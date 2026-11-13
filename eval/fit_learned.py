"""Fit the engine's learned geometry models (`mct.learned`) from an EXPLICIT list of lab sessions.

    PYTHONPATH=src:. python -m eval.fit_learned \\
        --fixtures-dir data/lab/gpu_s42/data/fixtures --gt-dir data/fixtures \\
        --topology configs/lab/topology.yaml --homography-dir configs/lab/homography \\
        --sessions s2_1 s2_1b s2_1c s2_2 s2_2b s2_2c s2_3 s2_3b --calib calib calib_2 \\
        --out-dir configs/lab/learned

Four files, one per `association.*` key (`mct.learned`). Each records `fit_sessions` and
`fit_date`; the engine only reads them.

**What is learned, and from what.** Only geometry: box sizes, foot positions, where and which
way people walk. Never appearance (the same ~7 people are in every lab session, so anything
learned from embeddings would leak identity across sessions).

- `foot_height`    pipeline boxes only, no labels at all;
- `ground_sigma`   pipeline boxes of ONE person seen by two overlapping cameras at the same
                   instant (the ground truth only says "same person", never who);
- `exit_direction` ground-truth walks between non-overlapping cameras (training + calibration);
- `entry_zones`    where pipeline tracks start, labelled arrival vs fragment by the ground truth.

**Leakage guard.** `fit_all` takes the session list explicitly and calls
`eval.lab_geometry.check_sessions`: a held-out session (or the test session of a leave-one-out
fold, passed as `exclude`) raises before any file is read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from eval import roi_filter_fixture
from eval.lab_geometry import CALIB_SESSIONS, Boxes, LabData, check_sessions
from mct.homography import HomographyMapper, estimate_homography
from mct.learned import FootModel, exit_velocity, head_to_image_matrix, pair_key
from tools.estimate_transit import Appearance, handovers, load_overlaps, pair_kind

BOTTOM_MARGIN_PX = 30.0
"""A box whose bottom is within this many pixels of the frame bottom counts as cut. Detector
boxes of cut people cluster 10-20 px above the border, not on it (measured, session 45)."""

FILES = {
    "foot": "foot_height.yaml",
    "sigma": "ground_sigma.yaml",
    "exit": "exit_direction.yaml",
    "entry": "entry_zones.yaml",
}
CONFIG_KEYS = {
    "foot": "association.foot_from_height",
    "sigma": "association.ground_sigma_model",
    "exit": "association.exit_direction_gate",
    "entry": "association.entry_zone_prior",
}
SIGMA_MODES = ("sigma", "bias", "bias_sigma")


def _round(values: Any, digits: int = 6) -> Any:
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


# --------------------------------------------------------------------------- foot


def fit_foot(
    runs: Sequence[Boxes], mapper: HomographyMapper, *, method: str = "head_plane"
) -> dict[str, Any]:
    """Per camera: the height line and the head-plane mapping, from boxes that are not cut."""
    cameras: dict[str, Any] = {}
    for cam_id in sorted({str(c) for r in runs for c in np.unique(r.cam)}):
        boxes = np.concatenate([r.bbox[r.of_cam(cam_id)] for r in runs])
        height = float(next(r.frame_size[1] for r in runs))
        x, y, w, h = boxes.T
        bottom = y + h
        # Whole people only: not cut at the bottom or top, and not a half-body box (wide aspect).
        whole = (bottom < height - BOTTOM_MARGIN_PX) & (y > 6.0) & (w / np.maximum(h, 1.0) < 0.55)
        centres, medians, picked = [], [], []
        rng = np.random.default_rng(0)
        for lo in range(0, int(height), 50):
            rows = np.flatnonzero(whole & (bottom >= lo) & (bottom < lo + 50))
            if len(rows) < 15:
                continue
            centres.append(float(np.median(bottom[rows])))
            medians.append(float(np.median(h[rows])))
            if lo >= height / 2:  # the head plane is only used near the camera: fit it there
                picked.extend(rng.choice(rows, min(len(rows), 300), replace=False).tolist())
        entry: dict[str, Any] = {"n": int(whole.sum())}
        if len(centres) >= 3:
            slope, intercept = np.polyfit(centres, medians, 1)
            entry |= {"slope": round(float(slope), 6), "intercept": round(float(intercept), 3)}
        camera = mapper.cameras.get(cam_id)
        if camera is not None and len(picked) >= 50:
            rows = np.array(picked)
            mid = x[rows] + w[rows] / 2.0
            world = mapper.project_many(cam_id, np.column_stack([mid, bottom[rows]]))
            fit = estimate_homography(np.column_stack([mid, y[rows]]), world)
            entry["head_to_image"] = _round(head_to_image_matrix(fit.matrix, camera.matrix), 9)
        cameras[cam_id] = entry
    return {"method": method, "bottom_margin_px": BOTTOM_MARGIN_PX, "cameras": cameras}


def _world(
    boxes: np.ndarray, cam_id: str, mapper: HomographyMapper, foot: FootModel | None, height: float
) -> np.ndarray:
    """Ground position of each box, through the SAME foot rule the engine will use."""
    if foot is None:
        points = np.column_stack([boxes[:, 0] + boxes[:, 2] / 2.0, boxes[:, 1] + boxes[:, 3]])
    else:
        points = np.array([foot.foot(cam_id, tuple(b), height) for b in boxes]).reshape(-1, 2)
    return mapper.project_many(cam_id, points)


# --------------------------------------------------------------------------- ground error


def same_person_pairs(
    run: Boxes,
    cam_a: str,
    cam_b: str,
    mapper: HomographyMapper,
    foot: FootModel | None,
    *,
    tol_ms: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """World points of one person in two cameras at the same instant: two (n, 2) arrays."""
    height = float(run.frame_size[1])
    out_a, out_b = [np.empty((0, 2))], [np.empty((0, 2))]
    in_a = run.of_cam(cam_a) & (run.person >= 0)
    in_b = run.of_cam(cam_b) & (run.person >= 0)
    for person in np.unique(run.person[in_a]):
        a = np.flatnonzero(in_a & (run.person == person))
        b = np.flatnonzero(in_b & (run.person == person))
        if not len(a) or not len(b):
            continue
        b = b[np.argsort(run.ts_ms[b], kind="stable")]
        nearest = np.clip(np.searchsorted(run.ts_ms[b], run.ts_ms[a]), 0, len(b) - 1)
        hit = np.abs(run.ts_ms[b][nearest] - run.ts_ms[a]) <= tol_ms
        if hit.any():
            out_a.append(_world(run.bbox[a[hit]], cam_a, mapper, foot, height))
            out_b.append(_world(run.bbox[b[nearest[hit]]], cam_b, mapper, foot, height))
    return np.concatenate(out_a), np.concatenate(out_b)


def fit_ground_error(
    runs: Sequence[Boxes],
    mapper: HomographyMapper,
    overlaps: dict[str, set[str]],
    foot: FootModel | None,
    *,
    mode: str = "bias",
    cell_m: float = 1.0,
    min_n: int = 50,
) -> dict[str, Any]:
    """Offset and spread of same-person foot points per floor cell, for every overlapping pair."""
    if mode not in SIGMA_MODES:
        raise ValueError(f"mode phải thuộc {SIGMA_MODES}, nhận {mode!r}")
    pairs: dict[str, Any] = {}
    for cam_a, cam_b in sorted({tuple(sorted((a, b))) for a, bs in overlaps.items() for b in bs}):
        parts = [same_person_pairs(r, cam_a, cam_b, mapper, foot) for r in runs]
        first = np.concatenate([p[0] for p in parts])
        second = np.concatenate([p[1] for p in parts])
        ok = np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
        first, second = first[ok], second[ok]
        if len(first) < min_n:
            continue
        delta = second - first
        keys = np.floor((first + second) / 2.0 / cell_m).astype(np.int64)
        by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, key in enumerate(map(tuple, keys.tolist())):
            by_cell[key].append(i)
        dense = {k: np.array(v) for k, v in by_cell.items() if len(v) >= min_n}

        use_bias, use_sigma = mode != "sigma", mode != "bias"
        default_mu = np.median(delta, axis=0) if use_bias else np.zeros(2)
        mu = np.tile(default_mu, (len(delta), 1))
        if use_bias:
            for rows in dense.values():
                mu[rows] = np.median(delta[rows], axis=0)
        residual = np.hypot(*(delta - mu).T)
        sigma_ref = float(np.sqrt(np.mean(residual**2)))
        cells = [
            {
                "ix": int(k[0]),
                "iy": int(k[1]),
                "mu": _round(mu[rows[0]], 4),
                "sigma": round(
                    float(np.sqrt(np.mean(residual[rows] ** 2))) if use_sigma else sigma_ref, 4
                ),
                "n": len(rows),
            }
            for k, rows in sorted(dense.items())
        ]
        pairs[pair_key(cam_a, cam_b)] = {
            "cell_m": cell_m,
            "sigma_ref": round(sigma_ref, 4),
            "default": {"mu": _round(default_mu, 4), "sigma": round(sigma_ref, 4)},
            "n": len(delta),
            "raw_median_m": round(float(np.median(np.hypot(*delta.T))), 4),
            "corrected_median_m": round(float(np.median(residual)), 4),
            "cells": cells,
        }
    return {"mode": mode, "pairs": pairs}


# --------------------------------------------------------------------------- exit direction


def _appearances(gt: Boxes, gap_ms: int = 2000) -> list[Appearance]:
    """Ground-truth boxes -> `tools.estimate_transit.Appearance` (same definition as the scorer)."""
    out: list[Appearance] = []
    labelled = gt.person >= 0
    for person in np.unique(gt.person[labelled]):
        for cam_id in np.unique(gt.cam[labelled & (gt.person == person)]):
            rows = np.flatnonzero(labelled & (gt.person == person) & gt.of_cam(str(cam_id)))
            rows = rows[np.argsort(gt.ts_ms[rows], kind="stable")]
            current = Appearance(int(person), str(cam_id))
            for i in rows:
                ts = int(gt.ts_ms[i])
                if current.boxes and ts - current.boxes[-1][1] > gap_ms:
                    out.append(current)
                    current = Appearance(int(person), str(cam_id))
                current.boxes.append((0, ts, tuple(gt.bbox[i])))  # type: ignore[arg-type]
            out.append(current)
    return out


def fit_exit_direction(
    gts: Sequence[Boxes],
    mapper: HomographyMapper,
    overlaps: dict[str, set[str]],
    foot: FootModel | None,
    *,
    window_ms: float = 1000.0,
    min_speed_m_s: float = 0.3,
    min_cos: float = 0.0,
    min_n: int = 3,
) -> dict[str, Any]:
    """Mean walking direction at the moment of leaving `src`, per non-overlapping `src > dst`."""
    units: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for gt in gts:
        height = float(gt.frame_size[1])
        for h in handovers(_appearances(gt)):
            if pair_kind(h.src.cam_id, h.dst.cam_id, overlaps) != "non_overlap":
                continue
            boxes = np.array([b for _, _, b in h.src.boxes], dtype=np.float64)
            world = _world(boxes, h.src.cam_id, mapper, foot, height)
            if world is None:
                continue
            path = np.column_stack([[ts for _, ts, _ in h.src.boxes], world])
            velocity = exit_velocity(path[np.isfinite(path).all(axis=1)], window_ms)
            if velocity is not None and np.hypot(*velocity) >= min_speed_m_s:
                units[h.pair].append(velocity / np.hypot(*velocity))
    pairs: dict[str, Any] = {}
    for (src, dst), vectors in sorted(units.items()):
        if len(vectors) < min_n:
            continue
        mean = np.mean(vectors, axis=0)
        direction = mean / np.hypot(*mean)
        cos = np.array(vectors) @ direction
        pairs[f"{src}>{dst}"] = {
            "direction": _round(direction, 4),
            "n": len(vectors),
            "cos_min": round(float(cos.min()), 3),
            "cos_median": round(float(np.median(cos)), 3),
        }
    return {
        "window_ms": window_ms,
        "min_speed_m_s": min_speed_m_s,
        "min_cos": min_cos,
        "pairs": pairs,
    }


# --------------------------------------------------------------------------- entry zones


def fit_entry_zones(
    runs: Sequence[Boxes],
    *,
    rows: int = 4,
    cols: int = 6,
    idle_ms: int = 2000,
    lookback_ms: int = 2000,
    min_p: float = 0.8,
    min_n: int = 10,
) -> dict[str, Any]:
    """Share of tracklet starts per image cell that are new arrivals rather than fragments.

    A start is an ARRIVAL when the same person had no box in that camera during the
    `lookback_ms` before it; otherwise the tracker merely restarted on someone already there.
    Tracklets are cut like the engine cuts them (silence longer than `idle_ms`).
    """
    from mct.learned import EntryZoneModel

    counts: dict[str, np.ndarray] = {}
    arrivals: dict[str, np.ndarray] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for run in runs:
        labelled = run.person >= 0
        for cam_id in np.unique(run.cam[labelled]):
            cam = str(cam_id)
            in_cam = labelled & run.of_cam(cam)
            sizes[cam] = run.frame_size
            n = counts.setdefault(cam, np.zeros((rows, cols), dtype=np.int64))
            arrived = arrivals.setdefault(cam, np.zeros((rows, cols), dtype=np.int64))
            for track in np.unique(run.track[in_cam]):
                idx = np.flatnonzero(in_cam & (run.track == track))
                idx = idx[np.argsort(run.ts_ms[idx], kind="stable")]
                ts = run.ts_ms[idx]
                starts = idx[np.r_[True, np.diff(ts) > idle_ms]]
                person_ts = run.ts_ms[in_cam & (run.person == run.person[idx[0]])]
                for i in starts:
                    t0 = run.ts_ms[i]
                    seen_before = bool(((person_ts >= t0 - lookback_ms) & (person_ts < t0)).any())
                    cell = EntryZoneModel.cell(tuple(run.bbox[i]), run.frame_size, (rows, cols))
                    n[cell] += 1
                    arrived[cell] += not seen_before
    cameras = {
        cam: {
            "image_size": list(sizes[cam]),
            "n": counts[cam].tolist(),
            "p_entry": _round(arrivals[cam] / np.maximum(counts[cam], 1), 3),
        }
        for cam in sorted(counts)
    }
    return {"min_p": min_p, "min_n": min_n, "rows": rows, "cols": cols, "cameras": cameras}


# --------------------------------------------------------------------------- orchestration


def write_model(path: Path, kind: str, sessions: Iterable[str], body: dict[str, Any]) -> Path:
    """One learned file, with the provenance the engine insists on (`mct.learned.load_learned`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "kind": kind,
        "fit_sessions": sorted(sessions),
        "fit_date": dt.date.today().isoformat(),
        "fit_tool": "eval.fit_learned",
        **body,
    }
    path.write_text(
        "# Fitted by eval.fit_learned. Do not edit by hand; refit instead.\n"
        + yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def fit_all(
    lab: LabData,
    sessions: Sequence[str],
    out_dir: Path,
    *,
    mapper: HomographyMapper,
    overlaps: dict[str, set[str]],
    calib: Sequence[str] = (),
    exclude: Iterable[str] = (),
    foot_method: str | None = "head_plane",
    sigma_mode: str = "bias",
    components: Iterable[str] = ("foot", "sigma", "exit", "entry"),
) -> dict[str, Path]:
    """Fit the requested components on `sessions` (+ `calib` for walking directions).

    `foot_method=None` fits the later components on plain bottom-of-box feet (the engine then
    must run without `foot_from_height` too). `exclude` is the test session of a leave-one-out
    fold: it, like any held-out session, makes this raise instead of fit.
    """
    sessions = check_sessions(sessions, exclude=exclude)
    calib = check_sessions(calib, exclude=exclude) if calib else []
    bad = sorted(set(calib) - set(CALIB_SESSIONS))
    if bad:
        raise ValueError(f"--calib chỉ nhận {CALIB_SESSIONS}, nhận thêm {bad}")
    components = list(components)
    runs = lab.pipeline(sessions)
    out: dict[str, Path] = {}
    foot: FootModel | None = None
    if foot_method is not None and ({"foot", "sigma", "exit"} & set(components)):
        path = write_model(
            out_dir / FILES["foot"],
            "foot_height",
            sessions,
            fit_foot(runs, mapper, method=foot_method),
        )
        foot = FootModel.load(path)
        if "foot" in components:
            out["foot"] = path
    if "sigma" in components:
        body = fit_ground_error(runs, mapper, overlaps, foot, mode=sigma_mode)
        out["sigma"] = write_model(out_dir / FILES["sigma"], "ground_sigma", sessions, body)
    if "exit" in components:
        walks = lab.ground_truth([*sessions, *calib])
        body = fit_exit_direction(walks, mapper, overlaps, foot)
        out["exit"] = write_model(
            out_dir / FILES["exit"], "exit_direction", [*sessions, *calib], body
        )
    if "entry" in components:
        out["entry"] = write_model(
            out_dir / FILES["entry"], "entry_zones", sessions, fit_entry_zones(runs)
        )
    return out


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fixtures-dir", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--topology", type=Path, required=True)
    p.add_argument("--homography-dir", type=Path, required=True)
    p.add_argument("--sessions", nargs="+", required=True, help="buổi dùng để fit (tường minh)")
    p.add_argument("--calib", nargs="*", default=[], help="buổi hiệu chỉnh, chỉ cho hướng đi")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--foot-method", default="head_plane", choices=["height_line", "head_plane"])
    p.add_argument("--sigma-mode", default="bias", choices=SIGMA_MODES)
    args = p.parse_args(argv)
    lab = LabData(
        args.fixtures_dir,
        args.gt_dir,
        roi_filter_fixture.topology_filter(args.topology, args.homography_dir),
        args.cache_dir,
    )
    written = fit_all(
        lab,
        args.sessions,
        args.out_dir,
        mapper=HomographyMapper.load(args.homography_dir),
        overlaps=load_overlaps(args.topology),
        calib=args.calib,
        foot_method=args.foot_method,
        sigma_mode=args.sigma_mode,
    )
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
