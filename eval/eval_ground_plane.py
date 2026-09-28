"""Ground-plane tracking metrics on WildTrack, the way the multi-view literature reports them.

    PYTHONPATH=src ~/.venvs/mct-eval/Scripts/python.exe -m eval.eval_ground_plane \\
        --trackeval-path ~/TrackEval --wildtrack-dir data/wildtrack \\
        --homography-dir configs/cameras/homography/wildtrack \\
        --run R640:r1 data/fixtures/ds_wildtrack_7cam_r640n_r1.jsonl data/s25/R640_r1/mct.db

    ... --self-check          # wiring test with no fixture or DB (see below)

Why this exists (docs/worklog/2026-09-20-25-*, question "why do WildTrack SOTA papers report
90+ and we report HOTA 15"): MVDet / EarlyBird / TrackTacular / MVTrajecter / MCBLT score people
as POINTS on the ground plane, matched within a metric distance (1 m for tracking), on the 12 m
x 36 m annotated grid only. Our other evaluations (`compare_oracle_tracker`) score image boxes
at IoU 0.5 on seven concatenated per-camera sequences. This tool re-scores the SAME engine runs
with the ground-plane protocol so the metric side is at least comparable.

Protocol, and where it still differs from the papers:

- One point per (frame, Global ID): every camera's box foot (bottom-centre) is projected with its
  ground-plane homography and the coordinate-wise MEDIAN over the cameras that see that Global ID
  in that frame is taken. Only points inside the annotated grid are kept (``--area both`` also
  reports the unfiltered variant).
- Ground truth: one point per (frame, personID), from ``positionID``.
- Similarity ``max(0, 1 - d / (2 * T))``: alpha = 0.5 corresponds to a distance of ``T`` metres, so
  TrackEval's CLEAR/Identity (threshold 0.5) match within T metres. HOTA is reported both as the
  mean over alpha 0.05..0.95 (TrackEval's definition) and at alpha = 0.5 (= T metres).
- Frames: ``all`` = the 400 annotated frames, ``test`` = the last 40 (the split the papers test
  on). Unlike them, nothing here is trained on WildTrack.
- NOT matched to the papers: they fuse the 7 views inside one network (recall ~90%); this engine
  detects per camera at 2-D IoU and associates afterwards.

Metrics come from TrackEval's own `HOTA`, `CLEAR` and `Identity` classes (numpy 1.23.5 -> run it
with the `mct-eval` venv), fed a hand-built sequence, so no metric formula is re-implemented here.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

AREA_X_M = (-3.0, 9.0)
AREA_Y_M = (-9.0, 27.0)
N_TEST_FRAMES = 40
ALPHA_HALF = 9  # index of alpha = 0.5 in TrackEval's np.arange(0.05, 0.99, 0.05)

Point = tuple[float, float]


def in_annotated_area(point: Point) -> bool:
    x, y = point
    return AREA_X_M[0] <= x < AREA_X_M[1] and AREA_Y_M[0] <= y < AREA_Y_M[1]


def load_gt_points(ann_dir: Path, max_frames: int | None = None) -> list[dict[int, Point]]:
    """``annotations_positions/*.json`` -> per frame ``{personID: (x_m, y_m)}``.

    Frame ``i`` is the i-th annotation file in name order, which is exactly the frame id that
    ``tools/wildtrack_to_video.py`` and the pipeline fixtures use.
    """
    from tools.wildtrack_to_fixture import position_id_to_world_m

    frames: list[dict[int, Point]] = []
    for path in sorted(ann_dir.glob("*.json"))[:max_frames]:
        frame: dict[int, Point] = {}
        for person in json.loads(path.read_text(encoding="utf-8")):
            frame[int(person["personID"])] = position_id_to_world_m(int(person["positionID"]))
        frames.append(frame)
    return frames


def fuse_median(points: Sequence[Point]) -> Point:
    return (
        float(statistics.median(p[0] for p in points)),
        float(statistics.median(p[1] for p in points)),
    )


def ground_nms(
    points: dict[int, Point], support: dict[int, int], radius_m: float
) -> dict[int, Point]:
    """Greedy ground-plane NMS: keep the best-supported point, drop any within ``radius_m`` of it.

    "Best" = seen by the most cameras, ties broken by the lower Global ID so the survivor is the
    same one frame after frame. The multi-view papers apply the same NMS (0.5 m) to their
    detections, so a person never yields two points; without it a person that the engine gives
    two Global IDs at once counts one of them as a false positive.
    """
    kept: dict[int, Point] = {}
    for gid in sorted(points, key=lambda g: (-support[g], g)):
        x, y = points[gid]
        if all(math.hypot(x - qx, y - qy) > radius_m for qx, qy in kept.values()):
            kept[gid] = points[gid]
    return kept


def fuse_frames(
    raw: Sequence[dict[int, list[Point]]], *, area_filter: bool, nms_m: float = 0.0
) -> list[dict[int, Point]]:
    """Per frame ``{gid: [points from each camera]}`` -> ``{gid: fused point}``."""
    out: list[dict[int, Point]] = []
    for frame in raw:
        fused = {gid: fuse_median(pts) for gid, pts in frame.items()}
        if area_filter:
            fused = {gid: p for gid, p in fused.items() if in_annotated_area(p)}
        if nms_m > 0:
            fused = ground_nms(fused, {gid: len(frame[gid]) for gid in fused}, nms_m)
        out.append(fused)
    return out


def predicted_points(
    fixture: Path, db: Path | None, homography_dir: Path, n_frames: int
) -> list[dict[int, list[Point]]]:
    """Fixture boxes + the engine's Global IDs -> per frame ``{gid: [ground point per camera]}``.

    ``db=None`` is the perfect-linking oracle: the Global ID is the fixture's ``local_track_id``,
    which is the WildTrack personID in the ``*_oracle.jsonl`` fixtures. It exists to test the
    geometry (frame alignment, homography, fusion) without any linking error in the way.
    """
    from common.schema import read_jsonl
    from mct.homography import CameraHomography
    from tools.export_trackeval import load_global_ids

    global_ids = load_global_ids(db) if db is not None else None
    homographies: dict[str, CameraHomography] = {}
    frames: list[dict[int, list[Point]]] = [{} for _ in range(n_frames)]
    for msg in read_jsonl(fixture):
        if not 0 <= msg.frame_id < n_frames:
            continue
        if msg.cam_id not in homographies:
            homographies[msg.cam_id] = CameraHomography.load(homography_dir / f"{msg.cam_id}.yaml")
        for det in msg.detections:
            if global_ids is None:
                gid: int | None = int(det.local_track_id)
            else:
                gid = global_ids.get(msg.cam_id, int(det.local_track_id), int(msg.ts_ms))
            if gid is None:
                continue  # the engine never assigned a Global ID: counts as a miss, as elsewhere
            x, y, w, h = det.bbox
            ground = homographies[msg.cam_id].project((float(x + w / 2), float(y + h)))
            if ground is not None:
                frames[msg.frame_id].setdefault(gid, []).append(
                    (float(ground[0]), float(ground[1]))
                )
    return frames


def build_sequence(
    gt: Sequence[dict[int, Point]],
    pred: Sequence[dict[int, Point]],
    frames: Iterable[int],
    zero_distance: float,
) -> dict[str, Any]:
    """The `data` dict TrackEval's metrics consume, with a distance-based similarity."""
    gt_map: dict[int, int] = {}
    pred_map: dict[int, int] = {}
    gt_ids: list[np.ndarray] = []
    tr_ids: list[np.ndarray] = []
    sims: list[np.ndarray] = []
    n_gt = n_tr = 0
    n_steps = 0
    for t in frames:
        g = gt[t] if t < len(gt) else {}
        p = pred[t] if t < len(pred) else {}
        gt_ids.append(np.array([gt_map.setdefault(k, len(gt_map)) for k in g], dtype=int))
        tr_ids.append(np.array([pred_map.setdefault(k, len(pred_map)) for k in p], dtype=int))
        gp = np.array(list(g.values()), dtype=float).reshape(-1, 2)
        pp = np.array(list(p.values()), dtype=float).reshape(-1, 2)
        dist = np.linalg.norm(gp[:, None, :] - pp[None, :, :], axis=2)
        sims.append(np.maximum(0.0, 1.0 - dist / zero_distance))
        n_gt += len(g)
        n_tr += len(p)
        n_steps += 1
    return {
        "num_timesteps": n_steps,
        "num_gt_ids": len(gt_map),
        "num_tracker_ids": len(pred_map),
        "num_gt_dets": n_gt,
        "num_tracker_dets": n_tr,
        "gt_ids": gt_ids,
        "tracker_ids": tr_ids,
        "similarity_scores": sims,
    }


def score_sequence(data: dict[str, Any], trackeval_path: Path) -> dict[str, float]:
    """Run TrackEval's HOTA / CLEAR / Identity on one sequence. Values in percent."""
    sys.path.insert(0, str(trackeval_path))
    from trackeval.metrics import CLEAR, HOTA, Identity  # type: ignore[import-not-found]

    quiet = {"PRINT_CONFIG": False}
    hota = HOTA(quiet).eval_sequence(data)
    clear = CLEAR(quiet).eval_sequence(data)
    ident = Identity(quiet).eval_sequence(data)
    return {
        "HOTA": 100 * float(np.mean(hota["HOTA"])),
        "HOTA@0.5": 100 * float(hota["HOTA"][ALPHA_HALF]),
        "DetA": 100 * float(np.mean(hota["DetA"])),
        "DetA@0.5": 100 * float(hota["DetA"][ALPHA_HALF]),
        "AssA": 100 * float(np.mean(hota["AssA"])),
        "AssA@0.5": 100 * float(hota["AssA"][ALPHA_HALF]),
        "MOTA": 100 * float(clear["MOTA"]),
        "IDF1": 100 * float(ident["IDF1"]),
        "IDP": 100 * float(ident["IDP"]),
        "IDR": 100 * float(ident["IDR"]),
        "TP": float(clear["CLR_TP"]),
        "FP": float(clear["CLR_FP"]),
        "FN": float(clear["CLR_FN"]),
        "IDSW": float(clear["IDSW"]),
        "n_gt_ids": float(data["num_gt_ids"]),
        "n_pred_ids": float(data["num_tracker_ids"]),
        "n_pred_dets": float(data["num_tracker_dets"]),
    }


def frame_ranges(n_frames: int) -> dict[str, range]:
    return {"all": range(n_frames), "test": range(max(0, n_frames - N_TEST_FRAMES), n_frames)}


def self_check(gt: Sequence[dict[int, Point]], trackeval_path: Path) -> tuple[list[str], bool]:
    """Feed the ground truth back as the prediction: the metrics must say what they should.

    The shifts are chosen against WildTrack's crowd density: people stand closer than 1.5 m, so
    a 1.5 m shift still lands most points within 1 m of a NEIGHBOUR and "no match" would be a
    wrong expectation. 40 m is far outside the annotated grid, so nothing can match.
    """
    n = len(gt)
    shift = lambda dx: [{k: (x + dx, y) for k, (x, y) in f.items()} for f in gt]  # noqa: E731
    checks = (
        ("GT as prediction", gt, 1.0, lambda r: min(r["MOTA"], r["IDF1"], r["HOTA"]) > 99.9),
        (
            "GT shifted 40 m (nothing can match)",
            shift(40.0),
            1.0,
            lambda r: r["TP"] == 0 and r["IDF1"] == 0 and r["MOTA"] < -99.9,
        ),
        (
            "GT shifted 0.3 m, T = 1 m (only neighbour swaps may cost anything)",
            shift(0.3),
            1.0,
            lambda r: min(r["MOTA"], r["IDF1"], r["HOTA@0.5"]) > 99.0,
        ),
    )
    lines, ok = [], True
    for label, pred, threshold, expect in checks:
        data = build_sequence(gt, pred, range(n), zero_distance=2 * threshold)
        r = score_sequence(data, trackeval_path)
        passed = bool(expect(r))
        ok &= passed
        lines.append(
            f"[{'PASS' if passed else 'FAIL'}] {label}\n"
            f"    MOTA {r['MOTA']:.1f}  IDF1 {r['IDF1']:.1f}  "
            f"HOTA {r['HOTA']:.1f}  HOTA@0.5 {r['HOTA@0.5']:.1f}  "
            f"TP {r['TP']:.0f}  FP {r['FP']:.0f}  FN {r['FN']:.0f}"
        )
    return lines, ok


COLUMNS = ("HOTA", "HOTA@0.5", "DetA@0.5", "AssA@0.5", "MOTA", "IDF1", "IDP", "IDR")


def aggregate(results: dict[str, dict[str, float]], columns: Sequence[str]) -> dict[str, str]:
    """{'R640:r1': {...}, 'R640:r2': {...}} -> mean +- sample std per scenario prefix."""
    groups: dict[str, list[dict[str, float]]] = {}
    for label, res in results.items():
        groups.setdefault(label.split(":")[0], []).append(res)
    out = {}
    for name, rows in groups.items():
        cells = []
        for col in columns:
            vals = [r[col] for r in rows]
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{statistics.mean(vals):.1f} ± {sd:.1f}")
        out[name] = f"n={len(rows)} | " + " | ".join(cells)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--trackeval-path", type=Path, required=True)
    p.add_argument(
        "--wildtrack-dir", type=Path, required=True, help="contains annotations_positions/"
    )
    p.add_argument("--homography-dir", type=Path, default=None)
    p.add_argument(
        "--run",
        nargs=3,
        action="append",
        default=[],
        metavar=("LABEL:RUN", "FIXTURE", "DB"),
        help="pipeline fixture + the engine DB that labelled it (or 'oracle': Global ID = "
        "local_track_id, perfect linking); runs sharing a LABEL are averaged",
    )
    p.add_argument("--threshold-m", type=float, nargs="+", default=[1.0, 0.5])
    p.add_argument("--area", choices=("in", "all", "both"), default="both")
    p.add_argument(
        "--nms-m",
        type=float,
        nargs="+",
        default=[0.0],
        help="ground-plane NMS radius(es) in metres; 0 = off (the papers use 0.5)",
    )
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    gt = load_gt_points(args.wildtrack_dir / "annotations_positions")
    if args.self_check:
        lines, ok = self_check(gt, args.trackeval_path)
        print("\n".join(lines))
        return 0 if ok else 1
    if not args.run or args.homography_dir is None:
        p.error("--run and --homography-dir are required unless --self-check")

    n_frames = len(gt)
    area_modes = {"in": [True], "all": [False], "both": [True, False]}[args.area]
    results: dict[str, dict[str, float]] = {}
    for label, fixture, db in args.run:
        raw = predicted_points(
            Path(fixture), None if db == "oracle" else Path(db), args.homography_dir, n_frames
        )
        name, _, run = label.partition(":")
        for area_filter in area_modes:
            for nms_m in args.nms_m:
                pred = fuse_frames(raw, area_filter=area_filter, nms_m=nms_m)
                for frames_name, frames in frame_ranges(n_frames).items():
                    for threshold in args.threshold_m:
                        data = build_sequence(gt, pred, frames, zero_distance=2 * threshold)
                        key = (
                            f"{name}|{'area' if area_filter else 'noarea'}|nms{nms_m:g}"
                            f"|{frames_name}|{threshold:g}m:{run}"
                        )
                        results[key] = score_sequence(data, args.trackeval_path)
        print(f"{label}: done", file=sys.stderr)

    header = "scenario|area|nms|frames|T | " + " | ".join(COLUMNS)
    print(header)
    for name, row in aggregate(results, COLUMNS).items():
        print(f"{name} | {row}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
