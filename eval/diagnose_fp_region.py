"""Are the detector's "false" boxes inside the area WildTrack actually annotates?

    PYTHONPATH=src python -m eval.diagnose_fp_region \\
        --gt-fixture data/fixtures/wildtrack_7cam.jsonl \\
        --homography-dir configs/cameras/homography/wildtrack \\
        --fixture r640 data/fixtures/ds_wildtrack_7cam_r640n_r1.jsonl \\
        --fixture r1280 data/fixtures/ds_wildtrack_7cam_r1280_r1.jsonl

Why this exists (docs/worklog/2026-09-20-25-*): raising the detector's input size from 640 to
1280 adds ~24 000 boxes that TrackEval counts as false positives, and HOTA drops. But WildTrack
only annotates people standing on a 12 m x 36 m ground grid; a person outside it is a correct
detection that the ground truth simply does not contain. `diagnose_junk_ids.py` cannot tell
those apart because it only asks "does an annotated person overlap this box".

Every detection that is not matched to a ground-truth box (one-to-one, IoU >= `--min-iou`) is
put in exactly one bucket, using the foot point (bottom-centre of the box) projected through the
camera's ground-plane homography:

- ``outside_area``    foot is off the annotated grid -> an evaluation artifact, not a detector
                      error.
- ``inside_near_gt``  foot is on the grid and the box overlaps a person at IoU >= 0.3 (poor
                      localisation, or a duplicate box on an already matched person).
- ``inside_no_gt``    foot is on the grid and no person overlaps -> a ghost box, or a person the
                      annotators missed.

Only numpy + scipy. No GPU, no Redis.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.schema import read_jsonl
from mct.homography import CameraHomography

# WildTrack ground grid: 480 x 1440 cells of 2.5 cm, origin (-3.0, -9.0) m
# (see tools/wildtrack_to_fixture.py, position_id_to_world_m).
AREA_X_M = (-3.0, 9.0)
AREA_Y_M = (-9.0, 27.0)

NEAR_GT_IOU = 0.3
OUTSIDE = "outside_area"
NEAR_GT = "inside_near_gt"
NO_GT = "inside_no_gt"
BUCKETS = (OUTSIDE, NEAR_GT, NO_GT)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of two ``[x, y, w, h]`` box arrays, shape (len(a), len(b))."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(
        np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None
    )
    ih = np.clip(
        np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None
    )
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def in_annotated_area(point: tuple[float, float]) -> bool:
    x, y = point
    return AREA_X_M[0] <= x < AREA_X_M[1] and AREA_Y_M[0] <= y < AREA_Y_M[1]


def classify_frame(
    det_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    homography: CameraHomography,
    min_iou: float,
) -> tuple[int, list[str]]:
    """Return ``(matched count, bucket of every unmatched detection)`` for one (camera, frame)."""
    det_boxes = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)
    n = len(det_boxes)
    if n == 0:
        return 0, []
    iou = iou_matrix(det_boxes, gt_boxes)
    matched = np.zeros(n, dtype=bool)
    if iou.shape[1]:
        # Maximise the NUMBER of pairs above the threshold, IoU only breaks ties.
        score = np.where(iou >= min_iou, 1.0 + iou, 0.0)
        rows, cols = linear_sum_assignment(-score)
        matched[[r for r, c in zip(rows, cols, strict=True) if score[r, c] > 0]] = True
    best = iou.max(axis=1) if iou.shape[1] else np.zeros(n)
    buckets: list[str] = []
    for i in np.flatnonzero(~matched):
        x, y, w, h = det_boxes[i]
        ground = homography.project((float(x + w / 2), float(y + h)))
        if ground is None or not in_annotated_area(ground):
            buckets.append(OUTSIDE)
        elif best[i] >= NEAR_GT_IOU:
            buckets.append(NEAR_GT)
        else:
            buckets.append(NO_GT)
    return int(matched.sum()), buckets


def load_boxes(path: Path) -> dict[tuple[str, int], np.ndarray]:
    boxes: dict[tuple[str, int], np.ndarray] = {}
    for msg in read_jsonl(path):
        boxes[(msg.cam_id, msg.frame_id)] = np.array(
            [d.bbox for d in msg.detections], dtype=np.float64
        ).reshape(-1, 4)
    return boxes


def scan(
    fixture: Path,
    gt: dict[tuple[str, int], np.ndarray],
    homography_dir: Path,
    min_iou: float,
) -> dict:
    homographies: dict[str, CameraHomography] = {}
    empty = np.zeros((0, 4))
    total, tp = 0, 0
    fp: Counter[str] = Counter()
    per_cam: dict[str, Counter[str]] = defaultdict(Counter)
    for msg in read_jsonl(fixture):
        if msg.cam_id not in homographies:
            homographies[msg.cam_id] = CameraHomography.load(homography_dir / f"{msg.cam_id}.yaml")
        det = np.array([d.bbox for d in msg.detections], dtype=np.float64).reshape(-1, 4)
        n_tp, buckets = classify_frame(
            det, gt.get((msg.cam_id, msg.frame_id), empty), homographies[msg.cam_id], min_iou
        )
        total += len(det)
        tp += n_tp
        fp.update(buckets)
        per_cam[msg.cam_id].update(buckets)
        per_cam[msg.cam_id]["tp"] += n_tp
    return {
        "detections": total,
        "tp": tp,
        "fp": {b: fp[b] for b in BUCKETS},
        "per_cam": {c: dict(v) for c, v in sorted(per_cam.items())},
    }


def format_report(results: Sequence[tuple[str, dict]]) -> str:
    lines = [
        f"{'':10} {'detections':>10} {'TP':>7} {'FP':>7} | "
        + " ".join(f"{b:>15}" for b in BUCKETS)
        + f" | {'outside %':>9}"
    ]
    for label, r in results:
        fp_total = sum(r["fp"].values())
        share = 100 * r["fp"][OUTSIDE] / fp_total if fp_total else 0.0
        lines.append(
            f"{label:10} {r['detections']:>10} {r['tp']:>7} {fp_total:>7} | "
            + " ".join(f"{r['fp'][b]:>15}" for b in BUCKETS)
            + f" | {share:>8.1f}%"
        )
    if len(results) > 1:
        base_label, base = results[0]
        lines.append("")
        lines.append(f"Extra false positives relative to {base_label}:")
        for label, r in results[1:]:
            extra = {b: r["fp"][b] - base["fp"][b] for b in BUCKETS}
            total = sum(extra.values())
            parts = ", ".join(
                f"{b} {v:+d} ({100 * v / total:.1f}%)" if total else f"{b} {v:+d}"
                for b, v in extra.items()
            )
            lines.append(f"  {label:8} {total:+d}:  {parts}   (extra TP {r['tp'] - base['tp']:+d})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--gt-fixture", type=Path, required=True, help="ground-truth fixture (wildtrack_7cam.jsonl)"
    )
    p.add_argument("--homography-dir", type=Path, required=True)
    p.add_argument(
        "--fixture",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        required=True,
        help="pipeline fixture to scan; the first one is the reference for the delta table",
    )
    p.add_argument("--min-iou", type=float, default=0.5)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    gt = load_boxes(args.gt_fixture)
    results = [
        (label, scan(Path(path), gt, args.homography_dir, args.min_iou))
        for label, path in args.fixture
    ]
    print(format_report(results))
    if args.json:
        args.json.write_text(
            json.dumps(dict(results), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
