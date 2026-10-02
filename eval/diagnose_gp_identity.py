"""Vì sao HOTA mặt đất thấp: phân loại từng điểm mặt đất theo danh tính (phiên 27, mục 4 và 6).

    F=data/fixtures
    PYTHONPATH=src python -m eval.diagnose_gp_identity --wildtrack-dir data/wildtrack \\
        --run R640:r1 $F/ds_wildtrack_7cam_r640n_r1.jsonl data/s25/R640_r1/mct.db \\
        --run R640:r2 $F/ds_wildtrack_7cam_r640n_r2.jsonl data/s25/R640_r2/mct.db \\
        --json data/s25/gp_fp_identity.json

Bản đưa vào repo của `data/s27/scripts/gp_fp_identity.py` (trước chỉ có trên máy dev, nên số
"38% điểm sai là một người bị tách qua camera, 15% gộp nhầm, 37% hộp không khớp ai" của phiên
27 không tái lập được từ repo). Cùng thuật toán, cùng kết quả từng số — đối chiếu ở phiên 33.

Mỗi detection nhận `personID` bằng Hungarian trên IoU ≥ 0.5 với hộp chú thích của chính
(camera, khung) đó — đúng quy tắc của `tools/ds_wildtrack_gt`. Rồi trong từng khung:

- **độ vỡ** (`frag`): với mỗi người được ≥ 2 camera thấy, người đó nằm trong mấy Global ID
  (1, 2, 3, ≥4). Chỉ 1 mới là "liên kết đa camera thành công" ở khung đó;
- mỗi điểm hợp nhất (Global ID, khung) — trung vị điểm chân qua các camera, chỉ trong lưới chú
  thích — được ghép TP/FP với điểm chú thích trong `--threshold-m` (Hungarian). Điểm được xếp
  loại theo các detection mà Global ID đó gom trong khung:

    junk     không detection nào khớp hộp chú thích (hộp báo nhầm của detector)
    split    một người, và người đó CŨNG nằm dưới Global ID khác trong khung này (tách qua camera)
    solo     một người, không bị tách → nếu là FP thì do định vị (> ngưỡng mét)
    mixed    detection của ≥ 2 người dưới một Global ID (gộp nhầm)
    partial  một phần là junk, phần còn lại là một người

Chỉ numpy + scipy + `common/` + `mct/` + `tools/`; không cần GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.schema import read_jsonl
from eval.eval_ground_plane import Point, fuse_median, in_annotated_area, load_gt_points
from mct.homography import CameraHomography
from tools.ds_wildtrack_gt import gt_index, match_frame, view_idx_for_cam
from tools.export_trackeval import load_global_ids
from tools.wildtrack_to_fixture import parse_raw_detections

KINDS = ("junk", "split", "solo", "mixed", "partial")

Obs = tuple[str, Point, int]
"""(cam_id, điểm mặt đất, personID hoặc -1 nếu detection không khớp hộp chú thích nào)."""


def classify_point(observations: list[Obs], person_gids: dict[int, set[int]]) -> str:
    """Loại của một điểm hợp nhất (một Global ID trong một khung). Xem docstring module."""
    persons = [o[2] for o in observations]
    real = {p for p in persons if p >= 0}
    if not real:
        return "junk"
    if len(real) >= 2:
        return "mixed"
    if -1 in persons:
        return "partial"
    if len(person_gids[next(iter(real))]) >= 2:
        return "split"
    return "solo"


def score_frame(
    frame: dict[int, list[Obs]],
    gt_points: dict[int, Point],
    *,
    threshold_m: float = 1.0,
    area: Callable[[Point], bool] = in_annotated_area,
) -> tuple[Counter[int], Counter[str], Counter[str]]:
    """Một khung → (độ vỡ, loại các điểm TP, loại các điểm FP)."""
    person_gids: dict[int, set[int]] = defaultdict(set)
    person_cams: dict[int, set[str]] = defaultdict(set)
    for gid, obs in frame.items():
        for cam, _, pid in obs:
            if pid >= 0:
                person_gids[pid].add(gid)
                person_cams[pid].add(cam)
    frag: Counter[int] = Counter()
    for pid, cams in person_cams.items():
        if len(cams) >= 2:
            frag[min(len(person_gids[pid]), 4)] += 1

    points = {}
    for gid, obs in frame.items():
        q = fuse_median([o[1] for o in obs])
        if area(q):
            points[gid] = q
    ids = list(points)
    gt = np.array(list(gt_points.values()), dtype=float).reshape(-1, 2)
    pred = np.array([points[g] for g in ids], dtype=float).reshape(-1, 2)
    matched: set[int] = set()
    if len(gt) and len(pred):
        dist = np.linalg.norm(gt[:, None] - pred[None], axis=2)
        rows, cols = linear_sum_assignment(np.where(dist <= threshold_m, dist, 1e6))
        matched = {ids[j] for i, j in zip(rows, cols, strict=True) if dist[i, j] <= threshold_m}

    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    for gid in ids:
        kind = classify_point(frame[gid], person_gids)
        (tp if gid in matched else fp)[kind] += 1
    return frag, tp, fp


def collect_observations(
    fixture: Path, db: Path, wildtrack_dir: Path, homography_dir: Path, n_frames: int
) -> list[dict[int, list[Obs]]]:
    """Fixture + SQLite → mỗi khung: {Global ID: [(camera, điểm mặt đất, personID)]}."""
    raw, _ = parse_raw_detections(
        wildtrack_dir / "annotations_positions",
        view_indices=list(range(7)),
        stride=1,
        max_frames=0,
        min_box_area=0.0,
    )
    gt_by = gt_index(raw)
    gids = load_global_ids(db)
    homs: dict[str, CameraHomography] = {}
    frames: list[dict[int, list[Obs]]] = [defaultdict(list) for _ in range(n_frames)]
    for msg in read_jsonl(fixture):
        t = int(msg.frame_id)
        if not 0 <= t < n_frames or not msg.detections:
            continue
        hom = homs.setdefault(
            msg.cam_id, CameraHomography.load(homography_dir / f"{msg.cam_id}.yaml")
        )
        g_dets = gt_by.get((view_idx_for_cam(msg.cam_id), t), [])
        pairs = match_frame(
            [tuple(float(v) for v in d.bbox) for d in msg.detections],  # type: ignore[misc]
            [g.bbox for g in g_dets],
            min_iou=0.5,
        )
        pid_of = {di: g_dets[gi].person_id for di, gi, _ in pairs}
        for di, det in enumerate(msg.detections):
            gid = gids.get(msg.cam_id, int(det.local_track_id), int(msg.ts_ms))
            if gid is None:
                continue
            x, y, w, h = det.bbox
            p = hom.project((float(x + w / 2), float(y + h)))
            if p is None:
                continue
            frames[t][gid].append((msg.cam_id, (float(p[0]), float(p[1])), pid_of.get(di, -1)))
    return frames


def run(
    fixture: Path,
    db: Path,
    *,
    wildtrack_dir: Path,
    homography_dir: Path,
    threshold_m: float = 1.0,
) -> dict:
    gt_pts = load_gt_points(wildtrack_dir / "annotations_positions")
    frames = collect_observations(fixture, db, wildtrack_dir, homography_dir, len(gt_pts))
    frag: Counter[int] = Counter()
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    for t, frame in enumerate(frames):
        f, a, b = score_frame(frame, gt_pts[t], threshold_m=threshold_m)
        frag += f
        tp += a
        fp += b
    return {"frag": dict(sorted(frag.items())), "TP": dict(tp), "FP": dict(fp)}


def shares(result: dict) -> dict[str, float]:
    """Tỉ lệ từng loại trong các điểm FP — dạng số mà phiên 27 báo cáo."""
    total = sum(result["FP"].values())
    return {k: result["FP"].get(k, 0) / total for k in KINDS} if total else {}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run", nargs=3, action="append", required=True, metavar=("NHÃN", "FIXTURE", "DB")
    )
    p.add_argument("--wildtrack-dir", type=Path, default=Path("data/wildtrack"))
    p.add_argument(
        "--homography-dir", type=Path, default=Path("configs/cameras/homography/wildtrack")
    )
    p.add_argument("--threshold-m", type=float, default=1.0)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out = {}
    for label, fixture, db in args.run:
        res = run(
            Path(fixture),
            Path(db),
            wildtrack_dir=args.wildtrack_dir,
            homography_dir=args.homography_dir,
            threshold_m=args.threshold_m,
        )
        out[label] = res
        share = ", ".join(f"{k} {100 * v:.0f}%" for k, v in shares(res).items())
        print(f"{label}: {json.dumps(res)}\n  FP: {share}", flush=True)
    if args.json is not None:
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
