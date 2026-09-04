"""Đánh giá engine liên kết trên fixture WildTrack (dữ liệu thật + ground truth).

    PYTHONPATH=src python -m eval.eval_wildtrack \\
        --fixture data/fixtures/wildtrack_3cam.jsonl --sweep

Khác `eval/sweep_synthetic.py` ở chỗ đây là **dữ liệu thật**: ảnh người thật ở 7 camera
nhìn chung một quảng trường, embedding trích bằng OSNet, `gt_global_id` do WildTrack gán
nhất quán xuyên camera. Đây là phép thử nghiêm túc đầu tiên của `src/mct/` trước khi có
dataset tự thu (M6).

**Đặc thù WildTrack phải nhớ khi đọc số:** mọi camera đều chồng lấn nhau và cùng nhìn một
lúc, nên bài toán ở đây là "liên kết đồng thời", KHÔNG có pha chuyển tiếp non-overlap mà
đề cương quan tâm nhất. Ràng buộc thời gian di chuyển vì thế gần như không lọc được gì
(mọi cặp `min_ms=0`), và toàn bộ gánh nặng dồn lên đặc trưng ngoại hình. Coi kết quả ở đây
là **cận dưới**: phần topology chưa được dùng tới.

Chỉ số tính theo cặp tracklet, giống `sweep_synthetic.py` để so sánh được giữa hai bộ dữ liệu.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.schema import read_jsonl
from mct.affinity import AffinityConfig
from mct.associator import run_offline
from mct.gallery import GalleryConfig
from mct.homography import HomographyMapper
from mct.topology import Topology
from mct.tracklet import TrackletConfig, build_tracklets

# Hai đầu của dải, vì hai chế độ cần hai vùng ngưỡng khác hẳn nhau:
#   - CHỈ ngoại hình: trên embedding thật (khác domain) cosine giữa hai người khác nhau đã
#     ~0.70, nên phải siết xuống 0.04-0.15 mới mong tách được;
#   - CÓ hình học (`geo=reject`): vị trí lo phần loại bỏ, ngưỡng ngoại hình siết chặt chỉ
#     tổ ném đi những cặp đúng có cosine thấp. Đo được 2026-09-04: 0.30 → F1 0.446,
#     0.40 → F1 0.712 trên cùng cấu hình.
MAX_COSTS = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60)


def overlapping_topology(cam_ids: list[str]) -> Topology:
    """Topology cho WildTrack: mọi camera chồng lấn nhau, không giới hạn thời gian.

    Dựng bằng code chứ không thêm vào `configs/cameras/topology.yaml` — file đó mô tả
    hệ thống camera THẬT của đồ án, không phải dataset mượn để thử thuật toán.
    """
    return Topology.from_mapping(
        {"cameras": {cam: {"overlaps_with": [c for c in cam_ids if c != cam]} for cam in cam_ids}}
    )


def load_gt(path: Path) -> dict[tuple[str, int], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        (t["cam_id"], int(t["local_track_id"])): int(t["gt_global_id"]) for t in data["tracklets"]
    }


def score(results, gt: dict[tuple[str, int], int]) -> dict[str, float]:
    """P/R/F1 theo cặp tracklet + số danh tính bị vỡ / Global ID bị gộp.

    Chỉ tính trên các cặp tracklet KHÁC CAMERA: cặp cùng camera bị ràng buộc loại trừ xử
    lý riêng và luôn đúng theo thiết kế, gộp vào sẽ thổi phồng điểm số.
    """
    gid_of = {r.tracklet.key: r.global_id for r in results if r.tracklet.key in gt}
    keys = sorted(gid_of)
    tp = fp = fn = 0
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if a[0] == b[0]:
                continue
            same_gt = gt[a] == gt[b]
            same_gid = gid_of[a] == gid_of[b]
            tp += same_gt and same_gid
            fp += same_gid and not same_gt
            fn += same_gt and not same_gid

    per_identity: dict[int, set[int]] = defaultdict(set)
    per_gid: dict[int, set[int]] = defaultdict(set)
    for key, gid in gid_of.items():
        per_identity[gt[key]].add(gid)
        per_gid[gid].add(gt[key])

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "n_tracklet": len(keys),
        "n_identity": len(per_identity),
        "n_gid": len(set(gid_of.values())),
        "broken": sum(1 for g in per_identity.values() if len(g) > 1),
        "merged": sum(1 for ids in per_gid.values() if len(ids) > 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def evaluate(
    tracklets,
    gt: dict[tuple[str, int], int],
    cam_ids: list[str],
    *,
    max_cost: float,
    mode: str,
    use_topology: bool,
    ground_mapper: HomographyMapper | None = None,
    homography_weight: float = 0.4,
    max_ground_dist_m: float = 3.0,
    ground_gap_policy: str = "allow",
) -> dict[str, float]:
    """Gán trên tracklet ĐÃ dựng sẵn — dựng lại cho mỗi tổ hợp tham số là phí thời gian.

    `ground_mapper` chỉ có tác dụng khi `use_topology=True`: affinity cần topology mới
    biết cặp camera nào chồng lấn, mà thành phần hình học chỉ áp cho cặp chồng lấn.
    """
    results, _ = run_offline(
        tracklets,
        topology=overlapping_topology(cam_ids) if use_topology else None,
        config=AffinityConfig(
            max_cost=max_cost,
            similarity_mode=mode,  # type: ignore[arg-type]
            homography_weight=homography_weight,
            max_ground_dist_m=max_ground_dist_m,
            ground_gap_policy=ground_gap_policy,
        ),
        gallery_config=GalleryConfig(similarity_mode=mode),  # type: ignore[arg-type]
        ground_mapper=ground_mapper,
    )
    return score(results, gt)


def diagnose(tracklets, gt: dict[tuple[str, int], int], topk: int = 8) -> None:
    """Phân bố cosine giữa các cặp tracklet KHÁC camera, tách theo cùng/khác danh tính.

    Đây là thứ quyết định `max_cost` chọn được hay không: nếu hai phân bố chồng lên nhau
    thì không ngưỡng nào tách được, và mọi công sức chỉnh tham số là vô ích.
    """
    items = [
        (t.cam_id, gt[t.key], q)
        for t in tracklets
        if t.key in gt and (q := t.query_embedding(topk)) is not None
    ]
    same: list[float] = []
    diff: list[float] = []
    for i, (cam_a, id_a, q_a) in enumerate(items):
        for cam_b, id_b, q_b in items[i + 1 :]:
            if cam_a == cam_b:
                continue
            (same if id_a == id_b else diff).append(float(q_a @ q_b))

    print(
        f"cosine cặp tracklet khác camera: {len(same)} cặp cùng người, {len(diff)} cặp khác người"
    )
    print(f"{'nhóm':12s} {'p05':>7s} {'p25':>7s} {'trung vị':>9s} {'p75':>7s} {'p95':>7s}")
    for name, values in (("cùng người", same), ("khác người", diff)):
        if not values:
            continue
        qs = np.percentile(np.array(values), [5, 25, 50, 75, 95])
        print(f"{name:12s} " + " ".join(f"{v:7.3f}" for v in qs))

    if not (same and diff):
        return
    # Trần lý thuyết nếu CHỈ dùng ngoại hình: quét mọi mốc cosine, lấy F1 tốt nhất.
    # Không thuật toán gán nào vượt được con số này trên chính bộ embedding đó.
    s_arr, d_arr = np.array(same), np.array(diff)
    best_f1, best_thr = 0.0, 0.0
    for thr in np.unique(np.round(np.concatenate([s_arr, d_arr]), 3)):
        tp = int((s_arr >= thr).sum())
        if not tp:
            continue
        fp = int((d_arr >= thr).sum())
        fn = int((s_arr < thr).sum())
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    print(
        f"trần lý thuyết chỉ với ngoại hình: F1={best_f1:.3f} tại cosine>={best_thr:.3f} "
        f"(tương ứng max_cost={1 - best_thr:.3f})"
    )


def diagnose_ground(messages, gt: dict[tuple[str, int], int], mapper: HomographyMapper) -> None:
    """Khoảng cách mặt đất giữa các cặp tracklet KHÁC camera, tách theo cùng/khác danh tính.

    Song song với `diagnose()` của phần ngoại hình, và trả lời đúng một câu hỏi: hình học
    có tách được người hay không, TRƯỚC khi bàn tới trọng số λ. Ở đây so tại **cùng mốc
    thời gian** (WildTrack đồng bộ khung giữa 7 camera) rồi lấy trung vị trên các mốc
    chung — tức là cận trên của những gì thành phần hình học có thể đem lại; affinity thật
    chỉ so một điểm đầu với một điểm cuối nên sẽ kém hơn con số này.
    """
    tracks: dict[tuple[str, int], dict[int, tuple[float, float]]] = defaultdict(dict)
    for msg in messages:
        for det in msg.detections:
            key = (msg.cam_id, det.local_track_id)
            if key in gt:
                point = mapper.project(msg.cam_id, det.ground_point)
                if point is not None:
                    tracks[key][msg.ts_ms] = point

    keys = sorted(tracks)
    same: list[float] = []
    diff: list[float] = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if a[0] == b[0]:
                continue
            common = tracks[a].keys() & tracks[b].keys()
            if not common:
                continue
            d = float(
                np.median([np.hypot(*(np.subtract(tracks[a][t], tracks[b][t]))) for t in common])
            )
            (same if gt[a] == gt[b] else diff).append(d)

    print(
        f"khoảng cách mặt đất cặp tracklet khác camera (trung vị trên mốc chung): "
        f"{len(same)} cặp cùng người, {len(diff)} cặp khác người"
    )
    print(f"{'nhóm':12s} {'p05':>7s} {'p25':>7s} {'trung vị':>9s} {'p75':>7s} {'p95':>7s}")
    for name, values in (("cùng người", same), ("khác người", diff)):
        if not values:
            continue
        qs = np.percentile(np.array(values), [5, 25, 50, 75, 95])
        print(f"{name:12s} " + " ".join(f"{v:7.3f}" for v in qs))

    if not (same and diff):
        return
    s_arr, d_arr = np.array(same), np.array(diff)
    best_f1, best_thr = 0.0, 0.0
    for thr in np.unique(np.round(np.concatenate([s_arr, d_arr]), 2)):
        tp = int((s_arr <= thr).sum())
        if not tp:
            continue
        fp = int((d_arr <= thr).sum())
        fn = int((s_arr > thr).sum())
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    print(f"trần lý thuyết chỉ với hình học: F1={best_f1:.3f} tại d_ground<={best_thr:.2f} m")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--gt", type=Path, default=None, help="mặc định: <fixture>.gt.json")
    p.add_argument("--max-cost", type=float, default=0.30)
    p.add_argument("--min-frames", type=int, default=3, help="WildTrack chú thích ~2 fps")
    p.add_argument(
        "--idle-timeout-ms",
        type=int,
        default=2000,
        help=(
            "khoảng lặng bao lâu thì CẮT tracklet. Phụ thuộc frame rate: ở 2 fps, 2000 ms "
            "chỉ là 4 khung, nên hụt vài detection là tracklet bị chẻ đôi. Đo 2026-09-04 "
            "trên fixture DeepStream: 2000 -> 30000 ms nâng trần recall của "
            "ground_gap_policy=reject từ 13.8% lên 34.5% (eval/diagnose_tracklets.py)"
        ),
    )
    p.add_argument("--mode", default="max", choices=("max", "centroid"))
    p.add_argument("--sweep", action="store_true", help="quét max_cost x similarity_mode")
    p.add_argument("--diagnose", action="store_true", help="in phân bố cosine cùng/khác danh tính")
    p.add_argument(
        "--homography-dir",
        type=Path,
        default=None,
        help="thư mục file hiệu chỉnh homography (mct.homography). Có thì sweep thêm cột `geo`",
    )
    p.add_argument("--lambda", dest="lam", type=float, default=0.4, help="trọng số d_ground")
    p.add_argument("--max-ground-dist", type=float, default=3.0, help="mét, vượt là loại thẳng")
    p.add_argument(
        "--ground-gap-policy",
        default="allow",
        choices=("allow", "reject", "both"),
        help="cặp camera chồng lấn không có mốc thời gian chung: thả qua / loại / đo cả hai",
    )
    args = p.parse_args(argv)

    gt_path = args.gt or args.fixture.with_suffix("").with_suffix(".gt.json")
    if not gt_path.is_file():
        gt_path = Path(str(args.fixture).replace(".jsonl", ".gt.json"))
    gt = load_gt(gt_path)

    messages = list(read_jsonl(args.fixture))
    cam_ids = sorted({m.cam_id for m in messages})
    print(
        f"{args.fixture.name}: {len(messages)} message, {len(cam_ids)} camera, "
        f"{len(gt)} tracklet ground-truth, {len(set(gt.values()))} danh tính"
    )

    tracklets = build_tracklets(
        messages,
        TrackletConfig(min_frames=args.min_frames, idle_timeout_ms=args.idle_timeout_ms),
    )
    print(f"gom được {len(tracklets)} tracklet (min_frames={args.min_frames})")

    mapper = None
    if args.homography_dir is not None:
        mapper = HomographyMapper.load(args.homography_dir)
        print(
            f"homography: {len(mapper.calibrated)} camera đã hiệu chỉnh "
            f"({', '.join(mapper.calibrated)})"
        )

    if args.diagnose:
        diagnose(tracklets, gt)
        if mapper is not None:
            diagnose_ground(messages, gt, mapper)

    header = (
        f"{'mode':9s} {'topo':5s} {'geo':7s} {'max_cost':>8s} {'#tracklet':>9s} {'#gid':>5s} "
        f"{'vỡ':>4s} {'gộp':>4s} {'P':>6s} {'R':>6s} {'F1':>6s}"
    )
    print(header)

    if mapper is None:
        geo_options: tuple[str | None, ...] = (None,)
    elif args.ground_gap_policy == "both":
        geo_options = (None, "allow", "reject")
    else:
        geo_options = (None, args.ground_gap_policy)
    combos = (
        [
            (m, t, g, c)
            for m in ("max", "centroid")
            for t in (True, False)
            for g in geo_options
            for c in MAX_COSTS
            if not (g and not t)  # hình học chỉ áp cho cặp overlap -> cần topology
        ]
        if args.sweep
        else [(args.mode, True, g, args.max_cost) for g in geo_options]
    )
    for mode, use_topology, use_geo, max_cost in combos:
        s = evaluate(
            tracklets,
            gt,
            cam_ids,
            max_cost=max_cost,
            mode=mode,
            use_topology=use_topology,
            ground_mapper=mapper if use_geo else None,
            ground_gap_policy=use_geo or "allow",
            homography_weight=args.lam,
            max_ground_dist_m=args.max_ground_dist,
        )
        geo_label = use_geo or "-"
        print(
            f"{mode:9s} {use_topology!s:5s} {geo_label:7s} {max_cost:8.2f} "
            f"{s['n_tracklet']:9.0f} {s['n_gid']:5.0f} {s['broken']:4.0f} {s['merged']:4.0f} "
            f"{s['precision']:6.3f} {s['recall']:6.3f} {s['f1']:6.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
