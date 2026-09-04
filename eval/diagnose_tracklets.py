"""Định lượng độ vỡ tracklet và trần recall của ràng buộc hình học.

    PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m eval.diagnose_tracklets \\
        --fixture data/fixtures/ds_wildtrack_7cam.jsonl \\
        --homography-dir configs/cameras/homography/wildtrack

**Vì sao có file này.** Phiên 11 (2026-09-04) đo được F1 tụt từ 0.752 trên fixture SCT lý
tưởng xuống 0.170 trên fixture pipeline thật, và truy ra nguyên nhân là tracklet vỡ vụn chứ
không phải embedding. Nhưng "vỡ vụn" mới là chẩn đoán định tính. Trước khi động vào bất kỳ
tham số nào cần biết **vỡ đến mức nào, và ràng buộc hình học còn cho phép ghép bao nhiêu
phần trăm số cặp đúng** — nếu trần đó đã thấp thì chỉnh ngưỡng ngoại hình là vô ích, phải
sửa ở tầng tracker.

Ba phần báo cáo:

1. **Độ vỡ** — mỗi danh tính bị cắt thành mấy tracklet, tracklet dài bao nhiêu.
2. **Trùng thời gian** — bao nhiêu phần trăm cặp tracklet khác camera của CÙNG người có mốc
   thời gian chung. Đây là điều kiện sống còn của thành phần hình học: `_ground_term` lấy
   trung vị khoảng cách trên các mốc chung, không có mốc chung thì `ground_gap_policy`
   quyết định thả qua hay loại thẳng.
3. **Trần recall** dưới từng `ground_gap_policy` — tỉ lệ cặp đúng mà ràng buộc hình học
   CÒN CHO PHÉP ghép. Không thuật toán gán nào vượt được con số này.

Dùng thẳng `_world_path` và `_synchronized_distance` của `mct.affinity` chứ không viết lại:
một bản sao chép sẽ trôi khỏi bản gốc và cho ra con số không nói về hệ thống thật.

LƯU Ý KHI ĐỌC TRẦN RECALL: associator so tracklet với `GlobalTrack` đã gộp quỹ đạo của
nhiều camera, nên trên thực tế nó có nhiều cơ hội tìm mốc chung hơn phép so từng-cặp ở đây.
Con số này vì thế là **thước đo chẩn đoán**, không phải cận trên chặt. Nó trả lời đúng một
câu hỏi: ràng buộc hình học có phải là thứ đang chặn recall hay không.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.schema import read_jsonl
from mct.affinity import AffinityConfig, _synchronized_distance, _world_path
from mct.homography import HomographyMapper
from mct.tracklet import Tracklet, TrackletConfig, build_tracklets

PERCENTILES = (5, 25, 50, 75, 95)


def load_gt(path: Path) -> dict[tuple[str, int], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        (t["cam_id"], int(t["local_track_id"])): int(t["gt_global_id"]) for t in data["tracklets"]
    }


def _quantiles(values: list[float]) -> list[float]:
    if not values:
        return [0.0] * len(PERCENTILES)
    return [float(v) for v in np.percentile(np.array(values, dtype=np.float64), PERCENTILES)]


def _print_quantile_row(name: str, values: list[float], fmt: str = "{:8.1f}") -> None:
    print(f"{name:26s}" + "".join(fmt.format(v) for v in _quantiles(values)))


def fragmentation(tracklets: list[Tracklet], gt: dict[tuple[str, int], int]) -> dict:
    """Mỗi danh tính bị cắt thành mấy tracklet, và tracklet dài bao nhiêu."""
    scored = [t for t in tracklets if t.key in gt]

    # (danh tính, camera) -> số tracklet. Vỡ TRONG một camera mới là thứ tracker gây ra;
    # một người xuất hiện ở 5 camera thì đương nhiên có >= 5 tracklet, đó không phải vỡ.
    per_id_cam: dict[tuple[int, str], int] = defaultdict(int)
    per_id_cams: dict[int, set[str]] = defaultdict(set)
    for t in scored:
        per_id_cam[(gt[t.key], t.cam_id)] += 1
        per_id_cams[gt[t.key]].add(t.cam_id)

    splits = [float(v) for v in per_id_cam.values()]
    n_frames = [float(t.n_frames) for t in scored]
    durations = [float(t.duration_ms) for t in scored]
    multi_cam = [i for i, cams in per_id_cams.items() if len(cams) >= 2]

    print(f"\n{'=' * 78}\n1. ĐỘ VỠ TRACKLET\n{'=' * 78}")
    print(f"tracklet dựng được          {len(tracklets)}")
    print(f"tracklet có trong bảng GT   {len(scored)}  (phần còn lại không được chấm)")
    print(f"danh tính                   {len(per_id_cams)}")
    print(f"danh tính thấy ở >=2 camera {len(multi_cam)}  <- chỉ nhóm này mới có việc để liên kết")
    print(f"trung bình tracklet/(danh tính, camera)  {np.mean(splits) if splits else 0:.2f}")
    print(f"\n{'':26s}" + "".join(f"{'p' + str(p):>8s}" for p in PERCENTILES))
    _print_quantile_row("tracklet/(danh tính,cam)", splits, "{:8.1f}")
    _print_quantile_row("độ dài tracklet (khung)", n_frames, "{:8.0f}")
    _print_quantile_row("thời lượng tracklet (s)", [d / 1000.0 for d in durations], "{:8.1f}")

    return {
        "n_tracklets_built": len(tracklets),
        "n_tracklets_scored": len(scored),
        "n_identities": len(per_id_cams),
        "n_identities_multi_cam": len(multi_cam),
        "mean_tracklets_per_id_cam": float(np.mean(splits)) if splits else 0.0,
        "tracklets_per_id_cam_quantiles": _quantiles(splits),
        "frames_quantiles": _quantiles(n_frames),
        "duration_s_quantiles": [v / 1000.0 for v in _quantiles(durations)],
    }


def _pair_geometry(
    a: Tracklet,
    b: Tracklet,
    mapper: HomographyMapper,
    config: AffinityConfig,
    cache: dict,
) -> tuple[float | None, float | None, int]:
    """(khoảng cách đồng bộ, khoảng cách đầu-cuối, Δt ms) cho một cặp tracklet.

    Khoảng cách đồng bộ là `None` khi hai quỹ đạo không có mốc thời gian chung — đúng điều
    kiện mà `ground_gap_policy` xử lý.
    """
    path_a = _world_path(a.cam_id, a.ground_path, mapper, cache)
    path_b = _world_path(b.cam_id, b.ground_path, mapper, cache)
    synced = _synchronized_distance(path_a, path_b, config.ground_time_tol_ms)

    # Đường dự phòng của `_ground_term`: điểm cuối của cái sớm hơn với điểm đầu của cái muộn hơn.
    first, second = (a, b) if a.end_ms <= b.end_ms else (b, a)
    endpoint = mapper.distance_m(
        first.cam_id, first.last_ground_point, second.cam_id, second.first_ground_point
    )
    gap_ms = max(0, second.start_ms - first.end_ms)
    return synced, endpoint, gap_ms


def temporal_overlap_and_ceiling(
    tracklets: list[Tracklet],
    gt: dict[tuple[str, int], int],
    mapper: HomographyMapper,
    config: AffinityConfig,
    *,
    max_negative_pairs: int,
    seed: int,
) -> dict:
    """Phần 2 + 3: tỉ lệ cặp có mốc chung, và trần recall theo từng chính sách."""
    items = [t for t in tracklets if t.key in gt]
    cache: dict = {}

    positives: list[tuple[Tracklet, Tracklet]] = []
    negatives: list[tuple[Tracklet, Tracklet]] = []
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            if a.cam_id == b.cam_id:
                continue  # cặp cùng camera do ràng buộc loại trừ lo, không tính ở đây
            (positives if gt[a.key] == gt[b.key] else negatives).append((a, b))

    rng = random.Random(seed)
    if len(negatives) > max_negative_pairs:
        negatives = rng.sample(negatives, max_negative_pairs)

    def measure(pairs: list[tuple[Tracklet, Tracklet]]) -> dict:
        synced_ok = 0
        synced_dists: list[float] = []
        feasible_reject = 0
        feasible_allow = 0
        gaps: list[float] = []
        for a, b in pairs:
            synced, endpoint, gap_ms = _pair_geometry(a, b, mapper, config, cache)
            gaps.append(gap_ms / 1000.0)
            if synced is not None:
                synced_ok += 1
                synced_dists.append(synced)
                ok = synced <= config.max_ground_dist_m
                feasible_reject += ok
                feasible_allow += ok
                continue
            # Không có mốc chung: `reject` loại thẳng; `allow` rơi xuống so đầu-cuối
            # với ngân sách nới theo tốc độ đi bộ.
            if endpoint is None:
                feasible_allow += 1  # chưa hiệu chỉnh homography -> hình học không chặn
                continue
            budget = config.max_ground_dist_m + config.max_speed_m_s * (gap_ms / 1000.0)
            feasible_allow += endpoint <= budget
        n = max(1, len(pairs))
        return {
            "n_pairs": len(pairs),
            "share_with_common_timestamp": synced_ok / n,
            "synced_distance_quantiles": _quantiles(synced_dists),
            "gap_s_quantiles": _quantiles(gaps),
            "feasible_reject": feasible_reject / n,
            "feasible_allow": feasible_allow / n,
        }

    pos = measure(positives)
    neg = measure(negatives)

    print(f"\n{'=' * 78}\n2. TRÙNG THỜI GIAN GIỮA CẶP TRACKLET KHÁC CAMERA\n{'=' * 78}")
    print(f"(dung sai mốc chung: ground_time_tol_ms = {config.ground_time_tol_ms} ms)")
    print(f"{'':26s}{'#cặp':>10s}{'có mốc chung':>15s}")
    print(f"{'cùng người':26s}{pos['n_pairs']:10d}{pos['share_with_common_timestamp']:14.1%}")
    print(f"{'khác người (mẫu)':26s}{neg['n_pairs']:10d}{neg['share_with_common_timestamp']:14.1%}")
    print(f"\n{'':26s}" + "".join(f"{'p' + str(p):>8s}" for p in PERCENTILES))
    _print_quantile_row("d_ground cùng người (m)", pos["synced_distance_quantiles"], "{:8.2f}")
    _print_quantile_row("d_ground khác người (m)", neg["synced_distance_quantiles"], "{:8.2f}")
    _print_quantile_row("khoảng trống thời gian (s)", pos["gap_s_quantiles"], "{:8.1f}")

    print(f"\n{'=' * 78}\n3. TRẦN RECALL CỦA RÀNG BUỘC HÌNH HỌC\n{'=' * 78}")
    print("Tỉ lệ cặp tracklet ĐÚNG mà ràng buộc còn cho phép ghép. Không thuật toán gán")
    print("nào vượt được con số này — nếu nó thấp, chỉnh ngưỡng ngoại hình là vô ích.")
    print(f"\n{'chính sách':26s}{'trần recall':>14s}{'chặn oan':>12s}")
    for policy in ("reject", "allow"):
        share = pos[f"feasible_{policy}"]
        print(f"{'ground_gap_policy=' + policy:26s}{share:13.1%}{1 - share:11.1%}")
    print(f"\n{'lọt lưới (cặp khác người)':26s}{'reject':>14s}{'allow':>12s}")
    print(f"{'':26s}{neg['feasible_reject']:13.1%}{neg['feasible_allow']:11.1%}")

    return {"positive": pos, "negative": neg}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--gt", type=Path, default=None, help="mặc định: <fixture>.gt.json")
    p.add_argument("--homography-dir", type=Path, required=True)
    p.add_argument("--min-frames", type=int, default=3)
    p.add_argument("--idle-timeout-ms", type=int, default=2000)
    p.add_argument("--ground-time-tol-ms", type=int, default=400)
    p.add_argument("--max-ground-dist", type=float, default=1.0)
    p.add_argument("--max-speed", type=float, default=2.5)
    p.add_argument("--max-negative-pairs", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", type=Path, default=None, help="ghi toàn bộ số liệu ra file JSON")
    args = p.parse_args(argv)

    gt_path = args.gt or Path(str(args.fixture).replace(".jsonl", ".gt.json"))
    gt = load_gt(gt_path)
    messages = list(read_jsonl(args.fixture))
    cam_ids = sorted({m.cam_id for m in messages})

    print(f"{args.fixture.name}: {len(messages)} msg, {len(cam_ids)} cam, {len(gt)} tracklet GT")

    tracklets = build_tracklets(
        messages,
        TrackletConfig(min_frames=args.min_frames, idle_timeout_ms=args.idle_timeout_ms),
    )
    mapper = HomographyMapper.load(args.homography_dir)
    print(f"homography: {len(mapper.calibrated)} camera đã hiệu chỉnh")

    config = AffinityConfig(
        ground_time_tol_ms=args.ground_time_tol_ms,
        max_ground_dist_m=args.max_ground_dist,
        max_speed_m_s=args.max_speed,
    )

    report = {
        "fixture": args.fixture.name,
        "n_messages": len(messages),
        "cam_ids": cam_ids,
        "config": {
            "min_frames": args.min_frames,
            "idle_timeout_ms": args.idle_timeout_ms,
            "ground_time_tol_ms": args.ground_time_tol_ms,
            "max_ground_dist_m": args.max_ground_dist,
            "max_speed_m_s": args.max_speed,
        },
        "fragmentation": fragmentation(tracklets, gt),
        "pairs": temporal_overlap_and_ceiling(
            tracklets,
            gt,
            mapper,
            config,
            max_negative_pairs=args.max_negative_pairs,
            seed=args.seed,
        ),
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nsố liệu đầy đủ: {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
