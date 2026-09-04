"""Đo chênh lệch giữa chế độ `online` và `offline` trên cùng một fixture.

    PYTHONPATH=src python -m eval.compare_online_offline \\
        --fixture data/fixtures/wildtrack_7cam.jsonl \\
        --homography-dir configs/cameras/homography/wildtrack \\
        --max-cost 0.5 --ground-gap-policy reject

Đây là con số mà CLAUDE.md §6 gọi là **cái giá phải trả của ràng buộc thời gian thực**.
Hai chế độ dùng chung đúng một hàm `Associator.assign()`; khác nhau ở chỗ tracklet được
đưa vào lúc nào:

  - `offline` — tracklet đã đóng, mang theo TOÀN BỘ đặc trưng và quỹ đạo. Cận trên của
    độ chính xác: hệ thống không thể làm tốt hơn thế với cùng bộ tham số.
  - `online`  — tracklet vào vòng gán ngay khi vừa động, lúc mới có vài khung. Quyết định
    gán đưa ra sớm và **không rút lại được** — đó chính là chỗ mất mát.

Cùng một fixture, cùng tham số, cùng chỉ số (cặp tracklet khác camera, xem
`eval_wildtrack.score`). Chênh lệch giữa hai dòng là thứ đi vào báo cáo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common.schema import read_jsonl
from eval.eval_wildtrack import load_gt, overlapping_topology, score
from mct.__main__ import Engine
from mct.affinity import AffinityConfig
from mct.associator import Associator, run_offline
from mct.gallery import Gallery, GalleryConfig
from mct.homography import HomographyMapper
from mct.tracklet import TrackletConfig, build_tracklets


class _Result:
    """Vỏ tối thiểu để dùng lại `eval_wildtrack.score()` cho kết quả của chế độ online."""

    __slots__ = ("global_id", "tracklet")

    def __init__(self, tracklet, global_id: int) -> None:
        self.tracklet = tracklet
        self.global_id = global_id


def run_online(messages, *, tracklet_config, associator, window_ms: int):
    """Chạy đúng vòng lặp của `python -m mct`, trả về kết quả gán CUỐI CÙNG mỗi tracklet.

    Một tracklet được gán lại qua nhiều cửa sổ; cái đáng tính là Global ID nó mang lúc
    kết thúc — đúng thứ dashboard và SQLite hiển thị.
    """
    engine = Engine(
        tracklet_config=tracklet_config, associator=associator, window_ms=window_ms, store=None
    )
    final: dict[int, int] = {}
    for msg in messages:
        for update in engine.feed(msg):
            final[update.tracklet_id] = update.global_id
    for update in engine.finish():
        final[update.tracklet_id] = update.global_id

    return [
        _Result(tracklet, gid)
        for tid, gid in final.items()
        if (tracklet := engine.builder.by_id(tid)) is not None
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--gt", type=Path, default=None)
    p.add_argument("--homography-dir", type=Path, default=None)
    p.add_argument("--max-cost", type=float, default=0.5)
    p.add_argument("--min-frames", type=int, default=3)
    p.add_argument("--window-ms", type=int, default=1000)
    p.add_argument("--mode", default="max", choices=("max", "centroid"))
    p.add_argument("--ground-gap-policy", default="reject", choices=("allow", "reject"))
    p.add_argument("--max-ground-dist", type=float, default=0.5)
    args = p.parse_args(argv)

    gt_path = args.gt or Path(str(args.fixture).replace(".jsonl", ".gt.json"))
    gt = load_gt(gt_path)
    messages = list(read_jsonl(args.fixture))
    cam_ids = sorted({m.cam_id for m in messages})
    mapper = HomographyMapper.load(args.homography_dir) if args.homography_dir else None

    tracklet_config = TrackletConfig(min_frames=args.min_frames, idle_timeout_ms=2000)
    affinity = AffinityConfig(
        max_cost=args.max_cost,
        similarity_mode=args.mode,  # type: ignore[arg-type]
        max_ground_dist_m=args.max_ground_dist,
        ground_gap_policy=args.ground_gap_policy,
    )

    def _associator() -> Associator:
        return Associator(
            topology=overlapping_topology(cam_ids),
            gallery=Gallery(GalleryConfig(similarity_mode=args.mode)),  # type: ignore[arg-type]
            config=affinity,
            ground_mapper=mapper,
        )

    print(
        f"{args.fixture.name}: {len(messages)} message, {len(cam_ids)} camera, "
        f"{len(set(gt.values()))} danh tính | max_cost={args.max_cost} "
        f"geo={args.ground_gap_policy if mapper else 'tắt'}"
    )

    online = run_online(
        messages,
        tracklet_config=tracklet_config,
        associator=_associator(),
        window_ms=args.window_ms,
    )
    offline_results, _ = run_offline(
        build_tracklets(messages, tracklet_config),
        topology=overlapping_topology(cam_ids),
        config=affinity,
        gallery_config=GalleryConfig(similarity_mode=args.mode),  # type: ignore[arg-type]
        ground_mapper=mapper,
        window_ms=args.window_ms,
    )

    header = (
        f"{'chế độ':8s} {'#tracklet':>9s} {'#gid':>5s} {'vỡ':>4s} {'gộp':>4s} "
        f"{'P':>6s} {'R':>6s} {'F1':>6s}"
    )
    print(header)
    scores = {}
    for name, results in (("online", online), ("offline", offline_results)):
        s = scores[name] = score(results, gt)
        print(
            f"{name:8s} {s['n_tracklet']:9.0f} {s['n_gid']:5.0f} {s['broken']:4.0f} "
            f"{s['merged']:4.0f} {s['precision']:6.3f} {s['recall']:6.3f} {s['f1']:6.3f}"
        )

    delta = scores["online"]["f1"] - scores["offline"]["f1"]
    ratio = scores["online"]["f1"] / scores["offline"]["f1"] if scores["offline"]["f1"] else 0.0
    print(
        f"giá của thời gian thực: ΔF1 = {delta:+.3f} "
        f"(online giữ được {100 * ratio:.1f}% F1 của offline)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
