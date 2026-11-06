"""Oracle tracker cho dữ liệu tự thu (M6): tách lỗi tracker đơn camera khỏi lỗi liên kết.

    PYTHONPATH=src python -m tools.lab_oracle_fixture \\
        --fixture data/fixtures/lab_s2_1_r1.jsonl --gt-fixture data/fixtures/lab_s2_1_gt.jsonl \\
        --det-out data/fixtures/lab_s2_1_r1_gtdet.jsonl \\
        --oracle-out data/fixtures/lab_s2_1_r1_oracle.jsonl

Bản lab của `tools.reembed_fixture --oracle-out` (phiên 24, WildTrack). Sinh HAI fixture từ
một fixture pipeline, chỉ khác nhau đúng một cột:

- `--det-out` (C): các detection của pipeline khớp được một hộp GT (IoU >= `--min-iou`),
  giữ nguyên `local_track_id` của NvDCF.
- `--oracle-out` (B): đúng các detection đó, nhưng `local_track_id` = Global ID ground-truth.

Hộp, embedding, `ts_ms`, `stamps` giữ nguyên từng bit. Vậy C → B chỉ đổi danh tính cục bộ, và
chênh lệch điểm quy được về đúng một nguyên nhân: tracker đơn camera vỡ/trộn tracklet. Cặp
C/B không qua bộ trích embedding nào khác, nên không cần ảnh gốc hay ONNX.

**Khác WildTrack: chú thích nhảy khung** (5 fps trên video 25 fps). Engine phải thấy đủ 25 fps
như lần chạy thật (`min_frames`, `max_embeddings`, `ground_time_tol_ms` đều tính theo khung
hoặc ms), nên hộp GT được NỘI SUY TUYẾN TÍNH giữa hai khung chú thích liền kề của cùng một
track — đúng cách CVAT nội suy giữa hai khung khoá. Không nội suy qua khoảng hở lớn hơn bước
chú thích (người ra khỏi khung hình). Phương án bị loại: chỉ giữ các khung chú thích (5 fps);
nó đổi luôn tốc độ khung mà engine thấy, tức thêm một biến thứ hai.

Message không còn detection nào vẫn được giữ (đồng hồ khung của camera không đổi).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

from common.logging import get_logger
from common.schema import FrameMessage, read_jsonl, write_jsonl
from tools.ds_wildtrack_gt import match_frame
from tools.export_trackeval import load_gt_table

log = get_logger("tools.lab_oracle_fixture")

Bbox = tuple[float, float, float, float]


def interpolated_gt(
    gt_messages: list[FrameMessage],
    table: dict[tuple[str, int], int],
    *,
    step: int,
) -> dict[tuple[str, int], list[tuple[int, Bbox]]]:
    """(cam_id, frame_id) -> [(gt_global_id, bbox)] ở MỌI khung, nội suy giữa hai khung chú thích.

    Chỉ nội suy khi hai khung chú thích liền kề của cùng một track cách nhau đúng `step`;
    khoảng hở lớn hơn là người đã ra khỏi khung hình.
    """
    tracks: dict[tuple[str, int], list[tuple[int, Bbox]]] = defaultdict(list)
    for msg in gt_messages:
        for det in msg.detections:
            gid = table.get((msg.cam_id, int(det.local_track_id)))
            if gid is None:
                continue
            box = tuple(float(v) for v in det.bbox)
            tracks[(msg.cam_id, gid)].append((int(msg.frame_id), box))  # type: ignore[arg-type]

    out: dict[tuple[str, int], list[tuple[int, Bbox]]] = defaultdict(list)
    for (cam, gid), points in tracks.items():
        points.sort()
        for (f0, b0), (f1, b1) in pairwise(points):
            if f1 - f0 > step:
                continue
            for f in range(f0, f1):
                t = (f - f0) / (f1 - f0)
                box = tuple(a + (b - a) * t for a, b in zip(b0, b1, strict=True))
                out[(cam, f)].append((gid, box))  # type: ignore[arg-type]
        for f, b in points:
            # Khung chú thích cuối của mỗi đoạn (và track chỉ có một khung) chưa được thêm ở trên.
            if not any(g == gid for g, _ in out.get((cam, f), [])):
                out[(cam, f)].append((gid, b))
    return dict(out)


def split_fixture(
    messages: list[FrameMessage],
    gt_by_frame: dict[tuple[str, int], list[tuple[int, Bbox]]],
    *,
    min_iou: float,
) -> tuple[list[FrameMessage], list[FrameMessage], dict[str, int]]:
    """Fixture pipeline -> (C: id NvDCF, B: id GT) trên cùng tập detection khớp GT."""
    det_out: list[FrameMessage] = []
    oracle_out: list[FrameMessage] = []
    stats = {"n_detections": 0, "n_kept": 0}
    for msg in messages:
        gt_rows = gt_by_frame.get((msg.cam_id, int(msg.frame_id)), [])
        pairs = match_frame(
            [tuple(float(v) for v in d.bbox) for d in msg.detections],  # type: ignore[misc]
            [box for _, box in gt_rows],
            min_iou=min_iou,
        )
        gid_of = {det_i: gt_rows[gt_i][0] for det_i, gt_i, _ in pairs}
        kept = [i for i in range(len(msg.detections)) if i in gid_of]
        stats["n_detections"] += len(msg.detections)
        stats["n_kept"] += len(kept)
        det_out.append(replace(msg, detections=[msg.detections[i] for i in kept]))
        oracle_out.append(
            replace(
                msg,
                detections=[
                    replace(msg.detections[i], local_track_id=int(gid_of[i])) for i in kept
                ],
            )
        )
    return det_out, oracle_out, stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fixture", type=Path, required=True, help="fixture .jsonl của pipeline")
    p.add_argument("--gt-fixture", type=Path, required=True)
    p.add_argument("--gt-fixture-table", type=Path, default=None)
    p.add_argument("--det-out", type=Path, required=True, help="C: id NvDCF, chỉ detection khớp GT")
    p.add_argument("--oracle-out", type=Path, required=True, help="B: cùng detection, id GT")
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--min-iou", type=float, default=0.5)
    p.add_argument("--step", type=int, default=5, help="bước chú thích (khung video)")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    table_path = args.gt_fixture_table or Path(str(args.gt_fixture).replace(".jsonl", ".gt.json"))
    gt_by_frame = interpolated_gt(
        list(read_jsonl(args.gt_fixture)), load_gt_table(table_path), step=args.step
    )
    det_msgs, oracle_msgs, stats = split_fixture(
        list(read_jsonl(args.fixture)), gt_by_frame, min_iou=args.min_iou
    )
    for path, msgs in ((args.det_out, det_msgs), (args.oracle_out, oracle_msgs)):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, msgs)
    stats["keep_rate"] = (
        round(stats["n_kept"] / stats["n_detections"], 4) if stats["n_detections"] else 0.0
    )
    if args.report is not None:
        args.report.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    log.info(
        "%s: giữ %d/%d detection (%.3f)",
        args.fixture.name,
        stats["n_kept"],
        stats["n_detections"],
        stats["keep_rate"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
