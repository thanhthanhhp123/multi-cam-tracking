"""Gán ground-truth Global ID cho fixture của pipeline, dùng một fixture ground-truth BẤT KỲ.

    PYTHONPATH=src python -m tools.assign_gt \\
        --fixture data/fixtures/lab_s1_r1.jsonl \\
        --gt-fixture data/fixtures/lab_s1_gt.jsonl \\
        --report data/fixtures/lab_s1_r1.gt-report.json

**Vì sao cần, khi đã có `tools/ds_wildtrack_gt.py`.** Công cụ đó đọc thẳng chú thích của
WildTrack (`annotations_positions/*.json`, `personID`, `cam_id` dạng `camNN` ↔ view 1..7).
Dữ liệu tự thu ở M6 đi qua CVAT: `tools/cvat_to_mot.py --fixture-out` cho ra fixture ground-
truth cùng schema với kết quả pipeline. Công cụ này là bản TỔNG QUÁT của bước gán: hai fixture
cùng schema vào, bảng `.gt.json` ra — không biết gì về dataset cụ thể.

Cách gán giống hệt `ds_wildtrack_gt` (dùng lại chính hàm của nó, để hai đường không lệch nhau):

1. `(cam_id, frame_id)` của fixture pipeline tra thẳng về khung ground-truth — đúng khi
   pipeline chạy trên CHÍNH file video đã chú thích (`tools/sync_recordings.py` sinh ra cả
   hai). `--frame-offset` cho trường hợp hai bên lệch một hằng số khung.
2. Trong từng khung: ghép hộp bằng IoU, Hungarian một-một, giữ cặp `IoU >= --min-iou`.
3. Mỗi `local_track_id` bỏ phiếu theo `gt_global_id`; track thiếu khung khớp hoặc độ thuần
   khiết thấp bị LOẠI khỏi bảng thay vì gán bừa.

Khung không có trong fixture ground-truth (chú thích nhảy khung) đơn giản là không bỏ phiếu;
khác với khung CÓ chú thích mà rỗng, nơi mọi hộp của pipeline đều không khớp.

Bảng ra có `start_ms`/`end_ms` theo thời gian của LẦN CHẠY pipeline (từ fixture pipeline),
khớp với SQLite của engine — khác với bảng của fixture ground-truth.

Chỉ cần numpy + scipy (đã có trong venv test), không cần GPU.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.logging import get_logger
from common.schema import FrameMessage, read_jsonl
from tools.ds_wildtrack_gt import TrackVotes, match_frame
from tools.export_trackeval import load_gt_table

log = get_logger("tools.assign_gt")

Bbox = tuple[float, float, float, float]


def index_gt(
    gt_messages: list[FrameMessage], table: dict[tuple[str, int], int]
) -> dict[tuple[str, int], list[tuple[int, Bbox]]]:
    """(cam_id, frame_id) -> [(gt_global_id, bbox)] của các hộp có danh tính trong bảng.

    Khung có chú thích nhưng không có ai vẫn có khoá (danh sách rỗng): `collect_votes` cần
    phân biệt "khung rỗng" với "khung không chú thích" để đếm đúng.
    """
    out: dict[tuple[str, int], list[tuple[int, Bbox]]] = {}
    for msg in gt_messages:
        rows = out.setdefault((msg.cam_id, int(msg.frame_id)), [])
        for det in msg.detections:
            gid = table.get((msg.cam_id, int(det.local_track_id)))
            if gid is not None:
                rows.append((gid, tuple(float(v) for v in det.bbox)))  # type: ignore[arg-type]
    return out


def collect_votes(
    messages: list[FrameMessage],
    gt_by_frame: dict[tuple[str, int], list[tuple[int, Bbox]]],
    *,
    min_iou: float,
    frame_offset: int = 0,
) -> tuple[dict[tuple[str, int], TrackVotes], dict[str, int]]:
    """Duyệt fixture pipeline, ghép IoU theo từng khung CÓ chú thích, cộng phiếu cho track."""
    tracks: dict[tuple[str, int], TrackVotes] = {}
    stats = {
        "n_messages": 0,
        "n_messages_annotated": 0,
        "n_detections": 0,
        "n_detections_annotated": 0,
        "n_matched": 0,
    }
    for msg in messages:
        stats["n_messages"] += 1
        for det in msg.detections:
            key = (msg.cam_id, int(det.local_track_id))
            if key not in tracks:
                tracks[key] = TrackVotes(msg.cam_id, int(det.local_track_id))
            tracks[key].n_detections += 1
            tracks[key].ts.append(int(msg.ts_ms))
        stats["n_detections"] += len(msg.detections)

        gt_rows = gt_by_frame.get((msg.cam_id, int(msg.frame_id) + frame_offset))
        if gt_rows is None:
            continue  # khung không chú thích: không có gì để so
        stats["n_messages_annotated"] += 1
        stats["n_detections_annotated"] += len(msg.detections)
        pairs = match_frame(
            [tuple(float(v) for v in d.bbox) for d in msg.detections],  # type: ignore[misc]
            [box for _, box in gt_rows],
            min_iou=min_iou,
        )
        for det_i, gt_i, _ in pairs:
            det = msg.detections[det_i]
            track = tracks[(msg.cam_id, int(det.local_track_id))]
            track.votes[gt_rows[gt_i][0]] += 1
            track.n_matched += 1
            stats["n_matched"] += 1
    return tracks, stats


def build_table(
    tracks: dict[tuple[str, int], TrackVotes], *, min_purity: float, min_matched: int
) -> tuple[list[dict], dict[str, int]]:
    """Phiếu -> các dòng của bảng `.gt.json`. Track không đủ tin cậy bị loại, không gán bừa."""
    rows: list[dict] = []
    drops = {"khong_khop": 0, "it_khung": 0, "khong_thuan": 0}
    for track in sorted(tracks.values(), key=lambda t: t.key):
        gid, purity = track.winner()
        if gid < 0:
            drops["khong_khop"] += 1
        elif track.n_matched < min_matched:
            drops["it_khung"] += 1
        elif purity < min_purity:
            drops["khong_thuan"] += 1
        else:
            rows.append(
                {
                    "cam_id": track.cam_id,
                    "local_track_id": track.local_track_id,
                    "gt_global_id": gid,
                    "start_ms": min(track.ts),
                    "end_ms": max(track.ts),
                    "n_frames": track.n_detections,
                    "n_matched": track.n_matched,
                    "purity": round(purity, 4),
                }
            )
    return rows, drops


def summarize(
    tracks: dict[tuple[str, int], TrackVotes],
    rows: list[dict],
    stats: dict[str, int],
    drops: dict[str, int],
    *,
    n_gt_boxes_annotated: int,
) -> dict:
    purities = [t.winner()[1] for t in tracks.values() if t.n_matched > 0]
    by_cam: dict[str, int] = defaultdict(int)
    for r in rows:
        by_cam[r["cam_id"]] += 1
    n_det = stats["n_detections_annotated"]
    return {
        **stats,
        # Trên khung CÓ chú thích: tỉ lệ hộp pipeline khớp người thật (≈ precision ở IoU)
        # và tỉ lệ hộp người thật được khớp (≈ recall). Đây chưa phải chỉ số báo cáo —
        # TrackEval mới là — mà là phép kiểm nhanh rằng hai fixture thẳng hàng: lệch số
        # khung thì cả hai tụt về gần 0.
        "match_rate": stats["n_matched"] / n_det if n_det else 0.0,
        "gt_recall": stats["n_matched"] / n_gt_boxes_annotated if n_gt_boxes_annotated else 0.0,
        "n_gt_boxes_annotated": n_gt_boxes_annotated,
        "n_tracks": len(tracks),
        "n_tracks_kept": len(rows),
        "n_tracks_kept_by_cam": dict(sorted(by_cam.items())),
        "n_tracks_multi_identity": sum(1 for t in tracks.values() if len(t.votes) > 1),
        "drops": drops,
        "purity_median": float(np.median(purities)) if purities else 0.0,
        "purity_p10": float(np.percentile(purities, 10)) if purities else 0.0,
        "n_identities": len({r["gt_global_id"] for r in rows}),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fixture", type=Path, required=True, help="fixture .jsonl của pipeline")
    p.add_argument("--gt-fixture", type=Path, required=True, help="fixture ground-truth .jsonl")
    p.add_argument(
        "--gt-fixture-table", type=Path, default=None, help="mặc định: <gt-fixture>.gt.json"
    )
    p.add_argument("--out", type=Path, default=None, help="mặc định: <fixture>.gt.json")
    p.add_argument("--report", type=Path, default=None, help="ghi thống kê ghép nối ra JSON")
    p.add_argument("--min-iou", type=float, default=0.5)
    p.add_argument("--min-purity", type=float, default=0.7)
    p.add_argument("--min-matched", type=int, default=3)
    p.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="khung GT = khung pipeline + offset (0 khi chạy trên đúng file đã chú thích)",
    )
    args = p.parse_args(argv)

    table_path = args.gt_fixture_table or Path(str(args.gt_fixture).replace(".jsonl", ".gt.json"))
    table = load_gt_table(table_path)
    gt_by_frame = index_gt(list(read_jsonl(args.gt_fixture)), table)
    messages = list(read_jsonl(args.fixture))
    if not messages:
        raise SystemExit(f"{args.fixture}: không có message nào")

    missing = sorted({m.cam_id for m in messages} - {cam for cam, _ in gt_by_frame})
    if missing:
        log.warning("camera không có ground-truth, sẽ không gán được: %s", ", ".join(missing))

    tracks, stats = collect_votes(
        messages, gt_by_frame, min_iou=args.min_iou, frame_offset=args.frame_offset
    )
    rows, drops = build_table(tracks, min_purity=args.min_purity, min_matched=args.min_matched)
    if not rows:
        raise SystemExit(
            "không gán được tracklet nào — kiểm tra hai fixture có cùng file video / cùng "
            "cam_id / --frame-offset, rồi mới tới --min-iou"
        )

    annotated = {(m.cam_id, int(m.frame_id) + args.frame_offset) for m in messages}
    n_gt_boxes = sum(len(v) for k, v in gt_by_frame.items() if k in annotated)
    report = summarize(tracks, rows, stats, drops, n_gt_boxes_annotated=n_gt_boxes)
    meta = {
        "source": "pipeline+gt_fixture",
        "fixture": args.fixture.name,
        "gt_fixture": args.gt_fixture.name,
        "min_iou": args.min_iou,
        "min_purity": args.min_purity,
        "min_matched": args.min_matched,
        "frame_offset": args.frame_offset,
        "n_tracklets": len(rows),
        "n_identities": report["n_identities"],
        "matching": report,
        "notes": (
            "local_track_id do tracker của pipeline cấp, gt_global_id suy ra bằng ghép IoU với "
            "fixture ground-truth rồi bỏ phiếu đa số; start_ms/end_ms theo đồng hồ lần chạy."
        ),
    }
    out = args.out or Path(str(args.fixture).replace(".jsonl", ".gt.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"scenario": out.name.removesuffix(".gt.json"), "meta": meta, "tracklets": rows},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    log.info(
        "%s: %d/%d track gán được (%d danh tính), match_rate %.3f, gt_recall %.3f, loại %s",
        out,
        len(rows),
        len(tracks),
        report["n_identities"],
        report["match_rate"],
        report["gt_recall"],
        drops,
    )
    if report["match_rate"] < 0.2:
        log.warning(
            "match_rate %.3f rất thấp — dấu hiệu lệch số khung/cam_id giữa hai fixture",
            report["match_rate"],
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
