"""Xuất kết quả của hệ thống sang bố cục TrackEval — cả đơn camera lẫn xuyên camera.

    PYTHONPATH=src python -m tools.export_trackeval \\
        --fixture data/fixtures/lab_4cam.jsonl --db data/mct.db \\
        --gt eval/gt/lab_4cam.gt.json --out eval/trackeval

**Vì sao cần** (CLAUDE.md §7): tới giờ đồ án mới chỉ có F1 theo cặp tracklet — một chỉ số
tự chế, không so được với bất kỳ công bố nào. Chương 6/7 cần MOTA/MOTP/IDF1/HOTA theo đúng
định nghĩa MOT Challenge, tức phải xuất ra định dạng chuẩn rồi để TrackEval chấm.

**Hai chế độ, hai câu hỏi khác nhau:**

- `sct` (single-camera tracking) — mỗi camera là một chuỗi riêng, `id` = `local_track_id`.
  Đo chất lượng của **nvtracker**, không liên quan tới `src/mct`. Đây là con số cần khi nói
  "tracker đơn camera tốt đến đâu".
- `mct` (multi-camera) — nối mọi camera thành MỘT chuỗi ảo bằng offset khung, `id` =
  `global_id`. Đo chất lượng của **module liên kết** — đóng góp chính của đồ án.

Hộp theo từng khung chỉ có trong **fixture**; `global_id` chỉ có trong **SQLite store** (bảng
`appearances` ánh xạ `(cam_id, local_track_id) -> global_id`). Nên công cụ này ghép hai
nguồn: fixture cho hình học, store cho danh tính. Không có `--db` thì chỉ xuất được `sct`.

Ground-truth lấy từ bảng `.gt.json` (cùng định dạng mà `wildtrack_to_fixture.py` và
`ds_wildtrack_gt.py` sinh ra, và `tools/cvat_to_mot.py` sẽ sinh cho dữ liệu tự thu):
`(cam_id, local_track_id) -> gt_global_id`. Với `sct`, `id` của ground-truth chính là
`local_track_id` — nghĩa là **giả định tracklet ground-truth trùng local track**, đúng khi
bảng GT sinh từ chú thích, KHÔNG đúng khi sinh bằng ghép IoU. Xem chú thích ở `--mode sct`.

Chỉ stdlib + `common/`. Chạy được ở đâu cũng được, không cần GPU.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from common.logging import get_logger
from common.motformat import (
    MotRow,
    TrackEvalLayout,
    frame_offset_for,
    to_mot_frame,
    virtual_frame,
    write_gt,
    write_results,
    write_seqinfo,
    write_seqmap,
)
from common.schema import FrameMessage, read_jsonl

log = get_logger("tools.export_trackeval")

MODES = ("sct", "mct")
CROSS_CAMERA_SEQ = "all"


def load_gt_table(path: Path) -> dict[tuple[str, int], int]:
    """`.gt.json` -> {(cam_id, local_track_id): gt_global_id}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        (t["cam_id"], int(t["local_track_id"])): int(t["gt_global_id"]) for t in data["tracklets"]
    }


def load_global_ids(db_path: Path) -> dict[tuple[str, int], int]:
    """SQLite store -> {(cam_id, local_track_id): global_id}.

    Một `(cam_id, local_track_id)` có thể có nhiều dòng `appearances` nếu tracklet bị cắt
    rồi cấp lại cùng local id; lấy dòng MỚI NHẤT theo `start_ms`, khớp với thứ dashboard
    hiển thị.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT cam_id, local_track_id, global_id FROM appearances ORDER BY start_ms"
        ).fetchall()
    finally:
        con.close()
    return {(str(cam), int(local)): int(gid) for cam, local, gid in rows}


def _messages_by_cam(messages: list[FrameMessage]) -> dict[str, list[FrameMessage]]:
    out: dict[str, list[FrameMessage]] = defaultdict(list)
    for msg in messages:
        out[msg.cam_id].append(msg)
    return out


def export_sct(
    messages: list[FrameMessage],
    gt: dict[tuple[str, int], int],
    layout: TrackEvalLayout,
    *,
    tracker: str,
    fps: float,
) -> list[str]:
    """Mỗi camera một chuỗi. `id` = `local_track_id` ở CẢ hai phía.

    Ground-truth ở chế độ này chỉ đúng khi bảng `.gt.json` sinh TỪ CHÚ THÍCH (mỗi
    `local_track_id` là một người thật, như `wildtrack_to_fixture.py` hoặc `cvat_to_mot.py`
    tạo ra). Nếu bảng sinh bằng ghép IoU với đầu ra tracker (`ds_wildtrack_gt.py`) thì
    `local_track_id` CHÍNH LÀ id của tracker, và chấm nó với chính nó ra MOTA hoàn hảo một
    cách vô nghĩa. Công cụ cảnh báo chứ không chặn — có lúc vẫn cần xuất để soi bằng mắt.
    """
    seqs: list[str] = []
    for cam_id, msgs in sorted(_messages_by_cam(messages).items()):
        gt_rows: list[MotRow] = []
        res_rows: list[MotRow] = []
        for msg in msgs:
            frame = to_mot_frame(msg.frame_id)
            for det in msg.detections:
                row = MotRow(
                    frame=frame,
                    track_id=int(det.local_track_id),
                    x=det.bbox[0],
                    y=det.bbox[1],
                    w=det.bbox[2],
                    h=det.bbox[3],
                    confidence=max(0.0, float(det.confidence)),
                )
                res_rows.append(row)
                if (cam_id, int(det.local_track_id)) in gt:
                    gt_rows.append(row)

        write_gt(layout.gt_file(cam_id), gt_rows)
        write_results(layout.result_file(tracker, cam_id), res_rows)
        write_seqinfo(
            layout.seqinfo_file(cam_id),
            name=cam_id,
            width=msgs[0].frame_width,
            height=msgs[0].frame_height,
            length=max(msg.frame_id for msg in msgs) + 1,
            fps=fps,
        )
        seqs.append(cam_id)
        log.info("sct %s: %d dòng GT, %d dòng kết quả", cam_id, len(gt_rows), len(res_rows))
    return seqs


def export_mct(
    messages: list[FrameMessage],
    gt: dict[tuple[str, int], int],
    global_ids: dict[tuple[str, int], int],
    layout: TrackEvalLayout,
    *,
    tracker: str,
    fps: float,
) -> tuple[str, dict[str, int]]:
    """Nối mọi camera thành một chuỗi ảo. `id` = `gt_global_id` / `global_id`.

    Detection nào không tra được danh tính thì bỏ ở phía tương ứng: thiếu ở ground-truth
    nghĩa là người đó không được chú thích (không chấm), thiếu ở kết quả nghĩa là engine
    chưa gán Global ID (tính là bỏ sót — đúng như vậy).
    """
    by_cam = _messages_by_cam(messages)
    cam_ids = sorted(by_cam)
    offset = frame_offset_for(max(msg.frame_id for msg in msgs) + 1 for msgs in by_cam.values())

    gt_rows: list[MotRow] = []
    res_rows: list[MotRow] = []
    stats = {"n_detections": 0, "n_gt": 0, "n_result": 0, "n_unassigned": 0}

    for cam_index, cam_id in enumerate(cam_ids):
        for msg in by_cam[cam_id]:
            frame = virtual_frame(cam_index, msg.frame_id, offset=offset)
            for det in msg.detections:
                stats["n_detections"] += 1
                key = (cam_id, int(det.local_track_id))
                box = (det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3])
                if key in gt:
                    gt_rows.append(MotRow(frame, gt[key], *box))
                    stats["n_gt"] += 1
                gid = global_ids.get(key)
                if gid is None:
                    stats["n_unassigned"] += 1
                    continue
                res_rows.append(
                    MotRow(frame, gid, *box, confidence=max(0.0, float(det.confidence)))
                )
                stats["n_result"] += 1

    write_gt(layout.gt_file(CROSS_CAMERA_SEQ), gt_rows)
    write_results(layout.result_file(tracker, CROSS_CAMERA_SEQ), res_rows)
    first = by_cam[cam_ids[0]][0]
    write_seqinfo(
        layout.seqinfo_file(CROSS_CAMERA_SEQ),
        name=CROSS_CAMERA_SEQ,
        width=first.frame_width,
        height=first.frame_height,
        length=len(cam_ids) * offset,
        fps=fps,
    )
    stats["frame_offset"] = offset
    stats["n_cameras"] = len(cam_ids)
    return CROSS_CAMERA_SEQ, stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fixture", type=Path, required=True, help="nguồn hộp theo từng khung")
    p.add_argument("--gt", type=Path, default=None, help="bảng .gt.json; mặc định <fixture>")
    p.add_argument("--db", type=Path, default=None, help="SQLite store — bắt buộc cho --mode mct")
    p.add_argument("--out", type=Path, default=Path("eval/trackeval"))
    p.add_argument("--benchmark", default="MCT")
    p.add_argument("--tracker", default="mct-engine")
    p.add_argument("--fps", type=float, default=25.0, help="chỉ ghi vào seqinfo.ini")
    p.add_argument("--mode", choices=(*MODES, "both"), default="both")
    args = p.parse_args(argv)

    gt_path = args.gt or Path(str(args.fixture).replace(".jsonl", ".gt.json"))
    gt = load_gt_table(gt_path)
    messages = [m for m in read_jsonl(args.fixture) if m.detections]
    if not messages:
        raise SystemExit(f"{args.fixture}: không có detection nào")

    modes = MODES if args.mode == "both" else (args.mode,)
    if "mct" in modes and args.db is None:
        raise SystemExit("--mode mct cần --db (bảng appearances chứa Global ID)")

    for mode in modes:
        layout = TrackEvalLayout(root=args.out, benchmark=args.benchmark, split=mode)
        if mode == "sct":
            seqs = export_sct(messages, gt, layout, tracker=args.tracker, fps=args.fps)
        else:
            global_ids = load_global_ids(args.db)
            seq, stats = export_mct(
                messages, gt, global_ids, layout, tracker=args.tracker, fps=args.fps
            )
            seqs = [seq]
            log.info(
                "mct: %d camera, offset %d khung, %d dòng GT / %d dòng kết quả "
                "(%d detection chưa có Global ID)",
                stats["n_cameras"],
                stats["frame_offset"],
                stats["n_gt"],
                stats["n_result"],
                stats["n_unassigned"],
            )
        write_seqmap(layout.seqmap_file(), seqs)
        log.info("%s: %d chuỗi -> %s", mode, len(seqs), layout.gt_folder / layout.dataset)

    log.info("chấm bằng: PYTHONPATH=src python eval/run_trackeval.py --root %s", args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
