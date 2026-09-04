"""Gán ground-truth Global ID cho fixture do pipeline DeepStream sinh ra trên video WildTrack.

    PYTHONPATH=src python -m tools.ds_wildtrack_gt \\
        --fixture tests/fixtures/ds_wildtrack_7cam.jsonl \\
        --wildtrack-dir data/wildtrack --report data/ds_wildtrack_gt.report.json

**Mắt xích còn thiếu của M3.** `tools/wildtrack_to_fixture.py` sinh fixture *cùng lúc* với
bảng ground-truth, vì `local_track_id` ở đó do chính nó đặt ra từ `personID`. Fixture chạy
qua DeepStream thì không: bbox do YOLO11s tìm, `local_track_id` do NvDCF cấp, không cái nào
biết `personID` của WildTrack. Không có bảng đáp án thì không đánh giá được, và câu hỏi
"ngưỡng `max_cost` chỉnh trên đường ONNX Runtime có chuyển sang đường DeepStream không"
vẫn treo.

Cách gán ở đây:

1. `frame_id` của mỗi `FrameMessage` tra thẳng về khung chú thích WildTrack — đúng nhờ quy
   ước đóng video của `tools/wildtrack_to_video.py` (khung thứ i của video = khung chú thích
   thứ i). Không khớp theo `ts_ms`: đồng hồ của lần chạy pipeline không liên quan gì tới
   `BASE_TS_MS` của fixture ONNX Runtime.
2. Trong từng (camera, khung): ghép bbox của DeepStream với bbox ground-truth bằng **IoU**,
   gán tối ưu một-một bằng Hungarian (`scipy`), giữ lại cặp có `IoU >= --min-iou`.
3. Mỗi `local_track_id` bỏ phiếu: `personID` chiếm đa số thắng. Track có độ thuần khiết
   dưới `--min-purity`, hoặc khớp được quá ít khung, bị **loại khỏi bảng** thay vì gán bừa —
   `eval/eval_wildtrack.py` chỉ chấm những tracklet có trong bảng, nên loại là an toàn, còn
   gán sai thì đầu độc cả điểm số.

Đầu ra `<fixture>.gt.json` **cùng định dạng** với `tools/wildtrack_to_fixture.py`, nên
`eval/eval_wildtrack.py --diagnose --sweep` chạy được ngay, không sửa gì. Đó chính là phép
so sánh cần: cùng dataset, cùng bộ chỉ số, khác đúng một biến — đường trích embedding.

HẠN CHẾ, ghi để đọc số cho đúng: tracklet ở đây do tracker thật cắt nên phân mảnh hơn hẳn
"SCT lý tưởng" của fixture ONNX Runtime, và một phần người bị detector bỏ sót. Vì vậy **F1
tuyệt đối giữa hai fixture không so trực tiếp được**; thứ so được là *vị trí ngưỡng tốt
nhất* và phân bố cosine cùng/khác danh tính (`--diagnose`).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.logging import get_logger
from common.schema import FrameMessage, read_jsonl
from tools.wildtrack_to_fixture import (
    N_VIEWS,
    RawDetection,
    TrackletGT,
    parse_raw_detections,
    write_ground_truth,
)

log = get_logger("tools.ds_wildtrack_gt")

Bbox = tuple[float, float, float, float]


def view_idx_for_cam(cam_id: str) -> int:
    """ "cam01" -> 0. Nghịch đảo của `cam_id_for_view`."""
    try:
        idx = int(str(cam_id).removeprefix("cam")) - 1
    except ValueError as exc:
        raise ValueError(f"cam_id không đúng dạng camNN: {cam_id!r}") from exc
    if not 0 <= idx < N_VIEWS:
        raise ValueError(f"cam_id ngoài phạm vi WildTrack 1..{N_VIEWS}: {cam_id!r}")
    return idx


def iou(a: Bbox, b: Bbox) -> float:
    """IoU của hai hộp (x, y, w, h)."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def match_frame(
    det_boxes: list[Bbox], gt_boxes: list[Bbox], *, min_iou: float
) -> list[tuple[int, int, float]]:
    """Ghép một-một tối ưu theo IoU. Trả về [(chỉ số detection, chỉ số gt, iou)].

    Hungarian chứ không tham lam: hai người đứng chồng nhau thì cách tham lam dễ ăn nhầm
    hộp của người bên cạnh, và một lá phiếu sai ở đây kéo cả tracklet sang nhầm danh tính.
    """
    if not det_boxes or not gt_boxes:
        return []
    matrix = np.array([[iou(d, g) for g in gt_boxes] for d in det_boxes], dtype=np.float64)
    rows, cols = linear_sum_assignment(-matrix)
    return [
        (int(r), int(c), float(matrix[r, c]))
        for r, c in zip(rows, cols, strict=True)
        if matrix[r, c] >= min_iou
    ]


@dataclass(slots=True)
class TrackVotes:
    """Phiếu bầu danh tính của một `local_track_id` trên một camera."""

    cam_id: str
    local_track_id: int
    votes: Counter[int] = field(default_factory=Counter)
    n_detections: int = 0
    n_matched: int = 0
    ts: list[int] = field(default_factory=list)
    world: list[tuple[float, float]] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, int]:
        return (self.cam_id, self.local_track_id)

    def winner(self) -> tuple[int, float]:
        """(personID thắng, độ thuần khiết). Không có phiếu -> (-1, 0.0)."""
        if not self.votes:
            return (-1, 0.0)
        pid, n = self.votes.most_common(1)[0]
        return (int(pid), n / max(1, self.n_matched))


def gt_index(
    raw: list[RawDetection],
) -> dict[tuple[int, int], list[RawDetection]]:
    """(view_idx, frame_idx) -> các quan sát ground-truth trong khung đó."""
    out: dict[tuple[int, int], list[RawDetection]] = defaultdict(list)
    for det in raw:
        out[(det.view_idx, det.frame_idx)].append(det)
    return out


def collect_votes(
    messages: list[FrameMessage],
    gt_by_frame: dict[tuple[int, int], list[RawDetection]],
    *,
    min_iou: float,
) -> tuple[dict[tuple[str, int], TrackVotes], dict[str, int]]:
    """Duyệt fixture, ghép IoU theo từng khung, cộng dồn phiếu cho từng local track."""
    tracks: dict[tuple[str, int], TrackVotes] = {}
    stats = {"n_messages": 0, "n_detections": 0, "n_matched": 0, "n_frames_no_gt": 0}

    for msg in messages:
        stats["n_messages"] += 1
        if not msg.detections:
            continue
        view_idx = view_idx_for_cam(msg.cam_id)
        gt_dets = gt_by_frame.get((view_idx, int(msg.frame_id)), [])
        stats["n_detections"] += len(msg.detections)
        if not gt_dets:
            stats["n_frames_no_gt"] += 1

        for det in msg.detections:
            key = (msg.cam_id, int(det.local_track_id))
            track = tracks.get(key)
            if track is None:
                track = tracks[key] = TrackVotes(msg.cam_id, int(det.local_track_id))
            track.n_detections += 1

        pairs = match_frame(
            [tuple(float(v) for v in d.bbox) for d in msg.detections],  # type: ignore[misc]
            [g.bbox for g in gt_dets],
            min_iou=min_iou,
        )
        for det_i, gt_i, _ in pairs:
            det = msg.detections[det_i]
            gt = gt_dets[gt_i]
            track = tracks[(msg.cam_id, int(det.local_track_id))]
            track.votes[gt.person_id] += 1
            track.n_matched += 1
            track.ts.append(int(msg.ts_ms))
            track.world.append(gt.world_xy)
            stats["n_matched"] += 1

    return tracks, stats


def build_tracklet_table(
    tracks: dict[tuple[str, int], TrackVotes], *, min_purity: float, min_matched: int
) -> tuple[list[TrackletGT], dict[str, int]]:
    """Phiếu -> bảng ground-truth. Track không đủ tin cậy bị loại, không gán bừa."""
    kept: list[TrackletGT] = []
    drops = {"khong_khop": 0, "it_khung": 0, "khong_thuan": 0}

    for track in sorted(tracks.values(), key=lambda t: t.key):
        pid, purity = track.winner()
        if pid < 0:
            drops["khong_khop"] += 1
            continue
        if track.n_matched < min_matched:
            drops["it_khung"] += 1
            continue
        if purity < min_purity:
            drops["khong_thuan"] += 1
            continue
        world = np.array(track.world, dtype=np.float64).mean(axis=0)
        kept.append(
            TrackletGT(
                cam_id=track.cam_id,
                local_track_id=track.local_track_id,
                gt_global_id=pid,
                start_ms=min(track.ts),
                end_ms=max(track.ts),
                world_xy_m=(float(world[0]), float(world[1])),
                n_frames=track.n_matched,
            )
        )
    return kept, drops


def summarize(
    tracks: dict[tuple[str, int], TrackVotes],
    tracklets: list[TrackletGT],
    stats: dict[str, int],
    drops: dict[str, int],
    *,
    n_gt_detections: int,
    n_ann_frames: int,
    max_frame_id: int,
) -> dict:
    purities = [t.winner()[1] for t in tracks.values() if t.n_matched > 0]
    split = sum(1 for t in tracks.values() if len(t.votes) > 1)
    return {
        "n_messages": stats["n_messages"],
        "n_detections": stats["n_detections"],
        "n_detections_matched": stats["n_matched"],
        "match_rate": stats["n_matched"] / stats["n_detections"] if stats["n_detections"] else 0.0,
        "n_gt_detections": n_gt_detections,
        "gt_recall": stats["n_matched"] / n_gt_detections if n_gt_detections else 0.0,
        "n_tracks": len(tracks),
        "n_tracks_kept": len(tracklets),
        "n_tracks_multi_identity": split,
        "drops": drops,
        "purity_median": float(np.median(purities)) if purities else 0.0,
        "purity_p10": float(np.percentile(purities, 10)) if purities else 0.0,
        "n_identities": len({t.gt_global_id for t in tracklets}),
        "n_ann_frames": n_ann_frames,
        "max_frame_id": max_frame_id,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fixture", type=Path, required=True, help="fixture .jsonl từ DeepStream")
    p.add_argument("--wildtrack-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="mặc định: <fixture>.gt.json")
    p.add_argument("--report", type=Path, default=None, help="ghi thống kê ghép nối ra JSON")
    p.add_argument("--min-iou", type=float, default=0.5, help="ngưỡng ghép bbox")
    p.add_argument(
        "--min-purity",
        type=float,
        default=0.7,
        help="tỉ lệ phiếu của personID thắng; dưới ngưỡng thì loại track khỏi bảng",
    )
    p.add_argument("--min-matched", type=int, default=3, help="số khung khớp tối thiểu")
    p.add_argument(
        "--min-box-area",
        type=float,
        default=0.0,
        help="lọc bbox ground-truth nhỏ (px^2). 0 = giữ tất cả, khác wildtrack_to_fixture",
    )
    p.add_argument("--frame-stride", type=int, default=1, help="phải khớp lúc đóng video")
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args(argv)

    messages = list(read_jsonl(args.fixture))
    if not messages:
        raise SystemExit(f"{args.fixture}: không có message nào")
    cam_ids = sorted({m.cam_id for m in messages})
    view_indices = sorted(view_idx_for_cam(c) for c in cam_ids)

    raw, n_ann_frames = parse_raw_detections(
        args.wildtrack_dir / "annotations_positions",
        view_indices=view_indices,
        stride=args.frame_stride,
        max_frames=args.max_frames,
        min_box_area=args.min_box_area,
    )
    max_frame_id = max(int(m.frame_id) for m in messages)
    if max_frame_id >= n_ann_frames:
        log.warning(
            "frame_id lớn nhất là %d nhưng chỉ có %d khung chú thích — video có thể chứa "
            "khung ngoài tập chú thích, ánh xạ sẽ lệch",
            max_frame_id,
            n_ann_frames,
        )

    tracks, stats = collect_votes(messages, gt_index(raw), min_iou=args.min_iou)
    tracklets, drops = build_tracklet_table(
        tracks, min_purity=args.min_purity, min_matched=args.min_matched
    )
    if not tracklets:
        raise SystemExit("không gán được tracklet nào — xem lại --min-iou / ánh xạ frame_id")

    report = summarize(
        tracks,
        tracklets,
        stats,
        drops,
        n_gt_detections=len(raw),
        n_ann_frames=n_ann_frames,
        max_frame_id=max_frame_id,
    )
    meta = {
        "source": "deepstream+wildtrack",
        "fixture": args.fixture.name,
        "cam_ids": cam_ids,
        "min_iou": args.min_iou,
        "min_purity": args.min_purity,
        "min_matched": args.min_matched,
        "n_tracklets": len(tracklets),
        "n_identities": report["n_identities"],
        "matching": report,
        "notes": (
            "local_track_id do NvDCF cấp, gt_global_id suy ra bằng ghép IoU với chú thích "
            "WildTrack rồi bỏ phiếu đa số. Tracklet phân mảnh hơn fixture ONNX Runtime nên "
            "F1 tuyệt đối giữa hai bên KHÔNG so trực tiếp được."
        ),
    }

    out = args.out or Path(str(args.fixture).replace(".jsonl", ".gt.json"))
    write_ground_truth(out, tracklets, meta)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    log.info(
        "%s: %d/%d track giữ lại, %d danh tính; khớp %d/%d detection (%.1f%%), "
        "phủ %.1f%% ground-truth, thuần khiết trung vị %.3f",
        out,
        report["n_tracks_kept"],
        report["n_tracks"],
        report["n_identities"],
        report["n_detections_matched"],
        report["n_detections"],
        100 * report["match_rate"],
        100 * report["gt_recall"],
        report["purity_median"],
    )
    log.info("loại bỏ: %s", drops)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
