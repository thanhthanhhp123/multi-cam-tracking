"""Confidence của detector có tách được hộp báo nhầm khỏi người thật không — đo trước khi chỉnh.

    # chẩn đoán
    PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m eval.diagnose_confidence \\
        --fixture data/fixtures/ds_wildtrack_7cam.jsonl \\
        --wildtrack-dir data/wildtrack --json data/confidence.json

    # sinh fixture đã lọc để chạy lại trọn đường online -> TrackEval
    PYTHONPATH=src python -m eval.diagnose_confidence \\
        --fixture data/fixtures/ds_wildtrack_7cam.jsonl \\
        --write-filtered data/fixtures/ds_wildtrack_7cam_conf040.jsonl --min-det-conf 0.40

**Vì sao có file này.** Phiên 21 (`eval/diagnose_junk_ids.py`) tìm ra 73.5% khối lượng
Global ID rác là hộp detector báo nhầm — tức đòn bẩy lớn nhất còn lại nằm TRƯỚC `src/mct`.
Chỉnh nó tận gốc (`pre-cluster-threshold` của nvinfer, `minDetectorConfidence` của NvDCF)
cần thuê GPU chạy lại pipeline. Trước khi tốn tiền đó phải biết một điều rẻ hơn nhiều:
**confidence có mang thông tin phân biệt không?** Nếu hộp báo nhầm và hộp người thật có
cùng phân bố confidence thì mọi ngưỡng đều vô ích, và phiên thuê GPU chỉ để xác nhận điều
đó. Công cụ trả lời hai câu:

1. **Mức detection** — mỗi hộp trong fixture được ghép IoU với chú thích WildTrack (đúng
   phép ghép của `tools/ds_wildtrack_gt.py`) thành TP/FP, rồi quét ngưỡng: nâng ngưỡng
   lên `t` thì giữ lại bao nhiêu TP, bỏ được bao nhiêu FP.
2. **Mức tracklet** — thống kê nào (trung bình / trung vị / lớn nhất) trên confidence của
   một local track tách track báo nhầm (`match_rate < 0.30`, cùng định nghĩa phiên 21)
   khỏi track người thật, đo bằng AUC và bằng khối lượng khung một cổng lọc sẽ bỏ.

**Lọc fixture là phép XẤP XỈ của việc nâng ngưỡng detector, và chỉ theo chiều NÂNG.**
Bỏ hộp khỏi fixture không chạy lại nvtracker: tracker thật, khi bị cắt bớt detection, có
thể bám tiếp bằng shadow tracking hoặc cấp id khác. Nên số đo trên fixture đã lọc là ước
lượng của hướng đi, không phải con số của pipeline sau khi đổi config. Hạ ngưỡng xuống
dưới 0.25 thì không mô phỏng được — hộp đó chưa bao giờ tới fixture.

**Detection `confidence < 0`** (DeepStream gán −0.1 cho target do tracker suy ra, không có
detection trong khung đó — phiên 9) không mang điểm của detector: đếm riêng, không vào
thống kê mức detection, và khi lọc thì GIỮ NGUYÊN — ngưỡng detector không trực tiếp chạm
tới chúng.

Chỉ stdlib + numpy + scipy. Không GPU, không Redis. Chạy trên máy dev.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from common.schema import FrameMessage, read_jsonl, write_jsonl
from tools.ds_wildtrack_gt import gt_index, match_frame, view_idx_for_cam
from tools.wildtrack_to_fixture import RawDetection, parse_raw_detections

DEFAULT_MIN_IOU = 0.5
# Cùng ngưỡng với `diagnose_junk_ids.DEFAULT_FP_MATCH_RATE`: một track khớp người thật
# dưới 30% số khung là track báo nhầm. Phiên 21 đo được hai lớp tách bạch quanh ngưỡng
# này (trung vị 0.00 vs 0.78) nên chọn 0.30 hay 0.50 cho cùng kết luận.
DEFAULT_FP_MATCH_RATE = 0.30
# 0.25 = `pre-cluster-threshold` hiện tại của nvinfer, tức sàn của mọi confidence trong
# fixture. Hàng đầu tiên của bảng quét vì vậy là đường đối chứng (không bỏ gì).
DEFAULT_THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70)
PERCENTILES = (5, 25, 50, 75, 95)

TRACK_STATS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda c: float(c.mean()),
    "median": lambda c: float(np.median(c)),
    "max": lambda c: float(c.max()),
}


@dataclass(frozen=True, slots=True)
class DetRecord:
    """Một detection của fixture, đã biết có trùng người thật hay không."""

    cam_id: str
    local_track_id: int
    confidence: float
    matched: bool

    @property
    def tracker_only(self) -> bool:
        """Target do tracker suy ra (confidence âm) — không có điểm của detector."""
        return self.confidence < 0.0


@dataclass(slots=True)
class TrackConf:
    """Confidence của một local track (cam_id, local_track_id) gom qua cả fixture."""

    cam_id: str
    local_track_id: int
    n_detections: int = 0
    n_matched: int = 0
    confidences: list[float] = field(default_factory=list)
    """Chỉ điểm của detector (>= 0); khung tracker-only vẫn tính vào `n_detections`."""

    @property
    def match_rate(self) -> float:
        return self.n_matched / self.n_detections if self.n_detections else 0.0

    def is_fp(self, fp_match_rate: float) -> bool:
        return self.match_rate < fp_match_rate

    def stat(self, name: str) -> float:
        """Thống kê `name` trên confidence; track không có điểm detector nào -> 0.0."""
        if not self.confidences:
            return 0.0
        return TRACK_STATS[name](np.asarray(self.confidences, dtype=np.float64))


# --------------------------------------------------------------------------- tính toán


def label_detections(
    messages: Iterable[FrameMessage],
    gt_by_frame: dict[tuple[int, int], list[RawDetection]],
    *,
    min_iou: float,
) -> list[DetRecord]:
    """Mỗi detection -> TP/FP bằng ghép IoU một-một với chú thích của đúng khung đó.

    Ghép trên TOÀN BỘ detection của khung (kể cả tracker-only), giống hệt
    `ds_wildtrack_gt.collect_votes`, để `match_rate` ở đây trùng với phiên 21.
    """
    out: list[DetRecord] = []
    for msg in messages:
        if not msg.detections:
            continue
        gt_dets = gt_by_frame.get((view_idx_for_cam(msg.cam_id), int(msg.frame_id)), [])
        pairs = match_frame(
            [tuple(float(v) for v in d.bbox) for d in msg.detections],  # type: ignore[misc]
            [g.bbox for g in gt_dets],
            min_iou=min_iou,
        )
        hit = {det_i for det_i, _, _ in pairs}
        out.extend(
            DetRecord(msg.cam_id, int(d.local_track_id), float(d.confidence), i in hit)
            for i, d in enumerate(msg.detections)
        )
    return out


def auc(positive: Iterable[float], negative: Iterable[float]) -> float:
    """Xác suất một mẫu dương ngẫu nhiên có điểm cao hơn một mẫu âm ngẫu nhiên (hoà = 1/2).

    Đúng bằng diện tích dưới ROC, tính bằng tổng hạng Mann–Whitney nên không phải chọn
    ngưỡng nào. 0.5 = confidence không mang thông tin; 1.0 = tách hoàn toàn.
    """
    pos = np.asarray(list(positive), dtype=np.float64)
    neg = np.asarray(list(negative), dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return math.nan
    ranks = rankdata(np.concatenate([pos, neg]))
    u = float(ranks[: pos.size].sum()) - pos.size * (pos.size + 1) / 2.0
    return u / (pos.size * neg.size)


def _ratio(num: float, den: float) -> float:
    return num / den if den else math.nan


def detection_sweep(
    records: Sequence[DetRecord], thresholds: Sequence[float]
) -> list[dict[str, float]]:
    """Giữ detection có confidence >= t: còn bao nhiêu TP, bỏ được bao nhiêu FP."""
    scored = [r for r in records if not r.tracker_only]
    conf = np.array([r.confidence for r in scored], dtype=np.float64)
    matched = np.array([r.matched for r in scored], dtype=bool)
    n_tp, n_fp = int(matched.sum()), int((~matched).sum())

    rows: list[dict[str, float]] = []
    for t in thresholds:
        keep = conf >= t
        tp = int((keep & matched).sum())
        fp = int((keep & ~matched).sum())
        rows.append(
            {
                "threshold": float(t),
                "kept_tp": tp,
                "kept_fp": fp,
                "precision": _ratio(tp, tp + fp),
                "tp_retained": _ratio(tp, n_tp),
                "fp_removed": 1.0 - _ratio(fp, n_fp),
            }
        )
    return rows


def track_table(records: Iterable[DetRecord]) -> list[TrackConf]:
    tracks: dict[tuple[str, int], TrackConf] = {}
    for r in records:
        key = (r.cam_id, r.local_track_id)
        tc = tracks.get(key)
        if tc is None:
            tc = tracks[key] = TrackConf(r.cam_id, r.local_track_id)
        tc.n_detections += 1
        tc.n_matched += int(r.matched)
        if not r.tracker_only:
            tc.confidences.append(r.confidence)
    return [tracks[k] for k in sorted(tracks)]


def gate_sweep(
    tracks: Sequence[TrackConf],
    stat: str,
    thresholds: Sequence[float],
    *,
    fp_match_rate: float,
) -> list[dict[str, float]]:
    """Cổng mức tracklet: bỏ track có `stat < t`. Đo bằng KHỐI LƯỢNG KHUNG, không bằng số track.

    Cùng lý do với phiên 21: một track báo nhầm 3 khung và một track 187 khung ăn vào
    AssA khác nhau hai bậc độ lớn.
    """
    fp_frames = sum(t.n_detections for t in tracks if t.is_fp(fp_match_rate))
    real_frames = sum(t.n_detections for t in tracks if not t.is_fp(fp_match_rate))
    rows: list[dict[str, float]] = []
    for th in thresholds:
        gone = [t for t in tracks if t.stat(stat) < th]
        gone_fp = [t for t in gone if t.is_fp(fp_match_rate)]
        gone_real = [t for t in gone if not t.is_fp(fp_match_rate)]
        rows.append(
            {
                "threshold": float(th),
                "n_fp_tracks_removed": len(gone_fp),
                "n_real_tracks_removed": len(gone_real),
                "fp_frames_removed": _ratio(sum(t.n_detections for t in gone_fp), fp_frames),
                "real_frames_lost": _ratio(sum(t.n_detections for t in gone_real), real_frames),
            }
        )
    return rows


def filter_messages(
    messages: Iterable[FrameMessage], *, min_det_conf: float
) -> tuple[list[FrameMessage], int]:
    """Bỏ detection có 0 <= confidence < ngưỡng. Trả (message mới, số detection đã bỏ).

    Giữ message rỗng: engine đóng cửa sổ theo `ts_ms` của message tới, bỏ cả message thì
    nhịp cửa sổ đổi và phép so với đường đối chứng lẫn thêm một biến. Không sửa input.
    """
    out: list[FrameMessage] = []
    dropped = 0
    for msg in messages:
        kept = [d for d in msg.detections if d.confidence < 0.0 or d.confidence >= min_det_conf]
        dropped += len(msg.detections) - len(kept)
        out.append(dataclasses.replace(msg, detections=kept))
    return out, dropped


# --------------------------------------------------------------------------- báo cáo


def _quantiles(values: Sequence[float]) -> list[float]:
    if not values:
        return [math.nan] * len(PERCENTILES)
    return [float(v) for v in np.percentile(np.asarray(values, dtype=np.float64), PERCENTILES)]


def _fmt(v: float, spec: str = ".3f") -> str:
    return "—" if v != v else format(v, spec)


def report(
    records: Sequence[DetRecord],
    *,
    thresholds: Sequence[float],
    fp_match_rate: float,
) -> dict[str, Any]:
    scored = [r for r in records if not r.tracker_only]
    tp_conf = [r.confidence for r in scored if r.matched]
    fp_conf = [r.confidence for r in scored if not r.matched]
    det_auc = auc(tp_conf, fp_conf)

    print(f"\n{'=' * 78}")
    print("1. MỨC DETECTION — confidence của hộp trùng người thật vs. hộp báo nhầm")
    print("=" * 78)
    print(f"detection trong fixture        {len(records)}")
    print(f"  tracker-only (conf < 0)      {len(records) - len(scored)}  (không vào thống kê)")
    print(f"  TP (IoU khớp người thật)     {len(tp_conf)}")
    print(f"  FP (không khớp ai)           {len(fp_conf)}")
    print(f"AUC(conf, TP vs FP)            {_fmt(det_auc)}   (0.5 = vô dụng, 1.0 = tách hẳn)")
    print(f"\n{'':22s}" + "".join(f"{'p' + str(p):>8s}" for p in PERCENTILES))
    print(f"{'conf TP':22s}" + "".join(f"{_fmt(v):>8s}" for v in _quantiles(tp_conf)))
    print(f"{'conf FP':22s}" + "".join(f"{_fmt(v):>8s}" for v in _quantiles(fp_conf)))

    det_rows = detection_sweep(records, thresholds)
    print(
        f"\n{'ngưỡng':>8s}{'TP còn':>9s}{'FP còn':>9s}{'precision':>11s}{'%TP giữ':>10s}"
        f"{'%FP bỏ':>9s}"
    )
    for row in det_rows:
        print(
            f"{row['threshold']:>8.2f}{row['kept_tp']:>9d}{row['kept_fp']:>9d}"
            f"{_fmt(row['precision']):>11s}{_fmt(100 * row['tp_retained'], '.1f'):>10s}"
            f"{_fmt(100 * row['fp_removed'], '.1f'):>9s}"
        )

    tracks = track_table(records)
    fp_tracks = [t for t in tracks if t.is_fp(fp_match_rate)]
    real_tracks = [t for t in tracks if not t.is_fp(fp_match_rate)]
    track_auc = {
        name: auc([t.stat(name) for t in real_tracks], [t.stat(name) for t in fp_tracks])
        for name in TRACK_STATS
    }
    gates = {
        name: gate_sweep(tracks, name, thresholds, fp_match_rate=fp_match_rate)
        for name in TRACK_STATS
    }

    print(f"\n{'=' * 78}")
    print(
        f"2. MỨC TRACKLET — cổng lọc track theo confidence (track FP: match_rate < {fp_match_rate})"
    )
    print("=" * 78)
    fp_frames = sum(t.n_detections for t in fp_tracks)
    real_frames = sum(t.n_detections for t in real_tracks)
    print(f"local track FP                 {len(fp_tracks)}  ({fp_frames} khung)")
    print(f"local track người thật         {len(real_tracks)}  ({real_frames} khung)")
    for name in TRACK_STATS:
        print(f"AUC({name:6s})                   {_fmt(track_auc[name])}")
    for name, rows in gates.items():
        print(
            f"\ncổng '{name}' < t: {'ngưỡng':>7s}{'#FP bỏ':>8s}{'#thật bỏ':>10s}"
            f"{'%khung FP bỏ':>14s}{'%khung thật mất':>17s}"
        )
        for row in rows:
            print(
                f"{'':17s}{row['threshold']:>7.2f}{row['n_fp_tracks_removed']:>8d}"
                f"{row['n_real_tracks_removed']:>10d}"
                f"{_fmt(100 * row['fp_frames_removed'], '.1f'):>14s}"
                f"{_fmt(100 * row['real_frames_lost'], '.1f'):>17s}"
            )

    return {
        "detections": {
            "n_total": len(records),
            "n_tracker_only": len(records) - len(scored),
            "n_tp": len(tp_conf),
            "n_fp": len(fp_conf),
            "auc": det_auc,
            "conf_quantiles": {"tp": _quantiles(tp_conf), "fp": _quantiles(fp_conf)},
        },
        "detection_sweep": det_rows,
        "tracks": {
            "n_fp": len(fp_tracks),
            "n_real": len(real_tracks),
            "fp_frames": fp_frames,
            "real_frames": real_frames,
            "fp_match_rate": fp_match_rate,
        },
        "track_auc": track_auc,
        "gate_sweep": gates,
    }


def _json_safe(obj: Any) -> Any:
    """NaN -> null: `json.dumps` mặc định ghi `NaN`, không phải JSON hợp lệ."""
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    return obj


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.split(",") if v.strip())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fixture", type=Path, required=True, help="fixture .jsonl của pipeline")
    p.add_argument(
        "--wildtrack-dir",
        type=Path,
        default=None,
        help="thư mục chứa annotations_positions/; bỏ trống thì chỉ lọc (--write-filtered)",
    )
    p.add_argument("--min-iou", type=float, default=DEFAULT_MIN_IOU)
    p.add_argument("--fp-match-rate", type=float, default=DEFAULT_FP_MATCH_RATE)
    p.add_argument("--frame-stride", type=int, default=1, help="phải khớp lúc đóng video")
    p.add_argument(
        "--thresholds",
        type=_floats,
        default=DEFAULT_THRESHOLDS,
        help="danh sách ngưỡng quét, phân cách bằng dấu phẩy",
    )
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--write-filtered", type=Path, default=None, help="ghi fixture đã lọc ra đây")
    p.add_argument(
        "--min-det-conf", type=float, default=None, help="ngưỡng lọc cho --write-filtered"
    )
    args = p.parse_args(argv)

    if (args.write_filtered is None) != (args.min_det_conf is None):
        p.error("--write-filtered và --min-det-conf phải đi cùng nhau")
    if args.wildtrack_dir is None and args.write_filtered is None:
        p.error("cần --wildtrack-dir (chẩn đoán) hoặc --write-filtered (lọc), hoặc cả hai")

    messages = list(read_jsonl(args.fixture))

    if args.wildtrack_dir is not None:
        view_indices = sorted({view_idx_for_cam(m.cam_id) for m in messages})
        raw, n_ann_frames = parse_raw_detections(
            args.wildtrack_dir / "annotations_positions",
            view_indices=view_indices,
            stride=args.frame_stride,
            max_frames=0,
            min_box_area=0.0,
        )
        records = label_detections(messages, gt_index(raw), min_iou=args.min_iou)
        print(
            f"{args.fixture.name}: {len(messages)} message, {len(records)} detection | "
            f"WildTrack: {len(raw)} hộp chú thích / {n_ann_frames} khung"
        )
        out = report(records, thresholds=args.thresholds, fp_match_rate=args.fp_match_rate)
        out["fixture"] = str(args.fixture)
        out["min_iou"] = args.min_iou
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(_json_safe(out), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(f"\nsố liệu đầy đủ: {args.json}")

    if args.write_filtered is not None:
        filtered, dropped = filter_messages(messages, min_det_conf=args.min_det_conf)
        n = write_jsonl(args.write_filtered, filtered)
        total = sum(len(m.detections) for m in messages)
        print(
            f"\n{args.write_filtered}: {n} message, bỏ {dropped}/{total} detection "
            f"(0 <= conf < {args.min_det_conf})"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
