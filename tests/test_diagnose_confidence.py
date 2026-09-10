"""Test `eval/diagnose_confidence.py` — confidence có tách hộp báo nhầm khỏi người thật không.

Con số công cụ này in ra quyết định có đáng thuê GPU để quét ngưỡng detector hay không, và
nó cũng sinh fixture đã lọc để chấm lại. Sai ở phép gắn nhãn TP/FP, ở AUC, hay ở chỗ lọc
nhầm target tracker-only là sai cả kết luận. Mọi kịch bản dưới đây có đáp án đếm tay.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from eval.diagnose_confidence import (
    DetRecord,
    auc,
    detection_sweep,
    filter_messages,
    gate_sweep,
    label_detections,
    main,
    track_table,
)

from common.schema import Detection, FrameMessage
from tools.wildtrack_to_fixture import RawDetection


def frame(cam: str, frame_id: int, dets: list[Detection]) -> FrameMessage:
    return FrameMessage(
        cam_id=cam,
        frame_id=frame_id,
        ts_ms=1000 + frame_id,
        frame_pts_ns=0,
        frame_width=1920,
        frame_height=1080,
        detections=dets,
    )


def det(local: int, bbox: tuple[float, float, float, float], conf: float) -> Detection:
    return Detection(local_track_id=local, bbox=bbox, confidence=conf)


def rec(local: int, conf: float, matched: bool) -> DetRecord:
    return DetRecord("cam01", local, conf, matched)


# --------------------------------------------------------------------------- AUC


def test_auc_tach_hoan_toan_la_1_nguoc_lai_la_0() -> None:
    assert auc([0.9, 0.8], [0.3, 0.2]) == 1.0
    assert auc([0.3, 0.2], [0.9, 0.8]) == 0.0


def test_auc_hoa_tinh_mot_nua() -> None:
    assert auc([0.5, 0.5], [0.5]) == 0.5
    # 1 dương 0.6 vs âm {0.4, 0.6, 0.8}: thắng 1, hoà 1, thua 1 -> (1 + 0.5) / 3
    assert auc([0.6], [0.4, 0.6, 0.8]) == pytest.approx(0.5)


def test_auc_mot_phia_rong_la_nan() -> None:
    assert math.isnan(auc([], [0.1]))
    assert math.isnan(auc([0.1], []))


# --------------------------------------------------------------------------- gắn nhãn


def test_gan_nhan_theo_iou_voi_chu_thich_cua_dung_khung() -> None:
    gt = {
        (0, 0): [RawDetection(0, 0, 0, person_id=7, bbox=(100, 100, 50, 150), world_xy=(0, 0))],
    }
    msgs = [
        frame(
            "cam01",
            0,
            [det(1, (102, 101, 50, 150), 0.9), det(2, (900, 900, 40, 80), 0.3)],
        ),
        # khung 1 không có chú thích -> mọi hộp là FP
        frame("cam01", 1, [det(1, (102, 101, 50, 150), 0.8)]),
    ]
    recs = label_detections(msgs, gt, min_iou=0.5)
    assert [(r.local_track_id, r.matched) for r in recs] == [(1, True), (2, False), (1, False)]


# --------------------------------------------------------------------------- quét ngưỡng


def test_quet_nguong_detection_dem_tay() -> None:
    recs = [
        rec(1, 0.90, True),
        rec(1, 0.40, True),
        rec(2, 0.30, False),
        rec(2, 0.50, False),
        rec(3, -0.1, False),  # tracker-only: không vào thống kê
    ]
    rows = {r["threshold"]: r for r in detection_sweep(recs, [0.25, 0.45])}
    base, cut = rows[0.25], rows[0.45]
    assert (base["kept_tp"], base["kept_fp"]) == (2, 2)
    assert base["fp_removed"] == 0.0 and base["tp_retained"] == 1.0
    # >= 0.45 giữ 0.90 (TP) và 0.50 (FP)
    assert (cut["kept_tp"], cut["kept_fp"]) == (1, 1)
    assert cut["precision"] == 0.5
    assert cut["tp_retained"] == 0.5 and cut["fp_removed"] == 0.5


def test_bang_track_va_cong_loc_tinh_theo_khung() -> None:
    recs = (
        [rec(1, 0.8, True)] * 20  # người thật, 20 khung
        + [rec(2, 0.3, False)] * 10  # báo nhầm, 10 khung
        + [rec(3, -0.1, False)] * 4  # báo nhầm, chỉ có khung tracker-only
        + [rec(4, 0.9, False)] * 2  # báo nhầm nhưng confidence cao
    )
    tracks = {t.local_track_id: t for t in track_table(recs)}
    assert tracks[1].match_rate == 1.0 and tracks[2].match_rate == 0.0
    assert tracks[3].n_detections == 4 and tracks[3].confidences == []
    assert tracks[3].stat("mean") == 0.0  # không có điểm detector nào

    rows = gate_sweep(list(tracks.values()), "mean", [0.5], fp_match_rate=0.3)
    row = rows[0]
    # bỏ track 2 (0.3) và 3 (0.0); track 4 (0.9) lọt cổng
    assert row["n_fp_tracks_removed"] == 2 and row["n_real_tracks_removed"] == 0
    assert row["fp_frames_removed"] == pytest.approx(14 / 16)
    assert row["real_frames_lost"] == 0.0


def test_thong_ke_track_mean_median_max() -> None:
    tracks = track_table([rec(1, c, True) for c in (0.3, 0.4, 0.9)])
    t = tracks[0]
    assert t.stat("mean") == pytest.approx(np.mean([0.3, 0.4, 0.9]))
    assert t.stat("median") == pytest.approx(0.4)
    assert t.stat("max") == pytest.approx(0.9)


# --------------------------------------------------------------------------- lọc fixture


def test_loc_bo_duoi_nguong_giu_tracker_only_va_khong_sua_input() -> None:
    msgs = [
        frame("cam01", 0, [det(1, (0, 0, 10, 20), 0.30), det(2, (50, 0, 10, 20), 0.60)]),
        frame("cam01", 1, [det(3, (0, 0, 10, 20), -0.1)]),
        frame("cam01", 2, [det(1, (0, 0, 10, 20), 0.20)]),
    ]
    out, dropped = filter_messages(msgs, min_det_conf=0.40)
    assert dropped == 2
    assert [[d.local_track_id for d in m.detections] for m in out] == [[2], [3], []]
    # message rỗng vẫn giữ để nhịp cửa sổ của engine không đổi
    assert len(out) == len(msgs)
    assert len(msgs[0].detections) == 2  # input nguyên vẹn


# --------------------------------------------------------------------------- đầu cuối


def test_main_dau_cuoi(tmp_path) -> None:
    ann_dir = tmp_path / "wt" / "annotations_positions"
    ann_dir.mkdir(parents=True)
    (ann_dir / "00000000.json").write_text(
        json.dumps(
            [
                {
                    "personID": 1,
                    "positionID": 200000,
                    "views": [{"viewNum": 0, "xmin": 100, "ymin": 100, "xmax": 150, "ymax": 250}],
                }
            ]
        ),
        encoding="utf-8",
    )

    row = {
        "schema_version": 1,
        "cam_id": "cam01",
        "frame_id": 0,
        "ts_ms": 1000,
        "frame_pts_ns": 0,
        "frame_width": 1920,
        "frame_height": 1080,
        "embed_dim": 0,
        "detections": [
            {"local_track_id": 1, "bbox": [100, 100, 50, 150], "confidence": 0.9},
            {"local_track_id": 2, "bbox": [900, 900, 40, 80], "confidence": 0.3},
            {"local_track_id": 3, "bbox": [600, 600, 40, 80], "confidence": -0.1},
        ],
    }
    fixture = tmp_path / "fx.jsonl"
    fixture.write_text("\n".join(json.dumps(row) for _ in range(5)) + "\n", encoding="utf-8")
    filtered = tmp_path / "fx_conf050.jsonl"

    rc = main(
        [
            "--fixture",
            str(fixture),
            "--wildtrack-dir",
            str(tmp_path / "wt"),
            "--json",
            str(tmp_path / "out.json"),
            "--write-filtered",
            str(filtered),
            "--min-det-conf",
            "0.5",
        ]
    )
    assert rc == 0
    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    d = out["detections"]
    assert (d["n_total"], d["n_tracker_only"], d["n_tp"], d["n_fp"]) == (15, 5, 5, 5)
    assert d["auc"] == 1.0
    assert out["tracks"]["n_real"] == 1 and out["tracks"]["n_fp"] == 2
    assert out["track_auc"]["mean"] == 1.0

    lines = filtered.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    kept = [dd["local_track_id"] for dd in json.loads(lines[0])["detections"]]
    assert kept == [1, 3]  # hộp 0.3 bị lọc, tracker-only giữ nguyên


def test_main_loc_can_ca_hai_co() -> None:
    with pytest.raises(SystemExit):
        main(["--fixture", "x.jsonl", "--write-filtered", "y.jsonl"])
