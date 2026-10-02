"""Test cho `eval/check_detector_cpu.py` và `tools/export_yolo26.py` — phần không cần ONNX.

Điểm cần ghim: bước giải mã phải làm ĐÚNG như `NvDsInferParseYolo` + nvinfer
(`maintain-aspect-ratio=1`, `symmetric-padding=1`), nếu không thì số so YOLO11/YOLO26 trên
CPU không nói gì về pipeline thật.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from eval.check_detector_cpu import Letterbox, cluster, decode, pick_frames, wildtrack_gt_xywh

from tools import export_yolo26


def test_letterbox_1080p_dem_deu_tren_duoi():
    box = Letterbox.fit(1920, 1080)
    assert box.scale == pytest.approx(1 / 3)
    assert box.pad_x == pytest.approx(0.0)
    assert box.pad_y == pytest.approx(140.0)


def test_decode_tra_hop_ve_toa_do_khung_goc():
    box = Letterbox.fit(1920, 1080)
    # Một người ở (300, 600)-(360, 900) trong khung gốc -> toạ độ mạng: /3, cộng đệm 140.
    person = [100.0, 340.0, 120.0, 440.0, 0.9, 0.0]
    xyxy, scores = decode(np.array([person]), box, threshold=0.25)
    assert xyxy[0].tolist() == pytest.approx([300.0, 600.0, 360.0, 900.0])
    assert scores.tolist() == pytest.approx([0.9])


def test_decode_chi_giu_nguoi_tren_nguong_va_bo_hop_suy_bien():
    box = Letterbox.fit(1920, 1080)
    rows = np.array(
        [
            [10, 200, 50, 300, 0.95, 2.0],  # ô tô, tự tin -> bỏ (ngưỡng lớp khác là 1.0)
            [10, 200, 50, 300, 0.20, 0.0],  # người dưới ngưỡng -> bỏ
            [10, 200, 10.5, 300, 0.90, 0.0],  # rộng < 1 px trong toạ độ mạng -> bỏ
            [10, 200, 50, 300, 0.30, 0.0],  # giữ
        ],
        dtype=np.float32,
    )
    xyxy, scores = decode(rows, box, threshold=0.25)
    assert len(xyxy) == 1
    assert scores[0] == pytest.approx(0.30)


def test_decode_kep_hop_trong_khung_mang_truoc_khi_doi_toa_do():
    box = Letterbox.fit(1920, 1080)
    xyxy, _ = decode(np.array([[-20, 150, 30, 250, 0.9, 0]], dtype=np.float32), box, threshold=0.25)
    assert xyxy[0, 0] == pytest.approx(0.0)


def test_nms_gop_hop_trung_con_none_thi_giu_het():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.6, 0.9, 0.5], dtype=np.float32)
    assert cluster(boxes, scores, "nms").tolist() == [1, 2]
    assert cluster(boxes, scores, "none").tolist() == [1, 0, 2]
    assert cluster(boxes, scores, "none", top_k=2).tolist() == [1, 0]
    with pytest.raises(ValueError):
        cluster(boxes, scores, "dbscan")


def test_chon_khung_rai_deu_ca_chuoi(tmp_path):
    for i in range(400):
        (tmp_path / f"{i * 5:08d}.json").write_text("[]", encoding="utf-8")
    frames = pick_frames(tmp_path, 10)
    assert len(frames) == 10
    assert frames[0] == "00000000" and frames[-1] == "00001995"


def test_gt_wildtrack_bo_view_khong_thay_nguoi(tmp_path):
    people = [
        {"views": [{"xmin": 10, "ymin": 20, "xmax": 40, "ymax": 120}]},
        {"views": [{"xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1}]},
    ]
    (tmp_path / "00000000.json").write_text(json.dumps(people), encoding="utf-8")
    assert wildtrack_gt_xywh(tmp_path, "00000000", 0).tolist() == [[10, 20, 30, 100]]


def test_export_tu_choi_file_sai_sha256(tmp_path):
    """Không chạy script hay nạp weight khác thứ đã kiểm (file đã có thì không tải lại)."""
    path = tmp_path / "export_yolo26.py"
    path.write_text("print('khác')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sha256"):
        export_yolo26.fetch(
            export_yolo26.EXPORT_SCRIPT_URL, path, export_yolo26.EXPORT_SCRIPT_SHA256
        )


def test_export_ghim_commit_trong_url():
    assert export_yolo26.DSYOLO_COMMIT in export_yolo26.EXPORT_SCRIPT_URL
