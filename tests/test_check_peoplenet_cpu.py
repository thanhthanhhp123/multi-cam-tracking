"""Test phần numpy thuần của `tools/check_peoplenet_cpu.py`.

Đây là phần mà mọi kết luận của công cụ dựa vào: op thay cho plugin TensorRT, và bộ giải
mã chép lại parser của NVIDIA. Không cần onnxruntime.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.check_peoplenet_cpu import (
    NET_H,
    NET_W,
    decode_ddetr,
    match_count,
    msda_reference,
)


def _one_level(h: int, w: int, m: int = 1, d: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(1, h * w, m, d)).astype(np.float32)


def test_msda_diem_dung_tam_pixel_tra_dung_gia_tri_pixel() -> None:
    """Toạ độ chuẩn hoá (x+0.5)/W là tâm pixel x (align_corners=False) -> không nội suy."""
    h, w = 3, 4
    value = _one_level(h, w)
    loc = np.zeros((1, 1, 1, 1, 1, 2), dtype=np.float32)
    loc[..., 0] = (2 + 0.5) / w
    loc[..., 1] = (1 + 0.5) / h
    out = msda_reference(
        value, np.array([[h, w]]), np.array([0]), loc, np.ones((1, 1, 1, 1, 1), np.float32)
    )
    np.testing.assert_allclose(out[0, 0, 0], value[0, 1 * w + 2, 0], rtol=1e-6)


def test_msda_noi_suy_song_tuyen_va_dem_0_ngoai_bien() -> None:
    h, w = 2, 2
    value = _one_level(h, w, d=1)
    v = value[0, :, 0, 0].reshape(h, w)
    loc = np.zeros((1, 2, 1, 1, 1, 2), dtype=np.float32)
    loc[0, 0, 0, 0, 0] = (0.5, 0.5)  # giữa 4 tâm pixel -> trung bình
    loc[0, 1, 0, 0, 0] = (0.0, 0.25)  # mép trái, giữa hàng 0 -> nửa pixel (0,0), nửa là 0
    out = msda_reference(
        value, np.array([[h, w]]), np.array([0]), loc, np.ones((1, 2, 1, 1, 1), np.float32)
    )
    assert out[0, 0, 0, 0] == pytest.approx(v.mean(), rel=1e-6)
    assert out[0, 1, 0, 0] == pytest.approx(0.5 * v[0, 0], rel=1e-6)


def test_msda_cong_theo_trong_so_qua_cac_tang_va_nhan_float64() -> None:
    """Hai tầng, mỗi tầng một điểm tâm pixel: kết quả = tổng có trọng số. Bản v2 (DINO)
    đưa sampling_locations kiểu float64 vào, nên dùng float64 ở đây."""
    shapes = np.array([[2, 2], [1, 1]])
    value = np.concatenate([_one_level(2, 2, seed=1), _one_level(1, 1, seed=2)], axis=1)
    loc = np.zeros((1, 1, 1, 2, 1, 2), dtype=np.float64)
    loc[0, 0, 0, 0, 0] = (1.5 / 2, 0.5 / 2)  # tầng 0, pixel (hàng 0, cột 1)
    loc[0, 0, 0, 1, 0] = (0.5, 0.5)  # tầng 1, pixel duy nhất
    weights = np.array([0.25, 0.75], np.float32).reshape(1, 1, 1, 2, 1)
    out = msda_reference(value, shapes, np.array([0, 4]), loc, weights)
    expected = 0.25 * value[0, 1, 0] + 0.75 * value[0, 4, 0]
    np.testing.assert_allclose(out[0, 0, 0], expected, rtol=1e-6)


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def test_decode_bo_lop_nen_va_loc_nguong_sau_sigmoid() -> None:
    logits = np.full((3, 4), -10.0, dtype=np.float32)
    logits[0, 1] = _logit(0.9)  # người, đạt ngưỡng
    logits[1, 0] = _logit(0.99)  # nền thắng argmax -> bỏ dù rất chắc
    logits[2, 1] = _logit(0.2)  # người nhưng dưới ngưỡng
    boxes = np.tile(np.array([0.5, 0.5, 0.1, 0.2], np.float32), (3, 1))
    dets = decode_ddetr(logits, boxes, threshold=0.3, scale_x=1.0, scale_y=1.0)
    assert [(d.class_id, round(d.confidence, 3)) for d in dets] == [(1, 0.9)]


def test_decode_doi_cxcywh_ra_goc_trai_tren_roi_scale_ve_anh_goc() -> None:
    logits = np.array([[-10.0, 5.0, -10.0, -10.0]], dtype=np.float32)
    boxes = np.array([[0.5, 0.5, 0.25, 0.5]], dtype=np.float32)
    (d,) = decode_ddetr(logits, boxes, threshold=0.3, scale_x=2.0, scale_y=1080 / NET_H)
    assert d.x1 == pytest.approx((0.5 - 0.125) * NET_W * 2.0)
    assert d.x2 == pytest.approx((0.5 + 0.125) * NET_W * 2.0)
    assert d.y1 == pytest.approx(0.25 * 1080, rel=1e-5)
    assert d.y2 == pytest.approx(0.75 * 1080, rel=1e-5)


def test_match_count_mot_mot_theo_iou() -> None:
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=np.float32)
    dets = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [100, 100, 110, 110]], dtype=np.float32)
    tp, hit = match_count(dets, gt)
    assert tp == 1  # hai hộp trùng GT đầu, chỉ một được tính
    assert hit.tolist() == [True, False]
