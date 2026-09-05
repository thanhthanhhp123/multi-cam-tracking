"""Test phép tách sai số điểm chân (eval/diagnose_foot_error.py).

Chỉ test `decompose()` — phần đọc fixture và chiếu homography đã có test riêng ở
`test_wildtrack_to_fixture.py` / `test_homography.py`. Thứ cần khoá ở đây là PHÉP TOÁN:
kết luận "hướng làm mượt không có gì để lấy" của phiên 16 dựa vào tỉ lệ độ chệch/nhiễu,
nên nếu phép tách sai thì kết luận sai theo.
"""

from __future__ import annotations

import numpy as np
import pytest
from eval.diagnose_foot_error import decompose


def _pair(offsets: list[tuple[float, float]], *, cam: str = "cam01", track: int = 1):
    """Hai bảng điểm chân lệch nhau đúng `offsets` (bên tham chiếu đặt ở gốc)."""
    detector = {(cam, track, i * 500): offset for i, offset in enumerate(offsets)}
    reference = {(cam, track, i * 500): (0.0, 0.0) for i in range(len(offsets))}
    return detector, reference


def test_lech_hang_so_thi_toan_bo_la_do_chech():
    """Hộp detector lệch y hệt ở mọi khung: không có gì cho bộ lọc khử."""
    detector, reference = _pair([(0.3, 0.4)] * 6)

    report = decompose(detector, reference, min_frames=3)

    assert report["bias_share_of_energy"] == pytest.approx(1.0)
    assert report["rms_bias_m"] == pytest.approx(0.5)  # hypot(0.3, 0.4)
    assert report["rms_noise_m"] == pytest.approx(0.0)


def test_lech_trung_binh_khong_thi_toan_bo_la_nhieu():
    """Nhiễu quanh 0: độ chệch bằng 0, và đây là phần bộ lọc thời gian khử được."""
    detector, reference = _pair([(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)])

    report = decompose(detector, reference, min_frames=3)

    assert report["rms_bias_m"] == pytest.approx(0.0)
    assert report["rms_noise_m"] == pytest.approx(1.0)
    assert report["bias_share_of_energy"] == pytest.approx(0.0)


def test_phan_ra_nang_luong_dung_bang_tong():
    """E|Δ|² = |độ chệch|² + E|nhiễu|² — đẳng thức này là toàn bộ lập luận của phép đo."""
    detector, reference = _pair([(0.5, 0.2), (1.5, -0.4), (0.1, 0.9), (-0.3, 0.1), (0.7, 0.5)])

    report = decompose(detector, reference, min_frames=3)

    total = report["rms_total_m"] ** 2
    assert total == pytest.approx(report["rms_bias_m"] ** 2 + report["rms_noise_m"] ** 2)
    deltas = np.array([(0.5, 0.2), (1.5, -0.4), (0.1, 0.9), (-0.3, 0.1), (0.7, 0.5)])
    assert total == pytest.approx(float((deltas**2).sum(axis=1).mean()))


def test_chi_so_khop_o_ca_hai_ben_moi_duoc_tinh():
    """Khoá thiếu một bên bị bỏ — hai fixture phải cùng tập detection mới so được."""
    detector, reference = _pair([(1.0, 0.0)] * 5)
    detector[("cam01", 1, 99_999)] = (100.0, 100.0)  # chỉ có ở bên detector
    reference[("cam02", 7, 0)] = (0.0, 0.0)  # chỉ có ở bên tham chiếu

    report = decompose(detector, reference, min_frames=3)

    assert report["n_detections"] == 5
    assert report["rms_total_m"] == pytest.approx(1.0)


def test_tracklet_qua_ngan_bi_bo():
    """Tracklet 2 khung không tách được độ chệch khỏi nhiễu một cách có nghĩa."""
    detector, reference = _pair([(1.0, 0.0), (1.0, 0.0)])

    report = decompose(detector, reference, min_frames=3)

    assert report["n_tracklets"] == 0
    assert report["n_detections"] == 0
    assert report["rms_total_m"] == 0.0
