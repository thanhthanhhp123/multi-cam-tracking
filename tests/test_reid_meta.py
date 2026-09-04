"""Test `ds_pipeline/reid_meta.py` — phần logic thuần, không cần pyds/GPU.

Module đó import `pyds` **bên trong hàm** chứ không ở đầu file, chính là để chỗ này test
được trên máy dev. Thứ đáng test không phải việc gọi API DeepStream (chỉ máy GPU mới nói
được đúng/sai), mà là hai quyết định dễ sai âm thầm:

  1. vector phải được COPY ra khỏi bộ nhớ của DeepStream trước khi probe trả về;
  2. vector hỏng (sai chiều, NaN) phải bị loại chứ không đi tiếp vào ma trận chi phí,
     nơi nó sẽ biến thành một Global ID sai mà không ai truy ra nguồn.
"""

from __future__ import annotations

import numpy as np
import pytest

from ds_pipeline import reid_meta


@pytest.fixture(autouse=True)
def _reset_warnings() -> None:
    """`_warn_once` giữ trạng thái toàn cục — dọn giữa các test."""
    reid_meta._warned.clear()


class TestSanitizeEmbedding:
    def test_tra_ve_ban_sao_khong_phai_view(self) -> None:
        """Nguồn bị ghi đè (DeepStream giải phóng meta) thì kết quả phải không đổi."""
        nguon = np.arange(512, dtype=np.float32)
        vec = reid_meta.sanitize_embedding(nguon)

        assert vec is not None
        nguon[:] = -1.0  # mô phỏng vùng nhớ bị tái sử dụng sau khi probe trả về

        assert vec[0] == 0.0
        assert vec[-1] == 511.0
        assert not np.shares_memory(vec, nguon)

    def test_ep_ve_float32_mot_chieu(self) -> None:
        vec = reid_meta.sanitize_embedding(np.ones((1, 512), dtype=np.float64))
        assert vec is not None
        assert vec.dtype == np.float32
        assert vec.shape == (512,)

    def test_loai_vector_sai_chieu(self) -> None:
        sai = np.ones(256, dtype=np.float32)
        assert reid_meta.sanitize_embedding(sai, expected_dim=512) is None

    def test_nhan_vector_dung_chieu(self) -> None:
        vec = reid_meta.sanitize_embedding(np.ones(512, dtype=np.float32), expected_dim=512)
        assert vec is not None and vec.size == 512

    @pytest.mark.parametrize("gia_tri", [np.nan, np.inf, -np.inf])
    def test_loai_vector_co_nan_hoac_inf(self, gia_tri: float) -> None:
        raw = np.ones(512, dtype=np.float32)
        raw[7] = gia_tri
        assert reid_meta.sanitize_embedding(raw) is None

    def test_vector_rong_hoac_none(self) -> None:
        assert reid_meta.sanitize_embedding(None) is None
        assert reid_meta.sanitize_embedding(np.empty(0, dtype=np.float32)) is None

    def test_khong_tu_l2_normalize(self) -> None:
        """Chuẩn hoá là việc của `common.schema.l2_normalize`, đúng một chỗ (CLAUDE.md §5)."""
        vec = reid_meta.sanitize_embedding(np.full(512, 3.0, dtype=np.float32))
        assert vec is not None
        assert np.isclose(float(np.linalg.norm(vec)), 3.0 * np.sqrt(512))


class _StubMetaTypes:
    NVDS_TRACKER_OBJ_REID_META = 42


class _StubPyds:
    NvDsMetaType = _StubMetaTypes


class _StubPydsCu:
    """Bản DeepStream không có hằng meta type của tracker (chỉ còn đường B)."""

    class NvDsMetaType:
        NVDSINFER_TENSOR_OUTPUT_META = 12


class TestTrackerMetaType:
    def test_tim_thay_hang_khi_co(self) -> None:
        assert reid_meta._tracker_meta_type(_StubPyds()) == 42

    def test_tra_none_khi_ban_pyds_khong_co_hang(self) -> None:
        """Phải trả None để bên gọi rơi xuống đường (B), KHÔNG ném AttributeError."""
        assert reid_meta._tracker_meta_type(_StubPydsCu()) is None

    def test_tra_none_khi_khong_co_ca_NvDsMetaType(self) -> None:
        assert reid_meta._tracker_meta_type(object()) is None


class TestWarnOnce:
    def test_chi_canh_bao_mot_lan_cho_moi_khoa(self, caplog: pytest.LogCaptureFixture) -> None:
        """Probe chạy trên mọi đối tượng mọi frame — log không chặn sẽ tự làm tụt FPS."""
        raw = np.ones(256, dtype=np.float32)
        with caplog.at_level("WARNING"):
            for _ in range(100):
                reid_meta.sanitize_embedding(raw, expected_dim=512)

        assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1
