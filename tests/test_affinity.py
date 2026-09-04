"""Test ma trận chi phí (src/mct/affinity.py).

Mỗi ô bị loại phải bị loại vì ĐÚNG lý do của nó — đó là thứ dùng để truy nguyên khi một
match đúng bị trượt ở tuần tinh chỉnh tham số, nên lý do được test như một phần của hành vi.
"""

from __future__ import annotations

import numpy as np
import pytest

from common.schema import CLASS_PERSON, Detection, FrameMessage, l2_normalize
from mct.affinity import (
    INFEASIBLE,
    AffinityConfig,
    build_cost_matrix,
    costs_for_hungarian,
)
from mct.gallery import Gallery, GalleryConfig
from mct.topology import Topology
from mct.tracklet import Tracklet, TrackletBuilder, TrackletConfig

SEED = 7
DIM = 32

TOPO = Topology.from_mapping(
    {
        "cameras": {"cam01": {}, "cam02": {}},
        "transitions": [
            {"from": "cam01", "to": "cam02", "bidirectional": True, "min_ms": 3000, "max_ms": 15000}
        ],
    }
)

OVERLAP_TOPO = Topology.from_mapping(
    {
        "cameras": {
            "cam01": {"overlaps_with": ["cam02"]},
            "cam02": {"overlaps_with": ["cam01"]},
        }
    }
)


def _tracklet(
    cam_id: str,
    local_track_id: int,
    *,
    start_ms: int,
    embedding: np.ndarray | None = None,
    n_frames: int = 6,
    tracklet_id: int | None = None,
    bbox: tuple[float, float, float, float] = (100.0, 200.0, 80.0, 200.0),
) -> Tracklet:
    builder = TrackletBuilder(TrackletConfig(min_frames=1))
    for i in range(n_frames):
        builder.update(
            FrameMessage(
                cam_id=cam_id,
                frame_id=i,
                ts_ms=start_ms + i * 66,
                frame_pts_ns=i * 66_666_667,
                frame_width=1920,
                frame_height=1080,
                detections=[
                    Detection(
                        local_track_id=local_track_id,
                        bbox=bbox,
                        confidence=0.9,
                        embedding=embedding,
                        class_id=CLASS_PERSON,
                    )
                ],
                embed_dim=0 if embedding is None else embedding.shape[0],
            )
        )
    tracklet = builder.flush()[0]
    if tracklet_id is not None:
        tracklet.tracklet_id = tracklet_id
    return tracklet


class _FakeGround:
    """GroundMapper giả: trả khoảng cách đặt sẵn, hoặc None = cặp chưa hiệu chỉnh."""

    def __init__(self, distance: float | None) -> None:
        self.distance = distance
        self.calls: list[tuple[str, str]] = []

    def distance_m(self, cam_a, point_a, cam_b, point_b):
        self.calls.append((cam_a, cam_b))
        return self.distance


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
# Chi phí ngoại hình
# --------------------------------------------------------------------------- #


def test_chi_phi_bang_mot_tru_cosine(rng):
    a = l2_normalize(rng.standard_normal(DIM))
    b = l2_normalize(0.8 * a + 0.6 * l2_normalize(rng.standard_normal(DIM)))

    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=a, tracklet_id=1))
    later = _tracklet("cam02", 1, start_ms=10_000, embedding=b, tracklet_id=2)

    matrix = build_cost_matrix([later], [track], topology=TOPO)
    expected = 1.0 - float(track.similarity(b, "max"))

    assert matrix.shape == (1, 1)
    assert matrix.costs[0, 0] == pytest.approx(expected)
    assert matrix.reason(0, 0) == ""
    assert matrix.feasible_pairs() == [(0, 0, pytest.approx(expected))]


def test_tracklet_khong_co_embedding_thi_bat_kha_thi(rng):
    vec = l2_normalize(rng.standard_normal(DIM))
    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1))

    no_emb = _tracklet("cam02", 1, start_ms=10_000, tracklet_id=2)
    matrix = build_cost_matrix([no_emb], [track], topology=TOPO)

    assert matrix.costs[0, 0] == INFEASIBLE
    assert "chưa có embedding" in matrix.reason(0, 0)
    assert matrix.feasible_pairs() == []


def test_global_track_chua_co_ngoai_hinh_thi_bat_kha_thi(rng):
    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, tracklet_id=1))  # ReID tắt

    vec = l2_normalize(rng.standard_normal(DIM))
    tracklet = _tracklet("cam02", 1, start_ms=10_000, embedding=vec, tracklet_id=2)
    matrix = build_cost_matrix([tracklet], [track], topology=TOPO)

    assert matrix.costs[0, 0] == INFEASIBLE
    assert "chưa có embedding để so" in matrix.reason(0, 0)


def test_tracklet_dang_thuoc_track_thi_chi_phi_bang_khong(rng):
    """Tracklet đang chạy phải giữ nguyên Global ID của nó, không phải cạnh tranh lại."""
    vec = l2_normalize(rng.standard_normal(DIM))
    gallery = Gallery()
    tracklet = _tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1)
    track = gallery.create(tracklet)

    longer = _tracklet("cam01", 1, start_ms=0, embedding=vec, n_frames=20, tracklet_id=1)
    matrix = build_cost_matrix([longer], [track], topology=TOPO)

    assert matrix.costs[0, 0] == 0.0


def test_global_track_da_dong_bi_loai(rng):
    vec = l2_normalize(rng.standard_normal(DIM))
    gallery = Gallery(GalleryConfig(global_track_ttl_ms=1_000))
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1))
    gallery.prune(now_ms=500_000)

    tracklet = _tracklet("cam02", 1, start_ms=600_000, embedding=vec, tracklet_id=2)
    matrix = build_cost_matrix([tracklet], [track], topology=TOPO)

    assert matrix.costs[0, 0] == INFEASIBLE
    assert "đã đóng" in matrix.reason(0, 0)


# --------------------------------------------------------------------------- #
# Ràng buộc
# --------------------------------------------------------------------------- #


def test_di_qua_nhanh_bi_loai_du_ngoai_hinh_giong_het(rng):
    """Ràng buộc thời gian phải chặn được thứ mà đặc trưng Re-ID không tự chặn nổi."""
    vec = l2_normalize(rng.standard_normal(DIM))
    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1))

    too_soon = _tracklet("cam02", 1, start_ms=1_000, embedding=vec, tracklet_id=2)
    matrix = build_cost_matrix([too_soon], [track], topology=TOPO)

    assert matrix.costs[0, 0] == INFEASIBLE
    assert "quá nhanh" in matrix.reason(0, 0)
    assert "LOẠI" in matrix.explain(0, 0)


def test_di_qua_lau_bi_loai(rng):
    vec = l2_normalize(rng.standard_normal(DIM))
    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1))

    too_late = _tracklet("cam02", 1, start_ms=90_000, embedding=vec, tracklet_id=2)
    matrix = build_cost_matrix([too_late], [track], topology=TOPO)

    assert matrix.costs[0, 0] == INFEASIBLE
    assert "quá lâu" in matrix.reason(0, 0)


def test_khong_truyen_topology_thi_bo_qua_rang_buoc_thoi_gian(rng):
    """Dùng khi cố ý đo phần đóng góp riêng của đặc trưng Re-ID (chương 6)."""
    vec = l2_normalize(rng.standard_normal(DIM))
    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1))

    too_soon = _tracklet("cam02", 1, start_ms=1_000, embedding=vec, tracklet_id=2)
    matrix = build_cost_matrix([too_soon], [track], topology=None)

    assert matrix.costs[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_rang_buoc_loai_tru_cung_camera(rng):
    vec = l2_normalize(rng.standard_normal(DIM))
    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1))

    other = _tracklet("cam01", 2, start_ms=100, embedding=vec, tracklet_id=2)
    matrix = build_cost_matrix(
        [other], [track], topology=TOPO, config=AffinityConfig(exclusion_window_ms=2_000)
    )

    assert matrix.costs[0, 0] == INFEASIBLE
    assert "ràng buộc loại trừ" in matrix.reason(0, 0)


def test_het_cua_so_loai_tru_thi_khop_lai_duoc(rng):
    """Người rời cam01 rồi quay lại chính cam01 vẫn phải nhận lại Global ID cũ."""
    vec = l2_normalize(rng.standard_normal(DIM))
    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1))

    comeback = _tracklet("cam01", 9, start_ms=60_000, embedding=vec, tracklet_id=2)
    matrix = build_cost_matrix(
        [comeback], [track], topology=TOPO, config=AffinityConfig(exclusion_window_ms=2_000)
    )

    assert np.isfinite(matrix.costs[0, 0])


# --------------------------------------------------------------------------- #
# Thành phần hình học (homography)
# --------------------------------------------------------------------------- #


def test_cap_chong_lan_cong_them_khoang_cach_mat_dat(rng):
    vec = l2_normalize(rng.standard_normal(DIM))
    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1))
    tracklet = _tracklet("cam02", 1, start_ms=200, embedding=vec, tracklet_id=2)

    config = AffinityConfig(homography_weight=0.4, max_ground_dist_m=3.0)
    mapper = _FakeGround(1.5)
    matrix = build_cost_matrix(
        [tracklet], [track], topology=OVERLAP_TOPO, config=config, ground_mapper=mapper
    )

    base = 1.0 - float(track.similarity(vec, "max"))
    assert matrix.costs[0, 0] == pytest.approx(base + 0.4 * 1.5)
    assert mapper.calls == [("cam01", "cam02")]


def test_vuot_khoang_cach_mat_dat_thi_loai_thang(rng):
    vec = l2_normalize(rng.standard_normal(DIM))
    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1))
    tracklet = _tracklet("cam02", 1, start_ms=200, embedding=vec, tracklet_id=2)

    matrix = build_cost_matrix(
        [tracklet],
        [track],
        topology=OVERLAP_TOPO,
        config=AffinityConfig(max_ground_dist_m=3.0),
        ground_mapper=_FakeGround(9.0),
    )

    assert matrix.costs[0, 0] == INFEASIBLE
    assert "mặt phẳng tham chiếu" in matrix.reason(0, 0)


def test_cap_khong_chong_lan_khong_dung_toi_homography(rng):
    """Hai đầu hành lang cách nhau chục mét vẫn là cùng một người — khoảng cách vô nghĩa."""
    vec = l2_normalize(rng.standard_normal(DIM))
    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1))
    tracklet = _tracklet("cam02", 1, start_ms=10_000, embedding=vec, tracklet_id=2)

    mapper = _FakeGround(50.0)
    matrix = build_cost_matrix([tracklet], [track], topology=TOPO, ground_mapper=mapper)

    assert np.isfinite(matrix.costs[0, 0])
    assert mapper.calls == [], "cặp không chồng lấn thì không được gọi tới homography"


def test_cap_chua_hieu_chinh_thi_bo_qua_thanh_phan_hinh_hoc(rng):
    vec = l2_normalize(rng.standard_normal(DIM))
    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1))
    tracklet = _tracklet("cam02", 1, start_ms=200, embedding=vec, tracklet_id=2)

    matrix = build_cost_matrix(
        [tracklet],
        [track],
        topology=OVERLAP_TOPO,
        ground_mapper=_FakeGround(None),  # chưa hiệu chỉnh cặp này
    )

    assert matrix.costs[0, 0] == pytest.approx(1.0 - float(track.similarity(vec, "max")))


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #


def test_costs_for_hungarian_thay_inf_bang_gia_tri_lon_hon_nguong():
    raw = np.array([[0.1, INFEASIBLE], [INFEASIBLE, INFEASIBLE]])
    padded = costs_for_hungarian(raw, max_cost=0.3)

    assert np.isfinite(padded).all()
    assert padded[0, 0] == 0.1
    assert (padded[0, 1] > 0.3) and (padded[1, 1] > 0.3)


def test_ma_tran_rong_khong_no():
    matrix = build_cost_matrix([], [], topology=TOPO)
    assert matrix.shape == (0, 0)
    assert matrix.feasible_pairs() == []


def test_config_doc_duoc_tu_mct_yaml(repo_root):
    """AffinityConfig phải khớp đúng các khoá đang có trong configs/mct.yaml."""
    from common.config import load_yaml

    data = load_yaml(repo_root / "configs" / "mct.yaml")
    config = AffinityConfig.from_mapping(data)

    assert config.max_cost == data["association"]["max_cost"]
    assert config.homography_weight == data["association"]["homography_weight"]
    assert config.max_ground_dist_m == data["association"]["max_ground_dist_m"]
    assert config.exclusion_window_ms == data["tracklet"]["idle_timeout_ms"]
    assert config.topk_query == data["gallery"]["topk_query"]
