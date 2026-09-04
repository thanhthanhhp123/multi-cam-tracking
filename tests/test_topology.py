"""Test đồ thị camera + ràng buộc thời gian di chuyển (src/mct/topology.py).

Gồm cả một test chạy trên `configs/cameras/topology.yaml` thật: file cấu hình đó là thứ
duy nhất mô tả bố trí camera, sai ở đó thì mọi kết quả liên kết đều sai mà không có
triệu chứng rõ ràng.
"""

from __future__ import annotations

import pytest

from mct.topology import Topology, TopologyError

BASE = {
    "cameras": {
        "cam01": {"name": "đầu hành lang", "resolution": [1920, 1080], "fps": 15},
        "cam02": {"name": "cuối hành lang", "resolution": [1920, 1080], "fps": 15},
        "cam03": {"name": "sảnh", "resolution": [1280, 720], "fps": 15},
    },
    "transitions": [
        {
            "from": "cam01",
            "to": "cam02",
            "bidirectional": True,
            "min_ms": 3000,
            "max_ms": 15000,
            "distance_m": 12.0,
        },
    ],
}


def _topo(**overrides) -> Topology:
    data = {**BASE, **overrides}
    return Topology.from_mapping(data)


# --------------------------------------------------------------------------- #
# Nạp cấu hình
# --------------------------------------------------------------------------- #


def test_nap_camera_va_transition():
    topo = _topo()

    assert topo.camera_ids == ["cam01", "cam02", "cam03"]
    assert len(topo) == 3
    assert "cam02" in topo
    assert topo.cameras["cam03"].resolution == (1280, 720)

    transition = topo.transition("cam01", "cam02")
    assert transition is not None
    assert (transition.min_ms, transition.max_ms) == (3000, 15000)
    assert topo.distance_m("cam01", "cam02") == 12.0


def test_bidirectional_sinh_ca_chieu_nguoc():
    topo = _topo()
    back = topo.transition("cam02", "cam01")

    assert back is not None
    assert (back.min_ms, back.max_ms) == (3000, 15000)
    assert back.src == "cam02" and back.dst == "cam01"
    assert topo.neighbours("cam02") == ["cam01"]


def test_mot_chieu_khong_tu_sinh_chieu_nguoc():
    topo = _topo(transitions=[{"from": "cam01", "to": "cam02", "min_ms": 1000, "max_ms": 5000}])

    assert topo.transition("cam01", "cam02") is not None
    assert topo.transition("cam02", "cam01") is None


def test_transition_tro_toi_camera_khong_ton_tai_thi_bao_loi():
    with pytest.raises(TopologyError, match="không có trong khối"):
        _topo(transitions=[{"from": "cam01", "to": "cam99", "min_ms": 0, "max_ms": 1000}])


def test_khoang_thoi_gian_nguoc_thi_bao_loi():
    with pytest.raises(TopologyError, match="không hợp lệ"):
        _topo(transitions=[{"from": "cam01", "to": "cam02", "min_ms": 9000, "max_ms": 1000}])


def test_overlaps_with_mot_chieu_bi_bat_ngay_luc_nap():
    """Khai chồng lấn một chiều gần như luôn là quên — hậu quả rất khó nhận ra về sau."""
    cameras = {
        "cam01": {"overlaps_with": ["cam02"]},
        "cam02": {},
    }
    with pytest.raises(TopologyError, match="không đối xứng"):
        Topology.from_mapping({"cameras": cameras})


def test_overlaps_with_tro_toi_camera_la():
    with pytest.raises(TopologyError, match="chưa khai báo"):
        Topology.from_mapping({"cameras": {"cam01": {"overlaps_with": ["camX"]}}})


def test_policy_la_gia_tri_la_thi_bao_loi():
    with pytest.raises(TopologyError, match="unknown_pair_policy"):
        _topo(unknown_pair_policy="maybe")


# --------------------------------------------------------------------------- #
# Ràng buộc thời gian
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("elapsed_ms", "expected"),
    [(2_999, False), (3_000, True), (9_000, True), (15_000, True), (15_001, False)],
)
def test_bien_cua_khoang_transit(elapsed_ms, expected):
    assert _topo().is_feasible("cam01", "cam02", elapsed_ms) is expected


def test_ly_do_loai_noi_ro_qua_nhanh_hay_qua_lau():
    topo = _topo()

    assert "quá nhanh" in topo.check("cam01", "cam02", 500).reason
    assert "quá lâu" in topo.check("cam01", "cam02", 60_000).reason
    assert topo.check("cam01", "cam02", 500).transition is not None


def test_thoi_gian_am_bi_loai_o_cap_khong_chong_lan():
    """Xuất hiện ở camera đích TRƯỚC khi rời camera nguồn là bất khả thi."""
    assert not _topo().is_feasible("cam01", "cam02", -4_000)


def test_cap_chong_lan_bo_qua_chieu_thoi_gian():
    """Hai camera nhìn chung một vùng thấy cùng lúc; message hai luồng tới lệch nhau là thường."""
    topo = Topology.from_mapping(
        {
            "cameras": {
                "cam01": {"overlaps_with": ["cam02"]},
                "cam02": {"overlaps_with": ["cam01"]},
            }
        }
    )

    assert topo.is_overlapping("cam01", "cam02")
    assert topo.is_feasible("cam01", "cam02", 0)
    assert topo.is_feasible("cam01", "cam02", -800), "lệch âm vài trăm ms vẫn phải chấp nhận"


def test_cap_chong_lan_van_ton_trong_max_ms_neu_co_khai():
    topo = Topology.from_mapping(
        {
            "cameras": {
                "cam01": {"overlaps_with": ["cam02"]},
                "cam02": {"overlaps_with": ["cam01"]},
            },
            "transitions": [
                {"from": "cam01", "to": "cam02", "bidirectional": True, "min_ms": 0, "max_ms": 2000}
            ],
        }
    )

    assert topo.is_feasible("cam01", "cam02", -1_500)
    assert not topo.is_feasible("cam01", "cam02", 5_000)


def test_cung_mot_camera_luon_thoa():
    """Rời khung rồi quay lại là hợp lệ; chặn trùng trong cùng camera là việc của gallery."""
    topo = _topo()
    assert topo.is_feasible("cam01", "cam01", 50_000)
    assert "cùng camera" in topo.check("cam01", "cam01", 0).reason


def test_cap_chua_khai_bao_theo_policy_allow():
    topo = _topo()
    assert topo.transition("cam01", "cam03") is None
    assert topo.is_feasible("cam01", "cam03", 1)
    assert "chưa khai báo" in topo.check("cam01", "cam03", 1).reason


def test_cap_chua_khai_bao_theo_policy_reject():
    topo = _topo(unknown_pair_policy="reject")
    assert not topo.is_feasible("cam01", "cam03", 5_000)
    # Cặp đã khai vẫn hoạt động bình thường khi siết policy.
    assert topo.is_feasible("cam01", "cam02", 5_000)


def test_camera_la_thi_bao_loi_thay_vi_am_tham_cho_qua():
    with pytest.raises(TopologyError, match="không có trong topology"):
        _topo().check("cam01", "cam99", 1_000)


def test_ket_qua_dung_duoc_nhu_bool():
    result = _topo().check("cam01", "cam02", 5_000)
    assert result and result.feasible
    assert result.elapsed_ms == 5_000


# --------------------------------------------------------------------------- #
# File cấu hình thật
# --------------------------------------------------------------------------- #


def test_topology_yaml_that_nap_duoc_va_khop_fixture(repo_root):
    """configs/cameras/topology.yaml phải khớp fixture tổng hợp two_cam_walk.

    Fixture đặt transit danh nghĩa 8s, lệch ngẫu nhiên ±25% → [6s, 10s]; khoảng khai báo
    [3s, 15s] phải phủ trọn, nếu không engine sẽ cắt mất chính những match đúng mà cả
    bộ test M4 dựa vào.
    """
    topo = Topology.load(repo_root / "configs" / "cameras" / "topology.yaml")

    assert topo.camera_ids == ["cam01", "cam02"]
    transition = topo.transition("cam01", "cam02")
    assert transition is not None
    assert transition.min_ms <= 6_000
    assert transition.max_ms >= 10_000
    assert topo.transition("cam02", "cam01") is not None, "phải khai bidirectional"
