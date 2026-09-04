"""Test `eval/diagnose_tracklets.py`.

Công cụ này sinh ra con số dùng để QUYẾT ĐỊNH sửa tracker hay sửa `src/mct`. Nó sai thì
quyết định sai theo mà không có gì báo. Vì vậy test dựng những cặp tracklet có đáp án biết
trước — trùng thời gian / không trùng, trong tầm / ngoài tầm — rồi soi đúng ô số tương ứng.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from eval.diagnose_tracklets import fragmentation, load_gt, temporal_overlap_and_ceiling

from mct.affinity import AffinityConfig
from mct.tracklet import Tracklet


class IdentityMapper:
    """Homography giả: toạ độ ảnh CHÍNH LÀ toạ độ mặt đất, mét.

    Nhờ vậy khoảng cách trong test đọc thẳng ra được từ số liệu đầu vào, không phải suy
    qua ma trận nào.
    """

    @property
    def calibrated(self) -> list[str]:
        return ["cam01", "cam02", "cam03"]

    def project(self, cam_id: str, point: tuple[float, float]) -> tuple[float, float] | None:
        return point

    def distance_m(self, cam_a, point_a, cam_b, point_b) -> float | None:
        return float(((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5)


_next_id = [0]


def make_tracklet(
    cam_id: str, local_id: int, path: list[tuple[int, tuple[float, float]]]
) -> Tracklet:
    _next_id[0] += 1
    t = Tracklet(tracklet_id=_next_id[0], cam_id=cam_id, local_track_id=local_id)
    t.ground_path = list(path)
    t.start_ms = path[0][0]
    t.end_ms = path[-1][0]
    t.first_ground_point = path[0][1]
    t.last_ground_point = path[-1][1]
    t.n_frames = len(path)
    return t


def straight(cam_id: str, local_id: int, t0: int, x: float, *, n: int = 5, step: int = 500):
    """Tracklet đứng yên tại x, lấy mẫu mỗi `step` ms."""
    return make_tracklet(cam_id, local_id, [(t0 + i * step, (x, 0.0)) for i in range(n)])


CONFIG = AffinityConfig(ground_time_tol_ms=400, max_ground_dist_m=1.0, max_speed_m_s=2.5)


def run(tracklets, gt, config=CONFIG):
    return temporal_overlap_and_ceiling(
        tracklets, gt, IdentityMapper(), config, max_negative_pairs=1000, seed=0
    )


# --------------------------------------------------------------------------------------
# Phần 2: trùng thời gian
# --------------------------------------------------------------------------------------


def test_cap_trung_thoi_gian_va_dung_cho_thi_ghep_duoc():
    """Hai camera thấy cùng một người, cùng lúc, cùng chỗ -> ràng buộc phải cho qua."""
    a = straight("cam01", 1, 1000, 0.0)
    b = straight("cam02", 1, 1000, 0.2)
    r = run([a, b], {a.key: 7, b.key: 7})

    assert r["positive"]["n_pairs"] == 1
    assert r["positive"]["share_with_common_timestamp"] == 1.0
    assert r["positive"]["feasible_reject"] == 1.0
    assert r["positive"]["feasible_allow"] == 1.0


def test_trung_thoi_gian_nhung_cach_qua_xa_thi_bi_loai_o_ca_hai_chinh_sach():
    """Khoảng cách mới là thứ loại, không phải chính sách — `allow` không cứu được cặp này."""
    a = straight("cam01", 1, 1000, 0.0)
    b = straight("cam02", 1, 1000, 50.0)
    r = run([a, b], {a.key: 7, b.key: 7})

    assert r["positive"]["share_with_common_timestamp"] == 1.0
    assert r["positive"]["feasible_reject"] == 0.0
    assert r["positive"]["feasible_allow"] == 0.0


def test_khong_co_moc_chung_thi_reject_loai_con_allow_cho_qua():
    """ĐÂY là ô số quan trọng nhất của cả công cụ.

    Hai tracklet của cùng một người, hai camera, nhưng lệch thời gian hoàn toàn — đúng thứ
    mà tracker vỡ vụn sinh ra hàng loạt. `reject` loại sạch (recall về 0), `allow` rơi
    xuống so đầu-cuối với ngân sách theo tốc độ đi bộ và cho qua.
    """
    a = straight("cam01", 1, 1000, 0.0)
    b = straight("cam02", 1, 100_000, 0.5)
    r = run([a, b], {a.key: 7, b.key: 7})

    assert r["positive"]["share_with_common_timestamp"] == 0.0
    assert r["positive"]["feasible_reject"] == 0.0
    assert r["positive"]["feasible_allow"] == 1.0


def test_khong_co_moc_chung_va_xa_hon_ca_ngan_sach_toc_do():
    """`allow` không phải là thả cửa: vẫn có ngân sách max_ground_dist + v·Δt."""
    a = straight("cam01", 1, 1000, 0.0)
    b = straight("cam02", 1, 4000, 500.0)  # Δt ~1.5s, ngân sách ~4.75m, cách 500m
    r = run([a, b], {a.key: 7, b.key: 7})

    assert r["positive"]["share_with_common_timestamp"] == 0.0
    assert r["positive"]["feasible_allow"] == 0.0


def test_cap_cung_camera_khong_duoc_tinh():
    """Cặp cùng camera do ràng buộc loại trừ xử lý riêng; gộp vào là thổi phồng số liệu."""
    a = straight("cam01", 1, 1000, 0.0)
    b = straight("cam01", 2, 1000, 0.1)
    r = run([a, b], {a.key: 7, b.key: 7})

    assert r["positive"]["n_pairs"] == 0
    assert r["negative"]["n_pairs"] == 0


def test_cap_khac_nguoi_vao_nhom_negative():
    a = straight("cam01", 1, 1000, 0.0)
    b = straight("cam02", 1, 1000, 0.2)
    r = run([a, b], {a.key: 7, b.key: 9})

    assert r["positive"]["n_pairs"] == 0
    assert r["negative"]["n_pairs"] == 1
    # Hai người khác nhau đứng chồng chỗ: hình học KHÔNG tách được, và công cụ phải nói
    # đúng như vậy thay vì che đi.
    assert r["negative"]["feasible_reject"] == 1.0


def test_tracklet_ngoai_bang_gt_bi_bo_qua():
    a = straight("cam01", 1, 1000, 0.0)
    b = straight("cam02", 1, 1000, 0.2)
    r = run([a, b], {a.key: 7})  # b không có trong bảng GT

    assert r["positive"]["n_pairs"] == 0
    assert r["negative"]["n_pairs"] == 0


def test_dung_sai_moc_chung_duoc_ton_trong():
    """Lệch 300 ms thì trong dung sai 400 ms; lệch 900 ms thì không."""
    a = make_tracklet("cam01", 1, [(1000, (0.0, 0.0)), (1500, (0.0, 0.0))])
    gan = make_tracklet("cam02", 1, [(1300, (0.1, 0.0)), (1800, (0.1, 0.0))])
    xa = make_tracklet("cam03", 1, [(2400, (0.1, 0.0)), (2900, (0.1, 0.0))])

    assert run([a, gan], {a.key: 7, gan.key: 7})["positive"]["share_with_common_timestamp"] == 1.0
    assert run([a, xa], {a.key: 7, xa.key: 7})["positive"]["share_with_common_timestamp"] == 0.0


def test_lay_mau_cap_am_khong_dung_cham_cap_duong():
    """Giới hạn `max_negative_pairs` chỉ được cắt bớt cặp ÂM — cắt nhầm cặp dương là làm
    hỏng chính con số trần recall."""
    tracklets = [straight(f"cam{1 + i % 3:02d}", i, 1000, 0.1 * i) for i in range(12)]
    gt = {t.key: (7 if i < 4 else 100 + i) for i, t in enumerate(tracklets)}

    r = temporal_overlap_and_ceiling(
        tracklets, gt, IdentityMapper(), CONFIG, max_negative_pairs=3, seed=0
    )
    assert r["negative"]["n_pairs"] == 3
    # 4 tracklet cùng danh tính trên 3 camera -> 5 cặp khác camera (cặp cam01-cam01 bị loại)
    assert r["positive"]["n_pairs"] == 5


# --------------------------------------------------------------------------------------
# Phần 1: độ vỡ
# --------------------------------------------------------------------------------------


def test_do_vo_dem_theo_cap_danh_tinh_camera():
    """Một người thấy ở 3 camera KHÔNG phải là vỡ — vỡ là bị cắt nhiều mảnh TRONG một camera."""
    dep = [straight(f"cam{i:02d}", 1, 1000, 0.0) for i in (1, 2, 3)]
    vo = [straight("cam01", i, 1000 * i, 0.0) for i in (1, 2, 3)]

    r_dep = fragmentation(dep, {t.key: 7 for t in dep})
    r_vo = fragmentation(vo, {t.key: 7 for t in vo})

    assert r_dep["mean_tracklets_per_id_cam"] == 1.0
    assert r_dep["n_identities_multi_cam"] == 1
    assert r_vo["mean_tracklets_per_id_cam"] == 3.0
    assert r_vo["n_identities_multi_cam"] == 0


def test_do_vo_bo_qua_tracklet_ngoai_bang_gt():
    a = straight("cam01", 1, 1000, 0.0)
    b = straight("cam02", 1, 1000, 0.0)
    r = fragmentation([a, b], {a.key: 7})

    assert r["n_tracklets_built"] == 2
    assert r["n_tracklets_scored"] == 1
    assert r["n_identities"] == 1


def test_load_gt_doc_dung_dinh_dang_chung(tmp_path: Path):
    """Cùng định dạng với wildtrack_to_fixture.py và ds_wildtrack_gt.py."""
    import json

    path = tmp_path / "x.gt.json"
    path.write_text(
        json.dumps({"tracklets": [{"cam_id": "cam01", "local_track_id": 3, "gt_global_id": 9}]}),
        encoding="utf-8",
    )
    assert load_gt(path) == {("cam01", 3): 9}


def test_khong_co_cap_nao_thi_khong_chia_cho_khong():
    r = run([], {})
    assert r["positive"]["n_pairs"] == 0
    assert r["positive"]["feasible_reject"] == pytest.approx(0.0)
