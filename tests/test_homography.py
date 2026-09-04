"""Test cho `mct.homography`: ước lượng H, đọc/ghi file hiệu chỉnh, và giao thức GroundMapper."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from mct.affinity import AffinityConfig, GroundMapper, build_cost_matrix
from mct.gallery import Gallery
from mct.homography import (
    CameraHomography,
    HomographyMapper,
    apply_homography,
    estimate_homography,
    reprojection_errors,
)
from mct.topology import Topology
from mct.tracklet import Observation, Tracklet

# H "thật" dùng để sinh dữ liệu: một phép chiếu phối cảnh có thật, không phải affine
# (hàng cuối khác 0 0 1 — nếu không thì bài toán suy biến thành affine và test dễ dãi).
TRUE_H = np.array(
    [
        [0.0042, 0.0009, -3.15],
        [0.0006, 0.0130, -8.40],
        [0.00002, 0.00060, 1.0],
    ],
    dtype=np.float64,
)


def _grid_points(n: int = 6) -> np.ndarray:
    xs = np.linspace(200.0, 1700.0, n)
    ys = np.linspace(400.0, 1000.0, n)
    return np.array([(x, y) for x in xs for y in ys], dtype=np.float64)


def _project_all(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return np.array([apply_homography(matrix, tuple(p)) for p in pts], dtype=np.float64)


# ------------------------------------------------------------------ ước lượng


def test_khoi_phuc_dung_h_tu_du_lieu_khong_nhieu():
    img = _grid_points()
    world = _project_all(TRUE_H, img)

    fit = estimate_homography(img, world, trim_ratio=0.0)

    assert fit.rmse_m < 1e-9
    assert np.allclose(fit.matrix, TRUE_H / TRUE_H[2, 2], atol=1e-9)


def test_bon_diem_la_du():
    img = np.array([[100.0, 900.0], [1800.0, 900.0], [1500.0, 500.0], [400.0, 500.0]])
    world = _project_all(TRUE_H, img)

    fit = estimate_homography(img, world, trim_ratio=0.0)

    assert fit.n_points == 4
    assert fit.rmse_m < 1e-9


def test_it_hon_bon_diem_thi_bao_loi():
    img = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="ít nhất 4"):
        estimate_homography(img, img)


def test_so_diem_hai_ve_lech_nhau_thi_bao_loi():
    with pytest.raises(ValueError, match="khác số điểm"):
        estimate_homography(_grid_points(), _grid_points()[:-1])


def test_diem_thang_hang_thi_bao_loi():
    img = np.array([[float(i), 2.0 * i] for i in range(1, 8)])
    world = np.array([[float(i), 2.0 * i] for i in range(1, 8)])
    with pytest.raises(ValueError):
        estimate_homography(img, world, trim_ratio=0.0)


def test_chuan_hoa_giup_uoc_luong_on_dinh_voi_toa_do_pixel_lon():
    # Không chuẩn hoá thì DLT trên toạ độ cỡ 10^3 trộn với mét cỡ 10^0 sai rõ rệt.
    # Ở đây chỉ khẳng định kết quả CÓ chuẩn hoá là chính xác tới mức máy.
    img = _grid_points(8) * np.array([1.0, 1.0]) + 1000.0
    world = _project_all(TRUE_H, img)

    fit = estimate_homography(img, world, trim_ratio=0.0)

    assert fit.rmse_m < 1e-8


def test_cat_tia_lam_giam_sai_so_khi_co_diem_ngoai_lai():
    rng = np.random.default_rng(7)
    img = _grid_points(8)
    world = _project_all(TRUE_H, img)
    # 8% điểm hỏng: mô phỏng bbox bị cắt ở mép khung -> điểm chân lệch hẳn.
    n_bad = max(1, len(img) // 12)
    bad = rng.choice(len(img), size=n_bad, replace=False)
    world[bad] += rng.normal(0.0, 6.0, size=(n_bad, 2))

    plain = estimate_homography(img, world, trim_ratio=0.0)
    trimmed = estimate_homography(img, world, trim_ratio=0.15, trim_rounds=2)

    # So trên tập điểm SẠCH: cắt tỉa phải bám dữ liệu đúng sát hơn.
    clean = np.setdiff1d(np.arange(len(img)), bad)
    err_plain = reprojection_errors(plain.matrix, img[clean], world[clean]).mean()
    err_trimmed = reprojection_errors(trimmed.matrix, img[clean], world[clean]).mean()
    assert err_trimmed < err_plain
    assert trimmed.n_points < trimmed.n_input


def test_diem_tren_duong_chan_troi_chieu_ra_none():
    # Chọn điểm làm mẫu số w = 0: H[2] . (u, v, 1) = 0.
    h2 = TRUE_H[2]
    u = 500.0
    v = -(h2[0] * u + h2[2]) / h2[1]
    assert apply_homography(TRUE_H, (u, v)) is None


# ------------------------------------------------- CameraHomography: đọc/ghi/kiểm tra


def test_luu_va_nap_lai_giu_nguyen_ma_tran(tmp_path):
    cam = CameraHomography(
        cam_id="cam01",
        matrix=TRUE_H,
        plane="ground",
        image_size=(1920, 1080),
        rmse_m=0.21,
        n_points=1234,
        source="test",
    )
    path = cam.save(tmp_path / "cam01.yaml")

    again = CameraHomography.load(path)

    assert again.cam_id == "cam01"
    assert again.image_size == (1920, 1080)
    assert again.rmse_m == pytest.approx(0.21)
    assert np.allclose(again.matrix, TRUE_H)
    # File phải đọc được bằng mắt: ma trận là list lồng, không phải blob numpy.
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw["matrix"][0][0], float)


def test_ma_tran_suy_bien_bi_tu_choi():
    with pytest.raises(ValueError, match="suy biến"):
        CameraHomography(cam_id="cam01", matrix=np.ones((3, 3)))


def test_ma_tran_sai_kich_thuoc_bi_tu_choi():
    with pytest.raises(ValueError, match="3x3"):
        CameraHomography(cam_id="cam01", matrix=np.eye(2))


# ------------------------------------------------------------- HomographyMapper


def _two_camera_mapper() -> HomographyMapper:
    """Hai camera nhìn cùng một mặt phẳng bằng hai phép chiếu khác nhau."""
    other = TRUE_H.copy()
    other[0, 2] += 4.0  # camera thứ hai lệch 4 m theo X
    other[0, 0] *= 1.3
    return HomographyMapper(
        {
            "cam01": CameraHomography("cam01", TRUE_H, image_size=(1920, 1080)),
            "cam02": CameraHomography("cam02", other, image_size=(1920, 1080)),
        }
    )


def test_mapper_thoa_giao_thuc_ground_mapper():
    assert isinstance(_two_camera_mapper(), GroundMapper)


def test_khoang_cach_bang_khong_khi_hai_camera_cung_thay_mot_diem():
    mapper = _two_camera_mapper()
    point_a = (800.0, 700.0)
    world = mapper.project("cam01", point_a)
    assert world is not None

    # Tìm điểm ảnh của cam02 chiếu về đúng `world`: giải ngược bằng H^-1.
    inv = np.linalg.inv(mapper.cameras["cam02"].matrix)
    point_b = apply_homography(inv, world)
    assert point_b is not None

    assert mapper.distance_m("cam01", point_a, "cam02", point_b) == pytest.approx(0.0, abs=1e-9)


def test_camera_chua_hieu_chinh_tra_none():
    mapper = _two_camera_mapper()
    assert mapper.distance_m("cam01", (800.0, 700.0), "cam09", (800.0, 700.0)) is None
    assert mapper.project("cam09", (1.0, 2.0)) is None
    assert mapper.has("cam09") is False


def test_hai_mat_phang_khac_nhau_thi_tu_choi():
    with pytest.raises(ValueError, match="mặt phẳng khác nhau"):
        HomographyMapper(
            {
                "cam01": CameraHomography("cam01", TRUE_H, plane="ground"),
                "cam02": CameraHomography("cam02", TRUE_H, plane="tang2"),
            }
        )


def test_nap_ca_thu_muc(tmp_path):
    mapper = _two_camera_mapper()
    mapper.save(tmp_path)

    again = HomographyMapper.load(tmp_path)

    assert again.calibrated == ["cam01", "cam02"]
    assert np.allclose(again.cameras["cam01"].matrix, TRUE_H)


def test_nap_file_gop(tmp_path):
    path = tmp_path / "all.yaml"
    cameras = {cam_id: cam.to_mapping() for cam_id, cam in _two_camera_mapper().cameras.items()}
    path.write_text(yaml.safe_dump({"cameras": cameras}), encoding="utf-8")

    mapper = HomographyMapper.load(path)

    assert mapper.calibrated == ["cam01", "cam02"]


def test_thu_muc_rong_bao_loi(tmp_path):
    with pytest.raises(FileNotFoundError):
        HomographyMapper.load(tmp_path)


def test_canh_bao_khi_do_phan_giai_khac_luc_hieu_chinh():
    mapper = _two_camera_mapper()
    assert mapper.check_frame_size("cam01", 1920, 1080) is None
    warning = mapper.check_frame_size("cam01", 1280, 720)
    assert warning is not None and "1280x720" in warning
    # Camera chưa hiệu chỉnh thì không có gì để cảnh báo.
    assert mapper.check_frame_size("cam09", 640, 480) is None
    assert len(mapper.warn_frame_sizes([("cam01", 1280, 720), ("cam02", 1920, 1080)])) == 1


# ------------------------------------------------- tích hợp với affinity


def _tracklet(cam_id: str, local_id: int, point: tuple[float, float], embedding) -> Tracklet:
    tracklet = Tracklet(tracklet_id=local_id, cam_id=cam_id, local_track_id=local_id)
    x, y = point
    tracklet.add(
        Observation(
            ts_ms=1_000,
            frame_id=1,
            bbox=(x - 20.0, y - 100.0, 40.0, 100.0),
            confidence=0.9,
            embedding=embedding,
            ground_point=(x, y),
        ),
        max_embeddings=8,
    )
    return tracklet


def test_affinity_cong_them_khoang_cach_mat_dat():
    """Hai tracklet ngoại hình giống hệt nhau: chi phí chỉ còn khác nhau ở hình học."""
    mapper = _two_camera_mapper()
    embedding = np.ones(8, dtype=np.float32) / np.sqrt(8.0)
    topology = Topology.from_mapping(
        {"cameras": {"cam01": {"overlaps_with": ["cam02"]}, "cam02": {"overlaps_with": ["cam01"]}}}
    )
    config = AffinityConfig(max_cost=0.5, homography_weight=0.4, max_ground_dist_m=50.0)

    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, (800.0, 700.0), embedding))
    # Điểm của cam02 chiếu về đúng chỗ cam01 đang đứng -> d_ground = 0.
    inv = np.linalg.inv(mapper.cameras["cam02"].matrix)
    world = mapper.project("cam01", (800.0, 700.0))
    same_spot = apply_homography(inv, world)

    near = _tracklet("cam02", 2, same_spot, embedding)
    far = _tracklet("cam02", 3, (same_spot[0] + 400.0, same_spot[1]), embedding)

    matrix = build_cost_matrix(
        [near, far], [track], topology=topology, config=config, ground_mapper=mapper
    )

    assert matrix.costs[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert matrix.costs[1, 0] > matrix.costs[0, 0]


def test_vuot_max_ground_dist_thi_loai_thang():
    mapper = _two_camera_mapper()
    embedding = np.ones(8, dtype=np.float32) / np.sqrt(8.0)
    topology = Topology.from_mapping(
        {"cameras": {"cam01": {"overlaps_with": ["cam02"]}, "cam02": {"overlaps_with": ["cam01"]}}}
    )
    config = AffinityConfig(max_cost=0.5, max_ground_dist_m=0.5)

    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, (400.0, 900.0), embedding))
    far = _tracklet("cam02", 2, (1700.0, 500.0), embedding)

    matrix = build_cost_matrix(
        [far], [track], topology=topology, config=config, ground_mapper=mapper
    )

    assert not np.isfinite(matrix.costs[0, 0])
    assert "mặt phẳng tham chiếu" in matrix.reason(0, 0)


def test_cap_khong_chong_lan_thi_bo_qua_hinh_hoc():
    """Hai camera cách nhau một hành lang: khoảng cách mặt đất vô nghĩa, không được cộng."""
    mapper = _two_camera_mapper()
    embedding = np.ones(8, dtype=np.float32) / np.sqrt(8.0)
    topology = Topology.from_mapping(
        {
            "cameras": {"cam01": {}, "cam02": {}},
            "transitions": [{"from": "cam01", "to": "cam02", "min_ms": 0, "max_ms": 60_000}],
        }
    )
    config = AffinityConfig(max_cost=0.5, max_ground_dist_m=0.5)

    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, (400.0, 900.0), embedding))
    far = _tracklet("cam02", 2, (1700.0, 500.0), embedding)

    matrix = build_cost_matrix(
        [far], [track], topology=topology, config=config, ground_mapper=mapper
    )

    assert matrix.costs[0, 0] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------- so vị trí theo thời gian (quỹ đạo)


def _walking_tracklet(
    cam_id: str,
    local_id: int,
    points: list[tuple[int, tuple[float, float]]],
    embedding,
) -> Tracklet:
    """Tracklet có nhiều quan sát theo thời gian — cần cho phép so quỹ đạo."""
    tracklet = Tracklet(tracklet_id=local_id, cam_id=cam_id, local_track_id=local_id)
    for frame_id, (ts_ms, (x, y)) in enumerate(points):
        tracklet.add(
            Observation(
                ts_ms=ts_ms,
                frame_id=frame_id,
                bbox=(x - 20.0, y - 100.0, 40.0, 100.0),
                confidence=0.9,
                embedding=embedding,
                ground_point=(x, y),
            ),
            max_embeddings=8,
            max_points=64,
        )
    return tracklet


def _overlap_topology() -> Topology:
    return Topology.from_mapping(
        {"cameras": {"cam01": {"overlaps_with": ["cam02"]}, "cam02": {"overlaps_with": ["cam01"]}}}
    )


def _mirror(mapper: HomographyMapper, cam_id: str, world: tuple[float, float]):
    """Điểm ảnh của `cam_id` chiếu về đúng vị trí `world`."""
    point = apply_homography(np.linalg.inv(mapper.cameras[cam_id].matrix), world)
    assert point is not None
    return point


def test_so_quy_dao_theo_thoi_gian_thay_vi_mot_diem():
    """Người đi qua khung: cùng lúc thì cùng chỗ, dù điểm đầu/điểm cuối cách nhau xa.

    Đây chính là ca mà cách cũ (điểm cuối của track ↔ điểm đầu của tracklet) làm sai:
    hai mốc thời gian khác nhau nên khoảng cách lớn dù là cùng một người.
    """
    mapper = _two_camera_mapper()
    embedding = np.ones(8, dtype=np.float32) / np.sqrt(8.0)
    world_path = [(1_000 + 500 * i, (2.0 + 0.8 * i, 3.0)) for i in range(8)]

    a = _walking_tracklet(
        "cam01", 1, [(ts, _mirror(mapper, "cam01", w)) for ts, w in world_path], embedding
    )
    b = _walking_tracklet(
        "cam02", 2, [(ts, _mirror(mapper, "cam02", w)) for ts, w in world_path], embedding
    )

    gallery = Gallery()
    track = gallery.create(a)
    matrix = build_cost_matrix(
        [b],
        [track],
        topology=_overlap_topology(),
        config=AffinityConfig(max_cost=0.5, max_ground_dist_m=0.5),
        ground_mapper=mapper,
    )

    # Cùng quỹ đạo, cùng mốc thời gian -> khoảng cách ~0 dù đi hết 5.6 m trong 3.5 s.
    assert matrix.costs[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_hai_nguoi_di_song_song_khac_cho_bi_loai():
    mapper = _two_camera_mapper()
    embedding = np.ones(8, dtype=np.float32) / np.sqrt(8.0)
    path_a = [(1_000 + 500 * i, (2.0 + 0.8 * i, 3.0)) for i in range(8)]
    path_b = [(ts, (x, y + 6.0)) for ts, (x, y) in path_a]  # lệch 6 m theo Y

    a = _walking_tracklet(
        "cam01", 1, [(ts, _mirror(mapper, "cam01", w)) for ts, w in path_a], embedding
    )
    b = _walking_tracklet(
        "cam02", 2, [(ts, _mirror(mapper, "cam02", w)) for ts, w in path_b], embedding
    )

    gallery = Gallery()
    track = gallery.create(a)
    matrix = build_cost_matrix(
        [b],
        [track],
        topology=_overlap_topology(),
        config=AffinityConfig(max_cost=0.5, max_ground_dist_m=1.0),
        ground_mapper=mapper,
    )

    assert not np.isfinite(matrix.costs[0, 0])
    assert "CÙNG thời điểm" in matrix.reason(0, 0)


def test_lech_thoi_gian_qua_dung_sai_thi_khong_con_moc_chung():
    """Lệch quá `ground_time_tol_ms` thì rơi về đường một-điểm, không phải đường quỹ đạo."""
    mapper = _two_camera_mapper()
    embedding = np.ones(8, dtype=np.float32) / np.sqrt(8.0)
    path_a = [(1_000 + 500 * i, (2.0 + 0.8 * i, 3.0)) for i in range(6)]
    path_b = [(ts + 20_000, w) for ts, w in path_a]  # 20 s sau, không mốc nào trùng

    a = _walking_tracklet(
        "cam01", 1, [(ts, _mirror(mapper, "cam01", w)) for ts, w in path_a], embedding
    )
    b = _walking_tracklet(
        "cam02", 2, [(ts, _mirror(mapper, "cam02", w)) for ts, w in path_b], embedding
    )

    gallery = Gallery()
    track = gallery.create(a)
    topology = _overlap_topology()

    allow = build_cost_matrix(
        [b],
        [track],
        topology=topology,
        config=AffinityConfig(max_cost=0.9, max_ground_dist_m=0.5, ground_gap_policy="allow"),
        ground_mapper=mapper,
    )
    reject = build_cost_matrix(
        [b],
        [track],
        topology=topology,
        config=AffinityConfig(max_cost=0.9, max_ground_dist_m=0.5, ground_gap_policy="reject"),
        ground_mapper=mapper,
    )

    # allow: ngân sách nới theo max_speed_m_s * 20 s nên cặp vẫn khả thi.
    assert np.isfinite(allow.costs[0, 0])
    # reject: không có mốc chung giữa hai camera chồng lấn -> loại thẳng.
    assert not np.isfinite(reject.costs[0, 0])
    assert "không có mốc thời gian chung" in reject.reason(0, 0)


def test_ground_gap_policy_sai_gia_tri_bi_tu_choi():
    with pytest.raises(ValueError, match="ground_gap_policy"):
        AffinityConfig(ground_gap_policy="maybe")


def test_quy_dao_bi_tia_khi_vuot_tran():
    tracklet = Tracklet(tracklet_id=1, cam_id="cam01", local_track_id=1)
    for i in range(50):
        tracklet.add(
            Observation(
                ts_ms=1_000 + 100 * i,
                frame_id=i,
                bbox=(0.0, 0.0, 10.0, 10.0),
                confidence=0.5,
                ground_point=(float(i), 0.0),
            ),
            max_embeddings=8,
            max_points=8,
        )

    assert 2 <= len(tracklet.ground_path) <= 8
    # Vẫn phủ trọn quãng thời gian, chỉ thưa hơn.
    assert tracklet.ground_path[0][0] == 1_000
    assert tracklet.ground_path[-1][0] >= 1_000 + 100 * 40
