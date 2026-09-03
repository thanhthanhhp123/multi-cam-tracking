"""Test gallery GlobalTrack (src/mct/gallery.py).

Trọng tâm là ba tính chất mà nếu sai thì associator phía sau không thể đúng được:
  - hạn ngạch theo camera: người đứng lâu trước một camera không được đẩy hết ngoại hình
    của camera khác ra khỏi gallery;
  - ràng buộc loại trừ: GlobalTrack đang hiện diện ở camera c không được làm ứng viên cho
    một tracklet KHÁC cũng thuộc c, nhưng vẫn phải nhận được cập nhật của chính nó;
  - similarity giữ đúng biên giữa "cùng người, khác camera" và "hai người mặc đồ giống nhau".
"""

from __future__ import annotations

import numpy as np
import pytest

from common.schema import CLASS_PERSON, Detection, FrameMessage, l2_normalize
from mct.gallery import Gallery, GalleryConfig, similarity_matrix
from mct.tracklet import Tracklet, TrackletBuilder, TrackletConfig, build_tracklets
from tools.make_synthetic_fixture import build_scenario

SEED = 42
DIM = 64

PARAMS = dict(
    identities=3,
    fps=15,
    embed_dim=256,
    intra_sim=0.80,
    cross_cam_sim=0.75,
    inter_sim=0.65,
    dwell_s=6.0,
    transit_s=8.0,
    stagger_s=4.0,
    miss_rate=0.04,
    seed=SEED,
)


def _tracklet(
    cam_id: str,
    local_track_id: int,
    *,
    start_ms: int,
    embedding: np.ndarray | None = None,
    confidence: float = 0.9,
    n_frames: int = 6,
    tracklet_id: int | None = None,
) -> Tracklet:
    """Tracklet dựng qua TrackletBuilder để không nhân bản logic gom ở test."""
    builder = TrackletBuilder(TrackletConfig(min_frames=1))
    for i in range(n_frames):
        det = Detection(
            local_track_id=local_track_id,
            bbox=(100.0, 200.0, 80.0, 200.0),
            confidence=confidence,
            embedding=embedding,
            class_id=CLASS_PERSON,
        )
        builder.update(
            FrameMessage(
                cam_id=cam_id,
                frame_id=i,
                ts_ms=start_ms + i * 66,
                frame_pts_ns=i * 66_666_667,
                frame_width=1920,
                frame_height=1080,
                detections=[det],
                embed_dim=0 if embedding is None else embedding.shape[0],
            )
        )
    tracklet = builder.flush()[0]
    if tracklet_id is not None:
        tracklet.tracklet_id = tracklet_id
    return tracklet


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
# Vòng đời
# --------------------------------------------------------------------------- #


def test_tracklet_dau_tien_tao_global_id_moi(rng):
    gallery = Gallery()
    vec = l2_normalize(rng.standard_normal(DIM))
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec))

    assert track.global_id == 1
    assert track.cameras == {"cam01"}
    assert track.n_tracklets == 1
    assert len(gallery) == 1
    assert track.centroid is not None
    assert float(np.linalg.norm(track.centroid)) == pytest.approx(1.0, abs=1e-5)


def test_gan_them_tracklet_camera_khac_mo_rong_global_track(rng):
    gallery = Gallery()
    vec = l2_normalize(rng.standard_normal(DIM))
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec))
    gallery.assign(track, _tracklet("cam02", 4, start_ms=10_000, embedding=vec))

    assert track.cameras == {"cam01", "cam02"}
    assert track.n_tracklets == 2
    assert track.last_cam_id == "cam02"
    assert track.last_seen_in("cam01") == 5 * 66
    assert track.last_seen_in("cam02") == 10_000 + 5 * 66


def test_cung_tracklet_gan_lai_khong_dem_thanh_luot_moi(rng):
    """Tracklet dài được xét lại ở nhiều cửa sổ — không được nhân bản thành nhiều lượt."""
    gallery = Gallery()
    vec = l2_normalize(rng.standard_normal(DIM))

    first = _tracklet("cam01", 1, start_ms=0, embedding=vec, n_frames=5, tracklet_id=7)
    track = gallery.create(first)
    later = _tracklet("cam01", 1, start_ms=0, embedding=vec, n_frames=12, tracklet_id=7)
    gallery.assign(track, later)

    assert track.n_tracklets == 1
    assert len(track.members) == 1
    assert track.members[-1].end_ms == later.end_ms


def test_prune_dong_global_track_qua_han(rng):
    gallery = Gallery(GalleryConfig(global_track_ttl_ms=10_000))
    vec = l2_normalize(rng.standard_normal(DIM))
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec))

    assert gallery.prune(5_000) == []
    closed = gallery.prune(60_000)

    assert [t.global_id for t in closed] == [track.global_id]
    assert track.closed
    assert len(gallery) == 0
    assert gallery.get(track.global_id) is None
    with pytest.raises(ValueError, match="đã đóng"):
        gallery.assign(track, _tracklet("cam02", 1, start_ms=61_000, embedding=vec))


# --------------------------------------------------------------------------- #
# Ràng buộc loại trừ cùng camera
# --------------------------------------------------------------------------- #


def test_track_dang_o_camera_do_bi_loai_khoi_ung_vien(rng):
    gallery = Gallery()
    vec = l2_normalize(rng.standard_normal(DIM))
    gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1))

    other = _tracklet("cam01", 2, start_ms=100, embedding=vec, tracklet_id=2)
    assert gallery.candidates(other, exclusion_window_ms=2_000) == []

    # Cũng người đó nhưng ở camera khác thì vẫn là ứng viên hợp lệ.
    cross = _tracklet("cam02", 2, start_ms=100, embedding=vec, tracklet_id=3)
    assert [t.global_id for t in gallery.candidates(cross, exclusion_window_ms=2_000)] == [1]


def test_rang_buoc_loai_tru_khong_chan_cap_nhat_chinh_no(rng):
    """Tracklet đang chạy phải gán lại được vào đúng GlobalTrack của nó ở cửa sổ sau."""
    gallery = Gallery()
    vec = l2_normalize(rng.standard_normal(DIM))
    tracklet = _tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=11)
    track = gallery.create(tracklet)

    longer = _tracklet("cam01", 1, start_ms=0, embedding=vec, n_frames=20, tracklet_id=11)
    assert [t.global_id for t in gallery.candidates(longer, exclusion_window_ms=2_000)] == [
        track.global_id
    ]
    assert gallery.find_by_tracklet(longer) is track


def test_het_cua_so_loai_tru_thi_lai_thanh_ung_vien(rng):
    """Người rời camera rồi quay lại (local_track_id mới) vẫn phải khớp lại được."""
    gallery = Gallery()
    vec = l2_normalize(rng.standard_normal(DIM))
    gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1))

    comeback = _tracklet("cam01", 9, start_ms=30_000, embedding=vec, tracklet_id=2)
    assert len(gallery.candidates(comeback, exclusion_window_ms=2_000)) == 1


# --------------------------------------------------------------------------- #
# Hạn ngạch gallery
# --------------------------------------------------------------------------- #


def test_gallery_khong_vuot_max_size(rng):
    gallery = Gallery(GalleryConfig(max_size=4))
    track = None
    for i in range(10):
        vec = l2_normalize(rng.standard_normal(DIM))
        t = _tracklet("cam01", i, start_ms=i * 1_000, embedding=vec, tracklet_id=i + 1)
        track = gallery.create(t) if track is None else gallery.assign(track, t)

    assert track is not None
    assert len(track.entries) == 4


def test_camera_hiem_gap_khong_bi_day_khoi_gallery(rng):
    """Đứng lâu trước cam01 không được xoá sạch ngoại hình chụp ở cam02.

    Đây là lý do hạn ngạch chia theo camera thay vì chỉ xếp hạng theo confidence.
    """
    gallery = Gallery(GalleryConfig(max_size=6))
    cam02_vec = l2_normalize(rng.standard_normal(DIM))

    track = gallery.create(
        _tracklet("cam02", 1, start_ms=0, embedding=cam02_vec, confidence=0.5, tracklet_id=1)
    )
    for i in range(12):
        vec = l2_normalize(rng.standard_normal(DIM))
        gallery.assign(
            track,
            _tracklet(
                "cam01",
                i,
                start_ms=1_000 + i * 500,
                embedding=vec,
                confidence=0.99,
                tracklet_id=100 + i,
            ),
        )

    cams = [entry.cam_id for entry in track.entries]
    assert "cam02" in cams, "ngoại hình của camera hiếm gặp bị confidence cao của cam01 đè mất"
    assert len(track.entries) == 6


def test_khong_co_embedding_van_theo_doi_duoc_vong_doi():
    """Pipeline M1/M2 chạy ReID tắt: gallery rỗng nhưng vòng đời vẫn phải đúng."""
    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, start_ms=0))

    assert track.entries == []
    assert track.centroid is None
    assert track.similarity(np.zeros(DIM, dtype=np.float32)) == -1.0
    assert track.n_tracklets == 1


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #


def test_similarity_max_khong_bi_pha_loang_boi_camera_khac(rng):
    """`max` phải giữ nguyên độ giống với ngoại hình đã thấy ở đúng camera đó."""
    a = l2_normalize(rng.standard_normal(DIM))
    b = l2_normalize(rng.standard_normal(DIM))

    gallery = Gallery(GalleryConfig(ema_alpha=0.5))
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=a, tracklet_id=1))
    gallery.assign(track, _tracklet("cam02", 1, start_ms=10_000, embedding=b, tracklet_id=2))

    assert track.similarity(a, "max") == pytest.approx(1.0, abs=1e-5)
    assert track.similarity(b, "max") == pytest.approx(1.0, abs=1e-5)
    # Centroid nằm giữa hai vector gần trực giao nên giống mỗi cái chỉ khoảng 0.7.
    assert track.similarity(a, "centroid") < 0.9
    assert track.similarity(b, "centroid") < 0.9


def test_similarity_matrix_danh_dau_tracklet_khong_co_embedding(rng):
    vec = l2_normalize(rng.standard_normal(DIM))
    gallery = Gallery()
    track = gallery.create(_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1))

    with_emb = _tracklet("cam02", 1, start_ms=10_000, embedding=vec, tracklet_id=2)
    without = _tracklet("cam02", 2, start_ms=10_000, tracklet_id=3)

    matrix = similarity_matrix([with_emb, without], [track])
    assert matrix.shape == (2, 1)
    assert matrix[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert matrix[1, 0] == -1.0


# --------------------------------------------------------------------------- #
# Trên fixture tổng hợp
# --------------------------------------------------------------------------- #


def test_gallery_phan_biet_dung_danh_tinh_tren_fixture():
    """Dựng gallery từ tracklet ở cam01 rồi chấm điểm tracklet cam02 bằng ground truth.

    Đây là bài kiểm tra "gallery có đủ sức phân biệt không" tách khỏi thuật toán gán:
    với mỗi tracklet của cam02, GlobalTrack giống nhất phải đúng là người đó.
    """
    messages, appearances = build_scenario(**PARAMS)  # type: ignore[arg-type]
    tracklets = build_tracklets(messages, TrackletConfig(min_frames=5, topk_query=8))
    gt = {(a.cam_id, a.local_track_id): a.gt_global_id for a in appearances}

    gallery = Gallery(GalleryConfig(max_size=32, topk_query=8, similarity_mode="max"))
    gid_of_identity: dict[int, int] = {}
    for tracklet in (t for t in tracklets if t.cam_id == "cam01"):
        track = gallery.create(tracklet)
        gid_of_identity[gt[tracklet.key]] = track.global_id

    for tracklet in (t for t in tracklets if t.cam_id == "cam02"):
        query = tracklet.query_embedding(8)
        assert query is not None
        best = max(gallery, key=lambda track: track.similarity(query, "max"))
        assert best.global_id == gid_of_identity[gt[tracklet.key]], (
            f"tracklet {tracklet.key} khớp nhầm GlobalTrack {best.global_id}"
        )
