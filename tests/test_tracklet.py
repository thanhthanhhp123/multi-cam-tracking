"""Test bộ gom tracklet (src/mct/tracklet.py).

Hai nhóm test:
  1. Hành vi máy trạng thái, dựng bằng message viết tay — kiểm những trường hợp biên
     mà fixture tổng hợp không có (local_track_id dùng lại, message tới lệch thứ tự,
     tracklet quá ngắn).
  2. Chạy trên fixture tổng hợp — số tracklet gom được phải khớp số lượt xuất hiện
     trong ground truth, và query embedding phải giữ được biên phân biệt giữa
     cross-cam-sim và inter-sim (nếu không thì mọi kết quả của associator phía sau
     đều vô nghĩa).
"""

from __future__ import annotations

import numpy as np
import pytest

from common.schema import CLASS_PERSON, Detection, FrameMessage, l2_normalize
from mct.tracklet import TrackletBuilder, TrackletConfig, build_tracklets
from tools.make_synthetic_fixture import BASE_TS_MS, build_scenario

SEED = 42

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


def _msg(
    cam_id: str,
    frame_id: int,
    ts_ms: int,
    tracks: dict[int, float],
    *,
    embeddings: dict[int, np.ndarray] | None = None,
) -> FrameMessage:
    """FrameMessage tối giản: `tracks` là {local_track_id: confidence}."""
    embeddings = embeddings or {}
    detections = [
        Detection(
            local_track_id=ltid,
            bbox=(100.0 + 10 * ltid, 200.0, 80.0, 200.0),
            confidence=conf,
            embedding=embeddings.get(ltid),
            class_id=CLASS_PERSON,
        )
        for ltid, conf in sorted(tracks.items())
    ]
    return FrameMessage(
        cam_id=cam_id,
        frame_id=frame_id,
        ts_ms=ts_ms,
        frame_pts_ns=frame_id * 66_666_667,
        frame_width=1920,
        frame_height=1080,
        detections=detections,
        embed_dim=next(iter(embeddings.values())).shape[0] if embeddings else 0,
    )


def _stream(cam_id: str, ltid: int, n: int, *, start_ms: int, step_ms: int = 66):
    for i in range(n):
        yield _msg(cam_id, i, start_ms + i * step_ms, {ltid: 0.9})


# --------------------------------------------------------------------------- #
# 1. Máy trạng thái
# --------------------------------------------------------------------------- #


def test_gom_detection_lien_tuc_thanh_mot_tracklet():
    tracklets = build_tracklets(_stream("cam01", 7, 10, start_ms=1_000))

    assert len(tracklets) == 1
    t = tracklets[0]
    assert t.key == ("cam01", 7)
    assert t.n_frames == 10
    assert t.start_ms == 1_000
    assert t.end_ms == 1_000 + 9 * 66
    assert t.duration_ms == 9 * 66
    assert t.closed


def test_tracklet_ngan_hon_min_frames_bi_bo():
    config = TrackletConfig(min_frames=5)
    builder = TrackletBuilder(config)
    builder.update_many(_stream("cam01", 1, 3, start_ms=0))

    assert builder.flush() == []
    assert builder.n_dropped_short == 1
    assert builder.n_closed == 0


def test_local_track_id_dung_lai_sinh_tracklet_moi():
    """Cạm bẫy chính: nvtracker cấp lại ID cũ cho người khác sau khi track chết.

    Nếu gộp chung, hai người khác nhau chung một gallery và Global ID phía sau sai theo.
    """
    config = TrackletConfig(min_frames=3, idle_timeout_ms=2_000)
    builder = TrackletBuilder(config)

    builder.update_many(_stream("cam01", 5, 6, start_ms=0))
    # Cùng local_track_id nhưng cách 10s — vượt idle_timeout_ms.
    closed = builder.update_many(_stream("cam01", 5, 6, start_ms=10_000))

    assert len(closed) == 1, "lượt xuất hiện đầu phải bị đóng ngay khi ID được dùng lại"
    assert closed[0].end_ms == 5 * 66

    remaining = builder.flush()
    assert len(remaining) == 1
    assert remaining[0].start_ms == 10_000
    assert remaining[0].tracklet_id != closed[0].tracklet_id
    assert builder.n_started == 2


def test_khoang_lang_ngan_hon_timeout_khong_cat_tracklet():
    """Detector trượt vài frame (miss_rate) không được cắt tracklet làm đôi."""
    config = TrackletConfig(min_frames=3, idle_timeout_ms=2_000)
    builder = TrackletBuilder(config)

    builder.update_many(_stream("cam01", 5, 5, start_ms=0))
    builder.update_many(_stream("cam01", 5, 5, start_ms=1_500))  # lặng 1.5s < 2s

    assert builder.update_many([]) == []
    tracklets = builder.flush()
    assert len(tracklets) == 1
    assert tracklets[0].n_frames == 10


def test_close_expired_dong_dung_track_het_han():
    config = TrackletConfig(min_frames=3, idle_timeout_ms=1_000)
    builder = TrackletBuilder(config)

    builder.update_many(_stream("cam01", 1, 5, start_ms=0))
    builder.update_many(_stream("cam01", 2, 5, start_ms=5_000))

    # Ở mốc 5.5s: track 1 im lặng >1s, track 2 vẫn đang sống.
    expired = builder.close_expired(5_500)
    assert [t.local_track_id for t in expired] == [1]
    assert [t.local_track_id for t in builder.active()] == [2]

    # Không truyền mốc thời gian thì lấy ts muộn nhất đã thấy trong dữ liệu.
    assert builder.close_expired() == []


def test_take_updated_chi_tra_tracklet_vua_thay_doi():
    config = TrackletConfig(min_frames=3)
    builder = TrackletBuilder(config)

    builder.update_many(_stream("cam01", 1, 5, start_ms=0))
    first = builder.take_updated()
    assert [t.local_track_id for t in first] == [1]
    assert builder.take_updated() == [], "không có cập nhật mới thì không trả lại nữa"

    builder.update_many(_stream("cam02", 9, 5, start_ms=1_000))
    assert [t.cam_id for t in builder.take_updated()] == ["cam02"]


def test_take_updated_bo_qua_tracklet_chua_du_dai():
    builder = TrackletBuilder(TrackletConfig(min_frames=5))
    builder.update_many(_stream("cam01", 1, 4, start_ms=0))
    assert builder.take_updated() == []

    builder.update_many(_stream("cam01", 1, 1, start_ms=400))
    assert len(builder.take_updated()) == 1


def test_message_lech_thu_tu_khong_lam_lui_moc_ket_thuc():
    builder = TrackletBuilder(TrackletConfig(min_frames=2))
    builder.update(_msg("cam01", 10, 1_000, {1: 0.9}))
    builder.update(_msg("cam01", 11, 1_100, {1: 0.9}))
    builder.update(_msg("cam01", 9, 900, {1: 0.9}))  # tới muộn, thuộc về quá khứ

    t = builder.flush()[0]
    assert t.n_frames == 3
    assert t.end_ms == 1_100, "end_ms phải là mốc muộn nhất, không phải message cuối cùng"
    assert t.end_frame_id == 11
    assert t.start_ms == 1_000


def test_hai_camera_khong_lan_id_cua_nhau():
    builder = TrackletBuilder(TrackletConfig(min_frames=3))
    builder.update_many(_stream("cam01", 1, 5, start_ms=0))
    builder.update_many(_stream("cam02", 1, 5, start_ms=0))

    tracklets = builder.flush()
    assert len(tracklets) == 2
    assert {t.cam_id for t in tracklets} == {"cam01", "cam02"}


def test_ground_point_lay_dung_quy_uoc_day_giua_bbox():
    builder = TrackletBuilder(TrackletConfig(min_frames=1))
    builder.update(_msg("cam01", 0, 1_000, {3: 0.9}))
    t = builder.flush()[0]

    x, y, w, h = t.last_bbox
    assert t.last_ground_point == (x + w / 2.0, y + h)
    assert t.first_ground_point == t.last_ground_point


# --------------------------------------------------------------------------- #
# 2. Query embedding
# --------------------------------------------------------------------------- #


def test_khong_co_embedding_thi_query_tra_none():
    """Pipeline M1/M2 chạy với ReID tắt — tracklet vẫn phải gom được, chỉ là không có vector."""
    t = build_tracklets(_stream("cam01", 1, 6, start_ms=0))[0]
    assert t.query_embedding() is None
    assert t.n_embeddings == 0


def test_query_embedding_uu_tien_detection_confidence_cao():
    """Crop mờ (confidence thấp) không được kéo query embedding đi xa."""
    rng = np.random.default_rng(SEED)
    good = l2_normalize(rng.standard_normal(64))
    bad = l2_normalize(rng.standard_normal(64))

    builder = TrackletBuilder(TrackletConfig(min_frames=1, topk_query=3))
    for i in range(6):
        builder.update(_msg("cam01", i, i * 66, {1: 0.95}, embeddings={1: good}))
    for i in range(6, 9):
        builder.update(_msg("cam01", i, i * 66, {1: 0.20}, embeddings={1: bad}))

    t = builder.flush()[0]
    query = t.query_embedding(top_k=3)
    assert query is not None
    assert float(query @ good) > 0.99
    assert float(query @ bad) < 0.2


def test_query_embedding_da_l2_normalize():
    rng = np.random.default_rng(SEED)
    vecs = [l2_normalize(rng.standard_normal(64)) for _ in range(5)]

    builder = TrackletBuilder(TrackletConfig(min_frames=1))
    for i, vec in enumerate(vecs):
        builder.update(_msg("cam01", i, i * 66, {1: 0.5 + 0.1 * i}, embeddings={1: vec}))

    query = builder.flush()[0].query_embedding()
    assert query is not None
    assert query.dtype == np.float32
    assert float(np.linalg.norm(query)) == pytest.approx(1.0, abs=1e-5)


def test_max_embeddings_chan_bo_nho_va_giu_lai_cai_tot_nhat():
    rng = np.random.default_rng(SEED)
    builder = TrackletBuilder(TrackletConfig(min_frames=1, max_embeddings=4))

    for i in range(20):
        vec = l2_normalize(rng.standard_normal(32))
        builder.update(_msg("cam01", i, i * 66, {1: i / 20.0}, embeddings={1: vec}))

    t = builder.flush()[0]
    assert t.n_embeddings == 4
    assert t.n_embeddings_seen == 20
    # Bốn detection cuối có confidence cao nhất (0.75..0.95) nên phải là bốn cái còn lại.
    assert t.mean_confidence == pytest.approx(sum(i / 20.0 for i in range(20)) / 20.0)


def test_query_embedding_duoc_cache_va_huy_khi_co_them_du_lieu():
    rng = np.random.default_rng(SEED)
    a = l2_normalize(rng.standard_normal(32))
    b = l2_normalize(rng.standard_normal(32))

    builder = TrackletBuilder(TrackletConfig(min_frames=1))
    builder.update(_msg("cam01", 0, 0, {1: 0.9}, embeddings={1: a}))
    t = builder.get("cam01", 1)
    assert t is not None

    first = t.query_embedding()
    assert first is not None and t.query_embedding() is first, "gọi lại phải dùng cache"

    builder.update(_msg("cam01", 1, 66, {1: 0.9}, embeddings={1: b}))
    second = t.query_embedding()
    assert second is not None and second is not first
    assert not np.allclose(first, second)


# --------------------------------------------------------------------------- #
# 3. Chạy trên fixture tổng hợp
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def scenario():
    return build_scenario(**PARAMS)  # type: ignore[arg-type]


def test_so_tracklet_khop_ground_truth(scenario):
    messages, appearances = scenario
    tracklets = build_tracklets(messages, TrackletConfig(min_frames=5, idle_timeout_ms=2_000))

    got = {(t.cam_id, t.local_track_id) for t in tracklets}
    expected = {(a.cam_id, a.local_track_id) for a in appearances}
    assert got == expected
    assert len(tracklets) == len(appearances), "không được cắt vụn hay gộp nhầm lượt xuất hiện"


def test_bien_thoi_gian_tracklet_bam_sat_ground_truth(scenario):
    messages, appearances = scenario
    tracklets = build_tracklets(messages, TrackletConfig(min_frames=5, idle_timeout_ms=2_000))
    by_key = {(t.cam_id, t.local_track_id): t for t in tracklets}

    fps_ms = 1000.0 / PARAMS["fps"]
    for app in appearances:
        t = by_key[(app.cam_id, app.local_track_id)]
        # Appearance ghi mốc tương đối từ đầu kịch bản, message ghi epoch thật.
        # Lệch tối đa vài frame vì miss_rate làm rụng frame ở hai đầu.
        assert abs(t.start_ms - (BASE_TS_MS + app.start_ms)) <= 4 * fps_ms
        assert abs(t.end_ms - (BASE_TS_MS + app.end_ms)) <= 4 * fps_ms


def test_query_embedding_giu_duoc_bien_phan_biet(scenario):
    """Gom tracklet không được làm mất khoảng cách giữa cross-cam-sim và inter-sim.

    Trung bình top-k khử nhiễu i.i.d. theo frame, nên similarity giữa hai query embedding
    tiến TỪ DƯỚI LÊN về đúng hai tham số đặt cho kịch bản (cross_cam_sim với cùng người,
    inter_sim với cặp "mặc đồ giống nhau") — chỉ còn thiếu phần nhiễu chưa bị triệt tiêu
    hết bởi k=8 mẫu. Điều phải giữ là biên: cùng người xuyên camera vẫn giống nhau hơn
    hẳn hai người khác nhau.
    """
    messages, appearances = scenario
    tracklets = build_tracklets(messages, TrackletConfig(min_frames=5, topk_query=8))
    gt = {(a.cam_id, a.local_track_id): a.gt_global_id for a in appearances}

    same_id: list[float] = []
    diff_id: list[float] = []
    for i, t1 in enumerate(tracklets):
        for t2 in tracklets[i + 1 :]:
            if t1.cam_id == t2.cam_id:
                continue
            q1, q2 = t1.query_embedding(8), t2.query_embedding(8)
            assert q1 is not None and q2 is not None
            sim = float(q1 @ q2)
            bucket = same_id if gt[t1.key] == gt[t2.key] else diff_id
            bucket.append(sim)

    assert same_id and diff_id
    assert min(same_id) > max(diff_id), (
        f"biên bị mất: cùng người xuyên camera thấp nhất {min(same_id):.3f}, "
        f"khác người cao nhất {max(diff_id):.3f}"
    )
    # Không được vượt trần lý thuyết, và phải đủ gần trần (k=8 đã khử phần lớn nhiễu).
    assert max(same_id) <= PARAMS["cross_cam_sim"] + 0.02
    assert min(same_id) > PARAMS["cross_cam_sim"] - 0.06
    # Cặp "mặc đồ giống nhau" cũng phải bám sát tham số inter_sim của kịch bản.
    assert max(diff_id) == pytest.approx(PARAMS["inter_sim"], abs=0.06)


def test_online_va_offline_cho_cung_bo_tracklet(scenario):
    """Chế độ online (đóng theo cửa sổ) phải ra cùng kết quả với offline (một lần flush)."""
    messages, _ = scenario
    config = TrackletConfig(min_frames=5, idle_timeout_ms=2_000)

    offline = build_tracklets(messages, config)

    builder = TrackletBuilder(config)
    online: list = []
    window_end = messages[0].ts_ms + 1_000
    for msg in messages:
        online.extend(builder.update(msg))
        if msg.ts_ms >= window_end:
            online.extend(builder.close_expired(msg.ts_ms))
            window_end = msg.ts_ms + 1_000
    online.extend(builder.flush())

    assert [(t.cam_id, t.local_track_id, t.n_frames) for t in sorted(online, key=_sort_key)] == [
        (t.cam_id, t.local_track_id, t.n_frames) for t in sorted(offline, key=_sort_key)
    ]


def _sort_key(t):
    return (t.cam_id, t.local_track_id, t.start_ms)
