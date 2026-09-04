"""Test cho `mct.store`: ghi theo lô, truy vấn hành trình, và bất biến "một nguồn sự thật"."""

from __future__ import annotations

import pytest

from mct.associator import Assignment
from mct.store import Appearance, Store, StoreConfig
from mct.tracklet import Observation, Tracklet


def _tracklet(cam_id: str, local_id: int, *, tracklet_id: int, start_ms: int, n: int = 5):
    tracklet = Tracklet(tracklet_id=tracklet_id, cam_id=cam_id, local_track_id=local_id)
    for i in range(n):
        tracklet.add(
            Observation(
                ts_ms=start_ms + 100 * i,
                frame_id=i,
                bbox=(10.0 + i, 20.0, 40.0, 90.0),
                confidence=0.8,
                ground_point=(30.0 + i, 110.0),
            ),
            max_embeddings=8,
        )
    return tracklet


def _assign(track, global_id: int, *, cost: float = 0.1, is_new: bool = False) -> Assignment:
    return Assignment(tracklet=track, global_id=global_id, cost=cost, is_new=is_new)


@pytest.fixture
def store(tmp_path):
    with Store(StoreConfig(db_path=str(tmp_path / "mct.db"), batch_size=2)) as st:
        yield st


def test_ghi_va_doc_lai_mot_hanh_trinh(store):
    store.record(_assign(_tracklet("cam01", 1, tracklet_id=1, start_ms=1_000), 7))
    store.record(_assign(_tracklet("cam02", 3, tracklet_id=2, start_ms=9_000), 7))
    store.flush()

    trip = store.trajectory(7)

    assert [a.cam_id for a in trip] == ["cam01", "cam02"]
    assert isinstance(trip[0], Appearance)
    assert trip[0].duration_ms == 400
    assert store.summary()["n_cross_camera"] == 1


def test_gan_lai_cung_tracklet_khong_de_them_dong(store):
    """Tracklet dài được gán lại ở nhiều cửa sổ — bảng `appearances` chỉ được nới end_ms."""
    first = _tracklet("cam01", 1, tracklet_id=1, start_ms=1_000, n=3)
    store.record(_assign(first, 1))
    store.flush()

    longer = _tracklet("cam01", 1, tracklet_id=1, start_ms=1_000, n=9)
    store.record(_assign(longer, 1))
    store.flush()

    trip = store.trajectory(1)
    assert len(trip) == 1
    assert trip[0].end_ms == longer.end_ms
    assert trip[0].n_frames == 9


def test_dem_camera_suy_ra_tu_bang_appearances(store):
    for idx, cam in enumerate(("cam01", "cam02", "cam03", "cam01"), start=1):
        store.record(_assign(_tracklet(cam, idx, tracklet_id=idx, start_ms=1_000 * idx), 5))
    store.flush()

    row = store.conn.execute("SELECT * FROM global_tracks WHERE global_id = 5").fetchone()

    assert row["n_tracklets"] == 4
    assert row["n_cameras"] == 3  # cam01 xuất hiện hai lần nhưng vẫn là một camera
    assert row["created_ms"] == 1_000
    assert row["last_cam_id"] == "cam01"  # lượt muộn nhất (t=4000)


def test_flush_tu_dong_khi_du_lo(tmp_path):
    with Store(StoreConfig(db_path=str(tmp_path / "b.db"), batch_size=2)) as st:
        st.record(_assign(_tracklet("cam01", 1, tracklet_id=1, start_ms=0), 1))
        assert st.n_written == 0
        st.record(_assign(_tracklet("cam01", 2, tracklet_id=2, start_ms=0), 2))
        assert st.n_written == 2


def test_cost_vo_cuc_ghi_thanh_am_mot(store):
    """Global ID mới có cost = inf; SQLite không có kiểu đó và 0.0 dễ đọc nhầm."""
    store.record(_assign(_tracklet("cam01", 1, tracklet_id=1, start_ms=0), 1, cost=float("inf")))
    store.flush()

    assert store.trajectory(1)[0].cost == -1.0


def test_dong_track_va_loc_theo_thoi_gian(store):
    store.record(_assign(_tracklet("cam01", 1, tracklet_id=1, start_ms=1_000), 1))
    store.record(_assign(_tracklet("cam02", 2, tracklet_id=2, start_ms=50_000), 2))
    store.flush()

    assert [r["global_id"] for r in store.active_tracks(since_ms=40_000)] == [2]
    store.close_tracks([2])
    assert store.active_tracks(since_ms=0) == [
        r for r in store.active_tracks(since_ms=0) if r["global_id"] == 1
    ]


def test_loc_theo_khoang_thoi_gian(store):
    store.record(_assign(_tracklet("cam01", 1, tracklet_id=1, start_ms=1_000), 1))
    store.record(_assign(_tracklet("cam02", 2, tracklet_id=2, start_ms=80_000), 2))
    store.flush()

    hits = store.appearances_between(0, 5_000)

    assert [a.global_id for a in hits] == [1]


def test_chi_lay_track_xuyen_camera(store):
    store.record(_assign(_tracklet("cam01", 1, tracklet_id=1, start_ms=0), 1))
    store.record(_assign(_tracklet("cam01", 2, tracklet_id=2, start_ms=9_000), 1))
    store.record(_assign(_tracklet("cam01", 3, tracklet_id=3, start_ms=0), 2))
    store.record(_assign(_tracklet("cam02", 4, tracklet_id=4, start_ms=9_000), 2))
    store.flush()

    assert [r["global_id"] for r in store.cross_camera_tracks()] == [2]


def test_batch_size_khong_hop_le_bi_tu_choi():
    with pytest.raises(ValueError, match="batch_size"):
        StoreConfig(batch_size=0)


def test_doc_config_tu_mct_yaml():
    config = StoreConfig.from_mapping(
        {"store": {"db_path": "/tmp/x.db", "batch_size": 8, "flush_interval_ms": 250}}
    )
    assert (config.db_path, config.batch_size, config.flush_interval_ms) == ("/tmp/x.db", 8, 250)
