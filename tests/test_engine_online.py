"""Test cho vòng chạy online (`mct.__main__.Engine`).

Điểm cần ghim: engine chỉ dùng `ts_ms` trong message, KHÔNG dùng đồng hồ hệ thống — nhờ
vậy phát lại một fixture luôn cho ra đúng một kết quả, điều kiện cần để số liệu chương 6
tái lập được.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict

import numpy as np
import pytest

from common.schema import CLASS_PERSON, Detection, FrameMessage, l2_normalize
from mct.__main__ import Engine
from mct.affinity import AffinityConfig
from mct.associator import Associator
from mct.store import Store, StoreConfig
from mct.tracklet import TrackletConfig

DIM = 16
BASE_TS = 1_700_000_000_000


def _msg(cam_id: str, frame_id: int, tracks: dict[int, np.ndarray], *, step_ms: int = 100):
    return FrameMessage(
        cam_id=cam_id,
        frame_id=frame_id,
        ts_ms=BASE_TS + frame_id * step_ms,
        frame_pts_ns=frame_id * step_ms * 1_000_000,
        frame_width=1920,
        frame_height=1080,
        detections=[
            Detection(
                local_track_id=tid,
                bbox=(100.0 + frame_id, 200.0, 60.0, 150.0),
                confidence=0.9,
                embedding=emb,
                class_id=CLASS_PERSON,
            )
            for tid, emb in tracks.items()
        ],
        embed_dim=DIM,
    )


def _engine(**kwargs) -> Engine:
    return Engine(
        tracklet_config=TrackletConfig(min_frames=2, idle_timeout_ms=2_000),
        associator=Associator(config=AffinityConfig(max_cost=0.5)),
        window_ms=1_000,
        **kwargs,
    )


def test_hai_camera_cung_nguoi_nhan_cung_global_id():
    person = l2_normalize(np.ones(DIM, dtype=np.float32))
    engine = _engine()

    for frame in range(6):
        engine.feed(_msg("cam01", frame, {1: person}))
    for frame in range(6, 12):
        engine.feed(_msg("cam02", frame, {5: person}))
    assert engine.finish(), "vòng cuối phải phát ít nhất một cập nhật"

    tracks = engine.associator.gallery.open_tracks()
    assert len(tracks) == 1
    assert tracks[0].cameras == {"cam01", "cam02"}


def test_khong_dung_dong_ho_he_thong_ket_qua_lap_lai_duoc():
    person = l2_normalize(np.ones(DIM, dtype=np.float32))
    other = l2_normalize(np.array([1.0] + [-1.0] * (DIM - 1), dtype=np.float32))

    def run() -> list[tuple[str, int]]:
        engine = _engine()
        out = []
        for frame in range(20):
            out.extend(engine.feed(_msg("cam01", frame, {1: person, 2: other})))
        out.extend(engine.finish())
        return [(u.cam_id, u.global_id) for u in out]

    assert run() == run()


def test_cua_so_dong_theo_ts_ms_khong_theo_so_message():
    person = l2_normalize(np.ones(DIM, dtype=np.float32))
    engine = _engine()

    # 5 message trong 400 ms: chưa hết cửa sổ 1000 ms -> chưa vòng gán nào chạy.
    for frame in range(5):
        assert engine.feed(_msg("cam01", frame, {1: person})) == []
    assert engine.associator.stats.windows == 0

    # Message ở 1500 ms vượt mốc cửa sổ -> vòng gán chạy.
    engine.feed(_msg("cam01", 15, {1: person}))
    assert engine.associator.stats.windows == 1


def test_tracklet_ngan_hon_min_frames_khong_duoc_gan():
    person = l2_normalize(np.ones(DIM, dtype=np.float32))
    engine = Engine(
        tracklet_config=TrackletConfig(min_frames=10, idle_timeout_ms=2_000),
        associator=Associator(config=AffinityConfig(max_cost=0.5)),
        window_ms=1_000,
    )

    for frame in range(3):
        engine.feed(_msg("cam01", frame, {1: person}))
    engine.finish()

    assert engine.associator.gallery.open_tracks() == []


def test_ghi_xuong_sqlite_khop_voi_ket_qua_gan(tmp_path):
    person = l2_normalize(np.ones(DIM, dtype=np.float32))
    store = Store(StoreConfig(db_path=str(tmp_path / "engine.db"), batch_size=1))
    engine = _engine(store=store)

    for frame in range(6):
        engine.feed(_msg("cam01", frame, {1: person}))
    for frame in range(6, 12):
        engine.feed(_msg("cam02", frame, {5: person}))
    engine.finish()

    summary = store.summary()
    assert summary["n_cross_camera"] == 1
    assert summary["n_cameras"] == 2
    gid = engine.associator.gallery.open_tracks()[0].global_id
    assert [a.cam_id for a in store.trajectory(gid)] == ["cam01", "cam02"]
    store.close()


def test_global_update_mang_du_thong_tin_cho_dashboard():
    person = l2_normalize(np.ones(DIM, dtype=np.float32))
    engine = _engine()

    for frame in range(6):
        engine.feed(_msg("cam01", frame, {1: person}))
    updates = engine.finish()

    update = updates[-1]
    assert update.cam_id == "cam01"
    assert update.local_track_id == 1
    assert update.bbox[2:] == (60.0, 150.0)
    assert update.ts_ms >= BASE_TS
    assert update.n_cameras == 1
    # ground_point = đáy-giữa bbox theo contract common/schema.py
    assert update.ground_point == pytest.approx((update.bbox[0] + 30.0, 350.0))


# --------------------------------------------------------------------------- đường phát vị trí


def _walk() -> list[FrameMessage]:
    """Hai người ở cam01 rồi một người sang cam02, 100 ms/khung — đủ để có cả gán mới lẫn
    ghép xuyên camera, và nhiều khung sau khi đã có Global ID."""
    person = l2_normalize(np.ones(DIM, dtype=np.float32))
    other = l2_normalize(np.array([1.0] + [-1.0] * (DIM - 1), dtype=np.float32))
    msgs = [_msg("cam01", frame, {1: person, 2: other}) for frame in range(25)]
    msgs += [_msg("cam02", frame, {5: person}) for frame in range(25, 50)]
    return msgs


def _run_walk(tmp_path, name: str, position_interval_ms: int | None):
    store = Store(StoreConfig(db_path=str(tmp_path / f"{name}.db"), batch_size=1))
    windows: list[tuple[int, list[tuple[int, int, bool, bool]]]] = []
    engine = _engine(
        store=store,
        position_interval_ms=position_interval_ms,
        window_observer=lambda now, results: windows.append(
            (now, [(r.tracklet.tracklet_id, r.global_id, r.is_new, r.is_update) for r in results])
        ),
    )
    updates = []
    for msg in _walk():
        updates.append((int(msg.ts_ms), engine.feed(msg)))
    updates.append((None, engine.finish()))
    store.close()
    with sqlite3.connect(tmp_path / f"{name}.db") as conn:
        tables = {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall()
            for table in ("global_tracks", "appearances")
        }
    return engine, windows, updates, tables


@pytest.mark.parametrize("interval", [0, 100, 250])
def test_duong_phat_vi_tri_khong_doi_ket_qua_lien_ket(tmp_path, interval):
    """Điều kiện để bật mặc định: chỉ ĐỌC gallery, nên mọi vòng gán, SQLite và gallery cuối
    phải y hệt khi tắt."""
    off, windows_off, _, tables_off = _run_walk(tmp_path, "off", None)
    on, windows_on, _, tables_on = _run_walk(tmp_path, f"on{interval}", interval)

    assert windows_on == windows_off
    assert tables_on == tables_off
    assert [(t.global_id, sorted(t.cameras)) for t in on.associator.gallery.open_tracks()] == [
        (t.global_id, sorted(t.cameras)) for t in off.associator.gallery.open_tracks()
    ]
    assert off.n_position_updates == 0 and on.n_position_updates > 0


def _publish_times(updates, tracklet_id: int, *, until_ms: int) -> list[int]:
    return [
        ts - BASE_TS
        for ts, batch in updates
        if ts is not None and ts < until_ms
        for update in batch
        if update.tracklet_id == tracklet_id
    ]


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        (None, [1000, 2000]),
        (250, [1000, 1300, 1600, 1900, 2000, 2300]),
        (100, list(range(1000, 2500, 100))),
        (0, list(range(1000, 2500, 100))),
    ],
)
def test_nhip_phat_cua_nguoi_da_co_global_id(tmp_path, interval, expected):
    """cam01, 100 ms/khung, cửa sổ 1000 ms: vòng gán chạy ở 1000 và 2000 ms.

    - Tắt (`None`): chỉ phát khi cửa sổ đóng, như trước phiên 32.
    - Bật: từ vòng gán đầu, phát ngay khi khung tới, tối đa một lần mỗi khoảng. Lần phát
      của vòng gán cũng tính vào nhịp (2000 rồi mới tới 2300 ở khoảng 250).
    - Trước vòng gán đầu (0–900 ms) chưa ai có Global ID nên không có gì để phát.
    """
    _, _, updates, _ = _run_walk(tmp_path, "run", interval)
    assert _publish_times(updates, 1, until_ms=BASE_TS + 2500) == expected


def test_vi_tri_la_cua_khung_vua_toi_va_global_id_khong_doi(tmp_path):
    _, _, updates, _ = _run_walk(tmp_path, "run", 0)
    ids: dict[int, set[int]] = defaultdict(set)
    for ts, batch in updates:
        for update in batch:
            ids[update.tracklet_id].add(update.global_id)
            if ts is not None and ts < BASE_TS + 2500:
                assert update.ts_ms == ts, "vị trí phát ngay phải là của chính khung vừa tới"
    assert ids and all(len(gids) == 1 for gids in ids.values())
