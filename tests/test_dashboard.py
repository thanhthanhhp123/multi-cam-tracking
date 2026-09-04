"""Test dashboard: API, trạng thái live, fan-out WebSocket.

Không cần Redis cũng không cần GPU. `TestClient` chỉ chạy vòng đời (lifespan) khi dùng
làm context manager, nên các test dưới đây gọi thẳng `client.get(...)` để `RedisBridge`
không khởi động — phần đó được test riêng bằng cách bơm `GlobalUpdate` vào `LiveState`.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from common.schema import GlobalUpdate
from dashboard.app import Dashboard, create_app
from dashboard.live import Hub, LiveState
from mct.associator import Assignment
from mct.homography import CameraHomography, HomographyMapper
from mct.store import Store, StoreConfig
from mct.tracklet import Observation, Tracklet

# Homography giả lập một camera nhìn chéo xuống mặt phẳng (hàng cuối khác 0 0 1).
H = np.array(
    [
        [0.0042, 0.0009, -3.15],
        [0.0006, 0.0130, -8.40],
        [0.00002, 0.00060, 1.0],
    ],
    dtype=np.float64,
)


def _mapper() -> HomographyMapper:
    other = H.copy()
    other[0, 2] += 4.0
    return HomographyMapper(
        {
            "cam01": CameraHomography("cam01", H, image_size=(1920, 1080)),
            "cam02": CameraHomography("cam02", other, image_size=(1920, 1080)),
        }
    )


def _update(gid: int, cam_id: str, ts_ms: int, point=(960.0, 900.0), n_cameras: int = 1):
    return GlobalUpdate(
        global_id=gid,
        cam_id=cam_id,
        local_track_id=1,
        tracklet_id=gid * 10,
        ts_ms=ts_ms,
        bbox=(point[0] - 30.0, point[1] - 150.0, 60.0, 150.0),
        ground_point=point,
        n_cameras=n_cameras,
    )


# --------------------------------------------------------------------- LiveState


def test_quy_vi_tri_ve_met_ngay_tai_server():
    state = LiveState(mapper=_mapper())

    track = state.apply(_update(1, "cam01", 1_000))

    assert track.x_m is not None and track.y_m is not None
    assert track.trail and len(track.trail) == 1


def test_camera_chua_hieu_chinh_thi_khong_co_toa_do():
    """Không đoán bừa vị trí: giao diện xếp track đó ra danh sách riêng thay vì vẽ sai."""
    state = LiveState(mapper=_mapper())

    track = state.apply(_update(2, "cam09", 1_000))

    assert track.x_m is None
    assert list(track.trail) == []


def test_track_im_lang_qua_ttl_thi_bien_mat():
    state = LiveState(mapper=_mapper(), ttl_ms=5_000)
    state.apply(_update(1, "cam01", 1_000))
    state.apply(_update(2, "cam01", 20_000))

    gone = state.expire()

    assert gone == [1]
    assert set(state.tracks) == {2}


def test_moc_thoi_gian_lay_tu_du_lieu_khong_phai_dong_ho_server():
    """Phát lại fixture cũ (ts_ms năm ngoái) vẫn phải hiện đúng, không bị hết hạn ngay."""
    state = LiveState(mapper=_mapper(), ttl_ms=5_000)

    state.apply(_update(1, "cam01", 1_600_000_000_000))

    assert state.expire() == []


def test_snapshot_du_cho_client_moi_ket_noi():
    state = LiveState(mapper=_mapper())
    state.apply(_update(1, "cam01", 1_000))
    state.apply(_update(1, "cam02", 2_000, n_cameras=2))

    snapshot = state.snapshot()

    assert snapshot["type"] == "snapshot"
    assert len(snapshot["tracks"]) == 1
    entry = snapshot["tracks"][0]
    assert entry["cameras"] == ["cam01", "cam02"]
    assert entry["n_cameras"] == 2
    assert len(entry["trail"]) == 2


# --------------------------------------------------------------------------- Hub


class _FakeSocket:
    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[dict] = []
        self.fail = fail

    async def send_json(self, payload: dict) -> None:
        if self.fail:
            raise RuntimeError("client đã đóng")
        self.sent.append(payload)


def test_hub_gui_cho_moi_client():
    hub = Hub()
    a, b = _FakeSocket(), _FakeSocket()
    hub.add(a)
    hub.add(b)

    sent = asyncio.run(hub.broadcast({"type": "update"}))

    assert sent == 2
    assert a.sent == b.sent == [{"type": "update"}]


def test_client_chet_bi_loai_chu_khong_lam_hong_vong_phat():
    hub = Hub()
    good, dead = _FakeSocket(), _FakeSocket(fail=True)
    hub.add(good)
    hub.add(dead)

    sent = asyncio.run(hub.broadcast({"type": "update"}))

    assert sent == 1
    assert hub.n_clients == 1


# --------------------------------------------------------------------------- API


@pytest.fixture
def db(tmp_path):
    """SQLite có sẵn một Global ID đi qua hai camera."""
    path = tmp_path / "mct.db"
    with Store(StoreConfig(db_path=str(path), batch_size=1)) as store:
        for tid, (cam, start) in enumerate(
            [("cam01", 1_000), ("cam02", 9_000), ("cam01", 30_000)], start=1
        ):
            tracklet = Tracklet(tracklet_id=tid, cam_id=cam, local_track_id=tid)
            for i in range(4):
                tracklet.add(
                    Observation(
                        ts_ms=start + 250 * i,
                        frame_id=i,
                        bbox=(10.0, 20.0, 40.0, 90.0),
                        confidence=0.8,
                        ground_point=(30.0, 110.0),
                    ),
                    max_embeddings=8,
                )
            gid = 7 if cam != "cam01" or tid != 3 else 8
            store.record(Assignment(tracklet=tracklet, global_id=gid, cost=0.1, is_new=False))
    return path


@pytest.fixture
def client(db):
    return TestClient(create_app(Dashboard(db_path=str(db), topology_path=None)))


def test_trang_chu_tra_ve_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Multi-Camera Tracking" in res.text


def test_so_do_camera_co_vung_phu_va_khung_nhin(tmp_path):
    mapper_dir = tmp_path / "hom"
    _mapper().save(mapper_dir)
    app = create_app(
        Dashboard(db_path=str(tmp_path / "x.db"), topology_path=None, homography_dir=mapper_dir)
    )

    data = TestClient(app).get("/api/layout").json()

    assert [c["cam_id"] for c in data["cameras"]] == ["cam01", "cam02"]
    assert all(len(c["footprint"]) >= 3 for c in data["cameras"])
    bounds = data["bounds"]
    assert bounds["min_x"] < bounds["max_x"] and bounds["min_y"] < bounds["max_y"]


def test_chua_hieu_chinh_thi_khong_co_khung_nhin(client):
    data = client.get("/api/layout").json()
    assert data["bounds"] is None


def test_danh_sach_track_tu_sqlite(client):
    data = client.get("/api/tracks?min_cameras=2").json()

    assert data["summary"]["n_appearances"] == 3
    assert [row["global_id"] for row in data["tracks"]] == [7]
    assert data["tracks"][0]["n_cameras"] == 2


def test_hanh_trinh_theo_global_id(client):
    trip = client.get("/api/tracks/7").json()

    assert trip["n_cameras"] == 2
    assert [a["cam_id"] for a in trip["appearances"]] == ["cam01", "cam02"]
    assert trip["appearances"][0]["duration_ms"] == 750
    assert trip["start_ms"] < trip["end_ms"]


def test_global_id_khong_ton_tai_tra_404(client):
    assert client.get("/api/tracks/999").status_code == 404


def test_chua_co_file_sqlite_thi_bao_503_chu_khong_bao_rong(tmp_path):
    """Bảng rỗng gây hiểu nhầm "chưa ai đi qua"; 503 nói thẳng là engine chưa chạy."""
    app = create_app(Dashboard(db_path=str(tmp_path / "chua-co.db"), topology_path=None))

    res = TestClient(app).get("/api/tracks")

    assert res.status_code == 503
    assert "chưa" in res.json()["detail"].lower()


def test_api_live_tra_ve_snapshot(client):
    data = client.get("/api/live").json()
    assert data["type"] == "snapshot"
    assert data["tracks"] == []


def test_health_khong_chet_khi_thieu_redis(client):
    data = client.get("/health").json()
    assert data["status"] in ("ok", "degraded")
    assert data["db"]["exists"] is True


def test_store_chi_doc_tu_choi_ghi(db):
    store = Store.open_readonly(db)
    try:
        with pytest.raises(RuntimeError, match="chỉ-đọc"):
            store.record(
                Assignment(
                    tracklet=Tracklet(tracklet_id=99, cam_id="cam01", local_track_id=9),
                    global_id=1,
                    cost=0.0,
                    is_new=True,
                )
            )
    finally:
        store.close()


def test_websocket_gui_snapshot_ngay_khi_ket_noi(db):
    """Client mở trang lúc vắng người vẫn phải thấy toàn cảnh, không phải chờ ai đi qua."""
    board = Dashboard(db_path=str(db), topology_path=None)
    board.state.mapper = _mapper()
    board.state.apply(_update(3, "cam01", 5_000))
    client = TestClient(create_app(board))

    with client.websocket_connect("/ws/live") as ws:
        message = ws.receive_json()

    assert message["type"] == "snapshot"
    assert [t["global_id"] for t in message["tracks"]] == [3]


def test_client_ngat_ket_noi_thi_bi_go_khoi_hub(db):
    board = Dashboard(db_path=str(db), topology_path=None)
    client = TestClient(create_app(board))

    with client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        assert board.hub.n_clients == 1

    assert board.hub.n_clients == 0


# ------------------------------------------------------------------ RedisBridge


class _FakeRedis:
    """Client Redis giả: trả một lô entry rồi im. Đủ để chạy đúng đường `read_global`."""

    def __init__(self, batches: list[list[GlobalUpdate]]) -> None:
        self.batches = list(batches)
        self.pings = 0
        self.closed = False
        self.seen_last_id: list[str] = []

    def ping(self) -> bool:
        self.pings += 1
        return True

    def xread(self, streams, count=None, block=None):
        from common.schema import encode_global_msgpack

        self.seen_last_id.append(next(iter(streams.values())))
        if not self.batches:
            return []
        batch = self.batches.pop(0)
        entries = [
            (f"{i}-0".encode(), {b"data": encode_global_msgpack(u)})
            for i, u in enumerate(batch, start=1)
        ]
        return [(b"mct:global", entries)]

    def close(self) -> None:
        self.closed = True


async def _drain(bridge, hub_socket, *, ticks: int = 4) -> None:
    await bridge.start()
    for _ in range(ticks):
        await asyncio.sleep(0)
        await asyncio.sleep(0.01)
    await bridge.stop()


def test_bridge_doc_redis_cap_nhat_state_va_phat_cho_client(monkeypatch):
    import common.streams as streams

    fake = _FakeRedis([[_update(1, "cam01", 1_000), _update(2, "cam02", 1_100)]])
    monkeypatch.setattr(streams, "connect", lambda url=None: fake)

    state = LiveState(mapper=_mapper())
    hub = Hub()
    socket = _FakeSocket()
    hub.add(socket)
    from dashboard.live import RedisBridge

    bridge = RedisBridge(state, hub, block_ms=1)
    asyncio.run(_drain(bridge, socket))

    assert fake.pings == 1
    assert set(state.tracks) == {1, 2}
    assert any(msg["type"] == "update" for msg in socket.sent)
    # Lần đọc đầu từ "$" (chỉ lấy cái mới), các lần sau nối tiếp từ entry id cuối.
    assert fake.seen_last_id[0] == "$"
    assert fake.seen_last_id[1] == "2-0"


def test_bridge_mat_redis_thi_bao_loi_chu_khong_chet(monkeypatch):
    import common.streams as streams

    def _boom(url=None):
        raise ConnectionError("Connection refused")

    monkeypatch.setattr(streams, "connect", _boom)
    from dashboard.live import RedisBridge

    bridge = RedisBridge(LiveState(), Hub(), block_ms=1)
    asyncio.run(_drain(bridge, None, ticks=2))

    assert bridge.connected is False
    assert "refused" in (bridge.last_error or "")
