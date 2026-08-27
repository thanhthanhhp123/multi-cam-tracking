"""Test wrapper Redis Streams — cần một instance Redis đang chạy (`make up`).

Tự bỏ qua khi không kết nối được, để `make test` trên máy chưa bật Docker vẫn chạy trọn.
Chạy riêng nhóm này: pytest -m redis
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from common.schema import Detection, FrameMessage, l2_normalize
from common.streams import FrameConsumer, FramePublisher, connect

pytestmark = pytest.mark.redis

DIM = 16


def _redis_available() -> bool:
    try:
        client = connect()
        client.ping()
        client.close()
        return True
    except Exception:
        return False


requires_redis = pytest.mark.skipif(
    not _redis_available(), reason="cần Redis đang chạy — bật bằng `make up`"
)


@pytest.fixture
def temp_stream():
    """Stream riêng cho mỗi test, dọn sạch sau khi xong."""
    name = f"mct:test:{uuid.uuid4().hex[:8]}"
    yield name
    client = connect()
    client.delete(name)
    client.close()


def make_message(frame_id: int, cam_id: str = "cam01", n_det: int = 2) -> FrameMessage:
    rng = np.random.default_rng(frame_id)
    return FrameMessage(
        cam_id=cam_id,
        frame_id=frame_id,
        ts_ms=1_788_231_600_000 + frame_id * 66,
        frame_pts_ns=frame_id * 66_666_666,
        frame_width=1920,
        frame_height=1080,
        detections=[
            Detection(
                local_track_id=i + 1,
                bbox=(100.0 * (i + 1), 200.0, 80.0, 220.0),
                confidence=0.9,
                embedding=l2_normalize(rng.standard_normal(DIM)),
            )
            for i in range(n_det)
        ],
        embed_dim=DIM,
    )


@requires_redis
def test_publish_roi_consume_giu_nguyen_du_lieu(temp_stream) -> None:
    sent = [make_message(i) for i in range(5)]

    with FramePublisher(stream=temp_stream) as pub:
        for msg in sent:
            pub.publish(msg)

    with FrameConsumer(stream=temp_stream, group="test-group", consumer="c1") as con:
        got = con.read(count=10, block_ms=500)

    assert [m.frame_id for _, m in got] == [0, 1, 2, 3, 4]
    for original, (_entry_id, received) in zip(sent, got, strict=True):
        assert received.cam_id == original.cam_id
        assert received.ts_ms == original.ts_ms
        np.testing.assert_array_equal(
            received.detections[0].embedding, original.detections[0].embedding
        )


@requires_redis
def test_message_chua_ack_van_doc_lai_duoc(temp_stream) -> None:
    """Consumer chết giữa chừng thì message chưa ack phải còn nguyên — đây là lý do
    chọn Redis Streams thay vì ZeroMQ."""
    with FramePublisher(stream=temp_stream) as pub:
        pub.publish(make_message(1))

    with FrameConsumer(stream=temp_stream, group="g", consumer="c1") as con:
        first = con.read(count=10, block_ms=500)
        assert len(first) == 1
        assert con.pending_count() == 1
        # Không ack, giả lập crash.

    with FrameConsumer(stream=temp_stream, group="g", consumer="c1") as con:
        recovered = con.read_pending(count=10)
        assert [m.frame_id for _, m in recovered] == [1]

        assert con.ack([entry_id for entry_id, _ in recovered]) == 1
        assert con.pending_count() == 0


@requires_redis
def test_consumer_tao_duoc_group_truoc_khi_co_message(temp_stream) -> None:
    """mkstream=True cho phép bật engine trước cả pipeline."""
    with FrameConsumer(stream=temp_stream, group="g", consumer="c1") as con:
        assert con.read(count=1, block_ms=100) == []

        with FramePublisher(stream=temp_stream) as pub:
            pub.publish(make_message(42))

        assert [m.frame_id for _, m in con.read(count=1, block_ms=500)] == [42]


@requires_redis
def test_tao_group_hai_lan_khong_loi(temp_stream) -> None:
    with FrameConsumer(stream=temp_stream, group="g", consumer="c1"):
        pass
    with FrameConsumer(stream=temp_stream, group="g", consumer="c2"):
        pass


@requires_redis
def test_nhieu_camera_tren_cung_mot_stream(temp_stream) -> None:
    with FramePublisher(stream=temp_stream) as pub:
        pub.publish(make_message(1, cam_id="cam01"))
        pub.publish(make_message(1, cam_id="cam02"))

    with FrameConsumer(stream=temp_stream, group="g", consumer="c1") as con:
        got = con.read(count=10, block_ms=500)

    assert {m.cam_id for _, m in got} == {"cam01", "cam02"}
