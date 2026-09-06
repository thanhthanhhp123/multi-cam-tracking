"""Test hàng đợi có trần giữa probe DeepStream và Redis (`common.streams`).

Probe chạy trên luồng streaming của GStreamer, nên tính chất phải giữ bằng mọi giá là
**`publish()` không bao giờ chặn và không bao giờ ném lỗi**. Mọi test ở đây dùng publisher
GIẢ — không cần Redis, không cần GPU (CLAUDE.md §2 quy tắc 3).
"""

from __future__ import annotations

import threading
import time

import pytest

from common.schema import Detection, FrameMessage
from common.streams import QueuedFramePublisher


def _msg(frame_id: int) -> FrameMessage:
    return FrameMessage(
        cam_id="cam01",
        frame_id=frame_id,
        ts_ms=1_000 + frame_id * 40,
        frame_pts_ns=frame_id * 40_000_000,
        frame_width=1920,
        frame_height=1080,
        detections=[Detection(local_track_id=1, bbox=(0.0, 0.0, 10.0, 20.0), confidence=0.9)],
    )


class _FakePublisher:
    """Ghi lại message đã nhận; `gate` giữ luồng nền lại để mô phỏng Redis chậm."""

    stream = "mct:frames"

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.sent: list[FrameMessage] = []
        self.gate = gate
        self.closed = False
        self._lock = threading.Lock()

    def publish_many(self, messages) -> int:
        if self.gate is not None:
            self.gate.wait(timeout=5.0)
        batch = list(messages)
        with self._lock:
            self.sent.extend(batch)
        return len(batch)

    def close(self) -> None:
        self.closed = True


def test_publish_khong_chan_khi_redis_dung_hinh():
    """Tính chất sống còn: Redis treo thì probe vẫn phải trả về ngay."""
    gate = threading.Event()
    fake = _FakePublisher(gate)
    pub = QueuedFramePublisher(fake, maxsize=8, batch=4)

    t0 = time.perf_counter()
    for i in range(200):  # nhiều hơn hẳn trần hàng đợi
        pub.publish(_msg(i))
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, "publish() đã chặn luồng gọi"
    assert pub.n_dropped > 0  # có bỏ, và bỏ thì phải đếm

    gate.set()
    pub.close(timeout=5.0)


def test_hang_doi_thoang_thi_khong_bo_khung_nao():
    fake = _FakePublisher()
    pub = QueuedFramePublisher(fake, maxsize=256, batch=8)

    for i in range(100):
        assert pub.publish(_msg(i)) is True
    pub.close(timeout=5.0)

    assert pub.n_dropped == 0
    assert pub.n_published == 100
    assert [m.frame_id for m in fake.sent] == list(range(100))
    assert fake.closed is True


def test_day_thi_bo_khung_CU_nhat_va_giu_khung_moi():
    """Hệ thống trả lời 'người đó đang ở đâu' nên khung mới có giá trị hơn khung cũ."""
    gate = threading.Event()
    fake = _FakePublisher(gate)
    pub = QueuedFramePublisher(fake, maxsize=4, batch=1)

    for i in range(40):
        pub.publish(_msg(i))
    gate.set()
    pub.close(timeout=5.0)

    ids = [m.frame_id for m in fake.sent]
    assert ids, "không đẩy được khung nào"
    assert max(ids) >= 30, f"khung mới nhất bị bỏ thay vì khung cũ: {ids}"


def test_publisher_nem_loi_thi_luong_nen_van_song():
    """Một lô hỏng không được giết luồng nền — nếu chết thì mất im lặng cả luồng dữ liệu."""

    class _Flaky(_FakePublisher):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def publish_many(self, messages) -> int:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Redis rớt kết nối")
            return super().publish_many(messages)

    flaky = _Flaky()
    pub = QueuedFramePublisher(flaky, maxsize=64, batch=1)
    pub.publish(_msg(0))
    time.sleep(0.3)
    pub.publish(_msg(1))
    pub.close(timeout=5.0)

    assert pub.n_failed == 1
    assert [m.frame_id for m in flaky.sent] == [1]


@pytest.mark.parametrize("kwargs", [{"maxsize": 0}, {"batch": 0}])
def test_tham_so_vo_ly_bi_tu_choi(kwargs):
    with pytest.raises(ValueError):
        QueuedFramePublisher(_FakePublisher(), **kwargs)
