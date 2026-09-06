"""Wrapper Redis Streams — ranh giới giữa pipeline DeepStream và engine liên kết.

Chọn Redis Streams thay vì Kafka/ZeroMQ vì nó có persistence: ghi lại được luồng
metadata thật từ máy GPU rồi phát lại trên máy không GPU để phát triển engine liên kết
(CLAUDE.md §3, ADR trong docs/worklog/2026-08-27).
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable
from types import TracebackType
from typing import Any

import redis
from redis.exceptions import ResponseError

from common.config import redis_url
from common.logging import get_logger
from common.schema import (
    FrameMessage,
    GlobalUpdate,
    decode_global_msgpack,
    decode_msgpack,
    encode_global_msgpack,
    encode_msgpack,
)

log = get_logger(__name__)

STREAM_FRAMES = "mct:frames"
"""Metadata từ pipeline: bbox + local track ID + embedding."""

STREAM_GLOBAL = "mct:global"
"""Cập nhật Global ID cho dashboard."""

GROUP_ENGINE = "mct-engine"

# Giữ khoảng vài phút dữ liệu ở 4 camera x 15fps, đủ để replay khi consumer chết.
MAXLEN_FRAMES = 100_000
MAXLEN_GLOBAL = 10_000

_FIELD = b"data"


def connect(url: str | None = None) -> redis.Redis:
    """Client Redis dùng bytes thô — không decode_responses, vì payload là msgpack."""
    return redis.Redis.from_url(url or redis_url(), decode_responses=False)


class FramePublisher:
    """Đẩy FrameMessage lên stream. Dùng bởi probe của DeepStream và tool replay."""

    def __init__(
        self,
        url: str | None = None,
        *,
        stream: str = STREAM_FRAMES,
        maxlen: int = MAXLEN_FRAMES,
        client: redis.Redis | None = None,
    ) -> None:
        self.stream = stream
        self.maxlen = maxlen
        self._client = client or connect(url)
        self._owns_client = client is None

    def publish(self, msg: FrameMessage) -> str:
        # approximate=True cho phép Redis trim theo lô, rẻ hơn nhiều so với trim chính xác.
        entry_id = self._client.xadd(
            self.stream,
            {_FIELD: encode_msgpack(msg)},
            maxlen=self.maxlen,
            approximate=True,
        )
        return entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)

    def publish_many(self, messages: Iterable[FrameMessage]) -> int:
        """Gửi cả lô bằng một pipeline — một vòng mạng thay vì một vòng cho mỗi khung."""
        pipe = self._client.pipeline(transaction=False)
        count = 0
        for msg in messages:
            pipe.xadd(
                self.stream,
                {_FIELD: encode_msgpack(msg)},
                maxlen=self.maxlen,
                approximate=True,
            )
            count += 1
        if count:
            pipe.execute()
        return count

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> FramePublisher:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class QueuedFramePublisher:
    """Đẩy FrameMessage lên Redis từ MỘT LUỒNG NỀN, qua hàng đợi có trần.

    **Vấn đề nó giải quyết.** Probe của DeepStream chạy trên luồng streaming của GStreamer:
    mọi thứ làm trong đó đều nằm chắn ngang đường đi của khung hình. Bản trước gọi thẳng
    `FramePublisher.publish()` trong probe, tức mỗi khung của mỗi camera là một vòng gọi
    mạng đồng bộ; Redis khựng một nhịp (đang lưu RDB, mạng nghẽn) là cả pipeline suy luận
    khựng theo, và pipeline lại không có phần tử `queue` nào để hấp thụ.

    **Vì sao hàng đợi phải CÓ TRẦN.** Hàng đợi không trần chỉ đổi một chỗ hỏng ồn ào (khung
    hình đứng) lấy một chỗ hỏng im lặng (RAM phình tới khi bị kill). Có trần thì khi bên
    tiêu thụ không theo kịp, ta **biết** mình mất gì và mất bao nhiêu.

    **Đầy thì bỏ cái CŨ nhất.** Hệ thống này trả lời "người đó đang ở đâu", nên khung mới
    có giá trị hơn khung cũ; đã tụt lại thì giữ dữ liệu tươi mới là đúng. Mất một khung chỉ
    tạo một lỗ trong quỹ đạo, thứ mà `TrackletBuilder` vốn đã phải chịu được (detector bỏ
    sót 44.9% ở mức đo hiện tại).

    Số khung rơi được ĐẾM và log định kỳ, không log từng cái — log mỗi lần rơi thì chính
    nó thành chỗ nghẽn mới.
    """

    def __init__(
        self,
        publisher: FramePublisher | None = None,
        *,
        url: str | None = None,
        maxsize: int = 2000,
        batch: int = 32,
        log_every: int = 500,
    ) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize phải >= 1, nhận {maxsize}")
        if batch < 1:
            raise ValueError(f"batch phải >= 1, nhận {batch}")
        self.publisher = publisher or FramePublisher(url)
        self.batch = int(batch)
        self.log_every = int(log_every)
        self._queue: queue.Queue[FrameMessage] = queue.Queue(maxsize=int(maxsize))
        self._stop = threading.Event()
        self.n_published = 0
        self.n_dropped = 0
        self.n_failed = 0
        self.max_depth = 0
        self._thread = threading.Thread(
            target=self._run, name="frame-publisher", daemon=True
        )
        self._thread.start()

    @property
    def stream(self) -> str:
        return self.publisher.stream

    def publish(self, msg: FrameMessage) -> bool:
        """Xếp hàng, KHÔNG BAO GIỜ chặn và không bao giờ ném lỗi. True = đã nhận."""
        depth = self._queue.qsize()
        if depth > self.max_depth:
            self.max_depth = depth
        try:
            self._queue.put_nowait(msg)
            return True
        except queue.Full:
            pass

        # Đầy: nhả một chỗ bằng cách bỏ cái cũ nhất rồi thử lại đúng một lần. Thất bại lần
        # nữa nghĩa là luồng nền vừa lấy mất chỗ đó — bỏ khung này, không quay vòng.
        try:
            self._queue.get_nowait()
            self.n_dropped += 1
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(msg)
            return True
        except queue.Full:
            self.n_dropped += 1
            if self.n_dropped % self.log_every == 0:
                log.warning(
                    "hàng đợi Redis đầy: đã bỏ %d khung (trần %d). Engine hoặc Redis "
                    "không theo kịp pipeline.",
                    self.n_dropped,
                    self._queue.maxsize,
                )
            return False

    def _run(self) -> None:
        while True:
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            chunk = [first]
            while len(chunk) < self.batch:
                try:
                    chunk.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            try:
                self.n_published += self.publisher.publish_many(chunk)
            # Bắt rộng là CÓ CHỦ Ý: luồng nền chết nghĩa là mất im lặng cả luồng dữ liệu,
            # tệ hơn nhiều so với bỏ một lô và chạy tiếp.
            except Exception:
                self.n_failed += len(chunk)
                log.exception("đẩy %d khung lên Redis thất bại, bỏ lô này", len(chunk))

    def close(self, timeout: float = 5.0) -> None:
        """Vét nốt hàng đợi rồi đóng. Quá `timeout` thì bỏ phần còn lại và ghi log."""
        self._stop.set()
        self._thread.join(timeout=timeout)
        remaining = self._queue.qsize()
        if remaining:
            log.warning("còn %d khung chưa đẩy được khi đóng", remaining)
        log.info(
            "publisher: %d khung đã đẩy, %d bỏ vì hàng đợi đầy, %d lỗi, sâu nhất %d",
            self.n_published,
            self.n_dropped,
            self.n_failed,
            self.max_depth,
        )
        self.publisher.close()

    def __enter__(self) -> QueuedFramePublisher:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class FrameConsumer:
    """Đọc FrameMessage qua consumer group (ack thủ công, không mất message khi crash)."""

    def __init__(
        self,
        url: str | None = None,
        *,
        stream: str = STREAM_FRAMES,
        group: str = GROUP_ENGINE,
        consumer: str = "engine-1",
        start_id: str = "0",
        client: redis.Redis | None = None,
    ) -> None:
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self._client = client or connect(url)
        self._owns_client = client is None
        self._ensure_group(start_id)

    def _ensure_group(self, start_id: str) -> None:
        try:
            # mkstream=True để chạy được engine trước cả khi pipeline gửi message đầu tiên.
            self._client.xgroup_create(self.stream, self.group, id=start_id, mkstream=True)
            log.info("Đã tạo consumer group %r trên stream %r", self.group, self.stream)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def read(self, *, count: int = 64, block_ms: int = 1000) -> list[tuple[str, FrameMessage]]:
        """Đọc message mới. Trả về list rỗng khi hết thời gian chờ."""
        response = self._client.xreadgroup(
            self.group, self.consumer, {self.stream: ">"}, count=count, block=block_ms
        )
        return self._parse(response)

    def read_pending(self, *, count: int = 64) -> list[tuple[str, FrameMessage]]:
        """Đọc lại message đã giao cho consumer này nhưng chưa ack (khôi phục sau crash)."""
        response = self._client.xreadgroup(
            self.group, self.consumer, {self.stream: "0"}, count=count
        )
        return self._parse(response)

    def _parse(self, response: Any) -> list[tuple[str, FrameMessage]]:
        out: list[tuple[str, FrameMessage]] = []
        for _stream, entries in response or []:
            for entry_id, fields in entries:
                raw = fields.get(_FIELD)
                if raw is None:
                    log.warning("Entry %s thiếu field %r, bỏ qua", entry_id, _FIELD)
                    continue
                out.append((entry_id.decode(), decode_msgpack(raw)))
        return out

    def ack(self, entry_ids: list[str]) -> int:
        if not entry_ids:
            return 0
        return int(self._client.xack(self.stream, self.group, *entry_ids))

    def pending_count(self) -> int:
        info = self._client.xpending(self.stream, self.group)
        return int(info["pending"]) if info else 0

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> FrameConsumer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class GlobalPublisher:
    """Đẩy `GlobalUpdate` lên `mct:global`. Dùng bởi engine liên kết (`python -m mct`).

    Stream này chỉ để dashboard theo dõi realtime, KHÔNG phải nguồn sự thật — nguồn sự
    thật là SQLite (`mct.store`). `MAXLEN` nhỏ hơn `mct:frames` mười lần vì mỗi tracklet
    chỉ sinh vài cập nhật, và mất vài cập nhật cũ không ảnh hưởng gì.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        stream: str = STREAM_GLOBAL,
        maxlen: int = MAXLEN_GLOBAL,
        client: redis.Redis | None = None,
    ) -> None:
        self.stream = stream
        self.maxlen = maxlen
        self._client = client or connect(url)
        self._owns_client = client is None

    def publish(self, update: GlobalUpdate) -> str:
        entry_id = self._client.xadd(
            self.stream,
            {_FIELD: encode_global_msgpack(update)},
            maxlen=self.maxlen,
            approximate=True,
        )
        return entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)

    def publish_many(self, updates: Iterable[GlobalUpdate]) -> int:
        """Gửi theo lô bằng pipeline — một vòng gán sinh hàng chục cập nhật một lúc."""
        pipe = self._client.pipeline(transaction=False)
        count = 0
        for update in updates:
            pipe.xadd(
                self.stream,
                {_FIELD: encode_global_msgpack(update)},
                maxlen=self.maxlen,
                approximate=True,
            )
            count += 1
        if count:
            pipe.execute()
        return count

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> GlobalPublisher:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_global(
    client: redis.Redis, *, last_id: str = "$", block_ms: int = 1000, count: int = 64
) -> list[tuple[str, GlobalUpdate]]:
    """Đọc `mct:global` bằng XREAD thường (không consumer group).

    Dashboard là bên đọc duy nhất và nó chỉ cần "những gì mới từ lúc tôi kết nối" — dùng
    consumer group ở đây chỉ tổ phải quản lý ack cho một thứ không cần đảm bảo giao nhận.
    """
    response = client.xread({STREAM_GLOBAL: last_id}, count=count, block=block_ms)
    out: list[tuple[str, GlobalUpdate]] = []
    for _stream, entries in response or []:
        for entry_id, fields in entries:
            raw = fields.get(_FIELD)
            if raw is None:
                log.warning("Entry %s trên %s thiếu field %r", entry_id, STREAM_GLOBAL, _FIELD)
                continue
            out.append((entry_id.decode(), decode_global_msgpack(raw)))
    return out
