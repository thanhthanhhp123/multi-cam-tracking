"""Trạng thái realtime của dashboard: đọc `mct:global` một lần, phát cho mọi client.

**Một bên đọc Redis, N client WebSocket.** Không mở một consumer cho mỗi trình duyệt: vừa
nhân tải lên Redis, vừa khiến mỗi client thấy một tập dữ liệu khác nhau. Một `RedisBridge`
duy nhất đọc stream rồi đẩy vào `LiveState` và fan-out qua `Hub`.

**Vì sao `XREAD` từ `$` chứ không dùng consumer group.** Dashboard chỉ cần "những gì xảy ra
từ lúc tôi mở trang"; lịch sử là việc của SQLite (`mct.store`). Consumer group ở đây chỉ tổ
phải quản lý ack và pending list cho một thứ không cần đảm bảo giao nhận — mất một cập nhật
vị trí thì khung sau đã có cập nhật mới.

**Vị trí quy ra mét ngay tại server.** `GlobalUpdate` mang điểm chân theo toạ độ ảnh của
từng camera; phép ánh xạ về mặt phẳng chung cần homography, mà thứ đó nằm ở server. Trình
duyệt chỉ nhận `(x_m, y_m)` đã sẵn sàng để vẽ — client không phải biết gì về homography.
Camera chưa hiệu chỉnh thì `x_m/y_m = None` và giao diện xếp track đó vào danh sách riêng
thay vì vẽ bừa lên bản đồ.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from common.logging import get_logger
from common.schema import GlobalUpdate
from mct.homography import HomographyMapper

log = get_logger("dashboard.live")

DEFAULT_TTL_MS = 15_000
"""Không thấy lại trong khoảng này thì track biến khỏi bản đồ (không phải bị xoá khỏi SQLite)."""

TRAIL_LENGTH = 32
"""Số điểm vệt di chuyển giữ cho mỗi Global ID — đủ để thấy hướng đi, không phình bộ nhớ."""


@dataclass(slots=True)
class LiveTrack:
    """Trạng thái hiện tại của một Global ID theo con mắt dashboard."""

    global_id: int
    cam_id: str
    ts_ms: int
    n_cameras: int
    x_m: float | None = None
    y_m: float | None = None
    cameras: set[str] = field(default_factory=set)
    trail: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))

    def to_json(self) -> dict[str, Any]:
        return {
            "global_id": self.global_id,
            "cam_id": self.cam_id,
            "ts_ms": self.ts_ms,
            "n_cameras": self.n_cameras,
            "cameras": sorted(self.cameras),
            "x_m": self.x_m,
            "y_m": self.y_m,
            "trail": [[round(x, 3), round(y, 3)] for x, y in self.trail],
        }


class LiveState:
    """Ai đang ở đâu, ngay lúc này. Bộ nhớ trong, mất khi restart — đúng bản chất của nó."""

    def __init__(self, *, mapper: HomographyMapper | None = None, ttl_ms: int = DEFAULT_TTL_MS):
        self.mapper = mapper
        self.ttl_ms = ttl_ms
        self.tracks: dict[int, LiveTrack] = {}
        self.latest_ts_ms = 0
        self.n_updates = 0

    def apply(self, update: GlobalUpdate) -> LiveTrack:
        track = self.tracks.get(update.global_id)
        if track is None:
            track = LiveTrack(
                global_id=update.global_id,
                cam_id=update.cam_id,
                ts_ms=update.ts_ms,
                n_cameras=update.n_cameras,
            )
            self.tracks[update.global_id] = track

        track.cam_id = update.cam_id
        track.ts_ms = max(track.ts_ms, update.ts_ms)
        track.n_cameras = max(track.n_cameras, update.n_cameras)
        track.cameras.add(update.cam_id)

        point = self.mapper.project(update.cam_id, update.ground_point) if self.mapper else None
        if point is not None:
            track.x_m, track.y_m = point
            track.trail.append(point)

        self.latest_ts_ms = max(self.latest_ts_ms, update.ts_ms)
        self.n_updates += 1
        return track

    def apply_many(self, updates: list[GlobalUpdate]) -> list[LiveTrack]:
        return [self.apply(update) for update in updates]

    def expire(self, now_ms: int | None = None) -> list[int]:
        """Bỏ track im lặng quá `ttl_ms`. Mốc so sánh là `ts_ms` của DỮ LIỆU, không phải
        đồng hồ server — để phát lại fixture cũ vẫn hiện đúng như lúc chạy thật."""
        cutoff = (self.latest_ts_ms if now_ms is None else now_ms) - self.ttl_ms
        gone = [gid for gid, track in self.tracks.items() if track.ts_ms < cutoff]
        for gid in gone:
            del self.tracks[gid]
        return gone

    def snapshot(self) -> dict[str, Any]:
        """Toàn cảnh cho client vừa kết nối — trước đó nó chưa nhận được cập nhật nào."""
        return {
            "type": "snapshot",
            "latest_ts_ms": self.latest_ts_ms,
            "n_updates": self.n_updates,
            "tracks": [track.to_json() for track in sorted(self.tracks.values(), key=_by_id)],
        }


def _by_id(track: LiveTrack) -> int:
    return track.global_id


class Hub:
    """Fan-out WebSocket đơn giản. Client chậm hoặc đã đóng thì bị loại, không chặn vòng đọc."""

    def __init__(self) -> None:
        self._clients: set[Any] = set()

    def add(self, websocket: Any) -> None:
        self._clients.add(websocket)

    def remove(self, websocket: Any) -> None:
        self._clients.discard(websocket)

    @property
    def n_clients(self) -> int:
        return len(self._clients)

    async def broadcast(self, payload: dict[str, Any]) -> int:
        sent = 0
        for websocket in list(self._clients):
            try:
                await websocket.send_json(payload)
                sent += 1
            except Exception:  # client đóng tab, mạng rớt — không phải lỗi của server
                self._clients.discard(websocket)
        return sent


class RedisBridge:
    """Vòng nền: `mct:global` → `LiveState` → `Hub`.

    `redis-py` là client đồng bộ và `XREAD` có block, nên phần đọc chạy trong thread riêng
    (`asyncio.to_thread`) — chặn event loop ở đây là treo luôn mọi WebSocket đang mở.
    """

    def __init__(
        self,
        state: LiveState,
        hub: Hub,
        *,
        url: str | None = None,
        block_ms: int = 1000,
        count: int = 256,
    ) -> None:
        self.state = state
        self.hub = hub
        self.url = url
        self.block_ms = block_ms
        self.count = count
        self.connected = False
        self.last_error: str | None = None
        self._task: asyncio.Task | None = None
        self._client: Any = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="dashboard-redis-bridge")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._client is not None:
            self._client.close()
            self._client = None

    async def _run(self) -> None:
        from common.streams import connect, read_global

        last_id = "$"
        while True:
            try:
                if self._client is None:
                    self._client = await asyncio.to_thread(connect, self.url)
                    await asyncio.to_thread(self._client.ping)
                    self.connected = True
                    self.last_error = None
                    log.info("Đã nối Redis, theo dõi mct:global")

                batch = await asyncio.to_thread(
                    read_global,
                    self._client,
                    last_id=last_id,
                    block_ms=self.block_ms,
                    count=self.count,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.connected = False
                self.last_error = str(exc)
                if self._client is not None:
                    with contextlib.suppress(Exception):
                        self._client.close()
                    self._client = None
                log.warning("Mất kết nối Redis (%s), thử lại sau 2 s", exc)
                await asyncio.sleep(2.0)
                continue

            if not batch:
                if self.state.expire():
                    await self.hub.broadcast(self.state.snapshot())
                continue

            last_id = batch[-1][0]
            tracks = self.state.apply_many([update for _entry_id, update in batch])
            self.state.expire()
            await self.hub.broadcast(
                {
                    "type": "update",
                    "server_ts_ms": int(time.time() * 1000),
                    "tracks": [track.to_json() for track in tracks],
                }
            )
