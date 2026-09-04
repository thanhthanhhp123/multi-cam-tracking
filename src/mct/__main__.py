"""Engine liên kết đa camera — vòng chạy online.

    # chạy thật: đọc mct:frames từ Redis, ghi mct:global + SQLite
    python -m mct --config configs/mct.yaml

    # chạy không cần Redis: nạp thẳng fixture, vẫn đi qua đúng vòng lặp online
    python -m mct --source tests/fixtures/two_cam_walk.jsonl --db /tmp/mct.db

Sơ đồ (CLAUDE.md §3): `mct:frames` → `TrackletBuilder` → `Associator` → `mct:global` +
SQLite. Không có GPU ở đây và không được có: module này phải chạy được trên máy dev.

**Cửa sổ.** Cứ mỗi `association.window_ms` (mốc thời gian lấy từ `ts_ms` trong message,
KHÔNG phải đồng hồ hệ thống) thì: đóng tracklet hết hạn, lấy danh sách tracklet vừa động,
chạy một vòng gán, phát `GlobalUpdate`, ghi SQLite, rồi đóng GlobalTrack quá TTL.
Dùng `ts_ms` chứ không dùng đồng hồ hệ thống để phát lại một fixture cho ra **đúng** kết
quả như lúc chạy thật — nếu không thì không có cách nào tái lập số liệu của chương 6.

**Vì sao có `--source <fixture>`.** Chênh lệch giữa chế độ online và offline chính là cái
giá phải trả của ràng buộc thời gian thực (CLAUDE.md §6), và muốn đo nó thì phải cho cùng
một dữ liệu chạy qua cả hai đường. Đường offline là `eval/eval_wildtrack.py`; đường online
là đúng file này với `--source`. Không cần Redis, không cần GPU, chạy lại được bao nhiêu
lần cũng ra một kết quả.
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
from collections.abc import Iterator
from pathlib import Path
from types import FrameType
from typing import Any

import yaml

from common.logging import get_logger
from common.schema import FrameMessage, GlobalUpdate, read_jsonl
from mct.associator import Assignment, Associator
from mct.homography import HomographyMapper
from mct.store import Store, StoreConfig
from mct.topology import Topology
from mct.tracklet import TrackletBuilder, TrackletConfig

log = get_logger("mct")


class Engine:
    """Máy trạng thái của vòng online. Không biết gì về Redis — chỉ nhận `FrameMessage`.

    Tách ra khỏi phần I/O để test được toàn bộ logic cửa sổ bằng fixture, không cần
    Redis cũng không cần đồng hồ thật (CLAUDE.md §2 quy tắc 3).
    """

    def __init__(
        self,
        *,
        tracklet_config: TrackletConfig | None = None,
        associator: Associator | None = None,
        store: Store | None = None,
        window_ms: int = 1000,
    ) -> None:
        self.builder = TrackletBuilder(tracklet_config)
        self.associator = associator or Associator()
        self.store = store
        self.window_ms = window_ms
        self._window_end_ms: int | None = None
        self.n_messages = 0
        self.n_updates = 0

    def feed(self, msg: FrameMessage) -> list[GlobalUpdate]:
        """Nạp một message. Trả về các cập nhật Global ID nếu cửa sổ vừa đóng."""
        self.n_messages += 1
        closed = self.builder.update(msg)
        if self._window_end_ms is None:
            self._window_end_ms = int(msg.ts_ms) + self.window_ms

        if int(msg.ts_ms) < self._window_end_ms:
            # Tracklet bị đóng sớm (local_track_id được cấp lại) vẫn phải vào vòng gán
            # ngay, nếu không nó biến mất khỏi `take_updated()` của cửa sổ sau.
            return self._run_window(int(msg.ts_ms), extra=closed) if closed else []

        self._window_end_ms = int(msg.ts_ms) + self.window_ms
        return self._run_window(int(msg.ts_ms), extra=closed)

    def finish(self) -> list[GlobalUpdate]:
        """Hết luồng: đóng mọi tracklet còn mở rồi chạy vòng gán cuối."""
        now_ms = self.builder.latest_ts_ms
        updates = self._run_window(now_ms, extra=self.builder.flush(), force=True)
        if self.store is not None:
            self.store.flush()
        return updates

    def _run_window(
        self, now_ms: int, *, extra: list[Any] | None = None, force: bool = False
    ) -> list[GlobalUpdate]:
        pending = list(extra or [])
        pending.extend(self.builder.close_expired(now_ms))
        pending.extend(self.builder.take_updated())
        # Cùng một tracklet có thể vừa bị đóng vừa nằm trong `take_updated` — gán hai lần
        # trong một vòng là vô hại nhưng làm thống kê sai, nên lọc trùng theo id.
        unique: dict[int, Any] = {t.tracklet_id: t for t in pending}
        if not unique and not force:
            return []

        results = self.associator.assign(list(unique.values()))
        # TTL của GlobalTrack nằm trong GalleryConfig (`association.global_track_ttl_ms`),
        # `prune` nhận mốc hiện tại chứ không nhận mốc cắt.
        closed_tracks = self.associator.prune(now_ms)

        if self.store is not None:
            self.store.record_many(results, now_ms=now_ms)
            if closed_tracks:
                self.store.close_tracks([t.global_id for t in closed_tracks])

        updates = [self._to_update(r) for r in results]
        self.n_updates += len(updates)
        return updates

    def _to_update(self, assignment: Assignment) -> GlobalUpdate:
        tracklet = assignment.tracklet
        track = self.associator.gallery.get(assignment.global_id)
        return GlobalUpdate(
            global_id=assignment.global_id,
            cam_id=tracklet.cam_id,
            local_track_id=tracklet.local_track_id,
            tracklet_id=tracklet.tracklet_id,
            ts_ms=tracklet.end_ms,
            bbox=tracklet.last_bbox,
            ground_point=tracklet.last_ground_point,
            cost=assignment.cost if assignment.cost == assignment.cost else 0.0,
            is_new=assignment.is_new,
            is_update=assignment.is_update,
            n_cameras=len(track.cameras) if track is not None else 1,
            reason=assignment.reason,
        )


# --------------------------------------------------------------------------- dựng


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_engine(
    config: dict[str, Any],
    *,
    topology_path: Path | None = None,
    homography_dir: Path | None = None,
    db_path: str | None = None,
) -> Engine:
    topology = Topology.load(topology_path) if topology_path else None
    mapper = None
    # Thư mục rỗng (chưa hiệu chỉnh camera nào) là trạng thái BÌNH THƯỜNG, không phải lỗi:
    # hệ thống vẫn chạy được bằng ngoại hình + ràng buộc thời gian, chỉ mất phần hình học.
    if homography_dir is not None and any(Path(homography_dir).glob("*.yaml")):
        mapper = HomographyMapper.load(homography_dir)
        log.info("Homography: %d camera đã hiệu chỉnh", len(mapper.calibrated))
    else:
        log.info("Không có file hiệu chỉnh trong %s — bỏ qua thành phần hình học", homography_dir)

    associator = Associator.from_mapping(config, topology=topology, ground_mapper=mapper)
    store_config = StoreConfig.from_mapping(config)
    if db_path:
        store_config.db_path = db_path

    association = dict(config.get("association", {}) or {})
    return Engine(
        tracklet_config=TrackletConfig.from_mapping(config),
        associator=associator,
        store=Store(store_config),
        window_ms=int(association.get("window_ms", 1000)),
    )


# --------------------------------------------------------------------------- nguồn


def _fixture_source(path: Path) -> Iterator[FrameMessage]:
    yield from read_jsonl(path)


def _redis_source(url: str | None, *, block_ms: int, idle_limit: int) -> Iterator[FrameMessage]:
    """Đọc `mct:frames` qua consumer group, ack sau khi engine đã nuốt xong message.

    Ack SAU chứ không phải trước: engine chết giữa chừng thì Redis giao lại đúng những
    message chưa xử lý, không mất tracklet nào.
    """
    from common.streams import FrameConsumer

    with FrameConsumer(url) as consumer:
        for entry_id, msg in consumer.read_pending():
            yield msg
            consumer.ack([entry_id])

        idle = 0
        while True:
            batch = consumer.read(block_ms=block_ms)
            if not batch:
                idle += 1
                if 0 < idle_limit <= idle:
                    log.info("Không có message mới sau %d lần chờ, dừng", idle)
                    return
                continue
            idle = 0
            for entry_id, msg in batch:
                yield msg
                consumer.ack([entry_id])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, default=Path("configs/mct.yaml"))
    p.add_argument(
        "--source",
        default="redis",
        help="'redis' (mặc định) hoặc đường dẫn fixture .jsonl để chạy không cần Redis",
    )
    p.add_argument("--redis-url", default=None)
    p.add_argument("--topology", type=Path, default=Path("configs/cameras/topology.yaml"))
    p.add_argument(
        "--homography-dir",
        type=Path,
        default=Path("configs/cameras/homography"),
        help="thư mục file hiệu chỉnh; không tồn tại thì bỏ qua thành phần hình học",
    )
    p.add_argument("--db", default=None, help="ghi đè store.db_path trong config")
    p.add_argument("--publish", action="store_true", help="đẩy GlobalUpdate lên mct:global")
    p.add_argument("--block-ms", type=int, default=1000)
    p.add_argument(
        "--idle-limit",
        type=int,
        default=0,
        help="dừng sau N lần chờ không có message (0 = chạy mãi). Dùng cho chạy theo lô",
    )
    args = p.parse_args(argv)

    config = load_config(args.config if args.config.is_file() else None)
    engine = build_engine(
        config,
        topology_path=args.topology if args.topology.is_file() else None,
        homography_dir=args.homography_dir,
        db_path=args.db,
    )

    publisher = None
    if args.publish:
        from common.streams import GlobalPublisher

        publisher = GlobalPublisher(args.redis_url)

    if args.source == "redis":
        source = _redis_source(args.redis_url, block_ms=args.block_ms, idle_limit=args.idle_limit)
    else:
        source = _fixture_source(Path(args.source))

    stop = {"now": False}

    def _handle(_signum: int, _frame: FrameType | None) -> None:
        log.info("Nhận tín hiệu dừng, đang kết thúc cửa sổ cuối...")
        stop["now"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        # Chạy trong thread phụ (test, dashboard nhúng) thì không đăng ký được handler.
        with contextlib.suppress(ValueError):
            signal.signal(sig, _handle)

    try:
        for msg in source:
            updates = engine.feed(msg)
            if updates and publisher is not None:
                publisher.publish_many(updates)
            if stop["now"]:
                break
        final = engine.finish()
        if final and publisher is not None:
            publisher.publish_many(final)
    finally:
        if engine.store is not None:
            summary = engine.store.summary()
            engine.store.close()
            log.info(
                "%d message, %d cập nhật Global ID, %d Global ID (%d xuyên camera), "
                "%d lượt xuất hiện",
                engine.n_messages,
                engine.n_updates,
                summary["n_tracks"],
                summary["n_cross_camera"],
                summary["n_appearances"],
            )
        stats = engine.associator.stats
        log.info(
            "vòng gán: %d cửa sổ, %d khớp, %d cập nhật, %d tạo mới "
            "(%d bị ngưỡng loại, %d hết ứng viên)",
            stats.windows,
            stats.matched,
            stats.updated,
            stats.created,
            stats.rejected_by_threshold,
            stats.no_candidate,
        )
        if publisher is not None:
            publisher.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
