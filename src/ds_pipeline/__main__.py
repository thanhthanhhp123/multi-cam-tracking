"""Entrypoint M1: file/RTSP -> YOLO -> nvtracker -> probe -> in ra console hoặc Redis.

    python -m ds_pipeline --config configs/pipeline/streams.yaml
    python -m ds_pipeline --config configs/pipeline/streams.yaml --publish
    python -m ds_pipeline --config configs/pipeline/streams_multi.yaml --stats

Không --publish: chỉ in message ra console (đúng mục tiêu M1, CLAUDE.md §9). Có
--publish: đẩy lên Redis qua FramePublisher — dùng khi đã sẵn sàng ghi fixture (M3).
--stats: không in từng frame, chỉ đếm và báo cáo throughput lúc kết thúc — con số đo
được mới là của pipeline chứ không phải của việc ghi log (CLAUDE.md §7).
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from common.logging import get_logger  # noqa: E402
from common.schema import FrameMessage  # noqa: E402
from ds_pipeline.builder import build_pipeline, load_pipeline_config  # noqa: E402
from ds_pipeline.probes import CameraGeometry, make_probe  # noqa: E402

log = get_logger(__name__)


def _print_sink(msg: FrameMessage) -> None:
    dets = ", ".join(
        f"id={d.local_track_id} conf={d.confidence:.2f} bbox={tuple(round(v) for v in d.bbox)}"
        for d in msg.detections
    )
    log.info(
        "[%s] frame=%d ts_ms=%d embed_dim=%d n_det=%d %s",
        msg.cam_id,
        msg.frame_id,
        msg.ts_ms,
        msg.embed_dim,
        len(msg.detections),
        dets,
    )


class _StatsSink:
    """Đếm frame/detection theo camera và báo cáo throughput khi pipeline kết thúc.

    Đồng hồ bắt đầu tính từ MESSAGE ĐẦU TIÊN, không phải từ lúc gọi set_state(PLAYING):
    lần chạy đầu còn phải build engine TensorRT (vài phút), gộp vào sẽ ra FPS vô nghĩa.
    """

    def __init__(self, inner=None) -> None:
        self._inner = inner
        self.frames: Counter[str] = Counter()
        self.detections: Counter[str] = Counter()
        self.embed_dims: set[int] = set()
        self.first_ts: float | None = None
        self.last_ts: float | None = None

    def __call__(self, msg: FrameMessage) -> None:
        now = time.perf_counter()
        if self.first_ts is None:
            self.first_ts = now
        self.last_ts = now
        self.frames[msg.cam_id] += 1
        self.detections[msg.cam_id] += len(msg.detections)
        self.embed_dims.add(msg.embed_dim)
        if self._inner is not None:
            self._inner(msg)

    @property
    def elapsed_s(self) -> float:
        if self.first_ts is None or self.last_ts is None:
            return 0.0
        return self.last_ts - self.first_ts

    def report(self) -> None:
        total_frames = sum(self.frames.values())
        elapsed = self.elapsed_s
        if not total_frames or elapsed <= 0:
            log.warning("không có frame nào đi qua probe — không báo cáo được throughput")
            return

        log.info(
            "throughput: %d frame / %.2f s = %.1f FPS gộp trên %d luồng "
            "(embed_dim=%s, tổng %d detection)",
            total_frames,
            elapsed,
            total_frames / elapsed,
            len(self.frames),
            sorted(self.embed_dims),
            sum(self.detections.values()),
        )
        for cam_id in sorted(self.frames):
            frames = self.frames[cam_id]
            log.info(
                "  %s: %d frame, %.1f FPS, %d detection (%.2f/frame)",
                cam_id,
                frames,
                frames / elapsed,
                self.detections[cam_id],
                self.detections[cam_id] / frames,
            )


def _redis_sink():
    from common.streams import FramePublisher

    publisher = FramePublisher()
    log.info("publish vào Redis stream '%s'", publisher.stream)

    def sink(msg: FrameMessage) -> None:
        publisher.publish(msg)

    return sink


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/pipeline/streams.yaml", help="đường dẫn streams.yaml"
    )
    parser.add_argument(
        "--publish", action="store_true", help="đẩy lên Redis thay vì chỉ in console"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="không in từng frame, chỉ báo cáo throughput lúc kết thúc (dùng khi đo FPS)",
    )
    args = parser.parse_args(argv)

    if not Path(args.config).is_absolute() and not Path(args.config).exists():
        # load_pipeline_config tự resolve theo project_root, nhưng thất bại sớm ở đây
        # thì thông báo lỗi rõ ràng hơn traceback từ YAML loader.
        pass

    cfg = load_pipeline_config(args.config)
    sink = _redis_sink() if args.publish else _print_sink
    stats: _StatsSink | None = None
    if args.stats:
        # --stats bỏ hẳn phần in từng frame: ở vài trăm FPS, chi phí ghi log lớn hơn
        # chi phí suy luận và sẽ trở thành thứ đang đo.
        stats = _StatsSink(_redis_sink() if args.publish else None)
        sink = stats

    geometries = {
        index: CameraGeometry(
            cam_id=source.cam_id, width=cfg.streammux.width, height=cfg.streammux.height
        )
        for index, source in enumerate(cfg.sources)
    }
    # CHÚ Ý: geometries dùng width/height của streammux làm "gốc" khi chưa biết độ
    # phân giải thật của từng camera. streams.yaml nên khai báo width/height riêng cho
    # từng nguồn khi cần scale bbox chính xác — xem CLAUDE.md §5. M1 chấp nhận giới hạn
    # này vì mục tiêu là chạy được pipeline 1 camera, chưa phải đo độ chính xác bbox.

    probe = make_probe(geometries, cfg.streammux.width, cfg.streammux.height, sink)
    pipeline = build_pipeline(cfg, probe=probe)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(_bus, message) -> None:
        t = message.type
        if t == Gst.MessageType.EOS:
            log.info("EOS — kết thúc pipeline")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            log.error("lỗi GStreamer: %s (%s)", err, debug)
            loop.quit()

    bus.connect("message", on_message)

    def on_sigint(*_args) -> None:
        log.info("nhận SIGINT — gửi EOS")
        pipeline.send_event(Gst.Event.new_eos())

    signal.signal(signal.SIGINT, on_sigint)

    log.info("bắt đầu chạy pipeline (%d nguồn)", len(cfg.sources))
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        if stats is not None:
            stats.report()

    return 0


if __name__ == "__main__":
    sys.exit(main())
