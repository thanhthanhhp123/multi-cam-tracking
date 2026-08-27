"""Phát lại fixture JSONL vào Redis, giữ đúng nhịp thời gian gốc.

Đây là thứ thay thế cho pipeline DeepStream khi làm việc trên máy không có GPU:
engine liên kết không phân biệt được message đến từ pipeline thật hay từ tool này.

    python -m tools.replay_metadata --fixture tests/fixtures/two_cam_walk.jsonl
    python -m tools.replay_metadata --fixture ... --speed 10      # chạy nhanh 10x
    python -m tools.replay_metadata --fixture ... --no-wait       # đẩy hết, không chờ
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from common.logging import get_logger
from common.schema import FrameMessage, read_jsonl
from common.streams import STREAM_FRAMES, FramePublisher

log = get_logger("tools.replay")


def replay(
    fixture: Path,
    *,
    stream: str = STREAM_FRAMES,
    url: str | None = None,
    speed: float = 1.0,
    wait: bool = True,
    loops: int = 1,
    realtime_ts: bool = False,
) -> int:
    messages: list[FrameMessage] = list(read_jsonl(fixture))
    if not messages:
        log.warning("%s không có message nào", fixture)
        return 0
    if speed <= 0:
        raise ValueError("--speed phải dương")

    span_ms = messages[-1].ts_ms - messages[0].ts_ms
    log.info(
        "Nạp %d frame từ %s (%.1fs dữ liệu, %d detection)",
        len(messages),
        fixture,
        span_ms / 1000.0,
        sum(len(m.detections) for m in messages),
    )

    sent = 0
    with FramePublisher(url, stream=stream) as pub:
        loop_idx = 0
        while loops <= 0 or loop_idx < loops:
            # Mỗi vòng lặp dịch timestamp về sau để ts_ms luôn tăng đơn điệu —
            # engine liên kết dựa vào ràng buộc thời gian nên thời gian lùi sẽ phá logic.
            base_shift = loop_idx * (span_ms + 1000)
            wall_start = time.monotonic()
            origin_ts = messages[0].ts_ms

            for msg in messages:
                if wait:
                    target = (msg.ts_ms - origin_ts) / 1000.0 / speed
                    lag = target - (time.monotonic() - wall_start)
                    if lag > 0:
                        time.sleep(lag)

                out = msg
                if base_shift or realtime_ts:
                    shift = base_shift
                    if realtime_ts:
                        # Gắn thời gian thực để so khớp với đồng hồ hệ thống khi demo.
                        shift += int(time.time() * 1000) - origin_ts - base_shift
                    out = replace(msg, ts_ms=msg.ts_ms + shift)

                pub.publish(out)
                sent += 1

            loop_idx += 1
            if loops <= 0 or loop_idx < loops:
                log.info("Hết fixture, lặp lại vòng %d", loop_idx + 1)

    log.info("Đã gửi %d message lên %r", sent, stream)
    return sent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fixture", type=Path, default=Path("tests/fixtures/two_cam_walk.jsonl"))
    p.add_argument("--stream", default=STREAM_FRAMES)
    p.add_argument("--url", default=None, help="mặc định lấy REDIS_URL")
    p.add_argument("--speed", type=float, default=1.0, help="hệ số tăng tốc thời gian")
    p.add_argument("--no-wait", action="store_true", help="đẩy hết ngay, bỏ qua nhịp thời gian")
    p.add_argument("--loops", type=int, default=1, help="0 = lặp vô hạn")
    p.add_argument("--realtime-ts", action="store_true", help="đổi ts_ms sang thời điểm hiện tại")
    args = p.parse_args(argv)

    try:
        replay(
            args.fixture,
            stream=args.stream,
            url=args.url,
            speed=args.speed,
            wait=not args.no_wait,
            loops=args.loops,
            realtime_ts=args.realtime_ts,
        )
    except KeyboardInterrupt:
        log.info("Dừng theo yêu cầu")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
