"""Ghi luồng metadata từ Redis ra file JSONL.

Chạy cái này trên máy GPU khi pipeline DeepStream đang chạy (M3) để tạo fixture THẬT,
rồi chép file JSONL về máy dev. Đó là toàn bộ lý do chọn Redis Streams làm ranh giới —
xem CLAUDE.md §3.

    python -m tools.record_metadata --out tests/fixtures/hanh_lang_2cam.jsonl --duration 60
"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path
from types import FrameType

from common.logging import get_logger
from common.schema import encode_jsonl, validate
from common.streams import STREAM_FRAMES, FrameConsumer

log = get_logger("tools.record")

_stop = False


def _handle_signal(_sig: int, _frame: FrameType | None) -> None:
    global _stop
    _stop = True


def record(
    out: Path,
    *,
    stream: str = STREAM_FRAMES,
    url: str | None = None,
    group: str = "recorder",
    duration_s: float = 0.0,
    max_frames: int = 0,
    check: bool = True,
) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + duration_s if duration_s > 0 else None
    written = 0
    problems = 0

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("Ghi %r -> %s (Ctrl-C để dừng)", stream, out)

    with (
        out.open("w", encoding="utf-8") as fh,
        FrameConsumer(
            url, stream=stream, group=group, consumer="recorder-1", start_id="0"
        ) as consumer,
    ):
        while not _stop:
            if deadline and time.monotonic() >= deadline:
                break
            batch = consumer.read(count=256, block_ms=1000)
            if not batch:
                continue

            for _entry_id, msg in batch:
                if check:
                    issues = validate(msg)
                    if issues:
                        problems += 1
                        # Không bỏ message — ghi lại rồi cảnh báo, vì fixture "xấu" chính
                        # là bằng chứng cần thiết khi truy bug ở pipeline.
                        log.warning("%s frame %d: %s", msg.cam_id, msg.frame_id, "; ".join(issues))
                fh.write(encode_jsonl(msg))
                fh.write("\n")
                written += 1

            consumer.ack([entry_id for entry_id, _ in batch])
            fh.flush()

            if max_frames and written >= max_frames:
                break

    log.info("Đã ghi %d frame vào %s", written, out)
    if problems:
        log.warning(
            "%d frame vi phạm contract — kiểm tra lại probe trước khi dùng làm fixture", problems
        )
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=Path("tests/fixtures/recorded.jsonl"))
    p.add_argument("--stream", default=STREAM_FRAMES)
    p.add_argument("--url", default=None, help="mặc định lấy REDIS_URL")
    p.add_argument("--group", default="recorder")
    p.add_argument("--duration", type=float, default=0.0, help="số giây, 0 = tới khi Ctrl-C")
    p.add_argument("--max-frames", type=int, default=0, help="0 = không giới hạn")
    p.add_argument("--no-check", action="store_true", help="bỏ qua kiểm tra contract")
    args = p.parse_args(argv)

    record(
        args.out,
        stream=args.stream,
        url=args.url,
        group=args.group,
        duration_s=args.duration,
        max_frames=args.max_frames,
        check=not args.no_check,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
