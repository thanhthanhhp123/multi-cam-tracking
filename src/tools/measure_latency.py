"""Đo độ trễ end-to-end: từ lúc probe đóng dấu thời gian tới lúc cập nhật Global ID đọc được.

Chuỗi được đo (CLAUDE.md §3):

    probe (ts_ms) -> XADD mct:frames -> engine liên kết -> XADD mct:global -> ĐÂY

`GlobalUpdate.ts_ms` là mốc muộn nhất của tracklet tại thời điểm gán — tức dấu thời gian
do probe gắn, KHÔNG phải giờ hệ thống lúc engine ghi. Vậy `now - ts_ms` chính là độ trễ
của cả chuỗi trên. Đọc bằng `XREAD` từ `$` (chỉ những gì xảy ra từ lúc chạy công cụ này),
cùng lý do như `dashboard/live.py`: đây là quan trắc thời gian thực, không phải tra lịch sử.

CHỈ có nghĩa khi nguồn phát ĐÚNG TỐC ĐỘ THẬT (`sink.sync: true` trong streams.yaml). Chạy
hết tốc lực thì engine bị dội và con số đo được là độ trễ lúc quá tải, không phải độ trễ
vận hành.

    python -m tools.measure_latency --duration 90

Độ trễ này BAO GỒM cửa sổ gom của engine (mặc định 1 s, `configs/mct.yaml`) — đó là thành
phần lớn nhất và là chủ ý thiết kế, không phải chi phí thừa.
"""

from __future__ import annotations

import argparse
import statistics
import time

from common.logging import get_logger
from common.streams import connect, read_global

log = get_logger("tools.latency")


def do(url: str | None, duration: float, warmup: float) -> int:
    client = connect(url)
    het_han = time.time() + duration
    bat_dau_ghi = time.time() + warmup

    tre_ms: list[float] = []
    bo_qua = 0
    last_id = "$"

    log.info("Đọc mct:global trong %.0f s (bỏ %.0f s đầu cho ổn định)", duration, warmup)
    while time.time() < het_han:
        con_lai = max(het_han - time.time(), 0.1)
        goi = read_global(client, last_id=last_id, block_ms=int(min(con_lai, 2.0) * 1000))
        for entry_id, cap_nhat in goi:
            last_id = entry_id
            if time.time() < bat_dau_ghi:
                bo_qua += 1
                continue
            tre_ms.append(time.time() * 1000.0 - cap_nhat.ts_ms)

    if not tre_ms:
        log.error("Không nhận được cập nhật nào — engine có chạy không? nguồn có phát không?")
        return 1

    tre_ms.sort()

    def pct(p: float) -> float:
        return tre_ms[min(int(len(tre_ms) * p), len(tre_ms) - 1)]

    log.info("mẫu           = %d (bỏ %d lúc khởi động)", len(tre_ms), bo_qua)
    log.info("trung vị      = %8.1f ms", statistics.median(tre_ms))
    log.info("trung bình    = %8.1f ms", statistics.fmean(tre_ms))
    log.info("p90           = %8.1f ms", pct(0.90))
    log.info("p99           = %8.1f ms", pct(0.99))
    log.info("nhỏ nhất/lớn nhất = %.1f / %.1f ms", tre_ms[0], tre_ms[-1])

    muc_tieu = 1000.0  # CLAUDE.md §7: độ trễ end-to-end < 1 s
    dat = pct(0.90) < muc_tieu
    log.info("mục tiêu đề cương < %.0f ms: %s (theo p90)", muc_tieu, "ĐẠT" if dat else "KHÔNG ĐẠT")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=None, help="mặc định lấy REDIS_URL")
    p.add_argument("--duration", type=float, default=60.0, help="số giây quan trắc")
    p.add_argument(
        "--warmup",
        type=float,
        default=5.0,
        help="bỏ qua N giây đầu — cửa sổ đầu tiên của engine luôn lệch",
    )
    args = p.parse_args(argv)
    return do(args.url, args.duration, args.warmup)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
