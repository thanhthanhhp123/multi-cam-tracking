"""Đọc file mốc thời gian của engine, tính trung vị/p90/p99 CHO TỪNG ĐOẠN.

    # engine ghi log khi chạy
    python -m mct --publish --latency-log data/latency.jsonl

    # rồi đọc
    python -m tools.latency_report --log data/latency.jsonl
    python -m tools.latency_report --log data/latency.jsonl --by cam_id --top 5
    python -m tools.latency_report --log data/latency.jsonl --json > data/latency.json

**Vì sao không gộp vào `tools/measure_latency.py`.** Cái kia quan trắc `mct:global` từ
bên ngoài và trả lời "hệ thống đang trễ bao nhiêu" — dùng được cả khi không bật đo, kể
cả với engine của người khác. Cái này mổ xẻ file mốc để trả lời "trễ ở ĐÂU". Hai câu hỏi
khác nhau, hai công cụ, và cái này không cần Redis.

**Ba thứ phải đọc cùng nhau, đừng chỉ nhìn p90 tổng:**

1. **Bảng đoạn thô** — bốn đoạn của đề bài, cộng lại ra end-to-end.
2. **Bảng đoạn nhỏ** — thụt lề dưới đoạn thô chứa nó. Đây mới là chỗ chỉ đích danh:
   `queue_wait` lớn là engine/Redis không theo kịp pipeline; `db_write` lớn là commit
   SQLite chặn vòng lặp; `window_wait` lớn là **chờ theo thiết kế**, và nó gộp hai thứ
   khác hẳn nhau — chờ cửa sổ hiện tại đóng (`association.window_ms`, cỡ ≤ 1 s) và chờ
   một tracklet im lặng đủ lâu để bị đóng (`tracklet.idle_timeout_ms`, cỡ vài giây).

   Đây chính là phép kiểm cho giả thuyết của phiên 9 (`docs/worklog/2026-09-04-9-*`):
   đuôi p90 2.1 s được cho là "độ trễ CHỐT danh tính" chứ không phải nghẽn hàng đợi.
   Nếu đúng thì đuôi nằm gần hết ở `window_wait` và `queue_wait` phẳng; nếu sai thì
   ngược lại. Trước đây không phân biệt được hai khả năng đó bằng số.
3. **Quy trách nhiệm cho đuôi** — lấy riêng các bản ghi nằm trong 10% chậm nhất rồi so
   trung vị từng đoạn của nhóm đó với trung vị chung. Đoạn nào phình ra ở nhóm đuôi
   chính là đoạn tạo ra đuôi. Trung bình toàn cục KHÔNG trả lời được câu này: một đoạn
   chậm 2 s ở 5% số mẫu gần như không nhúc nhích trung vị chung.

**Cột `âm`** đếm số mẫu có độ dài âm. Với đoạn nằm trong một máy thì đó là lỗi logic;
với đoạn bắc cầu hai máy (đánh dấu `*`) thì đó là LỆCH ĐỒNG HỒ giữa hai máy — số đo của
đoạn đó lệch đi đúng bằng độ lệch ấy, và không sửa được bằng cách đo lại (xem CLAUDE.md
§11: đồng bộ NTP là điều kiện sống còn).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from common.latency import (
    ALL_SEGMENTS,
    COARSE_SEGMENTS,
    END_TO_END,
    FINE_SEGMENTS,
    T0_CAPTURE,
    T1_PROBE,
    T1B_DEQUEUE,
    T2_XADD,
    T3_ASSOC,
    T3A_RECV,
    T3D_DB,
    T3W_WINDOW,
    T4_OUT,
    LatencyRecord,
    Segment,
    Stats,
    deltas,
    latency_log_path,
    percentile,
    read_records,
    summarize,
)
from common.logging import get_logger

log = get_logger("tools.latency_report")

# Mục tiêu đề cương (CLAUDE.md §7): độ trễ end-to-end < 1 s.
TARGET_MS = 1000.0


def select_run(records: list[LatencyRecord], run: str) -> list[LatencyRecord]:
    """`run`: `last` (mặc định), `all`, hoặc một `run_id` cụ thể.

    Mặc định lấy lần chạy MỚI NHẤT vì file mở kiểu nối thêm: trộn hai lần chạy khác cấu
    hình vào một bảng là cách chắc chắn nhất để rút ra kết luận sai.
    """
    if run == "all" or not records:
        return records
    if run != "last":
        chosen = [r for r in records if r.run_id == run]
        if not chosen:
            known = sorted({r.run_id for r in records})
            raise SystemExit(f"không có run_id {run!r} trong file. Đang có: {', '.join(known)}")
        return chosen
    last_id = records[-1].run_id
    return [r for r in records if r.run_id == last_id]


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:9.1f}"


def _row(name: str, stats: Stats | None, *, cross_host: bool) -> str:
    mark = "*" if cross_host else " "
    if stats is None:
        return f"{name:<24}{mark} {'—':>9} {'—':>9} {'—':>9} {'—':>9} {'—':>9} {'—':>9} {'—':>5}"
    return (
        f"{name:<24}{mark} {stats.n:>9d} {_fmt(stats.median)} {_fmt(stats.p90)} "
        f"{_fmt(stats.p99)} {_fmt(stats.mean)} {_fmt(stats.maximum)} {stats.n_negative:>5d}"
    )


_HEADER = (
    f"{'đoạn':<24}  {'n':>9} {'p50':>9} {'p90':>9} {'p99':>9} {'mean':>9} {'max':>9} {'âm':>5}"
)


def segment_stats(records: Sequence[LatencyRecord]) -> dict[str, Stats | None]:
    return {
        seg.name: summarize(deltas(records, seg), neg_tolerance_ms=seg.neg_tolerance_ms)
        for seg in ALL_SEGMENTS
    }


def report(
    records: list[LatencyRecord], *, top: int = 0, group_by: str = "", as_json: bool = False
) -> dict[str, Any]:
    """In báo cáo và trả về cùng nội dung dưới dạng dict (cho `--json`)."""
    stats = segment_stats(records)
    e2e = deltas(records, END_TO_END)

    payload: dict[str, Any] = {
        "n_records": len(records),
        "run_ids": sorted({r.run_id for r in records}),
        "segments": {
            name: (
                None
                if s is None
                else {
                    "n": s.n,
                    "median_ms": round(s.median, 2),
                    "p90_ms": round(s.p90, 2),
                    "p99_ms": round(s.p99, 2),
                    "mean_ms": round(s.mean, 2),
                    "min_ms": round(s.minimum, 2),
                    "max_ms": round(s.maximum, 2),
                    "n_negative": s.n_negative,
                }
            )
            for name, s in stats.items()
        },
    }

    if not as_json:
        log.info("%d bản ghi, run: %s", len(records), ", ".join(payload["run_ids"]) or "—")
        log.info("%s", _HEADER)
        log.info("%s", "-" * len(_HEADER))
        for seg in COARSE_SEGMENTS:
            log.info("%s", _row(seg.name, stats[seg.name], cross_host=seg.cross_host))
            for fine in FINE_SEGMENTS:
                if _inside(fine, seg):
                    log.info("%s", _row(fine.name, stats[fine.name], cross_host=fine.cross_host))
        log.info("%s", "-" * len(_HEADER))
        log.info("%s", _row(END_TO_END.name, stats[END_TO_END.name], cross_host=True))
        log.info("(*) đoạn bắc cầu hai máy — cột `âm` là bằng chứng lệch đồng hồ, xem docstring")

    payload["tail"] = _tail_attribution(records, e2e, as_json=as_json)
    if group_by:
        payload["groups"] = _by_group(records, group_by, as_json=as_json)
    if top:
        payload["worst"] = _worst(records, top, as_json=as_json)
    payload["warnings"] = _warnings(records, stats, e2e, as_json=as_json)
    return payload


# Thứ tự thời gian của các mốc — dùng để biết đoạn nhỏ nào nằm trong đoạn thô nào.
_CHAIN = (
    T0_CAPTURE,
    T1_PROBE,
    T1B_DEQUEUE,
    T2_XADD,
    T3A_RECV,
    T3W_WINDOW,
    T3_ASSOC,
    T3D_DB,
    T4_OUT,
)


def _inside(fine: Segment, coarse: Segment) -> bool:
    """Đoạn nhỏ có nằm trong đoạn thô không — dùng để thụt lề bảng.

    So theo THỨ TỰ MỐC chứ không so tên: bảng không bao giờ lệch với định nghĩa đoạn
    trong `common/latency.py`, kể cả khi thêm mốc mới vào giữa chuỗi.
    """
    try:
        return _CHAIN.index(coarse.start) <= _CHAIN.index(fine.start) and _CHAIN.index(
            fine.end
        ) <= _CHAIN.index(coarse.end)
    except ValueError:
        return False


def _tail_attribution(
    records: Sequence[LatencyRecord], e2e: Sequence[float], *, as_json: bool
) -> dict[str, Any] | None:
    """Đoạn nào phình ra ở nhóm 10% chậm nhất — câu trả lời cho "đuôi trễ do đâu"."""
    if len(e2e) < 10:
        return None
    cutoff = percentile(e2e, 0.90)
    tail = [r for r in records if (d := r.delta(END_TO_END)) is not None and d >= cutoff]
    if not tail:
        return None

    rows: list[dict[str, Any]] = []
    for seg in (*COARSE_SEGMENTS, *FINE_SEGMENTS):
        overall = summarize(deltas(records, seg), neg_tolerance_ms=seg.neg_tolerance_ms)
        in_tail = summarize(deltas(tail, seg), neg_tolerance_ms=seg.neg_tolerance_ms)
        if overall is None or in_tail is None:
            continue
        rows.append(
            {
                "segment": seg.name.strip(),
                "fine": seg in FINE_SEGMENTS,
                "median_ms": round(overall.median, 2),
                "tail_median_ms": round(in_tail.median, 2),
                "delta_ms": round(in_tail.median - overall.median, 2),
            }
        )
    # Hoà thì ĐOẠN NHỎ đứng trước: một đoạn thô luôn phình đúng bằng đoạn nhỏ gây ra nó,
    # mà câu trả lời hữu ích là cái cụ thể hơn ("chờ cửa sổ"), không phải cái bao ngoài.
    rows.sort(key=lambda r: (-r["delta_ms"], 0 if r["fine"] else 1))

    if not as_json:
        log.info("")
        log.info(
            "ĐUÔI TRỄ: %d bản ghi có end-to-end >= p90 = %.1f ms (tối đa %.1f ms)",
            len(tail),
            cutoff,
            max(e2e),
        )
        log.info("%-24s %12s %12s %12s", "đoạn", "p50 chung", "p50 ở đuôi", "chênh")
        for row in rows:
            log.info(
                "%-24s %12.1f %12.1f %+12.1f",
                row["segment"],
                row["median_ms"],
                row["tail_median_ms"],
                row["delta_ms"],
            )
        if rows:
            log.info("→ đuôi trễ chủ yếu sinh ra ở: %s", rows[0]["segment"])

    return {"cutoff_ms": round(cutoff, 2), "n_tail": len(tail), "segments": rows}


def _by_group(records: Sequence[LatencyRecord], field: str, *, as_json: bool) -> dict[str, Any]:
    """Tách end-to-end theo `cam_id` / `global_id` / `db_flushed` / `t0_source`."""
    buckets: dict[str, list[LatencyRecord]] = defaultdict(list)
    for record in records:
        buckets[str(getattr(record, field, ""))].append(record)

    out: dict[str, Any] = {}
    if not as_json:
        log.info("")
        log.info("END-TO-END theo %s:", field)
        log.info("%-24s  %9s %9s %9s %9s", "giá trị", "n", "p50", "p90", "p99")
    for key in sorted(buckets):
        stats = summarize(deltas(buckets[key], END_TO_END))
        if stats is None:
            continue
        out[key] = {
            "n": stats.n,
            "median_ms": round(stats.median, 2),
            "p90_ms": round(stats.p90, 2),
            "p99_ms": round(stats.p99, 2),
        }
        if not as_json:
            log.info(
                "%-24s  %9d %9.1f %9.1f %9.1f",
                key,
                stats.n,
                stats.median,
                stats.p90,
                stats.p99,
            )
    return out


def _worst(records: Sequence[LatencyRecord], top: int, *, as_json: bool) -> list[dict[str, Any]]:
    """N bản ghi chậm nhất — để mở file log gốc ra soi đúng thời điểm đó."""
    scored = [(d, r) for r in records if (d := r.delta(END_TO_END)) is not None]
    scored.sort(key=lambda item: item[0], reverse=True)
    rows = [
        {
            "e2e_ms": round(value, 2),
            "cam_id": record.cam_id,
            "frame_id": record.frame_id,
            "global_id": record.global_id,
            "window_n": record.window_n,
            "db_flushed": record.db_flushed,
        }
        for value, record in scored[:top]
    ]
    if not as_json and rows:
        log.info("")
        log.info("%d bản ghi chậm nhất:", len(rows))
        for row in rows:
            log.info(
                "  %9.1f ms  %s frame=%d gid=%d cửa_sổ=%d db_flush=%s",
                row["e2e_ms"],
                row["cam_id"],
                row["frame_id"],
                row["global_id"],
                row["window_n"],
                row["db_flushed"],
            )
    return rows


def _warnings(
    records: Sequence[LatencyRecord],
    stats: dict[str, Stats | None],
    e2e: Sequence[float],
    *,
    as_json: bool,
) -> list[str]:
    """Những thứ làm con số ở trên ĐỌC SAI nếu không biết — in ra chứ không giấu."""
    out: list[str] = []

    n_probe = sum(1 for r in records if r.t0_source == "probe")
    if n_probe:
        out.append(
            f"{n_probe}/{len(records)} bản ghi có t0 lấy từ đường lui (không có "
            "ntp_timestamp): đoạn `t1-t0` của chúng luôn = 0 ms, tức CHƯA ĐO ĐƯỢC phần "
            "detect+track+ReID, không phải phần đó tức thời. Bật attach-sys-ts trên "
            "nvstreammux hoặc dùng nguồn RTSP có RTCP."
        )
    for segment in ALL_SEGMENTS:
        stat = stats.get(segment.name)
        if stat is not None and stat.n_negative:
            kind = (
                "lệch đồng hồ giữa hai máy (hoặc lượng tử hoá 1 ms của entry ID Redis)"
                if segment.cross_host
                else "LỖI LOGIC ĐO"
            )
            out.append(
                f"đoạn `{segment.name.strip()}` có {stat.n_negative}/{stat.n} mẫu âm — {kind}."
            )
    if e2e:
        p90 = percentile(e2e, 0.90)
        out.append(
            f"mục tiêu đề cương < {TARGET_MS:.0f} ms: "
            f"{'ĐẠT' if p90 < TARGET_MS else 'KHÔNG ĐẠT'} (p90 = {p90:.1f} ms)"
        )

    if not as_json and out:
        log.info("")
        for line in out:
            log.info("! %s", line)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--log", default=None, help="file JSONL (mặc định lấy MCT_LATENCY_LOG)")
    p.add_argument(
        "--run",
        default="last",
        help="'last' (mặc định) | 'all' | một run_id cụ thể — file mở kiểu nối thêm",
    )
    p.add_argument(
        "--by",
        default="",
        choices=["", "cam_id", "global_id", "db_flushed", "t0_source", "final_flush"],
        help="tách end-to-end theo trường này",
    )
    p.add_argument("--top", type=int, default=0, help="liệt kê N bản ghi chậm nhất")
    p.add_argument("--json", action="store_true", help="in JSON thay vì bảng")
    args = p.parse_args(argv)

    path = latency_log_path(args.log)
    if not path:
        log.error("chưa có file log: truyền --log hoặc đặt MCT_LATENCY_LOG")
        return 2
    if not Path(path).is_file():
        log.error("không thấy file %s — engine đã chạy với --latency-log chưa?", path)
        return 2

    records = select_run(read_records(path), args.run)
    if not records:
        log.error("%s không có bản ghi nào đọc được", path)
        return 1

    payload = report(records, top=args.top, group_by=args.by, as_json=args.json)
    if args.json:
        # print() có chủ ý: JSON phải ra stdout sạch để pipe sang jq, còn log đi stderr.
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
