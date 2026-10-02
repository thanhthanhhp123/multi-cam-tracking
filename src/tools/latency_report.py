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
   **Đính chính 2026-09-28:** đuôi đó KHÔNG phải độ trễ chốt danh tính. Danh tính của một
   tracklet được chốt ở vòng gán ĐẦU TIÊN và không bao giờ đổi; đuôi 2–3 s là các bản ghi
   phát lại lúc tracklet ĐÓNG (`kind = close`). Vì vậy báo cáo in riêng độ trễ của các bản
   ghi `kind = first`, và `--by kind` tách ba loại (docs/worklog/2026-09-28-28-*).
3. **Quy trách nhiệm cho đuôi** — lấy riêng các bản ghi nằm trong 10% chậm nhất rồi so
   trung vị từng đoạn của nhóm đó với trung vị chung. Đoạn nào phình ra ở nhóm đuôi
   chính là đoạn tạo ra đuôi. Trung bình toàn cục KHÔNG trả lời được câu này: một đoạn
   chậm 2 s ở 5% số mẫu gần như không nhúc nhích trung vị chung.

**Mục tiêu < 1 s của đề cương** (định nghĩa chốt ở phiên 32, docs/worklog/2026-10-02-32-*)
chấm trên HAI đại lượng, cả hai đo tới lúc kết quả lên `mct:global` (nguồn duy nhất của
dashboard), và p95 của cả hai phải < 1 s:

- **độ trễ theo khung** (`frame_latencies`): khung chụp lúc `c` hiện ra lúc nào. Engine phát
  theo cửa sổ, nên khung đến đầu cửa sổ chờ gần trọn `window_ms`. Bảng end-to-end ở trên
  chỉ đo khung MỚI NHẤT của mỗi lần phát, tức đỉnh tốt nhất của hình răng cưa;
- **thời gian tới Global ID** (`time_to_id`): từ khung đầu tiên của một tracklet tới lần
  đầu nó được phát kèm Global ID.

**Cột `âm`** đếm số mẫu có độ dài âm. Với đoạn nằm trong một máy thì đó là lỗi logic;
với đoạn bắc cầu hai máy (đánh dấu `*`) thì đó là LỆCH ĐỒNG HỒ giữa hai máy — số đo của
đoạn đó lệch đi đúng bằng độ lệch ấy, và không sửa được bằng cách đo lại (xem CLAUDE.md
§11: đồng bộ NTP là điều kiện sống còn).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from common.latency import (
    ALL_SEGMENTS,
    COARSE_SEGMENTS,
    END_TO_END,
    FINE_SEGMENTS,
    T0_CAPTURE,
    T0_FIRST,
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

# Mục tiêu đề cương (CLAUDE.md §7): độ trễ end-to-end < 1 s, chấm ở p95 trên cả độ trễ
# theo khung lẫn thời gian tới Global ID (định nghĩa phiên 32, xem docstring module).
TARGET_MS = 1000.0
TARGET_QUANTILE = 0.95

FRAME_STEP_MS = 10.0
"""Bước lấy mẫu độ trễ theo khung (trần) — tự thu nhỏ để mỗi nhịp phát có ≥ 10 mẫu."""

GAP_SLACK = 1.5
"""Khung thuộc một lần phát chỉ được tính lùi tới `GAP_SLACK` × nhịp phát trung vị.

Đủ rộng để không cắt nhịp phát dao động (khung rơi lệch nhịp giới hạn tần số), đủ hẹp để
một quãng vắng dài (tracklet không có detection) không bị tính thành trễ. Quãng vắng ngắn
hơn mức này thì bị tính thừa, tức sai về phía bảo thủ.
"""

_LIVE_KINDS = ("first", "update", "repeat", "position")


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


def with_kinds(records: Sequence[LatencyRecord]) -> list[LatencyRecord]:
    """Điền `kind` cho log cũ (trước 2026-09-28, chưa có trường này).

    Log cũ không phân biệt được `update` với `close`, nhưng suy ra được `first`: danh tính
    chốt ở bản ghi đầu tiên của mỗi `(run_id, tracklet_id)` theo thứ tự ghi. Các bản ghi còn
    lại nhận `repeat`. Log mới đã mang sẵn `kind` thì giữ nguyên.
    """
    seen: set[tuple[str, int]] = set()
    out: list[LatencyRecord] = []
    for record in records:
        key = (record.run_id, record.tracklet_id)
        first = key not in seen
        seen.add(key)
        if record.kind:
            out.append(record)
        else:
            out.append(dataclasses.replace(record, kind="first" if first else "repeat"))
    return out


@dataclasses.dataclass(frozen=True, slots=True)
class FrameLatency:
    """Độ trễ theo khung, dựng lại từ các lần phát liên tiếp của từng tracklet."""

    samples: list[float]
    n_tracklets: int
    n_pairs: int
    update_gap_ms: float
    """Trung vị khoảng cách giữa hai lần phát của cùng một tracklet (≈ `window_ms`)."""


def _live_by_tracklet(records: Sequence[LatencyRecord]) -> list[list[LatencyRecord]]:
    """Các bản ghi mang thông tin MỚI của từng tracklet, theo thứ tự `t4`.

    Bỏ `close` (phát lại khung cũ lúc tracklet đóng) và `final_flush` (vòng gán lúc hết
    nguồn). Log cũ không có `kind = close`: bản ghi `repeat` cuối của một tracklet chính là
    lần phát lại lúc đóng (phiên 28: mọi bản ghi đuôi đều là bản ghi cuối), trừ khi tracklet
    còn mở tới lúc hết nguồn — khi đó bản ghi cuối là `final_flush`.
    """
    groups: dict[tuple[str, int], list[LatencyRecord]] = defaultdict(list)
    for record in with_kinds(records):
        if T0_CAPTURE in record.stamps and T4_OUT in record.stamps:
            groups[(record.run_id, record.tracklet_id)].append(record)
    out: list[list[LatencyRecord]] = []
    for group in groups.values():
        group.sort(key=lambda r: r.stamps[T4_OUT])
        if group[-1].kind == "repeat" and not group[-1].final_flush:
            group = group[:-1]
        live = [r for r in group if r.kind in _LIVE_KINDS and not r.final_flush]
        if live:
            out.append(live)
    return out


def frame_latencies(
    records: Sequence[LatencyRecord], *, step_ms: float = FRAME_STEP_MS
) -> FrameLatency | None:
    """Độ trễ THEO KHUNG: khung chụp lúc `c` hiện ra trên `mct:global` lúc nào.

    Mỗi bản ghi chỉ mang mốc của khung MỚI NHẤT lúc phát. Giữa hai lần phát `a → b` của
    cùng một tracklet, mọi khung chụp trong `(t0_a, t0_b]` hiện ra cùng lúc ở `t4_b`. Độ trễ
    của chúng vì thế trải từ `t4_b - t0_b` (khung mới nhất, con số duy nhất bảng end-to-end
    đo) tới `t4_b - t0_a` (khung ngay sau lần phát trước). Hàm lấy mẫu khoảng đó đều theo
    thời gian, tức giả định khung đến đều. Cũng chính là độ cũ của vị trí đang hiện trên
    dashboard, lấy trung bình theo thời gian. Chỉ phát theo cửa sổ thì nhịp phát là
    `window_ms`; bật đường phát vị trí thì là `publish.position_interval_ms`.

    Giả định khung đến LIÊN TỤC chỉ đúng khi nhịp phát dài hơn hẳn khoảng cách khung. Khi
    hai thứ xấp xỉ nhau (phát mọi khung, hoặc dữ liệu 2 fps như WildTrack) thì thật ra không
    có khung nào nằm giữa hai lần phát, và phép dựng lại thiên CAO tới một khoảng khung —
    sai về phía bảo thủ, nhưng số đó không còn đọc như độ trễ được.

    Hai giới hạn có chủ ý:
    - khoảng bị chặn ở `GAP_SLACK` × nhịp phát trung vị (`update_gap_ms`): tracklet vắng
      lâu (không có detection) thì quãng vắng không có khung nào để mà trễ;
    - khung TRƯỚC lần phát đầu tiên không tính ở đây. Chúng chờ danh tính, và đó là việc của
      `time_to_id`.
    """
    tracks = _live_by_tracklet(records)
    pairs = [pair for live in tracks for pair in pairwise(live)]
    if not pairs:
        return None
    gap = percentile([b.stamps[T4_OUT] - a.stamps[T4_OUT] for a, b in pairs], 0.5)
    step = min(step_ms, max(gap / 10.0, 1.0))
    samples: list[float] = []
    for a, b in pairs:
        t0_b, t4_b = b.stamps[T0_CAPTURE], b.stamps[T4_OUT]
        lower = max(a.stamps[T0_CAPTURE], t0_b - GAP_SLACK * gap)
        capture = t0_b
        while capture > lower:
            samples.append(t4_b - capture)
            capture -= step
    if not samples:
        return None
    return FrameLatency(
        samples=samples, n_tracklets=len(tracks), n_pairs=len(pairs), update_gap_ms=gap
    )


def time_to_id(records: Sequence[LatencyRecord]) -> list[float]:
    """Thời gian tới Global ID: `t4 - t0_first` của bản ghi `first`, mỗi tracklet một mẫu.

    Log trước phiên 32 không có `t0_first` nên trả rỗng; khi đó đo bằng
    `eval.latency_tradeoff` (thời gian dữ liệu) rồi cộng đoạn `t1-t0`.
    """
    return [
        r.stamps[T4_OUT] - r.stamps[T0_FIRST]
        for r in with_kinds(records)
        if r.kind == "first" and not r.final_flush and T0_FIRST in r.stamps and T4_OUT in r.stamps
    ]


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    p95 = percentile(values, TARGET_QUANTILE)
    return {
        "n": len(values),
        "p50_ms": round(percentile(values, 0.50), 2),
        "p90_ms": round(percentile(values, 0.90), 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(percentile(values, 0.99), 2),
        "max_ms": round(max(values), 2),
        "pass": p95 < TARGET_MS,
    }


def target_verdict(records: Sequence[LatencyRecord], *, as_json: bool) -> dict[str, Any]:
    """Chấm mục tiêu < 1 s theo định nghĩa phiên 32 (xem docstring module)."""
    out: dict[str, Any] = {"target_ms": TARGET_MS, "quantile": TARGET_QUANTILE}
    frames = frame_latencies(records)
    if frames is not None:
        out["frame"] = {
            **_quantiles(frames.samples),
            "n_tracklets": frames.n_tracklets,
            "n_pairs": frames.n_pairs,
            "update_gap_ms": round(frames.update_gap_ms, 2),
        }
    ttid = time_to_id(records)
    if ttid:
        out["time_to_id"] = _quantiles(ttid)

    if not as_json:
        log.info("")
        log.info(
            "MỤC TIÊU < %.0f ms ở p%.0f (định nghĩa phiên 32):",
            TARGET_MS,
            TARGET_QUANTILE * 100,
        )
        log.info("%-26s %7s %9s %9s %9s %9s", "", "n", "p50", "p90", "p95", "p99")
        rows = (("frame", "độ trễ theo khung"), ("time_to_id", "thời gian tới Global ID"))
        for key, name in rows:
            row = out.get(key)
            if row is None:
                log.info("%-26s %7s", name, "—")
                continue
            log.info(
                "%-26s %7d %9.1f %9.1f %9.1f %9.1f  → %s",
                name,
                row["n"],
                row["p50_ms"],
                row["p90_ms"],
                row["p95_ms"],
                row["p99_ms"],
                "ĐẠT" if row["pass"] else "KHÔNG ĐẠT",
            )
        if frames is not None:
            log.info(
                "(độ trễ theo khung: %d tracklet, %d cặp lần phát, nhịp phát trung vị %.0f ms)",
                frames.n_tracklets,
                frames.n_pairs,
                frames.update_gap_ms,
            )
    return out


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
    records = with_kinds(records)
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
    payload["target"] = target_verdict(records, as_json=as_json)
    payload["warnings"] = _warnings(records, stats, e2e, payload["target"], as_json=as_json)
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
    """Tách end-to-end theo `cam_id` / `global_id` / `db_flushed` / `t0_source` / `kind`."""
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
    target: dict[str, Any],
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
    if "time_to_id" not in target:
        out.append(
            "log không có mốc t0_first (ghi trước phiên 32) nên KHÔNG chấm được thời gian tới "
            "Global ID — vế thứ hai của mục tiêu. Đo bằng eval.latency_tradeoff (thời gian dữ "
            "liệu) rồi cộng đoạn t1-t0, hoặc ghi lại log bằng engine mới."
        )
    if e2e:
        p50 = percentile(e2e, 0.50)
        p90 = percentile(e2e, 0.90)
        out.append(
            f"bảng end-to-end (p50 = {p50:.1f}, p90 trên MỌI bản ghi = {p90:.1f} ms) KHÔNG phải "
            "số để chấm mục tiêu: nó đo khung MỚI NHẤT của mỗi lần phát (đỉnh tốt nhất của "
            "răng cưa) và gộp cả bản ghi phát lại lúc tracklet đóng; xem mục MỤC TIÊU và --by kind"
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
        choices=["", "cam_id", "global_id", "db_flushed", "t0_source", "final_flush", "kind"],
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
