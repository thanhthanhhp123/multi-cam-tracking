"""Đóng dấu thời gian tại từng điểm chuyển giao của pipeline — chỉ ĐO, không sửa logic.

**Vì sao cần.** `tools/measure_latency.py` đo được đúng một con số: `now - ts_ms` lúc
cập nhật Global ID xuất hiện trên `mct:global`. Con số đó nói "chậm", không nói "chậm ở
đâu". Khi trung vị 40 ms mà p90 lên 2.1 s thì đuôi trễ nằm ở MỘT khâu cụ thể, và không
tách được từng khâu thì mọi phỏng đoán đều không bác bỏ được.

**Các mốc.** Tên khoá là hằng trong file này, không rải chuỗi ma thuật khắp nơi:

    t0  capture      khung hình rời camera/NVDEC            (đồng hồ máy GPU)
    t1  probe        DeepStream xong: detect+track+ReID     (đồng hồ máy GPU)
    t1b dequeue      luồng nền của publisher nhấc khỏi hàng đợi (đồng hồ máy GPU)
    t2  xadd         Redis ghi entry vào stream             (đồng hồ máy Redis)
    t3a recv         engine nhận được message               (đồng hồ máy engine)
    t3w window       cửa sổ gán bắt đầu chạy                (đồng hồ máy engine)
    t3  assoc        gán Global ID xong, TRƯỚC khi ghi DB   (đồng hồ máy engine)
    t3d db           `Store.record_many` trả về             (đồng hồ máy engine)
    t4  out          đã đẩy lên `mct:global` cho dashboard  (đồng hồ máy engine)

`t0`, `t1`, `t1b` đi kèm message (`FrameMessage.stamps`) nên vượt được ranh giới tiến
trình; `t2` KHÔNG cần trường riêng vì entry ID của Redis (`<ms>-<seq>`) chính là thời
điểm server ghi entry — chính xác hơn bất kỳ dấu nào ta tự gắn, và lấy được ở phía đọc.

**BA ĐỒNG HỒ KHÁC NHAU.** Cột "đồng hồ" ở trên không phải chú thích cho vui: pipeline
chạy trên `vast-gpu`, engine chạy trên máy dev, Redis ở một trong hai. Mọi đoạn nằm gọn
trong một đồng hồ (`t1-t0`, `t3-t3a`) là số đo thật; mọi đoạn bắc cầu hai máy
(`t2-t1b`, `t3a-t2`) mang theo cả độ lệch đồng hồ giữa hai máy đó. `latency_report.py`
đếm riêng số mẫu ÂM cho từng đoạn — đoạn âm là bằng chứng lệch đồng hồ, không phải
bằng chứng "nhanh". Chạy cả hai đầu trên cùng một máy thì không có vấn đề này.

Module này KHÔNG chạm Redis, KHÔNG cần GPU, và không phụ thuộc numpy: nó nằm ở
`src/common/` nên phải nạp được ở mọi nơi (CLAUDE.md §2 quy tắc 1).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Tên mốc — dùng làm khoá trong FrameMessage.stamps và trong LatencyRecord
# --------------------------------------------------------------------------- #

T0_CAPTURE = "t0_capture_ms"
T1_PROBE = "t1_probe_ms"
T1B_DEQUEUE = "t1b_dequeue_ms"
T2_XADD = "t2_xadd_ms"
T3A_RECV = "t3a_recv_ms"
T3W_WINDOW = "t3w_window_ms"
T3_ASSOC = "t3_assoc_ms"
T3D_DB = "t3d_db_ms"
T4_OUT = "t4_out_ms"

T0_IS_NTP = "t0_is_ntp"
"""1.0 = `t0` lấy từ `ntp_timestamp` thật; 0.0 = đường lui (bằng chính `t1`).

Là cờ nhưng để trong cùng dict float thay vì thêm một trường schema riêng: `stamps` chỉ
có một kiểu giá trị nên msgpack/JSONL không phải xử lý ngoại lệ nào, và bên đọc chỉ cần
`t0_source_from()`.
"""

MESSAGE_STAMPS = (T0_CAPTURE, T1_PROBE, T1B_DEQUEUE, T2_XADD, T3A_RECV)
"""Các mốc đi kèm message. `t2`/`t3a` do phía ĐỌC gắn (xem `FrameConsumer._parse`)."""


def t0_source_from(stamps: dict[str, float]) -> str:
    """`"ntp"` / `"probe"` / `""` (không có `t0`) — xem `T0_IS_NTP` và `capture_ms()`."""
    if T0_CAPTURE not in stamps:
        return ""
    return "ntp" if float(stamps.get(T0_IS_NTP, 0.0)) >= 0.5 else "probe"


NEG_TOL_MS = 0.05
"""Dưới ngưỡng này thì một đoạn ÂM là nhiễu đo, không phải bằng chứng.

Hai mốc liền nhau trong cùng một tiến trình có thể lệch cỡ micro-giây theo chiều sai chỉ
vì `to_json()` làm tròn tới 3 chữ số thập phân, cộng với độ phân giải của `time.time()`
trên Windows. Đếm cả những cái đó vào `n_negative` thì báo cáo hô "LỖI LOGIC ĐO" ở mọi
lần chạy bình thường, và một cảnh báo luôn bật là một cảnh báo không ai đọc. 50 µs nhỏ
hơn ba bậc so với thứ nhỏ nhất đang đo (mili-giây), nên không giấu được lệch đồng hồ thật.
"""


@dataclass(slots=True, frozen=True)
class Segment:
    """Một đoạn giữa hai mốc. `name` là thứ hiện trong báo cáo."""

    name: str
    start: str
    end: str
    note: str
    cross_host: bool = False
    """True = hai đầu nằm trên hai máy khác nhau khi chạy phân tán → dính lệch đồng hồ."""

    @property
    def neg_tolerance_ms(self) -> float:
        """Dưới ngưỡng này thì giá trị âm là nhiễu ĐO, không phải bằng chứng.

        Đoạn có một đầu là `t2` được nới lên 1 ms vì entry ID của Redis chỉ mang
        mili-giây NGUYÊN: `t2` là phần nguyên bị cắt xuống, còn đầu kia có phần thập
        phân, nên hiệu có thể âm tới gần 1 ms ngay cả khi thời gian thật là dương. Đo
        được 2026-09-18: `t2-t1b` có 556/1018 mẫu âm với trung vị −0.1 ms **dù Redis
        chạy cùng máy** — toàn bộ là lượng tử hoá, không có đồng hồ nào lệch cả.
        """
        return 1.0 if T2_XADD in (self.start, self.end) else NEG_TOL_MS


COARSE_SEGMENTS: tuple[Segment, ...] = (
    Segment("t1-t0  deepstream", T0_CAPTURE, T1_PROBE, "detect + track + ReID"),
    Segment("t2-t1  push_redis", T1_PROBE, T2_XADD, "hàng đợi + XADD", cross_host=True),
    Segment("t3-t2  engine", T2_XADD, T3_ASSOC, "nhận + chờ cửa sổ + gán", cross_host=True),
    Segment("t4-t3  db+dashboard", T3_ASSOC, T4_OUT, "ghi SQLite + XADD mct:global"),
)
"""Bốn đoạn của đề bài. Tổng của chúng = end-to-end."""

FINE_SEGMENTS: tuple[Segment, ...] = (
    Segment("  t1b-t1 queue_wait", T1_PROBE, T1B_DEQUEUE, "nằm trong QueuedFramePublisher"),
    Segment("  t2-t1b xadd", T1B_DEQUEUE, T2_XADD, "vòng gọi Redis", cross_host=True),
    Segment("  t3a-t2 redis_pickup", T2_XADD, T3A_RECV, "XREADGROUP nhấc lên", cross_host=True),
    Segment(
        "  t3w-t3a window_wait",
        T3A_RECV,
        T3W_WINDOW,
        "chờ cửa sổ đóng + chờ tracklet hết idle_timeout",
    ),
    Segment("  t3-t3w associate", T3W_WINDOW, T3_ASSOC, "Hungarian + affinity"),
    Segment("  t3d-t3 db_write", T3_ASSOC, T3D_DB, "Store.record_many (ghi theo lô)"),
    Segment("  t4-t3d publish", T3D_DB, T4_OUT, "XADD mct:global"),
)
"""Đoạn nhỏ bên trong các đoạn thô — chỗ thật sự chỉ ra khâu nào tạo đuôi trễ."""

END_TO_END = Segment("t4-t0  END-TO-END", T0_CAPTURE, T4_OUT, "toàn chuỗi", cross_host=True)

ALL_SEGMENTS: tuple[Segment, ...] = (*COARSE_SEGMENTS, *FINE_SEGMENTS, END_TO_END)


# --------------------------------------------------------------------------- #
# Lấy mốc thời gian
# --------------------------------------------------------------------------- #


def now_ms() -> float:
    """Wall clock, epoch milliseconds, giữ phần thập phân.

    KHÔNG dùng `time.monotonic`: các mốc phải so được giữa hai tiến trình (và hai máy),
    mà đồng hồ monotonic thì mỗi tiến trình một gốc. Đổi lại là phải chịu NTP nhảy —
    đúng thứ `latency_report.py` cảnh báo bằng số mẫu âm.
    """
    return time.time() * 1000.0


# Khoảng epoch-ms coi là hợp lệ: 2020-01-01 .. 2100-01-01. Dùng để phân biệt "mốc thật"
# với 0 / giá trị rác — DeepStream trả ntp_timestamp = 0 khi nguồn không có RTCP.
_MIN_PLAUSIBLE_MS = 1_577_836_800_000.0
_MAX_PLAUSIBLE_MS = 4_102_444_800_000.0


def is_plausible_ms(value: float | None) -> bool:
    """Giá trị có ra dáng epoch milliseconds không (không phải 0, không phải giây, không NaN)."""
    if value is None:
        return False
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return False
    return _MIN_PLAUSIBLE_MS <= value <= _MAX_PLAUSIBLE_MS


def capture_ms(ntp_timestamp_ns: int | None, fallback_ms: float) -> tuple[float, str]:
    """Mốc `t0` từ `NvDsFrameMeta.ntp_timestamp`, có đường lui rõ ràng.

    Trả `(giá trị, nguồn)` với nguồn là `"ntp"` hoặc `"probe"`. Ghi lại NGUỒN chứ không
    chỉ giá trị: `t1-t0` đo bằng đường lui luôn bằng 0 (cùng một lần gọi đồng hồ), và
    một cột 0 ms phải đọc được là "chưa đo được", không phải "không tốn thời gian".

    `ntp_timestamp` là nanosecond kể từ epoch, do streammux gắn khi `attach-sys-ts=1`
    (giờ hệ thống lúc nhận buffer) hoặc từ RTCP sender report của camera RTSP. Bằng 0
    khi không có cả hai.
    """
    if ntp_timestamp_ns:
        candidate = float(ntp_timestamp_ns) / 1e6
        if is_plausible_ms(candidate):
            return candidate, "ntp"
    return float(fallback_ms), "probe"


def entry_id_to_ms(entry_id: str | bytes) -> float | None:
    """Thời điểm Redis ghi entry, lấy từ chính entry ID `<ms>-<seq>`.

    Đây là `t2` và nó MIỄN PHÍ: không cần thêm trường nào trên wire, và nó là đồng hồ
    của server Redis chứ không phải của bên ghi — nên đoạn `t2-t1b` đo được cả chi phí
    mạng lẫn thời gian Redis xử lý, thay vì chỉ đo đồng hồ của một bên.
    """
    if isinstance(entry_id, bytes):
        entry_id = entry_id.decode("ascii", "ignore")
    head = entry_id.split("-", 1)[0]
    try:
        value = float(int(head))
    except ValueError:
        return None
    return value if is_plausible_ms(value) else None


# --------------------------------------------------------------------------- #
# Bản ghi + file log
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class LatencyRecord:
    """Một `GlobalUpdate` kèm toàn bộ mốc thời gian dẫn tới nó.

    Mốc `t0..t1b` là của KHUNG MỚI NHẤT đóng góp vào tracklet đó, không phải khung đầu:
    câu hỏi vận hành là "vị trí tôi vừa hiển thị cũ bao nhiêu", chứ không phải "tracklet
    này bắt đầu từ bao giờ".
    """

    run_id: str
    cam_id: str = ""
    frame_id: int = 0
    tracklet_id: int = 0
    global_id: int = 0
    window_n: int = 0
    """Số tracklet được gán trong cùng cửa sổ — cửa sổ to thì `associate` lâu hơn."""

    db_flushed: bool = False
    """Cửa sổ này có commit SQLite thật không (Store ghi theo lô, xem `store.batch_size`)."""

    final_flush: bool = False
    """Bản ghi sinh từ vòng gán CUỐI CÙNG (`Engine.finish()`), khi nguồn đã hết.

    Phải tách ra vì nó là hiện vật của phép đo chứ không phải hành vi vận hành: lúc hết
    nguồn, mọi tracklet còn sống bị đóng một lượt, và `window_wait` của chúng bằng đúng
    thời gian từ khung cuối tới lúc engine chịu dừng — đo 2026-09-18 là **45.3 s**, chính
    là `--idle-limit`. Hệ thống chạy thật không bao giờ có khoảnh khắc đó.
    """

    t0_source: str = ""
    stamps: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "cam_id": self.cam_id,
            "frame_id": int(self.frame_id),
            "tracklet_id": int(self.tracklet_id),
            "global_id": int(self.global_id),
            "window_n": int(self.window_n),
            "db_flushed": bool(self.db_flushed),
            "final_flush": bool(self.final_flush),
            "t0_source": self.t0_source,
            # 3 chữ số thập phân = 1 µs: dư cho thứ đang đo, mà file nhỏ hơn hẳn so với
            # float đầy đủ (một buổi chạy sinh hàng trăm nghìn dòng).
            "stamps": {k: round(float(v), 3) for k, v in self.stamps.items()},
        }
        return json.dumps(payload, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> LatencyRecord:
        data = json.loads(line)
        return cls(
            run_id=str(data.get("run_id", "")),
            cam_id=str(data.get("cam_id", "")),
            frame_id=int(data.get("frame_id", 0)),
            tracklet_id=int(data.get("tracklet_id", 0)),
            global_id=int(data.get("global_id", 0)),
            window_n=int(data.get("window_n", 0)),
            db_flushed=bool(data.get("db_flushed", False)),
            final_flush=bool(data.get("final_flush", False)),
            t0_source=str(data.get("t0_source", "")),
            stamps={str(k): float(v) for k, v in (data.get("stamps") or {}).items()},
        )

    def delta(self, segment: Segment) -> float | None:
        """Độ dài một đoạn, `None` khi thiếu một trong hai mốc."""
        start = self.stamps.get(segment.start)
        end = self.stamps.get(segment.end)
        if start is None or end is None:
            return None
        return float(end) - float(start)


class LatencyLog:
    """Ghi `LatencyRecord` ra JSONL. Mở kiểu NỐI THÊM, không bao giờ xoá file cũ.

    Mỗi lần chạy có `run_id` riêng nên nối thêm không trộn lẫn hai lần đo: công cụ báo
    cáo mặc định chỉ lấy lần chạy MỚI NHẤT. Chọn nối thêm vì mất một lần đo đã chạy tốn
    tiền GPU thì không đo lại được, còn file phình thì xoá tay lúc nào cũng được.

    Có khoá: engine ghi từ luồng chính, nhưng công cụ khác có thể dùng lại lớp này từ
    luồng nền (ví dụ publisher). Chi phí khoá không đáng kể so với một lần `write`.
    """

    def __init__(self, path: str | Path, *, run_id: str | None = None, flush_every: int = 200):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or new_run_id()
        self.flush_every = max(1, int(flush_every))
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._since_flush = 0
        self.n_written = 0

    def write(self, record: LatencyRecord) -> None:
        self.write_many((record,))

    def write_many(self, records: Iterable[LatencyRecord]) -> int:
        count = 0
        with self._lock:
            for record in records:
                self._fh.write(record.to_json())
                self._fh.write("\n")
                count += 1
            self.n_written += count
            self._since_flush += count
            if self._since_flush >= self.flush_every:
                self._fh.flush()
                self._since_flush = 0
        return count

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()

    def __enter__(self) -> LatencyLog:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def new_run_id() -> str:
    """Định danh một lần chạy: đọc được bằng mắt (giờ) + đuôi ngẫu nhiên chống trùng."""
    return time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:4]


def read_records(path: str | Path) -> list[LatencyRecord]:
    """Đọc cả file JSONL. Dòng hỏng bị bỏ qua chứ không làm chết báo cáo.

    Cố tình khoan dung: file này thường được sao chép về từ máy thuê giữa lúc tiến trình
    còn đang ghi, nên dòng cuối cùng cụt là chuyện BÌNH THƯỜNG.
    """
    out: list[LatencyRecord] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(LatencyRecord.from_json(line))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return out


def latency_log_path(explicit: str | None = None) -> str | None:
    """Đường dẫn file log: tham số dòng lệnh thắng, rồi tới `MCT_LATENCY_LOG`, rồi tắt."""
    if explicit:
        return explicit
    return os.environ.get("MCT_LATENCY_LOG") or None


# --------------------------------------------------------------------------- #
# Thống kê
# --------------------------------------------------------------------------- #


def percentile(values: Sequence[float], p: float) -> float:
    """Phân vị theo nội suy tuyến tính (cùng quy ước với `numpy.percentile`).

    Tự viết thay vì gọi numpy để module này nạp được ở mọi nơi; với vài trăm nghìn mẫu
    thì chi phí sắp xếp một lần ở `summarize()` mới là phần đáng kể, không phải chỗ này.
    """
    if not values:
        raise ValueError("percentile() cần ít nhất một giá trị")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = max(0.0, min(1.0, float(p))) * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


@dataclass(slots=True, frozen=True)
class Stats:
    """Thống kê của một đoạn. `n_negative` là cột quan trọng nhất khi chạy phân tán."""

    n: int
    median: float
    p90: float
    p99: float
    mean: float
    minimum: float
    maximum: float
    n_negative: int
    """Số mẫu âm hơn `NEG_TOL_MS` — nhiễu làm tròn không tính vào đây."""

    @property
    def total_ms(self) -> float:
        """Tổng thời gian đoạn này chiếm (dùng để tính tỉ trọng, không phải để báo cáo)."""
        return self.mean * self.n


def summarize(values: Sequence[float], *, neg_tolerance_ms: float = NEG_TOL_MS) -> Stats | None:
    """`None` khi không có mẫu nào — bên gọi in "—" chứ không in số 0 gây hiểu nhầm."""
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    return Stats(
        n=n,
        median=percentile(ordered, 0.50),
        p90=percentile(ordered, 0.90),
        p99=percentile(ordered, 0.99),
        mean=sum(ordered) / n,
        minimum=float(ordered[0]),
        maximum=float(ordered[-1]),
        n_negative=sum(1 for v in ordered if v < -neg_tolerance_ms),
    )


def deltas(records: Iterable[LatencyRecord], segment: Segment) -> list[float]:
    """Mọi giá trị đo được của một đoạn, bỏ qua bản ghi thiếu mốc."""
    out: list[float] = []
    for record in records:
        value = record.delta(segment)
        if value is not None:
            out.append(value)
    return out
