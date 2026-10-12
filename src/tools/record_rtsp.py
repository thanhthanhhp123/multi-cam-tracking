"""Ghi luồng RTSP của các camera IP ra file trên MỘT máy, kèm số liệu để đồng bộ (M6).

    # trước mỗi buổi quay: thấy camera không, đúng mật khẩu, độ phân giải, fps không
    python src/tools/record_rtsp.py check
    # mỗi buổi quay: ghi mọi camera cùng lúc; Enter hoặc Ctrl+C để dừng
    python src/tools/record_rtsp.py record --session s1

Chạy từ gốc repo. Cách gọi `PYTHONPATH=src python -m tools.record_rtsp ...` cũng được; gọi
thẳng file thì khỏi đặt PYTHONPATH (tiện trên PowerShell của laptop mượn).

**Camera lấy từ `.env`** ở gốc repo (gitignored): mỗi camera một dòng `LAB_RTSP_CAMXX=...`,
vd `LAB_RTSP_CAM01=rtsp://admin:<mật khẩu>@192.168.1.3:554/unicast/c1/s0/live`. Tên camera
theo VỊ TRÍ, khớp `configs/lab/topology.yaml`, không theo IP. Biến môi trường thật đè lên
`.env`; `--camera cam01=rtsp://...` đè lên cả hai. `LAB_FFMPEG=<đường dẫn ffmpeg.exe>` trong
`.env` thay cho việc sửa PATH.

**Vì sao ghi ra file, không cho pipeline đọc RTSP trực tiếp.** Pipeline chạy trên `vast-gpu`,
máy đó không vào được mạng LAN của camera. File còn cho phép chạy lại pipeline nhiều lần trên
cùng dữ liệu (n = 3, CLAUDE.md §7) và đem chính dữ liệu đó lên CVAT. ffmpeg chỉ CHÉP luồng
nén của camera (`-c copy`), không giải mã hay mã hoá lại, nên laptop yếu vẫn ghi được 3–4 luồng.
Định dạng Matroska vì file vẫn mở được nếu tiến trình chết giữa chừng (MP4 thì mất hết).

**Đồng bộ theo đồng hồ của máy ghi, không cần vỗ tay.** Mọi camera được ghi cùng lúc trên cùng
một máy, nên giờ máy ghi NHẬN gói là một trục thời gian chung. ffmpeg báo tiến độ
(`-progress`) mỗi 0,1 s; tại lúc nhận báo cáo `t`, gói mới nhất đã ghi nằm ở giây `p` của file.
Hiệu `t − p` = (giờ máy ghi ứng với giây 0 của file) + độ trễ camera/mạng + độ trễ báo cáo, và
hai phần sau chỉ có thể DƯƠNG. Nên lấy đường bao DƯỚI của `t − p`: min trong từng khoảng `bin_s`
giây, rồi hồi quy tuyến tính qua các min — cùng ý với bộ lọc độ trễ nhỏ nhất của NTP. Cụm gói
dồn lại lúc ffmpeg dò luồng ở đầu file có `t − p` lớn nên tự bị loại.
Độ dốc của đường bao là độ trôi đồng hồ camera so với máy ghi: thạch anh lệch vài chục ppm, tức
vài chục ms trên một buổi vài phút. Độ lệch được tính ở GIỮA cửa sổ chung, nên sai số do trôi ở
hai mép chỉ còn một nửa, và được báo ra (`edge_error_ms`).

Kết quả là `lag_s` theo đúng nghĩa của `tools.sync_recordings` (sự kiện ở giây `t` của camera
tham chiếu nằm ở giây `t + lag_s` của camera này), in sẵn thành lệnh
`sync_recordings --offset ...`. Từ bước đó trở đi quy trình như cũ (docs/m6/README.md).

Giới hạn, phải ghi trong báo cáo:
- Phần không đo được là chênh lệch độ trễ GIỮA các camera (mã hoá + mạng). Cùng model, cùng cài
  đặt luồng, cùng switch thì chênh lệch nhỏ nhưng không bằng 0; cộng thêm pha chụp lệch nhau
  tới nửa khung (các camera chạy tự do, không khoá khung với nhau).
- Vì vậy cặp CHỒNG LẤN vẫn kiểm bằng mắt: vỗ tay trong vùng chung ở đầu và cuối buổi, hai bàn
  tay phải chạm nhau ở cùng một khung ±1 trên cả hai camera. Camera không chồng lấn không cần:
  ràng buộc thời gian di chuyển đã nới hàng giây.

**Chỉ dùng thư viện chuẩn** (có test canh): laptop mượn để ghi chỉ cần Python ≥ 3.10 và
ffmpeg ≥ 5, không venv, không pip. Vì thế công cụ không dùng `common.config`/`common.logging`
(kéo theo PyYAML/python-dotenv) và in thẳng ra màn hình như mọi CLI tương tác.

Mật khẩu camera không ra đĩa ở đâu ngoài `.env`: URL trong `record.json` và log ffmpeg (ffmpeg
in lại nguyên URL đầu vào) đều được che.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]

_ENV_KEY = re.compile(r"^LAB_RTSP_(CAM\d+)$", re.IGNORECASE)
_SESSION = re.compile(r"^[A-Za-z0-9_-]+$")
_URL_CREDS = re.compile(r"(rtsps?://[^:/@\s'\"]+):[^@\s'\"]+@", re.IGNORECASE)
DEFAULT_PORT = 554
MIN_FFMPEG_MAJOR = 5
"""`-timeout` của RTSP đổi nghĩa ở ffmpeg 5: trước đó nó là thời gian LẮNG NGHE (ffmpeg tự làm
server), thời gian chờ socket là `-stimeout`. Bản cũ hơn sẽ treo thay vì báo mất camera."""
SOCKET_TIMEOUT_US = 5_000_000
BITRATE_BUDGET_MBPS = 8.0
"""Ước lượng trên cao cho một luồng 1080p25 CBR, chỉ để kiểm dung lượng đĩa."""

Runner = Callable[[Sequence[str], float], "subprocess.CompletedProcess[bytes]"]
Reach = Callable[[str, int], "str | None"]


def _run(cmd: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(list(cmd), capture_output=True, check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(list(cmd), -1, b"", b"timed out")
    except OSError as exc:
        return subprocess.CompletedProcess(list(cmd), -1, b"", str(exc).encode())


# --------------------------------------------------------------------------------------
# Camera: .env, che mật khẩu
# --------------------------------------------------------------------------------------


def read_env_file(path: Path) -> dict[str, str]:
    """`.env` tối giản: `KEY=VALUE`, bỏ dòng trống/`#`, bỏ `export ` và cặp ngoặc bao ngoài.

    Không cắt chú thích cuối dòng: mật khẩu camera có thể chứa `#`. Đọc được cả UTF-16, vì
    `echo ... > .env` của Windows PowerShell 5.1 ghi ra UTF-16 — trên laptop mượn, không ai
    kiểm hộ chuyện đó.
    """
    if not path.is_file():
        return {}
    raw = path.read_bytes()
    text = (
        raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    )
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key] = value
    return out


def cameras_from_env(env: dict[str, str]) -> dict[str, str]:
    """`LAB_RTSP_CAM01=url` → `{"cam01": url}`, sắp theo tên camera."""
    cams: dict[str, str] = {}
    for key, value in env.items():
        m = _ENV_KEY.match(key)
        if m and value.strip():
            cams[m.group(1).lower()] = value.strip()
    return dict(sorted(cams.items()))


def merged_env(env_file: Path, environ: dict[str, str] | None = None) -> dict[str, str]:
    """`.env` rồi đè bằng biến môi trường thật — cùng thứ tự ưu tiên với `common.config`."""
    return {**read_env_file(env_file), **(dict(os.environ) if environ is None else environ)}


def redact(text: str, secrets: Sequence[str] = ()) -> str:
    """Che phần `user:pass@` của mọi URL RTSP, và mọi chỗ chuỗi mật khẩu hiện ra."""
    text = _URL_CREDS.sub(r"\1:***@", text)
    for secret in secrets:
        if len(secret) >= 3:
            text = text.replace(secret, "***")
    return text


def url_secrets(urls: Sequence[str]) -> list[str]:
    out: list[str] = []
    for url in urls:
        password = urlsplit(url).password
        if password:
            out.extend({password, unquote(password)})
    return out


_SPECIAL_HINT = (
    "mật khẩu có ký tự đặc biệt thì phải viết dạng mã hoá trong URL: # → %23, @ → %40, "
    "/ → %2F, ? → %3F, : → %3A"
)


def host_port(url: str) -> tuple[str, int]:
    """(host, cổng) của URL RTSP; báo lỗi rõ ràng khi mật khẩu làm hỏng cú pháp URL.

    `#`, `/`, `?`, `@` chưa mã hoá trong mật khẩu cắt URL ở chỗ sai: host đọc ra thành tên
    đăng nhập, phần còn lại thành đường dẫn. ffmpeg khi đó báo lỗi kết nối khó hiểu.
    """
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("rtsp", "rtsps"):
        raise ValueError(f"không phải URL RTSP: {redact(url)}")
    if parts.fragment or parts.netloc.count("@") > 1 or ("@" in url and "@" not in parts.netloc):
        raise ValueError(_SPECIAL_HINT)
    try:
        port = parts.port
    except ValueError:
        raise ValueError(_SPECIAL_HINT) from None
    if not parts.hostname:
        raise ValueError(f"URL thiếu địa chỉ camera: {redact(url)}")
    return parts.hostname, port or DEFAULT_PORT


def tcp_reachable(host: str, port: int, timeout_s: float = 2.0) -> str | None:
    """None nếu mở được kết nối TCP, ngược lại là lý do."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return None
    except OSError as exc:
        return str(exc) or exc.__class__.__name__


# --------------------------------------------------------------------------------------
# Kiểm trước buổi quay
# --------------------------------------------------------------------------------------

LEVELS = ("OK", "WARN", "FAIL")


@dataclass
class Report:
    """Cùng định dạng với `tools.check_lab_setup` (không import được: nó cần PyYAML)."""

    lines: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, level: str, topic: str, message: str) -> None:
        assert level in LEVELS
        self.lines.append((level, topic, message))

    @property
    def n_fail(self) -> int:
        return sum(1 for lv, _, _ in self.lines if lv == "FAIL")

    @property
    def n_warn(self) -> int:
        return sum(1 for lv, _, _ in self.lines if lv == "WARN")

    def render(self) -> str:
        return "\n".join(f"[{lv:4}] {topic}: {msg}" for lv, topic, msg in self.lines)


@dataclass(frozen=True, slots=True)
class StreamInfo:
    codec: str
    width: int
    height: int
    fps: float
    """`r_frame_rate` mà ffprobe đoán từ dấu thời gian; 0 nếu không đoán được."""
    has_audio: bool


def _rate(text: str | None) -> float:
    try:
        num, _, den = (text or "0").partition("/")
        return float(num) / float(den or 1) if float(den or 1) else 0.0
    except ValueError:
        return 0.0


def parse_stream_probe(data: dict) -> StreamInfo:
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("luồng không có video")
    return StreamInfo(
        codec=str(video.get("codec_name", "?")),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_rate(video.get("r_frame_rate")),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def explain_ffmpeg_error(stderr: str) -> str:
    low = stderr.lower()
    if "401" in low or "unauthorized" in low:
        return "sai tên đăng nhập/mật khẩu (401)"
    if "404" in low or "not found" in low:
        return "sai đường dẫn luồng (404), kiểm phần /unicast/c1/s0/live"
    if "timed out" in low or "timeout" in low:
        return "camera không gửi dữ liệu trong 5 s"
    if "refused" in low:
        return "camera từ chối kết nối (cổng RTSP tắt?)"
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "lỗi không rõ"


def probe_stream(url: str, *, ffprobe: str, runner: Runner, timeout_s: float = 20.0) -> StreamInfo:
    cmd = [ffprobe, "-v", "error", "-rtsp_transport", "tcp", "-timeout", str(SOCKET_TIMEOUT_US)]
    proc = runner([*cmd, "-show_streams", "-of", "json", url], timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(explain_ffmpeg_error(proc.stderr.decode("utf-8", "replace")))
    return parse_stream_probe(json.loads(proc.stdout or b"{}"))


def ffmpeg_version(ffmpeg: str, runner: Runner) -> tuple[str, int | None]:
    """(dòng đầu của `ffmpeg -version`, số phiên bản chính hoặc None nếu là bản git)."""
    proc = runner([ffmpeg, "-hide_banner", "-version"], 15.0)
    if proc.returncode != 0:
        return "", None
    lines = proc.stdout.decode("utf-8", "replace").splitlines()
    first = lines[0] if lines else ""
    m = re.search(r"version\s+n?(\d+)\.", first)
    return first, (int(m.group(1)) if m else None)


def _sibling_tool(ffmpeg: str, name: str) -> str:
    """ffprobe nằm cạnh ffmpeg khi ffmpeg được chỉ bằng đường dẫn (bản static giải nén)."""
    path = Path(ffmpeg)
    if path.parent == Path("."):
        return name
    return str(path.with_name(name + path.suffix))


def _resolve_tool(tool: str) -> str | None:
    return shutil.which(tool) or (tool if Path(tool).is_file() else None)


def run_check(
    cameras: dict[str, str],
    *,
    ffmpeg: str,
    ffprobe: str,
    out_root: Path,
    expect_size: tuple[int, int],
    expect_fps: float,
    runner: Runner = _run,
    reach: Reach = tcp_reachable,
) -> Report:
    rep = Report()
    secrets = url_secrets(list(cameras.values()))

    if not cameras:
        rep.add(
            "FAIL", "camera", "không có camera nào: thêm dòng LAB_RTSP_CAM01=rtsp://... vào .env"
        )
    else:
        rep.add("OK", "camera", f"{len(cameras)} camera: {', '.join(cameras)}")
    seen: dict[tuple[str, int, str], str] = {}
    for cam, url in cameras.items():
        try:
            host, port = host_port(url)
        except ValueError as exc:
            rep.add("FAIL", cam, str(exc))
            continue
        key = (host, port, urlsplit(url).path)
        if key in seen:
            rep.add(
                "FAIL",
                cam,
                f"cùng luồng với {seen[key]} ({host}): chép dòng .env quên đổi IP? Hai camera "
                "trùng nhau thì đồng bộ trông hoàn hảo mà dữ liệu vô dụng",
            )
        seen.setdefault(key, cam)

    ffmpeg_path = _resolve_tool(ffmpeg)
    if ffmpeg_path is None:
        rep.add(
            "FAIL",
            "ffmpeg",
            f"không tìm thấy {ffmpeg!r}: tải bản static (gyan.dev, 'release essentials'), giải "
            "nén, rồi đặt LAB_FFMPEG=<...>\\bin\\ffmpeg.exe trong .env",
        )
    else:
        first, major = ffmpeg_version(ffmpeg_path, runner)
        if major is None:
            rep.add(
                "WARN", "ffmpeg", f"không đọc được số phiên bản ({first or 'không chạy'}), cần ≥ 5"
            )
        elif major < MIN_FFMPEG_MAJOR:
            rep.add("FAIL", "ffmpeg", f"{first}: cần ≥ {MIN_FFMPEG_MAJOR} (nghĩa của -timeout)")
        else:
            rep.add("OK", "ffmpeg", first)
    ffprobe_path = _resolve_tool(ffprobe)
    if ffprobe_path is None:
        rep.add("FAIL", "ffprobe", f"không tìm thấy {ffprobe!r} (nằm cùng thư mục với ffmpeg)")

    out_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out_root).free / 1e9
    need_gb = max(1, len(cameras)) * BITRATE_BUDGET_MBPS / 8 * 30 * 60 / 1000
    level = "FAIL" if free_gb < need_gb else "WARN" if free_gb < 2 * need_gb else "OK"
    rep.add(level, "đĩa", f"trống {free_gb:.1f} GB ở {out_root}; 30 phút ghi cần ~{need_gb:.1f} GB")

    infos: dict[str, StreamInfo] = {}
    for cam, url in cameras.items():
        try:
            host, port = host_port(url)
        except ValueError:
            continue
        why = reach(host, port)
        if why is not None:
            rep.add(
                "FAIL",
                cam,
                f"không kết nối được {host}:{port} ({redact(why, secrets)}). Kiểm dây/nguồn "
                "camera, và IP của máy ghi phải cùng dải (vd 192.168.1.100, mask 255.255.255.0)",
            )
            continue
        if ffprobe_path is None:
            continue
        try:
            info = probe_stream(url, ffprobe=ffprobe_path, runner=runner)
        except (RuntimeError, ValueError) as exc:
            rep.add("FAIL", cam, f"{host}: {redact(str(exc), secrets)}")
            continue
        infos[cam] = info
        desc = f"{host}: {info.codec} {info.width}x{info.height} {info.fps:g} fps, " + (
            "có audio" if info.has_audio else "không audio"
        )
        problems = []
        if (info.width, info.height) != expect_size:
            problems.append(
                f"cần {expect_size[0]}x{expect_size[1]}: đặt lại luồng chính trên trang quản lý "
                "camera (hoặc --scale khi đồng bộ, rồi sửa nốt topology/ground_points)"
            )
        if info.fps <= 0:
            problems.append("ffprobe không đoán được fps; xem fps đo được sau lần ghi thử")
        elif abs(info.fps - expect_fps) / expect_fps > 0.02:
            problems.append(f"cần {expect_fps:g} fps")
        rep.add("WARN" if problems else "OK", cam, "; ".join([desc, *problems]))

    if len({(i.width, i.height, round(i.fps, 1)) for i in infos.values()}) > 1:
        rep.add(
            "WARN", "camera", "các camera khác nhau về độ phân giải/fps: nên đặt giống hệt nhau"
        )
    return rep


# --------------------------------------------------------------------------------------
# Tiến độ ffmpeg → đồng hồ → độ lệch
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProgressSample:
    t_s: float
    """Giờ máy ghi lúc nhận xong khối báo cáo (giây, tính từ lúc bắt đầu buổi)."""
    frame: int
    out_time_s: float
    """Vị trí (giây, theo dấu thời gian của camera, từ đầu file) của gói mới nhất đã ghi."""


class ProgressParser:
    """Gom các dòng `key=value` của `ffmpeg -progress` thành khối; khối kết thúc ở `progress=`."""

    def __init__(self) -> None:
        self._block: dict[str, str] = {}
        self.ended = False

    def feed(self, line: str, t_s: float) -> ProgressSample | None:
        if "=" not in line:
            return None
        key, value = line.split("=", 1)
        if key != "progress":
            self._block[key.strip()] = value.strip()
            return None
        block, self._block = self._block, {}
        self.ended = value.strip() == "end"
        try:
            frame = int(block.get("frame", "0"))
            out_us = int(block.get("out_time_us", "N/A"))
        except ValueError:
            return None
        if frame <= 0 or out_us < 0:
            return None
        return ProgressSample(t_s=t_s, frame=frame, out_time_s=out_us / 1e6)


@dataclass(frozen=True, slots=True)
class ClockFit:
    offset_s: float
    """Giờ máy ghi ứng với giây 0 của file (cộng độ trễ nhỏ nhất của camera/mạng)."""
    drift: float
    """d(giờ máy ghi)/d(giờ file) − 1; nhân 1e6 ra ppm."""
    jitter_ms: float
    """Trung vị khoảng cách từ mẫu tới đường bao: độ ồn của mạng/máy ghi."""
    n_samples: int
    n_bins: int

    def wall_at(self, file_s: float) -> float:
        return self.offset_s + (1.0 + self.drift) * file_s

    def file_at(self, wall_s: float) -> float:
        return (wall_s - self.offset_s) / (1.0 + self.drift)


def fit_clock(
    samples: Sequence[ProgressSample],
    *,
    bin_s: float = 5.0,
    skip_s: float = 2.0,
    min_bins_for_drift: int = 3,
) -> ClockFit:
    """Đường bao dưới của `t − p` theo `p` (docstring module)."""
    usable = [s for s in samples if s.out_time_s >= skip_s] or list(samples)
    if not usable:
        raise ValueError("không có mẫu tiến độ nào: ffmpeg chưa ghi được khung nào")
    lowest: dict[int, ProgressSample] = {}
    for s in usable:
        k = int(s.out_time_s // bin_s)
        best = lowest.get(k)
        if best is None or s.t_s - s.out_time_s < best.t_s - best.out_time_s:
            lowest[k] = s
    pts = sorted((s.out_time_s, s.t_s - s.out_time_s) for s in lowest.values())
    if len(pts) >= min_bins_for_drift:
        n = len(pts)
        mx = sum(p for p, _ in pts) / n
        my = sum(d for _, d in pts) / n
        sxx = sum((p - mx) ** 2 for p, _ in pts)
        slope = sum((p - mx) * (d - my) for p, d in pts) / sxx if sxx > 0 else 0.0
        offset = my - slope * mx
    else:
        slope, offset = 0.0, min(d for _, d in pts)
    resid = sorted((s.t_s - s.out_time_s) - (offset + slope * s.out_time_s) for s in usable)
    return ClockFit(
        offset_s=offset,
        drift=slope,
        jitter_ms=resid[len(resid) // 2] * 1000.0,
        n_samples=len(usable),
        n_bins=len(pts),
    )


@dataclass(frozen=True, slots=True)
class SyncPlan:
    reference: str
    lags_s: dict[str, float]
    """Nghĩa của `tools.sync_recordings`: giây `t` của tham chiếu = giây `t + lag` của camera."""
    window_s: tuple[float, float]
    """Cửa sổ mọi camera cùng có hình, theo giờ máy ghi."""
    edge_error_ms: dict[str, float]
    """|độ lệch ở mép cửa sổ − độ lệch ở giữa|: sai số còn lại do trôi đồng hồ."""


def plan_sync(fits: dict[str, ClockFit], durations: dict[str, float], reference: str) -> SyncPlan:
    start = max(f.wall_at(0.0) for f in fits.values())
    end = min(fits[c].wall_at(durations[c]) for c in fits)
    if end <= start:
        raise ValueError("các camera không có đoạn ghi chung")

    def lag_at(cam: str, wall: float) -> float:
        return fits[cam].file_at(wall) - fits[reference].file_at(wall)

    mid = (start + end) / 2
    lags = {c: lag_at(c, mid) for c in fits}
    edge = {
        c: 1000.0 * max(abs(lag_at(c, start) - lags[c]), abs(lag_at(c, end) - lags[c]))
        for c in fits
    }
    return SyncPlan(reference=reference, lags_s=lags, window_s=(start, end), edge_error_ms=edge)


def sync_command(files: dict[str, str], plan: SyncPlan, *, out_dir: str, fps: float) -> list[str]:
    cmd = ["python", "-m", "tools.sync_recordings"]
    for cam, path in files.items():
        cmd += ["--video", f"{cam}={path}"]
    cmd += ["--reference", plan.reference]
    for cam, lag in plan.lags_s.items():
        if cam != plan.reference:
            cmd += ["--offset", f"{cam}={lag:+.4f}"]
    return [*cmd, "--out-dir", out_dir, "--fps", f"{fps:g}", "--execute"]


# --------------------------------------------------------------------------------------
# Ghi
# --------------------------------------------------------------------------------------


def record_cmd(
    url: str, out_file: Path, ffmpeg_cmd: Sequence[str], *, transport: str = "tcp"
) -> list[str]:
    """ffmpeg chép luồng video đầu tiên, báo tiến độ ra stdout mỗi 0,1 s.

    `-rtsp_transport tcp`: mất gói UDP làm vỡ khung mà không báo gì; TCP thì chậm lại chứ
    không vỡ. `-timeout`: camera rút dây thì ffmpeg thoát sau 5 s thay vì treo mãi.
    """
    return [
        *ffmpeg_cmd,
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "info",
        "-y",
        "-rtsp_transport",
        transport,
        "-timeout",
        str(SOCKET_TIMEOUT_US),
        "-i",
        url,
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-progress",
        "pipe:1",
        "-stats_period",
        "0.1",
        "-f",
        "matroska",
        str(out_file),
    ]


class _CameraRecorder:
    def __init__(
        self,
        cam: str,
        url: str,
        out_dir: Path,
        *,
        ffmpeg_cmd: Sequence[str],
        transport: str,
        secrets: Sequence[str],
        clock: Callable[[], float],
        anchor: float,
    ) -> None:
        self.cam = cam
        self.url = url
        self.file = out_dir / f"{cam}.mkv"
        self.progress_path = out_dir / f"{cam}.progress.csv"
        self.log_path = out_dir / f"{cam}.ffmpeg.log"
        self.samples: list[ProgressSample] = []
        self.last_advance: float | None = None
        self.tail: deque[str] = deque(maxlen=5)
        self._cmd = record_cmd(url, self.file, ffmpeg_cmd, transport=transport)
        self._secrets = secrets
        self._clock = clock
        self._anchor = anchor
        self._threads: list[threading.Thread] = []
        self.proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        # Nhóm tiến trình riêng: Ctrl+C chỉ tới công cụ này, rồi công cụ dừng từng ffmpeg bằng
        # 'q' để file được đóng tử tế — không phải ffmpeg nào nhận tín hiệu trước dừng trước.
        kwargs: dict = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self.proc = subprocess.Popen(self._cmd, **kwargs)
        for target in (self._read_progress, self._read_log):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)

    def _read_progress(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        parser = ProgressParser()
        with self.progress_path.open("w", encoding="utf-8", newline="") as f:
            f.write("t_s,frame,out_time_s\n")
            for raw in self.proc.stdout:
                t = self._clock() - self._anchor
                sample = parser.feed(raw.decode("utf-8", "replace").strip(), t)
                if sample is None:
                    continue
                if not self.samples or sample.out_time_s > self.samples[-1].out_time_s:
                    self.last_advance = t
                self.samples.append(sample)
                f.write(f"{sample.t_s:.6f},{sample.frame},{sample.out_time_s:.6f}\n")

    def _read_log(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        with self.log_path.open("w", encoding="utf-8") as f:
            for raw in self.proc.stderr:
                text = redact(raw.decode("utf-8", "replace"), self._secrets)
                f.write(text)
                f.flush()
                if text.strip():
                    self.tail.append(text.strip())

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def request_stop(self) -> None:
        if self.proc is None or self.proc.stdin is None:
            return
        try:
            self.proc.stdin.write(b"q")
            self.proc.stdin.flush()
            self.proc.stdin.close()
        except (OSError, ValueError):
            pass

    def finish(self, deadline: float) -> int | None:
        if self.proc is None:
            return None
        try:
            self.proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        for thread in self._threads:
            thread.join(timeout=5)
        return self.proc.returncode

    def fps_recent(self, window_s: float = 2.0) -> float:
        if len(self.samples) < 2:
            return 0.0
        last = self.samples[-1]
        first = next((s for s in self.samples if s.t_s >= last.t_s - window_s), self.samples[0])
        dt = last.t_s - first.t_s
        return (last.frame - first.frame) / dt if dt > 0 else 0.0


@dataclass
class SessionResult:
    ok: bool
    stopped_by: str
    cameras: dict[str, dict]
    errors: list[str]
    warnings: list[str]
    sync: dict | None
    record_json: Path


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _rel_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def record_session(
    cameras: dict[str, str],
    out_dir: Path,
    *,
    session: str,
    ffmpeg_cmd: Sequence[str],
    fps: float = 25.0,
    duration_s: float | None = None,
    transport: str = "tcp",
    stop: threading.Event | None = None,
    clock: Callable[[], float] = time.perf_counter,
    status_every_s: float = 2.0,
    stall_s: float = 3.0,
    ffmpeg_info: str = "",
    printer: Callable[[str], None] = print,
) -> SessionResult:
    """Ghi mọi camera cùng lúc tới khi `stop`/hết `duration_s`/một camera chết; rồi tính độ lệch."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stop = stop or threading.Event()
    secrets = url_secrets(list(cameras.values()))
    anchor = clock()
    started_at = datetime.now().astimezone()
    recorders = {
        cam: _CameraRecorder(
            cam,
            url,
            out_dir,
            ffmpeg_cmd=ffmpeg_cmd,
            transport=transport,
            secrets=secrets,
            clock=clock,
            anchor=anchor,
        )
        for cam, url in cameras.items()
    }
    for rec in recorders.values():
        rec.start()

    errors: list[str] = []
    warnings: list[str] = []
    stopped_by = "error"
    stalled: set[str] = set()
    next_status = status_every_s
    try:
        while True:
            if stop.wait(0.1):
                stopped_by = "user"
                break
            now = clock() - anchor
            if duration_s is not None and now >= duration_s:
                stopped_by = "duration"
                break
            dead = [rec for rec in recorders.values() if not rec.alive()]
            if dead:
                for rec in dead:
                    rec.finish(time.monotonic() + 2)
                    last = rec.tail[-1] if rec.tail else "không có log"
                    errors.append(
                        f"{rec.cam}: ffmpeg dừng giữa chừng sau {_fmt_elapsed(now)} "
                        f"(mã thoát {rec.proc.returncode if rec.proc else '?'}): {last}"
                    )
                break
            for rec in recorders.values():
                if rec.last_advance is None:
                    if now > 10 and rec.cam not in stalled:
                        stalled.add(rec.cam)
                        printer(f"  !! {rec.cam}: sau 10 s vẫn chưa nhận được khung nào")
                elif now - rec.last_advance > stall_s:
                    if rec.cam not in stalled:
                        stalled.add(rec.cam)
                        printer(f"  !! {rec.cam}: MẤT HÌNH {now - rec.last_advance:.0f} s")
                        warnings.append(
                            f"{rec.cam}: mất hình > {stall_s:g} s lúc {_fmt_elapsed(now)}"
                        )
                else:
                    stalled.discard(rec.cam)
            if now >= next_status:
                next_status += status_every_s
                parts = [
                    f"{rec.cam} {rec.samples[-1].frame if rec.samples else 0} khung "
                    f"{rec.fps_recent():4.1f} fps"
                    for rec in recorders.values()
                ]
                printer(f"  {_fmt_elapsed(now)}  " + " | ".join(parts))
    except KeyboardInterrupt:
        stopped_by = "user"

    for rec in recorders.values():
        rec.request_stop()
    deadline = time.monotonic() + 10
    exit_codes = {cam: rec.finish(deadline) for cam, rec in recorders.items()}

    cams_out: dict[str, dict] = {}
    fits: dict[str, ClockFit] = {}
    durations: dict[str, float] = {}
    for cam, rec in recorders.items():
        item: dict = {
            "url": redact(rec.url, secrets),
            "file": _rel_posix(rec.file),
            "progress_csv": _rel_posix(rec.progress_path),
            "ffmpeg_log": _rel_posix(rec.log_path),
            "exit_code": exit_codes[cam],
            "samples": len(rec.samples),
        }
        if rec.samples:
            first, last = rec.samples[0], rec.samples[-1]
            span = last.out_time_s - first.out_time_s
            measured = (last.frame - first.frame) / span if span > 0 else 0.0
            item.update(
                frames=last.frame, duration_s=round(last.out_time_s, 3), fps=round(measured, 3)
            )
            durations[cam] = last.out_time_s
            if measured and abs(measured - fps) / fps > 0.02:
                warnings.append(f"{cam}: đo được {measured:.2f} fps, cần {fps:g}")
            try:
                fit = fit_clock(rec.samples)
            except ValueError as exc:
                errors.append(f"{cam}: {exc}")
            else:
                fits[cam] = fit
                item["clock"] = {**asdict(fit), "drift_ppm": round(fit.drift * 1e6, 1)}
                if fit.jitter_ms > 50:
                    warnings.append(f"{cam}: mạng/máy ghi ồn (jitter {fit.jitter_ms:.0f} ms)")
                if fit.n_bins < 3:
                    warnings.append(f"{cam}: buổi quá ngắn để ước lượng trôi đồng hồ")
        else:
            errors.append(f"{cam}: không ghi được khung nào")
        cams_out[cam] = item

    sync: dict | None = None
    if not errors and len(fits) == len(recorders):
        reference = next(iter(recorders))
        try:
            plan = plan_sync(fits, durations, reference)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            files = {cam: _rel_posix(rec.file) for cam, rec in recorders.items()}
            out_rel = f"data/lab/{session}"
            cmd = sync_command(files, plan, out_dir=out_rel, fps=fps)
            for cam, err in plan.edge_error_ms.items():
                if err > 20:
                    warnings.append(
                        f"{cam}: trôi đồng hồ làm lệch tới {err:.0f} ms ở hai đầu buổi; nên chia "
                        "buổi ngắn hơn"
                    )
            sync = {
                "method": "đường bao dưới của (giờ nhận báo cáo − vị trí trong file)",
                "reference": plan.reference,
                "lags_s": {c: round(v, 4) for c, v in plan.lags_s.items()},
                "edge_error_ms": {c: round(v, 1) for c, v in plan.edge_error_ms.items()},
                "common_window_s": [round(plan.window_s[0], 3), round(plan.window_s[1], 3)],
                "command": cmd,
            }

    ok = not errors
    record_json = out_dir / "record.json"
    report = {
        "session": session,
        "tool": "tools.record_rtsp",
        "started_at": started_at.isoformat(timespec="seconds"),
        "stopped_by": stopped_by,
        "ok": ok,
        "ffmpeg": ffmpeg_info,
        "transport": transport,
        "fps_expected": fps,
        "cameras": cams_out,
        "sync": sync,
        "errors": errors,
        "warnings": warnings,
    }
    record_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return SessionResult(ok, stopped_by, cams_out, errors, warnings, sync, record_json)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def wait_enter(stream, stop: threading.Event) -> None:
    """Đặt `stop` khi người dùng bấm Enter — CHỈ khi đọc được một dòng thật.

    EOF (`readline()` trả chuỗi rỗng) không phải Enter: có môi trường báo stdin là TTY mà đọc
    ra EOF ngay, và coi EOF là Enter thì mọi buổi ghi tự dừng sau chưa tới 1 giây (đo trên máy
    dev 2026-10-12: 24 khung rồi dừng). Khi đó vẫn dừng được bằng Ctrl+C hoặc `--duration`.
    """
    try:
        line = stream.readline()
    except (OSError, ValueError):
        return
    if line:
        stop.set()


def _parse_camera(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"cần dạng cam01=rtsp://..., nhận {redact(value)!r}")
    cam, url = value.split("=", 1)
    cam = cam.strip().lower()
    if not re.fullmatch(r"cam\d+", cam):
        raise argparse.ArgumentTypeError(f"tên camera phải dạng camNN, nhận {cam!r}")
    return cam, url.strip()


def _parse_size(value: str) -> tuple[int, int]:
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"cần dạng 1920x1080, nhận {value!r}") from exc


def _print_summary(result: SessionResult, printer: Callable[[str], None]) -> None:
    printer("")
    for cam, item in result.cameras.items():
        clock = item.get("clock") or {}
        printer(
            f"{cam}: {item.get('frames', 0)} khung, {item.get('duration_s', 0):.1f} s, "
            f"{item.get('fps', 0):.2f} fps, jitter {clock.get('jitter_ms', 0):.0f} ms, "
            f"trôi {clock.get('drift_ppm', 0):+.0f} ppm -> {item['file']}"
        )
    if result.sync:
        lags = ", ".join(f"{c} {v:+.3f} s" for c, v in result.sync["lags_s"].items())
        edge = max(result.sync["edge_error_ms"].values())
        printer(
            f"độ lệch so với {result.sync['reference']}: {lags} (sai số do trôi ≤ {edge:.0f} ms)"
        )
    for warning in result.warnings:
        printer(f"[WARN] {warning}")
    for error in result.errors:
        printer(f"[FAIL] {error}")
    printer(f"-> {_rel_posix(result.record_json)}")
    if not result.ok:
        printer("BUỔI NÀY HỎNG: quay lại (tên buổi mới, hoặc --overwrite).")
        return
    printer("")
    printer(
        "Kiểm bằng mắt cặp chồng lấn: vỗ tay trong vùng chung, hai tay chạm nhau cùng khung ±1."
    )
    printer("Bước sau (máy có ffmpeg + libx264, xem docs/m6/README.md mục 4):")
    printer("  PYTHONPATH=src " + " ".join(result.sync["command"] if result.sync else []))


def main(
    argv: list[str] | None = None,
    *,
    runner: Runner = _run,
    reach: Reach = tcp_reachable,
    ffmpeg_cmd: Sequence[str] | None = None,
    environ: dict[str, str] | None = None,
) -> int:
    # Trước cả argparse: `--help` và thông báo lỗi tiếng Việt cũng chết vì cp1252 khi đầu ra bị
    # chuyển hướng trên Windows (CLAUDE.md §11), và lỗi đi ra stderr.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    common.add_argument(
        "--camera", action="append", default=[], type=_parse_camera, metavar="CAM=URL"
    )
    common.add_argument("--ffmpeg", default=None, help="mặc định LAB_FFMPEG trong .env, rồi PATH")
    common.add_argument("--ffprobe", default=None, help="mặc định: cạnh ffmpeg")
    common.add_argument("--out-root", type=Path, default=REPO_ROOT / "data" / "lab" / "raw")
    common.add_argument("--expect-size", type=_parse_size, default=(1920, 1080))
    common.add_argument("--fps", type=float, default=25.0, help="fps camera phải phát")

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("check", parents=[common], help="kiểm camera + máy ghi trước buổi quay")
    rec = sub.add_parser("record", parents=[common], help="ghi một buổi quay")
    rec.add_argument("--session", required=True, help="tên buổi: calib, s1, s2...")
    rec.add_argument("--duration", type=float, default=None, help="tự dừng sau N giây")
    rec.add_argument("--overwrite", action="store_true", help="ghi đè buổi đã có")
    rec.add_argument("--transport", choices=("tcp", "udp"), default="tcp")
    args = p.parse_args(argv)

    env = merged_env(args.env_file, environ)
    cameras = cameras_from_env(env)
    cameras.update(dict(args.camera))
    cameras = dict(sorted(cameras.items()))
    ffmpeg = args.ffmpeg or env.get("LAB_FFMPEG") or "ffmpeg"
    ffprobe = args.ffprobe or _sibling_tool(ffmpeg, "ffprobe")

    if args.command == "check":
        rep = run_check(
            cameras,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            out_root=args.out_root,
            expect_size=args.expect_size,
            expect_fps=args.fps,
            runner=runner,
            reach=reach,
        )
        print(rep.render())
        print(f"\n{rep.n_fail} FAIL, {rep.n_warn} WARN")
        return 1 if rep.n_fail else 0

    if not _SESSION.match(args.session):
        p.error(f"tên buổi chỉ gồm chữ, số, '-', '_': {args.session!r}")
    if not cameras:
        p.error("không có camera nào: thêm LAB_RTSP_CAM01=rtsp://... vào .env (chạy `check`)")
    out_dir = args.out_root / args.session
    existing = sorted(out_dir.glob("cam*.mkv")) + sorted(out_dir.glob("record.json"))
    if existing and not args.overwrite:
        p.error(
            f"{_rel_posix(out_dir)} đã có bản ghi ({existing[0].name}). Đặt tên buổi khác "
            "(vd s1b) hoặc thêm --overwrite nếu chắc chắn bỏ bản cũ"
        )
    for cam, url in cameras.items():
        try:
            host, port = host_port(url)
        except ValueError as exc:
            p.error(f"{cam}: {exc}")
        why = reach(host, port)
        if why is not None:
            p.error(f"{cam}: không kết nối được {host}:{port} — chạy `check` để xem chi tiết")

    info = ""
    if ffmpeg_cmd is None:
        ffmpeg_path = _resolve_tool(ffmpeg)
        if ffmpeg_path is None:
            p.error(f"không tìm thấy {ffmpeg!r} (chạy `check`)")
        ffmpeg_cmd = [ffmpeg_path]
        info, _ = ffmpeg_version(ffmpeg_path, runner)

    stop = threading.Event()
    if sys.stdin is not None and sys.stdin.isatty():
        threading.Thread(target=wait_enter, args=(sys.stdin, stop), daemon=True).start()
    until = f"tự dừng sau {args.duration:g} s" if args.duration else "Enter hoặc Ctrl+C để dừng"
    print(f"Ghi buổi {args.session}: {', '.join(cameras)} -> {_rel_posix(out_dir)} ({until})")
    result = record_session(
        cameras,
        out_dir,
        session=args.session,
        ffmpeg_cmd=ffmpeg_cmd,
        fps=args.fps,
        duration_s=args.duration,
        transport=args.transport,
        stop=stop,
        ffmpeg_info=info,
    )
    _print_summary(result, print)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
