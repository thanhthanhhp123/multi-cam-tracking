"""Đồng bộ các video quay riêng (điện thoại) bằng tiếng vỗ tay, rồi cắt + chuyển sang fps cố định.

    PYTHONPATH=src python -m tools.sync_recordings \\
        --video cam01=data/lab/raw/s1/cam01.mp4 --video cam02=data/lab/raw/s1/cam02.mp4 \\
        --video cam03=data/lab/raw/s1/cam03.mp4 --video cam04=data/lab/raw/s1/cam04.mp4 \\
        --out-dir data/lab/s1 --fps 25 --execute

**Vì sao cần** (đề cương mục 4.3.2 bước 2: "quay video đồng bộ thời gian giữa các camera,
đồng bộ mốc bằng tín hiệu chung hoặc NTP"). Phương án dự phòng khi chưa có camera IP (worklog
phiên 29, bộ nhớ dự án): 3–4 điện thoại tự quay ra FILE. Mỗi máy bấm quay lúc một khác, đồng
hồ mỗi máy lệch một khác, và điện thoại quay **fps thay đổi** (VFR). Trong khi đó:

- ràng buộc thời gian di chuyển và so vị trí "tại cùng mốc thời gian" của cặp chồng lấn
  (CLAUDE.md §6) đều đòi các camera chung một trục thời gian, sai vài trăm ms đã đủ làm cặp
  chồng lấn so nhầm vị trí của người đang đi (≈ 1.4 m/s);
- `frame_id` của pipeline phải tra thẳng về khung chú thích CVAT — VFR làm "khung thứ i" của
  hai công cụ không còn là cùng một khung.

**Cách đồng bộ: tín hiệu chung = tiếng vỗ tay** (hoặc tiếng clapperboard) ở ĐẦU và CUỐI buổi
quay, đủ to để mọi máy thu được. Âm thanh truyền 1 m mất 3 ms, nhỏ hơn 1 khung — đứng giữa các
máy là đủ. Công cụ trích audio bằng ffmpeg, dựng đường bao onset (năng lượng tăng đột ngột), tương
quan chéo với camera tham chiếu và lấy đỉnh. Độ tin cậy = đỉnh cao nhất / đỉnh cao thứ nhì ngoài
±`--exclusion-ms`; dưới `--min-confidence` thì báo động thay vì lặng lẽ dùng một độ lệch sai.
Vỗ tay hai lần với nhịp KHÔNG đều (vd 1 tiếng, nghỉ 2 s, 2 tiếng) để đỉnh không bị nhân bản.
Có thể đè từng camera bằng `--offset camXX=<giây>` (vd khi đo bằng đèn flash trên hình).

**Sau khi tìm độ lệch:** cửa sổ chung = khoảng mọi camera cùng có hình; mỗi video được cắt đúng
cửa sổ đó và mã hoá lại ở fps CỐ ĐỊNH (`-vf fps=`), không B-frame, keyframe mỗi giây — để khung
thứ i ở mọi camera là CÙNG một thời điểm `i / fps` (sai số ≤ nửa khung). Đây là file cho cả
pipeline (`streams` của `configs/lab/`) lẫn CVAT: một file, hai nơi dùng, nên số khung khớp.
Audio bị bỏ (DeepStream không cần, và để audio lại thì tiếng nói của người tham gia đi theo).

Kết quả: `<out-dir>/<cam>.mp4` + `<out-dir>/sync.json` (độ lệch, độ tin cậy, cửa sổ, lệnh ffmpeg,
thông tin ffprobe của file gốc — ghi lại để báo cáo tái lập được).

**Phải kiểm bằng mắt một lần:** mở các file đầu ra ở đúng khung của tiếng vỗ tay (in trong
`sync.json`, `clap_frame`), tay phải chạm nhau ở cùng một khung ±1 trên mọi camera. Audio và
hình của điện thoại có thể lệch nhau vài chục ms; nếu lệch thấy được, sửa bằng `--offset`.

Chạy ở đâu: máy có `ffmpeg` + `libx264` (máy dev Windows hiện KHÔNG có ffmpeg; image DeepStream
7.1 phải cài thêm thư viện, CLAUDE.md §11). Phần tìm độ lệch chỉ đọc audio nên nhẹ; phần mã hoá
lại tốn CPU cỡ thời lượng video — làm trên `vast-gpu` trong lúc chờ build engine là hợp lý.

Chỉ numpy + stdlib + ffmpeg ngoài; không import gì cần GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from common.logging import get_logger

log = get_logger("tools.sync_recordings")

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[bytes]"]

AUDIO_RATE = 8000
HOP_S = 0.001
WINDOW_S = 0.010


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(list(cmd), capture_output=True, check=False)


# --------------------------------------------------------------------------------------
# ffprobe
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VideoInfo:
    path: str
    width: int
    height: int
    duration_s: float
    r_frame_rate: float
    avg_frame_rate: float
    rotation: int
    has_audio: bool

    @property
    def variable_rate(self) -> bool:
        """VFR: `r_frame_rate` (nhịp danh định) khác `avg_frame_rate` (thực tế) quá 1%."""
        if self.avg_frame_rate <= 0:
            return True
        return abs(self.r_frame_rate - self.avg_frame_rate) / self.avg_frame_rate > 0.01

    @property
    def display_size(self) -> tuple[int, int]:
        """Kích thước SAU khi xoay — ffmpeg tự xoay theo metadata khi mã hoá lại."""
        if self.rotation % 180:
            return (self.height, self.width)
        return (self.width, self.height)


def _rate(text: str | None) -> float:
    if not text or text in ("0/0", "N/A"):
        return 0.0
    return float(Fraction(text))


def parse_ffprobe(path: str, data: dict) -> VideoInfo:
    """JSON của `ffprobe -show_streams -show_format` → `VideoInfo`.

    Góc xoay của điện thoại nằm ở `tags.rotate` (ffmpeg cũ) hoặc `side_data_list[].rotation`
    (ffmpeg mới) — đọc cả hai. Quên xoay là quên rằng ảnh dọc 1080x1920 không phải 1920x1080.
    """
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"{path}: không có luồng video")
    rotation = 0
    if "rotate" in (video.get("tags") or {}):
        rotation = int(float(video["tags"]["rotate"]))
    for side in video.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(float(side["rotation"]))
    duration = float(video.get("duration") or (data.get("format") or {}).get("duration") or 0)
    return VideoInfo(
        path=path,
        width=int(video["width"]),
        height=int(video["height"]),
        duration_s=duration,
        r_frame_rate=_rate(video.get("r_frame_rate")),
        avg_frame_rate=_rate(video.get("avg_frame_rate")),
        rotation=rotation % 360,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def probe(path: Path, *, ffprobe: str = "ffprobe", runner: Runner | None = None) -> VideoInfo:
    run = runner or _run
    proc = run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe lỗi trên {path}: {proc.stderr[-300:]!r}")
    return parse_ffprobe(str(path), json.loads(proc.stdout))


def read_audio(
    path: Path, *, ffmpeg: str = "ffmpeg", rate: int = AUDIO_RATE, runner: Runner | None = None
) -> np.ndarray:
    """Audio mono `rate` Hz dạng float32 trong [-1, 1], đọc qua ống của ffmpeg."""
    run = runner or _run
    proc = run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(rate),
            "-f",
            "s16le",
            "-",
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg không trích được audio từ {path}: {proc.stderr[-300:]!r}")
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0


# --------------------------------------------------------------------------------------
# Tìm độ lệch
# --------------------------------------------------------------------------------------


def onset_envelope(
    samples: np.ndarray, rate: int = AUDIO_RATE, *, hop_s: float = HOP_S, window_s: float = WINDOW_S
) -> np.ndarray:
    """Đường bao onset, một giá trị mỗi `hop_s`: phần TĂNG của log-năng lượng theo cửa sổ.

    Dùng log để một tiếng vỗ tay ở máy xa (nhỏ hơn 10 lần) vẫn cho cùng hình dạng; chỉ giữ
    phần tăng để tiếng ồn nền đều đều (quạt, xe) không đóng góp gì vào tương quan.
    """
    hop = max(1, round(hop_s * rate))
    win = max(hop, round(window_s * rate))
    if samples.size < win:
        return np.zeros(0, dtype=np.float64)
    power = np.convolve(samples.astype(np.float64) ** 2, np.ones(win) / win, mode="valid")[::hop]
    log_power = np.log10(power + 1e-10)
    onset = np.maximum(np.diff(log_power, prepend=log_power[0]), 0.0)
    onset -= onset.mean()
    norm = np.linalg.norm(onset)
    return onset / norm if norm > 0 else onset


@dataclass(frozen=True, slots=True)
class LagEstimate:
    lag_s: float
    """Sự kiện ở thời điểm `t` của video tham chiếu nằm ở `t + lag_s` của video này."""
    confidence: float
    """Đỉnh cao nhất / đỉnh cao thứ nhì ngoài vùng loại trừ. ~1 là không phân biệt được."""
    peak: float


def estimate_lag(
    reference: np.ndarray,
    other: np.ndarray,
    *,
    hop_s: float = HOP_S,
    max_lag_s: float = 120.0,
    exclusion_s: float = 0.25,
) -> LagEstimate:
    """Tương quan chéo hai đường bao bằng FFT, tìm độ trễ trong ±`max_lag_s`."""
    if reference.size == 0 or other.size == 0:
        raise ValueError("đường bao rỗng — video không có audio?")
    n = reference.size + other.size - 1
    size = 1 << (n - 1).bit_length()
    spec = np.fft.rfft(other, size) * np.conj(np.fft.rfft(reference, size))
    corr = np.fft.irfft(spec, size)
    # corr[k] với k >= 0 ứng với other trễ k mẫu; phần cuối mảng là độ trễ âm.
    lags = np.concatenate([np.arange(0, other.size), np.arange(-(reference.size - 1), 0)])
    values = np.concatenate([corr[: other.size], corr[size - (reference.size - 1) :]])
    max_lag = int(max_lag_s / hop_s)
    keep = np.abs(lags) <= max_lag
    lags, values = lags[keep], values[keep]

    best = int(np.argmax(values))
    peak = float(values[best])
    far = np.abs(lags - lags[best]) > int(exclusion_s / hop_s)
    second = float(values[far].max()) if far.any() else 0.0
    confidence = peak / second if second > 1e-12 else float("inf")
    return LagEstimate(lag_s=float(lags[best]) * hop_s, confidence=confidence, peak=peak)


@dataclass(frozen=True, slots=True)
class Window:
    start_s: float
    """Trên trục thời gian của video THAM CHIẾU."""
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def common_window(lags: dict[str, float], durations: dict[str, float]) -> Window:
    """Khoảng thời gian (theo trục của video tham chiếu) mà MỌI camera cùng có hình.

    Video `c` có hình ở `τ ∈ [−lag_c, duration_c − lag_c]` trên trục tham chiếu.
    """
    start = max(-lags[c] for c in lags)
    end = min(durations[c] - lags[c] for c in lags)
    if end <= start:
        raise ValueError(
            f"các video không có đoạn chung (cửa sổ [{start:.2f}, {end:.2f}] s) — độ lệch sai?"
        )
    return Window(start, end)


def encode_cmd(
    src: Path,
    dst: Path,
    *,
    start_s: float,
    duration_s: float,
    fps: float,
    crf: int = 18,
    preset: str = "medium",
    scale: tuple[int, int] | None = None,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """Cắt [start, start + duration) và mã hoá lại ở fps cố định, H.264 không B-frame.

    `-ss` trước `-i` + mã hoá lại là chính xác tới khung với ffmpeg hiện hành. `fps=` chọn
    khung gần nhất cho mỗi mốc `i / fps` — đó là bước VFR → CFR. Không B-frame và GOP 1 giây
    để `nvv4l2decoder` không phải giữ khung, và tua trong CVAT nhanh.
    """
    vf = [f"fps={fps:g}"]
    if scale is not None:
        vf.append(f"scale={scale[0]}:{scale[1]}")
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration_s:.3f}",
        "-vf",
        ",".join(vf),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-bf",
        "0",
        "-g",
        str(max(1, round(fps))),
        "-movflags",
        "+faststart",
        str(dst),
    ]


def _parse_kv(value: str, *, cast=str):
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"cần dạng camXX=giá_trị, nhận {value!r}")
    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("cam_id rỗng")
    try:
        return key, cast(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r}: {exc}") from exc


def _parse_size(value: str) -> tuple[int, int]:
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"cần dạng 1920x1080, nhận {value!r}") from exc


def main(argv: list[str] | None = None, *, runner: Runner | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--video",
        action="append",
        required=True,
        type=lambda v: _parse_kv(v, cast=Path),
        metavar="CAM=FILE",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fps", type=float, default=25.0, help="fps cố định của video đầu ra")
    p.add_argument("--reference", default=None, help="camera tham chiếu (mặc định: cái đầu tiên)")
    p.add_argument(
        "--offset",
        action="append",
        default=[],
        type=lambda v: _parse_kv(v, cast=float),
        metavar="CAM=GIÂY",
        help="đè độ lệch tìm được (giây, cùng nghĩa với lag_s trong sync.json)",
    )
    p.add_argument("--max-lag-s", type=float, default=120.0)
    p.add_argument("--exclusion-ms", type=float, default=250.0)
    p.add_argument("--min-confidence", type=float, default=1.5)
    p.add_argument("--trim-start-s", type=float, default=0.0, help="bỏ thêm ở đầu cửa sổ chung")
    p.add_argument("--trim-end-s", type=float, default=0.0, help="bỏ thêm ở cuối cửa sổ chung")
    p.add_argument("--scale", type=_parse_size, default=None, help="vd 1920x1080")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", default="medium")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    p.add_argument("--execute", action="store_true", help="chạy ffmpeg (không có thì chỉ in lệnh)")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    videos: dict[str, Path] = dict(args.video)
    if len(videos) != len(args.video):
        p.error("trùng cam_id trong --video")
    reference = args.reference or args.video[0][0]
    if reference not in videos:
        p.error(f"--reference {reference} không có trong --video")
    overrides = dict(args.offset)
    if runner is None and (shutil.which(args.ffmpeg) is None or shutil.which(args.ffprobe) is None):
        p.error("không tìm thấy ffmpeg/ffprobe trên PATH (xem docstring: chạy ở đâu)")

    infos = {cam: probe(path, ffprobe=args.ffprobe, runner=runner) for cam, path in videos.items()}
    for cam, info in infos.items():
        if info.variable_rate:
            log.info(
                "%s: VFR (danh định %.3f, thực tế %.3f fps) — sẽ chuyển sang %g fps cố định",
                cam,
                info.r_frame_rate,
                info.avg_frame_rate,
                args.fps,
            )
        if info.rotation:
            log.warning(
                "%s: video xoay %d° — đầu ra sẽ là %dx%d; homography/CVAT phải dùng kích thước này",
                cam,
                info.rotation,
                *info.display_size,
            )

    lags: dict[str, float] = {reference: 0.0}
    estimates: dict[str, dict] = {}
    need_audio = [c for c in videos if c != reference and c not in overrides]
    if need_audio:
        ref_env = onset_envelope(read_audio(videos[reference], ffmpeg=args.ffmpeg, runner=runner))
    bad: list[str] = []
    for cam in videos:
        if cam == reference:
            continue
        if cam in overrides:
            lags[cam] = float(overrides[cam])
            estimates[cam] = {"lag_s": lags[cam], "source": "--offset"}
            continue
        env = onset_envelope(read_audio(videos[cam], ffmpeg=args.ffmpeg, runner=runner))
        est = estimate_lag(
            ref_env, env, max_lag_s=args.max_lag_s, exclusion_s=args.exclusion_ms / 1000.0
        )
        lags[cam] = est.lag_s
        estimates[cam] = {**asdict(est), "source": "audio"}
        if not est.confidence >= args.min_confidence:
            bad.append(f"{cam} (độ tin cậy {est.confidence:.2f})")
    if bad:
        raise SystemExit(
            "không tìm chắc được độ lệch cho: " + ", ".join(bad) + ". Vỗ tay to hơn / nhịp không "
            "đều, tăng --max-lag-s nếu bấm quay cách nhau lâu, hoặc đo tay rồi dùng --offset."
        )

    window = common_window(lags, {c: i.duration_s for c, i in infos.items()})
    start = window.start_s + args.trim_start_s
    duration = window.duration_s - args.trim_start_s - args.trim_end_s
    if duration <= 0:
        raise SystemExit(f"cửa sổ chung chỉ dài {window.duration_s:.1f} s, cắt bớt quá tay")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan: dict[str, dict] = {}
    for cam, src in videos.items():
        dst = args.out_dir / f"{cam}.mp4"
        cmd = encode_cmd(
            src,
            dst,
            start_s=start + lags[cam],
            duration_s=duration,
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
            scale=args.scale,
            ffmpeg=args.ffmpeg,
        )
        plan[cam] = {
            "src": str(src),
            "dst": str(dst),
            "start_s": round(start + lags[cam], 4),
            "cmd": cmd,
            "probe": asdict(infos[cam]),
        }
        print(f"{cam}: lệch {lags[cam]:+.3f} s, cắt từ {start + lags[cam]:.3f} s")

    n_frames = math.floor(duration * args.fps)
    clap_frame = None
    if need_audio and ref_env.size:
        # Onset mạnh nhất của video tham chiếu, quy ra số khung của video ĐẦU RA: chỗ để
        # kiểm bằng mắt rằng mọi camera thấy hai bàn tay chạm nhau ở cùng một khung.
        clap_ref_s = float(np.argmax(ref_env)) * HOP_S
        if start <= clap_ref_s < start + duration:
            clap_frame = round((clap_ref_s - start) * args.fps)
    report = {
        "reference": reference,
        "fps": args.fps,
        "lags_s": lags,
        "estimates": estimates,
        "window_ref_s": [window.start_s, window.end_s],
        "trim_start_s": args.trim_start_s,
        "trim_end_s": args.trim_end_s,
        "duration_s": duration,
        "expected_frames": n_frames,
        "clap_frame": clap_frame,
        "cameras": plan,
        "notes": (
            "khung i của mọi video đầu ra = thời điểm start_s(ref) + i/fps trên trục video tham "
            "chiếu. Kiểm bằng mắt khung của tiếng vỗ tay trên mọi camera trước khi chú thích."
        ),
    }
    out = args.out_dir / "sync.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"cửa sổ chung {duration:.1f} s ≈ {n_frames} khung ở {args.fps:g} fps -> {out}")
    if clap_frame is not None:
        print(f"kiểm bằng mắt: tiếng vỗ tay mạnh nhất ở khung {clap_frame} của mọi video đầu ra")

    if not args.execute:
        print("(chưa mã hoá — thêm --execute để chạy ffmpeg)")
        return 0
    run = runner or _run
    for cam, item in plan.items():
        proc = run(item["cmd"])
        if proc.returncode != 0:
            raise SystemExit(f"ffmpeg lỗi ở {cam}: {proc.stderr[-500:]!r}")
        log.info("%s -> %s", cam, item["dst"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
