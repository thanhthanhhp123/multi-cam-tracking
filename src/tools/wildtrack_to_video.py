"""Đóng chuỗi ảnh WildTrack thành video H.264 để chạy được qua pipeline DeepStream.

    python -m tools.wildtrack_to_video --wildtrack-dir data/wildtrack \\
        --out-dir data/wildtrack_video

**Vì sao cần:** ngưỡng `max_cost` của `configs/demo/wildtrack.mct.yaml` được chỉnh trên
fixture WildTrack có embedding trích bằng **ONNX Runtime trên CPU** (`tools/reid_onnx.py`).
Pipeline thật trích embedding bằng **TensorRT trong nvtracker**, với tiền xử lý khác một
chút (`netScaleFactor` vô hướng thay cho std từng kênh, FP16) — xem worklog phiên 8, quyết
định 4. Muốn biết ngưỡng có chuyển được sang đường DeepStream không thì phải cho **chính
WildTrack** đi qua pipeline DeepStream, mà pipeline chỉ nhận video/RTSP chứ không nhận thư
mục ảnh. Đây là bước đóng gói đó.

QUY ƯỚC QUAN TRỌNG NHẤT — khung thứ `i` của video là khung chú thích thứ `i`:

    video cam0N, frame index i  <->  annotation_frame_numbers()[i]  <->  frame_idx i

`probes.py` ghi `frame_id = frame_meta.frame_num` (đếm từ 0 theo từng nguồn), nên nhờ quy
ước này `frame_id` tra thẳng được về khung chú thích WildTrack, KHÔNG cần khớp theo thời
gian. `tools/ds_wildtrack_gt.py` dựa hoàn toàn vào đó để gán ground-truth. Thứ tự khung lấy
từ `tools.wildtrack_to_fixture.annotation_frame_numbers` — cùng một hàm với bộ dựng fixture,
để hai đường không thể lệch nhau.

Ảnh WildTrack là PNG 1920x1080 không nén mất mát. CRF mặc định 18 (gần như không nhìn thấy
suy hao) vì thứ đang đo là **chất lượng embedding**: nén mạnh tay sẽ trộn nhiễu nén vào kết
quả và không còn quy được về nguyên nhân nào.

CHẠY Ở ĐÂU: `ut-hpc` (ảnh gốc 7.2 GB nằm sẵn ở `~/mct/data/wildtrack/`, có `ffmpeg` 4.4.2
với `libx264` — đã kiểm 2026-09-04). Chỉ dùng CPU. Xong thì scp video sang `vast-gpu`.

Chỉ dùng stdlib + ffmpeg ngoài; không import gì cần GPU (quy tắc bất biến 1 của CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from common.logging import get_logger
from tools.wildtrack_to_fixture import (
    N_VIEWS,
    WILDTRACK_H,
    WILDTRACK_W,
    annotation_frame_numbers,
    cam_id_for_view,
)

log = get_logger("tools.wildtrack_video")

# Tên file trong index.json và trong configs/demo/streams_wildtrack.yaml phải khớp nhau.
INDEX_NAME = "index.json"

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


class VideoBuildError(RuntimeError):
    """Đóng video thất bại — ffmpeg lỗi, thiếu ảnh, hoặc số khung ra không như mong đợi."""


@dataclass(slots=True)
class CameraVideo:
    """Kết quả đóng gói một camera."""

    cam_id: str
    view: int  # 1-based, khớp thư mục C1..C7
    path: Path
    n_frames: int
    n_frames_probed: int = -1  # -1 = chưa kiểm bằng ffprobe
    size_bytes: int = 0


@dataclass(slots=True)
class EncodeSpec:
    """Tham số encode — gom lại để ghi nguyên vẹn vào index.json (tái lập được)."""

    fps: float = 2.0
    crf: int = 18
    preset: str = "medium"
    pix_fmt: str = "yuv420p"
    codec: str = "libx264"
    gop: int = 0  # 0 = tự suy ra 2 giây

    def keyint(self) -> int:
        return self.gop if self.gop > 0 else max(1, round(self.fps * 2))

    def as_dict(self) -> dict:
        return {
            "codec": self.codec,
            "crf": self.crf,
            "preset": self.preset,
            "pix_fmt": self.pix_fmt,
            "fps": self.fps,
            "gop": self.keyint(),
        }


@dataclass(slots=True)
class BuildResult:
    videos: list[CameraVideo] = field(default_factory=list)
    index_path: Path | None = None
    frame_numbers: list[int] = field(default_factory=list)


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), capture_output=True, text=True, check=False)


def _link_or_copy(src: Path, dst: Path) -> None:
    """symlink -> hardlink -> copy. Chỉ symlink là rẻ thật; hai bậc sau để chạy được cả
    trên máy dev Windows (test) lẫn NFS cấm hardlink."""
    for attempt in (os.symlink, os.link):
        try:
            attempt(src, dst)
            return
        except (OSError, NotImplementedError, AttributeError):
            continue
    shutil.copy2(src, dst)


def _sequence_dir(work_root: Path, cam_id: str, image_paths: list[Path]) -> Path:
    """ffmpeg chỉ đọc chuỗi ảnh theo mẫu tên liên tiếp (`%06d.png`), còn WildTrack đặt tên
    cách nhau 5. Dựng một thư mục link tạm đánh số liên tiếp thay vì tin vào demuxer
    `concat` — thứ tự khung ở đây là thứ mọi ánh xạ ground-truth về sau dựa vào, nên phải
    tường minh."""
    seq_dir = work_root / f".seq-{cam_id}"
    if seq_dir.exists():
        shutil.rmtree(seq_dir)
    seq_dir.mkdir(parents=True)
    for i, src in enumerate(image_paths):
        _link_or_copy(src.resolve(), seq_dir / f"{i:06d}.png")
    return seq_dir


def _ffmpeg_cmd(seq_dir: Path, out_path: Path, spec: EncodeSpec, ffmpeg: str) -> list[str]:
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{spec.fps:g}",
        "-start_number",
        "0",
        "-i",
        str(seq_dir / "%06d.png"),
        "-c:v",
        spec.codec,
        "-preset",
        spec.preset,
        "-crf",
        str(spec.crf),
        "-pix_fmt",
        spec.pix_fmt,
        "-g",
        str(spec.keyint()),
        "-movflags",
        "+faststart",
        str(out_path),
    ]


def probe_frame_count(path: Path, *, ffprobe: str = "ffprobe", runner: Runner | None = None) -> int:
    """Số khung thật trong file video, hoặc -1 nếu không hỏi được ffprobe.

    Kiểm bằng ffprobe chứ không tin ffmpeg báo thành công: thiếu/thừa một khung là toàn bộ
    ánh xạ `frame_id -> khung chú thích` lệch đi, và lỗi đó KHÔNG có triệu chứng — engine
    vẫn chạy, chỉ ra điểm số thấp một cách khó hiểu.
    """
    run = runner or _run
    proc = run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
    )
    if proc.returncode != 0:
        log.warning("ffprobe không chạy được trên %s: %s", path.name, proc.stderr.strip()[:200])
        return -1
    text = (proc.stdout or "").strip().splitlines()
    try:
        return int(text[0])
    except (IndexError, ValueError):
        return -1


def _normalize_views(views: list[int]) -> list[int]:
    out = sorted({int(v) for v in views})
    if not out or out[0] < 1 or out[-1] > N_VIEWS:
        raise ValueError(f"--views phải nằm trong 1..{N_VIEWS}, nhận {views}")
    return out


def build_videos(
    wildtrack_dir: str | Path,
    out_dir: str | Path,
    *,
    views: list[int] | None = None,
    spec: EncodeSpec | None = None,
    frame_stride: int = 1,
    max_frames: int = 0,
    image_subdir_fmt: str = "C{n}",
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    runner: Runner | None = None,
    verify: bool = True,
    keep_sequence: bool = False,
) -> BuildResult:
    """Đóng mỗi camera thành một file `<out_dir>/cam0N.mp4` + ghi `index.json`."""
    wildtrack_dir, out_dir = Path(wildtrack_dir), Path(out_dir)
    spec = spec or EncodeSpec()
    run = runner or _run
    view_list = _normalize_views(views or list(range(1, N_VIEWS + 1)))

    ann_dir = wildtrack_dir / "annotations_positions"
    if not ann_dir.is_dir():
        raise FileNotFoundError(
            f"{ann_dir} không tồn tại. Lấy annotation bằng "
            f"`python -m tools.fetch_wildtrack_annotations --dest {wildtrack_dir}`."
        )
    frame_numbers = annotation_frame_numbers(ann_dir, stride=frame_stride, max_frames=max_frames)
    if not frame_numbers:
        raise VideoBuildError(f"{ann_dir}: không có khung chú thích nào sau khi lọc")

    image_root = wildtrack_dir / "Image_subsets"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = BuildResult(frame_numbers=frame_numbers)

    for view in view_list:
        cam_id = cam_id_for_view(view - 1)
        cam_dir = image_root / image_subdir_fmt.format(n=view)
        paths = [cam_dir / f"{n:08d}.png" for n in frame_numbers]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{cam_dir}: thiếu {len(missing)} ảnh (ví dụ {missing[0].name}). "
                "Kiểm tra --wildtrack-dir và --image-subdir-fmt."
            )

        seq_dir = _sequence_dir(out_dir, cam_id, paths)
        out_path = out_dir / f"{cam_id}.mp4"
        try:
            proc = run(_ffmpeg_cmd(seq_dir, out_path, spec, ffmpeg))
            if proc.returncode != 0:
                raise VideoBuildError(
                    f"ffmpeg lỗi khi đóng {cam_id} (mã {proc.returncode}): "
                    f"{(proc.stderr or '').strip()[:500]}"
                )
        finally:
            if not keep_sequence:
                shutil.rmtree(seq_dir, ignore_errors=True)

        video = CameraVideo(
            cam_id=cam_id,
            view=view,
            path=out_path,
            n_frames=len(frame_numbers),
            size_bytes=out_path.stat().st_size if out_path.is_file() else 0,
        )
        if verify:
            video.n_frames_probed = probe_frame_count(out_path, ffprobe=ffprobe, runner=run)
            if 0 <= video.n_frames_probed != video.n_frames:
                raise VideoBuildError(
                    f"{out_path.name}: video có {video.n_frames_probed} khung nhưng cần "
                    f"{video.n_frames} — ánh xạ frame_id -> khung chú thích sẽ sai"
                )
        result.videos.append(video)
        log.info("%s: %d khung, %.1f MB", out_path.name, video.n_frames, video.size_bytes / 1e6)

    result.index_path = write_index(out_dir, result, spec, frame_stride=frame_stride)
    return result


def write_index(out_dir: Path, result: BuildResult, spec: EncodeSpec, *, frame_stride: int) -> Path:
    """Ghi `index.json` — hợp đồng giữa video và bộ gán ground-truth."""
    path = out_dir / INDEX_NAME
    payload = {
        "source": "wildtrack",
        "width": WILDTRACK_W,
        "height": WILDTRACK_H,
        "frame_stride": frame_stride,
        "n_frames": len(result.frame_numbers),
        "frame_numbers": result.frame_numbers,
        "encoder": spec.as_dict(),
        "cameras": {
            v.cam_id: {
                "view": v.view,
                "file": v.path.name,
                "n_frames": v.n_frames,
                "n_frames_probed": v.n_frames_probed,
                "size_bytes": v.size_bytes,
            }
            for v in result.videos
        },
        "contract": (
            "Khung thứ i của mỗi video là khung chú thích thứ i (frame_numbers[i]). "
            "FrameMessage.frame_id do probes.py ghi ra chính là i."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--wildtrack-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--views", default="1,2,3,4,5,6,7", help="số camera 1..7, phẩy ngăn cách")
    p.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="phải bằng --fps của wildtrack_to_fixture (chú thích ~2 fps) để ts_ms cùng nhịp",
    )
    p.add_argument("--crf", type=int, default=18, help="18 = gần như không suy hao")
    p.add_argument("--preset", default="medium")
    p.add_argument("--gop", type=int, default=0, help="0 = 2 giây")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0, help="0 = tất cả 400 khung")
    p.add_argument("--image-subdir-fmt", default="C{n}")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    p.add_argument("--no-verify", action="store_true", help="bỏ bước ffprobe đếm khung")
    args = p.parse_args(argv)

    result = build_videos(
        args.wildtrack_dir,
        args.out_dir,
        views=[int(v) for v in str(args.views).split(",") if v.strip()],
        spec=EncodeSpec(fps=args.fps, crf=args.crf, preset=args.preset, gop=args.gop),
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        image_subdir_fmt=args.image_subdir_fmt,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        verify=not args.no_verify,
    )
    total_mb = sum(v.size_bytes for v in result.videos) / 1e6
    log.info(
        "%d video, %d khung/camera, tổng %.1f MB, index tại %s",
        len(result.videos),
        len(result.frame_numbers),
        total_mb,
        result.index_path,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
