"""Giải nén ảnh WildTrack từ zip gốc của EPFL, tự vá offset lệch 4 GiB.

    python -m tools.unzip_wildtrack --zip Wildtrack_dataset_full.zip \\
        --dest data/wildtrack/Image_subsets

Zip gốc (`documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip`,
6.4 GB) được tạo bằng công cụ macOS **không dùng zip64** cho archive lớn hơn 4 GB, nên
offset local header ghi trong central directory bị lệch đúng `2^32` với một phần entry.
Hậu quả đã gặp (2026-09-04):

  - `7z` từ chối mở hẳn: "Can't open as archive";
  - `zipfile` của Python: `BadZipFile: Bad magic number for file header`;
  - `unzip`: cảnh báo "4294967296 extra bytes", tự bù offset nhưng chỉ đúng cho entry nằm
    SAU mốc 4 GiB (lấy được C5–C7, hỏng C1–C4), rồi chết vì "not enough memory for bomb
    detection" (còn cần `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE`).

Cách vá ở đây: với mỗi entry thử `header_offset` ở `+0`, `+2^32`, `−2^32` và nhận kết quả
đầu tiên giải nén ra đúng `file_size`. Chạy lại được: entry đã có sẵn và đúng kích thước
thì bỏ qua.

Chỉ dùng stdlib — không cần cài gì thêm, và `tests/test_no_gpu_imports.py` import được.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from common.logging import get_logger

log = get_logger("tools.unzip_wildtrack")

SKEW = 1 << 32
DEFAULT_PREFIX = "Wildtrack_dataset/Image_subsets/"


def extract(
    zip_path: str | Path,
    dest: str | Path,
    *,
    prefix: str = DEFAULT_PREFIX,
    cameras: list[str] | None = None,
) -> dict[str, int]:
    """Giải nén các entry dưới `prefix` vào `dest`. Trả về thống kê."""
    zip_path, dest = Path(zip_path), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    stats = {"ok": 0, "repaired": 0, "skipped": 0, "failed": 0}

    with zipfile.ZipFile(zip_path) as zf:
        members = [
            info
            for info in zf.infolist()
            if info.filename.startswith(prefix)
            and not info.is_dir()
            and "/._" not in info.filename
            and (cameras is None or info.filename[len(prefix) :].split("/")[0] in cameras)
        ]
        log.info("%d entry khớp prefix %r", len(members), prefix)

        for n, info in enumerate(members, start=1):
            out = dest / info.filename[len(prefix) :]
            if out.is_file() and out.stat().st_size == info.file_size:
                stats["skipped"] += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)

            base = info.header_offset
            for delta in (0, SKEW, -SKEW):
                info.header_offset = base + delta
                try:
                    with zf.open(info) as src, out.open("wb") as dst:
                        while chunk := src.read(1 << 20):
                            dst.write(chunk)
                except (zipfile.BadZipFile, OSError, EOFError, ValueError):
                    continue
                if out.stat().st_size == info.file_size:
                    stats["ok"] += 1
                    stats["repaired"] += delta != 0
                    break
            else:
                stats["failed"] += 1
                out.unlink(missing_ok=True)
                log.warning("không giải nén được %s", info.filename)
            info.header_offset = base

            if n % 200 == 0:
                log.info("%d/%d entry", n, len(members))

    log.info(
        "Xong: %d giải nén (%d phải vá offset), %d bỏ qua, %d lỗi",
        stats["ok"],
        stats["repaired"],
        stats["skipped"],
        stats["failed"],
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--zip", dest="zip_path", type=Path, required=True)
    p.add_argument("--dest", type=Path, default=Path("data/wildtrack/Image_subsets"))
    p.add_argument("--prefix", default=DEFAULT_PREFIX, help="tiền tố đường dẫn trong zip")
    p.add_argument(
        "--cameras",
        default=None,
        help="danh sách thư mục camera cần lấy, ví dụ C1,C4,C7 (mặc định: tất cả)",
    )
    args = p.parse_args(argv)

    cameras = args.cameras.split(",") if args.cameras else None
    stats = extract(args.zip_path, args.dest, prefix=args.prefix, cameras=cameras)
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
