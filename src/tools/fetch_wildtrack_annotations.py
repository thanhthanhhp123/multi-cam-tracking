"""Tải phần NHỎ của WildTrack: annotation vị trí + file hiệu chỉnh camera.

Đủ để chạy `tools.wildtrack_to_fixture --no-reid` (fixture chỉ hình học) và để dựng
homography cho `configs/cameras/` sau này. Ảnh gốc (~13GB) KHÔNG tải ở đây — lấy riêng
từ https://www.epfl.ch/labs/cvlab/data/data-wildtrack/ và giải nén vào cùng thư mục
(tạo `Image_subsets/C1..C7/`).

Nguồn: bản mirror trong repo crowdbotp/OpenTraj (đã đối chiếu schema khớp bản gốc EPFL).
Chỉ dùng stdlib, tải lại được (bỏ qua file đã có).

    python -m tools.fetch_wildtrack_annotations --dest data/wildtrack
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path

from common.logging import get_logger

log = get_logger("tools.fetch_wildtrack")

_RAW_BASE = "https://raw.githubusercontent.com/crowdbotp/OpenTraj/master/datasets/Wild-Track"

# WildTrack: 400 khung chú thích, tên file cách nhau 5 (00000000.json .. 00001995.json).
_N_ANN = 400
_ANN_STEP = 5

# 7 camera: 4 "CVLab" + 3 "IDIAP", theo thứ tự viewNum 0..6.
_CAM_NAMES = ("CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3")
_CALIB_GROUPS = ("extrinsic", "intrinsic_original", "intrinsic_zero")


def _relative_paths() -> list[str]:
    paths = [f"annotations_positions/{i * _ANN_STEP:08d}.json" for i in range(_N_ANN)]
    for group in _CALIB_GROUPS:
        prefix = "extr" if group == "extrinsic" else "intr"
        paths += [f"calibrations/{group}/{prefix}_{name}.xml" for name in _CAM_NAMES]
    return paths


def _download_one(rel: str, dest: Path, *, timeout: float) -> str:
    out = dest / rel
    if out.is_file() and out.stat().st_size > 0:
        return "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_RAW_BASE}/{rel}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # URL cố định, luôn https
            data = resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{url} -> HTTP {exc.code}") from exc
    out.write_bytes(data)
    return "ok"


def fetch(dest: str | Path, *, timeout: float = 30.0) -> dict[str, int]:
    dest = Path(dest)
    counts = {"ok": 0, "skip": 0}
    rels = _relative_paths()
    for i, rel in enumerate(rels, start=1):
        counts[_download_one(rel, dest, timeout=timeout)] += 1
        if i % 50 == 0 or i == len(rels):
            log.info("%d/%d file (mới %d, có sẵn %d)", i, len(rels), counts["ok"], counts["skip"])
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dest", type=Path, default=Path("data/wildtrack"))
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args(argv)

    counts = fetch(args.dest, timeout=args.timeout)
    log.info(
        "Xong: %s/annotations_positions + calibrations (%d file mới, %d đã có). "
        "Ảnh gốc ~13GB tải riêng từ EPFL vào %s/Image_subsets/",
        args.dest,
        counts["ok"],
        counts["skip"],
        args.dest,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
