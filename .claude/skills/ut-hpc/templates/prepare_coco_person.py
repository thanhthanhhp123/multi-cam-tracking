"""Download COCO and derive a person-only detection dataset. HEAD NODE ONLY.

Compute nodes on ut-hpc have no internet access, so every byte a training job needs
must already sit on disk. This script downloads COCO through Ultralytics, then rewrites
the label files keeping class 0 (person) only, and emits a one-class dataset YAML.

Usage (on the ut-hpc head node):
    ~/mct/env/bin/python ~/mct/jobs/prepare_coco_person.py
Roughly 20 GB. Check `df -h $HOME` first - home was 90% full on 2026-09-03.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path.home() / "mct"
DEST = ROOT / "data" / "coco-person"
PERSON_CLASS = 0


def download_coco() -> Path:
    """Let Ultralytics fetch COCO; return the directory it landed in."""
    from ultralytics.data.utils import check_det_dataset

    info = check_det_dataset("coco.yaml", autodownload=True)
    return Path(info["path"])


def filter_labels(src_labels: Path, dst_labels: Path) -> tuple[int, int]:
    """Copy label files, keeping only person rows. Returns (files_written, boxes_kept)."""
    dst_labels.mkdir(parents=True, exist_ok=True)
    files = boxes = 0
    for txt in sorted(src_labels.glob("*.txt")):
        kept = [
            line
            for line in txt.read_text().splitlines()
            if line.strip() and int(line.split()[0]) == PERSON_CLASS
        ]
        # An empty label file is a valid "no objects here" negative for Ultralytics.
        (dst_labels / txt.name).write_text("\n".join(kept) + ("\n" if kept else ""))
        files += 1
        boxes += len(kept)
    return files, boxes


def main() -> int:
    coco = download_coco()
    print(f"COCO at {coco}")
    DEST.mkdir(parents=True, exist_ok=True)

    for split in ("train2017", "val2017"):
        images = coco / "images" / split
        labels = coco / "labels" / split
        if not labels.is_dir():
            print(f"missing {labels} - COCO layout changed, fix this script", file=sys.stderr)
            return 1
        # Symlink images (no second 19 GB copy); rewrite labels for real.
        link = DEST / "images" / split
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(images, target_is_directory=True)
        files, boxes = filter_labels(labels, DEST / "labels" / split)
        print(f"{split}: {files} label files, {boxes} person boxes")

    yaml_path = DEST / "coco_person.yaml"
    yaml_path.write_text(
        f"path: {DEST}\ntrain: images/train2017\nval: images/val2017\n\nnc: 1\nnames:\n  0: person\n"
    )
    print(f"wrote {yaml_path}")
    print("free space:", shutil.disk_usage(Path.home()).free // 2**30, "GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
