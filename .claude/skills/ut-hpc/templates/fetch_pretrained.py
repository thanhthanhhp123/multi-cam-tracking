"""Pull pretrained weights onto the cluster. HEAD NODE ONLY (compute nodes have no network).

    ~/mct/env/bin/python ~/mct/jobs/fetch_pretrained.py

Downloads the YOLO checkpoint used as the fine-tuning starting point, and warms the
torch hub cache under $HOME/mct/.torch so training jobs never reach for the network.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

ROOT = Path.home() / "mct"
MODELS = ROOT / "models"

# Ultralytics release assets. Swap the tag/name if the project settles on another size.
YOLO_ASSETS = {
    "yolo11s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
}


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"skip {dest.name} (da co, {dest.stat().st_size / 2**20:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"tai {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest} ({dest.stat().st_size / 2**20:.1f} MB)")


def main() -> int:
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch"))
    for name, url in YOLO_ASSETS.items():
        fetch(url, MODELS / name)

    # OSNet weights come from the torchreid model zoo (Google Drive) and cannot be fetched
    # by plain URL reliably. Trigger torchreid's own downloader instead, if it is installed.
    try:
        import torchreid

        torchreid.models.build_model(
            name="osnet_x1_0", num_classes=751, pretrained=True, loss="softmax"
        )
        print("osnet_x1_0 pretrained: cached duoi ~/.cache/torch/checkpoints")
    except ImportError:
        print("torchreid chua cai - bo qua OSNet (xem train_reid.sbatch)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
