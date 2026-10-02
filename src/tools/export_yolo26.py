"""Xuất YOLO26s sang ONNX định dạng DeepStream-Yolo — trên CPU của máy dev, không cần GPU.

    # venv riêng, một lần (torch CPU ~200 MB; KHÔNG cài vào venv test):
    uv venv --python 3.10 ~/.venvs/mct-export
    uv pip install --python ~/.venvs/mct-export/Scripts/python.exe \\
        --index-url https://download.pytorch.org/whl/cpu torch torchvision
    uv pip install --python ~/.venvs/mct-export/Scripts/python.exe \\
        "ultralytics==8.4.7" onnx onnxslim onnxruntime python-dotenv PyYAML
    # mỗi lần:
    PYTHONPATH=src ~/.venvs/mct-export/Scripts/python.exe -m tools.export_yolo26

Ra `models/detector/yolo26s.onnx` (gitignored, rsync sang `vast-gpu` khi chạy pipeline —
CLAUDE.md §2 quy tắc 5), dùng với `configs/pipeline/config_infer_yolo26_b*.txt`.

**Vì sao xuất trên máy dev chứ không trên `vast-gpu` như YOLO11.** Cài ultralytics trong
image DeepStream nâng numpy lên 2.x và nvtracker segfault lúc build engine ReID (CLAUDE.md
§11). Xuất sẵn ở đây thì máy thuê không cần ultralytics/torch, và một bước dễ hỏng (lại tính
tiền theo giờ) biến mất.

**Ba chỗ phải ghim, mỗi chỗ đều đã gây lỗi thật khi để mặc định** (2026-10-02):

1. Script `export_yolo26.py` của DeepStream-Yolo — ghim commit `DSYOLO_COMMIT` và sha256.
   Script này thay `Detect.forward` bằng head một-một (không NMS) và trả `[batch, N, 6]` =
   x1, y1, x2, y2, score, class, đúng thứ `NvDsInferParseYolo` đọc.
2. `ultralytics==8.4.7` (bản mới nhất khi script được sửa lần cuối, 2026-01-25). Với 8.4.171,
   `Detect.fuse()` bỏ nhánh một-một của checkpoint cũ và export chết ở `KeyError: 'feats'`
   — đúng lỗi Ultralytics ghi nhận với bản 8.4.142.
3. Exporter TorchScript (`dynamo=False`). torch 2.14 mặc định exporter dynamo, và nó không
   trace được `.item()` trong model (`'float' object has no attribute 'node'`). Script gốc
   không cho chọn, nên ở đây bọc `torch.onnx.export` thay vì sửa file của bên thứ ba.

Thêm `--simplify` (onnxslim) để shape đầu ra là `[batch, 8400, 6]` thay vì tên ký hiệu — cho
TensorRT/nvinfer biết sẵn số hộp.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import os
import runpy
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

from common.logging import get_logger

log = get_logger("tools.export_yolo26")

DSYOLO_COMMIT = "2894babce8e75c49115dbe0c7b516289ed853565"
"""marcoslucianops/DeepStream-Yolo, 2026-01-25 — lần cuối `utils/export_yolo26.py` đổi."""

EXPORT_SCRIPT_URL = (
    f"https://raw.githubusercontent.com/marcoslucianops/DeepStream-Yolo/{DSYOLO_COMMIT}"
    "/utils/export_yolo26.py"
)
EXPORT_SCRIPT_SHA256 = "75136de2c55b5123b71d152facb28804301738ff20772116da11077c2adc2ea9"

WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt"
WEIGHTS_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"

ULTRALYTICS_VERSION = "8.4.7"
NET_SIZE = 640


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path, expected_sha256: str) -> Path:
    """Tải `url` về `dest` (nếu chưa có) và đòi đúng sha256 — không chạy file lạ."""
    if not dest.exists():
        log.info("tải %s", url)
        urllib.request.urlretrieve(url, dest)  # URL cố định ở đầu file
    actual = sha256(dest)
    if actual != expected_sha256:
        raise RuntimeError(f"{dest.name}: sha256 {actual} != {expected_sha256}")
    return dest


def check_output(onnx_path: Path, size: int) -> list[object]:
    """Shape đầu ra phải là [batch, N, 6]: thứ `NvDsInferParseYolo` đọc."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    shape = list(session.get_outputs()[0].shape)
    expected_n = (size // 8) ** 2 + (size // 16) ** 2 + (size // 32) ** 2
    if len(shape) != 3 or shape[1:] != [expected_n, 6]:
        raise RuntimeError(f"đầu ra {shape}, cần [batch, {expected_n}, 6]")
    return shape


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", type=Path, default=Path("models/detector"))
    p.add_argument("--size", type=int, default=NET_SIZE)
    p.add_argument("--work-dir", type=Path, default=None, help="mặc định: thư mục tạm")
    args = p.parse_args(argv)

    import torch
    import ultralytics

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise SystemExit(
            f"ultralytics {ultralytics.__version__}, cần đúng {ULTRALYTICS_VERSION} (xem docstring)"
        )

    work = args.work_dir or Path(tempfile.mkdtemp(prefix="yolo26_export_"))
    work.mkdir(parents=True, exist_ok=True)
    script = fetch(EXPORT_SCRIPT_URL, work / "export_yolo26.py", EXPORT_SCRIPT_SHA256)
    fetch(WEIGHTS_URL, work / "yolo26s.pt", WEIGHTS_SHA256)

    # Script gốc ghi ONNX và labels.txt vào cwd và đọc sys.argv: chạy nó đúng như tác giả
    # định, chỉ đổi exporter mặc định của torch.
    torch.onnx.export = functools.partial(torch.onnx.export, dynamo=False)
    cwd, argv_saved = Path.cwd(), sys.argv
    try:
        os.chdir(work)
        sys.argv = [
            "export_yolo26.py",
            "-w",
            "yolo26s.pt",
            "-s",
            str(args.size),
            "--dynamic",
            "--simplify",
        ]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        os.chdir(cwd)
        sys.argv = argv_saved

    onnx_path = work / "yolo26s.onnx"
    shape = check_output(onnx_path, args.size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / ("yolo26s.onnx" if args.size == NET_SIZE else f"yolo26s_{args.size}.onnx")
    shutil.copyfile(onnx_path, out)
    log.info("-> %s  đầu ra %s  sha256 %s", out, shape, sha256(out))
    log.info(
        "labels.txt của YOLO26 là 80 lớp COCO, trùng models/detector/labels.txt — không chép đè"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
