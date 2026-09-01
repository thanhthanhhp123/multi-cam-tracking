"""Xuất OSNet (torchreid) ra ONNX để `tools.reid_onnx` chạy trên CPU khi dựng fixture.

Chạy MỘT LẦN. Cần PyTorch + torchreid — cả hai KHÔNG nằm trong dependency dự án, chỉ
phục vụ bước chuẩn bị dữ liệu này. Kết quả để ở `models/reid/` (gitignored).

    pip install torch torchreid

    # (khuyến nghị) tải checkpoint re-id từ Model Zoo torchreid rồi trỏ vào:
    #   https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
    python -m tools.export_osnet_onnx \
        --weights ~/Downloads/osnet_x1_0_market1501.pth.tar \
        --out models/reid/osnet_x1_0_market1501.onnx

    # hoặc nhanh gọn (chỉ init ImageNet, KHÔNG được huấn luyện re-id — chỉ để chạy thử):
    python -m tools.export_osnet_onnx --out models/reid/osnet_x1_0_imagenet.onnx

Model mặc định osnet_x1_0 cho feature 512 chiều. Tiền xử lý (resize 256x128, chuẩn hoá
ImageNet) phải khớp `tools/reid_onnx.py`.

`torch`/`torchreid` chỉ import trong `main()` — file này vẫn phải import được trên môi
trường chỉ có deps lõi (tests/test_no_gpu_imports.py quét toàn bộ src/tools).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common.logging import get_logger

log = get_logger("tools.export_osnet")

_MODELS = ("osnet_x1_0", "osnet_x0_75", "osnet_x0_5", "osnet_x0_25", "osnet_ain_x1_0")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="osnet_x1_0", choices=_MODELS)
    p.add_argument("--out", type=Path, default=Path("models/reid/osnet_x1_0_market1501.onnx"))
    p.add_argument("--opset", type=int, default=12)
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="checkpoint re-id .pth.tar tải từ Model Zoo torchreid; bỏ trống = chỉ init ImageNet",
    )
    args = p.parse_args(argv)

    import torch
    import torchreid

    model = torchreid.models.build_model(
        name=args.model, num_classes=1000, pretrained=args.weights is None
    )
    if args.weights is not None:
        torchreid.utils.load_pretrained_weights(model, str(args.weights))
        log.info("Nạp weight re-id từ %s", args.weights)
    else:
        log.warning(
            "Không có --weights: model chỉ init ImageNet, CHƯA huấn luyện re-id. "
            "Embedding sẽ yếu — chỉ nên dùng để chạy thử đường ống."
        )
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 256, 128)
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["images"],
        output_names=["features"],
        opset_version=args.opset,
        dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
    )
    log.info("Đã xuất %s -> %s (feature 512-d, opset %d)", args.model, args.out, args.opset)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
