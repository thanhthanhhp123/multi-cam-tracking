"""M3 - fine-tune OSNet for person Re-ID and export the ONNX the pipeline will consume.

Runs inside an sbatch job on ut-hpc. Never touches the network: the dataset must already
be under --data-root and the pretrained checkpoint already cached (fetch_pretrained.py).

The exported ONNX must match what src/tools/reid_onnx.py expects on the dev machine:
input NCHW 1x3x256x128, ImageNet normalisation applied by the caller, 512-d output.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path.home() / "mct" / "data" / "reid")
    p.add_argument("--sources", default="market1501", help="comma-separated torchreid dataset names")
    p.add_argument("--targets", default="", help="eval datasets; defaults to --sources")
    p.add_argument("--model", default="osnet_x1_0")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-dir", type=Path, default=Path.home() / "mct" / "runs" / "reid")
    p.add_argument("--onnx-out", type=Path, default=Path.home() / "mct" / "models" / "reid" / "osnet_x1_0_ft.onnx")
    return p.parse_args()


def main() -> int:
    import torch
    import torchreid

    args = parse_args()
    sources = [s for s in args.sources.split(",") if s]
    targets = [s for s in (args.targets or args.sources).split(",") if s]

    datamanager = torchreid.data.ImageDataManager(
        root=str(args.data_root),
        sources=sources,
        targets=targets,
        height=args.height,
        width=args.width,
        batch_size_train=args.batch_size,
        batch_size_test=100,
        transforms=["random_flip", "random_crop", "random_erase"],
    )

    model = torchreid.models.build_model(
        name=args.model,
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True,
    ).cuda()

    optimizer = torchreid.optim.build_optimizer(model, optim="adam", lr=args.lr)
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, lr_scheduler="cosine", max_epoch=args.epochs
    )
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager, model, optimizer=optimizer, scheduler=scheduler, label_smooth=True
    )
    engine.run(
        save_dir=str(args.save_dir),
        max_epoch=args.epochs,
        eval_freq=10,
        print_freq=50,
        test_only=False,
    )

    # Export with the classifier detached: the pipeline needs the 512-d embedding, not logits.
    model.eval()
    model.classifier = torch.nn.Identity()
    dummy = torch.randn(1, 3, args.height, args.width, device="cuda")
    args.onnx_out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.onnx_out),
        input_names=["images"],
        output_names=["features"],
        opset_version=12,
        dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
    )
    with torch.no_grad():
        dim = model(dummy).shape[1]
    print(f"exported {args.onnx_out} (embed_dim={dim})")
    print("embed_dim phai khop header message trong src/common/schema.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
