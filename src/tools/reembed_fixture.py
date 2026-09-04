"""Trích lại embedding cho fixture DeepStream bằng ONNX Runtime, từ hộp detector HOẶC hộp GT.

    PYTHONPATH=src python -m tools.reembed_fixture \\
        --fixture data/fixtures/ds_wildtrack_7cam.jsonl --wildtrack-dir data/wildtrack \\
        --reid-onnx models/reid/osnet_x1_0_msdc_dg.onnx --boxes fixture \\
        --out data/fixtures/ds_wildtrack_7cam_onnx_detbox.jsonl

**Vì sao cần.** Phiên 11 đo được ngoại hình là nút thắt (trần F1 chỉ 0.181), nhưng hai
fixture đang có khác nhau ở HAI biến cùng lúc:

| fixture | hộp cắt crop | bộ trích embedding |
|---|---|---|
| `wildtrack_7cam.jsonl` | ground-truth | ONNX Runtime (CPU) |
| `ds_wildtrack_7cam.jsonl` | detector YOLO11s | TensorRT trong nvtracker |

So hai cái đó với nhau thì chênh lệch không quy được về nguyên nhân nào. Công cụ này sinh
fixture thứ ba và thứ tư, **cùng cấu trúc tracklet, cùng bộ trích, chỉ khác hộp**:

- `--boxes fixture` → hộp của detector + ONNX Runtime
- `--boxes gt`      → hộp ground-truth + ONNX Runtime

Đặt cạnh nhau thì tách được sạch hai biến:

- `--boxes gt` so với `--boxes fixture` → **ảnh hưởng của chất lượng hộp**, mọi thứ khác giữ nguyên.
- `--boxes fixture` so với fixture DeepStream gốc → **ảnh hưởng của đường trích**, hộp giữ nguyên.

**Chỉ giữ detection khớp được một hộp GT** (IoU ≥ `--min-iou`, ghép Hungarian như
`tools/ds_wildtrack_gt.py`), ở CẢ HAI chế độ. Nhờ vậy hai fixture ra có đúng cùng tập
detection, cùng `local_track_id`, cùng `ts_ms` — khác đúng một thứ là toạ độ hộp.

Ánh xạ `frame_id` → khung chú thích dựa vào quy ước đóng video của `tools/wildtrack_to_video.py`
(khung thứ i của video là khung chú thích thứ i). Cắt crop dùng chung
`wildtrack_to_fixture.crop_for_reid` để không có hai cách cắt khác nhau.

Cần `cv2` + `onnxruntime` (trên `ut-hpc`: `~/mct/venv-reid`), và ảnh gốc WildTrack.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.logging import get_logger
from common.schema import Detection, FrameMessage, l2_normalize, validate, write_jsonl
from common.schema import read_jsonl as read_fixture
from tools.ds_wildtrack_gt import gt_index, match_frame, view_idx_for_cam
from tools.wildtrack_to_fixture import crop_for_reid, parse_raw_detections

log = get_logger("tools.reembed")

BOX_SOURCES = ("fixture", "gt")


def _read_image(path: Path) -> np.ndarray | None:
    import cv2

    return cv2.imread(str(path))


def rebuild_messages(
    messages: list[FrameMessage],
    gt_by_frame: dict,
    *,
    box_source: str,
    min_iou: float,
) -> tuple[list[FrameMessage], dict[str, int]]:
    """Lọc lấy detection khớp GT, và thay bbox nếu `box_source == "gt"`.

    Trả về message MỚI (không sửa tại chỗ) để gọi hai lần trên cùng đầu vào cho ra hai kết
    quả độc lập — đúng thứ thí nghiệm này cần.
    """
    if box_source not in BOX_SOURCES:
        raise ValueError(f"--boxes phải là một trong {BOX_SOURCES}, nhận {box_source!r}")

    out: list[FrameMessage] = []
    stats = {"n_detections": 0, "n_matched": 0}

    for msg in messages:
        view_idx = view_idx_for_cam(msg.cam_id)
        gt_dets = gt_by_frame.get((view_idx, int(msg.frame_id)), [])
        stats["n_detections"] += len(msg.detections)
        if not msg.detections or not gt_dets:
            continue

        pairs = match_frame(
            [tuple(float(v) for v in d.bbox) for d in msg.detections],  # type: ignore[misc]
            [g.bbox for g in gt_dets],
            min_iou=min_iou,
        )
        kept: list[Detection] = []
        for det_i, gt_i, _ in sorted(pairs):
            src = msg.detections[det_i]
            bbox = gt_dets[gt_i].bbox if box_source == "gt" else src.bbox
            kept.append(
                Detection(
                    local_track_id=src.local_track_id,
                    bbox=tuple(float(v) for v in bbox),  # type: ignore[arg-type]
                    confidence=src.confidence,
                    embedding=None,
                )
            )
        stats["n_matched"] += len(kept)
        if not kept:
            continue
        out.append(
            FrameMessage(
                cam_id=msg.cam_id,
                frame_id=msg.frame_id,
                ts_ms=msg.ts_ms,
                frame_pts_ns=msg.frame_pts_ns,
                frame_width=msg.frame_width,
                frame_height=msg.frame_height,
                detections=kept,
                embed_dim=0,
            )
        )
    return out, stats


def attach_embeddings(
    messages: list[FrameMessage],
    *,
    wildtrack_dir: Path,
    frame_numbers: list[int],
    embedder,
    image_subdir_fmt: str = "C{n}",
    image_reader=None,
) -> None:
    """Trích embedding tại chỗ cho từng message, gom theo ảnh để mỗi PNG chỉ đọc một lần."""
    read_image = image_reader or _read_image
    image_root = wildtrack_dir / "Image_subsets"

    by_image: dict[tuple[str, int], list[Detection]] = defaultdict(list)
    for msg in messages:
        by_image[(msg.cam_id, int(msg.frame_id))].extend(msg.detections)

    items = sorted(by_image.items())
    for done, ((cam_id, frame_id), dets) in enumerate(items, start=1):
        if frame_id >= len(frame_numbers):
            raise IndexError(
                f"frame_id {frame_id} vượt quá {len(frame_numbers)} khung chú thích — "
                "fixture và dataset không cùng --frame-stride/--max-frames"
            )
        subdir = image_subdir_fmt.format(n=view_idx_for_cam(cam_id) + 1)
        path = image_root / subdir / f"{frame_numbers[frame_id]:08d}.png"
        image = read_image(path)
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh {path}")

        feats = embedder.embed([crop_for_reid(image, d.bbox) for d in dets])
        for det, feat in zip(dets, feats, strict=True):
            det.embedding = l2_normalize(feat)

        if done % 200 == 0 or done == len(items):
            log.info("trích embedding: %d/%d ảnh", done, len(items))

    for msg in messages:
        msg.embed_dim = msg.infer_embed_dim()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fixture", type=Path, required=True, help="fixture DeepStream đầu vào")
    p.add_argument("--wildtrack-dir", type=Path, required=True)
    p.add_argument("--reid-onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--boxes",
        choices=BOX_SOURCES,
        required=True,
        help="'fixture' = hộp của detector, 'gt' = hộp ground-truth đã khớp IoU",
    )
    p.add_argument("--min-iou", type=float, default=0.5)
    p.add_argument("--min-box-area", type=float, default=0.0)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--image-subdir-fmt", default="C{n}")
    p.add_argument("--reid-batch", type=int, default=32)
    args = p.parse_args(argv)

    from tools.reid_onnx import OsnetOnnxEmbedder
    from tools.wildtrack_to_fixture import annotation_frame_numbers

    messages = list(read_fixture(args.fixture))
    cam_ids = sorted({m.cam_id for m in messages})
    view_indices = sorted(view_idx_for_cam(c) for c in cam_ids)

    ann_dir = args.wildtrack_dir / "annotations_positions"
    frame_numbers = annotation_frame_numbers(
        ann_dir, stride=args.frame_stride, max_frames=args.max_frames
    )
    raw, _ = parse_raw_detections(
        ann_dir,
        view_indices=view_indices,
        stride=args.frame_stride,
        max_frames=args.max_frames,
        min_box_area=args.min_box_area,
    )

    rebuilt, stats = rebuild_messages(
        messages, gt_index(raw), box_source=args.boxes, min_iou=args.min_iou
    )
    if not rebuilt:
        raise SystemExit("không có detection nào khớp GT — xem lại --min-iou / ánh xạ frame_id")

    embedder = OsnetOnnxEmbedder(args.reid_onnx, batch_size=args.reid_batch)
    attach_embeddings(
        rebuilt,
        wildtrack_dir=args.wildtrack_dir,
        frame_numbers=frame_numbers,
        embedder=embedder,
        image_subdir_fmt=args.image_subdir_fmt,
    )

    # KHÔNG strict: fixture nguồn mang sẵn các detection `confidence = -0.1` — target do
    # nvtracker suy ra khi khung đó không có detection (đã biết từ phiên 9). Loại chúng đi
    # sẽ làm fixture này lệch TẬP DETECTION so với fixture DeepStream gốc, mà cả thí nghiệm
    # dựa trên việc hai bên có đúng cùng một tập. Cảnh báo rồi giữ nguyên, như record_metadata.
    problems = sum(1 for msg in rebuilt if validate(msg))
    if problems:
        log.warning("%d message vi phạm contract (giữ nguyên, xem chú thích trong code)", problems)
    n = write_jsonl(args.out, rebuilt)

    log.info(
        "%s: %d message, %d/%d detection giữ lại (hộp=%s), embed_dim=%d",
        args.out,
        n,
        stats["n_matched"],
        stats["n_detections"],
        args.boxes,
        rebuilt[0].embed_dim,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
