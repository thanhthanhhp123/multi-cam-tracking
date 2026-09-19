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
from dataclasses import replace
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


def relabel_with_gt_ids(
    messages: list[FrameMessage],
    gt_by_frame: dict,
    *,
    min_iou: float,
) -> list[FrameMessage]:
    """Oracle tracker: thay `local_track_id` bằng `personID` của WildTrack, giữ nguyên mọi thứ khác.

    Dùng để tách lỗi của tracker đơn camera (NvDCF) khỏi lỗi của module liên kết: fixture ra
    có CÙNG hộp, CÙNG embedding, CÙNG `ts_ms` với đầu vào, chỉ khác danh tính cục bộ — nên
    hiệu số điểm giữa hai fixture quy được về đúng một nguyên nhân là chất lượng tracker.

    - NvDCF vỡ một người thành hai id → sau đây thành một (`personID` là duy nhất trên camera).
    - NvDCF trộn hai người vào một id → sau đây tách lại thành hai.

    Đầu vào phải là detection ĐÃ khớp được GT (đầu ra của `rebuild_messages`): detection
    không ghép được với người nào thì không có `personID` để gán, và im lặng bỏ nó đi sẽ làm
    fixture oracle lệch tập detection so với fixture gốc — nên báo lỗi.
    """
    out: list[FrameMessage] = []
    for msg in messages:
        gt_dets = gt_by_frame.get((view_idx_for_cam(msg.cam_id), int(msg.frame_id)), [])
        pairs = match_frame(
            [tuple(float(v) for v in d.bbox) for d in msg.detections],  # type: ignore[misc]
            [g.bbox for g in gt_dets],
            min_iou=min_iou,
        )
        person_of = {det_i: gt_dets[gt_i].person_id for det_i, gt_i, _ in pairs}
        if len(person_of) != len(msg.detections):
            raise ValueError(
                f"{msg.cam_id} frame {msg.frame_id}: {len(msg.detections) - len(person_of)} "
                "detection không ghép được người WildTrack nào — chỉ dùng cho fixture đã qua "
                "`rebuild_messages`"
            )
        out.append(
            FrameMessage(
                cam_id=msg.cam_id,
                frame_id=msg.frame_id,
                ts_ms=msg.ts_ms,
                frame_pts_ns=msg.frame_pts_ns,
                frame_width=msg.frame_width,
                frame_height=msg.frame_height,
                detections=[
                    replace(det, local_track_id=int(person_of[i]))
                    for i, det in enumerate(msg.detections)
                ],
                embed_dim=msg.embed_dim,
            )
        )
    return out


class EmbeddingCache:
    """(cam_id, frame_id, hộp) -> embedding đã L2-normalize, lưu ra `.npz`.

    **Vì sao cần.** Embedding của một crop chỉ phụ thuộc ảnh, hộp và model — không phụ thuộc
    fixture nào yêu cầu nó. Với `--boxes gt`, ba lần chạy pipeline khác nhau (nhiễu giữa các
    lần, phiên 22) khớp gần như cùng một tập hộp GT, nên chỉ lần đầu phải trả giá suy luận
    (OSNet trên CPU ~0.2 s/crop, ~20 000 crop mỗi fixture); các lần sau gần như không tốn gì.

    Cache gắn với TÊN model: đọc file của model khác là lỗi, không phải im lặng dùng nhầm.
    Tuy nhiên nó KHÔNG biết crop được cắt bằng code nào — đổi `crop_for_reid` hay tiền xử lý
    của `OsnetOnnxEmbedder` thì phải xoá file cache.
    """

    def __init__(self, path: Path | None, *, model_tag: str = "") -> None:
        self.path = path
        self.model_tag = model_tag
        self._items: dict[tuple, np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        if path is not None and path.is_file():
            self._load(path)

    @staticmethod
    def key(cam_id: str, frame_id: int, bbox) -> tuple:
        return (str(cam_id), int(frame_id), tuple(round(float(v), 3) for v in bbox))

    def __len__(self) -> int:
        return len(self._items)

    def get(self, cam_id: str, frame_id: int, bbox) -> np.ndarray | None:
        return self._items.get(self.key(cam_id, frame_id, bbox))

    def put(self, cam_id: str, frame_id: int, bbox, embedding: np.ndarray) -> None:
        self._items[self.key(cam_id, frame_id, bbox)] = np.array(embedding, dtype=np.float32)

    def _load(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            stored = str(data["model"])
            if stored != self.model_tag:
                raise ValueError(
                    f"{path}: cache của model {stored!r}, đang dùng {self.model_tag!r} — "
                    "xoá file hoặc trỏ --embed-cache sang chỗ khác"
                )
            for cam, frame, box, emb in zip(
                data["cams"], data["frames"], data["boxes"], data["embs"], strict=True
            ):
                self._items[self.key(str(cam), int(frame), box)] = emb

    def save(self) -> None:
        if self.path is None or not self._items:
            return
        keys = list(self._items)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            tmp,
            model=np.array(self.model_tag),
            cams=np.array([k[0] for k in keys]),
            frames=np.array([k[1] for k in keys], dtype=np.int64),
            boxes=np.array([k[2] for k in keys], dtype=np.float64),
            embs=np.stack([self._items[k] for k in keys]),
        )
        tmp.replace(self.path)  # thay nguyên tử: ngắt giữa chừng không để lại file hỏng


def attach_embeddings(
    messages: list[FrameMessage],
    *,
    wildtrack_dir: Path,
    frame_numbers: list[int],
    embedder,
    image_subdir_fmt: str = "C{n}",
    image_reader=None,
    cache: EmbeddingCache | None = None,
) -> None:
    """Trích embedding tại chỗ cho từng message, gom theo ảnh để mỗi PNG chỉ đọc một lần.

    Có `cache` thì crop đã có sẵn không bị suy luận lại, và ảnh mà MỌI crop đều đã có thì
    không được đọc lên (đọc + giải mã một PNG 1080p đã tốn ~0.15 s).
    """
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

        todo: list[Detection] = []
        for det in dets:
            hit = cache.get(cam_id, frame_id, det.bbox) if cache is not None else None
            if hit is None:
                todo.append(det)
            else:
                det.embedding = hit.copy()
        if cache is not None:
            cache.hits += len(dets) - len(todo)
            cache.misses += len(todo)

        if todo:
            subdir = image_subdir_fmt.format(n=view_idx_for_cam(cam_id) + 1)
            path = image_root / subdir / f"{frame_numbers[frame_id]:08d}.png"
            image = read_image(path)
            if image is None:
                raise FileNotFoundError(f"Không đọc được ảnh {path}")

            feats = embedder.embed([crop_for_reid(image, d.bbox) for d in todo])
            for det, feat in zip(todo, feats, strict=True):
                det.embedding = l2_normalize(feat)
                if cache is not None:
                    cache.put(cam_id, frame_id, det.bbox, det.embedding)

        if done % 200 == 0 or done == len(items):
            if cache is not None:
                cache.save()
                log.info(
                    "trích embedding: %d/%d ảnh (cache: %d trúng, %d trượt)",
                    done,
                    len(items),
                    cache.hits,
                    cache.misses,
                )
            else:
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
    p.add_argument(
        "--oracle-out",
        type=Path,
        default=None,
        help="ghi THÊM một fixture 'oracle tracker': cùng hộp/embedding/ts_ms với --out nhưng "
        "local_track_id = personID WildTrack. So điểm hai fixture này = chi phí của tracker "
        "đơn camera. Dùng chung một lượt trích embedding nên không tốn thêm CPU",
    )
    p.add_argument(
        "--embed-cache",
        type=Path,
        default=None,
        help="file .npz nhớ embedding theo (camera, khung, hộp) — dùng chung giữa nhiều fixture "
        "cùng --boxes gt để mỗi crop chỉ bị suy luận một lần. Gắn với tên model ONNX",
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
    cache = None
    if args.embed_cache is not None:
        cache = EmbeddingCache(args.embed_cache, model_tag=args.reid_onnx.name)
        log.info("cache embedding: %s (%d mục có sẵn)", args.embed_cache, len(cache))
    attach_embeddings(
        rebuilt,
        wildtrack_dir=args.wildtrack_dir,
        frame_numbers=frame_numbers,
        embedder=embedder,
        image_subdir_fmt=args.image_subdir_fmt,
        cache=cache,
    )

    # KHÔNG strict: fixture nguồn mang sẵn các detection `confidence = -0.1` — target do
    # nvtracker suy ra khi khung đó không có detection (đã biết từ phiên 9). Loại chúng đi
    # sẽ làm fixture này lệch TẬP DETECTION so với fixture DeepStream gốc, mà cả thí nghiệm
    # dựa trên việc hai bên có đúng cùng một tập. Cảnh báo rồi giữ nguyên, như record_metadata.
    problems = sum(1 for msg in rebuilt if validate(msg))
    if problems:
        log.warning("%d message vi phạm contract (giữ nguyên, xem chú thích trong code)", problems)
    n = write_jsonl(args.out, rebuilt)

    if args.oracle_out is not None:
        oracle = relabel_with_gt_ids(rebuilt, gt_index(raw), min_iou=args.min_iou)
        n_oracle = write_jsonl(args.oracle_out, oracle)
        n_ids = len({(m.cam_id, d.local_track_id) for m in oracle for d in m.detections})
        n_ids_src = len({(m.cam_id, d.local_track_id) for m in rebuilt for d in m.detections})
        log.info(
            "%s: %d message, oracle tracker — %d local track (tracker thật: %d)",
            args.oracle_out,
            n_oracle,
            n_ids,
            n_ids_src,
        )

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
