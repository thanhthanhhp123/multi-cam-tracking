"""Chạy thử ONNX PeopleNet Transformer trên CPU, TRƯỚC khi thuê GPU.

File ONNX của TAO chứa op `MultiscaleDeformableAttnPlugin_TRT`, một plugin chỉ TensorRT
hiểu. Công cụ này thay op đó bằng bản numpy (`msda_reference`, đúng lõi PyTorch của
Deformable DETR), rồi chạy phần còn lại bằng ONNX Runtime. Đầu ra được giải mã y như
`NvDsInferParseCustomDDETRTAO` (`decode_ddetr`) và so với hộp GT của WildTrack.

Công cụ trả lời những câu mà đọc thẻ model không trả lời chắc được:
  1. Người là lớp mấy (kết quả: lớp 1, vì labels là BG, Person, Face, Bag).
  2. Chuẩn hoá đầu vào nào đúng. Config NVIDIA dùng ImageNet, còn thẻ NGC ghi 1/255.
     Kết quả: ImageNet đúng. Với v1, chuẩn hoá 1/255 cho recall 2/33, ImageNet cho 22/33.
  3. Parser có giải mã đúng hộp không (cx, cy, w, h chuẩn hoá theo kích thước mạng).
Xem docs/worklog/2026-09-28-30-*.

KHÔNG phải công cụ đo chất lượng detector. Op thay thế viết bằng Python nên chậm
(~16 s/ảnh với v1, ~36 s/ảnh với v2), vì vậy chỉ chạy vài ảnh. Số đo detector thật lấy
từ pipeline DeepStream (FP16, TensorRT).

Cài thêm: `pip install -e ".[detcheck]"`. `onnx`, `onnxruntime`, `onnxruntime_extensions`
và `cv2` chỉ được import khi chạy thật, vì `tests/test_no_gpu_imports.py` import mọi
file trong `src/tools`.

    python -m tools.check_peoplenet_cpu \\
        --onnx models/detector/peoplenet_transformer/resnet50_peoplenet_transformer_op17.onnx \\
        --wildtrack data/wildtrack --views 0 --norms imagenet unit
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.logging import get_logger

log = get_logger(__name__)

NET_W, NET_H = 960, 544
# Khớp offsets / net-scale-factor trong configs/pipeline/config_infer_peoplenet_transformer*.txt.
IMAGENET_OFFSETS = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_SCALE = 0.0173520735728
# Parser của NVIDIA tự cắt còn 200 hộp mỗi ảnh (keep_top_k).
PARSER_KEEP_TOP_K = 200
PLUGIN_OP = "MultiscaleDeformableAttnPlugin_TRT"


@dataclass(slots=True, frozen=True)
class Box:
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def _bilinear_zero_pad(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """img [B, H, W, D], x/y [B, Q] toạ độ pixel -> [B, Q, D].

    Tương đương `grid_sample(align_corners=False, padding_mode="zeros")`.
    """
    b, h, w, d = img.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    out = np.zeros((b, x.shape[1], d), dtype=np.float32)
    bidx = np.arange(b)[:, None]
    for dx in (0, 1):
        for dy in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            weight = (1 - np.abs(x - xi)) * (1 - np.abs(y - yi))
            inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            vals = img[bidx, np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)]
            out += vals * (weight * inside)[..., None]
    return out


def msda_reference(
    value: np.ndarray,
    spatial_shapes: np.ndarray,
    level_start_index: np.ndarray,
    sampling_locations: np.ndarray,
    attention_weights: np.ndarray,
) -> np.ndarray:
    """Multi-scale deformable attention, cùng đầu vào/đầu ra với plugin TensorRT.

    value [N, S, M, D], spatial_shapes [L, 2] (H, W), level_start_index [L],
    sampling_locations [N, Lq, M, L, P, 2] (x, y chuẩn hoá 0..1),
    attention_weights [N, Lq, M, L, P]  ->  [N, Lq, M, D].
    """
    n, _, m, d = value.shape
    _, lq, _, n_levels, n_points, _ = sampling_locations.shape
    loc = sampling_locations.astype(np.float32)
    out = np.zeros((n, lq, m, d), dtype=np.float32)
    for lvl in range(n_levels):
        h, w = int(spatial_shapes[lvl, 0]), int(spatial_shapes[lvl, 1])
        start = int(level_start_index[lvl])
        feat = value[:, start : start + h * w].reshape(n, h, w, m, d)
        feat = feat.transpose(0, 3, 1, 2, 4).reshape(n * m, h, w, d)
        grid = loc[:, :, :, lvl].transpose(0, 2, 1, 3, 4).reshape(n * m, lq * n_points, 2)
        sampled = _bilinear_zero_pad(feat, grid[..., 0] * w - 0.5, grid[..., 1] * h - 0.5)
        sampled = sampled.reshape(n, m, lq, n_points, d)
        weights = attention_weights[:, :, :, lvl].transpose(0, 2, 1, 3)  # [N, M, Lq, P]
        out += np.einsum("nmqpd,nmqp->nqmd", sampled, weights)
    return out


def decode_ddetr(
    logits: np.ndarray, boxes: np.ndarray, threshold: float, scale_x: float, scale_y: float
) -> list[Box]:
    """Chép lại `NvDsInferParseCustomDDETRTAO` (deepstream_tao_apps) cho một ảnh.

    logits [Q, C] chưa qua sigmoid, boxes [Q, 4] = (cx, cy, w, h) chuẩn hoá 0..1. Lấy argmax
    kể cả lớp nền, bỏ lớp 0, rồi lọc theo ngưỡng. Hộp tính theo kích thước mạng, cuối cùng
    nhân với scale_x/scale_y để đổi ra toạ độ ảnh gốc.
    """
    cls = logits.argmax(-1)
    conf = 1.0 / (1.0 + np.exp(-logits[np.arange(len(cls)), cls]))
    out: list[Box] = []
    for q in np.argsort(-conf, kind="stable")[:PARSER_KEEP_TOP_K]:
        if cls[q] == 0 or conf[q] < threshold:
            continue
        cx, cy, w, h = boxes[q]
        x1 = float(np.clip((cx - w / 2) * NET_W, 0, NET_W - 1))
        y1 = float(np.clip((cy - h / 2) * NET_H, 0, NET_H - 1))
        x2 = float(np.clip(x1 + w * NET_W, 0, NET_W - 1))
        y2 = float(np.clip(y1 + h * NET_H, 0, NET_H - 1))
        out.append(
            Box(int(cls[q]), float(conf[q]), x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
        )
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a [n, 4], b [k, 4] dạng (x1, y1, x2, y2) -> IoU [n, k]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match_count(dets: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5) -> tuple[int, np.ndarray]:
    """Ghép một-một (Hungarian trên IoU) -> (số cặp đạt ngưỡng, mặt nạ GT được khớp)."""
    hit = np.zeros(len(gt), dtype=bool)
    iou = iou_matrix(dets, gt)
    if iou.size == 0:
        return 0, hit
    rows, cols = linear_sum_assignment(-iou)
    good = iou[rows, cols] >= iou_thr
    hit[cols[good]] = True
    return int(good.sum()), hit


def wildtrack_gt_boxes(annotations_dir: Path, frame: str, view: int) -> np.ndarray:
    """Hộp GT (x1, y1, x2, y2) của một view trong một khung chú thích WildTrack."""
    rows = []
    for person in json.loads((annotations_dir / f"{frame}.json").read_text(encoding="utf-8")):
        v = person["views"][view]
        if v["xmin"] >= 0:
            rows.append((v["xmin"], v["ymin"], v["xmax"], v["ymax"]))
    return np.array(rows, dtype=np.float32).reshape(-1, 4)


def preprocess(bgr: np.ndarray, norm: str) -> np.ndarray:
    """Giống nvinfer: kéo về 960x544 (maintain-aspect-ratio=0), RGB, NCHW."""
    import cv2

    rgb = cv2.cvtColor(cv2.resize(bgr, (NET_W, NET_H)), cv2.COLOR_BGR2RGB).astype(np.float32)
    if norm == "imagenet":
        x = (rgb - IMAGENET_OFFSETS) * IMAGENET_SCALE
    elif norm == "unit":
        x = rgb / 255.0
    else:
        raise ValueError(f"norm không hợp lệ: {norm}")
    return x.transpose(2, 0, 1)[None]


def load_session(onnx_path: Path, threads: int = 4) -> Any:
    """Nạp ONNX, đổi mọi node plugin TensorRT sang op Python rồi tạo phiên ORT (CPU)."""
    import onnx
    import onnxruntime as ort
    from onnxruntime_extensions import PyCustomOpDef, get_library_path, onnx_op

    f32, i64, f64 = PyCustomOpDef.dt_float, PyCustomOpDef.dt_int64, PyCustomOpDef.dt_double
    # Bản v2 (DINO) đưa sampling_locations kiểu float64 vào một nửa số tầng, nên cần hai
    # chữ ký. onnx_op đăng ký theo tên op, nên gọi lại hàm này nhiều lần vẫn an toàn.
    onnx_op(op_type="MSDA", inputs=[f32, i64, i64, f32, f32], outputs=[f32])(msda_reference)
    onnx_op(op_type="MSDA_D", inputs=[f32, i64, i64, f64, f32], outputs=[f32])(msda_reference)

    model = onnx.load(str(onnx_path))
    inferred = onnx.shape_inference.infer_shapes(model)
    elem_type = {v.name: v.type.tensor_type.elem_type for v in inferred.graph.value_info}
    n_replaced = 0
    for node in model.graph.node:
        if node.op_type == PLUGIN_OP:
            is_double = elem_type.get(node.input[3]) == onnx.TensorProto.DOUBLE
            node.op_type = "MSDA_D" if is_double else "MSDA"
            node.domain = "ai.onnx.contrib"
            del node.attribute[:]
            n_replaced += 1
    model.opset_import.append(onnx.helper.make_opsetid("ai.onnx.contrib", 1))
    log.info("%s: thay %d node %s bằng bản numpy", onnx_path.name, n_replaced, PLUGIN_OP)

    opts = ort.SessionOptions()
    opts.register_custom_ops_library(get_library_path())
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--wildtrack", type=Path, required=True, help="thư mục WildTrack đã giải nén")
    ap.add_argument("--frames", nargs="+", default=["00000000"])
    ap.add_argument("--views", type=int, nargs="+", default=[0], help="0..6 (C1..C7)")
    ap.add_argument("--norms", nargs="+", default=["imagenet", "unit"])
    ap.add_argument("--threshold", type=float, default=0.3, help="pre-cluster-threshold")
    ap.add_argument("--person-class-id", type=int, default=1)
    args = ap.parse_args(argv)

    import cv2

    sess = load_session(args.onnx)
    for norm in args.norms:
        per_class: dict[int, list[int]] = {}  # lớp -> [số hộp, số hộp khớp người GT]
        n_gt = n_gt_hit = 0
        seconds = 0.0
        for frame in args.frames:
            for view in args.views:
                img_path = args.wildtrack / "Image_subsets" / f"C{view + 1}" / f"{frame}.png"
                img = cv2.imread(str(img_path))
                if img is None:
                    raise FileNotFoundError(img_path)
                t0 = time.perf_counter()
                logits, boxes = sess.run(
                    ["pred_logits", "pred_boxes"], {"inputs": preprocess(img, norm)}
                )
                seconds += time.perf_counter() - t0
                dets = decode_ddetr(
                    logits[0], boxes[0], args.threshold, img.shape[1] / NET_W, img.shape[0] / NET_H
                )
                gt = wildtrack_gt_boxes(args.wildtrack / "annotations_positions", frame, view)
                n_gt += len(gt)
                for c in sorted({d.class_id for d in dets}):
                    arr = np.array([(d.x1, d.y1, d.x2, d.y2) for d in dets if d.class_id == c])
                    tp, hit = match_count(arr.reshape(-1, 4), gt)
                    acc = per_class.setdefault(c, [0, 0])
                    acc[0] += len(arr)
                    acc[1] += tp
                    if c == args.person_class_id:
                        n_gt_hit += int(hit.sum())
        n_img = len(args.frames) * len(args.views)
        log.info(
            "norm=%s  %d ảnh  ngưỡng %.2f  CPU %.1f s/ảnh  %d hộp GT (toàn khung)",
            norm,
            n_img,
            args.threshold,
            seconds / n_img,
            n_gt,
        )
        for c, (n_det, tp) in sorted(per_class.items()):
            log.info("  lớp %d: %4d hộp, %4d khớp người GT @IoU0.5", c, n_det, tp)
        log.info("  recall lớp %d trên GT: %d/%d", args.person_class_id, n_gt_hit, max(n_gt, 1))


if __name__ == "__main__":
    main()
