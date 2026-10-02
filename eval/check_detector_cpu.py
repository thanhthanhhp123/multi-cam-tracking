"""Compare YOLO detectors (DeepStream-Yolo ONNX) on CPU against WildTrack, BEFORE renting a GPU.

    PYTHONPATH="src;." ~/.venvs/mct-reid/Scripts/python.exe -m eval.check_detector_cpu \\
        --model yolo11s models/detector/yolo11s.onnx \\
        --model yolo26s models/detector/yolo26s.onnx \\
        --wildtrack data/wildtrack --homography-dir configs/cameras/homography/wildtrack

Why this exists (docs/worklog/2026-10-02-32-*): swapping the detector is a full GPU session
(n = 3 runs, both scoring protocols, 4-stream FPS), and two things can sink it for free first.
The ONNX may produce no person boxes in the DeepStream-Yolo format (DeepStream-Yolo issue
#688). And a detector that "sees more people" may only see more people OUTSIDE the grid
WildTrack annotates, which is exactly what made PeopleNet lose (docs/worklog/2026-09-28-30-*:
84% of its extra false boxes had their foot off the grid).

What it mimics, so the numbers mean the same thing as in the pipeline
(configs/pipeline/config_infer_yolo*.txt):

- letterbox `maintain-aspect-ratio=1` + `symmetric-padding=1`, padded with 0 like nvinfer
  (not the 114 grey of Ultralytics), RGB, `net-scale-factor` 1/255;
- `NvDsInferParseYolo`: output `[N, 6]` = x1, y1, x2, y2 (network pixels), score, class; a box
  survives if its score reaches its class's `pre-cluster-threshold` (person only: every other
  class has threshold 1.0), is clamped to the network frame and is at least 1 px wide and high;
- `cluster-mode=2` (greedy NMS, IoU 0.45, as YOLO11 runs) or `cluster-mode=4` (no clustering,
  what DeepStream-Yolo recommends for the NMS-free YOLO26), then `topk` 300.

What it does NOT tell you: TensorRT FP16 numbers, tracking, or HOTA. Phiên 25 showed recall can
go up while HOTA goes down, so this is a filter before the GPU run, not a verdict. Every
detection is put in one bucket with `eval.diagnose_fp_region.classify_frame`: matched to an
annotated person (IoU >= 0.5), `inside_near_gt` (on the grid, overlaps a person: duplicate or
poor box), `inside_no_gt` (on the grid, nobody there) or `outside_area` (foot off the grid).

`cv2` and `onnxruntime` are imported only when running (the `mct-reid` venv has both), so the
pure functions below stay importable in the light test venv.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from common.logging import get_logger
from eval.diagnose_fp_region import classify_frame
from mct.homography import CameraHomography

log = get_logger("eval.check_detector_cpu")

NET_SIZE = 640
PERSON_CLASS_ID = 0
NMS_IOU = 0.45
TOP_K = 300
"""Same as `[class-attrs-*]` in configs/pipeline/config_infer_yolo11*.txt."""


@dataclass(slots=True, frozen=True)
class Letterbox:
    """How a frame was fitted into the square network input (nvinfer, symmetric padding)."""

    scale: float
    pad_x: float
    pad_y: float

    @classmethod
    def fit(cls, frame_w: int, frame_h: int, net: int = NET_SIZE) -> Letterbox:
        scale = min(net / frame_w, net / frame_h)
        return cls(scale, (net - frame_w * scale) / 2.0, (net - frame_h * scale) / 2.0)


def preprocess(bgr: np.ndarray, net: int = NET_SIZE) -> tuple[np.ndarray, Letterbox]:
    """BGR frame -> NCHW float32 network input, as nvinfer would build it."""
    import cv2

    h, w = bgr.shape[:2]
    box = Letterbox.fit(w, h, net)
    new_w, new_h = round(w * box.scale), round(h * box.scale)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((net, net, 3), dtype=np.uint8)
    top, left = int(box.pad_y), int(box.pad_x)
    canvas[top : top + new_h, left : left + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None], box


def decode(
    output: np.ndarray,
    box: Letterbox,
    *,
    threshold: float,
    person_class_id: int = PERSON_CLASS_ID,
    net: int = NET_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """`NvDsInferParseYolo` for one image: `[N, 6]` -> (boxes x1y1x2y2 in frame pixels, scores)."""
    out = np.asarray(output, dtype=np.float32).reshape(-1, 6)
    keep = (out[:, 5].astype(np.int64) == person_class_id) & (out[:, 4] >= threshold)
    out = out[keep]
    xyxy = np.clip(out[:, :4], 0.0, float(net))
    wide = (xyxy[:, 2] - xyxy[:, 0] >= 1.0) & (xyxy[:, 3] - xyxy[:, 1] >= 1.0)
    xyxy, scores = xyxy[wide], out[wide, 4]
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - box.pad_x) / box.scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - box.pad_y) / box.scale
    return xyxy, scores


def _iou_one(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def cluster(
    boxes: np.ndarray, scores: np.ndarray, mode: str, *, iou: float = NMS_IOU, top_k: int = TOP_K
) -> np.ndarray:
    """Indices kept by `cluster-mode` `nms` (2) or `none` (4), highest score first, `top_k` max."""
    order = np.argsort(-scores, kind="stable")
    if mode == "none":
        return order[:top_k]
    if mode != "nms":
        raise ValueError(f"unknown cluster mode: {mode}")
    kept: list[int] = []
    alive = order
    while alive.size and len(kept) < top_k:
        best = int(alive[0])
        kept.append(best)
        rest = alive[1:]
        alive = rest[_iou_one(boxes[best], boxes[rest]) < iou]
    return np.array(kept, dtype=np.int64)


def wildtrack_gt_xywh(annotations_dir: Path, frame: str, view: int) -> np.ndarray:
    """Annotated person boxes of one view in one frame, as (x, y, w, h)."""
    rows = []
    for person in json.loads((annotations_dir / f"{frame}.json").read_text(encoding="utf-8")):
        v = person["views"][view]
        if v["xmin"] >= 0:
            rows.append((v["xmin"], v["ymin"], v["xmax"] - v["xmin"], v["ymax"] - v["ymin"]))
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


def pick_frames(annotations_dir: Path, n_frames: int) -> list[str]:
    """`n_frames` annotated frames spread evenly over the sequence (not the first ones only)."""
    names = sorted(p.stem for p in annotations_dir.glob("*.json"))
    if not names:
        raise FileNotFoundError(f"no annotation in {annotations_dir}")
    idx = np.linspace(0, len(names) - 1, num=min(n_frames, len(names))).round().astype(int)
    return [names[i] for i in sorted(set(idx.tolist()))]


@dataclass(slots=True)
class Tally:
    n_images: int = 0
    seconds: float = 0.0
    n_gt: int = 0
    n_det: int = 0
    n_tp: int = 0
    buckets: Counter[str] = field(default_factory=Counter)

    def row(self) -> dict[str, Any]:
        inside_fp = self.buckets["inside_near_gt"] + self.buckets["inside_no_gt"]
        return {
            "images": self.n_images,
            "cpu_s_per_image": round(self.seconds / max(self.n_images, 1), 3),
            "gt": self.n_gt,
            "det": self.n_det,
            "tp": self.n_tp,
            "recall": round(self.n_tp / max(self.n_gt, 1), 3),
            "precision_full_frame": round(self.n_tp / max(self.n_det, 1), 3),
            "precision_in_area": round(self.n_tp / max(self.n_tp + inside_fp, 1), 3),
            "outside_area": self.buckets["outside_area"],
            "inside_near_gt": self.buckets["inside_near_gt"],
            "inside_no_gt": self.buckets["inside_no_gt"],
        }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", nargs=2, action="append", required=True, metavar=("LABEL", "ONNX"))
    p.add_argument("--wildtrack", type=Path, required=True, help="unpacked WildTrack directory")
    p.add_argument("--homography-dir", type=Path, required=True)
    p.add_argument("--views", type=int, nargs="+", default=list(range(7)), help="0..6 = C1..C7")
    p.add_argument("--n-frames", type=int, default=10, help="annotated frames, spread evenly")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.25])
    p.add_argument("--cluster", nargs="+", choices=("nms", "none"), default=["nms", "none"])
    p.add_argument("--min-iou", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    import cv2
    import onnxruntime as ort

    annotations = args.wildtrack / "annotations_positions"
    frames = pick_frames(annotations, args.n_frames)
    homographies = {
        view: CameraHomography.load(args.homography_dir / f"cam{view + 1:02d}.yaml")
        for view in args.views
    }
    images = {
        (frame, view): args.wildtrack / "Image_subsets" / f"C{view + 1}" / f"{frame}.png"
        for frame in frames
        for view in args.views
    }

    results: dict[str, dict[str, Any]] = {}
    for label, onnx_path in args.model:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = args.threads
        session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        tallies = {(t, m): Tally() for t in args.thresholds for m in args.cluster}
        for (frame, view), path in images.items():
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise FileNotFoundError(path)
            tensor, box = preprocess(bgr)
            t0 = time.perf_counter()
            (output,) = session.run(None, {input_name: tensor})
            seconds = time.perf_counter() - t0
            gt = wildtrack_gt_xywh(annotations, frame, view)
            for (threshold, mode), tally in tallies.items():
                xyxy, scores = decode(output[0], box, threshold=threshold)
                kept = xyxy[cluster(xyxy, scores, mode)]
                xywh = np.column_stack([kept[:, :2], kept[:, 2:] - kept[:, :2]])
                n_tp, buckets = classify_frame(xywh, gt, homographies[view], args.min_iou)
                tally.n_images += 1
                tally.seconds += seconds
                tally.n_gt += len(gt)
                tally.n_det += len(kept)
                tally.n_tp += n_tp
                tally.buckets.update(buckets)
        for (threshold, mode), tally in tallies.items():
            key = f"{label} thr={threshold:g} cluster={mode}"
            results[key] = tally.row()
            log.info("%s: %s", key, json.dumps(results[key], ensure_ascii=False))

    head = [
        "model / threshold / cluster",
        "det",
        "TP",
        "recall",
        "precision (full frame)",
        "precision (in area)",
        "outside area",
        "dup/poor in area",
        "ghost in area",
        "CPU s/img",
    ]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for key, r in results.items():
        cells = [
            key,
            r["det"],
            r["tp"],
            f"{r['recall']:.3f}",
            f"{r['precision_full_frame']:.3f}",
            f"{r['precision_in_area']:.3f}",
            r["outside_area"],
            r["inside_near_gt"],
            r["inside_no_gt"],
            f"{r['cpu_s_per_image']:.2f}",
        ]
        print("| " + " | ".join(str(c) for c in cells) + " |")
    first = next(iter(results.values()))
    n_images, n_gt = first["images"], first["gt"]
    print(f"\n{n_images} images ({len(frames)} frames x {len(args.views)} views), GT {n_gt}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"frames": frames, "views": args.views, "results": results}
        args.json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
