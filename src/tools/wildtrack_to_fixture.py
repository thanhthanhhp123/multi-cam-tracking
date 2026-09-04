"""Chuyển dataset WildTrack thành fixture metadata đúng contract `common/schema.py`.

Mục đích (CLAUDE.md §9, M4 kéo lên sớm): thay vì chỉ có fixture tổng hợp sinh tay
(`make_synthetic_fixture.py`), dựng một fixture từ dữ liệu THẬT có ground-truth Global ID
để thử engine liên kết (`src/mct/affinity.py`, `associator.py`) trước khi có pipeline
DeepStream. WildTrack: 7 camera HD tĩnh, cùng nhìn một quảng trường, chú thích ~2 fps,
mỗi người có `personID` nhất quán xuyên camera → đúng là bảng đáp án cho MTMCT.

Đầu ra giống hệt `make_synthetic_fixture.py`:
  - `<out>.jsonl`     — một `FrameMessage` mỗi (camera, frame)
  - `<out>.gt.json`   — bảng (cam_id, local_track_id) -> gt_global_id + vị trí mặt đất

QUY ƯỚC ÁNH XẠ:
  - `viewNum` 0..6  ->  `cam_id` "cam01".."cam07"
  - `personID`      ->  `gt_global_id` (WildTrack đã gán nhất quán xuyên camera)
  - `local_track_id` sinh mới: đếm tăng dần trong phạm vi từng camera. Một người rời FOV
    quá `--reentry-gap-frames` khung rồi quay lại thì được cấp local id MỚI — mô phỏng
    tracker thật mất dấu rồi bắt lại (idle_timeout trong `configs/mct.yaml`).

HẠN CHẾ ĐÃ BIẾT (ghi để báo cáo đối chiếu):
  - Mọi camera WildTrack đều overlap; KHÔNG có cặp non-overlap. Kịch bản non-overlap của
    đề cương (mục 4.2) phải chờ dataset tự thu ở M6.
  - `local_track_id` ở đây là SCT lý tưởng (chỉ đứt khi ra khỏi khung, không id-switch).
    Tracklet pipeline thật phân mảnh nhiều hơn.
  - Embedding trích bằng OSNet pretrained chạy ONNX Runtime trên CPU (`--reid-onnx`),
    KHÔNG phải model fine-tune sẽ chạy trong DeepStream. Coi như cận dưới về chất lượng.

Cài phần trích embedding:  pip install -e ".[reid]"
Lấy annotation (nhỏ):      python -m tools.fetch_wildtrack_annotations --dest data/wildtrack
Ảnh gốc (~13GB):           tải riêng từ https://www.epfl.ch/labs/cvlab/data/data-wildtrack/

Ví dụ:
    # fixture đầy đủ, 3 camera, có embedding
    python -m tools.wildtrack_to_fixture --wildtrack-dir data/wildtrack \
        --views 1,4,7 --reid-onnx models/reid/osnet_x1_0_market1501.onnx \
        --out tests/fixtures/wildtrack_3cam.jsonl

    # chỉ hình học (bbox + thời gian), không cần ảnh, chạy được trong CI
    python -m tools.wildtrack_to_fixture --wildtrack-dir data/wildtrack --no-reid \
        --out tests/fixtures/wildtrack_geom.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from common.logging import get_logger
from common.schema import Detection, FrameMessage, l2_normalize, validate, write_jsonl

log = get_logger("tools.wildtrack")

# Mốc thời gian trùng với make_synthetic_fixture để fixture hai nguồn cùng hệ quy chiếu:
# 2026-09-01T09:00:00Z.
BASE_TS_MS = 1_788_231_600_000

WILDTRACK_W = 1920
WILDTRACK_H = 1080
N_VIEWS = 7

# positionID nằm trên lưới 480 x 1440 (X trước), ô 2.5 cm, gốc (-3.0, -9.0) mét.
# Nguồn: WILDTRACK-toolkit + arxiv 1707.09299; đối chiếu code hou-yz/MVDet.
_GRID_COLS = 480
_GRID_ORIGIN_X_M = -3.0
_GRID_ORIGIN_Y_M = -9.0
_GRID_CELL_M = 0.025


class Embedder(Protocol):
    """Giao diện tối thiểu mà `build_fixture` cần — test tiêm bản giả, thật thì dùng
    `tools.reid_onnx.OsnetOnnxEmbedder`."""

    embed_dim: int

    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray: ...


# (đường dẫn ảnh) -> ảnh BGR HxWx3, hoặc None nếu không đọc được. Mặc định dùng OpenCV;
# tách ra tham số để test tiêm ảnh giả mà không cần cài opencv.
ImageReader = Callable[[Path], "np.ndarray | None"]


def position_id_to_world_m(position_id: int) -> tuple[float, float]:
    """positionID của WildTrack -> toạ độ (X, Y) mét trên mặt phẳng mặt đất tham chiếu."""
    grid_x = position_id % _GRID_COLS
    grid_y = position_id // _GRID_COLS
    return (
        _GRID_ORIGIN_X_M + _GRID_CELL_M * grid_x,
        _GRID_ORIGIN_Y_M + _GRID_CELL_M * grid_y,
    )


def cam_id_for_view(view_idx: int) -> str:
    """viewNum 0-based -> cam_id ("cam01"...), khớp key trong configs/cameras/topology.yaml."""
    return f"cam{view_idx + 1:02d}"


def frame_idx_to_ts_ms(frame_idx: int, fps: float) -> int:
    return BASE_TS_MS + round(frame_idx * 1000.0 / fps)


def frame_idx_to_pts_ns(frame_idx: int, fps: float) -> int:
    return round(frame_idx * 1_000_000_000.0 / fps)


def clip_bbox_xyxy(
    xmin: float, ymin: float, xmax: float, ymax: float, *, width: int, height: int
) -> tuple[float, float, float, float] | None:
    """(xmin, ymin, xmax, ymax) bất kỳ -> (x, y, w, h) đã clip vào khung, hoặc None nếu suy biến.

    WildTrack để bbox tràn khung khá nhiều (xmin xuống -177, xmax lên 2529) khi người
    đứng sát camera. Clip cứng vào [0, W] x [0, H] rồi loại hộp bị co gần về 0.
    """
    x0 = max(0.0, min(float(xmin), float(xmax)))
    y0 = max(0.0, min(float(ymin), float(ymax)))
    x1 = min(float(width), max(float(xmin), float(xmax)))
    y1 = min(float(height), max(float(ymin), float(ymax)))
    w, h = x1 - x0, y1 - y0
    if w <= 1.0 or h <= 1.0:
        return None
    return (x0, y0, w, h)


@dataclass(slots=True)
class RawDetection:
    """Một quan sát trước khi gán local_track_id."""

    frame_idx: int
    frame_number: int  # số trong tên file ảnh WildTrack (00000000.png ...)
    view_idx: int  # viewNum, 0-based
    person_id: int
    bbox: tuple[float, float, float, float]  # (x, y, w, h) đã clip
    world_xy: tuple[float, float]  # (X, Y) mét từ positionID
    local_track_id: int = -1
    embedding: np.ndarray | None = None


@dataclass(slots=True)
class TrackletGT:
    """Một tracklet cục bộ liên tục — đơn vị của bảng ground-truth."""

    cam_id: str
    local_track_id: int
    gt_global_id: int
    start_ms: int
    end_ms: int
    world_xy_m: tuple[float, float]
    n_frames: int


def _frame_files(ann_dir: Path, *, stride: int, max_frames: int) -> list[Path]:
    files = sorted(ann_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"{ann_dir} không có file .json nào")
    if stride > 1:
        files = files[::stride]
    if max_frames > 0:
        files = files[:max_frames]
    return files


def annotation_frame_numbers(
    ann_dir: str | Path, *, stride: int = 1, max_frames: int = 0
) -> list[int]:
    """Số khung WildTrack (00000000.png -> 0, 00000005.png -> 5...) theo ĐÚNG thứ tự mà
    `parse_raw_detections` đánh `frame_idx`.

    Public vì `tools.wildtrack_to_video` phải đóng video theo đúng thứ tự này: chỉ khi
    khung thứ i của video là `annotation_frame_numbers()[i]` thì `frame_id` mà probe
    DeepStream ghi ra mới tra thẳng được về khung chú thích. Gọi chung một hàm để hai
    đường không thể trôi khỏi nhau.
    """
    return [int(p.stem) for p in _frame_files(Path(ann_dir), stride=stride, max_frames=max_frames)]


def _normalize_views(views: list[int]) -> list[int]:
    out = sorted({int(v) - 1 for v in views})
    if not out or out[0] < 0 or out[-1] >= N_VIEWS:
        raise ValueError(f"--views phải nằm trong 1..{N_VIEWS}, nhận {views}")
    return out


def parse_raw_detections(
    ann_dir: Path,
    *,
    view_indices: list[int],
    stride: int,
    max_frames: int,
    min_box_area: float,
) -> tuple[list[RawDetection], int]:
    """Đọc annotations_positions/*.json -> danh sách RawDetection + số khung đã quét."""
    want = set(view_indices)
    files = _frame_files(ann_dir, stride=stride, max_frames=max_frames)
    raw: list[RawDetection] = []

    for frame_idx, path in enumerate(files):
        frame_number = int(path.stem)
        for person in json.loads(path.read_text(encoding="utf-8")):
            pid = int(person["personID"])
            world = position_id_to_world_m(int(person["positionID"]))
            for view in person["views"]:
                view_idx = int(view["viewNum"])
                if view_idx not in want:
                    continue
                if any(view[k] == -1 for k in ("xmin", "ymin", "xmax", "ymax")):
                    continue
                bbox = clip_bbox_xyxy(
                    view["xmin"],
                    view["ymin"],
                    view["xmax"],
                    view["ymax"],
                    width=WILDTRACK_W,
                    height=WILDTRACK_H,
                )
                if bbox is None or bbox[2] * bbox[3] < min_box_area:
                    continue
                raw.append(RawDetection(frame_idx, frame_number, view_idx, pid, bbox, world))

    return raw, len(files)


def _finalize_segment(
    segment: list[RawDetection], *, view_idx: int, person_id: int, local_id: int, fps: float
) -> TrackletGT:
    for det in segment:
        det.local_track_id = local_id
    frames = [d.frame_idx for d in segment]
    mean_xy = np.array([d.world_xy for d in segment], dtype=np.float64).mean(axis=0)
    return TrackletGT(
        cam_id=cam_id_for_view(view_idx),
        local_track_id=local_id,
        gt_global_id=person_id,
        start_ms=frame_idx_to_ts_ms(min(frames), fps),
        end_ms=frame_idx_to_ts_ms(max(frames), fps),
        world_xy_m=(float(mean_xy[0]), float(mean_xy[1])),
        n_frames=len(segment),
    )


def assign_local_tracks(
    raw: list[RawDetection], *, reentry_gap_frames: int, fps: float
) -> list[TrackletGT]:
    """Gán local_track_id (in-place lên `raw`) và trả về bảng tracklet ground-truth.

    Mỗi (camera, người) tách thành nhiều segment nếu có khoảng trống > reentry_gap_frames
    khung — quãng người đó ra khỏi khung hình rồi quay lại.
    """
    by_key: dict[tuple[int, int], list[RawDetection]] = defaultdict(list)
    for det in raw:
        by_key[(det.view_idx, det.person_id)].append(det)

    next_local: dict[int, int] = defaultdict(lambda: 1)
    tracklets: list[TrackletGT] = []

    for (view_idx, person_id), dets in sorted(by_key.items()):
        dets.sort(key=lambda d: d.frame_idx)
        segment: list[RawDetection] = []
        prev_idx: int | None = None

        for det in dets:
            if prev_idx is not None and det.frame_idx - prev_idx > reentry_gap_frames:
                local_id = next_local[view_idx]
                next_local[view_idx] += 1
                tracklets.append(
                    _finalize_segment(
                        segment, view_idx=view_idx, person_id=person_id, local_id=local_id, fps=fps
                    )
                )
                segment = []
            segment.append(det)
            prev_idx = det.frame_idx

        local_id = next_local[view_idx]
        next_local[view_idx] += 1
        tracklets.append(
            _finalize_segment(
                segment, view_idx=view_idx, person_id=person_id, local_id=local_id, fps=fps
            )
        )

    return tracklets


def crop_for_reid(image: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Cắt vùng ảnh cho ReID từ bbox `(x, y, w, h)`.

    Public vì `tools/reembed_fixture.py` phải cắt **y hệt** cách này: hai fixture sinh bằng
    hai đoạn code cắt khác nhau thì chênh lệch embedding đo được không quy về nguyên nhân
    nào. Một hàm, hai chỗ gọi.
    """
    x, y, w, h = (round(v) for v in bbox)
    x = max(0, min(x, WILDTRACK_W - 1))
    y = max(0, min(y, WILDTRACK_H - 1))
    x2 = min(WILDTRACK_W, x + max(w, 1))
    y2 = min(WILDTRACK_H, y + max(h, 1))
    return image[y:y2, x:x2]


def _default_image_reader(path: Path) -> np.ndarray | None:
    import cv2

    return cv2.imread(str(path))


def _attach_embeddings(
    raw: list[RawDetection],
    *,
    wildtrack_dir: Path,
    image_subdir_fmt: str,
    embedder: Embedder,
    image_reader: ImageReader | None = None,
) -> None:
    read_image = image_reader or _default_image_reader
    image_root = wildtrack_dir / "Image_subsets"
    if image_reader is None and not image_root.is_dir():
        raise FileNotFoundError(
            f"{image_root} không tồn tại — cần ảnh gốc WildTrack (~13GB) để trích embedding. "
            "Tải từ EPFL, hoặc chạy lại với --no-reid để tạo fixture chỉ hình học."
        )

    by_image: dict[tuple[int, int], list[RawDetection]] = defaultdict(list)
    for det in raw:
        by_image[(det.view_idx, det.frame_number)].append(det)

    items = sorted(by_image.items())
    for done, ((view_idx, frame_number), dets) in enumerate(items, start=1):
        subdir = image_subdir_fmt.format(n=view_idx + 1)
        img_path = image_root / subdir / f"{frame_number:08d}.png"
        image = read_image(img_path)
        if image is None:
            raise FileNotFoundError(
                f"Không đọc được ảnh {img_path} — kiểm tra --wildtrack-dir và --image-subdir-fmt"
            )

        crops = [crop_for_reid(image, det.bbox) for det in dets]

        feats = embedder.embed(crops)
        for det, feat in zip(dets, feats, strict=True):
            det.embedding = l2_normalize(feat)

        if done % 200 == 0 or done == len(items):
            log.info("trích embedding: %d/%d ảnh", done, len(items))


def _build_messages(
    raw: list[RawDetection], *, view_indices: list[int], n_frames: int, fps: float, embed_dim: int
) -> list[FrameMessage]:
    by_slot: dict[tuple[int, int], list[RawDetection]] = defaultdict(list)
    for det in raw:
        by_slot[(det.view_idx, det.frame_idx)].append(det)

    messages: list[FrameMessage] = []
    for view_idx in view_indices:
        for frame_idx in range(n_frames):
            dets = sorted(by_slot.get((view_idx, frame_idx), []), key=lambda d: d.local_track_id)
            messages.append(
                FrameMessage(
                    cam_id=cam_id_for_view(view_idx),
                    frame_id=frame_idx,
                    ts_ms=frame_idx_to_ts_ms(frame_idx, fps),
                    frame_pts_ns=frame_idx_to_pts_ns(frame_idx, fps),
                    frame_width=WILDTRACK_W,
                    frame_height=WILDTRACK_H,
                    detections=[
                        Detection(
                            local_track_id=d.local_track_id,
                            bbox=d.bbox,
                            confidence=1.0,  # ground truth: không có điểm tin cậy detector
                            embedding=d.embedding,
                        )
                        for d in dets
                    ],
                    embed_dim=embed_dim,
                )
            )

    messages.sort(key=lambda m: (m.ts_ms, m.cam_id))
    return messages


def build_fixture(
    wildtrack_dir: str | Path,
    *,
    views: list[int],
    fps: float = 2.0,
    frame_stride: int = 1,
    max_frames: int = 0,
    min_box_area: float = 400.0,
    reentry_gap_frames: int = 4,
    image_subdir_fmt: str = "C{n}",
    embedder: Embedder | None = None,
    image_reader: ImageReader | None = None,
) -> tuple[list[FrameMessage], list[TrackletGT], dict]:
    """WildTrack -> (messages, tracklets ground-truth, meta). `embedder=None` -> fixture
    chỉ hình học (không đọc ảnh, chạy được không cần dataset ảnh)."""
    wildtrack_dir = Path(wildtrack_dir)
    ann_dir = wildtrack_dir / "annotations_positions"
    if not ann_dir.is_dir():
        raise FileNotFoundError(
            f"{ann_dir} không tồn tại. Lấy annotation bằng "
            f"`python -m tools.fetch_wildtrack_annotations --dest {wildtrack_dir}`."
        )

    view_indices = _normalize_views(views)
    raw, n_frames = parse_raw_detections(
        ann_dir,
        view_indices=view_indices,
        stride=frame_stride,
        max_frames=max_frames,
        min_box_area=min_box_area,
    )
    if not raw:
        raise ValueError("Không có detection nào sau khi lọc — xem lại --views / --min-box-area")

    tracklets = assign_local_tracks(raw, reentry_gap_frames=reentry_gap_frames, fps=fps)

    embed_dim = 0
    reid_model: str | None = None
    if embedder is not None:
        _attach_embeddings(
            raw,
            wildtrack_dir=wildtrack_dir,
            image_subdir_fmt=image_subdir_fmt,
            embedder=embedder,
            image_reader=image_reader,
        )
        embed_dim = int(embedder.embed_dim)
        onnx_path = getattr(embedder, "onnx_path", None)
        reid_model = Path(onnx_path).name if onnx_path else type(embedder).__name__

    messages = _build_messages(
        raw, view_indices=view_indices, n_frames=n_frames, fps=fps, embed_dim=embed_dim
    )

    meta = {
        "source": "wildtrack",
        "views": [vi + 1 for vi in view_indices],
        "cam_ids": [cam_id_for_view(vi) for vi in view_indices],
        "fps": fps,
        "frame_stride": frame_stride,
        "n_frames_per_cam": n_frames,
        "n_messages": len(messages),
        "n_detections": len(raw),
        "n_tracklets": len(tracklets),
        "n_identities": len({t.gt_global_id for t in tracklets}),
        "embed_dim": embed_dim,
        "reid_model": reid_model,
        "reentry_gap_frames": reentry_gap_frames,
        "min_box_area": min_box_area,
        "notes": (
            "WildTrack: mọi camera đều overlap, không có cặp non-overlap. "
            "local_track_id mô phỏng SCT lý tưởng (chỉ đứt khi ra khỏi FOV)."
        ),
    }
    return messages, tracklets, meta


def write_ground_truth(path: Path, tracklets: list[TrackletGT], meta: dict) -> None:
    """Bảng (cam_id, local_track_id) -> gt_global_id, cùng định dạng make_synthetic_fixture."""
    payload = {
        "scenario": path.name.removesuffix(".gt.json"),
        "meta": meta,
        "tracklets": [
            {
                "cam_id": t.cam_id,
                "local_track_id": t.local_track_id,
                "gt_global_id": t.gt_global_id,
                "start_ms": t.start_ms,
                "end_ms": t.end_ms,
                "world_xy_m": [round(t.world_xy_m[0], 3), round(t.world_xy_m[1], 3)],
                "n_frames": t.n_frames,
            }
            for t in sorted(tracklets, key=lambda t: (t.cam_id, t.local_track_id))
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--wildtrack-dir",
        type=Path,
        required=True,
        help="thư mục chứa annotations_positions/ (và Image_subsets/ nếu trích embedding)",
    )
    p.add_argument("--out", type=Path, default=Path("tests/fixtures/wildtrack.jsonl"))
    p.add_argument("--views", default="1,2,3,4,5,6,7", help="số camera 1..7, phẩy ngăn cách")
    p.add_argument("--fps", type=float, default=2.0, help="WildTrack chú thích ở ~2 fps")
    p.add_argument("--frame-stride", type=int, default=1, help="lấy 1 trong mỗi N khung chú thích")
    p.add_argument("--max-frames", type=int, default=0, help="0 = tất cả 400 khung")
    p.add_argument("--min-box-area", type=float, default=400.0, help="loại bbox nhỏ hơn (px^2)")
    p.add_argument(
        "--reentry-gap-frames",
        type=int,
        default=4,
        help="ra khỏi FOV quá số khung này rồi quay lại -> cấp local_track_id mới",
    )
    p.add_argument(
        "--image-subdir-fmt", default="C{n}", help="tên thư mục con trong Image_subsets/ ({n}=1..7)"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--reid-onnx", type=Path, help="model OSNet .onnx để trích embedding")
    src.add_argument(
        "--no-reid", action="store_true", help="fixture chỉ hình học, không trích embedding"
    )
    p.add_argument("--reid-batch", type=int, default=32)
    args = p.parse_args(argv)

    embedder: Embedder | None = None
    if not args.no_reid:
        from tools.reid_onnx import OsnetOnnxEmbedder

        embedder = OsnetOnnxEmbedder(args.reid_onnx, batch_size=args.reid_batch)

    views = [int(v) for v in str(args.views).split(",") if v.strip()]
    messages, tracklets, meta = build_fixture(
        args.wildtrack_dir,
        views=views,
        fps=args.fps,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        min_box_area=args.min_box_area,
        reentry_gap_frames=args.reentry_gap_frames,
        image_subdir_fmt=args.image_subdir_fmt,
        embedder=embedder,
    )

    for msg in messages:
        validate(msg, strict=True)

    n = write_jsonl(args.out, messages)
    gt_path = args.out.with_name(args.out.stem + ".gt.json")
    write_ground_truth(gt_path, tracklets, meta)

    span_s = (messages[-1].ts_ms - messages[0].ts_ms) / 1000.0 if messages else 0.0
    log.info(
        "%s: %d message, %d detection, %.1fs, %d camera, %d danh tính, %d tracklet, embed_dim=%d",
        args.out,
        n,
        meta["n_detections"],
        span_s,
        len(meta["cam_ids"]),
        meta["n_identities"],
        meta["n_tracklets"],
        meta["embed_dim"],
    )
    log.info("%s: bảng ground-truth Global ID", gt_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
