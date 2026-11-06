"""Gán nhãn trước cho CVAT (M6): detector + tracker → track có sẵn hộp, người chỉ sửa lại.

    # trên máy GPU (docker/vast_prelabel.sh), cần torch + ultralytics==8.4.7:
    PYTHONPATH=src python -m tools.prelabel_cvat data/lab/calib/cam0*.mp4 \\
        --out-dir data/prelabel --model yolo11x.pt --imgsz 1280 --min-foot-y cam01=250

Ra `data/prelabel/<buổi>/camXX.xml`, định dạng "CVAT for video 1.1". Nạp vào task CVAT bằng
"Upload annotations" → "CVAT 1.1" (task phải tạo với Frame step = 5 như docs/m6/README.md §5).

**Vì sao.** 14 buổi × 3 camera ở 5 fps là khoảng 30 000 khung chú thích. Vẽ tay từ đầu thì
một lượt đi trên một camera mất khoảng 10 phút (thử 2026-10-28). Có track sẵn thì người gán
nhãn chỉ sửa hộp lệch, xoá track rác, thêm người bị sót và điền `person_id`.

**Ba chỗ cố ý để người quyết, không để máy:**

1. `person_id` để TRỐNG. `tools.cvat_to_mot` báo lỗi mọi hộp thiếu danh tính, nên không track
   nào lọt vào ground-truth mà chưa qua tay người.
2. Detector gán trước nên KHÁC detector đang được chấm (YOLO11s 640 của pipeline): dùng chính
   nó thì người nó bỏ sót dễ bị sót luôn trong ground-truth, và điểm của nó bị thổi lên. Báo
   cáo phải ghi detector + tracker đã dùng để gán trước, và rằng mọi hộp đã qua người.
3. Giữ cả hộp điểm thấp (mặc định của tracker): xoá một track rác dễ hơn vẽ lại một người
   bị sót.

**Khung.** Tracker chạy MỌI khung của video (ByteTrack cần khung liền nhau), nhưng chỉ giữ
khung thuộc task (`khung % step == 0`). Số khung trong XML là số khung VIDEO, giống bản CVAT
tự xuất, nên CVAT nhập lại được và `tools.cvat_to_mot` đọc thẳng được.

**Khung khoá.** Tracker cho hộp ở mọi khung; nạp nguyên như vậy thì khung nào cũng là khung
khoá, và sửa một hộp phải sửa từng khung. Nên chỉ giữ số khung khoá tối thiểu sao cho phép
nội suy tuyến tính của CVAT vẫn trùng hộp tracker với IoU ≥ `--key-iou` (kiểu Douglas–Peucker
trên quỹ đạo hộp). Track mất dấu giữa chừng thì chèn một khung `outside`.

**`--min-foot-y`.** cam01 nhìn dọc hành lang tới tận vùng cam03; pipeline bỏ hộp có chân ở
x ≥ 6 m (`ground_roi`), và quy ước chú thích chỉ đòi người có chân dưới y ≈ 270 (docs/m6/README.md
§5 mục 6). Hộp có chân cao hơn ngưỡng này bị bỏ trước khi tạo track — không thì người gán nhãn
phải điền `person_id` cho cả đám người tí hon ở cuối hành lang.

`ultralytics`/`cv2` chỉ import khi chạy thật; phần còn lại (rút khung khoá, ghi XML) là stdlib
và có test trên máy dev.
"""

from __future__ import annotations

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from common.logging import get_logger

log = get_logger("tools.prelabel_cvat")

Box = tuple[float, float, float, float]  # xtl, ytl, xbr, ybr — pixel ảnh gốc

PERSON_CLASS = 0  # COCO
DEFAULT_TRACKER = Path("configs/lab/prelabel_bytetrack.yaml")


@dataclass(slots=True)
class Track:
    track_id: int
    frames: list[int] = field(default_factory=list)  # số khung VIDEO, tăng dần
    boxes: list[Box] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Key:
    frame: int
    box: Box
    outside: bool = False


def iou(a: Box, b: Box) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def giou(a: Box, b: Box) -> float:
    """Generalized IoU: bằng IoU trừ phần hộp bao chung không phủ; âm khi hai hộp rời nhau.

    Chỉ dùng để XẾP HẠNG khung lệch nhất: hai hộp rời nhau đều có IoU = 0 dù cách nhau
    10 px hay 500 px, còn GIoU vẫn phân biệt được.
    """
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union -= iw * ih
    hull = (max(a[2], b[2]) - min(a[0], b[0])) * (max(a[3], b[3]) - min(a[1], b[1]))
    if union <= 0 or hull <= 0:
        return 0.0
    return iw * ih / union - (hull - union) / hull


def _lerp(a: Box, b: Box, t: float) -> Box:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    )


def keyframe_indices(frames: list[int], boxes: list[Box], min_iou: float) -> list[int]:
    """Chỉ số khung phải giữ làm khung khoá trong MỘT đoạn liền mạch của track.

    Giữ hai đầu, rồi lặp: trong các khung mà hộp nội suy tuyến tính giữa hai khung khoá kề
    nó có IoU với hộp thật dưới `min_iou`, khung lệch nhất (GIoU thấp nhất) thành khung khoá.
    """
    if not frames:
        return []
    keep = {0, len(frames) - 1}
    stack = [(0, len(frames) - 1)]
    while stack:
        lo, hi = stack.pop()
        worst, worst_giou = -1, 1.0
        for i in range(lo + 1, hi):
            t = (frames[i] - frames[lo]) / (frames[hi] - frames[lo])
            guess = _lerp(boxes[lo], boxes[hi], t)
            if iou(guess, boxes[i]) < min_iou and giou(guess, boxes[i]) < worst_giou:
                worst, worst_giou = i, giou(guess, boxes[i])
        if worst >= 0:
            keep.add(worst)
            stack += [(lo, worst), (worst, hi)]
    return sorted(keep)


def track_keys(track: Track, *, step: int, stop_frame: int, min_iou: float) -> list[Key]:
    """Khung khoá CVAT của một track: cắt đoạn ở chỗ hụt khung, cuối mỗi đoạn một `outside`."""
    keys: list[Key] = []
    start = 0
    n = len(track.frames)
    for end in range(n):
        if end < n - 1 and track.frames[end + 1] - track.frames[end] <= step:
            continue
        seg_frames = track.frames[start : end + 1]
        seg_boxes = track.boxes[start : end + 1]
        keys += [
            Key(seg_frames[i], seg_boxes[i])
            for i in keyframe_indices(seg_frames, seg_boxes, min_iou)
        ]
        after = seg_frames[-1] + step
        if after <= stop_frame:
            keys.append(Key(after, seg_boxes[-1], outside=True))
        start = end + 1
    return keys


def clean_tracks(
    tracks: list[Track], *, min_frames: int, min_foot_y: float | None = None
) -> list[Track]:
    """Bỏ hộp có chân cao hơn `min_foot_y`, bỏ track còn ít hơn `min_frames` khung.

    Đánh số lại 0, 1, 2… theo khung xuất hiện đầu tiên, để chạy lại ra cùng một file.
    """
    kept: list[Track] = []
    for track in tracks:
        pairs = [
            (f, b)
            for f, b in zip(track.frames, track.boxes, strict=True)
            if min_foot_y is None or b[3] >= min_foot_y
        ]
        if len(pairs) >= min_frames:
            kept.append(Track(track.track_id, [f for f, _ in pairs], [b for _, b in pairs]))
    kept.sort(key=lambda t: (t.frames[0], t.track_id))
    return [Track(i, t.frames, t.boxes) for i, t in enumerate(kept)]


def to_cvat_xml(
    tracks: list[Track],
    *,
    step: int,
    stop_frame: int,
    size: tuple[int, int],
    min_iou: float,
    label: str = "person",
    attribute: str = "person_id",
) -> ET.ElementTree:
    """Ghi "CVAT for video 1.1": `<meta>` đủ cho `tools.cvat_to_mot`, `person_id` để trống."""
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    task = ET.SubElement(ET.SubElement(root, "meta"), "task")
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str(stop_frame)
    ET.SubElement(task, "frame_filter").text = f"step={step}"
    original = ET.SubElement(task, "original_size")
    ET.SubElement(original, "width").text = str(size[0])
    ET.SubElement(original, "height").text = str(size[1])

    for track in tracks:
        node = ET.SubElement(root, "track", id=str(track.track_id), label=label, source="auto")
        for key in track_keys(track, step=step, stop_frame=stop_frame, min_iou=min_iou):
            box = ET.SubElement(
                node,
                "box",
                frame=str(key.frame),
                keyframe="1",
                outside="1" if key.outside else "0",
                occluded="0",
                xtl=f"{key.box[0]:.2f}",
                ytl=f"{key.box[1]:.2f}",
                xbr=f"{key.box[2]:.2f}",
                ybr=f"{key.box[3]:.2f}",
                z_order="0",
            )
            ET.SubElement(box, "attribute", name=attribute).text = ""
    ET.indent(root)
    return ET.ElementTree(root)


def run_tracker(
    video: Path,
    *,
    model: str,
    tracker: Path,
    imgsz: int,
    step: int,
    device: str,
    max_frames: int | None = None,
) -> tuple[list[Track], int, tuple[int, int]]:
    """Chạy detector + tracker trên MỌI khung, giữ hộp người ở khung thuộc task.

    Trả (track, số khung đã đọc, (rộng, cao)). Mỗi video một đối tượng YOLO mới, để tracker
    không mang trạng thái từ video trước sang.
    """
    import cv2
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    tracks: dict[int, Track] = {}
    n_read = 0
    results = YOLO(model).track(
        source=str(video),
        stream=True,
        persist=False,
        tracker=str(tracker),
        classes=[PERSON_CLASS],
        imgsz=imgsz,
        device=device,
        half=device != "cpu",
        verbose=False,
    )
    for frame, result in enumerate(results):
        if max_frames is not None and frame >= max_frames:
            break
        n_read = frame + 1
        boxes = result.boxes
        if frame % step or boxes is None or boxes.id is None:
            continue
        for tid, xyxy in zip(boxes.id.int().tolist(), boxes.xyxy.tolist(), strict=True):
            track = tracks.setdefault(tid, Track(tid))
            track.frames.append(frame)
            track.boxes.append(
                (
                    max(0.0, xyxy[0]),
                    max(0.0, xyxy[1]),
                    min(float(size[0]), xyxy[2]),
                    min(float(size[1]), xyxy[3]),
                )
            )
    return list(tracks.values()), n_read, size


def _parse_foot(values: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for value in values:
        cam, sep, y = value.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"--min-foot-y cần dạng cam01=250, nhận {value!r}")
        out[cam] = float(y)
    return out


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("videos", type=Path, nargs="+", help="data/lab/<buổi>/camXX.mp4 ĐÃ đồng bộ")
    p.add_argument("--out-dir", type=Path, default=Path("data/prelabel"))
    p.add_argument("--model", default="yolo11x.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    p.add_argument("--step", type=int, default=5, help="Frame step của task CVAT")
    p.add_argument("--key-iou", type=float, default=0.85)
    p.add_argument("--min-frames", type=int, default=5, help="khung TASK; 5 = 1 s ở 5 fps")
    p.add_argument("--min-foot-y", action="append", default=[], metavar="CAM=Y")
    p.add_argument("--device", default="0")
    p.add_argument("--max-frames", type=int, default=None, help="chỉ để thử")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    foot = _parse_foot(args.min_foot_y)

    for video in args.videos:
        out = args.out_dir / video.parent.name / f"{video.stem}.xml"
        if out.exists() and not args.overwrite:
            print(f"{out}: đã có, bỏ qua (--overwrite để chạy lại)")
            continue
        t0 = time.monotonic()
        raw, n_read, size = run_tracker(
            video,
            model=args.model,
            tracker=args.tracker,
            imgsz=args.imgsz,
            step=args.step,
            device=args.device,
            max_frames=args.max_frames,
        )
        tracks = clean_tracks(raw, min_frames=args.min_frames, min_foot_y=foot.get(video.stem))
        stop_frame = (n_read - 1) // args.step * args.step
        tree = to_cvat_xml(
            tracks, step=args.step, stop_frame=stop_frame, size=size, min_iou=args.key_iou
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out, encoding="utf-8", xml_declaration=True)
        n_keys = sum(1 for _ in tree.getroot().iter("box"))
        n_boxes = sum(len(t.frames) for t in tracks)
        elapsed = time.monotonic() - t0
        print(
            f"{out}: {n_read} khung video ({n_read / elapsed:.1f} khung/s), "
            f"{len(raw)} track thô -> {len(tracks)} track, {n_boxes} hộp, {n_keys} khung khoá"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
