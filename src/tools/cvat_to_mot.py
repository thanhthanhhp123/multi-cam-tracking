"""Chú thích CVAT (một task mỗi camera) → ground-truth MOT + bảng Global ID.

    PYTHONPATH=src python -m tools.cvat_to_mot \\
        --annotation cam01=data/cvat/cam01.xml --annotation cam02=data/cvat/cam02.xml \\
        --out-dir eval/gt/lab_2cam \\
        --fixture-out data/fixtures/lab_2cam_gt.jsonl --fps 25

**Vị trí trong quy trình M6** (CLAUDE.md §7): quay video → chú thích trên CVAT → công cụ này
→ `eval/gt/` → `eval/run_trackeval.py`. Đây là chỗ duy nhất biết cách đọc CVAT, để nếu đổi
công cụ chú thích thì chỉ phải sửa một file.

**Định dạng đầu vào: "CVAT for video 1.1" (XML).** Chọn bản này chứ không phải bản MOT mà
CVAT xuất sẵn, vì bản MOT **làm mất thuộc tính** — mà thuộc tính chính là chỗ mang danh tính
xuyên camera. Cấu trúc cần đọc:

    <track id="0" label="person">
      <box frame="0" xtl="..." ytl="..." xbr="..." ybr="..." outside="0" occluded="0">
        <attribute name="person_id">P03</attribute>
      </box>
    </track>

**QUY ƯỚC CHÚ THÍCH — phải phổ biến cho người gán nhãn trước khi họ bắt đầu:**

1. Mỗi người là **một track** trong task của camera đó.
2. Mỗi track mang thuộc tính `person_id` (đổi tên bằng `--attribute`) với giá trị **giống
   nhau ở mọi camera** cho cùng một người — ví dụ `P01`. Đây là thứ duy nhất nối danh tính
   giữa các task, vì `track id` của CVAT chỉ duy nhất trong một task.
3. Khung có `outside="1"` là người đã ra khỏi khung — CVAT vẫn ghi hộp ở đó, **không tính**.
4. `occluded="1"` vẫn tính (người bị che một phần vẫn là ground-truth hợp lệ), nhưng được
   ghi `visibility=0.5` để TrackEval biết mà xử lý.

Thiếu quy ước 2 thì công cụ vẫn chạy nhưng mỗi camera thành một tập danh tính riêng, và mọi
chỉ số xuyên camera trở nên vô nghĩa — nên nó **báo lỗi** thay vì đoán, trừ khi `--no-global`.

Đầu ra:

    <out-dir>/<cam_id>.gt.txt   ground-truth MOT một camera (id = track id của CVAT)
    <out-dir>/global_ids.gt.json  bảng (cam_id, local_track_id) -> gt_global_id

Bảng `.gt.json` **cùng định dạng** với `wildtrack_to_fixture.py` và `ds_wildtrack_gt.py`, nên
`eval/eval_wildtrack.py` và `tools/export_trackeval.py` dùng lại được ngay.

**Fixture ground-truth (`--fixture-out`, M6).** `tools/export_trackeval.py --gt-fixture` và
`tools/assign_gt.py` đọc ground-truth dưới dạng fixture JSONL (cùng schema với kết quả của
pipeline), không đọc `gt.txt`. Nên công cụ này ghi thêm `<fixture>.jsonl` + `<fixture>.gt.json`:
mỗi (camera, khung ĐÃ CHÚ THÍCH) một message — kể cả khung không có ai, vì "khung chú thích
rỗng" khác hẳn "khung không chú thích" khi chấm điểm (`export_trackeval --only-gt-frames`).

**Số khung: phải là số khung của VIDEO, không phải của task CVAT.** `frame_id` của pipeline
đếm từng khung của file video (từ 0). Task CVAT có thể chỉ lấy một phần video (`start_frame`,
`stop_frame`) hoặc nhảy khung (`frame_filter` = `step=N`, cách đề cương mục 4.3.2 bước 3 chú
thích ở 5–10 fps trên video 25–30 fps). CVAT bản hiện hành xuất số khung TUYỆT ĐỐI (đã cộng
`start_frame`, nhân `step`), nhưng không có gì bảo đảm mọi bản đều thế, mà lệch số khung thì
TrackEval vẫn chấm, chỉ là chấm hộp của khung này với hộp của khung khác. Nên công cụ đọc
`<meta>` và tự nhận ra hai kiểu đánh số (`--frame-numbering auto`), cùng một phép kiểm: mọi
số khung tuyệt đối phải rơi vào tập khung của task.

Chỉ stdlib (`xml.etree`) + `common/`, không cần cài gì.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from common.logging import get_logger
from common.motformat import MotRow, to_mot_frame, write_gt
from common.schema import Detection, FrameMessage, write_jsonl

log = get_logger("tools.cvat_to_mot")

DEFAULT_ATTRIBUTE = "person_id"
DEFAULT_LABEL = "person"
# Hộp `occluded="1"` vẫn là ground-truth hợp lệ, chỉ kém tin cậy hơn. TrackEval đọc cột này.
OCCLUDED_VISIBILITY = 0.5
FRAME_NUMBERING = ("auto", "abs", "rel")
# Cùng mốc với fixture tổng hợp/WildTrack: `validate` đòi ts_ms là epoch ms dương.
DEFAULT_BASE_TS_MS = 1_788_231_600_000


class CvatError(ValueError):
    """Chú thích không dùng được — sai ở đây thì cả bảng điểm sai theo."""


@dataclass(slots=True)
class CvatBox:
    frame: int  # đếm từ 0, như CVAT
    track_id: int
    person: str  # giá trị thuộc tính danh tính, "" nếu không có
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    occluded: bool = False

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.xtl, self.ytl, self.xbr - self.xtl, self.ybr - self.ytl)


def parse_cvat_video(
    xml_path: Path, *, label: str = DEFAULT_LABEL, attribute: str = DEFAULT_ATTRIBUTE
) -> list[CvatBox]:
    """Đọc XML "CVAT for video 1.1" → danh sách hộp, đã bỏ khung `outside="1"`.

    Thuộc tính danh tính đọc được ở CẢ hai chỗ CVAT có thể đặt: trên `<track>` (thuộc tính
    không đổi theo thời gian) hoặc trên từng `<box>` (thuộc tính thay đổi được). Ưu tiên
    box, vì nếu người gán nhãn sửa giữa chừng thì đó là ý định mới nhất của họ.
    """
    root = ET.parse(xml_path).getroot()
    boxes: list[CvatBox] = []

    for track in root.iter("track"):
        if label and track.get("label") != label:
            continue
        track_id = int(track.get("id", "-1"))
        track_person = ""
        for attr in track.findall("attribute"):
            if attr.get("name") == attribute:
                track_person = (attr.text or "").strip()

        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            person = track_person
            for attr in box.findall("attribute"):
                if attr.get("name") == attribute:
                    person = (attr.text or "").strip() or person
            try:
                boxes.append(
                    CvatBox(
                        # `attrib[...]`, KHÔNG phải `.get(..., "nan")`: hộp thiếu toạ độ phải
                        # nổ ở đây. Để nó thành NaN thì file gt.txt vẫn ghi ra được, TrackEval
                        # vẫn chấm, và bảng điểm sai mà không có gì báo.
                        frame=int(box.attrib["frame"]),
                        track_id=track_id,
                        person=person,
                        xtl=float(box.attrib["xtl"]),
                        ytl=float(box.attrib["ytl"]),
                        xbr=float(box.attrib["xbr"]),
                        ybr=float(box.attrib["ybr"]),
                        occluded=box.get("occluded") == "1",
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise CvatError(
                    f"{xml_path}: hộp thiếu hoặc sai toạ độ ở track {track_id} ({exc})"
                ) from exc

    if not boxes:
        raise CvatError(
            f"{xml_path}: không có hộp nào với label={label!r}. Kiểm tra --label và định "
            'dạng xuất (phải là "CVAT for video 1.1", không phải bản MOT).'
        )
    return boxes


@dataclass(frozen=True, slots=True)
class CvatMeta:
    """Phần `<meta>` của file XML: tập khung của task và kích thước ảnh gốc.

    Thiếu trường nào thì để mặc định (task trọn video, không nhảy khung) — file XML soạn
    tay trong test hay bản xuất cũ không có `<meta>` vẫn đọc được như trước.
    """

    start_frame: int = 0
    stop_frame: int | None = None
    step: int = 1
    width: int | None = None
    height: int | None = None

    def task_frames(self, last_frame: int) -> range:
        """Tập khung (số khung VIDEO) của task. `last_frame` dùng khi meta thiếu `stop_frame`."""
        stop = self.stop_frame if self.stop_frame is not None else last_frame
        return range(self.start_frame, stop + 1, self.step)


def _meta_int(root: ET.Element, tag: str) -> int | None:
    node = root.find(f"meta//{tag}")
    if node is None or not (node.text or "").strip():
        return None
    return int(node.text.strip())


def parse_cvat_meta(xml_path: Path) -> CvatMeta:
    """Đọc `start_frame`, `stop_frame`, `frame_filter` (`step=N`) và `original_size`.

    Tìm theo `meta//<tag>` để chịu được cả bản xuất theo task (`meta/task/...`) lẫn theo job
    (`meta/job/...`) của các bản CVAT khác nhau.
    """
    root = ET.parse(xml_path).getroot()
    step = 1
    node = root.find("meta//frame_filter")
    if node is not None and node.text:
        match = re.search(r"step\s*=\s*(\d+)", node.text)
        if match:
            step = int(match.group(1))
    if step < 1:
        raise CvatError(f"{xml_path}: frame_filter step={step} không hợp lệ")
    return CvatMeta(
        start_frame=_meta_int(root, "start_frame") or 0,
        stop_frame=_meta_int(root, "stop_frame"),
        step=step,
        width=_meta_int(root, "original_size/width"),
        height=_meta_int(root, "original_size/height"),
    )


def to_video_frames(
    boxes: list[CvatBox], meta: CvatMeta, *, numbering: str = "auto", source: str = ""
) -> tuple[list[CvatBox], str]:
    """Đưa số khung của CVAT về số khung của video gốc. Trả (hộp mới, kiểu đánh số đã dùng).

    - `abs`: số khung đã là của video — giữ nguyên.
    - `rel`: số khung đếm 0, 1, 2… trong task — đổi thành `start_frame + i * step`.
    - `auto`: task không nhảy khung và bắt đầu từ 0 thì hai kiểu trùng nhau. Còn lại, khung
      tuyệt đối luôn nằm trên lưới `start_frame + k * step`; một track nội suy liên tục có
      khung lệch lưới thì chắc chắn là đánh số tương đối.

    Sau khi đổi, mọi khung phải nằm trong tập khung của task — không thì nổ: lệch số khung
    là loại lỗi cho ra bảng điểm sai mà không có triệu chứng.
    """
    if numbering not in FRAME_NUMBERING:
        raise ValueError(f"numbering phải thuộc {FRAME_NUMBERING}, nhận {numbering!r}")
    if not boxes:
        return [], "abs"

    frames = {b.frame for b in boxes}
    if numbering == "auto":
        on_grid = all(
            f >= meta.start_frame and (f - meta.start_frame) % meta.step == 0 for f in frames
        )
        numbering = "abs" if on_grid else "rel"

    out = boxes
    if numbering == "rel":
        out = [replace(b, frame=meta.start_frame + b.frame * meta.step) for b in boxes]

    last = max(b.frame for b in out)
    valid = set(meta.task_frames(last))
    stray = sorted({b.frame for b in out} - valid)
    if stray:
        raise CvatError(
            f"{source or 'chú thích'}: {len(stray)} khung (ví dụ {stray[:3]}) không thuộc tập "
            f"khung của task (start={meta.start_frame}, stop={meta.stop_frame}, "
            f"step={meta.step}) sau khi đánh số kiểu {numbering!r}. Chỉ định "
            "--frame-numbering abs|rel cho đúng bản CVAT đã xuất."
        )
    return out, numbering


def build_gt_fixture(
    per_cam: dict[str, list[CvatBox]],
    frame_sets: dict[str, range],
    frame_sizes: dict[str, tuple[int, int]],
    *,
    fps: float,
    base_ts_ms: int = DEFAULT_BASE_TS_MS,
) -> list[FrameMessage]:
    """Hộp CVAT → fixture ground-truth: mỗi (camera, khung đã chú thích) một message.

    `local_track_id` = `track id` của CVAT (khoá của bảng `.gt.json`), `confidence` = 1,
    không có embedding. Khung chú thích không có ai vẫn có message rỗng: đó là thông tin
    "ở khung này mọi hộp của hệ thống đều là báo nhầm", khác với khung không chú thích.
    """
    if fps <= 0:
        raise ValueError(f"fps phải dương, nhận {fps}")
    messages: list[FrameMessage] = []
    for cam_id in sorted(per_cam):
        width, height = frame_sizes[cam_id]
        by_frame: dict[int, list[CvatBox]] = defaultdict(list)
        for box in per_cam[cam_id]:
            by_frame[box.frame].append(box)
        for frame in frame_sets[cam_id]:
            offset_ms = frame * 1000.0 / fps
            messages.append(
                FrameMessage(
                    cam_id=cam_id,
                    frame_id=int(frame),
                    ts_ms=int(base_ts_ms + round(offset_ms)),
                    frame_pts_ns=round(offset_ms * 1e6),
                    frame_width=int(width),
                    frame_height=int(height),
                    detections=[
                        Detection(local_track_id=b.track_id, bbox=b.bbox, confidence=1.0)
                        for b in sorted(by_frame.get(frame, []), key=lambda b: b.track_id)
                    ],
                )
            )
    return messages


def assign_global_ids(
    per_cam: dict[str, list[CvatBox]], *, require_global: bool
) -> dict[tuple[str, int], int]:
    """(cam_id, track_id) -> gt_global_id, suy từ thuộc tính danh tính dùng chung.

    Người có mặt ở nhiều camera phải mang cùng chuỗi `person_id`; công cụ đánh số các chuỗi
    đó theo thứ tự từ điển để `gt_global_id` ổn định giữa các lần chạy — chạy lại ra bảng
    khác là không tái lập được kết quả.
    """
    missing: list[str] = []
    per_track: dict[tuple[str, int], set[str]] = defaultdict(set)
    for cam_id, boxes in per_cam.items():
        for box in boxes:
            if box.person:
                per_track[(cam_id, box.track_id)].add(box.person)
            else:
                missing.append(f"{cam_id} track {box.track_id}")

    if missing and require_global:
        raise CvatError(
            f"{len(missing)} hộp thiếu thuộc tính danh tính (ví dụ: {missing[0]}). "
            "Không có nó thì mỗi camera là một tập danh tính riêng và mọi chỉ số xuyên "
            "camera vô nghĩa. Gán đủ, hoặc chạy lại với --no-global nếu chỉ cần đo đơn camera."
        )

    lan: list[str] = [
        f"{cam} track {tid}: {sorted(names)}"
        for (cam, tid), names in sorted(per_track.items())
        if len(names) > 1
    ]
    if lan:
        raise CvatError(
            "Một track mang nhiều danh tính khác nhau — chú thích mâu thuẫn, sửa trên CVAT "
            f"trước khi chấm: {'; '.join(lan[:5])}"
        )

    names = sorted({next(iter(v)) for v in per_track.values()})
    number = {name: i + 1 for i, name in enumerate(names)}
    return {key: number[next(iter(v))] for key, v in per_track.items()}


def write_ground_truth(
    out_dir: Path,
    per_cam: dict[str, list[CvatBox]],
    global_ids: dict[tuple[str, int], int],
    *,
    meta: dict,
) -> dict[str, int]:
    """Ghi `<cam>.gt.txt` cho từng camera + `global_ids.gt.json`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}

    for cam_id, boxes in sorted(per_cam.items()):
        rows = [
            MotRow(
                frame=to_mot_frame(b.frame),
                track_id=b.track_id,
                x=b.bbox[0],
                y=b.bbox[1],
                w=b.bbox[2],
                h=b.bbox[3],
                visibility=OCCLUDED_VISIBILITY if b.occluded else 1.0,
            )
            for b in boxes
        ]
        written[cam_id] = write_gt(out_dir / f"{cam_id}.gt.txt", rows)

    spans: dict[tuple[str, int], list[int]] = defaultdict(list)
    for cam_id, boxes in per_cam.items():
        for b in boxes:
            spans[(cam_id, b.track_id)].append(b.frame)

    payload = {
        "scenario": out_dir.name,
        "meta": meta,
        "tracklets": [
            {
                "cam_id": cam_id,
                "local_track_id": track_id,
                "gt_global_id": global_ids[(cam_id, track_id)],
                "start_frame": min(frames),
                "end_frame": max(frames),
                "n_frames": len(frames),
            }
            for (cam_id, track_id), frames in sorted(spans.items())
            if (cam_id, track_id) in global_ids
        ],
    }
    (out_dir / "global_ids.gt.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return written


def write_fixture_table(
    path: Path,
    per_cam: dict[str, list[CvatBox]],
    global_ids: dict[tuple[str, int], int],
    *,
    fps: float,
    base_ts_ms: int,
    meta: dict,
) -> int:
    """Bảng `.gt.json` đi kèm fixture ground-truth: như `global_ids.gt.json` + mốc `ts_ms`.

    `start_ms`/`end_ms` cùng hệ thời gian với fixture GT (không phải với lần chạy pipeline),
    để `tools/estimate_transit.py` đọc thời gian đi giữa hai camera thẳng từ bảng.
    """
    spans: dict[tuple[str, int], list[int]] = defaultdict(list)
    for cam_id, boxes in per_cam.items():
        for b in boxes:
            spans[(cam_id, b.track_id)].append(b.frame)

    def ts(frame: int) -> int:
        return int(base_ts_ms + round(frame * 1000.0 / fps))

    tracklets = [
        {
            "cam_id": cam_id,
            "local_track_id": track_id,
            "gt_global_id": global_ids[(cam_id, track_id)],
            "start_frame": min(frames),
            "end_frame": max(frames),
            "start_ms": ts(min(frames)),
            "end_ms": ts(max(frames)),
            "n_frames": len(frames),
        }
        for (cam_id, track_id), frames in sorted(spans.items())
        if (cam_id, track_id) in global_ids
    ]
    payload = {"scenario": path.name.removesuffix(".gt.json"), "meta": meta, "tracklets": tracklets}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return len(tracklets)


def identity_names(
    per_cam: dict[str, list[CvatBox]], global_ids: dict[tuple[str, int], int]
) -> dict[int, str]:
    """gt_global_id -> chuỗi `person_id` gốc, để đọc bảng điểm bằng tên người gán nhãn đặt."""
    out: dict[int, str] = {}
    for cam_id, boxes in per_cam.items():
        for b in boxes:
            gid = global_ids.get((cam_id, b.track_id))
            if gid is not None and b.person:
                out[gid] = b.person
    return dict(sorted(out.items()))


def _parse_size(value: str) -> tuple[int, int]:
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--frame-size cần dạng 1920x1080, nhận {value!r}"
        ) from exc


def _parse_annotation_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--annotation cần dạng cam_id=đường/dẫn.xml, nhận {value!r}"
        )
    cam_id, path = value.split("=", 1)
    if not cam_id.strip():
        raise argparse.ArgumentTypeError("cam_id rỗng")
    return cam_id.strip(), Path(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--annotation",
        action="append",
        required=True,
        type=_parse_annotation_arg,
        metavar="CAM_ID=FILE.xml",
        help="lặp lại cho từng camera, ví dụ --annotation cam01=cam01.xml",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--label", default=DEFAULT_LABEL, help="'' = nhận mọi label")
    p.add_argument("--attribute", default=DEFAULT_ATTRIBUTE, help="thuộc tính mang danh tính")
    p.add_argument(
        "--no-global",
        action="store_true",
        help="bỏ qua danh tính xuyên camera (chỉ đo đơn camera)",
    )
    p.add_argument(
        "--frame-numbering",
        choices=FRAME_NUMBERING,
        default="auto",
        help="số khung trong XML là của video (abs) hay của task (rel); auto = tự nhận",
    )
    p.add_argument(
        "--fixture-out",
        type=Path,
        default=None,
        help="ghi thêm fixture ground-truth .jsonl (+ .gt.json cạnh nó) cho export_trackeval "
        "--gt-fixture và tools.assign_gt",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="fps của VIDEO đã chú thích (bắt buộc với --fixture-out)",
    )
    p.add_argument(
        "--frame-size",
        type=_parse_size,
        default=None,
        help="kích thước ảnh, ví dụ 1920x1080 — chỉ cần khi XML thiếu <original_size>",
    )
    p.add_argument("--base-ts-ms", type=int, default=DEFAULT_BASE_TS_MS)
    args = p.parse_args(argv)
    if args.fixture_out is not None and not args.fps:
        p.error("--fixture-out cần --fps (fps của video sau khi đồng bộ, vd 25)")

    per_cam: dict[str, list[CvatBox]] = {}
    metas: dict[str, CvatMeta] = {}
    numberings: dict[str, str] = {}
    for cam_id, path in args.annotation:
        boxes = parse_cvat_video(path, label=args.label, attribute=args.attribute)
        metas[cam_id] = parse_cvat_meta(path)
        per_cam[cam_id], numberings[cam_id] = to_video_frames(
            boxes, metas[cam_id], numbering=args.frame_numbering, source=str(path)
        )
    global_ids = assign_global_ids(per_cam, require_global=not args.no_global)

    meta = {
        "source": "cvat",
        "cam_ids": sorted(per_cam),
        "label": args.label,
        "attribute": args.attribute,
        "n_identities": len(set(global_ids.values())),
        "n_tracks": len(global_ids),
        "identities": {str(k): v for k, v in identity_names(per_cam, global_ids).items()},
        "frames": {
            cam_id: {
                "start": m.start_frame,
                "stop": m.stop_frame,
                "step": m.step,
                "numbering": numberings[cam_id],
            }
            for cam_id, m in sorted(metas.items())
        },
        "notes": (
            "gt_global_id đánh số theo thứ tự từ điển của thuộc tính danh tính, ổn định "
            "giữa các lần chạy. Khung outside=1 của CVAT đã bị loại. Số khung là số khung "
            "của VIDEO (đã quy đổi từ task CVAT)."
        ),
    }
    written = write_ground_truth(args.out_dir, per_cam, global_ids, meta=meta)

    if args.fixture_out is not None:
        sizes: dict[str, tuple[int, int]] = {}
        frame_sets: dict[str, range] = {}
        for cam_id, m in metas.items():
            if m.width and m.height:
                sizes[cam_id] = (m.width, m.height)
            elif args.frame_size:
                sizes[cam_id] = args.frame_size
            else:
                p.error(f"{cam_id}: XML không có <original_size>, cần --frame-size WxH")
            frame_sets[cam_id] = m.task_frames(max(b.frame for b in per_cam[cam_id]))
        messages = build_gt_fixture(
            per_cam, frame_sets, sizes, fps=args.fps, base_ts_ms=args.base_ts_ms
        )
        n_msg = write_jsonl(args.fixture_out, messages)
        table = Path(str(args.fixture_out).replace(".jsonl", ".gt.json"))
        n_tracks = write_fixture_table(
            table,
            per_cam,
            global_ids,
            fps=args.fps,
            base_ts_ms=args.base_ts_ms,
            meta={**meta, "fps": args.fps, "base_ts_ms": args.base_ts_ms},
        )
        log.info(
            "fixture GT %s: %d message, bảng %s: %d track", args.fixture_out, n_msg, table, n_tracks
        )

    log.info(
        "%s: %d camera, %d track, %d danh tính; số dòng mỗi camera: %s",
        args.out_dir,
        len(per_cam),
        meta["n_tracks"],
        meta["n_identities"],
        ", ".join(f"{c}={n}" for c, n in sorted(written.items())),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
