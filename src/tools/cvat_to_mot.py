"""Chú thích CVAT (một task mỗi camera) → ground-truth MOT + bảng Global ID.

    PYTHONPATH=src python -m tools.cvat_to_mot \\
        --annotation cam01=data/cvat/cam01.xml --annotation cam02=data/cvat/cam02.xml \\
        --out-dir eval/gt/lab_2cam

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

Chỉ stdlib (`xml.etree`), không cần cài gì.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from common.logging import get_logger
from common.motformat import MotRow, to_mot_frame, write_gt

log = get_logger("tools.cvat_to_mot")

DEFAULT_ATTRIBUTE = "person_id"
DEFAULT_LABEL = "person"
# Hộp `occluded="1"` vẫn là ground-truth hợp lệ, chỉ kém tin cậy hơn. TrackEval đọc cột này.
OCCLUDED_VISIBILITY = 0.5


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
    args = p.parse_args(argv)

    per_cam = {
        cam_id: parse_cvat_video(path, label=args.label, attribute=args.attribute)
        for cam_id, path in args.annotation
    }
    global_ids = assign_global_ids(per_cam, require_global=not args.no_global)

    meta = {
        "source": "cvat",
        "cam_ids": sorted(per_cam),
        "label": args.label,
        "attribute": args.attribute,
        "n_identities": len(set(global_ids.values())),
        "n_tracks": len(global_ids),
        "notes": (
            "gt_global_id đánh số theo thứ tự từ điển của thuộc tính danh tính, ổn định "
            "giữa các lần chạy. Khung outside=1 của CVAT đã bị loại."
        ),
    }
    written = write_ground_truth(args.out_dir, per_cam, global_ids, meta=meta)

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
