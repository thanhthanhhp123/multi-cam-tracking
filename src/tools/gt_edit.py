"""Mô hình sửa ground-truth không cần CVAT: đọc/ghi "CVAT for video 1.1" + các thao tác sửa.

Dùng bởi `tools.gt_review` (giao diện web nhẹ). Tách riêng để test được mà không cần trình
duyệt: mọi thao tác là hàm trên `CamDoc`, kiểm tra lỗi bằng `validate`.

**Mô hình.** Một `CamDoc` là một camera của một buổi. Mỗi `Track` giữ hộp ở MỌI khung task
mà người đó hiện (đúng như bản CVAT tự xuất: có cả hộp nội suy, cờ `keyframe` đánh dấu khung
khoá). Khung task `k` ứng với khung video `start_frame + k * step`. Khung không có trong
`boxes` là `outside`.

**Sửa một hộp** (`set_box`) biến khung đó thành khung khoá rồi nội suy lại tuyến tính tới khung
khoá gần nhất mỗi bên, CHỈ trong đoạn hiện liên tục — không nội suy xuyên qua chỗ người đã ra
khỏi khung. `link=True` thì nối thêm các khung còn thiếu giữa lần thấy trước và khung vừa sửa
(kéo dài track sau một đoạn mất dấu).

**Ghi ra** đúng định dạng `tools.cvat_to_mot` đọc: giữ nguyên `<meta>` của file gốc, số khung
tuyệt đối, `person_id` trên từng hộp, và một hộp `outside="1"` sau mỗi đoạn hiện (để nhập
ngược lại CVAT vẫn đúng).
"""

from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

ATTRIBUTE = "person_id"
LABEL = "person"


class EditError(ValueError):
    """Thao tác không hợp lệ (track không tồn tại, khung ngoài phạm vi...)."""


@dataclass(slots=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    keyframe: bool = False
    occluded: bool = False

    def coords(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(slots=True)
class Track:
    tid: int
    person: str = ""
    source: str = "manual"
    boxes: dict[int, Box] = field(default_factory=dict)  # khung task -> hộp (chỉ khung hiện)

    def frames(self) -> list[int]:
        return sorted(self.boxes)

    def runs(self) -> list[tuple[int, int]]:
        """Các đoạn hiện liên tục [đầu, cuối]."""
        out: list[tuple[int, int]] = []
        for k in self.frames():
            if out and k == out[-1][1] + 1:
                out[-1] = (out[-1][0], k)
            else:
                out.append((k, k))
        return out


@dataclass(slots=True)
class CamDoc:
    session: str
    cam: str
    meta: ET.Element
    start: int
    step: int
    size: int
    tracks: dict[int, Track] = field(default_factory=dict)

    def track(self, tid: int) -> Track:
        if tid not in self.tracks:
            raise EditError(f"{self.session}/{self.cam}: không có track #{tid}")
        return self.tracks[tid]

    def check_frame(self, k: int) -> None:
        if not 0 <= k < self.size:
            raise EditError(f"khung {k} ngoài phạm vi 0..{self.size - 1}")

    def next_tid(self) -> int:
        return max(self.tracks, default=-1) + 1


# ------------------------------------------------------------------------------------ I/O


def _meta_int(meta: ET.Element, tag: str, default: int) -> int:
    node = meta.find(f".//{tag}")
    return int(node.text) if node is not None and (node.text or "").strip() else default


def load(path: Path, *, session: str = "", cam: str = "") -> CamDoc:
    root = ET.parse(path).getroot()
    meta = root.find("meta")
    if meta is None:
        raise EditError(f"{path}: thiếu <meta> — phải là bản xuất 'CVAT for video 1.1'")
    step = 1
    ff = meta.find(".//frame_filter")
    if ff is not None and ff.text:
        m = re.search(r"step\s*=\s*(\d+)", ff.text)
        step = int(m.group(1)) if m else 1
    start = _meta_int(meta, "start_frame", 0)
    stop = _meta_int(meta, "stop_frame", -1)
    size = _meta_int(meta, "size", (stop - start) // step + 1 if stop >= 0 else 0)
    doc = CamDoc(session or path.parent.name, cam or path.stem, meta, start, step, size)
    for tr in root.iter("track"):
        if tr.get("label") != LABEL:
            continue
        t = Track(int(tr.get("id")), source=tr.get("source", "manual"))
        for b in tr.iter("box"):
            person = next(
                ((a.text or "").strip() for a in b.iter("attribute") if a.get("name") == ATTRIBUTE),
                "",
            )
            t.person = person or t.person
            if b.get("outside") == "1":
                continue
            f = int(b.get("frame"))
            if (f - start) % step:
                raise EditError(f"{path}: khung {f} lệch lưới step={step}")
            t.boxes[(f - start) // step] = Box(
                *(float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")),
                keyframe=b.get("keyframe") == "1",
                occluded=b.get("occluded") == "1",
            )
        if t.boxes:
            doc.tracks[t.tid] = t
    return doc


def save(doc: CamDoc, path: Path) -> None:
    """Ghi atomically (file tạm rồi thay thế) — tắt ngang lúc ghi không làm hỏng file cũ."""
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    root.append(copy.deepcopy(doc.meta))
    for tid in sorted(doc.tracks):
        t = doc.tracks[tid]
        node = ET.SubElement(root, "track", id=str(tid), label=LABEL, source=t.source)
        runs = t.runs()
        for a, b in runs:
            for k in range(a, b + 1):
                _box_el(node, doc, k, t.boxes[k], t.person, outside=False, keyframe=k in (a, b))
            if b + 1 < doc.size:
                _box_el(node, doc, b + 1, t.boxes[b], t.person, outside=True, keyframe=True)
    ET.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".xml.tmp")
    ET.ElementTree(root).write(tmp, encoding="utf-8", xml_declaration=True)
    tmp.replace(path)


def _box_el(
    node: ET.Element, doc: CamDoc, k: int, box: Box, person: str, *, outside: bool, keyframe: bool
) -> None:
    el = ET.SubElement(
        node,
        "box",
        frame=str(doc.start + k * doc.step),
        keyframe="1" if (keyframe or box.keyframe) else "0",
        outside="1" if outside else "0",
        occluded="1" if box.occluded and not outside else "0",
        xtl=f"{box.x1:.2f}",
        ytl=f"{box.y1:.2f}",
        xbr=f"{box.x2:.2f}",
        ybr=f"{box.y2:.2f}",
        z_order="0",
    )
    ET.SubElement(el, "attribute", name=ATTRIBUTE).text = person


# ------------------------------------------------------------------------------ thao tác


def set_person(doc: CamDoc, tid: int, person: str) -> None:
    doc.track(tid).person = person.strip()


def delete_track(doc: CamDoc, tid: int) -> None:
    doc.track(tid)
    del doc.tracks[tid]


def split(doc: CamDoc, tid: int, k: int, person: str | None = None) -> int:
    """Từ khung `k` trở đi thành track mới (mang `person`, mặc định giữ mã cũ). Trả id mới."""
    t = doc.track(tid)
    tail = {f: b for f, b in t.boxes.items() if f >= k}
    if not tail or len(tail) == len(t.boxes):
        raise EditError(f"track #{tid} không có khung nào ở cả hai phía của khung {k}")
    for f in tail:
        del t.boxes[f]
    t.boxes[max(t.boxes)].keyframe = True
    new = Track(doc.next_tid(), t.person if person is None else person.strip(), t.source, tail)
    new.boxes[min(tail)].keyframe = True
    doc.tracks[new.tid] = new
    return new.tid


def hide(doc: CamDoc, tid: int, a: int, b: int) -> None:
    """Người không còn hiện ở các khung [a, b] (`outside`). Track rỗng thì bị xoá."""
    t = doc.track(tid)
    for f in [f for f in t.boxes if a <= f <= b]:
        del t.boxes[f]
    if not t.boxes:
        del doc.tracks[tid]
        return
    for f in (a - 1, b + 1):
        if f in t.boxes:
            t.boxes[f].keyframe = True


def set_occluded(doc: CamDoc, tid: int, a: int, b: int, value: bool) -> int:
    n = 0
    for f, box in doc.track(tid).boxes.items():
        if a <= f <= b:
            box.occluded = value
            n += 1
    return n


def _interpolate(t: Track, lo: int, hi: int) -> None:
    """Nội suy tuyến tính các khung (lo, hi) giữa hai khung khoá lo và hi (đã có hộp)."""
    a, b = t.boxes[lo], t.boxes[hi]
    for f in range(lo + 1, hi):
        r = (f - lo) / (hi - lo)
        old = t.boxes.get(f)
        t.boxes[f] = Box(
            *(p + (q - p) * r for p, q in zip(a.coords(), b.coords(), strict=True)),
            keyframe=False,
            occluded=old.occluded if old else False,
        )


def set_box(
    doc: CamDoc, tid: int, k: int, coords: tuple[float, float, float, float], *, link: bool = False
) -> None:
    doc.check_frame(k)
    t = doc.track(tid)
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        raise EditError("hộp phải có xbr > xtl và ybr > ytl")
    old = t.boxes.get(k)
    if old is None and link:
        before = [f for f in t.boxes if f < k]
        if before:  # nối liền đoạn mất dấu từ lần thấy trước tới khung k
            prev = max(before)
            t.boxes[k] = Box(x1, y1, x2, y2, keyframe=True)
            t.boxes[prev].keyframe = True
            _interpolate(t, prev, k)
            return
    t.boxes[k] = Box(x1, y1, x2, y2, keyframe=True, occluded=old.occluded if old else False)
    run = next(r for r in t.runs() if r[0] <= k <= r[1])
    keys = [f for f in range(run[0], run[1] + 1) if t.boxes[f].keyframe]
    prev = max((f for f in keys if f < k), default=None)
    nxt = min((f for f in keys if f > k), default=None)
    if prev is not None:
        _interpolate(t, prev, k)
    if nxt is not None:
        _interpolate(t, k, nxt)


def new_track(
    doc: CamDoc, k: int, coords: tuple[float, float, float, float], person: str = ""
) -> int:
    doc.check_frame(k)
    t = Track(doc.next_tid(), person.strip(), "manual")
    doc.tracks[t.tid] = t
    set_box(doc, t.tid, k, coords)
    return t.tid


def merge(doc: CamDoc, keep: int, other: int) -> None:
    """Gộp track `other` vào `keep` (cùng một người bị tracker cắt đôi). Không cho chồng khung."""
    a, b = doc.track(keep), doc.track(other)
    both = set(a.boxes) & set(b.boxes)
    if both:
        raise EditError(f"#{keep} và #{other} cùng hiện ở {len(both)} khung (vd {min(both)})")
    a.boxes.update(b.boxes)
    a.person = a.person or b.person
    del doc.tracks[other]


# ------------------------------------------------------------------------------ kiểm tra


def validate(doc: CamDoc) -> list[str]:
    """Các lỗi làm hỏng bảng điểm: track trống mã, hai track cùng mã hiện cùng một khung."""
    problems: list[str] = []
    seen: dict[str, dict[int, int]] = {}
    for tid in sorted(doc.tracks):
        t = doc.tracks[tid]
        if not t.person:
            problems.append(f"{doc.cam} #{tid}: chưa có {ATTRIBUTE} (khung {min(t.boxes)})")
            continue
        owner = seen.setdefault(t.person, {})
        clash = sorted(f for f in t.boxes if f in owner)
        if clash:
            problems.append(
                f"{doc.cam} {t.person}: #{owner[clash[0]]} và #{tid} cùng hiện ở {len(clash)} khung"
                f" ({clash[0]}–{clash[-1]})"
            )
        for f in t.boxes:
            owner.setdefault(f, tid)
    return problems
