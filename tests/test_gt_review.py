"""Test `tools/gt_edit.py` + `tools/gt_review.py` — công cụ duyệt GT thay CVAT.

Điều phải giữ bằng mọi giá: file ghi ra vẫn đi được qua `tools.cvat_to_mot` (đúng khung,
đúng danh tính), và không thao tác nào để lại hai track cùng mã trong một khung mà
`validate` không báo.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tools import gt_edit
from tools.cvat_to_mot import parse_cvat_meta, parse_cvat_video, to_video_frames
from tools.gt_edit import Box, EditError
from tools.gt_review import Workspace, create_app

STEP = 5


def _xml(tracks: dict[int, tuple[str, list[int]]], size: int = 20) -> str:
    """Bản xuất kiểu CVAT: mọi khung task có hộp, outside sau đoạn cuối."""
    out = [
        "<?xml version='1.0' encoding='utf-8'?>",
        "<annotations><version>1.1</version><meta><task>",
        f"<size>{size}</size><start_frame>0</start_frame>",
        f"<stop_frame>{(size - 1) * STEP}</stop_frame>",
        f"<frame_filter>step={STEP}</frame_filter>",
        "<original_size><width>1920</width><height>1080</height></original_size>",
        "</task></meta>",
    ]
    for tid, (person, frames) in tracks.items():
        out.append(f'<track id="{tid}" label="person" source="auto">')
        for k in frames:
            out.append(
                f'<box frame="{k * STEP}" keyframe="{int(k in (frames[0], frames[-1]))}" '
                f'outside="0" '
                f'occluded="0" xtl="{10 * k}" ytl="100" xbr="{10 * k + 50}" ybr="300" z_order="0">'
                f'<attribute name="person_id">{person}</attribute></box>'
            )
        out.append(
            f'<box frame="{(frames[-1] + 1) * STEP}" keyframe="1" outside="1" occluded="0" '
            f'xtl="0" ytl="0" xbr="1" ybr="1" z_order="0">'
            f'<attribute name="person_id">{person}</attribute></box>'
        )
        out.append("</track>")
    out.append("</annotations>")
    return "\n".join(out)


@pytest.fixture
def doc(tmp_path: Path) -> gt_edit.CamDoc:
    p = tmp_path / "s" / "cam01.xml"
    p.parent.mkdir()
    p.write_text(
        _xml({0: ("P01", list(range(0, 10))), 1: ("", list(range(5, 15)))}), encoding="utf-8"
    )
    return gt_edit.load(p)


def test_load_reads_task_frames_and_person(doc: gt_edit.CamDoc) -> None:
    assert (doc.step, doc.size, doc.start) == (STEP, 20, 0)
    assert doc.tracks[0].person == "P01" and doc.tracks[0].frames() == list(range(10))
    assert doc.tracks[0].boxes[3].coords() == (30.0, 100.0, 80.0, 300.0)
    assert doc.tracks[1].person == ""


def test_save_round_trips_through_cvat_to_mot(doc: gt_edit.CamDoc, tmp_path: Path) -> None:
    gt_edit.set_person(doc, 1, "X01")
    out = tmp_path / "out" / "cam01.xml"
    gt_edit.save(doc, out)
    boxes = parse_cvat_video(out)
    meta = parse_cvat_meta(out)
    boxes, numbering = to_video_frames(boxes, meta)
    assert numbering == "abs" and meta.step == STEP
    assert {b.person for b in boxes} == {"P01", "X01"}
    assert sorted({b.frame for b in boxes if b.person == "X01"}) == [k * STEP for k in range(5, 15)]
    again = gt_edit.load(out)
    assert again.tracks[1].frames() == list(range(5, 15))


def test_validate_reports_empty_and_same_id_overlap(doc: gt_edit.CamDoc) -> None:
    assert any("#1: chưa có" in p for p in gt_edit.validate(doc))
    gt_edit.set_person(doc, 1, "P01")
    assert any("P01: #0 và #1 cùng hiện ở 5 khung (5–9)" in p for p in gt_edit.validate(doc))
    gt_edit.hide(doc, 0, 5, 9)
    assert gt_edit.validate(doc) == []


def test_split_moves_tail_to_new_track(doc: gt_edit.CamDoc) -> None:
    new = gt_edit.split(doc, 1, 10, "P02")
    assert doc.tracks[1].frames() == list(range(5, 10))
    assert doc.tracks[new].frames() == list(range(10, 15)) and doc.tracks[new].person == "P02"
    assert doc.tracks[1].boxes[9].keyframe and doc.tracks[new].boxes[10].keyframe
    with pytest.raises(EditError):
        gt_edit.split(doc, 1, 5)  # nothing would stay in the head


def test_set_box_reinterpolates_between_keyframes(doc: gt_edit.CamDoc) -> None:
    t = doc.tracks[0]  # keyframes at 0 and 9, x1 = 10 * k
    gt_edit.set_box(doc, 0, 4, (100.0, 100.0, 150.0, 300.0))
    assert t.boxes[4].keyframe
    assert t.boxes[2].x1 == pytest.approx(50.0)  # 0 -> 100 over 4 frames
    assert t.boxes[6].x1 == pytest.approx(100 + (90 - 100) * 2 / 5)  # 100 -> 90 over 5 frames


def test_set_box_with_link_fills_gap(doc: gt_edit.CamDoc) -> None:
    gt_edit.set_box(doc, 0, 14, (140.0, 100.0, 190.0, 300.0), link=True)
    assert doc.tracks[0].frames() == list(range(15))
    assert doc.tracks[0].boxes[12].x1 == pytest.approx(90 + (140 - 90) * 3 / 5)


def test_new_track_and_merge(doc: gt_edit.CamDoc) -> None:
    tid = gt_edit.new_track(doc, 15, (0.0, 0.0, 10.0, 10.0), "P01")
    gt_edit.merge(doc, 0, tid)
    assert 15 in doc.tracks[0].boxes and tid not in doc.tracks
    with pytest.raises(EditError, match="cùng hiện"):
        gt_edit.merge(doc, 0, 1)


def test_occluded_survives_save(doc: gt_edit.CamDoc, tmp_path: Path) -> None:
    gt_edit.set_occluded(doc, 0, 2, 3, True)
    out = tmp_path / "o.xml"
    gt_edit.save(doc, out)
    occ = {b.frame for b in parse_cvat_video(out) if b.occluded}
    assert occ == {10, 15}


def test_bad_box_rejected(doc: gt_edit.CamDoc) -> None:
    with pytest.raises(EditError):
        gt_edit.set_box(doc, 0, 3, (50.0, 10.0, 40.0, 20.0))
    with pytest.raises(EditError):
        gt_edit.set_box(doc, 0, 99, (0.0, 0.0, 1.0, 1.0))
    assert isinstance(doc.tracks[0].boxes[3], Box)


def test_app_edits_save_all_cams_and_undo(tmp_path: Path) -> None:
    export, out, frames = tmp_path / "export", tmp_path / "cvat", tmp_path / "frames"
    for cam in ("cam01", "cam02", "cam03"):
        (export / "s1").mkdir(parents=True, exist_ok=True)
        (export / "s1" / f"{cam}.xml").write_text(_xml({0: ("", [0, 1, 2])}), encoding="utf-8")
    (frames / "s1" / "cam01").mkdir(parents=True)
    (frames / "s1" / "cam01" / "00002.jpg").write_bytes(b"\xff\xd8jpg")
    client = TestClient(create_app(Workspace(export, out, frames, None)))

    assert client.get("/api/sessions").json()[0] == {
        "name": "s1",
        "edited": False,
        "reviewed": False,
    }
    st = client.get("/api/session/s1").json()
    assert len(st["problems"]) == 3
    r = client.post(
        "/api/session/s1/op", json={"cam": "cam02", "op": "person", "tid": 0, "person": "P07"}
    )
    assert r.status_code == 200 and len(r.json()["problems"]) == 2
    assert {p.name for p in (out / "s1").iterdir()} == {
        "cam01.xml",
        "cam02.xml",
        "cam03.xml",
        "edits.jsonl",
    }
    assert gt_edit.load(out / "s1" / "cam02.xml").tracks[0].person == "P07"
    assert (
        client.post("/api/session/s1/reviewed", json={"value": True}).status_code == 400
    )  # still errors
    assert (
        client.post(
            "/api/session/s1/op", json={"cam": "cam01", "op": "delete", "tid": 9}
        ).status_code
        == 400
    )
    client.post("/api/session/s1/undo")
    assert gt_edit.load(out / "s1" / "cam02.xml").tracks[0].person == ""
    assert client.get("/frames/s1/cam01/2").content == b"\xff\xd8jpg"
    assert client.get("/frames/s1/cam09/2").status_code == 404
