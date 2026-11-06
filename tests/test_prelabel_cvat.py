"""Test `tools/prelabel_cvat.py` — phần không cần GPU: rút khung khoá, cắt đoạn, ghi XML.

Chỗ đáng canh nhất là XML ra phải đi được qua đúng cửa mà chú thích thật đi qua
(`tools.cvat_to_mot`): số khung là số khung VIDEO trên lưới `step`, khung `outside` bị bỏ, và
`person_id` trống thì NỔ — để không track máy gán nào vào ground-truth mà chưa qua người.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.cvat_to_mot import (
    CvatError,
    assign_global_ids,
    parse_cvat_meta,
    parse_cvat_video,
    to_video_frames,
)
from tools.prelabel_cvat import (
    Track,
    clean_tracks,
    giou,
    iou,
    keyframe_indices,
    to_cvat_xml,
    track_keys,
)


def _walk(frames: list[int], x0: float, dx: float) -> list[tuple[float, float, float, float]]:
    """Hộp 100×200 trượt đều theo x: nội suy tuyến tính khớp tuyệt đối."""
    return [(x0 + dx * i, 100.0, x0 + dx * i + 100.0, 300.0) for i in range(len(frames))]


def test_iou_basic() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_giou_ranks_disjoint_boxes_by_distance() -> None:
    assert giou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    near = giou((0, 0, 10, 10), (20, 0, 30, 10))
    far = giou((0, 0, 10, 10), (200, 0, 210, 10))
    assert far < near < 0


def test_linear_motion_needs_only_endpoints() -> None:
    frames = list(range(0, 100, 5))
    assert keyframe_indices(frames, _walk(frames, 0, 10), 0.9) == [0, len(frames) - 1]


def test_turn_adds_keyframe_at_the_corner() -> None:
    frames = list(range(0, 50, 5))
    there = _walk(frames[:5], 0, 40)  # x = 0 → 160
    back = _walk(frames[5:], 120, -40)  # x = 120 → -40
    assert keyframe_indices(frames, there + back, 0.9) == [0, 4, 9]


def test_gap_splits_track_and_inserts_outside() -> None:
    frames = [0, 5, 10, 30, 35, 40]
    track = Track(0, frames, _walk(frames, 0, 0))
    keys = track_keys(track, step=5, stop_frame=100, min_iou=0.9)
    assert [(k.frame, k.outside) for k in keys] == [
        (0, False),
        (10, False),
        (15, True),
        (30, False),
        (40, False),
        (45, True),
    ]


def test_no_outside_after_last_task_frame() -> None:
    frames = [90, 95, 100]
    keys = track_keys(Track(0, frames, _walk(frames, 0, 0)), step=5, stop_frame=100, min_iou=0.9)
    assert not any(k.outside for k in keys)


def test_clean_drops_short_and_far_and_renumbers() -> None:
    near = Track(7, [10, 15, 20], [(0, 0, 10, 400)] * 3)
    far = Track(3, [0, 5, 10], [(0, 0, 10, 200)] * 3)  # chân y = 200 < 250
    short = Track(9, [0], [(0, 0, 10, 400)])
    out = clean_tracks([near, far, short], min_frames=2, min_foot_y=250)
    assert [(t.track_id, t.frames) for t in out] == [(0, [10, 15, 20])]


def test_xml_round_trips_through_cvat_to_mot(tmp_path: Path) -> None:
    a = list(range(0, 50, 5))
    b = list(range(20, 60, 5))
    tracks = [Track(0, a, _walk(a, 0, 10)), Track(1, b, _walk(b, 500, -5))]
    path = tmp_path / "cam01.xml"
    to_cvat_xml(tracks, step=5, stop_frame=95, size=(1920, 1080), min_iou=0.9).write(
        path, encoding="utf-8", xml_declaration=True
    )

    meta = parse_cvat_meta(path)
    assert (meta.start_frame, meta.stop_frame, meta.step) == (0, 95, 5)
    assert (meta.width, meta.height) == (1920, 1080)

    boxes = parse_cvat_video(path)
    # Chỉ khung khoá nằm trong XML (CVAT tự nội suy phần giữa); khung outside đã bị bỏ.
    assert sorted({(x.track_id, x.frame) for x in boxes}) == [(0, 0), (0, 45), (1, 20), (1, 55)]
    _, numbering = to_video_frames(boxes, meta, source=str(path))
    assert numbering == "abs"

    # person_id trống → cvat_to_mot không cho vào ground-truth.
    with pytest.raises(CvatError, match="thiếu thuộc tính danh tính"):
        assign_global_ids({"cam01": boxes}, require_global=True)
