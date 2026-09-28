"""eval/roi_filter_fixture.py: keep only detections whose foot point is inside the ground ROI."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from eval.roi_filter_fixture import GroundRoi, filter_message, filter_messages

from common.schema import Detection, FrameMessage, read_jsonl, write_jsonl
from mct.homography import CameraHomography

# Identity homography: the foot pixel (x + w/2, y + h) IS the ground point in metres.
IDENTITY = CameraHomography(cam_id="cam01", matrix=np.eye(3))


def _det(track_id: int, foot_x: float, foot_y: float) -> Detection:
    return Detection(
        local_track_id=track_id, bbox=(foot_x - 0.5, foot_y - 2.0, 1.0, 2.0), confidence=0.9
    )


def _msg(dets: list[Detection], frame_id: int = 0, cam_id: str = "cam01") -> FrameMessage:
    return FrameMessage(
        cam_id=cam_id,
        frame_id=frame_id,
        ts_ms=1_000 + frame_id,
        frame_pts_ns=frame_id,
        frame_width=1920,
        frame_height=1080,
        detections=dets,
    )


def test_roi_defaults_to_the_wildtrack_grid() -> None:
    roi = GroundRoi()
    assert roi.contains((-3.0, -9.0))
    assert roi.contains((8.99, 26.99))
    assert not roi.contains((9.0, 0.0))  # half-open, like diagnose_fp_region.in_annotated_area
    assert not roi.contains((0.0, -9.5))


def test_margin_grows_every_side() -> None:
    roi = GroundRoi(margin_m=1.0)
    assert roi.contains((-3.9, 0.0))
    assert roi.contains((9.9, 27.9))
    assert not roi.contains((-4.1, 0.0))


def test_filter_keeps_inside_drops_outside_and_keeps_the_frame() -> None:
    inside, outside = _det(1, 0.0, 0.0), _det(2, 20.0, 0.0)
    out, n_dropped = filter_message(_msg([inside, outside]), IDENTITY, GroundRoi())
    assert n_dropped == 1
    assert [d.local_track_id for d in out.detections] == [1]

    empty, n_dropped = filter_message(_msg([outside]), IDENTITY, GroundRoi())
    assert n_dropped == 1 and empty.detections == []
    assert (empty.cam_id, empty.frame_id) == ("cam01", 0)


def test_unprojectable_foot_is_dropped() -> None:
    # Invertible, but w = 1 - y: a foot at image row y = 1 sits on the horizon, so
    # apply_homography reports it as degenerate (None) instead of a ground point.
    horizon = np.array([[1, 0, 0], [0, 1, 0], [0, -1, 1.0]])
    degenerate = CameraHomography(cam_id="cam01", matrix=horizon)
    out, n_dropped = filter_message(_msg([_det(1, 0.0, 1.0)]), degenerate, GroundRoi())
    assert n_dropped == 1 and out.detections == []


def test_round_trip_through_jsonl_keeps_every_message(tmp_path: Path) -> None:
    (tmp_path / "cam01.yaml").write_text(yaml.safe_dump({"matrix": np.eye(3).tolist()}))
    src = tmp_path / "in.jsonl"
    write_jsonl(src, [_msg([_det(1, 0.0, 0.0)], 0), _msg([_det(2, 50.0, 0.0)], 1)])

    dropped: Counter[str] = Counter()
    total: Counter[str] = Counter()
    out = tmp_path / "out.jsonl"
    write_jsonl(out, filter_messages(read_jsonl(src), tmp_path, GroundRoi(), dropped, total))

    msgs = list(read_jsonl(out))
    assert [m.frame_id for m in msgs] == [0, 1]
    assert [len(m.detections) for m in msgs] == [1, 0]
    assert (dropped["cam01"], total["cam01"]) == (1, 2)
