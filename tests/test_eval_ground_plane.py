"""eval/eval_ground_plane.py: point fusion, the annotated-area filter and TrackEval input building.

The metric classes themselves come from TrackEval (numpy 1.23.5, not installed in the light test
venv); their wiring is covered by ``python -m eval.eval_ground_plane --self-check``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from eval.eval_ground_plane import (
    aggregate,
    build_sequence,
    frame_ranges,
    fuse_frames,
    fuse_median,
    ground_nms,
    in_annotated_area,
    load_gt_points,
)


def test_area_is_half_open_on_the_grid() -> None:
    assert in_annotated_area((-3.0, -9.0))
    assert not in_annotated_area((9.0, 0.0))
    assert not in_annotated_area((0.0, 27.0))
    assert not in_annotated_area((-3.01, 0.0))


def test_fuse_median_is_coordinatewise() -> None:
    assert fuse_median([(0.0, 0.0), (1.0, 10.0), (100.0, 4.0)]) == (1.0, 4.0)


def test_fuse_frames_drops_points_outside_the_area_only_when_asked() -> None:
    raw = [{1: [(0.0, 0.0), (0.2, 0.0)], 2: [(50.0, 0.0)]}]
    assert fuse_frames(raw, area_filter=False) == [{1: (0.1, 0.0), 2: (50.0, 0.0)}]
    assert fuse_frames(raw, area_filter=True) == [{1: (0.1, 0.0)}]


def test_load_gt_points_orders_frames_by_file_name(tmp_path: Path) -> None:
    def person(pid: int, position_id: int) -> dict:
        return {"personID": pid, "positionID": position_id, "views": []}

    (tmp_path / "00000005.json").write_text(json.dumps([person(7, 480)]), encoding="utf-8")
    (tmp_path / "00000000.json").write_text(
        json.dumps([person(3, 0), person(4, 1)]), encoding="utf-8"
    )
    frames = load_gt_points(tmp_path)
    assert [sorted(f) for f in frames] == [[3, 4], [7]]
    assert frames[0][3] == pytest.approx((-3.0, -9.0))
    assert frames[0][4] == pytest.approx((-3.0 + 0.025, -9.0))
    assert frames[1][7] == pytest.approx((-3.0, -9.0 + 0.025))


def test_build_sequence_similarity_and_counts() -> None:
    gt = [{10: (0.0, 0.0), 11: (5.0, 0.0)}, {}]
    pred = [{"a": (1.0, 0.0)}, {"a": (0.0, 0.0)}]
    data = build_sequence(gt, pred, range(2), zero_distance=2.0)
    assert data["num_timesteps"] == 2
    assert (data["num_gt_dets"], data["num_tracker_dets"]) == (2, 2)
    assert (data["num_gt_ids"], data["num_tracker_ids"]) == (2, 1)
    np.testing.assert_allclose(data["similarity_scores"][0], [[0.5], [0.0]])  # 1 m -> 0.5, 4 m -> 0
    assert data["similarity_scores"][1].shape == (0, 1)
    assert data["gt_ids"][0].tolist() == [0, 1]
    assert data["tracker_ids"][1].tolist() == [0]  # the same Global ID keeps its remapped id


def test_build_sequence_restricts_to_the_requested_frames() -> None:
    gt = [{1: (0.0, 0.0)}, {2: (0.0, 0.0)}, {3: (0.0, 0.0)}]
    data = build_sequence(gt, [{}, {}, {}], range(1, 3), zero_distance=2.0)
    assert data["num_timesteps"] == 2
    assert data["num_gt_ids"] == 2
    assert data["num_gt_dets"] == 2


def test_build_sequence_pads_missing_frames_as_empty() -> None:
    data = build_sequence([{1: (0.0, 0.0)}], [], range(3), zero_distance=2.0)
    assert data["num_timesteps"] == 3
    assert data["num_gt_dets"] == 1
    assert data["similarity_scores"][2].shape == (0, 0)


def test_frame_ranges_test_split_is_the_last_forty() -> None:
    ranges = frame_ranges(400)
    assert ranges["all"] == range(400)
    assert ranges["test"] == range(360, 400)
    assert frame_ranges(10)["test"] == range(0, 10)


def test_aggregate_groups_by_scenario_prefix() -> None:
    res = {"R640|area|all|1m:r1": {"HOTA": 10.0}, "R640|area|all|1m:r2": {"HOTA": 20.0}}
    out = aggregate(res, ("HOTA",))
    assert out == {"R640|area|all|1m": "n=2 | 15.0 ± 7.1"}


def test_ground_nms_keeps_the_best_supported_point_and_breaks_ties_by_id() -> None:
    points = {5: (0.0, 0.0), 2: (0.3, 0.0), 9: (0.4, 0.0), 7: (3.0, 0.0)}
    support = {5: 1, 2: 1, 9: 3, 7: 1}
    # 9 has the most cameras and wins; 5 and 2 fall within 0.5 m of it; 7 is far away.
    assert ground_nms(points, support, 0.5) == {9: (0.4, 0.0), 7: (3.0, 0.0)}
    # Equal support: the lower Global ID survives, every frame.
    assert list(ground_nms({8: (0.0, 0.0), 3: (0.2, 0.0)}, {8: 2, 3: 2}, 0.5)) == [3]


def test_fuse_frames_applies_nms_after_the_area_filter() -> None:
    raw = [{1: [(0.0, 0.0)], 2: [(0.2, 0.0), (0.2, 0.0)], 3: [(50.0, 0.0)]}]
    assert fuse_frames(raw, area_filter=True, nms_m=0.5) == [{2: (0.2, 0.0)}]
    assert set(fuse_frames(raw, area_filter=True)[0]) == {1, 2}
    assert set(fuse_frames(raw, area_filter=False, nms_m=0.5)[0]) == {2, 3}
