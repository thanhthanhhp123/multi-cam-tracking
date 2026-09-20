"""eval/diagnose_fp_region.py: bucket unmatched detections by annotated area and GT overlap."""

from __future__ import annotations

import numpy as np
from eval.diagnose_fp_region import (
    NEAR_GT,
    NO_GT,
    OUTSIDE,
    classify_frame,
    in_annotated_area,
    iou_matrix,
)

from mct.homography import CameraHomography
from tools.wildtrack_to_fixture import position_id_to_world_m

# Identity homography: the foot pixel (x + w/2, y + h) IS the ground point in metres.
IDENTITY = CameraHomography(cam_id="cam01", matrix=np.eye(3))
GT = np.array([[0.0, 0.0, 2.0, 1.0]])  # foot (1, 1): inside the annotated area


def test_iou_matrix_known_values() -> None:
    a = np.array([[0.0, 0.0, 2.0, 1.0]])
    b = np.array([[0.0, 0.0, 2.0, 1.0], [1.0, 0.0, 2.0, 1.0], [9.0, 9.0, 1.0, 1.0]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]])


def test_iou_matrix_handles_empty_inputs() -> None:
    assert iou_matrix(np.zeros((0, 4)), GT).shape == (0, 1)
    assert iou_matrix(GT, np.zeros((0, 4))).shape == (1, 0)


def test_area_bounds_match_the_wildtrack_grid() -> None:
    assert in_annotated_area(position_id_to_world_m(0))
    assert in_annotated_area(position_id_to_world_m(480 * 1440 - 1))
    assert not in_annotated_area((-3.5, 0.0))
    assert not in_annotated_area((0.0, 27.5))


def test_every_bucket_and_the_match() -> None:
    det = np.array(
        [
            [0.0, 0.0, 2.0, 1.0],  # exact match -> TP
            [0.0, 0.0, 2.0, 1.0],  # duplicate of the matched person -> inside_near_gt (IoU 1.0)
            [1.0, 0.0, 2.0, 1.0],  # IoU 1/3 with the person, foot (2, 1) -> inside_near_gt
            [5.0, 5.0, 2.0, 1.0],  # foot (6, 6), overlaps nobody -> inside_no_gt
            [50.0, 0.0, 2.0, 1.0],  # foot (51, 1) is off the grid -> outside_area
        ]
    )
    n_tp, buckets = classify_frame(det, GT, IDENTITY, min_iou=0.5)
    assert n_tp == 1
    assert sorted(buckets) == sorted([NEAR_GT, NEAR_GT, NO_GT, OUTSIDE])


def test_outside_takes_priority_over_overlap() -> None:
    gt = np.array([[50.0, 0.0, 2.0, 1.0]])  # an annotated person, but off the grid
    det = np.array([[50.5, 0.0, 2.0, 1.0]])
    n_tp, buckets = classify_frame(det, gt, IDENTITY, min_iou=0.9)
    assert n_tp == 0
    assert buckets == [OUTSIDE]


def test_no_ground_truth_in_frame() -> None:
    det = np.array([[0.0, 0.0, 2.0, 1.0], [50.0, 0.0, 2.0, 1.0]])
    n_tp, buckets = classify_frame(det, np.zeros((0, 4)), IDENTITY, min_iou=0.5)
    assert n_tp == 0
    assert buckets == [NO_GT, OUTSIDE]


def test_no_detections_in_frame() -> None:
    assert classify_frame(np.zeros((0, 4)), GT, IDENTITY, min_iou=0.5) == (0, [])
