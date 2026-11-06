"""tools.lab_oracle_fixture: GT interpolation and the C/B fixture split."""

from __future__ import annotations

import numpy as np
import pytest

from common.schema import Detection, FrameMessage, l2_normalize
from tools.lab_oracle_fixture import interpolated_gt, split_fixture


def _msg(cam: str, frame: int, dets: list[Detection]) -> FrameMessage:
    return FrameMessage(
        cam_id=cam,
        frame_id=frame,
        ts_ms=1_000_000 + frame * 40,
        frame_pts_ns=frame * 40_000_000,
        frame_width=1920,
        frame_height=1080,
        detections=dets,
        embed_dim=4 if any(d.embedding is not None for d in dets) else 0,
    )


def _det(tid: int, box: tuple[float, float, float, float], emb: bool = False) -> Detection:
    return Detection(
        local_track_id=tid,
        bbox=box,
        confidence=0.9,
        embedding=l2_normalize(np.arange(1, 5, dtype=np.float32)) if emb else None,
    )


def test_interpolates_between_consecutive_annotated_frames() -> None:
    gt = [
        _msg("cam01", 0, [_det(7, (0.0, 0.0, 10.0, 10.0))]),
        _msg("cam01", 5, [_det(7, (50.0, 0.0, 10.0, 10.0))]),
    ]
    out = interpolated_gt(gt, {("cam01", 7): 3}, step=5)
    assert sorted(f for _, f in out) == [0, 1, 2, 3, 4, 5]
    gid, box = out[("cam01", 2)][0]
    assert gid == 3
    assert box == pytest.approx((20.0, 0.0, 10.0, 10.0))


def test_does_not_interpolate_across_gap_and_skips_unlabelled_tracks() -> None:
    gt = [
        _msg("cam01", 0, [_det(1, (0.0, 0.0, 10.0, 10.0)), _det(2, (100.0, 0.0, 10.0, 10.0))]),
        _msg("cam01", 50, [_det(1, (0.0, 0.0, 10.0, 10.0))]),
    ]
    out = interpolated_gt(gt, {("cam01", 1): 1}, step=5)  # track 2 has no identity
    assert set(out) == {("cam01", 0), ("cam01", 50)}
    assert all(len(rows) == 1 for rows in out.values())


def test_split_keeps_same_detections_and_only_changes_ids() -> None:
    gt = {("cam01", 0): [(5, (0.0, 0.0, 100.0, 200.0)), (6, (500.0, 0.0, 100.0, 200.0))]}
    pipeline = [
        _msg(
            "cam01",
            0,
            [
                _det(41, (2.0, 1.0, 100.0, 200.0), emb=True),  # person 5
                _det(42, (900.0, 0.0, 100.0, 200.0), emb=True),  # nobody: dropped
                _det(43, (498.0, 0.0, 100.0, 200.0), emb=True),  # person 6
            ],
        ),
        _msg("cam01", 1, [_det(41, (2.0, 1.0, 100.0, 200.0))]),  # no GT here: frame kept, empty
    ]
    det, oracle, stats = split_fixture(pipeline, gt, min_iou=0.5)
    assert stats == {"n_detections": 4, "n_kept": 2}
    assert [len(m.detections) for m in det] == [2, 0]
    assert [d.local_track_id for d in det[0].detections] == [41, 43]
    assert [d.local_track_id for d in oracle[0].detections] == [5, 6]
    for a, b in zip(det[0].detections, oracle[0].detections, strict=True):
        assert a.bbox == b.bbox
        assert np.array_equal(a.embedding, b.embedding)
    assert det[1].ts_ms == oracle[1].ts_ms == pipeline[1].ts_ms
