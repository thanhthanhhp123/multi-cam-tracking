"""Identity-free geometry of lab sessions, extracted once from fixtures for `eval.fit_learned`.

A pipeline fixture carries a 512-d embedding per detection; the fits only need boxes, timestamps
and (for some of them) which ground-truth person a pipeline track follows. This module reads the
fixtures once, drops the embeddings and caches plain arrays, so a leave-one-session-out loop can
refit on every fold without re-parsing ~25 MB per run.

**Leakage guard.** Every entry point goes through `check_sessions`: the held-out group
(`eval.aggregate_lab_eval.HELD_OUT`) can never be loaded here, whatever the caller asks for.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.schema import FrameMessage, read_jsonl
from eval.aggregate_lab_eval import HELD_OUT
from mct.roi import RoiFilter
from tools import assign_gt
from tools.export_trackeval import load_gt_table

CALIB_SESSIONS = ("calib", "calib_2")
RUNS = ("r1", "r2", "r3")


class LeakageError(AssertionError):
    """A fit or a development-time read touched the held-out sessions."""


def check_sessions(sessions: Iterable[str], *, exclude: Iterable[str] = ()) -> list[str]:
    """The session list, verified disjoint from the held-out group (and from `exclude`).

    `exclude` is the test session of a leave-one-out fold. Raises `LeakageError` (an
    `AssertionError`) rather than filtering: a caller that asks for a held-out session has a
    bug, and silently dropping it would hide that.
    """
    sessions = list(sessions)
    banned = set(HELD_OUT) | set(exclude)
    leaked = sorted(set(sessions) & banned)
    if leaked:
        raise LeakageError(f"buổi không được dùng để fit/phát triển: {leaked}")
    if not sessions:
        raise ValueError("danh sách buổi rỗng")
    return sessions


@dataclass(frozen=True, slots=True)
class Boxes:
    """Boxes of one fixture as parallel arrays (one row per detection)."""

    session: str
    cam: np.ndarray
    """Camera id per row (str)."""
    ts_ms: np.ndarray
    track: np.ndarray
    """Local track id (pipeline) or CVAT track id (ground truth)."""
    bbox: np.ndarray
    """(n, 4) x, y, w, h in pixels."""
    person: np.ndarray
    """Ground-truth person the track follows; -1 when the track could not be assigned."""
    frame_size: tuple[int, int]

    def of_cam(self, cam_id: str) -> np.ndarray:
        return self.cam == cam_id


def _to_boxes(
    session: str, messages: Sequence[FrameMessage], table: dict[tuple[str, int], int]
) -> Boxes:
    rows = [
        (m.cam_id, int(m.ts_ms), int(d.local_track_id), *map(float, d.bbox))
        for m in messages
        for d in m.detections
    ]
    size = (int(messages[0].frame_width), int(messages[0].frame_height))
    cam = np.array([r[0] for r in rows], dtype=str)
    track = np.array([r[2] for r in rows], dtype=np.int64)
    person = np.array([table.get((r[0], r[2]), -1) for r in rows], dtype=np.int64)
    return Boxes(
        session=session,
        cam=cam,
        ts_ms=np.array([r[1] for r in rows], dtype=np.int64),
        track=track,
        bbox=np.array([r[3:] for r in rows], dtype=np.float64).reshape(-1, 4),
        person=person,
        frame_size=size,
    )


def _filtered(path: Path, roi_filter: RoiFilter | None) -> list[FrameMessage]:
    messages = list(read_jsonl(path))
    if roi_filter is None:
        return messages
    return [roi_filter.filter(m)[0] for m in messages]


@dataclass(frozen=True, slots=True)
class LabData:
    """Where the fixtures live, and the engine's ROI filter (applied to everything read)."""

    fixtures_dir: Path
    """Pipeline fixtures: `lab_<session>_<run>.jsonl`."""
    gt_dir: Path
    """Ground-truth fixtures: `lab_<session>_gt.jsonl` + `.gt.json`."""
    roi_filter: RoiFilter | None = None
    cache_dir: Path | None = None

    def gt_fixture(self, session: str) -> Path:
        return self.gt_dir / f"lab_{session}_gt.jsonl"

    def run_fixture(self, session: str, run: str) -> Path:
        return self.fixtures_dir / f"lab_{session}_{run}.jsonl"

    def _cached(self, name: str, build) -> Boxes:  # type: ignore[no-untyped-def]
        path = None if self.cache_dir is None else self.cache_dir / f"{name}.npz"
        if path is not None and path.is_file():
            with np.load(path) as z:
                return Boxes(
                    session=str(z["session"]),
                    cam=z["cam"],
                    ts_ms=z["ts_ms"],
                    track=z["track"],
                    bbox=z["bbox"],
                    person=z["person"],
                    frame_size=tuple(int(v) for v in z["frame_size"]),  # type: ignore[arg-type]
                )
        boxes = build()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                session=boxes.session,
                cam=boxes.cam,
                ts_ms=boxes.ts_ms,
                track=boxes.track,
                bbox=boxes.bbox,
                person=boxes.person,
                frame_size=np.array(boxes.frame_size),
            )
        return boxes

    def gt_boxes(self, session: str) -> Boxes:
        """Ground-truth boxes of a session, with the person of each box."""
        check_sessions([session])

        def build() -> Boxes:
            table = load_gt_table(self.gt_fixture(session).with_suffix(".gt.json"))
            return _to_boxes(session, _filtered(self.gt_fixture(session), self.roi_filter), table)

        return self._cached(f"{session}_gt", build)

    def run_boxes(self, session: str, run: str) -> Boxes:
        """Pipeline boxes of one run; `person` comes from the same IoU vote as `tools.assign_gt`."""
        check_sessions([session])

        def build() -> Boxes:
            gt_table = load_gt_table(self.gt_fixture(session).with_suffix(".gt.json"))
            gt = _filtered(self.gt_fixture(session), self.roi_filter)
            messages = _filtered(self.run_fixture(session, run), self.roi_filter)
            tracks, _ = assign_gt.collect_votes(
                messages, assign_gt.index_gt(gt, gt_table), min_iou=0.5
            )
            rows, _ = assign_gt.build_table(tracks, min_purity=0.7, min_matched=3)
            table = {(r["cam_id"], int(r["local_track_id"])): int(r["gt_global_id"]) for r in rows}
            return _to_boxes(session, messages, table)

        return self._cached(f"{session}_{run}", build)

    def pipeline(self, sessions: Iterable[str]) -> list[Boxes]:
        """Every pipeline run of the given (non-calibration) sessions."""
        return [
            self.run_boxes(s, run)
            for s in check_sessions(sessions)
            if s not in CALIB_SESSIONS
            for run in RUNS
            if self.run_fixture(s, run).is_file()
        ]

    def ground_truth(self, sessions: Iterable[str]) -> list[Boxes]:
        return [self.gt_boxes(s) for s in check_sessions(sessions)]
