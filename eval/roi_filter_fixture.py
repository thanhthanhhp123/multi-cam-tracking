"""Drop detections whose foot point falls outside a ground-plane region of interest (ROI).

    PYTHONPATH=src python -m eval.roi_filter_fixture \\
        --homography-dir configs/cameras/homography/wildtrack --margin-m 0 \\
        --in data/fixtures/ds_wildtrack_7cam_pnt_r1.jsonl \\
        --out data/fixtures/ds_wildtrack_7cam_pnt_roi0_r1.jsonl

Why this exists (docs/worklog/2026-09-28-30-*, decisions 6-7): PeopleNet Transformer finds
more people than YOLO11s, but 84% of its extra "false positives" stand outside the 12 m x 36 m
grid that WildTrack annotates, and their tracks also compete for Global IDs in `src/mct`
(ID switches 406 -> 669). A deployed system knows the area it monitors, so dropping detections
whose foot point (bottom-centre of the box, through the camera's ground-plane homography) lies
outside that area is a legitimate system feature, not an evaluation trick -- as long as it is
reported as "the system knows its monitored area" and applied identically to every detector.

Caveats, so the numbers are read correctly:

- This filters a RECORDED fixture, i.e. AFTER NvDCF has already built its tracks from the
  unfiltered detections. Filtering before the tracker should do at least as well (worklog
  phiên 22 saw the same direction for confidence filtering), so this is a lower bound.
- On WildTrack the ROI coincides with the scored area. The ground-plane protocol already scores
  only inside that area, so any gain there comes from cleaner association, not from hiding
  false positives. The image-box protocol scores the full frame, so a gain there is partly the
  removal of correct detections WildTrack does not annotate: report it, but lean on ground plane.
- A box whose foot cannot be projected (degenerate homography, above the horizon) is dropped.

Every frame message is kept, even when all its detections are dropped: the frame count is part
of the contract with `tools.ds_wildtrack_gt` and the evaluators.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from common.schema import Detection, FrameMessage, read_jsonl, write_jsonl
from eval.diagnose_fp_region import AREA_X_M, AREA_Y_M
from mct.homography import CameraHomography


@dataclass(slots=True, frozen=True)
class GroundRoi:
    """Axis-aligned rectangle on the ground plane, in metres, grown by `margin_m` on every side."""

    x_m: tuple[float, float] = AREA_X_M
    y_m: tuple[float, float] = AREA_Y_M
    margin_m: float = 0.0

    def contains(self, point: tuple[float, float]) -> bool:
        x, y = point
        m = self.margin_m
        return self.x_m[0] - m <= x < self.x_m[1] + m and self.y_m[0] - m <= y < self.y_m[1] + m


def foot_point(det: Detection) -> tuple[float, float]:
    x, y, w, h = det.bbox
    return (x + w / 2.0, y + h)


def filter_message(
    msg: FrameMessage, homography: CameraHomography, roi: GroundRoi
) -> tuple[FrameMessage, int]:
    """Return the message with out-of-ROI detections removed, and how many were removed."""
    kept: list[Detection] = []
    for det in msg.detections:
        ground = homography.project(foot_point(det))
        if ground is not None and roi.contains(ground):
            kept.append(det)
    return dataclasses.replace(msg, detections=kept), len(msg.detections) - len(kept)


def filter_messages(
    messages: Iterable[FrameMessage],
    homography_dir: Path,
    roi: GroundRoi,
    dropped: Counter[str],
    total: Counter[str],
) -> Iterator[FrameMessage]:
    homographies: dict[str, CameraHomography] = {}
    for msg in messages:
        if msg.cam_id not in homographies:
            homographies[msg.cam_id] = CameraHomography.load(homography_dir / f"{msg.cam_id}.yaml")
        out, n_dropped = filter_message(msg, homographies[msg.cam_id], roi)
        dropped[msg.cam_id] += n_dropped
        total[msg.cam_id] += len(msg.detections)
        yield out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="src", type=Path, required=True, help="pipeline fixture .jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--homography-dir", type=Path, required=True)
    p.add_argument(
        "--margin-m",
        type=float,
        default=0.0,
        help="grow the ROI by this many metres on every side (default: the WildTrack grid)",
    )
    args = p.parse_args(argv)

    roi = GroundRoi(margin_m=args.margin_m)
    dropped: Counter[str] = Counter()
    total: Counter[str] = Counter()
    n_msg = write_jsonl(
        args.out, filter_messages(read_jsonl(args.src), args.homography_dir, roi, dropped, total)
    )
    n_total, n_dropped = sum(total.values()), sum(dropped.values())
    summary = {
        "in": str(args.src),
        "out": str(args.out),
        "roi_x_m": list(roi.x_m),
        "roi_y_m": list(roi.y_m),
        "margin_m": roi.margin_m,
        "messages": n_msg,
        "detections": n_total,
        "dropped": n_dropped,
        "dropped_per_cam": {c: [dropped[c], total[c]] for c in sorted(total)},
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
