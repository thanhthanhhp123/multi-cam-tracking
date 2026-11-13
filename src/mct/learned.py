"""Learned geometry models the engine READS (it never fits anything at run time).

Four optional, identity-free components, each behind its own `association.*` key whose value
is the path of a fitted YAML file (`null` = component off, old behaviour bit for bit):

- `foot_from_height`    -> `FootModel`: where the feet are when the box is cut by the image bottom;
- `ground_sigma_model`  -> `GroundErrorModel`: expected cross-camera foot offset per floor cell;
- `exit_direction_gate` -> `ExitDirectionModel`: which way people leave camera A to reach camera B;
- `entry_zone_prior`    -> `EntryZoneModel`: where in the image new arrivals first appear.

The files are written by `eval.fit_learned` from an explicit list of sessions. Every file must
carry `fit_sessions` and `fit_date`; a file without them is refused, so a hand-edited or
half-written model cannot slip into a run unnoticed. Nothing here learns from appearance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

Point = tuple[float, float]
Bbox = tuple[float, float, float, float]

FOOT_METHODS = ("height_line", "head_plane")


def load_learned(path: str | Path, kind: str) -> dict[str, Any]:
    """Parse a learned-model file and check it is the expected kind and carries provenance."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if data.get("kind") != kind:
        raise ValueError(f"{path}: cần kind={kind!r}, nhận {data.get('kind')!r}")
    if not data.get("fit_sessions") or not data.get("fit_date"):
        raise ValueError(
            f"{path}: thiếu fit_sessions/fit_date (file không do eval.fit_learned ghi)"
        )
    return data


def pair_key(cam_a: str, cam_b: str) -> str:
    """Unordered camera pair as one string key (`cam01|cam02`)."""
    return "|".join(sorted((cam_a, cam_b)))


@dataclass(frozen=True, slots=True)
class FootModel:
    """Foot point of a box whose bottom is cut by the image border.

    Two methods, both fitted per camera on boxes that are NOT cut:

    - `height_line`: person height in pixels as a line of the foot row, `h = a * y_foot + b`,
      so `y_foot = (y_top + b) / (1 - a)`. Exact for a level pinhole camera, but the relation
      bends close under a tilted camera, which is exactly where boxes get cut.
    - `head_plane`: a homography from the HEAD point (top-centre of the box) straight to the
      ground position, i.e. the plane at average head height. Same assumption (people have an
      average height) without the linearisation.

    The returned point is in image pixels and may lie below the frame; `head_plane` returns the
    image point whose ground-plane projection equals the head-plane projection, so everything
    downstream keeps using the one ground homography.
    """

    method: str
    bottom_margin_px: float
    lines: dict[str, tuple[float, float]]
    """cam_id -> (a, b) of `height_line`."""
    head_to_image: dict[str, np.ndarray]
    """cam_id -> 3x3 mapping the head point to the (virtual) foot point, for `head_plane`."""

    def foot(self, cam_id: str, bbox: Bbox, frame_height: float) -> Point:
        x, y, w, h = bbox
        bottom = y + h
        centre = x + w / 2.0
        if bottom < frame_height - self.bottom_margin_px:
            return (centre, bottom)
        if self.method == "height_line":
            line = self.lines.get(cam_id)
            if line is None or line[0] >= 1.0:
                return (centre, bottom)
            return (centre, max(bottom, (y + line[1]) / (1.0 - line[0])))
        matrix = self.head_to_image.get(cam_id)
        if matrix is None:
            return (centre, bottom)
        u, v, s = matrix @ np.array([centre, y, 1.0])
        # Never move the feet UP into the frame, and refuse a degenerate projection.
        if abs(s) < 1e-12 or v / s < bottom:
            return (centre, bottom)
        return (float(u / s), float(v / s))

    @classmethod
    def load(cls, path: str | Path) -> FootModel:
        data = load_learned(path, "foot_height")
        method = str(data.get("method", "height_line"))
        if method not in FOOT_METHODS:
            raise ValueError(f"{path}: method phải thuộc {FOOT_METHODS}, nhận {method!r}")
        cams = data.get("cameras") or {}
        return cls(
            method=method,
            bottom_margin_px=float(data.get("bottom_margin_px", 0.0)),
            lines={
                c: (float(v["slope"]), float(v["intercept"]))
                for c, v in cams.items()
                if "slope" in v
            },
            head_to_image={
                c: np.array(v["head_to_image"], dtype=np.float64).reshape(3, 3)
                for c, v in cams.items()
                if "head_to_image" in v
            },
        )


@dataclass(frozen=True, slots=True)
class GroundErrorModel:
    """Expected foot-point disagreement of ONE person seen by two overlapping cameras.

    Per floor cell (square of `cell_m`, indexed by the midpoint of the two projections): the
    mean offset `mu` (second camera minus first, cameras in sorted order) and its spread `sigma`.
    The engine replaces the raw distance `|delta|` by `|delta - mu| * sigma_ref / sigma`: a
    distance in metres-equivalent, so `homography_weight` and `max_ground_dist_m` keep their
    meaning. A `sigma`-only fit has `mu = 0` everywhere; a bias-only fit has `sigma = sigma_ref`.
    """

    pairs: dict[str, dict[str, Any]]
    """`pair_key` -> {cell_m, sigma_ref, default: (mu, sigma), cells: {(ix, iy): (mu, sigma)}}."""

    def has(self, cam_a: str, cam_b: str) -> bool:
        return pair_key(cam_a, cam_b) in self.pairs

    def distances(
        self, cam_a: str, pts_a: np.ndarray, cam_b: str, pts_b: np.ndarray
    ) -> np.ndarray | None:
        """Corrected distance per row of two (n, 2) world-point arrays; None for an unknown pair."""
        model = self.pairs.get(pair_key(cam_a, cam_b))
        if model is None:
            return None
        first, second = (pts_a, pts_b) if cam_a <= cam_b else (pts_b, pts_a)
        delta = second - first
        cells = np.floor((first + second) / 2.0 / model["cell_m"]).astype(np.int64)
        default_mu, default_sigma = model["default"]
        mu = np.tile(default_mu, (len(delta), 1))
        sigma = np.full(len(delta), default_sigma)
        table = model["cells"]
        for i, key in enumerate(map(tuple, cells.tolist())):
            hit = table.get(key)
            if hit is not None:
                mu[i], sigma[i] = hit
        return np.hypot(*(delta - mu).T) * model["sigma_ref"] / sigma

    @classmethod
    def load(cls, path: str | Path) -> GroundErrorModel:
        data = load_learned(path, "ground_sigma")
        pairs: dict[str, dict[str, Any]] = {}
        for key, raw in (data.get("pairs") or {}).items():
            default = raw["default"]
            pairs[pair_key(*key.split("|"))] = {
                "cell_m": float(raw["cell_m"]),
                "sigma_ref": float(raw["sigma_ref"]),
                "default": (np.array(default["mu"], dtype=np.float64), float(default["sigma"])),
                "cells": {
                    (int(c["ix"]), int(c["iy"])): (
                        np.array(c["mu"], dtype=np.float64),
                        float(c["sigma"]),
                    )
                    for c in raw.get("cells") or []
                },
            }
        return cls(pairs=pairs)


def exit_velocity(world_path: np.ndarray, window_ms: float) -> np.ndarray | None:
    """Ground velocity (m/s) over the last `window_ms` of a `[ts_ms, X, Y]` path; None if too short.

    Least-squares slope rather than last-minus-first: one jittery foot point at either end of a
    one-second window would otherwise decide the direction.
    """
    if len(world_path) < 4:
        return None
    tail = world_path[world_path[:, 0] >= world_path[-1, 0] - window_ms]
    if len(tail) < 4 or tail[-1, 0] - tail[0, 0] < window_ms / 2.0:
        return None
    t = (tail[:, 0] - tail[0, 0]) / 1000.0
    return np.array([np.polyfit(t, tail[:, 1], 1)[0], np.polyfit(t, tail[:, 2], 1)[0]])


@dataclass(frozen=True, slots=True)
class ExitDirectionModel:
    """Direction people walk when they leave camera `src` on their way to camera `dst`.

    A candidate GlobalTrack last seen at `src` is ruled out for a tracklet at `dst` when it was
    clearly moving (`min_speed_m_s`) the other way (`cos < min_cos`) as it left the frame. A
    track that stood still, or whose exit path is too short, is never ruled out: no evidence.
    """

    window_ms: float
    min_speed_m_s: float
    min_cos: float
    directions: dict[tuple[str, str], np.ndarray]
    """(src, dst) -> unit vector on the ground plane."""

    def contradicts(self, src: str, dst: str, world_path: np.ndarray) -> float | None:
        """Cosine between the exit velocity and the learned direction when it rules the pair out."""
        direction = self.directions.get((src, dst))
        if direction is None:
            return None
        velocity = exit_velocity(world_path, self.window_ms)
        if velocity is None:
            return None
        speed = float(np.hypot(*velocity))
        if speed < self.min_speed_m_s:
            return None
        cos = float(velocity @ direction) / speed
        return cos if cos < self.min_cos else None

    @classmethod
    def load(cls, path: str | Path) -> ExitDirectionModel:
        data = load_learned(path, "exit_direction")
        return cls(
            window_ms=float(data["window_ms"]),
            min_speed_m_s=float(data["min_speed_m_s"]),
            min_cos=float(data["min_cos"]),
            directions={
                tuple(key.split(">")): np.array(v["direction"], dtype=np.float64)  # type: ignore[misc]
                for key, v in (data.get("pairs") or {}).items()
            },
        )


@dataclass(frozen=True, slots=True)
class EntryZoneModel:
    """Image cells where a tracklet that STARTS there is almost always a new arrival.

    Per camera a coarse grid over the frame holds the share of tracklet starts that were real
    arrivals (the person was not in that camera just before) rather than a tracker fragment of
    someone already there. The engine uses it to stop treating "same camera, plausible walking
    distance" as position evidence for a tracklet that starts in such a cell.
    """

    min_p: float
    min_n: int
    grids: dict[str, tuple[np.ndarray, np.ndarray, tuple[float, float]]]
    """cam_id -> (p_entry, n, (frame width, frame height)); the two arrays are (rows, cols)."""

    @staticmethod
    def cell(
        bbox: Bbox, frame_size: tuple[float, float], shape: tuple[int, int]
    ) -> tuple[int, int]:
        """(row, col) of the bottom-centre of `bbox`, clipped to the frame."""
        x, y, w, h = bbox
        rows, cols = shape
        col = int(np.clip((x + w / 2.0) / frame_size[0] * cols, 0, cols - 1))
        row = int(np.clip((y + h) / frame_size[1] * rows, 0, rows - 1))
        return row, col

    def is_entry(self, cam_id: str, bbox: Bbox) -> bool:
        grid = self.grids.get(cam_id)
        if grid is None:
            return False
        p_entry, n, frame_size = grid
        row, col = self.cell(bbox, frame_size, p_entry.shape)  # type: ignore[arg-type]
        return bool(n[row, col] >= self.min_n and p_entry[row, col] >= self.min_p)

    @classmethod
    def load(cls, path: str | Path) -> EntryZoneModel:
        data = load_learned(path, "entry_zones")
        return cls(
            min_p=float(data["min_p"]),
            min_n=int(data["min_n"]),
            grids={
                c: (
                    np.array(v["p_entry"], dtype=np.float64),
                    np.array(v["n"], dtype=np.int64),
                    (float(v["image_size"][0]), float(v["image_size"][1])),
                )
                for c, v in (data.get("cameras") or {}).items()
            },
        )


def head_to_image_matrix(head_to_ground: np.ndarray, image_to_ground: np.ndarray) -> np.ndarray:
    """Head point -> virtual foot point in the image: `inv(H_ground) @ H_head`."""
    return np.linalg.inv(image_to_ground) @ head_to_ground


__all__ = [
    "EntryZoneModel",
    "ExitDirectionModel",
    "FootModel",
    "GroundErrorModel",
    "exit_velocity",
    "head_to_image_matrix",
    "load_learned",
    "pair_key",
]
