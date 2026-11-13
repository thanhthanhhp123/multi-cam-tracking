"""Learned geometry models (src/mct/learned.py) and the four `association.*` keys that use them.

Every key defaults to null, and null must leave the engine exactly as it was: the last tests of
this file pin that, next to tests showing each model does change the result when set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from common.schema import CLASS_PERSON, Detection, FrameMessage, l2_normalize
from mct.affinity import INFEASIBLE, AffinityConfig, _synchronized_distance, build_cost_matrix
from mct.gallery import Gallery
from mct.learned import (
    EntryZoneModel,
    ExitDirectionModel,
    FootModel,
    GroundErrorModel,
    exit_velocity,
    load_learned,
)
from mct.topology import Topology
from mct.tracklet import Tracklet, TrackletBuilder, TrackletConfig

DIM = 16
VEC = l2_normalize(np.ones(DIM, dtype=np.float32))
PROVENANCE = {"fit_sessions": ["s2_1"], "fit_date": "2026-11-13"}

NON_OVERLAP = Topology.from_mapping(
    {
        "cameras": {"cam01": {}, "cam03": {}},
        "transitions": [
            {"from": "cam01", "to": "cam03", "bidirectional": True, "min_ms": 0, "max_ms": 60000}
        ],
    }
)


class _MetricGround:
    """100 px = 1 m, every camera in the same frame."""

    def project(self, cam_id, point):
        return (point[0] / 100.0, point[1] / 100.0)

    def distance_m(self, cam_a, point_a, cam_b, point_b):
        return float(np.hypot(*(np.subtract(point_a, point_b) / 100.0)))


def _write(tmp_path: Path, name: str, body: dict) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump({**PROVENANCE, **body}), encoding="utf-8")
    return path


def _walk(
    cam_id: str,
    local_id: int,
    *,
    tracklet_id: int,
    start_ms: int = 0,
    x0: float = 100.0,
    dx: float = 20.0,
    y_bottom: float = 400.0,
    step_ms: int = 500,
    config: TrackletConfig | None = None,
) -> Tracklet:
    """Nine frames, `step_ms` apart, walking along x at `dx` px per frame (100 px = 1 m)."""
    builder = TrackletBuilder(config or TrackletConfig(min_frames=1))
    for i in range(9):
        det = Detection(
            local_track_id=local_id,
            bbox=(x0 + dx * i - 40.0, y_bottom - 200.0, 80.0, 200.0),
            confidence=0.9,
            embedding=VEC,
            class_id=CLASS_PERSON,
        )
        builder.update(
            FrameMessage(
                cam_id=cam_id,
                frame_id=i,
                ts_ms=start_ms + i * step_ms,
                frame_pts_ns=i * step_ms * 1_000_000,
                frame_width=1920,
                frame_height=1080,
                detections=[det],
                embed_dim=DIM,
            )
        )
    tracklet = builder.flush()[0]
    tracklet.tracklet_id = tracklet_id
    return tracklet


# --------------------------------------------------------------------------- provenance


def test_file_thieu_nguon_goc_bi_tu_choi(tmp_path):
    path = tmp_path / "foot.yaml"
    path.write_text(yaml.safe_dump({"kind": "foot_height", "cameras": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="fit_sessions"):
        load_learned(path, "foot_height")
    with pytest.raises(ValueError, match="kind"):
        load_learned(_write(tmp_path, "x.yaml", {"kind": "entry_zones"}), "foot_height")


# --------------------------------------------------------------------------- foot


def test_hop_khong_cham_mep_giu_nguyen_day_giua():
    model = FootModel("height_line", 30.0, {"cam01": (0.8, 0.0)}, {})
    assert model.foot("cam01", (100.0, 200.0, 80.0, 400.0), 1080.0) == (140.0, 600.0)


def test_height_line_suy_chan_tu_dinh_dau():
    """h = 0.8 * y_foot  =>  y_foot = y_top / (1 - 0.8)."""
    model = FootModel("height_line", 30.0, {"cam01": (0.8, 0.0)}, {})
    x, y = model.foot("cam01", (100.0, 300.0, 80.0, 780.0), 1080.0)
    assert (x, y) == (140.0, pytest.approx(1500.0))
    # Never above the visible bottom, and an unknown camera is left alone.
    assert model.foot("cam01", (100.0, 100.0, 80.0, 980.0), 1080.0)[1] == 1080.0
    assert model.foot("cam09", (100.0, 300.0, 80.0, 780.0), 1080.0) == (140.0, 1080.0)


def test_head_plane_chieu_dinh_dau_va_khong_keo_chan_len():
    shift = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 900.0], [0.0, 0.0, 1.0]])  # foot = head + 900 px
    model = FootModel("head_plane", 30.0, {}, {"cam01": shift})
    assert model.foot("cam01", (100.0, 300.0, 80.0, 780.0), 1080.0) == pytest.approx(
        (140.0, 1200.0)
    )
    assert model.foot("cam01", (100.0, 100.0, 80.0, 980.0), 1080.0) == (140.0, 1080.0)


def test_foot_model_doi_diem_chan_cua_tracklet(tmp_path):
    path = _write(
        tmp_path,
        "foot.yaml",
        {
            "kind": "foot_height",
            "method": "height_line",
            "bottom_margin_px": 30,
            "cameras": {"cam01": {"slope": 0.8, "intercept": 0.0}},
        },
    )
    config = TrackletConfig.from_mapping(
        {"tracklet": {"min_frames": 1}, "association": {"foot_from_height": str(path)}}
    )
    cut = _walk("cam01", 1, tracklet_id=1, y_bottom=1080.0, config=config)
    plain = _walk("cam01", 1, tracklet_id=1, y_bottom=1080.0)
    assert plain.last_ground_point[1] == 1080.0
    assert cut.last_ground_point[1] == pytest.approx(880.0 / 0.2)  # y_top = 880
    assert cut.last_bbox == plain.last_bbox, "the box itself is never touched"


# --------------------------------------------------------------------------- ground error


def _error_model(mu=(0.5, 0.0), sigma=0.2, sigma_ref=0.4) -> GroundErrorModel:
    return GroundErrorModel(
        {
            "cam01|cam02": {
                "cell_m": 1.0,
                "sigma_ref": sigma_ref,
                "default": (np.zeros(2), sigma_ref),
                "cells": {(0, 0): (np.array(mu), sigma)},
            }
        }
    )


def test_ground_error_tru_do_lech_va_chia_do_tan():
    model = _error_model()
    a = np.array([[0.2, 0.5], [5.0, 0.5]])  # cam01
    b = np.array([[0.7, 0.5], [5.5, 0.5]])  # cam02: +0.5 m in x
    d = model.distances("cam01", a, "cam02", b)
    # Row 0 sits in cell (0, 0): the offset is the learned bias -> 0. Row 1 has no cell.
    assert d == pytest.approx([0.0, 0.5])
    # Same pairs given in the other camera order: same answer.
    assert model.distances("cam02", b, "cam01", a) == pytest.approx([0.0, 0.5])
    # No bias left: 0.3 m off, in a cell twice as tight as the reference -> counts double.
    assert model.distances("cam01", a[:1], "cam02", b[:1] + np.array([0.3, 0.0])) == pytest.approx(
        [0.6]
    )
    assert model.distances("cam01", a, "cam03", b) is None


def test_khoang_cach_dong_bo_dung_mo_hinh_khi_co():
    a = np.array([[0.0, 0.2, 0.5], [500.0, 0.3, 0.5]])
    b = np.array([[0.0, 0.7, 0.5], [500.0, 0.8, 0.5]])
    assert _synchronized_distance(a, b, 100) == pytest.approx(0.5)
    corrected = _synchronized_distance(
        a, b, 100, error_model=_error_model(), cams=("cam01", "cam02")
    )
    assert corrected == pytest.approx(0.0)


# --------------------------------------------------------------------------- exit direction


def test_exit_velocity_va_cong_huong():
    t = np.arange(0, 2001, 200.0)
    toward = np.column_stack([t, 1.0 + t / 1000.0, np.zeros_like(t)])  # +1 m/s along x
    away = np.column_stack([t, 5.0 - t / 1000.0, np.zeros_like(t)])
    still = np.column_stack([t, np.full_like(t, 3.0), np.zeros_like(t)])
    assert exit_velocity(toward, 1000.0) == pytest.approx([1.0, 0.0], abs=1e-6)
    assert exit_velocity(toward[:2], 1000.0) is None

    gate = ExitDirectionModel(1000.0, 0.3, 0.0, {("cam01", "cam03"): np.array([1.0, 0.0])})
    assert gate.contradicts("cam01", "cam03", toward) is None
    assert gate.contradicts("cam01", "cam03", away) == pytest.approx(-1.0)
    assert gate.contradicts("cam01", "cam03", still) is None, "standing still is not evidence"
    assert gate.contradicts("cam03", "cam01", away) is None, "unknown pair: no gate"


def test_cong_huong_loai_track_roi_khung_nguoc_huong():
    gate = ExitDirectionModel(1000.0, 0.3, 0.0, {("cam01", "cam03"): np.array([1.0, 0.0])})
    arrival = _walk("cam03", 5, tracklet_id=9, start_ms=10_000)

    def cost(dx: float, config: AffinityConfig) -> tuple[float, str]:
        gallery = Gallery()
        track = gallery.create(_walk("cam01", 1, tracklet_id=1, x0=900.0, dx=dx, step_ms=200))
        matrix = build_cost_matrix(
            [arrival], [track], topology=NON_OVERLAP, config=config, ground_mapper=_MetricGround()
        )
        return float(matrix.costs[0, 0]), matrix.reason(0, 0)

    gated = AffinityConfig(max_cost=0.9, exit_direction_model=gate)
    assert np.isfinite(cost(+20.0, gated)[0])
    away_cost, reason = cost(-20.0, gated)
    assert away_cost == INFEASIBLE and "ngược hướng" in reason
    assert np.isfinite(cost(-20.0, AffinityConfig(max_cost=0.9))[0]), "null: no gate"


# --------------------------------------------------------------------------- entry zones


def _entry_model() -> EntryZoneModel:
    p = np.zeros((2, 2))
    p[1, 0] = 0.9  # bottom-left quarter of the frame is where people arrive
    return EntryZoneModel(0.8, 10, {"cam01": (p, np.full((2, 2), 50), (1920.0, 1080.0))})


def test_vung_vao_theo_o_anh():
    model = _entry_model()
    assert model.is_entry("cam01", (100.0, 700.0, 80.0, 200.0))
    assert not model.is_entry("cam01", (1500.0, 700.0, 80.0, 200.0))
    assert not model.is_entry("cam02", (100.0, 700.0, 80.0, 200.0))
    # A box hanging below the frame is clipped into the last row, not dropped.
    assert model.is_entry("cam01", (100.0, 900.0, 80.0, 400.0))


def test_tracklet_bat_dau_o_vung_vao_mat_ngan_sach_noi_cung_camera():
    """Same camera, plausible walking distance: evidence in mid-frame, not at the door."""
    config = AffinityConfig(max_cost=0.3, max_cost_geometric=0.9)
    with_prior = AffinityConfig(
        max_cost=0.3, max_cost_geometric=0.9, entry_zone_model=_entry_model()
    )
    mapper = _MetricGround()

    def geometric(y_bottom: float, cfg: AffinityConfig) -> bool:
        gallery = Gallery()
        track = gallery.create(_walk("cam01", 1, tracklet_id=1, y_bottom=y_bottom))
        later = _walk("cam01", 2, tracklet_id=2, start_ms=6_000, x0=300.0, y_bottom=y_bottom)
        matrix = build_cost_matrix([later], [track], config=cfg, ground_mapper=mapper)
        assert np.isfinite(matrix.costs[0, 0])
        return bool(matrix.geometric[0, 0])  # type: ignore[index]

    assert geometric(900.0, config) and geometric(400.0, config)
    assert not geometric(900.0, with_prior), "starts in the entry cell: appearance budget only"
    assert geometric(400.0, with_prior), "starts mid-frame: unchanged"


# --------------------------------------------------------------------------- null = old behaviour


def test_bon_khoa_mac_dinh_null():
    for data in ({}, {"association": {}}, {"association": dict.fromkeys(KEYS)}):
        cfg = AffinityConfig.from_mapping(data)
        assert cfg.ground_error_model is None
        assert cfg.exit_direction_model is None
        assert cfg.entry_zone_model is None
        assert TrackletConfig.from_mapping(data).foot_model is None
    shipped = yaml.safe_load(Path("configs/mct.yaml").read_text(encoding="utf-8"))
    assert all(shipped["association"][key] is None for key in KEYS)


KEYS = ("foot_from_height", "ground_sigma_model", "exit_direction_gate", "entry_zone_prior")


def test_null_cho_ma_tran_chi_phi_trung_tung_bit():
    """Explicit None for the three affinity models = the config that never heard of them."""
    gallery = Gallery()
    tracks = [
        gallery.create(_walk("cam01", 1, tracklet_id=1)),
        gallery.create(_walk("cam03", 7, tracklet_id=3, x0=500.0)),
    ]
    rows = [
        _walk("cam01", 2, tracklet_id=2, start_ms=6_000, x0=300.0),
        _walk("cam03", 8, tracklet_id=4, start_ms=9_000, x0=700.0),
    ]
    old = AffinityConfig(max_cost=0.4, max_cost_geometric=0.9, stitch_contested_max_gap_ms=0)
    new = AffinityConfig.from_mapping(
        {
            "association": {
                "max_cost": 0.4,
                "max_cost_geometric": 0.9,
                "stitch_contested_max_gap_ms": 0,
                **dict.fromkeys(KEYS),
            }
        }
    )
    a, b = (
        build_cost_matrix(
            rows, tracks, topology=NON_OVERLAP, config=cfg, ground_mapper=_MetricGround()
        )
        for cfg in (old, new)
    )
    assert a.costs.tobytes() == b.costs.tobytes()
    assert a.geometric.tobytes() == b.geometric.tobytes()  # type: ignore[union-attr]
    assert a.reasons == b.reasons
