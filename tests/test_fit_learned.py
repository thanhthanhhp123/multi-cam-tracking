"""`eval.fit_learned` + `eval.lab_geometry`: leakage guard, and fits recovering known geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from eval.aggregate_lab_eval import HELD_OUT
from eval.fit_learned import (
    fit_all,
    fit_entry_zones,
    fit_exit_direction,
    fit_foot,
    fit_ground_error,
    write_model,
)
from eval.lab_geometry import Boxes, LabData, LeakageError, check_sessions

from mct.homography import CameraHomography, HomographyMapper
from mct.learned import EntryZoneModel, ExitDirectionModel, FootModel, GroundErrorModel

SIZE = (1920, 1080)
CM = np.diag([0.01, 0.01, 1.0])  # 100 px = 1 m
MAPPER = HomographyMapper({c: CameraHomography(c, CM) for c in ("cam01", "cam02", "cam03")})
OVERLAPS = {"cam01": {"cam02"}, "cam02": {"cam01"}}


def _boxes(rows: list[tuple[str, int, int, tuple[float, float, float, float], int]]) -> Boxes:
    """rows = (cam, ts_ms, track, bbox, person)."""
    return Boxes(
        session="s2_1",
        cam=np.array([r[0] for r in rows], dtype=str),
        ts_ms=np.array([r[1] for r in rows], dtype=np.int64),
        track=np.array([r[2] for r in rows], dtype=np.int64),
        bbox=np.array([r[3] for r in rows], dtype=np.float64).reshape(-1, 4),
        person=np.array([r[4] for r in rows], dtype=np.int64),
        frame_size=SIZE,
    )


def _standing(cam: str, n: int = 2000, seed: int = 0) -> Boxes:
    """Whole people whose pixel height is 0.6 * foot row + 20."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        foot, centre = rng.uniform(300.0, 1040.0), rng.uniform(150.0, 1750.0)
        h = 0.6 * foot + 20.0
        rows.append((cam, i * 40, i, (centre - 0.15 * h, foot - h, 0.3 * h, h), i % 7))
    return _boxes(rows)


# --------------------------------------------------------------------------- leakage guard


def test_leakage_error_la_assertion():
    assert issubclass(LeakageError, AssertionError)


@pytest.mark.parametrize("held", HELD_OUT)
def test_buoi_giu_rieng_bi_chan_o_moi_cua(tmp_path, held):
    """The directories do not exist: reaching the disk would raise FileNotFoundError instead."""
    lab = LabData(tmp_path / "none", tmp_path / "none")
    with pytest.raises(LeakageError):
        check_sessions(["s2_1", held])
    for read in (
        lambda: lab.gt_boxes(held),
        lambda: lab.run_boxes(held, "r1"),
        lambda: lab.pipeline([held]),
    ):
        with pytest.raises(LeakageError):
            read()
    with pytest.raises(LeakageError):
        lab.ground_truth(["s2_1", held])
    with pytest.raises(LeakageError):
        fit_all(lab, ["s2_1", held], tmp_path / "out", mapper=MAPPER, overlaps=OVERLAPS)
    with pytest.raises(LeakageError):
        fit_all(lab, ["s2_1"], tmp_path / "out", mapper=MAPPER, overlaps=OVERLAPS, calib=[held])
    assert not (tmp_path / "out").exists()


def test_buoi_test_cua_fold_bi_chan():
    assert check_sessions(["s2_1", "s2_2"]) == ["s2_1", "s2_2"]
    with pytest.raises(LeakageError, match="s2_2"):
        check_sessions(["s2_1", "s2_2"], exclude=["s2_2"])
    with pytest.raises(ValueError, match="rỗng"):
        check_sessions([])


# --------------------------------------------------------------------------- fits


def test_fit_foot_khoi_phuc_duong_chieu_cao_va_mat_dau(tmp_path):
    body = fit_foot([_standing("cam01")], MAPPER, method="head_plane")
    cam = body["cameras"]["cam01"]
    assert cam["slope"] == pytest.approx(0.6, abs=0.01)
    assert cam["intercept"] == pytest.approx(20.0, abs=8.0)

    # A box cut by the image bottom: head at row 600 -> feet at (600 + 20) / (1 - 0.6) = 1550.
    cut = (860.0, 600.0, 200.0, 480.0)
    for method in ("head_plane", "height_line"):
        path = write_model(
            tmp_path / f"{method}.yaml", "foot_height", ["s2_1"], body | {"method": method}
        )
        x, y = FootModel.load(path).foot("cam01", cut, 1080.0)
        assert (x, y) == pytest.approx((960.0, 1550.0), abs=25.0), method


def _two_views(n: int = 200, shift_px: float = 50.0) -> Boxes:
    """One person per instant seen by cam01 and, `shift_px` further along x, by cam02."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        x = rng.uniform(110.0, 140.0)
        rows.append(("cam01", i * 40, 1, (x - 40.0, 250.0, 80.0, 200.0), 3))
        jitter = rng.normal(0.0, 2.0)  # a perfect fit would have zero spread to normalise by
        rows.append(("cam02", i * 40, 9, (x + shift_px + jitter - 40.0, 250.0, 80.0, 200.0), 3))
    return _boxes(rows)


def test_fit_ground_error_khoi_phuc_do_lech(tmp_path):
    body = fit_ground_error([_two_views()], MAPPER, OVERLAPS, None, mode="bias")
    pair = body["pairs"]["cam01|cam02"]
    assert pair["raw_median_m"] == pytest.approx(0.5, abs=0.01)
    assert pair["corrected_median_m"] < 0.05
    (cell,) = pair["cells"]
    assert (cell["ix"], cell["iy"], cell["n"]) == (1, 4, 200)
    assert cell["mu"] == pytest.approx([0.5, 0.0], abs=0.01)

    model = GroundErrorModel.load(write_model(tmp_path / "s.yaml", "ground_sigma", ["s2_1"], body))
    a, b = np.array([[1.2, 4.5]]), np.array([[1.7, 4.5]])
    assert model.distances("cam01", a, "cam02", b) == pytest.approx([0.0], abs=0.01)

    # The literal "sigma only" form learns no offset at all.
    sigma_only = fit_ground_error([_two_views()], MAPPER, OVERLAPS, None, mode="sigma")
    assert sigma_only["pairs"]["cam01|cam02"]["default"]["mu"] == [0.0, 0.0]
    with pytest.raises(ValueError, match="mode"):
        fit_ground_error([_two_views()], MAPPER, OVERLAPS, None, mode="logistic")


def _corridor_walks(people: int = 4) -> Boxes:
    """Each person walks +x through cam01 (1 m/s), then shows up in cam03 five seconds later."""
    rows = []
    for p in range(people):
        t0 = p * 60_000
        for i in range(75):  # 3 s at 25 fps
            rows.append(
                ("cam01", t0 + i * 40, p, (500.0 + 4.0 * i, 300.0 + 10 * p, 80.0, 200.0), p)
            )
        for i in range(25):
            rows.append(("cam03", t0 + 8_000 + i * 40, p, (200.0 + 4.0 * i, 300.0, 80.0, 200.0), p))
    return _boxes(rows)


def test_fit_exit_direction_khoi_phuc_huong_di(tmp_path):
    body = fit_exit_direction([_corridor_walks()], MAPPER, OVERLAPS, None)
    assert set(body["pairs"]) == {"cam01>cam03"}, "no one walked cam03 -> cam01: no gate there"
    pair = body["pairs"]["cam01>cam03"]
    assert pair["n"] == 4
    assert pair["direction"] == pytest.approx([1.0, 0.0], abs=1e-3)
    gate = ExitDirectionModel.load(
        write_model(tmp_path / "e.yaml", "exit_direction", ["s2_1"], body)
    )
    assert set(gate.directions) == {("cam01", "cam03")}
    # Fewer walks than min_n: no direction is learned from two samples.
    assert fit_exit_direction([_corridor_walks(2)], MAPPER, OVERLAPS, None)["pairs"] == {}


def _arrivals_and_fragments(people: int = 12) -> Boxes:
    """Arrive bottom-left (track A); the tracker restarts mid-frame 200 ms later (track B)."""
    rows = []
    for p in range(people):
        t0 = p * 10_000
        for i in range(25):
            rows.append(("cam01", t0 + i * 40, 2 * p, (100.0, 800.0, 80.0, 200.0), p))
            rows.append(("cam01", t0 + 1_200 + i * 40, 2 * p + 1, (900.0, 400.0, 80.0, 200.0), p))
    return _boxes(rows)


def test_fit_entry_zones_tach_nguoi_moi_den_khoi_manh_tracker(tmp_path):
    body = fit_entry_zones([_arrivals_and_fragments()])
    cam = body["cameras"]["cam01"]
    assert (cam["n"][3][0], cam["p_entry"][3][0]) == (12, 1.0)
    assert (cam["n"][2][2], cam["p_entry"][2][2]) == (12, 0.0)
    model = EntryZoneModel.load(write_model(tmp_path / "z.yaml", "entry_zones", ["s2_1"], body))
    assert model.is_entry("cam01", (100.0, 800.0, 80.0, 200.0))
    assert not model.is_entry("cam01", (900.0, 400.0, 80.0, 200.0))
    assert not model.is_entry("cam01", (1700.0, 100.0, 80.0, 100.0)), "empty cell: no evidence"


# --------------------------------------------------------------------------- orchestration


class _Lab:
    """Stands in for `LabData`: `fit_all` itself guards, before any data is asked for."""

    def __init__(self) -> None:
        self.asked: list[list[str]] = []

    def pipeline(self, sessions):
        self.asked.append(list(sessions))
        return [_standing("cam01"), _two_views(), _arrivals_and_fragments()]

    def ground_truth(self, sessions):
        self.asked.append(list(sessions))
        return [_corridor_walks()]


def test_fit_all_ghi_nguon_goc_va_chan_buoi_test(tmp_path):
    lab = _Lab()
    paths = fit_all(
        lab, ["s2_2", "s2_1"], tmp_path, mapper=MAPPER, overlaps=OVERLAPS, calib=["calib"]
    )
    assert set(paths) == {"foot", "sigma", "exit", "entry"}
    for component, path in paths.items():
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        expected = ["calib", "s2_1", "s2_2"] if component == "exit" else ["s2_1", "s2_2"]
        assert data["fit_sessions"] == expected and data["fit_date"], component

    lab = _Lab()
    with pytest.raises(LeakageError, match="s2_1"):
        fit_all(
            lab,
            ["s2_1", "s2_2"],
            tmp_path / "x",
            mapper=MAPPER,
            overlaps=OVERLAPS,
            exclude=["s2_1"],
        )
    assert lab.asked == [], "refused before a single box was loaded"
    with pytest.raises(ValueError, match="calib"):
        fit_all(lab, ["s2_1"], tmp_path / "x", mapper=MAPPER, overlaps=OVERLAPS, calib=["s2_2"])
