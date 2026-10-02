"""Test `tools/make_synthetic_lab.py` + phần đầu chuỗi M6 chạy NỐI TIẾP trên nó.

Bộ sinh chỉ có ích nếu các tệp nó sinh ra đi trót lọt qua đúng các công cụ của dữ liệu thật:
CVAT (hai kiểu đánh số khung) → `cvat_to_mot --fixture-out` → `assign_gt`. Engine và TrackEval
không chạy ở đây (nặng, và TrackEval không có trong venv test); toàn bộ chuỗi đã diễn tập tay ở
phiên 33 (`docs/worklog/2026-10-03-33-*`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from common.schema import read_jsonl
from mct.homography import HomographyMapper
from mct.topology import Topology
from tools import assign_gt, cvat_to_mot
from tools.make_synthetic_lab import VIEWS, camera_homography, main


@pytest.fixture(scope="module")
def lab(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("synth")
    args = ["--out", str(out), "--session", "t", "--runs", "1", "--seconds", "40", "--people", "3"]
    assert main(args) == 0
    return out


def test_homography_chieu_dung_vung_san(lab: Path):
    mapper = HomographyMapper.load(lab / "configs" / "homography")
    for cam, (x0, x1, y0, y1) in VIEWS.items():
        assert mapper.cameras[cam].image_size == (1920, 1080)
        # tâm cạnh gần của hình thang phối cảnh rơi đúng giữa cạnh gần của vùng sàn
        x, y = mapper.project(cam, (960.0, 1000.0))
        assert x == pytest.approx((x0 + x1) / 2, abs=0.05)
        assert y == pytest.approx(y1 if cam == "cam02" else y0, abs=0.05)
    assert np.allclose(camera_homography("cam01"), mapper.cameras["cam01"].matrix)


def test_topology_doc_duoc_va_dung_bo_tri(lab: Path):
    topo = Topology.load(lab / "configs" / "topology.yaml")
    assert topo.is_overlapping("cam01", "cam02")
    assert not topo.is_overlapping("cam02", "cam03")
    assert topo.transition("cam03", "cam04") is not None


def test_cvat_va_assign_gt_thang_hang(lab: Path):
    gt = lab / "fixtures" / "lab_t_gt.jsonl"
    cams = sorted(VIEWS)
    argv = [x for c in cams for x in ("--annotation", f"{c}={lab / 'cvat' / 't' / f'{c}.xml'}")]
    argv += ["--out-dir", str(lab / "gt"), "--fixture-out", str(gt), "--fps", "25"]
    assert cvat_to_mot.main(argv) == 0

    table = json.loads(gt.with_name("lab_t_gt.gt.json").read_text(encoding="utf-8"))
    numbering = {cam: v["numbering"] for cam, v in table["meta"]["frames"].items()}
    assert numbering == {"cam01": "abs", "cam02": "rel", "cam03": "abs", "cam04": "rel"}
    # nhảy 5 khung: mọi khung của fixture GT là bội của 5, ở mọi camera (cả kiểu tương đối)
    assert {m.frame_id % 5 for m in read_jsonl(gt)} == {0}

    report = lab / "assign.json"
    fixture = lab / "fixtures" / "lab_t_r1.jsonl"
    assert (
        assign_gt.main(
            ["--fixture", str(fixture), "--gt-fixture", str(gt), "--report", str(report)]
        )
        == 0
    )
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["gt_recall"] > 0.85  # bỏ sót 8% + rung hộp
    assert rep["match_rate"] > 0.75  # có hộp báo nhầm
    assert rep["n_identities"] == 3
