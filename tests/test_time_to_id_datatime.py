from __future__ import annotations

from pathlib import Path

import pytest
from eval import time_to_id_datatime as tt

from mct.__main__ import load_config


def test_summarize_va_dong_bang():
    stats = tt.summarize([100.0, 200.0, 300.0, 4000.0])
    assert stats["n"] == 4
    assert stats["p50"] == pytest.approx(250.0)
    assert stats["under_1s"] == pytest.approx(0.75)
    assert tt.summarize([]) == {"n": 0}
    assert tt.format_row("on", tt.summarize([])).startswith("| on | 0 |")
    assert "75.0%" in tt.format_row("on", stats)


def test_phat_lai_fixture_that():
    values = tt.collect_time_to_id(
        load_config(Path("configs/mct.yaml")),
        [Path("tests/fixtures/two_cam_walk.jsonl")],
        topology=Path("configs/cameras/topology.yaml"),
        homography_dir=None,
    )
    assert values and all(v >= 0 for v in values)
