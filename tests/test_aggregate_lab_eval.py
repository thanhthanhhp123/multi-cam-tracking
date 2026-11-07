from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval import aggregate_lab_eval as agg


def _handover(dung: int, tach: int, nham: int) -> dict:
    return {"dung": dung, "tach": tach, "nham": nham, "sot": 0}


def _summary(hota: float, counts: dict[str, tuple[int, int, int]], sct: bool = False) -> dict:
    total = tuple(sum(c[i] for c in counts.values()) for i in range(3))
    result = {
        "mct": {"n": 3, "HOTA": [hota, 0.0], "AssA": [hota - 10, 0.0], "IDF1": [hota + 5, 0.0]},
        "sct": (
            {"n": 3, "HOTA": [50.0, 0.0], "AssA": [40.0, 0.0], "IDF1": [60.0, 0.0]} if sct else None
        ),
        "handover": {k: _handover(*v) for k, v in {**counts, "total": total}.items()},
    }
    return result


def _write(root: Path, session: str, variants: dict[str, dict]) -> None:
    path = root / session / "pipeline" / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(variants), encoding="utf-8")


def test_wilson_khop_gia_tri_tra_cuu():
    lo, hi = agg.wilson(216, 489)
    assert lo == pytest.approx(0.398, abs=1e-3)
    assert hi == pytest.approx(0.486, abs=1e-3)
    assert all(v != v for v in agg.wilson(0, 0))  # NaN


def test_scenario_groups_theo_tien_to():
    groups = agg.scenario_groups(["s3_2", "s2_1b", "s1_1", "s2_1", "s4_1", "s3_1"])
    assert list(groups) == ["s1", "s2", "s3", "s4"]
    assert groups["s2"] == ["s2_1", "s2_1b"]
    assert groups["s3"] == ["s3_1", "s3_2"]


def test_gop_dem_ban_giao_va_trung_binh_theo_buoi(tmp_path):
    counts_a = {"overlap": (3, 1, 0), "non_overlap": (1, 1, 0), "same_camera": (1, 0, 0)}
    counts_b = {"overlap": (1, 0, 1), "non_overlap": (0, 2, 0), "same_camera": (2, 0, 0)}
    _write(
        tmp_path,
        "s2_1",
        {"base": _summary(60.0, counts_a, sct=True), "c0": _summary(70.0, counts_a)},
    )
    _write(
        tmp_path,
        "s2_2",
        {"base": _summary(50.0, counts_b, sct=True), "c0": _summary(80.0, counts_b)},
    )
    data = agg.load_summaries(tmp_path, "pipeline", agg.ALL_SESSIONS)

    assert set(data) == {"s2_1", "s2_2"}
    pooled = agg.pooled_counts(data, ["s2_1", "s2_2"], "c0", "overlap")
    assert (pooled.dung, pooled.tach, pooled.nham, pooled.n) == (4, 1, 1, 6)
    assert agg.mean_metric(data, ["s2_1", "s2_2"], "c0", "mct", "HOTA") == pytest.approx(75.0)
    # SCT only on the first variant: still found.
    assert agg.mean_metric(data, ["s2_1"], "c0", "sct", "HOTA") is None
    assert agg.sct_block(data["s2_1"]) is not None

    text = agg.render(data, ["base", "c0"])
    assert "### s2 (2 buổi: s2_1, s2_2)" in text
    # total over both sessions for c0: dung 3+1+1 + 1+0+2 = 8 of 13
    assert "62% [" in text and "8/4/1" in text
    assert "Đơn camera" in text and "| s2 | 2 | 50.0 | 40.0 | 60.0 |" in text


def test_main_ghi_file(tmp_path, capsys):
    counts = {"overlap": (1, 0, 0), "non_overlap": (1, 0, 0), "same_camera": (1, 0, 0)}
    _write(tmp_path, "s1_1", {"c0": _summary(55.0, counts)})
    out = tmp_path / "out" / "table.md"

    assert agg.main([str(tmp_path), "--out", str(out)]) == 0
    assert "### s1 (1 buổi: s1_1)" in out.read_text(encoding="utf-8")
    assert "thiếu:" in capsys.readouterr().out
