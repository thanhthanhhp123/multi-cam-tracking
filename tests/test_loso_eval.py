"""`eval.loso_eval`: parsing, the selection rule, pooling, the cache key, and the fold protocol."""

from __future__ import annotations

import argparse

import pytest
import yaml
from eval.lab_geometry import LeakageError
from eval.loso_eval import (
    Candidate,
    Step,
    assert_models_clean,
    config_hash,
    evaluate,
    format_table,
    parse_candidate,
    parse_step,
    pool,
    select,
)


def test_parse_step_va_candidate():
    assert parse_step("base=") == ("base", ())
    assert parse_step("+1+2=foot, sigma") == ("+1+2", ("foot", "sigma"))
    for bad in ("base", "=foot", "+1=feet"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_step(bad)

    step, cand = parse_candidate("+1:head:fit.foot_method=head_plane,association.max_cost=0.4")
    assert (step, cand.name) == ("+1", "head")
    assert cand.fit == {"foot_method": "head_plane"}
    assert cand.overrides == {"association.max_cost": 0.4}
    assert parse_candidate("+1:plain:")[1] == Candidate("plain")
    for bad in ("+1", "+1:x:novalue"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_candidate(bad)


def test_select_theo_quy_tac_phien_43():
    scores = {
        "a": {"hota": 70.0, "ttid_p95": 900.0},
        "b": {"hota": 75.0, "ttid_p95": 1200.0},  # best HOTA, but too slow: not eligible
        "c": {"hota": 72.0, "ttid_p95": 999.0},
        "d": {"hota": 72.0, "ttid_p95": 500.0},  # ties with c: the earlier one wins
    }
    assert select(scores) == "c"
    with pytest.raises(ValueError, match="p95"):
        select({"b": scores["b"]})


def _score(hota: float, counts: tuple[int, int, int], ttid: list[float]) -> dict:
    block = dict(zip(("dung", "tach", "nham"), counts, strict=True))
    zero = dict.fromkeys(("dung", "tach", "nham"), 0)
    return {
        "mct": {"HOTA": hota, "AssA": hota - 5, "IDF1": hota + 5},
        "handover": {"overlap": block, "non_overlap": zero, "same_camera": zero, "total": block},
        "ttid": ttid,
    }


def test_pool_gop_ban_giao_va_khoang_wilson():
    row = pool(
        {"s2_1": _score(70.0, (8, 1, 1), [400.0, 600.0]), "s2_2": _score(60.0, (2, 6, 2), [800.0])}
    )
    assert row["HOTA"] == 65.0
    assert row["total"] | {"wilson": None} == {
        "dung": 10,
        "tach": 7,
        "nham": 3,
        "n": 20,
        "wilson": None,
    }
    lo, hi = row["total"]["wilson"]
    assert lo < 0.5 < hi and hi - lo > 0.3, "20 handovers: the interval is wide, and reported"
    assert row["non_overlap"]["n"] == 0
    assert (row["ttid_p50"], row["ttid_n"]) == (600.0, 3)
    assert "50% [" in format_table({"base": row}, {"base": {"s2_1": "default", "s2_2": "default"}})


def _model(path, sessions, extra="a: 1", date="2026-11-13"):
    path.write_text(f"fit_sessions: {sessions}\nfit_date: '{date}'\n{extra}\n", encoding="utf-8")
    return path


def test_config_hash_theo_noi_dung_khong_theo_duong_dan(tmp_path):
    config = {"association": {"max_cost": 0.4, "foot_from_height": None}}
    first = config_hash(config, {"foot": _model(tmp_path / "a.yaml", ["s2_1"])})
    moved = config_hash(config, {"foot": _model(tmp_path / "b.yaml", ["s2_1"], date="2026-12-01")})
    refit = config_hash(config, {"foot": _model(tmp_path / "c.yaml", ["s2_1"], extra="a: 2")})
    other_fold = config_hash(config, {"foot": _model(tmp_path / "d.yaml", ["s2_2"])})
    assert first == moved, "same model at another path, fitted another day: same run"
    assert len({first, refit, other_fold, config_hash(config, {})}) == 4
    assert config["association"]["foot_from_height"] is None, "the input config is not mutated"


def test_assert_models_clean(tmp_path):
    ok = {"foot": _model(tmp_path / "ok.yaml", ["s2_1", "s2_2"])}
    assert_models_clean(ok, "s2_3")
    assert_models_clean(ok, None)
    with pytest.raises(LeakageError, match="s2_2"):
        assert_models_clean(ok, "s2_2")
    with pytest.raises(LeakageError, match="s3_1"):
        assert_models_clean({"foot": _model(tmp_path / "h.yaml", ["s2_1", "s3_1"])}, None)
    bare = tmp_path / "bare.yaml"
    bare.write_text(yaml.safe_dump({"kind": "foot_height"}), encoding="utf-8")
    with pytest.raises(LeakageError, match="fit_sessions"):
        assert_models_clean({"foot": bare}, None)


class _Runner:
    """Scores from a table instead of the engine; records what each fit was allowed to see."""

    def __init__(self, hota: dict[str, dict[str, float]], ttid: dict[str, float]) -> None:
        self.hota, self.ttid = hota, ttid
        self.fits: list[tuple[str, tuple[str, ...], str | None]] = []

    def learned(self, step, candidate, train, test):
        self.fits.append((candidate.name, tuple(train), test))
        return {}

    def config_for(self, candidate, learned):
        return candidate.name, {}

    def score(self, jobs):
        self.jobs = jobs

    def get(self, key, session):
        return _score(self.hota[key][session], (1, 0, 0), [self.ttid[key]])


def test_evaluate_chon_ung_vien_khong_nhin_buoi_test():
    sessions = ["s2_1", "s2_2", "s2_3"]
    hota = {
        "steady": dict.fromkeys(sessions, 70.0),
        "lucky": {"s2_1": 100.0, "s2_2": 60.0, "s2_3": 60.0},  # brilliant on one session only
        "slow": dict.fromkeys(sessions, 99.0),
    }
    runner = _Runner(hota, {"steady": 600.0, "lucky": 600.0, "slow": 1500.0})
    step = Step("+1", ("foot",), tuple(Candidate(n) for n in hota))
    out = evaluate(runner, [step], sessions, final=True)

    # Fold s2_1 cannot see that `lucky` shines there (train mean 60 < 70) and takes `steady`;
    # the other two folds are fooled by s2_1 and pay for it on their own test session.
    assert out["chosen"]["+1"] == {"s2_1": "steady", "s2_2": "lucky", "s2_3": "lucky"}
    assert out["rows"]["+1"]["HOTA"] == pytest.approx((70.0 + 60.0 + 60.0) / 3)
    assert out["rows"]["+1"]["total"]["n"] == 3
    assert out["final"]["+1"]["candidate"] == "lucky", "final pick: all sessions, `slow` never"
    for _, train, test in runner.fits:
        assert test not in train
        assert len(train) == (3 if test is None else 2)
