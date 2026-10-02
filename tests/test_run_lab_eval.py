"""Test `eval/run_lab_eval.py` — phần NỐI các công cụ chấm cho dữ liệu tự thu.

TrackEval không có trong venv test, nên ba công cụ bên dưới được thay bằng bản giả ghi ra
đúng tệp mà công cụ thật ghi. Thứ được kiểm là chỗ hay sai nhất khi chạy tay: mỗi biến thể
chạy đúng cấu hình dẫn xuất của nó, mọi biến thể dùng chung bảng GT, đơn camera chỉ chấm một
lần, và cờ `--only-gt-frames` luôn được bật.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml
from eval import run_lab_eval
from eval.run_lab_eval import aggregate_handover, apply_overrides, format_summary, parse_variant


def test_doc_bien_the():
    name, ov = parse_variant("w500:association.window_ms=500,association.max_cost_geometric=null")
    assert name == "w500"
    assert ov == {"association.window_ms": 500, "association.max_cost_geometric": None}
    assert parse_variant("base") == ("base", {})
    with pytest.raises(argparse.ArgumentTypeError):
        parse_variant("x:khong_co_dau_bang")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_variant("ten co cach:a=1")


def test_dat_khoa_khong_sua_ban_goc():
    base = {"association": {"window_ms": 1000, "max_cost": 0.3}}
    out = apply_overrides(base, {"association.window_ms": 500, "publish.position_interval_ms": 0})
    assert out["association"] == {"window_ms": 500, "max_cost": 0.3}
    assert out["publish"] == {"position_interval_ms": 0}
    assert base["association"]["window_ms"] == 1000
    with pytest.raises(ValueError):
        apply_overrides({"a": 1}, {"a.b": 2})


def _handover(acc_overlap, acc_non, n=4):
    def kind(acc):
        if acc is None:
            return {
                "n": 0,
                "dung": 0,
                "tach": 0,
                "nham": 0,
                "sot": 0,
                "accuracy": None,
                "end_to_end": None,
            }
        return {
            "n": n,
            "dung": round(acc * n),
            "tach": n - round(acc * n),
            "nham": 0,
            "sot": 0,
            "accuracy": acc,
            "end_to_end": acc,
        }

    return {
        "total": kind(acc_overlap),
        "by_kind": {
            "overlap": kind(acc_overlap),
            "non_overlap": kind(acc_non),
            "same_camera": kind(None),
        },
    }


def test_gop_ban_giao_theo_lan_chay():
    agg = aggregate_handover({"r1": _handover(1.0, 0.5), "r2": _handover(0.5, None)})
    assert agg["overlap"]["accuracy"] == (0.75, pytest.approx(0.3535, abs=1e-3))
    # lần chạy không có lần chuyển không chồng lấn nào thì bỏ, không tính là 0
    assert agg["non_overlap"]["accuracy"] == (0.5, None)
    assert agg["same_camera"]["accuracy"] is None
    text = format_summary(
        {"base": {"mct": {"n": 2, "HOTA": (40.0, 1.0), "IDs": (12.0, 2.0)}, "handover": agg}}
    )
    assert "75.0 ± 35.4%" in text
    assert "| base | 2 | 40.00 ± 1.00" in text


def test_noi_cac_cong_cu(tmp_path: Path, monkeypatch, capsys):
    calls: dict[str, list] = {"assign": [], "compare": [], "handover": []}

    def fake_assign(argv):
        calls["assign"].append(argv)
        out = Path(argv[argv.index("--out") + 1])
        out.write_text(json.dumps({"tracklets": []}), encoding="utf-8")
        return 0

    def fake_compare(argv):
        calls["compare"].append(argv)
        work = Path(argv[argv.index("--work-dir") + 1])
        labels = [argv[i + 1] for i, a in enumerate(argv) if a == "--run"]
        mct = {
            label: {
                c: 1.0
                for c in ("HOTA", "DetA", "AssA", "AssRe", "AssPr", "IDF1", "IDs", "Dets", "IDSW")
            }
            for label in labels
        }
        sct = mct if "--sct" in argv else {}
        (work / "compare_oracle_tracker.json").write_text(
            json.dumps({"mct": mct, "sct": sct}), encoding="utf-8"
        )
        return 0

    def fake_handover(argv):
        calls["handover"].append(argv)
        out = Path(argv[argv.index("--json") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_handover(1.0, 0.5)), encoding="utf-8")
        return 0

    monkeypatch.setattr(run_lab_eval.assign_gt, "main", fake_assign)
    monkeypatch.setattr(run_lab_eval.compare_oracle_tracker, "main", fake_compare)
    monkeypatch.setattr(run_lab_eval.eval_handover, "main", fake_handover)

    config = tmp_path / "lab.mct.yaml"
    config.write_text(yaml.safe_dump({"association": {"window_ms": 1000}}), encoding="utf-8")
    work = tmp_path / "work"
    argv = [
        "--config", str(config), "--topology", "t.yaml", "--homography-dir", "h",
        "--gt-fixture", "gt.jsonl",
        "--run", "r1", "fx1.jsonl", "--run", "r2", "fx2.jsonl",
        "--variant", "base", "--variant", "w500:association.window_ms=500",
        "--work-dir", str(work),
    ]  # fmt: skip
    assert run_lab_eval.main(argv) == 0

    assert len(calls["assign"]) == 2  # một bảng GT mỗi lần chạy, dùng chung cho mọi biến thể
    assert len(calls["compare"]) == 2  # một lần mỗi biến thể
    assert ["--sct" in c for c in calls["compare"]] == [True, False]
    assert all("--only-gt-frames" in c for c in calls["compare"])
    w500 = yaml.safe_load(
        (work / "w500" / "mct.yaml").read_text(encoding="utf-8").split("\n", 1)[1]
    )
    assert w500["association"]["window_ms"] == 500
    second = calls["compare"][1]
    assert second[second.index("--config") + 1] == str(work / "w500" / "mct.yaml")
    assert len(calls["handover"]) == 4
    dbs = {c[c.index("--db") + 1] for c in calls["handover"]}
    assert str(work / "w500" / "w500_r2" / "mct.db") in dbs
    assert (work / "summary.md").is_file()
    assert "w500" in capsys.readouterr().out
