"""Chấm trọn một buổi quay tự thu (M6): n lần chạy pipeline × các biến thể cấu hình engine.

    F=data/fixtures
    PYTHONPATH=src python -m eval.run_lab_eval \\
        --config configs/lab/lab.mct.yaml --topology configs/lab/topology.yaml \\
        --homography-dir configs/lab/homography --gt-fixture $F/lab_s1_gt.jsonl \\
        --run r1 $F/lab_s1_r1.jsonl --run r2 $F/lab_s1_r2.jsonl --run r3 $F/lab_s1_r3.jsonl \\
        --variant base \\
        --variant chung030:association.max_cost_geometric=null \\
        --variant w500:association.window_ms=500 \\
        --engine-python ~/.venvs/mct-test/Scripts/python.exe \\
        --eval-python ~/.venvs/mct-eval/Scripts/python.exe --trackeval-path ~/TrackEval \\
        --work-dir data/lab/eval/s1 --fps 25

**Vì sao một công cụ riêng.** Mọi mảnh đã có: `tools.assign_gt` (bảng GT cho fixture
pipeline), `eval.compare_oracle_tracker` (engine → TrackEval, gom n lần chạy), và
`eval.eval_handover` (bàn giao theo loại cặp). Nhưng chạy tay ba công cụ × 3 lần chạy × vài
biến thể là chỗ dễ trộn nhầm cấu hình giữa các ô của bảng nhất — đúng loại lỗi phiên 17 đã gặp
("bộ chấm điểm chạy cấu hình khác engine"). Công cụ này chỉ NỐI chúng lại theo một quy ước:

- **cùng ground-truth, cùng fixture cho mọi biến thể** — thứ duy nhất khác giữa hai hàng
  của bảng là các khoá ghi trong `--variant`;
- mỗi biến thể có một file cấu hình dẫn xuất ghi ra đĩa (`<work>/<biến thể>/mct.yaml`) để
  đối chiếu được về sau;
- đơn camera (`--sct`) chỉ chấm một lần (biến thể đầu tiên): nó không phụ thuộc engine;
- chú thích nhảy khung được xử lý đúng (`--only-gt-frames` ở mọi bước chấm).

`--variant TÊN:khoá.con=giá_trị,khoá=giá_trị` — giá trị đọc bằng YAML (`null`, `0.5`,
`true`, `reject`). Không có `--variant` nào thì chạy đúng `--config` dưới tên `base`.

Đầu ra: `<work>/summary.md` (bảng dán thẳng vào worklog/báo cáo) + `<work>/summary.json`.
Số n lần chạy là số lần chạy PIPELINE (nhiễu của detector/tracker, phiên 22: ≈ 0.3–0.9
HOTA), không phải lặp lại engine — engine là tất định.

Chỉ stdlib + `common/` + các công cụ trên; engine và TrackEval chạy trong tiến trình con.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

from eval import compare_oracle_tracker, eval_handover
from tools import assign_gt

KINDS = ("overlap", "non_overlap", "same_camera")
KIND_NAMES = {"overlap": "chồng lấn", "non_overlap": "không chồng lấn", "same_camera": "quay lại"}
MCT_COLUMNS = ("HOTA", "DetA", "AssA", "IDF1", "IDs")


def parse_variant(text: str) -> tuple[str, dict[str, Any]]:
    """`"w500:association.window_ms=500,tracklet.min_frames=3"` -> (tên, {khoá: giá trị})."""
    name, _, body = text.partition(":")
    name = name.strip()
    if not name or any(c in name for c in " /\\:"):
        raise argparse.ArgumentTypeError(f"tên biến thể không hợp lệ: {name!r}")
    overrides: dict[str, Any] = {}
    for item in filter(None, (x.strip() for x in body.split(","))):
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise argparse.ArgumentTypeError(f"{text!r}: cần dạng khoá.con=giá_trị")
        overrides[key.strip()] = yaml.safe_load(raw)
    return name, overrides


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Bản sao của `config` với các khoá chấm-phân-cấp được đặt lại. Không sửa bản gốc."""
    out = copy.deepcopy(config)
    for dotted, value in overrides.items():
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if child is None:
                child = node[part] = {}
            if not isinstance(child, dict):
                raise ValueError(f"{dotted}: '{part}' không phải một khối cấu hình")
            node = child
        node[parts[-1]] = value
    return out


def _mean_sd(values: list[float]) -> tuple[float, float | None] | None:
    if not values:
        return None
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else None


def aggregate_handover(per_run: dict[str, dict]) -> dict[str, dict]:
    """{run: summary của eval_handover} -> {loại cặp: {n, accuracy (mean, sd), ...}}.

    Độ chính xác lấy trung bình THEO LẦN CHẠY (mỗi lần chạy pipeline là một mẫu của nhiễu),
    bỏ lần chạy không có lần chuyển nào của loại đó thay vì tính nó là 0.
    """
    out: dict[str, dict] = {}
    for kind in (*KINDS, "total"):
        rows = [(s["total"] if kind == "total" else s["by_kind"][kind]) for s in per_run.values()]
        acc = [r["accuracy"] for r in rows if r.get("accuracy") is not None]
        e2e = [r["end_to_end"] for r in rows if r.get("end_to_end") is not None]
        out[kind] = {
            "n_runs": len(rows),
            "n_handovers": [r["n"] for r in rows],
            "accuracy": _mean_sd(acc),
            "end_to_end": _mean_sd(e2e),
            **{k: sum(r[k] for r in rows) for k in eval_handover.OUTCOMES},
        }
    return out


def _fmt(pair: tuple[float, float | None] | None, *, pct: bool = False, digits: int = 2) -> str:
    if pair is None:
        return "—"
    mean, sd = pair
    scale = 100.0 if pct else 1.0
    text = f"{mean * scale:.{1 if pct else digits}f}"
    if sd is not None:
        text += f" ± {sd * scale:.{1 if pct else digits}f}"
    return text + ("%" if pct else "")


def format_summary(variants: dict[str, dict]) -> str:
    """Hai bảng: TrackEval xuyên camera, rồi bàn giao theo loại cặp."""
    lines = ["## Xuyên camera (hộp ảnh, chỉ khung đã chú thích, TrackEval)", ""]
    lines.append("| biến thể | n | " + " | ".join(MCT_COLUMNS) + " |")
    lines.append("|---|---|" + "---|" * len(MCT_COLUMNS))
    for name, v in variants.items():
        agg = v["mct"]
        cells = [
            _fmt(tuple(agg[c]) if agg.get(c) else None, digits=0 if c == "IDs" else 2)  # type: ignore[arg-type]
            for c in MCT_COLUMNS
        ]
        lines.append(f"| {name} | {agg.get('n', 0)} | " + " | ".join(cells) + " |")
    lines += ["", "## Bàn giao danh tính theo loại cặp (eval.eval_handover)", ""]
    header = " | ".join(f"{KIND_NAMES[k]}" for k in KINDS)
    lines.append(f"| biến thể | {header} | tổng |")
    lines.append("|---|" + "---|" * (len(KINDS) + 1))
    for name, v in variants.items():
        h = v["handover"]
        cells = [
            f"{_fmt(h[k]['accuracy'], pct=True)} (n={sum(h[k]['n_handovers'])})"
            for k in (*KINDS, "total")
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Độ chính xác bàn giao = đúng / (đúng + tách + nhầm người), trung bình theo lần chạy "
        "pipeline; n = tổng số lần chuyển qua mọi lần chạy. Lần chuyển mà một đầu không có dự "
        "đoán (sót) không tính vào mẫu số — xem cột `sot` trong summary.json.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--topology", type=Path, required=True)
    p.add_argument("--homography-dir", type=Path, required=True)
    p.add_argument("--gt-fixture", type=Path, required=True)
    p.add_argument("--gt-fixture-table", type=Path, default=None)
    p.add_argument("--run", nargs=2, action="append", required=True, metavar=("TÊN", "FIXTURE"))
    p.add_argument("--variant", type=parse_variant, action="append", default=[])
    p.add_argument("--engine-python", default=sys.executable)
    p.add_argument("--eval-python", default=sys.executable)
    p.add_argument("--trackeval-path", type=Path, default=None)
    p.add_argument("--work-dir", type=Path, default=Path("data/lab/eval"))
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--min-iou", type=float, default=0.5)
    p.add_argument("--force", action="store_true", help="chạy lại engine dù đã có DB")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    work: Path = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    gt_table = args.gt_fixture_table or Path(str(args.gt_fixture).replace(".jsonl", ".gt.json"))
    runs = [(name, Path(fx)) for name, fx in args.run]
    variants = args.variant or [("base", {})]
    if len({n for n, _ in variants}) != len(variants):
        p.error("trùng tên biến thể")

    # 1. Bảng GT cho từng lần chạy pipeline — một lần, dùng chung cho mọi biến thể.
    tables: dict[str, Path] = {}
    for name, fixture in runs:
        table = work / "tables" / f"{name}.gt.json"
        table.parent.mkdir(parents=True, exist_ok=True)
        if args.force or not table.is_file():
            assign_gt.main(
                [
                    "--fixture",
                    str(fixture),
                    "--gt-fixture",
                    str(args.gt_fixture),
                    "--gt-fixture-table",
                    str(gt_table),
                    "--out",
                    str(table),
                    "--report",
                    str(table.with_suffix(".report.json")),
                    "--min-iou",
                    str(args.min_iou),
                ]
            )
        tables[name] = table

    base_config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    results: dict[str, dict] = {}
    for index, (variant, overrides) in enumerate(variants):
        vdir = work / variant
        vdir.mkdir(parents=True, exist_ok=True)
        cfg_path = vdir / "mct.yaml"
        cfg_path.write_text(
            f"# Dẫn xuất từ {args.config} với {overrides or 'không đổi gì'}\n"
            + yaml.safe_dump(apply_overrides(base_config, overrides), allow_unicode=True),
            encoding="utf-8",
        )
        # 2. Engine -> TrackEval cho mọi lần chạy của biến thể này.
        cmd = [
            "--config",
            str(cfg_path),
            "--topology",
            str(args.topology),
            "--homography-dir",
            str(args.homography_dir),
            "--gt-fixture",
            str(args.gt_fixture),
            "--gt-fixture-table",
            str(gt_table),
            "--engine-python",
            str(args.engine_python),
            "--eval-python",
            str(args.eval_python),
            "--work-dir",
            str(vdir),
            "--fps",
            str(args.fps),
            "--only-gt-frames",
        ]
        if args.trackeval_path:
            cmd += ["--trackeval-path", str(args.trackeval_path)]
        if index == 0:
            cmd.append("--sct")
        if args.force:
            cmd.append("--force")
        for name, fixture in runs:
            cmd += ["--run", f"{variant}:{name}", str(fixture), str(tables[name])]
        compare_oracle_tracker.main(cmd)
        scored = json.loads((vdir / "compare_oracle_tracker.json").read_text(encoding="utf-8"))
        mct = compare_oracle_tracker.aggregate(scored["mct"], MCT_COLUMNS).get(variant, {})

        # 3. Bàn giao theo loại cặp, dùng đúng SQLite mà bước 2 vừa sinh.
        handover: dict[str, dict] = {}
        for name, fixture in runs:
            run_dir = vdir / f"{variant}_{name}"
            out = run_dir / "handover.json"
            eval_handover.main(
                [
                    "--fixture",
                    str(fixture),
                    "--db",
                    str(run_dir / "mct.db"),
                    "--gt-fixture",
                    str(args.gt_fixture),
                    "--gt-fixture-table",
                    str(gt_table),
                    "--topology",
                    str(args.topology),
                    "--min-iou",
                    str(args.min_iou),
                    "--json",
                    str(out),
                ]
            )
            handover[name] = json.loads(out.read_text(encoding="utf-8"))
        results[variant] = {
            "overrides": overrides,
            "mct": mct,
            "sct": compare_oracle_tracker.aggregate(
                scored.get("sct") or {}, compare_oracle_tracker.SCT_COLUMNS
            ).get(variant),
            "handover": aggregate_handover(handover),
        }

    summary = format_summary(results)
    (work / "summary.md").write_text(summary + "\n", encoding="utf-8")
    (work / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=list) + "\n", encoding="utf-8"
    )
    print("\n" + summary)
    print(f"\n-> {work / 'summary.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
