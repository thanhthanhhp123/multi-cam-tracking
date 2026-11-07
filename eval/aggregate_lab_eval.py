"""Aggregate `eval.run_lab_eval` summaries across lab sessions into report tables.

`run_lab_eval` writes one `summary.json` per session (`<root>/<session>/<kind>/summary.json`).
This tool pools them the way chapter 6 reports them:

- **HOTA / AssA / IDF1**: mean over sessions of each session's mean over pipeline runs.
- **Handover accuracy**: counts (correct / split / wrong person) pooled over sessions and runs,
  accuracy = correct / (correct + split + wrong), with a Wilson 95% interval on the pooled count.
  Handovers whose one end has no prediction (`sot`) stay out of the denominator, as in
  `eval.eval_handover`.
- **SCT** (single-camera NvDCF, `sct` block): identical for every variant because it does not go
  through `src/mct`; `run_lab_eval` stores it on the first variant only, so it is taken from
  whichever variant carries it.

Groups: the selection/held-out split used since session 40 (s2 sessions select, s1/s3/s4 check)
plus one group per recording scenario (s1..s4, docs/m6/README.md section 3).

    PYTHONPATH="src;." python -m eval.aggregate_lab_eval data/lab/oracle/eval_v4 \\
        --variant combo --variant c0 --out data/s42/per_scenario.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SELECT = ["s2_1", "s2_1b", "s2_1c", "s2_2", "s2_2b", "s2_2c", "s2_3", "s2_3b"]
HELD_OUT = ["s1_1", "s3_1", "s3_2", "s4_1"]
ALL_SESSIONS = SELECT + HELD_OUT
SPLIT_GROUPS: dict[str, list[str]] = {
    "s2 (chọn)": SELECT,
    "giữ riêng": HELD_OUT,
    "cả 12": ALL_SESSIONS,
}
HANDOVER_TYPES: list[tuple[str, str]] = [
    ("overlap", "chồng lấn"),
    ("non_overlap", "không chồng lấn"),
    ("same_camera", "quay lại"),
]
OUTCOMES = ("dung", "tach", "nham")


def scenario_of(session: str) -> str:
    """`s2_1b` -> `s2`: the recording scenario a session belongs to."""
    return session.split("_", 1)[0]


def scenario_groups(sessions: list[str]) -> dict[str, list[str]]:
    """One group per scenario, in s1..s4 order, keeping only scenarios that have sessions."""
    groups: dict[str, list[str]] = {}
    for session in sorted(sessions, key=lambda s: (scenario_of(s), s)):
        groups.setdefault(scenario_of(session), []).append(session)
    return groups


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n (NaN bounds when n == 0)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


@dataclass(frozen=True)
class Counts:
    dung: int = 0
    tach: int = 0
    nham: int = 0

    @property
    def n(self) -> int:
        return self.dung + self.tach + self.nham

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.dung + other.dung, self.tach + other.tach, self.nham + other.nham)


def load_summaries(root: Path, kind: str, sessions: list[str]) -> dict[str, dict[str, Any]]:
    """session -> parsed summary.json, for the sessions that have one."""
    out: dict[str, dict[str, Any]] = {}
    for session in sessions:
        path = root / session / kind / "summary.json"
        if path.exists():
            out[session] = json.loads(path.read_text(encoding="utf-8"))
    return out


def pooled_counts(
    data: dict[str, dict[str, Any]], sessions: list[str], variant: str, key: str
) -> Counts:
    total = Counts()
    for session in sessions:
        block = data[session][variant]["handover"][key]
        total = total + Counts(*(int(block[o]) for o in OUTCOMES))
    return total


def mean_metric(
    data: dict[str, dict[str, Any]], sessions: list[str], variant: str, block: str, metric: str
) -> float | None:
    values = [
        float(data[s][variant][block][metric][0])
        for s in sessions
        if data[s].get(variant, {}).get(block)
    ]
    return sum(values) / len(values) if values else None


def sct_block(summary: dict[str, Any]) -> dict[str, Any] | None:
    """The single-camera block of a session, from whichever variant carries it."""
    for result in summary.values():
        if result.get("sct"):
            return result["sct"]
    return None


def _pct(counts: Counts) -> str:
    return f"{100 * counts.dung / counts.n:.0f}% ({counts.dung}/{counts.n})" if counts.n else "–"


def _num(value: float | None) -> str:
    return "–" if value is None else f"{value:.1f}"


def group_table(
    data: dict[str, dict[str, Any]], name: str, sessions: list[str], variants: list[str]
) -> list[str]:
    sessions = [s for s in sessions if s in data]
    lines = [f"### {name} ({len(sessions)} buổi: {', '.join(sessions)})", ""]
    if not sessions:
        return [*lines, "(không có kết quả)", ""]
    header = ["biến thể", "HOTA", "AssA", "IDF1"] + [label for _, label in HANDOVER_TYPES]
    header += ["tổng [Wilson 95%]", "đúng/tách/nhầm"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for variant in variants:
        if not all(variant in data[s] for s in sessions):
            continue
        cells = [variant]
        cells += [
            _num(mean_metric(data, sessions, variant, "mct", m)) for m in ("HOTA", "AssA", "IDF1")
        ]
        cells += [_pct(pooled_counts(data, sessions, variant, key)) for key, _ in HANDOVER_TYPES]
        total = pooled_counts(data, sessions, variant, "total")
        lo, hi = wilson(total.dung, total.n)
        cells.append(
            f"{100 * total.dung / total.n:.0f}% [{100 * lo:.0f}–{100 * hi:.0f}]" if total.n else "–"
        )
        cells.append(f"{total.dung}/{total.tach}/{total.nham}")
        lines.append("| " + " | ".join(cells) + " |")
    return [*lines, ""]


def sct_table(data: dict[str, dict[str, Any]], groups: dict[str, list[str]]) -> list[str]:
    rows = []
    for name, sessions in groups.items():
        blocks = [b for s in sessions if s in data and (b := sct_block(data[s])) is not None]
        if not blocks:
            continue

        def mean(metric: str, blocks: list[dict[str, Any]] = blocks) -> str:
            return f"{sum(float(b[metric][0]) for b in blocks) / len(blocks):.1f}"

        rows.append(
            f"| {name} | {len(blocks)} | {mean('HOTA')} | {mean('AssA')} | {mean('IDF1')} |"
        )
    if not rows:
        return []
    return [
        "### Đơn camera (NvDCF, không qua src/mct)",
        "",
        "| nhóm | buổi | HOTA | AssA | IDF1 |",
        "|---|---|---|---|---|",
        *rows,
        "",
    ]


def render(
    data: dict[str, dict[str, Any]], variants: list[str] | None = None, root: Path | None = None
) -> str:
    if not data:
        return "(không có summary.json nào)\n"
    if not variants:
        variants = list(next(iter(data.values())))
    missing = [s for s in ALL_SESSIONS if s not in data]
    lines = [
        f"# Tổng hợp lab-eval{f' — {root}' if root else ''}",
        "",
        f"Buổi có kết quả: {len(data)}/{len(ALL_SESSIONS)}"
        + (f" (thiếu: {', '.join(missing)})" if missing else ""),
        "",
        "HOTA/AssA/IDF1 = trung bình theo buổi (mỗi buổi đã là trung bình theo lần chạy). "
        "Bàn giao = đúng / (đúng + tách + nhầm), gộp số lần qua mọi buổi và lần chạy.",
        "",
        "## Theo nhóm chọn / giữ riêng",
        "",
    ]
    for name, sessions in SPLIT_GROUPS.items():
        lines += group_table(data, name, sessions, variants)
    lines += ["## Theo kịch bản", ""]
    scenarios = scenario_groups(list(data))
    for name, sessions in scenarios.items():
        lines += group_table(data, name, sessions, variants)
    lines += sct_table(data, scenarios)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("root", type=Path, help="thư mục chứa <buổi>/<kind>/summary.json")
    p.add_argument("--kind", default="pipeline", help="thư mục con: pipeline | oracle")
    p.add_argument("--variant", action="append", default=None, help="chỉ in các biến thể này")
    p.add_argument("--out", type=Path, default=None, help="ghi thêm markdown ra file")
    args = p.parse_args(argv)
    data = load_summaries(args.root, args.kind, ALL_SESSIONS)
    text = render(data, args.variant, args.root)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
