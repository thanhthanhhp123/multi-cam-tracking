"""Leave-one-session-out evaluation of the learned association components (session 45).

    PYTHONPATH=src:. python -m eval.loso_eval \\
        --fixtures-dir data/lab/gpu_s42/data/fixtures --gt-dir data/fixtures \\
        --config configs/lab/lab.mct.yaml --topology configs/lab/topology.yaml \\
        --homography-dir configs/lab/homography --calib calib calib_2 \\
        --step base= --step +1=foot --step +1+2=foot,sigma \\
        --candidate +1:line:fit.foot_method=height_line \\
        --candidate +1:head:fit.foot_method=head_plane \\
        --eval-python ~/.venvs/mct-eval/bin/python --trackeval-path ~/TrackEval \\
        --work-dir data/lab/s45/loso --jobs 3

**Protocol.** The sessions (default: the 8 selection sessions, never the held-out group) are
folded one at a time. In each fold, with test session `t`:

1. every learned component of the step is REFITTED on the other sessions (+ calibration
   sessions for walking directions) by `eval.fit_learned.fit_all(exclude=[t])`;
2. every candidate of the step (fit options and/or config overrides) is scored on the training
   sessions and one is chosen by the rule used since session 43: highest mean HOTA among the
   candidates whose time to Global ID (data time, p95 over the training sessions) is under 1 s;
3. the chosen candidate is scored on `t`.

Each session is a test session exactly once, so handover counts are pooled over the folds
(Wilson 95% interval) and HOTA / AssA / IDF1 are averaged over them. `--step NAME=` (no
component) is a fixed configuration: the baseline row.

`--final-dir DIR` additionally fits on ALL sessions, applies the same selection rule on all of
them, and writes the chosen learned files there: the configuration to freeze before the one
look at the held-out group. That look is NOT done by this tool.

**Leakage guards** (all raise): the session list is checked against the held-out group before
anything is read; every fit gets the fold's test session as `exclude`; and every learned file is
re-read after fitting and its `fit_sessions` checked against the test session and the held-out
group before any engine run may use it.

Scoring itself is `eval.run_lab_eval` (one subprocess per configuration and session) and
`eval.time_to_id_datatime`; results are cached by a hash of the configuration INCLUDING the
content of the learned files, so identical configurations are never run twice.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from eval import roi_filter_fixture
from eval.aggregate_lab_eval import HANDOVER_TYPES, OUTCOMES, SELECT, Counts, wilson
from eval.fit_learned import CONFIG_KEYS, fit_all
from eval.lab_geometry import RUNS, LabData, LeakageError, check_sessions
from eval.run_lab_eval import apply_overrides
from mct.homography import HomographyMapper
from tools.estimate_transit import load_overlaps

TTID_LIMIT_MS = 1000.0
ALL = "ALL"
"""Pseudo-fold: fit on every session, no test session (`--final-dir`)."""


@dataclass(frozen=True)
class Candidate:
    """One way to configure a step: fit options (`fit.*`) plus engine overrides."""

    name: str
    fit: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Step:
    name: str
    components: tuple[str, ...]
    candidates: tuple[Candidate, ...] = (Candidate("default"),)


def parse_step(text: str) -> tuple[str, tuple[str, ...]]:
    """`"+1+2=foot,sigma"` -> ("+1+2", ("foot", "sigma")); `"base="` -> ("base", ())."""
    name, sep, body = text.partition("=")
    components = tuple(c.strip() for c in body.split(",") if c.strip())
    unknown = sorted(set(components) - set(CONFIG_KEYS))
    if not sep or not name.strip() or unknown:
        raise argparse.ArgumentTypeError(
            f"{text!r}: cần TÊN=thành_phần,... với thành phần thuộc {sorted(CONFIG_KEYS)}"
        )
    return name.strip(), components


def parse_candidate(text: str) -> tuple[str, Candidate]:
    """`"+1:head:fit.foot_method=head_plane,association.max_cost=0.4"` -> (step, Candidate)."""
    step, _, rest = text.partition(":")
    name, _, body = rest.partition(":")
    if not step or not name:
        raise argparse.ArgumentTypeError(f"{text!r}: cần BƯỚC:TÊN:khoá=giá_trị,...")
    fit: dict[str, Any] = {}
    overrides: dict[str, Any] = {}
    for item in filter(None, (x.strip() for x in body.split(","))):
        key, sep, raw = item.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"{text!r}: {item!r} thiếu '='")
        value = yaml.safe_load(raw)
        if key.startswith("fit."):
            fit[key[4:]] = value
        else:
            overrides[key.strip()] = value
    return step, Candidate(name, fit, overrides)


def select(scores: dict[str, dict[str, float]], limit_ms: float = TTID_LIMIT_MS) -> str:
    """Rule of session 43: best mean HOTA among candidates with time-to-Global-ID p95 < 1 s.

    `scores` keeps the order the candidates were declared in; ties go to the earlier one.
    """
    allowed = {name: s for name, s in scores.items() if s["ttid_p95"] < limit_ms}
    if not allowed:
        raise ValueError(f"không ứng viên nào có thời gian tới Global ID p95 < {limit_ms:.0f} ms")
    return max(allowed, key=lambda name: allowed[name]["hota"])


def assert_models_clean(paths: dict[str, Path], test: str | None) -> None:
    """A learned file fitted on the test session or on the held-out group must never be run."""
    for component, path in paths.items():
        sessions = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("fit_sessions")
        if not sessions:
            raise LeakageError(f"{path} ({component}): thiếu fit_sessions")
        check_sessions(sessions, exclude=[test] if test else [])


def config_hash(config: dict[str, Any], learned: dict[str, Path]) -> str:
    """Hash of a configuration with each learned-file PATH replaced by that file's content."""
    resolved = copy.deepcopy(config)
    for component, path in learned.items():
        text = path.read_text(encoding="utf-8")
        # The fit date is provenance, not behaviour: a rerun tomorrow must hit the cache.
        body = "\n".join(line for line in text.splitlines() if not line.startswith("fit_date:"))
        resolved = apply_overrides(resolved, {CONFIG_KEYS[component]: body})
    blob = json.dumps(resolved, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def session_score(summary: dict[str, Any]) -> dict[str, Any]:
    """The numbers of one `run_lab_eval` summary (single variant `base`) this tool pools."""
    block = summary["base"]
    return {
        "mct": {m: float(block["mct"][m][0]) for m in ("HOTA", "AssA", "IDF1")},
        "handover": {
            kind: {o: int(block["handover"][kind][o]) for o in OUTCOMES}
            for kind in (*(k for k, _ in HANDOVER_TYPES), "total")
        },
    }


def pool(per_session: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pool test-session scores: metrics averaged, handover counts summed, Wilson on the total."""
    sessions = sorted(per_session)
    out: dict[str, Any] = {
        m: float(np.mean([per_session[s]["mct"][m] for s in sessions]))
        for m in ("HOTA", "AssA", "IDF1")
    }
    for kind in (*(k for k, _ in HANDOVER_TYPES), "total"):
        counts = Counts()
        for s in sessions:
            counts = counts + Counts(*(per_session[s]["handover"][kind][o] for o in OUTCOMES))
        out[kind] = {
            "dung": counts.dung,
            "tach": counts.tach,
            "nham": counts.nham,
            "n": counts.n,
            "wilson": list(wilson(counts.dung, counts.n)),
        }
    ttid = np.array([v for s in sessions for v in per_session[s].get("ttid", [])], dtype=float)
    if len(ttid):
        out["ttid_p50"], out["ttid_p95"] = (float(v) for v in np.percentile(ttid, [50, 95]))
        out["ttid_n"] = len(ttid)
    return out


def _pct(block: dict[str, Any], *, interval: bool = False) -> str:
    if not block["n"]:
        return "–"
    text = f"{100 * block['dung'] / block['n']:.0f}%"
    if interval:
        lo, hi = block["wilson"]
        return f"{text} [{100 * lo:.0f}–{100 * hi:.0f}]"
    return f"{text} ({block['dung']}/{block['n']})"


def format_table(rows: dict[str, dict[str, Any]], chosen: dict[str, dict[str, str]]) -> str:
    header = ["bước", "HOTA", "AssA", "IDF1", *(label for _, label in HANDOVER_TYPES)]
    header += [
        "tổng [Wilson 95%]",
        "đúng/tách/nhầm",
        "tới Global ID p50 / p95 (ms)",
        "chọn theo fold",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, row in rows.items():
        total = row["total"]
        picks: dict[str, int] = {}
        for candidate in chosen.get(name, {}).values():
            picks[candidate] = picks.get(candidate, 0) + 1
        cells = [name, *(f"{row[m]:.1f}" for m in ("HOTA", "AssA", "IDF1"))]
        cells += [_pct(row[kind]) for kind, _ in HANDOVER_TYPES]
        cells += [
            _pct(total, interval=True),
            f"{total['dung']}/{total['tach']}/{total['nham']}",
            f"{row['ttid_p50']:.0f} / {row['ttid_p95']:.0f}" if "ttid_p50" in row else "–",
            ", ".join(f"{c}×{n}" for c, n in sorted(picks.items())) or "–",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


class Runner:
    """Fits per fold, runs `run_lab_eval` + time-to-ID per (configuration, session), caches both."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.work: Path = args.work_dir
        self.base_config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        self.lab = LabData(
            args.fixtures_dir,
            args.gt_dir,
            roi_filter_fixture.topology_filter(args.topology, args.homography_dir),
            self.work / "geom",
        )
        self.mapper = HomographyMapper.load(args.homography_dir)
        self.overlaps = load_overlaps(args.topology)
        self._scores: dict[tuple[str, str], dict[str, Any]] = {}

    # -- fitting ----------------------------------------------------------------

    def learned(self, step: Step, candidate: Candidate, train: list[str], test: str | None) -> dict:
        """Learned files of this candidate, fitted on `train` only. {} for a fixed step."""
        if not step.components:
            return {}
        fit_key = ",".join(f"{k}={v}" for k, v in sorted(candidate.fit.items())) or "default"
        out_dir = self.work / "folds" / (test or ALL) / ("+".join(step.components) + "__" + fit_key)
        options = dict(candidate.fit)
        if "foot" not in step.components:
            options.setdefault("foot_method", None)
        paths = fit_all(
            self.lab,
            train,
            out_dir,
            mapper=self.mapper,
            overlaps=self.overlaps,
            calib=self.args.calib,
            exclude=[test] if test else [],
            components=step.components,
            **options,
        )
        assert_models_clean(paths, test)
        return paths

    def config_for(self, candidate: Candidate, learned: dict[str, Path]) -> tuple[str, dict]:
        overrides = dict(candidate.overrides)
        overrides |= {CONFIG_KEYS[c]: str(p.resolve()) for c, p in learned.items()}
        config = apply_overrides(self.base_config, overrides)
        return config_hash(apply_overrides(self.base_config, candidate.overrides), learned), config

    # -- scoring ----------------------------------------------------------------

    def _run_one(self, key: str, config: dict[str, Any], session: str) -> dict[str, Any]:
        a = self.args
        run_dir = self.work / "runs" / key / session
        done = run_dir / "score.json"
        if done.is_file():
            return json.loads(done.read_text(encoding="utf-8"))
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = run_dir.parent / "mct.yaml"
        if not cfg.is_file():
            cfg.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        fixtures = [
            (r, self.lab.run_fixture(session, r))
            for r in RUNS
            if self.lab.run_fixture(session, r).is_file()
        ]
        common = ["--topology", str(a.topology), "--homography-dir", str(a.homography_dir)]
        cmd = [a.engine_python, "-m", "eval.run_lab_eval", "--config", str(cfg), *common]
        cmd += ["--gt-fixture", str(self.lab.gt_fixture(session)), "--no-sct"]
        cmd += ["--engine-python", a.engine_python, "--eval-python", a.eval_python]
        cmd += ["--work-dir", str(run_dir), "--fps", str(a.fps)]
        if a.trackeval_path:
            cmd += ["--trackeval-path", str(a.trackeval_path)]
        for name, fixture in fixtures:
            cmd += ["--run", name, str(fixture)]
        ttid_json = run_dir / "ttid.json"
        ttid = [a.engine_python, "-m", "eval.time_to_id_datatime", "--config", str(cfg), *common]
        ttid += ["--json", str(ttid_json), *(str(f) for _, f in fixtures)]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(["src", "."])}
        steps = [(cmd, "lab_eval.log"), (ttid, "ttid.log")]
        strict = (config.get("association") or {}).get("revise_after_frames") is not None
        if strict:  # rescore on the patched DBs: run_lab_eval reuses a DB that exists
            steps.append((cmd, "lab_eval_strict.log"))
        patched = 0
        for i, (command, log) in enumerate(steps):
            if i == 2:
                patched = _strict_patch(run_dir, [name for name, _ in fixtures])
            with (run_dir / log).open("w", encoding="utf-8") as fh:
                proc = subprocess.run(
                    command, stdout=fh, stderr=subprocess.STDOUT, check=False, env=env
                )
            if proc.returncode:
                raise RuntimeError(f"{' '.join(command[:4])} thất bại, xem {run_dir / log}")
        score = session_score(json.loads((run_dir / "summary.json").read_text(encoding="utf-8")))
        if strict:
            score["strict_patched"] = patched
        score["ttid"] = json.loads(ttid_json.read_text(encoding="utf-8"))["values"]
        done.write_text(json.dumps(score), encoding="utf-8")
        if not a.keep_runs:  # SQLite + ROI copies of the fixtures: ~80 MB per run, all regenerable
            for child in run_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
        return score

    def score(self, jobs: list[tuple[str, dict[str, Any], str]]) -> None:
        """Run every (config key, config, session) not scored yet, `--jobs` at a time."""
        todo = {(k, s): c for k, c, s in jobs if (k, s) not in self._scores}
        with ThreadPoolExecutor(max_workers=self.args.jobs) as pool_:
            futures = {ks: pool_.submit(self._run_one, ks[0], c, ks[1]) for ks, c in todo.items()}
            for ks, future in futures.items():
                self._scores[ks] = future.result()

    def get(self, key: str, session: str) -> dict[str, Any]:
        return self._scores[(key, session)]


def _strict_patch(run_dir: Path, runs: list[str]) -> int:
    """Strict scoring of `revise_after_frames` (session 44): frames shown before an ID-changing
    revision keep the provisional ID. Patches each run's SQLite in place."""
    from eval.strict_revise_score import patch, revisions

    work = run_dir / "base"
    return sum(
        patch(
            work / f"base_{r}" / "mct.db",
            revisions(work / "mct.yaml", run_dir / "roi" / f"{r}.jsonl"),
        )
        for r in runs
    )


def evaluate(runner: Runner, steps: list[Step], sessions: list[str], *, final: bool) -> dict:
    """Run the whole protocol; returns {rows, chosen, folds, final}."""
    folds: list[str | None] = [*sessions, *([None] if final else [])]
    plan: dict[tuple[str, str], dict[str, str]] = {}  # (step, fold) -> {candidate: config key}
    learned_of: dict[tuple[str, str, str], dict[str, Path]] = {}
    jobs: list[tuple[str, dict[str, Any], str]] = []
    for step in steps:
        for test in folds:
            train = [s for s in sessions if s != test]
            keys: dict[str, str] = {}
            for candidate in step.candidates:
                learned = runner.learned(step, candidate, train, test)
                key, config = runner.config_for(candidate, learned)
                keys[candidate.name] = key
                learned_of[(step.name, test or ALL, candidate.name)] = learned
                jobs += [(key, config, s) for s in sessions]
            plan[(step.name, test or ALL)] = keys
    runner.score(jobs)

    def train_scores(keys: dict[str, str], train: list[str]) -> dict[str, dict[str, float]]:
        out = {}
        for name, key in keys.items():
            ttid = [v for s in train for v in runner.get(key, s)["ttid"]]
            out[name] = {
                "hota": float(np.mean([runner.get(key, s)["mct"]["HOTA"] for s in train])),
                "ttid_p95": float(np.percentile(ttid, 95)) if ttid else float("inf"),
            }
        return out

    rows: dict[str, dict[str, Any]] = {}
    chosen: dict[str, dict[str, str]] = {}
    detail: dict[str, Any] = {}
    final_pick: dict[str, Any] = {}
    for step in steps:
        tested: dict[str, dict[str, Any]] = {}
        chosen[step.name] = {}
        for test in folds:
            keys = plan[(step.name, test or ALL)]
            scores = train_scores(keys, [s for s in sessions if s != test])
            pick = select(scores)
            detail[f"{step.name}/{test or ALL}"] = {"train": scores, "chosen": pick}
            if test is None:
                final_pick[step.name] = {
                    "candidate": pick,
                    "learned": {c: str(p) for c, p in learned_of[(step.name, ALL, pick)].items()},
                }
                continue
            chosen[step.name][test] = pick
            tested[test] = runner.get(keys[pick], test)
        rows[step.name] = pool(tested)
    return {"rows": rows, "chosen": chosen, "folds": detail, "final": final_pick}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fixtures-dir", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--topology", type=Path, required=True)
    p.add_argument("--homography-dir", type=Path, required=True)
    p.add_argument("--sessions", nargs="+", default=list(SELECT))
    p.add_argument("--calib", nargs="*", default=[])
    p.add_argument("--step", type=parse_step, action="append", required=True)
    p.add_argument("--candidate", type=parse_candidate, action="append", default=[])
    p.add_argument("--final-dir", type=Path, default=None)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--engine-python", default=sys.executable)
    p.add_argument("--eval-python", default=sys.executable)
    p.add_argument("--trackeval-path", type=Path, default=None)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--keep-runs", action="store_true", help="giữ SQLite/TrackEval của từng lượt")
    args = p.parse_args(argv)

    sessions = check_sessions(args.sessions)  # raises on any held-out session
    by_step: dict[str, list[Candidate]] = {}
    for step_name, candidate in args.candidate:
        by_step.setdefault(step_name, []).append(candidate)
    unknown = sorted(set(by_step) - {name for name, _ in args.step})
    if unknown:
        p.error(f"--candidate cho bước không khai: {unknown}")
    steps = [
        Step(name, components, tuple(by_step.get(name, [Candidate("default")])))
        for name, components in args.step
    ]

    result = evaluate(Runner(args), steps, sessions, final=args.final_dir is not None)
    table = format_table(result["rows"], result["chosen"])
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "loso.md").write_text(table + "\n", encoding="utf-8")
    (args.work_dir / "loso.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.final_dir is not None:
        last = steps[-1].name
        args.final_dir.mkdir(parents=True, exist_ok=True)
        for path in result["final"][last]["learned"].values():
            shutil.copy(path, args.final_dir / Path(path).name)
        print(f"cấu hình cuối ({last}): {result['final'][last]} -> {args.final_dir}")
    print(table)
    print(f"\n-> {args.work_dir / 'loso.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
