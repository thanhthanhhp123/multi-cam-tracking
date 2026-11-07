"""Time to Global ID measured in DATA time (frame timestamps), without a GPU or a wall clock.

One sample per non-update assignment (the same records `tools.latency_report` counts as
`kind=first`): `latest frame ts_ms seen by the builder - first frame of the tracklet`. It
reproduces the shape of the wall-clock measurement (session 41: code-before-fix p95 10.7 s here
vs 10.8 s on the GPU logs of session 40) minus the pipeline stage before the engine (t0 -> t1,
~75 ms), so it is a way to compare engine variants on recorded fixtures, NOT a reportable latency.

`--former-owner both` replays each fixture twice, with `Gallery.pop_former_owner` disabled and
enabled, to show the effect of the session-41 fix.

    PYTHONPATH="src;." python -m eval.time_to_id_datatime --config configs/lab/lab.mct.yaml \\
        --topology configs/lab/topology.yaml --homography-dir configs/lab/homography \\
        data/fixtures/lab_s2_1_r1.jsonl data/fixtures/lab_s2_2_r1.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common.schema import read_jsonl
from mct.__main__ import build_engine, load_config

PERCENTILES = (50, 90, 95, 99)


def collect_time_to_id(
    config: dict[str, Any],
    fixtures: list[Path],
    *,
    topology: Path | None,
    homography_dir: Path | None,
    former_owner_fix: bool = True,
) -> list[float]:
    """Data-time delay (ms) of every first assignment of a tracklet, over all fixtures."""
    values: list[float] = []
    for fixture in fixtures:
        engine = build_engine(
            config, topology_path=topology, homography_dir=homography_dir, db_path=":memory:"
        )
        if not former_owner_fix:
            engine.associator.gallery.pop_former_owner = lambda tracklet: None  # type: ignore[method-assign]
        builder = engine.builder
        original = engine.associator.assign

        def assign(ts_ms: int, original: Any = original, builder: Any = builder) -> Any:
            result = original(ts_ms)
            for assignment in result:
                if not assignment.is_update:
                    values.append(float(builder.latest_ts_ms - assignment.tracklet.start_ms))
            return result

        engine.associator.assign = assign  # type: ignore[method-assign]
        for message in read_jsonl(fixture):
            engine.feed(message)
        engine.finish()
    return values


def summarize(values: list[float]) -> dict[str, float]:
    """n, percentiles (ms) and the share under one second."""
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    out: dict[str, float] = {"n": float(len(array))}
    for p, v in zip(PERCENTILES, np.percentile(array, PERCENTILES), strict=True):
        out[f"p{p}"] = float(v)
    out["under_1s"] = float(np.mean(array < 1000.0))
    return out


def format_row(label: str, stats: dict[str, float]) -> str:
    if not stats.get("n"):
        return f"| {label} | 0 | – | – | – | – | – |"
    cells = [f"{stats[f'p{p}']:.0f}" for p in PERCENTILES]
    return (
        f"| {label} | {int(stats['n'])} | "
        + " | ".join(cells)
        + f" | {100 * stats['under_1s']:.1f}% |"
    )


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("fixtures", type=Path, nargs="+")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--topology", type=Path, default=None)
    p.add_argument("--homography-dir", type=Path, default=None)
    p.add_argument(
        "--former-owner",
        choices=("on", "off", "both"),
        default="on",
        help="Gallery.pop_former_owner (bản sửa phiên 41): bật, tắt, hay chạy cả hai để so",
    )
    args = p.parse_args(argv)
    config = load_config(args.config)
    modes = {"on": [True], "off": [False], "both": [False, True]}[args.former_owner]
    print("| former owner | n | p50 ms | p90 ms | p95 ms | p99 ms | < 1 s |")
    print("|---|---|---|---|---|---|---|")
    for fix in modes:
        values = collect_time_to_id(
            config,
            args.fixtures,
            topology=args.topology,
            homography_dir=args.homography_dir,
            former_owner_fix=fix,
        )
        print(format_row("on" if fix else "off", summarize(values)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
