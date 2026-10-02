"""Đánh đổi độ trễ chốt danh tính ↔ độ chính xác liên kết, quét `window_ms` × `min_frames`.

    F=data/fixtures; T=$F/ds_wildtrack_7cam_r640n_r1.gt.json
    PYTHONPATH="src;." ~/.venvs/mct-test/Scripts/python.exe -m eval.latency_tradeoff \\
        --config configs/demo/wildtrack_ds.mct.yaml \\
        --topology configs/demo/wildtrack.topology.yaml \\
        --homography-dir configs/cameras/homography/wildtrack \\
        --window-ms 250 500 1000 2000 --min-frames 1 2 3 5 \\
        --gt-fixture $F/wildtrack_7cam.jsonl --gt-fixture-table $F/wildtrack_7cam.gt.json \\
        --eval-python ~/.venvs/mct-eval/Scripts/python.exe --trackeval-path ~/TrackEval \\
        --run r1 $F/ds_wildtrack_7cam_r640n_r1.jsonl $T

Bỏ các cờ `--gt-*` thì chỉ đo độ trễ (dùng cho fixture không có ground-truth, vd fixture
tốc độ thật ~30 fps của `ds_4cam_*`).

**Độ trễ chốt danh tính là gì, và vì sao KHÔNG phải đuôi p90 của phiên 23.** Engine gán
Global ID cho một tracklet ở vòng gán ĐẦU TIÊN mà tracklet đã đủ `min_frames`; từ đó
`Gallery.find_by_tracklet` giữ nguyên chủ sở hữu, các vòng sau chỉ là `is_update`
(`Associator.assign`). Tức danh tính được chốt đúng một lần. Đuôi 2–3 s đo ở phiên 23 là
bản ghi CUỐI của mỗi tracklet (phát lại lúc tracklet đóng sau `idle_timeout_ms`), không
phải lúc chốt danh tính — kiểm trên chính hai file log của phiên đó: 100% bản ghi có
`window_wait` > 1.5 s là bản ghi cuối của tracklet, và tính riêng lần gán đầu thì
end-to-end chỉ p50 107 / p90 ~600 / max ~1040 ms. Hệ quả: `idle_timeout_ms` KHÔNG nằm trên
đường chốt danh tính; hai núm vặn thật là:

- `min_frames` — tracklet phải có đủ ngần ấy khung mới được gán (≈ (min_frames − 1) / fps);
- `window_ms` — sau đó còn chờ cửa sổ hiện tại đóng (≤ `window_ms`).

**Thước đo ở đây: `time_to_id` = mốc dữ liệu của vòng gán đầu tiên − `start_ms` của
tracklet**, tính hoàn toàn bằng `ts_ms` trong message (engine tất định, không đồng hồ tường).
Đó là thời gian từ khung đầu tiên một người xuất hiện ở một camera tới lúc engine gán cho họ
Global ID, CHƯA gồm DeepStream + vận chuyển (phiên 23: t1−t0 ≈ 67 ms, các chặng khác < 10 ms).
Tracklet chỉ được gán ở vòng cuối lúc hết nguồn (`finish`) bị tách riêng (`n_final`): đó là
hiện vật của phát lại offline, cùng loại `final_flush` của phiên 23.

**Đọc số trên WildTrack cẩn thận.** WildTrack chạy 2 fps: một khung = 500 ms, nên `min_frames`
3 tự nó đã ≥ 1 s và độ phân giải của mọi độ trễ là 500 ms. Phía ĐỘ CHÍNH XÁC (HOTA/AssA/IDF1)
lấy từ WildTrack vì chỉ ở đó có ground-truth; phía ĐỘ TRỄ ở tốc độ khung thật phải đo trên
fixture ~30 fps. Không chép `time_to_id` của WildTrack sang hệ thống 25 fps.

Chấm điểm đi đúng đường của `eval.compare_oracle_tracker` (engine → `tools.export_trackeval
--mode mct` → TrackEval), nên HOTA ở đây so được thẳng với phiên 22/24/25.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from common.schema import read_jsonl
from mct.__main__ import build_engine, load_config
from mct.associator import Assignment

PERCENTILES = (0.5, 0.9, 0.95, 0.99)
SCORE_COLUMNS = ("HOTA", "DetA", "AssA", "IDF1", "IDs")


@dataclass
class FirstAssignment:
    """Lần gán đầu tiên của một tracklet — lúc danh tính của nó được chốt."""

    time_to_id_ms: int
    n_frames: int
    is_new: bool
    at_close: bool
    final: bool


class FirstAssignmentLog:
    """`window_observer` của `Engine`: ghi lại vòng gán ĐẦU TIÊN của mỗi tracklet."""

    def __init__(self) -> None:
        self.first: dict[int, FirstAssignment] = {}
        self.final = False
        """Bật trước `Engine.finish()` để đánh dấu các tracklet chỉ được gán ở vòng cuối."""

    def __call__(self, now_ms: int, results: list[Assignment]) -> None:
        for assignment in results:
            tracklet = assignment.tracklet
            if tracklet.tracklet_id in self.first:
                continue
            self.first[tracklet.tracklet_id] = FirstAssignment(
                time_to_id_ms=int(now_ms) - int(tracklet.start_ms),
                n_frames=tracklet.n_frames,
                is_new=assignment.is_new,
                at_close=tracklet.closed,
                final=self.final,
            )


def percentile(values: list[float], q: float) -> float:
    """Phân vị theo hạng gần nhất (không nội suy — độ trễ lượng tử theo khung)."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))])


def summarize(log: FirstAssignmentLog) -> dict[str, Any]:
    live = [f for f in log.first.values() if not f.final]
    delays = [float(f.time_to_id_ms) for f in live]
    out: dict[str, Any] = {
        "n_tracklets": len(log.first),
        "n_final": len(log.first) - len(live),
        "n_at_close": sum(f.at_close for f in live),
        "n_new": sum(f.is_new for f in live),
    }
    for q in PERCENTILES:
        out[f"p{round(q * 100)}_ms"] = percentile(delays, q)
    out["max_ms"] = max(delays) if delays else float("nan")
    out["frac_le_1s"] = sum(d <= 1000 for d in delays) / len(delays) if delays else float("nan")
    return out


def frame_interval_ms(fixture: Path, limit: int = 20000) -> float:
    """Trung vị khoảng cách giữa hai khung liên tiếp của cùng camera (để đọc `min_frames`)."""
    last: dict[str, int] = {}
    gaps: list[int] = []
    for i, msg in enumerate(read_jsonl(fixture)):
        if i >= limit:
            break
        prev = last.get(msg.cam_id)
        if prev is not None and msg.ts_ms > prev:
            gaps.append(int(msg.ts_ms) - prev)
        last[msg.cam_id] = int(msg.ts_ms)
    return float(statistics.median(gaps)) if gaps else float("nan")


def derive_config(base: dict[str, Any], window_ms: int, min_frames: int) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config.setdefault("association", {})["window_ms"] = int(window_ms)
    config.setdefault("tracklet", {})["min_frames"] = int(min_frames)
    return config


def run_engine(
    config: dict[str, Any],
    fixture: Path,
    db: Path,
    *,
    topology: Path | None,
    homography_dir: Path | None,
) -> tuple[dict[str, Any], FirstAssignmentLog]:
    """Chạy đường online (`Engine.feed` từng message rồi `finish`) trong cùng tiến trình."""
    for suffix in ("", "-shm", "-wal"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    engine = build_engine(
        config, topology_path=topology, homography_dir=homography_dir, db_path=str(db)
    )
    log = FirstAssignmentLog()
    engine.window_observer = log
    for msg in read_jsonl(fixture):
        engine.feed(msg)
    log.final = True
    engine.finish()
    summary: dict[str, Any] = {}
    if engine.store is not None:
        summary = engine.store.summary()
        engine.store.close()
    stats = summarize(log)
    stats["n_global_ids"] = summary.get("n_tracks")
    stats["n_dropped_short"] = engine.builder.n_dropped_short
    return stats, log


def score(label: str, fixture: Path, table: Path, work: Path, args: argparse.Namespace):
    """HOTA/AssA/IDF1 bằng đúng đường của `compare_oracle_tracker` (DB đã có sẵn)."""
    from eval.compare_oracle_tracker import score_run

    ns = argparse.Namespace(
        force=False,
        engine_python=sys.executable,
        eval_python=args.eval_python,
        trackeval_path=args.trackeval_path,
        gt_fixture=args.gt_fixture,
        gt_fixture_table=args.gt_fixture_table,
        fps=args.fps,
        sct=False,
        config=None,
        topology=None,
        homography_dir=None,
    )
    mct, _ = score_run(label, fixture, table, work, ns)
    return mct


def aggregate(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    if not values:
        return float("nan"), float("nan")
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def format_table(results: dict[str, list[dict[str, Any]]], scored: bool) -> str:
    head = ["cấu hình", "time_to_id p50", "p90", "p99", "≤1 s", "Global ID", "bỏ vì ngắn"]
    if scored:
        head += list(SCORE_COLUMNS[:-1])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, rows in results.items():
        cells = [label]
        for key in ("p50_ms", "p90_ms", "p99_ms"):
            mean, _ = aggregate(rows, key)
            cells.append(f"{mean:.0f}")
        cells.append(f"{aggregate(rows, 'frac_le_1s')[0] * 100:.1f}%")
        cells.append(f"{aggregate(rows, 'n_global_ids')[0]:.0f}")
        cells.append(f"{aggregate(rows, 'n_dropped_short')[0]:.0f}")
        if scored:
            for col in SCORE_COLUMNS[:-1]:
                mean, sd = aggregate(rows, col)
                cells.append(f"{mean:.2f} ± {sd:.2f}" if len(rows) > 1 else f"{mean:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run",
        nargs="+",
        action="append",
        required=True,
        metavar="ARG",
        help="'<nhãn> <fixture> [<bảng .gt.json>]' — bảng chỉ cần khi chấm điểm",
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--topology", type=Path, default=None)
    p.add_argument("--homography-dir", type=Path, default=None)
    p.add_argument("--window-ms", type=int, nargs="+", default=[1000])
    p.add_argument("--min-frames", type=int, nargs="+", default=None)
    p.add_argument("--gt-fixture", type=Path, default=None)
    p.add_argument("--gt-fixture-table", type=Path, default=None)
    p.add_argument("--eval-python", default=sys.executable)
    p.add_argument("--trackeval-path", type=Path, default=None)
    p.add_argument("--fps", type=float, default=2.0, help="chỉ ghi vào seqinfo.ini")
    p.add_argument("--work-dir", type=Path, default=Path("data/latency_tradeoff"))
    p.add_argument("--json", type=Path, default=None, help="mặc định <work-dir>/tradeoff.json")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    runs = []
    for spec in args.run:
        if len(spec) not in (2, 3):
            p.error("--run cần '<nhãn> <fixture> [<bảng .gt.json>]'")
        runs.append((spec[0], Path(spec[1]), Path(spec[2]) if len(spec) == 3 else None))
    scored = args.gt_fixture is not None and args.gt_fixture_table is not None
    if scored and any(table is None for _, _, table in runs):
        p.error("chấm điểm cần bảng .gt.json cho mọi --run")

    base = load_config(args.config)
    base_min = int((base.get("tracklet") or {}).get("min_frames", 5))
    min_frames_grid = args.min_frames or [base_min]
    topology = args.topology if args.topology and args.topology.is_file() else None

    results: dict[str, list[dict[str, Any]]] = {}
    for window_ms in args.window_ms:
        for min_frames in min_frames_grid:
            cfg_label = f"w{window_ms}_m{min_frames}"
            config = derive_config(base, window_ms, min_frames)
            for run_label, fixture, table in runs:
                label = f"{cfg_label}:{run_label}"
                run_dir = args.work_dir / label.replace(":", "_")
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "config.yaml").write_text(
                    yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
                )
                stats, _ = run_engine(
                    config,
                    fixture,
                    run_dir / "mct.db",
                    topology=topology,
                    homography_dir=args.homography_dir,
                )
                stats["frame_interval_ms"] = frame_interval_ms(fixture)
                if scored and table is not None:
                    stats.update(
                        {
                            k: v
                            for k, v in score(label, fixture, table, args.work_dir, args).items()
                            if k in SCORE_COLUMNS
                        }
                    )
                results.setdefault(cfg_label, []).append({"run": run_label, **stats})
                line = (
                    f"{label}: time_to_id p50 {stats['p50_ms']:.0f} p90 {stats['p90_ms']:.0f} "
                    f"p99 {stats['p99_ms']:.0f} ms, {stats['n_global_ids']} Global ID"
                )
                if "HOTA" in stats:
                    line += f", HOTA {stats['HOTA']:.3f} AssA {stats['AssA']:.3f}"
                print(line, flush=True)

    print()
    print(format_table(results, scored))
    out = args.json or (args.work_dir / "tradeoff.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
