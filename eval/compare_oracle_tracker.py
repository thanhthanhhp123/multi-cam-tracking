"""Tách lỗi HOTA thành ba tầng: detector / tracker đơn camera / module liên kết.

    F=data/fixtures; T=$F/ds_wildtrack_7cam.gt.json
    PYTHONPATH=src python -m eval.compare_oracle_tracker \\
        --config configs/demo/wildtrack_ds.mct.yaml \\
        --topology configs/demo/wildtrack.topology.yaml \\
        --homography-dir configs/cameras/homography/wildtrack \\
        --gt-fixture $F/wildtrack_7cam.jsonl --gt-fixture-table $F/wildtrack_7cam.gt.json \\
        --engine-python ~/.venvs/mct-test/Scripts/python.exe \\
        --eval-python ~/.venvs/mct-eval/Scripts/python.exe --trackeval-path ~/TrackEval \\
        --run A:r1 $F/ds_wildtrack_7cam.jsonl $T \\
        --run C:r1 $F/ds_wildtrack_7cam_onnx_gtbox.jsonl $T \\
        --run B:r1 $F/ds_wildtrack_7cam_onnx_oracle.jsonl $T

Mỗi `--run <kịch bản>:<lần> <fixture> <bảng .gt.json>` đi trọn đường thật: engine online
(`python -m mct --source`) → `tools.export_trackeval --mode mct` → TrackEval. Cùng cấu hình
engine, cùng ground-truth cho MỌI lần chạy — thứ duy nhất được phép khác là fixture.

Ba kịch bản (xem `tools/reembed_fixture.py`):

- **A** — thật: hộp detector YOLO11s, id NvDCF, embedding của nvtracker.
- **C** — `--boxes gt`: hộp ground-truth, id NvDCF, embedding ONNX trên crop GT.
- **B** — `--oracle-out`: như C nhưng id = personID. Cùng từng bit embedding với C.

Hiệu C→B chỉ có một nguyên nhân (id cục bộ), nên đó là chi phí của tracker đơn camera.
A→C trộn ba thứ (hộp, lọc FP, bộ trích embedding) — chỉ dùng để đo detector nói chung.

**Không đọc AssA của B như "khoảng cách tới 100".** AssA của HOTA phạt cả detection bị bỏ
sót (FNA), không chỉ id sai: tập detection ở đây chỉ phủ ~43% ground-truth, nên ngay cả hệ
hoàn hảo tuyệt đối cũng không đạt AssA 100. Muốn biết còn lại bao nhiêu là lỗi liên kết phải
so với TRẦN của cùng tập detection — hai kịch bản không qua engine (`--ceiling`, Global ID =
người thật, chọn theo đa số trong từng track):

- **LB** — id tracker hoàn hảo + liên kết hoàn hảo: trần của tập detection này.
- **LC** — id NvDCF + liên kết hoàn hảo: mức tốt nhất mà BẤT KỲ module liên kết nào đạt được
  trên tracklet NvDCF (track trộn hai người thì liên kết không cứu được). Hơi CHẶT tay: chọn
  người đa số theo cả (camera, id), không theo từng tracklet sau khi cắt bởi idle_timeout, nên
  trần thật ở mức tracklet có thể cao hơn chút (chưa đo mức chênh; idle_timeout 30 s ở 2 fps
  là 60 khung im lặng nên nhiều khả năng nhỏ).

Bốn góc LB / B / LC / C tách lỗi theo cả hai thứ tự, nên kết luận không phụ thuộc "sửa cái
nào trước": LB→LC là thiệt hại của tracker mà liên kết không sửa nổi, LB→B và LC→C là
thiệt hại của khâu liên kết ứng với từng loại tracker.

`--sct` chấm thêm từng camera riêng lẻ (id = `local_track_id`, không qua `src/mct`): đó là
đo TRỰC TIẾP tracker đơn camera, để đối chiếu với hiệu C→B của đường xuyên camera.

Chỉ stdlib + `common/` — chạy bằng venv nhẹ; engine và TrackEval chạy trong tiến trình con bằng đúng
interpreter của chúng (`--engine-python`, `--eval-python`, vì TrackEval ghim numpy 1.23.5).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common.schema import FrameMessage, read_jsonl

REPORT_COLUMNS = ("HOTA", "DetA", "AssA", "AssRe", "AssPr", "IDF1", "IDs", "Dets")
SCT_COLUMNS = ("HOTA", "DetA", "AssA", "IDF1", "IDSW")


def parse_summary(text: str) -> dict[str, float]:
    """`pedestrian_summary.txt` của TrackEval (một dòng tên cột + một dòng giá trị) -> dict."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("summary TrackEval phải có dòng tên cột và dòng giá trị")
    names, values = lines[0].split(), lines[1].split()
    if len(names) != len(values):
        raise ValueError(f"{len(names)} tên cột nhưng {len(values)} giá trị")
    return {n: float(v) for n, v in zip(names, values, strict=True)}


def aggregate(runs: dict[str, dict[str, float]], columns: tuple[str, ...]) -> dict[str, dict]:
    """{"A:r1": {...}, "A:r2": {...}} -> {"A": {"n": 2, "HOTA": (mean, std), ...}}.

    Độ lệch chuẩn mẫu (ddof=1) như bảng nhiễu của phiên 22; n=1 thì std = None.
    """
    by_scenario: dict[str, list[dict[str, float]]] = {}
    for label, metrics in runs.items():
        by_scenario.setdefault(label.split(":", 1)[0], []).append(metrics)

    out: dict[str, dict] = {}
    for scenario, group in by_scenario.items():
        row: dict = {"n": len(group)}
        for col in columns:
            vals = [g[col] for g in group]
            row[col] = (
                statistics.fmean(vals),
                statistics.stdev(vals) if len(vals) > 1 else None,
            )
        out[scenario] = row
    return out


def format_table(agg: dict[str, dict], columns: tuple[str, ...]) -> str:
    header = "| kịch bản | n | " + " | ".join(columns) + " |"
    rule = "|---|---|" + "---|" * len(columns)
    rows = [header, rule]
    for scenario in sorted(agg):
        cells = []
        for col in columns:
            mean, std = agg[scenario][col]
            digits = 0 if col in ("IDs", "Dets", "IDSW") else 2
            cells.append(f"{mean:.{digits}f}" + (f" ± {std:.{digits}f}" if std is not None else ""))
        rows.append(f"| {scenario} | {agg[scenario]['n']} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def majority_person_map(
    tracker_msgs: list[FrameMessage], oracle_msgs: list[FrameMessage]
) -> dict[tuple[str, int], int]:
    """(cam_id, id của fixture tracker) -> personID chiếm đa số trong track đó.

    Hai fixture phải CÙNG hàng (cùng message, cùng thứ tự detection) — đúng với cặp C/B mà
    `tools/reembed_fixture.py --oracle-out` sinh ra, vì oracle chỉ đổi `local_track_id`.
    Hoà phiếu thì lấy personID nhỏ hơn cho kết quả tất định.

    **Ràng buộc loại trừ.** Hai track cùng camera có mặt trong cùng một khung không được
    chung Global ID — engine thật cũng bị ràng buộc đó (CLAUDE.md §6), và TrackEval từ chối
    chấm kết quả có hai hộp cùng id trong một khung. NvDCF thường để lại hai track cùng
    "người đa số" chạy song song (một track chính, một track chứa vài hộp lạc): track nhiều
    phiếu hơn giữ personID, track xung đột nhận một Global ID MỚI (lớn hơn mọi personID).
    Hai mảnh nối tiếp không trùng khung của cùng một người vẫn được gộp — đúng việc của liên kết.
    """
    if len(tracker_msgs) != len(oracle_msgs):
        raise ValueError("hai fixture không cùng số message — không phải cặp tracker/oracle")
    votes: dict[tuple[str, int], Counter[int]] = defaultdict(Counter)
    frames: dict[tuple[str, int], set[int]] = defaultdict(set)
    for mt, mo in zip(tracker_msgs, oracle_msgs, strict=True):
        if (mt.cam_id, mt.frame_id) != (mo.cam_id, mo.frame_id) or len(mt.detections) != len(
            mo.detections
        ):
            raise ValueError(f"{mt.cam_id} frame {mt.frame_id}: hai fixture lệch hàng")
        for dt, do in zip(mt.detections, mo.detections, strict=True):
            key = (mt.cam_id, int(dt.local_track_id))
            votes[key][int(do.local_track_id)] += 1
            frames[key].add(int(mt.frame_id))

    majority = {k: min(c, key=lambda person: (-c[person], person)) for k, c in votes.items()}

    by_person: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for key, person in majority.items():
        by_person[(key[0], person)].append(key)

    next_fresh = max(majority.values(), default=0) + 1
    out: dict[tuple[str, int], int] = {}
    for (_cam, person), keys in sorted(by_person.items()):
        used: set[int] = set()
        for key in sorted(keys, key=lambda k: (-votes[k][person], k[1])):
            if frames[key] & used:
                out[key] = next_fresh
                next_fresh += 1
            else:
                out[key] = person
                used |= frames[key]
    return out


def write_identity_db(mapping: dict[tuple[str, int], int], path: Path) -> None:
    """SQLite tối thiểu mà `export_trackeval.load_global_ids` đọc được: mỗi track một Global ID
    cho TOÀN BỘ thời gian (không có engine nào chạy)."""
    for suffix in ("", "-shm", "-wal"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute(
            "CREATE TABLE appearances (cam_id TEXT, local_track_id INTEGER, "
            "start_ms INTEGER, end_ms INTEGER, global_id INTEGER)"
        )
        con.executemany(
            "INSERT INTO appearances VALUES (?, ?, ?, ?, ?)",
            [(cam, tid, 0, 2**62, person) for (cam, tid), person in sorted(mapping.items())],
        )
        con.commit()
    finally:
        con.close()


def _run(cmd: list[str], *, env_src: bool = True) -> None:
    import os

    env = dict(os.environ)
    if env_src:
        env["PYTHONPATH"] = "src"
    # errors="replace": log của tiến trình con trên Windows không phải lúc nào cũng là UTF-8
    # hợp lệ; ta chỉ đọc nó để in khi lỗi, đừng để việc giải mã làm chết luồng đọc.
    proc = subprocess.run(
        cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-2000:])
        raise SystemExit(f"lệnh lỗi ({proc.returncode}): {' '.join(cmd[:4])} ...")


def score_run(
    label: str,
    fixture: Path,
    table: Path,
    work: Path,
    args: argparse.Namespace,
    *,
    ceiling_of: Path | None = None,
) -> tuple[dict[str, float], dict[str, float] | None]:
    """Một lượt: engine -> export mct (+ sct) -> TrackEval. Trả (mct, sct hoặc None).

    `ceiling_of` (đường dẫn fixture oracle) thì KHÔNG chạy engine: Global ID = người thật.
    """
    run_dir = work / label.replace(":", "_")
    run_dir.mkdir(parents=True, exist_ok=True)
    db = run_dir / "mct.db"
    if ceiling_of is not None:
        write_identity_db(
            majority_person_map(list(read_jsonl(fixture)), list(read_jsonl(ceiling_of))), db
        )
    elif args.force and db.exists():
        for suffix in ("", "-shm", "-wal"):
            Path(str(db) + suffix).unlink(missing_ok=True)

    if ceiling_of is None and not db.exists():
        _run(
            [
                args.engine_python,
                "-m",
                "mct",
                "--config",
                str(args.config),
                "--source",
                str(fixture),
                "--topology",
                str(args.topology),
                "--homography-dir",
                str(args.homography_dir),
                "--db",
                str(db),
            ]
        )

    def export_and_eval(mode: str) -> dict[str, float]:
        root = run_dir / f"trackeval-{mode}"
        cmd = [
            args.engine_python,
            "-m",
            "tools.export_trackeval",
            "--fixture",
            str(fixture),
            "--gt",
            str(table),
            "--gt-fixture",
            str(args.gt_fixture),
            "--gt-fixture-table",
            str(args.gt_fixture_table),
            "--out",
            str(root),
            "--mode",
            mode,
            "--fps",
            str(args.fps),
        ]
        if mode == "mct":
            cmd += ["--db", str(db)]
        if getattr(args, "only_gt_frames", False):
            cmd.append("--only-gt-frames")
        _run(cmd)
        eval_cmd = [
            args.eval_python,
            "eval/run_trackeval.py",
            "--root",
            str(root),
            "--split",
            mode,
        ]
        if args.trackeval_path:
            eval_cmd += ["--trackeval-path", str(args.trackeval_path)]
        _run(eval_cmd)
        summary = root / "trackers" / f"MCT-{mode}" / "mct-engine" / "pedestrian_summary.txt"
        return parse_summary(summary.read_text(encoding="utf-8"))

    mct = export_and_eval("mct")
    sct = export_and_eval("sct") if args.sct and ceiling_of is None else None
    return mct, sct


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("KỊCH_BẢN:LẦN", "FIXTURE", "BẢNG_GT"),
        default=[],
        help="nhãn dạng 'A:r1' (kịch bản trước dấu ':' quyết định nhóm lấy trung bình)",
    )
    p.add_argument(
        "--ceiling",
        nargs=4,
        action="append",
        default=[],
        metavar=("KỊCH_BẢN:LẦN", "FIXTURE_TRACKER", "FIXTURE_ORACLE", "BẢNG_GT"),
        help="trần liên kết hoàn hảo, không chạy engine: mỗi track của FIXTURE_TRACKER nhận "
        "personID đa số theo FIXTURE_ORACLE (vd LC = fixture C + fixture B; LB = B + B)",
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--topology", type=Path, required=True)
    p.add_argument("--homography-dir", type=Path, required=True)
    p.add_argument("--gt-fixture", type=Path, required=True)
    p.add_argument("--gt-fixture-table", type=Path, required=True)
    p.add_argument("--engine-python", default=sys.executable)
    p.add_argument("--eval-python", default=sys.executable)
    p.add_argument("--trackeval-path", type=Path, default=None)
    p.add_argument("--work-dir", type=Path, default=Path("data/s24"))
    p.add_argument("--fps", type=float, default=2.0, help="chỉ ghi vào seqinfo.ini")
    p.add_argument("--sct", action="store_true", help="chấm thêm từng camera riêng (đo tracker)")
    p.add_argument("--force", action="store_true", help="chạy lại engine dù đã có DB")
    p.add_argument(
        "--only-gt-frames",
        action="store_true",
        help="chỉ chấm khung có trong --gt-fixture (chú thích nhảy khung, dữ liệu tự thu M6)",
    )
    args = p.parse_args(argv)
    if not args.run and not args.ceiling:
        p.error("cần ít nhất một --run hoặc --ceiling")

    # Windows chuyển hướng stdout sang file bằng cp1252 — tiêu đề tiếng Việt sẽ làm chương
    # trình chết SAU KHI đã chấm xong, tức mất kết quả vì một dòng in.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    mct_runs: dict[str, dict[str, float]] = {}
    sct_runs: dict[str, dict[str, float]] = {}
    jobs: list[tuple[str, Path, Path, Path | None]] = [
        (label, Path(fixture), Path(table), None) for label, fixture, table in args.run
    ] + [
        (label, Path(tracker), Path(table), Path(oracle))
        for label, tracker, oracle, table in args.ceiling
    ]
    for label, fixture, table, ceiling_of in jobs:
        mct, sct = score_run(label, fixture, table, args.work_dir, args, ceiling_of=ceiling_of)
        mct_runs[label] = mct
        if sct is not None:
            sct_runs[label] = sct
        print(
            f"{label}: HOTA {mct['HOTA']:.3f} DetA {mct['DetA']:.3f} AssA {mct['AssA']:.3f} "
            f"IDF1 {mct['IDF1']:.3f} IDs {mct['IDs']:.0f}",
            flush=True,
        )

    print("\n## Xuyên camera (đường online đầy đủ: engine -> TrackEval, chuỗi ảo 7 camera)\n")
    print(format_table(aggregate(mct_runs, REPORT_COLUMNS), REPORT_COLUMNS))
    if sct_runs:
        print("\n## Đơn camera (id = local_track_id, KHÔNG qua src/mct)\n")
        print(format_table(aggregate(sct_runs, SCT_COLUMNS), SCT_COLUMNS))

    out = args.work_dir / "compare_oracle_tracker.json"
    out.write_text(
        json.dumps({"mct": mct_runs, "sct": sct_runs}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
