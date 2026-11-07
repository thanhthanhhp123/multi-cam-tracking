"""Strict scoring for revise_after_frames: frames shown BEFORE a revision keep the provisional ID.

SQLite keeps one Global ID per tracklet (upsert), so a revision that changes the ID relabels the
whole tracklet retroactively. Live, the first part was published under the provisional ID. This
replays the engine in-process (deterministic, same tracklet ids as the DB), records each
ID-changing revision, and patches the run's mct.db: the main row starts after the revision, a
synthetic row carries the provisional ID for [start, end-at-revision].

    python strict_revise.py <work_root_copy> <variant>...
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path[:0] = ["src", "."]
from common.schema import read_jsonl  # noqa: E402
from mct.__main__ import build_engine, load_config  # noqa: E402

SYN = 1_000_000_000


def revisions(config_path: Path, fixture: Path) -> list[tuple[int, int, int]]:
    engine = build_engine(
        load_config(config_path),
        topology_path=Path("configs/lab/topology.yaml"),
        homography_dir=Path("configs/lab/homography"),
        db_path=":memory:",
    )
    assoc = engine.associator
    out: list[tuple[int, int, int]] = []
    original = assoc._revise

    def hook(tracklet, owner):  # type: ignore[no-untyped-def]
        old_gid, end_ms = owner.global_id, int(tracklet.end_ms)
        res = original(tracklet, owner)
        if res.global_id != old_gid:
            out.append((tracklet.tracklet_id, old_gid, end_ms))
        return res

    assoc._revise = hook  # type: ignore[method-assign]
    for msg in read_jsonl(fixture):
        engine.feed(msg)
    engine.finish()
    return out


def patch(db: Path, revs: list[tuple[int, int, int]]) -> int:
    con = sqlite3.connect(db)
    n = 0
    with con:
        for tid, old_gid, end_ms in revs:
            row = con.execute(
                "SELECT cam_id, local_track_id, start_ms, end_ms FROM appearances"
                " WHERE tracklet_id=?",
                (tid,),
            ).fetchone()
            if row is None:
                print("  thiếu tracklet", tid, "trong", db)
                continue
            cam, ltid, start, end = row
            con.execute("UPDATE appearances SET start_ms=? WHERE tracklet_id=?", (end_ms + 1, tid))
            con.execute(
                "INSERT INTO appearances (tracklet_id, global_id, cam_id, local_track_id, start_ms,"
                " end_ms, n_frames, cost, reason) VALUES (?,?,?,?,?,?,0,0,'provisional')",
                (SYN + tid, old_gid, cam, ltid, start, min(end_ms, end)),
            )
            n += 1
        # A provisional span may overlap another tracklet that later took the same Global ID on
        # the same camera (TrackEval rejects duplicate IDs in a frame). Conservative: give the
        # provisional part an ID of its own, i.e. score it as a split.
        rows = con.execute(
            "SELECT tracklet_id, global_id, cam_id, start_ms, end_ms FROM appearances"
            " WHERE reason='provisional'"
        ).fetchall()
        for tid, gid, cam, start, end in rows:
            clash = con.execute(
                "SELECT COUNT(*) FROM appearances WHERE global_id=? AND cam_id=?"
                " AND tracklet_id!=? AND start_ms<=? AND end_ms>=?",
                (gid, cam, tid, end, start),
            ).fetchone()[0]
            if clash:
                con.execute(
                    "UPDATE appearances SET global_id=? WHERE tracklet_id=?",
                    (900_000 + tid % SYN, tid),
                )
    con.close()
    return n


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(sys.argv[1])
    for variant in sys.argv[2:]:
        total = 0
        for sdir in sorted(root.iterdir()):
            work = sdir / "pipeline"
            if not (work / variant).is_dir():
                continue
            for run in ("r1", "r2", "r3"):
                revs = revisions(work / variant / "mct.yaml", work / "roi" / f"{run}.jsonl")
                total += patch(work / variant / f"{variant}_{run}" / "mct.db", revs)
        print(f"{variant}: {total} lần đổi ID khi xét lại, đã tách phần ID tạm")


if __name__ == "__main__":
    main()
