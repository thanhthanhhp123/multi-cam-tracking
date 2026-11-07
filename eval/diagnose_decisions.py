"""Root-cause association errors on lab data: log the FIRST decision the engine makes per tracklet.

Replays a `run_lab_eval` work dir (`<root>/<session>/<kind>/`: `roi/r1.jsonl` is the fixture after
the per-camera floor filter, `tables/r1.gt.json` maps `(cam_id, local_track_id)` to a GT person,
`<variant>/mct.yaml` is the exact engine config of that variant) and hooks `Gallery.assign` /
`Gallery.create`. Each tracklet's first decision is one of:

- `merge_ok`    joined the GlobalTrack of the same GT person,
- `merge_wrong` joined another person's GlobalTrack,
- `new_ok`      created a new Global ID and no open track of this person existed,
- `split`       created a new Global ID although this person's track was open.

For every decision the cost-matrix row is kept: cost and budget (limit) of the chosen column and
of the true person's column, the true column's infeasibility reason, and the pair kind relative
to the track's last camera. A track's person is a duration-weighted vote over its member
tracklets. Session 41 used this to find that 27/45 wrong merges were same-camera stitches.

    PYTHONPATH="src;." python -m eval.diagnose_decisions data/lab/oracle/eval_v4 \\
        --kind pipeline --variant c0 --sessions s2_1,s2_1b --out data/s42/decisions_c0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from common.schema import read_jsonl
from mct.__main__ import build_engine, load_config
from tools.export_trackeval import load_gt_table

GtTable = dict[tuple[str, int], int]


def load_overlaps(topology: Path) -> dict[str, set[str]]:
    cameras = yaml.safe_load(topology.read_text(encoding="utf-8"))["cameras"]
    return {cam: set((spec or {}).get("overlaps_with") or []) for cam, spec in cameras.items()}


def pair_kind(overlaps: dict[str, set[str]], last_cam: str | None, cam: str) -> str:
    """Handover kind from the track's last camera to the tracklet's camera."""
    if last_cam is None:
        return "-"
    if last_cam == cam:
        return "same_camera"
    return "overlap" if cam in overlaps.get(last_cam, set()) else "non_overlap"


def person_of(members: Iterable[Any], table: GtTable) -> int | None:
    """Majority GT person of a track's member tracklets, weighted by duration."""
    votes: Counter[int] = Counter()
    for ref in members:
        person = table.get((ref.cam_id, ref.local_track_id))
        if person is not None:
            votes[person] += max(1, ref.end_ms - ref.start_ms)
    return votes.most_common(1)[0][0] if votes else None


def split_reason(record: dict[str, Any]) -> str:
    """Why the true person's column lost, for a `split` record."""
    if "true_reason" in record:
        text = record["true_reason"]
        if "loại trừ" in text:
            return "infeasible:exclusion"
        if "quá" in text:
            return "infeasible:time"
        if " m " in text or "m trên" in text or "mảnh" in text:
            return "infeasible:ground"
        return "infeasible:" + text[:40]
    if "true_cost" not in record:
        return "unknown"
    return "over_threshold" if record["true_cost"] >= record["true_limit"] else "lost_to_other"


def run_session(
    config: dict[str, Any],
    fixture: Path,
    table: GtTable,
    session: str,
    *,
    topology: Path,
    homography_dir: Path,
    overlaps: dict[str, set[str]],
) -> list[dict[str, Any]]:
    engine = build_engine(
        config, topology_path=topology, homography_dir=homography_dir, db_path=":memory:"
    )
    assoc = engine.associator
    gallery, cfg = assoc.gallery, assoc.config
    rowinfo: dict[int, dict[str, Any]] = {}
    decided: set[int] = set()
    records: list[dict[str, Any]] = []
    t0: list[int | None] = [None]
    original_cost_matrix = assoc.cost_matrix

    def cost_matrix(tracklets: Any, tracks: Any = None) -> Any:
        matrix = original_cost_matrix(tracklets, tracks)
        limits = matrix.limits(cfg) if matrix.tracks else None
        for i, tracklet in enumerate(matrix.tracklets):
            info: dict[str, Any] = {
                "tracks": list(matrix.tracks),
                "costs": matrix.costs[i].copy(),
                "limits": (
                    limits[i].copy()
                    if limits is not None
                    else np.full(len(matrix.tracks), cfg.max_cost)
                ),
                "reasons": {
                    j: matrix.reason(i, j)
                    for j in range(len(matrix.tracks))
                    if not np.isfinite(matrix.costs[i, j])
                },
                "app": {},
            }
            query = tracklet.query_embedding(cfg.topk_query)
            if query is not None:
                for j, track in enumerate(matrix.tracks):
                    info["app"][j] = 1.0 - float(track.similarity(query, cfg.similarity_mode))
            rowinfo[tracklet.tracklet_id] = info
        return matrix

    assoc.cost_matrix = cost_matrix  # type: ignore[method-assign]
    original_assign, original_create = gallery.assign, gallery.create

    def record(tracklet: Any, chosen: Any) -> None:
        if tracklet.tracklet_id in decided:
            return
        decided.add(tracklet.tracklet_id)
        person = table.get((tracklet.cam_id, tracklet.local_track_id))
        if person is None:
            return
        info = rowinfo.get(tracklet.tracklet_id)
        tracks = info["tracks"] if info else []
        persons = [person_of(t.members, table) for t in tracks]
        true_cols = [j for j, q in enumerate(persons) if q == person]
        rec: dict[str, Any] = {
            "session": session,
            "person": person,
            "cam": tracklet.cam_id,
            "t": round((tracklet.start_ms - (t0[0] or 0)) / 1000, 2),
        }
        if chosen is not None:
            col = next((j for j, t in enumerate(tracks) if t is chosen), None)
            chosen_person = person_of(chosen.members, table)
            rec["decision"] = "merge_ok" if chosen_person == person else "merge_wrong"
            rec["chosen_person"] = chosen_person
            rec["chosen_kind"] = pair_kind(overlaps, chosen.last_cam_id, tracklet.cam_id)
            if col is not None and info is not None:
                rec.update(
                    chosen_cost=float(info["costs"][col]),
                    chosen_limit=float(info["limits"][col]),
                    chosen_app=info["app"].get(col),
                )
        else:
            rec["decision"] = "split" if true_cols else "new_ok"
        if true_cols and info:
            feasible = [j for j in true_cols if np.isfinite(info["costs"][j])]
            col = (
                min(feasible, key=lambda c: info["costs"][c] / info["limits"][c])
                if feasible
                else true_cols[0]
            )
            track = tracks[col]
            rec.update(
                true_kind=pair_kind(overlaps, track.last_cam_id, tracklet.cam_id),
                true_pair=f"{track.last_cam_id}->{tracklet.cam_id}",
                true_gap_ms=int(tracklet.start_ms - track.last_seen_ms),
                true_app=info["app"].get(col),
                true_track_pure=all(
                    table.get((r.cam_id, r.local_track_id)) in (person, None) for r in track.members
                ),
            )
            if feasible:
                rec.update(
                    true_cost=float(info["costs"][col]), true_limit=float(info["limits"][col])
                )
            else:
                rec["true_reason"] = info["reasons"].get(col, "")
        records.append(rec)

    def assign(track: Any, tracklet: Any, *args: Any, **kwargs: Any) -> Any:
        record(tracklet, track)
        return original_assign(track, tracklet, *args, **kwargs)

    def create(tracklet: Any, *args: Any, **kwargs: Any) -> Any:
        record(tracklet, None)
        return original_create(tracklet, *args, **kwargs)

    gallery.assign, gallery.create = assign, create  # type: ignore[method-assign]
    for message in read_jsonl(fixture):
        if t0[0] is None:
            t0[0] = message.ts_ms
        engine.feed(message)
    engine.finish()
    return records


def report(records: list[dict[str, Any]]) -> list[str]:
    splits = [r for r in records if r["decision"] == "split"]
    return [
        f"decisions: {dict(Counter(r['decision'] for r in records))}",
        "merge_wrong by kind: "
        + str(dict(Counter(r["chosen_kind"] for r in records if r["decision"] == "merge_wrong"))),
        f"split by true_kind: {dict(Counter(r.get('true_kind') for r in splits))}",
        "split with contaminated true track: "
        + str(sum(1 for r in splits if not r.get("true_track_pure", True))),
        "split reasons: "
        + str(dict(Counter(f"{r.get('true_kind')}/{split_reason(r)}" for r in splits))),
    ]


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("root", type=Path, help="thư mục làm việc của run_lab_eval (chứa <buổi>/)")
    p.add_argument("--kind", default="pipeline", help="pipeline | oracle")
    p.add_argument("--variant", required=True, help="tên biến thể (thư mục chứa mct.yaml)")
    p.add_argument("--sessions", required=True, help="danh sách buổi, phân cách bằng dấu phẩy")
    p.add_argument("--run", default="r1", help="lần chạy pipeline (r1/r2/r3)")
    p.add_argument("--topology", type=Path, default=Path("configs/lab/topology.yaml"))
    p.add_argument("--homography-dir", type=Path, default=Path("configs/lab/homography"))
    p.add_argument("--out", type=Path, default=None, help="ghi mọi bản ghi ra JSON")
    args = p.parse_args(argv)

    overlaps = load_overlaps(args.topology)
    records: list[dict[str, Any]] = []
    for session in args.sessions.split(","):
        work = args.root / session / args.kind
        records += run_session(
            load_config(work / args.variant / "mct.yaml"),
            work / "roi" / f"{args.run}.jsonl",
            load_gt_table(work / "tables" / f"{args.run}.gt.json"),
            session,
            topology=args.topology,
            homography_dir=args.homography_dir,
            overlaps=overlaps,
        )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print("\n".join(report(records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
