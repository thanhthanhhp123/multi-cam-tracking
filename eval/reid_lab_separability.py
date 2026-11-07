"""Re-ID separability on lab data, using the exact OSNet embeddings the engine sees.

For every session/run: pipeline tracklets mapped to a GT person (tables/rN.gt.json, purity >= 0.7),
descriptor = L2-normalised mean of the tracklet's embeddings. Reports retrieval (rank-1, mAP vs.
chance) and pair-distance separability (AUC, rates under the engine threshold).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from common.schema import read_jsonl

ROOT = Path("data/lab/oracle/eval_gpu42")
SESSIONS = [
    "s1_1",
    "s2_1",
    "s2_1b",
    "s2_1c",
    "s2_2",
    "s2_2b",
    "s2_2c",
    "s2_3",
    "s2_3b",
    "s3_1",
    "s3_2",
    "s4_1",
]
OVERLAP = {frozenset(("cam01", "cam02"))}
THR = 0.40  # association.max_cost of c0


def tracklets(session: str, run: str):
    d = ROOT / session / "pipeline"
    tab = json.loads((d / "tables" / f"{run}.gt.json").read_text(encoding="utf-8"))["tracklets"]
    meta = {(t["cam_id"], int(t["local_track_id"])): t for t in tab}
    embs: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for msg in read_jsonl(d / "roi" / f"{run}.jsonl"):
        for det in msg.detections:
            key = (msg.cam_id, int(det.local_track_id))
            if key in meta and det.embedding is not None:
                embs[key].append(det.embedding)
    out = []
    for key, es in embs.items():
        if len(es) < 5:
            continue
        v = np.mean(es, axis=0)
        out.append(
            dict(
                cam=key[0],
                pid=meta[key]["gt_global_id"],
                s=meta[key]["start_ms"],
                e=meta[key]["end_ms"],
                v=v / np.linalg.norm(v),
            )
        )
    return out


def kind(a, b):
    if a["cam"] == b["cam"]:
        return "same_camera"
    return "overlap" if frozenset((a["cam"], b["cam"])) in OVERLAP else "non_overlap"


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    # P(dist_pos < dist_neg)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    rp = ranks[: len(pos)].sum()
    return 1 - (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    groups = {
        "tất cả": SESSIONS,
        "s2 (2–4 người)": [s for s in SESSIONS if s.startswith("s2")],
        "s3 (đông)": ["s3_1", "s3_2"],
    }
    per_session = {}
    for s in SESSIONS:
        per_session[s] = [tracklets(s, r) for r in ("r1", "r2", "r3")]
    for gname, ss in groups.items():
        dist = defaultdict(lambda: {"pos": [], "neg": []})
        r1 = defaultdict(list)
        ap = defaultdict(list)
        chance = defaultdict(list)
        hard = defaultdict(list)
        for s in ss:
            for T in per_session[s]:
                for i, q in enumerate(T):
                    for mode in ("cross", "same_camera"):
                        if mode == "cross":
                            gal = [g for g in T if g["cam"] != q["cam"]]
                        else:  # sequential tracklets on the same camera (no time overlap)
                            gal = [
                                g
                                for j, g in enumerate(T)
                                if j != i
                                and g["cam"] == q["cam"]
                                and (g["e"] < q["s"] or g["s"] > q["e"])
                            ]
                        if not gal:
                            continue
                        d = np.array([1 - float(q["v"] @ g["v"]) for g in gal])
                        lab = np.array([g["pid"] == q["pid"] for g in gal])
                        if not lab.any():
                            continue
                        o = d.argsort()
                        hits = lab[o]
                        r1[mode].append(hits[0])
                        prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
                        ap[mode].append(prec[hits].mean())
                        chance[mode].append(lab.mean())
                        if (~lab).any():
                            hard[mode].append(d[~lab].min() < d[lab].min())
                for i in range(len(T)):
                    for j in range(i + 1, len(T)):
                        a, b = T[i], T[j]
                        k = kind(a, b)
                        if k == "same_camera" and not (a["e"] < b["s"] or b["e"] < a["s"]):
                            continue
                        dd = 1 - float(a["v"] @ b["v"])
                        dist[k]["pos" if a["pid"] == b["pid"] else "neg"].append(dd)
        print(f"\n## {gname}")
        for mode in ("cross", "same_camera"):
            if r1[mode]:
                print(
                    f"{mode:12s} n={len(r1[mode]):4d} rank-1 {np.mean(r1[mode]):.1%} "
                    f"(đoán bừa {np.mean(chance[mode]):.1%})  mAP {np.mean(ap[mode]):.1%}  "
                    f"người sai gần nhất gần hơn người đúng gần nhất: {np.mean(hard[mode]):.1%}"
                )
        for k in ("overlap", "non_overlap", "same_camera"):
            p, n = dist[k]["pos"], dist[k]["neg"]
            if p and n:
                print(
                    f"  {k:12s} cặp đúng {len(p):4d} trung vị {np.median(p):.2f} | "
                    f"cặp sai {len(n):5d} trung vị {np.median(n):.2f} | AUC {auc(p, n):.3f} | "
                    f"<{THR}: đúng {np.mean(np.array(p) < THR):.0%}, "
                    f"sai {np.mean(np.array(n) < THR):.0%}"
                )


if __name__ == "__main__":
    main()
