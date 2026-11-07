"""Re-ID rank-1 when the query sees only the first N frames of a tracklet (session 43).

Gallery = full descriptors of tracklets that started earlier, i.e. what the engine faces at its
first decision. Run from the repo root after `run_lab_eval` (reads data/lab/oracle/eval_gpu42).

    PYTHONPATH="src;." python eval/reid_lab_early.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from common.schema import read_jsonl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("data/lab/oracle/eval_gpu42")
SESS = [
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
NS = [3, 5, 10, 25, 50, 10**9]


def load(s, r):
    d = ROOT / s / "pipeline"
    tab = json.loads((d / "tables" / f"{r}.gt.json").read_text(encoding="utf-8"))["tracklets"]
    meta = {(t["cam_id"], int(t["local_track_id"])): t for t in tab}
    E = defaultdict(list)
    for m in read_jsonl(d / "roi" / f"{r}.jsonl"):
        for det in m.detections:
            k = (m.cam_id, int(det.local_track_id))
            if k in meta and det.embedding is not None:
                E[k].append((det.confidence, det.embedding))
    return [
        dict(cam=k[0], pid=meta[k]["gt_global_id"], s=meta[k]["start_ms"], e=meta[k]["end_ms"], E=v)
        for k, v in E.items()
        if len(v) >= 5
    ]


def nrm(v):
    return v / np.linalg.norm(v)


res = {n: dict(r1=[], fp=[], tp=[]) for n in NS}
for s in SESS:
    for r in ("r1", "r2", "r3"):
        T = load(s, r)
        for t in T:
            t["full"] = nrm(np.mean([e for _, e in t["E"]], axis=0))
        for q in T:
            gal = [
                g
                for g in T
                if g is not q
                and g["s"] < q["s"]
                and not (g["cam"] == q["cam"] and g["e"] >= q["s"])
            ]
            if not gal:
                continue
            lab = np.array([g["pid"] == q["pid"] for g in gal])
            for n in NS:
                es = q["E"][:n]
                # engine-like: top-k by confidence among the first n
                top = sorted(es, key=lambda x: -x[0])[:10]
                v = nrm(np.mean([e for _, e in top], axis=0))
                d = np.array([1 - float(v @ g["full"]) for g in gal])
                if lab.any():
                    res[n]["r1"].append(lab[d.argmin()])
                    res[n]["tp"].append(d[lab].min() < 0.40)
                if (~lab).any():
                    res[n]["fp"].append(d[~lab].min() < 0.40)
for n in NS:
    R = res[n]
    print(
        f"N={'cả tracklet' if n > 1e6 else n:>11} | rank-1 {np.mean(R['r1']):.1%} "
        f"(n={len(R['r1'])}) | người đúng < 0.40: {np.mean(R['tp']):.0%} | "
        f"có người sai < 0.40: {np.mean(R['fp']):.0%}"
    )
