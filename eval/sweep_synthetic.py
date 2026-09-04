"""Sweep tham số của engine liên kết trên fixture tổng hợp.

    PYTHONPATH=src python -m eval.sweep_synthetic

KHÔNG phải bộ đánh giá chính thức của đồ án — đó là TrackEval trên dữ liệu thật
(`eval/run_trackeval.py`, M6). Đây là phép đo nhanh, tái lập được, trả lời đúng một câu:
engine nhạy thế nào với `association.max_cost` — tham số nhạy nhất của cả hệ thống —
và ngưỡng hiện tại trong `configs/mct.yaml` nằm ở đâu so với vùng chạy đúng.

Chỉ số dùng ở đây tính theo CẶP lượt xuất hiện (không phải MOTA/IDF1 chuẩn):
  - P = trong các cặp bị gán chung Global ID, bao nhiêu phần trăm đúng là một người;
  - R = trong các cặp thật sự cùng người, bao nhiêu phần trăm được gán chung ID;
  - vỡ = số danh tính bị xé thành nhiều Global ID; gộp = số Global ID ôm nhiều người.
Đủ để so sánh tham số với nhau, và đọc thẳng ra được cái giá của từng lựa chọn.
"""

from collections import defaultdict

from mct.affinity import AffinityConfig
from mct.associator import assign_messages
from mct.gallery import GalleryConfig
from mct.topology import Topology
from mct.tracklet import TrackletConfig
from tools.make_synthetic_fixture import build_scenario

TOPO = Topology.from_mapping(
    {
        "cameras": {"cam01": {}, "cam02": {}},
        "transitions": [
            {"from": "cam01", "to": "cam02", "bidirectional": True, "min_ms": 3000, "max_ms": 15000}
        ],
    }
)


def score(results, gt):
    """(#global id, #danh tính bị vỡ, #global id gộp nhầm, precision, recall) theo cặp."""
    gid_of = {r.tracklet.key: r.global_id for r in results}
    keys = sorted(gid_of)
    tp = fp = fn = 0
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            same_gt = gt[a] == gt[b]
            same_gid = gid_of[a] == gid_of[b]
            if same_gt and same_gid:
                tp += 1
            elif same_gid:
                fp += 1
            elif same_gt:
                fn += 1
    per_identity = defaultdict(set)
    per_gid = defaultdict(set)
    for key, gid in gid_of.items():
        per_identity[gt[key]].add(gid)
        per_gid[gid].add(gt[key])
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (
        len(set(gid_of.values())),
        sum(1 for g in per_identity.values() if len(g) > 1),
        sum(1 for ids in per_gid.values() if len(ids) > 1),
        precision,
        recall,
        f1,
    )


def run(name, params):
    messages, appearances = build_scenario(**params)
    gt = {(a.cam_id, a.local_track_id): a.gt_global_id for a in appearances}
    print(
        f"\n=== {name}: {params['identities']} danh tính, "
        f"cross_cam_sim={params['cross_cam_sim']}, inter_sim={params['inter_sim']} "
        f"({len(appearances)} lượt xuất hiện) ==="
    )
    print(
        f"{'mode':9s} {'max_cost':>8s} {'#gid':>5s} {'vỡ':>4s} {'gộp':>4s} "
        f"{'P':>6s} {'R':>6s} {'F1':>6s}"
    )
    for mode in ("max", "centroid"):
        for max_cost in MAX_COSTS:
            results, _ = assign_messages(
                messages,
                topology=TOPO,
                tracklet_config=TrackletConfig(min_frames=5, idle_timeout_ms=2000),
                config=AffinityConfig(max_cost=max_cost, similarity_mode=mode),
                gallery_config=GalleryConfig(similarity_mode=mode),
            )
            n_gid, broken, merged, p, r, f1 = score(results, gt)
            print(
                f"{mode:9s} {max_cost:8.2f} {n_gid:5d} {broken:4d} {merged:4d} "
                f"{p:6.3f} {r:6.3f} {f1:6.3f}"
            )


BASE = dict(
    fps=15,
    embed_dim=256,
    intra_sim=0.80,
    dwell_s=6.0,
    transit_s=8.0,
    stagger_s=4.0,
    miss_rate=0.04,
    seed=42,
)

# Ba kịch bản: mặc định, "nhiều người mặc đồ giống nhau", và Re-ID yếu tới mức cùng một
# người ở hai camera còn kém giống hơn hai người khác nhau ở kịch bản trước.
SCENARIOS = [
    ("Dễ (mặc định)", dict(BASE, identities=3, cross_cam_sim=0.75, inter_sim=0.65)),
    (
        "Khó: nhiều người, đồ giống nhau",
        dict(BASE, identities=6, cross_cam_sim=0.75, inter_sim=0.72),
    ),
    ("Rất khó: Re-ID yếu", dict(BASE, identities=6, cross_cam_sim=0.65, inter_sim=0.62)),
]

MAX_COSTS = (0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)


def main() -> None:
    for name, params in SCENARIOS:
        run(name, params)


if __name__ == "__main__":
    main()
