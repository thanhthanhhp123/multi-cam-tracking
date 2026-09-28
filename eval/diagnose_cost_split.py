"""Tracklet bị ngưỡng đẩy sang Global ID mới: thua vì NGOẠI HÌNH hay vì HÌNH HỌC?

    F=data/fixtures
    PYTHONPATH="src;." ~/.venvs/mct-test/Scripts/python.exe -m eval.diagnose_cost_split \\
        --config configs/demo/wildtrack_ds.mct.yaml \\
        --topology configs/demo/wildtrack.topology.yaml \\
        --homography-dir configs/cameras/homography/wildtrack \\
        --run r1 $F/ds_wildtrack_7cam_r640n_r1.jsonl $F/ds_wildtrack_7cam_r640n_r1.gt.json

Bước 1 của phiên 27 và bước đầu của "gói 2" (hợp nhất trên mặt đất). Phiên 27 thấy 38% điểm
sai là một người bị tách thành nhiều Global ID, và ~300 tracklet mỗi lần chạy nhận ID mới vì
ứng viên tốt nhất vượt `max_cost`. Chi phí là `(1 − cos) + λ·d_ground (+ nối mảnh cùng
camera)`, nhưng bảng `appearances.reason` chỉ ghi tổng. Công cụ này chạy lại engine (đường
online, tất định), chặn MỌI ma trận chi phí mà `Associator` dựng, và với từng hàng:

- nhãn người thật của tracklet (bảng `.gt.json`, khoá `(cam_id, local_track_id)`);
- nhãn của từng GlobalTrack ứng viên = người chiếm đa số theo THỜI LƯỢNG các tracklet thành
  viên của nó TẠI LÚC ĐÓ;
- tách chi phí từng ô: `app = 1 − similarity` (tính lại đúng như `_pair_cost`), `geo = cost − app`.

Rồi phân loại các ca "vượt ngưỡng" (ứng viên tốt nhất ≥ `max_cost`):

- `no_true_track`: chưa có GlobalTrack nào của người này → tạo ID mới là ĐÚNG;
- `true_infeasible`: có, nhưng ô đó bị ràng buộc loại (loại trừ / topology / hình học);
- `true_feasible`: có và khả thi, nhưng đắt hơn ngưỡng → tách người. Chia tiếp theo thành
  phần: `app ≥ max_cost` (ngoại hình một mình đã đủ loại) hay ngoại hình dưới ngưỡng và
  phần hình học đẩy qua.

Và với các ca ĐÃ ghép (tốt nhất < ngưỡng) thì đo tỉ lệ ghép đúng người — để biết nới ngưỡng
cho riêng một loại ca có an toàn không.

Chỉ đo, không sửa engine. Tracklet không có nhãn (hộp detector không khớp ai) bị bỏ qua.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.schema import read_jsonl
from mct.__main__ import build_engine, load_config
from mct.affinity import CostMatrix
from mct.gallery import GlobalTrack
from tools.export_trackeval import load_gt_table


@dataclass
class RowRecord:
    """Một hàng của một ma trận chi phí: một tracklet so với mọi GlobalTrack đang mở."""

    cam_id: str
    person: int
    n_frames: int
    best_cost: float
    best_app: float
    best_geo: float
    best_is_true: bool
    second_cost: float | None
    """Chi phí ô rẻ thứ nhì (khả thi) của hàng — độ chênh với `best_cost` đo mức mơ hồ."""
    true_state: str
    """`none` (chưa có track của người này) | `infeasible` | `feasible`."""
    true_cost: float | None
    true_app: float | None
    true_geo: float | None
    true_rank: int | None
    """Hạng của ô đúng người trong các ô khả thi của hàng (1 = rẻ nhất)."""


def track_person(track: GlobalTrack, table: dict[tuple[str, int], int]) -> int | None:
    """Người chiếm đa số theo thời lượng trong các tracklet thành viên của GlobalTrack."""
    votes: Counter[int] = Counter()
    for ref in track.members:
        person = table.get((ref.cam_id, ref.local_track_id))
        if person is not None:
            votes[person] += max(1, ref.end_ms - ref.start_ms)
    return votes.most_common(1)[0][0] if votes else None


def split_rows(
    matrix: CostMatrix,
    table: dict[tuple[str, int], int],
    *,
    similarity_mode: str,
    topk_query: int,
) -> list[RowRecord]:
    """Tách chi phí từng hàng có nhãn của một ma trận chi phí."""
    if not matrix.tracks:
        return []
    persons = [track_person(t, table) for t in matrix.tracks]
    out: list[RowRecord] = []
    for i, tracklet in enumerate(matrix.tracklets):
        person = table.get((tracklet.cam_id, tracklet.local_track_id))
        if person is None:
            continue
        costs = matrix.costs[i]
        finite = np.flatnonzero(np.isfinite(costs))
        if finite.size == 0:
            continue
        query = tracklet.query_embedding(topk_query)

        def parts(
            j: int, costs: np.ndarray = costs, query: np.ndarray | None = query
        ) -> tuple[float, float, float]:
            cost = float(costs[j])
            sim = matrix.tracks[j].similarity(query, similarity_mode)  # type: ignore[arg-type]
            app = 1.0 - float(sim)
            return cost, app, cost - app

        best_j = int(finite[np.argmin(costs[finite])])
        b_cost, b_app, b_geo = parts(best_j)
        ordered = np.sort(costs[finite])
        second = float(ordered[1]) if ordered.size > 1 else None

        true_cols = [j for j, p in enumerate(persons) if p == person]
        feasible_true = [j for j in true_cols if np.isfinite(costs[j])]
        if not true_cols:
            state, t = "none", None
        elif not feasible_true:
            state, t = "infeasible", None
        else:
            state = "feasible"
            t = min(feasible_true, key=lambda j: costs[j])
        if t is not None:
            t_cost, t_app, t_geo = parts(t)
            rank = int(np.sum(costs[finite] < costs[t])) + 1
        else:
            t_cost = t_app = t_geo = None
            rank = None
        out.append(
            RowRecord(
                cam_id=tracklet.cam_id,
                person=person,
                n_frames=tracklet.n_frames,
                best_cost=b_cost,
                best_app=b_app,
                best_geo=b_geo,
                best_is_true=persons[best_j] == person,
                second_cost=second,
                true_state=state,
                true_cost=t_cost,
                true_app=t_app,
                true_geo=t_geo,
                true_rank=rank,
            )
        )
    return out


def run(
    config: dict[str, Any],
    fixture: Path,
    table: dict[tuple[str, int], int],
    *,
    topology: Path | None,
    homography_dir: Path | None,
) -> tuple[list[RowRecord], float]:
    """Chạy engine online; trả mọi hàng của MỌI ma trận chi phí đã dựng, và `max_cost`."""
    engine = build_engine(
        config, topology_path=topology, homography_dir=homography_dir, db_path=":memory:"
    )
    associator = engine.associator
    affinity = associator.config
    rows: list[RowRecord] = []
    original = associator.cost_matrix

    def spy(tracklets, tracks=None):  # type: ignore[no-untyped-def]
        matrix = original(tracklets, tracks)
        rows.extend(
            split_rows(
                matrix,
                table,
                similarity_mode=affinity.similarity_mode,
                topk_query=affinity.topk_query,
            )
        )
        return matrix

    associator.cost_matrix = spy  # type: ignore[method-assign]
    for msg in read_jsonl(fixture):
        engine.feed(msg)
    engine.finish()
    if engine.store is not None:
        engine.store.close()
    return rows, float(affinity.max_cost)


def _q(values: list[float]) -> str:
    if not values:
        return "—"
    qs = np.quantile(values, [0.1, 0.5, 0.9])
    return f"{qs[0]:.2f} / {qs[1]:.2f} / {qs[2]:.2f}"


def summarize(rows: list[RowRecord], max_cost: float, homography_weight: float) -> dict[str, Any]:
    over = [r for r in rows if r.best_cost >= max_cost]
    under = [r for r in rows if r.best_cost < max_cost]
    feas = [r for r in over if r.true_state == "feasible"]
    app_alone = [r for r in feas if r.true_app is not None and r.true_app >= max_cost]
    pushed = [r for r in feas if r not in app_alone]
    lam = homography_weight or 1.0
    return {
        "n_rows": len(rows),
        "under": {
            "n": len(under),
            "best_is_true": sum(r.best_is_true for r in under),
        },
        "over": {
            "n": len(over),
            "states": dict(Counter(r.true_state for r in over)),
            "true_feasible": {
                "n": len(feas),
                "true_is_best": sum(r.true_rank == 1 for r in feas),
                "app_alone_over": len(app_alone),
                "pushed_by_geo": len(pushed),
                "true_app_q10_50_90": _q([r.true_app for r in feas if r.true_app is not None]),
                "true_geo_m_q10_50_90": _q(
                    [r.true_geo / lam for r in feas if r.true_geo is not None]
                ),
                "true_cost_q10_50_90": _q([r.true_cost for r in feas if r.true_cost is not None]),
                "wrong_best_app_q10_50_90": _q([r.best_app for r in feas if not r.best_is_true]),
                "wrong_best_geo_m_q10_50_90": _q(
                    [r.best_geo / lam for r in feas if not r.best_is_true]
                ),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run", nargs=3, action="append", required=True, metavar=("NHÃN", "FIXTURE", "BẢNG_GT")
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--topology", type=Path, default=None)
    p.add_argument("--homography-dir", type=Path, default=None)
    p.add_argument("--json", type=Path, default=Path("data/s28/cost_split.json"))
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    topology = args.topology if args.topology and args.topology.is_file() else None
    lam = float((config.get("association") or {}).get("homography_weight", 0.0))
    out: dict[str, Any] = {}
    for label, fixture, table_path in args.run:
        table = load_gt_table(Path(table_path))
        rows, max_cost = run(
            config, Path(fixture), table, topology=topology, homography_dir=args.homography_dir
        )
        summary = summarize(rows, max_cost, lam)
        out[label] = {"summary": summary, "rows": [asdict(r) for r in rows]}
        print(f"## {label}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
