"""Test cho `eval/diagnose_cost_split.py`.

Ghim ba điều mà bảng số của phiên 29 dựa vào:
- nhãn của GlobalTrack = người chiếm đa số THEO THỜI LƯỢNG, không theo số tracklet;
- tách chi phí đúng: `app = 1 − similarity`, `geo = cost − app`; ô `inf` không bao giờ là
  "tốt nhất" hay "tốt nhì";
- trạng thái của track đúng người phân biệt được `none` / `infeasible` / `feasible`, kèm hạng.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest
from eval.diagnose_cost_split import RowRecord, split_rows, summarize, track_person

from mct.affinity import CostMatrix
from mct.gallery import TrackletRef


@dataclass
class _Tracklet:
    cam_id: str
    local_track_id: int
    n_frames: int = 5

    def query_embedding(self, top_k: int | None = None) -> np.ndarray:
        return np.ones(4, dtype=np.float32)


@dataclass
class _Track:
    members: list[TrackletRef]
    sim: float = 0.5
    seen: list[str] = field(default_factory=list)

    def similarity(self, query: np.ndarray, mode: str = "max") -> float:
        self.seen.append(mode)
        return self.sim


def _ref(cam: str, tid: int, start: int, end: int) -> TrackletRef:
    return TrackletRef(tracklet_id=tid, cam_id=cam, local_track_id=tid, start_ms=start, end_ms=end)


def _track(*refs: TrackletRef, sim: float = 0.5) -> _Track:
    return _Track(members=list(refs), sim=sim)


INF = math.inf


def test_track_person_theo_thoi_luong_khong_theo_so_tracklet():
    # Người 2 có 2 tracklet ngắn, người 1 có 1 tracklet dài hơn tổng của cả hai.
    track = _track(_ref("c1", 1, 0, 5_000), _ref("c2", 2, 0, 1_000), _ref("c3", 3, 0, 1_000))
    table = {("c1", 1): 1, ("c2", 2): 2, ("c3", 3): 2}
    assert track_person(track, table) == 1  # type: ignore[arg-type]


def test_track_person_khong_nhan_tra_none():
    assert track_person(_track(_ref("c1", 9, 0, 100)), {}) is None  # type: ignore[arg-type]


def _matrix(tracklets, tracks, costs) -> CostMatrix:
    return CostMatrix(tracklets=tracklets, tracks=tracks, costs=np.array(costs, dtype=float))


def test_split_rows_tach_chi_phi_va_xep_hang_o_dung_nguoi():
    table = {("c1", 1): 10, ("c2", 5): 10, ("c3", 6): 20, ("c4", 7): 30}
    tracks = [
        _track(_ref("c3", 6, 0, 100), sim=0.7),  # người 20
        _track(_ref("c2", 5, 0, 100), sim=0.6),  # người 10 (đúng)
        _track(_ref("c4", 7, 0, 100), sim=0.9),  # người 30, bị loại
    ]
    matrix = _matrix([_Tracklet("c1", 1)], tracks, [[0.5, 0.8, INF]])
    rows = split_rows(matrix, table, similarity_mode="mean", topk_query=4)  # type: ignore[arg-type]

    assert len(rows) == 1
    r = rows[0]
    assert r.person == 10
    assert r.best_cost == pytest.approx(0.5)
    assert r.best_app == pytest.approx(0.3)
    assert r.best_geo == pytest.approx(0.2)
    assert not r.best_is_true
    assert r.second_cost == pytest.approx(0.8)  # ô inf không được tính là tốt nhì
    assert r.true_state == "feasible"
    assert r.true_cost == pytest.approx(0.8)
    assert r.true_app == pytest.approx(0.4)
    assert r.true_rank == 2
    assert all(m == "mean" for t in tracks for m in t.seen)


def test_split_rows_phan_biet_none_infeasible_va_bo_hang_khong_nhan():
    table = {("c1", 1): 10, ("c1", 2): 20, ("c2", 5): 10, ("c3", 6): 30}
    tracks = [_track(_ref("c2", 5, 0, 100)), _track(_ref("c3", 6, 0, 100))]
    tracklets = [
        _Tracklet("c1", 1),  # người 10: track đúng có nhưng bị chặn
        _Tracklet("c1", 2),  # người 20: chưa có track nào
        _Tracklet("c1", 3),  # không nhãn → bỏ
        _Tracklet("c1", 1),  # hàng toàn inf → bỏ
    ]
    matrix = _matrix(tracklets, tracks, [[INF, 0.6], [0.4, 0.7], [0.1, 0.1], [INF, INF]])
    rows = split_rows(matrix, table, similarity_mode="max", topk_query=4)  # type: ignore[arg-type]

    assert [r.true_state for r in rows] == ["infeasible", "none"]
    assert rows[0].second_cost is None
    assert rows[0].true_rank is None and rows[0].true_cost is None


def test_split_rows_khong_co_track_tra_rong():
    matrix = _matrix([_Tracklet("c1", 1)], [], np.zeros((1, 0)))
    assert split_rows(matrix, {("c1", 1): 1}, similarity_mode="max", topk_query=4) == []  # type: ignore[arg-type]


def _row(best: float, *, state: str, true_app: float | None = None, is_true: bool = False):
    return RowRecord(
        cam_id="c1", person=1, n_frames=5,
        best_cost=best, best_app=best, best_geo=0.0, best_is_true=is_true, second_cost=None,
        true_state=state,
        true_cost=None if true_app is None else true_app + 0.2,
        true_app=true_app, true_geo=None if true_app is None else 0.2,
        true_rank=None if true_app is None else 1,
    )  # fmt: skip


def test_summarize_chia_theo_nguong_va_theo_thanh_phan():
    rows = [
        _row(0.3, state="feasible", true_app=0.3, is_true=True),  # ghép đúng
        _row(0.4, state="none"),  # ghép nhầm người mới
        _row(1.2, state="feasible", true_app=1.0),  # ngoại hình một mình đã vượt
        _row(1.0, state="feasible", true_app=0.8),  # hình học đẩy qua ngưỡng
        _row(1.1, state="infeasible"),
    ]
    s = summarize(rows, max_cost=0.9, homography_weight=0.4)
    assert s["n_rows"] == 5
    assert s["under"] == {"n": 2, "best_is_true": 1}
    assert s["over"]["n"] == 3
    assert s["over"]["states"] == {"feasible": 2, "infeasible": 1}
    feas = s["over"]["true_feasible"]
    assert (feas["n"], feas["app_alone_over"], feas["pushed_by_geo"]) == (2, 1, 1)
    # geo 0.2 / λ 0.4 = 0.5 m
    assert feas["true_geo_m_q10_50_90"] == "0.50 / 0.50 / 0.50"
