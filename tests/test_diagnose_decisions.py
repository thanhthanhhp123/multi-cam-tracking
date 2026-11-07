from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eval import diagnose_decisions as dd


@dataclass
class _Ref:
    cam_id: str
    local_track_id: int
    start_ms: int
    end_ms: int


def test_pair_kind():
    overlaps = {"cam01": {"cam02"}, "cam02": {"cam01"}, "cam03": set()}
    assert dd.pair_kind(overlaps, None, "cam01") == "-"
    assert dd.pair_kind(overlaps, "cam01", "cam01") == "same_camera"
    assert dd.pair_kind(overlaps, "cam01", "cam02") == "overlap"
    assert dd.pair_kind(overlaps, "cam01", "cam03") == "non_overlap"


def test_person_of_bo_phieu_theo_thoi_luong():
    table = {("cam01", 1): 7, ("cam02", 4): 9}
    members = [_Ref("cam01", 1, 0, 1000), _Ref("cam02", 4, 0, 5000), _Ref("cam03", 2, 0, 9000)]
    assert dd.person_of(members, table) == 9
    assert dd.person_of([_Ref("cam03", 2, 0, 10)], table) is None


def test_split_reason():
    assert dd.split_reason({"true_reason": "ràng buộc loại trừ"}) == "infeasible:exclusion"
    assert dd.split_reason({"true_reason": "đi quá nhanh"}) == "infeasible:time"
    assert dd.split_reason({"true_cost": 0.5, "true_limit": 0.4}) == "over_threshold"
    assert dd.split_reason({"true_cost": 0.3, "true_limit": 0.4}) == "lost_to_other"


def test_load_overlaps_tu_topology_lab():
    overlaps = dd.load_overlaps(Path("configs/lab/topology.yaml"))
    assert "cam02" in overlaps["cam01"]
    assert not overlaps["cam03"] & {"cam01", "cam02"}
