"""Test phần thuần của `eval/compare_oracle_tracker.py` (đọc summary, gộp trung bình ± std).

Phần điều phối tiến trình con (engine, TrackEval) không test ở đây — nó chạy trên fixture
thật và đã được đối chiếu với con số đo tay của phiên 22 (A r1: HOTA 15.609).
"""

from __future__ import annotations

import sqlite3

import pytest
from eval.compare_oracle_tracker import (
    REPORT_COLUMNS,
    aggregate,
    format_table,
    majority_person_map,
    parse_summary,
    write_identity_db,
)

from common.schema import Detection, FrameMessage
from tools.export_trackeval import load_global_ids

SUMMARY = "HOTA DetA AssA IDF1 IDs Dets\n15.609 24.199 10.281 20.349 392 34556\n"


def test_parse_summary_doc_dong_ten_cot_va_dong_gia_tri():
    m = parse_summary(SUMMARY)
    assert m["HOTA"] == pytest.approx(15.609)
    assert m["IDs"] == 392.0


def test_parse_summary_lech_so_cot_thi_bao_loi():
    with pytest.raises(ValueError, match="giá trị"):
        parse_summary("HOTA DetA\n1.0\n")


def test_parse_summary_thieu_dong_gia_tri_thi_bao_loi():
    with pytest.raises(ValueError):
        parse_summary("HOTA DetA\n")


def _metrics(hota: float) -> dict[str, float]:
    return {c: hota for c in REPORT_COLUMNS}


def test_aggregate_nhom_theo_kich_ban_truoc_dau_hai_cham():
    agg = aggregate(
        {"A:r1": _metrics(10.0), "A:r2": _metrics(12.0), "B:r1": _metrics(30.0)}, REPORT_COLUMNS
    )
    assert agg["A"]["n"] == 2 and agg["B"]["n"] == 1
    mean, std = agg["A"]["HOTA"]
    assert mean == pytest.approx(11.0)
    assert std == pytest.approx(2**0.5)  # độ lệch chuẩn MẪU (ddof=1) của {10, 12}
    assert agg["B"]["HOTA"] == (30.0, None)  # n=1 thì không có std, đừng bịa 0


def test_format_table_khong_in_std_khi_n_bang_1():
    table = format_table(aggregate({"B:r1": _metrics(30.0)}, REPORT_COLUMNS), REPORT_COLUMNS)
    assert "±" not in table
    assert "| B | 1 | 30.00 |" in table


# --------------------------------------------------------------------------------------
# Trần liên kết hoàn hảo (oracle linker)
# --------------------------------------------------------------------------------------


def _msg(frame: int, ids: list[int], cam: str = "cam01") -> FrameMessage:
    return FrameMessage(
        cam_id=cam,
        frame_id=frame,
        ts_ms=1000 + frame * 500,
        frame_pts_ns=frame * 500_000_000,
        frame_width=1920,
        frame_height=1080,
        detections=[Detection(i, (10.0 * k, 0.0, 5.0, 9.0), 0.9) for k, i in enumerate(ids)],
        embed_dim=0,
    )


def test_majority_person_map_gop_manh_vo_va_lay_nguoi_da_so_cua_id_tron():
    # Tracker: id 5 rồi id 9 đều là người 11 (vỡ); id 7 là người 12 hai khung, người 13 một khung.
    tracker = [_msg(0, [5, 7]), _msg(1, [9, 7]), _msg(2, [9, 7])]
    oracle = [_msg(0, [11, 12]), _msg(1, [11, 12]), _msg(2, [11, 13])]
    m = majority_person_map(tracker, oracle)
    assert m == {("cam01", 5): 11, ("cam01", 9): 11, ("cam01", 7): 12}


def test_majority_person_map_hoa_phieu_lay_person_nho_hon():
    tracker = [_msg(0, [1]), _msg(1, [1])]
    oracle = [_msg(0, [20]), _msg(1, [10])]
    assert majority_person_map(tracker, oracle) == {("cam01", 1): 10}


def test_majority_person_map_hai_fixture_lech_hang_thi_bao_loi():
    with pytest.raises(ValueError, match="lệch hàng"):
        majority_person_map([_msg(0, [1, 2])], [_msg(0, [1])])
    with pytest.raises(ValueError, match="số message"):
        majority_person_map([_msg(0, [1])], [])


def test_write_identity_db_doc_lai_duoc_bang_load_global_ids(tmp_path):
    """Phải đi đúng cửa của bộ xuất TrackEval — lệch schema là chấm trần bằng bảng rỗng."""
    db = tmp_path / "ceil.db"
    write_identity_db({("cam01", 5): 11, ("cam02", 5): 11, ("cam01", 7): 12}, db)
    index = load_global_ids(db)
    assert index.get("cam01", 5, 123_456_789_012) == 11
    assert index.get("cam02", 5, 0) == 11  # cùng người ở camera khác -> cùng Global ID
    assert index.get("cam01", 7, 5) == 12
    assert index.get("cam01", 99, 5) is None
    write_identity_db({("cam01", 5): 1}, db)  # ghi đè, không cộng dồn
    con = sqlite3.connect(db)
    try:
        assert con.execute("SELECT COUNT(*) FROM appearances").fetchone()[0] == 1
    finally:
        con.close()


def test_majority_person_map_hai_track_trung_khung_cung_nguoi_thi_track_yeu_nhan_id_moi():
    """Ràng buộc loại trừ: hai track cùng camera cùng có mặt ở một khung không chung Global ID.

    Track 5 (người 11) chạy khung 0-2; track 8 cũng có người 11 là đa số nhưng chỉ ở khung
    1-2 và ít phiếu hơn, lại TRÙNG khung với track 5 -> track 8 phải nhận ID mới. Track 9
    nối tiếp track 5 (khung 3) không trùng khung nào -> vẫn gộp về 11.
    """
    tracker = [_msg(0, [5]), _msg(1, [5, 8]), _msg(2, [5, 8]), _msg(3, [9])]
    oracle = [_msg(0, [11]), _msg(1, [11, 11]), _msg(2, [11, 11]), _msg(3, [11])]
    m = majority_person_map(tracker, oracle)
    assert m[("cam01", 5)] == 11
    assert m[("cam01", 9)] == 11
    assert m[("cam01", 8)] > 11  # ID mới, lớn hơn mọi personID


def test_majority_person_map_khong_gay_trung_id_trong_mot_khung():
    tracker = [_msg(0, [1, 2]), _msg(1, [1, 2])]
    oracle = [_msg(0, [7, 7]), _msg(1, [7, 7])]  # ép cả hai track cùng là người 7
    m = majority_person_map(tracker, oracle)
    assert m[("cam01", 1)] != m[("cam01", 2)]
