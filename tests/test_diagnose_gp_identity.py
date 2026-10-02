"""Test `eval/diagnose_gp_identity.py` — phân loại điểm mặt đất theo danh tính (phiên 27).

Đã đối chiếu với script gốc trên WildTrack R640 r1 (trùng từng số, phiên 33). Ở đây canh
logic phân loại trên khung tổng hợp: năm loại điểm, TP/FP theo ngưỡng mét, và độ vỡ chỉ đếm
người được ≥ 2 camera thấy.
"""

from __future__ import annotations

from eval.diagnose_gp_identity import classify_point, score_frame, shares


def _everywhere(_point) -> bool:
    return True


def test_nam_loai_diem():
    person_gids = {1: {10}, 2: {20, 21}}
    assert classify_point([("cam01", (0, 0), -1)], person_gids) == "junk"
    assert classify_point([("cam01", (0, 0), 1), ("cam02", (0, 0), 2)], person_gids) == "mixed"
    assert classify_point([("cam01", (0, 0), 1), ("cam02", (0, 0), -1)], person_gids) == "partial"
    assert classify_point([("cam01", (0, 0), 2)], person_gids) == "split"
    assert classify_point([("cam01", (0, 0), 1)], person_gids) == "solo"


def test_mot_khung():
    # người 1 ở (0, 0): cam01 dưới gid 10, cam02 dưới gid 11 → bị tách qua camera
    # người 2 ở (5, 5): một gid 20, thấy ở 2 camera → liên kết đúng
    # gid 30: hộp báo nhầm ở (9, 9)
    frame = {
        10: [("cam01", (0.1, 0.0), 1)],
        11: [("cam02", (0.2, 0.1), 1)],
        20: [("cam01", (5.0, 5.0), 2), ("cam02", (5.2, 5.0), 2)],
        30: [("cam03", (9.0, 9.0), -1)],
    }
    gt = {1: (0.0, 0.0), 2: (5.0, 5.0)}
    frag, tp, fp = score_frame(frame, gt, threshold_m=1.0, area=_everywhere)
    assert frag == {2: 1, 1: 1}  # người 1 nằm trong 2 gid, người 2 trong 1 gid
    # người 1 chỉ ghép được với MỘT trong hai điểm 10/11; điểm còn lại là FP loại split
    assert tp == {"split": 1, "solo": 1}
    assert fp == {"split": 1, "junk": 1}
    assert shares({"FP": dict(fp)})["junk"] == 0.5


def test_ngoai_vung_chu_thich_khong_tinh():
    frame = {10: [("cam01", (100.0, 100.0), -1)]}
    _, tp, fp = score_frame(frame, {}, area=lambda p: p[0] < 50)
    assert not tp and not fp


def test_xa_qua_nguong_la_fp_solo():
    frame = {10: [("cam01", (2.0, 0.0), 1)]}
    _, tp, fp = score_frame(frame, {1: (0.0, 0.0)}, threshold_m=1.0, area=_everywhere)
    assert not tp
    assert fp == {"solo": 1}
