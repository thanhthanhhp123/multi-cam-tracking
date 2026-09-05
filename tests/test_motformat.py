"""Test `common/motformat.py` — hợp đồng định dạng MOT / bố cục TrackEval.

Sai ở đây là loại lỗi tệ nhất: bảng điểm VẪN RA SỐ, chỉ là số sai, và không có gì báo. Hai
chỗ nguy hiểm nhất được canh riêng: khung đếm-từ-1 và offset của chuỗi ảo nối nhiều camera.
"""

from __future__ import annotations

import pytest

from common.motformat import (
    MotFormatError,
    MotRow,
    TrackEvalLayout,
    frame_offset_for,
    from_mot_frame,
    parse_mot,
    to_mot_frame,
    virtual_frame,
    write_gt,
    write_results,
    write_seqinfo,
    write_seqmap,
)

# --------------------------------------------------------------------------------------
# Khung đếm từ 1
# --------------------------------------------------------------------------------------


def test_khung_mot_dem_tu_mot():
    """`frame_id` 0 của schema là khung 1 của MOT. Quên +1 là lệch cả bảng điểm."""
    assert to_mot_frame(0) == 1
    assert to_mot_frame(399) == 400


def test_doi_qua_lai_khong_mat_mat():
    for frame_id in (0, 1, 42, 9999):
        assert from_mot_frame(to_mot_frame(frame_id)) == frame_id


def test_khung_am_bi_tu_choi():
    with pytest.raises(MotFormatError):
        to_mot_frame(-1)
    with pytest.raises(MotFormatError):
        from_mot_frame(0)  # MOT không có khung 0


# --------------------------------------------------------------------------------------
# Chuỗi ảo nối nhiều camera
# --------------------------------------------------------------------------------------


def test_camera_dau_tien_giu_nguyen_so_khung():
    assert virtual_frame(0, 0, offset=1000) == 1
    assert virtual_frame(0, 99, offset=1000) == 100


def test_cac_camera_khong_dam_len_nhau():
    """Khung cuối của camera i phải nhỏ hơn khung đầu của camera i+1."""
    cuoi_cam0 = virtual_frame(0, 999, offset=1000)
    dau_cam1 = virtual_frame(1, 0, offset=1000)
    assert cuoi_cam0 < dau_cam1


def test_frame_id_vuot_offset_bi_tu_choi():
    """Đây chính là cái bẫy: offset nhỏ hơn độ dài camera thì hai camera trộn khung vào
    nhau, chỉ số vẫn tính ra được và vẫn sai."""
    with pytest.raises(MotFormatError, match="dẫm lên"):
        virtual_frame(0, 1000, offset=1000)


def test_offset_tu_suy_lon_hon_camera_dai_nhat():
    assert frame_offset_for([400, 400, 400]) > 400
    assert frame_offset_for([150_000]) > 150_000
    assert frame_offset_for([]) > 0


def test_offset_bam_sat_do_dai_that():
    """TrackEval duyệt MỌI timestep của chuỗi ảo — offset thừa là thời gian chấm vứt đi.

    Đo được: offset 100000 cho 400 khung/camera biến 2800 khung thật thành 700000 timestep
    và một lần chấm mất ~1 phút; bám sát độ dài thì còn ~3 giây.
    """
    assert frame_offset_for([400] * 7) == 1000
    assert frame_offset_for([1500]) == 10_000
    assert frame_offset_for([1000]) == 10_000  # phải LỚN HƠN, không được bằng
    assert frame_offset_for([]) == 1000


def test_offset_va_cam_index_phai_hop_le():
    with pytest.raises(MotFormatError):
        virtual_frame(-1, 0, offset=1000)
    with pytest.raises(MotFormatError):
        virtual_frame(0, 0, offset=0)


# --------------------------------------------------------------------------------------
# Dòng dữ liệu
# --------------------------------------------------------------------------------------


def test_dong_ground_truth_du_cot_va_dung_thu_tu():
    row = MotRow(frame=1, track_id=7, x=10.0, y=20.0, w=30.0, h=40.0)
    assert row.as_gt_line() == "1,7,10.00,20.00,30.00,40.00,1,1,1.00"


def test_dong_ket_qua_giu_confidence_va_ba_cot_cuoi_am_mot():
    row = MotRow(frame=2, track_id=9, x=1.5, y=2.5, w=3.5, h=4.5, confidence=0.87)
    assert row.as_result_line() == "2,9,1.50,2.50,3.50,4.50,0.870,-1,-1,-1"


def test_ghi_roi_doc_lai_khong_doi(tmp_path):
    rows = [
        MotRow(3, 1, 0.0, 0.0, 10.0, 20.0, 0.9),
        MotRow(1, 2, 5.0, 6.0, 7.0, 8.0, 0.5),
    ]
    path = tmp_path / "res.txt"
    assert write_results(path, rows) == 2

    doc = parse_mot(path)
    assert [(r.frame, r.track_id) for r in doc] == [(1, 2), (3, 1)]  # đã sắp theo khung
    assert doc[0].confidence == pytest.approx(0.5)
    assert (doc[1].x, doc[1].y, doc[1].w, doc[1].h) == (0.0, 0.0, 10.0, 20.0)


def test_sap_xep_theo_khung_roi_theo_id(tmp_path):
    """TrackEval không đòi thứ tự, nhưng file sắp sẵn thì diff giữa hai lần chạy đọc được."""
    path = tmp_path / "gt.txt"
    write_gt(path, [MotRow(2, 9, 0, 0, 1, 1), MotRow(2, 3, 0, 0, 1, 1), MotRow(1, 5, 0, 0, 1, 1)])
    assert [(r.frame, r.track_id) for r in parse_mot(path)] == [(1, 5), (2, 3), (2, 9)]


def test_file_thieu_cot_bao_loi_kem_so_dong(tmp_path):
    path = tmp_path / "hong.txt"
    path.write_text("1,2,3\n", encoding="utf-8")
    with pytest.raises(MotFormatError, match=r"hong\.txt:1"):
        parse_mot(path)


def test_dong_trong_duoc_bo_qua(tmp_path):
    path = tmp_path / "res.txt"
    path.write_text("1,1,0,0,10,20,1\n\n\n2,1,0,0,10,20,1\n", encoding="utf-8")
    assert len(parse_mot(path)) == 2


def test_file_ghi_ra_dung_lf_du_tren_windows(tmp_path):
    """File này được TrackEval đọc trên Linux; CRLF lọt vào là rắc rối không triệu chứng."""
    path = tmp_path / "gt.txt"
    write_gt(path, [MotRow(1, 1, 0, 0, 10, 20)])
    assert b"\r\n" not in path.read_bytes()


# --------------------------------------------------------------------------------------
# Bố cục thư mục
# --------------------------------------------------------------------------------------


def test_bo_cuc_dung_chuan_motchallenge(tmp_path):
    lay = TrackEvalLayout(root=tmp_path, benchmark="MCT", split="mct")
    assert lay.dataset == "MCT-mct"
    assert lay.gt_file("all") == tmp_path / "gt" / "MCT-mct" / "all" / "gt" / "gt.txt"
    assert lay.seqinfo_file("all") == tmp_path / "gt" / "MCT-mct" / "all" / "seqinfo.ini"
    assert lay.seqmap_file() == tmp_path / "gt" / "seqmaps" / "MCT-mct.txt"
    assert (
        lay.result_file("mct-engine", "all")
        == tmp_path / "trackers" / "MCT-mct" / "mct-engine" / "data" / "all.txt"
    )


def test_seqinfo_co_seq_length(tmp_path):
    """TrackEval đọc `seqLength` để biết chuỗi dài bao nhiêu; thiếu là nó bỏ chuỗi."""
    path = tmp_path / "seqinfo.ini"
    write_seqinfo(path, name="cam01", width=1920, height=1080, length=400, fps=25.0)
    noi_dung = path.read_text(encoding="utf-8")
    assert "seqLength=400" in noi_dung
    assert "imWidth=1920" in noi_dung
    assert "frameRate=25" in noi_dung


def test_seqmap_co_dong_tieu_de(tmp_path):
    path = tmp_path / "MCT-sct.txt"
    write_seqmap(path, ["cam01", "cam02"])
    assert path.read_text(encoding="utf-8").splitlines() == ["name", "cam01", "cam02"]
