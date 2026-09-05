"""Test `tools/export_trackeval.py` và `tools/cvat_to_mot.py` — hạ tầng đánh giá M6.

Hai công cụ này quyết định con số MOTA/IDF1/HOTA sẽ vào chương 6. Chúng không có "triệu
chứng khi sai": TrackEval vẫn chấm, vẫn in bảng, chỉ là bảng sai. Test vì thế soi vào đúng
những chỗ dễ lệch âm thầm — khung đếm từ 1, id dùng ở mỗi chế độ, và ánh xạ danh tính.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from common.motformat import TrackEvalLayout, parse_mot
from common.schema import Detection, FrameMessage
from tools.cvat_to_mot import CvatError, assign_global_ids, parse_cvat_video, write_ground_truth
from tools.export_trackeval import (
    GtSource,
    export_mct,
    export_sct,
    load_global_ids,
    load_gt_table,
)


def msg(cam_id: str, frame_id: int, dets: list[Detection]) -> FrameMessage:
    return FrameMessage(
        cam_id=cam_id,
        frame_id=frame_id,
        ts_ms=1000 + frame_id * 40,
        frame_pts_ns=frame_id * 40_000_000,
        frame_width=1920,
        frame_height=1080,
        detections=dets,
        embed_dim=0,
    )


def det(track_id: int, x: float = 10.0, conf: float = 0.9) -> Detection:
    return Detection(local_track_id=track_id, bbox=(x, 20.0, 30.0, 60.0), confidence=conf)


@pytest.fixture
def messages() -> list[FrameMessage]:
    return [
        msg("cam01", 0, [det(1), det(2, x=100.0)]),
        msg("cam01", 1, [det(1)]),
        msg("cam02", 0, [det(7, x=200.0)]),
    ]


# --------------------------------------------------------------------------------------
# Chế độ sct — mỗi camera một chuỗi, id = local_track_id
# --------------------------------------------------------------------------------------


def test_sct_moi_camera_mot_chuoi(tmp_path, messages):
    lay = TrackEvalLayout(root=tmp_path, benchmark="MCT", split="sct")
    seqs = export_sct(messages, {("cam01", 1): 5}, lay, tracker="t", fps=25.0)

    assert seqs == ["cam01", "cam02"]
    assert lay.result_file("t", "cam01").is_file()
    assert lay.result_file("t", "cam02").is_file()


def test_sct_dung_local_track_id_va_khung_dem_tu_mot(tmp_path, messages):
    lay = TrackEvalLayout(root=tmp_path, benchmark="MCT", split="sct")
    export_sct(messages, {("cam01", 1): 5, ("cam01", 2): 6}, lay, tracker="t", fps=25.0)

    rows = parse_mot(lay.result_file("t", "cam01"))
    assert [(r.frame, r.track_id) for r in rows] == [(1, 1), (1, 2), (2, 1)]


def test_sct_ground_truth_chi_lay_track_co_trong_bang(tmp_path, messages):
    """Track không được chú thích thì không phải là bỏ sót của hệ thống — đừng chấm nó."""
    lay = TrackEvalLayout(root=tmp_path, benchmark="MCT", split="sct")
    export_sct(messages, {("cam01", 1): 5}, lay, tracker="t", fps=25.0)

    assert {r.track_id for r in parse_mot(lay.gt_file("cam01"))} == {1}
    assert {r.track_id for r in parse_mot(lay.result_file("t", "cam01"))} == {1, 2}


def test_sct_seqinfo_dung_do_dai_va_kich_thuoc(tmp_path, messages):
    lay = TrackEvalLayout(root=tmp_path, benchmark="MCT", split="sct")
    export_sct(messages, {}, lay, tracker="t", fps=25.0)

    noi_dung = lay.seqinfo_file("cam01").read_text(encoding="utf-8")
    assert "seqLength=2" in noi_dung  # frame_id lớn nhất là 1 -> 2 khung
    assert "imWidth=1920" in noi_dung


def test_sct_confidence_am_bi_kep_ve_khong(tmp_path):
    """Target do tracker suy ra có confidence=-0.1 (phiên 9). MOT không nhận số âm."""
    lay = TrackEvalLayout(root=tmp_path, benchmark="MCT", split="sct")
    export_sct([msg("cam01", 0, [det(1, conf=-0.1)])], {}, lay, tracker="t", fps=25.0)
    assert parse_mot(lay.result_file("t", "cam01"))[0].confidence == 0.0


# --------------------------------------------------------------------------------------
# Chế độ mct — một chuỗi ảo, id = global_id
# --------------------------------------------------------------------------------------


def test_mct_dung_global_id_chu_khong_phai_local(tmp_path, messages):
    lay = TrackEvalLayout(root=tmp_path, benchmark="MCT", split="mct")
    gt = {("cam01", 1): 100, ("cam02", 7): 100}
    gids = {("cam01", 1): 55, ("cam02", 7): 55}
    _, stats = export_mct(messages, gt, gids, lay, tracker="t", fps=25.0)

    assert {r.track_id for r in parse_mot(lay.gt_file("all"))} == {100}
    assert {r.track_id for r in parse_mot(lay.result_file("t", "all"))} == {55}
    assert stats["n_cameras"] == 2


def test_mct_hai_camera_khong_dam_khung_len_nhau(tmp_path, messages):
    """Đây là chỗ thủ thuật chuỗi ảo hỏng lặng lẽ nếu offset sai."""
    lay = TrackEvalLayout(root=tmp_path, benchmark="MCT", split="mct")
    gt = {("cam01", 1): 100, ("cam02", 7): 200}
    _, stats = export_mct(messages, gt, {}, lay, tracker="t", fps=25.0)

    # Lấy khung ĐẦU của mỗi id: track 100 có mặt ở hai khung, dict comprehension thường
    # sẽ giữ khung cuối và làm test nói sai điều mình định nói.
    khung: dict[int, int] = {}
    for r in parse_mot(lay.gt_file("all")):
        khung.setdefault(r.track_id, r.frame)
    assert khung[100] == 1  # cam01 giữ nguyên
    assert khung[200] == stats["frame_offset"] + 1  # cam02 dời hẳn sang sau
    assert stats["frame_offset"] > 2


def test_mct_detection_chua_co_global_id_tinh_la_bo_sot(tmp_path, messages):
    """Engine chưa gán Global ID cho tracklet nào thì đó là bỏ sót thật, phải hiện ra ở
    recall — không được lặng lẽ thêm vào kết quả bằng id bịa."""
    lay = TrackEvalLayout(root=tmp_path, benchmark="MCT", split="mct")
    gt = {("cam01", 1): 100, ("cam02", 7): 200}
    _, stats = export_mct(messages, gt, {("cam01", 1): 55}, lay, tracker="t", fps=25.0)

    assert {r.track_id for r in parse_mot(lay.result_file("t", "all"))} == {55}
    # cam01 track 2 (một khung) và cam02 track 7 (một khung) đều chưa có Global ID.
    assert stats["n_unassigned"] == 2
    assert stats["n_gt"] == 3  # cam01 track1 x2 khung + cam02 track7


def test_mct_detection_khong_co_trong_bang_gt_van_vao_ket_qua(tmp_path, messages):
    """Đối xứng với sct: hệ thống báo cáo mọi thứ nó thấy, ground-truth mới là bên lọc."""
    lay = TrackEvalLayout(root=tmp_path, benchmark="MCT", split="mct")
    _, stats = export_mct(messages, {}, {("cam01", 2): 77}, lay, tracker="t", fps=25.0)
    assert stats["n_gt"] == 0 and stats["n_result"] == 1


# --------------------------------------------------------------------------------------
# Đọc nguồn danh tính
# --------------------------------------------------------------------------------------


def test_doc_global_id_tu_sqlite_lay_dong_moi_nhat(tmp_path):
    db = tmp_path / "mct.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE appearances (tracklet_id INTEGER PRIMARY KEY, global_id INTEGER, "
        "cam_id TEXT, local_track_id INTEGER, start_ms INTEGER)"
    )
    con.executemany(
        "INSERT INTO appearances VALUES (?,?,?,?,?)",
        [(1, 10, "cam01", 3, 1000), (2, 20, "cam01", 3, 5000)],
    )
    con.commit()
    con.close()

    assert load_global_ids(db) == {("cam01", 3): 20}


def test_doc_bang_gt_dung_dinh_dang_chung(tmp_path):
    path = tmp_path / "x.gt.json"
    path.write_text(
        json.dumps({"tracklets": [{"cam_id": "cam01", "local_track_id": 3, "gt_global_id": 9}]}),
        encoding="utf-8",
    )
    assert load_gt_table(path) == {("cam01", 3): 9}


# --------------------------------------------------------------------------------------
# CVAT
# --------------------------------------------------------------------------------------

CVAT_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <track id="0" label="person">
    <attribute name="person_id">P02</attribute>
    <box frame="0" xtl="10" ytl="20" xbr="40" ybr="80" outside="0" occluded="0"></box>
    <box frame="1" xtl="12" ytl="21" xbr="42" ybr="81" outside="0" occluded="1"></box>
    <box frame="2" xtl="0" ytl="0" xbr="0" ybr="0" outside="1" occluded="0"></box>
  </track>
  <track id="1" label="person">
    <box frame="0" xtl="100" ytl="30" xbr="130" ybr="90" outside="0" occluded="0">
      <attribute name="person_id">P01</attribute>
    </box>
  </track>
  <track id="2" label="car">
    <box frame="0" xtl="500" ytl="500" xbr="600" ybr="600" outside="0" occluded="0"></box>
  </track>
</annotations>
"""


@pytest.fixture
def cvat_file(tmp_path) -> Path:
    path = tmp_path / "cam01.xml"
    path.write_text(CVAT_XML, encoding="utf-8")
    return path


def test_cvat_bo_khung_outside(cvat_file):
    """`outside="1"` là người đã ra khỏi khung — CVAT vẫn ghi hộp, tính vào là bịa dữ liệu."""
    boxes = parse_cvat_video(cvat_file)
    assert [(b.track_id, b.frame) for b in boxes] == [(0, 0), (0, 1), (1, 0)]


def test_cvat_loc_theo_label(cvat_file):
    assert all(b.track_id != 2 for b in parse_cvat_video(cvat_file))
    assert len(parse_cvat_video(cvat_file, label="")) == 4  # '' = nhận mọi label


def test_cvat_doc_thuoc_tinh_o_ca_track_lan_box(cvat_file):
    """CVAT đặt thuộc tính ở hai chỗ tuỳ người gán nhãn khai là cố định hay thay đổi được."""
    theo_track = {b.track_id: b.person for b in parse_cvat_video(cvat_file)}
    assert theo_track == {0: "P02", 1: "P01"}


def test_cvat_bbox_doi_tu_goc_sang_xywh(cvat_file):
    box = parse_cvat_video(cvat_file)[0]
    assert box.bbox == (10.0, 20.0, 30.0, 60.0)


def test_global_id_on_dinh_giua_cac_lan_chay(cvat_file):
    """Đánh số theo thứ tự từ điển: chạy lại phải ra đúng bảng cũ, nếu không thì kết quả
    của hai lần chấm không so được với nhau."""
    boxes = parse_cvat_video(cvat_file)
    a = assign_global_ids({"cam01": boxes}, require_global=True)
    b = assign_global_ids({"cam01": list(reversed(boxes))}, require_global=True)
    assert a == b == {("cam01", 1): 1, ("cam01", 0): 2}  # P01 -> 1, P02 -> 2


def test_cung_person_id_o_hai_camera_thanh_mot_global_id(cvat_file):
    boxes = parse_cvat_video(cvat_file)
    ids = assign_global_ids({"cam01": boxes, "cam02": boxes}, require_global=True)
    assert ids[("cam01", 0)] == ids[("cam02", 0)]


def test_thieu_thuoc_tinh_danh_tinh_thi_bao_loi(tmp_path):
    """Thiếu nó thì mỗi camera là một tập danh tính riêng — im lặng cho qua là tệ nhất."""
    path = tmp_path / "x.xml"
    path.write_text(
        '<annotations><track id="0" label="person">'
        '<box frame="0" xtl="1" ytl="2" xbr="3" ybr="4" outside="0"></box>'
        "</track></annotations>",
        encoding="utf-8",
    )
    boxes = parse_cvat_video(path)
    with pytest.raises(CvatError, match="danh tính"):
        assign_global_ids({"cam01": boxes}, require_global=True)
    assert assign_global_ids({"cam01": boxes}, require_global=False) == {}


def test_mot_track_mang_hai_danh_tinh_thi_bao_loi(tmp_path):
    path = tmp_path / "x.xml"
    path.write_text(
        '<annotations><track id="0" label="person">'
        '<box frame="0" xtl="1" ytl="2" xbr="3" ybr="4" outside="0">'
        '<attribute name="person_id">P01</attribute></box>'
        '<box frame="1" xtl="1" ytl="2" xbr="3" ybr="4" outside="0">'
        '<attribute name="person_id">P02</attribute></box>'
        "</track></annotations>",
        encoding="utf-8",
    )
    with pytest.raises(CvatError, match="mâu thuẫn"):
        assign_global_ids({"cam01": parse_cvat_video(path)}, require_global=True)


def test_file_khong_co_hop_nao_bao_loi_kem_goi_y(tmp_path):
    path = tmp_path / "rong.xml"
    path.write_text("<annotations></annotations>", encoding="utf-8")
    with pytest.raises(CvatError, match="CVAT for video"):
        parse_cvat_video(path)


def test_cvat_ghi_ra_gt_va_bang_global_id(tmp_path, cvat_file):
    boxes = parse_cvat_video(cvat_file)
    per_cam = {"cam01": boxes}
    ids = assign_global_ids(per_cam, require_global=True)
    written = write_ground_truth(tmp_path / "gt", per_cam, ids, meta={"source": "test"})

    assert written == {"cam01": 3}
    rows = parse_mot(tmp_path / "gt" / "cam01.gt.txt")
    assert [(r.frame, r.track_id) for r in rows] == [(1, 0), (1, 1), (2, 0)]

    bang = json.loads((tmp_path / "gt" / "global_ids.gt.json").read_text(encoding="utf-8"))
    assert {t["local_track_id"]: t["gt_global_id"] for t in bang["tracklets"]} == {0: 2, 1: 1}
    assert bang["tracklets"][0]["n_frames"] == 2


# --------------------------------------------------------------------------------------
# Nguồn ground-truth độc lập (--gt-fixture)
# --------------------------------------------------------------------------------------


@pytest.fixture
def annotation() -> GtSource:
    """Chú thích của người gán nhãn: hộp KHÁC hộp của detector, id là người thật."""
    messages = [
        msg(
            "cam01",
            0,
            [Detection(local_track_id=41, bbox=(12.0, 22.0, 28.0, 58.0), confidence=1.0)],
        ),
        msg(
            "cam02",
            0,
            [Detection(local_track_id=42, bbox=(205.0, 21.0, 31.0, 61.0), confidence=1.0)],
        ),
    ]
    return GtSource(messages=messages, table={("cam01", 41): 900, ("cam02", 42): 900})


def test_sct_gt_rieng_lay_hop_cua_chu_thich(tmp_path, messages, annotation):
    """GT phải là hộp của người chú thích; nếu nó lấy lại hộp của kết quả thì MOTP luôn đẹp."""
    lay = TrackEvalLayout(root=tmp_path, benchmark="MCT", split="sct")
    export_sct(messages, {}, lay, tracker="t", fps=2.0, gt_source=annotation)

    gt_rows = parse_mot(lay.gt_file("cam01"))
    assert [(r.track_id, r.x) for r in gt_rows] == [(900, 12.0)]
    assert {r.track_id for r in parse_mot(lay.result_file("t", "cam01"))} == {1, 2}


def test_mct_gt_rieng_dung_gt_global_id_cua_bang_chu_thich(tmp_path, messages, annotation):
    lay = TrackEvalLayout(root=tmp_path, benchmark="MCT", split="mct")
    _, stats = export_mct(
        messages, {}, {("cam01", 1): 7}, lay, tracker="t", fps=2.0, gt_source=annotation
    )

    assert stats["n_gt"] == 2  # hai hộp chú thích, không phải hộp của kết quả
    assert {r.track_id for r in parse_mot(lay.gt_file("all"))} == {900}
    assert {r.track_id for r in parse_mot(lay.result_file("t", "all"))} == {7}


def test_mct_mot_danh_tinh_khong_bao_gio_co_hai_hop_trong_MOT_khung(tmp_path, annotation):
    """Ràng buộc cứng của MOT Challenge, và là lý do chuỗi ảo phải NỐI TIẾP các camera.

    Người 900 đứng trước hai camera cùng lúc (`frame_id` 0 ở cả hai). Xếp xen kẽ thì hai
    hộp rơi vào cùng một khung ảo và TrackEval từ chối chấm cả chuỗi.
    """
    lay = TrackEvalLayout(root=tmp_path, benchmark="MCT", split="mct")
    export_mct(annotation.messages, {}, {}, lay, tracker="t", fps=2.0, gt_source=annotation)

    rows = parse_mot(lay.gt_file("all"))
    assert len(rows) == 2
    assert len({(r.frame, r.track_id) for r in rows}) == 2
    assert rows[0].frame != rows[1].frame
