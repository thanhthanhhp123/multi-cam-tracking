"""Test `tools/cvat_to_mot.py` — chú thích CVAT → ground-truth MOT + bảng Global ID.

Đây là cửa duy nhất mà dữ liệu tự thu ở M6 đi vào phần chấm điểm, nên mọi lỗi ở đây đều
thuộc loại tệ nhất: bảng điểm VẪN RA SỐ, chỉ là số sai. Ba chỗ được canh kỹ nhất:

1. khung của CVAT đếm từ 0, của MOT đếm từ 1,
2. `gt_global_id` phải nối được danh tính XUYÊN camera và phải ổn định giữa các lần chạy,
3. hộp hỏng (thiếu toạ độ, mâu thuẫn danh tính) phải NỔ chứ không được đi tiếp im lặng.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from common.motformat import parse_mot
from tools.cvat_to_mot import (
    CvatBox,
    CvatError,
    _parse_annotation_arg,
    assign_global_ids,
    main,
    parse_cvat_video,
    write_ground_truth,
)
from tools.export_trackeval import load_gt_table

# --------------------------------------------------------------------------------------
# Dựng XML "CVAT for video 1.1" tối thiểu
# --------------------------------------------------------------------------------------


def _box(
    frame: int,
    *,
    xtl: str = "10.0",
    ytl: str = "20.0",
    xbr: str = "30.0",
    ybr: str = "60.0",
    outside: str = "0",
    occluded: str = "0",
    person: str | None = None,
    drop: str = "",
) -> str:
    attrs = {
        "frame": str(frame),
        "xtl": xtl,
        "ytl": ytl,
        "xbr": xbr,
        "ybr": ybr,
        "outside": outside,
        "occluded": occluded,
    }
    attrs.pop(drop, None)
    body = f'<attribute name="person_id">{person}</attribute>' if person else ""
    head = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    return f"<box {head}>{body}</box>"


def _track(
    track_id: int, boxes: list[str], *, label: str = "person", person: str | None = None
) -> str:
    body = f'<attribute name="person_id">{person}</attribute>' if person else ""
    return f'<track id="{track_id}" label="{label}">{body}{"".join(boxes)}</track>'


def _write_xml(path: Path, tracks: list[str]) -> Path:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<annotations><version>1.1</version>'
        + "".join(tracks)
        + "</annotations>",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------------------
# Đọc XML
# --------------------------------------------------------------------------------------


def test_bo_khung_outside(tmp_path: Path):
    """`outside="1"` là người đã ra khỏi khung — CVAT vẫn ghi hộp, ground-truth thì không."""
    xml = _write_xml(
        tmp_path / "cam01.xml",
        [_track(0, [_box(0), _box(1, outside="1"), _box(2)], person="P01")],
    )
    assert [b.frame for b in parse_cvat_video(xml)] == [0, 2]


def test_toa_do_doi_sang_xywh():
    """CVAT cho hai góc, schema của đồ án dùng [x, y, w, h]."""
    box = CvatBox(frame=0, track_id=0, person="P01", xtl=10.0, ytl=20.0, xbr=30.0, ybr=60.0)
    assert box.bbox == (10.0, 20.0, 20.0, 40.0)


def test_thuoc_tinh_tren_box_thang_tren_track(tmp_path: Path):
    """Người gán nhãn sửa danh tính giữa chừng thì ý định mới nhất nằm ở box."""
    xml = _write_xml(
        tmp_path / "cam01.xml",
        [_track(0, [_box(0), _box(1, person="P09")], person="P01")],
    )
    assert [b.person for b in parse_cvat_video(xml)] == ["P01", "P09"]


def test_loc_theo_label(tmp_path: Path):
    xml = _write_xml(
        tmp_path / "cam01.xml",
        [
            _track(0, [_box(0)], label="person", person="P01"),
            _track(1, [_box(0)], label="car", person="P02"),
        ],
    )
    assert {b.track_id for b in parse_cvat_video(xml)} == {0}
    assert {b.track_id for b in parse_cvat_video(xml, label="")} == {0, 1}


def test_khong_co_hop_nao_thi_bao_loi(tmp_path: Path):
    """Nhầm định dạng xuất là lỗi thường gặp nhất — thông báo phải nói ra điều đó."""
    xml = _write_xml(tmp_path / "cam01.xml", [_track(0, [_box(0)], label="car")])
    with pytest.raises(CvatError, match="label"):
        parse_cvat_video(xml)


@pytest.mark.parametrize("drop", ["xtl", "ybr", "frame"])
def test_hop_thieu_toa_do_thi_no(tmp_path: Path, drop: str):
    """Thiếu toạ độ mà thành NaN thì gt.txt vẫn ghi được và bảng điểm sai không triệu chứng."""
    xml = _write_xml(tmp_path / "cam01.xml", [_track(0, [_box(0, drop=drop)], person="P01")])
    with pytest.raises(CvatError, match="toạ độ"):
        parse_cvat_video(xml)


def test_hop_toa_do_khong_phai_so_thi_no(tmp_path: Path):
    xml = _write_xml(tmp_path / "cam01.xml", [_track(0, [_box(0, xtl="abc")], person="P01")])
    with pytest.raises(CvatError, match="toạ độ"):
        parse_cvat_video(xml)


# --------------------------------------------------------------------------------------
# Danh tính xuyên camera
# --------------------------------------------------------------------------------------


def _cam(person: str, *, track_id: int = 0, frames: tuple[int, ...] = (0, 1)) -> list[CvatBox]:
    return [
        CvatBox(frame=f, track_id=track_id, person=person, xtl=0.0, ytl=0.0, xbr=10.0, ybr=20.0)
        for f in frames
    ]


def test_cung_person_id_o_hai_camera_cho_cung_global_id():
    """`track id` của CVAT chỉ duy nhất trong một task — thuộc tính mới là thứ nối danh tính."""
    per_cam = {"cam01": _cam("P01", track_id=0), "cam02": _cam("P01", track_id=7)}
    ids = assign_global_ids(per_cam, require_global=True)
    assert ids[("cam01", 0)] == ids[("cam02", 7)]


def test_hai_nguoi_khac_nhau_khong_dung_chung_global_id():
    per_cam = {"cam01": _cam("P01"), "cam02": _cam("P02", track_id=1)}
    ids = assign_global_ids(per_cam, require_global=True)
    assert ids[("cam01", 0)] != ids[("cam02", 1)]


def test_danh_so_on_dinh_khong_phu_thuoc_thu_tu_camera():
    """Chạy lại ra bảng khác thì kết quả không tái lập được — đánh số theo thứ tự từ điển."""
    a = {"cam01": _cam("P02"), "cam02": _cam("P01", track_id=1)}
    b = {"cam02": _cam("P01", track_id=1), "cam01": _cam("P02")}
    assert assign_global_ids(a, require_global=True) == assign_global_ids(b, require_global=True)
    assert assign_global_ids(a, require_global=True)[("cam02", 1)] == 1  # "P01" đứng trước


def test_thieu_thuoc_tinh_thi_bao_loi_va_chi_duong_thoat():
    with pytest.raises(CvatError, match="--no-global"):
        assign_global_ids({"cam01": _cam("")}, require_global=True)


def test_no_global_bo_qua_track_khong_co_danh_tinh():
    ids = assign_global_ids({"cam01": _cam("") + _cam("P01", track_id=1)}, require_global=False)
    assert ("cam01", 0) not in ids
    assert ids[("cam01", 1)] == 1


def test_mot_track_mang_hai_danh_tinh_thi_bao_loi():
    """Chú thích mâu thuẫn: đoán bừa ở đây là bịa ground-truth."""
    per_cam = {"cam01": _cam("P01") + _cam("P02", frames=(2,))}
    with pytest.raises(CvatError, match="mâu thuẫn"):
        assign_global_ids(per_cam, require_global=True)


# --------------------------------------------------------------------------------------
# Ghi ra đĩa
# --------------------------------------------------------------------------------------


def test_khung_ghi_ra_dem_tu_mot(tmp_path: Path):
    per_cam = {"cam01": _cam("P01", frames=(0, 1))}
    ids = assign_global_ids(per_cam, require_global=True)
    write_ground_truth(tmp_path, per_cam, ids, meta={})
    rows = parse_mot(tmp_path / "cam01.gt.txt")
    assert [r.frame for r in rows] == [1, 2]
    assert (rows[0].x, rows[0].y, rows[0].w, rows[0].h) == (0.0, 0.0, 10.0, 20.0)


def test_hop_bi_che_ghi_visibility_thap(tmp_path: Path):
    """`occluded=1` vẫn là ground-truth hợp lệ, nhưng TrackEval cần biết nó kém tin cậy."""
    per_cam = {
        "cam01": [
            CvatBox(0, 0, "P01", 0.0, 0.0, 10.0, 20.0, occluded=False),
            CvatBox(1, 0, "P01", 0.0, 0.0, 10.0, 20.0, occluded=True),
        ]
    }
    ids = assign_global_ids(per_cam, require_global=True)
    write_ground_truth(tmp_path, per_cam, ids, meta={})
    visibility = [
        line.split(",")[8]
        for line in (tmp_path / "cam01.gt.txt").read_text(encoding="utf-8").splitlines()
    ]
    assert visibility == ["1.00", "0.50"]


def test_bang_gt_json_doc_duoc_boi_export_trackeval(tmp_path: Path):
    """Cùng định dạng với `wildtrack_to_fixture.py` — lệch thì M6 đứt ở giữa quy trình."""
    per_cam = {"cam01": _cam("P01"), "cam02": _cam("P01", track_id=7, frames=(4, 5, 6))}
    ids = assign_global_ids(per_cam, require_global=True)
    write_ground_truth(tmp_path, per_cam, ids, meta={"source": "cvat"})

    table = load_gt_table(tmp_path / "global_ids.gt.json")
    assert table[("cam01", 0)] == table[("cam02", 7)]

    payload = json.loads((tmp_path / "global_ids.gt.json").read_text(encoding="utf-8"))
    cam02 = next(t for t in payload["tracklets"] if t["cam_id"] == "cam02")
    assert (cam02["start_frame"], cam02["end_frame"], cam02["n_frames"]) == (4, 6, 3)


def test_track_khong_co_danh_tinh_khong_vao_bang_json(tmp_path: Path):
    """Ở chế độ --no-global, gt.txt vẫn có hộp nhưng bảng danh tính thì không được bịa."""
    per_cam = {"cam01": _cam("") + _cam("P01", track_id=1)}
    ids = assign_global_ids(per_cam, require_global=False)
    written = write_ground_truth(tmp_path, per_cam, ids, meta={})

    assert written["cam01"] == 4  # cả bốn hộp vẫn được ghi
    payload = json.loads((tmp_path / "global_ids.gt.json").read_text(encoding="utf-8"))
    assert [t["local_track_id"] for t in payload["tracklets"]] == [1]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def test_doi_so_annotation_sai_dang():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_annotation_arg("cam01.xml")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_annotation_arg("=cam01.xml")
    assert _parse_annotation_arg(" cam01 =a/b.xml") == ("cam01", Path("a/b.xml"))


def test_chay_tron_hai_camera(tmp_path: Path):
    """Đường đi thật của M6: hai task CVAT → gt.txt mỗi camera + một bảng danh tính."""
    a = _write_xml(tmp_path / "a.xml", [_track(0, [_box(0), _box(1)], person="P01")])
    b = _write_xml(tmp_path / "b.xml", [_track(3, [_box(5)], person="P01")])
    out = tmp_path / "gt"

    code = main(["--annotation", f"cam01={a}", "--annotation", f"cam02={b}", "--out-dir", str(out)])
    assert code == 0

    assert len(parse_mot(out / "cam01.gt.txt")) == 2
    assert len(parse_mot(out / "cam02.gt.txt")) == 1
    table = load_gt_table(out / "global_ids.gt.json")
    assert table[("cam01", 0)] == table[("cam02", 3)]

    meta = json.loads((out / "global_ids.gt.json").read_text(encoding="utf-8"))["meta"]
    assert meta["n_identities"] == 1
    assert meta["n_tracks"] == 2
