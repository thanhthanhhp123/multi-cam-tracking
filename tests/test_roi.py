"""Test `mct/roi.py`: lọc vùng sàn theo camera, khai trong topology, áp ở engine và ở bước chấm.

Điểm cần ghim: (1) khai sai thì báo lỗi ngay chứ không thành "không chặn gì"; (2) camera không
khai ROI đi qua nguyên vẹn; (3) engine lọc TRƯỚC khi dựng tracklet; (4) bộ cấu hình lab của repo
cắt đúng vùng xa của cam01 mà vẫn giữ vùng chồng lấn.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from common.schema import CLASS_PERSON, Detection, FrameMessage, read_jsonl, write_jsonl
from mct.homography import CameraHomography, HomographyMapper
from mct.roi import GroundRoi, RoiError, RoiFilter, foot_point
from mct.topology import Topology, TopologyError

# Ma trận đơn vị: điểm chân pixel (x + w/2, y + h) CHÍNH LÀ toạ độ sàn tính bằng mét.
IDENTITY = {c: CameraHomography(cam_id=c, matrix=np.eye(3)) for c in ("cam01", "cam02")}


def _det(track_id: int, foot_x: float, foot_y: float = 0.0) -> Detection:
    return Detection(
        local_track_id=track_id,
        bbox=(foot_x - 0.5, foot_y - 2.0, 1.0, 2.0),
        confidence=0.9,
        class_id=CLASS_PERSON,
    )


def _msg(cam_id: str, dets: list[Detection], frame_id: int = 0) -> FrameMessage:
    return FrameMessage(
        cam_id=cam_id,
        frame_id=frame_id,
        ts_ms=1_700_000_000_000 + 40 * frame_id,
        frame_pts_ns=frame_id,
        frame_width=1920,
        frame_height=1080,
        detections=dets,
    )


def _topology(roi: dict | None) -> dict:
    cam01: dict = {"overlaps_with": ["cam02"]}
    if roi is not None:
        cam01["ground_roi"] = roi
    return {"cameras": {"cam01": cam01, "cam02": {"overlaps_with": ["cam01"]}}}


# ------------------------------------------------------------------ GroundRoi


def test_roi_nua_mo_va_canh_null_khong_chan():
    roi = GroundRoi(x_m=(None, 6.0))
    assert roi.contains((-100.0, 5.0))
    assert roi.contains((5.999, -3.0))
    assert not roi.contains((6.0, 0.0))  # [lo, hi): đúng mốc thì thuộc bên kia
    assert GroundRoi().contains((1e6, -1e6))


def test_roi_doc_tu_yaml():
    roi = GroundRoi.from_mapping({"x_m": [None, 6.0], "y_m": [-0.5, 1.2]})
    assert roi == GroundRoi(x_m=(None, 6.0), y_m=(-0.5, 1.2))
    assert GroundRoi.from_mapping(roi.to_mapping()) == roi


@pytest.mark.parametrize(
    "data, match",
    [
        ({"x": [None, 6.0]}, "khoá lạ"),  # gõ sai tên khoá: không được thành "không chặn gì"
        ({"x_m": 6.0}, r"\[min, max\]"),
        ({"x_m": [None]}, r"\[min, max\]"),
        ({"x_m": [6.0, 2.0]}, "min < max"),
    ],
)
def test_roi_khai_sai_thi_bao_loi(data: dict, match: str):
    with pytest.raises(RoiError, match=match):
        GroundRoi.from_mapping(data)


def test_diem_chan_la_giua_canh_day():
    assert foot_point(_det(1, 3.0, 4.0)) == (3.0, 4.0)


# ------------------------------------------------------------------ topology


def test_topology_nap_ground_roi():
    topo = Topology.from_mapping(_topology({"x_m": [None, 6.0]}))
    assert topo.cameras["cam01"].ground_roi == GroundRoi(x_m=(None, 6.0))
    assert topo.cameras["cam02"].ground_roi is None
    assert topo.ground_rois == {"cam01": GroundRoi(x_m=(None, 6.0))}


def test_topology_ground_roi_sai_bao_loi_kem_ten_camera():
    with pytest.raises(TopologyError, match=r"cam01.*khoá lạ"):
        Topology.from_mapping(_topology({"xmax": 6.0}))


# ------------------------------------------------------------------ RoiFilter


def test_loc_bo_detection_ngoai_roi_giu_camera_khac_nguyen_ven():
    flt = RoiFilter({"cam01": GroundRoi(x_m=(None, 6.0))}, HomographyMapper(dict(IDENTITY)))
    out, n = flt.filter(_msg("cam01", [_det(1, 2.0), _det(2, 18.4), _det(3, 5.9)]))
    assert n == 1
    assert [d.local_track_id for d in out.detections] == [1, 3]
    far = _msg("cam02", [_det(1, 18.4)])
    same, n = flt.filter(far)
    assert n == 0 and same is far  # camera không có ROI: không sao chép, không đụng tới


def test_khung_rong_sau_khi_loc_van_giu_message():
    flt = RoiFilter({"cam01": GroundRoi(x_m=(None, 6.0))}, HomographyMapper(dict(IDENTITY)))
    out, n = flt.filter(_msg("cam01", [_det(1, 9.0)], frame_id=7))
    assert n == 1 and out.detections == [] and out.frame_id == 7


def test_detection_khong_chieu_duoc_bi_bo():
    # Hàng cuối = (1, 0, 0): mọi điểm có x = 0 rơi đúng đường chân trời (w = 0).
    horizon = CameraHomography(cam_id="cam01", matrix=np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    flt = RoiFilter({"cam01": GroundRoi(x_m=(None, 6.0))}, HomographyMapper({"cam01": horizon}))
    _, n = flt.filter(_msg("cam01", [_det(1, 0.0, 1.0)]))
    assert n == 1


def test_khai_roi_ma_khong_co_homography_la_loi():
    assert RoiFilter.from_rois({}, None) is None
    with pytest.raises(RoiError, match="homography"):
        RoiFilter.from_rois({"cam01": GroundRoi(x_m=(None, 6.0))}, None)
    only_cam02 = HomographyMapper({"cam02": IDENTITY["cam02"]})
    with pytest.raises(RoiError, match="cam01"):
        RoiFilter.from_rois({"cam01": GroundRoi(x_m=(None, 6.0))}, only_cam02)


# ------------------------------------------------------------------ engine


def test_engine_loc_truoc_khi_dung_tracklet():
    from mct.__main__ import Engine

    flt = RoiFilter({"cam01": GroundRoi(x_m=(None, 6.0))}, HomographyMapper(dict(IDENTITY)))
    engine = Engine(roi_filter=flt)
    for f in range(5):
        engine.feed(_msg("cam01", [_det(1, 2.0), _det(2, 18.4)], frame_id=f))
    assert engine.n_roi_dropped == 5
    assert engine.builder.get("cam01", 1) is not None
    assert engine.builder.get("cam01", 2) is None  # người ở vùng xa không bao giờ thành tracklet


def _write_lab(tmp_path: Path, roi: dict | None) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    topo = tmp_path / "topology.yaml"
    topo.write_text(yaml.safe_dump(_topology(roi)), encoding="utf-8")
    hdir = tmp_path / "homography"
    HomographyMapper(dict(IDENTITY)).save(hdir)
    return topo, hdir


def test_build_engine_dung_bo_loc_tu_topology(tmp_path: Path):
    from mct.__main__ import build_engine

    topo, hdir = _write_lab(tmp_path, {"x_m": [None, 6.0]})
    engine = build_engine({}, topology_path=topo, homography_dir=hdir, db_path=":memory:")
    assert engine.roi_filter is not None and engine.roi_filter.rois == {
        "cam01": GroundRoi(x_m=(None, 6.0))
    }
    plain, _ = _write_lab(tmp_path / "plain", None)
    assert build_engine({}, topology_path=plain, homography_dir=hdir).roi_filter is None
    with pytest.raises(RoiError):  # ROI mà quên --homography-dir: dừng ngay, không chạy lặng lẽ
        build_engine({}, topology_path=topo, homography_dir=tmp_path / "empty")


# ------------------------------------------------------------------ fixture (bước chấm)


def test_loc_fixture_theo_topology(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from eval.roi_filter_fixture import main

    topo, hdir = _write_lab(tmp_path, {"x_m": [None, 6.0]})
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(
        src,
        [
            _msg("cam01", [_det(1, 2.0), _det(2, 18.4)], frame_id=0),
            _msg("cam01", [_det(2, 19.0)], frame_id=1),
            _msg("cam02", [_det(1, 18.4)], frame_id=0),
        ],
    )
    assert (
        main(
            [
                *("--topology", str(topo), "--homography-dir", str(hdir)),
                *("--in", str(src), "--out", str(out)),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["dropped_per_cam"] == {"cam01": [2, 3], "cam02": [0, 1]}
    msgs = list(read_jsonl(out))
    assert len(msgs) == 3  # khung rỗng vẫn giữ: số khung là một phần của contract khi chấm
    assert [len(m.detections) for m in msgs] == [1, 0, 1]


# ------------------------------------------------------------------ cấu hình lab của repo


def test_roi_lab_cat_vung_xa_cam01_ma_giu_vung_chong_lan(repo_root: Path):
    """Chiếu ngược bằng homography thật của cam01: vùng cam03 bị bỏ, vùng chồng lấn được giữ."""
    lab = repo_root / "configs" / "lab"
    topo = Topology.load(lab / "topology.yaml")
    mapper = HomographyMapper.load(lab / "homography")
    flt = RoiFilter.from_rois(topo.ground_rois, mapper)
    assert flt is not None and set(flt.rois) == {"cam01"}

    inverse = np.linalg.inv(mapper.cameras["cam01"].matrix)

    def det_at(track_id: int, x_m: float, y_m: float = 0.34) -> Detection:
        u, v, w = inverse @ [x_m, y_m, 1.0]
        return _det(track_id, u / w, v / w)

    near = [det_at(1, -1.0), det_at(2, 0.0), det_at(3, 2.0), det_at(4, 5.5)]
    far = [det_at(10, 6.5), det_at(11, 18.4), det_at(12, 23.0)]
    out, n = flt.filter(_msg("cam01", near + far))
    assert n == len(far)
    assert [d.local_track_id for d in out.detections] == [1, 2, 3, 4]
