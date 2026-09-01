"""Test bộ chuyển WildTrack -> fixture.

Chạy hoàn toàn không cần dataset ảnh (~13GB) và không cần opencv/onnxruntime: phần đọc
annotation dùng dữ liệu giả sinh trong test, phần embedding tiêm `image_reader` + embedder
giả. Một test đối chiếu thêm với 2 file annotation THẬT vendor ở `tests/data/wildtrack/`
để bắt trường hợp schema WildTrack đổi.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from common.schema import decode_jsonl, encode_jsonl, validate
from tools.wildtrack_to_fixture import (
    BASE_TS_MS,
    build_fixture,
    clip_bbox_xyxy,
    parse_raw_detections,
    position_id_to_world_m,
    write_ground_truth,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_WILDTRACK_DIR = REPO_ROOT / "tests" / "data" / "wildtrack"

# positionID = 480*grid_y + grid_x ; world = (-3.0 + 0.025*grid_x, -9.0 + 0.025*grid_y)
POS_P7 = 480 * 100 + 200  # -> (2.0, -6.5)
POS_P8 = 480 * 300 + 100  # -> (-0.5, -1.5)
POS_P9 = 480 * 500 + 50  # -> (-1.75, 3.5)

BOX_A = (100, 150, 260, 900)  # 160 x 750, trong khung
BOX_LEFT_OVERFLOW = (-40, 100, 90, 620)  # tràn trái -> x0 phải bị clip về 0
BOX_RIGHT = (1600, 80, 1780, 700)


def _person(pid: int, pos_id: int, boxes: dict[int, tuple[int, int, int, int]]) -> dict:
    views = []
    for v in range(7):
        if v in boxes:
            xmin, ymin, xmax, ymax = boxes[v]
            views.append({"viewNum": v, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
        else:
            views.append({"viewNum": v, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1})
    return {"personID": pid, "positionID": pos_id, "views": views}


def _make_frames() -> dict[int, list[dict]]:
    """10 khung, 3 người:

    - p7: view0 + view1 toàn bộ  -> tín hiệu xuyên camera (cam01 <-> cam02)
    - p8: view2 toàn bộ; view0 khung 0-3 và 8-9 (khoảng trống 4-7) -> tách 2 local track ở cam01
    - p9: chỉ view1, khung 2-7
    """
    frames: dict[int, list[dict]] = {}
    for i in range(10):
        persons = [_person(7, POS_P7, {0: BOX_A, 1: BOX_RIGHT})]

        p8_boxes: dict[int, tuple[int, int, int, int]] = {2: BOX_LEFT_OVERFLOW}
        if i <= 3 or i >= 8:
            p8_boxes[0] = BOX_RIGHT
        persons.append(_person(8, POS_P8, p8_boxes))

        if 2 <= i <= 7:
            persons.append(_person(9, POS_P9, {1: BOX_A}))

        frames[i] = persons
    return frames


@pytest.fixture
def wildtrack_dir(tmp_path: Path) -> Path:
    ann = tmp_path / "annotations_positions"
    ann.mkdir(parents=True)
    for i, persons in _make_frames().items():
        (ann / f"{i * 5:08d}.json").write_text(json.dumps(persons), encoding="utf-8")
    return tmp_path


class FakeEmbedder:
    """Trả embedding suy ra từ kích thước crop — đủ để kiểm tra đường ghép, không cần model."""

    embed_dim = 8

    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(crops_bgr), self.embed_dim), dtype=np.float32)
        for i, crop in enumerate(crops_bgr):
            out[i, 0] = crop.shape[0]
            out[i, 1] = crop.shape[1]
            out[i, 2] = 1.0
        return out


def _blank_image(_path: Path) -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Hàm thuần
# --------------------------------------------------------------------------- #


def test_position_id_ve_toa_do_the_gioi() -> None:
    assert position_id_to_world_m(0) == pytest.approx((-3.0, -9.0))
    assert position_id_to_world_m(POS_P7) == pytest.approx((2.0, -6.5))
    assert position_id_to_world_m(POS_P9) == pytest.approx((-1.75, 3.5))


def test_clip_bbox_chuyen_xyxy_sang_xywh_va_clip() -> None:
    assert clip_bbox_xyxy(100, 150, 260, 900, width=1920, height=1080) == (
        100.0,
        150.0,
        160.0,
        750.0,
    )
    # tràn trái: x0 kéo về 0, chiều rộng co lại
    assert clip_bbox_xyxy(-40, 100, 90, 620, width=1920, height=1080) == (0.0, 100.0, 90.0, 520.0)


def test_clip_bbox_loai_hop_suy_bien() -> None:
    assert clip_bbox_xyxy(1920, 10, 1930, 40, width=1920, height=1080) is None  # ngoài khung phải
    assert clip_bbox_xyxy(500, 500, 500, 900, width=1920, height=1080) is None  # rộng 0


# --------------------------------------------------------------------------- #
# Đọc annotation
# --------------------------------------------------------------------------- #


def test_parse_loc_view_va_clip_bbox(wildtrack_dir: Path) -> None:
    raw, n_frames = parse_raw_detections(
        wildtrack_dir / "annotations_positions",
        view_indices=[0, 2],
        stride=1,
        max_frames=0,
        min_box_area=100.0,
    )
    assert n_frames == 10
    assert {d.view_idx for d in raw} == {0, 2}
    # mọi bbox nằm gọn trong khung sau khi clip
    for d in raw:
        x, y, w, h = d.bbox
        assert x >= 0 and y >= 0 and x + w <= 1920 and y + h <= 1080
    # p8 ở view2 dùng hộp tràn trái -> x phải bằng 0
    assert any(d.person_id == 8 and d.view_idx == 2 and d.bbox[0] == 0.0 for d in raw)


def test_min_box_area_loai_bbox_nho(wildtrack_dir: Path) -> None:
    raw, _ = parse_raw_detections(
        wildtrack_dir / "annotations_positions",
        view_indices=[0, 1, 2],
        stride=1,
        max_frames=0,
        min_box_area=1_000_000.0,  # lớn hơn mọi hộp -> không còn detection nào
    )
    assert raw == []


# --------------------------------------------------------------------------- #
# build_fixture — chế độ hình học
# --------------------------------------------------------------------------- #


@pytest.fixture
def geom(wildtrack_dir: Path):
    return build_fixture(wildtrack_dir, views=[1, 2, 3], fps=2.0, embedder=None)


def test_moi_message_dung_contract(geom) -> None:
    messages, _, _ = geom
    for msg in messages:
        assert validate(msg) == []


def test_so_message_bang_so_camera_nhan_so_khung(geom) -> None:
    messages, _, meta = geom
    assert len(messages) == 3 * 10
    assert meta["cam_ids"] == ["cam01", "cam02", "cam03"]
    assert meta["embed_dim"] == 0


def test_chi_hinh_hoc_thi_khong_co_embedding(geom) -> None:
    messages, _, _ = geom
    assert all(det.embedding is None for msg in messages for det in msg.detections)


def test_timestamp_theo_fps_va_tang_dan(geom) -> None:
    messages, _, _ = geom
    assert [m.ts_ms for m in messages] == sorted(m.ts_ms for m in messages)
    by_cam = [m for m in messages if m.cam_id == "cam01"]
    assert by_cam[0].ts_ms == BASE_TS_MS
    assert by_cam[1].ts_ms - by_cam[0].ts_ms == 500  # 2 fps


def test_global_id_nhat_quan_theo_khoa_camera_local(geom) -> None:
    messages, tracklets, _ = geom
    gid_of: dict[tuple[str, int], int] = {}
    for t in tracklets:
        key = (t.cam_id, t.local_track_id)
        assert key not in gid_of, "local_track_id trùng trong một camera"
        gid_of[key] = t.gt_global_id
    # mọi detection trong stream phải khớp bảng ground-truth
    for msg in messages:
        for det in msg.detections:
            assert (msg.cam_id, det.local_track_id) in gid_of


def test_nguoi_roi_fov_roi_quay_lai_bi_tach_tracklet(wildtrack_dir: Path) -> None:
    _, split, _ = build_fixture(
        wildtrack_dir, views=[1], fps=2.0, reentry_gap_frames=4, embedder=None
    )
    p8_cam01 = [t for t in split if t.gt_global_id == 8 and t.cam_id == "cam01"]
    assert len(p8_cam01) == 2  # khung 0-3 và 8-9 -> hai local track riêng
    assert len({t.local_track_id for t in p8_cam01}) == 2

    _, no_split, _ = build_fixture(
        wildtrack_dir, views=[1], fps=2.0, reentry_gap_frames=100, embedder=None
    )
    assert len([t for t in no_split if t.gt_global_id == 8 and t.cam_id == "cam01"]) == 1


def test_nguoi_xuyen_camera_co_tracklet_o_nhieu_cam(geom) -> None:
    _, tracklets, meta = geom
    cams_of_p7 = {t.cam_id for t in tracklets if t.gt_global_id == 7}
    assert cams_of_p7 == {"cam01", "cam02"}
    assert meta["n_identities"] == 3


def test_world_xy_trong_ground_truth(geom) -> None:
    _, tracklets, _ = geom
    p7 = next(t for t in tracklets if t.gt_global_id == 7 and t.cam_id == "cam01")
    assert p7.world_xy_m == pytest.approx((2.0, -6.5))


def test_views_ngoai_pham_vi_bao_loi(wildtrack_dir: Path) -> None:
    for bad in ([0], [8], [1, 9]):
        with pytest.raises(ValueError, match="--views"):
            build_fixture(wildtrack_dir, views=bad, embedder=None)


# --------------------------------------------------------------------------- #
# build_fixture — có embedding (tiêm ảnh + embedder giả)
# --------------------------------------------------------------------------- #


def test_embedding_duoc_l2_normalize_va_khop_embed_dim(wildtrack_dir: Path) -> None:
    messages, _, meta = build_fixture(
        wildtrack_dir,
        views=[1, 2, 3],
        fps=2.0,
        embedder=FakeEmbedder(),
        image_reader=_blank_image,
    )
    assert meta["embed_dim"] == 8
    assert meta["reid_model"] == "FakeEmbedder"

    seen = 0
    for msg in messages:
        assert validate(msg) == []
        for det in msg.detections:
            assert det.embedding is not None
            assert det.embedding.shape == (8,)
            assert float(np.linalg.norm(det.embedding)) == pytest.approx(1.0, abs=1e-5)
            seen += 1
    assert seen == meta["n_detections"]


def test_jsonl_round_trip(wildtrack_dir: Path) -> None:
    messages, _, _ = build_fixture(
        wildtrack_dir, views=[1, 2], fps=2.0, embedder=FakeEmbedder(), image_reader=_blank_image
    )
    for msg in messages:
        again = decode_jsonl(encode_jsonl(msg))
        assert again.cam_id == msg.cam_id
        assert len(again.detections) == len(msg.detections)


# --------------------------------------------------------------------------- #
# Ground-truth JSON
# --------------------------------------------------------------------------- #


def test_gt_json_dung_dinh_dang_make_synthetic(tmp_path: Path, geom) -> None:
    _, tracklets, meta = geom
    gt_path = tmp_path / "wildtrack.gt.json"
    write_ground_truth(gt_path, tracklets, meta)
    data = json.loads(gt_path.read_text(encoding="utf-8"))

    assert set(data) == {"scenario", "meta", "tracklets"}
    assert data["scenario"] == "wildtrack"
    for row in data["tracklets"]:
        assert {"cam_id", "local_track_id", "gt_global_id", "start_ms", "end_ms"} <= set(row)
    assert len(data["tracklets"]) == meta["n_tracklets"]


# --------------------------------------------------------------------------- #
# Đối chiếu annotation thật
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not (REAL_WILDTRACK_DIR / "annotations_positions").is_dir(),
    reason="chưa vendor annotation WildTrack thật ở tests/data/wildtrack/",
)
def test_parse_annotation_wildtrack_that() -> None:
    raw, n_frames = parse_raw_detections(
        REAL_WILDTRACK_DIR / "annotations_positions",
        view_indices=list(range(7)),
        stride=1,
        max_frames=0,
        min_box_area=100.0,
    )
    assert n_frames == 2
    assert len(raw) > 50
    assert {d.view_idx for d in raw}.issubset(set(range(7)))
    for d in raw:
        x, y, w, h = d.bbox
        assert x >= 0 and y >= 0 and x + w <= 1920 and y + h <= 1080
        assert w > 1 and h > 1


@pytest.mark.skipif(
    not (REAL_WILDTRACK_DIR / "annotations_positions").is_dir(),
    reason="chưa vendor annotation WildTrack thật ở tests/data/wildtrack/",
)
def test_build_fixture_tren_annotation_that_hop_le() -> None:
    messages, tracklets, meta = build_fixture(
        REAL_WILDTRACK_DIR, views=[1, 4, 7], fps=2.0, embedder=None
    )
    assert meta["n_identities"] > 0
    assert tracklets
    for msg in messages:
        assert validate(msg) == []
