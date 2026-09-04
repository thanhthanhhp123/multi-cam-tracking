"""Test `tools/ds_wildtrack_gt.py` — gán ground-truth cho fixture sinh từ DeepStream.

Bảng ground-truth này là thước đo; sai ở đây thì mọi kết luận về ngưỡng `max_cost` của
đường DeepStream đều sai theo mà không có dấu hiệu gì. Vì vậy test tập trung vào hai chỗ
dễ sai nhất: ghép bbox khi hai người đứng cạnh nhau, và quyết định LOẠI một track thay vì
gán bừa cho nó một danh tính.
"""

from __future__ import annotations

import numpy as np
import pytest

from common.schema import Detection, FrameMessage
from tools.ds_wildtrack_gt import (
    TrackVotes,
    build_tracklet_table,
    collect_votes,
    gt_index,
    iou,
    match_frame,
    view_idx_for_cam,
)
from tools.wildtrack_to_fixture import RawDetection, cam_id_for_view


def raw(frame_idx: int, view_idx: int, person_id: int, bbox, world=(0.0, 0.0)) -> RawDetection:
    return RawDetection(
        frame_idx=frame_idx,
        frame_number=frame_idx * 5,
        view_idx=view_idx,
        person_id=person_id,
        bbox=bbox,
        world_xy=world,
    )


def msg(cam_id: str, frame_id: int, dets: list[Detection], ts_ms: int = 1000) -> FrameMessage:
    return FrameMessage(
        cam_id=cam_id,
        frame_id=frame_id,
        ts_ms=ts_ms,
        frame_pts_ns=frame_id * 500_000_000,
        frame_width=1920,
        frame_height=1080,
        detections=dets,
        embed_dim=0,
    )


def det(track_id: int, bbox) -> Detection:
    return Detection(local_track_id=track_id, bbox=bbox, confidence=0.9)


# --------------------------------------------------------------------------------------
# IoU + ghép khung
# --------------------------------------------------------------------------------------


def test_iou_trung_khit_va_roi_nhau():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0
    # Chồng nửa theo trục x: giao 50, hợp 150.
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_ghep_mot_mot_khong_dung_mot_gt_hai_lan():
    pairs = match_frame([(0, 0, 10, 10), (2, 0, 10, 10)], [(0, 0, 10, 10)], min_iou=0.1)
    assert len(pairs) == 1
    assert pairs[0][0] == 0  # hộp khớp nhất thắng


def test_hungarian_khac_tham_lam_khi_hai_nguoi_dung_canh_nhau():
    """Cặp có IoU cao nhất KHÔNG phải cặp nên chọn — đây là chỗ tham lam sai.

    det0 khớp gt0 tốt nhất (IoU .818), nhưng lấy cặp đó thì det1 chỉ còn gt1 (.333),
    tổng .151 kém hơn phương án chéo (.667 + .667). Tình huống thật của WildTrack:
    quảng trường đông, hộp người này đè lên người kia; một lá phiếu sai ở đây kéo cả
    tracklet sang nhầm danh tính.
    """
    dets = [(11, 0, 10, 10), (8, 0, 10, 10)]
    gts = [(10, 0, 10, 10), (13, 0, 10, 10)]
    assert iou(dets[0], gts[0]) == max(iou(d, g) for d in dets for g in gts)

    pairs = sorted(match_frame(dets, gts, min_iou=0.0))
    assert [(d, g) for d, g, _ in pairs] == [(0, 1), (1, 0)]  # tham lam sẽ ra [(0,0),(1,1)]


def test_duoi_nguong_iou_thi_khong_ghep():
    assert match_frame([(0, 0, 10, 10)], [(9, 0, 10, 10)], min_iou=0.5) == []


def test_khung_rong_khong_lam_chet():
    assert match_frame([], [(0, 0, 10, 10)], min_iou=0.5) == []
    assert match_frame([(0, 0, 10, 10)], [], min_iou=0.5) == []


def test_cam_id_va_view_idx_la_nghich_dao_cua_nhau():
    for view_idx in range(7):
        assert view_idx_for_cam(cam_id_for_view(view_idx)) == view_idx


@pytest.mark.parametrize("xau", ["cam00", "cam08", "camXX", "cam1x"])
def test_cam_id_sai_bi_tu_choi(xau: str):
    with pytest.raises(ValueError):
        view_idx_for_cam(xau)


# --------------------------------------------------------------------------------------
# Bỏ phiếu -> bảng ground-truth
# --------------------------------------------------------------------------------------


def test_frame_id_tra_thang_ve_khung_chu_thich():
    """Ghép theo `frame_id`, KHÔNG theo `ts_ms` — đồng hồ lúc chạy pipeline không liên
    quan gì tới mốc thời gian giả lập của fixture ONNX Runtime."""
    gt = gt_index([raw(0, 0, 7, (0, 0, 10, 10)), raw(1, 0, 9, (0, 0, 10, 10))])
    messages = [
        msg("cam01", 0, [det(1, (0, 0, 10, 10))], ts_ms=999_999),
        msg("cam01", 1, [det(2, (0, 0, 10, 10))], ts_ms=1),
    ]
    tracks, _ = collect_votes(messages, gt, min_iou=0.5)
    assert tracks[("cam01", 1)].votes == {7: 1}
    assert tracks[("cam01", 2)].votes == {9: 1}


def test_track_thuan_khiet_duoc_giu_lai():
    tracks, stats = collect_votes(
        [msg("cam01", i, [det(1, (0, 0, 10, 10))], ts_ms=1000 + 500 * i) for i in range(4)],
        gt_index([raw(i, 0, 42, (0, 0, 10, 10), world=(1.0, 2.0)) for i in range(4)]),
        min_iou=0.5,
    )
    tracklets, drops = build_tracklet_table(tracks, min_purity=0.7, min_matched=3)

    assert stats["n_matched"] == 4
    assert drops == {"khong_khop": 0, "it_khung": 0, "khong_thuan": 0}
    assert len(tracklets) == 1
    t = tracklets[0]
    assert (t.cam_id, t.local_track_id, t.gt_global_id) == ("cam01", 1, 42)
    assert (t.start_ms, t.end_ms, t.n_frames) == (1000, 2500, 4)
    assert t.world_xy_m == pytest.approx((1.0, 2.0))


def test_track_lan_danh_tinh_bi_loai_chu_khong_gan_bua():
    """Tracker đổi người giữa chừng (id-switch): 3 phiếu cho A, 3 cho B.

    Gán bừa cho một trong hai là bơm nhiễu thẳng vào thước đo. Loại khỏi bảng thì
    `eval_wildtrack.score()` bỏ qua track đó — mất một mẫu, nhưng không sai một mẫu.
    """
    gt = gt_index([raw(i, 0, 1 if i < 3 else 2, (0, 0, 10, 10)) for i in range(6)])
    messages = [msg("cam01", i, [det(1, (0, 0, 10, 10))]) for i in range(6)]
    tracks, _ = collect_votes(messages, gt, min_iou=0.5)

    assert tracks[("cam01", 1)].winner()[1] == pytest.approx(0.5)
    tracklets, drops = build_tracklet_table(tracks, min_purity=0.7, min_matched=3)
    assert tracklets == []
    assert drops["khong_thuan"] == 1


def test_track_it_khung_bi_loai():
    gt = gt_index([raw(0, 0, 5, (0, 0, 10, 10))])
    tracks, _ = collect_votes([msg("cam01", 0, [det(1, (0, 0, 10, 10))])], gt, min_iou=0.5)
    tracklets, drops = build_tracklet_table(tracks, min_purity=0.7, min_matched=3)
    assert tracklets == [] and drops["it_khung"] == 1


def test_track_khong_khop_gt_nao_bi_loai():
    """Detector bắt được thứ WildTrack không chú thích (người ngoài rìa, bóng phản chiếu)."""
    gt = gt_index([raw(0, 0, 5, (500, 500, 10, 10))])
    tracks, _ = collect_votes([msg("cam01", 0, [det(9, (0, 0, 10, 10))])] * 3, gt, min_iou=0.5)
    tracklets, drops = build_tracklet_table(tracks, min_purity=0.7, min_matched=3)
    assert tracklets == [] and drops["khong_khop"] == 1
    assert tracks[("cam01", 9)].n_detections == 3


def test_thieu_thuan_khiet_van_giu_neu_da_so_ap_dao():
    """80% phiếu cho một người: chấp nhận, vì tracker nào cũng lem vài khung ở chỗ che khuất."""
    gt = gt_index([raw(i, 0, 1 if i < 4 else 2, (0, 0, 10, 10)) for i in range(5)])
    tracks, _ = collect_votes(
        [msg("cam01", i, [det(1, (0, 0, 10, 10))]) for i in range(5)], gt, min_iou=0.5
    )
    tracklets, _ = build_tracklet_table(tracks, min_purity=0.7, min_matched=3)
    assert [t.gt_global_id for t in tracklets] == [1]


def test_nhieu_camera_khong_lan_phieu_cua_nhau():
    """`local_track_id` chỉ duy nhất trong một camera (CLAUDE.md §5) — khoá phải là cặp."""
    gt = gt_index(
        [raw(i, 0, 11, (0, 0, 10, 10)) for i in range(3)]
        + [raw(i, 1, 22, (0, 0, 10, 10)) for i in range(3)]
    )
    messages = [msg("cam01", i, [det(1, (0, 0, 10, 10))]) for i in range(3)] + [
        msg("cam02", i, [det(1, (0, 0, 10, 10))]) for i in range(3)
    ]
    tracks, _ = collect_votes(messages, gt, min_iou=0.5)
    tracklets, _ = build_tracklet_table(tracks, min_purity=0.7, min_matched=3)
    assert {(t.cam_id, t.gt_global_id) for t in tracklets} == {("cam01", 11), ("cam02", 22)}


def test_thong_ke_khop_dem_ca_detection_khong_khop():
    gt = gt_index([raw(0, 0, 1, (0, 0, 10, 10))])
    tracks, stats = collect_votes(
        [msg("cam01", 0, [det(1, (0, 0, 10, 10)), det(2, (900, 900, 10, 10))])],
        gt,
        min_iou=0.5,
    )
    assert stats["n_detections"] == 2
    assert stats["n_matched"] == 1
    assert len(tracks) == 2


def test_track_khong_phieu_tra_ve_am_mot():
    assert TrackVotes("cam01", 1).winner() == (-1, 0.0)


def test_bang_ket_qua_dung_dinh_dang_eval_doc_duoc(tmp_path):
    """`eval/eval_wildtrack.load_gt` chỉ cần 3 khoá này — ghim lại để đừng đổi lệch."""
    from tools.wildtrack_to_fixture import write_ground_truth

    gt = gt_index([raw(i, 0, 3, (0, 0, 10, 10)) for i in range(3)])
    tracks, _ = collect_votes(
        [msg("cam01", i, [det(1, (0, 0, 10, 10))]) for i in range(3)], gt, min_iou=0.5
    )
    tracklets, _ = build_tracklet_table(tracks, min_purity=0.7, min_matched=3)

    out = tmp_path / "x.gt.json"
    write_ground_truth(out, tracklets, {"source": "test"})

    import json

    data = json.loads(out.read_text(encoding="utf-8"))
    assert {"cam_id", "local_track_id", "gt_global_id"} <= set(data["tracklets"][0])
    assert np.isfinite(data["tracklets"][0]["world_xy_m"]).all()
