"""Test `tools/assign_gt.py` — gán GT cho fixture pipeline từ một fixture ground-truth bất kỳ.

Đường của M6: CVAT → `cvat_to_mot --fixture-out` → công cụ này → `export_trackeval`. Lỗi ở
đây cho ra bảng điểm sai mà vẫn chạy, nên canh: khung không chú thích không bỏ phiếu, track
lẫn người bị loại, và lệch số khung thì phải lộ ra (match_rate ≈ 0) chứ không gán bừa.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.schema import Detection, FrameMessage, write_jsonl
from tools.assign_gt import build_table, collect_votes, index_gt, main
from tools.export_trackeval import load_gt_table

BASE = 1_788_231_600_000


def _msg(cam: str, frame: int, dets: list[tuple[int, tuple[float, float, float, float]]]):
    return FrameMessage(
        cam_id=cam,
        frame_id=frame,
        ts_ms=BASE + frame * 40,
        frame_pts_ns=frame * 40_000_000,
        frame_width=1920,
        frame_height=1080,
        detections=[Detection(local_track_id=t, bbox=b, confidence=0.9) for t, b in dets],
    )


A = (100.0, 100.0, 50.0, 150.0)
B = (600.0, 120.0, 55.0, 160.0)


def _shift(box, dx=3.0):
    return (box[0] + dx, box[1], box[2], box[3])


def _write_gt(tmp_path: Path, frames: range) -> Path:
    """GT: người 1 = hộp A (track CVAT 0), người 2 = hộp B (track CVAT 1), ở cam01."""
    path = tmp_path / "gt.jsonl"
    write_jsonl(path, [_msg("cam01", f, [(0, A), (1, B)]) for f in frames])
    table = {
        "tracklets": [
            {"cam_id": "cam01", "local_track_id": 0, "gt_global_id": 1},
            {"cam_id": "cam01", "local_track_id": 1, "gt_global_id": 2},
        ]
    }
    path.with_name("gt.gt.json").write_text(json.dumps(table), encoding="utf-8")
    return path


def test_gan_dung_track_cua_tracker():
    gt = index_gt([_msg("cam01", f, [(0, A), (1, B)]) for f in range(5)], {("cam01", 0): 7})
    # track 1 của GT không có trong bảng → không phải ground-truth, không bỏ phiếu được
    assert all(len(v) == 1 for v in gt.values())

    pipe = [_msg("cam01", f, [(42, _shift(A))]) for f in range(5)]
    tracks, stats = collect_votes(pipe, gt, min_iou=0.5)
    rows, _ = build_table(tracks, min_purity=0.7, min_matched=3)
    assert [(r["local_track_id"], r["gt_global_id"]) for r in rows] == [(42, 7)]
    assert stats["n_matched"] == 5


def test_khung_khong_chu_thich_khong_bo_phieu():
    """Chú thích nhảy 3 khung: chỉ khung 0, 3, 6 được so; các khung khác không tính."""
    gt = index_gt([_msg("cam01", f, [(0, A)]) for f in (0, 3, 6)], {("cam01", 0): 1})
    pipe = [_msg("cam01", f, [(5, A)]) for f in range(7)]
    tracks, stats = collect_votes(pipe, gt, min_iou=0.5)
    assert stats["n_messages_annotated"] == 3
    assert tracks[("cam01", 5)].n_matched == 3
    assert tracks[("cam01", 5)].n_detections == 7


def test_khung_chu_thich_rong_van_dem():
    """Khung có chú thích nhưng không có ai: hộp của pipeline ở đó là báo nhầm."""
    gt = index_gt([_msg("cam01", 0, [])], {})
    _, stats = collect_votes([_msg("cam01", 0, [(1, A)])], gt, min_iou=0.5)
    assert stats["n_detections_annotated"] == 1
    assert stats["n_matched"] == 0


def test_track_lan_hai_nguoi_bi_loai():
    gt = index_gt(
        [_msg("cam01", f, [(0, A), (1, B)]) for f in range(10)],
        {("cam01", 0): 1, ("cam01", 1): 2},
    )
    # track 9 nửa đầu là A, nửa sau nhảy sang B: độ thuần khiết 0.5 < 0.7
    pipe = [_msg("cam01", f, [(9, A if f < 5 else B)]) for f in range(10)]
    tracks, _ = collect_votes(pipe, gt, min_iou=0.5)
    rows, drops = build_table(tracks, min_purity=0.7, min_matched=3)
    assert rows == []
    assert drops["khong_thuan"] == 1


def test_frame_offset():
    gt = index_gt([_msg("cam01", f + 10, [(0, A)]) for f in range(4)], {("cam01", 0): 1})
    pipe = [_msg("cam01", f, [(3, A)]) for f in range(4)]
    tracks, _ = collect_votes(pipe, gt, min_iou=0.5)
    assert tracks[("cam01", 3)].n_matched == 0
    tracks, _ = collect_votes(pipe, gt, min_iou=0.5, frame_offset=10)
    assert tracks[("cam01", 3)].n_matched == 4


def test_chay_tron(tmp_path: Path):
    gt_path = _write_gt(tmp_path, range(0, 30, 3))
    fixture = tmp_path / "run.jsonl"
    write_jsonl(
        fixture, [_msg("cam01", f, [(11, _shift(A)), (12, _shift(B, -2))]) for f in range(30)]
    )
    report = tmp_path / "report.json"
    args = ["--fixture", str(fixture), "--gt-fixture", str(gt_path), "--report", str(report)]
    assert main(args) == 0
    table = load_gt_table(tmp_path / "run.gt.json")
    assert table == {("cam01", 11): 1, ("cam01", 12): 2}
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["match_rate"] == pytest.approx(1.0)
    assert rep["gt_recall"] == pytest.approx(1.0)
    payload = json.loads((tmp_path / "run.gt.json").read_text(encoding="utf-8"))
    row = payload["tracklets"][0]
    assert row["start_ms"] == BASE and row["end_ms"] == BASE + 29 * 40


def test_lech_so_khung_thi_khong_gan_bua(tmp_path: Path):
    """Hai fixture không thẳng hàng (người ở chỗ khác hẳn) → dừng, không viết bảng rỗng."""
    gt_path = _write_gt(tmp_path, range(5))
    fixture = tmp_path / "run.jsonl"
    write_jsonl(fixture, [_msg("cam01", f, [(1, (1500.0, 800.0, 40.0, 90.0))]) for f in range(5)])
    with pytest.raises(SystemExit, match="không gán được"):
        main(["--fixture", str(fixture), "--gt-fixture", str(gt_path)])
