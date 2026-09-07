"""Test `eval/diagnose_junk_ids.py` — tách Global ID "rác" làm detector-FP vs. nhãn-từ-chối.

Con số công cụ này in ra quyết định phiên sau đi sửa detector hay đi nới bộ gán nhãn. Sai
phép phân loại là sai cả hướng làm. Test dựng những tracklet có đáp án rõ: hộp không bao
giờ chạm người thật (detector báo nhầm), hộp bám sát một người nhưng quá ngắn, hộp bám
người nhưng phiếu lẫn — rồi soi đúng ô số.
"""

from __future__ import annotations

import json
import sqlite3

from eval.diagnose_global_ids import Appearance
from eval.diagnose_junk_ids import (
    CLASS_FP,
    CLASS_REAL,
    aggregate_by_gid,
    classify_tracks,
    junk_global_ids,
    main,
)

from tools.ds_wildtrack_gt import TrackVotes


def ap(gid: int, cam: str, local: int, frames: int, gt_id: int | None) -> Appearance:
    return Appearance(
        global_id=gid,
        cam_id=cam,
        local_track_id=local,
        n_frames=frames,
        start_ms=local * 1000,
        reason="",
        gt_id=gt_id,
    )


def votes(cam: str, local: int, n_det: int, n_match: int, pid: int) -> TrackVotes:
    tv = TrackVotes(cam, local)
    tv.n_detections = n_det
    tv.n_matched = n_match
    if n_match:
        tv.votes[pid] = n_match
    return tv


def test_junk_gid_can_moi_tracklet_khong_nhan() -> None:
    aps = [
        ap(1, "cam01", 1, 10, None),
        ap(1, "cam02", 2, 10, None),  # gid 1: toàn rác
        ap(2, "cam01", 3, 10, 99),  # gid 2: có một tracklet nhãn -> không rác
        ap(2, "cam02", 4, 10, None),
    ]
    assert junk_global_ids(aps) == {1}


def test_phan_loai_ba_kich_ban() -> None:
    aps = [
        ap(10, "cam01", 1, 40, None),  # hộp báo nhầm: khớp 2/40
        ap(11, "cam01", 2, 5, None),  # người thật nhưng chỉ 2 khung khớp
        ap(12, "cam01", 3, 30, None),  # người thật, phiếu lẫn 60%
    ]
    tracks = {
        ("cam01", 1): votes("cam01", 1, 40, 2, 7),
        ("cam01", 2): votes("cam01", 2, 5, 2, 7),
        ("cam01", 3): votes("cam01", 3, 30, 27, 7),
    }
    # track 3: 27 khớp nhưng phiếu chia 60/40 -> purity 0.6 < 0.7
    tracks[("cam01", 3)].votes = type(tracks[("cam01", 3)].votes)({7: 16, 8: 11})
    tracks[("cam01", 3)].n_matched = 27

    junk = {10, 11, 12}
    vs = classify_tracks(aps, tracks, junk, fp_match_rate=0.30, min_purity=0.7, min_matched=3)
    by_local = {v.local_track_id: v for v in vs}
    assert by_local[1].klass == CLASS_FP  # 2/40 = 0.05 < 0.30
    assert by_local[2].klass == CLASS_REAL and by_local[2].subreason == "it_khung"
    assert by_local[3].klass == CLASS_REAL and by_local[3].subreason == "khong_thuan"


def test_track_vang_trong_fixture_thanh_fp() -> None:
    aps = [ap(5, "cam01", 9, 12, None)]
    vs = classify_tracks(aps, {}, {5}, fp_match_rate=0.3, min_purity=0.7, min_matched=3)
    assert vs[0].klass == "not_in_fixture"
    assert aggregate_by_gid(vs) == {5: CLASS_FP}


def test_aggregate_bo_phieu_theo_khung() -> None:
    aps = [
        ap(7, "cam01", 1, 100, None),  # người thật, nặng
        ap(7, "cam02", 2, 5, None),  # báo nhầm, nhẹ
    ]
    tracks = {
        ("cam01", 1): votes("cam01", 1, 100, 95, 3),
        ("cam02", 2): votes("cam02", 2, 5, 0, -1),
    }
    vs = classify_tracks(aps, tracks, {7}, fp_match_rate=0.3, min_purity=0.7, min_matched=3)
    assert aggregate_by_gid(vs) == {7: CLASS_REAL}  # 100 khung người thật > 5 khung nhầm


def test_main_dau_cuoi(tmp_path) -> None:
    """Đường end-to-end nhỏ: 1 khung, 2 camera, hộp GT + hộp lạc."""

    # --- WildTrack annotation: person 1 ở view 0 và view 1, khung 0
    ann_dir = tmp_path / "wt" / "annotations_positions"
    ann_dir.mkdir(parents=True)
    (ann_dir / "00000000.json").write_text(
        json.dumps(
            [
                {
                    "personID": 1,
                    "positionID": 200000,
                    "views": [
                        {"viewNum": 0, "xmin": 100, "ymin": 100, "xmax": 150, "ymax": 250},
                        {"viewNum": 1, "xmin": 300, "ymin": 100, "xmax": 350, "ymax": 250},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    # --- fixture: cam01 có hộp trùng người 1 + hộp lạc; cam02 có hộp trùng người 1
    fixture = tmp_path / "fx.jsonl"
    rows = []
    for _ in range(5):  # 5 khung để vượt min_matched
        rows.append(
            {
                "schema_version": 1,
                "cam_id": "cam01",
                "frame_id": 0,
                "ts_ms": 1000,
                "frame_pts_ns": 0,
                "frame_width": 1920,
                "frame_height": 1080,
                "embed_dim": 0,
                "detections": [
                    {"local_track_id": 1, "bbox": [100, 100, 50, 150], "confidence": 0.9},
                    {"local_track_id": 2, "bbox": [900, 900, 40, 80], "confidence": 0.9},
                ],
            }
        )
        rows.append(
            {
                "schema_version": 1,
                "cam_id": "cam02",
                "frame_id": 0,
                "ts_ms": 1000,
                "frame_pts_ns": 0,
                "frame_width": 1920,
                "frame_height": 1080,
                "embed_dim": 0,
                "detections": [
                    {"local_track_id": 1, "bbox": [300, 100, 50, 150], "confidence": 0.9},
                ],
            }
        )
    fixture.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    # --- .gt.json rỗng danh tính phủ (mọi tracklet đều "rác")
    gt = tmp_path / "fx.gt.json"
    gt.write_text(json.dumps({"tracklets": []}), encoding="utf-8")

    # --- DB: 3 Global ID rác
    db = tmp_path / "s.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE appearances (global_id INTEGER, cam_id TEXT, local_track_id INTEGER, "
        "n_frames INTEGER, start_ms INTEGER, reason TEXT)"
    )
    con.executemany(
        "INSERT INTO appearances VALUES (?,?,?,?,?,?)",
        [
            (1, "cam01", 1, 5, 0, "threshold: x"),
            (2, "cam01", 2, 5, 0, "threshold: x"),
            (3, "cam02", 1, 5, 0, "threshold: x"),
        ],
    )
    con.commit()
    con.close()

    rc = main(
        [
            "--db",
            str(db),
            "--gt",
            str(gt),
            "--fixture",
            str(fixture),
            "--wildtrack-dir",
            str(tmp_path / "wt"),
            "--json",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 0
    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert out["n_junk_gids"] == 3
    # gid 1 (cam01/local1) và gid 3 (cam02/local1) bám người thật; gid 2 là hộp lạc
    assert out["by_class"][CLASS_REAL]["n_gids"] == 2
    assert out["by_class"][CLASS_FP]["n_gids"] == 1
    assert out["n_real_person_ids_new"] == 1  # person 1 chưa có trong .gt.json
