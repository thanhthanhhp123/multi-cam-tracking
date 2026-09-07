"""Test `eval/diagnose_global_ids.py`.

Công cụ này phân rã tổng số Global ID thành rác / vỡ / gộp, và con số đó quyết định phiên
sau đi sửa ngưỡng liên kết hay đi sửa detector. Sai một phép đếm là sai cả hướng làm, mà
báo cáo vẫn in ra đẹp đẽ. Vì vậy test dựng những kịch bản có đáp án đếm được bằng tay —
một người liền mạch, một người bị xé đôi, một Global ID ôm hai người, một Global ID toàn
tracklet không có nhãn — rồi soi đúng ô số tương ứng.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from eval.diagnose_global_ids import (
    Appearance,
    birth_reasons,
    birth_rows,
    decompose,
    fragmentation,
    load_appearances,
    load_gt,
    main,
    main_fragment_ids,
    merges,
    surplus_mass,
)

from mct.associator import reason_kind
from mct.store import Store

# (global_id, cam_id, local_track_id, n_frames, gt_id | None, reason)
Row = tuple[int, str, int, int, "int | None", str]


def ap(
    gid: int, cam: str, local: int, frames: int, gt_id: int | None, reason: str = ""
) -> Appearance:
    return Appearance(
        global_id=gid,
        cam_id=cam,
        local_track_id=local,
        n_frames=frames,
        start_ms=local * 1000,
        reason=reason,
        gt_id=gt_id,
    )


# --------------------------------------------------------------------- phân rã


def test_lien_ket_hoan_hao_khong_co_phan_du() -> None:
    """Hai người, mỗi người 3 tracklet ở 3 camera, gộp đúng thành 2 Global ID."""
    aps = [
        ap(1, "cam01", 1, 10, 100),
        ap(1, "cam02", 2, 10, 100),
        ap(1, "cam03", 3, 10, 100),
        ap(2, "cam01", 4, 10, 200),
        ap(2, "cam02", 5, 10, 200),
    ]
    out = decompose(aps, n_gt_identities=2)
    assert out["n_global_ids"] == 2
    assert out["n_identities_covered"] == 2
    assert out["surplus_split"] == 0
    assert out["merge_overlap"] == 0
    assert out["n_unlabeled"] == 0


def test_mot_nguoi_bi_xe_ba_manh() -> None:
    aps = [
        ap(1, "cam01", 1, 10, 100),
        ap(2, "cam02", 2, 10, 100),
        ap(3, "cam03", 3, 10, 100),
    ]
    out = decompose(aps, n_gt_identities=1)
    assert out["n_global_ids"] == 3
    assert out["n_identities_covered"] == 1
    assert out["surplus_split"] == 2  # 3 Global ID cho 1 danh tính
    assert out["merge_overlap"] == 0


def test_mot_global_id_om_hai_nguoi() -> None:
    aps = [ap(1, "cam01", 1, 10, 100), ap(1, "cam02", 2, 10, 200)]
    out = decompose(aps, n_gt_identities=2)
    assert out["n_global_ids"] == 1
    assert out["n_identities_covered"] == 2
    assert out["surplus_split"] == 0
    assert out["merge_overlap"] == 1  # 1 = 0 + 2 + 0 - 1


def test_global_id_toan_tracklet_khong_nhan_la_rac() -> None:
    """Rác đếm riêng: nó không nói gì về chất lượng của bước liên kết."""
    aps = [
        ap(1, "cam01", 1, 10, 100),
        ap(2, "cam01", 2, 10, None),
        ap(2, "cam02", 3, 5, None),
        ap(3, "cam03", 4, 7, None),
    ]
    out = decompose(aps, n_gt_identities=1)
    assert out["n_global_ids"] == 3
    assert out["n_unlabeled"] == 2
    assert out["n_labeled"] == 1
    assert out["frames_unlabeled"] == 22


def test_mot_tracklet_co_nhan_la_du_de_khong_bi_tinh_la_rac() -> None:
    aps = [ap(1, "cam01", 1, 10, 100), ap(1, "cam02", 2, 90, None)]
    out = decompose(aps, n_gt_identities=1)
    assert out["n_unlabeled"] == 0
    assert out["frames_unlabeled"] == 90


def test_dang_thuc_phan_ra_bat_duoc_moi_truong_hop_tron() -> None:
    """Vỡ và gộp cùng lúc: đẳng thức phải khớp, nếu không `decompose` đã ném lỗi."""
    aps = [
        ap(1, "cam01", 1, 10, 100),  # người 100, mảnh chính
        ap(2, "cam02", 2, 4, 100),  # người 100, mảnh dư
        ap(2, "cam03", 3, 20, 200),  # ... mảnh dư đó lại lẫn người 200
        ap(9, "cam01", 4, 6, None),  # rác
    ]
    out = decompose(aps, n_gt_identities=2)
    assert out["n_global_ids"] == 3
    assert (
        out["n_unlabeled"]
        + out["n_identities_covered"]
        + out["surplus_split"]
        - out["merge_overlap"]
        == out["n_global_ids"]
    )


# ------------------------------------------------------------------ khối lượng


def test_manh_chinh_chon_theo_so_khung_khong_theo_so_tracklet() -> None:
    """5 tracklet ngắn không phải chỗ hệ thống nhận ra người đó; 1 tracklet dài mới là."""
    aps = [ap(7, "cam01", 1, 100, 100)] + [
        ap(8, "cam02", i, 3, 100)
        for i in range(2, 7)  # 5 tracklet, tổng 15 khung
    ]
    assert main_fragment_ids(aps) == {7}
    out = surplus_mass(aps)
    assert out["n_surplus_fragments"] == 1
    assert out["frames_outside_main"] == 15
    assert out["frames_labeled_total"] == 115
    assert out["n_surplus_single_tracklet"] == 0  # mảnh dư này gồm 5 tracklet


def test_nguoi_bi_chia_doi_mat_nua_so_khung() -> None:
    aps = [ap(1, "cam01", 1, 50, 100), ap(2, "cam02", 2, 50, 100)]
    out = surplus_mass(aps)
    assert out["frames_outside_main"] == 50
    assert out["share_outside_main_pct"] == pytest.approx(50.0)
    assert out["main_share_pct_quantiles"][2] == pytest.approx(50.0)


def test_lien_ket_hoan_hao_thi_khong_mat_khung_nao() -> None:
    aps = [ap(1, "cam01", 1, 10, 100), ap(1, "cam02", 2, 10, 100)]
    out = surplus_mass(aps)
    assert out["n_surplus_fragments"] == 0
    assert out["frames_outside_main"] == 0
    assert out["share_outside_main_pct"] == pytest.approx(0.0)


def test_vo_dem_theo_danh_tinh_va_xep_hang_ca_nang_nhat() -> None:
    aps = [
        ap(1, "cam01", 1, 10, 100),
        ap(2, "cam02", 2, 10, 100),
        ap(3, "cam03", 3, 10, 100),
        ap(4, "cam01", 4, 10, 200),
    ]
    out = fragmentation(aps, top=2)
    assert out["n_identities"] == 2
    assert out["gids_per_identity_hist"] == {"1": 1, "3": 1}
    assert out["mean_gids_per_identity"] == pytest.approx(2.0)
    assert out["worst"][0] == {"gt_id": 100, "n_gids": 3, "n_frames": 30, "n_cams": 3}


def test_gop_dem_khung_thieu_so() -> None:
    aps = [
        ap(1, "cam01", 1, 80, 100),
        ap(1, "cam02", 2, 20, 200),  # thiểu số trong Global ID 1
        ap(2, "cam03", 3, 30, 300),
    ]
    out = merges(aps)
    assert out["n_labeled_gids"] == 2
    assert out["n_impure_gids"] == 1
    assert out["frames_minority"] == 20
    assert out["frames_labeled_total"] == 130
    assert out["identities_per_gid_hist"] == {"1": 1, "2": 1}


# ------------------------------------------------------------------- lý do sinh


def test_reason_kind_cat_tien_to_chu_khong_doan_cau_tieng_viet() -> None:
    assert reason_kind("threshold: ứng viên tốt nhất có cost=0.95 >= max_cost=0.9") == "threshold"
    khong_ung_vien = "no_candidate: không còn ứng viên khả thi sau ràng buộc (x)"
    assert reason_kind(khong_ung_vien) == "no_candidate"
    assert reason_kind("") == ""
    # Tiền tố lạ không được nhận bừa thành một loại.
    assert reason_kind("linh tinh: gì đó") == ""


def test_ly_do_sinh_tach_manh_chinh_manh_du_va_rac() -> None:
    aps = [
        ap(1, "cam01", 1, 50, 100, "empty: gallery đang rỗng, đây là người đầu tiên"),
        ap(2, "cam02", 2, 10, 100, "threshold: ứng viên tốt nhất có cost=0.95 >= max_cost=0.9"),
        ap(3, "cam03", 3, 5, None, "no_candidate: không còn ứng viên khả thi sau ràng buộc (x)"),
    ]
    out = birth_reasons(aps)
    assert out["empty"] == {"mảnh chính": 1, "mảnh dư": 0, "rác": 0}
    assert out["threshold"] == {"mảnh chính": 0, "mảnh dư": 1, "rác": 0}
    assert out["no_candidate"] == {"mảnh chính": 0, "mảnh dư": 0, "rác": 1}


def test_dong_khai_sinh_la_dong_CO_ly_do_khong_phai_dong_som_nhat() -> None:
    """Tracklet bắt đầu sớm nhưng được GHÉP vào track có sẵn không phải người khai sinh.

    Lấy theo `start_ms` thì dòng rỗng lý do ở trên sẽ che mất dòng thật, và mục 5 báo cáo
    một Global ID "không rõ vì sao ra đời" trong khi lý do nằm ngay dòng dưới.
    """
    aps = [
        Appearance(1, "cam01", 1, 10, start_ms=1000, reason="", gt_id=100),
        Appearance(1, "cam02", 2, 10, start_ms=5000, reason="threshold: x", gt_id=100),
    ]
    assert birth_rows(aps)[1].local_track_id == 2
    out = birth_reasons(aps)
    assert out["threshold"] == {"mảnh chính": 1, "mảnh dư": 0, "rác": 0}


def test_nhieu_dong_co_ly_do_thi_lay_dong_som_nhat() -> None:
    """Tracklet từng tạo Global ID khác rồi bị gán lại mang `reason` cũ theo sang."""
    aps = [
        Appearance(1, "cam01", 1, 10, start_ms=1000, reason="empty: x", gt_id=100),
        Appearance(1, "cam02", 2, 10, start_ms=5000, reason="threshold: y", gt_id=100),
    ]
    assert birth_rows(aps)[1].local_track_id == 1


def test_global_id_khong_con_dong_nao_giu_ly_do_dem_rieng() -> None:
    aps = [Appearance(1, "cam01", 1, 10, start_ms=1000, reason="", gt_id=100)]
    out = birth_reasons(aps)
    assert out[""] == {"mảnh chính": 1, "mảnh dư": 0, "rác": 0}


# ----------------------------------------------------------------- đọc từ đĩa


def _write_db(path: Path, rows: list[Row]) -> None:
    """Tạo file bằng CHÍNH `Store` (để schema không trôi khỏi bản thật), rồi chèn tay."""
    store = Store(db_path=str(path))
    store.close()
    con = sqlite3.connect(path)
    with con:
        con.executemany(
            "INSERT INTO appearances (tracklet_id, global_id, cam_id, local_track_id, "
            "start_ms, end_ms, n_frames, cost, reason) VALUES (?, ?, ?, ?, ?, ?, ?, 0.0, ?)",
            [
                (i + 1, gid, cam, local, local * 1000, local * 1000 + 500, frames, reason)
                for i, (gid, cam, local, frames, _gt, reason) in enumerate(rows)
            ],
        )
    con.close()


def _write_gt(path: Path, rows: list[Row]) -> None:
    path.write_text(
        json.dumps(
            {
                "tracklets": [
                    {"cam_id": cam, "local_track_id": local, "gt_global_id": gt}
                    for _gid, cam, local, _frames, gt, _reason in rows
                    if gt is not None
                ]
            }
        ),
        encoding="utf-8",
    )


ROWS: list[Row] = [
    (1, "cam01", 1, 50, 100, "empty: gallery đang rỗng"),
    (2, "cam02", 2, 10, 100, "threshold: cost=0.95 >= max_cost=0.9"),
    (3, "cam03", 3, 7, None, "no_candidate: hết ứng viên"),
]


def test_doc_db_va_bang_gt_ghep_dung_nhan(tmp_path: Path) -> None:
    db, gt_path = tmp_path / "s.db", tmp_path / "s.gt.json"
    _write_db(db, ROWS)
    _write_gt(gt_path, ROWS)

    aps = load_appearances(db, load_gt(gt_path))
    assert [a.gt_id for a in aps] == [100, 100, None]
    assert [a.n_frames for a in aps] == [50, 10, 7]
    assert reason_kind(aps[0].reason) == "empty"


def test_chay_ca_cli_tren_db_that(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db, gt_path, out = tmp_path / "s.db", tmp_path / "s.gt.json", tmp_path / "out.json"
    _write_db(db, ROWS)
    _write_gt(gt_path, ROWS)

    code = main(["--db", str(db), "--gt", str(gt_path), "--json", str(out)])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["decomposition"] == {
        "n_global_ids": 3,
        "n_unlabeled": 1,
        "n_labeled": 2,
        "n_identities_covered": 1,
        "n_identities_gt": 1,
        "surplus_split": 1,
        "merge_overlap": 0,
        "n_appearances": 3,
        "frames_total": 67,
        "frames_unlabeled": 7,
        "n_local_tracks": 3,
        "n_local_tracks_labeled": 2,
    }
    assert report["surplus_mass"]["frames_outside_main"] == 10
    assert "PHÂN RÃ TỔNG SỐ GLOBAL ID" in capsys.readouterr().out


def test_db_rong_bao_loi_thay_vi_in_bang_toan_so_khong(tmp_path: Path) -> None:
    db, gt_path = tmp_path / "s.db", tmp_path / "s.gt.json"
    _write_db(db, [])
    _write_gt(gt_path, ROWS)
    assert main(["--db", str(db), "--gt", str(gt_path)]) == 1
