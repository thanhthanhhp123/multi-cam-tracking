"""Test `tools/estimate_transit.py` và `eval/eval_handover.py`.

Kịch bản tổng hợp, 3 camera ở 10 fps: cam01–cam02 chồng lấn, cam03 không chồng lấn với ai.

- Người 1: cam01 khung 0–19, cam02 khung 10–29 (cùng lúc ở cả hai), cam03 khung 60–79.
- Người 2: cam03 khung 0–9, ra khỏi khung 3 giây, quay lại cam03 khung 40–49.

Canh: Δt âm cho cặp chồng lấn, chọn đúng "lần thấy cuối" làm nguồn, cắt lần xuất hiện theo
khoảng trống, và bốn kết cục đúng / tách / nhầm người / sót.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.compare_oracle_tracker import write_identity_db
from eval.eval_handover import (
    edge_id,
    id_owners,
    main,
    predicted_ids,
    score_handovers,
    summarize,
)

from common.schema import Detection, FrameMessage, write_jsonl
from tools.estimate_transit import (
    PairStats,
    build_appearances,
    handovers,
    load_overlaps,
    pair_kind,
    pair_stats,
)
from tools.estimate_transit import main as transit_main
from tools.export_trackeval import load_global_ids

BASE = 1_788_231_600_000
BOX = {1: (100.0, 100.0, 50.0, 150.0), 2: (700.0, 100.0, 50.0, 150.0)}
OVERLAPS = {"cam01": {"cam02"}, "cam02": {"cam01"}}

# (người, camera, khung đầu, khung cuối)
SCRIPT = [
    (1, "cam01", 0, 19),
    (1, "cam02", 10, 29),
    (1, "cam03", 60, 79),
    (2, "cam03", 0, 9),
    (2, "cam03", 40, 49),
]


def _msg(cam: str, frame: int, dets: list[Detection]) -> FrameMessage:
    return FrameMessage(
        cam_id=cam,
        frame_id=frame,
        ts_ms=BASE + frame * 100,
        frame_pts_ns=frame * 100_000_000,
        frame_width=1920,
        frame_height=1080,
        detections=dets,
    )


def gt_messages() -> list[FrameMessage]:
    """GT: track CVAT = người (một track mỗi người mỗi camera, kể cả lúc quay lại)."""
    out = []
    for cam in ("cam01", "cam02", "cam03"):
        for frame in range(80):
            dets = [
                Detection(local_track_id=p, bbox=BOX[p], confidence=1.0)
                for p, c, a, b in SCRIPT
                if c == cam and a <= frame <= b
            ]
            out.append(_msg(cam, frame, dets))
    return out


TABLE = {(cam, p): p for p, cam, _, _ in SCRIPT}


def result_messages(local_of) -> list[FrameMessage]:
    """Kết quả hệ thống: cùng hộp (lệch 2 px), `local_of(người, camera, khung)` cấp local id."""
    out = []
    for cam in ("cam01", "cam02", "cam03"):
        for frame in range(80):
            dets = []
            for p, c, a, b in SCRIPT:
                if c == cam and a <= frame <= b:
                    x, y, w, h = BOX[p]
                    tid = local_of(p, cam, frame)
                    if tid is not None:
                        dets.append(Detection(tid, (x + 2, y, w, h), confidence=0.9))
            out.append(_msg(cam, frame, dets))
    return out


# --------------------------------------------------------------------------------------
# estimate_transit
# --------------------------------------------------------------------------------------


def test_cat_lan_xuat_hien_theo_khoang_trong():
    apps = build_appearances(gt_messages(), TABLE, gap_ms=2000)
    assert [(a.person, a.cam_id, a.boxes[0][0], a.boxes[-1][0]) for a in apps] == [
        (1, "cam01", 0, 19),
        (1, "cam02", 10, 29),
        (1, "cam03", 60, 79),
        (2, "cam03", 0, 9),
        (2, "cam03", 40, 49),
    ]
    # khoảng trống 3 s < gap 5 s thì là MỘT lần xuất hiện
    assert len(build_appearances(gt_messages(), TABLE, gap_ms=5000)) == 4


def test_lan_chuyen_va_dt():
    items = handovers(build_appearances(gt_messages(), TABLE))
    got = sorted((h.dst.person, h.src.cam_id, h.dst.cam_id, h.elapsed_ms) for h in items)
    assert got == [
        (1, "cam01", "cam02", 1000 - 1900),  # cam02 hiện khi cam01 còn thấy: Δt âm
        # nguồn là cam02 (thấy cuối muộn nhất), không phải cam01
        (1, "cam02", "cam03", 6000 - 2900),
        (2, "cam03", "cam03", 4000 - 900),  # quay lại cùng camera
    ]


def test_loai_cap():
    assert pair_kind("cam01", "cam02", OVERLAPS) == "overlap"
    assert pair_kind("cam02", "cam03", OVERLAPS) == "non_overlap"
    assert pair_kind("cam03", "cam03", OVERLAPS) == "same_camera"


def test_de_xuat_transit():
    items = handovers(build_appearances(gt_messages(), TABLE))
    stats = {(s.src, s.dst): s for s in pair_stats(items)}
    assert ("cam03", "cam03") not in stats  # quay lại cùng camera không phải transit
    non = stats[("cam02", "cam03")].suggestion(margin=0.5, slack_ms=1000)
    assert non == {"from": "cam02", "to": "cam03", "min_ms": 1550, "max_ms": 5650}
    ov = stats[("cam01", "cam02")]
    assert ov.n_negative == 1
    assert ov.suggestion(margin=0.0, slack_ms=0) == {
        "from": "cam01",
        "to": "cam02",
        "min_ms": 0,
        "max_ms": 900,
    }


def test_de_xuat_khong_bao_gio_am():
    s = PairStats("a", "b", n=3, min_ms=100, median_ms=200.0, max_ms=300, n_negative=0)
    assert s.suggestion(margin=2.0, slack_ms=0)["min_ms"] == 0


def test_doc_overlaps_tu_topology(tmp_path: Path):
    topo = tmp_path / "t.yaml"
    topo.write_text(
        "cameras:\n  cam01: {overlaps_with: [cam02]}\n  cam02: {}\n  cam03: {}\n",
        encoding="utf-8",
    )
    assert load_overlaps(topo) == {"cam01": {"cam02"}, "cam02": {"cam01"}}
    assert load_overlaps(None) == {}


def test_estimate_transit_chay_tron(tmp_path: Path, capsys):
    gt = tmp_path / "gt.jsonl"
    write_jsonl(gt, gt_messages())
    _write_table(gt.with_name("gt.gt.json"))
    out = tmp_path / "sug.yaml"
    assert transit_main(["--gt-fixture", str(gt), "--yaml-out", str(out), "--min-n", "1"]) == 0
    text = out.read_text(encoding="utf-8")
    assert "cam02" in text and "overlap_pairs_detected" in text
    assert "cam01 → cam02" in capsys.readouterr().out


def _write_table(path: Path) -> None:
    rows = [{"cam_id": c, "local_track_id": t, "gt_global_id": g} for (c, t), g in TABLE.items()]
    path.write_text(json.dumps({"tracklets": rows}), encoding="utf-8")


# --------------------------------------------------------------------------------------
# eval_handover
# --------------------------------------------------------------------------------------


def _score(local_of, gid_of, tmp_path: Path, *, edge_frames: int = 3):
    db = tmp_path / "mct.db"
    write_identity_db(gid_of, db)
    gt = gt_messages()
    pred = predicted_ids(result_messages(local_of), gt, TABLE, load_global_ids(db), min_iou=0.5)
    items = handovers(build_appearances(gt, TABLE))
    return score_handovers(items, pred, OVERLAPS, edge_frames=edge_frames), pred


def _outcomes(scored) -> dict[tuple[str, str], str]:
    return {(s.handover.src.cam_id, s.handover.dst.cam_id): s.outcome for s in scored}


def test_he_thong_hoan_hao_thi_dung_het(tmp_path: Path):
    scored, _ = _score(
        lambda p, cam, f: p,
        {("cam01", 1): 10, ("cam02", 1): 10, ("cam03", 1): 10, ("cam03", 2): 20},
        tmp_path,
    )
    assert set(_outcomes(scored).values()) == {"dung"}
    summary = summarize(scored)
    assert summary["by_kind"]["overlap"]["accuracy"] == 1.0
    assert summary["by_kind"]["non_overlap"]["n"] == 1
    assert summary["by_kind"]["same_camera"]["n"] == 1


def test_tach(tmp_path: Path):
    """Người 2 quay lại cam03 với local id mới và Global ID mới 30: tách. Người 1 sang cam03
    nhận Global ID 20 mà người 2 cũng mang, nhưng người 1 chiếm đa số hộp của 20 (20 so với
    10) nên 20 là "của" người 1: đó là tách, không phải nhầm người."""

    def local_of(p, cam, f):
        if p == 2 and f >= 40:
            return 5  # tracker cấp id mới khi quay lại
        return p

    gid_of = {("cam01", 1): 10, ("cam02", 1): 10, ("cam03", 1): 20, ("cam03", 2): 20}
    gid_of[("cam03", 5)] = 30
    scored, pred = _score(local_of, gid_of, tmp_path)
    assert id_owners(pred)[20] == 1
    assert _outcomes(scored) == {
        ("cam01", "cam02"): "dung",
        ("cam02", "cam03"): "tach",
        ("cam03", "cam03"): "tach",
    }


def test_nham_nguoi_ro_rang(tmp_path: Path):
    """Global ID của người 2 (nhiều hộp hơn) bị gán cho người 1 ở cam03 → nhầm người."""
    gid_of = {("cam01", 1): 10, ("cam02", 1): 10, ("cam03", 1): 20, ("cam03", 2): 20}

    def local_of(p, cam, f):
        if p == 1 and cam == "cam03" and f >= 66:
            return None  # hệ thống chỉ thấy người 1 ở cam03 sáu khung đầu
        return p

    scored, pred = _score(local_of, gid_of, tmp_path)
    assert id_owners(pred)[20] == 2
    assert _outcomes(scored)[("cam02", "cam03")] == "nham"


def test_sot_khi_mot_dau_khong_co_du_doan(tmp_path: Path):
    def local_of(p, cam, f):
        return None if cam == "cam02" else p  # detector bỏ sót cả cam02

    scored, _ = _score(local_of, {("cam01", 1): 10, ("cam03", 1): 10, ("cam03", 2): 20}, tmp_path)
    got = _outcomes(scored)
    # cam02 không có dự đoán: lần chuyển cam01→cam02 và cam02→cam03 đều "sót"
    assert got[("cam01", "cam02")] == "sot"
    assert got[("cam02", "cam03")] == "sot"
    summary = summarize(scored)
    assert summary["by_kind"]["overlap"]["accuracy"] is None


def test_id_o_mep_lay_khung_khop_gan_mep_nhat():
    from tools.estimate_transit import Appearance

    app = Appearance(1, "cam01", [(f, BASE + f * 100, BOX[1]) for f in range(10)])
    pred = {("cam01", f, 1): (7 if f < 8 else 9) for f in range(10)}
    assert edge_id(app, pred, edge_frames=2, tail=True) == 9
    assert edge_id(app, pred, edge_frames=3, tail=False) == 7
    assert edge_id(app, {}, edge_frames=3, tail=False) is None


def test_eval_handover_chay_tron(tmp_path: Path, capsys):
    gt = tmp_path / "gt.jsonl"
    write_jsonl(gt, gt_messages())
    _write_table(gt.with_name("gt.gt.json"))
    fixture = tmp_path / "run.jsonl"
    write_jsonl(fixture, result_messages(lambda p, cam, f: p))
    db = tmp_path / "mct.db"
    write_identity_db({("cam01", 1): 10, ("cam02", 1): 10, ("cam03", 1): 10, ("cam03", 2): 20}, db)
    topo = tmp_path / "topo.yaml"
    topo.write_text(
        "cameras:\n  cam01: {overlaps_with: [cam02]}\n  cam02: {overlaps_with: [cam01]}\n"
        "  cam03: {}\n",
        encoding="utf-8",
    )
    out = tmp_path / "h.json"
    args = ["--fixture", str(fixture), "--db", str(db), "--gt-fixture", str(gt)]
    assert main([*args, "--topology", str(topo), "--json", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["total"]["dung"] == 3
    assert len(data["handovers"]) == 3
    assert "chồng lấn" in capsys.readouterr().out


@pytest.mark.parametrize("offset", [0, 5])
def test_frame_offset(tmp_path: Path, offset: int):
    """Kết quả lệch `offset` khung so với GT thì --frame-offset kéo về thẳng hàng."""
    gt = tmp_path / "gt.jsonl"
    write_jsonl(gt, gt_messages())
    _write_table(gt.with_name("gt.gt.json"))
    shifted = result_messages(lambda p, cam, f: p)
    for m in shifted:
        m.frame_id -= offset
    fixture = tmp_path / "run.jsonl"
    write_jsonl(fixture, shifted)
    db = tmp_path / "mct.db"
    write_identity_db({("cam01", 1): 10, ("cam02", 1): 10, ("cam03", 1): 10, ("cam03", 2): 20}, db)
    topo = tmp_path / "topo.yaml"
    topo.write_text("cameras:\n  cam01: {}\n  cam02: {}\n  cam03: {}\n", encoding="utf-8")
    out = tmp_path / "h.json"
    args = ["--fixture", str(fixture), "--db", str(db), "--gt-fixture", str(gt)]
    args += ["--topology", str(topo), "--json", str(out), "--frame-offset", str(offset)]
    assert main(args) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["total"]["dung"] == 3
