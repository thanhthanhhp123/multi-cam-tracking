"""Test `tools/check_lab_setup.py` và các bản mẫu trong `configs/lab/`.

Công cụ kiểm trước khi thuê GPU chỉ có giá trị nếu (1) bộ cấu hình đúng thì im lặng (0 FAIL),
và (2) từng lỗi im lặng đã gặp trong đồ án làm nó báo FAIL. Thêm: bản mẫu đi kèm repo phải
báo FAIL ĐÚNG ở chỗ còn số giữ chỗ — không ít hơn (lọt qua), không nhiều hơn (mẫu tự hỏng).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from common.schema import Detection, FrameMessage, write_jsonl
from mct.topology import Topology
from tools.check_lab_setup import Report, check_ground_truth, check_session, main

TOPOLOGY = {
    "status": "measured",
    "cameras": {
        "cam01": {"resolution": [1920, 1080], "fps": 25, "overlaps_with": ["cam02"]},
        "cam02": {"resolution": [1920, 1080], "fps": 25, "overlaps_with": ["cam01"]},
        "cam03": {"resolution": [1920, 1080], "fps": 25, "overlaps_with": []},
    },
    "transitions": [
        {"from": "cam02", "to": "cam03", "bidirectional": True, "min_ms": 2000, "max_ms": 20000}
    ],
}


def _lab(tmp_path: Path, *, topology: dict | None = None, homography: list[str] | None = None):
    lab = tmp_path / "lab"
    (lab / "homography").mkdir(parents=True)
    (lab / "topology.yaml").write_text(yaml.safe_dump(topology or TOPOLOGY), encoding="utf-8")
    streams = {
        "sources": [
            {"cam_id": c, "uri": f"file:///x/data/lab/${{LAB_SESSION}}/{c}.mp4"}
            for c in ("cam01", "cam02", "cam03")
        ],
        "streammux": {"width": 1920, "height": 1080},
        "sink": {"sync": True},
    }
    (lab / "streams_lab.yaml").write_text(yaml.safe_dump(streams), encoding="utf-8")
    engine = {
        "tracklet": {"ground_path_max_points": 750},
        "association": {"max_cost": 0.3, "max_cost_geometric": 0.9, "ground_time_tol_ms": 200},
    }
    (lab / "lab.mct.yaml").write_text(yaml.safe_dump(engine), encoding="utf-8")
    for cam in homography if homography is not None else ["cam01", "cam02", "cam03"]:
        data = {
            "cam_id": cam,
            "plane": "ground",
            "unit": "m",
            "matrix": [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]],
            "image_size": [1920, 1080],
        }
        (lab / "homography" / f"{cam}.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return lab


def _run(lab: Path, capsys, *extra: str) -> tuple[int, str]:
    code = main(["--lab-dir", str(lab), "--session", "s1", "--data-dir", str(lab.parent), *extra])
    return code, capsys.readouterr().out


def test_cau_hinh_dung_thi_khong_co_fail(tmp_path: Path, capsys):
    code, out = _run(_lab(tmp_path), capsys)
    assert code == 0, out
    assert "0 FAIL" in out


def test_cap_chong_lan_thieu_homography_thi_fail(tmp_path: Path, capsys):
    code, out = _run(_lab(tmp_path, homography=["cam01", "cam03"]), capsys)
    assert code == 1
    assert "cam02 chồng lấn" in out


def test_camera_khong_chong_lan_thieu_homography_chi_warn(tmp_path: Path, capsys):
    code, out = _run(_lab(tmp_path, homography=["cam01", "cam02"]), capsys)
    assert code == 0
    assert "cam03 chưa hiệu chỉnh" in out


def test_homography_khac_do_phan_giai_thi_fail(tmp_path: Path, capsys):
    topo = json.loads(json.dumps(TOPOLOGY))
    topo["cameras"]["cam03"]["resolution"] = [1080, 1920]  # điện thoại quay dọc
    code, out = _run(_lab(tmp_path, topology=topo), capsys)
    assert code == 1
    assert "hiệu chỉnh ở" in out


def test_topology_con_la_ban_mau_thi_fail(tmp_path: Path, capsys):
    topo = {**TOPOLOGY, "status": "template"}
    code, out = _run(_lab(tmp_path, topology=topo), capsys)
    assert code == 1
    assert "template" in out


def test_khong_sync_voi_nguon_file_thi_fail(tmp_path: Path, capsys):
    lab = _lab(tmp_path)
    streams = yaml.safe_load((lab / "streams_lab.yaml").read_text(encoding="utf-8"))
    streams["sink"]["sync"] = False
    (lab / "streams_lab.yaml").write_text(yaml.safe_dump(streams), encoding="utf-8")
    code, out = _run(lab, capsys)
    assert code == 1
    assert "sync" in out


def test_thieu_session_thi_bien_khong_the_duoc(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.delenv("LAB_SESSION", raising=False)
    code = main(["--lab-dir", str(_lab(tmp_path))])
    assert code == 1
    assert "LAB_SESSION" in capsys.readouterr().out


def _sync(session: Path, *, fps: float = 25.0, frames: int = 100) -> None:
    session.mkdir(parents=True, exist_ok=True)
    data = {
        "fps": fps,
        "duration_s": frames / fps,
        "expected_frames": frames,
        "clap_frame": 10,
        "cameras": {c: {} for c in ("cam01", "cam02", "cam03")},
        "estimates": {"cam02": {"source": "audio", "confidence": 5.0}},
    }
    (session / "sync.json").write_text(json.dumps(data), encoding="utf-8")


def test_fps_buoi_quay_khac_topology_thi_fail(tmp_path: Path):
    _sync(tmp_path / "s1", fps=30.0)
    rep = Report()
    check_session(tmp_path / "s1", Topology.from_mapping(TOPOLOGY), rep)
    assert rep.n_fail == 1


def _gt(tmp_path: Path, *, last_frame: int, cams: dict[str, list[int]]) -> Path:
    """cams: {cam_id: [person...]} — mỗi người một track ở camera đó."""
    path = tmp_path / "gt.jsonl"
    msgs = []
    rows = []
    for cam, persons in cams.items():
        for frame in range(0, last_frame + 1, 5):
            dets = [
                Detection(local_track_id=p, bbox=(10.0 * p, 10.0, 50.0, 100.0), confidence=1.0)
                for p in persons
            ]
            msgs.append(
                FrameMessage(cam, frame, 1_000_000 + frame * 40, 0, 1920, 1080, detections=dets)
            )
        rows += [{"cam_id": cam, "local_track_id": p, "gt_global_id": p} for p in persons]
    write_jsonl(path, msgs)
    (tmp_path / "gt.gt.json").write_text(json.dumps({"tracklets": rows}), encoding="utf-8")
    return path


def test_ground_truth_vuot_so_khung_video_thi_fail(tmp_path: Path):
    _sync(tmp_path / "s1", frames=100)
    topo = Topology.from_mapping(TOPOLOGY)
    rep = Report()
    sync = check_session(tmp_path / "s1", topo, rep)
    gt = _gt(tmp_path, last_frame=150, cams={"cam01": [1], "cam02": [1], "cam03": [1]})
    check_ground_truth(gt, topo, sync, rep)
    assert any(lv == "FAIL" and "vượt số khung" in msg for lv, _, msg in rep.lines)


def test_ground_truth_thieu_loai_cap_thi_warn(tmp_path: Path):
    topo = Topology.from_mapping(TOPOLOGY)
    rep = Report()
    # người 1 chỉ đi cam01–cam02 (chồng lấn), không ai sang cam03
    check_ground_truth(
        _gt(tmp_path, last_frame=50, cams={"cam01": [1], "cam02": [1]}), topo, None, rep
    )
    assert rep.n_fail == 0
    assert any("không chồng lấn" in msg for lv, _, msg in rep.lines if lv == "WARN")


# --------------------------------------------------------------------------------------
# Bản mẫu đi kèm repo
# --------------------------------------------------------------------------------------


def test_ban_mau_cua_repo_chi_fail_o_cho_giu_cho(repo_root: Path, capsys, tmp_path: Path):
    code = main(
        [
            "--lab-dir",
            str(repo_root / "configs" / "lab"),
            "--session",
            "s1",
            "--data-dir",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert code == 1
    fails = [line for line in out.splitlines() if line.startswith("[FAIL]")]
    assert len(fails) == 2, out
    assert any("topology" in f and "template" in f for f in fails)
    assert any("homography" in f and "template" in f for f in fails)


@pytest.mark.parametrize("name", ["lab.mct.yaml"])
def test_cau_hinh_engine_lab_dung_duoc_voi_engine(repo_root: Path, name: str):
    from mct.__main__ import build_engine, load_config

    config = load_config(repo_root / "configs" / "lab" / name)
    engine = build_engine(
        config,
        topology_path=repo_root / "configs" / "lab" / "topology.yaml",
        db_path=":memory:",
    )
    assert engine is not None
    assert config["association"]["max_cost_geometric"] > config["association"]["max_cost"]


def test_hieu_chinh_tu_choi_file_diem_ban_mau(repo_root: Path, tmp_path: Path):
    """Toạ độ giữ chỗ vẫn khớp ra một ma trận hợp lệ về hình thức — phải chặn từ cửa."""
    from tools.calibrate_homography import load_point_file

    template = repo_root / "configs" / "lab" / "ground_points.yaml"
    with pytest.raises(ValueError, match="template"):
        load_point_file(template)
    # Bỏ dòng status thì cấu trúc của bản mẫu đọc được: đủ 4 camera, mỗi camera >= 4 điểm.
    data = yaml.safe_load(template.read_text(encoding="utf-8"))
    data.pop("status")
    measured = tmp_path / "points.yaml"
    measured.write_text(yaml.safe_dump(data), encoding="utf-8")
    points, meta = load_point_file(measured)
    assert sorted(points) == ["cam01", "cam02", "cam03", "cam04"]
    assert all(len(img) >= 4 for img, _ in points.values())
    assert meta["image_size"] == (1920, 1080)
