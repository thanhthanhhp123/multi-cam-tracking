"""Test bộ sinh fixture tổng hợp.

Fixture này là bàn thử nghiệm cho toàn bộ engine liên kết (M4). Nếu các tham số độ khó
không đúng bằng con số người dùng đặt thì mọi kết quả thực nghiệm báo cáo trên nó đều
sai độ khó — nên tính chất thống kê của nó phải được test như code thường.

Hai bẫy đã từng mắc, giữ test lại để không tái phạm:
  1. Nhầm "similarity giữa hai mẫu" với "similarity mẫu-với-vector-gốc" (lệch một luỹ
     thừa bậc hai: đặt 0.80 nhưng thực tế ra 0.64).
  2. Chỉ có nhiễu i.i.d. theo frame, thiếu sai lệch hệ thống theo camera. Nhiễu i.i.d.
     bị triệt tiêu khi lấy trung bình gallery, khiến cùng một người ở hai camera giống
     nhau tới ~0.99 và bài toán MTMCT trở nên vô nghĩa.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
import pytest

from common.schema import validate
from tools.make_synthetic_fixture import build_scenario

SEED = 42
TOL = 0.05

PARAMS = dict(
    identities=3,
    fps=15,
    embed_dim=256,
    intra_sim=0.80,
    cross_cam_sim=0.75,
    inter_sim=0.65,
    dwell_s=6.0,
    transit_s=8.0,
    stagger_s=4.0,
    miss_rate=0.04,
    seed=SEED,
)


@pytest.fixture(scope="module")
def scenario():
    return build_scenario(**PARAMS)  # type: ignore[arg-type]


def _embeddings_by_track(messages) -> dict[tuple[str, int], list[np.ndarray]]:
    out: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for msg in messages:
        for det in msg.detections:
            out[(msg.cam_id, det.local_track_id)].append(det.embedding)
    return dict(out)


def _centroid(embs: list[np.ndarray]) -> np.ndarray:
    vec = np.stack(embs).mean(axis=0)
    return vec / np.linalg.norm(vec)


def _mean_pairwise_cos(embs: list[np.ndarray]) -> float:
    mat = np.stack(embs)
    sims = mat @ mat.T
    iu = np.triu_indices(len(mat), k=1)
    return float(sims[iu].mean())


# --------------------------------------------------------------------------- #
# Ba tham số độ khó phải đúng bằng con số đặt
# --------------------------------------------------------------------------- #


def test_intra_sim_khop_tham_so(scenario) -> None:
    """Hai frame của cùng một người trong cùng camera."""
    messages, _ = scenario
    measured = [_mean_pairwise_cos(e) for e in _embeddings_by_track(messages).values()]
    assert float(np.mean(measured)) == pytest.approx(PARAMS["intra_sim"], abs=TOL)


def test_cross_cam_sim_khop_tham_so(scenario) -> None:
    """Cùng một người ở hai camera khác nhau — phải bị sai lệch hệ thống làm giảm xuống.

    Đây là test quan trọng nhất file này: nếu bỏ tầng sai lệch theo camera thì giá trị
    đo được sẽ vọt lên ~0.99 và test đổ.
    """
    messages, appearances = scenario
    gid_of = {(a.cam_id, a.local_track_id): a.gt_global_id for a in appearances}
    cent = {k: _centroid(v) for k, v in _embeddings_by_track(messages).items()}

    measured = [
        float(cent[a] @ cent[b])
        for a, b in itertools.product(cent, repeat=2)
        if a[0] == "cam01" and b[0] == "cam02" and gid_of[a] == gid_of[b]
    ]
    assert len(measured) == PARAMS["identities"]
    assert float(np.mean(measured)) == pytest.approx(PARAMS["cross_cam_sim"], abs=TOL)


def test_inter_sim_khop_tham_so(scenario) -> None:
    """Cặp danh tính 1-2 được cố ý làm giống nhau ("trang phục tương tự")."""
    messages, appearances = scenario
    gid_of = {(a.cam_id, a.local_track_id): a.gt_global_id for a in appearances}
    cent = {k: _centroid(v) for k, v in _embeddings_by_track(messages).items()}

    measured = [
        float(cent[a] @ cent[b])
        for a, b in itertools.combinations(cent, 2)
        if a[0] == b[0] and {gid_of[a], gid_of[b]} == {1, 2}
    ]
    assert measured
    assert float(np.mean(measured)) == pytest.approx(PARAMS["inter_sim"], abs=TOL)


def test_danh_tinh_khac_cap_giong_nhau_thi_gan_truc_giao(scenario) -> None:
    """Danh tính 3 là ngẫu nhiên, không được vô tình giống ai."""
    messages, appearances = scenario
    gid_of = {(a.cam_id, a.local_track_id): a.gt_global_id for a in appearances}
    cent = {k: _centroid(v) for k, v in _embeddings_by_track(messages).items()}

    measured = [
        abs(float(cent[a] @ cent[b]))
        for a, b in itertools.combinations(cent, 2)
        if 3 in {gid_of[a], gid_of[b]} and gid_of[a] != gid_of[b]
    ]
    assert max(measured) < 0.35


def test_bai_toan_van_giai_duoc_bang_ngoai_hinh(scenario) -> None:
    """Với tham số mặc định phải còn biên dương giữa positive và negative xuyên camera.

    Biên hẹp là cố ý — nhưng âm thì fixture vô nghiệm và mọi test M4 dựa trên nó sẽ
    thất bại vì lý do sai.
    """
    messages, appearances = scenario
    gid_of = {(a.cam_id, a.local_track_id): a.gt_global_id for a in appearances}
    cent = {k: _centroid(v) for k, v in _embeddings_by_track(messages).items()}

    pairs = [
        (float(cent[a] @ cent[b]), gid_of[a] == gid_of[b])
        for a, b in itertools.product(cent, repeat=2)
        if a[0] == "cam01" and b[0] == "cam02"
    ]
    positives = [s for s, same in pairs if same]
    negatives = [s for s, same in pairs if not same]
    assert min(positives) > max(negatives), (
        f"positive min {min(positives):.3f} <= negative max {max(negatives):.3f} — "
        "fixture không phân tách được bằng ngoại hình"
    )


def test_inter_sim_lon_hon_cross_cam_sim_thi_bao_loi() -> None:
    with pytest.raises(ValueError, match="không được lớn hơn"):
        build_scenario(**{**PARAMS, "inter_sim": 0.90, "cross_cam_sim": 0.75})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Tính toàn vẹn
# --------------------------------------------------------------------------- #


def test_moi_message_dung_contract(scenario) -> None:
    messages, _ = scenario
    for msg in messages:
        assert validate(msg) == []


def test_ground_truth_phu_het_tracklet(scenario) -> None:
    messages, appearances = scenario
    seen = set(_embeddings_by_track(messages))
    declared = {(a.cam_id, a.local_track_id) for a in appearances}
    assert seen == declared


def test_moi_danh_tinh_xuat_hien_o_ca_hai_camera(scenario) -> None:
    _, appearances = scenario
    by_gid: dict[int, set[str]] = defaultdict(set)
    for a in appearances:
        by_gid[a.gt_global_id].add(a.cam_id)
    assert all(cams == {"cam01", "cam02"} for cams in by_gid.values())


def test_message_sap_theo_thoi_gian(scenario) -> None:
    """Replay dựa vào thứ tự này để mô phỏng đúng cách message đến từ nhiều camera."""
    messages, _ = scenario
    assert [m.ts_ms for m in messages] == sorted(m.ts_ms for m in messages)


def test_thoi_gian_di_chuyen_nam_trong_khoang_topology(scenario) -> None:
    """Khoảng trống giữa hai camera phải khớp cửa sổ trong configs/cameras/topology.yaml."""
    _, appearances = scenario
    by_gid: dict[int, dict[str, tuple[int, int]]] = defaultdict(dict)
    for a in appearances:
        by_gid[a.gt_global_id][a.cam_id] = (a.start_ms, a.end_ms)

    for gid, cams in by_gid.items():
        gap_ms = cams["cam02"][0] - cams["cam01"][1]
        assert 3000 <= gap_ms <= 15000, f"gid {gid}: gap {gap_ms}ms ngoài khoảng topology"


def test_tai_lap_duoc_voi_cung_seed() -> None:
    a, _ = build_scenario(**PARAMS)  # type: ignore[arg-type]
    b, _ = build_scenario(**PARAMS)  # type: ignore[arg-type]
    assert len(a) == len(b)
    np.testing.assert_array_equal(a[0].detections[0].embedding, b[0].detections[0].embedding)
