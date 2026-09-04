"""Test gán Global ID (src/mct/associator.py) — đóng góp kỹ thuật chính của đồ án.

Ba tầng test:
  1. Hành vi từng bước trên tracklet dựng tay (khớp, tạo mới, cập nhật, không bị cướp ID).
  2. Chạy trọn kịch bản fixture tổng hợp có ground truth: mỗi danh tính phải nhận đúng
     MỘT Global ID, và hai danh tính khác nhau không được dùng chung ID.
  3. Trường hợp suy biến: pipeline chạy ReID tắt (đúng tình trạng M1/M2).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from common.schema import CLASS_PERSON, Detection, FrameMessage, l2_normalize
from mct.affinity import AffinityConfig
from mct.associator import Associator, assign_messages, run_offline
from mct.gallery import Gallery
from mct.topology import Topology
from mct.tracklet import Tracklet, TrackletBuilder, TrackletConfig, build_tracklets
from tools.make_synthetic_fixture import build_scenario

SEED = 11
DIM = 64

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
    seed=42,
)

TOPO = Topology.from_mapping(
    {
        "cameras": {"cam01": {}, "cam02": {}},
        "transitions": [
            {"from": "cam01", "to": "cam02", "bidirectional": True, "min_ms": 3000, "max_ms": 15000}
        ],
    }
)


def _tracklet(
    cam_id: str,
    local_track_id: int,
    *,
    start_ms: int,
    embedding: np.ndarray | None = None,
    n_frames: int = 6,
    tracklet_id: int | None = None,
) -> Tracklet:
    builder = TrackletBuilder(TrackletConfig(min_frames=1))
    for i in range(n_frames):
        builder.update(
            FrameMessage(
                cam_id=cam_id,
                frame_id=i,
                ts_ms=start_ms + i * 66,
                frame_pts_ns=i * 66_666_667,
                frame_width=1920,
                frame_height=1080,
                detections=[
                    Detection(
                        local_track_id=local_track_id,
                        bbox=(100.0, 200.0, 80.0, 200.0),
                        confidence=0.9,
                        embedding=embedding,
                        class_id=CLASS_PERSON,
                    )
                ],
                embed_dim=0 if embedding is None else embedding.shape[0],
            )
        )
    tracklet = builder.flush()[0]
    if tracklet_id is not None:
        tracklet.tracklet_id = tracklet_id
    return tracklet


def _similar(rng: np.random.Generator, base: np.ndarray, cos: float) -> np.ndarray:
    """Vector hợp với `base` đúng cosine yêu cầu — mô phỏng cùng người ở camera khác."""
    noise = rng.standard_normal(base.shape[0])
    perp = l2_normalize(noise - float(noise @ base) * base)
    return l2_normalize(cos * base + float(np.sqrt(1.0 - cos**2)) * perp)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
# 1. Hành vi từng bước
# --------------------------------------------------------------------------- #


def test_tracklet_dau_tien_nhan_global_id_moi(rng):
    associator = Associator(topology=TOPO)
    vec = l2_normalize(rng.standard_normal(DIM))

    [result] = associator.assign([_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1)])

    assert result.is_new
    assert result.global_id == 1
    assert "gallery đang rỗng" in result.reason
    assert associator.stats.created == 1


def test_cung_nguoi_o_camera_khac_nhan_lai_global_id(rng):
    associator = Associator(topology=TOPO)
    base = l2_normalize(rng.standard_normal(DIM))

    first = associator.assign([_tracklet("cam01", 1, start_ms=0, embedding=base, tracklet_id=1)])
    second = associator.assign(
        [_tracklet("cam02", 5, start_ms=8_000, embedding=_similar(rng, base, 0.85), tracklet_id=2)]
    )

    assert second[0].global_id == first[0].global_id
    assert not second[0].is_new
    assert second[0].cost < 0.30
    assert associator.stats.matched == 1
    assert len(associator.gallery) == 1


def test_nguoi_khac_mac_do_giong_nhau_van_nhan_id_rieng(rng):
    """Kịch bản 4 của đề cương: ngoại hình giống ở mức inter_sim=0.65 → cost 0.35 > 0.30."""
    associator = Associator(topology=TOPO, config=AffinityConfig(max_cost=0.30))
    base = l2_normalize(rng.standard_normal(DIM))

    first = associator.assign([_tracklet("cam01", 1, start_ms=0, embedding=base, tracklet_id=1)])
    second = associator.assign(
        [_tracklet("cam02", 5, start_ms=8_000, embedding=_similar(rng, base, 0.65), tracklet_id=2)]
    )

    assert second[0].is_new
    assert second[0].global_id != first[0].global_id
    assert "max_cost" in second[0].reason
    assert associator.stats.rejected_by_threshold == 1


def test_rang_buoc_thoi_gian_chan_match_du_ngoai_hinh_trung_khop(rng):
    associator = Associator(topology=TOPO)
    base = l2_normalize(rng.standard_normal(DIM))

    first = associator.assign([_tracklet("cam01", 1, start_ms=0, embedding=base, tracklet_id=1)])
    too_soon = associator.assign(
        [_tracklet("cam02", 5, start_ms=900, embedding=base, tracklet_id=2)]
    )

    assert too_soon[0].is_new
    assert too_soon[0].global_id != first[0].global_id
    assert "quá nhanh" in too_soon[0].reason
    assert associator.stats.no_candidate == 1


def test_tracklet_dang_chay_duoc_cap_nhat_chu_khong_gan_lai(rng):
    associator = Associator(topology=TOPO)
    vec = l2_normalize(rng.standard_normal(DIM))

    first = associator.assign([_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=7)])
    again = associator.assign(
        [_tracklet("cam01", 1, start_ms=0, embedding=vec, n_frames=20, tracklet_id=7)]
    )

    assert again[0].global_id == first[0].global_id
    assert again[0].is_update and not again[0].is_new
    assert associator.stats.updated == 1
    assert associator.stats.created == 1, "không được sinh thêm Global ID nào"


def test_hungarian_khong_cuop_id_cua_tracklet_dang_chay(rng):
    """Tracklet đang chạy không được đem ra cạnh tranh lại chỉ vì có kẻ giống hơn.

    Hungarian tối ưu TỔNG chi phí nên sẵn sàng hy sinh một cặp để đổi lấy tổng rẻ hơn —
    lý do associator tách phần cập nhật ra trước khi dựng ma trận.
    """
    associator = Associator(topology=TOPO)
    base = l2_normalize(rng.standard_normal(DIM))

    running = _tracklet("cam01", 1, start_ms=0, embedding=base, tracklet_id=1)
    [first] = associator.assign([running])

    newcomer = _tracklet("cam02", 2, start_ms=8_000, embedding=base, tracklet_id=2)
    longer = _tracklet("cam01", 1, start_ms=0, embedding=base, n_frames=30, tracklet_id=1)
    results = {r.tracklet.tracklet_id: r for r in associator.assign([longer, newcomer])}

    assert results[1].global_id == first.global_id
    assert results[1].is_update
    assert results[2].global_id == first.global_id, "người mới vẫn khớp đúng vào track đó"


def test_hai_tracklet_cung_camera_khong_gan_chung_mot_id(rng):
    """Ràng buộc loại trừ: hai người trong cùng khung hình không thể là một."""
    associator = Associator(topology=TOPO)
    base = l2_normalize(rng.standard_normal(DIM))

    associator.assign([_tracklet("cam01", 1, start_ms=0, embedding=base, tracklet_id=1)])
    [second] = associator.assign(
        [_tracklet("cam01", 2, start_ms=100, embedding=base, tracklet_id=2)]
    )

    assert second.is_new
    assert "loại trừ" in second.reason
    assert len(associator.gallery) == 2


def test_hai_tracklet_trong_cung_vong_duoc_ghep_toi_uu(rng):
    """Hungarian phải ghép theo tổng tốt nhất, không phải tham lam theo từng hàng."""
    associator = Associator(topology=TOPO)
    a = l2_normalize(rng.standard_normal(DIM))
    b = l2_normalize(rng.standard_normal(DIM))

    first = associator.assign([_tracklet("cam01", 1, start_ms=0, embedding=a, tracklet_id=1)])
    second = associator.assign([_tracklet("cam01", 2, start_ms=0, embedding=b, tracklet_id=2)])
    gid_a, gid_b = first[0].global_id, second[0].global_id

    pair = associator.assign(
        [
            _tracklet("cam02", 11, start_ms=8_000, embedding=_similar(rng, a, 0.9), tracklet_id=3),
            _tracklet("cam02", 12, start_ms=8_000, embedding=_similar(rng, b, 0.9), tracklet_id=4),
        ]
    )
    by_tracklet = {r.tracklet.tracklet_id: r.global_id for r in pair}

    assert by_tracklet[3] == gid_a
    assert by_tracklet[4] == gid_b


def test_prune_dong_track_qua_han_va_tra_ve_de_ghi_sqlite(rng):
    associator = Associator(topology=TOPO)
    vec = l2_normalize(rng.standard_normal(DIM))
    [result] = associator.assign([_tracklet("cam01", 1, start_ms=0, embedding=vec, tracklet_id=1)])

    closed = associator.prune(now_ms=10_000_000)

    assert [t.global_id for t in closed] == [result.global_id]
    assert len(associator.gallery) == 0


# --------------------------------------------------------------------------- #
# 2. Trên fixture tổng hợp có ground truth
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def scenario():
    return build_scenario(**PARAMS)  # type: ignore[arg-type]


def _gt_map(appearances) -> dict[tuple[str, int], int]:
    return {(a.cam_id, a.local_track_id): a.gt_global_id for a in appearances}


def test_moi_danh_tinh_nhan_dung_mot_global_id(scenario):
    messages, appearances = scenario
    results, associator = assign_messages(
        messages,
        topology=TOPO,
        tracklet_config=TrackletConfig(min_frames=5, idle_timeout_ms=2_000),
    )
    gt = _gt_map(appearances)

    gid_per_identity: dict[int, set[int]] = defaultdict(set)
    identity_per_gid: dict[int, set[int]] = defaultdict(set)
    for r in results:
        identity = gt[r.tracklet.key]
        gid_per_identity[identity].add(r.global_id)
        identity_per_gid[r.global_id].add(identity)

    assert len(gid_per_identity) == PARAMS["identities"]
    for identity, gids in gid_per_identity.items():
        assert len(gids) == 1, f"danh tính {identity} bị vỡ thành {len(gids)} Global ID"
    for gid, identities in identity_per_gid.items():
        assert len(identities) == 1, f"Global ID {gid} gộp nhầm {len(identities)} người"

    assert associator.stats.created == PARAMS["identities"]
    assert associator.stats.matched == PARAMS["identities"], (
        "mỗi người phải khớp đúng một lần khi chuyển sang camera thứ hai"
    )


def test_ket_qua_khong_phu_thuoc_thu_tu_message(scenario):
    """Message tới lệch thứ tự (Redis, nhiều luồng) không được đổi kết quả gán."""
    messages, _ = scenario
    config = TrackletConfig(min_frames=5, idle_timeout_ms=2_000)

    straight, _ = assign_messages(messages, topology=TOPO, tracklet_config=config)
    shuffled_messages = sorted(messages, key=lambda m: (m.ts_ms, m.cam_id), reverse=False)
    shuffled, _ = assign_messages(shuffled_messages, topology=TOPO, tracklet_config=config)

    def signature(results):
        return sorted((r.tracklet.key, r.global_id) for r in results)

    assert signature(straight) == signature(shuffled)


def test_bo_rang_buoc_thoi_gian_van_dung_tren_kich_ban_de(scenario):
    """Không có topology thì chỉ còn ngoại hình — vẫn phải đúng trên kịch bản này.

    Chạy để định lượng phần đóng góp riêng của Re-ID (chương 6): nếu bỏ ràng buộc thời
    gian mà kết quả tệ đi, chênh lệch đó chính là giá trị của topology.
    """
    messages, appearances = scenario
    results, _ = assign_messages(
        messages, topology=None, tracklet_config=TrackletConfig(min_frames=5)
    )
    gt = _gt_map(appearances)

    gid_per_identity: dict[int, set[int]] = defaultdict(set)
    for r in results:
        gid_per_identity[gt[r.tracklet.key]].add(r.global_id)

    assert all(len(g) == 1 for g in gid_per_identity.values())


def test_nguong_qua_gat_lam_vo_danh_tinh(scenario):
    """max_cost quá nhỏ thì mọi lượt xuất hiện thành một Global ID riêng.

    Ghim lại hành vi ở hai đầu để phần sweep tham số ở M6 có mốc so sánh: cùng dữ liệu,
    chỉ đổi một tham số, kết quả đổi hẳn.
    """
    messages, _ = scenario
    results, _ = assign_messages(
        messages,
        topology=TOPO,
        tracklet_config=TrackletConfig(min_frames=5),
        config=AffinityConfig(max_cost=0.05),
    )

    assert len({r.global_id for r in results}) == len(results)


def test_run_offline_nhan_thang_tracklet_da_dong(scenario):
    messages, _ = scenario
    tracklets = build_tracklets(messages, TrackletConfig(min_frames=5, idle_timeout_ms=2_000))

    results, associator = run_offline(tracklets, topology=TOPO)

    assert len(results) == len(tracklets)
    assert len(associator.gallery) == PARAMS["identities"]


# --------------------------------------------------------------------------- #
# 3. Suy biến: ReID tắt
# --------------------------------------------------------------------------- #


def test_reid_tat_thi_moi_tracklet_mot_id_va_khong_no():
    """Pipeline M1/M2 (embed_dim=0): suy biến thành theo dõi trong từng camera."""
    associator = Associator(topology=TOPO, gallery=Gallery())

    first = associator.assign([_tracklet("cam01", 1, start_ms=0, tracklet_id=1)])
    second = associator.assign([_tracklet("cam02", 1, start_ms=8_000, tracklet_id=2)])
    again = associator.assign([_tracklet("cam01", 1, start_ms=0, n_frames=20, tracklet_id=1)])

    assert first[0].is_new and second[0].is_new
    assert first[0].global_id != second[0].global_id
    assert "chưa có embedding" in second[0].reason
    assert again[0].is_update, "tracklet cũ vẫn phải được nhận ra, không sinh ID mới mỗi cửa sổ"
    assert associator.stats.created == 2
