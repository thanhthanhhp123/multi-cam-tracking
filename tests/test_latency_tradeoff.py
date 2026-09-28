"""Test cho `eval/latency_tradeoff.py` và điểm móc `Engine.window_observer`.

Hai điều phải ghim:
- `time_to_id` đo lúc danh tính được CHỐT (vòng gán đầu tiên của tracklet), không phải các
  lần phát lại sau đó — đúng chỗ phiên 23 đọc nhầm đuôi p90.
- Gắn observer KHÔNG đổi kết quả gán (công cụ đo không được chạm vào thuật toán).
"""

from __future__ import annotations

import math

import numpy as np
from eval.latency_tradeoff import (
    FirstAssignmentLog,
    derive_config,
    percentile,
    summarize,
)

from common.schema import CLASS_PERSON, Detection, FrameMessage, l2_normalize
from mct.__main__ import Engine
from mct.affinity import AffinityConfig
from mct.associator import Associator
from mct.tracklet import TrackletConfig

DIM = 16
BASE_TS = 1_700_000_000_000


def _msg(cam_id: str, frame_id: int, tracks: dict[int, np.ndarray], *, step_ms: int = 100):
    return FrameMessage(
        cam_id=cam_id,
        frame_id=frame_id,
        ts_ms=BASE_TS + frame_id * step_ms,
        frame_pts_ns=frame_id * step_ms * 1_000_000,
        frame_width=1920,
        frame_height=1080,
        detections=[
            Detection(
                local_track_id=tid,
                bbox=(100.0 + frame_id, 200.0, 60.0, 150.0),
                confidence=0.9,
                embedding=emb,
                class_id=CLASS_PERSON,
            )
            for tid, emb in tracks.items()
        ],
        embed_dim=DIM,
    )


def _engine(*, min_frames: int = 2, window_ms: int = 1_000, observer=None) -> Engine:
    return Engine(
        tracklet_config=TrackletConfig(min_frames=min_frames, idle_timeout_ms=2_000),
        associator=Associator(config=AffinityConfig(max_cost=0.5)),
        window_ms=window_ms,
        window_observer=observer,
    )


def _scenario() -> list[FrameMessage]:
    person = l2_normalize(np.ones(DIM, dtype=np.float32))
    other = l2_normalize(np.array([1.0] + [-1.0] * (DIM - 1), dtype=np.float32))
    msgs = [_msg("cam01", f, {1: person, 2: other}) for f in range(15)]
    msgs += [_msg("cam02", f, {7: person}) for f in range(15, 30)]
    return msgs


def _run(engine: Engine, log: FirstAssignmentLog | None = None) -> list[tuple]:
    out = []
    for msg in _scenario():
        out.extend(engine.feed(msg))
    if log is not None:
        log.final = True
    out.extend(engine.finish())
    return [(u.cam_id, u.local_track_id, u.global_id, u.is_new, u.is_update) for u in out]


def test_observer_khong_doi_ket_qua_gan():
    log = FirstAssignmentLog()
    assert _run(_engine()) == _run(_engine(observer=log), log)
    assert log.first, "observer phải được gọi"


def test_chi_ghi_lan_gan_dau_tien_cua_moi_tracklet():
    log = FirstAssignmentLog()
    _run(_engine(observer=log), log)
    # 3 tracklet (cam01 x2, cam02 x1), mỗi cái đúng một bản ghi dù được phát nhiều vòng.
    assert len(log.first) == 3
    # Theo ts_ms, không theo đồng hồ: hai tracklet cam01 bắt đầu ở 0 ms, vòng đầu chạy khi
    # message 1000 ms vượt mốc cửa sổ -> 1000 ms. Cửa sổ kế neo ở 2000 ms, nên tracklet cam02
    # bắt đầu ở 1500 ms được gán sau 500 ms.
    assert sorted(f.time_to_id_ms for f in log.first.values()) == [500, 1_000, 1_000]
    assert not any(f.final for f in log.first.values())


def test_cua_so_ngan_hon_thi_chot_som_hon():
    slow, fast = FirstAssignmentLog(), FirstAssignmentLog()
    _run(_engine(window_ms=1_000, observer=slow), slow)
    _run(_engine(window_ms=200, observer=fast), fast)
    assert summarize(fast)["p50_ms"] < summarize(slow)["p50_ms"]


def test_tracklet_chi_duoc_gan_o_vong_cuoi_bi_tach_rieng():
    log = FirstAssignmentLog()
    engine = _engine(min_frames=2, window_ms=10_000, observer=log)
    person = l2_normalize(np.ones(DIM, dtype=np.float32))
    for f in range(3):  # 300 ms: cửa sổ 10 s chưa bao giờ đóng
        engine.feed(_msg("cam01", f, {1: person}))
    log.final = True
    engine.finish()
    stats = summarize(log)
    assert stats["n_final"] == 1
    assert math.isnan(stats["p50_ms"]), "hiện vật của vòng cuối không được lẫn vào phân vị"


def test_percentile_theo_hang_gan_nhat():
    values = [100.0, 200.0, 300.0, 400.0, 500.0]
    assert percentile(values, 0.5) == 300.0
    assert percentile(values, 0.0) == 100.0
    assert percentile(values, 1.0) == 500.0
    assert math.isnan(percentile([], 0.5))


def test_derive_config_khong_sua_ban_goc():
    base = {"association": {"window_ms": 1000, "max_cost": 0.9}, "tracklet": {"min_frames": 3}}
    derived = derive_config(base, 500, 1)
    assert derived["association"] == {"window_ms": 500, "max_cost": 0.9}
    assert derived["tracklet"]["min_frames"] == 1
    assert base["association"]["window_ms"] == 1000
    assert base["tracklet"]["min_frames"] == 3
