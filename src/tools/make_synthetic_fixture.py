"""Sinh fixture metadata giả lập 2 camera, kèm ground-truth Global ID.

Mục đích: cho phép phát triển và test engine liên kết đa camera (M4) TRƯỚC KHI có máy
GPU và dữ liệu thật. Khi đã có fixture thật ghi từ pipeline (M3), fixture tổng hợp vẫn
giữ lại vì nó điều khiển được độ khó — chỉnh đúng một tham số để dựng lại các kịch bản
thực nghiệm ở chương 6 đề cương.

MÔ HÌNH EMBEDDING — ba tầng, tương ứng ba nguồn biến thiên trong Re-ID thật:

  1. vector gốc của danh tính        b_i
  2. sai lệch hệ thống theo camera   b_ic = mix(b_i, hướng ngẫu nhiên riêng của camera)
  3. nhiễu ngẫu nhiên từng frame     emb  = normalize(b_ic + sigma * n)

Tầng 2 là tầng quan trọng nhất và dễ bị bỏ sót. Nhiễu ở tầng 3 độc lập giữa các frame
nên lấy trung bình gallery là triệt tiêu gần hết; nếu chỉ có tầng 3 thì cùng một người
ở hai camera sẽ giống nhau tới ~0.99 và bài toán trở nên vô nghĩa. Trong thực tế, ánh
sáng và góc nhìn khác nhau làm lệch CẢ CỤM embedding của một người ở mỗi camera —
đúng thách thức (1) nêu ở đề cương mục 2.2 — và sai lệch đó không trung bình đi được.

Ba tham số, tất cả đều là cosine similarity đo được trực tiếp:

  --intra-sim      giữa hai frame của cùng người trong CÙNG camera      (mặc định 0.80)
  --cross-cam-sim  giữa gallery của cùng người ở HAI camera khác nhau   (mặc định 0.75)
  --inter-sim      giữa gallery của hai người "mặc đồ giống nhau"       (mặc định 0.65)

Bài toán của engine liên kết nằm gọn ở khoảng cách giữa `cross-cam-sim` (phải khớp) và
`inter-sim` (không được khớp). Thu hẹp hai số này lại là dựng được kịch bản 4 của đề
cương; khi chúng bằng nhau thì ngoại hình mất hoàn toàn khả năng phân biệt và chỉ còn
ràng buộc không–thời gian cứu được.

Mô hình này buộc `inter-sim <= cross-cam-sim`: hai danh tính khác nhau không thể giống
nhau hơn mức chính một danh tính tự giống mình khi đổi camera, vì cùng chịu chung một
hệ số suy giảm theo camera.

Ví dụ:
    python -m tools.make_synthetic_fixture --out tests/fixtures/two_cam_walk.jsonl
    python -m tools.make_synthetic_fixture --cross-cam-sim 0.62 --inter-sim 0.60 --out hard.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.schema import Detection, FrameMessage, l2_normalize, validate, write_jsonl

# Mốc thời gian cố định để fixture tái lập được: 2026-09-01T09:00:00Z.
BASE_TS_MS = 1_788_231_600_000

FRAME_W = 1920
FRAME_H = 1080

CAMERAS = ("cam01", "cam02")


@dataclass(slots=True)
class Appearance:
    """Một lượt xuất hiện của một danh tính trong một camera."""

    cam_id: str
    local_track_id: int
    gt_global_id: int
    start_ms: int
    end_ms: int


def _sigma_for_intra_sim(target_sim: float, dim: int) -> float:
    """Độ lệch chuẩn nhiễu để hai frame của cùng người đạt cosine `target_sim`.

    Với b đơn vị và n1, n2 ~ N(0, I_d) độc lập, đặt e_k = normalize(b + sigma*n_k):

        e1 . e2 = (1 + sigma*(b.n1 + b.n2) + sigma^2*(n1.n2)) / |b+sigma*n1| / |b+sigma*n2|

    Khi d lớn: b.n_k ~= 0, n1.n2 ~= 0, |b + sigma*n_k| ~= sqrt(1 + sigma^2*d), nên

        e1 . e2 ~= 1 / (1 + sigma^2 * d)

    Lưu ý đây là similarity GIỮA HAI MẪU, không phải mẫu-với-vector-gốc (cái sau bằng
    1/sqrt(1 + sigma^2*d), tức căn bậc hai của số này). Nhầm hai đại lượng đó làm độ
    nhiễu thực tế lệch hẳn một bậc.
    """
    if not 0.0 < target_sim < 1.0:
        raise ValueError(f"--intra-sim phải nằm trong (0, 1), nhận {target_sim}")
    return math.sqrt((1.0 / target_sim - 1.0) / dim)


def _random_unit(rng: np.random.Generator, dim: int) -> np.ndarray:
    return l2_normalize(rng.standard_normal(dim))


def _correlated_unit(rng: np.random.Generator, base: np.ndarray, target_cos: float) -> np.ndarray:
    """Vector đơn vị hợp với `base` một góc sao cho cosine đúng bằng `target_cos`."""
    noise = rng.standard_normal(base.shape[0])
    perp = l2_normalize(noise - float(noise @ base) * base)  # bỏ thành phần song song
    return l2_normalize(target_cos * base + math.sqrt(1.0 - target_cos**2) * perp)


def _camera_views(
    rng: np.random.Generator, base: np.ndarray, cross_cam_sim: float
) -> dict[str, np.ndarray]:
    """Biến thể của một danh tính ở từng camera.

    Mỗi camera lấy b_ic = mix(b_i, hướng riêng) với hệ số alpha = sqrt(cross_cam_sim).
    Hai hướng riêng ở hai camera độc lập nên gần trực giao, do đó
        cos(b_ic1, b_ic2) ~= alpha^2 = cross_cam_sim
    đúng bằng giá trị người dùng đặt.
    """
    if not 0.0 < cross_cam_sim <= 1.0:
        raise ValueError(f"--cross-cam-sim phải nằm trong (0, 1], nhận {cross_cam_sim}")
    alpha = math.sqrt(cross_cam_sim)
    return {cam: _correlated_unit(rng, base, alpha) for cam in CAMERAS}


def _bbox_at(progress: float, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """bbox của người đi ngang khung hình; càng đi càng lại gần nên cao dần lên."""
    height = 240.0 + 140.0 * progress + rng.normal(0.0, 6.0)
    width = height * 0.42
    cx = (0.06 + 0.88 * progress) * FRAME_W + rng.normal(0.0, 4.0)
    cy_bottom = 0.62 * FRAME_H + 0.30 * FRAME_H * progress + rng.normal(0.0, 4.0)

    x = float(np.clip(cx - width / 2.0, 0.0, FRAME_W - 1.0))
    y = float(np.clip(cy_bottom - height, 0.0, FRAME_H - 1.0))
    # Clip về trong khung như detector thật vẫn làm.
    return (x, y, float(min(width, FRAME_W - x)), float(min(height, FRAME_H - y)))


def build_scenario(
    *,
    identities: int,
    fps: int,
    embed_dim: int,
    intra_sim: float,
    cross_cam_sim: float,
    inter_sim: float,
    dwell_s: float,
    transit_s: float,
    stagger_s: float,
    miss_rate: float,
    seed: int,
) -> tuple[list[FrameMessage], list[Appearance]]:
    rng = np.random.default_rng(seed)
    sigma = _sigma_for_intra_sim(intra_sim, embed_dim)

    # Vector gốc từng danh tính. Danh tính 1 và 2 được ép giống nhau để tái hiện kịch bản
    # "trang phục tương tự" (đề cương mục 6.2, kịch bản 4).
    #
    # Độ giống đặt lên VECTOR GỐC phải chia cho cross_cam_sim: hai gallery mà ta thực sự
    # đo được là b_1c và b_2c, mỗi cái đã bị hệ số alpha = sqrt(cross_cam_sim) làm suy
    # giảm thành phần theo vector gốc, nên cos(b_1c, b_2c) = cross_cam_sim * cos(b_1, b_2).
    # Chia ngược lại để --inter-sim đúng bằng con số đo được trên gallery.
    if inter_sim > cross_cam_sim:
        raise ValueError(
            f"--inter-sim ({inter_sim}) không được lớn hơn --cross-cam-sim ({cross_cam_sim}): "
            "hai người khác nhau không thể giống nhau hơn mức một người tự giống mình "
            "khi đổi camera, vì cùng chịu chung hệ số suy giảm theo camera."
        )
    base_cos = inter_sim / cross_cam_sim

    bases = [_random_unit(rng, embed_dim)]
    if identities > 1:
        bases.append(_correlated_unit(rng, bases[0], base_cos))
    while len(bases) < identities:
        bases.append(_random_unit(rng, embed_dim))

    views = {gid + 1: _camera_views(rng, bases[gid], cross_cam_sim) for gid in range(identities)}

    dwell_ms = int(dwell_s * 1000)
    transit_ms = int(transit_s * 1000)
    stagger_ms = int(stagger_s * 1000)

    appearances: list[Appearance] = []
    next_local_id = dict.fromkeys(CAMERAS, 1)

    for idx in range(identities):
        gid = idx + 1
        enter_first = idx * stagger_ms
        # Thời gian di chuyển lệch ngẫu nhiên +-25% quanh giá trị danh nghĩa, để ràng buộc
        # thời gian ở engine phải chịu được sai số chứ không khớp cứng.
        gap = int(transit_ms * float(rng.uniform(0.75, 1.25)))
        starts = (enter_first, enter_first + dwell_ms + gap)

        for cam_id, start in zip(CAMERAS, starts, strict=True):
            appearances.append(
                Appearance(
                    cam_id=cam_id,
                    local_track_id=next_local_id[cam_id],
                    gt_global_id=gid,
                    start_ms=start,
                    end_ms=start + dwell_ms,
                )
            )
            next_local_id[cam_id] += 1

    messages = _render_frames(appearances, views, fps, sigma, miss_rate, embed_dim, rng)
    return messages, appearances


def _render_frames(
    appearances: list[Appearance],
    views: dict[int, dict[str, np.ndarray]],
    fps: int,
    sigma: float,
    miss_rate: float,
    embed_dim: int,
    rng: np.random.Generator,
) -> list[FrameMessage]:
    frame_interval_ms = 1000.0 / fps
    messages: list[FrameMessage] = []

    for cam_id in CAMERAS:
        cam_apps = [a for a in appearances if a.cam_id == cam_id]
        if not cam_apps:
            continue
        span_start = min(a.start_ms for a in cam_apps)
        span_end = max(a.end_ms for a in cam_apps)
        n_frames = int((span_end - span_start) / frame_interval_ms) + 1

        for frame_id in range(n_frames):
            offset_ms = span_start + frame_id * frame_interval_ms
            detections: list[Detection] = []

            for app in cam_apps:
                if not app.start_ms <= offset_ms <= app.end_ms:
                    continue
                if rng.random() < miss_rate:  # detector thỉnh thoảng bỏ sót
                    continue

                progress = (offset_ms - app.start_ms) / max(app.end_ms - app.start_ms, 1)
                cam_base = views[app.gt_global_id][cam_id]

                detections.append(
                    Detection(
                        local_track_id=app.local_track_id,
                        bbox=_bbox_at(progress, rng),
                        confidence=float(np.clip(rng.normal(0.88, 0.06), 0.3, 0.999)),
                        embedding=l2_normalize(cam_base + sigma * rng.standard_normal(embed_dim)),
                    )
                )

            messages.append(
                FrameMessage(
                    cam_id=cam_id,
                    frame_id=frame_id,
                    ts_ms=BASE_TS_MS + int(offset_ms),
                    frame_pts_ns=int(frame_id * frame_interval_ms * 1_000_000),
                    frame_width=FRAME_W,
                    frame_height=FRAME_H,
                    detections=detections,
                    embed_dim=embed_dim,
                )
            )

    # Sắp theo thời gian thực để replay giống thứ tự message đến từ nhiều camera.
    messages.sort(key=lambda m: (m.ts_ms, m.cam_id))
    return messages


def write_ground_truth(path: Path, appearances: list[Appearance], meta: dict) -> None:
    """Bảng ánh xạ (cam_id, local_track_id) -> Global ID đúng, để test assert kết quả."""
    payload = {
        "scenario": path.name.removesuffix(".gt.json"),
        "meta": meta,
        "tracklets": [
            {
                "cam_id": a.cam_id,
                "local_track_id": a.local_track_id,
                "gt_global_id": a.gt_global_id,
                "start_ms": BASE_TS_MS + a.start_ms,
                "end_ms": BASE_TS_MS + a.end_ms,
            }
            for a in sorted(appearances, key=lambda a: (a.cam_id, a.local_track_id))
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="tests/fixtures/two_cam_walk.jsonl", type=Path)
    p.add_argument("--identities", type=int, default=3)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--embed-dim", type=int, default=256, help="256 = model re-id kèm DeepStream")
    p.add_argument("--intra-sim", type=float, default=0.80, help="2 frame cùng người, cùng camera")
    p.add_argument("--cross-cam-sim", type=float, default=0.75, help="cùng người, khác camera")
    p.add_argument("--inter-sim", type=float, default=0.65, help="2 người 'mặc đồ giống nhau'")
    p.add_argument("--dwell", type=float, default=6.0, help="số giây ở trong mỗi camera")
    p.add_argument("--transit", type=float, default=8.0, help="số giây di chuyển giữa 2 camera")
    p.add_argument("--stagger", type=float, default=4.0, help="giãn cách giữa các danh tính (giây)")
    p.add_argument("--miss-rate", type=float, default=0.04, help="tỉ lệ detector bỏ sót")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    messages, appearances = build_scenario(
        identities=args.identities,
        fps=args.fps,
        embed_dim=args.embed_dim,
        intra_sim=args.intra_sim,
        cross_cam_sim=args.cross_cam_sim,
        inter_sim=args.inter_sim,
        dwell_s=args.dwell,
        transit_s=args.transit,
        stagger_s=args.stagger,
        miss_rate=args.miss_rate,
        seed=args.seed,
    )

    # Fixture sai contract thì mọi test dựa trên nó đều vô nghĩa — chặn ngay tại đây.
    for msg in messages:
        validate(msg, strict=True)

    out = Path(args.out)
    n = write_jsonl(out, messages)

    meta = {
        "identities": args.identities,
        "fps": args.fps,
        "embed_dim": args.embed_dim,
        "intra_sim": args.intra_sim,
        "cross_cam_sim": args.cross_cam_sim,
        "inter_sim": args.inter_sim,
        "dwell_s": args.dwell,
        "transit_s": args.transit,
        "miss_rate": args.miss_rate,
        "seed": args.seed,
    }
    gt_path = out.with_name(out.stem + ".gt.json")
    write_ground_truth(gt_path, appearances, meta)

    total_det = sum(len(m.detections) for m in messages)
    span_s = (messages[-1].ts_ms - messages[0].ts_ms) / 1000.0 if messages else 0.0
    print(f"{out}: {n} frame, {total_det} detection, {span_s:.1f}s, {args.identities} danh tính")
    print(f"{gt_path}: {len(appearances)} tracklet ground-truth")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
