"""Sinh một "buổi quay" lab GIẢ LẬP — đủ mọi tệp mà buổi quay thật sinh ra — để diễn tập M6.

    PYTHONPATH=src python -m tools.make_synthetic_lab --out data/lab_synth --session syn1 --runs 3

**Vì sao cần.** Chuỗi công cụ của M6 (`cvat_to_mot --fixture-out` → `assign_gt` → engine →
`export_trackeval --only-gt-frames` → TrackEval → `eval_handover`, nối bởi `eval.run_lab_eval`)
có test cho từng mắt xích, nhưng chưa từng chạy NỐI TIẾP trên một bộ dữ liệu có đủ mọi đặc
điểm của dữ liệu thật: chú thích nhảy khung, hai hệ đánh số khung, đồng hồ lần chạy khác đồng
hồ chú thích, id tracker khác id chú thích, cặp chồng lấn lẫn không chồng lấn. Lỗi ghép nối
lộ ra lúc có dữ liệu thật là lỗi đắt nhất (đã có hàng GPU-giờ chạy rồi). Bộ sinh này cho chạy
diễn tập toàn bộ trên máy dev, và cũng là cách kiểm nhanh khi sửa một mắt xích sau này.

**Thế giới giả lập** (khớp bố trí mẫu `configs/lab/topology.yaml`):

- Sàn phẳng, đơn vị mét. cam01 nhìn x ∈ [0, 6], cam02 nhìn x ∈ [2, 8] (CHỒNG LẤN ở [2, 6]),
  cam03 nhìn hành lang x ∈ [18, 24], cam04 nhìn x ∈ [34, 40]; y ∈ [0, 4] (hành lang [0, 2]).
- Mỗi camera là một homography thật (hình thang phối cảnh: xa thì hẹp và nhỏ), nên điểm chân
  chiếu về đúng toạ độ sàn — thành phần hình học của engine có việc thật để làm.
- Người đi tuyến A → hành lang → đầu kia (và một số quay về), 1.1–1.5 m/s, xuất phát lệch nhau.

**Chú thích CVAT** (`cvat/<buổi>/camXX.xml`): định dạng "CVAT for video 1.1" với `<meta>` có
`frame_filter step=5`, đánh số khung TUYỆT ĐỐI ở cam01/cam03 và TƯƠNG ĐỐI ở cam02/cam04 — để
`cvat_to_mot` phải nhận ra cả hai kiểu.

**Fixture "pipeline"** (`fixtures/lab_<buổi>_r<n>.jsonl`, n lần chạy khác hạt giống):
detection bỏ sót ngẫu nhiên, hộp rung, id cục bộ do "tracker" cấp (đổi id giữa chừng với xác
suất `--switch-prob`), hộp báo nhầm ngắn, embedding 3 tầng như `tools.make_synthetic_fixture`
(gốc danh tính → lệch theo camera → nhiễu từng khung). `ts_ms` lệch một hằng số khác nhau mỗi
lần chạy — như đồng hồ của lần chạy pipeline thật, không liên quan đồng hồ chú thích.

Kèm `configs/topology.yaml` (status measured) và `configs/homography/camXX.yaml` thật.
KHÔNG dùng số đo trên dữ liệu này cho báo cáo: đây là phép thử đường ống, không phải đánh giá.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from common.schema import Detection, FrameMessage, l2_normalize, write_jsonl
from mct.homography import CameraHomography, estimate_homography
from tools.make_synthetic_fixture import _correlated_unit, _random_unit, _sigma_for_intra_sim

FRAME_W, FRAME_H = 1920, 1080
BASE_TS_MS = 1_788_231_600_000
STEP = 5

# Vùng sàn mỗi camera nhìn thấy: (x0, x1, y0, y1) mét. Cạnh y1 là cạnh XA camera.
VIEWS: dict[str, tuple[float, float, float, float]] = {
    "cam01": (0.0, 6.0, 0.0, 4.0),
    "cam02": (2.0, 8.0, 0.0, 4.0),
    "cam03": (18.0, 24.0, 0.0, 2.0),
    "cam04": (34.0, 40.0, 0.0, 2.0),
}
OVERLAPS = {"cam01": ["cam02"], "cam02": ["cam01"], "cam03": [], "cam04": []}
RELATIVE_NUMBERING = {"cam02", "cam04"}


def camera_homography(cam_id: str) -> np.ndarray:
    """H (ảnh → sàn) của một camera: hình thang phối cảnh phủ vùng `VIEWS[cam_id]`.

    cam02 nhìn từ phía đối diện cam01 (trục x đảo chiều trong ảnh) — hai góc nhìn khác nhau
    như cặp chồng lấn thật.
    """
    x0, x1, y0, y1 = VIEWS[cam_id]
    near_l, near_r, far_l, far_r = (160, 1000), (1760, 1000), (560, 380), (1360, 380)
    world = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    if cam_id == "cam02":
        world = [(x1, y1), (x0, y1), (x1, y0), (x0, y0)]
    fit = estimate_homography([near_l, near_r, far_l, far_r], world, trim_ratio=0.0)
    return fit.matrix


@dataclass(slots=True)
class Person:
    pid: int
    start_s: float
    speed: float
    lane_y: float
    route: list[float]
    """Các mốc x mà người đi qua theo thứ tự (đi thẳng giữa hai mốc liên tiếp)."""

    def position(self, t: float) -> tuple[float, float] | None:
        """Toạ độ sàn lúc `t` giây, None nếu chưa xuất phát hoặc đã đi hết tuyến."""
        d = (t - self.start_s) * self.speed
        if d < 0:
            return None
        for a, b in zip(self.route, self.route[1:], strict=False):
            seg = abs(b - a)
            if d <= seg:
                x = a + math.copysign(d, b - a)
                y = self.lane_y + 0.3 * math.sin(0.7 * x)
                return (x, y)
            d -= seg
        return None


def build_people(n: int, rng: np.random.Generator) -> list[Person]:
    people = []
    for i in range(n):
        back = i % 2 == 1  # nửa số người đi tới đầu kia rồi quay lại khu A
        route = [0.5, 40.0, 1.0] if back else [0.5, 40.0]
        lane = float(rng.uniform(0.6, 1.6))
        people.append(
            Person(
                pid=i + 1,
                start_s=float(i * 4.0 + rng.uniform(0.0, 1.5)),
                speed=float(rng.uniform(1.1, 1.5)),
                lane_y=lane,
                route=route,
            )
        )
    return people


def foot_box(
    world_to_image: np.ndarray, point: tuple[float, float]
) -> tuple[float, float, float, float] | None:
    """Hộp (x, y, w, h) của người đứng ở `point`; None nếu chân ra ngoài khung."""
    v = world_to_image @ np.array([point[0], point[1], 1.0])
    if v[2] <= 0:
        return None
    fx, fy = v[0] / v[2], v[1] / v[2]
    if not (0 <= fx < FRAME_W and 300 <= fy < FRAME_H):
        return None
    h = 160.0 + (fy - 380.0) * 0.55  # xa (fy nhỏ) thì thấp
    w = 0.42 * h
    x, y = fx - w / 2.0, fy - h
    if x < 0 or y < 0 or x + w > FRAME_W:
        return None
    return (float(x), float(y), float(w), float(h))


def ground_truth(
    people: list[Person], seconds: float, fps: int
) -> dict[str, dict[int, list[tuple[int, int, tuple[float, float, float, float]]]]]:
    """{cam: {person: [(frame, đoạn xuất hiện, bbox)]}} ở MỌI khung (chưa nhảy khung)."""
    inv = {cam: np.linalg.inv(camera_homography(cam)) for cam in VIEWS}
    out: dict[str, dict[int, list]] = {cam: {} for cam in VIEWS}
    last_seen: dict[tuple[str, int], int] = {}
    segment: dict[tuple[str, int], int] = {}
    for frame in range(int(seconds * fps)):
        t = frame / fps
        for person in people:
            pos = person.position(t)
            if pos is None:
                continue
            for cam, (x0, x1, y0, y1) in VIEWS.items():
                if not (x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1):
                    continue
                box = foot_box(inv[cam], pos)
                if box is None:
                    continue
                key = (cam, person.pid)
                if key in last_seen and frame - last_seen[key] > fps:
                    segment[key] = segment.get(key, 0) + 1
                last_seen[key] = frame
                out[cam].setdefault(person.pid, []).append((frame, segment.get(key, 0), box))
    return out


def write_cvat(path: Path, cam: str, tracks: dict[int, list], n_frames: int) -> None:
    """XML "CVAT for video 1.1": một track mỗi (người, đoạn xuất hiện), nhảy khung STEP."""
    relative = cam in RELATIVE_NUMBERING
    parts = [
        '<?xml version="1.0" encoding="utf-8"?>\n<annotations><version>1.1</version><meta><task>',
        f"<size>{(n_frames - 1) // STEP + 1}</size><start_frame>0</start_frame>",
        f"<stop_frame>{n_frames - 1}</stop_frame><frame_filter>step={STEP}</frame_filter>",
        f"<original_size><width>{FRAME_W}</width><height>{FRAME_H}</height></original_size>",
        "</task></meta>",
    ]
    track_id = 0
    for pid, rows in sorted(tracks.items()):
        for seg in sorted({s for _, s, _ in rows}):
            boxes = [(f, b) for f, s, b in rows if s == seg and f % STEP == 0]
            if not boxes:
                continue
            parts.append(f'<track id="{track_id}" label="person" source="manual">')
            for frame, (x, y, w, h) in boxes:
                number = frame // STEP if relative else frame
                parts.append(
                    f'<box frame="{number}" keyframe="1" outside="0" occluded="0" '
                    f'xtl="{x:.2f}" ytl="{y:.2f}" xbr="{x + w:.2f}" ybr="{y + h:.2f}" z_order="0">'
                    f'<attribute name="person_id">P{pid:02d}</attribute></box>'
                )
            last = boxes[-1][0] + STEP
            if last < n_frames:
                number = last // STEP if relative else last
                x, y, w, h = boxes[-1][1]
                parts.append(
                    f'<box frame="{number}" keyframe="1" outside="1" occluded="0" '
                    f'xtl="{x:.2f}" ytl="{y:.2f}" xbr="{x + w:.2f}" ybr="{y + h:.2f}" z_order="0">'
                    f'<attribute name="person_id">P{pid:02d}</attribute></box>'
                )
            parts.append("</track>")
            track_id += 1
    parts.append("</annotations>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def pipeline_run(
    gt: dict[str, dict[int, list]],
    n_frames: int,
    fps: int,
    *,
    seed: int,
    dim: int,
    miss_prob: float,
    switch_prob: float,
    fp_rate: float,
    intra_sim: float,
    cross_cam_sim: float,
) -> list[FrameMessage]:
    """Một lần chạy "pipeline" trên cùng thế giới: bỏ sót, rung hộp, đổi id, báo nhầm."""
    rng = np.random.default_rng(seed)
    persons = sorted({pid for tracks in gt.values() for pid in tracks})
    base = {pid: _random_unit(np.random.default_rng(1000 + pid), dim) for pid in persons}
    alpha = math.sqrt(cross_cam_sim)
    view = {
        (cam, pid): _correlated_unit(
            np.random.default_rng(int(cam[-2:]) * 1000 + pid), base[pid], alpha
        )
        for cam in gt
        for pid in persons
    }
    sigma = _sigma_for_intra_sim(intra_sim, dim)
    clock = BASE_TS_MS + 3_600_000 * seed  # đồng hồ của lần chạy, khác đồng hồ chú thích

    by_frame: dict[tuple[str, int], list[Detection]] = {}
    for cam, tracks in gt.items():
        next_id = 1
        for pid, rows in sorted(tracks.items()):
            for seg in sorted({s for _, s, _ in rows}):
                local = next_id
                next_id += 1
                seg_rows = [(f, b) for f, s, b in rows if s == seg]
                switch_at = seg_rows[len(seg_rows) // 2][0] if rng.random() < switch_prob else None
                for frame, (x, y, w, h) in seg_rows:
                    if switch_at is not None and frame == switch_at:
                        local = next_id
                        next_id += 1
                    if rng.random() < miss_prob:
                        continue
                    jitter = rng.normal(0.0, 0.03, 4) * np.array([w, h, w, h])
                    emb = l2_normalize(view[(cam, pid)] + sigma * rng.standard_normal(dim))
                    by_frame.setdefault((cam, frame), []).append(
                        Detection(
                            local_track_id=local,
                            bbox=(
                                float(x + jitter[0]),
                                float(y + jitter[1]),
                                float(w + jitter[2]),
                                float(h + jitter[3]),
                            ),
                            confidence=float(rng.uniform(0.5, 0.95)),
                            embedding=emb,
                        )
                    )
        # Báo nhầm: các track ngắn (~1 s) ở chỗ không có ai.
        n_fp = int(fp_rate * n_frames / fps)
        for _ in range(n_fp):
            start = int(rng.integers(0, max(1, n_frames - fps)))
            x, y = float(rng.uniform(100, 1700)), float(rng.uniform(100, 600))
            local = next_id
            next_id += 1
            emb = _random_unit(rng, dim)
            for frame in range(start, min(n_frames, start + fps)):
                by_frame.setdefault((cam, frame), []).append(
                    Detection(local, (x, y, 60.0, 140.0), 0.4, embedding=emb)
                )

    out = []
    for cam in sorted(gt):
        for frame in range(n_frames):
            dets = by_frame.get((cam, frame), [])
            out.append(
                FrameMessage(
                    cam_id=cam,
                    frame_id=frame,
                    ts_ms=clock + round(frame * 1000 / fps),
                    frame_pts_ns=round(frame * 1e9 / fps),
                    frame_width=FRAME_W,
                    frame_height=FRAME_H,
                    detections=dets,
                    embed_dim=dim if dets else 0,
                )
            )
    out.sort(key=lambda m: (m.ts_ms, m.cam_id))
    return out


def write_configs(out: Path, fps: int, gt: dict[str, dict[int, list]]) -> None:
    hdir = out / "configs" / "homography"
    hdir.mkdir(parents=True, exist_ok=True)
    for cam in VIEWS:
        CameraHomography(
            cam_id=cam,
            matrix=camera_homography(cam),
            plane="ground",
            image_size=(FRAME_W, FRAME_H),
            source="tools.make_synthetic_lab (homography đúng của thế giới giả lập)",
        ).save(hdir / f"{cam}.yaml")
    topo = {
        "status": "measured",
        "cameras": {
            cam: {"resolution": [FRAME_W, FRAME_H], "fps": fps, "overlaps_with": OVERLAPS[cam]}
            for cam in VIEWS
        },
        "unknown_pair_policy": "allow",
        "transitions": [
            {"from": a, "to": "cam03", "bidirectional": True, "min_ms": 4000, "max_ms": 20000}
            for a in ("cam01", "cam02")
        ]
        + [
            {"from": "cam03", "to": "cam04", "bidirectional": True, "min_ms": 4000, "max_ms": 20000}
        ],
        "homography": {},
    }
    (out / "configs" / "topology.yaml").write_text(
        yaml.safe_dump(topo, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=Path("data/lab_synth"))
    p.add_argument("--session", default="syn1")
    p.add_argument("--people", type=int, default=6)
    p.add_argument("--seconds", type=float, default=90.0)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--miss-prob", type=float, default=0.08)
    p.add_argument("--switch-prob", type=float, default=0.25)
    p.add_argument("--fp-rate", type=float, default=0.05, help="track báo nhầm mỗi giây mỗi camera")
    p.add_argument("--intra-sim", type=float, default=0.80)
    p.add_argument("--cross-cam-sim", type=float, default=0.78)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    people = build_people(args.people, rng)
    n_frames = int(args.seconds * args.fps)
    gt = ground_truth(people, args.seconds, args.fps)
    for cam, tracks in gt.items():
        write_cvat(args.out / "cvat" / args.session / f"{cam}.xml", cam, tracks, n_frames)
    write_configs(args.out, args.fps, gt)
    for r in range(1, args.runs + 1):
        msgs = pipeline_run(
            gt,
            n_frames,
            args.fps,
            seed=args.seed * 100 + r,
            dim=args.dim,
            miss_prob=args.miss_prob,
            switch_prob=args.switch_prob,
            fp_rate=args.fp_rate,
            intra_sim=args.intra_sim,
            cross_cam_sim=args.cross_cam_sim,
        )
        write_jsonl(args.out / "fixtures" / f"lab_{args.session}_r{r}.jsonl", msgs)
    summary = {
        "session": args.session,
        "n_frames": n_frames,
        "people": [asdict(p) for p in people],
        "boxes_per_cam": {cam: sum(len(v) for v in t.values()) for cam, t in gt.items()},
    }
    (args.out / f"{args.session}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["boxes_per_cam"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
