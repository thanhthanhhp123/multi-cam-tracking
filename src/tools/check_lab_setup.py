"""Kiểm cấu hình dữ liệu tự thu (M6) TRƯỚC khi thuê GPU — gom mọi lỗi im lặng đã gặp.

    PYTHONPATH=src python -m tools.check_lab_setup --session s1
    PYTHONPATH=src python -m tools.check_lab_setup --session s1 \\
        --gt-fixture data/fixtures/lab_s1_gt.jsonl

**Vì sao cần.** Gần như mọi lỗi tốn tiền của đồ án là lỗi KHÔNG có triệu chứng: quên khai
topology thì thành phần hình học bị bỏ qua (2800 message ra 42 Global ID, worklog phiên 6),
camera lạ trong topology thì engine chết giữa chừng (phiên 9), hiệu chỉnh ở độ phân giải khác
khung hình thì toạ độ mét sai mà không ai báo (CLAUDE.md §5), số khung chú thích lệch số khung
pipeline thì TrackEval vẫn chấm. Mỗi lỗi đó, nếu lộ ra trên `vast-gpu`, là một lượt thuê máy.
Công cụ này đọc toàn bộ cấu hình của một buổi quay và đối chiếu chéo, trên máy dev, miễn phí.

Kiểm những gì (mỗi dòng in OK / WARN / FAIL; có FAIL thì mã thoát 1):

1. topology: đọc được, không còn `status: template`, `overlaps_with` đối xứng.
2. streams: cam_id trùng khớp topology, `${LAB_SESSION}` thế được, `sink.sync: true`,
   kích thước streammux = độ phân giải trong topology.
3. homography: cặp chồng lấn PHẢI có đủ hai camera (FAIL), camera khác nên có (WARN: mất
   nối mảnh cùng camera); `image_size` khớp topology; cùng một `plane`.
4. ground_points: không còn `status: template` (nếu thư mục homography chưa có gì).
5. engine: cấu hình đọc được bằng chính hàm của engine, và `ground_path_max_points` đủ cho
   fps (≥ fps × 20 s, không thì quỹ đạo bị tỉa thưa — phiên 16).
6. buổi quay (`data/lab/<buổi>/sync.json`, nếu có): đủ camera, fps khớp topology, độ tin cậy
   đồng bộ, có file video.
7. ground-truth (`--gt-fixture`, nếu có): cam_id khớp, kích thước khung khớp, số khung chú
   thích không vượt số khung video, có người đi qua cả cặp chồng lấn lẫn cặp không chồng lấn
   (không thì chỉ số bàn giao của loại đó rỗng).

Chỉ đọc file, không chạy gì nặng. Không import GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from common.schema import read_jsonl
from mct.affinity import AffinityConfig
from mct.homography import HomographyMapper
from mct.topology import Topology, TopologyError

LEVELS = ("OK", "WARN", "FAIL")


@dataclass
class Report:
    lines: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, level: str, topic: str, message: str) -> None:
        assert level in LEVELS
        self.lines.append((level, topic, message))

    def ok(self, topic: str, message: str) -> None:
        self.add("OK", topic, message)

    def warn(self, topic: str, message: str) -> None:
        self.add("WARN", topic, message)

    def fail(self, topic: str, message: str) -> None:
        self.add("FAIL", topic, message)

    @property
    def n_fail(self) -> int:
        return sum(1 for level, _, _ in self.lines if level == "FAIL")

    def render(self) -> str:
        return "\n".join(f"[{lv:4}] {topic}: {msg}" for lv, topic, msg in self.lines)


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def check_topology(path: Path, rep: Report) -> Topology | None:
    if not path.is_file():
        rep.fail("topology", f"không có {path}")
        return None
    raw = _yaml(path)
    try:
        topo = Topology.from_mapping(raw)
    except (TopologyError, KeyError, ValueError) as exc:
        rep.fail("topology", f"{path}: {exc}")
        return None
    if raw.get("status") == "template":
        rep.fail("topology", "còn `status: template` — số transit/độ phân giải chưa đo thật")
    cams = sorted(topo.cameras)
    pairs = [(a, b) for i, a in enumerate(cams) for b in cams[i + 1 :]]
    overlap = [p for p in pairs if topo.is_overlapping(*p)]
    linked = [p for p in pairs if not topo.is_overlapping(*p) and topo.transition(*p)]
    rep.ok(
        "topology",
        f"{len(cams)} camera, chồng lấn: {overlap or 'không có'}, có transit: "
        f"{linked or 'không có'}, unknown_pair_policy={topo.unknown_pair_policy}",
    )
    if not overlap:
        rep.warn("topology", "không có cặp chồng lấn nào — kịch bản 1 của đề cương không đo được")
    if pairs and len(overlap) == len(pairs):
        rep.warn("topology", "không có cặp KHÔNG chồng lấn nào — kịch bản 2 không đo được")
    return topo


def check_streams(path: Path, topo: Topology | None, session: str | None, rep: Report) -> dict:
    if not path.is_file():
        rep.fail("streams", f"không có {path}")
        return {}
    raw = _yaml(path)
    fails_before = rep.n_fail
    sources = raw.get("sources") or []
    cams = [str(s.get("cam_id")) for s in sources]
    if len(set(cams)) != len(cams):
        rep.fail("streams", f"cam_id trùng: {cams}")
    if topo is not None:
        missing = sorted(set(cams) - set(topo.cameras))
        extra = sorted(set(topo.cameras) - set(cams))
        if missing:
            rep.fail("streams", f"camera không có trong topology: {missing} (engine áp policy lạ)")
        if extra:
            rep.warn("streams", f"topology có nhưng streams không chạy: {extra}")
    env = dict(os.environ)
    if session:
        env["LAB_SESSION"] = session
    unresolved = []
    for s in sources:
        uri = str(s.get("uri", ""))
        if "${" in uri:
            expanded = uri
            for key, value in env.items():
                expanded = expanded.replace("${" + key + "}", value)
            if "${" in expanded:
                unresolved.append(uri)
    if unresolved:
        rep.fail("streams", f"biến chưa thế được (thiếu --session?): {unresolved[0]}")
    if not (raw.get("sink") or {}).get("sync", False) and any(
        str(s.get("uri", "")).startswith("file://") for s in sources
    ):
        rep.fail(
            "streams", "nguồn file mà sink.sync=false — ts_ms bị nén, ràng buộc thời gian vô nghĩa"
        )
    mux = raw.get("streammux") or {}
    if topo is not None:
        sizes = {topo.cameras[c].resolution for c in cams if c in topo.cameras}
        mux_size = (int(mux.get("width", 0)), int(mux.get("height", 0)))
        if sizes and sizes != {mux_size}:
            rep.warn(
                "streams",
                f"streammux {mux_size} khác độ phân giải camera {sorted(sizes)} — probe phải scale "
                "ngược (CLAUDE.md §5); kiểm lại nếu đổi",
            )
    if rep.n_fail == fails_before:
        rep.ok("streams", f"{len(sources)} nguồn: {', '.join(cams)}")
    return raw


def check_homography(
    hdir: Path, points: Path | None, topo: Topology | None, rep: Report
) -> HomographyMapper | None:
    files = sorted(hdir.glob("*.yaml")) if hdir.is_dir() else []
    if not files:
        if points is not None and points.is_file() and _yaml(points).get("status") == "template":
            rep.fail(
                "homography",
                f"{hdir} rỗng và {points} còn `status: template` — đo điểm sàn rồi chạy "
                "tools.calibrate_homography",
            )
        else:
            rep.fail("homography", f"{hdir} rỗng — thành phần hình học sẽ bị bỏ qua im lặng")
        return None
    try:
        mapper = HomographyMapper.load(hdir)
    except (ValueError, KeyError) as exc:
        rep.fail("homography", f"{hdir}: {exc}")
        return None
    calibrated = set(mapper.calibrated)
    rep.ok("homography", f"đã hiệu chỉnh: {sorted(calibrated)}")
    if topo is None:
        return mapper
    for cam in sorted(topo.cameras):
        if cam not in calibrated:
            overlapping = sorted(topo.cameras[cam].overlaps_with)
            if overlapping:
                rep.fail(
                    "homography",
                    f"{cam} chồng lấn với {overlapping} nhưng chưa hiệu chỉnh — mất hình học",
                )
            else:
                rep.warn("homography", f"{cam} chưa hiệu chỉnh — mất nối mảnh tracklet cùng camera")
            continue
        size = mapper.cameras[cam].image_size
        want = topo.cameras[cam].resolution
        if size is not None and tuple(size) != tuple(want):
            rep.fail("homography", f"{cam}: hiệu chỉnh ở {size} nhưng topology ghi {want}")
    return mapper


def check_engine_config(path: Path, topo: Topology | None, rep: Report) -> None:
    if not path.is_file():
        rep.fail("engine", f"không có {path}")
        return
    raw = _yaml(path)
    try:
        affinity = AffinityConfig.from_mapping(raw)
    except (ValueError, TypeError) as exc:
        rep.fail("engine", f"{path}: {exc}")
        return
    points = int((raw.get("tracklet") or {}).get("ground_path_max_points", 64))
    fps = max((c.fps for c in topo.cameras.values()), default=0) if topo else 0
    if fps and points < fps * 20:
        rep.warn(
            "engine",
            f"ground_path_max_points={points} < fps×20 = {fps * 20}: quỹ đạo người đứng lâu bị "
            "tỉa thưa, mất mốc thời gian chung (phiên 16)",
        )
    tol = int((raw.get("association") or {}).get("ground_time_tol_ms", 400))
    if fps and tol < 1000 / fps:
        rep.warn("engine", f"ground_time_tol_ms={tol} nhỏ hơn một khung ở {fps} fps")
    rep.ok(
        "engine",
        f"max_cost={affinity.max_cost}, max_cost_geometric={affinity.max_cost_geometric}, "
        f"window_ms={(raw.get('association') or {}).get('window_ms')}",
    )


def check_session(session_dir: Path, topo: Topology | None, rep: Report) -> dict | None:
    sync = session_dir / "sync.json"
    if not sync.is_file():
        rep.warn("buổi quay", f"chưa có {sync} (chạy tools.sync_recordings trước)")
        return None
    data = json.loads(sync.read_text(encoding="utf-8"))
    cams = sorted(data.get("cameras") or {})
    if topo is not None and set(cams) != set(topo.cameras):
        rep.fail("buổi quay", f"camera trong sync.json {cams} khác topology {sorted(topo.cameras)}")
    fps = float(data.get("fps", 0))
    if topo is not None:
        bad = [c for c in cams if c in topo.cameras and topo.cameras[c].fps != round(fps)]
        if bad:
            rep.fail("buổi quay", f"fps đầu ra {fps:g} khác fps trong topology của {bad}")
    for cam, est in (data.get("estimates") or {}).items():
        conf = est.get("confidence")
        if est.get("source") == "audio" and conf is not None and conf < 2.0:
            rep.warn(
                "buổi quay",
                f"{cam}: độ tin cậy đồng bộ {conf:.2f} thấp — kiểm bằng mắt khung vỗ tay",
            )
    missing = [c for c in cams if not (session_dir / f"{c}.mp4").is_file()]
    if missing:
        rep.warn("buổi quay", f"chưa có video đầu ra cho {missing} (chưa --execute?)")
    rep.ok(
        "buổi quay",
        f"{len(cams)} camera, {data.get('duration_s', 0):.0f} s ≈ {data.get('expected_frames')} "
        f"khung ở {fps:g} fps, khung vỗ tay {data.get('clap_frame')}",
    )
    return data


def check_ground_truth(
    gt_fixture: Path, topo: Topology | None, sync: dict | None, rep: Report
) -> None:
    table_path = Path(str(gt_fixture).replace(".jsonl", ".gt.json"))
    if not gt_fixture.is_file() or not table_path.is_file():
        rep.fail("ground-truth", f"thiếu {gt_fixture} hoặc {table_path}")
        return
    msgs = list(read_jsonl(gt_fixture))
    cams = sorted({m.cam_id for m in msgs})
    if topo is not None and set(cams) - set(topo.cameras):
        rep.fail(
            "ground-truth", f"cam_id lạ so với topology: {sorted(set(cams) - set(topo.cameras))}"
        )
    if topo is not None:
        for m in msgs:
            want = topo.cameras.get(m.cam_id)
            if want is not None and (m.frame_width, m.frame_height) != tuple(want.resolution):
                rep.fail(
                    "ground-truth",
                    f"{m.cam_id}: khung chú thích {m.frame_width}x{m.frame_height} khác topology "
                    f"{want.resolution} — CVAT có nhận video đã xoay/đã đồng bộ không?",
                )
                break
    if sync is not None and sync.get("expected_frames"):
        last = max(m.frame_id for m in msgs)
        if last >= int(sync["expected_frames"]) + 1:
            rep.fail(
                "ground-truth",
                f"khung chú thích lớn nhất {last} vượt số khung video {sync['expected_frames']} — "
                "chú thích trên file KHÁC file đã đồng bộ?",
            )
    table = json.loads(table_path.read_text(encoding="utf-8"))
    persons_by_cam: dict[int, set[str]] = {}
    for row in table.get("tracklets", []):
        persons_by_cam.setdefault(int(row["gt_global_id"]), set()).add(str(row["cam_id"]))
    n_frames = Counter(m.cam_id for m in msgs)
    rep.ok(
        "ground-truth",
        f"{len(persons_by_cam)} người, khung chú thích mỗi camera: "
        + ", ".join(f"{c}={n}" for c, n in sorted(n_frames.items())),
    )
    if topo is None:
        return
    kinds = Counter()
    for cams_seen in persons_by_cam.values():
        seen = sorted(cams_seen)
        for i, a in enumerate(seen):
            for b in seen[i + 1 :]:
                kinds["overlap" if topo.is_overlapping(a, b) else "non_overlap"] += 1
    for kind, name in (("overlap", "chồng lấn"), ("non_overlap", "không chồng lấn")):
        if not kinds[kind]:
            rep.warn("ground-truth", f"không người nào đi qua cặp {name} — chỉ số loại đó sẽ rỗng")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--lab-dir", type=Path, default=Path("configs/lab"))
    p.add_argument("--topology", type=Path, default=None, help="mặc định <lab-dir>/topology.yaml")
    p.add_argument("--streams", type=Path, default=None, help="mặc định <lab-dir>/streams_lab.yaml")
    p.add_argument(
        "--engine-config", type=Path, default=None, help="mặc định <lab-dir>/lab.mct.yaml"
    )
    p.add_argument(
        "--homography-dir", type=Path, default=None, help="mặc định <lab-dir>/homography"
    )
    p.add_argument("--session", default=None, help="tên buổi quay (thư mục data/lab/<buổi>)")
    p.add_argument("--data-dir", type=Path, default=Path("data/lab"))
    p.add_argument("--gt-fixture", type=Path, default=None)
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    lab = args.lab_dir
    rep = Report()
    topo = check_topology(args.topology or lab / "topology.yaml", rep)
    check_streams(args.streams or lab / "streams_lab.yaml", topo, args.session, rep)
    check_homography(
        args.homography_dir or lab / "homography", lab / "ground_points.yaml", topo, rep
    )
    check_engine_config(args.engine_config or lab / "lab.mct.yaml", topo, rep)
    sync = check_session(args.data_dir / args.session, topo, rep) if args.session else None
    if args.gt_fixture is not None:
        check_ground_truth(args.gt_fixture, topo, sync, rep)

    print(rep.render())
    print(f"\n{rep.n_fail} FAIL, {sum(1 for lv, _, _ in rep.lines if lv == 'WARN')} WARN")
    return 1 if rep.n_fail else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
