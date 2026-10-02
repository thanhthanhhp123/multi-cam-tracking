"""Đo thời gian di chuyển giữa các camera từ ground-truth → đề xuất khối `transitions`.

    PYTHONPATH=src python -m tools.estimate_transit \\
        --gt-fixture data/fixtures/lab_calib_gt.jsonl \\
        --topology configs/lab/topology.yaml --yaml-out data/lab/transitions.suggest.yaml

**Vị trí trong M6** (đề cương mục 4.3.2 bước 1: "ghi nhận khoảng cách và thời gian di chuyển
tối thiểu giữa các camera"). Đo tay bằng đồng hồ bấm giờ được, nhưng đã có chú thích CVAT thì
thời gian đi giữa hai camera đọc thẳng từ đó, đúng theo định nghĩa mà engine dùng:

    Δt = lúc xuất hiện ở camera đích − lúc thấy LẦN CUỐI ở camera nguồn

(`mct.affinity` gọi `topology.check(track.last_cam_id, tracklet.cam_id, Δt)`). Cặp chồng lấn
cho Δt ÂM (người còn ở camera nguồn khi đã hiện ở camera đích) — công cụ nhận ra và đề xuất
`overlaps_with` cho cặp đó.

**ĐỪNG đo trên chính đoạn video dùng để chấm điểm.** Lấy `min_ms`/`max_ms` từ tập test rồi
chấm trên tập test là chỉnh tham số trên đáp án: ràng buộc thời gian sẽ vừa khít mọi cặp đúng
của tập đó. Quay riêng một đoạn hiệu chỉnh (một người đi hết các tuyến, đi chậm / bình thường /
nhanh — `docs/m6/README.md`), đo trên đoạn đó, rồi nới thêm biên (`--margin`).

**Lần xuất hiện** (`Appearance`): các hộp liên tiếp của một người ở một camera, cắt khi
khoảng trống > `--gap-ms`. Người gán nhãn CVAT có thể để một track xuyên suốt lúc người đó
ra khỏi khung rồi quay lại (khung `outside`) — cắt theo khoảng trống mới ra đúng hai lần xuất
hiện. `eval/eval_handover.py` dùng lại đúng định nghĩa này.

Chỉ stdlib + `common/` + PyYAML.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from common.logging import get_logger
from common.schema import FrameMessage, read_jsonl
from tools.export_trackeval import load_gt_table

log = get_logger("tools.estimate_transit")

Bbox = tuple[float, float, float, float]

DEFAULT_GAP_MS = 2000
PAIR_KINDS = ("overlap", "non_overlap", "same_camera")


@dataclass(slots=True)
class Appearance:
    """Một lần một người (theo ground-truth) có mặt liên tục ở một camera."""

    person: int
    cam_id: str
    boxes: list[tuple[int, int, Bbox]] = field(default_factory=list)
    """[(frame_id, ts_ms, bbox)] theo thời gian."""

    @property
    def start_ms(self) -> int:
        return self.boxes[0][1]

    @property
    def end_ms(self) -> int:
        return self.boxes[-1][1]

    @property
    def n_boxes(self) -> int:
        return len(self.boxes)


def build_appearances(
    gt_messages: list[FrameMessage],
    table: dict[tuple[str, int], int],
    *,
    gap_ms: int = DEFAULT_GAP_MS,
) -> list[Appearance]:
    """Fixture ground-truth → các lần xuất hiện, sắp theo (người, lúc bắt đầu).

    Gom theo (người, camera) chứ không theo track CVAT: hai track của cùng một người ở cùng
    camera (người gán nhãn tạo track mới khi người đó quay lại) nối tiếp nhau vẫn phải
    được cắt/gộp theo khoảng trống, giống hệt một track có khung `outside` ở giữa.
    """
    per_key: dict[tuple[int, str], list[tuple[int, int, Bbox]]] = defaultdict(list)
    for msg in gt_messages:
        for det in msg.detections:
            person = table.get((msg.cam_id, int(det.local_track_id)))
            if person is None:
                continue
            box = tuple(float(v) for v in det.bbox)
            per_key[(person, msg.cam_id)].append((int(msg.frame_id), int(msg.ts_ms), box))  # type: ignore[arg-type]

    out: list[Appearance] = []
    for (person, cam_id), rows in per_key.items():
        rows.sort(key=lambda r: (r[1], r[0]))
        current = Appearance(person, cam_id)
        for row in rows:
            if current.boxes and row[1] - current.boxes[-1][1] > gap_ms:
                out.append(current)
                current = Appearance(person, cam_id)
            current.boxes.append(row)
        out.append(current)
    out.sort(key=lambda a: (a.person, a.start_ms, a.cam_id))
    return out


@dataclass(frozen=True, slots=True)
class Handover:
    """Lần `dst` xuất hiện, nối với lần thấy cuối cùng trước đó của cùng người (`src`)."""

    src: Appearance
    dst: Appearance
    elapsed_ms: int
    """`dst.start_ms − src.end_ms` — âm khi hai camera cùng thấy người đó (chồng lấn)."""

    @property
    def pair(self) -> tuple[str, str]:
        return (self.src.cam_id, self.dst.cam_id)


def handovers(appearances: list[Appearance]) -> list[Handover]:
    """Mọi lần một người "đến" một camera sau khi đã được thấy ở đâu đó.

    Nguồn = lần xuất hiện trước đó (bắt đầu trước `dst`) có lúc thấy cuối MUỘN NHẤT — đúng
    `track.last_cam_id` mà engine dùng: với cặp chồng lấn, người còn đang ở camera cũ khi
    hiện ở camera mới, nên nguồn có thể kết thúc SAU lúc `dst` bắt đầu (Δt âm).
    """
    by_person: dict[int, list[Appearance]] = defaultdict(list)
    for app in appearances:
        by_person[app.person].append(app)

    out: list[Handover] = []
    for apps in by_person.values():
        apps.sort(key=lambda a: (a.start_ms, a.cam_id))
        for i, dst in enumerate(apps):
            earlier = [a for a in apps[:i] if a.start_ms <= dst.start_ms]
            if not earlier:
                continue
            src = max(earlier, key=lambda a: (min(a.end_ms, dst.start_ms), a.end_ms))
            out.append(Handover(src, dst, dst.start_ms - src.end_ms))
    return out


def pair_kind(src: str, dst: str, overlaps: dict[str, set[str]]) -> str:
    if src == dst:
        return "same_camera"
    return "overlap" if dst in overlaps.get(src, set()) else "non_overlap"


def load_overlaps(topology_path: Path | None) -> dict[str, set[str]]:
    """{cam: {camera chồng lấn}} từ topology.yaml. Không có file thì coi như không cặp nào."""
    if topology_path is None:
        return {}
    data = yaml.safe_load(topology_path.read_text(encoding="utf-8")) or {}
    out: dict[str, set[str]] = defaultdict(set)
    for cam, spec in (data.get("cameras") or {}).items():
        for other in (spec or {}).get("overlaps_with") or []:
            out[str(cam)].add(str(other))
            out[str(other)].add(str(cam))
    return dict(out)


@dataclass(frozen=True, slots=True)
class PairStats:
    src: str
    dst: str
    n: int
    min_ms: int
    median_ms: float
    max_ms: int
    n_negative: int

    def suggestion(self, *, margin: float, slack_ms: int) -> dict:
        """Khối `transitions` đề xuất. Biên nới về CẢ hai phía: chặt quá là cắt match đúng."""
        if self.n_negative:
            # Có lúc hai camera cùng thấy người đó → cặp chồng lấn: min 0, max = độ lệch
            # lớn nhất từng gặp (theo |Δt|) + biên. Engine xét |Δt| cho cặp chồng lấn.
            span = max(abs(self.min_ms), abs(self.max_ms))
            return {
                "from": self.src,
                "to": self.dst,
                "min_ms": 0,
                "max_ms": math.ceil(span * (1 + margin) + slack_ms),
            }
        return {
            "from": self.src,
            "to": self.dst,
            "min_ms": max(0, math.floor(self.min_ms * (1 - margin))),
            "max_ms": math.ceil(self.max_ms * (1 + margin) + slack_ms),
        }


def pair_stats(items: list[Handover]) -> list[PairStats]:
    by_pair: dict[tuple[str, str], list[int]] = defaultdict(list)
    for h in items:
        if h.src.cam_id != h.dst.cam_id:
            by_pair[h.pair].append(h.elapsed_ms)
    return [
        PairStats(
            src=src,
            dst=dst,
            n=len(v),
            min_ms=min(v),
            median_ms=float(statistics.median(v)),
            max_ms=max(v),
            n_negative=sum(1 for x in v if x < 0),
        )
        for (src, dst), v in sorted(by_pair.items())
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--gt-fixture", type=Path, required=True)
    p.add_argument("--gt-fixture-table", type=Path, default=None)
    p.add_argument("--topology", type=Path, default=None, help="để biết cặp nào đã khai chồng lấn")
    p.add_argument("--gap-ms", type=int, default=DEFAULT_GAP_MS)
    p.add_argument("--margin", type=float, default=0.3, help="nới min/max theo tỉ lệ")
    p.add_argument("--slack-ms", type=int, default=2000, help="cộng thêm vào max_ms")
    p.add_argument("--min-n", type=int, default=2, help="cặp ít mẫu hơn thì chỉ cảnh báo")
    p.add_argument("--yaml-out", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    table_path = args.gt_fixture_table or Path(str(args.gt_fixture).replace(".jsonl", ".gt.json"))
    apps = build_appearances(
        list(read_jsonl(args.gt_fixture)), load_gt_table(table_path), gap_ms=args.gap_ms
    )
    items = handovers(apps)
    stats = pair_stats(items)
    overlaps = load_overlaps(args.topology)

    print(f"{len(apps)} lần xuất hiện, {len(items)} lần chuyển (kể cả quay lại cùng camera)\n")
    print("| nguồn → đích | n | min (s) | trung vị (s) | max (s) | Δt<0 | topology | đề xuất |")
    print("|---|---|---|---|---|---|---|---|")
    suggestions: list[dict] = []
    new_overlaps: set[tuple[str, str]] = set()
    for s in stats:
        kind = pair_kind(s.src, s.dst, overlaps)
        sug = s.suggestion(margin=args.margin, slack_ms=args.slack_ms)
        if s.n_negative and kind != "overlap":
            new_overlaps.add(tuple(sorted((s.src, s.dst))))  # type: ignore[arg-type]
        note = f"[{sug['min_ms'] / 1000:.1f}, {sug['max_ms'] / 1000:.1f}] s"
        if s.n < args.min_n:
            note += " (ít mẫu)"
        else:
            suggestions.append(sug)
        print(
            f"| {s.src} → {s.dst} | {s.n} | {s.min_ms / 1000:.1f} | {s.median_ms / 1000:.1f} | "
            f"{s.max_ms / 1000:.1f} | {s.n_negative} | {kind} | {note} |"
        )
    for a, b in sorted(new_overlaps):
        log.warning(
            "%s và %s cùng thấy một người (Δt < 0) nhưng topology không khai chồng lấn — "
            "thêm vào overlaps_with của cả hai",
            a,
            b,
        )

    if args.yaml_out is not None:
        args.yaml_out.parent.mkdir(parents=True, exist_ok=True)
        args.yaml_out.write_text(
            "# Đề xuất từ tools.estimate_transit — soát tay trước khi chép vào topology.yaml.\n"
            f"# margin={args.margin}, slack_ms={args.slack_ms}, gap_ms={args.gap_ms}\n"
            + yaml.safe_dump(
                {
                    "overlap_pairs_detected": [list(x) for x in sorted(new_overlaps)],
                    "transitions": suggestions,
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        print(f"\n-> {args.yaml_out}")
    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "n_appearances": len(apps),
                    "n_handovers": len(items),
                    "pairs": [
                        {**asdict(s), "kind": pair_kind(s.src, s.dst, overlaps)} for s in stats
                    ],
                    "suggestions": suggestions,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
