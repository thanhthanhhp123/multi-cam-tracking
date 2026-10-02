"""Bàn giao danh tính giữa các camera, tách theo loại cặp: chồng lấn / không chồng lấn / quay lại.

    PYTHONPATH=src python -m eval.eval_handover \\
        --fixture data/fixtures/lab_s1_r1.jsonl --db data/lab/s1_r1/mct.db \\
        --gt-fixture data/fixtures/lab_s1_gt.jsonl \\
        --topology configs/lab/topology.yaml --json data/lab/s1_r1/handover.json

**Vì sao cần, khi đã có HOTA/IDF1.** HOTA và IDF1 trên chuỗi ảo ghép mọi camera trả lời "hệ
thống giữ danh tính tốt đến đâu, tính chung". Đóng góp chính của đồ án lại là liên kết cho
cấu hình HỖN HỢP chồng lấn/không chồng lấn (CLAUDE.md §1, §7), và đề cương tách đúng hai kịch
bản đó (mục 6.2, kịch bản 1 và 2). Một con số chung không cho biết phần nào tốt, phần nào kém;
khối lượng khung của cặp chồng lấn (người đứng trong vùng chung lâu) còn có thể che kết quả của
cặp không chồng lấn (vài lần đi qua). Nên đo thẳng đơn vị của bài toán: **mỗi lần một người
đến một camera mới, hệ thống có giữ được danh tính của người đó không?**

**Định nghĩa** (`tools/estimate_transit.py` dựng các lần chuyển từ ground-truth):

1. Lần xuất hiện = hộp liên tiếp của một người ở một camera, cắt khi trống > `--gap-ms`.
2. Lần chuyển (handover) `src → dst`: `dst` là một lần xuất hiện, `src` là lần xuất hiện
   trước đó của cùng người có lúc thấy cuối muộn nhất — đúng `track.last_cam_id` của engine.
   Loại cặp lấy theo `overlaps_with` của topology; `src` và `dst` cùng camera là "quay lại".
3. Danh tính DỰ ĐOÁN của một lần xuất hiện ở đầu/cuối: ghép hộp GT với hộp của hệ thống
   (IoU, Hungarian trong từng khung, như `tools/assign_gt.py`), lấy Global ID qua bảng
   `appearances` của SQLite (tra theo thời điểm), rồi bỏ phiếu đa số trên `--edge-frames`
   khung KHỚP cuối cùng của `src` và đầu tiên của `dst`.
4. Kết quả của lần chuyển:
   - `dung` — hai đầu cùng một Global ID;
   - `tach` — khác Global ID, và Global ID ở `dst` không thuộc người nào khác (hệ thống coi là
     người mới: lỗi bỏ sót liên kết);
   - `nham` — Global ID ở `dst` là của NGƯỜI KHÁC (chủ đa số của Global ID đó khác người này:
     gộp nhầm/tráo danh tính — lỗi nặng hơn với giám sát an ninh);
   - `sot` — một trong hai đầu không có dự đoán (detector/tracker bỏ sót, hoặc tracklet chưa
     được gán vì `min_frames`): không phải lỗi của bước liên kết, báo riêng.

**Độ chính xác bàn giao** = `dung / (dung + tach + nham)` — chỉ trên các lần chuyển mà bước
liên kết THẤY được cả hai đầu, nên nó là chỉ số của `src/mct`, không trộn lỗi detector vào.
`dung / tổng` báo kèm để không giấu phần `sot`.

Chỉ stdlib + numpy/scipy (qua `tools.ds_wildtrack_gt.match_frame`) + `common/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from common.logging import get_logger
from common.schema import FrameMessage, read_jsonl
from tools.ds_wildtrack_gt import match_frame
from tools.estimate_transit import (
    DEFAULT_GAP_MS,
    PAIR_KINDS,
    Appearance,
    Handover,
    build_appearances,
    handovers,
    load_overlaps,
    pair_kind,
)
from tools.export_trackeval import GlobalIdIndex, load_global_ids, load_gt_table

log = get_logger("eval.eval_handover")

OUTCOMES = ("dung", "tach", "nham", "sot")


def predicted_ids(
    result: list[FrameMessage],
    gt_messages: list[FrameMessage],
    table: dict[tuple[str, int], int],
    global_ids: GlobalIdIndex,
    *,
    min_iou: float,
) -> dict[tuple[str, int, int], int]:
    """(cam_id, frame_id, người GT) -> Global ID của hộp hệ thống khớp với người đó.

    Hộp hệ thống chưa có Global ID (chưa gán) thì không có mục — với người đó ở khung đó,
    hệ thống chưa nói gì.
    """
    by_frame: dict[tuple[str, int], FrameMessage] = {(m.cam_id, int(m.frame_id)): m for m in result}
    out: dict[tuple[str, int, int], int] = {}
    for gt in gt_messages:
        persons = [(table.get((gt.cam_id, int(d.local_track_id))), d) for d in gt.detections]
        persons = [(p, d) for p, d in persons if p is not None]
        res = by_frame.get((gt.cam_id, int(gt.frame_id)))
        if not persons or res is None or not res.detections:
            continue
        pairs = match_frame(
            [tuple(float(v) for v in d.bbox) for d in res.detections],  # type: ignore[misc]
            [tuple(float(v) for v in d.bbox) for _, d in persons],  # type: ignore[misc]
            min_iou=min_iou,
        )
        for res_i, gt_i, _ in pairs:
            det = res.detections[res_i]
            gid = global_ids.get(res.cam_id, int(det.local_track_id), int(res.ts_ms))
            if gid is not None:
                out[(gt.cam_id, int(gt.frame_id), int(persons[gt_i][0]))] = gid  # type: ignore[arg-type]
    return out


def edge_id(
    app: Appearance, pred: dict[tuple[str, int, int], int], *, edge_frames: int, tail: bool
) -> int | None:
    """Global ID đa số trên `edge_frames` khung KHỚP ở đầu (`tail=False`) hoặc cuối."""
    frames = [f for f, _, _ in (reversed(app.boxes) if tail else app.boxes)]
    ids: list[int] = []
    for frame in frames:
        gid = pred.get((app.cam_id, frame, app.person))
        if gid is not None:
            ids.append(gid)
            if len(ids) >= edge_frames:
                break
    if not ids:
        return None
    counts = Counter(ids).most_common()
    best = counts[0][1]
    # Hoà phiếu: lấy Global ID gần mép nhất (đầu danh sách) cho kết quả tất định.
    return next(g for g in ids if dict(counts)[g] == best)


def id_owners(pred: dict[tuple[str, int, int], int]) -> dict[int, int]:
    """Global ID -> người GT chiếm đa số các hộp mang Global ID đó (chủ của nó)."""
    votes: dict[int, Counter[int]] = defaultdict(Counter)
    for (_, _, person), gid in pred.items():
        votes[gid][person] += 1
    return {gid: c.most_common(1)[0][0] for gid, c in votes.items()}


@dataclass(slots=True)
class Scored:
    handover: Handover
    kind: str
    outcome: str
    src_id: int | None
    dst_id: int | None


def score_handovers(
    items: list[Handover],
    pred: dict[tuple[str, int, int], int],
    overlaps: dict[str, set[str]],
    *,
    edge_frames: int,
) -> list[Scored]:
    owners = id_owners(pred)
    out: list[Scored] = []
    for h in items:
        src_id = edge_id(h.src, pred, edge_frames=edge_frames, tail=True)
        dst_id = edge_id(h.dst, pred, edge_frames=edge_frames, tail=False)
        if src_id is None or dst_id is None:
            outcome = "sot"
        elif src_id == dst_id:
            outcome = "dung"
        elif owners.get(dst_id, h.dst.person) != h.dst.person:
            outcome = "nham"
        else:
            outcome = "tach"
        kind = pair_kind(h.src.cam_id, h.dst.cam_id, overlaps)
        out.append(Scored(h, kind, outcome, src_id, dst_id))
    return out


@dataclass(slots=True)
class Tally:
    counts: Counter[str] = field(default_factory=Counter)

    @property
    def n(self) -> int:
        return sum(self.counts.values())

    @property
    def seen(self) -> int:
        return self.counts["dung"] + self.counts["tach"] + self.counts["nham"]

    @property
    def accuracy(self) -> float | None:
        """Độ chính xác bàn giao: chỉ trên các lần chuyển liên kết thấy được cả hai đầu."""
        return self.counts["dung"] / self.seen if self.seen else None

    @property
    def end_to_end(self) -> float | None:
        return self.counts["dung"] / self.n if self.n else None

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            **{k: self.counts[k] for k in OUTCOMES},
            "accuracy": self.accuracy,
            "end_to_end": self.end_to_end,
        }


def summarize(scored: list[Scored]) -> dict:
    by_kind: dict[str, Tally] = {k: Tally() for k in PAIR_KINDS}
    by_pair: dict[str, Tally] = defaultdict(Tally)
    total = Tally()
    for s in scored:
        by_kind[s.kind].counts[s.outcome] += 1
        by_pair[f"{s.handover.src.cam_id}->{s.handover.dst.cam_id}"].counts[s.outcome] += 1
        total.counts[s.outcome] += 1
    return {
        "total": total.as_dict(),
        "by_kind": {k: t.as_dict() for k, t in by_kind.items()},
        "by_pair": {k: t.as_dict() for k, t in sorted(by_pair.items())},
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def format_table(summary: dict) -> str:
    names = {"overlap": "chồng lấn", "non_overlap": "không chồng lấn", "same_camera": "quay lại"}
    rows = [
        "| loại cặp | n | đúng | tách | nhầm người | sót | độ chính xác bàn giao | đúng / tổng |",
        "|---|---|---|---|---|---|---|---|",
    ]
    items = [(names[k], summary["by_kind"][k]) for k in PAIR_KINDS]
    items.append(("**tổng**", summary["total"]))
    for name, t in items:
        rows.append(
            f"| {name} | {t['n']} | {t['dung']} | {t['tach']} | {t['nham']} | {t['sot']} | "
            f"{_pct(t['accuracy'])} | {_pct(t['end_to_end'])} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fixture", type=Path, required=True, help="fixture của pipeline")
    p.add_argument("--db", type=Path, required=True, help="SQLite của engine (bảng appearances)")
    p.add_argument("--gt-fixture", type=Path, required=True)
    p.add_argument("--gt-fixture-table", type=Path, default=None)
    p.add_argument("--topology", type=Path, required=True)
    p.add_argument("--min-iou", type=float, default=0.5)
    p.add_argument("--edge-frames", type=int, default=5, help="số khung khớp ở mỗi mép")
    p.add_argument("--gap-ms", type=int, default=DEFAULT_GAP_MS)
    p.add_argument("--frame-offset", type=int, default=0, help="khung GT = khung pipeline + offset")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    table_path = args.gt_fixture_table or Path(str(args.gt_fixture).replace(".jsonl", ".gt.json"))
    table = load_gt_table(table_path)
    gt_messages = list(read_jsonl(args.gt_fixture))
    result = list(read_jsonl(args.fixture))
    if args.frame_offset:
        for m in result:
            m.frame_id = int(m.frame_id) + args.frame_offset

    apps = build_appearances(gt_messages, table, gap_ms=args.gap_ms)
    items = handovers(apps)
    pred = predicted_ids(result, gt_messages, table, load_global_ids(args.db), min_iou=args.min_iou)
    scored = score_handovers(
        items, pred, load_overlaps(args.topology), edge_frames=args.edge_frames
    )
    summary = summarize(scored)
    summary["params"] = {
        "min_iou": args.min_iou,
        "edge_frames": args.edge_frames,
        "gap_ms": args.gap_ms,
        "n_appearances": len(apps),
    }

    print(format_table(summary))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        detail = [
            {
                "person": s.handover.dst.person,
                "src": s.handover.src.cam_id,
                "dst": s.handover.dst.cam_id,
                "elapsed_ms": s.handover.elapsed_ms,
                "dst_start_ms": s.handover.dst.start_ms,
                "kind": s.kind,
                "outcome": s.outcome,
                "src_id": s.src_id,
                "dst_id": s.dst_id,
            }
            for s in scored
        ]
        args.json.write_text(
            json.dumps({**summary, "handovers": detail}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
