"""Tách 217 Global ID "rác" làm hai: detector báo nhầm vs. bộ gán nhãn từ chối.

    PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m eval.diagnose_junk_ids \\
        --db data/mct-ds.db --gt data/fixtures/ds_wildtrack_7cam.gt.json \\
        --fixture data/fixtures/ds_wildtrack_7cam.jsonl \\
        --wildtrack-dir data/wildtrack --json data/junk_ids.json

**Vì sao có file này.** `eval/diagnose_global_ids.py` (phiên 20) phân rã 433 Global ID và
tìm ra 217 (50.1% số ID, 52.0% số khung) là "rác" — không một tracklet nào của chúng tra
được nhãn qua bảng `.gt.json`. Nhưng "rác" ở đó trộn HAI thứ đòi hai cách sửa khác nhau:

1. **hộp detector báo nhầm** — YOLO11s dựng một hộp không trùng người thật nào. Sửa ở
   detector (fine-tune, đổi ngưỡng conf, lọc lớp). `src/mct` không làm gì được.
2. **người thật mà `ds_wildtrack_gt.py` từ chối** — track có trùng một người WildTrack,
   nhưng bị loại khỏi bảng `.gt.json` vì độ thuần khiết < `--min-purity` hoặc khớp < 3
   khung. Track NÀY là dữ liệu thật; nó "rác" chỉ vì bảng chấm bỏ nó, và sửa nằm ở bộ gán
   nhãn (hoặc ở chính bước liên kết, nếu track đủ dài).

`diagnose_global_ids.py` không phân biệt được vì nó chỉ nhìn qua `.gt.json`. Công cụ này
đối chiếu hộp của từng tracklet rác **thẳng với chú thích WildTrack gốc** (`annotations_
positions/*.json`) bằng IoU — đúng cách `ds_wildtrack_gt.py` gán nhãn, nhưng KHÔNG áp
ngưỡng loại — rồi phân loại theo tỉ lệ khung trùng người thật.

Con số đầu ra quyết định phiên sau: nếu phần lớn khối lượng rác là (1) thì đi sửa detector;
nếu là (2) thì bộ gán nhãn đang giấu mất một phần dataset và cần nới.

Chỉ stdlib + numpy + scipy. Không GPU, không Redis. Chạy được trên head node `ut-hpc`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.schema import read_jsonl
from eval.diagnose_global_ids import Appearance, load_appearances, load_gt
from tools.ds_wildtrack_gt import collect_votes, gt_index, view_idx_for_cam
from tools.wildtrack_to_fixture import parse_raw_detections

PERCENTILES = (5, 25, 50, 75, 95)

# Ngưỡng phân loại một tracklet rác. Cùng con số với `ds_wildtrack_gt.py` để câu trả lời
# "vì sao bảng nhãn từ chối" đọc thẳng được: track vượt cả ba mà vẫn không có trong
# `.gt.json` là một mâu thuẫn cần ghi lại, không phải một loại.
DEFAULT_MIN_IOU = 0.5
DEFAULT_MIN_PURITY = 0.7
DEFAULT_MIN_MATCHED = 3
# Dưới tỉ lệ này thì hộp gần như không bao giờ chạm người thật -> detector báo nhầm.
DEFAULT_FP_MATCH_RATE = 0.30

CLASS_FP = "detector_fp"
CLASS_REAL = "real_person"
CLASS_NOT_IN_FIXTURE = "not_in_fixture"

REAL_SUBREASON = {
    "it_khung": "trùng người thật nhưng < min_matched khung -> bảng nhãn loại",
    "khong_thuan": "trùng người thật nhưng độ thuần khiết < min_purity -> bảng nhãn loại",
    "du_dieu_kien": "trùng người thật, vượt cả ba ngưỡng, nhưng vẫn vắng trong .gt.json (!)",
}


@dataclass(frozen=True)
class TrackVerdict:
    """Phán quyết cho một tracklet (cam_id, local_track_id) thuộc một Global ID rác."""

    global_id: int
    cam_id: str
    local_track_id: int
    n_frames: int  # từ bảng appearances — trọng số khi gộp lên Global ID
    n_detections: int  # số khung tracklet xuất hiện trong fixture
    n_matched: int  # số khung khớp IoU với một người WildTrack bất kỳ
    match_rate: float
    person_id: int  # người WildTrack chiếm đa số phiếu (-1 nếu không có)
    purity: float
    klass: str
    subreason: str  # chỉ có nghĩa khi klass == CLASS_REAL


def _quantiles(values: list[float]) -> list[float]:
    if not values:
        return [0.0] * len(PERCENTILES)
    return [float(v) for v in np.percentile(np.array(values, dtype=np.float64), PERCENTILES)]


def _quantile_row(name: str, values: list[float], fmt: str = "{:8.2f}") -> None:
    cells = "".join(fmt.format(v) for v in _quantiles(values))
    print(f"{name:28s}{cells}   (n={len(values)})")


def _quantile_header() -> None:
    print(f"\n{'':28s}" + "".join(f"{'p' + str(p):>8s}" for p in PERCENTILES))


def junk_global_ids(appearances: list[Appearance]) -> set[int]:
    """Global ID mà KHÔNG một tracklet nào tra được nhãn — định nghĩa "rác" của phiên 20.

    Giống hệt `decompose()` trong `diagnose_global_ids.py`: chỉ cần một tracklet có nhãn
    là Global ID đó đã nói về một người thật và không còn là rác.
    """
    by_gid: dict[int, list[Appearance]] = defaultdict(list)
    for a in appearances:
        by_gid[a.global_id].append(a)
    return {gid for gid, aps in by_gid.items() if all(a.gt_id is None for a in aps)}


def classify_tracks(
    appearances: list[Appearance],
    tracks: dict[tuple[str, int], object],
    junk_gids: set[int],
    *,
    fp_match_rate: float,
    min_purity: float,
    min_matched: int,
) -> list[TrackVerdict]:
    """Mỗi tracklet của một Global ID rác -> một phán quyết."""
    verdicts: list[TrackVerdict] = []
    for a in appearances:
        if a.global_id not in junk_gids:
            continue
        tv = tracks.get((a.cam_id, a.local_track_id))
        if tv is None:
            verdicts.append(
                TrackVerdict(
                    a.global_id,
                    a.cam_id,
                    a.local_track_id,
                    a.n_frames,
                    0,
                    0,
                    0.0,
                    -1,
                    0.0,
                    CLASS_NOT_IN_FIXTURE,
                    "",
                )
            )
            continue
        n_det = int(tv.n_detections)  # type: ignore[attr-defined]
        n_match = int(tv.n_matched)  # type: ignore[attr-defined]
        pid, purity = tv.winner()  # type: ignore[attr-defined]
        rate = n_match / n_det if n_det else 0.0

        if rate < fp_match_rate or pid < 0:
            klass, sub = CLASS_FP, ""
        else:
            klass = CLASS_REAL
            if n_match < min_matched:
                sub = "it_khung"
            elif purity < min_purity:
                sub = "khong_thuan"
            else:
                sub = "du_dieu_kien"
        verdicts.append(
            TrackVerdict(
                a.global_id,
                a.cam_id,
                a.local_track_id,
                a.n_frames,
                n_det,
                n_match,
                rate,
                int(pid),
                float(purity),
                klass,
                sub,
            )
        )
    return verdicts


def aggregate_by_gid(verdicts: list[TrackVerdict]) -> dict[int, str]:
    """Gộp phán quyết tracklet lên Global ID bằng bỏ phiếu CÓ TRỌNG SỐ theo số khung.

    Một Global ID rác có thể ôm cả tracklet báo nhầm lẫn tracklet người thật; phần khối
    lượng (số khung của bảng appearances) nghiêng về đâu thì Global ID thuộc về đó. Hoà
    (hoặc chỉ có `not_in_fixture`) xếp vào `detector_fp` — phía an toàn, vì không có bằng
    chứng nó là người thật.
    """
    frames: dict[int, Counter[str]] = defaultdict(Counter)
    for v in verdicts:
        key = v.klass if v.klass != CLASS_NOT_IN_FIXTURE else CLASS_FP
        frames[v.global_id][key] += max(v.n_frames, 1)
    return {
        gid: (CLASS_REAL if c[CLASS_REAL] > c[CLASS_FP] else CLASS_FP) for gid, c in frames.items()
    }


def report(
    verdicts: list[TrackVerdict],
    gid_class: dict[int, str],
    appearances: list[Appearance],
    junk_gids: set[int],
    covered_identities: set[int],
    *,
    fp_match_rate: float,
) -> dict:
    frames_by_gid: dict[int, int] = defaultdict(int)
    for a in appearances:
        if a.global_id in junk_gids:
            frames_by_gid[a.global_id] += a.n_frames
    junk_frames_total = sum(frames_by_gid.values())
    all_frames_total = sum(a.n_frames for a in appearances)

    n_fp = sum(1 for k in gid_class.values() if k == CLASS_FP)
    n_real = sum(1 for k in gid_class.values() if k == CLASS_REAL)
    fr_fp = sum(f for gid, f in frames_by_gid.items() if gid_class[gid] == CLASS_FP)
    fr_real = sum(f for gid, f in frames_by_gid.items() if gid_class[gid] == CLASS_REAL)

    print(f"\n{'=' * 78}")
    print("1. 217 GLOBAL ID RÁC — DETECTOR BÁO NHẦM hay BỘ GÁN NHÃN TỪ CHỐI")
    print("=" * 78)
    print(f"Global ID rác                          {len(junk_gids)}")
    print(
        f"  ({junk_frames_total} khung = "
        f"{100.0 * junk_frames_total / max(all_frames_total, 1):.1f}% toàn hệ thống)"
    )
    print(f"\n{'':38s}{'#ID':>8s}{'%ID':>8s}{'#khung':>10s}{'%khung rác':>12s}")
    print(
        f"{'DETECTOR BÁO NHẦM (sửa ở detector)':38s}"
        f"{n_fp:>8d}{100.0 * n_fp / max(len(junk_gids), 1):>7.1f}%"
        f"{fr_fp:>10d}{100.0 * fr_fp / max(junk_frames_total, 1):>11.1f}%"
    )
    print(
        f"{'NGƯỜI THẬT, BẢNG NHÃN LOẠI (sửa ở gán nhãn)':38s}"
        f"{n_real:>8d}{100.0 * n_real / max(len(junk_gids), 1):>7.1f}%"
        f"{fr_real:>10d}{100.0 * fr_real / max(junk_frames_total, 1):>11.1f}%"
    )

    # --- mục 2: người thật thì vì sao bảng nhãn loại, và có phải danh tính MỚI không
    real_tracks = [
        v for v in verdicts if gid_class.get(v.global_id) == CLASS_REAL and v.klass == CLASS_REAL
    ]
    sub_hist = Counter(v.subreason for v in real_tracks)
    real_pids = {v.person_id for v in real_tracks if v.person_id >= 0}
    new_pids = real_pids - covered_identities

    print(f"\n{'=' * 78}")
    print("2. PHẦN 'NGƯỜI THẬT' — VÌ SAO BỊ LOẠI, CÓ MỞ RỘNG TẬP CHẤM KHÔNG")
    print("=" * 78)
    print(f"tracklet người thật trong nhóm này     {len(real_tracks)}")
    for sub, n in sub_hist.most_common():
        print(f"  {sub:14s} {n:4d}   {REAL_SUBREASON.get(sub, '?')}")
    print(f"\ndanh tính WildTrack mà chúng trùng     {len(real_pids)}")
    print(f"  đã có trong .gt.json (146 phủ được)   {len(real_pids & covered_identities)}")
    print(f"  CHƯA có — nới bảng nhãn sẽ thêm       {len(new_pids)}")
    if new_pids:
        print(f"  personID mới: {sorted(new_pids)}")

    # --- mục 3: phân bố tỉ lệ khớp, để thấy hai lớp có tách bạch không
    fp_rates = [v.match_rate for v in verdicts if v.klass == CLASS_FP and v.n_detections]
    real_rates = [v.match_rate for v in verdicts if v.klass == CLASS_REAL]
    print(f"\n{'=' * 78}")
    print(f"3. PHÂN BỐ TỈ LỆ KHUNG TRÙNG NGƯỜI THẬT (ngưỡng cắt = {fp_match_rate})")
    print("=" * 78)
    _quantile_header()
    _quantile_row("detector_fp  (match_rate)", fp_rates)
    _quantile_row("real_person  (match_rate)", real_rates)
    fp_len = [float(v.n_detections) for v in verdicts if v.klass == CLASS_FP]
    real_len = [float(v.n_detections) for v in verdicts if v.klass == CLASS_REAL]
    _quantile_row("detector_fp  (#khung/track)", fp_len, "{:8.0f}")
    _quantile_row("real_person  (#khung/track)", real_len, "{:8.0f}")

    verdict = "đi SỬA DETECTOR" if fr_fp >= fr_real else "đi NỚI BỘ GÁN NHÃN / bước liên kết"
    print(f"\n{'=' * 78}")
    lean = "detector_fp" if fr_fp >= fr_real else "real_person"
    print(
        f"KẾT LUẬN: khối lượng rác nghiêng về '{lean}'"
        f" ({max(fr_fp, fr_real)}/{junk_frames_total} khung) -> phiên sau {verdict}."
    )
    print(f"{'=' * 78}")

    return {
        "n_junk_gids": len(junk_gids),
        "junk_frames_total": junk_frames_total,
        "all_frames_total": all_frames_total,
        "by_class": {
            CLASS_FP: {"n_gids": n_fp, "frames": fr_fp},
            CLASS_REAL: {"n_gids": n_real, "frames": fr_real},
        },
        "real_subreason_hist": dict(sub_hist),
        "real_person_ids": sorted(real_pids),
        "real_person_ids_new": sorted(new_pids),
        "n_real_person_ids_new": len(new_pids),
        "match_rate_quantiles": {
            "detector_fp": _quantiles(fp_rates),
            "real_person": _quantiles(real_rates),
        },
        "track_len_quantiles": {
            "detector_fp": _quantiles(fp_len),
            "real_person": _quantiles(real_len),
        },
        "verdict": "fix_detector" if fr_fp >= fr_real else "fix_labeler_or_linking",
        "tracks": [
            {
                "global_id": v.global_id,
                "cam_id": v.cam_id,
                "local_track_id": v.local_track_id,
                "n_frames": v.n_frames,
                "n_detections": v.n_detections,
                "n_matched": v.n_matched,
                "match_rate": v.match_rate,
                "person_id": v.person_id,
                "purity": v.purity,
                "class": v.klass,
                "subreason": v.subreason,
            }
            for v in verdicts
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--db", type=Path, required=True, help="SQLite store của lần chạy cần soi")
    p.add_argument("--gt", type=Path, required=True, help="bảng .gt.json tương ứng")
    p.add_argument("--fixture", type=Path, required=True, help="fixture .jsonl mà lần chạy đọc")
    p.add_argument(
        "--wildtrack-dir", type=Path, required=True, help="thư mục chứa annotations_positions/"
    )
    p.add_argument("--min-iou", type=float, default=DEFAULT_MIN_IOU)
    p.add_argument("--min-purity", type=float, default=DEFAULT_MIN_PURITY)
    p.add_argument("--min-matched", type=int, default=DEFAULT_MIN_MATCHED)
    p.add_argument(
        "--fp-match-rate",
        type=float,
        default=DEFAULT_FP_MATCH_RATE,
        help="dưới tỉ lệ này thì tracklet bị xếp là hộp detector báo nhầm",
    )
    p.add_argument("--frame-stride", type=int, default=1, help="phải khớp lúc đóng video")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    gt = load_gt(args.gt)
    appearances = load_appearances(args.db, gt)
    if not appearances:
        print("Bảng appearances rỗng.")
        return 1
    junk_gids = junk_global_ids(appearances)

    messages = list(read_jsonl(args.fixture))
    cam_ids = sorted({m.cam_id for m in messages})
    view_indices = sorted(view_idx_for_cam(c) for c in cam_ids)
    raw, n_ann_frames = parse_raw_detections(
        args.wildtrack_dir / "annotations_positions",
        view_indices=view_indices,
        stride=args.frame_stride,
        max_frames=0,
        min_box_area=0.0,
    )
    tracks, _ = collect_votes(messages, gt_index(raw), min_iou=args.min_iou)

    verdicts = classify_tracks(
        appearances,
        tracks,
        junk_gids,
        fp_match_rate=args.fp_match_rate,
        min_purity=args.min_purity,
        min_matched=args.min_matched,
    )
    gid_class = aggregate_by_gid(verdicts)
    covered = set(gt.values())

    print(
        f"{args.db.name}: {len(appearances)} lượt xuất hiện, {len(junk_gids)} Global ID rác | "
        f"{args.gt.name}: {len(covered)} danh tính phủ | "
        f"WildTrack: {len(raw)} hộp chú thích / {n_ann_frames} khung"
    )
    out = report(
        verdicts,
        gid_class,
        appearances,
        junk_gids,
        covered,
        fp_match_rate=args.fp_match_rate,
    )
    out["db"] = str(args.db)
    out["gt"] = str(args.gt)
    out["fixture"] = str(args.fixture)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nsố liệu đầy đủ: {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
