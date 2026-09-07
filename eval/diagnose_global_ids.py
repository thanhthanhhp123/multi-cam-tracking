"""Phần dư của các Global ID: engine sinh ra nhiều hơn số danh tính, và nhiều hơn ở ĐÂU.

    PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m eval.diagnose_global_ids \\
        --db data/mct-ds.db --gt data/fixtures/ds_wildtrack_7cam.gt.json \\
        --gt-identities 313 --json data/gid_surplus.json

**Vì sao có file này.** Phiên 18 chốt `max_cost: 0.90` với lập luận "433 > 313 danh tính
nghĩa là nghiêng về TÁCH thay vì GỘP, mà tách thì còn sửa được". Lập luận đó dựa trên đúng
một con số — tổng số Global ID — và con số đó không phân biệt nổi ba tình huống khác hẳn
nhau:

1. một người bị xé thành nhiều Global ID (**vỡ**) — sửa được ở bước liên kết;
2. Global ID không ứng với danh tính nào trong bảng GT (**rác**) — detector báo nhầm, hoặc
   `ds_wildtrack_gt.py` đã loại track vì không đủ thuần khiết. Không phải lỗi của `src/mct`
   và cũng không sửa được ở đó;
3. một Global ID ôm nhiều danh tính (**gộp**) — thứ mà lập luận trên khẳng định là hiếm,
   nhưng chưa từng được đếm.

Nếu phần lớn phần dư là (2) thì mọi nỗ lực chỉnh ngưỡng liên kết đều đi sai chỗ. Vì thế công
cụ này phân rã tổng số Global ID thành một đẳng thức KIỂM TRA ĐƯỢC (mục 1), rồi đo kích
thước các mảnh dư (mục 3) để biết phần dư đó có thật sự ăn vào AssA hay chỉ là bụi.

**Đọc từ kết quả đã chạy, không chạy lại engine.** Nguồn là bảng `appearances` của SQLite
store — chính thứ mà `tools/export_trackeval.py` dùng để xuất điểm — nên số ở đây nói về
đúng lần chạy đã cho ra HOTA/IDF1, không phải về một lần chạy khác.

Chỉ stdlib + numpy. Không cần GPU, không cần Redis.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mct.associator import reason_kind

PERCENTILES = (5, 25, 50, 75, 95)

REASON_LABELS = {
    "no_candidate": "ràng buộc không-thời gian loại sạch ứng viên",
    "threshold": "có ứng viên nhưng ngoại hình không đủ giống",
    "taken": "ứng viên rẻ nhất đã bị tracklet khác cùng camera lấy",
    "empty": "gallery đang rỗng (người đầu tiên)",
    "": "không còn dòng nào giữ lý do (người khai sinh đã bị gán đi nơi khác)",
}

GROUP_NAMES = ("mảnh chính", "mảnh dư", "rác")


@dataclass(frozen=True)
class Appearance:
    """Một dòng `appearances` đã tra được (hoặc không) về danh tính ground-truth."""

    global_id: int
    cam_id: str
    local_track_id: int
    n_frames: int
    start_ms: int
    reason: str
    gt_id: int | None


def load_gt(path: Path) -> dict[tuple[str, int], int]:
    """`.gt.json` -> {(cam_id, local_track_id): gt_global_id}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        (t["cam_id"], int(t["local_track_id"])): int(t["gt_global_id"]) for t in data["tracklets"]
    }


def load_appearances(db_path: Path, gt: dict[tuple[str, int], int]) -> list[Appearance]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT global_id, cam_id, local_track_id, n_frames, start_ms, reason "
            "FROM appearances ORDER BY start_ms"
        ).fetchall()
    finally:
        con.close()

    return [
        Appearance(
            global_id=int(gid),
            cam_id=str(cam),
            local_track_id=int(local),
            n_frames=int(n_frames),
            start_ms=int(start_ms),
            reason=str(reason),
            gt_id=gt.get((str(cam), int(local))),
        )
        for gid, cam, local, n_frames, start_ms, reason in rows
    ]


def _quantiles(values: list[float]) -> list[float]:
    if not values:
        return [0.0] * len(PERCENTILES)
    return [float(v) for v in np.percentile(np.array(values, dtype=np.float64), PERCENTILES)]


def _print_quantile_row(name: str, values: list[float], fmt: str = "{:8.1f}") -> None:
    print(f"{name:30s}" + "".join(fmt.format(v) for v in _quantiles(values)))


def _quantile_header() -> None:
    print(f"\n{'':30s}" + "".join(f"{'p' + str(p):>8s}" for p in PERCENTILES))


def _frames_by_identity_gid(appearances: list[Appearance]) -> dict[tuple[int, int], int]:
    """(danh tính, global_id) -> số khung. Bỏ qua lượt xuất hiện không có nhãn."""
    frames: dict[tuple[int, int], int] = defaultdict(int)
    for a in appearances:
        if a.gt_id is not None:
            frames[(a.gt_id, a.global_id)] += a.n_frames
    return frames


def main_fragment_ids(appearances: list[Appearance]) -> set[int]:
    """Global ID giữ NHIỀU KHUNG NHẤT của ít nhất một danh tính = mảnh chính của người đó.

    Mọi Global ID có nhãn khác là mảnh dư. Chọn theo số khung chứ không theo số tracklet:
    một mảnh gồm 5 tracklet ngắn 3 khung không phải là chỗ hệ thống nhận ra người đó.
    """
    by_identity: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (ident, gid), n in _frames_by_identity_gid(appearances).items():
        by_identity[ident].append((gid, n))
    # Hoà thì lấy global_id nhỏ hơn — chỉ để kết quả tái lập được, không mang ý nghĩa gì.
    return {max(items, key=lambda kv: (kv[1], -kv[0]))[0] for items in by_identity.values()}


# --------------------------------------------------------------------- phân rã


def decompose(appearances: list[Appearance], *, n_gt_identities: int | None) -> dict:
    """Đẳng thức phân rã tổng số Global ID. Mục 1 của báo cáo.

    `n_gid = n_rác + n_danh_tính_phủ + n_mảnh_dư − n_chồng_do_gộp`

    Vế phải là bốn đại lượng đo được độc lập nhau, nên đẳng thức này vừa là kết quả vừa là
    phép kiểm: lệch một đơn vị nghĩa là một trong bốn phép đếm sai. Kiểm nó ngay tại đây,
    vì một báo cáo tự mâu thuẫn mà không ai phát hiện là thứ đi thẳng vào chương 6.
    """
    by_gid: dict[int, list[Appearance]] = defaultdict(list)
    for a in appearances:
        by_gid[a.global_id].append(a)

    # Rác = KHÔNG có lấy một tracklet nào tra được về danh tính. Chỉ cần một tracklet có
    # nhãn là Global ID đó đã nói về một người thật, dù phần còn lại của nó là gì.
    unlabeled = {gid for gid, aps in by_gid.items() if all(a.gt_id is None for a in aps)}
    labeled = set(by_gid) - unlabeled

    # (Global ID, danh tính) — mỗi cặp là một lần một người xuất hiện dưới một Global ID.
    pairs = {(a.global_id, a.gt_id) for a in appearances if a.gt_id is not None}
    gids_per_identity: dict[int, set[int]] = defaultdict(set)
    identities_per_gid: dict[int, set[int]] = defaultdict(set)
    for gid, ident in pairs:
        gids_per_identity[ident].add(gid)
        identities_per_gid[gid].add(ident)

    n_covered = len(gids_per_identity)
    surplus_split = sum(len(g) - 1 for g in gids_per_identity.values())
    merge_overlap = sum(len(i) - 1 for i in identities_per_gid.values())

    n_gid = len(by_gid)
    balance = len(unlabeled) + n_covered + surplus_split - merge_overlap
    if balance != n_gid:  # pragma: no cover - chỉ xảy ra khi một phép đếm sai
        raise AssertionError(
            f"đẳng thức phân rã không khớp: {balance} != {n_gid} — một phép đếm sai"
        )

    frames_total = sum(a.n_frames for a in appearances)
    frames_unlabeled = sum(a.n_frames for a in appearances if a.gt_id is None)
    local_tracks = {(a.cam_id, a.local_track_id) for a in appearances}
    local_labeled = {(a.cam_id, a.local_track_id) for a in appearances if a.gt_id is not None}

    print(f"\n{'=' * 78}\n1. PHÂN RÃ TỔNG SỐ GLOBAL ID\n{'=' * 78}")
    print(f"Global ID engine sinh ra                {n_gid}")
    print(
        f"  ├─ RÁC (không tracklet nào có nhãn)   {len(unlabeled):5d}"
        f"   {100.0 * len(unlabeled) / max(n_gid, 1):5.1f}% số ID,"
        f" {100.0 * frames_unlabeled / max(frames_total, 1):5.1f}% số khung"
    )
    print(f"  └─ CÓ NHÃN                            {len(labeled):5d}")
    print(f"       ├─ danh tính phủ được            {n_covered:5d}  (mỗi danh tính 1 mảnh chính)")
    print(f"       ├─ mảnh DƯ do vỡ               + {surplus_split:5d}")
    print(f"       └─ chồng do GỘP                − {merge_overlap:5d}")
    if n_gt_identities:
        print(
            f"\ndanh tính trong ground-truth            {n_gt_identities}"
            f"  → phủ {n_covered} ({100.0 * n_covered / n_gt_identities:.1f}%),"
            f" trượt {n_gt_identities - n_covered}"
        )
    # Trần của phép đo: "rác" chỉ chặt bằng bảng nhãn. Track mà `ds_wildtrack_gt.py` loại
    # vì không đủ thuần khiết rơi vào đây y hệt như một hộp detector báo nhầm, và hai thứ
    # đó đòi hai cách sửa khác nhau. In tỉ lệ phủ ra để không ai đọc "rác" thành "nhiễu".
    print(
        f"local track của pipeline                {len(local_tracks)}"
        f"  → tra được nhãn {len(local_labeled)}"
        f" ({100.0 * len(local_labeled) / max(len(local_tracks), 1):.1f}%)"
    )
    print(
        f"\nĐọc: dư {n_gid - n_covered} Global ID so với {n_covered} danh tính phủ được,"
        f" trong đó {len(unlabeled)} là rác"
        f" và {surplus_split - merge_overlap} là do vỡ (đã trừ phần gộp)."
    )

    return {
        "n_global_ids": n_gid,
        "n_unlabeled": len(unlabeled),
        "n_labeled": len(labeled),
        "n_identities_covered": n_covered,
        "n_identities_gt": n_gt_identities,
        "surplus_split": surplus_split,
        "merge_overlap": merge_overlap,
        "n_appearances": len(appearances),
        "frames_total": frames_total,
        "frames_unlabeled": frames_unlabeled,
        "n_local_tracks": len(local_tracks),
        "n_local_tracks_labeled": len(local_labeled),
    }


def fragmentation(appearances: list[Appearance], *, top: int = 10) -> dict:
    """Mỗi danh tính bị xé thành mấy Global ID. Mục 2."""
    gids_per_identity: dict[int, set[int]] = defaultdict(set)
    frames_per_identity: dict[int, int] = defaultdict(int)
    cams_per_identity: dict[int, set[str]] = defaultdict(set)
    for a in appearances:
        if a.gt_id is None:
            continue
        gids_per_identity[a.gt_id].add(a.global_id)
        frames_per_identity[a.gt_id] += a.n_frames
        cams_per_identity[a.gt_id].add(a.cam_id)

    n_ident = len(gids_per_identity)
    splits = [float(len(g)) for g in gids_per_identity.values()]
    hist = Counter(len(g) for g in gids_per_identity.values())
    # Một người thấy ở k camera vẫn PHẢI là 1 Global ID; số camera chỉ để biết ca nào khó.
    multi_cam = [i for i, cams in cams_per_identity.items() if len(cams) >= 2]

    print(f"\n{'=' * 78}\n2. VỠ: MỘT DANH TÍNH → MẤY GLOBAL ID\n{'=' * 78}")
    print(f"danh tính phủ được                      {n_ident}")
    print(f"  trong đó thấy ở >= 2 camera           {len(multi_cam)}")
    print(f"trung bình Global ID / danh tính        {np.mean(splits) if splits else 0:.2f}")
    print("\nsố Global ID   số danh tính   % tích luỹ")
    running = 0
    for k in sorted(hist):
        running += hist[k]
        mark = "   <- liên kết ĐÚNG" if k == 1 else ""
        pct = 100.0 * running / max(n_ident, 1)
        print(f"{k:>12d}   {hist[k]:>12d}   {pct:>9.1f}%{mark}")
    _quantile_header()
    _print_quantile_row("Global ID / danh tính", splits, "{:8.1f}")

    worst = sorted(gids_per_identity.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top]
    print(f"\n{len(worst)} danh tính vỡ nặng nhất:")
    print(f"{'danh tính':>10s}{'#gid':>7s}{'#khung':>9s}{'#cam':>6s}")
    for ident, gids in worst:
        print(
            f"{ident:>10d}{len(gids):>7d}{frames_per_identity[ident]:>9d}"
            f"{len(cams_per_identity[ident]):>6d}"
        )

    return {
        "n_identities": n_ident,
        "n_identities_multi_cam": len(multi_cam),
        "mean_gids_per_identity": float(np.mean(splits)) if splits else 0.0,
        "gids_per_identity_hist": {str(k): v for k, v in sorted(hist.items())},
        "gids_per_identity_quantiles": _quantiles(splits),
        "worst": [
            {
                "gt_id": ident,
                "n_gids": len(gids),
                "n_frames": frames_per_identity[ident],
                "n_cams": len(cams_per_identity[ident]),
            }
            for ident, gids in worst
        ],
    }


def surplus_mass(appearances: list[Appearance]) -> dict:
    """Mảnh dư to hay là bụi. Mục 3.

    Với mỗi danh tính, mảnh CHÍNH là Global ID giữ nhiều khung nhất CỦA danh tính đó; các
    Global ID còn lại là mảnh dư. Câu hỏi cần trả lời không phải "có bao nhiêu mảnh dư" —
    mục 1 đã đếm — mà **bao nhiêu phần trăm số khung của người đó rơi ra ngoài mảnh
    chính**. Đó chính là phần mà AssA mất: một mảnh dư 3 khung gần như không đáng gì, còn
    một người bị chia đôi 50/50 thì mất nửa số cặp liên kết đúng.
    """
    frames = _frames_by_identity_gid(appearances)
    tracklets: dict[tuple[int, int], int] = defaultdict(int)
    for a in appearances:
        if a.gt_id is not None:
            tracklets[(a.gt_id, a.global_id)] += 1

    by_identity: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (ident, gid), n in frames.items():
        by_identity[ident].append((gid, n))

    surplus_frames: list[float] = []
    surplus_tracklets: list[float] = []
    main_share: list[float] = []
    total_frames = 0
    lost_frames = 0
    for ident, items in by_identity.items():
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        total = sum(n for _, n in items)
        total_frames += total
        lost_frames += total - items[0][1]
        main_share.append(100.0 * items[0][1] / max(total, 1))
        for gid, n in items[1:]:
            surplus_frames.append(float(n))
            surplus_tracklets.append(float(tracklets[(ident, gid)]))

    singleton = sum(1 for v in surplus_tracklets if v == 1)

    print(f"\n{'=' * 78}\n3. MẢNH DƯ TO CỠ NÀO\n{'=' * 78}")
    print(f"mảnh dư (cặp danh tính × Global ID)     {len(surplus_frames)}")
    print(
        f"  chỉ gồm 1 tracklet                    {singleton}"
        f"  ({100.0 * singleton / max(len(surplus_frames), 1):.1f}%)"
    )
    print(
        f"khung nằm ngoài mảnh chính              {lost_frames}/{total_frames}"
        f"  ({100.0 * lost_frames / max(total_frames, 1):.1f}%)  <- phần AssA mất vì vỡ"
    )
    _quantile_header()
    _print_quantile_row("khung / mảnh dư", surplus_frames, "{:8.0f}")
    _print_quantile_row("tracklet / mảnh dư", surplus_tracklets, "{:8.1f}")
    _print_quantile_row("% khung ở mảnh chính", main_share, "{:8.1f}")

    return {
        "n_surplus_fragments": len(surplus_frames),
        "n_surplus_single_tracklet": singleton,
        "frames_outside_main": lost_frames,
        "frames_labeled_total": total_frames,
        "share_outside_main_pct": 100.0 * lost_frames / max(total_frames, 1),
        "surplus_frames_quantiles": _quantiles(surplus_frames),
        "surplus_tracklets_quantiles": _quantiles(surplus_tracklets),
        "main_share_pct_quantiles": _quantiles(main_share),
    }


def merges(appearances: list[Appearance], *, top: int = 10) -> dict:
    """Một Global ID ôm mấy danh tính. Mục 4 — đối trọng của mục 2.

    Vỡ và gộp là hai hướng hỏng ngược nhau và một ngưỡng chỉ đánh đổi giữa chúng. Báo cáo
    số mảnh dư mà không báo cáo số ca gộp thì không ai kiểm được lập luận "nghiêng về
    tách" của phiên 18.
    """
    frames: dict[tuple[int, int], int] = defaultdict(int)  # (gid, danh tính) -> khung
    for a in appearances:
        if a.gt_id is not None:
            frames[(a.global_id, a.gt_id)] += a.n_frames

    by_gid: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (gid, ident), n in frames.items():
        by_gid[gid].append((ident, n))

    impure = {gid: items for gid, items in by_gid.items() if len(items) >= 2}
    contaminated = 0
    total = 0
    for items in by_gid.values():
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        total += sum(n for _, n in items)
        contaminated += sum(n for _, n in items[1:])

    hist = Counter(len(items) for items in by_gid.values())

    print(f"\n{'=' * 78}\n4. GỘP: MỘT GLOBAL ID → MẤY DANH TÍNH\n{'=' * 78}")
    print(f"Global ID có nhãn                       {len(by_gid)}")
    print(
        f"  ôm >= 2 danh tính                     {len(impure)}"
        f"  ({100.0 * len(impure) / max(len(by_gid), 1):.1f}%)"
    )
    print(
        f"khung thuộc danh tính THIỂU SỐ          {contaminated}/{total}"
        f"  ({100.0 * contaminated / max(total, 1):.1f}%)  <- phần AssA mất vì gộp"
    )
    print("\nsố danh tính   số Global ID")
    for k in sorted(hist):
        mark = "   <- thuần khiết" if k == 1 else ""
        print(f"{k:>12d}   {hist[k]:>12d}{mark}")

    worst = sorted(impure.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top]
    if worst:
        print(f"\n{len(worst)} Global ID lẫn nặng nhất:")
        print(f"{'global_id':>10s}{'#danh tính':>12s}{'#khung':>9s}")
        for gid, items in worst:
            print(f"{gid:>10d}{len(items):>12d}{sum(n for _, n in items):>9d}")

    return {
        "n_labeled_gids": len(by_gid),
        "n_impure_gids": len(impure),
        "frames_minority": contaminated,
        "frames_labeled_total": total,
        "share_minority_pct": 100.0 * contaminated / max(total, 1),
        "identities_per_gid_hist": {str(k): v for k, v in sorted(hist.items())},
    }


def birth_rows(appearances: list[Appearance]) -> dict[int, Appearance]:
    """Global ID -> dòng `appearances` đã KHAI SINH ra nó.

    Chỉ tracklet nhận Global ID MỚI mới có `reason`; tracklet được ghép vào track có sẵn
    để trống, và `ON CONFLICT` trong `store.flush()` không đụng tới cột đó. Nên dòng khai
    sinh là dòng có `reason` khác rỗng — KHÔNG phải dòng sớm nhất theo `start_ms`: một
    tracklet bắt đầu sớm nhưng đóng muộn có thể được ghép vào một Global ID đã tồn tại, và
    lấy theo thời gian bắt đầu sẽ đọc nhầm nó thành người khai sinh.

    Vẫn còn một chỗ mờ không khử được từ phía bảng: `global_id` thì `ON CONFLICT` CÓ cập
    nhật, nên một tracklet từng tạo ra Global ID khác rồi bị gán lại sẽ mang `reason` của
    nó sang track mới. Khi một Global ID có nhiều dòng mang `reason`, lấy dòng sớm nhất.
    Global ID không còn dòng nào mang `reason` (người khai sinh đã bị gán đi nơi khác) rơi
    vào nhóm `""` và được đếm riêng chứ không nhét bừa vào một loại.
    """
    out: dict[int, Appearance] = {}
    for a in appearances:  # đã ORDER BY start_ms
        prev = out.get(a.global_id)
        if prev is None or (reason_kind(prev.reason) == "" and reason_kind(a.reason) != ""):
            out[a.global_id] = a
    return out


def birth_reasons(appearances: list[Appearance]) -> dict:
    """Global ID ra đời vì lý do gì, tách theo mảnh chính / mảnh dư / rác. Mục 5.

    Đây là chỗ biến phép đếm thành việc phải làm: nếu mảnh dư chủ yếu sinh ra vì
    `threshold` thì nút thắt là ngoại hình, còn nếu vì `no_candidate` thì là ràng buộc
    không–thời gian. Hai kết luận đó dẫn tới hai phiên làm việc hoàn toàn khác nhau.

    Đếm theo GLOBAL ID (một Global ID vào đúng một nhóm), không theo cặp danh tính × Global
    ID như mục 3 — nên tổng ở đây nhỏ hơn số mảnh dư của mục 3, vì một Global ID có thể là
    mảnh dư của nhiều người cùng lúc.
    """
    first = birth_rows(appearances)
    main = main_fragment_ids(appearances)
    labeled_gids = {a.global_id for a in appearances if a.gt_id is not None}
    groups: dict[str, list[Appearance]] = {
        "mảnh chính": [a for gid, a in first.items() if gid in main],
        "mảnh dư": [a for gid, a in first.items() if gid in labeled_gids and gid not in main],
        "rác": [a for gid, a in first.items() if gid not in labeled_gids],
    }

    print(f"\n{'=' * 78}\n5. GLOBAL ID RA ĐỜI VÌ LÝ DO GÌ (đếm theo Global ID)\n{'=' * 78}")
    kinds = sorted({reason_kind(a.reason) for a in first.values()})
    print(f"{'lý do':>14s}" + "".join(f"{name:>14s}" for name in GROUP_NAMES) + "   nghĩa là")
    out: dict[str, dict[str, int]] = {}
    for kind in kinds:
        counts = {
            name: sum(1 for a in aps if reason_kind(a.reason) == kind)
            for name, aps in groups.items()
        }
        out[kind] = counts
        print(
            f"{kind or '(trống)':>14s}"
            + "".join(f"{counts[name]:>14d}" for name in GROUP_NAMES)
            + f"   {REASON_LABELS.get(kind, '?')}"
        )
    print(f"{'TỔNG':>14s}" + "".join(f"{len(groups[name]):>14d}" for name in GROUP_NAMES))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--db", type=Path, required=True, help="SQLite store của lần chạy cần soi")
    p.add_argument("--gt", type=Path, required=True, help="bảng .gt.json của fixture tương ứng")
    p.add_argument(
        "--gt-identities",
        type=int,
        default=None,
        help=(
            "số danh tính trong chú thích gốc (WildTrack: 313). Bảng .gt.json thường ít hơn "
            "vì track không đủ thuần khiết đã bị loại; đưa số gốc vào để đọc tỉ lệ phủ"
        ),
    )
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--json", type=Path, default=None, help="ghi toàn bộ số liệu ra file JSON")
    args = p.parse_args(argv)

    gt = load_gt(args.gt)
    appearances = load_appearances(args.db, gt)
    n_gt_identities = args.gt_identities or len(set(gt.values()))

    print(
        f"{args.db.name}: {len(appearances)} lượt xuất hiện | "
        f"{args.gt.name}: {len(gt)} tracklet GT, {len(set(gt.values()))} danh tính"
    )
    if not appearances:
        print("Bảng appearances rỗng — không có gì để phân rã.")
        return 1

    report = {
        "db": str(args.db),
        "gt": str(args.gt),
        "decomposition": decompose(appearances, n_gt_identities=n_gt_identities),
        "fragmentation": fragmentation(appearances, top=args.top),
        "surplus_mass": surplus_mass(appearances),
        "merges": merges(appearances, top=args.top),
        "birth_reasons": birth_reasons(appearances),
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nsố liệu đầy đủ: {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
