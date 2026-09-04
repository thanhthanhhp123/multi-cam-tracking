"""Gán Global ID cho tracklet bằng Hungarian trên ma trận chi phí (CLAUDE.md §6 bước 4–6).

Vòng gán của một cửa sổ:

1. Tracklet nào đang thuộc về một GlobalTrack rồi thì **cập nhật thẳng**, không đưa vào
   ma trận. Hungarian tối ưu TỔNG chi phí, nên nó sẵn sàng hy sinh một cặp chi phí 0 để
   đổi lấy hai cặp khác rẻ hơn một chút — tức là cướp Global ID của một tracklet đang
   chạy ngon lành. Tách ra trước là cách rẻ nhất để chuyện đó không xảy ra.
2. Phần còn lại: dựng ma trận chi phí (`affinity.build_cost_matrix`), chạy
   `scipy.optimize.linear_sum_assignment`.
3. Cặp có chi phí `< max_cost` thì nhận; còn lại **tạo Global ID mới**. Ngưỡng đặt sau
   Hungarian chứ không phải trước: Hungarian cần thấy toàn bộ ma trận mới tối ưu đúng.
4. Tracklet không khớp được ai → Global ID mới.

**Online và offline khác nhau ở chỗ nào.** Không phải ở thuật toán — cùng đúng một hàm
`assign()`. Khác ở chỗ tracklet được đưa vào lúc nào: chế độ `online` đưa tracklet vào
ngay khi nó vừa động (còn dang dở, embedding mới gom được vài frame), chế độ `offline`
chỉ đưa vào khi tracklet đã đóng và có đủ toàn bộ đặc trưng. Chênh lệch kết quả giữa hai
chế độ chính là **cái giá phải trả của ràng buộc thời gian thực** — con số cần báo cáo ở
chương 6. Vòng lặp quyết định điều đó nằm ở `run_offline()` (dưới đây) và ở tầng chạy
online (`mct/__main__.py`, chưa viết).

**Khi ReID tắt** (`embed_dim=0`, đúng tình trạng pipeline M1/M2): mọi ô đều bất khả thi,
nên mỗi tracklet nhận một Global ID riêng ngay lần đầu, rồi các cửa sổ sau chỉ cập nhật
GlobalTrack đó. Hệ thống suy biến thành theo dõi trong từng camera — đúng như mong đợi,
không phải lỗi.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.logging import get_logger
from mct.affinity import (
    AffinityConfig,
    CostMatrix,
    GroundMapper,
    build_cost_matrix,
    costs_for_hungarian,
)
from mct.gallery import Gallery, GalleryConfig, GlobalTrack
from mct.topology import Topology
from mct.tracklet import Tracklet, TrackletConfig, build_tracklets

log = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class Assignment:
    """Kết quả gán một tracklet trong một vòng."""

    tracklet: Tracklet
    global_id: int
    cost: float
    is_new: bool
    """True = vừa tạo Global ID mới cho tracklet này."""

    is_update: bool = False
    """True = tracklet đã thuộc GlobalTrack này từ trước, vòng này chỉ cập nhật thêm."""

    reason: str = ""
    """Vì sao phải tạo ID mới (rỗng nếu khớp được)."""


@dataclass(slots=True)
class AssociatorStats:
    windows: int = 0
    matched: int = 0
    """Số lần khớp một tracklet vào GlobalTrack đã có (không tính cập nhật)."""

    updated: int = 0
    created: int = 0
    rejected_by_threshold: int = 0
    """Hungarian có ghép, nhưng chi phí vượt `max_cost` nên phải tạo ID mới."""

    no_candidate: int = 0
    """Không còn ứng viên khả thi nào sau khi áp ràng buộc."""


class Associator:
    """Gán Global ID. Giữ gallery, không giữ tracklet — tracklet do `TrackletBuilder` quản."""

    def __init__(
        self,
        *,
        topology: Topology | None = None,
        gallery: Gallery | None = None,
        config: AffinityConfig | None = None,
        ground_mapper: GroundMapper | None = None,
    ) -> None:
        self.topology = topology
        self.gallery = gallery or Gallery()
        self.config = config or AffinityConfig()
        self.ground_mapper = ground_mapper
        self.stats = AssociatorStats()

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        topology: Topology | None = None,
        ground_mapper: GroundMapper | None = None,
    ) -> Associator:
        """Dựng từ dict đã load của configs/mct.yaml."""
        return cls(
            topology=topology,
            gallery=Gallery(GalleryConfig.from_mapping(data)),
            config=AffinityConfig.from_mapping(data),
            ground_mapper=ground_mapper,
        )

    # ------------------------------------------------------------------ vòng gán

    def assign(self, tracklets: Sequence[Tracklet]) -> list[Assignment]:
        """Một vòng gán cho các tracklet vừa cập nhật/kết thúc."""
        self.stats.windows += 1
        results: list[Assignment] = []

        pending: list[Tracklet] = []
        for tracklet in tracklets:
            owner = self.gallery.find_by_tracklet(tracklet)
            if owner is None:
                pending.append(tracklet)
                continue
            self.gallery.assign(owner, tracklet)
            self.stats.updated += 1
            results.append(
                Assignment(
                    tracklet=tracklet,
                    global_id=owner.global_id,
                    cost=0.0,
                    is_new=False,
                    is_update=True,
                )
            )

        if pending:
            results.extend(self._match(pending))

        return sorted(results, key=lambda a: (a.tracklet.end_ms, a.tracklet.tracklet_id))

    def _match(self, tracklets: list[Tracklet]) -> list[Assignment]:
        """Ghép theo TỪNG CAMERA, không phải một lần Hungarian cho cả vòng.

        Phép ghép một-một chỉ đúng trong phạm vi một camera: hai local track khác nhau
        của cùng một camera không thể là một người (ràng buộc loại trừ). GIỮA các camera
        thì ngược lại — camera chồng lấn nhìn thấy đúng một người cùng lúc, nên một
        Global ID phải được phép nhận nhiều tracklet trong cùng một vòng. Gộp tất cả vào
        một ma trận là áp nhầm ràng buộc một-một lên cả chiều liên camera: người xuất hiện
        ở 7 camera thì 6 tracklet còn lại bị đẩy sang Global ID mới, và danh tính vỡ vụn
        ngay tại vòng đầu tiên. Đo trên WildTrack (7 camera chồng lấn): recall 0.06 khi
        gộp chung, xem worklog M4.

        Đổi lại, thứ tự xử lý camera ảnh hưởng tới kết quả (camera sau nhìn thấy gallery
        đã cập nhật bởi camera trước). Sắp theo `cam_id` để chạy lại là tất định.
        """
        by_cam: dict[str, list[Tracklet]] = {}
        for tracklet in tracklets:
            by_cam.setdefault(tracklet.cam_id, []).append(tracklet)

        results: list[Assignment] = []
        for cam_id in sorted(by_cam):
            results.extend(self._match_one_camera(by_cam[cam_id]))
        return results

    def _match_one_camera(self, tracklets: list[Tracklet]) -> list[Assignment]:
        tracks = self.gallery.open_tracks()
        matrix = self.cost_matrix(tracklets, tracks)
        results: list[Assignment] = []
        matched_rows: dict[int, tuple[int, float]] = {}

        if tracks:
            padded = costs_for_hungarian(matrix.costs, self.config.max_cost)
            rows, cols = linear_sum_assignment(padded)
            for row, col in zip(rows, cols, strict=True):
                cost = float(matrix.costs[row, col])
                if cost < self.config.max_cost:
                    matched_rows[int(row)] = (int(col), cost)
                elif np.isfinite(cost):
                    self.stats.rejected_by_threshold += 1

        for row, tracklet in enumerate(tracklets):
            hit = matched_rows.get(row)
            if hit is not None:
                col, cost = hit
                track = tracks[col]
                self.gallery.assign(track, tracklet)
                self.stats.matched += 1
                results.append(
                    Assignment(
                        tracklet=tracklet, global_id=track.global_id, cost=cost, is_new=False
                    )
                )
                continue

            reason = _why_new(matrix, row, self.config.max_cost)
            if "không còn ứng viên" in reason:
                self.stats.no_candidate += 1
            track = self.gallery.create(tracklet)
            self.stats.created += 1
            results.append(
                Assignment(
                    tracklet=tracklet,
                    global_id=track.global_id,
                    cost=float("inf"),
                    is_new=True,
                    reason=reason,
                )
            )
        return results

    def cost_matrix(
        self, tracklets: Sequence[Tracklet], tracks: Sequence[GlobalTrack] | None = None
    ) -> CostMatrix:
        """Ma trận chi phí của vòng hiện tại — tách ra để soi khi tinh chỉnh tham số."""
        return build_cost_matrix(
            tracklets,
            self.gallery.open_tracks() if tracks is None else tracks,
            topology=self.topology,
            config=self.config,
            ground_mapper=self.ground_mapper,
        )

    def prune(self, now_ms: int) -> list[GlobalTrack]:
        """Đóng GlobalTrack quá hạn. Trả về chúng để `store.py` ghi xuống SQLite."""
        return self.gallery.prune(now_ms)

    # ------------------------------------------------------------------ tiện ích

    def global_id_of(self, tracklet: Tracklet) -> int | None:
        track = self.gallery.find_by_tracklet(tracklet)
        return None if track is None else track.global_id


def _why_new(matrix: CostMatrix, row: int, max_cost: float) -> str:
    """Lý do một tracklet phải nhận Global ID mới — ghi vào log để truy được về sau."""
    if not matrix.tracks:
        return "gallery đang rỗng, đây là người đầu tiên"

    costs = matrix.costs[row]
    finite = costs[np.isfinite(costs)]
    if finite.size == 0:
        reasons = {matrix.reason(row, col) for col in range(len(matrix.tracks))}
        detail = "; ".join(sorted(r for r in reasons if r)[:3])
        return f"không còn ứng viên khả thi sau ràng buộc ({detail})"

    best = float(finite.min())
    return f"ứng viên tốt nhất có cost={best:.4f} >= max_cost={max_cost}"


def run_offline(
    tracklets: Iterable[Tracklet],
    *,
    topology: Topology | None = None,
    config: AffinityConfig | None = None,
    gallery_config: GalleryConfig | None = None,
    ground_mapper: GroundMapper | None = None,
    window_ms: int = 1000,
) -> tuple[list[Assignment], Associator]:
    """Chế độ `offline` — cận trên độ chính xác (CLAUDE.md §6).

    Tracklet đã đóng hết, nên mỗi cái vào vòng gán với TOÀN BỘ đặc trưng của nó, và thứ
    tự xử lý là thứ tự kết thúc thật. Vẫn chia theo cửa sổ `window_ms` để những tracklet
    kết thúc gần nhau được xét cùng một lần Hungarian — đó mới là chỗ Hungarian có ích;
    gán từng cái một sẽ thành tham lam và mất hết ý nghĩa của phép ghép tối ưu.
    """
    associator = Associator(
        topology=topology,
        gallery=Gallery(gallery_config) if gallery_config else Gallery(),
        config=config,
        ground_mapper=ground_mapper,
    )
    ordered = sorted(tracklets, key=lambda t: (t.end_ms, t.cam_id, t.local_track_id))
    results: list[Assignment] = []

    batch: list[Tracklet] = []
    window_end: int | None = None
    for tracklet in ordered:
        if window_end is not None and tracklet.end_ms > window_end:
            results.extend(associator.assign(batch))
            batch = []
        if not batch:
            window_end = tracklet.end_ms + window_ms
        batch.append(tracklet)
    if batch:
        results.extend(associator.assign(batch))

    return results, associator


def assign_messages(
    messages: Iterable[Any],
    *,
    topology: Topology | None = None,
    tracklet_config: TrackletConfig | None = None,
    config: AffinityConfig | None = None,
    gallery_config: GalleryConfig | None = None,
    ground_mapper: GroundMapper | None = None,
    window_ms: int = 1000,
) -> tuple[list[Assignment], Associator]:
    """Đường tắt: `FrameMessage` → tracklet → Global ID, chạy offline một lần.

    Dùng cho đánh giá trên fixture (`eval/`) và cho test. Đường chạy thật là online, đi
    qua Redis — xem docstring module.
    """
    tracklets = build_tracklets(messages, tracklet_config)
    return run_offline(
        tracklets,
        topology=topology,
        config=config,
        gallery_config=gallery_config,
        ground_mapper=ground_mapper,
        window_ms=window_ms,
    )
