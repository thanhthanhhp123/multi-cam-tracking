"""Gán Global ID cho tracklet bằng Hungarian trên ma trận chi phí (CLAUDE.md §6 bước 4–6).

Vòng gán của một cửa sổ:

1. Tracklet nào đang thuộc về một GlobalTrack rồi thì **cập nhật thẳng**, không đưa vào
   ma trận. Hungarian tối ưu TỔNG chi phí, nên nó sẵn sàng hy sinh một cặp chi phí 0 để
   đổi lấy hai cặp khác rẻ hơn một chút — tức là cướp Global ID của một tracklet đang
   chạy ngon lành. Tách ra trước là cách rẻ nhất để chuyện đó không xảy ra.
2. Phần còn lại: dựng ma trận chi phí (`affinity.build_cost_matrix`), chạy
   `scipy.optimize.linear_sum_assignment`.
3. Ô vượt `max_cost` bị chặn **trước** khi chạy Hungarian (`costs_for_hungarian`), không
   lọc sau. Lọc sau thì Hungarian còn thấy ô sẽ-bị-loại là "rẻ" so với ô chặn và sẵn sàng
   đẩy một hàng vào đó để hàng khác lấy ô rẻ hơn — mất một cặp hợp lệ. Hàng không ghép
   được ô nào **tạo Global ID mới**.
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
    GroundCache,
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

    revised: int = 0
    """Số lần xét lại quyết định đầu (`revise_after_frames`)."""

    revised_changed: int = 0
    """Trong đó, số lần Global ID của tracklet đổi sau khi xét lại."""


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
        # Cache quỹ đạo đã chiếu, DÙNG CHUNG qua mọi cửa sổ: quỹ đạo của một GlobalTrack
        # gần như không đổi giữa hai cửa sổ liên tiếp, nhưng bản trước tạo cache mới cho
        # mỗi ma trận nên chiếu lại từ đầu mỗi lần. Xem `affinity.GroundCache` về vì sao
        # khoá theo `id()` vẫn an toàn khi cache sống lâu.
        self._ground_cache: GroundCache = {}
        self._provisional: set[int] = set()
        """tracklet_id được gán khi còn ít khung hơn `revise_after_frames`, chờ xét lại."""

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
        revise: list[tuple[Tracklet, GlobalTrack]] = []
        for tracklet in tracklets:
            owner = self.gallery.find_by_tracklet(tracklet)
            if owner is not None and self._due_revision(tracklet):
                revise.append((tracklet, owner))
                continue
            if owner is None and tracklet.closed:
                # Tracklet đã có danh tính nhưng bị một tracklet mới hơn cùng camera thế chỗ
                # trong GlobalTrack đó (`Gallery.pop_former_owner`). Nó đóng mà không có khung
                # mới nào, nên không có gì để xét lại: giữ nguyên danh tính, KHÔNG nạp lại vào
                # gallery (nạp lại sẽ đẩy tracklet mới hơn ra khỏi vị trí "đang sở hữu").
                former = self.gallery.pop_former_owner(tracklet)
                if former is not None:
                    self.stats.updated += 1
                    results.append(
                        Assignment(
                            tracklet=tracklet,
                            global_id=former.global_id,
                            cost=0.0,
                            is_new=False,
                            is_update=True,
                        )
                    )
                    continue
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

        for tracklet, owner in revise:
            results.append(self._revise(tracklet, owner))

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

        # Ngưỡng theo từng ô chỉ khi `max_cost_geometric` được đặt. Khi đó Hungarian chạy
        # trên TỈ LỆ chi phí / ngưỡng của ô (ngân sách đã dùng) và chấp nhận khi < 1; không
        # đặt thì đi đúng đường cũ — giữ nguyên từng bit kết quả của mọi lần chạy trước.
        limits = matrix.limits(self.config)
        if limits is None:
            work, threshold = matrix.costs, self.config.max_cost
        else:
            work, threshold = matrix.costs / limits, 1.0

        if tracks:
            padded = costs_for_hungarian(work, threshold)
            rows, cols = linear_sum_assignment(padded)
            for row, col in zip(rows, cols, strict=True):
                # Ô vượt ngưỡng đã bị chặn trước khi ghép, nhưng Hungarian vẫn phải trả về
                # một cột cho mọi hàng — nên vẫn đối chiếu với chi phí GỐC ở đây.
                if float(work[row, col]) < threshold:
                    matched_rows[int(row)] = (int(col), float(matrix.costs[row, col]))

        for row, tracklet in enumerate(tracklets):
            hit = matched_rows.get(row)
            if hit is not None:
                col, cost = hit
                track = tracks[col]
                self.gallery.assign(track, tracklet)
                self.stats.matched += 1
                self._mark_provisional(tracklet)
                results.append(
                    Assignment(
                        tracklet=tracklet, global_id=track.global_id, cost=cost, is_new=False
                    )
                )
                continue

            kind, reason = _why_new(matrix, row, self.config.max_cost, limits)
            if kind == "no_candidate":
                self.stats.no_candidate += 1
            elif kind == "threshold":
                self.stats.rejected_by_threshold += 1
            track = self.gallery.create(tracklet)
            self.stats.created += 1
            self._mark_provisional(tracklet)
            results.append(
                Assignment(
                    tracklet=tracklet,
                    global_id=track.global_id,
                    cost=float("inf"),
                    is_new=True,
                    reason=f"{kind}: {reason}",
                )
            )
        return results

    # ------------------------------------------------------------------ xét lại

    def _mark_provisional(self, tracklet: Tracklet) -> None:
        limit = self.config.revise_after_frames
        if limit is not None and not tracklet.closed and tracklet.n_frames < limit:
            self._provisional.add(tracklet.tracklet_id)

    def _due_revision(self, tracklet: Tracklet) -> bool:
        limit = self.config.revise_after_frames
        if limit is None or tracklet.tracklet_id not in self._provisional:
            return False
        return tracklet.closed or tracklet.n_frames >= limit

    def _revise(self, tracklet: Tracklet, owner: GlobalTrack) -> Assignment:
        """Gỡ tracklet khỏi GlobalTrack đang giữ nó rồi gán lại với query đầy đủ hơn.

        Chỉ một lần cho mỗi tracklet. Một hàng duy nhất nên Hungarian suy biến thành chọn ô
        rẻ nhất (theo chi phí / ngưỡng của ô, như `_match_one_camera`). Ba kết cục:
        giữ nguyên track cũ; chuyển sang track khác (track cũ rỗng thì bỏ ID đó đi); hoặc
        tách ra Global ID mới (nếu track cũ rỗng thì giữ luôn ID cũ thay vì cấp số mới).
        """
        self._provisional.discard(tracklet.tracklet_id)
        self.stats.revised += 1
        self.gallery.detach(owner, tracklet)
        orphan = not owner.members
        tracks = [t for t in self.gallery.open_tracks() if not (orphan and t is owner)]

        target: GlobalTrack | None = None
        cost = float("inf")
        if tracks:
            matrix = self.cost_matrix([tracklet], tracks)
            limits = matrix.limits(self.config)
            if limits is None:
                work, threshold = matrix.costs, self.config.max_cost
            else:
                work, threshold = matrix.costs / limits, 1.0
            col = int(np.argmin(work[0]))
            if float(work[0, col]) < threshold:
                target, cost = tracks[col], float(matrix.costs[0, col])

        if target is None:
            target = owner if orphan else self.gallery.create(tracklet)
            if target is owner:
                self.gallery.assign(owner, tracklet)
            is_new = target is not owner
        else:
            self.gallery.assign(target, tracklet)
            if orphan and target is not owner:
                self.gallery.drop(owner)
            is_new = False
        if target is not owner:
            self.stats.revised_changed += 1
            log.debug(
                "xét lại tracklet #%d: Global ID %d -> %d",
                tracklet.tracklet_id,
                owner.global_id,
                target.global_id,
            )
        return Assignment(
            tracklet=tracklet,
            global_id=target.global_id,
            cost=cost if not is_new else float("inf"),
            is_new=is_new,
            # Không phải lần phát đầu (Global ID đã cấp từ trước), nên `latency_report` không
            # đếm nó là "first": thời gian tới Global ID vẫn tính ở lần gán tạm.
            is_update=True,
            reason="revised" if target is not owner else "",
        )

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
            cache=self._ground_cache,
        )

    def prune(self, now_ms: int) -> list[GlobalTrack]:
        """Đóng GlobalTrack quá hạn. Trả về chúng để `store.py` ghi xuống SQLite."""
        return self.gallery.prune(now_ms)

    # ------------------------------------------------------------------ tiện ích

    def global_id_of(self, tracklet: Tracklet) -> int | None:
        track = self.gallery.find_by_tracklet(tracklet)
        return None if track is None else track.global_id


NEW_TRACK_KINDS = ("empty", "no_candidate", "threshold", "taken")
"""Vì sao một tracklet phải nhận Global ID mới. Xem `_why_new`."""


def reason_kind(reason: str) -> str:
    """Lấy lại `loại` từ chuỗi `Assignment.reason` đã lưu xuống SQLite.

    `Assignment.reason` được ghép theo dạng `"<loại>: <mô tả>"` ngay tại chỗ gọi `_why_new`,
    nên hàm này chỉ cắt tiền tố chứ KHÔNG đoán nghĩa câu tiếng Việt phía sau — chính là điều
    docstring của `_why_new` cấm. Tiền tố lạ (hoặc chuỗi rỗng của tracklet được ghép vào
    track có sẵn) trả về `""`.
    """
    kind, sep, _ = reason.partition(": ")
    return kind if sep and kind in NEW_TRACK_KINDS else ""


def _why_new(
    matrix: CostMatrix, row: int, max_cost: float, limits: np.ndarray | None = None
) -> tuple[str, str]:
    """Vì sao tracklet này phải nhận Global ID mới: `(loại, mô tả)`.

    Trả về `loại` chứ không để bên gọi dò chuỗi tiếng Việt: thống kê `no_candidate` và
    `rejected_by_threshold` phân biệt hai tình huống khác hẳn nhau (ràng buộc không-thời
    gian loại sạch ứng viên, so với ngoại hình không đủ giống), và một lần đổi chữ trong
    câu mô tả không được phép làm hỏng bảng thống kê của chương 6.

    Chỗ gọi ghép `loại` vào đầu `mô tả` trước khi ghi xuống SQLite, vì bảng `appearances`
    chỉ có một cột `reason`: không có tiền tố đó thì hậu kiểm (`eval/diagnose_global_ids.py`)
    buộc phải dò chuỗi, đúng thứ vừa cấm.
    """
    if not matrix.tracks:
        return "empty", "gallery đang rỗng, đây là người đầu tiên"

    costs = matrix.costs[row]
    finite = costs[np.isfinite(costs)]
    if finite.size == 0:
        reasons = {matrix.reason(row, col) for col in range(len(matrix.tracks))}
        detail = "; ".join(sorted(r for r in reasons if r)[:3])
        return "no_candidate", f"không còn ứng viên khả thi sau ràng buộc ({detail})"

    if limits is not None:
        # Ngưỡng theo ô: "vượt ngưỡng" nghĩa là MỌI ô khả thi đều vượt ngưỡng của chính nó.
        ratio = np.where(np.isfinite(costs), costs / limits[row], np.inf)
        col = int(np.argmin(ratio))
        best, limit = float(costs[col]), float(limits[row, col])
        geo = matrix.geometric is not None and bool(matrix.geometric[row, col])
        if ratio[col] >= 1.0:
            return (
                "threshold",
                f"ứng viên tốt nhất có cost={best:.4f} >= ngưỡng của ô {limit} "
                f"({'có' if geo else 'không có'} bằng chứng vị trí)",
            )
        return "taken", f"ứng viên tốt nhất (cost={best:.4f}) đã bị tracklet khác lấy mất"

    best = float(finite.min())
    if best >= max_cost:
        return "threshold", f"ứng viên tốt nhất có cost={best:.4f} >= max_cost={max_cost}"
    # Còn ô rẻ hơn ngưỡng nhưng Hungarian đã trao nó cho tracklet khác của cùng camera.
    return "taken", f"ứng viên tốt nhất (cost={best:.4f}) đã bị tracklet khác lấy mất"


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
