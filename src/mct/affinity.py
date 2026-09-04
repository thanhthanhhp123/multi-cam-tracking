"""Ma trận chi phí giữa tracklet và GlobalTrack (CLAUDE.md §6 bước 2–3).

Chi phí cơ bản là `1 − cosine_similarity` của đặc trưng ngoại hình. Trước khi tính, mọi
cặp bất khả thi bị đánh `inf`:

  - **ràng buộc loại trừ cùng camera** — một người không thể đồng thời là hai local track
    khác nhau trong cùng một camera;
  - **ràng buộc thời gian di chuyển** — `topology.check()` với `Δt` = lúc tracklet bắt đầu
    trừ lúc GlobalTrack được thấy lần cuối;
  - **ngoại hình không so được** — tracklet chưa có embedding (pipeline đang chạy ReID tắt),
    hoặc GlobalTrack chưa có ngoại hình nào.

Với cặp camera **chồng lấn**, cộng thêm `λ · d_ground`: khoảng cách giữa hai điểm chân sau
khi ánh xạ homography về mặt phẳng tham chiếu chung, và loại thẳng nếu vượt
`max_ground_dist_m`. Phần hình học đó không nằm ở đây: module này chỉ nhận một
`GroundMapper` (giao thức ở dưới) và gọi nó. `homography.py` sẽ hiện thực giao thức đó;
chưa có thì để `None` và thành phần hình học bị bỏ qua — đúng tình trạng hiện tại khi
chưa có cặp camera nào được hiệu chỉnh.

Vì sao tách khỏi `associator.py`: ma trận chi phí là thứ cần *nhìn thấy* khi tinh chỉnh
tham số (tuần 16–17) — mỗi ô bị loại đều kèm lý do đọc được, thay vì chỉ thấy kết quả gán
cuối cùng rồi đoán ngược.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mct.gallery import GlobalTrack, SimilarityMode
from mct.topology import Topology
from mct.tracklet import Tracklet

INFEASIBLE = float("inf")


@runtime_checkable
class GroundMapper(Protocol):
    """Ánh xạ điểm chân trong ảnh về mặt phẳng tham chiếu chung (do homography.py cấp).

    Trả `None` khi cặp camera này chưa được hiệu chỉnh — khi đó thành phần hình học
    được bỏ qua thay vì đoán bừa một khoảng cách.
    """

    def project(self, cam_id: str, point: tuple[float, float]) -> tuple[float, float] | None:
        """Điểm chân trong ảnh → toạ độ mét trên mặt phẳng chung. `None` = chưa hiệu chỉnh."""
        ...

    def distance_m(
        self,
        cam_a: str,
        point_a: tuple[float, float],
        cam_b: str,
        point_b: tuple[float, float],
    ) -> float | None: ...


@dataclass(slots=True)
class AffinityConfig:
    """Tham số tính chi phí. Nguồn: khối `association` + `gallery` trong configs/mct.yaml."""

    max_cost: float = 0.30
    """Ngưỡng chấp nhận một cặp gán. Đây là tham số nhạy nhất của cả hệ thống."""

    homography_weight: float = 0.4
    """λ — trọng số của khoảng cách mặt đất, chỉ áp dụng cho cặp camera chồng lấn."""

    max_ground_dist_m: float = 3.0
    """Vượt khoảng cách này trên mặt phẳng tham chiếu thì loại thẳng, không xét ngoại hình."""

    exclusion_window_ms: int = 2000
    """GlobalTrack thấy ở camera `c` trong khoảng này thì không nhận tracklet KHÁC của `c`.

    Lấy bằng `tracklet.idle_timeout_ms`: đó chính là ngưỡng mà bên gom tracklet coi một
    track cục bộ là đã kết thúc, nên hai chỗ hiểu "còn đang hiện diện" giống nhau.
    """

    ground_time_tol_ms: int = 400
    """Hai quan sát ở hai camera lệch nhau dưới ngưỡng này thì coi là CÙNG thời điểm.

    Đặt theo độ lệch đồng hồ + chu kỳ khung: camera 15 fps lệch NTP ~100 ms thì 400 ms
    là rộng rãi. Rộng quá thì người đi bộ đã dịch chuyển đáng kể giữa hai mẫu được ghép.
    """

    ground_gap_policy: str = "allow"
    """Cặp camera CHỒNG LẤN mà hai bên không có mốc thời gian chung thì xử lý thế nào.

      allow  — thả qua, để ngoại hình quyết định (mặc định: an toàn, không bịa ràng buộc).
      reject — loại thẳng. Lập luận: hai camera nhìn chung một vùng thì cùng một người
               phải được cả hai thấy vào cùng lúc; không có mốc chung nghĩa là không có
               bằng chứng hình học nào, mà ngoại hình xuyên camera thì đã đo được là yếu.
               Đổi lấy precision bằng recall — phải đo trước khi bật (xem worklog M4).
    """

    max_speed_m_s: float = 2.5
    """Tốc độ đi bộ nhanh nhất còn hợp lý, dùng cho cặp KHÔNG trùng thời gian.

    Khi hai bên không có mốc thời gian chung, khoảng cách mặt đất vẫn nói lên điều gì đó
    nhưng phải cộng thêm quãng người ta kịp đi trong `Δt`. Ngân sách =
    `max_ground_dist_m + max_speed_m_s · Δt`.
    """

    similarity_mode: SimilarityMode = "max"
    topk_query: int = 8

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> AffinityConfig:
        association = dict(data.get("association", data) or {})
        gallery = dict(data.get("gallery", {}) or {})
        tracklet = dict(data.get("tracklet", {}) or {})
        defaults = cls()
        return cls(
            max_cost=float(association.get("max_cost", defaults.max_cost)),
            homography_weight=float(
                association.get("homography_weight", defaults.homography_weight)
            ),
            max_ground_dist_m=float(
                association.get("max_ground_dist_m", defaults.max_ground_dist_m)
            ),
            ground_time_tol_ms=int(
                association.get("ground_time_tol_ms", defaults.ground_time_tol_ms)
            ),
            max_speed_m_s=float(association.get("max_speed_m_s", defaults.max_speed_m_s)),
            ground_gap_policy=str(association.get("ground_gap_policy", defaults.ground_gap_policy)),
            exclusion_window_ms=int(tracklet.get("idle_timeout_ms", defaults.exclusion_window_ms)),
            similarity_mode=str(  # type: ignore[arg-type]
                gallery.get("similarity_mode", defaults.similarity_mode)
            ),
            topk_query=int(gallery.get("topk_query", defaults.topk_query)),
        )

    def __post_init__(self) -> None:
        if not 0.0 < self.max_cost <= 2.0:
            raise ValueError(f"max_cost phải nằm trong (0, 2], nhận {self.max_cost}")
        if self.homography_weight < 0.0:
            raise ValueError(f"homography_weight phải >= 0, nhận {self.homography_weight}")
        if self.max_ground_dist_m <= 0.0:
            raise ValueError(f"max_ground_dist_m phải > 0, nhận {self.max_ground_dist_m}")
        if self.ground_time_tol_ms < 0:
            raise ValueError(f"ground_time_tol_ms phải >= 0, nhận {self.ground_time_tol_ms}")
        if self.max_speed_m_s <= 0.0:
            raise ValueError(f"max_speed_m_s phải > 0, nhận {self.max_speed_m_s}")
        if self.ground_gap_policy not in ("allow", "reject"):
            raise ValueError(
                f"ground_gap_policy phải là 'allow' hoặc 'reject', nhận {self.ground_gap_policy!r}"
            )


@dataclass(slots=True)
class CostMatrix:
    """Ma trận chi phí đã mask, kèm lý do loại của từng ô bất khả thi."""

    tracklets: list[Tracklet]
    tracks: list[GlobalTrack]
    costs: np.ndarray
    """(n_tracklet, n_track) float64; `inf` = không được phép gán."""

    reasons: dict[tuple[int, int], str] = field(default_factory=dict)
    """(hàng, cột) → vì sao ô đó bị loại. Chỉ lưu ô bị loại, không lưu ô hợp lệ."""

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.tracklets), len(self.tracks))

    def reason(self, row: int, col: int) -> str:
        """Vì sao ô này bị loại (chuỗi rỗng nếu nó hợp lệ)."""
        return self.reasons.get((row, col), "")

    def feasible_pairs(self) -> list[tuple[int, int, float]]:
        """Các ô còn khả thi, sắp theo chi phí tăng dần — tiện để soi bằng mắt."""
        rows, cols = np.where(np.isfinite(self.costs))
        pairs = [(int(r), int(c), float(self.costs[r, c])) for r, c in zip(rows, cols, strict=True)]
        return sorted(pairs, key=lambda item: (item[2], item[0], item[1]))

    def explain(self, row: int, col: int) -> str:
        """Một dòng mô tả ô — dùng khi tìm hiểu vì sao một match đúng lại trượt."""
        tracklet, track = self.tracklets[row], self.tracks[col]
        cost = self.costs[row, col]
        head = (
            f"{tracklet.cam_id}/{tracklet.local_track_id} (tracklet #{tracklet.tracklet_id}) "
            f"↔ GlobalTrack #{track.global_id}"
        )
        if not np.isfinite(cost):
            return f"{head}: LOẠI — {self.reason(row, col)}"
        return f"{head}: cost={cost:.4f}"


def build_cost_matrix(
    tracklets: Sequence[Tracklet],
    tracks: Sequence[GlobalTrack],
    *,
    topology: Topology | None = None,
    config: AffinityConfig | None = None,
    ground_mapper: GroundMapper | None = None,
) -> CostMatrix:
    """Chi phí gán từng tracklet vào từng GlobalTrack.

    `topology=None` thì bỏ qua ràng buộc thời gian di chuyển (chỉ còn ngoại hình) — chỉ
    dùng khi cố tình muốn đo phần đóng góp riêng của đặc trưng Re-ID.
    """
    config = config or AffinityConfig()
    tracklets = list(tracklets)
    tracks = list(tracks)
    costs = np.full((len(tracklets), len(tracks)), INFEASIBLE, dtype=np.float64)
    reasons: dict[tuple[int, int], str] = {}
    # Chiếu quỹ đạo về mặt phẳng chung là phần đắt nhất của vòng lặp và lặp lại y hệt cho
    # mọi cột — nhớ lại trong đúng một lần dựng ma trận.
    cache: dict[tuple[str, int], np.ndarray] = {}

    for i, tracklet in enumerate(tracklets):
        query = tracklet.query_embedding(config.topk_query)
        for j, track in enumerate(tracks):
            cost, reason = _pair_cost(
                tracklet, query, track, topology, config, ground_mapper, cache
            )
            costs[i, j] = cost
            if reason:
                reasons[(i, j)] = reason

    return CostMatrix(tracklets=tracklets, tracks=tracks, costs=costs, reasons=reasons)


def _pair_cost(
    tracklet: Tracklet,
    query: np.ndarray | None,
    track: GlobalTrack,
    topology: Topology | None,
    config: AffinityConfig,
    ground_mapper: GroundMapper | None,
    cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> tuple[float, str]:
    if track.closed:
        return INFEASIBLE, f"GlobalTrack #{track.global_id} đã đóng"

    # Tracklet đang chạy và đã thuộc về track này: giữ nguyên, đừng để tracklet khác
    # giành mất chỉ vì tình cờ giống hơn ở cửa sổ hiện tại.
    if track.owns_tracklet(tracklet):
        return 0.0, ""

    if track.is_active_in(tracklet.cam_id, tracklet.end_ms, config.exclusion_window_ms):
        return INFEASIBLE, (
            f"GlobalTrack #{track.global_id} đang hiện diện ở chính {tracklet.cam_id} "
            "dưới một local track khác (ràng buộc loại trừ)"
        )

    if query is None:
        return INFEASIBLE, "tracklet chưa có embedding (ReID đang tắt?)"

    if topology is not None:
        elapsed = tracklet.start_ms - track.last_seen_ms
        verdict = topology.check(track.last_cam_id, tracklet.cam_id, elapsed)
        if not verdict.feasible:
            return INFEASIBLE, verdict.reason

    similarity = track.similarity(query, config.similarity_mode)
    if similarity < -1.0 + 1e-9:  # -1.0 = GlobalTrack chưa có ngoại hình nào
        return INFEASIBLE, f"GlobalTrack #{track.global_id} chưa có embedding để so"

    cost = 1.0 - float(similarity)

    if ground_mapper is not None and topology is not None:
        ground_cost, reason = _ground_term(tracklet, track, topology, config, ground_mapper, cache)
        if reason:
            return INFEASIBLE, reason
        cost += ground_cost

    return cost, ""


def _ground_term(
    tracklet: Tracklet,
    track: GlobalTrack,
    topology: Topology,
    config: AffinityConfig,
    ground_mapper: GroundMapper,
    cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> tuple[float, str]:
    """Thành phần hình học, CHỈ cho cặp camera chồng lấn.

    Với cặp không chồng lấn, hai người ở hai đầu hành lang cách nhau vài chục mét vẫn là
    cùng một người — khoảng cách mặt đất lúc đó vô nghĩa, chỉ ràng buộc thời gian mới nói
    lên điều gì.

    Hai đường tính, theo thứ tự ưu tiên:

    1. **Trùng thời gian** — GlobalTrack có quỹ đạo ở một camera chồng lấn với camera của
       tracklet, và hai quỹ đạo có mốc thời gian chung (lệch dưới `ground_time_tol_ms`).
       Lấy TRUNG VỊ khoảng cách trên các mốc chung. Đây là bằng chứng mạnh nhất mà hệ
       thống có: cùng một lúc, hai camera thấy người ở cùng một chỗ.
    2. **Không trùng thời gian** — quay về so điểm cuối của track với điểm đầu của
       tracklet, nhưng nới ngưỡng theo quãng đường người ta kịp đi trong `Δt`
       (`max_speed_m_s`). Không nới thì mọi cặp lệch vài giây đều bị loại oan.
    """
    cams = [
        cam_id
        for cam_id in track.cam_ground_path
        if cam_id != tracklet.cam_id and topology.is_overlapping(cam_id, tracklet.cam_id)
    ]
    if not topology.is_overlapping(track.last_cam_id, tracklet.cam_id) and not cams:
        return 0.0, ""

    own = _world_path(tracklet.cam_id, tracklet.ground_path, ground_mapper, cache)
    matched: list[float] = []
    for cam_id in cams:
        other = _world_path(cam_id, track.cam_ground_path[cam_id], ground_mapper, cache)
        distance = _synchronized_distance(own, other, config.ground_time_tol_ms)
        if distance is not None:
            matched.append(distance)

    if matched:
        distance = min(matched)
        if distance > config.max_ground_dist_m:
            return 0.0, (
                f"cách {distance:.2f} m trên mặt phẳng tham chiếu tại CÙNG thời điểm "
                f"> {config.max_ground_dist_m} m (cặp camera chồng lấn)"
            )
        return config.homography_weight * distance, ""

    if cams and config.ground_gap_policy == "reject" and len(own):
        return 0.0, (
            f"cặp camera chồng lấn ({', '.join(sorted(cams))} ↔ {tracklet.cam_id}) nhưng "
            "không có mốc thời gian chung để so vị trí (ground_gap_policy=reject)"
        )

    if not topology.is_overlapping(track.last_cam_id, tracklet.cam_id):
        return 0.0, ""

    # Không có mốc thời gian chung: so một điểm với một điểm, ngưỡng nới theo Δt.
    distance = ground_mapper.distance_m(
        track.last_cam_id,
        track.last_ground_point,
        tracklet.cam_id,
        tracklet.first_ground_point,
    )
    if distance is None:  # cặp chưa hiệu chỉnh homography
        return 0.0, ""

    elapsed_s = abs(tracklet.start_ms - track.last_seen_ms) / 1000.0
    budget = config.max_ground_dist_m + config.max_speed_m_s * elapsed_s
    if distance > budget:
        return 0.0, (
            f"cách {distance:.2f} m trên mặt phẳng tham chiếu > ngân sách {budget:.2f} m "
            f"(đi nhanh nhất {config.max_speed_m_s} m/s trong {elapsed_s:.1f}s)"
        )
    return config.homography_weight * min(distance, config.max_ground_dist_m), ""


def _world_path(
    cam_id: str,
    path: Sequence[tuple[int, tuple[float, float]]],
    ground_mapper: GroundMapper,
    cache: dict[tuple[str, int], np.ndarray] | None,
) -> np.ndarray:
    """Quỹ đạo (ts_ms, điểm ảnh) → mảng (n, 3) [ts_ms, X, Y] đã sắp theo thời gian.

    Điểm chiếu ra vô cực bị bỏ. Cache theo `id()` của list quỹ đạo — an toàn vì cache chỉ
    sống trong đúng một lần dựng ma trận, lúc đó không tracklet nào đang dài ra.
    """
    key = (cam_id, id(path))
    if cache is not None and key in cache:
        return cache[key]

    rows = [
        (float(ts), point[0], point[1])
        for ts, image_point in path
        if (point := ground_mapper.project(cam_id, image_point)) is not None
    ]
    world = np.array(sorted(rows), dtype=np.float64) if rows else np.empty((0, 3), dtype=np.float64)
    if cache is not None:
        cache[key] = world
    return world


def _synchronized_distance(a: np.ndarray, b: np.ndarray, tol_ms: int) -> float | None:
    """Trung vị khoảng cách giữa hai quỹ đạo tại các mốc thời gian khớp nhau.

    Trung vị chứ không phải trung bình: một khung có bbox bị che cho điểm chân lệch hàng
    mét, và với quỹ đạo vài chục điểm thì trung bình bị kéo lệch còn trung vị thì không.
    `None` khi không có mốc nào khớp (hai camera không cùng thấy người vào một lúc).
    """
    if len(a) == 0 or len(b) == 0:
        return None

    idx = np.searchsorted(b[:, 0], a[:, 0])
    left = np.clip(idx - 1, 0, len(b) - 1)
    right = np.clip(idx, 0, len(b) - 1)
    d_left = np.abs(b[left, 0] - a[:, 0])
    d_right = np.abs(b[right, 0] - a[:, 0])
    nearest = np.where(d_left <= d_right, left, right)
    gap = np.minimum(d_left, d_right)

    hit = gap <= tol_ms
    if not hit.any():
        return None
    delta = a[hit, 1:] - b[nearest[hit], 1:]
    return float(np.median(np.hypot(delta[:, 0], delta[:, 1])))


def costs_for_hungarian(matrix: np.ndarray, max_cost: float) -> np.ndarray:
    """Thay `inf` bằng một giá trị hữu hạn đủ lớn để `linear_sum_assignment` chạy được.

    `scipy.optimize.linear_sum_assignment` ném lỗi khi ma trận có `inf` mà không tồn tại
    phép gán hoàn chỉnh — chuyện xảy ra liên tục ở đây vì phần lớn ô bị ràng buộc loại bỏ.
    Cách chuẩn: đổi `inf` thành hằng số lớn hơn hẳn `max_cost`, chạy Hungarian, rồi bỏ mọi
    cặp có chi phí GỐC vượt ngưỡng. Giá trị thay thế không ảnh hưởng kết quả vì mọi cặp
    dùng tới nó đều bị loại ở bước sau.
    """
    blocked = max_cost * 10.0 + 1.0
    return np.where(np.isfinite(matrix), matrix, blocked)


def summarize(matrix: CostMatrix, *, limit: int = 10) -> Iterable[str]:
    """Vài dòng tóm tắt ma trận — để log ở mức DEBUG khi tinh chỉnh tham số."""
    n_rows, n_cols = matrix.shape
    feasible = matrix.feasible_pairs()
    yield (
        f"ma trận {n_rows}x{n_cols}: {len(feasible)}/{n_rows * n_cols} ô khả thi, "
        f"{len(matrix.reasons)} ô bị loại"
    )
    for row, col, _cost in feasible[:limit]:
        yield "  " + matrix.explain(row, col)
