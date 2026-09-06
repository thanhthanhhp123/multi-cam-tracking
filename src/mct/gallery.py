"""Gallery các GlobalTrack — bộ nhớ ngoại hình của engine liên kết (CLAUDE.md §6 bước 6).

Một `GlobalTrack` là một người, được nhận ra qua nhiều camera: nó gom các tracklet cục bộ
đã được gán cùng một Global ID, giữ lại một tập embedding đại diện và mốc thời gian
lần cuối thấy ở từng camera.

Ba quyết định gói trong module này:

1. **Giữ nhiều embedding, không chỉ một vector trung bình.** Cùng một người ở hai camera
   khác góc/ánh sáng cho hai vector lệch nhau đáng kể (fixture tổng hợp đặt mức này bằng
   `--cross-cam-sim`). Gộp tất cả vào một centroid làm nhoè đúng phần thông tin cần để
   khớp camera thứ ba. Nên gallery giữ tối đa `max_size` embedding (`similarity_mode: max`)
   và **song song** duy trì một centroid EMA (`similarity_mode: centroid`) — chọn được
   bằng config để tuần 16-17 sweep và báo cáo so sánh, không phải sửa code.

2. **Ưu tiên giữ embedding của các camera KHÁC NHAU khi gallery đầy.** Một người đứng lâu
   trước camera 1 sẽ đẩy hết embedding của camera 2 ra ngoài nếu chỉ xét confidence —
   đúng lúc cần chúng nhất. Nên hạn ngạch chia theo camera trước, trong mỗi camera mới
   xét confidence.

3. **GlobalTrack hết hạn thì đóng lại, không xoá.** Quá `global_track_ttl_ms` không thấy
   lại thì không còn là ứng viên (tránh gán nhầm người đã rời khỏi khu vực từ lâu), nhưng
   bản ghi vẫn trả về cho `store.py` ghi xuống SQLite phục vụ tra cứu hành trình.

Module này chỉ giữ trạng thái và tính similarity ngoại hình. Ràng buộc không-thời gian
(topology, homography) nằm ở `topology.py`/`affinity.py`, còn quyết định gán là của
`associator.py`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from common.schema import l2_normalize
from mct.tracklet import Tracklet

SimilarityMode = Literal["max", "centroid"]

# Số lượt xuất hiện gần nhất giữ trong RAM cho mỗi GlobalTrack. Hành trình đầy đủ nằm ở
# SQLite (store.py) — đây chỉ là phần đủ dùng cho ràng buộc gán ở cửa sổ hiện tại.
_MAX_MEMBERS = 32

# Số khoảng thời gian giữ cho mỗi camera trong `GlobalTrack.cam_spans` (ràng buộc loại trừ).
# Một người quay lại cùng một camera nhiều lần thì chỉ vài lượt gần nhất còn khả năng trùng
# thời gian với tracklet đang xét; giữ tất cả là rò rỉ bộ nhớ cho một phiên chạy dài.
_MAX_SPANS_PER_CAM = 8


@dataclass(slots=True, frozen=True)
class TrackletRef:
    """Một tracklet cục bộ đã được gán vào GlobalTrack."""

    tracklet_id: int
    cam_id: str
    local_track_id: int
    start_ms: int
    end_ms: int

    @classmethod
    def of(cls, tracklet: Tracklet) -> TrackletRef:
        return cls(
            tracklet_id=tracklet.tracklet_id,
            cam_id=tracklet.cam_id,
            local_track_id=tracklet.local_track_id,
            start_ms=tracklet.start_ms,
            end_ms=tracklet.end_ms,
        )


@dataclass(slots=True)
class GalleryConfig:
    """Tham số gallery. Nguồn: khối `gallery` + `association` trong configs/mct.yaml."""

    max_size: int = 32
    """Số embedding tối đa giữ cho mỗi GlobalTrack."""

    topk_query: int = 8
    """Số embedding của tracklet dùng để tính query embedding."""

    ema_alpha: float = 0.3
    """Trọng số cập nhật centroid EMA khi khớp thêm một tracklet."""

    similarity_mode: SimilarityMode = "max"
    """`max` = cosine lớn nhất với embedding đang giữ; `centroid` = cosine với vector EMA."""

    global_track_ttl_ms: int = 120_000
    """Không xuất hiện lại trong khoảng này thì đóng GlobalTrack, thôi làm ứng viên."""

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> GalleryConfig:
        gallery = dict(data.get("gallery", data) or {})
        association = dict(data.get("association", {}) or {})
        defaults = cls()
        mode = str(gallery.get("similarity_mode", defaults.similarity_mode))
        return cls(
            max_size=int(gallery.get("max_size", defaults.max_size)),
            topk_query=int(gallery.get("topk_query", defaults.topk_query)),
            ema_alpha=float(gallery.get("ema_alpha", defaults.ema_alpha)),
            similarity_mode=mode,  # type: ignore[arg-type]
            global_track_ttl_ms=int(
                association.get("global_track_ttl_ms", defaults.global_track_ttl_ms)
            ),
        )

    def __post_init__(self) -> None:
        if self.max_size < 1:
            raise ValueError(f"max_size phải >= 1, nhận {self.max_size}")
        if self.topk_query < 1:
            raise ValueError(f"topk_query phải >= 1, nhận {self.topk_query}")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha phải nằm trong (0, 1], nhận {self.ema_alpha}")
        if self.similarity_mode not in ("max", "centroid"):
            raise ValueError(
                f"similarity_mode phải là 'max' hoặc 'centroid', nhận {self.similarity_mode!r}"
            )
        if self.global_track_ttl_ms <= 0:
            raise ValueError(f"global_track_ttl_ms phải > 0, nhận {self.global_track_ttl_ms}")


# eq=False: so sánh mặc định của dataclass sẽ so cả ndarray và ném "truth value is
# ambiguous" ngay ở `list.remove()`. Ở đây entry chỉ cần phân biệt theo danh tính đối tượng.
@dataclass(slots=True, eq=False)
class GalleryEntry:
    """Một embedding trong gallery, kèm xuất xứ để áp hạn ngạch theo camera."""

    embedding: np.ndarray
    cam_id: str
    confidence: float
    ts_ms: int
    tracklet_id: int


@dataclass(slots=True)
class GlobalTrack:
    """Một danh tính xuyên camera. `global_id` là thứ dashboard và báo cáo nhìn thấy."""

    global_id: int
    created_ms: int
    last_seen_ms: int
    last_cam_id: str

    n_tracklets: int = 0
    closed: bool = False

    last_ground_point: tuple[float, float] = (0.0, 0.0)
    """Điểm chân (đáy-giữa bbox) ở lần thấy gần nhất, theo toạ độ ảnh của `last_cam_id`.

    Đầu vào của thành phần hình học trong `affinity.py` với cặp camera chồng lấn. Giữ ở
    toạ độ ảnh chứ không quy đổi sẵn: phép ánh xạ homography phụ thuộc CẶP camera, mà lúc
    ghi lại thì chưa biết sẽ so với camera nào.
    """

    cam_last_seen: dict[str, int] = field(default_factory=dict)
    """cam_id → ts_ms lần cuối thấy ở camera đó. Đầu vào của ràng buộc thời gian di chuyển."""

    cam_last_tracklet: dict[str, int] = field(default_factory=dict)
    """cam_id → tracklet_id gần nhất ở camera đó. Dùng cho ràng buộc loại trừ cùng camera."""

    cam_spans: dict[str, list[tuple[int, int, int]]] = field(default_factory=dict)
    """cam_id → [(start_ms, end_ms, tracklet_id)] của các tracklet gần đây ở camera đó.

    Ràng buộc loại trừ hỏi "hai tracklet có TRÙNG THỜI GIAN không", không phải "cái kia
    vừa mới xuất hiện gần đây không" — nên cần cả khoảng, không chỉ mốc cuối (xem
    `overlaps_in()`).

    Và phải giữ NHIỀU khoảng chứ không chỉ khoảng của tracklet mới nhất: một GlobalTrack
    có thể ôm vài tracklet nối tiếp nhau ở cùng một camera, mà thứ tự hấp thụ không nhất
    thiết theo thời gian. Chỉ nhớ cái cuối thì một tracklet đến muộn nhưng nằm ở khoảng
    SỚM hơn sẽ lọt qua và Global ID đó có hai hộp trong cùng một khung — TrackEval từ chối
    chấm cả chuỗi khi gặp chuyện này (đo 2026-09-06).
    """

    cam_ground_path: dict[str, list[tuple[int, tuple[float, float]]]] = field(default_factory=dict)
    """cam_id → quỹ đạo (ts_ms, điểm chân ảnh) của tracklet gần nhất ở camera đó.

    Vì sao cần cả quỹ đạo chứ không chỉ `last_ground_point`: với cặp camera chồng lấn,
    bằng chứng mạnh nhất là "hai camera thấy người ở cùng một chỗ TẠI CÙNG MỘT LÚC".
    So điểm cuối của track với điểm đầu của tracklet là so hai thời điểm khác nhau —
    người đi 2 m/s thì chỉ cần lệch 3 giây là sai 6 m, đủ để loại nhầm mọi cặp đúng.
    Giữ tham chiếu tới chính list của tracklet nên nó tự dài ra khi tracklet còn chạy;
    kích thước đã bị `TrackletConfig.ground_path_max_points` chặn.
    """

    members: deque[TrackletRef] = field(default_factory=lambda: deque(maxlen=_MAX_MEMBERS))
    entries: list[GalleryEntry] = field(default_factory=list)
    centroid: np.ndarray | None = None
    """Vector EMA. Cũng là thứ ghi xuống SQLite để khởi động lại không mất ngoại hình."""

    @property
    def cameras(self) -> set[str]:
        return set(self.cam_last_seen)

    def last_seen_in(self, cam_id: str) -> int | None:
        return self.cam_last_seen.get(cam_id)

    def overlaps_in(
        self,
        cam_id: str,
        start_ms: int,
        end_ms: int,
        slack_ms: int = 0,
        *,
        ignore_tracklet_id: int | None = None,
    ) -> bool:
        """Tracklet ở camera này có TRÙNG THỜI GIAN với `[start_ms, end_ms]` không.

        Đây là ràng buộc loại trừ (CLAUDE.md §6 bước 2), và mệnh đề của nó là *đồng thời*:
        một người không thể là hai local track khác nhau của cùng một camera **tại cùng
        một lúc**. Hai tracklet NỐI TIẾP nhau thì không mâu thuẫn gì — đó chính là hình
        dạng của một lần tracker đổi id giữa chừng, và ghép chúng lại là việc engine phải
        làm được.

        Bản trước hỏi "lần cuối thấy ở camera này có gần đây không" (`now − last ≤ window`,
        `window` lấy bằng `idle_timeout_ms`). Với `idle_timeout_ms` lớn — 30 s trên fixture
        WildTrack — điều đó cấm luôn việc nối lại mảnh vỡ ngay sau khi tracker cắt id, tức
        tự tay chặn đúng thứ AssA đang mất điểm. Đo 2026-09-06: 12/15 cặp mảnh nối tiếp
        cùng camera cùng người nằm trong cửa sổ đó (`docs/worklog/2026-09-06-17-*`).

        `slack_ms` nới hai đầu khoảng để chịu jitter mốc thời gian; 0 là mặc định và đúng
        về mặt ngữ nghĩa (chồng lấn thật sự mới là mâu thuẫn). `ignore_tracklet_id` để một
        tracklet đang được cập nhật không tự phủ quyết chính nó.
        """
        return any(
            tracklet_id != ignore_tracklet_id
            and start_ms <= end + slack_ms
            and end_ms >= start - slack_ms
            for start, end, tracklet_id in self.cam_spans.get(cam_id, ())
        )

    def owns_tracklet(self, tracklet: Tracklet) -> bool:
        """Tracklet này có phải chính tracklet đang gán ở camera đó không.

        Cần để ràng buộc loại trừ không tự chặn việc CẬP NHẬT một tracklet đang chạy:
        tracklet dài được gán lại nhiều lần qua nhiều cửa sổ.
        """
        return self.cam_last_tracklet.get(tracklet.cam_id) == tracklet.tracklet_id

    def similarity(self, query: np.ndarray, mode: SimilarityMode = "max") -> float:
        """Cosine similarity với GlobalTrack này. Trả -1.0 nếu chưa có ngoại hình nào.

        Query và mọi embedding lưu trong gallery đều đã L2-normalize (contract §5), nên
        tích vô hướng chính là cosine — không chuẩn hoá lại ở đây.
        """
        if mode == "centroid":
            if self.centroid is None:
                return -1.0
            return float(self.centroid @ query)
        if not self.entries:
            return -1.0
        return float(max(entry.embedding @ query for entry in self.entries))

    def __repr__(self) -> str:  # pragma: no cover - chỉ phục vụ debug
        state = "closed" if self.closed else "open"
        return (
            f"GlobalTrack(#{self.global_id} {state} cams={sorted(self.cameras)} "
            f"tracklets={self.n_tracklets} emb={len(self.entries)} last={self.last_seen_ms}ms)"
        )


class Gallery:
    """Tập GlobalTrack đang mở + vòng đời của chúng.

    Không tự quyết định gán — `associator.py` gọi `candidates()` để lấy danh sách ứng viên,
    tự tính chi phí, rồi gọi `assign()` hoặc `create()` theo kết quả Hungarian.
    """

    def __init__(self, config: GalleryConfig | None = None) -> None:
        self.config = config or GalleryConfig()
        self._tracks: dict[int, GlobalTrack] = {}
        self._owner_of: dict[int, int] = {}
        """tracklet_id → global_id đang giữ nó. Chỉ mục cho `find_by_tracklet`."""

        self._next_id = 1
        self.n_created = 0
        self.n_closed = 0

    # ------------------------------------------------------------------ vòng đời

    def create(self, tracklet: Tracklet) -> GlobalTrack:
        """Global ID mới cho tracklet chưa khớp được với ai (CLAUDE.md §6 bước 5)."""
        track = GlobalTrack(
            global_id=self._next_id,
            created_ms=tracklet.start_ms,
            last_seen_ms=tracklet.end_ms,
            last_cam_id=tracklet.cam_id,
        )
        self._next_id += 1
        self.n_created += 1
        self._tracks[track.global_id] = track
        self._absorb(track, tracklet)
        return track

    def assign(self, track: GlobalTrack, tracklet: Tracklet) -> GlobalTrack:
        """Gắn tracklet vào một GlobalTrack đã có và cập nhật gallery của nó."""
        if track.closed:
            raise ValueError(f"GlobalTrack {track.global_id} đã đóng, không gán thêm được")
        self._absorb(track, tracklet)
        return track

    def _absorb(self, track: GlobalTrack, tracklet: Tracklet) -> None:
        previous = track.cam_last_tracklet.get(tracklet.cam_id)
        is_update = previous == tracklet.tracklet_id

        if not is_update:
            track.n_tracklets += 1
            track.members.append(TrackletRef.of(tracklet))
        elif track.members and track.members[-1].tracklet_id == tracklet.tracklet_id:
            # Cùng tracklet được gán lại ở cửa sổ sau: nới mốc kết thúc thay vì thêm bản ghi.
            track.members[-1] = TrackletRef.of(tracklet)

        track.cam_last_tracklet[tracklet.cam_id] = tracklet.tracklet_id
        # Chỉ mục chỉ giữ tracklet ĐANG được sở hữu: tracklet cũ của chính camera này vừa
        # bị thay nên bỏ khỏi chỉ mục, nếu không nó phình theo tổng số tracklet từng thấy
        # thay vì theo số track đang mở.
        if previous is not None and previous != tracklet.tracklet_id:
            self._owner_of.pop(previous, None)
        self._owner_of[tracklet.tracklet_id] = track.global_id
        track.cam_ground_path[tracklet.cam_id] = tracklet.ground_path
        spans = [
            span
            for span in track.cam_spans.get(tracklet.cam_id, [])
            if span[2] != tracklet.tracklet_id
        ]
        spans.append((tracklet.start_ms, tracklet.end_ms, tracklet.tracklet_id))
        track.cam_spans[tracklet.cam_id] = sorted(spans)[-_MAX_SPANS_PER_CAM:]
        track.cam_last_seen[tracklet.cam_id] = max(
            tracklet.end_ms, track.cam_last_seen.get(tracklet.cam_id, tracklet.end_ms)
        )
        if tracklet.end_ms >= track.last_seen_ms:
            track.last_seen_ms = tracklet.end_ms
            track.last_cam_id = tracklet.cam_id
            track.last_ground_point = tracklet.last_ground_point
        track.created_ms = min(track.created_ms, tracklet.start_ms)

        query = tracklet.query_embedding(self.config.topk_query)
        if query is None:  # pipeline chạy với ReID tắt (M1/M2) — vẫn theo dõi được vòng đời
            return

        # Cùng một tracklet được gán lại ở mỗi cửa sổ (chế độ online), nên phải THAY bản
        # ghi cũ chứ không chồng thêm: không có bước này thì một tracklet dài chiếm trọn
        # `max_size` ô và `_enforce_quota` đẩy hết ngoại hình của camera khác ra ngoài —
        # đúng lúc cần chúng nhất. Đo 2026-09-06 trên fixture DeepStream: 32/32 ô thuộc về
        # một tracklet, khử trùng lặp thì còn trung vị 3 và F1 online 0.153 → 0.163.
        if track.entries and any(e.tracklet_id == tracklet.tracklet_id for e in track.entries):
            track.entries = [e for e in track.entries if e.tracklet_id != tracklet.tracklet_id]

        track.entries.append(
            GalleryEntry(
                embedding=query,
                cam_id=tracklet.cam_id,
                confidence=tracklet.mean_confidence,
                ts_ms=tracklet.end_ms,
                tracklet_id=tracklet.tracklet_id,
            )
        )
        self._enforce_quota(track)

        alpha = self.config.ema_alpha
        track.centroid = (
            query
            if track.centroid is None
            else l2_normalize((1.0 - alpha) * track.centroid + alpha * query)
        )

    def _enforce_quota(self, track: GlobalTrack) -> None:
        """Chặn gallery ở `max_size`, chia hạn ngạch đều cho các camera đã thấy.

        Loại bỏ theo vòng: camera nào đang giữ nhiều nhất thì bỏ bản ghi yếu nhất của
        camera đó (confidence thấp nhất, cũ nhất). Nhờ vậy camera hiếm gặp không bị
        camera mà người đó đứng lâu nhất đè mất.
        """
        while len(track.entries) > self.config.max_size:
            by_cam: dict[str, list[GalleryEntry]] = {}
            for entry in track.entries:
                by_cam.setdefault(entry.cam_id, []).append(entry)
            crowded = max(by_cam.values(), key=len)
            weakest = min(crowded, key=lambda e: (e.confidence, e.ts_ms))
            track.entries.remove(weakest)

    def prune(self, now_ms: int) -> list[GlobalTrack]:
        """Đóng các GlobalTrack quá hạn `global_track_ttl_ms`. Trả về chúng để ghi SQLite."""
        expired = [
            track
            for track in self._tracks.values()
            if now_ms - track.last_seen_ms > self.config.global_track_ttl_ms
        ]
        for track in expired:
            track.closed = True
            self._tracks.pop(track.global_id, None)
            for tracklet_id in track.cam_last_tracklet.values():
                self._owner_of.pop(tracklet_id, None)
            self.n_closed += 1
        return sorted(expired, key=lambda t: t.global_id)

    # ------------------------------------------------------------------ truy vấn

    def candidates(self, tracklet: Tracklet, *, exclusion_slack_ms: int = 0) -> list[GlobalTrack]:
        """Ứng viên có thể khớp với `tracklet`, sau ràng buộc loại trừ cùng camera.

        Chỉ lọc phần phụ thuộc camera; ràng buộc thời gian di chuyển giữa cặp camera là
        việc của `topology.py` vì nó cần đồ thị camera mà gallery không giữ.
        """
        out = [
            track
            for track in self._tracks.values()
            if track.owns_tracklet(tracklet)
            or not track.overlaps_in(
                tracklet.cam_id, tracklet.start_ms, tracklet.end_ms, exclusion_slack_ms
            )
        ]
        return sorted(out, key=lambda t: t.global_id)

    def get(self, global_id: int) -> GlobalTrack | None:
        return self._tracks.get(global_id)

    def find_by_tracklet(self, tracklet: Tracklet) -> GlobalTrack | None:
        """GlobalTrack đang giữ tracklet này (None nếu tracklet chưa được gán).

        Tra bảng chỉ mục chứ không quét gallery: hàm này chạy cho mọi tracklet của mọi
        cửa sổ, nên quét tuyến tính biến vòng gán thành bậc hai theo số GlobalTrack đang
        mở (2,9 triệu lời gọi `owns_tracklet` trong một lượt chạy 2800 message).
        """
        global_id = self._owner_of.get(tracklet.tracklet_id)
        if global_id is None:
            return None
        track = self._tracks.get(global_id)
        # Chỉ mục có thể trỏ tới track đã bị `prune` đóng, hoặc tới một tracklet mà camera
        # đó đã gán cho tracklet khác — `owns_tracklet` vẫn là nguồn sự thật.
        return track if track is not None and track.owns_tracklet(tracklet) else None

    def open_tracks(self) -> list[GlobalTrack]:
        return sorted(self._tracks.values(), key=lambda t: t.global_id)

    def __len__(self) -> int:
        return len(self._tracks)

    def __iter__(self) -> Iterator[GlobalTrack]:
        return iter(self.open_tracks())


def similarity_matrix(
    tracklets: Iterable[Tracklet],
    tracks: Iterable[GlobalTrack],
    *,
    topk_query: int = 8,
    mode: SimilarityMode = "max",
) -> np.ndarray:
    """Ma trận similarity ngoại hình (hàng = tracklet, cột = GlobalTrack).

    Ô của tracklet chưa có embedding đặt -1.0 — `affinity.py` biến nó thành `inf` trong
    ma trận chi phí, tức "không khớp được bằng ngoại hình", chứ không phải "rất khác".
    """
    tracklets = list(tracklets)
    tracks = list(tracks)
    matrix = np.full((len(tracklets), len(tracks)), -1.0, dtype=np.float32)

    for i, tracklet in enumerate(tracklets):
        query = tracklet.query_embedding(topk_query)
        if query is None:
            continue
        for j, track in enumerate(tracks):
            matrix[i, j] = track.similarity(query, mode)
    return matrix
