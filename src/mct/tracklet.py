"""Gom FrameMessage rời rạc thành tracklet cục bộ — bước 1 của engine liên kết (CLAUDE.md §6).

Tracklet = toàn bộ lần xuất hiện liên tục của MỘT local_track_id trong MỘT camera.
Đây là đơn vị mà thuật toán liên kết làm việc: so khớp tracklet ↔ GlobalTrack, không so
khớp từng detection lẻ (detection đơn lẻ quá nhiễu — crop mờ, bị che, bbox lệch).

Hai điểm dễ sai, đã xử lý sẵn ở đây:

1. **`local_track_id` bị dùng lại.** ID của nvtracker chỉ duy nhất *tại một thời điểm*
   trong phạm vi một camera; sau khi track chết, tracker có quyền cấp lại số đó cho
   người khác. Nếu cứ khoá theo `(cam_id, local_track_id)` mãi mãi thì hai người khác
   nhau bị gộp thành một tracklet và mọi Global ID phía sau sai theo. Nên mỗi tracklet
   có `tracklet_id` nội bộ tăng dần, và khoảng lặng dài hơn `idle_timeout_ms` sẽ cắt
   sang tracklet mới.

2. **Query embedding không phải embedding frame cuối.** Lấy trung bình có trọng số của
   top-k detection có confidence cao nhất rồi L2-normalize (CLAUDE.md §6 bước 1). Giữ
   toàn bộ embedding của một tracklet dài là phí bộ nhớ, nên chỉ giữ `max_embeddings`
   cái tốt nhất bằng một min-heap theo confidence.

Module này KHÔNG chạm Redis và KHÔNG cần GPU: nạp vào bằng `FrameMessage` từ đâu cũng
được (Redis thật, hay fixture JSONL phát lại trên Mac).
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from common.schema import Detection, FrameMessage, l2_normalize

# Số observation gần nhất giữ lại cho mỗi tracklet (để ước lượng hướng di chuyển ở
# affinity.py sau này). Không phải ngưỡng thuật toán — chỉ là trần bộ nhớ.
_RECENT_HISTORY = 16


@dataclass(slots=True, frozen=True)
class Observation:
    """Một detection đã gắn mốc thời gian của frame chứa nó."""

    ts_ms: int
    frame_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    ground_point: tuple[float, float]
    embedding: np.ndarray | None = None

    @classmethod
    def from_detection(cls, msg: FrameMessage, det: Detection) -> Observation:
        # ground_point lấy từ Detection để quy ước "đáy-giữa bbox" chỉ định nghĩa một
        # chỗ duy nhất (common/schema.py), không chép lại ở đây.
        return cls(
            ts_ms=int(msg.ts_ms),
            frame_id=int(msg.frame_id),
            bbox=det.bbox,
            confidence=float(det.confidence),
            ground_point=det.ground_point,
            embedding=det.embedding,
        )


@dataclass(slots=True)
class TrackletConfig:
    """Tham số gom tracklet. Nguồn: khối `tracklet` + `gallery` trong configs/mct.yaml."""

    min_frames: int = 5
    """Tracklet ngắn hơn ngưỡng này bị bỏ: quá ít frame thì embedding trung bình không đáng tin."""

    idle_timeout_ms: int = 2000
    """Không thấy lại trong khoảng này thì coi tracklet đã kết thúc."""

    max_embeddings: int = 32
    """Trần số embedding giữ cho mỗi tracklet (giữ những cái confidence cao nhất)."""

    topk_query: int = 8
    """Số embedding dùng khi tính query embedding."""

    ground_path_max_points: int = 64
    """Trần số điểm quỹ đạo mặt đất giữ cho mỗi tracklet.

    Thành phần hình học của `affinity.py` cần so vị trí tại CÙNG mốc thời gian giữa hai
    camera, nên một điểm chân duy nhất là không đủ. Đầy trần thì tỉa bớt một nửa (giữ
    điểm chẵn) — vẫn phủ trọn quãng thời gian, chỉ thưa hơn.
    """

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> TrackletConfig:
        """Đọc từ dict đã load của configs/mct.yaml (chấp nhận cả dict con `tracklet`)."""
        tracklet = dict(data.get("tracklet", data) or {})
        gallery = dict(data.get("gallery", {}) or {})
        defaults = cls()
        return cls(
            min_frames=int(tracklet.get("min_frames", defaults.min_frames)),
            idle_timeout_ms=int(tracklet.get("idle_timeout_ms", defaults.idle_timeout_ms)),
            max_embeddings=int(tracklet.get("max_embeddings", defaults.max_embeddings)),
            topk_query=int(gallery.get("topk_query", defaults.topk_query)),
            ground_path_max_points=int(
                tracklet.get("ground_path_max_points", defaults.ground_path_max_points)
            ),
        )

    def __post_init__(self) -> None:
        if self.min_frames < 1:
            raise ValueError(f"min_frames phải >= 1, nhận {self.min_frames}")
        if self.idle_timeout_ms <= 0:
            raise ValueError(f"idle_timeout_ms phải > 0, nhận {self.idle_timeout_ms}")
        if self.max_embeddings < 1:
            raise ValueError(f"max_embeddings phải >= 1, nhận {self.max_embeddings}")
        if self.topk_query < 1:
            raise ValueError(f"topk_query phải >= 1, nhận {self.topk_query}")
        if self.ground_path_max_points < 2:
            raise ValueError(
                f"ground_path_max_points phải >= 2, nhận {self.ground_path_max_points}"
            )


@dataclass(slots=True)
class Tracklet:
    """Một lần xuất hiện liên tục của một local_track_id trong một camera."""

    tracklet_id: int
    """Duy nhất trong toàn bộ phiên chạy, kể cả khi camera dùng lại local_track_id."""

    cam_id: str
    local_track_id: int

    start_ms: int = 0
    end_ms: int = 0
    start_frame_id: int = 0
    end_frame_id: int = 0
    n_frames: int = 0
    n_embeddings_seen: int = 0
    """Tổng số detection CÓ embedding đã đi qua, kể cả cái đã bị loại khỏi heap."""

    confidence_sum: float = 0.0
    closed: bool = False

    first_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    first_ground_point: tuple[float, float] = (0.0, 0.0)
    last_ground_point: tuple[float, float] = (0.0, 0.0)

    ground_path: list[tuple[int, tuple[float, float]]] = field(default_factory=list)
    """(ts_ms, điểm chân trong ảnh) đã tỉa thưa — quỹ đạo để so vị trí theo thời gian.

    Giữ ở toạ độ ảnh, không quy đổi sẵn ra mét: phép ánh xạ phụ thuộc camera và lúc ghi
    thì chưa chắc đã có file hiệu chỉnh.
    """

    recent: deque[Observation] = field(default_factory=lambda: deque(maxlen=_RECENT_HISTORY))
    """Vài observation gần nhất — dùng cho ràng buộc chuyển động, không phải toàn bộ lịch sử."""

    # Min-heap (confidence, seq, embedding): đỉnh heap là ứng viên bị loại tiếp theo.
    # `seq` chỉ để so sánh tất định khi confidence bằng nhau (ndarray không so sánh được).
    _embeddings: list[tuple[float, int, np.ndarray]] = field(default_factory=list)
    _emb_seq: int = 0
    _query_cache: tuple[int, int, np.ndarray] | None = None
    """(top_k, _emb_seq lúc tính, vector) — huỷ hiệu lực khi có embedding mới."""

    @property
    def key(self) -> tuple[str, int]:
        """Khoá cục bộ theo camera.

        KHÔNG phải khoá toàn cục qua thời gian — camera có thể cấp lại số này cho
        người khác; xem docstring module.
        """
        return (self.cam_id, self.local_track_id)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def n_embeddings(self) -> int:
        """Số embedding thực sự đang giữ (đã bị chặn bởi max_embeddings)."""
        return len(self._embeddings)

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.n_frames if self.n_frames else 0.0

    def add(self, obs: Observation, *, max_embeddings: int, max_points: int = 64) -> None:
        if self.closed:
            raise ValueError(f"tracklet {self.tracklet_id} đã đóng, không thêm được observation")

        if self.n_frames == 0:
            self.start_ms = obs.ts_ms
            self.start_frame_id = obs.frame_id
            self.first_bbox = obs.bbox
            self.first_ground_point = obs.ground_point

        self.n_frames += 1
        self.confidence_sum += obs.confidence
        # Message có thể tới lệch thứ tự (Redis consumer group, replay nhiều camera) —
        # end_* phải là mốc muộn nhất đã thấy, không phải mốc của message cuối cùng.
        if obs.ts_ms >= self.end_ms:
            self.end_ms = obs.ts_ms
            self.end_frame_id = obs.frame_id
            self.last_bbox = obs.bbox
            self.last_ground_point = obs.ground_point
        self.recent.append(obs)
        self._push_ground_point(obs, max_points)

        if obs.embedding is not None:
            self._push_embedding(obs.embedding, obs.confidence, max_embeddings)

    def _push_ground_point(self, obs: Observation, max_points: int) -> None:
        self.ground_path.append((obs.ts_ms, obs.ground_point))
        if len(self.ground_path) > max(2, int(max_points)):
            self.ground_path = self.ground_path[::2]

    def _push_embedding(
        self, embedding: np.ndarray, confidence: float, max_embeddings: int
    ) -> None:
        self.n_embeddings_seen += 1
        self._emb_seq += 1
        self._query_cache = None
        item = (float(confidence), self._emb_seq, embedding)
        # Min-heap theo confidence: khi đầy thì đẩy vào rồi lấy ra cái yếu nhất, nên
        # cái vừa vào cũng có thể chính là cái bị loại — đúng ý "giữ top-N tốt nhất".
        if len(self._embeddings) < max(1, int(max_embeddings)):
            heapq.heappush(self._embeddings, item)
        else:
            heapq.heappushpop(self._embeddings, item)

    def query_embedding(self, top_k: int | None = None) -> np.ndarray | None:
        """Trung bình có trọng số confidence của top-k embedding tốt nhất, đã L2-normalize.

        `top_k=None` dùng toàn bộ embedding đang giữ. Trả None nếu tracklet chưa có
        embedding nào (pipeline chạy với ReID tắt — đúng tình trạng M1/M2, `embed_dim=0`).
        """
        if not self._embeddings:
            return None
        k = len(self._embeddings) if top_k is None else max(1, int(top_k))

        cache = self._query_cache
        if cache is not None and cache[0] == k and cache[1] == self._emb_seq:
            return cache[2]

        best = heapq.nlargest(k, self._embeddings, key=lambda item: (item[0], item[1]))
        weights = np.array([item[0] for item in best], dtype=np.float32)
        if float(weights.sum()) <= 0.0:
            # Mọi confidence bằng 0 (không nên xảy ra, nhưng đừng chia cho 0): lấy trung bình đều.
            weights = np.ones_like(weights)
        stacked = np.stack([item[2] for item in best]).astype(np.float32)
        vector = l2_normalize(weights @ stacked)

        self._query_cache = (k, self._emb_seq, vector)
        return vector

    def embeddings(self) -> list[np.ndarray]:
        """Các embedding đang giữ, sắp theo confidence giảm dần."""
        ordered = sorted(self._embeddings, key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in ordered]

    def __repr__(self) -> str:  # pragma: no cover - chỉ phục vụ debug
        state = "closed" if self.closed else "active"
        return (
            f"Tracklet(#{self.tracklet_id} {self.cam_id}/{self.local_track_id} {state} "
            f"frames={self.n_frames} emb={self.n_embeddings} "
            f"span={self.start_ms}..{self.end_ms}ms)"
        )


class TrackletBuilder:
    """Máy trạng thái gom FrameMessage thành tracklet.

    Dùng cho cả hai chế độ ở CLAUDE.md §6:
      - `online`  — mỗi cửa sổ: `update()` theo message tới, rồi `close_expired()` +
        `take_updated()` để lấy phần việc cho vòng gán.
      - `offline` — nạp hết message rồi `flush()` một lần để lấy toàn bộ tracklet.

    Không tự lấy giờ hệ thống: mốc thời gian luôn là `ts_ms` trong message, nên phát lại
    fixture cho kết quả y hệt lúc chạy thật (và test không phụ thuộc đồng hồ).
    """

    def __init__(self, config: TrackletConfig | None = None) -> None:
        self.config = config or TrackletConfig()
        self._active: dict[tuple[str, int], Tracklet] = {}
        self._updated: set[int] = set()
        self._by_id: dict[int, Tracklet] = {}
        self._next_id = 1
        self._latest_ts_ms = 0
        self.n_started = 0
        self.n_closed = 0
        self.n_dropped_short = 0
        """Số tracklet bị bỏ vì ngắn hơn min_frames — theo dõi để biết ngưỡng có quá gắt không."""

    # ------------------------------------------------------------------ nạp dữ liệu

    def update(self, msg: FrameMessage) -> list[Tracklet]:
        """Nạp một FrameMessage. Trả về tracklet bị đóng vì local_track_id được dùng lại.

        Chỉ đóng theo khoảng lặng của CHÍNH track đó; các track khác hết hạn được xử lý
        ở `close_expired()` — vì một track im lặng không có nghĩa là camera đã tắt.
        """
        self._latest_ts_ms = max(self._latest_ts_ms, int(msg.ts_ms))
        closed: list[Tracklet] = []

        for det in msg.detections:
            key = (msg.cam_id, int(det.local_track_id))
            obs = Observation.from_detection(msg, det)
            tracklet = self._active.get(key)

            if tracklet is not None and obs.ts_ms - tracklet.end_ms > self.config.idle_timeout_ms:
                # Cùng số local_track_id nhưng cách nhau quá lâu → coi như người khác.
                emitted = self._close(tracklet)
                if emitted is not None:
                    closed.append(emitted)
                tracklet = None

            if tracklet is None:
                tracklet = self._start(key)

            tracklet.add(
                obs,
                max_embeddings=self.config.max_embeddings,
                max_points=self.config.ground_path_max_points,
            )
            self._updated.add(tracklet.tracklet_id)

        return closed

    def update_many(self, messages: Iterable[FrameMessage]) -> list[Tracklet]:
        closed: list[Tracklet] = []
        for msg in messages:
            closed.extend(self.update(msg))
        return closed

    # ------------------------------------------------------------------ đóng / lấy ra

    def close_expired(self, now_ms: int | None = None) -> list[Tracklet]:
        """Đóng mọi tracklet im lặng lâu hơn `idle_timeout_ms` tính tới `now_ms`.

        `now_ms=None` thì lấy mốc muộn nhất đã thấy trong dữ liệu — đúng cho replay
        fixture; chạy thật với nguồn live nên truyền thẳng wall clock vào.
        """
        cutoff = self._latest_ts_ms if now_ms is None else int(now_ms)
        expired = [
            tracklet
            for tracklet in self._active.values()
            if cutoff - tracklet.end_ms > self.config.idle_timeout_ms
        ]
        out = [t for t in (self._close(tracklet) for tracklet in expired) if t is not None]
        return _ordered(out)

    def flush(self) -> list[Tracklet]:
        """Đóng toàn bộ tracklet còn mở (hết stream / chế độ offline)."""
        out = [t for t in (self._close(tracklet) for tracklet in list(self._active.values())) if t]
        return _ordered(out)

    def take_updated(self) -> list[Tracklet]:
        """Tracklet đã đủ dài và có cập nhật kể từ lần gọi trước; xoá dấu sau khi trả.

        Đây là đầu vào của một vòng gán ở chế độ online: chỉ những tracklet vừa động
        mới cần xét lại, không quét toàn bộ gallery mỗi cửa sổ.
        """
        out = [
            tracklet
            for tid in self._updated
            if (tracklet := self._by_id.get(tid)) is not None
            and tracklet.n_frames >= self.config.min_frames
        ]
        self._updated.clear()
        return _ordered(out)

    # ------------------------------------------------------------------ truy vấn

    def active(self) -> list[Tracklet]:
        return _ordered(self._active.values())

    def get(self, cam_id: str, local_track_id: int) -> Tracklet | None:
        """Tracklet đang mở của khoá này (None nếu track đó không còn sống)."""
        return self._active.get((cam_id, int(local_track_id)))

    def by_id(self, tracklet_id: int) -> Tracklet | None:
        return self._by_id.get(tracklet_id)

    @property
    def latest_ts_ms(self) -> int:
        return self._latest_ts_ms

    def __len__(self) -> int:
        return len(self._active)

    def __iter__(self) -> Iterator[Tracklet]:
        return iter(self.active())

    # ------------------------------------------------------------------ nội bộ

    def _start(self, key: tuple[str, int]) -> Tracklet:
        tracklet = Tracklet(tracklet_id=self._next_id, cam_id=key[0], local_track_id=key[1])
        self._next_id += 1
        self.n_started += 1
        self._active[key] = tracklet
        self._by_id[tracklet.tracklet_id] = tracklet
        return tracklet

    def _close(self, tracklet: Tracklet) -> Tracklet | None:
        """Đóng tracklet. Trả None nếu nó bị loại vì quá ngắn."""
        self._active.pop(tracklet.key, None)
        self._updated.discard(tracklet.tracklet_id)
        tracklet.closed = True

        if tracklet.n_frames < self.config.min_frames:
            self.n_dropped_short += 1
            self._by_id.pop(tracklet.tracklet_id, None)
            return None

        self.n_closed += 1
        return tracklet


def _ordered(tracklets: Iterable[Tracklet]) -> list[Tracklet]:
    """Thứ tự tất định — dict/set không đảm bảo, mà kết quả gán phải tái lập được."""
    return sorted(tracklets, key=lambda t: (t.end_ms, t.cam_id, t.local_track_id, t.tracklet_id))


def build_tracklets(
    messages: Iterable[FrameMessage], config: TrackletConfig | None = None
) -> list[Tracklet]:
    """Tiện ích chế độ offline: cả stream → danh sách tracklet đã đóng.

    Dùng cho `mode: offline` (cận trên độ chính xác, CLAUDE.md §6) và cho test.
    """
    builder = TrackletBuilder(config)
    closed = builder.update_many(messages)
    closed.extend(builder.flush())
    return _ordered(closed)
