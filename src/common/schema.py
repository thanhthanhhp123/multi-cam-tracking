"""Contract dữ liệu giữa pipeline DeepStream (máy GPU) và engine liên kết (CPU).

Đây là ranh giới DUY NHẤT giữa hai nửa codebase (CLAUDE.md §2, §5). Sửa file này là
breaking change: phải cập nhật đồng thời producer (`src/ds_pipeline`), consumer
(`src/mct`) và toàn bộ fixture trong `tests/fixtures/`.

Hai định dạng, một nguồn sự thật:
  - msgpack  — dùng trên wire (Redis). Embedding là raw bytes float32.
  - JSONL    — dùng cho fixture. Embedding base64, các trường khác đọc/sửa tay được.
Cả hai đi qua cùng bộ hàm `_message_to_dict` / `_message_from_dict`, nên không thể
lệch nhau.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgpack
import numpy as np

SCHEMA_VERSION = 1

CLASS_PERSON = 0

EMB_DTYPE = np.float32

# Sai số cho phép khi kiểm tra embedding đã L2-normalize.
_NORM_TOL = 1e-2

# bbox được phép tràn khỏi khung một chút (detector đôi khi không clip sát mép).
_BBOX_MARGIN_RATIO = 0.02


class SchemaError(ValueError):
    """Dữ liệu không đúng contract."""


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Chuẩn hoá L2. Producer PHẢI gọi hàm này trước khi gửi embedding đi."""
    vec = np.asarray(vec, dtype=EMB_DTYPE).ravel()
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise SchemaError("embedding có norm bằng 0, không chuẩn hoá được")
    return (vec / norm).astype(EMB_DTYPE)


@dataclass(slots=True)
class Detection:
    """Một đối tượng được phát hiện và gán local track ID trong phạm vi một camera."""

    local_track_id: int
    """Chỉ duy nhất trong phạm vi một camera. Khoá toàn cục là (cam_id, local_track_id)."""

    bbox: tuple[float, float, float, float]
    """(x, y, w, h) pixel, gốc trên-trái, THEO ĐỘ PHÂN GIẢI GỐC CỦA CAMERA.

    DeepStream trả `rect_params` theo toạ độ của nvstreammux — probe phải scale ngược
    về độ phân giải camera trước khi tạo Detection. Xem CLAUDE.md §5.
    """

    confidence: float
    embedding: np.ndarray | None = None
    """float32, đã L2-normalize. None khi frame đó chưa trích được Re-ID."""

    class_id: int = CLASS_PERSON

    @property
    def ground_point(self) -> tuple[float, float]:
        """Điểm chân = đáy-giữa bbox.

        Đây là điểm dùng cho phép biến đổi homography ở cặp camera overlap. Chốt trong
        contract để pipeline và engine không mỗi bên hiểu một kiểu.
        """
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h)


@dataclass(slots=True)
class FrameMessage:
    """Toàn bộ đối tượng phát hiện được trong một frame của một camera."""

    cam_id: str
    """String ổn định ("cam01"...), khớp key trong configs/cameras/topology.yaml.
    KHÔNG dùng index của nvstreammux — index đổi khi bật/tắt camera."""

    frame_id: int
    ts_ms: int
    """Epoch milliseconds theo wall clock (NTP). Dùng cho ràng buộc thời gian xuyên camera."""

    frame_pts_ns: int
    """PTS của GStreamer. Dùng để đồng bộ nội bộ pipeline, KHÔNG dùng so khớp xuyên camera."""

    frame_width: int
    frame_height: int
    """Độ phân giải gốc của camera, không phải của streammux."""

    detections: list[Detection] = field(default_factory=list)
    embed_dim: int = 0
    """0 nghĩa là message này không mang embedding."""

    schema_version: int = SCHEMA_VERSION

    def infer_embed_dim(self) -> int:
        for det in self.detections:
            if det.embedding is not None:
                return int(det.embedding.shape[0])
        return 0


# --------------------------------------------------------------------------- #
# Biểu diễn trung gian — cả msgpack và JSONL đều đi qua đây
# --------------------------------------------------------------------------- #

_EmbEncoder = Callable[[np.ndarray], Any]
_EmbDecoder = Callable[[Any], np.ndarray]


def _emb_to_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=EMB_DTYPE).tobytes()


def _emb_from_bytes(raw: Any) -> np.ndarray:
    return np.frombuffer(raw, dtype=EMB_DTYPE).copy()


def _emb_to_b64(vec: np.ndarray) -> str:
    return base64.b64encode(_emb_to_bytes(vec)).decode("ascii")


def _emb_from_b64(raw: Any) -> np.ndarray:
    return _emb_from_bytes(base64.b64decode(raw))


def _message_to_dict(msg: FrameMessage, encode_emb: _EmbEncoder) -> dict[str, Any]:
    detections: list[dict[str, Any]] = []
    for det in msg.detections:
        item: dict[str, Any] = {
            "local_track_id": int(det.local_track_id),
            "bbox": [float(v) for v in det.bbox],
            "confidence": float(det.confidence),
            "class_id": int(det.class_id),
        }
        if det.embedding is not None:
            item["embedding"] = encode_emb(det.embedding)
        detections.append(item)

    return {
        "schema_version": int(msg.schema_version),
        "cam_id": str(msg.cam_id),
        "frame_id": int(msg.frame_id),
        "ts_ms": int(msg.ts_ms),
        "frame_pts_ns": int(msg.frame_pts_ns),
        "frame_width": int(msg.frame_width),
        "frame_height": int(msg.frame_height),
        "embed_dim": int(msg.embed_dim or msg.infer_embed_dim()),
        "detections": detections,
    }


def _message_from_dict(data: dict[str, Any], decode_emb: _EmbDecoder) -> FrameMessage:
    version = int(data.get("schema_version", -1))
    if version != SCHEMA_VERSION:
        raise SchemaError(
            f"schema_version {version} không khớp {SCHEMA_VERSION}. "
            "Sinh lại fixture hoặc cập nhật producer/consumer cùng lúc."
        )

    detections = [
        Detection(
            local_track_id=int(item["local_track_id"]),
            bbox=tuple(float(v) for v in item["bbox"]),  # type: ignore[arg-type]
            confidence=float(item["confidence"]),
            embedding=decode_emb(item["embedding"]) if item.get("embedding") is not None else None,
            class_id=int(item.get("class_id", CLASS_PERSON)),
        )
        for item in data.get("detections", [])
    ]

    return FrameMessage(
        cam_id=str(data["cam_id"]),
        frame_id=int(data["frame_id"]),
        ts_ms=int(data["ts_ms"]),
        frame_pts_ns=int(data["frame_pts_ns"]),
        frame_width=int(data["frame_width"]),
        frame_height=int(data["frame_height"]),
        detections=detections,
        embed_dim=int(data.get("embed_dim", 0)),
        schema_version=version,
    )


# --------------------------------------------------------------------------- #
# msgpack — định dạng trên wire
# --------------------------------------------------------------------------- #


def encode_msgpack(msg: FrameMessage) -> bytes:
    return msgpack.packb(_message_to_dict(msg, _emb_to_bytes), use_bin_type=True)


def decode_msgpack(raw: bytes) -> FrameMessage:
    data = msgpack.unpackb(raw, raw=False)
    return _message_from_dict(data, _emb_from_bytes)


# --------------------------------------------------------------------------- #
# JSONL — định dạng fixture
# --------------------------------------------------------------------------- #


def encode_jsonl(msg: FrameMessage) -> str:
    """Một message thành đúng một dòng JSON."""
    return json.dumps(_message_to_dict(msg, _emb_to_b64), separators=(",", ":"))


def decode_jsonl(line: str) -> FrameMessage:
    return _message_from_dict(json.loads(line), _emb_from_b64)


def write_jsonl(path: str | Path, messages: Iterable[FrameMessage]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for msg in messages:
            fh.write(encode_jsonl(msg))
            fh.write("\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[FrameMessage]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield decode_jsonl(line)
            except (SchemaError, KeyError, ValueError) as exc:
                raise SchemaError(f"{path}:{lineno}: {exc}") from exc


# --------------------------------------------------------------------------- #
# mct:global — cập nhật Global ID cho dashboard
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class GlobalUpdate:
    """Một lần engine gán/cập nhật Global ID cho một tracklet.

    Đây là toàn bộ nội dung stream `mct:global` (CLAUDE.md §5). Cố tình KHÔNG mang
    embedding: dashboard không cần ngoại hình, mà 256 float mỗi cập nhật thì stream phình
    vô ích. Ai cần ngoại hình thì đọc `mct:frames`.
    """

    global_id: int
    cam_id: str
    local_track_id: int
    tracklet_id: int
    ts_ms: int
    """Mốc muộn nhất của tracklet tại thời điểm gán (không phải giờ hệ thống lúc ghi)."""

    bbox: tuple[float, float, float, float]
    """Vị trí gần nhất trong ảnh, [x, y, w, h] theo độ phân giải GỐC của camera."""

    ground_point: tuple[float, float] = (0.0, 0.0)
    cost: float = 0.0
    """Chi phí ghép (0.0 khi tạo mới hoặc khi chỉ cập nhật tracklet đang chạy)."""

    is_new: bool = False
    """True = Global ID này vừa được tạo cho tracklet đó."""

    is_update: bool = False
    """True = tracklet đã thuộc Global ID này từ trước, lần này chỉ nới thêm."""

    n_cameras: int = 1
    """Số camera mà Global ID này đã đi qua — dashboard dùng để lọc "người xuyên camera"."""

    reason: str = ""
    """Vì sao phải tạo ID mới (rỗng nếu ghép được). Để truy nguyên khi soi dashboard."""


def _global_to_dict(update: GlobalUpdate) -> dict[str, Any]:
    return {
        "global_id": int(update.global_id),
        "cam_id": update.cam_id,
        "local_track_id": int(update.local_track_id),
        "tracklet_id": int(update.tracklet_id),
        "ts_ms": int(update.ts_ms),
        "bbox": [float(v) for v in update.bbox],
        "ground_point": [float(v) for v in update.ground_point],
        "cost": float(update.cost),
        "is_new": bool(update.is_new),
        "is_update": bool(update.is_update),
        "n_cameras": int(update.n_cameras),
        "reason": update.reason,
    }


def _global_from_dict(data: dict[str, Any]) -> GlobalUpdate:
    bbox = tuple(float(v) for v in data["bbox"])
    if len(bbox) != 4:
        raise SchemaError(f"bbox phải có 4 phần tử, nhận {len(bbox)}")
    point = tuple(float(v) for v in data.get("ground_point", (0.0, 0.0)))
    return GlobalUpdate(
        global_id=int(data["global_id"]),
        cam_id=str(data["cam_id"]),
        local_track_id=int(data["local_track_id"]),
        tracklet_id=int(data["tracklet_id"]),
        ts_ms=int(data["ts_ms"]),
        bbox=bbox,  # type: ignore[arg-type]
        ground_point=point,  # type: ignore[arg-type]
        cost=float(data.get("cost", 0.0)),
        is_new=bool(data.get("is_new", False)),
        is_update=bool(data.get("is_update", False)),
        n_cameras=int(data.get("n_cameras", 1)),
        reason=str(data.get("reason", "")),
    )


def encode_global_msgpack(update: GlobalUpdate) -> bytes:
    return msgpack.packb(_global_to_dict(update), use_bin_type=True)


def decode_global_msgpack(raw: bytes) -> GlobalUpdate:
    return _global_from_dict(msgpack.unpackb(raw, raw=False))


def encode_global_jsonl(update: GlobalUpdate) -> str:
    return json.dumps(_global_to_dict(update), separators=(",", ":"), ensure_ascii=False)


def decode_global_jsonl(line: str) -> GlobalUpdate:
    return _global_from_dict(json.loads(line))


# --------------------------------------------------------------------------- #
# Kiểm tra tính hợp lệ
# --------------------------------------------------------------------------- #


def validate(msg: FrameMessage, *, strict: bool = False) -> list[str]:
    """Trả về danh sách vấn đề tìm thấy. `strict=True` thì raise SchemaError.

    Đáng chú ý nhất là kiểm tra bbox tràn khung: đó là triệu chứng điển hình của việc
    quên scale toạ độ từ nvstreammux về độ phân giải camera. Lưu ý nó chỉ bắt được
    chiều streammux LỚN HƠN camera; chiều ngược lại (bbox bị co nhỏ) nằm gọn trong
    khung nên phải phát hiện bằng cách xem trực quan overlay.
    """
    issues: list[str] = []

    if msg.schema_version != SCHEMA_VERSION:
        issues.append(f"schema_version={msg.schema_version}, mong đợi {SCHEMA_VERSION}")
    if not msg.cam_id:
        issues.append("cam_id rỗng")
    if msg.frame_width <= 0 or msg.frame_height <= 0:
        issues.append(f"kích thước frame không hợp lệ: {msg.frame_width}x{msg.frame_height}")
    if msg.ts_ms <= 0:
        issues.append(f"ts_ms={msg.ts_ms} không hợp lệ (cần epoch milliseconds)")

    seen: set[int] = set()
    margin_x = msg.frame_width * _BBOX_MARGIN_RATIO
    margin_y = msg.frame_height * _BBOX_MARGIN_RATIO

    for det in msg.detections:
        tag = f"track {det.local_track_id}"

        if det.local_track_id in seen:
            issues.append(f"{tag}: local_track_id trùng trong cùng một frame")
        seen.add(det.local_track_id)

        x, y, w, h = det.bbox
        if w <= 0 or h <= 0:
            issues.append(f"{tag}: bbox có cạnh không dương ({w}x{h})")
        if (
            x < -margin_x
            or y < -margin_y
            or x + w > msg.frame_width + margin_x
            or y + h > msg.frame_height + margin_y
        ):
            issues.append(
                f"{tag}: bbox {det.bbox} tràn khỏi khung {msg.frame_width}x{msg.frame_height} "
                "— rất có thể probe quên scale toạ độ nvstreammux về độ phân giải camera"
            )

        if not 0.0 <= det.confidence <= 1.0:
            issues.append(f"{tag}: confidence={det.confidence} ngoài khoảng [0, 1]")

        if det.embedding is not None:
            if det.embedding.dtype != EMB_DTYPE:
                issues.append(f"{tag}: embedding dtype {det.embedding.dtype}, cần {EMB_DTYPE}")
            if msg.embed_dim and det.embedding.shape[0] != msg.embed_dim:
                issues.append(
                    f"{tag}: embedding dim {det.embedding.shape[0]} lệch header {msg.embed_dim}"
                )
            norm = float(np.linalg.norm(det.embedding))
            if not math.isclose(norm, 1.0, abs_tol=_NORM_TOL):
                issues.append(
                    f"{tag}: embedding chưa L2-normalize (norm={norm:.4f}) "
                    "— producer phải gọi l2_normalize() trước khi gửi"
                )

    if strict and issues:
        raise SchemaError(f"{msg.cam_id} frame {msg.frame_id}: " + "; ".join(issues))
    return issues
