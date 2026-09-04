"""Đọc embedding Re-ID ra khỏi metadata DeepStream (M3).

Tách riêng khỏi `probes.py` vì đây là chỗ duy nhất trong repo phụ thuộc vào **chi tiết
theo phiên bản** của DeepStream: tên hằng meta type và hình dạng struct đổi giữa các bản
(CLAUDE.md §11). Gom vào một file để khi đổi bản DeepStream chỉ phải đọc lại một chỗ.

Hai đường lấy embedding, thử theo thứ tự:

  (A) **mặc định** — ReID tích hợp trong `nvtracker` (NvDCF, `reidType != 0` và
      `outputReidTensor: 1`). Tracker gắn `NvDsObjReid` vào `obj_user_meta_list`.
      Xem `configs/pipeline/config_tracker_NvDCF_reid.yml`.
  (B) **dự phòng** — SGIE `nvinfer` thứ hai (`process-mode=2`, `output-tensor-meta=1`)
      chạy OSNet trên crop, ra `NVDSINFER_TENSOR_OUTPUT_META`.

API dùng cho đường (A), khớp `pyds` 1.2.0 (bản đi kèm DeepStream 7.1, chốt trong
`docker/deepstream.Dockerfile`):

    user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_OBJ_REID_META
    reid = pyds.NvDsObjReid.cast(user_meta.user_meta_data)
    vec  = reid.get_host_reid_vector()   # numpy float32, kích thước reid.featureSize

`pyds` được import **bên trong hàm**, không phải ở đầu file, để module này vẫn nạp được
trên máy dev không có DeepStream — nhờ vậy phần logic thuần numpy dưới đây test được mà
không cần GPU. Ngoại lệ có chủ ý so với `probes.py` (nơi import thẳng ở đầu file).
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from common.logging import get_logger

log = get_logger(__name__)

# Tên hằng meta type của tracker khi ReID bật. Tra từ binding pyds v1.2.0
# (deepstream_python_apps, bindings/src/bindtrackermeta.cpp) — KHÔNG đoán từ bản khác.
TRACKER_REID_META_TYPE = "NVDS_TRACKER_OBJ_REID_META"

# Nhiều nhất một cảnh báo cho mỗi loại sự cố: probe chạy trên MỌI đối tượng của MỌI
# frame, log không chặn sẽ nhấn chìm mọi thứ khác và tự nó làm tụt FPS.
_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args: object) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args)


def sanitize_embedding(raw: Any, expected_dim: int = 0) -> np.ndarray | None:
    """Kiểm tra vector thô rồi COPY ra khỏi bộ nhớ do DeepStream quản lý.

    `get_host_reid_vector()` trả về numpy array *bọc* con trỏ `ptr_host` của meta, không
    sao chép. Meta bị giải phóng ngay khi probe trả về, nên giữ lại view đó là đọc bộ nhớ
    đã chết — dữ liệu hỏng âm thầm, không crash. Luôn `.copy()`.

    Trả `None` (không ném lỗi) khi vector không dùng được: một đối tượng thiếu embedding
    không phải lý do để cả pipeline chết.
    """
    if raw is None:
        return None
    vec = np.asarray(raw, dtype=np.float32).reshape(-1)
    if vec.size == 0:
        return None
    if expected_dim and vec.size != expected_dim:
        _warn_once(
            "dim",
            "Embedding %d chiều, lệch featureSize %d — bỏ qua. Kiểm tra reidFeatureSize "
            "trong config tracker có khớp model không.",
            vec.size,
            expected_dim,
        )
        return None
    if not np.isfinite(vec).all():
        _warn_once("nan", "Embedding chứa NaN/Inf — bỏ qua. Nghi engine FP16 tràn số.")
        return None
    return vec.copy()


def _tracker_meta_type(pyds: Any) -> Any | None:
    """Hằng meta type của tracker, hoặc None nếu bản pyds này không có.

    Không có nghĩa là DeepStream quá cũ để hỗ trợ đường (A). Trả None để bên gọi rơi
    xuống đường (B) thay vì ném AttributeError giữa probe.
    """
    meta_types = getattr(pyds, "NvDsMetaType", None)
    if meta_types is None:
        return None
    return getattr(meta_types, TRACKER_REID_META_TYPE, None)


def _from_tracker_meta(pyds: Any, user_meta: Any) -> np.ndarray | None:
    """Đường (A): NvDsObjReid do nvtracker gắn."""
    try:
        reid = pyds.NvDsObjReid.cast(user_meta.user_meta_data)
    except Exception:  # pragma: no cover - cần pyds thật
        _warn_once("cast", "Không cast được NvDsObjReid — bản pyds không khớp?")
        return None

    size = int(getattr(reid, "featureSize", 0) or 0)
    if size <= 0:
        # Bình thường: target đang shadow-tracking chưa có đặc trưng mới. Không cảnh báo.
        return None
    return sanitize_embedding(reid.get_host_reid_vector(), expected_dim=size)


def _from_tensor_meta(pyds: Any, user_meta: Any) -> np.ndarray | None:
    """Đường (B): tensor output của SGIE nvinfer thứ hai."""
    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
    if tensor_meta.num_output_layers < 1:
        return None
    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
    n = int(np.prod(layer.dims.d[: layer.dims.numDims]))
    if n <= 0:
        return None
    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
    return sanitize_embedding(np.ctypeslib.as_array(ptr, shape=(n,)))


def extract_reid_embedding(obj_meta: Any) -> np.ndarray | None:
    """Embedding Re-ID của một đối tượng, hoặc None nếu không có.

    Duyệt `obj_user_meta_list` một lần, ưu tiên đường (A). Chưa L2-normalize — bên gọi
    làm bằng `common.schema.l2_normalize` để đúng một chỗ chuẩn hoá (CLAUDE.md §5).
    """
    import pyds  # import muộn có chủ ý — xem docstring đầu file

    tracker_type = _tracker_meta_type(pyds)
    tensor_type = getattr(pyds.NvDsMetaType, "NVDSINFER_TENSOR_OUTPUT_META", None)

    fallback: np.ndarray | None = None
    l_user = obj_meta.obj_user_meta_list
    while l_user is not None:
        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        meta_type = user_meta.base_meta.meta_type
        if tracker_type is not None and meta_type == tracker_type:
            vec = _from_tracker_meta(pyds, user_meta)
            if vec is not None:
                return vec
        elif tensor_type is not None and meta_type == tensor_type and fallback is None:
            fallback = _from_tensor_meta(pyds, user_meta)
        l_user = l_user.next

    if fallback is None and tracker_type is None:
        _warn_once(
            "no_tracker_type",
            "pyds không có NvDsMetaType.%s — bản DeepStream này không hỗ trợ đường (A), "
            "chỉ còn SGIE (đường B). Xem CLAUDE.md §11.",
            TRACKER_REID_META_TYPE,
        )
    return fallback
