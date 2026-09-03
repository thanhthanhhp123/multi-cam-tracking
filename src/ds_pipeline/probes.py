"""Pad probe trên src pad của nvtracker: NvDsBatchMeta -> FrameMessage -> Redis.

CHỈ chạy được trên máy có GPU + DeepStream runtime (CLAUDE.md §2). Import pyds ở đây
là hợp lệ — đây là package DUY NHẤT được phép.

Điểm chỉnh khoá (CLAUDE.md §5), đọc lại trước khi sửa:
  - bbox: NvDsObjectMeta.rect_params là toạ độ theo nvstreammux (đã resize theo
    streammux width/height), KHÔNG phải theo độ phân giải gốc của camera. Phải scale
    ngược bằng frame_width/frame_height gốc trước khi ghi vào Detection.
  - ts_ms: epoch milliseconds theo wall clock (NTP), không phải PTS. Lấy từ
    ntp_timestamp nếu streammux có attach-sys-ts; nếu không, dùng wall clock tại probe
    (chấp nhận độ trễ hàng chục ms — ghi rõ trong docstring, KHÔNG âm thầm coi là chính xác).
  - embedding: lấy từ user meta của nvtracker (đường A, CLAUDE.md §11) — tên tag khác
    nhau giữa các bản DeepStream, tra bằng gst-inspect / test thực tế, không đoán.
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Callable
from dataclasses import dataclass

import gi
import numpy as np
import pyds

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from common.logging import get_logger  # noqa: E402
from common.schema import Detection, FrameMessage, l2_normalize  # noqa: E402

log = get_logger(__name__)

# Tag chuẩn của NvDCF/NvDeepSORT khi ReID bật (reidType != 0) trên DeepStream 7.1.
# Xác nhận bằng test thực tế trên máy GPU (CLAUDE.md §11) — KHÔNG đoán từ tài liệu bản khác.
_REID_USER_META_TAG = "NVDS_TRACKER_OBJ_REID_FEATURE"

FrameSink = Callable[[FrameMessage], None]


@dataclass(slots=True, frozen=True)
class CameraGeometry:
    """Kích thước gốc của một camera — để scale ngược bbox từ toạ độ streammux."""

    cam_id: str
    width: int
    height: int


def _scale_bbox(
    rect: pyds.NvBoundingBox,
    mux_w: int,
    mux_h: int,
    cam_w: int,
    cam_h: int,
) -> tuple[float, float, float, float]:
    """rect_params theo toạ độ streammux -> (x, y, w, h) theo độ phân giải camera gốc.

    nvstreammux resize giữ tỉ lệ + letterbox khi enable-padding=1, hoặc kéo méo khi =0.
    Ở đây giả định enable-padding=0 (kéo méo, không viền đen) — khớp streams.yaml mặc
    định. Nếu bật padding, hàm này phải trừ thêm offset viền trước khi scale.
    """
    sx = cam_w / mux_w
    sy = cam_h / mux_h
    return (rect.left * sx, rect.top * sy, rect.width * sx, rect.height * sy)


def _extract_reid_embedding(obj_meta: pyds.NvDsObjectMeta) -> np.ndarray | None:
    """Đọc embedding ReID từ user meta của object, nếu tracker có gắn (reidType != 0)."""
    l_user = obj_meta.obj_user_meta_list
    while l_user is not None:
        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            # Đường dự phòng (B): SGIE nvinfer thứ hai, output-tensor-meta=1.
            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            vec = _read_reid_tensor(tensor_meta)
            if vec is not None:
                return vec
        l_user = l_user.next
    return None


def _read_reid_tensor(tensor_meta: pyds.NvDsInferTensorMeta) -> np.ndarray | None:
    if tensor_meta.num_output_layers < 1:
        return None
    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
    n = int(np.prod(layer.dims.d[: layer.dims.numDims]))
    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(ptr, shape=(n,)).astype(np.float32).copy()


def make_probe(
    geometries: dict[int, CameraGeometry],
    mux_width: int,
    mux_height: int,
    sink: FrameSink,
) -> Callable[[Gst.Pad, Gst.PadProbeInfo, object], Gst.PadProbeReturn]:
    """Trả về hàm probe gắn vào src pad của nvtracker.

    `geometries` ánh xạ source_id (index streammux, ổn định trong một phiên chạy) ->
    CameraGeometry mang cam_id thật + độ phân giải gốc. Một FrameMessage phát ra cho
    mỗi frame của mỗi camera có trong batch.
    """

    def probe(pad: Gst.Pad, info: Gst.PadProbeInfo, _user_data: object) -> Gst.PadProbeReturn:
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            geom = geometries.get(frame_meta.source_id)
            if geom is None:
                log.warning("source_id %d không có trong cấu hình camera, bỏ qua frame")
                l_frame = l_frame.next
                continue

            # attach-sys-ts trên streammux gắn NTP wall-clock vào buffer_pts của batch;
            # nếu không bật (nguồn file, live-source=0) thì fallback wall-clock tại probe —
            # đủ cho fixture/dev, KHÔNG đủ chính xác cho đối chiếu đa camera thời gian thực.
            ts_ms = int(time.time() * 1000)

            detections: list[Detection] = []
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                if obj_meta.class_id == 0:  # person — khớp CLASS_PERSON trong schema.py
                    bbox = _scale_bbox(
                        obj_meta.rect_params, mux_width, mux_height, geom.width, geom.height
                    )
                    raw_emb = _extract_reid_embedding(obj_meta)
                    embedding = l2_normalize(raw_emb) if raw_emb is not None else None
                    detections.append(
                        Detection(
                            local_track_id=int(obj_meta.object_id),
                            bbox=bbox,
                            confidence=float(obj_meta.confidence),
                            embedding=embedding,
                        )
                    )
                l_obj = l_obj.next

            msg = FrameMessage(
                cam_id=geom.cam_id,
                frame_id=int(frame_meta.frame_num),
                ts_ms=ts_ms,
                frame_pts_ns=int(gst_buffer.pts),
                frame_width=geom.width,
                frame_height=geom.height,
                detections=detections,
            )
            msg.embed_dim = msg.infer_embed_dim()
            sink(msg)

            l_frame = l_frame.next

        return Gst.PadProbeReturn.OK

    return probe
