"""Trích Re-ID embedding bằng OSNet chạy trên ONNX Runtime (CPU).

Dùng để trích embedding OFFLINE trên máy không GPU khi dựng fixture từ dataset có sẵn
(WildTrack...) — xem `tools/wildtrack_to_fixture.py`. Đây KHÔNG phải đường Re-ID của
pipeline thật: pipeline chạy engine TensorRT bên trong DeepStream (CLAUDE.md §11). Module
này chỉ là công cụ chuẩn bị dữ liệu, cố tình dùng chung tiền xử lý với OSNet để embedding
sinh ra ở đây gần với embedding pipeline sẽ tạo ra sau này.

`onnxruntime` và `cv2` chỉ được import khi thực sự khởi tạo embedder — module phải import
được trên môi trường chỉ có dependency lõi, vì `tests/test_no_gpu_imports.py` quét và import
mọi file trong `src/tools`. Cài thêm bằng: `pip install -e ".[reid]"`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Tiền xử lý mặc định của OSNet trong torchreid: resize (h=256, w=128), RGB, chia 255,
# rồi chuẩn hoá theo thống kê ImageNet. Sai bước này thì embedding lệch hẳn.
INPUT_H = 256
INPUT_W = 128
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class OsnetOnnxEmbedder:
    """Bọc một ONNX OSNet, nhận list crop BGR (như `cv2.imread` trả về) -> ma trận feature.

    Không tự L2-normalize: bên gọi chuẩn hoá bằng `common.schema.l2_normalize` để đi đúng
    một cửa với phần còn lại của contract.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        batch_size: int = 32,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.is_file():
            raise FileNotFoundError(
                f"Không thấy model ONNX: {self.onnx_path}. "
                "Xuất bằng `python -m tools.export_osnet_onnx`, hoặc trỏ --reid-onnx tới file khác."
            )
        if batch_size < 1:
            raise ValueError("batch_size phải >= 1")
        self.batch_size = int(batch_size)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=opts,
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        out_dims = self._session.get_outputs()[0].shape
        # Chiều cuối là số chiều embedding nếu model khai báo tĩnh; nếu động thì suy ra khi chạy.
        self.embed_dim = int(out_dims[-1]) if isinstance(out_dims[-1], int) else 0

    @staticmethod
    def _preprocess(crops_bgr: list[np.ndarray]) -> np.ndarray:
        import cv2

        batch = np.empty((len(crops_bgr), 3, INPUT_H, INPUT_W), dtype=np.float32)
        for i, crop in enumerate(crops_bgr):
            if crop.size == 0:
                raise ValueError("crop rỗng — bbox nằm ngoài ảnh hoặc chưa clip")
            resized = cv2.resize(crop, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
            rgb = np.ascontiguousarray(resized[:, :, ::-1], dtype=np.float32) / 255.0
            chw = np.transpose(rgb, (2, 0, 1))
            batch[i] = (chw - _IMAGENET_MEAN) / _IMAGENET_STD
        return batch

    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """(N crop BGR) -> ndarray (N, embed_dim) float32, CHƯA chuẩn hoá."""
        if not crops_bgr:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        chunks: list[np.ndarray] = []
        for start in range(0, len(crops_bgr), self.batch_size):
            batch = self._preprocess(crops_bgr[start : start + self.batch_size])
            out = self._session.run(None, {self._input_name: batch})[0]
            chunks.append(np.asarray(out, dtype=np.float32).reshape(len(batch), -1))

        feats = np.concatenate(chunks, axis=0)
        if self.embed_dim == 0:
            self.embed_dim = int(feats.shape[1])
        return feats
