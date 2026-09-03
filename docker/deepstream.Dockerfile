# Pipeline DeepStream — CHỈ build/chạy trên máy Ubuntu + GPU NVIDIA (CLAUDE.md §2).
# Không build được trên Mac/Windows dev, không chạy được trên ut-hpc (không có
# DeepStream runtime — chỉ CUDA/TensorRT để train, xem CLAUDE.md §11).
#
# Image nền chính thức của NVIDIA, không cài native (rủi ro số 1, CLAUDE.md §11).
# Phiên bản chốt bằng cách chạy thật trên vast.ai RTX 3090 (2026-09-03) — xem
# docs/worklog/2026-09-03-3-m1-vast-deepstream.md.
FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

# pyds: binding Python cho DeepStream 7.1. Bản cp310 khớp Python 3.10 có sẵn trong
# image (KHÔNG dùng bản cp312 — đó là cho DeepStream 8.0/Ubuntu 24.04).
ARG PYDS_URL=https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_x86_64.whl

# DeepStream-Yolo (MIT license): custom bbox parser cho model YOLO của Ultralytics.
# Không có sẵn parser YOLO11 trong DeepStream gốc — đây là đường chuẩn của cộng đồng.
ARG DEEPSTREAM_YOLO_REF=master

WORKDIR /opt/nvidia/deepstream/deepstream
RUN ./user_additional_install.sh

RUN pip install --no-cache-dir "${PYDS_URL}" \
    && python3 -c "import pyds; print('pyds', pyds.__file__)"

WORKDIR /workspace
RUN git clone --depth 1 --branch "${DEEPSTREAM_YOLO_REF}" \
        https://github.com/marcoslucianops/DeepStream-Yolo.git

# Ultralytics chỉ cần để export ONNX (utils/export_yolo11.py) — không phải runtime infer,
# pipeline thật suy luận bằng TensorRT qua nvinfer. onnxscript cần cho torch.onnx export.
RUN pip install --no-cache-dir ultralytics onnx onnxslim onnxscript onnxruntime

# Build native lib parse bbox YOLO. CUDA_VER phải khớp CUDA trong image (kiểm tra lại
# nếu đổi base image — CLAUDE.md §11: DeepStream rất kén cặp driver/CUDA/TensorRT).
RUN CUDA_VER=12.6 make -C /workspace/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo -j"$(nproc)"

WORKDIR /workspace/mct-repo
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
# third_party trỏ tới lib đã build ở trên — khớp custom-lib-path trong
# configs/pipeline/config_infer_yolo11.txt (tương đối: ../../third_party/DeepStream-Yolo/...)
RUN mkdir -p third_party models/detector \
    && ln -sfn /workspace/DeepStream-Yolo third_party/DeepStream-Yolo

RUN pip install --no-cache-dir -e .

# Model YOLO (.onnx/.pt) và ReID không đi qua image — mount vào models/ lúc chạy
# (CLAUDE.md §11: trọng số không qua git, không qua image; xem docker/compose.gpu.yml).
# Bước dựng ONNX từ .pt tham khảo docs/worklog/2026-09-03-3-m1-vast-deepstream.md
# (utils/export_yolo11.py trong /workspace/DeepStream-Yolo).

ENTRYPOINT ["python3", "-m", "ds_pipeline"]
CMD ["--config", "configs/pipeline/streams.yaml"]
