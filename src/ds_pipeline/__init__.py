"""Pipeline DeepStream chạy trên máy Ubuntu + GPU NVIDIA (M1-M3).

nvurisrcbin -> nvstreammux -> nvinfer (YOLO) -> nvtracker (+ ReID) -> pad probe -> Redis.

Đây là package DUY NHẤT được phép import pyds, gi, tensorrt, pycuda, cuda. Mọi thứ khác
phải chạy được trên máy không GPU (CLAUDE.md §2). Giữ ranh giới này sạch: probe chỉ nên
dựng đối tượng từ common.schema rồi publish, không nhét logic liên kết vào đây.

M1 xong (2026-09-03, xem docs/worklog/2026-09-03-3-m1-vast-deepstream.md): pipeline 1
camera chạy thật trên vast.ai (DeepStream 7.1, CUDA 12.6, TensorRT 10.3, RTX 3090),
YOLO11s weight gốc COCO (chưa fine-tune) -> nvtracker NvDCF -> probe in console.
410.8 FPS đo được (1080p, batch=1, FP16, không ReID).

Hai điểm CLAUDE.md §11 để ngỏ đã xác minh trên máy thật:
  1. DeepStream 7.1.0 / CUDA 12.6 / TensorRT 10.3.0.26 — chốt cho .env.
  2. Đường lấy Re-ID embedding: (A) — nvtracker NvDCF có sẵn khối `ReID:` trong config,
     `reidFeatureSize: 256`, tự L2-normalize (`addFeatureNormalization: 1`). Không cần
     SGIE thứ hai. Model resnet50_market1501 chưa có trong image, chưa bật (M3).

Còn thiếu cho M2/M3: fine-tune YOLO trên COCO-person (`ut-hpc`, xem
.claude/skills/ut-hpc/), tải model ReID, bật `reidType: 2` trong
configs/pipeline/config_tracker_NvDCF_perf.yml, ghi fixture thật qua Redis.
"""
