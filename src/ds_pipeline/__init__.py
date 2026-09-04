"""Pipeline DeepStream chạy trên máy Ubuntu + GPU NVIDIA (M1-M3).

nvurisrcbin -> nvstreammux -> nvinfer (YOLO) -> nvtracker (+ ReID) -> pad probe -> Redis.

Đây là package DUY NHẤT được phép import pyds, gi, tensorrt, pycuda, cuda. Mọi thứ khác
phải chạy được trên máy không GPU (CLAUDE.md §2). Giữ ranh giới này sạch: probe chỉ nên
dựng đối tượng từ common.schema rồi publish, không nhét logic liên kết vào đây.

M1 xong (2026-09-03, xem docs/worklog/2026-09-03-3-m1-vast-deepstream.md): pipeline 1
camera chạy thật trên vast.ai (DeepStream 7.1, CUDA 12.6, TensorRT 10.3, RTX 3090),
YOLO11s weight gốc COCO -> nvtracker NvDCF -> probe in console.
618.5 FPS đo được (1080p, batch=1, FP16, không ReID, đo bằng --stats); 4 luồng 186.4
FPS/luồng.

Hai điểm CLAUDE.md §11 để ngỏ đã xác minh trên máy thật:
  1. DeepStream 7.1.0 / CUDA 12.6 / TensorRT 10.3.0.26 — chốt cho .env.
  2. Đường lấy Re-ID embedding: (A) — nvtracker NvDCF có sẵn khối `ReID:` trong config,
     tự L2-normalize (`addFeatureNormalization: 1`). Không cần SGIE thứ hai.

M2 xong (2026-09-04): detector chốt là **weight YOLO11s gốc COCO, không fine-tune**.
Lớp `person` của weight này đã được train trên chính COCO nên fine-tune lại trên tập con
COCO-person là học lại đúng dữ liệu cũ. Thứ thật sự cần — tracker chỉ bám người — lấy bằng
khối `[class-attrs-*]` trong configs/pipeline/config_infer_yolo11*.txt, không tốn GPU-hour.
Xem docs/worklog/2026-09-04-7-m2-detector-pretrained.md.

M3 soạn xong phần chạy được ngoài máy GPU (2026-09-04): model OSNet pretrained
(bản khái quát hoá miền, 512-d) ở models/reid/, config bật ReID ở
configs/pipeline/config_tracker_NvDCF_reid.yml, và đường đọc embedding từ user meta của
tracker ở ds_pipeline/reid_meta.py. KHÔNG fine-tune (cùng lý do như M2 — OSNet pretrained
đã học chính Market-1501/MSMT17); fine-tune dời sang M6, đóng khung như một ablation.

Còn thiếu cho M3, PHẢI làm trên vast-gpu: xác minh tracker nhận đúng các khoá ReID (log
khởi động), xác nhận probe đọc được embedding 512-d, đo chi phí FPS của ReID
(streams_reid.yaml vs streams_multi.yaml), rồi ghi fixture thật qua Redis với sink.sync
bật. Xem docs/worklog/2026-09-04-8-m3-reid-pretrained.md.
"""
