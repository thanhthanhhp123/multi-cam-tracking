"""Pipeline DeepStream chạy trên máy Ubuntu + GPU NVIDIA (M1-M3).

nvurisrcbin -> nvstreammux -> nvinfer (YOLO) -> nvtracker (+ ReID) -> pad probe -> Redis.

Đây là package DUY NHẤT được phép import pyds, gi, tensorrt, pycuda, cuda. Mọi thứ khác
phải chạy được trên máy không GPU (CLAUDE.md §2). Giữ ranh giới này sạch: probe chỉ nên
dựng đối tượng từ common.schema rồi publish, không nhét logic liên kết vào đây.

CHƯA HIỆN THỰC — chờ máy GPU. Hai việc phải xác minh trên bản DeepStream thực tế trước
khi viết code (CLAUDE.md §11):
  1. phiên bản DeepStream tương thích với driver đang cài;
  2. đường lấy Re-ID embedding: ReID extractor tích hợp trong nvtracker (ưu tiên) hay
     SGIE nvinfer thứ hai (dự phòng) — tên meta type khác nhau giữa các phiên bản.
"""
