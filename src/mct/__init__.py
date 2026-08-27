"""Engine liên kết đa camera — đóng góp kỹ thuật chính của đồ án (M4, tuần 11-13).

Nhận metadata từ Redis Stream, gom thành tracklet, so khớp xuyên camera bằng đặc trưng
Re-ID kết hợp ràng buộc không-thời gian, gán Global ID. Thuật toán mô tả ở CLAUDE.md §6.

QUY TẮC BẤT BIẾN: package này KHÔNG được import pyds, gi, tensorrt, pycuda hay cuda —
nó phải phát triển và chạy được trên máy không GPU bằng fixture ghi sẵn. Xem CLAUDE.md §2.
"""
