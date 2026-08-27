"""Code dùng chung giữa pipeline DeepStream và engine liên kết đa camera.

QUY TẮC BẤT BIẾN: package này KHÔNG được import pyds, gi, tensorrt, pycuda hay cuda.
Nó phải chạy được trên máy không có GPU. Xem CLAUDE.md §2.
"""
