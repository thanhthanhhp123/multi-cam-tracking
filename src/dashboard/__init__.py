"""Giao diện giám sát: sơ đồ camera, vị trí đối tượng, tra cứu hành trình theo Global ID.

Hai nguồn dữ liệu, hai vai trò: Redis `mct:global` cho realtime, SQLite (`mct.store`) cho
lịch sử. Chi tiết ở `dashboard/app.py`.

    uvicorn dashboard.app:app --port 8000

QUY TẮC BẤT BIẾN: không import pyds, gi, tensorrt, pycuda hay cuda (CLAUDE.md §2).
"""
