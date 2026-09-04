# Image CPU cho engine liên kết (`python -m mct`) và dashboard.
#
# Cố tình KHÔNG dựa trên image DeepStream: hai thứ này không cần GPU, không cần CUDA,
# và phải chạy được cả trên máy dev lẫn trên máy GPU thuê theo giờ (CLAUDE.md §2 quy tắc 1).
# Image DeepStream nặng vài GB — dùng nó ở đây là trả tiền cho thứ không dùng tới.
#
# python:3.10-slim để khớp phiên bản Python trong container DeepStream 7.x (quy tắc 4):
# cùng một bộ mã chạy hai nơi thì đừng để hai phiên bản Python khác nhau.

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Cài dependency trước, copy mã sau: sửa mã thì không phải cài lại toàn bộ.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[dashboard]"

# `configs/` được mount lúc chạy (xem docker/compose.yml) để chỉnh ngưỡng mà không build lại.
COPY configs/ ./configs/

# SQLite nằm ở volume dùng chung; engine ghi, dashboard mở chỉ-đọc.
ENV MCT_DB_PATH=/data/mct.sqlite3
VOLUME ["/data"]

EXPOSE 8000

CMD ["python", "-m", "mct", "--config", "configs/mct.yaml", "--publish"]
