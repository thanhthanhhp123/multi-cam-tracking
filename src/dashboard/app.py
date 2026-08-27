"""Khung dashboard tối thiểu — đủ để kiểm tra kết nối, chưa có nghiệp vụ (M5 sẽ làm tiếp).

uvicorn dashboard.app:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from common.logging import get_logger
from common.streams import STREAM_FRAMES, STREAM_GLOBAL, connect

log = get_logger("dashboard")

app = FastAPI(title="Multi-Camera Tracking", version="0.1.0")


@app.get("/health")
def health() -> dict[str, object]:
    """Kiểm tra Redis và độ dài hai stream — dùng để xác nhận replay đang chạy."""
    status: dict[str, object] = {"status": "ok", "redis": False}
    try:
        client = connect()
        client.ping()
        status["redis"] = True
        status["streams"] = {
            STREAM_FRAMES: client.xlen(STREAM_FRAMES),
            STREAM_GLOBAL: client.xlen(STREAM_GLOBAL),
        }
        client.close()
    except Exception as exc:
        status["status"] = "degraded"
        status["error"] = str(exc)
    return status


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (
        "<!doctype html><meta charset='utf-8'>"
        "<title>Multi-Camera Tracking</title>"
        "<h1>Multi-Camera Tracking</h1>"
        "<p>Khung dashboard (M0). Sơ đồ camera và hành trình Global ID sẽ làm ở M5.</p>"
        "<p><a href='/health'>/health</a> &middot; <a href='/docs'>/docs</a></p>"
    )
