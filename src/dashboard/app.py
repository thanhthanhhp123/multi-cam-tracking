"""Dashboard giám sát: sơ đồ camera, vị trí hiện tại, tra cứu hành trình theo Global ID.

    uvicorn dashboard.app:app --port 8000
    # hoặc: make dashboard

**Hai nguồn dữ liệu, hai vai trò khác nhau — không phải trùng lặp:**

  - **Redis `mct:global`** → *đang* xảy ra chuyện gì. Stream bị trim còn vài nghìn bản ghi,
    nên nó không nhớ được quá khứ và cũng không cần nhớ.
  - **SQLite (`mct.store`)** → *đã* xảy ra chuyện gì. Nguồn sự thật, mở ở chế độ chỉ-đọc
    vì engine mới là bên ghi.

Thiếu nguồn nào thì phần tương ứng tắt chứ trang không hỏng: chưa chạy engine thì không có
file SQLite (bảng hành trình báo "chưa có dữ liệu"); Redis chưa lên thì bản đồ trống và
`/health` nói rõ vì sao. Đây là trạng thái bình thường lúc phát triển, không phải lỗi.

**Không có GPU, không import pyds/gi/tensorrt** (CLAUDE.md §2 quy tắc 1).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from common.logging import get_logger
from common.streams import STREAM_FRAMES, STREAM_GLOBAL, connect
from dashboard.live import Hub, LiveState, RedisBridge
from mct.homography import HomographyMapper
from mct.store import Store
from mct.topology import Topology

log = get_logger("dashboard")

HERE = Path(__file__).parent
DEFAULT_DB = os.getenv("MCT_DB_PATH", "data/mct.db")
DEFAULT_TOPOLOGY = Path(os.getenv("MCT_TOPOLOGY", "configs/cameras/topology.yaml"))
DEFAULT_HOMOGRAPHY = Path(os.getenv("MCT_HOMOGRAPHY_DIR", "configs/cameras/homography"))

templates = Jinja2Templates(directory=str(HERE / "templates"))


class Dashboard:
    """Gom mọi phụ thuộc vào một chỗ để test tiêm được bản giả (không cần Redis/SQLite thật)."""

    def __init__(
        self,
        *,
        db_path: str = DEFAULT_DB,
        topology_path: Path | None = DEFAULT_TOPOLOGY,
        homography_dir: Path | None = DEFAULT_HOMOGRAPHY,
        redis_url: str | None = None,
    ) -> None:
        self.db_path = db_path
        self.topology = _load_topology(topology_path)
        self.mapper = _load_homography(homography_dir)
        self.state = LiveState(mapper=self.mapper)
        self.hub = Hub()
        self.bridge = RedisBridge(self.state, self.hub, url=redis_url)

    # ---------------------------------------------------------------- SQLite

    def store(self) -> Store:
        """Mở SQLite chỉ-đọc cho MỘT request rồi đóng.

        Mở theo request chứ không giữ một kết nối dùng chung: `sqlite3.Connection` không
        an toàn khi dùng chéo thread, mà FastAPI chạy route đồng bộ trong threadpool.
        Chi phí mở một file SQLite là không đáng kể so với việc phải đồng bộ hoá bằng tay.
        """
        return Store.open_readonly(self.db_path)

    @contextlib.contextmanager
    def open_store(self) -> Any:
        try:
            store = self.store()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            yield store
        finally:
            store.close()

    # ---------------------------------------------------------------- sơ đồ

    def layout(self) -> dict[str, Any]:
        """Sơ đồ camera cho bản đồ: vùng phủ mặt đất + quan hệ chồng lấn."""
        footprints = self.mapper.footprints() if self.mapper else {}
        cameras = []
        cam_ids = sorted(set(footprints) | set(self.topology.cameras if self.topology else []))
        for cam_id in cam_ids:
            spec = self.topology.cameras.get(cam_id) if self.topology else None
            polygon = footprints.get(cam_id, ())
            cameras.append(
                {
                    "cam_id": cam_id,
                    "name": getattr(spec, "name", "") or cam_id,
                    "calibrated": cam_id in footprints,
                    "footprint": [[round(x, 3), round(y, 3)] for x, y in polygon],
                    "overlaps_with": sorted(getattr(spec, "overlaps_with", ()) or ()),
                }
            )
        return {"cameras": cameras, "bounds": _bounds(footprints)}


def _load_topology(path: Path | None) -> Topology | None:
    if path is None or not Path(path).is_file():
        log.info("Không có %s — sơ đồ camera chỉ dựa vào file hiệu chỉnh", path)
        return None
    return Topology.load(path)


def _load_homography(directory: Path | None) -> HomographyMapper | None:
    if directory is None or not any(Path(directory).glob("*.yaml")):
        log.info("Không có file hiệu chỉnh trong %s — bản đồ mặt đất sẽ trống", directory)
        return None
    mapper = HomographyMapper.load(directory)
    log.info("Bản đồ mặt đất: %d camera đã hiệu chỉnh", len(mapper.calibrated))
    return mapper


def _bounds(footprints: dict[str, list[tuple[float, float]]]) -> dict[str, float] | None:
    """Khung nhìn mặc định của bản đồ = bao của mọi vùng phủ, nới thêm 10%."""
    points = [point for polygon in footprints.values() for point in polygon]
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad_x = max(1.0, 0.1 * (max(xs) - min(xs)))
    pad_y = max(1.0, 0.1 * (max(ys) - min(ys)))
    return {
        "min_x": round(min(xs) - pad_x, 3),
        "max_x": round(max(xs) + pad_x, 3),
        "min_y": round(min(ys) - pad_y, 3),
        "max_y": round(max(ys) + pad_y, 3),
    }


def create_app(dashboard: Dashboard | None = None) -> FastAPI:
    board = dashboard or Dashboard()

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await board.bridge.start()
        try:
            yield
        finally:
            await board.bridge.stop()

    app = FastAPI(title="Multi-Camera Tracking", version="0.5.0", lifespan=lifespan)
    app.state.dashboard = board
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    # ------------------------------------------------------------------ trang

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Any:
        return templates.TemplateResponse(request, "index.html", {"db_path": board.db_path})

    # ------------------------------------------------------------------ API

    @app.get("/health")
    def health() -> dict[str, Any]:
        status: dict[str, Any] = {
            "status": "ok",
            "redis": board.bridge.connected,
            "clients": board.hub.n_clients,
            "live_tracks": len(board.state.tracks),
            "calibrated_cameras": board.mapper.calibrated if board.mapper else [],
            "db": {"path": board.db_path, "exists": Path(board.db_path).is_file()},
        }
        if board.bridge.last_error:
            status["redis_error"] = board.bridge.last_error
        try:
            client = connect()
            client.ping()
            status["streams"] = {
                STREAM_FRAMES: client.xlen(STREAM_FRAMES),
                STREAM_GLOBAL: client.xlen(STREAM_GLOBAL),
            }
            client.close()
        except Exception as exc:
            status["status"] = "degraded"
            status["error"] = str(exc)
        return status

    @app.get("/api/layout")
    def layout() -> dict[str, Any]:
        """Sơ đồ camera (vùng phủ mặt đất) — client vẽ một lần lúc nạp trang."""
        return board.layout()

    @app.get("/api/live")
    def live() -> dict[str, Any]:
        """Ảnh chụp trạng thái hiện tại, cho client chưa kịp nhận cập nhật WebSocket nào."""
        return board.state.snapshot()

    @app.get("/api/tracks")
    def tracks(
        min_cameras: int = Query(1, ge=1, le=16),
        limit: int = Query(100, ge=1, le=1000),
    ) -> dict[str, Any]:
        """Global ID đã ghi xuống SQLite. `min_cameras=2` = chỉ người đi xuyên camera."""
        with board.open_store() as store:
            rows = store.cross_camera_tracks(min_cameras=min_cameras)[:limit]
            return {
                "summary": store.summary(),
                "tracks": [dict(row) for row in rows],
            }

    @app.get("/api/tracks/{global_id}")
    def trajectory(global_id: int) -> dict[str, Any]:
        """Hành trình đầy đủ của một Global ID: qua camera nào, lúc nào, bao lâu."""
        with board.open_store() as store:
            appearances = store.trajectory(global_id)
            if not appearances:
                raise HTTPException(status_code=404, detail=f"Không có Global ID {global_id}")
            return {
                "global_id": global_id,
                "n_cameras": len({a.cam_id for a in appearances}),
                "start_ms": min(a.start_ms for a in appearances),
                "end_ms": max(a.end_ms for a in appearances),
                "appearances": [
                    {
                        "cam_id": a.cam_id,
                        "local_track_id": a.local_track_id,
                        "tracklet_id": a.tracklet_id,
                        "start_ms": a.start_ms,
                        "end_ms": a.end_ms,
                        "duration_ms": a.duration_ms,
                        "n_frames": a.n_frames,
                        "cost": a.cost,
                        "reason": a.reason,
                    }
                    for a in appearances
                ],
            }

    # ------------------------------------------------------------------ WebSocket

    @app.websocket("/ws/live")
    async def ws_live(websocket: WebSocket) -> None:
        await websocket.accept()
        board.hub.add(websocket)
        try:
            # Gửi ngay toàn cảnh: nếu không, client mở trang lúc vắng người sẽ thấy bản đồ
            # trống cho tới khi có ai đó đi qua camera.
            await websocket.send_json(board.state.snapshot())
            while True:
                # Không chờ client gửi gì — chỉ dùng để phát hiện lúc nó đóng kết nối.
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # pragma: no cover - đường lỗi mạng
            log.debug("WebSocket đóng bất thường: %s", exc)
        finally:
            board.hub.remove(websocket)

    return app


app = create_app()
