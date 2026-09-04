"""Lưu kết quả liên kết xuống SQLite — nguồn sự thật của hệ thống.

Redis `mct:global` là kênh realtime cho dashboard, sống vài nghìn bản ghi rồi bị trim.
Cái ở đây mới là thứ trả lời được "Global ID 47 đã đi qua những camera nào, lúc mấy giờ"
sau khi hệ thống chạy xong — tức là đầu vào của phần đánh giá (chương 6) và của màn tra
cứu hành trình trên dashboard (CLAUDE.md §3).

**Vì sao SQLite chứ không Postgres:** một tiến trình ghi, vài tiến trình đọc, dữ liệu cỡ
vài trăm nghìn dòng cho một buổi thu. Không có gì trong đó cần một server riêng, mà thêm
một dịch vụ nữa là thêm một thứ phải cài trên máy GPU thuê theo giờ.

**Hai bảng, không hơn:**

  - `global_tracks` — mỗi Global ID một dòng, trạng thái mới nhất.
  - `appearances` — mỗi (Global ID, tracklet) một dòng: người đó xuất hiện ở camera nào,
    từ lúc nào tới lúc nào. Khoá duy nhất là `tracklet_id` nên gán lại cùng một tracklet
    ở cửa sổ sau chỉ nới `end_ms` chứ không đẻ thêm dòng.

Ghi theo lô (`batch_size`, `flush_interval_ms` trong `configs/mct.yaml`): vòng gán chạy
mỗi giây và sinh hàng chục bản ghi, commit từng cái một là phí I/O vô ích.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.logging import get_logger
from mct.associator import Assignment

log = get_logger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS global_tracks (
    global_id    INTEGER PRIMARY KEY,
    created_ms   INTEGER NOT NULL,
    last_seen_ms INTEGER NOT NULL,
    last_cam_id  TEXT    NOT NULL,
    n_tracklets  INTEGER NOT NULL DEFAULT 0,
    n_cameras    INTEGER NOT NULL DEFAULT 1,
    closed       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS appearances (
    tracklet_id    INTEGER PRIMARY KEY,
    global_id      INTEGER NOT NULL,
    cam_id         TEXT    NOT NULL,
    local_track_id INTEGER NOT NULL,
    start_ms       INTEGER NOT NULL,
    end_ms         INTEGER NOT NULL,
    n_frames       INTEGER NOT NULL,
    cost           REAL    NOT NULL DEFAULT 0.0,
    reason         TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_appearances_global ON appearances(global_id, start_ms);
CREATE INDEX IF NOT EXISTS idx_appearances_cam    ON appearances(cam_id, start_ms);
CREATE INDEX IF NOT EXISTS idx_tracks_last_seen   ON global_tracks(last_seen_ms);
"""


@dataclass(slots=True, frozen=True)
class Appearance:
    """Một lượt xuất hiện của một Global ID tại một camera (một dòng `appearances`)."""

    global_id: int
    cam_id: str
    local_track_id: int
    tracklet_id: int
    start_ms: int
    end_ms: int
    n_frames: int
    cost: float = 0.0
    reason: str = ""

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(slots=True)
class StoreConfig:
    db_path: str = "mct.db"
    batch_size: int = 64
    flush_interval_ms: int = 1000

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> StoreConfig:
        store = dict(data.get("store", data) or {})
        defaults = cls()
        return cls(
            db_path=str(store.get("db_path", defaults.db_path)),
            batch_size=int(store.get("batch_size", defaults.batch_size)),
            flush_interval_ms=int(store.get("flush_interval_ms", defaults.flush_interval_ms)),
        )

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size phải >= 1, nhận {self.batch_size}")
        if self.flush_interval_ms <= 0:
            raise ValueError(f"flush_interval_ms phải > 0, nhận {self.flush_interval_ms}")


class Store:
    """Ghi/đọc kết quả liên kết. Một tiến trình ghi; dashboard mở riêng ở chế độ đọc."""

    def __init__(
        self,
        config: StoreConfig | None = None,
        *,
        db_path: str | Path | None = None,
        readonly: bool = False,
    ):
        self.config = config or StoreConfig()
        if db_path is not None:
            self.config.db_path = str(db_path)
        self.readonly = readonly

        path = self.config.db_path
        if readonly:
            # Dashboard chạy trong tiến trình KHÁC engine. Mở chỉ-đọc để chắc chắn nó
            # không bao giờ giành khoá ghi của engine, và để lỗi lộ ra ngay tại đây nếu
            # ai đó lỡ gọi nhầm hàm ghi.
            self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            self.conn.row_factory = sqlite3.Row
            self._pending: list[Assignment] = []
            self._last_flush_ms = 0
            self.n_written = 0
            return

        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        # WAL cho phép dashboard đọc trong khi engine đang ghi; NORMAL đủ an toàn ở đây
        # (mất vài bản ghi cuối khi mất điện không phải rủi ro của đồ án này).
        if path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(_SCHEMA)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

        self._pending: list[Assignment] = []
        self._last_flush_ms = 0
        self.n_written = 0

    # ------------------------------------------------------------------ ghi

    def record(self, assignment: Assignment, *, now_ms: int | None = None) -> None:
        """Xếp một kết quả gán vào hàng chờ. Tự flush khi đủ lô hoặc hết thời gian."""
        if self.readonly:
            raise RuntimeError("Store mở ở chế độ chỉ-đọc, không ghi được")
        self._pending.append(assignment)
        stamp = assignment.tracklet.end_ms if now_ms is None else int(now_ms)
        if self._last_flush_ms == 0:
            self._last_flush_ms = stamp
        if (
            len(self._pending) >= self.config.batch_size
            or stamp - self._last_flush_ms >= self.config.flush_interval_ms
        ):
            self.flush(now_ms=stamp)

    def record_many(self, assignments: Iterable[Assignment], *, now_ms: int | None = None) -> None:
        for assignment in assignments:
            self.record(assignment, now_ms=now_ms)

    def flush(self, *, now_ms: int | None = None) -> int:
        """Ghi hàng chờ xuống đĩa trong một transaction. Trả về số bản ghi đã ghi."""
        if not self._pending:
            return 0

        rows = [_appearance_row(a) for a in self._pending]
        with self.conn:  # transaction: hoặc vào hết, hoặc không vào gì
            self.conn.executemany(
                """
                INSERT INTO appearances
                    (tracklet_id, global_id, cam_id, local_track_id,
                     start_ms, end_ms, n_frames, cost, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tracklet_id) DO UPDATE SET
                    global_id = excluded.global_id,
                    end_ms    = MAX(appearances.end_ms, excluded.end_ms),
                    n_frames  = excluded.n_frames
                """,
                rows,
            )
            # Trạng thái GlobalTrack suy ra từ chính bảng appearances, không lưu trùng:
            # một nguồn sự thật thì không bao giờ lệch nhau.
            self.conn.executemany(
                """
                INSERT INTO global_tracks
                    (global_id, created_ms, last_seen_ms, last_cam_id, n_tracklets, n_cameras)
                VALUES (?, ?, ?, ?, 1, 1)
                ON CONFLICT(global_id) DO UPDATE SET
                    created_ms   = MIN(global_tracks.created_ms, excluded.created_ms),
                    last_seen_ms = MAX(global_tracks.last_seen_ms, excluded.last_seen_ms),
                    last_cam_id  = CASE
                        WHEN excluded.last_seen_ms >= global_tracks.last_seen_ms
                        THEN excluded.last_cam_id ELSE global_tracks.last_cam_id END
                """,
                [(r[1], r[4], r[5], r[2]) for r in rows],
            )
            touched = sorted({r[1] for r in rows})
            self.conn.executemany(
                """
                UPDATE global_tracks SET
                    n_tracklets = (SELECT COUNT(*) FROM appearances WHERE global_id = ?),
                    n_cameras   = (SELECT COUNT(DISTINCT cam_id) FROM appearances
                                   WHERE global_id = ?)
                WHERE global_id = ?
                """,
                [(gid, gid, gid) for gid in touched],
            )

        written = len(self._pending)
        self.n_written += written
        self._pending.clear()
        if now_ms is not None:
            self._last_flush_ms = int(now_ms)
        return written

    def close_tracks(self, global_ids: Sequence[int]) -> int:
        """Đánh dấu các Global ID đã đóng (GlobalTrack hết TTL trong gallery)."""
        if not global_ids:
            return 0
        with self.conn:
            cur = self.conn.executemany(
                "UPDATE global_tracks SET closed = 1 WHERE global_id = ?",
                [(int(gid),) for gid in global_ids],
            )
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else len(global_ids)

    # ------------------------------------------------------------------ đọc

    def trajectory(self, global_id: int) -> list[Appearance]:
        """Hành trình của một Global ID, theo thứ tự thời gian — màn tra cứu của dashboard."""
        rows = self.conn.execute(
            "SELECT * FROM appearances WHERE global_id = ? ORDER BY start_ms, cam_id",
            (int(global_id),),
        ).fetchall()
        return [_appearance(row) for row in rows]

    def appearances_between(self, start_ms: int, end_ms: int) -> list[Appearance]:
        rows = self.conn.execute(
            "SELECT * FROM appearances WHERE end_ms >= ? AND start_ms <= ? ORDER BY start_ms",
            (int(start_ms), int(end_ms)),
        ).fetchall()
        return [_appearance(row) for row in rows]

    def active_tracks(self, since_ms: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM global_tracks WHERE last_seen_ms >= ? AND closed = 0 "
            "ORDER BY last_seen_ms DESC",
            (int(since_ms),),
        ).fetchall()

    def cross_camera_tracks(self, *, min_cameras: int = 2) -> list[sqlite3.Row]:
        """Global ID đã đi qua từ `min_cameras` camera trở lên — kết quả đáng quan tâm nhất.

        Đây là thứ phân biệt hệ thống này với bốn bộ theo dõi đơn camera chạy song song.
        """
        return self.conn.execute(
            "SELECT * FROM global_tracks WHERE n_cameras >= ? ORDER BY n_cameras DESC, global_id",
            (int(min_cameras),),
        ).fetchall()

    def summary(self) -> dict[str, int]:
        row = self.conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM global_tracks)                     AS n_tracks,
                (SELECT COUNT(*) FROM appearances)                       AS n_appearances,
                (SELECT COUNT(DISTINCT cam_id) FROM appearances)         AS n_cameras,
                (SELECT COUNT(*) FROM global_tracks WHERE n_cameras > 1) AS n_cross_camera
            """
        ).fetchone()
        # `.keys()` là bắt buộc: lặp thẳng trên sqlite3.Row cho ra GIÁ TRỊ, không phải tên cột.
        return {key: int(row[key]) for key in row.keys()}  # noqa: SIM118

    # ------------------------------------------------------------------ vòng đời

    @classmethod
    def open_readonly(cls, db_path: str | Path) -> Store:
        """Mở một file đã có ở chế độ chỉ-đọc. `FileNotFoundError` nếu chưa có file.

        SQLite với `mode=ro` KHÔNG tạo file mới — đó chính là điều mong muốn: dashboard
        chỉ ra bảng rỗng khi engine chưa từng chạy thì gây hiểu nhầm hơn là báo thẳng.
        """
        path = Path(db_path)
        if not path.is_file():
            raise FileNotFoundError(f"{path} chưa tồn tại — engine (`python -m mct`) chưa chạy?")
        return cls(StoreConfig(db_path=str(path)), readonly=True)

    def close(self) -> None:
        if not self.readonly:
            self.flush()
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _appearance_row(assignment: Assignment) -> tuple:
    tracklet = assignment.tracklet
    return (
        int(tracklet.tracklet_id),
        int(assignment.global_id),
        tracklet.cam_id,
        int(tracklet.local_track_id),
        int(tracklet.start_ms),
        int(tracklet.end_ms),
        int(tracklet.n_frames),
        # Tracklet mới có cost = inf; SQLite không có kiểu đó, mà 0.0 lại dễ đọc nhầm
        # thành "khớp hoàn hảo" — ghi -1.0 để phân biệt rõ "không qua phép ghép nào".
        float(assignment.cost) if math.isfinite(assignment.cost) else -1.0,
        assignment.reason,
    )


def _appearance(row: sqlite3.Row) -> Appearance:
    return Appearance(
        global_id=int(row["global_id"]),
        cam_id=str(row["cam_id"]),
        local_track_id=int(row["local_track_id"]),
        tracklet_id=int(row["tracklet_id"]),
        start_ms=int(row["start_ms"]),
        end_ms=int(row["end_ms"]),
        n_frames=int(row["n_frames"]),
        cost=float(row["cost"]),
        reason=str(row["reason"]),
    )
