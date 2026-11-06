"""Công cụ duyệt/sửa ground-truth M6 chạy cục bộ — thay CVAT (quá nặng RAM trên máy dev).

    # một lần: trích các khung đã chú thích ra JPEG 960x540 (giải mã cả video, vài phút)
    PYTHONPATH=src python -m tools.gt_review extract
    # mỗi lần duyệt:
    PYTHONPATH=src python -m tools.gt_review serve      # mở http://127.0.0.1:8770

**Đọc** bản xuất "CVAT for video 1.1" ở `data/prelabel/export_draft/<buổi>/camXX.xml` (hoặc bản
đã sửa ở `data/cvat/<buổi>/`, nếu có). **Ghi** cả ba camera của buổi vào `data/cvat/<buổi>/`
sau mỗi thao tác — đúng chỗ `make lab-gt` đọc — kèm `edits.jsonl` (mọi thao tác, có giờ) để
báo cáo nói được người đã sửa những gì. Nút "Đã duyệt" ghi `REVIEWED` vào thư mục buổi.

Chỉ nghe ở 127.0.0.1: video có người thật (docs/m6/phieu-dong-thuan.md). Không cần GPU,
database hay worker nào; RAM chỉ là một tiến trình Python + trình duyệt.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from common.config import env
from common.logging import get_logger
from tools import gt_edit
from tools.gt_edit import CamDoc, EditError

log = get_logger("tools.gt_review")

HERE = Path(__file__).parent
CAMS = ("cam01", "cam02", "cam03")
UNDO_DEPTH = 50


class Workspace:
    """Các buổi đang mở + ngăn xếp hoàn tác. Một người dùng, khoá thô cho mọi thao tác."""

    def __init__(self, export_dir: Path, out_dir: Path, frames_dir: Path, review_md: Path | None):
        self.export_dir, self.out_dir, self.frames_dir = export_dir, out_dir, frames_dir
        self.review_md = review_md
        self.docs: dict[str, dict[str, CamDoc]] = {}
        self.undo: dict[str, list[dict[str, CamDoc]]] = {}
        self.lock = threading.Lock()

    def sessions(self) -> list[dict[str, Any]]:
        names = sorted(p.name for p in self.export_dir.iterdir() if p.is_dir())
        return [
            {
                "name": s,
                "edited": (self.out_dir / s / "edits.jsonl").exists(),
                "reviewed": (self.out_dir / s / "REVIEWED").exists(),
            }
            for s in names
        ]

    def session(self, s: str) -> dict[str, CamDoc]:
        if s not in self.docs:
            if not (self.export_dir / s).is_dir():
                raise HTTPException(404, f"không có buổi {s}")
            docs = {}
            for cam in CAMS:
                edited = self.out_dir / s / f"{cam}.xml"
                src = edited if edited.exists() else self.export_dir / s / f"{cam}.xml"
                docs[cam] = gt_edit.load(src, session=s, cam=cam)
            self.docs[s] = docs
            self.undo[s] = []
        return self.docs[s]

    def save(self, s: str, entry: dict[str, Any] | None) -> None:
        for cam, doc in self.docs[s].items():
            gt_edit.save(doc, self.out_dir / s / f"{cam}.xml")
        if entry is not None:
            with (self.out_dir / s / "edits.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}) + "\n")

    def notes(self, s: str) -> list[dict[str, Any]]:
        """Các dòng `- [ ]` của REVIEW.md thuộc ba task của buổi, kèm khung đầu tiên nhắc tới."""
        if self.review_md is None or not self.review_md.exists():
            return []
        out, cam = [], None
        for line in self.review_md.read_text(encoding="utf-8").splitlines():
            head = re.match(r"## (\S+)_(cam0\d)\b", line)
            if head:
                cam = head.group(2) if head.group(1) == s else None
                continue
            inline = re.search(rf"`{re.escape(s)}_(cam0\d)`", line)
            c = inline.group(1) if inline else cam
            if c and re.match(r"\s*- \[[ x]\]", line):
                m = re.search(r"khung (\d+)", line)
                text = re.sub(r"^\s*- \[[ x]\]\s*", "", line)
                out.append({"cam": c, "frame": int(m.group(1)) if m else 0, "text": text})
        return out


def doc_json(doc: CamDoc) -> dict[str, Any]:
    width = doc.meta.find(".//original_size/width")
    return {
        "size": doc.size,
        "step": doc.step,
        "width": int(width.text) if width is not None else 1920,
        "tracks": [
            {
                "tid": t.tid,
                "person": t.person,
                "source": t.source,
                "boxes": [
                    [
                        k,
                        round(b.x1, 1),
                        round(b.y1, 1),
                        round(b.x2, 1),
                        round(b.y2, 1),
                        int(b.keyframe),
                        int(b.occluded),
                    ]
                    for k, b in sorted(t.boxes.items())
                ],
            }
            for t in sorted(doc.tracks.values(), key=lambda t: t.tid)
        ],
    }


def problems(docs: dict[str, CamDoc]) -> list[str]:
    return [p for cam in CAMS for p in gt_edit.validate(docs[cam])]


def apply_op(docs: dict[str, CamDoc], op: dict[str, Any]) -> dict[str, Any]:
    """Một thao tác của giao diện → hàm của `gt_edit`. Trả thông tin thêm (vd id track mới)."""
    doc = docs[op["cam"]]
    kind = op["op"]
    box = tuple(float(v) for v in op["box"]) if "box" in op else None
    if kind == "person":
        gt_edit.set_person(doc, op["tid"], op["person"])
    elif kind == "delete":
        gt_edit.delete_track(doc, op["tid"])
    elif kind == "split":
        return {"tid": gt_edit.split(doc, op["tid"], op["k"], op.get("person"))}
    elif kind == "hide":
        gt_edit.hide(doc, op["tid"], op["a"], op["b"])
    elif kind == "occluded":
        gt_edit.set_occluded(doc, op["tid"], op["a"], op["b"], bool(op["value"]))
    elif kind == "box":
        gt_edit.set_box(doc, op["tid"], op["k"], box, link=bool(op.get("link")))  # type: ignore[arg-type]
    elif kind == "new":
        return {"tid": gt_edit.new_track(doc, op["k"], box, op.get("person", ""))}  # type: ignore[arg-type]
    elif kind == "merge":
        gt_edit.merge(doc, op["tid"], op["other"])
    else:
        raise EditError(f"thao tác lạ: {kind}")
    return {}


def create_app(ws: Workspace) -> FastAPI:
    app = FastAPI(title="GT review M6")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (HERE / "gt_review_ui" / "index.html").read_text(encoding="utf-8")

    @app.get("/api/sessions")
    def sessions() -> list[dict[str, Any]]:
        return ws.sessions()

    def state(s: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        docs = ws.session(s)
        return {
            "session": s,
            "cams": {cam: doc_json(docs[cam]) for cam in CAMS},
            "problems": problems(docs),
            "undo": len(ws.undo[s]),
            **(extra or {}),
        }

    @app.get("/api/session/{s}")
    def get_session(s: str) -> dict[str, Any]:
        with ws.lock:
            return state(s)

    @app.get("/api/session/{s}/notes")
    def notes(s: str) -> list[dict[str, Any]]:
        return ws.notes(s)

    @app.post("/api/session/{s}/op")
    def op(s: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        with ws.lock:
            docs = ws.session(s)
            before = {cam: copy.deepcopy(doc) for cam, doc in docs.items()}
            try:
                extra = apply_op(docs, body)
            except (EditError, KeyError, TypeError, ValueError) as exc:
                ws.docs[s] = before
                raise HTTPException(400, str(exc)) from exc
            ws.undo[s] = (ws.undo[s] + [before])[-UNDO_DEPTH:]
            ws.save(s, {**body, **extra})
            return state(s, extra)

    @app.post("/api/session/{s}/undo")
    def undo(s: str) -> dict[str, Any]:
        with ws.lock:
            ws.session(s)
            if not ws.undo[s]:
                raise HTTPException(400, "không còn gì để hoàn tác")
            ws.docs[s] = ws.undo[s].pop()
            ws.save(s, {"op": "undo"})
            return state(s)

    @app.post("/api/session/{s}/reviewed")
    def reviewed(s: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        with ws.lock:
            docs = ws.session(s)
            if body.get("value", True):
                if problems(docs):
                    raise HTTPException(
                        400, "còn lỗi trong tab Lỗi — sửa hết rồi mới đánh dấu đã duyệt"
                    )
                ws.save(s, {"op": "reviewed"})
                (ws.out_dir / s / "REVIEWED").write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
            else:
                (ws.out_dir / s / "REVIEWED").unlink(missing_ok=True)
            return state(s)

    @app.get("/frames/{s}/{cam}/{k}")
    def frame(s: str, cam: str, k: int) -> FileResponse:
        if cam not in CAMS or not re.fullmatch(r"[\w-]+", s):
            raise HTTPException(404)
        path = ws.frames_dir / s / cam / f"{k:05d}.jpg"
        if not path.exists():
            raise HTTPException(404, f"chưa trích khung {s}/{cam}/{k} — chạy 'extract'")
        return FileResponse(path, headers={"Cache-Control": "max-age=86400"})

    return app


def extract(
    video_root: Path, frames_dir: Path, sessions: list[str], ffmpeg: str, step: int
) -> None:
    """Mọi khung task (khung video chia hết cho `step`) → JPEG 960x540, đánh số theo khung task."""
    for s in sessions:
        for cam in CAMS:
            out = frames_dir / s / cam
            if (out / ".done").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.monotonic()
            subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-threads",
                    "4",
                    "-i",
                    str(video_root / s / f"{cam}.mp4"),
                    "-vf",
                    f"select='not(mod(n\\,{step}))',scale=960:540",
                    "-fps_mode",
                    "passthrough",
                    "-q:v",
                    "5",
                    "-start_number",
                    "0",
                    "-y",
                    str(out / "%05d.jpg"),
                ],
                check=True,
            )
            n = len(list(out.glob("*.jpg")))
            (out / ".done").write_text(f"{n}\n")
            print(f"{s}/{cam}: {n} khung, {time.monotonic() - t0:.0f} s", flush=True)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("command", choices=("serve", "extract"))
    p.add_argument("--export-dir", type=Path, default=Path("data/prelabel/export_draft"))
    p.add_argument("--out-dir", type=Path, default=Path("data/cvat"))
    p.add_argument("--frames-dir", type=Path, default=Path("data/gt_review/frames"))
    p.add_argument("--video-root", type=Path, default=Path("data/lab"))
    p.add_argument("--review", type=Path, default=Path("data/prelabel/REVIEW.md"))
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--ffmpeg", default=None, help="mặc định LAB_FFMPEG trong .env, rồi PATH")
    p.add_argument("sessions", nargs="*")
    args = p.parse_args(argv)

    if args.command == "extract":
        sessions = args.sessions or sorted(d.name for d in args.export_dir.iterdir() if d.is_dir())
        extract(
            args.video_root,
            args.frames_dir,
            sessions,
            args.ffmpeg or env("LAB_FFMPEG") or "ffmpeg",
            args.step,
        )
        return 0

    import uvicorn

    ws = Workspace(args.export_dir, args.out_dir, args.frames_dir, args.review)
    print(f"http://127.0.0.1:{args.port}  (ghi vào {args.out_dir})", flush=True)
    uvicorn.run(create_app(ws), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
