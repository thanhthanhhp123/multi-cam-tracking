"""Định dạng MOT Challenge và bố cục thư mục TrackEval — hợp đồng dùng chung.

Cả bên SINH dữ liệu (`tools/cvat_to_mot.py` cho ground-truth, `tools/export_trackeval.py`
cho kết quả) lẫn bên CHẤM (`eval/run_trackeval.py`) đều đi qua đây. Hai bản cài đặt của
cùng một định dạng là cách chắc chắn nhất để có một bảng điểm sai mà không ai biết.

**Một dòng MOT** (dấu phẩy ngăn cách, KHÔNG khoảng trắng):

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

- `frame` **đếm từ 1**. `FrameMessage.frame_id` của đồ án đếm từ 0 → phải +1. Đây là lỗi
  lệch-một kinh điển, và nó không có triệu chứng: bảng điểm vẫn ra số, chỉ là số sai.
  Mọi chuyển đổi đi qua `to_mot_frame` / `from_mot_frame`, không cộng tay ở chỗ khác.
- `bb_left, bb_top` là góc trên-trái, `bb_width/height` là kích thước — trùng đúng quy ước
  `bbox = [x, y, w, h]` của `common/schema.py`, nên không phải đổi hệ toạ độ.
- Ground-truth: `conf=1`, rồi `class=1` (pedestrian) và `visibility` thay cho x/y/z —
  đúng thứ TrackEval đọc ở nhánh MotChallenge2DBox. `visibility` là **thuộc tính của hộp**
  (trường của `MotRow`), không phải tham số lúc ghi: nó phải sống sót qua `write_gt`, nếu
  không thì bên gọi có đặt cũng vô ích và mọi hộp bị che đều ra 1.00.
- Kết quả: `conf` là điểm tin cậy thật, ba cột cuối `-1`.

**Bố cục TrackEval** (nhánh MotChallenge2DBox) mà `build_layout` dựng:

    <root>/gt/<BENCH>-<SPLIT>/<SEQ>/gt/gt.txt
    <root>/gt/<BENCH>-<SPLIT>/<SEQ>/seqinfo.ini
    <root>/gt/seqmaps/<BENCH>-<SPLIT>.txt
    <root>/trackers/<BENCH>-<SPLIT>/<tracker>/data/<SEQ>.txt

**Đánh giá xuyên camera** dùng thủ thuật quen thuộc của AI City Challenge: nối các camera
thành MỘT chuỗi ảo bằng cách cộng offset vào `frame`, rồi lấy `global_id` làm `id`. Nhờ vậy
bộ chỉ số Identity/HOTA vốn viết cho một camera đo được luôn tính nhất quán xuyên camera.
Offset phải LỚN HƠN số khung của camera dài nhất, nếu không hai camera chồng khung lên nhau
và mọi chỉ số sai — `virtual_frame` kiểm điều đó thay vì tin người gọi.

Chỉ dùng stdlib. Không import gì cần GPU (CLAUDE.md §2 quy tắc bất biến 1).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Lớp `pedestrian` của MOT Challenge. Đồ án chỉ theo dõi người nên mọi dòng đều lớp này.
MOT_CLASS_PEDESTRIAN = 1


class MotFormatError(ValueError):
    """Dữ liệu không hợp lệ với định dạng MOT — sai ở đây là bảng điểm sai."""


def to_mot_frame(frame_id: int) -> int:
    """`frame_id` đếm từ 0 của schema → `frame` đếm từ 1 của MOT."""
    if frame_id < 0:
        raise MotFormatError(f"frame_id phải >= 0, nhận {frame_id}")
    return int(frame_id) + 1


def from_mot_frame(frame: int) -> int:
    """Nghịch đảo của `to_mot_frame`."""
    if frame < 1:
        raise MotFormatError(f"frame của MOT phải >= 1, nhận {frame}")
    return int(frame) - 1


def virtual_frame(cam_index: int, frame_id: int, *, offset: int) -> int:
    """Số khung trong chuỗi ảo nối nhiều camera, đã ở hệ đếm-từ-1 của MOT.

    `offset` phải lớn hơn số khung của camera dài nhất; bằng hoặc nhỏ hơn là hai camera
    dẫm lên khung của nhau và mọi chỉ số mất nghĩa.
    """
    if cam_index < 0:
        raise MotFormatError(f"cam_index phải >= 0, nhận {cam_index}")
    if offset <= 0:
        raise MotFormatError(f"offset phải > 0, nhận {offset}")
    if frame_id >= offset:
        raise MotFormatError(
            f"frame_id {frame_id} >= offset {offset}: camera này sẽ dẫm lên khung của "
            "camera kế tiếp trong chuỗi ảo"
        )
    return cam_index * offset + to_mot_frame(frame_id)


def frame_offset_for(n_frames_per_cam: Iterable[int]) -> int:
    """Offset an toàn: luỹ thừa 10 nhỏ nhất LỚN HƠN camera dài nhất, tối thiểu 1000.

    Luỹ thừa 10 để đọc log không phải nhẩm (khung ảo 3001 = camera thứ 3, khung 1). Nhưng
    phải BÁM SÁT độ dài thật: TrackEval duyệt mọi timestep của chuỗi ảo, kể cả khối rỗng,
    nên offset 100000 cho 400 khung/camera biến 2800 khung thật thành 700000 timestep —
    chạy một lần chấm mất ~1 phút thay vì vài giây, mà không thêm thông tin nào.
    """
    longest = max(list(n_frames_per_cam) or [0])
    offset = 1000
    while offset <= longest:
        offset *= 10
    return offset


@dataclass(slots=True, frozen=True)
class MotRow:
    """Một hộp trong một khung. `frame` đã ở hệ đếm-từ-1."""

    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0
    visibility: float = 1.0

    def as_gt_line(self) -> str:
        return (
            f"{self.frame},{self.track_id},{self.x:.2f},{self.y:.2f},"
            f"{self.w:.2f},{self.h:.2f},1,{MOT_CLASS_PEDESTRIAN},{self.visibility:.2f}"
        )

    def as_result_line(self) -> str:
        return (
            f"{self.frame},{self.track_id},{self.x:.2f},{self.y:.2f},"
            f"{self.w:.2f},{self.h:.2f},{self.confidence:.3f},-1,-1,-1"
        )


def _sorted(rows: Iterable[MotRow]) -> list[MotRow]:
    return sorted(rows, key=lambda r: (r.frame, r.track_id))


def write_gt(path: Path, rows: Iterable[MotRow]) -> int:
    """Ghi `gt.txt`. Trả về số dòng."""
    return _write(path, [r.as_gt_line() for r in _sorted(rows)])


def write_results(path: Path, rows: Iterable[MotRow]) -> int:
    """Ghi file kết quả của một tracker. Trả về số dòng."""
    return _write(path, [r.as_result_line() for r in _sorted(rows)])


def _write(path: Path, lines: Sequence[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": file này được đọc trên Linux (TrackEval), và máy dev là Windows.
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for line in lines:
            fh.write(line + "\n")
    return len(lines)


def parse_mot(path: Path) -> list[MotRow]:
    """Đọc lại file MOT. Bỏ qua dòng trống; dòng thiếu cột thì báo lỗi kèm số dòng."""
    rows: list[MotRow] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.split(",")
        if len(parts) < 6:
            raise MotFormatError(f"{path}:{n}: cần ít nhất 6 cột, có {len(parts)}")
        try:
            rows.append(
                MotRow(
                    frame=int(float(parts[0])),
                    track_id=int(float(parts[1])),
                    x=float(parts[2]),
                    y=float(parts[3]),
                    w=float(parts[4]),
                    h=float(parts[5]),
                    confidence=float(parts[6]) if len(parts) > 6 else 1.0,
                )
            )
        except ValueError as exc:
            raise MotFormatError(f"{path}:{n}: không đọc được số — {exc}") from exc
    return rows


def write_seqinfo(
    path: Path, *, name: str, width: int, height: int, length: int, fps: float
) -> None:
    """`seqinfo.ini` — TrackEval đọc `seqLength` để biết chuỗi dài bao nhiêu khung."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[Sequence]\n"
        f"name={name}\n"
        "imDir=img1\n"
        f"frameRate={fps:g}\n"
        f"seqLength={length}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        "imExt=.jpg\n",
        encoding="utf-8",
        newline="\n",
    )


def write_seqmap(path: Path, seq_names: Sequence[str]) -> None:
    """Seqmap của TrackEval: dòng đầu là tiêu đề `name`, rồi mỗi chuỗi một dòng."""
    _write(path, ["name", *seq_names])


@dataclass(slots=True, frozen=True)
class TrackEvalLayout:
    """Đường dẫn theo bố cục MotChallenge2DBox. Một chỗ định nghĩa, mọi nơi dùng lại."""

    root: Path
    benchmark: str
    split: str

    @property
    def dataset(self) -> str:
        return f"{self.benchmark}-{self.split}"

    @property
    def gt_folder(self) -> Path:
        return self.root / "gt"

    @property
    def trackers_folder(self) -> Path:
        return self.root / "trackers"

    def seq_dir(self, seq: str) -> Path:
        return self.gt_folder / self.dataset / seq

    def gt_file(self, seq: str) -> Path:
        return self.seq_dir(seq) / "gt" / "gt.txt"

    def seqinfo_file(self, seq: str) -> Path:
        return self.seq_dir(seq) / "seqinfo.ini"

    def seqmap_file(self) -> Path:
        return self.gt_folder / "seqmaps" / f"{self.dataset}.txt"

    def result_file(self, tracker: str, seq: str) -> Path:
        return self.trackers_folder / self.dataset / tracker / "data" / f"{seq}.txt"
