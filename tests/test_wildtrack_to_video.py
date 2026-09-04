"""Test `tools/wildtrack_to_video.py` — bộ đóng ảnh WildTrack thành video cho DeepStream.

Không gọi ffmpeg thật: tiêm `runner` giả. Thứ đang được canh ở đây KHÔNG phải chất lượng
video mà là **thứ tự khung** — nếu khung thứ i của video không phải khung chú thích thứ i
thì `tools/ds_wildtrack_gt.py` gán ground-truth lệch hàng loạt, và lỗi đó không có triệu
chứng nào ngoài điểm số thấp một cách khó hiểu.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.wildtrack_to_video import (
    EncodeSpec,
    VideoBuildError,
    build_videos,
    probe_frame_count,
)

# Khung chú thích WildTrack cách nhau 5 — đúng chỗ mà một bộ đóng video ngây thơ
# (`%08d.png` liên tiếp) sẽ hỏng.
FRAME_NUMBERS = (0, 5, 10, 15)


@pytest.fixture
def wildtrack_dir(tmp_path: Path) -> Path:
    root = tmp_path / "wildtrack"
    ann = root / "annotations_positions"
    ann.mkdir(parents=True)
    for n in FRAME_NUMBERS:
        (ann / f"{n:08d}.json").write_text("[]", encoding="utf-8")
    for view in (1, 2):
        cam_dir = root / "Image_subsets" / f"C{view}"
        cam_dir.mkdir(parents=True)
        for n in FRAME_NUMBERS:
            (cam_dir / f"{n:08d}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([view, n]))
    return root


class FakeFfmpeg:
    """Ghi lại lệnh, tạo file đầu ra, và trả về số khung mà ffprobe "đếm được"."""

    def __init__(self, *, returncode: int = 0, n_frames: int | None = None) -> None:
        self.returncode = returncode
        self.n_frames = n_frames
        self.commands: list[list[str]] = []
        self.sequences: list[list[bytes]] = []

    def __call__(self, cmd) -> subprocess.CompletedProcess[str]:
        cmd = list(cmd)
        self.commands.append(cmd)
        if "ffprobe" in cmd[0]:
            n = len(self.sequences[-1]) if self.n_frames is None else self.n_frames
            return subprocess.CompletedProcess(cmd, 0, f"{n}\n", "")
        # Chụp lại nội dung thư mục chuỗi ảnh TRƯỚC khi build_videos xoá nó — đây là
        # bằng chứng duy nhất về thứ tự khung mà ffmpeg thật sự sẽ đọc.
        seq_dir = Path(cmd[cmd.index("-i") + 1]).parent
        self.sequences.append(
            [p.read_bytes() for p in sorted(seq_dir.iterdir(), key=lambda p: p.name)]
        )
        out = Path(cmd[-1])
        if self.returncode == 0:
            out.write_bytes(b"fake mp4")
        return subprocess.CompletedProcess(cmd, self.returncode, "", "loi gia dinh")


def test_thu_tu_khung_dung_bang_thu_tu_khung_chu_thich(wildtrack_dir: Path, tmp_path: Path):
    """Khung thứ i của chuỗi đưa cho ffmpeg phải là ảnh của khung chú thích thứ i.

    Đây là hợp đồng mà cả `ds_wildtrack_gt.py` lẫn `streams_wildtrack.yaml` dựa vào.
    """
    fake = FakeFfmpeg()
    build_videos(wildtrack_dir, tmp_path / "out", views=[1], runner=fake)

    goc = [
        (wildtrack_dir / "Image_subsets" / "C1" / f"{n:08d}.png").read_bytes()
        for n in FRAME_NUMBERS
    ]
    assert fake.sequences[0] == goc


def test_ten_file_lien_tiep_du_ten_goc_cach_nhau_5(wildtrack_dir: Path, tmp_path: Path):
    """ffmpeg đọc chuỗi bằng mẫu `%06d.png`, nên link phải được đánh số liên tiếp."""
    fake = FakeFfmpeg()
    build_videos(wildtrack_dir, tmp_path / "out", views=[1], runner=fake, keep_sequence=True)

    seq_dir = tmp_path / "out" / ".seq-cam01"
    assert sorted(p.name for p in seq_dir.iterdir()) == [
        f"{i:06d}.png" for i in range(len(FRAME_NUMBERS))
    ]


def test_index_ghi_du_hop_dong_anh_xa(wildtrack_dir: Path, tmp_path: Path):
    fake = FakeFfmpeg()
    result = build_videos(wildtrack_dir, tmp_path / "out", views=[1, 2], runner=fake)

    index = json.loads(result.index_path.read_text(encoding="utf-8"))
    assert index["frame_numbers"] == list(FRAME_NUMBERS)
    assert index["n_frames"] == len(FRAME_NUMBERS)
    assert set(index["cameras"]) == {"cam01", "cam02"}
    assert index["cameras"]["cam02"]["view"] == 2
    assert index["cameras"]["cam02"]["file"] == "cam02.mp4"
    assert index["width"] == 1920 and index["height"] == 1080


def test_encode_ghi_lai_tham_so_de_tai_lap(wildtrack_dir: Path, tmp_path: Path):
    """CRF/preset/fps phải nằm trong index.json: số đo chương 6 phải tái lập được."""
    fake = FakeFfmpeg()
    spec = EncodeSpec(fps=2.0, crf=15, preset="slow")
    result = build_videos(wildtrack_dir, tmp_path / "out", views=[1], runner=fake, spec=spec)

    index = json.loads(result.index_path.read_text(encoding="utf-8"))
    assert index["encoder"] == {
        "codec": "libx264",
        "crf": 15,
        "preset": "slow",
        "pix_fmt": "yuv420p",
        "fps": 2.0,
        "gop": 4,
    }
    cmd = fake.commands[0]
    assert cmd[cmd.index("-crf") + 1] == "15"
    assert cmd[cmd.index("-framerate") + 1] == "2"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"


def test_so_khung_lech_thi_bao_loi(wildtrack_dir: Path, tmp_path: Path):
    """ffmpeg báo thành công nhưng thiếu khung = mọi ánh xạ frame_id lệch. Phải chết ngay.

    Không có bước ffprobe này thì lỗi chỉ lộ ra ở cuối chuỗi, dưới dạng F1 thấp khó hiểu.
    """
    fake = FakeFfmpeg(n_frames=len(FRAME_NUMBERS) - 1)
    with pytest.raises(VideoBuildError, match="khung"):
        build_videos(wildtrack_dir, tmp_path / "out", views=[1], runner=fake)


def test_ffmpeg_loi_thi_bao_loi(wildtrack_dir: Path, tmp_path: Path):
    fake = FakeFfmpeg(returncode=1)
    with pytest.raises(VideoBuildError, match="ffmpeg"):
        build_videos(wildtrack_dir, tmp_path / "out", views=[1], runner=fake)


def test_thieu_anh_thi_bao_loi_kem_ten_file(wildtrack_dir: Path, tmp_path: Path):
    (wildtrack_dir / "Image_subsets" / "C1" / "00000010.png").unlink()
    with pytest.raises(FileNotFoundError, match=r"00000010\.png"):
        build_videos(wildtrack_dir, tmp_path / "out", views=[1], runner=FakeFfmpeg())


def test_thieu_annotation_thi_bao_cach_tai(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="fetch_wildtrack_annotations"):
        build_videos(tmp_path / "trong", tmp_path / "out", runner=FakeFfmpeg())


def test_views_ngoai_pham_vi_bi_tu_choi(wildtrack_dir: Path, tmp_path: Path):
    with pytest.raises(ValueError, match=r"1\.\.7"):
        build_videos(wildtrack_dir, tmp_path / "out", views=[8], runner=FakeFfmpeg())


def test_max_frames_cat_ngan_ca_video_lan_index(wildtrack_dir: Path, tmp_path: Path):
    fake = FakeFfmpeg()
    result = build_videos(wildtrack_dir, tmp_path / "out", views=[1], runner=fake, max_frames=2)
    assert result.frame_numbers == [0, 5]
    assert len(fake.sequences[0]) == 2


def test_don_dep_thu_muc_chuoi_anh(wildtrack_dir: Path, tmp_path: Path):
    build_videos(wildtrack_dir, tmp_path / "out", views=[1], runner=FakeFfmpeg())
    assert not (tmp_path / "out" / ".seq-cam01").exists()


def test_ffprobe_hong_thi_tra_ve_am_mot_chu_khong_chet(tmp_path: Path):
    """ffprobe thiếu trên máy không phải lý do để hỏng cả mẻ encode."""

    def runner(cmd):
        return subprocess.CompletedProcess(list(cmd), 1, "", "not found")

    assert probe_frame_count(tmp_path / "x.mp4", runner=runner) == -1


def test_gop_mac_dinh_bang_hai_giay():
    assert EncodeSpec(fps=2.0).keyint() == 4
    assert EncodeSpec(fps=25.0).keyint() == 50
    assert EncodeSpec(fps=2.0, gop=1).keyint() == 1
