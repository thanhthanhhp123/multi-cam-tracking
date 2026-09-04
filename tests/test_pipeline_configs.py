"""Test các file cấu hình nvinfer trong `configs/pipeline/`.

Chạy được trên máy không GPU: chỉ đọc file `.txt` bằng `configparser`, không đụng tới
DeepStream. Lý do có bộ test này: từ M2 (2026-09-04) đồ án dùng THẲNG weight YOLO11s gốc
COCO 80 lớp thay vì fine-tune một head 1 lớp, nên việc "chỉ lấy người" nằm hoàn toàn trong
khối `[class-attrs-*]`. Xoá nhầm khối đó thì pipeline vẫn chạy, vẫn ra số FPS đẹp, chỉ có
điều nvtracker đi bám cả ô tô và ghế — một lỗi không có triệu chứng, đúng loại mà test
cấu hình sinh ra để chặn.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest

# Lớp person trong labels.txt của COCO — trùng CLASS_PERSON mà probes.py lọc lần hai.
PERSON_CLASS_ID = 0

PGIE_CONFIGS = ["config_infer_yolo11.txt", "config_infer_yolo11_b4.txt"]


@pytest.fixture(scope="module")
def pipeline_dir(repo_root: Path) -> Path:
    return repo_root / "configs" / "pipeline"


def _read(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
    parser.read(path, encoding="utf-8")
    return parser


@pytest.fixture(params=PGIE_CONFIGS)
def pgie(request: pytest.FixtureRequest, pipeline_dir: Path) -> configparser.ConfigParser:
    return _read(pipeline_dir / request.param)


def test_chi_lop_person_qua_duoc_nguong(pgie: configparser.ConfigParser) -> None:
    """Mặc định mọi lớp bị chặn, chỉ lớp 0 được hạ ngưỡng xuống mức dùng thật."""
    mac_dinh = pgie.getfloat("class-attrs-all", "pre-cluster-threshold")
    person = pgie.getfloat(f"class-attrs-{PERSON_CLASS_ID}", "pre-cluster-threshold")

    # confidence luôn thuộc [0, 1] nên ngưỡng >= 1.0 là không lớp nào vượt qua được.
    assert mac_dinh >= 1.0
    assert 0.0 < person < 1.0


def test_khong_co_lop_nao_khac_duoc_mo(pgie: configparser.ConfigParser) -> None:
    """Chặn việc lỡ thêm `[class-attrs-2]` (car) rồi quên mất."""
    duoc_phep = {"class-attrs-all", f"class-attrs-{PERSON_CLASS_ID}"}
    mo_them = [s for s in pgie.sections() if s.startswith("class-attrs-") and s not in duoc_phep]
    assert mo_them == []


def test_so_lop_khop_voi_labels(pgie: configparser.ConfigParser, repo_root: Path) -> None:
    """`num-detected-classes` mô tả đầu ra ONNX (80 lớp COCO), không phải số lớp ta cần.

    Bỏ qua nếu `models/` chưa có (gitignored — máy dev có thể chưa kéo weight về).
    """
    labels = repo_root / "models" / "detector" / "labels.txt"
    if not labels.exists():
        pytest.skip("models/detector/labels.txt chưa có trên máy này (gitignored)")

    ten_lop = [d for d in labels.read_text(encoding="utf-8").splitlines() if d.strip()]
    assert pgie.getint("property", "num-detected-classes") == len(ten_lop)
    assert ten_lop[PERSON_CLASS_ID] == "person"


def test_hai_ban_config_chi_khac_batch_va_ten_engine(pipeline_dir: Path) -> None:
    """b1 và b4 phải trỏ cùng ONNX/labels/parser — chỉ khác batch size và tên engine.

    Lệch một trong những khoá đó là đo FPS 4 luồng trên một model khác với 1 luồng, và
    bảng số liệu chương 6 mất giá trị so sánh.
    """
    b1 = _read(pipeline_dir / "config_infer_yolo11.txt")["property"]
    b4 = _read(pipeline_dir / "config_infer_yolo11_b4.txt")["property"]

    for khoa in (
        "onnx-file",
        "labelfile-path",
        "custom-lib-path",
        "parse-bbox-func-name",
        "network-mode",
        "num-detected-classes",
    ):
        assert b1[khoa] == b4[khoa], f"{khoa} lệch giữa hai bản config"

    assert b1["batch-size"] == "1"
    assert b4["batch-size"] == "4"
    assert "_b1_" in b1["model-engine-file"]
    assert "_b4_" in b4["model-engine-file"]
