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
import yaml

# Lớp person trong labels.txt của COCO — trùng CLASS_PERSON mà probes.py lọc lần hai.
PERSON_CLASS_ID = 0

# Một file config nvinfer cho mỗi batch size: engine TensorRT gắn chặt với batch, và
# nvinfer chỉ nạp lại engine đã build khi `model-engine-file` trùng đúng tên.
PGIE_BATCHES = {
    "config_infer_yolo11.txt": 1,
    "config_infer_yolo11_b4.txt": 4,
    "config_infer_yolo11_b7.txt": 7,
}
PGIE_CONFIGS = sorted(PGIE_BATCHES)


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


def test_moi_ban_config_chi_khac_batch_va_ten_engine(pipeline_dir: Path) -> None:
    """Mọi bản phải trỏ cùng ONNX/labels/parser — chỉ khác batch size và tên engine.

    Lệch một trong những khoá đó là đo FPS 4 hay 7 luồng trên một model khác với 1 luồng,
    và bảng số liệu chương 6 mất giá trị so sánh.
    """
    goc = _read(pipeline_dir / "config_infer_yolo11.txt")["property"]

    for ten, batch in sorted(PGIE_BATCHES.items()):
        prop = _read(pipeline_dir / ten)["property"]
        for khoa in (
            "onnx-file",
            "labelfile-path",
            "custom-lib-path",
            "parse-bbox-func-name",
            "network-mode",
            "num-detected-classes",
        ):
            assert goc[khoa] == prop[khoa], f"{khoa} lệch ở {ten}"

        assert prop["batch-size"] == str(batch), ten
        # Tên engine PHẢI mang đúng batch: đây là thứ quyết định engine có được nạp lại
        # hay build lại ~4 phút mỗi lần chạy.
        assert f"_b{batch}_" in prop["model-engine-file"], ten


# --------------------------------------------------------------------------------------
# Tracker + ReID (M3)
# --------------------------------------------------------------------------------------
#
# `config_tracker_*.yml` mở đầu bằng chỉ thị `%YAML:1.0` của OpenCV FileStorage, không
# phải YAML hợp lệ — PyYAML từ chối. Cắt bỏ dòng chỉ thị rồi parse phần còn lại (đúng là
# YAML thuần).

TRACKER_PERF = "config_tracker_NvDCF_perf.yml"
TRACKER_REID = "config_tracker_NvDCF_reid.yml"

# osnet_x1_0 trả feature 512 chiều (xác nhận trên chính file ONNX: output ('features',
# ['batch', 512])). Model resnet50_market1501 kèm DeepStream là 256 — khác model, khác số.
OSNET_FEATURE_SIZE = 512


def _read_tracker(path: Path) -> dict:
    noi_dung = "\n".join(
        d for d in path.read_text(encoding="utf-8").splitlines() if not d.startswith("%")
    )
    return yaml.safe_load(noi_dung)


@pytest.fixture(scope="module")
def tracker_reid(pipeline_dir: Path) -> dict:
    return _read_tracker(pipeline_dir / TRACKER_REID)


def test_reid_duoc_bat_va_xuat_ra_metadata(tracker_reid: dict) -> None:
    """Thiếu `outputReidTensor` là lỗi kinh điển: tracker vẫn trích embedding và vẫn dùng
    nội bộ, nhưng KHÔNG gắn vào user meta — probe đọc ra None và cả pipeline chạy
    "thành công" mà không sinh ra dữ liệu nào cho src/mct."""
    reid = tracker_reid["ReID"]
    assert reid["reidType"] != 0
    assert reid["outputReidTensor"] == 1


def test_kich_thuoc_feature_khop_osnet(tracker_reid: dict) -> None:
    assert tracker_reid["ReID"]["reidFeatureSize"] == OSNET_FEATURE_SIZE


def test_tien_xu_ly_khop_duong_onnx_o_may_dev(tracker_reid: dict) -> None:
    """Hai đường sinh embedding phải cùng tiền xử lý, nếu không so cosine giữa chúng là vô nghĩa.

    Đường 1: pipeline DeepStream (file config này).
    Đường 2: `tools/reid_onnx.py` trên máy dev — thứ đã sinh fixture WildTrack và là chỗ
    ngưỡng `max_cost` trong configs/mct.yaml được chỉnh.
    """
    from tools import reid_onnx

    reid = tracker_reid["ReID"]

    assert reid["inferDims"] == [3, reid_onnx.INPUT_H, reid_onnx.INPUT_W]
    assert reid["inputOrder"] == 0  # NCHW, như batch của reid_onnx
    assert reid["colorFormat"] == 0  # RGB — reid_onnx đảo BGR->RGB trước khi chuẩn hoá

    # reid_onnx resize thẳng bằng cv2.resize, KHÔNG letterbox giữ tỉ lệ.
    assert reid["keepAspc"] == 0

    # y = netScaleFactor * (x - offset) với x là pixel 0..255, so với
    # (x/255 - mean) / std của reid_onnx  =>  offset = 255*mean, scale = 1/(255*std).
    mean = reid_onnx._IMAGENET_MEAN.reshape(-1)
    assert reid["offsets"] == pytest.approx((255.0 * mean).tolist(), abs=1e-3)

    # netScaleFactor là MỘT số vô hướng trong khi std của ImageNet khác nhau theo kênh —
    # buộc phải dùng std trung bình. Test ghim sai số đó lại để nó là lựa chọn có ý thức,
    # không phải thứ trôi đi lúc nào không hay.
    std_tb = float(reid_onnx._IMAGENET_STD.reshape(-1).mean())
    assert reid["netScaleFactor"] == pytest.approx(1.0 / (255.0 * std_tb), rel=2e-3)


def test_hai_config_tracker_chi_khac_khoi_reid(pipeline_dir: Path) -> None:
    """Chênh lệch FPS đo được giữa hai file phải quy về đúng một biến: ReID bật hay tắt."""
    perf = _read_tracker(pipeline_dir / TRACKER_PERF)
    reid = _read_tracker(pipeline_dir / TRACKER_REID)

    bo_qua = {"ReID", "TrajectoryManagement"}  # TrajectoryManagement chứa tham số re-assoc của ReID
    assert {k: v for k, v in perf.items() if k not in bo_qua} == {
        k: v for k, v in reid.items() if k not in bo_qua
    }


def test_streams_reid_chi_khac_streams_multi_o_config_tracker(pipeline_dir: Path) -> None:
    """Cặp đối chứng để đo chi phí ReID: mọi thứ khác phải giống hệt."""
    multi = yaml.safe_load((pipeline_dir / "streams_multi.yaml").read_text(encoding="utf-8"))
    reid = yaml.safe_load((pipeline_dir / "streams_reid.yaml").read_text(encoding="utf-8"))

    assert multi["sources"] == reid["sources"]
    assert multi["streammux"] == reid["streammux"]
    assert multi["pgie"] == reid["pgie"]
    assert multi["sink"] == reid["sink"]

    khac = {k for k in multi["tracker"] if multi["tracker"][k] != reid["tracker"][k]}
    assert khac == {"ll_config_file"}
    assert reid["tracker"]["ll_config_file"].endswith(TRACKER_REID)


# --------------------------------------------------------------------------------------
# WildTrack qua pipeline DeepStream (configs/demo/streams_wildtrack.yaml)
# --------------------------------------------------------------------------------------
#
# Cấu hình này tồn tại để trả lời câu hỏi treo của M3: ngưỡng `max_cost` chỉnh trên
# embedding của đường ONNX Runtime có chuyển sang đường DeepStream không. Nó chỉ trả lời
# được nếu hai bên cùng nhịp thời gian và cùng bộ camera — đó là thứ các test dưới canh.


@pytest.fixture(scope="module")
def streams_wildtrack(repo_root: Path) -> dict:
    path = repo_root / "configs" / "demo" / "streams_wildtrack.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_wildtrack_du_7_camera_va_cam_id_khop_ten_file(streams_wildtrack: dict) -> None:
    """cam_id phải khớp tên file video, nếu không là hoán vị camera — sai mà không lỗi.

    Hoán vị camera làm ràng buộc homography sai hoàn toàn trong khi pipeline chạy bình
    thường và điểm số chỉ tụt một cách khó hiểu.
    """
    sources = streams_wildtrack["sources"]
    assert [s["cam_id"] for s in sources] == [f"cam{i:02d}" for i in range(1, 8)]
    for s in sources:
        assert s["uri"].endswith(f"/{s['cam_id']}.mp4"), s


def test_wildtrack_phat_dung_toc_do_that(streams_wildtrack: dict) -> None:
    """`sink.sync` PHẢI true. Chạy hết tốc lực thì 400 khung dồn vào vài giây ts_ms, cửa
    sổ gán 1000 ms của src/mct nuốt trọn cả đoạn và ràng buộc thời gian mất nghĩa."""
    assert streams_wildtrack["sink"]["sync"] is True


def test_wildtrack_dung_engine_dung_batch(streams_wildtrack: dict) -> None:
    n_luong = len(streams_wildtrack["sources"])
    assert PGIE_BATCHES[Path(streams_wildtrack["pgie"]["config_file"]).name] == n_luong


def test_wildtrack_bat_reid(streams_wildtrack: dict) -> None:
    """Không có ReID thì cả thí nghiệm vô nghĩa: thứ đang đo chính là embedding."""
    assert streams_wildtrack["tracker"]["ll_config_file"].endswith(TRACKER_REID)


def test_wildtrack_streammux_giong_ban_4_luong(streams_wildtrack: dict, pipeline_dir: Path) -> None:
    """Cùng streammux với streams_reid.yaml: bbox scale ngược trong probe phụ thuộc kích
    thước này, và độ phân giải tracker quyết định chất lượng crop đưa vào ReID."""
    reid = yaml.safe_load((pipeline_dir / "streams_reid.yaml").read_text(encoding="utf-8"))
    assert streams_wildtrack["streammux"] == reid["streammux"]
    assert streams_wildtrack["tracker"]["width"] == reid["tracker"]["width"]
    assert streams_wildtrack["tracker"]["height"] == reid["tracker"]["height"]


def test_wildtrack_cam_id_khop_topology_demo(streams_wildtrack: dict, repo_root: Path) -> None:
    """cam_id phải trùng key trong wildtrack.topology.yaml, nếu không mọi tracklet rơi vào
    nhánh "camera lạ" và thành phần hình học bị bỏ qua im lặng (phiên 9, quyết định 3)."""
    topo = yaml.safe_load(
        (repo_root / "configs" / "demo" / "wildtrack.topology.yaml").read_text(encoding="utf-8")
    )
    assert {s["cam_id"] for s in streams_wildtrack["sources"]} == set(topo["cameras"])
