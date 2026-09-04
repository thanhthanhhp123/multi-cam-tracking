"""Test `tools/reembed_fixture.py`.

Cả giá trị của thí nghiệm nằm ở chỗ hai fixture sinh ra **chỉ khác nhau đúng một biến**.
Nếu chúng lệch thêm ở tập detection, ở `local_track_id` hay ở `ts_ms` thì chênh lệch
embedding đo được không quy về nguyên nhân nào — nên đó chính là thứ test này canh.
"""

from __future__ import annotations

import numpy as np
import pytest

from common.schema import Detection, FrameMessage
from tools.ds_wildtrack_gt import gt_index
from tools.reembed_fixture import attach_embeddings, rebuild_messages
from tools.wildtrack_to_fixture import RawDetection


def raw(frame_idx: int, view_idx: int, person_id: int, bbox) -> RawDetection:
    return RawDetection(
        frame_idx=frame_idx,
        frame_number=frame_idx * 5,
        view_idx=view_idx,
        person_id=person_id,
        bbox=bbox,
        world_xy=(0.0, 0.0),
    )


def msg(cam_id: str, frame_id: int, dets: list[Detection], ts_ms: int = 1000) -> FrameMessage:
    return FrameMessage(
        cam_id=cam_id,
        frame_id=frame_id,
        ts_ms=ts_ms,
        frame_pts_ns=frame_id * 500_000_000,
        frame_width=1920,
        frame_height=1080,
        detections=dets,
        embed_dim=0,
    )


def det(track_id: int, bbox, conf: float = 0.9) -> Detection:
    return Detection(local_track_id=track_id, bbox=bbox, confidence=conf)


class FakeEmbedder:
    """Trả về embedding phụ thuộc KÍCH THƯỚC crop, để test thấy được crop đã đổi hay chưa."""

    embed_dim = 4

    def __init__(self) -> None:
        self.batches: list[list[tuple[int, int]]] = []

    def embed(self, crops):
        self.batches.append([(c.shape[0], c.shape[1]) for c in crops])
        return np.array([[float(c.shape[0]), float(c.shape[1]), 1.0, 0.0] for c in crops])


# --------------------------------------------------------------------------------------
# Hai chế độ hộp phải cho ra cấu trúc giống hệt nhau
# --------------------------------------------------------------------------------------


def _one_frame():
    gt = gt_index([raw(0, 0, 11, (100.0, 100.0, 40.0, 90.0))])
    messages = [msg("cam01", 0, [det(5, (104.0, 96.0, 40.0, 90.0))], ts_ms=777)]
    return messages, gt


def test_hai_che_do_chi_khac_bbox():
    """Cùng detection, cùng local_track_id, cùng ts_ms — khác đúng toạ độ hộp."""
    messages, gt = _one_frame()
    a, _ = rebuild_messages(messages, gt, box_source="fixture", min_iou=0.5)
    b, _ = rebuild_messages(messages, gt, box_source="gt", min_iou=0.5)

    assert len(a) == len(b) == 1
    for x, y in zip(a, b, strict=True):
        assert (x.cam_id, x.frame_id, x.ts_ms) == (y.cam_id, y.frame_id, y.ts_ms)
        assert [d.local_track_id for d in x.detections] == [d.local_track_id for d in y.detections]
        assert [d.confidence for d in x.detections] == [d.confidence for d in y.detections]

    assert a[0].detections[0].bbox == (104.0, 96.0, 40.0, 90.0)  # hộp detector
    assert b[0].detections[0].bbox == (100.0, 100.0, 40.0, 90.0)  # hộp GT


def test_khong_sua_message_dau_vao():
    """Gọi hai lần trên cùng đầu vào phải cho hai kết quả độc lập."""
    messages, gt = _one_frame()
    goc = messages[0].detections[0].bbox
    rebuild_messages(messages, gt, box_source="gt", min_iou=0.5)
    assert messages[0].detections[0].bbox == goc


def test_detection_khong_khop_gt_bi_loai_o_ca_hai_che_do():
    """Detector bắt được thứ WildTrack không chú thích — giữ lại là làm hai fixture lệch tập."""
    gt = gt_index([raw(0, 0, 11, (100.0, 100.0, 40.0, 90.0))])
    messages = [
        msg("cam01", 0, [det(5, (104.0, 96.0, 40.0, 90.0)), det(6, (900.0, 900.0, 30.0, 60.0))])
    ]
    for source in ("fixture", "gt"):
        out, stats = rebuild_messages(messages, gt, box_source=source, min_iou=0.5)
        assert [d.local_track_id for d in out[0].detections] == [5]
        assert stats == {"n_detections": 2, "n_matched": 1}


def test_khung_khong_co_gt_thi_bo_han_message():
    gt = gt_index([raw(0, 0, 11, (100.0, 100.0, 40.0, 90.0))])
    messages = [msg("cam01", 9, [det(5, (104.0, 96.0, 40.0, 90.0))])]
    out, stats = rebuild_messages(messages, gt, box_source="fixture", min_iou=0.5)
    assert out == [] and stats["n_matched"] == 0


def test_bo_hop_source_la_bat_buoc_va_duoc_kiem():
    messages, gt = _one_frame()
    with pytest.raises(ValueError, match="--boxes"):
        rebuild_messages(messages, gt, box_source="ground_truth", min_iou=0.5)


# --------------------------------------------------------------------------------------
# Trích embedding
# --------------------------------------------------------------------------------------


def test_anh_xa_frame_id_theo_bang_khung_chu_thich(tmp_path):
    """`frame_id` i phải đọc ảnh `frame_numbers[i]`, không phải ảnh thứ i."""
    doc: list[str] = []

    def reader(path):
        doc.append(path.name)
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

    messages = [msg("cam01", 2, [det(1, (0.0, 0.0, 10.0, 20.0))])]
    attach_embeddings(
        messages,
        wildtrack_dir=tmp_path,
        frame_numbers=[0, 5, 10, 15],
        embedder=FakeEmbedder(),
        image_reader=reader,
    )
    assert doc == ["00000010.png"]


def test_cam_id_quyet_dinh_thu_muc_anh(tmp_path):
    duong_dan: list[str] = []

    def reader(path):
        duong_dan.append(str(path).replace("\\", "/"))
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

    messages = [msg("cam03", 0, [det(1, (0.0, 0.0, 10.0, 20.0))])]
    attach_embeddings(
        messages,
        wildtrack_dir=tmp_path,
        frame_numbers=[0],
        embedder=FakeEmbedder(),
        image_reader=reader,
    )
    assert duong_dan[0].endswith("Image_subsets/C3/00000000.png")


def test_crop_theo_dung_bbox_va_embedding_duoc_chuan_hoa(tmp_path):
    embedder = FakeEmbedder()
    messages = [msg("cam01", 0, [det(1, (10.0, 20.0, 30.0, 40.0))])]
    attach_embeddings(
        messages,
        wildtrack_dir=tmp_path,
        frame_numbers=[0],
        embedder=embedder,
        image_reader=lambda _p: np.zeros((1080, 1920, 3), dtype=np.uint8),
    )

    assert embedder.batches == [[(40, 30)]]  # (cao, rộng) đúng bằng h, w của bbox
    emb = messages[0].detections[0].embedding
    assert emb is not None
    assert float(np.linalg.norm(emb)) == pytest.approx(1.0, abs=1e-6)
    assert messages[0].embed_dim == 4


def test_moi_anh_chi_doc_mot_lan(tmp_path):
    """2800 khung x 3 MB PNG: đọc lại mỗi detection một lần là hỏng cả job."""
    dem = {"n": 0}

    def reader(_path):
        dem["n"] += 1
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

    messages = [msg("cam01", 0, [det(1, (0.0, 0.0, 10.0, 20.0)), det(2, (50.0, 50.0, 10.0, 20.0))])]
    attach_embeddings(
        messages,
        wildtrack_dir=tmp_path,
        frame_numbers=[0],
        embedder=FakeEmbedder(),
        image_reader=reader,
    )
    assert dem["n"] == 1


def test_frame_id_vuot_bang_khung_thi_bao_loi_ro_rang(tmp_path):
    messages = [msg("cam01", 5, [det(1, (0.0, 0.0, 10.0, 20.0))])]
    with pytest.raises(IndexError, match="frame_id"):
        attach_embeddings(
            messages,
            wildtrack_dir=tmp_path,
            frame_numbers=[0, 5],
            embedder=FakeEmbedder(),
            image_reader=lambda _p: np.zeros((1080, 1920, 3), dtype=np.uint8),
        )


def test_khong_doc_duoc_anh_thi_bao_loi_kem_duong_dan(tmp_path):
    messages = [msg("cam01", 0, [det(1, (0.0, 0.0, 10.0, 20.0))])]
    with pytest.raises(FileNotFoundError, match="00000000"):
        attach_embeddings(
            messages,
            wildtrack_dir=tmp_path,
            frame_numbers=[0],
            embedder=FakeEmbedder(),
            image_reader=lambda _p: None,
        )
