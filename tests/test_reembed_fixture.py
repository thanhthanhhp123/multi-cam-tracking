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
from tools.reembed_fixture import (
    EmbeddingCache,
    attach_embeddings,
    rebuild_messages,
    relabel_with_gt_ids,
)
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
# Oracle tracker: local_track_id := personID, mọi thứ khác giữ nguyên
# --------------------------------------------------------------------------------------


def _fragmented_and_switched():
    """Người 11 bị NvDCF vỡ thành id 5 rồi id 9; id 7 lúc đầu bám người 12 rồi nhảy sang 13."""
    gt = gt_index(
        [
            raw(0, 0, 11, (100.0, 100.0, 40.0, 90.0)),
            raw(0, 0, 12, (400.0, 100.0, 40.0, 90.0)),
            raw(1, 0, 11, (110.0, 100.0, 40.0, 90.0)),
            raw(1, 0, 13, (400.0, 100.0, 40.0, 90.0)),
        ]
    )
    e = np.array([1.0, 0.0], dtype=np.float32)
    messages = [
        msg(
            "cam01",
            0,
            [
                Detection(5, (100.0, 100.0, 40.0, 90.0), 0.9, e),
                Detection(7, (400.0, 100.0, 40.0, 90.0), 0.8, e),
            ],
            ts_ms=1000,
        ),
        msg(
            "cam01",
            1,
            [
                Detection(9, (110.0, 100.0, 40.0, 90.0), 0.7, e),
                Detection(7, (400.0, 100.0, 40.0, 90.0), 0.6, e),
            ],
            ts_ms=1500,
        ),
    ]
    return messages, gt


def test_oracle_gop_id_bi_vo_va_tach_id_bi_tron():
    messages, gt = _fragmented_and_switched()
    out = relabel_with_gt_ids(messages, gt, min_iou=0.5)

    ids = [[d.local_track_id for d in m.detections] for m in out]
    assert ids == [[11, 12], [11, 13]]
    # NvDCF: id 5 và 9 là MỘT người (vỡ) -> gộp về 11; id 7 là HAI người (trộn) -> tách 12/13.
    assert len({i for frame in ids for i in frame}) == 3


def test_oracle_chi_doi_local_track_id():
    messages, gt = _fragmented_and_switched()
    out = relabel_with_gt_ids(messages, gt, min_iou=0.5)

    for x, y in zip(messages, out, strict=True):
        assert (x.cam_id, x.frame_id, x.ts_ms, x.frame_pts_ns) == (
            y.cam_id,
            y.frame_id,
            y.ts_ms,
            y.frame_pts_ns,
        )
        assert len(x.detections) == len(y.detections)
        for a, b in zip(x.detections, y.detections, strict=True):
            assert a.bbox == b.bbox
            assert a.confidence == b.confidence
            assert a.embedding is not None and b.embedding is not None
            assert np.array_equal(a.embedding, b.embedding)


def test_oracle_khong_sua_message_dau_vao():
    messages, gt = _fragmented_and_switched()
    relabel_with_gt_ids(messages, gt, min_iou=0.5)
    assert [d.local_track_id for m in messages for d in m.detections] == [5, 7, 9, 7]


def test_oracle_khong_cap_trung_id_trong_mot_khung():
    """MOT Challenge cấm hai hộp cùng id trong một khung — ghép một-một phải đảm bảo điều đó."""
    messages, gt = _fragmented_and_switched()
    for m in relabel_with_gt_ids(messages, gt, min_iou=0.5):
        ids = [d.local_track_id for d in m.detections]
        assert len(ids) == len(set(ids))


def test_oracle_bao_loi_khi_detection_khong_ghep_duoc_nguoi_nao():
    gt = gt_index([raw(0, 0, 11, (100.0, 100.0, 40.0, 90.0))])
    messages = [
        msg("cam01", 0, [det(5, (100.0, 100.0, 40.0, 90.0)), det(6, (900.0, 900.0, 30.0, 60.0))])
    ]
    with pytest.raises(ValueError, match="không ghép được"):
        relabel_with_gt_ids(messages, gt, min_iou=0.5)


def test_oracle_sau_rebuild_giu_dung_tap_detection_cua_fixture_tracker_that():
    """Đường dùng thật: rebuild (lọc + hộp GT) rồi relabel — hai bên phải cùng tập detection."""
    messages, gt = _fragmented_and_switched()
    c, _ = rebuild_messages(messages, gt, box_source="gt", min_iou=0.5)
    b = relabel_with_gt_ids(c, gt, min_iou=0.5)

    assert [(m.cam_id, m.frame_id, len(m.detections)) for m in c] == [
        (m.cam_id, m.frame_id, len(m.detections)) for m in b
    ]
    assert [d.bbox for m in c for d in m.detections] == [d.bbox for m in b for d in m.detections]


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


# --------------------------------------------------------------------------------------
# Cache embedding
# --------------------------------------------------------------------------------------


def _blank(_path):
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def test_cache_trung_thi_khong_goi_embedder_va_khong_doc_anh(tmp_path):
    cache = EmbeddingCache(None, model_tag="m")
    cache.put("cam01", 0, (10.0, 20.0, 30.0, 40.0), np.array([0.6, 0.8], dtype=np.float32))

    messages = [msg("cam01", 0, [det(1, (10.0, 20.0, 30.0, 40.0))])]
    embedder = FakeEmbedder()
    doc = {"n": 0}

    def reader(_p):
        doc["n"] += 1
        return _blank(_p)

    attach_embeddings(
        messages,
        wildtrack_dir=tmp_path,
        frame_numbers=[0],
        embedder=embedder,
        image_reader=reader,
        cache=cache,
    )
    assert embedder.batches == [] and doc["n"] == 0
    assert messages[0].detections[0].embedding.tolist() == pytest.approx([0.6, 0.8])
    assert (cache.hits, cache.misses) == (1, 0)


def test_cache_chi_embed_phan_con_thieu(tmp_path):
    cache = EmbeddingCache(None, model_tag="m")
    cache.put("cam01", 0, (0.0, 0.0, 10.0, 20.0), np.array([1.0, 0.0], dtype=np.float32))

    messages = [msg("cam01", 0, [det(1, (0.0, 0.0, 10.0, 20.0)), det(2, (50.0, 50.0, 12.0, 24.0))])]
    embedder = FakeEmbedder()
    attach_embeddings(
        messages,
        wildtrack_dir=tmp_path,
        frame_numbers=[0],
        embedder=embedder,
        image_reader=_blank,
        cache=cache,
    )
    assert embedder.batches == [[(24, 12)]]  # chỉ crop của detection thứ hai
    assert (cache.hits, cache.misses) == (1, 1)
    assert cache.get("cam01", 0, (50.0, 50.0, 12.0, 24.0)) is not None  # miss được nhớ lại


def test_cache_khu_hoi_npz_giu_nguyen_embedding(tmp_path):
    path = tmp_path / "c.npz"
    a = EmbeddingCache(path, model_tag="osnet.onnx")
    a.put("cam03", 7, (1.0, 2.0, 3.0, 4.0), np.array([0.0, 1.0], dtype=np.float32))
    a.save()

    b = EmbeddingCache(path, model_tag="osnet.onnx")
    assert len(b) == 1
    assert b.get("cam03", 7, (1.0, 2.0, 3.0, 4.0)).tolist() == [0.0, 1.0]
    assert b.get("cam03", 8, (1.0, 2.0, 3.0, 4.0)) is None  # khác khung thì không trúng


def test_cache_cua_model_khac_bi_tu_choi(tmp_path):
    path = tmp_path / "c.npz"
    a = EmbeddingCache(path, model_tag="osnet_a.onnx")
    a.put("cam01", 0, (0.0, 0.0, 1.0, 1.0), np.array([1.0], dtype=np.float32))
    a.save()
    with pytest.raises(ValueError, match="model"):
        EmbeddingCache(path, model_tag="osnet_b.onnx")


def test_cache_ket_qua_giong_het_khong_cache(tmp_path):
    """Có cache hay không thì fixture ra phải như nhau — đó là điều làm cache an toàn."""
    boxes = [(0.0, 0.0, 10.0, 20.0), (50.0, 50.0, 12.0, 24.0)]

    def run(cache):
        messages = [msg("cam01", 0, [det(i, b) for i, b in enumerate(boxes, start=1)])]
        attach_embeddings(
            messages,
            wildtrack_dir=tmp_path,
            frame_numbers=[0],
            embedder=FakeEmbedder(),
            image_reader=_blank,
            cache=cache,
        )
        return [d.embedding for d in messages[0].detections]

    warm = EmbeddingCache(None, model_tag="m")
    run(warm)  # lần đầu: điền cache
    for x, y in zip(run(None), run(warm), strict=True):
        assert np.array_equal(x, y)
