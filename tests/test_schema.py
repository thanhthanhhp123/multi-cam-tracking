"""Test contract dữ liệu giữa pipeline DeepStream và engine liên kết (CLAUDE.md §5)."""

from __future__ import annotations

import numpy as np
import pytest

from common.schema import (
    SCHEMA_VERSION,
    Detection,
    FrameMessage,
    SchemaError,
    decode_jsonl,
    decode_msgpack,
    encode_jsonl,
    encode_msgpack,
    l2_normalize,
    read_jsonl,
    validate,
    write_jsonl,
)

DIM = 32


def make_embedding(seed: int) -> np.ndarray:
    return l2_normalize(np.random.default_rng(seed).standard_normal(DIM))


def make_message(**overrides) -> FrameMessage:
    defaults = dict(
        cam_id="cam01",
        frame_id=7,
        ts_ms=1_788_231_600_000,
        frame_pts_ns=466_666_666,
        frame_width=1920,
        frame_height=1080,
        detections=[
            Detection(
                local_track_id=1,
                bbox=(100.0, 200.0, 80.0, 220.0),
                confidence=0.91,
                embedding=make_embedding(1),
            ),
            Detection(
                local_track_id=2,
                bbox=(900.0, 300.0, 70.0, 190.0),
                confidence=0.77,
                embedding=make_embedding(2),
            ),
        ],
        embed_dim=DIM,
    )
    defaults.update(overrides)
    return FrameMessage(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #


def test_msgpack_round_trip_giu_nguyen_du_lieu() -> None:
    msg = make_message()
    back = decode_msgpack(encode_msgpack(msg))

    assert back.cam_id == msg.cam_id
    assert back.frame_id == msg.frame_id
    assert back.ts_ms == msg.ts_ms
    assert back.frame_pts_ns == msg.frame_pts_ns
    assert back.embed_dim == DIM
    assert len(back.detections) == 2
    for orig, got in zip(msg.detections, back.detections, strict=True):
        assert got.local_track_id == orig.local_track_id
        assert got.bbox == pytest.approx(orig.bbox)
        assert got.confidence == pytest.approx(orig.confidence)
        # float32 qua raw bytes phải khớp BIT-CHÍNH-XÁC, không chỉ xấp xỉ.
        np.testing.assert_array_equal(got.embedding, orig.embedding)


def test_jsonl_round_trip_giu_nguyen_du_lieu() -> None:
    msg = make_message()
    back = decode_jsonl(encode_jsonl(msg))
    np.testing.assert_array_equal(back.detections[0].embedding, msg.detections[0].embedding)
    assert back.detections[1].bbox == pytest.approx(msg.detections[1].bbox)


def test_hai_dinh_dang_cho_ket_qua_giong_nhau() -> None:
    """msgpack (wire) và JSONL (fixture) phải không bao giờ lệch nhau."""
    msg = make_message()
    from_pack = decode_msgpack(encode_msgpack(msg))
    from_json = decode_jsonl(encode_jsonl(msg))

    assert from_pack.cam_id == from_json.cam_id
    assert from_pack.ts_ms == from_json.ts_ms
    for a, b in zip(from_pack.detections, from_json.detections, strict=True):
        assert a.bbox == pytest.approx(b.bbox)
        np.testing.assert_array_equal(a.embedding, b.embedding)


def test_jsonl_moi_message_dung_mot_dong() -> None:
    assert "\n" not in encode_jsonl(make_message())


def test_detection_khong_co_embedding() -> None:
    msg = make_message(
        detections=[Detection(local_track_id=5, bbox=(0.0, 0.0, 10.0, 20.0), confidence=0.5)],
        embed_dim=0,
    )
    assert decode_msgpack(encode_msgpack(msg)).detections[0].embedding is None


def test_write_va_read_jsonl(tmp_path) -> None:
    messages = [make_message(frame_id=i) for i in range(5)]
    path = tmp_path / "sub" / "out.jsonl"
    assert write_jsonl(path, messages) == 5
    assert [m.frame_id for m in read_jsonl(path)] == [0, 1, 2, 3, 4]


def test_schema_version_lech_thi_bao_loi() -> None:
    raw = encode_jsonl(make_message()).replace(
        f'"schema_version":{SCHEMA_VERSION}', f'"schema_version":{SCHEMA_VERSION + 99}'
    )
    with pytest.raises(SchemaError, match="schema_version"):
        decode_jsonl(raw)


# --------------------------------------------------------------------------- #
# validate()
# --------------------------------------------------------------------------- #


def test_message_hop_le_khong_co_van_de() -> None:
    assert validate(make_message()) == []


def test_bat_bbox_tran_khung() -> None:
    """Triệu chứng điển hình của việc quên scale toạ độ nvstreammux về độ phân giải camera."""
    msg = make_message(
        detections=[
            Detection(
                local_track_id=1,
                bbox=(1850.0, 900.0, 400.0, 500.0),
                confidence=0.9,
                embedding=make_embedding(1),
            )
        ]
    )
    issues = validate(msg)
    assert any("tràn khỏi khung" in i for i in issues)
    assert any("nvstreammux" in i for i in issues), "thông báo lỗi phải chỉ ra nguyên nhân khả dĩ"


def test_bat_embedding_chua_chuan_hoa() -> None:
    msg = make_message(
        detections=[
            Detection(
                local_track_id=1,
                bbox=(10.0, 10.0, 50.0, 100.0),
                confidence=0.9,
                embedding=(make_embedding(1) * 3.0).astype(np.float32),
            )
        ]
    )
    assert any("L2-normalize" in i for i in validate(msg))


def test_bat_local_track_id_trung_trong_mot_frame() -> None:
    det = Detection(local_track_id=1, bbox=(10.0, 10.0, 50.0, 100.0), confidence=0.9)
    msg = make_message(detections=[det, det], embed_dim=0)
    assert any("trùng" in i for i in validate(msg))


def test_bat_embed_dim_lech_header() -> None:
    msg = make_message(embed_dim=DIM + 1)
    assert any("lệch header" in i for i in validate(msg))


def test_strict_thi_nem_loi() -> None:
    msg = make_message(ts_ms=0)
    with pytest.raises(SchemaError):
        validate(msg, strict=True)


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #


def test_ground_point_la_day_giua_bbox() -> None:
    det = Detection(local_track_id=1, bbox=(100.0, 200.0, 80.0, 220.0), confidence=0.9)
    assert det.ground_point == (140.0, 420.0)


def test_l2_normalize() -> None:
    vec = l2_normalize(np.array([3.0, 4.0]))
    assert np.linalg.norm(vec) == pytest.approx(1.0)
    assert vec.dtype == np.float32


def test_l2_normalize_vector_khong() -> None:
    with pytest.raises(SchemaError):
        l2_normalize(np.zeros(8))
