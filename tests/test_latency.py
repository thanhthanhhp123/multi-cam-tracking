"""Test cho hạ tầng đo độ trễ (`common/latency.py` + `tools/latency_report.py`).

Điểm cần ghim, theo thứ tự quan trọng:

1. Phần đo KHÔNG được đổi kết quả gán — cùng một fixture, bật hay tắt `--latency-log`
   thì Global ID phải y hệt. Nếu không giữ được điều này thì mọi số liệu chương 6 đo
   trước đây hết so sánh được.
2. Mốc `t0..t4` phải đi trọn chuỗi từ message tới bản ghi, kể cả khi thiếu mốc giữa
   (chạy từ fixture thì không có t2/t3a).
3. Đoạn ÂM phải đếm được chứ không bị nuốt: đó là cách duy nhất phát hiện lệch đồng hồ
   giữa máy GPU và máy engine.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from common.latency import (
    COARSE_SEGMENTS,
    END_TO_END,
    FINE_SEGMENTS,
    T0_CAPTURE,
    T0_IS_NTP,
    T1_PROBE,
    T1B_DEQUEUE,
    T2_XADD,
    T3_ASSOC,
    T3A_RECV,
    T3D_DB,
    T3W_WINDOW,
    T4_OUT,
    LatencyLog,
    LatencyRecord,
    capture_ms,
    entry_id_to_ms,
    is_plausible_ms,
    now_ms,
    percentile,
    read_records,
    summarize,
    t0_source_from,
)
from common.schema import (
    CLASS_PERSON,
    Detection,
    FrameMessage,
    decode_jsonl,
    decode_msgpack,
    encode_jsonl,
    encode_msgpack,
    l2_normalize,
)
from common.streams import FrameConsumer, QueuedFramePublisher
from mct.__main__ import Engine
from mct.affinity import AffinityConfig
from mct.associator import Associator
from mct.tracklet import TrackletConfig
from tools import latency_report

DIM = 16
BASE_TS = 1_700_000_000_000
BASE_WALL = 1_757_000_000_000.0


def _msg(cam_id: str, frame_id: int, tracks: dict[int, np.ndarray], *, stamps=None):
    return FrameMessage(
        cam_id=cam_id,
        frame_id=frame_id,
        ts_ms=BASE_TS + frame_id * 100,
        frame_pts_ns=frame_id * 100 * 1_000_000,
        frame_width=1920,
        frame_height=1080,
        detections=[
            Detection(
                local_track_id=tid,
                bbox=(100.0 + frame_id, 200.0, 60.0, 150.0),
                confidence=0.9,
                embedding=emb,
                class_id=CLASS_PERSON,
            )
            for tid, emb in tracks.items()
        ],
        embed_dim=DIM,
        stamps=dict(stamps or {}),
    )


def _producer_stamps(frame_id: int) -> dict[str, float]:
    """Mốc phía producer như probe + publisher sẽ gắn."""
    t0 = BASE_WALL + frame_id * 100.0
    return {
        T0_CAPTURE: t0,
        T0_IS_NTP: 1.0,
        T1_PROBE: t0 + 30.0,
        T1B_DEQUEUE: t0 + 35.0,
        T2_XADD: t0 + 40.0,
        T3A_RECV: t0 + 45.0,
    }


def _engine(**kwargs) -> Engine:
    return Engine(
        tracklet_config=TrackletConfig(min_frames=2, idle_timeout_ms=2_000),
        associator=Associator(config=AffinityConfig(max_cost=0.5)),
        window_ms=1_000,
        **kwargs,
    )


# --------------------------------------------------------------------------- mốc lẻ


def test_capture_ms_uu_tien_ntp():
    ntp_ns = int(BASE_WALL * 1e6)
    value, source = capture_ms(ntp_ns, BASE_WALL + 999.0)
    assert source == "ntp"
    assert value == pytest.approx(BASE_WALL)


def test_capture_ms_lui_ve_probe_khi_khong_co_ntp():
    for bad in (0, None, 12345, -1):
        value, source = capture_ms(bad, BASE_WALL)
        assert (value, source) == (BASE_WALL, "probe")


def test_t0_source_doc_lai_duoc_tu_stamps():
    assert t0_source_from({T0_CAPTURE: BASE_WALL, T0_IS_NTP: 1.0}) == "ntp"
    assert t0_source_from({T0_CAPTURE: BASE_WALL, T0_IS_NTP: 0.0}) == "probe"
    assert t0_source_from({T0_CAPTURE: BASE_WALL}) == "probe"
    assert t0_source_from({}) == ""


def test_is_plausible_ms_loai_gia_tri_rac():
    assert is_plausible_ms(BASE_WALL)
    assert not is_plausible_ms(0)
    assert not is_plausible_ms(None)
    assert not is_plausible_ms(float("nan"))
    # Giây thay vì mili-giây: lỗi kinh điển, phải bị loại chứ không lọt thành năm 1970.
    assert not is_plausible_ms(BASE_WALL / 1000.0)


def test_entry_id_cho_ra_thoi_diem_redis_ghi():
    assert entry_id_to_ms("1757000000000-0") == pytest.approx(1_757_000_000_000.0)
    assert entry_id_to_ms(b"1757000000000-7") == pytest.approx(1_757_000_000_000.0)
    assert entry_id_to_ms("khong-phai-so") is None
    assert entry_id_to_ms("42-0") is None  # quá nhỏ để là epoch ms


# --------------------------------------------------------------------------- schema


def test_stamps_di_qua_msgpack_va_jsonl():
    msg = _msg("cam01", 1, {1: l2_normalize(np.ones(DIM, dtype=np.float32))})
    msg.stamps = _producer_stamps(1)

    for decoded in (decode_msgpack(encode_msgpack(msg)), decode_jsonl(encode_jsonl(msg))):
        assert decoded.stamps == pytest.approx(msg.stamps)


def test_message_khong_do_do_tre_thi_khong_phinh_them_truong():
    """Fixture cũ đọc-ghi lại phải ra đúng như trước — không có khoá `stamps` thừa."""
    msg = _msg("cam01", 1, {1: l2_normalize(np.ones(DIM, dtype=np.float32))})
    assert msg.stamps == {}
    assert "stamps" not in json.loads(encode_jsonl(msg))
    assert decode_msgpack(encode_msgpack(msg)).stamps == {}


# --------------------------------------------------------------------------- thống kê


def test_percentile_khop_voi_numpy():
    values = [float(v) for v in range(1, 101)]
    for p in (0.0, 0.5, 0.9, 0.99, 1.0):
        assert percentile(values, p) == pytest.approx(float(np.percentile(values, p * 100)))


def test_summarize_dem_rieng_mau_am():
    stats = summarize([-5.0, 1.0, 2.0, 3.0])
    assert stats is not None
    assert stats.n == 4
    assert stats.n_negative == 1
    assert stats.minimum == -5.0
    assert summarize([]) is None


def test_thieu_moc_thi_doan_do_bi_bo_qua_chu_khong_tinh_bang_khong():
    record = LatencyRecord(run_id="x", stamps={T0_CAPTURE: 10.0, T1_PROBE: 25.0})
    assert record.delta(COARSE_SEGMENTS[0]) == pytest.approx(15.0)
    assert record.delta(END_TO_END) is None


# --------------------------------------------------------------------------- file log


def test_log_ghi_va_doc_lai_duoc(tmp_path):
    path = tmp_path / "latency.jsonl"
    with LatencyLog(path, run_id="run-a") as log:
        log.write(LatencyRecord(run_id="run-a", cam_id="cam01", stamps={T0_CAPTURE: 1.5}))
        log.write(LatencyRecord(run_id="run-a", cam_id="cam02", stamps={T0_CAPTURE: 2.5}))

    records = read_records(path)
    assert [r.cam_id for r in records] == ["cam01", "cam02"]
    assert records[0].stamps[T0_CAPTURE] == pytest.approx(1.5)


def test_log_noi_them_khong_de_len_lan_chay_truoc(tmp_path):
    path = tmp_path / "latency.jsonl"
    with LatencyLog(path, run_id="run-a") as log:
        log.write(LatencyRecord(run_id="run-a", cam_id="cam01"))
    with LatencyLog(path, run_id="run-b") as log:
        log.write(LatencyRecord(run_id="run-b", cam_id="cam02"))

    records = read_records(path)
    assert [r.run_id for r in records] == ["run-a", "run-b"]
    # Mặc định báo cáo chỉ lấy lần chạy mới nhất, không trộn hai lần đo vào một bảng.
    assert [r.run_id for r in latency_report.select_run(records, "last")] == ["run-b"]
    assert len(latency_report.select_run(records, "all")) == 2


def test_dong_hong_khong_lam_chet_bao_cao(tmp_path):
    """File chép về từ máy thuê lúc tiến trình còn đang ghi thì dòng cuối hay bị cụt."""
    path = tmp_path / "latency.jsonl"
    path.write_text(
        json.dumps({"run_id": "r", "cam_id": "cam01", "stamps": {}}) + '\n{"run_id": "r", cam',
        encoding="utf-8",
    )
    records = read_records(path)
    assert [r.cam_id for r in records] == ["cam01"]


# --------------------------------------------------------------------------- engine


def _run_engine(latency=None) -> tuple[Engine, list]:
    person_a = l2_normalize(np.ones(DIM, dtype=np.float32))
    person_b = l2_normalize(np.arange(1, DIM + 1, dtype=np.float32))
    engine = _engine(latency=latency)
    updates = []
    for frame in range(12):
        msg = _msg("cam01", frame, {1: person_a, 2: person_b}, stamps=_producer_stamps(frame))
        updates.extend(engine.feed(msg))
        engine.mark_published()
    updates.extend(engine.finish())
    engine.mark_published()
    return engine, updates


def test_bat_do_do_tre_khong_doi_ket_qua_gan(tmp_path):
    """Điều kiện sống còn: đo là ĐO, không được chạm vào kết quả liên kết."""
    _, plain = _run_engine()
    with LatencyLog(tmp_path / "l.jsonl") as log:
        _, measured = _run_engine(latency=log)

    assert [(u.global_id, u.cam_id, u.tracklet_id, u.ts_ms) for u in plain] == [
        (u.global_id, u.cam_id, u.tracklet_id, u.ts_ms) for u in measured
    ]


def test_engine_ghi_du_moc_t0_den_t4(tmp_path):
    path = tmp_path / "latency.jsonl"
    with LatencyLog(path) as log:
        _run_engine(latency=log)

    records = read_records(path)
    assert records, "phải có ít nhất một bản ghi"
    for record in records:
        for key in (T0_CAPTURE, T1_PROBE, T1B_DEQUEUE, T2_XADD, T3A_RECV):
            assert key in record.stamps, f"thiếu mốc producer {key}"
        for key in (T3W_WINDOW, T3_ASSOC, T3D_DB, T4_OUT):
            assert key in record.stamps, f"thiếu mốc engine {key}"
        # Thứ tự thời gian bên trong engine phải đúng chiều.
        assert record.stamps[T3W_WINDOW] <= record.stamps[T3_ASSOC] <= record.stamps[T3D_DB]
        assert record.stamps[T3D_DB] <= record.stamps[T4_OUT]
        assert record.cam_id == "cam01"
        assert record.global_id > 0


def test_moc_lay_tu_khung_moi_nhat_cua_tracklet(tmp_path):
    """Độ trễ phải trả lời "vị trí vừa hiện cũ bao nhiêu", nên mốc là của khung MỚI NHẤT."""
    path = tmp_path / "latency.jsonl"
    with LatencyLog(path) as log:
        engine, _ = _run_engine(latency=log)

    records = read_records(path)
    last = records[-1]
    tracklet = engine.builder.by_id(last.tracklet_id)
    assert tracklet is not None
    assert last.stamps[T0_CAPTURE] == pytest.approx(tracklet.last_stamps[T0_CAPTURE])
    assert last.frame_id == tracklet.end_frame_id


def test_khong_bat_do_thi_khong_sinh_ban_ghi(tmp_path):
    engine, _ = _run_engine()
    assert engine.latency is None
    assert engine.mark_published() == 0
    assert not list(tmp_path.iterdir())


def test_engine_chay_duoc_khi_message_khong_co_moc(tmp_path):
    """Fixture cũ (không có `stamps`) vẫn phải chạy — chỉ là báo cáo thiếu vài đoạn."""
    path = tmp_path / "latency.jsonl"
    person = l2_normalize(np.ones(DIM, dtype=np.float32))
    with LatencyLog(path) as log:
        engine = _engine(latency=log)
        for frame in range(6):
            engine.feed(_msg("cam01", frame, {1: person}))
            engine.mark_published()
        engine.finish()
        engine.mark_published()

    records = read_records(path)
    assert records
    assert all(T0_CAPTURE not in r.stamps for r in records)
    assert all(T4_OUT in r.stamps for r in records)
    assert all(r.delta(END_TO_END) is None for r in records)


# --------------------------------------------------------------------------- báo cáo


def _synthetic_records(n: int = 100) -> list[LatencyRecord]:
    """n bản ghi, trong đó 10 cái cuối có `window_wait` phình lên 2 s — đuôi trễ giả lập."""
    out: list[LatencyRecord] = []
    for i in range(n):
        base = BASE_WALL + i * 100.0
        window_wait = 2000.0 if i >= n - 10 else 20.0
        stamps = {
            T0_CAPTURE: base,
            T0_IS_NTP: 1.0,
            T1_PROBE: base + 30.0,
            T1B_DEQUEUE: base + 32.0,
            T2_XADD: base + 35.0,
            T3A_RECV: base + 40.0,
            T3W_WINDOW: base + 40.0 + window_wait,
            T3_ASSOC: base + 45.0 + window_wait,
            T3D_DB: base + 46.0 + window_wait,
            T4_OUT: base + 48.0 + window_wait,
        }
        out.append(
            LatencyRecord(
                run_id="r1",
                cam_id=f"cam0{i % 2 + 1}",
                frame_id=i,
                tracklet_id=i,
                global_id=i % 7,
                window_n=3,
                db_flushed=i % 10 == 0,
                t0_source="ntp",
                stamps=stamps,
            )
        )
    return out


def test_bao_cao_tinh_dung_tung_doan():
    payload = latency_report.report(_synthetic_records(), as_json=True)
    seg = payload["segments"]
    assert seg["t1-t0  deepstream"]["median_ms"] == pytest.approx(30.0)
    assert seg["t1-t0  deepstream"]["n"] == 100
    # Trung vị vẫn nhỏ trong khi đuôi đã 2 s — đúng hình dạng mà phiên này đi tìm.
    e2e = seg["t4-t0  END-TO-END"]
    assert e2e["median_ms"] == pytest.approx(68.0)
    assert e2e["p99_ms"] > 2000.0
    assert e2e["max_ms"] == pytest.approx(2048.0)


def test_bao_cao_chi_dung_ten_doan_gay_ra_duoi_tre():
    payload = latency_report.report(_synthetic_records(), as_json=True)
    tail = payload["tail"]
    assert tail is not None
    assert tail["n_tail"] == 10
    # Đoạn đứng đầu bảng quy trách nhiệm phải là chỗ ta cố tình làm chậm.
    assert tail["segments"][0]["segment"] == "t3w-t3a window_wait"
    assert tail["segments"][0]["delta_ms"] == pytest.approx(1980.0)


def test_bao_cao_canh_bao_khi_co_mau_am():
    records = _synthetic_records(20)
    # Máy engine chạy chậm hơn máy GPU 500 ms: đoạn bắc cầu ra số âm.
    for record in records[:5]:
        record.stamps[T3A_RECV] -= 500.0
    payload = latency_report.report(records, as_json=True)
    assert payload["segments"]["  t3a-t2 redis_pickup"]["n_negative"] == 5
    assert any("lệch đồng hồ" in w for w in payload["warnings"])


def test_bao_cao_canh_bao_khi_t0_la_duong_lui():
    records = _synthetic_records(20)
    for record in records:
        record.t0_source = "probe"
    payload = latency_report.report(records, as_json=True)
    assert any("đường lui" in w for w in payload["warnings"])


def test_bao_cao_tach_duoc_theo_camera_va_liet_ke_cham_nhat():
    payload = latency_report.report(_synthetic_records(), group_by="cam_id", top=3, as_json=True)
    assert set(payload["groups"]) == {"cam01", "cam02"}
    assert sum(g["n"] for g in payload["groups"].values()) == 100
    worst = payload["worst"]
    assert len(worst) == 3
    assert worst[0]["e2e_ms"] >= worst[-1]["e2e_ms"] > 2000.0


def test_cli_doc_file_va_in_json(tmp_path, capsys):
    path = tmp_path / "latency.jsonl"
    with LatencyLog(path, run_id="r1") as log:
        log.write_many(_synthetic_records(30))

    assert latency_report.main(["--log", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_records"] == 30
    assert payload["run_ids"] == ["r1"]


def test_cli_bao_loi_ro_rang_khi_thieu_file(tmp_path):
    assert latency_report.main(["--log", str(tmp_path / "khong-ton-tai.jsonl")]) == 2


def test_now_ms_la_epoch_milliseconds():
    assert is_plausible_ms(now_ms())


# --------------------------------------------------------------------------- Redis


class _FakePublisher:
    """Bên gửi giả — chỉ giữ lại message, không cần Redis (như test_queued_publisher)."""

    stream = "mct:frames"

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.sent: list[FrameMessage] = []
        self.gate = gate
        self.closed = False

    def publish_many(self, messages) -> int:
        if self.gate is not None:
            self.gate.wait(timeout=5.0)
        batch = list(messages)
        self.sent.extend(batch)
        return len(batch)

    def close(self) -> None:
        self.closed = True


def test_hang_doi_publisher_dong_dau_luc_nhac_message_ra():
    """`t1b - t1` đo thời gian nằm CHỜ trong hàng đợi, không phải thời gian gửi.

    Message đầu tiên gần như không chờ: luồng nền đang rảnh nên nhấc nó ra ngay rồi mới
    chặn ở lệnh gửi — phần bị chặn đó thuộc đoạn `t2-t1b` (XADD). Message thứ hai mới là
    cái nằm lại trong hàng đợi, và đó chính là hình dạng của đuôi trễ khi Redis hoặc
    engine không theo kịp pipeline.
    """
    gate = threading.Event()
    fake = _FakePublisher(gate)
    pub = QueuedFramePublisher(fake, maxsize=8, batch=4)
    try:
        for frame in (1, 2):
            msg = _msg("cam01", frame, {1: l2_normalize(np.ones(DIM, dtype=np.float32))})
            stamp = now_ms()
            msg.stamps = {T0_CAPTURE: stamp, T0_IS_NTP: 0.0, T1_PROBE: stamp}
            pub.publish(msg)
            time.sleep(0.05)  # để luồng nền kịp nhấc cái đầu ra trước khi có cái thứ hai
        time.sleep(0.2)  # Redis "chậm": luồng nền vẫn đang bị giữ ở gate
        gate.set()
        deadline = time.monotonic() + 5.0
        while len(fake.sent) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        pub.close(timeout=2.0)

    assert len(fake.sent) == 2
    waited = [m.stamps[T1B_DEQUEUE] - m.stamps[T1_PROBE] for m in fake.sent]
    assert waited[0] < 50.0, "message đầu không phải chờ hàng đợi"
    assert waited[1] >= 150.0, "message thứ hai phải ghi nhận thời gian nằm chờ"


def test_hang_doi_khong_them_dau_vao_message_khong_do():
    """Message không mang mốc (fixture cũ, tool khác) thì đi qua nguyên vẹn."""
    fake = _FakePublisher()
    pub = QueuedFramePublisher(fake, maxsize=8, batch=4)
    try:
        pub.publish(_msg("cam01", 1, {1: l2_normalize(np.ones(DIM, dtype=np.float32))}))
        deadline = time.monotonic() + 5.0
        while not fake.sent and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        pub.close(timeout=2.0)

    assert fake.sent and fake.sent[0].stamps == {}


class _FakeRedis:
    """Đủ để `FrameConsumer` chạy: tạo group rồi trả về đúng một entry đã đóng gói."""

    def __init__(self, entry_id: bytes, payload: bytes) -> None:
        self.entry_id = entry_id
        self.payload = payload

    def xgroup_create(self, *args, **kwargs) -> None:
        return None

    def xreadgroup(self, *args, **kwargs):
        return [(b"mct:frames", [(self.entry_id, {b"data": self.payload})])]

    def close(self) -> None:
        return None


def test_consumer_dong_dau_t2_tu_entry_id_va_t3a_luc_nhan():
    """`t2` không cần trường riêng trên wire: entry ID của Redis chính là nó."""
    msg = _msg("cam01", 1, {1: l2_normalize(np.ones(DIM, dtype=np.float32))})
    msg.stamps = _producer_stamps(1)
    del msg.stamps[T2_XADD]
    del msg.stamps[T3A_RECV]

    xadd_ms = 1_757_000_123_456
    fake = _FakeRedis(f"{xadd_ms}-0".encode(), encode_msgpack(msg))
    with FrameConsumer(client=fake) as consumer:
        entry_id, received = consumer.read()[0]

    assert entry_id == f"{xadd_ms}-0"
    assert received.stamps[T2_XADD] == pytest.approx(float(xadd_ms))
    assert received.stamps[T3A_RECV] >= received.stamps[T2_XADD] is not None
    # Mốc của producer đi qua nguyên vẹn, không bị ghi đè.
    assert received.stamps[T1_PROBE] == pytest.approx(msg.stamps[T1_PROBE])


def test_consumer_bo_qua_entry_id_khong_doc_duoc_thay_vi_bia_so():
    msg = _msg("cam01", 1, {1: l2_normalize(np.ones(DIM, dtype=np.float32))})
    fake = _FakeRedis(b"khong-phai-so-0", encode_msgpack(msg))
    with FrameConsumer(client=fake) as consumer:
        _, received = consumer.read()[0]

    assert T2_XADD not in received.stamps
    assert T3A_RECV in received.stamps


def test_nhieu_lam_tron_khong_bi_dem_thanh_lech_dong_ho():
    """Hai mốc cách nhau 1 µs theo chiều sai là nhiễu làm tròn, không phải bằng chứng."""
    stats = summarize([-0.001, -0.002, 0.5, 1.0])
    assert stats is not None
    assert stats.n_negative == 0
    assert summarize([-0.5, 0.5]).n_negative == 1


def test_vong_gan_cuoi_duoc_danh_dau_final_flush(tmp_path):
    """Hiện vật của phép đo phải TỰ khai báo, không bắt người đọc đi đào ra.

    Lúc hết nguồn, `Engine.finish()` đóng mọi tracklet còn sống một lượt; `window_wait`
    của chúng bằng thời gian từ khung cuối tới lúc engine chịu dừng (đo thật 2026-09-18:
    45.3 s = đúng `--idle-limit`). Đó không phải hành vi vận hành.
    """
    path = tmp_path / "latency.jsonl"
    with LatencyLog(path) as log:
        _run_engine(latency=log)

    records = read_records(path)
    assert any(r.final_flush for r in records), "vòng gán cuối phải được đánh dấu"
    assert any(not r.final_flush for r in records), "các vòng thường thì không"


def test_doan_cham_t2_duoc_noi_dung_sai_mot_mili_giay():
    """Entry ID của Redis chỉ mang mili-giây NGUYÊN — hiệu âm dưới 1 ms là lượng tử hoá."""
    xadd = next(s for s in FINE_SEGMENTS if s.end == T2_XADD)
    khac = next(s for s in FINE_SEGMENTS if T2_XADD not in (s.start, s.end))
    assert xadd.neg_tolerance_ms == 1.0
    assert khac.neg_tolerance_ms < 1.0

    # -0.1 ms trên đoạn chạm t2 là nhiễu; trên đoạn khác thì không.
    assert summarize([-0.1, 0.5], neg_tolerance_ms=xadd.neg_tolerance_ms).n_negative == 0
    assert summarize([-0.1, 0.5], neg_tolerance_ms=khac.neg_tolerance_ms).n_negative == 1


# --------------------------------------------------------------------------- loại bản ghi


def test_engine_gan_nhan_first_dung_mot_lan_moi_tracklet(tmp_path):
    """Danh tính chốt ở vòng gán ĐẦU TIÊN: mỗi tracklet đúng một bản ghi `first`, và nó đứng
    trước mọi bản ghi khác của tracklet đó (phiên 28 — đuôi phiên 23 là bản ghi `close`)."""
    path = tmp_path / "latency.jsonl"
    with LatencyLog(path) as log:
        _run_engine(latency=log)

    records = read_records(path)
    assert {r.kind for r in records} <= {"first", "update", "close"}
    seen: set[int] = set()
    for record in records:
        if record.tracklet_id not in seen:
            assert record.kind == "first"
            seen.add(record.tracklet_id)
        else:
            assert record.kind in ("update", "close")


def test_kind_di_qua_json():
    record = LatencyRecord(run_id="r", tracklet_id=3, kind="close")
    assert LatencyRecord.from_json(record.to_json()).kind == "close"
    assert LatencyRecord.from_json('{"run_id": "r"}').kind == ""


def test_log_cu_suy_ra_first_theo_thu_tu_tracklet():
    records = [
        LatencyRecord(run_id="a", tracklet_id=1),
        LatencyRecord(run_id="a", tracklet_id=1),
        LatencyRecord(run_id="a", tracklet_id=2),
        LatencyRecord(run_id="b", tracklet_id=1),
        LatencyRecord(run_id="b", tracklet_id=1, kind="close"),
    ]
    kinds = [r.kind for r in latency_report.with_kinds(records)]
    assert kinds == ["first", "repeat", "first", "first", "close"]
    assert records[0].kind == "", "không được sửa bản ghi gốc"


def test_muc_tieu_1s_cham_tren_ban_ghi_first_khong_tren_moi_ban_ghi():
    """Đuôi 2 s nằm ở bản ghi phát lại (cùng tracklet với bản ghi trước) -> không được tính
    vào độ trễ chốt danh tính, dù p90 trên mọi bản ghi vượt 1 s."""
    records = _synthetic_records(20)
    for record in records[-10:]:  # 10 bản ghi chậm = phát lại của tracklet 0..9
        record.tracklet_id -= 10
    payload = latency_report.report(records, as_json=True)
    identity = [w for w in payload["warnings"] if "CHỐT DANH TÍNH" in w]
    assert identity and "ĐẠT" in identity[0] and "KHÔNG ĐẠT" not in identity[0]
    assert any("MỌI bản ghi" in w for w in payload["warnings"])
    groups = latency_report._by_group(latency_report.with_kinds(records), "kind", as_json=True)
    assert groups["first"]["n"] == 10 and groups["repeat"]["n"] == 10
