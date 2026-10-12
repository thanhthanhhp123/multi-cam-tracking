"""Test `tools/record_rtsp.py` — ghi camera IP ra file + độ lệch đồng bộ theo đồng hồ máy ghi.

Công cụ chạy trên laptop mượn ở hiện trường, không có ai sửa hộ, nên test canh ba thứ:
(1) độ lệch tính ra đúng nghĩa với `tools.sync_recordings` (sai dấu là lệch gấp đôi mà không
triệu chứng); (2) ước lượng đường bao chịu được nhiễu mạng, cụm gói đầu file và trôi đồng hồ;
(3) mật khẩu camera không lọt ra đĩa, và công cụ chỉ cần thư viện chuẩn.
"""

from __future__ import annotations

import ast
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools import record_rtsp
from tools.record_rtsp import (
    ClockFit,
    ProgressParser,
    ProgressSample,
    cameras_from_env,
    fit_clock,
    main,
    merged_env,
    plan_sync,
    read_env_file,
    redact,
    run_check,
    sync_command,
    url_secrets,
)

URL1 = "rtsp://admin:Pa55w0rd!@192.168.1.3:554/unicast/c1/s0/live"
URL2 = "rtsp://admin:Pa55w0rd!@192.168.1.2:554/unicast/c1/s0/live"
URL3 = "rtsp://admin:Pa55w0rd!@192.168.1.4:554/unicast/c1/s0/live"


# --------------------------------------------------------------------------------------
# Chỉ thư viện chuẩn
# --------------------------------------------------------------------------------------


def test_chi_dung_thu_vien_chuan() -> None:
    """Laptop ghi hình chỉ có Python + ffmpeg: một import ngoài stdlib là hỏng cả buổi quay."""
    tree = ast.parse(Path(record_rtsp.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "không import tương đối: file phải chạy được trực tiếp"
            roots.add((node.module or "").split(".")[0])
    assert roots - {"__future__"} <= set(sys.stdlib_module_names), roots


def test_chay_truc_tiep_file_khong_can_pythonpath(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-I", record_rtsp.__file__, "check", "--help"],
        capture_output=True,
        cwd=tmp_path,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")


# --------------------------------------------------------------------------------------
# .env và mật khẩu
# --------------------------------------------------------------------------------------


def test_doc_env_bom_ngoac_chu_thich_export(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "﻿# camera lab\n"
        f"LAB_RTSP_CAM01={URL1}\n"
        f'export LAB_RTSP_CAM02="{URL2}"\n'
        "LAB_RTSP_CAM03=\n"
        "REDIS_URL=redis://localhost\n",
        encoding="utf-8",
    )
    cams = cameras_from_env(read_env_file(env))
    # `#` trong mật khẩu KHÔNG phải chú thích; biến rỗng bị bỏ.
    assert cams == {"cam01": URL1, "cam02": URL2}


@pytest.mark.parametrize("password", ["ab#cd", "ab/cd", "ab?cd", "ab@cd"])
def test_mat_khau_ky_tu_dac_biet_chua_ma_hoa_thi_bao_ro(tmp_path: Path, password: str) -> None:
    """`#` trong .env KHÔNG bị cắt như chú thích (cắt thì thành sai mật khẩu, khó hiểu), và
    URL hỏng cú pháp vì nó thì báo cách mã hoá thay vì một lỗi kết nối mơ hồ."""
    url = f"rtsp://admin:{password}@192.168.1.3:554/unicast/c1/s0/live"
    env = tmp_path / ".env"
    env.write_text(f"LAB_RTSP_CAM01={url}\n", encoding="utf-8")
    assert cameras_from_env(read_env_file(env)) == {"cam01": url}
    with pytest.raises(ValueError, match="%23"):
        record_rtsp.host_port(url)
    encoded = url.replace(
        password,
        password.replace("#", "%23").replace("/", "%2F").replace("?", "%3F").replace("@", "%40"),
    )
    assert record_rtsp.host_port(encoded) == ("192.168.1.3", 554)


def test_doc_env_utf16_cua_powershell(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_bytes(f"LAB_RTSP_CAM01={URL1}\r\n".encode("utf-16"))
    assert cameras_from_env(read_env_file(env)) == {"cam01": URL1}


def test_bien_moi_truong_de_len_env(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(f"LAB_RTSP_CAM01={URL1}\n", encoding="utf-8")
    merged = merged_env(env, {"lab_rtsp_cam01": URL3, "LAB_RTSP_CAM02": URL2})
    assert cameras_from_env(merged) == {"cam01": URL3, "cam02": URL2}


def test_che_mat_khau() -> None:
    line = f"Input #0, rtsp, from '{URL1}':"
    assert "Pa55w0rd!" not in redact(line)
    assert "rtsp://admin:***@192.168.1.3" in redact(line)
    # mật khẩu đứng một mình (vd trong thông báo lỗi) cũng phải che
    assert redact("auth failed for Pa55w0rd!", url_secrets([URL1])) == "auth failed for ***"


# --------------------------------------------------------------------------------------
# Tiến độ ffmpeg
# --------------------------------------------------------------------------------------


def test_progress_bo_khoi_chua_co_khung() -> None:
    parser = ProgressParser()
    lines = ["frame=0", "out_time_us=N/A", "progress=continue"]
    assert [parser.feed(line, 1.0) for line in lines] == [None, None, None]
    lines = ["frame=50", "fps=25.0", "out_time_us=2000000", "progress=continue"]
    out = [parser.feed(line, 3.5) for line in lines]
    assert out[-1] == ProgressSample(t_s=3.5, frame=50, out_time_s=2.0)
    assert not parser.ended
    parser.feed("frame=51", 3.6)
    parser.feed("out_time_us=2040000", 3.6)
    parser.feed("progress=end", 3.6)
    assert parser.ended


def _simulate(
    offset: float,
    *,
    duration: float = 120.0,
    drift: float = 0.0,
    latency: float = 0.08,
    burst_s: float = 1.5,
    report_s: float = 0.1,
    seed: int = 0,
) -> list[ProgressSample]:
    """Báo cáo tiến độ giả như ffmpeg: khung `p` tới máy ghi lúc offset + (1+drift)·p + trễ.

    Trễ = `latency` + nhiễu chỉ DƯƠNG (mũ, trung bình 15 ms). `burst_s` giây đầu file tới cùng
    lúc (ffmpeg dò luồng xong mới nhả ra) — đúng cái mà đường bao dưới phải bỏ qua.
    """
    rng = random.Random(seed)
    frames = [i / 25 for i in range(int(duration * 25))]
    arrive = [offset + (1 + drift) * p + latency + rng.expovariate(1 / 0.015) for p in frames]
    release = offset + (1 + drift) * burst_s + latency
    arrive = [max(a, release) if p < burst_s else a for p, a in zip(frames, arrive, strict=True)]
    samples = []
    t = offset + rng.uniform(0, report_s)
    i = 0
    while t < arrive[-1]:
        while i + 1 < len(frames) and arrive[i + 1] <= t:
            i += 1
        if arrive[i] <= t:
            samples.append(ProgressSample(t_s=t, frame=i + 1, out_time_s=frames[i]))
        t += report_s + rng.uniform(-0.01, 0.01)
    return samples


def test_duong_bao_duoi_tim_dung_goc_thoi_gian() -> None:
    fit = fit_clock(_simulate(100.0))
    # gốc + độ trễ nhỏ nhất (80 ms); độ trễ chung của mọi camera triệt tiêu khi lấy hiệu
    assert fit.offset_s == pytest.approx(100.08, abs=0.01)
    assert abs(fit.drift) < 20e-6
    assert fit.n_bins >= 20


def test_duong_bao_duoi_uoc_luong_duoc_troi_dong_ho() -> None:
    fit = fit_clock(_simulate(5.0, drift=80e-6, duration=240.0, seed=1))
    assert fit.drift * 1e6 == pytest.approx(80, abs=10)


def test_cum_goi_dau_file_khong_keo_lech() -> None:
    long_burst = fit_clock(_simulate(0.0, burst_s=4.0, seed=2))
    assert long_burst.offset_s == pytest.approx(0.08, abs=0.01)


def test_khong_co_mau_thi_bao_loi() -> None:
    with pytest.raises(ValueError, match="khung"):
        fit_clock([])


# --------------------------------------------------------------------------------------
# Độ lệch: đúng nghĩa của sync_recordings
# --------------------------------------------------------------------------------------


def test_do_lech_dung_dau() -> None:
    """cam02 bắt đầu ghi muộn hơn 0.5 s: sự kiện ở giây 10 của cam01 là giây 9.5 của cam02."""
    fits = {
        "cam01": ClockFit(offset_s=10.0, drift=0.0, jitter_ms=0, n_samples=1, n_bins=1),
        "cam02": ClockFit(offset_s=10.5, drift=0.0, jitter_ms=0, n_samples=1, n_bins=1),
    }
    plan = plan_sync(fits, {"cam01": 60.0, "cam02": 60.0}, "cam01")
    assert plan.lags_s == {"cam01": 0.0, "cam02": pytest.approx(-0.5)}
    assert plan.window_s == (pytest.approx(10.5), pytest.approx(70.0))
    assert plan.edge_error_ms["cam02"] == pytest.approx(0.0, abs=1e-6)


def test_troi_dong_ho_tinh_o_giua_va_bao_sai_so_mep() -> None:
    fits = {
        "cam01": ClockFit(offset_s=0.0, drift=0.0, jitter_ms=0, n_samples=1, n_bins=9),
        "cam02": ClockFit(offset_s=0.0, drift=100e-6, jitter_ms=0, n_samples=1, n_bins=9),
    }
    plan = plan_sync(fits, {"cam01": 200.0, "cam02": 200.0}, "cam01")
    # ở giữa (≈100 s) cam02 chậm 100 ppm → lệch ≈ −10 ms; hai mép lệch thêm ≈ 10 ms
    assert plan.lags_s["cam02"] == pytest.approx(-0.010, abs=1e-4)
    assert plan.edge_error_ms["cam02"] == pytest.approx(10.0, abs=0.1)


def test_khong_co_doan_chung_thi_bao_loi() -> None:
    fits = {
        "cam01": ClockFit(offset_s=0.0, drift=0.0, jitter_ms=0, n_samples=1, n_bins=1),
        "cam02": ClockFit(offset_s=100.0, drift=0.0, jitter_ms=0, n_samples=1, n_bins=1),
    }
    with pytest.raises(ValueError, match="chung"):
        plan_sync(fits, {"cam01": 30.0, "cam02": 30.0}, "cam01")


def test_lenh_in_ra_chay_duoc_voi_sync_recordings(tmp_path: Path) -> None:
    """Lệnh in sẵn phải được `sync_recordings` hiểu đúng: cắt cửa sổ đúng chỗ, không cần audio."""
    from tools import sync_recordings

    fits = {
        "cam01": ClockFit(offset_s=10.0, drift=0.0, jitter_ms=0, n_samples=1, n_bins=1),
        "cam02": ClockFit(offset_s=10.5, drift=0.0, jitter_ms=0, n_samples=1, n_bins=1),
        "cam03": ClockFit(offset_s=9.8, drift=0.0, jitter_ms=0, n_samples=1, n_bins=1),
    }
    durations = {"cam01": 60.0, "cam02": 60.0, "cam03": 61.0}
    plan = plan_sync(fits, durations, "cam01")
    files = {c: str(tmp_path / f"{c}.mkv") for c in fits}
    cmd = sync_command(files, plan, out_dir=str(tmp_path / "out"), fps=25)
    assert cmd[:3] == ["python", "-m", "tools.sync_recordings"]

    def runner(args):
        if args[0] == "ffprobe":
            cam = Path(args[-1]).stem
            data = {
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1920,
                        "height": 1080,
                        "duration": str(durations[cam]),
                        "r_frame_rate": "25/1",
                        "avg_frame_rate": "25/1",
                    }
                ]
            }
            return subprocess.CompletedProcess(args, 0, json.dumps(data).encode(), b"")
        raise AssertionError(f"không được gọi gì ngoài ffprobe khi đã có --offset: {args}")

    argv = [a for a in cmd[3:] if a != "--execute"]
    assert sync_recordings.main(argv, runner=runner) == 0
    report = json.loads((tmp_path / "out" / "sync.json").read_text(encoding="utf-8"))
    # cửa sổ chung theo giờ máy ghi: [10.5, 70.0] → mỗi camera cắt từ đúng giây ứng với 10.5
    starts = {c: item["start_s"] for c, item in report["cameras"].items()}
    assert starts == {
        "cam01": pytest.approx(0.5, abs=1e-3),
        "cam02": pytest.approx(0.0, abs=1e-3),
        "cam03": pytest.approx(0.7, abs=1e-3),
    }
    assert report["duration_s"] == pytest.approx(59.5, abs=1e-3)


# --------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------


def _probe_json(width=1920, height=1080, fps="25/1", audio=False) -> bytes:
    streams = [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": width,
            "height": height,
            "r_frame_rate": fps,
        }
    ]
    if audio:
        streams.append({"codec_type": "audio", "codec_name": "pcm_alaw"})
    return json.dumps({"streams": streams}).encode()


def _fake_runner(per_host: dict[str, subprocess.CompletedProcess] | None = None, version="7.1"):
    per_host = per_host or {}

    def runner(cmd, timeout_s):
        if "-version" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, f"ffmpeg version {version}-essentials_build\n".encode(), b""
            )
        host = cmd[-1].split("@")[1].split(":")[0]
        return per_host.get(host) or subprocess.CompletedProcess(cmd, 0, _probe_json(), b"")

    return runner


def _check(tmp_path: Path, cameras: dict[str, str], **kw):
    tool = tmp_path / "ffmpeg.exe"
    tool.write_bytes(b"")
    (tmp_path / "ffprobe.exe").write_bytes(b"")
    return run_check(
        cameras,
        ffmpeg=str(tool),
        ffprobe=str(tmp_path / "ffprobe.exe"),
        out_root=tmp_path / "raw",
        expect_size=(1920, 1080),
        expect_fps=25.0,
        runner=kw.pop("runner", _fake_runner()),
        reach=kw.pop("reach", lambda host, port: None),
    )


def _levels(rep, topic):
    return [lv for lv, t, _ in rep.lines if t == topic]


def test_check_dung_het_thi_khong_fail(tmp_path: Path) -> None:
    rep = _check(tmp_path, {"cam01": URL1, "cam02": URL2, "cam03": URL3})
    assert rep.n_fail == 0, rep.render()
    assert "Pa55w0rd!" not in rep.render()


def test_check_khong_ket_noi_duoc_thi_fail_va_goi_y_ip(tmp_path: Path) -> None:
    rep = _check(
        tmp_path,
        {"cam01": URL1, "cam02": URL2},
        reach=lambda host, port: "timed out" if host == "192.168.1.2" else None,
    )
    assert _levels(rep, "cam02") == ["FAIL"]
    assert "cùng dải" in rep.render()


def test_check_sai_mat_khau_thi_fail(tmp_path: Path) -> None:
    bad = subprocess.CompletedProcess(
        [], 1, b"", f"[rtsp @ 0x1] method DESCRIBE failed: 401 Unauthorized {URL1}".encode()
    )
    rep = _check(tmp_path, {"cam01": URL1}, runner=_fake_runner({"192.168.1.3": bad}))
    assert _levels(rep, "cam01") == ["FAIL"]
    assert "mật khẩu" in rep.render()
    assert "Pa55w0rd!" not in rep.render()


def test_check_trung_camera_thi_fail(tmp_path: Path) -> None:
    rep = _check(tmp_path, {"cam01": URL1, "cam02": URL1})
    assert "FAIL" in _levels(rep, "cam02")


def test_check_sai_do_phan_giai_thi_warn(tmp_path: Path) -> None:
    big = subprocess.CompletedProcess([], 0, _probe_json(2304, 1296), b"")
    rep = _check(
        tmp_path, {"cam01": URL1, "cam02": URL2}, runner=_fake_runner({"192.168.1.3": big})
    )
    assert _levels(rep, "cam01") == ["WARN"]
    assert rep.n_fail == 0


def test_check_khong_co_camera_hoac_ffmpeg_cu_thi_fail(tmp_path: Path) -> None:
    assert _levels(_check(tmp_path, {}), "camera") == ["FAIL"]
    rep = _check(tmp_path, {"cam01": URL1}, runner=_fake_runner(version="4.4"))
    assert _levels(rep, "ffmpeg") == ["FAIL"]


# --------------------------------------------------------------------------------------
# record — với ffmpeg giả
# --------------------------------------------------------------------------------------

FAKE_FFMPEG = r"""
import sys, threading, time
from urllib.parse import parse_qs, urlsplit
args = sys.argv[1:]
url = args[args.index("-i") + 1]
q = {k: float(v[0]) for k, v in parse_qs(urlsplit(url).query).items()}
t0 = q["t0"] + q.get("delay", 0.0)
die = q.get("die")
sys.stderr.write(f"Input #0, rtsp, from '{url}':\n")
sys.stderr.flush()
open(args[-1], "wb").write(b"\x1aE\xdf\xa3")
stop = threading.Event()
def watch():
    while True:
        ch = sys.stdin.read(1)
        if not ch or ch == "q":
            stop.set()
            return
threading.Thread(target=watch, daemon=True).start()
start = time.time()
frame = 0
while not stop.is_set():
    now = time.time()
    if die is not None and now - start > die:
        sys.stderr.write("Connection reset by peer\n")
        sys.exit(1)
    p = now - t0
    if p >= 0:
        frame = int(p * 25) + 1
        sys.stdout.write(f"frame={frame}\nout_time_us={int(p * 1e6)}\nprogress=continue\n")
    else:
        sys.stdout.write("frame=0\nout_time_us=N/A\nprogress=continue\n")
    sys.stdout.flush()
    time.sleep(0.02)
sys.stdout.write(f"frame={frame}\nout_time_us={int((time.time() - t0) * 1e6)}\nprogress=end\n")
"""


def _record(tmp_path: Path, delays: dict[str, float], *, extra=(), die: dict | None = None):
    fake = tmp_path / "fake_ffmpeg.py"
    fake.write_text(FAKE_FFMPEG, encoding="utf-8")
    t0 = time.time() + 0.3
    env = tmp_path / ".env"
    lines = []
    for i, (cam, delay) in enumerate(delays.items()):
        query = f"t0={t0:.6f}&delay={delay}"
        if die and cam in die:
            query += f"&die={die[cam]}"
        lines.append(
            f"LAB_RTSP_{cam.upper()}=rtsp://admin:Pa55w0rd!@10.0.0.{i + 2}:554/live?{query}"
        )
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    argv = [
        "record",
        "--session",
        "t1",
        "--duration",
        "2.5",
        "--env-file",
        str(env),
        "--out-root",
        str(tmp_path / "raw"),
        *extra,
    ]
    return main(
        argv,
        reach=lambda host, port: None,
        ffmpeg_cmd=[sys.executable, str(fake)],
        environ={},
    )


def test_record_ghi_du_file_va_tinh_dung_do_lech(tmp_path: Path, capsys) -> None:
    code = _record(tmp_path, {"cam01": 0.0, "cam02": 0.4, "cam03": -0.2})
    out = capsys.readouterr().out
    assert code == 0, out
    session = tmp_path / "raw" / "t1"
    report = json.loads((session / "record.json").read_text(encoding="utf-8"))
    assert report["ok"] and report["stopped_by"] == "duration"
    for cam in ("cam01", "cam02", "cam03"):
        assert (session / f"{cam}.mkv").exists()
        assert (session / f"{cam}.progress.csv").read_text().startswith("t_s,frame,out_time_s")
        assert report["cameras"][cam]["exit_code"] == 0
    lags = report["sync"]["lags_s"]
    # cam02 bắt đầu muộn 0.4 s → lag −0.4; cam03 sớm 0.2 s → lag +0.2
    assert lags["cam02"] == pytest.approx(-0.4, abs=0.06)
    assert lags["cam03"] == pytest.approx(0.2, abs=0.06)
    assert "--offset" in report["sync"]["command"]
    # mật khẩu không lọt ra đĩa
    for path in session.iterdir():
        assert "Pa55w0rd!" not in path.read_text(encoding="utf-8", errors="replace"), path.name
    assert "Pa55w0rd!" not in out


def test_record_camera_chet_giua_chung_thi_buoi_hong(tmp_path: Path, capsys) -> None:
    code = _record(tmp_path, {"cam01": 0.0, "cam02": 0.0}, die={"cam02": 1.0})
    out = capsys.readouterr().out
    assert code == 1
    report = json.loads((tmp_path / "raw" / "t1" / "record.json").read_text(encoding="utf-8"))
    assert not report["ok"] and report["sync"] is None
    assert any("cam02" in e and "Connection reset" in e for e in report["errors"])
    assert "HỎNG" in out


def test_eof_tren_stdin_khong_phai_enter() -> None:
    """Máy dev 2026-10-12: stdin báo TTY nhưng đọc ra EOF ngay → buổi ghi dừng sau 0.9 s."""
    import io
    import threading

    stop = threading.Event()
    record_rtsp.wait_enter(io.StringIO(""), stop)
    assert not stop.is_set()
    record_rtsp.wait_enter(io.StringIO("\n"), stop)
    assert stop.is_set()


def test_record_tu_choi_ghi_de_buoi_da_co(tmp_path: Path) -> None:
    session = tmp_path / "raw" / "t1"
    session.mkdir(parents=True)
    (session / "cam01.mkv").write_bytes(b"old")
    with pytest.raises(SystemExit):
        _record(tmp_path, {"cam01": 0.0})
    assert (session / "cam01.mkv").read_bytes() == b"old"
