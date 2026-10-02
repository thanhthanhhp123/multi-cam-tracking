"""Test `tools/sync_recordings.py` — đồng bộ video điện thoại bằng tiếng vỗ tay.

Không cần ffmpeg: tín hiệu tổng hợp cho phần tìm độ lệch, và một `runner` giả đứng thay
ffprobe/ffmpeg cho phần chạy trọn. Canh: độ lệch đúng tới ~1 ms cả hai chiều, máy xa (tiếng
nhỏ hơn 10 lần) vẫn khớp, không có tín hiệu thì BÁO ĐỘNG chứ không trả một số bừa, và cửa sổ
chung + lệnh cắt dùng đúng độ lệch.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools.sync_recordings import (
    AUDIO_RATE,
    VideoInfo,
    common_window,
    encode_cmd,
    estimate_lag,
    main,
    onset_envelope,
    parse_ffprobe,
)


def _audio(events_s: list[float], *, seconds: float, gain: float, seed: int) -> np.ndarray:
    """Ồn nền + các tiếng vỗ tay (5 ms nhiễu dải rộng) ở các mốc cho trước."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.01, int(seconds * AUDIO_RATE))
    for t in events_s:
        i = int(t * AUDIO_RATE)
        n = int(0.005 * AUDIO_RATE)
        x[i : i + n] += gain * rng.normal(0.0, 1.0, n) * np.exp(-np.arange(n) / (n / 3))
    return np.clip(x, -1, 1).astype(np.float32)


CLAPS = [3.0, 5.0, 5.4]  # một tiếng, nghỉ, hai tiếng: nhịp không đều


@pytest.mark.parametrize("lag", [1.234, -0.75, 0.0])
def test_tim_do_lech(lag: float):
    ref = _audio(CLAPS, seconds=12, gain=0.8, seed=1)
    other = _audio([t + lag for t in CLAPS], seconds=12, gain=0.08, seed=2)  # máy xa: nhỏ 10x
    est = estimate_lag(onset_envelope(ref), onset_envelope(other), max_lag_s=5)
    assert est.lag_s == pytest.approx(lag, abs=0.003)
    assert est.confidence > 1.5


def test_khong_co_tieng_vo_tay_thi_do_tin_cay_thap():
    ref = _audio([], seconds=10, gain=0.0, seed=1)
    other = _audio([], seconds=10, gain=0.0, seed=2)
    est = estimate_lag(onset_envelope(ref), onset_envelope(other), max_lag_s=5)
    assert est.confidence < 1.5


def test_duong_bao_rong_thi_bao_loi():
    with pytest.raises(ValueError):
        estimate_lag(np.zeros(0), np.ones(10))


def test_cua_so_chung():
    # cam02 bấm quay muộn 2 s (sự kiện ở t của cam01 nằm ở t − 2 của cam02 → lag = −2)
    w = common_window({"cam01": 0.0, "cam02": -2.0}, {"cam01": 60.0, "cam02": 50.0})
    assert (w.start_s, w.end_s) == (2.0, 52.0)
    with pytest.raises(ValueError):
        common_window({"a": 0.0, "b": 100.0}, {"a": 10.0, "b": 10.0})


def test_doc_ffprobe_xoay_va_vfr():
    data = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
                "avg_frame_rate": "2675/91",
                "duration": "91.0",
                "side_data_list": [{"rotation": -90}],
            },
            {"codec_type": "audio"},
        ],
        "format": {"duration": "91.2"},
    }
    info = parse_ffprobe("x.mp4", data)
    assert info.variable_rate
    assert info.rotation == 270
    assert info.display_size == (1080, 1920)
    assert info.has_audio
    cfr = VideoInfo("y", 1920, 1080, 10.0, 25.0, 25.0, 0, False)
    assert not cfr.variable_rate


def test_lenh_ffmpeg():
    cmd = encode_cmd(Path("in.mp4"), Path("out.mp4"), start_s=3.5, duration_s=60, fps=25)
    assert cmd[cmd.index("-ss") + 1] == "3.500"
    assert cmd.index("-ss") < cmd.index("-i")  # cắt nhanh, mã hoá lại nên vẫn đúng khung
    assert cmd[cmd.index("-vf") + 1] == "fps=25"
    assert cmd[cmd.index("-bf") + 1] == "0"
    assert "-an" in cmd


def _video_json(duration: float, *, audio: bool = True) -> bytes:
    streams = [
        {
            "codec_type": "video",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "30/1",
            "avg_frame_rate": "30/1",
            "duration": str(duration),
        }
    ]
    if audio:
        streams.append({"codec_type": "audio"})
    return json.dumps({"streams": streams}).encode()


def _fake_runner(lags: dict[str, float], durations: dict[str, float]):
    """Đứng thay ffprobe (trả JSON) và ffmpeg (trả audio s16le hoặc "mã hoá" thành công)."""
    calls: list[list[str]] = []

    def run(cmd):
        cmd = list(cmd)
        calls.append(cmd)
        if cmd[0] == "ffprobe":
            cam = Path(cmd[-1]).stem
            return subprocess.CompletedProcess(cmd, 0, _video_json(durations[cam]), b"")
        if cmd[-1] == "-":
            cam = Path(cmd[cmd.index("-i") + 1]).stem
            x = _audio(
                [t + lags[cam] for t in CLAPS],
                seconds=durations[cam],
                gain=0.5,
                seed=int(cam[-2:]),
            )
            raw = (x * 32767).astype("<i2").tobytes()
            return subprocess.CompletedProcess(cmd, 0, raw, b"")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    return run, calls


def test_chay_tron(tmp_path: Path):
    lags = {"cam01": 0.0, "cam02": 1.5, "cam03": -0.5}
    durations = {"cam01": 12.0, "cam02": 12.0, "cam03": 11.0}
    runner, calls = _fake_runner(lags, durations)
    args = [f"--video={c}=raw/{c}.mp4" for c in lags]
    out = tmp_path / "s1"
    args += ["--out-dir", str(out), "--max-lag-s", "4", "--execute"]
    assert main(args, runner=runner) == 0

    report = json.loads((out / "sync.json").read_text(encoding="utf-8"))
    for cam, lag in lags.items():
        assert report["lags_s"][cam] == pytest.approx(lag, abs=0.003)
    # cửa sổ chung trên trục cam01: [max(−lag), min(dur − lag)] = [0.5, 10.5]
    assert report["window_ref_s"][0] == pytest.approx(0.5, abs=0.003)
    assert report["window_ref_s"][1] == pytest.approx(10.5, abs=0.003)
    assert report["cameras"]["cam02"]["start_s"] == pytest.approx(2.0, abs=0.003)
    # tiếng vỗ mạnh nhất nằm đâu đó trong 3.0–5.4 s trục cam01 → khung (t − 0.5) × 25
    assert 2.5 * 25 - 1 <= report["clap_frame"] <= 4.9 * 25 + 1
    encodes = [c for c in calls if c[0] == "ffmpeg" and c[-1].endswith(".mp4")]
    assert len(encodes) == 3


def test_offset_tay_de_len_va_khong_doc_audio(tmp_path: Path):
    runner, calls = _fake_runner({"cam01": 0.0, "cam02": 0.0}, {"cam01": 10.0, "cam02": 10.0})
    args = ["--video", "cam01=a/cam01.mp4", "--video", "cam02=a/cam02.mp4"]
    assert main([*args, "--out-dir", str(tmp_path), "--offset", "cam02=0.8"], runner=runner) == 0
    report = json.loads((tmp_path / "sync.json").read_text(encoding="utf-8"))
    assert report["lags_s"]["cam02"] == 0.8
    assert report["estimates"]["cam02"]["source"] == "--offset"
    assert not any(c[-1] == "-" for c in calls)  # không trích audio khi mọi camera có --offset
    assert not any(c[0] == "ffmpeg" and c[-1].endswith(".mp4") for c in calls)  # chưa --execute


def test_khong_tim_chac_duoc_thi_dung(tmp_path: Path):
    def run(cmd):
        cmd = list(cmd)
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, _video_json(10.0), b"")
        seed = 1 if "cam01" in cmd[cmd.index("-i") + 1] else 2
        silent = _audio([], seconds=10, gain=0.0, seed=seed)
        return subprocess.CompletedProcess(cmd, 0, (silent * 32767).astype("<i2").tobytes(), b"")

    args = ["--video", "cam01=a/cam01.mp4", "--video", "cam02=a/cam02.mp4"]
    with pytest.raises(SystemExit, match="không tìm chắc"):
        main([*args, "--out-dir", str(tmp_path), "--max-lag-s", "4"], runner=run)
