"""Tách sai số điểm chân thành ĐỘ CHỆCH và NHIỄU — trần lý thuyết của mọi bộ lọc.

    PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m eval.diagnose_foot_error \\
        --detector data/fixtures/ds_wildtrack_7cam_onnx_detbox.jsonl \\
        --reference data/fixtures/ds_wildtrack_7cam_onnx_gtbox.jsonl \\
        --homography-dir configs/cameras/homography/wildtrack

**Vì sao có file này.** Phiên 13 đo được điểm chân từ hộp detector lệch xa hơn điểm chân từ
hộp ground-truth (d_ground giữa hai tracklet cùng người: trung vị 0.74 m so với 0.21 m) và
đề xuất một hướng rẻ: làm mượt quỹ đạo theo thời gian để khử nhiễu, khỏi phải đổi detector.
Phiên 16 hiện thực bộ lọc đó (`mct.tracklet.smooth_ground_path`) và đo được **gần như
không đổi**. Script này trả lời câu hỏi còn lại: vì sao?

Phép đo dựa trên đúng một tính chất của cặp fixture do `tools/reembed_fixture.py` sinh ra:
hai bản có **cùng tập detection, cùng `local_track_id`, cùng `ts_ms`**, khác đúng toạ độ hộp
(một bên hộp detector, một bên hộp ground-truth đã khớp IoU). Nên với từng detection, hiệu
hai điểm chân sau khi chiếu về mặt phẳng chung là **sai số điểm chân của detector**, đo bằng
mét, không cần giả định gì thêm.

Tách sai số đó theo từng tracklet thành hai phần:

    Δ_i = b + n_i        b   = trung bình Δ trong tracklet (ĐỘ CHỆCH, có hệ thống)
                         n_i = phần còn lại (NHIỄU, trung bình 0)

Bộ lọc thời gian chỉ khử được `n_i`; `b` thì không bộ lọc nào chạm tới, vì nó không phải
nhiễu mà là chỗ khác nhau CÓ HỆ THỐNG giữa hộp detector và hộp thật (hộp cao hơn/thấp hơn,
chân bị che, hộp cắt ở biên ảnh). Nên `|b|` chính là **trần dưới của sai số điểm chân sau
khi làm mượt hoàn hảo** — con số cần biết trước khi đầu tư thêm vào hướng này.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.schema import read_jsonl
from mct.homography import HomographyMapper

PERCENTILES = (5, 25, 50, 75, 95)


def _quantiles(values: list[float]) -> list[float]:
    if not values:
        return [0.0] * len(PERCENTILES)
    return [float(v) for v in np.percentile(np.array(values, dtype=np.float64), PERCENTILES)]


def _print_row(name: str, values: list[float]) -> None:
    print(f"{name:30s}" + "".join(f"{v:8.2f}" for v in _quantiles(values)))


def world_points(
    path: Path, mapper: HomographyMapper
) -> dict[tuple[str, int, int], tuple[float, float]]:
    """(cam_id, local_track_id, ts_ms) → điểm chân trên mặt phẳng chung (mét).

    Detection không chiếu được (camera chưa hiệu chỉnh, điểm ra vô cực) bị bỏ ở cả hai
    phía một cách tự nhiên: khoá nào thiếu một bên thì không vào phép so.
    """
    out: dict[tuple[str, int, int], tuple[float, float]] = {}
    for msg in read_jsonl(path):
        for det in msg.detections:
            point = mapper.project(msg.cam_id, det.ground_point)
            if point is not None:
                out[(msg.cam_id, int(det.local_track_id), int(msg.ts_ms))] = point
    return out


def decompose(
    detector: dict[tuple[str, int, int], tuple[float, float]],
    reference: dict[tuple[str, int, int], tuple[float, float]],
    *,
    min_frames: int,
) -> dict:
    """Sai số từng detection → độ chệch + nhiễu theo từng tracklet."""
    per_track: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for key, point_a in detector.items():
        point_b = reference.get(key)
        if point_b is not None:
            per_track[(key[0], key[1])].append((point_a[0] - point_b[0], point_a[1] - point_b[1]))

    errors: list[float] = []  # |Δ| của từng detection
    bias: list[float] = []  # |b| của từng tracklet
    jitter: list[float] = []  # RMS |n_i| của từng tracklet
    energy_bias = 0.0
    energy_noise = 0.0
    n_used = 0

    for deltas in per_track.values():
        if len(deltas) < min_frames:
            continue
        arr = np.array(deltas, dtype=np.float64)
        errors.extend(float(v) for v in np.hypot(arr[:, 0], arr[:, 1]))
        mean = arr.mean(axis=0)
        residual = arr - mean
        bias.append(float(np.hypot(mean[0], mean[1])))
        jitter.append(float(np.sqrt((residual**2).sum(axis=1).mean())))
        # Phân rã năng lượng: E|Δ|² = |b|² + E|n|², cộng dồn theo số detection để
        # tracklet dài đóng góp đúng trọng số của nó.
        energy_bias += float((mean**2).sum()) * len(arr)
        energy_noise += float((residual**2).sum())
        n_used += len(arr)

    total = energy_bias + energy_noise
    return {
        "n_tracklets": len(bias),
        "n_detections": n_used,
        "error_m_quantiles": _quantiles(errors),
        "bias_m_quantiles": _quantiles(bias),
        "jitter_m_quantiles": _quantiles(jitter),
        "rms_total_m": float(np.sqrt(total / n_used)) if n_used else 0.0,
        "rms_bias_m": float(np.sqrt(energy_bias / n_used)) if n_used else 0.0,
        "rms_noise_m": float(np.sqrt(energy_noise / n_used)) if n_used else 0.0,
        "bias_share_of_energy": energy_bias / total if total else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detector", type=Path, required=True, help="fixture cắt theo hộp detector")
    p.add_argument("--reference", type=Path, required=True, help="fixture cắt theo hộp GT")
    p.add_argument("--homography-dir", type=Path, required=True)
    p.add_argument("--min-frames", type=int, default=3, help="tracklet ngắn hơn thì bỏ")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    mapper = HomographyMapper.load(args.homography_dir)
    print(f"homography: {len(mapper.calibrated)} camera đã hiệu chỉnh")
    detector = world_points(args.detector, mapper)
    reference = world_points(args.reference, mapper)
    print(f"{args.detector.name}: {len(detector)} điểm chiếu được")
    print(f"{args.reference.name}: {len(reference)} điểm chiếu được")

    report = decompose(detector, reference, min_frames=args.min_frames)

    print(f"\n{'=' * 78}")
    print("SAI SỐ ĐIỂM CHÂN CỦA DETECTOR (mét trên mặt phẳng tham chiếu)")
    print(f"{'=' * 78}")
    print(f"tracklet đủ dài để tách: {report['n_tracklets']}, {report['n_detections']} detection")
    print(f"\n{'':30s}" + "".join(f"{'p' + str(v):>8s}" for v in PERCENTILES))
    _print_row("|Δ| từng detection", report["error_m_quantiles"])
    _print_row("|độ chệch| theo tracklet", report["bias_m_quantiles"])
    _print_row("nhiễu (RMS) theo tracklet", report["jitter_m_quantiles"])

    print(f"\n{'-' * 78}")
    print("PHÂN RÃ NĂNG LƯỢNG SAI SỐ  (E|Δ|² = |độ chệch|² + E|nhiễu|²)")
    print(f"{'-' * 78}")
    bias_share = report["bias_share_of_energy"]
    print(f"RMS tổng                  {report['rms_total_m']:6.3f} m")
    print(f"  phần độ chệch           {report['rms_bias_m']:6.3f} m  ({bias_share:.1%} năng lượng)")
    print(f"  phần nhiễu              {report['rms_noise_m']:6.3f} m  ({1 - bias_share:.1%})")
    print(
        "\nBộ lọc thời gian chỉ khử được phần NHIỄU. Trần lý thuyết của việc làm mượt hoàn"
        f"\nhảo là RMS {report['rms_bias_m']:.3f} m — nếu nó không nhỏ hơn hẳn "
        f"{report['rms_total_m']:.3f} m thì hướng này\nkhông có gì để lấy."
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nsố liệu đầy đủ: {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
