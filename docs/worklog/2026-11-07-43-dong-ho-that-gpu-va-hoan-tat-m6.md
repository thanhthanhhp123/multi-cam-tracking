# 2026-11-07 (phiên 43): Đồng hồ thật trên GPU với code mới, tái lập lab-eval, đóng các mục treo của M6

- **Mốc:** M6 (đo) | **Máy:** máy dev (Windows 10) + `vast-gpu` (RTX A4000, instance 59523966, đã huỷ) | **Thời lượng:** ~2h

Tiếp phiên 42. Người dùng yêu cầu hoàn tất phần còn lại của M6 (trừ khóa API Vast.ai), cho phép thao tác trên Vast.ai không cần hỏi lại.

## Mục tiêu phiên

- Đo vế (B) "thời gian tới Global ID" bằng đồng hồ thật trên code sau phiên 41–42 (mục treo từ phiên 32).
- Chạy lại pipeline 12 buổi × 3 lần để kiểm tái lập con số c0.
- Đóng các mục treo của phiên 41–42: Python 3.10, bảng theo kịch bản, sweep `window_ms`/`min_frames`, cặp cam02 ↔ cam03, `.gitignore`.

## Đã làm

- **Subagent (4 việc song song, máy dev):**
  - Test trên Python 3.10.22 (`~/.venvs/mct-test`, dựng bằng uv): 881 passed, 5 skipped, không cần sửa gì. Ruff sạch.
  - Đưa script phân tích vào `eval/` (commit 70673ac), kèm test:
    - `eval/aggregate_lab_eval.py`: bảng theo nhóm và theo kịch bản;
    - `eval/diagnose_decisions.py`;
    - `eval/time_to_id_datatime.py`, có cờ `--former-owner on|off|both`.
  - Bảng theo kịch bản trên c0 (`data/s42/per_scenario.md`).
  - Sweep `window_ms` / `min_frames` (`data/s42/latency_sweep.md`).
  - Thử ước lượng thời gian đi cam02 ↔ cam03.
- **`.gitignore`:** thêm `/eval/gt/lab_*/` vì phiếu đồng thuận cấm đưa quỹ đạo người thật lên GitHub/cloud, và thêm `/logs/`.
- **Kiểm GT:** fixture GT hiện tại của s3_2, s4_1, s1_1 trùng từng byte với bản xuất lại bằng `cvat_to_mot` từ XML trong `data/cvat` (XML mới nhất 11-06 23:30).
- **GPU:**
  - Máy: RTX A4000 (driver 580.178.04), image `nvcr.io/nvidia/deepstream:7.1-triton-multiarch`.
  - Thứ tự: kiểm NVDEC trước, rồi `vast_bootstrap.sh`, rồi `vast_lab.sh engine`.
  - Engine: YOLO b3 23.6 MB, OSNet b100.
  - Chạy 12 buổi × 3 lần (`vast_lab.sh run <buổi> 3`), sau đó `fps s2_1` (2 lần). Trong lúc chạy ghi VRAM mỗi 2 s.
  - Kết quả ở `data/lab/gpu_s42/` (gitignored), tách riêng khỏi fixture phiên 40 ở `data/fixtures/`.
  - Instance đã huỷ; `show instances` = `[]`. Tốn khoảng $0.37 (credit $4.37 → $4.00).
- **Phân tích đuôi trễ:** gộp độ trễ bằng `tools.latency_report` (`frame_latencies`, `time_to_id`) qua 36 log, rồi tìm vị trí các mẫu chậm trong từng lần chạy. Script ở scratchpad: `pool_latency.py`, `slow_ttid.py`, `ttid_warm.py`.
- **Chấm lại 36 fixture mới:** `run_lab_eval` biến thể `base` (= c0), `data/lab/oracle/eval_gpu42/`, bảng ở `data/s42/gpu42_eval.md`.

## Quyết định kỹ thuật

1. **Giữ `window_ms` 500 / `min_frames` 5.**
   - Quy tắc chọn: HOTA cao nhất trên s2 trong số cấu hình có (B) p95 < 1 s.
   - `w1000` có HOTA s2 cao nhất (75.3) nhưng (B) p95 ≈ 1.12 s, nên bị loại. Đây là cùng kết luận với phiên 32.
   - `m10` tốt nhất trên nhóm giữ riêng (HOTA 58.1, IDF1 62.9, bàn giao 48%). Nó không được chọn vì thua trên s2 và (B) p95 ≈ 0.99 s sát ngưỡng. Chỉ ghi lại làm quan sát; chọn theo nhóm giữ riêng là phá quy trình.
2. **Giữ khoảng giữ chỗ 3–30 s cho cam02 ↔ cam03.**
   - Chỉ có n = 3 lần chuyển thật (7.4–10.4 s), cả 3 đều trong nhóm giữ riêng.
   - Khoảng đề xuất 5180–15520 ms cho kết quả lẫn lộn.
   - n = 3 không đủ để đặt cửa sổ, và đặt từ nhóm giữ riêng là rò dữ liệu kiểm.
   - Ghi thêm: transit thật cam01 → cam03 sau ROI bắt đầu từ khoảng 5.6–6.2 s.
3. **Báo vế (B) kèm hai cách đếm: toàn bộ, và bỏ 3 s khởi động.**
   - 37/51 mẫu trên 1 s là tracklet có mặt ngay từ khung đầu của video.
   - Ở các mẫu này, đoạn pipeline + engine chỉ khoảng 80 ms. Còn khung kích hoạt (`t0_capture`) đến sau `t0_first` khoảng 2.2 s đồng hồ thật.
   - Tính theo thời gian dữ liệu trên cùng fixture thì không có đuôi (s2_1b/1c/3b: p95 680 ms, 98.4% < 1 s).
   - Vậy đây là pipeline khởi động (khung đầu được đóng dấu `t0` rồi phải chờ suy luận lần đầu), không phải hành vi liên kết. Camera chạy liên tục chỉ gặp nó một lần lúc bật hệ thống.
   - Số chính trong báo cáo vẫn phải kèm cả cách đếm toàn bộ.
4. **Không bắt buộc quay thêm buổi đông người.**
   - Người dùng đã tự kiểm toàn bộ GT.
   - Nhóm giữ riêng (s1_1, s3_1, s3_2, s4_1) đã dùng để kiểm qua 3 phiên (41, 42, 43). Vì vậy con số giữ riêng nên được trình bày là "kiểm lặp", không phải "kiểm một lần chưa từng thấy".
   - Hỏi GVHD có cần thêm buổi mới hay chỉ cần cross-validation bỏ một buổi.

## Số liệu đo được

Cấu hình chung cho các bảng dưới:
- GPU: RTX A4000 16 GB, driver 580.178.04, DeepStream 7.1;
- model: YOLO11s 640 TensorRT FP16 batch 3, OSNet x1.0 MSDC 512 qua NvDCF ReID;
- 3 luồng video lab 25 fps phát thời gian thực;
- engine: `configs/lab/lab.mct.yaml` (combo + c0, `window_ms` 500), Redis trên cùng máy;
- định nghĩa độ trễ: phiên 32.

**Độ trễ đồng hồ thật, 12 buổi × 3 lần chạy:**

| Nhóm | Độ trễ theo khung (A): n, p50 / p95 / p99 / max | Thời gian tới Global ID (B), toàn bộ: n, p50 / p95 / p99, < 1 s | (B) bỏ 3 s đầu mỗi lần chạy |
|---|---|---|---|
| s2 | 595 508; 143 / 205 / 236 / 780 ms | 527; 525 / **2278** / 2395 ms; 92.0% | 414; 525 / **764** / 1467 ms; 97.8% |
| Giữ riêng | 474 123; 143 / 207 / 243 / 736 ms | 494; 521 / **768** / 2243 ms; 98.2% | 455; 525 / **768** / 942 ms; 98.9% |
| Cả 12 | 1 069 631; 143 / **206** / 242 / 780 ms; 100% < 1 s | 1021; 524 / **888** / 2365 ms; 95.0% | 869; 525 / **766** / 1243 ms; 98.4% |

→ **Đạt mục tiêu < 1 s trên cả 12 buổi theo định nghĩa phiên 32** (A p95 206 ms, B p95 888 ms).
- Trước sửa `pop_former_owner`: p95 (B) 9.6 s (phiên 40).
- Nhóm s2 tính toàn bộ có p95 (B) 2.28 s. Lý do là các buổi s2 ngắn, nên người có mặt từ khung đầu chiếm tỷ lệ lớn (đuôi khởi động, quyết định 3).
- Theo buổi, (B) p95 đạt ở s2_2, s2_3, s3_1, s3_2, s4_1. Ở các buổi còn lại, đuôi ~2.3 s đều nằm ở khung đầu.

**Thông lượng và tài nguyên:**
- FPS (`vast_lab.sh fps s2_1`, phát nhanh nhất có thể, 2 lần): **526–533 FPS gộp 3 luồng, khoảng 176 FPS/luồng**.
  - Mục tiêu đề cương là ≥ 15 FPS/luồng.
  - Khi chạy: nhiệt tối đa 74–80 °C, xung SM thấp nhất 1560 MHz. Cờ throttle `0x4` là SW power cap, không phải giảm xung vì nhiệt.
- VRAM tối đa trong cả lượt 36 lần chạy: **khoảng 1.2 GB / 16 GB**. GPU dùng trung bình 13% khi phát thời gian thực 3 × 25 fps.

**Tái lập độ chính xác:** chấm 36 fixture mới, c0, hộp ảnh, chỉ khung chú thích, IoU 0.5.

| Nhóm | Fixture | HOTA | IDF1 | Chồng lấn | Không chồng lấn | Quay lại | Tổng [95%] | đúng / tách / nhầm |
|---|---|---|---|---|---|---|---|---|
| s2 | phiên 40 | 73.0 | 80.5 | 82% | 79% | 56% | 76% [70–81] | 189 / 47 / 13 |
| | phiên 43 | 69.9 | 76.7 | 79% | 66% | 58% | 70% [64–75] | 174 / 56 / 19 |
| Giữ riêng | phiên 40 | 54.9 | 58.3 | 32% | 39% | 81% | 40% [34–46] | 96 / 100 / 44 |
| | phiên 43 | 53.4 | 55.7 | 32% | 44% | 67% | 39% [33–45] | 93 / 96 / 51 |
| Cả 12 | phiên 40 | 67.0 | 73.1 | 51% | 68% | 67% | 58% [54–63] | 285 / 147 / 57 |
| | phiên 43 | 64.4 | 69.7 | 50% | 60% | 62% | 55% [50–59] | 267 / 152 / 70 |

→ **Tái lập trong khoảng Wilson ở cả ba nhóm.**
- Nhóm giữ riêng gần như trùng (40 → 39%).
- s2 thấp hơn 6 điểm. c0 được chọn trên chính fixture phiên 40 của s2, nên lần chạy pipeline mới hồi về một phần là điều dự kiến. Ngoài ra NvDCF không tất định giữa các lần chạy, và đây là instance khác.
- Vì vậy con số nên báo cho cả 12 buổi là **HOTA 64–67, bàn giao 55–58% qua 6 lần chạy pipeline**, so với combo 44% và base 28%.

**Theo kịch bản (fixture phiên 40, c0 so với combo):**
- s3 (cảnh đông) gần như không đổi: HOTA 49.3 → 49.6.
- s4: bàn giao 12 → 35%.
- s1, s2 tăng rõ.
- Bảng đầy đủ ở `data/s42/per_scenario.md`.

## Vướng mắc / chưa xong

- **Cảnh đông (s3) vẫn yếu và c0 không giúp được.** Chồng lấn ở nhóm giữ riêng chỉ 32%, vì homography lệch ở xa: cam02 mất chân, khoảng 0.7 m.
- **Đuôi khởi động của vế (B).** Nếu muốn bỏ hẳn, có thể warm-up engine TensorRT trước khi mở nguồn video, hoặc đóng dấu `t0` sau khi pipeline đã PLAYING. Chưa làm, vì không đổi kết quả liên kết.
- **Cần hỏi người dùng:** có sửa GT trong giao diện CVAT sau lần xuất XML 11-06 23:30 không? Nếu có thì phải xuất lại và chấm lại.
- **Cần hỏi GVHD:**
  - có cần thêm buổi quay mới làm nhóm kiểm sạch không, hay dùng cross-validation bỏ một buổi;
  - có chạy baseline MV3DT không.

## Bước tiếp theo

1. Báo cáo GVHD với số trên. Chốt hai câu hỏi ở mục trên.
2. Nếu dùng cross-validation: chạy chọn tham số theo kiểu bỏ một buổi trên 12 buổi bằng fixture có sẵn, không cần GPU.
3. M7: viết chương 6 từ worklog phiên 40–43.
