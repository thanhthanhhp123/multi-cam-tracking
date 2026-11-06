# 2026-11-06 (phiên 40): Chạy pipeline thật 12 buổi trên GPU, lab-eval, sweep biến thể, quét nhãn

- **Mốc:** M6 (đo) | **Máy:** máy dev (Windows, user `admin`) + `vast-gpu` (RTX A4000, thuê ~1.5 h) | **Thời lượng:** ~5h

Tiếp phiên 39 cùng ngày (cắt s1_1 ở khung 477). Phiên này đi từ "GT đã duyệt" tới "có số đo đầu tiên của đóng góp chính".

## Mục tiêu phiên

- Từ GT đã duyệt: xuất GT, đo thời gian đi, điền topology, chạy pipeline thật trên GPU cho 12 buổi chấm, chấm điểm.
- Tìm xem lỗi nằm ở nhãn, detector, ReID hay liên kết (nguyên tắc 1, CLAUDE.md §1).

## Đã làm

**GT và cấu hình.**
- 13 buổi trong `data/cvat/` đều `REVIEWED`; `tools.cvat_to_mot` chạy sạch cho cả 13 (s1_1 duyệt sau, không cần sửa nhãn nào so với bản nháp đã cắt).
- `dataset_stats` trên GT đã duyệt → `data/prelabel/dataset_stats_reviewed.json`.
- `tools.estimate_transit` trên `calib` và `calib_2`. Chép vào `configs/lab/topology.yaml`: cam01 ↔ cam03 `min_ms` 3220, `max_ms` 19420; `status: measured`; cam02 ↔ cam03 giữ số giữ chỗ 3–30 s (ghi rõ trong comment). `lab-check` 14 buổi: 0 FAIL. Commit `7748a34`.

**Môi trường.**
- Máy dev có cấu hình mới: i5-12400F, 15.8 GB RAM, RTX 3060 12 GB, Windows 10; WSL2 chưa có bản phân phối, Docker Desktop chưa chạy → chưa chạy được `src/ds_pipeline` tại chỗ. Chọn thuê GPU.
- Cài `vastai` 1.11.0 (`~/venvs/vast`), tạo khóa SSH `~/.ssh/id_ed25519`, gắn vào tài khoản.
- venv `~/venvs/mct-stats` (Python 3.12) cho thống kê/chấm điểm. TrackEval clone về `~/TrackEval` và **vá 81 chỗ `np.float/np.int/np.bool`** để chạy với numpy 2.x (bản clone ngoài repo).
- ffmpeg 9.0.2 (phiên 39).

**Thuê GPU (ba lần).**

| Hợp đồng | Offer | GPU | Driver | NVDEC | Kết quả |
|---|---|---|---|---|---|
| 59408046 | 37930663 | RTX 3090, Tây Ban Nha | 590.48 | **hỏng** (`PREROLLING`, `Failed to query video capabilities`) | huỷ |
| 59408718 | 43543261 | RTX 3090, Tây Ban Nha | 615.78 | **hỏng**, thư viện `libnvcuvid` có đủ | huỷ |
| 59409343 | 51587985 | RTX A4000, Estonia, $0.136/h | 580.178 | tốt (`EOS` sau 1 s) | dùng |

**Chạy trên A4000.** Đẩy repo + `models/` (tar|ssh), bootstrap (`pyds` 1.2.0, `numpy` 1.26.4), build engine YOLO b3, rồi `vast_lab.sh run <buổi> 3` cho s2_1 (thử) rồi 11 buổi còn lại. Đẩy video của buổi sau chồng lên lúc buổi trước chạy GPU. Mỗi video kiểm md5. Kéo kết quả về từng buổi. Huỷ máy ngay khi xong, `show instances` = 0. **Chi phí: credit $5.15 → $4.46** (~$0.69, gồm hai lần thử hỏng).

**Chấm điểm và phân tích** (`eval.run_lab_eval`, `tools.estimate_transit`, các script phụ ở scratchpad):
- Chấm 12 buổi × 3 lần chạy, biến thể `base`.
- Chẩn đoán s1_1 (bàn giao chồng lấn 0%): ngoại hình tốt, hình học nhiễu.
- Sweep 8 biến thể × 12 buổi (lần chạy r1).
- Quét nhãn (ngoại hình vs mã nhãn) → `data/prelabel/label_scan.json`.
- Quét hộp nhãn vs hộp hệ thống từng khung → `data/prelabel/SCAN_REPORT.md`.

## Quyết định kỹ thuật

- **Thuê card chuyên nghiệp (A4000), không thuê GeForce.** Hai RTX 3090 liên tiếp hỏng NVDEC trong container dù thư viện đầy đủ; A4000 đã từng chạy tốt (phiên 30). Máy 3090 "chạy bình thường" ở phiên 38 chỉ chạy PyTorch, chưa từng thử NVDEC, nên không phải bằng chứng. Bước thử NVDEC 45 s ở đầu mỗi máy bắt được lỗi trước khi đẩy 6 GB video; mỗi lần thử hỏng tốn ~$0.03–0.05.
- **Vá TrackEval cục bộ** (ngoài repo) thay vì hạ numpy: `requirements.txt` của TrackEval ghim `numpy==1.18.1`, không cài được trên Python 3.12 và sẽ làm hỏng venv.
- **Sweep chọn trên nhóm s2 (8 buổi), kiểm trên nhóm giữ riêng (s1_1, s3_1, s3_2, s4_1)** để lộ chuyện khớp quá mức. Chỉ dùng lần chạy r1 vì engine tất định.
- **Không fine-tune lúc này.** Đo được: detector tốt (DetA ~82, `match_rate` 0.998), ReID phân biệt tốt (cosine cùng người 0.81–0.92, khác người 0.61–0.75), nên model không phải điểm nghẽn. Dữ liệu nhỏ (~7 người, 18 phút), không có tập kiểm độc lập (cùng người ở mọi buổi), GT là bản nháp qua duyệt, `ut-hpc` đã mất. Chỉ cân nhắc như ablation nếu sau khi chỉnh tham số và homography mà ReID vẫn là điểm nghẽn.
- **Giữ `max_cost_geometric = 0.9`**: bỏ nó (`mcgnull`) làm HOTA giảm 56 → 47.7, IDs 126 → 257.
- **Biến thể `gd3` (`max_ground_dist_m` 3.0) trùng hệt `base`**: cổng 2.0 m không phải thứ đang loại cặp.

## Số liệu đo được

Cấu hình chung: RTX A4000 16 GB (driver 580.178), YOLO11s 640 (TensorRT FP16 batch 3), OSNet x1.0 MSDC 512 chiều, 3 luồng 25 fps `sync=true`, `configs/lab/lab.mct.yaml`; GT nháp đã duyệt; chấm hộp ảnh, chỉ khung chú thích, IoU 0.5.

**Tốc độ và nhiệt (s2_1):** 76.7–76.9 FPS gộp (25.6 mỗi luồng, đúng tốc độ video); engine không đồng bộ 179.5 FPS mỗi luồng; nhiệt tối đa 61 °C, SM 1920 MHz, cờ hạ xung `0x0`.

**Độ trễ (s2_1, 3 lần chạy, đồng hồ thật):** theo khung p95 **203 ms (ĐẠT)**; thời gian tới Global ID p95 **9601 ms (KHÔNG ĐẠT)**, p50 484 ms, p90 3287 ms, n = 20 tracklet. Chưa đo 11 buổi còn lại.

**Xuyên camera, `base`, 12 buổi (lần chạy r1):** HOTA trung bình 55.7, AssA 38.8, IDF1 55.2, DetA ~82.5. Dao động theo buổi: s2_1 HOTA 69.3 / s2_2b 68.4 (cao) đến s4_1 33.5 / s3_2 43.9 / s3_1 46.3 (thấp).

**Độ chính xác bàn giao** (12 buổi; cấu hình `base` gộp 3 lần chạy; sweep chỉ r1):

| Loại | n | `base` (3 lần) | `base` (r1) | `combo` (r1) |
|---|---|---|---|---|
| Chồng lấn | 89 | 21.3% (Wilson 95% 14–31%) | 20% (18/89) | **38%** (34/89) |
| Không chồng lấn | 46 | 21.0% (12–35%) | 30% (14/46) | **39%** (18/46) |
| Quay lại cùng camera | 27 | 64.2% (45–79%) | 59% (16/27) | **70%** (19/27) |
| Tổng | 162 | 28.4% | 30% | **44%** |

Nhiễu giữa các lần chạy pipeline ở loại không chồng lấn khoảng ±9 điểm (r1 30% so với trung bình 3 lần 21%), nên chênh lệch nhỏ hơn mức đó không đáng tin.

**Sweep (HOTA / IDF1 / bàn giao tổng):**

| Nhóm | `base` | `combo` |
|---|---|---|
| s2 (chọn, 8 buổi) | 61.1 / 62.7 / 39% | **64.0 / 69.5 / 56%** |
| Giữ riêng (s1_1, s3_1, s3_2, s4_1) | 44.9 / 40.2 / 20% | **53.0 / 53.6 / 31%** |
| Cả 12 | 55.7 / 55.2 / 30% | **60.3 / 64.2 / 44%** |

`combo` = `max_cost` 0.4 + `homography_weight` 0.2 + `max_ground_dist_m` 3.0. Biến thể đơn lẻ ít tác dụng; `mc05` (`max_cost` 0.5) giúp riêng cặp không chồng lấn (s2: 68% so với 35%); `hw00` giúp chồng lấn ở nhóm giữ riêng (20%).

**Chẩn đoán s1_1 (bàn giao chồng lấn 0%, 8 lần/lần chạy, 6 tách + 2 nhầm):** cosine cùng người trung vị 0.89 (min 0.53), khác người 0.60 (max 0.87); khoảng cách điểm chân cùng người giữa cam01 và cam02 trung vị 0.54–1.53 m (3/11 cặp ≥ 2.06 m), khác người 1.24–9.83 m (3 cặp chỉ 1.2–2.3 m: hai người đi sát nhau); `max_ground_dist_m` = 2.0 m.

**Quét nhãn (12 buổi):** cặp cùng mã mà ngoại hình lệch (cos < 0.60): 1/379; cặp khác mã mà quá giống (cos > 0.85): 18/1214, trong đó 12 là s4_1 (áo giống nhau, chủ ý).

**Quét hộp nhãn vs hộp hệ thống (IoU 0.5):** khớp 95–98% ở mọi buổi. Chỗ lệch tập trung: s2_2 cam02 P02 khung 4–23 và 55–71; s1_1 cam03 P04 khung 442–477; s2_2c cam02 P01 (79 khung); s3_2 cam01 khung 177–203; s4_1 cam03 khung 286–373 (người sát mép dưới).

## Vướng mắc / chưa xong

- **Lỗi chính nằm ở liên kết, chưa chắc nằm ở nhãn.** Bàn giao 44% (`combo`) vẫn thấp, nhất là nhóm giữ riêng (chồng lấn 21%, 12/56). Lỗi chủ yếu là "tách" (cùng người, hai Global ID). Sai số homography 0.5–1 m gần bằng khoảng cách hai người đi sát nhau.
- **Sweep chỉ r1**, chưa có số cuối cho báo cáo (cần chạy lại biến thể tốt nhất 3 lần, tính khoảng tin cậy theo số bàn giao).
- **Thời gian tới Global ID chưa đạt** (s2_1 p95 9.6 s), mới đo 1/12 buổi.
- **Cặp cam02 ↔ cam03 chưa đo thời gian đi**, đang dùng khoảng chờ rộng 3–30 s; `measured` trong topology không bao gồm cặp này.
- **Cần người xem lại nhãn** (danh sách trong `SCAN_REPORT.md` và bảng quét nhãn): s2_3 cam01 khung 0–244 vs cam02 khung 0–71 (nghi tráo P01/P02, đúng đoạn đổi P06 → P01); s1_1 cam03 khung 442–477; s3_1 và s3_2 (4 cặp); và các chỗ lệch hộp ở mục trên.
- **Chưa so với MV3DT** (baseline camera chồng lấn).
- Cấu hình lab chưa được tinh chỉnh lần nào ngoài sweep này; kết quả trên là mốc xuất phát.
- **Khóa API Vast.ai đã dán vào chat** → nên tạo lại khóa mới ở Vast.ai (Account → API Key). Khóa hiện ở `~/.config/vastai/vast_api_key`.
- Script chạy 11 buổi (`run_all.sh`) và sweep (`sweep.py`) nằm ở scratchpad, **chưa vào repo**; nên chép thành công cụ trong repo nếu muốn tái lập.
- Chưa commit: file worklog này; `eval/gt/` (GT MOT, 1.5 MB, quỹ đạo người thật, chưa quyết định commit hay ignore) và `logs/` chưa nằm trong `.gitignore`.
- CLAUDE.md §2 vẫn ghi máy dev "không có GPU NVIDIA dùng được cho DeepStream"; máy hiện có RTX 3060 12 GB nhưng thiếu WSL2/Docker, nên mục đó chưa cần sửa cho tới khi cài.

## Bước tiếp theo

1. Xem lại nhãn theo danh sách ưu tiên (s2_2, s1_1, s2_3 đầu tiên), rồi chạy lại `make lab-gt` + `lab-eval` + `dataset_stats` cho buổi đã sửa (không cần GPU).
2. **Vòng 2 của sweep** (chọn trên s2, kiểm trên nhóm giữ riêng): `mc05` ghép với `hw02`/`hw00`, `combo` với `max_cost` 0.5. Nếu bàn giao vẫn dưới 50%, thử hiệu chỉnh lại homography bằng điểm chân trong GT trước khi nghĩ tới fine-tune.
3. Chạy lại biến thể tốt nhất với 3 lần chạy → số cuối cho chương 6; tính Wilson theo số bàn giao.
4. `tools.latency_report` cho cả 12 buổi (thời gian tới Global ID).
5. Quyết định về `eval/gt/` và `logs/` (`.gitignore`), tạo lại khóa API Vast.ai.
