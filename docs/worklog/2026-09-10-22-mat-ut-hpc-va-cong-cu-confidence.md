# 2026-09-10 (phiên 22) — Mất `ut-hpc`, sinh lại WildTrack trên `vast-gpu`, và quét ngưỡng detector THẬT

- **Mốc:** M4/M6 (hạ tầng + detector) | **Máy:** máy dev (Windows) + `vast-gpu` (Tesla T4) | **Thời lượng:** ~3h, **GPU $0.23**

## Mục tiêu phiên

- Bước 1 của phiên 21: đi vào detector/tracker đơn camera, vì 73.5% khối lượng Global ID rác
  là hộp detector báo nhầm. Đo trước bằng dữ liệu đã có xem **confidence có tách được hộp báo
  nhầm khỏi người thật không**, rồi mới thuê GPU quét `pre-cluster-threshold`.

Giữa phiên: **tài khoản `ut-hpc` bị khoá** (người dùng báo), kéo theo mọi fixture WildTrack
của pipeline (chỉ nằm trên cụm). Người dùng không có bản sao → quyết định thuê `vast-gpu` sinh
lại, gộp luôn phép quét ngưỡng thật vào cùng chuyến.

## Đã làm

**Hạ tầng thay `ut-hpc`**
- Venv test `~/.venvs/mct-test` (uv, CPython **3.10.20**, deps nhẹ) — thay head node cụm.
  Venv chấm điểm `~/.venvs/mct-eval` (numpy 1.23.5 / scipy 1.10.1) + `~/TrackEval`.
- Cập nhật CLAUDE.md §2 (khung cảnh báo, quy trình test mới, quy tắc 4–5), §11 (các mục
  `ut-hpc` thành lịch sử + 4 cạm bẫy mới), skill `ut-hpc` (lịch sử), `Makefile`,
  `pyproject.toml`, chú thích `configs/demo/streams_wildtrack.yaml`.
- Dò toàn bộ ổ A–G của máy dev: **không có bản sao** fixture nào.

**Công cụ mới**
- **`eval/diagnose_confidence.py`** + **`tests/test_diagnose_confidence.py`** (10 test): AUC
  confidence TP/FP mức detection (ghép IoU với chú thích WildTrack, tái dùng `match_frame`),
  quét ngưỡng, AUC của `mean`/`median`/`max` mức tracklet + cổng lọc tính theo khung, và
  `--write-filtered` sinh fixture lọc để mô phỏng việc nâng ngưỡng.

**Sửa lỗi**
- **`src/tools/unzip_wildtrack.py`**: 971/2807 ảnh không giải nén được trên Python có bản vá
  CVE-2024-0450 (xem QĐ 5). Sửa bằng cách tắt `_end_offset` cho từng entry. Sau sửa: 7×401
  ảnh, 0 lỗi. Không thêm test hồi quy: tái hiện cần một zip > 4 GiB có offset cuộn vòng.

**Chuyến `vast-gpu`** (instance `50499316`, Tesla T4 15 GB, driver 560.35.03, $0.167/h)
- Thử NVDEC trước tiên: PREROLLED → PLAYING → EOS. Đẩy repo + `models/` + chú thích bằng tar.
  `vast_bootstrap.sh` chạy sạch.
- Tải zip WildTrack thẳng từ EPFL trên instance (~45 MB/s, 6.8 GB), giải nén, sửa ffmpeg
  (QĐ 6), đóng 7 video — **cam01 173.0 MB, 400 khung, trùng từng MB với phiên 10**.
- Pipeline 7 luồng có ReID, `--publish` + `record_metadata`, với `pre-cluster-threshold`
  (lớp person) = **0.25 / 0.40 / 0.55**, rồi lặp **0.25 và 0.40 thêm 2 lần** (người dùng duyệt)
  để đo nhiễu giữa các lần chạy. 7 fixture × 2800 message, kéo về `data/fixtures/`.
- Kéo 7 video về `data/wildtrack_video/` (1.1 GB) để lần sau khỏi đóng lại.
- Zip WildTrack gốc (6 807 496 358 byte) tải thẳng từ EPFL về `data/` trên máy dev (nối qua
  3 lần đứt kết nối bằng `curl -C -`), giải nén ra `data/wildtrack/Image_subsets`: 2807 ảnh,
  0 lỗi (CRC từng entry khớp) — bản sửa `unzip_wildtrack.py` chạy đúng cả trên Python
  3.10.20 của máy dev (1714 entry phải vá offset).
- Toàn bộ các bước chạy trên instance gom vào **`docker/vast_wildtrack.sh`** (`nvdec` /
  `data` / `run <ngưỡng>`), cạnh `vast_bootstrap.sh` — tái lập được chuyến này bằng 4 lệnh.
- Sinh lại `data/fixtures/wildtrack_7cam.jsonl` (ground-truth, `--no-reid`) trên máy dev:
  2800 message / **42 606 hộp** / 313 danh tính — trùng khít phiên 15.

## Quyết định kỹ thuật

**1. Một lần chạy pipeline KHÔNG đủ để so hai cấu hình — nhiễu giữa các lần chạy lớn cỡ hiệu
ứng.** Cùng video, cùng config, cùng code, cùng GPU-loại: HOTA ngưỡng 0.25 lần lượt 16.210
(phiên 21) / 15.609 / 15.294 / 16.208. Số detection gần như trùng (34 597 / 34 673 / 34 627 /
34 619), nhưng AssA và số Global ID dao động (433 / 392 / 418 / 411) — nhiễu nằm ở **cách NvDCF cấp id**
theo nhịp gom batch của streammux (`sync=true`, đồng hồ thật), không ở detector. Hệ quả cho
chương 6: mọi chênh lệch dưới ~1 HOTA giữa hai cấu hình pipeline phải có **≥3 lần chạy mỗi
bên**, báo cáo trung bình ± độ lệch. Các so sánh trong `src/mct` (phiên 16–21) KHÔNG dính
nhiễu này vì chúng chạy lại engine trên CÙNG một fixture — engine thì tất định.

**2. Confidence có mang thông tin, nhưng chỉ ở mức vừa: AUC 0.750 (detection), 0.789
(`mean` mức tracklet).** Đủ để một ngưỡng có ích, không đủ để một ngưỡng giải quyết được vấn
đề: nâng lên 0.40 bỏ 38.4% hộp báo nhầm nhưng cũng mất 11.7% hộp đúng.

**3. Lọc fixture (mô phỏng) ĐÁNH GIÁ THẤP lợi ích của việc nâng ngưỡng — không dùng nó để
chốt ngưỡng.** Ở 0.40: mô phỏng HOTA 15.51 vs chạy thật 15.995 ± 0.395 (n=3) — riêng HOTA thì chênh nằm
trong nhiễu; ở 0.55: 14.67 vs 15.42 (n=1). Bằng chứng chắc hơn là CẤU TRÚC track, lặp lại ở
cả 3 lần chạy thật: khi tracker thật được ăn ít hộp rác hơn, nó dựng track SẠCH hơn (thuần khiết
trung vị 0.857–0.860 vs 0.833, 804–805 vs 891 local track, Global ID 294–340 vs 353 ở 0.40) — mô phỏng giữ nguyên track của lần
chạy 0.25 nên không thấy được hiệu ứng đó. Công cụ vẫn giá trị cho câu hỏi "confidence có
thông tin không" (AUC), chỉ không thay được việc chạy lại pipeline.

**4. Không đổi `pre-cluster-threshold` của hệ thống thật, và không đưa cổng tracklet vào
`src/mct`.** Với n=3, thứ CHẮC là 0.25 → 0.40 đổi DetA (−0.84, t≈−12) lấy ít Global ID hơn (407 → 318)
và nhiều khả năng IDF1 cao hơn (+2.05, t≈3.1); HOTA thì KHÔNG phân biệt được (+0.29, t≈0.8).
Tức trên WildTrack, nâng ngưỡng là một đánh đổi gần hoà về HOTA, không phải một cải tiến —
và chỉ thấy được điều đó nhờ chạy lặp (hai lần đầu cho +0.73, dễ đọc thành "cải tiến").
0.55 (n=1) mất DetA rõ (20.4) mà HOTA không hơn. Nhưng WildTrack chạy 2 fps, mọi
camera chồng lấn — đúng loại dataset mà phiên 11/12 đã chốt KHÔNG được dùng để chỉnh tham số
tracker/detector. Ghi xu hướng, chốt ở M6 trên dữ liệu tự thu 25 fps. Cổng tracklet `mean`
cũng chờ: cùng lượng hộp rác bỏ được, ngưỡng detector thật cho kết quả tốt hơn cổng hậu kỳ
(QĐ 3), nên cổng trong `src/mct` chỉ đáng làm nếu M6 cho thấy không chỉnh được detector.

**5. `unzip_wildtrack.py` hỏng im lặng theo bản vá bảo mật của Python, không theo code.**
Bản vá CVE-2024-0450 (Ubuntu backport vào 3.10.12) thêm phép kiểm "Overlapped entries
(possible zip bomb)" dựa trên offset của entry kế tiếp. Offset của zip WildTrack cuộn vòng qua
mốc 4 GiB (phiên 4), nên mọi entry sau mốc bị từ chối ngay ở offset đúng; ba độ lệch đều
thất bại vì những lý do khác nhau. `unzip` hệ thống cũng từ chối. Tắt phép kiểm cho từng
entry là an toàn vì tool vẫn đòi kích thước giải nén khớp `file_size`. Phương án bị loại:
dùng `unzip` (cùng lỗi) hoặc ghim Python cũ (không kiểm soát được trên máy thuê).

**6. Hàng đợi `QueuedFramePublisher` (nợ phiên 19): trần 2000 dư xa ở tải này.** 7 lần
chạy, mỗi lần 2800 khung: **0 khung bỏ, 0 lỗi, sâu nhất 5–6**. Tải là 7 luồng × 2 fps = 14
khung/s — thấp hơn nhiều so với 4 luồng × 25 fps = 100 khung/s của hệ thống thật, nên con số
này đóng nợ cho WildTrack, chưa đóng cho kịch bản 25 fps.

## Số liệu đo được

**Cấu hình pipeline:** vast.ai `50499316`, Tesla T4 15 GB, driver 560.35.03, DeepStream
7.1.0 (image `nvcr.io/nvidia/deepstream:7.1-triton-multiarch`), YOLO11s COCO FP16 input 640
(lọc lớp person tại nvinfer), NvDCF + ReID OSNet `osnet_x1_0_msdc_dg` (TensorRT FP16), tracker
960×544, streammux 1920×1080 batch 7, `sink.sync: true`. Nguồn: 7 video WildTrack (400
khung/camera, 2 fps, CRF 18). Đổi đúng MỘT biến: `pre-cluster-threshold` của `[class-attrs-0]`.

**Cấu hình chấm:** engine `configs/demo/wildtrack_ds.mct.yaml` + `wildtrack.topology.yaml` +
homography 7 camera, đường online `python -m mct --source`; `tools.export_trackeval --mode mct
--gt-fixture data/fixtures/wildtrack_7cam.jsonl`; TrackEval MotChallenge2DBox. Máy dev
Windows, `~/.venvs/mct-test` / `mct-eval`. Engine ~20–27 s mỗi fixture.

### 1. Quét ngưỡng detector — pipeline THẬT

| ngưỡng | lần | HOTA | DetA | AssA | DetRe | DetPr | IDF1 | Global ID | #det |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 1 | 15.609 | 24.199 | 10.281 | 33.07 | 40.77 | 20.349 | 392 | 34 673 |
| 0.25 | r2 | 15.294 | 24.124 | 9.928 | 32.96 | 40.69 | 18.995 | 418 | 34 627 |
| 0.25 | r3 | 16.208 | 24.035 | 11.189 | 32.85 | 40.57 | 20.882 | 411 | 34 619 |
| 0.40 | 1 | 15.893 | 23.239 | 11.028 | 28.14 | 48.06 | 21.656 | 294 | 25 077 |
| 0.40 | r2 | 16.431 | 23.381 | 11.742 | 28.28 | 48.29 | 22.787 | 320 | 25 078 |
| 0.40 | r3 | 15.660 | 23.224 | 10.756 | 28.13 | 48.03 | 21.943 | 340 | 25 080 |
| 0.55 | 1 | 15.419 | 20.396 | 11.868 | 22.63 | 55.57 | 22.674 | 260 | 17 469 |

**Trung bình ± độ lệch chuẩn (n = 3 lần chạy mỗi ngưỡng), và t Welch của hiệu 0.40 − 0.25:**

| | HOTA | DetA | AssA | IDF1 | Global ID |
|---|---|---|---|---|---|
| 0.25 | 15.704 ± 0.464 | 24.119 ± 0.082 | 10.466 ± 0.650 | 20.075 ± 0.973 | 407 ± 13 |
| 0.40 | 15.995 ± 0.395 | 23.281 ± 0.087 | 11.175 ± 0.509 | 22.129 ± 0.588 | 318 ± 23 |
| hiệu | +0.29 | −0.84 | +0.71 | +2.05 | −89 |
| t | **0.8** | **−12** | 1.5 | **3.1** | **−5.8** |

Đọc: DetA giảm và số Global ID giảm là CHẮC; IDF1 tăng là có khả năng (t≈3 với n=3);
AssA và HOTA **không tách được khỏi nhiễu** giữa các lần chạy.

Đối chiếu: phiên 18–21 (cùng cấu hình 0.25, fixture sinh 2026-09-04) HOTA 16.210 / AssA
11.131 / IDF1 20.918 / 433 Global ID.

### 2. Mô phỏng bằng lọc fixture 0.25 (`--write-filtered`)

| ngưỡng | HOTA | DetA | AssA | IDF1 | Global ID | local track | thuần khiết p50 |
|---|---|---|---|---|---|---|---|
| 0.40 mô phỏng | 15.509 | 23.741 | 10.352 | 20.966 | 353 | 891 | 0.833 |
| 0.40 thật (lần 1) | 15.893 | 23.239 | 11.028 | 21.656 | 294 | 804 | 0.857 |
| 0.55 mô phỏng | 14.671 | 21.314 | 10.277 | 20.995 | 305 | 788 | 0.889 |
| 0.55 thật | 15.419 | 20.396 | 11.868 | 22.674 | 260 | 634 | 0.923 |

### 3. Confidence tách TP/FP tới đâu (fixture 0.25 lần 1, `eval/diagnose_confidence.py`)

34 673 detection = 19 168 TP + 15 408 FP + 97 tracker-only (`conf = −0.1`, loại khỏi thống kê).

| | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| conf TP | 0.322 | 0.529 | 0.685 | 0.792 | 0.883 |
| conf FP | 0.268 | 0.344 | 0.454 | 0.601 | 0.814 |

**AUC detection 0.750.** Mức tracklet (279 track FP / 11 928 khung, 646 track thật / 22 745
khung): AUC `mean` **0.789**, `median` 0.787, `max` 0.748.

| ngưỡng | %TP giữ | %FP bỏ | precision | cổng `mean`: %khung FP bỏ | %khung thật mất |
|---|---|---|---|---|---|
| 0.30 | 96.6 | 13.7 | 0.582 | 0.8 | 0.1 |
| 0.40 | 88.3 | 38.4 | 0.641 | 23.3 | 3.1 |
| 0.50 | 78.5 | 58.5 | 0.702 | 63.3 | 18.7 |
| 0.60 | 65.0 | 74.8 | 0.763 | 89.6 | 41.1 |

### 4. Phân rã rác (fixture 0.25 lần 1, cùng công cụ phiên 20/21)

208/392 Global ID rác (53.1% số ID); trong đó **75.9% khối lượng là hộp detector báo nhầm**
(phiên 21: 73.5%) — kết luận của phiên 21 tái lập trên fixture sinh lại. Ở 0.40 thật: rác
116/294, 71.6% khối lượng vẫn là báo nhầm.

### 5. Hạ tầng

- Test: **518 passed, 5 skipped**, ruff sạch (`~/.venvs/mct-test`, CPython 3.10.20).
- Tải EPFL trên instance ~45 MB/s; đường về máy dev chỉ ~1–1.5 MB/s (nút thắt của cả chuyến).
- Chi phí: **$0.23** (credit vast.ai 6.424 → 6.196), instance 14:57 → 16:05 UTC. Đã huỷ,
  `vastai show instances` = 0. Phần lớn thời gian thuê là chờ kéo dữ liệu về máy dev.

## Vướng mắc / chưa xong

- `ds_wildtrack_7cam_onnx_gtbox.jsonl` (phiên 13/21) CHƯA sinh lại. Ảnh đã có trên máy dev;
  còn thiếu `pip install -e ".[reid]"` (onnxruntime + opencv) rồi `tools/reembed_fixture.py`.
- Mọi con số phiên 12–21 nay đã có nền dữ liệu mới nhưng **không khớp từng chữ số** với bản cũ
  (QĐ 1). Số của chương 6 nên lấy từ fixture mới + báo cáo kèm độ lệch giữa các lần chạy.
- Chỗ fine-tune ở M6 (nếu cần) chưa chốt — quy tắc cũ dựa trên việc có `ut-hpc`.

## Bước tiếp theo

1. Chương 6: báo cáo mọi so sánh cấu hình PIPELINE bằng trung bình ± độ lệch trên ≥3 lần
   chạy (bảng 1 là mẫu). So sánh chỉ trong `src/mct` (cùng fixture) thì một lần là đủ — engine
   tất định.
2. Sinh lại `onnx_gtbox` trên máy dev (ảnh đã sẵn ở `data/wildtrack/Image_subsets`).
3. Chuẩn bị M6: kịch bản thu dữ liệu 25 fps có đồng thuận; ở đó mới quét lại
   `pre-cluster-threshold` (0.25 / 0.40) với ≥3 lần chạy mỗi mức.
