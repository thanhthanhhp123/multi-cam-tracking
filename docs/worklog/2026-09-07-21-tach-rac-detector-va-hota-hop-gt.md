# 2026-09-07 (phiên 21) — 217 ID rác: 3/4 là hộp detector báo nhầm; và hộp GT quy ra +55% HOTA

- **Mốc:** M4 (đóng góp chính) + M6 (đánh giá) | **Máy:** máy dev (soạn) + `ut-hpc` (chạy + đo) | **Thời lượng:** ~2h, **không tốn GPU**

## Mục tiêu phiên

Trả hai trong ba món nợ mà phiên 20 liệt kê, theo đúng thứ tự khuyến nghị:

1. **Tách 217 Global ID "rác"** (phiên 20) làm hai — hộp detector báo nhầm vs. người thật
   mà bộ gán nhãn từ chối — bằng cách đối chiếu THẲNG với chú thích WildTrack gốc thay vì
   qua bảng `.gt.json`. Đây là thứ quyết định phiên sau đi sửa detector hay sửa liên kết.
2. **Chấm `onnx_gtbox` bằng HOTA** (nợ từ phiên 15, nay sang phiên thứ năm) — quy đổi mức
   +22% F1 của hộp ground-truth (phiên 13) sang điểm HOTA.

Món thứ ba (chạy `--publish` trên `vast-gpu` để chốt trần hàng đợi) **chưa làm** — tốn tiền
thuê và cần xác nhận người dùng cho từng lệnh.

## Đã làm

- **`eval/diagnose_junk_ids.py`** (mới) — với mỗi Global ID rác, lấy các tracklet
  `(cam_id, local_track_id)` của nó, tra hộp từng khung trong fixture, ghép IoU với chú
  thích WildTrack gốc (`annotations_positions/*.json`) — đúng cách `ds_wildtrack_gt.py` gán
  nhãn nhưng **KHÔNG áp ngưỡng loại** — rồi phân loại theo tỉ lệ khung trùng người thật.
  Tái dùng `collect_votes`/`gt_index` của `ds_wildtrack_gt.py` và `parse_raw_detections`
  của `wildtrack_to_fixture.py` để phép ghép giống hệt bộ gán nhãn.
- **`tests/test_diagnose_junk_ids.py`** (6 test) — ba kịch bản phân loại có đáp án đếm tay
  (hộp không bao giờ chạm người thật / bám người nhưng quá ngắn / bám người nhưng phiếu
  lẫn), bỏ phiếu gộp lên Global ID theo số khung, và một đường end-to-end nhỏ dựng
  annotation + fixture + DB SQLite tạm.
- Chạy lại **trọn đường online → export → TrackEval** cho hai fixture:
  `ds_wildtrack_7cam` (mốc) và `ds_wildtrack_7cam_onnx_gtbox` (hộp GT, embedding ONNX).
  Mốc **tái lập chính xác** HOTA 16.210 / AssA 11.131 / IDF1 20.918 / 433 ID trước khi đo
  bản GT-box.
- **503 passed, 8 skipped**, ruff sạch trên `ut-hpc` (Python 3.10.12, `venv-test`).

## Quyết định kỹ thuật

**1. "Rác" phải được đo bằng KHỐI LƯỢNG KHUNG, không bằng số ID, và ngưỡng cắt là
`match_rate < 0.30`.** `match_rate` = tỉ lệ khung của tracklet ghép được IoU≥0.5 với một
người WildTrack BẤT KỲ. Phân bố tách bạch sạch: nhóm `detector_fp` có trung vị `match_rate`
= **0.00** (p95 chỉ 0.17), nhóm `real_person` trung vị **0.78** (p5 0.35). Không có vùng
xám — chọn 0.30 hay 0.50 cho cùng kết luận. Phương án bị loại: đếm số ID rồi chia đôi. Một
ID rác 3 khung và một ID rác 187 khung ăn vào AssA khác nhau hai bậc độ lớn.

**2. Con số phiên 20 nói "52.0% số khung là rác" là của `frames_unlabeled`, RỘNG hơn "khung
của Global ID rác".** `diagnose_global_ids.py` in tỉ lệ rác kèm `frames_unlabeled /
frames_total` = 17916/34475 = 52.0%, nhưng `frames_unlabeled` đếm MỌI lượt xuất hiện không
nhãn, kể cả những cái nằm trong Global ID CÓ nhãn (mảnh dư/chồng không nhãn). Khung thuộc
riêng 217 Global ID rác là **12174 = 35.3%** toàn hệ thống. Đây là con số đúng để đặt cạnh
"detector báo nhầm chiếm 73.5% phần đó". Ghi lại vì phiên 20 đã đưa 52.0% vào bảng chương 6.

**3. Nhóm "người thật bị từ chối" KHÔNG sửa được ở bộ gán nhãn — 114/122 tracklet trượt vì
ĐỘ THUẦN KHIẾT, không vì số khung.** Tức NvDCF đã trộn ≥2 danh tính WildTrack vào một
`local_track_id` (id-switch trong một camera); hộp nằm trên người thật nhưng track "bẩn".
Nới `--min-purity` không cứu được vì track thật sự lẫn người. Nên 26.5% khối lượng rác này
là **lỗi tracker đơn camera**, không phải lỗi `src/mct` và cũng không phải chỗ nới bảng
nhãn. Chỉ 8/122 trượt vì `it_khung` (dưới 3 khung khớp). 0 tracklet vượt cả ba ngưỡng mà
vẫn vắng trong `.gt.json` — phép phân loại nhất quán với `ds_wildtrack_gt.py`.

**4. Chấm `onnx_gtbox` xác nhận: chất lượng hộp là đòn bẩy lớn nhất còn lại, và nó nằm
NGOÀI `src/mct`.** Fixture `onnx_gtbox` chỉ giữ detection ghép được GT rồi cắt crop từ hộp
GT (phiên 13) — tức mô phỏng detector có precision hoàn hảo và hộp khít. Kết quả: HOTA
16.21 → **25.18** (+55%), gần như toàn bộ do DetA (24.1 → 42.8, vì DetPr 40.7% → 97.5%,
CLR_FP 15407 → 0). Nhưng **AssA cũng lên 11.13 → 14.81 (+33%)** — phần này LÀ bước liên kết
tốt lên, do (a) hình học chính xác (hộp GT siết `d_ground` của cặp cùng người, phiên 13),
và (b) ít tracklet rác hơn để làm nhiễu bước lọc ứng viên. Số Global ID rơi 433 → 262, và
`diagnose_global_ids` trên bản GT-box cho **rác 217 → 59** — 158/217 ID rác trên pipeline
thật đúng là hộp detector, biến mất khi bỏ FP. Phần vỡ (133 → 120) và gộp (63 → 63) gần như
không đổi: hộp GT không sửa được hai thứ đó, chúng là giới hạn của khâu so ngoại hình.

## Số liệu đo được

**Cấu hình:** fixture `ds_wildtrack_7cam.jsonl` / `ds_wildtrack_7cam_onnx_gtbox.jsonl`
(WildTrack 7 camera, 2 fps), config `configs/demo/wildtrack_ds.mct.yaml` +
`wildtrack.topology.yaml` + homography 7 camera, đường online đầy đủ
(`python -m mct --source` → `tools.export_trackeval --mode mct --gt-fixture wildtrack_7cam.jsonl`
→ TrackEval MotChallenge2DBox, IoU 0.5, `DO_PREPROC=False`). Head node `ut-hpc`, Python
3.10.12. Một lượt online ~14.5 s + ~3 s chấm. DB mốc: `mct-ds-s20.db` (433 ID, HOTA 16.210,
đúng lần chạy của phiên 18–20).

### 1. Phân rã 217 Global ID rác (`eval/diagnose_junk_ids.py`)

| | #ID | %ID | #khung | % khung rác | sửa ở đâu |
|---|---|---|---|---|---|
| **DETECTOR BÁO NHẦM** (`match_rate < 0.30`) | **146** | 67.3% | **8944** | **73.5%** | detector / nvtracker |
| **NGƯỜI THẬT, BẢNG NHÃN LOẠI** | **71** | 32.7% | **3230** | **26.5%** | tracker đơn camera (id-switch) |

217 Global ID rác = 12174 khung = **35.3%** toàn hệ thống (không phải 52.0% — xem QĐ 2).

**Phân bố `match_rate` (tỉ lệ khung trùng người thật):**

| | p5 | p25 | p50 | p75 | p95 | n |
|---|---|---|---|---|---|---|
| `detector_fp` | 0.00 | 0.00 | 0.00 | 0.03 | 0.17 | 196 |
| `real_person` | 0.35 | 0.60 | 0.78 | 0.88 | 1.00 | 124 |
| `detector_fp` #khung/track | 3 | 8 | 21 | 63 | 187 | 196 |
| `real_person` #khung/track | 4 | 8 | 16 | 32 | 63 | 124 |

Track rác của detector còn DÀI HƠN track người thật (p50 21 vs 16, p95 187 vs 63) — nvtracker
dựng track bền từ một hộp báo nhầm dai dẳng (cột, phản chiếu, đồ vật).

**Nhóm "người thật bị loại" vì sao:** 114 `khong_thuan` (purity < 0.7 — track lẫn người),
8 `it_khung` (< 3 khung khớp). Chúng trùng **93 danh tính** WildTrack, trong đó **50 danh
tính CHƯA có** trong `.gt.json` — tức bộ gán nhãn đang giấu 50/167 danh tính bị trượt, và
bảng chấm hiện tại đánh giá trên mẫu số hẹp hơn thực tế (nhắc lại vướng mắc phiên 20).

### 2. Chấm `onnx_gtbox` bằng HOTA (nợ phiên 15)

| | mốc `ds_wildtrack_7cam` | `onnx_gtbox` (hộp GT) | Δ |
|---|---|---|---|
| **HOTA** | 16.210 | **25.184** | **+55%** |
| DetA | 24.123 | 42.816 | +77% |
| **AssA** | 11.131 | **14.813** | **+33%** |
| DetRe / DetPr | 32.96 / 40.74 | 43.29 / **97.49** | |
| IDF1 | 20.918 | 25.688 | +23% |
| MOTA / CLR_FP | 2.68 / 15407 | 38.80 / **0** | |
| Global ID sinh ra / GT | 433 / 313 | **262** / 313 | |
| rác (`diagnose_global_ids`) | 217 (50.1%) | **59 (22.5%)** | |
| vỡ / gộp | 133 / 63 | 120 / 63 | ~0 |

**Đọc:** +22% F1 tự chế của phiên 13 quy ra **+8.97 HOTA tuyệt đối (+55%)**. Nhưng 2/3 mức
đó là DetA (bỏ 15407 hộp FP) — thứ `src/mct` không đụng tới. Phần của bước liên kết là
**AssA +33%**, đến từ hình học sạch hơn + ít rác hơn. Vỡ và gộp không nhúc nhích: hộp đẹp
không sửa được khâu ngoại hình.

## Vướng mắc / chưa xong

- **Nợ phiên 19 CHƯA trả:** chạy `--publish` trên `vast-gpu`, đọc `n_dropped`/`max_depth`
  để chốt trần hàng đợi `QueuedFramePublisher`. Cần thuê GPU + xác nhận người dùng từng lệnh.
- `match_rate` thấp không CHỈ nghĩa "hộp ma": WildTrack chú thích ~2 fps và bỏ sót người ở
  rìa quảng trường, nên một phần nhỏ `detector_fp` có thể là người thật mà GT không đánh
  dấu. Trung vị `match_rate` = 0.00 nói phần đó nhỏ, nhưng không phải 0.
- `onnx_gtbox` chạy được vì nó dùng lại `.gt.json` của `ds_wildtrack_7cam` (local id giữ
  nguyên qua `reembed_fixture.py`). Nếu sau này sinh lại fixture đó phải sinh lại cả bảng.
- Kết quả `onnx_gtbox` là **cận trên "nếu detector hoàn hảo"**, không phải mục tiêu đạt
  được: fixture đã lọc sạch FP bằng chính GT. Đừng đặt 25.18 HOTA cạnh 16.21 như "cải tiến".
- Mọi con số vẫn trên WildTrack 2 fps, mọi camera chồng lấn. **Đừng chốt tham số nào theo
  dataset này.**

## Bước tiếp theo

1. **Phiên sau nên đi vào detector/tracker đơn camera, không phải `max_cost`.** 73.5% khối
   lượng rác là hộp FP (25.9% toàn hệ thống) và 26.5% là id-switch của NvDCF — cả hai nằm
   ngoài `src/mct`. Cụ thể: quét `pre-cluster-threshold`/`nms-iou` của nvinfer, hoặc
   `minTrackerConfidence`/`maxShadowTrackingAge` của NvDCF, đo lại DetPr + số ID rác.
2. Nợ phiên 19: `--publish` trên `vast-gpu` để chốt trần hàng đợi.
3. (nhỏ) Cân nhắc nới `ds_wildtrack_gt.py` để thu hồi 8 tracklet `it_khung` + xét lại 50
   danh tính bị trượt — mở rộng mẫu số của bảng chấm.
