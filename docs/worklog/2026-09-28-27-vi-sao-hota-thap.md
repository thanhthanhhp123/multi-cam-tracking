# 2026-09-28 (phiên 27): Vì sao HOTA thấp. Khoảng 54% điểm sai trong vùng là lỗi liên kết (tách người 38%, gộp nhầm 15%), 37% là hộp detector không khớp ai. Nới `max_cost` làm TỆ đi trên cả hai cách chấm

- **Mốc:** M4 + M6 | **Máy:** máy dev (CPU, không thuê GPU) | **Thời lượng:** ~2h

## Mục tiêu phiên

- Đầu phiên: chạy bù NMS mặt đất cho phiên 26 (đã ghi vào worklog phiên 26), cập nhật CLAUDE.md §7.
- Sau đó trả lời câu hỏi "HOTA như vậy hơi thấp": điểm sai trong vùng chú thích đến từ đâu,
  và có đòn bẩy nào rẻ (CPU, không thuê GPU) để tăng không.

## Đã làm

Mọi phép đo dùng lại 3 lần chạy pipeline 640 của phiên 25 (`data/fixtures/ds_wildtrack_7cam_r640n_r{1,2,3}.jsonl`
+ `data/s25/R640_r*/mct.db`). Chấm theo điểm mặt đất, chỉ trong vùng, T = 1 m, không NMS,
trừ khi ghi khác. Ba script chẩn đoán là **script tạm, CHƯA vào git**. Chúng được lưu ở
`data/s27/scripts/` (gitignored, chỉ có trên máy dev). Xem "chưa xong".

1. **`gp_fp_breakdown.py`**: với mỗi điểm dự đoán, đếm số camera góp vào điểm đó, khoảng cách
   tới người thật gần nhất, và khoảng cách tới điểm dự đoán khác gần nhất. Thử bộ lọc "chỉ nhận
   điểm có ≥ 2 hoặc ≥ 3 camera" rồi chấm lại HOTA. → `data/s25/gp_fp_breakdown.json`.
   Phép khớp Hungarian tự viết cho TP/FP lệch dưới 1% so với `CLR_TP`/`CLR_FP` của TrackEval
   (7 157 / 9 828 so với 7 069 / 9 916).
2. **`gp_merge_estimate.py`**: ước lượng xem một bước GỘP Global ID sau cùng được bao nhiêu. Gộp hai
   Global ID khi điểm của chúng ở gần nhau (≤ 1–1.5 m) trong ≥ 3–5 khung và ≥ 60–80% số khung cả
   hai cùng có mặt, đồng thời không vi phạm ràng buộc loại trừ (hai detection của cùng một camera
   trong cùng một khung). Phép gộp dùng thông tin của toàn chuỗi, nên đây là ước lượng lạc quan.
   → `data/s25/gp_merge_estimate.json`.
3. **`gp_fp_identity.py`**: gán mỗi detection cho một personID bằng Hungarian IoU ≥ 0.5 với hộp
   chú thích, theo từng (camera, khung), giống quy tắc của `tools/ds_wildtrack_gt`. Sau đó (a) đo
   mức một người bị tách qua các camera, (b) phân loại mỗi điểm TP/FP theo thành phần detection của
   Global ID trong khung đó. → `data/s25/gp_fp_identity.json`.
4. **Lý do tạo Global ID mới**: đọc cột `appearances.reason` trong DB của engine.
5. **Quét `max_cost` 0.90 → 1.00 / 1.10**: chạy lại engine (`eval.compare_oracle_tracker`, kịch bản A)
   trên 3 fixture 640, chấm theo cả hộp ảnh lẫn điểm mặt đất. Config được sinh bằng `sed` từ
   `configs/demo/wildtrack_ds.mct.yaml`, chỉ khác dòng `max_cost` (đã kiểm bằng `grep -c`), lưu ở
   `data/s27/cfg/`. Kết quả ở `data/s27/{box,gp}_mc{1.00,1.10}.json`. Mốc 0.90 lấy từ phiên 25
   (`data/s25/compare_oracle_tracker.json`, `data/s25/ground_plane_nms.json`). `src/mct` không đổi
   từ 2026-09-19, trước khi phiên 25 chạy, nên các số so sánh được với nhau.

Không sửa `src/` hay `configs/`.

**Đã thử rồi bỏ**
- Đòn bẩy "chỉ nhận điểm được nhiều camera xác nhận" (đề xuất đầu phiên): **bác bỏ** (số liệu mục 2).
- Bước gộp Global ID sau cùng: được tối đa +0.7 HOTA dù đã "nhìn trước" toàn chuỗi (số liệu mục 3).
  Không đáng đưa vào engine.
- Nới `max_cost`: tệ hơn (số liệu mục 5).

## Quyết định kỹ thuật

**1. Giữ `max_cost: 0.90`.** Nới ra 1.00 / 1.10 làm số Global ID giảm (406 → 329 → 203), đúng
như mong muốn là gộp được nhiều hơn. Nhưng AssA giảm trên cả hai cách chấm, tức phần gộp thêm
chủ yếu là gộp NHẦM người. Với tracklet bị loại vì ngưỡng, ứng viên tốt nhất thường không phải
đúng người. Vì vậy ngưỡng không phải chỗ đặt sai: chi phí hiện tại không phân biệt được đúng
với sai ở vùng biên. Phiên 18 đã thấy điều này khi CHƯA bật nối mảnh cùng camera; phiên này
xác nhận lại khi đã bật.

**2. Không làm bước gộp Global ID sau cùng, không lọc điểm theo số camera.** Cả hai bị các số
đo dưới đây bác bỏ trước khi viết vào engine, tức là tốn một ước lượng CPU thay vì một vòng sửa
engine rồi chạy lại. Nguyên nhân chung: Global ID vốn đã lẫn người (15% điểm sai là gộp nhầm,
13% điểm đúng cũng vậy), nên ràng buộc loại trừ chặn gần hết các cặp muốn gộp (65 ứng viên chỉ
gộp được 20). Còn bộ lọc theo số camera thì đụng đúng hệ quả của việc tách người: 67% điểm ĐÚNG
chỉ có 1 camera góp vào.

**3. Kết luận cho báo cáo: nút thắt của `src/mct` trên WildTrack là CHẤT LƯỢNG ĐẶC TRƯNG, không
phải cách chọn tham số.** Tách người và gộp nhầm đánh đổi lẫn nhau qua ngưỡng, không giảm cùng
lúc được. Muốn giảm cả hai thì phải có đặc trưng phân biệt tốt hơn. Theo phiên 13, hộp detector
làm hình học kém hẳn: sai số điểm chân khi so vị trí 0.21 m với hộp GT, 0.74 m với hộp detector.
Còn ngoại hình DeepStream ở phiên 11–12 có ngưỡng khác hẳn fixture ONNX. WildTrack đông người
và chỉ 2 fps, là trường hợp khó cho cả hai loại đặc trưng.

## Số liệu đo được

**Cấu hình.** Như phiên 25/26: WildTrack 7 camera, 400 khung, 2 fps; pipeline trên `vast-gpu`
T4, DeepStream 7.1, YOLO11s COCO FP16 640, `pre-cluster-threshold` 0.25, NvDCF + ReID OSNet
`osnet_x1_0_msdc_dg`. Engine chạy trên máy dev bằng CPython 3.10.20 (`mct-test`) với
`configs/demo/wildtrack_ds.mct.yaml` (`max_cost` 0.90, `homography_weight` 0.4,
`max_ground_dist_m` 2.0, `ground_gap_policy: reject`, `same_camera_stitch: true`). Chấm bằng
TrackEval (`mct-eval`). n = 3, trung bình ± độ lệch chuẩn mẫu.

### 1. Số camera góp vào mỗi điểm (trong vùng)

| số camera | TP | FP |
|---|---|---|
| 1 | 4 780 (**67%**) | 6 724 |
| 2 | 1 407 | 2 170 |
| 3 | 759 | 708 |
| ≥ 4 | 210 | 225 |

Về khoảng cách, 69% FP có một điểm dự đoán khác trong vòng 1 m. Tính theo người thật gần nhất:
~48% FP ở trong vòng 1 m (người đó đã khớp với điểm khác), 18% ở 1–2 m, 22% ở 2–5 m, 12% xa
hơn 5 m. Cách phân loại theo khoảng cách này lẫn hai thứ với nhau: bản trùng, và người thật
đứng sát nhau trong đám đông. Mục 4 thay nó bằng cách phân loại theo danh tính.

### 2. Lọc theo số camera tối thiểu (HOTA trong vùng)

| tối thiểu | HOTA | DetA | AssA | TP | FP | FN |
|---|---|---|---|---|---|---|
| 1 (hiện tại) | 31.10 ± 0.85 | 31.06 | 31.64 | 7 069 | 9 916 | 2 449 |
| 2 | 30.80 ± 1.44 | 25.85 | 37.58 | 3 418 | 2 063 | 6 100 |
| 3 | 21.44 ± 3.03 | 13.87 | 33.48 | 1 589 | 314 | 7 929 |

Kết quả giống vậy ở 960 (30.20 → 29.12 → 24.06) và 1280 (28.16 → 28.25 → 20.30).

### 3. Bước gộp Global ID sau cùng (ước lượng lạc quan, nhìn trước toàn chuỗi)

| tiêu chí gộp (bán kính, số khung tối thiểu, tỉ lệ) | HOTA | AssA | FP | Global ID | số lần gộp (r1/r2/r3) |
|---|---|---|---|---|---|
| không gộp | 31.10 ± 0.85 | 31.64 | 9 916 | 324 | — |
| 1.0 m, 3, 0.7 | 31.50 ± 1.06 | 31.68 | 9 354 | 297 | 20 / 38 / 23 |
| 1.0 m, 5, 0.8 | 31.29 ± 0.56 | 31.54 | 9 564 | 313 | 12 / 10 / 11 |
| 1.5 m, 3, 0.6 | 31.81 ± 1.09 | 31.38 | 8 333 | 252 | 62 / 90 / 69 |

Tất cả đều nằm trong nhiễu (≤ +0.7). Ở r1, tiêu chí 1.0 m / 3 / 0.7 có 65 cặp ứng viên nhưng
ràng buộc loại trừ chặn mất 45.

### 4. Phân loại theo danh tính (hộp khớp chú thích ở IoU ≥ 0.5)

**Một người bị tách qua các camera.** Có trung bình 5 074 cặp (khung, người) mà người đó được ≥ 2 camera phát hiện.
Số Global ID đang giữ người đó:

| 1 Global ID | 2 | 3 | ≥ 4 |
|---|---|---|---|
| **9.6%** | 42.1% | 27.7% | 20.6% |

**Thành phần của mỗi điểm** (trung bình 3 lần chạy):

| loại | ý nghĩa | FP (9 828) | TP (7 157) |
|---|---|---|---|
| split | 1 người, người này đồng thời nằm ở Global ID khác trong cùng khung | **3 767 (38.3%)** | 3 554 (49.7%) |
| junk | không detection nào khớp hộp chú thích | **3 631 (37.0%)** | 489 (6.8%) |
| mixed | detection của ≥ 2 người trong cùng một Global ID | **1 503 (15.3%)** | 912 (12.7%) |
| partial | một phần không khớp ai, phần còn lại là 1 người | 590 (6.0%) | 346 (4.8%) |
| solo | 1 người, không bị tách, nhưng điểm lệch quá 1 m | 336 (3.4%) | 1 856 (25.9%) |

Tức là **~54% FP do liên kết** (split + mixed) và **~37% do detector** (junk). Chỉ ~3% là do
hình học/định vị. Lưu ý: "junk" nghĩa là không khớp hộp chú thích ở IoU 0.5, nên nó gồm cả
hộp lệch nhiều, không chỉ phát hiện ma. Và vì điểm đã lọc trong vùng, người thật ngoài vùng
(xem phiên 25) không nằm trong nhóm này.

### 5. Vì sao engine tạo Global ID mới (`appearances.reason`, r1 / r2 / r3)

| lý do | số tracklet | số khung |
|---|---|---|
| ghép / cập nhật | 454 / 412 / 433 | ~17.9 nghìn |
| **vượt `max_cost`** | **290 / 332 / 314** | **~11.6 nghìn** |
| không còn ứng viên khả thi (loại trừ, hình học) | 68 / 67 / 67 | ~4.3 nghìn |
| bị tracklet khác lấy mất / gallery rỗng | 6–9 / 9–11 | < 1 nghìn |

Chi phí của ứng viên tốt nhất ở các ca vượt ngưỡng, thập phân vị 10–90%: 0.94 / 0.97 / 0.99 /
1.01 / 1.02 / 1.04 / 1.06 / 1.09 / 1.14. 35% dưới 1.0, 84% dưới 1.1. Giống nhau ở cả ba lần chạy.

### 6. Quét `max_cost` (chạy lại engine, n = 3)

| `max_cost` | Global ID | HOTA hộp ảnh | AssA hộp | IDF1 hộp | HOTA mặt đất | HOTA mặt đất, NMS 0.5 m | AssA mặt đất | FP mặt đất |
|---|---|---|---|---|---|---|---|---|
| **0.90** | 406 ± 23 | **15.79 ± 0.47** | 10.62 | 20.27 | **31.10 ± 0.85** | **32.14 ± 0.82** | 31.64 | 9 916 |
| 1.00 | 329 ± 16 | 15.47 ± 0.25 | 10.22 | 19.30 | 30.22 ± 0.16 | 31.08 ± 0.62 | 29.78 | 9 698 |
| 1.10 | 203 ± 9 | 15.30 ± 0.50 | 9.99 | 18.79 | 28.51 ± 0.99 | 29.59 ± 0.65 | 25.76 | 9 178 |

DetA hộp ảnh đứng yên (24.05 / 23.99 / 23.99). Toàn bộ thay đổi nằm ở phần liên kết, và đi theo
chiều xấu. Cột "Global ID" ở đây là số ID tính toàn khung; cột `n_pred_ids` của giao thức mặt
đất chỉ tính trong vùng: 324 → 271 → 177.

## Vướng mắc / chưa xong

- **Ba script chẩn đoán chưa vào git** (`data/s27/scripts/gp_{fp_breakdown,merge_estimate,fp_identity}.py`,
  chỉ có trên máy dev). Nếu mục 4 và 6 vào chương 6 thì phải chuyển ít nhất `gp_fp_identity.py`
  vào `eval/` kèm test. Nếu không, số không tái lập được từ repo.
- Mới đo ở 640. Chưa chạy `gp_fp_identity` và quét `max_cost` cho 960/1280.
- `max_cost` mới quét lên trên, chưa quét xuống (0.80). Siết ngưỡng thì tách người tăng, nhưng
  gộp nhầm (15% FP) có thể giảm. Chưa đo cán cân.
- Chưa tách chi phí thành phần ngoại hình và phần hình học cho các ca vượt ngưỡng. `reason` chỉ
  ghi tổng chi phí. Chưa biết ca nào thua vì ngoại hình, ca nào thua vì hình học.
- Mọi số chỉ trên WildTrack 2 fps, rất đông người. Chưa suy ra được cho camera tự thu 25 fps.

## Bước tiếp theo

1. Tách chi phí của các ca vượt ngưỡng thành `1 − cos` và `λ · d_ground`, đối chiếu với nhãn
   người thật: cặp đúng người thua vì thành phần nào? Chỉ cần CPU (`Associator.cost_matrix` +
   bảng `.gt.json`). Nếu cặp đúng người có d_ground nhỏ mà thua vì ngoại hình, thử tăng trọng số
   hình học cho cặp chồng lấn; nếu thua vì hình học, xem lại điểm chân của hộp detector.
2. Chuyển `gp_fp_identity.py` vào `eval/` kèm test nếu mục 4 được dùng trong báo cáo.
3. Chương 6: dùng mục 4 + 6 làm bằng chứng rằng nút thắt là chất lượng đặc trưng, và rằng
   ngưỡng hiện tại đã nằm gần điểm tối ưu của đánh đổi tách người / gộp nhầm.
