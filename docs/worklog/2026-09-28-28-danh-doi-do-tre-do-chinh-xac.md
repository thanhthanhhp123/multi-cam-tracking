# 2026-09-28 (phiên 28): Đánh đổi độ trễ chốt danh tính ↔ độ chính xác. Đuôi 2–3 s của phiên 23 KHÔNG phải độ trễ chốt danh tính; núm vặn thật là `window_ms` và `min_frames`

- **Mốc:** M4/M5 (độ trễ) + M6 | **Máy:** máy dev (CPU, không thuê GPU) | **Thời lượng:** ~2.5h (trong đó ~30 phút CPU chạy nền)

## Mục tiêu phiên

- Đầu phiên: soát tiến độ để chuẩn bị báo cáo GVHD cuối tuần (03–04/10). Chưa mua được phần
  cứng và không mượn được phòng lab. Chọn "gói 1" trong ba việc làm được mà không cần thu dữ liệu.
- Gói 1: trả lời câu "<1 s" của đề cương (treo từ phiên 9) bằng một đường cong đánh đổi độ trễ
  ↔ độ chính xác, thay vì một con số.

## Đã làm

1. **Kiểm lại cách đọc đuôi p90 của phiên 23** (script tạm, không vào git) trên chính
   `data/latency.jsonl` và `data/latency-run2.jsonl`. Gom bản ghi theo `tracklet_id`: **mọi**
   bản ghi có `window_wait` > 1.5 s (128/128 ở lần 1, 116/116 ở lần 2) đều là bản ghi CUỐI của
   tracklet. Không có bản ghi đầu tiên nào nằm ở đuôi. Xem QĐ 1.
2. **`src/mct/__main__.py`**: thêm tham số `window_observer` cho `Engine`. Hook chỉ đọc, được
   gọi mỗi vòng gán với `now_ms` (thời gian dữ liệu) và kết quả gán. Không đổi logic.
3. **`eval/latency_tradeoff.py`** (mới): quét `window_ms` × `min_frames`, chạy engine online
   trong cùng tiến trình, đo `time_to_id`. Chấm điểm theo đúng đường của
   `eval.compare_oracle_tracker` (export → TrackEval). Bỏ cờ `--gt-*` thì chỉ đo độ trễ.
4. **`tests/test_latency_tradeoff.py`** (6 test): gắn observer không đổi kết quả gán; chỉ ghi
   lần gán đầu; cửa sổ ngắn hơn thì chốt sớm hơn; vòng `finish` được tách riêng; phân vị; config
   gốc không bị sửa. Chạy các file liên quan (`test_latency_tradeoff`, `test_engine_online`,
   `test_latency`, `test_no_gpu_imports`): **135 passed**. Ruff sạch trên `src tests eval`.
   Sau phần `latency_report` (mục "chưa xong"), toàn bộ bộ test: **613 passed, 5 skipped**.
5. **Kiểm tái lập**: `w1000_m3` (cấu hình WildTrack hiện tại) trên `r640n_r1` cho HOTA 15.878 /
   AssA 10.726 / 385 Global ID, trùng từng chữ số với phiên 25. Trên n = 3 cho 15.79 ± 0.47,
   cũng trùng đối chứng 640 của phiên 25.
6. **Chạy**: (a) độ trễ trên fixture tốc độ thật; (b) WildTrack r1 với lưới 3 × 4; (c) thêm
   r2, r3 cho ba điểm. Kết quả ở `data/s28/rt/tradeoff.json`, `data/s28/wt/tradeoff.json` và
   `data/s28/wt/step2_*.json`. Log ở `data/s28/wt_step{1,2}.log`.
7. `tests/fixtures/ds_4cam_reid_realtime.jsonl` vẫn cụt ở dòng 1905 (phiên 23 đã ghi). Đã cắt
   1904 dòng nguyên vẹn sang `data/s28/ds_4cam_reid_realtime_head1904.jsonl`, không sửa file gốc.

## Quyết định kỹ thuật

**1. Định nghĩa lại "độ trễ chốt danh tính" = thời gian tới vòng gán ĐẦU TIÊN của tracklet.**
Đọc code: `Associator.assign` tra `Gallery.find_by_tracklet`. Tracklet đã có chủ thì chỉ
nhận `is_update`, chủ không bao giờ đổi. Tức danh tính được chốt đúng một lần: ở vòng gán đầu
tiên sau khi tracklet đủ `min_frames`. Phiên 23 gọi đuôi 2–3 s là "độ trễ chốt danh tính" và
kết luận "KHÔNG ĐẠT nếu tính chung". **Sai**: đuôi đó là lần phát lại lúc tracklet đóng (sau
`idle_timeout_ms` + cửa sổ), khi danh tính đã được chốt từ trước. Hệ quả:
- `idle_timeout_ms` KHÔNG nằm trên đường chốt danh tính. Hai núm vặn thật là `min_frames`
  (chờ đủ khung) và `window_ms` (chờ cửa sổ đóng).
- Tính lại từ log phiên 23, chỉ lấy lần gán đầu: end-to-end (t0 → t4) **p50 107 / p90 598–622 /
  p99 877–1012 / max 1040–1045 ms**. Đây đo từ khung MỚI NHẤT của tracklet lúc gán, không từ
  khung đầu tiên. Thước đo của phiên này (`time_to_id`) thì tính từ khung đầu tiên.
- Bản ghi phát lại lúc đóng vẫn hữu ích cho dashboard (báo tracklet kết thúc), chỉ không được
  tính là độ trễ chốt danh tính. `latency_report.py` đã tách hai loại trong phiên này
  (`LatencyRecord.kind`, xem "chưa xong").

**2. `time_to_id` đo bằng thời gian dữ liệu, không bằng đồng hồ tường.** `time_to_id` = `now_ms`
của vòng gán đầu tiên − `start_ms` của tracklet, cả hai đều lấy từ `ts_ms`. Nhờ vậy engine tất
định và chạy lại trên máy dev cho cùng một số. Phần DeepStream và vận chuyển (~67 + <10 ms,
phiên 23) cộng thêm ở ngoài. Phương án bị loại: chạy lại chuyến `vast-gpu` với nhiều cấu hình
engine. Tốn tiền, mà phần engine thì tất định nên đo offline cho cùng kết quả.

**3. Tách hai phía của phép đo.** Phía ĐỘ CHÍNH XÁC chỉ đo được trên WildTrack (có ground-truth,
nhưng 2 fps). Phía ĐỘ TRỄ phải đo ở tốc độ khung thật (fixture ~31 fps, không có ground-truth).
Không chép `time_to_id` của WildTrack sang hệ thống thật: ở 2 fps, `min_frames` 5 đã là 2.5 s,
còn ở 30 fps chỉ ~130 ms.

**4. Không đổi config nào trong phiên này.** Chênh lệch giữa `w1000_m3` và `w2000_m5` nằm trong
nhiễu (t ≈ 1.4). Chênh lệch với `w500_m1` thì tách khỏi nhiễu (t ≈ 3.0), nhưng mới đo trên
WildTrack. Chốt `window_ms` / `min_frames` cho hệ thống thật ở M6, trên dữ liệu 25–30 fps.

## Số liệu đo được

### 1. Độ trễ ở tốc độ khung thật (`configs/mct.yaml`, chỉ đo độ trễ)

Fixture: 1904 message đầu của `ds_4cam_reid_realtime.jsonl` (phiên 9, RTX 3090, 4 luồng
`sample_1080p_h264`, ~31 fps, khoảng cách khung trung vị 33 ms, ReID OSNet 512-d, `sync: true`),
khoảng 15 giây/luồng, **64 tracklet**: mẫu nhỏ. `idle_timeout_ms` 2000. `time_to_id` tính bằng ms,
**chưa gồm** ~70 ms DeepStream + vận chuyển.

| `window_ms` | `min_frames` | p50 | p90 | p99 | ≤ 1 s |
|---|---|---|---|---|---|
| 250 | 1 / 5 / 10 | 166 / 300 / 331 | 265 / 532 / 532 | 266 / 699 / 565 | 100% |
| 500 | 1 / 5 / 10 | 379 / 470 / 470 | 470 / 598 / 732 | 499 / 867 / 798 | 100% |
| **1000** | 1 / **5** / 10 | 567 / **865** / 970 | 970 / **971** / 1232 | 971 / **1033** / 1298 | 100% / **93.8%** / 73.1% |
| 2000 | 1 / 5 / 10 | 1499 / 1598 / 1879 | 1970 / 1971 / 1971 | 1971 / 2032 / 2032 | 21% / 12.5% / 0% |

Cấu hình hiện tại (`w1000_m5`, in đậm): p90 971 ms, cộng ~70 ms thì **p90 ≈ 1.04 s**, sát mốc 1 s.
Với `window_ms` 500: p90 598 ms, cộng thêm thì ≈ 0.67 s.

### 2. Độ chính xác trên WildTrack (lần r1, lưới đầy đủ)

WildTrack 7 camera, 400 khung, 2 fps. Pipeline như phiên 25 (T4, DeepStream 7.1, YOLO11s COCO
FP16 640, `pre-cluster-threshold` 0.25, NvDCF + ReID `osnet_x1_0_msdc_dg`). Engine chạy
`configs/demo/wildtrack_ds.mct.yaml`, chỉ đổi `window_ms` / `min_frames` (`idle_timeout_ms`
30000). Chấm theo giao thức hộp ảnh, toàn khung, IoU 0.5, TrackEval. `time_to_id` tính bằng ms
thời gian dữ liệu.

| cấu hình | p50 | p90 | ≤ 1 s | Global ID | bỏ vì ngắn | HOTA | DetA | AssA | IDF1 |
|---|---|---|---|---|---|---|---|---|---|
| w500_m1 | 490 | 498 | 100% | 469 | 0 | 14.47 | 24.07 | 8.90 | 17.54 |
| w500_m2 | 991 | 1496 | 82.2% | 425 | 33 | 14.81 | 24.09 | 9.30 | 18.29 |
| w500_m3 | 1492 | 2995 | 15.7% | 392 | 73 | 15.31 | 23.99 | 10.00 | 19.18 |
| w500_m5 | 2495 | 6988 | 2.2% | 342 | 166 | 16.28 | 23.75 | 11.38 | 21.06 |
| w1000_m1 | 500 | 997 | 93.5% | 461 | 0 | 15.16 | 24.00 | 9.78 | 18.84 |
| w1000_m2 | 1115 | 1986 | 45.9% | 420 | 33 | 15.40 | 24.07 | 10.07 | 19.41 |
| **w1000_m3** | 1609 | 3489 | 8.9% | 385 | 73 | **15.88** | 23.99 | 10.73 | 20.51 |
| w1000_m5 | 2989 | 7000 | 2.2% | 341 | 166 | 16.29 | 23.99 | 11.28 | 21.26 |
| w2000_m1 | 1487 | 1995 | 47.1% | 430 | 0 | 15.24 | 24.11 | 9.85 | 19.36 |
| w2000_m2 | 1985 | 2498 | 21.2% | 400 | 33 | 15.60 | 23.97 | 10.37 | 20.04 |
| w2000_m3 | 2486 | 3987 | 2.4% | 380 | 73 | 16.19 | 23.97 | 11.12 | 20.72 |
| w2000_m5 | 3493 | 7986 | 0% | 336 | 166 | 16.80 | 23.98 | 12.03 | 21.60 |

### 3. Ba điểm có n = 3 (r1–r3), trung bình ± độ lệch chuẩn mẫu

| cấu hình | time_to_id p50 / p90 (ms) | HOTA | DetA | AssA | IDF1 | Global ID |
|---|---|---|---|---|---|---|
| w500_m1 (nhanh nhất) | 489 / 498 | 14.77 ± 0.37 | 24.14 | 9.25 ± 0.45 | 18.07 ± 0.55 | 490 ± 21 |
| **w1000_m3 (hiện tại)** | 1541 / 3163 | **15.79 ± 0.47** | 24.05 | 10.62 ± 0.66 | 20.27 ± 0.59 | 406 ± 23 |
| w2000_m5 (chậm nhất) | 3496 / 7989 | 16.31 ± 0.42 | 23.98 | 11.34 ± 0.60 | 20.76 ± 0.79 | 356 ± 19 |

Welch thô so với `w1000_m3`: `w500_m1` ΔHOTA −1.02, t ≈ 3.0 (tách khỏi nhiễu); `w2000_m5`
ΔHOTA +0.52, t ≈ 1.4 (**trong nhiễu**).

**Đọc bảng này thế nào:**
- **Đánh đổi có thật, và nằm hoàn toàn ở phần liên kết.** DetA đứng yên ở ~24, mọi chênh lệch
  là AssA. Chốt sớm thì embedding trung bình của tracklet dựa trên ít khung hơn, và tracklet rác
  ngắn chưa bị lọc (`min_frames` 1 không bỏ tracklet nào). Kết quả là nhiều Global ID thừa
  (490 so với 356).
- **Đồ thị có dạng "đầu gối".** Tăng tốc từ `w1000_m3` xuống `w500_m1` mất ~1 HOTA. Chậm hơn nữa
  (`w2000_m5`) chỉ được ~0.5 HOTA, và mức đó nằm trong nhiễu. Cấu hình hiện tại ở gần đầu gối.
- **`min_frames` tác động mạnh hơn `window_ms`** (so từng cột trong bảng 2), nhưng đó là tính
  theo khung. Ở 2 fps mỗi khung đáng 500 ms. Ở 30 fps, `min_frames` 5 chỉ tốn ~130 ms, nên phần
  độ chính xác mà `min_frames` mang lại gần như miễn phí về độ trễ trên hệ thống thật.
- **Câu trả lời cho "<1 s" (đề xuất để chốt với GVHD):** độ trễ vị trí trung vị 106 ms (phiên 23)
  → ĐẠT. Độ trễ chốt danh tính trên fixture ~31 fps với cấu hình hiện tại có p90 ≈ 1.04 s, nên
  đạt ở trung vị và **sát ngưỡng ở p90**. Hạ `window_ms` xuống 500 cho p90 ≈ 0.67 s, nhưng giá về
  độ chính xác ở 25–30 fps **chưa đo được** (WildTrack 2 fps không trả lời được câu này).

## Vướng mắc / chưa xong

- **Mẫu độ trễ tốc độ thật rất nhỏ**: 64 tracklet, ~15 giây, 4 bản sao cùng một video mẫu. Cần
  đo lại trên dữ liệu tự thu 25–30 fps (M6), ở đó mới có cả độ trễ lẫn độ chính xác trên cùng
  một nguồn.
- ~~`tools/latency_report.py` chưa tách "lần gán đầu" với "phát lại lúc đóng"~~ **xong trong
  phiên**: `LatencyRecord.kind` = `first` / `update` / `close` do engine ghi; log cũ được suy ra
  `first` theo thứ tự `tracklet_id` (còn lại là `repeat`); `--by kind`; mục tiêu 1 s giờ chấm
  trên bản ghi `first`. Trên log phiên 23 (lần 2): `first` n = 128, p50 106.6 / p90 621.5 /
  p99 996.2 ms → ĐẠT; `repeat` n = 891, p90 2405 ms. +4 test. Toàn bộ bộ test: **613 passed,
  5 skipped**, ruff sạch.
- Chưa chấm theo giao thức điểm mặt đất cho các cấu hình này (mới có hộp ảnh).
- `idle_timeout_ms` không ảnh hưởng độ trễ chốt danh tính, nhưng vẫn ảnh hưởng độ vỡ tracklet
  (phiên 12). Không nằm trong lưới quét này.
- ~~README sai số~~ **đã sửa trong phiên**: bảng kết quả dùng số n = 3 (A/C/B/LB của phiên 24)
  thay cho số một lần chạy trên fixture đã mất; bỏ hàng "Upper bound" ghép nhầm (HOTA/IDF1 của
  cận trên nhưng AssA/DetA của đơn camera pipeline thật, phiên 15); 189 FPS/luồng là số KHÔNG
  ReID (có ReID là 175, phiên 9); độ trễ tách vị trí / chốt danh tính; thêm HOTA mặt đất;
  các số "cận trên (SCT lý tưởng)" được ghi rõ nhãn.
- Worklog phiên 23 (QĐ/bước tiếp theo 1) vẫn ghi đuôi là "độ trễ chốt danh tính". Đã thêm ghi
  chú đính chính ở đầu file đó, nội dung gốc giữ nguyên.

## Bước tiếp theo

1. Báo cáo GVHD (03–04/10): mang bảng 1 + bảng 3 và đề xuất định nghĩa ở mục "Đọc bảng này thế
   nào". Xin thầy chốt: mục tiêu "<1 s" áp vào độ trễ vị trí, hay chốt danh tính ở p50 / p90?
2. ~~Vẽ biểu đồ đánh đổi~~ **xong**: `data/s28/latency_tradeoff.png` (script
   `data/s28/plot_tradeoff.py`, chạy bằng `uv run --no-project --with matplotlib`; cả hai đều
   gitignored, chỉ có trên máy dev). Trục x p90 `time_to_id` (log, WildTrack 2 fps), trục y
   HOTA hộp ảnh; ba điểm n = 3 có thanh sai số, 12 điểm r1 làm nền.
3. M6: đo lại `time_to_id` + HOTA trên cùng một nguồn 25–30 fps khi có dữ liệu tự thu.
