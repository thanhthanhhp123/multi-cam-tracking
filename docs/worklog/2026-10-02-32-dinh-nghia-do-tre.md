# 2026-10-02 (phiên 32): Chốt định nghĩa độ trễ "< 1 s". Theo định nghĩa này, hệ hiện tại KHÔNG ĐẠT, dù chỉ thiếu một chút

- **Mốc:** M5 (độ trễ) + chuẩn bị báo cáo GVHD | **Máy:** máy dev (CPU, không thuê GPU) | **Thời lượng:** ~1.5h

## Mục tiêu phiên

- Người dùng giao: **tự chốt định nghĩa độ trễ** cho mục tiêu "< 1 s" của đề cương, câu hỏi treo
  từ phiên 9, rồi giải thích các vấn đề liên quan. Trước đây định chờ GVHD chốt (phiên 28).

## Đã làm

1. Đọc lại câu chữ của đề cương (`docs/DoAn_MultiCameraTracking_DeepStream.docx`):
   - đoạn 96: "Xử lý thời gian thực với độ trễ end-to-end ở mức chấp nhận được cho giám sát an
     ninh (mục tiêu dưới 1 giây)";
   - đoạn 199: chỉ số "độ trễ end-to-end".
   Đề cương không nói đo từ đâu tới đâu, cũng không nói đo ở phân vị nào.
2. **Phát hiện chính.** Dashboard chỉ đọc `mct:global` (`src/dashboard/live.py`). Engine chỉ phát
   lên đó khi một cửa sổ gán đóng (`Engine.feed` → `_run_window`). Vì vậy vị trí trên màn hình
   được làm mới mỗi `window_ms` một lần (1 Hz với cấu hình mặc định), kể cả với người đã có
   Global ID từ lâu.
   - Mỗi bản ghi độ trễ chỉ mang mốc của khung **mới nhất** lúc phát.
   - Nên "106 ms trung vị" của phiên 23 chỉ đo **đỉnh tốt nhất của hình răng cưa**. Một khung tới
     đầu cửa sổ phải chờ gần trọn 1 s mới lên dashboard.
   - Kiểm trên log phiên 23 (script tạm, sau đó đưa vào công cụ): nhịp phát giữa hai lần của cùng
     một tracklet có trung vị 1000 ms, p99 1018 ms. Độ cũ của vị trí đang hiện, tính trung bình
     theo thời gian, khoảng 607 ms; ngay trước lần phát kế tiếp thì lên 1.10 s.
3. `src/tools/latency_report.py`:
   - `frame_latencies()` dựng lại độ trễ theo khung từ các cặp lần phát liên tiếp của từng
     tracklet.
   - `time_to_id()` tính thời gian tới Global ID từ mốc mới `t0_first`.
   - `target_verdict()` chấm mục tiêu theo định nghĩa mới. Bỏ dòng "độ trễ CHỐT DANH TÍNH" cũ:
     dòng đó đo khung mới nhất của bản ghi `first`, không phải thời gian tới Global ID.
4. `src/common/latency.py` thêm `T0_FIRST`. `src/mct/tracklet.py` thêm `Tracklet.first_stamps`.
   `src/mct/__main__.py` gắn `t0_first` vào bản ghi `kind = first`. Không đổi logic gán, không đổi
   schema message.
5. `eval/latency_tradeoff.py`: thêm p95 vào `PERCENTILES`. Chạy lại `w250/500/750/1000 × m5` trên
   fixture tốc độ thật của phiên 28 (`data/s28/ds_4cam_reid_realtime_head1904.jsonl`). Kết quả ở
   `data/s32/rt/tradeoff.json`: p50/p90/p99 **trùng từng số** với phiên 28.
6. Test (`tests/test_latency.py`): viết lại 1 test chấm mục tiêu, thêm 8 test (răng cưa; nhịp phát
   ngắn thì đạt; `close`/`final_flush` không tính; log cũ bỏ `repeat` cuối; quãng vắng không bị
   tính thành trễ; thời gian tới Global ID; log thiếu `t0_first` thì cảnh báo; engine gắn
   `t0_first`). Đã chạy các file liên quan, ruff sạch (xem "Số liệu").
7. Cập nhật README (bảng hiệu năng và lời giải thích) và CLAUDE.md §7 (một mục định nghĩa).

Lệnh tái lập (Git Bash, máy dev):
```
PYTHONPATH=src ~/.venvs/mct-test/Scripts/python.exe -m tools.latency_report --log data/latency-run2.jsonl
PYTHONPATH=src ~/.venvs/mct-test/Scripts/python.exe -m eval.latency_tradeoff \
    --run reid data/s28/ds_4cam_reid_realtime_head1904.jsonl --config configs/mct.yaml \
    --window-ms 250 500 750 1000 --min-frames 5 --work-dir data/s32/rt
```

## Quyết định kỹ thuật

### 1. Định nghĩa: hai đại lượng, cùng một ranh giới đo, chấm ở p95

Mục tiêu "< 1 s" của đề cương ĐẠT khi và chỉ khi **p95 của cả hai đại lượng dưới đây < 1000 ms**:

| Đại lượng | Một mẫu là | Bắt đầu | Kết thúc |
|---|---|---|---|
| **(A) Độ trễ theo khung** | một khung của một người đã có Global ID | `t0`: khung tới DeepStream (`ntp_timestamp` của streammux) | lúc kết quả phản ánh khung đó lên `mct:global` (`t4` của lần phát kế tiếp) |
| **(B) Thời gian tới Global ID** | một tracklet (một người xuất hiện ở một camera) | `t0` của khung ĐẦU TIÊN của tracklet | lần đầu tracklet lên `mct:global` kèm Global ID |

**Điều kiện đo:**
- nguồn chạy đúng tốc độ thật (RTSP, hoặc file với `sync: true`), 4 luồng;
- pipeline giữ ≥ 15 FPS/luồng suốt lần đo;
- engine cùng máy với pipeline, hoặc lệch NTP < 10 ms (cột `âm` của báo cáo phải bằng 0);
- lần đo đủ dài để p95 của (B) có nghĩa: ≥ 200 tracklet, khoảng ≥ 10 phút.

**Không tính:**
- bản ghi `close`: lần phát lại lúc tracklet đóng, mang khung cũ đã hiện từ trước;
- bản ghi `final_flush`: vòng gán lúc hết nguồn, chỉ có khi phát lại fixture;
- tracklet ngắn hơn `min_frames`: không bao giờ được hiện. Đó là chuyện độ chính xác (bỏ sót),
  không phải độ trễ; đếm và báo riêng.

**Nằm ngoài ranh giới, báo riêng ở M6:**
- phần trước `t0`: camera mã hoá, mạng, jitter buffer RTSP (`latency = 200` ms trong
  `src/ds_pipeline/builder.py`), giải mã;
- phần sau `t4`: WebSocket và trình duyệt vẽ.
Đo bằng một lần quay glass-to-glass khi có camera thật.

### 2. Vì sao chọn như vậy, và các phương án bị loại

- **Điểm cuối là `mct:global`, không phải lúc gán xong (`t3`).** `mct:global` là thứ người vận
  hành thấy, và là nguồn duy nhất của dashboard.
- **Hai đại lượng, vì người vận hành hỏi hai câu khác nhau:** "người X đang ở đâu" (A) và "người mới
  vào camera này là ai" (B). (B) là câu đặc trưng của MTMCT: đó là độ trễ bàn giao danh tính khi
  người đi sang camera khác.
  - *Loại: gộp làm một phân bố theo khung.* Khung khởi đầu chỉ chiếm vài phần trăm tổng số khung
    (tracklet dài cả chục giây), nên vế danh tính bị pha loãng, và che đúng chỗ MTMCT khó.
- **Chấm theo khung, không theo bản ghi.**
  - *Loại: cách cũ, chỉ lấy khung mới nhất của mỗi bản ghi.* Cách này đo đỉnh tốt nhất của răng
    cưa: 106 ms so với p50 thật khoảng 615 ms.
  - Độ cũ trên màn hình tính theo thời gian (age of information) cho cùng hình răng cưa. Chọn cách
    "theo khung" vì khớp chữ "end-to-end" của đề cương.
- **p95.** Các phiên trước dùng p90. Từ phiên này đổi sang p95.
  - *Loại p50:* che hết đuôi, mà giám sát an ninh quan tâm đúng phần đuôi.
  - *Loại max:* bị chi phối bởi sự kiện hiếm (khởi động, hết nguồn).
  - *Loại p99:* với (B), vài trăm tracklet thì p99 chỉ còn vài mẫu.
  - p95 là mức SLO thông dụng.
- **Ranh giới bắt đầu ở `t0` (khung tới DeepStream), không phải glass-to-glass.**
  - Đây cũng là cách DeepStream tự đo độ trễ: tính từ lúc khung vào pipeline.
  - Phần camera và mạng phụ thuộc phần cứng chưa mua (CLAUDE.md §11: điện thoại phát RTSP trễ
    và giật hơn camera IP), nên không tái lập được trên máy thuê.
  - *Loại: glass-to-glass làm số chính.* Vẫn đo nó ở M6 và đặt cạnh, để người đọc biết ngân sách
    thật còn bao nhiêu.

### 3. Hệ quả: với cấu hình mặc định, mục tiêu KHÔNG ĐẠT, và thiếu chỉ một chút

- (A) p95 = 1.05 s (log phiên 23).
- (B) p95 ≈ 1.03 s thời gian dữ liệu, cộng khoảng 75 ms pipeline thì ≈ 1.11 s.
- Câu "met for position" trong README là kết luận từ thước đo sai, đã sửa.
- Đây không phải chuyện tính toán chậm. Phần không phải chờ cửa sổ chỉ khoảng 75–80 ms, kể cả ở
  p99 (bảng 1). Toàn bộ phần còn lại là chờ cửa sổ, tức một **tham số thiết kế**.

### 4. Hướng sửa (chưa làm, chờ người dùng/GVHD)

1. **Tách đường phát vị trí khỏi cửa sổ gán** để sửa (A).
   - Tracklet đã có chủ thì không bao giờ đổi chủ (phiên 28, QĐ 1). Vậy có thể phát vị trí của nó
     ngay khi có khung mới, có giới hạn tần số, ví dụ 5–10 Hz/tracklet, mà không cần chờ vòng gán.
   - Kết quả gán và SQLite không đổi, nên **độ chính xác không đổi theo cấu trúc**.
   - Ước lượng (chưa đo): (A) còn khoảng 80 ms.
   - Cái giá là nhiều message hơn trên `mct:global`. Margin thông lượng của engine chỉ còn khoảng
     1.5 lần (phiên 19), nên phải đo lại.
2. **Chọn `window_ms` cho (B) theo quy tắc ràng buộc:** ở M6, trên dữ liệu 25–30 fps, chọn cấu hình
   có HOTA cao nhất **trong số** các cấu hình đạt (B) p95 < 1 s.
   - Ứng viên: `window_ms` 500, (B) p95 ≈ 0.94 s.
   - Bằng chứng duy nhất hiện có về giá độ chính xác: WildTrack r1 với `min_frames` 5 cho `w500`
     16.28 so với `w1000` 16.29. Chỉ là một lần chạy, ở 2 fps.
   - Không đổi `configs/mct.yaml` trong phiên này, theo phiên 28 QĐ 4.

## Số liệu đo được

### 1. Độ trễ theo khung, log phiên 23

T4, 4 × `sample_1080p_h264`, có ReID, `sync=true`, engine cùng máy, `configs/mct.yaml`
(`window_ms` 1000, `min_frames` 5, `idle_timeout_ms` 2000). 128 tracklet. Tính từ khoảng 762 cặp
lần phát, lấy mẫu mỗi 10 ms.

| Log | Mẫu | p50 | p90 | p95 | p99 | Chấm |
|---|---|---|---|---|---|---|
| `data/latency.jsonl` | 71 926 | 617 | 1 006 | **1 055** | 1 099 | KHÔNG ĐẠT |
| `data/latency-run2.jsonl` | 72 067 | 614 | 1 002 | **1 047** | 1 097 | KHÔNG ĐẠT |

Các đoạn ngoài cửa sổ (run2, p50 / p99, ms):
- DeepStream 67.4 / 69.4;
- đẩy Redis 0.1 / 1.1;
- engine nhận 0.8 / 2.0;
- gán 3.0 / 5.1;
- ghi DB + phát 1.5 / 2.9.

Tổng ≈ 73 / 80 ms. `window_wait` chiếm phần còn lại.

### 2. Thời gian tới Global ID trên fixture tốc độ thật (thời gian dữ liệu)

Fixture: 1904 message đầu của `ds_4cam_reid_realtime.jsonl` (phiên 9, RTX 3090, 4 luồng,
khoảng 31 fps), `configs/mct.yaml`, `min_frames` 5. **64 tracklet**, nên p95 xấp xỉ mẫu lớn thứ tư:
nhiễu. Cột cuối cộng khoảng 75 ms pipeline (bảng 1).

| `window_ms` | p50 | p90 | p95 | p99 | p95 + 75 ms | Chấm |
|---|---|---|---|---|---|---|
| 250 | 300 | 532 | 697 | 699 | ≈ 0.77 s | ĐẠT |
| 500 | 470 | 598 | 865 | 867 | ≈ 0.94 s | ĐẠT (sát) |
| 750 | 705 | 866 | 964 | 966 | ≈ 1.04 s | KHÔNG ĐẠT |
| **1000** (hiện tại) | 865 | 971 | 1 032 | 1 033 | **≈ 1.11 s** | KHÔNG ĐẠT |

### 3. Kiểm test

`pytest tests/test_latency.py tests/test_latency_tradeoff.py tests/test_engine_online.py
tests/test_no_gpu_imports.py tests/test_tracklet.py tests/test_associator.py`: **198 passed**. Ruff check và format sạch trên
`src tests eval`.

## Vướng mắc / chưa xong

- **(B) chưa từng đo bằng đồng hồ tường.** Hiện chỉ có số theo thời gian dữ liệu cộng một hằng số.
  Engine mới đã ghi `t0_first`, nhưng cần một lượt chạy GPU để có log.
- **Mẫu nhỏ:** 64–128 tracklet, khoảng 15 s, 4 bản sao của cùng một video mẫu. Định nghĩa đòi
  ≥ 200 tracklet và khoảng 10 phút, nên phải đo lại ở M6.
- Đường phát vị trí tách khỏi cửa sổ (QĐ 4.1) mới là đề xuất. Con số khoảng 80 ms là ước lượng
  từ các đoạn ngoài cửa sổ, chưa đo.
- Glass-to-glass (camera → màn hình) chưa đo. Cần camera thật (đang chờ phần cứng).
- Phiên 28 báo "giá ≈ 1 HOTA khi hạ `window_ms` 1000 → 500". Số đó so `w1000_m3` với `w500_m1`,
  tức đổi cả `min_frames`. Giữ nguyên `min_frames` 5 thì một lần chạy r1 cho chênh lệch gần 0.
  Cần n = 3 trước khi dùng.

## Bước tiếp theo

1. Báo cáo GVHD (03–04/10): trình định nghĩa ở QĐ 1, kết quả "chưa đạt, thiếu khoảng 5–10%"
   ở QĐ 3, và hai hướng sửa ở QĐ 4. Hỏi thầy có chấp nhận ranh giới `t0` không, hay muốn
   glass-to-glass làm số chính.
2. Nếu làm tiếp: cài đường phát vị trí tách khỏi cửa sổ trong `src/mct/__main__.py` (có giới hạn
   tần số, có test "kết quả gán không đổi"), rồi đo lại (A) bằng phát lại fixture.
3. Chạy n = 3 trên WildTrack cho `w500_m5` so với `w1000_m5`, để biết giá độ chính xác khi chỉ
   đổi cửa sổ.
