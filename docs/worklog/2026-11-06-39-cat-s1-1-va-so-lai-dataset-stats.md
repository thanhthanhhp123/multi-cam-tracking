# 2026-11-06 (phiên 39): Cắt s1_1 ở khung 477, cài ffmpeg, đếm lại bàn giao

- **Mốc:** M6 (chú thích) | **Máy:** máy dev (Windows, user `admin`) | **Thời lượng:** ~1h

## Mục tiêu phiên

- Nạp lại ngữ cảnh sau phiên 38 và bật công cụ kiểm nhãn `tools.gt_review`.
- Người dùng kiểm tay s1_1 và thấy từ khung 477 trở đi nhãn không tin cậy được. Cắt buổi này ở khung 477.

## Đã làm

- **`gt_review` đã chạy sẵn** ở `127.0.0.1:8770` (PID 11528). Máy này không có `~/.venvs/mct-test`, nên tôi không tự khởi động lại.
- **Cắt GT nháp của s1_1 ở khung task 477** (= khung video 2385 = 95.4 s), ở `data/prelabel/export_draft/s1_1/cam0{1,2,3}.xml`:
  - bỏ mọi hộp có `frame > 2385`; track hết hộp thì xoá (cam01 giữ 4 / xoá 11, cam02 giữ 9 / xoá 9, cam03 giữ 3 / xoá 2);
  - sửa `<size>` = 478, `<stop_frame>` = 2385, `segment/stop` = 477;
  - `gt_edit.validate` sạch cả ba camera;
  - bản gốc ở `data/prelabel/export_draft_full/s1_1/`.
- **Cài ffmpeg 9.0.2** bằng `winget install Gyan.FFmpeg`. `LAB_FFMPEG` trong `.env` trỏ tới bản mới, vì đường dẫn cũ (`C:/Users/thanh/...`) không có trên máy `admin`.
- **Cắt video** `data/lab/s1_1/cam0X.mp4` bằng `-c copy -t 95.44`. ffprobe đo lại: 2386 khung, 25 fps, 95.44 s, cả ba camera. Bản đầy đủ ở `data/lab_backup/s1_1_full/`. `sync.json`: `duration_s` 95.44, `expected_frames` 2386, `trim_end_s` 174.3753, thêm `note`.
- **Chạy lại `data/prelabel/_scripts/dataset_stats.py`** trong venv mới `~/venvs/mct-stats` (Python 3.12, numpy/scipy/msgpack/PyYAML/python-dotenv). Kết quả ở `data/prelabel/dataset_stats.json`; bản cũ ở `data/prelabel/dataset_stats_before_cut_s1_1.json`.

## Quyết định kỹ thuật

- **Cắt buổi chứ không chỉ xoá hộp.** Pipeline vẫn chạy trên toàn bộ video. Nếu chỉ xoá nhãn sau 477, mọi người hệ thống thấy sau đó thành báo nhầm và điểm s1_1 bị kéo xuống oan. Nên GT, `stop_frame` và video phải cắt cùng một mốc.
- **Cắt video bằng `-c copy`, không đồng bộ lại.** Chỉ cắt đuôi nên chỉ số khung phía trước không đổi. Tránh được ~10 phút mã hoá lại và việc mã hoá lại hai lần. `-t 95.44` giữ đúng khung 0–2385. Video thô ở `data/lab/raw/s1_1/` vẫn nguyên.
- **Giữ mã P04 cho người đi một mình ở cam03 trong s1_1.** Người ngoài kịch bản vẫn phải có hộp (quy ước 2), nhưng nếu không chắc là cùng một người thì đặt mã X, không đặt mã tạm kiểu "P04?".
- **Lý do cắt phải ghi vào báo cáo:** phần sau khung 477 bị loại vì không gán nhãn tin cậy được (người qua đường, che khuất, hộp ở cuối hành lang), không phải vì hệ thống làm sai.

## Số liệu đo được

Cấu hình: GT nháp (chưa người kiểm), 12 buổi chấm (không tính `calib`, `calib_2`), sau `ground_roi` của cam01, định nghĩa bàn giao như `eval_handover`, track trống mã tính là người riêng.

| Loại bàn giao | Trước cắt | Sau cắt | Đến cùng camera trong ±3 s (trước → sau) | Wilson 95% ở p = 0.8 (sau) |
|---|---|---|---|---|
| Chồng lấn | 102 | 92 | 58 → 54 | ±8 điểm |
| Không chồng lấn | 48 | 45 | 7 → 5 | ±11 điểm |
| Quay lại cùng camera | 39 | 30 | 15 → 12 | ±14 điểm |

- Tổng thời lượng 12 buổi chấm: 21.3 → 18.4 phút.
- Riêng s1_1 (trước → sau): thời lượng 269.8 → 95.6 s (lưới chú thích), người 8 → 5, bàn giao chồng lấn 18 → 8, quay lại cùng camera 12 → 3, không chồng lấn 3 → 0.

## Vướng mắc / chưa xong

- **Toàn bộ GT vẫn là bản nháp.** Chưa buổi nào được người kiểm xong.
- `s2_2c` và `s3_2` còn track trống mã (mã U), được đếm như người riêng nên số bàn giao của chúng là cận dưới.
- Quay lại cùng camera chỉ còn n = 30 (±14 điểm), không chồng lấn n = 45. Phải ghi rõ trong chương 6. Quay bổ sung phải hiệu chỉnh lại camera (cam02/cam03 đã bị chạm sau buổi cuối, phiên 37).
- Server `gt_review` có thể còn nhớ s1_1 cũ (1348 khung); thư mục ảnh `data/gt_review/frames/s1_1/` còn 1349 ảnh, phần sau 477 không dùng tới. Nếu refresh trang mà vẫn hiện 1348 thì phải khởi động lại server. Không có venv đúng trên máy này.
- `git` báo "dubious ownership" nên chưa xem được commit hay trạng thái working tree. Cần `git config --global --add safe.directory E:/Projects/multi-cam-tracking`.
- Chưa commit: thay đổi `record_rtsp` từ phiên 37.
- `make lab-gt` cho s1_1 chưa chạy, vì buổi này còn đang chờ kiểm nhãn.

## Bước tiếp theo

1. Refresh `gt_review`, kiểm nhãn s1_1 trong 477 khung (chú ý hộp P01/P02 chồng nhau ở cam01 khi hai người đi ngang nhau), rồi các buổi còn lại theo `data/prelabel/REVIEW.md` (calib → s2 → s1 → s4 → s3).
2. Điền 4 track trống ở `s2_2c`, `s3_2`.
3. Khi tab "Lỗi" trống: `make lab-gt LAB_SESSION=<buổi>`, `make lab-transit LAB_SESSION=calib`, điền `configs/lab/topology.yaml`, `make lab-check`.
4. Quyết định có quay bổ sung không (số bàn giao không chồng lấn và quay lại cùng camera đang mỏng).
