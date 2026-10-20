# 2026-10-20 (phiên 37): Quay dữ liệu M6 ở hành lang: 14 buổi, dừng ghi từ xa, laptop tự ngủ làm hỏng buổi

- **Mốc:** M6 (thu dữ liệu) | **Máy:** laptop LOQ (Windows 11), 3 × Tiandy TC-C131N | **Thời lượng:** ~3.5h (12:30–16:05)

## Mục tiêu phiên

- Quay `calib` và các kịch bản `s1`–`s4` theo `docs/m6/checklist-quay-hanh-lang.md`, ngay sau phiên 36
  (cùng ngày, camera không chạm vào).
- Người dùng điều khiển từ xa qua Remote Control, không ngồi ở laptop: mọi lần ghi do Claude chạy.

## Đã làm

1. **Kiểm trước buổi quay:**
   - `record_rtsp check` 0 FAIL.
   - Khung trung vị từ RTSP so với `report_media/2026-10-20/lan_e/*_median.png` (phase correlation):
     cả ba camera lệch ≤ 0.2 px → homography phiên 36 dùng được.
   - Không có OSD. `.env` thêm `LAB_FFMPEG` (gyan.dev 9.0.2 ở `~/tools/ffmpeg`).
2. **Kịch bản chi tiết in ra giấy** (`data/lab/kich-ban-quay.pdf`, gitignored): bản đồ mốc, quy
   tắc, bảng giờ cho `calib`/`s1`–`s4`, phiếu ghi chép. Sinh từ HTML bằng Edge headless
   (`--print-to-pdf`); nguồn HTML để ở scratchpad, không vào repo.
3. **`record_rtsp`: dừng sạch từ xa bằng file `STOP`** trong thư mục buổi (`watch_stop_file`,
   kiểm mỗi 0.5 s, tự xoá file; xoá file cũ trước khi ghi). Test `test_file_stop_dung_buoi_ghi_va_tu_xoa`.
   Trước đó buổi `calib` đầu phải chờ hết `--duration`, vì buộc tắt tiến trình thì mất `record.json`.
4. **Quay 14 buổi** (đều `"ok": true`, không cảnh báo), ghi chú vai từng buổi ở
   `data/lab/raw/ghi-chu-buoi-quay.md`. Người dùng xoá khoảng 10 lần quay hỏng (sai vai P1/P2,
   người qua đường, diễn sai).
5. **Hai buổi hỏng vì laptop vào Modern Standby** (xem Vướng mắc), sửa bằng `powercfg`.
6. **Kiểm camera có xê dịch trong từng buổi** (đầu và cuối mỗi file `.mkv`, so với `lan_e`): mọi
   buổi ≤ 1.6 px. Cuối phiên cam02 lệch 26 px và cam03 mất kết nối, nhưng xảy ra SAU buổi cuối
   (lúc tháo dọn).

## Quyết định kỹ thuật

### 1. Chia kịch bản thành buổi ngắn (~90 s), mỗi lượt một buổi

- `s2` dự kiến 3 phút với 3 lượt; thực tế người dùng quay từng lượt (`s2_1`, `s2_1b`, `s2_1c`).
- Lý do: hành lang công cộng, người qua đường làm hỏng cả buổi dài; buổi 90 s hỏng thì chỉ quay lại
  90 s. Đổi lại phải gộp các buổi cùng một lần quay khi chấm (cộng dồn số bàn giao).
- *Loại: ghi liền một mạch cả buổi rồi cắt sau* (người dùng hỏi). Lỗi chỉ lộ khi dừng
  (`BUỔI NÀY HỎNG`), nên ghi 2 giờ liền có thể mất cả buổi. Thêm nữa, đoạn nghỉ cũng bị ghi.

### 2. Dừng từ xa bằng file, không bằng tín hiệu

- Trên Windows không gửi Ctrl+C tới một tiến trình khác console được, còn buộc tắt thì mất
  `record.json` — chỗ duy nhất giữ độ lệch đồng hồ giữa các camera.
- File `STOP` không cần thêm cổng mạng hay thư viện, hợp với ràng buộc "chỉ thư viện chuẩn" của công cụ.

### 3. Vùng xa của cam01 không cần xử lý thêm khi quay

- Người dùng hỏi lại chuyện cam01 thấy vùng cam03. `ground_roi` của phiên 36 đã lọc ở engine và
  GT, nên lúc quay chỉ cần không đứng lại ở x ≈ 5–7 m.

## Số liệu đo được

Cấu hình: 3 × Tiandy TC-C131N, H.264 1920×1080 25 fps, RTSP/TCP, LAN link-local 169.254.249.x,
laptop LOQ, `record_rtsp` (`-c copy` ra MKV).

| Buổi | Nội dung | Dài (s) | Lệch cam02 / cam03 (ms) |
|---|---|---|---|
| `calib` | calib chậm ×2 | 479.8 | +3 / +69 |
| `calib_2` | calib thường ×2, nhanh ×2 | 182.8 | −34 / −52 |
| `s1_1` | chồng lấn, 2 người | 269.8 | −25 / −15 |
| `s2_1`, `b`, `c` | s2 lần 1, lượt 1/2/3 | 88.2 / 77.6 / 89.4 | |
| `s2_2`, `b`, `c` | s2 lần 2, lượt 1/2/3 | 81.1 / 94.4 / 93.3 | |
| `s2_3`, `b` | s2 lần 3 (đổi thứ tự người), cả hai khuất, lượt 1/2 | 80.1 / 92.9 | |
| `s3_1` | đông người | 105.7 | 0 / +45 |
| `s3_2` | đông người, 5 người | 60.1 | −24 / 0 |
| `s4_1` | áo cùng màu | 147.6 | +4 / +16 |

- Tổng **~32 phút**, 2.9 GB. Mọi buổi 25.00 fps, jitter 18–21 ms, độ lệch giữa camera < 80 ms.
- Trôi đồng hồ thường 3–46 ppm. Có hai buổi lên **~170–190 ppm**: `s2_3` (ngay sau lần laptop ngủ)
  và `s3_2` (60 s, ngắn nên ước lượng nhiễu). Sai số do trôi vẫn ≤ 7 ms.
- Camera xê dịch trong lúc quay (phase correlation so với `lan_e`):
  - cam01 ≤ 0.4 px;
  - cam02 ≤ 0.3 px cho tới `s4_1` (14:18), rồi **+1.5 px theo dọc từ `s2_3` (15:45)**. Có thể ai đó
    chạm nhẹ trong khoảng 14:20–15:28; quy ra khoảng 1–2 cm, không cần hiệu chỉnh lại;
  - cam03 trôi dần từ 0.1 tới 1.0 px.
- Ước lượng số bàn giao (theo kịch bản, chưa đếm từ GT): không chồng lấn ~32, quay lại cùng camera
  ~12, chồng lấn ~20. **Dưới mục tiêu ~50/loại** của checklist (khoảng tin cậy 95% ±14% ở n ≈ 50).
- Test: `tests/test_record_rtsp.py` 30 passed (venv `mct-lab`, **Python 3.12**), ruff sạch.

## Vướng mắc / chưa xong

- **Laptop tự ngủ (Modern Standby) giữa buổi ghi**: tắt màn hình sau 3 phút không ai chạm = ngủ.
  Khi ngủ, cả Python lẫn ffmpeg đóng băng (log tiến độ nhảy từ 00:30 sang 02:33), rồi ffmpeg thoát
  vì mất luồng. Đã dính hai lần (15:41–15:42, 15:47–15:50, khớp `Kernel-Power` 506/507).
  - Ban đầu đoán nhầm là dây mạng, vì cả ba camera mất cùng lúc.
  - Sửa: `powercfg /change monitor-timeout-ac 0` và `standby-timeout-ac 0` (chỉ khi cắm sạc). Gập
    màn hình vẫn làm ngủ.
  - **Trả lại sau M6:** `powercfg /change monitor-timeout-ac 3`, `powercfg /change standby-timeout-ac 3`.
  - Nên thêm vào `record_rtsp check` một dòng cảnh báo khi máy sẽ ngủ trong lúc ghi; và/hoặc gọi
    `SetThreadExecutionState` trong lúc ghi (chưa làm; với Modern Standby cách này có thể không đủ).
- **Thiếu so với kế hoạch:**
  - `s2_3c` (lượt 3), `s1_2`, `s1_3`, `s4_2`, `s4_3`, `s3_3`, cả nhóm B.
  - Chỉ có 2 người cố định, cùng một bộ đồ. Có 5 người ở `s3_2`, nhưng ba người kia chưa quay
    riêng làm nhóm B.
- **Ghi chú chưa đủ:**
  - vai P1/P2 của `s2_1` lượt 1;
  - nội dung `s4_1` (có đổi chỗ không, ai tới vùng cam03);
  - số người ở `s3_1`.
- **Camera đã bị tháo/chạm sau buổi cuối** (cam02 lệch 26 px, cam03 mất kết nối). Muốn quay bổ
  sung phải hiệu chỉnh lại theo quy trình cái bát (phiên 36, quyết định 4), và dữ liệu mới sẽ có
  homography khác dữ liệu hôm nay.
- Thay đổi `record_rtsp` chưa commit; chưa chạy bằng Python 3.10.
- `.env` (mật khẩu camera), venv, `data/lab/raw/` (2.9 GB) chỉ nằm trên laptop LOQ — chưa chép ra
  ổ ngoài.

## Bước tiếp theo

1. **Chép `data/lab/raw/` và `data/lab/report_media/` ra ổ ngoài**, so dung lượng hai bên.
2. Bổ sung ghi chú còn thiếu (mục trên) vào `data/lab/raw/ghi-chu-buoi-quay.md` khi người dùng còn nhớ.
3. Đồng bộ + chuẩn hoá từng buổi bằng lệnh `sync_recordings` in sẵn trong `record.json`, rồi chạy
   `estimate_transit` trên `calib` + `calib_2`. Kiểm `estimate_transit` có áp `ground_roi` của
   cam01 không: nếu không, khoảng cam01 → cam03 bị đo từ chỗ cam01 thấy người ở tận vùng cam03.
4. Chú thích CVAT theo thứ tự `calib` → `s2_*` → `s1_1` → `s4_1` → `s3_*`. Cần thêm bước gộp
   `s2_Nb/c` vào `s2_N` khi chấm.
5. Quyết định với người dùng: có quay bổ sung (nhóm B, `s3_3`, thêm `s2` cả 3 lượt khuất) hay chấp
   nhận n bàn giao hiện có và ghi rõ khoảng tin cậy trong chương 6.
