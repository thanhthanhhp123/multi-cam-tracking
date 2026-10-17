# 2026-10-17 (phiên 35): Dựng hiện trường M6 ở hành lang — camera sang H.264 1080p, tắt OSD, hiệu chỉnh homography 3 camera bằng đường ron gạch

- **Mốc:** M6 (hiện trường) | **Máy:** laptop ghi hình ZephyrusG14 (Windows 11, RTX 4060 Laptop — không dùng GPU), 3 × Tiandy TC-C131N | **Thời lượng:** ~3h

## Mục tiêu phiên

- Dựng môi trường trên laptop ghi hình để sẵn sàng thu dữ liệu M6 (`docs/m6/laptop-ghi-hinh.md`).
- Đặt cấu hình camera đúng yêu cầu, xác nhận vùng chồng lấn cam01–cam02.
- Hiệu chỉnh homography cả 3 camera và đo khoảng cách cho topology.

## Đã làm

1. **Laptop**: repo clone sẵn ở `D:\hoc\multi-cam-tracking` (`fe8c7db`), Python 3.14 (Anaconda),
   ffmpeg 8.1.2 essentials (`C:\ffmpeg-8.1.2-essentials_build`), ổ D trống 215 GB, cổng LAN
   qua USB–LAN ASIX (100 Mbps), máy không ngủ khi cắm sạc. Tạo `.env` (gitignored).
2. **Mạng**: không có router → camera không ở 192.168.1.x như ở nhà mà ở **IP tĩnh
   169.254.237.201 / .85 / .55** (ONVIF `GetNetworkInterfaces`: DHCP false); laptop có sẵn
   169.254.237.7/24. `.55` = MAC `…cf:b3` = con thứ ba của phiên 34. Lần TCP đầu sau khi mạng
   nghỉ có thể timeout (ARP), chạy `check` lại là qua.
3. **Cấu hình camera qua ONVIF** (script SOAP thư viện chuẩn trong scratchpad, không vào repo):
   - `SetVideoEncoderConfiguration` (Media1): H.265 2304×1296 → **H.264 High 1920×1080 25 fps**.
   - Firmware **nhận nhưng bỏ qua** `GovLength` 25 và `BitrateLimit` 6144, qua cả Media1 lẫn
     Media2 → giữ GOP 100 (khung khoá 4 s), 4096 kbps CBR. Không B-frame.
   - `DeleteOSD time_0001` + `title_0001` trên cả 3: tắt chữ "1970/01/01 …" và "Channel 1".
     Chưa thử khởi động lại camera xem còn giữ không.
4. **`record_rtsp check`** 0 FAIL 0 WARN; **ghi thử 60 s** (`data/lab/raw/thu/`, số ở dưới).
5. **Vùng chồng lấn**: cam01 (.201) và cam02 (.85) treo hai đầu hành lang nhìn vào nhau (~180°).
   Người đứng sát cam02 thì mất chân ở cam02; dời ~1 m về phía cam01 thì cả hai thấy trọn người.
6. **Hiệu chỉnh homography 3 camera** — `configs/lab/ground_points.yaml` (đầu file ghi đủ cách
   làm), `configs/lab/homography/cam0{1,2,3}.yaml` qua `tools.calibrate_homography`.
   - Khung trung vị 30 ảnh / 20 s (1.5 fps) để xoá người đi qua.
   - Canny + HoughLinesP trong đa giác sàn → gom đoạn thành đường ron → khớp đường TLS → giao
     điểm hai họ đường. Điểm ảnh lấy từ giao điểm, không click tay.
   - Fit thử từng camera trên lưới viên, rồi chiếu mọi đường ron khác về toạ độ viên: rơi đúng
     số nguyên (lệch ≤ 0.05 viên) → kiểm được việc đánh số hàng/cột.
   - Nối cam01–cam02: 5 đồ vật trên sàn, cùng một quan hệ hàng(cam02) = 9 − hàng(cam01),
     cột(cam02) = 1 − cột(cam01), lệch ≤ 0.15 viên.
   - Nối cam03: túi đỏ sát cùng bức tường, cách hộp **27 viên = 18.4 m** (thước dây, người dùng
     đo); túi nằm trên ron ngang hàng 0 của cam03 (lệch 0.08 viên). cam03 cùng hành lang thẳng,
     về phía cam02, quay lưng về phía hộp.
   - Chiếu ngược lưới gạch lên ảnh để kiểm bằng mắt: trùng gạch thật ở cả 3 camera, chân tường
     ở y ≈ −0.5 m trùng chân tường thật.
7. **Thử rồi bỏ**: dò góc bằng `goodFeaturesToTrack` + snap lưới — nhận nhầm dây điện trắng dọc
   sàn thành đường ron (lưới sai hẳn). Chuyển sang dò đường (Hough) mới ổn.
8. `configs/lab/topology.yaml`: điền `distance_m` (giữ `status: template`, transit chưa đo).
9. Test: `tests/test_check_lab_setup.py` — hai test vốn khẳng định `configs/lab` còn là bản mẫu;
   sửa thành: homography OK, chỉ topology còn FAIL; phép từ chối file `status: template` kiểm
   trên một bản sao tạm của file điểm thật.
10. Ảnh tư liệu cho báo cáo: `data/lab/report_media/2026-10-17/` (gitignored, có README; ảnh có
    người phải làm mờ mặt).

## Quyết định kỹ thuật

### 1. Lấy điểm hiệu chỉnh từ giao điểm đường ron gạch, không click tay dấu băng dính

- Hiện trường không có băng dính. Sàn caro là một lưới chuẩn sẵn có, dày điểm, phủ cả vùng nhìn.
- Giao điểm của hai đường khớp từ nhiều đoạn chính xác dưới pixel; click tay cỡ ±5 px.
- Đồ vật chỉ dùng để nối chỉ số lưới giữa các camera — chỉ cần đúng tới nửa viên. Người dùng
  đếm vị trí đồ vật lệch gấp đôi so với ảnh (đếm "1 viên" khi đồ cách 2 viên); cách này không
  phụ thuộc vào số đếm đó.
- Giới hạn: mỗi camera chỉ có 2 đường ron dọc trọn trong hành lang (giữa có 1 viên nguyên, hai
  bên là dải cắt ~0.6–0.75 viên), nên điểm trải ngang chỉ 0.68 m; phần ra tới chân tường là
  ngoại suy (hình chiếu chân tường vẫn trùng ảnh thật). Méo ống kính 4 mm chưa hiệu chỉnh —
  sai số trong vùng lưới ≤ 1.7 cm p95 nên chưa cần.

### 2. Bước lưới = 18.4 m / 27 viên = 0.6815 m, không phải 67 cm

- Một viên đo 67 cm, nhưng bước lưới gồm cả ron (~1 cm). Đo dài 27 viên chính xác hơn đo một
  viên; dùng 67 cm thì cả hệ co 1.7% (≈ 30 cm sai ở cam03).
- Phụ thuộc vào độ chính xác số 18.4 m (người dùng ghi "khoảng") — sai 10 cm ≈ 0.5%.

### 3. Hệ toạ độ sàn chung

- Gốc = góc gạch cạnh cái hộp, trên ron dọc gần bức tường có cửa kính/cửa phòng; x dọc hành lang
  dương về phía cam02 (và cam03), y ngang hành lang dương ra xa bức tường đó. Hành lang rộng
  ~2.4 viên ≈ 1.6 m (cả 3 camera khớp nhau về vị trí chân tường).
- Vùng sàn: cam01 x 0 … 4.7 m (thấy xa hơn nữa), cam02 x −1.3 … 2.0 m, cam03 x 18.4 … ~23 m.
  Vùng chung cam01–cam02: x 0 … 2 m.

### 4. ĐANG TREO — cam01 nhìn thấy vùng cam03 ở xa: cần ROI theo camera

- Hành lang thẳng, cam01 nhìn về phía cam03: x = 18.4 m chiếu vào pixel ~(561, 418) của cam01;
  người 1.7 m ở ~20 m cao ~80 px → YOLO bắt được. Cặp cam01–cam03 khi đó không còn là không
  chồng lấn đúng nghĩa, và người đi A → cam03 bị cam01 thấy suốt quãng đường.
- Chúc cam01 xuống không giải được (cần ~20°, mất luôn vùng chung với cam02).
- Đề xuất: lọc vùng sàn **theo từng camera** (giữ cam01 ở x ≤ 6 m) — tạo khoảng mù ~12 m. Phải
  chốt trước khi chú thích CVAT để áp cùng một vạch cho GT và kết quả. `eval/roi_filter_fixture.py`
  (phiên 31) hiện chỉ có một hình chữ nhật chung cho mọi camera. Chờ người dùng duyệt.
- `distance_m` cam01→cam03 = 12.4 m đã tính theo giả định cắt ở x = 6 m.

## Số liệu đo được

Cấu hình: 3 × Tiandy TC-C131N, luồng chính H.264 High 1920×1080 25 fps CBR 4 Mbps GOP 100, RTSP/TCP,
laptop ZephyrusG14 qua USB–LAN 100 Mbps, ffmpeg 8.1.2 `-c copy`.

- **Ghi thử 60 s** (`data/lab/raw/thu/record.json`, `ok: true`):

  | | cam01 | cam02 | cam03 |
  |---|---|---|---|
  | khung / thời lượng | 1494 / 59.7 s | 1494 / 59.7 s | 1494 / 59.7 s |
  | fps | 25.00 | 25.00 | 25.00 |
  | jitter | 19 ms | 20 ms | 20 ms |
  | trôi so với laptop | −9 ppm | −52 ppm | −7 ppm |
  | lệch so với cam01 | 0 | −0.023 s | +0.011 s |

  Sai số do trôi ≤ 1 ms. ~31 MB/phút/camera.
- **Homography** (`tools.calibrate_homography`, bước 0.6815 m):

  | | điểm dùng | RMSE | p95 |
  |---|---|---|---|
  | cam01 | 13/16 | 0.010 m | 0.015 m |
  | cam02 | 10/12 | 0.007 m | 0.010 m |
  | cam03 | 12/14 | 0.011 m | 0.017 m |

  Gộp: trung vị 0.01 m, p95 0.03 m → cặp hai camera lệch cỡ 0.05 m (p95).
- Camera không xê dịch suốt buổi (tương quan pha giữa các lần chụp: ≤ 0.1 px).
- Test: `tests/test_check_lab_setup.py` + `test_pipeline_configs.py` 55 passed, 7 skipped.
  **Chưa chạy bộ test bằng venv 3.10** (laptop không có `uv`); Python 3.14 hệ thống: 727 passed,
  4 failed, cả 4 failed giống hệt trên `main` chưa sửa (môi trường: thiếu `fastapi`/`redis`,
  3.14), không do thay đổi phiên này.

## Vướng mắc / chưa xong

- **ROI theo camera** (quyết định 4) — chờ duyệt, chưa code.
- Transit `min_ms`/`max_ms` còn giữ chỗ → quay buổi `calib`, chú thích, `make lab-transit`.
- GOP 25 / bitrate 6 Mbps chưa đặt được (firmware bỏ qua ONVIF); muốn thì đặt tay trên trang web.
- OSD đã xoá — chưa biết có giữ sau khi camera mất điện.
- 18.4 m là số "khoảng" — nên đo lại cẩn thận.
- Chưa quay buổi nào (calib, s1–s4). VLC còn mở thì đóng trước khi ghi.

## Bước tiếp theo

1. Đóng VLC, dọn đồ, không chạm camera; quay `calib` rồi `s1`–`s4` theo `docs/m6/README.md` mục 3.
2. Chốt ROI theo camera (cam01 x ≤ 6 m) → mở rộng `eval/roi_filter_fixture.py` thành đa giác
   theo camera, trước khi chú thích CVAT.
3. Chép `data/lab/raw/` về máy dev, xoá `.env` + video trên laptop mượn.
