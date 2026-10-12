# 2026-10-12 (phiên 34): Có 3 camera IP — công cụ ghi RTSP trên laptop mượn, đồng bộ theo đồng hồ máy ghi, chuyển cấu hình lab sang 3 camera

- **Mốc:** M6 (thiết bị + công cụ ghi) | **Máy:** máy dev (CPU, không thuê GPU) | **Thời lượng:** ~1.5h

## Mục tiêu phiên

- Người dùng báo đã có camera IP: Tiandy TC-C131N (3MP, ống kính 4 mm).
  - Lúc đầu hai con cắm vào LAN của máy dev: 192.168.1.3 và 192.168.1.2.
  - Sau đó báo có **3 con** (con thứ ba chưa cắm).
  - Kế hoạch: clone repo vào **laptop của một người bạn**, mang ra chỗ khác để thu.
- Mục tiêu: dựng đường thu dữ liệu cho camera IP. Quy trình M6 của phiên 33 viết cho 4 điện
  thoại đồng bộ bằng tiếng vỗ tay.

## Đã làm

1. **Kiểm từ máy dev** (máy dev = 192.168.1.224, Ethernet, máy bàn không pin):
   - cả hai camera mở cổng RTSP 554;
   - DESCRIBE không xác thực trả `401`, `WWW-Authenticate: Digest realm="RTSP SERVER"`;
   - chưa có mật khẩu nên chưa đọc được luồng;
   - máy dev chưa có ffmpeg (có VLC).
2. **`src/tools/record_rtsp.py`** (mới). Hai lệnh `check` và `record`.
   - **Chỉ thư viện chuẩn**, gọi thẳng được bằng `python src/tools/record_rtsp.py`, không cần
     PYTHONPATH.
   - Camera đọc từ `.env` (`LAB_RTSP_CAMXX`, `LAB_FFMPEG`). Đọc được cả `.env` UTF-16 do
     PowerShell 5.1 ghi.
   - `check`:
     - ffmpeg ≥ 5 và ffprobe;
     - dung lượng đĩa;
     - TCP tới từng camera;
     - ffprobe từng luồng: codec, kích thước, fps, có audio không;
     - hai `cam_id` trùng một luồng;
     - mật khẩu có ký tự làm hỏng URL;
     - lỗi 401/404/timeout được dịch ra lời.
   - `record`:
     - mỗi camera một ffmpeg `-c copy` ra Matroska, `-rtsp_transport tcp`, `-timeout 5 s`,
       `-progress pipe:1 -stats_period 0.1`;
     - trạng thái mỗi 2 s, báo camera mất hình;
     - dừng bằng Enter/Ctrl+C/`--duration`; camera chết giữa chừng thì dừng cả buổi và báo HỎNG;
     - không cho ghi đè buổi đã có;
     - ra `record.json`: độ lệch, trôi đồng hồ, jitter, cảnh báo, và lệnh `sync_recordings` in sẵn;
     - mật khẩu che trong `record.json` và log ffmpeg.
3. **`tests/test_record_rtsp.py`** (mới, 28 test):
   - không import gì ngoài thư viện chuẩn; chạy được bằng `python -I <file>`;
   - đọc `.env`; che mật khẩu; ký tự đặc biệt trong mật khẩu;
   - đường bao dưới tìm đúng gốc thời gian, độ trôi 80 ppm, bỏ qua cụm gói đầu file;
   - dấu của độ lệch;
   - **lệnh in sẵn chạy được qua `sync_recordings.main` thật và cắt đúng cửa sổ**;
   - các trường hợp của `check`;
   - ghi trọn một buổi bằng ffmpeg giả (độ lệch ±0.06 s, không lọt mật khẩu), camera chết giữa
     chừng, từ chối ghi đè.
   - Test bắt được một lỗi thật: `--help` chết vì cp1252 (CLAUDE.md §11). Đã chuyển
     `reconfigure` lên đầu `main`, cho cả stderr.
4. **Cấu hình lab chuyển sang 3 camera**:
   - `configs/lab/{topology,ground_points,streams_lab}.yaml` bỏ cam04 và chuyển tiếp
     cam03→cam04;
   - `configs/pipeline/config_infer_yolo11_b3.txt` mới (QĐ 3), `streams_lab` trỏ vào nó;
   - `Makefile LAB_CAMS`.
5. **`docker/vast_lab.sh`**:
   - danh sách camera lấy từ `streams_lab.yaml`;
   - `cat_engine_yolo` theo số camera;
   - `sync <buổi>` có `record.json` thì chạy đúng lệnh trong đó, buổi hỏng thì dừng; không có
     thì đồng bộ bằng vỗ tay như cũ.
   - Kiểm: `bash -n`, và chạy riêng đoạn Python nhúng với `record.json` tốt và hỏng.
   - Lần đầu, chuỗi `\n` trong đoạn nhúng bị biến thành xuống dòng thật mà `bash -n` không bắt.
     Đã sửa bằng `sep=chr(10)`.
6. **Tài liệu**:
   - `docs/m6/laptop-ghi-hinh.md` (mới): cài một lần, mạng hiện trường, mỗi buổi quay, sau buổi
     quay;
   - `docs/m6/README.md` mục 1–5, 10;
   - `docs/m6/checklist-ngay-quay.md` viết lại cho camera IP;
   - `.env.example`, `README.md`, CLAUDE.md §4/§9/§11/§12.

## Quyết định kỹ thuật

### 1. Ba camera IP thì đặt cấu hình HỖN HỢP trong cùng một buổi

- cam01 và cam02 chồng lấn ở khu A, cam03 ở lối đi.
- Đi A → cam03 → A cho đủ ba loại bàn giao (chồng lấn / không chồng lấn / quay lại cùng camera)
  trong mọi buổi, đúng đối tượng của đóng góp chính.
- *Loại: 2 camera, quay hai lần với hai cách bố trí.* Không buổi nào có cấu hình hỗn hợp.
- *Loại: camera IP + điện thoại.* Hai cách đồng bộ khác nhau trong một buổi. Vẫn ghi lại làm
  phương án dự phòng nếu một camera hỏng.

### 2. Ghi trên một laptop, đồng bộ theo đồng hồ máy ghi (đường bao dưới), không dựa vỗ tay

- **Ghi ra file.** `vast-gpu` không vào được LAN của camera. Ghi ra file còn cho n = 3 lần chạy
  trên cùng dữ liệu, và cho chính file đó lên CVAT. `-c copy` không mã hoá lại, nên laptop yếu
  vẫn chịu được.
- **Đồng hồ chung.** Cả 3 luồng được nhận trên cùng một máy. Hiệu `t − p` (giờ nhận báo cáo
  tiến độ − vị trí trong file) chỉ cộng thêm độ trễ dương. Đường bao dưới (min theo khoảng 5 s,
  rồi hồi quy) cho gốc thời gian của từng file, và cho cả độ trôi đồng hồ camera.
  - Độ lệch tính ở giữa cửa sổ chung, nên sai số do trôi ở hai mép chỉ còn một nửa.
  - Phần sai số đó được báo ra trong `edge_error_ms`.
  - Dùng `time.perf_counter` vì `time.monotonic` trên Windows chỉ phân giải ~15.6 ms.
- **Vì sao không dùng vỗ tay.**
  - Chưa biết camera có micro không.
  - Vỗ tay chỉ kiểm được bằng mắt cho cặp chồng lấn.
  - Ở hiện trường không có Internet, nên camera không có NTP và giờ in trên hình không tin được.
- **Giới hạn, phải ghi vào chương 4:**
  - không đo được chênh lệch độ trễ mã hoá/mạng GIỮA các camera;
  - pha chụp lệch nhau tới nửa khung.
  Vì vậy vẫn vỗ tay trong vùng chung để kiểm cặp cam01–cam02 bằng mắt (±1 khung). cam03
  không cần, vì `transitions` đã nới hàng giây.
- *Loại: `-use_wallclock_as_timestamps`.* Dấu thời gian của khung sẽ mang jitter mạng, rồi
  bước CFR của `sync_recordings` nhân hoặc bỏ khung theo jitter đó. Gốc thời gian lại chỉ lấy
  từ một gói đầu, đúng gói I-frame to bị trễ nhất.
- *Loại: thời gian NTP trong RTCP Sender Report.* Đó là đồng hồ của camera, và ở hiện trường
  không có NTP.
- *Loại: tự viết client RTSP.* Rủi ro, trong khi ffmpeg đã có sẵn.

### 3. Engine YOLO batch 3 riêng (`config_infer_yolo11_b3.txt`)

- `builder.py` đặt `batch-size` = số nguồn. Ba nguồn mà trỏ vào engine b4 thì DeepStream-Yolo
  build engine b3 vào `<cwd>`, `cat_engine_yolo` tìm b4 không thấy, và **mỗi lần chạy build lại
  ~4 phút** (CLAUDE.md §11).
- Theo đúng quy ước "mỗi batch một config" có test canh (`PGIE_FAMILIES`). Test của
  `streams_lab` giờ đòi batch của config = số nguồn, thay cho đòi trùng `pgie` với
  `streams_reid`.

### 4. Công cụ ghi chỉ dùng thư viện chuẩn, tự kiểm, không ghi đè

- Laptop mượn, ở hiện trường không có ai sửa hộ. Mọi lỗi đã thấy được chuyển thành `[FAIL]`
  có lời giải thích:
  - sai IP/dải mạng, sai mật khẩu, sai đường dẫn luồng;
  - hai camera trùng luồng (đồng bộ trông hoàn hảo mà dữ liệu vô dụng);
  - mật khẩu có `#` (bị hiểu thành fragment của URL);
  - ffmpeg < 5 (khi đó `-timeout` nghĩa là LẮNG NGHE, ffmpeg treo thay vì báo mất camera).
- Không cho ghi đè buổi đã có: ghi đè nhầm một buổi là mất cả một lần quay.

## Số liệu đo được

- Chưa đo gì trên camera thật (chưa có mật khẩu, máy dev chưa có ffmpeg).
- Test: **810 passed, 5 skipped** trong 29.8 s (phiên 33: 777). `ruff check` và
  `ruff format --check` sạch trên `src tests eval`.
- Ghi bằng ffmpeg giả (3 camera bắt đầu lệch nhau 0 / +0.4 / −0.2 s, 2.5 s): độ lệch tìm ra
  nằm trong ±0.06 s. Đây chỉ là kiểm cơ chế ghép nối, không phải độ chính xác đồng bộ thật.

## Vướng mắc / chưa xong

- **Chưa chạy với ffmpeg thật và camera thật.** Bốn giả định cần kiểm ở lần ghi thử đầu tiên:
  1. `-progress` báo `frame` và `out_time_us` cho luồng chép (`-c copy`);
  2. dấu thời gian RTP của Tiandy vào Matroska không lỗi (`Timestamps are unset` thì thêm
     `-fflags +genpts`);
  3. `-timeout` (µs) làm ffmpeg thoát khi rút dây camera;
  4. jitter và độ trôi trong `record.json` ở mức hợp lý (jitter vài ms, trôi vài chục ppm).
- `sync_recordings` chưa chạy với ffmpeg thật (vẫn treo từ phiên 33). File vào giờ là `.mkv`.
- Chưa biết camera thứ ba, nguồn điện (PoE hay adapter) và địa điểm quay.
- Laptop của bạn: Python, ffmpeg, git, cổng LAN đều chưa kiểm.
- Thay đổi chưa commit. Laptop clone từ GitHub nên phải commit và push trước khi đem đi.
- Vẫn treo: kết quả buổi gặp GVHD.

## Bước tiếp theo

1. Người dùng thêm mật khẩu vào `.env` trên máy dev. Đồng ý thì tải ffmpeg static về máy dev
   (~100 MB, chỉ chép luồng nên nhẹ máy), rồi chạy `check` và `record --session thu
   --duration 60` với 2 camera đang cắm. Đọc `record.json` và log để kiểm 4 giả định ở trên.
2. Commit, rồi push khi người dùng đồng ý, để laptop clone được.
3. Diễn tập trên laptop ở nhà với cả 3 camera (`docs/m6/laptop-ghi-hinh.md` mục A.6).
4. Quay theo `docs/m6/README.md` mục 3, rồi `vast_lab.sh sync` (đọc `record.json`) → hiệu chỉnh
   homography → CVAT.
