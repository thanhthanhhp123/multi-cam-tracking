# Laptop ghi hình cho 3 camera IP

Tài liệu cho buổi quay M6 bằng 3 camera IP Tiandy TC-C131N, ghi trên một laptop mang ra hiện
trường (laptop mượn, không phải máy dev). Quy trình M6 đầy đủ ở [README.md](README.md).
Công cụ ghi là `src/tools/record_rtsp.py`. Lý do thiết kế và giới hạn của cách đồng bộ nằm
trong docstring của file đó.

Laptop chỉ cần **Python ≥ 3.10** và **ffmpeg ≥ 5**: không venv, không `pip install`, không
GPU. ffmpeg chỉ chép luồng của camera, không mã hoá lại, nên laptop yếu cũng ghi được.

## A. Cài một lần, ở nơi có Internet, trước ngày quay

1. **Python**: tải từ python.org, lúc cài tích **"Add python.exe to PATH"**. Mở PowerShell,
   chạy `python --version` phải in ra 3.10 trở lên.
2. **ffmpeg**: tải bản "release essentials" (file zip) ở gyan.dev, giải nén thành `C:\ffmpeg`.
   Phải có `C:\ffmpeg\bin\ffmpeg.exe` và `C:\ffmpeg\bin\ffprobe.exe`. Không cần sửa PATH.
3. **Repo**: `git clone https://github.com/thanhthanhhp123/multi-cam-tracking.git`. Nếu
   laptop không có git, hoặc không muốn đăng nhập GitHub trên máy mượn, chép cả thư mục repo
   từ máy dev sang bằng USB.
4. **`.env`** ở gốc repo, soạn bằng Notepad (file này đã gitignore):

   ```
   LAB_FFMPEG=C:\ffmpeg\bin\ffmpeg.exe
   LAB_RTSP_CAM01=rtsp://admin:<mật khẩu>@192.168.1.3:554/unicast/c1/s0/live
   LAB_RTSP_CAM02=rtsp://admin:<mật khẩu>@192.168.1.2:554/unicast/c1/s0/live
   LAB_RTSP_CAM03=rtsp://admin:<mật khẩu>@192.168.1.4:554/unicast/c1/s0/live
   ```

   - **Tên camera theo VỊ TRÍ, không theo IP:** cam01 và cam02 là cặp nhìn chung khu A, cam03
     là camera ở lối đi. Lúc lắp xong thì sửa IP cho khớp với vị trí thật.
   - Nếu mật khẩu có `#`, `@`, `/`, `?` hoặc `:`, viết chúng thành `%23`, `%40`, `%2F`, `%3F`,
     `%3A`. Viết sai thì `check` sẽ báo ra.
   - `s0` là luồng chính (độ phân giải đầy đủ). Không dùng `s1` (luồng phụ, độ phân giải thấp).
5. **Cài đặt camera** trên trang quản lý `http://<IP camera>`. Ba con phải đặt **giống hệt
   nhau**:
   - luồng chính **1920×1080, 25 fps, H.264, CBR**, I-frame interval 25;
   - tắt smart codec / mã hoá thông minh nếu có;
   - khoá chế độ ngày, để camera không tự chuyển sang hồng ngoại đen trắng;
   - IP tĩnh: camera thứ 3 đặt 192.168.1.4 (lúc mới cắm, nó có thể trùng 192.168.1.2 của con
     kia).
6. **Diễn tập ở nhà**, cắm đủ 3 camera:

   ```powershell
   cd <thư mục repo>
   python src/tools/record_rtsp.py check
   python src/tools/record_rtsp.py record --session thu --duration 60
   ```

   - `check` phải ra **0 FAIL**.
   - Bản ghi thử nằm ở `data/lab/raw/thu/`. Mở từng file `.mkv` bằng VLC, rồi xem
     `record.json`: mục `ok` phải là `true`, và mục `fps` của mỗi camera phải gần 25.
   - Chép `data/lab/raw/thu/` về máy dev để kiểm lại trước ngày quay.

## B. Mạng ở hiện trường

- **3 camera và laptop cắm chung một switch.** Dùng switch PoE nếu camera nhận nguồn qua dây
  mạng. Nếu không, mỗi camera cần một adapter 12 V và một ổ điện gần đó. Kiểm trước dây mạng
  tới cam03 có đủ dài không.
- **Không có router thì không có DHCP**, nên phải đặt IP tĩnh cho cổng mạng dây của laptop:
  Settings → Network & internet → Ethernet → IP assignment → Edit → Manual → IPv4, đặt
  `192.168.1.100`, mask `255.255.255.0`, gateway để trống. Laptop không có cổng LAN thì dùng
  bộ chuyển USB–LAN.
- **Tắt Wi-Fi của laptop lúc ghi.** Nếu Wi-Fi nơi đó cũng là dải 192.168.1.x, Windows có thể
  đi nhầm đường sang Wi-Fi và không thấy camera.
- **Không có Internet thì camera không đồng bộ giờ (NTP) được.** Điều này không ảnh hưởng,
  vì đồng bộ dựa vào đồng hồ của laptop. Đừng dùng giờ in trên hình để đồng bộ.
- Laptop luôn **cắm sạc**, và đặt Settings → System → Power → Screen and sleep → **Never**
  khi cắm sạc.

## C. Mỗi buổi quay (calib, s1, s2, s3, s4)

1. `python src/tools/record_rtsp.py check` phải ra **0 FAIL**. Chạy lại mỗi khi cắm lại dây
   hoặc dời camera.
2. `python src/tools/record_rtsp.py record --session calib`, thay `calib` bằng tên buổi. Cứ
   2 giây màn hình in một dòng trạng thái; cả 3 camera phải chạy khoảng 25 fps.
3. **Vỗ tay trong vùng chung của cam01 và cam02**, sao cho cả hai camera cùng thấy bàn tay:
   1 tiếng, nghỉ 2 giây, rồi 2 tiếng nhanh.
4. Thực hiện kịch bản.
5. Vỗ tay lần nữa, cùng chỗ và cùng nhịp.
6. Bấm **Enter** để dừng (Ctrl+C cũng được). Đọc phần tóm tắt:
   - không có dòng `[FAIL]` thì bản ghi dùng được;
   - nếu có `BUỔI NÀY HỎNG`, hoặc giữa chừng hiện `!! camXX: MẤT HÌNH`, thì kiểm dây rồi quay
     lại buổi đó với tên mới (`s1b`, `s1c`…). Công cụ không cho ghi đè buổi đã có, trừ khi
     thêm `--overwrite`.
7. Ghi lên giấy: tên buổi, ai tham gia, mặc gì, có gì bất thường.

Mỗi buổi tạo ra thư mục `data/lab/raw/<buổi>/` gồm:
- `camXX.mkv`: luồng gốc của camera, chưa mã hoá lại;
- `camXX.progress.csv`: số liệu dùng để đồng bộ;
- `camXX.ffmpeg.log`: log của ffmpeg, mật khẩu đã che;
- `record.json`: độ lệch giữa các camera, cảnh báo, và lệnh đồng bộ in sẵn.

Dung lượng khoảng 1,5–3 GB cho 15 phút của cả 3 camera.

## D. Sau buổi quay

1. Chép cả thư mục `data/lab/raw/` sang USB, rồi chép vào đúng `data/lab/raw/` của repo trên
   máy dev.
2. Kiểm trên máy dev xem đủ mọi buổi và mỗi buổi có `record.json` với `ok: true`, rồi mới xoá
   trên laptop.
3. **Xoá khỏi laptop mượn:** file `.env` (có mật khẩu camera) và thư mục `data/lab/raw/`
   (video người thật, đề cương mục 4.3.3).

Bước tiếp theo là đồng bộ: `docker/vast_lab.sh sync <buổi>` tự đọc độ lệch trong
`record.json` (README mục 4).
