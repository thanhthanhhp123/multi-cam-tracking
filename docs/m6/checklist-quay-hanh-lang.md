# Checklist quay lặp — hành lang (in ra, mang theo)

Bản dành riêng cho hiện trường đã dựng và hiệu chỉnh ở phiên 35. Bố trí, IP và homography xem
`docs/worklog/2026-10-17-35-*`. Bản checklist chung (dựng từ đầu, dán băng dính) là
[checklist-ngay-quay.md](checklist-ngay-quay.md), hôm nay **không cần** phần đó.

**Vì sao quay lặp** (GVHD yêu cầu đủ dữ liệu): độ chính xác bàn giao chỉ tin được khi có
≥ 50 lần bàn giao mỗi loại cặp. Với 10 lần, khoảng tin cậy 95% rộng ±25%; với 50 lần còn ±11%.
Ba lần chạy pipeline (n = 3) chỉ đo dao động của GPU, không thay được nhiều đoạn video khác nhau.

## Mốc trên sàn (nhắc lại)

| Tên | Ở đâu | Ghi chú |
|---|---|---|
| **Khu A** | x = 0–2 m, ô gạch cạnh cái hộp | cam01 và cam02 cùng thấy trọn người, kể cả chân |
| **Vùng cam03** | từ túi đỏ trở đi (x ≈ 18,4 m) | không camera nào khác thấy |
| **Đoạn giữa** | x ≈ 6–18 m | **đi qua, không dừng** (có thể bị lọc khỏi chấm điểm) |
| **Điểm vỗ tay** | giữa khu A, cạnh cái hộp | |
| **Điểm khuất cam03** | ______________________ | chỗ đứng mà cam03 không thấy, dùng cho `s2` |

## Người tham gia

Chia hai nhóm. **Nhóm B chỉ xuất hiện ở các lần quay đánh dấu (B)**: đó là tập test giữ riêng,
không trùng người với phần còn lại, để sau này làm thí nghiệm có/không fine-tune mà không bị
lẫn người giữa train và test.

| Nhóm | Người | Tên | Trang phục 1 | Trang phục 2 | Đã ký phiếu |
|---|---|---|---|---|---|
| A | 1 | | | | [ ] |
| A | 2 | | | | [ ] |
| A | 3 | | | | [ ] |
| B | 4 | | | — | [ ] |
| B | 5 | | | — | [ ] |
| B | 6 | | | — | [ ] |

- Tối thiểu 4 người (A = 2, B = 2). Đủ 6 thì tốt nhất.
- Nhóm A mang thêm **một áo khoác hoặc áo khác màu** (trang phục 2).
- `s4` cần **áo cùng màu** cho 2–3 người mỗi nhóm.

## Trước khi bắt đầu

- [ ] Laptop cắm sạc, Wi-Fi tắt, Ethernet 2 có `169.254.237.7`
- [ ] **Đóng hết cửa sổ VLC**
- [ ] `python src/tools/record_rtsp.py check` → 0 FAIL (lần đầu timeout thì chạy lại)
- [ ] Không có chữ giờ/“Channel 1” trên hình (camera mất điện có thể bật lại OSD)
- [ ] Đĩa trống ≥ 10 GB (cả buổi khoảng 45 phút × 3 camera ≈ 4–5 GB)
- [ ] Nhắc mọi người: **không chạm vào camera** suốt buổi. Lỡ chạm thì báo ngay, phải hiệu chỉnh lại
- [ ] Mọi người đã ký phiếu đồng thuận

## Mỗi lần quay (lặp lại đúng y như nhau)

1. [ ] `python src/tools/record_rtsp.py record --session <tên>` (tên ở bảng dưới)
2. [ ] Một người vỗ tay ở điểm vỗ tay: **1 tiếng, nghỉ 2 giây, 2 tiếng nhanh**
3. [ ] Thực hiện kịch bản. Đi tự nhiên, không nhìn camera
4. [ ] Vỗ tay lại cùng chỗ, cùng nhịp
5. [ ] Bấm Enter. Tóm tắt **không** có `[FAIL]` hay `BUỔI NÀY HỎNG`. Có thì quay lại, thêm `b` vào tên (`s2_3b`)
6. [ ] Điền dòng tương ứng ở bảng dưới (giờ, ai, ghi chú)

## Kịch bản

**calib — đo thời gian đi (không chấm điểm).** Một người, 6 lượt. Mỗi lượt: đứng trong khu A
3 giây → đi sang vùng cam03, đứng 3 giây → đi về khu A. Tốc độ: chậm, chậm, thường, thường,
nhanh (không chạy), nhanh.

**s1 — chồng lấn cam01–cam02.** 1–2 người, ~2 phút. Đi vào khu A, ra khỏi khu A, rồi vào lại,
nhiều lần, đổi hướng. Có lúc hai người đi ngang nhau trong khu A.

**s2 — không chồng lấn + quay lại cùng camera.** 2 người, ~3 phút, mỗi người 3 lượt
khu A → vùng cam03 → khu A. Ở **2 trong 3 lượt**: tới vùng cam03, bước ra điểm khuất,
đứng 3–5 giây rồi quay lại vào vùng cam03 trước khi về.
Hai người đi lệch nhau vài giây, không đi song song.

**s3 — đông người, che khuất.** 4–6 người, ~3 phút. Đi cùng lúc, cắt ngang nhau trong khu A,
đi ngược chiều nhau trên hành lang, có lúc đi thành nhóm sát nhau. **Chỉ dùng để chấm, không
bao giờ dùng để fine-tune** (trộn cả hai nhóm).

**s4 — áo cùng màu.** 2–3 người cùng nhóm mặc áo cùng màu, ~3 phút. Đi các tuyến của s1 và s2,
có lúc đổi chỗ cho nhau ở đoạn giữa (khi không camera nào thấy).

## Thứ tự quay

| # | Tên buổi | Nhóm | Trang phục | Thời lượng | Giờ bắt đầu | Người | Ghi chú | Xong |
|---|---|---|---|---|---|---|---|---|
| 1 | `calib` | 1 người nhóm A | 1 | ~5 phút | | | | [ ] |
| 2 | `s1_1` | A | 1 | ~2 phút | | | | [ ] |
| 3 | `s2_1` | A | 1 | ~3 phút | | | | [ ] |
| 4 | `s2_2` | A (cặp khác) | 1 | ~3 phút | | | | [ ] |
| | *nhóm A đổi sang trang phục 2* | | | | | | | |
| 5 | `s1_2` | A | 2 | ~2 phút | | | | [ ] |
| 6 | `s2_3` | A | 2 | ~3 phút | | | | [ ] |
| | *nhóm A mặc áo cùng màu* | | | | | | | |
| 7 | `s4_1` | A | cùng màu | ~3 phút | | | | [ ] |
| 8 | `s4_2` | A (đổi tuyến) | cùng màu | ~3 phút | | | | [ ] |
| | *nhóm B vào* | | | | | | | |
| 9 | `s1_3` (B) | B | 1 | ~2 phút | | | | [ ] |
| 10 | `s2_4` (B) | B | 1 | ~3 phút | | | | [ ] |
| 11 | `s4_3` (B) | B | cùng màu | ~3 phút | | | | [ ] |
| | *cả hai nhóm* | | | | | | | |
| 12 | `s3_1` | A + B | tự do | ~3 phút | | | | [ ] |
| 13 | `s3_2` | A + B | tự do | ~3 phút | | | | [ ] |
| 14 | `s3_3` | A + B | tự do | ~3 phút | | | | [ ] |
| 15 | `s2_5` *(nếu còn thời gian)* | A | 2 | ~3 phút | | | | [ ] |

Tổng khoảng 45 phút video, khoảng 1,5–2 giờ cả buổi.

Ước lượng số lần bàn giao (cộng các lần quay của một kịch bản):
- chồng lấn: dư, mỗi lần ra vào khu A là một lần;
- không chồng lấn: 2 người × 3 lượt × 2 chiều × 4 lần quay `s2` ≈ 48 (thêm `s2_5` thì ≈ 60);
- quay lại cùng camera: 2 người × 2 lượt × 4 lần ≈ 16.

## Trước khi rời hiện trường

- [ ] Mỗi buổi có `data/lab/raw/<tên>/record.json` với `"ok": true`
- [ ] Mở thử một file `.mkv` của mỗi camera ở buổi cuối (đóng VLC sau đó)
- [ ] Chép `data/lab/raw/` sang ổ ngoài, so dung lượng hai bên
- [ ] Thu lại phiếu đồng thuận
- [ ] Về tới máy dev, kiểm bản chép xong mới xoá `.env` và `data/lab/raw/` trên laptop mượn

## Sau buổi quay (máy dev)

Chú thích CVAT theo thứ tự ưu tiên: `calib` → `s2_*` → `s1_*` → `s4_*` → `s3_*`. Lần quay
chưa kịp chú thích vẫn dùng được cho demo và đo FPS/độ trễ. Chấm điểm hiện chạy theo từng buổi;
cần thêm bước gộp các lần quay của cùng kịch bản (cộng dồn số lần bàn giao, tính khoảng tin cậy).
