# Checklist ngày quay (in ra, mang theo)

Quy trình đầy đủ: [README.md](README.md). Laptop và mạng: [laptop-ghi-hinh.md](laptop-ghi-hinh.md).

> Hiện trường hành lang (phiên 35) đã dựng và hiệu chỉnh xong; ngày quay lặp dùng
> [checklist-quay-hanh-lang.md](checklist-quay-hanh-lang.md) thay cho bản này.

## Mang theo
- [ ] 3 camera IP + giá đỡ/ốc vít để treo cao 2–2.5 m
- [ ] Switch (PoE nếu dùng PoE) hoặc 3 adapter 12 V; 4 dây mạng đủ dài; ổ cắm kéo dài
- [ ] Laptop ghi hình + sạc + bộ chuyển USB–LAN (nếu laptop không có cổng LAN)
- [ ] Laptop đã diễn tập ở nhà: `check` 0 FAIL, bản ghi thử `thu` mở được (laptop-ghi-hinh.md mục A)
- [ ] USB/ổ cứng trống ≥ 10 GB để chép dữ liệu về
- [ ] Thước dây ≥ 10 m, băng dính màu, bút, giấy, bảng ghi toạ độ (dưới)
- [ ] Phiếu đồng thuận đã in, đủ cho mọi người tham gia
- [ ] Áo cùng màu cho kịch bản 4 (2–3 người)

## Mạng và laptop
- [ ] Camera + laptop cắm chung switch, đèn mạng của cả 3 camera sáng
- [ ] IP tĩnh của laptop: 192.168.1.100 / 255.255.255.0; Wi-Fi TẮT
- [ ] Laptop cắm sạc, không tự ngủ
- [ ] `.env` gán đúng IP → vị trí: cam01 = ______ cam02 = ______ cam03 = ______
- [ ] `python src/tools/record_rtsp.py check` → 0 FAIL

## Bố trí
- [ ] cam01, cam02 nhìn chung vùng sàn khu A, lệch góc ≥ 45°
- [ ] cam03 ở lối đi, không thấy khu A
- [ ] Mọi camera cao 2–2.5 m, thấy trọn người kể cả bàn chân
- [ ] Đã chọn gốc toạ độ sàn: ______________________ (trục x: ______________)
- [ ] 6–8 dấu băng dính trong vùng nhìn MỖI camera, không 3 dấu thẳng hàng
- [ ] Đã đo toạ độ mọi dấu (bảng dưới), đo 2 lần dấu ở xa
- [ ] Đã đo quãng đi bộ: khu A → cam03 ____ m
- [ ] Đã chụp ảnh sơ đồ, ghi chiều cao/góc từng camera

| Camera | Dấu | x (m) | y (m) | Ghi chú |
|---|---|---|---|---|
| cam01 | 1 | | | |
| cam01 | 2 | | | |
| cam01 | 3 | | | |
| cam01 | 4 | | | |
| cam01 | 5 | | | |
| cam01 | 6 | | | |
| cam02 | 1 | | | |
| cam02 | 2 | | | |
| cam02 | 3 | | | |
| cam02 | 4 | | | |
| cam02 | 5 | | | |
| cam02 | 6 | | | |
| cam03 | 1 | | | |
| cam03 | 2 | | | |
| cam03 | 3 | | | |
| cam03 | 4 | | | |
| cam03 | 5 | | | |
| cam03 | 6 | | | |

## Mỗi buổi quay (calib, s1, s2, s3, s4)
- [ ] `python src/tools/record_rtsp.py record --session <buổi>`; cả 3 camera ~25 fps
- [ ] Vỗ tay trong vùng chung cam01–cam02: 1 tiếng — nghỉ 2 giây — 2 tiếng nhanh
- [ ] Thực hiện kịch bản
- [ ] Vỗ tay lại cùng chỗ, cùng nhịp
- [ ] Enter để dừng; tóm tắt KHÔNG có `[FAIL]` / `BUỔI NÀY HỎNG` (có thì quay lại, tên `s1b`…)
- [ ] Ghi: tên buổi ____ giờ bắt đầu ____ người tham gia ____ trang phục ____ bất thường ____

| Buổi | Kịch bản | Người | Thời lượng | Xong |
|---|---|---|---|---|
| calib | 1 người đi A → cam03 và ngược lại, chậm / thường / nhanh | 1 | ~3 phút | [ ] |
| s1 | khu A, qua lại vùng chung cam01–cam02 | 1–2 | ~2 phút | [ ] |
| s2 | A → cam03 → quay lại ×3, một lần khuất cam03 rồi quay lại | 1–2 | ~3 phút | [ ] |
| s3 | 4–6 người cùng lúc, cắt ngang, che nhau | 4–6 | ~3 phút | [ ] |
| s4 | 2–3 người áo cùng màu | 2–3 | ~3 phút | [ ] |

## Trước khi rời hiện trường
- [ ] Mở thử một file `.mkv` của mỗi camera
- [ ] Đủ 5 buổi, mỗi buổi có `record.json` với `"ok": true`
- [ ] Chép `data/lab/raw/` sang USB
- [ ] Chụp ảnh dấu sàn từ đúng vị trí mỗi camera
- [ ] Thu lại phiếu đồng thuận đã ký
- [ ] Về nhà, kiểm bản chép trên máy dev xong mới xoá `.env` và video khỏi laptop mượn
