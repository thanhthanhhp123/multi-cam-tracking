# Hiệu chỉnh homography ảnh → mặt phẳng mặt đất

Mỗi camera một file `<cam_id>.yaml`, đọc bằng `mct.homography.HomographyMapper.load(<thư mục>)`.
Tạo bằng `python -m tools.calibrate_homography` (xem docstring của module để biết hai cách
lấy cặp điểm).

Ma trận 3x3 biến điểm chân trong ảnh (**pixel, theo độ phân giải GỐC của camera**) thành
toạ độ mét trên mặt phẳng tham chiếu chung. `image_size` ghi lại độ phân giải lúc hiệu
chỉnh chính là để bắt lỗi kinh điển "bbox đang theo toạ độ streammux" (CLAUDE.md §5) —
`HomographyMapper.check_frame_size()` cảnh báo khi lệch.

Chỉ những camera nằm chung một `plane` mới so khoảng cách với nhau được; nạp lẫn hai mặt
phẳng khác nhau thì `HomographyMapper` từ chối ngay.

## Thư mục

- **`./`** (gốc) — camera THẬT của đồ án. Chưa có: chưa lắp camera, chưa đo điểm mặt đất.
- **`wildtrack/`** — 7 camera của dataset WildTrack, khớp tự động từ chú thích của dataset
  (`--wildtrack-dir`). Đây là **dataset mượn để thử thuật toán**, không phải hệ thống thật;
  để riêng thư mục cho khỏi lẫn. Sai số hiệu chỉnh rất nhỏ (RMSE 0.010–0.032 m) vì bbox
  của WildTrack vốn được sinh ra bằng cách chiếu vị trí lưới qua calibration của dataset,
  nên phép khớp gần như khôi phục lại đúng phép chiếu đó. **Đừng lấy con số đó làm kỳ vọng
  cho camera thật**: ở đó điểm chân đến từ bbox của detector, sai số sẽ lớn hơn nhiều bậc.
