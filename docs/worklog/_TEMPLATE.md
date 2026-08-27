# YYYY-MM-DD — <Tiêu đề ngắn gọn>

- **Mốc:** M<n> (tuần <x>) | **Máy:** Mac / GPU box / cloud GPU | **Thời lượng:** ~<n>h

## Mục tiêu phiên

<1–3 gạch đầu dòng: định làm gì khi bắt đầu.>

## Đã làm

<Việc cụ thể, kèm đường dẫn file đã tạo/sửa.
Ghi cả thứ đã thử rồi bỏ — biết cái gì KHÔNG chạy cũng là kết quả, và tiết kiệm thời gian sau này.>

## Quyết định kỹ thuật

<Chốt cái gì, **vì sao**, phương án nào bị loại và vì sao loại.
Phần này trích thẳng vào báo cáo — viết đủ để 3 tháng sau đọc lại vẫn hiểu.
Quyết định lớn → tạo `docs/adr/NNN-<slug>.md` rồi link tới đây.>

## Số liệu đo được

<FPS/luồng, độ trễ end-to-end, GPU/VRAM, MOTA/IDF1/HOTA...
**Luôn kèm cấu hình đã dùng**: GPU gì, model gì, độ phân giải, số luồng.
Số không tái lập được thì vô nghĩa. Không đo gì thì ghi "—".>

## Vướng mắc / chưa xong

<Bug còn treo, thứ đang chờ (thiết bị, dữ liệu, phản hồi GVHD),
giả định đang dùng nhưng **chưa xác minh**.>

## Bước tiếp theo

<Việc đầu tiên của phiên sau. Càng cụ thể càng tốt — "viết `src/common/schema.py`"
tốt hơn "làm tiếp M0".>
