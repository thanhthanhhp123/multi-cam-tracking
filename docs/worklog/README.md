# Nhật ký thực hiện đồ án (worklog)

Mỗi phiên làm việc ghi **một file** trong thư mục này.

## Quy ước

- Tên file: `YYYY-MM-DD-<slug-khong-dau>.md` — ví dụ `2026-09-03-pipeline-1-camera.md`.
  Không dấu, không khoảng trắng, để sắp xếp theo thời gian tự nhiên.
- Nếu một ngày có nhiều phiên tách bạch: thêm hậu tố `-2`, `-3`.
- Copy `_TEMPLATE.md` rồi điền. Phần nào không có thì ghi "—", **đừng xoá đề mục** —
  đề mục trống cũng là thông tin (ví dụ: phiên này không đo được số liệu nào).
- Viết ngay cuối phiên, khi còn nhớ **lý do** đã chọn phương án đó. Ghi bù sau vài ngày
  thì phần "Quyết định kỹ thuật" luôn mất giá trị nhất.

## Vì sao cần

Ba mục đích, theo thứ tự quan trọng:

1. **Nạp lại ngữ cảnh.** Phiên sau (người hoặc Claude) đọc 2–3 file gần nhất là biết đang ở đâu,
   thứ gì đang treo, giả định nào chưa xác minh.
2. **Vật liệu cho báo cáo.** Mục "Quyết định kỹ thuật" và "Số liệu đo được" trích thẳng vào
   chương 3 (thiết kế), chương 5 (triển khai) và chương 6 (đánh giá). Cuối kỳ ngồi nhớ lại
   *vì sao chọn NvDCF thay vì ByteTrack* là không khả thi — phải ghi lúc đang quyết.
3. **Báo cáo tiến độ với GVHD.** Mỗi lần gặp, đọc lại các file từ lần gặp trước là đủ nội dung.

## Quan hệ với `docs/adr/`

- **worklog** = nhật ký theo thời gian, mọi thứ xảy ra trong phiên.
- **`docs/adr/`** = chỉ những quyết định kiến trúc lớn, đủ sức nặng để cần một trang riêng
  (ví dụ: chọn Redis Streams làm ranh giới hệ thống). Mỗi ADR là `NNN-<slug>.md`.

Quyết định nhỏ ghi thẳng trong worklog. Quyết định lớn: tạo ADR, rồi trong worklog chỉ link tới nó.

## Mục lục

| Ngày | Mốc | Nội dung |
|---|---|---|
| [2026-08-27](2026-08-27-khoi-tao-kien-truc.md) | trước M0 | Đọc đề cương, chốt kiến trúc 3 tầng, viết CLAUDE.md |
| [2026-08-27 (2)](2026-08-27-2-m0-khung-du-an.md) | M0 xong | Khung repo, contract dữ liệu, wrapper Redis, bộ sinh fixture |
