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
| [2026-08-27 (3)](2026-08-27-3-khao-sat-ut-hpc.md) | trước M1 | Khảo sát ut-hpc, chốt vai trò 3 máy (Mac / ut-hpc / vast-gpu) |
| [2026-09-01](2026-09-01-wildtrack-fixture-m4.md) | chuẩn bị M4 | Bộ chuyển WildTrack → fixture (OSNet ONNX/CPU), kéo M4 lên sớm bằng dữ liệu thật |
| [2026-09-03](2026-09-03-skill-ut-hpc.md) | chuẩn bị M2/M3 | Vọc lại ut-hpc (partition `main-gpu`, node tính toán không có mạng), skill `ut-hpc` |
| [2026-09-03 (3)](2026-09-03-3-m1-vast-deepstream.md) | M1 | Thuê vast.ai, pipeline DeepStream 1 camera chạy thật (410.8 FPS), chốt DS 7.1/CUDA 12.6, Docker hoá |
| [2026-09-04](2026-09-04-vast-image-pull-stuck.md) | M1 (chưa xong) | Thuê lại vast.ai để build Docker + đo FPS multi-stream — image pull kẹt ~28 phút, huỷ instance giữa chừng |
| [2026-09-04 (2)](2026-09-04-2-m1-fps-multistream-va-m4-tracklet-gallery.md) | M1 xong + M4 | Đóng phần treo M1: Dockerfile tái lập được, 4 luồng 189 FPS/luồng, fixture thật qua Redis. Khởi động M4: tracklet + gallery + topology |
| [2026-09-04 (3)](2026-09-04-3-m4-affinity-associator.md) | M4 | affinity + associator: engine liên kết chạy trọn vòng. Sweep max_cost (vùng đúng rất hẹp [0.30, 0.37]). Chuyển chỗ chạy test sang ut-hpc (Python 3.10.12) |
| [2026-09-04 (4)](2026-09-04-4-wildtrack-fixture-va-danh-gia-that.md) | M4 | Fixture WildTrack thật (7 camera, OSNet ONNX) + lần đầu đo engine trên người thật: max_cost 0.30 sai hẳn trên embedding thật, checkpoint domain-generalization hơn Market-1501 25% |
| [2026-09-04 (5)](2026-09-04-5-homography-va-rang-buoc-hinh-hoc.md) | M4 | `homography.py` + hiệu chỉnh WildTrack: hình học mạnh hơn ngoại hình ~20 lần (trần F1 0.929 vs 0.053). Sửa 3 lỗi thiết kế lộ ra khi đo — F1 0.014 → 0.736 |
