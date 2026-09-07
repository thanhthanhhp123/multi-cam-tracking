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
| [2026-09-04 (5)](2026-09-04-5-homography-va-rang-buoc-hinh-hoc.md) | M4 | `homography.py` + vòng online (`python -m mct`, SQLite). Hình học mạnh hơn ngoại hình ~20 lần; sửa 3 lỗi thiết kế lộ ra khi đo — F1 0.014 → 0.929. **Online tốt hơn offline** |
| [2026-09-04 (6)](2026-09-04-6-m5-dashboard.md) | M5 | Dashboard realtime: bản đồ mặt phẳng mặt đất (vùng phủ camera suy ra từ homography), WebSocket, tra cứu hành trình. Chạy thật cả chuỗi Redis → engine → dashboard |
| [2026-09-04 (7)](2026-09-04-7-m2-detector-pretrained.md) | M2 xong + M3 | Bỏ fine-tune ở M2/M3 (weight pretrained đã train trên chính COCO/Market-1501/MSMT17). Lọc lớp person tại nvinfer thay cho head 1 lớp. Fine-tune dời sang M6 như ablation trên dữ liệu tự thu |
| [2026-09-04 (8)](2026-09-04-8-m3-reid-pretrained.md) | M3 | ReID bằng OSNet pretrained (bản DG): đường (A) qua nvtracker, `reid_meta.py` gom phần phụ thuộc phiên bản DeepStream, cặp config đối chứng có/không ReID. Chưa chạy trên phần cứng |
| [2026-09-04 (9)](2026-09-04-9-vast-xac-minh-m2-m3.md) | M2+M3 xong, M5 | Xác minh trên RTX 3090: đường (A) chạy, 100% detection có embedding 512-d. ReID tốn 9.4% FPS; lọc lớp đáng +66% **khi có ReID**. VRAM 1567 MiB. Độ trễ trung vị 40 ms, p90 2.1 s. Sửa lỗi camera lạ giết engine |
| [2026-09-04 (10)](2026-09-04-10-wildtrack-qua-deepstream.md) | M3 | Đưa WildTrack qua DeepStream — phần không cần GPU: bộ đóng ảnh thành video (hợp đồng khung thứ i = chú thích thứ i), bộ gán ground-truth bằng IoU + Hungarian + bỏ phiếu, cấu hình 7 luồng. Phát hiện node `ut-hpc` không đồng nhất (3/6 node thiếu libx264) |
| [2026-09-04 (11)](2026-09-04-11-wildtrack-qua-deepstream-do-that.md) | M3 xong | WildTrack chạy thật qua DeepStream trên Tesla T4: ngưỡng ngoại hình KHÔNG chuyển được (0.094 → 0.227), nhưng embedding DeepStream lại tách người TỐT HƠN. Điểm nghẽn thật là detector recall 44.9% + 53% track lẫn danh tính → F1 0.752 (SCT lý tưởng) tụt còn 0.170. Tracker 1080p: gấp đôi VRAM, không được gì. 2/3 máy thuê hỏng NVDEC |
| [2026-09-04 (12)](2026-09-04-12-do-vo-tracklet-va-tran-recall.md) | M4 | `diagnose_tracklets.py`: đo trần recall của ràng buộc hình học. Hai tham số đặt sai — `idle_timeout_ms` phụ thuộc frame rate (tự tạo 1.6× độ vỡ), `max_ground_dist_m` chỉnh trên bbox GT nên quá chặt với hộp detector. Sửa cả hai: F1 0.170 → 0.224. Nút thắt chuyển sang ngoại hình |
| [2026-09-05 (13)](2026-09-05-13-tach-bien-chat-luong-hop-va-duong-trich.md) | M4 | Tách hai biến bằng hai fixture chỉ khác hộp: hộp detector KHÔNG làm hỏng ngoại hình (trần 0.381 vs 0.379) mà làm hỏng hình học (d_ground 0.74 → 0.21 m, F1 +22%). Tiền xử lý ReID của DeepStream đáng −14% trần nhưng ~0 ở F1 — không đáng sửa |
| [2026-09-05 (14)](2026-09-05-14-bo-danh-gia-trackeval.md) | M6 | Bộ đánh giá chuẩn MOT/TrackEval + đọc chú thích CVAT vào git (7 file). Test cho `cvat_to_mot` (19), sửa hai lỗ im lặng: `visibility` không sống sót qua `write_gt`, hộp thiếu toạ độ thành NaN. 430 passed, ruff sạch. Chưa chạy trên dữ liệu thật |
| [2026-09-05 (15)](2026-09-05-15-hota-idf1-dau-tien.md) | M6 | **Bộ HOTA/IDF1/MOTA đầu tiên**: HOTA 94.7 / IDF1 94.0 ở cận trên (SCT lý tưởng) so với 14.2 / 17.3 trên pipeline thật. DetA gần như không đổi (24.3 → 24.1) trong khi AssA rơi 33.9 → 8.6 — toàn bộ chênh lệch là giá của bước liên kết. GT phải đến từ nguồn ĐỘC LẬP với kết quả (`--gt-fixture`), nếu không TrackEval từ chối chấm |
| [2026-09-05 (16)](2026-09-05-16-lam-muot-diem-chan-va-quy-dao-bi-tia.md) | M4 | Làm mượt điểm chân: **bác bỏ** (F1 0.224 → 0.229, trần recall không đổi) — `_synchronized_distance` vốn đã là bộ lọc thời gian, và 36% năng lượng sai số điểm chân là ĐỘ CHỆCH không bộ lọc nào khử được. Thứ thật sự mất điểm là `ground_path_max_points: 64` tự tỉa thưa quỹ đạo: nới lên 256 + siết `max_ground_dist_m` 5.0 → 2.0 cho F1 0.256, HOTA 14.245 → 14.374 |
| [2026-09-06 (17)](2026-09-06-17-soat-logic-mct.md) | M4 | Soát logic `src/mct` và **sửa 6 lỗi**: bộ chấm điểm chạy cấu hình khác engine; ràng buộc loại trừ hỏi "vừa thấy gần đây" thay vì "trùng thời gian" nên cấm luôn việc nối lại mảnh do tracker đổi id; một Global ID ôm hai hộp trong một khung (TrackEval từ chối chấm); gallery chồng bản ghi mỗi cửa sổ; ngưỡng đặt sau Hungarian; `reject` bị kích hoạt bởi quỹ đạo cũ. Cộng thêm: `export_trackeval` bịa ra id-switch. Giá phải trả đo được: HOTA 14.374 → 14.167, và các sửa đổi TƯƠNG TÁC mạnh (bỏ riêng cái nào cũng tệ hơn giữ cả) |
| [2026-09-06 (18)](2026-09-06-18-noi-manh-tracklet-cung-camera.md) | M4 | **Nối mảnh tracklet cùng camera** bằng liên tục vị trí + tốc độ đi bộ (`same_camera_stitch`): HOTA 14.374 → **16.210**, AssA 8.752 → **11.131** (+27%), IDF1 17.514 → **20.918**, DetA đứng yên nên toàn bộ mức tăng là của bước liên kết. Đây cũng là ràng buộc thay thế mà phiên 17 còn thiếu. Đối chứng: nới ngưỡng mà không nối mảnh thì TỆ ĐI (14.167 → 13.694) |
| [2026-09-06 (19)](2026-09-06-19-toi-uu-engine-va-hang-doi-co-tran.md) | M4/M5 | Chỗ nghẽn không ở GPU mà ở engine: **36.8 → 151.9 msg/s (4.1 lần)** bằng chiếu quỹ đạo theo lô, cache sống qua nhiều cửa sổ, bỏ `np.clip`, và chỉ mục cho `find_by_tracklet`. HOTA/AssA/IDF1 trùng khít từng chữ số nên chứng minh được là tối ưu thuần tuý. Thêm `QueuedFramePublisher`: probe thôi gọi Redis đồng bộ trên luồng streaming, hàng đợi CÓ TRẦN, đầy thì bỏ khung cũ nhất và đếm |
| [2026-09-07 (20)](2026-09-07-20-phan-ra-phan-du-global-id.md) | M4 | **Phân rã 433 Global ID** bằng đẳng thức `n_gid = rác + danh tính phủ + mảnh dư − chồng do gộp`: **50.1% là RÁC** (không tracklet nào tra được nhãn), chỉ 70/287 phần dư là do vỡ. 54.8% danh tính liên kết ĐÚNG thành 1 ID; 43.7% khung mất vì vỡ so với 13.9% mất vì gộp — nên "nghiêng về tách" của phiên 18 đúng chiều nhưng sai lập luận (24.1% Global ID vẫn lẫn ≥2 người). 73% mảnh dư sinh ra vì `threshold`, tức nút thắt vẫn là NGOẠI HÌNH |
