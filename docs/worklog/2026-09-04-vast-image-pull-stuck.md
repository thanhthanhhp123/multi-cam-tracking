# 2026-09-04 — Thuê lại vast.ai để build/test M1: image pull bị kẹt, huỷ giữa chừng

- **Mốc:** M1 (tuần 3–4, phần còn treo) | **Máy:** máy dev (Windows) + `vast-gpu` | **Thời lượng:** ~45 phút (chủ yếu chờ)

## Mục tiêu phiên

- Thuê lại instance Vast.ai (instance cũ `49751272` đã bị huỷ/hết hạn từ trước).
- Build thử `docker/deepstream.Dockerfile` trên máy GPU thật để xác nhận tái lập được
  (bước tiếp theo còn treo từ worklog 2026-09-03 phiên 3).
- Đo FPS multi-stream (proxy: lặp lại sample video vài lần trong `streams.yaml`, vì
  `tools/rtsp_sim.py` chưa viết).

## Đã làm

- Xác nhận instance cũ `49751272` đã không còn tồn tại (`vastai show instances` → 0 instances) —
  không tốn phí thêm, không cần huỷ tay.
- Tìm offer mới qua `vastai search offers`: chọn offer `49509921` (RTX 3090 24GB, driver
  580.159.03, $0.1127/h GPU, tổng $0.1789/h kèm disk) — rẻ hơn lần thuê trước.
- `vastai create instance 49509921 --image nvcr.io/nvidia/deepstream:7.1-triton-multiarch
  --disk 60 --ssh --direct` → tạo instance `49767510` thành công.
- Poll trạng thái (`vastai show instance 49767510 --raw`) liên tục ~28 phút — `actual_status`
  không thoát khỏi `loading`, `status_msg` đứng yên ở `"762bedf4b1b7: Already exists"` suốt
  từ phút ~10 tới phút ~28 (không đổi digest/layer nào thêm) → dấu hiệu kẹt thật, không phải
  đang tải chậm bình thường.
- Theo chỉ dẫn người dùng ("nếu kẹt bất thường thì huỷ hết"): `vastai destroy instance
  49767510 -y` → xác nhận `0 instances`, dừng tính phí.
- **Chưa SSH vào được lần nào** — instance không bao giờ đạt trạng thái `running`, nên chưa
  cập nhật `~/.ssh/config`, chưa build Docker, chưa đo FPS.

## Quyết định kỹ thuật

**Huỷ instance kẹt thay vì tiếp tục chờ.** `status_msg` đứng yên nhiều phút liên tục là tín
hiệu đáng tin hơn thời gian trôi qua đơn thuần (image DS 7.1 nặng, biết trước sẽ mất vài phút
thật — nhưng không tiến triển gì suốt 18 phút thì không phải vậy). Không có cách nào chẩn đoán
sâu hơn từ phía client (Vast.ai không lộ log pull chi tiết qua CLI `show instance`) — huỷ và
thuê lại (khả năng cao là vấn đề của riêng host `142739` đó) rẻ hơn tiếp tục chờ không giới hạn.

## Số liệu đo được

—  (không build/chạy được gì trên GPU trong phiên này)

## Vướng mắc / chưa xong

- Vẫn treo y hệt worklog 2026-09-03 phiên 3: chưa build thử `deepstream.Dockerfile` trên máy
  GPU thật, chưa đo FPS multi-stream/live-source.
- Chưa rõ nguyên nhân kẹt (host cụ thể, mạng route tới registry NVIDIA, hay vấn đề phía
  Vast.ai) — nếu lặp lại ở phiên sau, thử offer ở host/khu vực khác (offer lần trước —
  Bỉ, lần này — US — cả hai đều từng vướng vấn đề tốc độ/kẹt khác nhau, chưa đủ dữ liệu
  để kết luận pattern).
- Số dư tài khoản Vast.ai: $7.87 credit (kiểm tra 2026-09-03/04) — đủ cho vài lần thuê ngắn nữa.

## Bước tiếp theo

1. Thuê lại instance mới (ưu tiên offer khác host/region nếu có), poll tối đa ~10–12 phút —
   nếu `status_msg` không tiến triển sau ~10 phút thì huỷ sớm hơn lần này thay vì chờ 28 phút.
2. Build `docker/deepstream.Dockerfile` thật trên instance mới, xác nhận tái lập được.
3. Đo FPS multi-stream (lặp `sample_1080p_h264.mp4` 3–4 lần trong `streams.yaml` làm proxy)
   trước khi có `tools/rtsp_sim.py` thật.
4. Song song, không phụ thuộc cụm/GPU: `src/mct/tracklet.py` (bước tiếp theo không đổi từ
   worklog 2026-09-01 / 2026-09-03 phiên 1).
