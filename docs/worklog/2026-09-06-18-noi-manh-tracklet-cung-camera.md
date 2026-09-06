# 2026-09-06 (phiên 18) — Nối mảnh tracklet cùng camera: HOTA 14.4 → 16.2, AssA 8.8 → 11.1

- **Mốc:** M4 (đóng góp chính) | **Máy:** máy dev (soạn) + `ut-hpc` (chạy + đo) | **Thời lượng:** ~1.5h, **không tốn GPU**

## Mục tiêu phiên

- Việc số 1 của phiên 17 (và đã hoãn từ phiên 16): **ghép tracklet trong cùng camera**.
  AssA 8.52 nói đây là chỗ mất điểm lớn nhất, và phiên 17 vừa dọn đường bằng cách sửa ràng
  buộc loại trừ (trước đó nó cấm thẳng việc này).
- Lấy lại 0.21 HOTA đã mất ở phiên 17, nếu lập luận về nguyên nhân là đúng.

## Đã làm

- **`affinity._same_camera_term()`** (mới) — nối hai mảnh tracklet của CÙNG một camera bằng
  liên tục vị trí + tốc độ đi bộ: lấy điểm cuối của mảnh trước và điểm đầu của mảnh sau (đã
  chiếu về mét qua homography của chính camera đó), loại nếu khoảng cách vượt ngân sách
  `max_ground_dist_m + max_speed_m_s · Δt`, ngược lại cộng `λ · d` vào chi phí. Chỉ xét hai
  mảnh RỜI nhau về thời gian; chồng thời gian đã bị ràng buộc loại trừ chặn trước đó.
- Khoá `association.same_camera_stitch` (mặc định `true`) trong `configs/mct.yaml` và
  `configs/demo/wildtrack_ds.mct.yaml` — có cờ tắt nên ablation chạy được mà không sửa code.
- Quét lại `max_cost` vì thang chi phí đổi: `0.80 → 0.90` trong cấu hình WildTrack DeepStream.
- **467 passed, 8 skipped**, ruff sạch. 4 test mới: nối được mảnh liên tục, loại mảnh "nhảy"
  60 m trong 5 giây, tắt cờ thì trả về đúng hành vi cũ, và không nói gì về cặp chồng thời gian.

## Quyết định kỹ thuật

**1. Nối mảnh là một THÀNH PHẦN CHI PHÍ, không phải một bước tiền xử lý riêng.** Phương án
kia — gom mảnh thành tracklet dài trước rồi mới đưa vào liên kết xuyên camera — cần một tầng
trạng thái mới và một vòng Hungarian riêng. Không chọn, vì `_match_one_camera` vốn đã ghép
một-một giữa tracklet và GlobalTrack **trong phạm vi một camera**, tức khung sẵn có đã đúng
hình dạng bài toán; chỉ thiếu đúng một tín hiệu. Thêm 40 dòng vào chỗ đã có thay vì một tầng
mới là lựa chọn rẻ hơn nhiều và ablation được bằng một cờ YAML.

**2. Vì sao tín hiệu này mạnh hơn mọi thứ hệ thống đang dùng.** Hai mảnh nằm trong **cùng một
mặt phẳng ảnh**, nên so vị trí không đi qua hiệu chỉnh chéo giữa hai camera: sai số duy nhất
là sai số điểm chân (RMS 0.768 m, phiên 16), không cộng thêm sai số ghép camera. So với ngoại
hình xuyên camera — thứ mà phiên 11 đo được là cosine giữa hai người KHÁC nhau đã ~0.69 — đây
là bằng chứng tách người tốt hơn hẳn, và nó gần như miễn phí vì homography đã có sẵn.

**3. Đây chính là ràng buộc thay thế mà phiên 17 còn thiếu.** Ràng buộc loại trừ cũ chặn mọi
tracklet "vừa thấy ở camera này gần đây" — sai về mệnh đề, nhưng nó vô tình cũng chặn việc
ghép nhầm hai NGƯỜI KHÁC NHAU ở cùng một camera. Phiên 17 sửa mệnh đề cho đúng và bỏ trống
chỗ lọc đó, nên engine ghép quá tay (194 Global ID cho 313 danh tính, HOTA tụt 0.21). Ràng
buộc đúng đắn để lấp vào không phải "vừa mới thấy" mà là "người ta có kịp đi từ đó tới đây
không". Kết quả xác nhận lập luận: phiên 17 + nối mảnh cho HOTA **16.210**, cao hơn cả bản
trước phiên 17 (14.374) lẫn bản sau (14.167).

**4. `max_cost` 0.80 → 0.90, và con số này KHÔNG phải đỉnh HOTA.** Thành phần mới cộng
`λ · d` vào mọi cặp cùng camera nên thang chi phí đổi; giữ ngưỡng cũ thì 381 cặp bị ngưỡng
loại và engine tách quá tay (468 Global ID). Quét lại: HOTA đi ngang trong cả dải 0.90–1.40
(biên độ 0.39 và **không đơn điệu** — 1.00 tụt dưới 0.90 rồi 1.20 vọt lên), nên 1.20 là đỉnh
nhọn của nhiễu chứ không phải một điểm làm việc. Chọn **0.90**: IDF1 cao nhất (20.918 so với
20.098 ở 1.20), HOTA nằm trong cao nguyên, và 433 > 313 danh tính nghĩa là nghiêng về TÁCH
thay vì GỘP — gộp sai thì không có đường quay lại, tách thì còn sửa được ở bước sau.

**5. Đối chứng bắt buộc: mức tăng KHÔNG phải do nới ngưỡng.** Tắt nối mảnh rồi nới ngưỡng y
hệt cho kết quả **tệ đi** (0.80 → 14.167, 0.90 → 14.120, 1.20 → 13.694). Nếu bỏ qua phép đối
chứng này thì toàn bộ phiên có thể bị đọc thành "chỉnh một tham số".

## Số liệu đo được

**Cấu hình chung:** fixture `ds_wildtrack_7cam.jsonl` (WildTrack 7 camera, 2 fps, 2800 message
— hộp YOLO11s, `local_track_id` của NvDCF, embedding ReID của nvtracker), config
`configs/demo/wildtrack_ds.mct.yaml` + `wildtrack.topology.yaml` + homography 7 camera. Ground
truth: chú thích WildTrack (42.606 hộp, 313 danh tính). Đường đo là **online đầy đủ**
(`python -m mct` → `export_trackeval` → TrackEval MotChallenge2DBox, IoU 0.5, `DO_PREPROC=False`).
Head node `ut-hpc`; mỗi vòng ~70 s + ~6 s chấm.

### Ba mốc

| | trước phiên 17 | sau phiên 17 | **phiên 18** |
|---|---|---|---|
| **HOTA** | 14.374 | 14.167 | **16.210** |
| DetA / **AssA** | 24.117 / 8.752 | 24.163 / 8.522 | 24.123 / **11.131** |
| **IDF1** (IDR / IDP) | 17.514 (15.84 / 19.58) | 16.154 (14.61 / 18.06) | **20.918** (18.92 / 23.39) |
| Global ID / GT | 297 / 313 | 194 / 313 | 433 / 313 |

**AssA +27% so với bản trước phiên 17** (8.752 → 11.131) trong khi DetA đứng yên (24.12) —
cùng một tập detection, nên toàn bộ mức tăng là của bước liên kết. Đây là con số của đóng góp
chính, đo bằng thước đo chuẩn.

### Quét `max_cost` (nối mảnh BẬT)

| max_cost | 0.80 | **0.90** | 1.00 | 1.10 | 1.20 | 1.30 | 1.40 |
|---|---|---|---|---|---|---|---|
| HOTA | 15.194 | **16.210** | 16.144 | 16.350 | *16.530* | 16.230 | 16.236 |
| AssA | 9.807 | 11.131 | 11.043 | 11.404 | *11.680* | 11.227 | 11.233 |
| IDF1 | 19.206 | **20.918** | 20.075 | 19.681 | 20.098 | 19.489 | 19.496 |
| #gid | 468 | 433 | 349 | 216 | 176 | 169 | 167 |

### Đối chứng: nới ngưỡng mà KHÔNG nối mảnh

| max_cost | 0.80 | 0.90 | 1.20 |
|---|---|---|---|
| HOTA | 14.167 | 14.120 | 13.694 |
| #gid | 194 | 191 | 186 |

Ngưỡng lỏng hơn mà không có ràng buộc hình học thay thế thì chỉ ghép bừa thêm.

## Vướng mắc / chưa xong

- **433 Global ID cho 313 danh tính** — giờ engine nghiêng về tách. Chưa rõ phần dư là mảnh
  của cùng một người bị bỏ sót hay là người thật mà detector chỉ thấy thoáng qua; cần một
  phép đếm theo danh tính chứ không chỉ tổng số.
- Nối mảnh **cần homography của camera đó**. Camera chưa hiệu chỉnh thì thành phần này im
  lặng bỏ qua — đúng thiết kế, nhưng nghĩa là uplift ở trên chỉ có khi đã hiệu chỉnh xong.
  Chưa có phương án dự phòng bằng toạ độ ảnh (chuẩn hoá theo chiều cao bbox).
- Ngân sách dùng chung `max_ground_dist_m` với thành phần xuyên camera, trong khi hai chỗ đo
  hai loại sai số khác nhau (cùng camera không có sai số ghép chéo). Tách thành hai khoá
  riêng là việc nên làm khi có dữ liệu M6 để đo.
- Mọi con số vẫn trên WildTrack 2 fps, mọi camera chồng lấn. **Đừng chốt tham số nào theo
  dataset này** — nhắc lại lần thứ bảy.
- Nợ cũ chưa trả: chấm `onnx_gtbox` bằng HOTA (từ phiên 15, nay sang phiên thứ tư).

## Bước tiếp theo

1. Đếm phần dư của 433 Global ID theo danh tính: bao nhiêu người bị chẻ, chẻ thành mấy mảnh.
   Đây là thứ quyết định có cần thêm một vòng ghép nữa hay không.
2. Chấm `onnx_gtbox` bằng HOTA để quy đổi ảnh hưởng của chất lượng hộp.
3. Tách `max_ground_dist_m` thành hai khoá (cùng camera / xuyên camera) khi có dữ liệu tự thu.
