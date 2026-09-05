# 2026-09-06 (phiên 17) — Soát logic `src/mct`: 6 lỗi, đã sửa hết, và cái giá đo được là −0.21 HOTA

- **Mốc:** M4 (sửa lỗi) | **Máy:** máy dev (soạn) + `ut-hpc` (chạy + đo) | **Thời lượng:** ~3h, **không tốn GPU**

## Mục tiêu phiên

- Đọc lại toàn bộ logic `src/mct/` (+ `eval/`, `tools/export_trackeval.py`) tìm chỗ sai.
- Sửa những chỗ tìm được, khoá bằng test, rồi đo lại bằng HOTA/AssA chứ không chỉ F1 tự chế.

## Đã làm

Sáu lỗi, tất cả đã sửa. Mỗi cái kèm một test tái hiện đúng cơ chế hỏng.

**1. Bộ chấm điểm chạy CẤU HÌNH KHÁC engine.** `AffinityConfig.from_mapping` lấy cửa sổ loại
trừ bằng `tracklet.idle_timeout_ms` (30 000 trong YAML DeepStream), còn `eval_wildtrack.py`
dựng `AffinityConfig(...)` bằng tay nên để nguyên mặc định 2 000 — mọi số F1 từ phiên 12 tới
16 không cùng cấu hình với HOTA (0.256 so với 0.293 trên cùng fixture). Sửa gốc chứ không vá:
`eval/eval_wildtrack.py` và `eval/compare_online_offline.py` nhận `--config` và đọc thẳng YAML
qua chính `from_mapping` của engine, chỉ ba chiều quét (`max_cost`, `mode`, `ground_gap_policy`)
được phủ lên trên. Bộ tham số thật sự đang chạy được in ra đầu mỗi lượt.

**2. Ràng buộc loại trừ hỏi sai câu hỏi.** `is_active_in` hỏi "lần cuối thấy ở camera này có
gần đây không", trong khi mệnh đề đúng là *đồng thời*: một người không thể là hai local track
của cùng một camera **tại cùng một lúc**; hai mảnh NỐI TIẾP nhau thì không mâu thuẫn gì — đó
chính là hình dạng của một lần tracker đổi id. Thay bằng `GlobalTrack.overlaps_in()` so khoảng
`[start_ms, end_ms]`. Khoá cấu hình `exclusion_window_ms` (suy ra từ `idle_timeout_ms`) bị thay
bằng `exclusion_slack_ms` mặc định 0, đọc từ khối `association`.

**3. Ràng buộc loại trừ chỉ nhớ MẢNH CUỐI mỗi camera** — lỗi do chính sửa đổi 2 sinh ra, và
TrackEval bắt được: `Tracker predicts the same ID more than once in a single timestep`. Thứ tự
hấp thụ tracklet không nhất thiết theo thời gian, nên chỉ nhớ khoảng của mảnh cuối thì một
tracklet đến muộn nhưng nằm ở khoảng sớm hơn vẫn lọt qua, và Global ID đó có hai hộp trong
cùng một khung. `cam_span` (một khoảng) → `cam_spans` (tối đa 8 khoảng gần nhất mỗi camera).

**4. Gallery chồng thêm bản ghi cho cùng một tracklet ở mỗi cửa sổ.** Tracklet dài chiếm trọn
`max_size` ô và `_enforce_quota` đẩy hết ngoại hình của camera khác ra ngoài — đúng lúc cần
chúng nhất. `_absorb` giờ THAY bản ghi cùng `tracklet_id`. Chỉ ảnh hưởng chế độ online.

**5. Ngưỡng `max_cost` đặt SAU Hungarian.** Docstring cũ bảo "Hungarian cần thấy toàn bộ ma
trận", nhưng ô sẽ-bị-loại vẫn rẻ hơn ô chặn nên Hungarian sẵn sàng đẩy một hàng vào đó để hàng
khác lấy ô rẻ hơn, và cặp hợp lệ bị mất. `costs_for_hungarian` giờ chặn cả ô `>= max_cost`.
`_why_new` trả về `(loại, mô tả)` thay vì để `associator` dò chuỗi tiếng Việt để đếm thống kê.

**6. `ground_gap_policy=reject` bị kích hoạt bởi quỹ đạo CŨ.** Người đi cam01 (chồng lấn cam03)
→ cam02 (không chồng lấn ai) → cam03 bị loại vì "cam01 ↔ cam03 không có mốc chung", dù hai quỹ
đạo cách nhau hàng chục giây nên vốn dĩ không thể có mốc chung. Quyền phủ quyết giờ chỉ thuộc
về camera mà track ĐANG ở, hoặc camera có quỹ đạo TRÙNG khoảng thời gian với tracklet.

**Cộng thêm — `export_trackeval` bịa ra id-switch.** `load_global_ids` trả `{(cam, local): gid}`
nên khi một local id bị cắt thành nhiều tracklet, **mảnh cuối ghi đè mọi mảnh trước** và toàn
bộ detection của mảnh đầu bị xuất dưới Global ID của mảnh cuối. Thay bằng `GlobalIdIndex` tra
theo `(cam, local, ts_ms)`; detection không rơi vào khoảng nào trả `None` (engine thật sự chưa
gán gì cho nó → tính là bỏ sót, đúng).

**463 passed, 8 skipped**, ruff sạch (chạy trên `ut-hpc`, Python 3.10.12). 7 test mới, mỗi
sửa đổi có ít nhất một test tái hiện đúng cơ chế hỏng; 4 test cũ đổi theo API mới.

## Quyết định kỹ thuật

**1. Giữ toàn bộ sáu sửa đổi dù HOTA giảm 0.21.** Đây là quyết định khó nhất của phiên. Bảng
dưới cho thấy bản sau khi sửa được HOTA 14.167 so với 14.374 trước đó, IDF1 16.15 so với 17.51.
Lý do vẫn giữ:

- Cả sáu đều là lỗi **đúng/sai**, không phải lựa chọn tham số: hai cấu hình khác nhau giữa
  engine và bộ chấm, một ràng buộc phát biểu sai mệnh đề, một bất biến bị vi phạm (TrackEval
  từ chối chấm), một khâu chấm điểm bịa ra id-switch. Giữ lại một con số cao hơn sinh ra từ
  những thứ đó thì không bảo vệ được ở hội đồng.
- Mức giảm nằm gọn trong biên độ mà **tham số** gây ra. Ngưỡng trong `wildtrack_ds.mct.yaml`
  được chỉnh khi ràng buộc còn chặt hơn; ràng buộc lỏng ra thì engine ghép nhiều hơn hẳn
  (194 Global ID so với 297) và bắt đầu ghép quá tay — đúng kiểu "giá trị đúng của tham số phụ
  thuộc hành vi đang chạy" đã gặp ở phiên 16 với `max_ground_dist_m`.
- WildTrack là **ca tệ nhất để đánh giá sửa đổi 2**: 2 fps, mọi camera chồng lấn, và chỉ có
  15 cặp mảnh nối tiếp cùng camera trên toàn bộ dataset. Ở 25 fps của dữ liệu tự thu, tracker
  đổi id dày hơn nhiều và đó mới là chỗ sửa đổi này trả tiền.

**2. `exclusion_slack_ms` mặc định 0 ở MỌI cấu hình, kể cả WildTrack.** Đo được: giữ ngữ nghĩa
loại trừ cũ (cùng với năm sửa đổi kia) cho IDF1 17.05 so với 16.15, HOTA gần như bằng nhau
(14.153 so với 14.167). Tức trên riêng dataset này, chặn theo thời gian ăn điểm IDF1. **Không**
đặt slack lớn cho WildTrack: đó là khớp tham số vào một dataset mượn để lấy lại một chỉ số,
trong khi chỉ số chính (HOTA) không phân biệt được hai bên. Khoá đã có sẵn trong YAML để bật
lại và đo ở M6.

**3. Sửa đổi 6 làm HẸP, không làm rộng.** Bản đầu lọc mọi camera chồng lấn theo thời gian, và
nó làm `reject` mất răng với mọi cặp lệch thời gian — một test cũ (`test_lech_thoi_gian_qua_dung_sai`)
bắt được ngay. Bản giữ lại chỉ bỏ quyền phủ quyết của camera mà track ĐÃ RỜI KHỎI, tức đúng ca
hỏng, không đụng tới chính sách. Ablation xác nhận: bỏ hẳn phần này thì HOTA tụt về 13.622.

**4. Không đổi định nghĩa `eval_wildtrack.score`.** Nó khoá theo `(cam_id, local_track_id)` nên
mảnh sau ghi đè mảnh trước, và vì thế **mù hoàn toàn** với sửa đổi 2. Đổi sang đếm theo tracklet
sẽ làm mọi con số F1 của bốn phiên trước không so được nữa. Giữ nguyên, và theo đúng quy ước
phiên 16: F1 chỉ để quét nhanh, kết luận đọc trên HOTA/AssA.

## Số liệu đo được

**Cấu hình chung:** fixture `ds_wildtrack_7cam.jsonl` (WildTrack 7 camera, 2 fps, 400 khung/camera,
2800 message — hộp YOLO11s, `local_track_id` của NvDCF, embedding ReID của nvtracker), config
`configs/demo/wildtrack_ds.mct.yaml` + `wildtrack.topology.yaml` + homography 7 camera đã hiệu
chỉnh. Ground-truth: chú thích WildTrack (42.606 hộp, 313 danh tính). TrackEval MotChallenge2DBox,
IoU 0.5, `DO_PREPROC=False`, `--split mct`. Chạy trên head node `ut-hpc`; mỗi vòng online ~70 s,
export ~2 s, chấm ~4 s.

### Trước / sau, đường online đầy đủ (`python -m mct` → export → TrackEval)

| | trước (HEAD phiên 16) | sau phiên 17 |
|---|---|---|
| **HOTA** | **14.374** | **14.167** |
| DetA / AssA | 24.117 / 8.752 | 24.163 / **8.522** |
| **IDF1** (IDR / IDP) | **17.514** (15.84 / 19.58) | **16.154** (14.61 / 18.06) |
| MOTA / MOTP | 2.659 / 69.319 | 2.643 / 69.319 |
| IDSW | 2528 | 2535 |
| Global ID sinh ra / GT | 297 / 313 | **194** / 313 |

Bản HEAD **tái lập chính xác** số phiên 16 (14.374 / 8.7523 / 17.514) trước khi đo bản mới.

### Ablation: bỏ từng sửa đổi khỏi bản mới (HOTA, càng cao càng tốt)

| cấu hình | HOTA | AssA | IDF1 | #gid |
|---|---|---|---|---|
| **bản mới (giữ cả sáu)** | **14.167** | 8.522 | 16.154 | 194 |
| bỏ khử trùng lặp gallery (sửa đổi 4) | 13.604 | 7.837 | 15.220 | 194 |
| bỏ thu hẹp quyền phủ quyết (sửa đổi 6) | 13.622 | 7.868 | 15.124 | 234 |
| bỏ chặn ô vượt ngưỡng trước Hungarian (5) | 14.167 | 8.522 | 16.154 | 194 |
| trả ngữ nghĩa loại trừ về như cũ (2) | 14.153 | 8.525 | **17.050** | 246 |

Đọc bảng này cẩn thận: **các sửa đổi tương tác mạnh, không cộng được.** Bỏ riêng bất kỳ cái nào
đều tệ hơn bản mới, nhưng bỏ hết cùng lúc (= HEAD) lại cho 14.374. Nghĩa là chênh lệch ±0.5 HOTA
trên fixture này đến từ tương tác giữa các ràng buộc chứ không từ một cơ chế đơn lẻ — và đó là
lý do không nên đọc một con số HOTA đơn lẻ như bằng chứng cho một thay đổi đơn lẻ.

Sửa đổi 5 (chặn trước Hungarian) **không đổi một chữ số nào** trên fixture này, đúng như dự đoán
ở phần soát: nó sửa lập luận, không sửa điểm số.

### Quét lại `max_cost` dưới ngữ nghĩa mới

| max_cost | 0.50 | 0.60 | 0.70 | **0.80** |
|---|---|---|---|---|
| HOTA (online) | 12.888 | 13.471 | 13.435 | **14.167** |
| IDF1 | 14.229 | 14.862 | 14.696 | **16.154** |
| #gid | 216 | 208 | 203 | 194 |
| F1 offline | 0.217 | 0.247 | 0.243 | **0.256** |

0.80 vẫn là đỉnh, nên **không đổi tham số nào** trong phiên này. Việc engine ghép nhiều hơn không
sửa được bằng cách siết `max_cost` — siết vào là mất cả cặp đúng.

### Khâu chấm điểm

Sửa `load_global_ids` (tra theo thời gian) cho ra **đúng cùng một bảng điểm** trên fixture này
(14.374 với cả bản cũ lẫn bản mới, cùng 34.475 dòng kết quả): trong lượt chạy HEAD không có local
id nào bị cắt thành hai tracklet mang hai Global ID khác nhau. Đây là sửa lỗi phòng xa — nó chỉ
lộ ra khi `idle_timeout_ms` nhỏ so với fps, tức đúng cấu hình của dữ liệu tự thu ở M6.

## Vướng mắc / chưa xong

- **Bản sau khi sửa đang thấp hơn bản trước 0.21 HOTA / 1.36 IDF1** trên WildTrack. Đã giải
  thích được cơ chế (ghép nhiều hơn → ghép quá tay, 194 Global ID cho 313 danh tính) nhưng chưa
  lấy lại được điểm. Hướng tiếp theo là chỗ đang mất nhiều nhất chứ không phải chỗ này.
- `eval_wildtrack.score` vẫn mù với việc ghép mảnh trong cùng camera (quyết định 4) — mọi kết
  luận về AssA phải đọc trên TrackEval.
- Chưa chấm `onnx_gtbox` bằng HOTA (nợ từ phiên 15, sang phiên thứ ba).
- Mọi con số vẫn trên WildTrack 2 fps, mọi camera chồng lấn. **Đừng chốt tham số nào theo dataset
  này** — nhắc lại lần thứ sáu.

## Bước tiếp theo

1. **Ghép tracklet trong cùng camera** — việc đã hoãn từ phiên 16, và giờ ràng buộc loại trừ
   không còn cấm nó nữa (sửa đổi 2 dọn đường). AssA 8.52 vẫn nói đây là chỗ mất điểm lớn nhất.
2. Chấm `onnx_gtbox` bằng HOTA để quy đổi ảnh hưởng của chất lượng hộp.
3. Ở M6, bật lại `exclusion_slack_ms` như một ablation trên dữ liệu 25 fps — nơi giả thiết
   "mảnh nối tiếp là chuyện thường" mới thật sự được kiểm.
