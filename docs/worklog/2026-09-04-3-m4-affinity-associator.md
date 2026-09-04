# 2026-09-04 (phiên 3) — M4: affinity + associator, engine liên kết chạy trọn vòng

- **Mốc:** M4 (tuần 11–13) | **Máy:** máy dev (chỉ soạn code) + `ut-hpc` (chạy test) | **Thời lượng:** ~1.5h

## Mục tiêu phiên

- Viết nốt hai module còn lại của lõi engine liên kết: `affinity.py` (ma trận chi phí)
  và `associator.py` (Hungarian + gán Global ID).
- Chuyển chỗ chạy test ra khỏi máy dev theo yêu cầu người dùng.

## Đã làm

- `src/mct/affinity.py` — ma trận chi phí `1 − cosine`, mask `inf` cho mọi cặp bất khả thi
  (loại trừ cùng camera, ràng buộc thời gian di chuyển, thiếu embedding, GlobalTrack đã
  đóng), cộng `λ · d_ground` cho cặp camera chồng lấn. Mỗi ô bị loại **kèm lý do đọc được**.
- `src/mct/associator.py` — Hungarian (`scipy.optimize.linear_sum_assignment`) + ngưỡng
  `max_cost` + tạo Global ID mới. Có `run_offline()` và `assign_messages()` cho đường
  đánh giá trên fixture.
- Bổ sung `GlobalTrack.last_ground_point` (đầu vào thành phần homography).
- `eval/sweep_synthetic.py` — sweep `max_cost` × `similarity_mode` trên ba kịch bản độ khó.
- **Dựng môi trường test trên `ut-hpc`**: `~/mct/venv-test` (venv từ `python3` hệ thống =
  **Python 3.10.12**, đúng bằng phiên bản trong container DeepStream), cài numpy/scipy/
  msgpack/redis/PyYAML/python-dotenv/pytest/ruff/fastapi. Repo đồng bộ sang `~/mct/repo`.
  Toàn bộ test + lint từ nay chạy ở đây: **172 passed, 5 skipped**, `ruff` sạch.
- 36 test mới (`tests/test_affinity.py` 18, `tests/test_associator.py` 18).

## Quyết định kỹ thuật

**1. Tracklet đang chạy được cập nhật thẳng, KHÔNG đưa vào ma trận Hungarian.**
Hungarian tối ưu *tổng* chi phí, nên nó sẵn sàng hy sinh một cặp chi phí 0 để đổi lấy hai
cặp khác rẻ hơn một chút — tức cướp Global ID của một tracklet đang chạy ngon lành và làm
đứt danh tính giữa chừng. Tách phần cập nhật ra trước là cách rẻ nhất để chuyện đó không
xảy ra. Có test ghim (`test_hungarian_khong_cuop_id_cua_tracklet_dang_chay`).

**2. Ngưỡng `max_cost` áp SAU Hungarian, không phải trước.** Nếu mask trước, ma trận mất
thông tin về những cặp "hơi đắt nhưng là lựa chọn tốt nhất trong bối cảnh", và phép ghép
tối ưu không còn tối ưu. Chạy Hungarian trên toàn ma trận rồi mới bỏ cặp vượt ngưỡng.
Kèm theo đó, `inf` phải đổi thành hằng số hữu hạn lớn trước khi gọi `linear_sum_assignment`
(SciPy ném lỗi khi không tồn tại phép gán hoàn chỉnh) — giá trị thay thế không ảnh hưởng
kết quả vì mọi cặp dùng tới nó đều bị ngưỡng loại ngay sau đó.

**3. Online và offline dùng chung đúng một hàm `assign()`.** Khác nhau không nằm ở thuật
toán mà ở *thời điểm* tracklet được đưa vào: online đưa vào khi tracklet còn dang dở,
offline đưa vào khi tracklet đã đóng và có đủ đặc trưng. Chênh lệch kết quả giữa hai chế
độ **chính là cái giá của ràng buộc thời gian thực** — đúng con số cần cho chương 6, và
đo được bằng cách chạy cùng dữ liệu qua hai đường.

**4. Phần hình học chỉ khai qua giao thức `GroundMapper`, không nhúng homography vào
affinity.** `homography.py` chưa có; thay vì để trống hoặc đoán bừa khoảng cách, affinity
gọi một giao thức và **bỏ qua thành phần hình học khi cặp camera chưa được hiệu chỉnh**
(mapper trả `None`). Nhờ vậy test được bằng mapper giả, và ngày lắp homography thật không
phải sửa affinity.

**5. Chuyển toàn bộ test sang `ut-hpc` (venv Python 3.10.12).** Máy dev Windows chỉ có
Python 3.13 nên phải cài bằng `--ignore-requires-python`, và test ở đó không bao giờ bắt
được lỗi vi phạm quy tắc "nhắm Python 3.10" (CLAUDE.md §2 quy tắc 4). Head node `ut-hpc`
có sẵn Python 3.10.12 — **đúng bằng phiên bản trong container DeepStream 7.1** — nên đây
vừa là chỗ chạy test, vừa là chỗ *kiểm chứng* ràng buộc phiên bản. Chỉ dùng venv nhẹ
(không torch), chạy 1.3 s, không vi phạm quy tắc "không chạy gì nặng trên head node".

## Số liệu đo được

**Cấu hình:** `eval/sweep_synthetic.py`, fixture tổng hợp seed 42, `embed_dim=256`,
`intra_sim=0.80`, 15 fps, dwell 6 s, transit 8 s ±25%, miss_rate 0.04. Topology cam01↔cam02
`[3 s, 15 s]`. Tracklet: `min_frames=5`, `idle_timeout=2000 ms`, `topk_query=8`.
Chỉ số theo **cặp lượt xuất hiện** (không phải MOTA/IDF1 chuẩn): P = tỉ lệ cặp bị gán
chung ID mà đúng là một người; R = tỉ lệ cặp cùng người được gán chung ID.

Kịch bản **dễ** (3 người, `cross_cam_sim=0.75`, `inter_sim=0.65`, 6 lượt xuất hiện):

| max_cost | #Global ID | danh tính bị vỡ | ID gộp nhầm | P | R | F1 |
|---|---|---|---|---|---|---|
| 0.10–0.25 | 6 | 3 | 0 | 1.000 | 0.000 | 0.000 |
| **0.30** | **3** | **0** | **0** | **1.000** | **1.000** | **1.000** |
| 0.35 | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| 0.40–0.50 | 2 | 0 | 1 | 0.429 | 1.000 | 0.600 |

Kịch bản **khó** (6 người, `inter_sim=0.72` — đồ rất giống nhau): mọi ngưỡng ≥0.30 đều cho
P=0.600, R=1.000, F1=0.750 — cặp "mặc đồ giống nhau" **luôn bị gộp**, không ngưỡng nào cứu
được. Kịch bản **rất khó** (`cross_cam_sim=0.65 < inter_sim` của kịch bản trước): phải tới
0.40 mới bắt đầu khớp được gì, và khớp thì gộp luôn (F1=0.750); ở 0.30 thì không khớp gì cả.

**Ba kết luận rút ra:**

1. **Vùng chạy đúng của `max_cost` rất hẹp: `[0.30, ~0.37]`.** Giá trị 0.30 đang có trong
   `configs/mct.yaml` nằm **đúng ở mép dưới** — thấp hơn một chút là mất sạch match (R=0),
   cao hơn 0.40 là bắt đầu gộp người. Không đổi giá trị mặc định (nó đúng), nhưng đây là
   bằng chứng định lượng cho việc phải sweep lại ngưỡng này mỗi khi đổi model Re-ID.
2. **Khi ngoại hình không đủ tách người, không ngưỡng nào cứu được.** Ở `inter_sim=0.72`,
   mọi ngưỡng cho R=1 đều kéo theo P=0.6. Đây là lý do phải có ràng buộc không–thời gian —
   nhưng ở kịch bản này cả hai người đi cùng tuyến cam01→cam02 trong cùng cửa sổ transit
   nên topology cũng không phân biệt nổi. Kết luận cho chương 6: **giới hạn trên của hệ
   thống bị chặn bởi chất lượng Re-ID**, và dataset tự thu ở M6 nên cố tình có cặp người
   mặc đồ giống nhau đi *khác tuyến* để đo phần đóng góp riêng của topology.
3. **`similarity_mode: max` và `centroid` cho kết quả GIỐNG HỆT nhau trên fixture này** —
   vì mỗi GlobalTrack chỉ có đúng một embedding mỗi camera (2 lượt xuất hiện/người), nên
   max và trung bình EMA trùng nhau. Muốn so sánh hai chế độ phải có ≥3 camera hoặc có
   người quay lại camera cũ → chờ fixture WildTrack (7 camera) hoặc dataset M6. Ghi lại để
   khỏi kết luận nhầm rằng "hai chế độ tương đương".

## Vướng mắc / chưa xong

- **Chưa chạy trên dữ liệu thật có ground truth.** Fixture WildTrack chưa dựng được: cần
  annotation (nhỏ) + ảnh gốc (~13 GB) + OSNet ONNX, máy dev chưa có gì trong `data/` và
  `models/reid/`. Đây là việc đầu tiên của phiên sau.
- **`homography.py` chưa viết** — affinity đã chừa sẵn giao thức, nhưng chưa có cặp camera
  nào được hiệu chỉnh nên thành phần hình học chưa từng chạy với dữ liệu thật.
- **`mct/__main__.py` chưa có** — engine chưa nối vào Redis; hiện mới chạy được qua
  `assign_messages()` (offline, cho test và đánh giá). Chế độ online thật, `mct:global`,
  và `store.py` (SQLite) đều chưa làm.
- **Chưa đo chênh lệch online vs offline** — cần `__main__.py` mới đo được, đó mới là con
  số "giá của thời gian thực" cho chương 6.
- Fixture thật từ pipeline DeepStream (ghi hôm nay) **không có embedding** nên chỉ dùng
  kiểm tra vòng đời tracklet, chưa kiểm được phần liên kết.

## Bước tiếp theo

1. Dựng fixture WildTrack: `tools/fetch_wildtrack_annotations.py` + tải ảnh + OSNet ONNX
   (`tools/export_osnet_onnx.py`), rồi chạy `associator` trên đó — lần đầu có embedding
   thật + ground truth xuyên camera. Cân nhắc làm trên `ut-hpc` (head node có mạng, home
   còn 113 G) thay vì máy dev.
2. `src/mct/store.py` (SQLite) + `src/mct/__main__.py` (vòng online: Redis → tracklet →
   associator → `mct:global` + SQLite).
3. `homography.py` hiện thực giao thức `GroundMapper` — cần khi có cặp camera chồng lấn.
