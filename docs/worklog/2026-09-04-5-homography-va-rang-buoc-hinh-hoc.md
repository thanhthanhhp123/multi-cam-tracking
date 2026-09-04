# 2026-09-04 (phiên 5) — Homography + vòng online: F1 trên WildTrack 0.014 → 0.929

- **Mốc:** M4 (tuần 11–13) | **Máy:** máy dev (soạn code) + `ut-hpc` (test, hiệu chỉnh, sweep) | **Thời lượng:** ~5h

## Mục tiêu phiên

- Viết `src/mct/homography.py` — mảnh cuối còn thiếu của lõi engine, hiện thực giao thức
  `GroundMapper` mà `affinity.py` đã chừa sẵn từ phiên 3.
- Đo lại trên fixture WildTrack: phiên trước kết luận "ngoại hình đã hết đất ở F1 ≈ 0.4–0.5,
  hình học gần như chắc chắn là thứ kéo lên nhiều nhất" — kiểm chứng câu đó.
- Nối nốt hai tầng cuối của sơ đồ: `store.py` (SQLite) + `__main__.py` (vòng online), để
  đo được chênh lệch online vs offline — con số "giá của thời gian thực" cho chương 6.

## Đã làm

- **`src/mct/homography.py`** — DLT chuẩn hoá Hartley + cắt tỉa điểm sai lớn
  (`estimate_homography`), `CameraHomography` (H + siêu dữ liệu, đọc/ghi YAML),
  `HomographyMapper` (nhiều camera, hiện thực `GroundMapper`). **Chỉ numpy** — `src/mct/`
  không được phụ thuộc OpenCV (CLAUDE.md §2 quy tắc 1), và `cv2.findHomography` không đem
  lại gì mà 40 dòng DLT không làm được ở quy mô này.
- **`src/tools/calibrate_homography.py`** — hai nguồn cặp điểm: (A) file YAML điểm đo tay
  cho camera thật ở M6; (B) khớp tự động từ chú thích WildTrack (đáy-giữa bbox ↔ `positionID`).
  Cách (B) cho hàng nghìn cặp điểm miễn phí, không phải chạm vào file calibration OpenCV
  của dataset.
- **`configs/cameras/homography/wildtrack/cam0{1..7}.yaml`** + README giải thích quy ước.
- **Ba sửa đổi lớn trong lõi engine** — đều do *đo* mới lòi ra, không phải do đọc code
  (chi tiết ở mục Quyết định kỹ thuật): quỹ đạo theo thời gian, Hungarian theo từng camera,
  `ground_gap_policy`.
- `eval/eval_wildtrack.py`: thêm `--homography-dir`, `--ground-gap-policy`, và chẩn đoán
  `diagnose_ground` (phân bố khoảng cách mặt đất cùng/khác danh tính + trần lý thuyết).
- **`src/mct/store.py`** — SQLite hai bảng (`global_tracks`, `appearances`). Trạng thái
  của `global_tracks` suy ra hoàn toàn từ `appearances` nên hai bảng không bao giờ lệch;
  khoá chính của `appearances` là `tracklet_id` nên tracklet dài được gán lại qua nhiều
  cửa sổ chỉ nới `end_ms` chứ không đẻ thêm dòng.
- **`src/mct/__main__.py`** — vòng online thật: `mct:frames` → tracklet → associator →
  `mct:global` + SQLite. Cửa sổ tính theo `ts_ms` **trong message**, không theo đồng hồ
  hệ thống; `--source <fixture.jsonl>` chạy đúng vòng lặp đó mà không cần Redis.
- **`common/schema.py`: `GlobalUpdate`** + `GlobalPublisher`/`read_global` trong
  `common/streams.py` — nội dung stream `mct:global`, cố tình không mang embedding.
- **`eval/compare_online_offline.py`** — cùng dữ liệu, cùng tham số, hai đường.
- Makefile: `wildtrack-homography`, `engine`, `engine-fixture`, `compare`.
- 57 test mới → **233 passed, 5 skipped**, `ruff` sạch (chạy trên `ut-hpc`, Python 3.10.12).

## Quyết định kỹ thuật

**1. So vị trí phải TẠI CÙNG MỘT THỜI ĐIỂM, không phải "điểm cuối ↔ điểm đầu".**
Bản đầu của `_ground_term` so `GlobalTrack.last_ground_point` với `Tracklet.first_ground_point`.
Hai mốc đó cách nhau tuỳ ý; người đi 2 m/s thì lệch 3 giây là sai 6 m — đủ để ngưỡng
`max_ground_dist_m` loại sạch những cặp ĐÚNG. Đo được: bật hình học kiểu đó làm F1 **tệ đi**
(0.014 → 0.000 ở lần chạy đầu). Sửa: `Tracklet` giữ quỹ đạo `(ts_ms, điểm chân)` có chặn
kích thước (`ground_path_max_points`, đầy thì tỉa một nửa), `GlobalTrack` giữ quỹ đạo gần
nhất **theo từng camera**, affinity ghép hai quỹ đạo theo mốc thời gian (dung sai
`ground_time_tol_ms`) rồi lấy **trung vị** khoảng cách. Trung vị chứ không phải trung bình:
một khung có bbox bị che cho điểm chân lệch hàng mét.

**2. Hungarian phải chạy THEO TỪNG CAMERA, không phải một lần cho cả vòng.**
Đây là lỗi thiết kế nặng nhất tìm được trong phiên. Phép ghép một-một chỉ đúng **trong phạm
vi một camera** (hai local track khác nhau của cùng camera không thể là một người — đúng là
ràng buộc loại trừ). Giữa các camera thì ngược lại: camera chồng lấn nhìn thấy cùng một
người **cùng lúc**, nên một Global ID phải nhận được nhiều tracklet trong cùng một vòng.
Gộp tất cả vào một ma trận là áp nhầm ràng buộc một-một lên cả chiều liên camera: người xuất
hiện ở 7 camera thì 6 tracklet còn lại bị đẩy sang Global ID mới, danh tính vỡ vụn ngay
vòng đầu. **Recall 0.060 → 0.365 chỉ nhờ sửa chỗ này.** Giá phải trả: thứ tự xử lý camera
ảnh hưởng kết quả (camera sau nhìn thấy gallery đã cập nhật) — sắp theo `cam_id` cho tất định,
và ghi nhận đây là một điểm chưa tối ưu toàn cục.

**3. `ground_gap_policy` — cặp camera chồng lấn KHÔNG có mốc thời gian chung.**
Hình học chỉ phán được khi hai bên cùng thấy người vào một lúc; phần lớn cặp SAI lại không
có mốc chung nên lọt qua và ngoại hình (vốn là nhiễu) quyết định. Hai lựa chọn, giữ cả hai
làm tham số vì đây là đánh đổi thật:
- `allow` (mặc định) — thả qua, chỉ nới ngưỡng theo `max_speed_m_s · Δt` (ngân sách vật lý:
  người ta kịp đi bao xa trong khoảng đó). An toàn, không bịa ràng buộc.
- `reject` — loại thẳng. Lập luận: hai camera nhìn chung một vùng thì cùng một người phải
  được cả hai thấy vào cùng lúc; không có mốc chung nghĩa là **không có bằng chứng hình học**,
  mà ngoại hình xuyên camera đã đo được là yếu. Đổi recall lấy precision.
Trên WildTrack (mọi camera chồng lấn): `reject` cho P 0.060 → 0.544. Mặc định trong
`configs/mct.yaml` vẫn để `allow` — hệ thống thật có cặp non-overlap, ở đó lập luận trên
không áp dụng.

**4. Có hình học thì `max_cost` phải NỚI RA, không siết vào.** Ngược hẳn kết luận của phiên
trước (khi chỉ có ngoại hình, phải siết xuống 0.08–0.15). Lý do: khi vị trí đã lo phần loại
bỏ, ngưỡng ngoại hình chặt chỉ tổ ném đi những cặp đúng có cosine thấp (p05 của cặp cùng
người = 0.589, tức cost 0.41). Sweep đầy đủ: 0.30 → 0.446, 0.40 → 0.712, 0.50 → 0.737,
0.60 → 0.765, ≥0.80 bão hoà ở 0.768.

**Vẫn giữ `max_cost: 0.30` làm mặc định trong `configs/mct.yaml`, và đây là quyết định có
chủ ý.** WildTrack chỉ có cặp chồng lấn, nên mọi cặp ở đó đều được hình học bảo vệ. Hệ thống
thật có cặp non-overlap, và ở đó ngoại hình là bằng chứng DUY NHẤT — thả ngưỡng lên 0.60
tại đó nghĩa là gộp gần như mọi người lại với nhau. Lời giải đúng là **ngưỡng theo từng cặp
camera** (chặt khi thiếu hình học, lỏng khi có), nhưng chưa làm vì chưa có dữ liệu non-overlap
thật để đo — ghi lại thành việc của M6 thay vì tối ưu bừa lên một dataset mượn.

**`max_ground_dist_m` hạ 3.0 → 1.0.** Có căn cứ đo: sai số hiệu chỉnh p95 cho cặp hai camera
là 0.14 m, nên 1.0 m vẫn rộng gấp bảy lần sai số; và sweep cho đỉnh phẳng quanh 1.0
(0.5 → 0.736, 1.0 → 0.768, 2.0 → 0.764).

**5. Homography chứ không hiệu chỉnh camera đầy đủ.** Giả thiết duy nhất: người đứng trên
một mặt phẳng. Khi đó điểm chân và điểm mặt đất liên hệ bằng đúng một ma trận 3x3, cần ≥4
cặp điểm, không cần nội/ngoại tham số. Rẻ hơn nhiều, sai số đủ dùng, và giải thích được
trong báo cáo bằng một công thức.

## Số liệu đo được

**Cấu hình chung:** fixture `wildtrack_7cam.jsonl` (WildTrack 7 camera 1920x1080, 400 khung
chú thích ~2 fps, 313 danh tính), embedding OSNet chạy ONNX Runtime **CPU**, tracklet
`min_frames=3`, `idle_timeout=2000 ms` → **1544 tracklet**. Chỉ số theo **cặp tracklet KHÁC
camera** (cặp cùng camera bị ràng buộc loại trừ xử lý riêng, gộp vào sẽ thổi phồng điểm).
Chạy trên `ut-hpc` (head node cho các lần đo lẻ, `sbatch --partition=main-gpu` cho sweep).

### Hiệu chỉnh homography (`tools/calibrate_homography.py --wildtrack-dir`)

400 khung, 42607 detection, loại 3592 cái chạm mép khung (8.4% — bbox bị cắt thì điểm chân
không nằm trên mặt đất).

| camera | điểm dùng/đầu vào | RMSE | trung vị | p95 |
|---|---|---|---|---|
| cam01 | 6727/8304 | 0.018 m | 0.014 m | 0.033 m |
| cam02 | 6041/7458 | 0.030 m | 0.024 m | 0.053 m |
| cam03 | 5016/6192 | 0.022 m | 0.016 m | 0.043 m |
| cam04 | 1141/1409 | 0.022 m | 0.019 m | 0.034 m |
| cam05 | 2882/3558 | 0.010 m | 0.009 m | 0.018 m |
| cam06 | 7216/8909 | 0.032 m | 0.024 m | 0.060 m |
| cam07 | 2579/3185 | 0.012 m | 0.006 m | 0.028 m |

Sai số gộp mọi camera: trung vị 0.02 m, p95 0.10 m → cặp hai camera lệch cỡ 0.14 m (p95).

> **Đừng lấy con số này làm kỳ vọng cho camera thật.** Bbox của WildTrack vốn được sinh ra
> bằng cách chiếu vị trí lưới qua calibration của dataset, nên phép khớp gần như khôi phục
> lại đúng phép chiếu đó — sai số cỡ centimet là *tất yếu*, không phải thành tích. Ở camera
> thật, điểm chân đến từ bbox của detector, sai số sẽ lớn hơn nhiều bậc.

### Trần lý thuyết: ngoại hình so với hình học

Cặp tracklet khác camera, tách theo cùng/khác danh tính:

| tín hiệu | nhóm | p05 | p25 | trung vị | p75 | p95 |
|---|---|---|---|---|---|---|
| cosine | cùng người (3496 cặp) | 0.589 | 0.671 | 0.736 | 0.802 | 0.891 |
| cosine | khác người (995184 cặp) | 0.555 | 0.634 | 0.690 | 0.746 | 0.824 |
| d_ground | cùng người (3723 cặp) | 0.017 | 0.032 | **0.055** | 0.144 | 1.102 |
| d_ground | khác người (148783 cặp) | 2.088 | 5.795 | **9.349** | 13.768 | 20.810 |

- **Chỉ ngoại hình: F1 ≤ 0.053** (tại cosine ≥ 0.906). Hai phân bố chồng gần như hoàn toàn.
- **Chỉ hình học: F1 ≤ 0.929** (tại d_ground ≤ 0.53 m). Hai phân bố tách bạch.

Đây là con số quan trọng nhất của phiên: trên bộ camera chồng lấn, **hình học mạnh hơn
ngoại hình gần 20 lần**, và mọi công sức chỉnh ngưỡng ngoại hình đều vô ích.

### Engine chạy thật

`similarity_mode=max`, topology "mọi camera chồng lấn", `max_ground_dist_m=0.5`, λ=0.4:

| cấu hình | max_cost | #Global ID | danh tính vỡ | ID gộp nhầm | P | R | F1 |
|---|---|---|---|---|---|---|---|
| chỉ ngoại hình | 0.30 | 32 | 292 | 29 | 0.007 | 0.135 | 0.014 |
| + hình học, `allow` | 0.30 | 120 | 286 | 90 | 0.060 | 0.134 | 0.083 |
| + hình học, `reject` | 0.30 | 350 | 256 | 151 | 0.555 | 0.373 | 0.446 |
| + hình học, `reject` | **0.40** | 317 | 166 | 96 | **0.760** | **0.670** | **0.712** |

Ba lần nhảy bậc, mỗi lần là một sửa đổi ở mục Quyết định kỹ thuật:
`0.014 → 0.083` (thêm hình học so theo thời gian) → `0.446` (Hungarian theo từng camera +
`reject`) → `0.712` (nới `max_cost`).

### Sweep đầy đủ `max_ground_dist_m` × `max_cost` (chế độ `reject`, job SLURM 581533)

| d_max \ max_cost | 0.30 | 0.40 | 0.50 | 0.60 | 0.80 | 1.00 |
|---|---|---|---|---|---|---|
| **0.5 m** | 0.446 | 0.712 | 0.736 | 0.736 | 0.736 | 0.736 |
| **1.0 m** | 0.437 | 0.705 | 0.737 | 0.765 | **0.768** | **0.768** |
| **2.0 m** | 0.443 | 0.708 | 0.734 | 0.764 | 0.748 | 0.739 |

(giá trị = F1). Hai đọc được: (a) `max_cost` là trục nhạy hơn hẳn `max_ground_dist_m`;
(b) ở `d_max = 2.0` m, nới `max_cost` lên 0.80–1.00 bắt đầu **giảm** F1 (0.768 → 0.748 →
0.739) — cả hai ràng buộc cùng lỏng thì không còn gì chặn việc gộp người.

### Online so với offline — "giá của thời gian thực" (job SLURM 581534)

Cùng fixture, cùng tham số, khác nhau duy nhất ở thời điểm tracklet được đưa vào vòng gán.

| cấu hình | chế độ | #Global ID | danh tính vỡ | ID gộp nhầm | P | R | F1 |
|---|---|---|---|---|---|---|---|
| hình học `reject`, max_cost 0.60, d_max 1.0 | **online** | 321 | 62 | 14 | **0.976** | **0.886** | **0.929** |
| hình học `reject`, max_cost 0.60, d_max 1.0 | offline | 310 | 131 | 85 | 0.780 | 0.751 | 0.765 |
| chỉ ngoại hình, max_cost 0.10 | online | 840 | 297 | 206 | 0.126 | 0.024 | 0.041 |
| chỉ ngoại hình, max_cost 0.10 | offline | 699 | 297 | 128 | 0.033 | 0.032 | 0.033 |

**Online TỐT HƠN offline (F1 0.929 vs 0.765), và chạm đúng trần lý thuyết 0.929 của hình
học.** Đây là kết quả ngược với giả định trong CLAUDE.md §6 ("offline cho cận trên của độ
chính xác"), và nó có lý do rõ ràng, không phải may mắn:

> **Ràng buộc hình học là một hàm của THỜI GIAN, nên nó mạnh nhất khi phép gán diễn ra gần
> thời gian thực.** Chế độ online đưa tracklet vào vòng gán ngay lúc nó đang chạy, nên quỹ
> đạo của nó và quỹ đạo trong gallery **đương nhiên trùng khoảng thời gian** — đúng điều
> kiện để `_synchronized_distance` phán được. Chế độ offline xếp tracklet theo `end_ms`;
> lúc một tracklet đã đóng được đem đi ghép, quỹ đạo mà GlobalTrack đang giữ cho camera kia
> có thể thuộc một quãng thời gian hoàn toàn khác, và bằng chứng hình học biến mất.

Với đường **chỉ ngoại hình** thì khoảng cách giữa hai chế độ gần như không có (0.041 vs
0.033) — đúng như dự đoán: ngoại hình không phụ thuộc thời điểm gán.

Hệ quả cho báo cáo (chương 6): **câu "thời gian thực phải trả giá bằng độ chính xác" không
đúng một cách phổ quát.** Nó đúng với phần ngoại hình, nhưng với cặp camera chồng lấn thì
thời gian thực lại là *điều kiện* để ràng buộc không–thời gian phát huy. Cận trên thật của
hệ thống nằm ở đường online, không phải đường offline — và câu tương ứng trong CLAUDE.md §6
cần sửa lại sau khi kiểm chứng thêm trên dataset có cặp non-overlap (M6).

## Vướng mắc / chưa xong

- **`ground_gap_policy=reject` chưa được kiểm trên cặp non-overlap.** WildTrack không có
  cặp nào như vậy, mà chính ở đó lập luận của `reject` ("không cùng lúc thì không phải một
  người") là SAI. Mặc định vẫn `allow`; chỉ cân nhắc đổi sau dataset tự thu M6.
- **`max_cost` mặc định (0.30) không phải giá trị tốt nhất trên WildTrack (0.60–0.80).**
  Cố ý — xem Quyết định kỹ thuật 4. Việc đúng phải làm là ngưỡng theo từng cặp camera.
- **Kết luận "online tốt hơn offline" mới đo trên MỘT dataset toàn cặp chồng lấn.** Cơ chế
  giải thích được và nhất quán với số liệu, nhưng đừng viết vào báo cáo như một quy luật
  cho tới khi đo lại trên dữ liệu có cặp non-overlap.
- **`_ground_term` chậm**: mỗi cặp (tracklet, GlobalTrack) phải ghép hai quỹ đạo. Đã cache
  phép chiếu trong phạm vi một lần dựng ma trận nhưng vẫn O(n·m·k) — một cấu hình sweep
  mất ~4 phút cho 1544 tracklet. Chưa chạm giới hạn thật (fixture 2 fps), nhưng ở 15 fps
  ×4 luồng thì phải đo lại độ trễ end-to-end trước khi kết luận đạt mục tiêu <1 s.
- **Thứ tự camera ảnh hưởng kết quả** (hệ quả của Hungarian theo từng camera). Chưa đo độ
  nhạy: chạy lại với `sorted(by_cam)` đảo ngược sẽ cho biết ảnh hưởng lớn hay nhỏ.
- **Dashboard chưa đọc `mct:global`** — publisher và schema đã có, bên tiêu thụ thì chưa.
- **Chưa đo độ trễ end-to-end thật** (từ lúc probe đẩy message tới lúc có Global ID). Cần
  chạy engine cạnh pipeline trên `vast-gpu`, chưa làm.

## Bước tiếp theo

1. `src/dashboard/` — đọc `mct:global` qua WebSocket + tra cứu hành trình từ SQLite
   (`Store.trajectory`). Toàn bộ phía dữ liệu đã sẵn sàng, đây là M5.
2. Đo độ nhạy theo thứ tự camera trong `Associator._match` (đảo `sorted(by_cam)`) — biết
   cái giá của phép ghép tham lam theo camera.
3. Đo độ trễ end-to-end: chạy `python -m mct` cạnh pipeline DeepStream trên `vast-gpu`
   (nhớ xác nhận với người dùng trước, tính phí theo giờ).
