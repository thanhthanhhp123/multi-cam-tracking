# 2026-09-04 (phiên 5) — Homography: hình học vào cuộc, F1 trên WildTrack 0.014 → 0.712

- **Mốc:** M4 (tuần 11–13) | **Máy:** máy dev (soạn code) + `ut-hpc` (test, hiệu chỉnh, sweep) | **Thời lượng:** ~3h

## Mục tiêu phiên

- Viết `src/mct/homography.py` — mảnh cuối còn thiếu của lõi engine, hiện thực giao thức
  `GroundMapper` mà `affinity.py` đã chừa sẵn từ phiên 3.
- Đo lại trên fixture WildTrack: phiên trước kết luận "ngoại hình đã hết đất ở F1 ≈ 0.4–0.5,
  hình học gần như chắc chắn là thứ kéo lên nhiều nhất" — kiểm chứng câu đó.

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
- 32 test mới (`tests/test_homography.py` 30, `tests/test_associator.py` 2) →
  **208 passed, 5 skipped**, `ruff` sạch (chạy trên `ut-hpc`, Python 3.10.12).

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
người = 0.589, tức cost 0.41). Đo được: `max_cost` 0.30 → F1 0.446, 0.40 → **0.712**.
Con số trong `configs/mct.yaml` **chưa chốt** — chờ sweep đầy đủ và chờ model Re-ID cuối cùng.

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
`reject`) → `0.712` (nới `max_cost`). Còn cách trần 0.929 khoảng 0.22 — phần chênh nằm ở
những cặp không có mốc thời gian chung, chỗ mà hệ thống buộc phải tin vào ngoại hình.

## Vướng mắc / chưa xong

- **Sweep đầy đủ `max_cost` × `max_ground_dist_m` chưa xong** (job SLURM 581533, ~4 phút/cấu
  hình vì phần chiếu quỹ đạo còn chậm). Bốn dòng trong bảng trên là các mốc đã có; giá trị
  chốt cho `configs/mct.yaml` chờ kết quả đầy đủ.
- **Chưa chốt `max_cost` và `ground_gap_policy` trong `configs/mct.yaml`** — mặc định vẫn là
  0.30/`allow`. Chốt sau khi có sweep đầy đủ *và* sau khi chọn xong model Re-ID (M3).
- **`_ground_term` chậm**: mỗi cặp (tracklet, GlobalTrack) phải ghép hai quỹ đạo. Đã cache
  phép chiếu trong phạm vi một lần dựng ma trận, nhưng vẫn là O(n·m·k). Chưa tối ưu vì chưa
  chạy online thật — đo lại khi có `__main__.py`.
- **Thứ tự camera ảnh hưởng kết quả** (hệ quả của Hungarian theo từng camera). Chưa đo độ
  nhạy: chạy lại với thứ tự camera đảo ngược sẽ cho biết ảnh hưởng lớn hay nhỏ.
- **WildTrack không có cặp non-overlap** nên `ground_gap_policy=reject` và toàn bộ phần
  ràng buộc thời gian di chuyển vẫn chưa được kiểm bằng dữ liệu thật. Chờ dataset tự thu M6.
- `src/mct/store.py` và `src/mct/__main__.py` vẫn chưa có → **chưa đo được chênh lệch
  online vs offline**, con số "giá của thời gian thực" cho chương 6.

## Bước tiếp theo

1. Đọc kết quả sweep 581533, chốt `max_cost` (và cân nhắc `ground_gap_policy`) vào
   `configs/mct.yaml` kèm lý do.
2. `src/mct/store.py` (SQLite) + `src/mct/__main__.py` (vòng online: Redis → tracklet →
   associator → `mct:global` + SQLite) — mở đường đo online vs offline.
3. Đo độ nhạy theo thứ tự camera trong `_match` (đảo `sorted(by_cam)`) để biết cái giá của
   phép ghép tham lam theo camera.
