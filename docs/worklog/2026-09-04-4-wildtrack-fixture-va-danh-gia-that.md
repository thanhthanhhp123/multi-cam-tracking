# 2026-09-04 (phiên 4) — Fixture WildTrack thật + lần đầu đo engine liên kết trên dữ liệu người thật

- **Mốc:** M4 (tuần 11–13) | **Máy:** `ut-hpc` (toàn bộ: tải dữ liệu, dựng fixture, đánh giá) | **Thời lượng:** ~1.5h

## Mục tiêu phiên

- Dựng fixture WildTrack (7 camera thật, ground-truth Global ID xuyên camera) để đo
  `src/mct/` trên dữ liệu người thật thay vì fixture tổng hợp.

## Đã làm

- **Annotation WildTrack** (421 file, 13 MB) tải từ mirror OpenTraj bằng
  `tools/fetch_wildtrack_annotations.py` → `~/mct/data/wildtrack/`.
- **Ảnh gốc**: tìm được link EPFL còn sống
  (`documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip`,
  6.4 GB), tải trên head node ~45 s (~150 MB/s). Giải nén được 7×401 ảnh (7.2 GB) — xem
  mục cạm bẫy bên dưới, phần này tốn nhiều công nhất.
- **OSNet ONNX**: env riêng `~/mct/venv-reid` (torch CPU + torchreid + onnxruntime). Lấy
  đúng hai checkpoint từ Model Zoo torchreid (bằng cách *đọc* trang MODEL_ZOO rồi lấy
  drive id, không đoán): `osnet_x1_0` Market-1501 (rank-1 94.2) và `osnet_x1_0` bản
  **multi-source domain generalization** (MS+D+C). Xuất cả hai sang ONNX 512 chiều.
- **Fixture** dựng bằng job SLURM (`main-gpu`, 16 CPU, không GPU — dữ liệu đã nằm trên đĩa
  nên job không chạm mạng):
  - `wildtrack_7cam.jsonl` — 2800 message, 42 606 detection, 313 danh tính, 1678 tracklet;
  - `wildtrack_3cam.jsonl` (view 1,4,7) — 1200 message, 14 701 detection, 311 danh tính,
    546 tracklet;
  - `wildtrack_3cam_dg.jsonl` — như trên nhưng embedding bằng checkpoint DG.
  Tất cả ở `~/mct/data/fixtures/` trên `ut-hpc` (40–116 MB mỗi file, không kéo về máy dev).
- `eval/eval_wildtrack.py` — chấm điểm + `--diagnose` (phân bố cosine + trần lý thuyết).
- Toàn repo vẫn xanh trên `ut-hpc`: **175 passed, 5 skipped**, ruff sạch.

## Quyết định kỹ thuật

**1. Chấm điểm chỉ trên cặp tracklet KHÁC camera.** Cặp cùng camera đã bị ràng buộc loại
trừ xử lý và luôn đúng theo thiết kế; gộp chúng vào sẽ thổi phồng điểm số mà không nói lên
gì về năng lực liên kết xuyên camera — thứ duy nhất đồ án này quan tâm.

**2. Luôn in "trần lý thuyết chỉ với ngoại hình" trước khi chỉnh tham số.** Quét mọi mốc
cosine trên chính bộ embedding đó và lấy F1 tốt nhất: không thuật toán gán nào vượt được
con số này. Biết trần trước thì không mất một tuần chỉnh `max_cost` cho một bài toán mà
đặc trưng không đủ sức tách.

**3. Topology của WildTrack dựng bằng code, không đưa vào `configs/cameras/topology.yaml`.**
File config đó mô tả hệ thống camera THẬT của đồ án; dataset mượn để thử thuật toán không
được lẫn vào đấy.

**4. Checkpoint domain-generalization thay vì Market-1501 chuyên biệt — cần xem lại kế
hoạch M3.** Xem số liệu: trên WildTrack (domain khác hẳn), bản DG cho trần lý thuyết
**0.513** so với **0.398** của bản Market-1501, và F1 thực tế **0.346** so với **0.277**
(+25%). Lộ trình hiện tại (CLAUDE.md §9, M3) ghi "fine-tune OSNet trên Market-1501/MSMT17";
kết quả này gợi ý mục tiêu huấn luyện nên là **đa nguồn/khái quát hoá miền**, vì camera của
đồ án cũng sẽ là một domain chưa từng thấy y hệt WildTrack ở đây. Chưa sửa lộ trình — cần
xác nhận lại trên dataset tự thu ở M6.

## Số liệu đo được

**Cấu hình chung:** WildTrack view 1,4,7 (3 camera 1920×1080, cùng nhìn một quảng trường,
mọi cặp đều chồng lấn), 400 khung chú thích ở ~2 fps = 199.5 s. Detector/tracker = chú thích
ground-truth của dataset (KHÔNG phải pipeline DeepStream), nên tracklet ở đây "sạch" hơn
thực tế. Embedding OSNet x1_0 512 chiều, ONNX Runtime CPU, crop resize 256×128.
Tracklet: `min_frames=3`, `idle_timeout=2000 ms` → 497 tracklet từ 546 tracklet ground-truth.

### Phân bố cosine giữa cặp tracklet khác camera (259 cặp cùng người, 69 723 cặp khác người)

| Checkpoint | nhóm | p05 | p25 | trung vị | p75 | p95 |
|---|---|---|---|---|---|---|
| Market-1501 | cùng người | 0.663 | 0.759 | **0.820** | 0.916 | 0.957 |
| Market-1501 | khác người | 0.577 | 0.648 | **0.700** | 0.749 | 0.814 |
| DG (MS+D+C) | cùng người | 0.617 | 0.753 | **0.814** | 0.887 | 0.948 |
| DG (MS+D+C) | khác người | 0.516 | 0.580 | **0.626** | 0.672 | 0.735 |

**Trần lý thuyết nếu chỉ dùng ngoại hình:** Market-1501 **F1 = 0.398** (tại cosine ≥ 0.914,
tức `max_cost = 0.086`); DG **F1 = 0.513** (tại cosine ≥ 0.842, tức `max_cost = 0.158`).

### Sweep `max_cost` (mode `max`, có topology)

| max_cost | \#Global ID | vỡ | gộp | P | R | F1 (Market) | F1 (DG) |
|---|---|---|---|---|---|---|---|
| 0.04 | 487 | 137 | 1 | 1.000 | 0.027 | 0.053 | 0.008 |
| 0.06 | 464 | 128 | 7 | 1.000 | 0.089 | 0.163 | 0.067 |
| **0.08** | 421 | 120 | 25 | 0.827 | 0.166 | **0.277** | 0.176 |
| 0.10 | 358 | 116 | 55 | 0.500 | 0.181 | 0.266 | 0.272 |
| 0.12 | 280 | 115 | 57 | 0.227 | 0.201 | 0.213 | 0.337 |
| **0.15** | 177 | 110 | 46 | 0.074 | 0.236 | 0.112 | **0.346** |
| 0.20 | 72 | 110 | 26 | 0.027 | 0.255 | 0.048 | 0.201 |
| 0.30 (mặc định) | 45 | 112 | 28 | 0.020 | 0.224 | **0.037** | 0.045 |

(Cột P/R/#gid là của bản Market-1501; bản DG có cùng dạng nhưng đỉnh lệch sang phải.)

### Bốn kết luận

1. **`max_cost = 0.30` trong `configs/mct.yaml` hoàn toàn không dùng được cho embedding
   thật.** Trên WildTrack nó cho F1 = 0.037: 497 tracklet bị gộp thành 45 Global ID. Lý do
   nhìn thấy ngay ở bảng phân bố — cosine trung vị giữa **hai người khác nhau** đã là 0.700,
   đúng bằng ngưỡng mà 0.30 tương ứng. Giá trị 0.30 chỉ đúng cho fixture tổng hợp, nơi
   similarity được đặt bằng tay theo thang khác. **Ngưỡng phải chỉnh lại theo từng model
   Re-ID và từng domain** — đây là bằng chứng định lượng mạnh nhất từ trước tới nay cho
   việc đó, đưa thẳng vào chương 6.
2. **Bật/tắt topology cho kết quả GIỐNG HỆT nhau** (mọi dòng `topo=True` và `topo=False`
   trùng khớp). Đúng như dự đoán: mọi camera WildTrack đều chồng lấn nên `min_ms = 0` và
   ràng buộc thời gian không loại được cặp nào. Nói cách khác, **WildTrack không đo được
   phần đóng góp của topology** — muốn đo phải có cặp camera non-overlap, tức dataset tự thu
   ở M6.
3. **`max` vs `centroid` gần như bằng nhau** (0.277 vs 0.272 ở đỉnh). Lần này fixture có đủ
   3 camera nên hai chế độ *không* còn trùng nhau về mặt toán học như ở fixture tổng hợp,
   nhưng chênh lệch vẫn nằm trong nhiễu. Chưa có bằng chứng để bỏ chế độ nào.
4. **Engine đạt ~70% trần lý thuyết** (0.277/0.398 với Market, 0.346/0.513 với DG). Phần
   thiếu là do gán theo cửa sổ và ràng buộc một-một của Hungarian. Nhưng điểm nghẽn chính
   **không phải thuật toán gán — mà là chất lượng đặc trưng**: ngay cả ngưỡng hoàn hảo cũng
   chỉ tới 0.4–0.5.

## Cạm bẫy đã biết (mới)

- **Zip WildTrack của EPFL lệch offset đúng 4 GiB.** File được tạo bằng công cụ macOS không
  dùng zip64 cho archive >4 GB. Hậu quả: `7z` từ chối mở hẳn ("Can't open as archive"),
  `zipfile` của Python ném `BadZipFile`, `unzip` báo "4294967296 extra bytes" rồi tự
  "re-compensate" — nhưng chỉ đúng cho các entry NẰM SAU mốc 4 GiB (lấy được C5–C7, hỏng
  C1–C4), và còn chết giữa chừng vì "not enough memory for bomb detection".
  **Cách vá đã dùng** (`scratchpad/unzip_wt.py`): với mỗi entry, thử `header_offset` ở cả ba
  giá trị `+0`, `+2^32`, `−2^32` rồi kiểm tra kích thước giải nén — lấy đủ 2807/2807 ảnh,
  0 lỗi. Nếu phải làm lại: cũng cần `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE` khi dùng `unzip`.
- **Checkpoint DG của torchreid không nạp được bằng `torch.load` mặc định** (torch ≥ 2.6
  đặt `weights_only=True`; pickle của checkpoint tham chiếu `numpy.core.multiarray.scalar`,
  đường dẫn cũ mà numpy 2 đã đổi tên nên `add_safe_globals` không khớp). Đã vá bằng
  `Unpickler` riêng chỉ cho phép global thuộc `numpy`/`torch`/`collections` và ánh xạ
  `numpy.core` → `numpy._core`, rồi lưu lại state_dict sạch. **Không** mở
  `weights_only=False` trần.
- **`torch.onnx.export` với torch 2.14 luôn sinh ONNX kèm file `.data` ngoài** (833 KB +
  8.3 MB). ONNX Runtime nạp được miễn hai file nằm cạnh nhau — nhớ chép cả cặp khi chuyển máy.
- **Partition `debug` của `ut-hpc` chỉ mở cho account `admin,root`** (`AllowAccounts=admin,root`),
  dù `AllowGroups=ALL` nhìn như mở. Vẫn phải dùng `main-gpu` cho cả job CPU thuần.

## Vướng mắc / chưa xong

- **Chưa dùng hình học.** WildTrack có sẵn file hiệu chỉnh camera *và* `world_xy_m` trong
  `gt.json`, mọi camera chồng lấn — đây là điều kiện lý tưởng để thành phần `λ · d_ground`
  trong `affinity.py` phát huy, mà `homography.py` thì chưa viết. Gần như chắc chắn đây là
  thứ kéo F1 lên nhiều nhất, hơn mọi cách chỉnh ngưỡng.
- **Chưa chạy bản 7 camera** (`wildtrack_7cam.jsonl`, 1678 tracklet) — mới đo trên bản 3
  camera cho nhanh vòng lặp.
- Chưa đưa `max_cost` mới vào `configs/mct.yaml`: giá trị đúng khác nhau theo model
  (0.08 cho Market-1501, 0.15 cho DG) nên phải chốt model trước rồi mới chốt ngưỡng.
- Fixture WildTrack nằm trên `ut-hpc`, chưa kéo về `tests/fixtures/` của máy dev (40–116 MB).
- `data/wildtrack` + zip chiếm ~14 GB trên home `ut-hpc` (còn 99 G). Xoá zip được nếu cần
  chỗ — link tải đã ghi ở trên, tải lại mất ~45 s.

## Bước tiếp theo

1. **`src/mct/homography.py`** hiện thực giao thức `GroundMapper` — nạp file hiệu chỉnh
   WildTrack (`calibrations/`), ánh xạ điểm chân về mặt phẳng chung, rồi đo lại: đây là
   phép thử trực tiếp cho thấy hình học đóng góp bao nhiêu khi ngoại hình đã hết đất.
2. Chạy lại đánh giá trên bản 7 camera sau khi có homography.
3. `store.py` + `mct/__main__.py` (vòng online qua Redis) để đo được chênh lệch online vs
   offline — vẫn treo từ phiên trước.
