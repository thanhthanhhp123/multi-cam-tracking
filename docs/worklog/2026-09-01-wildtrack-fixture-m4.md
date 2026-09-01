# 2026-09-01 — Fixture từ WildTrack để kéo M4 lên sớm

- **Mốc:** chuẩn bị M4 (đáng lẽ tuần 11) | **Máy:** Mac M1 | **Thời lượng:** ~2h

## Mục tiêu phiên

- Thay vì chờ pipeline (M1–M3) rồi mới ghi fixture thật, dựng fixture từ một dataset
  multi-camera công khai CÓ ground-truth Global ID để phát triển engine liên kết ngay.
- Chọn dataset + model Re-ID + runtime, viết bộ chuyển đổi, test không cần GPU/ảnh gốc.

## Đã làm

- **`src/tools/wildtrack_to_fixture.py`** — WildTrack → `<out>.jsonl` + `<out>.gt.json`,
  cùng định dạng `make_synthetic_fixture.py`. `viewNum 0..6 → cam01..cam07`,
  `personID → gt_global_id`, `local_track_id` sinh mới đếm tăng trong từng camera.
  Tách tracklet khi một người rời FOV > `--reentry-gap-frames` khung rồi quay lại.
  Hai chế độ: `--no-reid` (chỉ hình học, không đọc ảnh) và `--reid-onnx` (có embedding).
- **`src/tools/reid_onnx.py`** — `OsnetOnnxEmbedder`: bọc OSNet ONNX chạy trên
  ONNX Runtime CPU, tiền xử lý khớp torchreid (resize 256×128, chuẩn hoá ImageNet).
  `onnxruntime`/`cv2` import trễ để không phá `test_no_gpu_imports`.
- **`src/tools/export_osnet_onnx.py`** — helper chạy một lần (cần `torch`+`torchreid`,
  không nằm trong deps dự án) xuất OSNet pretrained ra `models/reid/*.onnx`.
- **`src/tools/fetch_wildtrack_annotations.py`** — tải phần nhỏ của WildTrack
  (400 file annotation + 21 file calibration) từ mirror `crowdbotp/OpenTraj`, stdlib,
  tải lại được. Ảnh gốc ~13GB tải riêng từ EPFL.
- **`tests/test_wildtrack_to_fixture.py`** — 19 test. Dữ liệu WildTrack giả sinh trong
  test cho phần logic; `image_reader` + embedder giả cho phần embedding (không cần
  opencv/onnxruntime). Thêm 2 file annotation THẬT ở `tests/data/wildtrack/` để test
  đối chiếu schema.
- `pyproject.toml`: thêm extra `[reid]` = `[cv]` + `onnxruntime`.
- `Makefile`: `make wildtrack-annotations`, `make wildtrack-fixture`.
- `.gitignore`: neo `data/` → `/data/`, `models/` → `/models/` vào gốc repo, nếu không
  `tests/data/` cũng bị bỏ qua (rule `data/` khớp mọi cấp).
- `tests/fixtures/README.md`, `README.md`: cập nhật.
- Toàn bộ: **75 test pass** (trước 55), lint sạch. Chạy thử CLI `--no-reid` trên 2 frame
  mẫu: 6 message / 113 detection / 39 danh tính OK.

## Quyết định kỹ thuật

**1. Dùng WildTrack làm bàn thử M4 trước khi có pipeline.** Fixture tổng hợp (M0) điều
khiển được độ khó nhưng embedding của nó do chính ta sinh ra theo một mô hình 3 tầng —
không có "bất ngờ" nào mà mô hình đó không chứa sẵn. WildTrack: 7 camera HD tĩnh nhìn
chung một quảng trường, chú thích ~2 fps, `personID` gán nhất quán xuyên camera — tức là
đã có sẵn bảng đáp án cho MTMCT. Chạy Re-ID pretrained qua các bbox của nó (chỉ inference,
CPU Mac chịu được) là ra fixture "thật" đủ để thử `affinity.py`/`associator.py` với
Hungarian + cosine + ràng buộc thời gian, đối chiếu trực tiếp với ground-truth.

Phương án loại:
- *Chờ M3 ghi fixture từ pipeline* — đúng quy trình nhưng khoá M4 sau khi thuê được
  `vast-gpu` và làm xong M1–M3. WildTrack gỡ được phụ thuộc đó.
- *MOT17/MOT20* — chỉ single-camera, không có tín hiệu cross-camera.
- *Chỉ dùng fixture tổng hợp* — giữ lại (điều khiển độ khó cho sweep tuần 16–17) nhưng
  không thay được dữ liệu thật.

**2. OSNet chạy ONNX Runtime (CPU), không phải PyTorch/torchreid.** Runtime nhẹ (wheel
arm64 sạch, không kéo theo torch), và đường ONNX chính là đường Re-ID sẽ chạy trong
pipeline thật (ONNX → TensorRT trong DeepStream) nên tiền xử lý dùng lại được. `torch`
+ `torchreid` chỉ cần một lần để xuất `.onnx` (`export_osnet_onnx.py`), không vào deps
dự án. Model: `osnet_x1_0` / Market-1501 / feature 512-d.

**3. Tách chế độ `--no-reid`.** Fixture chỉ có bbox + thời gian + local id (embedding
rỗng) dựng được CHỈ từ 400 file annotation (~20MB), không cần ảnh gốc 13GB, không cần
`onnxruntime`, chạy trong CI. Đủ để test phần gom tracklet + lọc ứng viên không–thời
gian + (sau này) homography. Phần ngoại hình cần chế độ đầy đủ.

**4. `local_track_id` tách khi người ra/vào lại FOV.** WildTrack không phân mảnh track,
nếu cấp một local id cho suốt lượt xuất hiện thì fixture bỏ mất đúng ca thú vị nhất của
MTMCT (bắt lại Global ID sau khi mất dấu). `--reentry-gap-frames` mặc định 4 ≈ 2s ở
2 fps, khớp `idle_timeout_ms` trong `configs/mct.yaml`.

## Số liệu đo được

Chưa có số liệu chất lượng liên kết (chưa có `src/mct/`). Số về công cụ:

| Đại lượng | Giá trị |
|---|---|
| Test | 55 → 75 pass, 5 skip, lint sạch |
| CLI `--no-reid` trên 2 frame mẫu | 6 message, 113 detection, 39 danh tính, 60 tracklet |
| Kích thước fixture đầy đủ (ước tính) | ~110MB JSONL cho toàn bộ 400 frame × 7 cam, embed 512-d → dùng `--views` + `--max-frames` để cắt |

## Vướng mắc / chưa xong

- **Chưa tải ảnh gốc WildTrack (~13GB)** → chưa có fixture mang embedding thật. Mới xác
  minh được đường `--no-reid`.
- **Chưa có file `models/reid/osnet_x1_0_market1501.onnx`.** Cần chạy `export_osnet_onnx.py`
  với checkpoint re-id tải từ Model Zoo torchreid (link Google Drive trong bảng zoo).
- **Công thức `positionID → world (X,Y)` chưa đối chiếu bằng overlay.** Lấy từ code
  hou-yz/MVDet (`grid_x = pid % 480`, gốc (-3,-9)m, ô 2.5cm), khớp mô tả toolkit nhưng
  chưa vẽ lên ảnh để chắc. Ảnh hưởng `world_xy_m` trong `.gt.json` và phần homography sau.
- **WildTrack: mọi camera đều overlap** — không có cặp non-overlap. Kịch bản non-overlap
  của đề cương (mục 4.2) vẫn phải chờ dataset tự thu ở M6.
- **`local_track_id` là SCT lý tưởng** (chỉ đứt khi ra khỏi FOV, không id-switch). Tracklet
  pipeline thật phân mảnh hơn — cần một biến thể "khó" tiêm id-switch khi tới lúc sweep.
- `src/mct/` vẫn rỗng.
- Ghi nhận (không thuộc phiên này): `docs/DoAn_MultiCameraTracking_DeepStream.docx` bị
  thay bằng `..._v2.docx` trong working tree — CLAUDE.md §1 còn trỏ tên cũ, cần chỉnh khi
  chốt bản v2.

## Bước tiếp theo

1. `src/mct/tracklet.py` — gom detection theo `(cam_id, local_track_id)`, tính query
   embedding = trung bình top-k theo confidence (CLAUDE.md §6 bước 1), phát hiện tracklet
   kết thúc theo `idle_timeout_ms`.
2. `src/mct/topology.py` + `affinity.py` — nạp `configs/cameras/topology.yaml`, ma trận
   chi phí `1 − cosine` có mask ràng buộc không–thời gian.
3. `src/mct/associator.py` — Hungarian + ngưỡng `max_cost` → gán/ tạo Global ID.
4. Test đối chiếu với `wildtrack_geom.jsonl` (phần thời gian) trước, rồi fixture đầy đủ
   khi có ảnh + `.onnx`. Thêm chỉ số: độ chính xác gán Global ID so với `.gt.json`.
