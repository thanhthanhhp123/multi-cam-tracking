# CLAUDE.md

Hướng dẫn làm việc trong repo này. Đọc trước khi sửa bất kỳ file nào.

## 1. Bài toán

Hệ thống **MTMCT** (Multi-Target Multi-Camera Tracking): theo dõi người qua 3–4 camera IP,
gán **Global ID** nhất quán khi người đó di chuyển xuyên camera — cả cặp có vùng nhìn chồng lấn
(overlap) lẫn không chồng lấn (non-overlap). Nền tảng: NVIDIA DeepStream.

Đề cương đầy đủ (7 chương, tiếng Việt): `docs/DoAn_MultiCameraTracking_DeepStream.docx`.
Đây là **đồ án tốt nghiệp** — mọi quyết định kỹ thuật phải giải thích được trong báo cáo,
và mọi con số phải đo được lại. Ưu tiên đơn giản + đo được hơn là tối ưu cực đại.

Đóng góp kỹ thuật chính = **module liên kết đa camera** (`src/mct/`), không phải pipeline DeepStream.
Detector và Re-ID dùng model có sẵn (fine-tune nếu cần), không train from scratch.

## 2. Ràng buộc môi trường — ĐỌC KỸ

Có hai máy, code chạy ở hai nơi khác nhau:

| | Máy dev (hiện tại) | Máy chạy pipeline |
|---|---|---|
| Phần cứng | MacBook Apple M1 (arm64) | PC/laptop Ubuntu 22.04 + GPU NVIDIA rời (dự phòng: cloud GPU) |
| Chạy được | `src/common`, `src/mct`, `src/dashboard`, `src/tools`, `eval/`, toàn bộ `tests/` | tất cả, kể cả `src/ds_pipeline` |
| KHÔNG chạy được | `src/ds_pipeline` (cần pyds/CUDA/TensorRT) | — |

**DeepStream không chạy trên macOS.** Đừng đề xuất chạy `src/ds_pipeline` trên máy này,
đừng cài `pyds`/`tensorrt` vào venv của Mac, và đừng viết test cần GPU mà không đánh dấu skip.

### Quy tắc bất biến (vi phạm = hỏng cả quy trình dev)

1. **`src/common/`, `src/mct/`, `src/dashboard/`, `src/tools/`, `eval/` TUYỆT ĐỐI không được
   `import pyds`, `import gi` (GStreamer), `tensorrt`, `cuda`, hay bất kỳ thứ gì cần GPU.**
   Chỉ `src/ds_pipeline/` được phép. Có test tự động canh điều này (`tests/test_no_gpu_imports.py`).
2. Ranh giới giữa hai bên là **schema message trong `src/common/schema.py`** — không có kênh nào khác.
   Đổi schema là breaking change: phải cập nhật cả producer, consumer, và fixtures cùng lúc.
3. Mọi thứ chạy được trên Mac phải test được bằng **fixture ghi sẵn**, không cần Redis thật
   cũng không cần GPU. Fixture nằm ở `tests/fixtures/*.jsonl`.
4. **Nhắm Python 3.10** (bằng phiên bản trong container DeepStream 7.x / Ubuntu 22.04).
   Mac đang có 3.11 — không dùng cú pháp/thư viện chỉ có từ 3.11 trở lên trong code dùng chung.

## 3. Kiến trúc

Ba tầng, tách rời qua Redis Streams:

```
[Camera IP / video file]
   │ RTSP
   ▼
┌──────────────────────────── src/ds_pipeline/ (máy GPU, trong Docker) ────────────────────────────┐
│ nvurisrcbin → nvstreammux → nvinfer(PGIE: YOLO11→TensorRT) → nvtracker(NvDCF + ReID) → pad probe │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
   │ XADD mct:frames   (msgpack: bbox + local_track_id + embedding + timestamp)
   ▼
┌── Redis Streams ──┐   ← ranh giới duy nhất. Ghi lại được → replay được → dev trên Mac được.
└───────────────────┘
   │ XREADGROUP
   ▼
┌──────────────── src/mct/ (chạy đâu cũng được, không cần GPU) ─────────────┐
│ tracklet builder → gallery → affinity (cosine + spatio-temporal + homography) │
│                  → Hungarian → gán Global ID → SQLite                        │
└──────────────────────────────────────────────────────────────────────────────┘
   │ XADD mct:global + ghi SQLite
   ▼
src/dashboard/ (FastAPI + WebSocket): sơ đồ camera, vị trí hiện tại, tra cứu hành trình theo Global ID
```

**Vì sao tách bằng Redis Streams:** cho phép ghi lại luồng metadata thật từ máy GPU một lần,
rồi phát lại trên Mac để phát triển và tinh chỉnh module liên kết đa camera — phần chiếm
tuần 11–13 và là đóng góp chính của đồ án — mà không cần ngồi cạnh máy GPU. Redis kiêm luôn
state store và persistence (replay được khi consumer chết).

## 4. Cấu trúc thư mục

```
configs/
  pipeline/      nvinfer/nvtracker config (.txt/.yml) + streams.yaml (danh sách nguồn camera)
  cameras/       topology.yaml (đồ thị camera + transit time), homography/*.yaml
  mct.yaml       tham số association engine (ngưỡng, cửa sổ thời gian, trọng số)
src/
  common/        schema.py, streams.py (wrapper Redis), config.py, logging.py   ← dùng chung, KHÔNG GPU
  ds_pipeline/   builder.py, probes.py, reid_meta.py, sink.py                   ← CHỈ máy GPU
  mct/           tracklet.py, gallery.py, affinity.py, associator.py,
                 topology.py, homography.py, store.py
  dashboard/     app.py, static/, templates/
  tools/         replay_metadata.py, record_metadata.py, rtsp_sim.py,
                 calibrate_homography.py, cvat_to_mot.py, export_trackeval.py
eval/            run_trackeval.py, gt/ (ground-truth MOT format + bảng global_id)
tests/           fixtures/*.jsonl + test_*.py
docker/          deepstream.Dockerfile, compose.yml (Mac), compose.gpu.yml (máy GPU)
data/ models/    .gitignore — video, ảnh, weights, .onnx, .engine KHÔNG commit
docs/            đề cương + worklog/ (nhật ký từng phiên) + adr/ (quyết định kiến trúc lớn)
```

Mỗi package dưới `src/` chạy được bằng `python -m <tên>` (có `__main__.py`).

## 5. Contract dữ liệu — chốt một lần, đừng tự ý đổi

Định nghĩa duy nhất ở `src/common/schema.py` (dataclass + hàm encode/decode).

**Redis keys**
- `mct:frames` — metadata từ pipeline. Consumer group `mct-engine`. `MAXLEN ~ 100000`.
- `mct:global` — cập nhật Global ID cho dashboard. `MAXLEN ~ 10000`.

**Định dạng wire:** msgpack, embedding là raw bytes float32 (JSON làm phình ~3x và mất độ chính xác).
**Định dạng fixture:** JSONL, embedding base64 — để đọc/soạn tay được khi viết test.
Cả hai đi qua cùng bộ hàm encode/decode trong `schema.py`.

**Quy ước bắt buộc:**
- `bbox` = `[x, y, w, h]` pixel, gốc trên-trái, **theo độ phân giải gốc của camera**.
  DeepStream trả `rect_params` theo toạ độ của `nvstreammux` → **phải scale ngược lại** trong probe.
  Đây là nguồn bug kinh điển; kiểm tra lại mỗi khi đổi `streammux width/height`.
- `ts_ms` = epoch milliseconds (int). Giữ **cả** `frame_pts_ns` (PTS của GStreamer, để đồng bộ nội bộ)
  và `ts_ms` (wall clock từ NTP, để so khớp xuyên camera). Ràng buộc thời gian dùng `ts_ms`.
- `embedding` = float32, **L2-normalize ngay tại producer**, số chiều cố định.
  Ghi `embed_dim` vào header message (OSNet=512; model re-id kèm DeepStream `resnet50_market1501`=256).
- `cam_id` = string ổn định (`cam01`…), khớp với key trong `configs/cameras/topology.yaml`.
  Không dùng index của streammux — index đổi khi bật/tắt camera.
- `local_track_id` chỉ duy nhất trong phạm vi một camera. Khoá toàn cục = `(cam_id, local_track_id)`.

## 6. Thuật toán liên kết đa camera (`src/mct/`)

Chạy theo cửa sổ (mặc định 1s). Với mỗi tracklet cục bộ vừa cập nhật/kết thúc ở camera `c` tại `t`:

1. **Query embedding** = trung bình có trọng số của top-k embedding có confidence cao nhất
   của tracklet (bỏ crop mờ/bị che), rồi L2-normalize. Không dùng embedding của frame cuối.
2. **Lọc ứng viên** trong tập `GlobalTrack`:
   - loại bỏ track đang active ở chính camera `c` với local track khác (ràng buộc loại trừ),
   - **ràng buộc không–thời gian**: nếu track cuối thấy ở `c'`, yêu cầu
     `t − t_last ∈ [t_min(c'→c), t_max(c'→c)]` lấy từ `topology.yaml`
     (cặp overlap thì `t_min = 0`).
3. **Ma trận chi phí** = `1 − cosine_similarity`, ô không khả thi đặt `inf`.
   Với cặp camera overlap, cộng thêm `λ · d_ground` — khoảng cách giữa hai điểm chân
   sau khi ánh xạ homography về mặt phẳng tham chiếu chung.
4. **Hungarian** (`scipy.optimize.linear_sum_assignment`) trên ma trận đã mask.
5. Chấp nhận cặp gán nếu `cost < τ` (`configs/mct.yaml`); ngược lại **tạo Global ID mới**.
6. Cập nhật gallery của GlobalTrack (append có giới hạn kích thước + EMA), ghi SQLite.

**Hai chế độ, giữ cả hai:**
- `online` — chế độ thật, là sản phẩm bàn giao, dùng để đo độ trễ end-to-end.
- `offline` — chạy Hungarian theo lô trên toàn bộ tracklet đã hoàn tất. Không real-time,
  nhưng cho **cận trên** của độ chính xác; báo cáo nên đối chiếu online vs offline
  để định lượng cái giá phải trả của ràng buộc thời gian thực.

Mọi ngưỡng nằm trong `configs/mct.yaml`, **không hardcode trong code** — cần sweep tham số ở tuần 16–17.

## 7. Đánh giá

- **Single-camera** (MOTA/MOTP/IDF1/HOTA): xuất theo định dạng MOT Challenge, chạy TrackEval.
- **Cross-camera**: ghép các camera thành một chuỗi ảo (offset `frame_id` theo camera),
  dùng `global_id` làm ID, rồi chạy nhóm chỉ số Identity/HOTA — cách làm quen thuộc của AI City Challenge.
- **Hiệu năng**: FPS/luồng, độ trễ end-to-end, GPU/VRAM. Mục tiêu đề cương: 3–4 luồng, ≥15 FPS/luồng, <1s.
- Ground-truth tự gán bằng CVAT → `tools/cvat_to_mot.py` → `eval/gt/`.

Khi báo cáo số: luôn ghi kèm cấu hình GPU, model, độ phân giải, số luồng. Số không tái lập được thì vô nghĩa.

## 8. Quy ước code

- Định danh, docstring, comment: **tiếng Anh**. Tài liệu trong `docs/` và `README.md`: **tiếng Việt**.
- Type hints toàn bộ; dataclass cho mọi cấu trúc dữ liệu qua ranh giới module.
- Format/lint: `ruff format` + `ruff check`. Test: `pytest`.
- Config đọc từ YAML trong `configs/`, secret/đường dẫn máy từ `.env` (mẫu: `.env.example`).
  Không hardcode đường dẫn tuyệt đối, IP camera, hay ngưỡng thuật toán.
- Log dùng `src/common/logging.py` (structured), không dùng `print` trong code chạy production.
- Test nào cần GPU/DeepStream: đánh dấu `@pytest.mark.gpu` để CI trên Mac skip được.

## 9. Lộ trình (bám theo chương 5 đề cương, 18–20 tuần)

| Mốc | Tuần | Nội dung | Máy |
|---|---|---|---|
| M0 | 1–2 | Khung repo, `schema.py`, wrapper Redis, `replay_metadata.py`, fixture tổng hợp, dashboard rỗng | Mac |
| M1 | 3–4 | Pipeline DeepStream 1 camera: file/RTSP → YOLO → nvtracker → probe in ra console | GPU |
| M2 | 5–7 | YOLO → ONNX → TensorRT engine, đo FPS, `nvstreammux` 3–4 luồng | GPU |
| M3 | 8–10 | Trích Re-ID embedding vào metadata, publish Redis, **ghi fixture thật** | GPU |
| M4 | 11–13 | **Module liên kết đa camera** (đóng góp chính) — phát triển trên Mac bằng fixture M3 | Mac |
| M5 | 14–15 | SQLite store + dashboard realtime | Mac |
| M6 | 16–17 | Dataset tự thu + CVAT + TrackEval + sweep tham số | cả hai |
| M7 | 18–20 | (tuỳ chọn) Jetson + viết báo cáo | — |

**Trạng thái hiện tại: trước M0.** Repo mới chỉ có `docs/` và file này.

Thứ tự này cố ý đặt M3 (ghi fixture) trước M4: một khi có fixture thật, phần khó nhất của đồ án
phát triển được offline trên Mac. Nếu máy GPU chưa sẵn sàng khi tới M0, vẫn làm M0 được bằng
fixture tổng hợp sinh bằng tay.

## 10. Nhật ký làm việc — BẮT BUỘC

**Đầu mỗi phiên:** đọc 2–3 file mới nhất trong `docs/worklog/` để nạp lại ngữ cảnh —
đang ở mốc nào, thứ gì đang treo, giả định nào chưa xác minh.

**Cuối mỗi phiên:** ghi một file `docs/worklog/YYYY-MM-DD-<slug-khong-dau>.md`
theo `docs/worklog/_TEMPLATE.md`, rồi thêm một dòng vào bảng mục lục trong
`docs/worklog/README.md`. Quy ước đầy đủ nằm ở `docs/worklog/README.md`.

Hai mục quan trọng nhất, đừng bỏ trống nếu có nội dung:
- **Quyết định kỹ thuật** — chốt gì, **vì sao**, phương án nào bị loại. Trích thẳng vào
  chương 3/5 của báo cáo. Cuối kỳ ngồi nhớ lại lý do là không khả thi.
- **Số liệu đo được** — luôn kèm cấu hình (GPU, model, độ phân giải, số luồng). Cho chương 6.

Quyết định kiến trúc lớn (đủ sức nặng để cần một trang riêng) thì tạo `docs/adr/NNN-<slug>.md`
và trong worklog chỉ link tới nó.

## 11. Cạm bẫy đã biết

- **Phiên bản DeepStream chưa chốt.** Định hướng: DeepStream 7.x trên Ubuntu 22.04 (CUDA 12.x, TensorRT 10.x).
  Chốt phiên bản chính xác vào `.env` + `docker/deepstream.Dockerfile` **sau khi kiểm tra driver
  trên máy GPU thật** — DeepStream rất kén cặp driver/CUDA/TensorRT. Dùng Docker image chính thức
  của NVIDIA ngay từ đầu, đừng cài native (đề cương chương 5.2 đã liệt kê đây là rủi ro số 1).
- **Có hai đường lấy Re-ID embedding**, xác minh đường nào khả dụng trên phiên bản DeepStream thực tế
  trước khi viết code:
  - (A) **mặc định** — dùng ReID extractor tích hợp trong `nvtracker` (NvDCF/NvDeepSORT):
    bật trong config tracker, embedding ra qua user meta. Hiệu quả hơn vì tracker đã crop sẵn.
  - (B) **dự phòng** — SGIE `nvinfer` thứ hai (`process-mode=2`, `output-tensor-meta=1`) chạy OSNet
    trên crop. Đây là cách đề cương mô tả (mục 3.3); giữ lại như phương án B nếu (A) không dùng được.
  Tên chính xác của meta type khác nhau giữa các phiên bản DeepStream — **tra tài liệu bản đã cài,
  đừng đoán**.
- **Toạ độ bbox theo streammux, không phải theo camera** — xem mục 5.
- **Đồng bộ thời gian giữa các camera là điều kiện sống còn** cho ràng buộc thời gian di chuyển.
  Bật NTP trên mọi nguồn; điện thoại Android phát RTSP thường lệch — đo và ghi lại offset.
- **Điện thoại Android phát RTSP** (đề cương mục 4.1.2) có latency và jitter cao hơn camera IP.
  Đừng dùng nó cho cặp camera overlap cần homography chính xác.
- **VRAM**: chạy đồng thời YOLO + Re-ID trên 4 luồng dễ chạm trần trên GPU 8GB.
  Giảm `batch-size` của SGIE trước, rồi mới giảm độ phân giải suy luận.
- **Quyền riêng tư** (đề cương mục 4.3.3): dữ liệu người thật chỉ dùng cho học thuật, có đồng thuận,
  không commit vào git (`data/` đã ignore), làm mờ mặt trước khi đưa vào báo cáo/slide.

## 12. Lệnh (sẽ hiện thực ở M0, ghi ở đây để thống nhất tên)

```bash
# Trên Mac — không cần GPU
make dev                # cài deps CPU (pip install -e ".[dev]")
make test               # pytest, bỏ qua test có mark gpu
make lint               # ruff check + ruff format --check
make up                 # docker compose: redis + mct-engine + dashboard
make replay FIXTURE=tests/fixtures/two_cam_walk.jsonl   # phát lại metadata vào Redis
make eval               # chạy TrackEval trên kết quả trong eval/

# Trên máy GPU
make ds-build           # build docker/deepstream.Dockerfile
make ds-run             # chạy pipeline theo configs/pipeline/streams.yaml
make record OUT=tests/fixtures/<tên>.jsonl              # ghi Redis stream ra fixture
```
