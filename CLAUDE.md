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

Có ba máy, mỗi máy một vai trò. Đừng gộp việc của máy này sang máy khác.

| | máy dev (hiện tại: Windows) | `ut-hpc` (train/fine-tune + test) | `vast-gpu` (chạy pipeline, thuê khi cần) |
|---|---|---|---|
| Phần cứng | Windows 11, không GPU NVIDIA dùng được cho DeepStream (repo giai đoạn đầu dev trên Apple M1) | Cụm SLURM ĐH Twente. Partition mở cho mọi account: `main-gpu` (26 node, A40 46GB / L40 48GB / Lovelace). `students` chỉ có 1 node `hpc-node08` | GPU rời thuê trên Vast.ai, IP/cấu hình đổi mỗi lần thuê |
| OS | Windows 11 (Git Bash + PowerShell) | Ubuntu 22.04.5 LTS | tuỳ instance lúc thuê — kiểm tra lại mỗi lần |
| Cách vào | local | `ssh ut-hpc`, việc nặng qua `sbatch --partition=main-gpu` — **không chạy trên head node**, và **node tính toán không có Internet**, xem mục "Cạm bẫy" | `ssh vast-gpu` (đổi `HostName`/`Port` trong `~/.ssh/config` mỗi lần thuê mới) |
| Container | — | Không cần: conda env trong `$HOME` chạy được GPU trên node tính toán (đã đo). Có `module load singularity/3.9.5` làm dự phòng, không có Docker | Docker (kiểm tra lại — tuỳ image Vast.ai) |
| Dùng để | soạn `src/common`, `src/mct`, `src/dashboard`, `src/tools`, `eval/`, `tests/` | **chạy `pytest`/`ruff`** (xem dưới); xuất ONNX; fine-tune trên dữ liệu tự thu ở M6 nếu đo được domain gap (KHÔNG fine-tune trên COCO/Market-1501/MSMT17, xem §9) | `src/ds_pipeline` — pipeline DeepStream thật, đo FPS/độ trễ end-to-end |
| KHÔNG chạy được | `src/ds_pipeline` | `src/ds_pipeline` (không có nvstreammux/DeepStream runtime, chỉ có CUDA/TensorRT để train) | — |

Quy trình thao tác chi tiết trên `ut-hpc` (SSH, `sbatch`, module load...) đóng gói trong skill
`.claude/skills/ut-hpc/` — đọc trước khi thao tác lần đầu, đừng đoán cú pháp.

**Chạy test và lint ở `ut-hpc`, không phải máy dev** (chốt 2026-09-04). Máy dev hiện tại là
Windows và chỉ có Python 3.13, trong khi repo nhắm 3.10 — test chạy ở đó không bao giờ bắt
được vi phạm quy tắc phiên bản. Head node `ut-hpc` có sẵn Python 3.10.12, **đúng bằng bản
trong container DeepStream 7.1**, nên vừa là chỗ chạy test vừa là chỗ kiểm chứng ràng buộc đó:

```bash
tar czf - src tests configs pyproject.toml | ssh ut-hpc 'tar xzf - -C ~/mct/repo'
ssh ut-hpc 'cd ~/mct/repo && PYTHONPATH=src ~/mct/venv-test/bin/python -m pytest -q'
ssh ut-hpc 'cd ~/mct/repo && ~/mct/venv-test/bin/ruff check src tests eval'
```

`~/mct/venv-test` là venv NHẸ (numpy/scipy/msgpack/redis/PyYAML/dotenv/pytest/ruff/fastapi,
không có torch) — chạy hết bộ test trong ~1.3 s nên không vi phạm quy tắc "không chạy gì
nặng trên head node". `ut-hpc` cần VPN vào mạng ĐH Twente; `ssh` timeout thì kiểm tra VPN
trước khi đoán là cụm hỏng.

`vast-gpu` là **thuê theo phiên, không thường trực** — instance bị huỷ khi ngừng thuê, IP đổi
mỗi lần thuê lại. Đừng giả định nó đang chạy; luôn xác minh (`ssh vast-gpu echo ok`) trước khi
dùng, và đừng để job chạy quên trên đó (tính tiền theo giờ).

**Mọi lệnh chạy trên `vast-gpu` — kể cả `ssh vast-gpu echo ok` để kiểm tra — phải hỏi xác nhận
người dùng trước khi thực thi.** Máy này tính phí theo giờ; ngay cả một lệnh "chỉ để kiểm tra"
cũng có thể đánh thức billing nếu instance đang tắt hoặc tình cờ khởi động lại. Áp dụng cho
`make ds-build`, `make ds-run`, và mọi thao tác đánh dấu "Trên máy GPU" ở mục 12.

**DeepStream không chạy trên macOS, và không chạy trên `ut-hpc` (không có DeepStream runtime,
chỉ dùng để train).** Đừng đề xuất chạy `src/ds_pipeline` ở hai nơi đó, đừng cài `pyds`/`tensorrt`
vào venv của máy dev, và đừng viết test cần GPU mà không đánh dấu skip.

### Quy tắc bất biến (vi phạm = hỏng cả quy trình dev)

1. **`src/common/`, `src/mct/`, `src/dashboard/`, `src/tools/`, `eval/` TUYỆT ĐỐI không được
   `import pyds`, `import gi` (GStreamer), `tensorrt`, `cuda`, hay bất kỳ thứ gì cần GPU.**
   Chỉ `src/ds_pipeline/` được phép. Có test tự động canh điều này (`tests/test_no_gpu_imports.py`).
2. Ranh giới giữa hai bên là **schema message trong `src/common/schema.py`** — không có kênh nào khác.
   Đổi schema là breaking change: phải cập nhật cả producer, consumer, và fixtures cùng lúc.
3. Mọi thứ chạy được ngoài máy GPU phải test được bằng **fixture ghi sẵn**, không cần Redis thật
   cũng không cần GPU. Fixture nằm ở `tests/fixtures/*.jsonl`.
4. **Nhắm Python 3.10** (bằng phiên bản trong container DeepStream 7.x / Ubuntu 22.04).
   Máy dev đang có 3.13 — không dùng cú pháp/thư viện chỉ có từ 3.11 trở lên trong code dùng
   chung; test chạy ở `ut-hpc` (3.10.12) mới là chỗ bắt được vi phạm.
5. **Trọng số model train trên `ut-hpc` phải chuyển sang `vast-gpu` qua `models/`** (gitignored,
   không qua git). `ut-hpc` không bao giờ chạy pipeline; `vast-gpu` không bao giờ train.

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
┌── Redis Streams ──┐   ← ranh giới duy nhất. Ghi lại được → replay được → dev không cần GPU.
└───────────────────┘
   │ XREADGROUP
   ▼
┌────────────────  src/mct/ (chạy đâu cũng được, không cần GPU) ─────────────┐
│ tracklet builder → gallery → affinity (cosine + spatio-temporal + homography) │
│                  → Hungarian → gán Global ID → SQLite                        │
└──────────────────────────────────────────────────────────────────────────────┘
   │ XADD mct:global + ghi SQLite
   ▼
src/dashboard/ (FastAPI + WebSocket): sơ đồ camera, vị trí hiện tại, tra cứu hành trình theo Global ID
```

**Vì sao tách bằng Redis Streams:** cho phép ghi lại luồng metadata thật từ máy GPU một lần,
rồi phát lại trên máy dev để phát triển và tinh chỉnh module liên kết đa camera — phần chiếm
tuần 11–13 và là đóng góp chính của đồ án — mà không cần ngồi cạnh máy GPU. Redis kiêm luôn
state store và persistence (replay được khi consumer chết).

## 4. Cấu trúc thư mục

```
configs/
  pipeline/      nvinfer/nvtracker config (.txt/.yml) + streams.yaml (danh sách nguồn camera)
  cameras/       topology.yaml (đồ thị camera + transit time), homography/<cam>.yaml
  demo/          cấu hình cho dataset MƯỢN (WildTrack) — tách khỏi cấu hình hệ thống thật
  mct.yaml       tham số association engine (ngưỡng, cửa sổ thời gian, trọng số)
src/
  common/        schema.py, streams.py (wrapper Redis), config.py, logging.py   ← dùng chung, KHÔNG GPU
  ds_pipeline/   builder.py, probes.py, reid_meta.py, sink.py                   ← CHỈ máy GPU
  mct/           tracklet.py, gallery.py, affinity.py, associator.py,
                 topology.py, homography.py, store.py
  dashboard/     app.py, live.py, static/, templates/
  tools/         replay_metadata.py, record_metadata.py, rtsp_sim.py,
                 calibrate_homography.py, cvat_to_mot.py, export_trackeval.py
eval/            run_trackeval.py, gt/ (ground-truth MOT format + bảng global_id)
tests/           fixtures/*.jsonl + test_*.py
docker/          deepstream.Dockerfile, compose.yml (máy dev), compose.gpu.yml (máy GPU)
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
   sau khi ánh xạ homography về mặt phẳng tham chiếu chung. `d_ground` so vị trí **tại
   cùng mốc thời gian** (trung vị trên các mốc chung của hai quỹ đạo, dung sai
   `ground_time_tol_ms`); không có mốc chung thì `ground_gap_policy` quyết định thả qua
   hay loại thẳng. Đo trên WildTrack 2026-09-04: so "điểm cuối ↔ điểm đầu" như thiết kế
   ban đầu làm kết quả TỆ ĐI, xem `docs/worklog/2026-09-04-5-*`.
4. **Hungarian** (`scipy.optimize.linear_sum_assignment`) trên ma trận đã mask,
   chạy **riêng cho từng camera**. Ghép một-một chỉ đúng trong phạm vi một camera; áp nó
   lên cả chiều liên camera thì một người xuất hiện ở N camera sẽ mất N−1 tracklet sang
   Global ID mới (đo được: recall 0.06 → 0.37 khi sửa).
5. Chấp nhận cặp gán nếu `cost < τ` (`configs/mct.yaml`); ngược lại **tạo Global ID mới**.
6. Cập nhật gallery của GlobalTrack (append có giới hạn kích thước + EMA), ghi SQLite.

**Hai chế độ, giữ cả hai:**
- `online` — chế độ thật, là sản phẩm bàn giao, dùng để đo độ trễ end-to-end.
- `offline` — chạy Hungarian theo lô trên toàn bộ tracklet đã hoàn tất. Không real-time.
  Vốn được coi là **cận trên** của độ chính xác, nhưng đo trên WildTrack 2026-09-04 cho
  kết quả NGƯỢC LẠI: online F1 0.929 vs offline 0.765. Lý do: ràng buộc hình học là hàm
  của thời gian, gán gần thời gian thực thì hai quỹ đạo mới trùng khoảng thời gian để so
  vị trí. Vẫn giữ cả hai chế độ để đối chiếu — chỉ đừng giả định chiều thắng thua trước
  khi đo (`eval/compare_online_offline.py`).

Mọi ngưỡng nằm trong `configs/mct.yaml`, **không hardcode trong code** — cần sweep tham số ở tuần 16–17.

> Đặc tả trên là thiết kế tại thời điểm viết tài liệu này. Nếu lệch với `src/mct/associator.py`
> hoặc `configs/mct.yaml` đang chạy thực tế, **code và config luôn đúng hơn tài liệu** — coi
> phần này là ngữ cảnh lịch sử để hiểu ý đồ ban đầu, không phải đặc tả sống. Cập nhật lại đoạn
> này nếu phát hiện lệch đáng kể.

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
- Test nào cần GPU/DeepStream: đánh dấu `@pytest.mark.gpu` để bộ test ngoài máy GPU skip được.

## 9. Lộ trình (bám theo chương 5 đề cương, 18–20 tuần)

| Mốc | Tuần | Nội dung | Máy |
|---|---|---|---|
| M0 | 1–2 | Khung repo, `schema.py`, wrapper Redis, `replay_metadata.py`, fixture tổng hợp, dashboard rỗng | máy dev |
| M1 | 3–4 | Pipeline DeepStream 1 camera: file/RTSP → YOLO (weight gốc, chưa fine-tune) → nvtracker → probe in ra console | `vast-gpu` |
| M2 | 5–7 | Detector: YOLO11s **pretrained COCO, không fine-tune** + lọc lớp person tại nvinfer → ONNX; trên `vast-gpu`: TensorRT engine, đo FPS, `nvstreammux` 3–4 luồng | `vast-gpu` |
| M3 | 8–10 | Re-ID: OSNet **pretrained, không fine-tune** (ưu tiên checkpoint đa nguồn/khái quát hoá miền); trên `vast-gpu`: tích hợp vào pipeline, publish Redis, **ghi fixture thật** | `vast-gpu` |
| M4 | 11–13 | **Module liên kết đa camera** (đóng góp chính) — phát triển bằng fixture M3 | máy dev |
| M5 | 14–15 | SQLite store + dashboard realtime | máy dev |
| M6 | 16–17 | Dataset tự thu + CVAT + TrackEval + sweep tham số; **chỉ ở đây mới cân nhắc fine-tune**, và làm dưới dạng ablation có/không fine-tune trên chính dữ liệu lab | cả ba |
| M7 | 18–20 | (tuỳ chọn) Jetson + viết báo cáo | — |

**Vì sao M2/M3 không còn bước fine-tune** (chốt 2026-09-04, xem
`docs/worklog/2026-09-04-7-m2-detector-pretrained.md`): weight pretrained của cả hai model
đã được huấn luyện trên **chính** những bộ mà lộ trình cũ định fine-tune lại — YOLO11s trên
COCO (lớp `person` nằm sẵn trong đó), OSNet của torchreid trên Market-1501/MSMT17. Fine-tune
trên tập con của dữ liệu model đã thấy không tạo được uplift đáng kể, chỉ tốn GPU-hour và
~50G đĩa cụm. Fine-tune chỉ có nghĩa khi có **domain gap đo được**, và đồ án đã có bằng chứng
định lượng cho hướng đó: trên WildTrack, checkpoint domain-generalization vượt checkpoint
Market-1501 chuyên biệt **+25% F1** (0.346 vs 0.277, phiên 4) — tức khái quát hoá miền ăn
đứt chuyên biệt hoá trong miền. Nên: dùng pretrained, đo trên dữ liệu tự thu ở M6, và **nếu**
thấy tụt rõ thì mới fine-tune trên chính dữ liệu lab — trình bày như một *ablation* (có/không
fine-tune), không phải một bước bắt buộc của pipeline chính.

Bảng trên là **kế hoạch tham chiếu**, không phải tiến độ thật.

**Trạng thái hiện tại:** xem 2–3 file mới nhất trong `docs/worklog/` (quy ước ở mục 10) —
không lặp lại ở đây vì sẽ lỗi thời ngay khi tiến độ đổi.

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
- **Đường lấy Re-ID embedding: đã chốt (A)** — ReID extractor tích hợp trong `nvtracker`
  (NvDCF), bật bằng khối `ReID:` trong config tracker; embedding ra qua user meta. Đường (B)
  — SGIE `nvinfer` thứ hai (`process-mode=2`, `output-tensor-meta=1`) chạy OSNet trên crop,
  đúng như đề cương mục 3.3 — giữ lại làm dự phòng, code vẫn đọc được cả hai.
  Mọi chi tiết phụ thuộc phiên bản gom trong `src/ds_pipeline/reid_meta.py`; API dùng ở đó
  tra từ binding `pyds` v1.2.0 (bản của DeepStream 7.1), **không đoán từ bản khác**.
- **`outputReidTensor: 1` là bắt buộc nếu muốn đọc embedding từ probe.** Thiếu nó, tracker
  vẫn trích đặc trưng và vẫn dùng nội bộ để tái liên kết, nhưng KHÔNG gắn vào user meta —
  probe đọc ra `None`, `src/mct` không có gì để so, và pipeline chạy "thành công" mà không
  sinh ra dữ liệu nào có ích. Lại một lỗi không triệu chứng.
- **NvMultiObjectTracker bỏ qua IM LẶNG khoá lạ hoặc khoá đặt sai khối** trong file config
  tracker. Đặt `reidExtractionInterval` nhầm khối thì ReID vẫn "chạy" mà tham số không có
  tác dụng. Sau mỗi lần sửa config tracker phải đọc log khởi động của nvtracker.
- **Toạ độ bbox theo streammux, không phải theo camera** — xem mục 5.
- **Đồng bộ thời gian giữa các camera là điều kiện sống còn** cho ràng buộc thời gian di chuyển.
  Bật NTP trên mọi nguồn; điện thoại Android phát RTSP thường lệch — đo và ghi lại offset.
- **Điện thoại Android phát RTSP** (đề cương mục 4.1.2) có latency và jitter cao hơn camera IP.
  Đừng dùng nó cho cặp camera overlap cần homography chính xác.
- **VRAM**: chạy đồng thời YOLO + Re-ID trên 4 luồng dễ chạm trần trên GPU 8GB.
  Giảm `batch-size` của SGIE trước, rồi mới giảm độ phân giải suy luận.
- **Quyền riêng tư** (đề cương mục 4.3.3): dữ liệu người thật chỉ dùng cho học thuật, có đồng thuận,
  không commit vào git (`data/` đã ignore), làm mờ mặt trước khi đưa vào báo cáo/slide.
- **`ut-hpc`: node tính toán KHÔNG có Internet, head node CÓ.** Đã đo 2026-09-03: từ `ctit091`,
  `curl` tới pypi/github/docker registry đều không phản hồi; từ `hpc-head1` đều 200. Mọi thứ
  chạm mạng (pip, tải dataset, weight tự tải của Ultralytics, `singularity pull`) phải làm trên
  head node trước; job chỉ đọc từ đĩa. Đây là cạm bẫy số 1 — job có lệnh gọi mạng sẽ treo tới
  hết `--time` rồi chết.
- **`ut-hpc` không có Docker.** `docker/deepstream.Dockerfile` không dùng được ở đây (mà cũng
  không cần — `ut-hpc` không chạy DeepStream). Không cần container để fine-tune: conda env đặt
  trong `$HOME` (NFS) chạy được CUDA trên node tính toán, đã xác minh bằng job thật. Giữ
  `module load singularity/3.9.5` làm phương án dự phòng.
- **`ut-hpc` là cụm dùng chung — không chạy gì nặng trên head node (`hpc-head1`).** Mọi việc cần
  GPU phải qua `sbatch --partition=main-gpu --gres=gpu:...`. Dùng `main-gpu` (26 node,
  `AllowAccounts=ALL`), **không** dùng `students` (đúng 1 node `hpc-node08`, hay pending):
  đo 2026-09-03, `main-gpu` chờ ~40s trong khi `students` pending vô hạn định.
  `srun` foreground dễ bị treo chờ hàng đợi khi node đang bận —
  ưu tiên `sbatch` (job không đồng bộ) cho việc chạy lâu, dùng `squeue -u $USER` để theo dõi thay
  vì đoán thời gian chờ.
- **`vast-gpu` thuê theo phiên** — không giả định nó đang tồn tại. `~/.ssh/config` phải cập nhật
  `HostName`/`Port` mỗi lần thuê instance mới. Kiểm tra Docker và phiên bản driver ngay khi thuê,
  đừng giả định giống lần trước.
- **Home trên `ut-hpc` còn ~113G/1TB (2026-09-03).** Kiểm tra `df -h $HOME` trước mỗi lần tải
  dataset. Từ 2026-09-04 áp lực này giảm hẳn: bỏ fine-tune trên COCO/MSMT17 nghĩa là không
  cần ~50G dataset benchmark trên cụm nữa.
- **Trọng số model không đi qua git.** Fine-tune xong trên `ut-hpc`, chép `.onnx`/`.pt` sang
  `models/` (gitignored) rồi rsync/scp sang `vast-gpu` khi cần chạy pipeline.

## 12. Lệnh (tên chuẩn hoá cho Makefile — tham chiếu, không phải cam kết đã cài đặt đủ)

```bash
# Trên máy dev — không cần GPU
make dev                # cài deps CPU (pip install -e ".[dev]")
make test               # pytest, bỏ qua test có mark gpu
make lint               # ruff check + ruff format --check
make up                 # docker compose: redis + mct-engine + dashboard
make replay FIXTURE=tests/fixtures/two_cam_walk.jsonl   # phát lại metadata vào Redis
make eval               # chạy TrackEval trên kết quả trong eval/

# ⚠️ Trên máy GPU (vast-gpu) — xác nhận với người dùng trước khi chạy, tính phí theo giờ (mục 2)
make ds-build           # build docker/deepstream.Dockerfile
make ds-run             # chạy pipeline theo configs/pipeline/streams.yaml
make ds-run-reid        # 4 luồng CÓ ReID (streams_reid.yaml) — đối chứng của streams_multi.yaml
make record OUT=tests/fixtures/<tên>.jsonl              # ghi Redis stream ra fixture
```
