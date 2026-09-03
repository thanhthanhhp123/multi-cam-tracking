# 2026-09-03 (phiên 3) — M1: pipeline DeepStream 1 camera trên vast.ai

- **Mốc:** M1 (tuần 3–4) | **Máy:** máy dev (Windows) + `vast-gpu` (RTX 3090, Bỉ) | **Thời lượng:** ~2.5h

## Mục tiêu phiên

- Thuê GPU trên Vast.ai, xác minh môi trường DeepStream 7.x thật.
- Dựng pipeline 1 camera: file → YOLO (weight gốc COCO) → nvtracker → probe in console
  (đúng mục tiêu M1, CLAUDE.md §9).
- Trả lời hai câu hỏi để ngỏ ở CLAUDE.md §11: phiên bản DeepStream chốt, đường lấy Re-ID.

## Đã làm

- **Thuê instance `49751272`** trên Vast.ai qua CLI `vastai` (cài sẵn nhưng chưa có API
  key — 2 key đầu người dùng dán thiếu quyền do tài khoản bật 2FA, key thứ 3 mới đúng
  quyền). Đăng ký SSH key riêng `~/.ssh/id_ed25519_vast` (không dùng chung với các host
  khác trong `~/.ssh/config`). Offer chọn: RTX 3090 24GB, Bỉ, $0.188/h, image
  `nvcr.io/nvidia/deepstream:7.1-triton-multiarch`.
- Xác minh môi trường thật: Ubuntu 22.04.4, Python 3.10.12 (khớp CLAUDE.md §2 quy tắc 4),
  DeepStream 7.1.0, CUDA 12.6, TensorRT 10.3.0.26, driver 565.57.01. Plugin cần dùng đều có
  (nvurisrcbin, nvstreammux, nvinfer, nvtracker, nvvideoconvert, nvdsosd).
- Cài `pyds` (bản 1.2.0-cp310 — khớp DS 7.1/Py3.10, KHÔNG dùng 1.2.2-cp312 là cho DS 8.0).
- Clone `DeepStream-Yolo` (MIT, marcoslucianops) làm custom bbox parser cho YOLO11 —
  DeepStream gốc không có parser cho model Ultralytics. Build native lib
  `libnvdsinfer_custom_impl_Yolo.so` bằng `CUDA_VER=12.6 make`.
- Export `yolo11s.pt` (weight gốc COCO, 80 lớp, chưa fine-tune — đúng phạm vi M1) sang
  ONNX bằng `DeepStream-Yolo/utils/export_yolo11.py` (cần cài thêm `onnxscript` — script
  gốc không khai báo dependency này).
- Viết code thật cho `src/ds_pipeline/`: `builder.py` (dựng Gst pipeline từ
  `configs/pipeline/streams.yaml`), `probes.py` (đọc NvDsBatchMeta, scale bbox về độ
  phân giải camera gốc, đóng gói FrameMessage theo `common/schema.py`), `__main__.py`
  (entrypoint `python -m ds_pipeline`, có cờ `--publish` để đẩy Redis thay vì in console).
- Viết config: `configs/pipeline/streams.yaml`, `configs/pipeline/config_infer_yolo11.txt`,
  `configs/pipeline/config_tracker_NvDCF_perf.yml` — viết lại bằng tên tham số công khai,
  không sao chép số liệu đã tune của config mẫu NVIDIA (header ghi
  `LicenseRef-NvidiaProprietary ... strictly prohibited`, không phù hợp đưa vào git repo).
- Đồng bộ code lên instance (scp, không có rsync trong image), `pip install -e .`,
  chạy thật pipeline end-to-end: xử lý hết 1442 frame video mẫu 1080p, local_track_id
  bám object nhất quán qua frame, kết thúc sạch bằng EOS.
- Viết `docker/deepstream.Dockerfile` + `docker/compose.gpu.yml` để đóng gói lại toàn bộ
  các bước thủ công trên thành build tái lập được trên máy GPU hoàn toàn mới (image nền
  chính thức NVIDIA, không cài native — CLAUDE.md §11 rủi ro số 1). Chưa `docker build`
  thật được (máy dev không có Docker daemon chạy + không có GPU để test runtime) — các
  lệnh bên trong đều là lệnh đã chạy thành công thật trên instance, không phải suy đoán.

## Quyết định kỹ thuật

**1. DeepStream 7.1.0 / CUDA 12.6 / TensorRT 10.3.0.26 — chốt, không dùng DS 8.0.**
DS 8.0 chạy Ubuntu 24.04 + Python 3.12, phá quy tắc bất biến "nhắm Python 3.10" (CLAUDE.md
§2 quy tắc 4). DS 7.1 khớp Python 3.10.12 hệ thống, không cần venv riêng cho phần pipeline.

**2. Đường lấy Re-ID: (A), không cần SGIE dự phòng.** `config_tracker_NvDCF_accuracy.yml`
mẫu của NVIDIA có sẵn khối ReID với `reidType: 2`, `reidFeatureSize: 256`,
`addFeatureNormalization: 1` (tracker tự L2-normalize — khớp yêu cầu contract CLAUDE.md
§5 miễn phí). Bỏ được cả một tầng SGIE thứ hai khỏi pipeline. Model
`resnet50_market1501.etlt` mẫu chưa có sẵn trong image (thư mục `samples/models/Tracker/`
không tồn tại) — phải tải riêng hoặc dùng OSNet fine-tune từ `ut-hpc`, việc của M3.

**3. DeepStream-Yolo (bên thứ ba, MIT) thay vì SGIE nvinfer tự viết parser.** DeepStream
gốc không có bbox parser cho output YOLO11/Ultralytics. Viết parser CUDA từ đầu tốn nhiều
tuần, ngoài phạm vi đóng góp chính của đồ án (module liên kết đa camera). DeepStream-Yolo
là công cụ cộng đồng chuẩn cho việc này, MIT license — an toàn để đưa vào Docker image.
Native lib build từ source (kiến trúc-cụ thể) không đi qua git — thư mục `third_party/`
thêm vào `.gitignore`, dựng lại bằng Dockerfile.

**4. Không sao chép nguyên văn config tracker mẫu của NVIDIA vào git.** File mẫu
(config_tracker_NvDCF_*.yml trong container) ghi rõ
`SPDX-License-Identifier: LicenseRef-NvidiaProprietary` và "reproduction ... strictly
prohibited" khi không có thoả thuận license riêng. Viết lại bằng tên tham số (công khai,
tài liệu NvMultiObjectTracker) với giá trị mặc định hợp lý, chưa tune — sweep tham số thật
là việc của M6.

**5. `model-engine-file` của nvinfer bị custom engine-create-func của DeepStream-Yolo bỏ
qua khi GHI** (vẫn tôn trọng khi ĐỌC). Engine TensorRT luôn được serialize ra
`<cwd>/model_b<N>_gpu<M>_<precision>.engine`, bất kể đường dẫn khai trong config. Cách né:
chạy lần đầu, rồi copy engine sinh ra vào đúng đường dẫn config kỳ vọng
(models/detector/<tên>.onnx_b1_gpu0_fp16.engine) — các lần sau nvinfer đọc đúng chỗ và
bỏ qua bước build lại (tiết kiệm ~4 phút mỗi lần khởi động). Đáng ghi vào báo cáo chương 5
như một cạm bẫy cụ thể của công cụ, không phải lỗi cấu hình.

## Số liệu đo được

Cấu hình: Vast.ai instance `49751272`, RTX 3090 24GB (driver 565.57.01), DeepStream 7.1.0,
CUDA 12.6, TensorRT 10.3.0.26, Ubuntu 22.04.4, Python 3.10.12. Model: YOLO11s, weight gốc
COCO (80 lớp, chưa fine-tune), FP16, batch-size=1. Tracker: NvDCF (perf profile tự viết,
chưa tune), ReID tắt (reidType: 0). Nguồn: sample_1080p_h264.mp4 (1920x1080, file, không
phải RTSP live), streammux 1920x1080, 1 luồng.

| Đại lượng | Giá trị |
|---|---|
| Frame xử lý | 1443 |
| Tổng detection (person) | 5841 |
| Thời gian build TensorRT engine (lần đầu) | ~4 phút |
| Thời gian chạy hết video, đã cache engine | 3.51 s |
| Throughput | 410.8 FPS (1 luồng, không phải luồng real-time — nguồn file, sync=0) |

So với mục tiêu đề cương (3–4 luồng, >=15 FPS/luồng, <1s độ trễ end-to-end): con số này là
trần lý thuyết của riêng phần suy luận + tracking, đo trên nguồn file không giới hạn
tốc độ đọc — không phải FPS thực khi có 3–4 luồng RTSP đồng thời (I/O, decode, batch mux sẽ
kéo xuống). Cần đo lại với live-source=true + nhiều luồng RTSP thật (hoặc mô phỏng qua
tools/rtsp_sim.py, còn ở dạng rỗng) trước khi đưa vào chương 6 như số cuối cùng.

## Vướng mắc / chưa xong

- API key Vast.ai, 2 lần đầu bị 401 dù đúng định dạng — do tài khoản bật 2FA và key
  sinh ra ngoài phiên đã xác thực 2FA bị hạ quyền (chặn cả `show user` chỉ đọc). Sửa bằng
  cách đăng nhập lại qua 2FA rồi tạo key mới trong đúng phiên đó.
- Chưa fine-tune YOLO — weight vẫn là gốc COCO 80 lớp. M2 (COCO-person, ut-hpc) chưa
  làm trong phiên này.
- Chưa bật ReID — chưa có model resnet50_market1501 hay OSNet fine-tune trong
  models/reid/. embed_dim=0 trong mọi message ở phiên này.
- Chưa --publish lên Redis thật — mới test bằng _print_sink (console). Cần Redis
  chạy trên cùng máy GPU hoặc mở port để test --publish + tools/record_metadata.py.
- Chưa test với nguồn RTSP thật (live-source=true) — chỉ test với file. Độ trễ
  end-to-end và FPS thực với I/O mạng chưa đo.
- docker build chưa chạy thật — máy dev không có Docker daemon đang chạy (Windows,
  Docker Desktop chưa bật) và không có GPU để test runtime --gpus. Mỗi lệnh RUN trong
  Dockerfile là lệnh đã chạy thành công thật trên instance hôm nay, nhưng chưa build được
  cả image liền mạch — rủi ro nhỏ về thứ tự layer/cache. Cần build thử trên vast-gpu
  (hoặc máy GPU có Docker) ở phiên sau trước khi tin tưởng hoàn toàn.
- gst-plugin-scanner cảnh báo lặp lại mỗi lần chạy: libnvdsgst_udp.so:
  librivermax.so.1 not found — plugin SMPTE 2110, không liên quan RTSP, vô hại, chưa dọn.
- Instance 49751272 đang chạy tại thời điểm ghi worklog này — cần quyết định giữ hay
  huỷ (vastai destroy instance 49751272 -y) trước khi kết thúc phiên (tính tiền theo giờ).

## Bước tiếp theo

1. Build thử `docker/deepstream.Dockerfile` trên máy GPU thật (vast-gpu hoặc instance mới)
   để xác nhận Dockerfile tái lập đúng môi trường đã dựng thủ công hôm nay.
2. Đo FPS với live-source=true + RTSP thật hoặc mô phỏng nhiều luồng — số hiện tại chỉ
   là trần lý thuyết một luồng đọc từ file.
3. M2: fine-tune YOLO trên COCO-person qua `.claude/skills/ut-hpc/` (chưa chạy `hpc.sh
   setup` — xem worklog 2026-09-03 phiên 1).
4. Bật --publish, dựng Redis trên máy GPU, tools/record_metadata.py để ghi fixture
   thật đầu tiên từ pipeline (dù ReID vẫn tắt) — đối chiếu với fixture WildTrack hiện có.
