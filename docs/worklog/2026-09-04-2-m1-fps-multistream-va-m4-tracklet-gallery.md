# 2026-09-04 (phiên 2) — M1 đóng lại: FPS 4 luồng đo thật; M4 khởi động: tracklet + gallery + topology

- **Mốc:** M1 (đóng phần còn treo) + M4 (tuần 11–13, bắt đầu) | **Máy:** máy dev (Windows) + `vast-gpu` (RTX 3090, Czechia) | **Thời lượng:** ~2.5h

## Mục tiêu phiên

- Thuê lại vast.ai, làm nốt hai việc treo từ 2026-09-03 phiên 3: xác nhận
  `docker/deepstream.Dockerfile` tái lập được, và đo FPS nhiều luồng.
- Bắt đầu M4 trên máy dev (không phụ thuộc GPU): `src/mct/tracklet.py`.

## Đã làm

### Máy dev — M4 (đóng góp chính của đồ án)

- **Dựng lại môi trường dev trên Windows.** Máy này chưa từng cài deps (repo phát triển
  trên Mac trước đó). Tạo `.venv` và `pip install -e ".[dev]"` với
  `--ignore-requires-python`: máy chỉ có Python 3.13 trong khi `pyproject.toml` chốt
  `>=3.10,<3.12`. **Cảnh giác:** test chạy trên 3.13 sẽ KHÔNG bắt được việc lỡ dùng cú
  pháp/thư viện chỉ có từ 3.11 trở lên (CLAUDE.md §2 quy tắc 4) — chỗ duy nhất kiểm được
  điều đó hiện nay là container DeepStream (Python 3.10.12).
- `src/mct/tracklet.py` + 19 test — gom `FrameMessage` thành tracklet cục bộ, query
  embedding = trung bình có trọng số top-k confidence cao nhất.
- `src/mct/gallery.py` + 13 test — vòng đời `GlobalTrack`, gallery embedding có hạn ngạch
  theo camera, centroid EMA, TTL, ràng buộc loại trừ cùng camera.
- `src/mct/topology.py` + 23 test — nạp `configs/cameras/topology.yaml`, kiểm ràng buộc
  thời gian di chuyển, trả kèm lý do loại.
- Thêm vào config: `tracklet.max_embeddings`, `gallery.similarity_mode` (`configs/mct.yaml`),
  `unknown_pair_policy` (`configs/cameras/topology.yaml`).
- Toàn bộ: 136 passed, 5 skipped; `ruff check` + `ruff format` sạch.

### `vast-gpu` — đóng phần treo của M1

- Instance đầu (`49804063`, offer `43824378`, RTX 3090, US, host `155125`, 1185 Mbps) **kẹt
  lại y hệt hôm qua**: 11 phút không thoát khỏi dòng `7.1-triton-multiarch: Pulling from
  nvidia/deepstream` (hôm qua ít nhất còn nhảy được vài layer). Huỷ theo đúng quy tắc đã
  đặt ra hôm qua (≥10 phút không tiến triển thì huỷ).
- Instance thứ hai (`49804885`, offer `35740192`, RTX 3090 24GB, Czechia, host `151822`,
  **4948 Mbps**, driver 590.48.01, $0.187/h): **running sau 3 phút 15 giây**, pull chạy
  mượt qua từng layer.
- **Xác nhận Dockerfile tái lập được:** chạy tuần tự đúng các lệnh `RUN` của
  `docker/deepstream.Dockerfile` trên instance mới tinh (script
  `scratchpad/build_steps.sh`) — `user_additional_install.sh`, cài `pyds` 1.2.0-cp310,
  clone DeepStream-Yolo, cài ultralytics/onnx/onnxslim/onnxscript/onnxruntime, build
  `libnvdsinfer_custom_impl_Yolo.so`, `pip install -e .` — **tất cả xanh trong 2 phút 8
  giây**, `import pyds, ds_pipeline, common.schema` chạy được.
- Export lại `yolo11s.pt` → ONNX, chạy pipeline 1 luồng và 4 luồng, đo FPS + GPU/VRAM.
- **Chạy được đường `--publish` lần đầu:** cài `redis-server` trên chính instance, chạy
  pipeline với `--publish`, `tools/record_metadata.py` ghi ra fixture JSONL. 5772/5772
  message qua Redis, không mất frame nào, `validate()` báo 0 vấn đề.
- Kéo về máy dev: `tests/fixtures/ds_4cam_sample_noreid.jsonl` (chạy hết tốc lực) và
  `ds_4cam_sample_realtime.jsonl` (phát đúng tốc độ thật, 46.3 s), cùng
  `models/detector/yolo11s.onnx{,.data}` + `labels.txt` (đều gitignored).
- **Chạy `tracklet.py` trên fixture thật ngay trong phiên:** 5772 message → 128 tracklet,
  0 lỗi validate — module M4 viết sáng nay ăn khớp với đầu ra pipeline thật, không chỉ với
  fixture tổng hợp.
- Huỷ instance sau khi xong. Credit còn $7.743 (cả phiên tốn ~$0.09).

## Quyết định kỹ thuật

**1. Chọn offer vast.ai theo `inet_down`, không theo giá.** Hai lần kẹt liên tiếp đều rơi
vào host ~1.2 Gbps; host 4.9 Gbps pull xong trong 3 phút. Image DeepStream 7.1-triton nặng
(~20GB), nên tiền tiết kiệm được khi chọn host rẻ nhất bị chi phí chờ (và rủi ro kẹt hẳn)
nuốt hết. Từ nay lọc `inet_down>=2500` trước rồi mới xét giá — chênh lệch $0.07/h là không
đáng kể so với 30 phút chờ.

**2. Đo FPS phải tách khỏi việc ghi log — thêm cờ `--stats`.** Cùng pipeline, cùng máy,
1 luồng: **410.8 FPS** khi in log mỗi frame (số của worklog 2026-09-03) so với
**618.5 FPS** khi chỉ đếm. Tức con số 410.8 công bố hôm trước đo lẫn cả chi phí
`log.info()` — chênh 50%. Ghi lại chỗ này vào chương 6 như một lưu ý phương pháp đo, và
mọi số FPS từ nay đều phải đo bằng `--stats`. Đồng hồ trong `_StatsSink` tính từ message
đầu tiên, không tính từ `set_state(PLAYING)`, để không gộp ~4 phút build engine TensorRT
vào phép đo.

**3. Cần một file config nvinfer riêng cho mỗi batch size (`config_infer_yolo11_b4.txt`).**
Engine TensorRT gắn chặt với batch size, và nvinfer chỉ nạp lại được engine đã build khi
`model-engine-file` trỏ đúng tên file. Không có file riêng thì mỗi lần chạy 4 luồng phải
build lại engine ~4 phút. `builder.py` vẫn tự đặt thuộc tính `batch-size` = số nguồn lúc
chạy, nên giá trị `batch-size` trong file config chỉ mang tính tài liệu — thứ thực sự phải
khớp là **tên file engine**. (Đây là hệ quả trực tiếp của cạm bẫy đã ghi hôm trước: custom
engine-create-func của DeepStream-Yolo bỏ qua `model-engine-file` khi GHI, nhưng tôn trọng
khi ĐỌC.)

**4. `sink.sync` — nguồn file phải phát đúng tốc độ thật khi GHI FIXTURE.** `attach_sys_ts`
gắn wall clock lúc *xử lý*, nên chạy hết tốc lực nén 46 giây video vào **4.7 giây** `ts_ms`.
Fixture như vậy dùng được để test gom tracklet nhưng **vô dụng cho ràng buộc thời gian di
chuyển** (`src/mct/topology.py` làm việc với đơn vị giây). Ngược lại, đo FPS thì bắt buộc
`sync: false`, nếu không chỉ đo được đúng tốc độ phát của video (31.2 FPS = fps của file).
Giữ cả hai fixture để thấy rõ khác biệt.

**5. Không dùng Docker trên instance vast.ai — và không cần.** Instance vast.ai *bản thân
nó* là container, không có Docker daemon bên trong (`docker: command not found`), nên
`docker build` không chạy được ở đây; đây là giới hạn của nền tảng chứ không phải của
Dockerfile. Cách kiểm gần nhất — chạy lại đúng chuỗi lệnh `RUN` trên image nền y hệt —
đã xanh hết, nên rủi ro còn lại chỉ nằm ở ngữ nghĩa riêng của Docker (`COPY`, `WORKDIR`,
thứ tự layer), không nằm ở nội dung các bước.

**6. Hạn ngạch gallery chia theo camera trước, confidence sau.** Nếu chỉ xếp hạng theo
confidence, một người đứng lâu trước cam01 sẽ đẩy hết embedding chụp ở cam02 ra khỏi
gallery — đúng lúc cần chúng nhất để khớp với cam03. Có test riêng cho tình huống này
(`test_camera_hiem_gap_khong_bi_day_khoi_gallery`).

**7. `unknown_pair_policy: allow` làm mặc định.** Cặp camera chưa đo transit time thì thả
tự do, chỉ dựa vào ngoại hình. Siết mặc định (`reject`) sẽ khiến engine im lặng không khớp
gì khi ai đó thêm camera mới mà quên khai `transitions` — một lỗi không có triệu chứng.
Chuyển sang `reject` khi đã đo đủ mọi cặp (M6).

## Số liệu đo được

**Cấu hình chung:** Vast.ai instance `49804885`, RTX 3090 24GB (driver 590.48.01),
DeepStream 7.1.0, CUDA 12.6, TensorRT 10.3.0.26, Ubuntu 22.04, Python 3.10.12.
Model YOLO11s, **weight gốc COCO 80 lớp, chưa fine-tune**, FP16, input 640.
Tracker NvDCF (profile perf tự viết, chưa tune), **ReID TẮT** (`reidType: 0`, `embed_dim=0`).
Nguồn: `sample_1080p_h264.mp4` (1920×1080, 1443 frame) lặp lại cho từng luồng; streammux
1920×1080; `sink.sync=false` (không giới hạn tốc độ đọc) trừ dòng cuối bảng.

| Cấu hình | Frame | Thời gian | FPS gộp | FPS/luồng |
|---|---|---|---|---|
| 1 luồng, có log mỗi frame (đo 2026-09-03) | 1443 | 3.51 s | 410.8 | 410.8 |
| 1 luồng, `--stats` | 1443 | 2.33 s | 618.5 | 618.5 |
| 4 luồng, `--stats`, engine build lần đầu | 5772 | 7.74 s | 745.6 | 186.4 |
| 4 luồng, `--stats`, engine đã cache | 5772 | 7.62 s | 757.6 | 189.4 |
| 4 luồng, `--stats --publish` (qua Redis) | 5772 | 7.60 s | 759.5 | 189.8 |
| 4 luồng, `sink.sync=true` (phát tốc độ thật) | 5772 | 46.3 s | 124.8 | 31.2 |

**GPU trong lúc chạy 4 luồng** (`nvidia-smi -lms 200`, n=54 mẫu, gồm cả lúc khởi động):
utilization đỉnh **62%**, trung bình 41%; VRAM đỉnh **769 MiB**, trung bình 676 MiB.

**Thời gian dựng môi trường:** chạy hết chuỗi `RUN` của Dockerfile trên instance mới:
**2 phút 8 giây**. Build engine TensorRT FP16 lần đầu: ~3 phút 48 giây (batch 1),
~4 phút (batch 4). Pull image DeepStream 7.1-triton: 3 phút 15 giây trên host 4.9 Gbps
(host 1.2 Gbps: kẹt, không xong sau 11 phút).

**Đối chiếu với mục tiêu đề cương (3–4 luồng, ≥15 FPS/luồng, <1 s độ trễ):** 189 FPS/luồng
với 4 luồng — **vượt xa** ngưỡng 15 FPS/luồng, và GPU mới dùng 62%/769 MiB. Nhưng đây vẫn
là **trần**: nguồn file đọc không giới hạn tốc độ, chưa có ReID (M3 sẽ thêm một tầng trích
đặc trưng), chưa có I/O mạng RTSP, và độ trễ end-to-end vẫn **chưa đo** (cần đồng hồ ở hai
đầu, chưa làm). Utilization 62% cho thấy nút thắt hiện nằm ở phía CPU (decode + callback
Python trong probe), không phải ở GPU — đáng ghi vào chương 6 vì nó dự báo chỗ sẽ đụng trần
khi tăng số luồng.

**Fixture thật đầu tiên từ pipeline** (`tests/fixtures/ds_4cam_sample_realtime.jsonl`,
gitignored): 5772 message, 4 camera, span 46.3 s, 23380 detection, `validate()` 0 vấn đề.
Chạy `mct.tracklet.build_tracklets` (min_frames=5, idle_timeout=2000 ms) trên nó: **128
tracklet**, trung vị 61.5 frame/tracklet, trung vị 3.35 s, dài nhất 27.9 s.

**Đo trên fixture tổng hợp** (seed 42, embed_dim 256, `cross_cam_sim=0.75`,
`inter_sim=0.65`, top-k=8): similarity giữa hai query embedding của cùng một người ở hai
camera = **0.714–0.735**; cặp "mặc đồ giống nhau" = **0.615–0.620**; hai người khác hẳn ≈ 0.
Trung bình top-k tiến *từ dưới lên* về đúng tham số kịch bản chứ không vượt qua. Với
`association.max_cost = 0.30` (tức yêu cầu cosine > 0.70), biên an toàn chỉ còn **0.014** —
tham số nhạy nhất của hệ thống, cần sweep ở M6.

## Vướng mắc / chưa xong

- **Độ trễ end-to-end chưa đo** — mới đo throughput. Cần mốc thời gian ở cả hai đầu
  (frame vào pipeline → Global ID ra khỏi `mct`), làm sau khi associator chạy được.
- **Chưa test nguồn RTSP thật** (`live_source: true`). `tools/rtsp_sim.py` vẫn rỗng; số
  4 luồng ở trên là proxy bằng file.
- **ReID vẫn tắt** — `embed_dim=0` trong mọi fixture ghi được. Không có model
  `resnet50_market1501` trong image; phải chờ OSNet fine-tune từ `ut-hpc` (M3).
- **Dockerfile không ghim phiên bản pip package.** Lần này kéo về ultralytics 8.4.138 /
  onnx 1.22 / torch 2.14, và bước simplify ONNX ném
  `No Adapter To Version $17 for Resize` rồi rơi về export dạng external data
  (`yolo11s.onnx` + `yolo11s.onnx.data`). Engine vẫn build và chạy đúng, nhưng đây là
  phiên bản trôi tự do — nên ghim version trong Dockerfile trước khi coi image là ổn định.
- **Test đang chạy trên Python 3.13** (máy dev Windows chỉ có bản này) trong khi repo nhắm
  3.10. CI thật sự cho quy tắc này vẫn chưa có.
- `docker build` vẫn **chưa chạy thật một lần nào** — xem quyết định 5 ở trên về lý do và
  mức rủi ro còn lại.

## Bước tiếp theo

1. `src/mct/affinity.py` — ma trận chi phí `1 − cosine` + mask theo `topology.check()`,
   rồi `src/mct/associator.py` (Hungarian + ngưỡng `max_cost` + tạo Global ID mới).
   Đủ nguyên liệu rồi: tracklet, gallery, topology đều đã có và đã test.
2. Chạy associator trên cả hai fixture: tổng hợp (có ground truth) và WildTrack
   (`tools/wildtrack_to_fixture.py`, có embedding thật) — đo IDF1/HOTA sơ bộ.
3. Ghim phiên bản pip trong `docker/deepstream.Dockerfile` (ultralytics, onnx, torch)
   theo đúng bộ đã chạy được hôm nay.
4. Khi thuê vast.ai lần sau: lọc offer `inet_down>=2500` trước khi xét giá; nếu
   `status_msg` không tiến triển sau ~10 phút thì huỷ và đổi host ngay.
