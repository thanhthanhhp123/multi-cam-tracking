# 2026-09-04 (phiên 9) — Xác minh M2/M3 trên phần cứng thật, đo FPS/VRAM/độ trễ

- **Mốc:** M2 + M3 (đóng phần treo) + M5 (độ trễ end-to-end) | **Máy:** `vast-gpu` (RTX 3090) + máy dev | **Thời lượng:** ~1.5h, **máy GPU chạy 60 phút, tốn $0.165**

## Mục tiêu phiên

Gộp ba việc treo của ba phiên vào một chuyến thuê máy:

1. Phiên 8 (M3): đường (A) đọc embedding từ nvtracker có chạy không; `reidExtractionInterval`
   có đúng khối không; chi phí FPS của ReID.
2. Phiên 7 (M2): khối `[class-attrs-*]` lọc lớp person có thật sự hoạt động không.
3. Phiên 6 (M5): độ trễ end-to-end thật, và ghi fixture thật CÓ embedding.

## Đã làm

- **Tự thuê instance** bằng `vastai` CLI (API key đã có sẵn trên máy dev). Lọc offer theo
  `inet_down >= 2500` trước rồi mới xét giá, đúng quyết định phiên 2 → offer `47410737`,
  RTX 3090 24GB, **7535 Mbps**, $0.1614/h. Pull image xong sau **~2.5 phút** (hai phiên
  trước kẹt 11–28 phút trên host ~1.2 Gbps). Huỷ máy ngay sau khi xong, credit còn $7.576.
- **`docker/vast_bootstrap.sh`** (mới) — dựng môi trường DeepStream native trên instance,
  idempotent, phản chiếu đúng chuỗi `RUN` của Dockerfile. Phiên 2 làm việc này bằng script
  trong scratchpad rồi để mất; giờ nằm trong repo cạnh Dockerfile.
- **`src/tools/measure_latency.py`** (mới) — đọc `mct:global`, lấy `now − GlobalUpdate.ts_ms`
  làm độ trễ chuỗi probe → Redis → engine → Redis.
- **Sửa lỗi `topology.check()` giết engine** (xem Quyết định 3) + 4 test.
- Cập nhật `configs/pipeline/config_tracker_NvDCF_reid.yml` (đường dẫn ONNX + kết quả đã
  xác minh), `CLAUDE.md` §11 (4 cạm bẫy mới), `.gitattributes` (ghim `configs/**` = LF).
- Kéo về `tests/fixtures/ds_4cam_reid_realtime.jsonl` — 68 MB, gitignored.

## Quyết định kỹ thuật

**1. `nvinfer` và `nvtracker` phân giải đường dẫn tương đối khác nhau — phải biết mà tránh.**
Tôi đã cho `onnxFile` đường dẫn `../../models/...`, tương đối so với **file config**, đúng
quy ước mà `config_infer_yolo11.txt` đang dùng. nvtracker không dùng quy ước đó: nó lấy gốc
là **thư mục làm việc** của tiến trình. Kết quả: `!![ERROR] ONNX file does not exist` và cả
pipeline chết ngay lúc khởi động. Chốt: config tracker dùng đường dẫn tương đối so với gốc
repo (`models/reid/...`), và ghi rõ sự khác biệt này ở cả hai file để lần sau không lẫn.

Đây là loại lỗi mà không có phần cứng thì không cách nào phát hiện — test cấu hình của phiên
8 chỉ khẳng định file *nói* đúng điều ta muốn, không khẳng định DeepStream *hiểu* như vậy.

**2. "Không thấy cảnh báo" không phải bằng chứng — phải có thí nghiệm bác bỏ được.**
NvMultiObjectTracker bỏ qua im lặng khoá lạ hoặc khoá đặt sai khối, nên grep log tìm cảnh báo
là vô nghĩa. Nghi vấn lớn nhất của phiên 8 là `reidExtractionInterval` có đúng nằm trong
`TrajectoryManagement` không. Cách kiểm: đổi giá trị sang mức cực đoan và xem hành vi có đổi.
`0 → 100000` làm độ phủ embedding sập từ **100.0% xuống 0.6%** (23360 → 144, đúng bằng số
tracklet, tức chỉ còn frame đầu của mỗi target). Khoá được đọc, đúng khối. Nếu nó bị bỏ qua
thì độ phủ đã không đổi.

Ghi cách làm này vào CLAUDE.md §11 như một phương pháp, không chỉ một kết luận.

**3. Camera lạ phải suy giảm êm, không được giết engine.** `Topology.check()` ném
`TopologyError` khi gặp camera chưa khai báo. Chủ ý ban đầu (có test riêng, tên
`test_camera_la_thi_bao_loi_thay_vi_am_tham_cho_qua`) là "báo lỗi thay vì âm thầm cho qua" —
**đúng mục đích, sai cơ chế**. `check()` chạy trong vòng gán khi pipeline đang phát: ném lỗi
là giết cả engine. Đo được: chạy 4 luồng với `topology.yaml` chỉ khai cam01–cam02, engine
chết ở tracklet đầu tiên của cam03, **pipeline vẫn chạy tiếp bình thường**, và mọi cập nhật
Global ID biến mất — không ai đọc `mct:global` mà biết vì sao.

Sửa: camera lạ là trường hợp ngặt hơn của "cặp chưa khai báo transit" (ta còn không biết
camera đó ở đâu), nên áp cùng `unknown_pair_policy`, kèm **cảnh báo một lần cho mỗi camera**.
Giữ nguyên chủ ý "không im lặng", bỏ cơ chế gây chết. Test cũ được viết lại để ghi cả lý do
đổi, không xoá trắng.

**4. Hai thay đổi của hai phiên hoá ra phụ thuộc nhau.** Khối lọc lớp person (phiên 7) đo
riêng thì gần như **miễn phí về tốc độ** — 764.6 vs 769.8 FPS, nằm trong nhiễu. Nhưng khi
ReID bật, nó đáng **+66% throughput** (422.3 → 700.0 FPS), vì ReID chạy trên *từng đối tượng*
và bộ lọc cắt 66% số đối tượng. Nếu chỉ đo bộ lọc trước khi có ReID thì đã kết luận nó không
đáng gì. Bài học cho chương 6: đo tối ưu hoá trong đúng cấu hình sẽ vận hành, không đo lẻ.

**5. Bỏ `ultralytics`+`torch` khỏi bootstrap khi đã có sẵn `.onnx`.** Bước đó tồn tại chỉ để
export `.pt → .onnx`; pipeline suy luận bằng TensorRT qua nvinfer, không dùng ultralytics lúc
chạy. Trên máy này pip còn tải hỏng file (hash mismatch — host làm hỏng dữ liệu tải về). Bỏ
được ~1 GB tải về và một bước dễ hỏng. Có cờ `FORCE_EXPORT_DEPS=1` cho khi thật sự cần export.

## Số liệu đo được

**Cấu hình chung:** vast.ai instance `49832501`, RTX 3090 24GB, driver 595.84, DeepStream
7.1.0, CUDA 12.6, TensorRT 10.3, pyds 1.2.0, Python 3.10.12. YOLO11s weight gốc COCO, FP16,
input 640, `nvstreammux` 1920×1080. ReID = OSNet `osnet_x1_0_msdc_dg` (khái quát hoá miền),
ONNX → TensorRT FP16, batch 100. Nguồn: `sample_1080p_h264.mp4` (1443 frame) lặp cho 4 luồng.
`tracker-width/height = 960×544`. **Không có homography** (camera thật chưa hiệu chỉnh).

### Đường (A) — đọc embedding từ nvtracker

| hạng mục | kết quả |
|---|---|
| `pyds.NvDsMetaType.NVDS_TRACKER_OBJ_REID_META` | **có** |
| số chiều embedding probe đọc được | **512** |
| độ phủ | **23360/23364 detection = 100.0%** |
| 4 detection thiếu embedding | đúng 4 cái có `confidence = −0.1` (target do tracker suy ra, không có crop) |
| chuẩn L2 của embedding | **1.0000** → `addFeatureNormalization: 1` chạy |
| `reidExtractionInterval` 0 → 100000 | độ phủ 100.0% → **0.6%** (144 = số tracklet) |

### FPS (4 luồng, `--stats`, `sink.sync=false`)

| cấu hình | FPS gộp | FPS/luồng | detection ra khỏi detector |
|---|---|---|---|
| không ReID, có lọc lớp | **772.2** | 193.1 | 23 364 |
| **có ReID, có lọc lớp** | **700.0** | **175.0** | 23 364 |
| có ReID, **không** lọc lớp | 422.3 | 105.6 | 68 652 |
| không ReID, không lọc lớp | 769.8 | 192.5 | 68 652 |

- Chi phí của ReID: **−9.4%** throughput.
- Giá trị của khối lọc lớp **khi có ReID**: **+66%** throughput.
- Bộ lọc loại **45 288 / 68 652 = 66%** đầu ra của detector.
- Dao động giữa các lần chạy ~3% (700.0 và 718.7 ở hai lần cùng cấu hình) — mọi so sánh
  dưới 5% cần chạy lại nhiều lần mới kết luận được.
- Mục tiêu đề cương (3–4 luồng, ≥15 FPS/luồng): **vượt xa** — 175 FPS/luồng có ReID.

### VRAM

**1567 MiB đỉnh**, GPU util ~87–89%, 4 luồng có ReID. Thoải mái dưới trần GPU 8GB mà
CLAUDE.md §11 lo ngại. (Lưu ý: nguồn file, chưa có jitter RTSP.)

### Độ trễ end-to-end (probe → Redis → engine → `mct:global`)

Nguồn phát **đúng tốc độ thật** (`sink.sync: true`), 4 luồng, engine chạy cùng máy, 945 mẫu.

| | ms |
|---|---|
| trung vị | **40.4** |
| trung bình | 342.4 |
| p90 | 2077.7 |
| p99 | 2977.4 |
| nhỏ nhất / lớn nhất | 3.3 / 3009.6 |

**Phân bố hai đỉnh, và phải diễn giải đúng.** Đa số cập nhật về trong ~40 ms; đuôi kéo tới
~3 s. `GlobalUpdate.ts_ms` là mốc muộn nhất của *tracklet* tại thời điểm gán, nên: tracklet
đang sống được cập nhật mỗi cửa sổ → độ trễ nhỏ; tracklet đã kết thúc chỉ được gán sau khi
hết TTL → `ts_ms` đã cũ sẵn, tạo ra đuôi. Tức đuôi phản ánh **độ trễ chốt danh tính**, không
phải nghẽn hàng đợi.

Với câu hỏi "người này đang ở đâu" của dashboard, con số đúng là **trung vị 40 ms**. Với câu
hỏi "bao lâu thì một danh tính được chốt", con số là ~2–3 s. Mục tiêu "<1 s" của đề cương
(CLAUDE.md §7) chưa nói rõ là cái nào — **cần chốt định nghĩa trước khi tuyên bố đạt hay
không**. Chưa gọi là đạt.

### Chất lượng embedding (fixture thật, 23360 embedding / 144 tracklet)

| cặp | n | p10 | trung vị | p90 |
|---|---|---|---|---|
| cùng tracklet, frame kề nhau | 23 216 | 0.931 | **0.974** | 0.997 |
| khác tracklet, cùng camera | 12 211 | 0.393 | **0.491** | 0.658 |

Tách biệt rất rộng. **Nhưng chưa được suy ra ngưỡng từ đây:** dữ liệu là 4 bản sao cùng một
video và cặp dương là frame kề nhau — dễ hơn hẳn bài toán xuyên camera của WildTrack, nơi
phiên 4 đo được trần lý thuyết chỉ 0.513. Câu hỏi "`max_cost` chỉnh trên đường ONNX Runtime
có chuyển sang đường DeepStream không" **vẫn chưa có câu trả lời**.

## Cạm bẫy phát hiện trong phiên

- **Cấu hình CRLF làm `sed` im lặng không khớp.** `Path.write_text()` trên Windows tự đổi
  `\n → \r\n`, nên file config đẩy sang Linux là CRLF. YAML vẫn parse được (không triệu
  chứng), nhưng `sed 's/x: 0$/x: 100000/'` không khớp vì dòng kết thúc bằng `\r`. Lần đầu
  chạy thí nghiệm `reidExtractionInterval` tôi đã suýt kết luận sai là "đổi giá trị không
  ảnh hưởng gì" — trong khi giá trị chưa hề đổi. Đã ghim `configs/** eol=lf`.
- **`model-engine-file` của DeepStream-Yolo bị bỏ qua khi GHI** — xác nhận lần hai. Engine ra
  `<cwd>/model_b4_gpu0_fp16.engine` chứ không phải chỗ config khai; không chép sang đúng tên
  thì mỗi lần chạy lại build lại ~3 phút.
- **`nohup ... &` qua `ssh` treo phiên** nếu không đóng stdin (`</dev/null`). Mất một lượt.
- **pip trên host này làm hỏng file tải về** (hash mismatch). Bootstrap giờ thử lại 3 lần.

## Vướng mắc / chưa xong

- **Ngưỡng `max_cost` cho embedding của DeepStream vẫn chưa kiểm được.** Cần đưa chính
  WildTrack qua pipeline DeepStream (đóng ảnh thành video, chạy 7 luồng) rồi so với fixture
  đường ONNX Runtime. Đây là việc đáng làm nhất còn lại của M3.
- **Định nghĩa "độ trễ end-to-end" trong đề cương** cần chốt (xem trên) trước khi báo cáo.
- **Chưa đo leg dashboard** (engine → WebSocket) trên cùng máy; phiên 6 đã đo trên `ut-hpc`
  nhưng ghép số đo của hai máy lại là không chặt.
- `tracker-width/height = 960×544` vẫn chưa thử nâng lên 1080p — crop đưa vào ReID đang lấy
  từ frame nửa độ phân giải. Với người ở xa có thể ảnh hưởng chất lượng embedding.
- Fixture 68 MB là 4 bản sao cùng một video, **không dùng để đánh giá liên kết xuyên camera
  được** — chỉ dùng để kiểm đường ống và thống kê embedding.

## Bước tiếp theo

1. Đưa WildTrack qua pipeline DeepStream để trả lời câu hỏi ngưỡng (việc lớn nhất còn lại
   của M3) — chuẩn bị phần đóng ảnh thành video trên máy dev trước, rồi mới thuê máy.
2. Chốt định nghĩa độ trễ với GVHD, rồi báo cáo con số tương ứng.
3. Cảnh báo "hai Global ID quá gần nhau trên mặt phẳng" — treo từ phiên 6.
