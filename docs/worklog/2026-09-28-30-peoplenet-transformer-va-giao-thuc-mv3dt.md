# 2026-09-28 — PeopleNet Transformer trên DeepStream 7.1: không thắng YOLO11s; giao thức chấm của MV3DT

- **Mốc:** M2 (đổi detector) + M6 (baseline) | **Máy:** máy dev (Windows) + `vast-gpu` (2 × T4 hỏng, rồi RTX A4000) | **Thời lượng:** ~5h (GPU thuê ~1.5h) | **GPU thuê: ≈ $0.17 (ước tính theo giờ)**

## Mục tiêu phiên

Hai việc trên máy dev ở mục "Bước tiếp theo" của phiên 29 (bước 1 và 3). Làm để lần thuê `vast-gpu`
tới chỉ còn việc chạy.
- Bước 1: xác định định dạng file PeopleNet Transformer và yêu cầu plugin TensorRT, viết config
  `nvinfer`, kiểm bước hậu xử lý bbox.
- Bước 3: đọc bài MV3DT, chép đúng giao thức chấm WildTrack.
- Phần 2 (người dùng yêu cầu trong phiên): thuê `vast-gpu`, chạy WildTrack n = 3 cho v1 và v2, đo
  FPS 4 luồng, chấm cả hai giao thức ngay trên instance (người dùng chọn để không tải máy dev).

## Đã làm

- Tải và kiểm cả hai file ONNX trên NGC (gói `onnx`, không cần GPU). Có hai bản vì bài báo và
  reference app của MV3DT dùng hai bản khác nhau (quyết định 1).
- `src/tools/check_peoplenet_cpu.py` (+6 test, `tests/test_check_peoplenet_cpu.py`): chạy ONNX bằng
  ONNX Runtime trên CPU.
  - Op `MultiscaleDeformableAttnPlugin_TRT` chỉ TensorRT hiểu, nên thay bằng bản numpy
    (`msda_reference`, lõi PyTorch của Deformable DETR), đăng ký qua `onnxruntime-extensions`.
  - Giải mã đầu ra y như parser `NvDsInferParseCustomDDETRTAO` của NVIDIA.
  - Nhờ vậy trả lời được lớp, chuẩn hoá và parser mà không tốn tiền thuê máy.
  - Cài thêm: extra `detcheck` trong `pyproject.toml`.
- Config `nvinfer`:
  - `configs/pipeline/config_infer_peoplenet_transformer.txt` (v1.1)
  - `configs/pipeline/config_infer_peoplenet_transformer_v2.txt` (v2)
  - Streams WildTrack tương ứng: `configs/demo/streams_wildtrack_pnt.yaml`, `streams_wildtrack_pnt2.yaml`.
    Chỉ khác `streams_wildtrack.yaml` ở mục `pgie`.
- `src/ds_pipeline`: probe trước đây lọc cứng `class_id == 0`. Nay đọc lớp người từ
  `pgie.person_class_id` trong streams YAML (mặc định 0, nên cấu hình YOLO không đổi).
  - `builder.py`: thêm `PipelineConfig.person_class_id`.
  - `probes.py`: `make_probe(..., person_class_id)`.
  - `__main__.py`: truyền khoá này vào probe và ghi log.
  - Schema không đổi: `Detection` ra khỏi probe vẫn mang `CLASS_PERSON = 0`.
- `tests/test_pipeline_configs.py`: +8 hàm test (13 ca) cho ba lỗi không triệu chứng ở quyết định 2, và cho
  cặp v1/v2 chỉ khác file ONNX. Kiểm ngược: sửa hỏng từng chỗ (lớp, `topk=20`, `offsets=0`) thì
  4 test đỏ.
- Parser `libnvds_infercustomparser_tao.so`:
  - `docker/vast_bootstrap.sh` bước 5b và `docker/deepstream.Dockerfile` build nó từ
    `deepstream_tao_apps` (MIT). Chỉ lấy 2 file `post_processor/` ở commit ghim `bc1fa045` thuộc
    nhánh `release/tao_ds7.1ga`.
  - Symlink ra `third_party/deepstream_tao_apps`.
- `docker/fetch_peoplenet_transformer.sh`: tải hai ONNX và `labels.txt` từ NGC, kiểm sha256.
- `docker/vast_peoplenet_check.sh`: việc đầu tiên trên máy thuê.
  - Plugin có trong binary `libnvinfer_plugin` không.
  - `trtexec` build FP16 cả hai bản ở batch 1 và 4.
  - `nvinfer` 7.1 có cắt `topk` khi `cluster-mode=4` không.
- Đọc bài MV3DT (arXiv 2606.13127v1, bản HTML tải về đọc nguyên văn), reference app
  `deepstream-tracker-3d-multi-view`, và mục đánh giá của EarlyBird và TrackTacular để so giao thức.
- Test: 642 passed, 5 skipped (venv 3.10); `ruff check`/`format --check` sạch.

**Phần 2: chạy trên GPU**
- Script thêm:
  - `docker/vast_pnt.sh`: chạy WildTrack n lần và đo FPS. Mỗi lần in bằng chứng (lớp, engine, số
    detection) và ghi nhiệt độ/xung GPU mỗi 5 s.
  - `docker/vast_pnt_eval.sh`: chấm ngay trên instance. TrackEval ghim commit `12c8791b` như máy dev,
    venv numpy 1.23.5 dựng bằng `uv`.
  - `configs/pipeline/streams_reid_pnt{,2}.yaml`: 4 luồng, chỉ khác `streams_reid.yaml` ở `pgie` (+2 test).
- **Ba lần thuê** (quyết định 5):
  1. T4 offer `13790355` (instance `53170067`): plugin có, engine v1 build được, nhưng GPU bị
     **SW Thermal Slowdown** (84 °C, SM 300/1590 MHz). Hủy. Log ở `data/s30/host1_throttled/`.
  2. T4 offer `13080903` (instance `53171751`, driver 580): **không có `libnvcuvid.so`**. Bước kiểm
     NVDEC bắt được trong 30 s. Hủy.
  3. RTX A4000 offer `35674412` (instance `53173813`, Nhật, driver 595.71, $0.088/h): chạy toàn bộ.
     Hủy sau khi kéo dữ liệu về, `show instances` = 0.
- Kết quả (gitignored):
  - fixture `data/fixtures/ds_wildtrack_7cam_{pnt,pnt2}_r{1,2,3}.jsonl`, kèm `.gt.json` và `.gt-report.json`;
  - `data/s30/compare_oracle_tracker.json`, `data/s30/ground_plane.json`, `data/s30/fp_region_r1.json`;
  - DB engine từng lần ở `data/s30/<nhãn>_r<n>/`, log ở `data/s30/logs/`.
- Đối chứng YOLO11s 640 là 3 fixture R640 của phiên 25, **chấm lại** bằng code hiện tại cùng lúc với
  PNT. Ra đúng số cũ (hộp 15.79 ± 0.47, mặt đất 31.1 ± 0.8), nên code liên kết và code chấm không đổi
  kể từ phiên 25.

## Quyết định kỹ thuật

### 1. Đo cả hai bản PeopleNet Transformer, vì bài và code của MV3DT lệch nhau

- **Bài báo** (mục 4) ghi PNT = "PeopleNet Transformer 1.1", trích tới trang NGC của bản v1:
  Deformable DETR + ResNet50, 300 query.
- **Reference app** (`scripts/setup_prerequisites.sh`) lại tải
  `peoplenet_transformer_v2/deployable_v1.0/dino_fan_small_astro_delta.onnx`: DINO + FAN-Small,
  900 query. File được đổi tên thành `peoplenet_transformer_model_op17.onnx`, tên dễ khiến người
  đọc tưởng là v1.
- Không biết bảng WildTrack của bài dùng bản nào. Hai config giống hệt nhau trừ file ONNX (có test
  canh), nên so v1/v2 là so đúng một biến.
- **Đánh đổi tốc độ** (thẻ NGC, T4, FP16, batch 1): v1 43 FPS, v2 19 FPS. Mục tiêu đề cương là
  4 × 15 = 60 khung/s. Trên T4, cả hai đều chưa đạt ở batch 1. WildTrack 7 luồng × 2 fps chỉ cần
  14 khung/s nên chạy được, nhưng FPS 4 luồng phải đo thật.

### 2. Ba lỗi không triệu chứng khi đổi sang PeopleNet Transformer, cả ba đã được chặn

1. **Người là lớp 1, không phải 0.** Labels là `BG, Person, Face, Bag`, và parser bỏ lớp 0 (nền).
   - Probe cũ lọc cứng `class_id == 0`, nên sẽ bỏ **sạch** mọi hộp mà pipeline vẫn chạy
     "thành công", fixture ra rỗng.
   - Sửa: `pgie.person_class_id` trong streams YAML. Test kiểm, trên MỌI file streams, rằng lớp này
     đúng là lớp duy nhất lọt qua `nvinfer`.
   - Xác nhận bằng chạy thử trên CPU: chỉ hộp lớp 1 khớp người GT, lớp 2 và 3 không khớp hộp nào.
2. **Chuẩn hoá là ImageNet, dù thẻ NGC ghi "scale 1/255, không chuẩn hoá".**
   - Config NVIDIA dùng `offsets=123.675;116.28;103.53`, `net-scale-factor=0.01735`.
   - Chạy thử v1 trên CPU: ImageNet bắt 22/33 người, 1/255 chỉ bắt 2/33.
   - v2 ít nhạy hơn (19/33 cả hai cách), nhưng vẫn giữ ImageNet như config MV3DT.
3. **`topk=20` của config mẫu NVIDIA cắt mất người.**
   - Đọc `nvdsinfer_context_impl_output_parsing.cpp` (mã nguồn DS 9): `fillUnclusteredOutput` vẫn
     gọi `filterTopKOutputs` theo lớp khi `cluster-mode=4`.
   - Một khung WildTrack C1 có 33 người GT, và riêng lớp người đã ra 47 hộp ở ngưỡng 0.3.
   - Đặt `topk=200`, bằng `keep_top_k` mà parser tự cắt.
   - Bản 7.1 kiểm lại trên máy thuê (`vast_peoplenet_check.sh` bước 3).

Ngưỡng `pre-cluster-threshold=0.3` lấy theo config MV3DT (mẫu DS 7.1 dùng 0.5). Confidence ở đây
là sigmoid của logit DETR, khác thang với YOLO, nên không mang ngưỡng 0.25 của YOLO sang. Chỉ quét
ngưỡng sau khi có số n = 3 ở mức mặc định.

### 3. Engine batch 7 dùng luôn cho 4 luồng

`checkBackendParams` của `nvinfer` chỉ build lại khi engine có maxBatch **nhỏ hơn** batch được yêu
cầu. Vì vậy mỗi model chỉ cần một file config batch 7, không cần thêm các bản b1/b4 như YOLO. Khác
DeepStream-Yolo: ở đây `nvinfer` tự build, nên engine được ghi đúng
`<onnx-file>_b7_gpu0_fp16.engine` cạnh file ONNX, trùng tên đã khai trong config (có test canh).

### 4. Giao thức chấm của MV3DT: chưa đủ để đặt số cạnh nhau, và bài có chỗ tự mâu thuẫn

Xem bảng 3. Những điểm phải ghi vào chương 6:
- **Hai bộ số khác nhau cho cùng một cấu hình.**
  - Lời văn mục 4.2 và 4.4 ghi IDF1 94.3 / MOTA 93.3 / MOTP 93.5.
  - Bảng 1 và dòng cuối bảng 3 ghi 96.5 / 93.1 / 94.6.
  - Tóm tắt dùng 96.5 / 93.1.
- **Trích dẫn bảng 1 lệch.**
  - "EarlyBird [6]" trỏ vào bài dataset WILDTRACK.
  - "BEV-SUSHI [42]" trỏ vào bài TrackTacular.
  - "MVTr [50]" và "MVTrajecter [50]" cùng trỏ một bài nhưng ra hai bộ số khác nhau.
  - Đây là bản v1 của bài, nên tin số liệu ở mức vừa phải.
- **Bài không nêu:**
  - ngưỡng khoảng cách;
  - không gian chấm (mặt đất hay ảnh);
  - có giới hạn vùng không;
  - công cụ chấm;
  - phần cứng của con số 27 FPS (H100 NVL chỉ dùng cho thí nghiệm 100 camera);
  - model ReID (chỉ ghi "như [48]", tức bài DeepSORT).
- **Suy ra từ các hệ cùng bảng:** EarlyBird và TrackTacular ghi rõ ngưỡng r = 1 m trên mặt đất,
  tập test là 10% khung cuối. MV3DT đặt số của mình cạnh họ, nên nhiều khả năng dùng cùng giao thức.
  Đó là **suy luận, không phải trích dẫn**.
- **Hệ quả cho đồ án:** chỉ đặt số cạnh MV3DT theo biến thể "điểm mặt đất, 40 khung test, T = 1 m,
  trong vùng" của `eval/eval_ground_plane.py`, và ghi rõ các khoá trên là suy luận.
- **Còn một khác biệt đầu vào chưa rõ.** Bài mô tả WildTrack là "2,000 frames at 10 FPS with 400
  annotated frames at 2 FPS". Pipeline của đồ án chỉ chạy 400 khung chú thích (2 fps). Nếu MV3DT
  chạy video 10 fps, tracker của họ có mật độ khung gấp 5 lần. Chưa rõ họ chạy bản nào, và cũng
  chưa rõ gói WildTrack công khai có khung 10 fps hay không.

### 5. Chọn máy: kiểm nhiệt độ/xung GPU chứ không chỉ NVDEC; FPS đo trên RTX A4000

Hai máy T4 hỏng theo hai kiểu, mà bước kiểm của repo trước đây chỉ bắt được một:
- **Hạ xung vì nhiệt** không gây lỗi, chỉ làm chậm.
  - v1 ở batch 7 còn khoảng 12 ảnh/s, dưới mức 14 ảnh/s mà WildTrack 7 luồng × 2 fps cần.
  - Pipeline khi đó trễ dần, `ts_ms` giãn ra, và fixture không còn so được với YOLO.
  - Chỉ `nvidia-smi` (clocks và `clocks_throttle_reasons`) mới lộ ra. Từ phiên này `vast_pnt.sh` ghi
    nhiệt độ/xung suốt mỗi lần đo.
- **Thiếu `libnvcuvid.so`** dù `NVIDIA_DRIVER_CAPABILITIES=all`: host không mount thư viện video của
  driver vào container. Bước kiểm NVDEC của CLAUDE.md §11 bắt được ngay.

Người dùng chọn RTX A4000 thay vì chiếc T4 còn lại (Nevada, mạng 947 Mbps).
- **FPS phiên này không so thẳng được** với các số T4 (phiên 23) hay RTX 3090 (phiên 3–9). Vì vậy
  YOLO11s cũng được đo lại trên cùng máy.
- HOTA không phụ thuộc máy, miễn pipeline theo kịp thời gian thực. Cả 6 lần chạy đều đủ 2800 khung
  trong 198.6–198.8 s.

### 6. PeopleNet Transformer KHÔNG thay YOLO11s 640. Giữ YOLO, PNT chỉ làm ablation

- **Hộp ảnh:** v1 13.87 ± 0.33, v2 12.86 ± 0.39, so với 15.79 ± 0.47 của YOLO. v1 kém hơn với t ≈ 5.8.
- **Mặt đất trong vùng, 400 khung:** v1 28.4 ± 1.1, v2 26.3 ± 0.2, so với 31.1 ± 0.8 (v1: t ≈ 3.4).
- Bật NMS mặt đất 0.5 m, hay chỉ tính 40 khung test, đều không đảo được thứ tự (bảng 6).
- v2 còn quá chậm: 9.0 FPS/luồng trên A4000 (bảng 4), dưới mục tiêu 15. Bỏ v2.

Giả thuyết của phiên 29, rằng "đổi sang detector chuyên cho người sẽ lấy lại phần lớn điểm mất ở đầu
vào", **bị bác bỏ ở cấu hình mặc định.** Lý do ở quyết định 7.

### 7. Vì sao: PNT thấy nhiều người hơn, và phần lớn số người thêm nằm ngoài vùng WildTrack chú thích

Hai phân tích trên lần chạy r1 (bảng 7 và 8):
- **Không phải do ngưỡng.**
  - Ngay trong nhóm confidence ≥ 0.8, chỉ 42% hộp PNT khớp GT; YOLO là 79%.
  - Ở ngưỡng ≥ 0.8, PNT có precision 0.42 và recall 0.51. Precision đó vẫn dưới YOLO ở mọi ngưỡng,
    nên nâng ngưỡng không cứu được.
  - Hộp "sai" của PNT là hộp rất tự tin.
- **84.5% số hộp sai tăng thêm so với YOLO có chân nằm NGOÀI lưới 12 × 36 m**
  (`eval/diagnose_fp_region.py`). Đó là người thật mà WildTrack không chú thích.
  - Trong lưới, PNT thêm +5 282 hộp đúng nhưng cũng thêm +5 591 hộp sai: 1 394 hộp lệch hoặc trùng
    trên một người có thật, 4 197 hộp không trùng ai.
  - Nhóm "lệch/trùng" chỉ tăng ít, nên không có dấu hiệu lỗi toạ độ hệ thống.
- **Các track ngoài vùng làm hỏng cả bước liên kết.** `src/mct` không biết vùng nào được chấm, nên
  track của người ngoài vùng cũng tranh Global ID:
  - IDs trên hộp ảnh tăng 406 → 669;
  - IDP mặt đất giảm 27.6 → 20.7, dù IDR tăng 49.3 → 53.8.

Hệ quả cho cách đọc phiên 21/24/26: kịch bản "hộp GT" (HOTA mặt đất 31.1 → 63.8) cộng dồn hai hiệu ứng.
1. Hộp tốt hơn.
2. **Chỉ giữ đúng những người nằm trong vùng chấm.**

Một detector người tốt hơn theo nghĩa chung chỉ lấy được hiệu ứng 1, và lại làm hiệu ứng 2 tệ đi.

**Đề xuất, chưa chốt:** thêm **vùng quan tâm (ROI) trên mặt phẳng đất** như một tính năng của hệ
thống, áp như nhau cho mọi detector: bỏ detection có chân ngoài vùng giám sát trước khi vào `src/mct`.
- **Vì sao hợp lý:** triển khai thật nào cũng khai vùng giám sát, và homography đã có sẵn.
- **Phải trình bày minh bạch:** trên WildTrack, vùng giám sát trùng đúng vùng chấm. Vì vậy phải nói
  rõ là "hệ thống biết vùng giám sát", và đo cả YOLO lẫn PNT.
- **Làm được trên fixture có sẵn**, không cần GPU. Lọc ở fixture đánh giá thấp lợi ích so với lọc
  trước tracker (phiên 22), nên số thu được là cận dưới.

## Số liệu đo được

### 1. Cấu trúc hai file ONNX (đọc bằng `onnx`, không chạy)

| | v1.1 `resnet50_peoplenet_transformer_op17.onnx` | v2 `dino_fan_small_astro_delta.onnx` |
|---|---|---|
| NGC | `peoplenet_transformer:deployable_v1.1` | `peoplenet_transformer_v2:deployable_v1.0` |
| Kiến trúc | Deformable DETR + ResNet50, 2 tầng đặc trưng | DINO + FAN-Small, 4 tầng đặc trưng |
| Dung lượng | 94 MB | 215 MB |
| Opset / sinh bởi | 17 / PyTorch 1.14 | 17 / PyTorch 2.1 |
| Đầu vào | `inputs` [batch, 3, 544, 960], **batch động** | như v1 |
| Đầu ra | `pred_logits` [batch, 300, 4], `pred_boxes` [batch, 300, 4] (cx, cy, w, h chuẩn hoá) | như v1, 900 query |
| Op tuỳ biến | 12 × `MultiscaleDeformableAttnPlugin_TRT` (domain `nvidia`, version "1") | 12 × như v1; 6 node nhận `sampling_locations` kiểu float64 |
| FPS thẻ NGC (T4, FP16, batch 1) | 43 | 19 |

Plugin có trong mã nguồn TensorRT OSS nhánh `release/8.6` tới `release/10.3`, tên và version
khớp (`DMHA_VERSION "1"`). DeepStream 7.1 đi kèm TensorRT 10.3. Config mẫu chính thức của NVIDIA
cho DS 7.1 (`deepstream_tao_apps` nhánh `release/tao_ds7.1ga`) chạy chính file v1.1 này. **Chưa
kiểm** bản binary `libnvinfer_plugin` trong image.

### 2. Chạy thử trên CPU (`tools.check_peoplenet_cpu`)

- **Cấu hình:** 1 ảnh WildTrack C1 khung `00000000`, toàn khung, IoU 0.5, ngưỡng 0.3, ghép
  Hungarian. ONNX Runtime 1.23.2 CPU FP32, op plugin chạy bằng numpy.
- **Chỉ dùng để kiểm đúng/sai.** Một ảnh không đủ để so chất lượng detector, và không so được với
  YOLO11s.

| Model | Chuẩn hoá | Hộp lớp 1 | Hộp lớp 1 khớp GT | Recall lớp 1 | Lớp 2 / 3 khớp GT | CPU |
|---|---|---|---|---|---|---|
| v1.1 | ImageNet | 47 | 22 | 22/33 | 0 / 0 (20 và 26 hộp) | ~16–20 s/ảnh |
| v1.1 | 1/255 | 5 | 2 | 2/33 | 0 | ~16–19 s/ảnh |
| v2 | ImageNet | 51 | 19 | 19/33 | 0 / 0 (14 và 19 hộp) | ~36 s/ảnh |
| v2 | 1/255 | 58 | 19 | 19/33 | 0 / 0 | ~39 s/ảnh |

Nhiều hộp lớp 1 không khớp GT là chuyện bình thường khi chấm toàn khung: WildTrack chỉ chú thích
người trong lưới 12 × 36 m (phiên 25).

### 3. Giao thức chấm WildTrack: MV3DT so với đồ án

| Khoá | MV3DT (arXiv 2606.13127v1) | EarlyBird / TrackTacular | Đồ án (`eval_ground_plane.py`) |
|---|---|---|---|
| Tập khung | "last 10% of the 7 sequences" (bảng 1) → 40 khung | 10% khung cuối | 400 khung, hoặc 40 khung test |
| Không gian | không nêu (hệ ước lượng chân 3D trên mặt đất) | mặt đất | mặt đất (trung vị đa camera) |
| Ngưỡng | **không nêu** | r = 1 m | T = 1 m |
| Vùng | không nêu | lưới chú thích (mô hình chỉ dự đoán trong lưới) | trong lưới 12 × 36 m |
| Chỉ số | IDF1, MOTA, MOTP | MOTA, IDF1, MOTP | HOTA, IDF1, MOTA (TrackEval) |
| Công cụ | không nêu | — | TrackEval |
| Detector | PNT 1.1 (bài) / v2 (reference app) | học trên WildTrack | YOLO11s 640 → PNT |
| Học trên WildTrack | không | có | không |
| FPS | 27, phần cứng không nêu | — | đo riêng |

Bảng 1 của MV3DT (nguyên văn):

| Hệ | IDF1 | MOTA | MOTP | FPS |
|---|---|---|---|---|
| EarlyBird | 92.3 | 89.5 | 86.6 | – |
| MVTr | 93.1 | 92.3 | 92.7 | – |
| BEV-SUSHI | 93.4 | 87.5 | 94.3 | – |
| MVFlow | 93.5 | 91.3 | – | – |
| TrackTacular | 95.3 | 91.8 | 85.4 | – |
| UMPN | 96.3 | 93.9 | 86.9 | 2 |
| MVTrajecter | 96.5 | 94.3 | 93.0 | – |
| MV3DT w/ PNT | 96.5 | 93.1 | 94.6 | 27 |

Bảng 3 (từng bước truyền ID, WildTrack), IDF1 / MOTA:
- chỉ đơn camera: 25.0 / 15.9;
- thêm bước 1: 55.4 / 44.9;
- thêm bước 2 (liên kết lại muộn): 87.5 / 89.6;
- thêm bước 3 (sửa ID): 96.5 / 93.1.

Bước 2 và 3 đều là các bước **sửa ID sau khi đã gán**. Điều này ủng hộ hướng "cho phép sửa Global
ID" ở quyết định 1 của phiên 29.

### 4. FPS 4 luồng 1080p, CÓ ReID

- **Cấu hình:** RTX A4000, driver 595.71, DS 7.1, `sync: false`, `sample_1080p_h264.mp4` × 4.
- **Cách đo:** `docker/vast_pnt.sh fps`, mỗi cấu hình 2 lần, lấy lần 2. Lần 1 lệch dưới 0.3%.

| Detector | Engine | FPS gộp | FPS/luồng | Hộp/khung | Nhiệt max / xung SM min |
|---|---|---|---|---|---|
| YOLO11s 640 | b4 FP16 | 425.5 | 106.4 | 4.04 | 82 °C / 1515 MHz |
| PNT v1.1 | b7 FP16 (dùng cho batch 4) | 99.6 | 24.9 | 5.74 | 84 °C / 1320 MHz |
| PNT v2 | b7 FP16 (dùng cho batch 4) | 36.1 | 9.0 | 5.27 | 84 °C / 1290 MHz |

Cờ throttle suốt các lần đo chỉ là `0x4` (SW Power Cap, giới hạn 140 W). Có lúc thoáng thấy `0x4C`
(có HW thermal slowdown), ở lần WildTrack đầu và lần YOLO đầu.

`trtexec` trên chính engine đó (A4000, FP16, profile 1/7/7):

| Model | batch 1 | batch 4 | batch 7 |
|---|---|---|---|
| v1.1 | 107.5 ảnh/s (9.3 ms) | 124.4 ảnh/s | 129.6 ảnh/s |
| v2 | 35.8 ảnh/s (27.8 ms) | 39.1 ảnh/s | 40.5 ảnh/s |

Máy T4 bị hạ xung (84 °C, SM 300 MHz) chỉ đạt 9.9 / 11.6 / 12.2 ảnh/s cho v1, trong khi thẻ NGC ghi
43 FPS ở batch 1. Không dùng các số này.

### 5. WildTrack, hộp ảnh (IoU 0.5, toàn khung, 400 khung, kịch bản A, n = 3)

- **Công cụ:** `eval.compare_oracle_tracker`, `configs/demo/wildtrack_ds.mct.yaml`, TrackEval `12c8791b`.

| Detector | HOTA | DetA | AssA | AssRe | AssPr | IDF1 | IDs | Dets |
|---|---|---|---|---|---|---|---|---|
| YOLO11s 640 (fixture phiên 25, chấm lại) | 15.79 ± 0.47 | 24.05 ± 0.06 | 10.62 ± 0.66 | 13.42 ± 0.64 | 34.35 ± 1.30 | 20.27 ± 0.59 | 406 ± 23 | 34 289 |
| PNT v1.1 | 13.87 ± 0.33 | 19.61 ± 0.03 | 10.00 ± 0.49 | 13.28 ± 0.49 | 28.64 ± 1.27 | 15.12 ± 0.62 | 669 ± 4 | 75 696 |
| PNT v2 | 12.86 ± 0.39 | 18.33 ± 0.04 | 9.17 ± 0.54 | 12.77 ± 0.73 | 26.47 ± 0.20 | 13.60 ± 0.93 | 713 ± 12 | 87 932 |

Gán GT bằng `tools.ds_wildtrack_gt`:
- phủ ground-truth: v1 57.2–57.4%, v2 59.1–59.6%, so với 44.9% của YOLO 640 (phiên 25);
- số track giữ lại: v1 359–371, v2 357–366.

### 6. WildTrack, điểm mặt đất (trong vùng, T = 1 m, n = 3)

- **Công cụ:** `eval.eval_ground_plane`.
- **Cột "Khung":** 400 là mọi khung chú thích; 40 là 40 khung test, tập mà các bài WildTrack dùng.

| Detector | NMS | Khung | HOTA | DetA@0.5 | AssA@0.5 | MOTA | IDF1 | IDP | IDR |
|---|---|---|---|---|---|---|---|---|---|
| YOLO11s 640 | — | 400 | 31.1 ± 0.8 | 34.6 ± 0.4 | 34.3 ± 1.6 | −41.4 ± 3.5 | 35.4 ± 1.2 | 27.6 ± 0.9 | 49.3 ± 2.2 |
| PNT v1.1 | — | 400 | 28.4 ± 1.1 | 27.9 ± 0.1 | 35.6 ± 2.7 | −112.2 ± 1.4 | 29.9 ± 0.5 | 20.7 ± 0.3 | 53.8 ± 0.7 |
| PNT v2 | — | 400 | 26.3 ± 0.2 | 27.5 ± 0.3 | 31.9 ± 0.9 | −117.8 ± 4.2 | 29.4 ± 0.2 | 20.2 ± 0.0 | 54.0 ± 1.1 |
| YOLO11s 640 | 0.5 m | 400 | 32.1 ± 0.8 | 41.2 ± 0.3 | 31.0 ± 1.4 | −9.6 ± 1.0 | 36.3 ± 0.9 | 31.3 ± 0.5 | 43.1 ± 1.6 |
| PNT v1.1 | 0.5 m | 400 | 29.2 ± 0.9 | 34.2 ± 0.1 | 30.8 ± 1.8 | −62.9 ± 2.4 | 30.7 ± 0.7 | 23.0 ± 0.6 | 46.0 ± 0.9 |
| PNT v2 | 0.5 m | 400 | 26.7 ± 0.9 | 33.6 ± 0.2 | 27.0 ± 2.6 | −67.6 ± 2.8 | 29.1 ± 1.2 | 21.7 ± 0.9 | 44.4 ± 2.0 |
| YOLO11s 640 | — | 40 | 41.4 ± 1.1 | 38.0 ± 1.1 | 55.7 ± 2.5 | −48.5 ± 7.1 | 46.7 ± 1.1 | 34.2 ± 1.2 | 74.0 ± 1.2 |
| PNT v1.1 | — | 40 | 33.8 ± 0.3 | 27.6 ± 0.6 | 49.6 ± 1.1 | −141.9 ± 8.0 | 37.0 ± 0.3 | 24.4 ± 0.5 | 76.1 ± 1.6 |
| PNT v2 | — | 40 | 34.3 ± 0.6 | 27.1 ± 0.6 | 54.4 ± 3.1 | −151.0 ± 7.9 | 36.5 ± 1.0 | 23.9 ± 0.8 | 77.6 ± 0.8 |
| YOLO11s 640 | 0.5 m | 40 | 41.4 ± 0.8 | 47.2 ± 0.7 | 45.8 ± 1.4 | −9.2 ± 1.6 | 47.7 ± 0.6 | 38.3 ± 0.8 | 63.3 ± 0.2 |
| PNT v1.1 | 0.5 m | 40 | 36.8 ± 0.7 | 37.1 ± 1.2 | 45.6 ± 0.1 | −63.3 ± 2.5 | 39.6 ± 0.8 | 28.9 ± 0.6 | 63.2 ± 1.8 |
| PNT v2 | 0.5 m | 40 | 35.6 ± 2.2 | 35.0 ± 1.1 | 45.3 ± 5.3 | −78.9 ± 8.9 | 37.6 ± 3.0 | 26.9 ± 2.4 | 62.5 ± 3.8 |

### 7. Precision theo confidence (r1, hộp ảnh toàn khung, IoU 0.5)

- **Cách ghép:** một-một trong từng khung; tổng 41 499 hộp GT.
- **Cột "P / R từ ngưỡng":** precision/recall nếu chỉ giữ các hộp từ mức confidence đó trở lên.
- Đây là lọc SAU tracker, nên chỉ cho biết chiều hướng.

| Confidence | YOLO: hộp (khớp) | YOLO P / R từ ngưỡng | PNT v1: hộp (khớp) | PNT v1 P / R từ ngưỡng |
|---|---|---|---|---|
| ≥ 0.8 | 5 233 (79%) | 0.79 / 0.10 | 49 441 (42%) | 0.42 / 0.51 |
| 0.7–0.8 | 5 843 (77%) | 0.78 / 0.21 | 4 393 (12%) | 0.40 / 0.52 |
| 0.5–0.7 | 10 242 (56%) | 0.68 / 0.35 | 9 331 (10%) | 0.36 / 0.54 |
| 0.3–0.5 | 10 208 (32%) | 0.56 / 0.43 | 12 466 (9%) | 0.31 / 0.57 |
| < 0.3 | 2 753 (22%) | 0.53 / 0.44 | 0 | 0.31 / 0.57 |

v2 cho kết quả giống v1: ở nhóm ≥ 0.8 có 47% hộp khớp; tính cả bảng, P 0.28 / R 0.59.

### 8. Hộp sai nằm ở đâu (r1, `eval/diagnose_fp_region.py`, IoU 0.5)

| Detector | Hộp | TP | FP | FP ngoài lưới | FP trong lưới, gần GT (IoU ≥ 0.3) | FP trong lưới, không trùng ai |
|---|---|---|---|---|---|---|
| YOLO11s 640 | 34 388 | 19 144 | 15 244 | 9 184 (60.2%) | 1 823 | 4 237 |
| PNT v1.1 | 75 796 | 24 426 | 51 370 | 39 719 (77.3%) | 3 217 | 8 434 |
| PNT v2 | 88 265 | 25 462 | 62 803 | 49 322 (78.5%) | 3 220 | 10 261 |

FP tăng thêm so với YOLO:
- v1: +36 126, trong đó 84.5% nằm ngoài lưới; TP thêm +5 282.
- v2: +47 559, trong đó 84.4% nằm ngoài lưới; TP thêm +6 318.

## Vướng mắc / chưa xong

- ~~Chưa chạy trên GPU~~ **Xong** (phần 2):
  - plugin có sẵn trong binary TensorRT 10.3;
  - engine FP16 build được trên cả T4 lẫn A4000;
  - `nvinfer` 7.1 CÓ cắt `topk` khi `cluster-mode=4`, ở dòng 498 của
    `nvdsinfer_context_impl_output_parsing.cpp`.

  Mục 3 của script kiểm báo sai chỉ vì `grep -A8` quá ngắn. Đã sửa thành `-A15`.
- Hàm probe mới (`person_class_id`) **đã chạy thật**: log in "probe giữ lớp 1", fixture có 27–31 hộp/khung.
- Chưa đo VRAM lúc chạy: `nvidia-smi` gọi sau khi pipeline thoát nên chỉ thấy 1 MiB.
- MV3DT vẫn còn thiếu ngưỡng, vùng, công cụ chấm, phần cứng của 27 FPS, và chưa rõ họ chạy video 10 fps
  hay 2 fps.
  - Thêm một câu hỏi mới: họ dùng chính PNT mà đạt IDF1 96.5.
  - Có thể vì hệ của họ theo dõi trên mặt đất trong vùng đã hiệu chỉnh, tức là có sẵn thứ ROI ở quyết
    định 7. **Đó là suy luận.**
- Nhánh `s28-latency-tradeoff` và `s30-peoplenet-transformer` chưa gộp về `main`.

## Bước tiếp theo

1. **ROI mặt đất trên fixture có sẵn** (máy dev, CPU nhẹ; hỏi trước nếu chạy lâu):
   - lọc detection có chân ngoài lưới chú thích, dùng lại phép chiếu của `eval/diagnose_fp_region.py`;
   - chấm lại R640 và PNT, n = 3, cả hai giao thức.

   Chỉ khi PNT + ROI vượt YOLO + ROI ngoài nhiễu mới xét lại quyết định 6.
2. Nếu ROI đáng giữ:
   - đưa vào hệ thống thành một tham số cấu hình (`configs/cameras/`), áp trước `src/mct`;
   - chạy lại trên GPU với ROI lọc trước tracker.
3. Báo cáo thầy 03–04/10, mang theo:
   - bảng 5–6 (PNT không thắng) và bảng 8 (vì sao);
   - bảng 3 (giao thức MV3DT) và hai chỗ tự mâu thuẫn của bài;
   - câu hỏi định vị của phiên 29.

## Nguồn

- [MV3DT, arXiv 2606.13127v1](https://arxiv.org/html/2606.13127v1): mục 4, bảng 1–4
- [deepstream-tracker-3d-multi-view](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/deepstream-tracker-3d-multi-view):
  `config_templates/config_pgie.txt`, `scripts/setup_prerequisites.sh`, `models/PeopleNetTransformer/`
- [deepstream_tao_apps `release/tao_ds7.1ga`](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/tree/release/tao_ds7.1ga):
  config PeopleNet Transformer và `post_processor/` (commit `bc1fa045`)
- [PeopleNet Transformer (NGC)](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/peoplenet_transformer),
  [PeopleNet Transformer v2 (NGC)](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/peoplenet_transformer_v2)
- [TensorRT `multiscaleDeformableAttnPlugin` (release/10.3)](https://github.com/NVIDIA/TensorRT/tree/release/10.3/plugin/multiscaleDeformableAttnPlugin)
- [`nvdsinfer_context_impl_output_parsing.cpp` (NVIDIA/DeepStream)](https://github.com/NVIDIA/DeepStream/blob/main/src/utils/nvdsinfer/nvdsinfer_context_impl_output_parsing.cpp)
- [EarlyBird, arXiv 2310.13350](https://arxiv.org/html/2310.13350), [TrackTacular, arXiv 2403.12573](https://arxiv.org/html/2403.12573): mục đánh giá, r = 1 m
