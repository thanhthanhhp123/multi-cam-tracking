# 2026-09-28 — Chuẩn bị PeopleNet Transformer trên DeepStream 7.1, đọc giao thức chấm của MV3DT

- **Mốc:** M2 (đổi detector) + M6 (baseline) | **Máy:** máy dev (Windows), không thuê GPU | **Thời lượng:** ~2h

## Mục tiêu phiên

Hai việc trên máy dev ở mục "Bước tiếp theo" của phiên 29 (bước 1 và 3). Làm để lần thuê `vast-gpu`
tới chỉ còn việc chạy.
- Bước 1: xác định định dạng file PeopleNet Transformer và yêu cầu plugin TensorRT, viết config
  `nvinfer`, kiểm bước hậu xử lý bbox.
- Bước 3: đọc bài MV3DT, chép đúng giao thức chấm WildTrack.

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

## Vướng mắc / chưa xong

- **Chưa chạy trên GPU.** Chưa biết:
  - binary `libnvinfer_plugin` 10.3 có plugin không;
  - engine FP16 có build được không;
  - `nvinfer` 7.1 có cắt `topk` không.

  Cả ba nằm trong `docker/vast_peoplenet_check.sh`.
- Hàm probe mới (`person_class_id`) chỉ kiểm được bằng test cấu hình. Code `ds_pipeline` chưa chạy
  thật sau khi sửa.
- MV3DT còn thiếu: ngưỡng, vùng, công cụ chấm, phần cứng của 27 FPS, và video 10 fps hay 2 fps.
  Không có cách nào xác minh ngoài hỏi tác giả, hoặc tự chạy reference app (cần DS 9, Ubuntu 24.04,
  driver ≥ 580).
- Nhánh `s28-latency-tradeoff` vẫn chưa gộp về `main`. Việc của phiên này chưa commit.

## Bước tiếp theo

1. Thuê `vast-gpu` (**hỏi người dùng trước**; T4 cho khớp các số trước, khoảng 1–1.5 giờ):
   - kiểm NVDEC (CLAUDE.md §11);
   - chạy `vast_bootstrap.sh`, rồi `docker/fetch_peoplenet_transformer.sh`, rồi
     `docker/vast_peoplenet_check.sh`;
   - WildTrack n = 3 cho `streams_wildtrack_pnt.yaml` và `streams_wildtrack_pnt2.yaml`, ghi fixture
     theo cách của `docker/vast_detector_res.sh`;
   - FPS 4 luồng với engine b7.
2. Trên máy dev: chấm cả hai giao thức cho v1 và v2, so với YOLO11s 640:
   - hộp ảnh toàn khung: HOTA 15.70 ± 0.46;
   - mặt đất trong vùng: 31.10 ± 0.85.

   Thêm biến thể 40 khung test để đặt cạnh MV3DT kèm ghi chú ở quyết định 4.
3. Nếu PNT thắng rõ: đổi `streams_reid.yaml`/`streams_latency.yaml` sang detector mới, sau khi đo
   FPS. Nếu v2 quá chậm cho 4 luồng: dùng v1, hoặc `interval=1`.
4. Báo cáo thầy 03–04/10: thêm bảng 3 (giao thức) và hai chỗ tự mâu thuẫn của MV3DT.

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
