# 2026-09-04 (phiên 8) — M3: Re-ID bằng OSNet pretrained, đường (A) qua nvtracker

- **Mốc:** M3 (tuần 8–10) | **Máy:** máy dev (soạn) + `ut-hpc` (xuất model, test/lint) | **Thời lượng:** ~1.5h

## Mục tiêu phiên

- Làm M3 với model pretrained theo quyết định phiên 7: bật ReID trong `nvtracker`, đưa
  embedding qua `mct:frames` để `src/mct` có ngoại hình thật từ pipeline (chứ không chỉ từ
  fixture WildTrack sinh bằng ONNX Runtime trên CPU).
- Làm hết phần **không cần GPU**, để lần thuê `vast-gpu` tới chỉ còn việc xác minh và đo.

## Đã làm

### Tra tài liệu trước khi viết (CLAUDE.md §11: "đừng đoán")

Chốt API và tên tham số từ nguồn **khớp đúng phiên bản**, không suy từ bản khác:

- `pyds` v1.2.0 (bản đi kèm DeepStream 7.1, chính là bản `docker/deepstream.Dockerfile`
  cài): `bindings/src/bindtrackermeta.cpp` + `tests/integration/test.py` của
  `NVIDIA-AI-IOT/deepstream_python_apps` cho đường đọc embedding —
  `pyds.NvDsMetaType.NVDS_TRACKER_OBJ_REID_META` → `pyds.NvDsObjReid.cast(...)` →
  `.get_host_reid_vector()`, kích thước `.featureSize`.
- Danh sách khoá của khối `ReID:` và `TrajectoryManagement:` lấy từ tài liệu công khai
  NvMultiObjectTracker (`NVIDIA-AI-IOT/DeepStream_Coding_Agent`, reference tracker_config).

### Model

- Xuất lại checkpoint OSNet **khái quát hoá miền** (`osnet_x1_0_msdc_dg`, bản thắng +25%
  ở phiên 4) thành ONNX **một file** trên `ut-hpc`: bản cũ để weight ở file ngoài
  (`.onnx` 851 KB + `.onnx.data` 8.6 MB), thêm một đường hỏng khi TensorRT dựng engine trên
  máy khác. Bản mới 9.5 MB, tự chứa.
- Xác nhận trực tiếp trên file: input `images [batch, 3, 256, 128]` (NCHW, batch động),
  output `features [batch, 512]`, opset 18. Đây là nguồn cho `reidFeatureSize: 512`, không
  phải trí nhớ.
- Chép về `models/reid/osnet_x1_0_msdc_dg.onnx` (gitignored, đúng quy tắc bất biến 5).

### Code + config

- **`src/ds_pipeline/reid_meta.py`** (mới) — gom toàn bộ phần phụ thuộc phiên bản
  DeepStream vào một file. Thử đường (A) trước, (B) sau. `pyds` import **bên trong hàm**
  để module nạp được trên máy dev.
- **`src/ds_pipeline/probes.py`** — dùng `reid_meta`; xoá hai hàm đọc tensor cũ và hằng
  `_REID_USER_META_TAG` (khai báo từ M1 nhưng **chưa từng được dùng ở đâu**).
- **`configs/pipeline/config_tracker_NvDCF_reid.yml`** (mới) — bản có ReID, khác
  `config_tracker_NvDCF_perf.yml` đúng khối ReID + tham số re-assoc.
- **`configs/pipeline/streams_reid.yaml`** (mới) — 4 luồng có ReID, đối chứng của
  `streams_multi.yaml`.
- **`tests/test_reid_meta.py`** (13 test) + phần tracker trong **`tests/test_pipeline_configs.py`**
  (5 test). `Makefile`: `ds-run-reid`. `CLAUDE.md` §11 + §12.
- Tổng: **286 passed, 7 skipped**, ruff sạch (`ut-hpc`, Python 3.10.12).

## Quyết định kỹ thuật

**1. Đường (A) — ReID trong nvtracker — chốt, (B) giữ làm dự phòng trong code.**
CLAUDE.md §11 để ngỏ từ đầu dự án. Đường (A) rẻ hơn vì tracker đã crop sẵn đối tượng, và
DS 7.1 hỗ trợ đầy đủ. Nhưng `reid_meta.extract_reid_embedding` vẫn duyệt cả hai loại user
meta trong **một** vòng lặp: nếu đổi sang bản DeepStream không có hằng meta type của
tracker, code tự rơi xuống (B) và ghi log một lần, thay vì ném `AttributeError` giữa probe.
Có test cho đúng nhánh đó bằng pyds giả.

**2. `outputReidTensor: 1` — khoá quan trọng nhất của cả phiên.** Không có nó, tracker vẫn
trích embedding và vẫn dùng nội bộ để tái liên kết, nhưng **không gắn vào user meta**.
Triệu chứng: pipeline chạy "thành công", FPS đẹp, `probe` đọc ra `embedding=None`, `src/mct`
không có gì để so, mọi người thành Global ID riêng. Không log lỗi ở đâu cả. Đây đúng là họ
lỗi mà dự án này đã dính hai lần (homography thiếu topology ở phiên 6; lọc lớp ở phiên 7),
nên đưa thẳng vào CLAUDE.md §11 và có test canh.

**3. `keepAspc: 0`, ngược với config mẫu của NVIDIA.** Mẫu của NVIDIA để `1` vì model
`resnet50_market1501` của họ được train với letterbox giữ tỉ lệ. OSNet của torchreid thì
**resize thẳng** về 256×128 — và `src/tools/reid_onnx.py` (thứ đã sinh fixture WildTrack)
cũng đang `cv2.resize` trần. Bê nguyên `keepAspc: 1` sang là cho model ăn ảnh có viền đen
mà nó chưa từng thấy lúc train, đồng thời làm hai đường sinh embedding lệch nhau. Có test
ràng `keepAspc == 0` cùng với `inferDims`, `colorFormat`, `inputOrder`.

**4. Chấp nhận một chênh lệch không khử được, và ghim nó lại bằng test.** DeepStream chuẩn
hoá bằng `y = netScaleFactor * (x - offset)` với `netScaleFactor` là **một số vô hướng**,
trong khi ImageNet có std riêng từng kênh `[0.229, 0.224, 0.225]`. Buộc phải lấy std trung
bình 0.226 → `netScaleFactor = 1/(255×0.226) = 0.01735207`. Hệ quả: embedding từ pipeline
DeepStream **không** trùng khít embedding từ `tools/reid_onnx.py` — lệch ~1.3% ở kênh R và
B, cộng thêm FP16 vs FP32. Ngưỡng `max_cost = 0.60` trong `configs/demo/wildtrack.mct.yaml`
được chỉnh trên đường ONNX Runtime, nên **phải kiểm lại** trên fixture sinh từ pipeline
thật. Viết test so `offsets`/`netScaleFactor` với hằng số trong `reid_onnx.py` để sai số đó
là một lựa chọn có ý thức, không phải thứ trôi đi lúc nào không hay.

**5. `reidType: 2` (REASSOC), không phải 1 (NvDeepSORT).** NvDeepSORT đưa ReID vào thẳng
vòng data association của tracker — đổi hành vi bám của NvDCF và kéo theo một bộ ngưỡng nữa
phải tune. Đóng góp của đồ án nằm ở liên kết **xuyên** camera (`src/mct`), không ở tracker
đơn camera; cái cần từ tracker là (a) embedding chất lượng ổn định và (b) `local_track_id`
sống lâu qua che khuất. REASSOC cho đúng hai thứ đó với ít biến hơn.

**6. `reidExtractionInterval: 0` — trích đặc trưng mọi frame.** `src/mct` lấy query
embedding là trung bình có trọng số của top-k embedding confidence cao nhất trong cả
tracklet (CLAUDE.md §6), không phải embedding frame cuối. Lấy thưa thì top-k không còn gì
để chọn. Đây là tham số đầu tiên nên nới nếu FPS không đạt — ghi sẵn vào comment cho M6.

**7. Hai cặp file đối chứng, và test bắt chúng phải là đối chứng thật.**
`config_tracker_NvDCF_{perf,reid}.yml` và `streams_{multi,reid}.yaml`. Test khẳng định mọi
khối ngoài `ReID`/`TrajectoryManagement` giống hệt nhau, và hai file streams chỉ khác đúng
`ll_config_file`. Nếu không có ràng buộc đó thì chênh lệch FPS đo được sau này không quy
được về "chi phí của ReID" — mà một con số không quy được về nguyên nhân thì không viết
vào chương 6 được.

**8. Copy vector ra khỏi bộ nhớ của DeepStream.** `get_host_reid_vector()` trả về numpy
array *bọc* con trỏ `ptr_host` của meta, không sao chép; meta bị giải phóng ngay khi probe
trả về. Giữ view đó là đọc bộ nhớ đã chết — **không crash**, chỉ ra dữ liệu rác. Test mô
phỏng bằng cách ghi đè mảng nguồn sau khi gọi.

## Số liệu đo được

Không đo hiệu năng trong phiên này (không có GPU). Số liệu lấy trực tiếp từ file model,
làm căn cứ cho config:

| hạng mục | giá trị | nguồn |
|---|---|---|
| input ONNX | `images [batch, 3, 256, 128]`, NCHW | đọc từ chính `.onnx` |
| output ONNX | `features [batch, 512]` | như trên |
| opset | 18 (TensorRT 10.3 nhận) | như trên |
| kích thước ONNX một file | 9.5 MB (trước: 851 KB + 8.6 MB ngoài) | `ls` sau khi xuất lại |
| checkpoint chọn | `osnet_x1_0_msdc_dg` (đa nguồn, DG) | phiên 4: +25% F1 so với Market-1501 |

Test: **286 passed, 7 skipped** (trước phiên: 267/7), ruff sạch — `ut-hpc`, Python 3.10.12.

## Vướng mắc / chưa xong

**Toàn bộ phần M3 chạm phần cứng vẫn còn nguyên.** Những gì đã làm là "soạn đúng theo tài
liệu + có test canh", không phải "đã chạy". Checklist cho lần thuê `vast-gpu` tới, đã ghi
luôn vào đầu `config_tracker_NvDCF_reid.yml`:

1. **Đọc log khởi động nvtracker.** NvMultiObjectTracker bỏ qua im lặng khoá lạ hoặc khoá
   đặt sai khối. Nghi ngờ lớn nhất: `reidExtractionInterval` — tài liệu công khai xếp nó
   trong `TrajectoryManagement`, chưa xác nhận trên bản 7.1 thật.
2. **Xác nhận probe đọc được embedding 512-d** (`embed_dim=512` trong `FrameMessage`).
   Đây là chỗ `outputReidTensor` và cả đường (A) được kiểm thật sự.
3. **Đo chi phí FPS của ReID**: `streams_reid.yaml` vs `streams_multi.yaml`, cùng máy, cùng
   nguồn, `--stats`.
4. **Ghi fixture thật** với `sink.sync: true`, rồi chạy lại engine liên kết trên fixture đó
   để xem `max_cost` có phải chỉnh lại không (xem Quyết định 4).

Còn lại:

- **`tracker-width/height = 960×544`** trong `streams*.yaml`: tracker làm việc trên frame
  đã thu nhỏ một nửa từ 1080p, nên crop đưa vào ReID cũng từ ảnh nửa độ phân giải. Với
  người ở xa, crop có thể quá nhỏ để OSNet cho embedding tốt. Chưa đo — cần thử nâng lên
  1920×1080 và so chất lượng liên kết với chi phí VRAM.
- `batchSize: 100` cho mạng ReID chưa đối chiếu với `maxTargetsPerStream: 150` × 4 luồng.
  Chưa rõ tracker xử lý thế nào khi số đối tượng vượt batch; phải xem log.
- VRAM: YOLO + ReID cùng lúc trên 4 luồng là kịch bản CLAUDE.md §11 cảnh báo dễ chạm trần.
  RTX 3090 24GB thì thoải mái, nhưng số đo phải ghi kèm để biết ngưỡng ở GPU 8GB.

## Bước tiếp theo

1. Thuê `vast-gpu` (**hỏi xác nhận trước, tính phí theo giờ**) và chạy hết checklist 4 mục
   ở trên trong một chuyến, gộp luôn hai việc còn treo: xác minh khối lọc lớp person của
   phiên 7, và đo độ trễ end-to-end của phiên 6.
2. Sau khi có fixture thật có embedding: chạy lại `eval/` để xem ngưỡng `max_cost` đo trên
   đường ONNX Runtime có chuyển được sang đường DeepStream không.
3. Cảnh báo "hai Global ID quá gần nhau trên mặt phẳng" — treo từ phiên 6.
