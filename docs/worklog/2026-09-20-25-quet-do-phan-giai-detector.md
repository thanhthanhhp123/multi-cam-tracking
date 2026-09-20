# 2026-09-20 (phiên 25) — Quét độ phân giải detector (640/960/1280): HOTA toàn khung giảm, nhưng **82% hộp "sai" thêm nằm NGOÀI vùng WildTrack chú thích** — kết luận "không phải đòn bẩy" chưa đứng vững

- **Mốc:** M2 + M6 (đánh giá) | **Máy:** `vast-gpu` Tesla T4 (thuê bằng CLI `vastai`) + máy dev (chấm điểm, CPU) | **Thời lượng:** ~3h (GPU thuê ~1.4h) | **GPU thuê: $0.23**

## Mục tiêu phiên

- Phiên 24 chốt detector là tầng mất nhiều HOTA nhất (≈ 50 điểm; chỉ bắt được ~45% hộp
  ground-truth) và nêu một **giả thuyết chưa kiểm**: ảnh 1920×1080 bị nvinfer co xuống 640
  nên người đứng xa chỉ còn cỡ một phần ba kích thước gốc. Kiểm bằng cách đổi đúng một
  biến — kích thước đầu vào của YOLO11s — và chấm lại bằng cùng đường online.
- Chỉ đo, không đổi cấu hình thật (`pre-cluster-threshold` 0.25 giữ nguyên).

## Đã làm

**Hạ tầng thuê máy.** Instance `51712030` (offer `13790355`: Tesla T4, driver 560.35.03,
CUDA 12.6, 8 814 Mbps, $0.148/h, image `nvcr.io/nvidia/deepstream:7.1-triton-multiarch`,
`--disk 80 --ssh --direct`; lọc `inet_down ≥ 2500` trước như quy ước phiên 2). Lên `running`
sau ~6 phút. Kiểm NVDEC đầu tiên (`PREROLLED → PLAYING → EOS`, CLAUDE.md §11), rồi
`vast_bootstrap.sh` với `FORCE_EXPORT_DEPS=1`. Repo đẩy bằng `git archive HEAD | ssh ... tar x`,
kèm `models/`, chú thích WildTrack và **video 1.1 GB đẩy từ máy dev** (QĐ 4). Hủy bằng
`vastai destroy instance 51712030 -y`, `show instances` = 0, credit 4.126 → 3.896.

**Công cụ mới**
- `docker/vast_detector_res.sh` — chạy N lần pipeline cho một kích thước: tạo thư mục ONNX
  riêng (`models/detector/<tag>/`, vì `.onnx.data` gắn với tên `.onnx`), sinh config nvinfer và
  streams bằng `sed` có kiểm số dòng khớp, build/chép engine, ghi fixture, và **in dòng
  `nvinfer input: ...3xSxS` từ log** làm bằng chứng config có tác dụng. Header ghi cách tái lập.
- `eval/diagnose_fp_region.py` (+7 test, `tests/test_diagnose_fp_region.py`) — với mỗi hộp
  detector không khớp ground-truth, kiểm chân hộp (đáy-giữa, chiếu qua homography camera) có
  nằm trong lưới 12 m × 36 m mà WildTrack thực sự chú thích hay không. Viết sau khi thấy
  `diagnose_junk_ids.py` không trả lời được câu đó (nó chỉ hỏi "có người có chú thích nào trùng
  hộp này không"). Bản dựng lại TP/FP khớp `CLR_TP`/`CLR_FP` của TrackEval trong 0.5%.
- Không sửa `src/` hay `configs/` gốc.

**Dữ liệu sinh ra** (gitignored, trong `data/`): `data/fixtures/ds_wildtrack_7cam_{r640n,r960,r1280}_r{1,2,3}.jsonl`
(+ `.gt.json`, `.gt-report.json`), `data/s25/compare_oracle_tracker.json` (mọi con số),
`data/s25/logs/` (log pipeline + `run_all.log`), `data/s25_{control,r960,r1280,all}.out`.

**Đã thử rồi bỏ / làm sai**
- Lần chạy đầu bị **segfault** (QĐ/cạm bẫy 1) — sửa bằng hạ numpy.
- Lần export đầu thất bại cả ba kích thước (`Invalid weights file`) mà script của tôi không
  báo, vì `| tail -4` che mất mã thoát. Xem cạm bẫy 2.
- Lần chấm đối chứng đầu cho **DetA 2.24** — lỗi của tôi: truyền nhầm `--gt-fixture-table`
  (cạm bẫy 3). Kết quả đó bị loại hẳn, đã chấm lại.

## Quyết định kỹ thuật

**1. Có một đối chứng 640 chạy CÙNG lúc, cùng instance, cùng video — không dùng baseline
phiên 22 làm đối chứng duy nhất.** Ba thứ đã đổi so với phiên 22 và có thể làm lệch: (a) ONNX
640 export lại bằng torch 2.14 có **324 node**, file gốc có **331**; (b) video đóng bằng ffmpeg
của máy dev thay vì của instance; (c) instance khác. Nếu chỉ so 960/1280 với số phiên 22, ba
thứ đó lẫn vào "hiệu ứng độ phân giải". Đối chứng `R640` cho HOTA **15.79 ± 0.47** so với
**15.70 ± 0.46** của phiên 22 (DetA 24.05 vs 24.12, Dets 34 289 vs 34 523) — trùng trong nhiễu,
nên ba thay đổi trên không đáng kể. Chi phí thêm: ~15 phút GPU (~$0.04).

**2. Đổi đúng một biến: cạnh vuông của ONNX (640 / 960 / 1280), `--dynamic`, letterbox
đối xứng của nvinfer giữ nguyên.** Giữ `pre-cluster-threshold=0.25`, `nms-iou-threshold=0.45`,
`topk=300`, tracker, ReID, streammux 1920×1080, `sink.sync: true`. Phương án bị loại: đầu vào
chữ nhật (vd 1088×1920) — đổi hai thứ cùng lúc (độ phân giải *và* tỉ lệ khung/phần đệm), và
không quy được hiệu ứng về "độ phân giải". Cái giá: 1280×1280 vuông đệm gần nửa ảnh, nên thời
gian suy luận không đại diện cho cấu hình chữ nhật tối ưu (không đo, xem "chưa xong").

**3. Chấm bằng đúng đường phiên 24** (`eval.compare_oracle_tracker`, kịch bản A: engine online
→ `export_trackeval --mode mct` → TrackEval), cùng `configs/demo/wildtrack_ds.mct.yaml`, n = 3
mỗi mức. Không thêm `--sct` (giảm tải cho máy dev, và câu hỏi ở đây là detector).

**4. Đẩy video từ máy dev thay vì tải zip EPFL 6.8 GB lên instance.** Zip EPFL từng đứt kết
nối 3 lần, và bước `data` còn phải cài ffmpeg + mã hóa lại 2800 khung. Đẩy 1.1 GB bằng `tar |
ssh` là quy trình repo đã mô tả. Rủi ro (ffmpeg khác bản) được QĐ 1 bao.

**5. Hủy instance ngay sau khi kéo hết dữ liệu về**, không giữ để "phòng khi cần chạy thêm":
build lại 3 engine tốn ~25 phút và ~$0.06, rẻ hơn để máy chạy không.

## Số liệu đo được

**Cấu hình.** WildTrack 7 camera, 400 khung/camera, 2 fps, n = 3 lần chạy pipeline mỗi mức.
`vast-gpu` Tesla T4 15 GB, driver 560.35.03, DeepStream 7.1.0, YOLO11s COCO FP16 (lọc lớp
person), `pre-cluster-threshold` 0.25, NvDCF + ReID OSNet `osnet_x1_0_msdc_dg` TensorRT FP16,
streammux 1920×1080 batch 7, `sink.sync: true`. Engine detector xác nhận từ log nvinfer:
`3x640x640`, `3x960x960`, `3x1280x1280`. Chấm: máy dev, CPython 3.10.20 (`mct-test`) + TrackEval
(`mct-eval`), IoU 0.5, `DO_PREPROC=False`, cấu hình engine liên kết như phiên 24. Mỗi lần chạy
đủ 2800 message.

### 1. Xuyên camera (chuỗi ảo 7 camera), trung bình ± độ lệch chuẩn mẫu, n = 3

| cạnh vào | HOTA | DetA | AssA | AssRe | AssPr | IDF1 | Global ID | #det |
|---|---|---|---|---|---|---|---|---|
| **640** (đối chứng) | **15.79 ± 0.47** | 24.05 ± 0.06 | 10.62 ± 0.66 | 13.42 | 34.35 | 20.27 ± 0.59 | 406 ± 23 | 34 289 |
| **960** | **15.08 ± 0.24** | 22.86 ± 0.04 | 10.17 ± 0.36 | 13.27 | 31.54 | 17.77 ± 0.25 | 499 ± 21 | 48 655 |
| **1280** | **13.67 ± 0.28** | 21.13 ± 0.02 | 9.03 ± 0.36 | 12.10 | 29.82 | 15.24 ± 0.77 | 610 ± 12 | 62 466 |
| (phiên 22, ONNX 640 gốc) | 15.70 ± 0.46 | 24.12 ± 0.08 | 10.47 ± 0.65 | 13.17 | 34.13 | 20.08 ± 0.97 | 407 ± 13 | 34 523 |

### 2. Mức detection (IoU 0.5 với 42 606 hộp ground-truth; TrackEval `CLR_*`)

| cạnh vào | DetRe | DetPr | hộp đúng (TP) | hộp sai (FP) | bỏ sót (FN) | FP thêm / TP thêm so với 640 |
|---|---|---|---|---|---|---|
| 640 | 32.79 ± 0.07 | 40.75 ± 0.05 | 19 102 | 15 187 | 23 504 | — |
| 960 | 37.43 ± 0.06 | 32.77 ± 0.05 | 21 596 | 27 059 | 21 010 | 4.8 |
| 1280 | 40.53 ± 0.04 | 27.65 ± 0.02 | 23 050 | 39 416 | 19 556 | **6.1** |

Cùng chiều từ `tools.ds_wildtrack_gt` (gán track → người): phủ ground-truth **44.9% → 50.8% →
54.2%**, tỉ lệ detection khớp được **55.7% → 44.4% → 36.9%**, số Global ID **406 → 499 → 610**.

### 3. Ước lượng độ tin cậy của hiệu HOTA (Welch thô, n = 3)

| so với 640 | ΔHOTA | sai số chuẩn | t |
|---|---|---|---|
| 960 | −0.71 | ≈ 0.31 | ≈ 2.3 (biên giới) |
| 1280 | −2.12 | ≈ 0.32 | ≈ 6.7 (tách khỏi nhiễu) |

### 4. Chi phí (không phải phép đo hiệu năng — xem "chưa xong")

Build engine detector trên T4 (batch 7, FP16): 640 ≈ 4 phút, 960 ≈ 9 phút, 1280 ≈ 15 phút;
engine ReID ≈ 4 phút (chỉ lần đầu). Mỗi lần chạy 400 khung/camera ≈ 3.5 phút vì video 2 fps
+ `sync=true` — **thời gian này do tốc độ video quyết định, không phải tốc độ suy luận**.

### 5. Hộp "sai" nằm ở đâu — trong hay ngoài vùng chú thích (`eval/diagnose_fp_region.py`, lần chạy r1)

| | hộp | TP | FP | FP **ngoài** vùng | FP trong vùng, chồng lên người (IoU ≥ 0.3) | FP trong vùng, không chồng ai | % FP ngoài vùng |
|---|---|---|---|---|---|---|---|
| 640 | 34 388 | 19 144 | 15 244 | 9 184 | 1 823 | 4 237 | **60.2%** |
| 960 | 48 764 | 21 664 | 27 100 | 18 694 | 2 226 | 6 180 | **69.0%** |
| 1280 | 62 578 | 23 114 | 39 464 | 29 112 | 2 626 | 7 726 | **73.8%** |

**FP thêm so với 640:** 960 → +11 856 (80.2% ngoài vùng); 1280 → **+24 220, trong đó 19 928
(82.3%) ngoài vùng**, chỉ 3 489 (14.4%) là "trong vùng, không chồng ai" (ghost hoặc người bị
sót chú thích) và 803 (3.3%) là lệch vị trí/trùng lặp.

Kiểm chéo với TrackEval: TP/FP dựng lại 19 144/15 244, 21 664/27 100, 23 114/39 464 so với
`CLR_TP`/`CLR_FP` 19 102/15 187, 21 596/27 059, 23 050/39 416.

**Chỉ số detection nếu chỉ tính trong vùng chú thích** (TÔI TỰ TÍNH từ bảng trên, KHÔNG phải HOTA;
ground-truth luôn nằm trong vùng nên recall không đổi):

| | recall @0.5 | precision toàn khung | precision **trong vùng** | F1 toàn khung | F1 **trong vùng** |
|---|---|---|---|---|---|
| 640 | 0.449 | 0.557 | 0.760 | 0.497 | 0.565 |
| 960 | 0.508 | 0.444 | 0.720 | 0.474 | 0.596 |
| 1280 | 0.543 | 0.369 | 0.691 | 0.439 | **0.608** |

Cùng dữ liệu, hai cách tính cho hai chiều ngược nhau: toàn khung → càng phân giải cao càng
tệ; trong vùng chú thích → càng tốt.

### `diagnose_junk_ids.py` trên r1 (so sánh với phiên 21: 217 Global ID rác, 73.5% khối lượng "báo nhầm")

| | Global ID rác | % số khung toàn hệ thống | "báo nhầm" (%ID / %khung rác) | "người thật bị bảng nhãn loại" (%ID / %khung rác) |
|---|---|---|---|---|
| 640 | 202 | 29.7% | 70.3% / 77.1% | 29.7% / 22.9% |
| 1280 | 368 | 46.8% | 79.3% / 86.1% | 20.7% / 13.9% |

640 tái lập phiên 21 (217 → 202, 73.5% → 77.1%). **Nhưng "báo nhầm" ở công cụ này nghĩa là
"không có người có chú thích nào trùng hộp"** — mục 5 cho thấy 60–74% hộp như vậy nằm ngoài vùng
chú thích, nên nhãn "detector báo nhầm" của phiên 21 lẫn cả người thật ngoài vùng.

## Kết luận

**HOTA toàn khung giảm theo độ phân giải (15.79 → 15.08 → 13.67), nhưng đó phần lớn là
artifact của vùng chú thích, nên KHÔNG kết luận được "độ phân giải cao hơn là xấu hơn".**

- **Hiệu ứng thật, đúng giả thuyết:** recall @IoU 0.5 tăng 44.9% → 50.8% → 54.3% (+9.3 điểm) —
  640 đúng là co mất một phần người nhỏ.
- **Vì sao HOTA vẫn giảm:** 82.3% (19 928/24 220) số hộp "sai" thêm ở 1280 có chân nằm NGOÀI lưới
  12 m × 36 m mà WildTrack chú thích — người thật (hoặc vật) mà ground-truth không có, TrackEval
  vẫn đếm là FP. Ở 640 cũng đã 60.2% FP nằm ngoài vùng.
- **Chỉ tính trong vùng chú thích**, precision giảm nhẹ (0.760 → 0.720 → 0.691) trong khi recall
  tăng, và F1 tăng 0.565 → 0.596 → **0.608** — ngược chiều HOTA toàn khung. (Tôi tự tính từ số
  đếm hộp, **chưa phải HOTA chỉ-trong-vùng**; đó là phép đo còn thiếu, xem bước 1.)
- Chỉ 14.4% FP thêm là "trong vùng, không chồng ai" (ghost hoặc người bị sót chú thích).

**Tác động lên kết luận của các phiên trước** (chỉ ghi nhận, chưa sửa các worklog đó):

- Phiên 21 ("73.5% khối lượng Global ID rác là hộp detector báo nhầm") và DetA ≈ 24 / DetPr ≈ 41
  của kịch bản A phiên 24: "báo nhầm" ở đó nghĩa là "không có người có chú thích nào trùng hộp",
  lẫn cả người thật ngoài vùng. **Precision của detector bị đánh giá thấp** ở mọi số toàn khung
  của kịch bản A — ở 640 là 0.557 toàn khung so với 0.760 trong vùng.
- Phần chia tracker/liên kết của phiên 24 **không bị ảnh hưởng**: nó dựa trên các kịch bản B/C/LB/LC,
  vốn dùng hộp ground-truth (luôn trong vùng). Chỉ khoản "A → C ≈ 10.4 HOTA" bị phồng.

**Quyết định cấu hình:** không đổi trong phiên này (vẫn YOLO11s 640, ngưỡng 0.25). Bản đầu của
kết luận này ghi "giữ 640" như thể đã có bằng chứng; sau mục 5 thì chưa có. Chỉ nên quyết sau khi
có HOTA chỉ-trong-vùng.

## Vướng mắc / chưa xong

- **Phép phân loại "ngoài vùng" là xấp xỉ.** Dựa trên chân hộp (đáy-giữa) chiếu qua homography
  khớp trên các điểm TRONG lưới (RMSE 1.8 cm trong lưới; ngoài lưới không kiểm chứng). Hộp cắt ở
  mép dưới ảnh có chân nằm ngoài khung nên chiếu lệch. Nhóm "trong vùng, không chồng ai"
  (4 237 → 7 726 hộp) vẫn lẫn ghost với người bị sót chú thích — chưa tách được. Và mới chạy
  **lần r1** của mỗi mức (n = 1), chưa chạy r2/r3 (mức phủ ground-truth giữa r1–r3 lệch < 0.2%
  nên kỳ vọng ổn định, nhưng chưa kiểm).
- **Chưa quét lại ngưỡng ở 960/1280.** 0.25 chọn ở 640. Ở phiên 22, tại 640 nâng 0.25 → 0.40
  gần hòa (HOTA 15.70 → 16.00, trong nhiễu). Độ phân giải cao có thể cần ngưỡng cao hơn để trả
  lại precision — **chưa đo**, đừng suy diễn từ đường cong 640.
- **Không đo FPS/độ trễ.** Video 2 fps + `sync=true` nên tốc độ không bị chặn bởi suy luận;
  đầu vào vuông 1280 còn đệm ~nửa ảnh. Đừng chép thời gian ở mục 4 sang chương 6.
- Chỉ một model (YOLO11s), chỉ ba mức, chỉ đầu vào vuông, chỉ WildTrack 2 fps — người nhỏ
  ở camera thật của đồ án có thể khác.
- ONNX 640 export mới (324 node) ≠ file gốc (331 node): cờ export gốc không được ghi lại. QĐ 1
  chứng minh không ảnh hưởng đáng kể tới HOTA, nhưng chưa biết khác ở đâu (`--simplify`?).
- Máy dev chạy engine online 9 lần và TrackEval 21 lần (3 lần hỏng do lỗi ở cạm bẫy 3, 9 lần
  là chạy gộp lại để có một file JSON đầy đủ); tuần tự, mỗi lần dưới ~2 phút.

## Cạm bẫy phát hiện (đã thêm vào CLAUDE.md §11)

1. **`FORCE_EXPORT_DEPS=1` (ultralytics) nâng numpy `1.26.4 → 2.2.6` trong image
   DeepStream, và `nvtracker` SEGFAULT (exit 139) ngay lúc build engine ReID** — im lặng: log
   chỉ dừng sau `Begin building engine for tracker ReID...`. Chỉ đổi numpy về `1.26.4` thì
   pipeline chạy (exit 139 → chạy tới `timeout`). Bằng chứng một biến; không biết cơ chế bên
   trong. `import pyds` vẫn báo OK nên bootstrap không bắt được. Sau khi export xong phải
   `pip install numpy==1.26.4`.
2. **`export_yolo11.py` đòi `yolo11s.pt` có sẵn ở cwd** (`Invalid weights file`), không tự tải.
   Tải bằng `ultralytics.YOLO('yolo11s.pt')` trước. Đừng bọc lệnh trong `| tail`: che mất mã thoát.
3. **`eval.compare_oracle_tracker --gt-fixture-table` là bảng của fixture GROUND-TRUTH
   (`wildtrack_7cam.gt.json`), không phải bảng của fixture pipeline.** Truyền nhầm → không lỗi
   nào, DetA sụp 24 → 2.24 dù số detection giống hệt. Phát hiện nhờ đối chiếu `#det`
   với baseline. Ngoài ra công cụ **ghi đè** `compare_oracle_tracker.json` mỗi lần gọi: gọi một
   lượt cho mọi `--run` cần so.
4. **CLI `vastai` trên máy dev** nằm ở `%APPDATA%\Python\Python313\Scripts\vastai.exe` (không
   trên PATH), khóa ở `~/.config/vastai/vast_api_key`. `destroy instance` **hỏi xác nhận và
   tự hủy bỏ khi không có stdin** — cần `-y`, và phải kiểm `show instances` sau đó.

## Bước tiếp theo

1. **Chấm HOTA/DetA/AssA CHỈ TRONG vùng chú thích** cho 9 lần chạy có sẵn — phép đo còn thiếu để
   biết độ phân giải có thật sự giúp hay không. CPU, không cần GPU, không chạy lại engine: thêm cờ
   loại hộp tracker có chân ngoài lưới vào `tools/export_trackeval.py` (dùng homography 7 camera),
   rồi chạy TrackEval trên các DB `data/s25/R*_r*/mct.db`. Lưu ý đây là lọc **hậu kỳ khi chấm**
   (như vùng ignore), khác với lọc trước khi tracking — engine đã thấy các hộp ngoài vùng.
2. Nếu (1) cho thấy HOTA trong vùng tăng theo độ phân giải thì mới cân nhắc đổi detector sang
   960 và quét `pre-cluster-threshold` ở đó — cần GPU thuê lại, xin xác nhận trước (CLAUDE.md §2).
   Nếu HOTA trong vùng đi ngang hoặc giảm thì giữ 640.
3. Sau (1), cập nhật CLAUDE.md §7: số WildTrack toàn khung đánh giá **thấp** detector (60% FP nằm
   ngoài vùng chú thích) — mọi số DetA/HOTA của kịch bản A phải ghi kèm cách tính vùng.
4. Chạy `diagnose_fp_region.py` cho r2, r3 nếu cần khoảng tin cậy.
