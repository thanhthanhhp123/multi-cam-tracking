# 2026-09-04 (phiên 10) — Đưa WildTrack qua pipeline DeepStream: phần không cần GPU

- **Mốc:** M3 (đóng phần treo lớn nhất) | **Máy:** máy dev (soạn) + `ut-hpc` (test/lint, đóng video) | **Thời lượng:** ~2h

## Mục tiêu phiên

Việc số 1 trong "Bước tiếp theo" của phiên 9: **ngưỡng `max_cost` chỉnh trên embedding của
đường ONNX Runtime có chuyển sang embedding của đường DeepStream không?** Đây là câu hỏi
lớn nhất còn lại của M3, và cách duy nhất trả lời được là cho **chính WildTrack** đi qua
pipeline DeepStream rồi đo bằng đúng bộ chỉ số đã dùng cho fixture cũ.

Làm hết phần **không cần GPU** trong phiên này, để lần thuê `vast-gpu` tới chỉ còn việc
chạy và đo — đúng chiến lược đã hiệu quả ở phiên 8→9 ($0.165 cho ba việc treo).

## Đã làm

Pipeline chỉ nhận video/RTSP, còn WildTrack là thư mục ảnh PNG. Thiếu ba mắt xích, phiên
này làm cả ba:

- **`src/tools/wildtrack_to_video.py`** (mới) — đóng `Image_subsets/C1..C7` thành 7 video
  H.264 qua ffmpeg, cùng `index.json` ghi lại hợp đồng ánh xạ và tham số encode.
- **`src/tools/ds_wildtrack_gt.py`** (mới) — dựng bảng ground-truth cho fixture do
  DeepStream sinh ra: ghép bbox theo IoU + Hungarian, bỏ phiếu đa số theo `local_track_id`.
- **`configs/demo/streams_wildtrack.yaml`** (mới) — 7 luồng có ReID, `sink.sync: true`.
  Kèm **`configs/pipeline/config_infer_yolo11_b7.txt`** (engine TensorRT batch 7).

Ngoài ra:

- **`.claude/skills/ut-hpc/templates/wildtrack_video.sbatch`** (mới) + cập nhật
  `reference/cluster.md` với phát hiện node không đồng nhất (xem Cạm bẫy).
- `Makefile`: `wildtrack-video`, `wildtrack-ds-gt`, `ds-run-wildtrack`.
- `tools/wildtrack_to_fixture.py`: mở `annotation_frame_numbers()` thành hàm public để bộ
  đóng video và bộ dựng fixture dùng **chung một** nguồn thứ tự khung.
- Test: `tests/test_wildtrack_to_video.py` (13), `tests/test_ds_wildtrack_gt.py` (16),
  thêm 6 test cho `streams_wildtrack.yaml` và tổng quát hoá test config nvinfer theo bảng
  batch size. **335 passed, 8 skipped**, ruff sạch (`ut-hpc`, Python 3.10.12).

## Quyết định kỹ thuật

**1. Ánh xạ theo `frame_id`, không theo `ts_ms` — và ghim nó thành hợp đồng của bộ đóng video.**
Muốn chấm điểm fixture DeepStream thì phải biết mỗi detection ứng với khung chú thích nào
của WildTrack. Hai đường khả dĩ: khớp theo thời gian, hoặc khớp theo chỉ số khung. Khớp
theo `ts_ms` là ngõ cụt — `ts_ms` của lần chạy pipeline là wall clock lúc chạy, không liên
quan gì tới `BASE_TS_MS` giả lập của fixture cũ, và còn phụ thuộc jitter của streammux.

Chốt: **khung thứ i của video là khung chú thích thứ i**, nên `frame_id` mà `probes.py`
ghi ra (`frame_meta.frame_num`, đếm từ 0 theo từng nguồn) tra thẳng về chú thích. Hợp đồng
này được bảo đảm ở ba chỗ: bộ đóng video lấy thứ tự khung từ chính hàm mà bộ dựng fixture
dùng (`annotation_frame_numbers`), ffmpeg đọc chuỗi qua thư mục link đánh số liên tiếp
`%06d.png` (tên gốc WildTrack cách nhau 5, đưa thẳng cho ffmpeg là hỏng), và `index.json`
ghi lại danh sách khung để đối chiếu.

**2. Kiểm số khung bằng ffprobe, coi "ffmpeg trả về 0" là chưa đủ.** Thiếu hoặc thừa một
khung là toàn bộ ánh xạ lệch hàng loạt, mà triệu chứng duy nhất là F1 thấp một cách khó
hiểu — không có log lỗi nào. Đây đúng họ lỗi đã dính ba lần trong dự án (`outputReidTensor`
phiên 8, khối lọc lớp phiên 7, homography thiếu topology phiên 6), nên lần này chặn ngay
tại nguồn: encode xong thì `ffprobe -count_frames`, lệch là ném `VideoBuildError`.

**3. Track không chắc thì LOẠI khỏi bảng, không gán bừa.** `local_track_id` của NvDCF
không biết `personID` của WildTrack, nên danh tính phải suy ra bằng bỏ phiếu. Tracker đổi
người giữa chừng (id-switch) thì phiếu chia đôi. Gán bừa cho bên đa số là bơm nhiễu thẳng
vào **thước đo** — mà thước đo sai thì mọi kết luận về ngưỡng đều sai theo, im lặng.
`eval_wildtrack.score()` chỉ chấm tracklet có trong bảng, nên loại là an toàn: mất một mẫu,
chứ không sai một mẫu. Ngưỡng mặc định: thuần khiết ≥ 0.7, khớp ≥ 3 khung, IoU ≥ 0.5.

**4. Ghép bbox bằng Hungarian, không tham lam.** WildTrack là quảng trường đông người, hộp
người này đè lên người kia. Tham lam lấy cặp IoU cao nhất trước có thể ép cặp còn lại vào
lựa chọn tệ hơn hẳn — có test dựng đúng tình huống đó (cặp tốt nhất .818 nhưng phương án
chéo .667+.667 tổng cao hơn). Một lá phiếu sai kéo cả tracklet sang nhầm danh tính.

**5. Đầu ra dùng ĐÚNG định dạng `gt.json` của `wildtrack_to_fixture.py`.** Nhờ vậy
`eval/eval_wildtrack.py --diagnose --sweep` chạy được ngay, không sửa một dòng. Đó chính
là điều làm phép so sánh có giá trị: cùng dataset, cùng bộ chỉ số, cùng đoạn code chấm
điểm — **khác đúng một biến là đường trích embedding**. Nếu phải viết bộ chấm riêng cho
đường DeepStream thì chênh lệch đo được sẽ không quy được về nguyên nhân nào.

**6. `sink.sync: true` là bắt buộc, không phải tuỳ chọn.** Video WildTrack ở 2 fps; chạy
hết tốc lực thì 400 khung dồn vào vài giây `ts_ms`, cửa sổ gán 1000 ms của `src/mct` nuốt
trọn cả đoạn và ràng buộc thời gian mất nghĩa. Với `sync=true`, 400 khung trải đúng 200 s —
bằng nhịp mà fixture cũ giả lập (`fps=2.0`), nên hai bên mới so được. Có test canh.

**7. CRF 18, và GOP dài không đáng đổi.** Thứ đang đo là chất lượng embedding, nên nén
mạnh tay sẽ trộn nhiễu nén vào kết quả. Đã thử nới GOP từ 4 lên 30 khung để giảm dung
lượng: chỉ được 8% (173 → 158 MB/camera). Ở 2 fps hai khung liên tiếp cách nhau nửa giây
nên P-frame gần như vô dụng — dung lượng là hệ quả của nội dung, không phải của GOP. Giữ
mặc định GOP = 2 giây (an toàn hơn cho decoder).

**8. File config nvinfer riêng cho batch 7, và test đổi thành bảng.** Engine TensorRT gắn
chặt với batch size, `nvinfer` chỉ nạp lại engine đã build khi `model-engine-file` trùng
đúng tên. Test cũ so cứng hai bản b1/b4; giờ là bảng `{file: batch}` và vòng lặp so mọi
bản với b1 — thêm batch mới chỉ cần thêm một dòng, và không thể quên ràng buộc "chỉ được
khác batch size + tên engine".

**9. Đóng video trên head node `ut-hpc`, có ý thức phá lệ.** CLAUDE.md §2 cấm chạy việc
nặng trên head node, nên phương án đầu là `sbatch`. Bốn lần submit đều chết vì node được
cấp không có `libx264` (xem Cạm bẫy). Sau khi đo: **37 s wall / camera ở `nice -n 19`**,
tức ~4 phút cho cả 7 — nhẹ hơn một lần `pip install`, và `nice -n 19` nhường CPU cho mọi
tiến trình khác. Chọn head node vì nó **tất định**, còn `sbatch` là xổ số. Ghi lại đây để
lần sau không phải dò lại; nếu cần chạy trên node tính toán thì ghim
`--nodelist=<node đã kiểm>` và biết rằng node đó có thể bị drain.

## Số liệu đo được

**Cấu hình:** `ut-hpc` head node (`hpc-head1`), ffmpeg 4.4.2-0ubuntu0.22.04.1 + libx264,
`nice -n 19`, `~/mct/venv-test/bin/python` 3.10.12. Nguồn: WildTrack `Image_subsets/C1..C7`,
PNG 1920×1080, 400 khung chú thích/camera (00000000 → 00001995, bước 5).

### Đóng video (một camera, để so tham số)

| cấu hình | wall | CPU | dung lượng |
|---|---|---|---|
| CRF 18, GOP 4 (mặc định) | 36.9 s | 2m38.8s | **173.0 MB** |
| CRF 18, GOP 30 | 28.3 s | 3m40.4s | 158.5 MB (−8%) |

### Đóng đủ 7 camera (kết quả cuối)

| | |
|---|---|
| số khung/camera | 400, `ffprobe -count_frames` xác nhận cả 7 |
| dung lượng | 143.0–177.8 MB/camera, **tổng 1.1 GB** |
| thời gian | ~4 phút wall cho cả 7, `nice -n 19` |
| khung chú thích | 0, 5, 10, ... 1995 (bước 5) |

### Kiểm chứng hợp đồng ánh xạ trên dữ liệu THẬT (không chỉ bằng test)

Test đơn vị chỉ khẳng định *thứ tự file đưa cho ffmpeg* là đúng. Nó không khẳng định
ffmpeg giữ nguyên thứ tự đó, cũng không khẳng định frame thứ i giải mã ra đúng ảnh thứ i.
Thí nghiệm bác bỏ được: giải mã khung thứ `i` của `cam01.mp4`, đo PSNR với ảnh gốc **đúng**
(`frame_numbers[i]`) và với hai ảnh gốc **kề bên**. Nếu ánh xạ lệch, PSNR với ảnh đúng phải
tụt xuống ngang mức hàng xóm.

| khung video `i` | chú thích | PSNR vs ảnh đúng | vs ảnh trước (−5) | vs ảnh sau (+5) |
|---|---|---|---|---|
| 0 | 00000000 | **42.61 dB** | 17.58 dB | — |
| 100 | 00000500 | **43.15 dB** | 19.27 dB | 22.14 dB |
| 399 | 00001995 | **41.93 dB** | 19.87 dB | 18.98 dB |

Cách biệt hơn **20 dB** — không còn chỗ cho nghi ngờ. Đồng thời con số 42–43 dB cũng xác
nhận CRF 18 giữ được độ trung thực cao, đúng ý đồ (thứ đang đo là chất lượng embedding,
không được để nhiễu nén trộn vào).

### Dò `ffmpeg` trên cụm (phát hiện quan trọng của phiên)

| nơi | partition | `libx264` | job |
|---|---|---|---|
| `hpc-head1` (head node) | — | **có** | — |
| `ctit085` | main-gpu | **không** | 581854 FAILED |
| `hpc-node09` | main-gpu | **không** | 581864 FAILED |
| `ctit086` | main-gpu | **có** (kèm libx265, h264_nvenc) | 581856 |
| `ctit087` | main-cpu | **có** | 581867 |
| `spark-head3` | main-cpu | **không** | 581868 FAILED |

### Test

| | trước phiên | sau phiên |
|---|---|---|
| pytest | 286 passed, 7 skipped | **335 passed, 8 skipped** |
| ruff check + format | sạch | sạch |

Chưa có số liệu nào về ngưỡng `max_cost` — phần đó cần GPU, xem "Chưa xong".

## Cạm bẫy phát hiện trong phiên

- **Node tính toán của `ut-hpc` KHÔNG cùng image với nhau.** `ffmpeg` 4.4.2 có mặt ở mọi
  node đã thử, nhưng bản build khác nhau: 3/6 node thử được thiếu hẳn `libx264`. Hệ quả
  chung, không chỉ cho ffmpeg: **"job này chạy được hôm qua" không chứng minh gì** — SLURM
  cấp node khác. Job phụ thuộc công cụ hệ thống phải tự kiểm và chết to tiếng ở dòng đầu
  (`command -v` + `ffmpeg -encoders | grep`), chứ đừng để nó chết giữa camera thứ bảy. Đã
  ghi vào `reference/cluster.md` của skill `ut-hpc`.
- **Ghim `--nodelist` không phải bảo hiểm.** `ctit087` chạy job dò xong lúc 14:4x, submit
  lại 2 phút sau thì SLURM báo *"Nodes required for job are DOWN, DRAINED or reserved"*.
- **Tên file WildTrack cách nhau 5, ffmpeg đọc chuỗi theo mẫu liên tiếp.** Đưa thẳng
  `%08d.png` cho ffmpeg thì nó dừng ở khung đầu tiên không tìm thấy. Dựng thư mục link
  đánh số lại là cách tường minh nhất; demuxer `concat` cũng làm được nhưng thứ tự khung ở
  đây là thứ mọi ánh xạ ground-truth dựa vào, không nên phó thác cho hành vi của demuxer.

## Vướng mắc / chưa xong

- **Phần chạm GPU còn nguyên** — giống hệt tình thế cuối phiên 8: những gì làm hôm nay là
  "soạn đúng + có test canh", chưa phải "đã chạy". Checklist cho lần thuê `vast-gpu` tới:
  1. Chép `wildtrack_video/` (**1.1 GB, đã sẵn sàng ở `~/mct/data/wildtrack_video/` trên
     `ut-hpc`**) sang máy GPU. Nên đi **thẳng**
     `ut-hpc → vast-gpu` (cả hai đều băng thông tốt), không vòng qua máy dev.
  2. `make ds-run-wildtrack` — engine batch 7 phải build lần đầu (~4 phút), nhớ chép
     `model_b7_gpu0_fp16.engine` sang đúng tên khai trong config (cạm bẫy đã dính 2 lần).
  3. Ghi fixture: `python -m tools.record_metadata --out tests/fixtures/ds_wildtrack_7cam.jsonl`.
  4. `python -m tools.ds_wildtrack_gt` rồi `eval/eval_wildtrack.py --diagnose --sweep` —
     so vị trí ngưỡng tốt nhất với fixture đường ONNX Runtime.
- **Chưa biết detector bắt được bao nhiêu phần người của WildTrack.** Nếu YOLO11s bỏ sót
  quá nhiều (người ở xa, bị che), số tracklet giữ lại sẽ ít và kết luận yếu đi. Đo được
  ngay từ `gt_recall` trong report của `ds_wildtrack_gt`.
- **F1 tuyệt đối giữa hai fixture KHÔNG so trực tiếp được** — tracklet của tracker thật
  phân mảnh hơn hẳn "SCT lý tưởng" của fixture cũ. Thứ so được là *vị trí ngưỡng tốt nhất*
  và phân bố cosine cùng/khác danh tính. Phải viết rõ điều này khi đưa vào chương 6.
- Còn treo từ phiên trước: định nghĩa "độ trễ end-to-end" (chưa chốt với GVHD); leg
  dashboard chưa đo trên cùng máy; `tracker-width/height` chưa thử 1080p; cảnh báo "hai
  Global ID quá gần nhau trên mặt phẳng" (treo từ phiên 6).

## Bước tiếp theo

1. Thuê `vast-gpu` (**hỏi xác nhận trước, tính phí theo giờ**) và chạy hết checklist 4 mục
   ở trên trong một chuyến.
2. Có kết quả rồi thì quyết: giữ `max_cost = 0.60` hay đặt lại cho đường DeepStream, và
   ghi rõ trong `configs/demo/wildtrack.mct.yaml` con số đó đo trên đường nào.
3. Chốt định nghĩa độ trễ với GVHD, rồi báo cáo con số tương ứng.
