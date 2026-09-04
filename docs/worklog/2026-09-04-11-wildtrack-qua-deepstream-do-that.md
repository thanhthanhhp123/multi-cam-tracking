# 2026-09-04 (phiên 11) — WildTrack chạy thật qua DeepStream: ngưỡng KHÔNG chuyển được, và một sự thật khó chịu hơn

- **Mốc:** M3 (đóng câu hỏi treo lớn nhất) | **Máy:** `vast-gpu` (3 máy, Tesla T4 là máy chạy được) + `ut-hpc` (chấm điểm) | **Thời lượng:** ~2.5h, **GPU tốn $0.189**

## Mục tiêu phiên

Chạy checklist 4 mục mà phiên 10 đã chuẩn bị sẵn: đưa 7 video WildTrack qua pipeline
DeepStream thật, ghi fixture, gán ground-truth, rồi trả lời câu hỏi treo từ phiên 8:
**ngưỡng `max_cost` chỉnh trên embedding của đường ONNX Runtime có chuyển sang embedding
của đường DeepStream không?**

## Đã làm

- Thuê **3 instance vast.ai**: hai máy đầu (RTX 3090 driver 590.48.01; RTX 3060 driver
  595.84) **hỏng NVDEC**, huỷ; **Tesla T4** (driver 560.35.03) chạy được.
- Chuyển 1.13 GB video **thẳng `ut-hpc` → `vast-gpu`** (không vòng qua máy dev) bằng khoá
  ed25519 tạm sinh trên cụm, xoá sau khi xong. Đo được **~56 MB/s**, 3.5 phút.
- Chạy pipeline 7 luồng có ReID hai lần: tracker 960×544 và tracker 1920×1088.
- Ghi 2 fixture (100 MB mỗi cái), kéo về `ut-hpc`, gán ground-truth bằng
  `tools/ds_wildtrack_gt.py`, chấm bằng `eval/eval_wildtrack.py` — **không sửa một dòng
  code nào**, đúng như thiết kế phiên 10.
- Cập nhật `docker/vast_bootstrap.sh` không cần đổi; toàn bộ hạ tầng phiên 10 chạy đúng
  ngay lần đầu.

## Quyết định kỹ thuật

**1. Thử NVDEC trong 30 giây TRƯỚC KHI dựng bất cứ thứ gì trên máy mới thuê.**
Đây là bài học đắt nhất của phiên. Máy đầu tiên: tôi dựng môi trường, chép 1.1 GB video,
build engine TensorRT 5 phút — rồi mới phát hiện `nvv4l2decoder` treo ở PREROLLING và
không giải mã được gì. Mất ~30 phút. Phép thử đúng chỉ tốn 30 giây và không cần repo:

```bash
gst-launch-1.0 -e nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264 \
    ! fakesink sync=false 2>&1 | grep -E "PREROLLED|PLAYING|EOS"
```

Máy tốt in `PREROLLED → PLAYING → EOS`; máy hỏng dừng ở `PREROLLING` mãi mãi. Chốt: đây là
việc **đầu tiên** làm trên mọi instance mới, trước cả `vast_bootstrap.sh`.

**2. Video mẫu của DeepStream là biến đối chứng, không phải dữ liệu.** Khi máy đầu không ra
frame nào, nghi vấn tự nhiên là "video WildTrack tôi tự đóng bị sai". Cách phân biệt: cho
**video mẫu đi kèm DeepStream** qua đúng đường ống đó. Nó cũng treo → vấn đề ở máy, không ở
dữ liệu. Không có bước này thì rất dễ đi sửa bộ đóng video suốt buổi trong khi lỗi nằm ở
phần cứng thuê.

**3. Giữ `tracker-width/height = 960×544`.** Câu hỏi treo từ phiên 8 ("crop đưa vào ReID lấy
từ frame nửa độ phân giải, người ở xa có thể quá nhỏ") giờ đã có số: nâng lên 1920×1088 làm
**VRAM tăng 2.07 lần** (1003 → 2081 MiB) mà chất lượng embedding **không đổi**, thậm chí
nhích xuống (trần ngoại hình 0.168 so với 0.181). Lý do có thể là OSNet dù sao cũng resize
về 256×128, nên chi tiết thêm bị vứt đi ngay. Đóng câu hỏi này lại.

**4. Không đổi `max_cost` trong `configs/demo/wildtrack.mct.yaml`.** Ngưỡng ngoại hình tối ưu
CÓ dịch (xem Số liệu), nhưng ở **chế độ sản xuất** (có hình học, `ground_gap_policy=reject`)
thì trong 6 giá trị đã quét, `0.60` vẫn là giá trị tốt nhất trên cả hai đường. Đổi ngưỡng
không phải là việc cần làm — vấn đề thật nằm chỗ khác (xem quyết định 5).

**5. Chốt cách đọc mọi con số cũ của `src/mct`: chúng là CẬN TRÊN, không phải hiệu năng hệ
thống.** Fixture WildTrack cũ (`wildtrack_to_fixture.py`) dùng bbox ground-truth và
`local_track_id` sinh từ `personID` — tức **SCT lý tưởng**: tracklet dài, không id-switch,
không bỏ sót. Trên chính dataset đó, cùng cấu hình, cùng đoạn code chấm điểm:

| fixture | F1 (max_cost 0.60, geo reject) |
|---|---|
| ONNX Runtime, SCT lý tưởng | **0.752** |
| DeepStream, tracker thật | **0.170** |

Chênh 4.4 lần, và **không phải do embedding**. Nguyên nhân đã truy được: xem "Vì sao" dưới.
Từ nay mọi con số của `src/mct` đo trên fixture cũ phải ghi kèm chữ "cận trên (SCT lý tưởng)"
khi đưa vào chương 6, nếu không là báo cáo sai sự thật.

## Số liệu đo được

**Cấu hình chung:** vast.ai instance `49862670`, **Tesla T4 15 GB, driver 560.35.03**,
DeepStream 7.1.0, CUDA 12.6, TensorRT 10.3, pyds 1.2.0, Python 3.10.12. YOLO11s weight gốc
COCO, FP16, input 640, lọc lớp person tại nvinfer. `nvstreammux` 1920×1080, batch 7.
ReID = OSNet `osnet_x1_0_msdc_dg`, ONNX → TensorRT FP16, batch 100, `reidExtractionInterval: 0`.
Nguồn: 7 video WildTrack đóng ở phiên 10 (400 khung/camera, 2 fps, CRF 18, H.264).

### Chạy đúng tốc độ thật (`sink.sync: true`) — để ghi fixture

| | |
|---|---|
| khung | **2800** (400 × 7 camera, khớp chính xác) |
| thời gian | 196.62 s → **đúng 2.0 FPS mỗi luồng** |
| detection | **34 597** (11.1–14.0 mỗi khung tuỳ camera) |
| embed_dim | 512 |
| `max_frame_id` | **399** trên tổng 400 khung chú thích → **ánh xạ khung khớp tuyệt đối** |

### Throughput (`sink.sync: false`, 7 luồng có ReID, T4)

| | |
|---|---|
| FPS gộp | **94.8** (2800 khung / 29.53 s) |
| FPS mỗi luồng | **13.5** |
| VRAM đỉnh | **1003 MiB** (tracker 960×544) |

**Chưa đạt mục tiêu đề cương ≥15 FPS/luồng** ở 7 luồng trên T4. Đối chiếu phiên 9: RTX 3090,
4 luồng, 175 FPS/luồng. T4 yếu hơn nhiều và số luồng gần gấp đôi — con số này nói về T4,
không nói rằng hệ thống không đạt. Muốn kết luận thì phải đo 7 luồng trên đúng GPU triển khai.

### Tracker 960×544 vs 1920×1088 (cùng máy, cùng nguồn, cùng model)

| | 960×544 | 1920×1088 |
|---|---|---|
| VRAM đỉnh | **1003 MiB** | **2081 MiB** (+107%) |
| detection | 34 597 | 34 519 (−0.2%) |
| GT recall của detector | 44.9% | 45.1% |
| track giữ lại / tổng | 345 / 938 | 319 / 896 |
| thuần khiết trung vị | 0.833 | 0.818 |
| **trần ngoại hình (F1)** | **0.181** | **0.168** |

Kết luận: gấp đôi VRAM, không được gì.

### Gán ground-truth (fixture 960×544, IoU ≥ 0.5, Hungarian, thuần khiết ≥ 0.7)

| | |
|---|---|
| detection khớp được một hộp GT | 19 139 / 34 597 = **55.3%** |
| **GT recall** (phần người WildTrack chú thích mà detector bắt được) | **44.9%** |
| local track | 938 → giữ **345** (36.8%) |
| track dính ≥2 danh tính | **498 / 938 = 53%** |
| loại vì lẫn danh tính / ít khung / không khớp | 288 / 137 / 168 |
| danh tính phủ được | 146 (WildTrack có 313) |

### Câu trả lời cho câu hỏi của phiên 8 — phân bố cosine giữa cặp tracklet KHÁC camera

| đường trích embedding | cùng người (trung vị) | khác người (trung vị) | trần ngoại hình | ngưỡng tối ưu |
|---|---|---|---|---|
| ONNX Runtime (bbox GT, SCT lý tưởng) | 0.736 | 0.690 | F1 **0.053** | cosine ≥ 0.906 → `max_cost` **0.094** |
| **DeepStream** (960×544) | 0.645 | 0.588 | F1 **0.181** | cosine ≥ 0.773 → `max_cost` **0.227** |
| DeepStream (1920×1088) | 0.630 | 0.588 | F1 0.168 | cosine ≥ 0.768 → `max_cost` 0.232 |

**Ngưỡng KHÔNG chuyển được: điểm tối ưu dịch 0.094 → 0.227, gấp 2.4 lần.** Hai đường cho
embedding có thang cosine khác hẳn nhau, đúng như lo ngại ở phiên 8 quyết định 4.

Nhưng có hai điều bất ngờ, ngược với giả định:

1. **Embedding của DeepStream tách người TỐT HƠN, không tệ hơn** (trần 0.181 vs 0.053).
   Giả định ngầm suốt hai phiên là đường DeepStream sẽ kém hơn vì tiền xử lý lệch và FP16.
   Sai. (Lưu ý đọc số: tập tracklet của DeepStream đã bị lọc theo thuần khiết ≥ 0.7 nên
   sạch hơn, phần nào thổi con số lên — nhưng không thể lật ngược chiều so sánh.)
2. **Cả hai trần đều rất thấp.** Ngoại hình một mình **không giải được** WildTrack xuyên
   camera, dù đường nào. Khớp với kết luận phiên 5: hình học mạnh hơn ngoại hình ~20 lần.

### Quét `max_cost` trên fixture DeepStream (7 camera, có homography)

| `max_cost` | không hình học (P/R/F1) | `geo=reject` (P/R/F1) |
|---|---|---|
| 0.10 | 1.000 / 0.015 / 0.030 | 1.000 / 0.000 / 0.000 |
| 0.20 | 0.164 / 0.149 / **0.156** | 1.000 / 0.006 / 0.012 |
| 0.23 | 0.095 / 0.197 / 0.128 | 0.857 / 0.012 / 0.023 |
| 0.30 | 0.042 / 0.197 / 0.069 | 0.379 / 0.021 / 0.040 |
| 0.40 | 0.029 / 0.247 / 0.052 | 0.493 / 0.066 / 0.116 |
| 0.60 | 0.023 / 0.185 / 0.041 | 0.375 / 0.110 / **0.170** |

Đối chứng, cùng lệnh, fixture ONNX Runtime: `max_cost 0.60`, `geo=reject` → **P 0.776 /
R 0.729 / F1 0.752**.

### Vì sao F1 sụp từ 0.752 xuống 0.170 — và nó KHÔNG phải lỗi của embedding

Thành phần hình học so vị trí mặt đất **tại cùng mốc thời gian** (phiên 5 đã chốt: so
"điểm cuối ↔ điểm đầu" cho kết quả tệ hơn). `ground_gap_policy=reject` loại thẳng cặp
tracklet không có mốc chung. Trên fixture SCT lý tưởng, tracklet dài và luôn trùng thời
gian → hình học phát huy hết sức. Trên fixture tracker thật, tracklet **vỡ vụn**: 938 local
track cho 146 danh tính, 53% dính nhiều danh tính. Số cặp tracklet khác camera có mốc thời
gian chung tụt từ hàng nghìn xuống **27 cặp cùng người**. Hình học không có gì để so, chính
sách `reject` loại sạch, recall về 0.

Đây là điểm nghẽn thật của hệ thống, và nó nằm ở **tracking đơn camera + detector**, không ở
module liên kết:

- detector bỏ sót **55%** người WildTrack chú thích;
- tracker cắt tracklet thành mảnh và **đổi danh tính giữa chừng ở 53% số track**.

## Cạm bẫy phát hiện trong phiên

- **Instance vast.ai có thể hỏng NVDEC dù `nvidia-smi`, CUDA, TensorRT đều bình thường.**
  2/3 máy thuê hôm nay bị. Triệu chứng: `nvv4l2decoder` dừng ở `PREROLLING` vĩnh viễn,
  GPU 0%, kèm `gst_debug_log_valist: assertion 'category != NULL' failed`. `libnvcuvid.so`
  vẫn có mặt, `/dev/nvidia*` vẫn đủ — nên **không thể phát hiện bằng cách kiểm thư viện**,
  phải chạy thử thật. Không tương quan với driver (595.84 hỏng ở máy này nhưng chạy tốt ở
  phiên 9). Card datacenter (T4) chạy được; hai card tiêu dùng thì không.
- **`cd A && (vòng lặp nền) &` đặt CẢ `cd` vào subshell nền.** Tiến trình foreground sau đó
  chạy ở thư mục mặc định, và nvtracker — vốn phân giải đường dẫn theo **thư mục làm việc**
  (cạm bẫy đã ghi ở phiên 9) — báo `!![ERROR] ONNX file does not exist` rồi pipeline chết
  câm, tiến trình vẫn sống. Mất 7 phút tiền thuê. Phải `cd` lại tường minh trong nhánh
  foreground.
- **Đừng kill pipeline trong lúc build engine.** Lần chạy đầu tôi tưởng nó treo ở phút thứ 5
  và kill; đọc log sau mới thấy engine build xong đúng ở 4:57 và decoder vừa khởi động.
  Build engine TensorRT batch 7 mất ~5 phút và GPU util có lúc về 0% — **im lặng không có
  nghĩa là treo**. Cách phân biệt: `ls` xem file `.engine` có đang được ghi không.
- **stdout của lệnh chạy qua `ssh ... | tail -N` chỉ hiện khi lệnh kết thúc.** Chạy lâu thì
  phải `| tee /workspace/x.log` rồi `tail` file đó từ một phiên ssh khác, nếu không là mù
  hoàn toàn trong lúc chờ.
- **`~/.ssh/authorized_keys` trên instance không có newline cuối file.** `echo key >> file`
  nối khoá mới vào đuôi khoá cũ, làm hỏng cả hai. Dùng script đọc–chuẩn hoá–ghi lại.
- **`eval/eval_wildtrack.py` chết vì `UnicodeEncodeError` trên `ut-hpc`** khi stdout không
  phải TTY (locale latin-1). Chạy với `PYTHONIOENCODING=utf-8`.

## Vướng mắc / chưa xong

- **Điểm nghẽn đã đổi chỗ: giờ là detector recall (44.9%) và độ vỡ tracklet (53% track lẫn
  danh tính), không phải module liên kết.** Đây là việc lớn tiếp theo của đồ án. Hướng đáng
  thử, theo thứ tự rẻ→đắt: (a) hạ `pre-cluster-threshold` của lớp person từ 0.25 và đo lại
  recall; (b) nới `maxShadowTrackingAge`/tham số tái liên kết của NvDCF để tracklet sống
  lâu hơn; (c) xét lại `ground_gap_policy` — chính sách `reject` quá khắc nghiệt với tracklet
  ngắn, có thể cần cửa sổ dung sai thời gian rộng hơn `ground_time_tol_ms` hiện tại.
- **WildTrack ở 2 fps là kịch bản khắc nghiệt bất thường cho tracker.** Người đi bộ dịch
  ~0.5–1 m giữa hai khung liên tiếp; NvDCF vốn thiết kế cho 25–30 fps. Một phần độ vỡ
  tracklet là do đó chứ không phải do tracker kém. Camera thật của đồ án chạy 25 fps nên
  con số ở đây là **cận dưới**; phải nói rõ điều này trong báo cáo, và đừng vội chỉnh tham
  số tracker theo dataset 2 fps.
- Chưa đo 7 luồng trên GPU mạnh (T4 cho 13.5 FPS/luồng, dưới mục tiêu 15). Cần một lần đo
  trên đúng GPU sẽ triển khai trước khi kết luận về mục tiêu hiệu năng.
- Còn treo từ trước: định nghĩa "độ trễ end-to-end" (chưa chốt với GVHD); leg dashboard chưa
  đo trên cùng máy; cảnh báo "hai Global ID quá gần nhau trên mặt phẳng" (treo từ phiên 6).

Hai fixture mới nằm ở `~/mct/data/fixtures/` trên `ut-hpc` (100 MB mỗi cái, kèm `.gt.json`
và `.gt-report.json`), **không kéo về máy dev, không vào git**.

## Bước tiếp theo

1. Đo lại `gt_recall` của detector khi hạ `pre-cluster-threshold` — chạy được **ngay trên
   fixture đã có**? Không: cần chạy lại pipeline. Nhưng phần phân tích độ vỡ tracklet thì
   làm được offline trên hai fixture hiện có, làm trước.
2. Viết một công cụ nhỏ thống kê độ vỡ tracklet (số tracklet/danh tính, phân bố độ dài,
   số mốc thời gian chung giữa cặp tracklet khác camera) để định lượng điểm nghẽn trước khi
   động vào tham số.
3. Chốt định nghĩa độ trễ với GVHD.
