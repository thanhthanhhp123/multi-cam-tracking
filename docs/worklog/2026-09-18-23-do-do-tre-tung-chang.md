# 2026-09-18 (phiên 23) — Đo độ trễ theo TỪNG CHẶNG (t0..t4): đuôi p90 là độ trễ CHỐT DANH TÍNH

- **Mốc:** M4/M5 | **Máy:** máy dev (Windows, venv 3.10) + `vast-gpu` (Tesla T4) | **Thời lượng:** ~3h | **GPU $0.022**

## Mục tiêu phiên

- Phiên 9 đo được độ trễ end-to-end **trung vị 40 ms nhưng p90 2.1 s** và đưa ra giả thuyết
  "đuôi là độ trễ CHỐT danh tính, không phải nghẽn hàng đợi" — nhưng công cụ lúc đó
  (`tools/measure_latency.py`) chỉ cho ra MỘT con số, không bác bỏ được giả thuyết nào.
- Phiên này: gắn mốc tại từng điểm chuyển giao, tính trung vị/p90/p99 cho từng đoạn, rồi
  chạy thật trên GPU. **Chỉ đo, không sửa logic tracking/liên kết.**

## Đã làm

**Chuỗi mốc (`src/common/latency.py`, module mới, không GPU không numpy)**

| mốc | đóng ở đâu | đồng hồ |
|---|---|---|
| `t0` capture | probe, từ `NvDsFrameMeta.ntp_timestamp` | máy GPU |
| `t1` probe | probe, sau khi detect+track+ReID xong | máy GPU |
| `t1b` dequeue | luồng nền `QueuedFramePublisher` nhấc khỏi hàng đợi | máy GPU |
| `t2` xadd | **entry ID của Redis** (`<ms>-<seq>`) | máy Redis |
| `t3a` recv | `FrameConsumer` nhận được lô | máy engine |
| `t3w` window | cửa sổ gán bắt đầu chạy | máy engine |
| `t3` assoc | gán xong, TRƯỚC khi ghi DB | máy engine |
| `t3d` db | `Store.record_many` trả về | máy engine |
| `t4` out | đã đẩy lên `mct:global` | máy engine |

- `t0`, `t1`, `t1b` đi kèm message qua trường `FrameMessage.stamps` (thêm mới); `t2` lấy
  miễn phí từ entry ID; `t3*`/`t4` do engine tự đóng.
- Sửa: `ds_pipeline/probes.py` (t0/t1), `common/streams.py` (t1b ở publisher, t2+t3a ở
  consumer), `mct/tracklet.py` (tracklet mang mốc của khung MỚI NHẤT), `mct/__main__.py`
  (cờ `--latency-log` / `MCT_LATENCY_LOG`, mặc định TẮT), `tools/replay_metadata.py`
  (ghi đè mốc cũ của fixture — nếu không thì phát lại fixture tháng trước ra hàng triệu ms).
- **`src/tools/latency_report.py`**: bảng 4 đoạn thô + đoạn nhỏ thụt lề + **bảng quy trách
  nhiệm cho đuôi** (lấy riêng 10% chậm nhất, so trung vị từng đoạn với trung vị chung).
  `--by cam_id|final_flush|db_flushed|t0_source`, `--top N`, `--json`.
- **`configs/pipeline/streams_latency.yaml`** (4 luồng có ReID, `sync: true`) +
  **`docker/vast_latency.sh`** (`nvdec` | `engine` | `run`) — tái lập chuyến đo bằng 3 lệnh.
- Test: `tests/test_latency.py` (33 test) + 1 test ghim `streams_latency` chỉ khác
  `streams_reid` ở `sink.sync`. Toàn bộ: **554 passed, 5 skipped**, ruff sạch.

**Chuyến `vast-gpu`** (instance `51421345`, Tesla T4 15 GB, driver 615.71.09, $0.0708/h, Ý)
- NVDEC thử đầu tiên: PREROLLED → PLAYING → EOS. Đẩy repo + `models/` (40 MB, 15 s),
  `vast_bootstrap.sh` chạy sạch. Build engine TensorRT b4 + OSNet: ~6 phút trên T4.
- **Cả pipeline lẫn engine chạy trên CÙNG instance** — mọi đoạn nằm trong một đồng hồ.
- Chạy 2 lần, mỗi lần 5772 khung (4 luồng × 1443), ~1018 bản ghi độ trễ mỗi lần.

## Quyết định kỹ thuật

**1. `t2` lấy từ entry ID của Redis, KHÔNG thêm trường lên wire.** Entry ID `<ms>-<seq>`
chính là thời điểm server ghi entry: không tốn byte nào, và là đồng hồ của Redis chứ không
phải của bên ghi. Phương án bị loại: gắn `t2` vào message trước khi `XADD` — không làm được,
message đóng gói xong mới gửi, dấu đó là "trước khi gửi" chứ không phải "đã gửi xong".

**2. KHÔNG nâng `SCHEMA_VERSION` dù thêm trường `stamps`.** Phép thêm tương thích hai chiều:
bản đọc cũ bỏ qua khoá lạ, bản đọc mới gặp fixture cũ thì nhận dict rỗng, message không đo
thì không sinh khoá nên fixture cũ đọc-ghi lại ra đúng từng byte. Nâng version sẽ giết mọi
fixture WildTrack sinh ở phiên 22 (sinh lại phải thuê GPU). Đổi *ý nghĩa* một trường sẵn có
vẫn là breaking change và vẫn phải nâng version.

**3. Đo phải KHÔNG đổi kết quả gán — có test canh.** `stamps` chỉ để đo; `src/mct` không đọc
nó để ra quyết định, mốc cho thuật toán vẫn chỉ là `ts_ms`.
`test_bat_do_do_tre_khong_doi_ket_qua_gan` chạy cùng fixture hai lần và so từng cập nhật.

**4. Mẫu ÂM có dung sai RIÊNG cho từng đoạn, không dùng chung một ngưỡng.** Lần chạy đầu
báo `t2-t1b` có **556/1018 mẫu âm** và công cụ kết luận "lệch đồng hồ" — sai, vì Redis chạy
ngay trên cùng máy. Nguyên nhân thật: **entry ID của Redis chỉ mang mili-giây NGUYÊN**, tức
`t2` bị cắt phần thập phân trong khi `t1b` thì không, nên hiệu âm tới gần 1 ms dù thời gian
thật dương. Sửa: đoạn có một đầu là `t2` được nới dung sai lên 1 ms
(`Segment.neg_tolerance_ms`), các đoạn khác giữ 50 µs. Sau khi sửa: **0 mẫu âm ở mọi đoạn**.
Bài học chung: một cảnh báo sai ở hơn nửa số mẫu thì tệ hơn không có cảnh báo.

**5. Vòng gán CUỐI được đánh dấu `final_flush` và tách khỏi mọi con số vận hành.** Lúc hết
nguồn, `Engine.finish()` đóng mọi tracklet còn sống một lượt, nên `window_wait` của chúng
bằng đúng thời gian từ khung cuối tới lúc engine chịu dừng — lần 1 là **45.3 s** (đúng bằng
`--idle-limit 45`), kéo p99 từ 2.9 s lên 45.3 s. Đó là hiện vật của phép đo offline: hệ thống
chạy thật không bao giờ có khoảnh khắc "hết nguồn". Thay vì bỏ im lặng, bản ghi TỰ khai báo
cờ đó và `--by final_flush` tách được hai nhóm.

**6. Đo với `sync: true` là bắt buộc, và phải xác nhận pipeline theo kịp.** 126.6 FPS gộp
(4 luồng × 31.6 fps) so với 259.8 FPS lúc chạy hết tốc lực — tức nguồn phát đúng tốc độ thật
và pipeline dư sức, không có chỗ nào bị dồn. Nếu không xác nhận điều này thì mọi con số độ
trễ chỉ là độ trễ lúc quá tải.

## Số liệu đo được

**Cấu hình:** vast.ai `51421345`, Tesla T4 15 GB, driver 615.71.09, DeepStream 7.1.0
(`nvcr.io/nvidia/deepstream:7.1-triton-multiarch`), YOLO11s COCO FP16 input 640 batch 4 (lọc
lớp person tại nvinfer), NvDCF + ReID OSNet `osnet_x1_0_msdc_dg` (TensorRT FP16, 512-d),
streammux 1920×1080, tracker 960×544, `sink.sync: true`. Nguồn: 4 × `sample_1080p_h264.mp4`
(1443 khung/luồng, ~31.6 fps). Engine `src/mct` chạy **trên cùng instance** (EPYC 7V12),
`configs/mct.yaml` mặc định: `window_ms 1000`, `idle_timeout_ms 2000`, `store.batch_size 64`.
Throughput pipeline: **126.6 / 126.7 FPS gộp**, 5772 khung, **0 khung bỏ, hàng đợi sâu nhất 3**.

### 1. Từng chặng (lần 2, 1019 bản ghi; ms)

| đoạn | n | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| t1-t0 deepstream | 1019 | **67.4** | 68.3 | 69.4 | 73.2 |
| t2-t1 push_redis | 1019 | 0.1 | 0.6 | 1.1 | 2.6 |
| — t1b-t1 queue_wait | 1019 | 0.2 | 0.6 | 1.0 | 2.7 |
| — t2-t1b xadd | 1019 | −0.2 | 0.3 | 0.6 | 0.8 |
| t3-t2 engine | 1019 | 36.4 | **2235.9** | 15184.7 | 15266.7 |
| — t3a-t2 redis_pickup | 1019 | 0.8 | 1.3 | 2.0 | 2.0 |
| — **t3w-t3a window_wait** | 1019 | 32.8 | **2233.3** | 15182.3 | 15264.6 |
| — t3-t3w associate | 1019 | 3.0 | 4.8 | 5.1 | 5.1 |
| t4-t3 db+dashboard | 1019 | 1.5 | 1.8 | 2.9 | 2.9 |
| — t3d-t3 db_write | 1019 | 0.5 | 0.6 | 0.7 | 0.7 |
| — t4-t3d publish | 1019 | 0.9 | 1.2 | 2.9 | 2.9 |
| **t4-t0 END-TO-END** | 1019 | **105.7** | **2305.5** | 15200.7 | 15334.9 |

Bỏ 12 bản ghi `final_flush` (hiện vật, xem QĐ 5) thì con số **vận hành** là:
**p50 105.7 / p90 2172.6 / p99 2907.2 / max 3073.4 ms** (n = 1007).

### 2. Đuôi trễ sinh ra ở đâu — bảng quy trách nhiệm

102 bản ghi chậm nhất (end-to-end ≥ p90):

| đoạn | p50 chung | p50 ở đuôi | chênh |
|---|---|---|---|
| **t3w-t3a window_wait** | 32.8 | **2599.5** | **+2566.7** |
| t3a-t2 redis_pickup | 0.8 | 0.8 | +0.0 |
| t1b-t1 queue_wait | 0.2 | 0.2 | −0.0 |
| t1-t0 deepstream | 67.4 | 67.3 | −0.1 |
| t3-t3w associate | 3.0 | 2.4 | −0.6 |

Mọi đoạn khác **đứng yên** trong nhóm đuôi. Toàn bộ 2.5 s chênh lệch nằm ở một chỗ duy nhất.

### 3. `window_wait` có ĐÚNG BA CHẾ ĐỘ, không có gì ở giữa

| window_wait | lần 1 | lần 2 | cơ chế |
|---|---|---|---|
| ≤ 0.1 s | 73.7% | 74.8% | tracklet đang sống, gán ở cửa sổ kế tiếp |
| 0.1 – 1.0 s | 13.8% | 13.7% | chờ cửa sổ hiện tại đóng (`window_ms: 1000`) |
| 1.0 – 2.0 s | **0.0%** | **0.0%** | — |
| 2.0 – 3.0 s | 11.4% | 11.3% | **chờ `idle_timeout_ms` (2000) + cửa sổ** |
| > 3.0 s | 1.2% | 0.2% | vòng gán cuối (`final_flush`) |

**Kết luận: giả thuyết phiên 9 ĐÚNG, và giờ có cơ chế cụ thể kèm con số.** Đuôi p90 ≈ 2.2 s
là **độ trễ CHỐT DANH TÍNH**: 11.3% số cập nhật thuộc về tracklet chỉ được gán sau khi im
lặng đủ `idle_timeout_ms` rồi rơi vào cửa sổ kế — trần lý thuyết 2000 + 1000 = **3000 ms**,
đo được max 3073 ms và p99 2907 ms. Phiên 9 đo p99 **2977 ms** trên RTX 3090 với cùng bộ
tham số: hai máy khác nhau, cùng một trần, vì trần đó là **tham số cấu hình chứ không phải
phần cứng**.

Không có nghẽn ở đâu cả: hàng đợi publisher sâu nhất 3/2000 và 0 khung bỏ, `queue_wait`
p99 1.0 ms, `redis_pickup` p99 2.0 ms, `associate` p99 5.1 ms, ghi SQLite p99 0.7 ms, đẩy
`mct:global` p99 2.9 ms. **Tổng mọi khâu bị nghi ngờ < 10 ms.**

### 4. Thành phần lớn nhất của TRUNG VỊ lại là DeepStream

`t1-t0` = **67.4 ms (64% của trung vị 105.7 ms)**, và rất ổn định (p99 69.4, max 73.2). Đây
là thời gian từ lúc streammux nhận khung tới lúc probe đọc xong meta — gồm gom batch 4,
suy luận YOLO, NvDCF, trích 512-d ReID cho từng đối tượng (4.03 detection/khung). Muốn hạ
trung vị thì phải động vào đây, không phải vào `src/mct`.

### 5. Hạ tầng

- Test: **554 passed, 5 skipped**, ruff sạch (`~/.venvs/mct-test`, CPython 3.10.20).
- Chi phí: **$0.022** (credit 4.1494 → 4.1272), instance 11:00 → 11:19 UTC, đã huỷ,
  `vastai show instances` = 0.
- Hai lần chạy khớp nhau rất sát: p50 106.4 / 105.7 ms, p90 2315.9 / 2305.5 ms,
  `window_wait` p50 32.8 / 32.8 ms. Khác với so sánh cấu hình PIPELINE (phiên 22, nhiễu
  0.3–0.9 HOTA), phép đo độ trễ này lặp lại được.

## Vướng mắc / chưa xong

- **Chưa đo bản PHÂN TÁN** (pipeline ở `vast-gpu`, engine ở máy dev). Ở đó hai đoạn bắc cầu
  sẽ mang thêm lệch đồng hồ giữa hai máy — công cụ đã có cột `âm` để phát hiện, nhưng chưa
  có số thật. Chuyến này cố ý chạy chung một máy để có đường cơ sở sạch trước.
- `ts_ms` vẫn đóng ở probe (≈ `t1`), không phải lúc khung rời camera. Giờ đã biết `t1-t0`
  = 67 ms, tức ràng buộc thời gian xuyên camera đang làm việc với mốc trễ hơn thực tế 67 ms —
  **đồng đều giữa các camera** nên không lệch cặp, nhưng cần ghi vào chương 6.
- Nguồn là file 1080p lặp 4 lần, chưa có jitter RTSP thật. `t1-t0` với camera IP sẽ khác.
- `tests/fixtures/ds_4cam_reid_realtime.jsonl` trên máy dev **cụt ở dòng 1905** (không nằm
  trong git, lỗi có sẵn từ trước).

## Bước tiếp theo

1. **Chốt định nghĩa "độ trễ <1 s" của đề cương** (câu hỏi phiên 9 để ngỏ, giờ đã có số để
   chốt): độ trễ VỊ TRÍ = **p50 106 ms / p90 của nhóm tracklet đang sống < 1 s → ĐẠT**; độ
   trễ CHỐT DANH TÍNH = trần `idle_timeout_ms + window_ms` = 3 s → **KHÔNG ĐẠT nếu tính
   chung**. Hai câu hỏi khác nhau, phải báo cáo hai con số kèm định nghĩa, không gộp.
2. Nếu muốn hạ đuôi: `idle_timeout_ms` là núm vặn trực tiếp, nhưng phiên 12 đã chỉ ra nó
   phụ thuộc frame rate và hạ xuống sẽ làm VỠ tracklet (F1 giảm). Tức đây là **đánh đổi độ
   trễ ↔ độ chính xác**, phải đo cả hai chiều trước khi đổi — đừng sửa như sửa bug.
3. Hạ trung vị thì nhìn `t1-t0` (67 ms): thử `batched-push-timeout`, batch size, hoặc
   `reidExtractionInterval` — mỗi thay đổi cần ≥3 lần chạy (phiên 22 QĐ 1).
4. Đo lại bản phân tán để biết chi phí thật của việc đặt engine ở máy khác.
