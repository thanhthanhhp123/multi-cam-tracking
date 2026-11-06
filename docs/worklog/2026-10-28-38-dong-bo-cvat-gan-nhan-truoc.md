# 2026-10-28 — Đồng bộ 14 buổi, CVAT tự host, gán nhãn trước cho 42 task

- **Mốc:** M6 (chú thích) | **Máy:** máy dev (Windows) + `vast-gpu` (RTX 3090, ~1.2 h) | **Thời lượng:** ~2 buổi (2026-10-20 và 2026-10-28)

## Mục tiêu phiên

- Đưa 14 buổi quay ở hành lang (phiên 37) từ file thô tới chỗ chú thích được: đồng bộ, dựng CVAT.
- Người dùng yêu cầu: "label hết những gì m có thể đi, t chỉ việc check tay lại".

## Đã làm

**Dữ liệu.** Giải nén `data_zip.zip` (3.2 GB, kiểm CRC 237 file) vào `data/lab/raw/`. Chạy
`tools.sync_recordings` cho cả 14 buổi trên máy dev (khoảng 2.5 giờ) → `data/lab/<buổi>/camXX.mp4` +
`sync.json`, 6.0 GB. 10 buổi lệch ±1 khung ở cuối so với `expected_frames`; không ảnh hưởng gì.

**CVAT tự host.** Bản v2.78.0 ở `E:\Projects\cvat`, chạy ở `http://localhost:8081`, chỉ bind
`127.0.0.1`. Cổng 8080 đã bị container WordPress của dự án khác chiếm. 14 thư mục buổi đã đồng bộ được
gắn chỉ đọc vào `/home/django/share`; `data/lab/raw` cố ý không gắn.

- Có project "M6 hanh lang": label `person`, thuộc tính `person_id` kiểu text, không đổi theo khung.
- 42 task `<buổi>_<cam>`, Frame step 5.
- Lỗi đã biết: Traefik trả 404 ở `/` trong khi API vẫn 200. Sửa bằng `docker compose restart traefik`.

**Thử gán nhãn tay qua Chrome.** Một lượt đi của P01 trên `calib_cam01` có 11 khung khoá. Làm theo
quy ước (hộp tới mép dưới khi chân bị cắt, outside khi chân qua y ≈ 270) mất khoảng 10 phút. Như vậy
không thể làm tay hết 42 task.

**`tools/prelabel_cvat.py`** (+ `configs/lab/prelabel_bytetrack.yaml`, `docker/vast_prelabel.sh`, 9 test):

- Detector + ByteTrack chạy trên mọi khung, nhưng chỉ giữ khung thuộc task (`khung % 5 == 0`).
- Bỏ hộp có chân cao hơn `--min-foot-y` (cam01 = 250).
- Bỏ track ngắn hơn 5 khung task.
- Giữ ít khung khoá nhất có thể, sao cho nội suy tuyến tính của CVAT vẫn khớp hộp tracker với
  IoU ≥ 0.85. Thứ tự thêm khung khoá xếp theo GIoU, để chọn đúng góc đổi hướng khi các ứng viên đều có
  IoU = 0.
- Ghi "CVAT for video 1.1" với số khung VIDEO, nên `cvat_to_mot` đọc thẳng được (có test vòng tròn).
- `person_id` để trống, nên `cvat_to_mot` từ chối mọi track chưa có người kiểm.

**Chạy trên `vast-gpu`.** Máy 3090 ở Nhật mà người dùng duyệt bị thuê mất giữa lúc tìm và lúc tạo
(contract kẹt ở trạng thái `stopped`, đã huỷ). Người dùng duyệt lại máy RTX 3090 offer `37930674`
(Tây Ban Nha). Đẩy 6.0 GB video bằng 3 luồng `tar | ssh`, mất khoảng 6 phút. Chạy 3 tiến trình, mỗi
camera một tiến trình. Huỷ máy ngay khi xong, video không nằm lại trên đó.

**Nạp vào CVAT và gán danh tính.**

- Nạp track qua API (`PATCH …/annotations?action=create`, nguồn `auto`).
- Để điền `person_id`: tạo lưới ảnh cắt cho từng track (6 ảnh mẫu rải đều theo thời gian). Claude
  nhìn từng lưới, gán mã cho track nhận ra chắc chắn, xoá track bóng phản chiếu, và để trống track lẫn
  hai người.
- Nhận diện người theo trang phục:
  - P01–P07 là người trong nhóm, giữ cùng mã ở mọi buổi;
  - X.. là người khác, đánh số riêng trong từng buổi.
- Khi áp mã, máy không bao giờ để hai track cùng mã hiện trong cùng một khung của một camera:
  - chồng nhau ≤ 5 khung task ở đuôi track trước thì cắt đuôi track đó (thêm khung khoá nội suy +
    `outside`);
  - chồng dài hơn thì để trống track sau.
- Kết quả: `data/prelabel/REVIEW.md` liệt kê việc còn lại cho từng task, số khung theo CVAT.
- `data/prelabel/` (ngoài git) giữ XML, `person_ids.json`, log và các script phụ (`_scripts/`).

**Vòng 2: xử lý track trống bằng ảnh có khung.** Lưới ảnh cắt không đủ để gỡ track lẫn hai người. Vòng
này vẽ mọi hộp (nội suy) lên khung gốc, phóng to vùng có người, kèm nhãn hiện tại
(`_scripts/render_boxes.py`). Đo thêm độ phủ của từng track trống bởi track đã có mã
(`_scripts/overlap_report.py`). Mỗi quyết định ghi thành một thao tác trong `ops*.json`:
`set` (có thể kèm `keep` để giới hạn khoảng khung), `del`, hoặc `split` (cắt track tại khung tráo người, nửa sau
thành track mới). Các thao tác chạy qua `_scripts/cvat_ops.py` trong container: chạy thử trước, và từ chối
cả task nếu kết quả để hai track cùng mã hiện trong một khung.
- Gán thêm 33 track. Xoá 27 hộp trùng: hộp chỉ trúng đầu hoặc nửa người đã có hộp riêng, khoảng
  tối ở cửa, đỉnh đầu sát ống kính.
- Tách 2 track tráo người (s3_1 cam02 P02 → P05, s3_2 cam02 X03 → X01). Cắt khoảng cho 8 track.
- `calib`: nhóm sinh viên và nhóm đàn ông ở xa (25 track) nhận mã `Z<cam>-<số>`, mỗi track một mã. Mã
  không nối giữa các camera, nên không tạo cặp chuyển camera giả khi đo thời gian đi, mà `cvat_to_mot`
  vẫn chạy được.
- Kiểm chéo mã giữa các camera ở 12 buổi chấm điểm (`_scripts/id_sheet.py`: mỗi mã một hàng, 2 ảnh cắt
  mỗi camera). Không thấy ai mang hai mã. Ba chỗ trông lạ đều được xác minh lại trên khung gốc là ảnh cắt
  bị che.
- Lỗi tìm thấy và sửa trong lúc chạy thử: lúc giới hạn khoảng, `restrict` bỏ mất khung `outside` gốc ở
  cuối, làm track kéo dài tới hết video. Bộ kiểm chồng khung bắt được lỗi này ngay ở lần chạy thử.

**Công cụ duyệt GT thay CVAT (2026-10-29).** Người dùng thấy CVAT tốn RAM quá. CVAT có khoảng 18
container, trong đó 8 worker Python và bộ ClickHouse/Grafana; thêm vào đó WSL2 giữ bộ đệm file video
không trả. Vì vậy tôi viết công cụ riêng:
- `src/tools/gt_edit.py`: đọc và ghi "CVAT for video 1.1"; các thao tác đặt mã, xoá, tách, ẩn,
  occluded, sửa hộp có nội suy lại giữa các khung khoá, track mới, gộp; `validate`.
- `src/tools/gt_review.py` và `src/tools/gt_review_ui/index.html`: FastAPI chỉ nghe ở 127.0.0.1,
  trang HTML với JavaScript thuần. Ghi `data/cvat/<buổi>/` cùng `edits.jsonl` sau mỗi thao tác, có
  hoàn tác.
- Ảnh trích một lần: JPEG 960×540 ở các khung task, khoảng 70 khung video/giây trên CPU.
- 10 test, trong đó có phép kiểm file ghi ra vẫn đi qua `cvat_to_mot`.
- Thử trên dữ liệu thật trong Chrome: kéo cạnh hộp (khung khoá và nội suy hai phía đều đúng), tách
  track, hoàn tác, thêm người mới, tab Lỗi.
- Một lỗi tìm thấy lúc thử và đã sửa: bảng track lệch trạng thái sau khi hoàn tác.

**Kiểm trọn chuỗi.**
- Xuất `s2_1` (cả 3 camera đã gán đủ) theo "CVAT for video 1.1".
- Chạy `tools.cvat_to_mot` với đúng tham số của `make lab-gt`.
- Kết quả: 3 camera, 14 track, 3 danh tính, số khung tuyệt đối đúng lưới bước 5, 1 323 message GT.
- Sau vòng 2, xuất nháp cả 42 task (`data/prelabel/export_draft/`): **12/14 buổi qua được `cvat_to_mot`**.
  `s2_2c` và `s3_2` bị chặn đúng như thiết kế, vì còn 4 track trống.
- `estimate_transit` trên GT nháp (chỉ để xem trước, chưa điền topology):

  | Buổi | cam01→cam03 | cam03→cam01 |
  |---|---|---|
  | calib (đi chậm) | 4.8–13.4 s, n = 5 | 4.6–9.8 s, n = 5 |
  | calib_2 (thường / nhanh) | 3.8–6.2 s, n = 6 | 4.0–5.8 s, n = 4 |

  Cặp chồng lấn cam01↔cam02 đều có Δt < 0, đúng như mong đợi.

Sửa `docs/m6/README.md` §5: thêm quy trình gán nhãn trước và thứ tự kiểm tay.

## Quyết định kỹ thuật

- **CVAT tự host, không dùng app.cvat.ai.** `docs/m6/phieu-dong-thuan.md` mục 3 hứa video chỉ nằm
  trên máy sinh viên và máy chủ thuê theo phiên (xoá khi hết phiên).
- **Gán nhãn trước bằng một detector KHÁC detector đang được chấm** (YOLO11x 1280, không phải YOLO11s
  640). Dùng chính detector đang chấm thì người nó bỏ sót cũng dễ sót luôn trong GT, và điểm của nó bị
  thổi lên. Đây vẫn là thiên lệch (cùng họ YOLO, cùng COCO), nên báo cáo phải ghi: hộp do YOLO11x +
  ByteTrack vẽ trước, danh tính do Claude đề xuất, mọi thứ đã qua người kiểm.
- **Danh tính gán bằng mắt, không bằng gom cụm ReID.** Gom cụm bằng OSNet nhanh hơn, nhưng sẽ đưa đúng
  loại đặc trưng mà hệ thống dùng vào GT. Gom cụm cũng không phân biệt được s4_1, nơi hai người mặc cùng
  một áo khoác; ở đó phải nhìn quần. Track nào không chắc thì để trống, vì `cvat_to_mot` báo lỗi khi gặp
  track trống. Không dùng mã tạm kiểu "P01?", vì mã như vậy sẽ thành một danh tính mới mà không có gì
  báo.
- **Không chuyển sang BoT-SORT + ReID để bớt tráo người.** Đọc mã ultralytics 8.4.7 thì chi phí ghép là
  `min(khoảng cách IoU, khoảng cách ngoại hình)`. Ngoại hình chỉ làm việc ghép dễ hơn, không chặn được
  một lần ghép sai mà IoU ủng hộ. Kết luận này đến từ đọc mã, chưa đo.
- **Phương án bị loại:**
  - gán tay qua Chrome: khoảng 10 phút cho một lượt đi trên một camera;
  - chạy trên CPU máy dev: YOLO11n 640 chỉ được 0.9 khung/s, và máy đang yếu;
  - app.cvat.ai: trái phiếu đồng thuận.

## Số liệu đo được

- **Gán nhãn trước:**
  - Cấu hình: YOLO11x COCO, imgsz 1280, FP16, ultralytics 8.4.7, ByteTrack (`track_buffer` 75),
    RTX 3090 (driver 590.48.01, image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, 21 vCPU),
    3 tiến trình song song.
  - Tốc độ: mỗi tiến trình 15.4–16.6 khung/s; tổng 145 670 khung trong 51.5 phút ≈ 47 khung/s.
  - GPU dưới tải: tối đa 83 °C, SM ≥ 1 770 MHz; cờ hạ xung 0x4 là chạm giới hạn công suất, không phải
    nhiệt.
- **Chi phí:** máy thuê chạy 1.24 h ở $0.41/h. Credit giảm $5.73 → $5.16, tức khoảng $0.57 cả phần
  contract kẹt.
- **Track:**
  - 877 track thô → 472 track sau lọc (≥ 5 khung task, chân cam01 ≥ y 250);
  - 33 551 hộp ở khung task, chỉ 9 024 khung khoá (27%);
  - xoá 42 track bóng phản chiếu (ô kính cửa trái trên cam02, cửa kính phải trên cam01), còn
    **430 track**.
- **Danh tính:**
  - vòng 1 (lưới ảnh cắt): 333/430 track (77%) có `person_id`;
  - vòng 2 (ảnh có khung + kiểm chéo): **399/403 track có mã** (99%), trong đó 25 là mã `Z` ở `calib`.
    Còn 4 track trống, đều là mảnh nhỏ hoặc lẫn người ở s2_2c và s3_2.
- **CPU máy dev** (i5-10400F, YOLO11n 640): 0.9 khung/s.
- **Đánh giá bộ dữ liệu trên GT nháp** (`data/prelabel/_scripts/dataset_stats.py` → `data/prelabel/dataset_stats.json`):
  - Cách đếm: sau `ground_roi` của cam01, định nghĩa bàn giao giống `eval_handover`, track trống tính
    là người riêng.
  - Thời lượng: 14 buổi, 32.4 phút; 12 buổi chấm điểm dài 21.3 phút, có 19 977 hộp GT ở 5 fps.
  - **Bàn giao trong 12 buổi chấm:**

    | Loại | n | Có người khác đến cùng camera trong ±3 s | Khoảng tin cậy 95% Wilson ở p = 0.8 |
    |---|---|---|---|
    | Chồng lấn | 102 | 58 | ±8 điểm |
    | Không chồng lấn | 48 (44 ở cam01↔cam03) | 7 | ±11 điểm |
    | Quay lại cùng camera | 39 | 15 | ±12 điểm |

  - Mật độ: tối đa 7 người/khung (s3); s2 tối đa 4.
  - Hộp cao dưới 150 px chỉ chiếm 1–5%.
  - ROI cam01 bỏ 23% hộp cam01.
  - Kiểm độ phủ GT trên 32 khung ngẫu nhiên: khoảng 31 người trong vùng chú thích, không ai thiếu hộp.
  - Người chấm vẫn là 7 người tham gia lặp lại, nên mẫu không độc lập.
  - s4 (áo giống nhau) chỉ có 6 lần không chồng lấn.

## Vướng mắc / chưa xong

- **Toàn bộ GT mới là bản nháp.** Chưa task nào được người kiểm. Hộp có thể lệch (máy không bao giờ bật
  `occluded`), danh tính do Claude đề xuất có thể sai, và người bị sót chưa được thêm.
- Hai track ở s4_1 cam02 chỉ có hộp quanh đầu của P02 khi bị P06 che. Cần kéo hộp xuống tới chân và
  bật occluded; máy không làm được việc này.
- Các mã `Z` ở `calib` cố ý không nối danh tính giữa các camera. Chúng không ảnh hưởng đo thời gian đi,
  nhưng nếu sau này chấm `calib` thì phải gộp lại.
- Chưa kiểm vai P1/P2 trong `data/lab/raw/ghi-chu-buoi-quay.md` có khớp với P01–P07 hay không. Mã
  P01–P07 đặt theo trang phục, không theo vai trong kịch bản.
- Chưa chắc: P01 ở s3_2 (khoác áo navy trơn); X02 ở calib (ba lần đi qua có phải một người không). Cả
  hai đã ghi trong `REVIEW.md`.
- Hai tiến trình nền (vòng theo dõi GPU, server tĩnh cục bộ) bị Claude Code dừng vì máy dev thiếu RAM
  (còn khoảng 2.1/15.9 GB khi CVAT chạy). Từ đó bước nạp/áp chuyển sang chạy thẳng trong container
  (`docker exec cvat_server python manage.py shell`, script ở `data/prelabel/_scripts/`).
- Các việc treo từ phiên 37 vẫn còn: ghi chú buổi quay, `estimate_transit` với `ground_roi`, gộp
  `s2_Nb/c`, có quay bổ sung không, trả lại `powercfg` trên laptop.

## Bước tiếp theo

1. Người dùng kiểm trong CVAT theo `data/prelabel/REVIEW.md` (thứ tự calib → s2 → s1 → s4 → s3): điền
   4 track trống, xem các dòng "vừa chắc", sửa hộp, bật occluded, thêm người bị sót.
2. Xuất "CVAT for video 1.1" về `data/cvat/<buổi>/camXX.xml`. Xuất hàng loạt được bằng
   `data/prelabel/_scripts/cvat_export.py` trong container, hoặc làm tay trên giao diện.
3. `make lab-gt LAB_SESSION=calib` → `make lab-transit LAB_SESSION=calib` → điền
   `configs/lab/topology.yaml`.
