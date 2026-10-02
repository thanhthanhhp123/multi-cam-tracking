# M6 — Dữ liệu tự thu: từ ngày quay tới bảng điểm

Tài liệu này là quy trình trọn vẹn cho mốc M6 (đề cương mục 4.2, 4.3.2, 6.2). Mọi công cụ
đã có sẵn và có test. Phần **bạn phải tự làm** gồm quay, đo, chú thích, và xác nhận khi
thuê GPU. Phần còn lại chạy bằng lệnh.

| # | Việc | Ai | Ở đâu | Lệnh / tài liệu |
|---|---|---|---|---|
| 1 | Chuẩn bị thiết bị, địa điểm, phiếu đồng thuận | bạn | — | mục 1, [phiếu](phieu-dong-thuan.md) |
| 2 | Bố trí camera, dán dấu sàn, đo toạ độ | bạn | hiện trường | mục 2, [checklist](checklist-ngay-quay.md) |
| 3 | Quay 5 đoạn (1 hiệu chỉnh + 4 kịch bản) | bạn | hiện trường | mục 3 |
| 4 | Đồng bộ + chuyển fps cố định | lệnh | máy có ffmpeg / `vast-gpu` | `tools.sync_recordings` |
| 5 | Đọc pixel của dấu sàn → hiệu chỉnh homography | bạn + lệnh | máy dev | `make lab-homography` |
| 6 | Chú thích CVAT | bạn | CVAT | mục 5 |
| 7 | CVAT → ground-truth, đo transit, điền topology | lệnh + bạn soát | máy dev | `make lab-gt`, `make lab-transit` |
| 8 | Kiểm cấu hình trước khi thuê GPU | lệnh | máy dev | `make lab-check` |
| 9 | Pipeline 3 lần/đoạn + độ trễ đồng hồ thật + FPS | lệnh (**xác nhận thuê máy**) | `vast-gpu` | `docker/vast_lab.sh` |
| 10 | Chấm + quét tham số | lệnh | máy dev | `make lab-eval` |

Thứ tự 6 và 7 có thể đảo cho đoạn hiệu chỉnh: chú thích đoạn hiệu chỉnh trước để có transit
time sớm (mục 3).

**Máy dev Windows không có `make`.** Mỗi target `lab-*` chỉ là một lệnh Python. Chạy bằng venv
test (CLAUDE.md §2), với `PY=~/.venvs/mct-test/Scripts/python.exe` và `S=s1`:

```bash
PYTHONPATH=src $PY -m tools.calibrate_homography --points configs/lab/ground_points.yaml --out configs/lab/homography   # lab-homography
PYTHONPATH=src $PY -m tools.cvat_to_mot --annotation cam01=data/cvat/$S/cam01.xml --annotation cam02=data/cvat/$S/cam02.xml \
    --annotation cam03=data/cvat/$S/cam03.xml --annotation cam04=data/cvat/$S/cam04.xml \
    --out-dir eval/gt/lab_$S --fixture-out data/fixtures/lab_${S}_gt.jsonl --fps 25                               # lab-gt
PYTHONPATH=src $PY -m tools.estimate_transit --gt-fixture data/fixtures/lab_calib_gt.jsonl \
    --topology configs/lab/topology.yaml --yaml-out data/lab/transitions_calib.suggest.yaml                       # lab-transit
PYTHONPATH=src $PY -m tools.check_lab_setup --session $S --gt-fixture data/fixtures/lab_${S}_gt.jsonl             # lab-check
PYTHONPATH=src $PY -m eval.run_lab_eval --config configs/lab/lab.mct.yaml --topology configs/lab/topology.yaml \
    --homography-dir configs/lab/homography --gt-fixture data/fixtures/lab_${S}_gt.jsonl \
    --run r1 data/fixtures/lab_${S}_r1.jsonl --run r2 data/fixtures/lab_${S}_r2.jsonl --run r3 data/fixtures/lab_${S}_r3.jsonl \
    --engine-python $PY --eval-python ~/.venvs/mct-eval/Scripts/python.exe --trackeval-path ~/TrackEval \
    --work-dir data/lab/eval/$S --fps 25                                                                           # lab-eval
```

---

## 1. Chuẩn bị

**Thiết bị** (phương án dự phòng khi chưa mua camera IP, đã bàn ở phiên 29):
- 4 điện thoại (hoặc 3 nếu bỏ cam04), mỗi máy có kẹp hoặc chân đế đặt cao khoảng 2–2.5 m.
- Sạc dự phòng. Bộ nhớ trống: 10 phút 1080p tốn khoảng 1–1.5 GB mỗi máy.
- Thước dây ≥ 10 m, băng dính màu (dán dấu sàn), bút và giấy ghi số đo.
- Một người hô "bắt đầu" và vỗ tay đồng bộ.

**Cài đặt mọi điện thoại giống nhau:**
- Quay **ngang**, **1080p**, **cùng một fps** (25 hoặc 30).
- **Tắt chống rung (EIS/OIS điện tử).** Chống rung cắt và dịch khung theo thời gian, nên
  homography đúng ở khung này sẽ sai ở khung khác.
- Tắt HDR và "auto framing". Khoá lấy nét và phơi sáng nếu máy cho phép (giữ chạm vào màn hình).
- Bật chế độ máy bay để cuộc gọi không cắt ngang buổi quay. Bật **ghi âm** (đồng bộ cần tiếng).

**Địa điểm.** Cần một khu có sàn phẳng để làm cặp chồng lấn, nối với một hành lang hoặc lối
đi cho các camera không chồng lấn. Bố trí mẫu ở `configs/lab/topology.yaml`:

```
┌──────── khu A ────────┐                hành lang              đầu kia
│ cam01 ◣       ◢ cam02 │ ~10–15 m ─── cam03 ─── ~10 m ─── cam04
│  vùng sàn chung ≥3×4 m │
└───────────────────────┘
```

- cam01 và cam02 nhìn **chung** một vùng sàn từ hai góc khác nhau (lệch nhau ≥ 45°). Đây là kịch bản 1.
- cam03 và cam04 không thấy khu A và không thấy nhau. Đây là kịch bản 2.

Không mượn được phòng lab cố định cũng được (ghi nhận 2026-09-28). Một buổi quay 1–2 giờ ở
hành lang, sân trường hoặc sảnh là đủ.

**Đồng thuận** (đề cương 4.3.3): mỗi người tham gia ký [phiếu đồng thuận](phieu-dong-thuan.md)
trước khi quay. Video gốc KHÔNG đưa lên git (`data/` đã ignore) và không đưa lên dịch vụ công khai.

## 2. Bố trí và đo (đề cương 4.3.2 bước 1)

1. **Gắn camera** ở độ cao 2–2.5 m, chúc xuống 20–30°, sao cho người đi qua thấy được trọn
   người (cả bàn chân). Điểm chân là thứ được chiếu xuống sàn: chân bị khuất thì toạ độ mét sai.
2. **Chọn một gốc toạ độ sàn chung cho cả khu**, ví dụ một góc tường của khu A. Trục x chạy dọc
   hành lang, trục y vuông góc với nó, đơn vị mét. Mọi camera, kể cả cam03/cam04, đều dùng hệ
   toạ độ này: `HomographyMapper` chỉ nạp các camera cùng một mặt phẳng, và nối mảnh tracklet
   cùng camera cần homography của chính camera đó.
3. **Dán 6–8 dấu băng dính** trên sàn trong vùng nhìn của MỖI camera:
   - trải rộng khắp khung hình, không có ba dấu thẳng hàng;
   - khu A: phần lớn dấu nằm trong vùng chung để cả cam01 và cam02 cùng thấy.
4. **Đo toạ độ (x, y) của từng dấu** bằng thước rồi ghi lên giấy theo mẫu: `cam01 — dấu 1:
   (0.00, 0.00)`. Đo hai lần những dấu ở xa gốc.
5. **Đo quãng đường đi bộ** giữa vùng nhìn của các camera (khu A → cam03 → cam04). Số này
   điền vào `distance_m` của topology.
6. **Chụp ảnh và ghi lại để đưa vào chương 4:**
   - sơ đồ bố trí;
   - chiều cao và góc của từng camera;
   - mẫu điện thoại và cài đặt quay.

Giữ băng dính trên sàn suốt buổi quay. Phải có ít nhất một khung hình thấy rõ các dấu, vì mục 4
đọc pixel của dấu từ chính video.

## 3. Kịch bản quay (đề cương 4.3.2 bước 2, mục 6.2)

Mỗi đoạn là một **buổi quay** riêng, tức một thư mục `data/lab/raw/<buổi>/`. Tách riêng thì
mỗi kịch bản có bảng điểm riêng, đúng như đề cương mục 6.2 đòi.

| Buổi | Nội dung | Người | Thời lượng | Dùng để |
|---|---|---|---|---|
| `calib` | 1 người đi hết các tuyến A → cam03 → cam04 và ngược lại, 3 lượt: chậm / bình thường / nhanh | 1 | ~4 phút | đo transit time (KHÔNG chấm điểm) |
| `s1` | kịch bản 1: đi lại trong khu A, ra vào vùng chung giữa cam01 và cam02 | 1–2 | ~2 phút | liên kết chồng lấn (homography) |
| `s2` | kịch bản 2: A → cam03 → cam04 → quay lại, 3 lượt; một lượt dừng khuất khỏi cam03 vài giây rồi quay lại | 1–2 | ~3 phút | liên kết không chồng lấn + quay lại cùng camera |
| `s3` | kịch bản 3: 4–6 người đi cùng lúc, cắt ngang nhau, che nhau ở khu A và hành lang | 4–6 | ~3 phút | chống nhiễu, che khuất |
| `s4` | kịch bản 4: 2–3 người mặc áo cùng màu, đi cùng các tuyến | 2–3 | ~3 phút | độ phân biệt của Re-ID |

**Quy tắc cho MỌI buổi quay** (các bước 2–4 lấy từ `tools/sync_recordings.py`):
1. Bấm quay trên cả 4 máy, theo thứ tự nào cũng được.
2. Đứng ở chỗ mọi máy nghe được, **vỗ tay theo nhịp không đều**: 1 tiếng, nghỉ 2 giây, rồi 2
   tiếng nhanh. Nếu được, vỗ trong tầm nhìn của càng nhiều camera càng tốt để kiểm bằng mắt.
3. Thực hiện kịch bản. Người tham gia đi tự nhiên, không nhìn camera.
4. Vỗ tay lần nữa theo cùng nhịp, rồi dừng quay trên cả 4 máy.
5. Ghi lên giấy: tên buổi, giờ bắt đầu, ai tham gia, mặc gì, có gì bất thường.

**Đặt tên file khi chép về máy:** `data/lab/raw/<buổi>/cam01.mp4` … `cam04.mp4`. Tên camera
theo **vị trí**, không theo tên điện thoại. Phần đuôi file không quan trọng.

**Trước khi rời hiện trường:** mở thử một file của mỗi máy; kiểm đủ 5 buổi × 4 file; chụp
ảnh các dấu sàn từ đúng vị trí mỗi camera (để đo lại nếu lỡ dịch camera).

## 4. Đồng bộ, hiệu chỉnh

**Đồng bộ + fps cố định** (cần `ffmpeg` có `libx264`; máy dev hiện KHÔNG có, nên chạy trên
`vast-gpu` bằng `docker/vast_lab.sh ffmpeg` + `sync`, hoặc trên bất kỳ máy Linux/Mac nào có ffmpeg):

```bash
PYTHONPATH=src python -m tools.sync_recordings \
    --video cam01=data/lab/raw/s1/cam01.mp4 --video cam02=data/lab/raw/s1/cam02.mp4 \
    --video cam03=data/lab/raw/s1/cam03.mp4 --video cam04=data/lab/raw/s1/cam04.mp4 \
    --out-dir data/lab/s1 --fps 25 --execute
```

- Lệnh in độ lệch và độ tin cậy của từng máy. Độ tin cậy dưới 2 thì xem lại.
- Lệnh in "khung vỗ tay". **Mở cả 4 video ở khung đó**: hai bàn tay phải chạm nhau ở cùng
  khung, ±1 khung. Lệch nhiều hơn thì sửa bằng `--offset camXX=<giây>`.
- Video đầu ra `data/lab/<buổi>/camXX.mp4` là file DUY NHẤT dùng cho cả pipeline lẫn CVAT. Đừng
  đưa video gốc lên CVAT, vì số khung sẽ lệch.

**Hiệu chỉnh homography:**
1. Mở `data/lab/calib/camXX.mp4` (đã đồng bộ) ở một khung thấy rõ các dấu sàn. Đọc toạ độ
   pixel của từng dấu bằng trình xem ảnh bất kỳ; Paint của Windows hiện toạ độ ở góc dưới.
2. Điền vào `configs/lab/ground_points.yaml`: pixel vào `image`, mét đo ở mục 2 vào `world`.
   Xong thì **xoá dòng `status: template`**.
3. Chạy `make lab-homography`. Đọc p95 sai số chiếu của từng camera; trên 0.3 m nghĩa là có dấu
   đo sai hoặc đọc pixel sai.

## 5. Chú thích CVAT (đề cương 4.3.2 bước 3–5)

**Tạo task:**
- Mỗi camera của mỗi buổi là **một task**, tạo từ `data/lab/<buổi>/camXX.mp4` (bản đã đồng bộ).
- Ở "Advanced configuration", đặt **Frame step = 5**: video 25 fps được chú thích ở 5 fps, đúng
  khoảng 5–10 fps mà đề cương đề ra.
- Label `person`. Thêm attribute `person_id` kiểu **text**.

**Quy ước, phải phổ biến cho mọi người gán nhãn** (chi tiết trong docstring `tools/cvat_to_mot.py`):
1. Mỗi người là một **track**. Dùng chế độ track của CVAT: vẽ hộp ở các khung khoá, CVAT tự nội suy.
2. `person_id` của cùng một người phải **giống nhau ở mọi camera**, ví dụ `P01`, `P02`. Người
   qua đường không tham gia cũng phải chú thích, đặt `X01`, `X02`…; bỏ sót họ thì hộp đúng của
   hệ thống bị tính là báo nhầm.
3. Người ra khỏi khung: bấm **outside**. Người quay lại: dùng lại track cũ (bỏ outside) hoặc tạo
   track mới, đều được, công cụ cắt lần xuất hiện theo khoảng trống.
4. Người bị che quá nửa: đánh dấu **occluded**, vẫn giữ hộp.
5. Hộp ôm sát người, gồm cả bàn chân, vì điểm chân là đáy hộp.

**Xuất:** chọn định dạng **"CVAT for video 1.1"** (KHÔNG chọn MOT, vì bản MOT mất `person_id`),
lưu thành `data/cvat/<buổi>/camXX.xml`. Sau đó chạy:

```bash
make lab-gt LAB_SESSION=s1    # -> eval/gt/lab_s1/ + data/fixtures/lab_s1_gt.jsonl
```

Công cụ tự nhận frame step trong `<meta>`. Số khung chú thích nằm ngoài tập khung của task thì
công cụ dừng và báo lỗi, không chấm lệch.

**Ước lượng công:** khoảng 15 phút video × 4 camera × 5 fps ≈ 18 000 khung. Nhờ nội suy, mỗi
người chỉ cần khung khoá mỗi 1–2 giây cộng chỗ đổi hướng. Kịch bản 3 (đông người) tốn công nhất;
nếu thiếu thời gian, rút s3 xuống 1–2 phút.

## 6. Topology: transit time

Chú thích đoạn `calib` trước, rồi:

```bash
make lab-gt LAB_SESSION=calib
make lab-transit LAB_SESSION=calib    # -> data/lab/transitions_calib.suggest.yaml
```

- Chép các khoảng `min_ms`/`max_ms` đề xuất vào `configs/lab/topology.yaml`. Đề xuất đã nới
  ±30% và cộng 2 s.
- Kiểm `overlap_pairs_detected` khớp với `overlaps_with`.
- Đổi `status: template` → `status: measured`.

**Không đo transit trên s1–s4.** Ràng buộc thời gian lấy từ chính tập chấm điểm sẽ vừa khít
mọi cặp đúng của tập đó; đó là chỉnh tham số trên đáp án.

## 7. Kiểm trước khi thuê GPU

```bash
make lab-check LAB_SESSION=s1
```

Phải ra `0 FAIL`. Công cụ bắt các lỗi im lặng đã từng tốn tiền thuê máy:
- cặp chồng lấn chưa hiệu chỉnh;
- kích thước ảnh lệch;
- cam_id lệch;
- fps lệch;
- `sync: false` với nguồn file;
- chú thích trên nhầm file.

## 8. Pipeline trên `vast-gpu`

**Hỏi xác nhận trước mọi lệnh trên `vast-gpu`** (CLAUDE.md §2). Ước lượng:
- RTX A4000 khoảng $0.09/h;
- mỗi buổi 3 lần × (thời lượng + 30 s), cộng 5 phút build engine;
- cả 4 buổi s1–s4 xong trong khoảng 1 giờ.

Thứ tự lệnh nằm ở đầu `docker/vast_lab.sh`: kiểm NVDEC trước, rồi bootstrap → (ffmpeg, sync)
→ engine → run → fps → pack. Mỗi lần `run` cho ra hai thứ cùng lúc:
- fixture để chấm lại trên máy dev;
- log độ trễ theo đồng hồ thật. Đây là lần đầu đo được (A) độ trễ theo khung và (B) thời gian
  tới Global ID trên dữ liệu 25 fps (định nghĩa ở phiên 32).

Xong thì `vastai destroy instance <id> -y` và kiểm `vastai show instances` không còn gì.

## 9. Chấm điểm và quét tham số

```bash
make lab-eval LAB_SESSION=s1
make lab-eval LAB_SESSION=s2 LAB_EVAL_ARGS="--variant base \
    --variant mot_nguong:association.max_cost_geometric=null \
    --variant w500:association.window_ms=500 \
    --variant gap_reject:association.ground_gap_policy=reject"
```

Ra `data/lab/eval/<buổi>/summary.md`, gồm hai bảng:
- TrackEval xuyên camera: HOTA / DetA / AssA / IDF1 / IDs, trung bình ± độ lệch chuẩn trên 3 lần
  chạy pipeline;
- **độ chính xác bàn giao danh tính theo loại cặp**: chồng lấn / không chồng lấn / quay lại cùng
  camera.

Bảng thứ hai là con số trực tiếp nhất cho đóng góp chính của đồ án (`eval/eval_handover.py`).
Đơn camera (MOTA/IDF1 của NvDCF) nằm trong `summary.json`, mục `sct`.

**Lưới quét đề xuất** (mỗi biến thể chỉ đổi một khoá so với `base`):

| Khoá | Giá trị | Câu hỏi |
|---|---|---|
| `association.max_cost` | 0.20, 0.25, 0.30, 0.35 | ngưỡng ngoại hình cho cặp không chồng lấn |
| `association.max_cost_geometric` | null, 0.6, 0.9, 1.2 | ngưỡng riêng cho ô có vị trí có đáng không (null = một ngưỡng chung) |
| `association.window_ms` | 500, 1000 | giá độ chính xác của (B) < 1 s ở 25 fps (phiên 32 mới đo ở 2 fps) |
| `association.ground_gap_policy` | allow, reject | `reject` có hại cho cặp không chồng lấn không |
| `tracklet.min_frames` | 3, 5, 10 | đánh đổi độ trễ chốt danh tính / độ chính xác (phiên 28) |

Mỗi biến thể chạy lại engine trên 3 fixture, mất khoảng 1 phút mỗi biến thể mỗi buổi trên máy
dev. Đây là việc CPU, nặng vừa: quét cả lưới cùng lúc thì hỏi trước (bộ nhớ dự án: máy dev yếu).

## 10. Số cần mang vào chương 6

Mọi số phải ghi kèm cấu hình: GPU, model, độ phân giải, số luồng, giao thức chấm (CLAUDE.md §7).

- Từng kịch bản s1–s4: HOTA / AssA / IDF1 (hộp ảnh, chỉ khung chú thích, IoU 0.5), n = 3.
- Độ chính xác bàn giao theo loại cặp, n = 3, kèm số lần chuyển.
- Đơn camera: MOTA / IDF1 của NvDCF, để tách lỗi theo tầng (nguyên tắc 1, CLAUDE.md §1).
- Độ trễ (A) và (B) theo đồng hồ thật, p50 / p95, kèm số tracklet. Định nghĩa ở CLAUDE.md §7.
- FPS mỗi luồng ở 4 luồng, GPU, VRAM (`docker/vast_lab.sh fps`).
- Kết quả lưới quét: chọn cấu hình có HOTA cao nhất **trong số** các cấu hình đạt (B) p95 < 1 s
  (quy tắc ở phiên 32, QĐ 4.2).
- Fine-tune (đề cương M6) chỉ làm khi bảng bàn giao cho thấy ngoại hình tụt rõ so với WildTrack,
  và khi làm thì trình bày như một ablation (CLAUDE.md §9).

**Ảnh minh hoạ trong báo cáo/slide phải làm mờ mặt** (đề cương 4.3.3).

## 11. Demo trên dashboard bằng dữ liệu lab

Dashboard đọc topology và homography qua biến môi trường, nên chỉ cần trỏ sang `configs/lab/`.
Lệnh giống mục Demo của README gốc, cần Redis:

```bash
# shell 1 — engine
MCT_DB_PATH=data/lab/demo.db PYTHONPATH=src python -m mct --config configs/lab/lab.mct.yaml \
    --topology configs/lab/topology.yaml --homography-dir configs/lab/homography \
    --db data/lab/demo.db --publish
# shell 2 — dashboard
MCT_DB_PATH=data/lab/demo.db MCT_TOPOLOGY=configs/lab/topology.yaml \
    MCT_HOMOGRAPHY_DIR=configs/lab/homography PYTHONPATH=src uvicorn dashboard.app:app --port 8000
# shell 3 — phát lại một lần chạy pipeline đúng nhịp gốc
PYTHONPATH=src python -m tools.replay_metadata --fixture data/fixtures/lab_s2_r1.jsonl
```

Chưa có dữ liệu thật thì demo được bằng buổi quay giả lập (`tools.make_synthetic_lab`). Khi đó
trỏ topology/homography sang `<out>/configs/` của bộ sinh.
