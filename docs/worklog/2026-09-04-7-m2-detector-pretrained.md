# 2026-09-04 (phiên 7) — M2/M3 bỏ bước fine-tune: dùng thẳng weight pretrained

- **Mốc:** M2 (đóng lại) + M3 (thu hẹp phạm vi) | **Máy:** máy dev (soạn) + `ut-hpc` (test/lint) | **Thời lượng:** ~1h

## Mục tiêu phiên

- Rà lại M2: có thật sự cần fine-tune YOLO trên COCO-person không.
- Nếu không: sửa lộ trình, sửa cấu hình pipeline cho khớp, và dọn phần skill `ut-hpc`
  đang mô tả một quy trình sẽ không bao giờ chạy.

## Đã làm

- **`configs/pipeline/config_infer_yolo11.txt` + `config_infer_yolo11_b4.txt`** — thêm khối
  lọc lớp: `[class-attrs-all]` với `pre-cluster-threshold=1.0` (chặn hết) + `[class-attrs-0]`
  với `pre-cluster-threshold=0.25` (mở riêng lớp `person`). Viết lại toàn bộ khối comment M2
  ở cuối hai file.
- **`tests/test_pipeline_configs.py`** (mới, 7 test) — canh chính khối trên: mặc định phải
  chặn, lớp 0 phải mở, không lớp nào khác được mở thêm, `num-detected-classes` khớp số dòng
  `labels.txt`, và hai bản b1/b4 chỉ được khác nhau ở batch size + tên file engine.
- **`CLAUDE.md`** — sửa §2 (vai trò `ut-hpc`), §9 (dòng M2, M3, M6 + một đoạn giải thích
  "vì sao không còn fine-tune"), §11 (áp lực dung lượng home giảm).
- **`src/ds_pipeline/__init__.py`** — docstring: M2 ghi là xong, M3 thu hẹp còn "tải model
  ReID pretrained + bật `reidType: 2`". Nhân tiện sửa con số FPS trong docstring từ 410.8
  (số đo lẫn chi phí log, đã bị bác ở phiên 2) sang 618.5 đo bằng `--stats`.
- **Skill `ut-hpc`** — `SKILL.md` (mô tả + khung cảnh báo + cây thư mục `~/mct/data/`),
  `reference/workflows.md` (thay hai mục "M2 fine-tune YOLO" / "M3 fine-tune Re-ID" bằng
  một mục giải thích vì sao bỏ + một mục M6 cho dữ liệu tự thu), `reference/cluster.md`.
- **Xoá `templates/prepare_coco_person.py`** — script tải COCO rồi lọc lớp person, giờ không
  còn đường dùng nào. Còn trong lịch sử git nếu cần.
- **`templates/train_yolo.sbatch` / `train_reid.sbatch` / `train_reid.py` giữ lại nhưng đổi
  đích sang M6**: `DATA` mặc định trỏ `~/mct/data/lab/`, `SOURCES=lab`, tên output
  `yolo11s_lab` / `osnet_x1_0_lab`. Ghi rõ trong header rằng đây KHÔNG phải bước của pipeline
  chính, chỉ chạy sau khi đo được domain gap, và trình bày như ablation.

## Quyết định kỹ thuật

**1. Bỏ fine-tune ở M2 và M3 — dùng weight pretrained.** Cả hai bước fine-tune trong lộ
trình cũ đều nhắm vào **chính bộ dữ liệu mà weight pretrained đã được huấn luyện trên đó**:

| | lộ trình cũ | weight pretrained đã train trên |
|---|---|---|
| Detector | fine-tune YOLO11s trên COCO-person | COCO 80 lớp — `person` là lớp 0 |
| Re-ID | fine-tune OSNet trên Market-1501/MSMT17 | đúng Market-1501 / MSMT17 (torchreid) |

Fine-tune trên tập con của dữ liệu model đã thấy là học lại cái đã học: không tạo uplift đáng
kể, mà tốn GPU-hour và ~50G đĩa trên cụm dùng chung. Phương án bị loại vì lý do này, không
phải vì thiếu tài nguyên.

Fine-tune chỉ có nghĩa khi có **domain gap đo được**. Và đồ án đã có sẵn bằng chứng định
lượng chỉ đúng hướng đó, từ phiên 4: trên WildTrack, checkpoint OSNet **đa nguồn / khái quát
hoá miền** cho F1 **0.346** so với **0.277** của checkpoint Market-1501 chuyên biệt (**+25%**),
trần lý thuyết 0.513 vs 0.398. Tức là *khái quát hoá miền ăn đứt chuyên biệt hoá trong miền*
khi camera đích là một domain chưa từng thấy — mà camera của đồ án đúng là như vậy. Phiên 4
đã đặt câu hỏi "cần xem lại kế hoạch M3" và để ngỏ; phiên này đóng lại.

Hệ quả cho báo cáo: fine-tune dời sang **M6, trên dữ liệu tự thu**, và trình bày như một
**ablation** (có / không fine-tune) — một thí nghiệm có kết luận, chứ không phải một bước bắt
buộc của pipeline chính. Kể cả khi chênh lệch bằng 0 thì đó vẫn là kết quả đáng viết.

**2. Cái thật sự cần từ M2 — "tracker chỉ bám người" — lấy bằng cấu hình, không bằng train.**
Lợi ích duy nhất còn lại của một head 1 lớp là detector không trả về ô tô/ghế/túi. Điều đó
đạt được bằng ngưỡng theo lớp trong nvinfer, tốn 0 GPU-hour: đặt ngưỡng mặc định cho mọi lớp
là `1.0` (confidence luôn thuộc [0, 1] nên không lớp nào vượt qua), rồi mở riêng lớp 0 xuống
`0.25`.

Hàm parse của DeepStream-Yolo so `maxProb < preclusterThreshold[classId]` nên ngưỡng 1.0 loại
sạch 79 lớp còn lại **trước nvtracker**. Chỗ này quan trọng hơn vẻ ngoài của nó: trước thay
đổi, `probes.py` mới là nơi lọc `class_id == 0` — tức nvtracker vẫn đang bám và (từ M3) vẫn
sẽ trích embedding ReID cho ô tô, rồi ta vứt đi ở tầng sau. Lọc tại detector sửa cả lãng phí
tính toán lẫn chuyện ReID chạy trên vật thể không phải người.

Giữ **cả hai** tầng lọc: `probes.py` vẫn còn `if obj_meta.class_id == 0` làm lưới an toàn cho
schema, còn nvinfer lo phần chi phí. Hai tầng chặn hai loại hỏng khác nhau.

Phương án bị loại: `filter-out-class-ids=1;2;...;79` trong `[property]` (DS ≥ 6.1) — phải liệt
kê 79 id và lọc sau khi parse xong.

**3. Có test cho file cấu hình, không chỉ cho code.** Xoá nhầm khối `[class-attrs-*]` thì
pipeline vẫn chạy, vẫn ra FPS đẹp, chỉ có điều tracker đi bám ô tô — một lỗi **không có triệu
chứng**, đúng họ với cạm bẫy "có homography nhưng thiếu topology" của phiên 6. Test đọc file
bằng `configparser`, không cần GPU, nên chạy được trong bộ test thường.

**4. Giữ `train_*.sbatch` thay vì xoá.** Kịch bản M6 vẫn cần đúng những script đó, chỉ khác
dataset. Xoá đi rồi viết lại từ đầu ở tuần 16 là tự tạo việc — nhưng phải đổi mặc định sang
`data/lab/` ngay bây giờ, vì một template trỏ vào COCO là một cái bẫy cho phiên sau.

## Số liệu đo được

Không đo mới trong phiên này. Hai con số **đã có từ trước** được dùng làm căn cứ cho quyết
định 1, chép lại kèm nguồn:

| số đo | giá trị | nguồn |
|---|---|---|
| F1 liên kết, OSNet DG vs Market-1501 (WildTrack 3 cam) | 0.346 vs 0.277 (+25%) | phiên 4 |
| trần lý thuyết chỉ dùng ngoại hình, DG vs Market-1501 | 0.513 vs 0.398 | phiên 4 |
| FPS 1 luồng, YOLO11s pretrained COCO, RTX 3090, 1080p FP16 | 618.5 (`--stats`) | phiên 2 |
| FPS 4 luồng, cùng cấu hình | 186.4 /luồng | phiên 2 |

Test sau thay đổi: **267 passed, 7 skipped**, `ruff check` + `ruff format --check` sạch
(`ut-hpc`, Python 3.10.12). Bảy test mới: 5 pass + 2 skip trên `ut-hpc` (hai test so
`num-detected-classes` với `labels.txt` bị skip vì `models/` gitignored, không đẩy lên cụm);
cả 7 pass trên máy dev nơi có `models/detector/labels.txt`.

## Vướng mắc / chưa xong

- **Khối `[class-attrs-*]` chưa chạy trên phần cứng thật.** Đây là thay đổi cấu hình
  DeepStream, mà chỗ duy nhất kiểm được là `vast-gpu` (tính phí theo giờ). Test cấu hình chỉ
  xác nhận file *nói* đúng điều ta muốn, không xác nhận nvinfer *hiểu* như vậy. Cần xác minh
  ở lần thuê tới: đếm số đối tượng ra khỏi detector trước/sau, và đo lại FPS — kỳ vọng FPS
  **tăng** vì nvtracker nhận ít đối tượng hơn.
- Ngưỡng `pre-cluster-threshold=1.0` dựa trên việc confidence luôn `< 1.0`. Đúng về mặt toán
  (sigmoid không đạt 1.0) nhưng cũng thuộc diện phải xác minh cùng lần trên.
- M3 còn lại: chọn checkpoint OSNet nào để nhúng vào `nvtracker` (`reidType: 2`), và định
  dạng model ReID mà DeepStream 7.1 nhận — chưa tra tài liệu bản đã cài.

## Bước tiếp theo

1. M3 với model pretrained: tải OSNet DG về `models/reid/`, chuyển sang định dạng
   `nvtracker` nhận, bật `reidType: 2` trong `config_tracker_NvDCF_perf.yml`.
2. Lần thuê `vast-gpu` tới (hỏi xác nhận trước): xác minh khối lọc lớp + đo lại FPS, gộp
   chung chuyến với việc đo độ trễ end-to-end còn treo từ phiên 6.
3. Cảnh báo "hai Global ID quá gần nhau trên mặt phẳng" — vẫn treo từ phiên 6.
