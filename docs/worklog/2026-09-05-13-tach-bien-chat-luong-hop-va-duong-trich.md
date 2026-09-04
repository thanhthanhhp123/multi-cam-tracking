# 2026-09-05 (phiên 13) — Tách hai biến: hộp detector không làm hỏng ngoại hình, nó làm hỏng HÌNH HỌC

- **Mốc:** M4 (chẩn đoán) | **Máy:** máy dev (soạn) + `ut-hpc` (chạy) | **Thời lượng:** ~2h, **không tốn GPU**

## Mục tiêu phiên

Phiên 12 chốt nút thắt đã chuyển sang ngoại hình. Giả thuyết đứng đầu danh sách: **crop cắt
từ hộp detector kém hơn crop cắt từ hộp ground-truth** (hộp lệch thì crop lấy thêm nền, mất
mất một phần người). Nhưng hai fixture đang có khác nhau ở **hai** biến cùng lúc:

| fixture | hộp cắt crop | bộ trích embedding |
|---|---|---|
| `wildtrack_7cam.jsonl` | ground-truth | ONNX Runtime (CPU) |
| `ds_wildtrack_7cam.jsonl` | detector YOLO11s | TensorRT trong nvtracker |

So hai cái đó thì chênh lệch không quy về nguyên nhân nào. Phiên này sinh thêm hai fixture
để tách sạch từng biến.

## Đã làm

- **`src/tools/reembed_fixture.py`** (mới) — trích lại embedding cho fixture DeepStream bằng
  ONNX Runtime, cắt crop từ **hộp detector** (`--boxes fixture`) hoặc **hộp GT đã khớp IoU**
  (`--boxes gt`). Cả hai chế độ chỉ giữ detection khớp được GT, nên hai đầu ra có **đúng cùng
  tập detection, cùng `local_track_id`, cùng `ts_ms`** — khác đúng toạ độ hộp.
- **`tests/test_reembed_fixture.py`** (13 test) — phần lớn canh đúng tính chất đó: lệch tập
  detection là hỏng cả thí nghiệm.
- `wildtrack_to_fixture.py`: tách `crop_for_reid()` thành hàm public, cả hai công cụ gọi
  chung — hai cách cắt khác nhau thì chênh lệch đo được là của đoạn code, không phải của hộp.
- **`.claude/skills/ut-hpc/templates/reembed_fixture.sbatch`** (mới) — job CPU 16 core,
  ~11 phút cho cả hai chế độ.
- Chạy chẩn đoán trên bốn fixture, đo cả trần ngoại hình lẫn F1 thật.
- **363 passed, 8 skipped**, ruff sạch.

## Quyết định kỹ thuật

**1. Thiết kế thí nghiệm hai biến, không so hai fixture có sẵn.** Cám dỗ là so thẳng
`wildtrack_7cam` với `ds_wildtrack_7cam` rồi kết luận. Làm vậy sẽ sai: chúng khác nhau ở hộp,
ở bộ trích, **và** ở cấu trúc tracklet (SCT lý tưởng vs tracker thật). Sinh thêm hai fixture
tốn 11 phút CPU và cho phép đọc ra từng nguyên nhân riêng. Đây là kiểu chi phí nên trả.

**2. Giả thuyết "hộp detector làm hỏng ngoại hình" — BÁC BỎ.** Cùng bộ trích, cùng tracklet,
chỉ đổi hộp: trần ngoại hình **0.381 (hộp detector) vs 0.379 (hộp GT)** — chênh lệch nằm
trong nhiễu. OSNet resize crop về 256×128 nên vài pixel lệch ở biên hộp bị nuốt mất trong
phép nội suy. Không cần cải thiện độ chính xác hộp *vì lý do ngoại hình*.

**3. Nhưng hộp detector làm hỏng HÌNH HỌC, và nặng.** Điểm chân = đáy-giữa bbox, nên hộp
lệch bao nhiêu thì điểm chân lệch bấy nhiêu — rồi qua homography thành sai số mét. Đo trên
đúng cặp fixture đó:

| d_ground giữa cặp tracklet cùng người | hộp detector | hộp GT |
|---|---|---|
| p25 | 0.43 m | **0.08 m** |
| trung vị | 0.74 m | **0.21 m** |

Trong khi cặp **khác** người không đổi (7.77 m vs 7.78 m). Tức hộp GT không làm mọi thứ nhỏ
đi, nó **siết riêng phân bố của cặp đúng** — chính là thứ tạo ra khả năng phân biệt.

Hệ quả ở F1 thật: **0.218 (hộp detector) → 0.267 (hộp GT)**, +22%, dù trần ngoại hình y hệt.
Toàn bộ mức tăng đó đi qua thành phần hình học.

Đây là kết luận đảo ngược trực giác ban đầu và đáng viết vào báo cáo: **trong hệ MTMCT có
homography, độ chính xác của bounding box quan trọng vì nó là đầu vào của phép định vị, chứ
không vì nó ảnh hưởng đặc trưng ngoại hình.**

**4. Tiền xử lý ReID của DeepStream đáng 14% trần ngoại hình, và gần bằng 0 ở F1 thật —
không đáng sửa.** Cùng hộp, chỉ đổi bộ trích: trần **0.327 (TensorRT trong nvtracker) vs
0.381 (ONNX Runtime)**. Phiên 8 quyết định 4 đã dự đoán chênh lệch này (netScaleFactor là số
vô hướng trong khi ImageNet có std riêng từng kênh, cộng FP16) và ghim lại bằng test. Giờ có
số: nó thật, khoảng −14% trần. Nhưng ở F1 thật thì **0.224 vs 0.218** — đảo chiều và nằm
trong nhiễu. Chốt: **không sửa**, giữ đường DeepStream nguyên trạng, ghi con số vào báo cáo
như một sai số đã lượng hoá.

**5. Độ dài tracklet ảnh hưởng tới ngoại hình mạnh hơn cả hai biến trên cộng lại.** Trần
ngoại hình của cùng một fixture DeepStream: **0.181** với `idle_timeout_ms=2000` (phiên 11)
và **0.327** với 30000. Gần gấp đôi, chỉ vì tracklet dài hơn thì `query_embedding` có nhiều
mẫu confidence cao để chọn top-k. Điều này củng cố mạnh cho việc **ghép tracklet**: nó cải
thiện đồng thời cả hình học (có mốc thời gian chung) lẫn ngoại hình (query tốt hơn).

## Số liệu đo được

**Cấu hình:** `ut-hpc`, `~/mct/venv-reid` (onnxruntime 1.23.2, cv2 5.0.0) cho phần trích lại,
`~/mct/venv-test` (Python 3.10.12) cho phần chấm. OSNet `osnet_x1_0_msdc_dg`, cùng checkpoint
với đường DeepStream. Bốn fixture đều 7 camera WildTrack. `idle_timeout_ms=30000`,
`min_frames=3`, homography 7 camera đã hiệu chỉnh.

Hai fixture mới: **2770 message, 19139/34597 detection giữ lại** (đúng bằng tỉ lệ khớp IoU
55.3% mà `ds_wildtrack_gt` đo độc lập — một phép kiểm chéo tự nhiên).

### Trần lý thuyết chỉ với ngoại hình

| fixture | hộp | bộ trích | trần F1 | ngưỡng tối ưu |
|---|---|---|---|---|
| `ds_wildtrack_7cam` | detector | **TensorRT** | **0.327** | cosine ≥ 0.803 → `max_cost` 0.197 |
| `..._onnx_detbox` | detector | **ONNX RT** | **0.381** | cosine ≥ 0.781 → `max_cost` 0.219 |
| `..._onnx_gtbox` | **GT** | ONNX RT | **0.379** | cosine ≥ 0.777 → `max_cost` 0.223 |

- **Hộp**: 0.381 → 0.379. Không ảnh hưởng.
- **Bộ trích**: 0.381 → 0.327. −14%.

Phân bố cosine (cặp tracklet khác camera) của bản `onnx_gtbox`: cùng người trung vị 0.732,
khác người 0.575. Chồng lấn nặng kể cả khi hộp là ground-truth — **ReID xuyên camera trên
WildTrack là bài toán khó tự thân**, không phải lỗi của pipeline.

### Khoảng cách mặt đất (cùng tracklet, cùng bộ trích, chỉ khác hộp)

| | hộp detector | hộp GT |
|---|---|---|
| cặp cùng người, p25 | 0.43 m | **0.08 m** |
| cặp cùng người, trung vị | 0.74 m | **0.21 m** |
| cặp cùng người, p75 | 2.08 m | 1.69 m |
| cặp khác người, trung vị | 7.77 m | 7.78 m |
| cặp đúng có mốc thời gian chung | 62.1% | 62.1% |

### F1 thật (`geo=reject`, `idle 30000`, `max_cost 0.80`, `d_max 5.0`)

| fixture | P | R | F1 |
|---|---|---|---|
| `ds_wildtrack_7cam` (detector + TensorRT) | 0.417 | 0.153 | 0.224 |
| `..._onnx_detbox` (detector + ONNX RT) | 0.281 | 0.178 | 0.218 |
| **`..._onnx_gtbox` (GT + ONNX RT)** | 0.301 | 0.239 | **0.267** |

### Bảng quy nạp nguyên nhân

| biến | ảnh hưởng lên ngoại hình | ảnh hưởng lên hình học | ảnh hưởng lên F1 |
|---|---|---|---|
| chất lượng hộp (detector → GT) | **không** (0.381→0.379) | **lớn** (d_ground 0.74→0.21 m) | **+22%** (0.218→0.267) |
| đường trích (ONNX → TensorRT) | −14% (0.381→0.327) | không | ~0 (0.218→0.224) |
| độ dài tracklet (idle 2000→30000) | **+81%** (0.181→0.327) | **lớn** (mốc chung 27%→67%) | +6% (0.170→0.180) |

## Vướng mắc / chưa xong

- **Ghép tracklet trong cùng camera là việc lớn nhất còn lại, và giờ có ba lý do độc lập ủng
  hộ nó**: tracklet dài hơn → nhiều mốc thời gian chung hơn (hình học), query embedding tốt
  hơn (ngoại hình), và ít mảnh vụn để gán sai hơn.
- **Làm mượt điểm chân theo thời gian** — hướng rẻ vừa lộ ra từ phiên này. Sai số điểm chân
  của hộp detector là nhiễu quanh vị trí thật; lấy trung vị/khớp đường trong một tracklet có
  thể kéo d_ground từ 0.74 m về gần 0.21 m mà **không cần detector tốt hơn**. Làm và thử
  được hoàn toàn offline. Nên làm trước khi nghĩ tới đổi detector.
- Chưa thử ngưỡng `max_ground_dist_m` chặt lại trên fixture `gtbox`: với d_ground trung vị
  0.21 m thì ngưỡng 1.0 m có thể đủ, và siết ngưỡng sẽ tăng precision. Chưa quét.
- Mọi con số vẫn trên WildTrack 2 fps, mọi camera chồng lấn. **Đừng chốt tham số nào theo
  dataset này** — nhắc lại lần thứ ba.

Bốn fixture nằm ở `~/mct/data/fixtures/` trên `ut-hpc` (55–100 MB mỗi cái), không vào git.

## Bước tiếp theo

1. **Làm mượt điểm chân trong tracklet** (`src/mct/tracklet.py`), rồi đo lại d_ground bằng
   `diagnose_tracklets` trên fixture `onnx_detbox` — mục tiêu: kéo trung vị 0.74 m xuống gần
   mức của hộp GT mà không đụng tới detector.
2. Nếu (1) có tác dụng: quét lại `max_ground_dist_m` chặt hơn để lấy thêm precision.
3. Ghép tracklet trong cùng camera.
