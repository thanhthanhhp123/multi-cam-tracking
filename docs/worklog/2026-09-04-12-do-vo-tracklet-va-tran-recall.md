# 2026-09-04 (phiên 12) — Định lượng độ vỡ tracklet: nút thắt nằm ở đâu, và hai tham số đặt sai

- **Mốc:** M4 (chẩn đoán) | **Máy:** máy dev (soạn) + `ut-hpc` (chạy) | **Thời lượng:** ~1.5h, **không tốn GPU**

## Mục tiêu phiên

Phiên 11 kết luận điểm nghẽn đã chuyển từ module liên kết sang tracking đơn camera, nhưng
đó mới là chẩn đoán định tính. Trước khi động vào bất kỳ tham số nào — hoặc tệ hơn, trước
khi thuê máy để chỉnh tracker — cần biết **vỡ đến mức nào** và **ràng buộc hình học còn cho
phép ghép bao nhiêu phần trăm số cặp đúng**. Nếu trần đó đã thấp thì mọi công chỉnh ngưỡng
ngoại hình là vô ích.

## Đã làm

- **`eval/diagnose_tracklets.py`** (mới) — báo cáo ba phần: độ vỡ tracklet, tỉ lệ cặp
  tracklet khác camera có mốc thời gian chung, và **trần recall** của ràng buộc hình học
  theo từng `ground_gap_policy`. Dùng thẳng `_world_path` + `_synchronized_distance` của
  `mct.affinity` chứ không viết lại, để con số nói về hệ thống thật.
- **`tests/test_diagnose_tracklets.py`** (13 test) — dựng cặp tracklet có đáp án biết trước
  (trùng/không trùng thời gian, trong/ngoài tầm) rồi soi đúng ô số tương ứng.
- **`eval/eval_wildtrack.py`**: thêm `--idle-timeout-ms` (trước đây hardcode 2000).
- Chạy chẩn đoán trên cả hai fixture 7 camera (SCT lý tưởng và pipeline thật), quét
  `idle_timeout_ms` và `max_ground_dist_m`.
- **350 passed, 8 skipped**, ruff sạch (`ut-hpc`, Python 3.10.12).

## Quyết định kỹ thuật

**1. Đo trần trước, chỉnh tham số sau.** Việc đầu tiên định làm là hạ ngưỡng detector và
nới tham số tracker — cả hai đều cần thuê GPU. Đo trần trước cho thấy điều đó sẽ lãng phí:
ràng buộc hình học lúc ấy chỉ cho qua **13.8%** số cặp đúng, mà recall đo được đã là 11.0%
— tức associator đã lấy được **80% những gì ràng buộc cho phép**. Nút thắt không nằm ở
ngưỡng ngoại hình, cũng chưa nằm ở detector, mà ở chính hai tham số hình học/thời gian.

**2. `idle_timeout_ms` phụ thuộc frame rate, và đang đặt sai cho dữ liệu 2 fps.**
`configs/demo/wildtrack.mct.yaml` để 2000 ms. Ở 2 fps, đó là **đúng 4 khung** — hụt 4
detection liên tiếp (rất thường xuyên khi người bị che) là tracklet bị chẻ đôi. Hậu quả đo
được: bộ dựng tracklet tự tạo thêm **1.6 lần** độ vỡ ngoài phần do tracker gây ra (345 local
track → 550 tracklet). Nâng lên 30000 ms thì 345 local track ra đúng 345 tracklet, và trần
recall tăng **2.5 lần**. Đây là lỗi tham số, không phải lỗi thuật toán.

Chưa sửa giá trị trong config: nó đang dùng để tái lập số đo phiên 5 trên fixture cũ, và
30000 ms là con số chỉnh trên **một** dataset 2 fps mượn. Đã ghi chú vào config để phiên sau
quyết cùng dữ liệu tự thu M6.

**3. `max_ground_dist_m = 1.0` được chỉnh trên điểm chân CHÍNH XÁC, không dùng lại được với
hộp của detector.** Trên fixture SCT lý tưởng, khoảng cách mặt đất giữa hai tracklet cùng
người có trung vị **0.05 m** — bbox là ground-truth nên điểm chân gần như tuyệt đối. Trên
fixture pipeline thật, con số đó là **0.97 m**, xấu đi gần 20 lần, tức **nằm sát ngay ngưỡng
1.0 m**: hơn nửa số cặp đúng đang bị loại vì "quá xa". Đường cong đánh đổi (xem Số liệu) cho
thấy 3.0 m mới là điểm vận hành hợp lý — cho qua 53.1% cặp đúng mà chỉ lọt 2.2% cặp sai.

Bài học chung, đáng đưa vào báo cáo: **mọi ngưỡng chỉnh trên ground-truth đều phải chỉnh lại
khi chuyển sang đầu ra của detector.** Nguồn sai số đổi thì thang đo đổi theo.

**4. `ground_gap_policy=allow` KHÔNG phải lời giải cho tracklet ngắn.** Nó nâng trần recall
từ 34.5% lên 56.5%, nhưng đồng thời cho **80.8% cặp SAI** lọt qua — tức hình học thôi không
còn lọc gì nữa, toàn bộ gánh nặng dồn sang ngoại hình, mà ngoại hình đã đo được là yếu
(trần F1 0.181, phiên 11). Lý do kỹ thuật: đường dự phòng của `_ground_term` nới ngưỡng theo
`max_ground_dist_m + max_speed_m_s · Δt`, **không chặn trên theo thời gian**. Với Δt trung vị
14 s và p75 44.5 s, ngân sách thành 36–112 m trên một quảng trường chỉ khoảng 12×36 m — tức
là không còn ràng buộc nào. Cần một cổng có chặn trên, không phải `allow`.

**5. Không nướng giá trị đã tinh chỉnh vào config.** `max_cost 0.80` + `max_ground_dist 5.0`
+ `idle_timeout 30000` cho F1 tốt nhất (0.224) trên fixture này, nhưng đó là tinh chỉnh trên
**một** dataset mượn, 2 fps, mọi camera chồng lấn. Ghi vào worklog làm bằng chứng, để config
nguyên, quyết lại ở M6 với dữ liệu thật của đồ án.

## Số liệu đo được

**Cấu hình:** `ut-hpc`, Python 3.10.12, `~/mct/venv-test`. Hai fixture 7 camera WildTrack:
`wildtrack_7cam.jsonl` (bbox GT + SCT lý tưởng, OSNet qua ONNX Runtime) và
`ds_wildtrack_7cam.jsonl` (pipeline DeepStream thật trên Tesla T4, phiên 11). Homography
7 camera đã hiệu chỉnh. `min_frames=3`, `ground_time_tol_ms=400`.

### Độ vỡ và trùng thời gian — hai fixture cạnh nhau

| | SCT lý tưởng | pipeline thật (idle 2000) | pipeline thật (idle 30000) |
|---|---|---|---|
| tracklet chấm được | 1544 | 550 | 345 |
| danh tính | 298 | 146 | 146 |
| danh tính ở ≥2 camera | 298 | 77 | 77 |
| **tracklet/(danh tính, camera)** | **1.02** | **1.82** | **1.14** |
| độ dài tracklet, trung vị (khung) | 15 | 11 | 20 |
| **cặp đúng có mốc thời gian chung** | **94.8%** | **27.2%** | **66.7%** |
| d_ground cùng người, trung vị | **0.05 m** | **0.97 m** | — |
| d_ground khác người, trung vị | 9.63 m | 8.70 m | — |
| **trần recall (`reject`)** | **91.3%** | **13.8%** | **34.5%** |
| lọt lưới cặp sai (`reject`) | 0.2% | 0.1% | — |

Cột đầu giải thích trọn vẹn vì sao fixture cũ cho F1 0.752: 94.8% cặp đúng cùng nhìn thấy
một lúc, và khi cùng lúc thì khoảng cách 0.05 m so với 9.63 m — hình học là bộ phân biệt
gần như hoàn hảo. Không cột nào trong hai cột sau có được điều đó.

### Quét `idle_timeout_ms` (fixture DeepStream)

| `idle_timeout_ms` | tracklet | tracklet/(id,cam) | cặp đúng có mốc chung | trần recall `reject` |
|---|---|---|---|---|
| 2000 (đang dùng) | 550 | 1.82 | 27.2% | 13.8% |
| 10000 | 373 | 1.23 | 55.9% | 28.3% |
| 30000 | 345 | 1.14 | 66.7% | **34.5%** |

345 = đúng số local track trong bảng GT → ở 30000 ms bộ dựng tracklet không còn tự cắt thêm
mảnh nào. Phần vỡ còn lại là của tracker.

### Quét `max_ground_dist_m` (fixture DeepStream, `idle_timeout_ms=30000`)

| `max_ground_dist_m` | cặp ĐÚNG lọt qua | cặp SAI lọt qua |
|---|---|---|
| 1.0 (đang dùng) | 34.5% | 0.2% |
| 2.0 | 45.2% | 1.0% |
| **3.0** | **53.1%** | **2.2%** |
| 5.0 | 60.2% | 4.9% |
| 8.0 | 63.2% | 10.4% |

### F1 thật (fixture DeepStream, `geo=reject`, mode `max`)

| cấu hình | P | R | F1 |
|---|---|---|---|
| gốc (idle 2000, d_max 3.0, max_cost 0.60) | 0.375 | 0.110 | 0.170 |
| idle 30000 | 0.369 | 0.119 | 0.180 |
| idle 30000, d_max 3.0, max_cost 0.80 | 0.378 | 0.146 | 0.210 |
| **idle 30000, d_max 5.0, max_cost 0.80** | **0.417** | **0.153** | **0.224** |
| *(đối chiếu) fixture SCT lý tưởng, cấu hình gốc* | 0.776 | 0.729 | **0.752** |

### Nút thắt đã chuyển chỗ — bằng chứng

- Trước khi sửa: trần 13.8%, recall đạt 11.0% → associator lấy được **80%** phần cho phép.
  Ràng buộc hình học là thứ chặn.
- Sau khi sửa: trần 60.2%, recall đạt 15.3% → associator chỉ lấy được **25%** phần cho phép.
  Ràng buộc không còn là thứ chặn; **ngoại hình + logic gán mới là**.

Đây chính là câu trả lời mà phiên này đi tìm: sửa tham số hình học/thời gian nâng F1 được
32% (0.170 → 0.224) và đáng làm, nhưng nó **không** đưa hệ thống về gần 0.752. Muốn đi xa
hơn phải làm cho nhiều cặp đúng cùng nhìn thấy một lúc hơn — tức ghép tracklet trong cùng
camera, hoặc cải thiện tracker/detector.

## Vướng mắc / chưa xong

- **Ngoại hình yếu là nút thắt tiếp theo, và chưa rõ vì sao.** Trần F1 chỉ với ngoại hình là
  0.181 trên fixture DeepStream (phiên 11). Giả thuyết chưa kiểm: crop từ hộp detector lệch
  so với crop từ hộp GT (bao gồm cả phần nền), và WildTrack có nhiều người rất nhỏ. Kiểm
  được offline: so embedding của cùng một người lấy từ hộp GT với từ hộp detector.
- **Cổng hình học cho trường hợp không trùng thời gian vẫn chưa có lời giải.** `reject` quá
  khắc nghiệt, `allow` quá lỏng vì ngân sách tốc độ không chặn trên. Hướng: chặn trên ngân
  sách theo kích thước cảnh, hoặc phạt theo Δt thay vì cắt nhị phân. Làm và thử được hoàn
  toàn offline.
- **Ghép tracklet trong cùng camera** vẫn là việc lớn chưa làm. Số liệu phiên này ủng hộ nó:
  ngay cả sau khi sửa `idle_timeout_ms`, vẫn còn 1.14 tracklet/(danh tính, camera) và chỉ
  66.7% cặp đúng có mốc chung.
- Chưa đối chiếu chẩn đoán này với dữ liệu ở frame rate cao. WildTrack 2 fps làm mọi con số
  về độ vỡ xấu đi bất thường; camera thật 25 fps sẽ khác hẳn. **Đừng chốt tham số nào theo
  dataset này.**

## Bước tiếp theo

1. Kiểm giả thuyết "crop từ hộp detector làm hỏng embedding": so cosine cùng người giữa hai
   nguồn hộp trên chính hai fixture đã có. Offline, không cần GPU.
2. Thay cổng hình học nhị phân bằng phạt liên tục theo Δt, có chặn trên theo kích thước
   cảnh; đo lại bằng `diagnose_tracklets` rồi mới đo F1.
3. Ghép tracklet trong cùng camera (`src/mct`), nếu (1) và (2) chưa đủ.
