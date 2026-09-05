# 2026-09-05 (phiên 16) — Làm mượt điểm chân: không được gì. Thứ mất điểm là quỹ đạo bị TỈA

- **Mốc:** M4 (cải tiến) | **Máy:** máy dev (soạn) + `ut-hpc` (chạy) | **Thời lượng:** ~2h, **không tốn GPU**

## Mục tiêu phiên

- Làm việc số 1 trong "bước tiếp theo" của phiên 15: **làm mượt điểm chân theo thời gian**
  trong tracklet, rồi chấm bằng thước đo chuẩn (HOTA/AssA) chứ không chỉ F1 tự chế.
- Phiên 13 gọi đây là "hướng rẻ": sai số điểm chân của hộp detector là nhiễu quanh vị trí
  thật, nên lấy trung vị trong một tracklet có thể kéo d_ground từ 0.74 m về gần 0.21 m
  (mức của hộp ground-truth) mà **không cần detector tốt hơn**.

## Đã làm

- **`mct.tracklet.smooth_ground_path()`** (mới) — bộ lọc quỹ đạo điểm chân, bốn chế độ
  `none | median | mean | linear`, cửa sổ đối xứng, cắt quỹ đạo ở chỗ tracker mất dấu lâu
  (`max_gap_ms`) để không trộn hai vị trí cách nhau 30 giây.
- Nối vào `affinity._world_path` — **chỗ duy nhất** quỹ đạo được chuyển sang mét, nên cả
  quỹ đạo tracklet lẫn `GlobalTrack.cam_ground_path` đều đi qua một bộ lọc, không xử lý hai
  lần ở hai nơi. Đường dự phòng một-điểm (`_edge_point`) cũng lấy điểm đầu/cuối đã lọc.
- Ba khoá mới trong khối `association`: `ground_smooth`, `ground_smooth_window`,
  `ground_smooth_max_gap_ms`. Cờ dòng lệnh tương ứng ở `eval/diagnose_tracklets.py` và
  `eval/eval_wildtrack.py`, cộng `--ground-path-max-points`.
- **`eval/diagnose_foot_error.py`** (mới) — tách sai số điểm chân của detector thành **độ
  chệch** và **nhiễu**, để biết trần lý thuyết của mọi bộ lọc trước khi tin vào bộ lọc nào.
- Quét bốn chế độ × ba cỡ cửa sổ trên fixture DeepStream thật; **bác bỏ** hướng làm mượt.
- Truy ra thứ khác đang mất điểm: **`ground_path_max_points: 64` tự tỉa thưa quỹ đạo**, và
  sửa nó (+ siết lại `max_ground_dist_m` theo) đem lại F1 0.224 → 0.256, HOTA 14.245 → 14.374.
- **455 passed, 8 skipped**, ruff sạch. 19 test mới (9 cho bộ lọc, 5 cho tích hợp affinity,
  5 cho phép tách sai số).

## Quyết định kỹ thuật

**1. Lọc ở chỗ ĐỌC, không ghi đè lúc gom tracklet.** `ground_path` giữ nguyên số đo thô;
`affinity` lọc khi cần dùng. Hai lý do: (a) cùng một fixture chấm lại được cả hai cách nên
phép so có/không lọc là thật, không phải so hai lần ghi dữ liệu khác nhau; (b) ở chế độ
online, ghi đè dần theo từng khung có nghĩa là điểm mới nhất chỉ có láng giềng quá khứ —
tức một bộ lọc một phía, làm vị trí trễ nửa cửa sổ (ở 2 fps là ~1 s ≈ 1.5 m), tự tay tạo ra
đúng loại sai số đang muốn khử.

**2. Cửa sổ đối xứng, và `median` không phải lựa chọn hiển nhiên.** `median`/`mean` đều giả
định người **đứng yên** trong cửa sổ. Ở giữa quỹ đạo thì vô hại (hai phía triệt tiêu), nhưng
ở **hai đầu** cửa sổ bị cắt một bên nên điểm đầu/cuối lệch theo hướng di chuyển — mà đúng hai
điểm đó là thứ `_ground_term` dùng ở đường dự phòng. Vì thế có thêm `linear` (khớp đường
thẳng theo thời gian): tốc độ nằm trong mô hình nên không lệch, đổi lại mất tính bền với
ngoại lai. Cả hai tính chất này được khoá bằng test, không phải lập luận trên giấy.

**3. Hướng "làm mượt điểm chân" — BÁC BỎ.** Không chế độ nào, không cỡ cửa sổ nào cải thiện
được gì đáng kể (bảng dưới): d_ground của cặp đúng nhúc nhích 0.94 → 0.86 m, **trần recall
của ràng buộc hình học không đổi** (60.2% → 59.6%, tức tệ hơn một chút), F1 0.224 → 0.229 với
`median` và **tụt xuống 0.195** với `linear`. Đây là kết quả âm, và nó có lời giải thích:

> **`_synchronized_distance` vốn đã là một bộ lọc thời gian.** Nó lấy **trung vị** khoảng
> cách trên mọi mốc thời gian chung của hai quỹ đạo. Nhiễu trung bình 0 đã bị phép trung vị
> đó khử gần hết; lọc thêm ở đầu vào là làm hai lần cùng một việc. Còn `mean`/`linear` thì
> **rải** sai số của một khung xấu sang các khung lành, nên đường một-điểm và các mốc lân cận
> xấu đi — đúng chiều đo được ở F1.

**4. Vì sao phần còn lại không khử được: 36% năng lượng sai số là ĐỘ CHỆCH.** `diagnose_foot_error.py`
so từng detection giữa hai fixture chỉ khác toạ độ hộp (`onnx_detbox` vs `onnx_gtbox`, cùng
tập detection): sai số điểm chân của detector có RMS 0.768 m, trong đó **0.461 m là độ chệch
theo từng tracklet** (hộp cao/thấp có hệ thống, chân bị che, hộp cắt ở biên ảnh) và 0.614 m
là nhiễu. Cộng với việc phép trung vị theo thời gian đã lo phần nhiễu, ngân sách còn lại cho
bất kỳ bộ lọc nào là gần bằng không. **Muốn điểm chân chính xác hơn thì phải sửa hộp, không
phải làm mượt hộp** — kết luận này đóng lại một nhánh của kế hoạch phiên 13.

**5. Giữ code lọc lại, mặc định `none`.** Không xoá: nó là một *ablation* đã lượng hoá cho
chương 6, và giả định "nhiễu đã bị trung vị theo thời gian khử hết" **phụ thuộc mật độ mẫu**.
WildTrack chỉ có 2 fps; ở 25 fps của dữ liệu tự thu, nhiễu từng khung độc lập hơn và số mốc
chung nhiều hơn — chưa có cơ sở nào để nói bộ lọc vô dụng ở đó. Bật lại bằng một khoá YAML.

**6. `ground_path_max_points: 64` → `256` — thứ THẬT SỰ đang mất điểm.** Trong lúc đo, phát
hiện `_push_ground_point` tỉa quỹ đạo (`ground_path[::2]`) mỗi khi đầy trần 64 điểm. Trần đó
là trần **bộ nhớ**, nhưng hệ quả là **mất bằng chứng hình học**: quỹ đạo thưa đi thì hai
camera ít có mốc thời gian chung hơn, mà đó là điều kiện sống còn của `_ground_term`. Đo được:
tỉ lệ cặp đúng có mốc chung **66.7% → 74.7%**, trần recall của ràng buộc hình học
**60.2% → 66.5%**, trong khi cặp SAI lọt lưới gần như không tăng (4.9% → 5.6%). Bão hoà ở 256
trên WildTrack (tracklet dài nhất ~400 khung). 256 điểm ≈ 6 KB/tracklet — cái giá đã trả bằng
recall lớn hơn nhiều lần cái tiết kiệm được. Nới trần **không** đổi tiêu chí gán, chỉ thêm
bằng chứng, nên đây không phải khớp tham số theo dataset.

**7. Có quỹ đạo đầy đủ thì ngưỡng hình học phải SIẾT lại: `max_ground_dist_m` 5.0 → 2.0.**
Giá trị 5.0 của phiên 12 là giá trị đúng *khi quỹ đạo bị tỉa*: ít mốc chung thì trung vị
d_ground bị vài mốc xấu chi phối nên phải nới. Với quỹ đạo đầy đủ, quét lại: 1.0 → F1 0.206,
1.5 → 0.253, **2.0 → 0.256**, 2.5 → 0.253, 3.0 → 0.248, 5.0 → 0.240. Đỉnh **phẳng** quanh
1.5–2.5 nên 2.0 không phải một điểm khớp riêng cho WildTrack. Chỉ sửa trong
`configs/demo/wildtrack_ds.mct.yaml`; `configs/mct.yaml` giữ `max_ground_dist_m: 1.0` vì
con số đó đặt theo **sai số hiệu chỉnh homography** của hệ thống thật, không phải theo dataset
mượn — nhưng `ground_path_max_points: 256` thì áp cả hai (lý do ở quyết định 6 là cơ chế, không
phụ thuộc dataset).

**8. `configs/demo/wildtrack.mct.yaml` (cấu hình cận trên) giữ nguyên 64.** Nới lên 256 ở đó
cho F1 0.765 → 0.771, nhưng file đó tồn tại để **tái lập** số đo phiên 5; đổi giá trị là mất
khả năng đối chiếu. Ghi con số vào đây là đủ.

## Số liệu đo được

**Cấu hình chung:** fixture `ds_wildtrack_7cam.jsonl` (WildTrack 7 camera, 2 fps, 400
khung/camera, 2800 message — hộp YOLO11s, `local_track_id` của NvDCF, embedding ReID của
nvtracker). `min_frames=3`, `idle_timeout_ms=30000`, `max_cost=0.80`, `ground_gap_policy=reject`,
homography 7 camera đã hiệu chỉnh. Chạy trên head node `ut-hpc` (Python 3.10.12,
`~/mct/venv-test`), mỗi lượt `diagnose_tracklets` ~2.6 s, mỗi lượt `eval_wildtrack` ~15 s,
mỗi vòng online `python -m mct` ~54 s.

### Quét bộ lọc điểm chân (`max_ground_dist_m=5.0`, quỹ đạo trần 64 điểm)

| chế độ | cửa sổ | d_ground cặp đúng p25 / p50 / p75 (m) | mốc chung | trần recall (reject) |
|---|---|---|---|---|
| **none** | — | 0.48 / **0.94** / 2.55 | 66.7% | **60.2%** |
| median | 3 | 0.46 / 0.93 / 2.50 | 66.7% | — |
| median | 5 | 0.44 / 0.98 / 2.38 | 66.7% | — |
| median | 9 | 0.44 / **0.86** / 2.46 | 66.7% | 59.6% |
| mean | 5 | 0.45 / 0.97 / 2.46 | 66.7% | — |
| linear | 9 | 0.46 / 0.90 / 2.51 | 66.7% | 60.2% |

Cặp **khác** người không đổi ở mọi cấu hình (p50 = 9.0–9.1 m), tức bộ lọc không hề làm tăng
khả năng phân biệt. F1 end-to-end (offline): none **0.224**, median(9) 0.229, linear(9) 0.195.

### Tách sai số điểm chân của detector (`diagnose_foot_error`, 633 tracklet, 18.943 detection)

| | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| \|Δ\| từng detection (m) | 0.09 | 0.17 | 0.29 | 0.55 | 1.43 |
| \|độ chệch\| theo tracklet (m) | 0.05 | 0.12 | 0.25 | 0.52 | 1.07 |
| nhiễu RMS theo tracklet (m) | 0.13 | 0.21 | 0.43 | 0.73 | 1.15 |

Phân rã năng lượng `E|Δ|² = |độ chệch|² + E|nhiễu|²`: RMS tổng **0.768 m** = độ chệch
**0.461 m (36.0%)** + nhiễu **0.614 m (64.0%)**. Bộ lọc thời gian chỉ với tới phần 64% —
mà phần đó đã bị phép trung vị theo mốc chung khử trước rồi.

### Nới trần quỹ đạo (`ground_path_max_points`)

| trần | mốc chung (cặp đúng) | d_ground p50 (m) | trần recall reject | lọt lưới (cặp sai) |
|---|---|---|---|---|
| **64** | 66.7% | 0.94 | 60.2% | 4.9% |
| 128 | 73.4% | 0.82 | 65.3% | 5.5% |
| **256** | **74.7%** | **0.81** | **66.5%** | 5.6% |
| 1024 | 74.7% | 0.82 | 66.5% | 5.6% |

### F1 offline sau khi nới trần (max_points 256, quét `max_ground_dist_m`)

| d_max (m) | P | R | F1 | | P (median-9) | F1 (median-9) |
|---|---|---|---|---|---|---|
| 1.0 | 0.321 | 0.151 | 0.206 | | 0.369 | 0.230 |
| 1.5 | 0.435 | 0.178 | 0.253 | | — | — |
| **2.0** | **0.467** | **0.176** | **0.256** | | 0.331 | 0.235 |
| 2.5 | 0.451 | 0.176 | 0.253 | | — | — |
| 3.0 | 0.431 | 0.174 | 0.248 | | 0.275 | 0.201 |
| 5.0 | 0.415 | 0.169 | 0.240 | | 0.279 | 0.202 |

Mốc so sánh: trần 64 + d_max 5.0 (cấu hình phiên 12–15) = P 0.417 / R 0.153 / **F1 0.224**.

### Chấm lại bằng TrackEval — đường online, `--split mct`

Cùng lệnh và cùng ground-truth như phiên 15 (`--gt-fixture wildtrack_7cam.jsonl`, 42.606 hộp,
313 danh tính, MotChallenge2DBox, IoU 0.5, `DO_PREPROC=False`):

| | phiên 15 (trần 64, d_max 5.0) | phiên 16 (trần 256, d_max 2.0) |
|---|---|---|
| **HOTA** | 14.245 | **14.374** |
| DetA / AssA | 24.108 / 8.609 | 24.117 / **8.752** |
| **IDF1** (IDR / IDP) | 17.312 (15.66 / 19.35) | **17.514** (15.84 / 19.58) |
| MOTA / MOTP | 2.664 / 69.319 | 2.659 / 69.319 |
| IDSW | 2526 | 2528 |
| Global ID sinh ra / GT | 295 / 313 | 297 / 313 |

Con số cũ **tái lập chính xác** (14.245 / 8.6092 / 17.312) trước khi đo bản mới — nếu không
thì chênh lệch 0.13 điểm HOTA chẳng nói được gì.

### Đọc ra được gì

**F1 tự chế PHÓNG ĐẠI mức cải thiện so với HOTA: +14% F1 nhưng chỉ +0.9% HOTA.** Lý do là hai
thước đo có đơn vị khác nhau: F1 của `eval_wildtrack` đếm **cặp tracklet**, còn HOTA cân theo
**detection**. Nới trần quỹ đạo giúp đúng những tracklet DÀI (chỉ chúng mới bị tỉa), mà số cặp
tracklet dài thì ít so với tổng, còn HOTA lại bị DetA (recall detector 44.9%) đè xuống. Từ
phiên này, **mọi cải tiến của `src/mct` phải báo cáo kèm HOTA/AssA**, F1 tự chế chỉ dùng để
quét nhanh tham số.

Nhật ký gán cũng cho thấy đúng cơ chế: số Global ID tạo mới vì "bị ngưỡng loại" giảm
**210 → 86**, còn vì "hết ứng viên" tăng **68 → 190**. Ràng buộc hình học giờ loại sớm hơn
(ở bước lọc ứng viên) thay vì để ngoại hình quyết định — đúng ý đồ của việc siết `d_max`.

## Vướng mắc / chưa xong

- **AssA vẫn 8.75** — chênh lệch với cận trên (AssA 90.2) hầu như không đổi. Việc lớn nhất
  còn lại vẫn là **ghép tracklet trong cùng camera**, và phiên này không chạm vào nó.
- Giả định "bộ lọc điểm chân vô dụng" chỉ đúng ở **2 fps**. Ở 25 fps chưa đo; khoá YAML đã
  có sẵn để đo lại ở M6 mà không phải viết thêm code.
- `ground_path_max_points` đúng cho WildTrack là 256, nhưng con số đó là hàm của
  **fps × thời lượng tracklet**: ở 25 fps một tracklet 30 s cần ~750 điểm mới không bị tỉa.
  Chưa có cách tự đặt trần theo dữ liệu — hiện vẫn là hằng số trong YAML.
- Chưa chấm `onnx_gtbox` bằng HOTA (nợ từ phiên 15) — sẽ quy đổi mức +22% F1 của hộp GT sang
  điểm HOTA.
- Mọi con số vẫn trên WildTrack 2 fps, mọi camera chồng lấn. **Đừng chốt tham số nào theo
  dataset này** — nhắc lại lần thứ năm.

## Bước tiếp theo

1. **Ghép tracklet trong cùng camera** (`src/mct/`): AssA 8.75 nói đây là chỗ mất điểm lớn
   nhất, và phiên 13 đã có ba lý do độc lập ủng hộ (hình học, ngoại hình, ít mảnh vụn hơn).
   Chấm bằng `--split mct` (HOTA/AssA), không chỉ F1.
2. Chấm `onnx_gtbox` bằng HOTA để quy đổi ảnh hưởng của chất lượng hộp.
3. Nếu còn thời gian: xem lại `min_frames`/`topk_query` bằng đúng bộ thước đo chuẩn — hai
   tham số này chưa bao giờ được quét bằng HOTA.
