# 2026-09-20 (phiên 26): Chấm điểm trên mặt phẳng mặt đất theo giao thức của các bài WildTrack. Chỉ tính trong vùng chú thích, HOTA vẫn giảm theo độ phân giải (31.1 → 30.2 → 28.2), nên giữ 640

- **Mốc:** M6 (đánh giá) | **Máy:** máy dev (CPU, không thuê GPU) | **Thời lượng:** ~1h | **Ghi bù ngày 2026-09-28**

> Phiên này làm ngay sau phiên 25 (các file sinh ra lúc 16:30–17:05 ngày 2026-09-20) nhưng
> worklog viết bù 8 ngày sau, dựng lại từ code, test và `data/s25/*.json`. Phần "vì sao" dưới
> đây lấy từ docstring của công cụ; lý do nào không có ở đó thì được ghi là suy luận.

## Mục tiêu phiên

- Trả lời bước 1 của phiên 25: khi chỉ chấm trong vùng WildTrack chú thích, độ phân giải
  detector cao hơn có giúp không?
- Trả lời câu hỏi: vì sao các bài WildTrack báo 90+ còn mình báo HOTA 15. Muốn so thì phải
  chấm bằng cùng giao thức với họ trước đã.

## Đã làm

**Công cụ mới: `eval/eval_ground_plane.py` (+11 test, `tests/test_eval_ground_plane.py`).**
Chấm lại đúng các lần chạy engine có sẵn theo giao thức của MVDet / EarlyBird / TrackTacular /
MVTrajecter / MCBLT: người là một ĐIỂM trên mặt đất, khớp trong bán kính mét, chỉ tính trong
lưới 12 m × 36 m.

- Dự đoán: mỗi `(khung, Global ID)` cho một điểm. Chân hộp (đáy-giữa) của từng camera được
  chiếu qua homography mặt đất, rồi lấy trung vị theo từng toạ độ trên các camera cùng thấy
  Global ID đó trong khung đó. Hộp chưa được engine gán Global ID thì tính là bỏ sót, như
  các công cụ khác.
- Ground truth: mỗi `(khung, personID)` cho một điểm, lấy từ `positionID` (qua
  `tools.wildtrack_to_fixture.position_id_to_world_m`).
- Độ tương đồng là `max(0, 1 − d / 2T)`, nên α = 0.5 ứng với khoảng cách T mét. CLEAR/Identity
  (ngưỡng 0.5) vì vậy khớp trong T mét. HOTA báo cả trung bình α 0.05..0.95 lẫn HOTA@0.5.
- Chỉ số tính bằng chính các lớp `HOTA`/`CLEAR`/`Identity` của TrackEval (venv `mct-eval`) trên
  một chuỗi dựng tay. Không viết lại công thức nào.
- Tuỳ chọn: `--area in|all|both`, khung `all` (400) / `test` (40 khung cuối, tập test của các
  bài báo), `--threshold-m`, `--nms-m`, `--self-check`. Chế độ `oracle` lấy Global ID =
  `local_track_id` để kiểm hình học khi liên kết hoàn hảo.
- `--self-check` nạp lại GT làm dự đoán: khớp hoàn toàn thì phải > 99.9; dời 40 m thì TP phải
  = 0; dời 0.3 m với T = 1 m thì phải > 99. Độ dời được chọn theo mật độ đám đông WildTrack
  (người đứng gần nhau hơn 1.5 m, nên dời 1.5 m vẫn khớp nhầm hàng xóm).

**Chạy:** 9 lần chạy của phiên 25 (`data/s25/R{640,960,1280}_r{1,2,3}/mct.db` +
`data/fixtures/ds_wildtrack_7cam_{r640n,r960,r1280}_r*.jsonl`) → `data/s25/ground_plane.json`.
Kiểm hình học trên B và LB lần r1 của phiên 24 (`data/s24/{B,LB}_r1/mct.db`) →
`data/s25/gp_sanity_{B,LB}_r1.json`.

**Chưa chạy: NMS mặt đất.** `ground_nms` (`--nms-m`, các bài báo dùng 0.5 m) được thêm SAU
lần chạy cuối. Dấu hiệu: `ground_plane.json` ghi lúc 17:03, code lúc 17:05, và khoá trong
JSON không có đoạn `|nms…|` mà code hiện tại sinh ra. Mọi số dưới đây là **không NMS**, và
chạy lại với `--nms-m 0` thì khoá sẽ khác tên (thêm `|nms0|`).

Không sửa `src/` hay `configs/`.

## Quyết định kỹ thuật

**1. Chấm bằng giao thức mặt đất của giới nghiên cứu, bên cạnh (không thay) giao thức hộp ảnh.**
Phiên 15–25 chấm theo hộp ảnh, IoU 0.5, trên chuỗi ảo 7 camera ghép lại. Các bài WildTrack thì
chấm theo điểm mặt đất, trong vùng chú thích. Hai con số không so được với nhau, nên muốn đặt
kết quả đồ án cạnh các bài báo thì phải có số theo đúng giao thức của họ. Giao thức hộp ảnh vẫn
giữ, vì nó tách được lỗi theo từng camera (phiên 24 dựa vào nó).

**2. Gộp đa camera bằng trung vị từng toạ độ, không dùng trung bình.** Hộp bị cắt ở mép dưới ảnh
hoặc che khuất cho chân lệch nhiều mét. Trung vị chịu được một camera sai trong 3+ camera, trung
bình thì không. *(Suy luận, không có trong docstring.)*

**3. Không học gì trên WildTrack, và ghi rõ chỗ khác với các bài báo.** Các bài đó gộp 7 view
TRONG một mạng. Engine này phát hiện riêng từng camera ở IoU 2-D rồi mới liên kết. Số đo ở đây
nói về cùng một hệ thống như các phiên trước, chỉ đổi thước đo. Nó không phải một cấu hình mới
để đi đua bảng xếp hạng.

**4. Bước 1 của phiên 25 làm theo cách khác kế hoạch.** Kế hoạch là thêm cờ lọc vùng vào
`tools/export_trackeval.py` rồi chấm hộp ảnh chỉ trong vùng. Thay vào đó, phiên này lọc vùng
trên mặt đất. Cả hai cùng trả lời câu "bỏ phần ngoài vùng thì độ phân giải cao có giúp không",
nhưng **HOTA hộp ảnh trong vùng vẫn chưa được đo**.

**5. Quyết định cấu hình: giữ YOLO11s 640, không thuê GPU để quét ngưỡng ở 960.** Đây là tiêu chí
đã đặt trước ở bước 2 của phiên 25: "HOTA trong vùng đi ngang hoặc giảm thì giữ 640". Kết quả
ở mục 1 đáp ứng tiêu chí đó (960 ngang, 1280 giảm rõ). Điều kiện đi kèm: kết luận có thể đổi
nếu NMS mặt đất gỡ được phần lớn FP do trùng Global ID (xem "chưa xong").

## Số liệu đo được

**Cấu hình.** Dùng lại 9 lần chạy của phiên 25: WildTrack 7 camera, 400 khung, 2 fps, T4,
DeepStream 7.1, YOLO11s COCO FP16 ở 640/960/1280, `pre-cluster-threshold` 0.25, NvDCF + ReID OSNet
`osnet_x1_0_msdc_dg`, engine liên kết `configs/demo/wildtrack_ds.mct.yaml`. Chấm trên máy dev bằng
TrackEval (venv `mct-eval`), homography `configs/cameras/homography/wildtrack`, không NMS.
GT: 9 518 điểm, 313 danh tính (toàn bộ); 952 điểm, 41 danh tính (40 khung test).
Trung bình ± độ lệch chuẩn mẫu, n = 3.

### 1. Trong vùng chú thích, 400 khung, T = 1 m

| cạnh vào | HOTA | HOTA@0.5 | DetA | AssA | IDF1 | MOTA | TP | FP | FN | Global ID |
|---|---|---|---|---|---|---|---|---|---|---|
| **640** | **31.10 ± 0.85** | 34.43 ± 0.98 | 31.06 ± 0.49 | 31.64 ± 1.31 | 35.40 ± 1.22 | −41.4 | 7 069 | 9 916 | 2 449 | 324 |
| **960** | **30.20 ± 1.05** | 33.69 ± 0.95 | 28.40 ± 0.75 | 32.72 ± 1.55 | 32.77 ± 0.64 | −66.1 | 7 344 | 12 373 | 2 174 | 377 |
| **1280** | **28.16 ± 1.19** | 31.43 ± 1.55 | 26.19 ± 0.51 | 30.86 ± 2.88 | 31.17 ± 0.30 | −93.5 | 7 706 | 15 156 | 1 812 | 428 |

Welch thô so với 640: 960 ΔHOTA −0.90, t ≈ 1.2 (**trong nhiễu**); 1280 ΔHOTA −2.94, t ≈ 3.5
(tách khỏi nhiễu). AssA gần như không đổi, và **toàn bộ phần giảm nằm ở DetA**.

Recall/precision điểm (tự tính từ TP/FP/FN): recall 74.3% → 77.2% → 81.0%, precision
41.6% → 37.2% → 33.7%, **F1 0.533 → 0.502 → 0.476**. F1 giảm, ngược chiều với F1 hộp ảnh trong
vùng của phiên 25 (0.565 → 0.596 → 0.608).

### 2. Các biến thể (HOTA, n = 3)

| cạnh vào | trong vùng, T = 1 m | trong vùng, T = 0.5 m | toàn khung (không lọc vùng), T = 1 m | trong vùng, **40 khung test**, T = 1 m |
|---|---|---|---|---|
| 640 | 31.10 ± 0.85 | 24.85 ± 0.75 | 25.07 ± 0.74 | 41.39 ± 1.10 |
| 960 | 30.20 ± 1.05 | 23.89 ± 1.37 | 21.17 ± 0.50 | 37.00 ± 0.56 |
| 1280 | 28.16 ± 1.19 | 22.23 ± 0.89 | 18.22 ± 1.28 | 34.07 ± 1.42 |

Lọc vùng gỡ được nhiều hơn khi độ phân giải càng cao (+6.0 / +9.0 / +9.9 HOTA). Điều đó khớp với
phiên 25: hộp "sai" thêm phần lớn nằm ngoài vùng. Nhưng lọc xong thì thứ tự vẫn không đổi.

### 3. Kiểm hình học: các kịch bản của phiên 24, lần r1, trong vùng, 400 khung, T = 1 m

| kịch bản (phiên 24) | HOTA mặt đất | DetA | AssA | IDF1 | TP | FP | FN | HOTA hộp ảnh (phiên 24) |
|---|---|---|---|---|---|---|---|---|
| A = pipeline thật, 640 (lấy từ mục 1) | 31.10 | 31.06 | 31.64 | 35.40 | 7 069 | 9 916 | 2 449 | 15.70 |
| B = hộp GT + id GT + engine liên kết | 63.79 | 58.41 | 69.72 | 66.11 | 7 409 | 2 955 | 2 109 | 39.15 |
| LB = B + liên kết hoàn hảo | 83.06 | 80.02 | 86.22 | 89.34 | 7 734 | 58 | 1 784 | 49.31 |

LB trên 40 khung test: HOTA 94.17, IDF1 96.59. Toàn khung và trong vùng gần như trùng nhau
(83.10 vs 83.06), đúng như kỳ vọng vì hộp GT luôn nằm trong vùng.

Đọc bảng này thế nào:
- **Hình học của công cụ đúng.** LB có FP = 58 trên 7 792 điểm (IDP 99.2%): chiếu chân hộp qua
  homography, gộp bằng trung vị rồi khớp trong 1 m cho lại đúng vị trí chú thích.
- **Gộp đa view nâng recall mạnh.** Tập hộp của LB chỉ phủ ~45% hộp GT ảnh (DetA hộp 45.14),
  nhưng khi đo theo điểm thì phủ **81.3%** người-khung (IDR). Một người chỉ cần một camera thấy
  là đủ.
- 18.7% FN còn lại của LB là trần của tập detection này, không phải lỗi hình học.

## Vướng mắc / chưa xong

- **Chưa chạy NMS mặt đất.** Engine hay tách một người thành hai Global ID cùng lúc (phiên 20:
  43.7% khung mất vì vỡ). Khi đó mỗi Global ID thêm một điểm, và theo giao thức điểm thì điểm
  thứ hai là FP. Ngay trong vùng, precision chỉ 34–42%, nên một phần FP ở mục 1 có thể là trùng
  lặp chứ không phải hộp sai. Chưa đo được phần đó bao nhiêu, và **chưa biết** nó có đảo thứ tự
  640/960/1280 không. Code đã có sẵn (`--nms-m 0.5`), chỉ cần chạy.
- HOTA hộp ảnh chỉ trong vùng (kế hoạch gốc của bước 1, phiên 25) vẫn chưa đo.
- Kiểm hình học B/LB mới chạy lần r1 (n = 1), B chỉ có biến thể trong vùng. Chưa chạy A theo
  kiểu oracle (`db = oracle`).
- Số mặt đất vẫn **không so thẳng được** với các bài báo: họ gộp view trong mạng, có NMS, và
  học trên chính WildTrack. Muốn đặt cạnh nhau trong chương 6 thì phải ghi rõ ba điểm khác biệt đó.
- CLAUDE.md §7 chưa cập nhật (bước 3 của phiên 25): chưa ghi rằng số WildTrack toàn khung đánh
  giá thấp detector, và giờ còn có thêm giao thức mặt đất.

## Bước tiếp theo

1. Chạy lại `eval.eval_ground_plane` cho 9 lần chạy với `--nms-m 0 0.5` (CPU, venv `mct-eval`,
   không cần engine hay GPU). Nếu NMS 0.5 m đảo thứ tự 640/960/1280 thì mở lại QĐ 5.
   ```
   PYTHONPATH=src ~/.venvs/mct-eval/Scripts/python.exe -m eval.eval_ground_plane \
       --trackeval-path ~/TrackEval --wildtrack-dir data/wildtrack \
       --homography-dir configs/cameras/homography/wildtrack --nms-m 0 0.5 \
       --run R640:r1 data/fixtures/ds_wildtrack_7cam_r640n_r1.jsonl data/s25/R640_r1/mct.db \
       ... --json data/s25/ground_plane_nms.json
   ```
2. Cập nhật CLAUDE.md §7: có hai giao thức chấm (hộp ảnh IoU 0.5 và điểm mặt đất 1 m trong vùng),
   và mọi số WildTrack phải ghi kèm giao thức cùng cách tính vùng.
3. Chương 6: dùng bảng A/B/LB ở mục 3 làm "thang" theo giao thức mặt đất
   (31 → 64 → 83 HOTA), song song với thang hộp ảnh của phiên 24.
