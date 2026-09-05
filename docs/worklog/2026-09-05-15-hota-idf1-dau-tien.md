# 2026-09-05 (phiên 15) — Bộ HOTA/IDF1/MOTA đầu tiên: 94.7 ở cận trên, 14.2 trên pipeline thật

- **Mốc:** M6 (đánh giá) | **Máy:** máy dev (soạn) + `ut-hpc` (chạy) | **Thời lượng:** ~1.5h, **không tốn GPU**

## Mục tiêu phiên

- Đi trọn đường `python -m mct` → `export_trackeval` → `run_trackeval` lần đầu tiên: tới
  giờ đồ án mới chỉ có F1 theo cặp tracklet, một chỉ số tự chế không so được với công bố nào.
- Lấy số cho cả hai đầu: fixture bbox ground-truth (cận trên) và fixture DeepStream thật.

## Đã làm

- Cài TrackEval trên head node `ut-hpc` vào venv riêng `~/mct/venv-eval` (numpy 1.23.5 +
  scipy 1.10.1), tách khỏi `venv-test` để venv chạy pytest giữ nguyên độ nhẹ.
- **`configs/demo/wildtrack_ds.mct.yaml`** (mới) — tham số phiên 12/13 mới chỉ nằm trong
  comment và cờ dòng lệnh của script đánh giá, mà `python -m mct` thì chỉ đọc ngưỡng từ
  YAML. Không có file này thì số F1 0.224 không tái lập được bằng vòng chạy online.
- **`--gt-fixture` cho `export_trackeval`** (`GtSource`) — ground-truth lấy từ fixture CHÚ
  THÍCH thay vì từ chính detection của kết quả. Đây là thay đổi thiết kế, không phải sửa
  lặt vặt: xem quyết định 1.
- `frame_offset_for`: bội của 100000 → luỹ thừa 10 nhỏ nhất lớn hơn camera dài nhất.
- `run_trackeval.py`: tắt `PLOT_CURVES`, ghi lại cách cài đã chạy được (kèm lý do ghim numpy).
- Chạy trọn ba lượt chấm: `sct` và `mct` trên fixture DeepStream, `mct` trên fixture bbox GT.
- **434 passed, 8 skipped**, ruff sạch. Bốn test mới khoá đường `--gt-fixture` và offset.

## Quyết định kỹ thuật

**1. Ground-truth phải đến từ nguồn ĐỘC LẬP với kết quả.** Bản đầu dựng GT bằng cách lọc
chính detection của kết quả qua bảng `.gt.json`. Hai hệ quả, cái sau mới là cái giết chết
phép đo:

- Hình học hai phía trùng khít nhau, nên MOTP và phần Det của HOTA chỉ nói về chính nó.
- TrackEval **từ chối chấm**: `Ground-truth has the same ID more than once in a single
  timestep (frame 78, ids: 147)`. Vì bảng của `ds_wildtrack_gt.py` sinh bằng ghép IoU với
  đầu ra tracker, một người bị NvDCF chẻ thành hai track thì cả hai cùng mang một
  `gt_global_id` — và MOT Challenge cấm một danh tính có hai hộp trong một khung.

Không "sửa" bằng cách khử trùng lặp: hai track cho một người **là** lỗi tracking, và cách
tính đúng là để chúng cạnh tranh nhau, một cái khớp, cái kia thành FP. Nên `--gt-fixture`
nhận thẳng fixture chú thích (`wildtrack_7cam.jsonl`: hộp WildTrack, `local_track_id` =
`personID`), kiểm được là **42.606 hộp, 0 trùng lặp (cam, khung, người)**. Từ đó TrackEval
tự ghép hai phía bằng IoU đúng như MOT Challenge làm — và con số mới có nghĩa đối chiếu.

Đường cũ vẫn giữ (có lúc cần xuất để soi bằng mắt) nhưng giờ in cảnh báo.

**2. Offset của chuỗi ảo phải bám sát độ dài thật.** `frame_offset_for` làm tròn lên bội của
100000 "cho dễ đọc log", nhưng TrackEval duyệt **mọi** timestep của chuỗi, kể cả khối rỗng:
2800 khung thật thành 700.000 timestep, một lần chấm ~59 s. Đổi sang luỹ thừa 10 nhỏ nhất
lớn hơn camera dài nhất (400 khung → 1000): vẫn đọc log không phải nhẩm (khung ảo 3001 =
camera thứ 3, khung 1), nhưng còn **3.0 s**. Nhanh gấp ~20 lần, cùng kết quả.

**3. Ghim `numpy==1.23.5` thay vì vá bản clone TrackEval.** TrackEval còn dùng `np.float`,
bị xoá hẳn từ numpy 1.24 → nổ `AttributeError` ngay lúc nạp file GT. Vá bản clone thì ai
tái lập cũng phải vá lại y hệt; ghim phiên bản thì chỉ cần chép đúng ba dòng lệnh trong
docstring của `run_trackeval.py`. Cũng không cần `requirements.txt` đầy đủ — numpy + scipy
là đủ cho MotChallenge2DBox.

**4. `PLOT_CURVES=False`.** TrackEval vẽ đường cong HOTA bằng matplotlib **sau khi** đã tính
xong, và `ModuleNotFoundError` ở đó nuốt sạch kết quả vừa tính (đã mất một lượt chạy vì
đúng chuyện này). Số nằm trong CSV rồi; tắt hẳn thay vì kéo matplotlib vào venv.

## Số liệu đo được

**Cấu hình chung:** WildTrack 7 camera, 400 khung/camera ở 2 fps, 2800 message. Ground-truth
= chú thích WildTrack (42.606 hộp, 313 danh tính). TrackEval MotChallenge2DBox,
`DO_PREPROC=False`, ngưỡng IoU 0.5. Chấm trên head node `ut-hpc`, 3.0–3.2 s mỗi lượt.

### Xuyên camera (`mct`) — chuỗi ảo, `id` = Global ID

| | **cận trên (SCT lý tưởng)** | **pipeline DeepStream thật** |
|---|---|---|
| fixture | `wildtrack_7cam` (bbox GT) | `ds_wildtrack_7cam` (YOLO11s + NvDCF) |
| config | `wildtrack.mct.yaml` | `wildtrack_ds.mct.yaml` |
| **HOTA** | **94.736** | **14.245** |
| DetA / AssA | 99.542 / 90.162 | 24.108 / 8.609 |
| **IDF1** (IDR / IDP) | **93.980** (93.8 / 94.2) | **17.312** (15.7 / 19.4) |
| **MOTA** / MOTP | **99.272** / 100 | **2.664** / 69.319 |
| IDSW / Frag | 127 / 1414 | 2526 / 2991 |
| Global ID sinh ra / GT | 321 / 313 | 295 / 313 |

Cột cận trên phải đọc kèm cảnh báo: fixture đó **chính là** nguồn ground-truth, nên hộp hai
phía trùng khít và `MOTA 99.3 / DetA 99.5` là hệ quả của cách dựng, không phải thành tích.
Chỉ **AssA 90.2** và **IDF1 94.0** mang thông tin: đó là chất lượng liên kết khi tracking
đơn camera hoàn hảo. Khớp với F1 0.929 đo bằng chỉ số tự chế ở phiên 5.

### Đơn camera (`sct`) — `id` = `local_track_id` của NvDCF, chấm với chú thích

| | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|---|
| **COMBINED (7 cam)** | **28.336** | 24.315 | 33.928 | 4.481 | **36.001** | 1770 |
| cam07 (tốt nhất) | 41.283 | 31.639 | 54.368 | 3.297 | 54.927 | 63 |
| cam06 | 22.732 | 19.098 | 27.748 | 9.832 | 30.324 | 240 |
| cam04 (tệ nhất) | 23.274 | 14.494 | 37.474 | −108.9 | 26.609 | 85 |

`CLR_Re` của cả bảy camera = **44.9%**, trùng đúng con số recall của detector đo độc lập ở
phiên 11 (44.9%) bằng một đường code hoàn toàn khác. Một phép kiểm chéo tự nhiên cho cả hai
lần đo.

MOTA âm nặng ở cam04/cam05 là do **FP nhiều hơn GT**: cam04 có 2239 hộp chú thích nhưng
4443 detection. MOTA phạt FP theo số tuyệt đối nên tụt qua 0 rất nhanh; HOTA (23.3) không có
tính chất đó và là lý do nên báo cáo HOTA làm chỉ số chính.

### Đọc ra được gì

**DetA gần như không đổi giữa hai chế độ (24.315 đơn camera → 24.108 xuyên camera) trong khi
AssA rơi 33.928 → 8.609.** Cùng một tập detection, nên toàn bộ chênh lệch HOTA 28.3 → 14.2
là **giá của bước liên kết xuyên camera**, không phải của detector. Đây là cách tách biến
sạch nhất mà đồ án có được tới giờ, và nó chỉ thẳng vào chỗ cần cải thiện.

## Vướng mắc / chưa xong

- Cận trên và thực tế chênh nhau **6.6 lần HOTA** (94.7 → 14.2). Ba nguyên nhân đã định
  lượng ở phiên 11–13 (recall detector 44.9%, tracklet vỡ, ngoại hình xuyên camera yếu) đều
  còn nguyên; phiên này chỉ đổi thước đo chứ chưa sửa gì trong `src/mct`.
- `export_sct` đường **không** có `--gt-fixture` vẫn còn đó và vẫn cho ra số vô nghĩa nếu ai
  đó gọi nhầm. Có cảnh báo, nhưng cảnh báo thì đọc log mới thấy.
- Chưa chấm fixture `onnx_gtbox`/`onnx_detbox` của phiên 13 bằng HOTA — sẽ cho biết mức
  +22% F1 của hộp GT quy ra bao nhiêu điểm HOTA.
- Mọi con số vẫn trên WildTrack 2 fps, mọi camera chồng lấn. **Đừng chốt tham số nào theo
  dataset này** — nhắc lại lần thứ tư.
- `~/mct/venv-eval` và `~/TrackEval` chỉ tồn tại trên `ut-hpc`, không vào git. Cách dựng lại
  nằm trong docstring `eval/run_trackeval.py`.

## Bước tiếp theo

1. Quay lại việc M4 của phiên 13, giờ đã có thước đo chuẩn để chấm điểm cải tiến: làm mượt
   điểm chân trong `src/mct/tracklet.py`, đo lại bằng chính `--split mct` (HOTA/AssA), không
   chỉ bằng F1 tự chế.
2. Ghép tracklet trong cùng camera — AssA 8.6 nói rằng đây là chỗ mất điểm lớn nhất.
3. Chấm `onnx_gtbox` để quy đổi ảnh hưởng của chất lượng hộp sang HOTA.
