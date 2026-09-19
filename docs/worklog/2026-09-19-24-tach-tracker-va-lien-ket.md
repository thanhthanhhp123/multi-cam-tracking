# 2026-09-19 (phiên 24) — Tách lỗi AssA giữa tracker NvDCF và module liên kết: **nửa – nửa**, không có nút thắt đơn lẻ

- **Mốc:** M4 + M6 (đánh giá) | **Máy:** máy dev (Windows, venv 3.10; GTX 1660 SUPER không dùng) | **Thời lượng:** ~4h (phần lớn là CPU chạy nền) | **GPU thuê: $0**

## Mục tiêu phiên

- HOTA mới tách được hai tầng (detector vs "phần còn lại"). Cần tách "phần còn lại" làm hai:
  bao nhiêu AssA mất vì **NvDCF vỡ/trộn tracklet trong một camera**, bao nhiêu vì **module liên
  kết đa camera** (Re-ID + Hungarian + homography). **Chỉ đo, không sửa logic tracking/liên kết.**
- Kịch bản yêu cầu: A (pipeline thật), C (hộp GT + id NvDCF), B mới = **oracle tracker**
  (hộp GT + id GT, Re-ID và liên kết vẫn thật).

**Sửa một tiền đề của đề bài.** Đề bài viết "HOTA 25.18 ≈ AssA 6.34 khi DetA ≈ 100". Không
đúng: 6.34 = 25.18²/100 chỉ đúng nếu DetA = 100. Kịch bản C thật có **DetA 43 (DetRe 43%, vì
chỉ giữ detection khớp GT), AssA 14.8** (phiên 21) — đo lại phiên này: DetA 42.94 ± 0.26,
AssA 15.92 ± 0.44. Chưa từng có kịch bản nào với DetA ≈ 100.

## Đã làm

**Công cụ**
- `src/tools/reembed_fixture.py` — thêm (1) **`relabel_with_gt_ids`** + cờ `--oracle-out`:
  ghi thêm một fixture cùng hộp / embedding / `ts_ms` với `--out` nhưng `local_track_id` =
  `personID` WildTrack (một lượt trích embedding cho cả C lẫn B, nên hai fixture giống nhau
  **từng bit** ngoài cột id); (2) **`EmbeddingCache`** + `--embed-cache` (xem QĐ 5).
- `eval/compare_oracle_tracker.py` (mới) — điều phối engine online → `export_trackeval` →
  TrackEval cho nhiều lượt, gộp trung bình ± độ lệch chuẩn mẫu; `--sct` chấm thêm từng camera;
  `--ceiling` chấm trần liên kết hoàn hảo (không chạy engine). Tái lập lệnh nằm ở docstring.
- Test mới: `tests/test_reembed_fixture.py` (+11: oracle, cache), `tests/test_compare_oracle_tracker.py`
  (11). **579 passed, 5 skipped**, ruff sạch (`~/.venvs/mct-test`, CPython 3.10.20).

**Dữ liệu sinh ra** (đều trong `data/`, gitignored): `data/fixtures/ds_wildtrack_7cam{,_c025_r2,_c025_r3}_onnx_{gtbox,oracle}.jsonl`
(C và B cho 3 lần chạy pipeline), `data/s24/embed_cache_gtbox_osnet_msdc_dg.npz` (40.7 MB),
`data/s24/compare_oracle_tracker.json` (mọi con số), `data/s24/{A,B,C,LB,LC}_r{1,2,3}/` (DB + TrackEval).

**Đã thử rồi bỏ:** DirectML cho ONNX trên GTX 1660 SUPER (QĐ 5).

## Quyết định kỹ thuật

**1. Oracle tracker giữ NGUYÊN tập detection của C, không dùng toàn bộ GT.** B = đúng các
detection mà C có (19 238 hộp GT khớp được ở r1), chỉ đổi `local_track_id` → `personID`. Nhờ
đó C→B chỉ có **một** biến. Phương án bị loại: B_full (toàn bộ 42 606 hộp GT). Nó đổi hai
biến cùng lúc (id *và* độ phủ detection 43% → 100%), và AssA của HOTA phụ thuộc độ phủ
(QĐ 2), nên hiệu C→B_full không quy được về tracker. **B_full chưa chạy** — cần thêm ~22 000
crop ONNX (~75 phút CPU); nếu cần, thêm chỗ cắm cache cho `wildtrack_to_fixture`.

**2. Không đọc AssA của B như "khoảng cách tới 100" — AssA của HOTA phạt cả detection bị bỏ
sót.** Với `FNA` gồm mọi hộp GT của người đó không được ghép đúng id (kể cả hộp detector
không thấy), tập detection phủ 43% thì hệ hoàn hảo tuyệt đối cũng chỉ đạt AssA ≈ 54 (xuyên
camera) / 68 (đơn camera). Bằng chứng: B đơn camera có **0 id-switch mà AssA chỉ 68.34**; và
LB (bên dưới) có `AssPr = 100.00` nhưng `AssRe = AssA = 53.87`. Muốn quy lỗi về đúng chỗ phải
so với **trần của cùng tập detection**, nên thêm hai kịch bản không qua engine
(Global ID = người thật):
- **LB** = id tracker hoàn hảo + liên kết hoàn hảo (trần của tập detection này).
- **LC** = id NvDCF + liên kết hoàn hảo: mỗi track NvDCF nhận personID chiếm đa số. Cái này
  là mức tốt nhất mà *bất kỳ* module liên kết nào đạt được trên tracklet NvDCF (track trộn hai
  người thì liên kết không cứu được).

Bốn góc LB / B / LC / C cho phép tách lỗi theo **cả hai thứ tự** — kết luận không phụ thuộc
"sửa cái nào trước".

**3. LC phải tôn trọng ràng buộc loại trừ.** Lần đầu LC bị TrackEval từ chối ("predicts the
same ID more than once in a single timestep"): hai track NvDCF cùng camera cùng có "người đa
số" X chạy song song. Engine thật cũng bị cấm chuyện này (CLAUDE.md §6.2) nên oracle phải
chịu cùng luật: track nhiều phiếu hơn giữ personID, track trùng khung nhận Global ID mới;
mảnh nối tiếp không trùng khung vẫn được gộp. LC vì vậy hơi **chặt tay** (majority theo
`(camera, id)`, không theo tracklet sau khi cắt bởi `idle_timeout`) — trần thật có thể cao
hơn chút, chưa đo chênh. (Docstring ban đầu tôi ghi "rộng tay" — sai chiều, đã sửa.)

**4. Chạy cả ba lần pipeline (r1–r3), báo trung bình ± độ lệch chuẩn mẫu.** Theo quy ước
phiên 22 (nhiễu giữa các lần chạy NvDCF cỡ hiệu ứng). A tái lập từng chữ số của phiên 22
(HOTA 15.609 / 15.294 / 16.208), xác nhận môi trường chấm điểm không lệch.

**5. Embedding ONNX trên CPU chậm (~0.2 s/crop) nên phải có cache theo (camera, khung, hộp).**
Embedding của một crop không phụ thuộc fixture nào yêu cầu nó, và ba fixture khớp gần như
cùng tập hộp GT. `EmbeddingCache` gắn với **tên model** (đọc cache của model khác = lỗi, không
im lặng dùng nhầm), lưu `.npz` nguyên tử mỗi 200 ảnh. Kết quả: r1 mất **64 phút**, r2 **45 s**,
r3 **21 s**. Cache KHÔNG biết crop cắt bằng code nào — đổi `crop_for_reid` hay tiền xử lý thì
phải xoá file. Phương án bị loại: **DirectML** (`onnxruntime-directml` 1.23.0 trên GTX 1660 SUPER)
— `Reshape` node `node_view` báo `0x80070057` ở mọi mức tối ưu đồ thị; CUDA EP kéo theo cài
cuDNN/CUDA runtime, không đáng cho một phép đo phụ (chưa thử). Batch lớn không giúp
(0.259 / 0.232 / 0.201 / 0.232 s/crop ở batch 1 / 7 / 32 / 64).

## Số liệu đo được

**Cấu hình.** WildTrack 7 camera, 400 khung/camera, 2 fps, 3 lần chạy pipeline độc lập
(fixture `ds_wildtrack_7cam` = r1, `_c025_r2`, `_c025_r3`; phiên 22, `vast-gpu` Tesla T4,
YOLO11s FP16 640, `pre-cluster-threshold` 0.25, NvDCF + ReID OSNet `osnet_x1_0_msdc_dg`
TensorRT, streammux 1920×1080 batch 7). Embedding của B/C: OSNet `osnet_x1_0_msdc_dg.onnx`
chạy ONNX Runtime **CPU** (i5-10400F; gói `onnxruntime-directml` 1.23.0 nhưng ép
`CPUExecutionProvider` — tôi đã đổi gói này vào venv trước khi chạy job; GPU không tham gia)
trên crop hộp GT — **khác đường trích của A** (TensorRT trong nvtracker, crop hộp detector). Engine: `configs/demo/wildtrack_ds.mct.yaml`
+ `wildtrack.topology.yaml` + homography 7 camera, đường online `python -m mct --source`,
**cùng cấu hình cho mọi kịch bản**; chấm: `tools.export_trackeval --mode mct|sct
--gt-fixture wildtrack_7cam.jsonl`, TrackEval MotChallenge2DBox, IoU 0.5, `DO_PREPROC=False`.
Môi trường: máy dev Windows, `mct-test` / `mct-eval` (numpy 1.23.5) / `mct-reid` (mới).

### 1. Xuyên camera (chuỗi ảo 7 camera), trung bình ± độ lệch chuẩn, n = 3

| kịch bản | hộp | id cục bộ | liên kết | HOTA | DetA | **AssA** | AssRe | AssPr | IDF1 | Global ID | #det |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** thật | detector | NvDCF | engine | 15.70 ± 0.46 | 24.12 ± 0.08 | **10.47 ± 0.65** | 13.17 | 34.13 | 20.08 ± 0.97 | 407 ± 13 | 34 523 |
| **C** | GT | NvDCF | engine | 26.14 ± 0.28 | 42.94 ± 0.26 | **15.92 ± 0.44** | 18.45 | 50.24 | 26.76 ± 0.40 | 258 ± 10 | 19 026 |
| **B** oracle tracker | GT | **GT** | engine | 39.15 ± 1.78 | 43.85 ± 0.12 | **35.00 ± 3.05** | 36.01 | 87.65 | 43.31 ± 2.96 | 361 ± 11 | 18 740 |
| **LC** trần | GT | NvDCF | **hoàn hảo** | 37.70 ± 0.24 | 43.53 ± 0.00 | **32.64 ± 0.42** | 34.94 | 69.56 | 44.65 ± 0.26 | 310 | 19 232 |
| **LB** trần | GT | GT | **hoàn hảo** | 49.31 ± 0.02 | 45.14 ± 0.01 | **53.87 ± 0.02** | 53.87 | 100.00 | 62.20 ± 0.01 | 288 | 19 232 |

Từng lần chạy (r1 / r2 / r3) — AssA: A 10.28 / 9.93 / 11.19; C 15.65 / 15.69 / 16.43;
**B 31.58 / 35.98 / 37.43**; LC 32.16 / 32.88 / 32.88; LB 53.90 / 53.86 / 53.86.
Đối chiếu C: phiên 21 đo HOTA 25.18 / AssA 14.81 (một lần, trên fixture đã mất).

### 2. Đơn camera (id = `local_track_id`, không qua `src/mct`; 7 chuỗi, n = 3)

| | HOTA | DetA | AssA | IDF1 | IDSW |
|---|---|---|---|---|---|
| A | 28.10 ± 0.17 | 24.42 | 33.23 ± 0.38 | 35.31 | 1784 ± 7 |
| C (NvDCF trên hộp GT) | 44.95 ± 0.23 | 43.91 | **46.02 ± 0.48** | 44.02 | 1809 ± 8 |
| B (id GT) = trần đơn camera | 55.54 ± 0.01 | 45.14 | **68.34 ± 0.00** | 62.20 | 0 |

### 3. Chẩn đoán trực tiếp NvDCF trên 19 238 detection khớp GT (r1; r2, r3 lệch < 1 điểm)

- **Vỡ:** 726/1153 cặp (camera, người) — **63.0%** — bị chia thành ≥ 2 id NvDCF; 1700 id thừa.
- **Trộn:** 488/755 track — **64.6%** — chứa ≥ 2 người; 79.6% detection nằm trong track trộn;
  **22.8%** detection nằm ngoài người đa số của track của nó.
- 1153 cặp (camera, người) so với 755 track NvDCF: số track *ít hơn* số người-camera, tức trộn
  lấn át vỡ về số đếm.

### 4. Tách lỗi AssA so với trần LB = 53.87 (xuyên camera)

| bước | thứ tự "tracker trước" | thứ tự "liên kết trước" |
|---|---|---|
| lỗi TRACKER | LB → LC: **−21.23** (liên kết không sửa nổi) | B → C: **−19.08** (± ~3) |
| lỗi LIÊN KẾT | LC → C: **−16.72** | LB → B: **−18.87** (± ~3) |
| tổng LB → C | −37.95 | −37.95 |
| **phần của tracker** | **55.9%** | **50.3%** |

Tracker **50–56%**, liên kết **44–50%** phần AssA mất so với trần (trung bình 53% / 47%). Theo
HOTA (cùng cách tách): tracker −11.61 / −13.01, liên kết −11.56 / −10.16, trên tổng
−23.17 từ LB (49.31) xuống C (26.14). **Kiểm chéo độc lập:** tracker mất **22.3 điểm AssA**
ngay trong một camera (trần 68.34 → C 46.02, bảng 2), sát với 21.2 điểm "không ai sửa nổi" ở
LB → LC.

### 5. Toàn cảnh HOTA (từ A tới 100; cộng dồn, không phải tách rời sạch)

| khoản | điểm HOTA | chứng cứ |
|---|---|---|
| detector bỏ sót/không đủ (DetRe 43%) | ≈ 50.7 | 100 → LB 49.31 |
| **tracker NvDCF** | **≈ 12.3** (11.6–13.0) | LB→LC, B→C |
| **module liên kết** | **≈ 10.9** (10.2–11.6) | LB→B, LC→C |
| hộp báo nhầm + hộp lệch + khác đường trích embedding | ≈ 10.4 | C 26.14 → A 15.70 (không tách riêng được) |
| còn lại A | 15.70 | |

### 6. Chi phí hạ tầng

- OSNet ONNX CPU: ~0.2 s/crop (i5-10400F 6C/12T); r1 19 238 crop = 64 phút; cache 40.7 MB.
- Engine online: chỉ có một phép đo thời gian, **2m10s** cho A r1 lúc CPU đang bận trích
  embedding (phiên 22 đo 20–27 s khi CPU rảnh). Thời gian các lượt còn lại không ghi lại.
- Không thuê GPU.

## Vướng mắc / chưa xong

- **Kết luận chỉ có độ tin cậy ~±8 điểm phần trăm ở chỗ chia.** B dao động lớn giữa các lần
  chạy (AssA 31.6 / 36.0 / 37.4) dù đầu vào gần như y nhau (18 740 ± 6 hộp) — engine nhạy với
  thay đổi nhỏ (lỗi gán sớm lan ra theo cách online). Vì vậy "50.3%" của thứ tự thứ hai có
  khoảng ~42–58%; chỉ chắc được rằng **không bên nào chiếm áp đảo**. n = 3 là ít.
- **Đề bài dự đoán "B ≫ C ⇒ nút thắt ở tracker".** Con số đúng chiều đó (35.0 vs 15.9, ×2.2),
  nhưng tiêu chí ấy bỏ qua trần: B mới đạt **65% trần LB** (35.00 / 53.87). Kết luận theo
  trần khác kết luận theo tỉ số thô — nếu dùng tỉ số thô thì hiểu nhầm liên kết là ổn.
- **`same_camera_stitch: true`** trong `wildtrack_ds.mct.yaml` khiến engine đã tự vá một phần
  lỗi vỡ của NvDCF; số "lỗi tracker" ở trên là lỗi **như hệ thống thấy được** sau khi vá,
  không phải lỗi thô của NvDCF (thô tệ hơn). Chưa đo bằng cách tắt cờ đó.
- Cấu hình engine (`max_cost`, `max_ground_dist_m`…) được chỉnh trên fixture A (hộp detector)
  ở phiên 17–21. B/C chạy với cấu hình đó → phần "lỗi liên kết" của B có thể nhỏ hơn nếu chỉnh
  lại cho tracklet sạch; và ngược lại đó là dấu hiệu cấu hình đang ăn khớp với lỗi của tracker.
- **B_full (toàn bộ hộp GT, DetA ≈ 100) chưa chạy** (QĐ 1). Số của B/C/LB/LC đều bị chặn trên
  bởi độ phủ detection 43% — đừng đặt LB 49.31 cạnh "100".
- `min_frames: 3` của engine bỏ ~2.6% detection của B (18 740 / 19 232) vì tracklet ngắn chưa
  được gán Global ID; nó cũng tính vào phần "liên kết" ở trên.
- **Mọi con số là WildTrack 2 fps, mọi camera chồng lấn.** NvDCF ở 2 fps ngoài vùng làm việc
  (thiết kế cho 25–30 fps): mỗi người dịch chuyển cả nửa giây giữa hai khung. Phần 50–56% của
  tracker nhiều khả năng **nhỏ hơn** ở hệ thống thật 25 fps — đừng chép sang chương 6 như đặc
  tính của hệ thống, cũng đừng chỉnh tham số tracker theo dataset này (đã chốt phiên 11/12).
- CLAUDE.md §2 nói máy dev "không GPU NVIDIA dùng được"; máy thực có **GTX 1660 SUPER** (dùng
  được cho suy luận ONNX nếu cài CUDA EP, chưa thử) và venv Re-ID mới `~/.venvs/mct-reid`
  (Python 3.10 + `opencv-python-headless`; hiện đang cài `onnxruntime-directml` 1.23.0 thay cho
  `onnxruntime` 1.23.2 vì lần thử DirectML — **nên gỡ về `onnxruntime` thường** để khỏi lẫn;
  embedding B/C dùng build directml với provider CPU, chênh lệch số học so với build thường
  chưa kiểm, nhưng B/C/LB/LC cùng dùng nên không ảnh hưởng phép so sánh giữa chúng) chưa có trong CLAUDE.md. Chưa sửa CLAUDE.md vì tệp đang có thay đổi chưa commit của bạn.
- 69 message của fixture C/B vi phạm contract (`confidence = -0.1`, target do nvtracker suy ra)
  — giữ nguyên như `reembed_fixture.py` đã chú thích từ phiên 21.

## Bước tiếp theo

1. **Phần liên kết còn ~10–12 HOTA đáng thu, và nó nằm trong `src/mct`.** B cho `AssPr 87.65`
   nhưng `AssRe 36.01` (Global ID 361 cho 288 người): engine **tách nhầm nhiều hơn gộp nhầm**
   ngay cả khi tracklet hoàn hảo. Bước rẻ nhất: chạy `eval/diagnose_global_ids.py` trên
   `data/s24/B_r*/mct.db` để phân rã 361 ID (vỡ do ngoại hình? do hình học `max_ground_dist_m`?
   do `min_frames` bỏ tracklet ngắn?) — tracklet sạch nên đây là phép đo **khử nhiễu tracker**
   tốt nhất mà repo có. Không cần GPU.
2. **Tracker:** không chỉnh NvDCF theo WildTrack. Ghi nhận cần đo lại **AssA đơn camera của
   NvDCF ở 25 fps** trên dữ liệu tự thu (M6) trước khi quyết có đầu tư vào tracker hay không.
3. (Nếu cần số sạch cho chương 6) chạy B_full; thêm chỗ cắm `EmbeddingCache` vào
   `wildtrack_to_fixture.py`, ~22 000 crop ≈ 75 phút CPU.
4. Gỡ `onnxruntime-directml` khỏi `mct-reid`; bổ sung venv này và ghi chú GPU vào CLAUDE.md §2.
