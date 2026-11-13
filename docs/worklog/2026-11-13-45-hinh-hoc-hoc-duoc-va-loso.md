# 2026-11-13 — Ba cải tiến hình học học được, chấm bỏ-một-buổi (LOSO)

- **Mốc:** M6 (cải tiến liên kết) | **Máy:** máy dev macOS (CPU, 8 nhân), không thuê GPU | **Thời lượng:** ~7h (kéo sang 2026-11-14)

## Mục tiêu phiên

- Cải thiện bàn giao ở cặp chồng lấn và không chồng lấn bằng 3 cải tiến, mỗi cái một khoá cấu hình, mặc định `null` = hành vi cũ từng bit.
- Số chính là LOSO trên 8 buổi `s2_*`: mỗi fold fit lại mọi thành phần học được và chọn ứng viên chỉ trên 7 buổi còn lại.
- Không rò dữ liệu: nhóm giữ riêng (s1_1, s3_1, s3_2, s4_1) không được đọc trong lúc phát triển.

## Đã làm

- `src/mct/learned.py` (mới): bốn mô hình chỉ ĐỌC file, engine không fit lúc chạy. File thiếu `fit_sessions`/`fit_date` bị từ chối.
  - `association.foot_from_height` → `FootModel`: hộp chạm mép dưới (đáy ≥ H − 30 px) thì suy điểm chân từ đỉnh đầu. Hai dạng: `height_line` (đúng đề bài: chiều cao px tuyến tính theo y chân) và `head_plane` (homography từ điểm đỉnh đầu, tức mặt phẳng ở chiều cao đầu trung bình).
  - `association.ground_sigma_model` → `GroundErrorModel`: thay `d` bằng `|Δ − μ(ô)| · σ_ref / σ(ô)` theo ô sàn 1 m. Ba dạng fit: `sigma` (đúng đề bài, μ = 0), `bias` (chỉ bù lệch), `bias_sigma`.
  - `association.exit_direction_gate` → `ExitDirectionModel`: loại ứng viên rời camera nguồn ngược hướng tới camera đích (vận tốc mặt đất 1 s cuối, tốc độ ≥ 0.3 m/s, cos < 0).
  - `association.entry_zone_prior` → `EntryZoneModel`: tracklet bắt đầu ở ô ảnh "vùng vào" (lưới 4 × 6, P(mới đến) ≥ 0.8, n ≥ 10) thì ô nối cùng camera mất ngân sách `max_cost_geometric`.
- `src/mct/tracklet.py`, `src/mct/affinity.py`: nối bốn khoá vào. `configs/mct.yaml`: thêm bốn khoá, đều `null`. **Không đổi `configs/lab/lab.mct.yaml`.**
- `eval/lab_geometry.py` (mới): đọc fixture một lần, bỏ embedding, cache mảng hộp. `check_sessions` ném `LeakageError` nếu danh sách buổi giao với nhóm giữ riêng hoặc với buổi test của fold.
- `eval/fit_learned.py` (mới): `fit_all(lab, sessions, …, exclude=…)` ghi `foot_height.yaml`, `ground_sigma.yaml`, `exit_direction.yaml`, `entry_zones.yaml` kèm `fit_sessions`, `fit_date`.
- `eval/loso_eval.py` (mới): LOSO tái dùng `run_lab_eval` (thêm cờ `--no-sct`) và `time_to_id_datatime` (thêm `--json`). Cache theo băm (cấu hình + nội dung file học được).
- Test: `tests/test_learned.py`, `tests/test_fit_learned.py`, `tests/test_loso_eval.py`. Toàn bộ: 919 passed, 12 skipped (Python 3.10.21), ruff sạch.
- `configs/lab/learned/foot_height.yaml`: mô hình chân `head_plane`, fit trên cả 8 buổi `s2_*`. Chưa khoá nào trỏ tới nó.
- Đã thử rồi bỏ: giới hạn độ kéo dài điểm chân ảo (không đổi gì); σ là hàm của khoảng cách tới camera (AUC cùng người/khác người tụt 0.912 → 0.794).
- Bỏ qua biến thể tuỳ chọn hồi quy logistic của cải tiến 2.

## Quyết định kỹ thuật

1. **Baseline = c0 hiện tại trong `configs/lab/lab.mct.yaml`.** Ô baseline trong đề bài phiên để trống; `revise_after_frames: 25` chưa được duyệt đưa vào cấu hình lab nên không dùng. Vì thế không cần chấm nghiêm kiểu phiên 44.
2. **Làm trên worktree sạch** (`.claude/worktrees/phien-45`, nhánh `phien-45-cai-tien-lien-ket`, từ `85d3bae`). Cây chính đang có thay đổi chưa commit ở `src/mct/*` (bản sửa từ review, worklog 2026-09-21-26) làm đổi hành vi mặc định và chưa từng được đo, nên không trộn vào phép đo này. Các thay đổi đó không bị đụng tới.
3. **Cải tiến 1: giữ, dạng `head_plane`.** Dạng `height_line` đúng đề bài làm điểm chân hai camera lệch nhau HƠN (trung vị cùng người cam01–cam02: 0.66 → 0.84 m), vì quan hệ chiều cao–hàng chân cong đi ở sát camera, đúng chỗ hộp bị cắt. `head_plane` cùng giả định (người cao trung bình) nhưng không tuyến tính hoá: 0.66 → 0.51 m. Cả 8 fold đều chọn `head_plane`.
4. **Cải tiến 2: trong nhiễu, khoá giữ `null`.** So với +1: HOTA 71.5 → 71.2, bàn giao 74% → 75%, khoảng Wilson gần trùng nhau. Dạng `sigma` đúng đề bài thua `bias` ở cả 8 fold; `homography_weight` 0.4 thua 0.2 ở cả 8 fold.
5. **Cải tiến 3: không giữ, cả hai khoá `null`.**
   - Cổng hướng ra: kết quả TRÙNG +1 từng số. Chưa đếm cổng đã loại bao nhiêu ô; chỉ biết nó không đổi quyết định nào được chấm.
   - Vùng vào: âm tính. Quay lại cùng camera 53% → 40% (24 → 18 / 45), không chồng lấn 65% → 62%.
6. **Cấu hình cuối đề xuất = c0 + `foot_from_height: configs/lab/learned/foot_height.yaml`**, chờ người dùng duyệt trước khi chạy nhóm giữ riêng.
7. **Kiểm hồi quy WildTrack bằng so từng bit thay vì chấm lại**: bốn khoá `null` thì engine phải cho đúng SQLite cũ, và đó là phép kiểm mạnh hơn so điểm trong nhiễu.

## Số liệu đo được

Cấu hình chung: fixture pipeline thật `data/lab/gpu_s42` (YOLO11s 640 + NvDCF ReID, 3 camera, 25 fps, 3 lần chạy r1–r3 mỗi buổi), phát lại engine trên CPU, `configs/lab/lab.mct.yaml` (c0), TrackEval chỉ trên khung có GT. LOSO 8 buổi `s2_*`; HOTA/AssA/IDF1 là trung bình 8 buổi test, bàn giao gộp qua fold.

| bước | HOTA | AssA | IDF1 | chồng lấn | không chồng lấn | quay lại | tổng [Wilson 95%] | đúng/tách/nhầm | tới Global ID p50 / p95 (ms) | chọn theo fold |
|---|---|---|---|---|---|---|---|---|---|---|
| base (c0) | 69.9 | 59.9 | 76.7 | 79% (83/105) | 66% (65/99) | 58% (26/45) | 70% [64–75] | 174/56/19 | 439 / 679 | — |
| +1 | 71.5 | 62.0 | 78.8 | 91% (96/105) | 65% (64/99) | 53% (24/45) | 74% [68–79] | 184/47/18 | 439 / 679 | head×8 |
| +1+2 | 71.2 | 61.4 | 77.9 | 94% (99/105) | 65% (64/99) | 51% (23/45) | 75% [69–80] | 186/44/19 | 439 / 679 | bias_w0.2×5, bias_sigma_w0.2×3 |
| +1+2+3 | 70.5 | 60.2 | 76.4 | 96% (101/105) | 62% (61/99) | 38% (17/45) | 72% [66–77] | 179/53/17 | 439 / 679 | bias×5, bias_sigma×3 |
| +1 + cổng hướng ra | 71.5 | 62.0 | 78.8 | 91% (96/105) | 65% (64/99) | 53% (24/45) | 74% [68–79] | 184/47/18 | 439 / 679 | — |
| +1 + vùng vào | 70.8 | 60.7 | 77.3 | 93% (98/105) | 62% (61/99) | 40% (18/45) | 71% [65–76] | 177/56/16 | 439 / 679 | — |
| +1 + cả hai của 3 | 70.8 | 60.8 | 77.3 | 93% (98/105) | 62% (61/99) | 40% (18/45) | 71% [65–76] | 177/56/16 | 439 / 679 | — |

Đọc bảng:

- Dòng `base` trùng số c0 trên s2 của phiên 43 (HOTA 69.9, IDF1 76.7, 70% [64–75], 174/56/19): đường khoá `null` không đổi hành vi, và máy macOS tái lập số của máy Windows.
- **+1 so với base:** tổng bàn giao 70% → 74%, hai khoảng Wilson chồng nhau → **trong nhiễu ở mức tổng**. Riêng cặp chồng lấn 79% [70–86] → 91% [85–95] (Wilson tự tính, n = 105), hai khoảng gần như tách; đây là chỗ cải tiến nhắm tới. HOTA +1.6, IDF1 +2.1. Không chồng lấn và quay lại: trong nhiễu (−1 và −2 lượt).
- **Cặp không chồng lấn không cải thiện ở bất kỳ bước nào** (65–66% → 62–65%, nhiễu giữa các lần chạy ±9 điểm). Mục tiêu "cải thiện cặp không chồng lấn" của phiên KHÔNG đạt.
- Thời gian tới Global ID không đổi ở mọi bước (p95 679 ms < 1 s).

Số phụ (chỉ trên `s2_*`, không qua LOSO, dùng để thiết kế):

- Hộp bị cắt đáy (đáy ≥ 1060 px): cam01 18.5%, cam02 7.2%, cam03 51% số hộp.
- Lệch điểm chân cùng người cam01–cam02, hộp bị cắt, `head_plane`: trung vị 0.51 m, p90 1.17 m (không mô hình: 0.66 / 1.57).
- Sau `head_plane`, hộp không cắt vẫn lệch hệ thống theo trục hành lang: dx từ ≈ 0 ở x = −3 m đến +0.83 m ở x = 0 và −0.93 m ở x = 4 m. AUC cùng/khác người: d thô 0.912, bù lệch theo ô 0.972 (trong mẫu). Tách được tốt hơn nhưng không thành điểm bàn giao.
- Hướng ra `cam01>cam03` (0.9998, 0.0204), n = 27, cos nhỏ nhất 0.965; `cam03>cam01` n = 24, cos nhỏ nhất 0.984. Không có mẫu cam02↔cam03.

**WildTrack, kiểm hồi quy** (fixture `ds_wildtrack_7cam_r640n_r{1,2,3}`, `configs/demo/wildtrack_ds.mct.yaml`, bốn khoá mới `null`): `sqlite3 .dump` của engine ở `85d3bae` và ở đầu nhánh phiên này **trùng md5 cả ba lần chạy** (r1: 385 Global ID, đúng số IDs của phiên 42). Không hồi quy. Không chạy mô hình chân trên WildTrack: mô hình fit cho camera lab.

## Kiểm tra rò dữ liệu

- **Buổi dùng để fit từng thành phần** (mỗi fold LOSO: 7 buổi `s2_*` trừ buổi test; file cuối: cả 8):
  - điểm chân (`foot_height`): `s2_*`, hộp pipeline, không dùng danh tính;
  - độ lệch mặt đất (`ground_sigma`): `s2_*`, dùng nhãn GT chỉ để biết hai hộp là cùng một người ở cùng thời điểm;
  - hướng ra (`exit_direction`): `s2_*` + `calib`, `calib_2` (GT);
  - vùng vào (`entry_zones`): `s2_*`.
- Không học gì từ embedding; `lab_geometry` bỏ embedding ngay khi đọc. Không fine-tune.
- **Nhóm giữ riêng trong phiên này: đã chạy 0 lần, chưa đọc file nào** (GT, fixture, log, kết quả chấm). Lần chạy sắp tới, nếu được duyệt, là **lần nhìn thứ 5** vào nhóm này tính từ phiên 40 (sau các phiên 41, 42, 43, 44).
- **Assert bảo vệ:**
  - `eval.lab_geometry.check_sessions`: mọi đường đọc (`gt_boxes`, `run_boxes`, `pipeline`, `ground_truth`) và `fit_all` ném `LeakageError` trước khi mở file;
  - `fit_all(exclude=[buổi test])`: fold không fit được trên buổi test của chính nó;
  - `loso_eval.assert_models_clean`: đọc lại `fit_sessions` trong file đã ghi trước khi chạy engine;
  - `mct.learned.load_learned`: engine từ chối file không có nguồn gốc;
  - test: `test_buoi_giu_rieng_bi_chan_o_moi_cua` (4 buổi), `test_fit_all_ghi_nguon_goc_va_chan_buoi_test`, `test_evaluate_chon_ung_vien_khong_nhin_buoi_test`.
- Chọn ứng viên: theo quy tắc phiên 43 trên 7 buổi train của từng fold. Lưu ý: lưới ứng viên của bước +1+2 cố định `foot_method=head_plane` vì bước +1 đã chọn nó ở cả 8 fold.
- Không sửa nhãn GT. **Cần người kiểm:** không có mục nào.

## Vướng mắc / chưa xong

- **Chờ duyệt cấu hình cuối** rồi mới chạy nhóm giữ riêng một lần.
- Chưa biết vì sao cổng hướng ra không đổi gì: có thể các ứng viên ngược hướng vốn đã bị transit time hoặc ngoại hình loại.
- Cặp không chồng lấn vẫn ~65%; hình học không giúp được, phần mất nằm ở ngoại hình/thời gian.
- Nhánh `phien-45-cai-tien-lien-ket` chưa gộp vào `main`, chưa push. Cây chính còn thay đổi chưa commit ở `src/mct/*` và `docs/worklog/README.md`; khi gộp sẽ phải xử lý xung đột.
- Máy này là macOS (CLAUDE.md ghi Windows): đã dựng `~/.venvs/mct-eval` và clone `~/TrackEval`; không có ổ E, đĩa còn 44 GB.

## Bước tiếp theo

1. Người dùng duyệt cấu hình cuối → chạy nhóm giữ riêng một lần, ghi số vào file này, không chỉnh gì sau đó.
2. Nếu giữ: hỏi người dùng thêm `foot_from_height` vào `configs/lab/lab.mct.yaml`.
3. Đếm số ô cổng hướng ra loại (`diagnose_decisions`) để biết nó vô hại hay vô dụng.
