# 2026-11-07 — Re-ID cần ~1 s mới nhận chắc; xét lại quyết định gán đầu (`revise_after_frames`)

- **Mốc:** M6 (đo + sửa) | **Máy:** máy dev Windows (CPU), không thuê GPU | **Thời lượng:** ~4h

## Mục tiêu phiên

- Trả lời "vì sao vẫn nhận nhầm người" bằng số, sau phiên 43.
- Kiểm xem Re-ID có phải điểm nghẽn không, trước khi tính chuyện fine-tune.
- Thử cơ chế "cấp ID ngay, sửa một lần sau ~1 s".

## Đã làm

1. **Phân loại lỗi nhận nhầm trên c0** (`eval.diagnose_decisions`, `eval_gpu42`, 12 buổi × r1–r3):
   162 lần nhận nhầm ở quyết định đầu, ~54 mỗi lần chạy.
2. **Đo độ phân biệt Re-ID trên chính embedding engine thấy**, không chạy lại model:
   - `eval/reid_lab_separability.py`: mỗi tracklet pipeline đã nối với người GT (`tables/rN.gt.json`), descriptor là trung bình L2 cả tracklet. Đo rank-1/mAP so với mức đoán bừa, và AUC của khoảng cách cặp.
   - `eval/reid_lab_early.py`: query chỉ lấy N khung đầu (top-10 theo confidence trong N khung đó); gallery là các tracklet bắt đầu trước, tính cả đoạn.
3. **Cài `association.revise_after_frames`** (mặc định `null` = hành vi cũ từng bit):
   - `src/mct/affinity.py`: khoá trong `AffinityConfig`.
   - `src/mct/gallery.py`: `Gallery.detach`, `Gallery.drop`, `_paths`.
     - `detach` gỡ tracklet khỏi GlobalTrack.
     - Nó phục hồi trạng thái theo camera từ các tracklet còn lại: span, tracklet mới nhất kèm quỹ đạo, mốc thấy lần cuối, chủ cũ trong `_former_owner`.
     - Centroid EMA được tính lại bằng trung bình các entry còn giữ.
   - `src/mct/associator.py`: `_provisional`, `_due_revision`, `_revise`; thống kê `revised` / `revised_changed`.
   - `configs/mct.yaml`: khoá `revise_after_frames: null`.
   - `tests/test_associator.py`: +5 test:
     - chuyển sang đúng người;
     - bỏ ID tạm khi hoá ra là người cũ;
     - giữ ID cũ khi vẫn là người mới;
     - trả tracklet trước về làm chủ camera;
     - tắt mặc định.
4. **Sweep 12 buổi × 3 lần chạy** (`data/lab/oracle/eval_rv`, bảng `data/s42/rv_eval.md`):
   - các biến thể: c0, rv12, rv25, rv50;
   - fixture là của phiên 43 (`data/lab/gpu_s42`), nên engine chỉ phát lại trên CPU.
5. **Chấm nghiêm** (`eval/strict_revise_score.py`):
   - SQLite giữ một Global ID cho mỗi tracklet (upsert), nên xét lại đổi ID là ghi đè ngược về cả đoạn, kể cả phần màn hình thật đã hiện ID tạm.
   - Script phát lại engine trong tiến trình (tất định, cùng `tracklet_id`), ghi các lần đổi ID, rồi tách trong `mct.db` một dòng ID tạm cho `[start, end lúc xét lại]`.
   - Chấm lại bằng `run_lab_eval` không `--force` (dùng lại DB).
   - Ở s3_1 có 2 dòng ID tạm trùng khung với tracklet khác cùng ID trên cùng camera (TrackEval từ chối). Cách xử lý bảo thủ: cấp ID riêng, tức tính như một lần tách.
   - Làm trong scratchpad trên C: vì ổ E đầy.
6. **Deck M6** (`claude.ai/artifact/Bv5wdoAZBkntpjDjNV5QFb`) thêm 5 slide, giờ 22 slide:
   - "Liên kết đa camera hoạt động thế nào" (3 bước: đo độ giống, luật vật lý, ghép Hungarian);
   - "Chi phí = ngoại hình + λ × khoảng cách";
   - "Vì sao vẫn còn nhận nhầm người";
   - "HOTA thấp vì đâu" (DetA/AssA theo nhóm, 38 điểm mất theo tầng);
   - "Vì sao nối người giữa các camera lại khó".
7. **Dọn ổ E:** đầy 100% sau sweep, còn 14 MB rồi 920 KB.
   - Đã xoá `data/lab/oracle/{eval,eval_v2,eval_v3,eval_c23}` (~7.9 GB, kết quả trung gian sinh lại được, số đã ghi ở worklog 40–42).
   - Còn trống 7.6 GB. Giữ `eval_v4`/`eval_v5` vì worklog 42 trỏ tới.

## Quyết định kỹ thuật

1. **Không fine-tune Re-ID.**
   - Đo trên embedding thật: nhìn cả tracklet thì rank-1 khác camera là 94% (đoán bừa 31%), AUC cặp 0.91–0.94.
   - Mô hình đủ phân biệt; lỗi nằm ở chỗ engine quyết định khi query mới có vài khung.
   - Phương án bị loại: fine-tune trên dữ liệu lab. Lý do: ~7 người, cùng người ở cả tập học lẫn tập chấm, và phải đưa ảnh người thật kèm nhãn danh tính lên GPU thuê (vướng phiếu đồng thuận).
   - **Đính chính** giả thuyết đầu phiên rằng "mắt bị lừa vì model học ở chỗ khác": số đo không ủng hộ giả thuyết đó.
2. **Xét lại một lần, không trì hoãn quyết định.**
   - Tăng `min_frames` lên 25 sẽ đẩy thời gian tới Global ID thêm ~1 s, phá mục tiêu < 1 s.
   - Xét lại giữ nguyên lần cấp ID đầu (bản ghi latency vẫn là `first`; lần xét lại là `is_update=True`), nên (B) không đổi.
   - Chỉ một lần cho mỗi tracklet, để danh tính không nhảy qua lại.
3. **Ba kết cục của lần xét lại:**
   - giữ track cũ;
   - chuyển sang track khác, và bỏ ID tạm nếu nó rỗng;
   - tách ra ID mới, nhưng nếu track cũ chỉ có mình nó thì giữ luôn ID cũ, không cấp số mới vô ích.
4. **Chọn rv25 trên s2** (bàn giao 84% so với rv12 80%, rv50 82%; HOTA 78.5 / 78.3).
5. **Báo cáo theo chấm nghiêm, không theo chấm lỏng.** Chấm lỏng thổi phồng bàn giao (nhóm giữ riêng 48% so với 41%), vì bàn giao được chấm đúng lúc chuyển camera, tức đúng khoảng ID tạm.
6. **Chưa đặt `revise_after_frames: 25` vào `configs/lab/lab.mct.yaml`.** Chờ người dùng duyệt, vì mọi số trong deck và báo cáo sẽ đổi theo.

## Số liệu đo được

Cấu hình chung:
- fixture pipeline phiên 43 (RTX A4000, YOLO11s 640 TensorRT FP16 b3, OSNet x1.0 MSDC 512, NvDCF, 3 luồng 25 fps), 12 buổi × r1–r3;
- `configs/lab/lab.mct.yaml` (combo + c0);
- chấm hộp ảnh, chỉ khung chú thích, IoU 0.5;
- nhóm giữ riêng = s1_1, s3_1, s3_2, s4_1;
- engine phát lại trên máy dev.

**Nhận nhầm ở quyết định đầu, c0** (162 lần / 3 lần chạy):

| | Số lần |
|---|---|
| Lúc quay lại cùng camera | 77 (48%) |
| Lúc sang cam03 (không chồng lấn) | 46 (28%) |
| Giữa cam01 và cam02 (chồng lấn) | 39 (24%) |

- Ô người đúng dưới ngưỡng nhưng thua: 52. Ô người đúng vượt ngưỡng: 54. Ô người đúng bị loại bởi ràng buộc: 25. Không có track đúng để ghép (người mới bị ghép vào người cũ): 31.
- Trong 131 lần có người đúng để so: 73 lần người sai được chấm giống hơn, 58 lần người đúng giống hơn nhưng thua vì vị trí / thời gian / ngân sách ô.
- s3 chiếm 78/162.

**Re-ID trên cả tracklet** (`reid_lab_separability.py`):

| | rank-1 | Đoán bừa | mAP |
|---|---|---|---|
| Khác camera, cả 12 | 94.2% | 30.5% | 92.7% |
| Cùng camera (tracklet nối tiếp), cả 12 | 95.4% | 39.6% | 95.3% |
| Khác camera, s3 | 91.0% | 17.0% | 90.7% |

- AUC khoảng cách cặp: chồng lấn 0.938, không chồng lấn 0.905, cùng camera 0.909.
- Dưới ngưỡng 0.40: cặp đúng 92–97%, cặp sai 58–75%.

**Re-ID theo số khung query** (`reid_lab_early.py`, n = 573 query):

| N khung đầu | 3 | 5 | 10 | 25 | 50 | cả tracklet |
|---|---|---|---|---|---|---|
| rank-1 | 58.1% | 60.4% | 64.6% | 81.8% | 86.6% | 88.1% |

- Có ít nhất một người sai dưới 0.40: 70–81% số query.

**Sweep xét lại, chấm lỏng** (`data/s42/rv_eval.md`):

| Nhóm | Biến thể | HOTA | IDF1 | Bàn giao [Wilson 95%] |
|---|---|---|---|---|
| s2 | c0 | 69.9 | 76.7 | 70% [64–75] |
| s2 | rv12 / rv25 / rv50 | 74.8 / 78.5 / 78.3 | 83.7 / 88.9 / 88.3 | 80 / 84 / 82% |
| Giữ riêng | c0 | 53.4 | 55.7 | 39% [33–45] |
| Giữ riêng | rv25 | 58.9 | 64.4 | 48% [42–55] |

**rv25 chấm nghiêm** (98 lần đổi ID khi xét lại / 36 lần chạy; scratchpad `rv_strict_eval.md`):

| Nhóm | HOTA | AssA | IDF1 | Chồng lấn | Không chồng lấn | Quay lại | Tổng [Wilson 95%] | Đúng / tách / nhầm |
|---|---|---|---|---|---|---|---|---|
| s2 | 78.1 | 73.1 | 88.3 | 82% | 71% | 58% | 73% [67–78] | 182/46/21 |
| Giữ riêng | 58.4 | 43.7 | 63.8 | 35% | 28% | 83% | 41% [35–48] | 99/80/61 |
| Cả 12 | 71.5 | 63.3 | 80.2 | 53% | 59% | 69% | 57% [53–62] | 281/126/82 |

So với c0:
- cả 12: HOTA 64.4 → 71.5, AssA 52.1 → 63.3, IDF1 69.7 → 80.2, bàn giao 55 → 57% (trong nhiễu);
- nhóm giữ riêng: HOTA +5.0, IDF1 +8.1, bàn giao 39 → 41%;
- nhưng số nhận nhầm tại lúc chuyển ở nhóm giữ riêng tăng 51 → 61 (chưa chẩn đoán).

**Phân rã HOTA** (đọc lại từ số phiên 41, `combo`, chưa đo lại trên c0):
- A 62.0, C 62.0, B 66.4.
- Detector ~0 điểm, tracker đơn camera ~4.4, liên kết ~34 trong 38 điểm mất.
- DetA suy từ HOTA²/AssA trên c0: s2 ~82, nhóm giữ riêng ~78, s3 ~78. AssA lần lượt 60 / 37 / 29.

**Test:** 889 passed, 5 skipped trên Python 3.10.22 (`~/.venvs/mct-test`); ruff sạch.

## Vướng mắc / chưa xong

- **Chờ người dùng quyết:** có đặt `revise_after_frames: 25` vào `configs/lab/lab.mct.yaml` không. Nếu có, số trong deck và chương 6 phải chạy lại theo cấu hình mới. Hiện deck ghi rv là "thử nghiệm mới".
- Chưa chẩn đoán vì sao nhận nhầm lúc chuyển ở nhóm giữ riêng tăng 51 → 61 với rv25. Giả thuyết: gallery đổi sau mỗi lần xét lại nên các quyết định sau đó cũng đổi.
- **Chưa đo lại kịch bản B (tracker hoàn hảo) trên c0 / rv25.** Phân rã "liên kết ~34 điểm" là số phiên 41. Cũng chưa mở lại script để kiểm chính xác B dùng hộp nào trước khi đưa vào báo cáo.
- **Chấm nghiêm chỉ nghiêm với lần xét lại.** Mọi biến thể (kể cả c0) vẫn gán ngược Global ID cho ~0.2–0.5 s trước lần gán đầu, như mọi phiên trước.
- `eval_rv` (885 MB) nằm trên E:. Kết quả chấm nghiêm (`rv_strict`, ~850 MB) chỉ ở scratchpad, sẽ mất khi dọn scratchpad. Bảng đã chép vào đây.
- Ổ E còn 7.6 GB. Các chỗ có thể xoá thêm nếu cần: `data/Wildtrack_dataset_full.zip` (6.4 GB, đã giải nén), `data/lab/oracle/fixtures` (1.7 GB, bản trước khi sửa nhãn).

## Bước tiếp theo

1. Người dùng duyệt rv25 → đặt vào `lab.mct.yaml`, chạy lại `lab-eval` để có số chính thức. Chạy lại cả độ trễ đồng hồ thật nếu muốn khẳng định (B) không đổi; về lý thuyết nó không đổi.
2. Chẩn đoán nhận nhầm tăng ở nhóm giữ riêng (`diagnose_decisions` trên rv25).
3. Đo lại kịch bản B trên code hiện tại, để phân rã HOTA theo tầng dùng được cho chương 6.
4. Hướng cải thiện tiếp theo: λ theo vị trí (chia khoảng cách cho sai số homography dự kiến); dùng chỗ người xuất hiện (mép/cửa là người mới).
5. Còn từ phiên 43: ablation, kiểm chéo bỏ một buổi, baseline đơn giản, video demo; rồi M7.
