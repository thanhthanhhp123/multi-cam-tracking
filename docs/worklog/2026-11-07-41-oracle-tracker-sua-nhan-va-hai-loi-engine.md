# 2026-11-07 (phiên 41): Tách lỗi tracker và liên kết trên dữ liệu lab, sửa nhãn, tìm hai lỗi engine

- **Mốc:** M6 (đo + sửa) | **Máy:** máy dev (Windows 10, i5-12400F, RTX 3060 chưa dùng), không thuê GPU | **Thời lượng:** ~7h (tối 2026-11-06 sang 2026-11-07)

Tiếp phiên 40. Nguyên tắc 1 của CLAUDE.md §1: tách lỗi theo tầng trước khi tối ưu.

## Mục tiêu phiên

- Tách phần điểm mất do tracker đơn camera khỏi phần mất do liên kết trên dữ liệu lab (bước 1 đề xuất cuối phiên 40).
- Kiểm và sửa nhãn GT theo `data/prelabel/SCAN_REPORT.md`.
- Khắc phục hai điểm yếu: bàn giao thấp (47%) và thời gian tới Global ID p95 9.6 s.

## Đã làm

**Thí nghiệm oracle tracker cho lab** (`src/tools/lab_oracle_fixture.py` + `tests/test_lab_oracle_fixture.py`, 3 test):
- Từ một fixture pipeline sinh hai fixture cùng tập detection (khớp GT ở IoU ≥ 0.5), cùng hộp, embedding và `ts_ms`. **C** giữ id NvDCF; **B** thay id bằng Global ID ground-truth.
- GT chú thích nhảy khung (5 fps), nên hộp GT được nội suy tuyến tính giữa hai khung chú thích liền kề của cùng track. Không nội suy qua khoảng hở lớn hơn bước chú thích. Engine vì thế vẫn thấy 25 fps như lần chạy thật.
- Kiểm: sau lọc vùng sàn, C giữ 99.4% detection của A (phần bị loại trước khi lọc là người ngoài `ground_roi` của cam01).
- Chạy A/C/B × `base`/`combo` × 12 buổi × 3 lần chạy: `data/lab/oracle/eval{,_v2}/`.

**Sửa nhãn GT** (qua API của `tools.gt_review`, mọi thao tác ghi trong `data/cvat/<buổi>/edits.jsonl`; sao lưu ở `data/cvat_backup_20261106_2309/`):
- Gộp 106 lần các track cùng mã, cùng camera, nối tiếp nhau (76 ở s3_2, do người duyệt bấm "+ Người" mỗi khung; còn lại sinh từ thao tác cắt/tách ở phiên 38). s4_1 cam03 chỏm đầu: 36 track một khung → một track (làm mẫu trong Chrome).
- Lấp 27 lỗ một khung bằng nội suy. Bỏ một lần lấp sai ở s2_2c cam02 khung 133 (P01 bị P02 che kín, lỗ là cố ý). s2_2 cam02 khung 51: lỗ của P02 đã có mảnh riêng, hoàn tác lần lấp và gộp mảnh.
- s3_2 cam01: X01 (áo vàng "FINISHER", mặt trước "AMAZING HALONG 2023") khung 97 hộp nhảy sang đầu người khác, 98–104 thiếu hộp → đặt lại bằng nội suy giữa hai khung GT kề bên, `occluded` ở 101–104. X02 bỏ hộp trùng P01 ở khung 87, nối 177–178. P01 và P02 (cam02) gộp mảnh, lấp khung thiếu.
- s4_1 cam03 chỏm đầu: X01 → **X03** (X01 gốc là người áo đỏ đi qua ở khung 181–212).
- Xuất lại GT cả 12 buổi (`cvat_to_mot`, tham số như `make lab-gt`).
- Công cụ phụ ở scratchpad (chưa vào repo): `check_gt.py` (hộp trùng/hỏng/nhảy, lỗ một khung, mảnh vụn), `render_findings.py`, `id_spans.py` (một ảnh cắt cho mỗi đoạn của mỗi mã), `merge_fragments.py`, `fix_gt.py`.

**Chẩn đoán liên kết trên kịch bản B** (scratchpad `diag_decisions.py`): ghi lại quyết định ĐẦU TIÊN của engine cho từng tracklet (gán đúng / gộp nhầm / tạo mới đúng / tách), kèm chi phí, ngân sách của ô và lý do loại của ô đúng người.

**Hai lỗi engine và bản sửa:**
1. `association.evidence_max_gap_ms` (`src/mct/affinity.py`, mặc định `null` = hành vi cũ từng bit). Cổng vị trí nới theo `Δt` (nối mảnh cùng camera; cặp chồng lấn không có mốc chung) chỉ được tính là bằng chứng vị trí, tức được hưởng `max_cost_geometric`, khi `Δt` ≤ ngưỡng. Vượt ngưỡng thì cổng vẫn loại được, nhưng ô quay về `max_cost`. +3 test trong `tests/test_affinity.py`, thêm khoá vào `configs/mct.yaml`.
2. `Gallery.pop_former_owner` + nhánh trong `Associator.assign` (`src/mct/gallery.py`, `src/mct/associator.py`). Tracklet bị một tracklet mới hơn cùng camera thế chỗ trong GlobalTrack thì lúc đóng giữ nguyên danh tính, không bị ghép lại từ đầu. **Thay đổi hành vi mặc định** (không có cờ), xem Quyết định 4. +1 test hồi quy trong `tests/test_associator.py`.

**Khác:** `tests/test_check_lab_setup.py` còn đòi topology `template` dù phiên 40 đã chuyển `measured` → sửa test. Cài vào venv `~/venvs/mct-stats` (Python 3.12): pytest, ruff, pillow, redis, httpx. **864 passed, 5 skipped**, ruff sạch, nhưng là Python 3.12, chưa chạy 3.10 (máy này không có `~/.venvs/mct-test`, không có `uv`). Server `gt_review` khởi động lại bằng venv `mct-stats` (Python hệ thống không có fastapi).

## Quyết định kỹ thuật

1. **Oracle lab nội suy GT lên 25 fps, không giữ riêng khung chú thích.** Chỉ chạy 5 fps sẽ đổi luôn tốc độ khung engine thấy (`min_frames`, `max_embeddings`, `ground_time_tol_ms` tính theo khung/ms), tức hai biến cùng lúc.
2. **Sửa nhãn bằng nội suy giữa hai khung GT, không lấy hộp detector đang được chấm** (YOLO11s), kể cả ở khung người bị che. Cùng lý do phiên 38: tránh thổi điểm của chính detector được chấm.
3. **Mã X dùng lại giữa các buổi là đúng** (mỗi buổi chấm với bảng danh tính riêng). Chỉ sai khi trong CÙNG một buổi một mã gán cho hai người. Chỏm đầu không có ngoại hình để so → đặt mã riêng (X03): nếu là người khác mà chung mã thì phạt oan engine; nếu cùng người thì engine cũng không nối được.
4. **`pop_former_owner` sửa luôn hành vi mặc định, không đặt cờ.** Đây là lỗi chứ không phải lựa chọn thiết kế: danh tính của tracklet được chốt ở vòng gán đầu tiên (đính chính phiên 28), mà lỗi này phá đúng mệnh đề đó và ghi đè Global ID cả tracklet trong SQLite (`appearances` upsert theo `tracklet_id`). Hệ quả: số của phiên 40 và các phiên trước KHÔNG còn tái lập từng bit trên code mới (chênh nhỏ, xem số đo). Trường hợp tracklet bị thế chỗ mà còn sống và có khung mới (hai tracklet cùng camera song song: mơ hồ thật) vẫn để engine xét lại như cũ.
5. **`evidence_max_gap_ms` có cờ, mặc định `null`.** Đổi ngân sách ô là lựa chọn mô hình, phải chọn bằng sweep (s2 chọn, nhóm giữ riêng kiểm) như mọi ngưỡng khác.
6. **Giữ `evidence_max_gap_ms: null` (không đặt trong `configs/lab/lab.mct.yaml`).** Sweep chọn trên s2 cho e2000, nhưng nhóm giữ riêng tụt HOTA 7 / IDF1 10 (xem Số liệu). Giữ khoá trong code như một ablation âm tính có giải thích cho chương 6, không dùng làm mặc định. Phương án bị loại: lấy e2000 vì "tổng bàn giao 12 buổi tăng 44 → 53%". Con số gộp đó bị s2 kéo lên, trong khi nhóm không dùng để chọn tụt mạnh ở HOTA/IDF1.
7. **Không sửa `same_camera_stitch` bằng cách tắt nó.** Nối mảnh cùng camera vẫn cần cho người bị che rồi hiện lại (ID mới của tracker). Lỗi nằm ở chỗ coi "qua được cổng 3 m + 2.5 m/s · Δt" là bằng chứng khi Δt dài, không nằm ở việc nối.

## Số liệu đo được

Cấu hình chung: fixture pipeline phiên 40 (RTX A4000, YOLO11s 640 TensorRT FP16 b3, OSNet x1.0 MSDC 512, NvDCF, 3 luồng 25 fps), `configs/lab/lab.mct.yaml`, chấm hộp ảnh, chỉ khung chú thích, IoU 0.5; nhóm giữ riêng = s1_1, s3_1, s3_2, s4_1. Engine phát lại trên máy dev.

**A/C/B (GT sau khi sửa, code TRƯỚC hai bản sửa engine), `combo`, 12 buổi × 3 lần chạy:**

| Kịch bản | HOTA | IDF1 | Chồng lấn | Không chồng lấn | Quay lại | Tổng bàn giao |
|---|---|---|---|---|---|---|
| A pipeline | 62.0 | 66.7 | 41% | 45% | 70% | 47% |
| C detection khớp GT, id NvDCF | 62.0 | 67.4 | 42% | 45% | 58% | 45% |
| **B id GT (tracker hoàn hảo)** | **66.4** | **74.0** | 60% | 48% | 68% | **58%** |
| B, chỉ nhóm giữ riêng | 60.6 | 69.6 | 48% | 8% | 92% | 49% |

`base` cùng chiều: tổng bàn giao A 29%, C 26%, B 30%. → Tracker đơn camera chiếm khoảng 1/4 phần mất (45 → 58%), liên kết khoảng 3/4.

**Sửa nhãn gần như không đổi điểm:** mọi ô của bảng trên lệch ≤ 1 điểm so với GT trước khi sửa (`data/lab/oracle/eval` so với `eval_v2`).

**Quyết định đầu tiên của engine, kịch bản B, `combo`, r1, 12 buổi:** gán đúng 121, tạo mới đúng 51, **gộp nhầm 45** (cùng camera 27, không chồng lấn 10, chồng lấn 8), tách 17 (5 trong đó do track đúng đã bị gộp nhầm trước). Gộp nhầm cùng camera: ngân sách ô 0.90, ngoại hình trung vị 0.33 (gán đúng cùng camera: 0.24). Ví dụ s2_1b cam03 44.2 s: P2 vào cam03 → ô "nối cùng camera" của P1 (P1 vừa rời 2.4 s trước, cổng 9 m) chi phí 0.56 / ngân sách 0.9 = 0.62 thắng ô đúng của P2 (không chồng lấn) 0.31 / 0.4 = 0.78. 11/13 lần ô đúng người bị loại "quá nhanh" ở cặp không chồng lấn là hậu quả dây chuyền của một lần gộp nhầm như vậy (khoảng thời gian âm), không phải lỗi topology.

**Ghép lại tracklet lúc đóng (pipeline A, `lab.mct.yaml`, r1, 12 buổi):** 78/337 tracklet nhận lần gán "mới" thứ hai (69 lúc đóng), 4 lần đổi Global ID → sau bản sửa: 1 (tracklet song song còn sống), 1 lần đổi ID.

**Thời gian tới Global ID theo thời gian dữ liệu** (từ khung đầu của tracklet tới khung lúc được gán, mỗi lần gán "mới" một mẫu như `latency_report` đếm bản ghi `first`; cùng fixture, bản sửa tắt/bật):

| | n | p50 | p90 | p95 | p99 | < 1 s |
|---|---|---|---|---|---|---|
| Code cũ | 415 | 479 ms | 3039 ms | **10692 ms** | 15513 ms | 80.5% |
| Đã sửa | 338 | 440 ms | 640 ms | **679 ms** | 1070 ms | 98.8% |

Cột code cũ khớp log đồng hồ thật trên GPU của phiên 40 (p95 10.8 s gộp 12 buổi r1), nên phép đo theo thời gian dữ liệu tái lập được hiện tượng. Đồng hồ thật cộng thêm đoạn pipeline (t0→t1 khoảng 75 ms) và phát ra: p95 ước tính khoảng 0.8–0.9 s. **Chưa phải số báo cáo**: cần chạy lại trên GPU.

**Sweep `evidence_max_gap_ms`** ∈ {null, 0, 500, 1000, 2000} trên `combo`, 12 buổi × 3 lần chạy, code đã có bản sửa `pop_former_owner` (`data/lab/oracle/eval_v3/`). Bàn giao gộp số lần qua 3 lần chạy; HOTA/IDF1 trung bình theo buổi.

Pipeline thật (A):

| Nhóm | Biến thể | HOTA | IDF1 | Chồng lấn | Không chồng lấn | Quay lại | Tổng | đúng / tách / nhầm |
|---|---|---|---|---|---|---|---|---|
| s2 (chọn) | `combo` (null) | 68.2 | 75.5 | 67% | 45% | 47% | 55% | 136 / 72 / 41 |
| | e0 | 64.6 | 67.9 | 70% | 68% | 16% | 59% | 147 / 81 / 21 |
| | e500 | 66.4 | 70.6 | 79% | 74% | 20% | 66% | 165 / 63 / 21 |
| | e1000 | 67.7 | 72.3 | 82% | 67% | 33% | 67% | 167 / 63 / 19 |
| | e2000 | 70.3 | 76.0 | 83% | 67% | 40% | 69% | 171 / 59 / 19 |
| Giữ riêng (kiểm) | `combo` (null) | **53.5** | **55.2** | 27% | 8% | 86% | 33% | 80 / 106 / 54 |
| | e0 | 49.3 | 47.8 | 35% | 31% | 56% | 38% | 90 / 102 / 48 |
| | e500 | 46.5 | 44.6 | 30% | 33% | 58% | 35% | 83 / 103 / 54 |
| | e1000 | 46.2 | 44.4 | 29% | 33% | 58% | 34% | 82 / 101 / 57 |
| | e2000 | 46.6 | 44.6 | 32% | 33% | 58% | 36% | 87 / 98 / 55 |
| Cả 12 | `combo` (null) | 63.3 | 68.8 | 42% | 36% | 64% | 44% | 216 / 178 / 95 |
| | e2000 | 62.4 | 65.6 | 52% | 58% | 48% | 53% | 258 / 157 / 74 |

Tracker hoàn hảo (B), tổng bàn giao / HOTA: s2 `combo` 67% / 69.2 → e2000 71% / 71.5; giữ riêng `combo` **49% / 60.6** → e0 42% / 56.7, e2000 44% / 56.5. Cùng chiều với A.

→ **Kết quả âm tính.** Chọn trên s2 thì e2000 thắng rõ (+14 điểm bàn giao, +2 HOTA). Trên nhóm giữ riêng: bàn giao chỉ +3 điểm, còn HOTA −7, IDF1 −10. Cặp không chồng lấn tăng (8% → 33%: đúng lỗi đã chẩn đoán), nhưng quay lại cùng camera giảm (86% → 58%): các lần nối cùng camera ĐÚNG (người bị che rồi hiện lại sau 1–3 s, hay gặp ở s3/s4 có 5–7 người) cũng mất ngân sách rộng và bị tách. s2 (2–4 người, ít che) chủ yếu lỗi gộp nhầm; nhóm giữ riêng (đông, che nhiều) cần nối cùng camera. Khoảng hở thời gian một mình không phân biệt được "người khác vào cửa" với "cùng người hiện lại".

**Bản sửa `pop_former_owner` trên A `combo`, 12 buổi** (`eval_v2` → `eval_v3`, cùng GT, cùng fixture): HOTA 62.0 → 63.3, IDF1 66.7 → 68.8, tổng bàn giao 47% → 44% (chồng lấn 41 → 42, không chồng lấn 45 → 36, quay lại 70 → 64). Mức giảm bàn giao nằm trong nhiễu giữa các lần chạy đã đo ở phiên 40 (±9 điểm ở cặp không chồng lấn), nên chưa kết luận được về bàn giao; HOTA/IDF1 và thời gian tới Global ID thì cải thiện.

## Vướng mắc / chưa xong

- **Lỗi gộp nhầm qua nối cùng camera vẫn chưa sửa được**: `evidence_max_gap_ms` âm tính trên nhóm giữ riêng. Cần một tín hiệu phân biệt "người mới vào" với "cùng người hiện lại" mà khoảng hở thời gian không có.
- **Số của phiên 40 không còn tái lập từng bit** trên code mới (bản sửa `pop_former_owner`). Báo cáo phải dùng số chạy lại trên code này.
- Thời gian tới Global ID mới đo theo thời gian dữ liệu; số chính thức cần đồng hồ thật trên GPU (`vast_lab.sh`, phải hỏi người dùng trước khi thuê).
- Còn lại cho người kiểm nhãn: s3_2 X02 cam01 khung 0–68, X04 cam02 166–201, X05 cam01 219–220 (người ở cuối hành lang, quá nhỏ để nhận); "có thể sót người" chưa vẽ hộp: s4_1 cam03 khung 327–373 (X03), s3_2 cam01 159–203, s1_1 cam03 442–477.
- Test chạy bằng Python 3.12, chưa bằng 3.10 (quy tắc bất biến 4). Code mới không dùng gì ngoài 3.10 (`itertools.pairwise`, `int | None` dưới `from __future__ import annotations`), nhưng chưa kiểm bằng máy.
- Các script phân tích ở scratchpad (`diag_decisions.py`, `check_gt.py`, `id_spans.py`, `ttid_datatime.py`, `run_oracle.sh`, `sweep_evidence.sh`) chưa vào repo. Nên đưa `diag_decisions.py` và `ttid_datatime.py` vào `eval/` nếu dùng số của chúng trong báo cáo.
- Git trên máy này đã chạy được: người dùng thêm `safe.directory`; danh tính commit đặt giống commit cũ (Trần Quang Thành, noreply `thanhthanhhp123`); Git Credential Manager đã lưu đăng nhập GitHub (`git credential-manager github login`; chỉ đăng nhập trang github.com thì không đủ, push treo ở `credential-manager get`). Phiên 40–41 đã fast-forward vào `main` và push (`cff10cf..173178f`, kèm 2 commit phiên 38–40 trước đó chưa push). `eval/gt/`, `logs/` vẫn chưa quyết commit hay ignore.
- Vẫn còn từ phiên 40: khóa API Vast.ai đã lộ trong chat, chưa tạo lại; cặp cam02 ↔ cam03 chưa đo thời gian đi (3–30 s giữ chỗ).

## Bước tiếp theo

1. Thử tín hiệu phân biệt nối cùng camera đúng/sai, chọn trên s2 và kiểm trên nhóm giữ riêng như phiên này:
   (a) **có ứng viên khác camera khả thi** cho tracklet thì ô nối cùng camera không được hưởng `max_cost_geometric` (chỉ mất ngân sách khi có cạnh tranh, giữ nguyên ca người bị che hiện lại một mình);
   (b) **vị trí xuất hiện**: tracklet bắt đầu ở mép/cửa của camera là người mới tới, bắt đầu giữa khung là người hiện lại.
   Chạy `diag_decisions.py` (scratchpad, nên đưa vào `eval/`) trên biến thể mới để xem gộp nhầm cùng camera còn bao nhiêu.
2. Chạy lại `lab-eval` 12 buổi trên code hiện tại → số mới cho chương 6 (số phiên 40 không còn tái lập từng bit), tính Wilson theo số bàn giao.
3. Lỗi còn lại, theo thứ tự lớn: gộp nhầm ở cặp không chồng lấn và chồng lấn, tách vì hình học ở cặp chồng lấn (sai số homography), tracker đơn camera (~1/4 phần mất).
4. Khi thuê GPU lần tới: đo lại thời gian tới Global ID bằng đồng hồ thật trên code mới.
