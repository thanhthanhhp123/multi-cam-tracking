# 2026-09-28 — Đánh giá lại hướng đi: chẩn đoán gói 2, quét SOTA, định vị lại đóng góp

- **Mốc:** M4 + M6 (và định hướng toàn đồ án) | **Máy:** máy dev (Windows), không thuê GPU | **Thời lượng:** ~4h

## Mục tiêu phiên

- Bước đầu của "gói 2" (hợp nhất trên mặt phẳng mặt đất): đo xem tracklet bị ghép sai hay bị
  tách vì **ngoại hình** hay vì **hình học**, trước khi viết code hợp nhất.
- Người dùng hỏi: các bài gần đây dùng model nào để cân bằng tốc độ và độ chính xác, và **vì sao
  đồ án không dùng thẳng SOTA mà tự viết để ra số thấp**.
- Ghi lại một cách trung thực những quyết định trước đây đã sai, để trích vào chương 5.

## Đã làm

- `eval/diagnose_cost_split.py` (mới, +6 test ở `tests/test_diagnose_cost_split.py`): chạy engine online
  (`db_path=":memory:"`), gắn hàm gián điệp vào `associator.cost_matrix` để thu mọi ma trận chi
  phí. Với mỗi hàng có nhãn, công cụ lấy:
  - người thật của tracklet;
  - người của từng GlobalTrack, tính theo đa số thời lượng của các tracklet thành viên;
  - tách chi phí thành `app = 1 − similarity` và `geo = cost − app`;
  - chi phí rẻ nhì;
  - trạng thái của track đúng người (`none` / `infeasible` / `feasible`) và hạng của nó.

  Kết quả ở `data/s28/cost_split_r{1,2,3}.json` (gitignored).
- Quét tài liệu 2025–2026 cho từng tầng: detector, tracker, ReID, liên kết. Nguồn ở cuối file.

## Quyết định kỹ thuật

### 1. Tạm dừng gói 2 dạng "hợp nhất trên mặt đất": tiền đề yếu

Gói 2 giả định lỗi chính là **tách người** mà hình học có thể sửa. Số đo (bảng 1) cho thấy khác:
- Trong các lần ghép, lỗi chủ yếu là **gộp nhầm người mới vào track của người khác**: 62–82 ca mỗi
  lần chạy, trong khi chỉ 59–80 ca ghép đúng.
- Thành phần hình học tách ghép đúng/sai **kém hơn** ngoại hình: AUC 0.63–0.66 so với 0.67–0.72.
  Ở các ca gộp nhầm, `d_ground` trung vị chỉ ~0.8 m, so với ~0.48 m ở ca đúng. Trong đám đông
  WildTrack, 0.8 m không đủ để phân biệt người.

Viết thêm tầng hợp nhất hình học vì thế có trần thấp. Nếu làm tiếp liên kết, hướng được khuyên là
**cho phép sửa Global ID** trong một khoảng ngắn: liên kết lại muộn và gỡ xung đột. Đây cũng là
điều MV3DT và đội hạng 3 AI City 2025 làm (mục 3). **Chưa chốt**, chờ sau khi đổi detector
(quyết định 4), vì các ca gộp nhầm có thể đổi hẳn khi đầu vào sạch hơn.

### 2. Tự đánh giá: những quyết định trước đây đã sai

Câu hỏi của người dùng ("sao không dùng SOTA mà tự code để ra số thấp?") buộc phải nhìn lại cả
quá trình. Kết luận: **đề tài và kiến trúc không sai, nhưng có ba quyết định sai**.

**Sai:**
1. **Chốt detector mà không so sánh.**
   - M2 (phiên 7) chọn YOLO11s COCO 640 với lý do "pretrained, không fine-tune", nhưng không đặt
     nó cạnh một detector chuyên cho người.
   - Có ít nhất một ứng viên ngay trong hệ sinh thái DeepStream: PeopleNet / PeopleNet
     Transformer của NVIDIA.
   - Lý do "không fine-tune trên dữ liệu model đã thấy" vẫn đúng. Nhưng nó chỉ trả lời câu "có
     fine-tune không", chưa trả lời "chọn model nào".
2. **Không quét tài liệu và không dựng baseline có sẵn ngay từ đầu.**
   - Cho tới phiên này, đồ án chưa có một mốc so sánh chạy cùng giao thức.
   - Hệ quả: không ai nhận ra rằng trên chính DeepStream đã có một hệ MTMCT zero-shot đạt
     IDF1 ~96 trên WildTrack (MV3DT, mục 3).
3. **Tối ưu trước khi tách lỗi theo tầng.**
   - Phiên 17–20 tinh chỉnh `src/mct`.
   - Phải tới phiên 21 và 24 mới đo được rằng phần lớn điểm mất nằm **trước** module liên kết:
     hộp detector báo nhầm, và id-switch của tracker.
   - Phiên 26 xác nhận thêm (điểm mặt đất, trong vùng):
     - thay đầu vào bằng hộp GT + id GT, giữ nguyên engine: HOTA 31.1 → 63.8, IDF1 35.4 → 66.1;
     - thêm liên kết hoàn hảo: IDF1 89.3 (96.6 trên 40 khung test).

     Engine có lỗi thật (quyết định 1), nhưng chỗ nghẽn lớn nhất là đầu vào.

**Không sai (giữ nguyên):**
- Kiến trúc tách pipeline/engine qua Redis và fixture phát lại được. Nhờ nó mà mọi chẩn đoán trên
  chạy được không cần GPU, và mọi số đều tái lập được.
- Chọn liên kết đa camera làm đóng góp chính. Các hệ SOTA mạnh trên WildTrack đều **bắt buộc camera
  chồng lấn**, nên không giải được cặp không chồng lấn mà đề cương yêu cầu (mục 3).
- Kỷ luật đo: chính nó đã tự phát hiện fixture cũ là cận trên (phiên 11), độ trễ đo sai (phiên 28),
  và những cải tiến nằm trong nhiễu (phiên 22, 27).

**Bài học cho quy trình** (đã thêm vào `CLAUDE.md` §1):
- Tách lỗi theo tầng **trước** khi tối ưu tầng nào.
- Tầng nào không phải đóng góp thì dùng model/hệ tốt nhất có sẵn và so với nó.
- Quét tài liệu trước khi chốt một thành phần.

### 3. Định vị lại đóng góp: dùng SOTA ở những tầng không phải đóng góp

| Tầng | Hiện tại | Quyết định |
|---|---|---|
| Detector | YOLO11s COCO 640 | **Đổi**, ứng viên số 1: PeopleNet Transformer (quyết định 4) |
| Tracker đơn camera | NvDCF + ReID | Giữ. Không có bằng chứng đổi tracker đáng công bằng đổi detector |
| ReID | OSNet `osnet_x1_0_msdc_dg` | Giữ; CLIP-ReID làm ablation nếu còn thời gian (đo cả FPS) |
| Liên kết, camera chồng lấn | `src/mct` | Giữ, **so với MV3DT làm baseline**, nói rõ thua ở đâu và vì sao |
| Liên kết, camera không chồng lấn | `src/mct` | **Đóng góp chính.** SOTA overlap không làm được phần này |
| Hạ tầng (Redis, phát lại, dashboard, đo độ trễ) | tự viết | Giữ, là phần hệ thống của đồ án |

Luận điểm sẽ trình bày với thầy và hội đồng: *"Tầng nào có SOTA thì dùng SOTA. Đồ án tự làm phần
liên kết cho cấu hình camera hỗn hợp chồng lấn/không chồng lấn, chạy thời gian thực. Trên phần chỉ
có camera chồng lấn, đồ án so với MV3DT và phân tích khoảng cách."* **Chờ thầy chốt** (03–04/10):
định vị theo hướng này, hay ưu tiên tối đa độ chính xác trên WildTrack.

**Phương án bị loại:**
- **Dùng nguyên MV3DT làm sản phẩm.** Ba lý do:
  - không có phần tự thiết kế để bảo vệ;
  - không xử lý được camera không chồng lấn;
  - đòi DeepStream 8–9 (Ubuntu 24.04, driver ≥ 580), trong khi repo chốt DeepStream 7.1.
- **Dùng nguyên hệ của các đội AI City**: Co-DETR Swin + CLIP ViT + phân cụm offline không chạy
  được thời gian thực trên 3–4 luồng.
- **Tiếp tục tinh chỉnh `src/mct` trên đầu vào YOLO11s**: các đòn bẩy rẻ đã bị bác ở phiên 27, và
  quyết định 1 cho thấy trần thấp.

### 4. Detector: đổi sang detector chuyên cho người, bắt đầu từ PeopleNet Transformer

- **Vì sao chọn nó trước:** chạy thẳng trong `nvinfer`; là detector MV3DT dùng để đạt các số ở
  bảng 2; vẫn đúng quy tắc "pretrained, không fine-tune trên WildTrack".
- **Dự phòng:** RF-DETR, YOLO26. Hai model này tốt hơn YOLO11 trên đường cong tốc độ/độ chính xác
  COCO, nhưng vẫn là detector đa lớp.
- **Chưa xác minh:** PeopleNet Transformer có chạy được trên DeepStream 7.1 / TensorRT 10 không
  (định dạng model trên NGC, plugin Deformable DETR). Phải kiểm trước khi thuê máy.
- **Tiêu chí chấp nhận:** như phiên 25–26. Chạy n = 3, chấm cả hai giao thức (hộp ảnh toàn khung
  và điểm mặt đất trong vùng), kèm FPS/luồng trên 4 × 1080p. Tăng chỉ tính khi tách khỏi nhiễu.

## Số liệu đo được

### 1. Tách chi phí (WildTrack, pipeline thật YOLO11s 640, `configs/demo/wildtrack_ds.mct.yaml`, chỉ tracklet có nhãn)

| Lần chạy | Số lần ghép | Ghép đúng | Người mới bị gộp nhầm | Track đúng bị chặn | Track đúng khả thi nhưng thua track khác rẻ hơn | Vượt ngưỡng dù track đúng khả thi |
|---|---|---|---|---|---|---|
| r1 | 232 | 80 | 82 | 40 | 30 | 6 |
| r2 | 204 | 59 | 62 | 51 | 32 | 3 |
| r3 | 213 | 61 | 75 | 41 | 36 | 9 |

AUC tách ghép đúng/sai:

| Tín hiệu | AUC |
|---|---|
| Tổng chi phí | 0.71–0.73 |
| Ngoại hình | 0.67–0.72 |
| Hình học | 0.63–0.66 |
| Chênh lệch tốt nhất/tốt nhì | 0.61–0.63 |

Engine là tất định trên một fixture, nên khác biệt giữa r1, r2, r3 đến từ ba lần chạy pipeline.

### 2. Tài liệu: hệ gần bài toán nhất

Các số trong bảng lấy từ tóm tắt và bảng trong bản HTML của bài, **chưa kiểm lại giao thức chấm**.
Không so thẳng với số của đồ án cho tới khi kiểm xong.

| Hệ | Thành phần | WildTrack | Tốc độ | Học trên WildTrack? | Camera không chồng lấn? |
|---|---|---|---|---|---|
| MV3DT (NVIDIA, DS 8+) | PeopleNet Transformer + Kalman + ReID, liên kết 3 bước qua MQTT (gán, liên kết lại muộn, gỡ xung đột ID) | IDF1 96.5, MOTA 93.1 | 27 FPS | Không (chỉ cần hiệu chỉnh camera) | Không |
| TrackTacular | gộp BEV đầu-cuối | IDF1 95.3, MOTA 91.8, HOTA 68.2 | — | Có | Không |
| MVFlow | gộp BEV | IDF1 93.5, MOTA 91.3, HOTA 66.1 | — | Có | Không |
| EarlyBird | gộp BEV sớm | IDF1 92.3, MOTA 89.5, HOTA 64.5 | — | Có | Không |
| **Đồ án** (điểm mặt đất, trong vùng, 400 khung, T = 1 m) | YOLO11s 640 + NvDCF + OSNet DG + `src/mct` | IDF1 35.4, HOTA 31.1 | 175 FPS/luồng (RTX 3090, 4 × 1080p) | Không | Có (thiết kế) |
| Đồ án, LB (hộp GT + id GT + liên kết hoàn hảo, 40 khung test) | — | IDF1 96.6, HOTA 94.2 | — | — | — |

Dòng LB cho thấy bộ chấm điểm của đồ án có thể cho ra số cùng cỡ với tài liệu khi đầu vào đủ tốt.
Khoảng cách 35 → 96 vì vậy là khoảng cách của hệ thống, không phải của bộ chấm.

Các tầng khác:
- **AI City 2025, hạng 3:** Co-DETR Swin + Deep OC-SORT + CLIP-ReID. Riêng "MOT ID Consistency"
  (cho phép sửa/tách ID) đóng góp +20.46 HOTA trên tập validation 2D MTMC của họ.
- **AI City 2026, hạng 2 (Syn2RealTrack):** RF-DETR 2XL + ViTPose++ + KPR; không báo FPS.
- **ReID:** OSNet thường tổng quát hoá kém ngoài miền, CLIP-ReID tốt hơn nhiều nhưng ~35 lần số
  tham số. Đồ án dùng OSNet bản DG (huấn luyện đa nguồn), nên các số của OSNet thường không áp thẳng
  vào đây.
- **Detector** (T4, TensorRT FP16):
  - RF-DETR-L: 56.5 AP ở 6.8 ms;
  - YOLO11x: 50.9 AP ở 10.5 ms;
  - YOLO26 (01/2026, không cần NMS): n 40.9 AP ở 1.7 ms.

## Vướng mắc / chưa xong

- **Chưa xác minh** giao thức chấm WildTrack của MV3DT: tập khung, ngưỡng khoảng cách, có giới hạn
  vùng không. Phải đọc bảng gốc trong PDF trước khi đưa vào báo cáo.
- **Chưa xác minh** PeopleNet Transformer chạy được trên DeepStream 7.1 / TensorRT 10.
- Chờ thầy chốt định vị đồ án (quyết định 3) và định nghĩa độ trễ (phiên 28).
- Nhánh `s28-latency-tradeoff` chưa gộp về `main`.

## Bước tiếp theo

1. Trên máy dev, không cần GPU:
   - tải thẻ model PeopleNet Transformer từ NGC, xác định định dạng (ONNX hay etlt) và yêu cầu
     plugin TensorRT;
   - viết `configs/pipeline/config_infer_peoplenet_transformer.txt`;
   - kiểm bước hậu xử lý bbox (đầu ra DETR không cần NMS, lớp `person` là lớp 0?).
2. Thuê `vast-gpu` (**hỏi người dùng trước**, ~1 giờ): chạy WildTrack n = 3 với detector mới, chấm
   cả hai giao thức, đo FPS 4 luồng. So với YOLO11s 640 (HOTA hộp 15.70 ± 0.46, mặt đất 31.10 ± 0.85).
3. Đọc bản PDF của MV3DT, chép đúng giao thức chấm vào worklog.
4. Báo cáo thầy 03–04/10: mang bảng 2 và luận điểm ở quyết định 3.

## Nguồn

- [MV3DT — Fully Distributed Multi-View 3D Tracking in Real-Time](https://arxiv.org/html/2606.13127v1)
- [DeepStream — Multi-View 3D Tracking](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_MV3DT.html)
- [DeepStream 9.1 multi-camera 3D tracking (blog NVIDIA)](https://developer.nvidia.com/blog/build-a-multi-camera-3d-tracking-application-with-nvidia-deepstream-9-1-skills/)
- [AI City 2025, hạng 3](https://arxiv.org/html/2509.09946v1)
- [Syn2RealTrack, AI City 2026](https://arxiv.org/html/2608.24130)
- [Person Re-ID in 2025: What Works?](https://arxiv.org/html/2601.20598)
- [YOLO26](https://arxiv.org/html/2601.12882v2), [RF-DETR](https://arxiv.org/pdf/2511.09554)
- [PeopleNet Transformer (NGC)](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/peoplenet_transformer/-)
- [EarlyBird](https://arxiv.org/pdf/2310.13350), [TrackTacular](https://arxiv.org/pdf/2403.12573)
