# Quy trình từng mốc trên ut-hpc

Đọc `cluster.md` trước. Ràng buộc chi phối mọi thứ dưới đây: **node tính toán không có mạng**.

Ký hiệu: `[HEAD]` = chạy trên head node qua `ssh ut-hpc`, có mạng, phải ngắn.
`[JOB]` = chạy trong `sbatch`, có GPU, không mạng.

---

## Lần đầu — dựng môi trường

```bash
S=.claude/skills/ut-hpc/scripts/hpc.sh
bash $S status                  # xác nhận kết nối + còn bao nhiêu dung lượng home
bash $S setup                   # [HEAD] tạo ~/mct/* và conda env "mct" (~10 phút, ~10G)
bash $S push                    # đẩy templates lên ~/mct/jobs/
bash $S submit smoke_gpu.sbatch # [JOB] xác nhận env chạm được GPU
bash $S watch <jobid>
```

`smoke_gpu.sbatch` phải in `cuda_available True` + tên GPU. Nếu không: xem "Chẩn đoán" cuối file.

Env dùng cho đồ án là `~/mct/env` — **đừng** dùng chung với `fitroom`/`captcha` của người dùng.

---

## M2 / M3 — KHÔNG fine-tune nữa (chốt 2026-09-04)

Hai mốc này từng là "fine-tune YOLO trên COCO-person" và "fine-tune OSNet trên
Market-1501/MSMT17". **Đã bỏ cả hai.** Weight pretrained đã được huấn luyện trên chính
những bộ đó — YOLO11s trên COCO (có sẵn lớp `person`), OSNet của torchreid trên
Market-1501/MSMT17 — nên fine-tune lại là học lại dữ liệu cũ: không uplift đáng kể, tốn
GPU-hour và ~50G đĩa cụm.

Việc thay thế, **không cần cụm**:

- **M2 (detector):** dùng thẳng `yolo11s.pt` → ONNX, lọc lớp person bằng khối
  `[class-attrs-*]` trong `configs/pipeline/config_infer_yolo11*.txt`. Bước
  `.onnx` → TensorRT engine vẫn phải làm trên `vast-gpu` (engine gắn chặt với
  driver/GPU/phiên bản TensorRT), không làm ở đây.
- **M3 (Re-ID):** dùng OSNet pretrained, **ưu tiên checkpoint đa nguồn/khái quát hoá miền**
  thay vì bản Market-1501 chuyên biệt — đo được trên WildTrack: +25% F1 (0.346 vs 0.277).

Xem `docs/worklog/2026-09-04-7-m2-detector-pretrained.md` và CLAUDE.md §9.

---

## M6 — fine-tune trên dữ liệu tự thu (kịch bản DUY NHẤT còn dùng cụm để train)

Chỉ làm khi **đã đo được domain gap thật**: chạy model pretrained trên dữ liệu lab, thấy
mAP (detector) hoặc F1 liên kết (Re-ID) tụt rõ so với trên WildTrack. Không đo được gap
thì không train — và phải trình bày dưới dạng **ablation có/không fine-tune**, không phải
một bước bắt buộc của pipeline.

```bash
S=.claude/skills/ut-hpc/scripts/hpc.sh

# [HEAD] đẩy dữ liệu lab đã gán nhãn (CVAT -> YOLO/torchreid layout) lên cụm.
rsync -av data/lab/ ut-hpc:mct/data/lab/
ssh ut-hpc 'df -h $HOME | tail -1'

# [JOB] — cả hai template đều nhận DATA qua biến môi trường.
bash $S push
bash $S submit train_yolo.sbatch     # DATA=$ROOT/data/lab/detect/lab.yaml
bash $S submit train_reid.sbatch     # DATA=$ROOT/data/lab/reid
bash $S watch <jobid>

bash $S fetch mct/models/detector/yolo11s_lab.onnx models/detector/yolo11s_lab.onnx
```

Điều chỉnh qua `sbatch --export`, hoặc sửa thẳng trong file rồi `push` lại:
`MODEL`, `DATA`, `EPOCHS`, `IMGSZ`, `BATCH`, `NAME`.

VRAM 46–48G nên `BATCH=32` ở `imgsz=640` là thoải mái; muốn nhanh hơn thì `--gres=gpu:2`
và `device=0,1`.

**Ràng buộc Re-ID phải khớp với phần còn lại của hệ thống:**
- Input ONNX `1×3×256×128` NCHW, chuẩn hoá ImageNet — đúng như `src/tools/reid_onnx.py`
  đang làm ở máy dev. Đổi kích thước ở đây là breaking change cho cả hai phía.
- Output 512-d (osnet_x1_0). `embed_dim` phải khớp header message
  (`src/common/schema.py`, CLAUDE.md §5). Model re-id kèm DeepStream là 256-d — khác model,
  khác `embed_dim`, đừng lẫn.
- Embedding **L2-normalize tại producer**, không phải trong ONNX.

**Cách đo ablation:** chạy lại `make wildtrack-fixture` (hoặc bộ chuyển dữ liệu lab) với
`REID_ONNX=` trỏ lần lượt vào bản pretrained và bản fine-tune, rồi so F1 liên kết. Ghi cả
hai cột vào chương 6 — chênh lệch đó *chính là* kết quả của thí nghiệm, kể cả khi nó bằng 0.

**Ghi lại cho báo cáo (chương 6):** GPU nào (A40 hay L40), số GPU, batch, imgsz, epoch,
thời gian train, mAP50-95 / rank-1, kích thước `.onnx`. Không có mấy số này thì lần train
đó coi như bỏ.

---

## Chẩn đoán

| Triệu chứng | Nguyên nhân gần như chắc chắn |
|---|---|
| Job `PD` mãi, reason `(Resources)`/`(Priority)` | Đang dùng `--partition=students` (1 node). Đổi sang `main-gpu`. |
| Job chạy rồi treo tới hết `--time` | Có lệnh chạm mạng trong job (pip, tự tải weight/dataset). Node tính toán không có mạng. |
| `module: command not found` | Shell non-interactive. Trong sbatch: `source /etc/profile.d/modules.sh`. Qua ssh: `ssh ut-hpc 'bash -lc "..."'`. |
| `nvidia-smi` treo | Đang chạy trên head node. Head node không có GPU. |
| `srun` treo rồi mất job | `srun` foreground chết theo phiên SSH. Dùng `sbatch`. |
| `No space left on device` | Home 90% đầy. `df -h $HOME`; `~/miniconda3` chiếm 40G nếu cần dọn. |
| `CUDA error: no kernel image` | Wheel torch quá cũ cho A40/L40. Cài lại bản `cu124` trở lên. |
| Ultralytics đòi tải `yolo11s.pt` lúc train | `MODEL=` trỏ sai đường dẫn tuyệt đối, hoặc chưa chạy `fetch_pretrained.py`. |
| `FATAL: thieu .../data/lab/...` | Chưa `rsync` dữ liệu lab lên cụm. Từ 2026-09-04 template không còn dataset benchmark mặc định. |

Luôn kết thúc bằng `bash $S status` — `squeue` của mình phải rỗng.
