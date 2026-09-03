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

## M2 — fine-tune YOLO cho person detection

```bash
# [HEAD] tải weight gốc + dataset. Tốn ~20G, kiểm tra df trước.
ssh ut-hpc '~/mct/env/bin/python ~/mct/jobs/fetch_pretrained.py'
ssh ut-hpc 'df -h $HOME | tail -1'
ssh ut-hpc '~/mct/env/bin/python ~/mct/jobs/prepare_coco_person.py'

# [JOB]
bash $S submit train_yolo.sbatch
bash $S watch <jobid>

# về máy dev
bash $S fetch mct/models/detector/yolo11s_person.onnx models/detector/yolo11s_person.onnx
```

Điều chỉnh qua biến môi trường trong `sbatch --export`, hoặc sửa thẳng trong file rồi `push` lại:
`MODEL`, `DATA`, `EPOCHS`, `IMGSZ`, `BATCH`, `NAME`.

VRAM 46–48G nên `BATCH=32` ở `imgsz=640` là thoải mái; muốn nhanh hơn thì `--gres=gpu:2`
và `device=0,1`.

**Ghi lại cho báo cáo (chương 6):** GPU nào (A40 hay L40), số GPU, batch, imgsz, epoch,
thời gian train, mAP50-95 trên val, kích thước `.onnx`. Không có mấy số này thì lần train
đó coi như bỏ.

Bước tiếp theo **không** làm ở đây: chuyển `.onnx` → TensorRT engine. Việc đó phải làm trên
đúng máy sẽ chạy pipeline (`vast-gpu`), vì engine gắn chặt với driver/GPU/phiên bản TensorRT.

---

## M3 — fine-tune Re-ID (OSNet)

```bash
# [HEAD]
ssh ut-hpc '~/mct/env/bin/pip install torchreid gdown'
# Dataset Market-1501 / MSMT17 phải tự tải (licence). Đặt đúng layout torchreid:
#   ~/mct/data/reid/market1501/Market-1501-v15.09.15/{bounding_box_train,bounding_box_test,query}
# Nếu link chết, rsync từ máy dev lên:
#   rsync -av data/market1501/ ut-hpc:mct/data/reid/market1501/

# [JOB]
bash $S submit train_reid.sbatch
bash $S watch <jobid>

bash $S fetch mct/models/reid/osnet_x1_0_ft.onnx models/reid/osnet_x1_0_ft.onnx
```

**Ràng buộc phải khớp với phần còn lại của hệ thống:**
- Input ONNX `1×3×256×128` NCHW, chuẩn hoá ImageNet — đúng như `src/tools/reid_onnx.py`
  đang làm ở máy dev. Đổi kích thước ở đây là breaking change cho cả hai phía.
- Output 512-d (osnet_x1_0). `embed_dim` phải khớp header message
  (`src/common/schema.py`, CLAUDE.md §5). Model re-id kèm DeepStream là 256-d — khác model,
  khác `embed_dim`, đừng lẫn.
- Embedding **L2-normalize tại producer**, không phải trong ONNX.

Sau khi có `.onnx` mới: chạy lại `make wildtrack-fixture` với `REID_ONNX=` trỏ vào file mới
để so chất lượng liên kết trước/sau fine-tune. Đó chính là số liệu cho chương 6.

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

Luôn kết thúc bằng `bash $S status` — `squeue` của mình phải rỗng.
