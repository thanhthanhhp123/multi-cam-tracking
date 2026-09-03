---
name: ut-hpc
description: Chạy việc train/fine-tune của đồ án MTMCT trên cụm SLURM ut-hpc (ĐH Twente) — fine-tune YOLO cho detection (M2), fine-tune OSNet cho Re-ID (M3), xuất ONNX và chuyển trọng số về models/. Dùng khi cần submit/theo dõi/huỷ job SLURM, chuẩn bị dataset hoặc môi trường Python trên ut-hpc, chẩn đoán job pending/failed, hoặc khi người dùng nhắc tới ut-hpc, hpc-head1, sbatch, srun, squeue, partition, GPU cluster, train, fine-tune.
---

# ut-hpc — cụm train/fine-tune của đồ án

`ut-hpc` = `hpc-head1.ewi.utwente.nl`, user `s3002152`, SLURM 21.08.5, Ubuntu 22.04.5.
Vai trò trong đồ án (CLAUDE.md §2): **chỉ train/fine-tune**. Không có DeepStream runtime,
**không bao giờ** chạy `src/ds_pipeline/` ở đây.

Sự thật cụ thể đã xác minh: `reference/cluster.md`.
Quy trình từng mốc (M2 YOLO, M3 Re-ID): `reference/workflows.md`.

## 5 điều sai là hỏng cả job — đọc trước khi viết bất kỳ script nào

1. **Node tính toán KHÔNG có Internet. Head node CÓ.**
   `pip install`, `yolo` tự tải weight/dataset, `torch.hub`, `singularity pull`, `wget` —
   mọi thứ chạm mạng phải làm **trên head node trước**, rồi job chỉ đọc từ đĩa.
   Một `sbatch` gọi `pip install` sẽ treo tới hết `--time` rồi chết. Đây là cạm bẫy số 1.
2. **Dùng `--partition=main-gpu`, không phải `students`.**
   `main-gpu` = 26 node GPU, `AllowAccounts=ALL`. `students` = đúng 1 node (`hpc-node08`)
   và hay pending. Đo hôm 2026-09-03: `main-gpu` chờ ~40s, `students` pending vô hạn định.
3. **Không chạy gì nặng trên head node.** Head node dùng chung, không có GPU.
   Việc cần GPU → `sbatch`. Việc chạm mạng (tải dataset, tạo env) → head node nhưng phải ngắn gọn.
4. **`sbatch` chứ không `srun` foreground.** `srun` treo khi hàng đợi bận và job tự huỷ
   khi mất kết nối SSH — dễ tưởng nhầm là lỗi cấu hình. `sbatch` + `squeue` là đường chuẩn.
5. **Trọng số không đi qua git.** Fine-tune xong → `.pt`/`.onnx` về `models/` (gitignored)
   ở máy dev, rồi mới scp sang `vast-gpu` khi chạy pipeline.

## Quy trình chuẩn

```
[1] Chuẩn bị (head node, CÓ mạng, một lần)   → scripts/hpc.sh setup
[2] Tải dataset (head node, CÓ mạng)          → scripts/hpc.sh prepare-<dataset>
[3] Đẩy job script lên                        → scripts/hpc.sh push
[4] Submit + theo dõi (SLURM)                 → scripts/hpc.sh submit <job> ; watch <id>
[5] Kéo trọng số về models/ (máy dev)         → scripts/hpc.sh fetch <remote> <local>
```

Mọi thứ của đồ án nằm dưới `~/mct/` trên cụm:

```
~/mct/
  env/        conda env "mct" (tạo bằng bước [1])
  data/       dataset đã tải sẵn (coco-person/, market1501/, msmt17/)
  jobs/       file .sbatch + script train đẩy lên từ templates/
  runs/       output của Ultralytics / torchreid
  models/     .pt và .onnx đã xuất — nguồn để fetch về máy dev
  logs/       slurm-<jobid>.out
```

## Lệnh nhanh

Script bọc sẵn: `.claude/skills/ut-hpc/scripts/hpc.sh` (chạy từ gốc repo, Git Bash trên Windows OK).

```bash
S=.claude/skills/ut-hpc/scripts/hpc.sh

bash $S status                     # kết nối, GPU rảnh, job của mình, dung lượng home
bash $S setup                      # tạo conda env "mct" trên head node (một lần, ~10 phút)
bash $S push                       # đẩy templates/ lên ~/mct/jobs/
bash $S submit train_yolo.sbatch   # sbatch -> in JOBID
bash $S watch 580546               # chờ xong rồi in log (poll squeue, không đoán thời gian)
bash $S logs 580546                # xem log job
bash $S cancel                     # huỷ TẤT CẢ job của mình (hoặc: cancel <jobid>)
bash $S fetch mct/models/best.onnx models/detector/yolo11s_person.onnx
bash $S shell                      # ssh vào head node
```

Không có script nào thì gọi thẳng: `ssh ut-hpc '<lệnh>'`. Luôn `-o BatchMode=yes` và
đặt `timeout` khi gọi từ tool để không treo phiên.

## Checklist trước khi bấm submit

- [ ] Mọi weight/dataset job cần đã **nằm sẵn trên đĩa** (`ls` xác nhận), không tải lúc chạy.
- [ ] `--partition=main-gpu`, `--gres=gpu:N`, `--time=` có giới hạn thật (đừng để mặc định vô hạn).
- [ ] `--output=` trỏ vào `~/mct/logs/`.
- [ ] Đường dẫn python là `~/mct/env/bin/python` (env đã cài sẵn), không phải `python3` hệ thống.
- [ ] Env biến `ULTRALYTICS_OFFLINE`/`HF_HUB_OFFLINE`/`YOLO_OFFLINE` bật, hoặc chắc chắn
      thư viện không tự gọi mạng.
- [ ] Job ghi kết quả vào `~/mct/runs/`, xuất model cuối vào `~/mct/models/`.

## Sau khi job xong — BẮT BUỘC

- `squeue -u $USER` phải rỗng. Không để job chạy quên (cụm dùng chung).
- Ghi **số đo** vào worklog kèm cấu hình: GPU (A40 hay L40), số GPU, batch size, epoch,
  thời gian, mAP/rank-1. CLAUDE.md §10 — số không tái lập được thì vô nghĩa.
- Trọng số về `models/` ở máy dev, **không commit**.

## Giới hạn cần nhớ

- **Home còn ~113G/1TB (2026-09-03).** COCO (~20G) + MSMT17 (~30G) + env torch (~10G) là vừa hết.
  Kiểm tra `df -h $HOME` trước mỗi lần tải dataset; `~/miniconda3` đang chiếm 40G nếu cần dọn.
- Chỉ `$HOME` là nơi ghi được **và chia sẻ giữa các node**. `/local` (2.2T) ghi được nhưng
  **riêng từng node** — chỉ dùng làm scratch trong một job, đừng để dữ liệu lâu dài ở đó.
- Không có Docker. Có `module load singularity/3.9.5` (đã test load được trên node tính toán),
  nhưng với đồ án này conda env đơn giản hơn — chỉ dùng Singularity nếu conda không đủ.
