"""Hiệu chỉnh homography ảnh → mặt phẳng mặt đất cho từng camera.

Sinh ra các file `configs/cameras/homography/<cam_id>.yaml` mà `mct.homography.HomographyMapper`
đọc được. Hai nguồn cặp điểm tương ứng:

**(A) Thủ công** — cách dùng cho camera thật của đồ án (M6). Chọn ≥4 điểm nhìn thấy được
trên mặt đất (góc gạch, chân cột, vạch sơn), đo toạ độ mét của chúng bằng thước trên sơ đồ
mặt bằng, ghi vào một file YAML:

```yaml
plane: ground
image_size: [1920, 1080]
cameras:
  cam01:
    points:
      - {image: [412, 903], world: [0.00, 0.00]}
      - {image: [1385, 887], world: [4.20, 0.00]}
      - {image: [1102, 615], world: [4.20, 6.50]}
      - {image: [530, 622], world: [0.00, 6.50]}
```

    python -m tools.calibrate_homography --points configs/cameras/ground_points.yaml \\
        --out configs/cameras/homography

**(B) Từ chú thích WildTrack** — mỗi detection đã có sẵn *cả hai vế*: bbox trong ảnh và
`positionID` (ô lưới 2.5 cm trên mặt đất). Tức là hàng nghìn cặp điểm miễn phí, đủ để
hiệu chỉnh cả 7 camera mà không cần chạm vào file calibration OpenCV của dataset:

    python -m tools.calibrate_homography --wildtrack-dir data/wildtrack \\
        --out configs/cameras/homography/wildtrack

**Lọc điểm — phần quan trọng nhất của cách (B).** Điểm chân chỉ đúng khi *cả bàn chân lẫn
hai mép bbox* nằm trong khung: bbox bị cắt ở đáy thì `ymax` không còn là mặt đất, bị cắt ở
mép trái/phải thì tâm ngang lệch. Những detection đó bị loại khỏi tập hiệu chỉnh (vẫn còn
trong fixture — lúc chạy thật vẫn phải xử lý chúng, xem "Hạn chế").

**Hạn chế phải nhớ khi đọc số:** homography chỉ đúng với người *đứng trên mặt phẳng đã hiệu
chỉnh*. Người bị che nửa dưới, hoặc bbox chạm mép khung, cho điểm chân sai vài mét sau khi
chiếu → `max_ground_dist_m` có thể loại nhầm một cặp đúng. Đó là lý do `affinity.py` chỉ
*cộng* `λ · d_ground` chứ không dùng hình học một mình.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from common.logging import get_logger
from mct.homography import HomographyFit, HomographyMapper, estimate_homography

log = get_logger("tools.calibrate_homography")

Correspondences = tuple[list[tuple[float, float]], list[tuple[float, float]]]


def load_point_file(path: Path) -> tuple[dict[str, Correspondences], dict[str, Any]]:
    """Đọc file cặp điểm thủ công. Trả về ({cam_id: (điểm ảnh, điểm mét)}, siêu dữ liệu)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cameras = data.get("cameras") or {}
    if not cameras:
        raise ValueError(f"{path}: thiếu khối `cameras`")

    out: dict[str, Correspondences] = {}
    for cam_id, entry in cameras.items():
        points = (entry or {}).get("points") or []
        image_pts = [(float(p["image"][0]), float(p["image"][1])) for p in points]
        world_pts = [(float(p["world"][0]), float(p["world"][1])) for p in points]
        out[str(cam_id)] = (image_pts, world_pts)

    size = data.get("image_size")
    meta = {
        "plane": str(data.get("plane", "ground")),
        "image_size": (int(size[0]), int(size[1])) if size else None,
        "source": f"điểm đo tay từ {path.name}",
    }
    return out, meta


def wildtrack_correspondences(
    wildtrack_dir: Path,
    *,
    margin_px: float = 2.0,
    max_frames: int = 0,
) -> tuple[dict[str, Correspondences], dict[str, Any]]:
    """Rút cặp (điểm chân trong ảnh, vị trí mét) từ `annotations_positions/*.json`.

    `margin_px` — bbox phải cách mọi mép khung ít nhất chừng này mới được dùng; đó là
    cách nhận ra hộp bị cắt mà không cần đọc ảnh.
    """
    from tools.wildtrack_to_fixture import (
        WILDTRACK_H,
        WILDTRACK_W,
        cam_id_for_view,
        position_id_to_world_m,
    )

    ann_dir = wildtrack_dir / "annotations_positions"
    files = sorted(ann_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"{ann_dir} không có file chú thích nào")
    if max_frames > 0:
        files = files[:max_frames]

    image_by_cam: dict[str, list[tuple[float, float]]] = defaultdict(list)
    world_by_cam: dict[str, list[tuple[float, float]]] = defaultdict(list)
    n_seen = n_clipped = 0

    for path in files:
        for person in json.loads(path.read_text(encoding="utf-8")):
            world = position_id_to_world_m(int(person["positionID"]))
            for view in person["views"]:
                xmin, ymin = float(view["xmin"]), float(view["ymin"])
                xmax, ymax = float(view["xmax"]), float(view["ymax"])
                if -1 in (view["xmin"], view["ymin"], view["xmax"], view["ymax"]):
                    continue
                n_seen += 1
                # Bỏ hộp chạm mép: điểm chân của nó không nằm trên mặt đất thật.
                if (
                    xmin < margin_px
                    or xmax > WILDTRACK_W - margin_px
                    or ymax > WILDTRACK_H - margin_px
                    or ymax <= ymin
                ):
                    n_clipped += 1
                    continue
                cam_id = cam_id_for_view(int(view["viewNum"]))
                image_by_cam[cam_id].append(((xmin + xmax) / 2.0, ymax))
                world_by_cam[cam_id].append(world)

    log.info(
        "%d khung: %d detection, loại %d cái chạm mép khung (%.1f%%)",
        len(files),
        n_seen,
        n_clipped,
        100.0 * n_clipped / n_seen if n_seen else 0.0,
    )
    pairs = {cam: (image_by_cam[cam], world_by_cam[cam]) for cam in sorted(image_by_cam)}
    meta = {
        "plane": "wildtrack_ground",
        "image_size": (WILDTRACK_W, WILDTRACK_H),
        "source": "khớp từ chú thích WildTrack (đáy-giữa bbox ↔ positionID)",
    }
    return pairs, meta


def fit_all(
    pairs: dict[str, Correspondences], *, trim_ratio: float, trim_rounds: int
) -> dict[str, HomographyFit]:
    fits: dict[str, HomographyFit] = {}
    for cam_id, (image_pts, world_pts) in pairs.items():
        try:
            fit = estimate_homography(
                image_pts, world_pts, trim_ratio=trim_ratio, trim_rounds=trim_rounds
            )
        except ValueError as exc:
            log.warning("%s: bỏ qua — %s", cam_id, exc)
            continue
        fits[cam_id] = fit
        log.info("%s: %s", cam_id, fit.summary())
    return fits


def cross_camera_check(mapper: HomographyMapper, pairs: dict[str, Correspondences]) -> None:
    """Sai số *giữa* các camera — con số thật sự quyết định `max_ground_dist_m`.

    RMSE của từng camera chỉ nói homography khớp với chính dữ liệu của nó. Cái mà affinity
    dùng là khoảng cách giữa hai điểm chân của **hai camera khác nhau**, nên sai số ở đó
    tích luỹ từ cả hai phía. Ở đây ước lượng nó bằng cách chiếu mọi điểm hiệu chỉnh về mặt
    phẳng chung rồi xét độ lệch so với vị trí thật.
    """
    deviations: list[float] = []
    for cam_id, (image_pts, world_pts) in pairs.items():
        for img, world in zip(image_pts, world_pts, strict=True):
            projected = mapper.project(cam_id, img)
            if projected is not None:
                deviations.append(float(np.hypot(projected[0] - world[0], projected[1] - world[1])))
    if not deviations:
        return
    arr = np.array(deviations)
    # Hai camera độc lập, sai số cộng theo phương chiều dài → nhân sqrt(2) cho cặp.
    p50, p95 = np.percentile(arr, [50, 95])
    log.info(
        "sai số chiếu gộp mọi camera: trung vị %.2f m, p95 %.2f m "
        "→ cặp hai camera lệch cỡ %.2f m (p95), gợi ý max_ground_dist_m >= %.1f",
        p50,
        p95,
        p95 * np.sqrt(2),
        np.ceil(p95 * np.sqrt(2)),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--points", type=Path, help="file YAML cặp điểm đo tay (cách A)")
    src.add_argument("--wildtrack-dir", type=Path, help="thư mục WildTrack (cách B)")
    p.add_argument("--out", type=Path, required=True, help="thư mục ghi <cam_id>.yaml")
    p.add_argument("--trim-ratio", type=float, default=0.1, help="tỉ lệ điểm sai nhất bị cắt")
    p.add_argument("--trim-rounds", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=0, help="(WildTrack) 0 = mọi khung")
    p.add_argument("--margin-px", type=float, default=2.0, help="(WildTrack) mép an toàn của bbox")
    args = p.parse_args(argv)

    if args.points:
        pairs, meta = load_point_file(args.points)
    else:
        pairs, meta = wildtrack_correspondences(
            args.wildtrack_dir, margin_px=args.margin_px, max_frames=args.max_frames
        )

    fits = fit_all(pairs, trim_ratio=args.trim_ratio, trim_rounds=args.trim_rounds)
    if not fits:
        log.error("không hiệu chỉnh được camera nào")
        return 1

    mapper = HomographyMapper.from_fits(
        fits, plane=meta["plane"], image_size=meta["image_size"], source=meta["source"]
    )
    cross_camera_check(mapper, {cam: pairs[cam] for cam in fits})

    written = mapper.save(args.out)
    log.info("ghi %d file hiệu chỉnh vào %s", len(written), args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
