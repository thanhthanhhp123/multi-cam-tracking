"""Vùng quan tâm (ROI) trên mặt sàn, theo từng camera: bỏ detection có điểm chân nằm ngoài.

Khai trong `topology.yaml`, ở khối của camera cần lọc:

```yaml
cameras:
  cam01:
    ground_roi: {x_m: [null, 6.0]}     # chỉ giữ người có chân ở x <= 6 m; null = không chặn
```

**Vì sao cần** (docs/worklog/2026-10-17-35-*, quyết định 4; phiên 36). Ở hiện trường M6,
cam01 nhìn dọc một hành lang thẳng và thấy tới tận vùng của cam03 ở xa 18 m. Chúc camera
xuống để giấu vùng đó thì cắt mất đầu người trong vùng chồng lấn với cam02: khi camera treo
cao hơn đầu người, đầu người ở mép xa vùng chồng lấn và chân người ở vùng cam03 nằm gần cùng
một tia nhìn (đo ở phiên 36). Nên camera giữ góc thấy trọn người, còn vùng xa thì lọc ở đây.
Một hệ thật biết vùng nó giám sát, nên đây là tính năng của hệ, không phải mẹo chấm điểm —
miễn là GROUND-TRUTH đi qua đúng bộ lọc này (`eval.run_lab_eval` làm việc đó).

**Lọc ở đâu.** Engine lọc ngay khi nhận khung (`Engine.feed`), TRƯỚC `TrackletBuilder`, nên
tracklet của người đi qua mốc bị cắt đúng tại mốc. Không lọc trong pipeline DeepStream: NvDCF
đã dựng track xong, và ROI cần homography, thứ chỉ có ở phía `src/mct`.

**Hộp không chiếu được** (suy biến, phía trên đường chân trời) bị bỏ khi camera có ROI —
cùng quy ước với `eval/roi_filter_fixture.py`. Camera không khai ROI thì không đụng tới.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from common.schema import Detection, FrameMessage
from mct.homography import HomographyMapper

Bounds = tuple[float | None, float | None]


class RoiError(ValueError):
    """Khai báo `ground_roi` không hợp lệ hoặc không dùng được."""


@dataclass(slots=True, frozen=True)
class GroundRoi:
    """Hình chữ nhật trên mặt sàn (mét), cạnh nào `None` thì không chặn phía đó.

    Nửa mở `[lo, hi)` như `eval.roi_filter_fixture.GroundRoi`, để một điểm nằm đúng trên
    mốc chung của hai ROI kề nhau chỉ thuộc về một bên.
    """

    x_m: Bounds = (None, None)
    y_m: Bounds = (None, None)

    def __post_init__(self) -> None:
        for name, (lo, hi) in (("x_m", self.x_m), ("y_m", self.y_m)):
            if lo is not None and hi is not None and not lo < hi:
                raise RoiError(f"ground_roi.{name}: cần min < max, nhận [{lo}, {hi}]")

    def contains(self, point: tuple[float, float]) -> bool:
        for value, (lo, hi) in zip(point, (self.x_m, self.y_m), strict=True):
            if lo is not None and value < lo:
                return False
            if hi is not None and value >= hi:
                return False
        return True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> GroundRoi:
        unknown = set(data) - {"x_m", "y_m"}
        if unknown:
            # Khoá gõ sai (`x:` thay cho `x_m:`) mà bỏ qua im lặng thì ROI thành "không chặn
            # gì" — đúng loại lỗi không triệu chứng CLAUDE.md §11 đã gặp với nvtracker.
            raise RoiError(f"ground_roi: khoá lạ {sorted(unknown)} (chỉ nhận x_m, y_m)")
        return cls(x_m=_bounds(data, "x_m"), y_m=_bounds(data, "y_m"))

    def to_mapping(self) -> dict[str, list[float | None]]:
        return {"x_m": list(self.x_m), "y_m": list(self.y_m)}


def _bounds(data: Mapping[str, Any], key: str) -> Bounds:
    raw = data.get(key)
    if raw is None:
        return (None, None)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise RoiError(f"ground_roi.{key}: cần [min, max] (null = không chặn), nhận {raw!r}")
    lo, hi = (None if v is None else float(v) for v in raw)
    return (lo, hi)


def foot_point(det: Detection) -> tuple[float, float]:
    """Điểm giữa cạnh đáy của bbox `[x, y, w, h]` — chỗ chân chạm sàn."""
    x, y, w, h = det.bbox
    return (x + w / 2.0, y + h)


@dataclass(slots=True)
class RoiFilter:
    """Lọc `FrameMessage` theo ROI của từng camera. Camera không có ROI đi qua nguyên vẹn."""

    rois: dict[str, GroundRoi]
    mapper: HomographyMapper

    def __post_init__(self) -> None:
        missing = sorted(c for c in self.rois if not self.mapper.has(c))
        if missing:
            # Không có homography thì không biết chân nằm đâu: lặng lẽ bỏ qua ROI nghĩa là
            # camera đó lại thấy cả vùng mà bố trí đã tính là "không thấy".
            raise RoiError(f"camera {missing} khai ground_roi nhưng chưa có file hiệu chỉnh")

    def filter(self, msg: FrameMessage) -> tuple[FrameMessage, int]:
        """Trả về message đã bỏ detection ngoài ROI, và số detection bị bỏ."""
        roi = self.rois.get(msg.cam_id)
        if roi is None or not msg.detections:
            return msg, 0
        kept = []
        for det in msg.detections:
            ground = self.mapper.project(msg.cam_id, foot_point(det))
            if ground is not None and roi.contains(ground):
                kept.append(det)
        n_dropped = len(msg.detections) - len(kept)
        if not n_dropped:
            return msg, 0
        return dataclasses.replace(msg, detections=kept), n_dropped

    @classmethod
    def from_rois(
        cls, rois: Mapping[str, GroundRoi], mapper: HomographyMapper | None
    ) -> RoiFilter | None:
        """`None` nếu không camera nào khai ROI. Có ROI mà không có homography là lỗi."""
        if not rois:
            return None
        if mapper is None:
            raise RoiError(f"camera {sorted(rois)} khai ground_roi nhưng không có homography")
        return cls(rois=dict(rois), mapper=mapper)
