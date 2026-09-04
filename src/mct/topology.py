"""Đồ thị camera + ràng buộc thời gian di chuyển (CLAUDE.md §6 bước 2).

Đọc `configs/cameras/topology.yaml` và trả lời đúng một câu hỏi cho engine liên kết:
*một người thấy lần cuối ở camera `c'` lúc `t_last` có kịp xuất hiện ở camera `c` lúc `t`
hay không?* Cặp không thoả bị loại thẳng trước khi so ngoại hình — vừa rẻ hơn, vừa chặn
được kiểu nhầm mà đặc trưng Re-ID không tự chặn nổi (hai người mặc đồ giống nhau ở hai
đầu toà nhà, cách nhau 2 giây).

Ba quy ước, chọn theo hướng "thà nhận nhầm còn hơn cắt mất match đúng" như đã ghi trong
chính file topology.yaml — match sai còn bị ngưỡng ngoại hình chặn lại một lần nữa, còn
match đúng đã bị cắt thì không có đường quay lại:

1. **Cặp camera chồng lấn (`overlaps_with`) không bị ràng buộc chiều thời gian.** Cùng một
   người xuất hiện đồng thời ở hai camera, và message hai luồng tới lệch nhau vài trăm ms
   là chuyện thường — nên xét `|Δt|` chứ không xét dấu.

2. **Cặp chưa khai báo `transitions` được thả tự do** (`unknown_pair_policy: allow`).
   Lúc mới lắp camera chưa ai đo transit time; siết mặc định sẽ làm engine im lặng không
   khớp gì và rất khó truy. Đổi sang `reject` khi đã đo đủ, để dùng topology như một bộ
   lọc thật.

3. **Cùng một camera thì luôn thoả.** Người rời khung rồi quay lại là hợp lệ; việc chặn
   trùng lặp trong cùng camera là của ràng buộc loại trừ ở `gallery.py`, không phải ở đây.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from common.config import load_yaml

UnknownPairPolicy = Literal["allow", "reject"]

DEFAULT_TOPOLOGY_PATH = "configs/cameras/topology.yaml"


class TopologyError(ValueError):
    """topology.yaml không hợp lệ."""


@dataclass(slots=True, frozen=True)
class CameraSpec:
    cam_id: str
    name: str = ""
    resolution: tuple[int, int] = (1920, 1080)
    fps: int = 15
    overlaps_with: frozenset[str] = frozenset()


@dataclass(slots=True, frozen=True)
class Transition:
    """Khoảng thời gian di chuyển hợp lý từ camera này sang camera kia."""

    src: str
    dst: str
    min_ms: int
    max_ms: int
    distance_m: float | None = None

    def contains(self, elapsed_ms: float) -> bool:
        return self.min_ms <= elapsed_ms <= self.max_ms


@dataclass(slots=True, frozen=True)
class FeasibilityResult:
    """Kết quả kiểm tra, kèm lý do — để log được vì sao một ứng viên bị loại."""

    feasible: bool
    reason: str
    elapsed_ms: float
    transition: Transition | None = None

    def __bool__(self) -> bool:
        return self.feasible


@dataclass(slots=True)
class Topology:
    """Đồ thị camera đã nạp từ YAML."""

    cameras: dict[str, CameraSpec] = field(default_factory=dict)
    transitions: dict[tuple[str, str], Transition] = field(default_factory=dict)
    unknown_pair_policy: UnknownPairPolicy = "allow"

    # ------------------------------------------------------------------ nạp cấu hình

    @classmethod
    def load(cls, path: str | Path = DEFAULT_TOPOLOGY_PATH) -> Topology:
        return cls.from_mapping(load_yaml(path))

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Topology:
        cameras: dict[str, CameraSpec] = {}
        for cam_id, spec in (data.get("cameras") or {}).items():
            spec = spec or {}
            resolution = spec.get("resolution") or [1920, 1080]
            cameras[str(cam_id)] = CameraSpec(
                cam_id=str(cam_id),
                name=str(spec.get("name", "")),
                resolution=(int(resolution[0]), int(resolution[1])),
                fps=int(spec.get("fps", 15)),
                overlaps_with=frozenset(str(c) for c in (spec.get("overlaps_with") or [])),
            )

        policy = str(data.get("unknown_pair_policy", "allow"))
        if policy not in ("allow", "reject"):
            raise TopologyError(
                f"unknown_pair_policy phải là 'allow' hoặc 'reject', nhận {policy!r}"
            )

        topo = cls(cameras=cameras, unknown_pair_policy=policy)  # type: ignore[arg-type]

        for item in data.get("transitions") or []:
            src, dst = str(item["from"]), str(item["to"])
            for cam in (src, dst):
                if cam not in cameras:
                    raise TopologyError(
                        f"transitions nhắc tới camera '{cam}' không có trong khối `cameras`"
                    )
            min_ms, max_ms = int(item.get("min_ms", 0)), int(item["max_ms"])
            if min_ms < 0 or max_ms < min_ms:
                raise TopologyError(
                    f"{src}->{dst}: khoảng [{min_ms}, {max_ms}] ms không hợp lệ "
                    "(cần 0 <= min <= max)"
                )
            distance = item.get("distance_m")
            transition = Transition(
                src=src,
                dst=dst,
                min_ms=min_ms,
                max_ms=max_ms,
                distance_m=None if distance is None else float(distance),
            )
            topo.transitions[(src, dst)] = transition
            if bool(item.get("bidirectional", False)):
                topo.transitions[(dst, src)] = Transition(
                    src=dst,
                    dst=src,
                    min_ms=min_ms,
                    max_ms=max_ms,
                    distance_m=transition.distance_m,
                )

        topo._check_overlaps_symmetric()
        return topo

    def _check_overlaps_symmetric(self) -> None:
        """`overlaps_with` phải khai hai chiều — khai một chiều gần như luôn là quên.

        Bắt sớm ở đây vì hậu quả của nó (ràng buộc thời gian áp một chiều, một chiều
        không) rất khó nhận ra khi nhìn kết quả gán.
        """
        for cam in self.cameras.values():
            for other_id in cam.overlaps_with:
                other = self.cameras.get(other_id)
                if other is None:
                    raise TopologyError(
                        f"{cam.cam_id}.overlaps_with nhắc tới camera '{other_id}' chưa khai báo"
                    )
                if cam.cam_id not in other.overlaps_with:
                    raise TopologyError(
                        f"overlaps_with không đối xứng: {cam.cam_id} khai chồng lấn với "
                        f"{other_id}, nhưng {other_id} thì không khai ngược lại"
                    )

    # ------------------------------------------------------------------ truy vấn

    def __contains__(self, cam_id: object) -> bool:
        return cam_id in self.cameras

    def __len__(self) -> int:
        return len(self.cameras)

    @property
    def camera_ids(self) -> list[str]:
        return sorted(self.cameras)

    def is_overlapping(self, cam_a: str, cam_b: str) -> bool:
        spec = self.cameras.get(cam_a)
        return spec is not None and cam_b in spec.overlaps_with

    def transition(self, src: str, dst: str) -> Transition | None:
        return self.transitions.get((src, dst))

    def distance_m(self, src: str, dst: str) -> float | None:
        transition = self.transitions.get((src, dst))
        return None if transition is None else transition.distance_m

    def neighbours(self, cam_id: str) -> list[str]:
        """Các camera đi tới được trực tiếp từ đây (theo `transitions`)."""
        return sorted(dst for (src, dst) in self.transitions if src == cam_id)

    # ------------------------------------------------------------------ ràng buộc

    def check(self, src: str, dst: str, elapsed_ms: float) -> FeasibilityResult:
        """`elapsed_ms` = thời gian từ lần cuối thấy ở `src` tới lúc xuất hiện ở `dst`."""
        for cam in (src, dst):
            if cam not in self.cameras:
                raise TopologyError(f"camera '{cam}' không có trong topology")

        if src == dst:
            return FeasibilityResult(True, "cùng camera, không ràng buộc di chuyển", elapsed_ms)

        if self.is_overlapping(src, dst):
            # Chồng lấn: hai camera thấy cùng lúc, message tới lệch nhau cũng bình thường.
            transition = self.transition(src, dst)
            if transition is None:
                return FeasibilityResult(
                    True, "cặp chồng lấn, không giới hạn thời gian", elapsed_ms
                )
            ok = abs(elapsed_ms) <= transition.max_ms
            return FeasibilityResult(
                ok,
                "cặp chồng lấn trong cửa sổ cho phép"
                if ok
                else f"cặp chồng lấn nhưng lệch {abs(elapsed_ms):.0f}ms > {transition.max_ms}ms",
                elapsed_ms,
                transition,
            )

        transition = self.transition(src, dst)
        if transition is None:
            allow = self.unknown_pair_policy == "allow"
            return FeasibilityResult(
                allow,
                f"chưa khai báo transit {src}->{dst} (policy={self.unknown_pair_policy})",
                elapsed_ms,
            )

        if transition.contains(elapsed_ms):
            return FeasibilityResult(
                True, "trong khoảng transit đã khai báo", elapsed_ms, transition
            )

        if elapsed_ms < transition.min_ms:
            reason = (
                f"quá nhanh: {elapsed_ms:.0f}ms < {transition.min_ms}ms "
                f"(không kịp đi từ {src} sang {dst})"
            )
        else:
            reason = f"quá lâu: {elapsed_ms:.0f}ms > {transition.max_ms}ms"
        return FeasibilityResult(False, reason, elapsed_ms, transition)

    def is_feasible(self, src: str, dst: str, elapsed_ms: float) -> bool:
        return self.check(src, dst, elapsed_ms).feasible
