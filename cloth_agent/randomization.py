"""Single-gripper garment randomization planning for the Viser console."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any

from .config import ExperimentConfig, RobotConfig


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _normalized_yaw(yaw_deg: float) -> float:
    return (yaw_deg + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class GarmentRandomizationPlan:
    """Reviewable waypoints for a slow gather, twist, and low-air release."""

    seed: int
    center_x_mm: float
    center_y_mm: float
    grasp_z_mm: float
    approach_z_mm: float
    gather_lift_z_mm: float
    drag_x_mm: float
    drag_y_mm: float
    drop_x_mm: float
    drop_y_mm: float
    release_z_mm: float
    base_yaw_deg: float
    twist_yaw_deg: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def waypoint_rows(self) -> list[tuple[str, float, float, float, float]]:
        return [
            ("approach", self.center_x_mm, self.center_y_mm, self.approach_z_mm, self.base_yaw_deg),
            ("grasp", self.center_x_mm, self.center_y_mm, self.grasp_z_mm, self.base_yaw_deg),
            ("lift", self.center_x_mm, self.center_y_mm, self.gather_lift_z_mm, self.base_yaw_deg),
            ("drag", self.drag_x_mm, self.drag_y_mm, self.gather_lift_z_mm, self.base_yaw_deg),
            ("twist_to_drop", self.drop_x_mm, self.drop_y_mm, self.gather_lift_z_mm, self.twist_yaw_deg),
            ("release", self.drop_x_mm, self.drop_y_mm, self.release_z_mm, self.twist_yaw_deg),
            ("retreat", self.drop_x_mm, self.drop_y_mm, self.gather_lift_z_mm, self.twist_yaw_deg),
        ]


def build_garment_randomization_plan(
    experiment: ExperimentConfig,
    robot: RobotConfig,
    *,
    seed: int,
) -> GarmentRandomizationPlan:
    """Build one deterministic, bounds-checked randomization plan.

    The grasp remains at the validated garment point. Randomness is applied to
    the inward drag direction, drop location, and wrist-yaw twist. The cloth is
    intentionally not required to clear the table before the drag: remaining
    contact and friction gather the garment into folds before the low-air drop.
    """

    x, y, grasp_z, approach_z, _, yaw = experiment.require_ready()
    bounds = robot.boundaries
    if None in (bounds.x_min, bounds.y_min, bounds.y_max, bounds.z_min, bounds.z_max):
        raise ValueError("garment randomization requires complete local XYZ safety bounds")

    margin = float(robot.workspace_margin_mm)
    x_low = float(bounds.x_min + margin)  # type: ignore[operator]
    x_high = (
        float(bounds.x_max - margin) if bounds.x_max is not None else math.inf
    )
    y_low = float(bounds.y_min + margin)  # type: ignore[operator]
    y_high = float(bounds.y_max - margin)  # type: ignore[operator]
    z_low = float(bounds.z_min + robot.lower_z_margin_mm)  # type: ignore[operator]
    z_high = float(bounds.z_max - margin)  # type: ignore[operator]
    if not (x_low <= x <= x_high and y_low <= y <= y_high):
        raise ValueError("validated garment point is outside the randomization workspace")

    rng = random.Random(int(seed))
    gather_lift_z = min(max(approach_z, grasp_z + 140.0), z_high)
    if gather_lift_z < grasp_z + 80.0:
        raise ValueError("available vertical range is too small for garment randomization")

    requested_drag_x = x - rng.uniform(100.0, 140.0)
    drag_x = _clamp(requested_drag_x, x_low, x_high)
    y_direction = -1.0 if rng.random() < 0.5 else 1.0
    drag_y = _clamp(y + y_direction * rng.uniform(35.0, 70.0), y_low, y_high)
    if math.hypot(drag_x - x, drag_y - y) < 50.0:
        raise ValueError("workspace leaves less than 50 mm for a useful gather drag")

    drop_x = _clamp(x + rng.uniform(-35.0, 20.0), x_low, x_high)
    drop_y = _clamp(y + rng.uniform(-35.0, 35.0), y_low, y_high)
    release_z = min(max(grasp_z + 55.0, z_low + 30.0), gather_lift_z - 20.0)
    twist_yaw = _normalized_yaw(yaw + (-90.0 if rng.random() < 0.5 else 90.0))

    plan = GarmentRandomizationPlan(
        seed=int(seed),
        center_x_mm=float(x),
        center_y_mm=float(y),
        grasp_z_mm=float(grasp_z),
        approach_z_mm=float(approach_z),
        gather_lift_z_mm=float(gather_lift_z),
        drag_x_mm=float(drag_x),
        drag_y_mm=float(drag_y),
        drop_x_mm=float(drop_x),
        drop_y_mm=float(drop_y),
        release_z_mm=float(release_z),
        base_yaw_deg=float(yaw),
        twist_yaw_deg=float(twist_yaw),
    )
    for _, waypoint_x, waypoint_y, waypoint_z, _ in plan.waypoint_rows():
        bounds.validate(
            waypoint_x,
            waypoint_y,
            waypoint_z,
            robot.workspace_margin_mm,
            z_lower_margin_mm=robot.lower_z_margin_mm,
        )
    return plan


def garment_randomization_source(plan: GarmentRandomizationPlan) -> str:
    """Return restricted RobotAPI source for one randomization rollout."""

    return (
        "def run():\n"
        "    home()\n"
        "    open_gripper()\n"
        f"    move({plan.center_x_mm!r}, {plan.center_y_mm!r}, {plan.approach_z_mm!r}, {plan.base_yaw_deg!r})\n"
        f"    move({plan.center_x_mm!r}, {plan.center_y_mm!r}, {plan.grasp_z_mm!r}, {plan.base_yaw_deg!r})\n"
        "    close_gripper()\n"
        f"    move({plan.center_x_mm!r}, {plan.center_y_mm!r}, {plan.gather_lift_z_mm!r}, {plan.base_yaw_deg!r})\n"
        f"    move({plan.drag_x_mm!r}, {plan.drag_y_mm!r}, {plan.gather_lift_z_mm!r}, {plan.base_yaw_deg!r})\n"
        f"    move({plan.drop_x_mm!r}, {plan.drop_y_mm!r}, {plan.gather_lift_z_mm!r}, {plan.twist_yaw_deg!r})\n"
        f"    move({plan.drop_x_mm!r}, {plan.drop_y_mm!r}, {plan.release_z_mm!r}, {plan.twist_yaw_deg!r})\n"
        "    open_gripper()\n"
        f"    move({plan.drop_x_mm!r}, {plan.drop_y_mm!r}, {plan.gather_lift_z_mm!r}, {plan.twist_yaw_deg!r})\n"
        "    home()\n"
    )
