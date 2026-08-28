"""Single authoritative conversion from measured garment surface to grasp TCP Z."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from .config import RobotConfig


class GraspHeightError(RuntimeError):
    """Raised when calibrated geometry cannot produce a legal engaged grasp Z."""


@dataclass(frozen=True)
class GraspHeightResolution:
    """Auditable result of the shared grasp-height policy."""

    surface_xyz_mm: tuple[float, float, float]
    target_xyz_mm: tuple[float, float, float]
    surface_z_mm: float
    local_table_z_mm: float | None
    local_table_source: str | None
    plane_table_z_mm: float
    authoritative_table_z_mm: float
    table_z_disagreement_mm: float | None
    robot_lower_z_mm: float
    table_clearance_lower_z_mm: float
    lower_z_mm: float
    desired_compression_mm: float
    achieved_compression_mm: float
    minimum_compression_mm: float
    maximum_compression_mm: float
    local_surface_z_spread_mm: float | None
    policy: str = "runtime_authoritative_surface_compression"
    valid: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraspHeightError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise GraspHeightError(f"{label} must be a finite number")
    return result


def _finite_xyz(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise GraspHeightError(f"{label} must contain exactly three finite values")
    return tuple(
        _finite_number(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def resolve_grasp_height(
    *,
    measurement: Mapping[str, Any],
    table_plane_abc: Sequence[float] | None,
    robot_config: RobotConfig,
) -> GraspHeightResolution:
    """Resolve one engaged grasp Z from one calibrated local-surface measurement.

    The measured surface is the observation. The commanded grasp TCP height is a
    runtime decision and is never delegated to a model. The target presses below
    the measured median surface by the configured amount, unless the robot lower
    bound or table-clearance floor makes that impossible.
    """

    if measurement.get("valid") is not True:
        raise GraspHeightError("grasp height requires a valid local-surface measurement")
    surface_xyz = _finite_xyz(
        measurement.get("base_xyz_median_mm"),
        "measurement.base_xyz_median_mm",
    )
    x_mm, y_mm, surface_z_mm = surface_xyz

    raw_local_table = measurement.get("table_z_median_mm")
    local_table_source: str | None = None
    if raw_local_table is not None:
        local_table_z_mm = _finite_number(
            raw_local_table,
            "measurement.table_z_median_mm",
        )
        local_table_source = "measurement.table_z_median_mm"
    else:
        raw_height = measurement.get("height_above_table_median_mm")
        if raw_height is not None:
            local_table_z_mm = surface_z_mm - _finite_number(
                raw_height,
                "measurement.height_above_table_median_mm",
            )
            local_table_source = "surface_z_minus_height_above_table_median"
        else:
            local_table_z_mm = None

    use_table_floor = bool(
        getattr(robot_config, "grasp_use_table_clearance_floor", True)
    )
    if table_plane_abc is None:
        if local_table_z_mm is None:
            # Absolute-camera mode deliberately does not require a tabletop
            # estimate.  Keep a finite diagnostic value so existing audit and
            # formatting code remains backwards compatible.
            if not use_table_floor:
                plane_table_z_mm = float("nan")
            else:
                raise GraspHeightError(
                    "grasp height requires either a fitted table plane or local table height"
                )
        else:
            plane_table_z_mm = local_table_z_mm
    else:
        plane = _finite_xyz(table_plane_abc, "table_plane_abc")
        plane_table_z_mm = plane[0] * x_mm + plane[1] * y_mm + plane[2]
    finite_table_values = [
        value
        for value in (
            plane_table_z_mm,
            local_table_z_mm,
        )
        if value is not None and math.isfinite(float(value))
    ]
    authoritative_table_z_mm = max(finite_table_values) if finite_table_values else float("nan")
    table_z_disagreement_mm = (
        abs(local_table_z_mm - plane_table_z_mm)
        if local_table_z_mm is not None and math.isfinite(float(plane_table_z_mm))
        else None
    )

    bounds = robot_config.boundaries
    if bounds.z_min is None:
        raise GraspHeightError("grasp height requires a configured robot z_min")
    configured_compression_mm = _finite_number(
        robot_config.grasp_surface_compression_mm,
        "robot_config.grasp_surface_compression_mm",
    )
    minimum_compression_mm = _finite_number(
        robot_config.grasp_min_compression_mm,
        "robot_config.grasp_min_compression_mm",
    )
    maximum_compression_mm = _finite_number(
        robot_config.grasp_max_compression_mm,
        "robot_config.grasp_max_compression_mm",
    )
    table_clearance_mm = _finite_number(
        robot_config.grasp_table_clearance_mm,
        "robot_config.grasp_table_clearance_mm",
    )
    if minimum_compression_mm <= 0 or maximum_compression_mm < minimum_compression_mm:
        raise GraspHeightError("configured grasp compression interval is invalid")
    if not minimum_compression_mm <= configured_compression_mm <= maximum_compression_mm:
        raise GraspHeightError(
            "configured grasp surface compression lies outside its allowed interval"
        )
    if table_clearance_mm < 0:
        raise GraspHeightError("configured grasp table clearance must be non-negative")
    robot_lower_z_mm = float(bounds.z_min + robot_config.lower_z_margin_mm)
    if use_table_floor and math.isfinite(authoritative_table_z_mm):
        table_clearance_lower_z_mm = authoritative_table_z_mm + table_clearance_mm
        lower_z_mm = max(robot_lower_z_mm, table_clearance_lower_z_mm)
    else:
        # Do not derive a grasp floor from the live tabletop.  The controller's
        # configured z_min and final IK validation remain hard safety gates.
        table_clearance_lower_z_mm = float("nan")
        lower_z_mm = robot_lower_z_mm

    desired_compression_mm = configured_compression_mm
    diagnostic = measurement.get("surface_shape_diagnostic")
    if isinstance(diagnostic, Mapping):
        raw_recommended = diagnostic.get("recommended_press_below_surface_mm")
        if raw_recommended is not None:
            recommended_compression_mm = _finite_number(
                raw_recommended,
                "surface_shape_diagnostic.recommended_press_below_surface_mm",
            )
            if recommended_compression_mm <= 0:
                raise GraspHeightError(
                    "surface diagnostic compression recommendation must be positive"
                )
            desired_compression_mm = max(
                desired_compression_mm,
                recommended_compression_mm,
            )
    desired_compression_mm = min(
        max(desired_compression_mm, minimum_compression_mm),
        maximum_compression_mm,
    )

    target_z_mm = max(surface_z_mm - desired_compression_mm, lower_z_mm)
    achieved_compression_mm = surface_z_mm - target_z_mm
    if achieved_compression_mm + 1e-9 < minimum_compression_mm:
        raise GraspHeightError(
            "no legal engaged grasp Z exists: "
            f"surface_z={surface_z_mm:.2f} mm, "
            f"local_table_z={local_table_z_mm!r} mm, "
            f"plane_table_z={plane_table_z_mm:.2f} mm, "
            f"robot_lower_z={robot_lower_z_mm:.2f} mm, "
            f"table_clearance_lower_z={table_clearance_lower_z_mm:.2f} mm, "
            f"required_compression>={minimum_compression_mm:.2f} mm"
        )

    raw_spread = measurement.get("base_z_p90_minus_p10_mm")
    local_surface_z_spread_mm = (
        _finite_number(raw_spread, "measurement.base_z_p90_minus_p10_mm")
        if raw_spread is not None
        else None
    )
    return GraspHeightResolution(
        surface_xyz_mm=surface_xyz,
        target_xyz_mm=(x_mm, y_mm, target_z_mm),
        surface_z_mm=surface_z_mm,
        local_table_z_mm=local_table_z_mm,
        local_table_source=local_table_source,
        plane_table_z_mm=plane_table_z_mm,
        authoritative_table_z_mm=authoritative_table_z_mm,
        table_z_disagreement_mm=table_z_disagreement_mm,
        robot_lower_z_mm=robot_lower_z_mm,
        table_clearance_lower_z_mm=table_clearance_lower_z_mm,
        lower_z_mm=lower_z_mm,
        desired_compression_mm=desired_compression_mm,
        achieved_compression_mm=achieved_compression_mm,
        minimum_compression_mm=minimum_compression_mm,
        maximum_compression_mm=maximum_compression_mm,
        local_surface_z_spread_mm=local_surface_z_spread_mm,
        policy=(
            "runtime_authoritative_absolute_camera_surface_no_table_floor"
            if not use_table_floor
            else "runtime_authoritative_surface_compression"
        ),
    )
