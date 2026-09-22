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
    local_support_z_mm: float | None = None
    local_support_ring_valid: bool = False
    local_support_ring_elevation_mm: float | None = None
    support_layer_active: bool = False
    support_layer_confirmed: bool = False
    support_layer_activation_source: str | None = None
    support_floor_z_mm: float | None = None
    support_layer_type: str = "none"
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
    proposed_descent_mm: float | None = None,
) -> GraspHeightResolution:
    """Resolve one engaged grasp Z from one calibrated local-surface measurement.

    The measured surface is the observation. The commanded grasp TCP height is a
    model decision when proposed_descent_mm is supplied, validated without clamping.
    Without a proposal, legacy callers use the configured default. The target lies below
    the measured median surface by the configured amount, unless the robot lower
    bound or support/table safety floor makes that impossible. A configured
    sponge allowance is considered when the local support-ring diagnostic
    confirms an elevated compliant layer or the operator has explicitly
    confirmed the support layer in the robot configuration.
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
    support_layer_type = str(
        getattr(robot_config, "support_layer_type", "none")
    ).strip().lower()
    support_layer_thickness_mm = _finite_number(
        getattr(robot_config, "support_layer_thickness_mm", 0.0),
        "robot_config.support_layer_thickness_mm",
    )
    support_layer_press_mm = _finite_number(
        getattr(robot_config, "support_layer_press_mm", 0.0),
        "robot_config.support_layer_press_mm",
    )
    support_layer_max_compression_mm = _finite_number(
        getattr(robot_config, "support_layer_max_compression_mm", 0.0),
        "robot_config.support_layer_max_compression_mm",
    )
    support_layer_hard_clearance_mm = _finite_number(
        getattr(robot_config, "support_layer_hard_clearance_mm", 0.0),
        "robot_config.support_layer_hard_clearance_mm",
    )
    support_layer_presence_threshold_mm = _finite_number(
        getattr(robot_config, "support_layer_presence_threshold_mm", 3.0),
        "robot_config.support_layer_presence_threshold_mm",
    )
    support_layer_confirmed = bool(
        getattr(robot_config, "support_layer_confirmed", False)
    )
    if minimum_compression_mm <= 0 or maximum_compression_mm < minimum_compression_mm:
        raise GraspHeightError("configured grasp compression interval is invalid")
    if not minimum_compression_mm <= configured_compression_mm <= maximum_compression_mm:
        raise GraspHeightError(
            "configured grasp surface compression lies outside its allowed interval"
        )
    if table_clearance_mm < 0:
        raise GraspHeightError("configured grasp table clearance must be non-negative")
    if support_layer_type not in {"none", "sponge"}:
        raise GraspHeightError(
            "configured support layer type must be either 'none' or 'sponge'"
        )
    if any(
        value < 0
        for value in (
            support_layer_thickness_mm,
            support_layer_press_mm,
            support_layer_max_compression_mm,
            support_layer_hard_clearance_mm,
            support_layer_presence_threshold_mm,
        )
    ):
        raise GraspHeightError(
            "configured support layer values must be finite and non-negative"
        )
    if support_layer_type == "sponge":
        if support_layer_thickness_mm <= 0:
            raise GraspHeightError("sponge support layer thickness must be positive")
        if support_layer_press_mm <= 0:
            raise GraspHeightError("sponge support layer press depth must be positive")
        if support_layer_max_compression_mm < support_layer_press_mm:
            raise GraspHeightError(
                "sponge support layer maximum compression must be at least its press depth"
            )
    robot_lower_z_mm = float(bounds.z_min + robot_config.lower_z_margin_mm)
    if use_table_floor and math.isfinite(authoritative_table_z_mm):
        table_clearance_lower_z_mm = authoritative_table_z_mm + table_clearance_mm
        lower_z_mm = max(robot_lower_z_mm, table_clearance_lower_z_mm)
    else:
        # Do not derive a grasp floor from the live tabletop.  The controller's
        # configured z_min and final IK validation remain hard safety gates.
        table_clearance_lower_z_mm = float("nan")
        lower_z_mm = robot_lower_z_mm

    local_support_z_mm: float | None = None
    local_support_ring_valid = bool(
        measurement.get("local_support_ring_valid") is True
    )
    raw_support_z = measurement.get("local_support_z_median_mm")
    if raw_support_z is not None:
        local_support_z_mm = _finite_number(
            raw_support_z,
            "measurement.local_support_z_median_mm",
        )
    local_support_ring_elevation_mm: float | None = None
    raw_support_elevation = measurement.get("local_support_ring_elevation_mm")
    if raw_support_elevation is not None:
        local_support_ring_elevation_mm = _finite_number(
            raw_support_elevation,
            "measurement.local_support_ring_elevation_mm",
        )
    ring_support_confirmed = bool(
        local_support_ring_valid
        and local_support_z_mm is not None
        and local_support_ring_elevation_mm is not None
        and local_support_ring_elevation_mm >= support_layer_presence_threshold_mm
    )
    support_layer_active = bool(
        support_layer_type == "sponge"
        and (support_layer_confirmed or ring_support_confirmed)
    )
    support_layer_activation_source: str | None = None
    if support_layer_active:
        support_layer_activation_source = (
            "declared_configuration"
            if support_layer_confirmed
            else "local_support_ring"
        )
    support_floor_z_mm: float | None = None
    if support_layer_active and local_support_z_mm is not None:
        # The ring measures the compliant support's top surface.  Convert it to
        # a conservative hard-table floor, retaining the robot z_min as an
        # independent absolute safety gate.
        support_floor_z_mm = (
            local_support_z_mm
            - support_layer_thickness_mm
            + support_layer_hard_clearance_mm
        )
        # Once the local ring confirms the sponge, replace (rather than stack
        # on top of) the global tabletop floor.  The latter may be the exposed
        # hard table several centimetres away and must not veto a valid press
        # into the local compliant patch.
        lower_z_mm = max(robot_lower_z_mm, support_floor_z_mm)

    desired_compression_mm = configured_compression_mm
    effective_maximum_compression_mm = maximum_compression_mm
    if support_layer_active:
        effective_maximum_compression_mm = max(
            effective_maximum_compression_mm,
            support_layer_max_compression_mm,
        )
        desired_compression_mm = max(desired_compression_mm, support_layer_press_mm)
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
        effective_maximum_compression_mm,
    )

    if proposed_descent_mm is not None:
        desired_compression_mm = _finite_number(proposed_descent_mm, "proposed_descent_mm")
        if not minimum_compression_mm <= desired_compression_mm <= effective_maximum_compression_mm:
            raise GraspHeightError("Claude descent is outside configured geometric limits")
        if surface_z_mm - desired_compression_mm < lower_z_mm:
            raise GraspHeightError("Claude contact Z is below the allowed floor; no automatic clamp")
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
            f"support_floor_z={support_floor_z_mm!r}, "
            f"support_layer_active={support_layer_active}, "
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
        maximum_compression_mm=effective_maximum_compression_mm,
        local_surface_z_spread_mm=local_surface_z_spread_mm,
        local_support_z_mm=local_support_z_mm,
        local_support_ring_valid=local_support_ring_valid,
        local_support_ring_elevation_mm=local_support_ring_elevation_mm,
        support_layer_active=support_layer_active,
        support_layer_confirmed=support_layer_confirmed,
        support_layer_activation_source=support_layer_activation_source,
        support_floor_z_mm=support_floor_z_mm,
        support_layer_type=support_layer_type,
        policy=(
            "runtime_authoritative_surface_compression_with_local_sponge_support"
            if support_layer_active
            else (
                "runtime_authoritative_absolute_camera_surface_no_table_floor"
                if not use_table_floor
                else "runtime_authoritative_surface_compression"
            )
        ),
    )
