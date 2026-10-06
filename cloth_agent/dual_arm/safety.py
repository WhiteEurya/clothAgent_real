"""Conservative joint-box sweep checks, including tracking and stop envelopes.

The proof is conditional on measured geometry and bounds, not a hardware safety
certification. An uncertifiable interval is rejected rather than sampled away.
"""

from __future__ import annotations

import numpy as np

from .geometry import Capsule, DualArmError, check_collision, finite


def safety_template(arms):
    return {
        "status": "unmeasured",
        "validation_id": None,
        "open_tools_and_attachments_verified": False,
        "controller_watchdog_verified": False,
        "joint_segment_tracking_verified": False,
        "max_feedback_age_s": 0.1,
        "controller_watchdog_s": 0.15,
        "stop_command_latency_s": 0.05,
        "max_sweep_nodes": 4096,
        "arms": {
            k: {
                "base_error_mm": None,
                "geometry_error_mm": None,
                "max_joint_speed_deg_s": None,
                "stop_excursion_deg": None,
            }
            for k in arms
        },
    }


def synthetic_safety(arms, speed=10):
    data = safety_template(arms)
    data["status"] = "synthetic"
    for k, arm in arms.items():
        axis = arm["axis"] if isinstance(arm, dict) else arm.axis
        data["arms"][k].update(
            base_error_mm=0,
            geometry_error_mm=0,
            max_joint_speed_deg_s=speed,
            stop_excursion_deg=[0] * axis,
        )
    return data


def validate_safety(raw, arms, limits, *, real=False):
    from .config import number

    if not isinstance(raw, dict) or raw.get("status") not in {"synthetic", "measured"}:
        raise DualArmError(
            "measured collision/stop safety bounds required (no legacy fallback)"
        )
    for key in (
        "max_feedback_age_s",
        "controller_watchdog_s",
        "stop_command_latency_s",
    ):
        number(raw[key], key, 0.001, 0.5)
    if type(raw["max_sweep_nodes"]) is not int:
        raise DualArmError("integer sweep node budget required")
    number(raw["max_sweep_nodes"], "sweep node budget", 16, 100000)
    if set(raw["arms"]) != set(arms):
        raise DualArmError("safety bounds for both arms required")
    for k, arm in arms.items():
        row = raw["arms"][k]
        for key in ("base_error_mm", "geometry_error_mm"):
            number(row[key], key, 0 if not real else 0.01, 100)
        number(
            row["max_joint_speed_deg_s"],
            "verified joint speed",
            limits["joint_speed_deg_s"],
            100,
        )
        stop = finite(row["stop_excursion_deg"], (arm.axis,), "stop excursion")
        if np.any(stop < (0.001 if real else 0)) or np.any(stop > 30):
            raise DualArmError(
                "measured positive per-joint stopping excursions required"
            )
    if real and (
        raw["status"] != "measured"
        or not isinstance(raw.get("validation_id"), str)
        or not raw["validation_id"].strip()
        or any(
            raw.get(k) is not True
            for k in (
                "open_tools_and_attachments_verified",
                "controller_watchdog_verified",
                "joint_segment_tracking_verified",
            )
        )
    ):
        raise DualArmError(
            "real motion requires measured safety bounds, open-tool geometry, controller watchdog and segment tracking verification"
        )


def padded_capsules(config, models, joints, half_ranges=None):
    """Contain every capsule for independent joint ranges and a complete stop.

    Both ends of a capsule move by at most sum(reach_j * abs(delta_j)). A
    capsule inflated by that amount contains the entire swept segment. Arms
    may progress independently within an interval; synchronized arrival is not
    assumed. The uncertainty padding is never removed for an inter-arm check.
    """
    safety = config.raw["safety"]
    reaction = (
        safety["max_feedback_age_s"]
        + safety["controller_watchdog_s"]
        + safety["stop_command_latency_s"]
        + 1 / config.limits["rate_hz"]
        + config.limits["max_tick_lateness_s"]
        + config.limits["dispatch_skew_s"]
    )
    result = {}
    for k, model in models.items():
        row = safety["arms"][k]
        # Synthetic motion has exact feedback; physical runs may never use this.
        deviation = np.zeros(config.arms[k].axis)
        error_mm = 0.0
        if safety["status"] == "measured":
            deviation = (
                np.asarray(row["stop_excursion_deg"])
                + config.limits["tracking_error_deg"]
                + row["max_joint_speed_deg_s"] * reaction
            )
            error_mm = (
                row["base_error_mm"]
                + row["geometry_error_mm"]
                + config.limits["tracking_error_mm"]
            )
        if half_ranges is not None:
            deviation = deviation + half_ranges[k]
        bounds = model.capsule_motion_bounds(deviation)
        result[k] = [
            Capsule(c.name, c.start, c.end, c.radius + error_mm + bounds[c.name])
            for c in model.capsules(joints[k])
        ]
    return result


def check_state(config, models, joints, *, tool_contact=False, half_ranges=None):
    check_collision(
        padded_capsules(config, models, joints, half_ranges),
        config.obstacles,
        config.limits["clearance_mm"],
        allow_tool_contact=tool_contact,
    )


def validate_sweep(config, models, start, end, *, tool_contact=False):
    """Certify the full joint-linear interval; reject when proof budget expires."""
    # Endpoints include tracking/stop padding as well as exact model geometry.
    check_state(config, models, start, tool_contact=tool_contact)
    check_state(config, models, end, tool_contact=tool_contact)
    pending = [(start, end, 0)]
    nodes = 0
    while pending:
        a, b, depth = pending.pop()
        nodes += 1
        if nodes > config.raw["safety"]["max_sweep_nodes"]:
            raise DualArmError("continuous collision proof exhausted its node budget")
        mid = {k: (np.asarray(a[k]) + b[k]) / 2 for k in a}
        radii = {k: np.abs(np.asarray(b[k]) - a[k]) / 2 for k in a}
        try:
            check_state(
                config, models, mid, half_ranges=radii, tool_contact=tool_contact
            )
        except DualArmError as exc:
            # Reject real collisions immediately; subdivision cannot remove them.
            check_state(config, models, mid, tool_contact=tool_contact)
            if depth >= 16 or max(np.max(v) for v in radii.values()) < 1e-5:
                raise DualArmError(
                    f"continuous collision clearance cannot be certified: {exc}"
                ) from exc
            # Split one arm at a time. Splitting both along one shared progress
            # parameter would miss an arm lagging behind or stopping early.
            arm = max(radii, key=lambda k: np.max(radii[k]))
            split_end, split_start = dict(b), dict(a)
            split_end[arm] = mid[arm]
            split_start[arm] = mid[arm]
            pending.extend(((a, split_end, depth + 1), (split_start, b, depth + 1)))
    return nodes
