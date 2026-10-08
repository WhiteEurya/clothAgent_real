"""Offline execution preflight. No method in this module can command hardware."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime

import numpy as np

from ..geometry import DualArmError, finite
from .trajectory import sample_segment, validate_timed
from .validation import scene_digest


def plan_digest(plan):
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in plan.items() if k != "plan_digest"},
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def validate_artifact(plan, planner):
    if (
        plan.get("success") is not True
        or plan.get("schema_version") != 1
        or plan.get("units") != "m_rad_s"
    ):
        raise DualArmError("successful SI trajectory artifact required")
    if not isinstance(plan.get("plan_digest"), str) or not hmac.compare_digest(
        plan["plan_digest"], plan_digest(plan)
    ):
        raise DualArmError("trajectory content changed after planning")
    if plan.get("scene_digest") != scene_digest(planner.scene):
        raise DualArmError("robot geometry/calibration scene changed after planning")
    result = validate_timed(plan, planner.validator)
    if not np.allclose(
        plan["segments"][0]["start_q_rad"],
        np.r_[plan["initial_q_a"], plan["initial_q_b"]],
        atol=1e-10,
        rtol=0,
    ):
        raise DualArmError("initial state differs from trajectory start")
    a, b = plan["trajectory_a"], plan["trajectory_b"]
    times = np.asarray(a["times_s"], dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or not np.array_equal(times, b["times_s"])
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
        or abs(times[-1] - plan["duration_s"]) > 1e-8
    ):
        raise DualArmError("both arms must have the same strictly increasing time axis")
    values = {}
    for key, width, row in (("a", 6, a), ("b", 7, b)):
        for field in ("positions_rad", "velocities_rad_s", "accelerations_rad_s2"):
            values[key, field] = finite(row[field], (len(times), width), field)
    boundaries = np.array(
        [s["start_time_s"] + s["duration_s"] for s in plan["segments"]]
    )
    for i, t in enumerate(times):
        index = min(
            int(np.searchsorted(boundaries, t, side="left")), len(boundaries) - 1
        )
        segment = plan["segments"][index]
        expected = sample_segment(segment, t - segment["start_time_s"])
        for field, row in zip(
            ("positions_rad", "velocities_rad_s", "accelerations_rad_s2"), expected
        ):
            actual = np.r_[values["a", field][i], values["b", field][i]]
            if not np.allclose(actual, row, atol=1e-8, rtol=0):
                raise DualArmError(
                    "sampled trajectory differs from certified quintic interpolation"
                )
    return result


def preflight(
    plan,
    planner,
    current_q_a,
    current_q_b,
    *,
    sampled_at_a,
    sampled_at_b,
    now=None,
    real=False,
):
    """Read-only paired-state gate; timestamps are Unix seconds from one host.

    Caller supplies already acquired state. This function neither connects to
    hardware nor considers a passing offline check to be execution permission.
    """
    if real:
        raise DualArmError(
            "physical execution is not commissioned for this planner; offline validation only"
        )
    now = time.time() if now is None else float(now)
    timestamps = finite([sampled_at_a, sampled_at_b, now], (3,), "state timestamps")
    if np.any(now - timestamps[:2] < 0) or np.any(now - timestamps[:2] > 0.1):
        raise DualArmError("paired feedback is stale or from the future")
    if abs(sampled_at_a - sampled_at_b) > 0.02:
        raise DualArmError("paired feedback capture skew exceeds 20 ms")
    created = datetime.fromisoformat(plan["created_at"])
    if created.tzinfo is None:
        raise DualArmError("trajectory creation timestamp requires timezone")
    ttl = float(plan["valid_for_s"])
    if (
        not np.isfinite(ttl)
        or not 0 < ttl <= 120
        or not 0 <= now - created.timestamp() <= ttl
    ):
        raise DualArmError("trajectory expired or has invalid creation time")
    current = np.r_[
        finite(current_q_a, (6,), "current q_a"),
        finite(current_q_b, (7,), "current q_b"),
    ]
    planned = np.r_[
        finite(plan["initial_q_a"], (6,), "planned q_a"),
        finite(plan["initial_q_b"], (7,), "planned q_b"),
    ]
    if np.max(np.abs(current - planned)) > 0.005:
        raise DualArmError("current joint state differs from planned initial state")
    result = validate_artifact(plan, planner)
    # Account for the measured-to-planned transition too, without dispatching it.
    planner.validator.certify(current, planned)
    return {
        "success": True,
        "mode": "offline_preflight",
        **result,
        "execution_authorized": False,
        "physical_execution_supported": False,
    }
