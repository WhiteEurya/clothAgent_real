"""Shared-time quintic joint trajectories with analytic speed/acceleration bounds."""

from __future__ import annotations

import numpy as np

from ..geometry import DualArmError, finite


def blend(u):
    return (
        10 * u**3 - 15 * u**4 + 6 * u**5,
        30 * u**2 - 60 * u**3 + 30 * u**4,
        60 * u - 180 * u**2 + 120 * u**3,
    )


def sample_segment(segment, local_time):
    duration = segment["duration_s"]
    u = np.clip(local_time / duration, 0, 1)
    s, ds, dds = blend(u)
    a, b = np.asarray(segment["start_q_rad"]), np.asarray(segment["end_q_rad"])
    return a + (b - a) * s, (b - a) * ds / duration, (b - a) * dds / duration**2


def parameterize(edges, velocity, acceleration, *, period_s=0.02):
    velocity, acceleration = (
        finite(velocity, (13,), "velocity limits"),
        finite(acceleration, (13,), "acceleration limits"),
    )
    if (
        np.any(velocity <= 0)
        or np.any(acceleration <= 0)
        or not 0.005 <= period_s <= 0.1
    ):
        raise DualArmError(
            "positive dynamic limits and sample period 0.005..0.1 s required"
        )
    segments, clock, boundaries = [], 0.0, [0.0]
    for row in edges:
        a, b = (
            finite(row["start_q_rad"], (13,), "segment start"),
            finite(row["end_q_rad"], (13,), "segment end"),
        )
        duration = max(
            period_s,
            float(np.max(1.875 * np.abs(b - a) / velocity)),
            float(np.max(np.sqrt((10 / np.sqrt(3)) * np.abs(b - a) / acceleration))),
        )
        segment = dict(
            row,
            start_time_s=clock,
            duration_s=duration,
            interpolation="quintic_zero_velocity_acceleration_at_knots",
        )
        segments.append(segment)
        clock += duration
        boundaries.append(clock)
    if not segments:
        raise DualArmError("trajectory requires at least one segment")
    times, qrows, vrows, arows = [], [], [], []
    for index, segment in enumerate(segments):
        count = int(np.ceil(segment["duration_s"] / period_s))
        if count + len(times) > 200000:
            raise DualArmError("trajectory sample budget exceeded")
        local = np.linspace(0, segment["duration_s"], count + 1)
        for t in local if index == 0 else local[1:]:
            q, v, a = sample_segment(segment, t)
            times.append(float(t + segment["start_time_s"]))
            qrows.append(q)
            vrows.append(v)
            arows.append(a)
    qrows, vrows, arows = map(np.asarray, (qrows, vrows, arows))
    result = {
        "segments": segments,
        "waypoint_times_s": boundaries,
        "duration_s": clock,
        "sample_period_s": period_s,
        "velocity_limits_rad_s": velocity.tolist(),
        "acceleration_limits_rad_s2": acceleration.tolist(),
    }
    for key, sl in (("trajectory_a", slice(0, 6)), ("trajectory_b", slice(6, 13))):
        result[key] = {
            "times_s": times,
            "positions_rad": qrows[:, sl].tolist(),
            "velocities_rad_s": vrows[:, sl].tolist(),
            "accelerations_rad_s2": arows[:, sl].tolist(),
        }
    return result


def validate_timed(trajectory, validator, *, deadline=None):
    velocity = finite(trajectory["velocity_limits_rad_s"], (13,), "velocity limits")
    acceleration = finite(
        trajectory["acceleration_limits_rad_s2"], (13,), "acceleration limits"
    )
    if np.any(velocity <= 0) or np.any(acceleration <= 0):
        raise DualArmError("positive dynamic limits required")
    minimum, nodes, expected_time, previous = float("inf"), 0, 0.0, None
    for segment in trajectory["segments"]:
        a, b = (
            finite(segment["start_q_rad"], (13,), "segment start"),
            finite(segment["end_q_rad"], (13,), "segment end"),
        )
        duration = float(segment["duration_s"])
        if (
            not np.isfinite(duration)
            or duration <= 0
            or not np.isfinite(segment["start_time_s"])
            or abs(segment["start_time_s"] - expected_time) > 1e-8
            or segment["interpolation"] != "quintic_zero_velocity_acceleration_at_knots"
        ):
            raise DualArmError("invalid trajectory interpolation or time axis")
        if previous is not None and not np.allclose(a, previous, atol=1e-10, rtol=0):
            raise DualArmError("trajectory joint discontinuity")
        if np.any(1.875 * np.abs(b - a) / duration > velocity + 1e-9) or np.any(
            (10 / np.sqrt(3)) * np.abs(b - a) / duration**2 > acceleration + 1e-9
        ):
            raise DualArmError("analytic velocity/acceleration limit exceeded")
        # Quintic progress is monotonic, so it traces exactly the certified
        # joint-line geometry. Revalidate after time parameterization anyway.
        report = validator.certify(
            a, b, deadline=deadline, cartesian=segment.get("cartesian")
        )
        minimum = min(minimum, report["minimum_clearance_bound_m"])
        nodes += report["nodes"]
        expected_time += duration
        previous = b
    if previous is None or abs(expected_time - trajectory["duration_s"]) > 1e-8:
        raise DualArmError("invalid trajectory duration")
    return {
        "minimum_clearance_bound_m": minimum,
        "interval_nodes": nodes,
        "continuous_geometry_checked": True,
        "analytic_dynamics_checked": True,
    }
