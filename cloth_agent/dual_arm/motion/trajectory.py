"""Shared-time smooth joint trajectories with certified polynomial bounds."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

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
    if segment["interpolation"] == "quintic_shared_knot_velocity":
        coefficients = finite(segment["coefficients_rad"], (6, 13), "polynomial coefficients")
        return tuple(np.polynomial.polynomial.polyval(
            u, np.polynomial.polynomial.polyder(coefficients, m=order)
        ) / duration**order for order in range(3))
    s, ds, dds = blend(u)
    a, b = np.asarray(segment["start_q_rad"]), np.asarray(segment["end_q_rad"])
    return a + (b - a) * s, (b - a) * ds / duration, (b - a) * dds / duration**2


def polynomial_peak(coefficients):
    """Exact scalar polynomial extrema per joint on the closed unit interval."""
    result = []
    for column in np.asarray(coefficients).T:
        roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(column))
        candidates = [0., 1., *[r.real for r in roots if abs(r.imag) < 1e-9 and 0 < r.real < 1]]
        result.append(max(abs(np.polynomial.polynomial.polyval(u, column)) for u in candidates))
    return np.asarray(result)


def smooth_segments(edges, velocity, acceleration, period_s):
    """Share velocities within each phase; stop at task events and arm changes.

    Edge count and endpoints stay unchanged so event waypoint indices retain
    their meaning. Curved interpolation is certified separately below.
    """
    segments = []
    groups = []
    for row in edges:
        a = finite(row['start_q_rad'], (13,), 'segment start')
        b = finite(row['end_q_rad'], (13,), 'segment end')
        active = tuple(bool(np.any(np.abs((b-a)[sl]) > 1e-12))
                       for sl in (slice(0, 6), slice(6, 13)))
        # An explicit zero-length edge is a stop, not a velocity bridge.
        identity = (row.get('phase'), active)
        if not groups or groups[-1][0] != identity or not any(active):
            groups.append((identity, []))
        groups[-1][1].append((row, a, b))
    for _, group in groups:
        points = np.asarray([group[0][1], *[b for _, a, b in group]])
        for i, (_, a, _) in enumerate(group):
            if not np.allclose(a, points[i], atol=1e-10, rtol=0):
                raise DualArmError('trajectory joint discontinuity')
        distances = np.maximum(1e-12, np.max(np.abs(np.diff(points, axis=0)) / velocity, axis=1))
        arc = np.r_[0., np.cumsum(distances)]
        total = arc[-1]
        scalar_accel = float(np.min(acceleration / velocity))
        ramp_time = min(1/scalar_accel, np.sqrt(total/scalar_accel))
        peak_speed = scalar_accel*ramp_time
        ramp_distance = .5*scalar_accel*ramp_time**2
        total_time = 2*ramp_time + max(0., total-2*ramp_distance)/peak_speed
        knots = np.array([
            np.sqrt(2*x/scalar_accel) if x < ramp_distance else
            total_time-np.sqrt(max(0., 2*(total-x)/scalar_accel)) if x > total-ramp_distance else
            ramp_time+(x-ramp_distance)/peak_speed for x in arc])
        durations = np.maximum(1e-9, np.diff(knots))
        knots = np.r_[0., np.cumsum(durations)]
        slopes = PchipInterpolator(knots, points, axis=0).derivative()(knots)
        slopes[[0, -1]] = 0.
        rows = []
        scale = max(1., period_s / durations.sum())
        for i, (row, a, b) in enumerate(group):
            d = b-a
            v0, v1 = slopes[i]*durations[i], slopes[i+1]*durations[i]
            c = np.array([a, v0, np.zeros(13), 10*d-6*v0-4*v1,
                          -15*d+8*v0+7*v1, 6*d-3*v0-3*v1])
            speed = polynomial_peak(np.polynomial.polynomial.polyder(c)) / durations[i]
            accel = polynomial_peak(np.polynomial.polynomial.polyder(c, m=2)) / durations[i]**2
            scale = max(scale, float(np.max(speed/velocity)), float(np.sqrt(np.max(accel/acceleration))))
            rows.append(dict(row, coefficients_rad=c.tolist(), duration_s=float(durations[i]),
                             interpolation='quintic_shared_knot_velocity'))
        for row in rows:
            row['duration_s'] *= scale * (1 + 1e-10)
        segments.extend(rows)
    return segments


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
    for segment in smooth_segments(edges, velocity, acceleration, period_s):
        duration = segment['duration_s']
        segment['start_time_s'] = clock
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
    previous_end = None
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
            or segment["interpolation"] not in {
                "quintic_zero_velocity_acceleration_at_knots", "quintic_shared_knot_velocity"}
        ):
            raise DualArmError("invalid trajectory interpolation or time axis")
        if previous is not None and not np.allclose(a, previous, atol=1e-10, rtol=0):
            raise DualArmError("trajectory joint discontinuity")
        padding = np.zeros(13)
        if segment['interpolation'] == 'quintic_shared_knot_velocity':
            c = finite(segment['coefficients_rad'], (6, 13), 'polynomial coefficients')
            first, last = sample_segment(segment, 0), sample_segment(segment, duration)
            if not np.allclose(first[0], a, atol=1e-10, rtol=0) or not np.allclose(last[0], b, atol=1e-10, rtol=0):
                raise DualArmError('polynomial endpoints differ from trajectory')
            speed = polynomial_peak(np.polynomial.polynomial.polyder(c))/duration
            accel = polynomial_peak(np.polynomial.polynomial.polyder(c, m=2))/duration**2
            delta = b-a
            residual = c.copy(); residual[0] -= a
            length2 = float(delta @ delta)
            if length2 > 1e-24:
                progress = residual @ delta / length2
                # Bound normalized progress about 1/2 to prove it stays in [0,1].
                centered = progress.copy(); centered[0] -= .5
                if polynomial_peak(centered[:, None])[0] > .5 + 1e-10:
                    raise DualArmError('smooth trajectory overshoots edge progress')
                residual -= progress[:, None]*delta
            padding = polynomial_peak(residual)
            padding[padding < 1e-12] = 0.
        else:
            first, last = sample_segment(segment, 0), sample_segment(segment, duration)
            speed = 1.875 * np.abs(b-a)/duration
            accel = (10/np.sqrt(3))*np.abs(b-a)/duration**2
        if np.any(speed > velocity + 1e-9) or np.any(accel > acceleration + 1e-9):
            raise DualArmError("analytic velocity/acceleration limit exceeded")
        if previous_end is None:
            if not np.allclose(first[1:], 0, atol=1e-8):
                raise DualArmError('trajectory must start at rest')
        elif not np.allclose(first[1:], previous_end[1:], atol=1e-8, rtol=0):
            raise DualArmError('trajectory velocity/acceleration discontinuity')
        # Cover the actual polynomial, including deviation from the old joint
        # chord. A line-only certificate is insufficient after smoothing.
        report = validator.certify(
            a, b, deadline=deadline, padding=padding, cartesian=segment.get("cartesian")
        )
        minimum = min(minimum, report["minimum_clearance_bound_m"])
        nodes += report["nodes"]
        expected_time += duration
        previous = b
        previous_end = last
    if previous is None or abs(expected_time - trajectory["duration_s"]) > 1e-8:
        raise DualArmError("invalid trajectory duration")
    if not np.allclose(previous_end[1:], 0, atol=1e-8):
        raise DualArmError('trajectory must end at rest')
    return {
        "minimum_clearance_bound_m": minimum,
        "interval_nodes": nodes,
        "continuous_geometry_checked": True,
        "analytic_dynamics_checked": True,
    }
