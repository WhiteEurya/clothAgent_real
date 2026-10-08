"""Explicit millimetre world/base transforms and conservative capsule checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.spatial.transform import Rotation


class DualArmError(ValueError):
    pass


def finite(value, shape, name):
    if any(
        isinstance(v, (bool, np.bool_, str))
        for v in np.asarray(value, dtype=object).flat
    ):
        raise DualArmError(f"{name}: numbers required, not booleans or strings")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DualArmError(f"{name}: numeric values required") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise DualArmError(f"{name}: expected {shape} finite values")
    return array


def transform(value, name="transform"):
    matrix = finite(value, (4, 4), name)
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
        or not math.isclose(np.linalg.det(matrix[:3, :3]), 1.0, abs_tol=1e-6)
    ):
        raise DualArmError(f"{name}: rigid right-handed transform required")
    return matrix


def pose_matrix(pose):
    pose = finite(pose, (6,), "pose_mm_deg")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", pose[3:], degrees=True).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def matrix_pose(matrix):
    matrix = transform(matrix)
    return np.r_[
        matrix[:3, 3],
        Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True),
    ]


def apply(matrix, point):
    return matrix[:3, :3] @ np.asarray(point) + matrix[:3, 3]


def pose_error(a, b):
    a, b = pose_matrix(a), pose_matrix(b)
    return (
        float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
        float(np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())),
    )


def segment_distance(a, b, c, d):
    """Distance between finite segments, including zero-length/parallel ones."""
    a, b, c, d = (np.asarray(p, dtype=float) for p in (a, b, c, d))
    u, v, w = b - a, d - c, a - c
    aa, bb, cc, dd, ee = u @ u, u @ v, v @ v, u @ w, v @ w
    candidates = []
    # Interior closest points plus every boundary of [0,1]^2.
    determinant = aa * cc - bb * bb
    if determinant > 1e-12:
        s, t = (bb * ee - cc * dd) / determinant, (aa * ee - bb * dd) / determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append((s, t))
    candidates.extend(
        (s, float(np.clip((ee + bb * s) / cc, 0, 1)) if cc else 0.0) for s in (0.0, 1.0)
    )
    candidates.extend(
        (float(np.clip((bb * t - dd) / aa, 0, 1)) if aa else 0.0, t) for t in (0.0, 1.0)
    )
    return min(float(np.linalg.norm(w + s * u - t * v)) for s, t in candidates)


def segment_box_distance(a, b, lower, upper):
    """Exact minimum of the piecewise quadratic distance to an axis-aligned box."""
    a, b, lower, upper = (np.asarray(p, dtype=float) for p in (a, b, lower, upper))
    delta = b - a
    cuts = [0.0, 1.0]
    for i in range(3):
        if abs(delta[i]) > 1e-12:
            cuts.extend(
                float(t)
                for t in ((lower[i] - a[i]) / delta[i], (upper[i] - a[i]) / delta[i])
                if 0 < t < 1
            )
    cuts = sorted(set(cuts))
    candidates = list(cuts)
    for left, right in pairwise(cuts):
        p = a + (left + right) / 2 * delta
        active = (p < lower) | (p > upper)
        target = np.where(p < lower, lower, upper)
        denom = float(delta[active] @ delta[active])
        if denom:
            candidates.append(
                float(
                    np.clip(-(a - target)[active] @ delta[active] / denom, left, right)
                )
            )
    return min(
        float(
            np.linalg.norm(
                np.maximum(lower - (a + t * delta), 0)
                + np.maximum(a + t * delta - upper, 0)
            )
        )
        for t in candidates
    )


@dataclass
class Capsule:
    name: str
    start: np.ndarray
    end: np.ndarray
    radius: float


def check_collision(capsules, boxes, clearance_mm, *, allow_tool_contact=False,
                    pair_separated=None, box_separated=None):
    ids = list(capsules)
    for left in capsules[ids[0]]:
        for right in capsules[ids[1]]:
            reach=left.radius+right.radius+clearance_mm
            if (np.any(np.minimum(left.start,left.end)-reach > np.maximum(right.start,right.end))
                    or np.any(np.minimum(right.start,right.end)-reach > np.maximum(left.start,left.end))):
                continue
            if (
                segment_distance(left.start, left.end, right.start, right.end)
                <= reach
            ):
                if pair_separated is not None and pair_separated(ids[0], left, ids[1], right, clearance_mm):
                    continue
                raise DualArmError(
                    f"inter-arm collision: {ids[0]}/{left.name}, {ids[1]}/{right.name}"
                )
    for arm_id, shapes in capsules.items():
        for shape in shapes:
            for box in boxes:
                if f"{arm_id}/{shape.name}" in box.get("excluded_capsules", []):
                    continue
                margin = clearance_mm
                if (
                    (allow_tool_contact is True or arm_id in (allow_tool_contact or ()))
                    and shape.name == "tool"
                    and "tool_contact_allowance_mm" in box
                ):
                    margin = -float(box["tool_contact_allowance_mm"])
                if (np.any(np.minimum(shape.start,shape.end)-shape.radius-margin > box['max_mm'])
                        or np.any(np.maximum(shape.start,shape.end)+shape.radius+margin < box['min_mm'])):
                    continue
                if (
                    segment_box_distance(
                        shape.start, shape.end, box["min_mm"], box["max_mm"]
                    )
                    < shape.radius + margin
                ):
                    if box_separated is not None and box_separated(arm_id, shape, box, margin):
                        continue
                    raise DualArmError(
                        f"obstacle collision: {arm_id}/{shape.name}, {box['name']}"
                    )
