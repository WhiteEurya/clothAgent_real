"""Empirical world-Y tilt correction about a fixed, measured world reference."""
import numpy as np


def rotate_cloud(xyz, pivot, angle_deg):
    pivot = np.asarray(pivot, dtype=float)
    if pivot.shape != (3,) or not np.isfinite(pivot).all() or not np.isfinite(angle_deg):
        raise ValueError('finite XYZ pivot and angle required')
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (np.asarray(xyz) - pivot) @ rotation.T + pivot, rotation
