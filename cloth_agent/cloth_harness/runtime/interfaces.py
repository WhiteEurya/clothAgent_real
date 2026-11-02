"""Versioned interfaces; future coding-agent edits must not alter these contracts."""
from dataclasses import dataclass
from typing import Any
import numpy as np

SCHEMA_VERSION = 1

@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray
    depth: np.ndarray | None
    timestamp: float
    camera_id: str
    camera_intrinsics: dict | None = None
    camera_extrinsics: dict | None = None
    world_xyz_mm: np.ndarray | None = None
    provenance: dict | None = None

@dataclass(frozen=True)
class Action:
    pick_px: tuple[int, int] | None
    place_px: tuple[int, int] | None
    gripper_angle_deg: float | None
    intent: str | None = None

@dataclass(frozen=True)
class Outcome:
    schema_version: int
    harness_version: str
    execution: dict[str, Any]
    coverage: dict[str, Any]
    motion: dict[str, Any]
    local_change: dict[str, Any]
    quality: dict[str, Any]
    latency_ms: float

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError('unsupported outcome schema')
