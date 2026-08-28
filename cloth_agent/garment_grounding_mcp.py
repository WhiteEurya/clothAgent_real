"""Read-only MCP tools for calibrated garment coordinate grounding.

The server is intentionally small and dependency-free apart from NumPy.  It
reads only one saved ``workspace/perception_views`` directory and exposes
measured geometry; it never selects a grasp candidate, writes files, executes
commands, or controls the robot.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


SERVER_NAME = "garment_grounding"
SERVER_VERSION = "1.0.0"
REFERENCE_PATTERN = re.compile(r"^R\d{3,}$")


class GroundingToolError(RuntimeError):
    """Raised when a requested saved measurement is unavailable or invalid."""


def _finite_list(values: np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise GroundingToolError("measurement contains non-finite values")
    return [float(value) for value in array.tolist()]


class GarmentGrounding:
    """Read calibrated coordinate guides and full-resolution geometry maps."""

    def __init__(self, perception_dir: Path):
        self.perception_dir = Path(perception_dir).expanduser().resolve()
        if not self.perception_dir.is_dir():
            raise GroundingToolError(
                f"perception directory does not exist: {self.perception_dir}"
            )
        self._json_cache: dict[Path, Any] = {}
        self._array_cache: dict[Path, np.ndarray] = {}

    @staticmethod
    def _camera(camera: str) -> str:
        label = str(camera).strip().upper()
        if label not in {"A", "B"}:
            raise GroundingToolError("camera must be A or B")
        return label

    def _path(self, name: str) -> Path:
        path = (self.perception_dir / name).resolve()
        if path.parent != self.perception_dir:
            raise GroundingToolError("measurement path escaped perception directory")
        return path

    def _json(self, name: str) -> Any:
        path = self._path(name)
        if not path.is_file():
            raise GroundingToolError(f"saved measurement file is missing: {name}")
        if path not in self._json_cache:
            self._json_cache[path] = json.loads(path.read_text(encoding="utf-8"))
        return self._json_cache[path]

    def _array(self, name: str) -> np.ndarray:
        path = self._path(name)
        if not path.is_file():
            raise GroundingToolError(f"saved measurement file is missing: {name}")
        if path not in self._array_cache:
            self._array_cache[path] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._array_cache[path]

    def _fused_surface_support(
        self,
        base_xy_mm: np.ndarray,
        *,
        radius_mm: float = 15.0,
    ) -> dict[str, Any] | None:
        """Find fused A/B surface support near a Camera-A selected XY.

        Camera A remains the source of the action pixel and XY.  When the
        perception workspace contains the dense fused artifacts, use Camera-B
        or A+B-supported points to repair a material/view-dependent Camera-A Z
        under-estimate.  Missing fused artifacts keep the legacy behavior.
        """

        required = (
            "fused_points_base_mm.npy",
            "fused_height_above_table_mm.npy",
            "fused_source_mask.npy",
        )
        if not all(self._path(name).is_file() for name in required):
            return None
        points = np.asarray(self._array(required[0]), dtype=np.float64)
        heights = np.asarray(self._array(required[1]), dtype=np.float64)
        sources = np.asarray(self._array(required[2]), dtype=np.uint8)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(heights):
            return None
        xy = np.asarray(base_xy_mm, dtype=np.float64)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            return None
        distance = np.linalg.norm(points[:, :2] - xy[None, :], axis=1)
        finite = np.all(np.isfinite(points), axis=1) & np.isfinite(heights)
        nearby = finite & (distance <= float(radius_mm))
        garment_path = self._path("fused_garment_mask.npy")
        if garment_path.is_file():
            garment = np.asarray(self._array(garment_path), dtype=bool)
            if garment.shape == nearby.shape:
                nearby &= garment
        if int(np.count_nonzero(nearby)) < 3:
            return None
        # Source bit 2 means Camera B support; bit 3 means A+B overlap.
        b_supported = nearby & ((sources & 2) != 0)
        preferred = b_supported if int(np.count_nonzero(b_supported)) >= 3 else nearby
        selected_points = points[preferred]
        selected_heights = heights[preferred]
        p10, p25, p50, p75, p90 = np.percentile(
            selected_points, [10, 25, 50, 75, 90], axis=0
        )
        h10, h25, h50, h75, h90 = np.percentile(
            selected_heights, [10, 25, 50, 75, 90]
        )
        source_values, source_counts = np.unique(sources[preferred], return_counts=True)
        return {
            "radius_mm": float(radius_mm),
            "nearby_count": int(np.count_nonzero(nearby)),
            "preferred_count": int(np.count_nonzero(preferred)),
            "camera_b_supported_count": int(np.count_nonzero(b_supported)),
            "preferred_source": "camera_B_or_AB" if np.any(preferred & b_supported) else "all_fused",
            "source_counts": {
                str(int(source)): int(count)
                for source, count in zip(source_values, source_counts)
            },
            "base_xyz_p10_mm": _finite_list(p10),
            "base_xyz_p25_mm": _finite_list(p25),
            "base_xyz_median_mm": _finite_list(p50),
            "base_xyz_p75_mm": _finite_list(p75),
            "base_xyz_p90_mm": _finite_list(p90),
            "base_z_p90_minus_p10_mm": float(p90[2] - p10[2]),
            "height_p10_mm": float(h10),
            "height_p25_mm": float(h25),
            "height_median_mm": float(h50),
            "height_p75_mm": float(h75),
            "height_p90_mm": float(h90),
        }

    def _guide(self, camera: str) -> dict[str, Any]:
        label = self._camera(camera)
        guide = self._json(f"camera_{label}_coordinate_guide.json")
        if not isinstance(guide, dict) or not isinstance(guide.get("samples"), list):
            raise GroundingToolError(f"Camera {label} coordinate guide is malformed")
        return guide

    def lookup_reference(self, camera: str, reference_id: str) -> dict[str, Any]:
        """Return the exact saved measurement for one Rxxx reference."""

        label = self._camera(camera)
        identifier = str(reference_id).strip().upper()
        if not REFERENCE_PATTERN.fullmatch(identifier):
            raise GroundingToolError("reference_id must look like R026")
        guide = self._guide(label)
        match = next(
            (
                sample
                for sample in guide["samples"]
                if str(sample.get("reference_id", "")).upper() == identifier
            ),
            None,
        )
        if match is None:
            available = [str(sample.get("reference_id")) for sample in guide["samples"]]
            raise GroundingToolError(
                f"reference {identifier} is not present for Camera {label}; "
                f"available={available}"
            )
        xyz = _finite_list(np.asarray(match["base_xyz_mm"], dtype=np.float64))
        height = float(match["height_above_table_mm"])
        if not math.isfinite(height):
            raise GroundingToolError("reference height is non-finite")
        is_fold_boundary = match.get("reference_source") == "fold_rgb_boundary_dense"
        result = {
            "measurement_kind": (
                "fold_rgb_boundary_calibrated_reference"
                if is_fold_boundary
                else str(guide.get("measurement_kind", "uniform_calibrated_reference"))
            ),
            "camera": label,
            "reference_id": identifier,
            "pixel_xy": [int(value) for value in match["pixel_xy"]],
            "base_xyz_mm": xyz,
            "table_z_mm": float(xyz[2] - height),
            "height_above_table_mm": height,
            "coordinate_frame": str(guide.get("coordinate_frame", "robot_base_mm")),
            "reference_semantics": (
                str(guide.get("fold_boundary_reference_semantics", ""))
                if is_fold_boundary
                else str(guide.get("reference_semantics", ""))
            ),
            "valid": True,
            "warning": (
                "Host-validated RGB sleeve-boundary reference; Base XYZ remains a "
                "direct calibrated pixel measurement and must still pass workspace, "
                "grasp-height, preflight, and controller validation."
                if is_fold_boundary
                else str(
                    guide.get(
                        "warning",
                        "Measured coordinate only; this reference is not a ranked grasp candidate.",
                    )
                )
            ),
        }
        if not is_fold_boundary and "sample_stride_px" in guide:
            result["sample_stride_px"] = guide["sample_stride_px"]
        for key in (
            "name",
            "description",
            "source_pixel_xy",
            "confidence",
            "confidence_threshold",
            "confidence_definition",
            "local_radius_px",
            "local_sample_count",
            "local_base_z_spread_mm",
        ):
            if key in match:
                result[key] = match[key]
        return result

    def nearest_reference(self, camera: str, x_px: int, y_px: int) -> dict[str, Any]:
        """Return the exact Rxxx sample nearest to a selected image pixel."""

        label = self._camera(camera)
        guide = self._guide(label)
        x_value, y_value = int(x_px), int(y_px)
        samples = guide["samples"]
        if not samples:
            raise GroundingToolError(f"Camera {label} has no coordinate references")
        nearest = min(
            samples,
            key=lambda sample: (
                (float(sample["pixel_xy"][0]) - x_value) ** 2
                + (float(sample["pixel_xy"][1]) - y_value) ** 2
            ),
        )
        result = self.lookup_reference(label, str(nearest["reference_id"]))
        dx = float(result["pixel_xy"][0] - x_value)
        dy = float(result["pixel_xy"][1] - y_value)
        result["query_pixel_xy"] = [x_value, y_value]
        result["pixel_distance"] = float(math.hypot(dx, dy))
        return result

    def _pixel_arrays(
        self, camera: str
    ) -> tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None]:
        label = self._camera(camera)
        xyz = self._array(f"camera_{label}_base_xyz_mm.npy")
        if xyz.ndim != 3 or xyz.shape[2] != 3:
            raise GroundingToolError(
                f"Camera {label} base XYZ map must have shape HxWx3"
            )
        height_path = self._path(f"camera_{label}_height_above_table_mm.npy")
        table_path = self._path(f"camera_{label}_table_z_mm.npy")
        height = self._array(height_path.name) if height_path.is_file() else None
        table = self._array(table_path.name) if table_path.is_file() else None
        return label, xyz, height, table

    def _garment_mask(self, camera: str) -> np.ndarray | None:
        """Load the saved semantic garment mask when this perception has one."""

        label = self._camera(camera)
        path = self._path(f"camera_{label}_garment_mask.npy")
        if not path.is_file():
            return None
        mask = np.asarray(self._array(path.name), dtype=bool)
        if mask.ndim != 2:
            return None
        return mask

    def _minimum_grasp_height_mm(self) -> float:
        """Read the run's shared minimum engagement height when available."""

        # The MCP normally receives only ``workspace/perception_views``.  The
        # run-local robot config sits one directory above it and is copied at
        # session creation.  Falling back to the shared default keeps older
        # perception workspaces compatible.
        config_path = self.perception_dir.parent / "robot_config.json"
        default = 0.75
        if not config_path.is_file():
            return default
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            value = payload.get("grasp_height", {}).get("min_compression_mm", default)
            value = float(value)
            return value if math.isfinite(value) and value > 0.0 else default
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return default

    def _table_height_support_enabled(self) -> bool:
        """Return whether table-relative height may influence support-pixel repair.

        Absolute-camera grasp mode intentionally keeps the saved height maps for
        diagnostics, but does not let a potentially biased tabletop estimate
        move a semantic RGB point to a different pixel.
        """

        config_path = self.perception_dir.parent / "robot_config.json"
        if not config_path.is_file():
            return True
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            value = payload.get("grasp_height", {}).get(
                "use_table_clearance_floor", True
            )
            return bool(value)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return True

    @staticmethod
    def _patch_stats(
        xyz: np.ndarray,
        height_map: np.ndarray | None,
        garment_mask: np.ndarray,
        x_px: int,
        y_px: int,
        radius_px: int,
        interior_distance: np.ndarray | None,
    ) -> dict[str, Any]:
        """Summarize whether one nearby pixel has a stable interior surface."""

        height, width = xyz.shape[:2]
        x0 = max(0, int(x_px) - radius_px)
        x1 = min(width, int(x_px) + radius_px + 1)
        y0 = max(0, int(y_px) - radius_px)
        y1 = min(height, int(y_px) + radius_px + 1)
        mask_patch = garment_mask[y0:y1, x0:x1]
        xyz_patch = np.asarray(xyz[y0:y1, x0:x1], dtype=np.float64)
        finite_patch = np.all(np.isfinite(xyz_patch), axis=2)
        garment_count = max(1, int(mask_patch.size))
        valid_garment = mask_patch & finite_patch
        valid_count = int(np.count_nonzero(valid_garment))
        values: dict[str, Any] = {
            "pixel_xy": [int(x_px), int(y_px)],
            "mask_fraction": float(np.count_nonzero(mask_patch)) / garment_count,
            "finite_garment_fraction": float(valid_count) / garment_count,
            "sample_count": valid_count,
            "interior_distance_px": (
                float(interior_distance[int(y_px), int(x_px)])
                if interior_distance is not None
                else None
            ),
        }
        if valid_count:
            local_points = xyz_patch[valid_garment]
            values["base_xyz_median_mm"] = _finite_list(
                np.percentile(local_points, 50, axis=0)
            )
            values["base_z_spread_mm"] = float(
                np.percentile(local_points[:, 2], 90)
                - np.percentile(local_points[:, 2], 10)
            )
        if height_map is not None:
            local_height = np.asarray(height_map[y0:y1, x0:x1], dtype=np.float64)
            local_height = local_height[valid_garment]
            local_height = local_height[np.isfinite(local_height)]
            if local_height.size:
                values["height_median_mm"] = float(np.percentile(local_height, 50))
                values["height_spread_mm"] = float(
                    np.percentile(local_height, 90)
                    - np.percentile(local_height, 10)
                )
        return values

    def _select_interior_support_pixel(
        self,
        camera: str,
        xyz: np.ndarray,
        height_map: np.ndarray | None,
        x_px: int,
        y_px: int,
        radius_px: int,
        minimum_height_mm: float | None,
    ) -> tuple[int, int, dict[str, Any]]:
        """Keep a semantic edge selection but measure its nearest stable cloth interior.

        A contour pixel can be visually correct while its depth sample sees the
        table/background.  When the saved garment mask is available, search only
        a small neighbourhood of that same garment region for the nearest patch
        with dense garment/depth support.  This is deliberately local: it does
        not rank the whole garment or replace Claude's semantic choice.
        """

        garment_mask = self._garment_mask(camera)
        if garment_mask is None or garment_mask.shape != xyz.shape[:2]:
            return x_px, y_px, {
                "applied": False,
                "reason": "saved garment mask unavailable",
                "query_pixel_xy": [int(x_px), int(y_px)],
                "support_pixel_xy": [int(x_px), int(y_px)],
            }

        try:
            from scipy.ndimage import distance_transform_edt

            interior_distance = distance_transform_edt(garment_mask)
        except ImportError:
            interior_distance = None

        patch_radius = max(1, min(2, int(radius_px)))
        query_stats = self._patch_stats(
            xyz,
            height_map,
            garment_mask,
            x_px,
            y_px,
            patch_radius,
            interior_distance,
        )
        query_height = query_stats.get("height_median_mm")
        query_bad = (
            not bool(garment_mask[y_px, x_px])
            or float(query_stats.get("finite_garment_fraction", 0.0)) < 0.60
            or (
                query_height is not None
                and math.isfinite(float(query_height))
                and minimum_height_mm is not None
                and float(query_height) < minimum_height_mm
            )
        )
        if not query_bad:
            return x_px, y_px, {
                "applied": False,
                "reason": "query patch has sufficient garment/depth support",
                "query_pixel_xy": [int(x_px), int(y_px)],
                "support_pixel_xy": [int(x_px), int(y_px)],
                "query_patch": query_stats,
            }

        search_radius = max(12, min(24, int(radius_px) * 8))
        height, width = xyz.shape[:2]
        candidates: list[tuple[tuple[float, float, float], int, int, dict[str, Any]]] = []
        for cy in range(max(0, y_px - search_radius), min(height, y_px + search_radius + 1)):
            for cx in range(max(0, x_px - search_radius), min(width, x_px + search_radius + 1)):
                if not garment_mask[cy, cx]:
                    continue
                stats = self._patch_stats(
                    xyz,
                    height_map,
                    garment_mask,
                    cx,
                    cy,
                    patch_radius,
                    interior_distance,
                )
                if float(stats.get("finite_garment_fraction", 0.0)) < 0.70:
                    continue
                if float(stats.get("mask_fraction", 0.0)) < 0.70:
                    continue
                interior = float(stats.get("interior_distance_px") or 0.0)
                if interior_distance is not None and interior < 1.5:
                    continue
                candidate_height = stats.get("height_median_mm")
                if (
                    candidate_height is not None
                    and minimum_height_mm is not None
                    and float(candidate_height) < minimum_height_mm
                ):
                    continue
                distance_px = math.hypot(float(cx - x_px), float(cy - y_px))
                spread = float(stats.get("height_spread_mm", stats.get("base_z_spread_mm", 99.0)))
                # Stay near Claude's semantic region first; use interior support
                # and a compact depth patch only to break ties.
                score = (
                    distance_px,
                    -interior,
                    min(spread, 99.0),
                )
                candidates.append((score, cx, cy, stats))

        if not candidates:
            return x_px, y_px, {
                "applied": False,
                "reason": "no nearby stable garment-interior support pixel",
                "query_pixel_xy": [int(x_px), int(y_px)],
                "support_pixel_xy": [int(x_px), int(y_px)],
                "query_patch": query_stats,
                "search_radius_px": search_radius,
            }

        _, support_x, support_y, support_stats = min(candidates, key=lambda item: item[0])
        return support_x, support_y, {
            "applied": True,
            "reason": "query edge/height patch was unstable; used nearest stable garment interior",
            "query_pixel_xy": [int(x_px), int(y_px)],
            "support_pixel_xy": [int(support_x), int(support_y)],
            "shift_px": float(math.hypot(support_x - x_px, support_y - y_px)),
            "search_radius_px": search_radius,
            "query_patch": query_stats,
            "support_patch": support_stats,
        }

    @staticmethod
    def _validate_pixel(xyz: np.ndarray, x_px: int, y_px: int) -> tuple[int, int]:
        x_value, y_value = int(x_px), int(y_px)
        height, width = xyz.shape[:2]
        if not 0 <= x_value < width or not 0 <= y_value < height:
            raise GroundingToolError(
                f"pixel ({x_value}, {y_value}) is outside image bounds {width}x{height}"
            )
        return x_value, y_value

    def sample_pixel_xyz(self, camera: str, x_px: int, y_px: int) -> dict[str, Any]:
        """Return calibrated Base XYZ at one exact full-resolution pixel."""

        label, xyz, height_map, table_map = self._pixel_arrays(camera)
        x_value, y_value = self._validate_pixel(xyz, x_px, y_px)
        point = np.asarray(xyz[y_value, x_value], dtype=np.float64)
        if not np.all(np.isfinite(point)):
            return {
                "measurement_kind": "full_resolution_calibrated_pixel",
                "camera": label,
                "pixel_xy": [x_value, y_value],
                "valid": False,
                "reason": "no finite calibrated XYZ is available at this pixel",
            }
        result: dict[str, Any] = {
            "measurement_kind": "full_resolution_calibrated_pixel",
            "camera": label,
            "pixel_xy": [x_value, y_value],
            "base_xyz_mm": _finite_list(point),
            "valid": True,
            "warning": "Measured geometry only; visual garment membership must be checked from the supplied images/masks.",
        }
        if height_map is not None:
            height = float(height_map[y_value, x_value])
            result["height_above_table_mm"] = height if math.isfinite(height) else None
        if table_map is not None:
            table_z = float(table_map[y_value, x_value])
            result["table_z_mm"] = table_z if math.isfinite(table_z) else None
        nearest = self.nearest_reference(label, x_value, y_value)
        result["nearest_reference"] = {
            key: nearest[key]
            for key in (
                "reference_id",
                "pixel_xy",
                "pixel_distance",
                "base_xyz_mm",
                "height_above_table_mm",
            )
        }
        return result

    def sample_local_surface(
        self,
        camera: str,
        x_px: int,
        y_px: int,
        radius_px: int = 3,
        *,
        include_nearest_reference: bool = True,
    ) -> dict[str, Any]:
        """Return robust local XYZ/height statistics around a selected pixel."""

        label, xyz, height_map, table_map = self._pixel_arrays(camera)
        x_value, y_value = self._validate_pixel(xyz, x_px, y_px)
        radius = int(radius_px)
        if radius < 0 or radius > 25:
            raise GroundingToolError("radius_px must be between 0 and 25")
        minimum_height_mm: float | None = (
            self._minimum_grasp_height_mm()
            if self._table_height_support_enabled()
            else None
        )
        support_x, support_y, support_diagnostic = self._select_interior_support_pixel(
            label,
            xyz,
            height_map,
            x_value,
            y_value,
            radius,
            minimum_height_mm,
        )
        y0, y1 = max(0, support_y - radius), min(xyz.shape[0], support_y + radius + 1)
        x0, x1 = max(0, support_x - radius), min(xyz.shape[1], support_x + radius + 1)
        local_xyz = np.asarray(xyz[y0:y1, x0:x1], dtype=np.float64).reshape(-1, 3)
        valid = np.all(np.isfinite(local_xyz), axis=1)
        points = local_xyz[valid]
        if not len(points):
            return {
                "measurement_kind": "robust_local_calibrated_surface",
                "camera": label,
                "query_pixel_xy": [x_value, y_value],
                "support_pixel_xy": [support_x, support_y],
                "support_pixel_diagnostic": support_diagnostic,
                "radius_px": radius,
                "valid": False,
                "reason": "local window contains no finite calibrated XYZ",
            }
        p10 = np.percentile(points, 10, axis=0)
        p50 = np.percentile(points, 50, axis=0)
        p90 = np.percentile(points, 90, axis=0)
        result: dict[str, Any] = {
            "measurement_kind": "robust_local_calibrated_surface",
            "camera": label,
            "query_pixel_xy": [x_value, y_value],
            "support_pixel_xy": [support_x, support_y],
            "support_pixel_diagnostic": support_diagnostic,
            "window_xyxy": [x0, y0, x1 - 1, y1 - 1],
            "radius_px": radius,
            "sample_count": int(len(points)),
            "window_pixel_count": int((y1 - y0) * (x1 - x0)),
            "base_xyz_median_mm": _finite_list(p50),
            "base_xyz_p10_mm": _finite_list(p10),
            "base_xyz_p90_mm": _finite_list(p90),
            "base_z_p90_minus_p10_mm": float(p90[2] - p10[2]),
            "valid": True,
            "warning": "Local statistics can mix surfaces across an occlusion edge; compare the reported spread with RGB/depth boundaries.",
            "table_height_support_enabled": minimum_height_mm is not None,
        }
        local_valid_grid = valid.reshape(y1 - y0, x1 - x0)
        if height_map is not None:
            local_height = np.asarray(
                height_map[y0:y1, x0:x1], dtype=np.float64
            )[local_valid_grid]
            local_height = local_height[np.isfinite(local_height)]
            if len(local_height):
                result["height_above_table_median_mm"] = float(
                    np.percentile(local_height, 50)
                )
                result["height_above_table_p10_mm"] = float(
                    np.percentile(local_height, 10)
                )
                result["height_above_table_p90_mm"] = float(
                    np.percentile(local_height, 90)
                )
        if table_map is not None:
            local_table = np.asarray(
                table_map[y0:y1, x0:x1], dtype=np.float64
            )[local_valid_grid]
            local_table = local_table[np.isfinite(local_table)]
            if len(local_table):
                result["table_z_median_mm"] = float(np.percentile(local_table, 50))
        # Camera A is still authoritative for the action pixel and XY, but a
        # dense fused cloud can provide a better surface-Z estimate when the
        # two cameras see the same cloth at different heights.  Keep the raw
        # Camera-A measurement in the record for auditability and only replace
        # the Z component when there is at least modest Camera-B/AB support.
        # In absolute-camera mode Camera A's measured XYZ is authoritative for
        # both XY and Z.  Do not let a table-relative A/B height heuristic
        # replace it with a fused support estimate.
        if label == "A" and self._table_height_support_enabled():
            fused_support = self._fused_surface_support(p50[:2])
            if fused_support is not None:
                raw_surface_z = float(p50[2])
                fused_surface_z = float(fused_support["base_xyz_median_mm"][2])
                raw_height = result.get("height_above_table_median_mm")
                fused_height = float(fused_support["height_median_mm"])
                fused_spread = float(fused_support["base_z_p90_minus_p10_mm"])
                fused_p25_z = float(fused_support["base_xyz_p25_mm"][2])
                fused_p25_height = float(fused_support["height_p25_mm"])
                compact_support = fused_spread <= 8.0
                support_mode: str | None = None
                selected_surface_z = raw_surface_z
                selected_height = float(raw_height) if raw_height is not None else None
                if int(fused_support["camera_b_supported_count"]) >= 3:
                    if compact_support and fused_surface_z > raw_surface_z + 1.0:
                        support_mode = "fused_camera_b_or_ab_median"
                        selected_surface_z = fused_surface_z
                        selected_height = fused_height
                    elif (
                        raw_height is not None
                        and float(raw_height) <= 0.0
                        and fused_p25_height > 0.0
                    ):
                        # A wide B-support cloud can mix an edge and a raised
                        # neighbouring layer.  For a Camera-A near-table point,
                        # use only the conservative lower quartile rather than
                        # the optimistic median; this repairs false negatives
                        # without jumping to the top of an occlusion stack.
                        support_mode = "fused_camera_b_or_ab_lower_quartile_uncertain"
                        selected_surface_z = fused_p25_z
                        selected_height = fused_p25_height
                result["fused_surface_support"] = fused_support
                result["camera_a_raw_surface_z_mm"] = raw_surface_z
                result["camera_a_raw_height_above_table_mm"] = raw_height
                result["fused_surface_support_used"] = support_mode is not None
                result["fused_surface_support_mode"] = support_mode
                if support_mode is not None:
                    repaired_xyz = list(result["base_xyz_median_mm"])
                    repaired_xyz[2] = selected_surface_z
                    result["base_xyz_median_mm"] = repaired_xyz
                    result["height_above_table_median_mm"] = selected_height
                    result["base_z_p90_minus_p10_mm"] = max(
                        float(result["base_z_p90_minus_p10_mm"]),
                        float(fused_support["base_z_p90_minus_p10_mm"]),
                    )
                    result["surface_measurement_policy"] = (
                        "camera_A_xy_with_fused_camera_B_or_AB_surface_z"
                    )
        # A local height jump is only a candidate-structure signal.  A narrow
        # ridge or isolated spike can be a rolled wrinkle rather than a free
        # ply that the gripper can peel away.  Report a conservative shape
        # diagnostic next to the robust surface statistics so the planner can
        # require an active lift/hold check before committing to transport.
        diagnostic_radius = max(6, radius * 2)
        dy0 = max(0, support_y - diagnostic_radius)
        dy1 = min(xyz.shape[0], support_y + diagnostic_radius + 1)
        dx0 = max(0, support_x - diagnostic_radius)
        dx1 = min(xyz.shape[1], support_x + diagnostic_radius + 1)
        diagnostic = np.asarray(
            height_map[dy0:dy1, dx0:dx1], dtype=np.float64
        ) if height_map is not None else None
        if diagnostic is not None:
            finite_diagnostic = diagnostic[np.isfinite(diagnostic)]
            if finite_diagnostic.size:
                d50 = float(np.percentile(finite_diagnostic, 50.0))
                d10 = float(np.percentile(finite_diagnostic, 10.0))
                d90 = float(np.percentile(finite_diagnostic, 90.0))
                dspread = max(0.0, d90 - d10)
                high_cut = d50 + max(4.0, 0.35 * dspread)
                high = np.isfinite(diagnostic) & (diagnostic >= high_cut)
                high_fraction = float(np.count_nonzero(high)) / float(
                    max(1, np.count_nonzero(np.isfinite(diagnostic)))
                )
                center_y = min(max(support_y - dy0, 0), diagnostic.shape[0] - 1)
                center_x = min(max(support_x - dx0, 0), diagnostic.shape[1] - 1)
                center_is_high = bool(
                    np.isfinite(diagnostic[center_y, center_x])
                    and diagnostic[center_y, center_x] >= high_cut
                )
                if dspread < 5.0:
                    surface_shape = "LOW_RELIEF"
                elif center_is_high and high_fraction <= 0.22:
                    surface_shape = "NARROW_RIDGE_OR_SPIKE"
                elif center_is_high:
                    surface_shape = "BROAD_RELIEF"
                else:
                    surface_shape = "MIXED_OR_OCCLUSION_EDGE"
                result["surface_shape_diagnostic"] = {
                    "status": "TRIAGE_ONLY",
                    "diagnostic_radius_px": diagnostic_radius,
                    "height_p10_mm": d10,
                    "height_p50_mm": d50,
                    "height_p90_mm": d90,
                    "height_spread_mm": dspread,
                    "high_region_fraction": high_fraction,
                    "center_is_high": center_is_high,
                    "requires_structure_hold_check": True,
                    "compression_probe_recommended": surface_shape
                    in {"NARROW_RIDGE_OR_SPIKE", "MIXED_OR_OCCLUSION_EDGE"},
                    "recommended_press_below_surface_mm": 1.0,
                    "surface_shape": surface_shape,
                    "interpretation": (
                        "A narrow ridge/spike may be a rolled wrinkle; height alone "
                        "does not prove a separable free ply. Use the shallow "
                        "compression probe, then require a vertical lift/hold and "
                        "short relative-motion check before transport."
                        if surface_shape == "NARROW_RIDGE_OR_SPIKE"
                        else "A mixed/occlusion edge may collapse under shallow "
                        "compression; verify independent motion before transport."
                        if surface_shape == "MIXED_OR_OCCLUSION_EDGE"
                        else "Shape is only a geometric prior; verify independent "
                        "motion of the intended layer before a long pull."
                    ),
                }
        if include_nearest_reference:
            nearest = self.nearest_reference(label, x_value, y_value)
            result["nearest_reference"] = {
                key: nearest[key]
                for key in (
                    "reference_id",
                    "pixel_xy",
                    "pixel_distance",
                    "base_xyz_mm",
                    "height_above_table_mm",
                )
            }
        return result


REFERENCE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "lookup_reference",
        "description": (
            "Look up the exact saved robot-base XYZ, pixel, table height, and garment "
            "height for the one Camera A/B Rxxx reference already selected by Claude. "
            "Call exactly once, only after visual reasoning is complete. This tool is "
            "not for comparing, ranking, scanning, or searching references."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "camera": {"type": "string", "enum": ["A", "B"]},
                "reference_id": {"type": "string", "pattern": "^R[0-9]{3,}$"},
            },
            "required": ["camera", "reference_id"],
            "additionalProperties": False,
        },
    },
]

PIXEL_TOOLS: list[dict[str, Any]] = [
    {
        "name": "sample_local_surface",
        "description": (
            "Measure robust calibrated robot-base XYZ and table-relative height around "
            "one final Camera A/B image pixel already selected by Claude from the full "
            "visual scene. Call exactly once, only after visual reasoning is complete. "
            "This tool is not for scanning, comparing, ranking, or searching pixels. "
            "The result includes a triage-only surface_shape_diagnostic; when it sets "
            "compression_probe_recommended=true, close at the recommended shallow depth "
            "and inspect whether the peak collapses toward its neighbours before any "
            "lateral transport."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "camera": {"type": "string", "enum": ["A", "B"]},
                "x_px": {"type": "integer", "minimum": 0},
                "y_px": {"type": "integer", "minimum": 0},
                "radius_px": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 25,
                    "default": 3,
                },
            },
            "required": ["camera", "x_px", "y_px"],
            "additionalProperties": False,
        },
    },
]

# Backward-compatible module constant for callers that use the original
# reference-grounding server mode.
TOOLS = REFERENCE_TOOLS


def _tool_result(data: Any, *, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            }
        ],
        "isError": bool(is_error),
    }


def _call_tool(grounding: GarmentGrounding, name: str, arguments: dict[str, Any]) -> Any:
    if name == "lookup_reference":
        return grounding.lookup_reference(**arguments)
    if name == "sample_local_surface":
        return grounding.sample_local_surface(
            **arguments,
            include_nearest_reference=False,
        )
    raise GroundingToolError(f"unknown grounding tool: {name}")


def serve_stdio(grounding: GarmentGrounding, *, mode: str = "reference") -> None:
    """Serve the minimal MCP JSON-RPC tool protocol over stdin/stdout."""

    if mode not in {"reference", "pixel"}:
        raise ValueError("grounding mode must be reference or pixel")
    tools = REFERENCE_TOOLS if mode == "reference" else PIXEL_TOOLS
    allowed_tool = "lookup_reference" if mode == "reference" else "sample_local_surface"
    successful_lookup_count = 0
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"parse error: {exc}"},
            }
            print(json.dumps(response, separators=(",", ":")), flush=True)
            continue
        request_id = message.get("id")
        method = message.get("method")
        if request_id is None:
            continue
        try:
            if method == "initialize":
                requested_version = str(
                    (message.get("params") or {}).get(
                        "protocolVersion", "2024-11-05"
                    )
                )
                result = {
                    "protocolVersion": requested_version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "instructions": (
                        (
                            "Choose one image pixel from the complete visual scene before "
                            "using this server. Call the single local-surface tool exactly "
                            "once at the end of planning, then compose the final run."
                            if mode == "pixel"
                            else
                            "Choose one Rxxx visually before using this server. Call the "
                            "single lookup tool exactly once at the end of planning, then "
                            "compose the final run."
                        )
                        + " The measurement is not a grasp recommendation and never "
                        "authorizes robot motion."
                    ),
                }
            elif method == "tools/list":
                result = {"tools": tools}
            elif method == "tools/call":
                params = message.get("params") or {}
                name = str(params.get("name", ""))
                arguments = params.get("arguments") or {}
                if not isinstance(arguments, dict):
                    raise GroundingToolError("tool arguments must be an object")
                if name != allowed_tool:
                    raise GroundingToolError(
                        f"only {allowed_tool} is exposed in {mode} grounding mode"
                    )
                if successful_lookup_count >= 1:
                    raise GroundingToolError(
                        "the one successful coordinate lookup has already been used; "
                        "return the final proposal without another tool call"
                    )
                measurement = _call_tool(grounding, name, arguments)
                successful_lookup_count += 1
                measurement["lookup_budget_remaining"] = 0
                measurement["next_step"] = (
                    (
                        "Use this chosen pixel's local surface measurement to compose the "
                        "final proposal now; do not call another coordinate tool."
                        if mode == "pixel"
                        else
                        "Use this chosen Rxxx measurement to compose the final proposal now; "
                        "do not call another coordinate tool."
                    )
                )
                result = _tool_result(measurement)
            elif method == "ping":
                result = {}
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"method not found: {method}"},
                }
                print(json.dumps(response, separators=(",", ":")), flush=True)
                continue
            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
        except BaseException as exc:
            if method == "tools/call":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": _tool_result(
                        {"error": f"{type(exc).__name__}: {exc}"}, is_error=True
                    ),
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"},
                }
        print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=("reference", "pixel"),
        default="reference",
        help="expose either legacy Rxxx lookup or one arbitrary-pixel surface lookup",
    )
    args = parser.parse_args(argv)
    try:
        grounding = GarmentGrounding(Path(args.perception_dir))
    except BaseException as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    serve_stdio(grounding, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
