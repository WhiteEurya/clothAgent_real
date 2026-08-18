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
        result = {
            "measurement_kind": str(
                guide.get("measurement_kind", "uniform_calibrated_reference")
            ),
            "camera": label,
            "reference_id": identifier,
            "pixel_xy": [int(value) for value in match["pixel_xy"]],
            "base_xyz_mm": xyz,
            "table_z_mm": float(xyz[2] - height),
            "height_above_table_mm": height,
            "coordinate_frame": str(guide.get("coordinate_frame", "robot_base_mm")),
            "reference_semantics": str(guide.get("reference_semantics", "")),
            "valid": True,
            "warning": str(
                guide.get(
                    "warning",
                    "Measured coordinate only; this reference is not a ranked grasp candidate.",
                )
            ),
        }
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
    ) -> dict[str, Any]:
        """Return robust local XYZ/height statistics around a selected pixel."""

        label, xyz, height_map, table_map = self._pixel_arrays(camera)
        x_value, y_value = self._validate_pixel(xyz, x_px, y_px)
        radius = int(radius_px)
        if radius < 0 or radius > 25:
            raise GroundingToolError("radius_px must be between 0 and 25")
        y0, y1 = max(0, y_value - radius), min(xyz.shape[0], y_value + radius + 1)
        x0, x1 = max(0, x_value - radius), min(xyz.shape[1], x_value + radius + 1)
        local_xyz = np.asarray(xyz[y0:y1, x0:x1], dtype=np.float64).reshape(-1, 3)
        valid = np.all(np.isfinite(local_xyz), axis=1)
        points = local_xyz[valid]
        if not len(points):
            return {
                "measurement_kind": "robust_local_calibrated_surface",
                "camera": label,
                "query_pixel_xy": [x_value, y_value],
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


TOOLS: list[dict[str, Any]] = [
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
    raise GroundingToolError(f"unknown grounding tool: {name}")


def serve_stdio(grounding: GarmentGrounding) -> None:
    """Serve the minimal MCP JSON-RPC tool protocol over stdin/stdout."""

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
                        "Choose one Rxxx visually before using this server. Call the "
                        "single lookup tool exactly once at the end of planning, then "
                        "compose the final run. The measurement is not a grasp "
                        "recommendation and never authorizes robot motion."
                    ),
                }
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                params = message.get("params") or {}
                name = str(params.get("name", ""))
                arguments = params.get("arguments") or {}
                if not isinstance(arguments, dict):
                    raise GroundingToolError("tool arguments must be an object")
                if name != "lookup_reference":
                    raise GroundingToolError(
                        "only lookup_reference is exposed in final-grounding mode"
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
                    "Use this chosen Rxxx measurement to compose the final proposal now; "
                    "do not call another coordinate tool."
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
    args = parser.parse_args(argv)
    try:
        grounding = GarmentGrounding(Path(args.perception_dir))
    except BaseException as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    serve_stdio(grounding)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
