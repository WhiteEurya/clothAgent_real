"""Primary-view MolmoPoint localization with auxiliary-camera depth support.

MolmoPoint consumes camera A RGB and returns one semantic garment pixel. Camera
B may see only a partial garment, so it is not asked for an independent center.
Instead, its calibrated RGB-D point cloud is reprojected into camera A and used
to estimate depth along A's semantic ray. No perception output reaches an
experiment until same-point depth checks and robot workspace checks pass.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import numpy as np
except ImportError:  # Keep non-perception CLI commands usable without NumPy.
    np = None  # type: ignore[assignment]

from .config import ExperimentConfig, RobotConfig


class PerceptionError(RuntimeError):
    """Hard perception failure; robot execution must not proceed."""


class AuxiliaryDepthUnavailable(PerceptionError):
    """The auxiliary camera cannot observe the primary semantic surface."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def _require_numpy():
    if np is None:
        raise PerceptionError("NumPy is required for two-camera perception; use the configured cali environment")
    return np


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class CameraSpec:
    label: str
    serial: str
    extrinsics_file: Path


@dataclass(frozen=True)
class MolmoConfig:
    python: Path
    model: str = "allenai/MolmoPoint-8B"
    dtype: str = "bf16"
    max_crops: int = 1
    max_new_tokens: int = 96
    timeout_s: int = 600
    local_files_only: bool = True


@dataclass(frozen=True)
class PerceptionConfig:
    cameras: tuple[CameraSpec, CameraSpec]
    molmo: MolmoConfig
    active_camera_labels: tuple[str, ...] = ("A", "B")
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 10
    depth_window_radius_px: int = 4
    min_depth_m: float = 0.15
    max_depth_m: float = 2.0
    max_view_disagreement_mm: float = 50.0
    grasp_contact_clearance_mm: float = 0.0
    approach_clearance_mm: float = 80.0
    lift_clearance_mm: float = 160.0

    @classmethod
    def load(cls, project_root: Path, path: Path) -> "PerceptionConfig":
        project_root = project_root.resolve()
        raw = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        camera_values = raw.get("cameras", [])
        if len(camera_values) != 2:
            raise PerceptionError("perception config must contain exactly two cameras")
        cameras: list[CameraSpec] = []
        serials: set[str] = set()
        labels: set[str] = set()
        for item in camera_values:
            label = str(item["label"]).strip().upper()
            serial = str(item["serial"]).strip()
            extrinsics = Path(item["extrinsics_file"])
            if not extrinsics.is_absolute():
                extrinsics = project_root / extrinsics
            if not label or not serial or serial in serials or label in labels:
                raise PerceptionError("camera labels/serials must be non-empty and unique")
            if not extrinsics.is_file():
                raise FileNotFoundError(extrinsics)
            serials.add(serial)
            labels.add(label)
            cameras.append(CameraSpec(label, serial, extrinsics.resolve()))

        molmo_raw = raw.get("molmo", {})
        active_camera_labels = tuple(
            str(label).strip().upper()
            for label in raw.get("active_cameras", [camera.label for camera in cameras])
        )
        molmo_python = Path(
            molmo_raw.get("python", "/home/CNS2026330003/miniconda3/envs/molmo/bin/python")
        ).expanduser()
        config = cls(
            cameras=(cameras[0], cameras[1]),
            molmo=MolmoConfig(
                python=molmo_python.resolve(),
                model=str(molmo_raw.get("model", "allenai/MolmoPoint-8B")),
                dtype=str(molmo_raw.get("dtype", "bf16")),
                max_crops=int(molmo_raw.get("max_crops", 1)),
                max_new_tokens=int(molmo_raw.get("max_new_tokens", 96)),
                timeout_s=int(molmo_raw.get("timeout_s", 600)),
                local_files_only=bool(molmo_raw.get("local_files_only", True)),
            ),
            active_camera_labels=active_camera_labels,
            width=int(raw.get("width", 640)),
            height=int(raw.get("height", 480)),
            fps=int(raw.get("fps", 30)),
            warmup_frames=int(raw.get("warmup_frames", 10)),
            depth_window_radius_px=int(raw.get("depth_window_radius_px", 4)),
            min_depth_m=float(raw.get("min_depth_m", 0.15)),
            max_depth_m=float(raw.get("max_depth_m", 2.0)),
            max_view_disagreement_mm=float(raw.get("max_view_disagreement_mm", 50.0)),
            grasp_contact_clearance_mm=float(raw.get("grasp_contact_clearance_mm", 0.0)),
            approach_clearance_mm=float(raw.get("approach_clearance_mm", 80.0)),
            lift_clearance_mm=float(raw.get("lift_clearance_mm", 160.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        configured_labels = {camera.label for camera in self.cameras}
        if len(self.active_camera_labels) not in {1, 2}:
            raise PerceptionError("active_cameras must select one or two configured cameras")
        if len(set(self.active_camera_labels)) != len(self.active_camera_labels):
            raise PerceptionError("active_cameras must be unique")
        if not set(self.active_camera_labels).issubset(configured_labels):
            raise PerceptionError(
                f"active_cameras {list(self.active_camera_labels)} are not a subset of {sorted(configured_labels)}"
            )
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise PerceptionError("camera width/height/fps must be positive")
        if self.warmup_frames < 0 or self.depth_window_radius_px < 0:
            raise PerceptionError("warmup frames and depth radius must be non-negative")
        if not 0 < self.min_depth_m < self.max_depth_m:
            raise PerceptionError("expected 0 < min_depth_m < max_depth_m")
        if self.max_view_disagreement_mm <= 0:
            raise PerceptionError("max_view_disagreement_mm must be positive")
        if self.grasp_contact_clearance_mm < 0:
            raise PerceptionError("grasp_contact_clearance_mm must be non-negative")
        if self.approach_clearance_mm <= 0:
            raise PerceptionError("approach_clearance_mm must be positive")
        if self.lift_clearance_mm < self.approach_clearance_mm:
            raise PerceptionError(
                "lift_clearance_mm must be greater than or equal to approach_clearance_mm"
            )
        if self.molmo.dtype not in {"bf16", "fp16", "fp32"}:
            raise PerceptionError("Molmo dtype must be bf16, fp16, or fp32")
        if self.molmo.max_crops <= 0 or self.molmo.max_new_tokens <= 0 or self.molmo.timeout_s <= 0:
            raise PerceptionError("Molmo limits and timeout must be positive")
        if not self.molmo.python.is_file():
            raise FileNotFoundError(self.molmo.python)


@dataclass(frozen=True)
class RGBDFrame:
    label: str
    serial: str
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    X_base_camera: np.ndarray


@dataclass(frozen=True)
class AuxiliaryDepthEstimate:
    """Depth for one primary-camera semantic ray, observed by another camera."""

    point_base_mm: np.ndarray
    source_pixel_xy: tuple[float, float]
    depth_m: float
    candidate_count: int
    clustered_count: int
    projection_error_px_median: float
    surface_spread_mm: float


@dataclass(frozen=True)
class PrimaryDepthQuality:
    """Local quality evidence required before using primary-camera depth alone."""

    radius_px: int
    valid_count: int
    sample_count: int
    valid_fraction: float
    median_depth_m: float
    selected_depth_m: float
    selected_to_median_mm: float
    p10_depth_m: float
    p90_depth_m: float
    spread_mm: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "radius_px": self.radius_px,
            "valid_count": self.valid_count,
            "sample_count": self.sample_count,
            "valid_fraction": self.valid_fraction,
            "median_depth_m": self.median_depth_m,
            "selected_depth_m": self.selected_depth_m,
            "selected_to_median_mm": self.selected_to_median_mm,
            "p10_depth_m": self.p10_depth_m,
            "p90_depth_m": self.p90_depth_m,
            "spread_mm": self.spread_mm,
        }


def load_extrinsics(path: Path, key: str = "X_CammountCam") -> np.ndarray:
    numpy = _require_numpy()
    try:
        import yaml
    except ImportError as exc:
        raise PerceptionError("PyYAML is required to load camera extrinsics") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if key not in data:
        raise PerceptionError(f"{path} does not contain {key}")
    transform = numpy.asarray(data[key], dtype=numpy.float64)
    if transform.shape != (4, 4) or not numpy.allclose(transform[3], [0, 0, 0, 1], atol=1e-6):
        raise PerceptionError(f"{path}:{key} must be a 4x4 homogeneous transform")
    return transform


class RealSenseRGBD:
    """Small aligned RGB-D adapter matching the existing calibration scripts."""

    def __init__(self, spec: CameraSpec, width: int, height: int, fps: int):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise PerceptionError("pyrealsense2 is required in the camera/robot Python environment") from exc
        self.rs = rs
        self.spec = spec
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.depth_scale: float | None = None
        self.intrinsics: np.ndarray | None = None
        self.started = False

    def start(self) -> None:
        numpy = _require_numpy()
        rs = self.rs
        self.config.enable_device(self.spec.serial)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        profile = self.pipeline.start(self.config)
        self.started = True
        self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.intrinsics = numpy.asarray(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=numpy.float64,
        )

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        numpy = _require_numpy()
        if not self.started or self.depth_scale is None:
            raise PerceptionError(f"camera {self.spec.label} has not been started")
        frames = self.align.process(self.pipeline.wait_for_frames())
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise PerceptionError(f"camera {self.spec.label} returned an incomplete RGB-D frame")
        rgb = numpy.asanyarray(color.get_data()).copy()
        depth_m = numpy.asanyarray(depth.get_data()).astype(numpy.float32) * self.depth_scale
        return rgb, depth_m

    def stop(self) -> None:
        if self.started:
            try:
                self.pipeline.stop()
            finally:
                self.started = False


def capture_two_view_rgbd(config: PerceptionConfig) -> list[RGBDFrame]:
    active_labels = set(config.active_camera_labels)
    active_specs = [spec for spec in config.cameras if spec.label in active_labels]
    cameras = [RealSenseRGBD(spec, config.width, config.height, config.fps) for spec in active_specs]
    available = {
        device.get_info(cameras[0].rs.camera_info.serial_number)
        for device in cameras[0].rs.context().query_devices()
    }
    active_serials = {camera.spec.serial for camera in cameras}
    missing = sorted(active_serials - available)
    if missing:
        raise PerceptionError(
            f"configured RealSense serials are not connected: {missing}; available={sorted(available)}"
        )
    try:
        for camera in cameras:
            camera.start()
        for _ in range(config.warmup_frames):
            for camera in cameras:
                camera.read()
        frames: list[RGBDFrame] = []
        for camera in cameras:
            rgb, depth = camera.read()
            if camera.intrinsics is None:
                raise PerceptionError(f"camera {camera.spec.label} intrinsics are unavailable")
            frames.append(
                RGBDFrame(
                    label=camera.spec.label,
                    serial=camera.spec.serial,
                    rgb=rgb,
                    depth_m=depth,
                    intrinsics=camera.intrinsics.copy(),
                    X_base_camera=load_extrinsics(camera.spec.extrinsics_file),
                )
            )
        return frames
    finally:
        for camera in reversed(cameras):
            camera.stop()


class MolmoPointClient:
    """Run MolmoPoint in its existing Conda environment as a bounded subprocess."""

    def __init__(self, project_root: Path, config: MolmoConfig):
        self.project_root = project_root.resolve()
        self.config = config

    def locate(self, image_paths: list[Path], output_path: Path, prompt: str) -> dict[str, Any]:
        if len(image_paths) not in {1, 2}:
            raise PerceptionError("cloth-center localization requires one or two images")
        worker = self.project_root / "cloth_agent" / "molmo_worker.py"
        command = [
            str(self.config.python),
            str(worker),
            "--output",
            str(output_path),
            "--model",
            self.config.model,
            "--dtype",
            self.config.dtype,
            "--max-crops",
            str(self.config.max_crops),
            "--max-new-tokens",
            str(self.config.max_new_tokens),
            "--prompt",
            prompt,
        ]
        for image_path in image_paths:
            command.extend(["--image", str(image_path)])
        if self.config.local_files_only:
            command.append("--local-files-only")
        try:
            completed = subprocess.run(
                command,
                cwd=self.project_root,
                text=True,
                capture_output=True,
                timeout=self.config.timeout_s,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PerceptionError(f"MolmoPoint invocation failed: {exc}") from exc
        log_path = output_path.with_suffix(".stdout.txt")
        log_path.write_text(completed.stdout + ("\n" + completed.stderr if completed.stderr else ""), encoding="utf-8")
        if completed.returncode != 0:
            raise PerceptionError(
                f"MolmoPoint exited with {completed.returncode}; inspect {log_path}"
            )
        if not output_path.is_file():
            raise PerceptionError("MolmoPoint completed without producing its JSON output")
        return json.loads(output_path.read_text(encoding="utf-8"))


def robust_depth_at_pixel(
    depth_m: np.ndarray,
    x_px: float,
    y_px: float,
    radius: int,
    min_depth_m: float,
    max_depth_m: float,
) -> float:
    numpy = _require_numpy()
    if depth_m.ndim != 2:
        raise PerceptionError("aligned depth must be a 2D array")
    height, width = depth_m.shape
    if not (math.isfinite(x_px) and math.isfinite(y_px) and 0 <= x_px < width and 0 <= y_px < height):
        raise PerceptionError(f"Molmo pixel ({x_px}, {y_px}) is outside a {width}x{height} image")
    cx, cy = int(round(x_px)), int(round(y_px))
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    values = depth_m[y0:y1, x0:x1].astype(numpy.float64).reshape(-1)
    valid = values[numpy.isfinite(values) & (values > min_depth_m) & (values < max_depth_m)]
    if valid.size == 0:
        raise PerceptionError(f"no valid aligned depth near pixel ({x_px:.1f}, {y_px:.1f})")
    return float(numpy.median(valid))


def primary_depth_quality_at_pixel(
    depth_m: np.ndarray,
    x_px: float,
    y_px: float,
    *,
    selected_depth_m: float,
    radius: int,
    min_depth_m: float,
    max_depth_m: float,
    max_spread_mm: float,
    min_valid_fraction: float = 0.5,
) -> PrimaryDepthQuality:
    """Validate a wider local depth patch before an A-only fallback is allowed."""

    numpy = _require_numpy()
    if depth_m.ndim != 2 or radius < 1:
        raise PerceptionError("primary depth quality requires a 2D depth map and positive radius")
    height, width = depth_m.shape
    if not (0 <= x_px < width and 0 <= y_px < height):
        raise PerceptionError("primary depth quality pixel is outside the image")
    cx, cy = int(round(x_px)), int(round(y_px))
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    samples = depth_m[y0:y1, x0:x1].astype(numpy.float64).reshape(-1)
    valid = samples[
        numpy.isfinite(samples)
        & (samples > min_depth_m)
        & (samples < max_depth_m)
    ]
    valid_fraction = float(valid.size / samples.size) if samples.size else 0.0
    if valid.size < 12 or valid_fraction < min_valid_fraction:
        raise PerceptionError(
            f"primary camera depth fallback has only {valid.size}/{samples.size} valid "
            f"samples ({valid_fraction:.1%}) near the Molmo point"
        )
    p10, median, p90 = (float(value) for value in numpy.percentile(valid, [10, 50, 90]))
    spread_mm = (p90 - p10) * 1000.0
    selected_to_median_mm = abs(float(selected_depth_m) - median) * 1000.0
    if spread_mm > max_spread_mm:
        raise PerceptionError(
            f"primary camera depth fallback crosses an unstable fold/edge: local p10-p90 "
            f"spread is {spread_mm:.1f} mm, limit is {max_spread_mm:.1f} mm"
        )
    if selected_to_median_mm > max_spread_mm:
        raise PerceptionError(
            f"primary camera selected depth differs from its local median by "
            f"{selected_to_median_mm:.1f} mm, limit is {max_spread_mm:.1f} mm"
        )
    return PrimaryDepthQuality(
        radius_px=radius,
        valid_count=int(valid.size),
        sample_count=int(samples.size),
        valid_fraction=valid_fraction,
        median_depth_m=median,
        selected_depth_m=float(selected_depth_m),
        selected_to_median_mm=selected_to_median_mm,
        p10_depth_m=p10,
        p90_depth_m=p90,
        spread_mm=spread_mm,
    )


def pixel_to_base_mm(x_px: float, y_px: float, depth_m: float, K: np.ndarray, X_base_camera: np.ndarray) -> np.ndarray:
    numpy = _require_numpy()
    if K.shape != (3, 3):
        raise PerceptionError("camera intrinsics must be 3x3")
    x_cam = (x_px - K[0, 2]) * depth_m / K[0, 0]
    y_cam = (y_px - K[1, 2]) * depth_m / K[1, 1]
    point_camera = numpy.asarray([x_cam, y_cam, depth_m], dtype=numpy.float64)
    point_base_m = X_base_camera[:3, :3] @ point_camera + X_base_camera[:3, 3]
    return point_base_m * 1000.0


def camera_ray_base(
    x_px: float,
    y_px: float,
    K: np.ndarray,
    X_base_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a base-frame ray origin and unit direction for one image pixel."""

    numpy = _require_numpy()
    if K.shape != (3, 3) or X_base_camera.shape != (4, 4):
        raise PerceptionError("camera ray requires 3x3 intrinsics and a 4x4 extrinsic transform")
    ray_camera = numpy.asarray(
        [(x_px - K[0, 2]) / K[0, 0], (y_px - K[1, 2]) / K[1, 1], 1.0],
        dtype=numpy.float64,
    )
    direction = X_base_camera[:3, :3] @ ray_camera
    norm = float(numpy.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 0:
        raise PerceptionError("camera ray direction is invalid")
    return X_base_camera[:3, 3].astype(numpy.float64), direction / norm


def auxiliary_depth_on_primary_ray(
    primary: RGBDFrame,
    auxiliary: RGBDFrame,
    x_px: float,
    y_px: float,
    *,
    projection_radius_px: float,
    min_depth_m: float,
    max_depth_m: float,
    cluster_width_mm: float,
    min_points: int,
    reference_point_base_mm: np.ndarray | None = None,
) -> AuxiliaryDepthEstimate:
    """Reproject auxiliary RGB-D points and estimate depth along a primary semantic ray.

    The auxiliary camera is not asked to find a second garment center. Its visible
    point cloud is transformed into the primary image, and only points that land
    near the primary Molmo pixel are considered. A densest local depth cluster is
    selected before the final point is placed exactly on the primary camera ray.
    """

    numpy = _require_numpy()
    if projection_radius_px <= 0 or cluster_width_mm <= 0 or min_points <= 0:
        raise PerceptionError("auxiliary depth fusion limits must be positive")
    if auxiliary.depth_m.ndim != 2 or auxiliary.rgb.shape[:2] != auxiliary.depth_m.shape:
        raise PerceptionError(f"camera {auxiliary.label} has invalid aligned RGB-D shapes")

    depth = auxiliary.depth_m.astype(numpy.float64)
    valid = (
        numpy.isfinite(depth)
        & (depth > min_depth_m)
        & (depth < max_depth_m)
    )
    y_aux, x_aux = numpy.nonzero(valid)
    if x_aux.size == 0:
        raise PerceptionError(f"camera {auxiliary.label} has no valid depth points")
    z_aux = depth[valid]
    K_aux = auxiliary.intrinsics
    points_aux = numpy.stack(
        [
            (x_aux - K_aux[0, 2]) * z_aux / K_aux[0, 0],
            (y_aux - K_aux[1, 2]) * z_aux / K_aux[1, 1],
            z_aux,
        ],
        axis=1,
    )
    points_base = (
        points_aux @ auxiliary.X_base_camera[:3, :3].T
        + auxiliary.X_base_camera[:3, 3]
    )

    rotation_primary = primary.X_base_camera[:3, :3]
    translation_primary = primary.X_base_camera[:3, 3]
    points_primary = (points_base - translation_primary) @ rotation_primary
    z_primary = points_primary[:, 2]
    in_front = numpy.isfinite(z_primary) & (z_primary > min_depth_m)
    points_base = points_base[in_front]
    points_primary = points_primary[in_front]
    x_aux = x_aux[in_front]
    y_aux = y_aux[in_front]
    z_aux = z_aux[in_front]
    if points_primary.shape[0] == 0:
        raise PerceptionError(
            f"camera {auxiliary.label} depth does not project in front of camera {primary.label}"
        )

    K_primary = primary.intrinsics
    projected_x = K_primary[0, 0] * points_primary[:, 0] / points_primary[:, 2] + K_primary[0, 2]
    projected_y = K_primary[1, 1] * points_primary[:, 1] / points_primary[:, 2] + K_primary[1, 2]
    projection_error = numpy.hypot(projected_x - x_px, projected_y - y_px)
    near_ray = numpy.isfinite(projection_error) & (projection_error <= projection_radius_px)
    points_base = points_base[near_ray]
    x_aux = x_aux[near_ray]
    y_aux = y_aux[near_ray]
    z_aux = z_aux[near_ray]
    projection_error = projection_error[near_ray]
    candidate_count = int(points_base.shape[0])
    if candidate_count < min_points:
        reference_note = ""
        diagnostics: dict[str, Any] = {
            "reason": "insufficient_reprojected_points",
            "candidate_count": candidate_count,
            "required_points": min_points,
        }
        if reference_point_base_mm is not None:
            reference_base_m = numpy.asarray(
                reference_point_base_mm, dtype=numpy.float64
            ) / 1000.0
            if reference_base_m.shape == (3,) and numpy.all(numpy.isfinite(reference_base_m)):
                reference_aux = (
                    reference_base_m - auxiliary.X_base_camera[:3, 3]
                ) @ auxiliary.X_base_camera[:3, :3]
                if reference_aux[2] > 0:
                    reference_u = (
                        auxiliary.intrinsics[0, 0]
                        * reference_aux[0]
                        / reference_aux[2]
                        + auxiliary.intrinsics[0, 2]
                    )
                    reference_v = (
                        auxiliary.intrinsics[1, 1]
                        * reference_aux[1]
                        / reference_aux[2]
                        + auxiliary.intrinsics[1, 2]
                    )
                    height, width = auxiliary.depth_m.shape
                    visibility = (
                        "inside"
                        if 0 <= reference_u < width and 0 <= reference_v < height
                        else "outside"
                    )
                    reference_note = (
                        f"; the primary depth estimate projects to {auxiliary.label} pixel "
                        f"({reference_u:.1f}, {reference_v:.1f}), {visibility} its "
                        f"{width}x{height} image"
                    )
                    diagnostics["primary_depth_projection_in_auxiliary"] = {
                        "pixel_xy": [float(reference_u), float(reference_v)],
                        "visibility": visibility,
                        "image_size": [width, height],
                    }
        raise AuxiliaryDepthUnavailable(
            f"camera {auxiliary.label} provides only {candidate_count} depth points near "
            f"camera {primary.label} Molmo pixel; need at least {min_points}"
            f"{reference_note}",
            diagnostics,
        )

    ray_origin, ray_direction = camera_ray_base(
        x_px, y_px, primary.intrinsics, primary.X_base_camera
    )
    ray_parameter = (points_base - ray_origin) @ ray_direction
    positive = numpy.isfinite(ray_parameter) & (ray_parameter > min_depth_m)
    points_base = points_base[positive]
    x_aux = x_aux[positive]
    y_aux = y_aux[positive]
    z_aux = z_aux[positive]
    projection_error = projection_error[positive]
    ray_parameter = ray_parameter[positive]
    if ray_parameter.size < min_points:
        raise AuxiliaryDepthUnavailable(
            f"camera {auxiliary.label} has too few positive-depth points on the primary ray",
            {
                "reason": "insufficient_positive_depth_points",
                "candidate_count": int(ray_parameter.size),
                "required_points": min_points,
            },
        )

    order = numpy.argsort(ray_parameter)
    sorted_parameter = ray_parameter[order]
    width_m = cluster_width_mm / 1000.0
    best_start = 0
    best_end = 0
    right = 0
    for left in range(sorted_parameter.size):
        if right < left:
            right = left
        while (
            right < sorted_parameter.size
            and sorted_parameter[right] - sorted_parameter[left] <= width_m
        ):
            right += 1
        if right - left > best_end - best_start:
            best_start, best_end = left, right
    cluster_order = order[best_start:best_end]
    if cluster_order.size < min_points:
        raise AuxiliaryDepthUnavailable(
            f"camera {auxiliary.label} has no stable auxiliary depth cluster near the primary point",
            {
                "reason": "no_stable_depth_cluster",
                "candidate_count": candidate_count,
                "clustered_count": int(cluster_order.size),
                "required_points": min_points,
            },
        )

    clustered_parameter = ray_parameter[cluster_order]
    parameter_m = float(numpy.median(clustered_parameter))
    point_base_mm = (ray_origin + parameter_m * ray_direction) * 1000.0
    source_x = float(numpy.median(x_aux[cluster_order]))
    source_y = float(numpy.median(y_aux[cluster_order]))
    surface_spread_mm = float(
        (numpy.percentile(clustered_parameter, 90) - numpy.percentile(clustered_parameter, 10))
        * 1000.0
    )
    return AuxiliaryDepthEstimate(
        point_base_mm=point_base_mm,
        source_pixel_xy=(source_x, source_y),
        depth_m=float(numpy.median(z_aux[cluster_order])),
        candidate_count=candidate_count,
        clustered_count=int(cluster_order.size),
        projection_error_px_median=float(numpy.median(projection_error[cluster_order])),
        surface_spread_mm=surface_spread_mm,
    )


def points_by_image(raw_points: Any, image_count: int = 2) -> dict[int, list[float]]:
    if not isinstance(raw_points, list):
        raise PerceptionError("Molmo points must be a JSON list")
    grouped: dict[int, list[list[float]]] = {index: [] for index in range(image_count)}
    for raw in raw_points:
        if not isinstance(raw, list) or len(raw) < 4:
            raise PerceptionError(f"invalid Molmo point record: {raw!r}")
        image_index = int(raw[-3])
        x_px, y_px = float(raw[-2]), float(raw[-1])
        if image_index not in grouped or not math.isfinite(x_px) or not math.isfinite(y_px):
            raise PerceptionError(f"invalid Molmo image index/pixel: {raw!r}")
        grouped[image_index].append([x_px, y_px])
    result: dict[int, list[float]] = {}
    for image_index, candidates in grouped.items():
        if len(candidates) != 1:
            raise PerceptionError(
                f"expected exactly one cloth-center point for image {image_index}, got {len(candidates)}"
            )
        result[image_index] = candidates[0]
    return result


def derive_grasp_plan(
    center_base_mm: np.ndarray,
    robot_config: RobotConfig,
    perception_config: PerceptionConfig,
) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Convert one validated garment surface point into a safe motion plan.

    The controller already expresses Cartesian poses at its configured TCP, so
    the physical 172 mm tool length is not added here.  It is independently
    checked when a real backend connects.
    """

    numpy = _require_numpy()
    center = numpy.asarray(center_base_mm, dtype=numpy.float64)
    if center.shape != (3,) or not numpy.all(numpy.isfinite(center)):
        raise PerceptionError("fused cloth center must be a finite xyz point")

    bounds = robot_config.boundaries
    if bounds.z_min is None or bounds.z_max is None:
        raise PerceptionError("z_min and z_max are required to derive grasp motion")
    safe_z_min = float(bounds.z_min + robot_config.lower_z_margin_mm)
    safe_z_max = float(bounds.z_max - robot_config.workspace_margin_mm)
    if safe_z_min >= safe_z_max:
        raise PerceptionError("workspace margin leaves no usable z range")

    surface_z = float(center[2])
    requested_grasp_z = surface_z + perception_config.grasp_contact_clearance_mm
    grasp_z = max(requested_grasp_z, safe_z_min)
    approach_z = grasp_z + perception_config.approach_clearance_mm
    requested_lift_z = grasp_z + perception_config.lift_clearance_mm
    lift_z = min(requested_lift_z, safe_z_max)
    if approach_z > safe_z_max:
        raise PerceptionError(
            f"automatic approach_z={approach_z:.1f} mm exceeds safe upper z={safe_z_max:.1f} mm"
        )
    if lift_z < approach_z:
        raise PerceptionError(
            f"automatic lift_z={lift_z:.1f} mm is below approach_z={approach_z:.1f} mm"
        )

    yaw = float(robot_config.init_pose_mm_deg[5])
    plan = ExperimentConfig(
        cloth_center_x=float(center[0]),
        cloth_center_y=float(center[1]),
        grasp_z=float(grasp_z),
        approach_z=float(approach_z),
        lift_z=float(lift_z),
        yaw_deg=yaw,
    )
    plan.require_ready()
    for z in (grasp_z, approach_z, lift_z):
        bounds.validate(
            plan.cloth_center_x,
            plan.cloth_center_y,
            z,
            robot_config.workspace_margin_mm,
            z_lower_margin_mm=robot_config.lower_z_margin_mm,
        )
    derivation = {
        "surface_z_mm": surface_z,
        "tcp_coordinates": "controller-configured tool center point",
        "grasp_contact_clearance_mm": perception_config.grasp_contact_clearance_mm,
        "requested_grasp_z_mm": requested_grasp_z,
        "safe_grasp_z_mm": grasp_z,
        "lower_z_safety_adjustment_mm": grasp_z - requested_grasp_z,
        "approach_clearance_mm": perception_config.approach_clearance_mm,
        "approach_z_mm": approach_z,
        "lift_clearance_mm": perception_config.lift_clearance_mm,
        "requested_lift_z_mm": requested_lift_z,
        "safe_lift_z_mm": lift_z,
        "upper_z_safety_adjustment_mm": lift_z - requested_lift_z,
        "yaw_deg": yaw,
        "yaw_source": "robot observation pose",
    }
    return plan, derivation


class ClothCenterPerception:
    def __init__(
        self,
        project_root: Path,
        robot_config: RobotConfig,
        config: PerceptionConfig,
        capture: Callable[[PerceptionConfig], list[RGBDFrame]] = capture_two_view_rgbd,
        molmo_client: MolmoPointClient | None = None,
    ):
        self.project_root = project_root.resolve()
        self.robot_config = robot_config
        self.config = config
        self.capture = capture
        self.molmo = molmo_client or MolmoPointClient(self.project_root, config.molmo)

    def locate(
        self,
        output_dir: Path,
        experiment: ExperimentConfig,
        frames: list[RGBDFrame] | None = None,
    ) -> tuple[dict[str, Any], ExperimentConfig]:
        numpy = _require_numpy()
        from PIL import Image, ImageDraw

        output_dir.mkdir(parents=True, exist_ok=False)
        frames = self.capture(self.config) if frames is None else frames
        expected_labels = set(self.config.active_camera_labels)
        frame_labels = {frame.label for frame in frames}
        if len(frames) not in {1, 2} or frame_labels != expected_labels:
            raise PerceptionError(
                f"expected RGB-D frames {sorted(expected_labels)}, got {sorted(frame_labels)}"
            )
        frame_by_label = {frame.label: frame for frame in frames}
        image_paths_by_label: dict[str, Path] = {}
        for index, frame in enumerate(frames):
            if frame.rgb.shape[:2] != frame.depth_m.shape[:2]:
                raise PerceptionError(f"camera {frame.label} RGB/depth shapes do not match")
            image_path = output_dir / f"camera_{index}_{frame.label}.png"
            Image.fromarray(frame.rgb.astype(numpy.uint8)).save(image_path)
            numpy.save(output_dir / f"camera_{index}_{frame.label}_depth_m.npy", frame.depth_m)
            image_paths_by_label[frame.label] = image_path

        primary_label = self.config.active_camera_labels[0]
        primary = frame_by_label[primary_label]
        primary_image_path = image_paths_by_label[primary_label]

        prompt = (
            "Point to the geometric center of the entire garment laid on the table. "
            "Ignore the robot arm and gripper, and infer the garment boundary behind their occlusion. "
            "Point to the garment torso center, not the lower hem, a sleeve, the robot, "
            "or the image center. Return exactly one point. "
            "If the garment center cannot be inferred, do not invent a location."
        )
        molmo_output_path = output_dir / "molmo_output.json"
        molmo_result = self.molmo.locate([primary_image_path], molmo_output_path, prompt)
        x_px, y_px = points_by_image(molmo_result.get("points"), image_count=1)[0]
        primary_depth = robust_depth_at_pixel(
            primary.depth_m,
            x_px,
            y_px,
            self.config.depth_window_radius_px,
            self.config.min_depth_m,
            self.config.max_depth_m,
        )
        primary_point = pixel_to_base_mm(
            x_px,
            y_px,
            primary_depth,
            primary.intrinsics,
            primary.X_base_camera,
        )
        primary_annotated_name = f"{primary_image_path.stem}_annotated.png"
        primary_annotated = Image.open(primary_image_path).convert("RGB")
        primary_draw = ImageDraw.Draw(primary_annotated)
        radius = 8
        primary_draw.ellipse(
            (x_px - radius, y_px - radius, x_px + radius, y_px + radius),
            fill="red",
            outline="white",
            width=2,
        )
        primary_annotated.save(output_dir / primary_annotated_name)

        views: list[dict[str, Any]] = [
            {
                "label": primary.label,
                "role": "primary_semantic",
                "serial": primary.serial,
                "image": primary_image_path.name,
                "annotated_image": primary_annotated_name,
                "pixel_xy": [x_px, y_px],
                "depth_m": primary_depth,
                "intrinsics": primary.intrinsics.tolist(),
                "X_base_camera": primary.X_base_camera.tolist(),
                "center_base_mm": primary_point.tolist(),
            }
        ]

        disagreement: float | None = None
        perception_warnings: list[str] = []
        auxiliary_labels = list(self.config.active_camera_labels[1:])
        if auxiliary_labels:
            auxiliary_label = auxiliary_labels[0]
            auxiliary = frame_by_label[auxiliary_label]
            auxiliary_image_path = image_paths_by_label[auxiliary.label]
            projection_radius_px = max(
                4.0, float(self.config.depth_window_radius_px) * 2.0
            )
            cluster_width_mm = max(
                15.0, float(self.config.max_view_disagreement_mm) / 2.0
            )
            min_points = max(6, (self.config.depth_window_radius_px + 1) * 2)
            try:
                auxiliary_estimate = auxiliary_depth_on_primary_ray(
                    primary,
                    auxiliary,
                    x_px,
                    y_px,
                    projection_radius_px=projection_radius_px,
                    min_depth_m=self.config.min_depth_m,
                    max_depth_m=self.config.max_depth_m,
                    cluster_width_mm=cluster_width_mm,
                    min_points=min_points,
                    reference_point_base_mm=primary_point,
                )
            except AuxiliaryDepthUnavailable as exc:
                quality = primary_depth_quality_at_pixel(
                    primary.depth_m,
                    x_px,
                    y_px,
                    selected_depth_m=primary_depth,
                    radius=max(8, self.config.depth_window_radius_px * 3),
                    min_depth_m=self.config.min_depth_m,
                    max_depth_m=self.config.max_depth_m,
                    max_spread_mm=self.config.max_view_disagreement_mm,
                )
                views[0]["depth_quality"] = quality.as_dict()
                auxiliary_annotated_name = f"{auxiliary_image_path.stem}_annotated.png"
                auxiliary_annotated = Image.open(auxiliary_image_path).convert("RGB")
                auxiliary_draw = ImageDraw.Draw(auxiliary_annotated)
                projection_info = exc.diagnostics.get(
                    "primary_depth_projection_in_auxiliary", {}
                )
                projected_pixel = projection_info.get("pixel_xy")
                if (
                    isinstance(projected_pixel, list)
                    and len(projected_pixel) == 2
                    and projection_info.get("visibility") == "inside"
                ):
                    projected_x, projected_y = map(float, projected_pixel)
                    auxiliary_draw.ellipse(
                        (
                            projected_x - radius,
                            projected_y - radius,
                            projected_x + radius,
                            projected_y + radius,
                        ),
                        fill=(255, 180, 0),
                        outline="white",
                        width=2,
                    )
                auxiliary_annotated.save(output_dir / auxiliary_annotated_name)
                views.append(
                    {
                        "label": auxiliary.label,
                        "role": "auxiliary_depth_unavailable",
                        "serial": auxiliary.serial,
                        "image": auxiliary_image_path.name,
                        "annotated_image": auxiliary_annotated_name,
                        "pixel_xy": projected_pixel,
                        "depth_m": None,
                        "intrinsics": auxiliary.intrinsics.tolist(),
                        "X_base_camera": auxiliary.X_base_camera.tolist(),
                        "center_base_mm": None,
                        "availability": "occluded_or_outside_view",
                        "error": str(exc),
                        "diagnostics": exc.diagnostics,
                    }
                )
                center = primary_point.copy()
                perception_mode = "primary_rgbd_auxiliary_unavailable"
                status = "VALIDATED_PRIMARY_DEPTH_FALLBACK"
                warning = (
                    f"camera {auxiliary.label} could not observe the camera {primary.label} "
                    f"semantic surface; validated camera {primary.label} depth was used"
                )
                perception_warnings.append(warning)
                depth_fusion = {
                    "mode": "primary_depth_fallback_auxiliary_unavailable",
                    "selected_depth_camera": primary.label,
                    "primary_depth_camera": primary.label,
                    "primary_depth_point_base_mm": primary_point.tolist(),
                    "primary_depth_quality": quality.as_dict(),
                    "auxiliary_depth_camera": auxiliary.label,
                    "auxiliary_status": "occluded_or_outside_view",
                    "auxiliary_error": str(exc),
                    "auxiliary_diagnostics": exc.diagnostics,
                    "source_disagreement_mm": None,
                    "candidate_count": exc.diagnostics.get("candidate_count", 0),
                    "clustered_count": exc.diagnostics.get("clustered_count"),
                }
            else:
                disagreement = float(
                    numpy.linalg.norm(primary_point - auxiliary_estimate.point_base_mm)
                )
                if disagreement > self.config.max_view_disagreement_mm:
                    raise PerceptionError(
                        f"primary camera {primary.label} depth and auxiliary camera "
                        f"{auxiliary.label} depth for the same Molmo point disagree by "
                        f"{disagreement:.1f} mm, limit is "
                        f"{self.config.max_view_disagreement_mm:.1f} mm"
                    )
                center = auxiliary_estimate.point_base_mm.copy()
                perception_mode = "primary_rgb_auxiliary_depth"
                status = "VALIDATED_PRIMARY_AUX_DEPTH"

                auxiliary_annotated_name = f"{auxiliary_image_path.stem}_annotated.png"
                auxiliary_annotated = Image.open(auxiliary_image_path).convert("RGB")
                auxiliary_draw = ImageDraw.Draw(auxiliary_annotated)
                aux_x, aux_y = auxiliary_estimate.source_pixel_xy
                auxiliary_draw.ellipse(
                    (aux_x - radius, aux_y - radius, aux_x + radius, aux_y + radius),
                    fill="cyan",
                    outline="white",
                    width=2,
                )
                auxiliary_annotated.save(output_dir / auxiliary_annotated_name)
                views.append(
                    {
                        "label": auxiliary.label,
                        "role": "auxiliary_depth",
                        "serial": auxiliary.serial,
                        "image": auxiliary_image_path.name,
                        "annotated_image": auxiliary_annotated_name,
                        "pixel_xy": [aux_x, aux_y],
                        "depth_m": auxiliary_estimate.depth_m,
                        "intrinsics": auxiliary.intrinsics.tolist(),
                        "X_base_camera": auxiliary.X_base_camera.tolist(),
                        "center_base_mm": auxiliary_estimate.point_base_mm.tolist(),
                        "candidate_count": auxiliary_estimate.candidate_count,
                        "clustered_count": auxiliary_estimate.clustered_count,
                        "projection_error_px_median": auxiliary_estimate.projection_error_px_median,
                        "surface_spread_mm": auxiliary_estimate.surface_spread_mm,
                    }
                )
                depth_fusion = {
                    "mode": "primary_semantic_ray_with_auxiliary_depth",
                    "selected_depth_camera": auxiliary.label,
                    "primary_depth_camera": primary.label,
                    "primary_depth_point_base_mm": primary_point.tolist(),
                    "auxiliary_depth_point_base_mm": auxiliary_estimate.point_base_mm.tolist(),
                    "source_disagreement_mm": disagreement,
                    "projection_radius_px": projection_radius_px,
                    "candidate_count": auxiliary_estimate.candidate_count,
                    "clustered_count": auxiliary_estimate.clustered_count,
                    "projection_error_px_median": auxiliary_estimate.projection_error_px_median,
                    "surface_spread_mm": auxiliary_estimate.surface_spread_mm,
                }
        else:
            center = primary_point.copy()
            perception_mode = "single_camera_rgbd"
            status = "VALIDATED_SINGLE_VIEW"
            depth_fusion = {
                "mode": "primary_camera_depth_only",
                "selected_depth_camera": primary.label,
                "primary_depth_point_base_mm": primary_point.tolist(),
                "source_disagreement_mm": None,
            }
        automatic_plan, motion_derivation = derive_grasp_plan(
            center, self.robot_config, self.config
        )
        updated = replace(experiment, **automatic_plan.as_dict())
        updated.require_ready()
        result = {
            "created_at": _now(),
            "status": status,
            "perception_mode": perception_mode,
            "active_cameras": list(self.config.active_camera_labels),
            "primary_camera": primary.label,
            "auxiliary_depth_cameras": auxiliary_labels,
            "prompt": prompt,
            "molmo": molmo_result,
            "views": views,
            "view_disagreement_mm": disagreement,
            "warnings": perception_warnings,
            "depth_fusion": depth_fusion,
            "center_base_mm": center.tolist(),
            "motion_derivation": motion_derivation,
            "experiment_config": updated.as_dict(),
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result, updated
