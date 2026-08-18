"""Dense two-camera RGB-D perception in the robot base frame.

Both calibrated cameras contribute depth points to one voxelized base-frame
cloud.  A robust table plane is estimated from the fused cloud, garment points
are selected from height-above-table evidence, and an action-relevant garment
center/surface observation is reported.  No semantic model or Molmo process is
needed in the execution path.
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
    color_exposure: float | None = None
    color_white_balance: float | None = None


@dataclass(frozen=True)
class MolmoConfig:
    """Deprecated compatibility record; the perception pipeline does not use it."""

    python: Path
    model: str = "allenai/MolmoPoint-8B"
    dtype: str = "bf16"
    max_crops: int = 1
    max_new_tokens: int = 96
    timeout_s: int = 600
    local_files_only: bool = True


@dataclass(frozen=True)
class PerceptionConfig:
    """Calibration/capture settings for dense A+B RGB-D observation.

    The three clearance fields are retained only to read older configuration
    files.  They are recorded as deprecated metadata and are never converted
    into grasp, lift, transfer, or release waypoints.
    """

    cameras: tuple[CameraSpec, CameraSpec]
    molmo: MolmoConfig | None = None
    active_camera_labels: tuple[str, ...] = ("A", "B")
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 10
    temporal_median_frames: int = 25
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
            color_exposure_raw = item.get("color_exposure")
            color_white_balance_raw = item.get("color_white_balance")
            cameras.append(
                CameraSpec(
                    label,
                    serial,
                    extrinsics.resolve(),
                    float(color_exposure_raw)
                    if color_exposure_raw is not None
                    else None,
                    float(color_white_balance_raw)
                    if color_white_balance_raw is not None
                    else None,
                )
            )

        active_camera_labels = tuple(
            str(label).strip().upper()
            for label in raw.get("active_cameras", [camera.label for camera in cameras])
        )
        config = cls(
            cameras=(cameras[0], cameras[1]),
            molmo=None,
            active_camera_labels=active_camera_labels,
            width=int(raw.get("width", 640)),
            height=int(raw.get("height", 480)),
            fps=int(raw.get("fps", 30)),
            warmup_frames=int(raw.get("warmup_frames", 10)),
            temporal_median_frames=int(raw.get("temporal_median_frames", 25)),
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
        if len(self.active_camera_labels) != 2:
            raise PerceptionError(
                "dense AB RGB-D fusion requires both configured cameras"
            )
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
        if self.temporal_median_frames < 1 or self.temporal_median_frames > 60:
            raise PerceptionError("temporal_median_frames must be between 1 and 60")
        if not 0 < self.min_depth_m < self.max_depth_m:
            raise PerceptionError("expected 0 < min_depth_m < max_depth_m")
        if self.max_view_disagreement_mm <= 0:
            raise PerceptionError("max_view_disagreement_mm must be positive")
        for camera in self.cameras:
            if camera.color_exposure is not None and camera.color_exposure <= 0:
                raise PerceptionError(
                    f"camera {camera.label} color_exposure must be positive"
                )
        if self.grasp_contact_clearance_mm < 0:
            raise PerceptionError("grasp_contact_clearance_mm must be non-negative")
        if self.approach_clearance_mm <= 0:
            raise PerceptionError("approach_clearance_mm must be positive")
        if self.lift_clearance_mm < self.approach_clearance_mm:
            raise PerceptionError(
                "lift_clearance_mm must be greater than or equal to approach_clearance_mm"
            )


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
        device = profile.get_device()
        self.depth_scale = float(device.first_depth_sensor().get_depth_scale())
        if (
            self.spec.color_exposure is not None
            or self.spec.color_white_balance is not None
        ):
            color_sensor = next(
                (
                    sensor
                    for sensor in device.query_sensors()
                    if sensor.get_info(rs.camera_info.name) == "RGB Camera"
                ),
                None,
            )
            if color_sensor is None:
                raise PerceptionError(f"camera {self.spec.label} has no RGB sensor")
            if self.spec.color_exposure is not None:
                if not color_sensor.supports(rs.option.enable_auto_exposure) or not color_sensor.supports(rs.option.exposure):
                    raise PerceptionError(
                        f"camera {self.spec.label} has no configurable RGB exposure sensor"
                    )
                exposure_range = color_sensor.get_option_range(rs.option.exposure)
                exposure = float(self.spec.color_exposure)
                if exposure < exposure_range.min or exposure > exposure_range.max:
                    raise PerceptionError(
                        f"camera {self.spec.label} color exposure {exposure} is outside "
                        f"[{exposure_range.min}, {exposure_range.max}]"
                    )
                color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
                color_sensor.set_option(rs.option.exposure, exposure)
            if self.spec.color_white_balance is not None:
                if not color_sensor.supports(rs.option.enable_auto_white_balance) or not color_sensor.supports(rs.option.white_balance):
                    raise PerceptionError(
                        f"camera {self.spec.label} has no configurable RGB white-balance sensor"
                    )
                white_balance_range = color_sensor.get_option_range(rs.option.white_balance)
                white_balance = float(self.spec.color_white_balance)
                if white_balance < white_balance_range.min or white_balance > white_balance_range.max:
                    raise PerceptionError(
                        f"camera {self.spec.label} color white balance {white_balance} is outside "
                        f"[{white_balance_range.min}, {white_balance_range.max}]"
                    )
                color_sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
                color_sensor.set_option(rs.option.white_balance, white_balance)
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
    numpy = _require_numpy()
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
        temporal_rgb: dict[str, list[np.ndarray]] = {camera.spec.label: [] for camera in cameras}
        temporal_depth: dict[str, list[np.ndarray]] = {camera.spec.label: [] for camera in cameras}
        for _ in range(config.temporal_median_frames):
            for camera in cameras:
                rgb, depth = camera.read()
                temporal_rgb[camera.spec.label].append(rgb)
                temporal_depth[camera.spec.label].append(depth)
        frames: list[RGBDFrame] = []
        for camera in cameras:
            if camera.intrinsics is None:
                raise PerceptionError(f"camera {camera.spec.label} intrinsics are unavailable")
            rgb_stack = numpy.stack(temporal_rgb[camera.spec.label], axis=0).astype(numpy.float32)
            depth_stack = numpy.stack(temporal_depth[camera.spec.label], axis=0).astype(numpy.float32)
            invalid = ~numpy.isfinite(depth_stack) | (depth_stack <= 0.0)
            depth_stack[invalid] = numpy.nan
            depth_m = numpy.nanmedian(depth_stack, axis=0).astype(numpy.float32)
            rgb = numpy.rint(numpy.median(rgb_stack, axis=0)).clip(0, 255).astype(numpy.uint8)
            frames.append(
                RGBDFrame(
                    label=camera.spec.label,
                    serial=camera.spec.serial,
                    rgb=rgb,
                    depth_m=depth_m,
                    intrinsics=camera.intrinsics.copy(),
                    X_base_camera=load_extrinsics(camera.spec.extrinsics_file),
                )
            )
        return frames
    finally:
        for camera in reversed(cameras):
            camera.stop()


def _frame_points_base_mm(
    frame: RGBDFrame,
    config: PerceptionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one aligned RGB-D frame into the calibrated base frame."""

    numpy = _require_numpy()
    depth = numpy.asarray(frame.depth_m, dtype=numpy.float64)
    valid = numpy.isfinite(depth) & (depth > config.min_depth_m) & (depth < config.max_depth_m)
    y_px, x_px = numpy.nonzero(valid)
    z_m = depth[valid]
    if z_m.size == 0:
        raise PerceptionError(f"camera {frame.label} has no valid depth points")
    K = numpy.asarray(frame.intrinsics, dtype=numpy.float64)
    camera_points = numpy.stack(
        [
            (x_px - K[0, 2]) * z_m / K[0, 0],
            (y_px - K[1, 2]) * z_m / K[1, 1],
            z_m,
        ],
        axis=1,
    )
    X = numpy.asarray(frame.X_base_camera, dtype=numpy.float64)
    base_points_mm = (
        camera_points @ X[:3, :3].T + X[:3, 3]
    ) * 1000.0
    colors = numpy.asarray(frame.rgb[y_px, x_px], dtype=numpy.uint8)
    finite = numpy.all(numpy.isfinite(base_points_mm), axis=1)
    return base_points_mm[finite], colors[finite]


def _voxel_fuse_base_points(
    frames: list[RGBDFrame],
    config: PerceptionConfig,
    robot_config: RobotConfig,
    *,
    voxel_size_mm: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fuse A/B points by median-free deterministic voxel aggregation.

    The returned source mask uses bit 1 for camera A and bit 2 for camera B,
    so mask value 3 identifies voxels observed by both cameras.
    """

    numpy = _require_numpy()
    if voxel_size_mm <= 0:
        raise PerceptionError("voxel_size_mm must be positive")
    bounds = robot_config.boundaries
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    source_chunks: list[np.ndarray] = []
    input_counts: dict[str, int] = {}
    for frame_index, frame in enumerate(frames):
        points, colors = _frame_points_base_mm(frame, config)
        # Keep only points plausibly belonging to the calibrated work envelope.
        mask = numpy.ones(len(points), dtype=bool)
        if bounds.x_min is not None:
            mask &= points[:, 0] >= bounds.x_min - 80.0
        if bounds.x_max is not None:
            mask &= points[:, 0] <= bounds.x_max + 80.0
        if bounds.y_min is not None:
            mask &= points[:, 1] >= bounds.y_min - 20.0
        if bounds.y_max is not None:
            mask &= points[:, 1] <= bounds.y_max + 20.0
        if bounds.z_min is not None:
            mask &= points[:, 2] >= bounds.z_min - 120.0
        if bounds.z_max is not None:
            mask &= points[:, 2] <= bounds.z_max + 80.0
        points = points[mask]
        colors = colors[mask]
        if len(points) == 0:
            raise PerceptionError(f"camera {frame.label} has no points in the robot work envelope")
        point_chunks.append(points)
        color_chunks.append(colors)
        source_chunks.append(numpy.full(len(points), 1 << frame_index, dtype=numpy.uint8))
        input_counts[frame.label] = int(len(points))

    points = numpy.concatenate(point_chunks, axis=0)
    colors = numpy.concatenate(color_chunks, axis=0)
    sources = numpy.concatenate(source_chunks, axis=0)
    voxel_keys = numpy.floor(points / float(voxel_size_mm)).astype(numpy.int64)
    _, inverse = numpy.unique(voxel_keys, axis=0, return_inverse=True)
    order = numpy.argsort(inverse, kind="stable")
    grouped = inverse[order]
    starts = numpy.r_[0, numpy.flatnonzero(numpy.diff(grouped)) + 1]
    counts = numpy.diff(numpy.r_[starts, len(grouped)]).astype(numpy.float64)
    fused_points = numpy.add.reduceat(points[order], starts, axis=0) / counts[:, None]
    fused_colors = numpy.rint(
        numpy.add.reduceat(colors[order].astype(numpy.float64), starts, axis=0)
        / counts[:, None]
    ).clip(0, 255).astype(numpy.uint8)
    source_sorted = sources[order]
    source_mask = numpy.bitwise_or.reduceat(source_sorted, starts).astype(numpy.uint8)
    source_counts = {
        "A": int(numpy.count_nonzero(source_mask & 1)),
        "B": int(numpy.count_nonzero(source_mask & 2)),
        "AB_overlap": int(numpy.count_nonzero(source_mask == 3)),
    }
    diagnostics = {
        "voxel_size_mm": float(voxel_size_mm),
        "input_point_count": int(len(points)),
        "input_point_counts": input_counts,
        "fused_point_count": int(len(fused_points)),
        "source_voxel_counts": source_counts,
    }
    return fused_points, fused_colors, source_mask, diagnostics


def _fit_table_plane(
    points_mm: np.ndarray,
    colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit z=a*x+b*y+c from low, bright fused points with robust refitting."""

    numpy = _require_numpy()
    z = points_mm[:, 2]
    luma = (
        0.2126 * colors[:, 0].astype(numpy.float64)
        + 0.7152 * colors[:, 1].astype(numpy.float64)
        + 0.0722 * colors[:, 2].astype(numpy.float64)
    )
    low_cut = float(numpy.percentile(z, 45.0))
    bright_cut = float(numpy.percentile(luma, 45.0))
    candidates = (z <= low_cut) & (luma >= bright_cut)
    if int(candidates.sum()) < 40:
        candidates = z <= float(numpy.percentile(z, 40.0))
    if int(candidates.sum()) < 10:
        raise PerceptionError("fused RGB-D cloud does not contain enough table-plane points")

    design = numpy.column_stack((points_mm[:, 0], points_mm[:, 1], numpy.ones(len(points_mm))))
    fit_mask = candidates.copy()
    coefficients = numpy.linalg.lstsq(design[fit_mask], z[fit_mask], rcond=None)[0]
    for _ in range(5):
        residual = z - design @ coefficients
        abs_residual = numpy.abs(residual[fit_mask])
        robust_scale = float(numpy.median(abs_residual)) * 1.4826
        threshold = max(3.0, min(12.0, 3.0 * robust_scale))
        fit_mask = candidates & (numpy.abs(residual) <= threshold)
        if int(fit_mask.sum()) < 10:
            break
        coefficients = numpy.linalg.lstsq(design[fit_mask], z[fit_mask], rcond=None)[0]
    residual = z - design @ coefficients
    inliers = fit_mask
    table_rgb_median = (
        numpy.median(colors[inliers].astype(numpy.float64), axis=0)
        if inliers.any()
        else numpy.asarray([255.0, 255.0, 255.0], dtype=numpy.float64)
    )
    table_color_distance = numpy.linalg.norm(
        colors.astype(numpy.float64) - table_rgb_median[None, :],
        axis=1,
    )
    diagnostics = {
        "model": "base_z_mm = a*base_x_mm + b*base_y_mm + c",
        "coefficients": {
            "a": float(coefficients[0]),
            "b": float(coefficients[1]),
            "c_mm": float(coefficients[2]),
        },
        "candidate_count": int(candidates.sum()),
        "inlier_count": int(inliers.sum()),
        "residual_median_mm": float(numpy.median(residual[inliers])) if inliers.any() else None,
        "residual_p95_abs_mm": float(numpy.percentile(numpy.abs(residual[inliers]), 95))
        if inliers.any()
        else None,
        "table_luma_median": float(numpy.median(luma[inliers])) if inliers.any() else None,
        "table_rgb_median": [float(value) for value in table_rgb_median],
        # The geometric table inliers can include flat garment pixels.  The
        # median distance is intentionally used instead of p95 so those dark
        # cloth pixels do not inflate the table-color noise estimate.
        "table_color_distance_p50": float(numpy.percentile(table_color_distance[inliers], 50))
        if inliers.any()
        else None,
    }
    return coefficients, residual, diagnostics


def _sample_table_reference_points(
    frame: RGBDFrame,
    config: PerceptionConfig,
    *,
    patch_radius_px: int = 12,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Sample robust depth references around image corners and table edges.

    The RGB-D view can contain the arm or fixture at a corner.  Each reference
    is therefore a small patch, biased toward bright table pixels, rather than
    one brittle corner pixel.  Outlying patches are removed later by the
    cross-camera plane RANSAC.
    """

    numpy = _require_numpy()
    depth = numpy.asarray(frame.depth_m, dtype=numpy.float64)
    rgb = numpy.asarray(frame.rgb, dtype=numpy.uint8)
    height, width = depth.shape[:2]
    if rgb.shape[:2] != depth.shape:
        raise PerceptionError(f"camera {frame.label} RGB/depth shapes do not match")
    if patch_radius_px < 1:
        raise PerceptionError("table reference patch radius must be positive")
    valid_depth = (
        numpy.isfinite(depth)
        & (depth > config.min_depth_m)
        & (depth < config.max_depth_m)
    )
    luma = (
        0.2126 * rgb[..., 0].astype(numpy.float64)
        + 0.7152 * rgb[..., 1].astype(numpy.float64)
        + 0.0722 * rgb[..., 2].astype(numpy.float64)
    )
    # Four corners plus the middle of each edge.  The latter helps when one
    # corner is occupied by the robot or falls outside the table.
    locations = (
        ("top_left", 0.08, 0.08),
        ("top_edge", 0.08, 0.50),
        ("top_right", 0.08, 0.92),
        ("left_edge", 0.50, 0.08),
        ("right_edge", 0.50, 0.92),
        ("bottom_left", 0.92, 0.08),
        ("bottom_edge", 0.92, 0.50),
        ("bottom_right", 0.92, 0.92),
    )
    K = numpy.asarray(frame.intrinsics, dtype=numpy.float64)
    X = numpy.asarray(frame.X_base_camera, dtype=numpy.float64)
    points: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for name, fy, fx in locations:
        cy = int(round(fy * (height - 1)))
        cx = int(round(fx * (width - 1)))
        y0, y1 = max(0, cy - patch_radius_px), min(height, cy + patch_radius_px + 1)
        x0, x1 = max(0, cx - patch_radius_px), min(width, cx + patch_radius_px + 1)
        local_valid = valid_depth[y0:y1, x0:x1]
        if not local_valid.any():
            records.append({"name": name, "pixel_xy": [cx, cy], "valid": False})
            continue
        local_luma = luma[y0:y1, x0:x1]
        bright_cut = float(numpy.percentile(local_luma[local_valid], 60.0))
        selected = local_valid & (local_luma >= bright_cut)
        if int(selected.sum()) < 8:
            selected = local_valid
        local_y, local_x = numpy.nonzero(selected)
        y_px = local_y + y0
        x_px = local_x + x0
        z_m = depth[y_px, x_px]
        camera_points = numpy.stack(
            [
                (x_px - K[0, 2]) * z_m / K[0, 0],
                (y_px - K[1, 2]) * z_m / K[1, 1],
                z_m,
            ],
            axis=1,
        )
        base_points_mm = (camera_points @ X[:3, :3].T + X[:3, 3]) * 1000.0
        finite = numpy.all(numpy.isfinite(base_points_mm), axis=1)
        if not finite.any():
            records.append({"name": name, "pixel_xy": [cx, cy], "valid": False})
            continue
        patch_point = numpy.median(base_points_mm[finite], axis=0)
        point_index = int(numpy.flatnonzero(finite)[len(numpy.flatnonzero(finite)) // 2])
        points.append(patch_point)
        records.append(
            {
                "name": name,
                "pixel_xy": [cx, cy],
                "valid": True,
                "sample_count": int(finite.sum()),
                "depth_median_m": float(numpy.median(z_m[finite])),
                "luma_cut": bright_cut,
                "base_xyz_mm": [float(value) for value in patch_point],
                "selected_pixel_xy": [int(x_px[point_index]), int(y_px[point_index])],
            }
        )
    if not points:
        return numpy.empty((0, 3), dtype=numpy.float64), records
    return numpy.asarray(points, dtype=numpy.float64), records


def _fit_table_plane_from_references(
    frames: list[RGBDFrame],
    config: PerceptionConfig,
    fallback_coefficients: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the table from sampled edge/corner depths with deterministic RANSAC."""

    numpy = _require_numpy()
    import itertools

    all_points: list[np.ndarray] = []
    camera_records: dict[str, list[dict[str, Any]]] = {}
    point_records: list[dict[str, Any]] = []
    for frame in frames:
        points, records = _sample_table_reference_points(frame, config)
        camera_records[frame.label] = records
        if len(points):
            all_points.append(points)
            point_records.extend(record for record in records if record.get("valid"))
    if not all_points:
        return numpy.asarray(fallback_coefficients, dtype=numpy.float64), {
            "mode": "fused_cloud_fallback",
            "reference_count": 0,
            "inlier_count": 0,
            "cameras": camera_records,
        }
    points = numpy.concatenate(all_points, axis=0)
    if len(points) < 3:
        return numpy.asarray(fallback_coefficients, dtype=numpy.float64), {
            "mode": "fused_cloud_fallback",
            "reference_count": int(len(points)),
            "inlier_count": 0,
            "cameras": camera_records,
        }
    design = numpy.column_stack((points[:, 0], points[:, 1], numpy.ones(len(points))))
    best_score: tuple[int, float] | None = None
    best_coefficients: np.ndarray | None = None
    best_inliers: np.ndarray | None = None
    for indices in itertools.combinations(range(len(points)), 3):
        sample_design = design[list(indices)]
        if numpy.linalg.matrix_rank(sample_design) < 3:
            continue
        coefficients = numpy.linalg.lstsq(
            sample_design, points[list(indices), 2], rcond=None
        )[0]
        residual = points[:, 2] - design @ coefficients
        inliers = numpy.abs(residual) <= 25.0
        score = (int(inliers.sum()), -float(numpy.abs(residual[inliers]).sum()))
        if best_score is None or score > best_score:
            best_score = score
            best_coefficients = coefficients
            best_inliers = inliers
    if best_coefficients is None or best_inliers is None or int(best_inliers.sum()) < 3:
        return numpy.asarray(fallback_coefficients, dtype=numpy.float64), {
            "mode": "fused_cloud_fallback",
            "reference_count": int(len(points)),
            "inlier_count": 0,
            "cameras": camera_records,
        }
    coefficients = numpy.linalg.lstsq(
        design[best_inliers], points[best_inliers, 2], rcond=None
    )[0]
    residual = points[:, 2] - design @ coefficients
    # A second tighter pass removes edge patches that are close to, but not
    # actually on, the table plane.
    refined = numpy.abs(residual) <= 15.0
    if int(refined.sum()) >= 3:
        coefficients = numpy.linalg.lstsq(
            design[refined], points[refined, 2], rcond=None
        )[0]
        residual = points[:, 2] - design @ coefficients
        best_inliers = refined
    for index, record in enumerate(point_records):
        record["plane_inlier"] = bool(best_inliers[index])
        record["plane_residual_mm"] = float(residual[index])
    return coefficients, {
        "mode": "corner_edge_depth_interpolation",
        "model": "base_z_mm = a*base_x_mm + b*base_y_mm + c",
        "coefficients": {
            "a": float(coefficients[0]),
            "b": float(coefficients[1]),
            "c_mm": float(coefficients[2]),
        },
        "reference_count": int(len(points)),
        "inlier_count": int(best_inliers.sum()),
        "residual_p95_abs_mm": float(
            numpy.percentile(numpy.abs(residual[best_inliers]), 95)
        ),
        "cameras": camera_records,
    }


def _largest_xy_component(
    points_mm: np.ndarray,
    candidate: np.ndarray,
    *,
    grid_size_mm: float = 8.0,
) -> np.ndarray:
    """Keep the largest connected XY component of garment candidates."""

    numpy = _require_numpy()
    indices = numpy.flatnonzero(candidate)
    if len(indices) < 20:
        return candidate
    xy = points_mm[indices, :2]
    origin = numpy.floor(xy.min(axis=0) / grid_size_mm).astype(numpy.int64)
    cells = numpy.floor(xy / grid_size_mm).astype(numpy.int64) - origin
    width = int(cells[:, 0].max()) + 1
    height = int(cells[:, 1].max()) + 1
    if width <= 0 or height <= 0 or width * height > 4_000_000:
        return candidate
    occupied = numpy.zeros((height, width), dtype=bool)
    occupied[cells[:, 1], cells[:, 0]] = True
    try:
        from scipy.ndimage import label

        labels, count = label(occupied, structure=numpy.ones((3, 3), dtype=numpy.uint8))
    except ImportError:
        return candidate
    if count <= 1:
        return candidate
    cell_labels = labels[cells[:, 1], cells[:, 0]]
    values, counts = numpy.unique(cell_labels[cell_labels > 0], return_counts=True)
    if len(values) == 0:
        return candidate
    largest_label = values[int(numpy.argmax(counts))]
    selected = cell_labels == largest_label
    result = numpy.zeros_like(candidate)
    result[indices[selected]] = True
    return result


def _height_display_max_mm(values_mm: np.ndarray) -> float:
    """Choose one stable physical upper bound for a capture's heatmaps."""

    numpy = _require_numpy()
    values = numpy.asarray(values_mm, dtype=numpy.float64)
    values = values[numpy.isfinite(values)]
    if values.size == 0:
        return 40.0
    return float(
        min(
            160.0,
            max(40.0, numpy.ceil(numpy.percentile(values, 98) / 10.0) * 10.0),
        )
    )


def _save_fused_height_map(
    output_dir: Path,
    points_mm: np.ndarray,
    heights_mm: np.ndarray,
    garment_mask: np.ndarray | None = None,
    *,
    grid_size_mm: float = 4.0,
    display_max_mm: float | None = None,
) -> dict[str, Any]:
    """Save fused scalar maps and a garment-boundary overlay.

    ``heights_mm`` is the surface height above the fitted table plane at each
    fused point.  Both the scalar preview and the color heatmap use that same
    physical quantity; no alternate reference-surface transform is applied
    here.
    """

    numpy = _require_numpy()
    x0, y0 = numpy.floor(points_mm[:, :2].min(axis=0) / grid_size_mm) * grid_size_mm
    x1, y1 = numpy.ceil(points_mm[:, :2].max(axis=0) / grid_size_mm) * grid_size_mm
    width = max(1, int(round((x1 - x0) / grid_size_mm)) + 1)
    height = max(1, int(round((y1 - y0) / grid_size_mm)) + 1)
    if width * height > 8_000_000:
        raise PerceptionError("fused height map would be unreasonably large")
    ix = numpy.clip(numpy.rint((points_mm[:, 0] - x0) / grid_size_mm).astype(numpy.int64), 0, width - 1)
    iy = numpy.clip(numpy.rint((points_mm[:, 1] - y0) / grid_size_mm).astype(numpy.int64), 0, height - 1)
    flat = iy * width + ix
    height_map = numpy.full(width * height, -numpy.inf, dtype=numpy.float32)
    finite = numpy.isfinite(heights_mm)
    numpy.maximum.at(height_map, flat[finite], heights_mm[finite].astype(numpy.float32))
    height_map = height_map.reshape(height, width)
    height_map[~numpy.isfinite(height_map)] = numpy.nan
    numpy.save(output_dir / "fused_height_map_mm.npy", height_map)
    finite_map = numpy.isfinite(height_map)
    garment_grid: np.ndarray | None = None
    if garment_mask is not None:
        garment_mask = numpy.asarray(garment_mask, dtype=bool)
        if garment_mask.shape != points_mm.shape[:1]:
            raise PerceptionError("garment boundary mask must match fused point count")
        garment_grid = numpy.zeros(height_map.shape, dtype=bool)
        garment_grid.reshape(-1)[flat[garment_mask]] = True
        garment_grid = _solidify_largest_mask(
            garment_grid,
            dilation_iterations=1,
            closing_iterations=2,
        )
    heatmap_values = height_map
    heatmap_quantity = "height_above_table_mm"
    if display_max_mm is None:
        finite_garment = finite_map & garment_grid if garment_grid is not None else finite_map
        display_max_mm = _height_display_max_mm(height_map[finite_garment])
    display_max_mm = float(max(1.0, display_max_mm))
    preview = numpy.zeros(height_map.shape, dtype=numpy.uint8)
    if finite_map.any():
        preview[finite_map] = numpy.rint(
            numpy.clip(height_map[finite_map] / display_max_mm, 0.0, 1.0) * 255.0
        ).astype(numpy.uint8)
    try:
        from PIL import Image

        Image.fromarray(preview, mode="L").save(output_dir / "fused_height_map_preview.png")
        heatmap = _scalar_heatmap_rgb(
            heatmap_values,
            finite_map,
            focus_mask=garment_grid,
            higher_is_bright=True,
            value_range_mm=(0.0, display_max_mm),
        )
        Image.fromarray(heatmap, mode="RGB").save(output_dir / "fused_height_map_heatmap.png")
        boundary_path: str | None = None
        fold_edge_path: str | None = None
        if garment_grid is not None:
            boundary = _mask_boundary(garment_grid)
            overlay = heatmap.copy()
            overlay[boundary] = numpy.asarray([255, 255, 255], dtype=numpy.uint8)
            boundary_path = "fused_height_map_boundary.png"
            Image.fromarray(overlay, mode="RGB").save(output_dir / boundary_path)
            fold_edges, _ = _fold_edge_mask(height_map, garment_grid)
            fold_overlay = overlay.copy()
            fold_overlay[fold_edges] = numpy.asarray([0, 255, 255], dtype=numpy.uint8)
            fold_edge_path = "fused_fold_edges.png"
            Image.fromarray(fold_overlay, mode="RGB").save(output_dir / fold_edge_path)
    except ImportError:
        boundary_path = None
    return {
        "path": "fused_height_map_mm.npy",
        "preview": "fused_height_map_preview.png",
        "heatmap": "fused_height_map_heatmap.png",
        "boundary_overlay": boundary_path,
        "height_gradient_overlay": fold_edge_path,
        # Compatibility alias retained for existing consumers.
        "fold_edge_overlay": fold_edge_path,
        "heatmap_quantity": heatmap_quantity,
        "heatmap_display_min_mm": 0.0,
        "heatmap_display_max_mm": display_max_mm,
        "heatmap_normalization": "absolute_table_zero_shared",
        "height_min_mm": float(numpy.percentile(heatmap_values[finite_map], 2))
        if finite_map.any()
        else None,
        "height_max_mm": float(numpy.percentile(heatmap_values[finite_map], 98))
        if finite_map.any()
        else None,
        "value_min_mm": float(numpy.percentile(heatmap_values[finite_map], 2))
        if finite_map.any()
        else None,
        "value_max_mm": float(numpy.percentile(heatmap_values[finite_map], 98))
        if finite_map.any()
        else None,
        "grid_size_mm": float(grid_size_mm),
        "origin_xy_mm": [float(x0), float(y0)],
        "shape_yx": [int(height), int(width)],
    }


def _scalar_heatmap_rgb(
    values: np.ndarray,
    valid: np.ndarray | None = None,
    focus_mask: np.ndarray | None = None,
    *,
    higher_is_bright: bool = False,
    value_range_mm: tuple[float, float] | None = None,
) -> np.ndarray:
    """Colorize a scalar map, optionally normalizing only inside a focus mask.

    When ``focus_mask`` is supplied, percentile statistics and the softmax
    palette are computed from that region only.  Pixels outside the region are
    rendered as a dark background so table/robot values cannot flatten the
    focused garment signal or erase its silhouette.  By default lower values
    are brighter (the historical depth-preview behavior); height maps pass
    ``higher_is_bright=True`` so larger garment/table height differences are
    visually hotter.
    """

    numpy = _require_numpy()
    values = numpy.asarray(values, dtype=numpy.float64)
    finite = numpy.isfinite(values)
    if valid is not None:
        finite &= numpy.asarray(valid, dtype=bool)
    render_mask = finite
    focused_region = False
    if focus_mask is not None:
        focus_mask = numpy.asarray(focus_mask, dtype=bool)
        if focus_mask.shape != values.shape:
            raise PerceptionError("focus mask must match scalar map shape")
        focused_finite = finite & focus_mask
        if focused_finite.any():
            finite_for_color = focused_finite
            render_mask = focused_finite
            focused_region = True
        else:
            finite_for_color = finite
    else:
        finite_for_color = finite
    rgb = numpy.zeros((*values.shape, 3), dtype=numpy.uint8)
    if not finite_for_color.any():
        return rgb
    if value_range_mm is not None:
        low, high = (float(value_range_mm[0]), float(value_range_mm[1]))
        if not numpy.isfinite(low) or not numpy.isfinite(high) or high <= low:
            raise PerceptionError("heatmap value range must contain finite low < high values")
    else:
        low = float(numpy.percentile(values[finite_for_color], 2))
        high = float(numpy.percentile(values[finite_for_color], 98))
    scale = max(1e-9, high - low)
    # The palette runs from bright yellow through red/purple to a dark tone.
    # The normalized scalar is oriented below so callers can choose whether
    # low or high physical values should be bright.
    stops = numpy.asarray(
        [
            [255.0, 245.0, 180.0],
            [255.0, 150.0, 50.0],
            [210.0, 45.0, 45.0],
            [80.0, 25.0, 120.0],
            [30.0, 10.0, 60.0],
        ],
        dtype=numpy.float64,
    )
    normalized = numpy.clip((values - low) / scale, 0.0, 1.0)
    if focused_region and value_range_mm is None:
        # A linear range leaves concentrated garment values in one color. Mix
        # it with an empirical CDF computed only from the focused region so the
        # available palette is spread over the observed height distribution.
        focused_values = values[finite_for_color].astype(numpy.float64, copy=False)
        sorted_values = numpy.sort(focused_values)
        ranks = numpy.searchsorted(sorted_values, focused_values, side="left")
        rank_normalized = ranks.astype(numpy.float64) / max(1, len(sorted_values) - 1)
        normalized_focus = normalized[render_mask].astype(numpy.float64, copy=False)
        normalized = normalized.copy()
        normalized[render_mask] = 0.35 * normalized_focus + 0.65 * rank_normalized
    if higher_is_bright:
        normalized = 1.0 - normalized
    centers = numpy.linspace(0.0, 1.0, len(stops), dtype=numpy.float64)
    finite_values = normalized[render_mask].astype(numpy.float32, copy=False)
    color_values = numpy.empty((len(finite_values), 3), dtype=numpy.float32)
    # Chunking avoids allocating a full HxWx5 softmax tensor for a large fused
    # map while retaining the same color result.
    temperature = 0.12 if focused_region else 0.055
    chunk_size = 250_000
    for start in range(0, len(finite_values), chunk_size):
        stop = min(len(finite_values), start + chunk_size)
        chunk = finite_values[start:stop].astype(numpy.float64, copy=False)
        logits = -((chunk[:, None] - centers[None, :]) ** 2) / (2.0 * temperature**2)
        logits -= logits.max(axis=1, keepdims=True)
        weights = numpy.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        color_values[start:stop] = weights @ stops
    rgb[render_mask] = numpy.rint(color_values).astype(numpy.uint8)
    return rgb


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Return the one-pixel outer boundary of a boolean raster mask."""

    numpy = _require_numpy()
    mask = numpy.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise PerceptionError("boundary mask must be a 2-D raster")
    padded = numpy.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~interior


def _outer_mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Return only the outer silhouette, without outlining internal holes."""

    numpy = _require_numpy()
    mask = numpy.asarray(mask, dtype=bool)
    try:
        from scipy.ndimage import binary_fill_holes

        outline_mask = binary_fill_holes(mask)
    except ImportError:
        outline_mask = mask
    return _mask_boundary(outline_mask)


def _solidify_largest_mask(
    mask: np.ndarray,
    *,
    dilation_iterations: int,
    closing_iterations: int,
    fill_holes: bool = True,
) -> np.ndarray:
    """Close small depth holes and keep one garment component for outlining."""

    numpy = _require_numpy()
    mask = numpy.asarray(mask, dtype=bool)
    try:
        from scipy.ndimage import (
            binary_closing,
            binary_dilation,
            binary_fill_holes,
            label,
        )

        cleaned = mask.copy()
        dilation_iterations = max(0, int(dilation_iterations))
        closing_iterations = max(0, int(closing_iterations))
        # scipy treats iterations=0 as "repeat until stable", not as a no-op.
        # Skip the operator explicitly when the caller requests zero passes.
        if dilation_iterations:
            cleaned = binary_dilation(
                cleaned,
                structure=numpy.ones((3, 3), dtype=bool),
                iterations=dilation_iterations,
            )
        if closing_iterations:
            cleaned = binary_closing(
                cleaned,
                structure=numpy.ones((3, 3), dtype=bool),
                iterations=closing_iterations,
            )
        if fill_holes:
            cleaned = binary_fill_holes(cleaned)
        labels, count = label(cleaned, structure=numpy.ones((3, 3), dtype=numpy.uint8))
        if count > 1:
            sizes = numpy.bincount(labels.reshape(-1))
            sizes[0] = 0
            cleaned = labels == int(numpy.argmax(sizes))
        return numpy.asarray(cleaned, dtype=bool)
    except ImportError:
        return mask


def _fold_edge_mask(
    surface_field: np.ndarray,
    focus_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect internal fold/occlusion edges from local surface changes."""

    numpy = _require_numpy()
    field = numpy.asarray(surface_field, dtype=numpy.float64)
    mask = numpy.asarray(focus_mask, dtype=bool)
    finite = numpy.isfinite(field) & mask
    if not finite.any():
        return numpy.zeros(mask.shape, dtype=bool), numpy.zeros(field.shape, dtype=numpy.float32)
    fill_value = float(numpy.median(field[finite]))
    filled = numpy.where(numpy.isfinite(field), field, fill_value)
    try:
        from scipy.ndimage import gaussian_filter

        weights = gaussian_filter(mask.astype(numpy.float64), sigma=1.25)
        weighted_surface = gaussian_filter(filled * mask, sigma=1.25)
        filled = numpy.where(weights > 1e-3, weighted_surface / numpy.maximum(weights, 1e-3), fill_value)
    except ImportError:
        pass
    gradient_y, gradient_x = numpy.gradient(filled)
    gradient = numpy.hypot(gradient_x, gradient_y)
    gradient[~mask] = 0.0
    values = gradient[finite]
    threshold = float(numpy.percentile(values, 90)) if len(values) else 0.0
    threshold = max(threshold, 1.25)
    edge = (gradient >= threshold) & mask
    # Keep a small eroded interior so dilation cannot put the cyan gradient
    # overlay back on the outer silhouette.  The normal boundary remains
    # drawn in white and this map is only for internal fold/occlusion cues.
    try:
        from scipy.ndimage import binary_erosion

        interior = binary_erosion(
            mask,
            structure=numpy.ones((3, 3), dtype=bool),
            iterations=2,
        )
    except ImportError:
        interior = mask & ~_mask_boundary(mask)
    edge &= interior
    try:
        from scipy.ndimage import binary_dilation, label

        edge = binary_dilation(edge, structure=numpy.ones((3, 3), dtype=bool), iterations=1)
        edge &= interior
        labels, count = label(edge, structure=numpy.ones((3, 3), dtype=numpy.uint8))
        if count:
            sizes = numpy.bincount(labels.reshape(-1))
            minimum_size = max(12, int(numpy.count_nonzero(mask) * 0.00015))
            keep = sizes >= minimum_size
            keep[0] = False
            edge = keep[labels]
    except ImportError:
        pass
    return edge, gradient.astype(numpy.float32)
def _project_base_points_to_frame(
    points_base_mm: np.ndarray,
    frame: RGBDFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project base-frame points into one camera's pixel raster."""

    numpy = _require_numpy()
    points = numpy.asarray(points_base_mm, dtype=numpy.float64) / 1000.0
    X = numpy.asarray(frame.X_base_camera, dtype=numpy.float64)
    camera_points = (points - X[:3, 3]) @ X[:3, :3]
    z = camera_points[:, 2]
    K = numpy.asarray(frame.intrinsics, dtype=numpy.float64)
    finite = numpy.isfinite(z) & (z > 1e-6)
    x = numpy.full(len(points), -1, dtype=numpy.int64)
    y = numpy.full(len(points), -1, dtype=numpy.int64)
    x[finite] = numpy.rint(K[0, 0] * camera_points[finite, 0] / z[finite] + K[0, 2]).astype(numpy.int64)
    y[finite] = numpy.rint(K[1, 1] * camera_points[finite, 1] / z[finite] + K[1, 2]).astype(numpy.int64)
    height, width = frame.depth_m.shape[:2]
    visible = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x, y, visible


def _project_base_points_with_camera_depth(
    points_base_mm: np.ndarray,
    frame: RGBDFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project base points and retain their camera-frame depth in metres."""

    numpy = _require_numpy()
    points = numpy.asarray(points_base_mm, dtype=numpy.float64) / 1000.0
    X = numpy.asarray(frame.X_base_camera, dtype=numpy.float64)
    camera_points = (points - X[:3, 3]) @ X[:3, :3]
    z = camera_points[:, 2]
    K = numpy.asarray(frame.intrinsics, dtype=numpy.float64)
    finite = numpy.isfinite(z) & (z > 1e-6)
    x = numpy.full(len(points), -1, dtype=numpy.int64)
    y = numpy.full(len(points), -1, dtype=numpy.int64)
    x[finite] = numpy.rint(
        K[0, 0] * camera_points[finite, 0] / z[finite] + K[0, 2]
    ).astype(numpy.int64)
    y[finite] = numpy.rint(
        K[1, 1] * camera_points[finite, 1] / z[finite] + K[1, 2]
    ).astype(numpy.int64)
    height, width = frame.depth_m.shape[:2]
    visible = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x, y, z, visible


def _camera_table_appearance_mask(
    rgb: np.ndarray,
    height_above_table_mm: np.ndarray,
    valid_depth: np.ndarray,
    *,
    minimum_color_distance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Separate garment appearance from the table in one camera's RGB space.

    The fused-cloud table color is a useful lower bound, but cameras A and B
    can have materially different white balance and illumination gradients.
    Estimate a per-camera table color from bright pixels close to table zero,
    then require the rendered garment silhouette to differ from that color.
    """

    numpy = _require_numpy()
    rgb = numpy.asarray(rgb, dtype=numpy.uint8)
    height_above_table_mm = numpy.asarray(height_above_table_mm, dtype=numpy.float64)
    valid_depth = numpy.asarray(valid_depth, dtype=bool)
    if rgb.shape[:2] != height_above_table_mm.shape:
        raise PerceptionError("RGB and height map shapes do not match")
    luma = (
        0.2126 * rgb[..., 0].astype(numpy.float64)
        + 0.7152 * rgb[..., 1].astype(numpy.float64)
        + 0.0722 * rgb[..., 2].astype(numpy.float64)
    )
    near_table = (
        valid_depth
        & numpy.isfinite(height_above_table_mm)
        & (numpy.abs(height_above_table_mm) <= 15.0)
    )
    if int(numpy.count_nonzero(near_table)) < 100:
        return numpy.ones(height_above_table_mm.shape, dtype=bool), {
            "applied": False,
            "reason": "fewer than 100 near-table pixels",
        }
    bright_cut = float(numpy.percentile(luma[near_table], 60.0))
    table_samples = near_table & (luma >= bright_cut)
    if int(numpy.count_nonzero(table_samples)) < 40:
        return numpy.ones(height_above_table_mm.shape, dtype=bool), {
            "applied": False,
            "reason": "fewer than 40 bright table samples",
        }
    table_rgb = numpy.median(
        rgb[table_samples].astype(numpy.float64),
        axis=0,
    )
    color_distance = numpy.linalg.norm(
        rgb.astype(numpy.float64) - table_rgb[None, None, :],
        axis=2,
    )
    table_color_noise = float(numpy.percentile(color_distance[table_samples], 50.0))
    threshold = float(
        max(
            24.0,
            float(minimum_color_distance),
            min(140.0, table_color_noise * 6.0),
        )
    )
    appearance_mask = color_distance >= threshold
    return appearance_mask, {
        "applied": True,
        "bright_luma_cut": bright_cut,
        "table_sample_count": int(numpy.count_nonzero(table_samples)),
        "table_rgb_median": [float(value) for value in table_rgb],
        "table_color_distance_p50": table_color_noise,
        "minimum_color_distance": float(minimum_color_distance),
        "applied_color_distance": threshold,
        "appearance_pixel_count": int(numpy.count_nonzero(appearance_mask)),
    }


def _occlusion_aware_garment_mask(
    garment_points_base_mm: np.ndarray,
    frame: RGBDFrame,
    height_above_table_mm: np.ndarray,
    valid_depth: np.ndarray,
    *,
    minimum_table_color_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rasterize garment points without coloring nearer robot/fixture pixels.

    The projected garment cloud supplies an expected camera depth.  A lightly
    expanded silhouette is accepted only where the observed RGB-D depth agrees
    with that expected surface.  This deliberately preserves holes caused by
    a gripper or fixture occluding the cloth.
    """

    numpy = _require_numpy()
    depth = numpy.asarray(frame.depth_m, dtype=numpy.float64)
    valid_depth = numpy.asarray(valid_depth, dtype=bool)
    x, y, projected_z_m, visible = _project_base_points_with_camera_depth(
        garment_points_base_mm, frame
    )
    sparse_mask = numpy.zeros(depth.shape, dtype=bool)
    expected_depth = numpy.full(depth.shape, numpy.inf, dtype=numpy.float64)
    if visible.any():
        numpy.minimum.at(expected_depth, (y[visible], x[visible]), projected_z_m[visible])
        sparse_mask[y[visible], x[visible]] = True
    # Build a complete outer silhouette before the depth check.  A broad hole
    # can be the gripper, but its nearer observed depth is rejected below; no
    # morphology is applied after that rejection step.
    silhouette = _solidify_largest_mask(
        sparse_mask,
        dilation_iterations=1,
        closing_iterations=1,
        fill_holes=True,
    )
    expected = expected_depth
    expected_valid = silhouette & numpy.isfinite(expected)
    try:
        from scipy.ndimage import distance_transform_edt

        if sparse_mask.any():
            _, nearest = distance_transform_edt(
                ~sparse_mask,
                return_distances=True,
                return_indices=True,
            )
            nearest_y, nearest_x = nearest
            expected = expected_depth[nearest_y, nearest_x]
            expected_valid = silhouette & numpy.isfinite(expected)
    except ImportError:
        pass
    observed = depth
    depth_consistent = (
        expected_valid
        & valid_depth
        & numpy.isfinite(observed)
        & (numpy.abs(observed - expected) <= 0.025)
    )
    # Keep the scalar map signed for diagnostics, but use a physical garment
    # envelope for visualization so robot/fixture heights cannot leak in.
    plausible_height = (
        numpy.isfinite(height_above_table_mm)
        & (height_above_table_mm >= -15.0)
        & (height_above_table_mm <= 160.0)
    )
    appearance_mask = numpy.ones(depth.shape, dtype=bool)
    appearance_diagnostics: dict[str, Any] = {
        "applied": False,
        "reason": "no minimum table-color distance supplied",
    }
    if minimum_table_color_distance is not None:
        appearance_mask, appearance_diagnostics = _camera_table_appearance_mask(
            frame.rgb,
            height_above_table_mm,
            valid_depth,
            minimum_color_distance=float(minimum_table_color_distance),
        )
    unconnected_mask = (
        silhouette & depth_consistent & plausible_height & appearance_mask
    )
    # Appearance filtering can split off dark cables, fixtures, or table-edge
    # shadows. Keep the largest remaining image component without filling the
    # holes back in, so occluders and rejected table pixels stay rejected.
    garment_mask = _solidify_largest_mask(
        unconnected_mask,
        dilation_iterations=0,
        closing_iterations=0,
        fill_holes=False,
    )
    diagnostics = {
        "projected_point_count": int(numpy.count_nonzero(visible)),
        "sparse_mask_pixels": int(numpy.count_nonzero(sparse_mask)),
        "silhouette_pixels": int(numpy.count_nonzero(silhouette)),
        "depth_consistent_pixels": int(numpy.count_nonzero(depth_consistent)),
        "pre_component_garment_mask_pixels": int(
            numpy.count_nonzero(unconnected_mask)
        ),
        "garment_mask_pixels": int(numpy.count_nonzero(garment_mask)),
        "depth_consistency_tolerance_mm": 25.0,
        "height_interval_mm": [-15.0, 160.0],
        "appearance_filter": appearance_diagnostics,
    }
    return garment_mask, sparse_mask, diagnostics


def camera_base_xyz_map_mm(
    frame: RGBDFrame,
    config: PerceptionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact calibrated robot-base XYZ for every valid camera pixel."""

    numpy = _require_numpy()
    depth = numpy.asarray(frame.depth_m, dtype=numpy.float64)
    valid = numpy.isfinite(depth) & (depth > config.min_depth_m) & (depth < config.max_depth_m)
    y_px, x_px = numpy.nonzero(valid)
    z_m = depth[valid]
    if z_m.size == 0:
        raise PerceptionError(f"camera {frame.label} has no valid depth points")
    K = numpy.asarray(frame.intrinsics, dtype=numpy.float64)
    camera_points = numpy.stack(
        [
            (x_px - K[0, 2]) * z_m / K[0, 0],
            (y_px - K[1, 2]) * z_m / K[1, 1],
            z_m,
        ],
        axis=1,
    )
    X = numpy.asarray(frame.X_base_camera, dtype=numpy.float64)
    base_points_mm = (camera_points @ X[:3, :3].T + X[:3, 3]) * 1000.0
    finite = numpy.all(numpy.isfinite(base_points_mm), axis=1)
    if not finite.any():
        raise PerceptionError(f"camera {frame.label} has no finite base-frame depth points")
    xyz_map = numpy.full((*depth.shape, 3), numpy.nan, dtype=numpy.float32)
    xyz_map[y_px[finite], x_px[finite]] = base_points_mm[finite].astype(numpy.float32)
    valid_map = numpy.zeros(depth.shape, dtype=bool)
    valid_map[y_px[finite], x_px[finite]] = True
    return xyz_map, valid_map


def camera_height_map_mm(
    frame: RGBDFrame,
    config: PerceptionConfig,
    table_coefficients: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a camera raster of surface height above the fitted table.

    The returned tuple is ``(height_map_mm, valid_mask, coefficients)``.  When
    no table plane is supplied, one is robustly fitted from this frame's base
    points.  This is used for live previews as well as the saved per-camera
    diagnostic maps; the dense A+B result still supplies the authoritative
    fused table plane to the saved artifacts.
    """

    numpy = _require_numpy()
    base_xyz_mm, valid_map = camera_base_xyz_map_mm(frame, config)
    y_px, x_px = numpy.nonzero(valid_map)
    base_points_mm = base_xyz_mm[y_px, x_px].astype(numpy.float64)
    if table_coefficients is None:
        colors = numpy.asarray(frame.rgb[y_px, x_px], dtype=numpy.uint8)
        reference_coefficients, reference_stats = _fit_table_plane_from_references(
            [frame], config, numpy.asarray([0.0, 0.0, 0.0], dtype=numpy.float64)
        )
        if reference_stats.get("mode") == "corner_edge_depth_interpolation":
            coefficients = reference_coefficients
        else:
            coefficients, _, _ = _fit_table_plane(base_points_mm, colors)
    else:
        coefficients = numpy.asarray(table_coefficients, dtype=numpy.float64)
        if coefficients.shape != (3,) or not numpy.all(numpy.isfinite(coefficients)):
            raise PerceptionError("table plane coefficients must contain three finite values")
    table_z_mm = (
        coefficients[0] * base_points_mm[:, 0]
        + coefficients[1] * base_points_mm[:, 1]
        + coefficients[2]
    )
    height_map = numpy.full(valid_map.shape, numpy.nan, dtype=numpy.float32)
    height_map[y_px, x_px] = (base_points_mm[:, 2] - table_z_mm).astype(numpy.float32)
    return height_map, valid_map, coefficients


def _save_camera_coordinate_guide(
    output_dir: Path,
    frame: RGBDFrame,
    config: PerceptionConfig,
    height_above_table_mm: np.ndarray,
    garment_mask: np.ndarray,
    *,
    sample_stride_px: int = 48,
) -> dict[str, str]:
    """Save an unranked image-to-base coordinate guide for Claude.

    The full-resolution ``.npy`` map stores calibrated XYZ for every valid
    pixel.  The PNG/JSON pair adds uniformly sampled reference markers so a
    visual planner can ground a self-selected garment region without
    perception proposing or ranking grasp candidates.
    """

    numpy = _require_numpy()
    from PIL import Image, ImageDraw

    if sample_stride_px < 8:
        raise PerceptionError("coordinate-guide sample stride must be at least 8 pixels")
    base_xyz_mm, valid = camera_base_xyz_map_mm(frame, config)
    garment_valid = (
        numpy.asarray(garment_mask, dtype=bool)
        & valid
        & numpy.all(numpy.isfinite(base_xyz_mm), axis=2)
        & numpy.isfinite(height_above_table_mm)
    )
    xyz_name = f"camera_{frame.label}_base_xyz_mm.npy"
    guide_name = f"camera_{frame.label}_coordinate_guide.json"
    overlay_name = f"camera_{frame.label}_coordinate_overlay.png"
    numpy.save(output_dir / xyz_name, base_xyz_mm)

    overlay = Image.fromarray(numpy.asarray(frame.rgb, dtype=numpy.uint8)).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    samples: list[dict[str, Any]] = []
    height, width = garment_valid.shape
    for y0 in range(0, height, sample_stride_px):
        for x0 in range(0, width, sample_stride_px):
            y1 = min(height, y0 + sample_stride_px)
            x1 = min(width, x0 + sample_stride_px)
            local_y, local_x = numpy.nonzero(garment_valid[y0:y1, x0:x1])
            if len(local_x) == 0:
                continue
            center_x = (x1 - x0 - 1) / 2.0
            center_y = (y1 - y0 - 1) / 2.0
            nearest = int(
                numpy.argmin((local_x - center_x) ** 2 + (local_y - center_y) ** 2)
            )
            x_px = int(x0 + local_x[nearest])
            y_px = int(y0 + local_y[nearest])
            xyz = base_xyz_mm[y_px, x_px].astype(numpy.float64)
            reference_id = f"R{len(samples) + 1:03d}"
            samples.append(
                {
                    "reference_id": reference_id,
                    "pixel_xy": [x_px, y_px],
                    "base_xyz_mm": [float(value) for value in xyz],
                    "height_above_table_mm": float(height_above_table_mm[y_px, x_px]),
                }
            )
            radius = 4
            draw.ellipse(
                (x_px - radius, y_px - radius, x_px + radius, y_px + radius),
                fill=(0, 255, 255),
                outline=(0, 0, 0),
                width=1,
            )
            draw.text(
                (x_px + 5, y_px - 6),
                reference_id,
                fill=(255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )

    overlay.save(output_dir / overlay_name)
    (output_dir / guide_name).write_text(
        json.dumps(
            {
                "camera_label": frame.label,
                "coordinate_frame": "robot_base_mm",
                "image_shape_yx": [int(height), int(width)],
                "sample_stride_px": int(sample_stride_px),
                "full_resolution_xyz_map": xyz_name,
                "overlay_image": overlay_name,
                "reference_semantics": (
                    "Uniform calibrated coordinate references over the garment mask; "
                    "these are not grasp candidates and have no ranking."
                ),
                "usage": (
                    "Claude selects a visually useful region, then uses the nearest "
                    "reference_id to obtain a measured robot-base XYZ. Do not invent "
                    "coordinates between references without explicitly stating uncertainty."
                ),
                "samples": samples,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "base_xyz_map": xyz_name,
        "coordinate_guide": guide_name,
        "coordinate_overlay": overlay_name,
    }


def _save_camera_height_heatmap(
    output_dir: Path,
    frame: RGBDFrame,
    config: PerceptionConfig,
    garment_points_base_mm: np.ndarray,
    table_coefficients: np.ndarray,
    *,
    display_max_mm: float | None = None,
    minimum_table_color_distance: float | None = None,
) -> dict[str, Any]:
    """Save camera-pixel height-above-table maps and heatmaps.

    Each valid RGB-D pixel is transformed into the robot base frame and
    compared with the fitted table plane at the same ``x/y`` location.  The
    saved scalar map and both heatmaps therefore represent the same quantity:
    ``surface_z_mm - table_z_mm``.
    """

    numpy = _require_numpy()
    from PIL import Image, ImageDraw

    depth = numpy.asarray(frame.depth_m, dtype=numpy.float64)
    height_above_table_mm, valid, _ = camera_height_map_mm(
        frame,
        config,
        table_coefficients,
    )
    garment_mask, coordinate_reference_mask, projection_diagnostics = (
        _occlusion_aware_garment_mask(
            garment_points_base_mm,
            frame,
            height_above_table_mm,
            valid,
            minimum_table_color_distance=minimum_table_color_distance,
        )
    )
    if display_max_mm is None:
        display_max_mm = _height_display_max_mm(height_above_table_mm[garment_mask])
    display_max_mm = float(max(1.0, display_max_mm))
    garment_valid = valid & garment_mask & numpy.isfinite(height_above_table_mm)
    height_map_name = f"camera_{frame.label}_height_above_table_mm.npy"
    numpy.save(output_dir / height_map_name, height_above_table_mm)
    # Save the actual table references used by the corner/edge interpolation
    # so the zero surface can be audited independently of the color image.
    _, table_reference_records = _sample_table_reference_points(frame, config)
    for record in table_reference_records:
        if record.get("valid") and "base_xyz_mm" in record:
            base_x, base_y, base_z = record["base_xyz_mm"]
            residual_mm = float(
                base_z
                - (
                    table_coefficients[0] * base_x
                    + table_coefficients[1] * base_y
                    + table_coefficients[2]
                )
            )
            record["plane_residual_mm"] = residual_mm
            record["plane_inlier"] = abs(residual_mm) <= 15.0
    table_reference_name = f"camera_{frame.label}_table_references.json"
    table_reference_overlay_name = f"camera_{frame.label}_table_references.png"
    table_reference_overlay = Image.fromarray(
        numpy.asarray(frame.rgb, dtype=numpy.uint8)
    ).convert("RGB")
    reference_draw = ImageDraw.Draw(table_reference_overlay)
    for record in table_reference_records:
        x_px, y_px = record["pixel_xy"]
        color = (
            (0, 255, 0)
            if record.get("plane_inlier")
            else (255, 80, 40)
            if record.get("valid")
            else (255, 0, 0)
        )
        reference_draw.ellipse(
            (x_px - 6, y_px - 6, x_px + 6, y_px + 6),
            outline=color,
            width=2,
        )
        reference_draw.text(
            (x_px + 7, y_px - 7),
            str(record["name"]),
            fill=color,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    table_reference_overlay.save(output_dir / table_reference_overlay_name)
    (output_dir / table_reference_name).write_text(
        json.dumps(
            {
                "camera_label": frame.label,
                "method": "corner_edge_depth_interpolation",
                "table_plane_coefficients": [float(value) for value in table_coefficients],
                "samples": table_reference_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    base_xyz_mm, xyz_valid = camera_base_xyz_map_mm(frame, config)
    table_surface_map = numpy.full(depth.shape, numpy.nan, dtype=numpy.float32)
    table_surface_map[xyz_valid] = (
        table_coefficients[0] * base_xyz_mm[xyz_valid, 0]
        + table_coefficients[1] * base_xyz_mm[xyz_valid, 1]
        + table_coefficients[2]
    ).astype(numpy.float32)
    table_surface_name = f"camera_{frame.label}_table_z_mm.npy"
    numpy.save(output_dir / table_surface_name, table_surface_map)
    heatmap = _scalar_heatmap_rgb(
        height_above_table_mm,
        valid,
        focus_mask=garment_mask,
        higher_is_bright=True,
        value_range_mm=(0.0, display_max_mm),
    )
    global_heatmap = _scalar_heatmap_rgb(
        height_above_table_mm,
        valid,
        higher_is_bright=True,
        value_range_mm=(0.0, display_max_mm),
    )
    boundary = _outer_mask_boundary(garment_mask)
    fold_edges, _ = _fold_edge_mask(height_above_table_mm, garment_mask)
    overlay = heatmap.copy()
    overlay[boundary] = numpy.asarray([255, 255, 255], dtype=numpy.uint8)
    fold_overlay = overlay.copy()
    fold_overlay[fold_edges] = numpy.asarray([0, 255, 255], dtype=numpy.uint8)
    heatmap_name = f"camera_{frame.label}_height_map_heatmap.png"
    global_heatmap_name = f"camera_{frame.label}_height_map_heatmap_global.png"
    boundary_name = f"camera_{frame.label}_height_map_boundary.png"
    fold_edge_name = f"camera_{frame.label}_height_gradient_edges.png"
    Image.fromarray(heatmap).save(output_dir / heatmap_name)
    Image.fromarray(global_heatmap).save(output_dir / global_heatmap_name)
    Image.fromarray(overlay).save(output_dir / boundary_name)
    Image.fromarray(fold_overlay).save(output_dir / fold_edge_name)
    coordinate_artifacts = _save_camera_coordinate_guide(
        output_dir,
        frame,
        config,
        height_above_table_mm,
        # Use the final occlusion/appearance/depth-consistent garment mask.
        # The sparse projected mask is retained for diagnostics, but it can
        # contain isolated projected points on table/fixture pixels.
        garment_mask,
    )
    return {
        "height_map": heatmap_name,
        "height_map_global": global_heatmap_name,
        "height_map_boundary": boundary_name,
        "height_map_path": height_map_name,
        "table_z_map": table_surface_name,
        "table_references": table_reference_name,
        "table_reference_overlay": table_reference_overlay_name,
        "heatmap": heatmap_name,
        "global_heatmap": global_heatmap_name,
        "boundary_overlay": boundary_name,
        "height_gradient_overlay": fold_edge_name,
        **coordinate_artifacts,
        # Compatibility alias retained for existing consumers.
        "fold_edge_overlay": fold_edge_name,
        "height_map_quantity": "height_above_table_mm",
        "heatmap_quantity": "height_above_table_mm",
        "heatmap_display_min_mm": 0.0,
        "heatmap_display_max_mm": display_max_mm,
        "heatmap_normalization": "absolute_table_zero_shared",
        "projection_diagnostics": projection_diagnostics,
        "height_min_mm": float(numpy.percentile(height_above_table_mm[garment_valid], 2))
        if garment_valid.any()
        else None,
        "height_max_mm": float(numpy.percentile(height_above_table_mm[garment_valid], 98))
        if garment_valid.any()
        else None,
    }


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
    """Convert a fused garment point into an observation-only experiment state.

    Perception is deliberately not a motion planner.  It reports the fused
    garment center and the observed surface height; Claude chooses the grasp,
    approach, lift, transfer, release, and yaw waypoints from the saved views.
    The historical function name is retained for callers, but the returned
    :class:`ExperimentConfig` intentionally leaves every motion waypoint
    deferred.
    """

    numpy = _require_numpy()
    center = numpy.asarray(center_base_mm, dtype=numpy.float64)
    if center.shape != (3,) or not numpy.all(numpy.isfinite(center)):
        raise PerceptionError("fused cloth center must be a finite xyz point")

    surface_z = float(center[2])
    bounds = robot_config.boundaries
    if bounds.z_min is None or bounds.z_max is None:
        raise PerceptionError("z_min and z_max are required to validate the observed surface")
    safe_z_min = float(bounds.z_min + robot_config.lower_z_margin_mm)
    safe_z_max = float(bounds.z_max - robot_config.workspace_margin_mm)
    if safe_z_min >= safe_z_max:
        raise PerceptionError("workspace margin leaves no usable z range")
    try:
        # This validates the observation against the measured workspace, but it
        # does not turn the observed surface height into a TCP target.
        bounds.validate(
            float(center[0]),
            float(center[1]),
            surface_z,
            robot_config.workspace_margin_mm,
            z_lower_margin_mm=robot_config.lower_z_margin_mm,
        )
    except Exception as exc:
        raise PerceptionError(f"fused garment center is outside the measured workspace: {exc}") from exc

    plan = ExperimentConfig(
        cloth_center_x=float(center[0]),
        cloth_center_y=float(center[1]),
        # ``grasp_z`` is retained as the serialized compatibility field for
        # the observed surface height.  It is not a grasp command.
        grasp_z=surface_z,
        approach_z=None,
        lift_z=None,
        yaw_deg=None,
    )
    derivation = {
        "surface_z_mm": surface_z,
        "tcp_coordinates": "controller-configured tool center point",
        "waypoint_authority": "Claude",
        "perception_outputs": ["cloth_center_x_mm", "cloth_center_y_mm", "surface_z_mm"],
        "motion_waypoints_generated": False,
        "approach_z_mm": None,
        "grasp_z_mm": None,
        "lift_z_mm": None,
        "transfer_waypoints_mm": None,
        "release_z_mm": None,
        "yaw_deg": None,
        "deprecated_clearance_config_ignored": {
            "grasp_contact_clearance_mm": perception_config.grasp_contact_clearance_mm,
            "approach_clearance_mm": perception_config.approach_clearance_mm,
            "lift_clearance_mm": perception_config.lift_clearance_mm,
        },
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
        # Compatibility only; dense AB fusion never launches Molmo.
        self.molmo = molmo_client

    def locate(
        self,
        output_dir: Path,
        experiment: ExperimentConfig,
        frames: list[RGBDFrame] | None = None,
    ) -> tuple[dict[str, Any], ExperimentConfig]:
        return self._locate_dense_ab(output_dir, experiment, frames=frames)

    def _locate_dense_ab(
        self,
        output_dir: Path,
        experiment: ExperimentConfig,
        frames: list[RGBDFrame] | None = None,
    ) -> tuple[dict[str, Any], ExperimentConfig]:
        numpy = _require_numpy()
        from PIL import Image

        output_dir.mkdir(parents=True, exist_ok=False)
        temporal_median_applied = frames is None and self.capture is capture_two_view_rgbd
        frames = self.capture(self.config) if frames is None else frames
        expected_labels = set(self.config.active_camera_labels)
        frame_labels = {frame.label for frame in frames}
        if len(frames) != 2 or frame_labels != expected_labels:
            raise PerceptionError(
                f"dense AB fusion expected frames {sorted(expected_labels)}, got {sorted(frame_labels)}"
            )

        views: list[dict[str, Any]] = []
        for index, frame in enumerate(frames):
            if frame.rgb.shape[:2] != frame.depth_m.shape[:2]:
                raise PerceptionError(f"camera {frame.label} RGB/depth shapes do not match")
            image_name = f"camera_{index}_{frame.label}.png"
            Image.fromarray(frame.rgb.astype(numpy.uint8)).save(output_dir / image_name)
            numpy.save(output_dir / f"camera_{index}_{frame.label}_depth_m.npy", frame.depth_m)
            valid = numpy.isfinite(frame.depth_m) & (
                (frame.depth_m > self.config.min_depth_m)
                & (frame.depth_m < self.config.max_depth_m)
            )
            views.append(
                {
                    "label": frame.label,
                    "role": "rgbd_fusion_source",
                    "serial": frame.serial,
                    "image": image_name,
                    "depth_m": f"camera_{index}_{frame.label}_depth_m.npy",
                    "intrinsics": frame.intrinsics.tolist(),
                    "X_base_camera": frame.X_base_camera.tolist(),
                    "valid_depth_fraction": float(valid.mean()),
                    "temporal_median_frames": int(self.config.temporal_median_frames)
                    if temporal_median_applied
                    else 1,
                }
            )

        fused_points, fused_colors, source_mask, fusion_stats = _voxel_fuse_base_points(
            frames,
            self.config,
            self.robot_config,
        )
        coefficients, table_residual, table_stats = _fit_table_plane(
            fused_points, fused_colors
        )
        coefficients, table_reference_stats = _fit_table_plane_from_references(
            frames,
            self.config,
            coefficients,
        )
        table_stats["reference_interpolation"] = table_reference_stats
        design = numpy.column_stack(
            (fused_points[:, 0], fused_points[:, 1], numpy.ones(len(fused_points)))
        )
        table_z = design @ coefficients
        height_above_table = fused_points[:, 2] - table_z
        table_residual = fused_points[:, 2] - table_z
        table_stats["coefficients"] = {
            "a": float(coefficients[0]),
            "b": float(coefficients[1]),
            "c_mm": float(coefficients[2]),
        }
        table_stats["coefficient_source"] = table_reference_stats.get(
            "mode", "fused_cloud"
        )
        table_noise_mm = float(
            table_stats.get("residual_p95_abs_mm")
            or (numpy.percentile(numpy.abs(table_residual), 90) if len(table_residual) else 5.0)
        )
        relief_threshold_mm = max(5.0, min(15.0, table_noise_mm * 2.5))
        table_rgb = numpy.asarray(
            table_stats.get("table_rgb_median", [255.0, 255.0, 255.0]),
            dtype=numpy.float64,
        )
        color_distance = numpy.linalg.norm(
            fused_colors.astype(numpy.float64) - table_rgb[None, :],
            axis=1,
        )
        table_color_noise = float(table_stats.get("table_color_distance_p50") or 0.0)
        garment_color_threshold = max(24.0, min(80.0, table_color_noise * 4.0))
        lower_surface_tolerance_mm = max(12.0, table_noise_mm * 2.0)
        # The complete garment mask must include fabric lying directly on the
        # table.  Height-only thresholding selected only raised folds and
        # produced the thin-strip boundary seen in Viser.  Use appearance
        # difference from the fitted table plus a loose table-height envelope
        # for the outer garment mask; keep height relief as a separate fold cue.
        garment_candidate = (
            (color_distance >= garment_color_threshold)
            & (height_above_table >= -lower_surface_tolerance_mm)
            & (height_above_table <= 160.0)
        )
        garment_candidate = _largest_xy_component(fused_points, garment_candidate)
        garment_indices = numpy.flatnonzero(garment_candidate)
        if len(garment_indices) < 100:
            raise PerceptionError(
                "dense AB fusion found fewer than 100 full-garment points distinct from the table"
            )
        relief_candidate = garment_candidate & (height_above_table >= relief_threshold_mm)
        garment_points = fused_points[garment_indices]
        garment_heights = height_above_table[garment_indices]
        heatmap_display_max_mm = _height_display_max_mm(garment_heights)
        xy_center = numpy.median(garment_points[:, :2], axis=0)
        distance = numpy.sum((garment_points[:, :2] - xy_center) ** 2, axis=1)
        nearest = numpy.argsort(distance)[: min(128, len(distance))]
        center_point = garment_points[nearest[numpy.argmax(garment_points[nearest, 2])]]
        # Use the robust component median for XY, while taking surface height
        # from a nearby upper-surface point.  This prevents one protruding edge
        # from moving the grasp target outside the calibrated work envelope.
        center = numpy.asarray(
            [xy_center[0], xy_center[1], center_point[2]], dtype=numpy.float64
        )

        fused_points_path = output_dir / "fused_points_base_mm.npy"
        fused_colors_path = output_dir / "fused_colors_rgb.npy"
        fused_source_path = output_dir / "fused_source_mask.npy"
        numpy.save(fused_points_path, fused_points.astype(numpy.float32))
        numpy.save(fused_colors_path, fused_colors.astype(numpy.uint8))
        numpy.save(fused_source_path, source_mask)
        numpy.save(
            output_dir / "fused_height_above_table_mm.npy",
            height_above_table.astype(numpy.float32),
        )
        numpy.save(output_dir / "fused_garment_mask.npy", garment_candidate)
        numpy.save(output_dir / "fused_relief_mask.npy", relief_candidate)
        height_map = _save_fused_height_map(
            output_dir,
            fused_points,
            height_above_table,
            garment_candidate,
            display_max_mm=heatmap_display_max_mm,
        )
        camera_heatmaps: dict[str, dict[str, Any]] = {}
        for frame, view in zip(frames, views):
            artifacts = _save_camera_height_heatmap(
                output_dir,
                frame,
                self.config,
                garment_points,
                coefficients,
                display_max_mm=heatmap_display_max_mm,
                minimum_table_color_distance=garment_color_threshold,
            )
            camera_heatmaps[frame.label] = artifacts
            view.update(
                {
                    "height_map": artifacts["height_map"],
                    "height_map_global": artifacts["height_map_global"],
                    "height_map_boundary": artifacts["height_map_boundary"],
                    # Keep the old keys as aliases for existing run viewers and
                    # saved workspaces.  Their contents are now height maps.
                    "depth_heatmap": artifacts["height_map"],
                    "depth_heatmap_global": artifacts["height_map_global"],
                    "depth_heatmap_boundary": artifacts["height_map_boundary"],
                    "height_map_path": artifacts.get("height_map_path"),
                    "fold_edge_overlay": artifacts["fold_edge_overlay"],
                    "height_gradient_overlay": artifacts.get("height_gradient_overlay"),
                    "base_xyz_map": artifacts.get("base_xyz_map"),
                    "coordinate_guide": artifacts.get("coordinate_guide"),
                    "coordinate_overlay": artifacts.get("coordinate_overlay"),
                    "table_reference_overlay": artifacts.get("table_reference_overlay"),
                    "table_references": artifacts.get("table_references"),
                    "table_z_map": artifacts.get("table_z_map"),
                    "height_map_min_mm": artifacts.get("height_min_mm"),
                    "height_map_max_mm": artifacts.get("height_max_mm"),
                    "heatmap_quantity": artifacts.get("heatmap_quantity"),
                }
            )

        observation_plan, motion_derivation = derive_grasp_plan(
            center, self.robot_config, self.config
        )
        updated = replace(experiment, **observation_plan.as_dict())
        updated.require_center()
        depth_fusion = {
            "mode": "dense_ab_voxel_fusion",
            "source_cameras": list(self.config.active_camera_labels),
            "temporal_median_frames": int(self.config.temporal_median_frames)
            if temporal_median_applied
            else 1,
            "temporal_median_applied": bool(temporal_median_applied),
            **fusion_stats,
            "table_plane": table_stats,
            "table_noise_p90_mm": table_noise_mm,
            "garment_relief_threshold_mm": relief_threshold_mm,
            "garment_color_distance_threshold": garment_color_threshold,
            "garment_lower_surface_tolerance_mm": lower_surface_tolerance_mm,
            "garment_point_count": int(len(garment_points)),
            "relief_point_count": int(numpy.count_nonzero(relief_candidate)),
            "garment_height_p50_mm": float(numpy.percentile(garment_heights, 50)),
            "garment_height_p95_mm": float(numpy.percentile(garment_heights, 95)),
            "heatmap_display_min_mm": 0.0,
            "heatmap_display_max_mm": heatmap_display_max_mm,
            "heatmap_normalization": "absolute_table_zero_shared",
            "center_base_mm": center.tolist(),
            "artifacts": {
                "fused_points_base_mm": fused_points_path.name,
                "fused_colors_rgb": fused_colors_path.name,
                "fused_source_mask": fused_source_path.name,
                "fused_height_above_table_mm": "fused_height_above_table_mm.npy",
                "fused_garment_mask": "fused_garment_mask.npy",
                "fused_relief_mask": "fused_relief_mask.npy",
                "camera_height_maps": camera_heatmaps,
                # Backward-compatible key for older consumers.
                "camera_depth_heatmaps": camera_heatmaps,
                **height_map,
            },
        }
        result = {
            "created_at": _now(),
            "status": "VALIDATED_DENSE_AB_FUSION",
            "perception_mode": "dense_ab_rgbd_fusion",
            "active_cameras": list(self.config.active_camera_labels),
            "primary_camera": "A",
            "auxiliary_depth_cameras": ["B"],
            "views": views,
            "warnings": [],
            "depth_fusion": depth_fusion,
            "center_base_mm": center.tolist(),
            "surface_z_mm": float(center[2]),
            "motion_derivation": motion_derivation,
            "waypoint_authority": "Claude",
            "waypoints": None,
            "experiment_config": updated.as_dict(),
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result, updated

    def _locate_legacy_molmo(
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
        observation_plan, motion_derivation = derive_grasp_plan(
            center, self.robot_config, self.config
        )
        updated = replace(experiment, **observation_plan.as_dict())
        updated.require_center()
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
            "surface_z_mm": float(center[2]),
            "motion_derivation": motion_derivation,
            "waypoint_authority": "Claude",
            "waypoints": None,
            "experiment_config": updated.as_dict(),
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result, updated
