#!/usr/bin/env python3
"""Manually align two live RealSense point clouds in Viser.

Camera A is fixed with its configured ``X_BaseCamera``.  Camera B is the
movable cloud: drag ``camera_pose_B`` until the two clouds overlap, then click
``save_target_camera_pose``.  The saved matrix uses the same convention as the
perception pipeline::

    base_point = X_BaseCamera @ camera_point

Example::

    /home/CNS2026330003/miniconda3/envs/cali/bin/python \
      scripts/drag_camera_pose.py \
      --reference-label A \
      --target-label B \
      --output-yaml config/extrinsics_B_manual.yaml

The script intentionally does not connect to or move the robot.  It only
opens the two cameras, renders their point clouds, and saves the target
camera's adjusted extrinsic transform.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.run_storage import auxiliary_dir
from cloth_agent.perception import (  # noqa: E402
    CameraSpec,
    PerceptionConfig,
    PerceptionError,
    RealSenseRGBD,
    load_extrinsics,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drag the target RealSense pose until two live point clouds align."
    )
    parser.add_argument(
        "--perception-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "perception.free_exploration.json",
    )
    parser.add_argument("--reference-label", default="A", help="Fixed camera label (default: A).")
    parser.add_argument("--target-label", default="B", help="Camera label to drag (default: B).")
    parser.add_argument(
        "--camera-label",
        dest="target_label_compat",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--serial", default=None, help="Override the target camera serial.")
    parser.add_argument(
        "--initial-pose",
        type=Path,
        default=None,
        help="Optional target .npy or YAML pose to load before dragging.",
    )
    parser.add_argument(
        "--output-npy",
        type=Path,
        default=None,
        help="Target pose output .npy; defaults to runs/manual_camera_pose/camera_<target>.npy.",
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=None,
        help="Optional target YAML output using the X_CammountCam key.",
    )
    parser.add_argument("--stride", type=int, default=4, help="Point-cloud pixel stride (default: 4).")
    parser.add_argument("--refresh-hz", type=float, default=5.0, help="Refresh rate (default: 5 Hz).")
    parser.add_argument("--point-size", type=float, default=0.003, help="Viser point size in metres.")
    return parser


def _resolve_path(path: Path, *, root: Path = PROJECT_ROOT) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def _load_pose(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        pose = np.load(path, allow_pickle=False)
    else:
        pose = load_extrinsics(path)
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise PerceptionError(f"camera pose must be a finite 4x4 matrix: {path}")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise PerceptionError(f"invalid homogeneous row in camera pose: {path}")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise PerceptionError(f"camera pose rotation is not orthonormal: {path}")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-4):
        raise PerceptionError(f"camera pose rotation must have determinant +1: {path}")
    return pose


def _save_pose_npy(path: Path, pose: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(pose, dtype=np.float64))


def _save_pose_yaml(path: Path, pose: np.ndarray) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise PerceptionError("PyYAML is required for --output-yaml") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"X_CammountCam": np.asarray(pose).tolist()}, sort_keys=False),
        encoding="utf-8",
    )


def _camera_point_cloud(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    X_base_camera: np.ndarray,
    *,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    if stride < 1:
        raise ValueError("--stride must be at least 1")
    depth = np.asarray(depth_m, dtype=np.float64)[::stride, ::stride]
    colors = np.asarray(rgb, dtype=np.uint8)[::stride, ::stride]
    K = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2 or colors.shape[:2] != depth.shape or K.shape != (3, 3):
        raise PerceptionError("RGB, depth, and intrinsics have incompatible shapes")
    y_px, x_px = np.indices(depth.shape)
    y_px *= stride
    x_px *= stride
    valid = np.isfinite(depth) & (depth > 0.15) & (depth < 2.0)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    z = depth[valid]
    camera_points = np.stack(
        [
            (x_px[valid] - K[0, 2]) * z / K[0, 0],
            (y_px[valid] - K[1, 2]) * z / K[1, 1],
            z,
        ],
        axis=1,
    )
    X = np.asarray(X_base_camera, dtype=np.float64)
    base_points = camera_points @ X[:3, :3].T + X[:3, 3]
    return base_points.astype(np.float32), colors[valid].astype(np.uint8)


def _pose_from_control(control: Any) -> np.ndarray:
    wxyz = np.asarray(control.wxyz, dtype=np.float64)
    position = np.asarray(control.position, dtype=np.float64)
    if wxyz.shape != (4,) or position.shape != (3,):
        raise PerceptionError("Viser transform control returned an invalid pose")
    norm = float(np.linalg.norm(wxyz))
    if not np.isfinite(norm) or norm < 1e-8:
        raise PerceptionError("Viser transform control returned an invalid quaternion")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat(wxyz[[1, 2, 3, 0]] / norm).as_matrix()
    pose[:3, 3] = position
    return pose


def _target_spec(config: PerceptionConfig, label: str, serial: str | None) -> CameraSpec:
    specs = {camera.label: camera for camera in config.cameras}
    if label not in specs:
        raise PerceptionError(f"camera label {label!r} is not configured; available={sorted(specs)}")
    spec = specs[label]
    if serial is None:
        return spec
    return CameraSpec(
        label=spec.label,
        serial=str(serial).strip(),
        extrinsics_file=spec.extrinsics_file,
        color_exposure=spec.color_exposure,
        color_white_balance=spec.color_white_balance,
    )


def run(options: argparse.Namespace) -> int:
    import viser

    config = PerceptionConfig.load(PROJECT_ROOT, _resolve_path(options.perception_config))
    reference_label = str(options.reference_label).strip().upper()
    target_label = str(options.target_label_compat or options.target_label).strip().upper()
    if reference_label == target_label:
        raise PerceptionError("reference and target cameras must be different")
    reference_spec = _target_spec(config, reference_label, None)
    target_spec = _target_spec(config, target_label, options.serial)
    specs = {camera.label: camera for camera in config.cameras}

    output_npy = (
        _resolve_path(options.output_npy)
        if options.output_npy is not None
        else auxiliary_dir(PROJECT_ROOT, "manual_camera_pose") / f"camera_{target_label}.npy"
    )
    target_pose_path = (
        _resolve_path(options.initial_pose)
        if options.initial_pose is not None
        else output_npy
        if output_npy.is_file()
        else target_spec.extrinsics_file
    )
    reference_pose = _load_pose(specs[reference_label].extrinsics_file)
    target_pose = _load_pose(target_pose_path)
    output_yaml = _resolve_path(options.output_yaml) if options.output_yaml else None
    refresh_hz = float(options.refresh_hz)
    if not np.isfinite(refresh_hz) or refresh_hz <= 0:
        raise ValueError("--refresh-hz must be finite and positive")

    server = viser.ViserServer()
    reference_camera = RealSenseRGBD(
        reference_spec, config.width, config.height, config.fps
    )
    target_camera = RealSenseRGBD(target_spec, config.width, config.height, config.fps)
    reference_camera.start()
    try:
        target_camera.start()
    except BaseException:
        reference_camera.stop()
        raise

    stop = False

    def request_stop(*_: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    target_wxyz = Rotation.from_matrix(target_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
    control = server.scene.add_transform_controls(
        f"camera_pose_{target_label}",
        opacity=0.85,
        disable_sliders=False,
        scale=0.25,
        line_width=2.5,
        wxyz=tuple(float(value) for value in target_wxyz),
        position=tuple(float(value) for value in target_pose[:3, 3]),
    )
    reference_cloud = server.scene.add_point_cloud(
        f"camera_pointcloud_{reference_label}",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        point_size=float(options.point_size),
        point_shape="circle",
    )
    target_cloud = server.scene.add_point_cloud(
        f"camera_pointcloud_{target_label}",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        point_size=float(options.point_size),
        point_shape="circle",
    )
    ref_rgb_handle = server.gui.add_image(
        np.zeros((config.height, config.width, 3), dtype=np.uint8),
        label=f"Camera {reference_label} RGB",
    )
    target_rgb_handle = server.gui.add_image(
        np.zeros((config.height, config.width, 3), dtype=np.uint8),
        label=f"Camera {target_label} RGB",
    )
    status = server.gui.add_markdown("### Camera alignment\n\nStarting cameras...")
    save_button = server.gui.add_button("save_target_camera_pose")
    reset_button = server.gui.add_button("reset_target_pose")

    def save_current_pose() -> None:
        current = _pose_from_control(control)
        _save_pose_npy(output_npy, current)
        if output_yaml is not None:
            _save_pose_yaml(output_yaml, current)
        status.content = (
            "### Camera alignment\n\nSaved target pose: "
            f"`{output_npy}`"
            + (f" and `{output_yaml}`" if output_yaml else "")
        )

    @save_button.on_click
    def _(_: Any) -> None:
        try:
            save_current_pose()
        except Exception as exc:
            status.content = f"### Camera alignment\n\nSave failed: `{exc}`"

    @reset_button.on_click
    def _(_: Any) -> None:
        control.wxyz = tuple(float(value) for value in target_wxyz)
        control.position = tuple(float(value) for value in target_pose[:3, 3])

    print(
        f"Viser camera alignment started: fixed={reference_label} ({reference_spec.serial}), "
        f"movable={target_label} ({target_spec.serial})",
        flush=True,
    )
    print(
        f"Drag camera_pose_{target_label} until the clouds overlap, then click "
        "save_target_camera_pose.",
        flush=True,
    )
    try:
        interval = 1.0 / refresh_hz
        while not stop:
            current_target_pose = _pose_from_control(control)
            ref_rgb, ref_depth = reference_camera.read()
            target_rgb, target_depth = target_camera.read()
            if reference_camera.intrinsics is None or target_camera.intrinsics is None:
                raise PerceptionError("camera intrinsics are unavailable after startup")
            ref_points, ref_colors = _camera_point_cloud(
                ref_rgb,
                ref_depth,
                reference_camera.intrinsics,
                reference_pose,
                stride=int(options.stride),
            )
            target_points, target_colors = _camera_point_cloud(
                target_rgb,
                target_depth,
                target_camera.intrinsics,
                current_target_pose,
                stride=int(options.stride),
            )
            if len(ref_points):
                reference_cloud.points = ref_points
                reference_cloud.colors = ref_colors
                reference_cloud.visible = True
            if len(target_points):
                target_cloud.points = target_points
                target_cloud.colors = target_colors
                target_cloud.visible = True
            ref_rgb_handle.image = ref_rgb
            target_rgb_handle.image = target_rgb
            ref_valid = float(np.isfinite(ref_depth).mean()) * 100.0
            target_valid = float(np.isfinite(target_depth).mean()) * 100.0
            status.content = (
                "### Camera alignment\n\n"
                f"- fixed: `{reference_label}` / `{reference_spec.serial}`\n"
                f"- movable: `{target_label}` / `{target_spec.serial}`\n"
                f"- resolution: `{ref_rgb.shape[1]} × {ref_rgb.shape[0]}`\n"
                f"- valid depth: `{reference_label} {ref_valid:.1f}%`, `{target_label} {target_valid:.1f}%`\n"
                f"- points: `{reference_label} {len(ref_points):,}`, `{target_label} {len(target_points):,}`\n"
                f"- output: `{output_npy}`\n\n"
                f"Drag `camera_pose_{target_label}` until the two clouds overlap."
            )
            time.sleep(interval)
    finally:
        target_camera.stop()
        reference_camera.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(_parser().parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"drag_camera_pose: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
