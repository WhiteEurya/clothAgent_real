"""Stationary RGB-D snapshots with explicit arm ownership and world geometry."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import DualArmError, finite, transform


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Observation:
    directory: Path
    meta: dict
    xyz: np.ndarray

    @property
    def image(self):
        return self.directory / "rgb.png"

    @classmethod
    def load(cls, directory):
        directory = Path(directory).resolve()
        meta = json.loads((directory / "observation.json").read_text())
        if meta.get("schema_version") != 1 or meta.get("frame_id") != "world_mm":
            raise DualArmError("observation frame/schema mismatch")
        for name in ("rgb.png", "world_xyz_mm.npy"):
            if sha(directory / name) != meta["sha256"][name]:
                raise DualArmError(f"observation artifact changed: {name}")
        xyz = np.load(directory / "world_xyz_mm.npy", allow_pickle=False)
        with Image.open(directory / "rgb.png") as image:
            if xyz.shape != (image.height, image.width, 3):
                raise DualArmError("RGB and geometry dimensions differ")
        return cls(directory, meta, xyz)

    def validate_for(self, config, *, live=False, check_age=True):
        """Validate provenance; callers may disable the wall-clock age gate.

        Astra plans from a stationary round snapshot and checks live joints
        before dispatch. Its model/planner latency is not an expiry condition.
        """
        if self.meta["calibration_id"] != config.raw["calibration_id"]:
            raise DualArmError("observation uses a different calibration")
        if live:
            if self.meta.get("synthetic") is not False:
                raise DualArmError(
                    "real execution requires a real captured observation"
                )
            if check_age:
                age = time.time() - float(self.meta["captured_unix_s"])
                if not 0 <= age <= config.limits["max_grasp_age_s"]:
                    raise DualArmError("grasp observation expired; capture and plan again")

    def sample(self, pixel, max_spread):
        x, y = map(int, finite(pixel, (2,), "pixel"))
        h, w, _ = self.xyz.shape
        if not (2 <= x < w - 2 and 2 <= y < h - 2):
            raise DualArmError("grasp pixel outside supported image interior")
        patch = self.xyz[y - 2 : y + 3, x - 2 : x + 3].reshape(-1, 3)
        valid = patch[np.isfinite(patch).all(axis=1)]
        if len(valid) < 18 or not np.isfinite(self.xyz[y, x]).all():
            raise DualArmError("grasp has insufficient measured depth")
        if np.ptp(valid[:, 2]) > max_spread:
            raise DualArmError("grasp depth patch crosses a discontinuity")
        return np.median(valid, axis=0)


def save_observation(
    directory, rgb, xyz, config, joints, *, synthetic=False, extra=None
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(directory / "rgb.png")
    extra = dict(extra or {})
    correction = config.raw.get('camera', {}).get('world_tilt_correction')
    if correction is not None:
        from ..depth_level import rotate_cloud
        if correction.get('mode') != 'world_y_rotation':
            raise ValueError('unknown world tilt correction mode')
        np.save(directory/'uncorrected_world_xyz_mm.npy', np.asarray(xyz,dtype=np.float32),allow_pickle=False)
        xyz, rotation = rotate_cloud(xyz, correction['pivot_world_mm'], correction['angle_deg'])
        pivot = np.asarray(correction['pivot_world_mm'])
        correction_matrix = np.eye(4)
        correction_matrix[:3,:3] = rotation
        correction_matrix[:3,3] = pivot - rotation @ pivot
        if 'world_from_camera_mm' in extra:
            extra['uncorrected_world_from_camera_mm'] = extra['world_from_camera_mm']
            extra['world_from_camera_mm'] = (correction_matrix @ np.asarray(extra['world_from_camera_mm'])).tolist()
        extra['world_tilt_correction'] = {**correction, 'matrix_mm':correction_matrix.tolist()}
    np.save(
        directory / "world_xyz_mm.npy",
        np.asarray(xyz, dtype=np.float32),
        allow_pickle=False,
    )
    meta = {
        "schema_version": 1,
        "observation_id": uuid.uuid4().hex,
        "frame_id": "world_mm",
        "calibration_id": config.raw["calibration_id"],
        "captured_unix_s": time.time(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": synthetic,
        "joints_deg": {k: np.asarray(v).tolist() for k, v in joints.items()},
        "sha256": {n: sha(directory / n) for n in ("rgb.png", "world_xyz_mm.npy")},
        **(extra or {}),
    }
    if correction is not None:
        meta['sha256']['uncorrected_world_xyz_mm.npy'] = sha(directory/'uncorrected_world_xyz_mm.npy')
    (directory / "observation.json").write_text(
        json.dumps(meta, indent=2, allow_nan=False) + "\n"
    )
    return Observation.load(directory)


def create_camera(config):
    from ..perception import CameraSpec, RealSenseRGBD

    camera_config = config.raw["camera"]
    spec = CameraSpec(
        camera_config.get("label", "A"),
        str(camera_config["serial"]),
        Path("."),
        camera_config.get("color_exposure"),
        camera_config.get("color_white_balance"),
        depth_exposure_us=camera_config.get('depth_exposure_us'),
        depth_gain=camera_config.get('depth_gain'),
    )
    return RealSenseRGBD(
        spec, camera_config["width"], camera_config["height"], camera_config["fps"]
    )


def capture(directory, config, models, snapshot, *, camera=None):
    """Capture stationary frames; a supplied camera stays open for its owner."""

    camera_config = config.raw["camera"]
    owner = camera_config["mount_arm"]
    if owner not in {None, *config.arms}:
        raise DualArmError(
            "camera mount_arm must name an arm or be null for a fixed camera"
        )
    before = snapshot()
    owns_camera = camera is None
    if owns_camera:
        camera = create_camera(config)
    rgb_frames, depths = [], []
    stability = camera_config.get('depth_stability')
    stability_report = None

    def stationary():
        current = snapshot()
        for k in config.arms:
            if (
                np.max(np.abs(np.asarray(before[k]["joints"]) - current[k]["joints"]))
                > config.limits["stationary_tolerance_deg"]
            ):
                raise DualArmError(
                    "arm moved during RGB-D capture; observation discarded"
                )
        return current

    try:
        if owns_camera or not camera.started:
            camera.start()
        for _ in range(stability['warmup_frames'] if stability else 10):
            camera.read()
            stationary()
        stable_windows = 0
        for index in range(stability['max_frames'] if stability else 5):
            rgb, depth = camera.read()
            stationary()
            rgb_frames.append(rgb)
            depths.append(depth)
            if stability:
                from ..depth_stability import depth_window_report
                window = stability['window_frames']
                rgb_frames, depths = rgb_frames[-2*window:], depths[-2*window:]
                if len(depths) == 2*window and (index+1) % window == 0:
                    stability_report = depth_window_report(depths, stability['max_range_mm'])
                    stable_windows = stable_windows + 1 if stability_report['stable'] else 0
                    if stable_windows >= stability['stable_windows']:
                        stability_report.update(frames_checked=index+1, consecutive_windows=stable_windows)
                        rgb_frames, depths = rgb_frames[-window:], depths[-window:]
                        break
        if stability and stable_windows < stability['stable_windows']:
            raise DualArmError(f'depth did not stabilize: {stability_report}')
        intrinsics = camera.intrinsics.copy()
    finally:
        if owns_camera:
            camera.stop()
    after = stationary()
    if owner is None:
        world_camera = transform(
            camera_config["world_from_camera_mm"], "fixed camera calibration"
        )
    else:
        model = models[owner]
        world_camera = (
            config.arms[owner].world_from_base
            @ model.frame_at(before[owner]["joints"], camera_config["mount_link"])
            @ transform(camera_config["mount_from_camera_mm"], "hand-eye calibration")
        )
    samples = np.asarray(depths)
    if stability:
        valid = np.isfinite(samples) & (samples > .1) & (samples <= 2.5)
        depth = np.ma.median(np.ma.array(samples, mask=~valid), axis=0).filled(np.nan)
        depth[valid.sum(axis=0) < 3] = np.nan
    else:
        depth = np.median(samples, axis=0)
    h, w = depth.shape
    y, x = np.mgrid[:h, :w]
    camera_xyz = (
        np.stack(
            [
                (x - intrinsics[0, 2]) / intrinsics[0, 0] * depth,
                (y - intrinsics[1, 2]) / intrinsics[1, 1] * depth,
                depth,
            ],
            axis=-1,
        )
        * 1000
    )
    xyz = camera_xyz @ world_camera[:3, :3].T + world_camera[:3, 3]
    xyz[(depth <= 0.1) | (depth > 2.5) | ~np.isfinite(depth)] = np.nan
    return save_observation(
        directory,
        np.median(rgb_frames, axis=0),
        xyz,
        config,
        {k: v["joints"] for k, v in before.items()},
        extra={
            "world_from_camera_mm": world_camera.tolist(),
            "camera_serial": camera_config["serial"],
            "depth_capture_policy": {k: camera_config[k] for k in
                ('depth_exposure_us', 'depth_gain', 'depth_stability') if k in camera_config},
            "depth_stability": stability_report,
            "extrinsic_tilt_correction": camera_config.get("extrinsic_tilt_correction"),
            "snapshot_before": before,
            "snapshot_after": after,
        },
    )
