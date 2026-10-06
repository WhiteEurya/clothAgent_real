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

    def validate_for(self, config, *, live=False):
        if self.meta["calibration_id"] != config.raw["calibration_id"]:
            raise DualArmError("observation uses a different calibration")
        if live:
            if self.meta.get("synthetic") is not False:
                raise DualArmError(
                    "real execution requires a real captured observation"
                )
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
    (directory / "observation.json").write_text(
        json.dumps(meta, indent=2, allow_nan=False) + "\n"
    )
    return Observation.load(directory)


def capture(directory, config, models, snapshot):
    """Both arms must remain still for the entire temporal capture window."""
    from ..perception import CameraSpec, RealSenseRGBD

    camera_config = config.raw["camera"]
    owner = camera_config["mount_arm"]
    if owner not in {None, *config.arms}:
        raise DualArmError(
            "camera mount_arm must name an arm or be null for a fixed camera"
        )
    before = snapshot()
    spec = CameraSpec(
        "A",
        str(camera_config["serial"]),
        Path("."),
        camera_config.get("color_exposure"),
        camera_config.get("color_white_balance"),
    )
    camera = RealSenseRGBD(
        spec, camera_config["width"], camera_config["height"], camera_config["fps"]
    )
    rgb_frames, depths = [], []

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
        camera.start()
        for _ in range(10):
            camera.read()
            stationary()
        for _ in range(5):
            rgb, depth = camera.read()
            stationary()
            rgb_frames.append(rgb)
            depths.append(depth)
        intrinsics = camera.intrinsics.copy()
    finally:
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
    depth = np.median(np.asarray(depths), axis=0)
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
            "snapshot_before": before,
            "snapshot_after": after,
        },
    )
