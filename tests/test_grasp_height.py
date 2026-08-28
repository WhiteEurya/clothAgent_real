from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import pytest

from cloth_agent.config import RobotConfig, WorkspaceBounds
from cloth_agent.grasp_height import GraspHeightError, resolve_grasp_height


def _config() -> RobotConfig:
    return RobotConfig(
        robot_ip="127.0.0.1",
        boundaries=WorkspaceBounds(
            x_min=350,
            x_max=800,
            y_min=-300,
            y_max=170,
            z_min=6,
            z_max=500,
        ),
        init_joints_deg=(0, 0, 0, 0, 0, 0, 0),
        init_pose_mm_deg=(500, 0, 280, 180, 0, 0),
        orientation_roll_deg=180,
        orientation_pitch_deg=0,
    )


def _measurement(*, surface_z: float = 30.0, table_z: float = 5.0) -> dict:
    return {
        "valid": True,
        "base_xyz_median_mm": [500.0, 40.0, surface_z],
        "base_z_p90_minus_p10_mm": 1.2,
        "table_z_median_mm": table_z,
    }


def test_shared_grasp_height_uses_one_configured_surface_compression() -> None:
    resolution = resolve_grasp_height(
        measurement=_measurement(),
        table_plane_abc=[0.0, 0.0, 5.0],
        robot_config=_config(),
    )

    assert resolution.target_xyz_mm == pytest.approx((500.0, 40.0, 27.0))
    assert resolution.desired_compression_mm == pytest.approx(3.0)
    assert resolution.achieved_compression_mm == pytest.approx(3.0)


def test_shared_grasp_height_uses_conservative_local_or_plane_table_z() -> None:
    resolution = resolve_grasp_height(
        measurement=_measurement(surface_z=14.5, table_z=8.5),
        table_plane_abc=[0.0, 0.0, 5.0],
        robot_config=_config(),
    )

    assert resolution.authoritative_table_z_mm == pytest.approx(8.5)
    assert resolution.table_z_disagreement_mm == pytest.approx(3.5)
    assert resolution.target_xyz_mm[2] == pytest.approx(11.5)
    assert resolution.achieved_compression_mm == pytest.approx(3.0)


def test_shared_grasp_height_honors_a_deeper_diagnostic_without_exceeding_cap() -> None:
    measurement = _measurement()
    measurement["surface_shape_diagnostic"] = {
        "recommended_press_below_surface_mm": 4.0,
    }
    resolution = resolve_grasp_height(
        measurement=measurement,
        table_plane_abc=[0.0, 0.0, 5.0],
        robot_config=_config(),
    )

    assert resolution.desired_compression_mm == pytest.approx(3.0)
    assert resolution.target_xyz_mm[2] == pytest.approx(27.0)


def test_shared_grasp_height_rejects_a_target_that_cannot_engage_surface() -> None:
    config = replace(_config(), grasp_min_compression_mm=0.75)

    with pytest.raises(GraspHeightError, match="no legal engaged grasp Z"):
        resolve_grasp_height(
            measurement=_measurement(surface_z=6.5, table_z=1.0),
            table_plane_abc=[0.0, 0.0, 1.0],
            robot_config=config,
        )


def test_absolute_camera_mode_ignores_table_clearance_floor() -> None:
    config = replace(
        _config(),
        boundaries=WorkspaceBounds(
            x_min=350,
            x_max=800,
            y_min=-300,
            y_max=170,
            z_min=0,
            z_max=500,
        ),
        grasp_use_table_clearance_floor=False,
        grasp_table_clearance_mm=5.0,
    )
    resolution = resolve_grasp_height(
        measurement=_measurement(surface_z=8.0, table_z=7.0),
        table_plane_abc=[0.0, 0.0, 7.0],
        robot_config=config,
    )

    assert resolution.target_xyz_mm[2] == pytest.approx(5.0)
    assert resolution.achieved_compression_mm == pytest.approx(3.0)
    assert resolution.lower_z_mm == pytest.approx(0.0)
    assert resolution.policy == (
        "runtime_authoritative_absolute_camera_surface_no_table_floor"
    )
    assert math.isnan(resolution.table_clearance_lower_z_mm)


def test_robot_config_loads_the_shared_grasp_height_policy() -> None:
    root = Path(__file__).resolve().parents[1]
    config = RobotConfig.load(root, root / "config" / "robot.example.json")

    assert config.grasp_surface_compression_mm == pytest.approx(3.0)
    assert config.grasp_min_compression_mm == pytest.approx(0.75)
    assert config.grasp_max_compression_mm == pytest.approx(3.0)
    assert config.grasp_table_clearance_mm == pytest.approx(0.0)
    assert config.online_camera_z_bias_correction is False
    assert config.grasp_use_table_clearance_floor is False
