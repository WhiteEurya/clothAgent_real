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


def test_sponge_support_ring_enables_local_deeper_press() -> None:
    config = replace(
        _config(),
        support_layer_type="sponge",
        support_layer_thickness_mm=20.0,
        support_layer_press_mm=6.0,
        support_layer_max_compression_mm=8.0,
        support_layer_hard_clearance_mm=1.0,
        support_layer_presence_threshold_mm=3.0,
    )
    measurement = _measurement(surface_z=25.0, table_z=5.0)
    measurement.update(
        {
            "local_support_ring_valid": True,
            "local_support_z_median_mm": 20.0,
            "local_support_ring_elevation_mm": 18.0,
        }
    )
    resolution = resolve_grasp_height(
        measurement=measurement,
        table_plane_abc=[0.0, 0.0, 5.0],
        robot_config=config,
    )

    assert resolution.support_layer_active is True
    assert resolution.desired_compression_mm == pytest.approx(6.0)
    assert resolution.maximum_compression_mm == pytest.approx(8.0)
    assert resolution.support_floor_z_mm == pytest.approx(1.0)
    assert resolution.target_xyz_mm[2] == pytest.approx(19.0)
    assert resolution.policy == (
        "runtime_authoritative_surface_compression_with_local_sponge_support"
    )


def test_confirmed_sponge_replaces_global_table_floor() -> None:
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
        grasp_use_table_clearance_floor=True,
        support_layer_type="sponge",
        support_layer_thickness_mm=20.0,
        support_layer_press_mm=6.0,
        support_layer_max_compression_mm=8.0,
        support_layer_hard_clearance_mm=1.0,
        support_layer_presence_threshold_mm=3.0,
    )
    measurement = _measurement(surface_z=7.0, table_z=5.0)
    measurement.update(
        {
            "local_support_ring_valid": True,
            "local_support_z_median_mm": 20.0,
            "local_support_ring_elevation_mm": 15.0,
        }
    )
    resolution = resolve_grasp_height(
        measurement=measurement,
        table_plane_abc=[0.0, 0.0, 5.0],
        robot_config=config,
    )

    assert resolution.support_layer_active is True
    assert resolution.lower_z_mm == pytest.approx(1.0)
    assert resolution.target_xyz_mm[2] == pytest.approx(1.0)


def test_sponge_support_ring_below_presence_threshold_keeps_legacy_depth() -> None:
    config = replace(
        _config(),
        support_layer_type="sponge",
        support_layer_thickness_mm=20.0,
        support_layer_press_mm=6.0,
        support_layer_max_compression_mm=8.0,
        support_layer_hard_clearance_mm=1.0,
        support_layer_presence_threshold_mm=3.0,
    )
    measurement = _measurement(surface_z=25.0, table_z=5.0)
    measurement.update(
        {
            "local_support_ring_valid": True,
            "local_support_z_median_mm": 5.0,
            "local_support_ring_elevation_mm": 0.5,
        }
    )
    resolution = resolve_grasp_height(
        measurement=measurement,
        table_plane_abc=[0.0, 0.0, 5.0],
        robot_config=config,
    )

    assert resolution.support_layer_active is False
    assert resolution.desired_compression_mm == pytest.approx(3.0)
    assert resolution.maximum_compression_mm == pytest.approx(3.0)
    assert resolution.target_xyz_mm[2] == pytest.approx(22.0)


def test_confirmed_sponge_enables_deeper_press_without_ring_elevation() -> None:
    config = replace(
        _config(),
        support_layer_type="sponge",
        support_layer_confirmed=True,
        support_layer_thickness_mm=20.0,
        support_layer_press_mm=6.0,
        support_layer_max_compression_mm=8.0,
        support_layer_hard_clearance_mm=1.0,
        support_layer_presence_threshold_mm=3.0,
    )
    measurement = _measurement(surface_z=25.0, table_z=5.0)
    measurement.update(
        {
            "local_support_ring_valid": True,
            "local_support_z_median_mm": 5.0,
            "local_support_ring_elevation_mm": 0.5,
        }
    )
    resolution = resolve_grasp_height(
        measurement=measurement,
        table_plane_abc=[0.0, 0.0, 5.0],
        robot_config=config,
    )

    assert resolution.support_layer_active is True
    assert resolution.support_layer_confirmed is True
    assert resolution.support_layer_activation_source == "declared_configuration"
    assert resolution.desired_compression_mm == pytest.approx(6.0)
    assert resolution.maximum_compression_mm == pytest.approx(8.0)
    assert resolution.target_xyz_mm[2] == pytest.approx(19.0)


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

    assert config.grasp_surface_compression_mm == pytest.approx(20.0)
    assert config.grasp_min_compression_mm == pytest.approx(0.75)
    assert config.grasp_max_compression_mm == pytest.approx(20.0)
    assert config.grasp_table_clearance_mm == pytest.approx(0.0)
    assert config.online_camera_z_bias_correction is False
    assert config.grasp_use_table_clearance_floor is False
    assert config.support_layer_type == "sponge"
    assert config.support_layer_confirmed is True
    assert config.support_layer_thickness_mm == pytest.approx(20.0)
    assert config.support_layer_press_mm == pytest.approx(20.0)
    assert config.support_layer_max_compression_mm == pytest.approx(20.0)
    assert config.support_layer_hard_clearance_mm == pytest.approx(1.0)
    assert config.gripper_width_mm == pytest.approx(86.0)


@pytest.mark.parametrize('descent', [1., 2.5])
def test_model_descent_overrides_configured_default(descent):
    value = resolve_grasp_height(measurement=_measurement(), table_plane_abc=None,
        robot_config=_config(), proposed_descent_mm=descent)
    assert value.target_xyz_mm[2] == 30 - descent


@pytest.mark.parametrize('descent', [0., 100., float('nan'), True])
def test_model_descent_rejected_without_clamping(descent):
    with pytest.raises(GraspHeightError):
        resolve_grasp_height(measurement=_measurement(), table_plane_abc=None,
            robot_config=_config(), proposed_descent_mm=descent)
