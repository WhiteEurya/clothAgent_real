from __future__ import annotations

import json
from pathlib import Path

import pytest

from cloth_agent.config import RobotConfig, SafetyError, WorkspaceBounds
from cloth_agent.experiment import _execute, validate_experiment_source
from cloth_agent.robot_api import RobotAPI, SimulatedBackend
from cloth_agent.shake_once import (
    FLICK_ACCELERATION_MM_S2,
    FLICK_SPEED_MM_S,
    PRELOAD_RETURN_SPEED_MM_S,
    build_shake_plan,
    execute_shake_plan,
    main,
    shake,
    validate_shake_plan_with_controller,
)


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
        init_pose_mm_deg=(500, -7, 300, 178, 3, 170),
        orientation_roll_deg=178,
        orientation_pitch_deg=3,
        expected_tcp_offset_mm_deg=(0, 0, 172, 0, 0, 0),
    )


def test_plan_centres_before_three_complete_y_flick_cycles() -> None:
    plan = build_shake_plan((520, 31, 420, 178, 3, 165), _config())

    assert [step.name for step in plan.steps] == [
        "center_y0",
        "preload_positive_y",
        "flick_1_negative_y",
        "flick_1_positive_y",
        "flick_2_negative_y",
        "flick_2_positive_y",
        "flick_3_negative_y",
        "flick_3_positive_y",
        "return_y0",
    ]
    assert [step.target_pose_mm_deg[1] for step in plan.steps] == [
        0,
        20,
        -20,
        20,
        -20,
        20,
        -20,
        20,
        0,
    ]
    assert all(step.target_pose_mm_deg[0] == 520 for step in plan.steps)
    assert all(step.target_pose_mm_deg[2] == 420 for step in plan.steps)
    assert all(step.target_pose_mm_deg[3:] == (178, 3, 165) for step in plan.steps)
    assert plan.steps[0].speed_mm_s == _config().speed_mm_s
    assert plan.steps[1].speed_mm_s == PRELOAD_RETURN_SPEED_MM_S
    assert plan.steps[2].speed_mm_s == FLICK_SPEED_MM_S
    assert plan.steps[2].acceleration_mm_s2 == FLICK_ACCELERATION_MM_S2


def test_plan_rejects_a_low_tcp_before_any_motion() -> None:
    with pytest.raises(SafetyError, match="already lifted"):
        build_shake_plan((500, 0, 249, 178, 3, 170), _config())


def test_plan_rejects_when_fixed_amplitude_exceeds_workspace() -> None:
    config = RobotConfig(
        **{
            **_config().__dict__,
            "boundaries": WorkspaceBounds(
                x_min=350,
                x_max=800,
                y_min=-8,
                y_max=8,
                z_min=6,
                z_max=500,
            ),
        }
    )
    with pytest.raises(SafetyError, match="safe upper bound"):
        build_shake_plan((500, 0, 300, 178, 3, 170), config)


class _FakeArm:
    connected = True
    tcp_offset = [0, 0, 172, 0, 0, 0]

    def __init__(self) -> None:
        self.pose = [500.0, 25.0, 400.0, 178.0, 3.0, 170.0]
        self.position_commands: list[dict[str, float]] = []
        self.state_commands: list[int] = []
        self.ik_poses: list[list[float]] = []

    def get_err_warn_code(self):
        return 0, [0, 14]

    def get_servo_angle(self, is_radian=False):
        return 0, [0, 10, 20, 30, 40, 50, 60]

    def get_inverse_kinematics(self, pose, **kwargs):
        self.ik_poses.append(list(pose))
        return 0, [0, 10, 20, 30, 40, 50, 60]

    def motion_enable(self, enable=True):
        return 0

    def set_mode(self, mode):
        return 0

    def set_state(self, state):
        self.state_commands.append(state)
        return 0

    def set_position(self, **kwargs):
        self.position_commands.append(dict(kwargs))
        self.pose = [
            kwargs["x"],
            kwargs["y"],
            kwargs["z"],
            kwargs["roll"],
            kwargs["pitch"],
            kwargs["yaw"],
        ]
        return 0

    def get_position(self, is_radian=False):
        return 0, list(self.pose)


def test_controller_validation_and_execution_use_only_the_fixed_y_moves() -> None:
    arm = _FakeArm()
    plan = build_shake_plan(arm.pose, _config())
    validation = validate_shake_plan_with_controller(arm, plan)
    records = execute_shake_plan(arm, plan)

    assert validation["controller_warning_code"] == 14
    assert validation["validated_sample_count"] == len(arm.ik_poses)
    assert [command["y"] for command in arm.position_commands] == [
        0,
        20,
        -20,
        20,
        -20,
        20,
        -20,
        20,
        0,
    ]
    assert arm.position_commands[2]["speed"] == FLICK_SPEED_MM_S
    assert arm.position_commands[2]["mvacc"] == FLICK_ACCELERATION_MM_S2
    assert all(command["x"] == 500 for command in arm.position_commands)
    assert all(command["z"] == 400 for command in arm.position_commands)
    assert [record["name"] for record in records] == [step.name for step in plan.steps]
    assert arm.state_commands == [0]


def test_shake_owns_live_preflight_and_fixed_execution() -> None:
    arm = _FakeArm()

    result = shake(arm, _config())

    assert result["plan"]["cycles"] == 3
    assert result["controller_validation"]["tcp_offset_mm_deg"] == [
        0,
        0,
        172,
        0,
        0,
        0,
    ]
    assert [command["y"] for command in arm.position_commands] == [
        0,
        20,
        -20,
        20,
        -20,
        20,
        -20,
        20,
        0,
    ]


def test_restricted_experiment_can_call_host_owned_shake() -> None:
    source = """def run():
    move(500, 0, 300, 0)
    shake()
"""
    validate_experiment_source(source)
    api = RobotAPI(_config(), SimulatedBackend(_config()))

    _, error = _execute(source, Path("experiment.py"), api)

    assert error is None
    assert [record.name for record in api.actions] == ["move", "shake"]
    assert api.actions[-1].actual_ee_pose is not None
    assert api.actions[-1].actual_ee_pose[1] == pytest.approx(0.0)


def test_dry_run_saves_a_plan_without_real_execution(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "plan.json"

    assert main(["--project-root", str(project_root), "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "DRY_RUN"
    assert payload["physical_commands_sent"] is False
    assert payload["gripper_commands"] == []
    assert [step["target_pose_mm_deg"][1] for step in payload["plan"]["steps"]] == [
        0,
        20,
        -20,
        20,
        -20,
        20,
        -20,
        20,
        0,
    ]
