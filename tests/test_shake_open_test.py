from __future__ import annotations

import json
from pathlib import Path

import pytest

from cloth_agent.config import RobotConfig, SafetyError, WorkspaceBounds
from cloth_agent.experiment import _execute, validate_experiment_source
from cloth_agent.robot_api import (
    RobotAPI,
    SimulatedBackend,
    _controller_trajectory_with_arm,
)
from cloth_agent.shake_open_test import (
    DIAGONAL_CYCLES,
    DIAGONAL_SCALE_CANDIDATES,
    DIAGONAL_X_MM,
    DIAGONAL_Y_MM,
    DIAGONAL_Z_MM,
    FAST_RISE_SPEED_MM_S,
    TARGET_TEST_HIGH_Z_MM,
    VERTICAL_SNAP_CYCLES,
    WORK_Z_DROP_MM,
    build_shake_open_plan,
    execute_shake_open,
    main,
    select_controller_valid_shake_open_plan,
    validate_shake_open_with_controller,
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
        init_pose_mm_deg=(500, 0, 280, 178, 3, 170),
        orientation_roll_deg=178,
        orientation_pitch_deg=3,
        expected_tcp_offset_mm_deg=(0, 0, 172, 0, 0, 0),
    )


def test_plan_has_vertical_snaps_then_two_diagonal_cycles() -> None:
    plan = build_shake_open_plan((500, 25, 495, 178, 3, 170), _config())

    assert plan.vertical_snap_cycles == VERTICAL_SNAP_CYCLES == 2
    assert plan.diagonal_cycles == DIAGONAL_CYCLES == 2
    assert plan.diagonal_scale == pytest.approx(1.0)
    assert plan.work_pose_mm_deg[:3] == (500, 0, 495 - WORK_Z_DROP_MM)
    assert [step.name for step in plan.steps[:7]] == [
        "center_y0",
        "move_to_test_high",
        "move_to_shake_height",
        "vertical_1_slow_drop",
        "vertical_1_fast_rise",
        "vertical_2_slow_drop",
        "vertical_2_fast_rise",
    ]
    assert plan.steps[4].speed_mm_s == FAST_RISE_SPEED_MM_S
    assert plan.steps[3].wait is False
    assert plan.steps[3].blend_radius_mm is not None
    assert plan.steps[-2].name == "return_center"
    assert plan.steps[-2].wait is False
    assert plan.steps[-1].name == "return_test_high"
    assert plan.steps[-1].wait is True
    assert plan.steps[-1].blend_radius_mm is None

    diagonal = [step for step in plan.steps if step.name.startswith("diagonal_")]
    assert len(diagonal) == DIAGONAL_CYCLES * 4
    offsets = {
        (
            step.target_pose_mm_deg[0] - 500,
            step.target_pose_mm_deg[1],
            step.target_pose_mm_deg[2] - plan.work_pose_mm_deg[2],
        )
        for step in diagonal
    }
    assert offsets == {
        (-DIAGONAL_X_MM, DIAGONAL_Y_MM, -DIAGONAL_Z_MM),
        (DIAGONAL_X_MM, -DIAGONAL_Y_MM, DIAGONAL_Z_MM),
        (-DIAGONAL_X_MM, -DIAGONAL_Y_MM, -DIAGONAL_Z_MM),
        (DIAGONAL_X_MM, DIAGONAL_Y_MM, DIAGONAL_Z_MM),
    }


def test_plan_automatically_lifts_from_home_height() -> None:
    plan = build_shake_open_plan((500, 0, 280, 178, 3, 170), _config())

    assert plan.steps[1].name == "move_to_test_high"
    assert plan.steps[1].target_pose_mm_deg[2] == TARGET_TEST_HIGH_Z_MM
    assert plan.work_pose_mm_deg[2] == TARGET_TEST_HIGH_Z_MM - WORK_Z_DROP_MM


def test_plan_rejects_an_entry_pose_too_close_to_the_table() -> None:
    with pytest.raises(SafetyError, match="entry z>=250"):
        build_shake_open_plan((500, 0, 249, 178, 3, 170), _config())


class _FakeArm:
    connected = True
    tcp_offset = [0, 0, 172, 0, 0, 0]

    def __init__(self) -> None:
        self.pose = [500.0, 25.0, 495.0, 178.0, 3.0, 170.0]
        self.commands: list[dict[str, object]] = []
        self.state_commands: list[int] = []
        self.ik_calls = 0

    def get_position(self, is_radian=False):
        return 0, list(self.pose)

    def get_servo_angle(self, is_radian=False):
        return 0, [0, 10, 20, 30, 40, 50, 60]

    def get_err_warn_code(self):
        return 0, [0, 14]

    def get_inverse_kinematics(self, pose, **kwargs):
        self.ik_calls += 1
        return 0, [0, 10, 20, 30, 40, 50, 60]

    def get_forward_kinematics(self, joints, **kwargs):
        return 0, [500, 0, 280, 178, 3, 170]

    def motion_enable(self, enable=True):
        return 0

    def set_mode(self, mode):
        return 0

    def set_state(self, state):
        self.state_commands.append(state)
        return 0

    def set_position(self, **kwargs):
        self.commands.append(dict(kwargs))
        self.pose = [
            kwargs["x"],
            kwargs["y"],
            kwargs["z"],
            kwargs["roll"],
            kwargs["pitch"],
            kwargs["yaw"],
        ]
        return 0


class _ScaleRejectingArm(_FakeArm):
    def get_inverse_kinematics(self, pose, **kwargs):
        self.ik_calls += 1
        if float(pose[0]) < 490.0 and float(pose[1]) < -12.0:
            return 10, []
        return 0, [0, 10, 20, 30, 40, 50, 60]


def test_controller_validation_and_execution_queue_blended_motion() -> None:
    arm = _FakeArm()
    plan = build_shake_open_plan(arm.pose, _config())

    validation = validate_shake_open_with_controller(arm, plan)
    records = execute_shake_open(arm, plan)

    assert validation["controller_warning_code"] == 14
    assert validation["validated_sample_count"] == arm.ik_calls
    assert len(arm.commands) == len(plan.steps)
    assert all(command["wait"] is False for command in arm.commands[3:-1])
    assert all(command["radius"] is not None for command in arm.commands[3:-1])
    assert arm.commands[-1]["wait"] is True
    assert arm.commands[-1]["radius"] is None
    assert records[-1]["final_position_error_mm"] == pytest.approx(0.0)
    assert arm.state_commands == [0]


def test_shake_open_automatically_reduces_only_diagonal_amplitude_on_ik_failure() -> None:
    arm = _ScaleRejectingArm()

    plan, validation, trials = select_controller_valid_shake_open_plan(
        arm,
        arm.pose,
        _config(),
    )

    assert DIAGONAL_SCALE_CANDIDATES == (1.0, 0.75, 0.5, 0.25)
    assert plan.diagonal_scale == pytest.approx(0.5)
    assert validation["selected_diagonal_scale"] == pytest.approx(0.5)
    assert [trial["status"] for trial in trials] == [
        "IK_REJECTED",
        "IK_REJECTED",
        "IK_ACCEPTED",
    ]


def test_dry_run_saves_composite_plan_without_robot_connection(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "shake_open.json"

    assert main(["--project-root", str(project_root), "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "DRY_RUN"
    assert payload["physical_commands_sent"] is False
    assert payload["gripper_commands"] == []
    assert payload["home_commanded"] is False
    assert payload["plan"]["vertical_snap_cycles"] == 2
    assert payload["plan"]["diagonal_cycles"] == 2


def test_restricted_experiment_can_call_shake_open_separately_from_shake() -> None:
    source = """def run():
    shake_open()
"""
    validate_experiment_source(source)
    api = RobotAPI(_config(), SimulatedBackend(_config()))

    _, error = _execute(source, Path("experiment.py"), api)

    assert error is None
    assert [record.name for record in api.actions] == ["shake_open"]
    assert api.actions[0].actual_ee_pose is not None
    assert api.actions[0].actual_ee_pose[1] == pytest.approx(0.0)
    assert api.actions[0].actual_ee_pose[2] == pytest.approx(TARGET_TEST_HIGH_Z_MM)
    assert api.actions[0].robot_state["shake_open"]["diagonal_cycles"] == 2


def test_controller_preflight_expands_all_shake_open_substeps() -> None:
    arm = _FakeArm()

    validation = _controller_trajectory_with_arm(
        arm,
        _config(),
        [{"name": "shake_open", "args": {}}],
    )

    assert 0 in validation.joint_targets_rad
    assert validation.validated_sample_count == arm.ik_calls
    assert validation.validated_sample_count > 20


def test_controller_preflight_records_adaptive_shake_open_scale() -> None:
    arm = _ScaleRejectingArm()

    validation = _controller_trajectory_with_arm(
        arm,
        _config(),
        [{"name": "shake_open", "args": {}}],
    )

    assert validation.shake_open_diagonal_scales[0] == pytest.approx(0.5)
