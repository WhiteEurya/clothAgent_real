from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cloth_agent.config import ConfigError, ExperimentConfig, RobotConfig, SafetyError, WorkspaceBounds
from cloth_agent.claude import ClaudeCodeClient
from cloth_agent.experiment import (
    ExperimentRunner,
    ExperimentValidationError,
    format_speed_profile,
    validate_experiment_source,
)
from cloth_agent.perception import (
    CameraSpec,
    ClothCenterPerception,
    MolmoConfig,
    PerceptionConfig,
    PerceptionError,
    RGBDFrame,
    derive_grasp_plan,
    pixel_to_base_mm,
    points_by_image,
    robust_depth_at_pixel,
    _scalar_heatmap_rgb,
    _fold_edge_mask,
    _fit_table_plane_from_references,
    _height_display_max_mm,
    _save_fused_height_map,
    DEFAULT_HEIGHT_MAP_DISPLAY_MIN_MM,
    DEFAULT_HEIGHT_MAP_DISPLAY_MAX_MM,
    _mask_boundary,
    _occlusion_aware_garment_mask,
    _save_camera_height_heatmap,
    _estimate_camera_table_z_offset_mm,
    _signed_height_display_range_mm,
    camera_height_map_mm,
    table_clip_tolerance_mm,
    table_height_and_clip_mask,
)
from cloth_agent.robot_api import (
    RobotAPI,
    SimulatedBackend,
    _controller_trajectory_with_arm,
    _validated_live_tcp_offset,
    move_robot_to_perception_position,
)
from cloth_agent.randomization import (
    build_garment_randomization_plan,
    garment_randomization_source,
)
from cloth_agent.session import AgentSession
from cloth_agent.viewer import (
    _ensure_experiment,
    _ensure_experiment_source,
    canonical_grasp_source,
    canonical_home_source,
    _frame_point_cloud,
    _load_fused_point_cloud,
    _view_point_cloud,
    path_waypoints_mm,
    run_viewer,
)
from cloth_agent.kinematics import XArm7Kinematics


def robot_config() -> RobotConfig:
    return RobotConfig(
        robot_ip="127.0.0.1",
        boundaries=WorkspaceBounds(x_min=0, x_max=800, y_min=-400, y_max=400, z_min=10, z_max=400),
        init_joints_deg=(0, 0, 0, 0, 0, 0, 0),
        init_pose_mm_deg=(500, 0, 180, 180, 0, 0),
        orientation_roll_deg=180,
        orientation_pitch_deg=0,
        workspace_margin_mm=1,
        speed_mm_s=15,
        acceleration_mm_s2=30,
        home_speed_deg_s=5,
        home_acceleration_deg_s2=10,
    )


def write_experiment(workspace: Path, name: str, source: str) -> Path:
    path = workspace / name
    path.write_text(source, encoding="utf-8")
    return path


def test_restricted_script_and_sequence(tmp_path: Path) -> None:
    config = robot_config()
    api = RobotAPI(config, SimulatedBackend(config))
    source = """def run():
    home()
    open_gripper()
    move(500, -20, 100, 5)
    move(500, -20, 40, 5)
    close_gripper()
    move(500, -20, 200, 5)
    open_gripper()
    move(500, -20, 100, 5)
    home()
"""
    validate_experiment_source(source)
    compile(source, "experiment.py", "exec")
    from cloth_agent.experiment import _execute

    output, error = _execute(source, Path("experiment.py"), api)
    assert not output
    assert error is None
    assert [item.name for item in api.actions] == [
        "home", "open_gripper", "move", "move", "close_gripper", "move", "open_gripper", "move", "home"
    ]


def test_zero_relative_yaw_preserves_calibrated_home_gripper_orientation() -> None:
    config = RobotConfig(
        **{**robot_config().__dict__, "init_pose_mm_deg": (500, 0, 180, 180, 0, 170.569135)}
    )
    api = RobotAPI(config, SimulatedBackend(config))
    api.move(500, 0, 100, 0)

    assert api.actions[0].actual_ee_pose is not None
    assert api.actions[0].actual_ee_pose[5] == pytest.approx(170.569135)
    assert config.command_yaw_deg(15) == pytest.approx(-174.430865)


def test_viewer_canonical_source_and_waypoints() -> None:
    config = robot_config()
    experiment = ExperimentConfig(500, -20, 40, 100, 200, 5)
    source = canonical_grasp_source(experiment)
    validate_experiment_source(source)
    assert "move(500, -20, 40, 5)" in source
    runner_api = RobotAPI(config, SimulatedBackend(config))
    from cloth_agent.experiment import _execute

    _execute(source, Path("viewer.py"), runner_api)
    waypoints = path_waypoints_mm(runner_api.action_dicts(), config.init_pose_mm_deg)
    assert [name for name, _ in waypoints] == [
        "home", "move_1", "move_2", "move_3", "move_4", "home"
    ]
    assert waypoints[2][1].tolist() == pytest.approx([500, -20, 40])


def test_viewer_init_source_contains_only_home() -> None:
    source = canonical_home_source()
    validate_experiment_source(source)
    api = RobotAPI(robot_config(), SimulatedBackend(robot_config()))
    from cloth_agent.experiment import _execute

    output, error = _execute(source, Path("_viser_init_home.py"), api)
    assert output == ""
    assert error is None
    assert [action.name for action in api.actions] == ["home"]


def test_viewer_real_execution_must_stay_loopback_only() -> None:
    with pytest.raises(PermissionError, match="loopback-only"):
        run_viewer(None, host="0.0.0.0", enable_real=True)  # type: ignore[arg-type]


def test_viewer_never_reuses_script_stale_against_latest_plan(tmp_path: Path) -> None:
    experiment = ExperimentConfig(500, 0, 40, 100, 300, 0)
    session = AgentSession.create(
        tmp_path, "sync plan", robot_config(), experiment, run_id="sync_plan"
    )
    stale = write_experiment(
        session.workspace,
        "experiment_001.py",
        canonical_grasp_source(ExperimentConfig(450, 20, 40, 100, 160, 0)),
    )
    selected = _ensure_experiment(session, stale.name)
    assert selected == "experiment_002.py"
    assert (session.workspace / selected).read_text() == canonical_grasp_source(experiment)


def test_garment_randomization_plan_is_deterministic_and_previewable(tmp_path: Path) -> None:
    experiment = ExperimentConfig(500, -20, 20, 100, 180, 5)
    first = build_garment_randomization_plan(experiment, robot_config(), seed=1234)
    second = build_garment_randomization_plan(experiment, robot_config(), seed=1234)
    different = build_garment_randomization_plan(experiment, robot_config(), seed=1235)
    assert first == second
    assert first != different
    assert first.drag_x_mm < first.center_x_mm
    assert abs(first.twist_yaw_deg - first.base_yaw_deg) == pytest.approx(90)
    assert first.grasp_z_mm < first.release_z_mm < first.gather_lift_z_mm

    source = garment_randomization_source(first)
    validate_experiment_source(source)
    session = AgentSession.create(
        tmp_path, "randomize garment", robot_config(), experiment, run_id="randomize"
    )
    selected = _ensure_experiment_source(session, None, source)
    assert (session.workspace / selected).read_text() == source
    preflight = session.runner.preflight(selected)
    assert preflight.error is None
    assert [action["name"] for action in preflight.actions] == [
        "home",
        "open_gripper",
        "move",
        "move",
        "close_gripper",
        "move",
        "move",
        "move",
        "move",
        "open_gripper",
        "move",
        "home",
    ]
    waypoints = path_waypoints_mm(preflight.actions, robot_config().init_pose_mm_deg)
    assert len(waypoints) == 9
    assert waypoints[4][1].tolist() == pytest.approx(
        [first.drag_x_mm, first.drag_y_mm, first.gather_lift_z_mm]
    )


@pytest.mark.parametrize(
    "source",
    [
        "import xarm\ndef run():\n    home()\n",
        "def run():\n    robot.arm.set_position(1)\n",
        "def run():\n    __import__('os')\n",
        "def run():\n    while True:\n        home()\n",
        "def run() -> move.__self__:\n    home()\n",
        "def run():\n    pass\n",
    ],
)
def test_generated_code_hard_failures_are_rejected(source: str) -> None:
    with pytest.raises(ExperimentValidationError):
        validate_experiment_source(source)


def test_bounds_and_failure_stop(tmp_path: Path) -> None:
    config = robot_config()
    api = RobotAPI(config, SimulatedBackend(config))
    with pytest.raises(SafetyError):
        api.move(500, 0, 5, 0)
    assert api.halted is True
    with pytest.raises(Exception):
        api.home()


def test_real_run_requires_complete_bounds() -> None:
    config = robot_config()
    incomplete = RobotConfig(
        **{
            **config.__dict__,
            "boundaries": WorkspaceBounds(x_min=0, y_min=-400, z_min=10, z_max=400),
        }
    )
    with pytest.raises(ConfigError, match="y_max"):
        incomplete.validate_for_real()


def test_real_run_allows_sdk_managed_x_max() -> None:
    config = robot_config()
    without_x_max = RobotConfig(
        **{
            **config.__dict__,
            "boundaries": WorkspaceBounds(x_min=0, y_min=-400, y_max=400, z_min=10, z_max=400),
        }
    )
    without_x_max.validate_for_real()


def test_center_can_be_deferred_until_perception() -> None:
    deferred = ExperimentConfig.from_mapping({}, allow_deferred=True)
    assert deferred.cloth_center_x is None
    assert deferred.grasp_z is None
    with pytest.raises(ConfigError, match="plan is incomplete"):
        deferred.require_ready()


def test_live_tcp_offset_is_one_time_hardware_guard() -> None:
    config = robot_config()
    config.validate_live_tcp_offset([0, 0, 172, 0, 0, 0])
    with pytest.raises(ConfigError, match="TCP offset changed"):
        config.validate_live_tcp_offset([0, 0, 160, 0, 0, 0])


def test_recorded_perception_position_loads_from_default_robot_config() -> None:
    root = Path(__file__).resolve().parents[1]
    config = RobotConfig.load(root, root / "config" / "robot.example.json")
    assert config.perception_joints_deg == pytest.approx(
        (
            13.596346,
            -6.945108,
            -11.614542,
            128.426026,
            1.351092,
            70.942374,
            187.543296,
        )
    )
    assert config.perception_pose_mm_deg == pytest.approx(
        (478.214813, 9.459489, 812.595703, 159.939106, 62.538974, 161.782139)
    )


def test_perception_position_transition_is_home_then_recorded_joint_pose() -> None:
    config = RobotConfig(
        **{
            **robot_config().__dict__,
            "perception_joints_deg": (1, 2, 3, 4, 5, 6, 7),
            "perception_pose_mm_deg": (480, 10, 810, 160, 63, 162),
        }
    )

    class RecordingBackend(SimulatedBackend):
        def __init__(self, robot: RobotConfig):
            super().__init__(robot)
            self.calls: list[str] = []

        def home(self, robot: RobotConfig):
            self.calls.append("home")
            return super().home(robot)

        def perception_position(self, robot: RobotConfig):
            self.calls.append("perception_position")
            return super().perception_position(robot)

        def close(self):
            self.calls.append("close")

    backend = RecordingBackend(config)
    outcome = move_robot_to_perception_position(config, backend=backend)
    assert backend.calls == ["home", "perception_position", "close"]
    assert outcome["sequence"] == ["home", "perception_position"]
    assert outcome["target_joint_angles_deg"] == [1, 2, 3, 4, 5, 6, 7]
    assert outcome["actual_tcp_pose_mm_deg"] == [480, 10, 810, 160, 63, 162]


class FakeReadOnlyArm:
    tcp_offset = [0, 0, 172, 0, 0, 0]

    def __init__(self, ik_code: int = 0):
        self.ik_code = ik_code
        self.ik_calls = 0
        self.fk_calls = 0
        self.tcp_limit_calls = 0
        self.ik_poses: list[list[float]] = []
        self.ik_kwargs: list[dict] = []

    def get_err_warn_code(self):
        return 0, [0, 14]

    def is_tcp_limit(self, pose, is_radian=False):
        self.tcp_limit_calls += 1
        return 0, True

    def get_forward_kinematics(self, angles, **kwargs):
        self.fk_calls += 1
        return 0, [500, 0, 180, 180, 0, 123]

    def get_inverse_kinematics(self, pose, **kwargs):
        self.ik_calls += 1
        self.ik_poses.append(list(pose))
        self.ik_kwargs.append(kwargs)
        return self.ik_code, [0, 10, 20, 30, 40, 50, 60] if self.ik_code == 0 else []


def test_controller_ik_is_a_hard_gate_before_real_motion() -> None:
    actions = [
        {"name": "home", "args": {}},
        {"name": "move", "args": {"x": 500, "y": 0, "z": 100, "yaw": 0}},
    ]
    arm = FakeReadOnlyArm()
    validated = _controller_trajectory_with_arm(arm, robot_config(), actions)
    assert set(validated.joint_targets_rad) == {0, 1}
    assert validated.controller_warning_code == 14
    assert validated.validated_sample_count >= 1
    assert arm.fk_calls == 1
    assert arm.tcp_limit_calls == 0
    assert arm.ik_poses[0][5] == pytest.approx(123 - 123 / 13)
    with pytest.raises(SafetyError, match="IK rejected"):
        _controller_trajectory_with_arm(FakeReadOnlyArm(ik_code=10), robot_config(), actions)


def test_controller_ik_uses_home_yaw_for_zero_relative_yaw() -> None:
    config = RobotConfig(
        **{**robot_config().__dict__, "init_pose_mm_deg": (500, 0, 180, 180, 0, 123)}
    )
    actions = [{"name": "move", "args": {"x": 500, "y": 0, "z": 100, "yaw": 0}}]
    arm = FakeReadOnlyArm()
    _controller_trajectory_with_arm(arm, config, actions)

    assert arm.ik_poses[0][5] == pytest.approx(123)


def test_controller_ik_deduplicates_repeated_cartesian_targets() -> None:
    arm = FakeReadOnlyArm()
    move = {"name": "move", "args": {"x": 500, "y": 0, "z": 100, "yaw": 0}}
    validated = _controller_trajectory_with_arm(arm, robot_config(), [move, move])
    assert set(validated.joint_targets_rad) == {0, 1}
    assert arm.ik_calls == validated.validated_sample_count
    assert arm.ik_calls >= 1
    assert all(call["limited"] is True for call in arm.ik_kwargs)


def test_controller_ik_rejects_an_intermediate_cartesian_sample() -> None:
    class IntermediateFailureArm(FakeReadOnlyArm):
        def get_inverse_kinematics(self, pose, **kwargs):
            self.ik_calls += 1
            self.ik_kwargs.append(kwargs)
            if self.ik_calls == 2:
                return 10, []
            return 0, [0, 10, 20, 30, 40, 50, 60]

    move = {"name": "move", "args": {"x": 500, "y": 0, "z": 100, "yaw": 0}}
    with pytest.raises(SafetyError, match="segment sample 2/"):
        _controller_trajectory_with_arm(
            IntermediateFailureArm(), robot_config(), [move]
        )


def test_tcp_validation_waits_for_initial_controller_report() -> None:
    class DeferredTcpArm:
        tcp_offset = [0, 0, 0, 0, 0, 0]

        def get_position(self, is_radian=False):
            self.tcp_offset = [0, 0, 172, 0, 0, 0]
            return 0, [500, 0, 100, 180, 0, 0]

    actual = _validated_live_tcp_offset(DeferredTcpArm(), robot_config())
    assert actual == pytest.approx((0, 0, 172, 0, 0, 0))


def test_slow_motion_profile_and_caps() -> None:
    config = robot_config()
    text = format_speed_profile(config)
    assert "15.0 mm/s" in text
    assert "5.0 deg/s" in text
    too_fast = RobotConfig(**{**config.__dict__, "speed_mm_s": 31})
    with pytest.raises(ConfigError, match="motion speed"):
        too_fast.validate_for_real()


def test_measured_z_min_can_be_used_without_removing_other_margins() -> None:
    config = robot_config()
    config.boundaries.validate(
        500,
        0,
        config.boundaries.z_min,
        config.workspace_margin_mm,
        z_lower_margin_mm=config.lower_z_margin_mm,
    )


def test_urdf_gripper_open_close_mapping() -> None:
    root = Path(__file__).resolve().parents[1]
    kinematics = XArm7Kinematics(root / "assets/robots/xarm7/xarm7.urdf")
    frames = kinematics.build_animation(
        [{"name": "open_gripper", "args": {}}, {"name": "close_gripper", "args": {}}],
        robot_config().init_joints_deg,
        robot_config().orientation_roll_deg,
        robot_config().orientation_pitch_deg,
        gripper_steps=2,
    )
    assert frames[0].configuration_rad[-1] == pytest.approx(0.0)
    assert frames[-1].configuration_rad[-1] == pytest.approx(0.85)


def test_runner_saves_preflight_result(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True)
    source = "def run():\n    home()\n    move(500, 0, 100, 0)\n"
    write_experiment(workspace, "experiment_001.py", source)
    runner = ExperimentRunner(run_dir, robot_config())
    result = runner.run_experiment("experiment_001.py")
    assert result["execution_completed"] is True
    assert result["physical_execution"] is False
    saved = json.loads((run_dir / "results/experiment_001.json").read_text())
    assert len(saved["requested_robot_actions"]) == 2
    assert saved["experiment_source"] == source
    assert (run_dir / "results/experiment_001.source.py").read_text() == source


def test_runner_saves_static_validation_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True)
    write_experiment(workspace, "experiment_bad.py", "import xarm\ndef run():\n    home()\n")
    runner = ExperimentRunner(run_dir, robot_config())
    with pytest.raises(ExperimentValidationError):
        runner.run_experiment("experiment_bad.py")
    saved = json.loads((run_dir / "results/experiment_bad.json").read_text())
    assert saved["execution_completed"] is False
    assert "imports are forbidden" in saved["robot_errors"][0]


def test_session_layout_and_memory(tmp_path: Path) -> None:
    experiment = ExperimentConfig(500, 0, 40, 100, 200, 0)
    session = AgentSession.create(tmp_path, "test goal", robot_config(), experiment, run_id="run_1")
    assert (session.workspace / "memory.md").is_file()
    assert (session.workspace / "ROBOT_API.md").is_file()
    assert (session.workspace / "experiment_config.json").is_file()
    session.update_memory("experiment_001", hypothesis="lower grasp", next_experiment="experiment_002.py", result="FAILED_GRASP")
    assert "Why this change" in (session.workspace / "memory.md").read_text()
    (session.results / "experiment_001.trace.json").write_text("[]")
    metadata = json.loads((session.run_dir / "run_metadata.json").read_text())
    assert "physical_rollout_limit" not in metadata
    with pytest.raises(ValueError):
        session.invoke_claude_code("bad", experiment_name="../core.py")


def test_claude_generation_waits_for_automatic_plan(tmp_path: Path) -> None:
    session = AgentSession.create(
        tmp_path, "automatic plan", robot_config(), ExperimentConfig(), run_id="deferred"
    )
    with pytest.raises(ConfigError, match="plan is incomplete"):
        session.invoke_claude_code("write a grasp")


def test_claude_code_prompt_supports_anchor_tests_and_laydown(tmp_path: Path) -> None:
    captured = {}

    class CaptureClaude:
        def invoke(self, prompt, workspace):
            captured["prompt"] = prompt
            captured["workspace"] = workspace
            return None

    session = AgentSession.create(
        tmp_path,
        "anchor discovery",
        robot_config(),
        ExperimentConfig(500, 0, 40, None, None, None),
        run_id="anchor_prompt",
        claude=CaptureClaude(),  # type: ignore[arg-type]
    )
    perception_views = session.workspace / "perception_views"
    perception_views.mkdir()
    (perception_views / "observation.json").write_text("{}", encoding="utf-8")
    (perception_views / "camera_A_coordinate_guide.json").write_text(
        "{}", encoding="utf-8"
    )
    session.invoke_claude_code("test an uncertain possible anchor")
    prompt = captured["prompt"]
    assert "usable garment lifting anchor" in prompt
    assert "minimal cautious test" in prompt
    assert "camera_*_coordinate_guide.json" in prompt
    assert "Skill: laydown" in prompt
    assert list((session.results / "claude").glob("*.json"))


def test_real_session_always_attempts_home_after_claude_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = AgentSession.create(
        tmp_path,
        "mandatory home",
        robot_config(),
        ExperimentConfig(500, 0, 40, None, None, None),
        run_id="mandatory_home",
    )
    calls = []

    def fake_run(path, *, real=False, confirmed=False, notes=""):
        calls.append((str(path), real, confirmed, notes))
        if str(path).startswith("_mandatory_return_home_"):
            return {
                "experiment": Path(path).stem,
                "execution_completed": True,
                "robot_errors": [],
            }
        return {
            "experiment": "experiment_001",
            "execution_completed": True,
            "robot_errors": [],
        }

    monkeypatch.setattr(session.runner, "run_experiment", fake_run)
    result = session.run_experiment(
        "experiment_001.py", real=True, confirmed=True, notes="Claude rollout"
    )
    assert len(calls) == 2
    assert calls[1][0].startswith("_mandatory_return_home_")
    assert result["mandatory_return_home"]["completed"] is True
    assert session.last_return_home_outcome == result["mandatory_return_home"]
    assert list((session.results / "mandatory_return_home").glob("*.json"))


def test_real_session_returns_home_between_perception_and_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = AgentSession.create(
        tmp_path,
        "pre-run home",
        robot_config(),
        ExperimentConfig(500, 0, 40, None, None, None),
        run_id="pre_run_home",
    )
    metadata_path = session.run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pre_run_home_required"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    calls = []

    def fake_run(path, *, real=False, confirmed=False, notes=""):
        calls.append(str(path))
        return {
            "experiment": Path(path).stem,
            "execution_completed": True,
            "robot_errors": [],
        }

    monkeypatch.setattr(session.runner, "run_experiment", fake_run)
    result = session.run_experiment("experiment_001.py", real=True, confirmed=True)

    assert calls[0].startswith("_mandatory_pre_run_home_")
    assert calls[1] == "experiment_001.py"
    assert calls[2].startswith("_mandatory_return_home_")
    assert result["mandatory_pre_run_home"]["completed"] is True
    assert session.last_pre_run_home_outcome == result["mandatory_pre_run_home"]
    assert json.loads(metadata_path.read_text(encoding="utf-8"))[
        "pre_run_home_required"
    ] is False
    assert list((session.results / "mandatory_pre_run_home").glob("*.json"))


def test_failed_pre_run_home_blocks_rollout_and_keeps_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = AgentSession.create(
        tmp_path,
        "failed pre-run home",
        robot_config(),
        ExperimentConfig(500, 0, 40, None, None, None),
        run_id="failed_pre_run_home",
    )
    metadata_path = session.run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pre_run_home_required"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    calls = []

    def fake_run(path, *, real=False, confirmed=False, notes=""):
        calls.append(str(path))
        return {
            "experiment": Path(path).stem,
            "execution_completed": False,
            "robot_errors": ["home failed"],
        }

    monkeypatch.setattr(session.runner, "run_experiment", fake_run)
    with pytest.raises(RuntimeError, match="physical rollout was blocked"):
        session.run_experiment("experiment_001.py", real=True, confirmed=True)

    assert len(calls) == 1
    assert calls[0].startswith("_mandatory_pre_run_home_")
    assert session.last_pre_run_home_outcome is not None
    assert session.last_pre_run_home_outcome["completed"] is False
    assert session.last_return_home_outcome is None
    assert json.loads(metadata_path.read_text(encoding="utf-8"))[
        "pre_run_home_required"
    ] is True


def test_real_session_attempts_home_even_when_claude_rollout_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = AgentSession.create(
        tmp_path,
        "mandatory home on failure",
        robot_config(),
        ExperimentConfig(500, 0, 40, None, None, None),
        run_id="mandatory_home_failure",
    )
    calls = []

    def fake_run(path, *, real=False, confirmed=False, notes=""):
        calls.append(str(path))
        if str(path).startswith("_mandatory_return_home_"):
            return {
                "experiment": Path(path).stem,
                "execution_completed": True,
                "robot_errors": [],
            }
        raise RuntimeError("rollout failed")

    monkeypatch.setattr(session.runner, "run_experiment", fake_run)
    with pytest.raises(RuntimeError, match="rollout failed"):
        session.run_experiment("experiment_001.py", real=True, confirmed=True)
    assert len(calls) == 2
    assert calls[1].startswith("_mandatory_return_home_")
    assert session.last_return_home_outcome is not None
    assert session.last_return_home_outcome["completed"] is True


def test_unconfirmed_real_session_does_not_send_home_motion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = AgentSession.create(
        tmp_path,
        "unconfirmed",
        robot_config(),
        ExperimentConfig(500, 0, 40, None, None, None),
        run_id="unconfirmed_no_home",
    )
    calls = []

    def fake_run(path, *, real=False, confirmed=False, notes=""):
        calls.append(str(path))
        raise PermissionError("confirmation required")

    monkeypatch.setattr(session.runner, "run_experiment", fake_run)
    with pytest.raises(PermissionError, match="confirmation required"):
        session.run_experiment("experiment_001.py", real=True, confirmed=False)
    assert calls == ["experiment_001.py"]
    assert session.last_return_home_outcome is None


def test_claude_invocation_has_workspace_only_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    seen = {}

    monkeypatch.setattr("cloth_agent.claude.shutil.which", lambda _: "/usr/bin/claude")

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout='{"result":"ok"}', stderr="")

    monkeypatch.setattr("cloth_agent.claude.subprocess.run", fake_run)
    ClaudeCodeClient().invoke("write experiment", workspace)
    assert seen["cwd"] == workspace.resolve()
    assert seen["shell"] is False
    assert "Read,Edit,Write" in seen["command"]
    assert "Bash" not in seen["command"]
    assert str(workspace.resolve()) in seen["command"]
    assert any("lifting anchor" in str(part) for part in seen["command"])


def perception_config(tmp_path: Path, disagreement_mm: float = 50.0) -> PerceptionConfig:
    dummy = tmp_path / "dummy.yaml"
    dummy.write_text("X_CammountCam: []")
    return PerceptionConfig(
        cameras=(CameraSpec("A", "A1", dummy), CameraSpec("B", "B1", dummy)),
        molmo=MolmoConfig(Path(sys.executable)),
        width=10,
        height=10,
        fps=30,
        warmup_frames=0,
        depth_window_radius_px=1,
        min_depth_m=0.1,
        max_depth_m=2.0,
        max_view_disagreement_mm=disagreement_mm,
    )


def test_perception_pixel_depth_and_multiview_contract() -> None:
    depth = np.ones((10, 10), dtype=np.float32)
    depth[5, 5] = 0
    assert robust_depth_at_pixel(depth, 5, 5, 1, 0.1, 2.0) == pytest.approx(1.0)
    K = np.asarray([[100, 0, 5], [0, 100, 5], [0, 0, 1]], dtype=float)
    X = np.eye(4)
    X[0, 3] = 0.5
    assert pixel_to_base_mm(5, 5, 1.0, K, X).tolist() == pytest.approx([500, 0, 1000])
    assert points_by_image([[0, 0, 5, 5], [0, 1, 6, 5]]) == {0: [5.0, 5.0], 1: [6.0, 5.0]}
    with pytest.raises(PerceptionError, match="exactly one"):
        points_by_image([[0, 0, 5, 5]])


def test_height_heatmap_makes_larger_table_clearance_brighter() -> None:
    values = np.asarray([[0.0, 100.0]], dtype=np.float64)
    heatmap = _scalar_heatmap_rgb(
        values,
        higher_is_bright=True,
    )
    assert int(heatmap[0, 1].sum()) > int(heatmap[0, 0].sum())


def test_height_heatmap_explicit_table_zero_range_is_physical() -> None:
    values = np.asarray([[-20.0, 0.0, 50.0, 100.0, 140.0]], dtype=np.float64)
    heatmap = _scalar_heatmap_rgb(
        values,
        higher_is_bright=True,
        value_range_mm=(0.0, 100.0),
    )
    # Values below table zero clip to the dark endpoint and values above the
    # shared range clip to the bright endpoint.
    assert int(heatmap[0, 0].sum()) == int(heatmap[0, 1].sum())
    assert int(heatmap[0, 2].sum()) < int(heatmap[0, 3].sum())
    assert int(heatmap[0, 3].sum()) == int(heatmap[0, 4].sum())
    assert _height_display_max_mm(np.asarray([2.0, 99.0])) == 100.0
    assert _height_display_max_mm(np.asarray([200.0])) == 160.0


def test_garment_softmax_heatmap_makes_higher_height_hotter() -> None:
    values = np.asarray([[0.0, 10.0, 20.0]], dtype=np.float64)
    mask = np.ones(values.shape, dtype=bool)
    heatmap = _scalar_heatmap_rgb(
        values,
        mask,
        focus_mask=mask,
        higher_is_bright=True,
        value_range_mm=(-20.0, 20.0),
        normalization="global_softmax",
    )
    assert int(heatmap[0, 2].sum()) > int(heatmap[0, 0].sum())


def test_fused_height_map_filters_non_garment_before_grid_median(tmp_path: Path) -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 100.0],  # non-garment high outlier in the same cell
            [1.0, 0.0, 20.0],
            [4.0, 0.0, 10.0],
        ],
        dtype=np.float64,
    )
    heights = np.asarray([0.0, 100.0, 20.0, 10.0], dtype=np.float64)
    garment_mask = np.asarray([True, False, True, True])

    metadata = _save_fused_height_map(
        tmp_path,
        points,
        heights,
        garment_mask,
        grid_size_mm=4.0,
    )
    height_map = np.load(tmp_path / "fused_height_map_mm.npy")

    assert height_map[0, 0] == pytest.approx(10.0)
    assert float(np.nanmax(height_map)) == pytest.approx(10.0)
    assert metadata["height_map_aggregation"] == "garment_only_grid_median"
    assert metadata["height_map_display_scale"] == "fixed_physical_-5_to_40mm"
    assert metadata["heatmap_display_min_mm"] == DEFAULT_HEIGHT_MAP_DISPLAY_MIN_MM
    assert DEFAULT_HEIGHT_MAP_DISPLAY_MAX_MM == 40.0


def test_table_clip_uses_tilted_plane_and_bounded_noise_tolerance() -> None:
    coefficients = np.asarray([0.1, -0.2, 50.0])
    xy = np.asarray([[0.0, 0.0], [100.0, 50.0], [20.0, -40.0], [5.0, 5.0]])
    table_z = coefficients[0] * xy[:, 0] + coefficients[1] * xy[:, 1] + coefficients[2]
    points = np.column_stack((xy, table_z + np.asarray([0.0, -11.0, -13.0, 40.0])))
    points = np.vstack((points, np.asarray([np.nan, 0.0, 0.0])))

    heights, keep = table_height_and_clip_mask(
        points,
        coefficients,
        tolerance_mm=12.0,
    )

    assert heights[:4].tolist() == pytest.approx([0.0, -11.0, -13.0, 40.0])
    assert keep.tolist() == [True, True, False, True, False]
    assert table_clip_tolerance_mm(None) == pytest.approx(12.0)
    assert table_clip_tolerance_mm(8.0) == pytest.approx(16.0)
    assert table_clip_tolerance_mm(100.0) == pytest.approx(30.0)


def test_camera_table_z_bias_correction_centers_bright_table(tmp_path: Path) -> None:
    config = perception_config(tmp_path)
    depth = np.ones((30, 30), dtype=np.float32)
    transform = np.eye(4)
    transform[2, 3] = -1.006
    frame = RGBDFrame(
        "B",
        "B1",
        np.full((30, 30, 3), 240, dtype=np.uint8),
        depth,
        np.asarray([[100.0, 0.0, 14.5], [0.0, 100.0, 14.5], [0.0, 0.0, 1.0]]),
        transform,
    )

    offset_mm, diagnostics = _estimate_camera_table_z_offset_mm(
        frame,
        config,
        np.asarray([0.0, 0.0, 0.0]),
    )
    height_map, valid, _ = camera_height_map_mm(
        frame,
        config,
        np.asarray([0.0, 0.0, 0.0]),
        base_z_offset_mm=offset_mm,
    )

    assert diagnostics["applied"] is True
    assert offset_mm == pytest.approx(6.0, abs=1e-3)
    assert np.nanmedian(height_map[valid]) == pytest.approx(0.0, abs=1e-3)
    assert _signed_height_display_range_mm(np.asarray([-11.0, 0.0, 12.0])) == 20.0


def test_frame_and_saved_view_point_clouds_clip_below_table(tmp_path: Path) -> None:
    rgb = np.asarray([[[10, 0, 0], [20, 0, 0], [30, 0, 0]]], dtype=np.uint8)
    depth = np.ones((1, 3), dtype=np.float32)
    height_map = np.asarray([[0.0, -20.0, 25.0]], dtype=np.float32)
    frame = RGBDFrame(
        "A",
        "A1",
        rgb,
        depth,
        np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.eye(4),
    )

    points, colors = _frame_point_cloud(
        frame,
        stride=1,
        height_above_table_mm=height_map,
    )
    assert len(points) == 2
    assert colors[:, 0].tolist() == [10, 30]

    Image.fromarray(rgb).save(tmp_path / "camera_0_A.png")
    np.save(tmp_path / "camera_0_A_depth_m.npy", depth)
    np.save(tmp_path / "camera_A_height_above_table_mm.npy", height_map)
    view = {
        "label": "A",
        "serial": "A1",
        "image": "camera_0_A.png",
        "depth_m": "camera_0_A_depth_m.npy",
        "height_map_path": "camera_A_height_above_table_mm.npy",
        "intrinsics": frame.intrinsics.tolist(),
        "X_base_camera": frame.X_base_camera.tolist(),
    }
    _, saved_colors = _view_point_cloud(view, tmp_path, stride=1)
    assert saved_colors[:, 0].tolist() == [10, 30]


def test_fused_cloud_loader_enforces_saved_height_contract(tmp_path: Path) -> None:
    np.save(
        tmp_path / "points.npy",
        np.asarray([[100.0, 0.0, 50.0], [200.0, 0.0, -40.0], [300.0, 0.0, 80.0]]),
    )
    np.save(
        tmp_path / "colors.npy",
        np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8),
    )
    np.save(tmp_path / "height.npy", np.asarray([0.0, -40.0, 30.0]))
    result = {
        "depth_fusion": {
            "table_clip": {"tolerance_mm": 12.0},
            "artifacts": {
                "fused_points_base_mm": "points.npy",
                "fused_colors_rgb": "colors.npy",
                "fused_height_above_table_mm": "height.npy",
            },
        }
    }

    points, colors = _load_fused_point_cloud(result, tmp_path)

    assert np.allclose(points, [[0.1, 0.0, 0.05], [0.3, 0.0, 0.08]])
    assert colors.tolist() == [[1, 2, 3], [7, 8, 9]]


def test_projected_garment_mask_rejects_nearer_occluder(tmp_path: Path) -> None:
    config = perception_config(tmp_path)
    intrinsics = np.asarray([[1.0, 0.0, 4.5], [0.0, 1.0, 4.5], [0.0, 0.0, 1.0]])
    depth = np.ones((10, 10), dtype=np.float32)
    depth[5, 5] = 0.90
    frame = RGBDFrame(
        "A",
        "A1",
        np.full((10, 10, 3), 120, dtype=np.uint8),
        depth,
        intrinsics,
        np.eye(4),
    )
    pixels_y, pixels_x = np.mgrid[0:10, 0:10]
    garment_points = np.column_stack(
        [
            (pixels_x.reshape(-1) - 4.5) * 1000.0,
            (pixels_y.reshape(-1) - 4.5) * 1000.0,
            np.full(100, 1000.0),
        ]
    )
    height_map = np.full((10, 10), 100.0, dtype=np.float32)
    height_map[5, 5] = 0.0
    mask, sparse, diagnostics = _occlusion_aware_garment_mask(
        garment_points,
        frame,
        height_map,
        np.ones((10, 10), dtype=bool),
    )
    assert sparse[5, 5]
    assert not mask[5, 5]
    assert mask[4, 4]
    assert diagnostics["depth_consistency_tolerance_mm"] == 25.0


def test_height_gradient_edges_stay_inside_eroded_garment() -> None:
    field = np.zeros((20, 20), dtype=np.float32)
    field[8:12, 8:12] = 50.0
    mask = np.ones((20, 20), dtype=bool)
    edges, _ = _fold_edge_mask(field, mask)
    assert not np.any(edges & _mask_boundary(mask))


def test_table_corner_interpolation_rejects_occluded_reference(tmp_path: Path) -> None:
    config = perception_config(tmp_path)
    height = width = 100
    intrinsics = np.asarray(
        [[100.0, 0.0, 49.5], [0.0, 100.0, 49.5], [0.0, 0.0, 1.0]]
    )
    depth = np.ones((height, width), dtype=np.float32)
    # One table corner is occupied by a much nearer object.
    depth[:25, 75:] = 0.5
    X_base_camera = np.eye(4)
    X_base_camera[2, 3] = -1.0
    frame = RGBDFrame(
        "A",
        "A1",
        np.full((height, width, 3), 240, dtype=np.uint8),
        depth,
        intrinsics,
        X_base_camera,
    )
    coefficients, diagnostics = _fit_table_plane_from_references(
        [frame],
        config,
        np.asarray([0.0, 0.0, 100.0]),
    )
    assert diagnostics["mode"] == "corner_edge_depth_interpolation"
    assert diagnostics["inlier_count"] >= 7
    assert coefficients.tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    top_right = next(
        item
        for item in diagnostics["cameras"]["A"]
        if item["name"] == "top_right"
    )
    assert not top_right["plane_inlier"]


def test_camera_height_map_is_surface_minus_table_height(tmp_path: Path) -> None:
    config = perception_config(tmp_path)
    intrinsics = np.asarray([[1.0, 0.0, 4.5], [0.0, 1.0, 4.5], [0.0, 0.0, 1.0]])
    frame = RGBDFrame(
        "A",
        "A1",
        np.full((10, 10, 3), 120, dtype=np.uint8),
        np.ones((10, 10), dtype=np.float32),
        intrinsics,
        np.eye(4),
    )
    pixels_y, pixels_x = np.mgrid[0:10, 0:10]
    garment_points = np.column_stack(
        [
            (pixels_x.reshape(-1) - 4.5) * 1000.0,
            (pixels_y.reshape(-1) - 4.5) * 1000.0,
            np.full(100, 1000.0),
        ]
    )
    artifacts = _save_camera_height_heatmap(
        tmp_path,
        frame,
        config,
        garment_points,
        np.asarray([0.0, 0.0, 900.0]),
    )
    height_map = np.load(tmp_path / artifacts["height_map_path"])
    assert artifacts["heatmap_quantity"] == "height_above_table_mm"
    assert height_map[5, 5] == pytest.approx(100.0)
    assert (tmp_path / artifacts["height_map"]).is_file()
    assert (tmp_path / artifacts["base_xyz_map"]).is_file()
    assert (tmp_path / artifacts["coordinate_overlay"]).is_file()
    coordinate_guide = json.loads(
        (tmp_path / artifacts["coordinate_guide"]).read_text(encoding="utf-8")
    )
    assert coordinate_guide["coordinate_frame"] == "robot_base_mm"
    assert "not grasp candidates" in coordinate_guide["reference_semantics"]
    assert coordinate_guide["samples"]
    assert coordinate_guide["samples"][0]["base_xyz_mm"][2] == pytest.approx(1000.0)
    assert coordinate_guide["samples"][0]["height_above_table_mm"] == pytest.approx(100.0)


class FakeMolmo:
    def __init__(self, points):
        self.points = points
        self.image_counts = []

    def locate(self, image_paths, output_path, prompt):
        self.image_counts.append(len(image_paths))
        payload = {"generated_text": "points", "points": self.points}
        output_path.write_text(json.dumps(payload))
        return payload


def make_frames(
    second_offset_m: float = 0.0,
    base_z_m: float = 0.02,
    primary_depth_m: float = 1.0,
    auxiliary_depth_m: float = 1.0,
) -> list[RGBDFrame]:
    K = np.asarray([[100, 0, 5], [0, 100, 5], [0, 0, 1]], dtype=float)
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    primary_depth = np.full((10, 10), primary_depth_m, dtype=np.float32)
    auxiliary_depth = np.full((10, 10), auxiliary_depth_m, dtype=np.float32)
    X_a = np.eye(4)
    X_a[0, 3] = 0.5
    X_a[2, 3] = base_z_m - 1.0
    X_b = X_a.copy()
    X_b[0, 3] += second_offset_m
    return [
        RGBDFrame("A", "A1", rgb, primary_depth, K, X_a),
        RGBDFrame("B", "B1", rgb, auxiliary_depth, K, X_b),
    ]


def test_successful_session_perception_requires_home_before_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = AgentSession.create(
        tmp_path,
        "perception home marker",
        robot_config(),
        ExperimentConfig(),
        run_id="perception_home_marker",
    )

    def fake_locate(self, output_dir, experiment, frames=None):
        output_dir.mkdir(parents=True)
        result = {
            "center_base_mm": [500.0, 0.0, 20.0],
            "surface_z_mm": 20.0,
            "perception_mode": "dual_camera_rgbd",
            "active_cameras": ["A", "B"],
            "views": [],
            "depth_fusion": {
                "fused_point_count": 100,
                "input_point_count": 120,
                "source_voxel_counts": {"AB_overlap": 80},
                "artifacts": {},
            },
        }
        (output_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return result, ExperimentConfig(500, 0, 20, 100, 180, 0)

    monkeypatch.setattr(ClothCenterPerception, "locate", fake_locate)
    before = json.loads(
        (session.run_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert before["pre_run_home_required"] is False

    session.locate_cloth_center(perception_config(tmp_path), frames=make_frames())

    after = json.loads(
        (session.run_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert after["pre_run_home_required"] is True
    assert after["pre_run_home_required_at"]


def test_two_view_molmo_center_updates_experiment(tmp_path: Path) -> None:
    config = perception_config(tmp_path)
    molmo = FakeMolmo([[0, 0, 5, 5]])
    service = ClothCenterPerception(
        tmp_path,
        robot_config(),
        config,
        capture=lambda _: make_frames(primary_depth_m=1.03, auxiliary_depth_m=1.0),
        molmo_client=molmo,
    )
    initial = ExperimentConfig()
    result, updated = service.locate(tmp_path / "perception", initial)
    assert molmo.image_counts == [1]
    assert result["status"] == "VALIDATED_PRIMARY_AUX_DEPTH"
    assert result["perception_mode"] == "primary_rgb_auxiliary_depth"
    assert result["primary_camera"] == "A"
    assert result["auxiliary_depth_cameras"] == ["B"]
    assert [view["role"] for view in result["views"]] == [
        "primary_semantic",
        "auxiliary_depth",
    ]
    assert updated.cloth_center_x == pytest.approx(500)
    assert updated.cloth_center_y == pytest.approx(0)
    assert updated.grasp_z == pytest.approx(20)
    assert updated.approach_z == pytest.approx(100)
    assert updated.lift_z == pytest.approx(180)
    assert updated.yaw_deg == pytest.approx(robot_config().init_pose_mm_deg[5])
    assert result["motion_derivation"]["surface_z_mm"] == pytest.approx(20)
    assert result["depth_fusion"]["selected_depth_camera"] == "B"
    assert result["depth_fusion"]["primary_depth_point_base_mm"][2] == pytest.approx(50)
    assert result["view_disagreement_mm"] == pytest.approx(30)
    assert (tmp_path / "perception/result.json").is_file()


def test_single_camera_a_rgbd_updates_experiment(tmp_path: Path) -> None:
    base = perception_config(tmp_path)
    config = PerceptionConfig(
        **{**base.__dict__, "active_camera_labels": ("A",)}
    )
    service = ClothCenterPerception(
        tmp_path,
        robot_config(),
        config,
        capture=lambda _: make_frames()[:1],
        molmo_client=FakeMolmo([[0, 0, 5, 5]]),
    )
    result, updated = service.locate(tmp_path / "single_a", ExperimentConfig())
    assert result["status"] == "VALIDATED_SINGLE_VIEW"
    assert result["perception_mode"] == "single_camera_rgbd"
    assert result["active_cameras"] == ["A"]
    assert result["view_disagreement_mm"] is None
    assert updated.cloth_center_x == pytest.approx(500)
    assert updated.cloth_center_y == pytest.approx(0)
    assert updated.grasp_z == pytest.approx(20)


def test_single_view_real_run_requires_extra_confirmation(tmp_path: Path) -> None:
    experiment = ExperimentConfig(500, 0, 40, 100, 200, 0)
    session = AgentSession.create(
        tmp_path, "single view guard", robot_config(), experiment, run_id="single_guard"
    )
    write_experiment(session.workspace, "experiment_001.py", "def run():\n    home()\n")
    metadata_path = session.run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["last_perception_mode"] = "single_camera_rgbd"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(PermissionError, match="single-camera"):
        session.run_experiment("experiment_001.py", real=True, confirmed=True)
    assert session.last_return_home_outcome is None


def test_automatic_plan_caps_lift_at_safe_upper_z(tmp_path: Path) -> None:
    config = perception_config(tmp_path)
    config = PerceptionConfig(
        **{
            **config.__dict__,
            "approach_clearance_mm": 40.0,
            "lift_clearance_mm": 160.0,
        }
    )
    plan, derivation = derive_grasp_plan(
        np.asarray([500.0, 0.0, 250.0]), robot_config(), config
    )
    assert plan.grasp_z == pytest.approx(250)
    assert plan.approach_z == pytest.approx(290)
    assert plan.lift_z == pytest.approx(399)
    assert derivation["upper_z_safety_adjustment_mm"] == pytest.approx(-11)


def test_two_view_disagreement_blocks_center(tmp_path: Path) -> None:
    config = perception_config(tmp_path, disagreement_mm=50)
    service = ClothCenterPerception(
        tmp_path,
        robot_config(),
        config,
        capture=lambda _: make_frames(primary_depth_m=1.0, auxiliary_depth_m=0.9),
        molmo_client=FakeMolmo([[0, 0, 5, 5]]),
    )
    with pytest.raises(PerceptionError, match="disagree"):
        service.locate(tmp_path / "perception", ExperimentConfig())


def test_auxiliary_occlusion_uses_validated_primary_depth_fallback(tmp_path: Path) -> None:
    config = perception_config(tmp_path, disagreement_mm=50)
    service = ClothCenterPerception(
        tmp_path,
        robot_config(),
        config,
        capture=lambda _: make_frames(second_offset_m=0.1),
        molmo_client=FakeMolmo([[0, 0, 5, 5]]),
    )
    result, updated = service.locate(tmp_path / "fallback", ExperimentConfig())
    assert result["status"] == "VALIDATED_PRIMARY_DEPTH_FALLBACK"
    assert result["perception_mode"] == "primary_rgbd_auxiliary_unavailable"
    assert result["depth_fusion"]["selected_depth_camera"] == "A"
    assert result["depth_fusion"]["auxiliary_status"] == "occluded_or_outside_view"
    assert result["depth_fusion"]["primary_depth_quality"]["spread_mm"] == pytest.approx(0)
    assert result["views"][1]["role"] == "auxiliary_depth_unavailable"
    assert result["warnings"]
    assert updated.grasp_z == pytest.approx(20)


def test_auxiliary_occlusion_blocks_unstable_primary_depth(tmp_path: Path) -> None:
    config = perception_config(tmp_path, disagreement_mm=50)
    frames = make_frames(second_offset_m=0.1)
    frames[0].depth_m[:, :5] = 0.8
    frames[0].depth_m[:, 5:] = 1.2
    service = ClothCenterPerception(
        tmp_path,
        robot_config(),
        config,
        capture=lambda _: frames,
        molmo_client=FakeMolmo([[0, 0, 5, 5]]),
    )
    with pytest.raises(PerceptionError, match="unstable fold/edge"):
        service.locate(tmp_path / "unstable_fallback", ExperimentConfig())
