from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cloth_agent.config import ExperimentConfig, RobotConfig, WorkspaceBounds
from cloth_agent.experiment import validate_experiment_source
from cloth_agent.free_exploration import (
    ClaudeExplorationClient,
    ExplorationPlanningError,
    _json_from_claude_text,
    exploration_prompt,
    exploration_source,
    _voxel_balance_cloud,
    validate_exploration_payload,
    _load_or_create_session,
    _controller_ik_failure_message,
)
from cloth_agent.session import AgentSession


def _robot_config() -> RobotConfig:
    return RobotConfig(
        robot_ip="127.0.0.1",
        boundaries=WorkspaceBounds(
            x_min=0, x_max=800, y_min=-400, y_max=400, z_min=10, z_max=400
        ),
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


def test_free_exploration_payload_is_strict_and_compiles():
    proposal = validate_exploration_payload(
        {
            "garment_observation": "A large fold hides the lower-left panel.",
            "reveal_strategy": "Approach above the fold, lift gently, then release.",
            "confidence": 0.72,
            "actions": [
                {"name": "home", "args": {}},
                {"name": "open_gripper", "args": {}},
                {"name": "move", "args": {"x": 500, "y": -20, "z": 100, "yaw": 0}},
                {"name": "close_gripper", "args": {}},
                {"name": "move", "args": {"x": 500, "y": -20, "z": 200, "yaw": 15}},
                {"name": "move", "args": {"x": 540, "y": -20, "z": 200, "yaw": 15}},
                {"name": "move", "args": {"x": 540, "y": -20, "z": 100, "yaw": 15}},
                {"name": "open_gripper", "args": {}},
                {"name": "home", "args": {}},
            ],
            "expected_observation": "The lower-left panel should become visible.",
            "safety_notes": ["Keep the lift low and review the path before execution."],
        }
    )
    source = exploration_source(proposal)
    validate_experiment_source(source)
    assert "garment_observation" not in source
    assert "move(500.0, -20.0, 100.0, 0.0)" in source


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {
            "garment_observation": "x",
            "reveal_strategy": "y",
            "confidence": 2,
            "actions": [{"name": "home", "args": {}}],
            "expected_observation": "z",
            "safety_notes": ["safe"],
        },
        {
            "garment_observation": "x",
            "reveal_strategy": "y",
            "confidence": 0.5,
            "actions": [{"name": "sdk_call", "args": {}}],
            "expected_observation": "z",
            "safety_notes": ["safe"],
        },
    ],
)
def test_free_exploration_rejects_hard_schema_failures(payload):
    with pytest.raises(ExplorationPlanningError):
        validate_exploration_payload(payload)


def test_free_exploration_requires_explicit_post_grasp_release_geometry():
    payload = {
        "garment_observation": "fold",
        "reveal_strategy": "lift",
        "confidence": 0.5,
        "actions": [
            {"name": "home", "args": {}},
            {"name": "open_gripper", "args": {}},
            {"name": "move", "args": {"x": 500, "y": 0, "z": 100, "yaw": 0}},
            {"name": "close_gripper", "args": {}},
            {"name": "move", "args": {"x": 500, "y": 0, "z": 200, "yaw": 0}},
            {"name": "open_gripper", "args": {}},
        ],
        "expected_observation": "more fabric",
        "safety_notes": ["review"],
    }
    with pytest.raises(ExplorationPlanningError, match="two post-grasp move"):
        validate_exploration_payload(payload)


def test_claude_json_extractor_accepts_cli_envelope_and_fence():
    payload = {
        "garment_observation": "fold",
        "reveal_strategy": "lift",
        "confidence": 0.5,
        "actions": [
            {"name": "home", "args": {}},
            {"name": "open_gripper", "args": {}},
            {"name": "move", "args": {"x": 500, "y": 0, "z": 100, "yaw": 0}},
            {"name": "move", "args": {"x": 500, "y": 0, "z": 80, "yaw": 0}},
            {"name": "close_gripper", "args": {}},
            {"name": "move", "args": {"x": 500, "y": 0, "z": 180, "yaw": 0}},
            {"name": "move", "args": {"x": 540, "y": 0, "z": 180, "yaw": 0}},
            {"name": "move", "args": {"x": 540, "y": 0, "z": 100, "yaw": 0}},
            {"name": "open_gripper", "args": {}},
        ],
        "expected_observation": "more fabric",
        "safety_notes": ["review"],
    }
    wrapped = json.dumps({"result": "```json\n" + json.dumps(payload) + "\n```"})
    assert _json_from_claude_text(wrapped) == payload


def test_exploration_client_is_read_only_and_logs_proposal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "run"
    images = run_dir / "results" / "perception"
    images.mkdir(parents=True)
    image = images / "camera_A.png"
    image.write_bytes(b"image")
    payload = {
        "garment_observation": "fold",
        "reveal_strategy": "lift",
        "confidence": 0.6,
        "actions": [
            {"name": "home", "args": {}},
            {"name": "open_gripper", "args": {}},
            {"name": "move", "args": {"x": 500, "y": 0, "z": 100, "yaw": 0}},
            {"name": "move", "args": {"x": 500, "y": 0, "z": 80, "yaw": 0}},
            {"name": "close_gripper", "args": {}},
            {"name": "move", "args": {"x": 500, "y": 0, "z": 180, "yaw": 0}},
            {"name": "move", "args": {"x": 540, "y": 0, "z": 180, "yaw": 0}},
            {"name": "move", "args": {"x": 540, "y": 0, "z": 100, "yaw": 0}},
            {"name": "open_gripper", "args": {}},
        ],
        "expected_observation": "more",
        "safety_notes": ["review"],
    }
    seen = {}
    monkeypatch.setattr("cloth_agent.free_exploration.shutil.which", lambda _: "/usr/bin/claude")

    def fake_run(command, **kwargs):
        seen.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"result": json.dumps(payload)}), stderr="")

    monkeypatch.setattr("cloth_agent.free_exploration.subprocess.run", fake_run)
    result = ClaudeExplorationClient().invoke([image], "inspect", run_dir)
    assert result.proposal.garment_observation == "fold"
    assert seen["cwd"] == run_dir.resolve()
    assert seen["shell"] is False
    assert "Bash" not in seen["command"]
    assert "Write" not in seen["command"]
    assert list((run_dir / "results" / "claude_exploration").glob("*.json"))


def test_exploration_prompt_surfaces_capabilities():
    prompt = exploration_prompt(
        ExperimentConfig(500, -20, 40, 100, 200, 0), _robot_config()
    )
    assert "move(x,y,z,yaw)" in prompt
    assert "workspace bounds" in prompt
    assert "occlusions" in prompt


def test_exploration_prompt_makes_all_motion_heights_agent_decisions():
    prompt = exploration_prompt(
        ExperimentConfig(500, -20, 40, 100, 200, 0), _robot_config()
    )
    assert "There is no fixed approach/lift/release clearance" in prompt
    assert "decide the approach height, grasp height, lift height" in prompt


def test_voxel_balance_cloud_normalizes_world_space_density():
    points = []
    colors = []
    for x in range(10):
        for y in range(10):
            points.append([x * 0.001, y * 0.001, 0.5])
            colors.append([x, y, 10])
    balanced_points, balanced_colors = _voxel_balance_cloud(
        np.asarray(points),
        np.asarray(colors),
        voxel_size_mm=5.0,
    )
    assert len(balanced_points) == 4
    assert balanced_colors.shape == (4, 3)


def test_existing_run_id_is_reopened_instead_of_created_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run = AgentSession.create(
        tmp_path,
        "free exploration",
        _robot_config(),
        ExperimentConfig(),
        run_id="claude_explore_real",
    )
    monkeypatch.setattr(
        "cloth_agent.free_exploration.RobotConfig.load",
        lambda project_root, config_path=None: _robot_config(),
    )
    reopened = _load_or_create_session(
        tmp_path,
        run_dir=None,
        run_id="claude_explore_real",
        robot_config=None,
    )
    assert reopened.run_dir == run.run_dir
    assert reopened.workspace.is_dir()


def test_controller_ik_failure_message_explains_code_10():
    message = _controller_ik_failure_message(
        RuntimeError(
            "controller IK rejected action 6 pose=[707.878, -26.478, 217.8, "
            "178.370814, 3.606941, 170.569], code=10"
        )
    )
    assert "invalid/failed IK" in message
    assert "fixed roll/pitch/yaw" in message
    assert "Do not bypass controller IK" in message
