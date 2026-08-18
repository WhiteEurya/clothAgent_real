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
    ExplorationTimeoutError,
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


def test_free_exploration_allows_minimal_anchor_test_before_release():
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
    proposal = validate_exploration_payload(payload)
    assert proposal.actions[-1]["name"] == "open_gripper"
    assert len([action for action in proposal.actions if action["name"] == "move"]) == 2


def test_free_exploration_requires_test_motion_before_release():
    payload = {
        "garment_observation": "uncertain region",
        "reveal_strategy": "touch and immediately release",
        "confidence": 0.3,
        "actions": [
            {"name": "move", "args": {"x": 500, "y": 0, "z": 100, "yaw": 0}},
            {"name": "close_gripper", "args": {}},
            {"name": "open_gripper", "args": {}},
            {"name": "move", "args": {"x": 500, "y": 0, "z": 180, "yaw": 0}},
        ],
        "expected_observation": "none",
        "safety_notes": ["review"],
    }
    with pytest.raises(ExplorationPlanningError, match="before release"):
        validate_exploration_payload(payload)


def test_free_exploration_accepts_laydown_skill_without_hidden_trajectory():
    proposal = validate_exploration_payload(
        {
            "garment_observation": "A broad section appears suspended from an uncertain boundary.",
            "reveal_strategy": "Use a quasi-static laydown from the current useful anchor.",
            "confidence": 0.7,
            "skill_invocations": [
                {"name": "laydown", "reason": "The grasp supports a useful hanging sheet."}
            ],
            "actions": [
                {"name": "move", "args": {"x": 500, "y": 0, "z": 100, "yaw": 0}},
                {"name": "close_gripper", "args": {}},
                {"name": "move", "args": {"x": 540, "y": 0, "z": 150, "yaw": 0}},
                {"name": "move", "args": {"x": 580, "y": 0, "z": 80, "yaw": 0}},
                {"name": "open_gripper", "args": {}},
            ],
            "expected_observation": "The hanging surface should settle progressively onto the table.",
            "safety_notes": ["Use controller IK and avoid a high drop."],
        }
    )
    assert proposal.skill_invocations == (
        {"name": "laydown", "reason": "The grasp supports a useful hanging sheet."},
    )
    source = exploration_source(proposal)
    assert "laydown" not in source
    assert "move(580.0, 0.0, 80.0, 0.0)" in source


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


def test_claude_json_extractor_prefers_structured_output():
    payload = {"status": "ok"}
    wrapped = json.dumps(
        {"result": "natural-language summary", "structured_output": payload}
    )
    assert _json_from_claude_text(wrapped) == payload


def test_exploration_client_is_read_only_and_logs_proposal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "run"
    images = run_dir / "results" / "perception"
    images.mkdir(parents=True)
    image = images / "camera_A.png"
    image.write_bytes(b"image")
    (run_dir / "workspace" / "perception_views").mkdir(parents=True)
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
    result = ClaudeExplorationClient().invoke(
        [image],
        exploration_prompt(ExperimentConfig(500, 0, 40, None, None, None), _robot_config()),
        run_dir,
    )
    assert result.proposal.garment_observation == "fold"
    assert seen["cwd"] == run_dir.resolve()
    assert seen["shell"] is False
    assert "Bash" not in seen["command"]
    assert "Write" not in seen["command"]
    assert "--strict-mcp-config" in seen["command"]
    assert "--safe-mode" not in seen["command"]
    permission_index = seen["command"].index("--permission-mode")
    assert seen["command"][permission_index + 1] == "dontAsk"
    tools_index = seen["command"].index("--tools")
    assert seen["command"][tools_index + 1] == "Read"
    assert any("mcp__garment_grounding__lookup_reference" in str(part) for part in seen["command"])
    assert not any("mcp__garment_grounding__sample_pixel_xyz" in str(part) for part in seen["command"])
    assert any("call `lookup_reference` exactly once" in str(part) for part in seen["command"])
    assert any("as open and spread" in str(part) for part in seen["command"])
    assert any("camera_*_coordinate_guide.json" in str(part) for part in seen["command"])
    assert any("Skill: laydown" in str(part) for part in seen["command"])
    assert list((run_dir / "results" / "claude_exploration").glob("*.json"))


def test_exploration_timeout_is_concise_and_not_wrapped_as_generic_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "run"
    image_dir = run_dir / "results" / "perception"
    image_dir.mkdir(parents=True)
    image = image_dir / "camera_A.png"
    image.write_bytes(b"image")
    (run_dir / "workspace" / "perception_views").mkdir(parents=True)
    monkeypatch.setattr(
        "cloth_agent.free_exploration.shutil.which", lambda _: "/usr/bin/claude"
    )

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("cloth_agent.free_exploration.subprocess.run", fake_run)
    with pytest.raises(
        ExplorationTimeoutError,
        match="timed out after 400 seconds",
    ) as caught:
        ClaudeExplorationClient(timeout_s=400).invoke(
            [image],
            exploration_prompt(
                ExperimentConfig(500, 0, 40, None, None, None), _robot_config()
            ),
            run_dir,
        )
    assert "--print" not in str(caught.value)
    failed_log = next(
        (run_dir / "results" / "claude_exploration").glob("*_failed.json")
    )
    payload = json.loads(failed_log.read_text(encoding="utf-8"))
    assert payload["error"] == (
        "ExplorationTimeoutError: Claude exploration timed out after 400 seconds"
    )


def test_exploration_prompt_surfaces_capabilities():
    prompt = exploration_prompt(
        ExperimentConfig(500, -20, 40, 100, 200, 0), _robot_config()
    )
    assert "move(x,y,z,yaw)" in prompt
    assert "workspace bounds" in prompt
    assert "usable garment lifting anchor" in prompt
    assert "as open and spread" in prompt
    assert "semantic garment part" in prompt
    assert "center_is_reference_only" in prompt
    assert "Skill: laydown" in prompt


def test_exploration_prompt_makes_all_motion_heights_agent_decisions():
    prompt = exploration_prompt(
        ExperimentConfig(500, -20, 40, 100, 200, 0), _robot_config()
    )
    assert "choose the approach height, grasp height" in prompt
    assert "direct opening maneuver" in prompt
    assert "not a required grasp target" in prompt


def test_exploration_prompt_includes_previous_physical_outcomes():
    prompt = exploration_prompt(
        ExperimentConfig(500, -20, 40, 100, 200, 0),
        _robot_config(),
        history=[
            {
                "iteration": 1,
                "proposal": {"reveal_strategy": "test central fold"},
                "evaluation": {"observed_change": "whole pile translated"},
            }
        ],
    )
    assert "whole pile translated" in prompt
    assert "rather than restarting" in prompt


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
