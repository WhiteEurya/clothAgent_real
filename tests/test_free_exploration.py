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
    ground_global_grasp_target,
    _voxel_balance_cloud,
    validate_global_probe_profile,
    validate_global_exploration_payload,
    validate_global_grasp_grounding,
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


def test_exploration_requires_post_grasp_lift_before_probe():
    payload = {
        "garment_observation": "A narrow raised ridge may be a rolled wrinkle.",
        "reveal_strategy": "Lift and hold before any short probe.",
        "confidence": 0.5,
        "actions": [
            {"name": "move", "args": {"x": 500, "y": 0, "z": 20, "yaw": 0}},
            {"name": "close_gripper", "args": {}},
            {"name": "move", "args": {"x": 520, "y": 0, "z": 20, "yaw": 0}},
            {"name": "open_gripper", "args": {}},
        ],
        "expected_observation": "The ridge either forms a hanging patch or is released.",
        "safety_notes": ["Do not commit to a long pull without a hold check."],
    }
    with pytest.raises(ExplorationPlanningError, match="first post-grasp move must lift"):
        validate_exploration_payload(payload)


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


def test_global_proposal_grounds_arbitrary_pixel_and_checks_grasp_xy(
    tmp_path: Path,
) -> None:
    perception_dir = tmp_path / "perception_views"
    perception_dir.mkdir()
    xyz = np.zeros((5, 6, 3), dtype=np.float32)
    xyz[:, :, 0] = 520.0
    xyz[:, :, 1] = -40.0
    xyz[:, :, 2] = 18.0
    np.save(perception_dir / "camera_A_base_xyz_mm.npy", xyz)
    np.save(
        perception_dir / "camera_A_height_above_table_mm.npy",
        np.full((5, 6), 8.0, dtype=np.float32),
    )
    guide = {
        "samples": [
            {
                "reference_id": "R001",
                "pixel_xy": [2, 2],
                "base_xyz_mm": [520.0, -40.0, 18.0],
                "height_above_table_mm": 8.0,
            }
        ]
    }
    (perception_dir / "camera_A_coordinate_guide.json").write_text(
        json.dumps(guide), encoding="utf-8"
    )
    proposal = validate_global_exploration_payload(
        {
            "selected_grasp": {
                "camera": "A",
                "pixel_xy": [2, 2],
                "reason": "Claude sees a useful free boundary here.",
            },
            "garment_observation": "One raised boundary overlaps the main sheet.",
            "reveal_strategy": "Probe the selected boundary and lift it outward.",
            "confidence": 0.7,
            "actions": [
                {"name": "move", "args": {"x": 520, "y": -40, "z": 60, "yaw": 0}},
                {"name": "move", "args": {"x": 520, "y": -40, "z": 20, "yaw": 0}},
                {"name": "close_gripper", "args": {}},
                {"name": "move", "args": {"x": 530, "y": -40, "z": 40, "yaw": 0}},
                {"name": "open_gripper", "args": {}},
            ],
            "expected_observation": "The selected boundary follows the grasp.",
            "safety_notes": ["Use the runtime workspace and IK gates."],
        }
    )

    result = validate_global_grasp_grounding(proposal, perception_dir)
    assert result["valid"] is True
    assert result["measurement"]["query_pixel_xy"] == [2, 2]
    assert result["xy_error_mm"] == pytest.approx(0.0)

    offset_payload = proposal.as_dict()
    offset_payload["actions"][0]["args"]["x"] = 530.0
    offset_payload["actions"][0]["args"]["y"] = -35.0
    offset_payload["actions"][1]["args"]["x"] = 530.0
    offset_payload["actions"][1]["args"]["y"] = -35.0
    offset_payload["actions"][3]["args"]["x"] = 530.0
    offset_payload["actions"][3]["args"]["y"] = -35.0
    offset = validate_global_exploration_payload(offset_payload)

    grounded, grounding = ground_global_grasp_target(offset, perception_dir)

    assert grounding["claude_requested_grasp_xy_mm"] == pytest.approx([530.0, -35.0])
    assert grounding["commanded_grasp_xy_mm"] == pytest.approx([520.0, -40.0])
    assert grounding["xy_correction_mm"] == pytest.approx(np.hypot(10.0, 5.0))
    assert grounding["post_grounding_xy_error_mm"] == pytest.approx(0.0)
    assert grounding["grounded_action_numbers"] == [1, 2, 4]
    assert [
        (action["args"]["x"], action["args"]["y"])
        for action in grounded.actions
        if action["name"] == "move"
    ] == [(520.0, -40.0), (520.0, -40.0), (520.0, -40.0)]
    assert offset.actions[1]["args"]["x"] == 530.0
    assert validate_global_grasp_grounding(grounded, perception_dir)["valid"] is True


def test_global_probe_profile_caps_unvalidated_lateral_pull(tmp_path: Path):
    proposal = validate_global_exploration_payload(
        {
            "selected_grasp": {
                "camera": "A",
                "pixel_xy": [2, 2],
                "reason": "The ridge is a possible free boundary.",
            },
            "garment_observation": "A narrow ridge may be a rolled wrinkle.",
            "reveal_strategy": "Lift, hold, and make a short probe.",
            "confidence": 0.5,
            "actions": [
                {"name": "move", "args": {"x": 500, "y": 0, "z": 60, "yaw": 0}},
                {"name": "close_gripper", "args": {}},
                {"name": "move", "args": {"x": 500, "y": 0, "z": 80, "yaw": 0}},
                {"name": "move", "args": {"x": 550, "y": 0, "z": 80, "yaw": 0}},
                {"name": "open_gripper", "args": {}},
            ],
            "expected_observation": "A free layer should form a hanging patch.",
            "safety_notes": ["Release if the ridge only curls upward."],
        }
    )
    with pytest.raises(ExplorationPlanningError, match="short lateral probe"):
        validate_global_probe_profile(proposal)


def _compression_probe_measurement(*, surface_z_mm: float = 25.0) -> dict:
    return {
        "base_xyz_median_mm": [500.0, 0.0, surface_z_mm],
        "surface_shape_diagnostic": {
            "surface_shape": "NARROW_RIDGE_OR_SPIKE",
            "compression_probe_recommended": True,
            "recommended_press_below_surface_mm": 1.0,
        },
    }


def test_global_probe_profile_requires_shallow_compression_for_narrow_peak():
    proposal = validate_global_exploration_payload(
        {
            "selected_grasp": {
                "camera": "A",
                "pixel_xy": [2, 2],
                "reason": "A narrow peak may be a separable layer or a rolled wrinkle.",
            },
            "garment_observation": "A narrow peak is visible.",
            "reveal_strategy": "Press shallowly, lift vertically, and inspect the response.",
            "confidence": 0.4,
            "actions": [
                {"name": "move", "args": {"x": 500, "y": 0, "z": 60, "yaw": 0}},
                {"name": "move", "args": {"x": 500, "y": 0, "z": 24, "yaw": 0}},
                {"name": "close_gripper", "args": {}},
                {"name": "move", "args": {"x": 500, "y": 0, "z": 44, "yaw": 0}},
                {"name": "move", "args": {"x": 520, "y": 0, "z": 44, "yaw": 0}},
                {"name": "open_gripper", "args": {}},
            ],
            "expected_observation": "The peak either hangs independently or compresses without separating.",
            "safety_notes": ["Keep compression shallow and lift before probing laterally."],
        }
    )
    result = validate_global_probe_profile(
        proposal,
        measurement=_compression_probe_measurement(),
    )
    assert result["compression_probe_required"] is True
    assert result["compression_probe_surface_shape"] == "NARROW_RIDGE_OR_SPIKE"
    assert result["compression_depth_mm"] == pytest.approx(1.0)


def test_global_probe_profile_rejects_missing_or_deep_compression_press():
    base_payload = {
        "selected_grasp": {
            "camera": "A",
            "pixel_xy": [2, 2],
            "reason": "A narrow peak needs a compression test.",
        },
        "garment_observation": "A narrow peak is visible.",
        "reveal_strategy": "Press shallowly and lift.",
        "confidence": 0.4,
        "actions": [
            {"name": "move", "args": {"x": 500, "y": 0, "z": 60, "yaw": 0}},
            {"name": "move", "args": {"x": 500, "y": 0, "z": 25, "yaw": 0}},
            {"name": "close_gripper", "args": {}},
            {"name": "move", "args": {"x": 500, "y": 0, "z": 44, "yaw": 0}},
            {"name": "open_gripper", "args": {}},
        ],
        "expected_observation": "The peak response is visible.",
        "safety_notes": ["Do not press deeply."],
    }
    proposal = validate_global_exploration_payload(base_payload)
    with pytest.raises(ExplorationPlanningError, match="at or slightly below"):
        validate_global_probe_profile(
            proposal,
            measurement=_compression_probe_measurement(),
        )

    deep_payload = dict(base_payload)
    deep_payload["actions"] = [dict(action) for action in base_payload["actions"]]
    deep_payload["actions"][1] = {
        "name": "move",
        "args": {"x": 500, "y": 0, "z": 20, "yaw": 0},
    }
    deep_proposal = validate_global_exploration_payload(deep_payload)
    with pytest.raises(ExplorationPlanningError, match="too deep"):
        validate_global_probe_profile(
            deep_proposal,
            measurement=_compression_probe_measurement(),
        )


def test_global_proposal_rejects_camera_b_as_action_source() -> None:
    payload = {
        "selected_grasp": {
            "camera": "B",
            "pixel_xy": [320, 240],
            "reason": "Only visible from the secondary view.",
        },
        "garment_observation": "One raised boundary overlaps the main sheet.",
        "reveal_strategy": "Probe the selected boundary.",
        "confidence": 0.5,
        "actions": [
            {"name": "move", "args": {"x": 520, "y": -40, "z": 20, "yaw": 0}},
            {"name": "close_gripper", "args": {}},
            {"name": "move", "args": {"x": 530, "y": -40, "z": 40, "yaw": 0}},
            {"name": "open_gripper", "args": {}},
        ],
        "expected_observation": "The selected boundary follows the grasp.",
        "safety_notes": ["Use the runtime workspace and IK gates."],
    }

    with pytest.raises(ExplorationPlanningError, match="Camera B is observation-only"):
        validate_global_exploration_payload(payload)


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


def test_free_exploration_accepts_flatten_garment_system_skill():
    proposal = validate_exploration_payload(
        {
            "garment_observation": "A supported grasp is visible on raised cloth.",
            "reveal_strategy": "Lift high, move to the far safe X, then retreat while descending.",
            "confidence": 0.8,
            "skill_invocations": [
                {"name": "flatten-garment", "reason": "whole-garment spread is supported."}
            ],
            "actions": [
                {"name": "move", "args": {"x": 500, "y": 0, "z": 100, "yaw": 0}},
                {"name": "close_gripper", "args": {}},
                {"name": "move", "args": {"x": 500, "y": 0, "z": 150, "yaw": 0}},
                {"name": "move", "args": {"x": 400, "y": 0, "z": 30, "yaw": 0}},
                {"name": "open_gripper", "args": {}},
            ],
            "expected_observation": "The footprint should widen and relief should decrease.",
            "safety_notes": ["Use validated workspace and release low."],
        }
    )

    assert proposal.skill_invocations == (
        {"name": "flatten-garment", "reason": "whole-garment spread is supported."},
    )


def test_global_payload_accepts_active_dynamic_skill_name():
    payload = {
        "selected_grasp": {
            "camera": "A",
            "pixel_xy": [3, 2],
            "reason": "supported edge",
        },
        "garment_observation": "supported edge",
        "reveal_strategy": "apply the approved edge-release procedure",
        "confidence": 0.8,
        "skill_invocations": [
            {"name": "edge-release", "reason": "repeated before/after response"}
        ],
        "actions": [
            {"name": "move", "args": {"x": 500, "y": 0, "z": 100, "yaw": 0}},
            {"name": "close_gripper", "args": {}},
            {"name": "move", "args": {"x": 540, "y": 0, "z": 150, "yaw": 0}},
            {"name": "open_gripper", "args": {}},
        ],
        "expected_observation": "edge opens",
        "safety_notes": ["check workspace"],
    }
    proposal = validate_global_exploration_payload(
        payload,
        allowed_skill_names=("laydown", "edge-release"),
    )
    assert proposal.skill_invocations[0]["name"] == "edge-release"


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
        "selected_grasp": {
            "camera": "A",
            "pixel_xy": [320, 240],
            "reason": "visible free boundary",
        },
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
    assert any(
        "mcp__garment_grounding__sample_local_surface" in str(part)
        for part in seen["command"]
    )
    assert not any(
        "mcp__garment_grounding__lookup_reference" in str(part)
        for part in seen["command"]
    )
    context_path = next(
        (run_dir / "workspace" / "claude_planning_contexts").glob("*.md")
    )
    context_text = context_path.read_text(encoding="utf-8")
    assert str(context_path.relative_to(run_dir)) in seen["input"]
    assert "`sample_local_surface` exactly once" in context_text
    assert any(
        "Camera B is observation-only secondary context" in str(part)
        for part in seen["command"]
    )
    assert "runtime owns the precise grasp target" in context_text
    assert "as open and spread" in context_text
    assert "No system-generated grasp candidates" in context_text
    assert "Skill: laydown" in context_text
    assert not any("`sample_local_surface` exactly once" in str(part) for part in seen["command"])
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


def test_exploration_prompt_bounds_accumulated_history():
    history = [
        {"iteration": index, "evaluation": {"reason": "x" * 30_000}}
        for index in range(8)
    ]
    prompt = exploration_prompt(
        ExperimentConfig(500, -20, 40, 100, 200, 0),
        _robot_config(),
        history=history,
    )

    assert len(prompt) < 120_000
    assert '"iteration": 7' in prompt
    assert '"iteration": 0' not in prompt


def test_exploration_prompt_references_persisted_history_instead_of_embedding_it():
    prompt = exploration_prompt(
        ExperimentConfig(500, -20, 40, 100, 200, 0),
        _robot_config(),
        history=[{"iteration": 7, "evaluation": {"reason": "x" * 30_000}}],
        history_file="workspace/global_experience.jsonl",
    )

    assert "workspace/global_experience.jsonl" in prompt
    assert "Use the Read tool" in prompt
    assert "x" * 1_000 not in prompt
    assert "workspace bounds" in prompt
    assert "usable garment lifting anchor" in prompt
    assert "as open and spread" in prompt
    assert "semantic garment part" in prompt
    assert "center_is_reference_only" in prompt
    assert "Skill: laydown" in prompt
    assert "flat-garment reference first" in prompt
    assert "printed-pattern correspondence" in prompt
    assert "heatmap pixel as the target" in prompt


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
