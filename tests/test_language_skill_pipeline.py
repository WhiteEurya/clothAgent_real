from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from cloth_agent.config import ExperimentConfig, RobotConfig, WorkspaceBounds
from cloth_agent.language_skill_pipeline import (
    ClaudeSkillAuthor,
    LanguageSkillPipeline,
    LanguageSkillPipelineError,
    SkillDraft,
    _validate_actions,
    actions_to_source,
)
from cloth_agent.session import AgentSession


def _robot_config() -> RobotConfig:
    return RobotConfig(
        robot_ip="127.0.0.1",
        boundaries=WorkspaceBounds(
            x_min=0,
            x_max=800,
            y_min=-400,
            y_max=400,
            z_min=10,
            z_max=400,
        ),
        init_joints_deg=(0, 0, 0, 0, 0, 0, 0),
        init_pose_mm_deg=(500, 0, 180, 180, 0, 0),
        orientation_roll_deg=180,
        orientation_pitch_deg=0,
    )


def _skill() -> SkillDraft:
    return SkillDraft.from_payload(
        {
            "name": "grasp-sleeve-outward",
            "description": "Use for moving a visually identified sleeve outward on the table.",
            "instructions": [
                "Inspect both camera views and identify the requested sleeve with uncertainty stated explicitly.",
                "Ground one reachable grasp location and choose a collision-free outward direction.",
                "Approach, grasp, lift minimally, move outward, lower, and release.",
            ],
            "molmo_prompt": "Point to the visible sleeve requested by the instruction.",
            "success_criteria": ["The selected sleeve finishes farther from the garment center."],
            "safety_constraints": ["Keep every waypoint inside the calibrated workspace."],
        }
    )


def _plan() -> dict[str, object]:
    return {
        "skill_name": "grasp-sleeve-outward",
        "interpretation": "Grasp the visible sleeve edge and translate it toward negative Y.",
        "confidence": 0.73,
        "actions": [
            {"name": "home", "args": {}},
            {"name": "open_gripper", "args": {}},
            {"name": "move", "args": {"x": 500, "y": 0, "z": 100, "yaw": 0}},
            {"name": "move", "args": {"x": 500, "y": 0, "z": 40, "yaw": 0}},
            {"name": "close_gripper", "args": {}},
            {"name": "move", "args": {"x": 500, "y": -120, "z": 90, "yaw": 0}},
            {"name": "move", "args": {"x": 500, "y": -120, "z": 40, "yaw": 0}},
            {"name": "open_gripper", "args": {}},
            {"name": "home", "args": {}},
        ],
        "expected_observation": "The sleeve is extended outward after a controlled low release.",
        "safety_notes": ["Use the calibrated reference and review all waypoints."],
    }


def test_skill_draft_renders_standard_skill_markdown() -> None:
    skill = _skill()
    source = skill.markdown()
    assert source.startswith('---\nname: "grasp-sleeve-outward"\n')
    assert "description:" in source
    assert "## Procedure" in source
    assert "## Runtime contract" in source


def test_execution_plan_is_strict_and_compiles() -> None:
    plan = _validate_actions(_plan(), "grasp-sleeve-outward")
    source = actions_to_source(plan["actions"])
    assert "move(500.0, -120.0, 40.0, 0.0)" in source


def test_execution_plan_requires_release_after_close() -> None:
    payload = _plan()
    payload["actions"] = list(payload["actions"][:-2])
    with pytest.raises(LanguageSkillPipelineError, match="release"):
        _validate_actions(payload, "grasp-sleeve-outward")


def test_execution_plan_normalizes_zero_padded_gripper_args() -> None:
    payload = _plan()
    payload["actions"] = [
        {
            "name": action["name"],
            "args": ({"x": 0, "y": 0, "z": 0, "yaw": 0} if action["name"] == "open_gripper" else action["args"]),
        }
        for action in payload["actions"]
    ]
    normalized = _validate_actions(payload, "grasp-sleeve-outward")
    assert normalized["actions"][1] == {"name": "open_gripper", "args": {}}


def test_execution_plan_rejects_grasp_without_post_grasp_motion() -> None:
    payload = _plan()
    payload["actions"] = [
        {"name": "open_gripper", "args": {}},
        {"name": "move", "args": {"x": 500, "y": 0, "z": 40, "yaw": 0}},
        {"name": "close_gripper", "args": {}},
        {"name": "open_gripper", "args": {}},
    ]
    with pytest.raises(LanguageSkillPipelineError, match="post-grasp move"):
        _validate_actions(payload, "grasp-sleeve-outward")


def test_skill_author_claude_process_is_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}
    response = json.dumps({"structured_output": _skill().as_dict()})
    monkeypatch.setattr(
        "cloth_agent.language_skill_pipeline.shutil.which", lambda _: "/usr/bin/claude"
    )

    def fake_run(command, **kwargs):
        seen.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, stdout=response, stderr="")

    monkeypatch.setattr("cloth_agent.language_skill_pipeline.subprocess.run", fake_run)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill, _ = ClaudeSkillAuthor().author(
        "抓住袖子往外移动", workspace=workspace, project_root=tmp_path
    )
    command = seen["command"]
    assert skill.name == "grasp-sleeve-outward"
    assert seen["shell"] is False
    assert "--json-schema" in command
    assert "--safe-mode" in command
    assert "Bash" not in command
    assert "Write" not in command
    assert "Edit" not in command


def test_pipeline_authors_skill_then_uses_it_in_simulation(tmp_path: Path) -> None:
    session = AgentSession.create(
        tmp_path,
        "language skill test",
        _robot_config(),
        ExperimentConfig(),
        run_id="language_skill_test",
    )
    perception_dir = session.workspace / "perception_views"
    perception_dir.mkdir()
    (perception_dir / "observation.json").write_text(
        json.dumps({"center_base_mm": [500, 0, 40]}), encoding="utf-8"
    )
    Image.new("RGB", (32, 24), "white").save(perception_dir / "camera_0_A.png")
    Image.new("RGB", (32, 24), "white").save(perception_dir / "camera_1_B.png")

    events: list[str] = []

    class Author:
        def author(self, instruction, **kwargs):
            events.append(f"author:{instruction}")
            return _skill(), {"stage": "author"}

    class Executor:
        def execute(self, instruction, skill, **kwargs):
            events.append(f"execute:{skill.name}:{instruction}")
            return _validate_actions(_plan(), skill.name), {"stage": "executor"}

    result = LanguageSkillPipeline(
        session=session,
        instruction="抓住袖子往外移动",
        project_root=tmp_path,
        capture=False,
        use_molmo=False,
        skill_author=Author(),
        skill_executor=Executor(),
    ).run()

    assert events == [
        "author:抓住袖子往外移动",
        "execute:grasp-sleeve-outward:抓住袖子往外移动",
    ]
    assert result["execution_completed"] is True
    assert result["physical_execution"] is False
    skill_path = session.run_dir / result["skill"]
    experiment_path = session.run_dir / result["experiment"]
    assert skill_path.is_file()
    assert experiment_path.is_file()
    saved_execution = json.loads(
        next((session.results / "language_skill_pipeline").glob("*/execution_result.json")).read_text(encoding="utf-8")
    )
    assert saved_execution["physical_execution"] is False
    assert [action["name"] for action in saved_execution["actual_robot_actions"]] == [
        "home",
        "open_gripper",
        "move",
        "move",
        "close_gripper",
        "move",
        "move",
        "open_gripper",
        "home",
    ]
