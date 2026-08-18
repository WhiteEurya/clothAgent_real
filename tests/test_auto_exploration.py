from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from cloth_agent.auto_exploration import (
    AutoExplorationError,
    CameraAWebMonitor,
    ClaudeAutoClient,
    ClaudeVisualPlanResult,
    ExplorationEvaluation,
    VisualPlanDecision,
    _depth_preview,
    _is_preexecution_replan_error,
    _json_default,
    _planning_mode_from_history,
    grasp_targets_from_actions,
    prepare_rollout_video_evidence,
    target_overlay_image,
    validate_evaluation_payload,
    validate_visual_plan_payload,
)
from cloth_agent.config import ExperimentConfig, RobotConfig, WorkspaceBounds
from cloth_agent.free_exploration import (
    ClaudeExplorationResult,
    ExplorationPlanningError,
    ExplorationTimeoutError,
    validate_exploration_payload,
)


def _stage_evaluation_payload() -> dict:
    return {
        "target_selection": {
            "status": "SUPPORTED",
            "confidence": 0.8,
            "evidence": ["The selected fold edge moved."],
        },
        "grasp_acquisition": {
            "status": "SUCCESS",
            "confidence": 0.7,
            "evidence": ["Cloth moved with the gripper."],
        },
        "target_structure_acquired": {
            "status": "SUPPORTED",
            "confidence": 0.75,
            "evidence": ["The sleeve layer moved instead of only a local wrinkle."],
        },
        "transport": {
            "status": "INSUFFICIENT",
            "confidence": 0.85,
            "evidence": ["The fold edge moved only a short distance."],
        },
        "laydown": {
            "status": "SUCCESS",
            "confidence": 0.65,
            "evidence": ["The moved layer remained on the table."],
        },
        "task_progress": {
            "status": "IMPROVED",
            "confidence": 0.8,
            "metrics": {
                "visible_area_delta": "INCREASED",
                "overlap_delta": "DECREASED",
                "relief_delta": "UNCHANGED",
                "boundary_change": "The selected fold boundary shifted outward.",
            },
        },
        "earliest_failure_stage": "TRANSPORT",
        "next_experiment": {
            "keep": ["grasp_anchor", "grasp_depth"],
            "change": ["pull_direction", "pull_distance"],
            "reason": (
                "Acquisition and target-layer motion were supported; transport was insufficient."
            ),
        },
    }


def test_evaluation_contract_is_strict():
    evaluation = validate_evaluation_payload(_stage_evaluation_payload())
    assert isinstance(evaluation, ExplorationEvaluation)
    assert evaluation.useful is True
    assert evaluation.stop is False
    assert evaluation.transport.status == "INSUFFICIENT"
    assert evaluation.task_progress.confidence == pytest.approx(0.8)
    assert evaluation.next_experiment.keep == ("grasp_anchor", "grasp_depth")
    assert "useful" not in evaluation.as_dict()


def test_planning_mode_starts_as_exploration_and_expands_after_validation():
    mode, instruction = _planning_mode_from_history([])
    assert mode == "EXPLORATION"
    assert "10–30 mm" in instruction

    evaluation = _stage_evaluation_payload()
    mode, instruction = _planning_mode_from_history([{"evaluation": evaluation}])
    assert mode == "VALIDATED_EXPANSION"
    assert "validated grasp anchor" in instruction


def test_planning_mode_corrects_transport_without_reprobing():
    evaluation = _stage_evaluation_payload()
    evaluation["transport"] = {
        "status": "BAD_DIRECTION",
        "confidence": 0.8,
        "evidence": ["The fold edge moved opposite the intended outward direction."],
    }
    mode, instruction = _planning_mode_from_history([{"evaluation": evaluation}])
    assert mode == "VALIDATED_TRANSPORT_CORRECTION"
    assert "do not repeat the same short motion" in instruction


def test_visual_plan_contract_selects_one_camera_reference():
    decision = validate_visual_plan_payload(
        {
            "garment_observation": "One overlapping boundary is visible.",
            "opening_strategy": "Lift the selected boundary and lay it down outward.",
            "confidence": 0.7,
            "selected_reference": {
                "camera": "a",
                "reference_id": "r026",
                "reason": "It lies on the visually selected raised boundary.",
            },
            "motion_intent": "Approach vertically, lift, retreat, descend, release.",
            "expected_observation": "The overlap should open.",
            "safety_notes": ["Use a shallow grasp and low release."],
        }
    )
    assert decision.selected_reference["camera"] == "A"
    assert decision.selected_reference["reference_id"] == "R026"


def test_auto_plan_runs_visual_then_final_grounding_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    decision = VisualPlanDecision(
        garment_observation="fold",
        opening_strategy="lift and lay down",
        confidence=0.7,
        selected_reference={
            "camera": "A",
            "reference_id": "R026",
            "reason": "raised boundary",
        },
        motion_intent="approach, lift, retreat, low release",
        expected_observation="more spread",
        safety_notes=("stay inside bounds",),
    )
    visual_result = ClaudeVisualPlanResult(
        prompt="visual",
        command=("claude",),
        returncode=0,
        stdout="{}",
        stderr="",
        created_at="now",
        duration_s=1.0,
        decision=decision,
    )
    proposal = validate_exploration_payload(
        {
            "garment_observation": "fold",
            "reveal_strategy": "ground R026 then lift",
            "confidence": 0.7,
            "actions": [
                {"name": "open_gripper", "args": {}},
                {"name": "move", "args": {"x": 522.1, "y": -197.1, "z": 80, "yaw": 75}},
                {"name": "move", "args": {"x": 522.1, "y": -197.1, "z": 12, "yaw": 75}},
                {"name": "close_gripper", "args": {}},
                {"name": "move", "args": {"x": 522.1, "y": -197.1, "z": 150, "yaw": 75}},
                {"name": "open_gripper", "args": {}},
            ],
            "expected_observation": "more spread",
            "safety_notes": ["review"],
        }
    )
    final_result = ClaudeExplorationResult(
        prompt="ground",
        command=("claude",),
        returncode=0,
        stdout="{}",
        stderr="",
        created_at="now",
        proposal=proposal,
    )
    client = ClaudeAutoClient(timeout_s=400, grounding_timeout_s=120)
    seen = {}

    def fake_visual_plan(image_paths, base_prompt, run_dir):
        seen["visual_prompt"] = base_prompt
        return visual_result

    def fake_ground_final_plan(visual, session, objective, history=None):
        seen["grounding_history"] = list(history or [])
        return final_result

    monkeypatch.setattr(client, "_visual_plan", fake_visual_plan)
    monkeypatch.setattr(client, "_ground_final_plan", fake_ground_final_plan)
    monkeypatch.setattr(
        client,
        "_validate_reference_for_stage2",
        lambda *args, **kwargs: {"base_xyz_mm": [522.1, -197.1, 19.8]},
    )
    robot = RobotConfig(
        robot_ip="127.0.0.1",
        boundaries=WorkspaceBounds(
            x_min=300, x_max=900, y_min=-400, y_max=300, z_min=5, z_max=350
        ),
        init_joints_deg=(0, 0, 0, 0, 0, 0, 0),
        init_pose_mm_deg=(500, 0, 280, 180, 0, 0),
        orientation_roll_deg=180,
        orientation_pitch_deg=0,
    )
    session = SimpleNamespace(
        experiment_config=ExperimentConfig(570, -40, 20, None, None, None),
        robot_config=robot,
        run_dir=tmp_path,
    )
    events = []
    result = client.plan(
        [],
        session,  # type: ignore[arg-type]
        "open garment",
        phase_callback=lambda phase, event, value: events.append((phase, event)),
        reference_policy="molmo_confidence_filtered_keypoints",
    )
    assert result == proposal
    assert events == [
        ("visual_planning", "started"),
        ("visual_planning", "completed"),
        ("final_grounding", "started"),
        ("final_grounding", "completed"),
    ]
    assert client.last_visual_plan_result == visual_result
    assert client.last_plan_result == final_result
    assert "zero-shot visual stage" in seen["visual_prompt"]
    assert "Previous physical outcomes" not in seen["visual_prompt"]
    assert "workspace bounds" not in seen["visual_prompt"].lower()
    assert "only allowed grasp references" in seen["visual_prompt"]
    assert "below-threshold/rejected keypoint" in seen["visual_prompt"]
    assert seen["grounding_history"] == []
    assert set(client.last_plan_timing) == {
        "visual_planning_s",
        "visual_planning_attempts",
        "visual_reselection_count",
        "final_grounding_s",
        "total_planning_s",
    }


def test_unexecutable_reference_returns_to_stage_one_before_grounding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    decisions = [
        VisualPlanDecision(
            garment_observation="fold",
            opening_strategy="pull right edge",
            confidence=0.6,
            selected_reference={
                "camera": "A",
                "reference_id": "R020",
                "reason": "visible overlap",
            },
            motion_intent="lift and lay down",
            expected_observation="spread",
            safety_notes=("review",),
        ),
        VisualPlanDecision(
            garment_observation="fold",
            opening_strategy="pull inner edge",
            confidence=0.7,
            selected_reference={
                "camera": "A",
                "reference_id": "R026",
                "reason": "different visible overlap",
            },
            motion_intent="lift and lay down",
            expected_observation="spread",
            safety_notes=("review",),
        ),
    ]
    visual_prompts = []

    def fake_visual_plan(image_paths, base_prompt, run_dir):
        visual_prompts.append(base_prompt)
        decision = decisions[len(visual_prompts) - 1]
        return ClaudeVisualPlanResult(
            prompt=base_prompt,
            command=("claude",),
            returncode=0,
            stdout="{}",
            stderr="",
            created_at="now",
            duration_s=1.0,
            decision=decision,
        )

    proposal = validate_exploration_payload(
        {
            "garment_observation": "fold",
            "reveal_strategy": "use R026",
            "confidence": 0.7,
            "actions": [
                {"name": "open_gripper", "args": {}},
                {
                    "name": "move",
                    "args": {"x": 522.1, "y": -197.1, "z": 80, "yaw": 75},
                },
                {
                    "name": "move",
                    "args": {"x": 522.1, "y": -197.1, "z": 12, "yaw": 75},
                },
                {"name": "close_gripper", "args": {}},
                {
                    "name": "move",
                    "args": {"x": 522.1, "y": -197.1, "z": 120, "yaw": 75},
                },
                {"name": "open_gripper", "args": {}},
            ],
            "expected_observation": "spread",
            "safety_notes": ["review"],
        }
    )
    final_result = ClaudeExplorationResult(
        prompt="ground",
        command=("claude",),
        returncode=0,
        stdout="{}",
        stderr="",
        created_at="now",
        proposal=proposal,
    )
    grounded_references = []

    def fake_ground_final_plan(visual, session, objective, history=None):
        grounded_references.append(visual.selected_reference["reference_id"])
        return final_result

    perception_dir = tmp_path / "workspace" / "perception_views"
    perception_dir.mkdir(parents=True)
    (perception_dir / "camera_A_coordinate_guide.json").write_text(
        json.dumps(
            {
                "coordinate_frame": "robot_base_mm",
                "reference_semantics": "not candidates",
                "samples": [
                    {
                        "reference_id": "R020",
                        "pixel_xy": [407, 216],
                        "base_xyz_mm": [557.27, 173.745, 13.67],
                        "height_above_table_mm": 0.76,
                    },
                    {
                        "reference_id": "R026",
                        "pixel_xy": [168, 263],
                        "base_xyz_mm": [522.1, -197.1, 19.8],
                        "height_above_table_mm": 15.1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    robot = RobotConfig(
        robot_ip="127.0.0.1",
        boundaries=WorkspaceBounds(
            x_min=363.595,
            x_max=None,
            y_min=-303.571,
            y_max=171.087,
            z_min=6.65,
            z_max=333.919,
        ),
        init_joints_deg=(0, 0, 0, 0, 0, 0, 0),
        init_pose_mm_deg=(500, 0, 280, 180, 0, 0),
        orientation_roll_deg=180,
        orientation_pitch_deg=0,
    )
    session = SimpleNamespace(
        experiment_config=ExperimentConfig(570, -40, 20, None, None, None),
        robot_config=robot,
        run_dir=tmp_path,
    )
    client = ClaudeAutoClient(max_reference_reselections=2)
    monkeypatch.setattr(client, "_visual_plan", fake_visual_plan)
    monkeypatch.setattr(client, "_ground_final_plan", fake_ground_final_plan)
    events = []

    result = client.plan(
        [],
        session,  # type: ignore[arg-type]
        "open garment",
        phase_callback=lambda phase, event, value: events.append((phase, event)),
    )

    assert result == proposal
    assert grounded_references == ["R026"]
    assert len(visual_prompts) == 2
    assert "A/R020" not in visual_prompts[0]
    assert "A/R020" in visual_prompts[1]
    assert client.last_visual_plan_result is not None
    assert (
        client.last_visual_plan_result.decision.selected_reference["reference_id"]
        == "R026"
    )
    assert client.last_plan_timing["visual_planning_attempts"] == 2
    assert client.last_plan_timing["visual_reselection_count"] == 1
    assert client.last_rejected_visual_references[0]["reference_id"] == "R020"
    assert events == [
        ("visual_planning", "started"),
        ("visual_planning", "reselecting"),
        ("visual_planning", "started"),
        ("visual_planning", "completed"),
        ("final_grounding", "started"),
        ("final_grounding", "completed"),
    ]


def test_visual_stage_uses_original_read_only_safe_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image = tmp_path / "camera_A.png"
    image.write_bytes(b"image")
    payload = {
        "garment_observation": "fold",
        "opening_strategy": "lift boundary",
        "confidence": 0.6,
        "selected_reference": {
            "camera": "A",
            "reference_id": "R026",
            "reason": "visible raised edge",
        },
        "motion_intent": "approach, lift, lay down, release",
        "expected_observation": "more spread",
        "safety_notes": ["avoid fixture"],
    }
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"result": json.dumps(payload)}),
            stderr="",
        )

    client = ClaudeAutoClient(binary="/usr/bin/claude", timeout_s=400)
    monkeypatch.setattr("cloth_agent.auto_exploration.subprocess.run", fake_run)
    result = client._visual_plan([image], "base visual prompt", tmp_path)
    assert result.decision.selected_reference["reference_id"] == "R026"
    assert "--safe-mode" in seen["command"]
    assert "--mcp-config" not in seen["command"]
    assert seen["command"][seen["command"].index("--permission-mode") + 1] == "plan"
    assert seen["timeout"] == 400


def test_final_stage_uses_one_lookup_and_short_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    perception_dir = tmp_path / "workspace" / "perception_views"
    perception_dir.mkdir(parents=True)
    (perception_dir / "camera_A_coordinate_guide.json").write_text(
        json.dumps(
            {
                "coordinate_frame": "robot_base_mm",
                "reference_semantics": "not candidates",
                "samples": [
                    {
                        "reference_id": "R026",
                        "pixel_xy": [168, 263],
                        "base_xyz_mm": [522.1, -197.1, 19.8],
                        "height_above_table_mm": 15.1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    final_payload = {
        "garment_observation": "fold",
        "reveal_strategy": "use the selected R026",
        "confidence": 0.7,
        "actions": [
            {"name": "open_gripper", "args": {}},
            {"name": "move", "args": {"x": 522.1, "y": -197.1, "z": 80, "yaw": 75}},
            {"name": "move", "args": {"x": 522.1, "y": -197.1, "z": 12, "yaw": 75}},
            {"name": "close_gripper", "args": {}},
            {"name": "move", "args": {"x": 522.1, "y": -197.1, "z": 150, "yaw": 75}},
            {"name": "open_gripper", "args": {}},
        ],
        "expected_observation": "more spread",
        "safety_notes": ["review"],
    }
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"result": json.dumps(final_payload)}),
            stderr="",
        )

    client = ClaudeAutoClient(
        binary="/usr/bin/claude", timeout_s=400, grounding_timeout_s=120
    )
    monkeypatch.setattr("cloth_agent.auto_exploration.subprocess.run", fake_run)
    decision = VisualPlanDecision(
        garment_observation="fold",
        opening_strategy="lift",
        confidence=0.7,
        selected_reference={
            "camera": "A",
            "reference_id": "R026",
            "reason": "edge",
        },
        motion_intent="lift and lay down",
        expected_observation="spread",
        safety_notes=("review",),
    )
    robot = RobotConfig(
        robot_ip="127.0.0.1",
        boundaries=WorkspaceBounds(
            x_min=300, x_max=900, y_min=-400, y_max=300, z_min=5, z_max=350
        ),
        init_joints_deg=(0, 0, 0, 0, 0, 0, 0),
        init_pose_mm_deg=(500, 0, 280, 180, 0, 0),
        orientation_roll_deg=180,
        orientation_pitch_deg=0,
    )
    session = SimpleNamespace(
        experiment_config=ExperimentConfig(570, -40, 20, None, None, None),
        robot_config=robot,
        run_dir=tmp_path,
    )
    result = client._ground_final_plan(
        decision, session, "open garment"  # type: ignore[arg-type]
    )
    assert result.proposal.actions[2]["args"]["x"] == pytest.approx(522.1)
    assert "--mcp-config" in seen["command"]
    assert "--safe-mode" not in seen["command"]
    assert seen["command"][seen["command"].index("--permission-mode") + 1] == "dontAsk"
    assert seen["timeout"] == 120
    allowed = seen["command"][seen["command"].index("--allowedTools") + 1]
    assert allowed == "mcp__garment_grounding__lookup_reference"
    assert seen["command"][seen["command"].index("--tools") + 1] == ""


def test_evaluation_empty_change_list_means_stop():
    payload = _stage_evaluation_payload()
    payload["next_experiment"] = {
        "keep": ["current_spread_state"],
        "change": [],
        "reason": "No further safe grounded opening action is visible.",
    }
    evaluation = validate_evaluation_payload(payload)
    assert evaluation.stop is True
    assert evaluation.next_objective.startswith("Stop:")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {
            "useful": "yes",
            "confidence": 0.8,
            "observed_change": "x",
            "next_objective": "y",
            "stop": False,
            "reason": "z",
        },
        {**_stage_evaluation_payload(), "earliest_failure_stage": "GRASP"},
    ],
)
def test_evaluation_contract_rejects_hard_failures(payload):
    with pytest.raises(AutoExplorationError):
        validate_evaluation_payload(payload)


def test_evaluation_rejects_keep_change_overlap():
    payload = _stage_evaluation_payload()
    payload["next_experiment"]["change"].append("grasp_anchor")
    with pytest.raises(AutoExplorationError, match="both keep and change"):
        validate_evaluation_payload(payload)


def test_evaluator_uses_native_json_schema_and_structured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _stage_evaluation_payload()
    seen: dict[str, object] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "result": "ignored prose",
                    "structured_output": payload,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    before.write_bytes(b"image")
    after.write_bytes(b"image")
    proposal = validate_exploration_payload(
        {
            "garment_observation": "fold",
            "reveal_strategy": "lift and pull",
            "confidence": 0.7,
            "actions": [
                {"name": "move", "args": {"x": 500, "y": 0, "z": 20, "yaw": 0}},
                {"name": "close_gripper", "args": {}},
                {"name": "move", "args": {"x": 540, "y": 0, "z": 60, "yaw": 0}},
                {"name": "open_gripper", "args": {}},
            ],
            "expected_observation": "less overlap",
            "safety_notes": ["stay in bounds"],
        }
    )
    client = ClaudeAutoClient(binary="/usr/bin/claude")

    evaluation = client.evaluate(
        [before], [after], proposal=proposal, run_dir=tmp_path
    )

    command = seen["command"]
    assert isinstance(command, list)
    assert "--json-schema" in command
    schema = json.loads(command[command.index("--json-schema") + 1])
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(payload)
    assert evaluation.transport.status == "INSUFFICIENT"


def test_rollout_video_evidence_extracts_camera_contact_sheets(tmp_path: Path):
    import cv2
    import numpy as np

    cameras = []
    for label, color in (("A", (20, 40, 200)), ("B", (180, 60, 20))):
        video_path = tmp_path / f"camera_{label}_rgb.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (64, 48)
        )
        assert writer.isOpened()
        for index in range(8):
            frame = np.full((48, 64, 3), color, dtype=np.uint8)
            frame[:, : index * 8] = 255
            writer.write(frame)
        writer.release()
        cameras.append({"label": label, "rgb_video": video_path.name})
    (tmp_path / "recording_manifest.json").write_text(
        json.dumps({"cameras": cameras}), encoding="utf-8"
    )

    sheets, references, errors = prepare_rollout_video_evidence(tmp_path)

    assert errors == []
    assert len(sheets) == 2
    assert len(references) == 2
    assert all(path.is_file() for path in sheets)
    manifest = json.loads(
        (tmp_path / "evaluator_video_evidence" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(manifest["items"]) == 2


def test_auto_module_requires_explicit_real_flag(tmp_path: Path):
    from cloth_agent.auto_exploration import run_auto_exploration_viewer

    with pytest.raises(PermissionError, match="real-execution only"):
        run_auto_exploration_viewer(  # type: ignore[arg-type]
            None,
            enable_real=False,
        )


def test_web_camera_monitor_uses_dedicated_viser_label():
    from cloth_agent.perception import CameraSpec, MolmoConfig, PerceptionConfig

    dummy = Path("/tmp/cam_a_extrinsics.yaml")
    spec = CameraSpec("A", "serial-A", dummy)
    config = PerceptionConfig(
        cameras=(spec, CameraSpec("B", "serial-B", dummy)),
        molmo=MolmoConfig(Path("/bin/true")),
    )
    monitor = CameraAWebMonitor(
        Path("/tmp/project"),
        Path("/tmp/project/perception.json"),
        spec,
        config,
        lambda: None,
    )
    assert monitor.window_name == "CamA live monitor (serial-A)"


def test_depth_preview_is_rgb_grayscale_and_masks_invalid_depth():
    import numpy as np

    preview = _depth_preview(
        np.asarray([[0.1, 0.5, 1.0, 2.1]], dtype=np.float32),
        min_depth_m=0.15,
        max_depth_m=2.0,
    )
    assert preview.shape == (1, 4, 3)
    assert preview.dtype == np.uint8
    assert np.all(preview[0, 0] == 0)
    assert np.all(preview[0, 3] == 0)
    assert int(preview[0, 1, 0]) > int(preview[0, 2, 0])


def test_continuous_iteration_defaults_and_json_serialization():
    import numpy as np
    from pathlib import Path

    assert _json_default(Path("results/x.json")) == "results/x.json"
    assert _json_default(np.asarray([1, 2])) == [1, 2]


def test_timeout_is_never_a_preexecution_replan_error():
    assert _is_preexecution_replan_error(
        ExplorationTimeoutError("Claude exploration timed out after 400 seconds")
    ) is False
    assert _is_preexecution_replan_error(
        ExplorationPlanningError("invalid proposal schema")
    ) is True


def test_web_monitor_serializes_duplicate_camera_stop_calls():
    from cloth_agent.perception import CameraSpec, MolmoConfig, PerceptionConfig

    dummy = Path("/tmp/cam_a_extrinsics.yaml")
    spec = CameraSpec("A", "serial-A", dummy)
    config = PerceptionConfig(
        cameras=(spec, CameraSpec("B", "serial-B", dummy)),
        molmo=MolmoConfig(Path("/bin/true")),
    )
    monitor = CameraAWebMonitor(
        Path("/tmp/project"),
        Path("/tmp/project/perception.json"),
        spec,
        config,
        lambda: None,
    )

    class FakeCamera:
        def __init__(self):
            self.started = True
            self.stop_calls = 0

        def stop(self):
            if not self.started:
                return
            self.started = False
            self.stop_calls += 1

    camera = FakeCamera()
    threads = [
        threading.Thread(target=monitor._stop_camera, args=(camera,))  # type: ignore[arg-type]
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert camera.stop_calls == 1


def test_grasp_targets_use_last_move_before_each_close():
    targets = grasp_targets_from_actions(
        [
            {"name": "home", "args": {}},
            {"name": "move", "args": {"x": 500, "y": -20, "z": 100, "yaw": 10}},
            {"name": "move", "args": {"x": 510, "y": -25, "z": 12, "yaw": 15}},
            {"name": "close_gripper", "args": {}},
            {"name": "open_gripper", "args": {}},
            {"name": "move", "args": {"x": 540, "y": -80, "z": 14, "yaw": 40}},
            {"name": "close_gripper", "args": {}},
        ]
    )
    assert [(target["x"], target["y"], target["z"]) for target in targets] == [
        (510.0, -25.0, 12.0),
        (540.0, -80.0, 14.0),
    ]
    assert targets[0]["move_action_index"] == 3
    assert targets[0]["close_action_index"] == 4


def test_grasp_target_is_unknown_without_grounded_move():
    assert grasp_targets_from_actions([{"name": "close_gripper", "args": {}}]) == []
    assert grasp_targets_from_actions(
        [
            {"name": "move", "args": {"x": 1, "y": 2, "z": 3, "yaw": 4}},
            {"name": "home", "args": {}},
            {"name": "close_gripper", "args": {}},
        ]
    ) == []


def test_target_overlay_projects_base_target_into_rgb_frame():
    import numpy as np

    from cloth_agent.perception import RGBDFrame

    frame = RGBDFrame(
        label="A",
        serial="test",
        rgb=np.zeros((80, 100, 3), dtype=np.uint8),
        depth_m=np.ones((80, 100), dtype=np.float32),
        intrinsics=np.asarray(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
        ),
        X_base_camera=np.eye(4),
    )
    targets = [
        {
            "target_index": 1,
            "move_action_index": 2,
            "close_action_index": 3,
            "x": 0.0,
            "y": 0.0,
            "z": 1000.0,
            "yaw": 0.0,
        }
    ]
    overlay, projections = target_overlay_image(frame, targets)
    assert overlay.shape == frame.rgb.shape
    assert projections[0]["visible"] is True
    assert projections[0]["pixel"] == pytest.approx([50.0, 40.0])
    assert np.any(overlay != 0)
