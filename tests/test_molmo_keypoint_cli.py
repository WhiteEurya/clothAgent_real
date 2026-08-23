from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

from cloth_agent.config import ExperimentConfig, RobotConfig, SafetyError, WorkspaceBounds
from cloth_agent.free_exploration import (
    ExplorationTimeoutError,
    validate_exploration_payload,
    validate_global_exploration_payload,
)
from cloth_agent.molmo_keypoint_cli import (
    CliReporter,
    KeypointCliOptions,
    _is_recoverable_loop_error,
    _override_camera_controls,
    _recovery_skill_proposal,
    _planning_images,
    run_keypoint_cli_loop,
)
from cloth_agent.molmo_keypoint_pipeline import (
    KeypointSpec,
    MolmoKeypointPipelineError,
)
from cloth_agent.perception import CameraSpec, PerceptionConfig
from cloth_agent.semantic_claude import SemanticActionResult
from cloth_agent.semantic_pipeline import (
    SemanticStateBuilder,
    validate_semantic_evaluation_payload,
    validate_semantic_strategy_payload,
)


@dataclass
class _Preflight:
    source: str
    actions: list[dict]
    stdout: str = ""
    error: str | None = None


def test_camera_control_overrides_apply_per_launch(tmp_path: Path) -> None:
    config = PerceptionConfig(
        cameras=(
            CameraSpec("A", "a", tmp_path / "a.yaml", 400.0, 3800.0),
            CameraSpec("B", "b", tmp_path / "b.yaml", 400.0, 3800.0),
        )
    )

    updated = _override_camera_controls(
        config,
        camera_a_exposure=300.0,
        camera_b_exposure=350.0,
        camera_b_white_balance=4000.0,
    )

    assert updated.cameras[0].color_exposure == 300.0
    assert updated.cameras[0].color_white_balance == 3800.0
    assert updated.cameras[1].color_exposure == 350.0
    assert updated.cameras[1].color_white_balance == 4000.0
    assert config.cameras[0].color_exposure == 400.0


class _Runner:
    def __init__(self, actions: list[dict]):
        self.actions = actions

    def preflight(self, _: str) -> _Preflight:
        return _Preflight("def run():\n    home()\n", self.actions)


class _Session:
    def __init__(self, tmp_path: Path, actions: list[dict]):
        self.project_root = tmp_path
        self.run_dir = tmp_path / "runs" / "cli_test"
        self.workspace = self.run_dir / "workspace"
        self.results = self.run_dir / "results"
        self.workspace.mkdir(parents=True)
        self.results.mkdir(parents=True)
        (self.workspace / "perception_views").mkdir()
        self.robot_config = RobotConfig(
            robot_ip="127.0.0.1",
            boundaries=WorkspaceBounds(
                x_min=300,
                x_max=900,
                y_min=-400,
                y_max=300,
                z_min=5,
                z_max=350,
            ),
            init_joints_deg=(0, 0, 0, 0, 0, 0, 0),
            init_pose_mm_deg=(500, 0, 280, 180, 0, 0),
            orientation_roll_deg=180,
            orientation_pitch_deg=0,
        )
        self.experiment_config = ExperimentConfig(520.0, -40.0, 18.0, None, None, None)
        self.runner = _Runner(actions)
        self.execution_calls = 0
        self.last_return_home_outcome = None

    def locate_cloth_center(self, config, frames=None):
        return {
            "status": "VALIDATED_DENSE_AB_FUSION",
            "center_base_mm": [520.0, -40.0, 18.0],
        }

    def run_experiment(self, *args, **kwargs):
        self.execution_calls += 1
        return {"execution_completed": True}


def test_planning_images_put_raw_flat_reference_before_geometry(tmp_path: Path) -> None:
    result_dir = tmp_path / "perception"
    result_dir.mkdir()
    result_path = result_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    for name in (
        "camera_A.png",
        "camera_A_garment_only.png",
        "camera_A_height_map_heatmap.png",
    ):
        (result_dir / name).write_bytes(b"image")
    annotation_dir = tmp_path / "iteration" / "semantic_anchors"
    overlay = annotation_dir / "camera_A_semantic_anchors.png"
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"overlay")
    reference_dir = annotation_dir / "flat_reference"
    reference_dir.mkdir()
    (reference_dir / "camera_A_flat_reference.png").write_bytes(b"raw")
    (reference_dir / "camera_A_flat_reference_anchors.png").write_bytes(b"anchors")

    paths = _planning_images(
        {
            "views": [
                {
                    "image": "camera_A.png",
                    "garment_rgb": "camera_A_garment_only.png",
                    "height_map": "camera_A_height_map_heatmap.png",
                }
            ]
        },
        result_path,
        {"views": [{"accepted_overlay": str(overlay)}]},
        tmp_path,
    )

    assert [path.name for path in paths[:2]] == [
        "camera_A_flat_reference.png",
        "camera_A_flat_reference_anchors.png",
    ]
    assert paths[2].name == "camera_A_garment_only.png"


class _Client:
    def __init__(self, proposal):
        self.proposal = proposal
        self.last_strategy_log = None
        self.last_action_log = None
        self.last_evaluation_log = None

    def plan_strategy(self, *, semantic_state, **kwargs):
        strategy = validate_semantic_strategy_payload(
            {
                "semantic_objective": {
                    "target_part": "sleeve_end",
                    "desired_change": "move outward from torso",
                },
                "hypothesis": {
                    "state": "possible_inward_fold",
                    "confidence": 0.72,
                    "rationale": "Anchor lies near the garment centroid.",
                },
                "local_search_region": {
                    "around_anchor_id": "S001",
                    "radius_px": 40,
                    "include_connected_fold_edge": True,
                },
                "grasp_requirement": {
                    "prefer": ["free_edge"],
                    "avoid": ["flat_interior"],
                },
                "expected_semantic_observation": "Sleeve-associated edge moves outward.",
                "safety_notes": ["Respect runtime action scope."],
            },
            semantic_state=semantic_state,
        )
        self.last_strategy_log = {"validated": strategy.as_dict()}
        return strategy

    def propose_action(self, *, local_geometry, **kwargs):
        candidate = local_geometry["candidates"][0]
        result = SemanticActionResult(
            selected_candidate_id=candidate["reference_id"],
            proposal=self.proposal,
            scope_validation={
                "grasp_xy_error_mm": 0.0,
                "max_lateral_mm": 3.0,
                "max_lift_mm": 15.0,
            },
        )
        self.last_action_log = {"validated": result.as_dict()}
        return result

    def evaluate(self, **kwargs):
        evaluation = validate_semantic_evaluation_payload(
            {
                "semantic_target": {
                    "status": "SUPPORTED",
                    "confidence": 0.8,
                    "evidence": ["Target region remained visible."],
                    "hypothesis": "sleeve_end:possible_inward_fold",
                },
                "grasp_acquisition": {
                    "status": "SUCCESS",
                    "confidence": 0.8,
                    "evidence": ["Associated fabric lifted."],
                },
                "structure_engagement": {
                    "status": "SUCCESS",
                    "confidence": 0.8,
                    "evidence": ["Sleeve-associated edge moved."],
                },
                "opening_relevance": {
                    "status": "SUPPORTED",
                    "confidence": 0.7,
                    "evidence": ["Local overlap decreased."],
                },
                "transport": {
                    "status": "GOOD",
                    "confidence": 0.7,
                    "evidence": ["Motion was outward."],
                },
                "laydown": {
                    "status": "SUCCESS",
                    "confidence": 0.7,
                    "evidence": ["Fabric was released on the table."],
                },
                "task_progress": {
                    "status": "IMPROVED",
                    "confidence": 0.8,
                    "metrics": {
                        "visible_area_delta": "INCREASED",
                        "overlap_delta": "DECREASED",
                        "relief_delta": "UNCHANGED",
                        "boundary_change": "Sleeve-associated boundary moved outward.",
                    },
                },
                "earliest_failure_stage": "NONE",
                "next_experiment": {
                    "keep": ["semantic_target"],
                    "change": [],
                    "reason": "Test iteration completed.",
                },
            }
        )
        self.last_evaluation_log = {"validated": evaluation.as_dict()}
        return evaluation


class _LocalGrounder:
    def ground(self, *, artifact_dir, strategy, **kwargs):
        artifact_dir.mkdir(parents=True)
        overlay = artifact_dir / "camera_A_local_grasp_candidates.png"
        Image.new("RGB", (8, 6), (0, 255, 0)).save(overlay)
        manifest = {
            "status": "READY",
            "semantic_anchor_id": strategy.anchor_id,
            "target_part": strategy.target_part,
            "camera": "A",
            "source_image": str(overlay),
            "semantic_anchor_pixel_xy": [4, 3],
            "candidate_count": 1,
            "candidates": [
                {
                    "reference_id": "R001",
                    "semantic_anchor_id": strategy.anchor_id,
                    "target_part": strategy.target_part,
                    "pixel_xy": [4, 3],
                    "base_xyz_mm": [520.0, -40.0, 18.0],
                    "height_above_table_mm": 8.0,
                    "feature": "free_edge",
                    "free_boundary": True,
                    "height_step_mm": 6.0,
                    "relief_above_local_median_mm": 5.0,
                    "local_gradient_mm_per_px": 2.0,
                    "suggested_yaw_deg": 0.0,
                    "graspability_score": 0.9,
                }
            ],
            "overlay": str(overlay),
            "coordinate_guide": str(artifact_dir / "guide.json"),
        }
        (artifact_dir / "local_geometry_candidates.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest


def _proposal():
    return validate_exploration_payload(
        {
            "garment_observation": "fold",
            "reveal_strategy": "lift high-confidence edge",
            "confidence": 0.8,
            "actions": [
                {"name": "open_gripper", "args": {}},
                {
                    "name": "move",
                    "args": {"x": 520, "y": -40, "z": 60, "yaw": 0},
                },
                {
                    "name": "move",
                    "args": {"x": 520, "y": -40, "z": 20, "yaw": 0},
                },
                {"name": "close_gripper", "args": {}},
                {
                    "name": "move",
                    "args": {"x": 523, "y": -40, "z": 35, "yaw": 0},
                },
                {"name": "open_gripper", "args": {}},
            ],
            "expected_observation": "edge moves",
            "safety_notes": ["review"],
        }
    )


def _global_proposal():
    return validate_global_exploration_payload(
        {
            "selected_grasp": {
                "camera": "A",
                "pixel_xy": [4, 3],
                "reason": "A visible raised free boundary can reveal overlap.",
            },
            "garment_observation": "A raised boundary overlaps the main sheet.",
            "reveal_strategy": "Probe and move the selected boundary outward.",
            "confidence": 0.75,
            "actions": [
                {"name": "open_gripper", "args": {}},
                {"name": "move", "args": {"x": 520, "y": -40, "z": 60, "yaw": 0}},
                {"name": "move", "args": {"x": 520, "y": -40, "z": 20, "yaw": 0}},
                {"name": "close_gripper", "args": {}},
                {"name": "move", "args": {"x": 530, "y": -40, "z": 35, "yaw": 0}},
                {"name": "open_gripper", "args": {}},
            ],
            "expected_observation": "The selected layer moves independently.",
            "safety_notes": ["Respect deterministic workspace and IK gates."],
        }
    )


def _saved_perception(tmp_path: Path):
    directory = tmp_path / "saved_perception"
    directory.mkdir()
    image = directory / "camera_0_A.png"
    Image.new("RGB", (8, 6), (20, 30, 40)).save(image)
    result_path = directory / "result.json"
    result = {
        "status": "VALIDATED_DENSE_AB_FUSION",
        "center_base_mm": [520.0, -40.0, 18.0],
        "active_cameras": ["A", "B"],
        "views": [{"label": "A", "image": image.name}],
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result, result_path, image


def _manifest(artifact_dir: Path) -> dict:
    artifact_dir.mkdir(parents=True)
    overlay = artifact_dir / "camera_A_semantic_anchors.png"
    Image.new("RGB", (8, 6), (0, 255, 0)).save(overlay)
    manifest = {
        "status": "READY",
        "anchor_count": 1,
        "confidence_threshold": 0.8,
        "anchors": [
            {
                "anchor_id": "S001",
                "type": "sleeve_end",
                "camera": "A",
                "pixel_xy": [4, 3],
                "base_xyz_mm": [520.0, -40.0, 18.0],
                "height_above_table_mm": 8.0,
                "local_base_z_spread_mm": 3.0,
                "confidence": 0.9,
            }
        ],
        "views": [
            {
                "camera": "A",
                "accepted_overlay": str(overlay),
                "records": [
                    {
                        "name": "edge",
                        "status": "point_returned",
                        "confidence": 0.9,
                        "accepted": True,
                        "anchor_id": "S001",
                        "rejection_reason": None,
                    },
                    {
                        "name": "center",
                        "status": "point_returned",
                        "confidence": 0.5,
                        "accepted": False,
                        "rejection_reason": "below_threshold",
                    },
                ],
            }
        ],
    }
    (artifact_dir / "molmo_semantic_anchors.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


class _GlobalClient:
    def __init__(self, proposal):
        self.proposal = proposal
        self.prompts: list[str] = []

    def invoke(self, image_paths, prompt, run_dir):
        self.prompts.append(prompt)
        return SimpleNamespace(
            proposal=self.proposal,
            prompt=prompt,
            command=("claude",),
            returncode=0,
            stdout="{}",
            stderr="",
        )


class _GlobalClientSequence(_GlobalClient):
    def __init__(self, proposals):
        super().__init__(proposals[0])
        self.proposals = list(proposals)

    def invoke(self, image_paths, prompt, run_dir):
        index = min(len(self.prompts), len(self.proposals) - 1)
        self.prompts.append(prompt)
        return SimpleNamespace(
            proposal=self.proposals[index],
            prompt=prompt,
            command=("claude",),
            returncode=0,
            stdout="{}",
            stderr="",
        )


class _GlobalEvaluation:
    def __init__(self):
        self.task_progress = SimpleNamespace(status="IMPROVED", confidence=0.8)
        self.next_experiment = SimpleNamespace(
            keep=("selected_pixel",),
            change=("transport_direction",),
        )
        self.stop = False
        self.reason = "Keep the acquired layer and revise transport direction."

    def as_dict(self):
        return {
            "task_progress": {"status": "IMPROVED", "confidence": 0.8},
            "next_experiment": {
                "keep": ["selected_pixel"],
                "change": ["transport_direction"],
                "reason": self.reason,
            },
        }


class _GlobalEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, *args, **kwargs):
        self.calls += 1
        return _GlobalEvaluation()


class _TimeoutThenGlobalEvaluator(_GlobalEvaluator):
    def __init__(self, timeout_count: int = 1):
        super().__init__()
        self.timeout_count = timeout_count

    def evaluate(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.timeout_count:
            raise ExplorationTimeoutError(
                "Claude evaluation timed out after 400 seconds"
            )
        return _GlobalEvaluation()


def test_global_cli_skips_molmo_and_local_candidates(tmp_path: Path, monkeypatch) -> None:
    proposal_payload = json.loads(json.dumps(_global_proposal().as_dict()))
    proposal_payload["actions"][1]["args"].update({"x": 535.0, "y": -30.0})
    proposal_payload["actions"][2]["args"].update({"x": 535.0, "y": -30.0})
    proposal = validate_global_exploration_payload(proposal_payload)
    session = _Session(tmp_path, list(proposal.actions))
    saved, result_path, _ = _saved_perception(tmp_path)
    monkeypatch.setattr(
        "cloth_agent.molmo_keypoint_cli._load_latest_perception",
        lambda session: (saved, result_path),
    )
    perception_dir = session.workspace / "perception_views"
    xyz = np.zeros((6, 8, 3), dtype=np.float32)
    xyz[:, :, :] = [520.0, -40.0, 18.0]
    np.save(perception_dir / "camera_A_base_xyz_mm.npy", xyz)
    np.save(
        perception_dir / "camera_A_height_above_table_mm.npy",
        np.full((6, 8), 8.0, dtype=np.float32),
    )
    (perception_dir / "camera_A_coordinate_guide.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "reference_id": "R001",
                        "pixel_xy": [4, 3],
                        "base_xyz_mm": [520.0, -40.0, 18.0],
                        "height_above_table_mm": 8.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    keypoint_started = False
    gpu_probed = False

    def forbidden_keypoints(**kwargs):
        nonlocal keypoint_started
        keypoint_started = True
        raise AssertionError("global planning must not start Molmo")

    def forbidden_gpu_probe():
        nonlocal gpu_probed
        gpu_probed = True
        raise AssertionError("global planning does not need a Molmo GPU gate")

    global_client = _GlobalClient(proposal)
    output = tmp_path / "global_cli"
    stream = StringIO()
    code = run_keypoint_cli_loop(
        session,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        output,
        KeypointCliOptions(
            max_iterations=1,
            enable_real=False,
            skip_controller_ik=True,
            keypoint_specs=(),
        ),
        capture=lambda config: [object(), object()],
        keypoint_runner=forbidden_keypoints,
        global_client=global_client,  # type: ignore[arg-type]
        global_evaluator=object(),  # type: ignore[arg-type]
        gpu_memory_probe=forbidden_gpu_probe,
        stream=stream,
    )

    assert code == 0
    assert keypoint_started is False
    assert gpu_probed is False
    result = json.loads((output / "iteration_001" / "result.json").read_text())
    assert result["candidate_policy"] == "NONE_CLAUDE_SELECTS_ARBITRARY_PIXEL"
    assert result["proposal"]["selected_grasp"]["pixel_xy"] == [4, 3]
    assert result["proposal"]["actions"][1]["args"]["x"] == 520.0
    assert result["proposal"]["actions"][1]["args"]["y"] == -40.0
    assert result["proposal"]["actions"][2]["args"]["x"] == 520.0
    assert result["proposal"]["actions"][2]["args"]["y"] == -40.0
    assert result["global_grounding"]["grounding_policy"] == (
        "runtime_authoritative_selected_pixel_xy"
    )
    assert result["global_grounding"]["claude_requested_grasp_xy_mm"] == [535.0, -30.0]
    assert result["global_grounding"]["commanded_grasp_xy_mm"] == [520.0, -40.0]
    assert result["global_grounding"]["grounded_action_numbers"] == [2, 3]
    assert "semantic_anchors" not in result
    assert "local_geometry" not in result
    assert "no Sxxx/Rxxx candidate generation" in stream.getvalue()
    assert "Previous exploration outcomes" in global_client.prompts[0]


def test_global_cli_feeds_before_after_learning_into_next_iteration(
    tmp_path: Path, monkeypatch
) -> None:
    proposal = _global_proposal()
    session = _Session(tmp_path, list(proposal.actions))
    saved, result_path, _ = _saved_perception(tmp_path)
    monkeypatch.setattr(
        "cloth_agent.molmo_keypoint_cli._load_latest_perception",
        lambda session: (saved, result_path),
    )
    perception_dir = session.workspace / "perception_views"
    xyz = np.zeros((6, 8, 3), dtype=np.float32)
    xyz[:, :, :] = [520.0, -40.0, 18.0]
    np.save(perception_dir / "camera_A_base_xyz_mm.npy", xyz)
    (perception_dir / "camera_A_coordinate_guide.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "reference_id": "R001",
                        "pixel_xy": [4, 3],
                        "base_xyz_mm": [520.0, -40.0, 18.0],
                        "height_above_table_mm": 8.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    global_client = _GlobalClient(proposal)
    evaluator = _GlobalEvaluator()
    code = run_keypoint_cli_loop(
        session,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        tmp_path / "global_learning_cli",
        KeypointCliOptions(
            max_iterations=2,
            settle_s=0,
            enable_real=True,
            keypoint_specs=(),
        ),
        capture=lambda config: [object(), object()],
        global_client=global_client,  # type: ignore[arg-type]
        global_evaluator=evaluator,  # type: ignore[arg-type]
        controller_validator=lambda *args: {"status": "PASS"},
        perception_positioner=lambda config: {
            "actual_tcp_pose_mm_deg": [500, 0, 800, 180, 0, 0]
        },
        stream=StringIO(),
    )

    assert code == 0
    assert session.execution_calls == 2
    assert evaluator.calls == 2
    assert len(global_client.prompts) == 2
    assert "workspace/global_experience.jsonl" in global_client.prompts[1]
    history_text = (session.workspace / "global_experience.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"transport_direction"' in history_text
    assert "Keep the acquired layer and revise transport direction" in history_text
    history = (session.workspace / "global_experience.jsonl").read_text(
        encoding="utf-8"
    )
    assert len(history.splitlines()) == 2


def test_global_evaluation_timeout_retries_without_repeating_robot(
    tmp_path: Path, monkeypatch
) -> None:
    proposal = _global_proposal()
    session = _Session(tmp_path, list(proposal.actions))
    saved, result_path, _ = _saved_perception(tmp_path)
    monkeypatch.setattr(
        "cloth_agent.molmo_keypoint_cli._load_latest_perception",
        lambda session: (saved, result_path),
    )
    perception_dir = session.workspace / "perception_views"
    xyz = np.zeros((6, 8, 3), dtype=np.float32)
    xyz[:, :, :] = [520.0, -40.0, 18.0]
    np.save(perception_dir / "camera_A_base_xyz_mm.npy", xyz)
    (perception_dir / "camera_A_coordinate_guide.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "reference_id": "R001",
                        "pixel_xy": [4, 3],
                        "base_xyz_mm": [520.0, -40.0, 18.0],
                        "height_above_table_mm": 8.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    evaluator = _TimeoutThenGlobalEvaluator(timeout_count=1)
    sleeps: list[float] = []
    stream = StringIO()
    output = tmp_path / "global_evaluation_retry"

    code = run_keypoint_cli_loop(
        session,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        output,
        KeypointCliOptions(
            max_iterations=1,
            settle_s=0,
            enable_real=True,
            keypoint_specs=(),
            max_evaluation_retries=0,
            evaluation_retry_backoff_s=1.5,
        ),
        capture=lambda config: [object(), object()],
        global_client=_GlobalClient(proposal),  # type: ignore[arg-type]
        global_evaluator=evaluator,  # type: ignore[arg-type]
        controller_validator=lambda *args: {"status": "PASS"},
        perception_positioner=lambda config: {
            "actual_tcp_pose_mm_deg": [500, 0, 800, 180, 0, 0]
        },
        sleep=sleeps.append,
        stream=stream,
    )

    assert code == 0
    assert session.execution_calls == 1
    assert evaluator.calls == 2
    assert sleeps == [1.5]
    result = json.loads(
        (output / "iteration_001" / "result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "COMPLETED"
    assert result["evaluation_attempt_count"] == 2
    assert result["evaluation_retry_count"] == 1
    assert result["evaluation_timeout_retries"][0]["robot_command_sent"] is False
    assert result["evaluation_timeout_retries"][0]["same_saved_before_after"] is True
    evaluation_recovery = result["evaluation_error_recovery"]
    assert evaluation_recovery["robot_command_repeated"] is False
    assert evaluation_recovery["skill_proposal"]["name"] == (
        "evaluation-timeout-retry"
    )
    assert evaluation_recovery["skill_review"]["status"] == "RUN_LOCAL_PENDING"
    assert "no robot or camera command" in stream.getvalue()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETED"


def test_global_validation_error_is_fed_back_and_forms_recovery_skill(
    tmp_path: Path, monkeypatch
) -> None:
    rejected = _global_proposal()
    corrected_payload = json.loads(json.dumps(rejected.as_dict()))
    corrected_payload["reveal_strategy"] = (
        "Shorten the transport after the controller IK rejection."
    )
    corrected_payload["actions"][4]["args"]["x"] = 525.0
    corrected = validate_global_exploration_payload(corrected_payload)
    session = _Session(tmp_path, list(corrected.actions))
    saved, result_path, _ = _saved_perception(tmp_path)
    monkeypatch.setattr(
        "cloth_agent.molmo_keypoint_cli._load_latest_perception",
        lambda session: (saved, result_path),
    )
    perception_dir = session.workspace / "perception_views"
    xyz = np.zeros((6, 8, 3), dtype=np.float32)
    xyz[:, :, :] = [520.0, -40.0, 18.0]
    np.save(perception_dir / "camera_A_base_xyz_mm.npy", xyz)
    (perception_dir / "camera_A_coordinate_guide.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "reference_id": "R001",
                        "pixel_xy": [4, 3],
                        "base_xyz_mm": [520.0, -40.0, 18.0],
                        "height_above_table_mm": 8.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    client = _GlobalClientSequence([rejected, corrected])
    controller_calls = 0

    def controller_validator(*args):
        nonlocal controller_calls
        controller_calls += 1
        if controller_calls == 1:
            raise SafetyError(
                "controller IK rejected action 7 segment sample 1/8, code=10"
            )
        return {"status": "PASS"}

    output = tmp_path / "global_error_feedback_skill"
    code = run_keypoint_cli_loop(
        session,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        output,
        KeypointCliOptions(
            max_iterations=1,
            enable_real=False,
            max_replans=1,
            keypoint_specs=(),
        ),
        capture=lambda config: [object(), object()],
        global_client=client,  # type: ignore[arg-type]
        global_evaluator=object(),  # type: ignore[arg-type]
        controller_validator=controller_validator,
        stream=StringIO(),
    )

    assert code == 0
    assert controller_calls == 2
    assert session.execution_calls == 0
    assert "controller IK rejected action 7" in client.prompts[1]
    assert '"rejected_actions"' in client.prompts[1]
    result = json.loads(
        (output / "iteration_001" / "result.json").read_text(encoding="utf-8")
    )
    feedback = result["error_feedback_to_claude"]
    assert feedback["error_type"] == "SafetyError"
    assert feedback["physical_command_sent"] is False
    recovery = result["preexecution_error_recovery"]
    assert recovery["validation_result"]["controller_ik"] == "PASSED"
    assert recovery["validation_result"]["materially_changed"] is True
    assert recovery["skill_proposal"]["name"] == "controller-ik-recovery"
    assert recovery["skill_review"]["status"] == "RUN_LOCAL_PENDING"
    approved = json.loads(
        (tmp_path / "data" / "skills" / "approved.json").read_text(
            encoding="utf-8"
        )
    )
    assert any(
        skill["name"] == "controller-ik-recovery"
        for skill in approved["skills"]
    )
    synthesis = json.loads(
        (output / "run_skill_synthesis.json").read_text(encoding="utf-8")
    )
    assert synthesis["candidate_count"] >= 1
    assert synthesis["persistent_reviews"][0]["review"]["approved"] is True
    history = (session.workspace / "global_experience.jsonl").read_text(
        encoding="utf-8"
    )
    assert "preexecution_error_recovery" in history


def test_preexecution_molmo_error_is_recoverable_after_annotation_stage():
    error = MolmoKeypointPipelineError("worker exited with 1")

    assert _is_recoverable_loop_error(
        error,
        {
            "execution": None,
            "last_completed_stage": "SEMANTIC_ANNOTATIONS_COMPLETED",
        },
    ) is True

    proposal = _recovery_skill_proposal(
        {
            "attempt": 1,
            "error_type": "MolmoKeypointPipelineError",
            "error": "worker exited with 1",
        },
        corrected_attempt=1,
    )
    assert proposal.name == "molmo-annotation-retry"


def test_cli_dry_run_prints_all_phases_and_checkpoints_iteration(
    tmp_path: Path, monkeypatch
) -> None:
    proposal = _proposal()
    session = _Session(tmp_path, list(proposal.actions))
    saved, result_path, image = _saved_perception(tmp_path)
    monkeypatch.setattr(
        "cloth_agent.molmo_keypoint_cli._load_latest_perception",
        lambda session: (saved, result_path),
    )
    monkeypatch.setattr(
        "cloth_agent.molmo_keypoint_cli.perception_image_paths",
        lambda result, path: [image],
    )

    def fake_keypoints(**kwargs):
        kwargs["worker_line_callback"]("loading fake model\n")
        return _manifest(kwargs["artifact_dir"])

    output = tmp_path / "cli_output"
    stream = StringIO()
    code = run_keypoint_cli_loop(
        session,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        output,
        KeypointCliOptions(
            planning_policy="semantic_local",
            max_iterations=1,
            enable_real=False,
            skip_controller_ik=True,
            keypoint_specs=(KeypointSpec("edge", "visible edge", (0, 255, 0)),),
        ),
        capture=lambda config: [object(), object()],
        keypoint_runner=fake_keypoints,
        client=_Client(proposal),  # type: ignore[arg-type]
        semantic_state_builder=SemanticStateBuilder(),
        local_geometry_grounder=_LocalGrounder(),  # type: ignore[arg-type]
        gpu_memory_probe=lambda: 24_000,
        stream=stream,
    )

    assert code == 0
    assert session.execution_calls == 0
    text = stream.getvalue()
    assert "ClothAgent · Global Garment CLI" in text
    assert "MOLMO/WORKER" in text
    assert "loading fake model" in text
    assert "confidence=0.9000 valid=true anchor=S001" in text
    assert "confidence=0.5000 valid=false" in text
    assert "target=sleeve_end" in text
    assert "selected=R001" in text
    assert "DRY_RUN" in text or "dry run" in text
    assert "I001" in text
    assert "+00:" in text
    assert "PERCEPTION" in text
    assert "SEMANTIC-STATE" in text
    assert "SEMANTIC-STRATEGY" in text
    assert "LOCAL-GEOMETRY" in text
    assert "phase 00:" in text
    assert "\x1b[" not in text
    result = json.loads(
        (output / "iteration_001" / "result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "DRY_RUN_VALIDATED"
    assert result["last_completed_stage"] == "PREEXECUTION_VALIDATED"
    assert (output / "iteration_001" / "claude_semantic_strategy_log.json").is_file()
    assert (
        output / "iteration_001" / "claude_semantic_action_attempt_01.json"
    ).is_file()
    assert (output / "iteration_001.json").is_file()
    assert (output / "events.jsonl").is_file()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETED"
    assert summary["iteration_count"] == 1


def test_real_cli_positions_home_then_perception_before_every_capture(
    tmp_path: Path, monkeypatch
) -> None:
    proposal = _proposal()
    session = _Session(tmp_path, list(proposal.actions))
    saved, result_path, image = _saved_perception(tmp_path)
    monkeypatch.setattr(
        "cloth_agent.molmo_keypoint_cli._load_latest_perception",
        lambda session: (saved, result_path),
    )
    monkeypatch.setattr(
        "cloth_agent.molmo_keypoint_cli.perception_image_paths",
        lambda result, path: [image],
    )
    order: list[str] = []

    def fake_positioner(config):
        order.append("position")
        return {
            "sequence": ["home", "perception_position"],
            "actual_tcp_pose_mm_deg": [478, 9, 813, 160, 63, 162],
        }

    def fake_capture(config):
        order.append("capture")
        return [
            SimpleNamespace(
                label=label,
                rgb=np.zeros((6, 8, 3), dtype=np.uint8),
                depth_m=np.ones((6, 8), dtype=np.float32),
            )
            for label in ("A", "B")
        ]

    output = tmp_path / "cli_real_positioning"
    stream = StringIO()
    code = run_keypoint_cli_loop(
        session,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        output,
        KeypointCliOptions(
            planning_policy="semantic_local",
            max_iterations=1,
            settle_s=0,
            enable_real=True,
            min_gpu_free_mib=0,
            keypoint_specs=(KeypointSpec("edge", "visible edge", (0, 255, 0)),),
        ),
        capture=fake_capture,
        keypoint_runner=lambda **kwargs: _manifest(kwargs["artifact_dir"]),
        client=_Client(proposal),  # type: ignore[arg-type]
        semantic_state_builder=SemanticStateBuilder(),
        local_geometry_grounder=_LocalGrounder(),  # type: ignore[arg-type]
        controller_validator=lambda *args: {"status": "PASS"},
        perception_positioner=fake_positioner,
        stream=stream,
    )

    assert code == 0
    assert order == ["position", "capture", "position", "capture"]
    assert session.execution_calls == 1
    result = json.loads(
        (output / "iteration_001" / "result.json").read_text(encoding="utf-8")
    )
    assert result["pre_perception_robot_positioning"]["sequence"] == [
        "home",
        "perception_position",
    ]
    assert result["post_action_perception_robot_positioning"]["sequence"] == [
        "home",
        "perception_position",
    ]
    assert "Home → perception_position" in stream.getvalue()


def test_cli_worker_failure_is_printed_and_saved(
    tmp_path: Path, monkeypatch
) -> None:
    proposal = _proposal()
    session = _Session(tmp_path, list(proposal.actions))
    saved, result_path, image = _saved_perception(tmp_path)
    monkeypatch.setattr(
        "cloth_agent.molmo_keypoint_cli._load_latest_perception",
        lambda session: (saved, result_path),
    )
    monkeypatch.setattr(
        "cloth_agent.molmo_keypoint_cli.perception_image_paths",
        lambda result, path: [image],
    )

    def failing_keypoints(**kwargs):
        directory = kwargs["artifact_dir"]
        directory.mkdir(parents=True)
        log = directory / "molmo_keypoints.stdout.txt"
        log.write_text("torch.OutOfMemoryError: CUDA out of memory\n", encoding="utf-8")
        kwargs["worker_line_callback"]("torch.OutOfMemoryError: CUDA out of memory\n")
        raise MolmoKeypointPipelineError("worker exited with 1")

    output = tmp_path / "cli_failure"
    stream = StringIO()
    code = run_keypoint_cli_loop(
        session,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        output,
        KeypointCliOptions(
            planning_policy="semantic_local",
            max_iterations=1,
            keypoint_specs=(KeypointSpec("edge", "visible edge", (0, 255, 0)),),
        ),
        capture=lambda config: [object(), object()],
        keypoint_runner=failing_keypoints,
        client=_Client(proposal),  # type: ignore[arg-type]
        gpu_memory_probe=lambda: 24_000,
        stream=stream,
    )

    assert code == 1
    assert session.execution_calls == 0
    text = stream.getvalue()
    assert "torch.OutOfMemoryError: CUDA out of memory" in text
    assert "MolmoKeypointPipelineError: worker exited with 1" in text
    result = json.loads(
        (output / "iteration_001" / "result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "FAILED"
    assert "worker exited with 1" in result["error"]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED"


def test_cli_rejects_low_gpu_memory_before_starting_worker(
    tmp_path: Path, monkeypatch
) -> None:
    proposal = _proposal()
    session = _Session(tmp_path, list(proposal.actions))
    saved, result_path, image = _saved_perception(tmp_path)
    monkeypatch.setattr(
        "cloth_agent.molmo_keypoint_cli._load_latest_perception",
        lambda session: (saved, result_path),
    )
    monkeypatch.setattr(
        "cloth_agent.molmo_keypoint_cli.perception_image_paths",
        lambda result, path: [image],
    )
    worker_started = False

    def must_not_start(**kwargs):
        nonlocal worker_started
        worker_started = True
        raise AssertionError("Molmo worker must not start with insufficient memory")

    output = tmp_path / "cli_low_memory"
    stream = StringIO()
    code = run_keypoint_cli_loop(
        session,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        output,
        KeypointCliOptions(
            planning_policy="semantic_local",
            max_iterations=1,
            min_gpu_free_mib=19_000,
            keypoint_specs=(KeypointSpec("edge", "visible edge", (0, 255, 0)),),
        ),
        capture=lambda config: [object(), object()],
        keypoint_runner=must_not_start,
        client=_Client(proposal),  # type: ignore[arg-type]
        gpu_memory_probe=lambda: 18_500,
        stream=stream,
    )

    assert code == 1
    assert worker_started is False
    assert "free=18500 MiB required>=19000 MiB" in stream.getvalue()
    assert "insufficient free GPU memory before Molmo model load" in stream.getvalue()
    result = json.loads(
        (output / "iteration_001" / "result.json").read_text(encoding="utf-8")
    )
    assert result["gpu_memory_preflight"]["valid"] is False


def test_cli_reporter_heartbeat_shows_active_phase_and_elapsed_time(
    tmp_path: Path,
) -> None:
    stream = StringIO()
    reporter = CliReporter(tmp_path / "events.jsonl", stream=stream, color=False)
    reporter.start_heartbeat(0.01)
    try:
        reporter.start_phase(
            "molmo",
            "loading model shards",
            iteration=1,
        )
        time.sleep(0.04)
        reporter.finish_phase("model ready")
    finally:
        reporter.stop_heartbeat()

    text = stream.getvalue()
    assert "MOLMO" in text
    assert "still running: loading model shards" in text
    assert "phase elapsed 00:" in text
    assert "model ready · phase 00:" in text
    assert "\x1b[" not in text
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["level"] == "WAIT" for event in events)
    assert all("local_time" in event and "run_elapsed_s" in event for event in events)


def test_cli_reporter_heartbeat_refreshes_one_terminal_line(tmp_path: Path) -> None:
    class _TtyBuffer(StringIO):
        def isatty(self) -> bool:
            return True

    stream = _TtyBuffer()
    reporter = CliReporter(tmp_path / "events.jsonl", stream=stream, color=False)
    reporter.start_phase("global-planning", "inspect complete scene", iteration=1)
    reporter.emit(
        "global-planning",
        "still running: inspect complete scene · phase elapsed 00:10.0",
        iteration=1,
        level="WAIT",
    )
    heartbeat_text = stream.getvalue()
    assert heartbeat_text.count("still running:") == 1
    assert not heartbeat_text.endswith("\n")

    reporter.finish_phase("validated proposal")
    text = stream.getvalue()
    assert "validated proposal" in text
    assert "\033[2K" in text
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["level"] for event in events] == ["START", "WAIT", "DONE"]
