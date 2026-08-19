from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

from cloth_agent.config import RobotConfig, WorkspaceBounds
from cloth_agent.free_exploration import validate_exploration_payload
from cloth_agent.molmo_keypoint_cli import (
    CliReporter,
    KeypointCliOptions,
    run_keypoint_cli_loop,
)
from cloth_agent.molmo_keypoint_pipeline import (
    KeypointSpec,
    MolmoKeypointPipelineError,
)
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
    assert "ClothAgent · Semantic Anchor CLI" in text
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
            max_iterations=1,
            min_gpu_free_mib=20_000,
            keypoint_specs=(KeypointSpec("edge", "visible edge", (0, 255, 0)),),
        ),
        capture=lambda config: [object(), object()],
        keypoint_runner=must_not_start,
        client=_Client(proposal),  # type: ignore[arg-type]
        gpu_memory_probe=lambda: 19_500,
        stream=stream,
    )

    assert code == 1
    assert worker_started is False
    assert "free=19500 MiB required>=20000 MiB" in stream.getvalue()
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
