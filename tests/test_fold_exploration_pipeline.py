from __future__ import annotations

import errno
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cloth_agent.auto_exploration import ReferenceReselectionExhaustedError
from cloth_agent.config import RobotConfig, WorkspaceBounds
from cloth_agent.fold_exploration_pipeline import (
    FOLD_STEP_IDS,
    FoldExplorationPipeline,
    FoldExperienceStore,
    FoldSupervisor,
    assess_screen_visibility,
    build_reverse_trajectory,
    _build_upright_camera_a_planning_images,
    _clockwise90_pixel,
    _compact_history,
    _filter_fold_sleeve_planning_overlay,
    _fold_acquisition_learning_state,
    _grasp_strategy_signature,
    _proposal_from_actions,
    _select_fold_planning_images,
    _select_supervisor_images,
    _validate_acquisition_strategy_change,
    validate_supervisor_payload,
)


def test_upright_camera_a_planning_images_keep_rxxx_identity(tmp_path: Path) -> None:
    perception = tmp_path / "perception"
    perception.mkdir()
    raw = np.zeros((3, 4, 3), dtype=np.uint8)
    raw[2, 1] = (255, 0, 0)
    Image.fromarray(raw).save(perception / "camera_0_A.png")
    (perception / "camera_A_coordinate_guide.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "reference_id": "R007",
                        "pixel_xy": [1, 2],
                        "base_xyz_mm": [500, 0, 10],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = {
        "views": [
            {
                "label": "A",
                "image": "camera_0_A.png",
                "coordinate_guide": "camera_A_coordinate_guide.json",
            }
        ]
    }

    images = _build_upright_camera_a_planning_images(
        result,
        perception / "result.json",
        tmp_path / "planning",
    )

    assert [path.name for path in images] == [
        "camera_A_rgb_upright.png",
        "camera_A_rxxx_overlay_upright.png",
    ]
    with Image.open(images[0]) as upright:
        assert upright.size == (3, 4)
    assert _clockwise90_pixel(1, 2, raw_height=3) == (0.0, 1.0)
    assert _select_fold_planning_images(images) == images


def test_argument_list_too_long_is_not_retried(tmp_path: Path) -> None:
    images = []
    for name in ("camera_A_rgb_upright.png", "camera_A_rxxx_overlay_upright.png"):
        path = tmp_path / name
        path.write_bytes(b"image")
        images.append(path)
    calls = []

    def fail_plan(*args, **kwargs):
        calls.append(1)
        raise OSError(errno.E2BIG, "Argument list too long")

    pipeline = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    pipeline.max_stage_retries = 2
    pipeline.retry_backoff_s = 0.0
    pipeline.client = SimpleNamespace(plan=fail_plan)
    pipeline.session = SimpleNamespace()
    pipeline._debug = lambda *args, **kwargs: None
    pipeline._debug_exception = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError, match="will not be retried"):
        pipeline._plan_fold_with_retries(
            images,
            "objective",
            [],
            iteration=1,
        )
    assert len(calls) == 1


def test_reference_reselection_exhaustion_is_not_retried(tmp_path: Path) -> None:
    images = []
    for name in ("camera_A_rgb_upright.png", "camera_A_rxxx_overlay_upright.png"):
        path = tmp_path / name
        path.write_bytes(b"image")
        images.append(path)
    calls = []

    def fail_plan(*args, **kwargs):
        calls.append(1)
        raise ReferenceReselectionExhaustedError("no executable sleeve reference")

    pipeline = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    pipeline.max_stage_retries = 2
    pipeline.retry_backoff_s = 0.0
    pipeline.client = SimpleNamespace(plan=fail_plan)
    pipeline.session = SimpleNamespace()
    pipeline._debug = lambda *args, **kwargs: None
    pipeline._debug_exception = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError, match="will not be retried"):
        pipeline._plan_fold_with_retries(
            images,
            "objective",
            [],
            iteration=1,
        )
    assert len(calls) == 1


def test_fold_planning_overlay_preserves_full_uniform_grid_with_molmo_hint(
    tmp_path: Path,
) -> None:
    upright_rgb = np.full((180, 120, 3), 240, dtype=np.uint8)
    upright_mask = np.zeros((180, 120), dtype=bool)
    upright_mask[35:150, 35:85] = True
    upright_mask[65:105, 5:35] = True
    upright_rgb[upright_mask] = (20, 25, 45)
    raw_rgb = np.rot90(upright_rgb, k=1)
    raw_mask = np.rot90(upright_mask, k=1)
    raw_height = raw_mask.shape[0]

    perception = tmp_path / "perception"
    planning = tmp_path / "planning"
    perception.mkdir()
    planning.mkdir()
    raw_path = perception / "camera_0_A.png"
    mask_path = perception / "camera_A_garment_mask.npy"
    guide_path = perception / "camera_A_coordinate_guide.json"
    Image.fromarray(raw_rgb).save(raw_path)
    np.save(mask_path, raw_mask)

    edge_raw = [85, raw_height - 1 - 10]
    interior_raw = [85, raw_height - 1 - 55]
    guide_path.write_text(
        json.dumps(
            {
                "samples": [
                    {"reference_id": "R001", "pixel_xy": edge_raw},
                    {"reference_id": "R002", "pixel_xy": interior_raw},
                ]
            }
        ),
        encoding="utf-8",
    )
    upright_path = planning / "camera_A_rgb_upright.png"
    overlay_path = planning / "camera_A_rxxx_overlay_upright.png"
    Image.fromarray(upright_rgb).save(upright_path)
    Image.fromarray(upright_rgb).save(overlay_path)
    (planning / "camera_A_upright_mapping.json").write_text(
        json.dumps(
            {
                "rotation": "clockwise90",
                "raw_image": str(raw_path),
                "coordinate_guide": str(guide_path),
            }
        ),
        encoding="utf-8",
    )

    report = _filter_fold_sleeve_planning_overlay(
        [upright_path, overlay_path],
        "The supervisor says the next incomplete step is left_sleeve.",
        molmo_hint={
            "status": "MOLMO_POINT_AVAILABLE",
            "upright_pixel_xy": [10, 85],
            "confidence": 0.8,
        },
    )

    assert report is not None
    assert [item["reference_id"] for item in report["accepted"]] == [
        "R001",
        "R002",
    ]
    assert report["rejected"] == []
    assert report["reference_mode"] == "uniform_full_garment"
    assert report["visible_reference_count"] == 2
    assert report["molmo_fusion"]["status"] == (
        "MOLMO_HINT_OVER_FULL_UNIFORM_GRID"
    )
    assert report["molmo_fusion"]["narrowed_candidates"] is False
    assert (planning / "camera_A_rxxx_overlay_upright_all.png").is_file()
    assert (planning / "camera_A_left_sleeve_candidate_filter.json").is_file()


def test_upright_builder_keeps_original_uniform_references_only(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    perception = run / "results" / "perception" / "center_test"
    workspace = run / "workspace" / "perception_views"
    output = run / "results" / "fold_exploration" / "test" / "iteration_001" / "before_raw"
    perception.mkdir(parents=True)
    workspace.mkdir(parents=True)

    upright_rgb = np.full((180, 120, 3), 240, dtype=np.uint8)
    upright_mask = np.zeros((180, 120), dtype=bool)
    upright_mask[35:150, 35:85] = True
    upright_mask[65:105, 5:35] = True
    upright_mask[65:105, 85:115] = True
    upright_rgb[upright_mask] = (20, 25, 45)
    raw_rgb = np.rot90(upright_rgb, k=1)
    raw_mask = np.rot90(upright_mask, k=1)
    xyz = np.zeros((*raw_mask.shape, 3), dtype=np.float32)
    yy, xx = np.indices(raw_mask.shape)
    xyz[..., 0] = 400 + xx
    xyz[..., 1] = -250 + yy
    xyz[..., 2] = 10
    height = np.where(raw_mask, 5.0, np.nan).astype(np.float32)
    Image.fromarray(raw_rgb).save(perception / "camera_0_A.png")
    np.save(perception / "camera_A_garment_mask.npy", raw_mask)
    np.save(perception / "camera_A_base_xyz_mm.npy", xyz)
    np.save(perception / "camera_A_height_above_table_mm.npy", height)
    guide = {
        "full_resolution_xyz_map": "camera_A_base_xyz_mm.npy",
        "samples": [
            {
                "reference_id": "R001",
                "pixel_xy": [85, 60],
                "base_xyz_mm": [460, -165, 10],
                "height_above_table_mm": 5,
            },
            {
                "reference_id": "R9001",
                "pixel_xy": [80, 65],
                "base_xyz_mm": [465, -170, 10],
                "height_above_table_mm": 5,
                "reference_source": "fold_rgb_boundary_dense",
                "fold_side": "left_sleeve",
            },
        ],
        "fold_boundary_reference_count": 1,
    }
    for directory in (perception, workspace):
        (directory / "camera_A_coordinate_guide.json").write_text(
            json.dumps(guide),
            encoding="utf-8",
        )
    result = {
        "views": [
            {
                "label": "A",
                "image": "camera_0_A.png",
                "coordinate_guide": "camera_A_coordinate_guide.json",
            }
        ]
    }

    _build_upright_camera_a_planning_images(
        result,
        perception / "result.json",
        output,
    )

    saved = json.loads(
        (perception / "camera_A_coordinate_guide.json").read_text(encoding="utf-8")
    )
    dense = [
        sample
        for sample in saved["samples"]
        if sample.get("reference_source") == "fold_rgb_boundary_dense"
    ]
    assert dense == []
    mapping = json.loads(
        (output / "camera_A_upright_mapping.json").read_text(encoding="utf-8")
    )
    assert mapping["reference_mode"] == "uniform_full_garment"
    assert mapping["dense_boundary_references"] == []
    workspace_saved = json.loads(
        (workspace / "camera_A_coordinate_guide.json").read_text(encoding="utf-8")
    )
    assert [item["reference_id"] for item in workspace_saved["samples"]] == [
        "R001"
    ]
    assert "fold_boundary_reference_count" not in workspace_saved


def test_molmo_sleeve_locator_is_camera_a_only_and_non_installing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    perception = run / "workspace" / "perception_views"
    iteration_dir = run / "results" / "fold" / "iteration_001"
    perception.mkdir(parents=True)
    iteration_dir.mkdir(parents=True)
    Image.new("RGB", (1280, 720), (255, 255, 255)).save(
        perception / "camera_0_A.png"
    )
    np.save(
        perception / "camera_A_base_xyz_mm.npy",
        np.zeros((720, 1280, 3), dtype=np.float32),
    )
    np.save(
        perception / "camera_A_height_above_table_mm.npy",
        np.zeros((720, 1280), dtype=np.float32),
    )
    seen = {}

    def fake_molmo(**kwargs):
        seen.update(kwargs)
        kwargs["artifact_dir"].mkdir(parents=True)
        return {
            "status": "READY",
            "references": [
                {
                    "name": "fold_image_left_sleeve_region",
                    "source_pixel_xy": [400, 700],
                    "pixel_xy": [400, 700],
                    "base_xyz_mm": [420, -260, 10],
                    "confidence": 0.8,
                }
            ],
            "views": [],
        }

    monkeypatch.setattr(
        "cloth_agent.fold_exploration_pipeline.run_molmo_keypoint_pipeline",
        fake_molmo,
    )
    pipeline = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    pipeline.molmo_sleeve_grounding = True
    pipeline.molmo_confidence_threshold = 0.5
    pipeline.molmo_timeout_s = 300
    pipeline.molmo_python = None
    pipeline.molmo_gpu_max_memory_gib = 17.0
    pipeline.molmo_load_in_8bit = True
    pipeline.project_root = tmp_path
    pipeline.session = SimpleNamespace(run_dir=run)
    pipeline._debug = lambda *args, **kwargs: None
    pipeline._debug_exception = lambda *args, **kwargs: None

    hint = pipeline._locate_sleeve_with_molmo(
        step="left_sleeve",
        iteration=1,
        iteration_dir=iteration_dir,
    )

    assert hint is not None
    assert hint["status"] == "MOLMO_POINT_AVAILABLE"
    assert hint["upright_pixel_xy"] == [400, 700]
    assert hint["raw_pixel_xy"] == [700, 319]
    assert seen["perception_dir"] == iteration_dir / "molmo_sleeve_input_upright"
    assert seen["cameras"] == ("A",)
    assert seen["install"] is False
    assert seen["allow_cpu_offload"] is False
    assert seen["gpu_max_memory_gib"] == pytest.approx(17.0)
    assert seen["load_in_8bit"] is True
    assert seen["direct_keypoints"] is True
    assert seen["keypoint_specs"][0].name == "fold_image_left_sleeve_region"
    assert (iteration_dir / "molmo_sleeve_hint.json").is_file()


def _robot() -> RobotConfig:
    return RobotConfig(
        robot_ip="127.0.0.1",
        boundaries=WorkspaceBounds(
            x_min=350, x_max=800, y_min=-300, y_max=170, z_min=-5, z_max=500
        ),
        init_joints_deg=(0, 0, 0, 0, 0, 0, 0),
        init_pose_mm_deg=(500, 0, 280, 180, 0, 0),
        orientation_roll_deg=180,
        orientation_pitch_deg=0,
    )


def _valid_supervisor() -> dict:
    return {
        "status": "READY",
        "current_step": FOLD_STEP_IDS[0],
        "completed_steps": [],
        "next_step": FOLD_STEP_IDS[0],
        "garment_visibility": "FULL",
        "trajectory_decision": "CONTINUE",
        "confidence": 0.8,
        "evidence": ["Both sleeves and the hem are visible."],
        "reason": "The first sleeve remains unfolded.",
    }


def test_supervisor_reads_run_local_context_instead_of_inlining_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "camera_0_A.png"
    image.write_bytes(b"image")
    huge = "x" * 300_000
    seen: dict[str, object] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(_valid_supervisor()),
            stderr="",
        )

    monkeypatch.setattr(
        "cloth_agent.fold_exploration_pipeline.subprocess.run",
        fake_run,
    )
    supervisor = FoldSupervisor(binary="/usr/bin/claude", timeout_s=30)
    result = supervisor.inspect(
        [image],
        tmp_path,
        history=[
            {
                "iteration": 1,
                "planned_step": "left_sleeve",
                "supervisor_after": {
                    **_valid_supervisor(),
                    "command": ["claude", "--print", huge],
                    "raw_stdout": huge,
                },
            }
        ],
        screen={"visibility": "FULL", "bbox_xyxy": [1, 2, 3, 4]},
    )

    command = seen["command"]
    assert isinstance(command, list)
    prompt = command[command.index("--print") + 1]
    assert len(prompt.encode("utf-8")) < 2_000
    assert huge not in prompt
    manifest_path = Path(result["context_bundle"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    history_path = tmp_path / manifest["read_order"][2]
    history_text = history_path.read_text(encoding="utf-8")
    assert "left_sleeve" in history_text
    assert "raw_stdout" not in history_text
    assert "command" not in history_text
    evidence_path = tmp_path / manifest["read_order"][3]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["images"] == ["camera_0_A.png"]


def test_supervisor_payload_is_strict() -> None:
    result = validate_supervisor_payload(_valid_supervisor())
    assert result["next_step"] == "left_sleeve"
    with pytest.raises(Exception):
        validate_supervisor_payload({**_valid_supervisor(), "extra": True})


def test_screen_visibility_detects_border_contact(tmp_path: Path) -> None:
    perception_dir = tmp_path / "perception"
    perception_dir.mkdir()
    mask = np.zeros((20, 30), dtype=bool)
    mask[4:16, 1:12] = True
    np.save(perception_dir / "camera_A_garment_mask.npy", mask)
    (perception_dir / "camera_0_A.png").write_bytes(b"not decoded")
    result = {
        "views": [{"label": "A", "garment_mask": "camera_A_garment_mask.npy", "image": "camera_0_A.png"}]
    }
    visibility = assess_screen_visibility(result, perception_dir / "result.json", margin_px=2)
    assert visibility["visibility"] == "PARTIAL"
    assert visibility["touching_edges"]["left"] is True


def test_reverse_trajectory_reaches_old_grasp_and_homes() -> None:
    actions = [
        {"name": "move", "args": {"x": 500, "y": 0, "z": 80, "yaw": 0}},
        {"name": "open_gripper", "args": {}},
        {"name": "move", "args": {"x": 500, "y": 0, "z": 20, "yaw": 0}},
        {"name": "close_gripper", "args": {}},
        {"name": "move", "args": {"x": 500, "y": 0, "z": 100, "yaw": 0}},
        {"name": "move", "args": {"x": 600, "y": 0, "z": 100, "yaw": 0}},
        {"name": "move", "args": {"x": 600, "y": 0, "z": 20, "yaw": 0}},
        {"name": "open_gripper", "args": {}},
        {"name": "move", "args": {"x": 600, "y": 0, "z": 90, "yaw": 0}},
        {"name": "home", "args": {}},
    ]
    reverse = build_reverse_trajectory(actions, _robot())
    assert reverse[3]["name"] == "close_gripper"
    assert reverse[-1]["name"] == "home"
    assert any(
        action["name"] == "move" and action["args"]["x"] == 500 and action["args"]["z"] == 20
        for action in reverse
    )
    proposal = _proposal_from_actions(reverse, reason="test recovery")
    assert len(proposal.actions) == len(reverse)


def test_experience_store_summarizes_steps(tmp_path: Path) -> None:
    store = FoldExperienceStore(tmp_path / "run")
    summary = store.append(
        {
            "status": "FOLD",
            "supervisor_after": {"completed_steps": ["left_sleeve"]},
        }
    )
    assert summary["next_step"] == "right_sleeve"
    assert json.loads(store.summary_path.read_text())["experience_count"] == 1


def test_supervisor_fallback_does_not_advance_failed_confirmed_step() -> None:
    pipeline = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    result = pipeline._fallback_supervisor(
        {"visibility": "FULL"},
        [
            {
                "planned_step": "left_sleeve",
                "supervisor_after": {
                    "completed_steps": [],
                    "next_step": "left_sleeve",
                },
            }
        ],
        RuntimeError("supervisor unavailable"),
    )

    assert result["completed_steps"] == []
    assert result["next_step"] == "left_sleeve"


def test_supervisor_fallback_advances_only_when_after_record_is_missing() -> None:
    pipeline = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    result = pipeline._fallback_supervisor(
        {"visibility": "FULL"},
        [{"planned_step": "left_sleeve"}],
        RuntimeError("supervisor unavailable"),
    )

    assert result["completed_steps"] == ["left_sleeve"]
    assert result["next_step"] == "right_sleeve"


def test_compact_history_drops_recursive_claude_command_payloads() -> None:
    huge = "x" * 200_000
    compact = _compact_history(
        [
            {
                "iteration": 2,
                "planned_step": "left_sleeve",
                "supervisor_after": {
                    "status": "READY",
                    "completed_steps": [],
                    "next_step": "left_sleeve",
                    "reason": "The sleeve is still unfolded.",
                    "command": ["claude", "--print", huge],
                    "raw_stdout": huge,
                },
                "evaluation": {
                    "grasp_acquisition": {
                        "status": "FAILURE",
                        "confidence": 0.9,
                        "evidence": [huge],
                    },
                    "task_progress": {
                        "status": "NEUTRAL",
                        "confidence": 0.9,
                        "metrics": {"visible_area_delta": "UNCHANGED"},
                    },
                    "earliest_failure_stage": "ACQUISITION",
                    "command": ["claude", "--print", huge],
                },
            }
        ]
    )

    encoded = json.dumps(compact)
    assert len(encoded) < 5_000
    assert "command" not in encoded
    assert "raw_stdout" not in encoded
    assert compact[0]["planned_step"] == "left_sleeve"
    assert compact[0]["evaluation"]["grasp_acquisition"]["status"] == "FAILURE"


def _failed_acquisition_record(
    iteration: int,
    *,
    x: float = 500.0,
    y: float = 0.0,
    z: float = 5.0,
    yaw: float = 0.0,
    entry_x: float | None = None,
) -> dict:
    entry_x = x if entry_x is None else entry_x
    actions = [
        {"name": "open_gripper", "args": {}},
        {"name": "move", "args": {"x": entry_x, "y": y, "z": 35.0, "yaw": yaw}},
        {"name": "move", "args": {"x": x, "y": y, "z": z, "yaw": yaw}},
        {"name": "close_gripper", "args": {}},
        {"name": "move", "args": {"x": x, "y": y, "z": z + 20.0, "yaw": yaw}},
        {"name": "open_gripper", "args": {}},
    ]
    return {
        "iteration": iteration,
        "planned_step": "left_sleeve",
        "proposal": {"actions": actions},
        "evaluation": {
            "grasp_acquisition": {
                "status": "FAILURE",
                "confidence": 0.9,
                "evidence": ["No cloth followed the gripper during the short lift."],
            },
            "earliest_failure_stage": "ACQUISITION",
            "next_experiment": {
                "keep": ["left sleeve target"],
                "change": ["test another contact hypothesis"],
                "reason": "The prior contact closed empty.",
            },
        },
    }


def test_acquisition_learning_escalates_from_probe_to_strategy_diversification() -> None:
    first = _fold_acquisition_learning_state(
        [_failed_acquisition_record(1)],
        "left_sleeve",
    )
    assert first["phase"] == "ACQUISITION_DIAGNOSIS"
    assert first["use_lift_only_probe"] is True
    assert first["require_non_height_change"] is False

    repeated = _fold_acquisition_learning_state(
        [_failed_acquisition_record(1), _failed_acquisition_record(2, z=4.0)],
        "left_sleeve",
    )
    assert repeated["phase"] == "STRATEGY_DIVERSIFICATION"
    assert repeated["consecutive_acquisition_failures"] == 2
    assert repeated["require_non_height_change"] is True
    assert repeated["height_only_retry_pattern"] is True
    assert "changing only grasp Z" in repeated["instruction"]


def test_successful_acquisition_exits_probe_mode() -> None:
    record = _failed_acquisition_record(1)
    record["evaluation"]["grasp_acquisition"]["status"] = "SUCCESS"
    record["evaluation"]["earliest_failure_stage"] = "NONE"
    learning = _fold_acquisition_learning_state([record], "left_sleeve")
    assert learning["phase"] == "ACQUISITION_VALIDATED"
    assert learning["use_lift_only_probe"] is False


def test_unknown_acquisition_after_failure_stays_in_probe_mode() -> None:
    unknown = _failed_acquisition_record(2)
    unknown["evaluation"]["grasp_acquisition"]["status"] = "UNKNOWN"
    unknown["evaluation"]["earliest_failure_stage"] = "UNKNOWN"
    learning = _fold_acquisition_learning_state(
        [_failed_acquisition_record(1), unknown],
        "left_sleeve",
    )
    assert learning["acquisition_validated"] is False
    assert learning["use_lift_only_probe"] is True
    assert learning["uncertain_since_last_evidence"] is True


def test_repeated_acquisition_rejects_height_only_change() -> None:
    learning = _fold_acquisition_learning_state(
        [_failed_acquisition_record(1), _failed_acquisition_record(2, z=4.0)],
        "left_sleeve",
    )
    height_only = _proposal_from_actions(
        _failed_acquisition_record(3, z=3.0)["proposal"]["actions"],
        reason="try a lower contact",
    )
    with pytest.raises(Exception, match="only changes grasp height"):
        _validate_acquisition_strategy_change(height_only, learning)

    changed_yaw = _proposal_from_actions(
        _failed_acquisition_record(3, z=3.0, yaw=25.0)["proposal"]["actions"],
        reason="test a different jaw alignment",
    )
    result = _validate_acquisition_strategy_change(changed_yaw, learning)
    assert result["status"] == "MATERIALLY_DIFFERENT"
    assert any(
        "jaw_alignment" in comparison["non_height_dimensions_changed"]
        for comparison in result["comparisons"]
    )


def test_acquisition_probe_requires_observable_short_lift() -> None:
    learning = _fold_acquisition_learning_state(
        [_failed_acquisition_record(1)],
        "left_sleeve",
    )
    too_small = _proposal_from_actions(
        _failed_acquisition_record(2, z=20.0)["proposal"]["actions"],
        reason="two millimetre lift",
    )
    actions = [dict(action) for action in too_small.actions]
    actions[4] = {
        "name": "move",
        "args": {"x": 500.0, "y": 0.0, "z": 22.0, "yaw": 0.0},
    }
    too_small = _proposal_from_actions(actions, reason="two millimetre lift")
    with pytest.raises(Exception, match="15-30 mm"):
        _validate_acquisition_strategy_change(too_small, learning)


def test_grasp_strategy_signature_distinguishes_lateral_entry() -> None:
    record = _failed_acquisition_record(1, entry_x=492.0)
    signature = _grasp_strategy_signature(record)
    assert signature is not None
    assert signature["entry_style"] == "LATERAL_ENTRY"
    assert signature["entry_lateral_mm"] == pytest.approx(8.0)


def test_supervisor_image_selection_keeps_compact_key_views(tmp_path: Path) -> None:
    names = [
        "camera_A_height_map_heatmap.png",
        "camera_A_height_gradient_edges.png",
        "camera_0_A.png",
        "camera_A_garment_only.png",
        "camera_A_height_map_boundary.png",
        "camera_1_B.png",
        "fused_height_map_boundary.png",
        "duplicate_coordinate_overlay.png",
        "extra.png",
    ]
    paths = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"x")
        paths.append(path)
    selected = _select_supervisor_images(paths, max_images=6)
    selected_names = {path.name for path in selected}
    assert {"camera_0_A.png", "camera_1_B.png", "camera_A_garment_only.png"} <= selected_names
    assert len(selected) == 6
