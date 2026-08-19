from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cloth_agent.semantic_claude import SemanticClaudeClient
from cloth_agent.semantic_pipeline import (
    ACTION_SCOPES,
    HypothesisBudgetDecision,
    LocalGeometryGrounder,
    SemanticPipelineError,
    SemanticStateBuilder,
    action_scope_from_experiences,
    build_structured_experience,
    refresh_local_geometry_artifacts,
    semantic_hypothesis_budget,
    validate_action_scope,
    validate_semantic_evaluation_payload,
    validate_semantic_strategy_payload,
)


def _anchor_manifest() -> dict:
    return {
        "status": "READY",
        "anchors": [
            {
                "anchor_id": "S001",
                "type": "right_sleeve_end",
                "camera": "A",
                "pixel_xy": [25, 25],
                "base_xyz_mm": [540.0, -20.0, 28.0],
                "height_above_table_mm": 18.0,
                "local_base_z_spread_mm": 5.0,
                "confidence": 0.92,
            }
        ],
    }


def _semantic_state() -> dict:
    return SemanticStateBuilder().build(
        _anchor_manifest(), {"center_base_mm": [500.0, 0.0, 20.0]}
    )


def _strategy_payload() -> dict:
    return {
        "semantic_objective": {
            "target_part": "right_sleeve_end",
            "desired_change": "move outward from torso",
        },
        "hypothesis": {
            "state": "possible_inward_fold",
            "confidence": 0.72,
            "rationale": "The sleeve-associated anchor lies near the centroid.",
        },
        "local_search_region": {
            "around_anchor_id": "S001",
            "radius_px": 22,
            "include_connected_fold_edge": True,
        },
        "grasp_requirement": {
            "prefer": ["free_edge", "discrete_height_step"],
            "avoid": ["flat_interior", "unrelated_height_peak"],
        },
        "expected_semantic_observation": "The sleeve-associated boundary moves outward.",
        "safety_notes": ["Keep the action inside runtime scope."],
    }


def _evaluation_payload(
    *,
    acquisition: str = "SUCCESS",
    engagement: str = "SUCCESS",
    relevance: str = "SUPPORTED",
    transport: str = "BAD_DIRECTION",
) -> dict:
    def stage(status: str) -> dict:
        return {
            "status": status,
            "confidence": 0.8,
            "evidence": ["direct visible evidence"],
        }

    return {
        "semantic_target": {
            **stage("SUPPORTED"),
            "hypothesis": "right_sleeve_end:possible_inward_fold",
        },
        "grasp_acquisition": stage(acquisition),
        "structure_engagement": stage(engagement),
        "opening_relevance": stage(relevance),
        "transport": stage(transport),
        "laydown": stage("SUCCESS"),
        "task_progress": {
            "status": "NEUTRAL",
            "confidence": 0.75,
            "metrics": {
                "visible_area_delta": "UNCHANGED",
                "overlap_delta": "UNCHANGED",
                "relief_delta": "UNCHANGED",
                "boundary_change": "Sleeve boundary moved in the wrong direction.",
            },
        },
        "earliest_failure_stage": "TRANSPORT",
        "next_experiment": {
            "keep": ["semantic_target", "local_grasp_geometry"],
            "change": ["transport_direction"],
            "reason": "Acquisition and structure engagement succeeded.",
        },
    }


def test_semantic_strategy_selects_anchor_but_not_grasp_reference() -> None:
    strategy = validate_semantic_strategy_payload(
        _strategy_payload(), semantic_state=_semantic_state()
    )
    assert strategy.anchor_id == "S001"
    assert strategy.target_part == "right_sleeve_end"
    assert "R" not in strategy.as_dict()["local_search_region"]["around_anchor_id"]


def test_semantic_strategy_rejects_target_anchor_mismatch() -> None:
    payload = _strategy_payload()
    payload["semantic_objective"]["target_part"] = "collar"
    with pytest.raises(SemanticPipelineError, match="must match"):
        validate_semantic_strategy_payload(payload, semantic_state=_semantic_state())


def test_semantic_strategy_rejects_rxxx_or_low_level_action_details() -> None:
    payload = _strategy_payload()
    payload["semantic_objective"]["desired_change"] = "grasp R024 then move(10, 0)"
    with pytest.raises(SemanticPipelineError, match="forbidden"):
        validate_semantic_strategy_payload(payload, semantic_state=_semantic_state())


def test_semantic_state_marks_identity_and_relation_as_hypotheses() -> None:
    state = _semantic_state()
    assert state["known"]["anchors"][0]["semantic_identity_status"] == "HYPOTHESIS"
    assert state["relations"]["S001"]["near_torso_overlap"] is True
    assert state["hypotheses"][0]["state"] == "possible_inward_fold"


def test_local_geometry_generates_rxxx_only_inside_selected_semantic_region(
    tmp_path: Path,
) -> None:
    perception = tmp_path / "perception_views"
    perception.mkdir()
    height, width = 60, 60
    Image.new("RGB", (width, height), "white").save(perception / "camera_0_A.png")
    mask = np.zeros((height, width), dtype=bool)
    mask[12:48, 10:50] = True
    yy, xx = np.indices((height, width))
    height_map = np.where(mask, 5.0 + np.maximum(0, xx - 25) * 0.8, np.nan)
    xyz = np.full((height, width, 3), np.nan, dtype=np.float32)
    xyz[..., 0] = 400.0 + xx
    xyz[..., 1] = -100.0 + yy
    xyz[..., 2] = 10.0 + np.nan_to_num(height_map, nan=0.0)
    np.save(perception / "camera_A_base_xyz_mm.npy", xyz)
    np.save(perception / "camera_A_height_above_table_mm.npy", height_map)
    np.save(perception / "camera_A_garment_mask.npy", mask)
    (perception / "camera_A_coordinate_guide.json").write_text(
        json.dumps({"samples": []}), encoding="utf-8"
    )
    (perception / "camera_B_coordinate_guide.json").write_text(
        json.dumps({"samples": []}), encoding="utf-8"
    )
    strategy = validate_semantic_strategy_payload(
        _strategy_payload(), semantic_state=_semantic_state()
    )
    manifest = LocalGeometryGrounder(max_candidates=4).ground(
        perception_dir=perception,
        artifact_dir=tmp_path / "local",
        semantic_state=_semantic_state(),
        strategy=strategy,
        install=True,
    )
    assert 1 <= manifest["candidate_count"] <= 4
    assert all(item["reference_id"].startswith("R") for item in manifest["candidates"])
    assert all(item["semantic_anchor_id"] == "S001" for item in manifest["candidates"])
    assert all(
        (item["pixel_xy"][0] - 25) ** 2 + (item["pixel_xy"][1] - 25) ** 2
        <= (22 * 1.75) ** 2
        for item in manifest["candidates"]
    )
    assert manifest["connected_fold_pixels_outside_radius"] > 0
    guide = json.loads(
        (perception / "camera_A_coordinate_guide.json").read_text(encoding="utf-8")
    )
    assert guide["measurement_kind"] == "semantic_region_local_geometry_grasp_candidate"
    assert "not Molmo anchors" in guide["reference_semantics"]
    other = json.loads(
        (perception / "camera_B_coordinate_guide.json").read_text(encoding="utf-8")
    )
    assert other["samples"] == []

    rejected = [
        {**item, "rejection_reason": "workspace_or_controller_ik"}
        for item in manifest["candidates"][1:]
    ]
    filtered = refresh_local_geometry_artifacts(
        perception_dir=perception,
        artifact_dir=tmp_path / "local",
        manifest={
            **manifest,
            "candidate_count": 1,
            "candidates": manifest["candidates"][:1],
            "capability_rejected_candidates": rejected,
        },
        install=True,
    )
    filtered_guide = json.loads(
        Path(filtered["coordinate_guide"]).read_text(encoding="utf-8")
    )
    assert filtered_guide["samples"] == manifest["candidates"][:1]
    assert filtered_guide["capability_rejected_count"] == len(rejected)
    assert "not Molmo anchors" in filtered_guide["reference_semantics"]


def test_action_scope_is_runtime_owned_and_progresses_by_stage_evidence() -> None:
    assert action_scope_from_experiences([]).name == "ACQUISITION_CHECK"
    hypothesis = "right_sleeve_end:possible_inward_fold"
    acquisition_only = [
        {
            "hypothesis_key": hypothesis,
            "evaluation": _evaluation_payload(
                engagement="UNKNOWN", relevance="UNKNOWN", transport="UNKNOWN"
            ),
        }
    ]
    assert (
        action_scope_from_experiences(acquisition_only, hypothesis_key=hypothesis).name
        == "STRUCTURE_CHECK"
    )
    assert (
        action_scope_from_experiences(
            acquisition_only,
            hypothesis_key=hypothesis,
            budget_disposition="KEEP_SEMANTIC_CHANGE_GRASP",
        ).name
        == "ACQUISITION_CHECK"
    )
    engaged = [
        {
            "hypothesis_key": hypothesis,
            "evaluation": _evaluation_payload(relevance="UNKNOWN", transport="UNKNOWN"),
        }
    ]
    assert (
        action_scope_from_experiences(engaged, hypothesis_key=hypothesis).name
        == "TRANSPORT_TEST"
    )
    wrong_direction = [
        {
            "hypothesis_key": hypothesis,
            "evaluation": _evaluation_payload(
                relevance="UNKNOWN", transport="BAD_DIRECTION"
            ),
        }
    ]
    assert (
        action_scope_from_experiences(wrong_direction, hypothesis_key=hypothesis).name
        == "TRANSPORT_CORRECTION"
    )
    supported = [
        {
            "hypothesis_key": hypothesis,
            "evaluation": _evaluation_payload(transport="GOOD"),
        }
    ]
    assert (
        action_scope_from_experiences(supported, hypothesis_key=hypothesis).name
        == "OPENING_COMPLETION"
    )


def test_acquisition_scope_blocks_claude_from_large_transport() -> None:
    candidate = {"base_xyz_mm": [500.0, 0.0, 20.0]}
    actions = [
        {"name": "move", "args": {"x": 500, "y": 0, "z": 25, "yaw": 0}},
        {"name": "close_gripper", "args": {}},
        {"name": "move", "args": {"x": 540, "y": 0, "z": 35, "yaw": 0}},
        {"name": "open_gripper", "args": {}},
    ]
    with pytest.raises(SemanticPipelineError, match="lateral authority"):
        validate_action_scope(
            actions, candidate=candidate, scope=ACTION_SCOPES["ACQUISITION_CHECK"]
        )


def test_action_scope_blocks_multiple_probe_moves_in_one_acquisition_cycle() -> None:
    candidate = {"base_xyz_mm": [500.0, 0.0, 20.0]}
    actions = [
        {"name": "move", "args": {"x": 500, "y": 0, "z": 20, "yaw": 0}},
        {"name": "close_gripper", "args": {}},
        {"name": "move", "args": {"x": 502, "y": 0, "z": 30, "yaw": 0}},
        {"name": "move", "args": {"x": 498, "y": 0, "z": 30, "yaw": 0}},
        {"name": "open_gripper", "args": {}},
    ]
    with pytest.raises(SemanticPipelineError, match="post-grasp move budget"):
        validate_action_scope(
            actions, candidate=candidate, scope=ACTION_SCOPES["ACQUISITION_CHECK"]
        )


def test_semantic_evaluator_splits_engagement_from_opening_relevance() -> None:
    evaluation = validate_semantic_evaluation_payload(_evaluation_payload())
    assert evaluation.structure_engagement.status == "SUCCESS"
    assert evaluation.opening_relevance.status == "SUPPORTED"
    assert evaluation.transport.status == "BAD_DIRECTION"
    assert evaluation.earliest_failure_stage == "TRANSPORT"


def test_semantic_evaluator_rejects_inconsistent_earliest_failure() -> None:
    payload = _evaluation_payload()
    payload["earliest_failure_stage"] = "ACQUISITION"
    with pytest.raises(SemanticPipelineError, match="inconsistent"):
        validate_semantic_evaluation_payload(payload)


def test_hypothesis_budget_persists_locally_then_escapes() -> None:
    hypothesis = "right_sleeve_end:possible_inward_fold"
    failed = _evaluation_payload(
        acquisition="FAILURE",
        engagement="UNKNOWN",
        relevance="UNKNOWN",
        transport="UNKNOWN",
    )
    one = [{"hypothesis_key": hypothesis, "evaluation": failed}]
    decision = semantic_hypothesis_budget(
        one, hypothesis_key=hypothesis, anchor_id="S001"
    )
    assert decision.disposition == "KEEP_SEMANTIC_CHANGE_GRASP"
    assert decision.forced_anchor_id == "S001"
    two = one * 2
    exhausted = semantic_hypothesis_budget(
        two, hypothesis_key=hypothesis, anchor_id="S001"
    )
    assert exhausted.disposition == "ESCAPE_HYPOTHESIS"
    assert exhausted.forced_anchor_id is None


def test_bad_transport_keeps_supported_grasp_geometry_family() -> None:
    hypothesis = "right_sleeve_end:possible_inward_fold"
    decision = semantic_hypothesis_budget(
        [
            {
                "hypothesis_key": hypothesis,
                "chosen_structure": {"geometry_type": "free_edge"},
                "evaluation": _evaluation_payload(transport="BAD_DIRECTION"),
            }
        ],
        hypothesis_key=hypothesis,
        anchor_id="S009",
    )
    assert decision.disposition == "KEEP_TARGET_AND_GRASP_CHANGE_TRANSPORT"
    assert decision.forced_anchor_id == "S009"
    assert decision.forced_geometry_type == "free_edge"


def test_unknown_structure_engagement_keeps_grasp_for_structure_check() -> None:
    hypothesis = "right_sleeve_end:possible_inward_fold"
    decision = semantic_hypothesis_budget(
        [
            {
                "hypothesis_key": hypothesis,
                "chosen_structure": {"geometry_type": "raised_fold_edge"},
                "evaluation": _evaluation_payload(
                    engagement="UNKNOWN",
                    relevance="UNKNOWN",
                    transport="UNKNOWN",
                ),
            }
        ],
        hypothesis_key=hypothesis,
        anchor_id="S004",
    )
    assert decision.disposition == "KEEP_TARGET_AND_GRASP_CHECK_STRUCTURE"
    assert decision.forced_anchor_id == "S004"
    assert decision.forced_geometry_type == "raised_fold_edge"


def test_strategy_client_rejects_reuse_of_exhausted_hypothesis(
    tmp_path: Path,
) -> None:
    payload = _strategy_payload()

    class PayloadClient(SemanticClaudeClient):
        def _invoke_json(self, **kwargs):
            return kwargs["validator"](payload), {"validated": payload}

    budget = HypothesisBudgetDecision(
        hypothesis_key="right_sleeve_end:possible_inward_fold",
        acquisition_attempts=2,
        transport_attempts=0,
        max_acquisition_attempts=2,
        max_transport_attempts=3,
        disposition="ESCAPE_HYPOTHESIS",
        forced_anchor_id=None,
        forced_geometry_type=None,
        reason="Acquisition retry budget is exhausted.",
    )
    with pytest.raises(SemanticPipelineError, match="exhausted its budget"):
        PayloadClient().plan_strategy(
            images=[],
            run_dir=tmp_path,
            semantic_state=_semantic_state(),
            experiences=[],
            budget=budget,
        )


def test_semantic_claude_can_read_cli_artifacts_outside_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    artifact_dir = tmp_path / "custom_cli_output"
    run_dir.mkdir()
    artifact_dir.mkdir()
    image = artifact_dir / "local.png"
    Image.new("RGB", (4, 4), "white").save(image)
    seen: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(
            command, 0, stdout='{"ok": true}', stderr=""
        )

    monkeypatch.setattr("cloth_agent.semantic_claude.subprocess.run", fake_run)
    result, _ = SemanticClaudeClient(binary="/bin/true")._invoke_json(
        run_dir=run_dir,
        stage="test",
        prompt="Return ok.",
        schema={"type": "object"},
        timeout_s=30,
        images=[image],
        validator=lambda value: value,
    )

    assert result == {"ok": True}
    add_dir_index = seen["command"].index("--add-dir")
    add_dirs = seen["command"][add_dir_index + 1 :]
    assert str(run_dir.resolve()) in add_dirs
    assert str(artifact_dir.resolve()) in add_dirs


def test_structured_experience_is_relation_based_not_coordinate_history() -> None:
    strategy = validate_semantic_strategy_payload(
        _strategy_payload(), semantic_state=_semantic_state()
    )
    evaluation = validate_semantic_evaluation_payload(_evaluation_payload())
    candidate = {
        "reference_id": "R003",
        "feature": "free_edge",
        "free_boundary": True,
        "height_step_mm": 6.2,
        "relief_above_local_median_mm": 5.5,
        "base_xyz_mm": [600, 20, 30],
    }
    experience = build_structured_experience(
        iteration=2,
        semantic_state=_semantic_state(),
        strategy=strategy,
        candidate=candidate,
        action_scope=ACTION_SCOPES["TRANSPORT_TEST"],
        evaluation=evaluation,
    )
    assert experience["semantic_state"]["target_part"] == "right_sleeve_end"
    assert experience["chosen_structure"]["geometry_type"] == "free_edge"
    assert "base_xyz_mm" not in experience["chosen_structure"]
    assert "R003" not in json.dumps(experience)
    assert "S001" not in json.dumps(experience)
