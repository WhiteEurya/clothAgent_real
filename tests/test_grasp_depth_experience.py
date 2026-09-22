"""Relative depth learning: no robot, model service, or network required."""
from copy import deepcopy
import json

import pytest

from cloth_agent.fold_experience_learning import validate_schema
from cloth_agent.grasp_depth_experience import (
    GRASP_DEPTH_DIAGNOSIS_SCHEMA, GRASP_EXPERIENCE_SCHEMA, gate_depth_experiment,
    initial_grasp_experience, make_grasp_experience, persist_depth_trial,
    runtime_geometry, unknown_diagnosis,
)
from cloth_agent.perception_comparison import apply_comparison_policy


def record(surface=30., commanded=27., planned=27.):
    return {"planning_diagnostics": {"grasp_height_resolution": {
        "requested_grasp_z_mm": planned, "resolved_grasp_z_mm": planned,
        "resolved_grasp_xy_mm": [500., 40.],
        "measurement": {"valid": True},
        "resolution": {"valid": True, "surface_z_mm": surface,
            "surface_xyz_mm": [500., 40., surface], "authoritative_table_z_mm": 5.}}},
        "proposal": {"actions": [{"name": "move", "args": {"z": planned}}]},
        "execution": {"physical_execution": True, "execution_completed": True,
            "requested_robot_actions": [{"name": "move", "args": {"z": planned}}],
            "actual_robot_actions": [
                {"name": "move", "success": True, "args": {"x": 500., "y": 40., "z": commanded},
                 "actual_ee_pose": [500., 40., 28., 180., 0., 0.]},
                {"name": "close_gripper", "success": True, "args": {}},
                {"name": "move", "success": True, "args": {"x": 500., "y": 40., "z": 60.}}]},
        "evaluation": {"grasp_acquisition": {"status": "SUCCESS"}}}


def context(row=None):
    row = row if row is not None else record()
    return {"trial_id": "trial_1", "step": "left_sleeve",
        "grasp_depth_geometry": runtime_geometry(row), "outcome": row["evaluation"],
        "evidence_catalog": [
            {"id": "policy_outcome", "kind": "POLICY_LABEL"},
            {"id": "execution_log", "kind": "ROBOT_ACTIONS_NOT_CLOTH_ACQUISITION"},
            {"id": "raw_visual_evaluation", "kind": "MODEL_INTERPRETATION_NOT_CAUSAL_PROOF"},
            {"id": "current_after_rgb", "kind": "END_STATE_OR_PRIOR_RGB"},
            {"id": "prior_after_lift_rgb", "kind": "END_STATE_OR_PRIOR_RGB"},
            {"id": "current_after_close_rgb", "kind": "INTERACTION_RGB"},
            {"id": "current_after_lift_rgb", "kind": "INTERACTION_RGB"},
        ]}


def diagnosis(interpretation="EFFECTIVE", observation="STABLE_CLOTH_ACQUISITION",
              evidence_id="current_after_lift_rgb"):
    return {"depth_interpretation": interpretation, "confidence": .9, "evidence": [{
        "observation": observation, "evidence_ids": [evidence_id],
        "description": "Direct, unoccluded current interaction evidence for the specified observation."}]}


@pytest.mark.parametrize("interpretation,observation,evidence_id,direction", [
    ("EFFECTIVE", "STABLE_CLOTH_ACQUISITION", "current_after_lift_rgb", "KEEP"),
    ("TOO_SHALLOW", "JAWS_ABOVE_CLOTH_AT_CLOSURE", "current_after_close_rgb", "DEEPER"),
    ("TOO_DEEP", "TABLE_CONTACT", "current_after_close_rgb", "SHALLOWER"),
    ("TOO_DEEP", "DOWNWARD_MOTION_BLOCKED_BY_TABLE", "current_after_close_rgb", "SHALLOWER"),
    ("TOO_DEEP", "EXCESSIVE_COMPRESSION", "current_after_close_rgb", "SHALLOWER"),
])
def test_a_b_c_depth_specific_evidence(interpretation, observation, evidence_id, direction):
    ctx = context()
    payload = diagnosis(interpretation, observation, evidence_id)
    original = deepcopy(payload)
    result = make_grasp_experience(payload, ctx)
    assert result["geometry"]["descent_below_surface_mm"] == 3
    assert result["geometry"]["clearance_above_table_mm"] == 22
    assert result["geometry"]["surface_z_confidence"] is None  # Not invented from valid=True.
    assert result["outcome"]["depth_interpretation"] == interpretation
    assert result["experience_update"]["update_direction"] == direction
    assert payload == original
    validate_schema(result, GRASP_EXPERIENCE_SCHEMA)


def test_d_unchanged_policy_is_not_a_shallow_depth_observation():
    payload = {"grasp_acquisition": {"status": "SUCCESS"}, "earliest_failure_stage": "NONE",
        "task_progress": {"status": "IMPROVED"}, "next_experiment": {"reason": "test", "change": []},
        "perception_comparison": {"status": "UNCHANGED", "confidence": .95, "evidence": ["Same final garment RGB"]},
        "skill_update": {"lesson": "lower grasp"}}
    normalized = apply_comparison_policy(payload)
    row = record()
    row["evaluation"] = normalized
    assert normalized["grasp_acquisition"]["status"] == "FAILURE"
    assert normalized["earliest_failure_stage"] == "ACQUISITION"
    assert normalized["task_progress"]["status"] == "NEUTRAL"
    assert "skill_update" not in normalized
    assert payload["grasp_acquisition"]["status"] == "SUCCESS"
    for source in ("policy_outcome", "current_after_rgb", "raw_visual_evaluation", "execution_log"):
        result = make_grasp_experience(diagnosis("TOO_SHALLOW", "JAWS_ABOVE_CLOTH_AT_CLOSURE", source), context(row))
        assert result["outcome"]["acquisition_status"] == "FAILURE"
        assert result["outcome"]["depth_interpretation"] == "UNKNOWN"
        assert result["experience_update"]["update_direction"] == "NONE"
    assert initial_grasp_experience(row)["outcome"]["depth_interpretation"] == "UNKNOWN"
    # Independent depth evidence can coexist with the configured failure label.
    result = make_grasp_experience(diagnosis("TOO_DEEP", "TABLE_CONTACT", "current_after_close_rgb"), context(row))
    assert result["outcome"]["acquisition_status"] == "FAILURE"
    assert result["experience_update"]["update_direction"] == "SHALLOWER"


def test_e_actual_command_has_priority_over_plan_resolved_target_and_measured_pose():
    geometry = runtime_geometry(record(planned=27., commanded=27.5))
    assert geometry["actual_commanded_grasp_z_mm"] == 27.5
    assert geometry["descent_below_surface_mm"] == 2.5
    assert geometry["command_source"] == "execution.actual_robot_actions[0].args.z"
    assert geometry["runtime_authoritative_surface_z_mm"] == 30


def test_f_point_or_motion_failure_without_depth_evidence_cannot_change_depth():
    row = record()
    row["evaluation"]["grasp_acquisition"]["status"] = "FAILURE"
    payload = unknown_diagnosis()
    payload["evidence"][0].update(evidence_ids=["current_after_lift_rgb"],
        description="The entire garment moved together: coupling or target selection, no depth evidence.")
    result = make_grasp_experience(payload, context(row))
    assert result["experience_update"]["update_direction"] == "NONE"
    # A wrong drag/release does not turn successful acquisition into shallow/deep.
    row["evaluation"].update(grasp_acquisition={"status": "SUCCESS"}, transport={"status": "FAILURE"})
    result = make_grasp_experience(diagnosis(), context(row))
    assert result["experience_update"]["update_direction"] == "KEEP"


@pytest.mark.parametrize("mutate", [
    lambda r: r["execution"].update(physical_execution=False),
    lambda r: r["execution"].pop("actual_robot_actions"),
    lambda r: r["execution"]["actual_robot_actions"][0].update(success=False),
    lambda r: r["execution"]["actual_robot_actions"][1].update(success=False),
    lambda r: r["execution"]["actual_robot_actions"][0]["args"].update(z=float("nan")),
    lambda r: r["execution"]["actual_robot_actions"][0]["args"].update(z=True),
    lambda r: r["execution"]["actual_robot_actions"][0]["args"].update(x=501),
    lambda r: r["execution"]["actual_robot_actions"].insert(1, {"name": "home", "success": True}),
    lambda r: r["execution"]["actual_robot_actions"].append({"name": "close_gripper", "success": True}),
    lambda r: r.pop("planning_diagnostics"),
    lambda r: r["planning_diagnostics"]["grasp_height_resolution"]["resolution"].update(valid=False),
    lambda r: r["planning_diagnostics"]["grasp_height_resolution"]["resolution"].update(surface_z_mm=float("inf")),
])
def test_missing_invalid_or_unpaired_geometry_never_falls_back_to_plan(mutate):
    row = record()
    mutate(row)
    result = make_grasp_experience(diagnosis(), context(row))
    assert result["geometry"]["descent_below_surface_mm"] is None
    assert result["experience_update"]["update_direction"] == "NONE"


def test_optional_table_and_confidence_are_nullable_and_legacy_surface_pair_is_supported():
    row = record()
    audit = row["planning_diagnostics"]["grasp_height_resolution"]
    audit.pop("resolved_grasp_xy_mm")
    audit["resolution"]["authoritative_table_z_mm"] = float("nan")
    geometry = runtime_geometry(row)
    assert geometry["descent_below_surface_mm"] == 3
    assert geometry["table_z_mm"] is None
    assert geometry["clearance_above_table_mm"] is None
    audit["measurement"]["surface_z_confidence"] = .86
    assert runtime_geometry(row)["surface_z_confidence"] == .86


def test_finite_inputs_cannot_produce_infinite_descent():
    row = record(surface=1e308, commanded=-1e308)
    assert runtime_geometry(row)["descent_below_surface_mm"] is None
    assert initial_grasp_experience(row)["experience_update"]["update_direction"] == "NONE"


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(confidence=float("inf")),
    lambda d: d.update(confidence=float("nan")),
    lambda d: d.update(confidence=True),
    lambda d: d.update(confidence=1.01),
    lambda d: d.update(evidence=[]),
    lambda d: d["evidence"][0].update(evidence_ids=[]),
    lambda d: d["evidence"][0].update(description=" "),
    lambda d: d["evidence"][0].update(observation="WHOLE_GARMENT_COUPLING"),
    lambda d: d["evidence"][0].update(evidence_ids=["invented_sensor"]),
    lambda d: d.update(actual_commanded_grasp_z_mm=27),
    lambda d: d.pop("confidence"),
])
def test_strict_schema_and_traceable_evidence(mutate):
    payload = diagnosis()
    mutate(payload)
    with pytest.raises(ValueError):
        make_grasp_experience(payload, context())


def test_conflicting_low_confidence_prior_or_wrong_phase_evidence_cannot_update():
    low = diagnosis()
    low["confidence"] = .69
    conflicting = diagnosis()
    conflicting["evidence"].extend(diagnosis("TOO_DEEP", "TABLE_CONTACT", "current_after_close_rgb")["evidence"])
    for payload in (low, conflicting, diagnosis(evidence_id="prior_after_lift_rgb"),
                    diagnosis(evidence_id="current_after_close_rgb"),
                    diagnosis("TOO_SHALLOW", "JAWS_ABOVE_CLOTH_AT_CLOSURE", "current_after_lift_rgb")):
        assert make_grasp_experience(payload, context())["experience_update"]["update_direction"] == "NONE"


def test_numeric_ledger_deduplicates_and_unknown_makes_no_permanent_update(tmp_path):
    ctx = context(record(planned=27., commanded=27.5))
    ctx["grasp_depth_diagnosis"] = unknown_diagnosis()
    result = make_grasp_experience(ctx["grasp_depth_diagnosis"], ctx)
    assert persist_depth_trial(tmp_path, result, ctx, source_record="trial.json")["status"] == "NO_DEPTH_UPDATE"
    assert not list(tmp_path.iterdir())
    ctx["grasp_depth_diagnosis"] = diagnosis()
    result = make_grasp_experience(ctx["grasp_depth_diagnosis"], ctx)
    assert persist_depth_trial(tmp_path, result, ctx, source_record="trial.json")["status"] == "RECORDED"
    assert persist_depth_trial(tmp_path, result, ctx, source_record="trial.json")["status"] == "ALREADY_APPLIED"
    path = tmp_path / "grasp_depth_trials.json"
    before = path.read_bytes()
    ctx.update(trial_id="trial_2", grasp_depth_diagnosis=unknown_diagnosis())
    persist_depth_trial(tmp_path, make_grasp_experience(unknown_diagnosis(), ctx), ctx, source_record="second.json")
    assert path.read_bytes() == before
    trials = json.loads(before)["trials"]
    assert len(trials) == 1
    assert trials["trial_1"]["grasp_experience"]["geometry"]["descent_below_surface_mm"] == 2.5
    tampered = deepcopy(result)
    tampered["geometry"]["descent_below_surface_mm"] = 3
    with pytest.raises(ValueError, match="host-validated"):
        persist_depth_trial(tmp_path, tampered, ctx, source_record="bad.json")


def test_relative_offsets_transfer_not_absolute_world_z():
    first = make_grasp_experience(diagnosis(), context(record(surface=30, commanded=27)))
    second = make_grasp_experience(diagnosis(), context(record(surface=42, commanded=39)))
    assert first["geometry"]["descent_below_surface_mm"] == second["geometry"]["descent_below_surface_mm"] == 3
    assert first["experience_update"] == second["experience_update"]


def test_next_experiment_leaves_other_domains_unchanged():
    other = {"status": "PROPOSED", "single_change": {"variable": "CONTACT_XY", "description": "test point"}}
    original = deepcopy(other)
    gate_depth_experiment(other, make_grasp_experience(unknown_diagnosis(), context()))
    assert other == original
    depth = {"status": "PROPOSED", "single_change": {"variable": "CONTACT_Z", "description": "deeper"}}
    gate_depth_experiment(depth, make_grasp_experience(unknown_diagnosis(), context()))
    assert depth["status"] == "NO_EXPERIMENT"
    assert depth["single_change"]["variable"] == "NONE"
    for interpretation, observation in (("TOO_SHALLOW", "JAWS_ABOVE_CLOTH_AT_CLOSURE"), ("TOO_DEEP", "TABLE_CONTACT")):
        depth["single_change"]["variable"] = "CONTACT_Z"
        gate_depth_experiment(depth, make_grasp_experience(diagnosis(interpretation, observation, "current_after_close_rgb"), context()))
        assert depth["status"] == "BLOCKED_BY_CAPABILITY"
        assert "relative" in depth["single_change"]["description"]


def test_all_schema_objects_are_closed_and_require_their_fields():
    def walk(schema):
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])
            for value in schema["properties"].values():
                walk(value)
        if schema.get("type") == "array":
            walk(schema["items"])
    walk(GRASP_DEPTH_DIAGNOSIS_SCHEMA)
    walk(GRASP_EXPERIENCE_SCHEMA)
    payload = make_grasp_experience(diagnosis(), context())
    payload["geometry"]["descent_below_surface_mm"] = float("nan")
    with pytest.raises(ValueError):
        validate_schema(payload, GRASP_EXPERIENCE_SCHEMA)
