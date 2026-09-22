"""Host geometry integrity and evidence-localized XYZ experience; no hardware."""
from copy import deepcopy
import json

import pytest

from cloth_agent.fold_experience_learning import validate_schema
from cloth_agent.grasp_execution_experience import (
    GRASP_EXECUTION_DIAGNOSIS_SCHEMA, GRASP_EXECUTION_EXPERIENCE_SCHEMA,
    apply_grasp_execution_experience, gate_execution_experiment,
    initial_execution_experience, persist_execution_trial, runtime_execution_trial,
    unresolved_diagnosis,
)
from cloth_agent.perception_comparison import apply_comparison_policy, validate_comparison_execution


def record():
    return {"planning_diagnostics": {"grasp_height_resolution": {
        "camera": "A", "pixel_xy": [120, 240], "requested_grasp_z_mm": 27,
        "runtime_authoritative_grasp_xyz_mm": [502, 38.5, 27],
        "resolved_grasp_xy_mm": [502, 38.5], "resolved_grasp_z_mm": 27,
        "resolution": {"valid": True, "surface_xyz_mm": [500, 40, 30], "surface_z_mm": 30}}},
        "proposal": {"actions": [{"name": "move", "args": {"z": 20}}]},
        "execution": {"physical_execution": True, "execution_completed": True,
            "actual_robot_actions": [
                {"name": "move", "success": True, "args": {"x": 502, "y": 38.5, "z": 27},
                 "actual_ee_pose": [503, 40, 28, 180, 0, 0]},
                {"name": "close_gripper", "success": True, "args": {}},
                {"name": "move", "success": True, "args": {"x": 502, "y": 38.5, "z": 57}}]},
        "evaluation": {"grasp_acquisition": {"status": "FAILURE"},
                       "perception_comparison": {"status": "UNCHANGED"}}}


def context(row=None):
    row = row or record()
    return {"trial_id": "trial_1", "step": "left_sleeve", "outcome": row["evaluation"],
        "grasp_execution_trial": runtime_execution_trial(row),
        "evidence_catalog": [
            {"id": "policy_outcome", "kind": "POLICY_LABEL"},
            {"id": "current_before_rgb", "kind": "END_STATE_OR_PRIOR_RGB"},
            {"id": "current_after_rgb", "kind": "END_STATE_OR_PRIOR_RGB"},
            {"id": "prior_after_close_rgb", "kind": "END_STATE_OR_PRIOR_RGB"},
            {"id": "raw_visual_evaluation", "kind": "MODEL_INTERPRETATION_NOT_CAUSAL_PROOF"},
            {"id": "execution_log", "kind": "ROBOT_ACTIONS_NOT_CLOTH_ACQUISITION"},
            *[{"id": f"current_{r}_rgb", "kind": "INTERACTION_RGB"}
              for r in ("after_close", "after_lift", "rollout")],
        ]}


def diagnosis(*observations):
    return {"confidence": .9, "evidence": [{
        "observation": observation,
        "evidence_ids": ["current_after_lift_rgb" if observation == "STABLE_CLOTH_ACQUISITION" else "current_after_close_rgb"],
        "description": "Selected endpoint and jaw contact identifiable in the interaction frame; direct observation.",
    } for observation in observations]}


@pytest.mark.parametrize("observations,status,axes,z", [
    (("XY_ALIGNED", "STABLE_CLOTH_ACQUISITION"), "ALIGNED_SUCCESS", ["X", "Y", "Z"], "KEEP"),
    (("XY_MISALIGNED",), "XY_MISALIGNED", ["X", "Y"], "NONE"),
    (("XY_ALIGNED", "JAWS_ABOVE_CLOTH_AT_CLOSURE"), "Z_MISALIGNED", ["Z"], "DEEPER"),
    (("XY_ALIGNED", "TABLE_CONTACT"), "Z_MISALIGNED", ["Z"], "SHALLOWER"),
    (("XY_MISALIGNED", "TABLE_CONTACT"), "XYZ_MISALIGNED", ["X", "Y", "Z"], "SHALLOWER"),
    (("XY_MISALIGNED", "JAWS_ABOVE_CLOTH_AT_CLOSURE"), "XYZ_MISALIGNED", ["X", "Y", "Z"], "DEEPER"),
    (("INCONCLUSIVE",), "UNRESOLVED", [], "NONE"),
    # A shallow/deep claim cannot become Z-only without XY alignment evidence.
    (("JAWS_ABOVE_CLOTH_AT_CLOSURE",), "UNRESOLVED", [], "NONE"),
    (("TABLE_CONTACT",), "UNRESOLVED", [], "NONE"),
    (("STABLE_CLOTH_ACQUISITION",), "UNRESOLVED", [], "NONE"),
    (("XY_ALIGNED",), "UNRESOLVED", [], "NONE"),
])
def test_localized_statuses_require_independent_axis_evidence(observations, status, axes, z):
    payload, ctx = diagnosis(*observations), context()
    before = deepcopy((payload, ctx))
    result = apply_grasp_execution_experience(payload, ctx)
    assert result["status"] == status
    assert result["update_allowed"] == (status != "UNRESOLVED")
    assert result["experience_update"]["axes"] == axes
    assert result["experience_update"]["z_direction"] == z
    assert (payload, ctx) == before
    if status == "ALIGNED_SUCCESS":
        assert result["experience_update"]["correction_xyz_mm"] == {"x": 2, "y": -1.5, "z": -3}
        assert result["observed_result"]["acquisition"] == "SUCCESS"
        assert result["observed_result"]["policy_acquisition"] == "FAILURE"
    else:
        assert result["experience_update"]["correction_xyz_mm"] == {"x": None, "y": None, "z": None}
    validate_schema(result, GRASP_EXECUTION_EXPERIENCE_SCHEMA)


def test_three_geometry_layers_and_observed_contact_are_not_conflated():
    trial = runtime_execution_trial(record())
    assert trial["selected_point"] == {"pixel_xy": [120, 240], "surface_xyz_mm": [500, 40, 30],
        "camera": "A", "pixel_frame": "RAW_PERCEPTION",
        "source": "planning_diagnostics.grasp_height_resolution.resolution.surface_xyz_mm"}
    assert trial["execution"]["runtime_authoritative_xyz_mm"] == [502, 38.5, 27]
    assert trial["execution"]["commanded_xyz_mm"] == [502, 38.5, 27]
    assert trial["execution"]["command_minus_surface_xyz_mm"] == [2, -1.5, -3]
    assert trial["execution"]["command_minus_authoritative_xyz_mm"] == [0, 0, 0]
    assert trial["execution"]["descent_below_surface_mm"] == 3
    assert trial["observed_contact"] is None  # Never use measured TCP as cloth contact.
    assert trial["command_integrity"]["status"] == "MATCHED"


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_authoritative_vs_command_mismatch_is_system_failure_not_compensation(axis, tmp_path):
    row = record()
    row["execution"]["actual_robot_actions"][0]["args"][axis] += .5
    ctx = context(row)
    payload = diagnosis("XY_ALIGNED", "STABLE_CLOTH_ACQUISITION")
    result = apply_grasp_execution_experience(payload, ctx)
    assert result["trial"]["command_integrity"]["status"] == "SYSTEM_CODE_FAILURE"
    assert result["trial"]["command_integrity"]["mismatched_axes"] == [axis.upper()]
    assert result["status"] == "UNRESOLVED"
    assert result["update_allowed"] is False
    assert result["experience_update"]["kind"] == "NONE"
    assert "Investigate" in result["next_experiment"]["recommendation"]
    if axis == "z":
        assert result["trial"]["execution"]["commanded_xyz_mm"][2] == 27.5
        assert result["trial"]["execution"]["descent_below_surface_mm"] == 2.5
    receipt = persist_execution_trial(tmp_path, result, {**ctx, "grasp_execution_diagnosis": payload}, source_record="r.json")
    assert receipt["status"] == "NO_EXECUTION_UPDATE"
    assert not list(tmp_path.iterdir())


def test_planned_z_can_differ_if_authoritative_target_and_actual_command_agree():
    row = record()
    row["planning_diagnostics"]["grasp_height_resolution"]["runtime_authoritative_grasp_xyz_mm"][2] = 27.5
    row["execution"]["actual_robot_actions"][0]["args"]["z"] = 27.5
    result = apply_grasp_execution_experience(diagnosis("XY_ALIGNED", "STABLE_CLOTH_ACQUISITION"), context(row))
    assert result["status"] == "ALIGNED_SUCCESS"
    assert result["experience_update"]["correction_xyz_mm"]["z"] == -2.5


def test_unchanged_policy_adds_unresolved_summary_removes_multi_variable_retry():
    raw = {"grasp_acquisition": {"status": "SUCCESS"}, "earliest_failure_stage": "NONE",
        "task_progress": {"status": "IMPROVED"}, "next_experiment": {"reason": "test", "keep": ["target"], "change": ["lower 1 mm"]},
        "perception_comparison": {"status": "UNCHANGED", "confidence": .95, "evidence": ["Same final outline"]},
        "skill_update": {"name": "lower"}, "grasp_execution_experience": {"status": "Z_MISALIGNED", "update_allowed": True}}
    result = apply_comparison_policy(raw)
    assert result["grasp_acquisition"]["status"] == "FAILURE"
    assert result["earliest_failure_stage"] == "ACQUISITION"
    assert result["task_progress"]["status"] == "NEUTRAL"
    assert result["grasp_execution_experience"]["status"] == "UNRESOLVED"
    assert result["grasp_execution_experience"]["update_allowed"] is False
    assert "contact-alignment evidence" in result["next_experiment"]["change"][0]
    assert "does not identify" in result["next_experiment"]["reason"]
    assert "skill_update" not in result
    assert raw["grasp_execution_experience"]["update_allowed"] is True
    assert apply_comparison_policy(result) == result
    validate_comparison_execution(result["grasp_execution_experience"])


def test_comparison_summary_survives_evaluation_dataclass_roundtrip():
    from cloth_agent.auto_exploration import validate_evaluation_payload
    def stage(status):
        return {"status": status, "confidence": .5, "evidence": ["Not localized"]}
    payload = {"target_selection": stage("UNKNOWN"), "grasp_acquisition": stage("UNKNOWN"),
        "target_structure_acquired": stage("UNKNOWN"), "transport": stage("UNKNOWN"), "laydown": stage("NOT_REACHED"),
        "task_progress": {"status": "NEUTRAL", "confidence": .5, "metrics": {
            "visible_area_delta": "UNKNOWN", "overlap_delta": "UNKNOWN", "relief_delta": "UNKNOWN", "boundary_change": "Unchanged"}},
        "earliest_failure_stage": "UNKNOWN", "next_experiment": {"keep": ["target"], "change": ["observe"], "reason": "uncertain"},
        "perception_comparison": {"status": "UNCHANGED", "confidence": .9, "evidence": ["Same shape"]}}
    normalized = apply_comparison_policy(payload)
    evaluation = validate_evaluation_payload(normalized)
    saved = evaluation.as_dict()
    assert saved["grasp_execution_experience"] == normalized["grasp_execution_experience"]
    saved["grasp_execution_experience"]["evidence"].append("mutated")
    assert saved["grasp_execution_experience"] != evaluation.as_dict()["grasp_execution_experience"]
    # Evidence collection is a valid next step, not an unintended stop signal.
    assert evaluation.stop is False


@pytest.mark.parametrize("source", ["policy_outcome", "current_before_rgb", "current_after_rgb",
    "prior_after_close_rgb", "raw_visual_evaluation", "execution_log", "current_after_lift_rgb"])
def test_no_xy_or_z_diagnosis_from_end_state_prior_or_wrong_phase(source):
    payload = diagnosis("XY_ALIGNED", "JAWS_ABOVE_CLOTH_AT_CLOSURE")
    for evidence in payload["evidence"]:
        evidence["evidence_ids"] = [source]
    result = apply_grasp_execution_experience(payload, context())
    assert result["status"] == "UNRESOLVED"
    assert result["update_allowed"] is False


@pytest.mark.parametrize("observations", [
    ("XY_ALIGNED", "XY_MISALIGNED", "STABLE_CLOTH_ACQUISITION"),
    ("XY_ALIGNED", "JAWS_ABOVE_CLOTH_AT_CLOSURE", "TABLE_CONTACT"),
    ("XY_ALIGNED", "JAWS_ABOVE_CLOTH_AT_CLOSURE", "STABLE_CLOTH_ACQUISITION"),
])
def test_conflicting_evidence_is_unresolved(observations):
    assert apply_grasp_execution_experience(diagnosis(*observations), context())["update_allowed"] is False


def test_point_coupling_or_wrong_transport_are_not_execution_mismatches():
    payload = unresolved_diagnosis()
    payload["evidence"][0].update(evidence_ids=["current_after_lift_rgb"],
        description="The whole garment moved as a bundle; no localized landing error.")
    assert apply_grasp_execution_experience(payload, context())["status"] == "UNRESOLVED"
    ctx = context()
    ctx["outcome"]["transport"] = {"status": "FAILURE"}
    result = apply_grasp_execution_experience(diagnosis("XY_ALIGNED", "STABLE_CLOTH_ACQUISITION"), ctx)
    assert result["status"] == "ALIGNED_SUCCESS"
    assert result["experience_update"]["z_direction"] == "KEEP"


@pytest.mark.parametrize("mutate", [
    lambda r: r["execution"].update(physical_execution=False),
    lambda r: r["execution"].pop("actual_robot_actions"),
    lambda r: r["execution"]["actual_robot_actions"][0].update(success=False),
    lambda r: r["execution"]["actual_robot_actions"][1].update(success=False),
    lambda r: r["execution"]["actual_robot_actions"][0]["args"].update(x=float("nan")),
    lambda r: r["execution"]["actual_robot_actions"][0]["args"].update(z=True),
    lambda r: r["execution"]["actual_robot_actions"].insert(1, {"name": "home", "success": True}),
    lambda r: r["execution"]["actual_robot_actions"].append({"name": "close_gripper", "success": True}),
    lambda r: r.pop("planning_diagnostics"),
    lambda r: r["planning_diagnostics"]["grasp_height_resolution"]["resolution"].update(valid=False),
    lambda r: r["planning_diagnostics"]["grasp_height_resolution"]["resolution"].update(surface_xyz_mm=[500, 40, float("inf")]),
    lambda r: r["planning_diagnostics"]["grasp_height_resolution"].update(runtime_authoritative_grasp_xyz_mm=[500, 40, None]),
])
def test_missing_geometry_never_falls_back_to_proposal_or_tcp(mutate):
    row = record()
    mutate(row)
    result = apply_grasp_execution_experience(diagnosis("XY_ALIGNED", "STABLE_CLOTH_ACQUISITION"), context(row))
    assert result["update_allowed"] is False
    assert result["experience_update"]["correction_xyz_mm"] == {"x": None, "y": None, "z": None}
    assert result["trial"]["observed_contact"] is None


def test_legacy_runtime_target_is_supported_but_depth_history_is_not_promoted():
    row = record()
    row["planning_diagnostics"]["grasp_height_resolution"].pop("runtime_authoritative_grasp_xyz_mm")
    assert runtime_execution_trial(row)["command_integrity"]["status"] == "MATCHED"
    row["grasp_experience"] = {"outcome": {"depth_interpretation": "EFFECTIVE"},
                               "experience_update": {"update_direction": "KEEP"}}
    assert initial_execution_experience(row)["status"] == "UNRESOLVED"


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(confidence=float("nan")),
    lambda d: d.update(confidence=float("inf")),
    lambda d: d.update(confidence=True),
    lambda d: d.update(confidence=1.1),
    lambda d: d.pop("confidence"),
    lambda d: d.update(evidence=[]),
    lambda d: d["evidence"][0].update(evidence_ids=[]),
    lambda d: d["evidence"][0].update(evidence_ids=["fabricated"]),
    lambda d: d["evidence"][0].update(description=" "),
    lambda d: d.update(observed_contact={"xyz_mm": [1, 2, 3]}),
    lambda d: d.update(correction_xyz_mm={"x": 1, "y": 2, "z": -3}),
])
def test_schema_rejects_fabricated_measurements_and_invalid_evidence(mutate):
    payload = diagnosis("XY_MISALIGNED")
    mutate(payload)
    with pytest.raises(ValueError):
        apply_grasp_execution_experience(payload, context())


def test_low_confidence_or_overflow_does_not_update():
    payload = diagnosis("XY_ALIGNED", "STABLE_CLOTH_ACQUISITION")
    payload["confidence"] = .69
    assert apply_grasp_execution_experience(payload, context())["update_allowed"] is False
    row = record()
    row["planning_diagnostics"]["grasp_height_resolution"]["resolution"]["surface_xyz_mm"][2] = -1e308
    row["planning_diagnostics"]["grasp_height_resolution"]["runtime_authoritative_grasp_xyz_mm"][2] = 1e308
    row["execution"]["actual_robot_actions"][0]["args"]["z"] = 1e308
    assert initial_execution_experience(row)["trial"]["execution"]["command_minus_surface_xyz_mm"] is None


def test_relative_xyz_transfers_with_surface_not_absolute_target():
    first = apply_grasp_execution_experience(diagnosis("XY_ALIGNED", "STABLE_CLOTH_ACQUISITION"), context())
    row = record()
    shift = [20, -10, 12]
    audit = row["planning_diagnostics"]["grasp_height_resolution"]
    audit["runtime_authoritative_grasp_xyz_mm"] = [x + d for x, d in zip(audit["runtime_authoritative_grasp_xyz_mm"], shift)]
    audit["resolution"]["surface_xyz_mm"] = [x + d for x, d in zip(audit["resolution"]["surface_xyz_mm"], shift)]
    for axis, delta in zip(("x", "y", "z"), shift):
        row["execution"]["actual_robot_actions"][0]["args"][axis] += delta
    second = apply_grasp_execution_experience(diagnosis("XY_ALIGNED", "STABLE_CLOTH_ACQUISITION"), context(row))
    assert first["experience_update"] == second["experience_update"]
    assert second["trial"]["execution"]["commanded_xyz_mm"][2] == 39


def test_persistence_dedup_and_no_update_for_unresolved_or_system_errors(tmp_path):
    ctx = context()
    ctx["grasp_execution_diagnosis"] = diagnosis("XY_ALIGNED", "STABLE_CLOTH_ACQUISITION")
    result = apply_grasp_execution_experience(ctx["grasp_execution_diagnosis"], ctx)
    assert persist_execution_trial(tmp_path, result, ctx, source_record="a.json")["status"] == "RECORDED"
    assert persist_execution_trial(tmp_path, result, ctx, source_record="a.json")["status"] == "ALREADY_APPLIED"
    path = tmp_path / "grasp_execution_trials.json"
    original = path.read_bytes()
    ctx.update(trial_id="trial_2", grasp_execution_diagnosis=unresolved_diagnosis())
    unresolved = apply_grasp_execution_experience(ctx["grasp_execution_diagnosis"], ctx)
    assert persist_execution_trial(tmp_path, unresolved, ctx, source_record="b.json")["status"] == "NO_EXECUTION_UPDATE"
    assert path.read_bytes() == original
    state = json.loads(original)
    saved = state["trials"]["trial_1"]["grasp_execution_experience"]
    assert saved["experience_update"]["correction_xyz_mm"] == {"x": 2, "y": -1.5, "z": -3}
    assert not (tmp_path / "grasp_depth_trials.json").exists()
    tampered = deepcopy(unresolved)
    tampered["update_allowed"] = True
    with pytest.raises(ValueError, match="host-validated"):
        persist_execution_trial(tmp_path, tampered, ctx, source_record="bad.json")


@pytest.mark.parametrize("variable", ["CONTACT_XY", "JAW_ALIGNMENT", "TRANSPORT", "RELEASE"])
def test_execution_gate_leaves_point_and_motion_domains_unchanged(variable):
    experiment = {"status": "PROPOSED", "single_change": {"variable": variable, "description": "test"}}
    original = deepcopy(experiment)
    gate_execution_experiment(experiment, initial_execution_experience(record()))
    assert experiment == original


@pytest.mark.parametrize("observations,variable,allowed", [
    (("INCONCLUSIVE",), "CONTACT_Z", False),
    (("INCONCLUSIVE",), "EXECUTION_XY", False),
    (("XY_MISALIGNED",), "EXECUTION_XY", True),
    (("XY_MISALIGNED",), "CONTACT_Z", False),
    (("XY_ALIGNED", "TABLE_CONTACT"), "CONTACT_Z", True),
    (("XY_ALIGNED", "TABLE_CONTACT"), "EXECUTION_XY", False),
    (("XY_MISALIGNED", "TABLE_CONTACT"), "EXECUTION_XY", True),
    (("XY_MISALIGNED", "TABLE_CONTACT"), "CONTACT_Z", False),
    (("XY_ALIGNED", "STABLE_CLOTH_ACQUISITION"), "CONTACT_Z", False),
])
def test_execution_experiment_requires_localization_and_one_axis_group(observations, variable, allowed):
    result = apply_grasp_execution_experience(diagnosis(*observations), context())
    experiment = {"status": "PROPOSED", "single_change": {"variable": variable, "description": "test"}}
    gate_execution_experiment(experiment, result)
    assert experiment["status"] == ("BLOCKED_BY_CAPABILITY" if allowed else "NO_EXPERIMENT")
    assert experiment["single_change"]["variable"] == (variable if allowed else "NONE")
    assert variable not in [c["variable"] for c in experiment["held_constant"]]


def test_schemas_are_closed_and_finite():
    def walk(schema):
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])
            for value in schema["properties"].values():
                walk(value)
        if schema.get("type") == "array":
            walk(schema["items"])
    walk(GRASP_EXECUTION_DIAGNOSIS_SCHEMA)
    walk(GRASP_EXECUTION_EXPERIENCE_SCHEMA)
    result = initial_execution_experience(record())
    result["experience_update"]["correction_xyz_mm"]["x"] = float("nan")
    with pytest.raises(ValueError):
        validate_schema(result, GRASP_EXECUTION_EXPERIENCE_SCHEMA)
