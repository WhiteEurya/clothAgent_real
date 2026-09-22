from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from cloth_agent.fold_experience_learning import (
    ConditionalExperienceStore, EXPERIENCE_UPDATE_SCHEMA, build_experience_request,
    capabilities, execution_contrast, validate_experience_update, validate_schema,
)
from cloth_agent.fold_exploration_pipeline import FoldExplorationPipeline, FoldExperienceStore, _compact_history
from cloth_agent.perception_comparison import apply_comparison_policy
from cloth_agent.skill_lifecycle import RunSkillLedger, SkillStore
from cloth_agent.grasp_execution_experience import unresolved_diagnosis


def analysis_payload(*, update=True):
    return {
        "grasp_execution_diagnosis": unresolved_diagnosis(),
        "failure_diagnosis": {"observed_outcome": "Garment returned to its initial configuration.",
            "physical_failure_mode": "UNKNOWN", "candidate_causes": [{
                "cause": "An ambiguous edge may not have coupled to the jaws.", "confidence": .3,
                "evidence_for": ["current_after_lift_rgb"], "evidence_against": []}],
            "uncertainties": ["Temporary contact does not establish why the final view was unchanged."]},
        "next_experiment": {"status": "PROPOSED", "primary_hypothesis": "Jaw alignment may affect retention.",
            "single_change": {"variable": "JAW_ALIGNMENT", "description": "Change jaw direction relative to the sleeve edge."},
            "held_constant": [{"variable": "CONTACT_XY", "description": "Keep a comparable visible contact region."},
                {"variable": "LIFT_HEIGHT", "description": "Keep the same validated lift height."}],
            "expected_observation": "Sleeve moves with the jaws and remains displaced after release.",
            "interpretation_if_success": "Support for alignment relevance increases, subject to cloth comparability.",
            "interpretation_if_failure": "Do not strengthen the alignment hypothesis; inspect other causes.",
            "comparability_limitations": ["Cloth may move between trials."]},
        "experience_update": {
            "operation": "CREATE", "rule_id": None,
            "context": ["Weak free-boundary evidence", "Ambiguous local overlap"],
            "action_property": "Pick, short lift and release",
            "supported_hypothesis": {"statement": "Weak boundary evidence may increase ineffective coupling risk.", "confidence": .3},
            "evidence_relation": "SUPPORT", "evidence_ids": ["current_after_lift_rgb"],
            "do_not_infer": ["Unchanged end state does not prove empty jaws.", "No universal rule about elevated points."],
            "counterexample_guard": {"must_not_generalize_to": ["Visible semantic endpoint with strong free-boundary support"],
                "reason": "This trial does not isolate height or endpoint semantics."},
            "contrast_with_prior": {"prior_trial_id": None, "comparable": False,
                "differences": [], "confounders": ["Only one trial is available."]},
            "policy_effect": {"candidate_ranking": "SLIGHT_PENALTY", "reason": "Tentative contextual association only."},
        } if update else None,
        "no_update_reason": "" if update else "No cause isolated; test alignment before learning a rule.",
    }


def request_context():
    return {"trial_id": "trial_1", "step": "left_sleeve", "prior_trial_id": None,
        "outcome": {"policy_failure_stage": "ACQUISITION", "grasp_acquisition": {"status": "FAILURE"},
            "perception_comparison": {"status": "UNCHANGED"}},
        "evidence_catalog": [{"id": "policy_outcome", "kind": "POLICY_LABEL"},
            {"id": "current_before_rgb", "kind": "END_STATE_OR_PRIOR_RGB"},
            {"id": "current_after_lift_rgb", "kind": "INTERACTION_RGB"}],
        "existing_rules": [], "capabilities": capabilities()}


def test_unchanged_policy_is_preserved_but_does_not_prove_a_physical_cause():
    payload = analysis_payload()
    payload["failure_diagnosis"]["physical_failure_mode"] = "EMPTY_GRASP"
    cause = payload["failure_diagnosis"]["candidate_causes"][0]
    cause.update(evidence_for=["policy_outcome"], confidence=.95)
    payload["experience_update"]["evidence_ids"] = ["policy_outcome", "current_before_rgb"]
    context = request_context()
    result = validate_experience_update(payload, context)
    assert context["outcome"]["policy_failure_stage"] == "ACQUISITION"
    assert result["failure_diagnosis"]["physical_failure_mode"] == "UNKNOWN"
    assert result["failure_diagnosis"]["candidate_causes"][0]["confidence"] == .25
    assert result["experience_update"] is None
    assert result["no_update_reason"]
    assert payload["failure_diagnosis"]["physical_failure_mode"] == "EMPTY_GRASP"


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(skill_update={}),
    lambda p: p["experience_update"].update(evidence_count={"support": 100}),
    lambda p: p["failure_diagnosis"]["candidate_causes"][0].update(evidence_for=["invented_video"]),
    lambda p: p["experience_update"]["policy_effect"].update(candidate_ranking="HARD_REJECT"),
    lambda p: p["experience_update"]["supported_hypothesis"].update(confidence=float("nan")),
    lambda p: p["experience_update"].update(rule_id="E_model_invented"),
    lambda p: p["experience_update"].update(operation="UPDATE", rule_id="E_missing"),
    lambda p: p["experience_update"]["counterexample_guard"].update(must_not_generalize_to=[]),
    lambda p: p["experience_update"]["contrast_with_prior"].update(comparable=True),
    lambda p: p["next_experiment"]["single_change"].update(variable="CONTACT_XY"),
])
def test_rejects_invalid_or_untraceable_learning(mutate):
    payload = analysis_payload()
    mutate(payload)
    with pytest.raises(ValueError):
        validate_experience_update(payload, request_context())


def test_unknown_depth_does_not_authorize_height_experiment():
    payload = analysis_payload(update=False)
    payload["next_experiment"]["single_change"] = {"variable": "CONTACT_Z", "description": "Lower contact with XY held fixed."}
    result = validate_experience_update(payload, request_context())
    assert result["next_experiment"]["status"] == "NO_EXPERIMENT"
    assert result["next_experiment"]["single_change"]["variable"] == "NONE"
    assert result["grasp_execution_experience"]["experience_update"]["kind"] == "NONE"
    assert result["experience_update"] is None


def test_no_new_knowledge_creates_no_rule_and_counts_no_support(tmp_path):
    store = ConditionalExperienceStore(tmp_path)
    context = request_context()
    result = validate_experience_update(analysis_payload(update=False), context)
    receipt = store.apply(result, context)
    assert receipt["status"] == "NO_NEW_KNOWLEDGE"
    assert store.rules("left_sleeve") == []
    assert store.apply(result, context)["status"] == "ALREADY_APPLIED"


def test_rule_reuse_counts_trials_once_and_counterevidence_reduces_confidence(tmp_path):
    store = ConditionalExperienceStore(tmp_path)
    context = request_context()
    payload = analysis_payload()
    first = store.apply(validate_experience_update(payload, context), context)
    store.apply(validate_experience_update(payload, context), context)
    rule = store.rules("left_sleeve")[0]
    assert rule["evidence_count"] == {"support": 1, "contradict": 0, "inconclusive": 0}
    assert 0 < rule["confidence"] < .3
    assert rule["task_outcome_count"]["failure"] == 1
    # CREATE of the same normalized rule also merges, instead of appending a lesson.
    context["trial_id"] = "trial_2"
    store.apply(validate_experience_update(payload, context), context)
    assert len(store.rules("left_sleeve")) == 1
    before = store.rules("left_sleeve")[0]["confidence"]
    context.update(trial_id="trial_3", existing_rules=store.rules("left_sleeve"), prior_trial_id="trial_2")
    context["outcome"]["grasp_acquisition"]["status"] = "SUCCESS"
    update = payload["experience_update"]
    update.update(operation="UPDATE", rule_id=first["rule_id"], evidence_relation="CONTRADICT")
    update["contrast_with_prior"].update(prior_trial_id="trial_2", comparable=True, confounders=[])
    receipt = store.apply(validate_experience_update(payload, context), context)
    assert receipt["evidence_count"] == {"support": 2, "contradict": 1, "inconclusive": 0}
    assert receipt["confidence"] < before
    rule = store.rules("left_sleeve")[0]
    assert rule["task_outcome_count"] == {"success": 1, "failure": 2, "unknown": 0}
    assert rule["counterexample_guard"]["must_not_generalize_to"]
    assert store.rules("right_sleeve") == []


def test_specialization_preserves_parent_and_requires_narrower_conditions(tmp_path):
    store = ConditionalExperienceStore(tmp_path)
    context = request_context()
    payload = analysis_payload()
    first = store.apply(validate_experience_update(payload, context), context)
    context.update(trial_id="trial_2", existing_rules=store.rules("left_sleeve"))
    payload["experience_update"].update(operation="SPECIALIZE", rule_id=first["rule_id"])
    with pytest.raises(ValueError, match="narrower"):
        validate_experience_update(payload, context)
    payload["experience_update"]["context"].append("Sleeve edge is visibly rolled")
    child = store.apply(validate_experience_update(payload, context), context)
    rules = store.rules("left_sleeve")
    assert len(rules) == 2
    assert next(r for r in rules if r["rule_id"] == child["rule_id"])["parent_rule_id"] == first["rule_id"]


def physical_record(root):
    before, after, lift = root / "before/camera_0_A.png", root / "after/camera_0_A.png", root / "camera_A_grasp_after_lift.png"
    for path in (before, after, lift):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 30), "pink").save(path)
    actions = [{"name": "move", "args": {"x": 500, "y": 40, "z": 30, "yaw": 0}, "success": True},
        {"name": "close_gripper", "args": {}, "success": True},
        {"name": "move", "args": {"x": 500, "y": 40, "z": 60, "yaw": 0}, "success": True},
        {"name": "open_gripper", "args": {}, "success": True}, {"name": "home", "args": {}, "success": True}]
    raw = {"grasp_acquisition": {"status": "SUCCESS", "confidence": .8, "evidence": ["Temporary lift appeared visible"]},
        "earliest_failure_stage": "NONE", "task_progress": {"status": "IMPROVED", "confidence": .8},
        "next_experiment": {"keep": ["contact"], "change": ["transport"], "reason": "Model suggestion before policy"},
        "perception_comparison": {"status": "UNCHANGED", "confidence": .95, "evidence": ["Same sleeve position after release"]},
        "skill_update": {"name": "unsupported-success-rule"}}
    return {"record_id": str(root / "iteration_001"), "iteration": 1, "planned_step": "left_sleeve", "mode": "FOLD",
        "proposal": {"actions": actions}, "execution_proposal": {"actions": actions},
        "execution": {"physical_execution": True, "execution_completed": True, "actual_robot_actions": actions},
        "before_images": [str(before)], "after_images": [str(after)],
        "recording": {"grasp_snapshots": {"after_lift": {"status": "CAPTURED", "image": str(lift)}}},
        "evaluation": apply_comparison_policy(raw), "evaluation_raw": {"evaluation": raw}}


def test_request_keeps_normalized_outcome_raw_observation_and_images_separate(tmp_path):
    record = physical_record(tmp_path)
    context, images = build_experience_request(record, [], [], tmp_path, tmp_path / "analysis")
    assert context["outcome"]["grasp_acquisition"]["status"] == "FAILURE"
    assert context["outcome"]["physical_failure_mode"] == "UNKNOWN"
    raw = next(e for e in context["evidence_catalog"] if e["id"] == "raw_visual_evaluation")
    assert raw["value"]["grasp_acquisition"]["status"] == "SUCCESS"
    assert len(images) == 3
    assert str(tmp_path) not in json.dumps(context)
    assert "skill_update" not in json.dumps(context)
    catalog = {e["id"]: e for e in context["evidence_catalog"]}
    assert catalog["current_after_lift_rgb"]["kind"] == "INTERACTION_RGB"


def test_command_contrast_checks_what_changed_and_marks_confounded_claims(tmp_path):
    prior = physical_record(tmp_path)
    prior["next_experiment"] = {"single_change": {"variable": "JAW_ALIGNMENT"}}
    current = deepcopy(prior)
    for action in current["execution"]["actual_robot_actions"]:
        if action["name"] == "move":
            action["args"]["yaw"] += 20
    contrast = execution_contrast(current, prior)
    assert contrast["changed_variables"] == ["JAW_ALIGNMENT"]
    assert contrast["prior_experiment_implemented"] is True
    # Host-resolved contact changed too; it is no longer a single-variable trial.
    current["execution"]["actual_robot_actions"][0]["args"]["z"] -= 2
    contrast = execution_contrast(current, prior)
    assert "CONTACT_Z" in contrast["changed_variables"]
    assert contrast["single_variable_command_comparison"] is False
    context = request_context()
    context.update(prior_trial_id="trial_prior", execution_contrast=contrast)
    payload = analysis_payload()
    payload["experience_update"]["contrast_with_prior"].update(
        prior_trial_id="trial_prior", comparable=True, confounders=[])
    analysis = validate_experience_update(payload, context)
    assert analysis["experience_update"]["contrast_with_prior"]["comparable"] is False
    assert "Host command logs" in analysis["experience_update"]["contrast_with_prior"]["confounders"][0]


@pytest.mark.parametrize("failed", [False, True])
def test_pipeline_generates_after_policy_and_never_fabricates_fallback_lessons(tmp_path, failed):
    pipe = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    pipe.session = SimpleNamespace(run_dir=tmp_path)
    pipe.experiences = FoldExperienceStore(tmp_path)
    pipe.skill_ledger = RunSkillLedger(tmp_path / "workspace")
    pipe.skill_store = SkillStore(tmp_path / "skills")
    pipe.conditional_experiences = ConditionalExperienceStore(tmp_path / "rules")
    pipe._debug = pipe._debug_exception = lambda *a, **kw: None
    calls = []
    def update(**kwargs):
        calls.append(kwargs)
        assert kwargs["context"]["outcome"]["grasp_acquisition"]["status"] == "FAILURE"
        if failed:
            raise RuntimeError("analysis service unavailable")
        return analysis_payload()
    pipe.client = SimpleNamespace(update_experience=update)
    record = physical_record(tmp_path)
    directory = tmp_path / "iteration_001"
    pipe._save_iteration_learning(directory, record)
    pipe._save_iteration_learning(directory, record)
    assert len(calls) == 1
    saved = pipe.experiences.history(limit=None)[0]
    assert saved["evaluation"]["grasp_acquisition"]["status"] == "FAILURE"
    assert saved["evaluation_raw"]["evaluation"]["grasp_acquisition"]["status"] == "SUCCESS"
    assert saved["grasp_execution_experience"]["status"] == "UNRESOLVED"
    assert saved["grasp_execution_experience"]["experience_update"]["kind"] == "NONE"
    assert not (pipe.conditional_experiences.root / "grasp_depth_trials.json").exists()
    assert not (pipe.conditional_experiences.root / "grasp_execution_trials.json").exists()
    assert not pipe.skill_ledger.candidates_path.exists()
    if failed:
        assert saved["experience_generation"]["status"] == "FAILED"
        assert "failure_diagnosis" not in saved
        assert pipe.conditional_experiences.rules("left_sleeve") == []
    else:
        assert saved["experience_generation"]["status"] == "COMPLETED"
        assert saved["next_experiment"]["single_change"]["variable"] == "JAW_ALIGNMENT"
        assert saved["outcome"]["physical_failure_mode"] == "UNKNOWN"
        assert (directory / "experience_update/analysis.json").exists()
        assert "next_experiment" not in _compact_history([saved])[0]["evaluation"]
    summary = json.loads(pipe.experiences.summary_path.read_text())
    assert summary["experience_count"] == 1
    assert summary["conditional_rule_update_count"] == (0 if failed else 1)


@pytest.mark.parametrize("execution_store_failed,command_mismatch", [(False, False), (True, False), (False, True)])
def test_pipeline_saves_execution_trial_separately_without_changing_point_or_motion_rules(tmp_path, monkeypatch, execution_store_failed, command_mismatch):
    pipe = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    pipe.session = SimpleNamespace(run_dir=tmp_path)
    pipe.experiences = FoldExperienceStore(tmp_path)
    pipe.conditional_experiences = ConditionalExperienceStore(tmp_path / "rules")
    pipe._debug = pipe._debug_exception = lambda *a, **kw: None
    payload = analysis_payload(update=False)
    payload["grasp_execution_diagnosis"] = {"confidence": .9,
        "evidence": [{"observation": "STABLE_CLOTH_ACQUISITION", "evidence_ids": ["current_after_lift_rgb"],
                      "description": "Cloth visibly retained through the lift, independently of release outcome."},
                     {"observation": "XY_ALIGNED", "evidence_ids": ["current_after_close_rgb"],
                      "description": "Contact lies on the same selected endpoint identifiable in the close frame."}]}
    def update(**kwargs):
        assert kwargs["context"]["grasp_execution_trial"]["execution"]["descent_below_surface_mm"] == 2.5
        return payload
    pipe.client = SimpleNamespace(update_experience=update)
    if execution_store_failed:
        def fail(*args, **kwargs):
            raise OSError("execution store unavailable")
        monkeypatch.setattr("cloth_agent.fold_exploration_pipeline.persist_execution_trial", fail)
    row = physical_record(tmp_path)
    close_image = tmp_path / "camera_A_grasp_after_close.png"
    Image.new("RGB", (40, 30), "pink").save(close_image)
    row["recording"]["grasp_snapshots"]["after_close"] = {"status": "CAPTURED", "image": str(close_image)}
    row["execution"] = deepcopy(row["execution"])
    row["execution"]["actual_robot_actions"][0]["args"]["z"] = 27.5
    row["planning_diagnostics"] = {"grasp_height_resolution": {
        "resolved_grasp_z_mm": 27 if command_mismatch else 27.5, "resolved_grasp_xy_mm": [500, 40],
        "resolution": {"valid": True, "surface_xyz_mm": [500, 40, 30]}}}
    pipe._update_fold_experience(tmp_path / "iteration_001", row)
    assert row["experience_generation"]["status"] == "COMPLETED"
    experience = row["grasp_execution_experience"]
    assert experience["status"] == ("UNRESOLVED" if command_mismatch else "ALIGNED_SUCCESS")
    assert experience["observed_result"]["policy_acquisition"] == "FAILURE"
    assert experience["observed_result"]["acquisition"] == "SUCCESS"
    assert experience["experience_update"]["kind"] == ("NONE" if command_mismatch else "REINFORCE")
    assert experience["experience_update"]["correction_xyz_mm"] == (
        {"x": None, "y": None, "z": None} if command_mismatch else {"x": 0, "y": 0, "z": -2.5})
    assert row["next_experiment"] == payload["next_experiment"]  # Alignment domain unchanged.
    assert row["experience_generation"]["store_receipt"]["status"] == "NO_NEW_KNOWLEDGE"
    assert pipe.conditional_experiences.rules("left_sleeve") == []
    assert not (tmp_path / "rules/grasp_depth_trials.json").exists()
    if command_mismatch:
        assert experience["trial"]["command_integrity"]["status"] == "SYSTEM_CODE_FAILURE"
        assert row["grasp_execution_store_receipt"]["status"] == "NO_EXECUTION_UPDATE"
        assert not (tmp_path / "rules/grasp_execution_trials.json").exists()
        return
    if execution_store_failed:
        assert row["grasp_execution_store_receipt"]["status"] == "FAILED"
        assert not (tmp_path / "rules/grasp_execution_trials.json").exists()
        return
    assert row["grasp_execution_store_receipt"]["status"] == "RECORDED"
    trials = json.loads((tmp_path / "rules/grasp_execution_trials.json").read_text())["trials"]
    assert len(trials) == 1
    assert next(iter(trials.values()))["grasp_execution_experience"]["trial"]["execution"]["descent_below_surface_mm"] == 2.5


def test_local_backend_uses_isolated_read_only_analysis_call(tmp_path, monkeypatch):
    from cloth_agent.auto_exploration import ClaudeAutoClient
    from cloth_agent import auto_exploration
    image = tmp_path / "rgb.png"
    Image.new("RGB", (20, 20), "white").save(image)
    client = ClaudeAutoClient(binary="claude")
    monkeypatch.setattr(client, "_binary", lambda: "claude")
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"structured_output": analysis_payload(update=False)}), stderr="")
    monkeypatch.setattr(auto_exploration.subprocess, "run", run)
    payload = client.update_experience(context=request_context(), image_paths=[image], run_dir=tmp_path, output_dir=tmp_path / "analysis")
    validate_schema(payload, EXPERIENCE_UPDATE_SCHEMA)
    command, kwargs = calls[0]
    assert command[command.index("--tools") + 1] == "Read"
    assert "--no-session-persistence" in command
    assert "policy_failure_stage" in kwargs["input"]
    assert "CONTACT_Z" in kwargs["input"]
