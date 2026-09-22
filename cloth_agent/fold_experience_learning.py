"""Post-outcome causal hypotheses and conditional, evidence-weighted memory.

Outcome labels are host policy. Diagnosis is uncertain model interpretation.
Experiment proposals are not learned rules. This module never emits robot code.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import re
from pathlib import Path

from .trajectory_memory import prepare_trajectory_memory
from .grasp_execution_experience import (
    GRASP_EXECUTION_DIAGNOSIS_SCHEMA, GRASP_EXECUTION_INSTRUCTION,
    gate_execution_experiment, apply_grasp_execution_experience, runtime_execution_trial,
)


def _object(properties):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


TEXT = {"type": "string", "minLength": 1, "maxLength": 800}
TEXTS = {"type": "array", "maxItems": 8, "uniqueItems": True, "items": TEXT}
CONFIDENCE = {"type": "number", "minimum": 0, "maximum": 1}
NULLABLE_ID = {"type": ["string", "null"], "minLength": 1, "maxLength": 80}
VARIABLES = ["CONTACT_XY", "JAW_ALIGNMENT", "ENTRY_PATH", "INITIAL_LIFT_PATH",
             "LIFT_HEIGHT", "TRANSPORT", "RELEASE", "CONTACT_Z", "EXECUTION_XY", "OBSERVATION", "NONE"]
VARIABLE = {"enum": VARIABLES}
FAILURE_MODES = ["UNKNOWN", "EMPTY_GRASP", "SLIP", "RETURN_AFTER_RELEASE",
                 "INSUFFICIENT_TRANSPORT", "OTHER", "NONE"]

DIAGNOSIS_SCHEMA = _object({
    "observed_outcome": TEXT,
    "physical_failure_mode": {"enum": FAILURE_MODES},
    "candidate_causes": {"type": "array", "maxItems": 4, "items": _object({
        "cause": TEXT, "confidence": CONFIDENCE, "evidence_for": TEXTS, "evidence_against": TEXTS})},
    "uncertainties": TEXTS,
})

NEXT_EXPERIMENT_SCHEMA = _object({
    "status": {"enum": ["PROPOSED", "BLOCKED_BY_CAPABILITY", "NO_EXPERIMENT"]},
    "primary_hypothesis": TEXT,
    "single_change": _object({"variable": VARIABLE, "description": TEXT}),
    "held_constant": {"type": "array", "maxItems": 8, "items": _object({
        "variable": VARIABLE, "description": TEXT})},
    "expected_observation": TEXT,
    "interpretation_if_success": TEXT,
    "interpretation_if_failure": TEXT,
    "comparability_limitations": TEXTS,
})

EXPERIENCE_SCHEMA = _object({
    "operation": {"enum": ["CREATE", "UPDATE", "SPECIALIZE"]},
    "rule_id": NULLABLE_ID,
    "context": TEXTS,
    "action_property": TEXT,
    "supported_hypothesis": _object({"statement": TEXT, "confidence": CONFIDENCE}),
    "evidence_relation": {"enum": ["SUPPORT", "CONTRADICT", "INCONCLUSIVE"]},
    "evidence_ids": TEXTS,
    "do_not_infer": TEXTS,
    "counterexample_guard": _object({"must_not_generalize_to": TEXTS, "reason": TEXT}),
    "contrast_with_prior": _object({"prior_trial_id": NULLABLE_ID, "comparable": {"type": "boolean"},
        "differences": TEXTS, "confounders": TEXTS}),
    "policy_effect": _object({"candidate_ranking": {"enum": ["NONE", "SLIGHT_PENALTY", "SLIGHT_PREFERENCE"]},
        "reason": TEXT}),
})

EXPERIENCE_UPDATE_SCHEMA = _object({
    "grasp_execution_diagnosis": GRASP_EXECUTION_DIAGNOSIS_SCHEMA,
    "failure_diagnosis": DIAGNOSIS_SCHEMA,
    "next_experiment": NEXT_EXPERIMENT_SCHEMA,
    "experience_update": {"anyOf": [{"type": "null"}, EXPERIENCE_SCHEMA]},
    "no_update_reason": {"type": "string", "maxLength": 800},
})

EXPERIENCE_INSTRUCTION = (
    "Analyze this completed attempt AFTER host outcome normalization. Keep three layers separate: "
    "outcome (supplied, immutable), failure_diagnosis (uncertain physical hypotheses), and "
    "experience_update (conditional knowledge supported by evidence already observed). "
    "UNCHANGED can impose policy_failure_stage=ACQUISITION while physical_failure_mode remains "
    "UNKNOWN. Never infer empty jaws, wrong XY, or wrong Z from unchanged end state alone. "
    "Inspect the supplied RGB and execution log; completed closure is not cloth acquisition. "
    "All evidence_for, evidence_against and evidence_ids entries must be exact evidence catalog IDs. "
    "A hypothesis with no discriminating evidence is uncertain; do not present it as a cause. "
    "For next_experiment select ONE changed variable, list held_constant variables, predict a "
    "discriminating observation and explain how either result would change the hypothesis. "
    "Cloth state, occlusion and uncontrolled trajectory changes can confound comparisons; name them. "
    "An untested correction is an experiment, not a persistent lesson. CONTACT_Z is currently "
    "host-resolved: you may diagnose it, but mark that experiment BLOCKED_BY_CAPABILITY. Other "
    "host restrictions are in capabilities. Do not recommend a different variable merely to "
    "pretend the original hypothesis was tested. No experiment directly authorizes motion. "
    "Set experience_update=null and explain no_update_reason when there is no new supported "
    "knowledge. Do not manufacture a lesson on every failure. A tentative single-trial risk "
    "association must be contextual, low-confidence, and have at most a slight ranking effect. "
    "Include do_not_infer and counterexample_guard; never hard-reject candidates or generalize "
    "from one image location/height to all raised points or semantic endpoints. "
    "Reuse an existing rule_id via UPDATE for the same condition/hypothesis, including contrary "
    "evidence. SPECIALIZE creates a narrower child rule and must explain extra conditions. "
    "Preserve rule identity on UPDATE. Counts and accumulated confidence are computed by the host, "
    "never invent historical trials. Rules are soft conditional advice, subordinate to current "
    "visual evidence, exceptions, workspace, IK and execution validation. "
    + GRASP_EXECUTION_INSTRUCTION
)

EXPERIENCE_PLANNING_INSTRUCTION = (
    "Use conditional_experience only where its context matches current RGB; first check each "
    "counterexample_guard and do_not_infer. Unsupported or contradicted rules cannot hard-reject "
    "a candidate. next_experiment is an untested proposal, not established knowledge or a command. "
    "When feasible, implement its one changed variable, keep the named controls comparable, and "
    "explain any necessary deviations in motion_intent. Reassess if cloth state has changed. "
    "The structured next_experiment supersedes generic evaluator keep/change suggestions. "
    "BLOCKED_BY_CAPABILITY experiments are not executable; do not silently substitute a different "
    "variable and claim to have tested the hypothesis. Select all targets from current evidence. "
    "grasp_execution_experience concerns landing at the selected point, never point ranking. "
    "UNRESOLVED authorizes no XYZ update. SYSTEM_CODE_FAILURE requires code investigation. "
    "Never replay historical absolute grasp XYZ or substitute CONTACT_XY point reselection "
    "for a blocked EXECUTION_XY correction."
)


def validate_schema(value, schema, path="experience_update"):
    """Validate the deliberately small schema vocabulary above, without dependencies."""
    if "anyOf" in schema:
        for variant in schema["anyOf"]:
            try:
                validate_schema(value, variant, path)
                return
            except ValueError:
                pass
        raise ValueError(f"{path}: no permitted schema variant")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: unsupported value")
    types = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "null": value is None,
             "boolean": type(value) is bool,
             "number": type(value) in (int, float) and math.isfinite(value)}
    expected = schema.get("type")
    if expected and not any(types[t] for t in (expected if isinstance(expected, list) else [expected])):
        raise ValueError(f"{path}: invalid type")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if set(schema.get("required", [])) - value.keys() or (
                schema.get("additionalProperties") is False and value.keys() - properties.keys()):
            raise ValueError(f"{path}: missing or unexpected fields")
        for key, item in value.items():
            validate_schema(item, properties[key], f"{path}.{key}")
    if isinstance(value, list):
        if len(value) > schema.get("maxItems", len(value)) or len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path}: invalid array length")
        if schema.get("uniqueItems") and len({json.dumps(v, sort_keys=True) for v in value}) != len(value):
            raise ValueError(f"{path}: duplicate items")
        for i, item in enumerate(value):
            validate_schema(item, schema["items"], f"{path}[{i}]")
    if isinstance(value, str) and (len(value.strip()) < schema.get("minLength", 0) or
                                  len(value) > schema.get("maxLength", len(value))):
        raise ValueError(f"{path}: invalid string length")
    if type(value) in (int, float) and not schema.get("minimum", value) <= value <= schema.get("maximum", value):
        raise ValueError(f"{path}: number outside bounds")


def trial_id(record):
    return "trial_" + hashlib.sha256(str(record["record_id"]).encode()).hexdigest()[:20]


def capabilities(learning=None):
    allowed = [v for v in VARIABLES if v not in {"CONTACT_Z", "EXECUTION_XY", "NONE"}]
    if (learning or {}).get("require_non_height_change"):
        allowed = ["CONTACT_XY", "JAW_ALIGNMENT", "ENTRY_PATH", "INITIAL_LIFT_PATH", "OBSERVATION"]
    return {"executable_single_changes": allowed, "contact_z": "HOST_RESOLVED_NOT_MODEL_ADJUSTABLE",
            "execution_xy": "HOST_RESOLVED_NOT_MODEL_ADJUSTABLE",
            "all_motion_requires_current_grounding_and_host_validation": True}


def execution_contrast(current, prior):
    """Check command geometry independently of a model's comparability claim.

    This never certifies identical cloth state, speed, force or actual contact.
    It only detects confounds visible in the actual action-command log.
    """
    def signature(record):
        if not record or (record.get("execution") or {}).get("execution_completed") is not True:
            return None
        actions = (record.get("execution") or {}).get("actual_robot_actions") or []
        if not actions or any(a.get("success") is not True for a in actions):
            return None
        close = next((i for i, a in enumerate(actions) if a.get("name") == "close_gripper"), None)
        if close is None:
            return None
        before = [a.get("args") or {} for a in actions[:close] if a.get("name") == "move"]
        release = next((i for i in range(close + 1, len(actions)) if actions[i].get("name") == "open_gripper"), len(actions))
        after = [a.get("args") or {} for a in actions[close + 1:release] if a.get("name") == "move"]
        if not before or not after:
            return None
        values = [*before, *after]
        if any(type(a.get(k)) not in (int, float) or not math.isfinite(a[k])
               for a in values for k in ("x", "y", "z", "yaw")):
            return None
        contact = before[-1]
        def delta(a, keys):
            return [a[k] - contact[k] for k in keys]
        return {"contact": contact, "entry": delta(before[-2], ("x", "y")) if len(before) > 1 else None,
            "lift_xy": delta(after[0], ("x", "y")), "lift_height": after[0]["z"] - contact["z"],
            "transport": delta(after[-1], ("x", "y")),
            "sequence": [a.get("name") for a in actions]}
    a, b = signature(current), signature(prior)
    if a is None or b is None:
        return {"status": "UNAVAILABLE", "single_variable_command_comparison": False,
                "reason": "Requires two completed action logs with explicit successful move records."}
    changes = []
    if math.dist([a["contact"][k] for k in ("x", "y")], [b["contact"][k] for k in ("x", "y")]) >= 4:
        changes.append("CONTACT_XY")
    if abs(a["contact"]["z"] - b["contact"]["z"]) >= .25:
        changes.append("CONTACT_Z")
    if abs((a["contact"]["yaw"] - b["contact"]["yaw"] + 180) % 360 - 180) >= 15:
        changes.append("JAW_ALIGNMENT")
    if (a["entry"] is None) != (b["entry"] is None) or (
            a["entry"] is not None and b["entry"] is not None and math.dist(a["entry"], b["entry"]) >= 4):
        changes.append("ENTRY_PATH")
    if math.dist(a["lift_xy"], b["lift_xy"]) >= 4:
        changes.append("INITIAL_LIFT_PATH")
    if abs(a["lift_height"] - b["lift_height"]) >= .5:
        changes.append("LIFT_HEIGHT")
    if math.dist(a["transport"], b["transport"]) >= 4:
        changes.append("TRANSPORT")
    sequence_changed = a["sequence"] != b["sequence"]
    planned = ((prior.get("next_experiment") or {}).get("single_change") or {}).get("variable")
    return {"status": "COMPARED_COMMANDS", "changed_variables": changes,
        "action_sequence_changed": sequence_changed,
        "single_variable_command_comparison": len(changes) == 1 and not sequence_changed,
        "prior_experiment_variable": planned,
        "prior_experiment_implemented": bool(planned and changes == [planned] and not sequence_changed),
        "limitations": "Command-log comparison only. Cloth state, speed, force, sub-threshold changes and actual contact remain potential confounds.",
        "thresholds": {"xy_mm": 4, "contact_z_mm": .25, "yaw_deg": 15, "lift_height_mm": .5}}


def build_experience_request(record, history, rules, run_dir, output):
    """Supply normalized labels, raw observations, current/prior action evidence."""
    from .remote_fold import semantic_history
    step = record.get("planned_step")
    current, images = prepare_trajectory_memory([record], step, run_dir, output / "current")
    attempt = (current or {}).get("previous_physical_attempt")
    if attempt is None:
        raise ValueError("experience requires a physical action log")
    previous, previous_images = prepare_trajectory_memory(history, step, run_dir, output / "prior")
    prior = (previous or {}).get("previous_physical_attempt")
    previous_record = next((r for r in reversed(history) if not r.get("inherited_lesson")
        and r.get("planned_step") == step and r.get("iteration") == (prior or {}).get("iteration")
        and r.get("record_id")), None) if prior else None
    evaluation = record.get("evaluation") or {}
    raw = (record.get("evaluation_raw") or {}).get("evaluation") or {}
    comparison = evaluation.get("perception_comparison") or {}
    outcome = {"policy_failure_stage": evaluation.get("earliest_failure_stage", "UNKNOWN"),
        "grasp_acquisition": semantic_history(evaluation.get("grasp_acquisition")),
        "task_progress": semantic_history(evaluation.get("task_progress")),
        "perception_comparison": comparison,
        "physical_failure_mode": "UNKNOWN",
        "label_basis": "END_STATE_POLICY" if comparison.get("status") == "UNCHANGED" else "VISUAL_EVALUATION"}
    contrast = execution_contrast(record, previous_record)
    evidence = [
        {"id": "policy_outcome", "kind": "POLICY_LABEL", "value": outcome},
        {"id": "execution_log", "kind": "ROBOT_ACTIONS_NOT_CLOTH_ACQUISITION", "value": attempt["execution_log"]},
        {"id": "raw_visual_evaluation", "kind": "MODEL_INTERPRETATION_NOT_CAUSAL_PROOF",
         "value": semantic_history(raw)},
        {"id": "execution_contrast", "kind": "ROBOT_ACTIONS_NOT_CLOTH_ACQUISITION", "value": contrast},
    ]
    all_images, image_catalog = [], []
    for prefix, memory, paths in (("current", attempt, images), ("prior", prior, previous_images)):
        if memory is None:
            continue
        by_name = {p.name: p for p in paths}
        for entry in memory["images"]:
            if entry["status"] != "AVAILABLE":
                continue
            index = len(all_images)
            all_images.append(by_name[entry["name"]])
            evidence_id = f"{prefix}_{entry['role']}_rgb"
            entry["image_id"] = f"image_{index}"
            if entry["role"] == "before" and memory.get("grasp_in_before_image"):
                memory["grasp_in_before_image"]["image_id"] = entry["image_id"]
            image_catalog.append({"image_index": index, "role": evidence_id,
                "capture_note": entry.get("capture_note", "At the perception pose; not a gripper close-up.")})
            evidence.append({"id": evidence_id, "kind": "INTERACTION_RGB" if prefix == "current" and
                entry["role"] in {"before_lift", "after_close", "after_lift", "rollout"} else "END_STATE_OR_PRIOR_RGB",
                "image_index": index})
    if not images:
        raise ValueError("experience requires available same-trial RGB evidence")
    context = {"trial_id": trial_id(record), "step": step, "outcome": outcome,
        "grasp_execution_trial": runtime_execution_trial(record),
        "current_attempt": attempt, "prior_attempt": prior,
        "prior_trial_id": trial_id(previous_record) if previous_record else None,
        "prior_next_experiment": (previous_record or {}).get("next_experiment"),
        "execution_contrast": contrast,
        "evidence_catalog": evidence, "images": image_catalog,
        "existing_rules": rules, "capabilities": capabilities(record.get("acquisition_learning")),
        "instruction": EXPERIENCE_INSTRUCTION}
    return context, all_images


def validate_experience_update(payload, context):
    validate_schema(payload, EXPERIENCE_UPDATE_SCHEMA)
    result = copy.deepcopy(payload)
    result["grasp_execution_experience"] = apply_grasp_execution_experience(result["grasp_execution_diagnosis"], context)
    catalog = {e["id"]: e for e in context["evidence_catalog"]}
    diagnosis = result["failure_diagnosis"]
    for cause in diagnosis["candidate_causes"]:
        if set(cause["evidence_for"] + cause["evidence_against"]) - catalog.keys():
            raise ValueError("diagnosis cites unknown evidence IDs")
        if not cause["evidence_for"] and cause["confidence"] > .25:
            raise ValueError("unsupported causal hypothesis has excessive confidence")
    notes = []
    # Neither task failure nor end-state images establish the physical mechanism.
    has_interaction = any(catalog[e]["kind"] == "INTERACTION_RGB"
        for c in diagnosis["candidate_causes"] for e in c["evidence_for"])
    if diagnosis["physical_failure_mode"] != "UNKNOWN" and not has_interaction:
        diagnosis["physical_failure_mode"] = "UNKNOWN"
        notes.append("Physical mechanism kept UNKNOWN: no cited same-trial interaction RGB.")
    for cause in diagnosis["candidate_causes"]:
        if not any(catalog[e]["kind"] == "INTERACTION_RGB" for e in cause["evidence_for"]):
            cause["confidence"] = min(.25, cause["confidence"])
    experiment = result["next_experiment"]
    variable = experiment["single_change"]["variable"]
    controls = [c["variable"] for c in experiment["held_constant"]]
    if variable in controls or len(controls) != len(set(controls)) or "NONE" in controls:
        raise ValueError("experiment changed and held-constant variables overlap or repeat")
    if experiment["status"] == "NO_EXPERIMENT" and variable != "NONE":
        raise ValueError("NO_EXPERIMENT must use variable NONE")
    if experiment["status"] != "NO_EXPERIMENT" and variable == "NONE":
        raise ValueError("an experiment must name one variable")
    if experiment["status"] == "PROPOSED" and variable not in context["capabilities"]["executable_single_changes"]:
        experiment["status"] = "BLOCKED_BY_CAPABILITY"
        notes.append(f"{variable} is not supported by the current execution contract.")
    gate_execution_experiment(experiment, result["grasp_execution_experience"])
    update = result["experience_update"]
    if update is None:
        if not result["no_update_reason"].strip():
            raise ValueError("no experience update requires an explicit reason")
    else:
        if (not update["context"] or not update["evidence_ids"] or not update["do_not_infer"]
                or not update["counterexample_guard"]["must_not_generalize_to"]):
            raise ValueError("conditional experience requires context, evidence and inference guards")
        if set(update["evidence_ids"]) - catalog.keys():
            raise ValueError("experience cites unknown evidence IDs")
        prior = update["contrast_with_prior"]
        if prior["prior_trial_id"] != context["prior_trial_id"]:
            raise ValueError("experience cites an unavailable prior trial")
        if prior["comparable"] and (not context["prior_trial_id"] or prior["confounders"]):
            raise ValueError("comparable contrast requires a prior trial and no stated confounders")
        if not prior["comparable"] and not prior["confounders"]:
            raise ValueError("non-comparable evidence must state its limitations")
        if prior["comparable"] and not context.get("execution_contrast", {}).get("single_variable_command_comparison"):
            prior["comparable"] = False
            prior["confounders"].append("Host command logs do not establish a single-variable contrast.")
            notes.append("Comparison weight reduced: multiple/no isolated changes or incomplete command evidence.")
        known = {r["rule_id"]: r for r in context["existing_rules"]}
        if update["operation"] == "CREATE" and update["rule_id"] is not None:
            raise ValueError("CREATE cannot assign its own rule ID")
        if update["operation"] != "CREATE" and update["rule_id"] not in known:
            raise ValueError("experience update refers to an unknown rule")
        if update["operation"] == "UPDATE" and rule_key(update, context["step"]) != rule_key(known[update["rule_id"]], context["step"]):
            raise ValueError("UPDATE cannot change rule conditions or hypothesis; use SPECIALIZE")
        if update["operation"] == "SPECIALIZE":
            parent = known[update["rule_id"]]
            if not set(update["context"]) > set(parent["context"]):
                raise ValueError("SPECIALIZE must retain parent conditions and add a narrower condition")
        # End-state failure alone adds an observation, not causal support.
        kinds = {catalog[e]["kind"] for e in update["evidence_ids"]}
        if not (kinds & {"INTERACTION_RGB", "ROBOT_ACTIONS_NOT_CLOTH_ACQUISITION"}) or (
                not prior["comparable"] and "INTERACTION_RGB" not in kinds):
            update["evidence_relation"] = "INCONCLUSIVE"
            update["policy_effect"]["candidate_ranking"] = "NONE"
            notes.append("End-state-only or confounded evidence does not strengthen a causal rule.")
        if update["evidence_relation"] == "INCONCLUSIVE" and update["operation"] in {"CREATE", "SPECIALIZE"}:
            result["experience_update"] = None
            result["no_update_reason"] = "No discriminating evidence supports creating this conditional rule. Retain the hypothesis as an experiment only."
            notes.append("Did not create a persistent rule from inconclusive evidence.")
    result["host_validation_notes"] = notes
    return result


def _canonical(text):
    return " ".join(re.findall(r"\w+", text.lower()))


def rule_key(rule, step):
    identity = [step, sorted(_canonical(c) for c in rule["context"]),
                _canonical(rule["action_property"]), _canonical(rule["supported_hypothesis"]["statement"])]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:20]


class ConditionalExperienceStore:
    """Atomically update rules; a trial contributes at most one vote per rule."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / "rules.json"

    def _read(self):
        if not self.path.exists():
            return {"schema_version": 1, "rules": {}, "processed_trials": {}}
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("schema_version") != 1:
            raise ValueError("unsupported conditional experience schema")
        return state

    def rules(self, step, limit=8):
        rows = [copy.deepcopy(r) for r in self._read()["rules"].values() if r["step"] == step]
        rows.sort(key=lambda r: r["last_update"], reverse=True)
        for row in rows:
            row.pop("evidence_trials", None)
        return rows[:limit]

    def apply(self, analysis, context, *, source_record=None):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "rules.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self._read()
            tid = context["trial_id"]
            if tid in state["processed_trials"]:
                return {**state["processed_trials"][tid], "status": "ALREADY_APPLIED"}
            update = analysis["experience_update"]
            receipt = {"rule_id": None, "status": "NO_NEW_KNOWLEDGE"}
            if update is not None:
                key = "E_" + rule_key(update, context["step"])
                if update["operation"] == "UPDATE":
                    key = update["rule_id"]
                row = state["rules"].get(key)
                if row is None:
                    row = {"rule_id": key, "step": context["step"],
                        "context": update["context"], "action_property": update["action_property"],
                        "supported_hypothesis": {"statement": update["supported_hypothesis"]["statement"]},
                        "parent_rule_id": update["rule_id"] if update["operation"] == "SPECIALIZE" else None,
                        "do_not_infer": [], "counterexample_guard": {"must_not_generalize_to": []},
                        "evidence_trials": {}}
                for field in ("do_not_infer",):
                    row[field] = list(dict.fromkeys(row[field] + update[field]))
                guard = row["counterexample_guard"]
                guard["must_not_generalize_to"] = list(dict.fromkeys(
                    guard["must_not_generalize_to"] + update["counterexample_guard"]["must_not_generalize_to"]))
                guard["reason"] = update["counterexample_guard"]["reason"]
                comparable = update["contrast_with_prior"]["comparable"]
                vote = {"relation": update["evidence_relation"], "evidence_ids": update["evidence_ids"],
                    "weight": update["supported_hypothesis"]["confidence"] * (1 if comparable else .25),
                    "source_record": source_record,
                    "task_result": (context["outcome"].get("grasp_acquisition") or {}).get("status", "UNKNOWN")}
                row["evidence_trials"][tid] = vote
                votes = list(row["evidence_trials"].values())
                row["evidence_count"] = {label.lower(): sum(v["relation"] == label for v in votes)
                                         for label in ("SUPPORT", "CONTRADICT", "INCONCLUSIVE")}
                row["task_outcome_count"] = {label.lower(): sum(v["task_result"] == label for v in votes)
                                             for label in ("SUCCESS", "FAILURE", "UNKNOWN")}
                support = sum(v["weight"] for v in votes if v["relation"] == "SUPPORT")
                total = sum(v["weight"] for v in votes if v["relation"] != "INCONCLUSIVE")
                row["confidence"] = round(support / (2 + total), 4)
                row["confidence_kind"] = "conservative_evidence_score_not_calibrated_probability"
                # All effects stay soft; contrary or inconclusive new evidence
                # cannot activate a stronger ranking recommendation.
                if update["evidence_relation"] == "SUPPORT":
                    row["policy_effect"] = update["policy_effect"]
                elif "policy_effect" not in row:
                    row["policy_effect"] = {"candidate_ranking": "NONE", "reason": "No supporting evidence yet."}
                if row["confidence"] < .1:
                    row["policy_effect"]["candidate_ranking"] = "NONE"
                from datetime import datetime, timezone
                row["last_update"] = datetime.now(timezone.utc).isoformat()
                row["last_trial_id"] = tid
                state["rules"][key] = row
                receipt = {"rule_id": key, "status": "UPDATED", "evidence_count": row["evidence_count"],
                           "confidence": row["confidence"], "policy_effect": row["policy_effect"]}
            receipt["trial_id"] = tid
            state["processed_trials"][tid] = receipt
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.path)
            return receipt
