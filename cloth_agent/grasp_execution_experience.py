"""Learn selected-surface -> commanded XYZ only from localized contact evidence.

Commands, observed contact and selected points are distinct. A command/host-target
mismatch is a system error, never a calibration sample. This module records
conditional evidence; it does not authorize or apply robot corrections.
"""
from __future__ import annotations

import copy
import fcntl
import json
import math
from pathlib import Path

def _object(properties):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


TEXT = {"type": "string", "minLength": 1, "maxLength": 800}
CONFIDENCE = {"type": "number", "minimum": 0, "maximum": 1}
NUMBER_OR_NULL = {"type": ["number", "null"]}


def _vector(size):
    return {"type": ["array", "null"], "minItems": size, "maxItems": size,
            "items": {"type": "number"}}


STATUSES = ["ALIGNED_SUCCESS", "XY_MISALIGNED", "Z_MISALIGNED", "XYZ_MISALIGNED", "UNRESOLVED"]
OBSERVATIONS = ["XY_ALIGNED", "XY_MISALIGNED", "STABLE_CLOTH_ACQUISITION",
    "JAWS_ABOVE_CLOTH_AT_CLOSURE", "TABLE_CONTACT", "DOWNWARD_MOTION_BLOCKED_BY_TABLE",
    "EXCESSIVE_COMPRESSION", "INCONCLUSIVE"]
EVIDENCE_SCHEMA = {"type": "array", "minItems": 1, "maxItems": 8, "items": _object({
    "observation": {"enum": OBSERVATIONS},
    "evidence_ids": {"type": "array", "minItems": 1, "maxItems": 8,
                     "uniqueItems": True, "items": TEXT},
    "description": TEXT,
})}
GRASP_EXECUTION_DIAGNOSIS_SCHEMA = _object({
    "confidence": CONFIDENCE,
    "evidence": EVIDENCE_SCHEMA,
})
TRIAL_SCHEMA = _object({
    "selected_point": _object({
        "pixel_xy": _vector(2), "surface_xyz_mm": _vector(3),
        "camera": {"type": ["string", "null"]},
        "pixel_frame": {"enum": ["RAW_PERCEPTION", "UNKNOWN"]},
        "source": {"type": ["string", "null"]},
    }),
    "execution": _object({
        "runtime_authoritative_xyz_mm": _vector(3), "commanded_xyz_mm": _vector(3),
        "command_minus_surface_xyz_mm": _vector(3),
        "command_minus_authoritative_xyz_mm": _vector(3),
        "descent_below_surface_mm": NUMBER_OR_NULL,
        "command_source": {"type": ["string", "null"]},
        "authoritative_source": {"type": ["string", "null"]},
    }),
    "command_integrity": _object({
        "status": {"enum": ["MATCHED", "SYSTEM_CODE_FAILURE", "UNAVAILABLE"]},
        "mismatched_axes": {"type": "array", "maxItems": 3, "uniqueItems": True,
                            "items": {"enum": ["X", "Y", "Z"]}},
        "tolerance_mm": {"type": "number", "minimum": 0}, "reason": TEXT,
    }),
    # No calibrated contact-localization producer exists in the current runtime.
    # In particular, actual_ee_pose is TCP feedback, not a cloth contact point.
    "observed_contact": {"type": "null"},
})
CORRECTION_SCHEMA = _object({axis: NUMBER_OR_NULL for axis in ("x", "y", "z")})
GRASP_EXECUTION_EXPERIENCE_SCHEMA = _object({
    "status": {"enum": STATUSES}, "update_allowed": {"type": "boolean"},
    "trial": TRIAL_SCHEMA,
    "observed_result": _object({
        "acquisition": {"enum": ["SUCCESS", "FAILURE", "UNKNOWN"]},
        "policy_acquisition": {"enum": ["SUCCESS", "FAILURE", "UNKNOWN"]},
        "contact_alignment": {"enum": ["ALIGNED", "MISALIGNED", "UNKNOWN"]},
        "depth_interpretation": {"enum": ["EFFECTIVE", "TOO_SHALLOW", "TOO_DEEP", "UNKNOWN"]},
        "evidence": EVIDENCE_SCHEMA,
    }),
    "experience_update": _object({
        "kind": {"enum": ["REINFORCE", "DIRECTIONAL", "NONE"]},
        "axes": {"type": "array", "maxItems": 3, "uniqueItems": True,
                 "items": {"enum": ["X", "Y", "Z"]}},
        "correction_xyz_mm": CORRECTION_SCHEMA,
        "xy_direction": {"enum": ["KEEP", "RELOCALIZE_CONTACT", "NONE"]},
        "z_direction": {"enum": ["KEEP", "DEEPER", "SHALLOWER", "NONE"]},
        "confidence": CONFIDENCE, "reason": TEXT,
    }),
    "next_experiment": _object({
        "execution_status": {"enum": ["NO_CHANGE", "BLOCKED_BY_CAPABILITY"]},
        "recommendation": TEXT,
    }),
    "do_not_infer": {"type": "array", "minItems": 1, "maxItems": 8, "items": TEXT},
})

GRASP_EXECUTION_INSTRUCTION = (
    "Return grasp_execution_diagnosis separately using evidence observations, never model-generated "
    "XYZ measurements or corrections. Grasp Execution Experience means p_command=p_surface+delta_xyz "
    "for the ALREADY SELECTED point. It is not point selection or transport/release learning. "
    "The supplied grasp_execution_trial distinguishes selected surface XYZ, runtime authoritative "
    "command target, actual logged command and observed contact. Command offsets do not prove "
    "physical contact offsets. A command/authoritative-target mismatch is SYSTEM_CODE_FAILURE, "
    "never something to learn as compensation. Cite exact current interaction RGB IDs. XY_ALIGNED "
    "or XY_MISALIGNED requires identifiable selected physical feature and gripper contact in the "
    "same interaction evidence; never compare raw pixels across moving-camera frames without "
    "registration. Describe that correspondence and any occlusion. No measured observed-contact "
    "XYZ is currently available; do not invent it from TCP feedback or visual left/right. "
    "STABLE_CLOTH_ACQUISITION requires retained cloth during lift, not closure status. "
    "JAWS_ABOVE_CLOTH_AT_CLOSURE, TABLE_CONTACT, DOWNWARD_MOTION_BLOCKED_BY_TABLE and "
    "EXCESSIVE_COMPRESSION require direct contact evidence, not failed acquisition. "
    "The host derives ALIGNED_SUCCESS, XY_MISALIGNED, Z_MISALIGNED, XYZ_MISALIGNED or UNRESOLVED. "
    "Z-only diagnosis requires XY_ALIGNED. UNCHANGED, coupling, bad point, slip and wrong drag "
    "alone do not localize an execution mismatch. Use INCONCLUSIVE when uncertain. "
    "Keep execution calibration out of generic experience_update; keep point/motion rules there. "
    "Use EXECUTION_XY for landing correction at a fixed selected point, not CONTACT_XY (point "
    "reselection). Corrections are blocked by current host capabilities. Test one axis group "
    "at a time; never silently change both XY and Z. UNRESOLVED does not justify a learned "
    "deeper correction. A separately authorized host-bounded Z retry is an experiment, "
    "not a diagnosis or permanent policy update."
)


def _point(value, size=3):
    if (isinstance(value, (list, tuple)) and len(value) == size and
            all(type(v) in (int, float) and math.isfinite(v) for v in value)):
        return [float(v) for v in value]
    return None


def _delta(a, b):
    return _point([x - y for x, y in zip(a, b)]) if a is not None and b is not None else None


def runtime_execution_trial(record):
    """Only final host resolution and real successful action logs are authoritative."""
    audit = (record.get("planning_diagnostics") or {}).get("grasp_height_resolution") or {}
    resolution = audit.get("resolution") or {}
    surface = _point(resolution.get("surface_xyz_mm")) if resolution.get("valid") is True else None
    pixel = _point(audit.get("pixel_xy"), 2)
    target = _point(audit.get("runtime_authoritative_grasp_xyz_mm"))
    target_source = "planning_diagnostics.grasp_height_resolution.runtime_authoritative_grasp_xyz_mm"
    if target is None and "runtime_authoritative_grasp_xyz_mm" not in audit:
        xy = _point(audit.get("resolved_grasp_xy_mm"), 2)
        target = _point([*xy, audit.get("resolved_grasp_z_mm")]) if xy is not None else None
        target_source = "planning_diagnostics.grasp_height_resolution.resolved_grasp_xy_mm+resolved_grasp_z_mm"
    execution = record.get("execution") or {}
    command, command_source = None, None
    actions = execution.get("actual_robot_actions") or []
    closes = [i for i, a in enumerate(actions) if a.get("name") == "close_gripper"]
    if (execution.get("physical_execution") is True and len(closes) == 1
            and actions[closes[0]].get("success") is True):
        for index, action in enumerate(actions[:closes[0]]):
            if action.get("name") == "move":
                args = action.get("args") or {}
                command = _point([args.get(k) for k in ("x", "y", "z")]) if action.get("success") is True else None
                command_source = f"execution.actual_robot_actions[{index}].args" if command is not None else None
            elif action.get("name") != "open_gripper":
                command, command_source = None, None
    discrepancy = _delta(command, target)
    offset = _delta(command, surface)
    mismatched = [axis for axis, value in zip(("X", "Y", "Z"), discrepancy or []) if abs(value) > 1e-4]
    integrity = "SYSTEM_CODE_FAILURE" if mismatched else "MATCHED" if discrepancy is not None else "UNAVAILABLE"
    return {
        "selected_point": {"pixel_xy": pixel, "surface_xyz_mm": surface,
            "camera": audit.get("camera") if isinstance(audit.get("camera"), str) else None,
            "pixel_frame": "RAW_PERCEPTION" if pixel is not None else "UNKNOWN",
            "source": "planning_diagnostics.grasp_height_resolution.resolution.surface_xyz_mm" if surface is not None else None},
        "execution": {"runtime_authoritative_xyz_mm": target, "commanded_xyz_mm": command,
            "command_minus_surface_xyz_mm": offset, "command_minus_authoritative_xyz_mm": discrepancy,
            "descent_below_surface_mm": -offset[2] if offset is not None else None,
            "command_source": command_source, "authoritative_source": target_source if target is not None else None},
        "command_integrity": {"status": integrity, "mismatched_axes": mismatched, "tolerance_mm": .0001,
            "reason": {"MATCHED": "Logged command agrees with final runtime target; this does not establish physical contact.",
                "SYSTEM_CODE_FAILURE": "Logged XYZ differs from runtime authoritative XYZ; investigate execution code, do not learn compensation.",
                "UNAVAILABLE": "Actual command or runtime authoritative target unavailable; never substitute a plan or measured TCP pose."}[integrity]},
        "observed_contact": None,
    }


def unresolved_diagnosis():
    return {"confidence": 0.0, "evidence": [{"observation": "INCONCLUSIVE",
        "evidence_ids": ["policy_outcome"],
        "description": "Task-level outcome does not localize an execution error to X, Y or Z."}]}


def apply_grasp_execution_experience(diagnosis, context):
    from .fold_experience_learning import validate_schema
    diagnosis = copy.deepcopy(diagnosis if diagnosis is not None else unresolved_diagnosis())
    validate_schema(diagnosis, GRASP_EXECUTION_DIAGNOSIS_SCHEMA, "grasp_execution_diagnosis")
    trial = copy.deepcopy(context.get("grasp_execution_trial") or runtime_execution_trial({}))
    validate_schema(trial, TRIAL_SCHEMA, "grasp_execution_trial")
    catalog = {e["id"]: e for e in context.get("evidence_catalog", [])}
    observations = set()
    for evidence in diagnosis["evidence"]:
        if set(evidence["evidence_ids"]) - catalog.keys():
            raise ValueError("grasp execution diagnosis cites unknown evidence IDs")
        observation = evidence["observation"]
        roles = {"after_lift", "rollout"} if observation == "STABLE_CLOTH_ACQUISITION" else {
            "after_close", "before_lift", "rollout"}
        if any(catalog[e].get("kind") == "INTERACTION_RGB" and
               e in {f"current_{r}_rgb" for r in roles} for e in evidence["evidence_ids"]):
            observations.add(observation)
    aligned, misaligned = "XY_ALIGNED" in observations, "XY_MISALIGNED" in observations
    shallow = "JAWS_ABOVE_CLOTH_AT_CLOSURE" in observations
    deep = bool(observations & {"TABLE_CONTACT", "DOWNWARD_MOTION_BLOCKED_BY_TABLE", "EXCESSIVE_COMPRESSION"})
    stable = "STABLE_CLOTH_ACQUISITION" in observations
    alignment = "ALIGNED" if aligned and not misaligned else "MISALIGNED" if misaligned and not aligned else "UNKNOWN"
    conflict = (aligned and misaligned) or (shallow and deep) or (stable and shallow)
    depth = "TOO_SHALLOW" if shallow and not deep else "TOO_DEEP" if deep and not shallow else "UNKNOWN"
    status = "UNRESOLVED"
    if (not conflict and diagnosis["confidence"] >= .7 and trial["command_integrity"]["status"] == "MATCHED"
            and trial["execution"]["command_minus_surface_xyz_mm"] is not None):
        if alignment == "MISALIGNED":
            status = "XYZ_MISALIGNED" if depth != "UNKNOWN" else "XY_MISALIGNED"
        elif alignment == "ALIGNED":
            if depth != "UNKNOWN":
                status = "Z_MISALIGNED"
            elif stable:
                status, depth = "ALIGNED_SUCCESS", "EFFECTIVE"
    allowed = status != "UNRESOLVED"
    correction = {k: None for k in ("x", "y", "z")}
    axes, xy_direction, z_direction, kind = [], "NONE", "NONE", "NONE"
    reason = "Current evidence does not localize a learnable execution mismatch; no XYZ update."
    if trial["command_integrity"]["status"] == "SYSTEM_CODE_FAILURE":
        reason = trial["command_integrity"]["reason"]
    if status == "ALIGNED_SUCCESS":
        kind, axes, xy_direction, z_direction = "REINFORCE", ["X", "Y", "Z"], "KEEP", "KEEP"
        correction = dict(zip(("x", "y", "z"), trial["execution"]["command_minus_surface_xyz_mm"]))
        reason = "This relative XYZ correction acquired the selected point in this trial context; not a universal calibration."
    elif allowed:
        kind = "DIRECTIONAL"
        if status in {"XY_MISALIGNED", "XYZ_MISALIGNED"}:
            axes.extend(["X", "Y"])
            xy_direction = "RELOCALIZE_CONTACT"
        if status in {"Z_MISALIGNED", "XYZ_MISALIGNED"}:
            axes.append("Z")
            z_direction = "DEEPER" if depth == "TOO_SHALLOW" else "SHALLOWER"
        reason = "Localized mismatch supplies directional evidence only; no calibrated metric contact error, so new correction components remain null."
    policy = (context.get("outcome", {}).get("grasp_acquisition") or {}).get("status", "UNKNOWN")
    recommendation = {
        "UNRESOLVED": "Collect discriminating contact evidence; do not automatically change X, Y or Z.",
        "ALIGNED_SUCCESS": "Prefer this relative correction in comparable conditions, recomputing command from the new selected surface.",
        "XY_MISALIGNED": "Keep the selected point and depth fixed; localize the XY landing error before testing one calibrated XY correction.",
        "Z_MISALIGNED": f"Keep selected point and XY alignment fixed; test {z_direction.lower()} relative descent subject to host validation.",
        "XYZ_MISALIGNED": "Both XY and Z have independent mismatch evidence. Resolve XY contact localization first while holding Z fixed; do not change both at once.",
    }[status]
    if trial["command_integrity"]["status"] == "SYSTEM_CODE_FAILURE":
        recommendation = "Investigate runtime-target versus command inconsistency before any compensation experiment."
    result = {"status": status, "update_allowed": allowed, "trial": trial,
        "observed_result": {"acquisition": "SUCCESS" if stable and not conflict and diagnosis["confidence"] >= .7 else "UNKNOWN",
            "policy_acquisition": policy if policy in {"SUCCESS", "FAILURE", "UNKNOWN"} else "UNKNOWN",
            "contact_alignment": alignment if not conflict and diagnosis["confidence"] >= .7 else "UNKNOWN",
            "depth_interpretation": depth if allowed else "UNKNOWN", "evidence": diagnosis["evidence"]},
        "experience_update": {"kind": kind, "axes": axes, "correction_xyz_mm": correction,
            "xy_direction": xy_direction, "z_direction": z_direction,
            "confidence": diagnosis["confidence"] if allowed else 0.0, "reason": reason},
        "next_experiment": {"execution_status": "BLOCKED_BY_CAPABILITY" if kind == "DIRECTIONAL" else "NO_CHANGE",
                            "recommendation": recommendation},
        "do_not_infer": ["Do not learn absolute world XYZ or a universal correction.",
            "Do not learn software command mismatches as physical compensation.",
            "Do not infer XY/Z causes from UNCHANGED, empty jaws, coupling or task failure alone.",
            "Do not change grasp-point selection or target/motion rules from execution correction.",
            "Do not convert visual left/right into robot-base millimetres without calibrated contact localization."]}
    validate_schema(result, GRASP_EXECUTION_EXPERIENCE_SCHEMA, "grasp_execution_experience")
    return result


def initial_execution_experience(record):
    return apply_grasp_execution_experience(None, {"grasp_execution_trial": runtime_execution_trial(record),
        "outcome": record.get("evaluation") or {},
        "evidence_catalog": [{"id": "policy_outcome", "kind": "POLICY_LABEL"}]})


def gate_execution_experiment(experiment, experience):
    variable = experiment["single_change"]["variable"]
    if variable not in {"CONTACT_Z", "EXECUTION_XY"}:
        return  # CONTACT_XY remains point selection, not landing calibration.
    update = experience["experience_update"]
    allowed = (update["kind"] == "DIRECTIONAL" and (
        variable == "CONTACT_Z" and "Z" in update["axes"] or
        variable == "EXECUTION_XY" and "X" in update["axes"]))
    # With two localized axes, choose XY first, never both in one experiment.
    if experience["status"] == "XYZ_MISALIGNED" and variable == "CONTACT_Z":
        allowed = False
    experiment.update(status="BLOCKED_BY_CAPABILITY" if allowed else "NO_EXPERIMENT",
        primary_hypothesis=update["reason"],
        single_change={"variable": variable if allowed else "NONE",
                       "description": experience["next_experiment"]["recommendation"]},
        held_constant=([{"variable": "CONTACT_XY", "description": "Keep the selected physical point fixed."},
                        {"variable": "CONTACT_Z" if variable == "EXECUTION_XY" else "EXECUTION_XY",
                         "description": "Hold the other execution axis group fixed."}] if allowed else []),
        expected_observation="Contact on the selected point and stable acquisition without table collision.",
        interpretation_if_success="Support only the tested relative correction in comparable conditions.",
        interpretation_if_failure="Do not infer XYZ causes from task failure alone.")


def persist_execution_trial(root, experience, context, *, source_record):
    checked = apply_grasp_execution_experience(context["grasp_execution_diagnosis"], context)
    if checked != experience:
        raise ValueError("grasp execution trial differs from host-validated evidence")
    if not checked["update_allowed"]:
        return {"status": "NO_EXECUTION_UPDATE", "trial_id": context["trial_id"]}
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "grasp_execution_trials.json"
    with (root / "grasp_execution_trials.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "trials": {}}
        if state.get("schema_version") != 1:
            raise ValueError("unsupported grasp execution trial schema")
        tid = context["trial_id"]
        if tid in state["trials"]:
            return {"status": "ALREADY_APPLIED", "trial_id": tid}
        state["trials"][tid] = {"trial_id": tid, "step": context["step"], "source_record": source_record,
            "grasp_execution_experience": copy.deepcopy(checked),
            "context_limitations": "Compare cloth, support, camera calibration, tool and selected-point context before transfer. No automatic control update."}
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    return {"status": "RECORDED", "trial_id": tid}
