"""Legacy depth-only schema/helpers, retained for historical record tooling.

The live fold pipeline now uses grasp_execution_experience. It does not call
this module or promote old depth-only evidence into XYZ correction evidence.

Geometry is computed by the host from runtime records, never supplied by the
model. Stored trials are observations, not executable robot targets or a global
calibration constant. No automatic interval learner is implemented here.
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
INTERPRETATIONS = ["EFFECTIVE", "TOO_SHALLOW", "TOO_DEEP", "UNKNOWN"]
DIRECTIONS = dict(zip(INTERPRETATIONS, ["KEEP", "DEEPER", "SHALLOWER", "NONE"]))
OBSERVATIONS = {
    "STABLE_CLOTH_ACQUISITION": "EFFECTIVE",
    "JAWS_ABOVE_CLOTH_AT_CLOSURE": "TOO_SHALLOW",
    "TABLE_CONTACT": "TOO_DEEP",
    "DOWNWARD_MOTION_BLOCKED_BY_TABLE": "TOO_DEEP",
    "EXCESSIVE_COMPRESSION": "TOO_DEEP",
    "NON_DEPTH_OR_INCONCLUSIVE": "UNKNOWN",
}
EVIDENCE_SCHEMA = {"type": "array", "minItems": 1, "maxItems": 8, "items": _object({
    "observation": {"enum": list(OBSERVATIONS)},
    "evidence_ids": {"type": "array", "minItems": 1, "maxItems": 8,
                     "uniqueItems": True, "items": TEXT},
    "description": TEXT,
})}
GRASP_DEPTH_DIAGNOSIS_SCHEMA = _object({
    "depth_interpretation": {"enum": INTERPRETATIONS},
    "confidence": CONFIDENCE,
    "evidence": EVIDENCE_SCHEMA,
})
GEOMETRY_SCHEMA = _object({
    **{key: NUMBER_OR_NULL for key in (
        "runtime_authoritative_surface_z_mm", "actual_commanded_grasp_z_mm",
        "descent_below_surface_mm", "table_z_mm", "clearance_above_table_mm")},
    "surface_z_confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "surface_source": {"type": ["string", "null"]},
    "command_source": {"type": ["string", "null"]},
    "pairing_status": {"enum": ["MATCHED", "UNAVAILABLE"]},
})
GRASP_EXPERIENCE_SCHEMA = _object({
    "geometry": GEOMETRY_SCHEMA,
    "outcome": _object({
        "acquisition_status": {"enum": ["SUCCESS", "FAILURE", "UNKNOWN"]},
        "depth_interpretation": {"enum": INTERPRETATIONS},
        "evidence": EVIDENCE_SCHEMA,
    }),
    "experience_update": _object({
        "update_direction": {"enum": list(DIRECTIONS.values())},
        "supported_statement": TEXT,
        "confidence": CONFIDENCE,
    }),
    "next_experiment": _object({
        "recommendation": TEXT,
        "execution_status": {"enum": ["NO_DEPTH_CHANGE", "BLOCKED_BY_CAPABILITY"]},
    }),
    "do_not_infer": {"type": "array", "minItems": 1, "maxItems": 8, "items": TEXT},
})
GUARDS = [
    "Do not learn an absolute world-frame grasp Z or a universal descent constant.",
    "Do not modify grasp-point-selection or target/motion experience from this depth result.",
    "Do not conclude unchanged final RGB means the grasp was too shallow or jaws were empty.",
    "Do not infer stable cloth acquisition from successful robot motion or gripper closure.",
]
GRASP_DEPTH_INSTRUCTION = (
    "Return grasp_depth_diagnosis separately. It only concerns how far to descend below the "
    "runtime estimated surface at an already selected point. Geometry is host supplied; never "
    "invent command Z, surface confidence or a replacement measurement. Keep all depth calibration "
    "out of the generic experience_update (reserved for point and motion knowledge). "
    "Classify EFFECTIVE only with clearly observed stable cloth acquisition; TOO_SHALLOW only "
    "when jaws visibly remain above cloth at closure; TOO_DEEP only with clear table contact, "
    "table-blocked descent or excessive compression. Cite exact current interaction evidence IDs "
    "and describe the discriminating observation, not merely a hypothesis. Mechanical grasp "
    "status, failed acquisition, coupling, wrong drag/release, CHANGED or UNCHANGED alone cannot "
    "diagnose depth. Use UNKNOWN and NON_DEPTH_OR_INCONCLUSIVE when uncertain. A confidence below "
    "0.7, conflicting depth observations or missing authoritative geometry cannot update depth. "
    "Directions are host derived: EFFECTIVE=KEEP, TOO_SHALLOW=DEEPER, TOO_DEEP=SHALLOWER, "
    "UNKNOWN=NONE. KEEP refers to relative descent in comparable conditions, never absolute Z. "
    "Do not propose deeper Z on UNKNOWN. Depth adjustments remain BLOCKED_BY_CAPABILITY."
)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def runtime_geometry(record):
    """Read the executed closure command and the final host surface resolution.

    RobotAPI.move logs its float args, passes them unchanged to XArmBackend.move,
    which passes z unchanged to set_position. actual_ee_pose is measured feedback,
    not the commanded target. Neither proposal nor requested_robot_actions is a
    fallback. Ambiguous multiple grasps or mismatched XY cannot form depth trials.
    """
    geometry = {key: None for key in GEOMETRY_SCHEMA["properties"]}
    geometry["pairing_status"] = "UNAVAILABLE"
    execution = record.get("execution") or {}
    if execution.get("physical_execution") is not True:
        return geometry
    actions = execution.get("actual_robot_actions") or []
    closes = [i for i, a in enumerate(actions) if a.get("name") == "close_gripper"]
    if len(closes) != 1 or actions[closes[0]].get("success") is not True:
        return geometry
    contact = None
    for i, action in enumerate(actions[:closes[0]]):
        if action.get("name") == "move":
            contact = (i, action) if action.get("success") is True else None
        elif action.get("name") not in {"open_gripper"}:
            contact = None  # Home/shake/unknown motion invalidates the last move.
    if contact is None:
        return geometry
    index, action = contact
    args = action.get("args") or {}
    if not all(_number(args.get(k)) for k in ("x", "y", "z")):
        return geometry
    geometry["actual_commanded_grasp_z_mm"] = float(args["z"])
    geometry["command_source"] = f"execution.actual_robot_actions[{index}].args.z"
    audit = (record.get("planning_diagnostics") or {}).get("grasp_height_resolution") or {}
    resolution = audit.get("resolution") or {}
    if resolution.get("valid") is not True or not _number(resolution.get("surface_z_mm")):
        return geometry
    xy = audit.get("resolved_grasp_xy_mm", (resolution.get("surface_xyz_mm") or [])[:2])
    if (not isinstance(xy, (list, tuple)) or len(xy) != 2 or not all(_number(v) for v in xy)
            or any(abs(args[k] - v) > 1e-4 for k, v in zip(("x", "y"), xy))):
        return geometry
    surface = float(resolution["surface_z_mm"])
    table = resolution.get("authoritative_table_z_mm")
    confidence = (audit.get("measurement") or {}).get("surface_z_confidence")
    descent = surface - args["z"]
    clearance = args["z"] - table if _number(table) else None
    if not _number(descent) or (clearance is not None and not _number(clearance)):
        return geometry
    geometry.update(runtime_authoritative_surface_z_mm=surface,
        descent_below_surface_mm=descent,
        table_z_mm=float(table) if _number(table) else None,
        clearance_above_table_mm=clearance,
        surface_z_confidence=float(confidence) if _number(confidence) and 0 <= confidence <= 1 else None,
        surface_source="planning_diagnostics.grasp_height_resolution.resolution.surface_z_mm",
        pairing_status="MATCHED")
    return geometry


def unknown_diagnosis():
    return {"depth_interpretation": "UNKNOWN", "confidence": 0.0, "evidence": [{
        "observation": "NON_DEPTH_OR_INCONCLUSIVE", "evidence_ids": ["policy_outcome"],
        "description": "Task-level outcome alone contains no discriminating grasp-depth evidence."}]}


def make_grasp_experience(diagnosis, context):
    """Gate model observations using actual geometry and current evidence sources."""
    from .fold_experience_learning import validate_schema
    diagnosis = copy.deepcopy(diagnosis if diagnosis is not None else unknown_diagnosis())
    validate_schema(diagnosis, GRASP_DEPTH_DIAGNOSIS_SCHEMA, "grasp_depth_diagnosis")
    geometry = copy.deepcopy(context.get("grasp_depth_geometry") or runtime_geometry({}))
    validate_schema(geometry, GEOMETRY_SCHEMA, "grasp_depth_geometry")
    catalog = {e["id"]: e for e in context.get("evidence_catalog", [])}
    supported = set()
    for evidence in diagnosis["evidence"]:
        if set(evidence["evidence_ids"]) - catalog.keys():
            raise ValueError("grasp depth diagnosis cites unknown evidence IDs")
        interpretation = OBSERVATIONS[evidence["observation"]]
        roles = {"after_lift", "rollout"} if interpretation == "EFFECTIVE" else {
            "after_close", "before_lift", "rollout"}
        if any(catalog[e].get("kind") == "INTERACTION_RGB" and e in {
                f"current_{role}_rgb" for role in roles} for e in evidence["evidence_ids"]):
            if interpretation != "UNKNOWN":
                supported.add(interpretation)
    interpretation = diagnosis["depth_interpretation"]
    if (interpretation == "UNKNOWN" or supported != {interpretation} or diagnosis["confidence"] < .7
            or geometry["pairing_status"] != "MATCHED"
            or geometry["descent_below_surface_mm"] is None):
        interpretation = "UNKNOWN"
    acquisition = (context.get("outcome", {}).get("grasp_acquisition") or {}).get("status", "UNKNOWN")
    if acquisition not in {"SUCCESS", "FAILURE", "UNKNOWN"}:
        acquisition = "UNKNOWN"
    direction = DIRECTIONS[interpretation]
    descent = geometry["descent_below_surface_mm"]
    statement = {
        "EFFECTIVE": f"Relative descent near {descent} mm supported stable acquisition in this trial context only.",
        "TOO_SHALLOW": f"In comparable conditions, sufficient descent likely exceeds the observed {descent} mm.",
        "TOO_DEEP": f"In comparable conditions, suitable descent likely lies below the observed {descent} mm.",
        "UNKNOWN": "Current evidence does not justify a grasp-depth update.",
    }[interpretation]
    recommendation = {
        "KEEP": "Prefer the observed relative descent in comparable conditions; recompute commanded Z from the new surface.",
        "DEEPER": "Test increased descent relative to the newly estimated surface, subject to host safety validation.",
        "SHALLOWER": "Test reduced descent relative to the newly estimated surface, subject to host safety validation.",
        "NONE": "Current evidence does not justify a grasp-depth update; do not automatically descend deeper.",
    }[direction]
    result = {"geometry": geometry,
        "outcome": {"acquisition_status": acquisition, "depth_interpretation": interpretation,
                    "evidence": diagnosis["evidence"]},
        "experience_update": {"update_direction": direction, "supported_statement": statement,
                              "confidence": diagnosis["confidence"] if interpretation != "UNKNOWN" else 0.0},
        "next_experiment": {"recommendation": recommendation,
            "execution_status": "BLOCKED_BY_CAPABILITY" if direction in {"DEEPER", "SHALLOWER"} else "NO_DEPTH_CHANGE"},
        "do_not_infer": copy.deepcopy(GUARDS)}
    validate_schema(result, GRASP_EXPERIENCE_SCHEMA, "grasp_experience")
    return result


def initial_grasp_experience(record):
    return make_grasp_experience(None, {"grasp_depth_geometry": runtime_geometry(record),
        "outcome": record.get("evaluation") or {},
        "evidence_catalog": [{"id": "policy_outcome", "kind": "POLICY_LABEL"}]})


def gate_depth_experiment(experiment, experience):
    """Leave point/motion proposals intact; depth proposals must agree with evidence."""
    if experiment["single_change"]["variable"] != "CONTACT_Z":
        return
    direction = experience["experience_update"]["update_direction"]
    if direction in {"NONE", "KEEP"}:
        experiment.update(status="NO_EXPERIMENT", held_constant=[],
            primary_hypothesis=experience["experience_update"]["supported_statement"],
            single_change={"variable": "NONE", "description": experience["next_experiment"]["recommendation"]},
            expected_observation="No depth change is justified by this analysis.",
            interpretation_if_success="Do not attribute a new outcome to an untested depth change.",
            interpretation_if_failure="Failure alone does not justify deeper descent.")
    else:
        experiment.update(status="BLOCKED_BY_CAPABILITY",
            primary_hypothesis=experience["experience_update"]["supported_statement"],
            single_change={"variable": "CONTACT_Z", "description": experience["next_experiment"]["recommendation"]},
            expected_observation="Stable acquisition without table contact or excessive compression.",
            interpretation_if_success="Supports the relative descent hypothesis only in comparable conditions.",
            interpretation_if_failure="Do not extrapolate a cause from task-level failure alone.")


def persist_depth_trial(root, experience, context, *, source_record):
    """Separate numeric trial ledger. UNKNOWN never mutates long-term depth state."""
    # Recompute the gate at the persistence boundary, not from a caller's direction.
    checked = make_grasp_experience(context["grasp_depth_diagnosis"], context)
    if checked != experience:
        raise ValueError("grasp depth trial differs from host-validated evidence")
    if checked["experience_update"]["update_direction"] == "NONE":
        return {"status": "NO_DEPTH_UPDATE", "trial_id": context["trial_id"]}
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "grasp_depth_trials.json"
    with (root / "grasp_depth_trials.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "trials": {}}
        if state.get("schema_version") != 1:
            raise ValueError("unsupported grasp depth trial schema")
        tid = context["trial_id"]
        if tid in state["trials"]:
            return {"status": "ALREADY_APPLIED", "trial_id": tid}
        state["trials"][tid] = {"trial_id": tid, "step": context["step"], "source_record": source_record,
            "grasp_experience": copy.deepcopy(checked),
            "context_limitations": "Single selected-point observation; geometry and cloth/support conditions must be comparable. No universal offset or automatic robot policy update."}
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    return {"status": "RECORDED", "trial_id": tid}
