"""One bounded, Z-only follow-up experiment after an unsuccessful acquisition.

Scheduling an experiment is not a causal depth diagnosis or a policy update.
The controller's existing geometry limits are never expanded here.
"""
from __future__ import annotations

import copy
import math
from dataclasses import replace

from .fold_recovery import released_and_homed
from .grasp_execution_experience import runtime_execution_trial


def retry_eligibility(record, *, enabled=True):
    reason = None
    evaluation = record.get("evaluation") or {}
    execution = record.get("execution") or {}
    trial = runtime_execution_trial(record)
    acquisition = evaluation.get("grasp_acquisition") or {}
    observation = (record.get("grasp_execution_experience") or {}).get("observed_result") or {}
    if not enabled:
        reason = "Height retry disabled."
    elif record.get("height_retry") or record.get("inherited_lesson"):
        reason = "A secondary or inherited attempt cannot schedule another height retry."
    elif record.get("status") in {"FAILED", "INTERRUPTED", "PLANNING_FAILURE"}:
        reason = "Only a completed physical attempt can schedule a height retry."
    elif record.get("mode") not in {"FOLD", "REPAIR_SLEEVE"}:
        reason = "Height retry requires a completed fold or sleeve-repair trajectory."
    elif acquisition.get("status") != "FAILURE" or evaluation.get("earliest_failure_stage") != "ACQUISITION":
        reason = "Evaluation did not report acquisition failure."
    elif (evaluation.get("perception_comparison") or {}).get("status") in {"CHANGED", "UNCOMPARABLE"}:
        reason = "The selected scene changed or cannot be compared; return to normal planning."
    elif execution.get("physical_execution") is not True or not released_and_homed(execution):
        reason = "Physical execution, release and Home must be confirmed first."
    elif any(a.get("success") is not True for a in execution.get("actual_robot_actions", [])):
        reason = "Every action must have a confirmed successful command result."
    elif trial["command_integrity"]["status"] != "MATCHED":
        reason = "Runtime command integrity is unavailable or reports a system/code failure."
    elif trial["selected_point"]["surface_xyz_mm"] is None or trial["selected_point"]["pixel_xy"] is None:
        reason = "Missing authoritative surface geometry or selected pixel."
    elif observation.get("acquisition") == "SUCCESS":
        reason = "Independent evidence reports stable acquisition; task failure does not justify a missed-grasp retry."
    elif observation.get("contact_alignment") == "MISALIGNED":
        reason = "Evidence localizes XY misalignment; a fixed-XY height retry is not applicable."
    after = record.get("supervisor_after") or {}
    if reason is None and (after.get("trajectory_decision") == "STOP" or after.get("status") in {"BLOCKED", "COMPLETE"}):
        reason = "Supervisor terminal decision takes precedence over a height retry."
    return {"status": "NOT_SCHEDULED" if reason else "SCHEDULED",
        "reason": reason or "Acquisition failed; reuse the executed commands for one bounded Z-only hypothesis test.",
        "parent_record_id": record.get("record_id"), "parent_iteration": record.get("iteration"),
        "causal_claim": "NONE", "maximum_secondary_attempts": 1}


def _contact_index(actions):
    closes = [i for i, a in enumerate(actions) if a.get("name") == "close_gripper"]
    if len(closes) != 1:
        raise ValueError("height retry requires exactly one closure")
    indices = [i for i, a in enumerate(actions[:closes[0]]) if a.get("name") == "move"]
    if not indices:
        raise ValueError("height retry requires a contact move")
    return indices[-1]


def height_options(record, *, step_mm):
    """Compute safe relative-descent options; never clamp into a no-op trial."""
    if type(step_mm) not in (int, float) or not math.isfinite(step_mm) or not 0 < step_mm <= 3:
        raise ValueError("height retry step must be finite and in (0, 3] mm")
    eligibility = retry_eligibility(record)
    if eligibility["status"] != "SCHEDULED":
        raise ValueError(eligibility["reason"])
    prior = runtime_execution_trial(record)
    resolution = record["planning_diagnostics"]["grasp_height_resolution"]["resolution"]
    surface = prior["selected_point"]["surface_xyz_mm"]
    if (not isinstance(surface, (list, tuple)) or len(surface) != 3 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in surface)):
        raise ValueError("current surface must contain finite XYZ")
    for key in ("lower_z_mm", "minimum_compression_mm", "maximum_compression_mm"):
        if type(resolution.get(key)) not in (int, float) or not math.isfinite(resolution[key]):
            raise ValueError(f"height retry requires finite {key}")
    actions = [{"name": a["name"], "args": copy.deepcopy(a["args"])}
               for a in record["execution"]["actual_robot_actions"]]
    contact_index = _contact_index(actions)
    prior_descent = prior["execution"]["descent_below_surface_mm"]
    prior_z = actions[contact_index]["args"]["z"]
    close_index = next(i for i, a in enumerate(actions) if a["name"] == "close_gripper")
    lift = next((a for a in actions[close_index + 1:] if a["name"] == "move"), None)
    if lift is None:
        raise ValueError("height retry needs a validated lift after closure")
    interpreted = ((record.get("grasp_execution_experience") or {}).get("observed_result") or {}).get("depth_interpretation")
    options, blocked = {}, {}
    for choice, delta in (("DEEPER", step_mm), ("SHALLOWER", -step_mm)):
        descent = prior_descent + delta
        z = prior_z - delta
        reason = None
        if not math.isfinite(descent) or not math.isfinite(z):
            reason = "Non-finite depth calculation."
        elif not resolution["minimum_compression_mm"] <= descent <= resolution["maximum_compression_mm"]:
            reason = "Outside configured surface compression limits."
        elif z < resolution["lower_z_mm"]:
            reason = "Below the current robot/support/table safety floor."
        elif abs(z - prior_z) < .05:
            reason = "Adjustment would leave command Z unchanged."
        elif lift["args"]["z"] - z < 30:
            reason = "Would violate the existing minimum 30 mm lift with the other targets fixed."
        elif choice == "DEEPER" and interpreted == "TOO_DEEP":
            reason = "Independent excessive-depth evidence prohibits a deeper test."
        if reason:
            blocked[choice] = reason
        else:
            options[choice] = {"descent_below_surface_mm": descent,
                "commanded_z_mm": z, "delta_descent_mm": delta,
                "delta_commanded_z_mm": z - prior_z}
    return {"options": options, "blocked_options": blocked, "contact_action_index": contact_index,
            "locked_actions": actions, "prior_descent_below_surface_mm": prior_descent}


def compile_height_retry(record, choices, *, allowed_skill_names=None):
    """Select a bounded host experiment without invoking a model or re-grounding."""
    from .free_exploration import validate_exploration_payload
    observed = (record.get("grasp_execution_experience") or {}).get("observed_result") or {}
    choice = "SHALLOWER" if observed.get("depth_interpretation") == "TOO_DEEP" else "DEEPER"
    if choice not in choices["options"]:
        raise ValueError("Requested height retry unavailable: " +
                         choices["blocked_options"].get(choice, choice))
    option = choices["options"][choice]
    actions = copy.deepcopy(choices["locked_actions"])
    actions[choices["contact_action_index"]]["args"]["z"] = option["commanded_z_mm"]
    payload = copy.deepcopy(record["execution_proposal"])
    selected = payload.pop("selected_grasp", None)
    payload.update(actions=actions, reveal_strategy="Z-only follow-up hypothesis test: " + choice)
    proposal = validate_exploration_payload(payload, allowed_skill_names=allowed_skill_names)
    proposal = replace(proposal, selected_grasp=selected)
    metadata = {"status": "PLANNED", "parent_record_id": record.get("record_id"),
        "parent_iteration": record.get("iteration"), "choice": choice,
        **option, "prior_descent_below_surface_mm": choices["prior_descent_below_surface_mm"],
        "single_change": "CONTACT_Z", "causal_claim": "UNTESTED_HYPOTHESIS",
        "held_constant": ["selected_point", "XY", "yaw", "other_trajectory_targets"],
        "command_source": "previous_actual_robot_actions"}
    return proposal, metadata


def validate_locked_retry(actions, choices, metadata):
    expected = copy.deepcopy(choices["locked_actions"])
    expected[choices["contact_action_index"]]["args"]["z"] = metadata["commanded_z_mm"]
    actual = [{"name": a["name"], "args": a["args"]} for a in actions]
    if actual != expected:
        raise ValueError("Z-only retry changed XY/yaw/action sequence or another trajectory target")
