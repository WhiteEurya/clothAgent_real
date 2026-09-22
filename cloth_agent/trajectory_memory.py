"""Bounded, evidence-backed previous-attempt context for fold planning.

Keep this separate from the generic semantic-history filter: relative motion
and execution provenance must survive, but robot state and absolute positions
must not be relayed to the remote visual planner.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, UnidentifiedImageError


HISTORY_RGB_NAMES = frozenset({
    "history_before_rgb.png", "history_before_lift_rgb.png",
    "history_after_close_rgb.png", "history_after_lift_rgb.png",
    "history_after_rgb.png", "history_rollout_rgb.png",
})

TRAJECTORY_MEMORY_INSTRUCTION = (
    "PREVIOUS ATTEMPT EVIDENCE: Compare the previous model proposal, host-validated "
    "commands, and execution log with its RGB outcome. Only completed execution-log "
    "entries confirm completed robot actions; a failed/unfinished entry may have moved "
    "partially. A proposal, preflight simulation, or successful jaw closure does not "
    "prove cloth acquisition. Missing evidence means UNKNOWN. In your strategy and "
    "motion_intent explain which previous segment may have failed, what you will keep, "
    "what you will change, and which observation would test that hypothesis. Treat "
    "history_* images and their pixels as historical evidence only: they are NOT the "
    "current scene or executable coordinate sources. Re-select all targets from the "
    "CURRENT RGB. Relative motion uses robot-base axes, not image/garment axes; never "
    "replay it as a current trajectory."
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _point(value: Any, size: int = 3) -> list[float] | None:
    if isinstance(value, (tuple, list)) and len(value) >= size and all(_number(v) for v in value[:size]):
        return list(value[:size])
    return None


def _actions(value: Any) -> list[Mapping[str, Any]]:
    return [a for a in value if isinstance(a, Mapping)] if isinstance(value, (tuple, list)) else []


def _closure_origin(actions: Sequence[Mapping[str, Any]]) -> list[float] | None:
    last_move = None
    for action in actions:
        if action.get("name") == "close_gripper":
            return last_move
        if action.get("name") == "move":
            args = _mapping(action.get("args"))
            last_move = _point([args.get(k) for k in ("x", "y", "z")])
    return None


def _relative(point: list[float] | None, origin: list[float] | None) -> list[float] | None:
    return [round(v - ref, 3) for v, ref in zip(point, origin)] if point and origin else None


def _summarize_actions(actions, origin, *, executed=False):
    result = []
    for index, action in enumerate(actions[:64], 1):
        name = action.get("name")
        item = {"action_index": index, "name": name}
        if name == "move":
            args = _mapping(action.get("args"))
            item["commanded_offset_from_contact_mm"] = _relative(
                _point([args.get(k) for k in ("x", "y", "z")]), origin)
            item["commanded_yaw_relative_home_deg"] = args.get("yaw") if _number(args.get("yaw")) else None
        if executed:
            item["completion"] = "COMPLETED" if action.get("success") is True else "ATTEMPTED_NOT_CONFIRMED"
            item["error_type"] = str(action["error"]).split(":", 1)[0][:80] if action.get("error") else None
            item["measured_offset_from_contact_mm"] = _relative(_point(action.get("actual_ee_pose")), origin)
        result.append(item)
    return {"action_count": len(actions), "truncated": len(actions) > 64, "actions": result}


def _attempt_status(row):
    execution = _mapping(row.get("execution"))
    if execution.get("physical_execution") is True:
        return "PHYSICAL_COMPLETED" if execution.get("execution_completed") is True else "PHYSICAL_INCOMPLETE_OR_UNKNOWN"
    if execution.get("physical_execution") is False:
        return "NOT_PHYSICALLY_EXECUTED"
    failure = _mapping(row.get("planning_failure"))
    if failure.get("fold_command_sent") is False:
        return "NOT_PHYSICALLY_EXECUTED"
    return "EXECUTION_UNKNOWN"


def prepare_trajectory_memory(history, step: str | None, run_dir: Path, output: Path):
    """Select from full history, independently of the recent text window.

    Only this run's same-step records are eligible. Copy up to six stable RGB
    artifacts with distinct names so they cannot masquerade as current images.
    """
    rows = [row for row in history if isinstance(row, Mapping)
            and not row.get("inherited_lesson") and step and row.get("planned_step") == step]
    if not rows:
        return None, []
    # Import lazily to share the established redaction for narrative outcomes,
    # without subjecting the separately constructed trajectory to its allowlist.
    from .remote_fold import semantic_history

    latest = rows[-1]
    context = {"schema_version": 1, "step": step, "instruction": TRAJECTORY_MEMORY_INSTRUCTION,
        "latest_attempt": {"iteration": latest.get("iteration"), "mode": latest.get("mode"),
            "execution_status": _attempt_status(latest),
            "failure_detection": semantic_history(latest.get("failure_detection"))},
        "previous_physical_attempt": None}
    physical = next((row for row in reversed(rows)
        if _mapping(row.get("execution")).get("physical_execution") is True
        and _actions(_mapping(row.get("execution")).get("actual_robot_actions"))), None)
    images = []
    if physical is not None:
        execution = _mapping(physical.get("execution"))
        actual = _actions(execution.get("actual_robot_actions"))
        proposed = _actions(_mapping(physical.get("proposal")).get("actions"))
        validated = _actions(_mapping(physical.get("execution_proposal")).get("actions"))
        if not validated:
            validated = _actions(_mapping(physical.get("trajectory")).get("actions"))
        origin = _closure_origin(validated)
        origin_source = "host_validated_closure_target" if origin else "logged_closure_command_target"
        if origin is None:
            origin = _closure_origin(actual)
        attempt = {"iteration": physical.get("iteration"), "mode": physical.get("mode"),
            "execution_status": _attempt_status(physical),
            "coordinate_frame": "robot_base_axes_relative_to_contact_origin",
            "contact_origin_source": origin_source if origin else "UNAVAILABLE",
            "contact_origin_available": origin is not None,
            "intended_strategy": semantic_history(_mapping(physical.get("proposal")).get("reveal_strategy")),
            "expected_observation": semantic_history(_mapping(physical.get("proposal")).get("expected_observation")),
            "model_proposed": _summarize_actions(proposed, origin),
            "host_validated": _summarize_actions(validated, origin),
            "execution_log": _summarize_actions(actual, origin, executed=True),
            "evaluation": semantic_history(physical.get("evaluation")),
            "failure_detection": semantic_history(physical.get("failure_detection")),
            "images": [], "grasp_in_before_image": None}
        context["previous_physical_attempt"] = attempt
        root = run_dir.resolve()

        def add_image(role, candidates, allowed_names, *, upright=False):
            entry = {"role": role, "status": "UNAVAILABLE"}
            attempt["images"].append(entry)
            for candidate in candidates:
                if not isinstance(candidate, (str, Path)):
                    continue
                path = Path(candidate).resolve()
                # workspace/perception_views is overwritten on each capture.
                if (root not in path.parents or not path.is_file() or
                        root / "workspace" / "perception_views" in path.parents or
                        path.name.lower() not in allowed_names):
                    continue
                try:
                    with Image.open(path) as source:
                        image = source.convert("RGB")
                        if upright and path.name.lower() == "camera_0_a.png":
                            image = image.rotate(-90, expand=True)
                        output.mkdir(parents=True, exist_ok=True)
                        target = output / f"history_{role}_rgb.png"
                        image.save(target)
                        entry.update(status="AVAILABLE", name=target.name, size=list(image.size),
                            coordinate_frame="clockwise90_upright" if upright else "native_capture")
                        images.append(target)
                        return entry
                except (OSError, UnidentifiedImageError):
                    continue
            return entry

        before = add_image("before", physical.get("before_images") or [],
            {"camera_a_rgb_upright.png", "camera_0_a.png"}, upright=True)
        snapshots = _mapping(_mapping(physical.get("recording")).get("grasp_snapshots"))
        roles = {"before_lift": "Contact frame; may be during closure, not proof of completed closure.",
            "after_close": "After confirmed closure, before lift; not proof of cloth acquisition.",
            "after_lift": "Asynchronous frame requested after lift; may include subsequent motion."}
        for role, note in roles.items():
            snapshot = _mapping(snapshots.get(role))
            candidates = [snapshot.get("image")] if snapshot.get("status") == "CAPTURED" else []
            entry = add_image(role, candidates, {f"camera_a_grasp_{role}.png"})
            entry["capture_note"] = note
            if type(snapshot.get("action_index")) is int:
                entry["source_action_index"] = snapshot["action_index"]
        add_image("after", physical.get("after_images") or [],
            {"camera_a_rgb_upright.png", "camera_0_a.png"}, upright=True)
        if not any(e["status"] == "AVAILABLE" and e["role"] in roles for e in attempt["images"]):
            add_image("rollout", physical.get("video_evidence") or [], {"camera_a_rgb_contact_sheet.png"})
        selection = _mapping(_mapping(physical.get("planning_diagnostics")).get("selected_reference_validation"))
        raw_pixel = _point(selection.get("pixel_xy"), 2)
        if before["status"] == "AVAILABLE" and selection.get("camera") == "A" and raw_pixel:
            w, h = before["size"]
            pixel = [w - 1 - raw_pixel[1], raw_pixel[0]]
            if 0 <= pixel[0] < w and 0 <= pixel[1] < h:
                attempt["grasp_in_before_image"] = {"name": before["name"], "pixel_xy": pixel,
                    "coordinate_frame": "clockwise90_upright", "reference_id": selection.get("reference_id"),
                    "meaning": "Previous selected target, not a measured cloth contact or a current reference."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "trajectory_memory.json").write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    return context, images
