"""Host interpreter for frozen visual policies; receives no historical answer."""
from __future__ import annotations

import copy
import json
import math
import shutil
import time
from pathlib import Path

from PIL import Image

from ..image_tools_mcp import IDENTITY, ImageTools, compose, pixel_hash, transform_point
from .common import digest, write_json
from .policy import INSPECTION_SCHEMA, PolicyError, validate_policy, validate_schema


class NeedsLearning(ValueError):
    pass


def inverse(matrix):
    a, b, c, d, e, f = matrix
    determinant = a * e - b * d
    if not math.isfinite(determinant) or abs(determinant) < 1e-12:
        raise NeedsLearning("INVALID_TRANSFORM")
    return [e / determinant, -b / determinant, (b * f - e * c) / determinant,
            -d / determinant, a / determinant, (d * c - a * f) / determinant]


def _coordinates(point, size):
    return (isinstance(point, list) and len(point) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in point)
            and 0 <= point[0] < size[0] and 0 <= point[1] < size[1])


def validate_observation(before):
    required = {"observation_id", "fold_goal", "images", "candidate_registry"}
    if not required <= before.keys() or not before["observation_id"] or not before["fold_goal"]:
        raise NeedsLearning("MISSING_OBSERVATION_OR_GOAL")
    registry = before["candidate_registry"]
    if not isinstance(registry, dict) or not registry.get("candidates"):
        raise NeedsLearning("MISSING_CANDIDATE_REGISTRY")
    if registry.get("observation_id") != before["observation_id"]:
        raise NeedsLearning("REGISTRY_OBSERVATION_MISMATCH")
    if registry.get("binding") != "RAW_RGB_HASH_VERIFIED":
        raise NeedsLearning("REGISTRY_IDENTITY_UNVERIFIED")
    pair = {}
    for role in ("clean", "overlay"):
        matches = [i for i in before["images"] if i.get("role") == role and i.get("status") == "AVAILABLE"]
        if len(matches) != 1:
            raise NeedsLearning(f"MISSING_OR_AMBIGUOUS_{role.upper()}")
        image = matches[0]
        path = Path(image["path"])
        if not path.is_file():
            raise NeedsLearning("MISSING_IMAGE")
        with Image.open(path) as im:
            if list(im.size) != image["size"] or pixel_hash(im) != image["rgb_sha256"]:
                raise NeedsLearning("OBSERVATION_IMAGE_CHANGED")
        pair[role] = image
    if pair["clean"]["size"] != pair["overlay"]["size"]:
        raise NeedsLearning("RGB_OVERLAY_SIZE_MISMATCH")
    size = registry.get("raw_size")
    matrix = registry.get("to_raw")
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise NeedsLearning("INVALID_RAW_IMAGE_SIZE")
    if (not isinstance(matrix, list) or len(matrix) != 6
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in matrix)):
        raise NeedsLearning("INVALID_REGISTRY_TRANSFORM")
    inverse(matrix)
    seen = set()
    for row in registry["candidates"]:
        cid = row.get("candidate_id")
        if not isinstance(cid, str) or not cid.startswith("R") or not cid[1:].isdigit() or cid in seen:
            raise NeedsLearning("INVALID_CANDIDATE_ID")
        seen.add(cid)
        if row.get("camera") != "A" or not _coordinates(row.get("pixel_xy"), pair["clean"]["size"]):
            raise NeedsLearning("INVALID_CANDIDATE_COORDINATES")
        if not _coordinates(row.get("raw_pixel_xy"), registry["raw_size"]):
            raise NeedsLearning("INVALID_RAW_CANDIDATE_COORDINATES")
        if any(abs(a - b) > 1e-6 for a, b in zip(transform_point(registry["to_raw"], row["pixel_xy"]), row["raw_pixel_xy"])):
            raise NeedsLearning("CANDIDATE_FRAME_MISMATCH")
    return pair


def prepare_views(pair, judgment, directory, *, budget_remaining):
    """Identical crop/rotate/resize using the existing pixel-center affine tools."""
    if budget_remaining < 6:
        raise NeedsLearning("HOST_IMAGE_BUDGET_EXHAUSTED")
    roi = judgment.get("roi")
    if (not isinstance(roi, list) or len(roi) != 4 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in roi)
            or not 0 <= roi[0] < roi[2] <= 1 or not 0 <= roi[1] < roi[3] <= 1):
        raise NeedsLearning("INVALID_DYNAMIC_ROI")
    angle, scale = judgment["rotation_clockwise"], judgment["scale"]
    if angle not in {0, 90, 180, 270} or type(scale) not in (int, float) or not math.isfinite(scale) or not .25 <= scale <= 4:
        raise NeedsLearning("INVALID_DYNAMIC_TRANSFORM")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for index, role in enumerate(("clean", "overlay")):
        shutil.copyfile(pair[role]["path"], directory / f"image_{index}.png")
    tools = ImageTools(directory, 2)
    width, height = pair["clean"]["size"]
    box = [math.floor(roi[0] * width), math.floor(roi[1] * height),
           math.ceil(roi[2] * width), math.ceil(roi[3] * height)]
    result, lineage, operations = {}, [], 0
    for index, role in enumerate(("clean", "overlay")):
        current = f"image_{index}"
        for operation, arguments in (
            ("crop_image", {"box": box}), ("rotate_image", {"degrees_clockwise": angle}),
            ("resize_image", {"scale": scale}),
        ):
            view = tools.call(operation, {"image_id": current, **arguments})
            operations += 1
            if not isinstance(view, dict) or "image_id" not in view:
                raise NeedsLearning("HOST_IMAGE_TRANSFORM_FAILED")
            lineage.append({k: view.get(k) for k in ("image_id", "parent_image_id", "operation", "arguments", "size", "to_parent", "to_original")})
            current = view["image_id"]
        result[role] = {"path": view["path"], "size": view["size"], "rgb_sha256": view["rgb_sha256"],
                        "to_original": compose(pair[role].get("to_original", IDENTITY), view["to_original"])}
    if result["clean"]["to_original"] != result["overlay"]["to_original"]:
        raise NeedsLearning("PAIR_TRANSFORM_MISMATCH")
    write_json(directory / "lineage.json", {"pair": result, "lineage": lineage, "host_image_ops": operations})
    return result, lineage, operations


def _candidate_views(registry, bindings):
    result = []
    for candidate in registry["candidates"]:
        positions = {}
        for name, pair in bindings.items():
            if not isinstance(pair, dict) or "clean" not in pair:
                continue
            xy = transform_point(inverse(pair["clean"].get("to_original", IDENTITY)), candidate["pixel_xy"])
            if _coordinates(xy, pair["clean"]["size"]):
                positions[name] = xy
        if positions:
            result.append({"candidate_id": candidate["candidate_id"], "camera": "A", "view_pixels": positions})
    return result


def execute_policy(frozen, before, model, output):
    """The API intentionally has no argument for trace, post_decision or provenance."""
    start = time.monotonic()
    policy = validate_policy(copy.deepcopy(frozen["policy"]))
    expected_hash = digest(policy)
    if frozen["policy_hash"] != expected_hash:
        raise PolicyError("Frozen policy hash mismatch")
    before = copy.deepcopy(before)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "NEEDS_LEARNING", "reason": "NO_TERMINAL_DECISION",
              "observation_id": before.get("observation_id"), "candidate_id": None,
              "candidate_legal": None, "raw_pixel_xy": None, "selected_reference": None,
              "policy_hash": expected_hash, "steps": [], "metrics": {
                  "claude_calls": 0, "inspection_attempts": 0,
                  "host_image_ops": 0, "visible_tool_round_trips": 0,
                  "visual_decision_seconds": None, "cache_replay": False}}
    call_start = len(model.calls)
    try:
        pair = validate_observation(before)
        for image in pair.values():
            image["to_original"] = IDENTITY[:]
        bindings = {"observation": pair}
        for role in ("reference", "hint"):
            items = [i for i in before["images"] if i.get("role") == role and i.get("status") == "AVAILABLE"]
            bindings[role] = items
            if role in policy["required_inputs"] and not items:
                raise NeedsLearning(f"REQUIRED_{role.upper()}_MISSING")
        judgments, judgment_views = {}, {}
        budget = policy["budget"]
        for step in policy["steps"]:
            if time.monotonic() - start >= budget["max_seconds"]:
                raise NeedsLearning("TIME_BUDGET_EXHAUSTED")
            row = {"id": step["id"], "op": step["op"], "status": "RUNNING"}
            result["steps"].append(row)
            when = step["when"]
            if when:
                if when["step"] not in judgments:
                    raise NeedsLearning("CONDITION_BINDING_UNAVAILABLE")
                if judgments[when["step"]][when["field"]] != when["equals"]:
                    row["status"] = "SKIPPED_CONDITION"
                    continue
            if step["op"] == "needs_learning":
                raise NeedsLearning(step["reason"])
            if step["op"] == "inspect":
                if result["metrics"]["inspection_attempts"] >= budget["max_claude_calls"]:
                    raise NeedsLearning("CLAUDE_CALL_BUDGET_EXHAUSTED")
                if any(name not in judgments for name in step["context"]):
                    raise NeedsLearning("JUDGMENT_BINDING_UNAVAILABLE")
                images, catalog, selected_bindings = [], [], {}
                for name in step["views"]:
                    view = bindings.get(name)
                    if not view:
                        raise NeedsLearning("IMAGE_BINDING_UNAVAILABLE")
                    selected_bindings[name] = view
                    records = list(view.values()) if isinstance(view, dict) else view
                    roles = ["clean", "overlay"] if isinstance(view, dict) else [name] * len(records)
                    for image, role in zip(records, roles):
                        with Image.open(image["path"]) as im:
                            if pixel_hash(im) != image["rgb_sha256"]:
                                raise NeedsLearning("OBSERVATION_IMAGE_CHANGED")
                        catalog.append({"image_id": f"image_{len(images)}", "view": name, "role": role, "size": image["size"]})
                        images.append(Path(image["path"]))
                if len(images) > budget["max_images_per_call"]:
                    raise NeedsLearning("IMAGE_CALL_BUDGET_EXHAUSTED")
                candidates = _candidate_views(before["candidate_registry"], selected_bindings)
                # Allowlist, not recursive filtering: no trace/history/paths reach model.
                payload = {"observation_id": before["observation_id"], "fold_goal": before["fold_goal"],
                           "applicability": policy["applicability"], "instruction": step["instruction"],
                           "images": catalog, "current_candidates": candidates,
                           "prior_policy_judgments": {name: judgments[name] for name in step["context"]}}
                prompt = ("Inspect only attached CURRENT inputs. Decide semantic applicability yourself. "
                          "Return NEEDS_LEARNING if unclear; this does not imply out-of-distribution. "
                          "Candidate IDs are local to this observation. ROI is normalized edges in the named view. "
                          "Do not invent coordinates or act on a robot.\n" + json.dumps(payload, ensure_ascii=False))
                result["metrics"]["inspection_attempts"] += 1
                judgment = model.invoke(prompt=prompt, schema=INSPECTION_SCHEMA, images=images,
                    output=output / "calls" / step["id"], stage="harness_replay",
                    timeout_s=max(0, budget["max_seconds"] - (time.monotonic() - start)))
                validate_schema(judgment, INSPECTION_SCHEMA)
                roi = judgment["roi"]
                if roi is not None and not (roi[0] < roi[2] and roi[1] < roi[3]):
                    raise NeedsLearning("INVALID_DYNAMIC_ROI")
                if judgment["observation_id"] != before["observation_id"]:
                    raise NeedsLearning("JUDGMENT_OBSERVATION_MISMATCH")
                if not judgment["applicable"] or judgment["status"] == "NEEDS_LEARNING":
                    raise NeedsLearning("VISUAL_INFORMATION_OR_APPLICABILITY_INSUFFICIENT: " + judgment["evidence"])
                if judgment["status"] == "READY":
                    if judgment["candidate_id"] not in {c["candidate_id"] for c in candidates}:
                        raise NeedsLearning("CANDIDATE_NOT_IN_CURRENT_VISIBLE_REGISTRY")
                elif judgment["candidate_id"] is not None:
                    raise NeedsLearning("UNEXPECTED_CANDIDATE_BEFORE_READY")
                judgments[step["id"]] = judgment
                judgment_views[step["id"]] = set(step["views"])
                row.update(status="COMPLETED", judgment=judgment)
            elif step["op"] == "prepare_views":
                if step["roi_from"] not in judgments or step["source"] not in bindings:
                    raise NeedsLearning("DYNAMIC_VIEW_BINDING_UNAVAILABLE")
                if judgment_views[step["roi_from"]] != {step["source"]}:
                    raise NeedsLearning("AMBIGUOUS_ROI_COORDINATE_FRAME")
                pair, lineage, count = prepare_views(bindings[step["source"]], judgments[step["roi_from"]],
                    output / "views" / step["id"], budget_remaining=budget["max_host_image_ops"] - result["metrics"]["host_image_ops"])
                result["metrics"]["host_image_ops"] += count
                bindings[step["id"]] = pair
                row.update(status="COMPLETED", lineage=lineage, pair=pair)
            else:
                judgment = judgments.get(step["from"])
                if not judgment:
                    raise NeedsLearning("RETURN_BINDING_UNAVAILABLE")
                if judgment["status"] != "READY":
                    row["status"] = "NO_READY_DECISION"
                    continue
                candidate = next((c for c in before["candidate_registry"]["candidates"] if c["candidate_id"] == judgment["candidate_id"]), None)
                if not candidate:
                    raise NeedsLearning("CANDIDATE_NOT_IN_REGISTRY")
                if time.monotonic() - start >= budget["max_seconds"]:
                    raise NeedsLearning("TIME_BUDGET_EXHAUSTED")
                result.update(status="READY", reason=judgment["evidence"], candidate_id=candidate["candidate_id"],
                              raw_pixel_xy=candidate["raw_pixel_xy"], candidate_legal=True,
                              selected_reference={"camera": candidate["camera"], "reference_id": candidate["candidate_id"], "reason": judgment["evidence"]})
                row["status"] = "RETURNED"
                break
    except (NeedsLearning, PolicyError, FileNotFoundError) as exc:
        result.update(status="NEEDS_LEARNING", reason=str(exc))
    except Exception as exc:
        result.update(status="ERROR", reason=f"{type(exc).__name__}: {exc}")
    finally:
        calls = model.calls[call_start:]
        result["metrics"]["backend_invocations"] = sum(c.get("backend_invoked", False) for c in calls)
        result["metrics"]["claude_calls"] = result["metrics"]["backend_invocations"]
        result["metrics"]["model_responses"] = sum(c.get("response_received", False) for c in calls)
        result["metrics"]["visible_tool_round_trips"] = (sum(c["tool_round_trips"] for c in calls)
            if all(c.get("tool_round_trips") is not None for c in calls) else None)
        result["metrics"]["visual_decision_seconds"] = time.monotonic() - start
        result["metrics"]["boundary"] = "visual policy including host transforms and Claude transport, excluding motion and execution"
        result["metrics"]["actual_replay"] = (model.configuration.get("actual_measurement", True)
            and result["metrics"]["model_responses"] > 0)
        if digest(policy) != expected_hash:
            result.update(status="ERROR", reason="POLICY_CHANGED_DURING_EXECUTION", candidate_id=None)
        write_json(output / "result.json", result)
    return result
