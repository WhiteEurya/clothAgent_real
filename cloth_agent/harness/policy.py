"""Small acyclic policy language; semantic judgments remain Claude's job."""
from __future__ import annotations

import copy
import re
import uuid
from pathlib import Path

from jsonschema import Draft202012Validator

from .common import canonical, digest, now, read_json, write_json


def obj(properties):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


TEXT = {"type": "string", "minLength": 1, "maxLength": 2000}
NAME = {"type": "string", "pattern": "^[a-z][a-z_]{0,47}$"}
INSPECTION_SCHEMA = obj({
    "status": {"enum": ["CONTINUE", "READY", "NEEDS_LEARNING"]},
    "observation_id": {"type": "string"},
    "applicable": {"type": "boolean"},
    "candidate_id": {"anyOf": [{"type": "null"}, {"type": "string", "pattern": "^R[0-9]+$"}]},
    "roi": {"anyOf": [{"type": "null"}, {"type": "array", "minItems": 4, "maxItems": 4,
                                    "items": {"type": "number", "minimum": 0, "maximum": 1}}]},
    "rotation_clockwise": {"enum": [0, 90, 180, 270]},
    "scale": {"type": "number", "minimum": 0.25, "maximum": 4},
    "needs_reference": {"type": "boolean"},
    "movable_region": TEXT, "preserved_region": TEXT, "evidence": TEXT,
})
WHEN = {"anyOf": [{"type": "null"}, obj({
    "step": NAME, "field": {"enum": ["needs_reference", "status"]},
    "equals": {"enum": [True, False, "CONTINUE", "READY"]},
})]}


def step_schema(op, props):
    return obj({"id": NAME, "op": {"const": op}, "when": WHEN, **props})


POLICY_SCHEMA = obj({
    "schema_version": {"const": 1}, "policy_id": NAME,
    "applicability": TEXT,
    "required_inputs": {"type": "array", "uniqueItems": True, "minItems": 4,
                        "items": {"enum": ["observation", "fold_goal", "candidate_registry", "observation_id", "reference", "hint"]}},
    "budget": obj({"max_claude_calls": {"type": "integer", "minimum": 1, "maximum": 6},
                   "max_host_image_ops": {"type": "integer", "minimum": 0, "maximum": 24},
                   "max_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                   "max_images_per_call": {"type": "integer", "minimum": 1, "maximum": 8}}),
    "steps": {"type": "array", "minItems": 2, "maxItems": 16, "items": {"oneOf": [
        step_schema("inspect", {"views": {"type": "array", "minItems": 1, "maxItems": 4, "uniqueItems": True, "items": NAME},
                                "context": {"type": "array", "maxItems": 6, "uniqueItems": True, "items": NAME},
                                "instruction": TEXT, "output_schema": {"const": INSPECTION_SCHEMA}}),
        step_schema("prepare_views", {"source": NAME, "roi_from": NAME}),
        step_schema("return_decision", {"from": NAME}),
        step_schema("needs_learning", {"reason": TEXT}),
    ]}},
})
COMPILATION_SCHEMA = obj({"policy": POLICY_SCHEMA, "evidence": {"type": "array", "minItems": 1, "items": obj({
    "decision_id": {"type": "string"}, "observed": TEXT, "inferred": TEXT, "unverified": TEXT,
})}})


class PolicyError(ValueError):
    pass


def validate_schema(value, schema):
    # JSON Schema numbers can accept NaN in Python; canonical rejects it first.
    try:
        canonical(value)
    except (ValueError, TypeError) as exc:
        raise PolicyError(f"Non-finite/non-JSON value: {exc}") from exc
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: str(e.path))
    if errors:
        raise PolicyError("; ".join(f"{list(e.path)}: {e.message}" for e in errors[:6]))


def validate_policy(policy):
    validate_schema(policy, POLICY_SCHEMA)
    mandatory = {"observation", "fold_goal", "candidate_registry", "observation_id"}
    if not mandatory <= set(policy["required_inputs"]):
        raise PolicyError("Missing mandatory inputs")
    # Instructions cannot carry historical IDs, coordinates, run names or paths.
    # Numeric budgets/schema constraints are host language, not action constants.
    texts = [policy["policy_id"], policy["applicability"]]
    types = {"observation": "views", "reference": "views", "hint": "views"}
    inspected = 0
    judgment_views = {}
    for step in policy["steps"]:
        name, op, when = step["id"], step["op"], step["when"]
        if name in types or name in mandatory:
            raise PolicyError(f"Duplicate/reserved variable: {name}")
        if when:
            if types.get(when["step"]) != "judgment":
                raise PolicyError("Conditional must refer to an earlier inspection")
            if (when["field"] == "needs_reference") != isinstance(when["equals"], bool):
                raise PolicyError("Conditional value has wrong type")
        if op == "inspect":
            if any(types.get(v) != "views" for v in step["views"]):
                raise PolicyError("View reference must be an earlier prepared view or declared image input")
            if any(types.get(v) != "judgment" for v in step["context"]):
                raise PolicyError("Context reference must be an earlier inspection")
            types[name] = "judgment"
            judgment_views[name] = set(step["views"])
            texts.append(step["instruction"])
            inspected += 1
        elif op == "prepare_views":
            if types.get(step["source"]) != "views" or types.get(step["roi_from"]) != "judgment":
                raise PolicyError("Prepare needs a view binding and an earlier dynamic ROI")
            if step["source"] in {"reference", "hint"}:
                raise PolicyError("Only the current RGB/overlay pair can be prepared")
            if judgment_views[step["roi_from"]] != {step["source"]}:
                raise PolicyError("ROI judgment must inspect exactly its source pair; mixed coordinate frames are ambiguous")
            types[name] = "views"
        elif op == "return_decision":
            if types.get(step["from"]) != "judgment":
                raise PolicyError("Return must reference an earlier inspection")
            types[name] = "terminal"
        else:
            texts.append(step["reason"])
            types[name] = "terminal"
    if not inspected or not any(s["op"] == "return_decision" for s in policy["steps"]):
        raise PolicyError("No visual inspection/return path")
    if policy["steps"][-1]["op"] != "needs_learning" or policy["steps"][-1]["when"] is not None:
        raise PolicyError("Last step must be unconditional NEEDS_LEARNING")
    for text in texts:
        if re.search(r"[0-9/]|https?:|base64|fold_", text, re.I) or chr(92) in text:
            raise PolicyError("Executable prose must not contain historical IDs, numeric constants, paths or encoded data")
    return copy.deepcopy(policy)


def freeze_policy(policy, provenance, output):
    policy = validate_policy(policy)
    version = now().replace(":", "").replace("+", "_") + "_" + uuid.uuid4().hex[:8]
    directory = Path(output) / "policies" / version
    directory.mkdir(parents=True, exist_ok=False)
    frozen = {"schema_version": 1, "version": version, "policy_hash": digest(policy),
              "policy": policy, "offline_candidate_only": True,
              "reliability": "UNVALIDATED", "created_at": now()}
    write_json(directory / "policy.json", frozen, exclusive=True)
    write_json(directory / "provenance.json", provenance, exclusive=True)
    write_json(directory / "validation.json", {"valid": True, "policy_hash": frozen["policy_hash"],
               "meaning": "Host contract validation, not semantic or physical reliability."}, exclusive=True)
    return directory / "policy.json"


def load_policy(path):
    frozen = read_json(path)
    policy = validate_policy(frozen["policy"])
    if frozen.get("schema_version") != 1 or frozen.get("policy_hash") != digest(policy):
        raise PolicyError("Frozen policy hash/version mismatch")
    if not frozen.get("version") or frozen.get("offline_candidate_only") is not True:
        raise PolicyError("Not an offline frozen policy")
    return frozen
