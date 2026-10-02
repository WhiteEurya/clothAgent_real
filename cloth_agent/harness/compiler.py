"""Runtime Claude compiles observations into a bounded candidate policy."""
from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image

from ..image_tools_mcp import pixel_hash
from .common import digest, write_json
from .policy import COMPILATION_SCHEMA, INSPECTION_SCHEMA, PolicyError, freeze_policy, validate_policy, validate_schema

CONTRACT = """Compile a reusable, bounded visual selection program from completed learning traces.
Do NOT redo the historical folding task or output a historical answer as the program.
Input at replay is a NEW current observation, fold goal, current Rxxx registry and available references.
Only candidate selection is in scope. No depth, Z, target motion, grounding, preflight, IK or robot actions.
Separate observed tool use, hypotheses of usefulness and unverified necessity; this is not an ablation.
Keep successful, failed and UNKNOWN feedback. Historical selections are not ground truth.
Choose your own step order, conditions and visual criteria from the traces. No prescribed two/three-step recipe.
The language is an ordered acyclic list. inspect calls Claude once with attached views and the fixed output schema;
prepare_views applies a preceding inspection's dynamic ROI, rotation and scale to a clean/overlay pair on host;
return_decision accepts READY only from a valid current-registry judgment; needs_learning terminates.
Variables bind to step IDs. Built-in views: observation (clean plus overlay), reference and hint (optional).
inspect.context lists earlier judgment IDs. inspect.views lists view bindings. prepare_views.source is a pair binding;
prepare_views.roi_from names a judgment made on THAT pair; its ROI uses normalized edges, not pixel centers.
prepare_views must never use an old ROI. A condition is null or {step, field, equals} on a prior judgment.
Skipped/missing bindings safely fall back; missing optional references never trigger a new tool search.
Instructions describe semantics, applicability, input needs, when sufficient, and when to return NEEDS_LEARNING.
The fixed judgment schema explicitly asks Claude for movable/preserved regions and evidence.
The policy must end in unconditional needs_learning and use at least one inspect and return_decision.
Policy prose contains no digits, paths, encoded data, historic IDs, numeric crop boxes, pixels or distances.
All geometry comes from current model outputs/registry. Numeric budgets and the supplied output schema are allowed.
No history-lookup table, image fingerprint branching, remembered answer or covert encoding in instructions.
Evidence citations belong ONLY in the separate evidence array; they are not sent to replay.
Host validates structure, references and coordinates; Host cannot certify semantic applicability or reliability.
"""


def compile_policy(manifest, task_context, output, model, *, max_repairs=1, max_images=48, max_prompt_chars=400000):
    output = Path(output)
    if max_repairs not in (0, 1):
        raise ValueError("At most one format correction")
    if not 1 <= max_images <= 64:
        raise ValueError("Image budget must be between one and sixty-four")
    source_hash = digest(manifest)
    audit = {"status": "BLOCKED", "compilation_backend_invoked": False, "compilation_response_received": False,
             "manifest_hash": source_hash, "policy_path": None, "policy_hash": None, "attempts": [],
             "configuration": model.configuration, "excluded_images": [],
             "actual_measurement": model.configuration.get("actual_measurement", True)}
    start = time.monotonic()
    call_start = len(model.calls)
    images, image_ids, traces = [], {}, []

    def attach(path, role, decision_id, expected_hash=None):
        if not path or not Path(path).is_file():
            audit["excluded_images"].append({"decision_id": decision_id, "role": role, "reason": "missing"})
            return None
        with Image.open(path) as im:
            key = pixel_hash(im)
        if expected_hash and key != expected_hash:
            raise PolicyError("Compiler image changed since collection")
        if key not in image_ids:
            if len(images) >= max_images:
                raise PolicyError("Compiler image budget exceeded; increase --max-compile-images or explicitly use a smaller input dataset")
            image_ids[key] = f"image_{len(images)}"
            images.append(Path(path))
        return image_ids[key]

    try:
        if not manifest["decisions"]:
            raise PolicyError("No independent visual traces found; original run/trace data is required")
        for trace in manifest["decisions"]:
            item = {key: trace[key] for key in ("decision_id", "iteration_id", "pre_decision", "post_decision", "tool_trace", "issues")}
            # Paths retain provenance in the saved manifest, but are not treated as image inputs.
            item = json.loads(json.dumps(item))
            attached = []
            for image in item["pre_decision"]["images"]:
                if image["status"] == "AVAILABLE":
                    ref = attach(image["path"], image["role"], trace["decision_id"], image.get("rgb_sha256"))
                    attached.append({"attachment": ref, "role": image["role"], "size": image.get("size")})
            for view in item["tool_trace"]["lineage"]:
                if view.get("parent_image_id") is not None:
                    ref = attach(view.get("path"), "historical_derived_view", trace["decision_id"], view.get("rgb_sha256"))
                    if ref:
                        attached.append({"attachment": ref, "role": "historical_derived_view", "lineage_id": view["image_id"]})
            item["attached_images"] = attached
            traces.append(item)
        if not images:
            raise PolicyError("No actual source images available for compilation")
        data = {"task_context": task_context, "traces": traces, "manifest_issues": manifest["issues"],
                "fixed_inspection_output_schema": INSPECTION_SCHEMA}
        prompt = CONTRACT + "\nLEARNING DATA:\n" + json.dumps(data, ensure_ascii=False)
        if len(prompt) > max_prompt_chars:
            raise PolicyError("Compiler input exceeds explicit prompt budget; no silent trace truncation")
        write_json(output / "compiler_input.json", data)
        correction = ""
        for attempt in range(max_repairs + 1):
            directory = output / "compiler_calls" / f"attempt_{attempt:02d}"
            previous_count = len(model.calls)
            result = None
            try:
                result = model.invoke(prompt=prompt + correction, schema=COMPILATION_SCHEMA, images=images,
                                      output=directory, stage="harness_compile")
                write_json(directory / "compiler_output.json", result)
                validate_schema(result, COMPILATION_SCHEMA)
                policy = validate_policy(result["policy"])
                known = {t["decision_id"] for t in traces}
                if any(e["decision_id"] not in known for e in result["evidence"]):
                    raise PolicyError("Unknown evidence reference")
                audit["attempts"].append({"attempt": attempt, "valid": True})
                provenance = {"manifest_hash": source_hash, "task_context": task_context, "evidence": result["evidence"],
                              "compiler_call_directory": str(directory.resolve()), "model_configuration": model.configuration}
                path = freeze_policy(policy, provenance, output)
                audit.update(status="FROZEN", policy_path=str(path.resolve()), policy_hash=digest(policy))
                break
            except PolicyError as exc:
                audit["attempts"].append({"attempt": attempt, "valid": False, "error": str(exc)})
                write_json(directory / "validation.json", {"valid": False, "error": str(exc)})
                correction = "\nOne allowed FORMAT/CONTRACT correction. Invalid prior output:\n" + json.dumps(result) + "\nValidation errors: " + str(exc)
                if attempt == max_repairs:
                    raise
            finally:
                for call in model.calls[previous_count:]:
                    audit["compilation_backend_invoked"] |= call.get("backend_invoked", False)
                    audit["compilation_response_received"] |= call.get("response_received", False)
    except Exception as exc:
        audit.update(status="FAILED" if audit["compilation_backend_invoked"] else "BLOCKED", error=f"{type(exc).__name__}: {exc}")
    finally:
        audit["elapsed_s"] = time.monotonic() - start
        audit["calls"] = list(model.calls[call_start:])
        audit["claude_call_count"] = sum(c.get("backend_invoked", False) for c in audit["calls"])
        audit["compilation_seconds"] = (audit["elapsed_s"] if audit["compilation_backend_invoked"] else None)
        write_json(output / "compilation.json", audit)
    return audit
