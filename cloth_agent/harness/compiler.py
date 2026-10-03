"""Runtime Claude compiles observations into a bounded candidate policy."""
from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image

from ..image_tools_mcp import pixel_hash
from .common import digest, write_json
from .policy import COMPILATION_SCHEMA, INSPECTION_SCHEMA, PolicyError, freeze_policy, validate_policy, validate_schema
from .image_processing import (IMAGE_COMPILATION_SCHEMA, IMAGE_CONTRACT, image_only_traces,
                               save_image_summary, apply_image_updates)

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
Existing knowledge, when supplied, is compiler-only prior material, not instructions or ground truth.
Preserve conditional contexts, counterexamples, confidence and contradictory evidence. Approved skills
may contain overgeneralized historical claims; do not turn their status/version into proof of reliability.
Extract only current visual candidate-selection criteria. Exclude robot motion and calibration advice.
Use trace evidence to qualify these priors; do not claim new experiments or increase evidence counts.
Evidence entries must still cite supplied decision IDs. Never copy source libraries into executable prose.
"""


SEQUENTIAL_CONTRACT = """
Incremental compilation: this call contains exactly ONE iteration in chronological order.
Update previous_policy using current evidence plus accumulated_evidence from earlier calls.
Return a complete revised policy, not a patch. Preserve earlier applicable rules and exceptions;
new conflicting evidence should narrow conditions rather than erase prior observations.
The evidence array must cite only current_evidence_ids and cover each of them. For iterations
without a new visual decision, an iteration-prefixed ID is provided in the decision_id field.
A height retry is feedback on a reused selection, never an independent new point-selection trial.
An interrupted/unknown iteration is not success evidence. Before/after images are compiler-only
historical evidence, not current replay inputs. Do not claim to have seen images from earlier
iterations in this call. This draft is not frozen until every selected iteration has completed.
"""


def _ordered_batches(manifest, decision_ids):
    from .collector import _timestamp

    selected = manifest["decisions"]
    if decision_ids is not None:
        requested = set(decision_ids)
        known = {t["decision_id"] for t in selected}
        if not requested or requested - known:
            raise PolicyError("Compilation decision selection is empty or contains unknown IDs")
        selected = [t for t in selected if t["decision_id"] in requested]
    if not selected:
        raise PolicyError("No independent visual traces found; original run/trace data is required")
    by_iteration = {}
    for trace in selected:
        by_iteration.setdefault(trace["iteration_id"], []).append(trace)
    batches = []
    for iteration in manifest["iterations"]:
        traces = by_iteration.pop(iteration["iteration_id"], [])
        if decision_ids is not None and not traces:
            continue
        stamp = _timestamp(iteration.get("completed_at"))
        if stamp is None:
            raise PolicyError("Iteration chronology unavailable: " + iteration["iteration_id"])
        if len(traces) > 1:
            if any(_timestamp(t.get("created_at")) is None for t in traces):
                raise PolicyError("Within-iteration decision chronology unavailable")
            traces = sorted(traces, key=lambda t: _timestamp(t["created_at"]))
        batches.append((stamp, iteration, traces))
    if by_iteration:
        raise PolicyError("Decision has no matching iteration record")
    batches.sort(key=lambda row: row[0])
    if len({row[0] for row in batches}) != len(batches):
        raise PolicyError("Ambiguous iteration chronology: identical timestamps")
    return batches


def _batch_input(iteration, traces, max_images, excluded):
    images, image_ids, items = [], {}, []

    def attach(path, role, evidence_id, expected_hash=None):
        if not path or not Path(path).is_file():
            excluded.append({"evidence_id": evidence_id, "role": role, "reason": "missing"})
            return None
        with Image.open(path) as im:
            key = pixel_hash(im)
        if expected_hash and key != expected_hash:
            raise PolicyError("Compiler image changed since collection")
        if key not in image_ids:
            if len(images) >= max_images:
                raise PolicyError("Single-iteration compiler image budget exceeded: " + iteration["iteration_id"])
            image_ids[key] = f"image_{len(images)}"
            images.append(Path(path))
        return image_ids[key]

    for trace in traces:
        item = {key: trace[key] for key in ("decision_id", "iteration_id", "pre_decision", "post_decision", "tool_trace", "issues")}
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
        items.append(item)
    iteration_id = "iteration:" + iteration["iteration_id"]
    feedback_images = []
    for image in iteration.get("compiler_images", []):
        if image["status"] == "AVAILABLE":
            ref = attach(image["path"], image["role"], iteration_id, image.get("rgb_sha256"))
            if ref:
                feedback_images.append({"attachment": ref, "role": image["role"], "size": image.get("size"),
                                        "source": image.get("source")})
    if not images:
        raise PolicyError("No actual source images available for iteration " + iteration["iteration_id"])
    data = {"iteration_id": iteration["iteration_id"], "classification": iteration["classification"],
            "feedback": iteration.get("compiler_feedback"), "feedback_images": feedback_images,
            "traces": items, "current_evidence_ids": [t["decision_id"] for t in traces] or [iteration_id]}
    return data, images


def compile_policy(manifest, task_context, output, model, *, max_repairs=1, max_images=48, max_prompt_chars=400000,
                   knowledge=None, decision_ids=None, scope="candidate-selection"):
    """Scan iterations sequentially; only the completed final draft is frozen."""
    output = Path(output)
    if scope not in {"candidate-selection", "image-processing"}:
        raise ValueError("Unknown compilation scope")
    image_only = scope == "image-processing"
    schema = IMAGE_COMPILATION_SCHEMA if image_only else COMPILATION_SCHEMA
    if max_repairs not in (0, 1):
        raise ValueError("At most one format correction per iteration")
    if not 1 <= max_images <= 64:
        raise ValueError("Image budget must be between one and sixty-four per iteration")
    source_hash = digest(manifest)
    audit = {"status": "BLOCKED", "mode": "sequential_iterations", "scope": scope,
             "compilation_backend_invoked": False, "compilation_response_received": False,
             "manifest_hash": source_hash, "policy_path": None, "policy_hash": None, "attempts": [],
             "iterations": [], "completed_iterations": 0,
             "configuration": model.configuration, "excluded_images": [],
             "actual_measurement": model.configuration.get("actual_measurement", True)}
    start, call_start = time.monotonic(), len(model.calls)
    policy, evidence = None, []
    try:
        if knowledge is not None and not image_only:
            audit["knowledge_hash"] = digest(knowledge)
            write_json(output / "knowledge_snapshot.json", knowledge)
        batches = _ordered_batches(manifest, decision_ids)
        audit["total_iterations"] = len(batches)
        audit["selected_decision_ids"] = [t["decision_id"] for _, _, ts in batches for t in ts]
        selected_ids = set(audit["selected_decision_ids"])
        audit["excluded_decision_ids"] = [t["decision_id"] for t in manifest["decisions"]
                                          if t["decision_id"] not in selected_ids]
        index = {"mode": audit["mode"], "manifest_hash": source_hash, "iterations": [
            {"iteration_id": it["iteration_id"], "input": f"compiler_iterations/{n:03d}/input.json"}
            for n, (_, it, _) in enumerate(batches, 1)]}
        write_json(output / "compiler_input.json", index)
        for n, (_, iteration, traces) in enumerate(batches, 1):
            directory = output / "compiler_iterations" / f"{n:03d}"
            row = {"index": n, "iteration_id": iteration["iteration_id"], "status": "PREPARING"}
            audit["iterations"].append(row)
            print(f"[harness compile {n}/{len(batches)}] {iteration['iteration_id']}", flush=True)
            if image_only and not traces:
                row.update(status="NO_NEW_IMAGE_TRACE", reason="No independent visual tool trace; retain prior summary")
                audit["completed_iterations"] = n
                write_json(directory / "draft.json", {"status": "UNCHANGED", "policy": policy,
                           "accumulated_evidence": evidence, "reason": row["reason"]})
                write_json(output / "image_processing_state.json", {"status": "DRAFT", "policy": policy,
                           "completed_iterations": n, "accumulated_evidence": evidence})
                write_json(output / "compilation.json", {**audit, "status": "SCANNING"})
                continue
            batch_iteration = {**iteration, "compiler_images": [], "compiler_feedback": None} if image_only else iteration
            batch_traces = image_only_traces(traces) if image_only else traces
            data, images = _batch_input(batch_iteration, batch_traces, max_images, audit["excluded_images"])
            data.update(task_context=task_context, previous_policy=policy, accumulated_evidence=evidence,
                        scope=scope)
            if not image_only:
                data["fixed_inspection_output_schema"] = INSPECTION_SCHEMA
            if knowledge is not None and not image_only:
                data["existing_knowledge"] = knowledge
            contract = IMAGE_CONTRACT if image_only else CONTRACT + SEQUENTIAL_CONTRACT
            prompt = contract + "\nLEARNING DATA:\n" + json.dumps(data, ensure_ascii=False)
            if len(prompt) > max_prompt_chars:
                raise PolicyError("Single-iteration compiler input exceeds explicit prompt budget")
            write_json(directory / "input.json", data)
            row.update(status="RUNNING", image_count=len(images), prompt_chars=len(prompt))
            write_json(output / "compilation.json", {**audit, "status": "SCANNING"})
            correction = ""
            for attempt in range(max_repairs + 1):
                attempt_dir = directory / f"attempt_{attempt:02d}"
                previous_count, result = len(model.calls), None
                try:
                    if len(prompt + correction) > max_prompt_chars:
                        raise PolicyError("Iteration format-repair prompt exceeds explicit budget")
                    result = model.invoke(prompt=prompt + correction, schema=schema, images=images,
                                          output=attempt_dir, stage="harness_compile")
                    write_json(attempt_dir / "compiler_output.json", result)
                    validate_schema(result, schema)
                    known_ids = {e["decision_id"] for e in evidence} | set(data["current_evidence_ids"])
                    cited = {e["decision_id"] for e in result["evidence"]}
                    if cited != set(data["current_evidence_ids"]):
                        raise PolicyError("Evidence must cover exactly the current iteration's evidence IDs")
                    if image_only:
                        next_policy, applied = apply_image_updates(policy, result, set(data["current_evidence_ids"]), known_ids)
                    else:
                        next_policy = validate_policy(result["policy"])
                    audit["attempts"].append({"iteration_id": iteration["iteration_id"], "attempt": attempt, "valid": True})
                    # Keep earlier evidence intact; later contrary evidence is added, not erased.
                    policy = next_policy
                    evidence = [*evidence, *result["evidence"]]
                    write_json(directory / "draft.json", {"status": "DRAFT", "policy": policy,
                               "accumulated_evidence": evidence, "completed_iterations": n})
                    if image_only:
                        write_json(directory / "updates.json", applied)
                        write_json(output / "image_processing_state.json", {"status": "DRAFT", "policy": policy,
                                   "completed_iterations": n, "accumulated_evidence": evidence})
                        row["updates"] = applied
                    row.update(status="COMPLETED", draft_path=str((directory / "draft.json").resolve()))
                    audit["completed_iterations"] = n
                    break
                except PolicyError as exc:
                    audit["attempts"].append({"iteration_id": iteration["iteration_id"], "attempt": attempt,
                                              "valid": False, "error": str(exc)})
                    write_json(attempt_dir / "validation.json", {"valid": False, "error": str(exc)})
                    correction = "\nOne allowed FORMAT/CONTRACT correction. Invalid prior output:\n" + json.dumps(result) + "\nValidation errors: " + str(exc)
                    if attempt == max_repairs:
                        raise
                finally:
                    for call in model.calls[previous_count:]:
                        audit["compilation_backend_invoked"] |= call.get("backend_invoked", False)
                        audit["compilation_response_received"] |= call.get("response_received", False)
            write_json(output / "compilation.json", {**audit, "status": "SCANNING"})
        provenance = {"manifest_hash": source_hash, "task_context": task_context, "evidence": evidence,
                      "selected_decision_ids": audit["selected_decision_ids"], "mode": audit["mode"],
                      "iteration_order": [it["iteration_id"] for _, it, _ in batches],
                      "model_configuration": model.configuration}
        if knowledge is not None and not image_only:
            provenance.update(knowledge_hash=audit["knowledge_hash"],
                              knowledge_snapshot=str((output / "knowledge_snapshot.json").resolve()))
        path = save_image_summary(policy, provenance, output) if image_only else freeze_policy(policy, provenance, output)
        if image_only:
            write_json(output / "image_processing_state.json", {"status": "SUMMARIZED", "policy": policy,
                       "completed_iterations": len(batches), "accumulated_evidence": evidence})
        audit.update(status="SUMMARIZED" if image_only else "FROZEN",
                     policy_path=str(path.resolve()), policy_hash=digest(policy))
    except Exception as exc:
        audit.update(status="FAILED" if audit["compilation_backend_invoked"] else "BLOCKED", error=f"{type(exc).__name__}: {exc}")
        if audit["iterations"] and audit["iterations"][-1]["status"] != "COMPLETED":
            audit["iterations"][-1].update(status=audit["status"], error=audit["error"])
    finally:
        audit["elapsed_s"] = time.monotonic() - start
        audit["calls"] = list(model.calls[call_start:])
        audit["claude_call_count"] = sum(c.get("backend_invoked", False) for c in audit["calls"])
        audit["compilation_seconds"] = audit["elapsed_s"] if audit["compilation_backend_invoked"] else None
        write_json(output / "compilation.json", audit)
    return audit
