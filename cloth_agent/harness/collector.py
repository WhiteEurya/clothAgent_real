"""Collect existing fold records without deriving policy or inventing observations."""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

from PIL import Image

from ..image_tools_mcp import IDENTITY, VERIFIED_DELIVERIES, image_content_summary, pixel_hash
from ..run_storage import storage_roots, validate_run_id
from .common import digest, now, read_json, write_json


ROLES = {
    "camera_a_rgb_upright.png": "clean", "camera_a_rxxx_overlay_upright.png": "overlay",
    "camera_0_a.png": "raw", "fold_reference_source.png": "reference",
    "fold_reference_target.png": "reference", "camera_a_flat_reference.png": "reference",
    "camera_a_molmo_frame_hint.png": "hint", "camera_a_molmo_hint_upright.png": "hint",
}


def _role(path):
    parts = Path(str(path)).parts
    if any("after" in p.lower() or p.lower() == "previous_attempt" for p in parts[:-1]):
        return "excluded_historical_or_post_action"
    return ROLES.get(Path(str(path)).name.lower(), "unclassified")


def _read(path, issues):
    try:
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError("Expected object")
        return value
    except (OSError, ValueError) as exc:
        issues.append({"code": "unreadable_json", "path": str(path), "detail": str(exc)})
        return {}


def _timestamp(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp.timestamp() if stamp.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def _belongs(data, run_id):
    if data.get("run_id") == run_id or Path(str(data.get("run_dir", ""))).name == run_id:
        return True
    # Copied process-review records often retain source paths but not run_id.
    for key in ("before_images", "after_images"):
        if any(run_id in Path(p).parts for p in data.get(key, []) if isinstance(p, str)):
            return True
    return False


def _resolve(raw, base, roots):
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [base / path]
    # Explicit relative suffix remapping handles run relocation, not basename guessing.
    for root in roots:
        if root.name in path.parts:
            index = path.parts.index(root.name)
            candidates.append(root.joinpath(*path.parts[index + 1:]))
    matches = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    return matches[0] if len(matches) == 1 else None


def collect_tool_trace(debug):
    """Classify evidence conservatively; a display transform is not an ablation."""
    views = {v["image_id"]: dict(v) for v in debug.get("views", []) if isinstance(v, dict) and "image_id" in v}
    calls, seen, delivered = [], set(), set()
    for event in debug.get("events", []):
        if not isinstance(event, dict):
            continue
        metadata = event.get("image_metadata") or {}
        image_info = event.get("image_content") or {}
        if image_info.get("identity_status") in VERIFIED_DELIVERIES:
            if metadata.get("image_id"):
                delivered.add(metadata["image_id"])
            for view_id, view in views.items():
                if event.get("arguments", {}).get("file_path") and event["arguments"]["file_path"] in {view.get("path"), view.get("remote_path")}:
                    delivered.add(view_id)
        if event.get("status") == "completed" and metadata.get("image_id") in views:
            view = views[metadata["image_id"]]
            if any(i.get("rgb_sha256") == view.get("rgb_sha256") for i in image_info.get("images", [])):
                delivered.add(metadata["image_id"])
        name = str(event.get("tool", "")).split("__")[-1]
        if name not in {"list_images", "view_image", "crop_image", "rotate_image", "resize_image", "map_point", "image_info", "Read"}:
            continue
        if event.get("kind") in {"tool_lifecycle", "read"} and event.get("status") not in {"completed", "failed"}:
            continue
        identity = event.get("event_id") or digest(event)
        if identity in seen:
            continue
        seen.add(identity)
        args, result = event.get("arguments") or {}, event.get("result") or {}
        view = views.get(args.get("image_id"), {})
        classification = "USAGE_UNKNOWN"
        if name in {"crop_image", "resize_image", "rotate_image"}:
            classification = "DISPLAY_TRANSFORM"
        elif name in {"view_image", "Read"}:
            classification = ("NEW_OBSERVATION_INPUT" if image_info.get("image_count", 0) > 0
                              else "OBSERVATION_REQUEST")
        elif (name == "map_point" and event.get("status") == "ok" and view
              and view.get("to_original") == IDENTITY and view.get("parent_image_id") is None
              and view.get("original_image_index") == result.get("original_image_index")
              and args.get("pixel_xy") is not None and args.get("pixel_xy") == result.get("pixel_xy")):
            classification = "CONFIRMED_IDENTITY_NO_OP"
        calls.append({"tool": name, "arguments": args, "result": result,
                      "status": event.get("status"), "duration_s": event.get("duration_s"),
                      "timestamp_ns": event.get("timestamp_ns"), "tool_use_id": event.get("tool_use_id"),
                      "classification": classification, "event_id": identity})
    lineage = []
    for identity, view in views.items():
        inspections = view.get("image_inspections", [])
        verified = identity in delivered or view.get("image_delivery_status") in VERIFIED_DELIVERIES or any(x.get("image_delivery_status") in VERIFIED_DELIVERIES
                                                for x in inspections if isinstance(x, dict))
        lineage.append({k: view.get(k) for k in ("image_id", "parent_image_id", "operation", "arguments", "size",
                        "original_image_index", "original_size", "to_parent", "to_original", "rgb_sha256", "path")} | {
            "model_input": "VERIFIED_TOOL_RETURN" if verified else "UNKNOWN",
            "note": "Observed image-content boundary; not proof of model understanding or necessity."})
    # Hook and MCP audit duplicates are evidence, not extra tool round trips.
    authoritative = [e for e in calls if e["tool_use_id"]]
    count = len({e["tool_use_id"] for e in authoritative}) if authoritative else None
    return {"calls": calls, "lineage": lineage, "visible_tool_round_trips": count,
            "audit_complete": debug.get("audit_complete", False)}


def _image(raw, role, base, roots, *, snapshot=None):
    path = snapshot if snapshot and snapshot.is_file() else _resolve(raw, base, roots)
    item = {"source": raw, "role": role, "path": str(path) if path else None, "status": "MISSING"}
    if path:
        try:
            with Image.open(path) as im:
                item.update(status="AVAILABLE", size=list(im.size), rgb_sha256=pixel_hash(im))
        except (OSError, ValueError) as exc:
            item.update(status="INVALID", error=str(exc))
    return item


def _stream_evidence(directory, debug, issues):
    """Add public tool inputs/results with receipt-time durations, never thinking."""
    stream = directory / "claude_events.jsonl"
    if not stream.is_file():
        return
    uses = {}
    known = {e.get("tool_use_id") for e in debug.get("events", []) if e.get("status") in {"completed", "failed"}}
    for line in stream.read_text().splitlines():
        try:
            row = json.loads(line)
            event = row.get("event", row)
            content = (event.get("message") or {}).get("content", [])
            elapsed = row.get("received_elapsed_s")
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") == "tool_use":
                    uses[block.get("id")] = (block, elapsed)
                elif block.get("type") == "tool_result":
                    identity = block.get("tool_use_id")
                    if identity not in uses or identity in known:
                        continue
                    use, started = uses[identity]
                    duration = elapsed - started if type(elapsed) in (int, float) and type(started) in (int, float) else None
                    summary = image_content_summary(block.get("content"))
                    debug.setdefault("events", []).append({"event_id": "stream_" + identity,
                        "kind": "tool_lifecycle", "tool_use_id": identity, "tool": use.get("name"),
                        "arguments": use.get("input", {}), "status": "failed" if block.get("is_error") else "completed",
                        "result": {"public_text": [b.get("text") for b in block.get("content", []) if isinstance(b, dict) and b.get("type") == "text"]},
                        "image_content": summary, "duration_s": duration,
                        "duration_boundary": "CLI tool-use to tool-result receipt; includes transport"})
                    known.add(identity)
        except (ValueError, TypeError, AttributeError) as exc:
            issues.append({"code": "unreadable_tool_stream_event", "path": str(stream), "detail": str(exc)})


def _registry(iter_dir, clean, roots, issues):
    """Follow the recorded mapping and bind immutable perception RGB by hash."""
    mappings = list(iter_dir.rglob("camera_A_upright_mapping.json"))
    mappings = [p for p in mappings if not any("after" in part.lower() for part in p.relative_to(iter_dir).parts)]
    registries = []
    for mapping_path in sorted(mappings):
        mapping = _read(mapping_path, issues)
        guide = _resolve(mapping.get("coordinate_guide"), mapping_path.parent, roots)
        raw = _resolve(mapping.get("raw_image"), mapping_path.parent, roots)
        # The original host saves perception batches outside iteration directories.
        # Recorded explicit links are valid; mutable workspace snapshots are not.
        if not raw or "workspace" in raw.parts:
            continue
        try:
            with Image.open(raw) as image:
                raw_size = list(image.size)
                current = image.convert("RGB").rotate(-90, expand=True)
                if pixel_hash(current) != clean.get("rgb_sha256"):
                    continue
        except OSError:
            continue
        if not guide or "workspace" in guide.parts:
            continue
        if (mapping.get("rotation") != "clockwise90"
                or mapping.get("raw_size_xy") != raw_size
                or mapping.get("upright_size_xy") != clean.get("size")):
            issues.append({"code": "invalid_registry_mapping", "path": str(mapping_path)})
            return None
        entries = _read(guide, issues).get("samples", [])
        if not isinstance(entries, list):
            issues.append({"code": "invalid_candidate_registry", "path": str(guide)})
            return None
        # Match the host's original upright overlay exclusions; no chosen-point filter.
        rows = []
        seen = set()
        for row in entries:
            if not isinstance(row, dict) or row.get("reference_source") == "fold_rgb_boundary_dense":
                continue
            cid, xy = row.get("reference_id"), row.get("pixel_xy")
            if not isinstance(cid, str) or not re.fullmatch(r"R[0-9]+", cid) or cid in seen:
                issues.append({"code": "invalid_candidate_registry", "path": str(guide)})
                return None
            if (not isinstance(xy, list) or len(xy) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy)
                    or not 0 <= xy[0] < raw_size[0] or not 0 <= xy[1] < raw_size[1]):
                issues.append({"code": "invalid_candidate_coordinates", "path": str(guide)})
                return None
            seen.add(cid)
            rows.append({"candidate_id": cid, "camera": "A", "raw_pixel_xy": xy,
                         "pixel_xy": [raw_size[1] - 1 - xy[1], xy[0]]})
        if rows:
            registries.append({"source": str(guide), "mapping_source": str(mapping_path),
                    "raw_image": str(raw), "raw_size": raw_size,
                    "coordinate_frame": "current_upright_pixel_centers",
                    "candidates": sorted(rows, key=lambda row: row["candidate_id"]),
                    "to_raw": [0, 1, 0, -1, 0, raw_size[1] - 1], "binding": "RAW_RGB_HASH_VERIFIED"})
    if registries:
        if len({digest({k: row[k] for k in ("raw_size", "candidates", "to_raw")}) for row in registries}) != 1:
            issues.append({"code": "conflicting_candidate_registries",
                           "sources": [row["source"] for row in registries]})
            return None
        return registries[0]
    issues.append({"code": "registry_missing_or_unbound", "path": str(iter_dir),
                   "detail": "Need a recorded mapping to an immutable guide whose raw RGB matches the input; mutable workspace is not evidence."})
    return None


def _decision(record, source, debug_dir, roots, run_id, segment, issues):
    visual = (record.get("planning_diagnostics") or {}).get("visual_plan_result") or {}
    debug = _read(debug_dir / "image_debug.json", issues) if debug_dir else {}
    if debug_dir:
        _stream_evidence(debug_dir, debug, issues)
    request = _read(debug_dir / "request.json", issues) if debug_dir and (debug_dir / "request.json").is_file() else {}
    images = []
    if debug:
        for view in debug.get("views", []):
            if view.get("parent_image_id") is not None:
                continue
            raw = view.get("source_local_path")
            index = view.get("original_image_index")
            if not raw and type(index) is int and index < len(request.get("image_paths", [])):
                raw = request["image_paths"][index]
            role = _role(raw)
            snapshot = debug_dir / "images" / (view["image_id"] + ".png")
            item = _image(raw, role, source.parent, roots, snapshot=snapshot)
            if item["status"] == "AVAILABLE" and view.get("rgb_sha256") and item["rgb_sha256"] != view["rgb_sha256"]:
                item.update(status="INPUT_HASH_MISMATCH", path=None)
            item["image_id"] = view["image_id"]
            images.append(item)
    else:
        for raw in record.get("before_images", []):
            item = _image(raw, _role(raw), source.parent, roots)
            # Run-wide workspace may have been overwritten by later iterations.
            if item["path"] and "workspace" in Path(item["path"]).parts:
                item.update(status="UNVERIFIED_MUTABLE_WORKSPACE", path=None)
            images.append(item)
    clean = next((v for v in images if v["role"] == "clean" and v["status"] == "AVAILABLE"), None)
    overlay = next((v for v in images if v["role"] == "overlay" and v["status"] == "AVAILABLE"), None)
    local_issues = []
    for item in images:
        if item["status"] != "AVAILABLE":
            local_issues.append({"code": "image_unavailable", "source": item["source"], "status": item["status"]})
    if not clean or not overlay:
        local_issues.append({"code": "missing_current_rgb_overlay"})
    if clean and overlay and clean["size"] != overlay["size"]:
        local_issues.append({"code": "rgb_overlay_size_mismatch"})
    registry = _registry(source.parent, clean, roots, local_issues) if clean else None
    goal = record.get("planned_step") or (record.get("supervisor_before") or {}).get("current_step")
    if not goal:
        local_issues.append({"code": "missing_fold_goal"})
    historical = (visual.get("decision") or {}) if not debug_dir else {}
    duration = visual.get("duration_s") if not debug_dir else None
    decision_start = visual.get("created_at") if not debug_dir else None
    if not debug_dir and _timestamp(decision_start) is not None and type(duration) in (int, float) and math.isfinite(duration) and duration >= 0:
        from datetime import timezone
        decision_start = datetime.fromtimestamp(_timestamp(decision_start) - duration, timezone.utc).isoformat()
    if debug_dir:
        terminal_path = next((p for p in (debug_dir / "claude_result.json", debug_dir / "claude_stdout.txt") if p.is_file()), debug_dir / "claude_stdout.txt")
        if terminal_path.is_file():
            from ..planner_backend import parse_claude_json
            try:
                historical = parse_claude_json(terminal_path.read_text())
            except ValueError:
                local_issues.append({"code": "invalid_historical_output"})
            except RuntimeError:
                local_issues.append({"code": "failed_historical_output"})
        else:
            local_issues.append({"code": "historical_result_missing", "path": str(terminal_path)})
        stream = debug_dir / "claude_events.jsonl"
        if stream.is_file():
            for line in stream.read_text().splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    local_issues.append({"code": "invalid_claude_stream_event", "path": str(stream)})
                    continue
                if not decision_start and _timestamp(event.get("received_at")) is not None:
                    decision_start = event["received_at"]
                    # Receipt minus elapsed time is the debug-session start.
                    elapsed = event.get("received_elapsed_s")
                    if type(elapsed) in (int, float) and math.isfinite(elapsed):
                        from datetime import timezone
                        decision_start = datetime.fromtimestamp(_timestamp(decision_start) - elapsed, timezone.utc).isoformat()
                    break
        # Exact application boundary: includes transport, excludes robot iteration.
        durations = [e.get("duration_s") for e in debug.get("progress", [])
                     if e.get("stage") == "call" and e.get("event") in {"completed", "failed"}]
        duration = durations[-1] if durations else None
    if _timestamp(decision_start) is None:
        local_issues.append({"code": "decision_chronology_unavailable"})
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        duration = None
    selection = historical.get("selected_reference") or {}
    observation_id = digest({"run": run_id, "segment": segment, "iteration": record.get("iteration"),
                             "image": clean.get("rgb_sha256") if clean else None})
    if registry:
        registry["observation_id"] = observation_id
    identity = digest({"record": digest(record), "call": str(debug_dir.relative_to(source.parent)) if debug_dir else visual.get("created_at")})[:24]
    tool_trace = collect_tool_trace(debug)
    for view in tool_trace["lineage"]:
        resolved = _resolve(view.get("path"), debug_dir or source.parent, roots)
        view["path"] = str(resolved) if resolved else None
        if view.get("parent_image_id") and not resolved:
            local_issues.append({"code": "derived_image_missing", "image_id": view["image_id"]})
    if not debug:
        local_issues.append({"code": "tool_trace_unavailable"})
    before = {"observation_id": observation_id, "fold_goal": goal, "images": images, "candidate_registry": registry}
    blocking = {"missing_current_rgb_overlay", "rgb_overlay_size_mismatch", "registry_missing_or_unbound",
                "invalid_registry_mapping", "conflicting_candidate_registries",
                "missing_fold_goal", "invalid_candidate_registry", "invalid_candidate_coordinates"}
    return {"run_id": run_id, "iteration_id": f"{segment}:{record.get('iteration')}", "decision_id": identity,
            "source_record": str(source), "debug_directory": str(debug_dir) if debug_dir else None,
            "created_at": decision_start, "pre_decision": before,
            "post_decision": {"candidate_id": selection.get("reference_id"), "reason": selection.get("reason"),
                              "visible_decision": historical, "evaluation": record.get("evaluation"),
                              "feedback_scope": "iteration outcome; linkage to this call is UNKNOWN unless final call; not proof of candidate correctness"},
            "tool_trace": tool_trace, "historical_metrics": {"visual_decision_seconds": duration,
                "boundary": "visual Claude invocation including transport, excluding motion and robot execution",
                "claude_calls": 1 if visual or debug else None,
                "host_image_ops": None,
                "recorded_image_edit_events": sum(c["classification"] == "DISPLAY_TRANSFORM" for c in tool_trace["calls"]) if debug else None,
                "visible_tool_round_trips": tool_trace["visible_tool_round_trips"]},
            "replayable": not any(i["code"] in blocking for i in local_issues), "issues": local_issues,
            "compiler_only_request": request}


def collect_run(run_id, project_root, *, search_roots=()):
    validate_run_id(run_id)
    project_root = Path(project_root).resolve()
    issues, roots = [], []
    try:
        roots.extend(storage_roots(project_root))
    except (RuntimeError, OSError, ValueError) as exc:
        issues.append({"code": "configured_storage_unavailable", "detail": str(exc)})
        roots.append(project_root / "runs")
    roots.extend([project_root / "results", *(Path(p).resolve() for p in search_roots)])
    roots = list(dict.fromkeys(roots))
    run_roots, summaries = [], {}
    for root in roots:
        if not root.exists():
            issues.append({"code": "search_root_missing", "path": str(root)})
            continue
        for path in root.rglob("run_metadata.json"):
            data = _read(path, issues)
            if path.parent.name == run_id or _belongs(data, run_id):
                run_roots.append(path.parent)
        for path in root.rglob("summary.json"):
            data = _read(path, issues)
            if run_id in path.parts or _belongs(data, run_id):
                summaries[path.parent] = data
        if root.name == run_id:
            run_roots.append(root)
    candidates = set()
    for root in roots:
        if not root.exists():
            continue
        candidates.update(p.resolve() for p in root.rglob("record.json"))
        candidates.update(p.resolve() for p in root.rglob("*_record.json"))
    collected, hashes, identities, duplicates, conflicting_paths = [], {}, {}, [], set()
    # Prefer original records with their adjacent evidence over review copies.
    for path in sorted(candidates, key=lambda p: (p.name != "record.json", str(p))):
        data = _read(path, issues)
        if "iteration" not in data:
            continue
        summary_dir = next((p for p in path.parents if p in summaries), None)
        if not (run_id in path.parts or summary_dir or _belongs(data, run_id)):
            continue
        fingerprint = digest(data)
        if fingerprint in hashes:
            duplicates.append({"path": str(path), "duplicate_of": str(hashes[fingerprint])})
            continue
        hashes[fingerprint] = path
        stamp = _timestamp(data.get("completed_at"))
        if stamp is None:
            stamp = _timestamp((data.get("planning_diagnostics", {}).get("visual_plan_result") or {}).get("created_at"))
        if stamp is None:
            issues.append({"code": "chronology_unavailable", "path": str(path), "detail": "No UTC timestamp; file ordering is not claimed chronological."})
        key = (data.get("iteration"), data.get("completed_at"))
        if key[1] and key in identities:
            issues.append({"code": "conflicting_record", "path": str(path), "other": str(identities[key])})
            conflicting_paths.update((path, identities[key]))
            continue
        identities[key] = path
        segment = summary_dir.name if summary_dir else path.parent.parent.name
        collected.append((stamp, path, data, segment))
    collected.sort(key=lambda item: (item[0] is None, item[0] or 0, str(item[1])))
    resolution_roots = list(dict.fromkeys([*run_roots, *summaries, *roots]))
    iterations, decisions = [], []
    for stamp, path, data, segment in collected:
        visual = (data.get("planning_diagnostics") or {}).get("visual_plan_result")
        debug_dirs = sorted(p for p in path.parent.glob("claude_image_tools/visual_planning_*") if p.is_dir())
        fresh = bool(visual or debug_dirs)
        reused = bool(data.get("height_retry")) and not fresh
        kind = "REPLANNED" if fresh else "REUSED" if reused else "UNKNOWN"
        iteration = {"iteration_id": f"{segment}:{data['iteration']}", "iteration": data["iteration"],
                     "segment": segment, "completed_at": data.get("completed_at"), "source": str(path),
                     "classification": kind, "reuse_evidence": data.get("height_retry") if reused else None}
        iterations.append(iteration)
        if fresh:
            for debug_dir in debug_dirs or [None]:
                decision = _decision(data, path, debug_dir, resolution_roots, run_id, segment, issues)
                if path in conflicting_paths:
                    decision["replayable"] = False
                    decision["issues"].append({"code": "conflicting_record"})
                decisions.append(decision)
        elif not reused:
            issues.append({"code": "decision_classification_unknown", "path": str(path)})
    for directory, summary in summaries.items():
        for entry in summary.get("iterations", []):
            if not isinstance(entry, dict) or type(entry.get("iteration")) is not int:
                continue
            expected = directory / f"iteration_{entry['iteration']:03d}" / "record.json"
            if not expected.is_file():
                issues.append({"code": "summary_record_missing", "path": str(expected)})
        for iteration_dir in directory.glob("iteration_*"):
            if iteration_dir.is_dir() and not (iteration_dir / "record.json").is_file():
                issues.append({"code": "partial_iteration_without_record", "path": str(iteration_dir)})
    order = {row["iteration_id"]: index for index, row in enumerate(iterations)}
    decisions.sort(key=lambda row: (order[row["iteration_id"]], _timestamp(row["created_at"]) is None,
                                    _timestamp(row["created_at"]) or 0, row["decision_id"]))
    if not collected:
        issues.append({"code": "run_records_missing", "run_id": run_id})
    return {"schema_version": 1, "run_id": run_id, "created_at": now(), "search_roots": [str(p) for p in roots],
            "run_roots": [str(p) for p in run_roots], "segments": sorted({r["segment"] for r in iterations}),
            "iterations": iterations, "decisions": decisions, "duplicates": duplicates, "issues": issues,
            "counts": {"iterations": len(iterations), "independent_visual_decisions": len(decisions),
                       "reused_iterations": sum(r["classification"] == "REUSED" for r in iterations),
                       "unknown_iterations": sum(r["classification"] == "UNKNOWN" for r in iterations),
                       "replayable_decisions": sum(d["replayable"] for d in decisions)},
            "chronology_complete": bool(collected) and all(row[0] is not None for row in collected) and all(_timestamp(d["created_at"]) is not None for d in decisions),
            "experiment_kind": "same-run trace compression; not generalization or grasp success"}


def save_manifest(manifest, output):
    """Persist explicit A/B split; executor only receives pre_decision."""
    output = Path(output)
    for decision in manifest["decisions"]:
        directory = output / "traces" / decision["decision_id"]
        write_json(directory / "pre_decision.json", decision["pre_decision"])
        write_json(directory / "post_decision.json", decision["post_decision"])
        write_json(directory / "trace.json", decision)
    write_json(output / "manifest.json", manifest)
