"""Post-replay comparisons. Agreement is not physical success."""
from __future__ import annotations

import math
from pathlib import Path

from .common import write_json

LIMITS = [
    "Same-run trace compression only; no cross-state generalization or physical grasp-success claim.",
    "Candidate agreement measures agreement with an earlier decision, not correctness; disagreement is not automatically error.",
    "Historical tool round trips and new Claude invocations are distinct metrics.",
    "Unavailable old visual-boundary timing is null; full robot iteration time is never used.",
    "No cached decision is timed as a live replay; a frozen policy is not proof of reliability.",
]


def comparison(trace, replay):
    # Historical answer and feedback are first read HERE, after execution.
    candidate = trace["post_decision"].get("candidate_id")
    registry = trace["pre_decision"].get("candidate_registry") or {}
    rows = {r["candidate_id"]: r for r in registry.get("candidates", [])}
    historic_xy = rows.get(candidate, {}).get("raw_pixel_xy")
    new_xy = replay.get("raw_pixel_xy")
    agreement = replay["candidate_id"] == candidate if replay["status"] == "READY" and candidate else None
    distance = math.dist(historic_xy, new_xy) if historic_xy and new_xy else None
    previous = trace["historical_metrics"].get("visual_decision_seconds")
    current = replay["metrics"].get("visual_decision_seconds")
    measured = replay["metrics"].get("actual_replay", False)
    return {"decision_id": trace["decision_id"], "iteration_id": trace["iteration_id"],
            "historical_candidate": candidate, "replay_candidate": replay["candidate_id"],
            "exact_candidate_agreement": agreement, "original_pixel_distance": distance,
            "historical_candidate_legal": candidate in rows if candidate and rows else None,
            "replay_candidate_legal": replay.get("candidate_legal"),
            "historical_metrics": trace["historical_metrics"], "replay_metrics": replay["metrics"],
            "visual_seconds_ratio_old_over_new": previous / current if previous is not None and current and measured and replay["status"] == "READY" else None,
            "status": replay["status"], "reason": replay["reason"]}


def write_report(output, manifest, compilation, rows, *, policy_hash=None):
    total = len(manifest["decisions"])
    counts = {name: sum(r["status"] == name for r in rows) for name in ("READY", "NEEDS_LEARNING", "ERROR", "SKIPPED")}
    available = [r["exact_candidate_agreement"] for r in rows if r.get("exact_candidate_agreement") is not None]
    report = {"schema_version": 1, "experiment_kind": manifest["experiment_kind"], "run_id": manifest["run_id"],
              "collection": {k: manifest[k] for k in ("counts", "segments", "duplicates", "issues", "chronology_complete")},
              "decision_issues": {d["decision_id"]: d["issues"] for d in manifest["decisions"] if d["issues"]},
              "compilation": compilation, "policy_hash": policy_hash or (compilation or {}).get("policy_hash"),
              "replays": rows, "counts": counts, "exact_candidate_agreement": sum(available) / len(available) if available else None,
              "agreement_denominator": len(available), "fallback_ratio": counts["NEEDS_LEARNING"] / total if total else None,
              "decisions_not_executed": total - len(rows),
              "error_ratio": counts["ERROR"] / total if total else None,
              "skip_ratio": counts["SKIPPED"] / total if total else None,
              "ready_coverage": counts["READY"] / total if total else None, "limitations": LIMITS}
    write_json(Path(output) / "report.json", report)
    lines = ["# Offline harness experiment", "", f"Run: {manifest['run_id']}",
             f"Iterations: {manifest['counts']['iterations']}; independent visual decisions: {total}; reused iterations: {manifest['counts']['reused_iterations']}.",
             f"Compilation: {(compilation or {}).get('status', 'loaded frozen policy')}; policy hash: {report['policy_hash'] or 'unavailable'}.",
             f"Replay counts: {counts}", f"Agreement: {report['exact_candidate_agreement']} (n={len(available)})", "",
             "| Decision | Status | Historical | Replay | Distance px | Old visual s | New visual s |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['decision_id']} | {row['status']} | {row.get('historical_candidate')} | {row.get('replay_candidate')} | {row.get('original_pixel_distance')} | {row.get('historical_metrics', {}).get('visual_decision_seconds')} | {row.get('replay_metrics', {}).get('visual_decision_seconds')} |")
    lines += ["", "## Measured work", "",
              "| Decision | Claude calls | Host image ops | Old tool round trips | New tool round trips | Reason |",
              "|---|---|---|---|---|---|"]
    for row in rows:
        metric = row.get("replay_metrics", {})
        reason = str(row.get("reason", "")).replace("|", "/").replace(chr(10), " ")
        lines.append(f"| {row['decision_id']} | {metric.get('claude_calls')} | {metric.get('host_image_ops')} | {row.get('historical_metrics', {}).get('visible_tool_round_trips')} | {metric.get('visible_tool_round_trips')} | {reason} |")
    lines += ["", f"Fallback ratio: {report['fallback_ratio']}; skip ratio: {report['skip_ratio']}; ready coverage: {report['ready_coverage']}."]
    if compilation:
        if compilation.get("mode") == "sequential_iterations":
            lines += [f"Sequential compilation: {compilation.get('completed_iterations', 0)}/{compilation.get('total_iterations', 0)} iterations completed."]
        lines += [f"Compilation backend invoked: {compilation.get('compilation_backend_invoked')}; response received: {compilation.get('compilation_response_received')}.",
                  f"Compilation elapsed seconds: {compilation.get('compilation_seconds')}; calls: {compilation.get('claude_call_count')}.",
                  f"Model configuration: {compilation.get('configuration')}."]
    lines += ["", "## Missing data and blockers", ""]
    lines += ["- " + str(issue) for issue in manifest["issues"]]
    lines += ["- " + key + ": " + str(value) for key, value in report["decision_issues"].items()]
    if compilation and compilation.get("error"):
        lines.append("- " + compilation["error"])
    lines += ["", "## Interpretation", "", *["- " + line for line in LIMITS]]
    (Path(output) / "report.md").write_text("\n".join(lines) + "\n")
    return report
