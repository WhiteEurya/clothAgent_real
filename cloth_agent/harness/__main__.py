"""Offline harness compilation/replay; never launches the robot loop."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .collector import collect_run, save_manifest
from .common import read_json, write_json
from .compiler import compile_policy
from .executor import execute_policy
from .model import RuntimeClaude
from .policy import load_policy
from .report import comparison, write_report


def replay_policy(frozen, manifest, output, model):
    rows = []
    for trace in manifest["decisions"]:
        if not trace["replayable"]:
            rows.append({"decision_id": trace["decision_id"], "iteration_id": trace["iteration_id"],
                         "status": "SKIPPED", "reason": "Pre-decision evidence incomplete",
                         "issues": trace["issues"], "exact_candidate_agreement": None})
            continue
        # Do not pass trace, historical result or compiler provenance to the executor.
        result = execute_policy(frozen, trace["pre_decision"], model,
                                Path(output) / "replays" / trace["decision_id"])
        rows.append(comparison(trace, result))
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "compile", "replay", "experiment"):
        p = commands.add_parser(name)
        p.add_argument("--run-id", required=True)
        p.add_argument("--project-root", type=Path, default=Path.cwd())
        p.add_argument("--search-root", type=Path, action="append", default=[],
                       help="Additional original/archived run roots; repeatable")
        p.add_argument("--output", type=Path, required=True, help="New directory; never overwrites a prior experiment")
        if name == "collect":
            continue
        p.add_argument("--backend", choices=("local", "remote"), default="local")
        p.add_argument("--claude-binary", default="claude")
        p.add_argument("--ssh-host", default="company-planner")
        p.add_argument("--model", help="Omit to use the installed Claude configuration")
        p.add_argument("--timeout-s", type=int, default=120)
        if name in {"compile", "experiment"}:
            p.add_argument("--task-context", default="Select one current Camera-A Rxxx grasp candidate for the current fold goal.")
            p.add_argument("--max-format-repairs", type=int, choices=(0, 1), default=1)
            p.add_argument("--max-compile-images", type=int, default=48)
        if name == "replay":
            p.add_argument("--policy", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command != "collect" and args.timeout_s <= 0:
        print("ERROR: --timeout-s must be positive", file=sys.stderr)
        return 2
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"ERROR: output already exists; choose a new directory: {output}", file=sys.stderr)
        return 2
    manifest, compilation = None, None
    try:
        manifest = collect_run(args.run_id, args.project_root, search_roots=args.search_root)
        save_manifest(manifest, output)
        if args.command == "collect":
            write_report(output, manifest, None, [])
            print(json.dumps({"output": str(output), "counts": manifest["counts"], "issues": manifest["issues"]}))
            return 0 if manifest["decisions"] else 2
        model = RuntimeClaude(backend=args.backend, binary=args.claude_binary, ssh_host=args.ssh_host,
                              model=args.model, timeout_s=args.timeout_s)
        if args.command in {"compile", "experiment"}:
            compilation = compile_policy(manifest, args.task_context, output, model,
                                         max_repairs=args.max_format_repairs, max_images=args.max_compile_images)
            if compilation["status"] != "FROZEN":
                write_report(output, manifest, compilation, [])
                print(json.dumps({"output": str(output), "compilation": compilation}))
                return 2
            frozen = load_policy(compilation["policy_path"])
        else:
            frozen = load_policy(args.policy)
            # Immutable run-local snapshot, excluding source/evidence sidecar.
            write_json(output / "replay_policy.json", frozen, exclusive=True)
        rows = replay_policy(frozen, manifest, output, model) if args.command != "compile" else []
        report = write_report(output, manifest, compilation, rows, policy_hash=frozen["policy_hash"])
        print(json.dumps({"output": str(output), "policy_path": (compilation or {}).get("policy_path"),
                          "policy_hash": frozen["policy_hash"], "counts": report["counts"]}))
        return 0
    except Exception as exc:
        error = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
        write_json(output / "error.json", error)
        if manifest:
            write_report(output, manifest, compilation or error, [])
        print(json.dumps(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
