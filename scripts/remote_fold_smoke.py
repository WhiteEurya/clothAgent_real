#!/usr/bin/env python3
"""Exercise fold's actual remote planner on a saved run, without hardware.

Requires workspace/perception_views containing a matching saved RGB, coordinate
guide and geometry maps. Writes smoke artifacts under the run's results/ only,
apart from the normal fold reference-guide normalization performed by planning.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloth_agent.fold_exploration_pipeline import (
    FOLD_STEP_IDS, FoldExplorationPipeline, _build_upright_camera_a_planning_images,
    _validate_action_mode_contract, _validate_model_acquisition_probe,
)
from cloth_agent.free_exploration import _load_or_create_session


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--run-dir", type=Path, required=True, help="existing run with saved perception")
    parser.add_argument("--host", default="company-planner")
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--step", choices=FOLD_STEP_IDS, default="left_sleeve")
    parser.add_argument("--mode", choices=("ACQUISITION_PROBE", "FOLD", "REPAIR_SLEEVE"), default="ACQUISITION_PROBE")
    args = parser.parse_args(argv)
    try:
        root, run = args.project_root.resolve(), args.run_dir.resolve()
        # Loading the saved session constructs no SDK connection and starts no capture.
        session = _load_or_create_session(root, run, None, None)
        views = session.workspace / "perception_views"
        required = ("camera_0_A.png", "camera_A_coordinate_guide.json", "camera_A_base_xyz_mm.npy", "camera_A_garment_mask.npy")
        for name in required:
            if not (views / name).is_file():
                raise FileNotFoundError(f"saved perception required: {views / name}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = session.results / "remote_fold_smoke" / stamp
        print("[1/4] Preparing saved RGB evidence; no capture or robot connection", flush=True)
        images = _build_upright_camera_a_planning_images({"views": [{"label": "A",
            "image": "camera_0_A.png", "coordinate_guide": "camera_A_coordinate_guide.json"}]},
            views / "result.json", output)
        pipeline = FoldExplorationPipeline(session,
            perception_config=root / "config" / "perception.free_exploration.json",
            planner_backend="remote", remote_planner_host=args.host,
            claude_timeout_s=args.timeout_s, supervisor_timeout_s=args.timeout_s,
            max_stage_retries=0, record_video=False, molmo_sleeve_grounding=False)
        print(f"[2/4] Remote fold supervisor via HTTPS + SSH ({args.host})", flush=True)
        state = pipeline.supervisor.inspect(images, run, history=[], screen={})
        (output / "supervisor.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        objective = ("Fold this shirt in order: left sleeve inward, right sleeve inward, first torso side inward, "
            "second torso side inward, bottom hem upward. Left/right mean the displayed upright image. "
            f"The requested smoke-test current_step is {args.step}. Plan exactly this step.")
        objective += ("\nEXECUTION CONTRACT — ACQUISITION PROBE: return a reversible lift, reverse, release and home."
            if args.mode == "ACQUISITION_PROBE" else f"\nACTION MODE — {args.mode}: complete this task mode.")
        print("[3/4] Remote visual/motion proposal + local grounding and mode validation", flush=True)
        proposal = pipeline._plan_fold_with_retries(images, objective, [], iteration=1)
        proposal, height = pipeline._resolve_fold_grasp_height(proposal)
        mode_check = _validate_action_mode_contract(proposal, mode=args.mode)
        if args.mode == "ACQUISITION_PROBE":
            _validate_model_acquisition_probe(proposal)
        result = {"backend": "remote", "ssh_host": args.host, "hardware_connected": False,
            "proposal": proposal.as_dict(), "grasp_height": height, "mode_validation": mode_check,
            "grounding": pipeline.client.last_grounding_verification,
            "limitation": "No IK, controller preflight or physical execution was performed."}
        destination = output / "proposal.json"
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[4/4] REMOTE FOLD PROPOSAL PASSED: {destination}", flush=True)
        return 0
    except Exception as exc:
        print(f"REMOTE FOLD SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
