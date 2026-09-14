#!/usr/bin/env python3
"""Render saved RGB/workspace targets offline; no Claude, camera, IK or robot."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from cloth_agent.auto_exploration import validate_visual_plan_payload
from cloth_agent.config import RobotConfig
from cloth_agent.garment_grounding_mcp import GarmentGrounding
from cloth_agent.remote_fold import compile_pixel_motion
from cloth_agent.workspace_debug import WorkspaceTargetError, save_workspace_debug


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--views-dir", type=Path, help="Matching saved RGB/XYZ/guide directory; defaults to run workspace")
    parser.add_argument("--robot-config", type=Path, help="Explicit comparison config; default is the run's saved config")
    parser.add_argument("--motion-json", type=Path, help="Optional saved remote motion or failed invocation JSON")
    parser.add_argument("--visual-json", type=Path, help="Matching saved visual decision or visual result JSON")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if bool(args.motion_json) != bool(args.visual_json):
        parser.error("--motion-json and --visual-json must be supplied together from the same attempt")
    run = args.run_dir.resolve()
    config_path = args.robot_config or run / "workspace" / "robot_config.json"
    views = (args.views_dir or run / "workspace" / "perception_views").resolve()
    output = args.output_dir or run / "results" / "workspace_debug" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        config = RobotConfig.load(args.project_root, config_path)
        trace = None
        if args.motion_json:
            motion = json.loads(args.motion_json.read_text())
            motion = motion.get("remote_motion", motion.get("response", motion))
            visual = json.loads(args.visual_json.read_text())
            visual = visual.get("visual_plan_result", visual)
            visual = visual.get("decision", visual.get("response", visual))
            with Image.open(views / "camera_0_A.png") as im:
                upright_size = (im.height, im.width)
            try:
                _, result = compile_pixel_motion(motion, validate_visual_plan_payload(visual),
                    GarmentGrounding(views), config, upright_size)
                trace = result["workspace_trace"]
            except WorkspaceTargetError as exc:
                trace = exc.trace
        report = save_workspace_debug(output, views, config, trace)
        print(f"Config: {config_path.resolve()}")
        print(f"Saved geometry: {views}")
        print(f"Status: {report['status']} (offline; no execution)")
        print(f"Images and measured targets: {output.resolve()}")
        if report.get("render_error"):
            raise RuntimeError(report["render_error"])
        return 0
    except Exception as exc:
        print(f"WORKSPACE DEBUG FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
