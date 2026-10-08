"""Offline solve/plan/preflight/view commands. No hardware execution option."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..collision.__main__ import write
from ..collision.scene import CollisionScene
from ..geometry import DualArmError
from .ik import IKOptions
from .planner import DualArmPlanner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("solve", "plan", "preflight", "view"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--scene", type=Path, required=True)
        if name in {"solve", "plan"}:
            cmd.add_argument("--request", type=Path, required=True)
            cmd.add_argument("--output", type=Path, required=True)
        else:
            cmd.add_argument("--plan", type=Path, required=True)
        if name == "preflight":
            cmd.add_argument("--state", type=Path, required=True)
            cmd.add_argument("--output", type=Path, required=True)
        if name == "view":
            cmd.add_argument("--port", type=int, default=8767)
    args = parser.parse_args(argv)
    try:
        scene = CollisionScene.load(args.scene)
        if args.command in {"solve", "plan"}:
            request = json.loads(args.request.read_text())
            if request.pop("units", None) != "m_rad":
                raise DualArmError("request must declare units=m_rad")
            planner = DualArmPlanner(
                scene, ik_options=IKOptions(**request.pop("ik_options", {}))
            )
            result = (
                planner.solve_grasp_pose if args.command == "solve" else planner.plan
            )(**request)
            write(args.output, result)
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "success",
                            "reason",
                            "failed_phase",
                            "minimum_clearance",
                            "elapsed_s",
                        )
                        if k in result
                    }
                )
            )
            return 0 if result["success"] else 2
        planner = DualArmPlanner(scene)
        plan = json.loads(args.plan.read_text())
        if args.command == "preflight":
            from .preflight import preflight

            state = json.loads(args.state.read_text())
            if state.pop("units", None) != "rad_s":
                raise DualArmError(
                    "state must declare units=rad_s; timestamps are Unix seconds"
                )
            write(args.output, preflight(plan, planner, **state))
        else:
            from .viewer import serve

            serve(planner, plan, args.port)
        return 0
    except (DualArmError, ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Motion review failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
