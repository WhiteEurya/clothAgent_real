"""Stages 1-3 CLI: audit, configure, query and visualize; never moves robots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..geometry import DualArmError
from .scene import ROOT, CollisionScene


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser(
        "audit", help="audit URDF, saved FK and imported calibration without hardware"
    )
    audit.add_argument("--output", type=Path, required=True)
    init = commands.add_parser(
        "init",
        help="write incomplete measured-data template or explicit synthetic demo",
    )
    init.add_argument("--synthetic", action="store_true")
    init.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check", help="check one complete 13DoF model state")
    check.add_argument("--scene", type=Path, required=True)
    check.add_argument(
        "--state", type=Path, help='JSON: {"units":"rad", "q_a":[6], "q_b":[7]}'
    )
    check.add_argument("--output", type=Path, required=True)
    evidence = commands.add_parser(
        "validate-evidence",
        help="compare paired saved controller readings and physical distances",
    )
    evidence.add_argument("--scene", type=Path, required=True)
    evidence.add_argument("--evidence", type=Path, required=True)
    evidence.add_argument("--output", type=Path, required=True)
    viewer = commands.add_parser(
        "view", help="offline Viser joint sliders, collision meshes and closest pair"
    )
    viewer.add_argument("--scene", type=Path, required=True)
    viewer.add_argument("--host", default="127.0.0.1")
    viewer.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            from .audit import model_audit

            report = model_audit(args.project_root)
            write(args.output, report)
            print(f"Audit saved to {args.output}; physical acceptance is unresolved")
        elif args.command == "init":
            from .calibration import template

            write(args.output, template(args.project_root, synthetic=args.synthetic))
        else:
            scene = CollisionScene.load(args.scene, args.project_root)
            if args.command == "check":
                from .checker import CollisionChecker

                q_a, q_b = scene.initial["left"], scene.initial["right"]
                if args.state:
                    state = json.loads(args.state.read_text())
                    if state.get("units") != "rad":
                        raise DualArmError("joint state units must explicitly be rad")
                    q_a, q_b = state["q_a"], state["q_b"]
                report = CollisionChecker(scene).check(q_a, q_b)
                write(args.output, report)
                print(
                    json.dumps(
                        {
                            k: report[k]
                            for k in (
                                "collision",
                                "min_distance",
                                "closest_pair",
                                "safe",
                            )
                        },
                        indent=2,
                    )
                )
                return 0 if report["safe"] else 2
            if args.command == "validate-evidence":
                from .audit import validate_evidence

                report = validate_evidence(scene, json.loads(args.evidence.read_text()))
                write(args.output, report)
                return 0 if report["numerical_consistency"] else 2
            if args.command == "view":
                from .viewer import serve

                serve(scene, args.host, args.port)
        return 0
    except (DualArmError, ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Collision review failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
