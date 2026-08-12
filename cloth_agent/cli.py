"""Command line entry points for the minimal Agent experimentation loop."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import ExperimentConfig, RobotConfig
from .experiment import ExperimentValidationError, format_action_sequence, format_speed_profile
from .perception import PerceptionConfig
from .session import AgentSession, MANUAL_RESULTS


def _robot_config(project_root: Path, path: str | None) -> RobotConfig:
    return RobotConfig.load(project_root, Path(path) if path else None)


def _experiment_config(args: argparse.Namespace, *, allow_deferred: bool = False) -> ExperimentConfig:
    if args.experiment_config:
        return ExperimentConfig.from_mapping(
            json.loads(Path(args.experiment_config).read_text(encoding="utf-8")),
            allow_deferred=allow_deferred,
        )
    values = {
        "center_x": args.center_x,
        "center_y": args.center_y,
        "grasp_z": args.grasp_z,
        "approach_z": args.approach_z,
        "lift_z": args.lift_z,
        "yaw": args.yaw,
    }
    missing = [
        name for name, value in values.items()
        if value is None and not allow_deferred
    ]
    if missing:
        raise ValueError(f"provide --experiment-config or all experiment values ({', '.join(missing)})")
    return ExperimentConfig.from_mapping(values, allow_deferred=allow_deferred)


def _add_config_args(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        "--experiment-config",
        help="optional manual JSON; with --detect-center all scene values are derived automatically",
    )
    for name, flag in (("center_x", "--center-x"), ("center_y", "--center-y"), ("grasp_z", "--grasp-z"), ("approach_z", "--approach-z"), ("lift_z", "--lift-z"), ("yaw", "--yaw")):
        parser.add_argument(flag, dest=name, type=float, required=False, help=f"cloth/target {name.replace('_', ' ')}")
    parser.set_defaults(yaw=None)


def _session(project_root: Path, run_dir: Path) -> AgentSession:
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    saved_robot_config = run_dir / "workspace" / "robot_config.json"
    robot = _robot_config(project_root, str(saved_robot_config) if saved_robot_config.exists() else None)
    saved_experiment_config = run_dir / "workspace" / "experiment_config.json"
    experiment_values = (
        json.loads(saved_experiment_config.read_text(encoding="utf-8"))
        if saved_experiment_config.exists()
        else metadata["experiment_config"]
    )
    experiment = ExperimentConfig.from_mapping(experiment_values, allow_deferred=True)
    return AgentSession(project_root, run_dir, robot, experiment)


def _perception_config(
    project_root: Path,
    raw_path: str | None,
    single_camera: str | None = None,
) -> PerceptionConfig:
    path = Path(raw_path or "config/perception.example.json")
    if not path.is_absolute():
        path = project_root / path
    config = PerceptionConfig.load(project_root, path)
    if single_camera:
        config = replace(config, active_camera_labels=(single_camera.strip().upper(),))
        config.validate()
    return config


def _cmd_create(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    robot = _robot_config(root, args.robot_config)
    experiment = _experiment_config(args, allow_deferred=True)
    session = AgentSession.create(root, args.goal, robot, experiment, run_id=args.run_id)
    print(session.run_dir)
    print("workspace:", session.workspace)
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    session = _session(root, Path(args.run_dir).resolve())
    target = args.experiment_name or session._next_experiment_name()
    result = session.invoke_claude_code(args.prompt, experiment_name=target)
    print(result.stdout)
    print("generated:", session.workspace / target)
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    session = _session(root, Path(args.run_dir).resolve())
    preflight = session.runner.preflight(args.experiment)
    if preflight.stdout:
        print(preflight.stdout, end="" if preflight.stdout.endswith("\n") else "\n")
    print(format_speed_profile(session.robot_config))
    print(format_action_sequence(preflight.actions))
    if preflight.error:
        print("PREFLIGHT_ERROR:", preflight.error, file=sys.stderr)
        return 2
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    session = _session(root, Path(args.run_dir).resolve())
    try:
        result = session.run_experiment(
            args.experiment,
            real=args.real,
            confirmed=args.confirm_real,
            single_view_confirmed=args.confirm_single_view,
            notes=args.notes,
        )
    except (PermissionError, ExperimentValidationError, RuntimeError) as exc:
        print(f"RUN_BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["execution_completed"] else 1


def _cmd_inspect(args: argparse.Namespace) -> int:
    session = _session(Path(args.project_root).resolve(), Path(args.run_dir).resolve())
    if args.file:
        print(session.inspect_file(args.file))
    else:
        print(json.dumps(session.inspect_result(args.experiment), ensure_ascii=False, indent=2))
    return 0


def _cmd_label(args: argparse.Namespace) -> int:
    session = _session(Path(args.project_root).resolve(), Path(args.run_dir).resolve())
    session.record_manual_result(args.experiment, args.status, args.notes)
    print(json.dumps(session.inspect_result(args.experiment), ensure_ascii=False, indent=2))
    return 0


def _cmd_memory(args: argparse.Namespace) -> int:
    session = _session(Path(args.project_root).resolve(), Path(args.run_dir).resolve())
    session.update_memory(args.experiment, hypothesis=args.hypothesis, next_experiment=args.next_experiment, result=args.result, notes=args.notes)
    print(session.workspace / "memory.md")
    return 0


def _cmd_perceive(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    session = _session(root, Path(args.run_dir).resolve())
    config = _perception_config(root, args.perception_config, args.single_camera)
    result = session.locate_cloth_center(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Updated experiment config:", session.workspace / "experiment_config.json")
    return 0


def _cmd_viewer(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    if args.run_dir:
        session = _session(root, Path(args.run_dir).resolve())
    else:
        existing = root / "runs" / args.run_id if args.run_id else None
        if existing is not None and existing.is_dir():
            session = _session(root, existing.resolve())
        else:
            robot = _robot_config(root, args.robot_config)
            session = AgentSession.create(
                root,
                "Viser-operated cloth center grasp",
                robot,
                ExperimentConfig(),
                run_id=args.run_id,
            )
    from .viewer import run_viewer

    perception_path = Path(args.perception_config) if args.perception_config else None
    if perception_path is not None and not perception_path.is_absolute():
        perception_path = root / perception_path
    urdf_path = Path(args.urdf) if args.urdf else None
    if urdf_path is not None and not urdf_path.is_absolute():
        urdf_path = root / urdf_path

    return run_viewer(
        session,
        experiment=args.experiment,
        host=args.host,
        port=args.port,
        enable_real=args.enable_real,
        perception_config_path=perception_path,
        urdf_path=urdf_path,
    )


def _cmd_session(args: argparse.Namespace) -> int:
    """Interactive end-to-end loop; physical motion remains opt-in per rollout."""
    root = Path(args.project_root).resolve()
    if args.single_camera and not args.detect_center:
        raise ValueError("--single-camera requires --detect-center")
    robot = _robot_config(root, args.robot_config)
    experiment = _experiment_config(args, allow_deferred=args.detect_center)
    session = AgentSession.create(root, args.goal, robot, experiment, run_id=args.run_id)
    print(f"Run workspace: {session.workspace}")
    perception_result: dict[str, object] = {}
    if args.detect_center:
        perception_config = _perception_config(
            root, args.perception_config, args.single_camera
        )
        perception_result = session.locate_cloth_center(perception_config)
        print("\n--- validated Molmo cloth center ---")
        print(json.dumps(perception_result, ensure_ascii=False, indent=2))
    print("The Agent will ask Claude Code to write the experiment in that workspace.")
    target = session._next_experiment_name()
    session.invoke_claude_code(args.intent, experiment_name=target)
    current = target
    while True:
        source = session.workspace / current
        if not source.is_file():
            print(f"Claude did not create {source}", file=sys.stderr)
            return 2
        print("\n--- generated experiment source ---")
        print(session.inspect_file(source))
        preflight = session.runner.preflight(current)
        print(format_speed_profile(robot))
        print(format_action_sequence(preflight.actions))
        if preflight.error:
            print("PREFLIGHT_ERROR:", preflight.error, file=sys.stderr)
            return 2
        if args.real:
            robot.validate_for_real()
            print("No robot command has run yet. Review every xyz/yaw above.")
            single_view = perception_result.get("perception_mode") == "single_camera_rgbd"
            confirmation = "EXECUTE_SINGLE_VIEW" if single_view else "EXECUTE"
            if input(f"Type {confirmation} to allow this one physical rollout: ").strip() != confirmation:
                print("Physical execution cancelled; no robot command was sent.")
                return 0
            result = session.run_experiment(
                current,
                real=True,
                confirmed=True,
                single_view_confirmed=single_view,
            )
        else:
            result = session.run_experiment(current, real=False, confirmed=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        status = ""
        while status not in MANUAL_RESULTS:
            status = input(f"Manual result ({'/'.join(sorted(MANUAL_RESULTS))}): ").strip().upper()
        stem = current[:-3] if current.endswith(".py") else current
        session.record_manual_result(stem, status)
        if status == "SUCCESS":
            session.update_memory(stem, hypothesis="Manual observation indicates success; no modification needed.", next_experiment="STOP", result=status)
            print("Agent session stopped.")
            return 0
        if input("STOP, or type MODIFY_EXPERIMENT: ").strip().upper() != "MODIFY_EXPERIMENT":
            session.update_memory(stem, hypothesis="Operator chose to stop without another experiment.", next_experiment="STOP", result=status)
            print("Agent session stopped.")
            return 0
        reason = input("Why should the next experiment change? ").strip()
        next_name = session._next_experiment_name()
        session.update_memory(stem, hypothesis=reason, next_experiment=next_name, result=status)
        session.invoke_claude_code(reason, experiment_name=next_name)
        current = next_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="create runs/<run_id>/workspace")
    create.add_argument("--goal", required=True)
    create.add_argument("--run-id")
    create.add_argument("--robot-config")
    _add_config_args(create)
    create.set_defaults(func=_cmd_create)

    generate = sub.add_parser("generate", help="invoke Claude Code inside one run workspace")
    generate.add_argument("--run-dir", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--experiment-name")
    generate.set_defaults(func=_cmd_generate)

    viewer = sub.add_parser("viewer", help="start the Viser grasp-path preview dashboard")
    viewer.add_argument(
        "--run-dir",
        help="existing run; omit to create a new deferred Viser run",
    )
    viewer.add_argument("--run-id", help="new run directory name when --run-dir is omitted")
    viewer.add_argument("--robot-config")
    viewer.add_argument("--perception-config")
    viewer.add_argument("--urdf", help="defaults to assets/robots/xarm7/xarm7.urdf")
    viewer.add_argument("--experiment", help="experiment_*.py; defaults to latest or a safe canonical plan")
    viewer.add_argument("--host", default="127.0.0.1")
    viewer.add_argument("--port", type=int, default=8080)
    viewer.add_argument(
        "--enable-real",
        action="store_true",
        help="allow the validated GUI plan to expose its one-click physical execution button",
    )
    viewer.set_defaults(func=_cmd_viewer)

    for name, func in (("preflight", _cmd_preflight), ("run", _cmd_run), ("inspect", _cmd_inspect), ("label", _cmd_label), ("memory", _cmd_memory), ("perceive", _cmd_perceive)):
        command = sub.add_parser(name)
        command.add_argument("--run-dir", required=True)
        if name in {"preflight", "run"}:
            command.add_argument("--experiment", required=True)
        if name == "run":
            command.add_argument("--real", action="store_true", help="allow physical xArm execution after confirmation")
            command.add_argument("--confirm-real", action="store_true", help="explicit confirmation already shown/reviewed")
            command.add_argument(
                "--confirm-single-view",
                action="store_true",
                help="also confirm that the plan came from one RGB-D camera",
            )
            command.add_argument("--notes", default="")
        if name == "inspect":
            command.add_argument("--experiment")
            command.add_argument("--file")
        if name == "label":
            command.add_argument("--status", required=True, choices=sorted(MANUAL_RESULTS))
            command.add_argument("--notes", default="")
        if name == "memory":
            command.add_argument("--experiment", required=True)
            command.add_argument("--hypothesis", required=True)
            command.add_argument("--next-experiment", required=True)
            command.add_argument("--result", required=True)
            command.add_argument("--notes", default="")
        if name == "perceive":
            command.add_argument(
                "--perception-config",
                help="defaults to config/perception.example.json",
            )
            command.add_argument(
                "--single-camera",
                choices=("A", "B"),
                help="temporary one-camera RGB-D mode; default uses A and B",
            )
        command.set_defaults(func=func)

    session = sub.add_parser("session", help="create a run and drive the interactive Agent loop")
    session.add_argument("--goal", required=True)
    session.add_argument("--intent", required=True, help="Agent's research intent sent to Claude Code")
    session.add_argument("--run-id")
    session.add_argument("--robot-config")
    session.add_argument(
        "--real",
        action="store_true",
        help="offer each rollout for explicit EXECUTE/EXECUTE_SINGLE_VIEW confirmation",
    )
    session.add_argument(
        "--detect-center",
        action="store_true",
        help="capture configured camera view(s) and run Molmo before code generation",
    )
    session.add_argument("--perception-config")
    session.add_argument(
        "--single-camera",
        choices=("A", "B"),
        help="temporary one-camera RGB-D mode; real execution requires EXECUTE_SINGLE_VIEW",
    )
    _add_config_args(session)
    session.set_defaults(func=_cmd_session)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, PermissionError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
