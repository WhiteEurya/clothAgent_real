"""Separate dual-arm entry point; the existing single-arm CLI is unchanged."""

from __future__ import annotations

import argparse
import json
import signal
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
from jsonschema import ValidationError

from .config import DualConfig
from .execution import DualArmCoordinator, SimulatedConnection, XArmConnection
from .geometry import DualArmError, matrix_pose, pose_matrix
from .kinematics import ArmModel
from .model import VisionPlanner
from .observation import Observation, capture, save_observation
from .planning import (
    Motion,
    Phase,
    build_phases,
    compile_program,
    digest,
    ground_targets,
)
from .preview import write_preview
from .safety import synthetic_safety
from .setup import fit_base, initial_config, mesh_capsules

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")


def save_plan(
    directory,
    config,
    proposal,
    observation,
    initial,
    phases,
    program,
    *,
    controller_ik=False,
):
    write_json(directory / "config.json", config.raw)
    write_json(directory / "proposal.json", proposal)
    serial_phases = []
    for p in phases:
        row = asdict(p)
        row["targets"] = {k: v.tolist() for k, v in p.targets.items()}
        serial_phases.append(row)
    write_json(
        directory / "plan.json",
        {
            "schema_version": 1,
            "frame_id": "world_mm",
            "config_sha256": digest(config.raw),
            "proposal_sha256": digest(proposal),
            "observation_id": observation.meta["observation_id"],
            "observation_directory": str(observation.directory),
            "initial_joints_deg": {
                k: np.asarray(q).tolist() for k, q in initial.items()
            },
            "phases": serial_phases,
            "motion_duration_s": sum(
                float(m.times[-1]) for m in program if isinstance(m, Motion)
            ),
            "validation": {
                "controller_ik": controller_ik,
                "urdf_fk": True,
                "sampled_collision": True,
                "continuous_joint_box_collision": True,
                "tracking_and_stop_envelopes": config.raw["safety"]["status"],
                "physical_trial": False,
            },
        },
    )
    motion_dir = directory / "trajectories"
    motion_dir.mkdir()
    for motion in program:
        if isinstance(motion, Motion):
            np.savez_compressed(
                motion_dir / f"{motion.phase.name}.npz",
                times=motion.times,
                left_joints_deg=motion.joints["left"],
                right_joints_deg=motion.joints["right"],
                left_pose_world_mm_deg=motion.poses["left"],
                right_pose_world_mm_deg=motion.poses["right"],
            )


def connect(config, models, cancel, real):
    connections = {}
    try:
        for k, a in config.arms.items():
            connections[k] = (
                XArmConnection(a, cancel, recover_stopped=getattr(config, 'home_recover_stopped', False))
                if real
                else SimulatedConnection(a, models[k], cancel)
            )
        return connections
    except BaseException:
        for c in connections.values():
            c.disconnect()
        raise


def run(args):
    if args.real and not args.confirm_real:
        raise DualArmError("--real requires --confirm-real")
    if args.confirm_real and not args.real:
        raise DualArmError("--confirm-real requires --real")
    if args.real and not args.preflight_only and args.mode == "pin_pull":
        raise DualArmError(
            "pin_pull requires a force-limited contact controller for physical execution; use simulation or preflight"
        )
    config = DualConfig.load(args.config, root=args.project_root)
    if args.real:
        config.require_real()
    if not args.real and args.observation is None:
        raise DualArmError(
            "simulation needs --observation; use demo for a complete synthetic example"
        )
    if not args.real and args.proposal is None:
        raise DualArmError(
            "simulation needs --proposal; plan it with the plan command first"
        )
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    models = {k: ArmModel(a) for k, a in config.arms.items()}
    cancel = threading.Event()
    connections = connect(config, models, cancel, args.real)
    coordinator = DualArmCoordinator(
        config, models, connections, cancel, directory, realtime=args.real
    )
    old_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        cancel.set()
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(signal.SIGTERM, terminate)
    try:
        observation = (
            Observation.load(args.observation)
            if args.observation
            else capture(
                directory / "initial_observation", config, models, coordinator.snapshots
            )
        )
        observation.validate_for(config, live=args.real)
        if not args.real:
            for k, c in connections.items():
                c.joints = np.asarray(observation.meta["joints_deg"][k], dtype=float)
        initial_state = coordinator.snapshots()
        initial = {k: np.asarray(v["joints"]) for k, v in initial_state.items()}
        for k, joints in initial.items():
            if (
                np.max(np.abs(joints - observation.meta["joints_deg"][k]))
                > config.limits["stationary_tolerance_deg"]
            ):
                raise DualArmError(
                    "arms changed pose since the grasp observation; recapture"
                )
        planner = VisionPlanner(
            args.backend,
            host=args.host,
            timeout_s=int(config.limits["max_vision_wait_s"]) - 1,
        )
        proposal = (
            json.loads(args.proposal.read_text())
            if args.proposal
            else planner.plan(
                observation, config, directory / "claude_plan", mode=args.mode
            )
        )
        if proposal.get("mode") != args.mode:
            raise DualArmError(
                "proposal mode differs from --mode; refusing implicit mode changes"
            )
        targets = ground_targets(proposal, observation, config)
        initial_world = {
            k: matrix_pose(
                config.arms[k].world_from_base @ pose_matrix(models[k].forward(q))
            )
            for k, q in initial.items()
        }
        phases = build_phases(proposal, targets, initial_world, config)
        program = compile_program(
            phases,
            initial,
            config,
            models,
            {k: c.inverse for k, c in connections.items()},
        )
        save_plan(
            directory,
            config,
            proposal,
            observation,
            initial,
            phases,
            program,
            controller_ik=args.real,
        )
        write_preview(program, config, models, directory / "preview.html")
        print(
            f"Validated {len(phases)} paired phases. Preview: {directory / 'preview.html'}",
            flush=True,
        )
        if args.preflight_only:
            write_json(
                directory / "execution.json",
                {"status": "PREFLIGHT_ONLY", "physical_execution": False},
            )
            return 0

        def checkpoint(phase):
            if not args.real:
                result = {
                    "left_holding": True,
                    "right_holding": True,
                    "slip": False,
                    "overstretched": False,
                    "confidence": 1.0,
                    "reason": "Synthetic hold assumption for simulation only; no physical grasp evidence.",
                }
                if phase.pin_arm:
                    result.update(pin_contact=True, pin_slip=False)
                    result[f"{phase.pin_arm}_holding"] = False
                return result
            current = capture(
                directory / phase.name / "observation",
                config,
                models,
                coordinator.snapshots,
            )
            return planner.check(
                observation,
                current,
                phase,
                config,
                directory / phase.name / "claude",
            )

        result = coordinator.execute(
            program, initial, observation, checkpoint, confirmed=args.confirm_real
        )
        print(f"{result['status']}: {directory}", flush=True)
        return 0 if result["status"] == "COMPLETED" else 1
    except BaseException as exc:
        coordinator.stop_all()
        write_json(
            directory / "failure.json",
            {
                "status": "FAILED",
                "physical_execution_requested": args.real,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        coordinator.close()


def demo(args):
    """Real URDF kinematics with clearly synthetic bases, cloth and feedback."""
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config_raw = initial_config(
        args.project_root, args.project_root / "data/robot/dual_arm_home.json"
    )
    config_raw.update(
        calibration_status="synthetic",
        calibration_id="synthetic-demo-not-for-hardware",
        arm_layout_description="Synthetic parallel bases, left at world Y=-350 mm and right at Y=+350 mm.",
    )
    config_raw["limits"].update(contact_descent_mm=0, max_spread_mm=30, max_lift_mm=40)
    config_raw["limits"]["center_radius_mm"] = 500
    config_raw["safety"] = synthetic_safety(config_raw["arms"])
    for k, a in config_raw["arms"].items():
        matrix = np.eye(4)
        matrix[1, 3] = -350 if k == "left" else 350
        a["world_from_base_mm"] = matrix.tolist()
        a["workspace"] = {"min_mm": [-1000, -1000, -100], "max_mm": [1000, 1000, 1500]}
        a["grasp_rpy_world_deg"] = [180, 0, 0]
        a["collision_capsules"] = mesh_capsules(
            args.project_root / a["urdf"],
            a["axis"],
            tool_radius_mm=45,
            tcp_offset=a["tcp_offset_mm_deg"],
        )
    config = DualConfig.parse(config_raw, args.project_root)
    models = {k: ArmModel(a) for k, a in config.arms.items()}
    poses = {
        k: matrix_pose(
            a.world_from_base @ pose_matrix(models[k].forward(a.home_joints))
        )
        for k, a in config.arms.items()
    }
    for k in poses:
        config_raw["arms"][k]["grasp_rpy_world_deg"] = poses[k][3:].tolist()
    config = DualConfig.parse(config_raw, args.project_root)
    left, right = poses["left"][:3].copy(), poses["right"][:3].copy()
    left[2] -= 25
    right[2] -= 25
    y, x = np.mgrid[:128, :128]
    across = (right - left) / 64
    perpendicular = np.cross(across, [0, 0, 1])
    perpendicular /= np.linalg.norm(perpendicular)
    xyz = left + (x[..., None] - 32) * across + (y[..., None] - 64) * perpendicular
    rgb = np.full((128, 128, 3), 180, dtype=np.uint8)
    rgb[40:88, 16:112] = [60, 130, 210]
    observation = save_observation(
        directory / "observation",
        rgb,
        xyz,
        config,
        {k: a.home_joints for k, a in config.arms.items()},
        synthetic=True,
    )
    proposal = {
        "schema_version": 2,
        "mode": args.mode,
        "pin_arm": "left" if args.mode == "pin_pull" else None,
        "center": {"pixel_xy": [64, 64], "reason": "synthetic fixed center"}
        if args.mode == "center_pair"
        else None,
        "observation_id": observation.meta["observation_id"],
        "grasps": {
            "left": {"pixel_xy": [32, 64], "reason": "synthetic left sample"},
            "right": {"pixel_xy": [96, 64], "reason": "synthetic right sample"},
        },
        "lift_mm": 15,
        "spread_mm": 10,
        "approach_mm": 25,
    }
    write_json(directory / "config.json", config_raw)
    write_json(directory / "proposal.json", proposal)
    return run(
        argparse.Namespace(
            project_root=args.project_root,
            config=directory / "config.json",
            observation=observation.directory,
            proposal=directory / "proposal.json",
            output=directory / "run",
            real=False,
            confirm_real=False,
            preflight_only=False,
            backend="remote",
            host="company-planner",
            mode=args.mode,
        )
    )


def commission(args):
    """Empty-gripper 3 mm world-up lift/return through the actual servo runtime."""
    if args.real != args.confirm_real:
        raise DualArmError("real commissioning requires both --real and --confirm-real")
    config = DualConfig.load(args.config, root=args.project_root)
    if args.real:
        config.require_real(commissioning=True)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    models = {k: ArmModel(a) for k, a in config.arms.items()}
    cancel = threading.Event()
    connections = connect(config, models, cancel, args.real)
    c = DualArmCoordinator(
        config, models, connections, cancel, directory, realtime=args.real
    )
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        cancel.set()
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(signal.SIGTERM, terminate)
    try:
        state = c.snapshots()
        for k, a in config.arms.items():
            if (
                abs(state[k]["gripper"]["position_pulse"] - a.gripper.gripper_open)
                > a.gripper.gripper_open_tolerance_pulse
            ):
                raise DualArmError(
                    "commissioning requires already open, empty grippers"
                )
        initial = {k: np.asarray(v["joints"]) for k, v in state.items()}
        poses = {
            k: matrix_pose(
                config.arms[k].world_from_base @ pose_matrix(models[k].forward(q))
            )
            for k, q in initial.items()
        }
        targets = {k: p + np.array([0, 0, 3, 0, 0, 0]) for k, p in poses.items()}
        program = compile_program(
            [
                Phase("commission_lift", "move", targets),
                Phase("commission_return", "move", poses),
            ],
            initial,
            config,
            models,
            {k: a.inverse for k, a in connections.items()},
        )
        write_preview(program, config, models, directory / "preview.html")
        write_json(directory / "config.json", config.raw)
        if args.preflight_only:
            write_json(
                directory / "execution.json",
                {"status": "PREFLIGHT_ONLY", "physical_execution": False},
            )
            return 0
        result = c.execute(
            program,
            initial,
            None,
            None,
            confirmed=args.confirm_real,
            commissioning=True,
        )
        ticks = [
            json.loads(line)
            for line in (directory / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        ticks = [e for e in ticks if e["event"] == "servo_tick"]
        write_json(
            directory / "commissioning.json",
            {
                "status": result["status"],
                "physical_execution": args.real,
                "config_sha256": digest(config.raw),
                "delta_world_z_mm": 3,
                "max_host_dispatch_skew_s": max(
                    (e["dispatch_skew_s"] for e in ticks), default=None
                ),
                "samples": len(ticks),
                "note": "Host timing and measured tracking checks; not hardware synchronization certification.",
            },
        )
        print(f"{result['status']}: {directory}", flush=True)
        return 0 if result["status"] == "COMPLETED" else 1
    except BaseException as exc:
        c.stop_all()
        write_json(
            directory / "failure.json", {"error": f"{type(exc).__name__}: {exc}"}
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        c.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser(
        "init-config",
        help="create an incomplete hardware config from saved dual-arm identities",
    )
    init.add_argument(
        "--home", type=Path, default=ROOT / "data/robot/dual_arm_home.json"
    )
    init.add_argument("--output", type=Path, required=True)
    fit = sub.add_parser(
        "fit-base",
        help="fit world_from_base using measured correspondences and held-out points",
    )
    fit.add_argument("--points", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    envelopes = sub.add_parser(
        "envelopes", help="generate conservative URDF link envelopes for review"
    )
    envelopes.add_argument("--urdf", type=Path, required=True)
    envelopes.add_argument("--axis", type=int, choices=(6, 7), required=True)
    envelopes.add_argument("--tool-radius-mm", type=float, required=True)
    envelopes.add_argument("--tcp-offset-mm-deg", nargs=6, type=float, required=True)
    envelopes.add_argument("--output", type=Path, required=True)
    plan = sub.add_parser(
        "plan", help="Claude selects two grasp pixels in an existing observation"
    )
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--observation", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--backend", choices=("local", "remote"), default="remote")
    plan.add_argument("--host", default="company-planner")
    plan.add_argument(
        "--mode", choices=("pin_pull", "center_pair"), default="center_pair"
    )
    execute = sub.add_parser(
        "run", help="compile/preview/simulate or explicitly execute a paired task"
    )
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--observation", type=Path)
    execute.add_argument("--proposal", type=Path)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--backend", choices=("local", "remote"), default="remote")
    execute.add_argument("--host", default="company-planner")
    execute.add_argument("--real", action="store_true")
    execute.add_argument("--confirm-real", action="store_true")
    execute.add_argument("--preflight-only", action="store_true")
    execute.add_argument(
        "--mode", choices=("pin_pull", "center_pair"), default="center_pair"
    )
    synthetic = sub.add_parser(
        "demo", help="offline synthetic example using both actual URDF models"
    )
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument(
        "--mode", choices=("pin_pull", "center_pair"), default="center_pair"
    )
    commissioning = sub.add_parser(
        "commission",
        help="paired empty-gripper 3 mm servo commissioning, simulated by default",
    )
    commissioning.add_argument("--config", type=Path, required=True)
    commissioning.add_argument("--output", type=Path, required=True)
    commissioning.add_argument("--real", action="store_true")
    commissioning.add_argument("--confirm-real", action="store_true")
    commissioning.add_argument("--preflight-only", action="store_true")
    home = sub.add_parser("home", help="collision-checked joint Home for BOTH arms, then open grippers")
    home.add_argument("--config", type=Path, default=ROOT / "config/dual_arm.local.json")
    from datetime import datetime, timezone
    home.add_argument("--output", type=Path, default=ROOT / "results/dual_home" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    home.add_argument("--real", action="store_true")
    home.add_argument("--confirm-real", action="store_true")
    home.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    args.project_root = args.project_root.resolve()
    try:
        if args.command == "init-config":
            exclusive_json(args.output, initial_config(args.project_root, args.home))
        elif args.command == "fit-base":
            exclusive_json(args.output, fit_base(json.loads(args.points.read_text())))
        elif args.command == "envelopes":
            from .config import number

            number(args.tool_radius_mm, "tool radius", 1, 500)
            exclusive_json(
                args.output,
                mesh_capsules(
                    args.urdf,
                    args.axis,
                    tool_radius_mm=args.tool_radius_mm,
                    tcp_offset=args.tcp_offset_mm_deg,
                ),
            )
        elif args.command == "plan":
            config = DualConfig.load(args.config, root=args.project_root)
            observation = Observation.load(args.observation)
            observation.validate_for(config)
            args.output.mkdir(parents=True, exist_ok=False)
            proposal = VisionPlanner(args.backend, host=args.host).plan(
                observation, config, args.output / "claude", mode=args.mode
            )
            ground_targets(proposal, observation, config)
            exclusive_json(args.output / "proposal.json", proposal)
        elif args.command == "run":
            return run(args)
        elif args.command == "demo":
            return demo(args)
        elif args.command == "commission":
            return commission(args)
        elif args.command == "home":
            from .homing import run_home
            return run_home(args)
        return 0
    except (
        DualArmError,
        ValueError,
        KeyError,
        OSError,
        RuntimeError,
        ValidationError,
    ) as exc:
        print(f"DUAL ARM FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
