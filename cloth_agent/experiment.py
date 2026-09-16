"""Validation, preflight, and execution of generated experiment scripts."""

from __future__ import annotations

import ast
import io
import json
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import RobotConfig
from .robot_api import RobotAPI, SimulatedBackend, XArmBackend, validate_controller_trajectory


ALLOWED_ROBOT_CALLS = frozenset(
    {"move", "open_gripper", "close_gripper", "shake", "shake_open", "home"}
)
MAX_ACTIONS_PER_EXPERIMENT = 15


class ExperimentValidationError(ValueError):
    """Raised for a hard generated-code contract violation."""


@dataclass(frozen=True)
class Preflight:
    source_path: Path
    source: str
    actions: list[dict[str, Any]]
    stdout: str
    error: str | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class _SafeExperimentValidator(ast.NodeVisitor):
    """Small AST contract; reject hard failures before any execution."""

    _allowed_expr = (
        ast.Constant,
        ast.Name,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )

    def __init__(self) -> None:
        self.assigned: set[str] = set()
        self.errors: list[str] = []
        self.function_count = 0
        self.call_count = 0

    def fail(self, message: str) -> None:
        self.errors.append(message)

    def visit_Module(self, node: ast.Module) -> None:
        for statement in node.body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                continue
            if isinstance(statement, ast.FunctionDef) and statement.name == "run":
                self.function_count += 1
                if (
                    statement.decorator_list
                    or statement.returns is not None
                    or statement.type_comment is not None
                    or statement.args.args
                    or statement.args.kwonlyargs
                    or statement.args.vararg
                    or statement.args.kwarg
                ):
                    self.fail("run() must take no arguments and have no decorators, annotations, or type comments")
                for child in statement.body:
                    self.visit(child)
                continue
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                self.fail("imports are forbidden; generated code receives only the restricted RobotAPI")
            else:
                self.fail(f"only a single run() function is allowed at module scope (got {type(statement).__name__})")
        if self.function_count != 1:
            self.fail("generated script must define exactly one run() function")
        if self.call_count == 0:
            self.fail("generated experiment must contain at least one robot action")
        if self.call_count > MAX_ACTIONS_PER_EXPERIMENT:
            self.fail(f"generated experiment exceeds the {MAX_ACTIONS_PER_EXPERIMENT}-action safety limit")

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return
        if not isinstance(node.value, ast.Call):
            self.fail(f"unsupported expression: {type(node.value).__name__}")
            return
        self.visit(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name) or node.targets[0].id.startswith("_"):
            self.fail("only simple non-private variable assignments are allowed")
            return
        self.assigned.add(node.targets[0].id)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.fail("annotations are not allowed in generated experiments")

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            self.fail("run() must not return a value")

    def visit_Pass(self, node: ast.Pass) -> None:
        return None

    def visit_Call(self, node: ast.Call) -> None:
        self.call_count += 1
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_ROBOT_CALLS:
            self.fail(
                "only move/open_gripper/close_gripper/shake/shake_open/home may be called"
            )
            return
        if node.keywords:
            self.fail(f"{node.func.id} does not accept keyword arguments")
        expected = 4 if node.func.id == "move" else 0
        if len(node.args) != expected:
            self.fail(f"{node.func.id} expects {expected} positional arguments")
        for argument in node.args:
            self.visit(argument)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id not in self.assigned and node.id not in ALLOWED_ROBOT_CALLS:
            self.fail(f"unknown name {node.id!r}; imports and external state are forbidden")

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, self._allowed_expr):
            if isinstance(node, ast.Name):
                self.visit_Name(node)
            elif isinstance(node, ast.BinOp):
                self.visit(node.left)
                self.visit(node.op)
                self.visit(node.right)
            elif isinstance(node, ast.UnaryOp):
                self.visit(node.operand)
                self.visit(node.op)
            return
        if isinstance(node, (ast.operator, ast.unaryop)):
            if not isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.USub, ast.UAdd)):
                self.fail(f"operator {type(node).__name__} is not allowed")
            return
        if isinstance(node, (ast.FunctionDef, ast.Expr, ast.Assign, ast.AnnAssign, ast.Return, ast.Pass, ast.Call, ast.Name, ast.Constant, ast.Module)):
            super().generic_visit(node)
            return
        self.fail(f"syntax {type(node).__name__} is not allowed in an experiment")


def validate_experiment_source(source: str, source_path: Path | None = None) -> ast.Module:
    try:
        tree = ast.parse(source, filename=str(source_path or "experiment.py"), mode="exec")
    except SyntaxError as exc:
        raise ExperimentValidationError(f"syntax error: {exc}") from exc
    validator = _SafeExperimentValidator()
    validator.visit(tree)
    if validator.errors:
        raise ExperimentValidationError("; ".join(dict.fromkeys(validator.errors)))
    return tree


def _execute(
    source: str,
    source_path: Path,
    robot: RobotAPI,
    *,
    action_callback: Callable[[int, Mapping[str, Any]], None] | None = None,
) -> tuple[str, str | None]:
    tree = validate_experiment_source(source, source_path)

    def invoke(name: str, method: Callable[..., Any], *args: Any) -> Any:
        result = method(*args)
        if action_callback is not None:
            actions = robot.action_dicts()
            if actions:
                action_callback(len(actions) - 1, actions[-1])
        return result

    namespace: dict[str, Any] = {
        "__builtins__": {},
        "move": lambda x, y, z, yaw: invoke("move", robot.move, x, y, z, yaw),
        "open_gripper": lambda: invoke("open_gripper", robot.open_gripper),
        "close_gripper": lambda: invoke("close_gripper", robot.close_gripper),
        "shake": lambda: invoke("shake", robot.shake),
        "shake_open": lambda: invoke("shake_open", robot.shake_open),
        "home": lambda: invoke("home", robot.home),
    }
    stdout, stderr = io.StringIO(), io.StringIO()
    error: str | None = None
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(tree, str(source_path), "exec"), namespace, namespace)
            namespace["run"]()
    except KeyboardInterrupt:
        # Ctrl-C is a control signal, not a rollout result.  Let the caller
        # close the backend and perform its mandatory return-Home cleanup
        # instead of converting the interrupt into a normal failed experiment.
        raise
    except BaseException as exc:  # record the failure and stop; never retry here
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=stderr)
    output = stdout.getvalue()
    if stderr.getvalue():
        output += ("\n" if output else "") + stderr.getvalue()
    return output, error


def _execute_action_sequence(
    robot: RobotAPI,
    actions: Sequence[Mapping[str, Any]],
) -> None:
    """Execute an already validated action slice on one persistent backend."""

    for action in actions:
        name = str(action["name"])
        args = action.get("args", {})
        if name == "move":
            robot.move(
                float(args["x"]),
                float(args["y"]),
                float(args["z"]),
                float(args["yaw"]),
            )
        elif name == "open_gripper":
            robot.open_gripper()
        elif name == "close_gripper":
            robot.close_gripper()
        elif name == "shake":
            robot.shake()
        elif name == "shake_open":
            robot.shake_open()
        elif name == "home":
            robot.home()
        else:  # pragma: no cover - guarded by source validation
            raise ExperimentValidationError(f"unsupported checkpoint action {name!r}")


def format_action_sequence(actions: list[dict[str, Any]]) -> str:
    lines = ["Planned action sequence (no physical motion yet):"]
    for index, action in enumerate(actions, start=1):
        args = action.get("args", {})
        if action["name"] == "move":
            lines.append(
                f"  {index}. move(x={args['x']:.3f}, y={args['y']:.3f}, z={args['z']:.3f}, yaw={args['yaw']:.3f})"
            )
        elif action["name"] == "shake":
            from .shake_once import (
                FLICK_ACCELERATION_MM_S2,
                FLICK_SPEED_MM_S,
                SHAKE_AMPLITUDE_MM,
                SHAKE_CYCLES,
            )

            lines.append(
                f"  {index}. shake(cycles={SHAKE_CYCLES}, "
                f"y=±{SHAKE_AMPLITUDE_MM:.1f} mm, "
                f"speed={FLICK_SPEED_MM_S:.1f} mm/s, "
                f"acceleration={FLICK_ACCELERATION_MM_S2:.1f} mm/s^2)"
            )
        elif action["name"] == "shake_open":
            from .shake_open_test import (
                DIAGONAL_ACCELERATION_MM_S2,
                DIAGONAL_CYCLES,
                DIAGONAL_SPEED_MM_S,
                VERTICAL_SNAP_CYCLES,
            )

            lines.append(
                f"  {index}. shake_open(vertical_snaps={VERTICAL_SNAP_CYCLES}, "
                f"diagonal_cycles={DIAGONAL_CYCLES}, "
                f"diagonal_speed={DIAGONAL_SPEED_MM_S:.1f} mm/s, "
                f"diagonal_acceleration={DIAGONAL_ACCELERATION_MM_S2:.1f} mm/s^2)"
            )
        else:
            lines.append(f"  {index}. {action['name']}()")
    return "\n".join(lines)


def format_speed_profile(config: RobotConfig) -> str:
    return (
        "Low-speed execution profile:\n"
        f"  Cartesian: speed={config.speed_mm_s:.1f} mm/s, "
        f"acceleration={config.acceleration_mm_s2:.1f} mm/s^2\n"
        f"  Home joints: speed={config.home_speed_deg_s:.1f} deg/s, "
        f"acceleration={config.home_acceleration_deg_s2:.1f} deg/s^2\n"
        f"  Gripper speed: {config.gripper_speed:.1f}"
    )


class ExperimentRunner:
    def __init__(self, run_dir: Path, config: RobotConfig):
        self.run_dir = run_dir.resolve()
        self.workspace = (self.run_dir / "workspace").resolve()
        self.results_dir = (self.run_dir / "results").resolve()
        self.config = config
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _safe_source_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        candidate = candidate.resolve()
        if candidate.parent != self.workspace or candidate.suffix != ".py":
            raise ExperimentValidationError("experiment path must be a .py file directly inside the current run workspace")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    def preflight(self, path: str | Path) -> Preflight:
        source_path = self._safe_source_path(path)
        source = source_path.read_text(encoding="utf-8")
        robot = RobotAPI(self.config, SimulatedBackend(self.config))
        try:
            output, error = _execute(source, source_path, robot)
        except ExperimentValidationError as exc:
            output, error = "", f"ExperimentValidationError: {exc}"
        robot.close()
        return Preflight(source_path, source, robot.action_dicts(), output, error)

    def run_experiment(
        self,
        path: str | Path,
        *,
        real: bool = False,
        confirmed: bool = False,
        notes: str = "",
        action_callback: Callable[[int, Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        preflight = self.preflight(path)
        print(preflight.stdout, end="" if preflight.stdout.endswith("\n") or not preflight.stdout else "\n")
        print(format_speed_profile(self.config), flush=True)
        print(format_action_sequence(preflight.actions), flush=True)
        if preflight.error:
            result = self._result(preflight, physical=False, completed=False, notes=notes, error=preflight.error)
            self._save_result(result)
            raise ExperimentValidationError(preflight.error)
        if real and not confirmed:
            raise PermissionError("physical execution requires explicit confirmation after reviewing the printed action sequence")
        if real:
            self.config.validate_for_real()
            try:
                validate_controller_trajectory(self.config, preflight.actions)
                backend = XArmBackend(self.config)
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                result = self._result(
                    preflight,
                    physical=True,
                    completed=False,
                    notes=notes,
                    error=error,
                    actual_actions=[],
                )
                self._save_result(result)
                raise
        else:
            backend = SimulatedBackend(self.config)
        robot = RobotAPI(self.config, backend)
        interrupted: KeyboardInterrupt | None = None
        try:
            output, error = _execute(
                preflight.source_path.read_text(encoding="utf-8"),
                preflight.source_path,
                robot,
                action_callback=action_callback,
            )
        except KeyboardInterrupt as exc:
            # Persist the partial action trace before re-raising.  AgentSession
            # will then attempt its mandatory post-rollout Home in ``finally``.
            interrupted = exc
            output = ""
            error = "KeyboardInterrupt: rollout interrupted by operator"
        result = self._result(
            preflight,
            physical=real,
            completed=error is None,
            notes=notes,
            error=error,
            actual_actions=robot.action_dicts(),
            stdout=output,
        )
        try:
            self._save_result(result)
        finally:
            robot.close()
        if interrupted is not None:
            raise interrupted
        return result

    def run_checkpointed_experiment(
        self,
        path: str | Path,
        *,
        checkpoint_action_index: int,
        checkpoint_action_indices: Sequence[int] | None = None,
        abort_actions: Sequence[Mapping[str, Any]],
        checkpoint_callback: Callable[..., Mapping[str, Any]],
        real: bool = False,
        confirmed: bool = False,
        notes: str = "",
    ) -> dict[str, Any]:
        """Execute one or more checkpoints, then continue or release on one backend."""

        preflight = self.preflight(path)
        print(preflight.stdout, end="" if preflight.stdout.endswith("\n") or not preflight.stdout else "\n")
        print(format_speed_profile(self.config), flush=True)
        print(format_action_sequence(preflight.actions), flush=True)
        if preflight.error:
            result = self._result(
                preflight,
                physical=False,
                completed=False,
                notes=notes,
                error=preflight.error,
            )
            self._save_result(result)
            raise ExperimentValidationError(preflight.error)
        indices = (
            tuple(int(index) for index in checkpoint_action_indices)
            if checkpoint_action_indices is not None
            else (int(checkpoint_action_index),)
        )
        if not indices:
            raise ExperimentValidationError(
                "checkpoint_action_indices must contain at least one action index"
            )
        if any(
            index < 0 or index >= len(preflight.actions)
            for index in indices
        ):
            raise ExperimentValidationError(
                "checkpoint action index is outside the validated action sequence"
            )
        if tuple(sorted(set(indices))) != indices:
            raise ExperimentValidationError(
                "checkpoint action indices must be strictly increasing"
            )
        final_checkpoint_index = indices[-1]
        acquisition = preflight.actions[: final_checkpoint_index + 1]
        continuation = preflight.actions[final_checkpoint_index + 1 :]
        clean_abort = [
            {"name": str(action["name"]), "args": dict(action.get("args", {}))}
            for action in abort_actions
        ]
        abort_path = acquisition + clean_abort
        if len(abort_path) > MAX_ACTIONS_PER_EXPERIMENT:
            raise ExperimentValidationError(
                "checkpoint abort path exceeds the experiment action limit"
            )

        # Run the alternate path through the exact same restricted simulator and
        # source validator before any physical command. The original continuation
        # was already checked by ``preflight``.
        abort_source = "def run():\n" + "".join(
            (
                f"    move({action['args']['x']!r}, {action['args']['y']!r}, "
                f"{action['args']['z']!r}, {action['args']['yaw']!r})\n"
                if action["name"] == "move"
                else f"    {action['name']}()\n"
            )
            for action in abort_path
        )
        abort_robot = RobotAPI(self.config, SimulatedBackend(self.config))
        try:
            _, abort_error = _execute(
                abort_source,
                preflight.source_path.with_name(preflight.source_path.stem + "_abort.py"),
                abort_robot,
            )
        finally:
            abort_robot.close()
        if abort_error:
            raise ExperimentValidationError(
                f"checkpoint abort path failed preflight: {abort_error}"
            )
        if real and not confirmed:
            raise PermissionError(
                "physical execution requires explicit confirmation after reviewing the action sequence"
            )
        if real:
            self.config.validate_for_real()
            try:
                validate_controller_trajectory(self.config, preflight.actions)
                validate_controller_trajectory(self.config, abort_path)
                backend = XArmBackend(self.config)
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                result = self._result(
                    preflight,
                    physical=True,
                    completed=False,
                    notes=notes,
                    error=error,
                    actual_actions=[],
                )
                result["checkpoint"] = {
                    "status": "NOT_REACHED",
                    "error": error,
                }
                self._save_result(result)
                raise
        else:
            backend = SimulatedBackend(self.config)

        robot = RobotAPI(self.config, backend)
        checkpoint: dict[str, Any] = {"status": "NOT_REACHED"}
        error: str | None = None
        interrupted: KeyboardInterrupt | None = None
        emergency_cleanup: dict[str, Any] | None = None
        try:
            multi_checkpoint = len(indices) > 1
            if multi_checkpoint:
                segment_start = 0
                checkpoint_records: list[dict[str, Any]] = []
                for checkpoint_number, index in enumerate(indices, start=1):
                    _execute_action_sequence(
                        robot,
                        preflight.actions[segment_start : index + 1],
                    )
                    try:
                        raw_decision = checkpoint_callback(checkpoint_number)
                        checkpoint_records.append(dict(raw_decision))
                    except BaseException as exc:
                        checkpoint_records.append(
                            {
                                "status": "FAILED_CLOSED",
                                "classification": "UNKNOWN",
                                "continue_transport": False,
                                "runtime_decision": "ABORT_RELEASE",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    segment_start = index + 1
                checkpoint = {
                    "mode": "LIFT_CHECKPOINT_EXPERIMENT",
                    "status": "COMPLETED",
                    "checkpoints": checkpoint_records,
                    "continue_transport": False,
                    "runtime_decision": "REVERSE_RELEASE",
                    "executed_branch": "REVERSE_RELEASE",
                }
                _execute_action_sequence(robot, clean_abort)
            else:
                _execute_action_sequence(robot, acquisition)
                try:
                    raw_decision = checkpoint_callback()
                    checkpoint = dict(raw_decision)
                except BaseException as exc:
                    checkpoint = {
                        "status": "FAILED_CLOSED",
                        "classification": "UNKNOWN",
                        "continue_transport": False,
                        "runtime_decision": "ABORT_RELEASE",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                continue_transport = bool(
                    checkpoint.get("continue_transport")
                    and checkpoint.get("classification")
                    == "INDEPENDENT_LAYER_SUPPORTED"
                )
                checkpoint["continue_transport"] = continue_transport
                checkpoint["executed_branch"] = (
                    "CONTINUATION" if continue_transport else "ABORT_RELEASE"
                )
                _execute_action_sequence(
                    robot,
                    continuation if continue_transport else clean_abort,
                )
        except KeyboardInterrupt as exc:
            interrupted = exc
            error = "KeyboardInterrupt: checkpointed rollout interrupted by operator"
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            gripper_failed = any(
                (a.get('gripper_result') or {}).get('completion', {}).get('status') == 'FAILED'
                for a in robot.action_dicts())
            if error is not None and gripper_failed:
                emergency_cleanup = {'attempted': False, 'released': False,
                    'home_completed': False, 'reason': 'gripper completion unconfirmed; operator inspection required',
                    'errors': []}
            elif error is not None:
                emergency_cleanup = {
                    "attempted": True,
                    "released": False,
                    "home_completed": False,
                    "errors": [],
                }
                try:
                    backend.open_gripper(self.config)
                    emergency_cleanup["released"] = True
                except BaseException as exc:
                    emergency_cleanup["errors"].append(
                        f"open_gripper: {type(exc).__name__}: {exc}"
                    )
                    emergency_cleanup['gripper_completion_failed'] = True
                    emergency_cleanup['gripper_completion'] = getattr(exc, 'gripper_completion', None)
                else:
                    try:
                        backend.home(self.config)
                        emergency_cleanup["home_completed"] = True
                    except BaseException as exc:
                        emergency_cleanup["errors"].append(
                            f"home: {type(exc).__name__}: {exc}"
                        )

        result = self._result(
            preflight,
            physical=real,
            completed=error is None,
            notes=notes,
            error=error,
            actual_actions=robot.action_dicts(),
        )
        result["checkpoint"] = checkpoint
        result["checkpoint_action_index"] = int(final_checkpoint_index)
        result["checkpoint_action_indices"] = list(indices)
        result["checkpoint_abort_actions"] = clean_abort
        result["emergency_cleanup"] = emergency_cleanup
        if emergency_cleanup and emergency_cleanup.get('gripper_completion_failed'):
            result['gripper_completion_failed'] = True
            self.gripper_completion_failed = True
        try:
            self._save_result(result)
        finally:
            robot.close()
        if interrupted is not None:
            raise interrupted
        return result

    def _result(self, preflight: Preflight, *, physical: bool, completed: bool, notes: str, error: str | None, actual_actions: list[dict[str, Any]] | None = None, stdout: str | None = None) -> dict[str, Any]:
        actions = actual_actions if actual_actions is not None else preflight.actions
        errors = [a["error"] for a in actions if a.get("error")]
        if error and error not in errors:
            errors.append(error)
        gripper_failed = physical and any(
            (a.get('gripper_result') or {}).get('completion', {}).get('status') == 'FAILED'
            for a in actions)
        self.gripper_completion_failed = gripper_failed
        return {
            "created_at": _now(),
            "started_at": actions[0].get("requested_at") if actions else None,
            "completed_at": actions[-1].get("completed_at") if actions else None,
            "experiment": preflight.source_path.stem,
            "source_path": str(preflight.source_path.relative_to(self.run_dir)),
            "experiment_source": preflight.source,
            "physical_execution": physical,
            "preflight_completed": preflight.error is None,
            "execution_completed": completed,
            "gripper_completion_failed": gripper_failed,
            "robot_errors": errors,
            "notes": notes,
            "stdout": stdout if stdout is not None else preflight.stdout,
            "requested_robot_actions": preflight.actions,
            "actual_robot_actions": actions,
            "actual_ee_states": [a.get("actual_ee_pose") for a in actions if a.get("actual_ee_pose") is not None],
            "execution_profile": {
                "cartesian_speed_mm_s": self.config.speed_mm_s,
                "cartesian_acceleration_mm_s2": self.config.acceleration_mm_s2,
                "home_speed_deg_s": self.config.home_speed_deg_s,
                "home_acceleration_deg_s2": self.config.home_acceleration_deg_s2,
                "gripper_speed": self.config.gripper_speed,
                "gripper_completion_timeout_s": self.config.gripper_completion_timeout_s,
                "gripper_open_tolerance_pulse": self.config.gripper_open_tolerance_pulse,
            },
        }

    def _save_result(self, result: dict[str, Any]) -> None:
        stem = result["experiment"]
        path = self.results_dir / f"{stem}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.results_dir / f"{stem}.source.py").write_text(result.get("experiment_source", ""), encoding="utf-8")
        (self.results_dir / f"{stem}.stdout.txt").write_text(result.get("stdout", ""), encoding="utf-8")
        (self.results_dir / f"{stem}.trace.json").write_text(
            json.dumps(result.get("actual_robot_actions", []), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def inspect_result(self, experiment: str | None = None) -> dict[str, Any]:
        files = sorted(self.results_dir.glob("*.json"))
        files = [path for path in files if not path.name.endswith(".trace.json")]
        if experiment:
            normalized = Path(experiment).name
            if normalized.endswith(".py"):
                normalized = normalized[:-3]
            files = [self.results_dir / (normalized if normalized.endswith(".json") else f"{normalized}.json")]
        if not files:
            raise FileNotFoundError("no rollout result has been saved")
        return json.loads(files[-1].read_text(encoding="utf-8"))
