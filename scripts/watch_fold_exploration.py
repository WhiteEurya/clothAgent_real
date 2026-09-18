#!/usr/bin/env python3
"""Restart fold exploration runs while carrying forward their experience.

The fold pipeline already keeps a run-local ``workspace/fold_experience``
store.  This watchdog adds a process boundary around it: when a child exits
after a safe, non-physical failure, the next child receives the previous
experience directory through ``--experience-dir``.  Physical execution
failures and explicit interrupts remain stop conditions because the robot's
state may be unknown.

Usage (pipeline arguments may be passed directly; ``--`` is optional)::

    bash scripts/start_fold_exploration_watchdog.sh

The default child mode is an unattended, continuous real run with Viser
enabled.  Use ``--dry-run``, ``--no-unattended``, or ``--no-viser`` when one of
those defaults should be disabled for a launch.

``--max-restarts 0`` means unlimited safe restarts.  A clean COMPLETE,
BLOCKED, or SUPERVISOR_STOPPED child exits the watchdog.  Use
``--restart-on-max-iterations`` only when a finite child limit is deliberately
being used as a repeated experiment.  When ``--viser`` is forwarded, each
child gets a different port by default because the read-only viewer is allowed
to remain open for inspecting the previous run.

The default real mode is intentional for this watchdog's overnight use. Pass
``--dry-run``/``--no-real`` before the optional ``--`` separator to launch the
same continuous loop without physical robot commands; ``--no-unattended`` and
``--no-viser`` similarly disable the other default child flags.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloth_agent.run_storage import new_run_path

TERMINAL_STATUSES = frozenset({"COMPLETE", "BLOCKED", "SUPERVISOR_STOPPED"})
FORBIDDEN_CHILD_OPTIONS = frozenset({"--run-id", "--run-dir", "--experience-dir"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _write_event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": _now(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    compact = ", ".join(f"{key}={value!r}" for key, value in fields.items())
    print(f"[fold-watchdog] {event}" + (f" | {compact}" if compact else ""), flush=True)


def _latest_summary(run_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    candidates = [
        path
        for path in (run_dir / "results" / "fold_exploration").glob("*/summary.json")
        if path.is_file()
    ]
    if not candidates:
        return None, None
    path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, path
    return (value if isinstance(value, dict) else None), path


def _latest_debug_events(run_dir: Path) -> list[dict[str, Any]]:
    paths = [
        path
        for path in (run_dir / "results" / "fold_exploration").glob("*/debug_events.jsonl")
        if path.is_file()
    ]
    if not paths:
        return []
    path = max(paths, key=lambda item: item.stat().st_mtime_ns)
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    except OSError:
        return rows
    return rows


def _last_debug_event(run_dir: Path) -> dict[str, Any] | None:
    events = _latest_debug_events(run_dir)
    return events[-1] if events else None


def _contains_physical_failure(text: str, *, last_event: dict[str, Any] | None) -> bool:
    """Conservatively detect errors after a fold command may have been sent."""

    lowered = text.lower()
    # These markers identify the execution/robot boundary.  In particular,
    # plain ``controller IK rejected`` is intentionally absent: that is a
    # read-only preflight failure and is safe for the watchdog to restart.
    unsafe_tokens = (
        "graspcheckpointrejected",
        "recoveryexhausted",
        "gripper completion unconfirmed",
        "operator interrupted",
        "robotexecutionerror",
        "robot execution",
        "execution failed",
        "hardware error",
        "emergency stop",
        "e-stop",
        "servo error",
        "motion command failed",
        "return_home failed",
        "xarm execution",
    )
    if any(token in lowered for token in unsafe_tokens):
        return True
    if isinstance(last_event, dict):
        stage = str(last_event.get("stage", "")).lower()
        message = str(last_event.get("message", "")).lower()
        if stage == "execution" and any(
            token in message
            for token in ("failed", "error", "exception", "aborted", "unknown")
        ):
            return True
    return False


def classify_child_exit(
    returncode: int,
    summary: dict[str, Any] | None,
    last_event: dict[str, Any] | None,
    *,
    restart_on_max_iterations: bool = False,
) -> str:
    """Return ``STOP`` or ``RESTART`` for one child process exit."""

    if returncode < 0 or returncode in {2, 130}:
        return "STOP"
    status = str(summary.get("status", "")) if isinstance(summary, dict) else ""
    error = str(summary.get("error", "")) if isinstance(summary, dict) else ""
    if isinstance(summary, dict) and summary.get("restart_safe") is False:
        return "STOP"
    event_text = ""
    if isinstance(last_event, dict):
        event_text = " ".join(
            str(last_event.get(key, ""))
            for key in ("stage", "message", "exception_type")
        )
    if _contains_physical_failure(f"{error} {event_text}", last_event=last_event):
        return "STOP"
    if returncode != 0 and isinstance(last_event, dict):
        # A nonzero process exit while the latest recorded phase is execution
        # leaves the physical state unknown, even if the exception text was
        # truncated before it reached summary.json.
        if str(last_event.get("stage", "")).lower() == "execution":
            return "STOP"
    if status in TERMINAL_STATUSES:
        return "STOP"
    if status == "MAX_ITERATIONS_REACHED" and not restart_on_max_iterations:
        return "STOP"
    # A missing summary, FAILED status, RUNNING status, or unexpected nonzero
    # exit is treated as a safe process-level failure unless physical markers
    # above indicate that motion may have happened.
    return "RESTART"


def _option_name(token: str) -> str:
    return token.split("=", 1)[0]


def _validate_child_args(child_args: Sequence[str]) -> None:
    for token in child_args:
        if _option_name(token) in FORBIDDEN_CHILD_OPTIONS:
            raise ValueError(
                f"{token!r} is managed by the watchdog; pass experience/run identity "
                "to the watchdog instead of the child pipeline"
            )


def _has_option(child_args: Sequence[str], name: str) -> bool:
    return any(_option_name(token) == name for token in child_args)


def _prepare_default_child_args(
    child_args: Sequence[str],
    *,
    real: bool = True,
    unattended: bool = True,
    viser: bool = True,
) -> list[str]:
    """Apply the watchdog's safe-to-override night-run defaults.

    The child pipeline still owns the actual options and validation.  This
    helper only supplies the repetitive launch flags and removes an explicitly
    disabled default so ``--dry-run``/``--no-*`` have predictable semantics.
    User-provided values always win (for example a finite ``--max-iterations``
    or a custom ``--viser-port``).
    """

    result = list(child_args)

    def remove_options(names: set[str]) -> None:
        nonlocal result
        filtered: list[str] = []
        skip_next = False
        takes_value = {"--viser-port", "--viser-refresh-s"}
        for token in result:
            if skip_next:
                skip_next = False
                continue
            option = _option_name(token)
            if option in names:
                if option in takes_value and "=" not in token:
                    skip_next = True
                continue
            filtered.append(token)
        result = filtered

    if not real:
        remove_options({"--real", "--confirm-real"})
    elif not _has_option(result, "--real"):
        result.append("--real")
    if real and not _has_option(result, "--confirm-real"):
        result.append("--confirm-real")

    if not unattended:
        remove_options({"--unattended", "--continue-on-error"})
    elif not _has_option(result, "--unattended") and not _has_option(
        result, "--continue-on-error"
    ):
        result.append("--unattended")

    if not _has_option(result, "--max-iterations"):
        result.extend(["--max-iterations", "0"])

    if not viser:
        remove_options({"--viser", "--viser-port", "--viser-refresh-s"})
    elif not _has_option(result, "--viser") and not _has_option(result, "--no-viser"):
        result.append("--viser")
    return result


def _safe_run_prefix(value: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return prefix or "fold_night"


def _experience_dir(run_dir: Path) -> Path | None:
    path = run_dir / "workspace" / "fold_experience"
    if (path / "experiences.jsonl").is_file():
        return path
    return None


def _snapshot_experience(source: Path | None, destination: Path) -> Path | None:
    """Copy the previous ledger for audit before it is used by the next run."""

    if source is None or not source.is_dir():
        return None
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("experiences.jsonl", "experience_summary.json", "garment_condition.json"):
        source_file = source / name
        if source_file.is_file():
            shutil.copy2(source_file, destination / name)
    return destination


def _default_python() -> str:
    configured = os.environ.get("PYTHON")
    if configured:
        return configured
    preferred = Path("/home/CNS2026330003/miniconda3/envs/cali/bin/python")
    return str(preferred) if preferred.is_file() else sys.executable


def _terminate_child(process: subprocess.Popen[Any], sig: signal.Signals = signal.SIGINT) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, OSError):
        try:
            process.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


def _wait_for_child_shutdown(
    process: subprocess.Popen[Any],
    *,
    interrupt_grace_s: float = 8.0,
    terminate_grace_s: float = 5.0,
) -> None:
    """Stop a child process group without leaving Claude/Viser descendants."""

    if process.poll() is not None:
        return
    _terminate_child(process, signal.SIGINT)
    try:
        process.wait(timeout=max(0.1, float(interrupt_grace_s)))
        return
    except subprocess.TimeoutExpired:
        pass
    _terminate_child(process, signal.SIGTERM)
    try:
        process.wait(timeout=max(0.1, float(terminate_grace_s)))
        return
    except subprocess.TimeoutExpired:
        pass
    _terminate_child(process, signal.SIGKILL)
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        # The process group may have disappeared while the leader was being
        # reaped.  There is no safe further action from this parent.
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restart safe fold-exploration child runs while carrying forward experience."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--python", dest="python_bin", default=_default_python())
    parser.add_argument("--run-prefix", default="fold_night")
    parser.add_argument("--restart-delay-s", type=float, default=10.0)
    parser.add_argument(
        "--viser-port-base",
        type=int,
        default=8765,
        help="base port for child Viser viewers; each restart gets the next port unless --viser-port is forwarded",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=0,
        help="maximum safe restarts after the first child; 0 means unlimited",
    )
    parser.add_argument(
        "--restart-on-max-iterations",
        action="store_true",
        help="restart after a child reports MAX_ITERATIONS_REACHED",
    )
    parser.add_argument(
        "--watchdog-dir",
        type=Path,
        help="directory for watchdog events and experience snapshots",
    )
    parser.add_argument(
        "--dry-run",
        "--no-real",
        dest="dry_run",
        action="store_true",
        help="disable the default --real/--confirm-real child mode",
    )
    parser.add_argument(
        "--no-unattended",
        action="store_true",
        help="do not inject the default child --unattended flag",
    )
    parser.add_argument(
        "--no-viser",
        action="store_true",
        help="do not inject the default child --viser flag",
    )
    return parser


def run_watchdog(args: argparse.Namespace) -> int:
    root = args.project_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"project root does not exist: {root}")
    child_args = list(getattr(args, "child_args", []))
    if child_args[:1] == ["--"]:
        child_args = child_args[1:]
    _validate_child_args(child_args)
    dry_run = bool(getattr(args, "dry_run", False))
    unattended = not bool(getattr(args, "no_unattended", False))
    viser = not bool(getattr(args, "no_viser", False))
    child_args = _prepare_default_child_args(
        child_args,
        real=not dry_run,
        unattended=unattended,
        viser=viser,
    )
    if args.max_restarts < 0:
        raise ValueError("--max-restarts must be non-negative")
    if args.restart_delay_s < 0:
        raise ValueError("--restart-delay-s must be non-negative")
    if not 1 <= args.viser_port_base <= 65535:
        raise ValueError("--viser-port-base must be between 1 and 65535")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    watchdog_dir = (
        args.watchdog_dir.expanduser().resolve()
        if args.watchdog_dir
        else new_run_path(root, f"_{_safe_run_prefix(args.run_prefix)}_watchdog_{stamp}")
    )
    watchdog_dir.mkdir(parents=True, exist_ok=True)
    events_path = watchdog_dir / "watchdog_events.jsonl"
    snapshots_dir = watchdog_dir / "experience_snapshots"
    _write_event(
        events_path,
        "watchdog_started",
        project_root=str(root),
        watchdog_dir=str(watchdog_dir),
        max_restarts=args.max_restarts,
        restart_delay_s=args.restart_delay_s,
        child_args=child_args,
    )

    # ``start_fold_exploration_watchdog.sh`` execs this process, while every
    # child is intentionally placed in its own session.  Relying only on the
    # default KeyboardInterrupt from ``Popen.wait`` is fragile: Ctrl-C can
    # arrive during a restart delay or while Python is blocked in a syscall.
    # Install an explicit relay so one terminal interrupt stops the active
    # child process group and prevents a fresh restart.
    stop_requested = False
    interrupt_signal: int | None = None
    active_process: subprocess.Popen[Any] | None = None

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested, interrupt_signal
        if not stop_requested:
            interrupt_signal = int(signum)
            stop_requested = True
        process = active_process
        if process is not None and process.poll() is None:
            _terminate_child(process, signal.SIGINT)

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    previous_run: Path | None = None
    restart_count = 0
    try:
        while True:
            if stop_requested:
                _write_event(
                    events_path,
                    "watchdog_stopped",
                    reason="operator_interrupt",
                    signal=interrupt_signal,
                )
                return 130
            child_index = restart_count + 1
            child_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            run_id = f"{_safe_run_prefix(args.run_prefix)}_{child_stamp}"
            inherited = _experience_dir(previous_run) if previous_run is not None else None
            snapshot = _snapshot_experience(
                inherited,
                snapshots_dir / f"before_child_{child_index:04d}",
            )
            command = [
                str(Path(args.python_bin).expanduser()),
                str(root / "scripts" / "claude_fold_exploration.py"),
                "--project-root",
                str(root),
                "--run-id",
                run_id,
            ]
            if inherited is not None:
                command.extend(["--experience-dir", str(inherited)])
            if _has_option(child_args, "--viser") and not _has_option(child_args, "--viser-port"):
                port = args.viser_port_base + child_index - 1
                if port > 65535:
                    raise ValueError("watchdog Viser port range exhausted")
                command.extend(["--viser-port", str(port)])
            command.extend(child_args)
            child_dir = new_run_path(root, run_id)
            _write_event(
                events_path,
                "child_starting",
                child_index=child_index,
                run_id=run_id,
                run_dir=str(child_dir),
                inherited_experience=str(inherited) if inherited else None,
                experience_snapshot=str(snapshot) if snapshot else None,
                command=command,
            )
            process = subprocess.Popen(
                command,
                cwd=root,
                start_new_session=True,
            )
            active_process = process
            try:
                while process.poll() is None:
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        if stop_requested:
                            _wait_for_child_shutdown(process)
                            break
                returncode = process.wait()
            except KeyboardInterrupt:
                # Keep compatibility with platforms that still deliver the
                # default exception despite the explicit signal handler.
                request_stop(signal.SIGINT, None)
                _wait_for_child_shutdown(process)
                return 130
            finally:
                active_process = None

            if stop_requested:
                _write_event(
                    events_path,
                    "watchdog_interrupt",
                    run_id=run_id,
                    signal=interrupt_signal,
                    returncode=returncode,
                )
                _write_event(
                    events_path,
                    "watchdog_stopped",
                    run_id=run_id,
                    reason="operator_interrupt",
                )
                return 130

            summary, summary_path = _latest_summary(child_dir)
            last_event = _last_debug_event(child_dir)
            decision = classify_child_exit(
                returncode,
                summary,
                last_event,
                restart_on_max_iterations=args.restart_on_max_iterations,
            )
            inherited_after = _experience_dir(child_dir)
            _write_event(
                events_path,
                "child_exited",
                child_index=child_index,
                run_id=run_id,
                returncode=returncode,
                decision=decision,
                status=summary.get("status") if isinstance(summary, dict) else None,
                summary=str(summary_path) if summary_path else None,
                latest_debug_stage=(last_event or {}).get("stage") if last_event else None,
                inherited_experience=str(inherited_after) if inherited_after else None,
            )
            if decision == "STOP":
                _write_event(
                    events_path,
                    "watchdog_stopped",
                    run_id=run_id,
                    reason=(summary or {}).get("status") if isinstance(summary, dict) else "child_exit",
                )
                return 0 if returncode == 0 else returncode
            if args.max_restarts and restart_count >= args.max_restarts:
                _write_event(
                    events_path,
                    "restart_limit_reached",
                    max_restarts=args.max_restarts,
                    completed_restarts=restart_count,
                )
                return returncode if returncode != 0 else 1
            previous_run = child_dir
            restart_count += 1
            if args.restart_delay_s:
                _write_event(events_path, "waiting_before_restart", delay_s=args.restart_delay_s)
                deadline = time.monotonic() + float(args.restart_delay_s)
                while not stop_requested:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(1.0, remaining))
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def main(argv: Sequence[str] | None = None) -> int:
    args, child_args = build_parser().parse_known_args(argv)
    # Keep the child command separate from watchdog options while allowing
    # convenient invocation without an explicit ``--`` separator.
    args.child_args = child_args
    return run_watchdog(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
