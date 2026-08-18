"""Headless CLI loop for confidence-filtered Molmo garment grasp references.

The loop mirrors the automatic Viser workflow without starting a web server or
browser: synchronized A/B perception, Molmo keypoints and confidence filtering,
Claude visual selection/final grounding, preflight/controller IK, optional real
execution, and before/after evaluation.  Every phase is printed to stdout and
checkpointed under a separate iteration directory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Callable, Sequence

import numpy as np

from .auto_exploration import (
    AutoExplorationError,
    ClaudeAutoClient,
    ExplorationProposal,
    _is_preexecution_replan_error,
    _load_session,
    _save_frame_images,
    grasp_targets_from_actions,
)
from .experiment import ExperimentValidationError
from .free_exploration import exploration_source, perception_image_paths
from .molmo_keypoint_pipeline import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    KeypointSpec,
    load_keypoint_specs,
    run_molmo_keypoint_pipeline,
    validate_confidence_threshold,
)
from .perception import PerceptionConfig, capture_two_view_rgbd
from .robot_api import move_robot_to_perception_position, validate_controller_trajectory
from .session import AgentSession
from .viewer import _load_latest_perception


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


class CliReporter:
    """Print phase output and retain the same messages as JSON Lines."""

    _COLORS = {
        "INFO": "\033[36m",
        "START": "\033[94m",
        "PASS": "\033[92m",
        "DONE": "\033[92m",
        "REJECT": "\033[90m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "WAIT": "\033[95m",
    }
    _SYMBOLS = {
        "INFO": "•",
        "START": "▶",
        "PASS": "✓",
        "DONE": "✓",
        "REJECT": "×",
        "WARNING": "!",
        "ERROR": "✗",
        "WAIT": "…",
    }

    def __init__(
        self,
        events_path: Path,
        stream: Any = sys.stdout,
        *,
        color: bool | None = None,
    ):
        self.events_path = events_path
        self.stream = stream
        events_path.parent.mkdir(parents=True, exist_ok=True)
        self.color = (
            bool(getattr(stream, "isatty", lambda: False)())
            if color is None
            else bool(color)
        )
        self.run_started_monotonic = time.monotonic()
        self._phase_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._current_phase: dict[str, Any] | None = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        minutes, remainder = divmod(seconds, 60.0)
        hours, minutes = divmod(int(minutes), 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{remainder:04.1f}"
        return f"{minutes:02d}:{remainder:04.1f}"

    def banner(self, payload: dict[str, Any]) -> None:
        width = 86
        with self._io_lock:
            print("╭" + "─" * width + "╮", file=self.stream)
            print(
                "│  ClothAgent · Molmo Confidence Keypoint CLI".ljust(width + 1)
                + "│",
                file=self.stream,
            )
            print("├" + "─" * width + "┤", file=self.stream)
            for key, value in payload.items():
                text = f"│  {key:<22} {value}"
                if len(text) > width + 1:
                    text = text[: width - 2] + "..."
                print(text.ljust(width + 1) + "│", file=self.stream)
            print("╰" + "─" * width + "╯", file=self.stream, flush=True)

    def start_heartbeat(self, interval_s: float) -> None:
        if interval_s <= 0 or self._heartbeat_thread is not None:
            return

        def heartbeat() -> None:
            while not self._heartbeat_stop.wait(interval_s):
                with self._phase_lock:
                    current = dict(self._current_phase) if self._current_phase else None
                if current is None:
                    continue
                elapsed = time.monotonic() - float(current["started_monotonic"])
                self.emit(
                    str(current["phase"]),
                    f"still running: {current['activity']} · phase elapsed {self._duration(elapsed)}",
                    iteration=current.get("iteration"),
                    level="WAIT",
                )

        self._heartbeat_thread = threading.Thread(
            target=heartbeat,
            daemon=True,
            name="molmo-keypoint-cli-heartbeat",
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._heartbeat_thread = None

    def start_phase(
        self,
        phase: str,
        activity: str,
        *,
        iteration: int | None = None,
    ) -> None:
        with self._phase_lock:
            self._current_phase = {
                "phase": phase,
                "activity": activity,
                "iteration": iteration,
                "started_monotonic": time.monotonic(),
            }
        self.emit(phase, activity, iteration=iteration, level="START")

    def finish_phase(
        self,
        message: str,
        *,
        success: bool = True,
        level: str | None = None,
        payload: Any | None = None,
    ) -> None:
        with self._phase_lock:
            current = self._current_phase
            self._current_phase = None
        if current is None:
            return
        elapsed = time.monotonic() - float(current["started_monotonic"])
        self.emit(
            str(current["phase"]),
            f"{message} · phase {self._duration(elapsed)}",
            iteration=current.get("iteration"),
            level=level or ("DONE" if success else "ERROR"),
            payload=payload,
        )

    def fail_current_phase(self, message: str) -> None:
        self.finish_phase(message, success=False)

    def emit(
        self,
        phase: str,
        message: str,
        *,
        iteration: int | None = None,
        level: str = "INFO",
        payload: Any | None = None,
    ) -> None:
        event = {
            "created_at": _now(),
            "local_time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "level": level,
            "iteration": iteration,
            "phase": phase,
            "message": message,
            "run_elapsed_s": time.monotonic() - self.run_started_monotonic,
        }
        if payload is not None:
            event["payload"] = _jsonable(payload)
        iteration_text = f"I{iteration:03d}" if iteration else " RUN"
        clock_text = datetime.now().astimezone().strftime("%H:%M:%S")
        elapsed_text = self._duration(float(event["run_elapsed_s"]))
        symbol = self._SYMBOLS.get(level, "•")
        prefix = (
            f"{clock_text}  +{elapsed_text}  {iteration_text}  "
            f"{phase[:20].upper():<20}  {symbol} "
        )
        if self.color:
            color = self._COLORS.get(level, "")
            reset = "\033[0m" if color else ""
            prefix = f"{color}{prefix}{reset}"
        with self._io_lock:
            print(f"{prefix}{message}", file=self.stream, flush=True)
            if payload is not None:
                print(
                    json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
                    file=self.stream,
                    flush=True,
                )
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def worker_line(self, line: str) -> None:
        clock_text = datetime.now().astimezone().strftime("%H:%M:%S")
        elapsed_text = self._duration(time.monotonic() - self.run_started_monotonic)
        prefix = f"{clock_text}  +{elapsed_text}        MOLMO/WORKER          │ "
        with self._io_lock:
            print(f"{prefix}{line}", end="", file=self.stream, flush=True)


@dataclass(frozen=True)
class KeypointCliOptions:
    max_iterations: int | None = 1
    settle_s: float = 2.0
    enable_real: bool = False
    skip_controller_ik: bool = False
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    molmo_python: Path | None = None
    molmo_model: str = "allenai/MolmoPoint-8B"
    keypoint_specs: tuple[KeypointSpec, ...] = ()
    keypoint_cameras: tuple[str, ...] = ("A", "B")
    molmo_timeout_s: int = 900
    molmo_allow_download: bool = False
    min_gpu_free_mib: int = 20_000
    heartbeat_s: float = 10.0
    color: bool | None = None
    claude_binary: str = "claude"
    claude_timeout_s: int = 400
    claude_grounding_timeout_s: int = 120
    max_replans: int = 2
    objective: str = (
        "Take one planning-mode-appropriate action that makes the current garment "
        "as open and spread as safely possible."
    )


def _validate_options(options: KeypointCliOptions) -> KeypointCliOptions:
    validate_confidence_threshold(options.confidence_threshold)
    if options.max_iterations is not None and not 1 <= options.max_iterations <= 100:
        raise ValueError("max_iterations must be 1..100 or None for continuous")
    if not options.enable_real and options.max_iterations is None:
        raise ValueError("continuous CLI mode requires --enable-real")
    if not 0 <= options.settle_s <= 60:
        raise ValueError("settle_s must be between 0 and 60 seconds")
    if options.enable_real and options.skip_controller_ik:
        raise ValueError("controller IK cannot be skipped with --enable-real")
    if not 0 <= options.max_replans <= 5:
        raise ValueError("max_replans must be between 0 and 5")
    if not 30 <= options.claude_timeout_s <= 1200:
        raise ValueError("claude_timeout_s must be between 30 and 1200 seconds")
    if not 15 <= options.claude_grounding_timeout_s <= 400:
        raise ValueError("claude_grounding_timeout_s must be between 15 and 400 seconds")
    if not 30 <= options.molmo_timeout_s <= 3600:
        raise ValueError("molmo_timeout_s must be between 30 and 3600 seconds")
    if not 0 <= options.min_gpu_free_mib <= 24_564:
        raise ValueError("min_gpu_free_mib must be between 0 and 24564")
    if not 0 <= options.heartbeat_s <= 300:
        raise ValueError("heartbeat_s must be between 0 and 300 seconds")
    if not options.keypoint_specs:
        raise ValueError("at least one keypoint spec is required")
    if not options.keypoint_cameras or any(
        camera not in {"A", "B"} for camera in options.keypoint_cameras
    ):
        raise ValueError("keypoint_cameras must contain A and/or B")
    if len(set(options.keypoint_cameras)) != len(options.keypoint_cameras):
        raise ValueError("keypoint_cameras must be unique")
    return options


def probe_gpu_free_mib() -> int:
    """Return current free memory on GPU 0 without allocating CUDA memory."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise AutoExplorationError(
            "GPU memory preflight could not query nvidia-smi: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    lines = completed.stdout.strip().splitlines()
    if not lines:
        raise AutoExplorationError("GPU memory preflight received no nvidia-smi output")
    first_line = lines[0]
    try:
        return int(first_line.strip())
    except ValueError as exc:
        raise AutoExplorationError(
            f"GPU memory preflight received an invalid value: {first_line!r}"
        ) from exc


def _iteration_checkpoint(
    output_dir: Path,
    iteration_dir: Path,
    iteration: int,
    record: dict[str, Any],
) -> None:
    _write_json(iteration_dir / "result.json", record)
    _write_json(output_dir / f"iteration_{iteration:03d}.json", record)


def _record_stage(
    output_dir: Path,
    iteration_dir: Path,
    iteration: int,
    record: dict[str, Any],
    stage: str,
) -> None:
    record["last_completed_stage"] = stage
    record.setdefault("stage_timestamps", {})[stage] = _now()
    _iteration_checkpoint(output_dir, iteration_dir, iteration, record)


def _print_keypoints(
    reporter: CliReporter,
    iteration: int,
    manifest: dict[str, Any],
) -> None:
    reporter.emit(
        "molmo-summary",
        (
            f"status={manifest['status']} accepted="
            f"{manifest['accepted_reference_count']} threshold>"
            f"{manifest['confidence_threshold']:.3f}"
        ),
        iteration=iteration,
    )
    for view in manifest.get("views", []):
        camera = str(view.get("camera", "?"))
        for candidate in view.get("candidates", []):
            accepted = bool(candidate.get("accepted"))
            reference = candidate.get("reference_id", "-") if accepted else "-"
            reason = "accepted" if accepted else candidate.get("rejection_reason", "rejected")
            reporter.emit(
                "molmo-keypoint",
                (
                    f"camera={camera} name={candidate.get('name')} "
                    f"status={candidate.get('status')} "
                    f"confidence={float(candidate.get('confidence', 0.0)):.4f} "
                    f"valid={str(accepted).lower()} reference={reference} reason={reason}"
                ),
                iteration=iteration,
                level="PASS" if accepted else "REJECT",
            )


def _planning_images(
    saved: dict[str, Any],
    saved_path: Path,
    keypoint_manifest: dict[str, Any],
) -> list[Path]:
    uniform_overlays = {
        "camera_A_coordinate_overlay.png",
        "camera_B_coordinate_overlay.png",
    }
    paths = [
        path
        for path in perception_image_paths(saved, saved_path)
        if path.name not in uniform_overlays
    ]
    for view in keypoint_manifest.get("views", []):
        overlay = Path(str(view.get("accepted_overlay", ""))).resolve()
        if overlay.is_file() and overlay not in paths:
            paths.append(overlay)
    return paths


def _replan_feedback(
    exc: BaseException,
    proposal: ExplorationProposal | None,
) -> str:
    lines = [f"Error type: {type(exc).__name__}", f"Error: {exc}"]
    if proposal is not None:
        lines.append("Rejected proposal actions:")
        lines.extend(
            json.dumps(action, ensure_ascii=False) for action in proposal.actions
        )
    lines.append(
        "Generate a materially different proposal that addresses this exact "
        "pre-execution failure and stays inside workspace/IK limits."
    )
    return "\n".join(lines)


def _print_proposal(
    reporter: CliReporter,
    iteration: int,
    client: ClaudeAutoClient,
    proposal: ExplorationProposal,
    source: str,
) -> None:
    visual = client.last_visual_plan_result
    if visual is not None:
        selected = visual.decision.selected_reference
        reporter.emit(
            "claude-visual-plan",
            (
                f"selected={selected['camera']}/{selected['reference_id']} "
                f"confidence={visual.decision.confidence:.3f} "
                f"reason={selected['reason']}"
            ),
            iteration=iteration,
            payload=visual.decision.as_dict(),
        )
    reporter.emit(
        "claude-final-plan",
        f"validated proposal with {len(proposal.actions)} action(s)",
        iteration=iteration,
        payload=proposal.as_dict(),
    )
    reporter.emit(
        "generated-source",
        "restricted RobotAPI source",
        iteration=iteration,
    )
    print(source, file=reporter.stream, flush=True)
    targets = grasp_targets_from_actions(proposal.actions)
    reporter.emit(
        "grasp-targets",
        f"derived {len(targets)} grasp target(s)",
        iteration=iteration,
        payload=targets,
    )


def run_keypoint_cli_loop(
    session: AgentSession,
    perception_config: PerceptionConfig,
    output_dir: Path,
    options: KeypointCliOptions,
    *,
    capture: Callable[[PerceptionConfig], list[Any]] = capture_two_view_rgbd,
    keypoint_runner: Callable[..., dict[str, Any]] = run_molmo_keypoint_pipeline,
    client: ClaudeAutoClient | None = None,
    controller_validator: Callable[..., Any] = validate_controller_trajectory,
    perception_positioner: Callable[..., dict[str, Any]] = (
        move_robot_to_perception_position
    ),
    gpu_memory_probe: Callable[[], int] = probe_gpu_free_mib,
    sleep: Callable[[float], None] = time.sleep,
    stream: Any = sys.stdout,
) -> int:
    """Run and checkpoint the headless keypoint loop."""

    options = _validate_options(options)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"CLI output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    reporter = CliReporter(
        output / "events.jsonl",
        stream=stream,
        color=options.color,
    )
    client = client or ClaudeAutoClient(
        binary=options.claude_binary,
        timeout_s=options.claude_timeout_s,
        grounding_timeout_s=options.claude_grounding_timeout_s,
        max_reference_reselections=options.max_replans,
    )
    summary: dict[str, Any] = {
        "created_at": _now(),
        "status": "RUNNING",
        "run_dir": str(session.run_dir),
        "output_dir": str(output),
        "enable_real": options.enable_real,
        "confidence_threshold": options.confidence_threshold,
        "max_iterations": options.max_iterations,
        "perception_position": {
            "sequence": ["home", "perception_position"],
            "target_joint_angles_deg": (
                list(session.robot_config.perception_joints_deg)
                if session.robot_config.perception_joints_deg is not None
                else None
            ),
            "recorded_tcp_pose_mm_deg": (
                list(session.robot_config.perception_pose_mm_deg)
                if session.robot_config.perception_pose_mm_deg is not None
                else None
            ),
            "enabled": options.enable_real,
        },
        "iterations": [],
    }
    _write_json(output / "summary.json", summary)
    local_now = datetime.now().astimezone()
    reporter.banner(
        {
            "Local start time": local_now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "Mode": "REAL ROBOT" if options.enable_real else "DRY RUN (no motion)",
            "Iterations": options.max_iterations or "continuous",
            "Cameras / keypoints": (
                f"{','.join(options.keypoint_cameras)} / {len(options.keypoint_specs)}"
            ),
            "Confidence gate": f"confidence > {options.confidence_threshold:.3f}",
            "Before each capture": (
                "Home → perception_position → settle"
                if options.enable_real
                else "disabled in dry run"
            ),
            "Heartbeat": (
                f"every {options.heartbeat_s:g}s" if options.heartbeat_s else "disabled"
            ),
            "Results": output,
        }
    )
    reporter.emit(
        "startup",
        (
            "headless Molmo-keypoint loop started; physical execution "
            + ("ENABLED" if options.enable_real else "DISABLED (dry run)")
        ),
        payload={
            "output_dir": str(output),
            "confidence_policy": f"confidence > {options.confidence_threshold}",
            "keypoint_cameras": list(options.keypoint_cameras),
            "keypoint_count": len(options.keypoint_specs),
            "perception_position_sequence": ["home", "perception_position"],
            "perception_position_enabled": options.enable_real,
        },
        level="WARNING" if options.enable_real else "INFO",
    )
    reporter.start_heartbeat(options.heartbeat_s)

    def prepare_real_perception_position(
        iteration_number: int,
        iteration_record: dict[str, Any],
        record_key: str,
    ) -> None:
        if not options.enable_real:
            return
        reporter.start_phase(
            "robot-positioning",
            "moving robot Home → perception_position before RGB-D capture",
            iteration=iteration_number,
        )
        outcome = perception_positioner(session.robot_config)
        iteration_record[record_key] = _jsonable(outcome)
        actual_pose = outcome.get("actual_tcp_pose_mm_deg")
        reporter.finish_phase(
            f"reached perception_position; actual TCP={actual_pose}",
            payload=outcome,
        )
        if options.settle_s > 0:
            reporter.start_phase(
                "camera-settle",
                (
                    f"waiting {options.settle_s:.1f}s after robot motion "
                    "before RGB-D capture"
                ),
                iteration=iteration_number,
            )
            sleep(options.settle_s)
            reporter.finish_phase("camera stabilization interval complete")

    history: list[dict[str, Any]] = []
    objective = options.objective
    iteration = 0
    exit_code = 0
    while options.max_iterations is None or iteration < options.max_iterations:
        iteration += 1
        iteration_dir = output / f"iteration_{iteration:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=False)
        record: dict[str, Any] = {
            "iteration": iteration,
            "started_at": _now(),
            "status": "RUNNING",
            "objective": objective,
            "artifacts": {},
        }
        source_path = session.workspace / "_molmo_keypoint_cli.py"
        try:
            reporter.emit(
                "iteration",
                f"starting iteration; checkpoints: {iteration_dir}",
                iteration=iteration,
                level="START",
            )
            prepare_real_perception_position(
                iteration,
                record,
                "pre_perception_robot_positioning",
            )
            reporter.start_phase(
                "perception",
                "capturing synchronized Camera A/B RGB-D",
                iteration=iteration,
            )
            frames = capture(perception_config)
            perception = session.locate_cloth_center(
                perception_config, frames=frames
            )
            saved, saved_path = _load_latest_perception(session)
            if saved is None or saved_path is None:
                raise AutoExplorationError(
                    "perception completed without a saved result"
                )
            record["perception"] = perception
            record["saved_perception_result"] = str(saved_path)
            _record_stage(
                output, iteration_dir, iteration, record, "PERCEPTION_COMPLETED"
            )
            reporter.finish_phase(
                f"saved dense A/B result: {saved_path}",
                payload={
                    "status": saved.get("status"),
                    "center_base_mm": saved.get("center_base_mm"),
                    "active_cameras": saved.get("active_cameras"),
                },
            )

            if options.min_gpu_free_mib:
                reporter.start_phase(
                    "gpu-preflight",
                    "checking GPU 0 free memory before loading Molmo",
                    iteration=iteration,
                )
                free_mib = int(gpu_memory_probe())
                record["gpu_memory_preflight"] = {
                    "gpu": 0,
                    "free_mib": free_mib,
                    "required_free_mib": options.min_gpu_free_mib,
                    "valid": free_mib >= options.min_gpu_free_mib,
                }
                reporter.finish_phase(
                    (
                        f"free={free_mib} MiB required>="
                        f"{options.min_gpu_free_mib} MiB"
                    ),
                    success=free_mib >= options.min_gpu_free_mib,
                    level=(
                        "PASS"
                        if free_mib >= options.min_gpu_free_mib
                        else "ERROR"
                    ),
                )
                if free_mib < options.min_gpu_free_mib:
                    raise AutoExplorationError(
                        "insufficient free GPU memory before Molmo model load: "
                        f"{free_mib} MiB available, at least "
                        f"{options.min_gpu_free_mib} MiB required; stop the Viser "
                        "process, close its browser tab and other GPU-heavy GUI apps, "
                        "then retry"
                    )
                _record_stage(
                    output,
                    iteration_dir,
                    iteration,
                    record,
                    "GPU_PREFLIGHT_COMPLETED",
                )
            else:
                reporter.emit(
                    "gpu-preflight",
                    "GPU free-memory gate disabled (--min-gpu-free-mib 0)",
                    iteration=iteration,
                    level="WARNING",
                )
            reporter.start_phase(
                "molmo",
                (
                    f"loading/inferencing {len(options.keypoint_specs)} keypoint "
                    f"query/queries on Camera {','.join(options.keypoint_cameras)}"
                ),
                iteration=iteration,
            )
            keypoint_dir = iteration_dir / "molmo_keypoints"
            manifest = keypoint_runner(
                project_root=session.project_root,
                perception_dir=session.workspace / "perception_views",
                artifact_dir=keypoint_dir,
                confidence_threshold=options.confidence_threshold,
                molmo_python=options.molmo_python,
                model=options.molmo_model,
                timeout_s=options.molmo_timeout_s,
                local_files_only=not options.molmo_allow_download,
                keypoint_specs=options.keypoint_specs,
                cameras=options.keypoint_cameras,
                install=True,
                worker_line_callback=reporter.worker_line,
            )
            record["molmo_keypoints"] = manifest
            record["artifacts"]["molmo_keypoints"] = str(
                keypoint_dir / "molmo_keypoint_grasp_references.json"
            )
            _print_keypoints(reporter, iteration, manifest)
            _record_stage(
                output, iteration_dir, iteration, record, "MOLMO_KEYPOINTS_COMPLETED"
            )
            if manifest.get("status") != "READY":
                reporter.finish_phase(
                    "finished, but no keypoint passed confidence/geometry gates",
                    success=False,
                )
                raise AutoExplorationError(
                    "no Molmo keypoint passed the confidence and calibrated-geometry gates"
                )
            reporter.finish_phase(
                (
                    f"accepted {manifest['accepted_reference_count']} reference(s); "
                    f"artifacts saved in {keypoint_dir}"
                )
            )
            before_images = _planning_images(saved, saved_path, manifest)
            record["before_images"] = [str(path) for path in before_images]

            feedback: str | None = None
            proposal: ExplorationProposal | None = None
            source = ""
            preflight = None
            controller = None
            for attempt in range(1, options.max_replans + 2):
                reporter.emit(
                    "planning",
                    f"Claude/preflight attempt {attempt}/{options.max_replans + 1}",
                    iteration=iteration,
                )

                def phase_callback(phase: str, event: str, value: float) -> None:
                    labels = {
                        "visual_planning": "Claude Stage 1: inspecting images and selecting Camera/Rxxx",
                        "final_grounding": "Claude Stage 2: grounding the selected Rxxx into robot actions",
                    }
                    if event == "started":
                        reporter.start_phase(
                            phase,
                            f"{labels.get(phase, phase)} · timeout {value:.0f}s",
                            iteration=iteration,
                        )
                    elif event == "completed":
                        reporter.finish_phase(f"completed · reported {value:.2f}s")
                    elif event == "failed":
                        reporter.finish_phase(
                            f"failed · reported {value:.2f}s",
                            success=False,
                        )
                    elif event == "reselecting":
                        reporter.finish_phase(
                            (
                                "selected reference failed deterministic validation; "
                                f"reselecting · attempt {value:.2f}s"
                            ),
                            success=False,
                            level="WARNING",
                        )
                    else:
                        reporter.emit(
                            phase,
                            f"{event} · reported {value:.2f}s",
                            iteration=iteration,
                        )

                try:
                    proposal = client.plan(
                        before_images,
                        session,
                        objective,
                        feedback=feedback,
                        history=history,
                        phase_callback=phase_callback,
                        reference_policy="molmo_confidence_filtered_keypoints",
                    )
                    source = exploration_source(proposal)
                    source_path.write_text(source, encoding="utf-8")
                    reporter.start_phase(
                        "preflight",
                        "validating restricted source and workspace limits",
                        iteration=iteration,
                    )
                    preflight = session.runner.preflight(source_path.name)
                    if preflight.error:
                        raise ExperimentValidationError(preflight.error)
                    reporter.finish_phase(
                        f"static validation passed for {len(preflight.actions)} action(s)"
                    )
                    if options.skip_controller_ik:
                        controller = {"status": "SKIPPED_DRY_RUN"}
                        reporter.emit(
                            "controller-ik",
                            "skipped by --skip-controller-ik (dry run only)",
                            iteration=iteration,
                            level="WARNING",
                        )
                    else:
                        reporter.start_phase(
                            "controller-ik",
                            "validating all action targets without motion",
                            iteration=iteration,
                        )
                        controller = controller_validator(
                            session.robot_config, preflight.actions
                        )
                        reporter.finish_phase("all action targets passed controller IK")
                    break
                except Exception as exc:
                    reporter.fail_current_phase(f"{type(exc).__name__}: {exc}")
                    if (
                        attempt >= options.max_replans + 1
                        or not _is_preexecution_replan_error(exc)
                    ):
                        raise
                    feedback = _replan_feedback(exc, proposal)
                    reporter.emit(
                        "replan",
                        f"pre-execution candidate rejected: {type(exc).__name__}: {exc}",
                        iteration=iteration,
                        level="WARNING",
                    )
            if proposal is None or preflight is None or controller is None:
                raise AutoExplorationError("planning ended without a validated proposal")
            _print_proposal(reporter, iteration, client, proposal, source)
            record["proposal"] = proposal.as_dict()
            record["proposal_source"] = source
            record["visual_plan"] = _jsonable(client.last_visual_plan_result)
            record["claude_plan"] = _jsonable(client.last_plan_result)
            record["planning_timing"] = dict(client.last_plan_timing)
            record["preflight"] = _jsonable(preflight)
            record["controller_ik"] = _jsonable(controller)
            (iteration_dir / "proposal.py").write_text(source, encoding="utf-8")
            _write_json(iteration_dir / "proposal.json", proposal.as_dict())
            _write_json(iteration_dir / "preflight.json", preflight)
            _write_json(iteration_dir / "controller_ik.json", controller)
            _record_stage(
                output, iteration_dir, iteration, record, "PREEXECUTION_VALIDATED"
            )
            reporter.emit(
                "preexecution",
                "proposal passed static preflight and controller gate",
                iteration=iteration,
                level="PASS",
            )

            if not options.enable_real:
                record["status"] = "DRY_RUN_VALIDATED"
                record["completed_at"] = _now()
                _iteration_checkpoint(output, iteration_dir, iteration, record)
                summary["iterations"].append(
                    {"iteration": iteration, "status": record["status"]}
                )
                reporter.emit(
                    "dry-run",
                    "physical execution disabled; iteration ends after validation",
                    iteration=iteration,
                    level="DONE",
                )
                continue

            reporter.start_phase(
                "execution",
                "sending one validated physical rollout; mandatory Home remains active",
                iteration=iteration,
            )
            execution = session.run_experiment(
                source_path.name,
                real=True,
                confirmed=True,
                notes=f"Molmo keypoint CLI iteration {iteration}.",
            )
            record["execution"] = execution
            record["mandatory_return_home"] = session.last_return_home_outcome
            _write_json(iteration_dir / "execution.json", execution)
            _write_json(
                iteration_dir / "mandatory_return_home.json",
                session.last_return_home_outcome,
            )
            _record_stage(
                output, iteration_dir, iteration, record, "EXECUTION_COMPLETED"
            )
            reporter.finish_phase(
                f"execution_completed={bool(execution.get('execution_completed'))}",
                success=bool(execution.get("execution_completed")),
                payload=execution,
            )
            if not execution.get("execution_completed"):
                raise AutoExplorationError(
                    f"physical rollout did not complete: {execution.get('robot_errors', [])}"
                )

            prepare_real_perception_position(
                iteration,
                record,
                "post_action_perception_robot_positioning",
            )
            reporter.start_phase(
                "after-capture",
                "capturing post-action Camera A/B frames",
                iteration=iteration,
            )
            after_frames = capture(perception_config)
            after_dir = iteration_dir / "after_capture"
            after_images = _save_frame_images(after_frames, after_dir)
            record["after_images"] = [str(path) for path in after_images]
            _record_stage(
                output, iteration_dir, iteration, record, "AFTER_CAPTURE_COMPLETED"
            )
            reporter.finish_phase(f"saved {len(after_images)} after image(s) in {after_dir}")
            reporter.start_phase(
                "evaluation",
                "Claude comparing before/after images",
                iteration=iteration,
            )
            evaluation = client.evaluate(
                before_images,
                after_images,
                proposal=proposal,
                run_dir=session.run_dir,
            )
            record["evaluation"] = evaluation.as_dict()
            record["claude_evaluation"] = _jsonable(client.last_evaluation_result)
            _write_json(iteration_dir / "evaluation.json", evaluation.as_dict())
            record["status"] = "COMPLETED"
            record["completed_at"] = _now()
            _iteration_checkpoint(output, iteration_dir, iteration, record)
            history_entry = {
                "iteration": iteration,
                "plan_status": "executed",
                "proposal": proposal.as_dict(),
                "execution_completed": True,
                "before_images": record["before_images"],
                "after_images": record["after_images"],
                "evaluation": evaluation.as_dict(),
            }
            history.append(history_entry)
            summary["iterations"].append(
                {
                    "iteration": iteration,
                    "status": record["status"],
                    "task_progress": evaluation.task_progress.status,
                    "confidence": evaluation.task_progress.confidence,
                }
            )
            reporter.finish_phase(
                (
                    f"task_progress={evaluation.task_progress.status} "
                    f"confidence={evaluation.task_progress.confidence:.3f} "
                    f"earliest_failure={evaluation.earliest_failure_stage}"
                ),
                payload=evaluation.as_dict(),
            )
            if evaluation.stop:
                reporter.emit(
                    "stop",
                    f"evaluator requested stop: {evaluation.reason}",
                    iteration=iteration,
                )
                summary["stop_reason"] = evaluation.reason
                break
            objective = evaluation.next_objective
        except KeyboardInterrupt:
            reporter.fail_current_phase("operator interrupted the active phase")
            record["status"] = "INTERRUPTED"
            record["error"] = "KeyboardInterrupt"
            record["completed_at"] = _now()
            _iteration_checkpoint(output, iteration_dir, iteration, record)
            summary["iterations"].append(
                {"iteration": iteration, "status": record["status"]}
            )
            summary["status"] = "INTERRUPTED"
            exit_code = 130
            reporter.emit(
                "interrupt",
                "operator interrupted the CLI loop",
                iteration=iteration,
                level="WARNING",
            )
            break
        except BaseException as exc:
            reporter.fail_current_phase(f"{type(exc).__name__}: {exc}")
            record["status"] = "FAILED"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
            record["completed_at"] = _now()
            _iteration_checkpoint(output, iteration_dir, iteration, record)
            summary["iterations"].append(
                {
                    "iteration": iteration,
                    "status": record["status"],
                    "error": record["error"],
                }
            )
            summary["status"] = "FAILED"
            summary["error"] = record["error"]
            exit_code = 1
            reporter.emit(
                "failure",
                record["error"],
                iteration=iteration,
                level="ERROR",
            )
            traceback.print_exc(file=stream)
            molmo_log = iteration_dir / "molmo_keypoints" / "molmo_keypoints.stdout.txt"
            if molmo_log.is_file():
                reporter.emit(
                    "molmo-log",
                    f"saved worker log: {molmo_log}",
                    iteration=iteration,
                    level="ERROR",
                )
            break
        finally:
            if source_path.is_file():
                source_path.unlink()

    if summary.get("status") == "RUNNING":
        summary["status"] = "COMPLETED" if exit_code == 0 else "FAILED"
    summary["completed_at"] = _now()
    summary["iteration_count"] = len(summary["iterations"])
    _write_json(output / "summary.json", summary)
    reporter.stop_heartbeat()
    reporter.emit(
        "shutdown",
        f"status={summary['status']} iterations={summary['iteration_count']}",
        payload={"summary": str(output / "summary.json")},
        level="INFO" if exit_code == 0 else "ERROR",
    )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--robot-config", type=Path)
    parser.add_argument(
        "--perception-config",
        type=Path,
        default=Path("config/perception.free_exploration.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument(
        "--settle-s",
        type=float,
        default=2.0,
        help=(
            "seconds to wait after reaching perception_position before each "
            "real RGB-D capture"
        ),
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--keypoints-json", type=Path)
    parser.add_argument("--keypoint-camera", action="append", choices=["A", "B"])
    parser.add_argument("--molmo-python", type=Path)
    parser.add_argument("--molmo-model", default="allenai/MolmoPoint-8B")
    parser.add_argument("--molmo-timeout-s", type=int, default=900)
    parser.add_argument("--molmo-allow-download", action="store_true")
    parser.add_argument(
        "--min-gpu-free-mib",
        type=int,
        default=20_000,
        help=(
            "hard-stop before Molmo unless GPU 0 has at least this much free "
            "memory; use 0 to disable"
        ),
    )
    parser.add_argument(
        "--heartbeat-s",
        type=float,
        default=10.0,
        help="print the active phase and its elapsed time at this interval; use 0 to disable",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="disable ANSI colors even when stdout is an interactive terminal",
    )
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--claude-timeout-s", type=int, default=400)
    parser.add_argument("--claude-grounding-timeout-s", type=int, default=120)
    parser.add_argument("--max-replans", type=int, default=2)
    parser.add_argument(
        "--objective",
        default=(
            "Take one planning-mode-appropriate action that makes the current garment "
            "as open and spread as safely possible."
        ),
    )
    parser.add_argument(
        "--enable-real",
        action="store_true",
        help="send each validated proposal to the physical xArm automatically",
    )
    parser.add_argument(
        "--skip-controller-ik",
        action="store_true",
        help="dry-run only: stop after static preflight without connecting to xArm",
    )
    args = parser.parse_args(argv)

    root = Path(args.project_root).expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    robot_config_path = (
        args.robot_config.expanduser().resolve() if args.robot_config else None
    )
    perception_path = args.perception_config
    if not perception_path.is_absolute():
        perception_path = root / perception_path
    session = _load_session(
        root,
        run_dir,
        args.run_id,
        robot_config_path,
    )
    perception_config = PerceptionConfig.load(root, perception_path.resolve())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else session.results / "molmo_keypoint_cli" / stamp
    )
    max_iterations = None if args.max_iterations == 0 else args.max_iterations
    options = KeypointCliOptions(
        max_iterations=max_iterations,
        settle_s=args.settle_s,
        enable_real=args.enable_real,
        skip_controller_ik=args.skip_controller_ik,
        confidence_threshold=args.confidence_threshold,
        molmo_python=(
            args.molmo_python.expanduser().resolve() if args.molmo_python else None
        ),
        molmo_model=args.molmo_model,
        keypoint_specs=load_keypoint_specs(args.keypoints_json),
        keypoint_cameras=tuple(args.keypoint_camera or ("A", "B")),
        molmo_timeout_s=args.molmo_timeout_s,
        molmo_allow_download=args.molmo_allow_download,
        min_gpu_free_mib=args.min_gpu_free_mib,
        heartbeat_s=args.heartbeat_s,
        color=False if args.no_color else None,
        claude_binary=args.claude_binary,
        claude_timeout_s=args.claude_timeout_s,
        claude_grounding_timeout_s=args.claude_grounding_timeout_s,
        max_replans=args.max_replans,
        objective=args.objective,
    )
    return run_keypoint_cli_loop(
        session,
        perception_config,
        output_dir,
        options,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
