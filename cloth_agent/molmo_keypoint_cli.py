"""Headless semantic-anchor garment-opening CLI.

The loop mirrors the automatic Viser workflow without starting a web server or
browser: synchronized A/B perception, high-confidence Molmo semantic anchors,
semantic-state construction, Claude relation strategy, local geometry Rxxx,
state-scoped action generation, preflight/controller IK, optional execution,
stage-wise evaluation, and structured experience. Every phase is printed to
stdout and checkpointed under a separate iteration directory.
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
    ExplorationProposal,
    _is_preexecution_replan_error,
    _load_session,
    _save_frame_images,
    grasp_targets_from_actions,
)
from .experiment import ExperimentValidationError
from .free_exploration import exploration_source, perception_image_paths
from .molmo_keypoint_pipeline import (
    DEFAULT_SEMANTIC_ANCHORS,
    DEFAULT_SEMANTIC_CONFIDENCE_THRESHOLD,
    KeypointSpec,
    load_keypoint_specs,
    run_molmo_semantic_anchor_pipeline,
    validate_confidence_threshold,
)
from .perception import PerceptionConfig, capture_two_view_rgbd
from .robot_api import move_robot_to_perception_position, validate_controller_trajectory
from .semantic_claude import SemanticActionResult, SemanticClaudeClient
from .semantic_pipeline import (
    LocalGeometryGrounder,
    SemanticPipelineError,
    SemanticStateBuilder,
    action_scope_from_experiences,
    append_structured_experience,
    build_structured_experience,
    load_structured_experiences,
    refresh_local_geometry_artifacts,
    semantic_hypothesis_budget,
)
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
                "│  ClothAgent · Semantic Anchor CLI".ljust(width + 1)
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
    confidence_threshold: float = DEFAULT_SEMANTIC_CONFIDENCE_THRESHOLD
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
    max_replans: int = 1
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
    if not 0 <= options.max_replans <= 1:
        raise ValueError(
            "max_replans must be 0 or 1; hard validation permits one correction"
        )
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


def _print_semantic_anchors(
    reporter: CliReporter,
    iteration: int,
    manifest: dict[str, Any],
) -> None:
    reporter.emit(
        "molmo-summary",
        (
            f"status={manifest['status']} semantic_anchors="
            f"{manifest['anchor_count']} threshold>"
            f"{manifest['confidence_threshold']:.3f}"
        ),
        iteration=iteration,
    )
    for view in manifest.get("views", []):
        camera = str(view.get("camera", "?"))
        for candidate in view.get("records", []):
            accepted = bool(candidate.get("accepted"))
            anchor = candidate.get("anchor_id", "-") if accepted else "-"
            reason = (
                "accepted_as_semantic_anchor"
                if accepted
                else candidate.get("rejection_reason", "rejected")
            )
            reporter.emit(
                "molmo-semantic-anchor",
                (
                    f"camera={camera} name={candidate.get('name')} "
                    f"status={candidate.get('status')} "
                    f"confidence={float(candidate.get('confidence', 0.0)):.4f} "
                    f"valid={str(accepted).lower()} anchor={anchor} reason={reason}"
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


def _print_semantic_action(
    reporter: CliReporter,
    iteration: int,
    strategy: Any,
    action_result: SemanticActionResult,
    candidate: dict[str, Any],
    proposal: ExplorationProposal,
    source: str,
) -> None:
    reporter.emit(
        "semantic-strategy",
        (
            f"target={strategy.target_part} hypothesis={strategy.hypothesis_state} "
            f"anchor={strategy.anchor_id} desired_change={strategy.desired_change}"
        ),
        iteration=iteration,
        payload=strategy.as_dict(),
    )
    reporter.emit(
        "local-grasp-selection",
        (
            f"selected={action_result.selected_candidate_id} "
            f"feature={candidate.get('feature')} "
            f"graspability={float(candidate.get('graspability_score', 0.0)):.3f}"
        ),
        iteration=iteration,
        payload=candidate,
    )
    reporter.emit(
        "semantic-action",
        f"validated proposal with {len(proposal.actions)} action(s)",
        iteration=iteration,
        payload=action_result.as_dict(),
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
    keypoint_runner: Callable[..., dict[str, Any]] = (
        run_molmo_semantic_anchor_pipeline
    ),
    client: SemanticClaudeClient | None = None,
    semantic_state_builder: SemanticStateBuilder | None = None,
    local_geometry_grounder: LocalGeometryGrounder | None = None,
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
    client = client or SemanticClaudeClient(
        binary=options.claude_binary,
        strategy_timeout_s=options.claude_timeout_s,
        action_timeout_s=options.claude_grounding_timeout_s,
        evaluation_timeout_s=options.claude_timeout_s,
    )
    semantic_state_builder = semantic_state_builder or SemanticStateBuilder()
    local_geometry_grounder = local_geometry_grounder or LocalGeometryGrounder()
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
            "Cameras / anchor queries": (
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
            "headless semantic-anchor loop started; physical execution "
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

    experience_path = session.workspace / "structured_experience.jsonl"
    experiences = load_structured_experiences(experience_path)
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
            "objective": options.objective,
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
                    f"loading/inferencing {len(options.keypoint_specs)} semantic-anchor "
                    f"query/queries on Camera {','.join(options.keypoint_cameras)}; "
                    f"strict confidence > {options.confidence_threshold:.3f}"
                ),
                iteration=iteration,
            )
            keypoint_dir = iteration_dir / "semantic_anchors"
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
            record["semantic_anchors"] = manifest
            record["artifacts"]["semantic_anchors"] = str(
                keypoint_dir / "molmo_semantic_anchors.json"
            )
            _print_semantic_anchors(reporter, iteration, manifest)
            _record_stage(
                output, iteration_dir, iteration, record, "SEMANTIC_ANCHORS_COMPLETED"
            )
            if manifest.get("status") != "READY":
                reporter.finish_phase(
                    "finished, but no semantic anchor passed confidence/consistency gates",
                    success=False,
                )
                raise AutoExplorationError(
                    "no high-confidence Molmo semantic anchor is available; "
                    "planning is blocked before Claude and robot motion"
                )
            reporter.finish_phase(
                (
                    f"accepted {manifest['anchor_count']} semantic anchor(s); "
                    f"artifacts saved in {keypoint_dir}"
                )
            )
            before_images = _planning_images(saved, saved_path, manifest)
            record["before_images"] = [str(path) for path in before_images]

            reporter.start_phase(
                "semantic-state",
                "building uncertain garment relations from Sxxx anchors",
                iteration=iteration,
            )
            semantic_state = semantic_state_builder.build(manifest, saved)
            record["semantic_state"] = semantic_state
            _write_json(iteration_dir / "semantic_state.json", semantic_state)
            _record_stage(
                output, iteration_dir, iteration, record, "SEMANTIC_STATE_COMPLETED"
            )
            reporter.finish_phase(
                (
                    f"built {len(semantic_state['known']['anchors'])} known anchor(s) "
                    f"and {len(semantic_state['hypotheses'])} relation hypothesis/hypotheses"
                ),
                payload=semantic_state,
            )

            previous_hypothesis_key = (
                str(experiences[-1].get("hypothesis_key")) if experiences else None
            )
            previous_part = (
                str(
                    experiences[-1]
                    .get("semantic_state", {})
                    .get("target_part", "")
                )
                if experiences
                else ""
            )
            current_previous_anchor = next(
                (
                    str(anchor["anchor_id"])
                    for anchor in semantic_state["known"]["anchors"]
                    if anchor.get("type") == previous_part
                ),
                None,
            )
            budget = semantic_hypothesis_budget(
                experiences,
                hypothesis_key=previous_hypothesis_key,
                anchor_id=current_previous_anchor,
            )
            record["semantic_hypothesis_budget_before_strategy"] = budget.as_dict()
            reporter.start_phase(
                "semantic-strategy",
                "Claude choosing the garment relation to change; no Rxxx/actions allowed",
                iteration=iteration,
            )
            strategy = client.plan_strategy(
                images=before_images,
                run_dir=session.run_dir,
                semantic_state=semantic_state,
                experiences=experiences,
                budget=budget,
            )
            reporter.finish_phase(
                (
                    f"target={strategy.target_part} hypothesis={strategy.hypothesis_state} "
                    f"anchor={strategy.anchor_id}"
                ),
                payload=strategy.as_dict(),
            )
            record["semantic_strategy"] = strategy.as_dict()
            record["claude_semantic_strategy"] = _jsonable(
                client.last_strategy_log
            )
            _write_json(iteration_dir / "semantic_strategy.json", strategy.as_dict())
            _write_json(
                iteration_dir / "claude_semantic_strategy_log.json",
                client.last_strategy_log,
            )
            _record_stage(
                output, iteration_dir, iteration, record, "SEMANTIC_STRATEGY_COMPLETED"
            )

            selected_budget = semantic_hypothesis_budget(
                experiences,
                hypothesis_key=strategy.hypothesis_key,
                anchor_id=strategy.anchor_id,
            )
            if selected_budget.disposition == "ESCAPE_HYPOTHESIS":
                raise SemanticPipelineError(
                    "Claude selected a semantic hypothesis whose finite budget is "
                    f"already exhausted: {selected_budget.reason}"
                )
            record["semantic_hypothesis_budget"] = selected_budget.as_dict()
            reporter.start_phase(
                "local-geometry",
                (
                    f"searching only around {strategy.anchor_id} for free edges, "
                    "height steps and ridges"
                ),
                iteration=iteration,
            )
            local_geometry_dir = iteration_dir / "local_geometry"
            local_geometry = local_geometry_grounder.ground(
                perception_dir=session.workspace / "perception_views",
                artifact_dir=local_geometry_dir,
                semantic_state=semantic_state,
                strategy=strategy,
                install=True,
            )
            record["local_geometry"] = local_geometry
            record["artifacts"]["local_geometry"] = str(
                local_geometry_dir / "local_geometry_candidates.json"
            )
            _record_stage(
                output, iteration_dir, iteration, record, "LOCAL_GEOMETRY_COMPLETED"
            )
            reporter.finish_phase(
                (
                    f"generated {local_geometry['candidate_count']} local Rxxx "
                    f"candidate(s) inside {strategy.target_part} region"
                ),
                payload=local_geometry,
            )

            reporter.start_phase(
                "local-capability",
                "checking workspace/controller IK for each local Rxxx before Claude action planning",
                iteration=iteration,
            )
            reachable_candidates: list[dict[str, Any]] = []
            rejected_candidates: list[dict[str, Any]] = []
            for candidate in local_geometry["candidates"]:
                checked = dict(candidate)
                base_x, base_y, surface_z = (
                    float(value) for value in checked["base_xyz_mm"]
                )
                yaw = float(checked.get("suggested_yaw_deg", 0.0))
                z_high = session.robot_config.boundaries.z_max
                approach_z = max(85.0, surface_z + 40.0)
                if z_high is not None:
                    approach_z = min(approach_z, float(z_high) - 5.0)
                grasp_check_z = max(surface_z + 5.0, approach_z - 45.0)
                capability_actions = [
                    {
                        "name": "move",
                        "args": {
                            "x": base_x,
                            "y": base_y,
                            "z": approach_z,
                            "yaw": yaw,
                        },
                    },
                    {
                        "name": "move",
                        "args": {
                            "x": base_x,
                            "y": base_y,
                            "z": grasp_check_z,
                            "yaw": yaw,
                        },
                    },
                ]
                try:
                    for action in capability_actions:
                        args = action["args"]
                        session.robot_config.boundaries.validate(
                            args["x"],
                            args["y"],
                            args["z"],
                            session.robot_config.workspace_margin_mm,
                            z_lower_margin_mm=session.robot_config.lower_z_margin_mm,
                        )
                    if options.skip_controller_ik:
                        checked["controller_reachability"] = "SKIPPED_DRY_RUN"
                    else:
                        controller_validator(
                            session.robot_config, capability_actions
                        )
                        checked["controller_reachability"] = "PASS"
                except Exception as exc:
                    checked["controller_reachability"] = "REJECTED"
                    checked["rejection_reason"] = "workspace_or_controller_ik"
                    checked["reachability_error"] = f"{type(exc).__name__}: {exc}"
                    rejected_candidates.append(checked)
                else:
                    reachable_candidates.append(checked)
            if not reachable_candidates:
                raise SemanticPipelineError(
                    "all local geometry candidates failed deterministic workspace/IK "
                    "capability checks before Claude action planning"
                )
            if selected_budget.forced_geometry_type is not None:
                geometry_rejected = [
                    {
                        **item,
                        "rejection_reason": (
                            f"previous_supported_geometry_family_"
                            f"{selected_budget.forced_geometry_type}"
                        ),
                    }
                    for item in reachable_candidates
                    if item.get("feature") != selected_budget.forced_geometry_type
                ]
                matching_geometry = [
                    item
                    for item in reachable_candidates
                    if item.get("feature") == selected_budget.forced_geometry_type
                ]
                if not matching_geometry:
                    raise SemanticPipelineError(
                        "the previous supported grasp geometry family is unavailable "
                        f"in the current local region: {selected_budget.forced_geometry_type}"
                    )
                reachable_candidates = matching_geometry
                rejected_candidates.extend(geometry_rejected)
                reporter.emit(
                    "local-persistence",
                    (
                        f"keeping previous supported geometry family "
                        f"{selected_budget.forced_geometry_type}; "
                        f"withheld {len(geometry_rejected)} other local candidate(s)"
                    ),
                    iteration=iteration,
                    level="PASS",
                )
            local_geometry = {
                **local_geometry,
                "candidate_count": len(reachable_candidates),
                "candidates": reachable_candidates,
                "capability_rejected_candidates": rejected_candidates,
            }
            local_geometry = refresh_local_geometry_artifacts(
                perception_dir=session.workspace / "perception_views",
                artifact_dir=local_geometry_dir,
                manifest=local_geometry,
                install=True,
            )
            record["local_geometry"] = local_geometry
            _write_json(
                local_geometry_dir / "local_geometry_candidates.json",
                local_geometry,
            )
            reporter.finish_phase(
                (
                    f"reachable={len(reachable_candidates)} "
                    f"rejected={len(rejected_candidates)}"
                ),
                payload={
                    "reachable": reachable_candidates,
                    "rejected": rejected_candidates,
                },
            )
            _record_stage(
                output,
                iteration_dir,
                iteration,
                record,
                "LOCAL_CAPABILITY_COMPLETED",
            )

            scope = action_scope_from_experiences(
                experiences,
                hypothesis_key=strategy.hypothesis_key,
                budget_disposition=selected_budget.disposition,
            )
            record["action_scope"] = scope.as_dict()
            reporter.emit(
                "action-scope",
                (
                    f"runtime authority={scope.name} lateral<={scope.max_lateral_mm:.1f}mm "
                    f"lift<={scope.max_lift_mm:.1f}mm"
                ),
                iteration=iteration,
                payload=scope.as_dict(),
                level="PASS",
            )

            proposal: ExplorationProposal | None = None
            action_result: SemanticActionResult | None = None
            selected_candidate: dict[str, Any] | None = None
            source = ""
            preflight = None
            controller = None
            action_candidates = list(local_geometry["candidates"])
            for attempt in range(1, options.max_replans + 2):
                reporter.emit(
                    "planning",
                    (
                        f"semantic action/preflight attempt {attempt}/"
                        f"{options.max_replans + 1}"
                    ),
                    iteration=iteration,
                )
                try:
                    reporter.start_phase(
                        "semantic-action",
                        (
                            f"Claude selecting among {len(action_candidates)} local "
                            f"Rxxx candidate(s) under {scope.name}"
                        ),
                        iteration=iteration,
                    )
                    action_geometry = {**local_geometry, "candidates": action_candidates}
                    action_result = client.propose_action(
                        run_dir=session.run_dir,
                        strategy=strategy,
                        local_geometry=action_geometry,
                        scope=scope,
                        robot_context={
                            "workspace_bounds_mm": _jsonable(
                                session.robot_config.boundaries
                            ),
                            "fixed_orientation_deg": {
                                "roll": session.robot_config.orientation_roll_deg,
                                "pitch": session.robot_config.orientation_pitch_deg,
                            },
                            "capabilities": [
                                "move(x,y,z,yaw)",
                                "open_gripper()",
                                "close_gripper()",
                                "home()",
                            ],
                        },
                        overlay_image=Path(local_geometry["overlay"]),
                    )
                    record.setdefault("claude_semantic_action_attempts", []).append(
                        _jsonable(client.last_action_log)
                    )
                    _write_json(
                        iteration_dir
                        / f"claude_semantic_action_attempt_{attempt:02d}.json",
                        client.last_action_log,
                    )
                    proposal = action_result.proposal
                    selected_candidate = next(
                        item
                        for item in action_candidates
                        if item["reference_id"]
                        == action_result.selected_candidate_id
                    )
                    reporter.finish_phase(
                        (
                            f"selected={action_result.selected_candidate_id} "
                            f"feature={selected_candidate['feature']}"
                        ),
                        payload=action_result.as_dict(),
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
                    if action_result is None:
                        raise
                    rejected_id = action_result.selected_candidate_id
                    action_candidates = [
                        item
                        for item in action_candidates
                        if item["reference_id"] != rejected_id
                    ]
                    if not action_candidates:
                        raise SemanticPipelineError(
                            "the only local grasp candidate failed deterministic "
                            "pre-execution validation"
                        ) from exc
                    reporter.emit(
                        "replan",
                        (
                            f"hard validation rejected local candidate {rejected_id}; "
                            f"one bounded correction remains: {type(exc).__name__}: {exc}"
                        ),
                        iteration=iteration,
                        level="WARNING",
                    )
            if (
                proposal is None
                or action_result is None
                or selected_candidate is None
                or preflight is None
                or controller is None
            ):
                raise AutoExplorationError("planning ended without a validated proposal")
            _print_semantic_action(
                reporter,
                iteration,
                strategy,
                action_result,
                selected_candidate,
                proposal,
                source,
            )
            record["proposal"] = proposal.as_dict()
            record["semantic_action"] = action_result.as_dict()
            record["selected_local_candidate"] = selected_candidate
            record["proposal_source"] = source
            record["claude_semantic_strategy"] = _jsonable(
                client.last_strategy_log
            )
            record["claude_semantic_action"] = _jsonable(client.last_action_log)
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
                (
                    "Claude evaluating semantic target → acquisition → structure "
                    "engagement → opening relevance → transport → laydown"
                ),
                iteration=iteration,
            )
            evaluation = client.evaluate(
                before_images=before_images,
                after_images=after_images,
                run_dir=session.run_dir,
                semantic_state=semantic_state,
                strategy=strategy,
                candidate=selected_candidate,
                action_result=action_result,
            )
            record["evaluation"] = evaluation.as_dict()
            record["claude_semantic_evaluation"] = _jsonable(
                client.last_evaluation_log
            )
            _write_json(iteration_dir / "evaluation.json", evaluation.as_dict())
            _write_json(
                iteration_dir / "claude_semantic_evaluation_log.json",
                client.last_evaluation_log,
            )
            experience = build_structured_experience(
                iteration=iteration,
                semantic_state=semantic_state,
                strategy=strategy,
                candidate=selected_candidate,
                action_scope=scope,
                evaluation=evaluation,
            )
            append_structured_experience(experience_path, experience)
            experiences.append(experience)
            record["structured_experience"] = experience
            _write_json(iteration_dir / "structured_experience.json", experience)
            record["status"] = "COMPLETED"
            record["completed_at"] = _now()
            _iteration_checkpoint(output, iteration_dir, iteration, record)
            summary["iterations"].append(
                {
                    "iteration": iteration,
                    "status": record["status"],
                    "semantic_target": evaluation.semantic_target.status,
                    "structure_engagement": evaluation.structure_engagement.status,
                    "opening_relevance": evaluation.opening_relevance.status,
                    "task_progress": evaluation.task_progress["status"],
                    "confidence": evaluation.task_progress["confidence"],
                }
            )
            reporter.finish_phase(
                (
                    f"semantic_target={evaluation.semantic_target.status} "
                    f"engagement={evaluation.structure_engagement.status} "
                    f"opening_relevance={evaluation.opening_relevance.status} "
                    f"task_progress={evaluation.task_progress['status']} "
                    f"confidence={evaluation.task_progress['confidence']:.3f} "
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
            molmo_log = (
                iteration_dir / "semantic_anchors" / "molmo_keypoints.stdout.txt"
            )
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
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_SEMANTIC_CONFIDENCE_THRESHOLD,
        help=(
            "strict Molmo semantic-anchor threshold; confidence equal to the "
            "threshold is rejected"
        ),
    )
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
    parser.add_argument(
        "--max-replans",
        type=int,
        default=1,
        help="hard validation correction budget; semantic mode permits at most one",
    )
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
        keypoint_specs=(
            load_keypoint_specs(args.keypoints_json)
            if args.keypoints_json
            else DEFAULT_SEMANTIC_ANCHORS
        ),
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
