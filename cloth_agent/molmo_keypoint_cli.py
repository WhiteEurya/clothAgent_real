"""Headless Claude-global garment-opening CLI.

The default loop gives Claude complete synchronized A/B perception and prior
physical outcomes, lets it choose one arbitrary image pixel and action, then
applies grounding, preflight, workspace, and controller safety gates. The
real dashboard launcher also runs an axis-first Molmo annotation pass and
shows those overlays to Claude as auxiliary evidence; Molmo never selects the
final grasp point in ``claude_global`` mode. Every phase is printed to stdout
and checkpointed under a separate iteration directory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass, replace
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
from .config import SafetyError
from .experiment import ExperimentValidationError
from .evidence_ledger import build_evidence_record, persist_evidence_record
from .free_exploration import (
    ClaudeExplorationClient,
    DEFAULT_EXPLORATION_OBJECTIVE,
    ExplorationPlanningError,
    ExplorationTimeoutError,
    exploration_prompt,
    exploration_source,
    evaluation_depth_ranges,
    evaluation_perception_image_paths,
    global_perception_image_paths,
    ground_global_grasp_target,
    invoke_direct_prompt,
    perception_image_paths,
    split_global_lift_checkpoint_plan,
    validate_global_probe_profile,
)
from .molmo_keypoint_pipeline import (
    DEFAULT_SEMANTIC_ANCHORS,
    DEFAULT_SEMANTIC_CONFIDENCE_THRESHOLD,
    KeypointSpec,
    MolmoKeypointPipelineError,
    load_keypoint_specs,
    run_molmo_semantic_anchor_pipeline,
    validate_confidence_threshold,
)
from .perception import PerceptionConfig, PerceptionError, capture_two_view_rgbd
from .robot_api import move_robot_to_perception_position, validate_controller_trajectory
from .rollout_recorder import (
    DualRealSenseRolloutRecorder,
    append_mp4_to_cumulative,
    build_rollout_phase_timeline,
    label_iteration_mp4,
    prune_rollout_video_files,
    speed_up_mp4,
)
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
from .skill_lifecycle import RunSkillLedger, SkillProposal, SkillStore
from .viewer import _load_latest_perception


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _override_camera_controls(
    config: PerceptionConfig,
    *,
    camera_a_exposure: float | None = None,
    camera_b_exposure: float | None = None,
    camera_a_white_balance: float | None = None,
    camera_b_white_balance: float | None = None,
) -> PerceptionConfig:
    exposures = {"A": camera_a_exposure, "B": camera_b_exposure}
    white_balances = {
        "A": camera_a_white_balance,
        "B": camera_b_white_balance,
    }
    cameras = tuple(
        replace(
            camera,
            color_exposure=(
                exposures[camera.label]
                if exposures.get(camera.label) is not None
                else camera.color_exposure
            ),
            color_white_balance=(
                white_balances[camera.label]
                if white_balances.get(camera.label) is not None
                else camera.color_white_balance
            ),
        )
        for camera in config.cameras
    )
    updated = replace(config, cameras=cameras)
    updated.validate()
    return updated


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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AutoExplorationError(
                f"invalid JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise AutoExplorationError(
                f"global experience at {path}:{line_number} must be an object"
            )
        records.append(value)
    return records


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), ensure_ascii=False) + "\n")


def _save_global_selected_pixel_overlay(
    proposal: ExplorationProposal,
    perception: dict[str, Any],
    result_path: Path,
    output_path: Path,
) -> Path:
    """Draw only Claude's final selected pixel, never a candidate set."""

    from PIL import Image, ImageDraw

    selected = proposal.selected_grasp
    if selected is None:
        raise ExplorationPlanningError("global proposal is missing selected_grasp")
    view = next(
        (
            item
            for item in perception.get("views", [])
            if isinstance(item, dict) and item.get("label") == selected["camera"]
        ),
        None,
    )
    if view is None or not view.get("image"):
        raise ExplorationPlanningError(
            f"Camera {selected['camera']} full RGB is unavailable for selected-pixel audit"
        )
    image_path = (result_path.parent / str(view["image"])).resolve()
    image = Image.open(image_path).convert("RGB")
    x_px, y_px = (int(value) for value in selected["pixel_xy"])
    if not 0 <= x_px < image.width or not 0 <= y_px < image.height:
        raise ExplorationPlanningError(
            f"selected pixel ({x_px}, {y_px}) is outside RGB bounds "
            f"{image.width}x{image.height}"
        )
    draw = ImageDraw.Draw(image)
    radius = 10
    color = (255, 40, 40)
    draw.ellipse(
        (x_px - radius, y_px - radius, x_px + radius, y_px + radius),
        outline=color,
        width=3,
    )
    draw.line((x_px - 15, y_px, x_px + 15, y_px), fill=color, width=2)
    draw.line((x_px, y_px - 15, x_px, y_px + 15), fill=color, width=2)
    draw.text(
        (min(x_px + 14, max(0, image.width - 150)), max(0, y_px - 24)),
        f"CLAUDE {selected['camera']} ({x_px},{y_px})",
        fill=color,
        stroke_width=2,
        stroke_fill=(255, 255, 255),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path.resolve()


def _save_hold_checkpoint_snapshot(
    snapshot: dict[str, Any],
    output_dir: Path,
    *,
    depth_ranges: dict[str, tuple[float, float]],
    stage: str = "HOLD",
) -> tuple[list[Path], dict[str, Any]]:
    """Persist unlabelled recorder RGB-D arrays as labelled checkpoint evidence."""

    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=False)
    safe_stage = str(stage).upper().replace(" ", "_")
    image_paths: list[Path] = []
    manifest_frames: list[dict[str, Any]] = []
    for label in sorted(snapshot):
        frame = snapshot[label]
        rgb = np.asarray(frame.rgb, dtype=np.uint8)
        depth = np.asarray(frame.depth_m, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
            raise AutoExplorationError(
                f"invalid Camera {label} checkpoint RGB-D shapes: rgb={rgb.shape} depth={depth.shape}"
            )
        rgb_path = output_dir / f"{safe_stage}_camera_{label}_rgb.png"
        rgb_image = Image.fromarray(rgb)
        rgb_draw = ImageDraw.Draw(rgb_image)
        rgb_draw.rectangle((0, 0, rgb_image.width, 38), fill=(0, 0, 0))
        rgb_draw.text(
            (12, 11),
            f"{safe_stage} | CAMERA {label} | RGB",
            fill=(255, 255, 255),
        )
        rgb_image.save(rgb_path, format="PNG", optimize=True)

        depth_path = output_dir / f"{safe_stage}_camera_{label}_depth.png"
        depth_npy = output_dir / f"camera_{label}_depth_m.npy"
        np.save(depth_npy, depth)
        finite = np.isfinite(depth) & (depth > 0.0)
        min_depth, max_depth = depth_ranges.get(
            label,
            (float(np.nanpercentile(depth[finite], 2)), float(np.nanpercentile(depth[finite], 98)))
            if np.any(finite)
            else (0.2, 1.5),
        )
        if max_depth <= min_depth:
            max_depth = min_depth + 1e-3
        normalized = np.zeros(depth.shape, dtype=np.float32)
        normalized[finite] = np.clip(
            (max_depth - depth[finite]) / (max_depth - min_depth), 0.0, 1.0
        )
        red = normalized
        green = 1.0 - np.abs(2.0 * normalized - 1.0)
        blue = 1.0 - normalized
        depth_rgb = np.rint(
            np.stack((red, green, blue), axis=2) * 255.0
        ).astype(np.uint8)
        depth_rgb[~finite] = 0
        depth_image = Image.fromarray(depth_rgb)
        depth_draw = ImageDraw.Draw(depth_image)
        depth_draw.rectangle((0, 0, depth_image.width, 38), fill=(0, 0, 0))
        depth_draw.text(
            (12, 11),
            f"{safe_stage} | CAMERA {label} | DEPTH ({min_depth:.3f}-{max_depth:.3f} m)",
            fill=(255, 255, 255),
        )
        depth_image.save(depth_path, format="PNG", optimize=True)
        image_paths.extend((rgb_path.resolve(), depth_path.resolve()))
        manifest_frames.append(
            {
                "camera": label,
                "serial": frame.serial,
                "host_utc": frame.host_utc,
                "host_monotonic_ns": frame.host_monotonic_ns,
                "color_frame_number": frame.color_frame_number,
                "depth_frame_number": frame.depth_frame_number,
                "valid_depth_fraction": frame.valid_depth_fraction,
                "rgb": str(rgb_path.resolve()),
                "depth_visualization": str(depth_path.resolve()),
                "depth_m": str(depth_npy.resolve()),
                "depth_range_m": [min_depth, max_depth],
            }
        )
    manifest = {
        "created_at": _now(),
        "stage": safe_stage,
        "frames": manifest_frames,
        "image_paths": [str(path) for path in image_paths],
    }
    _write_json(output_dir / "snapshot_manifest.json", manifest)
    return image_paths, manifest


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
        # Heartbeats are rendered in-place only on a real terminal. Pipes,
        # StringIO, and redirected worker logs retain one complete line per
        # event so machine-readable/non-interactive logs never lose history.
        self._dynamic_terminal = bool(
            getattr(stream, "isatty", lambda: False)()
        )
        self._dynamic_line_active = False

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
            self._clear_dynamic_line_locked(add_newline=True)
            print("╭" + "─" * width + "╮", file=self.stream)
            print(
                "│  ClothAgent · Global Garment CLI".ljust(width + 1)
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
        with self._io_lock:
            self._clear_dynamic_line_locked(add_newline=True)

    def _clear_dynamic_line_locked(self, *, add_newline: bool = False) -> None:
        """Erase a previously rendered in-place heartbeat under ``_io_lock``."""

        if not self._dynamic_line_active:
            return
        # CR + erase-line also works when color output is disabled; the ANSI
        # sequence is terminal control, not semantic color formatting.
        self.stream.write("\r\033[2K")
        if add_newline:
            self.stream.write("\n")
        self.stream.flush()
        self._dynamic_line_active = False

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
            dynamic_wait = level == "WAIT" and self._dynamic_terminal
            if dynamic_wait:
                self.stream.write(f"\r\033[2K{prefix}{message}")
                self.stream.flush()
                self._dynamic_line_active = True
            else:
                self._clear_dynamic_line_locked()
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
            self._clear_dynamic_line_locked(add_newline=True)
            print(f"{prefix}{line}", end="", file=self.stream, flush=True)


@dataclass(frozen=True)
class KeypointCliOptions:
    planning_policy: str = "claude_global"
    # Optional visual annotation pass for Claude-global.  It is opt-in here so
    # lightweight library/test callers keep the historical no-Molmo behavior;
    # the real dashboard launcher enables it explicitly.
    global_molmo_annotations: bool = False
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
    min_gpu_free_mib: int = 19_000
    heartbeat_s: float = 10.0
    color: bool | None = None
    claude_binary: str = "claude"
    claude_timeout_s: int = 900
    claude_grounding_timeout_s: int = 120
    hold_checkpoint_timeout_s: int = 600
    hold_checkpoint_settle_s: float = 0.75
    max_replans: int = 1
    continue_on_recoverable_errors: bool = False
    # Zero is the unattended-run default: pre-execution failures are recorded
    # and retried without a consecutive-failure cap. Physical/hardware errors
    # remain non-recoverable and are never auto-replayed.
    max_consecutive_recoverable_failures: int = 0
    recovery_backoff_s: float = 2.0
    max_evaluation_retries: int = 0
    evaluation_retry_backoff_s: float = 2.0
    # Library/test callers stay side-effect free by default.  The executable
    # CLI opts into recording unless --no-record-rollouts is supplied.
    record_rollouts: bool = False
    recording_native: bool = True
    recording_codec: str = "mp4v"
    recording_warmup_frames: int | None = None
    objective: str = DEFAULT_EXPLORATION_OBJECTIVE
    rgb_only_comparison: bool = False
    # Cumulative rollout review videos are intentionally highly accelerated so
    # long physical runs remain easy to inspect.  Callers can still override
    # this explicitly with --combined-video-speed.
    combined_video_speed: float = 32.0


def _validate_options(options: KeypointCliOptions) -> KeypointCliOptions:
    if options.planning_policy not in {"claude_global", "semantic_local"}:
        raise ValueError(
            "planning_policy must be claude_global or semantic_local"
        )
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
    if options.max_consecutive_recoverable_failures < 0 or options.max_consecutive_recoverable_failures > 20:
        raise ValueError("max_consecutive_recoverable_failures must be between 0 and 20 (0 means unlimited)")
    if not 0 <= options.recovery_backoff_s <= 300:
        raise ValueError("recovery_backoff_s must be between 0 and 300 seconds")
    if not 0 <= options.max_evaluation_retries <= 100:
        raise ValueError("max_evaluation_retries must be between 0 and 100")
    if not 0 <= options.evaluation_retry_backoff_s <= 300:
        raise ValueError(
            "evaluation_retry_backoff_s must be between 0 and 300 seconds"
        )
    if not math.isfinite(options.combined_video_speed) or options.combined_video_speed <= 0:
        raise ValueError("combined_video_speed must be finite and positive")
    if not 30 <= options.claude_timeout_s <= 1200:
        raise ValueError("claude_timeout_s must be between 30 and 1200 seconds")
    if not 15 <= options.claude_grounding_timeout_s <= 400:
        raise ValueError("claude_grounding_timeout_s must be between 15 and 400 seconds")
    if not 15 <= options.hold_checkpoint_timeout_s <= 900:
        raise ValueError("hold_checkpoint_timeout_s must be between 15 and 900 seconds")
    if not 0 <= options.hold_checkpoint_settle_s <= 5:
        raise ValueError("hold_checkpoint_settle_s must be between 0 and 5 seconds")
    if not 30 <= options.molmo_timeout_s <= 3600:
        raise ValueError("molmo_timeout_s must be between 30 and 3600 seconds")
    if not 0 <= options.min_gpu_free_mib <= 24_564:
        raise ValueError("min_gpu_free_mib must be between 0 and 24564")
    if not 0 <= options.heartbeat_s <= 300:
        raise ValueError("heartbeat_s must be between 0 and 300 seconds")
    if len(options.recording_codec) != 4:
        raise ValueError("recording_codec must be a four-character code")
    if options.recording_warmup_frames is not None and not 0 <= options.recording_warmup_frames <= 300:
        raise ValueError("recording_warmup_frames must be between 0 and 300")
    if options.planning_policy == "semantic_local" and not options.keypoint_specs:
        raise ValueError("at least one keypoint spec is required")
    if options.planning_policy == "semantic_local" and (
        not options.keypoint_cameras or any(
        camera not in {"A", "B"} for camera in options.keypoint_cameras
        )
    ):
        raise ValueError("keypoint_cameras must contain A and/or B")
    if len(set(options.keypoint_cameras)) != len(options.keypoint_cameras):
        raise ValueError("keypoint_cameras must be unique")
    return options


def _is_recoverable_loop_error(exc: BaseException, record: dict[str, Any]) -> bool:
    """Allow a fresh perception/replan whenever no physical rollout started."""

    if record.get("execution") is not None:
        return False
    if isinstance(
        exc,
        (
            ExplorationTimeoutError,
            ExplorationPlanningError,
            ExperimentValidationError,
            SafetyError,
            MolmoKeypointPipelineError,
            PerceptionError,
        ),
    ):
        return True
    if isinstance(exc, (AutoExplorationError, SemanticPipelineError)):
        message = str(exc).lower()
        return not any(
            token in message
            for token in ("robot", "xarm", "set_position", "set_servo_angle", "physical rollout", "perception_position")
        )
    return False


def _recovery_skill_proposal(
    rejection: dict[str, Any],
    *,
    corrected_attempt: int,
) -> SkillProposal:
    """Turn a deterministically corrected pre-execution error into skill guidance."""

    error_type = str(rejection.get("error_type") or "PreexecutionError")
    error = str(rejection.get("error") or "unknown pre-execution error")
    normalized = f"{error_type} {error}".lower()
    if "controller ik" in normalized:
        name = "controller-ik-recovery"
        purpose = (
            "Recover safely when controller IK rejects a garment manipulation path "
            "before physical execution."
        )
        guidance = (
            "When controller IK rejects a pose or interpolated segment, preserve the "
            "supported visual objective but shorten extreme reach, lift, or transport; "
            "keep every waypoint inside the workspace, include a controlled release, "
            "and rerun preflight and controller IK before execution."
        )
    elif error_type == "ExperimentValidationError":
        name = "preflight-contract-recovery"
        purpose = (
            "Recover from a rejected action schema or unsafe action sequence before motion."
        )
        guidance = (
            "When static preflight rejects a plan, keep only the high-level garment "
            "objective, correct the reported action contract or sequence, keep all "
            "waypoints inside the workspace, include an explicit release, and rerun "
            "preflight and IK before execution."
        )
    elif error_type == "ExplorationPlanningError":
        name = "grounding-plan-recovery"
        purpose = (
            "Recover from an invalid or inconsistent grounded garment proposal before motion."
        )
        guidance = (
            "When grounding validation rejects a selected point or its use in the plan, "
            "reinspect the current evidence, choose a newly grounded visible fabric point, "
            "keep the path inside the workspace, include a controlled release, and rerun "
            "preflight and IK before execution."
        )
    elif error_type == "MolmoKeypointPipelineError":
        name = "molmo-annotation-retry"
        purpose = (
            "Recover from an auxiliary Molmo annotation worker failure before motion."
        )
        guidance = (
            "When auxiliary Molmo annotation fails before execution, preserve the saved "
            "perception evidence, retry the annotation on a fresh iteration, and never "
            "treat a missing semantic anchor as garment-state evidence; keep workspace, "
            "release, preflight, and IK validation mandatory before any later motion."
        )
    elif error_type == "PerceptionError":
        name = "perception-recapture-recovery"
        purpose = (
            "Recover from an invalid Camera A/B garment perception before planning motion."
        )
        guidance = (
            "When perception validation fails, send no motion, recapture the scene, and "
            "require a fresh validated garment state before planning; keep workspace, "
            "release, preflight, and IK checks mandatory after perception recovers."
        )
    elif error_type == "ExplorationTimeoutError":
        name = "planning-timeout-recovery"
        purpose = (
            "Recover from a Claude planning timeout before any garment motion starts."
        )
        guidance = (
            "When planning times out before execution, retain the exact failure context, "
            "start a fresh planning attempt from saved perception, keep all waypoints "
            "inside the workspace, include a controlled release, and rerun preflight "
            "and IK before execution."
        )
    else:
        name = "preexecution-safety-recovery"
        purpose = (
            "Recover from a deterministic safety rejection before garment motion starts."
        )
        guidance = (
            "When a deterministic safety gate rejects a proposal, feed the exact error "
            "and rejected plan back to the planner, make a materially different correction, "
            "keep all motion inside the workspace, include a controlled release, and rerun "
            "preflight and IK before execution."
        )
    return SkillProposal(
        operation="create",
        name=name,
        purpose=purpose,
        guidance=guidance,
        rationale=(
            "A Claude proposal failed a deterministic pre-execution gate, and a later "
            "Claude correction passed grounding, static preflight, and controller IK "
            "without sending the rejected motion to the robot."
        ),
        evidence=(
            f"Rejected attempt {rejection.get('attempt')}: {error_type}: {error}",
            (
                f"Corrected attempt {corrected_attempt} passed all pre-execution "
                "validation gates before any physical command."
            ),
        ),
        confidence=0.8,
    )


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
    project_root: Path | None = None,
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
    annotation_dir: Path | None = None
    for view in keypoint_manifest.get("views", []):
        overlay = Path(str(view.get("accepted_overlay", ""))).resolve()
        if overlay.is_file() and overlay not in paths:
            paths.append(overlay)
            annotation_dir = overlay.parent
    # The Molmo pipeline copies both the raw flat RGB reference and its
    # annotated anchor overlay into the current run.  Put both before the
    # current heatmaps so Claude first builds a topology/pattern hypothesis,
    # then uses geometry to verify a graspable instance of that structure.
    reference_paths: list[Path] = []
    if annotation_dir is not None:
        reference_dir = annotation_dir / "flat_reference"
        for name in (
            "camera_A_flat_reference.png",
            "camera_A_flat_reference_anchors.png",
        ):
            reference_image = reference_dir / name
            if reference_image.is_file() and reference_image not in reference_paths:
                reference_paths.append(reference_image.resolve())
    elif project_root is not None:
        # Global Claude may run without the optional Molmo pass.  Copy the
        # project reference into the current run so the CLI sandbox can still
        # use the same topology/pattern prior without granting access to files
        # outside the run directory.
        source_dir = (
            Path(project_root).resolve()
            / "data"
            / "reference"
            / "flat_garment_reference"
        )
        run_reference_dir = saved_path.parent / "flat_reference"
        for name in (
            "camera_A_flat_reference.png",
            "camera_A_flat_reference_anchors.png",
        ):
            source = source_dir / name
            destination = run_reference_dir / name
            if source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.is_file():
                    shutil.copy2(source, destination)
                reference_paths.append(destination.resolve())
    return reference_paths + [path for path in paths if path not in reference_paths]


def _rgb_only_comparison_images(image_paths: Sequence[Path]) -> list[Path]:
    """Keep only RGB/reference files for the no-height-map ablation."""

    allowed_names = {
        "camera_0_A.png",
        "camera_1_B.png",
        "camera_A.png",
        "camera_B.png",
        "camera_A_garment_only.png",
        "camera_B_garment_only.png",
        "camera_A_flat_reference.png",
        "camera_A_flat_reference_anchors.png",
    }
    selected: list[Path] = []
    for path in image_paths:
        resolved = Path(path).resolve()
        if resolved.name in allowed_names and resolved.is_file() and resolved not in selected:
            selected.append(resolved)
    if not selected:
        raise FileNotFoundError(
            "RGB-only comparison found no RGB/reference images in the planning bundle"
        )
    return selected


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
    global_client: ClaudeExplorationClient | None = None,
    global_evaluator: ClaudeAutoClient | None = None,
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
    combined_video_path = output / "combined_rollout.mp4"
    reporter = CliReporter(
        output / "events.jsonl",
        stream=stream,
        color=options.color,
    )
    if options.planning_policy == "semantic_local":
        client = client or SemanticClaudeClient(
            binary=options.claude_binary,
            strategy_timeout_s=options.claude_timeout_s,
            action_timeout_s=options.claude_grounding_timeout_s,
            evaluation_timeout_s=options.claude_timeout_s,
        )
        semantic_state_builder = semantic_state_builder or SemanticStateBuilder()
        local_geometry_grounder = local_geometry_grounder or LocalGeometryGrounder()
    else:
        global_client = global_client or ClaudeExplorationClient(
            binary=options.claude_binary,
            timeout_s=options.claude_timeout_s,
        )
        global_evaluator = global_evaluator or ClaudeAutoClient(
            binary=options.claude_binary,
            timeout_s=options.claude_timeout_s,
            grounding_timeout_s=options.claude_grounding_timeout_s,
        )
    # The lift-checkpoint budget is run-scoped.  Initialise it before the
    # summary is built (the summary records the value at startup), then
    # replace it with the persisted history value below once the experience
    # ledger has been loaded.
    lift_checkpoint_experiment_used = False
    summary: dict[str, Any] = {
        "created_at": _now(),
        "status": "RUNNING",
        "run_dir": str(session.run_dir),
        "output_dir": str(output),
        "enable_real": options.enable_real,
        "planning_policy": options.planning_policy,
        "rgb_only_comparison": options.rgb_only_comparison,
        "combined_rollout_video": str(combined_video_path),
        "combined_video_speed": options.combined_video_speed,
        "hold_checkpoint": {
            "mandatory": False,
            "replaced_by": "LIFT_CHECKPOINT_EXPERIMENT",
            "min_lift_mm": 40.0,
            "max_lateral_mm": 5.0,
            "settle_s": options.hold_checkpoint_settle_s,
            "claude_timeout_s": None,
        },
        "lift_checkpoint": {
            "mandatory": False,
            "decision_policy": (
                "CLAUDE_DECIDES; at most one physical experiment per run, then direct execution"
            ),
            "checkpoint_count": 3,
            "max_physical_experiments_per_run": 1,
            "used_at_start": lift_checkpoint_experiment_used,
            "used": lift_checkpoint_experiment_used,
            "remaining_experiments": 0 if lift_checkpoint_experiment_used else 1,
            "capture_settle_s": options.hold_checkpoint_settle_s,
            "transport_decision": "DISABLED; reverse lift and release",
        },
        "confidence_threshold": options.confidence_threshold,
        "max_iterations": options.max_iterations,
        "evaluation_retry": {
            "max_retries": options.max_evaluation_retries or "unlimited",
            "backoff_s": options.evaluation_retry_backoff_s,
            "timeout_only": True,
            "reuses_saved_before_after": True,
            "sends_robot_commands": False,
        },
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
            "Planning policy": options.planning_policy,
            "Iterations": options.max_iterations or "continuous",
            "Visual input": (
                "complete A/B scene + Molmo annotations; no candidates"
                if options.planning_policy == "claude_global"
                and options.global_molmo_annotations
                else "complete A/B scene; no candidates"
                if options.planning_policy == "claude_global"
                else f"{','.join(options.keypoint_cameras)} / {len(options.keypoint_specs)} anchors"
            ),
            "Confidence gate": (
                "not applicable"
                if options.planning_policy == "claude_global"
                else f"confidence > {options.confidence_threshold:.3f}"
            ),
            "Before each capture": (
                "Home → perception_position → settle"
                if options.enable_real
                else "disabled in dry run"
            ),
            "Heartbeat": (
                f"every {options.heartbeat_s:g}s" if options.heartbeat_s else "disabled"
            ),
            "Evaluation timeout": (
                "retry forever with saved before/after evidence"
                if options.max_evaluation_retries == 0
                else f"retry up to {options.max_evaluation_retries} time(s)"
            ),
            "Lift checkpoints": (
                f"Claude-selected; when enabled: 3 rising A/B captures + reverse release; "
                f"settle={options.hold_checkpoint_settle_s:g}s"
                if options.planning_policy == "claude_global"
                else "not applicable"
            ),
            "Results": output,
            "Combined rollout video": combined_video_path,
            "Combined video speed": f"{options.combined_video_speed:g}x",
        }
    )
    reporter.emit(
        "startup",
        (
            f"headless {options.planning_policy} loop started; physical execution "
            + ("ENABLED" if options.enable_real else "DISABLED (dry run)")
        ),
        payload={
            "output_dir": str(output),
            "planning_policy": options.planning_policy,
            "confidence_policy": f"confidence > {options.confidence_threshold}",
            "keypoint_cameras": list(options.keypoint_cameras),
            "keypoint_count": len(options.keypoint_specs),
            "perception_position_sequence": ["home", "perception_position"],
            "perception_position_enabled": options.enable_real,
            "lift_checkpoint_count": 3,
            "lift_checkpoint_min_lift_mm": 40.0,
            "lift_checkpoint_max_lateral_mm": 5.0,
            "lift_checkpoint_settle_s": options.hold_checkpoint_settle_s,
        },
        level="WARNING" if options.enable_real else "INFO",
    )
    reporter.start_heartbeat(options.heartbeat_s)

    def append_recording_to_cumulative_video(
        iteration_number: int,
        iteration_record: dict[str, Any],
        recording_directory: Path,
        execution_record: dict[str, Any] | None,
    ) -> None:
        """Append the completed composite segment without affecting execution."""

        source_video = recording_directory / "composite_AB_depth.mp4"
        if not source_video.is_file():
            return
        labelled_source = recording_directory / ".composite_AB_depth.iteration.mp4"
        speed_segment = recording_directory / ".composite_AB_depth.speed.mp4"
        try:
            rollout_recording = iteration_record.get("rollout_recording")
            recording_manifest = (
                rollout_recording.get("manifest")
                if isinstance(rollout_recording, dict)
                else None
            )
            phase_timeline = build_rollout_phase_timeline(
                execution_record,
                recording_manifest,
            )
            label_info = label_iteration_mp4(
                source_video,
                labelled_source,
                iteration=iteration_number,
                phase_timeline=phase_timeline,
            )
            speed_info = speed_up_mp4(
                labelled_source,
                speed_segment,
                speed=options.combined_video_speed,
            )
            append_info = append_mp4_to_cumulative(
                speed_segment,
                combined_video_path,
            )
            append_info["playback_speed"] = options.combined_video_speed
            append_info["iteration"] = iteration_number
            append_info["iteration_label"] = label_info
            append_info["process_phase_count"] = len(phase_timeline)
            append_info["speed_up"] = speed_info
        except Exception as exc:
            iteration_record.setdefault("rollout_recording", {})[
                "cumulative_video_error"
            ] = f"{type(exc).__name__}: {exc}"
            reporter.emit(
                "combined-video",
                f"failed to append rollout video; keeping source segment: {exc}",
                iteration=iteration_number,
                level="WARNING",
            )
            return
        finally:
            labelled_source.unlink(missing_ok=True)
            speed_segment.unlink(missing_ok=True)
        iteration_record.setdefault("rollout_recording", {})[
            "cumulative_video"
        ] = str(combined_video_path)
        iteration_record.setdefault("rollout_recording", {})[
            "cumulative_video_append"
        ] = append_info
        iteration_record.setdefault("artifacts", {})[
            "combined_rollout_video"
        ] = str(combined_video_path)
        reporter.emit(
            "combined-video",
            f"appended rollout composite to {combined_video_path}",
            iteration=iteration_number,
            level="PASS",
            payload=append_info,
        )

    def prune_completed_recording(
        iteration_number: int,
        iteration_record: dict[str, Any],
        recording_directory: Path | None,
    ) -> None:
        """Remove per-iteration video/native files after evaluator completion."""

        if recording_directory is None:
            return
        removed = prune_rollout_video_files(recording_directory)
        if not removed:
            return
        iteration_record.setdefault("rollout_recording", {})[
            "pruned_video_files"
        ] = removed
        reporter.emit(
            "combined-video",
            f"removed {len(removed)} old per-iteration video/native file(s); cumulative video retained",
            iteration=iteration_number,
            level="INFO",
            payload={"removed": removed, "cumulative_video": str(combined_video_path)},
        )

    def prepare_real_perception_position(
        iteration_number: int,
        iteration_record: dict[str, Any],
        record_key: str,
    ) -> None:
        nonlocal perception_position_ready
        if not options.enable_real:
            return
        if perception_position_ready:
            iteration_record[record_key] = {
                "name": "perception_position",
                "skipped": True,
                "reason": "already at verified perception_position from previous capture",
            }
            reporter.emit(
                "robot-positioning",
                "already at verified perception_position; skipping redundant Home → perception_position",
                iteration=iteration_number,
                level="PASS",
            )
            return
        reporter.start_phase(
            "robot-positioning",
            "moving robot Home → perception_position before RGB-D capture",
            iteration=iteration_number,
        )
        outcome = perception_positioner(session.robot_config)
        iteration_record[record_key] = _jsonable(outcome)
        actual_pose = outcome.get("actual_tcp_pose_mm_deg")
        perception_position_ready = True
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

    def evaluate_with_timeout_retry(
        iteration_number: int,
        iteration_directory: Path,
        iteration_record: dict[str, Any],
        evaluate_once: Callable[[], Any],
    ) -> Any:
        """Retry only Claude evaluation timeouts without repeating robot work."""

        attempt = 0
        while True:
            attempt += 1
            iteration_record["evaluation_attempt_count"] = attempt
            try:
                evaluation_result = evaluate_once()
            except ExplorationTimeoutError as exc:
                retry_count = attempt
                retry_limit = options.max_evaluation_retries
                exhausted = retry_limit > 0 and retry_count > retry_limit
                retry_record = {
                    "attempt": attempt,
                    "created_at": _now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "same_saved_before_after": True,
                    "robot_command_sent": False,
                    "retry_scheduled": not exhausted,
                }
                iteration_record.setdefault(
                    "evaluation_timeout_retries", []
                ).append(retry_record)
                iteration_record["evaluation_retry_count"] = retry_count
                iteration_record["status"] = (
                    "EVALUATION_RETRY_EXHAUSTED"
                    if exhausted
                    else "EVALUATION_RETRYING"
                )
                iteration_record["evaluation_retry"] = {
                    "max_retries": retry_limit or "unlimited",
                    "backoff_s": options.evaluation_retry_backoff_s,
                    "next_attempt": None if exhausted else attempt + 1,
                    "evidence_policy": "reuse saved before/after images",
                    "robot_policy": "no robot or camera command during retry",
                }
                _iteration_checkpoint(
                    output,
                    iteration_directory,
                    iteration_number,
                    iteration_record,
                )
                if exhausted:
                    reporter.emit(
                        "evaluation-retry",
                        (
                            "Claude evaluation timeout retry limit exhausted; "
                            "no robot command was repeated"
                        ),
                        iteration=iteration_number,
                        level="ERROR",
                        payload=iteration_record["evaluation_retry"],
                    )
                    raise
                limit_text = (
                    "unlimited"
                    if retry_limit == 0
                    else str(retry_limit)
                )
                reporter.emit(
                    "evaluation-retry",
                    (
                        f"Claude evaluation timed out on attempt {attempt}; "
                        f"retrying the same saved before/after evidence "
                        f"(retry {retry_count}/{limit_text}); no robot or camera command"
                    ),
                    iteration=iteration_number,
                    level="WARNING",
                    payload=retry_record,
                )
                if options.evaluation_retry_backoff_s > 0:
                    sleep(options.evaluation_retry_backoff_s)
                continue
            iteration_record["evaluation_attempt_count"] = attempt
            iteration_record["evaluation_retry_count"] = attempt - 1
            iteration_record["status"] = "RUNNING"
            if attempt > 1:
                timeout_entries = iteration_record.get(
                    "evaluation_timeout_retries", []
                )
                timeout_evidence = tuple(
                    str(item.get("error"))
                    for item in timeout_entries
                    if isinstance(item, dict) and item.get("error")
                )
                recovery_skill = SkillProposal(
                    operation="create",
                    name="evaluation-timeout-retry",
                    purpose=(
                        "Recover from a Claude evaluation timeout after a completed "
                        "garment rollout without repeating physical work."
                    ),
                    guidance=(
                        "When evaluation times out after execution and return-Home, "
                        "reuse the saved before/after evidence, do not resend robot or "
                        "camera commands or repeat the release, retain the completed "
                        "workspace and IK results as provenance, and retry only the "
                        "evaluation call."
                    ),
                    rationale=(
                        "The evaluator timed out after physical execution was already "
                        "complete, and retrying only the same saved evidence later "
                        "returned a valid evaluation."
                    ),
                    evidence=(
                        *timeout_evidence[:7],
                        (
                            f"Evaluation attempt {attempt} succeeded using the same saved "
                            "before/after evidence with no robot or camera command."
                        ),
                    ),
                    confidence=0.9,
                )
                approved_names = {
                    skill.name for skill in skill_store.approved()
                }
                if recovery_skill.name in approved_names:
                    skill_review = {
                        "status": "ALREADY_APPROVED",
                        "approved": True,
                        "reason": f"skill {recovery_skill.name} is already active",
                    }
                else:
                    reviewed = run_skill_ledger.stage_skill_update(
                        recovery_skill,
                        iteration=iteration_number,
                        source="evaluation_timeout_recovery",
                    )
                    skill_review = (
                        reviewed.as_dict()
                        if reviewed is not None
                        else {
                            "status": "NOT_PROPOSED",
                            "approved": False,
                            "reason": "skill review returned no result",
                        }
                    )
                recovery_experience = {
                    "created_at": _now(),
                    "type": "evaluation_timeout_recovery",
                    "iteration": iteration_number,
                    "timeout_attempts": list(timeout_entries),
                    "successful_attempt": attempt,
                    "same_saved_before_after": True,
                    "robot_command_repeated": False,
                    "camera_command_repeated": False,
                    "skill_proposal": recovery_skill.as_dict(),
                    "skill_review": skill_review,
                }
                iteration_record["evaluation_error_recovery"] = (
                    recovery_experience
                )
                iteration_record["evaluation_recovery_skill_review"] = (
                    skill_review
                )
                _write_json(
                    iteration_directory / "evaluation_error_recovery.json",
                    recovery_experience,
                )
                if options.planning_policy == "claude_global":
                    _append_jsonl(global_experience_path, recovery_experience)
                    global_experiences.append(recovery_experience)
                run_skill_ledger.append_experience(recovery_experience)
                if global_client is not None:
                    global_client.skill_names = tuple(
                        skill.name for skill in skill_store.approved()
                    )
                reporter.emit(
                    "evaluation-recovery",
                    (
                        f"evaluation retry succeeded; skill={recovery_skill.name} "
                        f"review={skill_review.get('status')}; no robot or camera command"
                    ),
                    iteration=iteration_number,
                    level="PASS" if skill_review.get("approved") else "WARNING",
                    payload=recovery_experience,
                )
            return evaluation_result

    experience_path = session.workspace / "structured_experience.jsonl"
    experiences = load_structured_experiences(experience_path)
    global_experience_path = session.workspace / "global_experience.jsonl"
    global_experiences = _load_jsonl(global_experience_path)
    # A lift-checkpoint is an experiment, not a reusable action primitive. Once
    # one physical checkpoint experiment has been executed in this run, later
    # iterations must use direct Claude actions instead of repeating the probe.
    lift_checkpoint_experiment_used = any(
        isinstance(item, dict)
        and isinstance(item.get("proposal"), dict)
        and item["proposal"].get("requires_lift_checkpoint") is True
        for item in global_experiences
    )
    # Keep the already-written summary truthful when a resumed run contains a
    # prior checkpoint experiment.  New runs retain the initial False value.
    summary["lift_checkpoint"]["used_at_start"] = lift_checkpoint_experiment_used
    summary["lift_checkpoint"]["used"] = lift_checkpoint_experiment_used
    summary["lift_checkpoint"]["remaining_experiments"] = (
        0 if lift_checkpoint_experiment_used else 1
    )
    _write_json(output / "summary.json", summary)
    skill_store = SkillStore(session.project_root / "data" / "skills")
    run_skill_ledger = RunSkillLedger(session.workspace)

    def run_skill_prompt() -> str:
        appendix = run_skill_ledger.prompt_appendix()
        return skill_store.prompt() + (("\n\n" + appendix) if appendix else "")
    if global_client is not None:
        global_client.skill_names = tuple(
            skill.name for skill in skill_store.approved()
        )
    iteration = 0
    consecutive_recoverable_failures = 0
    pending_recoverable_error: dict[str, Any] | None = None
    perception_position_ready = False
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
            "lift_checkpoint_experiment_used_at_start": lift_checkpoint_experiment_used,
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

            global_molmo_manifest: dict[str, Any] | None = None
            if options.planning_policy == "claude_global" and options.global_molmo_annotations:
                if options.min_gpu_free_mib:
                    reporter.start_phase(
                        "gpu-preflight",
                        "checking GPU 0 free memory before loading Molmo annotations",
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
                        f"free={free_mib} MiB required>={options.min_gpu_free_mib} MiB",
                        success=free_mib >= options.min_gpu_free_mib,
                        level="PASS" if free_mib >= options.min_gpu_free_mib else "ERROR",
                    )
                    if free_mib < options.min_gpu_free_mib:
                        raise AutoExplorationError(
                            "insufficient free GPU memory before global Molmo annotation: "
                            f"{free_mib} MiB available, at least {options.min_gpu_free_mib} MiB required"
                        )
                    _record_stage(
                        output,
                        iteration_dir,
                        iteration,
                        record,
                        "GPU_PREFLIGHT_COMPLETED",
                    )
                reporter.start_phase(
                    "molmo",
                    (
                        "axis-first Molmo semantic annotation for Claude-global; "
                        f"Camera {','.join(options.keypoint_cameras)}"
                    ),
                    iteration=iteration,
                )
                keypoint_dir = iteration_dir / "semantic_anchors"
                global_molmo_manifest = keypoint_runner(
                    project_root=session.project_root,
                    perception_dir=session.workspace / "perception_views",
                    artifact_dir=keypoint_dir,
                    confidence_threshold=options.confidence_threshold,
                    molmo_python=options.molmo_python,
                    model=options.molmo_model,
                    timeout_s=options.molmo_timeout_s,
                    local_files_only=not options.molmo_allow_download,
                    keypoint_specs=(options.keypoint_specs or DEFAULT_SEMANTIC_ANCHORS),
                    cameras=options.keypoint_cameras,
                    install=True,
                    worker_line_callback=reporter.worker_line,
                )
                record["semantic_anchors"] = global_molmo_manifest
                record["artifacts"]["semantic_anchors"] = str(
                    keypoint_dir / "molmo_semantic_anchors.json"
                )
                _print_semantic_anchors(reporter, iteration, global_molmo_manifest)
                _record_stage(
                    output,
                    iteration_dir,
                    iteration,
                    record,
                    "SEMANTIC_ANNOTATIONS_COMPLETED",
                )
                reporter.finish_phase(
                    (
                        f"annotated {global_molmo_manifest.get('anchor_count', 0)} "
                        f"semantic region(s); artifacts saved in {keypoint_dir}"
                    ),
                    success=True,
                )

            if options.planning_policy == "claude_global":
                if global_client is None or global_evaluator is None:
                    raise AutoExplorationError(
                        "claude_global clients were not initialized"
                    )
                before_images = (
                    _planning_images(
                        saved,
                        saved_path,
                        global_molmo_manifest or {},
                        session.project_root,
                    )
                    if global_molmo_manifest is not None
                    else global_perception_image_paths(saved, saved_path)
                )
                record["before_images"] = [str(path) for path in before_images]
                record["candidate_policy"] = "NONE_CLAUDE_SELECTS_ARBITRARY_PIXEL"
                proposal: ExplorationProposal | None = None
                grounding: dict[str, Any] | None = None
                probe_profile: dict[str, Any] | None = None
                checkpoint_plan = None
                source = ""
                preflight = None
                controller = None
                checkpoint_abort_controller = None
                validation_feedback: str | None = None
                for attempt in range(1, options.max_replans + 2):
                    reporter.start_phase(
                        "global-planning",
                        (
                            "Claude inspecting the complete A/B scene with Camera A as the "
                            "action view, summarizing state, and choosing an arbitrary "
                            f"Camera A pixel; attempt {attempt}/"
                            f"{options.max_replans + 1}"
                        ),
                        iteration=iteration,
                    )
                    prompt = exploration_prompt(
                        session.experiment_config,
                        session.robot_config,
                        objective=options.objective,
                        history=global_experiences,
                        history_file=(
                            global_experience_path.relative_to(session.run_dir)
                            if global_experiences
                            else None
                        ),
                        skill_guidance=run_skill_prompt(),
                    )
                    prompt += (
                        "\n\nYou must explicitly decide whether this proposal needs a "
                        "lift-checkpoint experiment and return `requires_lift_checkpoint` as "
                        "true or false. Set it true when the selected structure, graspability, "
                        "layer identity, or expected cloth response is uncertain and direct "
                        "transport would be speculative. When true, runtime will grasp, capture "
                        "three rising Camera A/B checkpoints, reverse the lift, release, and "
                        "return Home; it will not execute lateral transport in this iteration. "
                        "Set it false only when the target and intended action are sufficiently "
                        "supported to execute the complete action list directly; runtime will "
                        "then skip all lift checkpoints and the online hold classifier.\n"
                    )
                    if lift_checkpoint_experiment_used:
                        prompt += (
                            "A physical lift-checkpoint experiment has already been used "
                            "in this run. Do not request another probe: return "
                            "`requires_lift_checkpoint=false` and plan the next direct "
                            "action from the new evidence. The one-probe budget is spent "
                            "even if the previous probe failed; change the grasp region, "
                            "transport, or release strategy instead of repeating the same "
                            "lift-and-release test.\n"
                        )
                    if global_molmo_manifest is not None:
                        prompt += (
                            "\n\nAn axis-first Molmo semantic annotation pass was run for "
                            "visual assistance. Its accepted semantic anchors are auxiliary "
                            "evidence, not a candidate list or mandatory grasp target; choose "
                            "the final arbitrary Camera A pixel yourself. See the annotated "
                            "overlay images and manifest at: "
                            f"{iteration_dir / 'semantic_anchors' / 'molmo_semantic_anchors.json'}\n"
                            f"Molmo status={global_molmo_manifest.get('status')} "
                            f"anchor_count={global_molmo_manifest.get('anchor_count', 0)}."
                        )
                    if validation_feedback:
                        prompt += (
                            "\n\nThe previous proposal was rejected before motion by a hard "
                            "runtime validation. Reinspect the complete scene and choose a "
                            "new safe pixel/action that directly corrects this exact failure; "
                            "do not bypass the gate:\n"
                            f"{validation_feedback}\n"
                        )
                    try:
                        response = invoke_direct_prompt(
                            global_client,
                            before_images,
                            prompt,
                            session.run_dir,
                        )
                        proposal, grounding = ground_global_grasp_target(
                            response.proposal,
                            session.workspace / "perception_views",
                            robot_config=session.robot_config,
                        )
                        if (
                            lift_checkpoint_experiment_used
                            and proposal.requires_lift_checkpoint
                        ):
                            record["checkpoint_decision_override"] = {
                                "requested": True,
                                "applied": False,
                                "reason": (
                                    "one physical lift-checkpoint experiment has already "
                                    "been used in this run; repeated probing is disabled"
                                ),
                            }
                            proposal = replace(
                                proposal,
                                requires_lift_checkpoint=False,
                            )
                        if proposal.requires_lift_checkpoint:
                            probe_profile = validate_global_probe_profile(
                                proposal,
                                global_experiences,
                                measurement=(grounding or {}).get("measurement"),
                            )
                            probe_profile["hold_settle_s"] = (
                                options.hold_checkpoint_settle_s
                            )
                        else:
                            probe_profile = {
                                "mode": "DIRECT_EXECUTION",
                                "requires_online_hold": False,
                                "online_checkpoint_required": False,
                                "transport_decision": "CLAUDE_DIRECT",
                            }
                        if proposal.requires_lift_checkpoint:
                            checkpoint_plan = split_global_lift_checkpoint_plan(
                                proposal,
                                checkpoint_count=3,
                            )
                            # This iteration is deliberately a lift-only experiment:
                            # capture evidence at three rising waypoints, then reverse
                            # those waypoints and release. Claude's lateral continuation
                            # remains in the proposal for provenance but is not executed.
                            lift_proposal = replace(
                                proposal,
                                actions=checkpoint_plan.actions,
                            )
                            source = exploration_source(lift_proposal)
                        else:
                            # Claude judged the structure sufficiently actionable;
                            # execute its validated proposal directly without adding
                            # an online hold/checkpoint experiment.
                            checkpoint_plan = None
                            source = exploration_source(proposal)
                        source_path.write_text(source, encoding="utf-8")
                        preflight = session.runner.preflight(source_path.name)
                        if preflight.error:
                            raise ExperimentValidationError(preflight.error)
                        if options.skip_controller_ik:
                            controller = {"status": "SKIPPED_DRY_RUN"}
                            checkpoint_abort_controller = {
                                "status": "SKIPPED_DRY_RUN"
                            }
                        else:
                            controller = controller_validator(
                                session.robot_config, preflight.actions
                            )
                            # Validate the reverse-laydown branch independently as
                            # well; it is the only branch that will run after the
                            # final checkpoint.
                            checkpoint_abort_controller = (
                                controller_validator(
                                    session.robot_config,
                                    list(
                                        checkpoint_plan.acquisition_actions
                                        + checkpoint_plan.return_actions
                                    ),
                                )
                                if checkpoint_plan is not None
                                else {"status": "NOT_REQUIRED"}
                            )
                    except Exception as exc:
                        reporter.fail_current_phase(
                            f"{type(exc).__name__}: {exc}"
                        )
                        rejected_proposal = (
                            proposal.as_dict() if proposal is not None else None
                        )
                        rejected_requires_lift = (
                            proposal.requires_lift_checkpoint
                            if proposal is not None
                            else None
                        )
                        if rejected_requires_lift is False:
                            decision_recovery_policy = (
                                "The rejected proposal explicitly chose "
                                "requires_lift_checkpoint=false. Preserve that decision in "
                                "the correction unless this exact error is evidence that "
                                "the grasp structure cannot be judged safely. For IK, "
                                "workspace, reach, or waypoint errors, keep false and only "
                                "repair the selected pixel or motion geometry; do not add "
                                "a lift-checkpoint experiment as a generic workaround."
                            )
                        elif rejected_requires_lift is True:
                            decision_recovery_policy = (
                                "The rejected proposal explicitly chose "
                                "requires_lift_checkpoint=true. Preserve that decision "
                                "unless the exact error proves the checkpoint experiment "
                                "itself is unnecessary."
                            )
                        else:
                            decision_recovery_policy = (
                                "The checkpoint decision was unavailable; make an explicit "
                                "requires_lift_checkpoint decision from the fresh evidence."
                            )
                        feedback_payload = {
                            "attempt": attempt,
                            "phase": "global_preexecution_validation",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "rejected_proposal": rejected_proposal,
                            "rejected_actions": (
                                rejected_proposal.get("actions")
                                if isinstance(rejected_proposal, dict)
                                else None
                            ),
                            "rejected_requires_lift_checkpoint": rejected_requires_lift,
                            "decision_recovery_policy": decision_recovery_policy,
                            "feedback_target": "next Claude planning attempt",
                            "required_response": (
                                "materially change the rejected pixel, waypoint geometry, "
                                "or action sequence as required by the exact error; pass all "
                                "grounding, preflight, workspace, and controller IK gates"
                            ),
                            "physical_command_sent": False,
                        }
                        validation_feedback = json.dumps(
                            feedback_payload, ensure_ascii=False, indent=2
                        )
                        record.setdefault("global_planning_rejections", []).append(
                            feedback_payload
                        )
                        record["error_feedback_to_claude"] = feedback_payload
                        _iteration_checkpoint(
                            output, iteration_dir, iteration, record
                        )
                        if (
                            attempt >= options.max_replans + 1
                            or not _is_preexecution_replan_error(exc)
                        ):
                            raise
                        proposal = None
                        grounding = None
                        checkpoint_plan = None
                        continue
                    record.setdefault("claude_global_attempts", []).append(
                        {
                            "prompt": response.prompt,
                            "command": list(response.command),
                            "returncode": response.returncode,
                            "stdout": response.stdout,
                            "stderr": response.stderr,
                            "created_at": getattr(response, "created_at", None),
                            "proposal": proposal.as_dict(),
                        }
                    )
                    rejections = record.get("global_planning_rejections") or []
                    if rejections:
                        last_rejection = dict(rejections[-1])
                        corrected_proposal = proposal.as_dict()
                        materially_changed = (
                            last_rejection.get("rejected_proposal")
                            != corrected_proposal
                        )
                        recovery_experience = {
                            "created_at": _now(),
                            "type": "preexecution_error_recovery",
                            "iteration": iteration,
                            "error_feedback": last_rejection,
                            "corrected_attempt": attempt,
                            "corrected_proposal": corrected_proposal,
                            "validation_result": {
                                "grounding": "PASSED",
                                "static_preflight": "PASSED",
                                "controller_ik": (
                                    "SKIPPED_DRY_RUN"
                                    if options.skip_controller_ik
                                    else "PASSED"
                                ),
                                "physical_command_sent_during_correction": False,
                                "materially_changed": materially_changed,
                            },
                        }
                        recovery_skill = _recovery_skill_proposal(
                            last_rejection,
                            corrected_attempt=attempt,
                        )
                        approved_names = {
                            skill.name for skill in skill_store.approved()
                        }
                        if not materially_changed:
                            recovery_skill_review = {
                                "status": "NOT_PROPOSED",
                                "approved": False,
                                "reason": (
                                    "corrected proposal is identical to the rejected "
                                    "proposal, so no reusable correction was evidenced"
                                ),
                            }
                        elif recovery_skill.name in approved_names:
                            recovery_skill_review = {
                                "status": "ALREADY_APPROVED",
                                "approved": True,
                                "reason": (
                                    f"skill {recovery_skill.name} is already active"
                                ),
                            }
                        else:
                            reviewed = run_skill_ledger.stage_skill_update(
                                recovery_skill,
                                iteration=iteration,
                                source="preexecution_error_recovery",
                            )
                            recovery_skill_review = (
                                reviewed.as_dict()
                                if reviewed is not None
                                else {
                                    "status": "NOT_PROPOSED",
                                    "approved": False,
                                    "reason": "skill review returned no result",
                                }
                            )
                        recovery_experience["skill_proposal"] = (
                            recovery_skill.as_dict()
                        )
                        recovery_experience["skill_review"] = (
                            recovery_skill_review
                        )
                        record["preexecution_error_recovery"] = recovery_experience
                        record["recovery_skill_review"] = recovery_skill_review
                        _write_json(
                            iteration_dir / "preexecution_error_recovery.json",
                            recovery_experience,
                        )
                        _append_jsonl(
                            global_experience_path, recovery_experience
                        )
                        run_skill_ledger.append_experience(recovery_experience)
                        global_experiences.append(recovery_experience)
                        if global_client is not None:
                            global_client.skill_names = tuple(
                                skill.name for skill in skill_store.approved()
                            )
                        reporter.emit(
                            "error-recovery",
                            (
                                f"Claude correction passed pre-execution gates; "
                                f"recovery skill={recovery_skill.name} "
                                f"review={recovery_skill_review.get('status')}"
                            ),
                            iteration=iteration,
                            level=(
                                "PASS"
                                if recovery_skill_review.get("approved")
                                else "WARNING"
                            ),
                            payload=recovery_experience,
                        )
                        pending_recoverable_error = None
                    reporter.finish_phase(
                        (
                            f"selected pixel={proposal.selected_grasp['camera']}/"
                            f"{proposal.selected_grasp['pixel_xy']} and passed grounding, "
                            f"static workspace, and controller gates"
                        ),
                        payload={
                            "selected_grasp": proposal.selected_grasp,
                            "grounding": grounding,
                        },
                    )
                    break
                if (
                    proposal is None
                    or grounding is None
                    or preflight is None
                    or controller is None
                    or checkpoint_abort_controller is None
                    or (
                        proposal.requires_lift_checkpoint
                        and checkpoint_plan is None
                    )
                ):
                    raise AutoExplorationError(
                        "global planning ended without a validated proposal"
                    )

                if pending_recoverable_error is not None and not record.get(
                    "global_planning_rejections"
                ):
                    recovery_skill = _recovery_skill_proposal(
                        pending_recoverable_error,
                        corrected_attempt=1,
                    )
                    approved_names = {
                        skill.name for skill in skill_store.approved()
                    }
                    if recovery_skill.name in approved_names:
                        recovery_skill_review = {
                            "status": "ALREADY_APPROVED",
                            "approved": True,
                            "reason": f"skill {recovery_skill.name} is already active",
                        }
                    else:
                        reviewed = run_skill_ledger.stage_skill_update(
                            recovery_skill,
                            iteration=iteration,
                            source="fresh_iteration_error_recovery",
                        )
                        recovery_skill_review = (
                            reviewed.as_dict()
                            if reviewed is not None
                            else {
                                "status": "NOT_PROPOSED",
                                "approved": False,
                                "reason": "skill review returned no result",
                            }
                        )
                    recovery_experience = {
                        "created_at": _now(),
                        "type": "fresh_iteration_error_recovery",
                        "iteration": iteration,
                        "error_feedback": pending_recoverable_error,
                        "corrected_attempt": 1,
                        "corrected_proposal": proposal.as_dict(),
                        "validation_result": {
                            "grounding": "PASSED",
                            "static_preflight": "PASSED",
                            "controller_ik": (
                                "SKIPPED_DRY_RUN"
                                if options.skip_controller_ik
                                else "PASSED"
                            ),
                            "physical_command_sent_during_correction": False,
                        },
                        "skill_proposal": recovery_skill.as_dict(),
                        "skill_review": recovery_skill_review,
                    }
                    record["preexecution_error_recovery"] = recovery_experience
                    record["recovery_skill_review"] = recovery_skill_review
                    _write_json(
                        iteration_dir / "preexecution_error_recovery.json",
                        recovery_experience,
                    )
                    _append_jsonl(global_experience_path, recovery_experience)
                    run_skill_ledger.append_experience(recovery_experience)
                    global_experiences.append(recovery_experience)
                    global_client.skill_names = tuple(
                        skill.name for skill in skill_store.approved()
                    )
                    reporter.emit(
                        "error-recovery",
                        (
                            "fresh iteration passed pre-execution gates after the "
                            f"previous error; recovery skill={recovery_skill.name} "
                            f"review={recovery_skill_review.get('status')}"
                        ),
                        iteration=iteration,
                        level=(
                            "PASS"
                            if recovery_skill_review.get("approved")
                            else "WARNING"
                        ),
                        payload=recovery_experience,
                    )
                    pending_recoverable_error = None

                record["proposal"] = proposal.as_dict()
                record["global_grounding"] = grounding
                record["probe_profile"] = probe_profile
                if checkpoint_plan is not None:
                    record["hold_checkpoint_plan"] = checkpoint_plan.as_dict()
                    record["lift_checkpoint_plan"] = checkpoint_plan.as_dict()
                selected_pixel_overlay = _save_global_selected_pixel_overlay(
                    proposal,
                    saved,
                    saved_path,
                    iteration_dir / "claude_selected_pixel.png",
                )
                record["artifacts"]["claude_selected_pixel"] = str(
                    selected_pixel_overlay
                )
                record["proposal_source"] = source
                record["preflight"] = _jsonable(preflight)
                record["controller_ik"] = _jsonable(controller)
                record["checkpoint_abort_controller_ik"] = _jsonable(
                    checkpoint_abort_controller
                )
                (iteration_dir / "proposal.py").write_text(source, encoding="utf-8")
                _write_json(iteration_dir / "proposal.json", proposal.as_dict())
                _write_json(iteration_dir / "global_grounding.json", grounding)
                _write_json(iteration_dir / "preflight.json", preflight)
                _write_json(iteration_dir / "controller_ik.json", controller)
                _write_json(
                    iteration_dir / "checkpoint_abort_controller_ik.json",
                    checkpoint_abort_controller,
                )
                if checkpoint_plan is not None:
                    _write_json(
                        iteration_dir / "hold_checkpoint_plan.json",
                        checkpoint_plan.as_dict(),
                    )
                    _write_json(
                        iteration_dir / "lift_checkpoint_plan.json",
                        checkpoint_plan.as_dict(),
                    )
                if options.rgb_only_comparison:
                    reporter.start_phase(
                        "rgb-only-comparison",
                        "running a non-executing Claude thought with RGB/reference only",
                        iteration=iteration,
                    )
                    comparison_images: list[Path] = []
                    try:
                        comparison_images = _rgb_only_comparison_images(before_images)
                        comparison = global_client.invoke_rgb_only_comparison(
                            comparison_images,
                            objective=options.objective,
                            run_dir=session.run_dir,
                        )
                        comparison["status"] = (
                            "COMPLETED" if comparison.get("parse_error") is None else "PARSE_ERROR"
                        )
                    except Exception as exc:
                        comparison = {
                            "mode": "rgb_only_no_height_map",
                            "objective": options.objective,
                            "image_paths": [str(path) for path in comparison_images],
                            "status": "FAILED",
                            "error": f"{type(exc).__name__}: {exc}",
                            "grasp_trajectory": [],
                            "physical_command_sent": False,
                        }
                        reporter.emit(
                            "rgb-only-comparison",
                            f"comparison thought failed; primary rollout is unaffected: {comparison['error']}",
                            iteration=iteration,
                            level="WARNING",
                            payload=comparison,
                        )
                    record["rgb_only_comparison"] = comparison
                    comparison_path = iteration_dir / "rgb_only_comparison.json"
                    trajectory_path = iteration_dir / "rgb_only_comparison_trajectory.json"
                    _write_json(comparison_path, comparison)
                    _write_json(
                        trajectory_path,
                        {
                            "mode": comparison.get("mode"),
                            "objective": comparison.get("objective"),
                            "selected_grasp": (
                                (comparison.get("parsed_output") or {}).get("selected_grasp")
                                if isinstance(comparison.get("parsed_output"), dict)
                                else None
                            ),
                            "grasp_trajectory": comparison.get("grasp_trajectory", []),
                            "physical_command_sent": False,
                        },
                    )
                    record["artifacts"]["rgb_only_comparison"] = str(comparison_path)
                    record["artifacts"]["rgb_only_comparison_trajectory"] = str(
                        trajectory_path
                    )
                    reporter.finish_phase(
                        (
                            f"saved RGB-only comparison output and trajectory; "
                            f"status={comparison.get('status')}"
                        ),
                        success=comparison.get("status") == "COMPLETED",
                        level=(
                            "PASS"
                            if comparison.get("status") == "COMPLETED"
                            else "WARNING"
                        ),
                        payload={
                            "comparison_output": str(comparison_path),
                            "comparison_trajectory": str(trajectory_path),
                            "physical_command_sent": False,
                        },
                    )
                _record_stage(
                    output,
                    iteration_dir,
                    iteration,
                    record,
                    "GLOBAL_PREEXECUTION_VALIDATED",
                )
                reporter.emit(
                    "global-plan",
                    (
                        "Claude selected the interaction directly from the complete scene; "
                        "no Sxxx/Rxxx candidate generation or ranking was run"
                    ),
                    iteration=iteration,
                    level="PASS",
                    payload=proposal.as_dict(),
                )
                print(source, file=reporter.stream, flush=True)

                if not options.enable_real:
                    record["status"] = "DRY_RUN_VALIDATED"
                    record["completed_at"] = _now()
                    _iteration_checkpoint(output, iteration_dir, iteration, record)
                    summary["iterations"].append(
                        {"iteration": iteration, "status": record["status"]}
                    )
                    reporter.emit(
                        "dry-run",
                        "global proposal validated; physical execution disabled",
                        iteration=iteration,
                        level="DONE",
                    )
                    continue

                reporter.start_phase(
                    "execution",
                    (
                        "sending one validated physical rollout; pre-run and post-run Home remain mandatory"
                        + (
                            "; Camera A/B RGB-D video recording is active"
                            if options.record_rollouts
                            else ""
                        )
                    ),
                    iteration=iteration,
                )
                checkpoint_depth_ranges = evaluation_depth_ranges(
                    (saved, saved_path)
                )
                checkpoint_before_images = evaluation_perception_image_paths(
                    saved,
                    saved_path,
                    stage="BEFORE",
                    depth_ranges=checkpoint_depth_ranges,
                )
                record["hold_checkpoint_before_images"] = [
                    str(path) for path in checkpoint_before_images
                ]
                perception_position_ready = False
                rollout_recorder: DualRealSenseRolloutRecorder | None = None
                recording_thread: threading.Thread | None = None
                recording_result: dict[str, Any] = {}
                recording_errors: list[str] = []
                recording_dir = (
                    iteration_dir
                    / (
                        "rollout_recording_"
                        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    )
                )
                if options.record_rollouts:
                    rollout_recorder = DualRealSenseRolloutRecorder(
                        perception_config,
                        recording_dir,
                        record_bag=options.recording_native,
                        record_depth_video=True,
                        record_composite=True,
                        codec=options.recording_codec,
                        warmup_frames=options.recording_warmup_frames,
                    )
                    try:
                        rollout_recorder.start()
                    except BaseException as exc:
                        raise AutoExplorationError(
                            "rollout recording failed before physical execution; "
                            "no robot command was sent: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc

                    def _record_global_rollout() -> None:
                        try:
                            recording_result["manifest"] = rollout_recorder.record()  # type: ignore[union-attr]
                        except BaseException as exc:
                            recording_errors.append(f"{type(exc).__name__}: {exc}")

                    recording_thread = threading.Thread(
                        target=_record_global_rollout,
                        daemon=True,
                        name=f"global-rollout-recorder-iteration-{iteration}",
                    )
                    recording_thread.start()
                    time.sleep(0.25)
                    if recording_errors:
                        rollout_recorder.request_stop(
                            "recording_failed_before_execution"
                        )
                        recording_thread.join(timeout=5.0)
                        raise AutoExplorationError(
                            "rollout recording failed before physical execution; "
                            "no robot command was sent: "
                            f"{recording_errors[-1]}"
                        )
                    record["rollout_recording"] = {
                        "status": "recording",
                        "directory": str(recording_dir),
                    }

                def _capture_lift_checkpoint(
                    checkpoint_number: int,
                ) -> dict[str, Any]:
                    """Capture one lift experiment checkpoint; never call Claude here."""

                    reporter.emit(
                        "lift-checkpoint",
                        (
                            f"robot reached lift checkpoint {checkpoint_number}/3; "
                            "capturing fresh Camera A/B RGB-D"
                        ),
                        iteration=iteration,
                        level="START",
                    )
                    try:
                        if rollout_recorder is None:
                            raise AutoExplorationError(
                                "lift checkpoint requires active Camera A/B rollout recording"
                            )
                        reporter.emit(
                            "lift-checkpoint",
                            (
                                "holding still before fresh RGB-D capture; "
                                f"settle={options.hold_checkpoint_settle_s:.2f}s"
                            ),
                            iteration=iteration,
                            level="WAIT",
                        )
                        sleep(options.hold_checkpoint_settle_s)
                        fresh_after_ns = time.monotonic_ns()
                        snapshot = rollout_recorder.wait_for_latest_rgbd(
                            after_monotonic_ns=fresh_after_ns,
                            timeout_s=3.0,
                            labels=("A", "B"),
                        )
                        stage = f"LIFT_CHECKPOINT_{int(checkpoint_number):02d}"
                        checkpoint_dir = (
                            iteration_dir / "lift_checkpoints" / stage
                        )
                        checkpoint_images, snapshot_manifest = _save_hold_checkpoint_snapshot(
                            snapshot,
                            checkpoint_dir,
                            depth_ranges=checkpoint_depth_ranges,
                            stage=stage,
                        )
                        checkpoint_record = {
                            "checkpoint_number": int(checkpoint_number),
                            "status": "CAPTURED",
                            "stage": stage,
                            "images": [str(path) for path in checkpoint_images],
                            "snapshot_manifest": snapshot_manifest,
                        }
                        record.setdefault("lift_checkpoints", []).append(
                            checkpoint_record
                        )
                        _write_json(
                            checkpoint_dir / "checkpoint.json",
                            checkpoint_record,
                        )
                        reporter.emit(
                            "lift-checkpoint",
                            (
                                f"checkpoint {checkpoint_number}/3 captured; "
                                "continuing the lift experiment"
                            ),
                            iteration=iteration,
                            level="DONE",
                            payload=checkpoint_record,
                        )
                        return {
                            "status": "CAPTURED",
                            "checkpoint_number": int(checkpoint_number),
                            "stage": stage,
                            "images": [str(path) for path in checkpoint_images],
                            "snapshot_manifest": snapshot_manifest,
                            "continue_transport": False,
                            "runtime_decision": "REVERSE_RELEASE",
                        }
                    except BaseException as exc:
                        failure = {
                            "status": "CAPTURE_FAILED",
                            "checkpoint_number": int(checkpoint_number),
                            "continue_transport": False,
                            "runtime_decision": "REVERSE_RELEASE",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        reporter.emit(
                            "lift-checkpoint",
                            (
                                f"checkpoint {checkpoint_number}/3 capture failed; "
                                f"reverse release will still be attempted: {failure['error']}"
                            ),
                            iteration=iteration,
                            level="WARNING",
                            payload=failure,
                        )
                        return failure

                execution: dict[str, Any] | None = None
                try:
                    checkpoint_executor = getattr(
                        session, "run_checkpointed_experiment", None
                    )
                    if (
                        proposal.requires_lift_checkpoint
                        and checkpoint_plan is not None
                        and callable(checkpoint_executor)
                    ):
                        lift_checkpoint_experiment_used = True
                        summary["lift_checkpoint"]["used"] = True
                        summary["lift_checkpoint"]["remaining_experiments"] = 0
                        # Persist this transition immediately.  If the robot or
                        # process stops during the physical probe, a resumed run
                        # must still know that the one-shot experiment was spent.
                        _write_json(output / "summary.json", summary)
                        record["lift_checkpoint_experiment_started"] = True
                        execution = checkpoint_executor(
                            source_path.name,
                            checkpoint_action_index=(
                                checkpoint_plan.checkpoint_action_indices[-1]
                            ),
                            checkpoint_action_indices=(
                                checkpoint_plan.checkpoint_action_indices
                            ),
                            abort_actions=checkpoint_plan.return_actions,
                            checkpoint_callback=_capture_lift_checkpoint,
                            real=True,
                            confirmed=True,
                            notes=(
                                f"Claude global CLI iteration {iteration}; "
                                "lift-checkpoint experiment with reverse release."
                            ),
                        )
                    else:
                        # Claude judged that a checkpoint experiment was not
                        # necessary, so execute the original validated proposal
                        # directly without any online hold/checkpoint callback.
                        execution = session.run_experiment(
                            source_path.name,
                            real=True,
                            confirmed=True,
                            notes=f"Claude global CLI iteration {iteration}.",
                        )
                finally:
                    if rollout_recorder is not None:
                        rollout_recorder.request_stop(
                            "rollout_and_return_home_completed"
                        )
                    if recording_thread is not None:
                        recording_thread.join(timeout=300.0)
                        if recording_thread.is_alive():
                            recording_errors.append(
                                "recording thread did not stop within 300 seconds"
                            )
                            rollout_recorder.close()  # type: ignore[union-attr]
                            recording_thread.join(timeout=3.0)
                    if rollout_recorder is not None:
                        manifest = recording_result.get("manifest")
                        manifest_path = recording_dir / "recording_manifest.json"
                        if manifest is None and manifest_path.is_file():
                            manifest = json.loads(
                                manifest_path.read_text(encoding="utf-8")
                            )
                        record["rollout_recording"] = {
                            "status": "failed" if recording_errors else "completed",
                            "directory": str(recording_dir),
                            "manifest": manifest,
                            "errors": list(recording_errors),
                        }
                        _write_json(
                            iteration_dir / "rollout_recording.json",
                            record["rollout_recording"],
                        )
                        append_recording_to_cumulative_video(
                            iteration,
                            record,
                            recording_dir,
                            execution,
                        )
                        _write_json(
                            iteration_dir / "rollout_recording.json",
                            record["rollout_recording"],
                        )
                if execution is None:
                    raise AutoExplorationError(
                        "physical rollout returned no result"
                    )
                record["execution"] = execution
                if isinstance(execution.get("checkpoint"), dict):
                    record["hold_checkpoint"] = {
                        **(
                            record.get("hold_checkpoint")
                            if isinstance(record.get("hold_checkpoint"), dict)
                            else {}
                        ),
                        **execution["checkpoint"],
                    }
                    _write_json(
                        iteration_dir / "hold_checkpoint.json",
                        record["hold_checkpoint"],
                    )
                if record.get("lift_checkpoints"):
                    record["artifacts"]["lift_checkpoints"] = str(
                        iteration_dir / "lift_checkpoints.json"
                    )
                    _write_json(
                        iteration_dir / "lift_checkpoints.json",
                        record["lift_checkpoints"],
                    )
                record["mandatory_return_home"] = session.last_return_home_outcome
                _write_json(iteration_dir / "execution.json", execution)
                _write_json(
                    iteration_dir / "mandatory_return_home.json",
                    session.last_return_home_outcome,
                )
                reporter.finish_phase(
                    f"execution_completed={bool(execution.get('execution_completed'))}",
                    success=bool(execution.get("execution_completed")),
                    payload=execution,
                )
                if not execution.get("execution_completed"):
                    raise AutoExplorationError(
                        "physical rollout did not complete: "
                        f"{execution.get('robot_errors', [])}"
                    )
                _record_stage(
                    output,
                    iteration_dir,
                    iteration,
                    record,
                    "EXECUTION_COMPLETED",
                )

                prepare_real_perception_position(
                    iteration,
                    record,
                    "post_action_perception_robot_positioning",
                )
                reporter.start_phase(
                    "after-capture",
                    "capturing and processing complete post-action Camera A/B state",
                    iteration=iteration,
                )
                after_frames = capture(perception_config)
                after_perception = session.locate_cloth_center(
                    perception_config, frames=after_frames
                )
                after_saved, after_saved_path = _load_latest_perception(session)
                if after_saved is None or after_saved_path is None:
                    raise AutoExplorationError(
                        "post-action perception completed without a saved result"
                    )
                after_images = global_perception_image_paths(
                    after_saved, after_saved_path
                )
                # Planning deliberately uses the richer A/B geometry bundle.  The
                # post-action judge receives a compact, explicitly labelled pair:
                # raw RGB plus metric depth for each camera, with one fixed depth
                # color scale shared by before and after.
                evaluation_depth_scale = evaluation_depth_ranges(
                    (saved, saved_path),
                    (after_saved, after_saved_path),
                )
                evaluation_before_images = evaluation_perception_image_paths(
                    saved,
                    saved_path,
                    stage="BEFORE",
                    depth_ranges=evaluation_depth_scale,
                )
                evaluation_after_images = evaluation_perception_image_paths(
                    after_saved,
                    after_saved_path,
                    stage="AFTER",
                    depth_ranges=evaluation_depth_scale,
                )
                lift_checkpoint_images = [
                    Path(path)
                    for checkpoint in record.get("lift_checkpoints", [])
                    if isinstance(checkpoint, dict)
                    for path in checkpoint.get("images", [])
                    if Path(path).is_file()
                ]
                evaluation_after_images.extend(lift_checkpoint_images)
                record["lift_checkpoint_evaluation_images"] = [
                    str(path) for path in lift_checkpoint_images
                ]
                record["after_perception"] = after_perception
                record["after_images"] = [str(path) for path in after_images]
                record["evaluation_before_images"] = [
                    str(path) for path in evaluation_before_images
                ]
                record["evaluation_after_images"] = [
                    str(path) for path in evaluation_after_images
                ]
                reporter.finish_phase(
                    f"saved {len(evaluation_after_images)} labelled RGB/depth after-state image(s)"
                )
                _record_stage(
                    output,
                    iteration_dir,
                    iteration,
                    record,
                    "AFTER_PERCEPTION_COMPLETED",
                )

                reporter.start_phase(
                    "evaluation",
                    "Claude comparing complete before/after state and choosing keep/change",
                    iteration=iteration,
                )
                evaluation = evaluate_with_timeout_retry(
                    iteration,
                    iteration_dir,
                    record,
                    lambda: global_evaluator.evaluate(
                        evaluation_before_images,
                        evaluation_after_images,
                        proposal=proposal,
                        objective=options.objective,
                        run_dir=session.run_dir,
                        skill_guidance=run_skill_prompt(),
                        hold_checkpoint=(
                            record.get("hold_checkpoint")
                            if isinstance(record.get("hold_checkpoint"), dict)
                            else None
                        ),
                        rollout_recording_dir=(
                            recording_dir
                            if options.record_rollouts
                            and record.get("rollout_recording", {}).get("status") == "completed"
                            else None
                        ),
                    ),
                )
                evaluation_dict = evaluation.as_dict()
                record["evaluation"] = evaluation_dict
                _write_json(iteration_dir / "evaluation.json", evaluation_dict)
                _record_stage(
                    output,
                    iteration_dir,
                    iteration,
                    record,
                    "EVALUATION_COMPLETED",
                )
                # Keep compatibility with lightweight/fake evaluators used by
                # integrations and older callers that predate skill proposals.
                skill_update = getattr(evaluation, "skill_update", None)
                skill_review = run_skill_ledger.stage_skill_update(
                    skill_update,
                    iteration=iteration,
                    source="evaluation",
                )
                if skill_review is not None:
                    record["skill_review"] = skill_review.as_dict()
                    _write_json(iteration_dir / "skill_review.json", skill_review.as_dict())
                    if global_client is not None:
                        global_client.skill_names = tuple(
                            skill.name for skill in skill_store.approved()
                        )
                prune_completed_recording(
                    iteration,
                    record,
                    recording_dir if options.record_rollouts else None,
                )
                _write_json(
                    iteration_dir / "rollout_recording.json",
                    record.get("rollout_recording", {}),
                )
                evidence = build_evidence_record(
                    record,
                    iteration=iteration,
                    run_dir=session.run_dir,
                )
                evidence_paths = persist_evidence_record(
                    session.run_dir,
                    evidence,
                    iteration_dir=iteration_dir,
                )
                record["evidence"] = evidence
                record["evidence_artifacts"] = evidence_paths
                experience = {
                    "created_at": _now(),
                    "iteration": iteration,
                    "objective": options.objective,
                    "before_images": [str(path) for path in before_images],
                    "after_images": [
                        str(path) for path in after_images
                    ] + [
                        str(path) for path in lift_checkpoint_images
                    ],
                    "selected_grasp": proposal.selected_grasp,
                    "grounding": grounding,
                    "proposal": proposal.as_dict(),
                    "evaluation": evaluation_dict,
                    "skill_review": skill_review.as_dict() if skill_review is not None else None,
                    "evidence": evidence,
                }
                _append_jsonl(global_experience_path, experience)
                run_skill_ledger.append_experience(experience)
                global_experiences.append(experience)
                record["global_experience"] = experience
                record["status"] = "COMPLETED"
                record["completed_at"] = _now()
                consecutive_recoverable_failures = 0
                _iteration_checkpoint(output, iteration_dir, iteration, record)
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
                        f"keep={list(evaluation.next_experiment.keep)} "
                        f"change={list(evaluation.next_experiment.change)}"
                    ),
                    payload=evaluation_dict,
                )
                if evaluation.stop:
                    summary["stop_reason"] = evaluation.reason
                    reporter.emit(
                        "stop",
                        f"evaluator requested stop: {evaluation.reason}",
                        iteration=iteration,
                    )
                    break
                continue

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
                    f"axis-first Molmo pass, then {len(options.keypoint_specs)} "
                    f"semantic-anchor query/queries on Camera {','.join(options.keypoint_cameras)}; "
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
            before_images = _planning_images(
                saved, saved_path, manifest, session.project_root
            )
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
                        session.robot_config.validate_workspace_pose(
                            args["x"],
                            args["y"],
                            args["z"],
                            relative_yaw_deg=args["yaw"],
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
                            "home_tcp_yaw_deg": session.robot_config.init_pose_mm_deg[5],
                            "yaw_semantics": "move yaw is relative to calibrated Home TCP yaw; 0 preserves gripper orientation",
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
                (
                    "sending one validated physical rollout; mandatory Home remains active"
                    + ("; Camera A/B RGB-D video recording is active" if options.record_rollouts else "")
                ),
                iteration=iteration,
            )
            perception_position_ready = False
            rollout_recorder: DualRealSenseRolloutRecorder | None = None
            recording_thread: threading.Thread | None = None
            recording_result: dict[str, Any] = {}
            recording_errors: list[str] = []
            recording_dir = (
                iteration_dir
                / ("rollout_recording_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
            )
            if options.record_rollouts:
                rollout_recorder = DualRealSenseRolloutRecorder(
                    perception_config,
                    recording_dir,
                    record_bag=options.recording_native,
                    record_depth_video=True,
                    record_composite=True,
                    codec=options.recording_codec,
                    warmup_frames=options.recording_warmup_frames,
                )
                try:
                    rollout_recorder.start()
                except BaseException as exc:
                    raise AutoExplorationError(
                        "rollout recording failed before physical execution; no robot command was sent: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

                def _record_rollout() -> None:
                    try:
                        recording_result["manifest"] = rollout_recorder.record()  # type: ignore[union-attr]
                    except BaseException as exc:
                        recording_errors.append(f"{type(exc).__name__}: {exc}")

                recording_thread = threading.Thread(
                    target=_record_rollout,
                    daemon=True,
                    name=f"rollout-recorder-iteration-{iteration}",
                )
                recording_thread.start()
                time.sleep(0.25)
                if recording_errors:
                    rollout_recorder.request_stop("recording_failed_before_execution")
                    recording_thread.join(timeout=5.0)
                    raise AutoExplorationError(
                        "rollout recording failed before physical execution; no robot command was sent: "
                        f"{recording_errors[-1]}"
                    )
                record["rollout_recording"] = {
                    "status": "recording",
                    "directory": str(recording_dir),
                }
            execution: dict[str, Any] | None = None
            try:
                execution = session.run_experiment(
                    source_path.name,
                    real=True,
                    confirmed=True,
                    notes=f"Molmo keypoint CLI iteration {iteration}.",
                )
            finally:
                home_outcome = session.last_return_home_outcome
                if rollout_recorder is not None:
                    rollout_recorder.request_stop("rollout_and_return_home_completed")
                if recording_thread is not None:
                    recording_thread.join(timeout=300.0)
                    if recording_thread.is_alive():
                        recording_errors.append("recording thread did not stop within 300 seconds")
                        rollout_recorder.close()  # type: ignore[union-attr]
                        recording_thread.join(timeout=3.0)
                if rollout_recorder is not None:
                    manifest = recording_result.get("manifest")
                    manifest_path = recording_dir / "recording_manifest.json"
                    if manifest is None and manifest_path.is_file():
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    record["rollout_recording"] = {
                        "status": "failed" if recording_errors else "completed",
                        "directory": str(recording_dir),
                        "manifest": manifest,
                        "errors": list(recording_errors),
                    }
                    _write_json(iteration_dir / "rollout_recording.json", record["rollout_recording"])
                    append_recording_to_cumulative_video(
                        iteration,
                        record,
                        recording_dir,
                        execution,
                    )
                    _write_json(iteration_dir / "rollout_recording.json", record["rollout_recording"])
            if execution is None:
                raise AutoExplorationError("physical rollout returned no result")
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
            evaluation = evaluate_with_timeout_retry(
                iteration,
                iteration_dir,
                record,
                lambda: client.evaluate(
                    before_images=before_images,
                    after_images=after_images,
                    run_dir=session.run_dir,
                    semantic_state=semantic_state,
                    strategy=strategy,
                    candidate=selected_candidate,
                    action_result=action_result,
                ),
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
            _record_stage(
                output,
                iteration_dir,
                iteration,
                record,
                "EVALUATION_COMPLETED",
            )
            prune_completed_recording(
                iteration,
                record,
                recording_dir if options.record_rollouts else None,
            )
            _write_json(
                iteration_dir / "rollout_recording.json",
                record.get("rollout_recording", {}),
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
            consecutive_recoverable_failures = 0
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
            if record.get("execution") is not None and "evidence_artifacts" not in record:
                evidence = build_evidence_record(
                    record,
                    iteration=iteration,
                    run_dir=session.run_dir,
                )
                record["evidence"] = evidence
                record["evidence_artifacts"] = persist_evidence_record(
                    session.run_dir,
                    evidence,
                    iteration_dir=iteration_dir,
                )
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
            # A physical rollout may fail before evaluator completion.  Persist
            # its partial evidence immediately so an unattended supervisor and
            # the next Claude iteration can distinguish an execution failure
            # from a pre-execution planning failure.
            if record.get("execution") is not None and "evidence_artifacts" not in record:
                try:
                    evidence = build_evidence_record(
                        record,
                        iteration=iteration,
                        run_dir=session.run_dir,
                    )
                    record["evidence"] = evidence
                    record["evidence_artifacts"] = persist_evidence_record(
                        session.run_dir,
                        evidence,
                        iteration_dir=iteration_dir,
                    )
                except BaseException as evidence_exc:
                    reporter.emit(
                        "evidence",
                        f"failed to persist partial operation evidence: {evidence_exc}",
                        iteration=iteration,
                        level="ERROR",
                    )
            if (
                options.continue_on_recoverable_errors
                and _is_recoverable_loop_error(exc, record)
                and (
                    options.max_consecutive_recoverable_failures == 0
                    or consecutive_recoverable_failures
                    < options.max_consecutive_recoverable_failures
                )
            ):
                consecutive_recoverable_failures += 1
                feedback_payload = record.get("error_feedback_to_claude")
                if not isinstance(feedback_payload, dict):
                    feedback_payload = {
                        "attempt": None,
                        "phase": record.get("last_completed_stage") or "startup",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "rejected_proposal": record.get("proposal"),
                        "rejected_actions": (
                            (record.get("proposal") or {}).get("actions")
                            if isinstance(record.get("proposal"), dict)
                            else None
                        ),
                        "feedback_target": "next fresh Claude planning iteration",
                        "required_response": (
                            "use this exact failure context; do not repeat the rejected "
                            "operation; pass perception, grounding, preflight, workspace, "
                            "and controller IK validation before motion"
                        ),
                        "physical_command_sent": False,
                    }
                record["error_feedback_to_claude"] = feedback_payload
                pending_recoverable_error = feedback_payload
                error_experience = {
                    "created_at": _now(),
                    "type": "recoverable_preexecution_error",
                    "iteration": iteration,
                    "error_feedback": feedback_payload,
                    "recovery_policy": "fresh perception and Claude planning iteration",
                    "physical_command_sent": False,
                }
                record["error_feedback_experience"] = error_experience
                if options.planning_policy == "claude_global":
                    _append_jsonl(global_experience_path, error_experience)
                    global_experiences.append(error_experience)
                run_skill_ledger.append_experience(error_experience)
                record["status"] = "RECOVERABLE_ERROR"
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc()
                record["recovery"] = {
                    "enabled": True,
                    "consecutive_failure": consecutive_recoverable_failures,
                    "max_consecutive_failures": (
                        "unlimited"
                        if options.max_consecutive_recoverable_failures == 0
                        else options.max_consecutive_recoverable_failures
                    ),
                    "next_step": "fresh perception and new planning iteration",
                }
                record["completed_at"] = _now()
                _iteration_checkpoint(output, iteration_dir, iteration, record)
                summary["iterations"].append(
                    {
                        "iteration": iteration,
                        "status": record["status"],
                        "error": record["error"],
                    }
                )
                reporter.emit(
                    "recovery",
                    (
                        f"recoverable pre-execution failure recorded; continuing with a fresh "
                        f"iteration ({consecutive_recoverable_failures}/"
                        f"{'∞' if options.max_consecutive_recoverable_failures == 0 else options.max_consecutive_recoverable_failures})"
                    ),
                    iteration=iteration,
                    level="WARNING",
                    payload=record["recovery"],
                )
                if options.recovery_backoff_s > 0:
                    time.sleep(options.recovery_backoff_s)
                continue
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

    try:
        run_skill_synthesis = run_skill_ledger.finalize(skill_store)
        _write_json(output / "run_skill_synthesis.json", run_skill_synthesis)
        summary["run_skill_synthesis"] = str(output / "run_skill_synthesis.json")
        reporter.emit(
            "run-skill-finalize",
            (
                f"synthesized {run_skill_synthesis['skill_group_count']} run-local skill group(s); "
                "global skill persistence was deferred until run completion"
            ),
            payload=run_skill_synthesis,
        )
    except Exception as exc:
        reporter.emit(
            "run-skill-finalize",
            f"run-local skill synthesis failed: {type(exc).__name__}: {exc}",
            level="ERROR",
        )
    if summary.get("status") == "RUNNING":
        summary["status"] = "COMPLETED" if exit_code == 0 else "FAILED"
    summary["completed_at"] = _now()
    summary["iteration_count"] = len(summary["iterations"])
    if combined_video_path.is_file():
        summary["combined_rollout_video"] = str(combined_video_path)
    else:
        summary["combined_rollout_video"] = None
    _write_json(output / "summary.json", summary)
    reporter.stop_heartbeat()
    reporter.emit(
        "shutdown",
        f"status={summary['status']} iterations={summary['iteration_count']}",
        payload={
            "summary": str(output / "summary.json"),
            "combined_rollout_video": summary["combined_rollout_video"],
            "combined_video_speed": options.combined_video_speed,
        },
        level="INFO" if exit_code == 0 else "ERROR",
    )
    if summary["combined_rollout_video"]:
        print(
            f"Combined rollout video: {summary['combined_rollout_video']}",
            file=stream,
            flush=True,
        )
    else:
        print(
            "Combined rollout video: unavailable (no completed rollout recording)",
            file=stream,
            flush=True,
        )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path("config/robot.example.json"),
        help=(
            "robot configuration JSON (default: config/robot.example.json; "
            "uses absolute camera depth without live tabletop Z flooring)"
        ),
    )
    parser.add_argument(
        "--perception-config",
        type=Path,
        default=Path("config/perception.free_exploration.json"),
    )
    parser.add_argument("--camera-a-exposure", type=float)
    parser.add_argument("--camera-b-exposure", type=float)
    parser.add_argument("--camera-a-white-balance", type=float)
    parser.add_argument("--camera-b-white-balance", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument(
        "--planning-policy",
        choices=("claude_global", "semantic_local"),
        default="claude_global",
        help=(
            "claude_global lets Claude choose any Camera A pixel while using Camera B as "
            "an observation-only second view; "
            "semantic_local enables the legacy Molmo Sxxx/local Rxxx pipeline"
        ),
    )
    parser.add_argument(
        "--global-molmo-annotations",
        action="store_true",
        help=(
            "run the axis-first Molmo semantic annotation pass before Claude-global "
            "planning and include its overlays as visual evidence"
        ),
    )
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
        default=19_000,
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
    parser.add_argument("--claude-timeout-s", type=int, default=900)
    parser.add_argument("--claude-grounding-timeout-s", type=int, default=120)
    parser.add_argument(
        "--hold-checkpoint-timeout-s",
        type=int,
        default=600,
        help=(
            "maximum seconds to hold fabric while Claude classifies fresh Camera A/B "
            "RGB-D; timeout always descends and releases (default: 600; max: 900)"
        ),
    )
    parser.add_argument(
        "--hold-checkpoint-settle-s",
        type=float,
        default=0.75,
        help=(
            "seconds to hold still before requesting post-lift Camera A/B RGB-D "
            "for the mandatory checkpoint (default: 0.75)"
        ),
    )
    recording_group = parser.add_mutually_exclusive_group()
    recording_group.add_argument(
        "--record-rollouts",
        dest="record_rollouts",
        action="store_true",
        default=True,
        help="record Camera A/B RGB, depth, and composite rollout video for Claude evaluation (default)",
    )
    recording_group.add_argument(
        "--no-record-rollouts",
        dest="record_rollouts",
        action="store_false",
        help="disable rollout video recording; temporal acquisition/transport/laydown evidence becomes UNKNOWN",
    )
    parser.add_argument(
        "--recording-no-native",
        action="store_true",
        help="disable native RealSense recordings while retaining MP4 RGB/depth/composite videos",
    )
    parser.add_argument("--recording-codec", default="mp4v")
    parser.add_argument("--recording-warmup-frames", type=int)
    parser.add_argument(
        "--max-replans",
        type=int,
        default=1,
        help="hard validation correction budget; at most one correction is permitted",
    )
    parser.add_argument(
        "--continue-on-recoverable-errors",
        action="store_true",
        help=(
            "after a pre-execution Claude/planning failure, save the failed iteration, "
            "capture a fresh scene, and continue until the consecutive-failure cap"
        ),
    )
    parser.add_argument(
        "--max-consecutive-recoverable-failures",
        type=int,
        default=0,
        help=(
            "stop continuous recovery after this many consecutive pre-execution failures; "
            "0 means unlimited recovery"
        ),
    )
    parser.add_argument(
        "--recovery-backoff-s",
        type=float,
        default=2.0,
        help="seconds to wait before starting the next recovered iteration",
    )
    parser.add_argument(
        "--max-evaluation-retries",
        type=int,
        default=0,
        help=(
            "Claude evaluation timeout retry count using the same saved before/after "
            "evidence; 0 means retry until success or operator interrupt"
        ),
    )
    parser.add_argument(
        "--evaluation-retry-backoff-s",
        type=float,
        default=2.0,
        help=(
            "seconds between Claude evaluation timeout retries; retries never resend "
            "robot or camera commands"
        ),
    )
    parser.add_argument(
        "--objective",
        default=DEFAULT_EXPLORATION_OBJECTIVE,
    )
    comparison_group = parser.add_mutually_exclusive_group()
    comparison_group.add_argument(
        "--rgb-only-comparison",
        dest="rgb_only_comparison",
        action="store_true",
        help=(
            "after each accepted Claude-global decision, run one non-executing RGB-only "
            "comparison thought and save its raw output and provisional grasp trajectory"
        ),
    )
    comparison_group.add_argument(
        "--no-rgb-only-comparison",
        dest="rgb_only_comparison",
        action="store_false",
    )
    parser.set_defaults(rgb_only_comparison=False)
    parser.add_argument(
        "--combined-video-speed",
        type=float,
        default=32.0,
        help="playback speed multiplier for the cumulative combined rollout video (default: 32x)",
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
    perception_config = _override_camera_controls(
        PerceptionConfig.load(root, perception_path.resolve()),
        camera_a_exposure=args.camera_a_exposure,
        camera_b_exposure=args.camera_b_exposure,
        camera_a_white_balance=args.camera_a_white_balance,
        camera_b_white_balance=args.camera_b_white_balance,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else session.results / "molmo_keypoint_cli" / stamp
    )
    max_iterations = None if args.max_iterations == 0 else args.max_iterations
    options = KeypointCliOptions(
        planning_policy=args.planning_policy,
        global_molmo_annotations=args.global_molmo_annotations,
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
        hold_checkpoint_timeout_s=args.hold_checkpoint_timeout_s,
        hold_checkpoint_settle_s=args.hold_checkpoint_settle_s,
        max_replans=args.max_replans,
        continue_on_recoverable_errors=args.continue_on_recoverable_errors,
        max_consecutive_recoverable_failures=args.max_consecutive_recoverable_failures,
        recovery_backoff_s=args.recovery_backoff_s,
        max_evaluation_retries=args.max_evaluation_retries,
        evaluation_retry_backoff_s=args.evaluation_retry_backoff_s,
        record_rollouts=args.record_rollouts,
        recording_native=not args.recording_no_native,
        recording_codec=args.recording_codec,
        recording_warmup_frames=args.recording_warmup_frames,
        objective=args.objective,
        rgb_only_comparison=args.rgb_only_comparison,
        combined_video_speed=args.combined_video_speed,
    )
    return run_keypoint_cli_loop(
        session,
        perception_config,
        output_dir,
        options,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
