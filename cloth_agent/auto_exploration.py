"""Standalone automatic Claude/RealSense/xArm exploration loop.

The existing manual free-exploration console remains unchanged.  This module
adds an opt-in state machine for one cautious real rollout at a time:

``Viser RGB-D preview -> RGB-D perception -> Claude plan -> preflight/IK -> execute ->
Viser RGB-D preview -> before/after Claude evaluation -> next iteration``.

Pre-execution validation failures are returned to Claude for a bounded
replanning attempt. The loop never retries after physical motion has started or
interrupts an in-progress physical command.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import SafetyError
from .experiment import ExperimentValidationError
from .free_exploration import (
    ClaudeExplorationClient,
    ExplorationPlanningError,
    ClaudeExplorationResult,
    ExplorationProposal,
    _json_from_claude_text,
    _proposal_markdown,
    _voxel_balance_cloud,
    exploration_prompt,
    exploration_source,
    perception_image_paths,
    validate_exploration_payload,
)
from .perception import (
    CameraSpec,
    PerceptionConfig,
    RGBDFrame,
    RealSenseRGBD,
    capture_two_view_rgbd,
    load_extrinsics,
)
from .robot_api import RobotExecutionError, validate_controller_trajectory
from .session import AgentSession
from .viewer import _frame_point_cloud, _load_latest_perception


AUTO_EVALUATION_REQUIRED_FIELDS = frozenset(
    {"useful", "confidence", "observed_change", "next_objective", "stop", "reason"}
)
AUTO_EVALUATION_OPTIONAL_FIELDS = frozenset({"caveats"})
AUTO_EVALUATION_FIELDS = AUTO_EVALUATION_REQUIRED_FIELDS | AUTO_EVALUATION_OPTIONAL_FIELDS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class AutoExplorationError(RuntimeError):
    """Raised when an automatic-loop contract or runtime phase fails."""


@dataclass(frozen=True)
class ExplorationEvaluation:
    useful: bool
    confidence: float
    observed_change: str
    next_objective: str
    stop: bool
    reason: str
    caveats: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "useful": self.useful,
            "confidence": self.confidence,
            "observed_change": self.observed_change,
            "next_objective": self.next_objective,
            "stop": self.stop,
            "reason": self.reason,
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True)
class ClaudeEvaluationResult:
    """Raw Claude evaluator call plus its validated structured judgement."""

    prompt: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    created_at: str
    evaluation: ExplorationEvaluation

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "command": list(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "created_at": self.created_at,
            "evaluation": self.evaluation.as_dict(),
        }


def _json_default(value: Any) -> Any:
    """Serialize paths, NumPy values, and dataclasses in run records."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _run_relative(path: Path, run_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def validate_evaluation_payload(payload: Any) -> ExplorationEvaluation:
    """Validate Claude's before/after judgement before another iteration."""

    if not isinstance(payload, dict):
        raise AutoExplorationError("Claude evaluation must be a JSON object")
    missing = AUTO_EVALUATION_REQUIRED_FIELDS.difference(payload)
    unknown = set(payload).difference(AUTO_EVALUATION_FIELDS)
    if missing:
        raise AutoExplorationError(f"evaluation is missing fields: {sorted(missing)}")
    if unknown:
        raise AutoExplorationError(f"evaluation has unknown fields: {sorted(unknown)}")
    for name in ("useful", "stop"):
        if not isinstance(payload[name], bool):
            raise AutoExplorationError(f"evaluation field {name} must be boolean")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise AutoExplorationError("evaluation confidence must be numeric")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise AutoExplorationError("evaluation confidence must be between 0 and 1")
    values: dict[str, str] = {}
    for name in ("observed_change", "next_objective", "reason"):
        if not isinstance(payload[name], str) or not payload[name].strip():
            raise AutoExplorationError(f"evaluation field {name} must be non-empty")
        values[name] = payload[name].strip()
    raw_caveats = payload.get("caveats", [])
    if isinstance(raw_caveats, str):
        raw_caveats = [raw_caveats]
    if not isinstance(raw_caveats, list) or len(raw_caveats) > 10:
        raise AutoExplorationError(
            "evaluation caveats must be a string or a list of at most 10 strings"
        )
    caveats: list[str] = []
    for caveat in raw_caveats:
        if not isinstance(caveat, str) or not caveat.strip():
            raise AutoExplorationError(
                "every evaluation caveat must be a non-empty string"
            )
        caveats.append(caveat.strip())
    return ExplorationEvaluation(
        useful=payload["useful"],
        confidence=confidence,
        observed_change=values["observed_change"],
        next_objective=values["next_objective"],
        stop=payload["stop"],
        reason=values["reason"],
        caveats=tuple(caveats),
    )


class ClaudeAutoClient:
    """Claude adapter that plans actions and judges before/after images."""

    def __init__(self, binary: str = "claude", timeout_s: int = 300):
        self.binary = binary
        self.timeout_s = timeout_s
        self.planner = ClaudeExplorationClient(binary=binary, timeout_s=timeout_s)
        self.last_plan_result: ClaudeExplorationResult | None = None
        self.last_evaluation_result: ClaudeEvaluationResult | None = None

    @staticmethod
    def _save_evaluation_log(root: Path, payload: dict[str, Any], *, failed: bool = False) -> None:
        log_dir = root / "results" / "claude_auto"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        name = f"{stamp}_evaluation{'_failed' if failed else ''}.json"
        _write_json(log_dir / name, payload)

    def plan(
        self,
        image_paths: list[Path],
        session: AgentSession,
        objective: str,
        feedback: str | None = None,
    ) -> ExplorationProposal:
        self.last_plan_result = None
        prompt_objective = objective
        if feedback:
            prompt_objective += (
                "\n\nThe previous candidate was rejected before physical execution. "
                "Use the failure report below to generate a materially different, "
                "more conservative proposal. Do not repeat the rejected pose or "
                "assume that local XYZ bounds imply IK reachability.\n"
                f"Failure report:\n{feedback}"
            )
        response = self.planner.invoke(
            image_paths,
            exploration_prompt(
                session.experiment_config,
                session.robot_config,
                objective=prompt_objective,
            ),
            session.run_dir,
        )
        self.last_plan_result = response
        return response.proposal

    def evaluate(
        self,
        before_images: list[Path],
        after_images: list[Path],
        *,
        proposal: ExplorationProposal,
        run_dir: Path,
    ) -> ExplorationEvaluation:
        self.last_evaluation_result = None
        root = run_dir.resolve()
        image_lines = ["Before images:"]
        image_lines.extend(f"- {path.resolve()}" for path in before_images)
        image_lines.append("After images:")
        image_lines.extend(f"- {path.resolve()}" for path in after_images)
        prompt = (
            "Compare the before and after garment images after one robot reveal action. "
            "Decide whether the action usefully exposed more garment surface, what visibly "
            "changed, and what the next objective should be. Stop if the garment is already "
            "sufficiently revealed, the action was unsafe/unclear, or another action is not "
            "justified. Do not invent pixel measurements. Return exactly one JSON object with "
            "useful (boolean), confidence (number 0..1), observed_change (string), "
            "next_objective (string), stop (boolean), reason (string), and optional "
            "caveats (list of strings; use an empty list when there are none).\n\n"
            f"Previous proposal strategy: {proposal.reveal_strategy}\n"
            f"Previous expected observation: {proposal.expected_observation}\n\n"
            + "\n".join(image_lines)
        )
        binary = self.planner.binary
        if Path(binary).name == binary:
            import shutil

            binary = shutil.which(binary)
        if binary is None:
            raise AutoExplorationError(f"Claude CLI not found: {self.planner.binary}")
        command = [
            binary,
            "--print",
            prompt,
            "--output-format",
            "json",
            "--permission-mode",
            "plan",
            "--allowedTools",
            "Read",
            "--tools",
            "Read",
            "--add-dir",
            str(root),
            "--safe-mode",
            "--system-prompt",
            (
                "You are a cautious visual evaluator for a robotics garment run. "
                "Read only the supplied images and return exactly the requested JSON. "
                "Do not edit files, execute commands, or control a robot."
            ),
        ]
        import subprocess

        try:
            completed = subprocess.run(
                command,
                cwd=root,
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._save_evaluation_log(
                root,
                {
                    "prompt": prompt,
                    "command": command,
                    "returncode": None,
                    "stdout": getattr(exc, "stdout", "") or "",
                    "stderr": getattr(exc, "stderr", "") or "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "created_at": _now(),
                },
                failed=True,
            )
            raise AutoExplorationError(f"Claude evaluation invocation failed: {exc}") from exc
        if completed.returncode != 0:
            self._save_evaluation_log(
                root,
                {
                    "prompt": prompt,
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "error": "non-zero Claude return code",
                    "created_at": _now(),
                },
                failed=True,
            )
            raise AutoExplorationError(
                f"Claude evaluation exited with {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        try:
            evaluation = validate_evaluation_payload(_json_from_claude_text(completed.stdout))
        except BaseException as exc:
            self._save_evaluation_log(
                root,
                {
                    "prompt": prompt,
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "error": f"{type(exc).__name__}: {exc}",
                    "created_at": _now(),
                },
                failed=True,
            )
            raise
        result = ClaudeEvaluationResult(
            prompt=prompt,
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            created_at=_now(),
            evaluation=evaluation,
        )
        self.last_evaluation_result = result
        self._save_evaluation_log(root, result.as_dict())
        return evaluation


def _depth_preview(depth_m: np.ndarray, *, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    """Convert metric depth into a browser-friendly grayscale RGB image."""

    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > min_depth_m) & (depth < max_depth_m)
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = np.clip(
        (max_depth_m - depth[valid]) / (max_depth_m - min_depth_m), 0.0, 1.0
    )
    image = np.rint(normalized * 255.0).astype(np.uint8)
    return np.repeat(image[..., None], 3, axis=2)


class CameraAWebMonitor:
    """Persistent CamA RGB-D preview rendered directly inside Viser.

    The monitor owns the CamA RealSense pipeline only while the automatic loop
    is idle. The loop stops it before synchronized A/B capture, eliminating the
    device contention caused by a second OpenCV process.
    """

    def __init__(
        self,
        project_root: Path,
        perception_config_path: Path,
        spec: CameraSpec,
        config: PerceptionConfig,
        on_close: Callable[[], None],
        on_frame: Callable[[RGBDFrame], None] | None = None,
    ):
        del project_root, perception_config_path
        self.spec = spec
        self.config = config
        self.on_close = on_close
        # Kept as a compatibility/debug label for callers that used the old
        # native-window monitor. No native window is created anymore.
        self.window_name = f"CamA live monitor ({spec.serial})"
        self.on_frame = on_frame or (lambda frame: None)
        self.X_base_camera: np.ndarray | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.camera: RealSenseRGBD | None = None
        self.error: BaseException | None = None
        self._lock = threading.Lock()
        self._camera_lifecycle_lock = threading.Lock()

    def _start_camera(self, camera: RealSenseRGBD) -> bool:
        """Start once unless a stop was requested before the worker acquired it."""

        with self._camera_lifecycle_lock:
            if self.stop_event.is_set():
                return False
            camera.start()
            return True

    def _stop_camera(self, camera: RealSenseRGBD) -> None:
        """Serialize all pipeline stops so RealSense never receives a duplicate."""

        with self._camera_lifecycle_lock:
            camera.stop()

    def start(self) -> None:
        with self._lock:
            if self.thread is not None and self.thread.is_alive():
                return
            self.stop_event.clear()
            self.error = None
            self.camera = RealSenseRGBD(
                self.spec, self.config.width, self.config.height, self.config.fps
            )
            self.thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="claude-auto-camera-a-web-monitor",
            )
            self.thread.start()

    def _run(self) -> None:
        camera = self.camera
        if camera is None:
            return
        try:
            if self.stop_event.is_set():
                return
            self.X_base_camera = load_extrinsics(self.spec.extrinsics_file)
            if not self._start_camera(camera):
                return
            for _ in range(min(self.config.warmup_frames, 5)):
                if self.stop_event.is_set():
                    return
                camera.read()
            while not self.stop_event.is_set():
                rgb, depth_m = camera.read()
                if camera.intrinsics is None:
                    raise AutoExplorationError("CamA intrinsics are unavailable")
                self.on_frame(
                    RGBDFrame(
                        label=self.spec.label,
                        serial=self.spec.serial,
                        rgb=rgb,
                        depth_m=depth_m,
                        intrinsics=camera.intrinsics.copy(),
                        X_base_camera=self.X_base_camera.copy(),
                    )
                )
                self.stop_event.wait(0.15)
        except BaseException as exc:
            if not self.stop_event.is_set():
                self.error = exc
                self.on_close()
        finally:
            self._stop_camera(camera)
            self.camera = None

    def stop(self) -> None:
        with self._lock:
            self.stop_event.set()
            # Release the RealSense pipeline before joining the reader. A
            # blocked wait_for_frames() must be interrupted when capture_two_view
            # is about to claim the same device.
            camera = self.camera
            if camera is not None:
                self._stop_camera(camera)
            if self.thread is not None and self.thread.is_alive():
                self.thread.join(timeout=3.0)
            self.thread = None
            self.camera = None


def _save_frame_images(
    frames: list[RGBDFrame],
    output_dir: Path,
) -> list[Path]:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    for index, frame in enumerate(frames):
        path = output_dir / f"camera_{index}_{frame.label}.png"
        Image.fromarray(frame.rgb.astype(np.uint8)).save(path)
        np.save(output_dir / f"camera_{index}_{frame.label}_depth_m.npy", frame.depth_m)
        paths.append(path)
    return paths


@dataclass
class _AutoState:
    running: bool = False
    stop_requested: bool = False
    iteration: int = 0
    objective: str = (
        "Reveal more previously occluded garment surface with one cautious action."
    )
    proposal: ExplorationProposal | None = None
    evaluation: ExplorationEvaluation | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


def run_auto_exploration_viewer(
    session: AgentSession,
    *,
    host: str = "127.0.0.1",
    port: int = 8082,
    max_iterations: int | None = None,
    settle_s: float = 2.0,
    enable_real: bool = False,
    perception_config_path: Path | None = None,
    claude_binary: str = "claude",
    claude_timeout_s: int = 300,
    max_replans: int = 2,
) -> int:
    """Run the continuous automatic real-agent loop with a Viser preview.

    ``max_iterations=None`` (the default) keeps iterating until Claude returns
    ``stop=true``, the operator requests a stop, or a hard validation/runtime
    failure occurs. A positive value provides an optional iteration cap.
    """

    if not enable_real:
        raise PermissionError(
            "automatic exploration is real-execution only; pass --enable-real explicitly"
        )
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise PermissionError("physical execution is allowed only on a loopback-only Viser server")
    if max_iterations == 0:
        max_iterations = None
    if max_iterations is not None and (max_iterations < 0 or max_iterations > 20):
        raise ValueError("max_iterations must be between 1 and 20 when a cap is supplied")
    if settle_s < 0 or settle_s > 60:
        raise ValueError("settle_s must be between 0 and 60 seconds")
    if max_replans < 0 or max_replans > 5:
        raise ValueError("max_replans must be between 0 and 5")
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError(
            "Viser is required; install it with: python -m pip install 'viser[urdf]>=1.0,<2'"
        ) from exc

    root = session.project_root
    perception_path = (
        perception_config_path
        or root / "config" / "perception.free_exploration.json"
    ).expanduser().resolve()
    config = PerceptionConfig.load(root, perception_path)
    camera_a_spec = next(
        camera for camera in config.cameras if camera.label == config.active_camera_labels[0]
    )
    if camera_a_spec.label != "A":
        raise AutoExplorationError(
            "automatic web preview requires camera A as the first active camera"
        )
    server = viser.ViserServer(host=host, port=port, label="Claude automatic garment exploration")
    state = _AutoState()
    state_lock = threading.Lock()
    auto_thread: threading.Thread | None = None
    auto_run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    auto_results_dir = session.results / "auto_exploration" / auto_run_stamp

    status = server.gui.add_markdown(
        "### Automatic exploration ready\n\n"
        "The web page owns the CamA RGB-D preview. The preview pauses during "
        "synchronized A/B capture so both cameras are never opened twice."
    )
    controls = server.gui.add_markdown(
        f"### Loop contract\n\n- max iterations: `{'continuous' if max_iterations is None else max_iterations}`\n"
        f"- settle time after motion: `{settle_s:.1f}s`\n"
        "- default: continuous iterations until Claude/user stop or hard failure\n"
        f"- pre-execution Claude replans on validation failure: `{max_replans}`\n"
        "- stop takes effect between phases; it cannot interrupt a command already sent"
    )
    run_log_panel = server.gui.add_markdown(
        f"### Agent log\n\nRun artifacts will be saved under `{_run_relative(auto_results_dir, session.run_dir)}`."
    )
    start_button = server.gui.add_button("Restart automatic exploration", color="red")
    stop_button = server.gui.add_button("Stop after current phase", disabled=True, color="orange")
    iteration_slider = server.gui.add_slider(
        "Maximum iterations (0 = continuous)",
        min=0,
        max=20,
        step=1,
        initial_value=0 if max_iterations is None else max_iterations,
    )
    history_panel = server.gui.add_markdown("### Agent history\n\nNo iteration has run.")
    proposal_panel = server.gui.add_markdown("### Current proposal\n\nNone.")
    evaluation_panel = server.gui.add_markdown("### Before/after judgement\n\nNone.")
    preview_panel = server.gui.add_markdown(
        "### Live CamA RGB-D\n\nWaiting for the camera preview to start."
    )
    preview_rgb_handle = server.gui.add_image(
        np.zeros((240, 320, 3), dtype=np.uint8), label="CamA live RGB"
    )
    preview_depth_handle = server.gui.add_image(
        np.zeros((240, 320, 3), dtype=np.uint8), label="CamA depth (near = bright)"
    )
    capture_rgb_handles: dict[str, Any] = {
        "A": server.gui.add_image(
            np.zeros((240, 320, 3), dtype=np.uint8), label="Latest capture RGB A"
        ),
        "B": server.gui.add_image(
            np.zeros((240, 320, 3), dtype=np.uint8), label="Latest capture RGB B"
        ),
    }
    capture_depth_handles: dict[str, Any] = {
        "A": server.gui.add_image(
            np.zeros((240, 320, 3), dtype=np.uint8), label="Latest capture depth A"
        ),
        "B": server.gui.add_image(
            np.zeros((240, 320, 3), dtype=np.uint8), label="Latest capture depth B"
        ),
    }
    live_cloud_handle = server.scene.add_point_cloud(
        "/live_preview/CamA",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        point_size=0.004,
        point_shape="circle",
    )
    capture_cloud_handles: dict[str, Any] = {
        label: server.scene.add_point_cloud(
            f"/live_preview/capture_{label}",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=np.zeros((1, 3), dtype=np.uint8),
            point_size=0.003,
            point_shape="circle",
            visible=False,
        )
        for label in ("A", "B")
    }

    def set_status(message: str) -> None:
        status.content = message

    def render_live_frame(frame: RGBDFrame) -> None:
        """Update the browser preview from the CamA reader thread."""

        try:
            preview_rgb_handle.image = np.asarray(frame.rgb, dtype=np.uint8)
            preview_depth_handle.image = _depth_preview(
                frame.depth_m,
                min_depth_m=config.min_depth_m,
                max_depth_m=config.max_depth_m,
            )
            points, colors = _frame_point_cloud(frame, stride=4)
            points, colors = _voxel_balance_cloud(
                points, colors, voxel_size_mm=5.0, max_points=25000
            )
            if len(points):
                live_cloud_handle.points = points
                live_cloud_handle.colors = colors
            depth = np.asarray(frame.depth_m)
            valid = np.isfinite(depth) & (depth > config.min_depth_m) & (depth < config.max_depth_m)
            preview_panel.content = (
                "### Live CamA RGB-D\n\n"
                f"- resolution: `{frame.rgb.shape[1]} × {frame.rgb.shape[0]}`\n"
                f"- valid depth: `{float(valid.mean()) * 100:.1f}%`\n"
                f"- point cloud: `{len(points):,}` points in base frame\n"
                f"- serial: `{frame.serial}`"
            )
        except Exception as exc:
            preview_panel.content = f"### Live CamA RGB-D\n\nPreview update failed: `{exc}`"

    def render_capture_frames(frames: list[RGBDFrame]) -> None:
        for frame in frames:
            label = frame.label.upper()
            if label not in capture_rgb_handles:
                continue
            capture_rgb_handles[label].image = np.asarray(frame.rgb, dtype=np.uint8)
            capture_depth_handles[label].image = _depth_preview(
                frame.depth_m,
                min_depth_m=config.min_depth_m,
                max_depth_m=config.max_depth_m,
            )
            points, colors = _frame_point_cloud(frame, stride=4)
            points, colors = _voxel_balance_cloud(
                points, colors, voxel_size_mm=5.0, max_points=25000
            )
            handle = capture_cloud_handles[label]
            if len(points):
                handle.points = points
                handle.colors = colors
                handle.visible = True

    def camera_window_closed() -> None:
        with state_lock:
            state.stop_requested = True
        status.content = (
            "### CamA preview stopped\n\n"
            "The CamA reader stopped unexpectedly. Automatic exploration will stop "
            "at the next safe phase boundary."
        )

    monitor = CameraAWebMonitor(
        root,
        perception_path,
        camera_a_spec,
        config,
        camera_window_closed,
        render_live_frame,
    )

    def render_history() -> None:
        with state_lock:
            entries = list(state.history)
        if not entries:
            history_panel.content = "### Agent history\n\nNo iteration has run."
            return
        lines = ["### Agent history", "", "| iteration | plan | useful | confidence | stop |", "|---:|---|---|---:|---|"]
        for entry in entries:
            evaluation = entry.get("evaluation") or {}
            lines.append(
                f"| {entry['iteration']} | `{entry.get('plan_status', 'unknown')}` | "
                f"`{evaluation.get('useful', '-')}` | `{evaluation.get('confidence', '-')}` | "
                f"`{evaluation.get('stop', '-')}` |"
            )
        history_panel.content = "\n".join(lines)

    def request_stop(_: Any) -> None:
        with state_lock:
            state.stop_requested = True
        set_status(
            "### Stop requested\n\nThe loop will stop after the current safe phase. "
            "An in-progress robot command is not interrupted automatically."
        )

    @stop_button.on_click
    def _(event: Any) -> None:
        request_stop(event)

    def stopped() -> bool:
        with state_lock:
            return state.stop_requested

    def set_running(value: bool) -> None:
        with state_lock:
            state.running = value
        start_button.disabled = value
        stop_button.disabled = not value
        iteration_slider.disabled = value

    def save_auto_record(name: str, payload: dict[str, Any]) -> None:
        directory = auto_results_dir
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / name, payload)

    def save_agent_artifact(
        iteration: int,
        record: dict[str, Any],
        *,
        phase: str,
        payload: Any,
        suffix: str = ".json",
    ) -> str:
        """Persist one agent phase in a stable per-iteration artifact file."""

        directory = auto_results_dir / f"iteration_{iteration:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{phase}{suffix}"
        if suffix == ".json":
            _write_json(path, payload)
        else:
            path.write_text(str(payload), encoding="utf-8")
        relative = _run_relative(path, session.run_dir)
        record.setdefault("artifacts", {})[phase] = relative
        return relative

    def _is_preexecution_replan_error(exc: BaseException) -> bool:
        """Only retry failures that occurred before any physical authority."""

        return isinstance(
            exc,
            (
                ExperimentValidationError,
                SafetyError,
                RobotExecutionError,
                AutoExplorationError,
                ExplorationPlanningError,
            ),
        ) and "physical rollout did not complete" not in str(exc)

    def _replan_feedback(exc: BaseException, *, proposal: ExplorationProposal | None) -> str:
        details = [f"Error type: {type(exc).__name__}", f"Error: {exc}"]
        if proposal is not None:
            details.append("Rejected proposal actions:")
            details.extend(json.dumps(action, ensure_ascii=False) for action in proposal.actions)
        details.append(
            "Generate a new proposal that addresses this exact failure. Keep the "
            "action sequence short and stay away from controller reachability boundaries."
        )
        return "\n".join(details)

    def run_loop(iterations: int | None) -> None:
        client = ClaudeAutoClient(binary=claude_binary, timeout_s=claude_timeout_s)
        try:
            objective = state.objective
            iteration = 0
            while iterations is None or iteration < iterations:
                iteration += 1
                if stopped():
                    break
                with state_lock:
                    state.iteration = iteration
                record: dict[str, Any] = {
                    "iteration": iteration,
                    "started_at": _now(),
                    "objective": objective,
                }
                record_saved = False
                try:
                    limit_label = "continuous" if iterations is None else str(iterations)
                    set_status(
                        f"### Iteration {iteration}/{limit_label}: perceiving\n\n"
                        "Pausing the CamA monitor and capturing synchronized A/B RGB-D."
                    )
                    monitor.stop()
                    frames = capture_two_view_rgbd(config)
                    render_capture_frames(frames)
                    perception = session.locate_cloth_center(config, frames=frames)
                    saved, saved_path = _load_latest_perception(session)
                    if saved is None or saved_path is None:
                        raise AutoExplorationError("perception completed without saved result")
                    before_images = perception_image_paths(saved, saved_path)
                    record["perception"] = perception
                    record["before_images"] = [
                        _run_relative(path, session.run_dir) for path in before_images
                    ]
                    save_agent_artifact(
                        iteration,
                        record,
                        phase="perception",
                        payload={
                            "result": perception,
                            "saved_result": saved,
                            "saved_result_path": _run_relative(saved_path, session.run_dir),
                            "before_images": record["before_images"],
                        },
                    )
                    if stopped():
                        break

                    set_status(
                        f"### Iteration {iteration}/{iterations}: Claude thinking\n\n"
                        "Planning one restricted reveal action from the current garment view."
                    )
                    proposal_feedback: str | None = None
                    proposal: ExplorationProposal | None = None
                    max_plan_attempts = max_replans + 1
                    for plan_attempt in range(1, max_plan_attempts + 1):
                        record["plan_attempt"] = plan_attempt
                        try:
                            proposal = client.plan(
                                before_images,
                                session,
                                objective,
                                feedback=proposal_feedback,
                            )
                            break
                        except Exception as exc:
                            if plan_attempt >= max_plan_attempts:
                                raise
                            proposal_feedback = _replan_feedback(exc, proposal=proposal)
                            save_agent_artifact(
                                iteration,
                                record,
                                phase=f"replan_{plan_attempt:02d}_feedback",
                                payload={
                                    "attempt": plan_attempt,
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "feedback": proposal_feedback,
                                },
                            )
                    if proposal is None:
                        raise AutoExplorationError("Claude did not return an exploration proposal")
                    plan_result = client.last_plan_result
                    if plan_result is None:
                        raise AutoExplorationError("Claude plan completed without a raw result")
                    with state_lock:
                        state.proposal = proposal
                    proposal_panel.content = _proposal_markdown(proposal, exploration_source(proposal))
                    source = exploration_source(proposal)
                    record["proposal"] = proposal.as_dict()
                    record["proposal_source"] = source
                    save_agent_artifact(iteration, record, phase="claude_plan", payload=plan_result)
                    save_agent_artifact(
                        iteration,
                        record,
                        phase="proposal",
                        payload=proposal.as_dict(),
                    )
                    save_agent_artifact(
                        iteration,
                        record,
                        phase="proposal_source",
                        payload=source,
                        suffix=".py",
                    )
                    source_path = session.workspace / "_auto_exploration.py"
                    source_path.write_text(source, encoding="utf-8")
                    try:
                        validation_feedback: str | None = None
                        controller = None
                        for validation_attempt in range(1, max_replans + 2):
                            try:
                                preflight = session.runner.preflight(source_path.name)
                                if preflight.error:
                                    raise ExperimentValidationError(preflight.error)
                                set_status(
                                    f"### Iteration {iteration}/{limit_label}: controller IK\n\n"
                                    "Validating every target without motion."
                                )
                                controller = validate_controller_trajectory(
                                    session.robot_config, preflight.actions
                                )
                                break
                            except Exception as exc:
                                if (
                                    validation_attempt >= max_replans + 1
                                    or not _is_preexecution_replan_error(exc)
                                ):
                                    raise
                                validation_feedback = _replan_feedback(exc, proposal=proposal)
                                set_status(
                                    f"### Iteration {iteration}/{limit_label}: Claude replanning "
                                    f"({validation_attempt}/{max_replans})\n\n"
                                    f"The candidate was rejected before execution: `{exc}`"
                                )
                                save_agent_artifact(
                                    iteration,
                                    record,
                                    phase=f"replan_{validation_attempt:02d}_ik_feedback",
                                    payload={
                                        "attempt": validation_attempt,
                                        "error": f"{type(exc).__name__}: {exc}",
                                        "feedback": validation_feedback,
                                    },
                                )
                                proposal = client.plan(
                                    before_images,
                                    session,
                                    objective,
                                    feedback=validation_feedback,
                                )
                                replanned_result = client.last_plan_result
                                source = exploration_source(proposal)
                                record["proposal"] = proposal.as_dict()
                                record["proposal_source"] = source
                                source_path.write_text(source, encoding="utf-8")
                                if replanned_result is not None:
                                    save_agent_artifact(
                                        iteration,
                                        record,
                                        phase=f"replan_{validation_attempt:02d}_claude_plan",
                                        payload=replanned_result,
                                    )
                                with state_lock:
                                    state.proposal = proposal
                                proposal_panel.content = _proposal_markdown(
                                    proposal, source
                                )
                                save_agent_artifact(
                                    iteration,
                                    record,
                                    phase=f"replan_{validation_attempt:02d}_proposal",
                                    payload=proposal.as_dict(),
                                )
                                save_agent_artifact(
                                    iteration,
                                    record,
                                    phase=f"replan_{validation_attempt:02d}_source",
                                    payload=source,
                                    suffix=".py",
                                )
                        if controller is None:
                            raise AutoExplorationError("controller validation returned no result")
                        record["requested_actions"] = preflight.actions
                        record["controller_warning_code"] = controller.controller_warning_code
                        save_agent_artifact(
                            iteration,
                            record,
                            phase="preflight",
                            payload={
                                "source": preflight.source,
                                "actions": preflight.actions,
                                "stdout": preflight.stdout,
                                "error": preflight.error,
                            },
                        )
                        save_agent_artifact(
                            iteration,
                            record,
                            phase="controller_ik",
                            payload=controller,
                        )
                        if stopped():
                            break

                        set_status(
                            f"### Iteration {iteration}/{limit_label}: executing\n\n"
                            "Executing exactly one validated physical rollout."
                        )
                        result = session.run_experiment(
                            source_path.name,
                            real=True,
                            confirmed=True,
                            single_view_confirmed=(
                                json.loads(
                                    (session.run_dir / "run_metadata.json").read_text(
                                        encoding="utf-8"
                                    )
                                ).get("last_perception_mode")
                                == "single_camera_rgbd"
                            ),
                            notes=f"Automatic Claude exploration iteration {iteration}.",
                        )
                        record["execution"] = result
                        save_agent_artifact(iteration, record, phase="execution", payload=result)
                        if not result.get("execution_completed"):
                            errors = result.get("robot_errors") or []
                            raise AutoExplorationError(
                                "physical rollout did not complete; automatic loop stopped: "
                                f"{errors}"
                            )
                    finally:
                        if source_path.is_file():
                            source_path.unlink()
                    if stopped():
                        break

                    set_status(
                        f"### Iteration {iteration}/{limit_label}: settling\n\n"
                        f"Waiting `{settle_s:.1f}s`, then resuming CamA monitoring."
                    )
                    time.sleep(settle_s)
                    monitor.start()
                    time.sleep(0.5)
                    set_status(
                        f"### Iteration {iteration}/{limit_label}: evaluating\n\n"
                        "Comparing before/after garment views with Claude."
                    )
                    monitor.stop()
                    after_frames = capture_two_view_rgbd(config)
                    render_capture_frames(after_frames)
                    after_dir = (
                        auto_results_dir
                        / f"iteration_{iteration:03d}_after"
                    )
                    after_images = _save_frame_images(after_frames, after_dir)
                    record["after_images"] = [
                        _run_relative(path, session.run_dir) for path in after_images
                    ]
                    save_agent_artifact(
                        iteration,
                        record,
                        phase="after_capture",
                        payload={
                            "images": record["after_images"],
                            "frame_labels": [frame.label for frame in after_frames],
                        },
                    )
                    evaluation = client.evaluate(
                        before_images,
                        after_images,
                        proposal=proposal,
                        run_dir=session.run_dir,
                    )
                    evaluation_result = client.last_evaluation_result
                    if evaluation_result is None:
                        raise AutoExplorationError(
                            "Claude evaluation completed without a raw result"
                        )
                    save_agent_artifact(
                        iteration,
                        record,
                        phase="claude_evaluation",
                        payload=evaluation_result,
                    )
                    with state_lock:
                        state.evaluation = evaluation
                        state.history.append(
                            {
                                "iteration": iteration,
                                "plan_status": "executed",
                                "evaluation": evaluation.as_dict(),
                            }
                        )
                    evaluation_panel.content = (
                        "### Before/after judgement\n\n"
                        f"- useful: `{evaluation.useful}`\n"
                        f"- confidence: `{evaluation.confidence:.2f}`\n"
                        f"- stop: `{evaluation.stop}`\n"
                        f"- observed change: {evaluation.observed_change}\n\n"
                        f"- next objective: {evaluation.next_objective}\n\n"
                        f"- reason: {evaluation.reason}"
                        + (
                            "\n\n- caveats:\n"
                            + "\n".join(f"  - {caveat}" for caveat in evaluation.caveats)
                            if evaluation.caveats
                            else ""
                        )
                    )
                    render_history()
                    save_auto_record(
                        f"iteration_{iteration:03d}.json",
                        {**record, "evaluation": evaluation.as_dict(), "completed_at": _now()},
                    )
                    run_log_panel.content = (
                        f"### Agent log\n\n"
                        f"Saved iteration `{iteration}` under `{_run_relative(auto_results_dir, session.run_dir)}`.\n\n"
                        f"Claude artifacts: `{len(record.get('artifacts', {}))}`"
                    )
                    record_saved = True
                    if evaluation.stop:
                        set_status(
                            f"### Automatic exploration stopped after iteration {iteration}\n\n"
                            f"Claude judged the current state sufficient or unsafe: {evaluation.reason}"
                        )
                        break
                    objective = evaluation.next_objective
                    with state_lock:
                        state.objective = objective
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    save_auto_record(
                        f"iteration_{iteration:03d}_failed.json",
                        {**record, "completed_at": _now()},
                    )
                    record_saved = True
                    with state_lock:
                        state.history.append(
                            {
                                "iteration": iteration,
                                "plan_status": "hard_failed",
                                "evaluation": {},
                            }
                        )
                    render_history()
                    set_status(
                        f"### Automatic exploration hard-stopped at iteration {iteration}\n\n"
                        f"`{type(exc).__name__}: {exc}`\n\n"
                        "No physical retry was attempted. Pre-execution replanning was "
                        "bounded by `--max-replans`; inspect the saved iteration record."
                    )
                    break
                finally:
                    if not record_saved:
                        save_auto_record(
                            f"iteration_{iteration:03d}_stopped.json",
                            {
                                **record,
                                "stop_requested": stopped(),
                                "completed_at": _now(),
                            },
                        )
                    if not stopped():
                        monitor.start()
            if iterations is not None and not stopped() and iteration >= iterations:
                set_status(
                    f"### Automatic exploration reached its limit ({iterations} iterations)\n\n"
                    "Review the CamA stream and saved before/after records."
                )
        finally:
            set_running(False)

    @start_button.on_click
    def _(event: Any) -> None:
        nonlocal auto_thread
        with state_lock:
            if state.running:
                return
            state.stop_requested = False
            state.history = []
            state.objective = (
                "Reveal more previously occluded garment surface with one cautious action."
            )
        selected_iterations = int(iteration_slider.value)
        iterations = None if selected_iterations == 0 else selected_iterations
        set_running(True)
        auto_thread = threading.Thread(
            target=run_loop,
            args=(iterations,),
            daemon=True,
            name="claude-auto-exploration-loop",
        )
        auto_thread.start()

    try:
        monitor.start()
        # Automatic mode starts immediately when the module is launched. The
        # restart button remains available for a fresh run after a stop.
        with state_lock:
            state.stop_requested = False
            state.history = []
            state.objective = (
                "Reveal more previously occluded garment surface with one cautious action."
            )
        selected_iterations = int(iteration_slider.value)
        initial_iterations = None if selected_iterations == 0 else selected_iterations
        set_running(True)
        auto_thread = threading.Thread(
            target=run_loop,
            args=(initial_iterations,),
            daemon=True,
            name="claude-auto-exploration-loop",
        )
        auto_thread.start()
        print(f"Viser Claude automatic exploration console: http://{host}:{port}")
        print(f"Run workspace: {session.workspace}")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        request_stop(None)
    finally:
        monitor.stop()
        if auto_thread is not None and auto_thread.is_alive():
            auto_thread.join(timeout=2.0)
        server.stop()
    return 0


def _load_session(
    root: Path,
    run_dir: Path | None,
    run_id: str | None,
    robot_config: Path | None,
) -> AgentSession:
    from .free_exploration import _load_or_create_session

    return _load_or_create_session(root, run_dir, run_id, robot_config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-dir")
    parser.add_argument("--run-id")
    parser.add_argument("--robot-config")
    parser.add_argument("--perception-config")
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--claude-timeout-s", type=int, default=300)
    parser.add_argument(
        "--max-replans",
        type=int,
        default=2,
        help="maximum Claude replans after pre-execution validation failure",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="number of automatic iterations; 0 means continuous until stop/hard failure",
    )
    parser.add_argument("--settle-s", type=float, default=2.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--enable-real",
        action="store_true",
        help="required: automatic mode sends physical xArm commands",
    )
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    robot_config = Path(args.robot_config).resolve() if args.robot_config else None
    perception_config = Path(args.perception_config).resolve() if args.perception_config else None
    session = _load_session(root, run_dir, args.run_id, robot_config)
    return run_auto_exploration_viewer(
        session,
        host=args.host,
        port=args.port,
        max_iterations=args.max_iterations,
        settle_s=args.settle_s,
        enable_real=args.enable_real,
        perception_config_path=perception_config,
        claude_binary=args.claude_binary,
        claude_timeout_s=args.claude_timeout_s,
        max_replans=args.max_replans,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
