"""Standalone Claude-driven garment folding experiment.

This module is intentionally independent from the generic opening/exploration
loop.  Each iteration plans exactly one fold-over-and-laydown action from the
current RGB-D/reference evidence, executes it only after the normal preflight
and controller-IK gates, captures a new observation, and asks a dedicated
evaluator whether the garment is now a neat folded stack.
"""

from __future__ import annotations

import argparse
import atexit
import itertools
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from .run_storage import find_run
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .config import ExperimentConfig, RobotConfig, SafetyError
from .experiment import ExperimentValidationError, format_action_sequence
from .free_exploration import (
    ClaudeExplorationClient,
    ExplorationPlanningError,
    ExplorationTimeoutError,
    _load_latest_perception,
    _load_or_create_session,
    exploration_source,
    global_perception_image_paths,
    ground_global_grasp_target,
    validate_global_exploration_payload,
)
from .garment_grounding_mcp import GroundingToolError
from .perception import PerceptionConfig, capture_two_view_rgbd
from .robot_api import validate_controller_trajectory
from .robot_api import move_robot_to_perception_position
from .session import AgentSession


class NeatFoldError(RuntimeError):
    """Raised when the standalone folding flow cannot continue safely."""


NEAT_FOLD_INSTRUCTION = (
    "Fold the garment into a neat, compact, flat stack. A successful fold means "
    "the visible garment body is aligned into a coherent garment-shaped stack, "
    "the major edges and hems are reasonably aligned, sleeves and loose panels "
    "are not left sticking out, the stack is not bunched or twisted, and it rests "
    "flat on the table."
)


NEAT_FOLD_PLANNING_OBJECTIVE = (
    "This is a top-down RGB-D view of one shirt. Decide which visible cloth "
    "point should be grasped next, then plan one action that folds that part "
    "into the shirt body and makes the shirt neater. Use the current RGB view "
    "as the main evidence; use the reference, depth, and workspace overlay only "
    "to check the choice."
)


NEAT_FOLD_PLAN_PROMPT = """
Plan the next grasp-and-fold action for the shirt shown in the top-down RGB-D
scene. First decide which visible cloth point should be grasped next. Then carry
that part into the shirt body so the shirt becomes neater. Choose the Camera A
pixel and all waypoints from the current scene; do not use fixed coordinates.
Return the required JSON proposal with the complete grasp, fold-over, laydown,
release, and home action sequence. The runtime will perform the grounding,
workspace, table-height, preflight, and controller checks before motion.
""".strip()


FOLD_EVALUATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["COMPLETE", "CONTINUE", "BLOCKED"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "stack_alignment": {"type": "string", "enum": ["ALIGNED", "MISALIGNED", "UNKNOWN"]},
        "flatness": {"type": "string", "enum": ["FLAT", "BUNCHED", "UNKNOWN"]},
        "protruding_parts": {"type": "string", "enum": ["NONE", "PRESENT", "UNKNOWN"]},
        "evidence": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {"type": "string", "minLength": 1},
        },
        "reason": {"type": "string", "minLength": 1},
        "next_fold_target": {"type": "string", "minLength": 1},
    },
    "required": [
        "status",
        "confidence",
        "stack_alignment",
        "flatness",
        "protruding_parts",
        "evidence",
        "reason",
        "next_fold_target",
    ],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _safe_claude(binary: str) -> str:
    resolved = shutil.which(binary) if Path(binary).name == binary else binary
    if resolved is None:
        raise NeatFoldError(f"Claude CLI not found: {binary}")
    return str(resolved)


def _json_from_claude(text: str) -> dict[str, Any]:
    try:
        outer = json.loads(text)
    except json.JSONDecodeError:
        outer = None
    if isinstance(outer, dict):
        if isinstance(outer.get("structured_output"), dict):
            return outer["structured_output"]
        if isinstance(outer.get("result"), str):
            try:
                result = json.loads(outer["result"])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
        if "status" in outer and "evidence" in outer:
            return outer
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise NeatFoldError("Claude folding evaluator did not return a JSON object")


def _claude_runtime_metrics(stdout: str) -> dict[str, Any]:
    """Extract the CLI envelope metrics emitted by Claude's JSON output."""

    try:
        envelope = json.loads(stdout)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(envelope, Mapping):
        return {}
    usage = envelope.get("usage")
    model_usage = envelope.get("modelUsage")
    metrics: dict[str, Any] = {}
    for key in (
        "duration_api_ms",
        "duration_ms",
        "num_turns",
        "stop_reason",
        "terminal_reason",
        "fast_mode_state",
        "fast_mode_disabled_reason",
        "total_cost_usd",
    ):
        if key in envelope:
            metrics[key] = envelope[key]
    if isinstance(usage, Mapping):
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            if key in usage:
                metrics[key] = usage[key]
    if isinstance(model_usage, Mapping):
        metrics["models"] = {
            str(name): {
                key: value
                for key, value in details.items()
                if key in {"inputTokens", "outputTokens", "costUSD", "canonicalModel"}
            }
            for name, details in model_usage.items()
            if isinstance(details, Mapping)
        }
    return metrics


def _validate_fold_evaluation(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = set(FOLD_EVALUATION_SCHEMA["required"])
    if set(payload) != required:
        raise NeatFoldError(
            f"fold evaluation fields mismatch; missing={sorted(required - set(payload))}, "
            f"extra={sorted(set(payload) - required)}"
        )
    if payload["status"] not in {"COMPLETE", "CONTINUE", "BLOCKED"}:
        raise NeatFoldError("fold evaluation status is invalid")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise NeatFoldError("fold evaluation confidence must be numeric")
    if not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
        raise NeatFoldError("fold evaluation confidence must be in [0,1]")
    for name, values in {
        "stack_alignment": {"ALIGNED", "MISALIGNED", "UNKNOWN"},
        "flatness": {"FLAT", "BUNCHED", "UNKNOWN"},
        "protruding_parts": {"NONE", "PRESENT", "UNKNOWN"},
    }.items():
        if payload[name] not in values:
            raise NeatFoldError(f"fold evaluation {name} is invalid")
    if not isinstance(payload["evidence"], list) or not payload["evidence"]:
        raise NeatFoldError("fold evaluation evidence must be non-empty")
    if any(not isinstance(item, str) or not item.strip() for item in payload["evidence"]):
        raise NeatFoldError("fold evaluation evidence must contain strings")
    for name in ("reason", "next_fold_target"):
        if not isinstance(payload[name], str) or not payload[name].strip():
            raise NeatFoldError(f"fold evaluation {name} must be non-empty")
    result = dict(payload)
    result["confidence"] = float(confidence)
    return result


def _validate_grounded_workspace_xy(
    grounding: Mapping[str, Any],
    robot_config: RobotConfig,
) -> None:
    """Reject a selected image pixel whose calibrated base XY is unreachable."""

    measurement = grounding.get("measurement")
    if not isinstance(measurement, Mapping):
        raise ExplorationPlanningError("grounding did not return a calibrated measurement")
    xyz = measurement.get("base_xyz_median_mm")
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
        raise ExplorationPlanningError("grounding measurement has no calibrated base XYZ")
    x_mm, y_mm, z_mm = (float(value) for value in xyz)
    bounds = robot_config.boundaries
    margin = float(robot_config.workspace_margin_mm)
    try:
        bounds.validate_lateral(x_mm, y_mm, margin)
    except SafetyError as exc:
        raise ExplorationPlanningError(str(exc)) from exc
    if bounds.x_min is not None and x_mm < bounds.x_min + margin:
        raise ExplorationPlanningError(
            "selected Camera A pixel grounds outside the robot workspace: "
            f"base_x={x_mm:.3f} below safe lower bound {bounds.x_min + margin:.3f}; "
            "choose a different pixel whose calibrated base XYZ is inside the workspace"
        )
    if bounds.x_max is not None and x_mm > bounds.x_max - margin:
        raise ExplorationPlanningError(
            "selected Camera A pixel grounds outside the robot workspace: "
            f"base_x={x_mm:.3f} above safe upper bound {bounds.x_max - margin:.3f}; "
            "choose a different pixel whose calibrated base XYZ is inside the workspace"
        )
    if bounds.y_min is not None and y_mm < bounds.y_min + margin:
        raise ExplorationPlanningError(
            "selected Camera A pixel grounds outside the robot workspace: "
            f"base_y={y_mm:.3f} below safe lower bound {bounds.y_min + margin:.3f}; "
            "choose a different pixel whose calibrated base XYZ is inside the workspace"
        )
    if bounds.y_max is not None and y_mm > bounds.y_max - margin:
        raise ExplorationPlanningError(
            "selected Camera A pixel grounds outside the robot workspace: "
            f"base_y={y_mm:.3f} above safe upper bound {bounds.y_max - margin:.3f}; "
            "choose a different pixel whose calibrated base XYZ is inside the workspace"
        )


def _workspace_prefilter_overlay(
    result: Mapping[str, Any],
    result_path: Path,
    robot_config: RobotConfig,
    output_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Render the calibrated Camera-A workspace as a visual planning gate.

    The overlay is built from the per-pixel base-XYZ map rather than from a
    guessed image-space rectangle.  Green garment pixels are inside the
    configured base-frame XY bounds; red garment pixels are visibly outside.
    The final grounding/IK checks remain authoritative after Claude returns.
    """

    camera_a = next(
        (
            view
            for view in result.get("views", [])
            if isinstance(view, Mapping) and str(view.get("label", "")).upper() == "A"
        ),
        None,
    )
    if camera_a is None:
        raise NeatFoldError("workspace prefilter requires a saved Camera A view")

    def resolve_artifact(raw: Any) -> Path:
        path = (result_path.parent / str(raw)).resolve()
        if not path.is_file():
            raise NeatFoldError(f"workspace prefilter artifact is missing: {path}")
        return path

    image_path = resolve_artifact(camera_a.get("image"))
    xyz_path = resolve_artifact(camera_a.get("base_xyz_map"))
    xyz = np.asarray(np.load(xyz_path), dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise NeatFoldError(f"Camera A base XYZ map has invalid shape: {xyz.shape}")
    image = Image.open(image_path).convert("RGBA")
    height, width = xyz.shape[:2]
    if image.size != (width, height):
        raise NeatFoldError(
            "Camera A RGB/base-XYZ dimensions disagree: "
            f"image={image.size}, xyz={(width, height)}"
        )

    finite = np.all(np.isfinite(xyz), axis=2)
    bounds = robot_config.boundaries
    margin = float(robot_config.workspace_margin_mm)
    x = xyz[:, :, 0]
    y = xyz[:, :, 1]
    inside = finite.copy()
    if bounds.lateral_points_mm is not None:
        nx, ny, low, high = bounds.lateral_geometry()
        lateral = nx * x + ny * y
        inside &= (lateral >= low + margin) & (lateral <= high - margin)
    if bounds.x_min is not None:
        inside &= x >= float(bounds.x_min) + margin
    if bounds.x_max is not None:
        inside &= x <= float(bounds.x_max) - margin
    if bounds.y_min is not None:
        inside &= y >= float(bounds.y_min) + margin
    if bounds.y_max is not None:
        inside &= y <= float(bounds.y_max) - margin

    garment = np.ones((height, width), dtype=bool)
    raw_mask = camera_a.get("garment_mask")
    if raw_mask:
        mask_path = resolve_artifact(raw_mask)
        loaded_mask = np.asarray(np.load(mask_path), dtype=bool)
        if loaded_mask.shape == (height, width):
            garment = loaded_mask
    safe_garment = inside & garment
    outside_garment = finite & garment & ~inside

    # Mark only transitions against the configured workspace, not every
    # garment silhouette edge.  This keeps the visual cue focused on the
    # prefilter boundary Claude needs for grasp selection.
    workspace_edge = np.zeros_like(inside)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbour_inside = np.zeros_like(inside)
        src_y = slice(max(0, dy), min(height, height + dy))
        src_x = slice(max(0, dx), min(width, width + dx))
        dst_y = slice(max(0, -dy), min(height, height - dy))
        dst_x = slice(max(0, -dx), min(width, width - dx))
        neighbour_inside[dst_y, dst_x] = inside[src_y, src_x]
        workspace_edge |= safe_garment & ~neighbour_inside

    rgba = np.asarray(image, dtype=np.uint8).copy()

    def tint(mask: np.ndarray, colour: tuple[int, int, int], alpha: int) -> None:
        if not np.any(mask):
            return
        source = np.asarray(colour, dtype=np.float64)
        rgba_rgb = rgba[mask, :3].astype(np.float64)
        rgba[mask, :3] = np.clip(
            rgba_rgb * (1.0 - alpha / 255.0) + source * (alpha / 255.0),
            0,
            255,
        ).astype(np.uint8)

    tint(outside_garment, (220, 40, 40), 78)
    tint(safe_garment, (30, 190, 70), 48)
    edge_mask = Image.fromarray((workspace_edge.astype(np.uint8) * 255), mode="L")
    edge_mask = edge_mask.filter(ImageFilter.MaxFilter(5))
    rgba[np.asarray(edge_mask) > 0, :3] = np.asarray((20, 255, 70), dtype=np.uint8)
    rgba[np.asarray(edge_mask) > 0, 3] = 255

    overlay = Image.fromarray(rgba, mode="RGBA")
    draw = ImageDraw.Draw(overlay, mode="RGBA")
    banner = (
        "WORKSPACE PREFILTER | GREEN: safe base XY | RED: outside bounds | "
        "final grounding still required"
    )
    draw.rectangle((0, 0, min(width, 1040), 32), fill=(10, 45, 20, 225))
    draw.text((10, 9), banner, fill=(255, 255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)

    metadata = {
        "camera": "A",
        "source_rgb": str(image_path),
        "source_base_xyz_map": str(xyz_path),
        "bounds_mm": {
            "x_min": bounds.x_min,
            "x_max": bounds.x_max,
            "y_min": bounds.y_min,
            "y_max": bounds.y_max,
            "margin": margin,
        },
        "image_size": [width, height],
        "finite_pixel_count": int(np.count_nonzero(finite)),
        "safe_pixel_count": int(np.count_nonzero(inside)),
        "safe_garment_pixel_count": int(np.count_nonzero(safe_garment)),
        "outside_garment_pixel_count": int(np.count_nonzero(outside_garment)),
        "overlay": str(output_path),
        "authority": "visual prefilter only; calibrated grounding and controller IK remain authoritative",
    }
    _write_json(output_path.with_suffix(".json"), metadata)
    return output_path.resolve(), metadata


class NeatFoldEvaluator:
    def __init__(self, binary: str = "claude", timeout_s: int = 400):
        self.binary = binary
        self.timeout_s = int(timeout_s)

    def evaluate(self, before: Sequence[Path], after: Sequence[Path], run_dir: Path) -> dict[str, Any]:
        if not before or not after:
            raise NeatFoldError("fold evaluation requires before and after images")
        prompt = (
            "Evaluate one completed action in a standalone garment-folding experiment.\n"
            "The garment is COMPLETE only when the after images visibly show a neat, compact, "
            "flat, untwisted stack: the main body and major edges/hems substantially overlap, "
            "sleeves and loose panels are tucked rather than protruding, and there is no large "
            "bunching or diagonal twist. A mere translation, spread, shake, or release is not "
            "completion. If one fold improved alignment but another panel remains loose, return "
            "CONTINUE and name the next visible fold target. If the state is too occluded or no "
            "safe fold target is visible, return BLOCKED. Read only the listed files and return "
            "the schema JSON.\n\nBEFORE:\n"
            + "\n".join(f"- {path.resolve()}" for path in before)
            + "\n\nAFTER:\n"
            + "\n".join(f"- {path.resolve()}" for path in after)
        )
        command = [
            _safe_claude(self.binary),
            "--print",
            prompt,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(FOLD_EVALUATION_SCHEMA, separators=(",", ":")),
            "--permission-mode",
            "dontAsk",
            "--allowedTools",
            "Read",
            "--tools",
            "Read",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--add-dir",
            str(run_dir.resolve()),
            "--system-prompt",
            "You are a strict visual evaluator for a garment-folding experiment. Read only the listed images and return JSON.",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=run_dir.resolve(),
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise NeatFoldError(f"fold evaluator timed out after {self.timeout_s} seconds") from exc
        if completed.returncode != 0:
            raise NeatFoldError(
                f"fold evaluator exited with {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        return _validate_fold_evaluation(_json_from_claude(completed.stdout))


class NeatFoldPipeline:
    """Run the independent fold-plan/execute/evaluate loop."""

    def __init__(
        self,
        session: Any,
        *,
        project_root: Path,
        perception_config: Path,
        claude_binary: str = "claude",
        claude_timeout_s: int = 900,
        max_folds: int = 4,
        max_plan_replans: int = 2,
        infinite_retries: bool = False,
        real: bool = False,
        confirm_real: bool = False,
        skip_controller_ik: bool = False,
        reuse_latest_perception: bool = False,
        viser: bool = False,
        viser_host: str = "127.0.0.1",
        viser_port: int = 8765,
        viser_refresh_s: float = 0.5,
    ):
        self.session = session
        self.project_root = project_root.resolve()
        self.perception_config = perception_config.resolve()
        self.claude_binary = claude_binary
        self.claude_timeout_s = int(claude_timeout_s)
        self.max_folds = int(max_folds)
        self.max_plan_replans = int(max_plan_replans)
        if self.max_plan_replans < 0:
            raise NeatFoldError("max_plan_replans must be non-negative")
        self.infinite_retries = bool(infinite_retries)
        self.real = bool(real)
        self.confirm_real = bool(confirm_real)
        self.skip_controller_ik = bool(skip_controller_ik)
        self.reuse_latest_perception = bool(reuse_latest_perception)
        self.planner = ClaudeExplorationClient(binary=claude_binary, timeout_s=claude_timeout_s)
        self.evaluator = NeatFoldEvaluator(binary=claude_binary, timeout_s=min(900, claude_timeout_s))
        self.viser = bool(viser)
        self.viser_host = str(viser_host)
        self.viser_port = int(viser_port)
        self.viser_refresh_s = float(viser_refresh_s)
        if self.viser_refresh_s < 0.1 or self.viser_refresh_s > 30.0:
            raise NeatFoldError("viser_refresh_s must be between 0.1 and 30 seconds")
        self._debug_started = time.monotonic()
        self._viser_process: subprocess.Popen[Any] | None = None

    def _debug(self, stage: str, message: str) -> None:
        elapsed = time.monotonic() - self._debug_started
        print(f"[neat-fold +{elapsed:7.1f}s] {stage}: {message}", flush=True)

    def _start_viser(self, output: Path) -> None:
        if not self.viser:
            return
        log_path = output / "viser.log"
        command = [
            sys.executable,
            "-m",
            "cloth_agent.molmo_artifact_viser",
            str(self.session.run_dir),
            "--host",
            self.viser_host,
            "--port",
            str(self.viser_port),
            "--refresh-s",
            str(self.viser_refresh_s),
        ]
        log = log_path.open("w", encoding="utf-8")
        try:
            self._viser_process = subprocess.Popen(
                command,
                cwd=self.project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        finally:
            log.close()
        atexit.register(self._stop_viser)
        self._debug(
            "viser",
            f"started read-only Viser at http://{self.viser_host}:{self.viser_port}; log={log_path}",
        )

    def _stop_viser(self) -> None:
        process = self._viser_process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
        self._viser_process = None

    def _output_dir(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = self.session.results / "neat_fold" / stamp
        output.mkdir(parents=True, exist_ok=False)
        return output

    def _copy_reference(self, output: Path) -> list[Path]:
        source = self.project_root / "data" / "reference" / "flat_garment_reference"
        destination = output / "reference"
        destination.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for name in ("camera_A_flat_reference.png", "camera_A_flat_reference_anchors.png"):
            path = source / name
            if path.is_file():
                target = destination / name
                shutil.copy2(path, target)
                paths.append(target.resolve())
        if not paths:
            raise NeatFoldError(f"flat garment reference is missing: {source}")
        return paths

    def _capture(self, config: PerceptionConfig, *, reuse: bool) -> tuple[dict[str, Any], Path]:
        if reuse:
            self._debug("perception", "reusing the latest saved RGB-D result")
            saved, saved_path = _load_latest_perception(self.session)
            if saved is None or saved_path is None:
                raise NeatFoldError("no saved perception is available for --reuse-latest-perception")
            return saved, saved_path
        if self.real:
            self._debug("perception", "moving to the calibrated perception pose")
            move_robot_to_perception_position(self.session.robot_config)
        self._debug("perception", "capturing synchronized Camera A/B RGB-D")
        frames = capture_two_view_rgbd(config)
        self._debug("perception", "running dense garment localization and depth fusion")
        self.session.locate_cloth_center(config, frames=frames)
        saved, saved_path = _load_latest_perception(self.session)
        if saved is None or saved_path is None:
            raise NeatFoldError("perception completed without a saved result")
        return saved, saved_path

    def run(self) -> dict[str, Any]:
        if self.max_folds < 1:
            raise NeatFoldError("max_folds must be positive")
        if self.real and not self.confirm_real:
            raise NeatFoldError("physical folding requires --real and --confirm-real")
        output = self._output_dir()
        summary: dict[str, Any] = {
            "created_at": _now(),
            "status": "RUNNING",
            "objective": NEAT_FOLD_INSTRUCTION,
            "run_dir": str(self.session.run_dir),
            "output_dir": str(output),
            "physical_execution": self.real,
            "planning_retry_policy": {
                "infinite_retries": self.infinite_retries,
                "max_plan_replans": self.max_plan_replans,
                "scope": "pre-execution planning, grounding, preflight, and controller IK only",
            },
            "iterations": [],
        }
        _write_json(output / "summary.json", summary)
        retry_text = "infinite pre-execution retries" if self.infinite_retries else f"max plan attempts={self.max_plan_replans + 1}"
        self._debug(
            "run",
            f"created output={output} real={self.real} max_folds={self.max_folds}; {retry_text}",
        )
        self._start_viser(output)
        config = PerceptionConfig.load(self.project_root, self.perception_config)
        reference_images = self._copy_reference(output)
        history: list[dict[str, Any]] = []
        rejected_pixels: list[dict[str, Any]] = []
        before_saved, before_path = self._capture(config, reuse=self.reuse_latest_perception)
        for iteration in range(1, self.max_folds + 1):
            self._debug("iteration", f"starting iteration {iteration}/{self.max_folds}")
            iteration_dir = output / f"iteration_{iteration:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=False)
            before_images = global_perception_image_paths(before_saved, before_path)
            workspace_overlay, workspace_prefilter = _workspace_prefilter_overlay(
                before_saved,
                before_path,
                self.session.robot_config,
                iteration_dir / "camera_A_workspace_prefilter.png",
            )
            planning_images = [workspace_overlay] + list(before_images) + reference_images
            self._debug(
                "planning",
                "Camera A workspace prefilter rendered; "
                f"safe garment pixels={workspace_prefilter['safe_garment_pixel_count']} "
                f"outside={workspace_prefilter['outside_garment_pixel_count']}",
            )
            prompt = (
                NEAT_FOLD_PLAN_PROMPT
                + "\n\nPlanning objective:\n"
                + NEAT_FOLD_PLANNING_OBJECTIVE
                + "\n\nAuthoritative robot workspace bounds in base-frame millimetres:\n"
                + json.dumps(asdict(self.session.robot_config.boundaries), ensure_ascii=False)
                + "\nKeep every move waypoint inside these bounds with margin; the runtime will reject any out-of-bounds waypoint.\n"
                + "\nCamera A workspace prefilter (inspect this overlay before selecting a grasp pixel):\n"
                + str(workspace_overlay)
                + "\nGreen garment pixels are inside the calibrated safe base-frame XY bounds; red garment pixels are outside and must not be selected. The overlay is a visual prefilter, not a replacement for final calibrated grounding.\n"
                + json.dumps(workspace_prefilter, ensure_ascii=False, indent=2)
                + "\nRejected grasp pixels from this iteration (do not select them again):\n"
                + json.dumps(rejected_pixels, ensure_ascii=False, indent=2)
                + "\n\nPrevious folding outcomes (evidence only):\n"
                + json.dumps(history[-4:], ensure_ascii=False, indent=2)
            )
            source_path = self.session.workspace / f"_neat_fold_{iteration:03d}.py"
            planning_attempts: list[dict[str, Any]] = []
            planning_attempts_path = iteration_dir / "planning_attempts.json"
            validation_feedback: str | None = None
            proposal = None
            grounding = None
            source = ""
            preflight = None
            controller = None
            if self.infinite_retries:
                attempt_numbers = itertools.count(1)
                attempt_limit_text = "∞"
            else:
                attempt_numbers = iter(range(1, self.max_plan_replans + 2))
                attempt_limit_text = str(self.max_plan_replans + 1)
            for plan_attempt in attempt_numbers:
                self._debug("planning", f"iteration {iteration} Claude attempt {plan_attempt}/{attempt_limit_text}")
                attempt_prompt = prompt
                if rejected_pixels:
                    attempt_prompt += (
                        "\n\nUpdated rejected grasp pixels for this iteration. These pixels are forbidden; "
                        "choose a different Camera A pixel and verify its calibrated base XYZ:\n"
                        + json.dumps(rejected_pixels, ensure_ascii=False, indent=2)
                    )
                if validation_feedback:
                    attempt_prompt += (
                        "\n\nThe previous plan was rejected before any robot command was sent. "
                        "Keep the folding objective, but correct the exact deterministic failure "
                        "below. Reinspect the current scene and return a materially corrected "
                        "proposal; do not repeat the rejected waypoint.\n"
                        + validation_feedback
                    )
                # Give Claude the most recent structured attempts as a compact,
                # machine-readable correction history.  Full prompts, raw CLI
                # output, and per-attempt artifacts remain on disk; only the
                # bounded recent window is injected to avoid prompt growth.
                if planning_attempts:
                    attempt_prompt += (
                        "\n\nRecent planning attempts from this same iteration (the files named "
                        "below contain the complete records; use this history to avoid repeating "
                        "the same pixel or waypoint failure):\n"
                        + json.dumps(planning_attempts[-6:], ensure_ascii=False, indent=2, default=str)
                    )
                attempt_prompt_path = iteration_dir / f"plan_attempt_{plan_attempt:02d}.prompt.md"
                attempt_prompt_path.write_text(attempt_prompt, encoding="utf-8")
                attempt_record: dict[str, Any] = {"attempt": plan_attempt, "status": "PLANNING"}
                attempt_record["started_at"] = _now()
                attempt_record["prompt_path"] = str(attempt_prompt_path)
                attempt_record["rejected_pixels_before_attempt"] = list(rejected_pixels)
                attempt_record["validation_feedback_before_attempt"] = validation_feedback
                candidate = None
                candidate_grounding = None
                raw_proposal = None
                candidate_source = ""
                candidate_preflight = None
                candidate_controller = None
                response = None
                try:
                    response = self.planner.invoke(
                        planning_images,
                        attempt_prompt,
                        self.session.run_dir,
                        direct_prompt=True,
                    )
                    claude_metrics = _claude_runtime_metrics(response.stdout)
                    attempt_record["claude_metrics"] = claude_metrics
                    self._debug(
                        "planning",
                        "Claude returned a structured fold proposal; "
                        f"duration={float(claude_metrics.get('duration_ms', 0.0)) / 1000.0:.1f}s "
                        f"turns={claude_metrics.get('num_turns', '?')} "
                        f"models={list((claude_metrics.get('models') or {}).keys())}",
                    )
                    raw_proposal = response.proposal
                    attempt_record["claude"] = {
                        "created_at": response.created_at,
                        "returncode": response.returncode,
                        "command": list(response.command),
                        "metrics": claude_metrics,
                        "proposal": response.proposal.as_dict(),
                    }
                    raw_response_path = iteration_dir / f"plan_attempt_{plan_attempt:02d}.claude.json"
                    _write_json(
                        raw_response_path,
                        {
                            "created_at": response.created_at,
                            "command": list(response.command),
                            "returncode": response.returncode,
                            "prompt": response.prompt,
                            "stdout": response.stdout,
                            "stderr": response.stderr,
                            "proposal": response.proposal.as_dict(),
                        },
                    )
                    attempt_record["claude_raw_path"] = str(raw_response_path)
                    candidate, candidate_grounding = ground_global_grasp_target(
                        response.proposal,
                        self.session.workspace / "perception_views",
                        robot_config=self.session.robot_config,
                    )
                    _validate_grounded_workspace_xy(
                        candidate_grounding,
                        self.session.robot_config,
                    )
                    if candidate.requires_lift_checkpoint:
                        candidate = replace(candidate, requires_lift_checkpoint=False)
                    self._debug(
                        "grounding",
                        f"selected={candidate.selected_grasp} measured={candidate_grounding.get('measurement', {}).get('base_xyz_median_mm')}",
                    )
                    candidate_source = exploration_source(candidate)
                    source_path.write_text(candidate_source, encoding="utf-8")
                    candidate_preflight = self.session.runner.preflight(source_path.name)
                    if candidate_preflight.error:
                        raise ExperimentValidationError(candidate_preflight.error)
                    self._debug("preflight", f"passed with {len(candidate_preflight.actions)} actions")
                    candidate_controller = (
                        {"status": "SKIPPED"}
                        if self.skip_controller_ik
                        else validate_controller_trajectory(
                            self.session.robot_config, candidate_preflight.actions
                        )
                    )
                    self._debug("controller", "IK trajectory validation passed")
                except ExplorationTimeoutError as exc:
                    # A model timeout is not a geometry correction problem;
                    # retrying it here would silently multiply the wall-clock
                    # wait. The caller can rerun the standalone process.
                    timeout_error = f"{type(exc).__name__}: {exc}"
                    attempt_record.update(
                        {
                            "status": "CLAUDE_TIMEOUT",
                            "error": timeout_error,
                            "finished_at": _now(),
                        }
                    )
                    planning_attempts.append(attempt_record)
                    _write_json(iteration_dir / f"plan_attempt_{plan_attempt:02d}.json", attempt_record)
                    _write_json(planning_attempts_path, planning_attempts)
                    raise
                except (ExperimentValidationError, ExplorationPlanningError, GroundingToolError, SafetyError) as exc:
                    selected = (
                        candidate.selected_grasp
                        if candidate is not None
                        else raw_proposal.selected_grasp
                        if raw_proposal is not None
                        else None
                    )
                    measurement = (
                        candidate_grounding.get("measurement")
                        if isinstance(candidate_grounding, Mapping)
                        else None
                    )
                    if isinstance(selected, Mapping):
                        rejected_pixels.append(
                            {
                                "camera": selected.get("camera"),
                                "pixel_xy": selected.get("pixel_xy"),
                                "reason": str(exc),
                                "measured_base_xyz_mm": (
                                    measurement.get("base_xyz_median_mm")
                                    if isinstance(measurement, Mapping)
                                    else None
                                ),
                            }
                        )
                    validation_feedback = (
                        f"{type(exc).__name__}: {exc}\n"
                        "This was a pre-execution validation failure; no physical command was sent. "
                        "The selected pixel must be changed, and the replacement must be validated "
                        "in calibrated robot-base coordinates, not judged only by image-space appearance."
                    )
                    attempt_record.update(
                        {
                            "status": "REJECTED_BEFORE_EXECUTION",
                            "error": validation_feedback,
                            "feedback_sent_to_claude": validation_feedback,
                            "proposal": raw_proposal.as_dict() if raw_proposal is not None else None,
                            "grounding": candidate_grounding,
                            "selected_grasp": (
                                candidate.selected_grasp
                                if candidate is not None
                                else raw_proposal.selected_grasp
                                if raw_proposal is not None
                                else None
                            ),
                            "source": candidate_source or None,
                            "preflight": asdict(candidate_preflight) if candidate_preflight is not None else None,
                            "controller_ik": candidate_controller,
                            "finished_at": _now(),
                        }
                    )
                    self._debug("planning", f"rejected before execution: {validation_feedback}")
                    planning_attempts.append(attempt_record)
                    _write_json(iteration_dir / f"plan_attempt_{plan_attempt:02d}.json", attempt_record)
                    _write_json(planning_attempts_path, planning_attempts)
                    _write_json(
                        iteration_dir / "result.json",
                        {
                            "iteration": iteration,
                            "status": "PLAN_REJECTED",
                            "objective": NEAT_FOLD_INSTRUCTION,
                            "saved_perception_result": str(before_path),
                            "before_images": [str(path) for path in before_images],
                            "after_images": [],
                            "planning_attempts": planning_attempts,
                            "planning_attempts_file": str(planning_attempts_path),
                            "workspace_prefilter": workspace_prefilter,
                            "workspace_prefilter_overlay": str(workspace_overlay),
                            "error": validation_feedback,
                        },
                    )
                    if not self.infinite_retries and plan_attempt > self.max_plan_replans:
                        raise NeatFoldError(
                            "Claude could not produce a workspace/controller-valid fold plan "
                            f"after {plan_attempt} attempt(s): {validation_feedback}"
                        ) from exc
                    continue
                proposal = candidate
                grounding = candidate_grounding
                source = candidate_source
                preflight = candidate_preflight
                controller = candidate_controller
                attempt_record.update(
                    {
                        "status": "ACCEPTED",
                        "proposal": proposal.as_dict(),
                        "grounding": grounding,
                        "controller_ik": controller,
                        "preflight": asdict(preflight),
                        "source": source,
                        "finished_at": _now(),
                    }
                )
                planning_attempts.append(attempt_record)
                _write_json(iteration_dir / f"plan_attempt_{plan_attempt:02d}.json", attempt_record)
                _write_json(planning_attempts_path, planning_attempts)
                self._debug("planning", f"accepted fold plan on attempt {plan_attempt}")
                break
            if proposal is None or grounding is None or preflight is None or controller is None:
                raise NeatFoldError("fold planning ended without a validated proposal")
            plan_record = {
                "proposal": proposal.as_dict(),
                "grounding": grounding,
                "source": source,
                "preflight": preflight,
                "controller_ik": controller,
                "planning_attempts": planning_attempts,
                "planning_attempts_file": str(planning_attempts_path),
                "workspace_prefilter": workspace_prefilter,
                "workspace_prefilter_overlay": str(workspace_overlay),
                "before_images": [str(path) for path in before_images],
                "planning_images": [str(path) for path in planning_images],
            }
            _write_json(iteration_dir / "plan.json", plan_record)
            _write_json(
                iteration_dir / "result.json",
                {
                    "iteration": iteration,
                    "status": "PLANNED",
                    "objective": NEAT_FOLD_INSTRUCTION,
                    "saved_perception_result": str(before_path),
                    "before_images": [str(path) for path in before_images],
                    "after_images": [],
                    "proposal": proposal.as_dict(),
                    "grounding": grounding,
                    "preflight": asdict(preflight),
                    "controller_ik": controller,
                    "planning_attempts": planning_attempts,
                    "planning_attempts_file": str(planning_attempts_path),
                    "workspace_prefilter": workspace_prefilter,
                    "workspace_prefilter_overlay": str(workspace_overlay),
                },
            )
            (iteration_dir / "proposal.py").write_text(source, encoding="utf-8")
            print(format_action_sequence(preflight.actions), flush=True)
            if not self.real:
                self._debug("run", "dry-run validated; no physical command sent")
                summary["iterations"].append({"iteration": iteration, "status": "DRY_RUN_VALIDATED"})
                summary["status"] = "DRY_RUN_VALIDATED"
                summary["completed_at"] = _now()
                _write_json(output / "summary.json", summary)
                return summary
            execution = self.session.run_experiment(
                source_path.name,
                real=True,
                confirmed=True,
                notes=f"Standalone neat-fold iteration {iteration}.",
            )
            self._debug("execution", f"completed={execution.get('execution_completed')} returning to perception")
            _write_json(iteration_dir / "execution.json", execution)
            after_saved, after_path = self._capture(config, reuse=False)
            after_images = global_perception_image_paths(after_saved, after_path)
            evaluation = self.evaluator.evaluate(before_images, after_images, self.session.run_dir)
            self._debug(
                "evaluation",
                f"status={evaluation.get('status')} confidence={evaluation.get('confidence'):.3f}",
            )
            _write_json(iteration_dir / "evaluation.json", evaluation)
            record = {
                "iteration": iteration,
                "proposal": proposal.as_dict(),
                "execution": execution,
                "evaluation": evaluation,
                "before_images": [str(path) for path in before_images],
                "after_images": [str(path) for path in after_images],
            }
            history.append(record)
            summary["iterations"].append(
                {"iteration": iteration, "status": evaluation["status"], "confidence": evaluation["confidence"]}
            )
            _write_json(iteration_dir / "record.json", record)
            _write_json(
                iteration_dir / "result.json",
                {
                    "iteration": iteration,
                    "status": evaluation["status"],
                    "objective": NEAT_FOLD_INSTRUCTION,
                    "saved_perception_result": str(after_path),
                    "before_images": [str(path) for path in before_images],
                    "after_images": [str(path) for path in after_images],
                    "proposal": proposal.as_dict(),
                    "grounding": grounding,
                    "preflight": asdict(preflight),
                    "controller_ik": controller,
                    "planning_attempts": planning_attempts,
                    "planning_attempts_file": str(planning_attempts_path),
                    "workspace_prefilter": workspace_prefilter,
                    "workspace_prefilter_overlay": str(workspace_overlay),
                    "execution": execution,
                    "evaluation": evaluation,
                },
            )
            _write_json(output / "summary.json", summary)
            if evaluation["status"] in {"COMPLETE", "BLOCKED"}:
                summary["status"] = evaluation["status"]
                summary["completed_at"] = _now()
                _write_json(output / "summary.json", summary)
                return summary
            before_saved, before_path = after_saved, after_path
            self.reuse_latest_perception = False
        summary["status"] = "MAX_FOLDS_REACHED"
        summary["completed_at"] = _now()
        _write_json(output / "summary.json", summary)
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-dir", type=Path)
    group.add_argument("--run-id")
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path("config/robot.example.json"),
        help=(
            "robot configuration JSON (default: config/robot.example.json; "
            "uses absolute camera depth without live tabletop Z flooring)"
        ),
    )
    parser.add_argument("--perception-config", type=Path, default=Path("config/perception.free_exploration.json"))
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--claude-timeout-s", type=int, default=900)
    parser.add_argument("--max-folds", type=int, default=4)
    parser.add_argument(
        "--max-plan-replans",
        type=int,
        default=2,
        help="extra Claude planning attempts after preflight/workspace/IK rejection (default: 2)",
    )
    parser.add_argument(
        "--infinite-retries",
        action="store_true",
        help=(
            "keep retrying pre-execution planning/grounding/preflight/IK failures until a "
            "validated plan is found; every attempt is saved and recent failures are sent "
            "to the next Claude call"
        ),
    )
    parser.add_argument("--reuse-latest-perception", action="store_true")
    parser.add_argument("--skip-controller-ik", action="store_true")
    parser.add_argument("--viser", action="store_true", help="start the read-only Viser artifact viewer")
    parser.add_argument("--viser-host", default="127.0.0.1")
    parser.add_argument("--viser-port", type=int, default=8765)
    parser.add_argument("--viser-refresh-s", type=float, default=0.5)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--confirm-real", action="store_true")
    return parser


def _load_fold_session(
    root: Path,
    *,
    run_dir: Path | None,
    run_id: str | None,
    robot_config: Path | None,
) -> AgentSession:
    """Load an existing run or create one whose goal is specifically folding."""

    if run_dir is not None or (run_id and find_run(root, run_id) is not None):
        return _load_or_create_session(root, run_dir, run_id, robot_config)
    robot = RobotConfig.load(root, robot_config)
    return AgentSession.create(
        root,
        NEAT_FOLD_INSTRUCTION,
        robot,
        ExperimentConfig(),
        run_id=run_id,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    perception = args.perception_config if args.perception_config.is_absolute() else root / args.perception_config
    session = _load_fold_session(
        root,
        run_dir=Path(args.run_dir).resolve() if args.run_dir else None,
        run_id=args.run_id,
        robot_config=args.robot_config.resolve() if args.robot_config else None,
    )
    try:
        summary = NeatFoldPipeline(
            session,
            project_root=root,
            perception_config=perception,
            claude_binary=args.claude_binary,
            claude_timeout_s=args.claude_timeout_s,
            max_folds=args.max_folds,
            max_plan_replans=args.max_plan_replans,
            infinite_retries=args.infinite_retries,
            real=args.real,
            confirm_real=args.confirm_real,
            skip_controller_ik=args.skip_controller_ik,
            reuse_latest_perception=args.reuse_latest_perception,
            viser=args.viser,
            viser_host=args.viser_host,
            viser_port=args.viser_port,
            viser_refresh_s=args.viser_refresh_s,
        ).run()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["status"] != "BLOCKED" else 1
    except (NeatFoldError, ExplorationPlanningError, GroundingToolError, ExperimentValidationError, ValueError, FileNotFoundError, PermissionError, RuntimeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
