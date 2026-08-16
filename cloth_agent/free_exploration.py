"""Claude-guided, safety-gated free exploration for the Viser console.

This module is intentionally separate from :mod:`cloth_agent.viewer`.  It adds
an experimental loop in which Claude observes the saved garment photographs,
describes what it sees, and proposes one restricted RobotAPI program intended
to search for a useful garment lifting anchor.  The proposal is only a plan until the normal
static preflight, workspace checks, controller IK, and animation review pass.

No existing viewer or runtime file is patched by this feature.  Start it with
``python -m cloth_agent.free_exploration`` after creating a run (or provide a
run directory created by the regular CLI).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import ExperimentConfig, RobotConfig
from .experiment import ExperimentValidationError, Preflight
from .kinematics import AnimationFrame, XArm7Kinematics
from .perception import PerceptionConfig, capture_two_view_rgbd
from .robot_api import ControllerTrajectoryValidation, validate_controller_trajectory
from .session import AgentSession
from .skills import skill_prompt, validate_skill_name
from .viewer import (
    _load_latest_perception,
    _preflight_markdown,
    _source_hash,
    _view_point_cloud,
    path_waypoints_mm,
)


EXPLORATION_FIELDS = frozenset(
    {
        "garment_observation",
        "reveal_strategy",
        "confidence",
        "actions",
        "expected_observation",
        "safety_notes",
    }
)
EXPLORATION_OPTIONAL_FIELDS = frozenset({"skill_invocations"})
EXPLORATION_ACTIONS = frozenset({"move", "open_gripper", "close_gripper", "home"})
MAX_EXPLORATION_ACTIONS = 12


def _voxel_balance_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    voxel_size_mm: float = 5.0,
    max_points: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """Make point density comparable in base-frame space, not image pixels.

    Two cameras can have identical RGB-D image strides but different physical
    point spacing because their intrinsics and distance to the garment differ.
    Keeping one representative point per fixed-size base-frame voxel makes the
    Viser overlays visually comparable without inventing any geometry.
    """

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if colors.ndim != 2 or colors.shape[1] != 3 or len(colors) != len(points):
        raise ValueError("colors must have shape (N, 3) and match points")
    if voxel_size_mm <= 0 or max_points <= 0:
        raise ValueError("voxel_size_mm and max_points must be positive")
    if len(points) == 0:
        return points.astype(np.float32), colors.astype(np.uint8)

    voxel_m = float(voxel_size_mm) / 1000.0
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    colors = colors[finite]
    if len(points) == 0:
        return points.astype(np.float32), colors.astype(np.uint8)
    keys = np.floor(points / voxel_m).astype(np.int64)
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    first_indices.sort()
    if len(first_indices) > max_points:
        # Deterministic budget cap; preserve the spatial ordering produced by
        # the RGB-D raster instead of introducing random visual flicker.
        selected = np.linspace(0, len(first_indices) - 1, max_points, dtype=np.int64)
        first_indices = first_indices[selected]
    return points[first_indices].astype(np.float32), colors[first_indices].astype(np.uint8)


def _normalized_view_point_cloud(
    view: dict[str, Any],
    result_dir: Path,
    *,
    voxel_size_mm: float = 5.0,
    max_points: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one saved view at full raster resolution and balance it in 3-D."""

    points, colors = _view_point_cloud(view, result_dir, stride=1)
    return _voxel_balance_cloud(
        points,
        colors,
        voxel_size_mm=voxel_size_mm,
        max_points=max_points,
    )


class ExplorationPlanningError(ValueError):
    """Raised when Claude's free-exploration response violates its contract."""


def _controller_ik_failure_message(exc: BaseException) -> str:
    """Turn the xArm's terse read-only IK failure into an operator diagnosis."""

    message = str(exc)
    if "controller IK rejected action" not in message:
        return message
    if "code=10" in message:
        return (
            f"{message}\n\n"
            "xArm code 10 means the controller returned an invalid/failed IK result "
            "for this pose. The target can pass local XYZ bounds and still be "
            "unreachable with the fixed roll/pitch/yaw. Reduce the lateral x/y "
            "offset or lower the lift height, then ask Claude for a new proposal. "
            "Do not bypass controller IK."
        )
    return (
        f"{message}\n\n"
        "The xArm controller rejected this Cartesian target during read-only IK. "
        "Review the pose and controller status before retrying."
    )


@dataclass(frozen=True)
class ExplorationProposal:
    """A validated observation, rationale, and restricted action sequence."""

    garment_observation: str
    reveal_strategy: str
    confidence: float
    actions: tuple[dict[str, Any], ...]
    expected_observation: str
    safety_notes: tuple[str, ...]
    skill_invocations: tuple[dict[str, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "garment_observation": self.garment_observation,
            "reveal_strategy": self.reveal_strategy,
            "confidence": self.confidence,
            "actions": [dict(action) for action in self.actions],
            "expected_observation": self.expected_observation,
            "safety_notes": list(self.safety_notes),
            "skill_invocations": [dict(skill) for skill in self.skill_invocations],
        }


@dataclass(frozen=True)
class ClaudeExplorationResult:
    prompt: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    created_at: str
    proposal: ExplorationProposal


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExplorationPlanningError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ExplorationPlanningError(f"{name} must be finite")
    return number


def validate_exploration_payload(payload: Any) -> ExplorationProposal:
    """Validate Claude's JSON before generating executable RobotAPI source."""

    if not isinstance(payload, dict):
        raise ExplorationPlanningError("Claude response must be a JSON object")
    missing = EXPLORATION_FIELDS.difference(payload)
    unknown = set(payload).difference(EXPLORATION_FIELDS | EXPLORATION_OPTIONAL_FIELDS)
    if missing:
        raise ExplorationPlanningError(f"missing exploration fields: {sorted(missing)}")
    if unknown:
        raise ExplorationPlanningError(f"unknown exploration fields: {sorted(unknown)}")

    strings = ("garment_observation", "reveal_strategy", "expected_observation")
    for name in strings:
        if not isinstance(payload[name], str) or not payload[name].strip():
            raise ExplorationPlanningError(f"{name} must be a non-empty string")
    confidence = _finite_number(payload["confidence"], "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ExplorationPlanningError("confidence must be between 0 and 1")

    raw_notes = payload["safety_notes"]
    if not isinstance(raw_notes, list) or not raw_notes or len(raw_notes) > 10:
        raise ExplorationPlanningError("safety_notes must contain 1 to 10 strings")
    notes: list[str] = []
    for note in raw_notes:
        if not isinstance(note, str) or not note.strip():
            raise ExplorationPlanningError("every safety note must be a non-empty string")
        notes.append(note.strip())

    raw_skills = payload.get("skill_invocations", [])
    if not isinstance(raw_skills, list) or len(raw_skills) > 4:
        raise ExplorationPlanningError(
            "skill_invocations must be a list with at most four entries"
        )
    skill_invocations: list[dict[str, str]] = []
    for index, raw_skill in enumerate(raw_skills, start=1):
        if not isinstance(raw_skill, dict) or set(raw_skill) != {"name", "reason"}:
            raise ExplorationPlanningError(
                f"skill invocation {index} must contain exactly name and reason"
            )
        try:
            name = validate_skill_name(raw_skill["name"])
        except (TypeError, ValueError) as exc:
            raise ExplorationPlanningError(str(exc)) from exc
        reason = raw_skill["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ExplorationPlanningError(
                f"skill invocation {index} reason must be a non-empty string"
            )
        skill_invocations.append({"name": name, "reason": reason.strip()})

    raw_actions = payload["actions"]
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ExplorationPlanningError("actions must contain at least one action")
    if len(raw_actions) > MAX_EXPLORATION_ACTIONS:
        raise ExplorationPlanningError(
            f"actions exceed the {MAX_EXPLORATION_ACTIONS}-action safety limit"
        )
    actions: list[dict[str, Any]] = []
    move_count = 0
    for index, raw in enumerate(raw_actions, start=1):
        if not isinstance(raw, dict) or set(raw) != {"name", "args"}:
            raise ExplorationPlanningError(
                f"action {index} must contain exactly name and args"
            )
        name = raw["name"]
        args = raw["args"]
        if not isinstance(name, str) or name not in EXPLORATION_ACTIONS:
            raise ExplorationPlanningError(f"action {index} has unsupported name {name!r}")
        if name == "move":
            move_count += 1
            if not isinstance(args, dict) or set(args) != {"x", "y", "z", "yaw"}:
                raise ExplorationPlanningError(
                    f"move action {index} args must be exactly x, y, z, yaw"
                )
            clean_args = {
                key: _finite_number(args[key], f"action {index} {key}")
                for key in ("x", "y", "z", "yaw")
            }
        else:
            if args != {}:
                raise ExplorationPlanningError(
                    f"{name} action {index} must use an empty args object"
                )
            clean_args = {}
        actions.append({"name": name, "args": clean_args})
    if move_count == 0:
        raise ExplorationPlanningError("exploration actions must contain at least one move target")

    # The action geometry is Claude's decision. Require only the safety-critical
    # grasp/release envelope: an approach move, a post-grasp move for either an
    # exploratory test or a larger maneuver, and an explicit release.  A probe
    # is allowed to end after one post-grasp move; the planner is not forced to
    # emit a complete reveal or laydown trajectory.
    close_index = next(
        (index for index, action in enumerate(actions) if action["name"] == "close_gripper"),
        None,
    )
    if close_index is None:
        raise ExplorationPlanningError(
            "exploration actions must include close_gripper after an approach move"
        )
    if not any(action["name"] == "move" for action in actions[:close_index]):
        raise ExplorationPlanningError("a move approach is required before close_gripper")
    first_release = next(
        (
            index
            for index, action in enumerate(actions[close_index + 1 :], start=close_index + 1)
            if action["name"] == "open_gripper"
        ),
        None,
    )
    if first_release is None:
        raise ExplorationPlanningError(
            "exploration actions must include open_gripper at Claude's chosen release"
        )
    moves_before_release = [
        action
        for action in actions[close_index + 1 : first_release]
        if action["name"] == "move"
    ]
    if not moves_before_release:
        raise ExplorationPlanningError(
            "at least one post-grasp move is required before release for an "
            "exploratory lift or maneuver"
        )
    return ExplorationProposal(
        garment_observation=payload["garment_observation"].strip(),
        reveal_strategy=payload["reveal_strategy"].strip(),
        confidence=confidence,
        actions=tuple(actions),
        expected_observation=payload["expected_observation"].strip(),
        safety_notes=tuple(notes),
        skill_invocations=tuple(skill_invocations),
    )


def _json_from_claude_text(text: str) -> dict[str, Any]:
    """Extract one JSON object from direct, wrapped, or fenced CLI output."""

    candidates = [text.strip()]
    try:
        outer = json.loads(text)
        if isinstance(outer, dict) and isinstance(outer.get("result"), str):
            candidates.insert(0, outer["result"])
        elif isinstance(outer, dict) and isinstance(outer.get("result"), dict):
            return outer["result"]
        elif isinstance(outer, dict):
            return outer
    except json.JSONDecodeError:
        pass
    for candidate in list(candidates):
        candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.S))
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ExplorationPlanningError("Claude response did not contain a JSON exploration proposal")


class ClaudeExplorationClient:
    """Read-only Claude CLI adapter for visual garment reasoning."""

    def __init__(self, binary: str = "claude", timeout_s: int = 300):
        self.binary = binary
        self.timeout_s = timeout_s

    @staticmethod
    def _save_invocation_log(root: Path, payload: dict[str, Any], *, failed: bool = False) -> None:
        log_dir = root / "results" / "claude_exploration"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        name = f"{stamp}{'_failed' if failed else ''}.json"
        (log_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def invoke(
        self,
        image_paths: Iterable[str | Path],
        prompt: str,
        run_dir: Path,
    ) -> ClaudeExplorationResult:
        root = run_dir.resolve()
        if not root.is_dir():
            raise ExplorationPlanningError(f"run directory does not exist: {root}")
        safe_images: list[Path] = []
        for raw in image_paths:
            path = Path(raw)
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            if path != root and root not in path.parents:
                raise PermissionError("Claude images must stay inside the current run")
            if not path.is_file():
                raise FileNotFoundError(path)
            safe_images.append(path)
        if not safe_images:
            raise ExplorationPlanningError("at least one garment image is required")
        binary = shutil.which(self.binary) if Path(self.binary).name == self.binary else self.binary
        if binary is None:
            raise ExplorationPlanningError(f"Claude CLI not found: {self.binary}")
        image_text = "\n".join(f"- {path}" for path in safe_images)
        system_prompt = (
            "You are a cautious robotics garment analyst searching for a usable lifting "
            "anchor. Read the supplied garment photographs and return only one JSON "
            "object. You may inspect files but may not edit anything, execute commands, "
            "or control a robot. The action contract uses only move(x,y,z,yaw), "
            "open_gripper(), close_gripper(), and home(). Do not return Python, SDK "
            "calls, joint angles, invented measurements, fixed candidate lists, or "
            "mandatory semantic garment-part labels. Treat uncertain structures as "
            "possible boundaries/flaps/regions and use previous physical outcomes "
            "when they are supplied."
        )
        full_prompt = (
            f"{prompt}\n\nGarment images to inspect:\n{image_text}\n\n"
            "Coordinate grounding files are available under "
            "`workspace/perception_views/observation.json` and the listed "
            "`camera_*_coordinate_guide.json` files. Cyan Rxxx markers in coordinate "
            "overlay images are uniform calibrated references, not ranked grasp "
            "candidates. Choose the visual region yourself, then ground it with the "
            "nearest measured reference and state any remaining spatial uncertainty.\n\n"
            "The workspace also contains the calibrated full-resolution base-XYZ maps "
            "and fused point cloud when produced by perception; use them only as measured "
            "geometry, never as a perception-selected grasp recommendation.\n\n"
            "Return exactly these fields and no others: "
            "garment_observation (string), reveal_strategy (string; compatibility name, "
            "describe the anchor-search/test/laydown strategy here), confidence (number "
            "0..1), actions (non-empty list of {name,args}), expected_observation (string), "
            "safety_notes (non-empty list of strings), and optional skill_invocations "
            "(list of {name,reason}; use name=laydown only when you choose that skill). "
            "For move, args must contain exactly numeric x,y,z,yaw in millimetres/degrees. "
            "You choose the grasp region, all waypoint geometry, and whether this is a "
            "small anchor test or a larger maneuver. A cautious exploratory proposal may "
            "use one post-grasp move and then release; it does not have to immediately "
            "reveal large coverage. If you invoke laydown, use its procedural guidance "
            "but still emit explicit Claude-chosen move waypoints; the skill never supplies "
            "fixed coordinates. Always release before the action list ends. Keep the action "
            "list at or below 12."
        )
        command = [
            binary,
            "--print",
            full_prompt,
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
            system_prompt,
        ]
        completed = None
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
            self._save_invocation_log(
                root,
                {
                    "prompt": full_prompt,
                    "command": list(command),
                    "returncode": None,
                    "stdout": getattr(exc, "stdout", "") or "",
                    "stderr": getattr(exc, "stderr", "") or "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "created_at": _now(),
                },
                failed=True,
            )
            raise ExplorationPlanningError(f"Claude exploration invocation failed: {exc}") from exc
        if completed.returncode != 0:
            self._save_invocation_log(
                root,
                {
                    "prompt": full_prompt,
                    "command": list(command),
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "error": "non-zero Claude return code",
                    "created_at": _now(),
                },
                failed=True,
            )
            raise ExplorationPlanningError(
                f"Claude exited with {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        try:
            proposal = validate_exploration_payload(_json_from_claude_text(completed.stdout))
        except BaseException as exc:
            self._save_invocation_log(
                root,
                {
                    "prompt": full_prompt,
                    "command": list(command),
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "error": f"{type(exc).__name__}: {exc}",
                    "created_at": _now(),
                },
                failed=True,
            )
            raise
        result = ClaudeExplorationResult(
            prompt=full_prompt,
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            created_at=_now(),
            proposal=proposal,
        )
        self._save_invocation_log(
            root,
            {
                "prompt": result.prompt,
                "command": list(result.command),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "created_at": result.created_at,
                "proposal": result.proposal.as_dict(),
            },
        )
        return result


def exploration_source(proposal: ExplorationProposal) -> str:
    """Compile a validated Claude proposal to the existing restricted script format."""

    lines = ["def run():"]
    for action in proposal.actions:
        name, args = action["name"], action["args"]
        if name == "move":
            lines.append(
                f"    move({args['x']!r}, {args['y']!r}, {args['z']!r}, {args['yaw']!r})"
            )
        else:
            lines.append(f"    {name}()")
    return "\n".join(lines) + "\n"


def exploration_prompt(
    experiment: ExperimentConfig,
    robot: RobotConfig,
    *,
    objective: str = (
        "Take one cautious agent-chosen exploratory action toward autonomously "
        "discovering a usable garment lifting anchor."
    ),
    history: Iterable[dict[str, Any]] | None = None,
) -> str:
    """Build a bounded anchor-discovery prompt with explicit robot capabilities."""

    center = {
        "x_mm": experiment.cloth_center_x,
        "y_mm": experiment.cloth_center_y,
        # The legacy ``grasp_z`` slot stores the observed fused surface height;
        # it is not a commanded TCP grasp height.
        "surface_z_mm": experiment.grasp_z,
        # This is a validated observation/reference, not a mandatory grasp
        # target. Claude may select another visible or geometrically grounded
        # region.
        "center_is_reference_only": True,
        "perception_waypoint_authority": "Claude",
    }
    bounds = asdict(robot.boundaries)
    history_items = list(history or [])[-8:]
    history_text = (
        json.dumps(history_items, ensure_ascii=False, indent=2)
        if history_items
        else "No previous exploration outcomes are available."
    )
    return (
        f"Objective: {objective}\n"
        "Your primary objective is to discover a usable garment lifting anchor. A usable "
        "anchor is a grasp location from which lifting creates a useful hanging configuration "
        "that can subsequently be laid down to make the garment substantially easier to "
        "perceive or manipulate. You do not need to identify a sleeve, collar, hem, or any "
        "other semantic garment part before grasping. Do not assume the garment center, the "
        "highest point, a fold-convergence point, or the most occluded region is a good anchor. "
        "Use RGB images, garment masks, height-map heatmaps, height-gradient/occlusion edges, "
        "depth/3-D geometry, coordinate guides, and previous outcomes to decide where to "
        "interact. Claude chooses the region; perception only provides coordinate grounding. "
        "You may make a cautious lift, drag, repositioning, test grasp, release-and-retry, "
        "or another exploratory action. An action may be an experiment whose immediate purpose "
        "is only to test or improve understanding; it need not immediately reveal a large area. "
        "Do not redesign this into a candidate list or SELECT/PROBE/VERIFY state machine.\n\n"
        "The only available low-level calls are move(x,y,z,yaw), open_gripper(), "
        "close_gripper(), and home(). Every Cartesian waypoint must be emitted explicitly as "
        "a move(x,y,z,yaw) action and the runtime executes those values without replacing them "
        "with a template. You must choose the approach height, grasp height, lift/retreat "
        "height, lateral destination, release height, and yaw from the current observation. "
        "The validated center surface point below is a reference and coordinate sanity check, "
        "not a required grasp target. Coordinate-guide images/files in `perception_views` map "
        "camera pixels or visible geometry to calibrated robot-base XYZ; do not invent a "
        "pixel-to-base transform or silently treat an ungrounded visual guess as a safe point. "
        "If grounding is uncertain, state the uncertainty and choose a conservative target. "
        "Keep every waypoint inside the stated bounds with margin; runtime workspace, IK, and "
        "safety checks remain authoritative. Release before the action list ends.\n\n"
        "The garment-focused height map heatmap shows surface height above the fitted table "
        "plane in millimeters; brighter colors mean a larger garment/table height difference. "
        "`height_gradient_overlay` highlights internal height-gradient/occlusion edges and "
        "the global height map is scene context. In a severely crumpled/OOD state, full garment "
        "topology may be unobservable and semantic keypoints may be unreliable. Semantic identity "
        "is uncertain when evidence is weak; prefer phrases such as possible boundary, "
        "possible flap, uncertain structure, "
        "or candidate lifting region rather than hallucinating garment-part labels. State "
        "uncertainty explicitly.\n\n"
        "Previous exploration outcomes (use them to revise the strategy rather than restarting; "
        "referenced before/after image paths remain readable inside the run directory):\n"
        f"{history_text}\n\n"
        f"Validated center/height context: {json.dumps(center, ensure_ascii=False)}\n"
        f"Robot workspace bounds (mm): {json.dumps(bounds, ensure_ascii=False)}\n"
        f"Fixed orientation: roll={robot.orientation_roll_deg} deg, pitch={robot.orientation_pitch_deg} deg\n\n"
        "Available procedural skill guidance:\n"
        f"{skill_prompt()}\n"
    )


def perception_image_paths(result: dict[str, Any], result_path: Path) -> list[Path]:
    """Return RGB views plus the dense fused height preview for Claude."""

    paths: list[Path] = []
    for view in result.get("views", []):
        if not isinstance(view, dict) or "image" not in view:
            continue
        annotated = result_path.parent / str(view.get("annotated_image", ""))
        original = result_path.parent / str(view["image"])
        candidate = annotated if annotated.is_file() else original
        if candidate.is_file():
            paths.append(candidate.resolve())
        for key in (
            "height_map",
            "height_map_global",
            "height_map_boundary",
            "height_gradient_overlay",
            "coordinate_overlay",
            "depth_heatmap",
            "depth_heatmap_global",
            "depth_heatmap_boundary",
            "fold_edge_overlay",
        ):
            heatmap = result_path.parent / str(view.get(key, ""))
            if heatmap.is_file():
                resolved = heatmap.resolve()
                if resolved not in paths:
                    paths.append(resolved)
    if not paths:
        raise FileNotFoundError("perception result contains no saved garment images")
    fusion_preview = (
        result_path.parent
        / str(result.get("depth_fusion", {}).get("artifacts", {}).get("preview", ""))
    )
    if fusion_preview.is_file():
        resolved = fusion_preview.resolve()
        if resolved not in paths:
            paths.append(resolved)
    for key in ("heatmap", "boundary_overlay"):
        fused_map = result_path.parent / str(
            result.get("depth_fusion", {}).get("artifacts", {}).get(key, "")
        )
        if fused_map.is_file():
            resolved = fused_map.resolve()
            if resolved not in paths:
                paths.append(resolved)
    return paths


@dataclass
class _ExplorationState:
    result: dict[str, Any] | None = None
    result_path: Path | None = None
    proposal: ExplorationProposal | None = None
    proposal_source: str | None = None
    experiment_name: str | None = None
    preflight: Preflight | None = None
    controller: ControllerTrajectoryValidation | None = None
    animation_frames: list[AnimationFrame] = field(default_factory=list)
    approved_hash: str | None = None
    busy: bool = False
    playing: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)


def run_exploration_viewer(
    session: AgentSession,
    *,
    host: str = "127.0.0.1",
    port: int = 8081,
    enable_real: bool = False,
    perception_config_path: Path | None = None,
    urdf_path: Path | None = None,
    claude_client: ClaudeExplorationClient | None = None,
) -> int:
    """Start the standalone Claude free-exploration Viser application."""

    if enable_real and host not in {"127.0.0.1", "localhost", "::1"}:
        raise PermissionError("physical execution is allowed only on a loopback-only Viser server")
    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError as exc:
        raise RuntimeError(
            "Viser with URDF support is required; install it with: "
            "python -m pip install 'viser[urdf]>=1.0,<2'"
        ) from exc

    root = session.project_root
    perception_path = (
        perception_config_path
        or root / "config" / "perception.free_exploration.json"
    ).expanduser().resolve()
    urdf = (urdf_path or root / "assets" / "robots" / "xarm7" / "xarm7.urdf").resolve()
    robot = session.robot_config
    state = _ExplorationState()
    latest, latest_path = _load_latest_perception(session)
    state.result, state.result_path = latest, latest_path
    planner = claude_client or ClaudeExplorationClient()
    operation_lock = threading.Lock()
    animation_lock = threading.Lock()

    server = viser.ViserServer(host=host, port=port, label="Claude garment exploration")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/workspace/table", width=1.2, height=0.8, cell_size=0.05, section_size=0.25, position=(0.4, 0.0, 0.0))
    server.scene.add_frame("/robot_base", axes_length=0.15, axes_radius=0.006)
    server.scene.add_frame("/xarm", show_axes=False)
    robot_model = ViserUrdf(server, urdf, root_node_name="/xarm", load_meshes=True, load_collision_meshes=False)
    kinematics = XArm7Kinematics(urdf)
    home_cfg = np.concatenate([np.radians(np.asarray(robot.init_joints_deg, dtype=np.float64)), [0.0]])
    robot_model.update_cfg(home_cfg)

    status = server.gui.add_markdown("### Claude free exploration\n\nNo garment analysis loaded yet.")
    capture_button = server.gui.add_button("1. Capture + validate garment views", color="blue")
    analyze_button = server.gui.add_button("2. Ask Claude: search for a usable lifting anchor", disabled=state.result is None, color="blue")
    proposal_panel = server.gui.add_markdown("### Claude's garment view\n\nNo proposal yet.")
    validate_button = server.gui.add_button("3. Validate Claude action + build animation", disabled=True, color="blue")
    validation_panel = server.gui.add_markdown("No action validation has run yet.")
    slider = server.gui.add_slider("Animation frame", min=0, max=1, step=1, initial_value=0, disabled=True)
    animation_panel = server.gui.add_markdown("Animation is not available yet.")
    play_button = server.gui.add_button("4. Play proposed anchor exploration", disabled=True, color="green")
    pause_button = server.gui.add_button("Pause", disabled=True)
    reset_button = server.gui.add_button("Reset", disabled=True)
    loop_checkbox = server.gui.add_checkbox("Loop animation", initial_value=False)
    execute_button = server.gui.add_button("5. Confirm and execute Claude exploration", disabled=True, color="red")
    server.gui.add_markdown(
        "### Safety gate\n\nClaude is read-only. Physical execution stays disabled until the generated restricted program passes static preflight, workspace checks, controller IK, and animation review."
        + ("" if enable_real else "\n\nThis server is preview-only; restart with `--enable-real` for live authority.")
    )
    image_handles: list[Any] = []

    def clear_images() -> None:
        while image_handles:
            image_handles.pop().remove()

    def render_result() -> None:
        server.scene.remove_by_name("/perception")
        clear_images()
        if state.result is None or state.result_path is None:
            return
        for index, view in enumerate(state.result.get("views", [])):
            try:
                points, colors = _normalized_view_point_cloud(
                    view,
                    state.result_path.parent,
                    voxel_size_mm=5.0,
                    max_points=50000,
                )
                server.scene.add_point_cloud(
                    f"/perception/{view['label']}",
                    points=points,
                    colors=colors,
                    # 5 mm is the same physical scale used for voxel balancing.
                    point_size=0.005,
                    point_shape="circle",
                )
            except Exception:
                pass
            image = state.result_path.parent / str(view.get("annotated_image", view.get("image", "")))
            if not image.is_file():
                image = state.result_path.parent / str(view.get("image", ""))
            if image.is_file():
                from PIL import Image
                image_handles.append(server.gui.add_image(np.asarray(Image.open(image).convert("RGB")), label=f"Garment camera {view.get('label', index)}"))
            heatmap = state.result_path.parent / str(
                view.get(
                    "height_map_boundary",
                    view.get(
                        "height_map",
                        view.get("depth_heatmap_boundary", view.get("depth_heatmap", "")),
                    ),
                )
            )
            if heatmap.is_file():
                from PIL import Image
                image_handles.append(
                    server.gui.add_image(
                        np.asarray(Image.open(heatmap).convert("RGB")),
                label=f"Camera {view.get('label', index)} garment height-above-table heatmap + boundary",
                    )
                )
            fold_edges = state.result_path.parent / str(
                view.get(
                    "height_gradient_overlay",
                    view.get("fold_edge_overlay", ""),
                )
            )
            if fold_edges.is_file():
                from PIL import Image
                image_handles.append(
                    server.gui.add_image(
                        np.asarray(Image.open(fold_edges).convert("RGB")),
                        label=f"Camera {view.get('label', index)} internal height-gradient/occlusion edges",
                    )
                )
            coordinate_overlay = state.result_path.parent / str(
                view.get("coordinate_overlay", "")
            )
            if coordinate_overlay.is_file():
                from PIL import Image
                image_handles.append(
                    server.gui.add_image(
                        np.asarray(Image.open(coordinate_overlay).convert("RGB")),
                        label=(
                            f"Camera {view.get('label', index)} unranked robot-base "
                            "coordinate references"
                        ),
                    )
                )
        artifacts = state.result.get("depth_fusion", {}).get("artifacts", {})
        for key, label in (
            ("heatmap", "Fused garment height-above-table heatmap"),
            ("boundary_overlay", "Fused height map + garment boundary"),
            ("fold_edge_overlay", "Fused height-gradient/occlusion edges"),
        ):
            heatmap = state.result_path.parent / str(artifacts.get(key, ""))
            if heatmap.is_file():
                from PIL import Image
                image_handles.append(
                    server.gui.add_image(
                        np.asarray(Image.open(heatmap).convert("RGB")),
                        label=label,
                    )
                )

    def render_path(preflight: Preflight) -> None:
        server.scene.remove_by_name("/exploration_path")
        waypoints = path_waypoints_mm(preflight.actions, robot.init_pose_mm_deg)
        if len(waypoints) < 2:
            return
        points = np.stack([point for _, point in waypoints], axis=0).astype(np.float32) / 1000.0
        segments = np.stack([points[:-1], points[1:]], axis=1)
        colors = np.repeat(np.asarray([[220, 80, 50]], dtype=np.uint8)[None, :, :], len(segments), axis=0)
        server.scene.add_line_segments("/exploration_path/tcp", points=segments, colors=np.repeat(colors, 2, axis=1), line_width=4.0)

    def update_buttons() -> None:
        busy = state.busy
        analyze_button.disabled = busy or state.result is None
        validate_button.disabled = busy or state.proposal is None
        play_button.disabled = busy or not state.animation_frames
        pause_button.disabled = busy or not state.animation_frames
        reset_button.disabled = busy or not state.animation_frames
        execute_button.disabled = not (enable_real and not busy and state.approved_hash and state.controller and state.animation_frames)

    def set_busy(value: bool) -> None:
        state.busy = value
        capture_button.disabled = value
        update_buttons()

    def apply_frame(index: int) -> None:
        if not state.animation_frames:
            return
        index = max(0, min(int(index), len(state.animation_frames) - 1))
        frame = state.animation_frames[index]
        robot_model.update_cfg(frame.configuration_rad)
        animation_panel.content = f"### Animation\n\nFrame `{index + 1}/{len(state.animation_frames)}` · `{frame.label}` · action `{frame.action_index + 1}`"

    @capture_button.on_click
    def _(event: Any) -> None:
        if not operation_lock.acquire(blocking=False):
            return
        set_busy(True)
        status.content = "### Capturing garment views\n\nRealSense capture and existing center/depth validation are running."
        try:
            config = PerceptionConfig.load(root, perception_path)
            frames = capture_two_view_rgbd(config)
            result = session.locate_cloth_center(config, frames=frames)
            state.result, state.result_path = _load_latest_perception(session)
            if state.result is None or state.result_path is None:
                raise RuntimeError("center validation completed without a saved result")
            state.proposal = None
            state.proposal_source = None
            state.controller = None
            state.animation_frames = []
            state.approved_hash = None
            render_result()
            status.content = f"### Views ready\n\nValidated garment center reference: `{result['center_base_mm']}`. Ask Claude to search for a lifting anchor."
        except Exception as exc:
            status.content = f"### Capture blocked\n\n`{type(exc).__name__}: {exc}`"
        finally:
            set_busy(False)
            operation_lock.release()

    @analyze_button.on_click
    def _(event: Any) -> None:
        if state.result is None or state.result_path is None or not operation_lock.acquire(blocking=False):
            return
        set_busy(True)
        status.content = "### Claude is thinking\n\nInspecting geometry and previous outcomes for a cautious anchor-discovery action."
        try:
            images = perception_image_paths(state.result, state.result_path)
            prompt = exploration_prompt(
                session.experiment_config,
                robot,
                history=state.history,
            )
            response = planner.invoke(images, prompt, session.run_dir)
            state.proposal = response.proposal
            state.proposal_source = exploration_source(response.proposal)
            state.controller = None
            state.animation_frames = []
            state.approved_hash = None
            proposal_panel.content = _proposal_markdown(response.proposal, state.proposal_source)
            validate_button.disabled = False
            status.content = "### Claude proposal ready\n\nReview the anchor hypothesis and exploratory path, then validate it."
        except Exception as exc:
            state.proposal = None
            state.proposal_source = None
            proposal_panel.content = f"### Claude proposal blocked\n\n`{type(exc).__name__}: {exc}`"
            status.content = "### Claude proposal blocked\n\nThe structured contract failed; no action was accepted."
        finally:
            set_busy(False)
            operation_lock.release()

    @validate_button.on_click
    def _(event: Any) -> None:
        if state.proposal is None or state.proposal_source is None or not operation_lock.acquire(blocking=False):
            return
        set_busy(True)
        status.content = "### Validating Claude exploration\n\nRunning static preflight, controller IK, and URDF animation."
        path = session.workspace / "_claude_exploration_preview.py"
        try:
            path.write_text(state.proposal_source, encoding="utf-8")
            preflight = session.runner.preflight(path.name)
            state.preflight = preflight
            validation_panel.content = _preflight_markdown(preflight)
            render_path(preflight)
            if preflight.error:
                raise ExperimentValidationError(preflight.error)
            controller = validate_controller_trajectory(robot, preflight.actions)
            frames = kinematics.build_animation(
                preflight.actions,
                robot.init_joints_deg,
                robot.orientation_roll_deg,
                robot.orientation_pitch_deg,
                joint_targets_rad=controller.joint_targets_rad,
            )
            state.controller = controller
            state.animation_frames = frames
            state.approved_hash = _source_hash(preflight.source)
            slider.max = max(1, len(frames) - 1)
            slider.value = 0
            slider.disabled = False
            apply_frame(0)
            status.content = f"### Claude exploration validated\n\n`{len(frames)}` animation frames are ready. Physical execution remains an explicit opt-in."
        except Exception as exc:
            state.controller = None
            state.animation_frames = []
            state.approved_hash = None
            diagnosis = _controller_ik_failure_message(exc)
            validation_panel.content = (
                f"### Hard validation failure\n\n`{type(exc).__name__}: {diagnosis}`"
            )
            status.content = (
                "### Claude exploration blocked\n\n"
                "The controller rejected at least one pose during read-only IK. "
                "Change the proposal or scene; the rejected path cannot execute."
            )
        finally:
            if path.is_file():
                path.unlink()
            set_busy(False)
            operation_lock.release()

    @slider.on_update
    def _(_: Any) -> None:
        apply_frame(int(slider.value))

    @play_button.on_click
    def _(_: Any) -> None:
        if not state.animation_frames or state.playing:
            return
        state.playing = True

        def play() -> None:
            with animation_lock:
                try:
                    index = int(slider.value)
                    while state.playing and state.animation_frames:
                        slider.value = index
                        apply_frame(index)
                        index += 1
                        if index >= len(state.animation_frames):
                            if loop_checkbox.value:
                                index = 0
                            else:
                                break
                        time.sleep(1.0 / 12.0)
                finally:
                    state.playing = False

        threading.Thread(target=play, daemon=True, name="claude-exploration-animation").start()

    @pause_button.on_click
    def _(_: Any) -> None:
        state.playing = False

    @reset_button.on_click
    def _(_: Any) -> None:
        state.playing = False
        slider.value = 0
        apply_frame(0)

    @execute_button.on_click
    def _(_: Any) -> None:
        if execute_button.disabled or not operation_lock.acquire(blocking=False):
            return
        set_busy(True)
        state.playing = False
        session.last_return_home_outcome = None
        try:
            path = session.workspace / "experiment_999_claude_exploration.py"
            if state.proposal_source is None:
                raise RuntimeError("no Claude proposal is approved")
            path.write_text(state.proposal_source, encoding="utf-8")
            current = session.runner.preflight(path.name)
            if current.error:
                raise RuntimeError(current.error)
            if state.approved_hash != _source_hash(current.source):
                raise PermissionError("Claude proposal changed after validation; validate again")
            result = session.run_experiment(
                path.name,
                real=True,
                confirmed=True,
                single_view_confirmed=(
                    json.loads((session.run_dir / "run_metadata.json").read_text(encoding="utf-8")).get(
                        "last_perception_mode"
                    )
                    == "single_camera_rgbd"
                ),
                notes="Confirmed standalone Claude free-exploration Viser action.",
            )
            prior_images: list[str] = []
            if state.result is not None and state.result_path is not None:
                for image_path in perception_image_paths(state.result, state.result_path):
                    try:
                        prior_images.append(str(image_path.relative_to(session.run_dir)))
                    except ValueError:
                        prior_images.append(str(image_path))
            state.history.append(
                {
                    "attempt": len(state.history) + 1,
                    "proposal": state.proposal.as_dict() if state.proposal else None,
                    "execution_completed": bool(result.get("execution_completed")),
                    "robot_errors": result.get("robot_errors", []),
                    "before_images": prior_images,
                    "note": (
                        "Capture a new observation before the next Claude turn so the "
                        "physical outcome can be judged visually."
                    ),
                }
            )
            status.content = f"### Claude exploration finished\n\nCompleted: `{result['execution_completed']}` · errors: `{result['robot_errors']}`"
        except Exception as exc:
            status.content = f"### Claude exploration stopped/failed\n\n`{type(exc).__name__}: {exc}`"
        finally:
            home_outcome = session.last_return_home_outcome
            if home_outcome is not None:
                if home_outcome.get("completed"):
                    robot_model.update_cfg(home_cfg)
                status.content += (
                    "\n\n### Mandatory return Home\n\n"
                    f"- attempted: `{home_outcome.get('attempted')}`\n"
                    f"- completed: `{home_outcome.get('completed')}`\n"
                    f"- error: `{home_outcome.get('error')}`\n"
                    f"- result: `{home_outcome.get('result_path')}`"
                )
            path = session.workspace / "experiment_999_claude_exploration.py"
            if path.is_file():
                path.unlink()
            state.approved_hash = None
            state.controller = None
            set_busy(False)
            operation_lock.release()

    @server.on_client_connect
    def _(client: Any) -> None:
        client.camera.position = (1.3, -0.9, 0.9)
        client.camera.look_at = (0.62, -0.07, 0.1)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    render_result()
    if state.result is not None:
        status.content = "### Garment views loaded\n\nAsk Claude to inspect the garment and search for a usable lifting anchor."
    update_buttons()
    print(f"Viser Claude free-exploration console: http://{host}:{port}")
    print(f"Run workspace: {session.workspace}")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Claude exploration console stopped; no additional robot command was sent.")
    finally:
        state.playing = False
        server.stop()
    return 0


def _proposal_markdown(proposal: ExplorationProposal, source: str) -> str:
    rows = ["| # | action | arguments |", "|---:|---|---|"]
    for index, action in enumerate(proposal.actions, start=1):
        rows.append(f"| {index} | `{action['name']}` | `{json.dumps(action['args'], ensure_ascii=False)}` |")
    safe_source = source.replace("```", "` ` `")
    skills = (
        "\n".join(
            f"- `{skill['name']}`: {skill['reason']}"
            for skill in proposal.skill_invocations
        )
        if proposal.skill_invocations
        else "- none"
    )
    return (
        "### Claude's garment view\n\n"
        f"**Observation:** {proposal.garment_observation}\n\n"
        f"**Anchor-search strategy:** {proposal.reveal_strategy}\n\n"
        f"**Confidence:** `{proposal.confidence:.2f}`\n\n"
        f"**Expected observation:** {proposal.expected_observation}\n\n"
        "**Invoked procedural skills:**\n\n"
        + skills
        + "\n\n**Safety notes:**\n\n"
        + "\n".join(f"- {note}" for note in proposal.safety_notes)
        + "\n\n**Proposed restricted actions:**\n\n"
        + "\n".join(rows)
        + f"\n\n```python\n{safe_source}\n```"
    )


def _load_or_create_session(root: Path, run_dir: Path | None, run_id: str | None, robot_config: Path | None) -> AgentSession:
    root = root.resolve()
    runs_root = (root / "runs").resolve()
    if run_dir is not None:
        run = run_dir.resolve()
        metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
        saved_robot = run / "workspace" / "robot_config.json"
        robot = RobotConfig.load(root, saved_robot if saved_robot.is_file() else robot_config)
        saved_experiment = run / "workspace" / "experiment_config.json"
        values = json.loads(saved_experiment.read_text(encoding="utf-8")) if saved_experiment.is_file() else metadata["experiment_config"]
        return AgentSession(root, run, robot, ExperimentConfig.from_mapping(values, allow_deferred=True))
    if run_id is not None:
        # ``--run-id`` is convenient for both first launch and reopening a
        # console.  AgentSession.create intentionally uses exist_ok=False, so
        # resolve an existing run through the read-only loader instead of
        # trying to create its workspace a second time.
        if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
            raise ValueError("run-id must be one simple directory name")
        existing = runs_root / run_id
        if existing.exists():
            if not existing.is_dir():
                raise FileExistsError(f"run-id path is not a directory: {existing}")
            return _load_or_create_session(root, existing, None, robot_config)
    robot = RobotConfig.load(root, robot_config)
    return AgentSession.create(root, "Claude free garment exploration", robot, ExperimentConfig(), run_id=run_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-dir")
    parser.add_argument("--run-id")
    parser.add_argument("--robot-config")
    parser.add_argument("--perception-config")
    parser.add_argument("--urdf")
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--enable-real", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    run = Path(args.run_dir).resolve() if args.run_dir else None
    robot_config = Path(args.robot_config).resolve() if args.robot_config else None
    perception = Path(args.perception_config).resolve() if args.perception_config else None
    urdf = Path(args.urdf).resolve() if args.urdf else None
    session = _load_or_create_session(root, run, args.run_id, robot_config)
    return run_exploration_viewer(
        session,
        host=args.host,
        port=args.port,
        enable_real=args.enable_real,
        perception_config_path=perception,
        urdf_path=urdf,
        claude_client=ClaudeExplorationClient(args.claude_binary),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
