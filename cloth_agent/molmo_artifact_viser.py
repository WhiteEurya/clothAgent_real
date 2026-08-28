"""Read-only Viser view of saved Molmo/Claude perception artifacts.

This module is deliberately separate from the interactive robot consoles.  It
only follows files written under a run's ``results``/``workspace`` tree.  It
shows a static xArm7 URDF, the measured clouds/images, the validated TCP path,
and Claude's structured proposal/evaluation summaries. It does not open
RealSense, connect to xArm, load Molmo/Claude, or expose action buttons. The
process is therefore safe to run beside the headless CLI while the CLI is
doing physical execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .config import RobotConfig
from .molmo_artifact_viewer import _latest_iteration_dir, discover_output_dir
from .viewer import _load_fused_point_cloud, _view_point_cloud, path_waypoints_mm


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def discover_perception_result(output_dir: Path) -> Path | None:
    """Return the newest saved perception result for one CLI output directory."""

    iteration = _latest_iteration_dir(output_dir.resolve())
    if iteration is None:
        return None
    record = _load_json(iteration / "result.json")
    saved = record.get("saved_perception_result")
    if not isinstance(saved, str) or not saved:
        return None
    path = Path(saved).expanduser().resolve()
    return path if path.is_file() else None


def _sample_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    max_points: int = 80_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound browser payload size without changing the measured coordinates."""

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if len(points) <= max_points:
        return points, colors
    stride = int(np.ceil(len(points) / max_points))
    return points[::stride], colors[::stride]


def _image(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError):
        return None


def _short(value: Any, limit: int = 1400) -> str:
    """Keep the Viser markdown readable while preserving the model's summary."""

    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _path_from_result(result_dir: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = result_dir / path
    return path.resolve()


def _jsonl_tail(path: Path, limit: int = 8) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines[-limit:]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _skill_markdown(
    record: dict[str, Any],
    iteration_dir: Path,
    skills_root: Path | None,
) -> str:
    """Render skill proposals, reviews, duplicate checks, and activations."""

    proposal = record.get("proposal") if isinstance(record.get("proposal"), dict) else {}
    invocations = proposal.get("skill_invocations", []) if isinstance(proposal, dict) else []
    evaluation = record.get("evaluation") if isinstance(record.get("evaluation"), dict) else {}
    skill_update = evaluation.get("skill_update") if isinstance(evaluation, dict) else None
    review = record.get("skill_review") if isinstance(record.get("skill_review"), dict) else {}
    if not review:
        review = _load_json(iteration_dir / "skill_review.json")

    lines = ["### Skill lifecycle and operations", ""]
    if invocations:
        lines.extend(["**Skills used in this proposal**", ""])
        for item in invocations:
            if isinstance(item, dict):
                lines.append(
                    f"- `{item.get('name', '?')}`: {_short(item.get('reason'), 500)}"
                )
        lines.append("")
    else:
        lines.extend(["- skills used in this proposal: `none recorded`", ""])

    if isinstance(skill_update, dict):
        lines.extend(
            [
                "**Claude skill operation proposal**",
                "",
                f"- operation/name: `{skill_update.get('operation', '?')}` / `{skill_update.get('name', '?')}`",
                f"- base skill: `{skill_update.get('base_skill', '—')}`",
                f"- confidence: `{skill_update.get('confidence', '—')}`",
                f"- purpose: {_short(skill_update.get('purpose'), 700)}",
                f"- rationale: {_short(skill_update.get('rationale'), 900)}",
                f"- guidance: {_short(skill_update.get('guidance'), 1200)}",
                f"- evidence: `{_short(skill_update.get('evidence'), 1000)}`",
                "",
            ]
        )
    else:
        lines.extend(["- skill create/modify proposal in this iteration: `none`", ""])

    if review:
        lines.extend(
            [
                "**Independent skill review**",
                "",
                f"- status: `{review.get('status', '—')}`",
                f"- approved/activated: `{review.get('approved', '—')}`",
                f"- reason: {_short(review.get('reason'), 1100)}",
                f"- similar skills checked: `{_short(review.get('similar_skills', []), 1000)}`",
                f"- activated skill: `{_short(review.get('activated_skill', '—'), 1200)}`",
                "",
            ]
        )
    else:
        lines.extend(["- independent skill review in this iteration: `not run`", ""])

    error_recovery = record.get("preexecution_error_recovery")
    if isinstance(error_recovery, dict):
        recovery_proposal = error_recovery.get("skill_proposal", {})
        recovery_review = error_recovery.get("skill_review", {})
        lines.extend(
            [
                "**Error-recovery skill proposal**",
                "",
                f"- error: `{_short(error_recovery.get('error_feedback', {}), 1200)}`",
                f"- operation/name: `{recovery_proposal.get('operation', '—') if isinstance(recovery_proposal, dict) else '—'}` / `{recovery_proposal.get('name', '—') if isinstance(recovery_proposal, dict) else '—'}`",
                f"- guidance: {_short(recovery_proposal.get('guidance', '—') if isinstance(recovery_proposal, dict) else '—', 1200)}",
                f"- review: `{recovery_review.get('status', '—') if isinstance(recovery_review, dict) else '—'}`",
                f"- activated: `{recovery_review.get('approved', '—') if isinstance(recovery_review, dict) else '—'}`",
                f"- reason: {_short(recovery_review.get('reason', '—') if isinstance(recovery_review, dict) else '—', 1000)}",
                "",
            ]
        )

    evaluation_recovery = record.get("evaluation_error_recovery")
    if isinstance(evaluation_recovery, dict):
        recovery_proposal = evaluation_recovery.get("skill_proposal", {})
        recovery_review = evaluation_recovery.get("skill_review", {})
        lines.extend(
            [
                "**Evaluation-timeout recovery skill**",
                "",
                f"- attempts: `{evaluation_recovery.get('successful_attempt', '—')}`",
                f"- operation/name: `{recovery_proposal.get('operation', '—') if isinstance(recovery_proposal, dict) else '—'}` / `{recovery_proposal.get('name', '—') if isinstance(recovery_proposal, dict) else '—'}`",
                f"- review: `{recovery_review.get('status', '—') if isinstance(recovery_review, dict) else '—'}`",
                f"- activated: `{recovery_review.get('approved', '—') if isinstance(recovery_review, dict) else '—'}`",
                f"- no robot/camera repeat: `{not evaluation_recovery.get('robot_command_repeated', True) and not evaluation_recovery.get('camera_command_repeated', True)}`",
                "",
            ]
        )

    if skills_root is not None:
        reviews = _jsonl_tail(skills_root / "reviews.jsonl", limit=8)
        approved = _load_json(skills_root / "approved.json")
        approved_patches = _load_json(skills_root / "approved_patches.json")
        if reviews:
            lines.extend(["**Recent skill audit log**", ""])
            for entry in reviews:
                proposal_item = entry.get("proposal", {})
                review_item = entry.get("review", {})
                if not isinstance(proposal_item, dict) or not isinstance(review_item, dict):
                    continue
                lines.append(
                    f"- `{entry.get('created_at', '?')}` "
                    f"`{proposal_item.get('operation', '?')}` `{proposal_item.get('name', '?')}` "
                    f"→ `{review_item.get('status', '?')}`: "
                    f"{_short(review_item.get('reason'), 500)}"
                )
            lines.append("")
        if isinstance(approved, dict):
            active = approved.get("skills", [])
            if isinstance(active, list):
                lines.append(
                    "**Currently approved skill library**\n\n"
                    + ", ".join(
                        f"`{item.get('name', '?')}@v{item.get('version', '?')}`"
                        for item in active
                        if isinstance(item, dict)
                    )
                )
        if isinstance(approved_patches, dict):
            patches = approved_patches.get("patches", [])
            if isinstance(patches, list) and patches:
                lines.extend(["", "**Approved system skill patches**", ""])
                for item in patches:
                    if not isinstance(item, dict):
                        continue
                    lines.append(
                        f"- `{item.get('base_skill', '?')}` from "
                        f"`{item.get('patch_file', '?')}`; reviewer "
                        f"`{item.get('reviewer', '?')}` at "
                        f"`{item.get('approved_at', '?')}`"
                    )
    return "\n".join(lines)


def _claude_output_markdown(
    record: dict[str, Any],
    iteration_dir: Path,
    run_root: Path | None,
) -> str:
    """Show explicit Claude process output and runtime intermediate variables."""

    lines = ["### Claude calls and intermediate variables", ""]
    attempts = record.get("claude_global_attempts", [])
    standalone_attempts = record.get("planning_attempts", [])
    if not attempts and isinstance(standalone_attempts, list):
        attempts = standalone_attempts
    if isinstance(attempts, list) and attempts:
        lines.extend(["**Global planning call**", ""])
        for index, attempt in enumerate(attempts, start=1):
            if not isinstance(attempt, dict):
                continue
            lines.extend(
                [
                    f"- attempt `{index}` return code: `{attempt.get('returncode', '—')}`",
                    f"- command: `{_short(attempt.get('command'), 1200)}`",
                    f"- prompt: `{_short(attempt.get('prompt'), 1600)}`",
                    "- stdout:",
                    "```text",
                    _short(attempt.get("stdout"), 6000) or "(empty)",
                    "```",
                    "- stderr:",
                    "```text",
                    _short(attempt.get("stderr"), 2500) or "(empty)",
                    "```",
                ]
            )
            if attempt.get("proposal") is not None:
                lines.extend(
                    [
                        "- parsed proposal:",
                        "```json",
                        _short(json.dumps(attempt.get("proposal"), ensure_ascii=False, indent=2), 6000),
                        "```",
                    ]
                )
        lines.append("")
    else:
        lines.extend(["- planning call: `no completed call recorded yet`", ""])

    rejections = record.get("global_planning_rejections", [])
    if not rejections and isinstance(standalone_attempts, list):
        rejections = [
            {
                "attempt": item.get("attempt", "?"),
                "error": item.get("error", ""),
                "feedback_target": "standalone fold planner",
                "physical_command_sent": False,
            }
            for item in standalone_attempts
            if isinstance(item, dict) and item.get("status") == "REJECTED_BEFORE_EXECUTION"
        ]
    if isinstance(rejections, list) and rejections:
        lines.extend(["**Rejected planning attempts**", ""])
        for item in rejections:
            if isinstance(item, dict):
                lines.append(
                    f"- attempt `{item.get('attempt', '?')}`: {_short(item.get('error'), 1200)}; "
                    f"feedback target=`{item.get('feedback_target', '—')}`; "
                    f"physical command sent=`{item.get('physical_command_sent', '—')}`"
                )
        lines.append("")

    lines.extend(
        [
            "**Runtime intermediate variables**",
            "",
            f"- objective: `{_short(record.get('objective'), 1200)}`",
            f"- candidate policy: `{record.get('candidate_policy', '—')}`",
            f"- before images: `{_short(record.get('before_images', []), 1400)}`",
            f"- after images: `{_short(record.get('after_images', []), 1400)}`",
            f"- grounding: `{_short(record.get('global_grounding', record.get('grounding', {})), 2200)}`",
            f"- preflight: `{_short(record.get('preflight', {}), 2200)}`",
            f"- controller IK: `{_short(record.get('controller_ik', {}), 1800)}`",
            f"- planning attempts: `{_short(record.get('planning_attempts', []), 2600)}`",
            f"- error feedback to Claude: `{_short(record.get('error_feedback_to_claude', {}), 2200)}`",
            f"- pre-execution error recovery: `{_short(record.get('preexecution_error_recovery', {}), 2600)}`",
            f"- evaluation error recovery: `{_short(record.get('evaluation_error_recovery', {}), 2200)}`",
            f"- stage timestamps: `{_short(record.get('stage_timestamps', {}), 1600)}`",
            "",
        ]
    )

    if run_root is not None:
        evaluation_logs = sorted(
            (run_root / "results" / "claude_auto").glob("*_evaluation*.json"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if evaluation_logs:
            raw = _load_json(evaluation_logs[-1])
            lines.extend(
                [
                    "**Latest evaluator process output**",
                    "",
                    f"- artifact: `{evaluation_logs[-1]}`",
                    f"- return code: `{raw.get('returncode', '—')}`",
                    "- stdout:",
                    "```text",
                    _short(raw.get("stdout"), 6000) or "(empty)",
                    "```",
                    "- stderr:",
                    "```text",
                    _short(raw.get("stderr"), 2500) or "(empty)",
                    "```",
                ]
            )
    return "\n".join(lines)


def _iteration_snapshot(
    output_dir: Path,
) -> tuple[Path | None, dict[str, Any], Path | None]:
    """Return latest iteration checkpoint, its record, and saved perception."""

    iteration_dir = _latest_iteration_dir(output_dir.resolve())
    if iteration_dir is None:
        return None, {}, None
    record_path = iteration_dir / "result.json"
    record = _load_json(record_path)
    saved = record.get("saved_perception_result")
    result_path = (
        Path(saved).expanduser().resolve()
        if isinstance(saved, str) and saved
        else None
    )
    if result_path is not None and not result_path.is_file():
        result_path = None
    return iteration_dir, record, result_path


def _claude_markdown(
    record: dict[str, Any],
    result_dir: Path,
) -> tuple[str, str, str, Path | None]:
    """Build safe, structured summaries instead of exposing hidden chain-of-thought."""

    proposal = record.get("proposal")
    if not isinstance(proposal, dict):
        proposal_path = _path_from_result(result_dir, "proposal.json")
        proposal = _load_json(proposal_path) if proposal_path else {}
    selected = proposal.get("selected_grasp", {}) if isinstance(proposal, dict) else {}
    grounding = record.get("global_grounding", record.get("grounding", {}))
    measurement = grounding.get("measurement", {}) if isinstance(grounding, dict) else {}
    observation = _short(proposal.get("garment_observation"), 1800)
    strategy = _short(proposal.get("reveal_strategy"), 1800)
    expected = _short(proposal.get("expected_observation"), 1400)
    confidence = proposal.get("confidence")
    skills = proposal.get("skill_invocations", [])
    skill_lines = [
        f"- `{item.get('name', '?')}`: {_short(item.get('reason'), 360)}"
        for item in skills
        if isinstance(item, dict)
    ]
    grasp_lines = [
        f"- Camera/pixel: `{selected.get('camera', '?')}` / `{selected.get('pixel_xy', '?')}`",
        f"- measured base XYZ (mm): `{measurement.get('base_xyz_median_mm', '?')}`",
        f"- height above table (mm): `{measurement.get('height_above_table_median_mm', '?')}`",
        f"- commanded XY correction (mm): `{grounding.get('xy_correction_mm', '?') if isinstance(grounding, dict) else '?'}`",
    ]
    if isinstance(confidence, (int, float)):
        grasp_lines.append(f"- Claude confidence: `{float(confidence):.2f}`")
    claude = (
        "### Claude structured scene summary\n\n"
        f"**Observation**\n\n{observation or 'No completed Claude proposal yet.'}\n\n"
        f"**Strategy**\n\n{strategy or '—'}\n\n"
        f"**Expected next observation**\n\n{expected or '—'}\n\n"
        "**Selected interaction and grounding**\n\n"
        + "\n".join(grasp_lines)
        + "\n\n**Approved procedural skills**\n\n"
        + ("\n".join(skill_lines) if skill_lines else "- none")
    )

    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        evaluation_path = _path_from_result(result_dir, "evaluation.json")
        evaluation = _load_json(evaluation_path) if evaluation_path else {}
    task = evaluation.get("task_progress", {}) if isinstance(evaluation, dict) else {}
    if isinstance(evaluation, dict) and not task and "status" in evaluation:
        task = {
            "status": evaluation.get("status"),
            "confidence": evaluation.get("confidence"),
            "outcome": (
                f"alignment={evaluation.get('stack_alignment', '—')}; "
                f"flatness={evaluation.get('flatness', '—')}; "
                f"protruding_parts={evaluation.get('protruding_parts', '—')}"
            ),
        }
    next_experiment = evaluation.get("next_experiment", {}) if isinstance(evaluation, dict) else {}
    evaluation_text = (
        "### Claude evaluation / learning result\n\n"
        f"- status: `{task.get('status', task.get('outcome', 'not available')) if isinstance(task, dict) else 'not available'}`\n"
        f"- confidence: `{task.get('confidence', '—') if isinstance(task, dict) else '—'}`\n"
        f"- earliest failure stage: `{evaluation.get('earliest_failure_stage', '—') if isinstance(evaluation, dict) else '—'}`\n"
        f"- stop: `{evaluation.get('stop', '—') if isinstance(evaluation, dict) else '—'}`\n"
        f"- reason: {_short(evaluation.get('reason', '—') if isinstance(evaluation, dict) else '—', 900)}\n\n"
        f"- keep: `{next_experiment.get('keep', '—') if isinstance(next_experiment, dict) else '—'}`\n"
        f"- change: `{next_experiment.get('change', '—') if isinstance(next_experiment, dict) else '—'}`"
    )

    error = record.get("error")
    recovery = record.get("recovery")
    rejections = record.get("global_planning_rejections", [])
    latest_rejection = rejections[-1] if isinstance(rejections, list) and rejections else {}
    execution = record.get("execution")
    execution_lines = ""
    if isinstance(execution, dict):
        execution_lines = (
            f"\n- physical execution completed: `{execution.get('execution_completed', '—')}`"
            f"\n- robot errors: `{_short(execution.get('robot_errors', []), 700)}`"
        )
    recovery_text = (
        "### Runtime state\n\n"
        f"- iteration: `{record.get('iteration', '—')}`\n"
        f"- status: `{record.get('status', '—')}`\n"
        f"- completed stage: `{record.get('last_completed_stage', '—')}`\n"
        f"- error: `{_short(error, 900) if error else 'none'}`\n"
        f"- recovery: `{_short(recovery, 900) if recovery else 'not active'}`\n"
        f"- error feedback to Claude: `{_short(record.get('error_feedback_to_claude', {}), 1200)}`\n"
        f"- recovery skill review: `{_short(record.get('recovery_skill_review', {}), 1000)}`\n"
        f"- evaluation recovery skill review: `{_short(record.get('evaluation_recovery_skill_review', {}), 1000)}`\n"
        f"- latest planning rejection: `{_short(latest_rejection.get('error', 'none'), 900) if isinstance(latest_rejection, dict) else 'none'}`"
        f"{execution_lines}"
    )
    selected_overlay = _path_from_result(
        result_dir,
        (record.get("artifacts") or {}).get("claude_selected_pixel")
        if isinstance(record.get("artifacts"), dict)
        else None,
    )
    return claude, evaluation_text, recovery_text, selected_overlay


class _ArtifactViserState:
    def __init__(
        self,
        server: Any,
        robot: RobotConfig,
        urdf_path: Path,
        *,
        run_root: Path | None = None,
        skills_root: Path | None = None,
    ) -> None:
        self.server = server
        self.robot = robot
        self.urdf_path = urdf_path
        self.run_root = run_root.resolve() if run_root is not None else None
        self.skills_root = skills_root.resolve() if skills_root is not None else None
        self.lock = threading.Lock()
        self.loaded_perception: str | None = None
        self.loaded_iteration: tuple[str, int] | None = None
        self.image_handles: list[Any] = []
        self.cloud_handles: list[Any] = []
        self.path_handles: list[Any] = []
        self.selected_overlay_path: str | None = None
        self.status = server.gui.add_markdown(
            "### Waiting for perception artifacts\n\n"
            "This Viser process is read-only and follows the latest CLI iteration."
        )
        self.claude_panel = server.gui.add_markdown(
            "### Claude structured scene summary\n\nWaiting for the first completed proposal."
        )
        self.evaluation_panel = server.gui.add_markdown(
            "### Claude evaluation / learning result\n\nWaiting for a before/after evaluation."
        )
        self.skill_panel = server.gui.add_markdown(
            "### Skill lifecycle and operations\n\nWaiting for skill audit data."
        )
        self.output_panel = server.gui.add_markdown(
            "### Claude calls and intermediate variables\n\nWaiting for an explicit Claude process output."
        )
        self.runtime_panel = server.gui.add_markdown(
            "### Runtime state\n\nWaiting for the CLI checkpoint."
        )
        self.path_panel = server.gui.add_markdown(
            "### Robot path\n\nNo validated path is available yet."
        )
        self.robot_model: Any | None = None
        self._init_robot_model()

    def _init_robot_model(self) -> None:
        try:
            from viser.extras import ViserUrdf

            self.server.scene.add_frame("/robot_base", axes_length=0.15, axes_radius=0.006)
            self.server.scene.add_frame("/xarm", show_axes=False)
            self.robot_model = ViserUrdf(
                self.server,
                self.urdf_path,
                root_node_name="/xarm",
                load_meshes=True,
                load_collision_meshes=False,
            )
            home_cfg = np.concatenate(
                [np.radians(np.asarray(self.robot.init_joints_deg, dtype=np.float64)), [0.0]]
            )
            self.robot_model.update_cfg(home_cfg)
        except Exception as exc:
            self.runtime_panel.content = (
                "### Runtime state\n\n"
                f"Robot model unavailable: `{type(exc).__name__}: {exc}`"
            )

    def clear(self) -> None:
        for handle in self.image_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.image_handles.clear()
        try:
            self.server.scene.remove_by_name("/perception")
        except Exception:
            pass
        self.cloud_handles.clear()
        for handle in self.path_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.path_handles.clear()
        try:
            self.server.scene.remove_by_name("/plan")
        except Exception:
            pass

    def render_perception(self, result_path: Path) -> bool:
        result = _load_json(result_path)
        if not result:
            return False
        result_dir = result_path.parent
        for handle in self.image_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.image_handles.clear()
        self.selected_overlay_path = None
        try:
            self.server.scene.remove_by_name("/perception")
        except Exception:
            pass
        self.cloud_handles.clear()
        views = result.get("views", [])
        loaded_clouds = 0
        for view in views:
            if not isinstance(view, dict):
                continue
            label = str(view.get("label", "camera")).upper()
            try:
                points, colors = _view_point_cloud(view, result_dir, stride=4)
                points, colors = _sample_cloud(points, colors)
            except Exception:
                points = np.empty((0, 3), dtype=np.float32)
                colors = np.empty((0, 3), dtype=np.uint8)
            if len(points):
                self.cloud_handles.append(
                    self.server.scene.add_point_cloud(
                        f"/perception/camera_{label}",
                        points=points,
                        colors=colors,
                        point_size=0.004,
                        point_shape="circle",
                    )
                )
                loaded_clouds += len(points)
            for key, title in (
                ("image", f"Camera {label} RGB"),
                ("height_map", f"Camera {label} height map (softmax heatmap)"),
                ("height_map_boundary", f"Camera {label} height map + boundary"),
                ("height_gradient_overlay", f"Camera {label} height-gradient edges"),
                ("coordinate_overlay", f"Camera {label} coordinate overlay"),
            ):
                raw = view.get(key)
                path = result_dir / str(raw) if isinstance(raw, str) and raw else None
                if path is None:
                    continue
                image = _image(path)
                if image is not None:
                    self.image_handles.append(self.server.gui.add_image(image, label=title))

        artifacts = result.get("depth_fusion", {}).get("artifacts", {})
        if isinstance(artifacts, dict):
            try:
                points, colors = _load_fused_point_cloud(result, result_dir)
                points, colors = _sample_cloud(points, colors)
            except Exception:
                points = np.empty((0, 3), dtype=np.float32)
                colors = np.empty((0, 3), dtype=np.uint8)
            if len(points):
                self.cloud_handles.append(
                    self.server.scene.add_point_cloud(
                        "/perception/fused_AB",
                        points=points,
                        colors=colors,
                        point_size=0.004,
                        point_shape="circle",
                    )
                )
                loaded_clouds += len(points)
            for key, title in (
                ("heatmap", "Fused height map (softmax heatmap)"),
                ("boundary_overlay", "Fused height map + garment boundary"),
                ("fold_edge_overlay", "Fused height-gradient edges"),
            ):
                raw = artifacts.get(key)
                path = result_dir / str(raw) if isinstance(raw, str) and raw else None
                if path is None:
                    continue
                image = _image(path)
                if image is not None:
                    self.image_handles.append(self.server.gui.add_image(image, label=title))

        self.status.content = (
            "### Read-only perception artifact\n\n"
            f"- source: `{result_path}`\n"
            f"- camera views: `{len([v for v in views if isinstance(v, dict)])}`\n"
            f"- displayed points: `{loaded_clouds}`\n"
            "- fused cloud is table-clipped and shown in robot-base metres\n"
            "- no camera, robot connection, Claude process, or action controls are connected"
        )
        return True

    def render_path(self, record: dict[str, Any]) -> None:
        for handle in self.path_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.path_handles.clear()
        try:
            self.server.scene.remove_by_name("/plan")
        except Exception:
            pass
        preflight = record.get("preflight")
        actions = preflight.get("actions") if isinstance(preflight, dict) else None
        if not isinstance(actions, list):
            proposal = record.get("proposal")
            actions = proposal.get("actions") if isinstance(proposal, dict) else None
        if not isinstance(actions, list):
            self.path_panel.content = "### Robot path\n\nNo validated Claude path is available yet."
            return
        try:
            waypoints = path_waypoints_mm(actions, self.robot.init_pose_mm_deg)
        except (KeyError, TypeError, ValueError):
            self.path_panel.content = "### Robot path\n\nThe current action proposal has no renderable move waypoints."
            return
        if not waypoints:
            self.path_panel.content = "### Robot path\n\nNo move/home actions in the current proposal."
            return
        points = np.stack([point for _, point in waypoints], axis=0).astype(np.float32) / 1000.0
        if len(points) >= 2:
            segments = np.stack([points[:-1], points[1:]], axis=1)
            colors = np.repeat(
                np.asarray([[255, 120, 40]], dtype=np.uint8)[None, :, :],
                len(segments),
                axis=0,
            )
            self.path_handles.append(
                self.server.scene.add_line_segments(
                    "/plan/tcp_path",
                    points=segments,
                    colors=np.repeat(colors, 2, axis=1),
                    line_width=4.0,
                )
            )
        self.path_handles.append(
            self.server.scene.add_point_cloud(
                "/plan/waypoints",
                points=points,
                colors=np.tile(np.asarray([[255, 210, 40]], dtype=np.uint8), (len(points), 1)),
                point_size=0.009,
                point_shape="circle",
            )
        )
        moves = sum(1 for name, _ in waypoints if name.startswith("move_"))
        self.path_panel.content = (
            "### Robot path (read-only preview)\n\n"
            f"- move/home waypoints: `{len(waypoints)}`\n"
            f"- Cartesian move count: `{moves}`\n"
            f"- first waypoint (mm): `{waypoints[0][1].round(2).tolist()}`\n"
            f"- last waypoint (mm): `{waypoints[-1][1].round(2).tolist()}`\n"
            "- orange line/yellow points are the validated TCP path; model remains at Home"
        )

    def update_text(
        self,
        record: dict[str, Any],
        result_dir: Path,
        iteration_dir: Path,
    ) -> None:
        claude, evaluation, runtime, selected_overlay = _claude_markdown(record, result_dir)
        self.claude_panel.content = claude
        self.evaluation_panel.content = evaluation
        self.runtime_panel.content = runtime
        self.skill_panel.content = _skill_markdown(record, iteration_dir, self.skills_root)
        self.output_panel.content = _claude_output_markdown(
            record, iteration_dir, self.run_root
        )
        if selected_overlay is not None and str(selected_overlay) != self.selected_overlay_path:
            image = _image(selected_overlay)
            if image is not None:
                # Replace only the optional selected-pixel image, keeping the
                # perception images above intact.
                handle = self.server.gui.add_image(image, label="Claude selected Camera A pixel")
                self.image_handles.append(handle)
                self.selected_overlay_path = str(selected_overlay)

    def update(
        self,
        iteration_dir: Path,
        record: dict[str, Any],
        result_path: Path | None,
    ) -> None:
        result_dir = result_path.parent if result_path is not None else iteration_dir
        if result_path is not None and str(result_path) != self.loaded_perception:
            if self.render_perception(result_path):
                self.loaded_perception = str(result_path)
        self.update_text(record, result_dir, iteration_dir)
        self.render_path(record)


def run_viewer(
    source: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    refresh_s: float = 1.0,
) -> int:
    """Serve the newest saved perception result without any physical side effect."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise PermissionError("read-only artifact Viser must bind to loopback")
    if not 0.1 <= refresh_s <= 30.0:
        raise ValueError("refresh_s must be between 0.1 and 30 seconds")
    try:
        import viser
        from viser.extras import ViserUrdf  # noqa: F401 - validated at startup
    except ImportError as exc:
        raise RuntimeError(
            "Viser is required for the artifact viewer; install it with "
            "python -m pip install 'viser>=1.0,<2'"
        ) from exc

    source = source.expanduser().resolve()
    # Resolve project/run roots from the stable ``runs/<run>`` ancestor so the
    # viewer can follow both the historical molmo_keypoint_cli output and the
    # standalone neat_fold output tree.
    runs_ancestor = next(
        (ancestor for ancestor in (source, *source.parents) if ancestor.name == "runs"),
        None,
    )
    project_root = runs_ancestor.parent.resolve() if runs_ancestor is not None else source.resolve()
    run_root = next(
        (
            ancestor
            for ancestor in (source, *source.parents)
            if ancestor.parent.name == "runs"
        ),
        source,
    ).resolve()
    robot_config_path = run_root / "workspace" / "robot_config.json"
    if not robot_config_path.is_file():
        robot_config_path = project_root / "config" / "robot.example.json"
    try:
        robot = RobotConfig.load(project_root, robot_config_path if robot_config_path.is_file() else None)
    except Exception as exc:
        raise RuntimeError(f"could not load read-only robot model configuration: {exc}") from exc
    urdf_path = project_root / "assets" / "robots" / "xarm7" / "xarm7.urdf"
    if not urdf_path.is_file():
        raise RuntimeError(f"xArm7 URDF is missing: {urdf_path}")

    server = viser.ViserServer(host=host, port=port, label="Molmo/Claude artifacts (read-only)")
    server.scene.set_up_direction("+z")
    bounds = robot.boundaries
    grid_x = ((bounds.x_min or 0.0) + (bounds.x_max or 900.0)) / 2000.0
    grid_y = ((bounds.y_min or -400.0) + (bounds.y_max or 400.0)) / 2000.0
    server.scene.add_grid(
        "/workspace/table",
        width=1.2,
        height=0.8,
        cell_size=0.05,
        section_size=0.25,
        position=(grid_x, grid_y, 0.0),
    )
    state = _ArtifactViserState(
        server,
        robot,
        urdf_path,
        run_root=run_root,
        skills_root=project_root / "data" / "skills",
    )
    stop = threading.Event()

    def follow() -> None:
        while not stop.is_set():
            output_dir = discover_output_dir(source)
            if output_dir is not None:
                iteration_dir, record, result_path = _iteration_snapshot(output_dir)
            else:
                iteration_dir, record, result_path = None, {}, None
            key = (
                str(iteration_dir),
                (iteration_dir / "result.json").stat().st_mtime_ns,
            ) if iteration_dir and (iteration_dir / "result.json").is_file() else None
            if iteration_dir is not None and key != state.loaded_iteration:
                try:
                    with state.lock:
                        state.update(iteration_dir, record, result_path)
                        state.loaded_iteration = key
                except Exception as exc:
                    state.status.content = f"### Artifact update failed\n\n`{type(exc).__name__}: {exc}`"
            stop.wait(refresh_s)

    thread = threading.Thread(target=follow, daemon=True, name="artifact-viser-follow")
    thread.start()
    print(f"Read-only artifact Viser: http://{host}:{port}", flush=True)
    print(f"Following run: {source}", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Artifact Viser stopped; no camera or robot command was sent.", flush=True)
    finally:
        stop.set()
        thread.join(timeout=max(1.0, refresh_s + 0.5))
        server.stop()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="run directory or CLI output directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh-s", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        return run_viewer(
            args.source,
            host=args.host,
            port=args.port,
            refresh_s=args.refresh_s,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Artifact Viser failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
