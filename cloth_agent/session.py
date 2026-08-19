"""Agent-facing tools and run workspace management."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .claude import ClaudeCodeClient, ClaudeResult
from .config import ExperimentConfig, RobotConfig
from .experiment import ExperimentRunner
from .perception import ClothCenterPerception, PerceptionConfig, RGBDFrame
from .skills import skill_prompt


MANUAL_RESULTS = frozenset({"SUCCESS", "FAILED_GRASP", "FAILED_LIFT", "OTHER_FAILURE"})


ROBOT_API_DOC = """# Restricted experiment RobotAPI

Generated scripts must define exactly one `run()` function and may call only:

```python
move(x, y, z, yaw)
open_gripper()
close_gripper()
home()
```

Coordinates are xArm base-frame millimetres; `yaw` is degrees around the
vertical axis. Roll and pitch are fixed to the configured safe grasp
orientation. Perception supplies observations and calibrated coordinate guides;
the fused garment center is a reference rather than a mandatory grasp target.
The Agent chooses the interaction region and every approach, grasp, lift,
transfer/release, and yaw waypoint. Do not import anything, access xArm SDK objects,
use shell or filesystem APIs, add retries, or catch errors. A command failure
stops the rollout immediately.
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AgentSession:
    """Small, explicit loop: inspect -> ask Claude -> preflight -> execute -> reflect."""

    def __init__(self, project_root: Path, run_dir: Path, robot_config: RobotConfig, experiment_config: ExperimentConfig, claude: ClaudeCodeClient | None = None):
        self.project_root = project_root.resolve()
        self.run_dir = run_dir.resolve()
        runs_root = (self.project_root / "runs").resolve()
        if runs_root not in self.run_dir.parents:
            raise PermissionError("run_dir must be inside the project's runs/ directory")
        self.workspace = (self.run_dir / "workspace").resolve()
        self.results = (self.run_dir / "results").resolve()
        self.robot_config = robot_config
        self.experiment_config = experiment_config
        self.claude = claude or ClaudeCodeClient()
        self.runner = ExperimentRunner(self.run_dir, robot_config)
        self.last_return_home_outcome: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        project_root: Path,
        goal: str,
        robot_config: RobotConfig,
        experiment_config: ExperimentConfig,
        run_id: str | None = None,
        claude: ClaudeCodeClient | None = None,
    ) -> "AgentSession":
        run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run_id must be one simple directory name")
        run_dir = (project_root / "runs" / run_id).resolve()
        workspace, results = run_dir / "workspace", run_dir / "results"
        workspace.mkdir(parents=True, exist_ok=False)
        results.mkdir(parents=True, exist_ok=True)
        (workspace / "ROBOT_API.md").write_text(ROBOT_API_DOC, encoding="utf-8")
        (workspace / "experiment_config.json").write_text(
            json.dumps(experiment_config.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (workspace / "robot_config.json").write_text(
            json.dumps(
                {
                    "robot_ip": robot_config.robot_ip,
                    "boundaries": asdict(robot_config.boundaries),
                    "workspace_margin_mm": robot_config.workspace_margin_mm,
                    "lower_z_margin_mm": robot_config.lower_z_margin_mm,
                    "expected_tcp_offset_mm_deg": list(robot_config.expected_tcp_offset_mm_deg),
                    "tcp_offset_tolerance": robot_config.tcp_offset_tolerance,
                    "fixed_orientation_deg": {
                        "roll": robot_config.orientation_roll_deg,
                        "pitch": robot_config.orientation_pitch_deg,
                    },
                    "perception_position": {
                        "joint_angles_deg": (
                            list(robot_config.perception_joints_deg)
                            if robot_config.perception_joints_deg is not None
                            else None
                        ),
                        "tcp_pose_mm_deg": (
                            list(robot_config.perception_pose_mm_deg)
                            if robot_config.perception_pose_mm_deg is not None
                            else None
                        ),
                    },
                    "motion": {
                        "speed_mm_s": robot_config.speed_mm_s,
                        "acceleration_mm_s2": robot_config.acceleration_mm_s2,
                        "home_speed_deg_s": robot_config.home_speed_deg_s,
                        "home_acceleration_deg_s2": robot_config.home_acceleration_deg_s2,
                    },
                    "gripper": {
                        "speed": robot_config.gripper_speed,
                        "open": robot_config.gripper_open,
                        "close": robot_config.gripper_close,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (workspace / "memory.md").write_text(
            f"# Agent working memory\n\nGoal: {goal}\n\n"
            "The first rollout has not run. Record why each later experiment is changed.\n",
            encoding="utf-8",
        )
        (run_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "created_at": _now(),
                    "goal": goal,
                    "experiment_config": experiment_config.as_dict(),
                    "robot_config": asdict(robot_config),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return cls(project_root, run_dir, robot_config, experiment_config, claude=claude)

    def _safe_read_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            workspace_candidate = self.workspace / candidate
            candidate = workspace_candidate if workspace_candidate.exists() else self.project_root / candidate
        candidate = candidate.resolve()
        if candidate != self.project_root and self.project_root not in candidate.parents:
            raise PermissionError("inspect_file is limited to the project and current run workspace")
        runs_root = (self.project_root / "runs").resolve()
        if runs_root in candidate.parents and self.run_dir not in candidate.parents:
            raise PermissionError("inspect_file cannot read another run workspace")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    def inspect_file(self, path: str | Path) -> str:
        """Read project or current-experiment code without granting write access."""
        return self._safe_read_path(path).read_text(encoding="utf-8")

    def _next_experiment_name(self) -> str:
        existing = sorted(self.workspace.glob("experiment_*.py"))
        index = len(existing) + 1
        return f"experiment_{index:03d}_grasp_lift_drop.py" if index == 1 else f"experiment_{index:03d}.py"

    @staticmethod
    def _validate_experiment_name(name: str) -> str:
        if Path(name).name != name or not name.startswith("experiment_") or not name.endswith(".py"):
            raise ValueError("experiment name must be experiment_*.py directly inside the run workspace")
        return name

    def invoke_claude_code(self, prompt: str, *, experiment_name: str | None = None) -> ClaudeResult:
        """Ask Claude to write/modify code inside this run's workspace only."""
        # Perception intentionally leaves point selection and motion geometry
        # to Claude.  The center is one reference; coordinate-guide files ground
        # other self-selected visual regions in calibrated robot-base XYZ.
        self.experiment_config.require_center()
        target = self._validate_experiment_name(experiment_name or self._next_experiment_name())
        perception_dir = self.workspace / "perception_views"
        perception_files = sorted(perception_dir.glob("*")) if perception_dir.is_dir() else []
        perception_context = (
            "Perception view files available for visual inspection:\n"
            + "\n".join(f"- perception_views/{path.name}" for path in perception_files)
            + "\n"
            if perception_files
            else "No copied perception images are available; use the fused center/surface observation only.\n"
        )
        context = (
            f"Goal for this run: {self.inspect_file(self.workspace / 'memory.md')}\n"
            f"Experiment parameters (JSON): {self.inspect_file(self.workspace / 'experiment_config.json')}\n"
            f"Create or modify `{target}` in the current workspace.\n"
            f"Research intent (the Agent decides this; implement it literally): {prompt}\n"
            "The research objective is to make the garment as open and spread as safely "
            "possible. A usable garment lifting anchor is an intermediate tool that supports "
            "a useful hanging configuration and controlled laydown; finding one is not the "
            "terminal goal while the garment can still be opened further. Do not "
            "require a semantic sleeve/collar/hem label, a fixed candidate list, or a "
            "hard-coded probe/verification state machine. The center is a validated "
            "reference only, not a required grasp target. Claude chooses the region; "
            "perception provides calibrated coordinate grounding.\n"
            f"{perception_context}"
            "The `height_map`/`height_map_boundary` files show the garment surface "
            "height above the fitted table plane in millimeters; brighter colors mean "
            "a larger garment/table height difference. Statistics are normalized inside "
            "the garment mask. `fold_edge_overlay` is retained as a height-gradient/"
            "occlusion diagnostic, and `height_map_global` is the whole-scene reference. "
            "Read `perception_views/observation.json` and the unranked "
            "`camera_*_coordinate_guide.json` files to ground a self-selected visual "
            "region in measured robot-base XYZ. Cyan Rxxx markers are coordinate "
            "references, not grasp candidates; do not invent a camera transform.\n"
            "Prefer a direct opening, hanging, regrasp, or laydown maneuver. A script may "
            "perform a minimal cautious test only when uncertainty prevents a grounded "
            "opening action. If the "
            "current grasp appears to be a useful anchor, you may apply the Laydown "
            "procedural skill below, but you must still choose and emit every concrete "
            "move/gripper command yourself.\n\n"
            f"{skill_prompt()}\n\n"
            "The file must be directly in the workspace and define `def run():`. "
            "Do not edit ROBOT_API.md, memory.md, results, or any file outside the workspace."
        )
        log_dir = self.results / "claude"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        try:
            result = self.claude.invoke(context, self.workspace)
        except BaseException as exc:
            (log_dir / f"{stamp}_failed.json").write_text(
                json.dumps(
                    {
                        "prompt": context,
                        "error": f"{type(exc).__name__}: {exc}",
                        "created_at": _now(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise
        payload = (
            asdict(result)
            if isinstance(result, ClaudeResult)
            else {"prompt": context, "result": repr(result), "created_at": _now()}
        )
        (log_dir / f"{stamp}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    def locate_cloth_center(
        self,
        config: PerceptionConfig,
        frames: list[RGBDFrame] | None = None,
    ) -> dict[str, Any]:
        """Capture both RGB-D views and return an observation for Claude planning."""
        output_dir = self.results / "perception" / datetime.now(timezone.utc).strftime("center_%Y%m%dT%H%M%S%fZ")
        perception = ClothCenterPerception(self.project_root, self.robot_config, config)
        try:
            result, updated = perception.locate(
                output_dir, self.experiment_config, frames=frames
            )
        except BaseException as exc:
            output_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "created_at": _now(),
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            }
            (output_dir / "failure.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise
        self.experiment_config = updated
        (self.workspace / "experiment_config.json").write_text(
            json.dumps(updated.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Claude Code is confined to the current workspace.  Copy the visual
        # evidence there so its waypoint choices can be based on the actual A/B
        # views and fused height preview instead of only numeric center values.
        perception_views_dir = self.workspace / "perception_views"
        perception_views_dir.mkdir(parents=True, exist_ok=True)
        copied_names: list[str] = []
        for view in result.get("views", []):
            if not isinstance(view, dict):
                continue
            for key in (
                "annotated_image",
                "image",
                "depth_m",
                "height_map",
                "height_map_global",
                "height_map_boundary",
                "height_map_path",
                "garment_mask",
                "height_gradient_overlay",
                "base_xyz_map",
                "coordinate_guide",
                "coordinate_overlay",
                "depth_heatmap",
                "depth_heatmap_global",
                "depth_heatmap_boundary",
                "fold_edge_overlay",
            ):
                raw_name = view.get(key)
                if not raw_name:
                    continue
                source = output_dir / str(raw_name)
                if not source.is_file():
                    continue
                destination = perception_views_dir / source.name
                shutil.copy2(source, destination)
                if destination.name not in copied_names:
                    copied_names.append(destination.name)
        fusion_artifacts = result.get("depth_fusion", {}).get("artifacts", {})
        for artifact_key in (
            "fused_points_base_mm",
            "fused_colors_rgb",
            "fused_source_mask",
            "path",
            "preview",
            "heatmap",
            "boundary_overlay",
        ):
            preview_name = fusion_artifacts.get(artifact_key)
            if not preview_name:
                continue
            source = output_dir / str(preview_name)
            if source.is_file():
                destination = perception_views_dir / source.name
                shutil.copy2(source, destination)
                if destination.name not in copied_names:
                    copied_names.append(destination.name)
        (perception_views_dir / "observation.json").write_text(
            json.dumps(
                {
                    "center_base_mm": result.get("center_base_mm"),
                    "surface_z_mm": result.get("surface_z_mm"),
                    "center_is_reference_only": True,
                    "waypoint_authority": "Claude",
                    "table_plane": result.get("depth_fusion", {}).get("table_plane"),
                    "coordinate_guides": [
                        {
                            "camera": view.get("label"),
                            "overlay": view.get("coordinate_overlay"),
                            "guide": view.get("coordinate_guide"),
                            "full_resolution_xyz_map": view.get("base_xyz_map"),
                        }
                        for view in result.get("views", [])
                        if isinstance(view, dict) and view.get("coordinate_guide")
                    ],
                    "files": copied_names,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        metadata_path = self.run_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["experiment_config"] = updated.as_dict()
        metadata["last_perception_mode"] = result["perception_mode"]
        metadata["last_active_cameras"] = result["active_cameras"]
        metadata.setdefault("perception_results", []).append(
            str((output_dir / "result.json").relative_to(self.run_dir))
        )
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        memory = self.workspace / "memory.md"
        with memory.open("a", encoding="utf-8") as handle:
            center = result["center_base_mm"]
            fusion = result.get("depth_fusion", {})
            validation_text = (
                f"Dense A/B voxel fusion used {fusion.get('fused_point_count')} fused points "
                f"from {fusion.get('input_point_count')} valid camera points; "
                f"shared A/B voxels: {fusion.get('source_voxel_counts', {}).get('AB_overlap')}."
            )
            handle.write(
                "\n## Perception observation\n\n"
                f"Fused garment center/surface observation in base frame: x={center[0]:.3f}, "
                f"y={center[1]:.3f}, z={center[2]:.3f} mm.\n\n"
                f"{validation_text}\n\n"
                "Reason for experiment coordinates: calibrated A/B RGB-D points were transformed "
                "to the robot base frame, voxel-fused, and segmented by height above the fitted table. "
                "The fused center is a reference only. Uniform per-camera coordinate guides map "
                "visual garment regions to calibrated robot-base XYZ without selecting or ranking "
                "grasp candidates; Claude must choose the interaction region and all waypoints.\n\n"
                f"Visual evidence copied to the workspace: perception_views/{', perception_views/'.join(copied_names)}\n"
            )
        return result

    def run_experiment(
        self,
        path: str | Path,
        *,
        real: bool = False,
        confirmed: bool = False,
        single_view_confirmed: bool = False,
        notes: str = "",
    ) -> dict[str, Any]:
        """Execute one rollout and always attempt Home after physical Claude motion."""

        if real:
            metadata = json.loads(
                (self.run_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            if (
                metadata.get("last_perception_mode") == "single_camera_rgbd"
                and not single_view_confirmed
            ):
                raise PermissionError(
                    "single-camera RGB-D plan requires explicit single-view confirmation"
                )

        result: dict[str, Any] | None = None
        self.last_return_home_outcome = None
        try:
            result = self.runner.run_experiment(
                path,
                real=real,
                confirmed=confirmed,
                notes=notes,
            )
            return result
        finally:
            if real and confirmed:
                outcome = self._attempt_return_home(
                    notes=(
                        "Mandatory post-Claude return to configured Home, attempted "
                        "regardless of rollout success or failure."
                    )
                )
                self.last_return_home_outcome = outcome
                if result is not None:
                    result["mandatory_return_home"] = outcome
                    result_path = self.results / f"{result['experiment']}.json"
                    result_path.write_text(
                        json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

    def _attempt_return_home(self, *, notes: str) -> dict[str, Any]:
        """Attempt a fresh, isolated Home command without masking the rollout result."""

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        source_name = f"_mandatory_return_home_{stamp}.py"
        source_path = self.workspace / source_name
        outcome: dict[str, Any] = {
            "attempted": True,
            "completed": False,
            "result_path": None,
            "error": None,
        }
        try:
            source_path.write_text("def run():\n    home()\n", encoding="utf-8")
            home_result = self.runner.run_experiment(
                source_name,
                real=True,
                confirmed=True,
                notes=notes,
            )
            outcome["completed"] = bool(home_result.get("execution_completed"))
            outcome["result_path"] = str(
                (self.results / f"{home_result['experiment']}.json").relative_to(
                    self.run_dir
                )
            )
            outcome["robot_errors"] = list(home_result.get("robot_errors", []))
        except BaseException as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"
            expected_result = self.results / f"{Path(source_name).stem}.json"
            if expected_result.is_file():
                outcome["result_path"] = str(expected_result.relative_to(self.run_dir))
        finally:
            if source_path.is_file():
                source_path.unlink()
            event_dir = self.results / "mandatory_return_home"
            event_dir.mkdir(parents=True, exist_ok=True)
            (event_dir / f"{stamp}.json").write_text(
                json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return outcome

    def inspect_result(self, experiment: str | None = None) -> dict[str, Any]:
        return self.runner.inspect_result(experiment)

    def record_manual_result(self, experiment: str, status: str, notes: str = "") -> dict[str, Any]:
        status = status.strip().upper()
        if status not in MANUAL_RESULTS:
            raise ValueError(f"status must be one of: {', '.join(sorted(MANUAL_RESULTS))}")
        result = self.inspect_result(experiment)
        result["human_result"] = status
        if notes:
            result["notes"] = (result.get("notes", "") + "\n" + notes).strip()
        (self.results / f"{result['experiment']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def update_memory(self, experiment: str, *, hypothesis: str, next_experiment: str, result: str, notes: str = "") -> None:
        memory = self.workspace / "memory.md"
        with memory.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n## {experiment}\n\n"
                f"Result: {result}\n\n"
                f"Why this change / current hypothesis: {hypothesis}\n\n"
                f"Next experiment: {next_experiment}\n"
            )
            if notes:
                handle.write(f"\nNotes: {notes}\n")

    def tool_manifest(self) -> list[str]:
        return [
            "inspect_file(path)",
            "invoke_claude_code(prompt)",
            "locate_cloth_center()",
            "run_experiment(path)",
            "inspect_result(experiment=None)",
        ]
