"""Agent-facing tools and run workspace management."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .claude import ClaudeCodeClient, ClaudeResult
from .config import ExperimentConfig, RobotConfig
from .experiment import ExperimentRunner
from .perception import ClothCenterPerception, MolmoPointClient, PerceptionConfig, PerceptionError, RGBDFrame


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
orientation. Do not import anything, access xArm SDK objects, use shell or
filesystem APIs, add retries, or catch errors. A command failure stops the
rollout immediately.
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
        self.experiment_config.require_ready()
        target = self._validate_experiment_name(experiment_name or self._next_experiment_name())
        context = (
            f"Goal for this run: {self.inspect_file(self.workspace / 'memory.md')}\n"
            f"Experiment parameters (JSON): {self.inspect_file(self.workspace / 'experiment_config.json')}\n"
            f"Create or modify `{target}` in the current workspace.\n"
            f"Research intent (the Agent decides this; implement it literally): {prompt}\n"
            "The file must be directly in the workspace and define `def run():`. "
            "Do not edit ROBOT_API.md, memory.md, results, or any file outside the workspace."
        )
        return self.claude.invoke(context, self.workspace)

    def invoke_molmo(self, image_paths: list[str | Path], prompt: str, config: PerceptionConfig) -> dict[str, Any]:
        """Agent tool: invoke MolmoPoint on one or two images from the current run."""
        safe_images: list[Path] = []
        for raw_path in image_paths:
            path = Path(raw_path)
            if not path.is_absolute():
                path = self.run_dir / path
            path = path.resolve()
            if self.run_dir not in path.parents or not path.is_file():
                raise PermissionError("Molmo images must be existing files inside the current run")
            safe_images.append(path)
        output_dir = self.results / "perception" / datetime.now(timezone.utc).strftime("molmo_%Y%m%dT%H%M%S%fZ")
        output_dir.mkdir(parents=True, exist_ok=False)
        output_path = output_dir / "molmo_output.json"
        return MolmoPointClient(self.project_root, config.molmo).locate(safe_images, output_path, prompt)

    def locate_cloth_center(
        self,
        config: PerceptionConfig,
        frames: list[RGBDFrame] | None = None,
    ) -> dict[str, Any]:
        """Capture selected RGB-D views, call Molmo, and derive a base-frame plan."""
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
            camera_text = ", ".join(result["active_cameras"])
            fusion = result.get("depth_fusion", {})
            if fusion.get("auxiliary_status") == "occluded_or_outside_view":
                quality = fusion.get("primary_depth_quality", {})
                validation_text = (
                    f"Camera B could not observe the A-selected surface, so validated camera A "
                    f"depth was used. Local depth spread: {quality.get('spread_mm')} mm; "
                    f"valid fraction: {quality.get('valid_fraction')}."
                )
            elif result["perception_mode"] == "single_camera_rgbd":
                validation_text = (
                    f"Single-camera RGB-D mode using camera {camera_text}; "
                    "there is no second-view consistency measurement."
                )
            else:
                validation_text = (
                    f"Primary/auxiliary depth disagreement for the same A-selected point: "
                    f"{result['view_disagreement_mm']:.3f} mm."
                )
            handle.write(
                "\n## Perception observation\n\n"
                f"A-selected garment point in base frame: x={center[0]:.3f}, "
                f"y={center[1]:.3f}, z={center[2]:.3f} mm.\n\n"
                f"{validation_text}\n\n"
                "Reason for experiment coordinates: camera A supplied the semantic garment point; "
                "when camera B was active, its reprojected point cloud supplied the selected depth. "
                "Grasp, approach, lift, and yaw were then derived automatically from that point and "
                "robot safety policy.\n"
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
        """Preflight and optionally execute one rollout; never retries automatically."""
        if real:
            metadata = json.loads((self.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            if (
                metadata.get("last_perception_mode") == "single_camera_rgbd"
                and not single_view_confirmed
            ):
                raise PermissionError(
                    "single-camera RGB-D plan requires explicit single-view confirmation"
                )
        return self.runner.run_experiment(path, real=real, confirmed=confirmed, notes=notes)

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
            "invoke_molmo(image_paths, prompt)",
            "locate_cloth_center()",
            "run_experiment(path)",
            "inspect_result(experiment=None)",
        ]
