"""Language instruction -> Claude-authored skill -> grounded robot action.

This is an intentionally separate experiment pipeline.  It accepts one natural
language instruction, asks Claude to author a task-specific ``SKILL.md``, then
asks a fresh Claude process to use that skill with the saved RGB-D evidence and
an optional MolmoPoint query.  The second stage can only return the restricted
RobotAPI action vocabulary; the existing static preflight and execution gates
remain the final authority.

Preview/simulated execution is the default.  Physical execution requires both
``--real`` and ``--confirm-real`` after the printed action sequence has been
reviewed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import ExperimentConfig, RobotConfig
from .experiment import ExperimentValidationError, format_action_sequence, format_speed_profile
from .perception import MolmoConfig, MolmoPointClient, PerceptionConfig
from .session import AgentSession


class LanguageSkillPipelineError(RuntimeError):
    """Raised when either Claude stage or its safety contract fails."""


SKILL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "name",
        "description",
        "instructions",
        "molmo_prompt",
        "success_criteria",
        "safety_constraints",
    ],
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 63},
        "description": {"type": "string", "minLength": 1, "maxLength": 500},
        "instructions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 16,
            "items": {"type": "string", "minLength": 1, "maxLength": 1200},
        },
        "molmo_prompt": {"type": "string", "maxLength": 600},
        "success_criteria": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "safety_constraints": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
        },
    },
}

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "skill_name",
        "interpretation",
        "confidence",
        "actions",
        "expected_observation",
        "safety_notes",
    ],
    "properties": {
        "skill_name": {"type": "string", "minLength": 1, "maxLength": 63},
        "interpretation": {"type": "string", "minLength": 1, "maxLength": 1200},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "actions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "args"],
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": ["move", "open_gripper", "close_gripper", "home"],
                    },
                    "args": {
                        "type": "object",
                        "additionalProperties": False,
                        "description": "Use exactly {} for gripper and home actions; use x,y,z,yaw only for move.",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                            "yaw": {"type": "number"},
                        },
                    },
                },
            },
        },
        "expected_observation": {"type": "string", "minLength": 1, "maxLength": 1000},
        "safety_notes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
        },
    },
}

ACTION_NAMES = frozenset({"move", "open_gripper", "close_gripper", "home"})
MAX_ACTIONS = 12


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _safe_cli_binary(binary: str) -> str:
    resolved = shutil.which(binary) if Path(binary).name == binary else binary
    if resolved is None:
        raise LanguageSkillPipelineError(f"Claude CLI not found: {binary}")
    return resolved


def _json_from_claude_text(text: str) -> dict[str, Any]:
    """Extract one JSON object from direct, wrapped, or fenced Claude output."""

    candidates = [text.strip()]
    try:
        outer = json.loads(text)
        if isinstance(outer, dict) and isinstance(outer.get("structured_output"), dict):
            return outer["structured_output"]
        if isinstance(outer, dict) and isinstance(outer.get("structuredOutput"), dict):
            return outer["structuredOutput"]
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
    raise LanguageSkillPipelineError("Claude response did not contain a JSON object")


def _grounding_mcp_config(run_dir: Path) -> dict[str, Any]:
    """Build the scoped grounding MCP config without importing the old viewer pipeline."""

    root = Path(run_dir).resolve()
    perception_dir = root / "workspace" / "perception_views"
    server_script = Path(__file__).resolve().with_name("garment_grounding_mcp.py")
    if not perception_dir.is_dir() or not server_script.is_file():
        raise LanguageSkillPipelineError("saved perception views and the grounding MCP server are required")
    return {
        "mcpServers": {
            "garment_grounding": {
                "type": "stdio",
                "command": sys.executable,
                "args": [str(server_script), "--perception-dir", str(perception_dir)],
            }
        }
    }


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    normalized = normalized[:63].rstrip("-")
    if not normalized or not re.fullmatch(r"[a-z][a-z0-9-]*", normalized):
        raise LanguageSkillPipelineError(
            "Claude skill name must contain lowercase ASCII letters, digits, and hyphens"
        )
    return normalized


def _nonempty_strings(
    value: Any,
    field: str,
    maximum_length: int,
    maximum_items: int,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum_items:
        raise LanguageSkillPipelineError(f"skill {field} must be a non-empty list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or len(item) > maximum_length:
            raise LanguageSkillPipelineError(f"skill {field} contains an invalid string")
        result.append(item.strip())
    return tuple(result)


@dataclass(frozen=True)
class SkillDraft:
    """The structured result of Claude's skill-authoring stage."""

    name: str
    description: str
    instructions: tuple[str, ...]
    molmo_prompt: str
    success_criteria: tuple[str, ...]
    safety_constraints: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SkillDraft":
        if not isinstance(payload, Mapping):
            raise LanguageSkillPipelineError("Claude skill response must be an object")
        required = {
            "name",
            "description",
            "instructions",
            "molmo_prompt",
            "success_criteria",
            "safety_constraints",
        }
        if set(payload) != required:
            raise LanguageSkillPipelineError(
                f"skill response fields mismatch; missing={sorted(required - set(payload))}, "
                f"extra={sorted(set(payload) - required)}"
            )
        name = _slug(str(payload.get("name", "")))
        description = payload.get("description")
        if not isinstance(description, str) or not description.strip() or len(description) > 500:
            raise LanguageSkillPipelineError("skill description must be a non-empty string")
        molmo_prompt = payload.get("molmo_prompt", "")
        if not isinstance(molmo_prompt, str) or len(molmo_prompt) > 600:
            raise LanguageSkillPipelineError("skill molmo_prompt must be a string")
        return cls(
            name=name,
            description=description.strip(),
            instructions=_nonempty_strings(payload.get("instructions"), "instructions", 1200, 16),
            molmo_prompt=molmo_prompt.strip(),
            success_criteria=_nonempty_strings(payload.get("success_criteria"), "success_criteria", 500, 8),
            safety_constraints=_nonempty_strings(payload.get("safety_constraints"), "safety_constraints", 500, 12),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "instructions": list(self.instructions),
            "molmo_prompt": self.molmo_prompt,
            "success_criteria": list(self.success_criteria),
            "safety_constraints": list(self.safety_constraints),
        }

    def markdown(self) -> str:
        lines = [
            "---",
            f"name: {json.dumps(self.name, ensure_ascii=False)}",
            f"description: {json.dumps(self.description, ensure_ascii=False)}",
            "---",
            "",
            f"# {self.name}",
            "",
            self.description,
            "",
            "## Procedure",
            "",
        ]
        lines.extend(f"{index}. {step}" for index, step in enumerate(self.instructions, 1))
        lines.extend(["", "## Success criteria", ""])
        lines.extend(f"- {criterion}" for criterion in self.success_criteria)
        lines.extend(["", "## Safety constraints", ""])
        lines.extend(f"- {constraint}" for constraint in self.safety_constraints)
        lines.extend([
            "",
            "## Runtime contract",
            "",
            "Use the current calibrated RGB-D evidence to choose geometry. The skill provides procedure only; it never supplies fixed coordinates or bypasses RobotAPI preflight.",
            "",
        ])
        return "\n".join(lines)


def _validate_actions(payload: Mapping[str, Any], skill_name: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise LanguageSkillPipelineError("Claude execution response must be an object")
    required = {"skill_name", "interpretation", "confidence", "actions", "expected_observation", "safety_notes"}
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        raise LanguageSkillPipelineError(f"execution response fields mismatch; missing={missing}, extra={extra}")
    if payload["skill_name"] != skill_name:
        raise LanguageSkillPipelineError("execution response used a different skill name")
    if not isinstance(payload["interpretation"], str) or not payload["interpretation"].strip() or len(payload["interpretation"]) > 1200:
        raise LanguageSkillPipelineError("execution interpretation must be non-empty")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
        raise LanguageSkillPipelineError("execution confidence must be a finite number in [0, 1]")
    actions = payload["actions"]
    if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_ACTIONS:
        raise LanguageSkillPipelineError(f"execution actions must contain 1..{MAX_ACTIONS} items")
    seen_move = False
    close_indices: list[int] = []
    open_indices: list[int] = []
    normalized: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping) or set(action) != {"name", "args"}:
            raise LanguageSkillPipelineError(f"action {index + 1} must contain exactly name and args")
        name = action["name"]
        args = action["args"]
        if name not in ACTION_NAMES or not isinstance(args, Mapping):
            raise LanguageSkillPipelineError(f"action {index + 1} uses an unsupported command")
        if name == "move":
            if set(args) != {"x", "y", "z", "yaw"}:
                raise LanguageSkillPipelineError(f"move action {index + 1} must contain x,y,z,yaw")
            values: dict[str, float] = {}
            for key in ("x", "y", "z", "yaw"):
                value = args[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise LanguageSkillPipelineError(f"move action {index + 1} has a non-finite {key}")
                values[key] = float(value)
            seen_move = True
            normalized.append({"name": name, "args": values})
        else:
            if args:
                # Claude occasionally serializes the no-argument calls with a
                # zero-valued move-shaped object.  It is semantically empty;
                # accept only that exact harmless padding and reject all other
                # hidden parameters.
                if set(args) != {"x", "y", "z", "yaw"} or any(
                    isinstance(args[key], bool)
                    or not isinstance(args[key], (int, float))
                    or not math.isfinite(float(args[key]))
                    or float(args[key]) != 0.0
                    for key in ("x", "y", "z", "yaw")
                ):
                    raise LanguageSkillPipelineError(f"{name} action {index + 1} must have empty args")
            normalized.append({"name": name, "args": {}})
            if name == "close_gripper":
                close_indices.append(index)
            if name == "open_gripper":
                open_indices.append(index)
    if not seen_move:
        raise LanguageSkillPipelineError("execution plan must contain at least one move")
    if close_indices and not any(open_index > close_indices[-1] for open_index in open_indices):
        raise LanguageSkillPipelineError("a plan that closes the gripper must release it before ending")
    if close_indices:
        first_close = close_indices[0]
        first_release = next((index for index in open_indices if index > first_close), None)
        if first_release is None or not any(
            action["name"] == "move" for action in normalized[first_close + 1 : first_release]
        ):
            raise LanguageSkillPipelineError("a grasp must include a post-grasp move before release")
    safety_notes = _nonempty_strings(payload["safety_notes"], "safety_notes", 500, 12)
    expected = payload["expected_observation"]
    if not isinstance(expected, str) or not expected.strip() or len(expected) > 1000:
        raise LanguageSkillPipelineError("expected_observation must be non-empty")
    return {
        "skill_name": skill_name,
        "interpretation": str(payload["interpretation"]).strip(),
        "confidence": float(confidence),
        "actions": normalized,
        "expected_observation": expected.strip(),
        "safety_notes": list(safety_notes),
    }


def actions_to_source(actions: Sequence[Mapping[str, Any]]) -> str:
    """Compile validated action dictionaries to the existing restricted script."""

    lines = ["def run():"]
    for action in actions:
        name = action["name"]
        args = action["args"]
        if name == "move":
            lines.append(f"    move({args['x']!r}, {args['y']!r}, {args['z']!r}, {args['yaw']!r})")
        else:
            lines.append(f"    {name}()")
    return "\n".join(lines) + "\n"


class _ClaudeJSONClient:
    def __init__(self, binary: str = "claude", timeout_s: int = 400):
        self.binary = binary
        self.timeout_s = timeout_s

    def _invoke(
        self,
        prompt: str,
        *,
        system_prompt: str,
        schema: Mapping[str, Any],
        workspace: Path,
        project_root: Path,
        allowed_tools: Sequence[str],
        mcp_config: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        binary = _safe_cli_binary(self.binary)
        command = [
            binary,
            "--print",
            prompt,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
            "--permission-mode",
            "dontAsk",
            "--allowedTools",
            ",".join(allowed_tools),
            "--tools",
            ",".join(tool for tool in allowed_tools if not tool.startswith("mcp__")),
            "--disable-slash-commands",
            "--no-session-persistence",
            "--add-dir",
            str(project_root.resolve()),
            "--system-prompt",
            system_prompt,
        ]
        if mcp_config is not None:
            command.extend(["--mcp-config", json.dumps(mcp_config, ensure_ascii=False, separators=(",", ":")), "--strict-mcp-config"])
        else:
            command.append("--safe-mode")
        try:
            completed = subprocess.run(
                command,
                cwd=workspace.resolve(),
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise LanguageSkillPipelineError(f"Claude timed out after {self.timeout_s} seconds") from exc
        except OSError as exc:
            raise LanguageSkillPipelineError(f"Claude invocation failed: {exc}") from exc
        invocation = {
            "prompt": prompt,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "created_at": _now(),
        }
        if completed.returncode != 0:
            raise LanguageSkillPipelineError(
                f"Claude exited with {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        try:
            payload = _json_from_claude_text(completed.stdout)
        except BaseException as exc:
            raise LanguageSkillPipelineError(f"Claude response was not valid JSON: {exc}") from exc
        return payload, invocation


class ClaudeSkillAuthor(_ClaudeJSONClient):
    """Claude stage that authors one portable, task-specific SKILL.md."""

    def author(self, instruction: str, *, workspace: Path, project_root: Path) -> tuple[SkillDraft, dict[str, Any]]:
        prompt = (
            "Author one task-specific skill for a robot garment manipulation experiment.\n"
            f"The user's language instruction is: {instruction}\n\n"
            "Return only the requested JSON object. Write concise imperative procedure steps for a second Claude planner. "
            "Do not choose image coordinates, invent measurements, emit Python, or directly control a robot. "
            "The optional molmo_prompt must be a single visual pointing query for the semantic region implied by the instruction; "
            "leave it empty when Molmo would not add evidence. The runtime will save your result as SKILL.md and, when requested, "
            "run the existing MolmoPoint worker before the execution-planning stage."
        )
        system_prompt = (
            "You are a robotics skill author. Design reusable procedural guidance, not a robot program. "
            "Keep the skill under 500 lines, avoid fixed coordinates and hidden trajectories, state uncertainty handling, "
            "and include explicit safety constraints. Use a lowercase ASCII hyphenated name. Return JSON only."
        )
        payload, invocation = self._invoke(
            prompt,
            system_prompt=system_prompt,
            schema=SKILL_SCHEMA,
            workspace=workspace,
            project_root=project_root,
            allowed_tools=("Read",),
        )
        return SkillDraft.from_payload(payload), invocation


class ClaudeSkillExecutor(_ClaudeJSONClient):
    """Claude stage that grounds one action plan using the generated skill."""

    def execute(
        self,
        instruction: str,
        skill: SkillDraft,
        *,
        image_paths: Iterable[Path],
        perception_dir: Path,
        molmo_path: Path | None,
        workspace: Path,
        project_root: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        images = [path.resolve() for path in image_paths if path.is_file()]
        if not images:
            raise LanguageSkillPipelineError("skill execution requires saved garment images")
        image_text = "\n".join(f"- {path}" for path in images)
        molmo_text = str(molmo_path.resolve()) if molmo_path is not None and molmo_path.is_file() else "disabled or unavailable"
        has_grounding = all(
            (perception_dir / f"camera_{camera}_coordinate_guide.json").is_file()
            for camera in ("A", "B")
        )
        grounding_text = (
            "A read-only garment_grounding MCP server exposes lookup_reference(camera, reference_id). "
            "Inspect all images first, select one final Rxxx visually, call lookup_reference exactly once, and then compose the JSON."
            if has_grounding
            else "No coordinate-guide MCP is available; use only saved measured observation files and remain conservative."
        )
        prompt = (
            f"User instruction: {instruction}\n\n"
            f"Generated skill ({skill.name}):\n{skill.markdown()}\n\n"
            f"Saved garment images and overlays:\n{image_text}\n\n"
            f"Optional Molmo result JSON: {molmo_text}\n"
            "Read `workspace/perception_views/observation.json` and the camera coordinate-guide files before grounding geometry. "
            f"{grounding_text}\n\n"
            "Use the generated skill as the procedure, but adapt its geometry to the current observation and the exact user instruction. "
            "The center observation is only a reference. Claude chooses the interaction region and every waypoint. "
            "Return exactly skill_name, interpretation, confidence, actions, expected_observation, and safety_notes. "
            "Each action must be {name,args}; move args must contain numeric x,y,z,yaw in robot-base millimetres/degrees. "
            "Use only move, open_gripper, close_gripper, and home. Keep at most 12 actions. If you close the gripper, release it before ending. "
            "Do not return Python, SDK calls, shell commands, candidate lists, or a hidden state machine."
        )
        system_prompt = (
            "You are the execution planner for a safety-gated xArm garment experiment. "
            "Read the supplied images and calibrated files. Follow the generated skill and satisfy the user's instruction, "
            "but never invent a transform or bypass measurement uncertainty. You may only read files and use the single read-only "
            "grounding lookup when it is configured. Return JSON only; the runtime will static-validate every action before execution."
        )
        mcp = _grounding_mcp_config(workspace.parent) if has_grounding else None
        payload, invocation = self._invoke(
            prompt,
            system_prompt=system_prompt,
            schema=PLAN_SCHEMA,
            workspace=workspace,
            project_root=project_root,
            allowed_tools=("Read", "mcp__garment_grounding__lookup_reference") if has_grounding else ("Read",),
            mcp_config=mcp,
        )
        return _validate_actions(payload, skill.name), invocation


def _workspace_images(perception_dir: Path) -> list[Path]:
    preferred = sorted(perception_dir.glob("camera_*_A.png")) + sorted(perception_dir.glob("camera_*_B.png"))
    all_images = sorted(perception_dir.glob("*.png"))
    return preferred + [path for path in all_images if path not in preferred]


def _load_saved_observation(session: AgentSession) -> dict[str, Any]:
    observation_path = session.workspace / "perception_views" / "observation.json"
    if not observation_path.is_file():
        raise LanguageSkillPipelineError(
            f"--skip-capture requires saved perception evidence at {observation_path}"
        )
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    center = observation.get("center_base_mm")
    if not isinstance(center, list) or len(center) != 3:
        raise LanguageSkillPipelineError("saved observation has no valid center_base_mm")
    session.experiment_config = ExperimentConfig.from_mapping(
        {"center_x": center[0], "center_y": center[1], "grasp_z": center[2]},
        allow_deferred=True,
    )
    (session.workspace / "experiment_config.json").write_text(
        json.dumps(session.experiment_config.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return observation


def _load_or_create_session(
    project_root: Path,
    *,
    run_dir: Path | None,
    run_id: str | None,
    robot_config: Path | None,
    instruction: str,
) -> AgentSession:
    root = project_root.resolve()
    if run_dir is not None:
        run = run_dir.resolve()
        metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
        saved_robot = run / "workspace" / "robot_config.json"
        robot = RobotConfig.load(root, saved_robot if saved_robot.is_file() else robot_config)
        saved_experiment = run / "workspace" / "experiment_config.json"
        values = json.loads(saved_experiment.read_text(encoding="utf-8")) if saved_experiment.is_file() else metadata.get("experiment_config", {})
        return AgentSession(root, run, robot, ExperimentConfig.from_mapping(values, allow_deferred=True))
    if run_id:
        existing = root / "runs" / run_id
        if existing.is_dir():
            return _load_or_create_session(root, run_dir=existing, run_id=None, robot_config=robot_config, instruction=instruction)
    robot = RobotConfig.load(root, robot_config)
    return AgentSession.create(root, f"Language instruction: {instruction}", robot, ExperimentConfig(), run_id=run_id)


@dataclass
class LanguageSkillPipeline:
    """Orchestrate one language-conditioned skill experiment."""

    session: AgentSession
    instruction: str
    project_root: Path
    perception_config: Path | None = None
    capture: bool = True
    use_molmo: bool = False
    molmo_python: Path | None = None
    molmo_model: str = "allenai/MolmoPoint-8B"
    claude_binary: str = "claude"
    claude_timeout_s: int = 400
    real: bool = False
    confirm_real: bool = False
    skill_author: ClaudeSkillAuthor | Any | None = None
    skill_executor: ClaudeSkillExecutor | Any | None = None

    def _results_dir(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.session.results / "language_skill_pipeline" / stamp
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _run_molmo(self, skill: SkillDraft, output_dir: Path, images: list[Path]) -> Path | None:
        if not self.use_molmo or not skill.molmo_prompt:
            _json_dump(output_dir / "molmo.json", {"status": "disabled", "reason": "--use-molmo not set or skill supplied no prompt"})
            return None
        if len(images) not in {1, 2}:
            raise LanguageSkillPipelineError("MolmoPoint requires one or two RGB camera images")
        python = self.molmo_python
        if python is None:
            python_env = os.environ.get("MOLMO_PYTHON")
            candidates = [Path(python_env)] if python_env else []
            candidates.extend([Path.home() / "miniconda3/envs/molmo/bin/python", Path(sys.executable)])
            python = next((candidate for candidate in candidates if candidate.is_file()), None)
        if python is None or not python.is_file():
            raise LanguageSkillPipelineError("Molmo Python was not found; pass --molmo-python or set MOLMO_PYTHON")
        output_path = output_dir / "molmo.json"
        config = MolmoConfig(python=python.resolve(), model=self.molmo_model)
        result = MolmoPointClient(self.project_root, config).locate(images, output_path, skill.molmo_prompt)
        _json_dump(output_path, result)
        return output_path

    def run(self) -> dict[str, Any]:
        instruction = self.instruction.strip()
        if not instruction:
            raise LanguageSkillPipelineError("instruction must be non-empty")
        if self.real and not self.confirm_real:
            raise LanguageSkillPipelineError("physical execution requires both --real and --confirm-real")
        root = self.project_root.resolve()
        output_dir = self._results_dir()
        _json_dump(output_dir / "instruction.json", {"instruction": instruction, "created_at": _now()})

        # The explicit experiment contract is instruction -> skill authoring ->
        # observation/tool use -> execution planning.  Authoring first keeps
        # the generated procedure independent from a stale visual snapshot.
        author = self.skill_author or ClaudeSkillAuthor(self.claude_binary, self.claude_timeout_s)
        skill, author_invocation = author.author(instruction, workspace=self.session.workspace, project_root=root)
        skill_dir = self.session.workspace / "generated_skills" / skill.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(skill.markdown(), encoding="utf-8")
        _json_dump(output_dir / "skill.json", skill.as_dict())
        _json_dump(output_dir / "skill_author_invocation.json", author_invocation)

        if self.capture:
            config_path = (self.perception_config or root / "config" / "perception.free_exploration.json").resolve()
            config = PerceptionConfig.load(root, config_path)
            perception = self.session.locate_cloth_center(config)
            _json_dump(output_dir / "perception.json", perception)
        else:
            perception = _load_saved_observation(self.session)
            _json_dump(output_dir / "perception.json", perception)

        images = _workspace_images(self.session.workspace / "perception_views")
        molmo_path = self._run_molmo(skill, output_dir, images[:2])
        executor = self.skill_executor or ClaudeSkillExecutor(self.claude_binary, self.claude_timeout_s)
        plan, executor_invocation = executor.execute(
            instruction,
            skill,
            image_paths=images,
            perception_dir=self.session.workspace / "perception_views",
            molmo_path=molmo_path,
            workspace=self.session.workspace,
            project_root=root,
        )
        _json_dump(output_dir / "execution_plan.json", plan)
        _json_dump(output_dir / "skill_executor_invocation.json", executor_invocation)

        source_name = "experiment_001_language_skill.py"
        while (self.session.workspace / source_name).exists():
            index = int(re.search(r"(\d+)", source_name).group(1)) + 1
            source_name = f"experiment_{index:03d}_language_skill.py"
        source = actions_to_source(plan["actions"])
        source_path = self.session.workspace / source_name
        source_path.write_text(source, encoding="utf-8")
        preflight = self.session.runner.preflight(source_name)
        if preflight.error:
            raise ExperimentValidationError(preflight.error)
        print(format_speed_profile(self.session.robot_config))
        print(format_action_sequence(preflight.actions))
        execution = self.session.run_experiment(source_name, real=self.real, confirmed=self.confirm_real, notes=f"Language instruction: {instruction}")
        _json_dump(output_dir / "execution_result.json", execution)
        manifest = {
            "created_at": _now(),
            "instruction": instruction,
            "skill": str(skill_path.relative_to(self.session.run_dir)),
            "molmo_result": str(molmo_path.relative_to(self.session.run_dir)) if molmo_path else None,
            "plan": str((output_dir / "execution_plan.json").relative_to(self.session.run_dir)),
            "experiment": str(source_path.relative_to(self.session.run_dir)),
            "execution_result": str((output_dir / "execution_result.json").relative_to(self.session.run_dir)),
            "physical_execution": bool(self.real),
            "execution_completed": bool(execution.get("execution_completed")),
        }
        _json_dump(output_dir / "pipeline.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--instruction", required=True, help="language instruction, e.g. '抓住袖子往外移动'")
    run = parser.add_mutually_exclusive_group()
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--run-id")
    parser.add_argument("--robot-config", type=Path)
    parser.add_argument("--perception-config", type=Path)
    parser.add_argument("--skip-capture", action="store_true", help="reuse workspace/perception_views from an existing run")
    parser.add_argument("--use-molmo", action="store_true", help="run the skill-selected MolmoPoint query")
    parser.add_argument("--molmo-python", type=Path)
    parser.add_argument("--molmo-model", default="allenai/MolmoPoint-8B")
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--claude-timeout-s", type=int, default=400)
    parser.add_argument("--real", action="store_true", help="allow physical xArm execution")
    parser.add_argument("--confirm-real", action="store_true", help="confirm the printed action sequence")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    try:
        session = _load_or_create_session(
            root,
            run_dir=args.run_dir,
            run_id=args.run_id,
            robot_config=args.robot_config.resolve() if args.robot_config else None,
            instruction=args.instruction,
        )
        pipeline = LanguageSkillPipeline(
            session=session,
            instruction=args.instruction,
            project_root=root,
            perception_config=args.perception_config.resolve() if args.perception_config else None,
            capture=not args.skip_capture,
            use_molmo=args.use_molmo,
            molmo_python=args.molmo_python.resolve() if args.molmo_python else None,
            molmo_model=args.molmo_model,
            claude_binary=args.claude_binary,
            claude_timeout_s=args.claude_timeout_s,
            real=args.real,
            confirm_real=args.confirm_real,
        )
        pipeline.run()
        return 0
    except (LanguageSkillPipelineError, ExperimentValidationError, ValueError, FileNotFoundError, PermissionError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
