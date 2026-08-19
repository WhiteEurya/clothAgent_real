"""Claude strategy/action/evaluation client for the semantic-anchor pipeline."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .free_exploration import (
    ExplorationProposal,
    _json_from_claude_text,
    validate_exploration_payload,
)
from .semantic_pipeline import (
    GRASP_FEATURES,
    SEMANTIC_EVALUATION_JSON_SCHEMA,
    ActionScope,
    HypothesisBudgetDecision,
    SemanticEvaluation,
    SemanticPipelineError,
    SemanticStrategy,
    validate_action_scope,
    validate_semantic_evaluation_payload,
    validate_semantic_strategy_payload,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


SEMANTIC_STRATEGY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "semantic_objective": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_part": {"type": "string", "minLength": 1},
                "desired_change": {"type": "string", "minLength": 1},
            },
            "required": ["target_part", "desired_change"],
        },
        "hypothesis": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "state": {"type": "string", "minLength": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string", "minLength": 1},
            },
            "required": ["state", "confidence", "rationale"],
        },
        "local_search_region": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "around_anchor_id": {"type": "string", "pattern": "^S[0-9]{3}$"},
                "radius_px": {"type": "integer", "minimum": 20, "maximum": 160},
                "include_connected_fold_edge": {"type": "boolean"},
            },
            "required": [
                "around_anchor_id",
                "radius_px",
                "include_connected_fold_edge",
            ],
        },
        "grasp_requirement": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "prefer": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"type": "string", "enum": sorted(GRASP_FEATURES)},
                },
                "avoid": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"type": "string", "enum": sorted(GRASP_FEATURES)},
                },
            },
            "required": ["prefer", "avoid"],
        },
        "expected_semantic_observation": {"type": "string", "minLength": 1},
        "safety_notes": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
    },
    "required": [
        "semantic_objective",
        "hypothesis",
        "local_search_region",
        "grasp_requirement",
        "expected_semantic_observation",
        "safety_notes",
    ],
}


SEMANTIC_ACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selected_candidate_id": {"type": "string", "pattern": "^R[0-9]{3}$"},
        "garment_observation": {"type": "string", "minLength": 1},
        "reveal_strategy": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "actions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": ["move", "open_gripper", "close_gripper", "home"],
                    },
                    "args": {"type": "object"},
                },
                "required": ["name", "args"],
            },
        },
        "expected_observation": {"type": "string", "minLength": 1},
        "safety_notes": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "skill_invocations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "enum": ["laydown"]},
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": ["name", "reason"],
            },
        },
    },
    "required": [
        "selected_candidate_id",
        "garment_observation",
        "reveal_strategy",
        "confidence",
        "actions",
        "expected_observation",
        "safety_notes",
    ],
}


@dataclass(frozen=True)
class SemanticActionResult:
    selected_candidate_id: str
    proposal: ExplorationProposal
    scope_validation: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "proposal": self.proposal.as_dict(),
            "scope_validation": dict(self.scope_validation),
        }


class SemanticClaudeClient:
    """One strategy decision, one local action proposal, one staged evaluation."""

    def __init__(
        self,
        binary: str = "claude",
        *,
        strategy_timeout_s: int = 400,
        action_timeout_s: int = 180,
        evaluation_timeout_s: int = 400,
    ):
        self.binary = binary
        self.strategy_timeout_s = int(strategy_timeout_s)
        self.action_timeout_s = int(action_timeout_s)
        self.evaluation_timeout_s = int(evaluation_timeout_s)
        self.last_strategy_log: dict[str, Any] | None = None
        self.last_action_log: dict[str, Any] | None = None
        self.last_evaluation_log: dict[str, Any] | None = None

    def _binary(self) -> str:
        if Path(self.binary).name != self.binary:
            resolved = Path(self.binary).expanduser().resolve()
            if resolved.is_file():
                return str(resolved)
        located = shutil.which(self.binary)
        if located is None:
            raise SemanticPipelineError(f"Claude CLI not found: {self.binary}")
        return located

    @staticmethod
    def _save_log(run_dir: Path, stage: str, payload: Mapping[str, Any]) -> None:
        output = Path(run_dir).resolve() / "results" / "claude_semantic"
        output.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        (output / f"{stamp}_{stage}.json").write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _invoke_json(
        self,
        *,
        run_dir: Path,
        stage: str,
        prompt: str,
        schema: Mapping[str, Any],
        timeout_s: int,
        images: Sequence[Path] = (),
        validator: Callable[[Any], Any],
    ) -> tuple[Any, dict[str, Any]]:
        root = Path(run_dir).resolve()
        safe_images: list[Path] = []
        for raw in images:
            path = Path(raw).resolve()
            if not path.is_file():
                raise SemanticPipelineError(
                    f"semantic Claude image does not exist: {path}"
                )
            safe_images.append(path)
        image_text = "\n".join(f"- {path}" for path in safe_images)
        base_prompt = prompt
        if safe_images:
            base_prompt += f"\n\nImages available for Read:\n{image_text}"
        last_error: BaseException | None = None
        for correction in range(2):
            current_prompt = base_prompt
            if correction and last_error is not None:
                current_prompt += (
                    "\n\nONE HARD-CONTRACT CORRECTION IS ALLOWED. The previous response "
                    f"failed deterministic validation: {type(last_error).__name__}: "
                    f"{last_error}. Correct only that contract error and return the full JSON."
                )
            allowed_dirs = [root, *[path.parent for path in safe_images]]
            unique_allowed_dirs = list(
                dict.fromkeys(str(path) for path in allowed_dirs)
            )
            command = [
                self._binary(),
                "--print",
                current_prompt,
                "--output-format",
                "json",
                "--json-schema",
                json.dumps(schema, separators=(",", ":")),
                "--permission-mode",
                "plan",
                "--allowedTools",
                "Read" if safe_images else "",
                "--tools",
                "Read" if safe_images else "",
                "--add-dir",
                *unique_allowed_dirs,
                "--safe-mode",
                "--no-session-persistence",
                "--system-prompt",
                (
                    "You are one stage in a validation-first garment robotics pipeline. "
                    "Return only the requested structured JSON. Do not write files, execute "
                    "commands, call robot APIs, invent measurements, or exceed the supplied "
                    "capabilities/action authority. Semantic identities are hypotheses."
                ),
            ]
            started = time.monotonic()
            completed = subprocess.run(
                command,
                cwd=root,
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
                shell=False,
            )
            duration = time.monotonic() - started
            log = {
                "stage": stage,
                "created_at": _now(),
                "correction": correction,
                "prompt": current_prompt,
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "duration_s": duration,
            }
            if completed.returncode != 0:
                self._save_log(root, f"{stage}_failed", log)
                raise SemanticPipelineError(
                    f"Claude {stage} exited with {completed.returncode}: "
                    f"{completed.stderr.strip() or completed.stdout.strip()}"
                )
            try:
                validated = validator(_json_from_claude_text(completed.stdout))
            except BaseException as exc:
                last_error = exc
                log["validation_error"] = f"{type(exc).__name__}: {exc}"
                self._save_log(root, f"{stage}_invalid", log)
                if correction:
                    raise SemanticPipelineError(
                        f"Claude {stage} failed its single hard-contract correction: {exc}"
                    ) from exc
                continue
            log["validated"] = (
                validated.as_dict() if hasattr(validated, "as_dict") else validated
            )
            self._save_log(root, stage, log)
            return validated, log
        raise SemanticPipelineError(f"Claude {stage} ended without a valid response")

    def plan_strategy(
        self,
        *,
        images: Sequence[Path],
        run_dir: Path,
        semantic_state: Mapping[str, Any],
        experiences: Sequence[Mapping[str, Any]],
        budget: HypothesisBudgetDecision,
    ) -> SemanticStrategy:
        anchors = semantic_state.get("known", {}).get("anchors", [])
        prompt = (
            "Choose the single garment relation to change next. Molmo Sxxx entries are "
            "high-confidence semantic anchors only; they are NOT grasp points. Do not emit "
            "Rxxx, pixels as grasp commands, robot coordinates, or actions. Form one explicit "
            "hypothesis, select its Sxxx region, and state local geometric grasp requirements. "
            "Prefer a semantic relation such as moving an inward sleeve outward from the torso, "
            "not a generic goal such as reducing the highest ridge. Treat all semantic identity "
            "and relation fields as hypotheses.\n\n"
            "Available measurements: anchor type/pixel/base location/confidence, garment centroid, "
            "relative-to-centroid relation, near-centroid/overlap hints, RGB, garment height, "
            "boundary, and height-gradient images. No graspability measurement exists yet.\n"
            "Available capability in this stage: choose semantic objective and define local search "
            "requirements only.\n\n"
            f"Semantic state:\n{json.dumps(semantic_state, ensure_ascii=False, indent=2)}\n\n"
            f"Available anchors:\n{json.dumps(anchors, ensure_ascii=False, indent=2)}\n\n"
            f"Hypothesis budget decision:\n{json.dumps(budget.as_dict(), ensure_ascii=False, indent=2)}\n\n"
            "Structured experience history (coordinate-independent):\n"
            f"{json.dumps(list(experiences)[-8:], ensure_ascii=False, indent=2)}"
        )

        def validate(payload: Any) -> SemanticStrategy:
            strategy = validate_semantic_strategy_payload(
                payload,
                semantic_state=semantic_state,
                forced_anchor_id=budget.forced_anchor_id,
            )
            persistent_dispositions = {
                "KEEP_SEMANTIC_CHANGE_GRASP",
                "KEEP_TARGET_AND_GRASP_CHECK_STRUCTURE",
                "KEEP_TARGET_AND_GRASP_CHANGE_TRANSPORT",
                "KEEP_HYPOTHESIS",
            }
            if (
                budget.disposition in persistent_dispositions
                and budget.hypothesis_key is not None
                and strategy.hypothesis_key != budget.hypothesis_key
            ):
                raise SemanticPipelineError(
                    "the runtime budget requires the existing semantic hypothesis "
                    f"{budget.hypothesis_key}; got {strategy.hypothesis_key}"
                )
            if (
                budget.disposition == "ESCAPE_HYPOTHESIS"
                and budget.hypothesis_key is not None
                and strategy.hypothesis_key == budget.hypothesis_key
            ):
                raise SemanticPipelineError(
                    f"semantic hypothesis {budget.hypothesis_key} exhausted its budget; "
                    "select a different supported anchor/hypothesis"
                )
            return strategy

        strategy, log = self._invoke_json(
            run_dir=run_dir,
            stage="semantic_strategy",
            prompt=prompt,
            schema=SEMANTIC_STRATEGY_JSON_SCHEMA,
            timeout_s=self.strategy_timeout_s,
            images=images,
            validator=validate,
        )
        self.last_strategy_log = log
        return strategy

    def propose_action(
        self,
        *,
        run_dir: Path,
        strategy: SemanticStrategy,
        local_geometry: Mapping[str, Any],
        scope: ActionScope,
        robot_context: Mapping[str, Any],
        overlay_image: Path | None = None,
    ) -> SemanticActionResult:
        candidates = local_geometry.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise SemanticPipelineError(
                "action planning requires local Rxxx candidates"
            )
        candidate_map = {str(item["reference_id"]): item for item in candidates}
        prompt = (
            "Select exactly one local Rxxx geometry candidate and propose exactly one normal "
            "action program. Rxxx exists only inside the already selected semantic region. "
            "Do not reconsider the semantic target and do not treat Molmo Sxxx as a grasp. "
            "The runtime, not you, chose the action scope. You may not request repeated probes, "
            "change scope, emit joint angles, call SDK/IK, or invent measurements. The grasp move "
            "immediately before close_gripper must use the selected candidate's exact base X/Y. "
            "Emit exactly one close_gripper/release cycle and no more post-grasp move targets than "
            "the runtime scope permits. Always release before the program ends.\n\n"
            "Available measurements: candidate base_xyz_mm, height_above_table_mm, free_boundary, "
            "height_step_mm, relief_above_local_median_mm, local_gradient_mm_per_px, suggested_yaw, "
            "and graspability_score. Token/semantic confidence is not graspability.\n"
            "Capabilities: move(x,y,z,yaw), open_gripper(), close_gripper(), home().\n\n"
            f"Semantic strategy:\n{json.dumps(strategy.as_dict(), ensure_ascii=False, indent=2)}\n\n"
            f"Runtime action scope:\n{json.dumps(scope.as_dict(), ensure_ascii=False, indent=2)}\n\n"
            f"Local geometry candidates:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
            f"Robot context:\n{json.dumps(robot_context, ensure_ascii=False, indent=2)}"
        )

        def validate(payload: Any) -> SemanticActionResult:
            if not isinstance(payload, dict):
                raise SemanticPipelineError("semantic action must be an object")
            selected = payload.get("selected_candidate_id")
            if not isinstance(selected, str) or selected not in candidate_map:
                raise SemanticPipelineError(
                    f"selected_candidate_id must be one of {sorted(candidate_map)}"
                )
            proposal_payload = dict(payload)
            proposal_payload.pop("selected_candidate_id", None)
            proposal = validate_exploration_payload(proposal_payload)
            scope_result = validate_action_scope(
                proposal.actions,
                candidate=candidate_map[selected],
                scope=scope,
            )
            return SemanticActionResult(selected, proposal, scope_result)

        images = [overlay_image] if overlay_image is not None else []
        result, log = self._invoke_json(
            run_dir=run_dir,
            stage="semantic_action",
            prompt=prompt,
            schema=SEMANTIC_ACTION_JSON_SCHEMA,
            timeout_s=self.action_timeout_s,
            images=images,
            validator=validate,
        )
        self.last_action_log = log
        return result

    def evaluate(
        self,
        *,
        before_images: Sequence[Path],
        after_images: Sequence[Path],
        run_dir: Path,
        semantic_state: Mapping[str, Any],
        strategy: SemanticStrategy,
        candidate: Mapping[str, Any],
        action_result: SemanticActionResult,
    ) -> SemanticEvaluation:
        prompt = (
            "Evaluate one semantic garment action stage by stage using only visible evidence. "
            "Do not infer success merely from commanded actions. The semantic target is a "
            "hypothesis. Distinguish: semantic_target (was the hypothesized part/relation "
            "supported), grasp_acquisition, structure_engagement (did the intended associated "
            "structure move), opening_relevance (did controlling it actually help unfolding), "
            "transport, laydown, and task_progress. If static before/after images cannot prove "
            "acquisition or engagement, return UNKNOWN. Keep supported upstream stages and change "
            "only the earliest failed stage/downstream consequences.\n\n"
            "Available measurements: visible area/overlap/relief/boundary changes when directly "
            "observable. INCREASED/DECREASED/UNCHANGED/UNKNOWN are valid when no exact measurement "
            "is supplied. No other measurement names are available.\n\n"
            f"Semantic state:\n{json.dumps(semantic_state, ensure_ascii=False, indent=2)}\n\n"
            f"Strategy:\n{json.dumps(strategy.as_dict(), ensure_ascii=False, indent=2)}\n\n"
            f"Chosen local structure:\n{json.dumps(dict(candidate), ensure_ascii=False, indent=2)}\n\n"
            f"Action:\n{json.dumps(action_result.as_dict(), ensure_ascii=False, indent=2)}"
        )
        evaluation, log = self._invoke_json(
            run_dir=run_dir,
            stage="semantic_evaluation",
            prompt=prompt,
            schema=SEMANTIC_EVALUATION_JSON_SCHEMA,
            timeout_s=self.evaluation_timeout_s,
            images=[*before_images, *after_images],
            validator=validate_semantic_evaluation_payload,
        )
        self.last_evaluation_log = log
        return evaluation
