"""Semantic-anchor garment opening contracts and deterministic local grounding.

The semantic pipeline deliberately keeps three concepts separate:

* Molmo ``Sxxx`` anchors are uncertain semantic observations, never grasp points.
* A semantic strategy states which garment relation should change.
* Local ``Rxxx`` references are geometry-derived grasp candidates scoped to one
  selected semantic region.

All schemas in this module are strict so deterministic failures are rejected
before another Claude call or physical action is attempted.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


class SemanticPipelineError(RuntimeError):
    """Raised when a semantic/geometry/action contract is invalid."""


SEMANTIC_STRATEGY_FIELDS = frozenset(
    {
        "semantic_objective",
        "hypothesis",
        "local_search_region",
        "grasp_requirement",
        "expected_semantic_observation",
        "safety_notes",
    }
)
SEMANTIC_OBJECTIVE_FIELDS = frozenset({"target_part", "desired_change"})
SEMANTIC_HYPOTHESIS_FIELDS = frozenset({"state", "confidence", "rationale"})
LOCAL_SEARCH_FIELDS = frozenset(
    {"around_anchor_id", "radius_px", "include_connected_fold_edge"}
)
GRASP_REQUIREMENT_FIELDS = frozenset({"prefer", "avoid"})

GRASP_FEATURES = frozenset(
    {
        "free_edge",
        "raised_fold_edge",
        "discrete_height_step",
        "interior_ridge",
        "flat_interior",
        "unrelated_height_peak",
    }
)
STRATEGY_LOW_LEVEL_PATTERN = re.compile(
    r"(?:\bR\d{3}\b|\bbase_xyz_mm\b|\bpixel_xy\b|\bjoint_angles?\b|"
    r"\bsdk_call\b|\bik_solution\b|\bmove\s*\(|\bclose_gripper\b|"
    r"\bopen_gripper\b)",
    re.IGNORECASE,
)

SEMANTIC_STAGE_STATUSES: dict[str, frozenset[str]] = {
    "semantic_target": frozenset({"SUPPORTED", "CONTRADICTED", "UNKNOWN"}),
    "grasp_acquisition": frozenset({"SUCCESS", "FAILURE", "UNKNOWN"}),
    "structure_engagement": frozenset({"SUCCESS", "FAILURE", "UNKNOWN"}),
    "opening_relevance": frozenset({"SUPPORTED", "CONTRADICTED", "UNKNOWN"}),
    "transport": frozenset(
        {"GOOD", "BAD_DIRECTION", "INSUFFICIENT", "OVERPULL", "UNKNOWN"}
    ),
    "laydown": frozenset({"SUCCESS", "FAILURE", "NOT_REACHED", "UNKNOWN"}),
}
SEMANTIC_FAILURE_STAGES = frozenset(
    {
        "SEMANTIC_TARGET",
        "ACQUISITION",
        "STRUCTURE_ENGAGEMENT",
        "OPENING_RELEVANCE",
        "TRANSPORT",
        "LAYDOWN",
        "NONE",
        "UNKNOWN",
    }
)
TASK_PROGRESS_STATUSES = frozenset({"IMPROVED", "NEUTRAL", "REGRESSED"})
METRIC_DELTAS = frozenset({"INCREASED", "DECREASED", "UNCHANGED", "UNKNOWN"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _strict_fields(
    value: Any, expected: frozenset[str], *, context: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SemanticPipelineError(f"{context} must be an object")
    missing = expected.difference(value)
    unknown = set(value).difference(expected)
    if missing:
        raise SemanticPipelineError(f"{context} is missing fields: {sorted(missing)}")
    if unknown:
        raise SemanticPipelineError(f"{context} has unknown fields: {sorted(unknown)}")
    return value


def _text(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticPipelineError(f"{context} must be a non-empty string")
    return value.strip()


def _confidence(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticPipelineError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SemanticPipelineError(f"{context} must be within [0, 1]")
    return result


def _string_list(
    value: Any,
    *,
    context: str,
    allowed: frozenset[str] | None = None,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise SemanticPipelineError(f"{context} must be {qualifier}")
    result: list[str] = []
    for index, raw in enumerate(value):
        item = _text(raw, context=f"{context}[{index}]")
        if allowed is not None and item not in allowed:
            raise SemanticPipelineError(
                f"{context}[{index}] must be one of {sorted(allowed)}"
            )
        if item in result:
            raise SemanticPipelineError(f"{context} must not contain duplicates")
        result.append(item)
    return tuple(result)


def _strategy_text(value: Any, *, context: str) -> str:
    result = _text(value, context=context)
    match = STRATEGY_LOW_LEVEL_PATTERN.search(result)
    if match is not None:
        raise SemanticPipelineError(
            f"{context} contains forbidden grasp/action/coordinate detail: {match.group(0)!r}"
        )
    return result


@dataclass(frozen=True)
class SemanticStrategy:
    target_part: str
    desired_change: str
    hypothesis_state: str
    hypothesis_confidence: float
    hypothesis_rationale: str
    anchor_id: str
    radius_px: int
    include_connected_fold_edge: bool
    prefer: tuple[str, ...]
    avoid: tuple[str, ...]
    expected_semantic_observation: str
    safety_notes: tuple[str, ...]

    @property
    def hypothesis_key(self) -> str:
        return f"{self.target_part}:{self.hypothesis_state}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "semantic_objective": {
                "target_part": self.target_part,
                "desired_change": self.desired_change,
            },
            "hypothesis": {
                "state": self.hypothesis_state,
                "confidence": self.hypothesis_confidence,
                "rationale": self.hypothesis_rationale,
            },
            "local_search_region": {
                "around_anchor_id": self.anchor_id,
                "radius_px": self.radius_px,
                "include_connected_fold_edge": self.include_connected_fold_edge,
            },
            "grasp_requirement": {
                "prefer": list(self.prefer),
                "avoid": list(self.avoid),
            },
            "expected_semantic_observation": self.expected_semantic_observation,
            "safety_notes": list(self.safety_notes),
        }


def validate_semantic_strategy_payload(
    payload: Any,
    *,
    semantic_state: Mapping[str, Any],
    forced_anchor_id: str | None = None,
) -> SemanticStrategy:
    value = _strict_fields(
        payload, SEMANTIC_STRATEGY_FIELDS, context="semantic strategy"
    )
    objective = _strict_fields(
        value["semantic_objective"],
        SEMANTIC_OBJECTIVE_FIELDS,
        context="semantic strategy.semantic_objective",
    )
    hypothesis = _strict_fields(
        value["hypothesis"],
        SEMANTIC_HYPOTHESIS_FIELDS,
        context="semantic strategy.hypothesis",
    )
    search = _strict_fields(
        value["local_search_region"],
        LOCAL_SEARCH_FIELDS,
        context="semantic strategy.local_search_region",
    )
    requirement = _strict_fields(
        value["grasp_requirement"],
        GRASP_REQUIREMENT_FIELDS,
        context="semantic strategy.grasp_requirement",
    )
    anchors = {
        str(item.get("anchor_id")): item
        for item in semantic_state.get("known", {}).get("anchors", [])
        if isinstance(item, dict) and item.get("anchor_id")
    }
    anchor_id = _text(
        search["around_anchor_id"],
        context="semantic strategy.local_search_region.around_anchor_id",
    )
    if anchor_id not in anchors:
        raise SemanticPipelineError(
            f"semantic strategy selected unknown anchor {anchor_id!r}; "
            f"available={sorted(anchors)}"
        )
    if forced_anchor_id is not None and anchor_id != forced_anchor_id:
        raise SemanticPipelineError(
            f"semantic budget requires anchor {forced_anchor_id}, got {anchor_id}"
        )
    target_part = _text(
        objective["target_part"],
        context="semantic strategy.semantic_objective.target_part",
    )
    anchor_type = str(anchors[anchor_id].get("type", ""))
    if target_part != anchor_type:
        raise SemanticPipelineError(
            "semantic objective target_part must match the selected anchor type: "
            f"target_part={target_part!r}, anchor_type={anchor_type!r}"
        )
    radius = search["radius_px"]
    if (
        isinstance(radius, bool)
        or not isinstance(radius, int)
        or not 20 <= radius <= 160
    ):
        raise SemanticPipelineError(
            "semantic strategy local-search radius_px must be an integer in [20, 160]"
        )
    connected = search["include_connected_fold_edge"]
    if not isinstance(connected, bool):
        raise SemanticPipelineError(
            "semantic strategy include_connected_fold_edge must be boolean"
        )
    prefer = _string_list(
        requirement["prefer"],
        context="semantic strategy.grasp_requirement.prefer",
        allowed=GRASP_FEATURES,
    )
    avoid = _string_list(
        requirement["avoid"],
        context="semantic strategy.grasp_requirement.avoid",
        allowed=GRASP_FEATURES,
        allow_empty=True,
    )
    overlap = set(prefer).intersection(avoid)
    if overlap:
        raise SemanticPipelineError(
            f"semantic strategy cannot both prefer and avoid {sorted(overlap)}"
        )
    return SemanticStrategy(
        target_part=target_part,
        desired_change=_strategy_text(
            objective["desired_change"],
            context="semantic strategy.semantic_objective.desired_change",
        ),
        hypothesis_state=_strategy_text(
            hypothesis["state"], context="semantic strategy.hypothesis.state"
        ),
        hypothesis_confidence=_confidence(
            hypothesis["confidence"],
            context="semantic strategy.hypothesis.confidence",
        ),
        hypothesis_rationale=_strategy_text(
            hypothesis["rationale"],
            context="semantic strategy.hypothesis.rationale",
        ),
        anchor_id=anchor_id,
        radius_px=radius,
        include_connected_fold_edge=connected,
        prefer=prefer,
        avoid=avoid,
        expected_semantic_observation=_strategy_text(
            value["expected_semantic_observation"],
            context="semantic strategy.expected_semantic_observation",
        ),
        safety_notes=tuple(
            _strategy_text(note, context=f"semantic strategy.safety_notes[{index}]")
            for index, note in enumerate(
                _string_list(
                    value["safety_notes"], context="semantic strategy.safety_notes"
                )
            )
        ),
    )


class SemanticStateBuilder:
    """Build a light, explicitly uncertain relation state from Molmo anchors."""

    def build(
        self,
        anchor_manifest: Mapping[str, Any],
        perception_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        if anchor_manifest.get("status") != "READY":
            raise SemanticPipelineError(
                "semantic state requires at least one high-confidence anchor"
            )
        raw_anchors = anchor_manifest.get("anchors")
        if not isinstance(raw_anchors, list) or not raw_anchors:
            raise SemanticPipelineError("semantic anchor manifest has no anchors")
        center = perception_result.get("center_base_mm")
        if (
            not isinstance(center, (list, tuple))
            or len(center) < 2
            or any(not isinstance(value, (int, float)) for value in center[:2])
        ):
            raise SemanticPipelineError(
                "semantic state requires a calibrated garment center"
            )
        center_xy = np.asarray(center[:2], dtype=np.float64)
        known: list[dict[str, Any]] = []
        relations: dict[str, Any] = {}
        hypotheses: list[dict[str, Any]] = []
        for raw in raw_anchors:
            if not isinstance(raw, dict):
                raise SemanticPipelineError("semantic anchor must be an object")
            anchor_id = _text(raw.get("anchor_id"), context="semantic anchor.anchor_id")
            semantic_type = _text(
                raw.get("type"), context=f"semantic anchor {anchor_id}.type"
            )
            confidence = _confidence(
                raw.get("confidence"), context=f"semantic anchor {anchor_id}.confidence"
            )
            base = raw.get("base_xyz_mm")
            if (
                not isinstance(base, list)
                or len(base) != 3
                or any(not isinstance(value, (int, float)) for value in base)
            ):
                raise SemanticPipelineError(
                    f"semantic anchor {anchor_id} needs calibrated base_xyz_mm"
                )
            delta = np.asarray(base[:2], dtype=np.float64) - center_xy
            distance = float(np.linalg.norm(delta))
            if abs(float(delta[0])) >= abs(float(delta[1])):
                direction = "positive_x" if delta[0] >= 0 else "negative_x"
            else:
                direction = "positive_y" if delta[1] >= 0 else "negative_y"
            near_centroid = distance <= 120.0
            local_height = float(raw.get("height_above_table_mm", 0.0))
            spread = float(raw.get("local_base_z_spread_mm", 0.0))
            near_overlap = bool(
                ("sleeve" in semantic_type or "hem" in semantic_type)
                and near_centroid
                and (local_height >= 8.0 or spread >= 3.0)
            )
            known_anchor = {
                "anchor_id": anchor_id,
                "type": semantic_type,
                "camera": str(raw.get("camera")),
                "pixel_xy": list(raw.get("pixel_xy", [])),
                "base_xyz_mm": [float(value) for value in base],
                "confidence": confidence,
                "semantic_identity_status": "HYPOTHESIS",
            }
            known.append(known_anchor)
            relations[anchor_id] = {
                "relative_to_centroid": direction,
                "distance_to_centroid_mm": distance,
                "near_centroid": near_centroid,
                "near_torso_overlap": near_overlap,
            }
            if "sleeve" in semantic_type and near_overlap:
                hypotheses.append(
                    {
                        "anchor_id": anchor_id,
                        "target_part": semantic_type,
                        "state": "possible_inward_fold",
                        "confidence": min(confidence, 0.78),
                        "basis": (
                            "Sleeve-associated anchor lies near the garment centroid "
                            "with local height/relief; semantic identity and fold state "
                            "remain hypotheses, not ground truth."
                        ),
                    }
                )
        return {
            "schema_version": 1,
            "created_at": _now(),
            "known": {
                "anchors": known,
                "garment_centroid_base_mm": [float(value) for value in center[:3]],
            },
            "relations": relations,
            "hypotheses": hypotheses,
            "uncertainty_policy": (
                "Anchor identity and inferred garment relations are hypotheses. "
                "They must be evaluated after action and must not be treated as grasp points."
            ),
        }


@dataclass(frozen=True)
class ActionScope:
    name: str
    max_lateral_mm: float
    max_lift_mm: float
    max_post_grasp_moves: int
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_lateral_mm": self.max_lateral_mm,
            "max_lift_mm": self.max_lift_mm,
            "max_post_grasp_moves": self.max_post_grasp_moves,
            "description": self.description,
        }


ACTION_SCOPES: dict[str, ActionScope] = {
    "ACQUISITION_CHECK": ActionScope(
        "ACQUISITION_CHECK",
        max_lateral_mm=5.0,
        max_lift_mm=20.0,
        max_post_grasp_moves=1,
        description="Short lift only; acquisition is not yet supported.",
    ),
    "STRUCTURE_CHECK": ActionScope(
        "STRUCTURE_CHECK",
        max_lateral_mm=30.0,
        max_lift_mm=35.0,
        max_post_grasp_moves=1,
        description="Small reversible motion to verify the intended structure moves.",
    ),
    "TRANSPORT_TEST": ActionScope(
        "TRANSPORT_TEST",
        max_lateral_mm=60.0,
        max_lift_mm=50.0,
        max_post_grasp_moves=2,
        description="Moderate transport after acquisition/engagement are supported.",
    ),
    "TRANSPORT_CORRECTION": ActionScope(
        "TRANSPORT_CORRECTION",
        max_lateral_mm=80.0,
        max_lift_mm=55.0,
        max_post_grasp_moves=2,
        description=(
            "Keep the supported target/grasp family and correct transport direction/profile."
        ),
    ),
    "OPENING_COMPLETION": ActionScope(
        "OPENING_COMPLETION",
        max_lateral_mm=140.0,
        max_lift_mm=80.0,
        max_post_grasp_moves=3,
        description="Larger completion stroke after opening relevance is supported.",
    ),
}


def action_scope_from_experiences(
    experiences: Sequence[Mapping[str, Any]],
    *,
    hypothesis_key: str | None = None,
    budget_disposition: str | None = None,
) -> ActionScope:
    if budget_disposition == "KEEP_SEMANTIC_CHANGE_GRASP":
        return ACTION_SCOPES["ACQUISITION_CHECK"]
    latest: Mapping[str, Any] | None = None
    for item in reversed(list(experiences)):
        if hypothesis_key is not None and item.get("hypothesis_key") != hypothesis_key:
            continue
        evaluation = item.get("evaluation")
        if isinstance(evaluation, Mapping):
            latest = evaluation
            break
    if latest is None:
        return ACTION_SCOPES["ACQUISITION_CHECK"]
    acquisition = (latest.get("grasp_acquisition") or {}).get("status")
    engagement = (latest.get("structure_engagement") or {}).get("status")
    relevance = (latest.get("opening_relevance") or {}).get("status")
    transport = (latest.get("transport") or {}).get("status")
    if acquisition != "SUCCESS":
        return ACTION_SCOPES["ACQUISITION_CHECK"]
    if engagement != "SUCCESS":
        return ACTION_SCOPES["STRUCTURE_CHECK"]
    if transport in {"BAD_DIRECTION", "INSUFFICIENT", "OVERPULL"}:
        return ACTION_SCOPES["TRANSPORT_CORRECTION"]
    if relevance != "SUPPORTED":
        return ACTION_SCOPES["TRANSPORT_TEST"]
    return ACTION_SCOPES["OPENING_COMPLETION"]


@dataclass(frozen=True)
class HypothesisBudgetDecision:
    hypothesis_key: str | None
    acquisition_attempts: int
    transport_attempts: int
    max_acquisition_attempts: int
    max_transport_attempts: int
    disposition: str
    forced_anchor_id: str | None
    forced_geometry_type: str | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_key": self.hypothesis_key,
            "acquisition_attempts": self.acquisition_attempts,
            "transport_attempts": self.transport_attempts,
            "max_acquisition_attempts": self.max_acquisition_attempts,
            "max_transport_attempts": self.max_transport_attempts,
            "disposition": self.disposition,
            "forced_anchor_id": self.forced_anchor_id,
            "forced_geometry_type": self.forced_geometry_type,
            "reason": self.reason,
        }


def semantic_hypothesis_budget(
    experiences: Sequence[Mapping[str, Any]],
    *,
    hypothesis_key: str | None,
    anchor_id: str | None,
    max_acquisition_attempts: int = 2,
    max_transport_attempts: int = 3,
) -> HypothesisBudgetDecision:
    relevant = [
        item
        for item in experiences
        if hypothesis_key is not None and item.get("hypothesis_key") == hypothesis_key
    ]
    acquisition_attempts = 0
    transport_attempts = 0
    latest: Mapping[str, Any] | None = None
    latest_experience: Mapping[str, Any] | None = None
    for item in relevant:
        evaluation = item.get("evaluation")
        if not isinstance(evaluation, Mapping):
            continue
        latest_experience = item
        latest = evaluation
        acquisition = (evaluation.get("grasp_acquisition") or {}).get("status")
        engagement = (evaluation.get("structure_engagement") or {}).get("status")
        if acquisition != "SUCCESS" or engagement != "SUCCESS":
            acquisition_attempts += 1
        else:
            transport_attempts += 1
    if latest is None:
        return HypothesisBudgetDecision(
            hypothesis_key,
            acquisition_attempts,
            transport_attempts,
            max_acquisition_attempts,
            max_transport_attempts,
            "NEW_HYPOTHESIS",
            None,
            None,
            "No evaluated attempt exists for this semantic hypothesis.",
        )
    semantic_target = (latest.get("semantic_target") or {}).get("status")
    relevance = (latest.get("opening_relevance") or {}).get("status")
    acquisition = (latest.get("grasp_acquisition") or {}).get("status")
    engagement = (latest.get("structure_engagement") or {}).get("status")
    transport = (latest.get("transport") or {}).get("status")
    forced_geometry: str | None = None
    if anchor_id is None:
        disposition = "ESCAPE_HYPOTHESIS"
        reason = "The previous semantic target has no current high-confidence anchor."
        forced = None
    elif semantic_target == "CONTRADICTED":
        disposition = "ESCAPE_HYPOTHESIS"
        reason = "The semantic target hypothesis was contradicted."
        forced = None
    elif relevance == "CONTRADICTED":
        disposition = "ESCAPE_HYPOTHESIS"
        reason = "Opening relevance was contradicted."
        forced = None
    elif (
        acquisition != "SUCCESS" or engagement != "SUCCESS"
    ) and acquisition_attempts >= max_acquisition_attempts:
        disposition = "ESCAPE_HYPOTHESIS"
        reason = "Local acquisition/structure-engagement retry budget is exhausted."
        forced = None
    elif transport_attempts >= max_transport_attempts and transport != "GOOD":
        disposition = "ESCAPE_HYPOTHESIS"
        reason = "Transport-hypothesis budget is exhausted without opening progress."
        forced = None
    elif acquisition != "SUCCESS" or engagement == "FAILURE":
        disposition = "KEEP_SEMANTIC_CHANGE_GRASP"
        reason = (
            "Keep the semantic target but revise local grasp geometry because "
            "acquisition or intended-structure engagement is not supported."
        )
        forced = anchor_id
    elif engagement == "UNKNOWN":
        disposition = "KEEP_TARGET_AND_GRASP_CHECK_STRUCTURE"
        reason = (
            "Acquisition succeeded, so keep the grasp geometry family and use the "
            "runtime-owned structure-engagement check."
        )
        forced = anchor_id
        chosen = (
            latest_experience.get("chosen_structure", {})
            if isinstance(latest_experience, Mapping)
            else {}
        )
        raw_geometry = (
            chosen.get("geometry_type") if isinstance(chosen, Mapping) else None
        )
        if isinstance(raw_geometry, str) and raw_geometry in GRASP_FEATURES:
            forced_geometry = raw_geometry
    elif transport in {"BAD_DIRECTION", "INSUFFICIENT", "OVERPULL", "UNKNOWN"}:
        disposition = "KEEP_TARGET_AND_GRASP_CHANGE_TRANSPORT"
        reason = (
            "Acquisition/engagement evidence supports stage-local transport correction."
        )
        forced = anchor_id
        chosen = (
            latest_experience.get("chosen_structure", {})
            if isinstance(latest_experience, Mapping)
            else {}
        )
        raw_geometry = (
            chosen.get("geometry_type") if isinstance(chosen, Mapping) else None
        )
        if isinstance(raw_geometry, str) and raw_geometry in GRASP_FEATURES:
            forced_geometry = raw_geometry
    else:
        disposition = "KEEP_HYPOTHESIS"
        reason = "Current semantic hypothesis remains within its evidence budget."
        forced = anchor_id
    return HypothesisBudgetDecision(
        hypothesis_key,
        acquisition_attempts,
        transport_attempts,
        max_acquisition_attempts,
        max_transport_attempts,
        disposition,
        forced,
        forced_geometry,
        reason,
    )


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def _connected_to_seed(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Return 8-connected ``mask`` pixels reachable from ``seed`` pixels."""

    allowed = np.asarray(mask, dtype=bool)
    raw_seed = np.asarray(seed, dtype=bool)
    if allowed.shape != raw_seed.shape:
        raise ValueError("connected-component mask/seed shapes must match")
    selected = raw_seed & allowed
    output = np.zeros_like(allowed)
    queue: deque[tuple[int, int]] = deque(
        (int(y), int(x)) for y, x in np.argwhere(selected)
    )
    height, width = allowed.shape
    while queue:
        y, x = queue.popleft()
        if output[y, x]:
            continue
        output[y, x] = True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not dx and not dy:
                    continue
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and allowed[ny, nx]
                    and not output[ny, nx]
                ):
                    queue.append((ny, nx))
    return output


def _normalized(value: float, scale: float) -> float:
    return float(max(0.0, min(1.0, value / max(scale, 1e-6))))


class LocalGeometryGrounder:
    """Generate local Rxxx grasp candidates around one semantic Sxxx anchor."""

    def __init__(self, *, max_candidates: int = 6, minimum_spacing_px: int = 14):
        if not 1 <= max_candidates <= 12:
            raise ValueError("max_candidates must be between 1 and 12")
        if not 4 <= minimum_spacing_px <= 50:
            raise ValueError("minimum_spacing_px must be between 4 and 50")
        self.max_candidates = max_candidates
        self.minimum_spacing_px = minimum_spacing_px

    @staticmethod
    def _anchor(semantic_state: Mapping[str, Any], anchor_id: str) -> dict[str, Any]:
        for item in semantic_state.get("known", {}).get("anchors", []):
            if isinstance(item, dict) and item.get("anchor_id") == anchor_id:
                return item
        raise SemanticPipelineError(f"semantic state has no anchor {anchor_id}")

    def ground(
        self,
        *,
        perception_dir: Path,
        artifact_dir: Path,
        semantic_state: Mapping[str, Any],
        strategy: SemanticStrategy,
        install: bool = True,
    ) -> dict[str, Any]:
        perception = Path(perception_dir).resolve()
        output = Path(artifact_dir).resolve()
        if output.exists():
            raise SemanticPipelineError(
                f"local geometry output already exists: {output}"
            )
        output.mkdir(parents=True, exist_ok=False)
        anchor = self._anchor(semantic_state, strategy.anchor_id)
        camera = str(anchor["camera"]).upper()
        pixel = anchor.get("pixel_xy")
        if not isinstance(pixel, list) or len(pixel) != 2:
            raise SemanticPipelineError("semantic anchor has no valid pixel_xy")
        xyz_path = perception / f"camera_{camera}_base_xyz_mm.npy"
        height_path = perception / f"camera_{camera}_height_above_table_mm.npy"
        mask_path = perception / f"camera_{camera}_garment_mask.npy"
        image_paths = sorted(perception.glob(f"camera_*_{camera}.png"))
        if (
            not xyz_path.is_file()
            or not height_path.is_file()
            or not mask_path.is_file()
        ):
            raise SemanticPipelineError(
                f"Camera {camera} local grounding requires XYZ, height, and final garment mask"
            )
        if len(image_paths) != 1:
            raise SemanticPipelineError(
                f"Camera {camera} local grounding requires exactly one raw RGB image"
            )
        xyz = np.load(xyz_path, mmap_mode="r", allow_pickle=False)
        height = np.asarray(
            np.load(height_path, mmap_mode="r", allow_pickle=False), dtype=np.float64
        )
        garment_mask = np.asarray(
            np.load(mask_path, mmap_mode="r", allow_pickle=False), dtype=bool
        )
        if xyz.shape[:2] != height.shape or garment_mask.shape != height.shape:
            raise SemanticPipelineError("local geometry maps have incompatible shapes")
        valid = garment_mask & np.isfinite(height) & np.all(np.isfinite(xyz), axis=2)
        if not np.any(valid):
            raise SemanticPipelineError("local geometry has no finite garment surface")
        anchor_x = round(float(pixel[0]))
        anchor_y = round(float(pixel[1]))
        yy, xx = np.indices(height.shape)
        base_roi = (
            (xx - anchor_x) ** 2 + (yy - anchor_y) ** 2 <= strategy.radius_px**2
        ) & valid
        if int(np.count_nonzero(base_roi)) < 25:
            raise SemanticPipelineError(
                "semantic local-search region contains fewer than 25 valid garment pixels"
            )
        fill_value = float(np.median(height[base_roi]))
        filled = np.where(valid, height, fill_value)
        grad_y, grad_x = np.gradient(filled)
        gradient = np.hypot(grad_x, grad_y)
        try:
            from scipy.ndimage import maximum_filter, minimum_filter

            local_max = maximum_filter(filled, size=7, mode="nearest")
            local_min = minimum_filter(filled, size=7, mode="nearest")
        except ImportError:
            local_max = filled
            local_min = filled
        height_step = np.maximum(0.0, local_max - local_min)
        roi = base_roi.copy()
        connected_fold_pixels = 0
        if strategy.include_connected_fold_edge:
            # Follow only a fold-like structure that actually intersects the
            # anchor neighbourhood. The 1.75x cap prevents a noisy component
            # from turning a semantic-local search back into whole-garment
            # parsing.
            connected_cap = (
                (xx - anchor_x) ** 2 + (yy - anchor_y) ** 2
                <= (strategy.radius_px * 1.75) ** 2
            ) & valid
            gradient_cut = float(np.percentile(gradient[base_roi], 75.0))
            step_seed_cut = float(np.percentile(height_step[base_roi], 70.0))
            fold_like = connected_cap & (
                (gradient >= gradient_cut) | (height_step >= step_seed_cut)
            )
            connected = _connected_to_seed(fold_like, fold_like & base_roi)
            connected_fold_pixels = int(np.count_nonzero(connected & ~base_roi))
            roi |= connected
        boundary = _binary_boundary(garment_mask) & roi
        roi_heights = height[roi]
        height_median = float(np.median(roi_heights))
        grad_cut = float(np.percentile(gradient[roi], 75.0))
        step_cut = float(np.percentile(height_step[roi], 70.0))
        high_cut = float(np.percentile(roi_heights, 75.0))
        feature_masks = {
            "free_edge": boundary,
            "raised_fold_edge": roi
            & (gradient >= grad_cut)
            & (height >= height_median),
            "discrete_height_step": roi & (height_step >= step_cut),
            "interior_ridge": roi & ~boundary & (height >= high_cut),
        }
        feature_priority = list(strategy.prefer) + [
            item
            for item in (
                "free_edge",
                "raised_fold_edge",
                "discrete_height_step",
                "interior_ridge",
            )
            if item not in strategy.prefer
        ]
        raw_candidates: list[dict[str, Any]] = []
        seen_pixels: set[tuple[int, int]] = set()
        for feature in feature_priority:
            mask = feature_masks.get(feature)
            if mask is None or not np.any(mask):
                continue
            indices = np.argwhere(mask)
            feature_bonus = 1.0 if feature in strategy.prefer else 0.4
            scored: list[tuple[float, int, int]] = []
            for y_px, x_px in indices:
                edge_score = 1.0 if boundary[y_px, x_px] else 0.0
                step_score = _normalized(float(height_step[y_px, x_px]), 12.0)
                relief_score = _normalized(
                    max(0.0, float(height[y_px, x_px]) - height_median), 12.0
                )
                interior_penalty = 0.5 if not boundary[y_px, x_px] else 0.0
                avoid_penalty = 0.0
                if (
                    "flat_interior" in strategy.avoid
                    and step_score < 0.15
                    and not edge_score
                ):
                    avoid_penalty += 1.0
                if (
                    "unrelated_height_peak" in strategy.avoid
                    and feature == "interior_ridge"
                ):
                    avoid_penalty += 0.5
                score = (
                    0.35 * feature_bonus
                    + 0.30 * edge_score
                    + 0.25 * step_score
                    + 0.20 * relief_score
                    - 0.20 * interior_penalty
                    - 0.40 * avoid_penalty
                )
                scored.append((score, int(y_px), int(x_px)))
            scored.sort(reverse=True)
            for score, y_px, x_px in scored:
                key = (x_px, y_px)
                if key in seen_pixels:
                    continue
                if any(
                    (x_px - item["pixel_xy"][0]) ** 2
                    + (y_px - item["pixel_xy"][1]) ** 2
                    < self.minimum_spacing_px**2
                    for item in raw_candidates
                ):
                    continue
                point = np.asarray(xyz[y_px, x_px], dtype=np.float64)
                table_z = float(point[2] - height[y_px, x_px])
                tangent_yaw = math.degrees(
                    math.atan2(float(grad_y[y_px, x_px]), float(grad_x[y_px, x_px]))
                )
                raw_candidates.append(
                    {
                        "pixel_xy": [x_px, y_px],
                        "base_xyz_mm": [float(value) for value in point],
                        "table_z_mm": table_z,
                        "height_above_table_mm": float(height[y_px, x_px]),
                        "feature": feature,
                        "free_boundary": bool(boundary[y_px, x_px]),
                        "height_step_mm": float(height_step[y_px, x_px]),
                        "relief_above_local_median_mm": max(
                            0.0, float(height[y_px, x_px]) - height_median
                        ),
                        "local_gradient_mm_per_px": float(gradient[y_px, x_px]),
                        "suggested_yaw_deg": float(
                            max(-180.0, min(180.0, tangent_yaw))
                        ),
                        "graspability_score": float(score),
                    }
                )
                seen_pixels.add(key)
                if len(raw_candidates) >= self.max_candidates:
                    break
            if len(raw_candidates) >= self.max_candidates:
                break
        if not raw_candidates:
            raise SemanticPipelineError(
                "local geometry produced no candidate satisfying the grasp requirements"
            )
        raw_candidates.sort(
            key=lambda item: (-float(item["graspability_score"]), item["pixel_xy"])
        )
        candidates: list[dict[str, Any]] = []
        for index, item in enumerate(raw_candidates, start=1):
            candidates.append(
                {
                    "reference_id": f"R{index:03d}",
                    "semantic_anchor_id": strategy.anchor_id,
                    "target_part": strategy.target_part,
                    **item,
                }
            )
        image = Image.open(image_paths[0]).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            (anchor_x - 10, anchor_y - 10, anchor_x + 10, anchor_y + 10),
            outline=(255, 210, 0),
            width=3,
        )
        draw.text(
            (anchor_x + 12, anchor_y - 10),
            f"{strategy.anchor_id} {strategy.target_part}",
            fill=(255, 230, 20),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        for item in candidates:
            x_px, y_px = item["pixel_xy"]
            draw.ellipse(
                (x_px - 7, y_px - 7, x_px + 7, y_px + 7),
                fill=(30, 255, 90),
                outline=(0, 0, 0),
                width=2,
            )
            draw.text(
                (x_px + 9, y_px - 8),
                f"{item['reference_id']} {item['feature']} {item['graspability_score']:.2f}",
                fill=(255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
        overlay_name = f"camera_{camera}_local_grasp_candidates.png"
        image.save(output / overlay_name)
        guide = {
            "camera_label": camera,
            "coordinate_frame": "robot_base_mm",
            "measurement_kind": "semantic_region_local_geometry_grasp_candidate",
            "semantic_anchor_id": strategy.anchor_id,
            "target_part": strategy.target_part,
            "overlay_image": overlay_name,
            "reference_semantics": (
                "Rxxx references are geometry-derived grasp candidates inside the "
                "selected semantic-anchor region. They are not Molmo anchors."
            ),
            "samples": candidates,
        }
        guide_name = f"camera_{camera}_local_geometry_coordinate_guide.json"
        (output / guide_name).write_text(
            json.dumps(guide, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "created_at": _now(),
            "status": "READY",
            "semantic_anchor_id": strategy.anchor_id,
            "target_part": strategy.target_part,
            "camera": camera,
            "source_image": str(image_paths[0]),
            "semantic_anchor_pixel_xy": [anchor_x, anchor_y],
            "search_radius_px": strategy.radius_px,
            "include_connected_fold_edge": strategy.include_connected_fold_edge,
            "connected_fold_pixels_outside_radius": connected_fold_pixels,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "overlay": str(output / overlay_name),
            "coordinate_guide": str(output / guide_name),
        }
        (output / "local_geometry_candidates.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if install:
            for label in ("A", "B"):
                canonical = perception / f"camera_{label}_coordinate_guide.json"
                backup = perception / f"camera_{label}_uniform_coordinate_guide.json"
                if canonical.is_file() and not backup.exists():
                    shutil.copy2(canonical, backup)
                if label == camera:
                    shutil.copy2(output / guide_name, canonical)
                    shutil.copy2(output / overlay_name, perception / overlay_name)
                elif canonical.is_file():
                    canonical.write_text(
                        json.dumps(
                            {
                                "camera_label": label,
                                "coordinate_frame": "robot_base_mm",
                                "measurement_kind": "semantic_region_local_geometry_grasp_candidate",
                                "semantic_anchor_id": strategy.anchor_id,
                                "reference_semantics": (
                                    "No Rxxx candidates: the active semantic region is in "
                                    f"Camera {camera}."
                                ),
                                "samples": [],
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
        return manifest


def refresh_local_geometry_artifacts(
    *,
    perception_dir: Path,
    artifact_dir: Path,
    manifest: Mapping[str, Any],
    install: bool = True,
) -> dict[str, Any]:
    """Rewrite overlay/guide after deterministic capability filtering.

    Green Rxxx markers are candidates Claude may select. Workspace/IK-rejected
    candidates remain visible as red crosses for diagnosis, but they are never
    included in the installed coordinate guide.
    """

    perception = Path(perception_dir).resolve()
    output = Path(artifact_dir).resolve()
    if not output.is_dir():
        raise SemanticPipelineError(f"local geometry output does not exist: {output}")
    camera = _text(manifest.get("camera"), context="local geometry.camera").upper()
    if camera not in {"A", "B"}:
        raise SemanticPipelineError("local geometry camera must be A or B")
    candidates = manifest.get("candidates")
    rejected = manifest.get("capability_rejected_candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise SemanticPipelineError(
            "filtered local geometry needs reachable candidates"
        )
    if not isinstance(rejected, list):
        raise SemanticPipelineError("capability_rejected_candidates must be a list")
    source_image = Path(str(manifest.get("source_image", ""))).resolve()
    if not source_image.is_file():
        raw_images = sorted(perception.glob(f"camera_*_{camera}.png"))
        if len(raw_images) != 1:
            raise SemanticPipelineError(
                f"Camera {camera} filtered local geometry needs exactly one raw image"
            )
        source_image = raw_images[0]
    anchor_pixel = manifest.get("semantic_anchor_pixel_xy")
    if (
        not isinstance(anchor_pixel, list)
        or len(anchor_pixel) != 2
        or any(not isinstance(value, (int, float)) for value in anchor_pixel)
    ):
        raise SemanticPipelineError(
            "local geometry manifest needs semantic anchor pixel"
        )
    anchor_x, anchor_y = (round(float(value)) for value in anchor_pixel)
    image = Image.open(source_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (anchor_x - 10, anchor_y - 10, anchor_x + 10, anchor_y + 10),
        outline=(255, 210, 0),
        width=3,
    )
    draw.text(
        (anchor_x + 12, anchor_y - 10),
        f"{manifest.get('semantic_anchor_id')} {manifest.get('target_part')}",
        fill=(255, 230, 20),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    for item in candidates:
        pixel = item.get("pixel_xy") if isinstance(item, Mapping) else None
        if not isinstance(pixel, list) or len(pixel) != 2:
            raise SemanticPipelineError("reachable local candidate has no pixel_xy")
        x_px, y_px = (round(float(value)) for value in pixel)
        draw.ellipse(
            (x_px - 7, y_px - 7, x_px + 7, y_px + 7),
            fill=(30, 255, 90),
            outline=(0, 0, 0),
            width=2,
        )
        draw.text(
            (x_px + 9, y_px - 8),
            f"{item.get('reference_id')} PASS {item.get('feature')}",
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    for item in rejected:
        pixel = item.get("pixel_xy") if isinstance(item, Mapping) else None
        if not isinstance(pixel, list) or len(pixel) != 2:
            continue
        x_px, y_px = (round(float(value)) for value in pixel)
        draw.line((x_px - 7, y_px - 7, x_px + 7, y_px + 7), fill=(255, 60, 60), width=3)
        draw.line((x_px - 7, y_px + 7, x_px + 7, y_px - 7), fill=(255, 60, 60), width=3)
        draw.text(
            (x_px + 9, y_px - 8),
            f"{item.get('reference_id')} REJECTED {item.get('rejection_reason', 'capability')}",
            fill=(255, 90, 90),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    overlay_name = f"camera_{camera}_local_grasp_candidates.png"
    image.save(output / overlay_name)
    guide = {
        "camera_label": camera,
        "coordinate_frame": "robot_base_mm",
        "measurement_kind": "semantic_region_local_geometry_grasp_candidate",
        "semantic_anchor_id": manifest.get("semantic_anchor_id"),
        "target_part": manifest.get("target_part"),
        "overlay_image": overlay_name,
        "reference_semantics": (
            "Only green Rxxx entries passed deterministic workspace/controller-IK "
            "capability gates and may be selected. Red-cross entries are diagnostic "
            "rejections and are not coordinate-guide samples. Rxxx entries are local "
            "geometry candidates, not Molmo anchors."
        ),
        "capability_rejected_count": len(rejected),
        "samples": candidates,
    }
    guide_name = f"camera_{camera}_local_geometry_coordinate_guide.json"
    (output / guide_name).write_text(
        json.dumps(guide, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    updated = {
        **dict(manifest),
        "candidate_count": len(candidates),
        "capability_rejected_count": len(rejected),
        "overlay": str(output / overlay_name),
        "coordinate_guide": str(output / guide_name),
    }
    (output / "local_geometry_candidates.json").write_text(
        json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if install:
        for label in ("A", "B"):
            canonical = perception / f"camera_{label}_coordinate_guide.json"
            backup = perception / f"camera_{label}_uniform_coordinate_guide.json"
            if canonical.is_file() and not backup.exists():
                shutil.copy2(canonical, backup)
            if label == camera:
                shutil.copy2(output / guide_name, canonical)
                shutil.copy2(output / overlay_name, perception / overlay_name)
            elif canonical.is_file():
                canonical.write_text(
                    json.dumps(
                        {
                            "camera_label": label,
                            "coordinate_frame": "robot_base_mm",
                            "measurement_kind": "semantic_region_local_geometry_grasp_candidate",
                            "semantic_anchor_id": manifest.get("semantic_anchor_id"),
                            "reference_semantics": (
                                "No Rxxx candidates: the active semantic region is in "
                                f"Camera {camera}."
                            ),
                            "samples": [],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
    return updated


@dataclass(frozen=True)
class SemanticStageEvaluation:
    status: str
    confidence: float
    evidence: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class SemanticEvaluation:
    hypothesis: str
    semantic_target: SemanticStageEvaluation
    grasp_acquisition: SemanticStageEvaluation
    structure_engagement: SemanticStageEvaluation
    opening_relevance: SemanticStageEvaluation
    transport: SemanticStageEvaluation
    laydown: SemanticStageEvaluation
    task_progress: dict[str, Any]
    earliest_failure_stage: str
    next_experiment: dict[str, Any]

    @property
    def stop(self) -> bool:
        return not self.next_experiment["change"]

    @property
    def reason(self) -> str:
        return str(self.next_experiment["reason"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "semantic_target": {
                **self.semantic_target.as_dict(),
                "hypothesis": self.hypothesis,
            },
            "grasp_acquisition": self.grasp_acquisition.as_dict(),
            "structure_engagement": self.structure_engagement.as_dict(),
            "opening_relevance": self.opening_relevance.as_dict(),
            "transport": self.transport.as_dict(),
            "laydown": self.laydown.as_dict(),
            "task_progress": self.task_progress,
            "earliest_failure_stage": self.earliest_failure_stage,
            "next_experiment": self.next_experiment,
        }


def _validate_stage(
    name: str, value: Any
) -> tuple[SemanticStageEvaluation, str | None]:
    expected = frozenset({"status", "confidence", "evidence"})
    if name == "semantic_target":
        expected = expected | {"hypothesis"}
    stage = _strict_fields(value, expected, context=f"semantic evaluation.{name}")
    status = _text(stage["status"], context=f"semantic evaluation.{name}.status")
    if status not in SEMANTIC_STAGE_STATUSES[name]:
        raise SemanticPipelineError(
            f"semantic evaluation.{name}.status must be one of "
            f"{sorted(SEMANTIC_STAGE_STATUSES[name])}"
        )
    evidence = _string_list(
        stage["evidence"], context=f"semantic evaluation.{name}.evidence"
    )
    hypothesis = None
    if name == "semantic_target":
        hypothesis = _text(
            stage["hypothesis"],
            context="semantic evaluation.semantic_target.hypothesis",
        )
    return (
        SemanticStageEvaluation(
            status=status,
            confidence=_confidence(
                stage["confidence"],
                context=f"semantic evaluation.{name}.confidence",
            ),
            evidence=evidence,
        ),
        hypothesis,
    )


def _expected_earliest_failure(stages: Mapping[str, SemanticStageEvaluation]) -> str:
    explicit_failures = (
        ("SEMANTIC_TARGET", stages["semantic_target"].status == "CONTRADICTED"),
        ("ACQUISITION", stages["grasp_acquisition"].status == "FAILURE"),
        (
            "STRUCTURE_ENGAGEMENT",
            stages["structure_engagement"].status == "FAILURE",
        ),
        (
            "OPENING_RELEVANCE",
            stages["opening_relevance"].status == "CONTRADICTED",
        ),
        (
            "TRANSPORT",
            stages["transport"].status in {"BAD_DIRECTION", "INSUFFICIENT", "OVERPULL"},
        ),
        ("LAYDOWN", stages["laydown"].status == "FAILURE"),
    )
    for name, failed in explicit_failures:
        if failed:
            return name
    fully_supported = (
        stages["semantic_target"].status == "SUPPORTED"
        and stages["grasp_acquisition"].status == "SUCCESS"
        and stages["structure_engagement"].status == "SUCCESS"
        and stages["opening_relevance"].status == "SUPPORTED"
        and stages["transport"].status == "GOOD"
        and stages["laydown"].status == "SUCCESS"
    )
    return "NONE" if fully_supported else "UNKNOWN"


def validate_semantic_evaluation_payload(payload: Any) -> SemanticEvaluation:
    fields = frozenset(
        {
            *SEMANTIC_STAGE_STATUSES,
            "task_progress",
            "earliest_failure_stage",
            "next_experiment",
        }
    )
    value = _strict_fields(payload, fields, context="semantic evaluation")
    stages: dict[str, SemanticStageEvaluation] = {}
    hypothesis = None
    for name in SEMANTIC_STAGE_STATUSES:
        stages[name], stage_hypothesis = _validate_stage(name, value[name])
        hypothesis = stage_hypothesis or hypothesis
    progress = _strict_fields(
        value["task_progress"],
        frozenset({"status", "confidence", "metrics"}),
        context="semantic evaluation.task_progress",
    )
    progress_status = _text(
        progress["status"], context="semantic evaluation.task_progress.status"
    )
    if progress_status not in TASK_PROGRESS_STATUSES:
        raise SemanticPipelineError(
            f"task progress status must be one of {sorted(TASK_PROGRESS_STATUSES)}"
        )
    metrics = _strict_fields(
        progress["metrics"],
        frozenset(
            {
                "visible_area_delta",
                "overlap_delta",
                "relief_delta",
                "boundary_change",
            }
        ),
        context="semantic evaluation.task_progress.metrics",
    )
    normalized_metrics: dict[str, Any] = {}
    for name in ("visible_area_delta", "overlap_delta", "relief_delta"):
        raw = metrics[name]
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            normalized_metrics[name] = float(raw)
        elif isinstance(raw, str) and raw.upper() in METRIC_DELTAS:
            normalized_metrics[name] = raw.upper()
        else:
            raise SemanticPipelineError(
                f"semantic evaluation metric {name} must be numeric or one of "
                f"{sorted(METRIC_DELTAS)}"
            )
    normalized_metrics["boundary_change"] = _text(
        metrics["boundary_change"],
        context="semantic evaluation.task_progress.metrics.boundary_change",
    )
    earliest = _text(
        value["earliest_failure_stage"],
        context="semantic evaluation.earliest_failure_stage",
    )
    if earliest not in SEMANTIC_FAILURE_STAGES:
        raise SemanticPipelineError(
            f"earliest_failure_stage must be one of {sorted(SEMANTIC_FAILURE_STAGES)}"
        )
    expected_earliest = _expected_earliest_failure(stages)
    if earliest != expected_earliest:
        raise SemanticPipelineError(
            "earliest_failure_stage is inconsistent with stage statuses: "
            f"expected {expected_earliest}, got {earliest}"
        )
    next_value = _strict_fields(
        value["next_experiment"],
        frozenset({"keep", "change", "reason"}),
        context="semantic evaluation.next_experiment",
    )
    keep = _string_list(
        next_value["keep"],
        context="semantic evaluation.next_experiment.keep",
        allow_empty=True,
    )
    change = _string_list(
        next_value["change"],
        context="semantic evaluation.next_experiment.change",
        allow_empty=True,
    )
    if set(keep).intersection(change):
        raise SemanticPipelineError("semantic evaluation keep/change must not overlap")
    return SemanticEvaluation(
        hypothesis=hypothesis or "unknown",
        semantic_target=stages["semantic_target"],
        grasp_acquisition=stages["grasp_acquisition"],
        structure_engagement=stages["structure_engagement"],
        opening_relevance=stages["opening_relevance"],
        transport=stages["transport"],
        laydown=stages["laydown"],
        task_progress={
            "status": progress_status,
            "confidence": _confidence(
                progress["confidence"],
                context="semantic evaluation.task_progress.confidence",
            ),
            "metrics": normalized_metrics,
        },
        earliest_failure_stage=earliest,
        next_experiment={
            "keep": list(keep),
            "change": list(change),
            "reason": _text(
                next_value["reason"],
                context="semantic evaluation.next_experiment.reason",
            ),
        },
    )


SEMANTIC_EVALUATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {},
    "required": [
        *SEMANTIC_STAGE_STATUSES,
        "task_progress",
        "earliest_failure_stage",
        "next_experiment",
    ],
}
for _name, _statuses in SEMANTIC_STAGE_STATUSES.items():
    _properties: dict[str, Any] = {
        "status": {"type": "string", "enum": sorted(_statuses)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
    }
    if _name == "semantic_target":
        _properties["hypothesis"] = {"type": "string", "minLength": 1}
    SEMANTIC_EVALUATION_JSON_SCHEMA["properties"][_name] = {
        "type": "object",
        "additionalProperties": False,
        "properties": _properties,
        "required": list(_properties),
    }
SEMANTIC_EVALUATION_JSON_SCHEMA["properties"].update(
    {
        "task_progress": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": sorted(TASK_PROGRESS_STATUSES)},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "metrics": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        name: {
                            "oneOf": [
                                {"type": "number"},
                                {"type": "string", "enum": sorted(METRIC_DELTAS)},
                            ]
                        }
                        for name in (
                            "visible_area_delta",
                            "overlap_delta",
                            "relief_delta",
                        )
                    }
                    | {"boundary_change": {"type": "string", "minLength": 1}},
                    "required": [
                        "visible_area_delta",
                        "overlap_delta",
                        "relief_delta",
                        "boundary_change",
                    ],
                },
            },
            "required": ["status", "confidence", "metrics"],
        },
        "earliest_failure_stage": {
            "type": "string",
            "enum": sorted(SEMANTIC_FAILURE_STAGES),
        },
        "next_experiment": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "keep": {
                    "type": "array",
                    "maxItems": 12,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                "change": {
                    "type": "array",
                    "maxItems": 12,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["keep", "change", "reason"],
        },
    }
)


def validate_action_scope(
    actions: Sequence[Mapping[str, Any]],
    *,
    candidate: Mapping[str, Any],
    scope: ActionScope,
) -> dict[str, float]:
    close_index = next(
        (
            index
            for index, action in enumerate(actions)
            if action.get("name") == "close_gripper"
        ),
        None,
    )
    if close_index is None:
        raise SemanticPipelineError("action proposal has no close_gripper")
    close_indices = [
        index
        for index, action in enumerate(actions)
        if action.get("name") == "close_gripper"
    ]
    if len(close_indices) != 1:
        raise SemanticPipelineError(
            "action proposal must contain exactly one close_gripper cycle"
        )
    post_close_open_indices = [
        index
        for index, action in enumerate(
            actions[close_index + 1 :], start=close_index + 1
        )
        if action.get("name") == "open_gripper"
    ]
    if len(post_close_open_indices) != 1:
        raise SemanticPipelineError(
            "action proposal must contain exactly one release after close_gripper"
        )
    grasp_move = next(
        (
            action
            for action in reversed(actions[:close_index])
            if action.get("name") == "move"
        ),
        None,
    )
    if grasp_move is None:
        raise SemanticPipelineError(
            "action proposal has no grasp move before close_gripper"
        )
    args = grasp_move.get("args")
    if not isinstance(args, Mapping):
        raise SemanticPipelineError("grasp move args must be an object")
    expected = candidate.get("base_xyz_mm")
    if not isinstance(expected, list) or len(expected) != 3:
        raise SemanticPipelineError("local candidate has no base_xyz_mm")
    grasp_xy_error = float(
        np.linalg.norm(
            np.asarray([float(args["x"]), float(args["y"])])
            - np.asarray(expected[:2], dtype=np.float64)
        )
    )
    if grasp_xy_error > 2.0:
        raise SemanticPipelineError(
            f"action grasp does not use selected local candidate; XY error={grasp_xy_error:.2f} mm"
        )
    grasp_xyz = np.asarray(
        [float(args["x"]), float(args["y"]), float(args["z"])], dtype=np.float64
    )
    post_grasp_moves: list[np.ndarray] = []
    release_index: int | None = None
    for action_index, action in enumerate(
        actions[close_index + 1 :], start=close_index + 1
    ):
        if action.get("name") == "open_gripper":
            release_index = action_index
            break
        if action.get("name") != "move":
            continue
        move_args = action.get("args")
        if isinstance(move_args, Mapping):
            post_grasp_moves.append(
                np.asarray(
                    [
                        float(move_args["x"]),
                        float(move_args["y"]),
                        float(move_args["z"]),
                    ],
                    dtype=np.float64,
                )
            )
    if not post_grasp_moves:
        raise SemanticPipelineError("action proposal has no post-grasp target state")
    if len(post_grasp_moves) > scope.max_post_grasp_moves:
        raise SemanticPipelineError(
            f"action exceeds {scope.name} post-grasp move budget: "
            f"{len(post_grasp_moves)} > {scope.max_post_grasp_moves}"
        )
    if release_index is None:
        raise SemanticPipelineError(
            "action proposal has no release after close_gripper"
        )
    if any(
        action.get("name") == "close_gripper" for action in actions[release_index + 1 :]
    ):
        raise SemanticPipelineError("action proposal cannot start a second grasp cycle")
    max_lateral = max(
        float(np.linalg.norm(point[:2] - grasp_xyz[:2])) for point in post_grasp_moves
    )
    max_lift = max(
        0.0, max(float(point[2] - grasp_xyz[2]) for point in post_grasp_moves)
    )
    if max_lateral > scope.max_lateral_mm + 1e-6:
        raise SemanticPipelineError(
            f"action exceeds {scope.name} lateral authority: "
            f"{max_lateral:.1f} > {scope.max_lateral_mm:.1f} mm"
        )
    if max_lift > scope.max_lift_mm + 1e-6:
        raise SemanticPipelineError(
            f"action exceeds {scope.name} lift authority: "
            f"{max_lift:.1f} > {scope.max_lift_mm:.1f} mm"
        )
    return {
        "grasp_xy_error_mm": grasp_xy_error,
        "max_lateral_mm": max_lateral,
        "max_lift_mm": max_lift,
    }


def build_structured_experience(
    *,
    iteration: int,
    semantic_state: Mapping[str, Any],
    strategy: SemanticStrategy,
    candidate: Mapping[str, Any],
    action_scope: ActionScope,
    evaluation: SemanticEvaluation,
) -> dict[str, Any]:
    direction = strategy.desired_change
    transport_result = evaluation.transport.status
    return {
        "schema_version": 1,
        "created_at": _now(),
        "iteration": iteration,
        "hypothesis_key": strategy.hypothesis_key,
        "semantic_state": {
            "target_part": strategy.target_part,
            "hypothesis": strategy.hypothesis_state,
            "desired_change": strategy.desired_change,
            "relations": semantic_state.get("relations", {}).get(
                strategy.anchor_id, {}
            ),
        },
        "chosen_structure": {
            "geometry_type": candidate.get("feature"),
            "free_boundary": candidate.get("free_boundary"),
            "height_step_mm": candidate.get("height_step_mm"),
            "relief_above_local_median_mm": candidate.get(
                "relief_above_local_median_mm"
            ),
        },
        "action_scope": action_scope.as_dict(),
        "acquisition": evaluation.grasp_acquisition.as_dict(),
        "structure_engagement": evaluation.structure_engagement.as_dict(),
        "opening_relevance": evaluation.opening_relevance.as_dict(),
        "transport": {
            "direction": direction,
            "result": transport_result,
            "evaluation": evaluation.transport.as_dict(),
        },
        "evaluation": evaluation.as_dict(),
        "conclusion": (
            f"For hypothesis {strategy.hypothesis_state} on {strategy.target_part}, "
            f"a {candidate.get('feature')} produced acquisition="
            f"{evaluation.grasp_acquisition.status}, structure_engagement="
            f"{evaluation.structure_engagement.status}, opening_relevance="
            f"{evaluation.opening_relevance.status}, transport={transport_result}."
        ),
    }


def append_structured_experience(path: Path, experience: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(experience), ensure_ascii=False) + "\n")


def load_structured_experiences(path: Path, *, limit: int = 32) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return []
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        target.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SemanticPipelineError(
                f"structured experience line {line_number} must be an object"
            )
        items.append(value)
    return items[-max(1, int(limit)) :]
