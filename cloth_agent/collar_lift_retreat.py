"""Standalone Claude-selected collar high-lift and retreat experiment.

Claude first selects a visible, graspable Camera A pixel on collar/neckline
fabric. Runtime grounds that pixel, then Claude chooses the numeric high-lift,
far +X transfer, gradual descending retreat, release, and retract waypoints.
Runtime owns validation and execution, including grounded-grasp consistency,
table/workspace checks, static preflight, and full controller IK.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .config import SafetyError
from .free_exploration import (
    ExplorationProposal,
    PIXEL_GROUNDING_MCP_TOOLS,
    _json_from_claude_text,
    grounding_mcp_config,
    _load_or_create_session,
    exploration_source,
    global_perception_image_paths,
    validate_global_exploration_payload,
)
from .garment_grounding_mcp import GarmentGrounding, GroundingToolError
from .grasp_height import GraspHeightError, resolve_grasp_height
from .molmo_keypoint_pipeline import (
    KeypointSpec,
    MolmoKeypointPipelineError,
    run_molmo_semantic_anchor_pipeline,
)
from .perception import PerceptionConfig, capture_two_view_rgbd
from .robot_api import _controller_trajectory_with_arm, move_robot_to_perception_position
from .rollout_recorder import DualRealSenseRolloutRecorder
from .viewer import _load_latest_perception


class CollarLiftRetreatError(RuntimeError):
    """Raised when the standalone experiment cannot be validated safely."""


COLLAR_MOTION_MAX_ACTIONS = 14
COLLAR_EXECUTION_MAX_ACTIONS = 15
COLLAR_GRASP_YAW_DEG = 90.0
AUTO_IK_FAR_X_SEARCH_STEPS = 7
AUTO_IK_MIN_TRANSPORT_MM = 20.0
AUTO_IK_BACKOFF_MM = 3.0
MIN_HIGH_LIFT_FRACTION_OF_AVAILABLE_Z = 0.5
# Keep the hanging garment at the high pose until it reaches the far +X point.
# The legacy ``pretransport_lower_xyz_mm`` waypoint remains in the contract for
# compatibility, but it is now a same-height transition point: all meaningful
# lowering starts only after the far transport.
PRETRANSPORT_MIN_Z_DROP_MM = 0.0
PRETRANSPORT_MAX_Z_DROP_MM = 0.0
PRETRANSPORT_XY_TOLERANCE_MM = 5.0
TRANSPORT_Z_TOLERANCE_MM = 5.0
MIN_DESCENT_RETREAT_MM = 10.0
MIN_DESCENT_Z_PER_X_RATIO = 0.5
MAX_DESCENT_Z_PER_X_RATIO = 2.0
MAX_RELEASE_HEIGHT_ABOVE_TABLE_MM = 80.0
NECK_SIDE_MIN_LABEL_DISTANCE_PX = 12.0
NECK_SIDE_MAX_LABEL_DISTANCE_PX = 120.0
NECK_SIDE_MIN_TORSO_DISTANCE_PX = 30.0
NECK_SIDE_MIN_OPPOSITION_COSINE = 0.35
COLLAR_MOLMO_SPECS: tuple[KeypointSpec, ...] = (
    KeypointSpec(
        "neck_label",
        (
            "the small sewn rectangular neck or size label attached inside the shirt collar; "
            "this is a topology guide, not a grasp point, and may be pointed to directly"
        ),
        (40, 220, 255),
    ),
    KeypointSpec(
        "collar",
        (
            "graspable collar-band fabric immediately adjacent to the shirt's neck opening and "
            "sewn neck label; use the label only as a topology clue, and do not substitute a "
            "sleeve, shoulder seam, chest fold, broad interior panel, printed graphic, or label"
        ),
        (255, 170, 0),
    ),
    KeypointSpec(
        "left_shoulder",
        (
            "the garment's left shoulder as worn, where the neck/torso transitions toward the "
            "left sleeve; this is a negative landmark that must not be used as the collar"
        ),
        (50, 180, 255),
    ),
    KeypointSpec(
        "right_shoulder",
        (
            "the garment's right shoulder as worn, where the neck/torso transitions toward the "
            "right sleeve; this is a negative landmark that must not be used as the collar"
        ),
        (80, 220, 80),
    ),
)


@dataclass(frozen=True)
class CollarSelection:
    status: str
    camera: str | None
    pixel_xy: tuple[int, int] | None
    neck_label_pixel_xy: tuple[int, int] | None
    torso_landmark_pixel_xy: tuple[int, int] | None
    neck_side_opposition_cosine: float | None
    confidence: float
    evidence: tuple[str, ...]
    reference_correspondence: tuple[str, ...]
    molmo_guided_region_evidence: tuple[str, ...]
    grasp_point_evidence: tuple[str, ...]
    molmo_relation: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "camera": self.camera,
            "pixel_xy": list(self.pixel_xy) if self.pixel_xy is not None else None,
            "neck_label_pixel_xy": (
                list(self.neck_label_pixel_xy)
                if self.neck_label_pixel_xy is not None
                else None
            ),
            "torso_landmark_pixel_xy": (
                list(self.torso_landmark_pixel_xy)
                if self.torso_landmark_pixel_xy is not None
                else None
            ),
            "neck_side_opposition_cosine": self.neck_side_opposition_cosine,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "reference_correspondence": list(self.reference_correspondence),
            "molmo_guided_region_evidence": list(self.molmo_guided_region_evidence),
            "grasp_point_evidence": list(self.grasp_point_evidence),
            "molmo_relation": self.molmo_relation,
            "reason": self.reason,
        }


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CollarLiftRetreatError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CollarLiftRetreatError(f"{name} must be finite")
    return result


def validate_collar_selection_payload(
    payload: Any,
    *,
    image_width: int,
    image_height: int,
    min_confidence: float = 0.0,
) -> CollarSelection:
    """Validate Claude's collar-only selection contract."""

    if not isinstance(payload, Mapping):
        raise CollarLiftRetreatError("Claude collar selection must be a JSON object")
    required = {
        "status",
        "camera",
        "pixel_xy",
        "neck_label_pixel_xy",
        "torso_landmark_pixel_xy",
        "neck_side_opposition_cosine",
        "confidence",
        "evidence",
        "reference_correspondence",
        "molmo_guided_region_evidence",
        "grasp_point_evidence",
        "molmo_relation",
        "reason",
    }
    if set(payload) != required:
        raise CollarLiftRetreatError(
            "Claude collar selection must contain exactly " + ", ".join(sorted(required))
        )
    status = str(payload["status"]).strip().upper()
    if status not in {"SELECTED", "NOT_FOUND"}:
        raise CollarLiftRetreatError("collar selection status must be SELECTED or NOT_FOUND")
    confidence = _finite_number(payload["confidence"], "collar confidence")
    if not 0.0 <= confidence <= 1.0:
        raise CollarLiftRetreatError("collar confidence must be between 0 and 1")
    def string_evidence(name: str) -> tuple[str, ...]:
        raw = payload[name]
        if (
            not isinstance(raw, list)
            or not raw
            or any(not isinstance(item, str) or not item.strip() for item in raw)
        ):
            raise CollarLiftRetreatError(f"collar {name} must be a non-empty string list")
        return tuple(item.strip() for item in raw)

    evidence = string_evidence("evidence")
    reference_correspondence = string_evidence("reference_correspondence")
    molmo_guided_region_evidence = string_evidence("molmo_guided_region_evidence")
    grasp_point_evidence = string_evidence("grasp_point_evidence")
    molmo_relation = str(payload["molmo_relation"]).strip().upper()
    if molmo_relation not in {
        "USES_MOLMO_COLLAR_REGION",
        "CORRECTS_MOLMO_COLLAR_WITH_NECK_LABEL",
        "MOLMO_COLLAR_UNAVAILABLE",
    }:
        raise CollarLiftRetreatError("invalid collar molmo_relation")
    reason = payload["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise CollarLiftRetreatError("collar reason must be a non-empty string")
    if status == "NOT_FOUND":
        if any(
            payload[name] is not None
            for name in (
                "camera",
                "pixel_xy",
                "neck_label_pixel_xy",
                "torso_landmark_pixel_xy",
                "neck_side_opposition_cosine",
            )
        ):
            raise CollarLiftRetreatError(
                "NOT_FOUND must use null camera, pixels, and neck-side score"
            )
        return CollarSelection(
            status=status,
            camera=None,
            pixel_xy=None,
            neck_label_pixel_xy=None,
            torso_landmark_pixel_xy=None,
            neck_side_opposition_cosine=None,
            confidence=confidence,
            evidence=evidence,
            reference_correspondence=reference_correspondence,
            molmo_guided_region_evidence=molmo_guided_region_evidence,
            grasp_point_evidence=grasp_point_evidence,
            molmo_relation=molmo_relation,
            reason=reason.strip(),
        )
    if payload["camera"] != "A":
        raise CollarLiftRetreatError("the actionable collar point must come from Camera A")
    pixel = payload["pixel_xy"]
    if (
        not isinstance(pixel, list)
        or len(pixel) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in pixel)
    ):
        raise CollarLiftRetreatError("collar pixel_xy must contain two integers")
    x_px, y_px = int(pixel[0]), int(pixel[1])
    if not 0 <= x_px < int(image_width) or not 0 <= y_px < int(image_height):
        raise CollarLiftRetreatError(
            f"collar pixel ({x_px}, {y_px}) is outside Camera A {image_width}x{image_height}"
        )
    if confidence < float(min_confidence):
        raise CollarLiftRetreatError(
            f"collar confidence {confidence:.3f} is below required {float(min_confidence):.3f}"
        )

    def required_pixel(name: str) -> tuple[int, int]:
        raw = payload[name]
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw
            )
        ):
            raise CollarLiftRetreatError(f"collar {name} must contain two integers")
        point = (int(raw[0]), int(raw[1]))
        if not 0 <= point[0] < int(image_width) or not 0 <= point[1] < int(image_height):
            raise CollarLiftRetreatError(
                f"collar {name} {point} is outside Camera A {image_width}x{image_height}"
            )
        return point

    neck_label_pixel = required_pixel("neck_label_pixel_xy")
    torso_landmark_pixel = required_pixel("torso_landmark_pixel_xy")
    candidate_vector = np.asarray(
        [x_px - neck_label_pixel[0], y_px - neck_label_pixel[1]],
        dtype=np.float64,
    )
    torso_vector = np.asarray(
        [
            torso_landmark_pixel[0] - neck_label_pixel[0],
            torso_landmark_pixel[1] - neck_label_pixel[1],
        ],
        dtype=np.float64,
    )
    candidate_distance = float(np.linalg.norm(candidate_vector))
    torso_distance = float(np.linalg.norm(torso_vector))
    if not (
        NECK_SIDE_MIN_LABEL_DISTANCE_PX
        <= candidate_distance
        <= NECK_SIDE_MAX_LABEL_DISTANCE_PX
    ):
        raise CollarLiftRetreatError(
            "collar grasp must stay within the validated neck-label neighborhood: "
            f"distance={candidate_distance:.2f}px, required="
            f"[{NECK_SIDE_MIN_LABEL_DISTANCE_PX:.0f}, "
            f"{NECK_SIDE_MAX_LABEL_DISTANCE_PX:.0f}]px"
        )
    if torso_distance < NECK_SIDE_MIN_TORSO_DISTANCE_PX:
        raise CollarLiftRetreatError(
            "torso landmark is too close to the neck label to establish neck-side direction"
        )
    opposition_cosine = float(
        -np.dot(candidate_vector, torso_vector)
        / max(candidate_distance * torso_distance, 1e-9)
    )
    reported_opposition = _finite_number(
        payload["neck_side_opposition_cosine"],
        "neck_side_opposition_cosine",
    )
    if not math.isclose(reported_opposition, opposition_cosine, abs_tol=0.02):
        raise CollarLiftRetreatError(
            "reported neck-side opposition does not match the submitted pixels: "
            f"reported={reported_opposition:.3f}, computed={opposition_cosine:.3f}"
        )
    if opposition_cosine < NECK_SIDE_MIN_OPPOSITION_COSINE:
        raise CollarLiftRetreatError(
            "collar grasp is not on the neck side of the label; it points toward/sideways "
            "to the torso and may select an overlying sleeve: "
            f"opposition_cosine={opposition_cosine:.3f}, required>="
            f"{NECK_SIDE_MIN_OPPOSITION_COSINE:.2f}"
        )
    return CollarSelection(
        status=status,
        camera="A",
        pixel_xy=(x_px, y_px),
        neck_label_pixel_xy=neck_label_pixel,
        torso_landmark_pixel_xy=torso_landmark_pixel,
        neck_side_opposition_cosine=opposition_cosine,
        confidence=confidence,
        evidence=evidence,
        reference_correspondence=reference_correspondence,
        molmo_guided_region_evidence=molmo_guided_region_evidence,
        grasp_point_evidence=grasp_point_evidence,
        molmo_relation=molmo_relation,
        reason=reason.strip(),
    )


def _selector_images(paths: Sequence[Path]) -> list[Path]:
    """Return a small starting bundle; supporting geometry remains on-demand."""

    suffixes = (
        "camera_0_a.png",
        "camera_a_flat_reference.png",
        "camera_a_flat_reference_anchors.png",
        "camera_a_semantic_anchors.png",
        "camera_a_semantic_anchor_diagnostics.png",
    )
    selected = [path for path in paths if path.name.lower() in suffixes]
    return selected or list(paths)


def invoke_claude_collar_selector(
    image_paths: Sequence[Path],
    *,
    run_dir: Path,
    image_width: int,
    image_height: int,
    molmo_manifest: Mapping[str, Any] | None = None,
    reference_anchors: Mapping[str, Any] | None = None,
    binary: str = "claude",
    timeout_s: int = 900,
    min_confidence: float = 0.0,
) -> tuple[CollarSelection, dict[str, Any]]:
    """Ask Claude only for the collar fabric pixel; it cannot plan robot motion."""

    root = Path(run_dir).resolve()
    all_images: list[Path] = []
    for raw in [Path(path).resolve() for path in image_paths]:
        if root not in raw.parents or not raw.is_file():
            raise CollarLiftRetreatError(f"collar image is outside the run or missing: {raw}")
        all_images.append(raw)
    safe_images = _selector_images(all_images)
    optional_images = [path for path in all_images if path not in safe_images]
    if not safe_images:
        raise CollarLiftRetreatError("no saved collar-selection images are available")
    executable = shutil.which(binary) if Path(binary).name == binary else binary
    if executable is None:
        raise CollarLiftRetreatError(f"Claude CLI not found: {binary}")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["SELECTED", "NOT_FOUND"]},
            "camera": {"type": ["string", "null"], "enum": ["A", None]},
            "pixel_xy": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    {"type": "null"},
                ]
            },
            "neck_label_pixel_xy": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    {"type": "null"},
                ]
            },
            "torso_landmark_pixel_xy": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    {"type": "null"},
                ]
            },
            "neck_side_opposition_cosine": {
                "anyOf": [
                    {"type": "number", "minimum": -1, "maximum": 1},
                    {"type": "null"},
                ]
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "items": {"type": "string", "minLength": 1},
            },
            "reference_correspondence": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "items": {"type": "string", "minLength": 1},
            },
            "molmo_guided_region_evidence": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "items": {"type": "string", "minLength": 1},
            },
            "grasp_point_evidence": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "items": {"type": "string", "minLength": 1},
            },
            "molmo_relation": {
                "type": "string",
                "enum": [
                    "USES_MOLMO_COLLAR_REGION",
                    "CORRECTS_MOLMO_COLLAR_WITH_NECK_LABEL",
                    "MOLMO_COLLAR_UNAVAILABLE",
                ],
            },
            "reason": {"type": "string", "minLength": 1},
        },
        "required": [
            "status",
            "camera",
            "pixel_xy",
            "neck_label_pixel_xy",
            "torso_landmark_pixel_xy",
            "neck_side_opposition_cosine",
            "confidence",
            "evidence",
            "reference_correspondence",
            "molmo_guided_region_evidence",
            "grasp_point_evidence",
            "molmo_relation",
            "reason",
        ],
    }
    molmo_available = isinstance(molmo_manifest, Mapping)
    molmo_context: dict[str, Any] = {
        "status": None if molmo_available else "UNAVAILABLE_SKIPPED_BY_EXPERIMENT",
        "anchors": [],
        "topology_guides": [],
        "axis_reference": None,
        "camera_a_records": [],
    }
    if isinstance(molmo_manifest, Mapping):
        molmo_context["status"] = molmo_manifest.get("status")
        molmo_context["anchors"] = [
            {
                key: anchor.get(key)
                for key in ("type", "camera", "pixel_xy", "confidence", "reference_comparison")
            }
            for anchor in (molmo_manifest.get("anchors") or [])
            if isinstance(anchor, Mapping) and anchor.get("camera") == "A"
        ]
        molmo_context["topology_guides"] = [
            {
                key: guide.get(key)
                for key in (
                    "type",
                    "camera",
                    "pixel_xy",
                    "confidence",
                    "query_mode",
                    "role",
                )
            }
            for guide in (molmo_manifest.get("topology_guides") or [])
            if isinstance(guide, Mapping) and guide.get("camera") == "A"
        ]
        axes = molmo_manifest.get("axis_references")
        if isinstance(axes, Mapping):
            molmo_context["axis_reference"] = axes.get("A")
        for view in (molmo_manifest.get("views") or []):
            if not isinstance(view, Mapping) or view.get("camera") != "A":
                continue
            molmo_context["camera_a_records"] = [
                {
                    key: record.get(key)
                    for key in (
                        "name",
                        "status",
                        "confidence",
                        "source_pixel_xy",
                        "accepted",
                        "rejection_reason",
                        "query_mode",
                        "topology_guide",
                        "topology_guide_name",
                        "topology_guide_pixel_xy",
                        "reference_comparison",
                    )
                }
                for record in (view.get("records") or [])
                if isinstance(record, Mapping)
            ]
            break
    reference_context: dict[str, Any] = {"axis_reference": None, "anchors": []}
    if isinstance(reference_anchors, Mapping):
        reference_context["axis_reference"] = reference_anchors.get("axis_reference")
        reference_context["anchors"] = [
            {
                key: anchor.get(key)
                for key in ("name", "description", "selected_pixel_xy")
            }
            for anchor in (reference_anchors.get("anchors") or [])
            if isinstance(anchor, Mapping)
            and anchor.get("name") in {"neckline", "left_shoulder", "right_shoulder"}
        ]
    def image_role(path: Path) -> str:
        name = path.name.lower()
        if name == "camera_a_flat_reference.png":
            return "FLAT REFERENCE CAMERA A RAW RGB"
        if name == "camera_a_flat_reference_anchors.png":
            return "FLAT REFERENCE CAMERA A ANNOTATED TOPOLOGY"
        if "semantic_anchor" in name:
            return "CURRENT CAMERA A MOLMO HYPOTHESES"
        if name == "camera_0_a.png":
            return "CURRENT CAMERA A RAW RGB — ONLY SOURCE OF ACTION PIXEL"
        return "CURRENT CAMERA A SUPPORTING GEOMETRY"
    if molmo_available:
        semantic_guidance = (
            "Treat every Molmo point as a fallible hypothesis, not an accepted fact. Molmo queries the "
            "sewn neck_label first as a topology guide and then performs a separate label-guided collar "
            "query. Use your own judgment to decide whether the label is real, whether the collar query "
            "is useful, and whether more evidence is needed. Do not copy either Molmo pixel blindly, but "
            "do not reject a nearby collar point merely because it is near a Molmo point."
        )
        molmo_decision_guidance = (
            "Set molmo_relation to CORRECTS_MOLMO_COLLAR_WITH_NECK_LABEL when the original collar "
            "hypothesis is on the wrong garment part. In molmo_guided_region_evidence, explicitly state "
            "whether each Molmo hypothesis was supported or rejected."
        )
        selector_system_guidance = "Use your own visual judgment; do not blindly trust or reject Molmo."
    else:
        semantic_guidance = (
            "Molmo is intentionally unavailable in this experiment. Do not invent Molmo points or "
            "claim Molmo support. Localize the collar from the current Camera A RGB, optional RGB-D "
            "geometry inspection, and flat-reference topology only."
        )
        molmo_decision_guidance = (
            "Set molmo_relation to MOLMO_COLLAR_UNAVAILABLE and explicitly state in "
            "molmo_guided_region_evidence that no Molmo result was used."
        )
        selector_system_guidance = "Molmo is disabled; use only the supplied RGB-D/reference evidence."
    prompt = (
        "Locate one graspable shirt-collar fabric pixel using Camera A only. The CURRENT Camera "
        f"A raw action image is {image_width}x{image_height}; pixel_xy must refer to that image. "
        "You are allowed to make your own visual investigation: start with the small core image "
        "bundle below, and use Read on the optional supporting RGB/height/gradient files only when "
        "they would resolve a real ambiguity. You may also call the supplied read-only local-surface "
        "tool to inspect candidate pixels or neighborhoods; it provides geometry, not semantic labels. "
        "Do not assume that all supplied images must be inspected, and do not let a large batch of "
        "heatmaps replace your own garment reasoning.\n\n"
        + semantic_guidance + " Use the FLAT "
        "REFERENCE raw and annotated images for topology and appearance correspondence, but never "
        "transfer a reference pixel into the current image. Folding, rotation, and partial occlusion "
        "may change the current shoulder layout and print orientation. Shoulder anchors and axis_top "
        "are weak layout hypotheses only. Do not use Camera B or infer a Camera A coordinate by "
        "rotating/transferring another camera.\n\n"
        "A clearly visible neck-opening contour is NOT required. Decide freely whether the collar can "
        "be localized from the neck label, current RGB, optional geometry inspection, and reference "
        "topology. In the reference, the sewn neck label sits immediately inside/below the neckline; "
        "use that as evidence, not as a hard-coded pixel offset. Avoid the label itself, the neck-hole "
        "center, shoulder/armhole seams, sleeves, unrelated folds, printed graphics, table, and depth "
        "spikes. You may search outside the immediate label neighborhood only when your own evidence "
        "shows that folding or occlusion makes the local collar region invalid. "
        + molmo_decision_guidance + " Return NOT_FOUND only when the collar cannot be localized even "
        "after using the neck-label/reference topology. Do not plan robot motion. "
        "For every SELECTED result, also return neck_label_pixel_xy and one "
        "torso_landmark_pixel_xy on the visible chest/print/body side. The grasp point must be "
        "on the opposite neck side of the label, not merely beside it: let g=grasp-label and "
        "t=torso-label, and report neck_side_opposition_cosine=(-g dot t)/(|g||t|). The runtime "
        f"requires this value to be at least {NECK_SIDE_MIN_OPPOSITION_COSINE:.2f}, recomputes it "
        "from the submitted pixels, and rejects sideways/toward-torso points before robot motion. "
        "For NOT_FOUND, return null for both landmark pixels and the opposition score. "
        "Return only the requested JSON.\n\n"
        f"Molmo current Camera A context:\n{json.dumps(molmo_context, ensure_ascii=False)}\n\n"
        f"Flat reference anchor context:\n{json.dumps(reference_context, ensure_ascii=False)}\n\n"
        "Core images to start with:\n"
        + "\n".join(f"- {image_role(path)}: {path}" for path in safe_images)
        + "\n\nOptional supporting files (read only if useful):\n"
        + ("\n".join(f"- {path}" for path in optional_images) or "- none")
    )
    mcp_config: dict[str, Any] | None = None
    enabled_tools = ("Read",)
    if (root / "workspace" / "perception_views").is_dir():
        mcp_config = grounding_mcp_config(root, mode="pixel")
        enabled_tools = ("Read", *PIXEL_GROUNDING_MCP_TOOLS)
    command = [
        str(executable),
        "--print",
        prompt,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema, separators=(",", ":")),
        "--permission-mode",
        "dontAsk",
        "--allowedTools",
        ",".join(enabled_tools),
        "--tools",
        "Read",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--add-dir",
        str(root),
        "--system-prompt",
        (
            "You are a read-only Camera-A collar-fabric investigator. You may inspect optional files "
            "and call the strictly scoped local-surface MCP tool when that helps resolve uncertainty. "
            f"{selector_system_guidance} Do not follow a fixed pixel offset. The flat reference supplies topology only. Never grasp the label, never use Camera "
            "B, never edit files, and never control the robot."
        ),
    ]
    if mcp_config is not None:
        command[command.index("--disable-slash-commands"):command.index("--disable-slash-commands")] = [
            "--mcp-config",
            json.dumps(mcp_config, ensure_ascii=False, separators=(",", ":")),
            "--strict-mcp-config",
        ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=int(timeout_s),
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CollarLiftRetreatError(
            f"Claude collar selector timed out after {int(timeout_s)} seconds; "
            "the selector may have been inspecting optional evidence or using the "
            "read-only geometry tool"
        ) from exc
    duration_s = time.monotonic() - started
    if completed.returncode != 0:
        raise CollarLiftRetreatError(
            f"Claude collar selector exited with {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    payload = _json_from_claude_text(completed.stdout)
    selection = validate_collar_selection_payload(
        payload,
        image_width=image_width,
        image_height=image_height,
        min_confidence=min_confidence,
    )
    return selection, {
        "prompt": prompt,
        "command": command,
        "duration_s": duration_s,
        "returncode": completed.returncode,
        "raw_stdout": completed.stdout,
        "raw_stderr": completed.stderr,
        "selection": selection.as_dict(),
    }


def invoke_claude_collar_motion_planner(
    *,
    run_dir: Path,
    selection: CollarSelection,
    grasp_height_plan: Mapping[str, Any],
    table_plane_abc: Sequence[float],
    robot_config: Any,
    binary: str = "claude",
    timeout_s: int = 900,
    previous_proposal: Mapping[str, Any] | None = None,
    validation_error: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ask Claude to choose the complete numeric high-lift/laydown trajectory."""

    if selection.status != "SELECTED" or selection.camera != "A" or selection.pixel_xy is None:
        raise CollarLiftRetreatError("motion planning requires a selected Camera A collar point")
    surface = np.asarray(grasp_height_plan.get("surface_xyz_mm"), dtype=np.float64)
    grasp_target = np.asarray(grasp_height_plan.get("target_xyz_mm"), dtype=np.float64)
    plane = np.asarray(table_plane_abc, dtype=np.float64)
    if surface.shape != (3,) or not np.all(np.isfinite(surface)):
        raise CollarLiftRetreatError("motion planning requires a finite collar surface XYZ")
    if grasp_target.shape != (3,) or not np.all(np.isfinite(grasp_target)):
        raise CollarLiftRetreatError("motion planning requires a finite authoritative grasp XYZ")
    if plane.shape != (3,) or not np.all(np.isfinite(plane)):
        raise CollarLiftRetreatError("motion planning requires a finite fitted table plane")
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise CollarLiftRetreatError(f"motion-planning run directory does not exist: {root}")
    executable = shutil.which(binary) if Path(binary).name == binary else binary
    if executable is None:
        raise CollarLiftRetreatError(f"Claude CLI not found: {binary}")
    bounds = robot_config.boundaries
    margin = float(robot_config.workspace_margin_mm)
    y_extension = robot_config.y_workspace_extension_mm(COLLAR_GRASP_YAW_DEG)
    hard_bounds = {
        "x_min_mm": float(bounds.x_min + margin) if bounds.x_min is not None else None,
        "x_max_mm": float(bounds.x_max - margin) if bounds.x_max is not None else None,
        "y_min_mm": float(bounds.y_min - y_extension + margin) if bounds.y_min is not None else None,
        "y_max_mm": float(bounds.y_max + y_extension - margin) if bounds.y_max is not None else None,
        "lateral_points_mm": bounds.lateral_points_mm,
        "z_min_mm": float(bounds.z_min + robot_config.lower_z_margin_mm),
        "z_max_mm": float(bounds.z_max - margin) if bounds.z_max is not None else None,
    }
    support_context = ""
    if grasp_height_plan.get("support_layer_active"):
        support_context = (
            " A local sponge support ring is confirmed for this grasp. The measured "
            f"support top is z={float(grasp_height_plan.get('local_support_z_mm')):.2f} mm "
            f"and the hard support floor is z={float(grasp_height_plan.get('support_floor_z_mm')):.2f} mm; "
            "use these local values for the final low release rather than treating the "
            "distant exposed tabletop as the contact surface."
        )
    retry_context = ""
    if previous_proposal is not None or validation_error:
        retry_context = (
            "\n\nCONTROLLER FEEDBACK / REPLAN REQUIRED: the previous numeric proposal was rejected "
            "before physical execution. Treat the exact feedback below as hard evidence and "
            "make a materially different numeric decision; do not merely restate or slightly "
            "perturb the rejected far-X/high-Z plan. Recompute the far transport and all three "
            "retreat waypoints together. If automatic far-X repair reports that no tested "
            "transport is controller-valid, choose a shorter reachable retreat and/or a safer "
            "high pose while preserving every structural rule. Remember that the minimum legal "
            "far_x is derived from the total balanced retreat distance, so shortening the retreat "
            "also lowers that required far_x. Return one complete corrected "
            "proposal. Never bypass controller IK or any workspace check.\n"
            f"Previous proposal: {json.dumps(previous_proposal, ensure_ascii=False)}\n"
            f"Exact validation error and controller trials: {validation_error or 'unknown'}"
        )
    prompt = (
        "Plan this one standalone garment manipulation task: grasp the already selected shirt "
        "collar point, lift the held collar to a high safe pose, travel toward +X as far as you "
        "judge useful and reachable, then lay the garment down gradually by moving backward "
        "toward -X while descending through multiple waypoints. You must choose all remaining "
        "numeric motion values after the runtime-fixed grasp target from the measured geometry "
        "and limits below. Do not use a "
        "fixed canned far-X, retreat distance, lift height, or descent spacing. The collar "
        f"grasp orientation is fixed at yaw={COLLAR_GRASP_YAW_DEG:.0f} degrees relative to Home "
        "so the gripper is perpendicular to the collar; do not choose another yaw. This is a "
        "full experiment, not a short probe and not an online hold-check; do not release early.\n\n"
        f"Selected collar pixel: Camera A {list(selection.pixel_xy)}\n"
        f"Visual selection reason: {selection.reason}\n"
        f"Visual evidence: {json.dumps(list(selection.evidence), ensure_ascii=False)}\n"
        f"Measured collar surface XYZ mm: {surface.tolist()}\n"
        f"Runtime-authoritative grasp TCP XYZ mm: {grasp_target.tolist()}\n"
        f"Grasp-height resolution: {json.dumps(dict(grasp_height_plan), ensure_ascii=False)}\n"
        f"Fitted table plane: z_mm = {plane[0]}*x_mm + {plane[1]}*y_mm + {plane[2]}\n"
        f"Support-layer context:{support_context or ' no confirmed local sponge support.'}\n"
        f"Hard Cartesian bounds after margins (fixed yaw={COLLAR_GRASP_YAW_DEG:.0f} degrees; Y includes "
        f"{y_extension:g} mm TCP-center allowance): {json.dumps(hard_bounds)}\n"
        f"Fixed tool roll/pitch deg: [{robot_config.orientation_roll_deg}, "
        f"{robot_config.orientation_pitch_deg}]; action yaw is relative to Home and fixed at "
        f"{COLLAR_GRASP_YAW_DEG:.0f} degrees.\n\n"
        "Required action shape: approach the grounded collar from above; the runtime will open, "
        "move to the exact runtime-authoritative grasp TCP XYZ, and close. Do not choose or revise "
        "the grasp Z. First lift nearly vertically "
        "to clear the garment/table; then, before any +X transport, move the held collar to "
        "Y=0 while maintaining the high Z. The runtime inserts shake_open at that high centered "
        "pose. After the shake, do not lower the held garment. Set the legacy "
        "pretransport_lower_xyz_mm to exactly the same [x, y, z] as y0_center_xyz_mm: "
        "despite its legacy name it is an in-place no-op transition, with no X/Y/Z change. "
        "Only after that transition move predominantly in +X to the farthest "
        "useful and controller-reachable pose. All meaningful Z decrease must begin only after "
        "the far +X point, during the backward retreat. The legacy pre-transport point must "
        "preserve the high Z exactly; it is not a lowering budget. If the available +X distance "
        "cannot support a low final release within the balanced descent/retreat ratio, choose "
        "a different reachable high pose or retreat geometry rather than releasing high. "
        "rather than lowering before the far transport. The vertical lift and Y=0 centering "
        "point must reach a genuinely high but controller-reachable pose: at least halfway "
        "from the grounded grasp Z to z_max when configured (null means no ceiling), but they do not need to be within "
        "8 mm of z_max. Keep the far +X point at that same high Z, then descend only during "
        "retreat. "
        "Use exactly three subsequent move "
        f"waypoints whose X decreases by at least {MIN_DESCENT_RETREAT_MM:.0f} mm on every leg "
        "and whose Z also decreases strictly. Every descending-retreat leg, including far point "
        "to descent point 1, must keep a balanced descent/retreat ratio: each Z descent divided "
        f"by X retreat must stay between {MIN_DESCENT_Z_PER_X_RATIO:.1f} and "
        f"{MAX_DESCENT_Z_PER_X_RATIO:.1f}. Stay near Y=0, so the garment is "
        "laid down slowly while retreating. The final release TCP must be no more than "
        f"{MAX_RELEASE_HEIGHT_ABOVE_TABLE_MM:.0f} mm above the fitted table at its final X/Y; "
        "a high-air release is invalid even if all three waypoints descend. Release only at "
        "that final low safe waypoint; "
        "retract upward; finish with home. Keep all waypoints above the fitted table, inside "
        "the hard bounds, and at 14 actions or fewer. Choose the +X reach and lowered transport "
        "height yourself. "
        "The controller will independently validate the complete interpolated trajectory. "
        "Return JSON only.\n\n"
        "Return only the compact numeric plan requested by the JSON schema. Every XYZ field is "
        "an array [x_mm, y_mm, z_mm]. descent_xyz_mm must contain exactly three XYZ points. "
        "The runtime will insert open_gripper, close_gripper, release, and home in the required "
        "order; do not return an action DSL. The visual collar decision is already complete; "
        "plan only from the supplied grounded measurements and do not request or inspect files."
        + retry_context
    )
    xyz_schema = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "garment_observation": {"type": "string", "minLength": 1},
            "reveal_strategy": {"type": "string", "minLength": 1},
            "expected_observation": {"type": "string", "minLength": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "safety_notes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {"type": "string", "minLength": 1},
            },
            "yaw_deg": {"type": "number"},
            "approach_xyz_mm": xyz_schema,
            "lift_xyz_mm": xyz_schema,
            "y0_center_xyz_mm": xyz_schema,
            "pretransport_lower_xyz_mm": xyz_schema,
            "far_transport_xyz_mm": xyz_schema,
            "descent_xyz_mm": {
                "type": "array",
                "items": xyz_schema,
                "minItems": 3,
                "maxItems": 3,
            },
            "retract_xyz_mm": xyz_schema,
        },
        "required": [
            "garment_observation",
            "reveal_strategy",
            "expected_observation",
            "confidence",
            "safety_notes",
            "yaw_deg",
            "approach_xyz_mm",
            "lift_xyz_mm",
            "y0_center_xyz_mm",
            "pretransport_lower_xyz_mm",
            "far_transport_xyz_mm",
            "descent_xyz_mm",
            "retract_xyz_mm",
        ],
    }
    command = [
        str(executable),
        "--print",
        prompt,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema, separators=(",", ":")),
        "--permission-mode",
        "dontAsk",
        "--tools",
        "",
        "--safe-mode",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--system-prompt",
        (
            "You are a read-only robot motion planner for one collar high-lift and gradual "
            "laydown experiment. Use only the supplied grounded measurements, task contract, "
            "and prior validation feedback. Choose the numeric Cartesian waypoints yourself, "
            "never control the robot, and return only the requested JSON."
        ),
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=int(timeout_s),
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CollarLiftRetreatError(
            f"Claude collar motion planner timed out after {int(timeout_s)} seconds"
        ) from exc
    duration_s = time.monotonic() - started
    if completed.returncode != 0:
        raise CollarLiftRetreatError(
            f"Claude collar motion planner exited with {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    payload = _json_from_claude_text(completed.stdout)
    return payload, {
        "prompt": prompt,
        "command": command,
        "duration_s": duration_s,
        "returncode": completed.returncode,
        "raw_stdout": completed.stdout,
        "raw_stderr": completed.stderr,
        "payload": payload,
    }


def insert_runtime_shake_open_after_y0_center(
    proposal: ExplorationProposal,
) -> ExplorationProposal:
    """Insert the fixed host-owned composite shake-open after Y=0 centering."""

    actions = [dict(action) for action in proposal.actions]
    if any(action["name"] == "shake_open" for action in actions):
        raise CollarLiftRetreatError(
            "collar proposal already contains a shake_open action"
        )
    close_index = next(
        (index for index, action in enumerate(actions) if action["name"] == "close_gripper"),
        None,
    )
    if close_index is None:
        raise CollarLiftRetreatError(
            "cannot insert shake_open without a grasp close action"
        )
    release_index = next(
        (
            index
            for index, action in enumerate(actions[close_index + 1 :], start=close_index + 1)
            if action["name"] == "open_gripper"
        ),
        None,
    )
    if release_index is None:
        raise CollarLiftRetreatError(
            "cannot insert shake_open without a post-grasp release"
        )
    post_grasp_move_indices = [
        index
        for index, action in enumerate(
            actions[close_index + 1 : release_index],
            start=close_index + 1,
        )
        if action["name"] == "move"
    ]
    if len(post_grasp_move_indices) < 2:
        raise CollarLiftRetreatError(
            "cannot insert shake_open before the high Y=0 centering move"
        )
    insertion_index = post_grasp_move_indices[1] + 1
    actions.insert(insertion_index, {"name": "shake_open", "args": {}})
    if len(actions) > COLLAR_EXECUTION_MAX_ACTIONS:
        raise CollarLiftRetreatError(
            "collar execution exceeds "
            f"{COLLAR_EXECUTION_MAX_ACTIONS} actions after shake_open insertion"
        )
    return ExplorationProposal(
        garment_observation=proposal.garment_observation,
        reveal_strategy=proposal.reveal_strategy,
        confidence=proposal.confidence,
        actions=tuple(actions),
        expected_observation=proposal.expected_observation,
        safety_notes=proposal.safety_notes,
        skill_invocations=proposal.skill_invocations,
        selected_grasp=proposal.selected_grasp,
    )


def _proposal_with_repaired_transport_x(
    proposal: ExplorationProposal,
    *,
    far_action_index: int,
    descent_action_indices: Sequence[int],
    far_x_mm: float,
    original_far_x_mm: float,
) -> ExplorationProposal:
    """Shorten far-X while preserving Claude's balanced descent/retreat geometry."""

    actions = [
        {"name": action["name"], "args": dict(action.get("args", {}))}
        for action in proposal.actions
    ]
    actions[far_action_index]["args"]["x"] = float(far_x_mm)
    if not descent_action_indices:
        raise CollarLiftRetreatError(
            "automatic far-X repair requires descending retreat waypoints"
        )
    x_shift_mm = float(far_x_mm) - float(original_far_x_mm)
    redistributed_x_mm: list[float] = []
    for action_index in descent_action_indices:
        x_mm = float(actions[action_index]["args"]["x"]) + x_shift_mm
        actions[action_index]["args"]["x"] = x_mm
        redistributed_x_mm.append(x_mm)
    repair_note = (
        "Runtime validation-first IK repair shortened far transport X from "
        f"{original_far_x_mm:.3f} to {far_x_mm:.3f} mm and rebuilt the descending "
        "retreat X waypoints while preserving Claude's balanced descent/retreat ratios: "
        f"{[round(value, 3) for value in redistributed_x_mm]}; all Z/Y values were preserved."
    )
    safety_notes = tuple(proposal.safety_notes[:7]) + (repair_note,)
    return ExplorationProposal(
        garment_observation=proposal.garment_observation,
        reveal_strategy=proposal.reveal_strategy + " " + repair_note,
        confidence=proposal.confidence,
        actions=tuple(actions),
        expected_observation=proposal.expected_observation,
        safety_notes=safety_notes,
        skill_invocations=proposal.skill_invocations,
        selected_grasp=proposal.selected_grasp,
    )


def _proposal_with_repaired_pretransport_transition(
    proposal: ExplorationProposal,
) -> ExplorationProposal | None:
    """Repair Claude's legacy transition waypoint without another model call.

    ``pretransport_lower_xyz_mm`` is retained in the compact schema for backward
    compatibility, but the current experiment intentionally performs no lowering
    before the far +X transfer.  The only valid value is therefore the exact
    high, centered pose immediately before ``shake_open``.  This deterministic
    repair prevents a harmless naming/formatting mistake from consuming another
    long Claude planning attempt.
    """

    actions = [
        {"name": action["name"], "args": dict(action.get("args", {}))}
        for action in proposal.actions
    ]
    shake_indices = [
        index for index, action in enumerate(actions) if action["name"] == "shake_open"
    ]
    if len(shake_indices) != 1:
        return None
    shake_index = shake_indices[0]
    center_index = shake_index - 1
    pretransport_index = shake_index + 1
    if (
        center_index < 0
        or pretransport_index >= len(actions)
        or actions[center_index]["name"] != "move"
        or actions[pretransport_index]["name"] != "move"
    ):
        return None
    center = actions[center_index]["args"]
    pretransport = actions[pretransport_index]["args"]
    if (
        abs(float(pretransport.get("x", 0.0)) - float(center["x"])) <= 1e-6
        and abs(float(pretransport.get("y", 0.0)) - float(center["y"])) <= 1e-6
        and abs(float(pretransport.get("z", 0.0)) - float(center["z"])) <= 1e-6
    ):
        return None
    original = {
        key: float(pretransport[key])
        for key in ("x", "y", "z")
        if key in pretransport
    }
    for key in ("x", "y", "z"):
        pretransport[key] = float(center[key])
    repair_note = (
        "Runtime structural repair made the legacy pretransport transition an exact in-place "
        f"copy of the Y=0 high pose; discarded Claude value={original}."
    )
    return ExplorationProposal(
        garment_observation=proposal.garment_observation,
        reveal_strategy=proposal.reveal_strategy + " " + repair_note,
        confidence=proposal.confidence,
        actions=tuple(actions),
        expected_observation=proposal.expected_observation,
        safety_notes=tuple(proposal.safety_notes[:7]) + (repair_note,),
        skill_invocations=proposal.skill_invocations,
        selected_grasp=proposal.selected_grasp,
    )


def _proposal_with_repaired_workspace_x(
    proposal: ExplorationProposal,
    robot_config: Any,
) -> ExplorationProposal | None:
    """Shift the far-X/retreat section inward when a descent waypoint crosses x_min."""

    actions = [
        {"name": action["name"], "args": dict(action.get("args", {}))}
        for action in proposal.actions
    ]
    shake_indices = [
        index for index, action in enumerate(actions) if action["name"] == "shake_open"
    ]
    if len(shake_indices) != 1:
        return None
    shake_index = shake_indices[0]
    pretransport_index = shake_index + 1
    far_index = shake_index + 2
    if (
        far_index >= len(actions)
        or actions[pretransport_index]["name"] != "move"
        or actions[far_index]["name"] != "move"
    ):
        return None
    release_index = next(
        (
            index
            for index, action in enumerate(actions[far_index + 1 :], start=far_index + 1)
            if action["name"] == "open_gripper"
        ),
        None,
    )
    if release_index is None:
        return None
    descent_indices = [
        index
        for index in range(far_index + 1, release_index)
        if actions[index]["name"] == "move"
    ]
    if len(descent_indices) != 3:
        return None
    safe_x_min = (float(robot_config.boundaries.x_min + robot_config.workspace_margin_mm)
                  if robot_config.boundaries.x_min is not None else -math.inf)
    affected_indices = [far_index, *descent_indices]
    minimum_x = min(float(actions[index]["args"]["x"]) for index in affected_indices)
    if minimum_x >= safe_x_min - 1e-6:
        return None
    # Add a small numerical buffer so subsequent interpolation/checks do not land
    # exactly on the controller boundary.
    shift_mm = safe_x_min - minimum_x + 0.1
    far_x = float(actions[far_index]["args"]["x"]) + shift_mm
    if robot_config.boundaries.x_max is not None:
        safe_x_max = float(robot_config.boundaries.x_max - robot_config.workspace_margin_mm)
        if far_x > safe_x_max + 1e-6:
            return None
    for index in affected_indices:
        actions[index]["args"]["x"] = float(actions[index]["args"]["x"]) + shift_mm
    repair_note = (
        "Runtime structural repair shifted the far +X transport and all balanced-retreat X "
        f"waypoints by {shift_mm:.3f} mm to keep the minimum X above the safe lower bound."
    )
    return ExplorationProposal(
        garment_observation=proposal.garment_observation,
        reveal_strategy=proposal.reveal_strategy + " " + repair_note,
        confidence=proposal.confidence,
        actions=tuple(actions),
        expected_observation=proposal.expected_observation,
        safety_notes=tuple(proposal.safety_notes[:7]) + (repair_note,),
        skill_invocations=proposal.skill_invocations,
        selected_grasp=proposal.selected_grasp,
    )


def _proposal_with_repaired_transport_height(
    proposal: ExplorationProposal,
) -> ExplorationProposal | None:
    """Keep far transport at the high pre-transport Z and rebuild descent Z."""

    actions = [
        {"name": action["name"], "args": dict(action.get("args", {}))}
        for action in proposal.actions
    ]
    shake_indices = [
        index for index, action in enumerate(actions) if action["name"] == "shake_open"
    ]
    if len(shake_indices) != 1:
        return None
    shake_index = shake_indices[0]
    pretransport_index = shake_index + 1
    far_index = shake_index + 2
    if (
        far_index >= len(actions)
        or actions[pretransport_index]["name"] != "move"
        or actions[far_index]["name"] != "move"
    ):
        return None
    release_index = next(
        (
            index
            for index, action in enumerate(actions[far_index + 1 :], start=far_index + 1)
            if action["name"] == "open_gripper"
        ),
        None,
    )
    if release_index is None:
        return None
    descent_indices = [
        index
        for index in range(far_index + 1, release_index)
        if actions[index]["name"] == "move"
    ]
    if len(descent_indices) != 3:
        return None
    pretransport_z = float(actions[pretransport_index]["args"]["z"])
    far_args = actions[far_index]["args"]
    old_far_z = float(far_args["z"])
    if abs(old_far_z - pretransport_z) <= TRANSPORT_Z_TOLERANCE_MM:
        return None
    old_far_x = float(far_args["x"])
    far_args["z"] = pretransport_z
    for index in descent_indices:
        point = actions[index]["args"]
        retreat_mm = old_far_x - float(point["x"])
        point["z"] = pretransport_z - retreat_mm
    final_z = float(actions[descent_indices[-1]]["args"]["z"])
    retract_index = release_index + 1
    if retract_index < len(actions) and actions[retract_index]["name"] == "move":
        retract = actions[retract_index]["args"]
        if float(retract["z"]) <= final_z:
            retract["z"] = final_z + 10.0
    repair_note = (
        "Runtime structural repair restored far +X transport to the high pre-transport Z "
        f"({pretransport_z:.3f} mm) and rebuilt all descent Z values from their X retreat "
        "distances to preserve the selected descent/retreat balance."
    )
    return ExplorationProposal(
        garment_observation=proposal.garment_observation,
        reveal_strategy=proposal.reveal_strategy + " " + repair_note,
        confidence=proposal.confidence,
        actions=tuple(actions),
        expected_observation=proposal.expected_observation,
        safety_notes=tuple(proposal.safety_notes[:7]) + (repair_note,),
        skill_invocations=proposal.skill_invocations,
        selected_grasp=proposal.selected_grasp,
    )


def validate_controller_with_auto_far_x_repair(
    robot_config: Any,
    proposal: ExplorationProposal,
    *,
    max_search_steps: int = AUTO_IK_FAR_X_SEARCH_STEPS,
    arm: Any | None = None,
    debug_callback: Any | None = None,
) -> tuple[ExplorationProposal, Any, dict[str, Any] | None]:
    """Validate once, then repair an unreachable far-X without another Claude call."""

    owns_arm = arm is None
    if arm is None:
        try:
            from xarm.wrapper import XArmAPI
        except ImportError as exc:
            raise CollarLiftRetreatError(
                "xarm package is required for controller IK validation"
            ) from exc
        arm = XArmAPI(robot_config.robot_ip)
    try:
        if not getattr(arm, "connected", True):
            raise CollarLiftRetreatError(
                f"unable to connect to xArm at {robot_config.robot_ip}"
            )
        try:
            validation = _controller_trajectory_with_arm(
                arm,
                robot_config,
                list(proposal.actions),
            )
            return proposal, validation, None
        except SafetyError as initial_error:
            if debug_callback is not None:
                debug_callback(f"controller IK initial trajectory: REJECTED {initial_error}")
            actions = list(proposal.actions)
            shake_open_indices = [
                index
                for index, action in enumerate(actions)
                if action["name"] == "shake_open"
            ]
            if len(shake_open_indices) != 1:
                raise
            shake_open_index = shake_open_indices[0]
            center_index = shake_open_index - 1
            pretransport_index = shake_open_index + 1
            far_index = pretransport_index + 1
            first_descent_index = far_index + 1
            if (
                center_index < 0
                or first_descent_index >= len(actions)
                or actions[center_index]["name"] != "move"
                or actions[pretransport_index]["name"] != "move"
                or actions[far_index]["name"] != "move"
                or actions[first_descent_index]["name"] != "move"
                or f"action {far_index + 1} " not in str(initial_error)
            ):
                raise

            pretransport_x = float(actions[pretransport_index]["args"]["x"])
            original_far_x = float(actions[far_index]["args"]["x"])
            release_index = next(
                (
                    index
                    for index, action in enumerate(actions[first_descent_index:], start=first_descent_index)
                    if action["name"] == "open_gripper"
                ),
                None,
            )
            if release_index is None:
                raise
            descent_action_indices = [
                index
                for index in range(first_descent_index, release_index)
                if actions[index]["name"] == "move"
            ]
            if len(descent_action_indices) != 3:
                raise
            safe_x_min = (float(robot_config.boundaries.x_min + robot_config.workspace_margin_mm)
                          if robot_config.boundaries.x_min is not None else -math.inf)
            required_total_retreat = original_far_x - float(
                actions[descent_action_indices[-1]]["args"]["x"]
            )
            minimum_far_x = max(
                pretransport_x + AUTO_IK_MIN_TRANSPORT_MM,
                safe_x_min + required_total_retreat,
            )
            if minimum_far_x >= original_far_x - 1e-6:
                raise

            trials: list[dict[str, Any]] = []

            def try_far_x(far_x: float) -> tuple[ExplorationProposal, Any] | None:
                candidate = _proposal_with_repaired_transport_x(
                    proposal,
                    far_action_index=far_index,
                    descent_action_indices=descent_action_indices,
                    far_x_mm=far_x,
                    original_far_x_mm=original_far_x,
                )
                try:
                    candidate_validation = _controller_trajectory_with_arm(
                        arm,
                        robot_config,
                        list(candidate.actions),
                    )
                except SafetyError as exc:
                    trials.append(
                        {
                            "far_x_mm": far_x,
                            "status": "IK_REJECTED",
                            "error": str(exc),
                        }
                    )
                    if debug_callback is not None:
                        debug_callback(
                            f"controller IK far-X trial={far_x:.3f}: REJECTED {exc}"
                        )
                    return None
                trials.append({"far_x_mm": far_x, "status": "IK_ACCEPTED"})
                if debug_callback is not None:
                    debug_callback(f"controller IK far-X trial={far_x:.3f}: ACCEPTED")
                return candidate, candidate_validation

            lowest = try_far_x(minimum_far_x)
            if lowest is None:
                raise CollarLiftRetreatError(
                    "automatic far-X repair found no controller-valid +X transport "
                    f"at the minimum legal far_x={minimum_far_x:.3f} mm; "
                    f"initial controller error={initial_error}; "
                    "controller trials for Claude's next decision: "
                    f"{json.dumps(trials, ensure_ascii=False)}"
                ) from initial_error

            best_x = minimum_far_x
            best_candidate, best_validation = lowest
            rejected_x = original_far_x
            for _ in range(max(1, int(max_search_steps))):
                if rejected_x - best_x <= 1.0:
                    break
                midpoint = 0.5 * (best_x + rejected_x)
                tested = try_far_x(midpoint)
                if tested is None:
                    rejected_x = midpoint
                else:
                    best_x = midpoint
                    best_candidate, best_validation = tested

            backed_off_x = max(minimum_far_x, best_x - AUTO_IK_BACKOFF_MM)
            if backed_off_x < best_x - 1e-6:
                backed_off = try_far_x(backed_off_x)
                if backed_off is not None:
                    best_x = backed_off_x
                    best_candidate, best_validation = backed_off

            repair = {
                "status": "AUTO_REPAIRED",
                "strategy": "binary_search_far_transport_x_preserving_balanced_retreat",
                "initial_error": str(initial_error),
                "far_action_index": far_index,
                "descent_action_indices": descent_action_indices,
                "original_far_x_mm": original_far_x,
                "minimum_far_x_mm": minimum_far_x,
                "selected_far_x_mm": best_x,
                "selected_descent_x_mm": [
                    float(best_candidate.actions[index]["args"]["x"])
                    for index in descent_action_indices
                ],
                "backoff_mm": AUTO_IK_BACKOFF_MM,
                "search_steps": max_search_steps,
                "trials": trials,
            }
            return best_candidate, best_validation, repair
    finally:
        if owns_arm and getattr(arm, "connected", False):
            arm.disconnect()


def collar_motion_payload_to_proposal(
    payload: Any,
    *,
    selection: CollarSelection,
    grasp_target_xyz_mm: Sequence[float],
) -> ExplorationProposal:
    """Convert Claude's compact numeric plan into the restricted RobotAPI action DSL."""

    required = {
        "garment_observation",
        "reveal_strategy",
        "expected_observation",
        "confidence",
        "safety_notes",
        "yaw_deg",
        "approach_xyz_mm",
        "lift_xyz_mm",
        "y0_center_xyz_mm",
        "pretransport_lower_xyz_mm",
        "far_transport_xyz_mm",
        "descent_xyz_mm",
        "retract_xyz_mm",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise CollarLiftRetreatError(
            "Claude compact collar motion payload has missing or unknown fields"
        )
    if selection.pixel_xy is None:
        raise CollarLiftRetreatError("compact motion conversion requires a selected collar pixel")
    authoritative_grasp = np.asarray(grasp_target_xyz_mm, dtype=np.float64)
    if authoritative_grasp.shape != (3,) or not np.all(np.isfinite(authoritative_grasp)):
        raise CollarLiftRetreatError(
            "compact motion conversion requires a finite authoritative grasp XYZ"
        )

    def xyz(name: str, value: Any) -> list[float]:
        if not isinstance(value, list) or len(value) != 3:
            raise CollarLiftRetreatError(f"{name} must contain exactly [x, y, z]")
        return [_finite_number(item, f"{name}[{index}]") for index, item in enumerate(value)]

    descent_raw = payload["descent_xyz_mm"]
    if not isinstance(descent_raw, list) or len(descent_raw) != 3:
        raise CollarLiftRetreatError("descent_xyz_mm must contain exactly three XYZ waypoints")
    # The collar grasp uses one calibrated perpendicular gripper orientation.
    # Claude still returns the field for schema compatibility, but host runtime
    # owns this safety-critical orientation and does not allow a 0/90 guess.
    _finite_number(payload["yaw_deg"], "yaw_deg")
    yaw = COLLAR_GRASP_YAW_DEG
    named_points = {
        name: xyz(name, payload[name])
        for name in (
            "approach_xyz_mm",
            "lift_xyz_mm",
            "y0_center_xyz_mm",
            "pretransport_lower_xyz_mm",
            "far_transport_xyz_mm",
            "retract_xyz_mm",
        )
    }
    descent = [xyz(f"descent_xyz_mm[{index}]", point) for index, point in enumerate(descent_raw)]

    def move(point: Sequence[float]) -> dict[str, Any]:
        return {
            "name": "move",
            "args": {
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "yaw": yaw,
            },
        }

    actions = [
        move(named_points["approach_xyz_mm"]),
        {"name": "open_gripper", "args": {}},
        move(authoritative_grasp.tolist()),
        {"name": "close_gripper", "args": {}},
        move(named_points["lift_xyz_mm"]),
        move(named_points["y0_center_xyz_mm"]),
        move(named_points["pretransport_lower_xyz_mm"]),
        move(named_points["far_transport_xyz_mm"]),
        *(move(point) for point in descent),
        {"name": "open_gripper", "args": {}},
        move(named_points["retract_xyz_mm"]),
        {"name": "home", "args": {}},
    ]
    claude_proposal = validate_global_exploration_payload(
        {
            "selected_grasp": {
                "camera": "A",
                "pixel_xy": list(selection.pixel_xy),
                "reason": selection.reason,
            },
            "garment_observation": payload["garment_observation"],
            "reveal_strategy": payload["reveal_strategy"],
            "confidence": payload["confidence"],
            "actions": actions,
            "expected_observation": payload["expected_observation"],
            "safety_notes": payload["safety_notes"],
        },
        max_actions=COLLAR_MOTION_MAX_ACTIONS,
    )
    return insert_runtime_shake_open_after_y0_center(claude_proposal)


def validate_claude_collar_motion_proposal(
    proposal: ExplorationProposal,
    *,
    selection: CollarSelection,
    grasp_height_plan: Mapping[str, Any],
    table_plane_abc: Sequence[float],
    robot_config: Any,
) -> dict[str, Any]:
    """Enforce task shape and physical envelopes without supplying Claude's numbers."""

    if selection.pixel_xy is None or proposal.selected_grasp is None:
        raise CollarLiftRetreatError("collar motion proposal is missing its selected grasp")
    if proposal.selected_grasp.get("camera") != "A" or proposal.selected_grasp.get(
        "pixel_xy"
    ) != list(selection.pixel_xy):
        raise CollarLiftRetreatError(
            "motion planner changed the collar pixel selected by the perception stage"
        )
    surface = np.asarray(grasp_height_plan.get("surface_xyz_mm"), dtype=np.float64)
    grasp = np.asarray(grasp_height_plan.get("target_xyz_mm"), dtype=np.float64)
    plane = np.asarray(table_plane_abc, dtype=np.float64)
    if surface.shape != (3,) or not np.all(np.isfinite(surface)):
        raise CollarLiftRetreatError("collar motion requires a finite measured surface XYZ")
    if grasp.shape != (3,) or not np.all(np.isfinite(grasp)):
        raise CollarLiftRetreatError("collar motion requires a finite authoritative grasp XYZ")
    actions = list(proposal.actions)
    close_indices = [i for i, action in enumerate(actions) if action["name"] == "close_gripper"]
    if len(close_indices) != 1:
        raise CollarLiftRetreatError("collar motion must close the gripper exactly once")
    close_index = close_indices[0]
    release_indices = [
        i
        for i, action in enumerate(actions[close_index + 1 :], start=close_index + 1)
        if action["name"] == "open_gripper"
    ]
    if len(release_indices) != 1:
        raise CollarLiftRetreatError("collar motion must release exactly once after grasping")
    release_index = release_indices[0]
    grasp_move_index = next(
        (i for i in range(close_index - 1, -1, -1) if actions[i]["name"] == "move"),
        None,
    )
    if grasp_move_index is None:
        raise CollarLiftRetreatError("collar motion requires a grounded grasp move before close")
    grasp_args = actions[grasp_move_index]["args"]
    commanded_grasp = np.asarray(
        [float(grasp_args["x"]), float(grasp_args["y"]), float(grasp_args["z"])],
        dtype=np.float64,
    )
    grounded_xyz_error = float(np.linalg.norm(commanded_grasp - grasp))
    if grounded_xyz_error > 0.25:
        raise CollarLiftRetreatError(
            "collar grasp move does not use the runtime-authoritative XYZ: "
            f"target={grasp.tolist()}, commanded={commanded_grasp.tolist()}, "
            f"error={grounded_xyz_error:.2f} mm"
        )
    grounded_xy_error = float(np.linalg.norm(commanded_grasp[:2] - surface[:2]))
    surface_offset = float(commanded_grasp[2] - surface[2])
    if not any(action["name"] == "open_gripper" for action in actions[:grasp_move_index]):
        raise CollarLiftRetreatError("collar approach must open the gripper before the grasp move")
    post_grasp_move_items = [
        (index, action["args"])
        for index, action in enumerate(
            actions[close_index + 1 : release_index],
            start=close_index + 1,
        )
        if action["name"] == "move"
    ]
    post_grasp_moves = [args for _, args in post_grasp_move_items]
    if len(post_grasp_moves) != 7:
        raise CollarLiftRetreatError(
            "collar motion requires a vertical lift, a high Y=0 centering move, an in-place "
            "pre-transport transition, a +X transfer, and exactly three descending "
            "retreat waypoints"
        )
    lift, center, pretransport, far = post_grasp_moves[:4]
    center_action_index = post_grasp_move_items[1][0]
    pretransport_action_index = post_grasp_move_items[2][0]
    far_action_index = post_grasp_move_items[3][0]
    shake_open_indices = [
        index
        for index, action in enumerate(
            actions[close_index + 1 : release_index],
            start=close_index + 1,
        )
        if action["name"] == "shake_open"
    ]
    if len(shake_open_indices) != 1:
        raise CollarLiftRetreatError(
            "collar motion must contain exactly one runtime shake_open after Y=0 centering"
        )
    shake_open_index = shake_open_indices[0]
    if (
        shake_open_index != center_action_index + 1
        or pretransport_action_index != shake_open_index + 1
        or far_action_index != pretransport_action_index + 1
    ):
        raise CollarLiftRetreatError(
            "runtime shake_open must be followed by an in-place transition before +X transport"
        )
    lift_lateral = math.dist(
        [float(lift["x"]), float(lift["y"])],
        [float(grasp_args["x"]), float(grasp_args["y"])],
    )
    if lift_lateral > 5.0 or float(lift["z"]) <= float(grasp_args["z"]):
        raise CollarLiftRetreatError(
            "first post-grasp move must be a near-vertical upward lift"
        )
    if abs(float(center["y"])) > 3.0:
        raise CollarLiftRetreatError(
            "second post-grasp move must center the held collar at Y=0 before +X transport"
        )
    if abs(float(center["x"]) - float(lift["x"])) > 10.0:
        raise CollarLiftRetreatError(
            "Y=0 centering move must not begin the +X transport at the same time"
        )
    if math.dist(
        [float(pretransport["x"]), float(pretransport["y"])],
        [float(center["x"]), float(center["y"])],
    ) > PRETRANSPORT_XY_TOLERANCE_MM:
        raise CollarLiftRetreatError(
            "post-shake pre-transport transition must stay in place; it must not begin +X transport"
        )
    pretransport_drop_mm = float(center["z"]) - float(pretransport["z"])
    if not (
        PRETRANSPORT_MIN_Z_DROP_MM - 1e-6
        <= pretransport_drop_mm
        <= PRETRANSPORT_MAX_Z_DROP_MM + 1e-6
    ):
        raise CollarLiftRetreatError(
            "post-shake transition must preserve Z so the far transport happens before descent: "
            f"required=[{PRETRANSPORT_MIN_Z_DROP_MM:.0f}, "
            f"{PRETRANSPORT_MAX_Z_DROP_MM:.0f}] mm, got={pretransport_drop_mm:.3f} mm"
        )
    if float(far["x"]) <= float(pretransport["x"]):
        raise CollarLiftRetreatError(
            "far transport must travel toward +X only after the in-place transition"
        )
    if abs(float(far["y"])) > 3.0:
        raise CollarLiftRetreatError("the +X transfer must remain centered near Y=0")
    if abs(float(far["z"]) - float(pretransport["z"])) > TRANSPORT_Z_TOLERANCE_MM:
        raise CollarLiftRetreatError(
            "far +X transport must preserve the pre-transport Z"
        )
    bounds = robot_config.boundaries
    safe_z_max = float(bounds.z_max - robot_config.workspace_margin_mm) if bounds.z_max is not None else math.inf
    grasp_z = float(grasp_args["z"])
    minimum_high_z = grasp_z + MIN_HIGH_LIFT_FRACTION_OF_AVAILABLE_Z * (
        safe_z_max - grasp_z
    ) if bounds.z_max is not None else grasp_z
    for label, point in (("vertical lift", lift), ("Y=0 centering", center)):
        point_z = float(point["z"])
        if point_z < minimum_high_z - 1e-6 or point_z > safe_z_max + 1e-6:
            raise CollarLiftRetreatError(
                f"{label} Z must be a high but controller-reachable height in "
                f"[{minimum_high_z:.3f}, {safe_z_max:.3f}] mm; z={point_z:.3f}"
            )
    descent = post_grasp_moves[4:]
    previous = far
    descent_leg_metrics: list[dict[str, float]] = []
    for index, point in enumerate(descent, start=1):
        if abs(float(point["y"])) > 5.0:
            raise CollarLiftRetreatError(
                f"descending retreat waypoint {index} must stay near Y=0"
            )
        retreat_mm = float(previous["x"]) - float(point["x"])
        if retreat_mm < MIN_DESCENT_RETREAT_MM - 1e-6:
            raise CollarLiftRetreatError(
                f"descending retreat waypoint {index} must move backward toward -X by at "
                f"least {MIN_DESCENT_RETREAT_MM:.0f} mm; got {retreat_mm:.3f} mm"
            )
        if float(point["z"]) >= float(previous["z"]):
            raise CollarLiftRetreatError(
                f"descending retreat waypoint {index} must be lower than the prior waypoint"
            )
        descent_mm = float(previous["z"]) - float(point["z"])
        descent_retreat_ratio = descent_mm / retreat_mm
        if not (
            MIN_DESCENT_Z_PER_X_RATIO - 1e-6
            <= descent_retreat_ratio
            <= MAX_DESCENT_Z_PER_X_RATIO + 1e-6
        ):
            raise CollarLiftRetreatError(
                f"descending retreat waypoint {index} must keep a balanced descent/retreat "
                f"ratio in [{MIN_DESCENT_Z_PER_X_RATIO:.1f}, {MAX_DESCENT_Z_PER_X_RATIO:.1f}]: "
                f"retreat={retreat_mm:.3f} mm, descent={descent_mm:.3f} mm, "
                f"ratio={descent_retreat_ratio:.3f}"
            )
        descent_leg_metrics.append(
            {
                "retreat_mm": retreat_mm,
                "descent_mm": descent_mm,
                "descent_retreat_ratio": descent_retreat_ratio,
            }
        )
        previous = point
    release = descent[-1]
    release_table_z = float(
        plane[0] * float(release["x"])
        + plane[1] * float(release["y"])
        + plane[2]
    )
    support_layer_active = bool(grasp_height_plan.get("support_layer_active"))
    support_surface_z = grasp_height_plan.get("local_support_z_mm")
    support_floor_z = grasp_height_plan.get("support_floor_z_mm")
    support_surface_z = (
        float(support_surface_z)
        if isinstance(support_surface_z, (int, float))
        and math.isfinite(float(support_surface_z))
        else None
    )
    support_floor_z = (
        float(support_floor_z)
        if isinstance(support_floor_z, (int, float))
        and math.isfinite(float(support_floor_z))
        else None
    )
    # When the local ring confirms a sponge, all release/floor checks use that
    # local support surface.  The exposed global tabletop is not the contact
    # surface for this trajectory and must not veto a valid low release.
    release_reference_z = (
        support_surface_z
        if support_layer_active and support_surface_z is not None
        else release_table_z
    )
    release_height_above_global_table = float(release["z"]) - release_table_z
    release_height_above_support = float(release["z"]) - release_reference_z
    use_table_floor = bool(getattr(robot_config, "grasp_use_table_clearance_floor", True))
    enforce_support_floor = support_layer_active and support_floor_z is not None
    if (use_table_floor or enforce_support_floor) and release_height_above_support > MAX_RELEASE_HEIGHT_ABOVE_TABLE_MM + 1e-6:
        required_total_descent = max(
            0.0,
            float(far["z"])
            - (release_reference_z + MAX_RELEASE_HEIGHT_ABOVE_TABLE_MM),
        )
        required_far_x = (
            (float(bounds.x_min + robot_config.workspace_margin_mm) if bounds.x_min is not None else float(release["x"]))
            + required_total_descent / MAX_DESCENT_Z_PER_X_RATIO
        )
        raise CollarLiftRetreatError(
            "final release is still too high above the support surface and would drop the garment: "
            f"release_z={float(release['z']):.2f} mm, "
            f"support_z={release_reference_z:.2f} mm, "
            f"height_above_support={release_height_above_support:.2f} mm, "
            f"required<={MAX_RELEASE_HEIGHT_ABOVE_TABLE_MM:.2f} mm. "
            "With the balanced descent/retreat ratio limits and the current far Z, the planner "
            f"needs approximately far_x>={required_far_x:.2f} mm, or it must choose a "
            "different controller-reachable geometry; do not release high in the air."
        )
    minimum_table_clearance = float("inf")
    for index, action in enumerate(actions, start=1):
        if action["name"] != "move":
            continue
        args = action["args"]
        bounds.validate(
            args["x"],
            args["y"],
            args["z"],
            robot_config.workspace_margin_mm,
            require_complete=True,
            z_lower_margin_mm=robot_config.lower_z_margin_mm,
        )
        if use_table_floor or enforce_support_floor:
            table_z = float(plane[0] * args["x"] + plane[1] * args["y"] + plane[2])
            floor_z = (
                support_floor_z
                if enforce_support_floor
                else table_z + float(getattr(robot_config, "grasp_table_clearance_mm", 0.0))
            )
            clearance = float(args["z"]) - floor_z
            minimum_table_clearance = min(minimum_table_clearance, clearance)
            required_floor_clearance = 0.0
            if clearance < required_floor_clearance - 1e-6:
                floor_label = (
                    "local support floor" if enforce_support_floor else "fitted table"
                )
                raise CollarLiftRetreatError(
                    f"move action {index} is too close to/below the {floor_label}: "
                    f"clearance={clearance:.2f} mm, "
                    f"required_clearance={required_floor_clearance:.2f} mm"
                )
    if release_index + 2 >= len(actions):
        raise CollarLiftRetreatError("release must be followed by an upward retract and home")
    if actions[release_index + 1]["name"] != "move" or actions[-1]["name"] != "home":
        raise CollarLiftRetreatError("release must be followed by an upward retract and final home")
    retract = actions[release_index + 1]["args"]
    if float(retract["z"]) <= float(descent[-1]["z"]):
        raise CollarLiftRetreatError("post-release retract must move upward")
    return {
        "grounded_grasp_xy_error_mm": grounded_xy_error,
        "grounded_grasp_xyz_error_mm": grounded_xyz_error,
        "grasp_surface_offset_mm": surface_offset,
        "runtime_authoritative_grasp_xyz_mm": grasp.tolist(),
        "fixed_collar_grasp_yaw_deg": COLLAR_GRASP_YAW_DEG,
        "post_grasp_move_count": len(post_grasp_moves),
        "descent_waypoint_count": len(descent),
        "descent_retreat_ratio": f"{MIN_DESCENT_Z_PER_X_RATIO:.1f}-{MAX_DESCENT_Z_PER_X_RATIO:.1f}_Z_per_X",
        "descent_leg_metrics": descent_leg_metrics,
        "claude_chosen_lift": dict(lift),
        "claude_chosen_y0_center": dict(center),
        "runtime_shake_open_action_index": shake_open_index,
        "claude_chosen_pretransport_lower": dict(pretransport),
        "pretransport_z_drop_mm": pretransport_drop_mm,
        "claude_chosen_far_transport": dict(far),
        "claude_chosen_release": dict(descent[-1]),
        "release_height_above_table_mm": release_height_above_global_table,
        "release_height_above_support_mm": release_height_above_support,
        "support_layer_active": support_layer_active,
        "support_surface_z_mm": support_surface_z,
        "support_floor_z_mm": support_floor_z,
        "maximum_release_height_above_table_mm": MAX_RELEASE_HEIGHT_ABOVE_TABLE_MM,
        "minimum_table_clearance_mm": minimum_table_clearance,
        "table_grasp_safety_checks_enabled": use_table_floor,
    }


def table_plane_coefficients(result: Mapping[str, Any]) -> tuple[float, float, float]:
    raw = (
        result.get("depth_fusion", {})
        .get("table_plane", {})
        .get("coefficients", {})
    )
    try:
        coefficients = (
            _finite_number(raw["a"], "table plane a"),
            _finite_number(raw["b"], "table plane b"),
            _finite_number(raw["c_mm"], "table plane c_mm"),
        )
    except (KeyError, TypeError) as exc:
        raise CollarLiftRetreatError("saved perception lacks a fitted table plane") from exc
    return coefficients


def validate_grounded_collar_grasp_feasibility(
    *,
    measurement: Mapping[str, Any],
    table_plane_abc: Sequence[float],
    robot_config: Any,
) -> dict[str, Any]:
    """Resolve the one runtime-authoritative collar grasp height before planning."""

    try:
        resolution = resolve_grasp_height(
            measurement=measurement,
            table_plane_abc=table_plane_abc,
            robot_config=robot_config,
        )
        # This is a cheap deterministic check.  Do it before invoking Claude's
        # long numeric planner: a measured collar point outside the calibrated
        # Cartesian workspace can never become legal by replanning the retreat.
        try:
            robot_config.validate_workspace_pose(
                resolution.target_xyz_mm[0],
                resolution.target_xyz_mm[1],
                resolution.target_xyz_mm[2],
                relative_yaw_deg=COLLAR_GRASP_YAW_DEG,
                require_complete=True,
            )
        except Exception as exc:
            # Keep the public error type stable while preserving the exact
            # calibrated boundary and target for a fast, actionable failure.
            raise CollarLiftRetreatError(
                "runtime-authoritative collar grasp target is outside the safe "
                "Cartesian workspace; choose another measured collar pixel rather "
                f"than replanning motion: target={list(resolution.target_xyz_mm)}, {exc}"
            ) from exc
        return resolution.as_dict()
    except GraspHeightError as exc:
        raise CollarLiftRetreatError(
            "Claude/reference located collar fabric, but the shared grasp-height policy "
            f"could not produce a legal engaged grasp: {exc}"
        ) from exc


def _write_json(path: Path, payload: Any) -> None:
    def jsonable(value: Any) -> Any:
        if is_dataclass(value):
            return jsonable(asdict(value))
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _selection_overlay(image_path: Path, selection: CollarSelection, output: Path) -> None:
    if selection.pixel_xy is None:
        return
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    x_px, y_px = selection.pixel_xy
    radius = 18
    draw.ellipse((x_px - radius, y_px - radius, x_px + radius, y_px + radius), outline=(255, 40, 40), width=5)
    draw.line((x_px - 28, y_px, x_px + 28, y_px), fill=(255, 230, 30), width=3)
    draw.line((x_px, y_px - 28, x_px, y_px + 28), fill=(255, 230, 30), width=3)
    if selection.neck_label_pixel_xy is not None:
        label_x, label_y = selection.neck_label_pixel_xy
        draw.ellipse(
            (label_x - 10, label_y - 10, label_x + 10, label_y + 10),
            outline=(40, 220, 255),
            width=4,
        )
        draw.text(
            (label_x + 12, label_y - 10),
            "NECK LABEL",
            fill=(40, 220, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    if (
        selection.neck_label_pixel_xy is not None
        and selection.torso_landmark_pixel_xy is not None
    ):
        torso_x, torso_y = selection.torso_landmark_pixel_xy
        draw.line(
            (*selection.neck_label_pixel_xy, torso_x, torso_y),
            fill=(255, 120, 40),
            width=4,
        )
        draw.ellipse(
            (torso_x - 8, torso_y - 8, torso_x + 8, torso_y + 8),
            outline=(255, 120, 40),
            width=3,
        )
        draw.text(
            (torso_x + 10, torso_y - 8),
            "TORSO SIDE",
            fill=(255, 120, 40),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    draw.rectangle((8, 8, 440, 48), fill=(0, 0, 0))
    draw.text((18, 18), f"CLAUDE COLLAR GRASP ({x_px}, {y_px})", fill=(255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def run_standalone_experiment(args: argparse.Namespace) -> int:
    debug_enabled = not bool(getattr(args, "quiet", False))
    debug_started = time.monotonic()

    def debug(message: str) -> None:
        if debug_enabled:
            elapsed = time.monotonic() - debug_started
            print(f"[collar-debug +{elapsed:7.1f}s] {message}", flush=True)

    root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    robot_path = Path(args.robot_config).resolve() if args.robot_config else None
    session = _load_or_create_session(root, run_dir, args.run_id, robot_path)
    perception_path = Path(args.perception_config)
    if not perception_path.is_absolute():
        perception_path = root / perception_path
    perception_config = PerceptionConfig.load(root, perception_path.resolve())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = session.results / "collar_lift_retreat" / stamp
    output.mkdir(parents=True, exist_ok=False)
    debug(
        f"START run_dir={session.run_dir} output={output} "
        f"real={bool(args.enable_real)} reuse_perception={bool(args.reuse_latest_perception)}"
    )
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "run_dir": str(session.run_dir),
        "output_dir": str(output),
        "enable_real": bool(args.enable_real),
        "physical_commands_sent": False,
    }
    _write_json(output / "summary.json", summary)
    try:
        if args.reuse_latest_perception:
            debug("perception: reusing latest saved perception")
            saved, saved_path = _load_latest_perception(session)
            if saved is None or saved_path is None:
                raise CollarLiftRetreatError("run has no saved perception to reuse")
            summary["perception"] = {"status": "reused", "result": str(saved_path)}
            debug(f"perception: reused {saved_path}")
        else:
            if args.enable_real:
                debug("perception: moving robot to calibrated perception pose")
                positioning = move_robot_to_perception_position(session.robot_config)
                summary["perception_position"] = positioning
                if args.settle_s > 0:
                    debug(f"perception: settling for {args.settle_s:.2f}s")
                    time.sleep(float(args.settle_s))
            debug("perception: capturing Camera A/B RGB-D frames")
            frames = capture_two_view_rgbd(perception_config)
            debug("perception: running garment localization and saving depth fusion")
            perception = session.locate_cloth_center(perception_config, frames=frames)
            saved, saved_path = _load_latest_perception(session)
            if saved is None or saved_path is None:
                raise CollarLiftRetreatError("perception completed without a saved result")
            summary["perception"] = {
                "status": "captured",
                "result": str(saved_path),
                "center_base_mm": perception.get("center_base_mm"),
            }
            debug(
                f"perception: saved={saved_path} center_base_mm={perception.get('center_base_mm')}"
            )
        images = global_perception_image_paths(saved, saved_path)
        molmo_dir = output / "molmo_collar_anchors"
        if args.no_molmo:
            molmo_manifest = None
            reference_source = root / "data" / "reference" / "flat_garment_reference"
            reference_dir = output / "no_molmo_reference"
            reference_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "camera_A_flat_reference.png",
                "camera_A_flat_reference_anchors.png",
                "reference_anchors.json",
            ):
                source = reference_source / name
                if not source.is_file():
                    raise CollarLiftRetreatError(
                        f"no-Molmo experiment is missing reference asset: {source}"
                    )
                shutil.copy2(source, reference_dir / name)
            reference_path = reference_dir / "reference_anchors.json"
            debug(
                "molmo: SKIPPED by --no-molmo; using Camera A RGB-D, Claude, and "
                f"flat reference from {reference_dir}"
            )
        else:
            debug(
                f"molmo: starting semantic anchors model={args.molmo_model} "
                f"timeout={args.molmo_timeout_s}s"
            )
            try:
                molmo_manifest = run_molmo_semantic_anchor_pipeline(
                    project_root=root,
                    perception_dir=session.workspace / "perception_views",
                    artifact_dir=molmo_dir,
                    confidence_threshold=args.molmo_confidence_threshold,
                    molmo_python=(
                        Path(args.molmo_python).expanduser().resolve()
                        if args.molmo_python
                        else None
                    ),
                    model=args.molmo_model,
                    timeout_s=args.molmo_timeout_s,
                    local_files_only=not args.molmo_allow_download,
                    keypoint_specs=COLLAR_MOLMO_SPECS,
                    cameras=("A",),
                    max_anchors=len(COLLAR_MOLMO_SPECS),
                    install=True,
                )
            except MolmoKeypointPipelineError as exc:
                raise CollarLiftRetreatError(f"Molmo collar annotation failed: {exc}") from exc
            debug(
                f"molmo: completed status={molmo_manifest.get('status')} "
                f"anchors={molmo_manifest.get('anchor_count')}"
            )
            reference_path = molmo_dir / "flat_reference" / "reference_anchors.json"
            if not reference_path.is_file():
                raise CollarLiftRetreatError(
                    "Molmo collar annotation did not copy flat reference anchors"
                )
        try:
            reference_anchors = json.loads(reference_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CollarLiftRetreatError("flat reference anchors are unreadable") from exc
        selector_evidence = list(images)
        evidence_candidates = (
            [path for path in reference_dir.iterdir() if path.is_file()]
            if args.no_molmo
            else [
                molmo_dir / "camera_A_semantic_anchors.png",
                molmo_dir / "camera_A_semantic_anchor_diagnostics.png",
                molmo_dir / "flat_reference" / "camera_A_flat_reference.png",
                molmo_dir / "flat_reference" / "camera_A_flat_reference_anchors.png",
            ]
        )
        for candidate in evidence_candidates:
            if candidate.is_file():
                selector_evidence.append(candidate.resolve())
        summary["molmo_collar_anchors"] = {
            "status": "SKIPPED" if args.no_molmo else molmo_manifest.get("status"),
            "anchor_count": 0 if args.no_molmo else molmo_manifest.get("anchor_count"),
            "manifest": None if args.no_molmo else str(molmo_dir / "molmo_semantic_anchors.json"),
            "reference_anchors": str(reference_path),
        }
        debug(
            f"claude-selector: starting with {len(selector_evidence)} evidence files "
            f"timeout={args.claude_timeout_s}s"
        )
        selection, claude_log = invoke_claude_collar_selector(
            selector_evidence,
            run_dir=session.run_dir,
            image_width=perception_config.width,
            image_height=perception_config.height,
            molmo_manifest=molmo_manifest,
            reference_anchors=reference_anchors,
            binary=args.claude_binary,
            timeout_s=args.claude_timeout_s,
            min_confidence=args.min_collar_confidence,
        )
        _write_json(output / "claude_collar_selection.json", claude_log)
        debug(
            f"claude-selector: status={selection.status} pixel={selection.pixel_xy} "
            f"confidence={selection.confidence:.3f} duration={claude_log.get('duration_s', 0.0):.1f}s"
        )
        summary["selection"] = selection.as_dict()
        if selection.status != "SELECTED" or selection.pixel_xy is None:
            raise CollarLiftRetreatError(f"Claude did not find a safe collar grasp: {selection.reason}")
        camera_a = next(
            (path for path in images if path.name == "camera_0_A.png"),
            None,
        )
        if camera_a is not None:
            _selection_overlay(camera_a, selection, output / "collar_grasp_overlay.png")
        try:
            debug(
                f"grounding: sampling Camera A local surface at pixel={selection.pixel_xy}"
            )
            measurement = GarmentGrounding(
                session.workspace / "perception_views"
            ).sample_local_surface("A", selection.pixel_xy[0], selection.pixel_xy[1], radius_px=3)
        except GroundingToolError as exc:
            raise CollarLiftRetreatError(f"collar grounding failed: {exc}") from exc
        if measurement.get("valid") is not True:
            raise CollarLiftRetreatError(
                f"selected collar pixel has no valid calibrated surface: {measurement.get('reason')}"
            )
        table_plane = table_plane_coefficients(saved)
        summary["measurement"] = measurement
        debug(
            "grounding: "
            f"surface={measurement.get('base_xyz_median_mm')} "
            f"table={measurement.get('table_z_median_mm')} valid={measurement.get('valid')}"
        )
        try:
            grasp_height_plan = validate_grounded_collar_grasp_feasibility(
                measurement=measurement,
                table_plane_abc=table_plane,
                robot_config=session.robot_config,
            )
        except CollarLiftRetreatError as exc:
            summary["grasp_height_resolution"] = {
                "valid": False,
                "error": str(exc),
            }
            raise
        summary["grasp_height_resolution"] = grasp_height_plan
        debug(
            f"grasp-height: target={grasp_height_plan.get('target_xyz_mm')} "
            f"compression={grasp_height_plan.get('achieved_compression_mm')} "
            f"table_clearance={grasp_height_plan.get('table_clearance_lower_z_mm')}"
        )
        source_path = session.workspace / "_claude_collar_lift_retreat.py"
        previous_proposal: Mapping[str, Any] | None = None
        validation_error: str | None = None
        planning_attempts: list[dict[str, Any]] = []
        proposal: ExplorationProposal | None = None
        source = ""
        preflight: Any = None
        controller: Any = None
        motion_contract: dict[str, Any] | None = None
        for attempt_index in range(1, int(args.max_motion_replans) + 2):
            candidate: ExplorationProposal | None = None
            debug(
                f"motion[{attempt_index}]: asking Claude for numeric plan "
                f"({('with controller feedback' if validation_error else 'initial decision')})"
            )
            candidate_payload, motion_log = invoke_claude_collar_motion_planner(
                run_dir=session.run_dir,
                selection=selection,
                grasp_height_plan=grasp_height_plan,
                table_plane_abc=table_plane,
                robot_config=session.robot_config,
                binary=args.claude_binary,
                timeout_s=args.claude_timeout_s,
                previous_proposal=previous_proposal,
                validation_error=validation_error,
            )
            _write_json(
                output / f"claude_motion_attempt_{attempt_index:02d}.json",
                motion_log,
            )
            debug(
                f"motion[{attempt_index}]: Claude returned in {motion_log.get('duration_s', 0.0):.1f}s "
                f"far={candidate_payload.get('far_transport_xyz_mm')} "
                f"descent={candidate_payload.get('descent_xyz_mm')}"
            )
            attempt_record: dict[str, Any] = {
                "attempt": attempt_index,
                "payload": candidate_payload,
                "status": "VALIDATING",
            }
            try:
                candidate = collar_motion_payload_to_proposal(
                    candidate_payload,
                    selection=selection,
                    grasp_target_xyz_mm=grasp_height_plan["target_xyz_mm"],
                )
                attempt_record["proposal"] = candidate.as_dict()
                debug(
                    f"motion[{attempt_index}]: parsed {len(candidate.actions)} actions; "
                    "running structural/table/workspace validation"
                )
                automatic_structure_repair: dict[str, Any] | None = None
                try:
                    candidate_contract = validate_claude_collar_motion_proposal(
                        candidate,
                        selection=selection,
                        grasp_height_plan=grasp_height_plan,
                        table_plane_abc=table_plane,
                        robot_config=session.robot_config,
                    )
                except (CollarLiftRetreatError, SafetyError) as structural_error:
                    if isinstance(structural_error, SafetyError) and "x=" in str(structural_error):
                        repaired_candidate = _proposal_with_repaired_workspace_x(
                            candidate, session.robot_config
                        )
                        repair_type = "workspace_x_boundary"
                    elif "far +X transport must preserve the pre-transport Z" in str(
                        structural_error
                    ):
                        repaired_candidate = _proposal_with_repaired_transport_height(candidate)
                        repair_type = "far_transport_high_z"
                    else:
                        repaired_candidate = _proposal_with_repaired_pretransport_transition(
                            candidate
                        )
                        repair_type = "pretransport_in_place_transition"
                    if repaired_candidate is None:
                        raise
                    candidate = repaired_candidate
                    candidate_contract = validate_claude_collar_motion_proposal(
                        candidate,
                        selection=selection,
                        grasp_height_plan=grasp_height_plan,
                        table_plane_abc=table_plane,
                        robot_config=session.robot_config,
                    )
                    automatic_structure_repair = {
                        "type": repair_type,
                        "original_validation_error": str(structural_error),
                    }
                    attempt_record["automatic_structure_repair"] = automatic_structure_repair
                    attempt_record["proposal"] = candidate.as_dict()
                    debug(
                        f"motion[{attempt_index}]: applied deterministic repair type={repair_type}"
                    )
                candidate_source = exploration_source(candidate)
                source_path.write_text(candidate_source, encoding="utf-8")
                candidate_preflight = session.runner.preflight(source_path.name)
                if candidate_preflight.error:
                    raise CollarLiftRetreatError(
                        f"static preflight failed: {candidate_preflight.error}"
                    )
                automatic_ik_repair: dict[str, Any] | None = None
                if args.skip_controller_ik:
                    debug(f"motion[{attempt_index}]: controller IK skipped by flag")
                    candidate_controller = {"status": "SKIPPED"}
                else:
                    debug(f"motion[{attempt_index}]: validating full interpolated trajectory with controller IK")
                    (
                        candidate,
                        candidate_controller,
                        automatic_ik_repair,
                    ) = validate_controller_with_auto_far_x_repair(
                        session.robot_config,
                        candidate,
                        debug_callback=lambda message, attempt=attempt_index: debug(
                            f"motion[{attempt}]: {message}"
                        ),
                    )
                    if automatic_ik_repair is not None:
                        debug(
                            f"motion[{attempt_index}]: automatic IK repair accepted "
                            f"far_x={automatic_ik_repair.get('selected_far_x_mm')}"
                        )
                        candidate_contract = validate_claude_collar_motion_proposal(
                            candidate,
                            selection=selection,
                            grasp_height_plan=grasp_height_plan,
                            table_plane_abc=table_plane,
                            robot_config=session.robot_config,
                        )
                        candidate_source = exploration_source(candidate)
                        source_path.write_text(candidate_source, encoding="utf-8")
                        candidate_preflight = session.runner.preflight(source_path.name)
                        if candidate_preflight.error:
                            raise CollarLiftRetreatError(
                                "static preflight failed after automatic IK repair: "
                                f"{candidate_preflight.error}"
                            )
                        attempt_record["automatic_ik_repair"] = automatic_ik_repair
                        attempt_record["proposal"] = candidate.as_dict()
            except Exception as exc:
                validation_error = f"{type(exc).__name__}: {exc}"
                attempt_record["status"] = "REJECTED_BEFORE_EXECUTION"
                attempt_record["validation_error"] = validation_error
                attempt_record["feedback_for_next_claude"] = {
                    "must_replan": True,
                    "reason": validation_error,
                    "previous_payload_is_invalid": True,
                }
                planning_attempts.append(attempt_record)
                _write_json(
                    output / f"motion_validation_attempt_{attempt_index:02d}.json",
                    attempt_record,
                )
                debug(f"motion[{attempt_index}]: REJECTED before execution: {validation_error}")
                if "safe lower bound" in str(exc) or "safe upper bound" in str(exc):
                    raise CollarLiftRetreatError(
                        "deterministic workspace-boundary validation failed; "
                        "Claude replanning is not useful for this error: "
                        f"{validation_error}"
                    ) from exc
                if attempt_index > int(args.max_motion_replans):
                    raise CollarLiftRetreatError(
                        "Claude could not produce a controller-valid collar motion after "
                        f"{attempt_index} attempt(s): {validation_error}"
                    ) from exc
                previous_proposal = candidate_payload
                continue
            attempt_record["status"] = "ACCEPTED"
            attempt_record["motion_contract"] = candidate_contract
            planning_attempts.append(attempt_record)
            proposal = candidate
            source = candidate_source
            preflight = candidate_preflight
            controller = candidate_controller
            motion_contract = candidate_contract
            debug(
                f"motion[{attempt_index}]: ACCEPTED actions={len(proposal.actions)} "
                f"min_table_clearance={candidate_contract.get('minimum_table_clearance_mm')}"
            )
            break
        if proposal is None or motion_contract is None:
            raise CollarLiftRetreatError("collar motion planning ended without an accepted plan")
        summary["motion_planning"] = {
            "attempt_count": len(planning_attempts),
            "rejected_count": sum(
                item["status"] == "REJECTED_BEFORE_EXECUTION" for item in planning_attempts
            ),
            "attempts": planning_attempts,
        }
        plan = {
            "selection": selection.as_dict(),
            "measurement": measurement,
            "grasp_height_resolution": grasp_height_plan,
            "proposal": proposal.as_dict(),
            "motion_contract": motion_contract,
            "planning_attempts": planning_attempts,
            "actions": list(proposal.actions),
            "source": source,
            "preflight": preflight,
            "controller_ik": controller,
        }
        _write_json(output / "validated_plan.json", plan)
        (output / "proposal.py").write_text(source, encoding="utf-8")
        summary["validated_plan"] = str(output / "validated_plan.json")
        if not args.enable_real:
            summary["status"] = "DRY_RUN_VALIDATED"
            summary["completed_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(output / "summary.json", summary)
            debug(f"DONE dry-run validated; artifacts={output}")
            print(output)
            return 0

        recording_dir = output / "rollout_recording"
        recorder: DualRealSenseRolloutRecorder | None = None
        recording_thread: threading.Thread | None = None
        recording_result: dict[str, Any] = {}
        recording_errors: list[str] = []
        if args.record_rollout:
            debug("recording: starting rollout recorder")
            recorder = DualRealSenseRolloutRecorder(
                perception_config,
                recording_dir,
                record_bag=not args.recording_no_native,
                record_depth_video=True,
                record_composite=True,
                codec=args.recording_codec,
                warmup_frames=args.recording_warmup_frames,
            )
            recorder.start()

            def record() -> None:
                try:
                    recording_result["manifest"] = recorder.record()  # type: ignore[union-attr]
                except BaseException as exc:
                    recording_errors.append(f"{type(exc).__name__}: {exc}")

            recording_thread = threading.Thread(target=record, daemon=True, name="collar-rollout-recorder")
            recording_thread.start()
            time.sleep(0.25)
            if recording_errors:
                recorder.request_stop("recording_failed_before_execution")
                recording_thread.join(timeout=5.0)
                raise CollarLiftRetreatError(
                    "rollout recording failed before robot execution: " + recording_errors[-1]
                )
        execution: dict[str, Any] | None = None
        try:
            summary["physical_commands_sent"] = True
            debug("execution: sending validated plan to the real robot")
            execution = session.run_experiment(
                source_path.name,
                real=True,
                confirmed=True,
                notes="Standalone Claude collar high-lift +X descending-retreat experiment.",
            )
        finally:
            if recorder is not None:
                recorder.request_stop("collar_experiment_completed")
            if recording_thread is not None:
                recording_thread.join(timeout=300.0)
                if recording_thread.is_alive() and recorder is not None:
                    recording_errors.append("recording thread did not stop within 300 seconds")
                    recorder.close()
                    recording_thread.join(timeout=3.0)
        if execution is None:
            raise CollarLiftRetreatError("physical experiment returned no execution record")
        _write_json(output / "execution.json", execution)
        debug(
            f"execution: completed={execution.get('execution_completed')} "
            f"actions={len(execution.get('actions', [])) if isinstance(execution.get('actions'), list) else 'n/a'}"
        )
        if recorder is not None:
            _write_json(
                output / "rollout_recording.json",
                {
                    "status": "failed" if recording_errors else "completed",
                    "directory": str(recording_dir),
                    "manifest": recording_result.get("manifest"),
                    "errors": recording_errors,
                },
            )
        summary["execution"] = str(output / "execution.json")
        summary["status"] = (
            "COMPLETED" if execution.get("execution_completed") else "FAILED"
        )
        summary["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(output / "summary.json", summary)
        debug(f"DONE status={summary['status']} artifacts={output}")
        print(output)
        return 0 if execution.get("execution_completed") else 1
    except BaseException as exc:
        debug(f"FAILED {type(exc).__name__}: {exc}")
        summary["status"] = "FAILED"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(output / "summary.json", summary)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument("--run-dir")
    session_group.add_argument("--run-id")
    parser.add_argument(
        "--robot-config",
        default="config/robot.example.json",
        help=(
            "robot configuration JSON (default: config/robot.example.json; "
            "this disables live tabletop Z correction/flooring)"
        ),
    )
    parser.add_argument(
        "--perception-config",
        default="config/perception.free_exploration.json",
    )
    parser.add_argument("--reuse-latest-perception", action="store_true")
    parser.add_argument("--settle-s", type=float, default=2.0)
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress live [collar-debug] progress messages",
    )
    parser.add_argument(
        "--claude-timeout-s",
        type=int,
        default=900,
        help="Claude selector/planner timeout in seconds (default: 900).",
    )
    parser.add_argument("--molmo-python")
    parser.add_argument("--molmo-model", default="allenai/MolmoPoint-8B")
    parser.add_argument(
        "--no-molmo",
        action="store_true",
        help="run the standalone collar experiment without Molmo; use Claude plus RGB-D/reference only",
    )
    parser.add_argument("--molmo-timeout-s", type=int, default=900)
    parser.add_argument("--molmo-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--molmo-allow-download", action="store_true")
    parser.add_argument(
        "--min-collar-confidence",
        type=float,
        default=0.0,
        help=(
            "Optional host-side confidence floor for Claude SELECTED results "
            "(default: 0, trust Claude's SELECTED/NOT_FOUND decision)."
        ),
    )
    parser.add_argument(
        "--max-motion-replans",
        type=int,
        default=2,
        help="Maximum Claude numeric motion replans after pre-execution validation rejects a plan.",
    )
    parser.add_argument("--skip-controller-ik", action="store_true")
    parser.add_argument("--enable-real", action="store_true")
    recording = parser.add_mutually_exclusive_group()
    recording.add_argument("--record-rollout", dest="record_rollout", action="store_true", default=True)
    recording.add_argument("--no-record-rollout", dest="record_rollout", action="store_false")
    parser.add_argument("--recording-no-native", action="store_true")
    parser.add_argument("--recording-codec", default="mp4v")
    parser.add_argument("--recording-warmup-frames", type=int)
    args = parser.parse_args(argv)
    if not 0.0 <= args.min_collar_confidence <= 1.0:
        parser.error("--min-collar-confidence must be between 0 and 1")
    if not 0.0 <= args.molmo_confidence_threshold < 1.0:
        parser.error("--molmo-confidence-threshold must be in [0, 1)")
    if args.molmo_timeout_s <= 0:
        parser.error("--molmo-timeout-s must be positive")
    if args.claude_timeout_s <= 0:
        parser.error("--claude-timeout-s must be positive")
    if not 0 <= args.max_motion_replans <= 5:
        parser.error("--max-motion-replans must be between 0 and 5")
    if args.reuse_latest_perception and not (args.run_dir or args.run_id):
        parser.error("--reuse-latest-perception requires --run-dir or an existing --run-id")
    if args.skip_controller_ik and args.enable_real:
        parser.error("real execution cannot skip controller IK")
    return run_standalone_experiment(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
