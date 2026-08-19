"""Confidence-filtered Molmo semantic anchors and legacy grasp references.

This module is deliberately separate from dense RGB-D perception.  Dense
perception first saves camera RGB, full-resolution robot-base XYZ maps, a final
garment mask, and height-above-table maps. Molmo then proposes a small set of
named semantic anchors. The default path exposes accepted observations as
``Sxxx`` anchors only; local geometry later creates task ``Rxxx`` candidates
inside one selected semantic region. The retired direct Molmo→Rxxx builder is
kept as an explicit compatibility mode.

The confidence is the geometric mean of the probabilities assigned by Molmo
to the three generated point-location tokens.  It is a useful model score for
filtering within this pipeline, but is not a calibrated success probability.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw


CONFIDENCE_DEFINITION = (
    "geometric_mean_probability_of_the_three_generated_molmo_point_tokens"
)
DEFAULT_CONFIDENCE_THRESHOLD = 0.60
DEFAULT_SEMANTIC_CONFIDENCE_THRESHOLD = 0.80
RAW_IMAGE_PATTERN = re.compile(r"^camera_[0-9]+_([A-Za-z0-9_-]+)\.png$")


class MolmoKeypointPipelineError(RuntimeError):
    """Raised when inference or deterministic keypoint validation fails."""


@dataclass(frozen=True)
class KeypointSpec:
    name: str
    description: str
    color: tuple[int, int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "color": list(self.color),
        }


DEFAULT_KEYPOINTS: tuple[KeypointSpec, ...] = (
    KeypointSpec("garment_center", "geometric center of the whole garment", (255, 40, 40)),
    KeypointSpec("neckline", "collar, neckline, or neck opening", (255, 170, 0)),
    KeypointSpec("left_shoulder", "image-left shoulder or upper-left shoulder seam", (50, 180, 255)),
    KeypointSpec("right_shoulder", "image-right shoulder or upper-right shoulder seam", (80, 220, 80)),
    KeypointSpec("left_sleeve_tip", "outermost image-left sleeve tip or upper edge", (180, 80, 255)),
    KeypointSpec("right_sleeve_tip", "outermost image-right sleeve tip or upper edge", (255, 80, 190)),
    KeypointSpec("left_bottom_hem", "image-left end of the bottom hem", (80, 220, 220)),
    KeypointSpec("right_bottom_hem", "image-right end of the bottom hem", (220, 220, 60)),
    KeypointSpec("lower_left_half_center", "center of the lower image-left half of the garment", (120, 255, 120)),
    KeypointSpec("lower_right_half_center", "center of the lower image-right half of the garment", (120, 180, 255)),
)

# Semantic mode intentionally asks for a small set of task-level garment
# anchors. These observations are never installed as Rxxx grasp references.
DEFAULT_SEMANTIC_ANCHORS: tuple[KeypointSpec, ...] = (
    KeypointSpec(
        "collar",
        "the visible collar, neckline, or neck opening; return no point when uncertain",
        (255, 170, 0),
    ),
    KeypointSpec(
        "left_sleeve_end",
        "the garment's visible left sleeve endpoint; return no point when side identity is uncertain",
        (180, 80, 255),
    ),
    KeypointSpec(
        "right_sleeve_end",
        "the garment's visible right sleeve endpoint; return no point when side identity is uncertain",
        (255, 80, 190),
    ),
    KeypointSpec(
        "left_hem_corner",
        "the visible left corner/end of the garment hem; return no point when uncertain",
        (80, 220, 220),
    ),
    KeypointSpec(
        "right_hem_corner",
        "the visible right corner/end of the garment hem; return no point when uncertain",
        (220, 220, 60),
    ),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _finite_confidence(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MolmoKeypointPipelineError(f"{context} confidence must be numeric")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise MolmoKeypointPipelineError(
            f"{context} confidence must be finite and within [0, 1]"
        )
    return confidence


def validate_confidence_threshold(value: float) -> float:
    """Validate a strict ``confidence > threshold`` cutoff."""

    threshold = _finite_confidence(value, context="keypoint")
    if threshold >= 1.0:
        raise MolmoKeypointPipelineError(
            "keypoint confidence threshold must be less than 1"
        )
    return threshold


def load_keypoint_specs(path: Path | None) -> tuple[KeypointSpec, ...]:
    if path is None:
        return DEFAULT_KEYPOINTS
    raw = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise MolmoKeypointPipelineError(
            "keypoint spec JSON must be a non-empty list"
        )
    specs: list[KeypointSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise MolmoKeypointPipelineError(
                f"keypoint spec {index} must be an object"
            )
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        color = item.get("color", [255, 255, 255])
        if not name or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise MolmoKeypointPipelineError(
                f"keypoint spec {index} has an invalid name"
            )
        if name in seen:
            raise MolmoKeypointPipelineError(f"duplicate keypoint name: {name}")
        if not description or len(description) > 300:
            raise MolmoKeypointPipelineError(
                f"keypoint {name} needs a description of at most 300 characters"
            )
        if (
            not isinstance(color, list)
            or len(color) != 3
            or any(
                isinstance(channel, bool)
                or not isinstance(channel, int)
                or not 0 <= channel <= 255
                for channel in color
            )
        ):
            raise MolmoKeypointPipelineError(
                f"keypoint {name} color must contain three integers in [0, 255]"
            )
        specs.append(KeypointSpec(name, description, tuple(color)))
        seen.add(name)
    if len(specs) > 20:
        raise MolmoKeypointPipelineError("at most 20 keypoints are allowed")
    return tuple(specs)


def _raw_image_for_camera(perception_dir: Path, camera: str) -> Path:
    matches = [
        path
        for path in perception_dir.glob(f"camera_*_{camera}.png")
        if (match := RAW_IMAGE_PATTERN.fullmatch(path.name))
        and match.group(1).upper() == camera
    ]
    if len(matches) != 1:
        raise MolmoKeypointPipelineError(
            f"expected exactly one raw Camera {camera} image in {perception_dir}, "
            f"found {[path.name for path in matches]}"
        )
    return matches[0].resolve()


def _validate_worker_payload(
    payload: Any,
    *,
    cameras: Sequence[str],
    specs: Sequence[KeypointSpec],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return validated records indexed by camera and keypoint name."""

    if not isinstance(payload, dict) or not isinstance(payload.get("views"), list):
        raise MolmoKeypointPipelineError("Molmo keypoint output needs a views list")
    expected_names = {spec.name for spec in specs}
    views: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_view in payload["views"]:
        if not isinstance(raw_view, dict):
            raise MolmoKeypointPipelineError("Molmo keypoint view must be an object")
        label = str(raw_view.get("label", "")).strip().upper()
        if label not in cameras or label in views:
            raise MolmoKeypointPipelineError(
                f"unexpected or duplicate Molmo keypoint camera: {label!r}"
            )
        image_size = raw_view.get("image_size")
        if (
            not isinstance(image_size, list)
            or len(image_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in image_size
            )
        ):
            raise MolmoKeypointPipelineError(
                f"Camera {label} image_size must contain positive integer width/height"
            )
        records = raw_view.get("records")
        if not isinstance(records, list):
            raise MolmoKeypointPipelineError(
                f"Camera {label} Molmo records must be a list"
            )
        indexed: dict[str, dict[str, Any]] = {}
        for raw_record in records:
            if not isinstance(raw_record, dict):
                raise MolmoKeypointPipelineError(
                    f"Camera {label} keypoint record must be an object"
                )
            name = str(raw_record.get("name", "")).strip()
            if name not in expected_names or name in indexed:
                raise MolmoKeypointPipelineError(
                    f"Camera {label} has unexpected or duplicate keypoint {name!r}"
                )
            status = str(raw_record.get("status", "")).strip()
            if status not in {"point_returned", "not_found", "ambiguous"}:
                raise MolmoKeypointPipelineError(
                    f"Camera {label}/{name} has invalid status {status!r}"
                )
            confidence = _finite_confidence(
                raw_record.get("confidence"), context=f"Camera {label}/{name}"
            )
            confidence_definition = str(
                raw_record.get("confidence_definition", "")
            )
            if confidence_definition != CONFIDENCE_DEFINITION:
                raise MolmoKeypointPipelineError(
                    f"Camera {label}/{name} has an unsupported confidence definition"
                )
            components = raw_record.get("point_token_probabilities")
            pixel = raw_record.get("pixel_xy")
            if status == "point_returned":
                if (
                    not isinstance(pixel, list)
                    or len(pixel) != 2
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        for value in pixel
                    )
                ):
                    raise MolmoKeypointPipelineError(
                        f"Camera {label}/{name} returned an invalid pixel"
                    )
                x_px, y_px = float(pixel[0]), float(pixel[1])
                width, height = image_size
                if not 0 <= x_px < width or not 0 <= y_px < height:
                    raise MolmoKeypointPipelineError(
                        f"Camera {label}/{name} pixel is outside {width}x{height}"
                    )
                if (
                    not isinstance(components, list)
                    or len(components) != 3
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or not 0.0 <= float(value) <= 1.0
                        for value in components
                    )
                ):
                    raise MolmoKeypointPipelineError(
                        f"Camera {label}/{name} needs exactly three bounded point-token probabilities"
                    )
                expected_confidence = math.exp(
                    sum(math.log(max(float(value), 1e-300)) for value in components)
                    / 3.0
                )
                if any(float(value) == 0.0 for value in components):
                    expected_confidence = 0.0
                if not math.isclose(
                    confidence, expected_confidence, rel_tol=1e-7, abs_tol=1e-9
                ):
                    raise MolmoKeypointPipelineError(
                        f"Camera {label}/{name} confidence does not match its three token probabilities"
                    )
            elif pixel is not None or confidence != 0.0 or components not in ([], None):
                raise MolmoKeypointPipelineError(
                    f"Camera {label}/{name} without one point must have pixel=null, confidence=0, and no point probabilities"
                )
            record = dict(raw_record)
            record["confidence"] = confidence
            record["image_size"] = image_size
            indexed[name] = record
        if set(indexed) != expected_names:
            missing = sorted(expected_names.difference(indexed))
            raise MolmoKeypointPipelineError(
                f"Camera {label} is missing keypoint records: {missing}"
            )
        views[label] = indexed
    missing_cameras = sorted(set(cameras).difference(views))
    if missing_cameras:
        raise MolmoKeypointPipelineError(
            f"Molmo keypoint output is missing cameras: {missing_cameras}"
        )
    return views


def _local_geometry(
    perception_dir: Path,
    camera: str,
    pixel_xy: Sequence[float],
    *,
    radius_px: int,
) -> dict[str, Any]:
    xyz_path = perception_dir / f"camera_{camera}_base_xyz_mm.npy"
    height_path = perception_dir / f"camera_{camera}_height_above_table_mm.npy"
    if not xyz_path.is_file() or not height_path.is_file():
        raise MolmoKeypointPipelineError(
            f"Camera {camera} is missing its calibrated XYZ or height map"
        )
    xyz = np.load(xyz_path, mmap_mode="r", allow_pickle=False)
    height_map = np.load(height_path, mmap_mode="r", allow_pickle=False)
    if xyz.ndim != 3 or xyz.shape[2] != 3 or height_map.shape != xyz.shape[:2]:
        raise MolmoKeypointPipelineError(
            f"Camera {camera} calibrated maps have incompatible shapes"
        )
    x_source, y_source = float(pixel_xy[0]), float(pixel_xy[1])
    x_px = min(xyz.shape[1] - 1, max(0, int(round(x_source))))
    y_px = min(xyz.shape[0] - 1, max(0, int(round(y_source))))
    x0, x1 = max(0, x_px - radius_px), min(xyz.shape[1], x_px + radius_px + 1)
    y0, y1 = max(0, y_px - radius_px), min(xyz.shape[0], y_px + radius_px + 1)
    local_xyz = np.asarray(xyz[y0:y1, x0:x1], dtype=np.float64).reshape(-1, 3)
    local_height = np.asarray(
        height_map[y0:y1, x0:x1], dtype=np.float64
    ).reshape(-1)
    valid = np.all(np.isfinite(local_xyz), axis=1) & np.isfinite(local_height)
    if not np.any(valid):
        raise MolmoKeypointPipelineError(
            f"Camera {camera} has no finite calibrated surface near ({x_source:.1f}, {y_source:.1f})"
        )
    points = local_xyz[valid]
    heights = local_height[valid]
    median_xyz = np.median(points, axis=0)
    height_mm = float(np.median(heights))
    p10 = np.percentile(points, 10, axis=0)
    p90 = np.percentile(points, 90, axis=0)
    return {
        "pixel_xy": [x_px, y_px],
        "source_pixel_xy": [x_source, y_source],
        "base_xyz_mm": [float(value) for value in median_xyz],
        "height_above_table_mm": height_mm,
        "table_z_mm": float(median_xyz[2] - height_mm),
        "local_radius_px": int(radius_px),
        "local_sample_count": int(np.count_nonzero(valid)),
        "local_base_xyz_p10_mm": [float(value) for value in p10],
        "local_base_xyz_p90_mm": [float(value) for value in p90],
        "local_base_z_spread_mm": float(p90[2] - p10[2]),
    }


def _draw_overlays(
    image_path: Path,
    camera: str,
    records: Sequence[dict[str, Any]],
    artifact_dir: Path,
) -> tuple[Path, Path]:
    accepted_image = Image.open(image_path).convert("RGB")
    diagnostic_image = accepted_image.copy()
    accepted_draw = ImageDraw.Draw(accepted_image)
    diagnostic_draw = ImageDraw.Draw(diagnostic_image)
    for record in records:
        pixel = record.get("source_pixel_xy")
        if pixel is None:
            continue
        x_px, y_px = float(pixel[0]), float(pixel[1])
        accepted = bool(record.get("accepted"))
        color = (30, 255, 80) if accepted else (255, 70, 50)
        radius = 8 if accepted else 6
        diagnostic_draw.ellipse(
            (x_px - radius, y_px - radius, x_px + radius, y_px + radius),
            outline=color,
            width=3,
        )
        diagnostic_draw.text(
            (x_px + 10, y_px - 9),
            f"{record['name']} {record['confidence']:.3f}",
            fill=color,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        if accepted:
            accepted_draw.ellipse(
                (x_px - radius, y_px - radius, x_px + radius, y_px + radius),
                fill=(30, 255, 80),
                outline=(0, 0, 0),
                width=2,
            )
            accepted_draw.text(
                (x_px + 10, y_px - 9),
                f"{record['reference_id']} {record['name']} c={record['confidence']:.3f}",
                fill=(255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
    accepted_path = artifact_dir / f"camera_{camera}_molmo_keypoint_references.png"
    diagnostic_path = artifact_dir / f"camera_{camera}_molmo_keypoint_candidates.png"
    accepted_image.save(accepted_path)
    diagnostic_image.save(diagnostic_path)
    return accepted_path, diagnostic_path


def _semantic_anchor_geometry(
    perception_dir: Path,
    camera: str,
    pixel_xy: Sequence[float],
    *,
    radius_px: int,
) -> dict[str, Any]:
    """Ground an anchor location while proving it lies on the final garment mask."""

    geometry = _local_geometry(
        perception_dir,
        camera,
        pixel_xy,
        radius_px=radius_px,
    )
    mask_path = perception_dir / f"camera_{camera}_garment_mask.npy"
    if not mask_path.is_file():
        raise MolmoKeypointPipelineError(
            f"Camera {camera} is missing its final garment mask"
        )
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    x_px, y_px = geometry["pixel_xy"]
    x0, x1 = max(0, x_px - radius_px), min(mask.shape[1], x_px + radius_px + 1)
    y0, y1 = max(0, y_px - radius_px), min(mask.shape[0], y_px + radius_px + 1)
    local_mask = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
    garment_fraction = float(local_mask.mean()) if local_mask.size else 0.0
    if not bool(mask[y_px, x_px]) and garment_fraction < 0.25:
        raise MolmoKeypointPipelineError(
            f"Camera {camera} semantic anchor is outside the final garment mask"
        )
    geometry["on_final_garment_mask"] = bool(mask[y_px, x_px])
    geometry["local_garment_mask_fraction"] = garment_fraction
    return geometry


def _draw_semantic_anchor_overlays(
    image_path: Path,
    camera: str,
    records: Sequence[dict[str, Any]],
    artifact_dir: Path,
) -> tuple[Path, Path]:
    accepted_image = Image.open(image_path).convert("RGB")
    diagnostic_image = accepted_image.copy()
    accepted_draw = ImageDraw.Draw(accepted_image)
    diagnostic_draw = ImageDraw.Draw(diagnostic_image)
    for record in records:
        pixel = record.get("source_pixel_xy")
        if pixel is None:
            continue
        x_px, y_px = float(pixel[0]), float(pixel[1])
        accepted = bool(record.get("accepted"))
        color = (255, 210, 20) if accepted else (255, 70, 50)
        diagnostic_draw.ellipse(
            (x_px - 7, y_px - 7, x_px + 7, y_px + 7),
            outline=color,
            width=3,
        )
        diagnostic_draw.text(
            (x_px + 9, y_px - 8),
            f"{record['name']} {record['confidence']:.3f}",
            fill=color,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        if accepted:
            accepted_draw.ellipse(
                (x_px - 8, y_px - 8, x_px + 8, y_px + 8),
                fill=(255, 210, 20),
                outline=(0, 0, 0),
                width=2,
            )
            accepted_draw.text(
                (x_px + 10, y_px - 9),
                f"{record['anchor_id']} {record['name']} c={record['confidence']:.3f}",
                fill=(255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
    accepted_path = artifact_dir / f"camera_{camera}_semantic_anchors.png"
    diagnostic_path = artifact_dir / f"camera_{camera}_semantic_anchor_diagnostics.png"
    accepted_image.save(accepted_path)
    diagnostic_image.save(diagnostic_path)
    return accepted_path, diagnostic_path


def build_semantic_anchor_manifest(
    worker_payload: Any,
    *,
    perception_dir: Path,
    artifact_dir: Path,
    image_paths: dict[str, Path],
    cameras: Sequence[str],
    specs: Sequence[KeypointSpec],
    confidence_threshold: float = DEFAULT_SEMANTIC_CONFIDENCE_THRESHOLD,
    local_radius_px: int = 3,
    max_anchors: int = 4,
    duplicate_radius_px: float = 10.0,
    cross_view_tolerance_mm: float = 80.0,
    install: bool = True,
) -> dict[str, Any]:
    """Return a small high-confidence Sxxx set, never an Rxxx grasp set."""

    threshold = validate_confidence_threshold(confidence_threshold)
    if not 1 <= max_anchors <= 8:
        raise MolmoKeypointPipelineError("max_anchors must be between 1 and 8")
    validated = _validate_worker_payload(worker_payload, cameras=cameras, specs=specs)
    by_camera: dict[str, list[dict[str, Any]]] = {}
    for camera in cameras:
        records: list[dict[str, Any]] = []
        for spec in specs:
            raw = validated[camera][spec.name]
            record: dict[str, Any] = {
                "camera": camera,
                "name": spec.name,
                "description": spec.description,
                "status": raw["status"],
                "confidence": float(raw["confidence"]),
                "confidence_threshold": threshold,
                "confidence_definition": CONFIDENCE_DEFINITION,
                "source_pixel_xy": raw.get("pixel_xy"),
                "accepted": False,
                "rejection_reason": None,
                "point_token_probabilities": raw.get(
                    "point_token_probabilities", []
                ),
            }
            if raw["status"] != "point_returned":
                record["rejection_reason"] = raw["status"]
            elif float(raw["confidence"]) <= threshold:
                record["rejection_reason"] = (
                    f"confidence_not_strictly_above_{threshold:.6f}"
                )
            else:
                try:
                    record.update(
                        _semantic_anchor_geometry(
                            perception_dir,
                            camera,
                            raw["pixel_xy"],
                            radius_px=local_radius_px,
                        )
                    )
                except MolmoKeypointPipelineError as exc:
                    record["rejection_reason"] = f"invalid_anchor_geometry: {exc}"
                else:
                    record["preliminarily_accepted"] = True
            records.append(record)

        # Molmo occasionally emits the same pixel for contradictory semantic
        # labels. Keep only the strongest observation in that local cluster.
        preliminary = sorted(
            (item for item in records if item.get("preliminarily_accepted")),
            key=lambda item: -float(item["confidence"]),
        )
        kept: list[dict[str, Any]] = []
        for item in preliminary:
            pixel = np.asarray(item["source_pixel_xy"], dtype=np.float64)
            duplicate = next(
                (
                    existing
                    for existing in kept
                    if float(
                        np.linalg.norm(
                            pixel
                            - np.asarray(
                                existing["source_pixel_xy"], dtype=np.float64
                            )
                        )
                    )
                    <= duplicate_radius_px
                ),
                None,
            )
            if duplicate is None:
                kept.append(item)
            else:
                item["preliminarily_accepted"] = False
                item["rejection_reason"] = (
                    "duplicate_semantic_location_with_"
                    f"{duplicate['name']}_confidence_{duplicate['confidence']:.3f}"
                )
        by_camera[camera] = records

    # Reconcile the same physical semantic type across A/B. If two confident
    # observations disagree in calibrated base space, neither is safe enough to
    # become a semantic state fact.
    canonical: list[dict[str, Any]] = []
    for spec in specs:
        observations = [
            item
            for camera in cameras
            for item in by_camera[camera]
            if item["name"] == spec.name and item.get("preliminarily_accepted")
        ]
        if not observations:
            continue
        observations.sort(key=lambda item: -float(item["confidence"]))
        if len(observations) > 1:
            base_points = [
                np.asarray(item["base_xyz_mm"][:2], dtype=np.float64)
                for item in observations
            ]
            disagreement = max(
                float(np.linalg.norm(first - second))
                for first in base_points
                for second in base_points
            )
            if disagreement > cross_view_tolerance_mm:
                for item in observations:
                    item["preliminarily_accepted"] = False
                    item["rejection_reason"] = (
                        f"cross_view_semantic_disagreement_{disagreement:.1f}mm"
                    )
                continue
        chosen = observations[0]
        chosen["corroborating_observations"] = [
            {
                "camera": item["camera"],
                "pixel_xy": item["pixel_xy"],
                "base_xyz_mm": item["base_xyz_mm"],
                "confidence": item["confidence"],
            }
            for item in observations[1:]
        ]
        for item in observations[1:]:
            item["preliminarily_accepted"] = False
            item["rejection_reason"] = (
                f"corroborates_canonical_camera_{chosen['camera']}"
            )
        canonical.append(chosen)
    canonical.sort(key=lambda item: (-float(item["confidence"]), item["name"]))
    for item in canonical[max_anchors:]:
        item["preliminarily_accepted"] = False
        item["rejection_reason"] = f"semantic_anchor_budget_max_{max_anchors}"
    canonical = canonical[:max_anchors]
    for index, item in enumerate(canonical, start=1):
        item["accepted"] = True
        item["anchor_id"] = f"S{index:03d}"

    views: list[dict[str, Any]] = []
    for camera in cameras:
        records = by_camera[camera]
        accepted_overlay, diagnostic_overlay = _draw_semantic_anchor_overlays(
            image_paths[camera], camera, records, artifact_dir
        )
        views.append(
            {
                "camera": camera,
                "input_image": str(image_paths[camera]),
                "accepted_overlay": str(accepted_overlay),
                "diagnostic_overlay": str(diagnostic_overlay),
                "query_count": len(records),
                "accepted_count": sum(bool(item["accepted"]) for item in records),
                "records": records,
            }
        )
    anchors = [
        {
            "anchor_id": item["anchor_id"],
            "type": item["name"],
            "description": item["description"],
            "camera": item["camera"],
            "pixel_xy": item["pixel_xy"],
            "source_pixel_xy": item["source_pixel_xy"],
            "base_xyz_mm": item["base_xyz_mm"],
            "height_above_table_mm": item["height_above_table_mm"],
            "local_base_z_spread_mm": item["local_base_z_spread_mm"],
            "confidence": item["confidence"],
            "confidence_definition": CONFIDENCE_DEFINITION,
            "corroborating_observations": item.get(
                "corroborating_observations", []
            ),
            "role": "semantic_anchor_not_grasp_point",
        }
        for item in canonical
    ]
    manifest = {
        "schema_version": 1,
        "created_at": _now(),
        "status": "READY" if anchors else "NO_HIGH_CONFIDENCE_SEMANTIC_ANCHORS",
        "confidence_threshold": threshold,
        "confidence_policy": "confidence > threshold",
        "confidence_definition": CONFIDENCE_DEFINITION,
        "max_anchors": max_anchors,
        "anchor_count": len(anchors),
        "anchors": anchors,
        "views": views,
        "semantic_contract": (
            "Sxxx observations define uncertain garment-part regions only. "
            "They must never be passed directly to grasp execution."
        ),
        "next_stage": "semantic_state_builder",
    }
    _write_json(artifact_dir / "molmo_semantic_anchors.json", manifest)
    if install:
        _write_json(perception_dir / "molmo_semantic_anchors.json", manifest)
        observation_path = perception_dir / "observation.json"
        if observation_path.is_file():
            observation = json.loads(observation_path.read_text(encoding="utf-8"))
            if not isinstance(observation, dict):
                raise MolmoKeypointPipelineError("observation.json must be an object")
            observation["semantic_anchor_manifest"] = "molmo_semantic_anchors.json"
            observation["semantic_anchor_count"] = len(anchors)
            observation["semantic_anchor_confidence_threshold"] = threshold
            observation["grasp_reference_policy"] = (
                "not_available_until_local_geometry_grounding"
            )
            _write_json(observation_path, observation)
    return manifest


def build_confidence_filtered_references(
    worker_payload: Any,
    *,
    perception_dir: Path,
    artifact_dir: Path,
    image_paths: dict[str, Path],
    cameras: Sequence[str],
    specs: Sequence[KeypointSpec],
    confidence_threshold: float,
    local_radius_px: int = 3,
    install: bool = True,
) -> dict[str, Any]:
    """Filter Molmo points and install accepted points as the only task Rxxx set."""

    threshold = validate_confidence_threshold(confidence_threshold)
    if not 0 <= local_radius_px <= 25:
        raise MolmoKeypointPipelineError("local_radius_px must be between 0 and 25")
    validated = _validate_worker_payload(
        worker_payload, cameras=cameras, specs=specs
    )
    output_views: list[dict[str, Any]] = []
    all_references: list[dict[str, Any]] = []
    for camera in cameras:
        candidates: list[dict[str, Any]] = []
        for spec in specs:
            raw = validated[camera][spec.name]
            candidate: dict[str, Any] = {
                "camera": camera,
                "name": spec.name,
                "description": spec.description,
                "status": raw["status"],
                "confidence": float(raw["confidence"]),
                "confidence_threshold": threshold,
                "confidence_policy": "strictly_greater_than",
                "confidence_definition": CONFIDENCE_DEFINITION,
                "source_pixel_xy": raw.get("pixel_xy"),
                "accepted": False,
                "rejection_reason": None,
                "generated_text": raw.get("generated_text", ""),
                "point_token_probabilities": raw.get(
                    "point_token_probabilities", []
                ),
                "termination_point_token_probability": raw.get(
                    "termination_point_token_probability"
                ),
            }
            if raw["status"] != "point_returned":
                candidate["rejection_reason"] = raw["status"]
            elif float(raw["confidence"]) <= threshold:
                candidate["rejection_reason"] = (
                    f"confidence_not_strictly_above_{threshold:.6f}"
                )
            else:
                try:
                    candidate.update(
                        _local_geometry(
                            perception_dir,
                            camera,
                            raw["pixel_xy"],
                            radius_px=local_radius_px,
                        )
                    )
                except MolmoKeypointPipelineError as exc:
                    candidate["rejection_reason"] = f"invalid_geometry: {exc}"
                else:
                    candidate["accepted"] = True
            candidates.append(candidate)

        accepted = [candidate for candidate in candidates if candidate["accepted"]]
        accepted.sort(key=lambda item: (-float(item["confidence"]), str(item["name"])))
        for index, candidate in enumerate(accepted, start=1):
            candidate["reference_id"] = f"R{index:03d}"
        accepted_by_name = {str(candidate["name"]): candidate for candidate in accepted}
        for candidate in candidates:
            if candidate["name"] in accepted_by_name:
                candidate.update(accepted_by_name[candidate["name"]])

        accepted_overlay, diagnostic_overlay = _draw_overlays(
            image_paths[camera], camera, candidates, artifact_dir
        )
        samples = [
            {
                key: candidate[key]
                for key in (
                    "reference_id",
                    "name",
                    "description",
                    "pixel_xy",
                    "source_pixel_xy",
                    "base_xyz_mm",
                    "table_z_mm",
                    "height_above_table_mm",
                    "confidence",
                    "confidence_threshold",
                    "confidence_definition",
                    "local_radius_px",
                    "local_sample_count",
                    "local_base_xyz_p10_mm",
                    "local_base_xyz_p90_mm",
                    "local_base_z_spread_mm",
                )
            }
            for candidate in accepted
        ]
        guide = {
            "camera_label": camera,
            "coordinate_frame": "robot_base_mm",
            "measurement_kind": "molmo_confidence_filtered_grasp_reference",
            "full_resolution_xyz_map": f"camera_{camera}_base_xyz_mm.npy",
            "overlay_image": accepted_overlay.name,
            "confidence_threshold": threshold,
            "confidence_policy": "confidence > threshold",
            "confidence_definition": CONFIDENCE_DEFINITION,
            "reference_semantics": (
                "Molmo keypoints whose point-token confidence is strictly above the "
                "configured threshold and whose local calibrated RGB-D surface is valid; "
                "these are the only grasp references for this task observation."
            ),
            "usage": (
                "Choose one visible Rxxx from the accepted Molmo keypoint overlay. "
                "Do not select an unmarked pixel or a rejected/below-threshold keypoint."
            ),
            "warning": (
                "Molmo token confidence is not a calibrated grasp-success probability; "
                "workspace, collision, preflight, controller IK, and operator gates remain authoritative."
            ),
            "samples": samples,
        }
        guide_artifact = artifact_dir / f"camera_{camera}_molmo_coordinate_guide.json"
        _write_json(guide_artifact, guide)
        if install:
            canonical_guide = perception_dir / f"camera_{camera}_coordinate_guide.json"
            uniform_backup = perception_dir / f"camera_{camera}_uniform_coordinate_guide.json"
            if canonical_guide.is_file() and not uniform_backup.exists():
                shutil.copy2(canonical_guide, uniform_backup)
            canonical_overlay = perception_dir / accepted_overlay.name
            shutil.copy2(accepted_overlay, canonical_overlay)
            shutil.copy2(guide_artifact, canonical_guide)
        view_payload = {
            "camera": camera,
            "input_image": str(image_paths[camera]),
            "accepted_overlay": str(accepted_overlay),
            "diagnostic_overlay": str(diagnostic_overlay),
            "coordinate_guide": str(guide_artifact),
            "candidate_count": len(candidates),
            "accepted_count": len(accepted),
            "rejected_count": len(candidates) - len(accepted),
            "candidates": candidates,
            "references": samples,
        }
        output_views.append(view_payload)
        all_references.extend(
            [{"camera": camera, **sample} for sample in samples]
        )

    disabled_unqueried_cameras: list[str] = []
    if install:
        for camera in sorted({"A", "B"}.difference(cameras)):
            canonical_guide = perception_dir / f"camera_{camera}_coordinate_guide.json"
            if not canonical_guide.is_file():
                continue
            uniform_backup = perception_dir / f"camera_{camera}_uniform_coordinate_guide.json"
            if not uniform_backup.exists():
                shutil.copy2(canonical_guide, uniform_backup)
            _write_json(
                canonical_guide,
                {
                    "camera_label": camera,
                    "coordinate_frame": "robot_base_mm",
                    "measurement_kind": "molmo_confidence_filtered_grasp_reference",
                    "confidence_threshold": threshold,
                    "confidence_policy": "confidence > threshold",
                    "confidence_definition": CONFIDENCE_DEFINITION,
                    "reference_semantics": (
                        "No task grasp references: this camera was not queried by the "
                        "current Molmo keypoint pipeline."
                    ),
                    "warning": "This camera is disabled for task grasp-reference selection.",
                    "samples": [],
                },
            )
            disabled_unqueried_cameras.append(camera)

    manifest = {
        "schema_version": 1,
        "created_at": _now(),
        "status": "READY" if all_references else "NO_VALID_GRASP_REFERENCES",
        "confidence_threshold": threshold,
        "confidence_policy": "confidence > threshold",
        "confidence_definition": CONFIDENCE_DEFINITION,
        "local_radius_px": local_radius_px,
        "keypoint_count_per_camera": len(specs),
        "camera_count": len(cameras),
        "queried_cameras": list(cameras),
        "disabled_unqueried_cameras": disabled_unqueried_cameras,
        "accepted_reference_count": len(all_references),
        "references": all_references,
        "views": output_views,
        "safety_gate": (
            "planning must not start when status is NO_VALID_GRASP_REFERENCES"
        ),
    }
    _write_json(artifact_dir / "molmo_keypoint_grasp_references.json", manifest)
    if install:
        _write_json(
            perception_dir / "molmo_keypoint_grasp_references.json", manifest
        )
        observation_path = perception_dir / "observation.json"
        if observation_path.is_file():
            observation = json.loads(observation_path.read_text(encoding="utf-8"))
            if not isinstance(observation, dict):
                raise MolmoKeypointPipelineError("observation.json must be an object")
            observation["grasp_reference_policy"] = (
                "molmo_confidence_filtered_keypoints_only"
            )
            observation["molmo_keypoint_confidence_threshold"] = threshold
            observation["molmo_keypoint_confidence_definition"] = CONFIDENCE_DEFINITION
            observation["molmo_keypoint_reference_manifest"] = (
                "molmo_keypoint_grasp_references.json"
            )
            observation["valid_grasp_reference_count"] = len(all_references)
            _write_json(observation_path, observation)
    return manifest


def resolve_molmo_python(value: Path | None = None) -> Path:
    candidates = []
    if value is not None:
        candidates.append(Path(value))
    if os.environ.get("MOLMO_PYTHON"):
        candidates.append(Path(os.environ["MOLMO_PYTHON"]))
    candidates.append(Path.home() / "miniconda3" / "envs" / "molmo" / "bin" / "python")
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    raise MolmoKeypointPipelineError(
        "Molmo Python was not found; pass --molmo-python or set MOLMO_PYTHON"
    )


def _run_worker_streaming(
    command: list[str],
    *,
    cwd: Path,
    timeout_s: int,
    line_callback: Callable[[str], None],
) -> subprocess.CompletedProcess[str]:
    """Run the GPU worker while teeing merged stdout/stderr to a callback."""

    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        bufsize=1,
    )
    chunks: list[str] = []

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            chunks.append(line)
            line_callback(line)

    reader = threading.Thread(
        target=read_output,
        daemon=True,
        name="molmo-keypoint-worker-output",
    )
    reader.start()
    try:
        returncode = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
        reader.join(timeout=10)
        raise
    reader.join(timeout=10)
    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout="".join(chunks),
        stderr="",
    )


def run_molmo_keypoint_pipeline(
    *,
    project_root: Path,
    perception_dir: Path,
    artifact_dir: Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    molmo_python: Path | None = None,
    model: str = "allenai/MolmoPoint-8B",
    dtype: str = "bf16",
    max_crops: int = 1,
    max_new_tokens: int = 96,
    timeout_s: int = 900,
    local_files_only: bool = True,
    keypoint_specs: Sequence[KeypointSpec] = DEFAULT_KEYPOINTS,
    cameras: Sequence[str] = ("A", "B"),
    local_radius_px: int = 3,
    install: bool = True,
    semantic_anchors: bool = False,
    max_semantic_anchors: int = 4,
    subprocess_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    worker_line_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run Molmo once and build legacy Rxxx references or semantic Sxxx anchors."""

    root = Path(project_root).expanduser().resolve()
    perception = Path(perception_dir).expanduser().resolve()
    output = Path(artifact_dir).expanduser().resolve()
    if not perception.is_dir():
        raise MolmoKeypointPipelineError(
            f"perception directory does not exist: {perception}"
        )
    if output.exists():
        raise MolmoKeypointPipelineError(f"artifact directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    threshold = validate_confidence_threshold(confidence_threshold)
    normalized_cameras = tuple(str(camera).strip().upper() for camera in cameras)
    if not normalized_cameras or len(set(normalized_cameras)) != len(normalized_cameras):
        raise MolmoKeypointPipelineError("camera labels must be non-empty and unique")
    if any(camera not in {"A", "B"} for camera in normalized_cameras):
        raise MolmoKeypointPipelineError("keypoint cameras must be A and/or B")
    specs = tuple(keypoint_specs)
    if not specs:
        raise MolmoKeypointPipelineError("at least one keypoint spec is required")
    image_paths = {
        camera: _raw_image_for_camera(perception, camera)
        for camera in normalized_cameras
    }
    specs_path = output / "keypoint_specs.json"
    _write_json(specs_path, [spec.as_dict() for spec in specs])
    raw_output = output / "molmo_keypoints_raw.json"
    worker = root / "cloth_agent" / "molmo_keypoint_worker.py"
    if not worker.is_file():
        raise MolmoKeypointPipelineError(f"Molmo keypoint worker is missing: {worker}")
    command = [
        str(resolve_molmo_python(molmo_python)),
        str(worker),
        "--output",
        str(raw_output),
        "--specs",
        str(specs_path),
        "--model",
        model,
        "--dtype",
        dtype,
        "--max-crops",
        str(max_crops),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    for camera in normalized_cameras:
        command.extend(["--image", str(image_paths[camera]), "--label", camera])
    if local_files_only:
        command.append("--local-files-only")
    try:
        if worker_line_callback is not None:
            completed = _run_worker_streaming(
                command,
                cwd=root,
                timeout_s=timeout_s,
                line_callback=worker_line_callback,
            )
        else:
            completed = subprocess_run(
                command,
                cwd=root,
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
                shell=False,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MolmoKeypointPipelineError(
            f"Molmo keypoint invocation failed: {exc}"
        ) from exc
    (output / "molmo_keypoints.stdout.txt").write_text(
        completed.stdout + ("\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise MolmoKeypointPipelineError(
            f"Molmo keypoint worker exited with {completed.returncode}; inspect "
            f"{output / 'molmo_keypoints.stdout.txt'}"
        )
    if not raw_output.is_file():
        raise MolmoKeypointPipelineError(
            "Molmo keypoint worker completed without its JSON output"
        )
    payload = json.loads(raw_output.read_text(encoding="utf-8"))
    if semantic_anchors:
        manifest = build_semantic_anchor_manifest(
            payload,
            perception_dir=perception,
            artifact_dir=output,
            image_paths=image_paths,
            cameras=normalized_cameras,
            specs=specs,
            confidence_threshold=threshold,
            local_radius_px=local_radius_px,
            max_anchors=max_semantic_anchors,
            install=install,
        )
    else:
        manifest = build_confidence_filtered_references(
            payload,
            perception_dir=perception,
            artifact_dir=output,
            image_paths=image_paths,
            cameras=normalized_cameras,
            specs=specs,
            confidence_threshold=threshold,
            local_radius_px=local_radius_px,
            install=install,
        )
    manifest["worker"] = {
        "command": command,
        "model": model,
        "dtype": dtype,
        "max_crops": max_crops,
        "max_new_tokens": max_new_tokens,
        "local_files_only": local_files_only,
    }
    manifest_name = (
        "molmo_semantic_anchors.json"
        if semantic_anchors
        else "molmo_keypoint_grasp_references.json"
    )
    _write_json(output / manifest_name, manifest)
    if install and not semantic_anchors:
        _write_json(
            perception / "molmo_keypoint_grasp_references.json", manifest
        )
    return manifest


def run_molmo_semantic_anchor_pipeline(
    *,
    project_root: Path,
    perception_dir: Path,
    artifact_dir: Path,
    confidence_threshold: float = DEFAULT_SEMANTIC_CONFIDENCE_THRESHOLD,
    molmo_python: Path | None = None,
    model: str = "allenai/MolmoPoint-8B",
    dtype: str = "bf16",
    max_crops: int = 1,
    max_new_tokens: int = 96,
    timeout_s: int = 900,
    local_files_only: bool = True,
    keypoint_specs: Sequence[KeypointSpec] = DEFAULT_SEMANTIC_ANCHORS,
    cameras: Sequence[str] = ("A", "B"),
    local_radius_px: int = 3,
    max_anchors: int = 4,
    install: bool = True,
    subprocess_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    worker_line_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run Molmo as a high-confidence semantic-anchor detector only."""

    return run_molmo_keypoint_pipeline(
        project_root=project_root,
        perception_dir=perception_dir,
        artifact_dir=artifact_dir,
        confidence_threshold=confidence_threshold,
        molmo_python=molmo_python,
        model=model,
        dtype=dtype,
        max_crops=max_crops,
        max_new_tokens=max_new_tokens,
        timeout_s=timeout_s,
        local_files_only=local_files_only,
        keypoint_specs=keypoint_specs,
        cameras=cameras,
        local_radius_px=local_radius_px,
        install=install,
        semantic_anchors=True,
        max_semantic_anchors=max_anchors,
        subprocess_run=subprocess_run,
        worker_line_callback=worker_line_callback,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--perception-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_SEMANTIC_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument("--keypoints-json", type=Path)
    parser.add_argument("--camera", action="append", choices=["A", "B"])
    parser.add_argument("--molmo-python", type=Path)
    parser.add_argument("--model", default="allenai/MolmoPoint-8B")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-crops", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--local-radius-px", type=int, default=3)
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="write artifacts without replacing workspace Rxxx guides",
    )
    parser.add_argument(
        "--legacy-grasp-references",
        action="store_true",
        help=(
            "use the retired direct Molmo→Rxxx behavior; default output is "
            "high-confidence semantic Sxxx anchors"
        ),
    )
    args = parser.parse_args(argv)
    specs = (
        load_keypoint_specs(args.keypoints_json)
        if args.keypoints_json
        else DEFAULT_KEYPOINTS
        if args.legacy_grasp_references
        else DEFAULT_SEMANTIC_ANCHORS
    )
    runner = (
        run_molmo_keypoint_pipeline
        if args.legacy_grasp_references
        else run_molmo_semantic_anchor_pipeline
    )
    manifest = runner(
        project_root=Path(args.project_root),
        perception_dir=args.perception_dir,
        artifact_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        molmo_python=args.molmo_python,
        model=args.model,
        dtype=args.dtype,
        max_crops=args.max_crops,
        max_new_tokens=args.max_new_tokens,
        timeout_s=args.timeout_s,
        local_files_only=not args.allow_download,
        keypoint_specs=specs,
        cameras=tuple(args.camera or ("A", "B")),
        local_radius_px=args.local_radius_px,
        install=not args.no_install,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "semantic_anchor_count": manifest.get("anchor_count"),
                "legacy_accepted_reference_count": manifest.get(
                    "accepted_reference_count"
                ),
                "confidence_threshold": manifest["confidence_threshold"],
                "output_dir": str(args.output_dir.expanduser().resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if manifest["status"] == "READY" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
