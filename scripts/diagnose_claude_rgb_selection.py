#!/usr/bin/env python3
"""Attribute RGB garment point-selection failures to Claude or candidate generation.

This experiment is deliberately separated from robot execution and 3-D grounding.
Claude receives only an upright Camera-A RGB image and a marker overlay.  It never
receives depth, XYZ, robot bounds, a point cloud, or a garment mask.

The workflow has two stages:

1. ``--prepare-only`` creates a stable full-image Pxxx grid and an annotation
   template.  A human fills the acceptable Pxxx IDs for semantic localization
   and point-to-point folding tasks.
2. A scored run compares three candidate conditions with independently shuffled
   Rxxx labels on every call:

   A: uniform points over the complete RGB image;
   B: human-approved garment points;
   C: points generated from the production garment mask.

Only the marker identities are model outputs.  The runtime owns the marker-to-
pixel mapping and rejects unknown IDs before scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.free_exploration import _json_from_claude_text  # noqa: E402


CONDITION_NAMES = {
    "A": "FULL_IMAGE_UNIFORM_POINTS",
    "B": "HUMAN_APPROVED_GARMENT_POINTS",
    "C": "PRODUCTION_MASK_POINTS",
}

TASK_COLORS = [
    (230, 45, 45),
    (40, 130, 255),
    (40, 180, 80),
    (230, 135, 25),
    (170, 70, 220),
    (220, 45, 150),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _latest_perception_result(root: Path) -> Path:
    candidates: list[Path] = []
    for path in (root / "runs").glob("*/results/perception/*/result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            view = next(
                item
                for item in payload.get("views", [])
                if str(item.get("label", "")).upper() == "A"
            )
            image_path = path.parent / str(view["image"])
        except (OSError, ValueError, KeyError, StopIteration, TypeError):
            continue
        if image_path.is_file():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("no saved Camera-A perception result found")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns).resolve()


def _paths_from_perception_result(result_path: Path) -> tuple[Path, Path | None]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    view = next(
        item
        for item in payload.get("views", [])
        if str(item.get("label", "")).upper() == "A"
    )
    image_path = (result_path.parent / str(view["image"])).resolve()
    mask_name = view.get("garment_mask")
    mask_path = (result_path.parent / str(mask_name)).resolve() if mask_name else None
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if mask_path is not None and not mask_path.is_file():
        mask_path = None
    return image_path, mask_path


def _rotate_image(image: Image.Image, rotation: str) -> Image.Image:
    if rotation == "none":
        return image.convert("RGB")
    if rotation == "clockwise90":
        return image.convert("RGB").rotate(-90, expand=True)
    if rotation == "counterclockwise90":
        return image.convert("RGB").rotate(90, expand=True)
    if rotation == "180":
        return image.convert("RGB").rotate(180, expand=True)
    raise ValueError(f"unsupported rotation {rotation!r}")


def _rotate_mask(mask: np.ndarray, rotation: str) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if rotation == "none":
        return array
    if rotation == "clockwise90":
        return np.rot90(array, k=3)
    if rotation == "counterclockwise90":
        return np.rot90(array, k=1)
    if rotation == "180":
        return np.rot90(array, k=2)
    raise ValueError(f"unsupported rotation {rotation!r}")


def _image_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _uniform_grid(width: int, height: int, stride_px: int, margin_px: int) -> list[dict[str, Any]]:
    if stride_px < 16:
        raise ValueError("stride_px must be at least 16")
    if margin_px < 0 or margin_px * 2 >= min(width, height):
        raise ValueError("margin_px is invalid for the image size")
    points: list[dict[str, Any]] = []
    row = 0
    for y_px in range(margin_px, height - margin_px, stride_px):
        x_offset = stride_px // 2 if row % 2 else 0
        for x_px in range(margin_px + x_offset, width - margin_px, stride_px):
            points.append(
                {
                    "stable_id": f"P{len(points) + 1:03d}",
                    "pixel_xy": [int(x_px), int(y_px)],
                }
            )
        row += 1
    if len(points) < 2:
        raise ValueError("uniform grid produced fewer than two points")
    return points


def _production_mask_points(mask: np.ndarray, stride_px: int) -> list[dict[str, Any]]:
    """Use the production mask to choose one cloth pixel per image grid cell."""

    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    points: list[dict[str, Any]] = []
    for y0 in range(0, height, stride_px):
        for x0 in range(0, width, stride_px):
            y1 = min(height, y0 + stride_px)
            x1 = min(width, x0 + stride_px)
            local_y, local_x = np.nonzero(mask[y0:y1, x0:x1])
            if len(local_x) == 0:
                continue
            center_x = (x1 - x0 - 1) / 2.0
            center_y = (y1 - y0 - 1) / 2.0
            nearest = int(np.argmin((local_x - center_x) ** 2 + (local_y - center_y) ** 2))
            points.append(
                {
                    "stable_id": f"C{len(points) + 1:03d}",
                    "pixel_xy": [int(x0 + local_x[nearest]), int(y0 + local_y[nearest])],
                }
            )
    if len(points) < 2:
        raise ValueError("production mask produced fewer than two candidate points")
    return points


def _render_points(
    image: Image.Image,
    points: Iterable[dict[str, Any]],
    output: Path,
    *,
    id_key: str = "stable_id",
    title: str | None = None,
) -> Path:
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    for point in points:
        x_px, y_px = point["pixel_xy"]
        label = str(point[id_key])
        draw.ellipse(
            (x_px - 5, y_px - 5, x_px + 5, y_px + 5),
            fill=(0, 225, 225),
            outline=(0, 0, 0),
            width=2,
        )
        draw.text(
            (x_px + 7, y_px - 8),
            label,
            fill=(0, 0, 0),
            font=font,
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
    if title:
        draw.rectangle((8, 8, min(rendered.width - 8, 520), 36), fill=(255, 255, 255))
        draw.text((14, 14), title, fill=(0, 0, 0), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output)
    return output


def _annotation_template(
    *,
    source_image: Path,
    production_mask: Path | None,
    upright: Image.Image,
    rotation: str,
    stride_px: int,
    margin_px: int,
    all_points: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": _now(),
        "source_camera_a_image": str(source_image),
        "production_mask_path": str(production_mask) if production_mask else None,
        "rotation": rotation,
        "upright_image_size": [upright.width, upright.height],
        "upright_image_sha256": _image_sha256(upright),
        "stride_px": stride_px,
        "margin_px": margin_px,
        "acceptance_radius_px": max(24, int(round(stride_px * 0.65))),
        "instructions": [
            "Inspect full_image_stable_Pxxx_overlay.png.",
            "Fill human_candidate_point_ids with Pxxx points that are visibly on useful garment fabric.",
            "Consecutive IDs may be abbreviated as a range such as P045-P051 inside any point-ID list.",
            "For each enabled localization task, list every Pxxx point that should count as a correct answer.",
            "For each enabled planning task, define one or more acceptable source/destination Pxxx sets.",
            "Do not use depth, XYZ, robot reachability, or the production mask while making these labels.",
        ],
        "human_candidate_point_ids": [],
        "localization_tasks": [
            {
                "id": "image_left_sleeve_cuff_distal",
                "enabled": False,
                "instruction": "请选择图像左侧短袖袖口最外端附近、位于布料上的一个点。",
                "accepted_point_ids": [],
            },
            {
                "id": "image_right_sleeve_cuff_distal",
                "enabled": False,
                "instruction": "请选择图像右侧短袖袖口最外端附近、位于布料上的一个点。",
                "accepted_point_ids": [],
            },
            {
                "id": "image_left_sleeve_root",
                "enabled": False,
                "instruction": "请选择图像左侧袖子与衣身连接的袖根位置。",
                "accepted_point_ids": [],
            },
            {
                "id": "image_right_sleeve_root",
                "enabled": False,
                "instruction": "请选择图像右侧袖子与衣身连接的袖根位置。",
                "accepted_point_ids": [],
            },
            {
                "id": "collar",
                "enabled": False,
                "instruction": "请选择衣领或领口布料所在位置，不要选择标签。",
                "accepted_point_ids": [],
            },
            {
                "id": "bottom_hem",
                "enabled": False,
                "instruction": "请选择衣服下摆自由边缘附近、位于布料上的一个点。",
                "accepted_point_ids": [],
            },
        ],
        "planning_tasks": [
            {
                "id": "next_fold_point_transfer",
                "enabled": False,
                "instruction": "为了把这件衣服叠得规整，下一步应该把哪个布料点移动到哪个可见目标点？",
                "acceptable_transfers": [
                    {
                        "source_point_ids": [],
                        "destination_point_ids": [],
                        "note": "Describe one acceptable next fold here.",
                    }
                ],
            }
        ],
        "stable_points": all_points,
    }


def _point_table(points: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(point["stable_id"]): point for point in points}


def _expand_point_id_tokens(raw_ids: list[Any]) -> list[str]:
    expanded: list[str] = []
    for raw in raw_ids:
        token = str(raw).strip().upper()
        match = re.fullmatch(r"P([0-9]+)-P?([0-9]+)", token)
        if match is None:
            expanded.append(token)
            continue
        start = int(match.group(1))
        end = int(match.group(2))
        if end < start or end - start > 999:
            raise ValueError(f"invalid or excessive Pxxx range {token!r}")
        width = max(3, len(match.group(1)), len(match.group(2)))
        expanded.extend(f"P{index:0{width}d}" for index in range(start, end + 1))
    return expanded


def _validate_annotations(
    annotations: dict[str, Any],
    *,
    upright: Image.Image,
    all_points: list[dict[str, Any]],
    conditions: list[str],
    task_filter: set[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if int(annotations.get("schema_version", -1)) != 1:
        raise ValueError("annotations schema_version must be 1")
    expected_hash = str(annotations.get("upright_image_sha256", ""))
    actual_hash = _image_sha256(upright)
    if expected_hash != actual_hash:
        raise ValueError(
            "annotation image hash does not match the current upright RGB; "
            "prepare a new template for this exact image"
        )
    by_id = _point_table(all_points)

    def validate_ids(raw_ids: Any, field: str, *, nonempty: bool) -> list[str]:
        if not isinstance(raw_ids, list):
            raise ValueError(f"{field} must be a JSON list")
        ids = _expand_point_id_tokens(raw_ids)
        unknown = sorted(set(ids) - set(by_id))
        if unknown:
            raise ValueError(f"{field} contains unknown Pxxx IDs: {unknown}")
        if nonempty and not ids:
            raise ValueError(f"{field} must contain at least one Pxxx ID")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{field} contains duplicate Pxxx IDs")
        return ids

    human_ids = validate_ids(
        annotations.get("human_candidate_point_ids", []),
        "human_candidate_point_ids",
        nonempty="B" in conditions,
    )
    human_points = [by_id[point_id] for point_id in human_ids]
    if "B" in conditions and len(human_points) < 6:
        raise ValueError(
            "condition B requires at least six human-approved garment candidates so the task "
            "contains meaningful visual distractors"
        )

    localization_tasks: list[dict[str, Any]] = []
    for raw in annotations.get("localization_tasks", []):
        if not isinstance(raw, dict) or not bool(raw.get("enabled")):
            continue
        task_id = str(raw.get("id", "")).strip()
        if not task_id or (task_filter is not None and task_id not in task_filter):
            continue
        instruction = str(raw.get("instruction", "")).strip()
        if not instruction:
            raise ValueError(f"localization task {task_id!r} has an empty instruction")
        accepted = validate_ids(
            raw.get("accepted_point_ids", []),
            f"localization task {task_id}.accepted_point_ids",
            nonempty=True,
        )
        if "B" in conditions and not set(accepted).intersection(human_ids):
            raise ValueError(
                f"localization task {task_id!r} has no accepted point in "
                "human_candidate_point_ids, so condition B cannot succeed"
            )
        localization_tasks.append(
            {
                "kind": "localization",
                "id": task_id,
                "instruction": instruction,
                "accepted_point_ids": accepted,
            }
        )

    planning_tasks: list[dict[str, Any]] = []
    for raw in annotations.get("planning_tasks", []):
        if not isinstance(raw, dict) or not bool(raw.get("enabled")):
            continue
        task_id = str(raw.get("id", "")).strip()
        if not task_id or (task_filter is not None and task_id not in task_filter):
            continue
        instruction = str(raw.get("instruction", "")).strip()
        if not instruction:
            raise ValueError(f"planning task {task_id!r} has an empty instruction")
        transfers: list[dict[str, Any]] = []
        for index, transfer in enumerate(raw.get("acceptable_transfers", []), start=1):
            if not isinstance(transfer, dict):
                raise ValueError(f"planning task {task_id!r} transfer {index} must be an object")
            source_ids = validate_ids(
                transfer.get("source_point_ids", []),
                f"planning task {task_id}.transfer[{index}].source_point_ids",
                nonempty=True,
            )
            destination_ids = validate_ids(
                transfer.get("destination_point_ids", []),
                f"planning task {task_id}.transfer[{index}].destination_point_ids",
                nonempty=True,
            )
            if "B" in conditions and (
                not set(source_ids).intersection(human_ids)
                or not set(destination_ids).intersection(human_ids)
            ):
                raise ValueError(
                    f"planning task {task_id!r} transfer {index} is unreachable in condition B; "
                    "add accepted source and destination points to human_candidate_point_ids"
                )
            transfers.append(
                {
                    "source_point_ids": source_ids,
                    "destination_point_ids": destination_ids,
                    "note": str(transfer.get("note", "")),
                }
            )
        if not transfers:
            raise ValueError(f"planning task {task_id!r} needs at least one acceptable transfer")
        planning_tasks.append(
            {
                "kind": "planning",
                "id": task_id,
                "instruction": instruction,
                "acceptable_transfers": transfers,
            }
        )

    if not localization_tasks and not planning_tasks:
        selected = "all enabled tasks" if task_filter is None else f"task filter {sorted(task_filter)}"
        raise ValueError(f"annotations contain no runnable task for {selected}")
    return human_points, localization_tasks, planning_tasks


def _randomized_markers(
    candidates: list[dict[str, Any]],
    *,
    seed: int,
    key: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    key_seed = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()[:8], "big")
    shuffled = list(candidates)
    random.Random(key_seed).shuffle(shuffled)
    markers: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(shuffled, start=1):
        reference_id = f"R{index:03d}"
        marker = {
            "reference_id": reference_id,
            "stable_id": str(candidate["stable_id"]),
            "pixel_xy": list(candidate["pixel_xy"]),
        }
        markers.append(marker)
        mapping[reference_id] = marker
    return markers, mapping


def _schema(task_kind: str) -> dict[str, Any]:
    if task_kind == "localization":
        properties = {
            "selected_reference_id": {"type": "string", "pattern": "^R[0-9]{3,}$"},
        }
        required = ["selected_reference_id"]
    elif task_kind == "planning":
        properties = {
            "source_reference_id": {"type": "string", "pattern": "^R[0-9]{3,}$"},
            "destination_reference_id": {"type": "string", "pattern": "^R[0-9]{3,}$"},
        }
        required = ["source_reference_id", "destination_reference_id"]
    else:
        raise ValueError(f"unknown task kind {task_kind!r}")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _build_prompt(task: dict[str, Any], marker_count: int) -> str:
    contract = (
        "Return exactly one field named selected_reference_id."
        if task["kind"] == "localization"
        else "Return exactly two fields named source_reference_id and destination_reference_id."
    )
    return (
        "RGB-ONLY GARMENT POINT-SELECTION DIAGNOSTIC. The first image is the raw upright "
        "Camera-A RGB observation. The second is the same RGB image with visible Rxxx candidate "
        f"markers. There are {marker_count} candidates. Use only visible RGB appearance and garment "
        "semantics. Do not infer or request depth, XYZ, calibration, robot reachability, force, or IK. "
        "Every returned marker must be visibly present in the overlay.\n\n"
        f"Task: {task['instruction']}\n\n{contract} Return only the requested JSON object."
    )


def _invoke_claude(
    *,
    binary: str,
    call_dir: Path,
    task: dict[str, Any],
    marker_count: int,
    timeout_s: int,
) -> tuple[dict[str, Any], str, str, float]:
    prompt = _build_prompt(task, marker_count)
    command = [
        binary,
        "--print",
        (
            f"{prompt}\n\nRGB files to inspect:\n"
            f"- {call_dir / 'camera_A_rgb_upright.png'}\n"
            f"- {call_dir / 'camera_A_Rxxx_overlay.png'}"
        ),
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(_schema(task["kind"]), separators=(",", ":")),
        "--permission-mode",
        "dontAsk",
        "--allowedTools",
        "Read",
        "--tools",
        "Read",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--add-dir",
        str(call_dir.resolve()),
        "--system-prompt",
        (
            "You are a read-only Camera-A RGB garment analyst. Inspect only the two supplied "
            "RGB PNG files. Never inspect other files, never use depth or robot geometry, and "
            "return only the schema fields. Marker IDs are randomized independently every trial."
        ),
    ]
    started = datetime.now(timezone.utc)
    completed = subprocess.run(
        command,
        cwd=call_dir,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    duration_s = (datetime.now(timezone.utc) - started).total_seconds()
    if completed.returncode != 0:
        raise RuntimeError(
            f"Claude exited with {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return _json_from_claude_text(completed.stdout), completed.stdout, completed.stderr, duration_s


def _resolve_model_output(
    task: dict[str, Any],
    payload: dict[str, Any],
    mapping: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if task["kind"] == "localization":
        reference_id = str(payload.get("selected_reference_id", "")).upper()
        if reference_id not in mapping:
            raise ValueError(f"Claude selected unknown marker {reference_id!r}")
        return {"selected": mapping[reference_id]}
    source_id = str(payload.get("source_reference_id", "")).upper()
    destination_id = str(payload.get("destination_reference_id", "")).upper()
    if source_id not in mapping:
        raise ValueError(f"Claude selected unknown source marker {source_id!r}")
    if destination_id not in mapping:
        raise ValueError(f"Claude selected unknown destination marker {destination_id!r}")
    return {"source": mapping[source_id], "destination": mapping[destination_id]}


def _distance(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay = a
    bx, by = b
    return math.hypot(float(ax) - float(bx), float(ay) - float(by))


def _nearest_distance(pixel_xy: list[int], stable_ids: Iterable[str], all_by_id: dict[str, dict[str, Any]]) -> float:
    distances = [_distance(pixel_xy, all_by_id[point_id]["pixel_xy"]) for point_id in stable_ids]
    return min(distances) if distances else math.inf


def _score_result(
    task: dict[str, Any],
    resolved: dict[str, Any],
    *,
    all_by_id: dict[str, dict[str, Any]],
    acceptance_radius_px: float,
) -> dict[str, Any]:
    if task["kind"] == "localization":
        distance_px = _nearest_distance(
            resolved["selected"]["pixel_xy"],
            task["accepted_point_ids"],
            all_by_id,
        )
        return {"hit": distance_px <= acceptance_radius_px, "nearest_accepted_distance_px": distance_px}

    transfer_scores: list[dict[str, Any]] = []
    for index, transfer in enumerate(task["acceptable_transfers"], start=1):
        source_distance = _nearest_distance(
            resolved["source"]["pixel_xy"], transfer["source_point_ids"], all_by_id
        )
        destination_distance = _nearest_distance(
            resolved["destination"]["pixel_xy"], transfer["destination_point_ids"], all_by_id
        )
        transfer_scores.append(
            {
                "transfer_index": index,
                "source_distance_px": source_distance,
                "destination_distance_px": destination_distance,
                "hit": source_distance <= acceptance_radius_px
                and destination_distance <= acceptance_radius_px,
            }
        )
    best = min(
        transfer_scores,
        key=lambda item: item["source_distance_px"] + item["destination_distance_px"],
    )
    return {"hit": any(item["hit"] for item in transfer_scores), "best_transfer": best, "transfers": transfer_scores}


def _pairwise_spread(pixels: list[list[int]]) -> dict[str, float | int | None]:
    if not pixels:
        return {"count": 0, "median_px": None, "max_px": None, "mean_px": None}
    if len(pixels) == 1:
        return {"count": 1, "median_px": 0.0, "max_px": 0.0, "mean_px": 0.0}
    distances = [
        _distance(pixels[left], pixels[right])
        for left in range(len(pixels))
        for right in range(left + 1, len(pixels))
    ]
    return {
        "count": len(pixels),
        "median_px": float(np.median(distances)),
        "max_px": float(np.max(distances)),
        "mean_px": float(np.mean(distances)),
    }


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((record["task_kind"], record["task_id"], record["condition"]), []).append(record)
    aggregates: list[dict[str, Any]] = []
    for (task_kind, task_id, condition), group in sorted(groups.items()):
        valid = [item for item in group if item["status"] == "COMPLETED"]
        hits = [bool(item["score"]["hit"]) for item in valid]
        aggregate: dict[str, Any] = {
            "task_kind": task_kind,
            "task_id": task_id,
            "condition": condition,
            "condition_name": CONDITION_NAMES[condition],
            "trial_count": len(group),
            "completed_count": len(valid),
            "error_count": len(group) - len(valid),
            "hit_count": sum(hits),
            "hit_rate": (sum(hits) / len(hits)) if hits else None,
        }
        if task_kind == "localization":
            pixels = [item["resolved"]["selected"]["pixel_xy"] for item in valid]
            aggregate["selection_spread"] = _pairwise_spread(pixels)
        else:
            source_pixels = [item["resolved"]["source"]["pixel_xy"] for item in valid]
            destination_pixels = [item["resolved"]["destination"]["pixel_xy"] for item in valid]
            aggregate["source_spread"] = _pairwise_spread(source_pixels)
            aggregate["destination_spread"] = _pairwise_spread(destination_pixels)
        aggregates.append(aggregate)
    return aggregates


def _diagnosis_hints(aggregates: list[dict[str, Any]]) -> list[str]:
    hints: list[str] = []
    by_task: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for item in aggregates:
        by_task.setdefault((item["task_kind"], item["task_id"]), {})[item["condition"]] = item
    for (kind, task_id), conditions in sorted(by_task.items()):
        rates = {
            name: conditions.get(name, {}).get("hit_rate")
            for name in ("A", "B", "C")
        }
        if rates["A"] is not None and rates["B"] is not None and rates["A"] < 0.5 and rates["B"] < 0.5:
            hints.append(
                f"{task_id}: A and B are both below 0.50; this supports a Claude RGB semantic/decision limitation more than a production-mask limitation."
            )
        if rates["B"] is not None and rates["C"] is not None and rates["B"] >= 0.75 and rates["C"] < 0.5:
            hints.append(
                f"{task_id}: B is strong while C is weak; inspect production mask/candidate coverage before blaming Claude."
            )
        if kind == "planning" and rates["A"] is not None and rates["A"] < 0.5:
            hints.append(
                f"{task_id}: planning accuracy is weak under the full-image RGB condition; compare localization tasks before deciding whether the failure is visual or strategic."
            )
    if not hints:
        hints.append(
            "No rule-based attribution threshold fired. Inspect per-task hit rates, selection spread, and overlays; collect more trials if completed_count is below 3."
        )
    return hints


def _render_results(upright: Image.Image, records: list[dict[str, Any]], output_dir: Path) -> list[str]:
    artifacts: list[str] = []
    keys = sorted({(item["task_kind"], item["task_id"], item["condition"]) for item in records})
    font = ImageFont.load_default()
    for task_kind, task_id, condition in keys:
        selected = [
            item
            for item in records
            if item["task_kind"] == task_kind
            and item["task_id"] == task_id
            and item["condition"] == condition
            and item["status"] == "COMPLETED"
        ]
        canvas = upright.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((8, 8, min(canvas.width - 8, 610), 38), fill=(255, 255, 255))
        draw.text(
            (14, 15),
            f"{task_id} | condition {condition} | {CONDITION_NAMES[condition]}",
            fill=(0, 0, 0),
            font=font,
        )
        for index, record in enumerate(selected):
            color = TASK_COLORS[index % len(TASK_COLORS)]
            trial = record["round"]
            if task_kind == "localization":
                x_px, y_px = record["resolved"]["selected"]["pixel_xy"]
                draw.ellipse((x_px - 16, y_px - 16, x_px + 16, y_px + 16), outline=color, width=6)
                draw.text((x_px + 18, y_px - 10), f"T{trial}", fill=color, font=font)
            else:
                sx, sy = record["resolved"]["source"]["pixel_xy"]
                dx, dy = record["resolved"]["destination"]["pixel_xy"]
                draw.line((sx, sy, dx, dy), fill=(255, 255, 255), width=9)
                draw.line((sx, sy, dx, dy), fill=color, width=5)
                draw.ellipse((sx - 14, sy - 14, sx + 14, sy + 14), outline=color, width=5)
                draw.rectangle((dx - 12, dy - 12, dx + 12, dy + 12), outline=color, width=5)
                draw.text((sx + 17, sy - 10), f"T{trial}", fill=color, font=font)
        path = output_dir / f"selections_{task_kind}_{task_id}_condition_{condition}.png"
        canvas.save(path)
        artifacts.append(str(path))
    return artifacts


def _resolve_inputs(
    args: argparse.Namespace,
    root: Path,
    annotations: dict[str, Any] | None,
) -> tuple[Path, Path | None, Path | None]:
    result_path = args.perception_result.resolve() if args.perception_result else None
    image_path = args.camera_a_image.resolve() if args.camera_a_image else None
    mask_path = args.production_mask.resolve() if args.production_mask else None
    if annotations is not None:
        if image_path is None and annotations.get("source_camera_a_image"):
            image_path = Path(str(annotations["source_camera_a_image"])).expanduser().resolve()
        if mask_path is None and annotations.get("production_mask_path"):
            mask_path = Path(str(annotations["production_mask_path"])).expanduser().resolve()
    if result_path is None and image_path is None:
        result_path = _latest_perception_result(root)
    if result_path is not None:
        result_image, result_mask = _paths_from_perception_result(result_path)
        if image_path is None:
            image_path = result_image
        if mask_path is None:
            mask_path = result_mask
    assert image_path is not None
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if mask_path is not None and not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    return image_path, mask_path, result_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--camera-a-image", type=Path, help="raw Camera-A RGB PNG")
    parser.add_argument("--perception-result", type=Path, help="saved result.json used only to locate RGB and optional production mask")
    parser.add_argument("--production-mask", type=Path, help="optional production garment_mask.npy for condition C")
    parser.add_argument("--run-dir", type=Path, help="new output run directory")
    parser.add_argument("--prepare-only", action="store_true", help="generate stable Pxxx overlay and annotation template without Claude")
    parser.add_argument("--annotations", type=Path, help="completed human annotation JSON from --prepare-only")
    parser.add_argument("--condition", action="append", choices=("A", "B", "C"), dest="conditions")
    parser.add_argument("--task", action="append", dest="tasks", help="run only this enabled task ID; repeatable")
    parser.add_argument("--rounds", type=int, default=3, help="independent Claude calls per task/condition (default: 3)")
    parser.add_argument("--stride-px", type=int, help="defaults to annotation value or 64")
    parser.add_argument("--margin-px", type=int, help="defaults to annotation value or 24")
    parser.add_argument("--rotation", choices=("none", "clockwise90", "counterclockwise90", "180"))
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--claude-timeout-s", type=int, default=900)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rounds < 1 or args.rounds > 20:
        raise SystemExit("--rounds must be between 1 and 20")
    if args.claude_timeout_s < 30:
        raise SystemExit("--claude-timeout-s must be at least 30")
    root = args.project_root.expanduser().resolve()
    annotations = None
    if args.annotations:
        annotations = json.loads(args.annotations.expanduser().resolve().read_text(encoding="utf-8"))
    rotation = args.rotation or (str(annotations.get("rotation")) if annotations else "clockwise90")
    stride_px = args.stride_px if args.stride_px is not None else int(
        annotations.get("stride_px", 64) if annotations else 64
    )
    margin_px = args.margin_px if args.margin_px is not None else int(
        annotations.get("margin_px", 24) if annotations else 24
    )
    if annotations is not None:
        if args.rotation is not None and args.rotation != str(annotations.get("rotation")):
            raise ValueError("--rotation must match the value stored in the human annotations")
        if args.stride_px is not None and args.stride_px != int(annotations.get("stride_px", -1)):
            raise ValueError("--stride-px must match the value stored in the human annotations")
        if args.margin_px is not None and args.margin_px != int(annotations.get("margin_px", -1)):
            raise ValueError("--margin-px must match the value stored in the human annotations")
    source_image, production_mask_path, perception_result = _resolve_inputs(args, root, annotations)
    with Image.open(source_image) as image:
        upright = _rotate_image(image, rotation)
    all_points = _uniform_grid(upright.width, upright.height, stride_px, margin_px)

    production_points: list[dict[str, Any]] = []
    if production_mask_path is not None:
        mask = _rotate_mask(np.load(production_mask_path), rotation)
        if mask.shape != (upright.height, upright.width):
            raise ValueError(
                f"rotated production mask shape {mask.shape} does not match upright RGB "
                f"{(upright.height, upright.width)}"
            )
        production_points = _production_mask_points(mask, stride_px)

    run_dir = (
        args.run_dir.expanduser().resolve()
        if args.run_dir
        else root / "runs" / f"claude_rgb_diagnosis_{_stamp()}"
    )
    if run_dir.exists():
        raise FileExistsError(run_dir)
    output = run_dir / "results" / "claude_rgb_selection"
    output.mkdir(parents=True, exist_ok=False)
    upright_path = output / "camera_A_rgb_upright.png"
    upright.save(upright_path)
    stable_overlay = _render_points(
        upright,
        all_points,
        output / "full_image_stable_Pxxx_overlay.png",
        title="Human annotation grid: stable Pxxx IDs over the complete RGB image",
    )
    production_overlay = None
    if production_points:
        production_overlay = _render_points(
            upright,
            production_points,
            output / "production_mask_candidates_overlay.png",
            title="Condition C candidates from production mask (Claude never receives the mask)",
        )

    if args.prepare_only:
        template = _annotation_template(
            source_image=source_image,
            production_mask=production_mask_path,
            upright=upright,
            rotation=rotation,
            stride_px=stride_px,
            margin_px=margin_px,
            all_points=all_points,
        )
        annotation_path = output / "human_annotations.template.json"
        _write_json(annotation_path, template)
        summary = {
            "status": "PREPARED",
            "created_at": _now(),
            "physical_execution": False,
            "claude_called": False,
            "depth_passed_to_claude": False,
            "source_camera_a_image": str(source_image),
            "perception_result": str(perception_result) if perception_result else None,
            "upright_rgb": str(upright_path),
            "stable_overlay": str(stable_overlay),
            "production_overlay": str(production_overlay) if production_overlay else None,
            "annotation_template": str(annotation_path),
            "point_count_A": len(all_points),
            "point_count_C": len(production_points),
        }
        _write_json(output / "prepare_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0

    if annotations is None:
        raise SystemExit("a scored run requires --annotations; run --prepare-only first")
    conditions = args.conditions or ["A", "B", "C"]
    if "C" in conditions and not production_points:
        raise ValueError("condition C requires a production garment mask")
    task_filter = set(args.tasks) if args.tasks else None
    human_points, localization_tasks, planning_tasks = _validate_annotations(
        annotations,
        upright=upright,
        all_points=all_points,
        conditions=conditions,
        task_filter=task_filter,
    )
    tasks = [*localization_tasks, *planning_tasks]
    candidate_sets = {"A": all_points, "B": human_points, "C": production_points}
    acceptance_radius_px = float(annotations.get("acceptance_radius_px", max(24, stride_px * 0.65)))
    if not 1 <= acceptance_radius_px <= 200:
        raise ValueError("acceptance_radius_px must be between 1 and 200")
    all_by_id = _point_table(all_points)

    records: list[dict[str, Any]] = []
    total = len(tasks) * len(conditions) * args.rounds
    call_index = 0
    for task in tasks:
        for condition in conditions:
            candidates = candidate_sets[condition]
            if len(candidates) < 2:
                raise ValueError(f"condition {condition} has fewer than two candidates")
            for round_index in range(1, args.rounds + 1):
                call_index += 1
                key = f"{task['kind']}:{task['id']}:{condition}:{round_index}"
                markers, mapping = _randomized_markers(candidates, seed=args.seed, key=key)
                call_dir = output / "claude_inputs" / f"call_{call_index:03d}"
                call_dir.mkdir(parents=True, exist_ok=False)
                upright.save(call_dir / "camera_A_rgb_upright.png")
                _render_points(
                    upright,
                    markers,
                    call_dir / "camera_A_Rxxx_overlay.png",
                    id_key="reference_id",
                    title=f"RGB candidates | task={task['id']} | independent trial={round_index}",
                )
                print(
                    f"[claude-rgb] call {call_index}/{total}: task={task['id']} "
                    f"condition={condition} round={round_index} candidates={len(markers)}",
                    flush=True,
                )
                record: dict[str, Any] = {
                    "created_at": _now(),
                    "call_index": call_index,
                    "task_kind": task["kind"],
                    "task_id": task["id"],
                    "condition": condition,
                    "condition_name": CONDITION_NAMES[condition],
                    "round": round_index,
                    "candidate_count": len(markers),
                    "input_dir": str(call_dir),
                    "status": "FAILED",
                }
                try:
                    payload, stdout, stderr, duration_s = _invoke_claude(
                        binary=args.claude_binary,
                        call_dir=call_dir,
                        task=task,
                        marker_count=len(markers),
                        timeout_s=args.claude_timeout_s,
                    )
                    resolved = _resolve_model_output(task, payload, mapping)
                    score = _score_result(
                        task,
                        resolved,
                        all_by_id=all_by_id,
                        acceptance_radius_px=acceptance_radius_px,
                    )
                    record.update(
                        {
                            "status": "COMPLETED",
                            "duration_s": duration_s,
                            "model_payload": payload,
                            "resolved": resolved,
                            "score": score,
                            "raw_stdout": stdout,
                            "raw_stderr": stderr,
                        }
                    )
                    print(
                        f"[claude-rgb] call {call_index}: hit={score['hit']} resolved={resolved}",
                        flush=True,
                    )
                except subprocess.TimeoutExpired as exc:
                    record.update(
                        {
                            "status": "TIMEOUT",
                            "error": f"Claude timed out after {exc.timeout} seconds",
                        }
                    )
                    print(f"[claude-rgb] call {call_index}: TIMEOUT", flush=True)
                except BaseException as exc:
                    record.update(
                        {
                            "status": "FAILED",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(f"[claude-rgb] call {call_index}: FAILED {record['error']}", flush=True)
                records.append(record)
                _write_json(output / "calls" / f"call_{call_index:03d}.json", record)
                _write_json(output / "partial_results.json", records)

    aggregates = _aggregate(records)
    visualizations = _render_results(upright, records, output)
    report = {
        "status": "COMPLETED",
        "created_at": _now(),
        "mode": "CLAUDE_RGB_SELECTION_ATTRIBUTION",
        "physical_execution": False,
        "robot_used": False,
        "depth_passed_to_claude": False,
        "xyz_passed_to_claude": False,
        "robot_bounds_passed_to_claude": False,
        "source_camera_a_image": str(source_image),
        "upright_rgb": str(upright_path),
        "annotations": str(args.annotations.expanduser().resolve()),
        "conditions": conditions,
        "rounds_per_task_condition": args.rounds,
        "acceptance_radius_px": acceptance_radius_px,
        "call_count": len(records),
        "aggregates": aggregates,
        "diagnosis_hints": _diagnosis_hints(aggregates),
        "visualizations": visualizations,
        "records": records,
    }
    _write_json(output / "report.json", report)
    print(f"[claude-rgb] report={output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
