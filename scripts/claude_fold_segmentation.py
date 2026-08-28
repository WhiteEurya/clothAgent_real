#!/usr/bin/env python3
"""Claude-assisted zero-shot visible garment-fold segmentation.

Claude returns only visible fold polylines/regions in Camera-A pixel space.  It
is never allowed to control the robot or edit the run.  The host validates the
geometry and renders the returned structure on top of the RGB image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw


class FoldSegmentationError(RuntimeError):
    pass


def _json_from_claude(text: str) -> dict[str, Any]:
    try:
        outer = json.loads(text)
        if isinstance(outer, dict) and isinstance(outer.get("structured_output"), dict):
            return outer["structured_output"]
        if isinstance(outer, dict):
            return outer
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            if isinstance(value.get("structured_output"), dict):
                return value["structured_output"]
            return value
    raise FoldSegmentationError("Claude response did not contain a JSON object")


def _point(value: Any, width: int, height: int, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 2:
        raise FoldSegmentationError(f"{label} must be [x,y]")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise FoldSegmentationError(f"{label} must contain integer pixels")
    x, y = int(value[0]), int(value[1])
    if not (0 <= x < width and 0 <= y < height):
        raise FoldSegmentationError(f"{label}={value} is outside {width}x{height}")
    return [x, y]


def validate_payload(payload: Any, *, width: int, height: int) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise FoldSegmentationError("fold segmentation payload must be an object")
    required = {"status", "outer_contour", "fold_boundaries", "regions", "notes"}
    if set(payload) != required:
        raise FoldSegmentationError(
            "fold segmentation payload must contain exactly " + ", ".join(sorted(required))
        )
    status = str(payload["status"]).upper()
    if status not in {"READY", "AMBIGUOUS", "NOT_FOUND"}:
        raise FoldSegmentationError("status must be READY, AMBIGUOUS, or NOT_FOUND")
    if not isinstance(payload["notes"], list) or not payload["notes"]:
        raise FoldSegmentationError("notes must be a non-empty string list")
    outer_raw = payload["outer_contour"]
    outer: list[list[int]] | None = None
    if outer_raw is not None:
        if not isinstance(outer_raw, list) or len(outer_raw) < 3:
            raise FoldSegmentationError("outer_contour must contain at least 3 points or null")
        outer = [_point(p, width, height, "outer_contour point") for p in outer_raw[:80]]

    boundaries: list[dict[str, Any]] = []
    if not isinstance(payload["fold_boundaries"], list):
        raise FoldSegmentationError("fold_boundaries must be a list")
    for index, item in enumerate(payload["fold_boundaries"][:30]):
        if not isinstance(item, Mapping):
            raise FoldSegmentationError(f"fold_boundaries[{index}] must be an object")
        name = str(item.get("name", f"fold_{index + 1}")).strip()
        points = item.get("points")
        if not isinstance(points, list) or len(points) < 2:
            raise FoldSegmentationError(f"fold_boundaries[{index}].points needs at least 2 points")
        boundaries.append(
            {
                "name": name,
                "points": [_point(p, width, height, f"fold_boundaries[{index}] point") for p in points[:40]],
                "confidence": float(item.get("confidence", 0.0)),
                "evidence": str(item.get("evidence", "")),
            }
        )

    regions: list[dict[str, Any]] = []
    if not isinstance(payload["regions"], list):
        raise FoldSegmentationError("regions must be a list")
    for index, item in enumerate(payload["regions"][:20]):
        if not isinstance(item, Mapping):
            raise FoldSegmentationError(f"regions[{index}] must be an object")
        polygon = item.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise FoldSegmentationError(f"regions[{index}].polygon needs at least 3 points")
        regions.append(
            {
                "name": str(item.get("name", f"region_{index + 1}")),
                "polygon": [_point(p, width, height, f"regions[{index}] point") for p in polygon[:80]],
                "confidence": float(item.get("confidence", 0.0)),
                "evidence": str(item.get("evidence", "")),
            }
        )
    return {
        "status": status,
        "outer_contour": outer,
        "fold_boundaries": boundaries,
        "regions": regions,
        "notes": [str(note) for note in payload["notes"][:12]],
    }


def _schema() -> dict[str, Any]:
    point = {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 2, "maxItems": 2}
    polyline = {"type": "array", "items": point, "minItems": 2, "maxItems": 40}
    polygon = {"type": "array", "items": point, "minItems": 3, "maxItems": 80}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["READY", "AMBIGUOUS", "NOT_FOUND"]},
            "outer_contour": {"anyOf": [{"type": "null"}, {"type": "array", "items": point, "minItems": 3, "maxItems": 80}]},
            "fold_boundaries": {"type": "array", "maxItems": 30, "items": {"type": "object", "additionalProperties": False, "properties": {"name": {"type": "string"}, "points": polyline, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "evidence": {"type": "string"}}, "required": ["name", "points", "confidence", "evidence"]}},
            "regions": {"type": "array", "maxItems": 20, "items": {"type": "object", "additionalProperties": False, "properties": {"name": {"type": "string"}, "polygon": polygon, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "evidence": {"type": "string"}}, "required": ["name", "polygon", "confidence", "evidence"]}},
            "notes": {"type": "array", "minItems": 1, "maxItems": 12, "items": {"type": "string"}},
        },
        "required": ["status", "outer_contour", "fold_boundaries", "regions", "notes"],
    }


def segment_with_claude(
    perception_dir: Path,
    *,
    output_dir: Path | None = None,
    binary: str = "claude",
    timeout_s: int = 900,
) -> dict[str, Any]:
    perception = perception_dir.resolve()
    rgb_path = next((p for p in (perception / "camera_0_A.png", perception / "camera_A.png") if p.is_file()), None)
    mask_path = perception / "camera_A_garment_mask.npy"
    height_path = perception / "camera_A_height_above_table_mm.npy"
    gradient_path = perception / "camera_A_height_gradient_edges.png"
    if rgb_path is None or not mask_path.is_file() or not height_path.is_file():
        raise FoldSegmentationError("perception directory lacks Camera A RGB, mask, or height map")
    print(f"[fold-seg] loading Camera A evidence from {perception}", flush=True)
    rgb = Image.open(rgb_path).convert("RGB")
    width, height = rgb.size
    root = perception.parent.parent.resolve()
    executable = shutil.which(binary) if Path(binary).name == binary else binary
    if executable is None:
        raise FoldSegmentationError(f"Claude CLI not found: {binary}")
    reference_dir = root.parent.parent / "data" / "reference" / "flat_garment_reference"
    images = [rgb_path, mask_path, height_path]
    if gradient_path.is_file():
        images.append(gradient_path)
    for candidate in (
        reference_dir / "camera_A_flat_reference.png",
        reference_dir / "camera_A_flat_reference_anchors.png",
    ):
        if candidate.is_file():
            images.append(candidate)
    prompt = (
        "Create a visible-fold structural segmentation for the current garment. Read the current "
        "Camera A RGB first, then inspect the saved garment mask, height map, and height-gradient "
        "image as needed. Use the flat reference only for topology; never transfer reference pixels. "
        "Return polylines for visible fold/occlusion boundaries and polygons for visible surface "
        "regions. Do not invent hidden seams. Do not trace every texture wrinkle or isolated depth "
        "speckle. Prefer long, continuous boundaries supported by RGB/depth structure. The outer "
        "garment contour is allowed, but internal fold boundaries are the main target. Coordinates "
        f"must refer to the {width}x{height} Camera A RGB image. If the evidence is insufficient, "
        "return AMBIGUOUS or NOT_FOUND and leave uncertain boundaries out. Return JSON only.\n\n"
        "Files:\n" + "\n".join(f"- {path}" for path in images)
    )
    command = [
        str(executable), "--print", prompt,
        "--output-format", "json",
        "--json-schema", json.dumps(_schema(), separators=(",", ":")),
        "--permission-mode", "dontAsk",
        "--allowedTools", "Read",
        "--tools", "Read",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--add-dir", str(root),
        "--system-prompt",
        "You are a read-only garment-structure annotator. Draw only visible RGB-D-supported fold "
        "boundaries and visible surface regions. Never edit files, run commands, or control a robot. "
        "A hidden seam must be omitted or marked uncertain in notes.",
    ]
    log_dir = (output_dir or perception / "claude_fold_segmentation").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / "claude_stdout.json.txt"
    stderr_log = log_dir / "claude_stderr.txt"
    print(
        f"[fold-seg] starting Claude fold annotation; timeout={int(timeout_s)}s",
        flush=True,
    )
    started = time.monotonic()
    try:
        with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=root,
                text=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                shell=False,
            )
            while process.poll() is None:
                elapsed = int(time.monotonic() - started)
                if elapsed >= int(timeout_s):
                    process.kill()
                    process.wait()
                    raise FoldSegmentationError(
                        f"Claude fold segmentation timed out after {int(timeout_s)} seconds; "
                        f"logs: {stdout_log}, {stderr_log}"
                    )
                print(f"[fold-seg] Claude still running ({elapsed}s)", flush=True)
                time.sleep(10.0)
            returncode = int(process.returncode)
        stdout_text = stdout_log.read_text(encoding="utf-8")
        stderr_text = stderr_log.read_text(encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        raise FoldSegmentationError(f"Claude fold segmentation timed out after {timeout_s} seconds") from exc
    if returncode != 0:
        raise FoldSegmentationError(stderr_text.strip() or stdout_text.strip() or f"Claude exited {returncode}; logs: {stdout_log}, {stderr_log}")
    print(f"[fold-seg] Claude finished in {int(time.monotonic() - started)}s; validating annotation", flush=True)
    payload = validate_payload(_json_from_claude(stdout_text), width=width, height=height)
    out = log_dir
    out.mkdir(parents=True, exist_ok=True)
    overlay = rgb.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    if payload["regions"]:
        for index, region in enumerate(payload["regions"]):
            color = ((70 + index * 37) % 255, (140 + index * 71) % 255, (220 + index * 29) % 255, 70)
            draw.polygon([tuple(p) for p in region["polygon"]], fill=color)
    if payload["outer_contour"]:
        draw.line([tuple(p) for p in payload["outer_contour"] + [payload["outer_contour"][0]]], fill=(255, 230, 20, 255), width=4, joint="curve")
    for boundary in payload["fold_boundaries"]:
        draw.line([tuple(p) for p in boundary["points"]], fill=(255, 215, 0, 255), width=4, joint="curve")
    overlay_path = out / "camera_A_claude_fold_segmentation.png"
    overlay.save(overlay_path)
    result = {**payload, "overlay": str(overlay_path), "prompt": prompt, "stdout_log": str(stdout_log), "stderr_log": str(stderr_log)}
    (out / "camera_A_claude_fold_segmentation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--claude-timeout-s", type=int, default=900)
    args = parser.parse_args(argv)
    result = segment_with_claude(args.perception_dir, output_dir=args.output_dir, binary=args.claude_binary, timeout_s=args.claude_timeout_s)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
