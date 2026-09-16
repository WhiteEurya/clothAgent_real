"""Claude-selected collar-up RGB passed to Molmo, with an audited pixel map."""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from .claude_image_debug import debug_directory
from .image_tools_mcp import pixel_hash
from .motion_image_sources import resolve_motion_sources
from .planner_backend import parse_claude_json


class MolmoOrientationError(ValueError):
    """A failed bounded orientation attempt must not restart with a fresh budget."""


ORIENTATION_EDIT_LIMIT = 6


_PIXEL = {"anyOf": [{"type": "null"}, {"type": "array", "minItems": 2,
    "maxItems": 2, "items": {"type": "number"}}]}
VIEW_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "status": {"enum": ["READY", "UNCERTAIN"]},
    "image_id": {"type": ["string", "null"]},
    "collar_pixel_xy": _PIXEL, "hem_pixel_xy": _PIXEL,
    "reason": {"type": "string"}},
    "required": ["status", "image_id", "collar_pixel_xy", "hem_pixel_xy", "reason"]}

VIEW_PROMPT = (
    "Prepare the CURRENT Camera-A RGB (image_0) for a downstream Molmo sleeve locator. "
    "Read the original. Identify the actual collar/neck opening and the opposite torso hem. "
    "Use rotate_image as needed to put the collar ABOVE the hem, with their centerline "
    "approximately vertical. Choose the rotation yourself from visible garment evidence; "
    "the camera's upright filename does NOT mean the garment is aligned. "
    "You may crop or resize for visibility, but keep the whole visible garment, both sleeve "
    "regions, collar and hem in the selected view. Never mirror. Read the final selected "
    "view to verify it before responding. If already aligned, you may select image_0 "
    "without a redundant transform. Return its exact image_id and collar/hem coordinates "
    "IN THAT VIEW, not coordinates mapped back to the original. "
    "In the selected collar-up, hem-down view, left_sleeve means the IMAGE LEFT sleeve "
    "and right_sleeve means the IMAGE RIGHT sleeve, never the wearer's anatomical side. "
    "Do not locate a sleeve or choose a grasp here. Molmo will next inspect the selected "
    "RGB, then Claude will judge its fallible annotation and decide the fold. "
    "If the collar/hem orientation is ambiguous or cannot be verified, return UNCERTAIN "
    "with null image_id and null coordinates. Return only the requested JSON."
    " You may try and refine, but have at most 6 edit attempts across rotate_image, "
    "crop_image and resize_image, including invalid attempts. Every response reports "
    "remaining edits. Stop as soon as the view is suitable; do not aim for perfection. "
    "At zero edits, do not request more edits: Read existing views, select and verify "
    "the best suitable one, or return UNCERTAIN. Do not restart to obtain more edits."
)


def map_molmo_pixel(selection, pixel):
    """Return floating and once-rounded pixels in the fixed Camera-A display frame."""
    payload = {"actions": [{"name": "move", "args": {
        "target": "pixel", "image_id": selection["image_id"], "pixel_xy": pixel}}]}
    _, trace = resolve_motion_sources(payload, [Path(selection["canonical_image"])],
                                      selection["image_sources"])
    return trace[0]


def prepare_molmo_view(backend, canonical_image: Path, output: Path, *, timeout_s: int):
    """Require a verified, actually Read view; never substitute a guessed orientation."""
    output.mkdir(parents=True, exist_ok=False)
    debug = debug_directory([canonical_image], output, "molmo_orientation")
    report = {"status": "RUNNING", "canonical_image": str(canonical_image),
              "edit_limit": ORIENTATION_EDIT_LIMIT, "automatic_retry_allowed": False,
              "image_debug_directory": str(debug)}
    report_path = output / "selection.json"
    try:
        result = backend.invoke(prompt=VIEW_PROMPT, image_paths=[canonical_image],
            schema=VIEW_SCHEMA, debug_dir=debug, timeout_s=timeout_s,
            image_edit_limit=ORIENTATION_EDIT_LIMIT,
            system_prompt="Inspect current RGB with Read and the image tools. Prepare a collar-up view for Molmo. No robot access.")
        payload = parse_claude_json(result.stdout)
        report.update(response=payload, timings=result.timings,
                      image_sources=list(result.image_sources))
        if (set(payload) != set(VIEW_SCHEMA["required"]) or
                not isinstance(payload.get("reason"), str)):
            raise ValueError("invalid Claude Molmo-view selection schema")
        if payload.get("status") != "READY":
            raise ValueError("Claude could not establish collar-up orientation: " + payload["reason"])
        selected_id = payload.get("image_id")
        if not isinstance(selected_id, str):
            raise ValueError("Claude must select an explicit image_id for Molmo")
        sources = {view["image_id"]: view for view in result.image_sources}
        selected = sources.get(selected_id, {})
        if selected.get("verification") != "VERIFIED" or selected.get("read_status") != "READ_COMPLETED":
            raise ValueError("Molmo input must be pixel-verified and Read by Claude")
        path = Path(selected["path"]).resolve(strict=True)
        if debug.resolve() not in path.parents:
            raise ValueError("selected Molmo image is outside this Claude invocation")
        with Image.open(path) as source:
            image = source.convert("RGB")
        if pixel_hash(image) != selected.get("rgb_sha256"):
            raise ValueError("selected Molmo image changed after verification")
        report["image_id"] = selected_id
        endpoints = []
        for key in ("collar_pixel_xy", "hem_pixel_xy"):
            point = payload[key]
            if (not isinstance(point, list) or len(point) != 2 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in point)):
                raise ValueError("READY requires finite collar and hem pixels")
            endpoints.append(point)
            report[key + "_mapping"] = map_molmo_pixel(report, point)
        collar, hem = endpoints
        dx, dy = hem[0] - collar[0], hem[1] - collar[1]
        # Check Claude's declared alignment, not an independent semantic detector.
        if dy < max(10., min(image.size) * .05) or abs(dx) > dy * .25:
            raise ValueError("selected Molmo view is not collar-up/hem-down within 14 degrees")
        input_dir = output / "molmo_input"
        input_dir.mkdir()
        image_path = input_dir / "camera_0_A.png"
        image.save(image_path)
        report.update(status="READY", selected_image=str(image_path),
                      selected_rgb_sha256=pixel_hash(image),
                      side_convention="COLLAR_UP_IMAGE_LEFT_RIGHT", reason=payload["reason"],
                      alignment_check="Claude-declared endpoints; not independent semantic verification")
        # An annotated copy for humans, never the image given to Molmo.
        draw = ImageDraw.Draw(image)
        draw.line([tuple(collar), tuple(hem)], fill="yellow", width=3)
        for label, point in (("COLLAR / TOP", collar), ("HEM / BOTTOM", hem)):
            draw.text(tuple(point), label, fill="yellow", stroke_width=1, stroke_fill="black")
        image.save(output / "claude_orientation_debug.png")
        return report
    except BaseException as exc:
        report.update(status="FAILED_NO_MOLMO", error=f"{type(exc).__name__}: {exc}")
        if isinstance(exc, Exception):
            raise MolmoOrientationError(f'Claude orientation failed; no automatic budget reset: {exc}') from exc
        raise
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
