#!/usr/bin/env python3
"""RGB-only point-to-point folding planner preview.

This experiment deliberately stops before 3-D grounding and robot motion. It
segments the current Camera-A RGB image, sprinkles stable display-only Rxxx
markers over that RGB garment mask, and asks Claude to choose a source marker
and a destination marker for one useful folding move.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.free_exploration import _json_from_claude_text  # noqa: E402


COLORS = [
    (235, 45, 45),
    (35, 115, 255),
    (30, 185, 75),
    (225, 125, 20),
    (170, 65, 220),
    (220, 40, 150),
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _latest_result(root: Path) -> Path:
    paths = list((root / "runs").glob("*/results/perception/*/result.json"))
    if not paths:
        raise FileNotFoundError("no saved perception result found")
    return max(paths, key=lambda path: path.stat().st_mtime_ns).resolve()


def _largest_component(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import label
    except ImportError:
        return mask
    labels, count = label(mask, structure=np.ones((3, 3), dtype=bool))
    if count <= 0:
        return mask
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _rgb_garment_mask(image: Image.Image) -> np.ndarray:
    """Segment the dominant dark garment using RGB only.

    This is intentionally a diagnostic baseline for the current black-shirt /
    white-table scene, not the production garment segmenter. No depth, table
    plane, calibration, or point cloud is consulted.
    """

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    # Estimate the white tabletop from the bright majority and keep a generous
    # margin so dark cloth survives exposure changes.
    table_luma = float(np.percentile(luma, 75.0))
    # The current Camera-A frame contains a blue/purple fixture strip whose
    # luminance is darker than the white board but much brighter than the black
    # shirt. Cap the dark-cloth threshold so that strip is not connected to the
    # garment component. This is an RGB-only diagnostic assumption for the
    # present black-shirt/white-table experiment.
    threshold = float(np.clip(table_luma - 100.0, 60.0, 130.0))
    dark = luma < threshold
    # Remove tiny RGB speckles, then retain the dominant connected garment.
    try:
        from scipy.ndimage import binary_closing, binary_opening

        dark = binary_opening(dark, structure=np.ones((3, 3), dtype=bool), iterations=1)
        dark = binary_closing(dark, structure=np.ones((5, 5), dtype=bool), iterations=2)
    except ImportError:
        pass
    mask = _largest_component(dark)
    # Keep the neckline hole visible instead of filling every interior hole;
    # candidate points should represent actual cloth pixels.
    return np.asarray(mask, dtype=bool)


def _sprinkle_points(image: Image.Image, mask: np.ndarray, stride_px: int = 48) -> list[dict[str, Any]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError("RGB-only segmentation found no garment pixels")
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    samples: list[dict[str, Any]] = []
    height, width = mask.shape
    for y0 in range(y_min, y_max + 1, stride_px):
        for x0 in range(x_min, x_max + 1, stride_px):
            y1 = min(height, y0 + stride_px)
            x1 = min(width, x0 + stride_px)
            local_y, local_x = np.nonzero(mask[y0:y1, x0:x1])
            if len(local_x) == 0:
                continue
            center_x = (x1 - x0 - 1) / 2.0
            center_y = (y1 - y0 - 1) / 2.0
            nearest = int(np.argmin((local_x - center_x) ** 2 + (local_y - center_y) ** 2))
            x_px = int(x0 + local_x[nearest])
            y_px = int(y0 + local_y[nearest])
            samples.append(
                {
                    "reference_id": f"R{len(samples) + 1:03d}",
                    "pixel_xy": [x_px, y_px],
                }
            )
    if len(samples) < 2:
        raise RuntimeError("RGB-only segmentation produced fewer than two points")
    return samples


def _render_rgb_inputs(
    raw_path: Path,
    output: Path,
) -> tuple[Path, Path, list[dict[str, Any]], dict[str, Any]]:
    with Image.open(raw_path) as raw:
        upright = raw.convert("RGB").rotate(-90, expand=True)
    mask = _rgb_garment_mask(upright)
    samples = _sprinkle_points(upright, mask)
    overlay = upright.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for sample in samples:
        x, y = sample["pixel_xy"]
        reference_id = sample["reference_id"]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(0, 220, 220), outline=(0, 0, 0), width=2)
        draw.text(
            (x + 7, y - 8),
            reference_id,
            fill=(0, 0, 0),
            font=font,
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
    output.mkdir(parents=True, exist_ok=True)
    rgb_path = output / "camera_A_rgb_upright_rgb_only.png"
    overlay_path = output / "camera_A_rgb_points_overlay.png"
    mask_path = output / "camera_A_rgb_mask.png"
    upright.save(rgb_path)
    overlay.save(overlay_path)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(mask_path)
    metadata = {
        "segmentation": "RGB_ONLY_DARK_DOMINANT_COMPONENT",
        "source_rgb": str(raw_path),
        "rotation": "clockwise_90_degrees",
        "stride_px": 48,
        "point_count": len(samples),
        "samples": samples,
    }
    _write_json(output / "rgb_point_samples.json", metadata)
    return rgb_path, overlay_path, samples, metadata


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source_reference_id": {
                "type": "string",
                "pattern": "^R[0-9]{3,}$",
            },
            "destination_reference_id": {
                "type": "string",
                "pattern": "^R[0-9]{3,}$",
            },
        },
        "required": ["source_reference_id", "destination_reference_id"],
    }


def _invoke_claude(
    *,
    binary: str,
    images: list[Path],
    prompt: str,
    run_dir: Path,
    timeout_s: int,
) -> tuple[dict[str, Any], str, str, float]:
    image_text = "\n".join(f"- {path}" for path in images)
    full_prompt = (
        f"{prompt}\n\nRGB files to inspect:\n{image_text}\n\n"
        "Return only the requested JSON object. Do not read depth, height maps, "
        "coordinate guides, point clouds, or any other files. Do not output XYZ, Z, "
        "yaw, robot actions, or gripper commands."
    )
    command = [
        binary,
        "--print",
        full_prompt,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(_schema(), separators=(",", ":")),
        "--permission-mode",
        "dontAsk",
        "--allowedTools",
        "Read",
        "--tools",
        "Read",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--add-dir",
        str(run_dir.resolve()),
        "--system-prompt",
        (
            "You are a read-only RGB garment-fold analyst. Select exactly one visible "
            "source Rxxx point and one visible destination Rxxx point from the supplied "
            "upright Camera-A RGB image and marker overlay. Reason from RGB only and "
            "return only their two reference IDs."
        ),
    ]
    started = datetime.now(timezone.utc)
    completed = subprocess.run(
        command,
        cwd=run_dir,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    duration = (datetime.now(timezone.utc) - started).total_seconds()
    if completed.returncode != 0:
        raise RuntimeError(
            f"Claude exited with {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    parsed = _json_from_claude_text(completed.stdout)
    return parsed, completed.stdout, completed.stderr, duration


def _validate_plan(plan: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {sample["reference_id"]: sample for sample in samples}
    source_id = str(plan.get("source_reference_id", "")).upper()
    destination_id = str(plan.get("destination_reference_id", "")).upper()
    if source_id not in by_id:
        raise ValueError(f"selected unknown RGB source reference {source_id!r}")
    if destination_id not in by_id:
        raise ValueError(f"selected unknown RGB destination reference {destination_id!r}")
    # Only the two Rxxx identities are Claude's decisions. The marker table is
    # authoritative for display pixels and is filled in after the response.
    return {
        "source_reference_id": source_id,
        "destination_reference_id": destination_id,
        "source": {
            "reference_id": source_id,
            "pixel_xy": list(by_id[source_id]["pixel_xy"]),
        },
        "destination": {
            "reference_id": destination_id,
            "pixel_xy": list(by_id[destination_id]["pixel_xy"]),
        },
    }


def _render_transfer_plans(
    base_overlay: Path,
    plans: list[dict[str, Any]],
    output: Path,
) -> Path:
    with Image.open(base_overlay) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for index, record in enumerate(plans):
        color = COLORS[index % len(COLORS)]
        plan = record["plan"]
        sx, sy = plan["source"]["pixel_xy"]
        dx, dy = plan["destination"]["pixel_xy"]
        draw.line((sx, sy, dx, dy), fill=(255, 255, 255), width=9)
        draw.line((sx, sy, dx, dy), fill=color, width=5)
        draw.ellipse((sx - 16, sy - 16, sx + 16, sy + 16), outline=color, width=6)
        draw.rectangle((dx - 14, dy - 14, dx + 14, dy + 14), outline=color, width=6)
        draw.text(
            (sx + 20, sy - 14),
            f"I{record['round']} {plan['source']['reference_id']}→{plan['destination']['reference_id']}",
            fill=color,
            font=font,
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--perception-result", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--claude-timeout-s", type=int, default=900)
    args = parser.parse_args()
    root = args.project_root.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir else root / "runs" / (
        "rgb_point_transfer_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    perception_result = (
        args.perception_result.resolve()
        if args.perception_result
        else _latest_result(root)
    )
    result = json.loads(perception_result.read_text(encoding="utf-8"))
    view = next(
        item
        for item in result["views"]
        if str(item.get("label", "")).upper() == "A"
    )
    raw_path = perception_result.parent / str(view["image"])
    output = run_dir / "results" / "rgb_point_transfer" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    rgb_path, overlay_path, samples, metadata = _render_rgb_inputs(raw_path, output)
    reference_images: list[Path] = []
    reference_dir = root / "data" / "reference" / "flat_garment_reference"
    for name in ("camera_A_flat_reference.png", "camera_A_flat_reference_anchors.png"):
        path = reference_dir / name
        if path.is_file():
            reference_images.append(path.resolve())
    images = [rgb_path, overlay_path, *reference_images]
    plans: list[dict[str, Any]] = []
    for round_index in range(1, args.rounds + 1):
        history = [
            {
                "round": record["round"],
                "source_reference_id": record["plan"]["source_reference_id"],
                "destination_reference_id": record["plan"]["destination_reference_id"],
            }
            for record in plans
        ]
        prompt = (
            "RGB-ONLY POINT-TRANSFER FOLDING PREVIEW. The current Camera A image is an "
            "upright RGB top view of one shirt. Cyan Rxxx markers are uniformly sprinkled "
            "on the RGB garment mask. Choose exactly one source marker to grasp and one "
            "destination marker where that cloth point should be moved for the next useful "
            "step toward a neat fold. Return exactly two fields: "
            "source_reference_id and destination_reference_id. Do not output a trajectory, "
            "pixel coordinates, reasons, intent, confidence, or any other fields. Do not use depth or "
            "any 3-D information. The destination may be on the torso, sleeve, or another "
            "garment region, but it must be one of the visible Rxxx markers. Use the current "
            "RGB image as the scene truth. The flat reference is RGB topology help only.\n\n"
            "Previous hypothetical RGB plans (the image is unchanged; nothing was executed):\n"
            + json.dumps(history, ensure_ascii=False, indent=2)
        )
        prompt_path = output / f"round_{round_index:02d}.prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        print(f"[rgb-point-transfer] round {round_index}/{args.rounds}: asking Claude", flush=True)
        parsed, raw_stdout, raw_stderr, duration = _invoke_claude(
            binary=args.claude_binary,
            images=images,
            prompt=prompt,
            run_dir=run_dir,
            timeout_s=args.claude_timeout_s,
        )
        plan = _validate_plan(parsed, samples)
        record = {
            "round": round_index,
            "duration_s": duration,
            "prompt_path": str(prompt_path),
            "plan": plan,
            "raw_stdout": raw_stdout,
            "raw_stderr": raw_stderr,
        }
        plans.append(record)
        _write_json(output / f"round_{round_index:02d}.json", record)
        _write_json(output / "plans.json", plans)
        _render_transfer_plans(
            overlay_path,
            plans,
            output / "rgb_point_transfer_all_rounds.png",
        )
        print(
            f"[rgb-point-transfer] round {round_index}: "
            f"{plan['source']['reference_id']} -> {plan['destination']['reference_id']}",
            flush=True,
        )
    summary = {
        "status": "COMPLETED",
        "mode": "RGB_ONLY_POINT_TRANSFER",
        "physical_execution": False,
        "depth_used_for_planning": False,
        "perception_result": str(perception_result),
        "output_dir": str(output),
        "rounds": plans,
        "rgb_inputs": [str(path) for path in images],
        "point_metadata": metadata,
    }
    _write_json(output / "summary.json", summary)
    print(f"[rgb-point-transfer] saved={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
