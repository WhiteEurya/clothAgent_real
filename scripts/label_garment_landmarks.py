"""Capture a flattened garment and label multiple independent Molmo landmarks.

This is intentionally a standalone baseline-generation script. It does not
change the automatic Claude loop. Molmo is queried separately once per named
garment part; the model is loaded once and reused for all queries.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.perception import (
    PerceptionConfig,
    capture_two_view_rgbd,
    pixel_to_base_mm,
    robust_depth_at_pixel,
)


@dataclass(frozen=True)
class LandmarkSpec:
    name: str
    description: str
    color: tuple[int, int, int]


DEFAULT_LANDMARKS = (
    LandmarkSpec("garment_center", "the geometric center of the whole garment", (255, 40, 40)),
    LandmarkSpec("neckline", "the collar, neckline, or neck opening", (255, 170, 0)),
    LandmarkSpec("left_shoulder", "the left shoulder or upper-left shoulder seam", (50, 180, 255)),
    LandmarkSpec("right_shoulder", "the right shoulder or upper-right shoulder seam", (80, 220, 80)),
    LandmarkSpec("left_sleeve_tip", "the outer tip of the left sleeve or left upper edge", (180, 80, 255)),
    LandmarkSpec("right_sleeve_tip", "the outer tip of the right sleeve or right upper edge", (255, 80, 190)),
    LandmarkSpec("left_bottom_hem", "the leftmost point on the bottom hem", (80, 220, 220)),
    LandmarkSpec("right_bottom_hem", "the rightmost point on the bottom hem", (220, 220, 60)),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _prompts(specs: tuple[LandmarkSpec, ...]) -> list[str]:
    return [
        (
            "The garment is intentionally laid flat for a landmark baseline. "
            "Return exactly one point and no other points. The point must lie on "
            f"visible garment fabric at the {spec.description}; do not point to "
            "the table, robot, or nearby objects."
        )
        for spec in specs
    ]


def _group_points(raw_points: Any, image_count: int, expected_count: int) -> dict[int, list[tuple[float, float]]]:
    if not isinstance(raw_points, list):
        raise RuntimeError("Molmo output does not contain landmark_points")
    grouped: dict[int, list[tuple[int, tuple[float, float]]]] = {
        index: [] for index in range(image_count)
    }
    for raw in raw_points:
        if not isinstance(raw, dict):
            raise RuntimeError(f"invalid independent Molmo landmark record: {raw!r}")
        image_index = int(raw["image_index"])
        prompt_index = int(raw["prompt_index"])
        pixel = raw["pixel_xy"]
        if image_index not in grouped or not isinstance(pixel, list) or len(pixel) != 2:
            raise RuntimeError(f"invalid Molmo landmark record: {raw!r}")
        x_px, y_px = float(pixel[0]), float(pixel[1])
        if not np.isfinite(x_px) or not np.isfinite(y_px):
            raise RuntimeError(f"Molmo returned a non-finite pixel: {raw!r}")
        grouped[image_index].append((prompt_index, (x_px, y_px)))
    result: dict[int, list[tuple[float, float]]] = {}
    for image_index, points in grouped.items():
        if len(points) != expected_count or {index for index, _ in points} != set(range(expected_count)):
            raise RuntimeError(
                f"Molmo returned {len(points)} independent points for image {image_index}; "
                f"expected one for each of {expected_count} queries"
            )
        points.sort(key=lambda item: item[0])
        result[image_index] = [pixel for _, pixel in points]
    return result


def _annotate(
    image_path: Path,
    points: list[tuple[float, float]],
    specs: tuple[LandmarkSpec, ...],
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for index, (spec, (x_px, y_px)) in enumerate(zip(specs, points), start=1):
        radius = 7
        draw.ellipse(
            (x_px - radius, y_px - radius, x_px + radius, y_px + radius),
            outline=spec.color,
            width=3,
        )
        draw.text(
            (x_px + 9, y_px - 8),
            f"{index}:{spec.name}",
            fill=spec.color,
            font=font,
        )
    image.save(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--perception-config",
        default="config/perception.free_exploration.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="output directory; defaults to runs/landmarks_<timestamp>",
    )
    parser.add_argument(
        "--landmarks-json",
        default=None,
        help="optional JSON list [{name, description, color:[r,g,b]}]",
    )
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve()
    config = PerceptionConfig.load(root, (root / args.perception_config).resolve())
    if len(config.active_camera_labels) != 2:
        raise RuntimeError("landmark baseline requires active_cameras to contain A and B")

    if args.landmarks_json:
        raw_specs = json.loads(Path(args.landmarks_json).expanduser().resolve().read_text(encoding="utf-8"))
        if not isinstance(raw_specs, list) or not raw_specs:
            raise RuntimeError("--landmarks-json must contain a non-empty JSON list")
        specs = tuple(
            LandmarkSpec(
                str(item["name"]),
                str(item["description"]),
                tuple(int(value) for value in item.get("color", [255, 255, 255])),
            )
            for item in raw_specs
        )
    else:
        specs = DEFAULT_LANDMARKS
    if not 2 <= len(specs) <= 10:
        raise RuntimeError("the landmark list must contain between 2 and 10 points")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / "runs" / f"landmarks_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    capture_dir = output_dir / "capture"
    capture_dir.mkdir(parents=True, exist_ok=False)

    print("Capturing A/B RGB-D baseline; keep the garment fully spread out.")
    frames = capture_two_view_rgbd(config)
    image_paths: list[Path] = []
    for index, frame in enumerate(frames):
        image_path = capture_dir / f"camera_{index}_{frame.label}.png"
        Image.fromarray(frame.rgb.astype(np.uint8)).save(image_path)
        np.save(capture_dir / f"camera_{index}_{frame.label}_depth_m.npy", frame.depth_m)
        image_paths.append(image_path)

    molmo_output = output_dir / "molmo_landmarks.json"
    prompt_list = _prompts(specs)
    print(
        f"Asking Molmo {len(specs)} independent one-point queries per camera "
        "(one model load, separate prompts)..."
    )
    worker = root / "cloth_agent" / "molmo_landmark_worker.py"
    command = [
        str(config.molmo.python),
        str(worker),
        "--output",
        str(molmo_output),
        "--model",
        config.molmo.model,
        "--dtype",
        config.molmo.dtype,
        "--max-crops",
        str(config.molmo.max_crops),
        "--max-new-tokens",
        str(config.molmo.max_new_tokens),
    ]
    if config.molmo.local_files_only:
        command.append("--local-files-only")
    for image_path in image_paths:
        command.extend(["--image", str(image_path)])
    for prompt in prompt_list:
        command.extend(["--prompt", prompt])
    import subprocess

    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        timeout=config.molmo.timeout_s,
        check=False,
        shell=False,
    )
    (output_dir / "molmo_landmarks.stdout.txt").write_text(
        completed.stdout + ("\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Molmo independent landmark worker exited with {completed.returncode}; "
            f"inspect {output_dir / 'molmo_landmarks.stdout.txt'}"
        )
    molmo_result = json.loads(molmo_output.read_text(encoding="utf-8"))
    grouped = _group_points(molmo_result.get("landmark_points"), len(frames), len(specs))

    views: list[dict[str, Any]] = []
    for index, (frame, image_path) in enumerate(zip(frames, image_paths)):
        points = grouped[index]
        observations: list[dict[str, Any]] = []
        for landmark_index, (spec, (x_px, y_px)) in enumerate(zip(specs, points), start=1):
            depth_m = robust_depth_at_pixel(
                frame.depth_m,
                x_px,
                y_px,
                config.depth_window_radius_px,
                config.min_depth_m,
                config.max_depth_m,
            )
            point_base_mm = pixel_to_base_mm(
                x_px,
                y_px,
                depth_m,
                frame.intrinsics,
                frame.X_base_camera,
            )
            observations.append(
                {
                    "index": landmark_index,
                    "name": spec.name,
                    "description": spec.description,
                    "pixel_xy": [x_px, y_px],
                    "depth_m": depth_m,
                    "point_base_mm": point_base_mm,
                }
            )
        annotated_path = output_dir / f"camera_{index}_{frame.label}_landmarks.png"
        _annotate(image_path, points, specs, annotated_path)
        views.append(
            {
                "label": frame.label,
                "serial": frame.serial,
                "image": str(image_path.relative_to(output_dir)),
                "annotated_image": str(annotated_path.relative_to(output_dir)),
                "intrinsics": frame.intrinsics,
                "X_base_camera": frame.X_base_camera,
                "landmarks": observations,
            }
        )

    baseline = {
        "created_at": _now(),
        "status": "BASELINE_LANDMARKS_LABELED",
        "landmark_order": [spec.name for spec in specs],
        "query_mode": "one_independent_point_prompt_per_landmark",
        "prompts": prompt_list,
        "molmo": molmo_result,
        "views": views,
    }
    _save_json(output_dir / "landmarks_baseline.json", baseline)
    print(json.dumps({"output_dir": str(output_dir), "landmarks": [spec.name for spec in specs]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
