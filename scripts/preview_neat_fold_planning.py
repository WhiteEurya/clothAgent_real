#!/usr/bin/env python3
"""Generate Claude garment-fold plans with Rxxx grasp anchoring but no IK or robot motion."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.free_exploration import (  # noqa: E402
    ClaudeExplorationClient,
    _load_latest_perception,
    _load_or_create_session,
)
from cloth_agent.neat_fold_pipeline import (  # noqa: E402
    NEAT_FOLD_PLANNING_OBJECTIVE,
    _claude_runtime_metrics,
    _workspace_prefilter_overlay,
)
from cloth_agent.planning_preview_viser import (  # noqa: E402
    run_viewer as run_planning_preview_viser,
    write_alignment_report,
)
from cloth_agent.perception import PerceptionConfig, capture_two_view_rgbd  # noqa: E402
from cloth_agent.robot_api import move_robot_to_perception_position  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _copy_reference(root: Path, output: Path) -> list[Path]:
    source = root / "data" / "reference" / "flat_garment_reference"
    target = output / "reference"
    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in ("camera_A_flat_reference.png", "camera_A_flat_reference_anchors.png"):
        path = source / name
        if path.is_file():
            destination = target / name
            shutil.copy2(path, destination)
            paths.append(destination.resolve())
    if not paths:
        raise FileNotFoundError(f"flat reference files are missing: {source}")
    return paths


def _rgb_only_planning_images(
    saved: dict[str, Any],
    saved_path: Path,
    reference_images: list[Path],
    output: Path,
) -> list[Path]:
    """Return canonical upright Camera-A RGB plus the visual Rxxx overlay."""

    paths: list[Path] = []

    def add(path: Path) -> None:
        path = path.resolve()
        if path.is_file() and path not in paths:
            paths.append(path)

    for view in saved.get("views", []):
        if not isinstance(view, dict):
            continue
        label = str(view.get("label", "")).upper()
        # Camera A is the action view. Rotate it clockwise so the garment's
        # collar-to-hem axis matches the upright flat reference (collar top,
        # hem bottom). Camera B is omitted here to avoid mixing orientations.
        if label != "A":
            continue
        raw_rgb = view.get("image")
        raw_rgb_source = (
            saved_path.parent / str(raw_rgb) if raw_rgb else None
        )
        raw_rgb_image = (
            Image.open(raw_rgb_source).convert("RGB")
            if raw_rgb_source is not None and raw_rgb_source.is_file()
            else None
        )
        for key, output_name in (
            ("image", "camera_A_rgb_upright.png"),
            ("coordinate_overlay", "camera_A_rxxx_overlay_upright.png"),
        ):
            raw = view.get(key)
            if not raw:
                continue
            source = saved_path.parent / str(raw)
            if not source.is_file():
                continue
            if key == "coordinate_overlay" and raw_rgb_image is not None:
                # Rebuild the marker layer on top of the rotated RGB instead of
                # rotating the old raster overlay. This keeps Rxxx text upright
                # and avoids making Claude read every label sideways.
                rotated = raw_rgb_image.rotate(-90, expand=True)
                try:
                    guide_path = saved_path.parent / str(view.get("coordinate_guide", ""))
                    guide = json.loads(guide_path.read_text(encoding="utf-8"))
                    draw = ImageDraw.Draw(rotated)
                    font = ImageFont.load_default()
                    for sample in guide.get("samples", []):
                        if not isinstance(sample, dict):
                            continue
                        reference_id = sample.get("reference_id")
                        pixel = sample.get("pixel_xy")
                        if not isinstance(reference_id, str) or not isinstance(pixel, list) or len(pixel) != 2:
                            continue
                        marker = _clockwise90_pixel(
                            float(pixel[0]),
                            float(pixel[1]),
                            raw_rgb_image.width,
                            raw_rgb_image.height,
                        )
                        mx, my = marker
                        draw.ellipse((mx - 5, my - 5, mx + 5, my + 5), fill=(0, 220, 220), outline=(0, 0, 0), width=2)
                        draw.text(
                            (mx + 7, my - 8),
                            reference_id,
                            fill=(0, 0, 0),
                            font=font,
                            stroke_width=2,
                            stroke_fill=(255, 255, 255),
                        )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    # Fall back to the saved overlay if the guide is unavailable.
                    rotated = Image.open(source).convert("RGB").rotate(-90, expand=True)
            else:
                rotated = Image.open(source).convert("RGB").rotate(-90, expand=True)
            destination = output / output_name
            rotated.save(destination)
            add(destination)
    for path in reference_images:
        add(path)
    if not paths:
        raise RuntimeError("RGB-only planning has no saved RGB images")
    return paths


def _clockwise90_pixel(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    """Map a source-image pixel into the PIL ``rotate(-90, expand=True)`` frame."""

    return float(height - 1 - y), float(x)


def _camera_a_coordinate_samples(saved: dict[str, Any], saved_path: Path) -> dict[str, dict[str, Any]]:
    view = next(
        (
            item
            for item in saved.get("views", [])
            if isinstance(item, dict) and str(item.get("label", "")).upper() == "A"
        ),
        None,
    )
    if view is None or not view.get("coordinate_guide"):
        raise RuntimeError("Camera A coordinate guide is unavailable")
    guide_path = saved_path.parent / str(view["coordinate_guide"])
    guide = json.loads(guide_path.read_text(encoding="utf-8"))
    samples = guide.get("samples") if isinstance(guide, dict) else None
    if not isinstance(samples, list) or not samples:
        raise RuntimeError(f"Camera A coordinate guide has no Rxxx samples: {guide_path}")
    result: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if isinstance(sample, dict) and isinstance(sample.get("reference_id"), str):
            result[str(sample["reference_id"]).upper()] = sample
    if not result:
        raise RuntimeError(f"Camera A coordinate guide has no valid Rxxx samples: {guide_path}")
    return result


def _anchor_preview_proposal(
    proposal: Any,
    samples: dict[str, dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    """Use Claude's Rxxx choice to anchor the complete preview path.

    The transport shape, relative displacements, Z profile, and yaw remain
    Claude's free-motion plan. Only the global XY translation and the grasp
    waypoint Z are grounded to the selected Rxxx sample so the displayed path
    actually starts at the selected cloth point.
    """

    selected = dict(proposal.selected_grasp or {})
    reference_id = str(selected.get("reference_id", "")).upper()
    sample = samples.get(reference_id)
    if sample is None:
        raise RuntimeError(f"Claude selected unknown Camera-A Rxxx reference: {reference_id!r}")
    pixel = sample.get("pixel_xy")
    base_xyz = sample.get("base_xyz_mm")
    if (
        not isinstance(pixel, list)
        or len(pixel) != 2
        or not isinstance(base_xyz, list)
        or len(base_xyz) != 3
    ):
        raise RuntimeError(f"Rxxx sample {reference_id} has incomplete pixel/base XYZ data")
    canonical_pixel = [int(round(float(pixel[0]))), int(round(float(pixel[1])))]
    target_xyz = np.asarray([float(value) for value in base_xyz], dtype=np.float64)
    actions = [dict(action) for action in proposal.actions]
    close_index = next(
        (index for index, action in enumerate(actions) if action.get("name") == "close_gripper"),
        None,
    )
    if close_index is None:
        raise RuntimeError("preview proposal has no close_gripper")
    grasp_index = next(
        (index for index in range(close_index - 1, -1, -1) if actions[index].get("name") == "move"),
        None,
    )
    if grasp_index is None:
        raise RuntimeError("preview proposal has no move before close_gripper")
    raw_grasp = actions[grasp_index].get("args", {})
    if not isinstance(raw_grasp, dict):
        raise RuntimeError("preview grasp move has invalid args")
    raw_xy = np.asarray([float(raw_grasp["x"]), float(raw_grasp["y"])], dtype=np.float64)
    delta_xy = target_xyz[:2] - raw_xy
    for action in actions:
        if action.get("name") != "move" or not isinstance(action.get("args"), dict):
            continue
        args = dict(action["args"])
        args["x"] = float(args["x"]) + float(delta_xy[0])
        args["y"] = float(args["y"]) + float(delta_xy[1])
        action["args"] = args
    grasp_args = dict(actions[grasp_index]["args"])
    grasp_args["x"] = float(target_xyz[0])
    grasp_args["y"] = float(target_xyz[1])
    grasp_args["z"] = float(target_xyz[2])
    actions[grasp_index]["args"] = grasp_args
    selected.update({"camera": "A", "reference_id": reference_id, "pixel_xy": canonical_pixel})
    anchored = replace(proposal, actions=tuple(actions), selected_grasp=selected)
    diagnostic = {
        "reference_id": reference_id,
        "pixel_xy": canonical_pixel,
        "base_xyz_mm": [float(value) for value in target_xyz],
        "raw_grasp_xy_mm": [float(value) for value in raw_xy],
        "xy_translation_applied_mm": [float(value) for value in delta_xy],
        "trajectory_anchor_mode": "RXXX_GROUNDED_TRANSLATION_FREE_TRANSPORT",
    }
    return anchored, diagnostic


def _capture_current_perception(
    *,
    root: Path,
    session: Any,
    perception_config: Path,
    move_to_perception: bool,
) -> tuple[dict[str, Any], Path]:
    """Capture and save a fresh synchronized Camera A/B perception result."""

    config_path = perception_config.expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    config = PerceptionConfig.load(root, config_path.resolve())
    if move_to_perception:
        print("[planning-preview] moving robot to calibrated perception pose", flush=True)
        move_robot_to_perception_position(session.robot_config)
    print("[planning-preview] capturing synchronized Camera A/B RGB-D", flush=True)
    frames = capture_two_view_rgbd(config)
    print("[planning-preview] processing dense garment perception", flush=True)
    session.locate_cloth_center(config, frames=frames)
    saved, saved_path = _load_latest_perception(session)
    if saved is None or saved_path is None:
        raise RuntimeError("camera capture completed without a saved perception result")
    return saved, saved_path


def _trajectory(proposal: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gripper = "unknown"
    for index, action in enumerate(proposal.actions, start=1):
        name = str(action["name"])
        args = dict(action.get("args", {}))
        if name == "open_gripper":
            gripper = "open"
        elif name == "close_gripper":
            gripper = "closed"
        row: dict[str, Any] = {
            "step": index,
            "action": name,
            "gripper_state": gripper,
        }
        if name == "move":
            row["pose_mm_deg"] = {
                key: float(args[key]) for key in ("x", "y", "z", "yaw")
            }
        rows.append(row)
    return rows


def _format_plan(proposal: Any) -> str:
    lines = []
    for row in _trajectory(proposal):
        if "pose_mm_deg" in row:
            pose = row["pose_mm_deg"]
            lines.append(
                f"  {row['step']:02d}. move(x={pose['x']:.1f}, y={pose['y']:.1f}, "
                f"z={pose['z']:.1f}, yaw={pose['yaw']:.1f}) [{row['gripper_state']}]"
            )
        else:
            lines.append(f"  {row['step']:02d}. {row['action']} [{row['gripper_state']}]" )
    return "\n".join(lines)


TRAJECTORY_COLORS = [
    (230, 45, 45),
    (40, 130, 255),
    (40, 180, 80),
    (220, 125, 25),
    (170, 70, 220),
    (220, 45, 150),
    (20, 170, 170),
    (180, 160, 30),
]


def _proposal_actions(proposal: Any) -> tuple[dict[str, Any], ...] | list[dict[str, Any]]:
    """Return actions from either a live proposal or its JSON-safe dict form."""

    if hasattr(proposal, "actions"):
        return proposal.actions
    if isinstance(proposal, dict):
        actions = proposal.get("actions", [])
        if isinstance(actions, list):
            return actions
    return []


def _proposal_selected_grasp(proposal: Any) -> dict[str, Any]:
    if hasattr(proposal, "selected_grasp"):
        selected = proposal.selected_grasp
    elif isinstance(proposal, dict):
        selected = proposal.get("selected_grasp")
    else:
        selected = None
    return dict(selected) if isinstance(selected, dict) else {}


def _move_rows(proposal: Any) -> list[dict[str, Any]]:
    """Extract move rows from either an ExplorationProposal or saved JSON."""

    rows: list[dict[str, Any]] = []
    gripper = "unknown"
    for index, action in enumerate(_proposal_actions(proposal), start=1):
        if not isinstance(action, dict):
            continue
        name = str(action.get("name", ""))
        args = action.get("args", {})
        if not isinstance(args, dict):
            args = {}
        if name == "open_gripper":
            gripper = "open"
        elif name == "close_gripper":
            gripper = "closed"
        if name != "move":
            continue
        try:
            pose = {key: float(args[key]) for key in ("x", "y", "z", "yaw")}
        except (KeyError, TypeError, ValueError):
            # Keep visualization best-effort if a future preview schema emits a
            # non-Cartesian or partially specified action.
            continue
        rows.append({"step": index, "action": name, "gripper_state": gripper, "pose_mm_deg": pose})
    return rows


def _project_base_xyz(view: dict[str, Any], xyz_mm: list[float]) -> tuple[float, float] | None:
    intrinsics = np.asarray(view["intrinsics"], dtype=np.float64)
    base_from_camera = np.asarray(view["X_base_camera"], dtype=np.float64)
    camera_from_base = np.linalg.inv(base_from_camera)
    homogeneous = np.concatenate((np.asarray(xyz_mm, dtype=np.float64) / 1000.0, [1.0]))
    camera = camera_from_base @ homogeneous
    if not np.all(np.isfinite(camera)) or camera[2] <= 0.0:
        return None
    return (
        float(intrinsics[0, 0] * camera[0] / camera[2] + intrinsics[0, 2]),
        float(intrinsics[1, 1] * camera[1] / camera[2] + intrinsics[1, 2]),
    )


def _render_trajectory_visuals(
    *,
    plans: list[dict[str, Any]],
    result: dict[str, Any],
    result_path: Path,
    robot_config: Any,
    output: Path,
) -> dict[str, str]:
    """Render every preview plan in Camera-A and base-frame XY views."""

    view = next(
        item for item in result["views"] if str(item.get("label", "")).upper() == "A"
    )
    rgb_path = result_path.parent / str(view["image"])
    # Keep the diagnostic trajectory view in the same upright frame Claude sees.
    # Grounding/projection still uses the raw Camera-A calibration; only the
    # rendered pixel is transformed for display.
    raw_camera_image = Image.open(rgb_path).convert("RGB")
    source_width, source_height = raw_camera_image.size
    camera_image = raw_camera_image.rotate(-90, expand=True)
    camera_draw = ImageDraw.Draw(camera_image)
    font = ImageFont.load_default()
    all_xyz: list[list[float]] = []

    for index, record in enumerate(plans):
        color = TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)]
        rows = _move_rows(record["proposal"])
        poses = [row["pose_mm_deg"] for row in rows]
        points = [[pose["x"], pose["y"], pose["z"]] for pose in poses]
        all_xyz.extend(points)
        projected = [_project_base_xyz(view, point) for point in points]
        previous: tuple[float, float] | None = None
        for step, pixel in zip(rows, projected):
            if pixel is None:
                previous = None
                continue
            upright_pixel = _clockwise90_pixel(
                pixel[0], pixel[1], source_width, source_height
            )
            if previous is not None:
                camera_draw.line(
                    (previous[0], previous[1], upright_pixel[0], upright_pixel[1]),
                    fill=color,
                    width=6,
                )
                camera_draw.line(
                    (previous[0], previous[1], upright_pixel[0], upright_pixel[1]),
                    fill=(20, 20, 20),
                    width=2,
                )
            x, y = upright_pixel
            if -30 <= x <= camera_image.width + 30 and -30 <= y <= camera_image.height + 30:
                camera_draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color, outline=(255, 255, 255), width=3)
                camera_draw.text((x + 12, y - 16), f"I{record['round']} M{step['step']}", fill=color, font=font)
            previous = upright_pixel

    camera_panel_width = 320
    camera_canvas = Image.new("RGB", (camera_image.width + camera_panel_width, camera_image.height), (28, 28, 28))
    camera_canvas.paste(camera_image, (0, 0))
    panel = ImageDraw.Draw(camera_canvas)
    panel.text((camera_image.width + 15, 15), "Camera A projected plans", fill=(255, 255, 255), font=font)
    for index, record in enumerate(plans):
        color = TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)]
        y = 45 + index * 22
        panel.line((camera_image.width + 15, y + 6, camera_image.width + 40, y + 6), fill=color, width=6)
        selected = _proposal_selected_grasp(record["proposal"])
        panel.text((camera_image.width + 50, y), f"Iter {record['round']} grasp={selected.get('pixel_xy')}", fill=color, font=font)
    camera_path = output / "trajectory_camera_A.png"
    camera_canvas.save(camera_path)

    bounds = robot_config.boundaries
    xs = [point[0] for point in all_xyz] or [500.0]
    ys = [point[1] for point in all_xyz] or [0.0]
    x_min = float(bounds.x_min if bounds.x_min is not None else min(xs) - 50.0)
    x_max = float(bounds.x_max if bounds.x_max is not None else max(xs) + 50.0)
    y_min = float(bounds.y_min if bounds.y_min is not None else min(ys) - 50.0)
    y_max = float(bounds.y_max if bounds.y_max is not None else max(ys) + 50.0)
    if x_max <= x_min:
        x_max = x_min + 100.0
    if y_max <= y_min:
        y_max = y_min + 100.0

    plot_w, plot_h, margin = 1100, 760, 70
    xy = Image.new("RGB", (plot_w, plot_h), (245, 245, 245))
    draw = ImageDraw.Draw(xy)

    def to_pixel(x_mm: float, y_mm: float) -> tuple[float, float]:
        px = margin + (x_mm - x_min) / (x_max - x_min) * (plot_w - 2 * margin)
        py = plot_h - margin - (y_mm - y_min) / (y_max - y_min) * (plot_h - 2 * margin)
        return px, py

    workspace_corners = [
        to_pixel(x_min, y_min),
        to_pixel(x_max, y_min),
        to_pixel(x_max, y_max),
        to_pixel(x_min, y_max),
    ]
    draw.line(workspace_corners + [workspace_corners[0]], fill=(80, 80, 80), width=3)
    draw.text((margin, 18), "Base-frame XY planning trajectories (IK intentionally ignored)", fill=(20, 20, 20), font=font)
    draw.text((margin, plot_h - 45), f"x: {x_min:.1f}..{x_max:.1f} mm", fill=(50, 50, 50), font=font)
    draw.text((plot_w - 280, plot_h - 45), f"y: {y_min:.1f}..{y_max:.1f} mm", fill=(50, 50, 50), font=font)

    for index, record in enumerate(plans):
        color = TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)]
        rows = _move_rows(record["proposal"])
        previous: tuple[float, float] | None = None
        for step in rows:
            pose = step["pose_mm_deg"]
            pixel = to_pixel(float(pose["x"]), float(pose["y"]))
            if previous is not None:
                draw.line((previous[0], previous[1], pixel[0], pixel[1]), fill=color, width=6)
                draw.line((previous[0], previous[1], pixel[0], pixel[1]), fill=(20, 20, 20), width=2)
            draw.ellipse((pixel[0] - 9, pixel[1] - 9, pixel[0] + 9, pixel[1] + 9), fill=color, outline=(255, 255, 255), width=2)
            draw.text((pixel[0] + 11, pixel[1] - 14), f"I{record['round']} M{step['step']}", fill=color, font=font)
            previous = pixel
    xy_path = output / "trajectory_base_xy.png"
    xy.save(xy_path)
    return {"camera_A": str(camera_path), "base_xy": str(xy_path)}


def run_preview(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.project_root).resolve()
    generated_run_id: str | None = None
    if args.run_id is None and args.run_dir is None:
        generated_run_id = datetime.now(timezone.utc).strftime(
            "neat_fold_preview_%Y%m%dT%H%M%S%fZ"
        )
        print(f"[planning-preview] generated run-id={generated_run_id}", flush=True)
    session = _load_or_create_session(
        root,
        Path(args.run_dir).resolve() if args.run_dir else None,
        args.run_id or generated_run_id,
        None,
    )
    if args.reuse_latest_perception:
        print("[planning-preview] reusing latest saved perception", flush=True)
        saved, saved_path = _load_latest_perception(session)
    else:
        saved, saved_path = _capture_current_perception(
            root=root,
            session=session,
            perception_config=args.perception_config,
            move_to_perception=args.move_to_perception,
        )
    if saved is None or saved_path is None:
        raise RuntimeError(
            "no saved perception result is available; capture failed or "
            "--reuse-latest-perception was used on an empty run"
        )
    output = session.results / "neat_fold_planning_preview" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    reference_images = _copy_reference(root, output)
    # Restore the previous Rxxx selection workflow, but keep the visual stage
    # RGB-only. The calibrated Rxxx JSON guide is read privately after Claude
    # selects a marker; it is never included in the Claude file list.
    before_images = _rgb_only_planning_images(
        saved,
        saved_path,
        reference_images,
        output,
    )
    upright_path = output / "camera_A_rgb_upright.png"
    if upright_path.is_file():
        with Image.open(upright_path) as upright_image:
            upright_size = list(upright_image.size)
    else:
        upright_size = None
    _write_json(
        output / "camera_A_display_transform.json",
        {
            "source_camera": "A",
            "operation": "clockwise_90_degrees",
            "implementation": "PIL.Image.rotate(-90, expand=True)",
            "display_size_px": upright_size,
            "reason": "match collar-to-hem direction to the upright flat reference",
            "rxxx_overlay": "re-rendered on rotated RGB with upright labels",
            "grounding": "Rxxx reference_id is mapped back to the original Camera-A calibration internally",
        },
    )
    coordinate_samples = _camera_a_coordinate_samples(saved, saved_path)
    # The workspace overlay is useful for operator diagnostics, but it must not
    # bias this planning-only experiment. Claude receives the RGB-D/reference
    # evidence without robot workspace, table-clearance, or IK boundaries.
    overlay_meta: dict[str, Any] = {
        "sent_to_claude": False,
        "reason": "free-motion planning preview ignores workspace constraints",
    }
    if args.save_workspace_overlay:
        _, overlay_meta = _workspace_prefilter_overlay(
            saved,
            saved_path,
            session.robot_config,
            output / "camera_A_workspace_prefilter.png",
        )
        overlay_meta["sent_to_claude"] = False
    planning_images = before_images
    planner = ClaudeExplorationClient(binary=args.claude_binary, timeout_s=args.claude_timeout_s)
    plans: list[dict[str, Any]] = []
    base_prompt = (
        NEAT_FOLD_PLANNING_OBJECTIVE
        + "\n\nEXPLICIT T-SHIRT FOLD ORDER — follow this phase order across iterations:\n"
        "PHASE 1 — SLEEVE ORGANIZATION: find a visible distal sleeve cuff/free edge "
        "and fold that sleeve inward onto the torso. Process the other sleeve in a "
        "later iteration; do not grab the shoulder/root when a distal sleeve marker "
        "is visible. Do not start the hem fold while an outer sleeve still protrudes.\n"
        "PHASE 2 — BODY SIDE FOLDS: only after both sleeves are planned as tucked, "
        "fold the garment's left and right body side panels inward toward the garment "
        "centerline, one side per iteration. Use the garment topology, not image-left "
        "or image-right, because the shirt may be rotated.\n"
        "PHASE 3 — HEM-UP FOLD: only after sleeves and side panels are tucked, grasp "
        "the bottom hem's distal free edge and fold the lower body upward toward the "
        "collar/shoulder end. This is the final length-halving fold.\n"
        "At the beginning of every iteration, identify the current phase, list which "
        "parts are already planned in the history ledger, and choose exactly one "
        "unfinished sub-action from the earliest incomplete phase. Never jump directly "
        "to the hem just because it is easy to see.\n"
        + "\n\nFREE-MOTION PLANNING PREVIEW: temporarily ignore robot workspace "
        "bounds, table-clearance limits, collision limits, speed limits, reachability, "
        "and controller IK. Do not reject a useful fold idea because a numeric waypoint "
        "would later be unsafe or unreachable. Coordinates are hypothetical diagnostics "
        "only. The purpose is to inspect the complete garment-level plan and every "
        "gripper waypoint. This visual stage is RGB-only: do not use height, depth, "
        "gradient, table, or coordinate-guide values to rank or reject a marker. "
        "Camera A is displayed after a clockwise 90-degree image rotation so the "
        "shirt is upright relative to the flat reference; reason in this rotated "
        "image orientation. The Rxxx ID remains the authoritative grasp identity. "
        "Grasp selection must use one displayed Rxxx reference; only later transport "
        "waypoints are free-motion."
    )
    summary: dict[str, Any] = {
        "created_at": _now(),
        "status": "RUNNING",
        "mode": "PLANNING_ONLY_PREVIEW",
        "execution": "DISABLED",
        "grounding": "RXXX_GRASP_ANCHORED_ONLY",
        "transport_grounding": "DISABLED_FREE_MOTION",
        "visual_evidence": "RGB_ONLY_WITH_CAMERA_A_RXXX_OVERLAY",
        "preflight": "DISABLED",
        "controller_ik": "DISABLED",
        "free_motion_mode": True,
        "workspace_constraints_sent_to_claude": False,
        "viser_requested": bool(args.viser),
        "capture": {
            "fresh_camera_capture": not bool(args.reuse_latest_perception),
            "perception_config": str(args.perception_config),
            "move_to_perception": bool(args.move_to_perception),
        },
        "run_dir": str(session.run_dir),
        "output_dir": str(output),
        "perception_result": str(saved_path),
        "rounds_requested": int(args.rounds),
        "rounds": [],
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "workspace_prefilter.json", overlay_meta)

    for round_index in range(1, int(args.rounds) + 1):
        prompt = base_prompt
        previous_plans: list[dict[str, Any]] = []
        if plans:
            for item in plans[-6:]:
                proposal_record = item.get("proposal", {})
                if not isinstance(proposal_record, dict):
                    proposal_record = {}
                previous_plans.append(
                    {
                        "iteration": item["round"],
                        "selected_grasp": proposal_record.get("selected_grasp"),
                        "planned_part_evidence": {
                            "garment_observation": proposal_record.get(
                                "garment_observation", ""
                            ),
                            "fold_intent": proposal_record.get("reveal_strategy", ""),
                            "expected_observation": proposal_record.get(
                                "expected_observation", ""
                            ),
                        },
                        "trajectory": item["trajectory"],
                    }
                )
            prompt += (
                "\n\nPLAN HISTORY / COVERAGE LEDGER FOR THIS SAME SCENE:\n"
                "The previous entries are hypothetical plans only; no robot action "
                "was executed and the garment image is unchanged. Nevertheless, you "
                "must remember which garment part each iteration already planned. "
                "Use the selected-grasp rationale, garment observation, fold intent, "
                "and trajectory together to identify the previously targeted part. "
                "Do not silently repeat the same part when another visible part is "
                "available. If you intentionally revise or repeat a target, state why. "
                "Use the explicit sleeve -> body-side -> hem-up phase order above: "
                "the ledger is a coverage record, not permission to skip an earlier "
                "unfinished phase. "
                "Produce a complete new overall plan and explicit trajectory for this "
                "iteration, and make the new target/what changed explicit in the "
                "garment_observation and reveal_strategy fields.\n"
                + json.dumps(previous_plans, ensure_ascii=False, indent=2, default=str)
            )
        prompt_path = output / f"round_{round_index:02d}.prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        print(f"[planning-preview] round {round_index}/{args.rounds}: asking Claude", flush=True)
        response = planner.invoke(
            planning_images,
            prompt,
            session.run_dir,
            direct_prompt=True,
            preview_only=True,
            reference_mode=True,
        )
        raw_proposal = response.proposal
        proposal, anchor_diagnostic = _anchor_preview_proposal(
            raw_proposal,
            coordinate_samples,
        )
        record = {
            "round": round_index,
            "created_at": response.created_at,
            "prompt_path": str(prompt_path),
            "claude_command": list(response.command),
            "claude_metrics": _claude_runtime_metrics(response.stdout),
            "previous_plan_ledger": previous_plans,
            "selected_reference_anchor": anchor_diagnostic,
            "raw_proposal_before_rxxx_anchor": raw_proposal.as_dict(),
            "proposal": proposal.as_dict(),
            "trajectory": _trajectory(proposal),
            "raw_stdout": response.stdout,
            "raw_stderr": response.stderr,
        }
        _write_json(output / f"round_{round_index:02d}.json", record)
        plans.append(record)
        _write_json(output / "plans.json", plans)
        try:
            summary["alignment_report"] = str(write_alignment_report(output, saved_path))
        except Exception as exc:
            summary["alignment_report_error"] = f"{type(exc).__name__}: {exc}"
        # Keep the legacy PNG renderer available for non-Viser runs. With
        # --viser, the final visualization is rendered directly in the fused
        # point cloud instead of producing trajectory images.
        if not args.viser:
            try:
                trajectory_visuals = _render_trajectory_visuals(
                    plans=plans,
                    result=saved,
                    result_path=saved_path,
                    robot_config=session.robot_config,
                    output=output,
                )
                summary["trajectory_visuals"] = trajectory_visuals
            except Exception as exc:  # visualization must not invalidate a plan preview
                summary["trajectory_visuals_error"] = f"{type(exc).__name__}: {exc}"
        summary["rounds"].append(
            {
                "round": round_index,
                "selected_grasp": proposal.selected_grasp,
                "action_count": len(proposal.actions),
                "planned_part_evidence": {
                    "garment_observation": proposal.garment_observation,
                    "fold_intent": proposal.reveal_strategy,
                    "expected_observation": proposal.expected_observation,
                },
            }
        )
        _write_json(output / "summary.json", summary)
        print(f"[planning-preview] round {round_index} proposal:", flush=True)
        print(json.dumps(proposal.as_dict(), ensure_ascii=False, indent=2), flush=True)
        print("[planning-preview] trajectory:", flush=True)
        print(_format_plan(proposal), flush=True)

    summary["status"] = "COMPLETED"
    summary["completed_at"] = _now()
    if not args.viser:
        # Re-render once at the end to guarantee the final files contain every
        # completed round, even if an earlier incremental render was interrupted.
        try:
            summary["trajectory_visuals"] = _render_trajectory_visuals(
                plans=plans,
                result=saved,
                result_path=saved_path,
                robot_config=session.robot_config,
                output=output,
            )
        except Exception as exc:
            summary["trajectory_visuals_error"] = f"{type(exc).__name__}: {exc}"
    else:
        summary["trajectory_visualization"] = {
            "mode": "VISER_FUSED_POINT_CLOUD",
            "host": args.viser_host,
            "port": int(args.viser_port),
            "refresh_s": float(args.viser_refresh_s),
        }
    _write_json(output / "summary.json", summary)
    print(f"[planning-preview] saved={output}", flush=True)
    if args.viser:
        print(
            f"[planning-preview] starting Viser at http://{args.viser_host}:{args.viser_port} "
            "(Ctrl+C closes it; no robot command was sent)",
            flush=True,
        )
        run_planning_preview_viser(
            output,
            perception_result=saved_path,
            host=args.viser_host,
            port=args.viser_port,
            refresh_s=args.viser_refresh_s,
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--run-id")
    group.add_argument("--run-dir", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--perception-config",
        type=Path,
        default=Path("config/perception.free_exploration.json"),
        help="perception configuration used for a fresh Camera A/B capture",
    )
    parser.add_argument(
        "--reuse-latest-perception",
        action="store_true",
        help="skip camera capture and use the newest saved perception in the run",
    )
    parser.add_argument(
        "--move-to-perception",
        action="store_true",
        help="move the robot to its calibrated perception pose before capture",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--claude-timeout-s", type=int, default=900)
    parser.add_argument(
        "--viser",
        action="store_true",
        help="show the final plans in a read-only fused point-cloud Viser viewer",
    )
    parser.add_argument("--viser-host", default="127.0.0.1")
    parser.add_argument("--viser-port", type=int, default=8765)
    parser.add_argument("--viser-refresh-s", type=float, default=1.0)
    parser.add_argument(
        "--save-workspace-overlay",
        action="store_true",
        help="save the workspace diagnostic overlay locally; never send it to Claude",
    )
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    run_preview(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
