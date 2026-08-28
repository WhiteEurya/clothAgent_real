#!/usr/bin/env python3
"""Run one minimal Claude-selected sleeve grasp and vertical lift diagnostic.

Claude sees only an upright Camera-A RGB image plus a full-image Rxxx grid and
selects exactly one point on either requested sleeve cuff/free edge.  The
runtime then performs the remaining deterministic diagnostic stages:

* map the selected upright pixel back to the raw Camera-A image;
* compare it with the production garment mask without showing that mask to
  Claude;
* ground the pixel to calibrated base XYZ and expose any query/support shift;
* resolve grasp Z through the shared grasp-height policy;
* validate workspace and controller IK;
* execute only approach, close, and vertical lifts of 15/30/45 mm;
* capture raw Camera A/B RGB-D at every lift checkpoint;
* reverse to the original point, release, retreat, and Home.

There is no folding transport, Molmo call, reference image, autonomous retry,
or Claude hold evaluation.  Dry-run is the default.  Physical motion requires
both ``--enable-real`` and ``--confirm-real``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.config import ExperimentConfig, RobotConfig  # noqa: E402
from cloth_agent.garment_grounding_mcp import GarmentGrounding  # noqa: E402
from cloth_agent.grasp_height import resolve_grasp_height  # noqa: E402
from cloth_agent.perception import (  # noqa: E402
    PerceptionConfig,
    RGBDFrame,
    capture_two_view_rgbd,
)
from cloth_agent.robot_api import (  # noqa: E402
    move_robot_to_perception_position,
    validate_controller_trajectory,
)
from cloth_agent.session import AgentSession  # noqa: E402
from scripts.diagnose_claude_rgb_selection import (  # noqa: E402
    _invoke_claude,
    _latest_perception_result,
    _paths_from_perception_result,
    _randomized_markers,
    _render_points,
    _resolve_model_output,
    _rotate_image,
    _uniform_grid,
    _write_json,
)


class SleeveGraspDiagnosticError(RuntimeError):
    """Raised when a diagnostic gate prevents physical execution."""


LIFT_DELTAS_MM = (15.0, 30.0, 45.0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _task_instruction(side: str) -> str:
    if side == "image_left":
        side_text = "图像左侧"
    elif side == "image_right":
        side_text = "图像右侧"
    else:
        side_text = "图像左侧或右侧中视觉上更清楚、更适合夹取的一侧"
    return (
        f"请选择{side_text}短袖的袖口最外端附近、确实位于布料上的一个抓取点。"
        "目标必须属于袖子自由边缘或袖口布料，不要选择袖根、肩膀、衣身内部、标签、"
        "印花、桌面或衣服外的空白区域。只选择一个点。"
    )


def _upright_to_raw(
    pixel_xy: Sequence[int],
    *,
    raw_width: int,
    raw_height: int,
    rotation: str,
) -> list[int]:
    u_px, v_px = (int(pixel_xy[0]), int(pixel_xy[1]))
    if rotation == "none":
        x_px, y_px = u_px, v_px
    elif rotation == "clockwise90":
        x_px, y_px = v_px, raw_height - 1 - u_px
    elif rotation == "counterclockwise90":
        x_px, y_px = raw_width - 1 - v_px, u_px
    elif rotation == "180":
        x_px, y_px = raw_width - 1 - u_px, raw_height - 1 - v_px
    else:
        raise ValueError(f"unsupported rotation {rotation!r}")
    if not 0 <= x_px < raw_width or not 0 <= y_px < raw_height:
        raise SleeveGraspDiagnosticError(
            f"upright pixel {list(pixel_xy)} mapped outside raw image: [{x_px}, {y_px}]"
        )
    return [x_px, y_px]


def _raw_to_upright(
    pixel_xy: Sequence[int],
    *,
    raw_width: int,
    raw_height: int,
    rotation: str,
) -> list[int]:
    x_px, y_px = (int(pixel_xy[0]), int(pixel_xy[1]))
    if rotation == "none":
        u_px, v_px = x_px, y_px
    elif rotation == "clockwise90":
        u_px, v_px = raw_height - 1 - y_px, x_px
    elif rotation == "counterclockwise90":
        u_px, v_px = y_px, raw_width - 1 - x_px
    elif rotation == "180":
        u_px, v_px = raw_width - 1 - x_px, raw_height - 1 - y_px
    else:
        raise ValueError(f"unsupported rotation {rotation!r}")
    return [u_px, v_px]


def _mask_diagnostic(mask: np.ndarray, x_px: int, y_px: int, radius_px: int = 4) -> dict[str, Any]:
    array = np.asarray(mask, dtype=bool)
    if not 0 <= x_px < array.shape[1] or not 0 <= y_px < array.shape[0]:
        raise SleeveGraspDiagnosticError("selected raw pixel lies outside the production mask")
    x0, x1 = max(0, x_px - radius_px), min(array.shape[1], x_px + radius_px + 1)
    y0, y1 = max(0, y_px - radius_px), min(array.shape[0], y_px + radius_px + 1)
    patch = array[y0:y1, x0:x1]
    return {
        "exact_pixel_is_garment": bool(array[y_px, x_px]),
        "local_mask_fraction": float(np.count_nonzero(patch)) / float(max(1, patch.size)),
        "window_xyxy": [x0, y0, x1 - 1, y1 - 1],
    }


def _render_selected_upright(
    upright: Image.Image,
    marker: dict[str, Any],
    output: Path,
) -> Path:
    image = upright.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x_px, y_px = marker["pixel_xy"]
    draw.ellipse((x_px - 22, y_px - 22, x_px + 22, y_px + 22), outline=(255, 35, 35), width=8)
    draw.text(
        (x_px + 26, y_px - 14),
        f"CLAUDE {marker['reference_id']} ({marker['stable_id']})",
        fill=(255, 35, 35),
        font=font,
        stroke_width=2,
        stroke_fill=(255, 255, 255),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def _render_mask_check(
    raw: Image.Image,
    mask: np.ndarray,
    query_xy: Sequence[int],
    output: Path,
) -> Path:
    image = raw.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    rgba[np.asarray(mask, dtype=bool)] = (30, 220, 80, 70)
    overlay = Image.fromarray(rgba, mode="RGBA")
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x_px, y_px = (int(query_xy[0]), int(query_xy[1]))
    draw.ellipse((x_px - 18, y_px - 18, x_px + 18, y_px + 18), outline=(255, 25, 25, 255), width=6)
    draw.text(
        (x_px + 22, y_px - 12),
        "CLAUDE RAW PIXEL",
        fill=(255, 25, 25, 255),
        font=font,
        stroke_width=2,
        stroke_fill=(255, 255, 255, 255),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output)
    return output


def _render_grounding_shift(
    raw: Image.Image,
    query_xy: Sequence[int],
    support_xy: Sequence[int],
    output: Path,
) -> Path:
    image = raw.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    qx, qy = (int(query_xy[0]), int(query_xy[1]))
    sx, sy = (int(support_xy[0]), int(support_xy[1]))
    draw.line((qx, qy, sx, sy), fill=(255, 220, 0), width=7)
    draw.ellipse((qx - 18, qy - 18, qx + 18, qy + 18), outline=(255, 30, 30), width=6)
    draw.ellipse((sx - 18, sy - 18, sx + 18, sy + 18), outline=(30, 255, 80), width=6)
    draw.text((qx + 22, qy - 14), "QUERY", fill=(255, 30, 30), font=font)
    draw.text((sx + 22, sy + 4), "SUPPORT", fill=(30, 255, 80), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def _save_frames(frames: Sequence[RGBDFrame], output: Path, stage: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    for frame in frames:
        label = str(frame.label).upper()
        rgb_path = output / f"{stage}_camera_{label}_rgb.png"
        depth_path = output / f"{stage}_camera_{label}_depth_m.npy"
        rgb = np.asarray(frame.rgb, dtype=np.uint8)
        depth = np.asarray(frame.depth_m, dtype=np.float32)
        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_path, depth)
        luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        saved.append(
            {
                "camera": label,
                "rgb": str(rgb_path),
                "depth_m": str(depth_path),
                "mean_luma": float(np.mean(luma)),
                "p50_luma": float(np.percentile(luma, 50)),
                "valid_depth_fraction": float(np.count_nonzero(np.isfinite(depth) & (depth > 0)))
                / float(max(1, depth.size)),
            }
        )
    return {"stage": stage, "frames": saved}


def _saved_before_images(result_path: Path, output: Path) -> tuple[dict[str, Any], Path, Path | None]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    camera_a_path: Path | None = None
    mask_path: Path | None = None
    for view in payload.get("views", []):
        label = str(view.get("label", "")).upper()
        if label not in {"A", "B"}:
            continue
        source = result_path.parent / str(view["image"])
        destination = output / f"BEFORE_camera_{label}_rgb.png"
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            rgb.save(destination)
            array = np.asarray(rgb, dtype=np.uint8)
        luma = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
        records.append(
            {
                "camera": label,
                "rgb": str(destination),
                "mean_luma": float(np.mean(luma)),
                "p50_luma": float(np.percentile(luma, 50)),
            }
        )
        if label == "A":
            camera_a_path = source.resolve()
            raw_mask = view.get("garment_mask")
            if raw_mask:
                candidate = (result_path.parent / str(raw_mask)).resolve()
                mask_path = candidate if candidate.is_file() else None
    if camera_a_path is None:
        raise SleeveGraspDiagnosticError("saved perception result has no Camera A RGB")
    return {"stage": "BEFORE", "frames": records}, camera_a_path, mask_path


def _build_contact_sheet(
    before: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    output: Path,
) -> Path | None:
    stage_records = [before, *checkpoints]
    if not stage_records:
        return None
    labels = ("A", "B")
    cell_width, cell_height = 320, 200
    canvas = Image.new("RGB", (cell_width * len(stage_records), cell_height * len(labels)), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    any_image = False
    for column, stage in enumerate(stage_records):
        by_camera = {str(item["camera"]): item for item in stage.get("frames", [])}
        for row, label in enumerate(labels):
            item = by_camera.get(label)
            if item is None:
                continue
            path = Path(str(item["rgb"]))
            if not path.is_file():
                continue
            with Image.open(path) as image:
                tile = image.convert("RGB")
                tile.thumbnail((cell_width, cell_height - 22))
            x0 = column * cell_width + (cell_width - tile.width) // 2
            y0 = row * cell_height + 22 + (cell_height - 22 - tile.height) // 2
            canvas.paste(tile, (x0, y0))
            draw.text(
                (column * cell_width + 6, row * cell_height + 5),
                f"{stage.get('stage')} | Camera {label} | luma={item.get('mean_luma', 0):.1f}",
                fill=(255, 255, 255),
                font=font,
            )
            any_image = True
    if not any_image:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _experiment_source(
    *,
    x_mm: float,
    y_mm: float,
    grasp_z_mm: float,
    approach_z_mm: float,
    lift_zs_mm: Sequence[float],
    yaw_deg: float,
) -> str:
    lines = [
        "def run():",
        f"    move({x_mm!r}, {y_mm!r}, {approach_z_mm!r}, {yaw_deg!r})",
        "    open_gripper()",
        f"    move({x_mm!r}, {y_mm!r}, {grasp_z_mm!r}, {yaw_deg!r})",
        "    close_gripper()",
    ]
    lines.extend(
        f"    move({x_mm!r}, {y_mm!r}, {z_mm!r}, {yaw_deg!r})"
        for z_mm in lift_zs_mm
    )
    return "\n".join(lines) + "\n"


def _abort_actions(
    *,
    x_mm: float,
    y_mm: float,
    grasp_z_mm: float,
    approach_z_mm: float,
    yaw_deg: float,
) -> list[dict[str, Any]]:
    return [
        {"name": "move", "args": {"x": x_mm, "y": y_mm, "z": grasp_z_mm, "yaw": yaw_deg}},
        {"name": "open_gripper", "args": {}},
        {"name": "move", "args": {"x": x_mm, "y": y_mm, "z": approach_z_mm, "yaw": yaw_deg}},
        {"name": "home", "args": {}},
    ]


def _validate_pose_sequence(
    robot: RobotConfig,
    *,
    x_mm: float,
    y_mm: float,
    z_values: Sequence[float],
) -> None:
    for z_mm in z_values:
        robot.boundaries.validate(
            x_mm,
            y_mm,
            z_mm,
            robot.workspace_margin_mm,
            require_complete=True,
            z_lower_margin_mm=robot.lower_z_margin_mm,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--run-id")
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path("config/robot.example.json"),
        help=(
            "robot configuration JSON (default: config/robot.example.json; "
            "this enables the calibrated absolute-camera-depth mode)"
        ),
    )
    parser.add_argument(
        "--perception-config",
        type=Path,
        default=Path("config/perception.free_exploration.json"),
    )
    parser.add_argument("--perception-result", type=Path, help="dry-run RGB-D result; defaults to latest valid result")
    parser.add_argument("--side", choices=("either", "image_left", "image_right"), default="either")
    parser.add_argument("--rotation", choices=("none", "clockwise90", "counterclockwise90", "180"), default="clockwise90")
    parser.add_argument("--stride-px", type=int, default=64)
    parser.add_argument("--margin-px", type=int, default=24)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--max-support-shift-px", type=float, default=12.0)
    parser.add_argument("--settle-s", type=float, default=2.0)
    parser.add_argument("--checkpoint-settle-s", type=float, default=0.75)
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--claude-timeout-s", type=int, default=900)
    parser.add_argument("--enable-real", action="store_true")
    parser.add_argument("--confirm-real", action="store_true")
    parser.add_argument("--skip-controller-ik", action="store_true", help="dry-run only")
    return parser


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    root = args.project_root.expanduser().resolve()
    if args.enable_real != args.confirm_real:
        raise PermissionError("physical execution requires both --enable-real and --confirm-real")
    if args.enable_real and args.skip_controller_ik:
        raise PermissionError("--skip-controller-ik is forbidden with --enable-real")
    if args.settle_s < 0 or args.checkpoint_settle_s < 0:
        raise ValueError("settle intervals must be non-negative")
    if not 0 <= args.max_support_shift_px <= 50:
        raise ValueError("--max-support-shift-px must be between 0 and 50")
    robot_path = args.robot_config.expanduser().resolve() if args.robot_config else None
    robot = RobotConfig.load(root, robot_path)
    perception_path = args.perception_config
    if not perception_path.is_absolute():
        perception_path = root / perception_path
    perception_config = PerceptionConfig.load(root, perception_path.resolve())
    run_id = args.run_id or f"single_sleeve_grasp_{_stamp()}"
    session = AgentSession.create(
        root,
        "Claude RGB-only single sleeve grasp diagnostic",
        robot,
        ExperimentConfig(),
        run_id=run_id,
    )
    output = session.results / "single_sleeve_grasp" / _stamp()
    output.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "created_at": _now(),
        "status": "RUNNING",
        "run_dir": str(session.run_dir),
        "output_dir": str(output),
        "physical_execution": bool(args.enable_real),
        "task": "select one sleeve cuff/free-edge point, grasp, lift vertically at 15/30/45 mm, reverse-release",
        "claude_input": {
            "camera": "A",
            "rgb_only": True,
            "full_image_uniform_points": True,
            "depth": False,
            "mask": False,
            "xyz": False,
            "robot_bounds": False,
            "molmo": False,
            "reference": False,
        },
    }
    _write_json(output / "summary.json", summary)
    pre_position_home_required = False
    execution_started = False
    try:
        before_frames: dict[str, Any]
        perception_result_path: Path
        if args.enable_real:
            print("[single-sleeve] moving Home -> perception_position", flush=True)
            position = move_robot_to_perception_position(robot)
            summary["pre_perception_robot_positioning"] = position
            pre_position_home_required = True
            if args.settle_s:
                time.sleep(args.settle_s)
            print("[single-sleeve] capturing fresh Camera A/B RGB-D", flush=True)
            frames = capture_two_view_rgbd(perception_config)
            before_frames = _save_frames(frames, output / "before", "BEFORE")
            perception = session.locate_cloth_center(perception_config, frames=list(frames))
            summary["perception"] = perception
            candidates = list((session.results / "perception").glob("*/result.json"))
            if not candidates:
                raise SleeveGraspDiagnosticError("fresh perception completed without result.json")
            perception_result_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
            perception_dir = session.workspace / "perception_views"
            view_a = next(
                view for view in perception["views"] if str(view.get("label", "")).upper() == "A"
            )
            raw_camera_a_path = perception_dir / str(view_a["image"])
            mask_path = perception_dir / str(view_a["garment_mask"])
        else:
            perception_result_path = (
                args.perception_result.expanduser().resolve()
                if args.perception_result
                else _latest_perception_result(root)
            )
            before_frames, raw_camera_a_path, mask_path = _saved_before_images(
                perception_result_path,
                output / "before",
            )
            perception_dir = perception_result_path.parent
        summary["perception_result"] = str(perception_result_path)
        summary["before_frames"] = before_frames
        if mask_path is None or not mask_path.is_file():
            raise SleeveGraspDiagnosticError("production Camera-A garment mask is unavailable")

        with Image.open(raw_camera_a_path) as raw_image_handle:
            raw_image = raw_image_handle.convert("RGB")
        upright = _rotate_image(raw_image, args.rotation)
        upright_path = output / "camera_A_rgb_upright.png"
        upright.save(upright_path)
        candidates = _uniform_grid(upright.width, upright.height, args.stride_px, args.margin_px)
        markers, mapping = _randomized_markers(
            candidates,
            seed=20260827,
            key=f"single-sleeve:{args.side}:{upright.width}x{upright.height}",
        )
        call_dir = output / "claude_rgb_input"
        call_dir.mkdir(parents=True, exist_ok=False)
        upright.save(call_dir / "camera_A_rgb_upright.png")
        _render_points(
            upright,
            markers,
            call_dir / "camera_A_Rxxx_overlay.png",
            id_key="reference_id",
            title="Select one sleeve cuff/free-edge grasp point from RGB only",
        )
        task = {
            "kind": "localization",
            "id": "single_sleeve_cuff_grasp",
            "instruction": _task_instruction(args.side),
        }
        print(f"[single-sleeve] asking Claude; candidates={len(markers)}", flush=True)
        payload, stdout, stderr, duration_s = _invoke_claude(
            binary=args.claude_binary,
            call_dir=call_dir,
            task=task,
            marker_count=len(markers),
            timeout_s=args.claude_timeout_s,
        )
        resolved = _resolve_model_output(task, payload, mapping)
        marker = resolved["selected"]
        selected_upright = list(marker["pixel_xy"])
        selected_raw = _upright_to_raw(
            selected_upright,
            raw_width=raw_image.width,
            raw_height=raw_image.height,
            rotation=args.rotation,
        )
        selected_overlay = _render_selected_upright(
            upright,
            marker,
            output / "claude_selected_sleeve_point.png",
        )
        summary["claude_selection"] = {
            "duration_s": duration_s,
            "payload": payload,
            "raw_stdout": stdout,
            "raw_stderr": stderr,
            "selected_reference_id": marker["reference_id"],
            "stable_grid_id": marker["stable_id"],
            "upright_pixel_xy": selected_upright,
            "raw_camera_a_pixel_xy": selected_raw,
            "overlay": str(selected_overlay),
        }
        _write_json(output / "summary.json", summary)

        mask = np.load(mask_path, allow_pickle=False)
        mask_check = _mask_diagnostic(mask, selected_raw[0], selected_raw[1])
        mask_overlay = _render_mask_check(
            raw_image,
            mask,
            selected_raw,
            output / "selected_point_vs_production_mask.png",
        )
        mask_check["overlay"] = str(mask_overlay)
        summary["production_mask_check"] = mask_check
        if not mask_check["exact_pixel_is_garment"]:
            summary["status"] = "BLOCKED_SELECTION_OR_MASK_DISAGREEMENT"
            _write_json(output / "summary.json", summary)
            raise SleeveGraspDiagnosticError(
                "Claude-selected RGB marker is outside the production garment mask; "
                "inspect the RGB and mask overlay before assigning blame"
            )

        grounding = GarmentGrounding(perception_dir)
        measurement = grounding.sample_local_surface(
            "A",
            selected_raw[0],
            selected_raw[1],
            radius_px=3,
            include_nearest_reference=False,
        )
        if measurement.get("valid") is not True:
            raise SleeveGraspDiagnosticError(
                f"selected sleeve pixel has no valid local 3-D surface: {measurement.get('reason')}"
            )
        support_raw = [int(value) for value in measurement.get("support_pixel_xy", selected_raw)]
        support_shift_px = math.dist(selected_raw, support_raw)
        grounding_overlay = _render_grounding_shift(
            raw_image,
            selected_raw,
            support_raw,
            output / "grounding_query_to_support.png",
        )
        summary["grounding"] = {
            "measurement": measurement,
            "query_pixel_xy": selected_raw,
            "support_pixel_xy": support_raw,
            "support_shift_px": support_shift_px,
            "support_upright_pixel_xy": _raw_to_upright(
                support_raw,
                raw_width=raw_image.width,
                raw_height=raw_image.height,
                rotation=args.rotation,
            ),
            "overlay": str(grounding_overlay),
        }
        if support_shift_px > args.max_support_shift_px:
            summary["status"] = "BLOCKED_GROUNDING_SHIFT"
            _write_json(output / "summary.json", summary)
            raise SleeveGraspDiagnosticError(
                f"grounding moved {support_shift_px:.1f}px from Claude's semantic point; "
                f"limit is {args.max_support_shift_px:.1f}px"
            )

        grasp_height = resolve_grasp_height(
            measurement=measurement,
            table_plane_abc=None,
            robot_config=robot,
        )
        x_mm, y_mm, grasp_z_mm = grasp_height.target_xyz_mm
        lift_zs_mm = [grasp_z_mm + delta for delta in LIFT_DELTAS_MM]
        approach_z_mm = max(grasp_z_mm + 70.0, lift_zs_mm[-1] + 25.0)
        _validate_pose_sequence(
            robot,
            x_mm=x_mm,
            y_mm=y_mm,
            z_values=[grasp_z_mm, *lift_zs_mm, approach_z_mm],
        )
        source = _experiment_source(
            x_mm=x_mm,
            y_mm=y_mm,
            grasp_z_mm=grasp_z_mm,
            approach_z_mm=approach_z_mm,
            lift_zs_mm=lift_zs_mm,
            yaw_deg=args.yaw_deg,
        )
        source_name = "experiment_001_single_sleeve_grasp.py"
        source_path = session.workspace / source_name
        source_path.write_text(source, encoding="utf-8")
        abort_actions = _abort_actions(
            x_mm=x_mm,
            y_mm=y_mm,
            grasp_z_mm=grasp_z_mm,
            approach_z_mm=approach_z_mm,
            yaw_deg=args.yaw_deg,
        )
        preflight = session.runner.preflight(source_name)
        if preflight.error:
            raise SleeveGraspDiagnosticError(f"static preflight failed: {preflight.error}")
        summary["grasp_plan"] = {
            "yaw_deg": float(args.yaw_deg),
            "grasp_height": grasp_height.as_dict(),
            "approach_z_mm": approach_z_mm,
            "lift_deltas_mm": list(LIFT_DELTAS_MM),
            "lift_zs_mm": lift_zs_mm,
            "actions": preflight.actions,
            "checkpoint_action_indices": [4, 5, 6],
            "abort_actions": abort_actions,
            "source": str(source_path),
        }

        if not args.enable_real and not args.skip_controller_ik:
            full_abort_path = list(preflight.actions[:7]) + abort_actions
            summary["controller_ik"] = {
                "main": asdict(validate_controller_trajectory(robot, list(preflight.actions))),
                "reverse_release": asdict(validate_controller_trajectory(robot, full_abort_path)),
            }
        elif args.skip_controller_ik:
            summary["controller_ik"] = {"status": "SKIPPED_DRY_RUN"}
        _write_json(output / "summary.json", summary)

        checkpoint_records: list[dict[str, Any]] = []

        def capture_checkpoint(checkpoint_number: int) -> dict[str, Any]:
            delta_mm = LIFT_DELTAS_MM[checkpoint_number - 1]
            stage = f"LIFT_{int(delta_mm):02d}MM"
            if not args.enable_real:
                record = {
                    "stage": stage,
                    "status": "SIMULATED_NO_CAMERA_CAPTURE",
                    "frames": [],
                    "continue_transport": False,
                    "runtime_decision": "REVERSE_RELEASE",
                }
            else:
                if args.checkpoint_settle_s:
                    time.sleep(args.checkpoint_settle_s)
                frames = capture_two_view_rgbd(perception_config)
                record = _save_frames(frames, output / "checkpoints" / stage, stage)
                record.update(
                    {
                        "status": "CAPTURED",
                        "continue_transport": False,
                        "runtime_decision": "REVERSE_RELEASE",
                    }
                )
            checkpoint_records.append(record)
            _write_json(output / "checkpoints.json", checkpoint_records)
            return record

        execution_started = True
        execution = session.run_checkpointed_experiment(
            source_name,
            checkpoint_action_index=6,
            checkpoint_action_indices=(4, 5, 6),
            abort_actions=abort_actions,
            checkpoint_callback=capture_checkpoint,
            real=bool(args.enable_real),
            confirmed=bool(args.confirm_real),
            notes="One-shot Claude RGB-only sleeve grasp; vertical lift checkpoints; no transport.",
        )
        pre_position_home_required = False
        contact_sheet = _build_contact_sheet(
            before_frames,
            checkpoint_records,
            output / "before_and_lift_checkpoints.png",
        )
        summary.update(
            {
                "status": "PHYSICAL_PROBE_COMPLETED" if args.enable_real else "DRY_RUN_COMPLETED",
                "execution": execution,
                "checkpoints": checkpoint_records,
                "contact_sheet": str(contact_sheet) if contact_sheet else None,
                "manual_review": [
                    "Is the red Claude point actually on the distal sleeve cuff/free edge?",
                    "Does the green grounding support remain on the same intended sleeve structure?",
                    "At LIFT_15MM, is cloth visibly trapped between the jaws or only displaced on the table?",
                    "If cloth is initially acquired, at which checkpoint does it slip out?",
                    "If the selected point and grounding are correct but no cloth rises, classify the primary failure as mechanical acquisition.",
                ],
                "completed_at": _now(),
            }
        )
        _write_json(output / "summary.json", summary)
        return summary
    except BaseException as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        if summary.get("status") == "RUNNING":
            summary["status"] = "FAILED"
        if args.enable_real and pre_position_home_required and not execution_started:
            try:
                summary["preexecution_return_home"] = session._attempt_return_home(  # noqa: SLF001
                    notes="Return Home after single-sleeve diagnostic failed before rollout."
                )
            except BaseException as home_exc:
                summary["preexecution_return_home"] = {
                    "attempted": True,
                    "completed": False,
                    "error": f"{type(home_exc).__name__}: {home_exc}",
                }
        summary["completed_at"] = _now()
        _write_json(output / "summary.json", summary)
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_diagnostic(args)
    print(json.dumps(
        {
            "status": result["status"],
            "run_dir": result["run_dir"],
            "output_dir": result["output_dir"],
            "selected": result.get("claude_selection"),
            "contact_sheet": result.get("contact_sheet"),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
