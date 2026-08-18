"""Run live A/B perception through Claude planning and stop before execution.

This entry point deliberately has no physical-execution branch.  It captures
the configured RealSense views, runs the project's dense perception, invokes
the same Claude exploration planner used by the automatic loop, performs
static preflight and optional read-only controller IK, writes plan
visualizations, and exits without calling ``AgentSession.run_experiment``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.auto_exploration import ClaudeAutoClient
from cloth_agent.config import ExperimentConfig, RobotConfig
from cloth_agent.experiment import format_action_sequence
from cloth_agent.free_exploration import (
    _load_or_create_session,
    exploration_source,
    perception_image_paths,
    validate_exploration_payload,
)
from cloth_agent.kinematics import XArm7Kinematics
from cloth_agent.perception import PerceptionConfig, capture_two_view_rgbd
from cloth_agent.robot_api import validate_controller_trajectory
from cloth_agent.session import AgentSession
from cloth_agent.viewer import _load_latest_perception, path_waypoints_mm


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
    )


def _move_rows(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    move_index = 0
    gripper_state = "open/unknown"
    for action_index, action in enumerate(actions):
        name = action.get("name")
        if name == "close_gripper":
            gripper_state = "closed"
        elif name == "open_gripper":
            gripper_state = "open"
        elif name == "move":
            move_index += 1
            args = action.get("args", {})
            rows.append(
                {
                    "label": f"M{move_index}",
                    "action_index": action_index,
                    "xyz_mm": [float(args[key]) for key in ("x", "y", "z")],
                    "yaw_deg": float(args["yaw"]),
                    "gripper_before_move": gripper_state,
                }
            )
    return rows


def _project_base_xyz(view: dict[str, Any], xyz_mm: np.ndarray) -> np.ndarray | None:
    intrinsics = np.asarray(view["intrinsics"], dtype=np.float64)
    base_from_camera = np.asarray(view["X_base_camera"], dtype=np.float64)
    camera_from_base = np.linalg.inv(base_from_camera)
    homogeneous = np.concatenate([np.asarray(xyz_mm, dtype=np.float64) / 1000.0, [1.0]])
    camera = camera_from_base @ homogeneous
    if not np.all(np.isfinite(camera)) or camera[2] <= 0:
        return None
    return np.array(
        [
            intrinsics[0, 0] * camera[0] / camera[2] + intrinsics[0, 2],
            intrinsics[1, 1] * camera[1] / camera[2] + intrinsics[1, 2],
        ],
        dtype=np.float64,
    )


def _plan_colors(count: int) -> list[tuple[int, int, int]]:
    palette = [
        (230, 45, 45),
        (255, 145, 20),
        (40, 170, 255),
        (55, 205, 95),
        (180, 75, 245),
        (245, 75, 175),
        (20, 200, 200),
        (220, 210, 45),
    ]
    return [palette[index % len(palette)] for index in range(count)]


def _render_camera_plan(
    view: dict[str, Any],
    result_dir: Path,
    move_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    image = Image.open(result_dir / str(view["image"])).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = _plan_colors(len(move_rows))
    projected: list[tuple[float, float] | None] = []
    for row in move_rows:
        pixel = _project_base_xyz(view, np.asarray(row["xyz_mm"], dtype=np.float64))
        projected.append(None if pixel is None else (float(pixel[0]), float(pixel[1])))

    previous: tuple[float, float] | None = None
    for pixel in projected:
        if pixel is not None and previous is not None:
            draw.line((previous[0], previous[1], pixel[0], pixel[1]), fill=(255, 255, 255), width=5)
            draw.line((previous[0], previous[1], pixel[0], pixel[1]), fill=(30, 30, 30), width=2)
        if pixel is not None:
            previous = pixel
    label_font = _font(14)
    for row, pixel, color in zip(move_rows, projected, colors):
        if pixel is None:
            continue
        x, y = pixel
        if not (-30 <= x <= image.width + 30 and -30 <= y <= image.height + 30):
            continue
        radius = 12
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(0, 0, 0),
            outline=(255, 255, 255),
            width=4,
        )
        draw.ellipse(
            (x - radius + 3, y - radius + 3, x + radius - 3, y + radius - 3),
            outline=color,
            width=4,
        )
        draw.text((x + 15, y - 18), row["label"], fill=color, font=label_font)

    panel_width = 520
    canvas = Image.new("RGB", (image.width + panel_width, image.height), (24, 24, 24))
    canvas.paste(image, (0, 0))
    panel = ImageDraw.Draw(canvas)
    x0 = image.width + 16
    panel.text(
        (x0, 14),
        f"Camera {view['label']} | Claude plan projection",
        fill=(255, 255, 255),
        font=_font(18),
    )
    panel.text(
        (x0, 43),
        "PLAN PREVIEW ONLY - physical execution terminated",
        fill=(255, 195, 70),
        font=_font(13),
    )
    y = 78
    body = _font(12)
    for row, pixel, color in zip(move_rows, projected, colors):
        xyz = row["xyz_mm"]
        pixel_text = (
            "off-camera"
            if pixel is None
            else f"px=({pixel[0]:.1f},{pixel[1]:.1f})"
        )
        text = (
            f"{row['label']} {row['gripper_before_move']} | "
            f"xyz=({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f}) "
            f"yaw={row['yaw_deg']:.1f} | {pixel_text}"
        )
        panel.ellipse((x0, y + 2, x0 + 11, y + 13), fill=color)
        panel.text((x0 + 18, y), text, fill=(235, 235, 235), font=body)
        y += 34
    canvas.save(output_path)


def _load_fused_cloud(
    result: dict[str, Any], result_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    artifacts = result.get("depth_fusion", {}).get("artifacts", {})
    points_path = result_dir / str(artifacts.get("fused_points_base_mm", ""))
    colors_path = result_dir / str(artifacts.get("fused_colors_rgb", ""))
    if not points_path.is_file() or not colors_path.is_file():
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
    points = np.asarray(np.load(points_path), dtype=np.float64)
    colors = np.asarray(np.load(colors_path), dtype=np.uint8)
    valid = points.ndim == 2 and points.shape[1] == 3 and len(points) == len(colors)
    if not valid:
        raise ValueError("invalid fused point/color artifacts")
    if len(points) > 120_000:
        stride = int(np.ceil(len(points) / 120_000))
        points = points[::stride]
        colors = colors[::stride]
    return points, colors


def _render_topdown_plan(
    result: dict[str, Any],
    result_dir: Path,
    move_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    points, colors = _load_fused_cloud(result, result_dir)
    move_points = np.asarray([row["xyz_mm"] for row in move_rows], dtype=np.float64)
    xy_sources: list[np.ndarray] = []
    if len(points):
        xy_sources.append(points[:, :2])
    if len(move_points):
        xy_sources.append(move_points[:, :2])
    if not xy_sources:
        raise ValueError("plan preview has no fused points or move targets")
    xy = np.concatenate(xy_sources, axis=0)
    low = np.nanmin(xy, axis=0) - 45.0
    high = np.nanmax(xy, axis=0) + 45.0
    span = np.maximum(high - low, 1.0)

    width, height = 1220, 820
    plot_left, plot_top, plot_width, plot_height = 55, 55, 790, 710
    canvas_array = np.full((height, width, 3), 247, dtype=np.uint8)

    def pixels(xy_mm: np.ndarray) -> np.ndarray:
        px = plot_left + (xy_mm[:, 0] - low[0]) / span[0] * plot_width
        py = plot_top + (high[1] - xy_mm[:, 1]) / span[1] * plot_height
        return np.stack([px, py], axis=1)

    if len(points):
        point_pixels = np.rint(pixels(points[:, :2])).astype(int)
        valid = (
            (point_pixels[:, 0] >= plot_left)
            & (point_pixels[:, 0] < plot_left + plot_width)
            & (point_pixels[:, 1] >= plot_top)
            & (point_pixels[:, 1] < plot_top + plot_height)
        )
        point_pixels = point_pixels[valid]
        point_colors = colors[valid]
        canvas_array[point_pixels[:, 1], point_pixels[:, 0]] = point_colors

    canvas = Image.fromarray(canvas_array, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (plot_left, plot_top, plot_left + plot_width, plot_top + plot_height),
        outline=(25, 25, 25),
        width=2,
    )
    draw.text((plot_left, 16), "Fused garment cloud + Claude TCP plan (top-down base XY)", fill=(15, 15, 15), font=_font(18))
    colors_plan = _plan_colors(len(move_rows))
    if len(move_points):
        move_pixels = pixels(move_points[:, :2])
        for start, end in zip(move_pixels[:-1], move_pixels[1:]):
            draw.line((start[0], start[1], end[0], end[1]), fill=(255, 255, 255), width=8)
            draw.line((start[0], start[1], end[0], end[1]), fill=(30, 30, 30), width=3)
        for row, pixel, color in zip(move_rows, move_pixels, colors_plan):
            x, y = pixel
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color, outline=(0, 0, 0), width=2)
            draw.text((x + 13, y - 17), row["label"], fill=(0, 0, 0), font=_font(14))

    panel_x = 875
    draw.text((panel_x, 24), "Validated plan preview", fill=(20, 20, 20), font=_font(19))
    draw.text((panel_x, 56), "NO PHYSICAL EXECUTION", fill=(205, 45, 35), font=_font(15))
    y = 100
    for row, color in zip(move_rows, colors_plan):
        xyz = row["xyz_mm"]
        draw.ellipse((panel_x, y + 2, panel_x + 12, y + 14), fill=color)
        draw.text(
            (panel_x + 20, y),
            (
                f"{row['label']} {row['gripper_before_move']}\n"
                f"  x={xyz[0]:.1f} y={xyz[1]:.1f} z={xyz[2]:.1f}\n"
                f"  yaw={row['yaw_deg']:.1f} deg"
            ),
            fill=(35, 35, 35),
            font=_font(12),
        )
        y += 68
    canvas.save(output_path)


def _combine_images(paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    canvas.save(output_path)


def _render_raw_depth(
    depth_path: Path,
    output_path: Path,
    *,
    camera_label: str,
    min_depth_m: float,
    max_depth_m: float,
) -> None:
    depth = np.asarray(np.load(depth_path), dtype=np.float64)
    valid = (
        np.isfinite(depth)
        & (depth > float(min_depth_m))
        & (depth < float(max_depth_m))
    )
    normalized = np.zeros(depth.shape, dtype=np.float64)
    normalized[valid] = np.clip(
        (max_depth_m - depth[valid]) / (max_depth_m - min_depth_m),
        0.0,
        1.0,
    )
    red = normalized
    green = 1.0 - np.abs(2.0 * normalized - 1.0)
    blue = 1.0 - normalized
    rgb = np.rint(np.stack([red, green, blue], axis=2) * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    image = Image.fromarray(rgb, mode="RGB")
    panel_width = 360
    canvas = Image.new("RGB", (image.width + panel_width, image.height), (24, 24, 24))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    x0 = image.width + 18
    draw.text(
        (x0, 18),
        f"Camera {camera_label} raw depth",
        fill=(255, 255, 255),
        font=_font(19),
    )
    draw.text(
        (x0, 54),
        "near = red | far = blue | invalid = black",
        fill=(230, 230, 230),
        font=_font(12),
    )
    draw.text(
        (x0, 85),
        f"display range: {min_depth_m:.3f} to {max_depth_m:.3f} m",
        fill=(230, 230, 230),
        font=_font(12),
    )
    if np.any(valid):
        values = depth[valid]
        draw.text(
            (x0, 116),
            (
                f"valid fraction: {float(np.mean(valid)):.4f}\n"
                f"p01/p50/p99: {np.quantile(values, 0.01):.3f} / "
                f"{np.quantile(values, 0.50):.3f} / {np.quantile(values, 0.99):.3f} m"
            ),
            fill=(230, 230, 230),
            font=_font(12),
            spacing=8,
        )
    canvas.save(output_path)


def _planning_images_with_raw_depth(
    result: dict[str, Any],
    result_path: Path,
    config: PerceptionConfig,
    preview_dir: Path,
) -> list[Path]:
    images = perception_image_paths(result, result_path)
    manifest: list[dict[str, str]] = [
        {"kind": "pipeline_visual", "path": str(path)} for path in images
    ]
    for view in result.get("views", []):
        if not isinstance(view, dict) or view.get("label") not in {"A", "B"}:
            continue
        raw_depth = result_path.parent / str(view.get("depth_m", ""))
        if not raw_depth.is_file():
            continue
        output = result_path.parent / f"camera_{view['label']}_raw_depth_visualization.png"
        _render_raw_depth(
            raw_depth,
            output,
            camera_label=str(view["label"]),
            min_depth_m=config.min_depth_m,
            max_depth_m=config.max_depth_m,
        )
        resolved = output.resolve()
        if resolved not in images:
            images.append(resolved)
        manifest.append(
            {
                "kind": "raw_depth_visualization",
                "camera": str(view["label"]),
                "source_npy": str(raw_depth.resolve()),
                "path": str(resolved),
            }
        )
    _write_json(
        preview_dir / "claude_planning_input_images.json",
        {
            "count": len(images),
            "images_in_exact_claude_input_order": [str(path) for path in images],
            "manifest": manifest,
        },
    )
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--run-dir",
        help="resume an existing preview run instead of capturing a new observation",
    )
    parser.add_argument(
        "--reuse-latest-plan",
        action="store_true",
        help="reuse the latest saved successful Claude proposal in --run-dir",
    )
    parser.add_argument("--robot-config")
    parser.add_argument(
        "--perception-config", default="config/perception.free_exploration.json"
    )
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--claude-timeout-s", type=int, default=400)
    parser.add_argument(
        "--objective",
        default=(
            "Inspect the current garment using every supplied Camera A/B RGB image, "
            "raw-depth visualization, per-camera height/edge/coordinate image, and "
            "fused height image. Plan one cautious grasp-based action that tests or "
            "establishes a useful lifting anchor."
        ),
    )
    parser.add_argument(
        "--skip-controller-ik",
        action="store_true",
        help="skip the read-only live-controller IK gate; physical execution remains absent",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    if args.run_dir and args.run_id:
        raise ValueError("use either --run-dir or --run-id, not both")
    if args.reuse_latest_plan and not args.run_dir:
        raise ValueError("--reuse-latest-plan requires --run-dir")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"claude_plan_preview_{stamp}"
    robot_config_path = Path(args.robot_config).resolve() if args.robot_config else None
    perception_path = Path(args.perception_config)
    if not perception_path.is_absolute():
        perception_path = root / perception_path

    robot = RobotConfig.load(root, robot_config_path)
    config = PerceptionConfig.load(root, perception_path)
    if args.run_dir:
        session = _load_or_create_session(
            root,
            Path(args.run_dir).resolve(),
            None,
            robot_config_path,
        )
        robot = session.robot_config
    else:
        session = AgentSession.create(
            root,
            "Claude plan preview only; terminate before physical execution",
            robot,
            ExperimentConfig(),
            run_id=run_id,
        )
    preview_dir = session.results / "claude_plan_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(session.run_dir),
        "objective": args.objective,
        "physical_execution_authority": False,
        "physical_commands_sent": False,
        "execution_status": "terminated_before_execution",
        "phases": {},
    }

    try:
        if args.run_dir:
            saved, result_path = _load_latest_perception(session)
            if saved is None or result_path is None:
                raise RuntimeError("resume run has no saved perception result")
            summary["phases"]["capture"] = {
                "status": "reused",
                "cameras": saved.get("active_cameras", []),
            }
            summary["phases"]["perception"] = {
                "status": "reused",
                "result_path": str(result_path),
                "center_base_mm": saved.get("center_base_mm"),
                "active_cameras": saved.get("active_cameras"),
            }
        else:
            frames = capture_two_view_rgbd(config)
            summary["phases"]["capture"] = {
                "status": "completed",
                "cameras": [frame.label for frame in frames],
            }
            perception = session.locate_cloth_center(config, frames=frames)
            saved, result_path = _load_latest_perception(session)
            if saved is None or result_path is None:
                raise RuntimeError("perception completed without a saved result")
            summary["phases"]["perception"] = {
                "status": "completed",
                "result_path": str(result_path),
                "center_base_mm": perception.get("center_base_mm"),
                "active_cameras": perception.get("active_cameras"),
            }

        if args.reuse_latest_plan:
            logs = sorted(
                (session.results / "claude_exploration").glob("*.json")
            )
            successful_logs = [
                path
                for path in logs
                if not path.name.endswith("_failed.json")
            ]
            if not successful_logs:
                raise RuntimeError("resume run has no successful Claude proposal log")
            raw_plan_payload = json.loads(
                successful_logs[-1].read_text(encoding="utf-8")
            )
            proposal = validate_exploration_payload(raw_plan_payload.get("proposal"))
            plan_status = "reused"
        else:
            images = _planning_images_with_raw_depth(
                saved, result_path, config, preview_dir
            )
            summary["phases"]["planning_inputs"] = {
                "status": "completed",
                "image_count": len(images),
                "manifest": str(
                    preview_dir / "claude_planning_input_images.json"
                ),
                "raw_depth_visualizations": [
                    str(path)
                    for path in images
                    if path.name.endswith("_raw_depth_visualization.png")
                ],
            }
            client = ClaudeAutoClient(
                binary=args.claude_binary, timeout_s=args.claude_timeout_s
            )
            proposal = client.plan(images, session, args.objective, history=[])
            raw_plan = client.last_plan_result
            if raw_plan is None:
                raise RuntimeError("Claude planning completed without a raw result")
            raw_plan_payload = asdict(raw_plan)
            plan_status = "completed"
        source = exploration_source(proposal)
        experiment_path = session.workspace / "experiment_001_claude_plan_preview.py"
        experiment_path.write_text(source, encoding="utf-8")
        _write_json(preview_dir / "claude_plan.json", raw_plan_payload)
        _write_json(preview_dir / "proposal.json", proposal.as_dict())
        summary["phases"]["claude_plan"] = {
            "status": plan_status,
            "proposal": proposal.as_dict(),
            "experiment_source": str(experiment_path),
        }

        preflight = session.runner.preflight(experiment_path.name)
        summary["phases"]["preflight"] = {
            "status": "failed" if preflight.error else "completed",
            "error": preflight.error,
            "actions": preflight.actions,
            "formatted_actions": format_action_sequence(preflight.actions),
        }
        if preflight.error:
            raise RuntimeError(f"static preflight rejected Claude plan: {preflight.error}")

        controller = None
        controller_error = None
        if not args.skip_controller_ik:
            try:
                controller = validate_controller_trajectory(robot, preflight.actions)
            except Exception as exc:
                controller_error = f"{type(exc).__name__}: {exc}"
        summary["phases"]["controller_ik"] = {
            "status": (
                "skipped"
                if args.skip_controller_ik
                else "failed"
                if controller_error
                else "completed"
            ),
            "error": controller_error,
            "result": controller,
            "read_only": True,
        }

        animation_error = None
        animation_frames = []
        try:
            kinematics = XArm7Kinematics(
                root / "assets" / "robots" / "xarm7" / "xarm7.urdf"
            )
            animation_frames = kinematics.build_animation(
                preflight.actions,
                robot.init_joints_deg,
                robot.orientation_roll_deg,
                robot.orientation_pitch_deg,
                joint_targets_rad=(
                    controller.joint_targets_rad if controller is not None else None
                ),
            )
            np.savez_compressed(
                preview_dir / "animation_frames.npz",
                configurations_rad=np.stack(
                    [frame.configuration_rad for frame in animation_frames], axis=0
                ),
                action_indices=np.asarray(
                    [frame.action_index for frame in animation_frames], dtype=np.int32
                ),
                labels=np.asarray([frame.label for frame in animation_frames]),
            )
        except Exception as exc:
            animation_error = f"{type(exc).__name__}: {exc}"
        summary["phases"]["animation"] = {
            "status": "failed" if animation_error else "completed",
            "error": animation_error,
            "frame_count": len(animation_frames),
            "artifact": "animation_frames.npz" if animation_frames else None,
        }

        move_rows = _move_rows(preflight.actions)
        topdown_path = preview_dir / "claude_plan_topdown.png"
        _render_topdown_plan(saved, result_path.parent, move_rows, topdown_path)
        camera_paths: list[Path] = []
        for view in saved.get("views", []):
            if not isinstance(view, dict) or view.get("label") not in {"A", "B"}:
                continue
            output_path = preview_dir / f"claude_plan_camera_{view['label']}.png"
            _render_camera_plan(view, result_path.parent, move_rows, output_path)
            camera_paths.append(output_path)
        combined_path = preview_dir / "claude_plan_visualization.png"
        _combine_images([topdown_path, *camera_paths], combined_path)
        summary["phases"]["visualization"] = {
            "status": "completed",
            "topdown": str(topdown_path),
            "camera_overlays": [str(path) for path in camera_paths],
            "combined": str(combined_path),
        }
    except BaseException as exc:
        summary["failure"] = f"{type(exc).__name__}: {exc}"
        _write_json(preview_dir / "plan_preview_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
        raise

    _write_json(preview_dir / "plan_preview_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
