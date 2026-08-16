"""End-to-end Viser console for perception, planning, animation, and execution."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import ExperimentConfig
from .experiment import Preflight, format_action_sequence, format_speed_profile
from .kinematics import AnimationFrame, XArm7Kinematics
from .perception import (
    PerceptionConfig,
    RGBDFrame,
    capture_two_view_rgbd,
)
from .robot_api import ControllerTrajectoryValidation, validate_controller_trajectory
from .randomization import (
    GarmentRandomizationPlan,
    build_garment_randomization_plan,
    garment_randomization_source,
)
from .session import AgentSession


def canonical_grasp_source(config: ExperimentConfig) -> str:
    """Return an explicitly manual center-grasp program for complete configs.

    Dense perception does not call this helper.  It remains available for a
    user-supplied/manual `ExperimentConfig` and is not a Claude waypoint plan.
    """

    x, y, grasp_z, approach_z, lift_z, yaw = config.require_ready()
    return (
        "def run():\n"
        "    home()\n"
        "    open_gripper()\n"
        f"    move({x!r}, {y!r}, {approach_z!r}, {yaw!r})\n"
        f"    move({x!r}, {y!r}, {grasp_z!r}, {yaw!r})\n"
        "    close_gripper()\n"
        f"    move({x!r}, {y!r}, {lift_z!r}, {yaw!r})\n"
        f"    move({x!r}, {y!r}, {approach_z!r}, {yaw!r})\n"
        "    open_gripper()\n"
        "    home()\n"
    )


def canonical_home_source() -> str:
    """Return the single-action program used by the Viser Init button."""

    return "def run():\n    home()\n"


def path_waypoints_mm(
    actions: list[dict[str, Any]], home_pose_mm_deg: tuple[float, ...]
) -> list[tuple[str, np.ndarray]]:
    """Extract the commanded TCP path from a preflight action trace."""

    home = np.asarray(home_pose_mm_deg[:3], dtype=np.float64)
    waypoints: list[tuple[str, np.ndarray]] = []
    move_index = 0
    for action in actions:
        if action.get("name") == "home":
            waypoints.append(("home", home.copy()))
        elif action.get("name") == "move":
            args = action.get("args", {})
            point = np.asarray([args["x"], args["y"], args["z"]], dtype=np.float64)
            move_index += 1
            waypoints.append((f"move_{move_index}", point))
    return waypoints


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _load_latest_perception(
    session: AgentSession,
) -> tuple[dict[str, Any] | None, Path | None]:
    metadata = json.loads((session.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    result_paths = metadata.get("perception_results", [])
    if not result_paths:
        return None, None
    result_path = (session.run_dir / result_paths[-1]).resolve()
    if session.run_dir not in result_path.parents or not result_path.is_file():
        raise PermissionError("saved perception result must stay inside the current run")
    return json.loads(result_path.read_text(encoding="utf-8")), result_path


def _frame_point_cloud(frame: RGBDFrame, stride: int = 5) -> tuple[np.ndarray, np.ndarray]:
    depth = frame.depth_m
    K = frame.intrinsics
    y_px, x_px = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[::stride, ::stride].astype(np.float64)
    valid = np.isfinite(z) & (z > 0.15) & (z < 2.0)
    x_px, y_px, z = x_px[valid], y_px[valid], z[valid]
    camera_points = np.stack(
        [
            (x_px - K[0, 2]) * z / K[0, 0],
            (y_px - K[1, 2]) * z / K[1, 1],
            z,
        ],
        axis=1,
    )
    points = camera_points @ frame.X_base_camera[:3, :3].T + frame.X_base_camera[:3, 3]
    colors = frame.rgb[::stride, ::stride][valid]
    return points.astype(np.float32), colors.astype(np.uint8)


def _view_point_cloud(
    view: dict[str, Any], result_dir: Path, stride: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    image_path = result_dir / view["image"]
    depth_path = result_dir / f"{Path(view['image']).stem}_depth_m.npy"
    frame = RGBDFrame(
        label=str(view["label"]),
        serial=str(view["serial"]),
        rgb=np.asarray(Image.open(image_path).convert("RGB")),
        depth_m=np.load(depth_path),
        intrinsics=np.asarray(view["intrinsics"], dtype=np.float64),
        X_base_camera=np.asarray(view["X_base_camera"], dtype=np.float64),
    )
    return _frame_point_cloud(frame, stride=stride)


def _latest_experiment(session: AgentSession, requested: str | None) -> str | None:
    if requested:
        name = session._validate_experiment_name(requested)
        if not (session.workspace / name).is_file():
            raise FileNotFoundError(session.workspace / name)
        return name
    existing = sorted(session.workspace.glob("experiment_*.py"))
    return existing[-1].name if existing else None


def _ensure_experiment_source(
    session: AgentSession,
    selected: str | None,
    expected_source: str,
) -> str:
    if selected is not None:
        selected_path = session.workspace / selected
        if selected_path.read_text(encoding="utf-8") == expected_source:
            return selected
    name = session._next_experiment_name()
    (session.workspace / name).write_text(expected_source, encoding="utf-8")
    return name


def _ensure_experiment(session: AgentSession, selected: str | None) -> str:
    return _ensure_experiment_source(
        session,
        selected,
        canonical_grasp_source(session.experiment_config),
    )


def _preflight_markdown(preflight: Preflight) -> str:
    source = preflight.source.replace("```", "` ` `")
    error = f"\n\n**Preflight error:** `{preflight.error}`" if preflight.error else ""
    return (
        f"### Experiment source\n\n```python\n{source}\n```\n\n"
        f"```text\n{format_action_sequence(preflight.actions)}\n```{error}"
    )


@dataclass
class _DashboardState:
    experiment_name: str | None = None
    captured_frames: list[RGBDFrame] | None = None
    perception_config: PerceptionConfig | None = None
    perception_result: dict[str, Any] | None = None
    perception_result_path: Path | None = None
    preflight: Preflight | None = None
    controller_validation: ControllerTrajectoryValidation | None = None
    animation_frames: list[AnimationFrame] = field(default_factory=list)
    approved_source_hash: str | None = None
    planned_source: str | None = None
    plan_kind: str = "center_grasp"
    randomization_plan: GarmentRandomizationPlan | None = None
    busy: bool = False
    animation_playing: bool = False


def run_viewer(
    session: AgentSession,
    *,
    experiment: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    enable_real: bool = False,
    perception_config_path: Path | None = None,
    urdf_path: Path | None = None,
) -> int:
    """Start the complete local Viser cloth-grasp workflow."""

    if enable_real and host not in {"127.0.0.1", "localhost", "::1"}:
        raise PermissionError("physical execution is allowed only on a loopback-only Viser server")
    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError as exc:
        raise RuntimeError(
            "Viser with URDF support is required in the current Python environment; "
            "install it with: python -m pip install 'viser[urdf]>=1.0,<2'"
        ) from exc

    root = session.project_root
    perception_path = (
        perception_config_path or root / "config" / "perception.example.json"
    ).expanduser().resolve()
    robot_urdf_path = (
        urdf_path or root / "assets" / "robots" / "xarm7" / "xarm7.urdf"
    ).expanduser().resolve()
    selected = _latest_experiment(session, experiment)
    if selected is not None and experiment is None:
        try:
            expected_source = canonical_grasp_source(session.experiment_config)
            if (session.workspace / selected).read_text(encoding="utf-8") != expected_source:
                selected = None
        except BaseException:
            selected = None
    latest_perception, latest_perception_path = _load_latest_perception(session)
    state = _DashboardState(
        experiment_name=selected,
        perception_result=latest_perception,
        perception_result_path=latest_perception_path,
    )
    operation_lock = threading.Lock()
    animation_lock = threading.Lock()

    server = viser.ViserServer(host=host, port=port, label="Cloth grasp console")
    server.scene.set_up_direction("+z")
    robot = session.robot_config
    bounds = robot.boundaries
    grid_x = ((bounds.x_min or 0.0) + (bounds.x_max or 900.0)) / 2000.0
    grid_y = ((bounds.y_min or -400.0) + (bounds.y_max or 400.0)) / 2000.0
    server.scene.add_grid(
        "/workspace/table",
        width=1.2,
        height=0.8,
        cell_size=0.05,
        section_size=0.25,
        position=(grid_x, grid_y, 0.0),
    )
    server.scene.add_frame("/robot_base", axes_length=0.15, axes_radius=0.006)
    server.scene.add_frame("/xarm", show_axes=False)
    robot_model = ViserUrdf(
        server,
        robot_urdf_path,
        root_node_name="/xarm",
        load_meshes=True,
        load_collision_meshes=False,
    )
    kinematics = XArm7Kinematics(robot_urdf_path)
    home_cfg = np.concatenate(
        [np.radians(np.asarray(robot.init_joints_deg, dtype=np.float64)), [0.0]]
    )
    robot_model.update_cfg(home_cfg)

    status = server.gui.add_markdown(
        "### Ready\n\nChoose cameras and capture RGB-D. No robot command has been sent."
    )
    workflow = server.gui.add_markdown(
        "### Workflow\n\n"
        "0. Optionally use Init to return the physical arm to its configured Home\n"
        "1. Capture RealSense RGB-D\n"
        "2. Fuse calibrated A/B RGB-D into one base-frame cloud\n"
        "3. Review the A/B garment height-above-table heatmaps and fused garment boundary; ask Claude free/automatic exploration to choose every motion waypoint\n"
        "4. Review the Claude path, then run static preflight + controller IK + URDF animation\n"
        "5. Click the red confirmation button to execute one physical rollout"
    )
    init_contract = server.gui.add_markdown(
        "### Robot Init / Home\n\n"
        f"- target TCP: `({robot.init_pose_mm_deg[0]:.1f}, "
        f"{robot.init_pose_mm_deg[1]:.1f}, {robot.init_pose_mm_deg[2]:.1f}) mm`\n"
        f"- joint speed / acceleration: `{robot.home_speed_deg_s:.1f} deg/s / "
        f"{robot.home_acceleration_deg_s2:.1f} deg/s²`\n"
        "- action: `home()` only; clicking the red button moves immediately"
    )
    init_button = server.gui.add_button(
        "Init: Return arm to Home", disabled=not enable_real, color="red"
    )
    camera_mode = server.gui.add_dropdown(
        "Camera mode",
        options=("Camera A + B dense RGB-D fusion",),
        initial_value="Camera A + B dense RGB-D fusion",
    )
    capture_button = server.gui.add_button("1. Capture RealSense RGB-D", color="blue")
    fusion_button = server.gui.add_button(
        "2. Fuse A + B depth + calculate center", disabled=True, color="blue"
    )
    plan_summary = server.gui.add_markdown("### Plan\n\nNo validated garment center yet.")
    randomization_summary = server.gui.add_markdown(
        "### Alternative/manual paths\n\nDense perception supplies only an observation. Use Claude free/automatic exploration to choose motion waypoints; the buttons below are available only for a complete manually supplied plan."
    )
    randomize_button = server.gui.add_button(
        "Generate manual randomization path", disabled=True, color="blue"
    )
    standard_grasp_button = server.gui.add_button(
        "Use manual center-grasp path", disabled=True
    )
    validate_button = server.gui.add_button(
        "3. Validate plan + build robot animation", disabled=True, color="blue"
    )
    validation_details = server.gui.add_markdown("No plan validation has run yet.")

    animation_slider = server.gui.add_slider(
        "Animation frame", min=0, max=1, step=1, initial_value=0, disabled=True
    )
    animation_state = server.gui.add_markdown("Animation is not available yet.")
    play_button = server.gui.add_button("4. Play arm + gripper", disabled=True, color="green")
    pause_button = server.gui.add_button("Pause animation", disabled=True)
    reset_button = server.gui.add_button("Reset animation", disabled=True)
    loop_animation = server.gui.add_checkbox("Loop animation", initial_value=False)

    execution_contract = server.gui.add_markdown(
        "### Physical execution\n\nThe red button is the explicit confirmation. It remains locked until perception, static validation, controller IK, and animation generation pass."
    )
    execute_button = server.gui.add_button(
        "5. Confirm and execute one physical rollout", disabled=True, color="red"
    )
    if not enable_real:
        server.gui.add_markdown(
            "Physical execution is disabled for this server. Restart with `--enable-real` after previewing."
        )

    image_handles: list[Any] = []

    def metadata() -> dict[str, Any]:
        return json.loads((session.run_dir / "run_metadata.json").read_text(encoding="utf-8"))

    def expected_plan_source() -> str:
        return state.planned_source or canonical_grasp_source(session.experiment_config)

    def randomization_markdown(plan: GarmentRandomizationPlan) -> str:
        rows = [
            "| phase | x mm | y mm | z mm | yaw deg |",
            "|---|---:|---:|---:|---:|",
        ]
        rows.extend(
            f"| {name} | {x:.2f} | {y:.2f} | {z:.2f} | {yaw:.2f} |"
            for name, x, y, z, yaw in plan.waypoint_rows()
        )
        return (
            "### Garment randomization path\n\n"
            f"- seed: `{plan.seed}`\n"
            "- policy: `grasp → lift → inward drag → ±90° wrist twist → low-air release`\n\n"
            + "\n".join(rows)
        )

    def clear_gui_images() -> None:
        while image_handles:
            image_handles.pop().remove()

    def render_captured_frames(frames: list[RGBDFrame]) -> None:
        server.scene.remove_by_name("/perception")
        clear_gui_images()
        for frame in frames:
            points, colors = _frame_point_cloud(frame)
            server.scene.add_point_cloud(
                f"/perception/{frame.label}_rgbd",
                points=points,
                colors=colors,
                point_size=0.004,
                point_shape="circle",
            )
            image_handles.append(
                server.gui.add_image(frame.rgb, label=f"RealSense {frame.label}")
            )

    def render_perception_result(result: dict[str, Any], result_path: Path) -> None:
        server.scene.remove_by_name("/perception")
        clear_gui_images()
        for index, view in enumerate(result.get("views", [])):
            points, colors = _view_point_cloud(view, result_path.parent)
            server.scene.add_point_cloud(
                f"/perception/{view['label']}_rgbd",
                points=points,
                colors=colors,
                point_size=0.004,
                point_shape="circle",
            )
            annotated = result_path.parent / view.get(
                "annotated_image", f"camera_{index}_{view['label']}_annotated.png"
            )
            image_path = annotated if annotated.is_file() else result_path.parent / view["image"]
            role = view.get("role")
            image_label = (
                f"Camera {view['label']} dense RGB-D fusion source"
                if role == "rgbd_fusion_source"
                else f"Camera {view['label']} perception"
            )
            image_handles.append(
                server.gui.add_image(
                    np.asarray(Image.open(image_path).convert("RGB")),
                    label=image_label,
                )
            )
            heatmap_path = result_path.parent / str(
                view.get(
                    "height_map_boundary",
                    view.get(
                        "height_map",
                        view.get("depth_heatmap_boundary", view.get("depth_heatmap", "")),
                    ),
                )
            )
            if heatmap_path.is_file():
                image_handles.append(
                    server.gui.add_image(
                        np.asarray(Image.open(heatmap_path).convert("RGB")),
                        label=f"Camera {view['label']} garment height-above-table heatmap + boundary",
                    )
                )
            fold_edge_path = result_path.parent / str(
                view.get(
                    "height_gradient_overlay",
                    view.get("fold_edge_overlay", ""),
                )
            )
            if fold_edge_path.is_file():
                image_handles.append(
                    server.gui.add_image(
                        np.asarray(Image.open(fold_edge_path).convert("RGB")),
                        label=f"Camera {view['label']} internal height-gradient/occlusion edges",
                    )
                )
            coordinate_overlay = result_path.parent / str(
                view.get("coordinate_overlay", "")
            )
            if coordinate_overlay.is_file():
                image_handles.append(
                    server.gui.add_image(
                        np.asarray(Image.open(coordinate_overlay).convert("RGB")),
                        label=(
                            f"Camera {view['label']} unranked robot-base coordinate references"
                        ),
                    )
                )
        artifacts = result.get("depth_fusion", {}).get("artifacts", {})
        fused_points_path = result_path.parent / str(artifacts.get("fused_points_base_mm", ""))
        fused_colors_path = result_path.parent / str(artifacts.get("fused_colors_rgb", ""))
        if fused_points_path.is_file() and fused_colors_path.is_file():
            fused_points = np.load(fused_points_path).astype(np.float32) / 1000.0
            fused_colors = np.load(fused_colors_path).astype(np.uint8)
            server.scene.add_point_cloud(
                "/perception/fused_AB",
                points=fused_points,
                colors=fused_colors,
                point_size=0.004,
                point_shape="circle",
            )
        for key, label in (
            ("heatmap", "Fused garment height-above-table heatmap"),
            ("boundary_overlay", "Fused height map + garment boundary"),
            ("fold_edge_overlay", "Fused height-gradient/occlusion edges"),
        ):
            heatmap_path = result_path.parent / str(artifacts.get(key, ""))
            if heatmap_path.is_file():
                image_handles.append(
                    server.gui.add_image(
                        np.asarray(Image.open(heatmap_path).convert("RGB")),
                        label=label,
                    )
                )

    def render_targets() -> None:
        server.scene.remove_by_name("/targets")
        try:
            x, y = session.experiment_config.require_center()
        except BaseException:
            return
        surface_z = (
            float(state.perception_result["center_base_mm"][2])
            if state.perception_result is not None
            else (session.experiment_config.surface_z_mm or 0.0)
        )
        server.scene.add_icosphere(
            "/targets/detected_surface_center",
            radius=0.016,
            color=(220, 40, 200),
            position=(x / 1000.0, y / 1000.0, surface_z / 1000.0),
        )
        server.scene.add_label(
            "/targets/detected_surface_center_label",
            "Dense A/B fused garment center",
            position=(x / 1000.0, y / 1000.0, surface_z / 1000.0 + 0.03),
        )
        server.scene.add_label(
            "/targets/detected_surface_center_height",
            f"Observed surface z={surface_z:.1f} mm; Claude chooses waypoints",
            position=(x / 1000.0, y / 1000.0, surface_z / 1000.0 - 0.03),
        )
        update_plan_summary()

    def render_path(preflight: Preflight) -> None:
        server.scene.remove_by_name("/path")
        waypoints = path_waypoints_mm(preflight.actions, robot.init_pose_mm_deg)
        if len(waypoints) < 2:
            return
        points_m = np.stack([point for _, point in waypoints], axis=0) / 1000.0
        segments = np.stack([points_m[:-1], points_m[1:]], axis=1)
        palette = np.asarray(
            [(65, 105, 225), (245, 166, 35), (225, 65, 65), (50, 190, 90), (125, 125, 210)],
            dtype=np.uint8,
        )
        colors = np.stack(
            [palette[index % len(palette)] for index in range(len(segments))], axis=0
        )
        server.scene.add_line_segments(
            "/path/tcp",
            points=segments,
            colors=np.repeat(colors[:, None, :], 2, axis=1),
            line_width=4.0,
        )
        for index, (name, point_mm) in enumerate(waypoints):
            point_m = point_mm / 1000.0
            server.scene.add_icosphere(
                f"/path/waypoints/{index}",
                radius=0.008,
                color=tuple(int(v) for v in colors[min(index, len(colors) - 1)]),
                position=point_m,
            )
            server.scene.add_label(
                f"/path/waypoints/{index}_label",
                f"{index + 1}: {name}",
                position=point_m + np.asarray([0, 0, 0.018]),
            )

    def update_plan_summary() -> None:
        try:
            x, y = session.experiment_config.require_center()
        except BaseException:
            plan_summary.content = "### Plan\n\nNo validated garment center yet."
            return
        info = metadata()
        fusion = (
            state.perception_result.get("depth_fusion", {})
            if state.perception_result
            else {}
        )
        surface_z = state.perception_result.get("center_base_mm", [None, None, None])[2] if state.perception_result else session.experiment_config.surface_z_mm
        motion_line = (
            f"- Claude randomization waypoints: `{len(state.randomization_plan.waypoint_rows())}`\n"
            if state.randomization_plan is not None
            else "- motion waypoints: `Claude chooses approach / grasp / lift / transfer / release / yaw`\n"
        )
        plan_summary.content = (
            "### Plan\n\n"
            f"- plan kind: `{state.plan_kind}`\n"
            f"- perception: `{info.get('last_perception_mode', 'manual')}`\n"
            f"- cameras: `{info.get('last_active_cameras', [])}`\n"
            f"- center x/y: `({x:.2f}, {y:.2f}) mm`\n"
            f"- observed surface z: `{float(surface_z or 0.0):.2f} mm`\n"
            f"{motion_line}"
            f"- fusion mode: `{fusion.get('mode')}`\n"
            f"- fused points / input points: `"
            f"{fusion.get('fused_point_count')} / {fusion.get('input_point_count')}`\n"
            f"- shared A/B voxels: `"
            f"{fusion.get('source_voxel_counts', {}).get('AB_overlap')}`"
        )

    def invalidate_plan(reason: str, *, discard_experiment: bool = False) -> None:
        state.preflight = None
        state.controller_validation = None
        state.animation_frames = []
        state.approved_source_hash = None
        state.animation_playing = False
        if discard_experiment:
            state.experiment_name = None
        animation_slider.disabled = True
        animation_slider.max = 1
        animation_slider.value = 0
        play_button.disabled = True
        pause_button.disabled = True
        reset_button.disabled = True
        animation_state.content = f"Animation unavailable: {reason}"
        validation_details.content = "Plan must be validated again."
        update_execute_enabled()

    def set_busy(value: bool) -> None:
        state.busy = value
        init_button.disabled = value or not enable_real
        capture_button.disabled = value
        fusion_button.disabled = value or state.captured_frames is None
        validate_button.disabled = value or not _plan_ready()
        randomize_button.disabled = value or not _plan_ready()
        standard_grasp_button.disabled = value or not _plan_ready()
        if value:
            play_button.disabled = True
        else:
            play_button.disabled = not bool(state.animation_frames)
        update_execute_enabled()

    @init_button.on_click
    def _(event: Any) -> None:
        if init_button.disabled or not enable_real:
            return
        if not operation_lock.acquire(blocking=False):
            return
        notification = (
            event.client.add_notification(
                "Robot Init / Home",
                "Returning the arm to the configured Home at low speed...",
                loading=True,
                with_close_button=False,
                color="red",
            )
            if event.client is not None
            else None
        )
        source_path = session.workspace / "_viser_init_home.py"
        set_busy(True)
        state.animation_playing = False
        status.content = (
            "### Robot Init in progress\n\n"
            "The arm is returning to Home at the configured low joint speed. "
            "Keep the hardware emergency stop within reach."
        )
        init_contract.content = (
            "### Robot Init / Home\n\n"
            "`home()` is executing now. No gripper or Cartesian command is included."
        )
        try:
            source_path.write_text(canonical_home_source(), encoding="utf-8")
            preflight = session.runner.preflight(source_path.name)
            if preflight.error:
                raise RuntimeError(preflight.error)
            if [action.get("name") for action in preflight.actions] != ["home"]:
                raise PermissionError("Init program must contain exactly one home action")
            result = session.runner.run_experiment(
                source_path.name,
                real=True,
                confirmed=True,
                notes="Confirmed Viser Init button: return arm to configured Home only.",
            )
            robot_model.update_cfg(home_cfg)
            if state.animation_frames:
                animation_slider.value = 0
                apply_animation_frame(0)
            result_path = session.results / (result["experiment"] + ".json")
            status.content = (
                "### Robot Init finished\n\n"
                f"- completed: `{result['execution_completed']}`\n"
                f"- errors: `{result['robot_errors']}`\n"
                f"- result: `{result_path}`"
            )
            init_contract.content = (
                "### Robot Init / Home\n\n"
                f"Arm returned to `({robot.init_pose_mm_deg[0]:.1f}, "
                f"{robot.init_pose_mm_deg[1]:.1f}, {robot.init_pose_mm_deg[2]:.1f}) mm` "
                f"at `{robot.home_speed_deg_s:.1f} deg/s`."
            )
            if notification is not None:
                notification.title = "Robot Init finished"
                notification.body = "The arm returned to the configured Home."
                notification.loading = False
                notification.color = "green"
                notification.with_close_button = True
                notification.auto_close_seconds = 8.0
        except BaseException as exc:
            message = f"{type(exc).__name__}: {exc}"
            status.content = f"### Robot Init stopped/failed\n\n`{message}`"
            init_contract.content = (
                "### Robot Init / Home\n\n"
                f"Home motion did not complete: `{message}`"
            )
            if notification is not None:
                notification.title = "Robot Init failed"
                notification.body = message
                notification.loading = False
                notification.color = "red"
                notification.with_close_button = True
        finally:
            if source_path.is_file():
                source_path.unlink()
            set_busy(False)
            operation_lock.release()

    def _plan_ready() -> bool:
        try:
            session.experiment_config.require_ready()
            return True
        except BaseException:
            return False

    def preview_planned_source(source: str) -> Preflight:
        preview_path = session.workspace / "_viewer_mode_preview.py"
        preview_path.write_text(source, encoding="utf-8")
        try:
            preview = session.runner.preflight(preview_path.name)
        finally:
            if preview_path.is_file():
                preview_path.unlink()
        state.preflight = preview
        validation_details.content = _preflight_markdown(preview)
        render_path(preview)
        return preview

    def apply_animation_frame(index: int) -> None:
        if not state.animation_frames:
            return
        index = max(0, min(int(index), len(state.animation_frames) - 1))
        frame = state.animation_frames[index]
        robot_model.update_cfg(frame.configuration_rad)
        animation_state.content = (
            "### Animation\n\n"
            f"- frame: `{index + 1}/{len(state.animation_frames)}`\n"
            f"- source action: `{frame.action_index + 1}`\n"
            f"- phase: `{frame.label}`\n"
            f"- gripper drive: `{frame.configuration_rad[-1]:.3f} rad`"
        )

    def update_execute_enabled() -> None:
        execute_button.disabled = not (
            enable_real
            and state.approved_source_hash is not None
            and state.controller_validation is not None
            and bool(state.animation_frames)
            and not state.busy
        )
        execution_contract.content = (
            "### Physical execution\n\n"
            f"- server authority: `{'enabled' if enable_real else 'preview only'}`\n"
            f"- static preflight: `{'passed' if state.preflight and not state.preflight.error else 'not passed'}`\n"
            f"- controller IK: `{'passed' if state.controller_validation else 'not passed'}`\n"
            f"- URDF animation: `{'ready' if state.animation_frames else 'not ready'}`\n"
            f"- selected depth mode: `"
            f"{(state.perception_result or {}).get('depth_fusion', {}).get('mode')}`\n"
            "- confirmation: `clicking the red button executes immediately`"
        )

    @capture_button.on_click
    def _(event: Any) -> None:
        if not operation_lock.acquire(blocking=False):
            return
        notification = (
            event.client.add_notification(
                "RealSense capture",
                "Reading aligned RGB-D frames...",
                loading=True,
                with_close_button=False,
                color="blue",
            )
            if event.client is not None
            else None
        )
        set_busy(True)
        status.content = "### Capturing RealSense\n\nReading aligned RGB and depth."
        try:
            config = PerceptionConfig.load(root, perception_path)
            labels = ("A", "B")
            config = replace(config, active_camera_labels=labels)
            config.validate()
            frames = capture_two_view_rgbd(config)
            state.perception_config = config
            state.captured_frames = frames
            state.plan_kind = "center_grasp"
            state.planned_source = None
            state.randomization_plan = None
            randomization_summary.content = (
                "### Garment randomization\n\nRun dense A/B fusion before generating a randomization path."
            )
            render_captured_frames(frames)
            invalidate_plan("new RGB-D capture has not been analyzed", discard_experiment=True)
            status.content = (
                "### RGB-D captured\n\n"
                f"Captured cameras `{list(labels)}`. Review the photos and point cloud, then run A/B fusion."
            )
            if notification is not None:
                notification.title = "RealSense capture complete"
                notification.body = f"Captured cameras {list(labels)}."
                notification.loading = False
                notification.color = "green"
                notification.with_close_button = True
                notification.auto_close_seconds = 6.0
        except BaseException as exc:
            state.captured_frames = None
            status.content = f"### RealSense capture failed\n\n`{type(exc).__name__}: {exc}`"
            if notification is not None:
                notification.title = "RealSense capture failed"
                notification.body = f"{type(exc).__name__}: {exc}"
                notification.loading = False
                notification.color = "red"
                notification.with_close_button = True
        finally:
            set_busy(False)
            operation_lock.release()

    @fusion_button.on_click
    def _(event: Any) -> None:
        if state.captured_frames is None or state.perception_config is None:
            return
        if not operation_lock.acquire(blocking=False):
            return
        notification = (
            event.client.add_notification(
                "A/B depth fusion running",
                "Building the calibrated fused point cloud and garment center...",
                loading=True,
                with_close_button=False,
                color="blue",
            )
            if event.client is not None
            else None
        )
        set_busy(True)
        status.content = (
            "### Perception running\n\n"
            "Fusing A/B RGB-D in the robot base frame and estimating the garment center."
        )
        try:
            result = session.locate_cloth_center(
                state.perception_config, frames=state.captured_frames
            )
            saved, saved_path = _load_latest_perception(session)
            if saved is None or saved_path is None:
                raise RuntimeError("perception completed without a saved result")
            state.perception_result = saved
            state.perception_result_path = saved_path
            state.experiment_name = None
            state.plan_kind = "center_grasp"
            state.planned_source = None
            state.randomization_plan = None
            randomization_summary.content = (
                "### Alternative/manual paths\n\n"
                "Perception is observation-only. Use Claude free/automatic exploration for motion waypoints; manual paths require a complete config."
            )
            render_perception_result(saved, saved_path)
            render_targets()
            invalidate_plan("new perception plan has not passed controller IK")
            validate_button.disabled = False
            fusion = result.get("depth_fusion", {})
            status.content = (
                "### Dense A/B depth fusion validated\n\n"
                f"Base-frame garment center: `{result['center_base_mm']}`. "
                f"Fused points: `{fusion.get('fused_point_count')}`. "
                f"Shared A/B voxels: `{fusion.get('source_voxel_counts', {}).get('AB_overlap')}`. "
                "Perception supplied no waypoints; ask Claude to choose the complete action sequence."
            )
            if notification is not None:
                notification.title = "Dense A/B depth fusion validated"
                notification.body = (
                    f"Center: {result['center_base_mm']}; "
                    f"fused points: {fusion.get('fused_point_count')}"
                )
                notification.loading = False
                notification.color = "green"
                notification.with_close_button = True
                notification.auto_close_seconds = 10.0
        except BaseException as exc:
            invalidate_plan("A/B depth fusion failed", discard_experiment=True)
            message = f"{type(exc).__name__}: {exc}"
            status.content = f"### A/B depth fusion blocked\n\n`{message}`"
            if notification is not None:
                notification.title = "A/B depth fusion blocked"
                notification.body = message
                notification.loading = False
                notification.color = "red"
                notification.with_close_button = True
        finally:
            set_busy(False)
            operation_lock.release()

    @randomize_button.on_click
    def _(_: Any) -> None:
        if not _plan_ready() or not operation_lock.acquire(blocking=False):
            return
        set_busy(True)
        try:
            seed = int(time.time_ns() & 0x7FFFFFFF)
            plan = build_garment_randomization_plan(
                session.experiment_config,
                robot,
                seed=seed,
            )
            source = garment_randomization_source(plan)
            state.plan_kind = "garment_randomization"
            state.randomization_plan = plan
            state.planned_source = source
            state.experiment_name = None
            invalidate_plan("randomization path has not passed controller IK", discard_experiment=True)
            preview = preview_planned_source(source)
            if preview.error:
                raise RuntimeError(preview.error)
            randomization_summary.content = randomization_markdown(plan)
            update_plan_summary()
            validate_button.disabled = bool(preview.error)
            status.content = (
                "### Garment randomization path generated\n\n"
                f"Seed `{plan.seed}` produced `{len(plan.waypoint_rows())}` Cartesian path points. "
                "Review the path and source, then run controller validation and animation."
            )
        except BaseException as exc:
            state.plan_kind = "center_grasp"
            state.randomization_plan = None
            state.planned_source = None
            invalidate_plan("randomization path generation failed", discard_experiment=True)
            status.content = (
                "### Garment randomization blocked\n\n"
                f"`{type(exc).__name__}: {exc}`"
            )
        finally:
            set_busy(False)
            operation_lock.release()

    @standard_grasp_button.on_click
    def _(_: Any) -> None:
        if not _plan_ready() or not operation_lock.acquire(blocking=False):
            return
        set_busy(True)
        try:
            state.plan_kind = "center_grasp"
            state.randomization_plan = None
            state.planned_source = None
            state.experiment_name = None
            invalidate_plan("standard grasp path has not passed controller IK", discard_experiment=True)
            preview = preview_planned_source(canonical_grasp_source(session.experiment_config))
            if preview.error:
                raise RuntimeError(preview.error)
            randomization_summary.content = (
                "### Garment randomization\n\nStandard center-grasp path selected."
            )
            update_plan_summary()
            validate_button.disabled = bool(preview.error)
            status.content = (
                "### Standard center-grasp path selected\n\n"
                "Review the source/path, then run controller validation and animation."
            )
        except BaseException as exc:
            invalidate_plan("standard grasp preview failed", discard_experiment=True)
            status.content = f"### Standard grasp blocked\n\n`{type(exc).__name__}: {exc}`"
        finally:
            set_busy(False)
            operation_lock.release()

    @validate_button.on_click
    def _(event: Any) -> None:
        if not _plan_ready() or not operation_lock.acquire(blocking=False):
            return
        notification = (
            event.client.add_notification(
                "Validating robot plan",
                "Static preflight and read-only controller IK are running. This can take about 30 seconds.",
                loading=True,
                with_close_button=False,
                color="blue",
            )
            if event.client is not None
            else None
        )
        set_busy(True)
        status.content = (
            "### Validating plan\n\n"
            "Running static simulation, then read-only controller TCP/IK checks."
        )
        try:
            state.experiment_name = _ensure_experiment_source(
                session,
                state.experiment_name,
                expected_plan_source(),
            )
            preflight = session.runner.preflight(state.experiment_name)
            state.preflight = preflight
            validation_details.content = _preflight_markdown(preflight)
            render_path(preflight)
            if preflight.error:
                raise RuntimeError(preflight.error)
            controller = validate_controller_trajectory(robot, preflight.actions)
            frames = kinematics.build_animation(
                preflight.actions,
                robot.init_joints_deg,
                robot.orientation_roll_deg,
                robot.orientation_pitch_deg,
            )
            state.controller_validation = controller
            state.animation_frames = frames
            state.approved_source_hash = _source_hash(preflight.source)
            animation_slider.max = max(1, len(frames) - 1)
            animation_slider.value = 0
            animation_slider.disabled = False
            play_button.disabled = False
            pause_button.disabled = False
            reset_button.disabled = False
            apply_animation_frame(0)
            status.content = (
                "### Plan and controller IK passed\n\n"
                f"Plan kind: `{state.plan_kind}`. "
                f"Built `{len(frames)}` URDF animation frames. "
                f"Controller warning code: `{controller.controller_warning_code}`."
            )
            if notification is not None:
                notification.title = "Plan validation passed"
                notification.body = (
                    f"Controller IK passed and {len(frames)} animation frames are ready."
                )
                notification.loading = False
                notification.color = "green"
                notification.with_close_button = True
                notification.auto_close_seconds = 8.0
        except BaseException as exc:
            state.controller_validation = None
            state.animation_frames = []
            state.approved_source_hash = None
            animation_slider.disabled = True
            play_button.disabled = True
            pause_button.disabled = True
            reset_button.disabled = True
            status.content = (
                "### Hard validation failure\n\n"
                f"`{type(exc).__name__}: {exc}`\n\n"
                "Animation and physical execution are disabled. Change the scene/target; do not retry the same unreachable plan."
            )
            if notification is not None:
                notification.title = "Plan validation blocked"
                notification.body = f"{type(exc).__name__}: {exc}"
                notification.loading = False
                notification.color = "red"
                notification.with_close_button = True
        finally:
            set_busy(False)
            operation_lock.release()

    @animation_slider.on_update
    def _(_: Any) -> None:
        apply_animation_frame(int(animation_slider.value))

    @play_button.on_click
    def _(_: Any) -> None:
        if not state.animation_frames or state.animation_playing:
            return
        state.animation_playing = True

        def play() -> None:
            with animation_lock:
                try:
                    index = int(animation_slider.value)
                    if index >= len(state.animation_frames) - 1:
                        index = 0
                    while state.animation_playing and state.animation_frames:
                        animation_slider.value = index
                        apply_animation_frame(index)
                        index += 1
                        if index >= len(state.animation_frames):
                            if loop_animation.value:
                                index = 0
                            else:
                                break
                        time.sleep(1.0 / 12.0)
                finally:
                    state.animation_playing = False

        threading.Thread(target=play, daemon=True, name="xarm-viser-animation").start()

    @pause_button.on_click
    def _(_: Any) -> None:
        state.animation_playing = False

    @reset_button.on_click
    def _(_: Any) -> None:
        state.animation_playing = False
        animation_slider.value = 0
        apply_animation_frame(0)

    @execute_button.on_click
    def _(_: Any) -> None:
        if execute_button.disabled or state.experiment_name is None:
            return
        if not operation_lock.acquire(blocking=False):
            return
        set_busy(True)
        state.animation_playing = False
        status.content = (
            "### Physical rollout in progress\n\n"
            "Use the hardware emergency stop if anything is unexpected. No retry will occur."
        )
        try:
            current = session.runner.preflight(state.experiment_name)
            if current.error:
                raise RuntimeError(f"preflight changed/failed: {current.error}")
            if _source_hash(current.source) != state.approved_source_hash:
                raise PermissionError("experiment source changed after preview; validate again")
            result = session.run_experiment(
                state.experiment_name,
                real=True,
                confirmed=True,
                single_view_confirmed=(
                    metadata().get("last_perception_mode") == "single_camera_rgbd"
                ),
                notes=(
                    f"Confirmed Viser {state.plan_kind} physical execution after "
                    "RealSense/A+B fusion/path/URDF preview."
                ),
            )
            status.content = (
                "### Physical rollout finished\n\n"
                f"- completed: `{result['execution_completed']}`\n"
                f"- errors: `{result['robot_errors']}`\n"
                f"- result: `{session.results / (result['experiment'] + '.json')}`"
            )
        except BaseException as exc:
            status.content = f"### Physical rollout stopped/failed\n\n`{type(exc).__name__}: {exc}`"
        finally:
            state.approved_source_hash = None
            state.controller_validation = None
            set_busy(False)
            operation_lock.release()

    @server.on_client_connect
    def _(client: Any) -> None:
        client.camera.position = (1.3, -0.9, 0.9)
        client.camera.look_at = (0.62, -0.07, 0.1)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    if latest_perception is not None and latest_perception_path is not None:
        render_perception_result(latest_perception, latest_perception_path)
        render_targets()
    if _plan_ready():
        validate_button.disabled = False
        randomize_button.disabled = False
        standard_grasp_button.disabled = False
        preview_source = (
            (session.workspace / selected).read_text(encoding="utf-8")
            if selected is not None
            else canonical_grasp_source(session.experiment_config)
        )
        preview_path = session.workspace / (selected or "_viewer_preview.py")
        temporary = selected is None
        if temporary:
            preview_path.write_text(preview_source, encoding="utf-8")
        try:
            preview = session.runner.preflight(preview_path.name)
            validation_details.content = _preflight_markdown(preview)
            render_path(preview)
        finally:
            if temporary and preview_path.is_file():
                preview_path.unlink()
    update_plan_summary()
    update_execute_enabled()

    print(f"Viser cloth grasp console: http://{host}:{port}")
    print(f"Run workspace: {session.workspace}")
    print(format_speed_profile(robot))
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Viser console stopped; no additional robot command was sent.")
    finally:
        state.animation_playing = False
        server.stop()
    return 0
