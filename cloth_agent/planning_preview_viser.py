"""Read-only Viser viewer for Claude planning-preview point clouds and paths."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .viewer import _load_fused_point_cloud, _view_point_cloud


TRAJECTORY_COLORS = np.asarray(
    [
        (230, 45, 45),
        (40, 130, 255),
        (40, 180, 80),
        (220, 125, 25),
        (170, 70, 220),
        (220, 45, 150),
        (20, 170, 170),
        (180, 160, 30),
    ],
    dtype=np.uint8,
)


def _load_json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value


def _sample_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    max_points: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if len(points) <= max_points:
        return points, colors
    stride = int(np.ceil(len(points) / max_points))
    return points[::stride], colors[::stride]


def _move_points(plan: dict[str, Any]) -> np.ndarray:
    proposal = plan.get("proposal") if isinstance(plan, dict) else None
    actions = proposal.get("actions", []) if isinstance(proposal, dict) else []
    points: list[list[float]] = []
    if not isinstance(actions, list):
        return np.empty((0, 3), dtype=np.float32)
    for action in actions:
        if not isinstance(action, dict) or action.get("name") != "move":
            continue
        args = action.get("args")
        if not isinstance(args, dict):
            continue
        try:
            points.append(
                [
                    float(args["x"]) / 1000.0,
                    float(args["y"]) / 1000.0,
                    float(args["z"]) / 1000.0,
                ]
            )
        except (KeyError, TypeError, ValueError):
            continue
    return np.asarray(points, dtype=np.float32).reshape((-1, 3))


def _grasp_base_point(
    result: dict[str, Any],
    result_dir: Path,
    plan: dict[str, Any],
) -> np.ndarray | None:
    """Resolve Claude's Camera-A grasp pixel to a measured base point.

    This is a diagnostic marker only. It does not replace or modify Claude's
    free-motion waypoint coordinates.
    """

    proposal = plan.get("proposal") if isinstance(plan, dict) else None
    selected = proposal.get("selected_grasp") if isinstance(proposal, dict) else None
    if not isinstance(selected, dict) or selected.get("camera") != "A":
        return None
    pixel = selected.get("pixel_xy")
    if not isinstance(pixel, list) or len(pixel) != 2:
        return None
    try:
        x_px, y_px = int(pixel[0]), int(pixel[1])
    except (TypeError, ValueError):
        return None
    view = next(
        (item for item in result.get("views", []) if isinstance(item, dict) and str(item.get("label", "")).upper() == "A"),
        None,
    )
    if view is None:
        return None
    raw_map = view.get("base_xyz_map")
    if not isinstance(raw_map, str) or not raw_map:
        return None
    path = result_dir / raw_map
    if not path.is_file():
        return None
    try:
        base_map = np.asarray(np.load(path), dtype=np.float64)
    except (OSError, ValueError):
        return None
    if base_map.ndim != 3 or base_map.shape[2] != 3:
        return None
    if not (0 <= x_px < base_map.shape[1] and 0 <= y_px < base_map.shape[0]):
        return None
    point = base_map[y_px, x_px]
    if not np.all(np.isfinite(point)):
        return None
    return (point / 1000.0).astype(np.float32)


def _planned_grasp_waypoint(plan: dict[str, Any]) -> np.ndarray | None:
    """Return the last move immediately before close_gripper, in metres."""

    proposal = plan.get("proposal") if isinstance(plan, dict) else None
    actions = proposal.get("actions", []) if isinstance(proposal, dict) else []
    if not isinstance(actions, list):
        return None
    close_index = next(
        (index for index, action in enumerate(actions) if isinstance(action, dict) and action.get("name") == "close_gripper"),
        None,
    )
    if close_index is None:
        return None
    for action in reversed(actions[:close_index]):
        if not isinstance(action, dict) or action.get("name") != "move":
            continue
        args = action.get("args")
        if not isinstance(args, dict):
            return None
        try:
            return np.asarray(
                [float(args["x"]), float(args["y"]), float(args["z"])],
                dtype=np.float32,
            ) / 1000.0
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _project_base_point(
    view: dict[str, Any],
    point_m: np.ndarray,
) -> tuple[float, float] | None:
    try:
        intrinsics = np.asarray(view["intrinsics"], dtype=np.float64)
        base_from_camera = np.asarray(view["X_base_camera"], dtype=np.float64)
        camera = np.linalg.inv(base_from_camera) @ np.concatenate(
            [np.asarray(point_m, dtype=np.float64), [1.0]]
        )
        if not np.all(np.isfinite(camera)) or camera[2] <= 0.0:
            return None
        return (
            float(intrinsics[0, 0] * camera[0] / camera[2] + intrinsics[0, 2]),
            float(intrinsics[1, 1] * camera[1] / camera[2] + intrinsics[1, 2]),
        )
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return None


def alignment_diagnostic(
    result: dict[str, Any],
    result_dir: Path,
    plan: dict[str, Any],
    *,
    xy_threshold_mm: float = 30.0,
    pixel_threshold_px: float = 40.0,
) -> dict[str, Any]:
    """Check whether the planned grasp waypoint starts at Claude's image target."""

    proposal = plan.get("proposal") if isinstance(plan, dict) else {}
    selected = proposal.get("selected_grasp") if isinstance(proposal, dict) else {}
    selected_pixel = selected.get("pixel_xy") if isinstance(selected, dict) else None
    measured = _grasp_base_point(result, result_dir, plan)
    planned = _planned_grasp_waypoint(plan)
    view = next(
        (item for item in result.get("views", []) if isinstance(item, dict) and str(item.get("label", "")).upper() == "A"),
        None,
    )
    projected = _project_base_point(view, planned) if view is not None and planned is not None else None
    pixel_error = None
    if (
        isinstance(selected_pixel, list)
        and len(selected_pixel) == 2
        and projected is not None
    ):
        pixel_error = float(
            np.linalg.norm(
                np.asarray(projected, dtype=np.float64)
                - np.asarray([float(selected_pixel[0]), float(selected_pixel[1])], dtype=np.float64)
            )
        )
    xyz_error = None
    xy_error = None
    z_error = None
    if measured is not None and planned is not None:
        delta_mm = (planned - measured) * 1000.0
        xyz_error = float(np.linalg.norm(delta_mm))
        xy_error = float(np.linalg.norm(delta_mm[:2]))
        z_error = float(abs(delta_mm[2]))
    aligned = (
        xy_error is not None
        and pixel_error is not None
        and xy_error <= float(xy_threshold_mm)
        and pixel_error <= float(pixel_threshold_px)
    )
    return {
        "iteration": plan.get("round"),
        "selected_grasp_pixel_xy": selected_pixel,
        "measured_grasp_base_mm": (
            np.round(measured * 1000.0, 3).tolist() if measured is not None else None
        ),
        "planned_grasp_waypoint_base_mm": (
            np.round(planned * 1000.0, 3).tolist() if planned is not None else None
        ),
        "projected_planned_grasp_pixel_xy": (
            [round(float(projected[0]), 3), round(float(projected[1]), 3)]
            if projected is not None
            else None
        ),
        "xyz_error_mm": xyz_error,
        "xy_error_mm": xy_error,
        "z_error_mm": z_error,
        "pixel_error_px": pixel_error,
        "thresholds": {
            "xy_error_mm": float(xy_threshold_mm),
            "pixel_error_px": float(pixel_threshold_px),
        },
        "status": "ALIGNED" if aligned else "MISALIGNED",
        "note": "grasp waypoint is expected to use Rxxx base XY; later transport remains free-motion",
    }


def write_alignment_report(
    preview_dir: Path,
    perception_result: Path,
) -> Path:
    """Write alignment diagnostics for every saved preview iteration."""

    preview_dir = preview_dir.resolve()
    perception_result = perception_result.resolve()
    plans_doc = _load_json(preview_dir / "plans.json")
    plans = plans_doc.get("plans", plans_doc) if isinstance(plans_doc, dict) else plans_doc
    if not isinstance(plans, list):
        plans = []
    result = _load_json(perception_result)
    if not isinstance(result, dict):
        result = {}
    report = {
        "preview_dir": str(preview_dir),
        "perception_result": str(perception_result),
        "mode": "FREE_MOTION_ALIGNMENT_DIAGNOSTIC",
        "iterations": [alignment_diagnostic(result, perception_result.parent, plan) for plan in plans if isinstance(plan, dict)],
    }
    output = preview_dir / "alignment_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _arrow_segments(points: np.ndarray) -> np.ndarray:
    """Return the path plus small 3-D arrowheads at each move endpoint."""

    points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    if len(points) < 2:
        return np.empty((0, 2, 3), dtype=np.float32)
    segments: list[np.ndarray] = []
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        length = float(np.linalg.norm(delta))
        segments.append(np.stack([start, end]))
        if not np.isfinite(length) or length < 1e-6:
            continue
        direction = delta / length
        arrow_length = min(0.035, max(0.012, length * 0.22))
        # Pick a stable vector perpendicular to the travel direction.
        reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(direction, reference))) > 0.9:
            reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        side = np.cross(direction, reference)
        side_norm = float(np.linalg.norm(side))
        if side_norm < 1e-6:
            continue
        side = side / side_norm
        back = end - direction * arrow_length
        wing = side * (arrow_length * 0.42)
        segments.append(np.stack([end, back + wing]))
        segments.append(np.stack([end, back - wing]))
    return np.asarray(segments, dtype=np.float32)


def _clear_handles(handles: list[Any]) -> None:
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass
    handles.clear()


class PlanningPreviewViser:
    """Display the saved fused cloud and all preview trajectories in Viser."""

    def __init__(self, server: Any, preview_dir: Path, perception_result: Path):
        self.server = server
        self.preview_dir = preview_dir.resolve()
        self.perception_result = perception_result.resolve()
        self.cloud_handles: list[Any] = []
        self.path_handles: list[Any] = []
        self.last_plans_mtime_ns: int | None = None
        self.status = server.gui.add_markdown(
            "### Planning preview Viser\n\nWaiting for saved plans."
        )
        self.history_panel = server.gui.add_markdown(
            "### Iteration plan history\n\nNo plans loaded yet."
        )

    def _clear_scene(self) -> None:
        _clear_handles(self.cloud_handles)
        _clear_handles(self.path_handles)
        for root in ("/preview_cloud", "/preview_trajectories"):
            try:
                self.server.scene.remove_by_name(root)
            except Exception:
                pass

    def _render_cloud(self) -> int:
        result = _load_json(self.perception_result)
        if not isinstance(result, dict):
            result = {}
        result_dir = self.perception_result.parent
        try:
            points, colors = _load_fused_point_cloud(result, result_dir)
            points, colors = _sample_cloud(points, colors)
        except Exception:
            points = np.empty((0, 3), dtype=np.float32)
            colors = np.empty((0, 3), dtype=np.uint8)
        if len(points):
            self.cloud_handles.append(
                self.server.scene.add_point_cloud(
                    "/preview_cloud/fused_AB",
                    points=points,
                    colors=colors,
                    point_size=0.004,
                    point_shape="circle",
                )
            )
            return len(points)

        # Older perception results may not contain the fused artifact. Fall
        # back to the saved camera clouds so the preview remains inspectable.
        loaded = 0
        for view in result.get("views", []):
            if not isinstance(view, dict):
                continue
            try:
                view_points, view_colors = _view_point_cloud(view, result_dir, stride=5)
                view_points, view_colors = _sample_cloud(view_points, view_colors)
            except Exception:
                continue
            if not len(view_points):
                continue
            label = str(view.get("label", "camera")).upper()
            self.cloud_handles.append(
                self.server.scene.add_point_cloud(
                    f"/preview_cloud/camera_{label}",
                    points=view_points,
                    colors=view_colors,
                    point_size=0.004,
                    point_shape="circle",
                )
            )
            loaded += len(view_points)
        return loaded

    def render(self) -> bool:
        plans_path = self.preview_dir / "plans.json"
        if not plans_path.is_file():
            return False
        try:
            mtime_ns = plans_path.stat().st_mtime_ns
        except OSError:
            return False
        plans_doc = _load_json(plans_path)
        raw_plans = plans_doc.get("plans", plans_doc) if isinstance(plans_doc, dict) else plans_doc
        if not isinstance(raw_plans, list):
            return False
        self._clear_scene()
        cloud_count = self._render_cloud()
        result = _load_json(self.perception_result)
        if not isinstance(result, dict):
            result = {}
        result_dir = self.perception_result.parent
        history: list[str] = []
        total_waypoints = 0
        for index, plan in enumerate(raw_plans):
            if not isinstance(plan, dict):
                continue
            points = _move_points(plan)
            if not len(points):
                continue
            total_waypoints += len(points)
            color = TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)]
            iteration = int(plan.get("round", index + 1))
            segments = _arrow_segments(points)
            if len(segments):
                segment_colors = np.repeat(color[None, None, :], len(segments), axis=0)
                segment_colors = np.repeat(segment_colors, 2, axis=1)
                self.path_handles.append(
                    self.server.scene.add_line_segments(
                        f"/preview_trajectories/iter_{iteration}/path",
                        points=segments,
                        colors=segment_colors,
                        line_width=5.0,
                    )
                )
            self.path_handles.append(
                self.server.scene.add_point_cloud(
                    f"/preview_trajectories/iter_{iteration}/waypoints",
                    points=points,
                    colors=np.tile(color[None, :], (len(points), 1)),
                    point_size=0.010,
                    point_shape="circle",
                )
            )
            # The white marker is the measured Camera-A grasp location. The
            # colored trajectory remains Claude's raw, ungrounded plan, so a
            # mismatch is visible instead of being silently hidden.
            grasp_point = _grasp_base_point(result, result_dir, plan)
            if grasp_point is not None:
                self.path_handles.append(
                    self.server.scene.add_point_cloud(
                        f"/preview_trajectories/iter_{iteration}/grounded_grasp",
                        points=grasp_point.reshape((1, 3)),
                        colors=np.asarray([[255, 255, 255]], dtype=np.uint8),
                        point_size=0.016,
                        point_shape="circle",
                    )
                )
            diagnostic = alignment_diagnostic(result, result_dir, plan)
            mismatch_text = ""
            if diagnostic.get("xy_error_mm") is not None:
                mismatch_text += f"; XY error={float(diagnostic['xy_error_mm']):.0f} mm"
            if diagnostic.get("pixel_error_px") is not None:
                mismatch_text += f"; pixel error={float(diagnostic['pixel_error_px']):.0f} px"
            grasp_base_text = (
                f"; grasp base mm={np.round(grasp_point * 1000.0, 1).tolist()}"
                if grasp_point is not None
                else ""
            )
            proposal = plan.get("proposal", {})
            selected = proposal.get("selected_grasp", {}) if isinstance(proposal, dict) else {}
            grasp = selected.get("pixel_xy") if isinstance(selected, dict) else None
            observation = proposal.get("garment_observation", "") if isinstance(proposal, dict) else ""
            strategy = proposal.get("reveal_strategy", "") if isinstance(proposal, dict) else ""
            history.append(
                f"- **Iter {iteration}** — color `rgb{tuple(int(v) for v in color)}`, "
                f"grasp pixel `{grasp}`, waypoints `{len(points)}`\n"
                f"  - white marker: measured Camera-A grasp{grasp_base_text}; "
                f"arrows: move direction; alignment=`{diagnostic.get('status')}`{mismatch_text}\n"
                f"  - observation: {observation}\n"
                f"  - intent: {strategy}"
            )
        self.status.content = (
            "### Planning preview Viser\n\n"
            f"- preview directory: `{self.preview_dir}`\n"
            f"- fused/cloud points: `{cloud_count}`\n"
            f"- iterations: `{len(raw_plans)}`\n"
            f"- trajectory waypoints: `{total_waypoints}`\n"
            "- arrowheads show waypoint order; white points show measured grasp pixels\n"
            "- grasp XY is anchored to the selected Rxxx reference; transport remains Claude's free-motion plan\n"
            "- planning coordinates are shown without workspace, table, collision, or IK filtering\n"
            "- colored lines/points are hypothetical Claude plans; no robot command was sent"
        )
        self.history_panel.content = "### Iteration plan history\n\n" + (
            "\n\n".join(history) if history else "No renderable move waypoints."
        )
        self.last_plans_mtime_ns = mtime_ns
        return True


def run_viewer(
    preview_dir: Path,
    *,
    perception_result: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    refresh_s: float = 1.0,
) -> int:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise PermissionError("planning preview Viser must bind to loopback")
    if not 0.1 <= refresh_s <= 30.0:
        raise ValueError("refresh_s must be between 0.1 and 30 seconds")
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError(
            "Viser is required; install it with python -m pip install 'viser>=1.0,<2'"
        ) from exc
    preview_dir = preview_dir.expanduser().resolve()
    perception_result = perception_result.expanduser().resolve()
    if not preview_dir.is_dir():
        raise FileNotFoundError(preview_dir)
    if not perception_result.is_file():
        raise FileNotFoundError(perception_result)
    server = viser.ViserServer(host=host, port=port, label="Claude planning preview")
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/preview_table",
        width=1.2,
        height=0.8,
        cell_size=0.05,
        section_size=0.25,
        position=(0.0, 0.0, 0.0),
    )
    state = PlanningPreviewViser(server, preview_dir, perception_result)
    stop = threading.Event()

    def follow() -> None:
        while not stop.is_set():
            plans_path = preview_dir / "plans.json"
            try:
                changed = (
                    plans_path.is_file()
                    and plans_path.stat().st_mtime_ns != state.last_plans_mtime_ns
                )
            except OSError:
                changed = False
            if changed:
                try:
                    state.render()
                except Exception as exc:
                    state.status.content = (
                        "### Planning preview Viser\n\n"
                        f"Update failed: `{type(exc).__name__}: {exc}`"
                    )
            stop.wait(refresh_s)

    thread = threading.Thread(target=follow, daemon=True, name="planning-preview-viser-follow")
    thread.start()
    print(f"Planning preview Viser: http://{host}:{port}", flush=True)
    print(f"Following preview: {preview_dir}", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Planning preview Viser stopped; no robot command was sent.", flush=True)
    finally:
        stop.set()
        thread.join(timeout=max(1.0, refresh_s + 0.5))
        server.stop()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preview_dir", type=Path)
    parser.add_argument("--perception-result", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh-s", type=float, default=1.0)
    args = parser.parse_args(argv)
    return run_viewer(
        args.preview_dir,
        perception_result=args.perception_result,
        host=args.host,
        port=args.port,
        refresh_s=args.refresh_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
