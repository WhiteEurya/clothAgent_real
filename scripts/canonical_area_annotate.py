#!/usr/bin/env python3
"""Interactively label sparse current nodes for canonical-graph evaluation.

This tool is deliberately a human evaluation aid, not a perception fallback.
It opens the flat reference and one current observation side by side. It can
label current-node-first or, for deformed garments, canonical-reference-first.
Only selected visible nodes need labels; the resulting JSON can be passed directly to
``canonical_area_graph.py match --ground-truth NAME=...``.

Controls:

* left click current, then reference: assign a canonical identity;
* right click current: remove that node's assignment;
* ``u``: undo the most recent assignment/removal;
* ``s``: save without closing;
* ``q`` or Enter: save and close;
* Escape: close without saving changes from this session.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.canonical_area_graph import CanonicalSurfaceGraph


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _point_map_from_visibility(payload: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    raw_nodes = payload.get("current_nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("visibility JSON must contain a non-empty current_nodes list")
    points: dict[str, tuple[float, float]] = {}
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            raise TypeError("each current_nodes entry must be an object")
        node_id = str(raw_node["node_id"])
        xy = raw_node.get("pixel_xy")
        if not isinstance(xy, list | tuple) or len(xy) != 2:
            raise ValueError(f"current node {node_id} has invalid pixel_xy")
        point = (float(xy[0]), float(xy[1]))
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f"current node {node_id} has non-finite pixel_xy")
        if node_id in points:
            raise ValueError(f"duplicate current node ID: {node_id}")
        points[node_id] = point
    return points


def nearest_id(
    points: Mapping[str, tuple[float, float]],
    xy: tuple[float, float],
    *,
    max_distance_px: float,
) -> str | None:
    """Return the nearest point ID inside a finite selection radius."""

    if max_distance_px <= 0:
        raise ValueError("max_distance_px must be positive")
    x, y = map(float, xy)
    best_id: str | None = None
    best_distance = float(max_distance_px)
    for point_id, (point_x, point_y) in points.items():
        distance = math.hypot(x - point_x, y - point_y)
        if distance <= best_distance:
            best_id = point_id
            best_distance = distance
    return best_id


def sparse_node_points(
    points: Mapping[str, tuple[float, float]],
    *,
    stride: int,
) -> dict[str, tuple[float, float]]:
    """Thin regular DINO-grid nodes so the garment remains visible."""

    if stride <= 0:
        raise ValueError("display stride must be positive")
    if stride == 1:
        return dict(points)
    selected: dict[str, tuple[float, float]] = {}
    fallback: list[tuple[str, tuple[float, float]]] = []
    for index, (node_id, xy) in enumerate(points.items()):
        match = re.fullmatch(r"X(\d+)_(\d+)", node_id)
        if match is None:
            if index % stride == 0:
                fallback.append((node_id, xy))
            continue
        py, px = int(match.group(1)), int(match.group(2))
        if py % stride == 0 and px % stride == 0:
            selected[node_id] = xy
    if selected:
        return selected
    return dict(fallback)


@dataclass
class AnnotationState:
    assignments: dict[str, str] = field(default_factory=dict)
    selected_node_id: str | None = None
    history: list[tuple[str, str | None, str | None]] = field(default_factory=list)

    def assign(self, node_id: str, area_id: str) -> None:
        previous = self.assignments.get(node_id)
        if previous == area_id:
            self.selected_node_id = None
            return
        self.history.append((node_id, previous, area_id))
        self.assignments[node_id] = area_id
        self.selected_node_id = None

    def remove(self, node_id: str) -> None:
        previous = self.assignments.get(node_id)
        if previous is None:
            return
        self.history.append((node_id, previous, None))
        del self.assignments[node_id]
        if self.selected_node_id == node_id:
            self.selected_node_id = None

    def undo(self) -> bool:
        if not self.history:
            return False
        node_id, previous, _current = self.history.pop()
        if previous is None:
            self.assignments.pop(node_id, None)
        else:
            self.assignments[node_id] = previous
        self.selected_node_id = None
        return True


def _load_existing_assignments(
    output_path: Path,
    *,
    valid_node_ids: set[str],
    valid_area_ids: set[str],
) -> dict[str, str]:
    if not output_path.is_file():
        return {}
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    raw = payload.get("ground_truth", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw, dict):
        raise TypeError("existing annotation must contain a ground_truth object")
    assignments = {str(node_id): str(area_id) for node_id, area_id in raw.items()}
    unknown_nodes = sorted(set(assignments) - valid_node_ids)
    unknown_areas = sorted(set(assignments.values()) - valid_area_ids)
    if unknown_nodes or unknown_areas:
        raise ValueError(
            "existing annotation is incompatible with this graph/visibility: "
            f"unknown_nodes={unknown_nodes}, unknown_areas={unknown_areas}"
        )
    return assignments


def save_annotations(
    output_path: Path,
    assignments: Mapping[str, str],
    *,
    graph_path: Path,
    visibility_path: Path,
    reference_path: Path | Mapping[str, Path],
    current_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": _now(),
        "graph": str(graph_path),
        "visibility": str(visibility_path),
        "current_image": str(current_path),
        "evaluated_node_count": len(assignments),
        "ground_truth": dict(sorted(assignments.items())),
    }
    if isinstance(reference_path, Mapping):
        payload["reference_images"] = {
            str(side).strip().upper(): str(path)
            for side, path in reference_path.items()
        }
    else:
        payload["reference_image"] = str(reference_path)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _fit_panel(image: np.ndarray, target_height: int) -> tuple[np.ndarray, float]:
    scale = target_height / image.shape[0]
    target_width = max(1, round(image.shape[1] * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation), scale


def annotate(
    graph_path: Path,
    visibility_path: Path,
    reference_path: Path | Mapping[str, Path],
    current_path: Path,
    output_path: Path,
    *,
    selection_radius_px: float = 28.0,
    max_display_width: int = 1500,
    max_display_height: int = 900,
    overwrite: bool = False,
    display_stride: int = 1,
    reference_first: bool = False,
) -> None:
    graph = CanonicalSurfaceGraph.load(graph_path)
    visibility = json.loads(visibility_path.read_text(encoding="utf-8"))
    all_node_points = _point_map_from_visibility(visibility)
    if isinstance(reference_path, Mapping):
        reference_sources = [
            (
                str(side).strip().upper(),
                path,
                np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8),
            )
            for side, path in reference_path.items()
        ]
        required_sides = {
            area.surface_side
            for area in graph.areas.values()
            if area.surface_side is not None
        }
        missing_sides = sorted(required_sides - {side for side, _, _ in reference_sources})
        if missing_sides:
            raise ValueError(f"reference images missing surface sides: {missing_sides}")
    else:
        reference_sources = [
            (
                "REFERENCE",
                reference_path,
                np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.uint8),
            )
        ]
    current_rgb = np.asarray(Image.open(current_path).convert("RGB"), dtype=np.uint8)
    for side, _, reference_rgb in reference_sources:
        if (reference_rgb.shape[1], reference_rgb.shape[0]) != graph.image_size:
            raise ValueError(
                f"reference {side} image size "
                f"{(reference_rgb.shape[1], reference_rgb.shape[0])} "
                f"does not match graph {graph.image_size}"
            )
    for node_id, (x, y) in all_node_points.items():
        if not (0 <= x < current_rgb.shape[1] and 0 <= y < current_rgb.shape[0]):
            raise ValueError(f"current node {node_id} lies outside the current image")

    assignments = (
        {}
        if overwrite
        else _load_existing_assignments(
            output_path,
            valid_node_ids=set(all_node_points),
            valid_area_ids=set(graph.areas),
        )
    )
    node_points = sparse_node_points(all_node_points, stride=display_stride)
    for node_id in assignments:
        node_points[node_id] = all_node_points[node_id]
    state = AnnotationState(assignments=assignments)
    reference_bgr_sources = [
        (side, cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR))
        for side, _, reference_rgb in reference_sources
    ]
    current_bgr = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2BGR)
    header_height = 38
    target_height = min(
        max_display_height - header_height,
        *(reference_bgr.shape[0] for _, reference_bgr in reference_bgr_sources),
        current_bgr.shape[0],
    )
    if target_height <= 0:
        raise ValueError("display height is too small")
    estimated_width = (
        sum(
            reference_bgr.shape[1] * target_height / reference_bgr.shape[0]
            for _, reference_bgr in reference_bgr_sources
        )
        + current_bgr.shape[1] * target_height / current_bgr.shape[0]
    )
    if estimated_width > max_display_width:
        target_height = max(1, round(target_height * max_display_width / estimated_width))
    reference_panels = [
        (side, *_fit_panel(reference_bgr, target_height))
        for side, reference_bgr in reference_bgr_sources
    ]
    current_panel, current_scale = _fit_panel(current_bgr, target_height)
    reference_width = sum(panel.shape[1] for _, panel, _ in reference_panels)
    multi_reference = isinstance(reference_path, Mapping)
    area_display_points: dict[str, tuple[float, float]] = {}
    offset = 0
    for side, panel, scale in reference_panels:
        for area_id, area in graph.areas.items():
            if multi_reference and area.surface_side != side:
                continue
            area_display_points[area_id] = (
                offset + float(area.canonical_xy[0]) * scale,
                float(area.canonical_xy[1]) * scale,
            )
        offset += panel.shape[1]
    window = "Canonical area ground truth"
    show_unassigned = not reference_first
    selected_area_id: str | None = None

    def render() -> None:
        left_panels: list[np.ndarray] = []
        for side, panel, scale in reference_panels:
            left = panel.copy()
            panel_area_ids = {
                area_id
                for area_id, area in graph.areas.items()
                if not multi_reference or area.surface_side == side
            }
            for area_id in panel_area_ids:
                area = graph.areas[area_id]
                x, y = area.canonical_xy
                point = (round(x * scale), round(y * scale))
                for neighbor_id in area.neighbor_area_ids:
                    if area_id >= neighbor_id or neighbor_id not in panel_area_ids:
                        continue
                    neighbor = graph.areas[neighbor_id]
                    neighbor_point = (
                        round(neighbor.canonical_xy[0] * scale),
                        round(neighbor.canonical_xy[1] * scale),
                    )
                    cv2.line(left, point, neighbor_point, (180, 180, 180), 1, cv2.LINE_AA)
                if area_id == selected_area_id:
                    area_color, radius = (40, 255, 40), 9
                elif area_id in state.assignments.values():
                    area_color, radius = (40, 210, 60), 6
                else:
                    area_color, radius = (0, 200, 255), 6
                cv2.circle(left, point, radius, area_color, -1, cv2.LINE_AA)
                cv2.putText(
                    left,
                    area_id,
                    (point[0] + 7, point[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    left,
                    area_id,
                    (point[0] + 7, point[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            if multi_reference:
                cv2.rectangle(left, (0, 0), (150, 25), (0, 0, 0), -1)
                cv2.putText(left, side, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            left_panels.append(left)
        right = current_panel.copy()
        for node_id, (x, y) in node_points.items():
            point = (round(x * current_scale), round(y * current_scale))
            if node_id == state.selected_node_id:
                node_color, radius = (0, 220, 255), 7
            elif node_id in state.assignments:
                node_color, radius = (40, 210, 60), 5
            else:
                if not show_unassigned:
                    continue
                node_color, radius = (210, 210, 210), 3
            thickness = -1 if node_id == state.selected_node_id or node_id in state.assignments else 1
            cv2.circle(right, point, radius, node_color, thickness, cv2.LINE_AA)
            if node_id in state.assignments:
                cv2.putText(
                    right,
                    state.assignments[node_id] if reference_first else f"{node_id}->{state.assignments[node_id]}",
                    (point[0] + 6, point[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,
                    (40, 230, 60),
                    1,
                    cv2.LINE_AA,
                )
        body = np.concatenate([*left_panels, right], axis=1)
        canvas = np.zeros((target_height + header_height, body.shape[1], 3), dtype=np.uint8)
        canvas[header_height:] = body
        selected = selected_area_id if reference_first else state.selected_node_id
        direction = "reference -> current" if reference_first else "current -> reference"
        status = (
            f"labels={len(state.assignments)} selected={selected or 'none'} | {direction} | "
            "right click current=remove | u undo | s save | q save+quit"
        )
        cv2.putText(
            canvas,
            status,
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window, canvas)

    def current_click(display_x: int, display_y: int) -> str | None:
        return nearest_id(
            all_node_points if reference_first else node_points,
            (
                (display_x - reference_width) / current_scale,
                display_y / current_scale,
            ),
            max_distance_px=selection_radius_px,
        )

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        nonlocal selected_area_id
        body_y = y - header_height
        if body_y < 0 or body_y >= target_height:
            return
        if reference_first and x < reference_width:
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            area_id = nearest_id(
                area_display_points,
                (x, body_y),
                max_distance_px=selection_radius_px * min(
                    scale for _, _, scale in reference_panels
                ),
            )
            if area_id is not None:
                selected_area_id = area_id
                render()
            return
        if x >= reference_width:
            node_id = current_click(x, body_y)
            if node_id is None:
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                if reference_first:
                    if selected_area_id is None:
                        return
                    state.assign(node_id, selected_area_id)
                    node_points[node_id] = all_node_points[node_id]
                    selected_area_id = None
                else:
                    state.selected_node_id = node_id
                render()
            elif event == cv2.EVENT_RBUTTONDOWN:
                assigned_points = {
                    assigned_node: all_node_points[assigned_node]
                    for assigned_node in state.assignments
                }
                removable = nearest_id(
                    assigned_points,
                    (
                        (x - reference_width) / current_scale,
                        body_y / current_scale,
                    ),
                    max_distance_px=selection_radius_px,
                )
                if removable is not None:
                    state.remove(removable)
                render()
            return
        if reference_first:
            return
        if event != cv2.EVENT_LBUTTONDOWN or state.selected_node_id is None:
            return
        area_id = nearest_id(
            area_display_points,
            (x, body_y),
            max_distance_px=selection_radius_px * min(scale for _, _, scale in reference_panels),
        )
        if area_id is not None:
            state.assign(state.selected_node_id, area_id)
            render()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, reference_width + current_panel.shape[1], target_height + header_height)
    cv2.setMouseCallback(window, on_mouse)
    render()
    cancelled = False
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("h"), ord("H")):
            show_unassigned = not show_unassigned
            render()
        elif key in (ord("u"), ord("U")):
            if state.undo():
                selected_area_id = None
                render()
        elif key in (ord("s"), ord("S")):
            save_annotations(
                output_path,
                state.assignments,
                graph_path=graph_path,
                visibility_path=visibility_path,
                reference_path=reference_path,
                current_path=current_path,
            )
            render()
        elif key in (ord("q"), ord("Q"), 13):
            save_annotations(
                output_path,
                state.assignments,
                graph_path=graph_path,
                visibility_path=visibility_path,
                reference_path=reference_path,
                current_path=current_path,
            )
            break
        elif key == 27:
            cancelled = True
            break
    cv2.destroyAllWindows()
    if cancelled:
        print("annotation cancelled; no unsaved changes were written")
    else:
        print(f"saved {len(state.assignments)} node labels to {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--visibility", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--reference-side",
        action="append",
        metavar="SIDE=IMAGE",
        help="surface-specific reference image; repeat for FRONT and BACK",
    )
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection-radius-px", type=float, default=28.0)
    parser.add_argument("--max-display-width", type=int, default=1500)
    parser.add_argument("--max-display-height", type=int, default=900)
    parser.add_argument(
        "--display-stride",
        type=int,
        default=1,
        help="show only every Nth DINO-grid row/column; use 4 for a sparse evaluation set",
    )
    parser.add_argument(
        "--reference-first",
        action="store_true",
        help="select a canonical reference area, then click its actual location in current image",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="start empty instead of resuming a compatible output JSON",
    )
    args = parser.parse_args(argv)
    if args.selection_radius_px <= 0:
        parser.error("--selection-radius-px must be positive")
    if args.max_display_width <= 0 or args.max_display_height <= 38:
        parser.error("display dimensions are too small")
    if args.display_stride <= 0:
        parser.error("--display-stride must be positive")
    if args.reference_side:
        reference_path: Path | dict[str, Path] = {}
        for raw in args.reference_side:
            if "=" not in raw:
                parser.error("--reference-side expects SIDE=IMAGE")
            side, path_text = raw.split("=", 1)
            side = side.strip().upper()
            if not side or side in reference_path:
                parser.error(f"invalid or duplicate reference side: {side!r}")
            reference_path[side] = Path(path_text).expanduser().resolve()
    elif args.reference is not None:
        reference_path = args.reference.expanduser().resolve()
    else:
        parser.error("provide --reference or one or more --reference-side SIDE=IMAGE")
    annotate(
        args.graph.expanduser().resolve(),
        args.visibility.expanduser().resolve(),
        reference_path,
        args.current.expanduser().resolve(),
        args.output.expanduser().resolve(),
        selection_radius_px=args.selection_radius_px,
        max_display_width=args.max_display_width,
        max_display_height=args.max_display_height,
        overwrite=args.overwrite,
        display_stride=args.display_stride,
        reference_first=args.reference_first,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
