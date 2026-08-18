#!/usr/bin/env python3
"""Capture and manually annotate reference pixels for the dense-feature experiment.

When ``--image`` is omitted, every launch captures a fresh aligned RGB-D frame
from the selected configured RealSense camera (Camera A by default), saves the
RGB/depth/base-XYZ artifacts, and opens the new RGB image. Passing ``--image``
retains the original offline relabeling mode. No Claude or robot motion is
involved. Left clicks are recorded in original-image coordinates. Press ``u``
to undo the last point, ``q`` or Enter to save, and Escape to cancel.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_LABELS = [
    "left_sleeve_outer",
    "left_sleeve_mid_outer",
    "left_sleeve_mid_inner",
    "left_sleeve_inner",
    "left_sleeve_cuff",
    "right_sleeve_outer",
    "right_sleeve_mid_outer",
    "right_sleeve_mid_inner",
    "right_sleeve_inner",
    "right_sleeve_cuff",
    "torso_left_shoulder",
    "torso_left_upper",
    "torso_left_mid",
    "torso_left_lower",
    "torso_center_upper",
    "torso_center_mid",
    "torso_center_lower",
    "torso_right_upper",
    "torso_right_mid",
    "torso_right_lower",
    "hem_left_outer",
    "hem_left_inner",
    "hem_center_left",
    "hem_center_right",
    "hem_right_inner",
    "collar_left_outer",
    "collar_left_inner",
    "collar_center",
    "collar_right_inner",
    "collar_right_outer",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _capture_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _parse_labels(raw: str | None, count: int) -> list[str]:
    if raw is None:
        labels = list(DEFAULT_LABELS)
        if count != len(labels):
            labels = [f"p{i:02d}" for i in range(1, count + 1)]
        return labels
    labels = [value.strip() for value in raw.split(",") if value.strip()]
    if len(labels) != count:
        raise ValueError(f"--labels contains {len(labels)} labels, expected {count}")
    if len(set(labels)) != len(labels):
        raise ValueError("--labels must be unique")
    return labels


def _load_guide_graph(
    points_path: Path | None,
    neighbors_path: Path | None,
    *,
    image_size: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Load a review guide while keeping it separate from saved BACK topology."""

    if points_path is None:
        if neighbors_path is not None:
            raise ValueError("--guide-neighbors requires --guide-points")
        return [], []
    payload = json.loads(points_path.expanduser().resolve().read_text(encoding="utf-8"))
    raw_points = payload.get("points") if isinstance(payload, dict) else payload
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError("guide points JSON must contain a non-empty points list")
    source_width = int(payload.get("image_width", image_size[0])) if isinstance(payload, dict) else image_size[0]
    source_height = int(payload.get("image_height", image_size[1])) if isinstance(payload, dict) else image_size[1]
    if source_width <= 0 or source_height <= 0:
        raise ValueError("guide image dimensions must be positive")
    x_scale = image_size[0] / source_width
    y_scale = image_size[1] / source_height
    guide_points: list[dict[str, Any]] = []
    known_ids: set[str] = set()
    for index, raw in enumerate(raw_points, start=1):
        if not isinstance(raw, dict):
            raise TypeError(f"guide point {index} must be an object")
        area_id = str(raw.get("area_id", f"G{index:02d}"))
        if area_id in known_ids:
            raise ValueError(f"duplicate guide area ID: {area_id}")
        known_ids.add(area_id)
        guide_points.append(
            {
                "area_id": area_id,
                "index": index,
                "x": float(raw["x"]) * x_scale,
                "y": float(raw["y"]) * y_scale,
            }
        )

    edges: set[tuple[str, str]] = set()
    if neighbors_path is not None:
        neighbor_payload = json.loads(
            neighbors_path.expanduser().resolve().read_text(encoding="utf-8")
        )
        raw_neighbors = (
            neighbor_payload.get("neighbors", neighbor_payload)
            if isinstance(neighbor_payload, dict)
            else neighbor_payload
        )
        if not isinstance(raw_neighbors, dict):
            raise TypeError("guide neighbors JSON must map area IDs to lists")
        for left, values in raw_neighbors.items():
            left_id = str(left)
            if left_id not in known_ids:
                raise ValueError(f"guide edge references unknown area {left_id}")
            if not isinstance(values, list):
                raise TypeError(f"guide neighbors for {left_id} must be a list")
            for value in values:
                right_id = str(value)
                if right_id not in known_ids:
                    raise ValueError(f"guide edge references unknown area {right_id}")
                if left_id != right_id:
                    edges.add(tuple(sorted((left_id, right_id))))
    return guide_points, sorted(edges)


def capture_reference_rgbd(
    project_root: Path,
    perception_config_path: Path,
    camera_label: str,
    capture_path: Path,
) -> dict[str, Any]:
    """Capture one configured RealSense and persist reusable RGB-D artifacts."""

    from cloth_agent.perception import (
        PerceptionConfig,
        camera_base_xyz_map_mm,
        capture_two_view_rgbd,
    )

    project_root = project_root.expanduser().resolve()
    config_path = perception_config_path.expanduser()
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = PerceptionConfig.load(project_root, config_path.resolve())
    label = camera_label.strip().upper()
    configured = {camera.label: camera for camera in config.cameras}
    if label not in configured:
        raise ValueError(
            f"camera {label!r} is not configured; available={sorted(configured)}"
        )
    # The existing capture function supports a one-camera subset. Replacing
    # only this immutable field leaves resolution, exposure, warmup, temporal
    # median, calibration, and all other camera settings authoritative.
    single_camera_config = replace(config, active_camera_labels=(label,))
    frames = capture_two_view_rgbd(single_camera_config)
    if len(frames) != 1 or frames[0].label != label:
        raise RuntimeError(
            f"single-camera capture returned {[frame.label for frame in frames]}"
        )
    frame = frames[0]
    xyz_map, xyz_valid = camera_base_xyz_map_mm(frame, single_camera_config)

    capture_path = capture_path.expanduser().resolve()
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame.rgb.astype(np.uint8)).save(capture_path)
    depth_path = capture_path.with_name(f"{capture_path.stem}_depth_m.npy")
    xyz_path = capture_path.with_name(f"{capture_path.stem}_base_xyz_mm.npy")
    valid_path = capture_path.with_name(f"{capture_path.stem}_xyz_valid.npy")
    metadata_path = capture_path.with_suffix(".json")
    np.save(depth_path, frame.depth_m.astype(np.float32))
    np.save(xyz_path, xyz_map.astype(np.float32))
    np.save(valid_path, xyz_valid.astype(bool))
    metadata = {
        "version": 1,
        "captured_at": _now(),
        "camera_label": frame.label,
        "camera_serial": frame.serial,
        "perception_config": str(config_path.resolve()),
        "rgb": str(capture_path),
        "depth_m": str(depth_path),
        "base_xyz_mm": str(xyz_path),
        "xyz_valid": str(valid_path),
        "image_width": int(frame.rgb.shape[1]),
        "image_height": int(frame.rgb.shape[0]),
        "intrinsics": frame.intrinsics.tolist(),
        "X_base_camera": frame.X_base_camera.tolist(),
        "valid_xyz_pixels": int(np.count_nonzero(xyz_valid)),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata["metadata_json"] = str(metadata_path)
    print(
        f"captured Camera {frame.label} serial={frame.serial} "
        f"to {capture_path} ({metadata['valid_xyz_pixels']} valid XYZ pixels)"
    )
    return metadata


def annotate(
    image_path: Path,
    output_path: Path,
    count: int,
    labels: list[str],
    *,
    capture_metadata: Mapping[str, Any] | None = None,
    surface_side: str = "FRONT",
    side_marker: str | None = None,
    guide_points_path: Path | None = None,
    guide_neighbors_path: Path | None = None,
    guide_opacity: float = 0.58,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    original_h, original_w = image.shape[:2]
    max_display_w, max_display_h = 1500, 950
    scale = min(1.0, max_display_w / original_w, max_display_h / original_h)
    display_w = max(1, round(original_w * scale))
    display_h = max(1, round(original_h * scale))
    points: list[tuple[int, int]] = []
    side = surface_side.strip().upper()
    if side not in {"FRONT", "BACK"}:
        raise ValueError("surface_side must be FRONT or BACK")
    if not 0.0 <= guide_opacity <= 1.0:
        raise ValueError("guide_opacity must be between 0 and 1")
    area_prefix = "F" if side == "FRONT" else "B"
    window = f"Dense reference points - {side}"
    guide_points, guide_edges = _load_guide_graph(
        guide_points_path,
        guide_neighbors_path,
        image_size=(original_w, original_h),
    )
    guide_by_id = {str(point["area_id"]): point for point in guide_points}

    def render() -> None:
        canvas = cv2.resize(image, (display_w, display_h), interpolation=cv2.INTER_AREA)
        if guide_points:
            guide_layer = canvas.copy()
            for left_id, right_id in guide_edges:
                left, right = guide_by_id[left_id], guide_by_id[right_id]
                cv2.line(
                    guide_layer,
                    (round(float(left["x"]) * scale), round(float(left["y"]) * scale)),
                    (round(float(right["x"]) * scale), round(float(right["y"]) * scale)),
                    (255, 190, 0),
                    3,
                    cv2.LINE_AA,
                )
            for guide in guide_points:
                index = int(guide["index"])
                dx = round(float(guide["x"]) * scale)
                dy = round(float(guide["y"]) * scale)
                is_next = index == len(points) + 1
                color = (40, 255, 40) if is_next else (255, 190, 0)
                radius = 10 if is_next else 7
                cv2.circle(guide_layer, (dx, dy), radius, color, -1, cv2.LINE_AA)
                cv2.putText(
                    guide_layer,
                    f"{index:02d}",
                    (dx + 9, dy - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            canvas = cv2.addWeighted(
                canvas,
                1.0 - guide_opacity,
                guide_layer,
                guide_opacity,
                0.0,
            )
        for index, (x, y) in enumerate(points):
            dx, dy = round(x * scale), round(y * scale)
            color = (0, 220, 255)
            cv2.circle(canvas, (dx, dy), 6, color, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(index + 1),
                (dx + 8, dy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        next_label = labels[len(points)] if len(points) < count else "done"
        marker_text = f" marker={side_marker}" if side_marker else ""
        guide_text = " guide=FRONT_LAYOUT_ONLY" if guide_points else ""
        status = (
            f"{side}{marker_text} {len(points)}/{count} next={next_label} | "
            f"{guide_text} left click=add  u=undo  q/Enter=save  Esc=cancel"
        )
        cv2.rectangle(canvas, (0, 0), (display_w, 30), (0, 0, 0), -1)
        cv2.putText(canvas, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, canvas)

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < count:
            points.append(
                (
                    min(original_w - 1, max(0, round(x / scale))),
                    min(original_h - 1, max(0, round(y / scale))),
                )
            )
            render()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, display_w, display_h)
    cv2.setMouseCallback(window, on_mouse)
    render()
    cancelled = False
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("u"), ord("U")) and points:
            points.pop()
            render()
        elif key in (ord("q"), ord("Q"), 13) and len(points) == count:
            break
        elif key == 27:
            cancelled = True
            break
    cv2.destroyAllWindows()
    if cancelled:
        raise KeyboardInterrupt("annotation cancelled")
    if len(points) != count:
        raise RuntimeError(f"expected {count} points, got {len(points)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xyz_map = None
    if capture_metadata is not None and capture_metadata.get("base_xyz_mm"):
        xyz_map = np.asarray(
            np.load(Path(str(capture_metadata["base_xyz_mm"]))),
            dtype=np.float32,
        )
        if xyz_map.shape != (original_h, original_w, 3):
            raise ValueError(
                f"captured base XYZ shape {xyz_map.shape} does not match "
                f"image {(original_h, original_w)}"
            )
    point_payloads: list[dict[str, Any]] = []
    for index, (x, y) in enumerate(points):
        point_payload: dict[str, Any] = {
            "id": index + 1,
            "area_id": f"{area_prefix}{index + 1:02d}",
            "surface_side": side,
            "label": labels[index],
            "x": x,
            "y": y,
        }
        if xyz_map is not None:
            base_xyz = xyz_map[y, x]
            if np.all(np.isfinite(base_xyz)):
                point_payload["base_xyz_mm"] = [
                    float(value) for value in base_xyz
                ]
        point_payloads.append(point_payload)
    payload = {
        "version": 1,
        "created_at": _now(),
        "image": str(image_path.resolve()),
        "image_width": original_w,
        "image_height": original_h,
        "surface_side": side,
        "side_marker": side_marker,
        "points": point_payloads,
    }
    if capture_metadata is not None:
        payload["capture"] = dict(capture_metadata)
    if guide_points_path is not None:
        payload["layout_guide"] = {
            "points_json": str(guide_points_path.expanduser().resolve()),
            "neighbors_json": (
                None
                if guide_neighbors_path is None
                else str(guide_neighbors_path.expanduser().resolve())
            ),
            "semantic_note": "display-only layout guide; no FRONT-BACK topology edges implied",
        }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(points)} points to {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        help="existing flat reference RGB; omit to capture a fresh configured camera frame",
    )
    parser.add_argument("--output", required=True, type=Path, help="annotation JSON output")
    parser.add_argument("--num-points", type=int, default=30)
    parser.add_argument("--labels", help="comma-separated labels in click order")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--perception-config",
        type=Path,
        default=Path("config/perception.free_exploration.json"),
    )
    parser.add_argument("--camera-label", default="A")
    parser.add_argument(
        "--surface-side",
        choices=("FRONT", "BACK"),
        default="FRONT",
        help="garment exterior surface being captured and annotated",
    )
    parser.add_argument(
        "--side-marker",
        help="human-visible cue defining this side, e.g. COLLAR_LABEL_VISIBLE",
    )
    parser.add_argument(
        "--capture-output",
        type=Path,
        help="optional RGB path for live capture; default is a timestamped file beside --output",
    )
    parser.add_argument(
        "--guide-points",
        type=Path,
        help="optional prior point JSON drawn over the newly captured image as a layout guide",
    )
    parser.add_argument(
        "--guide-neighbors",
        type=Path,
        help="optional neighbor JSON drawn with --guide-points; display only",
    )
    parser.add_argument("--guide-opacity", type=float, default=0.58)
    args = parser.parse_args(argv)
    if args.num_points <= 0:
        parser.error("--num-points must be positive")
    output_path = args.output.expanduser().resolve()
    capture_metadata = None
    if args.image is not None:
        image_path = args.image.expanduser().resolve()
    else:
        label = args.camera_label.strip().upper()
        side = args.surface_side.strip().upper()
        image_path = (
            args.capture_output.expanduser().resolve()
            if args.capture_output is not None
            else output_path.with_name(
                f"{output_path.stem}_{side.lower()}_camera_{label}_{_capture_stamp()}.png"
            )
        )
        capture_metadata = capture_reference_rgbd(
            args.project_root,
            args.perception_config,
            label,
            image_path,
        )
        capture_metadata["surface_side"] = side
        capture_metadata["side_marker"] = args.side_marker
    labels = _parse_labels(args.labels, args.num_points)
    annotate(
        image_path,
        output_path,
        args.num_points,
        labels,
        capture_metadata=capture_metadata,
        surface_side=args.surface_side,
        side_marker=args.side_marker,
        guide_points_path=args.guide_points,
        guide_neighbors_path=args.guide_neighbors,
        guide_opacity=args.guide_opacity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
