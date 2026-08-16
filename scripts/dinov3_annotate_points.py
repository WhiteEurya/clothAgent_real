#!/usr/bin/env python3
"""Manually annotate reference pixels for the standalone DINOv3 experiment.

This script has no dependency on Claude, xArm, or :mod:`cloth_agent`.  Left
clicks are recorded in original-image coordinates.  Press ``u`` to undo the
last point, ``q`` or Enter to save, and Escape to cancel.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2


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


def annotate(image_path: Path, output_path: Path, count: int, labels: list[str]) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    original_h, original_w = image.shape[:2]
    max_display_w, max_display_h = 1500, 950
    scale = min(1.0, max_display_w / original_w, max_display_h / original_h)
    display_w = max(1, int(round(original_w * scale)))
    display_h = max(1, int(round(original_h * scale)))
    points: list[tuple[int, int]] = []
    window = "DINOv3 reference points"

    def render() -> None:
        canvas = cv2.resize(image, (display_w, display_h), interpolation=cv2.INTER_AREA)
        for index, (x, y) in enumerate(points):
            dx, dy = int(round(x * scale)), int(round(y * scale))
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
        status = f"{len(points)}/{count}  next={next_label} | left click=add  u=undo  q/Enter=save  Esc=cancel"
        cv2.rectangle(canvas, (0, 0), (display_w, 30), (0, 0, 0), -1)
        cv2.putText(canvas, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, canvas)

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < count:
            points.append(
                (
                    min(original_w - 1, max(0, int(round(x / scale)))),
                    min(original_h - 1, max(0, int(round(y / scale)))),
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
    payload = {
        "version": 1,
        "created_at": _now(),
        "image": str(image_path.resolve()),
        "image_width": original_w,
        "image_height": original_h,
        "points": [
            {"id": i + 1, "label": labels[i], "x": x, "y": y}
            for i, (x, y) in enumerate(points)
        ],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(points)} points to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="flat reference RGB image")
    parser.add_argument("--output", required=True, type=Path, help="annotation JSON output")
    parser.add_argument("--num-points", type=int, default=30)
    parser.add_argument("--labels", help="comma-separated labels in click order")
    args = parser.parse_args()
    if args.num_points <= 0:
        parser.error("--num-points must be positive")
    labels = _parse_labels(args.labels, args.num_points)
    annotate(args.image.resolve(), args.output.resolve(), args.num_points, labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
