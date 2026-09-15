"""Image-plane garment frame; never changes camera/robot calibration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FRAME_RULE = (
    "GARMENT_FRAME_V1: left/right mean viewer-left/right after mentally rotating "
    "the garment so collar is above hem, without mirroring; never wearer anatomy. "
    "The filename upright denotes a fixed camera display rotation, NOT garment alignment. "
    "In displayed pixels let d=(hem-collar)/length and right=(d_y,-d_x). "
    "Left has negative projection onto right; right has positive projection. "
    "Use the supplied current garment_frame for sleeve and torso-side identity. "
    "All executable pixels and Rxxx IDs remain in the displayed camera frame. "
    "Static reference states convey the fold transition only; interpret their collar/hem "
    "orientation independently and never copy their pixels."
)


def image_digest(image: Image.Image) -> str:
    return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()


def build_frame(image: Image.Image, collar, hem) -> dict:
    endpoints = np.asarray([collar, hem], dtype=float)
    if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all():
        raise ValueError('garment frame needs two finite collar/hem pixels')
    if ((endpoints < 0).any() or (endpoints[:, 0] >= image.width).any()
            or (endpoints[:, 1] >= image.height).any()):
        raise ValueError('garment frame endpoints are outside the current image')
    delta = endpoints[1] - endpoints[0]
    length = float(np.linalg.norm(delta))
    if length < max(10., min(image.size) * .05):
        raise ValueError('collar/hem axis is degenerate or too short')
    down = delta / length
    return dict(schema_version=1, convention='GARMENT_FRAME_V1',
                image_sha256=image_digest(image), image_size=list(image.size),
                collar_pixel_xy=endpoints[0].tolist(), hem_pixel_xy=endpoints[1].tolist(),
                down_unit=down.tolist(), right_unit=[float(down[1]), float(-down[0])],
                length_px=length)


def load_frame(directory: Path, image: Image.Image) -> dict:
    data = json.loads((directory / 'garment_frame.json').read_text(encoding='utf-8'))
    rebuilt = build_frame(image, data['collar_pixel_xy'], data['hem_pixel_xy'])
    if any(data.get(k) != rebuilt[k] for k in rebuilt):
        raise ValueError('garment frame is stale or inconsistent with current RGB')
    return rebuilt


def project_pixels(points, frame: dict):
    relative = np.asarray(points, dtype=float) - np.asarray(frame['collar_pixel_xy'])
    return relative @ np.asarray(frame['right_unit']), relative @ np.asarray(frame['down_unit'])


def draw_frame(image: Image.Image, frame: dict, path: Path) -> None:
    overlay = image.convert('RGB').copy()
    draw = ImageDraw.Draw(overlay)
    collar, hem = frame['collar_pixel_xy'], frame['hem_pixel_xy']
    draw.line([tuple(collar), tuple(hem)], fill='yellow', width=3)
    for label, point in [('COLLAR', collar), ('HEM', hem)]:
        x, y = point
        draw.ellipse((x-5, y-5, x+5, y+5), fill='yellow')
        draw.text((x+7, y), label, fill='yellow', stroke_width=1, stroke_fill='black')
    origin = np.asarray(collar) + np.asarray(frame['down_unit']) * frame['length_px'] * .25
    for sign, label, color in [(-1, 'LEFT', 'cyan'), (1, 'RIGHT', 'magenta')]:
        end = origin + sign * np.asarray(frame['right_unit']) * frame['length_px'] * .3
        draw.line([tuple(origin), tuple(end)], fill=color, width=3)
        draw.text(tuple(end), label, fill=color, stroke_width=1, stroke_fill='black')
    overlay.save(path)
