#!/usr/bin/env python3
"""Zero-shot RGB-D garment fold segmentation preview.

The program does not train a garment-specific model.  It combines the saved
garment mask, table-relative height field, and RGB/depth edges to produce:

* a fold-boundary overlay on the current RGB image;
* a pseudo-coloured partition of visible fold regions; and
* JSON diagnostics describing the detected boundaries and regions.

These are visible-fold hypotheses.  A boundary hidden below another layer is
never claimed to be recovered from a single view.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

try:
    from scipy import ndimage
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("segment_garment_folds.py requires scipy in the active environment") from exc


PALETTE = np.asarray(
    [
        (36, 106, 255),
        (54, 201, 112),
        (232, 148, 42),
        (180, 76, 224),
        (40, 191, 214),
        (224, 80, 102),
        (128, 143, 235),
        (218, 198, 52),
    ],
    dtype=np.uint8,
)


def _load_first(directory: Path, names: list[str]) -> Path:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"none of the required files exist: {names}")


def _normalise(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values) & mask
    output = np.zeros(values.shape, dtype=np.float32)
    if not finite.any():
        return output
    low, high = np.percentile(values[finite], [5.0, 95.0])
    scale = max(float(high - low), 1e-6)
    output[finite] = np.clip((values[finite] - low) / scale, 0.0, 1.0)
    return output


def _gradient(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    finite = np.isfinite(field)
    fill = float(np.nanmedian(field[finite & mask])) if np.any(finite & mask) else 0.0
    smoothed = ndimage.gaussian_filter(np.where(finite, field, fill), sigma=1.2)
    gy, gx = np.gradient(smoothed)
    result = np.hypot(gx, gy).astype(np.float32)
    result[~mask] = 0.0
    return result


def _rgb_gradient(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = (
        0.2126 * rgb[..., 0].astype(np.float32)
        + 0.7152 * rgb[..., 1].astype(np.float32)
        + 0.0722 * rgb[..., 2].astype(np.float32)
    )
    return _gradient(gray, mask)


def _outer_boundary(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    return mask & ~eroded


def _fold_edges(
    height: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    edge_percentile: float,
    rgb_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height_grad = _gradient(height, mask)
    rgb_grad = _rgb_gradient(rgb, mask)
    score = _normalise(height_grad, mask) + float(rgb_weight) * _normalise(rgb_grad, mask)
    finite_score = score[mask & np.isfinite(score)]
    threshold = float(np.percentile(finite_score, edge_percentile)) if finite_score.size else 1.0
    interior = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), iterations=2)
    edges = (score >= threshold) & interior
    # Join nearby pixels into readable fold strokes, then remove tiny noise.
    edges = ndimage.binary_dilation(edges, structure=np.ones((3, 3), dtype=bool), iterations=1)
    labels, count = ndimage.label(edges, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    minimum = max(40, int(np.count_nonzero(mask) * 0.0005))
    keep = np.zeros_like(sizes, dtype=bool)
    for component_id in range(1, count + 1):
        area = int(sizes[component_id])
        if area < minimum:
            continue
        ys, xs = np.where(labels == component_id)
        span = max(
            int(xs.max() - xs.min() + 1) if xs.size else 0,
            int(ys.max() - ys.min() + 1) if ys.size else 0,
        )
        # Keep long fold strokes and genuinely broad boundaries; reject small
        # isolated depth speckles that otherwise look like false fold marks.
        keep[component_id] = span >= 24 or area >= minimum * 3
    edges = keep[labels] & interior
    return edges, score, height_grad


def _visible_regions(mask: np.ndarray, edges: np.ndarray, *, min_area_fraction: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    # Treat the detected fold strokes as barriers.  Connected components of the
    # remaining garment pixels are visible surface regions, not hidden layers.
    barriers = ndimage.binary_dilation(edges, structure=np.ones((3, 3), dtype=bool), iterations=1)
    open_regions = mask & ~barriers
    labels, count = ndimage.label(open_regions, structure=np.ones((3, 3), dtype=np.uint8))
    min_area = max(50, int(np.count_nonzero(mask) * float(min_area_fraction)))
    regions = np.zeros(labels.shape, dtype=np.int32)
    diagnostics: list[dict[str, Any]] = []
    next_id = 1
    for label_id in range(1, count + 1):
        pixels = labels == label_id
        area = int(np.count_nonzero(pixels))
        if area < min_area:
            continue
        yx = np.argwhere(pixels)
        centroid_y, centroid_x = np.mean(yx, axis=0)
        regions[pixels] = next_id
        diagnostics.append(
            {
                "region_id": next_id,
                "area_px": area,
                "centroid_xy": [float(centroid_x), float(centroid_y)],
            }
        )
        next_id += 1
    return regions, diagnostics


def _render(
    rgb: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
    regions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    overlay = rgb.copy()
    outer = _outer_boundary(mask)
    overlay[outer] = (255, 232, 20)  # garment silhouette
    overlay[edges] = (255, 232, 20)  # visible fold boundaries

    region_image = np.full_like(rgb, 245, dtype=np.uint8)
    region_image[mask] = (30, 30, 38)
    for region_id in range(1, int(regions.max()) + 1):
        region_image[regions == region_id] = PALETTE[(region_id - 1) % len(PALETTE)]
    region_image[edges] = (255, 232, 20)
    region_image[outer] = (255, 255, 255)
    return overlay, region_image


def segment(perception_dir: Path, *, camera: str = "A", edge_percentile: float = 90.0, rgb_weight: float = 0.35, min_area_fraction: float = 0.002) -> dict[str, Any]:
    label = camera.strip().upper()
    if label not in {"A", "B"}:
        raise ValueError("camera must be A or B")
    directory = perception_dir.resolve()
    rgb_path = _load_first(directory, [f"camera_0_{label}.png", f"camera_{label}.png"])
    mask_path = directory / f"camera_{label}_garment_mask.npy"
    height_path = directory / f"camera_{label}_height_above_table_mm.npy"
    if not mask_path.is_file() or not height_path.is_file():
        raise FileNotFoundError(
            f"Camera {label} requires {mask_path.name} and {height_path.name}"
        )
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    mask = np.asarray(np.load(mask_path, allow_pickle=False), dtype=bool)
    height = np.asarray(np.load(height_path, allow_pickle=False), dtype=np.float32)
    if rgb.shape[:2] != mask.shape or height.shape != mask.shape:
        raise ValueError(
            f"RGB/mask/height shapes disagree: rgb={rgb.shape[:2]}, mask={mask.shape}, height={height.shape}"
        )
    edges, score, height_gradient = _fold_edges(
        height,
        rgb,
        mask,
        edge_percentile=float(edge_percentile),
        rgb_weight=float(rgb_weight),
    )
    regions, region_diagnostics = _visible_regions(
        mask,
        edges,
        min_area_fraction=float(min_area_fraction),
    )
    overlay, region_image = _render(rgb, mask, edges, regions)
    output_dir = directory / "fold_segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / f"camera_{label}_fold_boundaries_overlay.png"
    regions_path = output_dir / f"camera_{label}_visible_fold_regions.png"
    np.save(output_dir / f"camera_{label}_fold_edge_mask.npy", edges.astype(np.bool_))
    np.save(output_dir / f"camera_{label}_visible_fold_regions.npy", regions.astype(np.int32))
    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(region_image).save(regions_path)
    finite_score = score[mask & np.isfinite(score)]
    report = {
        "status": "READY",
        "camera": label,
        "method": "zero_shot_rgbd_fold_edges_and_visible_region_components",
        "input_rgb": str(rgb_path),
        "input_mask": str(mask_path),
        "input_height": str(height_path),
        "overlay": str(overlay_path),
        "visible_regions": str(regions_path),
        "edge_percentile": float(edge_percentile),
        "rgb_weight": float(rgb_weight),
        "fold_edge_pixel_count": int(np.count_nonzero(edges)),
        "garment_pixel_count": int(np.count_nonzero(mask)),
        "fold_edge_fraction": float(np.count_nonzero(edges) / max(1, np.count_nonzero(mask))),
        "edge_score_p50_p95": [
            float(np.percentile(finite_score, 50.0)) if finite_score.size else None,
            float(np.percentile(finite_score, 95.0)) if finite_score.size else None,
        ],
        "region_count": len(region_diagnostics),
        "regions": region_diagnostics,
        "interpretation": (
            "Boundaries are visible RGB-D fold/occlusion hypotheses. Regions are connected visible "
            "surface components after removing those barriers; hidden garment layers are not recovered."
        ),
    }
    (output_dir / f"camera_{label}_fold_segmentation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception-dir", type=Path, required=True)
    parser.add_argument("--camera", default="A", choices=("A", "B"))
    parser.add_argument("--edge-percentile", type=float, default=95.0)
    parser.add_argument("--rgb-weight", type=float, default=0.15)
    parser.add_argument("--min-area-fraction", type=float, default=0.002)
    args = parser.parse_args(argv)
    report = segment(
        args.perception_dir,
        camera=args.camera,
        edge_percentile=args.edge_percentile,
        rgb_weight=args.rgb_weight,
        min_area_fraction=args.min_area_fraction,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
