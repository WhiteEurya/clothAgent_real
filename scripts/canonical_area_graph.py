#!/usr/bin/env python3
"""Build and evaluate a canonical garment-area graph prototype.

This is intentionally offline and perception-only.  It reuses the existing
``scripts/dinov3_annotate_points.py`` point JSON as the minimal canonical
surface representation, turns each point into a local area feature bank, and
compares feature-only against feature+topology matching.

Examples::

    python scripts/canonical_area_graph.py build \
      --reference flat.png --points flat_points.json \
      --output results/canonical_graph/reference_01 \
      --local-files-only

    python scripts/canonical_area_graph.py match \
      --graph results/canonical_graph/reference_01/canonical_graph.json \
      --reference flat.png --current folded=folded.png \
      --output results/canonical_graph/reference_01/matches \
      --local-files-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.canonical_area_graph import (
    CanonicalSurfaceGraph,
    evaluate_visibility,
    match_visibility,
    render_visibility_visualization,
)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._") or "current"


def _parse_named_paths(values: list[str], *, kind: str) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    used: set[str] = set()
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"--{kind} expects NAME=PATH, got {raw!r}")
        name, path_text = raw.split("=", 1)
        name = name.strip()
        path = Path(path_text).expanduser().resolve()
        if not name or name in used:
            raise ValueError(f"invalid or duplicate {kind} name: {name!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        used.add(name)
        parsed.append((name, path))
    return parsed


def _load_points(path: Path, image_size: tuple[int, int]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_points = payload.get("points") if isinstance(payload, dict) else payload
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError(f"{path} does not contain a non-empty points list")
    width, height = image_size
    points: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_points, start=1):
        if not isinstance(raw, dict):
            raise TypeError(f"point {index} is not an object")
        x, y = float(raw["x"]), float(raw["y"])
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"point {index} ({x}, {y}) outside {width}x{height}")
        point = dict(raw)
        point["id"] = int(raw.get("id", index))
        point["label"] = str(raw.get("label", f"p{index:02d}"))
        point["x"], point["y"] = x, y
        points.append(point)
    return points


def _load_neighbors(path: Path | None) -> dict[str, list[str]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("neighbors", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw, dict):
        raise TypeError("neighbors JSON must be an object mapping area IDs to lists")
    return {str(key): [str(item) for item in value] for key, value in raw.items()}


def _load_mask(path: Path | None, feature_shape: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None
    image = Image.open(path).convert("L")
    resized = image.resize((feature_shape[1], feature_shape[0]), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def _load_surface_xyz(path: Path | None, feature_shape: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None
    xyz = np.asarray(np.load(path), dtype=np.float32)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"{path} must contain an (H, W, 3) XYZ array")
    if xyz.shape[:2] == feature_shape:
        return xyz
    ys = np.rint(np.linspace(0, xyz.shape[0] - 1, feature_shape[0])).astype(np.int64)
    xs = np.rint(np.linspace(0, xyz.shape[1] - 1, feature_shape[1])).astype(np.int64)
    return xyz[np.ix_(ys, xs)]


def _save_region_visualization(
    reference: Image.Image,
    graph: CanonicalSurfaceGraph,
    output_path: Path,
) -> Path | None:
    if not graph.region_labels_by_side:
        return None
    if len(graph.region_labels_by_side) != 1:
        return None
    labels = next(iter(graph.region_labels_by_side.values()))
    area_ids = list(graph.areas)
    palette = np.zeros((len(area_ids), 3), dtype=np.uint8)
    for index in range(len(area_ids)):
        palette[index] = (
            (53 * index + 70) % 205 + 30,
            (97 * index + 40) % 205 + 30,
            (149 * index + 10) % 205 + 30,
        )
    region_rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    valid = labels >= 0
    region_rgb[valid] = palette[labels[valid]]
    region_image = Image.fromarray(region_rgb).resize(reference.size, Image.Resampling.NEAREST)
    overlay = Image.blend(reference.convert("RGB"), region_image, 0.34)
    draw = ImageDraw.Draw(overlay)
    for area_id, area in graph.areas.items():
        x, y = area.canonical_xy
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(255, 220, 0))
        draw.text(
            (x + 8, y - 12),
            f"{area_id} ({len(area.region_patch_xy)})",
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
    return output_path


def _otsu_threshold(values: np.ndarray) -> int:
    pixels = np.asarray(values, dtype=np.uint8).reshape(-1)
    histogram = np.bincount(pixels, minlength=256).astype(np.float64)
    total = float(histogram.sum())
    if total <= 0:
        return 0
    probability = histogram / total
    cumulative = np.cumsum(probability)
    cumulative_mean = np.cumsum(probability * np.arange(256, dtype=np.float64))
    global_mean = cumulative_mean[-1]
    denominator = cumulative * (1.0 - cumulative)
    variance = np.zeros(256, dtype=np.float64)
    valid = denominator > 1e-12
    variance[valid] = (
        (global_mean * cumulative[valid] - cumulative_mean[valid]) ** 2
        / denominator[valid]
    )
    return int(np.argmax(variance))


def prepare_reference_mask(args: argparse.Namespace) -> int:
    """Extract the largest non-table component for offline flat references."""

    from scipy.ndimage import binary_closing, binary_fill_holes, label

    image_path = args.image.expanduser().resolve()
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    border = max(2, round(min(height, width) * 0.05))
    border_pixels = np.concatenate(
        [
            rgb[:border].reshape(-1, 3),
            rgb[-border:].reshape(-1, 3),
            rgb[:, :border].reshape(-1, 3),
            rgb[:, -border:].reshape(-1, 3),
        ],
        axis=0,
    )
    table_rgb = np.median(border_pixels.astype(np.float32), axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - table_rgb[None, None, :], axis=2)
    scale = max(1.0, float(np.percentile(distance, 99)))
    distance_u8 = np.rint(np.clip(distance / scale, 0.0, 1.0) * 255.0).astype(np.uint8)
    threshold = _otsu_threshold(distance_u8)
    candidate = distance_u8 > threshold
    components, count = label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
    if count <= 0:
        raise ValueError("reference mask extraction found no non-table component")
    values, counts = np.unique(components[components > 0], return_counts=True)
    largest = int(values[int(np.argmax(counts))])
    mask = components == largest
    mask = binary_closing(mask, structure=np.ones((5, 5), dtype=bool), iterations=2)
    mask = binary_fill_holes(mask)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(output_path)
    overlay = rgb.copy()
    overlay[~mask] = np.rint(0.25 * overlay[~mask]).astype(np.uint8)
    overlay_path = output_path.with_name(f"{output_path.stem}_overlay.png")
    Image.fromarray(overlay).save(overlay_path)
    diagnostics = {
        "source_image": str(image_path),
        "mask": str(output_path),
        "overlay": str(overlay_path),
        "method": "border_table_color_distance_otsu_largest_component",
        "table_rgb_median": [float(value) for value in table_rgb],
        "otsu_threshold_u8": threshold,
        "garment_pixels": int(mask.sum()),
        "garment_fraction": float(mask.mean()),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return 0


def _propose_planar_neighbors(
    points: list[dict[str, Any]],
    *,
    max_edge_length_mm: float | None,
    max_edge_factor: float,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Propose local reference edges without treating them as ground truth."""

    from scipy.spatial import Delaunay
    from scipy.spatial._qhull import QhullError

    if len(points) < 3:
        raise ValueError("at least three reference points are required for Delaunay adjacency")
    if max_edge_length_mm is not None and max_edge_length_mm <= 0:
        raise ValueError("max edge length must be positive")
    if max_edge_factor <= 0:
        raise ValueError("max edge factor must be positive")

    area_ids: list[str] = []
    base_xy: list[tuple[float, float]] = []
    for point in points:
        area_id = str(point.get("area_id", f"A{int(point['id']):02d}"))
        raw_xyz = point.get("base_xyz_mm")
        if raw_xyz is None or len(raw_xyz) != 3:
            raise ValueError(f"{area_id} lacks calibrated base_xyz_mm")
        xyz = np.asarray(raw_xyz, dtype=np.float64)
        if not np.isfinite(xyz).all():
            raise ValueError(f"{area_id} has invalid calibrated base_xyz_mm")
        area_ids.append(area_id)
        base_xy.append((float(xyz[0]), float(xyz[1])))

    coordinates = np.asarray(base_xy, dtype=np.float64)
    try:
        triangulation = Delaunay(coordinates)
    except QhullError as exc:
        raise ValueError("reference base-XY points cannot form a planar triangulation") from exc

    all_edges: set[tuple[int, int]] = set()
    for triangle in triangulation.simplices:
        for left, right in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            all_edges.add(tuple(sorted((int(left), int(right)))))
    lengths = {
        edge: float(np.linalg.norm(coordinates[edge[0]] - coordinates[edge[1]]))
        for edge in all_edges
    }
    median_length_mm = float(np.median(list(lengths.values())))
    effective_limit_mm = (
        float(max_edge_length_mm)
        if max_edge_length_mm is not None
        else median_length_mm * float(max_edge_factor)
    )
    retained = sorted(edge for edge, length in lengths.items() if length <= effective_limit_mm)
    neighbors: dict[str, list[str]] = {area_id: [] for area_id in area_ids}
    edge_records: list[dict[str, Any]] = []
    for left, right in retained:
        left_id, right_id = area_ids[left], area_ids[right]
        neighbors[left_id].append(right_id)
        neighbors[right_id].append(left_id)
        edge_records.append(
            {
                "area_ids": [left_id, right_id],
                "length_mm": lengths[(left, right)],
            }
        )
    for values in neighbors.values():
        values.sort()
    isolated = [area_id for area_id, values in neighbors.items() if not values]
    if isolated:
        raise ValueError(
            "edge-length threshold isolates canonical areas; increase it before review: "
            + ", ".join(isolated)
        )
    diagnostics = {
        "proposal_only": True,
        "coordinate_space": "calibrated_base_xy_mm",
        "method": "delaunay_with_physical_length_threshold",
        "median_delaunay_edge_length_mm": median_length_mm,
        "max_edge_factor": float(max_edge_factor),
        "max_edge_length_mm": effective_limit_mm,
        "delaunay_edge_count": len(all_edges),
        "retained_edge_count": len(retained),
        "edges": edge_records,
    }
    return neighbors, diagnostics


def propose_neighbors(args: argparse.Namespace) -> int:
    """Write an explicitly unapproved reference-edge proposal and overlay."""

    reference_path = args.reference.expanduser().resolve()
    points_path = args.points.expanduser().resolve()
    reference = Image.open(reference_path).convert("RGB")
    points = _load_points(points_path, reference.size)
    neighbors, diagnostics = _propose_planar_neighbors(
        points,
        max_edge_length_mm=args.max_edge_length_mm,
        max_edge_factor=args.max_edge_factor,
    )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        **diagnostics,
        "neighbors": neighbors,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    point_by_id = {
        str(point.get("area_id", f"A{int(point['id']):02d}")): point
        for point in points
    }
    overlay = reference.copy()
    draw = ImageDraw.Draw(overlay)
    for left, right_values in neighbors.items():
        for right in right_values:
            if left >= right:
                continue
            left_point, right_point = point_by_id[left], point_by_id[right]
            draw.line(
                [(float(left_point["x"]), float(left_point["y"])),
                 (float(right_point["x"]), float(right_point["y"]))],
                fill=(0, 210, 255),
                width=3,
            )
    radius = 7
    for area_id, point in point_by_id.items():
        x, y = float(point["x"]), float(point["y"])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 210, 0))
        draw.text((x + 8, y - 12), area_id, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    visualization_path = args.visualization.expanduser().resolve()
    visualization_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(visualization_path)
    print(
        json.dumps(
            {
                "candidate_neighbors": str(output_path),
                "visualization": str(visualization_path),
                **diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _extract(extractor: Any, path: Path) -> np.ndarray:
    return extractor.encode(Image.open(path).convert("RGB")).detach().cpu().numpy().astype(np.float32)


def _extractor(args: argparse.Namespace) -> Any:
    try:
        from scripts.dinov3_correspondence import FrozenDINOv3
    except ImportError as exc:  # pragma: no cover - malformed installation
        raise RuntimeError(
            "scripts/dinov3_correspondence.py is required for model-backed features"
        ) from exc
    return FrozenDINOv3(
        args.model,
        input_size=args.input_size,
        input_width=args.input_width,
        input_height=args.input_height,
        device=args.device,
        local_files_only=args.local_files_only,
    )


def _save_ambiguity_crops(
    current_image: np.ndarray,
    ambiguities: list[dict[str, Any]],
    *,
    output_dir: Path,
    slug: str,
    crop_size: int,
) -> None:
    """Materialize small image packets for future ambiguity-only reasoning."""

    if crop_size <= 0:
        raise ValueError("ambiguity crop size must be positive")
    if not ambiguities:
        return
    crop_dir = output_dir / f"{slug}_ambiguity_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    image_h, image_w = current_image.shape[:2]
    half = crop_size / 2.0
    for ambiguity in ambiguities:
        center_x, center_y = map(float, ambiguity["pixel_xy"])
        left = max(0, round(center_x - half))
        top = max(0, round(center_y - half))
        right = min(image_w, max(left + 1, round(center_x + half)))
        bottom = min(image_h, max(top + 1, round(center_y + half)))
        crop = current_image[top:bottom, left:right]
        crop_name = f"{ambiguity['current_node_id']}.png"
        crop_path = crop_dir / crop_name
        Image.fromarray(crop.astype(np.uint8)).save(crop_path)
        ambiguity["current_crop"] = {
            "path": str(crop_path.relative_to(output_dir)),
            "bbox_xyxy": [left, top, right, bottom],
            "center_xy": [center_x, center_y],
        }


def prepare_mask(args: argparse.Namespace) -> int:
    """Project an existing fused garment mask into one saved RGB-D view."""

    from cloth_agent.perception import RGBDFrame, _occlusion_aware_garment_mask

    result_path = args.result.expanduser().resolve()
    result_dir = result_path.parent
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    views = payload.get("views")
    if not isinstance(views, list):
        raise TypeError("perception result does not contain a views list")
    view = next(
        (
            raw
            for raw in views
            if isinstance(raw, dict) and str(raw.get("label")) == args.camera_label
        ),
        None,
    )
    if view is None:
        raise ValueError(f"perception result has no camera {args.camera_label!r}")
    artifacts = payload.get("depth_fusion", {}).get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise TypeError("perception result does not contain depth_fusion artifacts")
    fused_points_name = artifacts.get("fused_points_base_mm")
    fused_mask_name = artifacts.get("fused_garment_mask")
    if not fused_points_name or not fused_mask_name:
        raise ValueError("perception result lacks fused points or fused garment mask")

    image_path = result_dir / str(view["image"])
    depth_path = result_dir / str(view["depth_m"])
    height_path = result_dir / str(view["height_map_path"])
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    depth_m = np.asarray(np.load(depth_path), dtype=np.float32)
    height_above_table_mm = np.asarray(np.load(height_path), dtype=np.float32)
    if rgb.shape[:2] != depth_m.shape or depth_m.shape != height_above_table_mm.shape:
        raise ValueError("saved RGB, depth, and height-map shapes do not match")
    frame = RGBDFrame(
        label=str(view["label"]),
        serial=str(view.get("serial", "unknown")),
        rgb=rgb,
        depth_m=depth_m,
        intrinsics=np.asarray(view["intrinsics"], dtype=np.float64),
        X_base_camera=np.asarray(view["X_base_camera"], dtype=np.float64),
    )
    fused_points = np.asarray(
        np.load(result_dir / str(fused_points_name)),
        dtype=np.float64,
    )
    fused_mask = np.asarray(
        np.load(result_dir / str(fused_mask_name)),
        dtype=bool,
    )
    if fused_points.ndim != 2 or fused_points.shape[1] != 3:
        raise ValueError("fused points must have shape (N, 3)")
    if fused_mask.shape != (len(fused_points),):
        raise ValueError("fused garment mask must match fused point count")
    valid_depth = np.isfinite(depth_m) & (depth_m > 0.0)
    garment_mask, sparse_mask, diagnostics = _occlusion_aware_garment_mask(
        fused_points[fused_mask],
        frame,
        height_above_table_mm,
        valid_depth,
    )
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(garment_mask.astype(np.uint8) * 255).save(output_path)
    sparse_path = output_path.with_name(f"{output_path.stem}_projected_sparse.png")
    Image.fromarray(sparse_mask.astype(np.uint8) * 255).save(sparse_path)
    diagnostics_path = output_path.with_suffix(".json")
    diagnostics_payload = {
        "source_result": str(result_path),
        "camera_label": args.camera_label,
        "rgb_image": str(image_path),
        "mask": str(output_path),
        "projected_sparse_mask": str(sparse_path),
        **diagnostics,
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics_payload, ensure_ascii=False, indent=2))
    return 0


def build(args: argparse.Namespace) -> int:
    if args.neighbors_approved and args.neighbors is None:
        raise ValueError("--neighbors-approved requires --neighbors")
    reference_path = args.reference.expanduser().resolve()
    points_path = args.points.expanduser().resolve()
    reference = Image.open(reference_path).convert("RGB")
    extractor = None
    if args.reference_features is not None:
        reference_features = np.asarray(np.load(args.reference_features.expanduser().resolve()), dtype=np.float32)
        feature_source = "precomputed_dense_feature_map"
    else:
        extractor = _extractor(args)
        reference_features = _extract(extractor, reference_path)
        feature_source = "FrozenDINOv3 adapter"
    points = _load_points(points_path, reference.size)
    graph = CanonicalSurfaceGraph.from_reference_points(
        points,
        reference_features,
        image_size=reference.size,
        patch_size=getattr(extractor, "patch_size", None) if extractor is not None else None,
        sample_radius=args.sample_radius,
        neighbor_k=args.neighbor_k,
        explicit_neighbors=_load_neighbors(args.neighbors),
        reference_mask=_load_mask(args.reference_mask, reference_features.shape[:2]),
    )
    graph.metadata.update(
        {
            "reference_image": str(reference_path),
            "points_json": str(points_path),
            "model": (
                args.model if extractor is not None else args.feature_model_id
            ),
            "input_size": (
                args.input_size
                if extractor is not None or args.feature_model_id is not None
                else None
            ),
            "input_shape_yx": (
                [extractor.input_height, extractor.input_width]
                if extractor is not None
                else [args.input_height, args.input_width]
                if args.feature_model_id is not None
                else None
            ),
            "device": args.device if extractor is not None else None,
            "feature_extractor": feature_source + "; replaceable dense feature API",
            "neighbors_json": (
                None if args.neighbors is None else str(args.neighbors.expanduser().resolve())
            ),
            "adjacency_review_status": (
                "user_approved" if args.neighbors_approved else "not_recorded"
            ),
            "reference_mask": (
                None
                if args.reference_mask is None
                else str(args.reference_mask.expanduser().resolve())
            ),
        }
    )
    output_dir = args.output.expanduser().resolve()
    graph_path, feature_path = graph.save(output_dir)
    region_visualization = _save_region_visualization(
        reference,
        graph,
        output_dir / "canonical_regions.png",
    )
    print(
        json.dumps(
            {
                "canonical_graph": str(graph_path),
                "feature_bank": str(feature_path),
                "areas": len(graph.areas),
                "canonical_regions": (
                    None if region_visualization is None else str(region_visualization)
                ),
            },
            indent=2,
        )
    )
    return 0


def combine(args: argparse.Namespace) -> int:
    """Combine separately approved surface graphs without automatic seam edges."""

    named_graphs = _parse_named_paths(args.graph, kind="graph")
    graphs = [CanonicalSurfaceGraph.load(path) for _, path in named_graphs]
    combined = CanonicalSurfaceGraph.combine(graphs)
    combined.metadata["component_graph_paths"] = {
        name: str(path) for name, path in named_graphs
    }
    graph_path, feature_path = combined.save(args.output.expanduser().resolve())
    edge_count = sum(len(area.neighbor_area_ids) for area in combined.areas.values()) // 2
    print(
        json.dumps(
            {
                "canonical_graph": str(graph_path),
                "feature_bank": str(feature_path),
                "areas": len(combined.areas),
                "edges": edge_count,
                "surface_sides": combined.metadata["surface_sides"],
                "automatic_cross_surface_edges": False,
            },
            indent=2,
        )
    )
    return 0


def match(args: argparse.Namespace) -> int:
    graph_path = args.graph.expanduser().resolve()
    graph = CanonicalSurfaceGraph.load(graph_path)
    if args.reference_side:
        reference_paths = _parse_named_paths(args.reference_side, kind="reference-side")
        reference: np.ndarray | dict[str, np.ndarray] = {
            name.strip().upper(): np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            for name, path in reference_paths
        }
    elif args.reference is not None:
        reference_path = args.reference.expanduser().resolve()
        reference = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.uint8)
    else:
        raise ValueError("match requires --reference or one or more --reference-side SIDE=IMAGE")
    currents = _parse_named_paths(args.current, kind="current")
    masks = dict(_parse_named_paths(args.mask, kind="mask")) if args.mask else {}
    surface_xyz_paths = (
        dict(_parse_named_paths(args.surface_xyz, kind="surface-xyz"))
        if args.surface_xyz
        else {}
    )
    current_feature_paths = (
        dict(_parse_named_paths(args.current_features, kind="current-features"))
        if args.current_features
        else {}
    )
    extractor = None
    missing_precomputed = [name for name, _ in currents if name not in current_feature_paths]
    if missing_precomputed:
        extractor = _extractor(args)
        if extractor.grid_h != graph.feature_shape[0] or extractor.grid_w != graph.feature_shape[1]:
            raise ValueError(
                f"dense feature grid mismatch: extractor={(extractor.grid_h, extractor.grid_w)} graph={graph.feature_shape}"
            )
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    ground_truths = dict(_parse_named_paths(args.ground_truth, kind="ground-truth")) if args.ground_truth else {}
    for name, path in currents:
        current_image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        if name in current_feature_paths:
            current_features = np.asarray(np.load(current_feature_paths[name]), dtype=np.float32)
        else:
            assert extractor is not None
            current_features = _extract(extractor, path)
        current_feature_shape = (int(current_features.shape[0]), int(current_features.shape[1]))
        valid_mask = _load_mask(masks.get(name), current_feature_shape)
        surface_xyz = _load_surface_xyz(surface_xyz_paths.get(name), current_feature_shape)
        result = match_visibility(
            graph,
            current_features,
            image_size=(current_image.shape[1], current_image.shape[0]),
            valid_mask=valid_mask,
            surface_xyz_mm=surface_xyz,
            max_surface_gap_mm=args.max_surface_gap_mm,
            top_k=args.top_k,
            topology_lambda=args.topology_lambda,
            ambiguity_margin=args.ambiguity_margin,
            confident_threshold=args.confident_threshold,
            minimum_similarity=args.minimum_similarity,
            adjacency_similarity_threshold=args.adjacency_similarity_threshold,
            current_sample_radius=args.current_sample_radius,
        )
        slug = _slug(name)
        result.parameters["current_image"] = str(path)
        _save_ambiguity_crops(
            current_image,
            result.ambiguities,
            output_dir=output_dir,
            slug=slug,
            crop_size=args.ambiguity_crop_size,
        )
        result_path = output_dir / f"{slug}_visibility.json"
        result_path.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        visualization_path = output_dir / f"{slug}_canonical_visibility.png"
        render_visibility_visualization(reference, current_image, graph, result, output_path=visualization_path)
        summary: dict[str, Any] = {
            "visibility": result_path.name,
            "visualization": visualization_path.name,
            "frontier_count": len(result.frontiers),
            "ambiguity_count": len(result.ambiguities),
            "area_status_counts": {
                status: sum(value["status"] == status for value in result.canonical_areas.values())
                for status in ("VISIBLE", "AMBIGUOUS", "UNOBSERVED")
            },
            "observation_adjacency_counts": {
                relation: sum(
                    adjacency["relation"] == relation
                    for adjacency in result.observation_adjacencies
                )
                for relation in (
                    "MATERIAL_CONSISTENT",
                    "UNEXPECTED_ADJACENCY",
                    "AMBIGUOUS",
                )
            },
        }
        if name in ground_truths:
            truth_payload = json.loads(ground_truths[name].read_text(encoding="utf-8"))
            raw_truth = truth_payload.get("ground_truth", truth_payload)
            if not isinstance(raw_truth, dict):
                raise ValueError(f"ground truth for {name} must be a node_id -> area_id object")
            truth = {str(node_id): str(area_id) for node_id, area_id in raw_truth.items()}
            unknown_truth_areas = sorted(set(truth.values()) - set(graph.areas))
            if unknown_truth_areas:
                raise ValueError(
                    f"ground truth for {name} references unknown canonical areas: "
                    f"{unknown_truth_areas}"
                )
            summary["evaluation"] = evaluate_visibility(result.current_nodes, truth, recall_ks=(3, 5))
        summaries[name] = summary
    (output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build a canonical graph from flat reference points")
    build_parser.add_argument("--reference", required=True, type=Path)
    build_parser.add_argument("--points", required=True, type=Path)
    build_parser.add_argument("--output", required=True, type=Path)
    build_parser.add_argument("--neighbors", type=Path, help="optional explicit area adjacency JSON")
    build_parser.add_argument(
        "--reference-mask",
        type=Path,
        help="garment mask used for constrained/geodesic Voronoi canonical regions",
    )
    build_parser.add_argument(
        "--neighbors-approved",
        action="store_true",
        help="record that the supplied explicit neighbors were manually approved",
    )
    build_parser.add_argument(
        "--reference-features",
        type=Path,
        help="optional precomputed (H,W,D) feature map; bypasses DINOv3",
    )
    build_parser.add_argument(
        "--feature-model-id",
        help="provenance label for --reference-features, e.g. facebook/dinov2-small",
    )
    build_parser.add_argument("--sample-radius", type=int, default=1)
    build_parser.add_argument("--neighbor-k", type=int, default=4)
    build_parser.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    build_parser.add_argument("--input-size", type=int, default=448)
    build_parser.add_argument("--input-width", type=int, default=448)
    build_parser.add_argument("--input-height", type=int, default=336)
    build_parser.add_argument("--device", default="auto")
    build_parser.add_argument("--local-files-only", action="store_true")
    build_parser.set_defaults(func=build)

    neighbors_parser = subparsers.add_parser(
        "propose-neighbors",
        help="propose review-only reference adjacency from calibrated base XY",
    )
    neighbors_parser.add_argument("--reference", required=True, type=Path)
    neighbors_parser.add_argument("--points", required=True, type=Path)
    neighbors_parser.add_argument("--output", required=True, type=Path)
    neighbors_parser.add_argument("--visualization", required=True, type=Path)
    neighbors_parser.add_argument(
        "--max-edge-length-mm",
        type=float,
        help="physical cutoff; defaults to median Delaunay edge length times --max-edge-factor",
    )
    neighbors_parser.add_argument("--max-edge-factor", type=float, default=1.25)
    neighbors_parser.set_defaults(func=propose_neighbors)

    reference_mask_parser = subparsers.add_parser(
        "prepare-reference-mask",
        help="extract a flat-reference garment mask from table appearance",
    )
    reference_mask_parser.add_argument("--image", required=True, type=Path)
    reference_mask_parser.add_argument("--output", required=True, type=Path)
    reference_mask_parser.set_defaults(func=prepare_reference_mask)

    combine_parser = subparsers.add_parser(
        "combine",
        help="combine independent canonical surface graphs without automatic cross-side edges",
    )
    combine_parser.add_argument("--graph", required=True, action="append", metavar="NAME=GRAPH_JSON")
    combine_parser.add_argument("--output", required=True, type=Path)
    combine_parser.set_defaults(func=combine)

    mask_parser = subparsers.add_parser(
        "prepare-mask",
        help="project a saved fused garment mask into one existing RGB-D camera view",
    )
    mask_parser.add_argument("--result", required=True, type=Path, help="saved perception result.json")
    mask_parser.add_argument("--camera-label", default="A")
    mask_parser.add_argument("--output", required=True, type=Path, help="binary camera-view mask PNG")
    mask_parser.set_defaults(func=prepare_mask)

    match_parser = subparsers.add_parser("match", help="match current dense features and refine with graph topology")
    match_parser.add_argument("--graph", required=True, type=Path)
    match_parser.add_argument("--reference", type=Path)
    match_parser.add_argument(
        "--reference-side",
        action="append",
        metavar="SIDE=IMAGE",
        help="surface-specific reference image; repeat for FRONT and BACK",
    )
    match_parser.add_argument("--current", required=True, action="append", metavar="NAME=IMAGE")
    match_parser.add_argument(
        "--current-features",
        action="append",
        metavar="NAME=NPY",
        help="optional precomputed (H,W,D) feature map per current image",
    )
    match_parser.add_argument("--mask", action="append", metavar="NAME=MASK", help="optional binary garment mask per current image")
    match_parser.add_argument(
        "--surface-xyz",
        action="append",
        metavar="NAME=NPY",
        help="optional calibrated per-pixel base XYZ array for conservative surface adjacency",
    )
    match_parser.add_argument("--ground-truth", action="append", metavar="NAME=JSON")
    match_parser.add_argument("--output", required=True, type=Path)
    match_parser.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    match_parser.add_argument("--input-size", type=int, default=448)
    match_parser.add_argument("--input-width", type=int, default=448)
    match_parser.add_argument("--input-height", type=int, default=336)
    match_parser.add_argument("--device", default="auto")
    match_parser.add_argument("--local-files-only", action="store_true")
    match_parser.add_argument("--top-k", type=int, default=5)
    match_parser.add_argument("--topology-lambda", type=float, default=0.25)
    match_parser.add_argument("--ambiguity-margin", type=float, default=0.06)
    match_parser.add_argument("--confident-threshold", type=float, default=0.58)
    match_parser.add_argument("--minimum-similarity", type=float, default=0.35)
    match_parser.add_argument("--adjacency-similarity-threshold", type=float, default=0.72)
    match_parser.add_argument("--max-surface-gap-mm", type=float, default=30.0)
    match_parser.add_argument(
        "--current-sample-radius",
        type=int,
        help="current local token radius; defaults to the reference graph sample radius",
    )
    match_parser.add_argument(
        "--ambiguity-crop-size",
        type=int,
        default=128,
        help="square crop size in source-image pixels for AMBIGUOUS node packets",
    )
    match_parser.set_defaults(func=match)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
