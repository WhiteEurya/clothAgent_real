"""Canonical garment-area graph matching with replaceable dense features.

This module is an offline perception prototype.  It deliberately stops at
visibility inference: it does not choose robot actions, modify the RGB-D
capture path, or call Claude.  The existing manual DINOv3 point annotation is
used as the smallest reference representation.  Each annotated point becomes
a small canonical surface area with a feature bank sampled from neighboring
patch tokens.

The matcher keeps top-k candidates for every current patch. A conservative
current-observation graph records visual/depth proximity without claiming
material adjacency. Canonical adjacency supplies positive-only evidence to
feature candidates, while disagreements are reported separately as possible
fold/contact/occlusion boundaries and never used as a negative penalty.
"""

from __future__ import annotations

import heapq
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

MATCH_STATUSES = ("VISIBLE", "UNOBSERVED", "AMBIGUOUS")


class DenseFeatureExtractor(Protocol):
    """Replaceable dense backbone contract used by the offline prototype."""

    def encode(self, image: Any) -> Any:
        """Return a dense ``(grid_h, grid_w, channels)`` feature map."""


def _normalize(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError("dense features must have shape (..., channels)")
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, 1e-8)


def bidirectional_chamfer_similarity(
    reference_samples: np.ndarray,
    current_samples: np.ndarray,
) -> float:
    """Symmetric local-set cosine similarity without fixed token alignment."""

    reference = _normalize(np.asarray(reference_samples, dtype=np.float32))
    current = _normalize(np.asarray(current_samples, dtype=np.float32))
    if reference.ndim != 2 or current.ndim != 2:
        raise ValueError("Chamfer feature samples must have shape (N, channels)")
    if not len(reference) or not len(current):
        raise ValueError("Chamfer feature sample sets must be non-empty")
    if reference.shape[1] != current.shape[1]:
        raise ValueError("Chamfer feature channels must match")
    cosine = reference @ current.T
    return 0.5 * float(
        np.mean(np.max(cosine, axis=1))
        + np.mean(np.max(cosine, axis=0))
    )


def _local_feature_samples(
    features: np.ndarray,
    px: int,
    py: int,
    *,
    radius: int,
    valid_mask: np.ndarray,
) -> np.ndarray:
    if radius < 0:
        raise ValueError("local feature radius must be non-negative")
    grid_h, grid_w = features.shape[:2]
    y0, y1 = max(0, py - radius), min(grid_h, py + radius + 1)
    x0, x1 = max(0, px - radius), min(grid_w, px + radius + 1)
    local_features = features[y0:y1, x0:x1]
    local_valid = valid_mask[y0:y1, x0:x1]
    samples = local_features[local_valid]
    if not len(samples):
        samples = features[py, px][None, :]
    return np.asarray(samples, dtype=np.float32)


def _finite_pair(x: Any, y: Any) -> tuple[float, float]:
    x, y = float(x), float(y)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("coordinates must be finite")
    return x, y


def _point_to_patch(
    x: float,
    y: float,
    image_size: tuple[int, int],
    feature_shape: tuple[int, int],
) -> tuple[int, int]:
    width, height = image_size
    grid_h, grid_w = feature_shape
    if width <= 0 or height <= 0 or grid_h <= 0 or grid_w <= 0:
        raise ValueError("image and feature dimensions must be positive")
    px = round(float(x) * (grid_w - 1) / max(1, width - 1))
    py = round(float(y) * (grid_h - 1) / max(1, height - 1))
    return min(grid_w - 1, max(0, px)), min(grid_h - 1, max(0, py))


def _patch_to_pixel(
    px: int,
    py: int,
    image_size: tuple[int, int],
    feature_shape: tuple[int, int],
) -> tuple[float, float]:
    width, height = image_size
    grid_h, grid_w = feature_shape
    x = (width - 1) / 2.0 if grid_w == 1 else float(px) * (width - 1) / (grid_w - 1)
    y = (height - 1) / 2.0 if grid_h == 1 else float(py) * (height - 1) / (grid_h - 1)
    return x, y


def constrained_geodesic_voronoi(
    garment_mask: np.ndarray,
    seed_patch_xy: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Partition a garment mask by shortest paths that cannot leave the mask."""

    mask = np.asarray(garment_mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("garment_mask must be a non-empty 2-D boolean array")
    if not seed_patch_xy:
        raise ValueError("at least one Voronoi seed is required")
    try:
        from scipy.ndimage import label as connected_components

        components, count = connected_components(
            mask,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        if count > 1:
            values, counts = np.unique(components[components > 0], return_counts=True)
            mask = components == int(values[int(np.argmax(counts))])
    except ImportError:
        pass
    valid_yx = np.argwhere(mask)
    snapped: list[tuple[int, int]] = []
    occupied: set[tuple[int, int]] = set()
    for px, py in seed_patch_xy:
        px, py = int(px), int(py)
        if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1] and mask[py, px]:
            snapped_xy = (px, py)
        else:
            delta = valid_yx - np.asarray([py, px], dtype=np.int64)
            nearest = valid_yx[int(np.argmin(np.sum(delta * delta, axis=1)))]
            snapped_xy = (int(nearest[1]), int(nearest[0]))
        if snapped_xy in occupied:
            raise ValueError(f"multiple canonical seeds snap to garment patch {snapped_xy}")
        occupied.add(snapped_xy)
        snapped.append(snapped_xy)

    distances = np.full(mask.shape, np.inf, dtype=np.float64)
    labels = np.full(mask.shape, -1, dtype=np.int32)
    queue: list[tuple[float, int, int, int]] = []
    for label_index, (px, py) in enumerate(snapped):
        distances[py, px] = 0.0
        labels[py, px] = label_index
        heapq.heappush(queue, (0.0, label_index, py, px))
    neighbors = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    while queue:
        distance, label_index, py, px = heapq.heappop(queue)
        if distance > distances[py, px] + 1e-9 or labels[py, px] != label_index:
            continue
        for dy, dx, cost in neighbors:
            qy, qx = py + dy, px + dx
            if not (0 <= qy < mask.shape[0] and 0 <= qx < mask.shape[1] and mask[qy, qx]):
                continue
            candidate = distance + cost
            better = candidate < distances[qy, qx] - 1e-9
            tie = abs(candidate - distances[qy, qx]) <= 1e-9 and label_index < labels[qy, qx]
            if better or tie:
                distances[qy, qx] = candidate
                labels[qy, qx] = label_index
                heapq.heappush(queue, (candidate, label_index, qy, qx))
    unassigned = mask & (labels < 0)
    if unassigned.any():
        raise ValueError(
            "garment_mask contains a disconnected component without a canonical seed: "
            f"{int(unassigned.sum())} patches"
        )
    return labels, snapped


@dataclass
class CanonicalArea:
    area_id: str
    label: str
    canonical_xy: tuple[float, float]
    canonical_patch_xy: tuple[int, int]
    surface_side: str | None = None
    canonical_base_xyz_mm: tuple[float, float, float] | None = None
    neighbor_area_ids: list[str] = field(default_factory=list)
    feature_sample_indices: list[int] = field(default_factory=list)
    region_patch_xy: list[tuple[int, int]] = field(default_factory=list)
    region_feature_sample_indices: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "area_id": self.area_id,
            "label": self.label,
            "canonical_xy": list(self.canonical_xy),
            "canonical_patch_xy": list(self.canonical_patch_xy),
            "surface_side": self.surface_side,
            "canonical_base_xyz_mm": (
                None
                if self.canonical_base_xyz_mm is None
                else list(self.canonical_base_xyz_mm)
            ),
            "neighbor_area_ids": list(self.neighbor_area_ids),
            "feature_sample_indices": list(self.feature_sample_indices),
            "region_patch_xy": [list(value) for value in self.region_patch_xy],
            "region_feature_sample_indices": list(self.region_feature_sample_indices),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CanonicalArea:
        base_xyz = payload.get("canonical_base_xyz_mm")
        return cls(
            area_id=str(payload["area_id"]),
            label=str(payload.get("label", payload["area_id"])),
            canonical_xy=(float(payload["canonical_xy"][0]), float(payload["canonical_xy"][1])),
            canonical_patch_xy=(int(payload["canonical_patch_xy"][0]), int(payload["canonical_patch_xy"][1])),
            surface_side=(
                None
                if payload.get("surface_side") is None
                else str(payload["surface_side"])
            ),
            canonical_base_xyz_mm=(
                None
                if base_xyz is None
                else (float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2]))
            ),
            neighbor_area_ids=[str(value) for value in payload.get("neighbor_area_ids", [])],
            feature_sample_indices=[int(value) for value in payload.get("feature_sample_indices", [])],
            region_patch_xy=[
                (int(value[0]), int(value[1]))
                for value in payload.get("region_patch_xy", [])
            ],
            region_feature_sample_indices=[
                int(value)
                for value in payload.get("region_feature_sample_indices", [])
            ],
        )


@dataclass
class CanonicalSurfaceGraph:
    """Fixed canonical areas and intrinsic material adjacency."""

    image_size: tuple[int, int]
    feature_shape: tuple[int, int]
    patch_size: int | None
    areas: dict[str, CanonicalArea]
    feature_bank: np.ndarray
    region_labels_by_side: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def area_ids(self) -> tuple[str, ...]:
        return tuple(self.areas)

    def neighbors(self, area_id: str) -> tuple[str, ...]:
        return tuple(self.areas[area_id].neighbor_area_ids)

    def has_edge(self, left: str, right: str) -> bool:
        return right in self.areas[left].neighbor_area_ids

    @classmethod
    def from_reference_points(
        cls,
        points: Sequence[Mapping[str, Any]],
        reference_features: np.ndarray,
        *,
        image_size: tuple[int, int],
        patch_size: int | None = None,
        area_prefix: str = "A",
        sample_radius: int = 1,
        neighbor_k: int = 4,
        explicit_neighbors: Mapping[str, Iterable[str]] | None = None,
        reference_mask: np.ndarray | None = None,
    ) -> CanonicalSurfaceGraph:
        """Build areas from the existing annotated point representation.

        If annotation records contain ``area_id`` and ``neighbor_area_ids``,
        those are honored.  Otherwise stable IDs are generated and a symmetric
        k-nearest-neighbor graph in reference image space is used as the
        minimal adjacency prior.  The latter is intentionally marked as a
        heuristic in metadata and should be replaced by manual edges when a
        garment-specific reference is available.
        """

        features = _normalize(np.asarray(reference_features, dtype=np.float32))
        if features.ndim != 3:
            raise ValueError("reference_features must have shape (grid_h, grid_w, channels)")
        if sample_radius < 0 or neighbor_k <= 0:
            raise ValueError("sample_radius must be non-negative and neighbor_k positive")
        if not points:
            raise ValueError("at least one canonical point is required")

        areas: dict[str, CanonicalArea] = {}
        sample_vectors: list[np.ndarray] = []
        locations: list[tuple[float, float]] = []
        base_locations: list[tuple[float, float] | None] = []
        ids: list[str] = []
        for index, raw in enumerate(points, start=1):
            x, y = _finite_pair(raw["x"], raw["y"])
            area_id = str(raw.get("area_id", f"{area_prefix}{int(raw.get('id', index)):02d}"))
            if area_id in areas:
                raise ValueError(f"duplicate canonical area id: {area_id}")
            label = str(raw.get("label", area_id))
            surface_side = (
                None
                if raw.get("surface_side") is None
                else str(raw["surface_side"]).strip().upper()
            )
            raw_base_xyz = raw.get("base_xyz_mm")
            canonical_base_xyz_mm = None
            if raw_base_xyz is not None:
                if len(raw_base_xyz) != 3:
                    raise ValueError(f"{area_id} base_xyz_mm must contain three values")
                canonical_base_xyz_mm = tuple(float(value) for value in raw_base_xyz)
                if not all(math.isfinite(value) for value in canonical_base_xyz_mm):
                    raise ValueError(f"{area_id} base_xyz_mm must be finite")
            patch_xy = _point_to_patch(x, y, image_size, features.shape[:2])
            px, py = patch_xy
            ids.append(area_id)
            locations.append((x, y))
            base_locations.append(
                None
                if canonical_base_xyz_mm is None
                else canonical_base_xyz_mm[:2]
            )
            areas[area_id] = CanonicalArea(
                area_id=area_id,
                label=label,
                canonical_xy=(x, y),
                canonical_patch_xy=patch_xy,
                surface_side=surface_side,
                canonical_base_xyz_mm=canonical_base_xyz_mm,
            )

        region_labels_by_side: dict[str, np.ndarray] = {}
        region_labels: np.ndarray | None = None
        if reference_mask is not None:
            mask = np.asarray(reference_mask, dtype=bool)
            if mask.shape != features.shape[:2]:
                raise ValueError(
                    f"reference_mask shape {mask.shape} must match feature grid {features.shape[:2]}"
                )
            region_labels, snapped = constrained_geodesic_voronoi(
                mask,
                [areas[area_id].canonical_patch_xy for area_id in ids],
            )
            for area_id, patch_xy in zip(ids, snapped):
                areas[area_id].canonical_patch_xy = patch_xy
            side_values = {
                areas[area_id].surface_side or "REFERENCE"
                for area_id in ids
            }
            if len(side_values) != 1:
                raise ValueError("one reference mask can only partition one surface side")
            region_labels_by_side[next(iter(side_values))] = region_labels

        # Build symmetric local feature sets after optional material-region
        # partitioning. When regions exist, local samples cannot leak across
        # the garment boundary or into another canonical area.
        for area_index, area_id in enumerate(ids):
            area = areas[area_id]
            px, py = area.canonical_patch_xy
            local_samples: list[np.ndarray] = []
            for sy in range(
                max(0, py - sample_radius),
                min(features.shape[0], py + sample_radius + 1),
            ):
                for sx in range(
                    max(0, px - sample_radius),
                    min(features.shape[1], px + sample_radius + 1),
                ):
                    if region_labels is not None and region_labels[sy, sx] != area_index:
                        continue
                    local_samples.append(features[sy, sx])
            if not local_samples:
                local_samples = [features[py, px]]
            local_start = len(sample_vectors)
            sample_vectors.extend(local_samples)
            area.feature_sample_indices = list(
                range(local_start, local_start + len(local_samples))
            )

            if region_labels is not None:
                region_yx = np.argwhere(region_labels == area_index)
                area.region_patch_xy = [
                    (int(sx), int(sy)) for sy, sx in region_yx
                ]
                region_start = len(sample_vectors)
                sample_vectors.extend(
                    [features[int(sy), int(sx)] for sy, sx in region_yx]
                )
                area.region_feature_sample_indices = list(
                    range(region_start, region_start + len(region_yx))
                )

        if explicit_neighbors is None:
            embedded_neighbors = {
                area_id: [str(value) for value in raw.get("neighbor_area_ids", [])]
                for area_id, raw in zip(ids, points)
                if raw.get("neighbor_area_ids") is not None
            }
            if embedded_neighbors:
                explicit_neighbors = embedded_neighbors

        if explicit_neighbors is not None:
            for area_id, neighbors in explicit_neighbors.items():
                if area_id not in areas:
                    raise ValueError(f"explicit neighbor references unknown area {area_id}")
                for neighbor in neighbors:
                    if neighbor not in areas:
                        raise ValueError(f"explicit neighbor references unknown area {neighbor}")
                    if neighbor != area_id and neighbor not in areas[area_id].neighbor_area_ids:
                        areas[area_id].neighbor_area_ids.append(neighbor)
                    if neighbor != area_id and area_id not in areas[neighbor].neighbor_area_ids:
                        areas[neighbor].neighbor_area_ids.append(area_id)
            adjacency_source = "explicit"
        else:
            distances = np.full((len(ids), len(ids)), np.inf, dtype=np.float32)
            all_base_locations = all(location is not None for location in base_locations)
            location_array = np.asarray(
                base_locations if all_base_locations else locations,
                dtype=np.float32,
            )
            for row in range(len(ids)):
                distances[row] = np.linalg.norm(location_array - location_array[row], axis=1)
                distances[row, row] = np.inf
                for column in np.argsort(distances[row])[: min(neighbor_k, len(ids) - 1)]:
                    if np.isfinite(distances[row, column]):
                        left, right = ids[row], ids[int(column)]
                        if right not in areas[left].neighbor_area_ids:
                            areas[left].neighbor_area_ids.append(right)
                        if left not in areas[right].neighbor_area_ids:
                            areas[right].neighbor_area_ids.append(left)
            adjacency_source = "symmetric_knn_heuristic"
            adjacency_coordinate_space = (
                "calibrated_base_xy_mm"
                if all_base_locations
                else "reference_image_xy"
            )

        if explicit_neighbors is not None:
            adjacency_coordinate_space = "explicit_intrinsic_edges"

        return cls(
            image_size=(int(image_size[0]), int(image_size[1])),
            feature_shape=(int(features.shape[0]), int(features.shape[1])),
            patch_size=None if patch_size is None else int(patch_size),
            areas=areas,
            feature_bank=np.asarray(sample_vectors, dtype=np.float32),
            region_labels_by_side=region_labels_by_side,
            metadata={
                "adjacency_source": adjacency_source,
                "adjacency_coordinate_space": adjacency_coordinate_space,
                "sample_radius": int(sample_radius),
                "neighbor_k": int(neighbor_k),
                "area_count": len(areas),
                "surface_sides": sorted(
                    {
                        area.surface_side
                        for area in areas.values()
                        if area.surface_side is not None
                    }
                ),
                "canonical_region_method": (
                    "constrained_geodesic_voronoi_on_feature_grid"
                    if region_labels is not None
                    else "implicit_seed_local_neighborhood"
                ),
                "reference_mask_applied": region_labels is not None,
                "local_reference_feature_scope": (
                    "seed_centered_window_clipped_to_canonical_region"
                    if region_labels is not None
                    else "seed_centered_window"
                ),
                "region_features_persisted": region_labels is not None,
            },
        )

    @classmethod
    def combine(
        cls,
        graphs: Sequence[CanonicalSurfaceGraph],
        *,
        seam_edges: Iterable[tuple[str, str]] = (),
    ) -> CanonicalSurfaceGraph:
        """Combine independent surface subgraphs without inventing cross-side edges."""

        components = list(graphs)
        if len(components) < 2:
            raise ValueError("at least two canonical surface graphs are required")
        first = components[0]
        for graph in components[1:]:
            if graph.image_size != first.image_size:
                raise ValueError("component canonical graphs must share image_size")
            if graph.feature_shape != first.feature_shape:
                raise ValueError("component canonical graphs must share feature_shape")
            if graph.feature_bank.shape[-1] != first.feature_bank.shape[-1]:
                raise ValueError("component canonical graphs must share feature channels")
            if graph.patch_size != first.patch_size:
                raise ValueError("component canonical graphs must share patch_size")

        areas: dict[str, CanonicalArea] = {}
        banks: list[np.ndarray] = []
        offset = 0
        component_summaries: list[dict[str, Any]] = []
        for graph in components:
            component_ids = set(graph.areas)
            overlap = sorted(component_ids.intersection(areas))
            if overlap:
                raise ValueError(f"duplicate canonical area IDs across components: {overlap}")
            for area in graph.areas.values():
                unknown_neighbors = sorted(set(area.neighbor_area_ids) - component_ids)
                if unknown_neighbors:
                    raise ValueError(
                        f"component area {area.area_id} references areas outside its subgraph: "
                        f"{unknown_neighbors}"
                    )
                clone = CanonicalArea.from_dict(area.as_dict())
                clone.feature_sample_indices = [offset + value for value in clone.feature_sample_indices]
                clone.region_feature_sample_indices = [
                    offset + value
                    for value in clone.region_feature_sample_indices
                ]
                areas[clone.area_id] = clone
            banks.append(np.asarray(graph.feature_bank, dtype=np.float32))
            offset += len(graph.feature_bank)
            component_summaries.append(
                {
                    "area_count": len(graph.areas),
                    "surface_sides": list(graph.metadata.get("surface_sides", [])),
                    "adjacency_source": graph.metadata.get("adjacency_source"),
                    "adjacency_review_status": graph.metadata.get("adjacency_review_status"),
                    "reference_image": graph.metadata.get("reference_image"),
                    "reference_mask": graph.metadata.get("reference_mask"),
                    "input_shape_yx": graph.metadata.get("input_shape_yx"),
                    "feature_shape_yx": list(graph.feature_shape),
                    "canonical_region_method": graph.metadata.get("canonical_region_method"),
                    "local_reference_feature_scope": graph.metadata.get(
                        "local_reference_feature_scope"
                    ),
                }
            )

        confirmed_seams: list[list[str]] = []
        for left, right in seam_edges:
            left_id, right_id = str(left), str(right)
            if left_id not in areas or right_id not in areas:
                raise ValueError(f"seam edge references unknown areas: {(left_id, right_id)}")
            if left_id == right_id:
                raise ValueError("seam edge endpoints must differ")
            if right_id not in areas[left_id].neighbor_area_ids:
                areas[left_id].neighbor_area_ids.append(right_id)
            if left_id not in areas[right_id].neighbor_area_ids:
                areas[right_id].neighbor_area_ids.append(left_id)
            confirmed_seams.append([left_id, right_id])

        feature_bank = np.concatenate(banks, axis=0)
        surface_sides = sorted(
            {
                area.surface_side
                for area in areas.values()
                if area.surface_side is not None
            }
        )
        region_labels_by_side: dict[str, np.ndarray] = {}
        for graph in components:
            overlap = sorted(set(region_labels_by_side).intersection(graph.region_labels_by_side))
            if overlap:
                raise ValueError(f"duplicate canonical region surfaces across components: {overlap}")
            region_labels_by_side.update(
                {
                    side: np.asarray(labels, dtype=np.int32).copy()
                    for side, labels in graph.region_labels_by_side.items()
                }
            )
        return cls(
            image_size=first.image_size,
            feature_shape=first.feature_shape,
            patch_size=first.patch_size,
            areas=areas,
            feature_bank=feature_bank,
            region_labels_by_side=region_labels_by_side,
            metadata={
                "adjacency_source": "combined_explicit_surface_subgraphs",
                "adjacency_coordinate_space": "intrinsic_component_edges",
                "area_count": len(areas),
                "surface_sides": surface_sides,
                "component_graphs": component_summaries,
                "automatic_cross_surface_edges": False,
                "confirmed_seam_edges": confirmed_seams,
            },
        )

    def save(self, directory: Path) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        graph_path = directory / "canonical_graph.json"
        feature_path = directory / "canonical_feature_bank.npz"
        archive: dict[str, np.ndarray] = {"feature_bank": self.feature_bank}
        region_arrays: dict[str, str] = {}
        for index, (side, labels) in enumerate(sorted(self.region_labels_by_side.items())):
            key = f"region_labels_{index}"
            archive[key] = np.asarray(labels, dtype=np.int32)
            region_arrays[side] = key
        np.savez_compressed(feature_path, **archive)
        payload = {
            "version": 1,
            "image_size": list(self.image_size),
            "feature_shape": list(self.feature_shape),
            "patch_size": self.patch_size,
            "feature_bank": feature_path.name,
            "areas": [area.as_dict() for area in self.areas.values()],
            "region_label_arrays": region_arrays,
            "metadata": dict(self.metadata),
        }
        graph_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return graph_path, feature_path

    @classmethod
    def load(cls, graph_path: Path) -> CanonicalSurfaceGraph:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
        feature_path = graph_path.parent / str(payload["feature_bank"])
        with np.load(feature_path) as archive:
            feature_bank = np.asarray(archive["feature_bank"], dtype=np.float32)
            region_labels_by_side = {
                str(side): np.asarray(archive[str(key)], dtype=np.int32)
                for side, key in payload.get("region_label_arrays", {}).items()
            }
        areas = {area.area_id: area for area in (CanonicalArea.from_dict(raw) for raw in payload["areas"])}
        return cls(
            image_size=(int(payload["image_size"][0]), int(payload["image_size"][1])),
            feature_shape=(int(payload["feature_shape"][0]), int(payload["feature_shape"][1])),
            patch_size=payload.get("patch_size"),
            areas=areas,
            feature_bank=feature_bank,
            region_labels_by_side=region_labels_by_side,
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class CurrentNode:
    node_id: str
    patch_xy: tuple[int, int]
    pixel_xy: tuple[float, float]
    neighbor_node_ids: tuple[str, ...]


def build_current_visible_graph(
    current_features: np.ndarray,
    *,
    image_size: tuple[int, int],
    valid_mask: np.ndarray | None = None,
    surface_xyz_mm: np.ndarray | None = None,
    max_surface_gap_mm: float = 30.0,
    adjacency_similarity_threshold: float = 0.72,
) -> tuple[dict[str, CurrentNode], list[tuple[str, str]], list[tuple[str, str]]]:
    """Build conservative local adjacency from dense patch features.

    Only four-connected patch neighbors with valid mask support and a high
    local feature cosine become current-observation adjacency. When a
    calibrated XYZ grid is supplied, the 3-D surface gap must also be below
    ``max_surface_gap_mm``. This edge means only that two observed patches
    currently touch or appear continuous; it never asserts intrinsic material
    adjacency. Remaining pixel neighbors stay UNKNOWN.
    """

    features = _normalize(np.asarray(current_features, dtype=np.float32))
    if features.ndim != 3:
        raise ValueError("current_features must have shape (grid_h, grid_w, channels)")
    grid_h, grid_w = features.shape[:2]
    if valid_mask is None:
        valid = np.ones((grid_h, grid_w), dtype=bool)
    else:
        valid_array = np.asarray(valid_mask, dtype=bool)
        if valid_array.shape != (grid_h, grid_w):
            raise ValueError("valid_mask must match current feature grid")
        valid = valid_array
    xyz = None
    if surface_xyz_mm is not None:
        xyz = np.asarray(surface_xyz_mm, dtype=np.float32)
        if xyz.shape != (grid_h, grid_w, 3):
            raise ValueError("surface_xyz_mm must have shape (grid_h, grid_w, 3)")
        if max_surface_gap_mm <= 0:
            raise ValueError("max_surface_gap_mm must be positive")

    nodes: dict[str, CurrentNode] = {}
    edge_pairs: list[tuple[str, str]] = []
    unknown_pairs: list[tuple[str, str]] = []
    for py in range(grid_h):
        for px in range(grid_w):
            if not valid[py, px]:
                continue
            node_id = f"X{py:03d}_{px:03d}"
            pixel_xy = _patch_to_pixel(px, py, image_size, (grid_h, grid_w))
            neighbors: list[str] = []
            for dy, dx in ((0, 1), (1, 0)):
                qy, qx = py + dy, px + dx
                if qy >= grid_h or qx >= grid_w or not valid[qy, qx]:
                    continue
                cosine = float(np.dot(features[py, px], features[qy, qx]))
                surface_continuous = True
                if xyz is not None:
                    surface_continuous = bool(
                        np.all(np.isfinite(xyz[py, px]))
                        and np.all(np.isfinite(xyz[qy, qx]))
                        and np.linalg.norm(xyz[py, px] - xyz[qy, qx]) <= max_surface_gap_mm
                    )
                if cosine >= adjacency_similarity_threshold and surface_continuous:
                    other_id = f"X{qy:03d}_{qx:03d}"
                    neighbors.append(other_id)
                    edge_pairs.append((node_id, other_id))
                else:
                    unknown_pairs.append((node_id, f"X{qy:03d}_{qx:03d}"))
            # Backfill reverse edges after all edge evidence is collected.
            nodes[node_id] = CurrentNode(node_id, (px, py), pixel_xy, tuple(neighbors))
    if edge_pairs:
        reverse: dict[str, list[str]] = {node_id: list(node.neighbor_node_ids) for node_id, node in nodes.items()}
        for left, right in edge_pairs:
            reverse[right].append(left)
        nodes = {
            node_id: CurrentNode(node.node_id, node.patch_xy, node.pixel_xy, tuple(sorted(set(reverse[node_id]))))
            for node_id, node in nodes.items()
        }
    return nodes, edge_pairs, unknown_pairs


@dataclass
class NodeMatch:
    node_id: str
    pixel_xy: tuple[float, float]
    candidates: list[dict[str, Any]]
    status: str
    selected_area_id: str | None
    feature_only_status: str
    feature_only_selected_area_id: str | None
    confidence: float = 0.0
    feature_only_confidence: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "pixel_xy": list(self.pixel_xy),
            "candidates": [dict(candidate) for candidate in self.candidates],
            "status": self.status,
            "selected_area_id": self.selected_area_id,
            "feature_only_status": self.feature_only_status,
            "feature_only_selected_area_id": self.feature_only_selected_area_id,
            "confidence": float(self.confidence),
            "feature_only_confidence": float(self.feature_only_confidence),
        }


@dataclass
class VisibilityResult:
    canonical_areas: dict[str, dict[str, Any]]
    current_nodes: list[NodeMatch]
    ambiguities: list[dict[str, Any]]
    frontiers: list[dict[str, Any]]
    current_edges: list[tuple[str, str]]
    current_unknown_edges: list[tuple[str, str]]
    observation_adjacencies: list[dict[str, Any]]
    parameters: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical_areas": self.canonical_areas,
            "current_nodes": [node.as_dict() for node in self.current_nodes],
            "ambiguities": [dict(item) for item in self.ambiguities],
            "frontiers": [dict(frontier) for frontier in self.frontiers],
            "current_edges": [list(edge) for edge in self.current_edges],
            "current_unknown_edges": [list(edge) for edge in self.current_unknown_edges],
            "observation_adjacencies": [
                dict(adjacency) for adjacency in self.observation_adjacencies
            ],
            "parameters": dict(self.parameters),
        }


def _top_k(values: np.ndarray, ids: Sequence[str], k: int) -> list[dict[str, Any]]:
    order = np.argsort(values)[::-1][: max(1, min(k, len(ids)))]
    return [
        {"area_id": ids[int(index)], "feature_similarity": float(values[int(index)])}
        for index in order
    ]


def _bounded_score_confidence(score: float, topology_lambda: float) -> float:
    """Convert a refined cosine score to a bounded display confidence.

    ``refined_score`` is intentionally left unbounded because it is the sum of
    a cosine similarity and a topology term.  Dividing its positive part by
    the largest possible positive score keeps the user-facing confidence in
    ``[0, 1]`` without hiding the raw, interpretable score.
    """

    scale = 1.0 + max(0.0, float(topology_lambda))
    return float(np.clip(float(score) / scale, 0.0, 1.0))


def _candidate_selection(
    top: Mapping[str, Any] | None,
    second: Mapping[str, Any] | None,
    *,
    score_key: str,
    top_confidence: float,
    minimum_similarity: float,
    confident_threshold: float,
    ambiguity_margin: float,
) -> tuple[str, str | None]:
    if top is None:
        return "AMBIGUOUS", None
    top_score = float(top[score_key])
    close_second = (
        second is not None
        and top_score - float(second[score_key]) < ambiguity_margin
    )
    if top_score < minimum_similarity or top_confidence < confident_threshold or close_second:
        return "AMBIGUOUS", None
    return "VISIBLE", str(top["area_id"])


def _canonical_graph_distance(
    graph: CanonicalSurfaceGraph,
    left: str,
    right: str,
) -> int | None:
    """Return intrinsic hop distance, or None for disconnected surface graphs."""

    if left == right:
        return 0
    visited = {left}
    frontier = [(left, 0)]
    while frontier:
        area_id, distance = frontier.pop(0)
        for neighbor in graph.neighbors(area_id):
            if neighbor == right:
                return distance + 1
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append((neighbor, distance + 1))
    return None


def match_visibility(
    graph: CanonicalSurfaceGraph,
    current_features: np.ndarray,
    *,
    image_size: tuple[int, int],
    valid_mask: np.ndarray | None = None,
    surface_xyz_mm: np.ndarray | None = None,
    max_surface_gap_mm: float = 30.0,
    top_k: int = 5,
    topology_lambda: float = 0.25,
    ambiguity_margin: float = 0.06,
    confident_threshold: float = 0.58,
    minimum_similarity: float = 0.35,
    adjacency_similarity_threshold: float = 0.72,
    current_sample_radius: int | None = None,
) -> VisibilityResult:
    """Run feature-only and topology-refined soft canonical matching."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if topology_lambda < 0:
        raise ValueError("topology_lambda must be non-negative")
    if ambiguity_margin < 0:
        raise ValueError("ambiguity_margin must be non-negative")
    if current_sample_radius is None:
        current_sample_radius = int(graph.metadata.get("sample_radius", 1))
    if current_sample_radius < 0:
        raise ValueError("current_sample_radius must be non-negative")
    current = _normalize(np.asarray(current_features, dtype=np.float32))
    if current.ndim != 3 or current.shape[-1] != graph.feature_bank.shape[-1]:
        raise ValueError("current feature channels must match canonical feature bank")
    nodes, edge_pairs, unknown_pairs = build_current_visible_graph(
        current,
        image_size=image_size,
        valid_mask=valid_mask,
        surface_xyz_mm=surface_xyz_mm,
        max_surface_gap_mm=max_surface_gap_mm,
        adjacency_similarity_threshold=adjacency_similarity_threshold,
    )
    current_valid = (
        np.ones(current.shape[:2], dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    area_ids = list(graph.areas)
    banks = _normalize(graph.feature_bank)
    node_scores: dict[str, np.ndarray] = {}
    node_candidates: dict[str, list[dict[str, Any]]] = {}
    for node_id, node in nodes.items():
        px, py = node.patch_xy
        current_samples = _local_feature_samples(
            current,
            px,
            py,
            radius=current_sample_radius,
            valid_mask=current_valid,
        )
        area_scores = np.full(len(area_ids), -1.0, dtype=np.float32)
        for area_index, area_id in enumerate(area_ids):
            area = graph.areas[area_id]
            samples = banks[area.feature_sample_indices]
            area_scores[area_index] = bidirectional_chamfer_similarity(
                samples,
                current_samples,
            )
        node_scores[node_id] = area_scores
        node_candidates[node_id] = _top_k(area_scores, area_ids, top_k)

    # Freeze feature-only high-confidence beliefs before any topology step.
    # Only these beliefs may propagate positive evidence. Ambiguous neighbors
    # and canonically incompatible observation adjacencies contribute zero,
    # never a negative penalty.
    feature_beliefs: dict[str, dict[str, Any]] = {}
    for node_id, candidates in node_candidates.items():
        top = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None
        confidence = (
            float(np.clip(float(top["feature_similarity"]), 0.0, 1.0))
            if top is not None
            else 0.0
        )
        status, selected = _candidate_selection(
            top,
            second,
            score_key="feature_similarity",
            top_confidence=confidence,
            minimum_similarity=minimum_similarity,
            confident_threshold=confident_threshold,
            ambiguity_margin=ambiguity_margin,
        )
        feature_beliefs[node_id] = {
            "status": status,
            "selected_area_id": selected,
            "confidence": confidence,
        }

    refined_candidates: dict[str, list[dict[str, Any]]] = {}
    for node_id, node in nodes.items():
        feature_scores = node_scores[node_id]
        refined = feature_scores.copy()
        known_neighbors = [neighbor for neighbor in node.neighbor_node_ids if neighbor in node_scores]
        high_confidence_neighbors = [
            (neighbor, feature_beliefs[neighbor])
            for neighbor in known_neighbors
            if feature_beliefs[neighbor]["selected_area_id"] is not None
        ]
        topology_supports_by_index: dict[int, list[dict[str, Any]]] = {}
        if known_neighbors and topology_lambda:
            for area_index, area_id in enumerate(area_ids):
                canonical_neighbors = set(graph.neighbors(area_id))
                # Current nodes are dense patches while canonical nodes are
                # larger sampled neighborhoods. Adjacent current patches may
                # therefore belong to the same canonical area, not only to an
                # adjacent canonical area. Treat self-or-neighbor as the
                # compatible local topology set.
                compatible_areas = canonical_neighbors | {area_id}
                supports = [
                    {
                        "current_neighbor_node_id": neighbor,
                        "anchor_area_id": str(belief["selected_area_id"]),
                        "anchor_confidence": float(belief["confidence"]),
                    }
                    for neighbor, belief in high_confidence_neighbors
                    if str(belief["selected_area_id"]) in compatible_areas
                ]
                if supports:
                    topology_supports_by_index[area_index] = supports
                    candidate_plausibility = float(
                        np.clip(
                            (float(feature_scores[area_index]) - minimum_similarity)
                            / max(1e-6, 1.0 - minimum_similarity),
                            0.0,
                            1.0,
                        )
                    )
                    refined[area_index] += float(topology_lambda) * candidate_plausibility * float(
                        sum(support["anchor_confidence"] for support in supports)
                        / max(1, len(high_confidence_neighbors))
                    )
        feature_order = list(np.argsort(feature_scores)[::-1][: max(1, min(top_k, len(area_ids)))])
        refined_order = list(np.argsort(refined)[::-1][: max(1, min(top_k, len(area_ids)))])
        union = list(dict.fromkeys([int(index) for index in refined_order + feature_order]))
        candidates: list[dict[str, Any]] = []
        for index in union:
            feature_rank = feature_order.index(index) + 1 if index in feature_order else None
            refined_rank = refined_order.index(index) + 1 if index in refined_order else None
            candidates.append(
                {
                    "area_id": area_ids[index],
                    "feature_similarity": float(feature_scores[index]),
                    "refined_score": float(refined[index]),
                    "topology_bonus": float(refined[index] - feature_scores[index]),
                    "positive_topology_supports": topology_supports_by_index.get(index, []),
                    "feature_rank": feature_rank,
                    "refined_rank": refined_rank,
                }
            )
        candidates.sort(key=lambda candidate: (candidate["refined_rank"] is None, candidate["refined_rank"] or 10**9))
        refined_candidates[node_id] = candidates

    node_matches: list[NodeMatch] = []
    area_evidence: dict[str, list[dict[str, Any]]] = {area_id: [] for area_id in area_ids}
    for node_id, node in nodes.items():
        candidates = refined_candidates[node_id]
        refined_ranked = [candidate for candidate in candidates if candidate["refined_rank"] is not None]
        top = refined_ranked[0] if refined_ranked else None
        second = refined_ranked[1] if len(refined_ranked) > 1 else None
        feature_belief = feature_beliefs[node_id]
        effective_topology_lambda = (
            topology_lambda
            if top is not None and top.get("positive_topology_supports")
            else 0.0
        )
        refined_confidence = (
            max(
                float(np.clip(float(top["feature_similarity"]), 0.0, 1.0)),
                _bounded_score_confidence(
                    float(top["refined_score"]), effective_topology_lambda
                ),
            )
            if top is not None
            else 0.0
        )
        feature_confidence = float(feature_belief["confidence"])
        status, selected = _candidate_selection(
            top,
            second,
            score_key="refined_score",
            top_confidence=refined_confidence,
            minimum_similarity=minimum_similarity,
            confident_threshold=confident_threshold,
            ambiguity_margin=ambiguity_margin,
        )
        feature_status = str(feature_belief["status"])
        feature_selected = feature_belief["selected_area_id"]
        if selected is not None:
            area_evidence[selected].append(
                {
                    "node_id": node_id,
                    "refined_score": float(top["refined_score"]),
                    "confidence": refined_confidence,
                    "status": status,
                }
            )
        elif top is not None and float(top["refined_score"]) >= minimum_similarity:
            # A close score tie means every candidate inside the ambiguity
            # margin is plausibly visible.  Keeping evidence for only rank 1
            # would incorrectly label the other hypotheses UNOBSERVED.
            top_score = float(top["refined_score"])
            plausible = [
                candidate
                for candidate in refined_ranked
                if float(candidate["refined_score"]) >= minimum_similarity
                and top_score - float(candidate["refined_score"]) < ambiguity_margin
            ]
            if not plausible:
                plausible = [top]
            for candidate in plausible:
                candidate_score = float(candidate["refined_score"])
                area_evidence[str(candidate["area_id"])].append(
                    {
                        "node_id": node_id,
                        "refined_score": candidate_score,
                        "confidence": _bounded_score_confidence(
                            candidate_score,
                            topology_lambda
                            if candidate.get("positive_topology_supports")
                            else 0.0,
                        ),
                        "status": status,
                    }
                )
        node_matches.append(
            NodeMatch(
                node_id=node_id,
                pixel_xy=node.pixel_xy,
                candidates=candidates,
                status=status,
                selected_area_id=selected,
                feature_only_status=feature_status,
                feature_only_selected_area_id=feature_selected,
                confidence=refined_confidence,
                feature_only_confidence=feature_confidence,
            )
        )

    canonical_areas: dict[str, dict[str, Any]] = {}
    for area_id, area in graph.areas.items():
        evidence = area_evidence[area_id]
        visible = [item for item in evidence if item["status"] == "VISIBLE"]
        ambiguous = [item for item in evidence if item["status"] == "AMBIGUOUS"]
        if visible:
            status = "VISIBLE"
            strongest = max(visible, key=lambda item: float(item["confidence"]))
        elif ambiguous:
            status = "AMBIGUOUS"
            strongest = max(ambiguous, key=lambda item: float(item["confidence"]))
        else:
            status = "UNOBSERVED"
            strongest = {"confidence": 0.0, "refined_score": None}
        canonical_areas[area_id] = {
            "area_id": area_id,
            "label": area.label,
            "status": status,
            "confidence": float(strongest["confidence"]),
            "refined_score": strongest["refined_score"],
            "neighbor_area_ids": list(area.neighbor_area_ids),
            "evidence_node_ids": [str(item["node_id"]) for item in evidence],
        }

    frontiers: list[dict[str, Any]] = []
    for area_id, summary in canonical_areas.items():
        if summary["status"] != "VISIBLE":
            continue
        for neighbor in graph.neighbors(area_id):
            neighbor_status = canonical_areas[neighbor]["status"]
            if neighbor_status in {"UNOBSERVED", "AMBIGUOUS"}:
                frontiers.append(
                    {
                        "visible_area_id": area_id,
                        "hidden_or_ambiguous_area_id": neighbor,
                        "target_status": neighbor_status,
                    }
                )

    node_match_map = {node.node_id: node for node in node_matches}
    observation_adjacencies: list[dict[str, Any]] = []
    for left_node_id, right_node_id in edge_pairs:
        left_match = node_match_map[left_node_id]
        right_match = node_match_map[right_node_id]
        left_area = left_match.selected_area_id
        right_area = right_match.selected_area_id
        record: dict[str, Any] = {
            "current_node_ids": [left_node_id, right_node_id],
            "current_pixel_xy": [
                list(left_match.pixel_xy),
                list(right_match.pixel_xy),
            ],
            "canonical_area_ids": [left_area, right_area],
            "observation_semantics": (
                "current visual/depth proximity only; does not imply material adjacency"
            ),
        }
        if left_area is None or right_area is None:
            record.update(
                {
                    "relation": "AMBIGUOUS",
                    "canonical_graph_distance": None,
                    "interpretation": "insufficient canonical identity evidence",
                }
            )
        else:
            graph_distance = _canonical_graph_distance(graph, left_area, right_area)
            if graph_distance in {0, 1}:
                record.update(
                    {
                        "relation": "MATERIAL_CONSISTENT",
                        "canonical_graph_distance": graph_distance,
                        "interpretation": (
                            "same canonical area"
                            if graph_distance == 0
                            else "canonical material neighbors"
                        ),
                    }
                )
            else:
                record.update(
                    {
                        "relation": "UNEXPECTED_ADJACENCY",
                        "canonical_graph_distance": graph_distance,
                        "interpretation": "possible fold/contact/occlusion boundary",
                    }
                )
        observation_adjacencies.append(record)

    ambiguities: list[dict[str, Any]] = []
    for node_match in node_matches:
        if node_match.status != "AMBIGUOUS":
            continue
        current_node = nodes[node_match.node_id]
        surrounding = []
        for neighbor_node_id in current_node.neighbor_node_ids:
            neighbor_match = node_match_map.get(neighbor_node_id)
            if neighbor_match is not None and neighbor_match.selected_area_id is not None:
                surrounding.append(
                    {
                        "current_node_id": neighbor_node_id,
                        "canonical_area_id": neighbor_match.selected_area_id,
                    }
                )
        ambiguities.append(
            {
                "current_node_id": node_match.node_id,
                "pixel_xy": list(node_match.pixel_xy),
                "top_k_candidates": [
                    {
                        **dict(candidate),
                        "canonical_neighbor_area_ids": list(graph.neighbors(str(candidate["area_id"]))),
                    }
                    for candidate in node_match.candidates
                    if candidate.get("refined_rank") is not None
                ],
                "matched_surrounding_areas": surrounding,
                "crop_request": {
                    "center_xy": list(node_match.pixel_xy),
                    "purpose": "optional Claude ambiguity reasoning only; no dense re-analysis",
                },
            }
        )

    return VisibilityResult(
        canonical_areas=canonical_areas,
        current_nodes=node_matches,
        ambiguities=ambiguities,
        frontiers=frontiers,
        current_edges=edge_pairs,
        current_unknown_edges=unknown_pairs,
        observation_adjacencies=observation_adjacencies,
        parameters={
            "top_k": int(top_k),
            "topology_lambda": float(topology_lambda),
            "ambiguity_margin": float(ambiguity_margin),
            "confident_threshold": float(confident_threshold),
            "minimum_similarity": float(minimum_similarity),
            "adjacency_similarity_threshold": float(adjacency_similarity_threshold),
            "max_surface_gap_mm": float(max_surface_gap_mm),
            "confidence_definition": (
                "clip(refined_score / (1 + effective_topology_lambda), 0, 1); "
                "effective lambda is zero for nodes without known current-surface neighbors"
            ),
            "confident_threshold_applies_to": "bounded confidence",
            "current_observation_graph_policy": (
                "four_connected_valid_patches_plus_local_feature_cosine"
                + ("_plus_calibrated_3d_continuity" if surface_xyz_mm is not None else "_without_3d_proxy")
            ),
            "current_observation_edge_semantics": (
                "visual/depth adjacency only; never automatically material adjacency"
            ),
            "topology_policy": (
                "positive_only propagation from high-confidence feature-only anchors; "
                "bonus gated by local candidate plausibility; canonical incompatibility "
                "contributes no penalty"
            ),
            "topology_compatibility": "same_canonical_area_or_intrinsic_neighbor",
            "local_similarity": "bidirectional_chamfer_cosine",
            "reference_sample_radius": int(graph.metadata.get("sample_radius", 1)),
            "current_sample_radius": int(current_sample_radius),
        },
    )


def evaluate_visibility(
    node_matches: Sequence[NodeMatch],
    ground_truth: Mapping[str, str],
    *,
    recall_ks: Sequence[int] = (3, 5),
) -> dict[str, Any]:
    """Evaluate top-k identity and ambiguity behavior against node labels."""

    evaluated = [node for node in node_matches if node.node_id in ground_truth]
    if not evaluated:
        raise ValueError("ground_truth does not overlap current node matches")
    def method_metrics(method: str) -> dict[str, Any]:
        top1 = 0
        recalls = {int(k): 0 for k in recall_ks}
        ambiguous = 0
        wrong_high_confidence = 0
        for node in evaluated:
            truth = ground_truth[node.node_id]
            rank_key = "feature_rank" if method == "feature_only" else "refined_rank"
            ranked_candidates = sorted(
                (candidate for candidate in node.candidates if candidate.get(rank_key) is not None),
                key=lambda candidate: int(candidate[rank_key]),
            )
            ranked = [str(candidate["area_id"]) for candidate in ranked_candidates]
            if ranked and ranked[0] == truth:
                top1 += 1
            for k in recalls:
                if truth in ranked[:k]:
                    recalls[k] += 1
            status = node.feature_only_status if method == "feature_only" else node.status
            selected = (
                node.feature_only_selected_area_id
                if method == "feature_only"
                else node.selected_area_id
            )
            if status == "AMBIGUOUS":
                ambiguous += 1
            elif selected != truth:
                wrong_high_confidence += 1
        count = len(evaluated)
        return {
            "top1_accuracy": top1 / count,
            "recall_at": {str(k): value / count for k, value in recalls.items()},
            "ambiguous_rate": ambiguous / count,
            "wrong_high_confidence_match_rate": wrong_high_confidence / count,
        }

    corrections: list[str] = []
    harms: list[str] = []
    for node in evaluated:
        truth = ground_truth[node.node_id]
        feature_ranked = sorted(
            (candidate for candidate in node.candidates if candidate.get("feature_rank") is not None),
            key=lambda candidate: int(candidate["feature_rank"]),
        )
        refined_ranked = sorted(
            (candidate for candidate in node.candidates if candidate.get("refined_rank") is not None),
            key=lambda candidate: int(candidate["refined_rank"]),
        )
        feature_top = str(feature_ranked[0]["area_id"]) if feature_ranked else None
        refined_top = str(refined_ranked[0]["area_id"]) if refined_ranked else None
        if feature_top != truth and refined_top == truth:
            corrections.append(node.node_id)
        elif feature_top == truth and refined_top != truth:
            harms.append(node.node_id)
    return {
        "evaluated_nodes": len(evaluated),
        "feature_only": method_metrics("feature_only"),
        "feature_plus_topology": method_metrics("feature_plus_topology"),
        "topology_corrected_count": len(corrections),
        "topology_corrected_node_ids": corrections,
        "topology_harmed_count": len(harms),
        "topology_harmed_node_ids": harms,
    }


def render_visibility_visualization(
    reference_rgb: np.ndarray | Mapping[str, np.ndarray],
    current_rgb: np.ndarray,
    graph: CanonicalSurfaceGraph,
    result: VisibilityResult,
    *,
    output_path: Path | None = None,
) -> np.ndarray:
    """Render reference graph, current matches, statuses, and frontiers."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional visualization dependency
        raise RuntimeError("OpenCV is required for canonical graph visualization") from exc
    current = cv2.cvtColor(np.asarray(current_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR).copy()

    if isinstance(reference_rgb, Mapping):
        reference_sources = [
            (str(side).strip().upper(), np.asarray(image, dtype=np.uint8))
            for side, image in reference_rgb.items()
        ]
        required_sides = {
            area.surface_side
            for area in graph.areas.values()
            if area.surface_side is not None
        }
        missing_sides = sorted(required_sides - {side for side, _ in reference_sources})
        if missing_sides:
            raise ValueError(f"reference images missing surface sides: {missing_sides}")
    else:
        reference_sources = [("REFERENCE", np.asarray(reference_rgb, dtype=np.uint8))]

    def color(status: str) -> tuple[int, int, int]:
        return {
            "VISIBLE": (40, 210, 60),
            "AMBIGUOUS": (0, 215, 255),
            "UNOBSERVED": (115, 115, 115),
        }.get(status, (155, 155, 155))

    frontier_targets = {
        str(frontier["hidden_or_ambiguous_area_id"])
        for frontier in result.frontiers
    }
    frontier_edges = {
        tuple(sorted((str(frontier["visible_area_id"]), str(frontier["hidden_or_ambiguous_area_id"]))))
        for frontier in result.frontiers
    }

    # Keep FRONT and BACK reference coordinates on their own image panels.
    references: list[np.ndarray] = []
    multi_reference = isinstance(reference_rgb, Mapping)
    for side, source in reference_sources:
        reference = cv2.cvtColor(source, cv2.COLOR_RGB2BGR).copy()
        panel_area_ids = {
            area_id
            for area_id, area in graph.areas.items()
            if not multi_reference or area.surface_side == side
        }
        for area_id in panel_area_ids:
            area = graph.areas[area_id]
            x, y = map(int, map(round, area.canonical_xy))
            for neighbor in area.neighbor_area_ids:
                if area_id >= neighbor or neighbor not in panel_area_ids:
                    continue
                other = graph.areas[neighbor]
                edge_color = (
                    (30, 30, 230)
                    if tuple(sorted((area_id, neighbor))) in frontier_edges
                    else (180, 180, 180)
                )
                cv2.line(
                    reference,
                    (x, y),
                    tuple(map(int, map(round, other.canonical_xy))),
                    edge_color,
                    2 if edge_color[2] > 200 else 1,
                    cv2.LINE_AA,
                )
            status = result.canonical_areas[area_id]["status"]
            area_color = (
                (30, 30, 230)
                if area_id in frontier_targets and status == "UNOBSERVED"
                else color(status)
            )
            cv2.circle(reference, (x, y), 7, area_color, -1, cv2.LINE_AA)
            cv2.putText(reference, area_id, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(reference, area_id, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        if multi_reference:
            cv2.rectangle(reference, (0, 0), (170, 28), (0, 0, 0), -1)
            cv2.putText(reference, f"REFERENCE {side}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        references.append(reference)

    # Draw G_O(t) with its comparison against canonical G_M, then every node. Text for
    # every DINO patch would cover the image, so exact mappings are placed in
    # a compact side panel below for ambiguity nodes and per-area exemplars.
    node_by_id = {node.node_id: node for node in result.current_nodes}
    observation_relation_by_edge = {
        tuple(sorted(map(str, adjacency["current_node_ids"]))): str(adjacency["relation"])
        for adjacency in result.observation_adjacencies
    }
    observation_edge_colors = {
        "MATERIAL_CONSISTENT": (60, 190, 70),
        "UNEXPECTED_ADJACENCY": (40, 40, 230),
        "AMBIGUOUS": (0, 210, 255),
    }
    for left, right in result.current_edges:
        if left not in node_by_id or right not in node_by_id:
            continue
        left_xy = tuple(map(int, map(round, node_by_id[left].pixel_xy)))
        right_xy = tuple(map(int, map(round, node_by_id[right].pixel_xy)))
        relation = observation_relation_by_edge.get(tuple(sorted((left, right))), "AMBIGUOUS")
        cv2.line(
            current,
            left_xy,
            right_xy,
            observation_edge_colors[relation],
            2 if relation == "UNEXPECTED_ADJACENCY" else 1,
            cv2.LINE_AA,
        )
    for node in result.current_nodes:
        x, y = map(int, map(round, node.pixel_xy))
        cv2.circle(current, (x, y), 5, color(node.status), -1, cv2.LINE_AA)

    height = max(280, current.shape[0], *(reference.shape[0] for reference in references))
    def fit(image: np.ndarray) -> np.ndarray:
        if image.shape[0] == height:
            return image
        width = max(1, round(image.shape[1] * height / image.shape[0]))
        interpolation = cv2.INTER_AREA if image.shape[0] > height else cv2.INTER_NEAREST
        return cv2.resize(image, (width, height), interpolation=interpolation)
    references = [fit(reference) for reference in references]
    current = fit(current)
    body = np.concatenate([*references, current], axis=1)

    ambiguous_nodes = sorted(
        (node for node in result.current_nodes if node.status == "AMBIGUOUS"),
        key=lambda node: node.confidence,
        reverse=True,
    )
    exemplar_by_area: dict[str, NodeMatch] = {}
    for node in result.current_nodes:
        if node.selected_area_id is None:
            continue
        previous = exemplar_by_area.get(node.selected_area_id)
        if previous is None or node.confidence > previous.confidence:
            exemplar_by_area[node.selected_area_id] = node
    panel_nodes: list[NodeMatch] = []
    seen_nodes: set[str] = set()
    for node in ambiguous_nodes + list(exemplar_by_area.values()):
        if node.node_id not in seen_nodes:
            panel_nodes.append(node)
            seen_nodes.add(node.node_id)

    panel_width = 330
    header_height = 42
    canvas = np.zeros((body.shape[0] + header_height, body.shape[1] + panel_width, 3), dtype=np.uint8)
    canvas[header_height:, : body.shape[1]] = body
    canvas[header_height:, body.shape[1] :] = (32, 32, 32)
    relation_counts = {
        relation: sum(
            adjacency["relation"] == relation
            for adjacency in result.observation_adjacencies
        )
        for relation in ("MATERIAL_CONSISTENT", "UNEXPECTED_ADJACENCY", "AMBIGUOUS")
    }
    title = (
        "Canonical visibility | G_O vs G_M: "
        f"consistent={relation_counts['MATERIAL_CONSISTENT']} "
        f"unexpected={relation_counts['UNEXPECTED_ADJACENCY']} "
        f"ambiguous={relation_counts['AMBIGUOUS']} | reference(s) / current"
    )
    cv2.putText(canvas, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    panel_x = body.shape[1] + 10
    cv2.putText(canvas, "Representative node matches", (panel_x, header_height + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, "node -> canonical  confidence", (panel_x, header_height + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    max_lines = max(1, (body.shape[0] - 72) // 18)
    for row, node in enumerate(panel_nodes[:max_lines]):
        candidate_id = node.selected_area_id
        if candidate_id is None and node.candidates:
            candidate_id = str(node.candidates[0]["area_id"]) + "?"
        label = f"{node.node_id} -> {candidate_id or '?'}  {node.confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            (panel_x, header_height + 64 + 18 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color(node.status),
            1,
            cv2.LINE_AA,
        )
    omitted = len(panel_nodes) - max_lines
    if omitted > 0:
        cv2.putText(
            canvas,
            f"... {omitted} more; see visibility JSON",
            (panel_x, header_height + body.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
