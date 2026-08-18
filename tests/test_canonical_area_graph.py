from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cloth_agent.canonical_area_graph import (
    CanonicalSurfaceGraph,
    bidirectional_chamfer_similarity,
    build_current_visible_graph,
    constrained_geodesic_voronoi,
    evaluate_visibility,
    match_visibility,
    render_visibility_visualization,
)
from scripts.canonical_area_graph import (
    _propose_planar_neighbors,
    prepare_mask,
    prepare_reference_mask,
)


def _prototype_graph() -> CanonicalSurfaceGraph:
    points = [
        {"id": 1, "label": "left", "x": 0, "y": 0},
        {"id": 2, "label": "middle", "x": 100, "y": 0},
        {"id": 3, "label": "right", "x": 200, "y": 0},
        {"id": 4, "label": "distractor", "x": 300, "y": 0},
    ]
    reference_features = np.asarray(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.8, 0.6, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    return CanonicalSurfaceGraph.from_reference_points(
        points,
        reference_features,
        image_size=(301, 1),
        sample_radius=0,
        explicit_neighbors={"A01": ["A02"], "A02": ["A03"], "A03": [], "A04": []},
    )


def test_canonical_graph_round_trip_preserves_feature_banks(tmp_path: Path) -> None:
    graph = _prototype_graph()
    graph_path, feature_path = graph.save(tmp_path)
    assert graph_path.is_file()
    assert feature_path.is_file()
    loaded = CanonicalSurfaceGraph.load(graph_path)
    assert loaded.neighbors("A02") == ("A01", "A03")
    assert np.allclose(loaded.feature_bank, graph.feature_bank)


def test_area_feature_bank_samples_unique_local_patch_neighborhood() -> None:
    features = np.arange(3 * 3 * 2, dtype=np.float32).reshape(3, 3, 2)
    graph = CanonicalSurfaceGraph.from_reference_points(
        [{"id": 1, "x": 1, "y": 1}],
        features,
        image_size=(3, 3),
        sample_radius=1,
        explicit_neighbors={"A01": []},
    )
    indices = graph.areas["A01"].feature_sample_indices
    assert len(indices) == 9
    assert len(np.unique(graph.feature_bank[indices], axis=0)) == 9


def test_bidirectional_chamfer_does_not_accept_one_lucky_token() -> None:
    reference = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    exact = reference.copy()
    one_lucky = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    assert bidirectional_chamfer_similarity(reference, exact) == pytest.approx(1.0)
    assert bidirectional_chamfer_similarity(reference, one_lucky) == pytest.approx(0.5)


def test_constrained_geodesic_voronoi_stays_inside_garment_mask() -> None:
    mask = np.asarray(
        [
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
        ],
        dtype=bool,
    )
    labels, snapped = constrained_geodesic_voronoi(mask, [(0, 0), (4, 0)])
    assert snapped == [(0, 0), (4, 0)]
    assert np.all(labels[~mask] == -1)
    assert set(np.unique(labels[mask])) == {0, 1}


def test_reference_mask_builds_persisted_material_regions(tmp_path: Path) -> None:
    features = np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4)
    mask = np.ones((3, 5), dtype=bool)
    graph = CanonicalSurfaceGraph.from_reference_points(
        [
            {"area_id": "B01", "surface_side": "BACK", "x": 0, "y": 1},
            {"area_id": "B02", "surface_side": "BACK", "x": 4, "y": 1},
        ],
        features,
        image_size=(5, 3),
        sample_radius=1,
        explicit_neighbors={"B01": ["B02"], "B02": []},
        reference_mask=mask,
    )
    assert graph.metadata["canonical_region_method"] == "constrained_geodesic_voronoi_on_feature_grid"
    assert set(graph.region_labels_by_side) == {"BACK"}
    assert sum(len(area.region_patch_xy) for area in graph.areas.values()) == int(mask.sum())
    assert all(area.region_feature_sample_indices for area in graph.areas.values())
    graph_path, _ = graph.save(tmp_path / "graph")
    loaded = CanonicalSurfaceGraph.load(graph_path)
    assert np.array_equal(
        loaded.region_labels_by_side["BACK"],
        graph.region_labels_by_side["BACK"],
    )
    assert loaded.areas["B01"].region_patch_xy == graph.areas["B01"].region_patch_xy


def test_knn_adjacency_prefers_calibrated_flat_reference_xy() -> None:
    graph = CanonicalSurfaceGraph.from_reference_points(
        [
            {
                "id": 1,
                "area_id": "F01",
                "surface_side": "FRONT",
                "x": 0,
                "y": 0,
                "base_xyz_mm": [0, 0, 5],
            },
            {
                "id": 2,
                "area_id": "F02",
                "surface_side": "FRONT",
                "x": 100,
                "y": 0,
                "base_xyz_mm": [1000, 0, 5],
            },
            {
                "id": 3,
                "area_id": "F03",
                "surface_side": "FRONT",
                "x": 200,
                "y": 0,
                "base_xyz_mm": [10, 0, 5],
            },
        ],
        np.asarray([[[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]], dtype=np.float32),
        image_size=(201, 1),
        sample_radius=0,
        neighbor_k=1,
    )
    assert graph.neighbors("F01") == ("F03",)
    assert graph.areas["F01"].surface_side == "FRONT"
    assert graph.areas["F01"].canonical_base_xyz_mm == (0.0, 0.0, 5.0)
    assert graph.metadata["adjacency_coordinate_space"] == "calibrated_base_xy_mm"


def test_planar_neighbor_proposal_prunes_long_delaunay_edges() -> None:
    points = [
        {"id": 1, "area_id": "F01", "base_xyz_mm": [0, 0, 0]},
        {"id": 2, "area_id": "F02", "base_xyz_mm": [100, 0, 0]},
        {"id": 3, "area_id": "F03", "base_xyz_mm": [0, 100, 0]},
        {"id": 4, "area_id": "F04", "base_xyz_mm": [100, 100, 0]},
        {"id": 5, "area_id": "F05", "base_xyz_mm": [300, 0, 0]},
        {"id": 6, "area_id": "F06", "base_xyz_mm": [300, 100, 0]},
    ]
    neighbors, diagnostics = _propose_planar_neighbors(
        points,
        max_edge_length_mm=180.0,
        max_edge_factor=1.25,
    )
    assert "F06" in neighbors["F05"]
    assert "F05" not in neighbors["F02"]
    assert "F06" not in neighbors["F04"]
    assert "F05" not in neighbors["F01"]
    assert "F05" not in neighbors["F03"]
    assert diagnostics["proposal_only"] is True
    assert diagnostics["retained_edge_count"] < diagnostics["delaunay_edge_count"]


def test_combined_surface_graph_keeps_front_and_back_disconnected(tmp_path: Path) -> None:
    features = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    front = CanonicalSurfaceGraph.from_reference_points(
        [
            {"area_id": "F01", "surface_side": "FRONT", "x": 0, "y": 0},
            {"area_id": "F02", "surface_side": "FRONT", "x": 1, "y": 0},
        ],
        features,
        image_size=(2, 1),
        sample_radius=0,
        explicit_neighbors={"F01": ["F02"], "F02": []},
    )
    back = CanonicalSurfaceGraph.from_reference_points(
        [
            {"area_id": "B01", "surface_side": "BACK", "x": 0, "y": 0},
            {"area_id": "B02", "surface_side": "BACK", "x": 1, "y": 0},
        ],
        features,
        image_size=(2, 1),
        sample_radius=0,
        explicit_neighbors={"B01": ["B02"], "B02": []},
    )
    combined = CanonicalSurfaceGraph.combine([front, back])
    assert combined.area_ids == ("F01", "F02", "B01", "B02")
    assert combined.neighbors("F01") == ("F02",)
    assert combined.neighbors("B01") == ("B02",)
    assert not any(neighbor.startswith("B") for neighbor in combined.neighbors("F01"))
    assert not any(neighbor.startswith("F") for neighbor in combined.neighbors("B01"))
    assert combined.feature_bank.shape == (4, 2)
    assert combined.areas["B01"].feature_sample_indices == [2]
    assert combined.metadata["automatic_cross_surface_edges"] is False

    result = match_visibility(
        combined,
        features,
        image_size=(2, 1),
        topology_lambda=0.0,
        top_k=4,
    )
    output = tmp_path / "combined.png"
    rendered = render_visibility_visualization(
        {
            "FRONT": np.full((40, 60, 3), 30, dtype=np.uint8),
            "BACK": np.full((40, 60, 3), 60, dtype=np.uint8),
        },
        np.full((40, 60, 3), 90, dtype=np.uint8),
        combined,
        result,
        output_path=output,
    )
    assert output.is_file()
    assert rendered.shape[1] > 3 * 60


def test_topology_corrects_feature_only_distractor() -> None:
    graph = _prototype_graph()
    current_features = np.asarray(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 0.7, 0.714, 0.0], [0.0, 0.0, 1.0, 0.0]]],
        dtype=np.float32,
    )
    result = match_visibility(
        graph,
        current_features,
        image_size=(201, 1),
        top_k=4,
        topology_lambda=1.0,
        adjacency_similarity_threshold=-1.0,
        confident_threshold=0.5,
        ambiguity_margin=0.02,
    )
    middle = next(node for node in result.current_nodes if node.node_id == "X000_001")
    assert middle.feature_only_selected_area_id == "A04"
    assert middle.selected_area_id == "A02"
    evaluation = evaluate_visibility(
        result.current_nodes,
        {"X000_000": "A01", "X000_001": "A02", "X000_002": "A03"},
    )
    assert evaluation["feature_only"]["top1_accuracy"] == 2 / 3
    assert evaluation["feature_plus_topology"]["top1_accuracy"] == 1.0
    assert evaluation["topology_corrected_node_ids"] == ["X000_001"]
    assert evaluation["topology_harmed_count"] == 0


def test_dense_neighbor_in_same_canonical_area_provides_topology_support() -> None:
    graph = CanonicalSurfaceGraph.from_reference_points(
        [{"id": 1, "x": 0, "y": 0}, {"id": 2, "x": 1, "y": 0}],
        np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        image_size=(2, 1),
        sample_radius=0,
        explicit_neighbors={"A01": ["A02"], "A02": []},
    )
    result = match_visibility(
        graph,
        np.asarray([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32),
        image_size=(2, 1),
        topology_lambda=0.5,
        adjacency_similarity_threshold=-1.0,
        top_k=2,
    )
    for node in result.current_nodes:
        a01 = next(candidate for candidate in node.candidates if candidate["area_id"] == "A01")
        assert a01["topology_bonus"] > 0.0
    assert result.parameters["topology_compatibility"] == "same_canonical_area_or_intrinsic_neighbor"


def test_unexpected_observation_adjacency_is_recorded_without_negative_penalty() -> None:
    graph = CanonicalSurfaceGraph.from_reference_points(
        [
            {"id": 1, "x": 0, "y": 0},
            {"id": 2, "x": 1, "y": 0},
            {"id": 3, "x": 2, "y": 0},
        ],
        np.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32),
        image_size=(3, 1),
        sample_radius=0,
        explicit_neighbors={"A01": ["A02"], "A02": ["A03"], "A03": []},
    )
    result = match_visibility(
        graph,
        np.asarray([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32),
        image_size=(2, 1),
        topology_lambda=0.25,
        adjacency_similarity_threshold=-1.0,
        ambiguity_margin=0.01,
        confident_threshold=0.5,
        top_k=3,
    )
    assert [node.selected_area_id for node in result.current_nodes] == ["A01", "A03"]
    adjacency = result.observation_adjacencies[0]
    assert adjacency["relation"] == "UNEXPECTED_ADJACENCY"
    assert adjacency["canonical_graph_distance"] == 2
    assert adjacency["interpretation"] == "possible fold/contact/occlusion boundary"
    assert all(
        candidate["topology_bonus"] >= -1e-7
        for node in result.current_nodes
        for candidate in node.candidates
    )
    assert "no penalty" in result.parameters["topology_policy"]


def test_unobserved_neighbor_becomes_visible_hidden_frontier() -> None:
    graph = _prototype_graph()
    current_features = np.asarray(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
        dtype=np.float32,
    )
    result = match_visibility(
        graph,
        current_features,
        image_size=(201, 1),
        valid_mask=np.asarray([[True, True, False]]),
        top_k=4,
        topology_lambda=0.2,
        adjacency_similarity_threshold=-1.0,
        confident_threshold=0.5,
        ambiguity_margin=0.02,
    )
    assert result.canonical_areas["A03"]["status"] == "UNOBSERVED"
    assert any(
        frontier["visible_area_id"] == "A02"
        and frontier["hidden_or_ambiguous_area_id"] == "A03"
        for frontier in result.frontiers
    )


def test_large_3d_gap_is_unknown_not_material_adjacency() -> None:
    features = np.asarray([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
    xyz = np.asarray([[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]], dtype=np.float32)
    _, edges, unknown = build_current_visible_graph(
        features,
        image_size=(2, 1),
        surface_xyz_mm=xyz,
        max_surface_gap_mm=30.0,
        adjacency_similarity_threshold=0.5,
    )
    assert edges == []
    assert unknown == [("X000_000", "X000_001")]


def test_close_candidates_both_remain_ambiguous_canonical_areas() -> None:
    graph = CanonicalSurfaceGraph.from_reference_points(
        [
            {"id": 1, "x": 0, "y": 0},
            {"id": 2, "x": 1, "y": 0},
        ],
        np.asarray([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32),
        image_size=(2, 1),
        sample_radius=0,
        explicit_neighbors={"A01": [], "A02": []},
    )
    result = match_visibility(
        graph,
        np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        image_size=(1, 1),
        top_k=2,
        topology_lambda=0.0,
        ambiguity_margin=0.01,
        confident_threshold=0.5,
    )
    assert result.current_nodes[0].status == "AMBIGUOUS"
    assert result.current_nodes[0].confidence == 1.0
    assert result.canonical_areas["A01"]["status"] == "AMBIGUOUS"
    assert result.canonical_areas["A02"]["status"] == "AMBIGUOUS"


def test_precomputed_feature_cli_supports_different_current_grid(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    current = tmp_path / "folded.png"
    mask = tmp_path / "mask.png"
    Image.fromarray(np.full((100, 301, 3), 80, dtype=np.uint8)).save(reference)
    Image.fromarray(np.full((100, 201, 3), 45, dtype=np.uint8)).save(current)
    Image.fromarray(np.full((100, 201), 255, dtype=np.uint8)).save(mask)

    reference_features = np.asarray(
        [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0.8, 0.6, 0]]],
        dtype=np.float32,
    )
    current_features = np.asarray(
        [[[1, 0, 0, 0], [0, 0.7, 0.714, 0], [0, 0, 1, 0]]],
        dtype=np.float32,
    )
    reference_features_path = tmp_path / "reference_features.npy"
    current_features_path = tmp_path / "current_features.npy"
    ambiguous_features_path = tmp_path / "ambiguous_features.npy"
    np.save(reference_features_path, reference_features)
    np.save(current_features_path, current_features)
    np.save(
        ambiguous_features_path,
        np.asarray([[[0.0, 1.8, 0.6, 0.0]]], dtype=np.float32),
    )

    points_path = tmp_path / "points.json"
    points_path.write_text(
        json.dumps(
            {
                "points": [
                    {"id": 1, "x": 0, "y": 50},
                    {"id": 2, "x": 100, "y": 50},
                    {"id": 3, "x": 200, "y": 50},
                    {"id": 4, "x": 300, "y": 50},
                ]
            }
        ),
        encoding="utf-8",
    )
    neighbors_path = tmp_path / "neighbors.json"
    neighbors_path.write_text(
        json.dumps({"A01": ["A02"], "A02": ["A03"], "A03": [], "A04": []}),
        encoding="utf-8",
    )
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "version": 1,
                "ground_truth": {
                    "X000_000": "A01",
                    "X000_001": "A02",
                    "X000_002": "A03",
                },
            }
        ),
        encoding="utf-8",
    )

    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "canonical_area_graph.py"
    graph_dir = tmp_path / "graph"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "build",
            "--reference",
            str(reference),
            "--points",
            str(points_path),
            "--neighbors",
            str(neighbors_path),
            "--reference-features",
            str(reference_features_path),
            "--sample-radius",
            "0",
            "--output",
            str(graph_dir),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    match_dir = tmp_path / "matches"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "match",
            "--graph",
            str(graph_dir / "canonical_graph.json"),
            "--reference",
            str(reference),
            "--current",
            f"folded={current}",
            "--current",
            f"ambiguous={current}",
            "--current-features",
            f"folded={current_features_path}",
            "--current-features",
            f"ambiguous={ambiguous_features_path}",
            "--mask",
            f"folded={mask}",
            "--ground-truth",
            f"folded={truth_path}",
            "--topology-lambda",
            "1.0",
            "--adjacency-similarity-threshold",
            "-1",
            "--confident-threshold",
            "0.5",
            "--ambiguity-margin",
            "0.02",
            "--output",
            str(match_dir),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((match_dir / "summary.json").read_text(encoding="utf-8"))["folded"]
    assert summary["evaluation"]["feature_only"]["top1_accuracy"] == 2 / 3
    assert summary["evaluation"]["feature_plus_topology"]["top1_accuracy"] == 1.0
    assert (match_dir / summary["visualization"]).is_file()
    ambiguous_result = json.loads(
        (match_dir / "ambiguous_visibility.json").read_text(encoding="utf-8")
    )
    assert len(ambiguous_result["ambiguities"]) == 1
    crop = ambiguous_result["ambiguities"][0]["current_crop"]
    assert (match_dir / crop["path"]).is_file()


def test_prepare_mask_reuses_saved_fused_perception_artifacts(tmp_path: Path) -> None:
    image = np.full((20, 20, 3), 80, dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "camera_A.png")
    np.save(tmp_path / "camera_A_depth.npy", np.ones((20, 20), dtype=np.float32))
    np.save(tmp_path / "camera_A_height.npy", np.zeros((20, 20), dtype=np.float32))
    axis = np.linspace(-90.0, 90.0, 10)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack(
        [xx.reshape(-1), yy.reshape(-1), np.full(xx.size, 1000.0)]
    )
    np.save(tmp_path / "fused_points.npy", points)
    np.save(tmp_path / "fused_mask.npy", np.ones(len(points), dtype=bool))
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "views": [
                    {
                        "label": "A",
                        "serial": "synthetic",
                        "image": "camera_A.png",
                        "depth_m": "camera_A_depth.npy",
                        "height_map_path": "camera_A_height.npy",
                        "intrinsics": [[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]],
                        "X_base_camera": np.eye(4).tolist(),
                    }
                ],
                "depth_fusion": {
                    "artifacts": {
                        "fused_points_base_mm": "fused_points.npy",
                        "fused_garment_mask": "fused_mask.npy",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "camera_A_mask.png"
    assert (
        prepare_mask(
            argparse.Namespace(result=result_path, camera_label="A", output=output)
        )
        == 0
    )
    assert output.is_file()
    mask = np.asarray(Image.open(output).convert("L"), dtype=np.uint8) > 0
    assert int(mask.sum()) > 100
    diagnostics = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert diagnostics["garment_mask_pixels"] == int(mask.sum())


def test_prepare_reference_mask_keeps_largest_non_table_component(tmp_path: Path) -> None:
    rgb = np.full((100, 140, 3), 235, dtype=np.uint8)
    rgb[20:90, 30:120] = np.asarray([25, 25, 30], dtype=np.uint8)
    rgb[2:8, 2:8] = 0
    image_path = tmp_path / "reference.png"
    Image.fromarray(rgb).save(image_path)
    output_path = tmp_path / "mask.png"
    assert (
        prepare_reference_mask(
            argparse.Namespace(image=image_path, output=output_path)
        )
        == 0
    )
    mask = np.asarray(Image.open(output_path).convert("L"), dtype=np.uint8) > 0
    assert mask[50, 60]
    assert not mask[4, 4]
    assert not mask[5, 130]
