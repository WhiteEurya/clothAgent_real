from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from cloth_agent.canonical_area_graph import CanonicalSurfaceGraph
from scripts.canonical_area_annotate import (
    AnnotationState,
    _load_existing_assignments,
    _point_map_from_visibility,
    annotate,
    nearest_id,
    save_annotations,
    sparse_node_points,
)


def test_nearest_id_respects_selection_radius() -> None:
    points = {"X0": (10.0, 10.0), "X1": (30.0, 10.0)}
    assert nearest_id(points, (12.0, 11.0), max_distance_px=5.0) == "X0"
    assert nearest_id(points, (20.0, 30.0), max_distance_px=5.0) is None
    with pytest.raises(ValueError, match="positive"):
        nearest_id(points, (10.0, 10.0), max_distance_px=0.0)


def test_sparse_node_points_thins_regular_patch_grid() -> None:
    points = {
        f"X{py:03d}_{px:03d}": (float(px), float(py))
        for py in range(8)
        for px in range(8)
    }
    selected = sparse_node_points(points, stride=4)
    assert set(selected) == {
        "X000_000",
        "X000_004",
        "X004_000",
        "X004_004",
    }


def test_annotation_state_assign_remove_and_undo() -> None:
    state = AnnotationState()
    state.selected_node_id = "X0"
    state.assign("X0", "A01")
    assert state.assignments == {"X0": "A01"}
    assert state.selected_node_id is None

    state.assign("X0", "A02")
    assert state.assignments == {"X0": "A02"}
    assert state.undo()
    assert state.assignments == {"X0": "A01"}

    state.remove("X0")
    assert state.assignments == {}
    assert state.undo()
    assert state.assignments == {"X0": "A01"}

    assert state.undo()
    assert state.assignments == {}
    assert not state.undo()


def test_visibility_points_and_annotation_resume_round_trip(tmp_path: Path) -> None:
    points = _point_map_from_visibility(
        {
            "current_nodes": [
                {"node_id": "X000_000", "pixel_xy": [12.5, 20.0]},
                {"node_id": "X000_001", "pixel_xy": [30.0, 20.0]},
            ]
        }
    )
    assert points["X000_000"] == (12.5, 20.0)

    output = tmp_path / "truth.json"
    save_annotations(
        output,
        {"X000_001": "A02", "X000_000": "A01"},
        graph_path=tmp_path / "graph.json",
        visibility_path=tmp_path / "visibility.json",
        reference_path=tmp_path / "reference.png",
        current_path=tmp_path / "current.png",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["evaluated_node_count"] == 2
    assert list(payload["ground_truth"]) == ["X000_000", "X000_001"]
    resumed = _load_existing_assignments(
        output,
        valid_node_ids=set(points),
        valid_area_ids={"A01", "A02"},
    )
    assert resumed == {"X000_000": "A01", "X000_001": "A02"}


def test_resume_rejects_stale_graph_or_visibility_ids(tmp_path: Path) -> None:
    output = tmp_path / "truth.json"
    output.write_text(
        json.dumps({"ground_truth": {"OLD_NODE": "OLD_AREA"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incompatible"):
        _load_existing_assignments(
            output,
            valid_node_ids={"X000_000"},
            valid_area_ids={"A01"},
        )


def test_headless_click_sequence_assigns_and_saves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path = tmp_path / "reference.png"
    current_path = tmp_path / "current.png"
    Image.fromarray(np.full((100, 200, 3), 80, dtype=np.uint8)).save(reference_path)
    Image.fromarray(np.full((100, 200, 3), 50, dtype=np.uint8)).save(current_path)
    graph = CanonicalSurfaceGraph.from_reference_points(
        [{"id": 1, "x": 50, "y": 50}, {"id": 2, "x": 150, "y": 50}],
        np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        image_size=(200, 100),
        sample_radius=0,
        explicit_neighbors={"A01": ["A02"], "A02": []},
    )
    graph_path, _ = graph.save(tmp_path / "graph")
    visibility_path = tmp_path / "visibility.json"
    visibility_path.write_text(
        json.dumps(
            {
                "current_nodes": [
                    {"node_id": "X000_000", "pixel_xy": [50.0, 50.0]}
                ]
            }
        ),
        encoding="utf-8",
    )

    callback_holder: dict[str, object] = {}
    monkeypatch.setattr(cv2, "namedWindow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(
        cv2,
        "setMouseCallback",
        lambda _window, callback: callback_holder.update(callback=callback),
    )

    def click_then_quit(_delay: int) -> int:
        callback = callback_holder["callback"]
        assert callable(callback)
        # Panels are 200 px wide and start below the 38 px header.
        callback(cv2.EVENT_LBUTTONDOWN, 250, 88, 0, None)
        callback(cv2.EVENT_LBUTTONDOWN, 50, 88, 0, None)
        return ord("q")

    monkeypatch.setattr(cv2, "waitKey", click_then_quit)
    output_path = tmp_path / "truth.json"
    annotate(
        graph_path,
        visibility_path,
        reference_path,
        current_path,
        output_path,
        selection_radius_px=20.0,
        max_display_width=400,
        max_display_height=200,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["ground_truth"] == {"X000_000": "A01"}


def test_headless_multi_reference_can_assign_back_area(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    front_path = tmp_path / "front.png"
    back_path = tmp_path / "back.png"
    current_path = tmp_path / "current.png"
    for path, value in ((front_path, 30), (back_path, 60), (current_path, 90)):
        Image.fromarray(np.full((100, 200, 3), value, dtype=np.uint8)).save(path)
    features = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    front = CanonicalSurfaceGraph.from_reference_points(
        [{"area_id": "F01", "surface_side": "FRONT", "x": 50, "y": 50}],
        features,
        image_size=(200, 100),
        sample_radius=0,
        explicit_neighbors={"F01": []},
    )
    back = CanonicalSurfaceGraph.from_reference_points(
        [{"area_id": "B01", "surface_side": "BACK", "x": 150, "y": 50}],
        features,
        image_size=(200, 100),
        sample_radius=0,
        explicit_neighbors={"B01": []},
    )
    graph_path, _ = CanonicalSurfaceGraph.combine([front, back]).save(tmp_path / "graph")
    visibility_path = tmp_path / "visibility.json"
    visibility_path.write_text(
        json.dumps({"current_nodes": [{"node_id": "X0", "pixel_xy": [50, 50]}]}),
        encoding="utf-8",
    )
    callback_holder: dict[str, object] = {}
    monkeypatch.setattr(cv2, "namedWindow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(
        cv2,
        "setMouseCallback",
        lambda _window, callback: callback_holder.update(callback=callback),
    )

    def click_back_then_quit(_delay: int) -> int:
        callback = callback_holder["callback"]
        assert callable(callback)
        # Three 200 px panels: FRONT, BACK, CURRENT. Select B01 then its current location.
        callback(cv2.EVENT_LBUTTONDOWN, 350, 88, 0, None)
        callback(cv2.EVENT_LBUTTONDOWN, 450, 88, 0, None)
        return ord("q")

    monkeypatch.setattr(cv2, "waitKey", click_back_then_quit)
    output_path = tmp_path / "truth.json"
    annotate(
        graph_path,
        visibility_path,
        {"FRONT": front_path, "BACK": back_path},
        current_path,
        output_path,
        selection_radius_px=20.0,
        max_display_width=600,
        max_display_height=200,
        reference_first=True,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["ground_truth"] == {"X0": "B01"}
    assert set(payload["reference_images"]) == {"FRONT", "BACK"}
