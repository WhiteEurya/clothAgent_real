from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from scripts.diagnose_claude_rgb_selection import (
    _aggregate,
    _annotation_template,
    _build_prompt,
    _expand_point_id_tokens,
    _image_sha256,
    _production_mask_points,
    _randomized_markers,
    _score_result,
    _uniform_grid,
    _validate_annotations,
)


def test_full_image_grid_includes_table_and_uses_stable_ids() -> None:
    points = _uniform_grid(160, 120, stride_px=32, margin_px=8)
    assert points[0]["stable_id"] == "P001"
    assert len(points) > 10
    assert any(point["pixel_xy"][0] < 40 for point in points)
    assert any(point["pixel_xy"][0] > 110 for point in points)


def test_production_candidates_follow_only_mask_cells() -> None:
    import numpy as np

    mask = np.zeros((96, 128), dtype=bool)
    mask[30:70, 45:95] = True
    points = _production_mask_points(mask, stride_px=32)
    assert points
    assert all(mask[y_px, x_px] for point in points for x_px, y_px in [point["pixel_xy"]])


def test_randomized_marker_ids_change_without_changing_pixels() -> None:
    points = _uniform_grid(160, 120, stride_px=32, margin_px=8)
    first, _ = _randomized_markers(points, seed=7, key="trial-1")
    second, _ = _randomized_markers(points, seed=7, key="trial-2")
    assert {tuple(item["pixel_xy"]) for item in first} == {tuple(item["pixel_xy"]) for item in second}
    first_by_pixel = {tuple(item["pixel_xy"]): item["reference_id"] for item in first}
    second_by_pixel = {tuple(item["pixel_xy"]): item["reference_id"] for item in second}
    assert first_by_pixel != second_by_pixel


def test_human_annotation_point_ranges_expand() -> None:
    assert _expand_point_id_tokens(["P001-P003", "P010-P012", "P020"]) == [
        "P001",
        "P002",
        "P003",
        "P010",
        "P011",
        "P012",
        "P020",
    ]


def test_annotations_are_bound_to_exact_rgb_and_condition_b_can_hit(tmp_path: Path) -> None:
    image = Image.new("RGB", (160, 120), (240, 240, 240))
    points = _uniform_grid(image.width, image.height, stride_px=32, margin_px=8)
    source = tmp_path / "camera_A.png"
    image.save(source)
    template = _annotation_template(
        source_image=source,
        production_mask=None,
        upright=image,
        rotation="none",
        stride_px=32,
        margin_px=8,
        all_points=points,
    )
    template["human_candidate_point_ids"] = ["P001-P006"]
    template["localization_tasks"][0].update(
        {"enabled": True, "accepted_point_ids": ["P001"]}
    )
    human, localization, planning = _validate_annotations(
        template,
        upright=image,
        all_points=points,
        conditions=["A", "B"],
        task_filter=None,
    )
    assert [point["stable_id"] for point in human] == [
        "P001",
        "P002",
        "P003",
        "P004",
        "P005",
        "P006",
    ]
    assert localization[0]["accepted_point_ids"] == ["P001"]
    assert planning == []

    changed = image.copy()
    changed.putpixel((0, 0), (0, 0, 0))
    assert _image_sha256(changed) != template["upright_image_sha256"]
    with pytest.raises(ValueError, match="image hash"):
        _validate_annotations(
            template,
            upright=changed,
            all_points=points,
            conditions=["A"],
            task_filter=None,
        )


def test_localization_and_planning_scoring_use_human_pixels() -> None:
    points = _uniform_grid(160, 120, stride_px=32, margin_px=8)
    by_id = {point["stable_id"]: point for point in points}
    localization = {
        "kind": "localization",
        "accepted_point_ids": ["P001"],
    }
    hit = _score_result(
        localization,
        {"selected": {"pixel_xy": by_id["P001"]["pixel_xy"]}},
        all_by_id=by_id,
        acceptance_radius_px=10,
    )
    miss = _score_result(
        localization,
        {"selected": {"pixel_xy": by_id["P010"]["pixel_xy"]}},
        all_by_id=by_id,
        acceptance_radius_px=10,
    )
    assert hit["hit"] is True
    assert miss["hit"] is False

    planning = {
        "kind": "planning",
        "acceptable_transfers": [
            {"source_point_ids": ["P001"], "destination_point_ids": ["P002"]}
        ],
    }
    score = _score_result(
        planning,
        {
            "source": {"pixel_xy": by_id["P001"]["pixel_xy"]},
            "destination": {"pixel_xy": by_id["P002"]["pixel_xy"]},
        },
        all_by_id=by_id,
        acceptance_radius_px=10,
    )
    assert score["hit"] is True


def test_prompt_contract_is_rgb_only_and_aggregate_reports_consistency() -> None:
    task = {"kind": "localization", "id": "cuff", "instruction": "select cuff"}
    prompt = _build_prompt(task, marker_count=20).lower()
    assert "rgb-only" in prompt
    assert "do not infer or request depth" in prompt
    records = [
        {
            "task_kind": "localization",
            "task_id": "cuff",
            "condition": "A",
            "status": "COMPLETED",
            "score": {"hit": True},
            "resolved": {"selected": {"pixel_xy": [10, 10]}},
        },
        {
            "task_kind": "localization",
            "task_id": "cuff",
            "condition": "A",
            "status": "COMPLETED",
            "score": {"hit": False},
            "resolved": {"selected": {"pixel_xy": [20, 10]}},
        },
    ]
    aggregate = _aggregate(records)[0]
    assert aggregate["hit_rate"] == pytest.approx(0.5)
    assert aggregate["selection_spread"]["max_px"] == pytest.approx(10.0)
