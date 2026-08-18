from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from cloth_agent.garment_grounding_mcp import GarmentGrounding
from cloth_agent.molmo_keypoint_pipeline import (
    CONFIDENCE_DEFINITION,
    KeypointSpec,
    MolmoKeypointPipelineError,
    build_confidence_filtered_references,
    run_molmo_keypoint_pipeline,
)
from cloth_agent.molmo_keypoint_worker import (
    geometric_mean_probability,
    point_location_probabilities,
)


SPECS = (
    KeypointSpec("center", "garment center", (255, 0, 0)),
    KeypointSpec("corner", "visible garment corner", (0, 255, 0)),
)


def _perception_dir(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    directory = tmp_path / "perception_views"
    directory.mkdir()
    image_paths: dict[str, Path] = {}
    for index, camera in enumerate(("A", "B")):
        image_path = directory / f"camera_{index}_{camera}.png"
        Image.new("RGB", (5, 4), (20 + index, 30, 40)).save(image_path)
        image_paths[camera] = image_path
        xyz = np.zeros((4, 5, 3), dtype=np.float32)
        for y_px in range(4):
            for x_px in range(5):
                xyz[y_px, x_px] = [500 + x_px, -200 + y_px, 12 + x_px]
        height = np.full((4, 5), 7.5, dtype=np.float32)
        np.save(directory / f"camera_{camera}_base_xyz_mm.npy", xyz)
        np.save(directory / f"camera_{camera}_height_above_table_mm.npy", height)
        (directory / f"camera_{camera}_coordinate_guide.json").write_text(
            json.dumps(
                {
                    "camera_label": camera,
                    "coordinate_frame": "robot_base_mm",
                    "samples": [
                        {
                            "reference_id": "R999",
                            "pixel_xy": [0, 0],
                            "base_xyz_mm": [500, -200, 12],
                            "height_above_table_mm": 7.5,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    (directory / "observation.json").write_text(
        json.dumps({"coordinate_guides": []}), encoding="utf-8"
    )
    return directory, image_paths


def _record(
    name: str,
    *,
    status: str,
    confidence: float,
    pixel: list[float] | None,
) -> dict[str, object]:
    return {
        "name": name,
        "description": next(spec.description for spec in SPECS if spec.name == name),
        "status": status,
        "pixel_xy": pixel,
        "confidence": confidence,
        "confidence_definition": CONFIDENCE_DEFINITION,
        "point_token_probabilities": [confidence] * 3 if pixel else [],
        "generated_text": "point" if pixel else "UNKNOWN",
    }


def _payload(
    a_records: list[dict[str, object]],
    b_records: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "views": [
            {
                "label": "A",
                "image_size": [5, 4],
                "records": a_records,
            },
            {
                "label": "B",
                "image_size": [5, 4],
                "records": b_records,
            },
        ],
    }


def test_filters_strictly_above_threshold_and_installs_only_valid_references(
    tmp_path: Path,
) -> None:
    perception_dir, image_paths = _perception_dir(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    payload = _payload(
        [
            _record("center", status="point_returned", confidence=0.91, pixel=[2.0, 1.0]),
            _record("corner", status="point_returned", confidence=0.60, pixel=[3.0, 2.0]),
        ],
        [
            _record("center", status="not_found", confidence=0.0, pixel=None),
            _record("corner", status="point_returned", confidence=0.75, pixel=[4.0, 3.0]),
        ],
    )

    manifest = build_confidence_filtered_references(
        payload,
        perception_dir=perception_dir,
        artifact_dir=artifact_dir,
        image_paths=image_paths,
        cameras=("A", "B"),
        specs=SPECS,
        confidence_threshold=0.60,
        local_radius_px=0,
        install=True,
    )

    assert manifest["status"] == "READY"
    assert manifest["accepted_reference_count"] == 2
    assert {(item["camera"], item["name"]) for item in manifest["references"]} == {
        ("A", "center"),
        ("B", "corner"),
    }
    camera_a = manifest["views"][0]
    equal_threshold = next(
        item for item in camera_a["candidates"] if item["name"] == "corner"
    )
    assert equal_threshold["accepted"] is False
    assert "not_strictly_above" in equal_threshold["rejection_reason"]

    installed = json.loads(
        (perception_dir / "camera_A_coordinate_guide.json").read_text(
            encoding="utf-8"
        )
    )
    assert [sample["name"] for sample in installed["samples"]] == ["center"]
    assert installed["samples"][0]["reference_id"] == "R001"
    assert (perception_dir / "camera_A_uniform_coordinate_guide.json").is_file()

    grounded = GarmentGrounding(perception_dir).lookup_reference("A", "R001")
    assert grounded["measurement_kind"] == "molmo_confidence_filtered_grasp_reference"
    assert grounded["name"] == "center"
    assert grounded["confidence"] == pytest.approx(0.91)
    assert grounded["confidence_threshold"] == pytest.approx(0.60)
    observation = json.loads(
        (perception_dir / "observation.json").read_text(encoding="utf-8")
    )
    assert observation["grasp_reference_policy"] == (
        "molmo_confidence_filtered_keypoints_only"
    )
    assert observation["valid_grasp_reference_count"] == 2


def test_no_point_above_threshold_produces_explicit_stop_status(tmp_path: Path) -> None:
    perception_dir, image_paths = _perception_dir(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    payload = _payload(
        [
            _record("center", status="point_returned", confidence=0.4, pixel=[2.0, 1.0]),
            _record("corner", status="not_found", confidence=0.0, pixel=None),
        ],
        [
            _record("center", status="not_found", confidence=0.0, pixel=None),
            _record("corner", status="point_returned", confidence=0.6, pixel=[4.0, 3.0]),
        ],
    )

    manifest = build_confidence_filtered_references(
        payload,
        perception_dir=perception_dir,
        artifact_dir=artifact_dir,
        image_paths=image_paths,
        cameras=("A", "B"),
        specs=SPECS,
        confidence_threshold=0.6,
        install=False,
    )

    assert manifest["status"] == "NO_VALID_GRASP_REFERENCES"
    assert manifest["accepted_reference_count"] == 0
    assert manifest["safety_gate"].startswith("planning must not start")


def test_restricting_queries_to_one_camera_disables_other_camera_references(
    tmp_path: Path,
) -> None:
    perception_dir, image_paths = _perception_dir(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    payload = {
        "views": [
            {
                "label": "A",
                "image_size": [5, 4],
                "records": [
                    _record(
                        "center",
                        status="point_returned",
                        confidence=0.9,
                        pixel=[2.0, 1.0],
                    ),
                    _record("corner", status="not_found", confidence=0.0, pixel=None),
                ],
            }
        ]
    }

    manifest = build_confidence_filtered_references(
        payload,
        perception_dir=perception_dir,
        artifact_dir=artifact_dir,
        image_paths={"A": image_paths["A"]},
        cameras=("A",),
        specs=SPECS,
        confidence_threshold=0.6,
        install=True,
    )

    assert manifest["queried_cameras"] == ["A"]
    assert manifest["disabled_unqueried_cameras"] == ["B"]
    camera_b = json.loads(
        (perception_dir / "camera_B_coordinate_guide.json").read_text(
            encoding="utf-8"
        )
    )
    assert camera_b["samples"] == []
    assert "not queried" in camera_b["reference_semantics"]


def test_worker_payload_requires_bounded_numeric_confidence(tmp_path: Path) -> None:
    perception_dir, image_paths = _perception_dir(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    payload = _payload(
        [
            _record("center", status="point_returned", confidence=0.8, pixel=[2.0, 1.0]),
            _record("corner", status="point_returned", confidence=0.7, pixel=[3.0, 2.0]),
        ],
        [
            _record("center", status="not_found", confidence=0.0, pixel=None),
            _record("corner", status="not_found", confidence=0.0, pixel=None),
        ],
    )
    payload["views"][0]["records"][0]["confidence"] = float("nan")  # type: ignore[index]

    with pytest.raises(MolmoKeypointPipelineError, match="finite"):
        build_confidence_filtered_references(
            payload,
            perception_dir=perception_dir,
            artifact_dir=artifact_dir,
            image_paths=image_paths,
            cameras=("A", "B"),
            specs=SPECS,
            confidence_threshold=0.6,
        )


def test_runner_invokes_worker_then_builds_installed_task_references(
    tmp_path: Path,
) -> None:
    perception_dir, _ = _perception_dir(tmp_path)
    artifact_dir = tmp_path / "pipeline_output"
    worker_payload = _payload(
        [
            _record("center", status="point_returned", confidence=0.9, pixel=[2.0, 1.0]),
            _record("corner", status="not_found", confidence=0.0, pixel=None),
        ],
        [
            _record("center", status="not_found", confidence=0.0, pixel=None),
            _record("corner", status="point_returned", confidence=0.8, pixel=[4.0, 3.0]),
        ],
    )
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["kwargs"] = kwargs
        output_path = Path(command[command.index("--output") + 1])
        output_path.write_text(json.dumps(worker_payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    manifest = run_molmo_keypoint_pipeline(
        project_root=Path.cwd(),
        perception_dir=perception_dir,
        artifact_dir=artifact_dir,
        confidence_threshold=0.6,
        molmo_python=Path("/bin/true"),
        keypoint_specs=SPECS,
        subprocess_run=fake_run,
    )

    assert manifest["status"] == "READY"
    assert manifest["accepted_reference_count"] == 2
    command = captured["command"]
    assert isinstance(command, list)
    assert command.count("--image") == 2
    assert command.count("--label") == 2
    assert captured["kwargs"]["shell"] is False  # type: ignore[index]
    assert (artifact_dir / "molmo_keypoint_grasp_references.json").is_file()


def test_geometric_mean_probability_is_stable_and_strictly_validated() -> None:
    assert geometric_mean_probability([0.8, 0.8, 0.8]) == pytest.approx(0.8)
    assert geometric_mean_probability([0.5, 0.25, 1.0]) == pytest.approx(0.5)
    assert geometric_mean_probability([]) == 0.0
    with pytest.raises(ValueError):
        geometric_mean_probability([1.1])


def test_point_location_confidence_excludes_no_more_points_token() -> None:
    dynamic = [0.9, 0.5, 0.7, 0.999]
    assert point_location_probabilities(dynamic, 1) == [0.9, 0.5, 0.7]
    assert point_location_probabilities(dynamic, 0) == []
    assert point_location_probabilities(dynamic, 2) == []
