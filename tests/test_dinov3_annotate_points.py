from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cloth_agent import perception
from scripts import dinov3_annotate_points as annotation


def test_live_capture_saves_rgbd_xyz_and_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = perception.PerceptionConfig(
        cameras=(
            perception.CameraSpec("A", "serial-A", tmp_path / "A.yaml"),
            perception.CameraSpec("B", "serial-B", tmp_path / "B.yaml"),
        )
    )
    monkeypatch.setattr(
        perception.PerceptionConfig,
        "load",
        classmethod(lambda _cls, _root, _path: config),
    )
    rgb = np.full((4, 5, 3), 80, dtype=np.uint8)
    depth = np.full((4, 5), 1.0, dtype=np.float32)
    frame = perception.RGBDFrame(
        label="A",
        serial="serial-A",
        rgb=rgb,
        depth_m=depth,
        intrinsics=np.eye(3, dtype=np.float64),
        X_base_camera=np.eye(4, dtype=np.float64),
    )
    captured_labels: list[tuple[str, ...]] = []

    def fake_capture(capture_config):
        captured_labels.append(capture_config.active_camera_labels)
        return [frame]

    monkeypatch.setattr(perception, "capture_two_view_rgbd", fake_capture)
    xyz = np.zeros((4, 5, 3), dtype=np.float32)
    valid = np.ones((4, 5), dtype=bool)
    monkeypatch.setattr(
        perception,
        "camera_base_xyz_map_mm",
        lambda _frame, _config: (xyz, valid),
    )

    capture_path = tmp_path / "capture.png"
    metadata = annotation.capture_reference_rgbd(
        tmp_path,
        Path("config.json"),
        "a",
        capture_path,
    )
    assert captured_labels == [("A",)]
    assert capture_path.is_file()
    assert Path(metadata["depth_m"]).is_file()
    assert Path(metadata["base_xyz_mm"]).is_file()
    assert Path(metadata["xyz_valid"]).is_file()
    saved = json.loads(Path(metadata["metadata_json"]).read_text(encoding="utf-8"))
    assert saved["camera_label"] == "A"
    assert saved["valid_xyz_pixels"] == 20


def test_live_mode_uses_a_fresh_timestamped_capture_each_launch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    stamps = iter(["FIRST", "SECOND"])
    monkeypatch.setattr(annotation, "_capture_stamp", lambda: next(stamps))
    captured_paths: list[Path] = []
    annotate_calls: list[tuple[Path, dict, str]] = []

    def fake_capture(_root, _config, label, path):
        captured_paths.append(path)
        return {"camera_label": label, "rgb": str(path)}

    def fake_annotate(
        image_path,
        _output_path,
        _count,
        _labels,
        *,
        capture_metadata=None,
        surface_side="FRONT",
        side_marker=None,
        guide_points_path=None,
        guide_neighbors_path=None,
        guide_opacity=0.58,
    ):
        annotate_calls.append(
            (
                image_path,
                {**dict(capture_metadata or {}), "side_marker_arg": side_marker},
                surface_side,
            )
        )

    monkeypatch.setattr(annotation, "capture_reference_rgbd", fake_capture)
    monkeypatch.setattr(annotation, "annotate", fake_annotate)
    output = tmp_path / "reference_points.json"
    argv = ["--output", str(output), "--num-points", "20"]
    assert annotation.main(argv) == 0
    assert annotation.main(argv) == 0

    assert captured_paths[0].name == "reference_points_front_camera_A_FIRST.png"
    assert captured_paths[1].name == "reference_points_front_camera_A_SECOND.png"
    assert captured_paths[0] != captured_paths[1]
    assert [call[0] for call in annotate_calls] == captured_paths
    assert [call[2] for call in annotate_calls] == ["FRONT", "FRONT"]


def test_explicit_image_keeps_offline_relabel_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "existing.png"
    image_path.write_bytes(b"placeholder")
    calls: list[tuple[Path, object]] = []
    monkeypatch.setattr(
        annotation,
        "capture_reference_rgbd",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected capture")),
    )
    monkeypatch.setattr(
        annotation,
        "annotate",
        lambda image, _output, _count, _labels, *, capture_metadata=None, surface_side="FRONT", side_marker=None, **_kwargs: calls.append(
            (image, capture_metadata, surface_side)
        ),
    )
    assert (
        annotation.main(
            [
                "--image",
                str(image_path),
                "--output",
                str(tmp_path / "points.json"),
                "--num-points",
                "2",
            ]
        )
        == 0
    )
    assert calls == [(image_path.resolve(), None, "FRONT")]


def test_back_annotation_writes_surface_and_b_prefixed_area_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "back.png"
    Image.fromarray(np.full((80, 100, 3), 60, dtype=np.uint8)).save(image_path)
    xyz_path = tmp_path / "back_base_xyz_mm.npy"
    xyz = np.zeros((80, 100, 3), dtype=np.float32)
    xyz[..., 0] = 100.0
    xyz[..., 1] = 200.0
    xyz[..., 2] = 5.0
    np.save(xyz_path, xyz)
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

    def click_two_then_quit(_delay: int) -> int:
        callback = callback_holder["callback"]
        assert callable(callback)
        callback(cv2.EVENT_LBUTTONDOWN, 20, 30, 0, None)
        callback(cv2.EVENT_LBUTTONDOWN, 70, 50, 0, None)
        return ord("q")

    monkeypatch.setattr(cv2, "waitKey", click_two_then_quit)
    output_path = tmp_path / "back_points.json"
    annotation.annotate(
        image_path,
        output_path,
        2,
        ["p01", "p02"],
        capture_metadata={"base_xyz_mm": str(xyz_path)},
        surface_side="BACK",
        side_marker="COLLAR_LABEL_NOT_VISIBLE",
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["surface_side"] == "BACK"
    assert payload["side_marker"] == "COLLAR_LABEL_NOT_VISIBLE"
    assert [point["area_id"] for point in payload["points"]] == ["B01", "B02"]
    assert all(point["surface_side"] == "BACK" for point in payload["points"])
    assert all(point["base_xyz_mm"] == [100.0, 200.0, 5.0] for point in payload["points"])


def test_guide_graph_scales_points_and_deduplicates_edges(tmp_path: Path) -> None:
    points_path = tmp_path / "front_points.json"
    points_path.write_text(
        json.dumps(
            {
                "image_width": 100,
                "image_height": 50,
                "points": [
                    {"area_id": "F01", "x": 10, "y": 20},
                    {"area_id": "F02", "x": 80, "y": 30},
                ],
            }
        ),
        encoding="utf-8",
    )
    neighbors_path = tmp_path / "neighbors.json"
    neighbors_path.write_text(
        json.dumps({"neighbors": {"F01": ["F02"], "F02": ["F01"]}}),
        encoding="utf-8",
    )
    points, edges = annotation._load_guide_graph(
        points_path,
        neighbors_path,
        image_size=(200, 100),
    )
    assert [(point["x"], point["y"]) for point in points] == [(20.0, 40.0), (160.0, 60.0)]
    assert edges == [("F01", "F02")]
