from __future__ import annotations

from pathlib import Path

from scripts.capture_camera_photo import _shot_directory, build_parser


def test_shot_directory_is_unique_and_timestamped(tmp_path: Path) -> None:
    first = _shot_directory(tmp_path, label="C", index=1, count=1)
    second = _shot_directory(tmp_path, label="C", index=1, count=1)
    assert first != second
    assert first.is_dir()
    assert second.is_dir()
    assert first.name.startswith("camera_C_")


def test_parser_defaults_to_camera_a() -> None:
    args = build_parser().parse_args([])
    assert args.serial == "317222073552"
    assert args.label == "A"
    assert args.count == 1

