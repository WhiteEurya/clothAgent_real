#!/usr/bin/env python3
"""Burn time-aligned robot-process phases into an existing cumulative rollout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.rollout_recorder import (  # noqa: E402
    build_rollout_phase_timeline,
    overlay_rollout_labels_mp4,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _find_output_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if (root / "combined_rollout.mp4").is_file():
        return root
    candidates = sorted(root.glob("results/molmo_keypoint_cli/*/combined_rollout.mp4"))
    if not candidates:
        raise FileNotFoundError(f"no combined_rollout.mp4 under {root}")
    return candidates[-1].parent


def _probe_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"cannot probe {path}")
    duration = float(completed.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid video duration: {duration}")
    return duration


def annotate(root: Path, output: Path | None, *, in_place: bool) -> Path:
    output_root = _find_output_root(root)
    source = output_root / "combined_rollout.mp4"
    summary = _load_json(output_root / "summary.json")
    speed = float(summary.get("combined_video_speed", 1.0))
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError(f"invalid combined video speed: {speed}")

    segments: list[dict[str, Any]] = []
    total_source_duration = 0.0
    for recording_path in sorted(output_root.glob("iteration_*/rollout_recording.json")):
        recording = _load_json(recording_path)
        if recording.get("status") != "completed":
            continue
        manifest = recording.get("manifest")
        if not isinstance(manifest, dict):
            continue
        frames = manifest.get("composite_encoded_frame_count")
        fps = manifest.get("fps")
        if not isinstance(frames, (int, float)) or not isinstance(fps, (int, float)):
            continue
        if float(frames) <= 0 or float(fps) <= 0:
            continue
        iteration = int(recording_path.parent.name.split("_")[-1])
        execution_path = recording_path.parent / "execution.json"
        execution = _load_json(execution_path) if execution_path.is_file() else None
        source_duration = float(frames) / float(fps)
        segments.append(
            {
                "iteration": iteration,
                "source_duration_s": source_duration,
                "phases": build_rollout_phase_timeline(execution, manifest),
            }
        )
        total_source_duration += source_duration
    if not segments or total_source_duration <= 0:
        raise ValueError("no completed rollout segments with timing metadata were found")

    actual_duration = _probe_duration(source)
    theoretical_duration = total_source_duration / speed
    duration_scale = actual_duration / theoretical_duration
    cursor = 0.0
    combined_timeline: list[dict[str, Any]] = []
    for segment in segments:
        for phase in segment["phases"]:
            combined_timeline.append(
                {
                    "start_s": cursor + float(phase["start_s"]) / speed * duration_scale,
                    "end_s": cursor + float(phase["end_s"]) / speed * duration_scale,
                    "label": str(phase["label"]),
                }
            )
        cursor += float(segment["source_duration_s"]) / speed * duration_scale

    destination = (
        output.expanduser().resolve()
        if output is not None
        else output_root / "combined_rollout_with_process.mp4"
    )
    if in_place:
        temporary = source.with_name(f".{source.stem}.process.tmp.mp4")
        temporary.unlink(missing_ok=True)
        result = overlay_rollout_labels_mp4(
            source,
            temporary,
            phase_timeline=combined_timeline,
        )
        temporary.replace(source)
        destination = source
    else:
        result = overlay_rollout_labels_mp4(
            source,
            destination,
            phase_timeline=combined_timeline,
        )
    manifest_path = output_root / "combined_rollout_process_labels.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "output": str(destination),
                "in_place": in_place,
                "playback_speed": speed,
                "duration_scale": duration_scale,
                "phase_count": len(combined_timeline),
                "segments": segments,
                "combined_timeline": combined_timeline,
                "overlay_result": result,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="atomically replace the run's combined_rollout.mp4",
    )
    args = parser.parse_args()
    if args.in_place and args.output is not None:
        parser.error("--output cannot be combined with --in-place")
    result = annotate(args.run, args.output, in_place=args.in_place)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
