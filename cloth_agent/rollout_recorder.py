"""Standalone dual-RealSense recorder for physical garment rollouts.

This module deliberately has no robot imports or execution authority.  It owns
Camera A/B only while recording and writes RGB video, depth-visualization
video, a four-panel composite, optional native RealSense ``.db3`` files, and a
per-frame timestamp table.  It must not run while another process owns either
configured RealSense device.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .perception import CameraSpec, PerceptionConfig


class RolloutRecorderError(RuntimeError):
    """Raised when standalone rollout recording cannot continue safely."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RolloutRecorderError(
            "OpenCV is required for MP4 output; use the configured cali environment"
        ) from exc
    return cv2


def depth_to_bgr(
    depth_m: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """Convert metric depth to a fixed-scale near-red/far-blue BGR image."""

    depth = np.asarray(depth_m, dtype=np.float32)
    valid = (
        np.isfinite(depth)
        & (depth > float(min_depth_m))
        & (depth < float(max_depth_m))
    )
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = np.clip(
        (max_depth_m - depth[valid]) / (max_depth_m - min_depth_m),
        0.0,
        1.0,
    )
    red = normalized
    green = 1.0 - np.abs(2.0 * normalized - 1.0)
    blue = 1.0 - normalized
    rgb = np.rint(np.stack([red, green, blue], axis=2) * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb[:, :, ::-1].copy()


def compose_four_panel(
    camera_a_bgr: np.ndarray,
    camera_b_bgr: np.ndarray,
    camera_a_depth_bgr: np.ndarray,
    camera_b_depth_bgr: np.ndarray,
) -> np.ndarray:
    """Return A/B RGB over A/B depth as a 2x2 frame."""

    frames = [
        np.asarray(camera_a_bgr, dtype=np.uint8),
        np.asarray(camera_b_bgr, dtype=np.uint8),
        np.asarray(camera_a_depth_bgr, dtype=np.uint8),
        np.asarray(camera_b_depth_bgr, dtype=np.uint8),
    ]
    shape = frames[0].shape
    if len(shape) != 3 or shape[2] != 3 or any(frame.shape != shape for frame in frames):
        raise ValueError("all four panel frames must have the same HxWx3 shape")
    return np.concatenate(
        [np.concatenate(frames[:2], axis=1), np.concatenate(frames[2:], axis=1)],
        axis=0,
    )


def _label_frame(frame: np.ndarray, label: str, elapsed_s: float) -> np.ndarray:
    cv2 = _require_cv2()
    result = np.asarray(frame, dtype=np.uint8).copy()
    text = f"{label}  t={elapsed_s:8.3f}s"
    cv2.putText(result, text, (13, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(result, text, (13, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def _open_writer(path: Path, width: int, height: int, fps: int, codec: str):
    cv2 = _require_cv2()
    if len(codec) != 4:
        raise RolloutRecorderError("video codec must be a four-character code")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), float(fps), (int(width), int(height))
    )
    if not writer.isOpened():
        writer.release()
        raise RolloutRecorderError(
            f"failed to open video writer {path} with codec {codec!r}"
        )
    return writer


def finalize_mp4_h264(
    path: Path,
    *,
    ffmpeg_binary: str = "ffmpeg",
    preset: str = "veryfast",
    crf: int = 20,
) -> dict[str, Any]:
    """Replace one OpenCV MP4 with a widely compatible H.264 MP4.

    OpenCV's available encoder on this machine writes MPEG-4 Part 2 (``mp4v``),
    which is valid but unsupported by several browsers and default media players.
    FFmpeg performs the compatibility encode only after the writer is closed.  The
    original file is replaced only after FFmpeg succeeds and emits a non-empty file.
    """

    source = Path(path).resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise RolloutRecorderError(f"video is missing or empty: {source}")
    binary = (
        shutil.which(ffmpeg_binary)
        if Path(ffmpeg_binary).name == ffmpeg_binary
        else ffmpeg_binary
    )
    if binary is None:
        raise RolloutRecorderError(
            f"FFmpeg executable not found: {ffmpeg_binary}; cannot finalize H.264 MP4"
        )
    temporary = source.with_name(f".{source.stem}.h264.tmp.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        str(binary),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            shell=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RolloutRecorderError(
                f"FFmpeg H.264 finalization failed for {source.name}: {detail}"
            )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RolloutRecorderError(
                f"FFmpeg produced no usable H.264 output for {source.name}"
            )
        source_size = source.stat().st_size
        output_size = temporary.stat().st_size
        temporary.replace(source)
        return {
            "status": "completed",
            "codec": "h264",
            "pixel_format": "yuv420p",
            "faststart": True,
            "source_size_bytes": source_size,
            "output_size_bytes": output_size,
        }
    finally:
        temporary.unlink(missing_ok=True)


def append_mp4_to_cumulative(
    source: Path,
    cumulative: Path,
    *,
    ffmpeg_binary: str = "ffmpeg",
) -> dict[str, Any]:
    """Append one finalized MP4 segment to a run-level cumulative MP4.

    The source is never modified.  The destination is replaced atomically only
    after FFmpeg has produced a valid combined file.  Segments are expected to
    come from the same recorder configuration (resolution, frame rate, and
    codec), which permits a fast stream-copy concat; a re-encode fallback keeps
    the append operation usable when codec metadata differs.
    """

    source_path = Path(source).expanduser().resolve()
    cumulative_path = Path(cumulative).expanduser().resolve()
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise RolloutRecorderError(f"video segment is missing or empty: {source_path}")
    binary = (
        shutil.which(ffmpeg_binary)
        if Path(ffmpeg_binary).name == ffmpeg_binary
        else ffmpeg_binary
    )
    if binary is None:
        raise RolloutRecorderError(
            f"FFmpeg executable not found: {ffmpeg_binary}; cannot append MP4"
        )
    cumulative_path.parent.mkdir(parents=True, exist_ok=True)
    if not cumulative_path.exists():
        shutil.copy2(source_path, cumulative_path)
        return {
            "status": "completed",
            "mode": "initial_segment",
            "source": str(source_path),
            "cumulative": str(cumulative_path),
            "segments_added": 1,
        }

    temporary = cumulative_path.with_name(
        f".{cumulative_path.stem}.append.{os.getpid()}.tmp.mp4"
    )
    concat_list = cumulative_path.with_name(
        f".{cumulative_path.stem}.append.{os.getpid()}.txt"
    )

    def _concat_entry(path: Path) -> str:
        # FFmpeg concat files use single-quoted paths; escape backslashes and
        # apostrophes so run directories with unusual names remain valid.
        escaped = str(path).replace("\\", "\\\\").replace("'", "'\\''")
        return f"file '{escaped}'\n"

    concat_list.write_text(
        _concat_entry(cumulative_path) + _concat_entry(source_path),
        encoding="utf-8",
    )
    temporary.unlink(missing_ok=True)
    stream_copy_command = [
        str(binary),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-map",
        "0:v:0",
        "-an",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    completed = subprocess.run(
        stream_copy_command,
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    mode = "stream_copy"
    if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size <= 0:
        temporary.unlink(missing_ok=True)
        reencode_command = [
            str(binary),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        completed = subprocess.run(
            reencode_command,
            text=True,
            capture_output=True,
            check=False,
            shell=False,
        )
        mode = "reencode"
    try:
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RolloutRecorderError(
                f"FFmpeg cumulative append failed: {detail or 'unknown error'}"
            )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RolloutRecorderError("FFmpeg produced no usable cumulative MP4")
        temporary.replace(cumulative_path)
        return {
            "status": "completed",
            "mode": mode,
            "source": str(source_path),
            "cumulative": str(cumulative_path),
            "segments_added": 1,
        }
    finally:
        temporary.unlink(missing_ok=True)
        concat_list.unlink(missing_ok=True)


def speed_up_mp4(
    source: Path,
    output: Path,
    *,
    speed: float = 4.0,
    ffmpeg_binary: str = "ffmpeg",
) -> dict[str, Any]:
    """Encode a video at a faster playback rate without dropping source frames."""

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise RolloutRecorderError(f"video is missing or empty: {source_path}")
    if source_path == output_path:
        raise RolloutRecorderError("speed-up output must differ from the source")
    if not np.isfinite(float(speed)) or float(speed) <= 0:
        raise RolloutRecorderError("video playback speed must be finite and positive")
    binary = (
        shutil.which(ffmpeg_binary)
        if Path(ffmpeg_binary).name == ffmpeg_binary
        else ffmpeg_binary
    )
    if binary is None:
        raise RolloutRecorderError(
            f"FFmpeg executable not found: {ffmpeg_binary}; cannot speed up MP4"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.speed.tmp.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        str(binary),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"setpts=PTS/{float(speed):.8g}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            shell=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RolloutRecorderError(
                f"FFmpeg speed-up failed for {source_path.name}: {detail or 'unknown error'}"
            )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RolloutRecorderError("FFmpeg produced no usable speed-up MP4")
        temporary.replace(output_path)
        return {
            "status": "completed",
            "speed": float(speed),
            "source": str(source_path),
            "output": str(output_path),
            "output_size_bytes": output_path.stat().st_size,
        }
    finally:
        temporary.unlink(missing_ok=True)


def _parse_recorded_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _recorded_video_duration_s(manifest: Mapping[str, Any]) -> float | None:
    frames = manifest.get("composite_encoded_frame_count")
    fps = manifest.get("fps")
    if isinstance(frames, (int, float)) and isinstance(fps, (int, float)):
        if float(frames) > 0 and float(fps) > 0:
            return float(frames) / float(fps)
    started = _parse_recorded_time(manifest.get("created_at"))
    ended = _parse_recorded_time(manifest.get("ended_at"))
    if started is not None and ended is not None and ended > started:
        return (ended - started).total_seconds()
    duration = manifest.get("duration_s")
    if isinstance(duration, (int, float)) and np.isfinite(float(duration)):
        if float(duration) > 0:
            return float(duration)
    return None


def _rollout_action_phase(
    actions: Sequence[Mapping[str, Any]],
    index: int,
    *,
    checkpoint_index: int | None,
    abort_branch: bool,
) -> str:
    action = actions[index]
    name = str(action.get("name", "action")).strip().lower()
    names = [str(item.get("name", "")).strip().lower() for item in actions]
    close_indices = [position for position, value in enumerate(names) if value == "close_gripper"]
    close_index = close_indices[0] if close_indices else None
    release_index = next(
        (
            position
            for position, value in enumerate(names)
            if value == "open_gripper"
            and close_index is not None
            and position > close_index
        ),
        None,
    )
    prefix = f"STEP {index + 1:02d}/{len(actions):02d} | "
    if name == "home":
        return prefix + "RETURN HOME"
    if name == "close_gripper":
        return prefix + "CLOSE GRIPPER | GRASP"
    if name == "open_gripper":
        if close_index is None or index < close_index:
            return prefix + "OPEN GRIPPER"
        if abort_branch and checkpoint_index is not None and index > checkpoint_index:
            return prefix + "ABORT | RELEASE AT ORIGIN"
        return prefix + "RELEASE GARMENT"
    if name != "move":
        return prefix + name.replace("_", " ").upper()
    if checkpoint_index is not None and index == checkpoint_index:
        return prefix + "LIFT TO HOLD CHECK"
    if (
        abort_branch
        and checkpoint_index is not None
        and index > checkpoint_index
        and (release_index is None or index < release_index)
    ):
        return prefix + "ABORT | DESCEND TO ORIGIN"
    if close_index is None or index < close_index:
        later_moves_before_close = any(
            names[position] == "move"
            for position in range(index + 1, close_index or len(actions))
        )
        return prefix + (
            "APPROACH TARGET" if later_moves_before_close else "DESCEND TO GRASP"
        )
    if release_index is not None and index < release_index:
        move_indices = [
            position
            for position in range(close_index + 1, release_index)
            if names[position] == "move"
        ]
        if move_indices and index == move_indices[0]:
            return prefix + "LIFT GARMENT"
        if move_indices and index == move_indices[-1]:
            return prefix + "DESCEND FOR LAYDOWN"
        return prefix + "TRANSPORT GARMENT"
    return prefix + "RETRACT GRIPPER"


def build_rollout_phase_timeline(
    execution: Mapping[str, Any] | None,
    recording_manifest: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Align robot-action phases to the unaccelerated rollout video clock."""

    if not isinstance(execution, Mapping) or not isinstance(recording_manifest, Mapping):
        return []
    raw_actions = execution.get("actual_robot_actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raw_actions = execution.get("requested_robot_actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        return []
    actions = [item for item in raw_actions if isinstance(item, Mapping)]
    if not actions:
        return []
    recording_start = _parse_recorded_time(recording_manifest.get("created_at"))
    if recording_start is None:
        recording_start = _parse_recorded_time(actions[0].get("requested_at"))
    duration_s = _recorded_video_duration_s(recording_manifest)
    if recording_start is None or duration_s is None:
        return []
    checkpoint_value = execution.get("checkpoint_action_index")
    checkpoint_index = (
        int(checkpoint_value)
        if isinstance(checkpoint_value, int)
        and not isinstance(checkpoint_value, bool)
        and 0 <= checkpoint_value < len(actions)
        else None
    )
    checkpoint = execution.get("checkpoint")
    abort_branch = bool(
        isinstance(checkpoint, Mapping)
        and str(checkpoint.get("executed_branch", checkpoint.get("runtime_decision", ""))).upper()
        == "ABORT_RELEASE"
    )
    parsed_actions: list[tuple[Mapping[str, Any], datetime, datetime]] = []
    for action in actions:
        requested = _parse_recorded_time(action.get("requested_at"))
        completed = _parse_recorded_time(action.get("completed_at")) or requested
        if requested is None or completed is None:
            continue
        parsed_actions.append((action, requested, max(requested, completed)))
    if not parsed_actions:
        return []
    # Preserve action indices after filtering malformed timestamp records.
    aligned_actions = [item[0] for item in parsed_actions]
    if len(aligned_actions) != len(actions):
        actions = aligned_actions
        checkpoint_index = None

    def offset(timestamp: datetime) -> float:
        return min(duration_s, max(0.0, (timestamp - recording_start).total_seconds()))

    timeline: list[dict[str, Any]] = []

    def append_interval(start_s: float, end_s: float, label: str) -> None:
        start = min(duration_s, max(0.0, float(start_s)))
        end = min(duration_s, max(start, float(end_s)))
        if end - start < 1e-3:
            return
        if timeline and timeline[-1]["label"] == label and abs(timeline[-1]["end_s"] - start) < 1e-3:
            timeline[-1]["end_s"] = end
            return
        timeline.append({"start_s": start, "end_s": end, "label": label})

    first_start = offset(parsed_actions[0][1])
    append_interval(0.0, first_start, "WAITING FOR ROBOT EXECUTION")
    for index, (_, requested, completed) in enumerate(parsed_actions):
        start_s = offset(requested)
        completed_s = offset(completed)
        next_start_s = (
            offset(parsed_actions[index + 1][1])
            if index + 1 < len(parsed_actions)
            else completed_s
        )
        label = _rollout_action_phase(
            actions,
            index,
            checkpoint_index=checkpoint_index,
            abort_branch=abort_branch,
        )
        append_interval(start_s, completed_s, label)
        settled_label = (
            "HOLD CHECK | CLAUDE EVALUATION"
            if checkpoint_index is not None and index == checkpoint_index
            else label
        )
        append_interval(completed_s, next_start_s, settled_label)
    last_end = offset(parsed_actions[-1][2])
    append_interval(last_end, duration_s, "ROLLOUT COMPLETE")
    return timeline


def _escape_drawtext(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace("%", "\\%")
    )


def overlay_rollout_labels_mp4(
    source: Path,
    output: Path,
    *,
    iteration: int | None = None,
    phase_timeline: Sequence[Mapping[str, Any]] = (),
    ffmpeg_binary: str = "ffmpeg",
) -> dict[str, Any]:
    """Burn iteration and time-aligned process phases into a rollout video.

    Labels are applied before playback acceleration in the normal recording
    path, so every phase remains attached to the correct physical action.
    """

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise RolloutRecorderError(f"video is missing or empty: {source_path}")
    if source_path == output_path:
        raise RolloutRecorderError("rollout-label output must differ from the source")
    if iteration is not None and (isinstance(iteration, bool) or int(iteration) < 1):
        raise RolloutRecorderError("iteration number must be a positive integer")
    normalized_timeline: list[dict[str, Any]] = []
    for item in phase_timeline:
        try:
            start_s = float(item["start_s"])
            end_s = float(item["end_s"])
            phase_label = str(item["label"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if not phase_label or not np.isfinite(start_s) or not np.isfinite(end_s):
            continue
        if end_s <= start_s:
            continue
        normalized_timeline.append(
            {"start_s": max(0.0, start_s), "end_s": end_s, "label": phase_label}
        )
    if iteration is None and not normalized_timeline:
        raise RolloutRecorderError("at least one rollout label is required")
    binary = (
        shutil.which(ffmpeg_binary)
        if Path(ffmpeg_binary).name == ffmpeg_binary
        else ffmpeg_binary
    )
    if binary is None:
        raise RolloutRecorderError(
            f"FFmpeg executable not found: {ffmpeg_binary}; cannot label MP4"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.label.tmp.mp4")
    temporary.unlink(missing_ok=True)
    label = f"ITER {int(iteration):03d}" if iteration is not None else None
    filters: list[str] = []
    if label is not None:
        # Keep the iteration box below the recorder's camera caption at y=29.
        filters.extend(
            [
                "drawbox=x=16:y=56:w=220:h=48:color=black@0.72:t=fill",
                f"drawtext=fontcolor=white:fontsize=30:expansion=none:text='{label}':x=29:y=64",
            ]
        )
    for item in normalized_timeline:
        start_s = float(item["start_s"])
        end_s = float(item["end_s"])
        enabled = f"between(t\\,{start_s:.6f}\\,{end_s:.6f})"
        phase_label = _escape_drawtext(str(item["label"]))
        filters.extend(
            [
                "drawbox=x=16:y=112:w=1180:h=50:color=black@0.72:t=fill:"
                f"enable='{enabled}'",
                "drawtext=fontcolor=white:fontsize=28:expansion=none:"
                f"text='{phase_label}':x=29:y=121:enable='{enabled}'",
            ]
        )
    filter_graph = ",".join(filters)
    command = [
        str(binary),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        filter_graph,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            shell=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RolloutRecorderError(
                f"FFmpeg rollout-label failed for {source_path.name}: "
                f"{detail or 'unknown error'}"
            )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RolloutRecorderError("FFmpeg produced no usable labelled MP4")
        temporary.replace(output_path)
        return {
            "status": "completed",
            "iteration": int(iteration) if iteration is not None else None,
            "label": label,
            "phase_timeline": normalized_timeline,
            "source": str(source_path),
            "output": str(output_path),
            "output_size_bytes": output_path.stat().st_size,
        }
    finally:
        temporary.unlink(missing_ok=True)


def label_iteration_mp4(
    source: Path,
    output: Path,
    *,
    iteration: int,
    phase_timeline: Sequence[Mapping[str, Any]] = (),
    ffmpeg_binary: str = "ffmpeg",
) -> dict[str, Any]:
    """Burn the iteration and current robot-process phase into one segment."""

    return overlay_rollout_labels_mp4(
        source,
        output,
        iteration=iteration,
        phase_timeline=phase_timeline,
        ffmpeg_binary=ffmpeg_binary,
    )


def prune_rollout_video_files(recording_dir: Path) -> list[str]:
    """Delete per-rollout video/native-recording files after evaluation.

    Manifests, timestamp CSVs, and evaluator contact sheets remain available as
    lightweight evidence; the run-level cumulative MP4 is the retained video.
    """

    root = Path(recording_dir).expanduser().resolve()
    if not root.is_dir():
        return []
    removed: list[str] = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".mp4", ".db3", ".bag"}:
            continue
        try:
            path.unlink()
            removed.append(str(path))
        except OSError:
            continue
    return removed


def _configure_color_exposure(device: Any, spec: CameraSpec, rs: Any) -> None:
    if spec.color_exposure is None and spec.color_white_balance is None:
        return
    color_sensor = next(
        (
            sensor
            for sensor in device.query_sensors()
            if sensor.get_info(rs.camera_info.name) == "RGB Camera"
        ),
        None,
    )
    if color_sensor is None:
        raise RolloutRecorderError(f"camera {spec.label} has no RGB sensor")
    if spec.color_exposure is not None:
        if not color_sensor.supports(rs.option.enable_auto_exposure) or not color_sensor.supports(rs.option.exposure):
            raise RolloutRecorderError(
                f"camera {spec.label} has no configurable RGB exposure sensor"
            )
        exposure_range = color_sensor.get_option_range(rs.option.exposure)
        exposure = float(spec.color_exposure)
        if not exposure_range.min <= exposure <= exposure_range.max:
            raise RolloutRecorderError(
                f"camera {spec.label} exposure {exposure} is outside "
                f"[{exposure_range.min}, {exposure_range.max}]"
            )
        color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
        color_sensor.set_option(rs.option.exposure, exposure)
    if spec.color_white_balance is not None:
        if not color_sensor.supports(rs.option.enable_auto_white_balance) or not color_sensor.supports(rs.option.white_balance):
            raise RolloutRecorderError(
                f"camera {spec.label} has no configurable RGB white-balance sensor"
            )
        white_balance_range = color_sensor.get_option_range(rs.option.white_balance)
        white_balance = float(spec.color_white_balance)
        if not white_balance_range.min <= white_balance <= white_balance_range.max:
            raise RolloutRecorderError(
                f"camera {spec.label} white balance {white_balance} is outside "
                f"[{white_balance_range.min}, {white_balance_range.max}]"
            )
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
        color_sensor.set_option(rs.option.white_balance, white_balance)


@dataclass
class _CameraState:
    spec: CameraSpec
    pipeline: Any
    align: Any
    depth_scale: float
    recorder: Any | None
    rgb_writer: Any
    depth_writer: Any | None
    frame_count: int = 0
    encoded_frame_count: int = 0
    last_frame_number: int | None = None


@dataclass(frozen=True)
class _CapturedFrame:
    label: str
    serial: str
    rgb: np.ndarray
    depth_m: np.ndarray
    rgb_bgr: np.ndarray
    depth_bgr: np.ndarray
    host_utc: str
    host_monotonic_ns: int
    color_frame_number: int
    color_device_timestamp_ms: float
    depth_frame_number: int
    depth_device_timestamp_ms: float
    valid_depth_fraction: float


@dataclass(frozen=True)
class RolloutRGBDFrame:
    """One unlabelled aligned frame copied from the active recorder thread."""

    label: str
    serial: str
    rgb: np.ndarray
    depth_m: np.ndarray
    host_utc: str
    host_monotonic_ns: int
    color_frame_number: int
    depth_frame_number: int
    valid_depth_fraction: float


@dataclass(frozen=True)
class ObserverRGBFrame:
    """One RGB frame copied from the uncalibrated Camera-C observer."""

    label: str
    serial: str
    rgb: np.ndarray
    host_utc: str
    host_monotonic_ns: int
    color_frame_number: int
    color_device_timestamp_ms: float


class DualRealSenseRolloutRecorder:
    """Own both configured RealSense devices and record one standalone rollout."""

    def __init__(
        self,
        perception_config: PerceptionConfig,
        output_dir: Path,
        *,
        record_bag: bool = True,
        record_depth_video: bool = True,
        record_composite: bool = True,
        codec: str = "mp4v",
        finalize_h264: bool = True,
        ffmpeg_binary: str = "ffmpeg",
        warmup_frames: int | None = None,
    ):
        self.config = perception_config
        self.output_dir = output_dir.expanduser().resolve()
        self.record_bag = bool(record_bag)
        self.record_depth_video = bool(record_depth_video)
        self.record_composite = bool(record_composite)
        self.codec = codec
        self.finalize_h264 = bool(finalize_h264)
        self.ffmpeg_binary = ffmpeg_binary
        self.warmup_frames = (
            perception_config.warmup_frames
            if warmup_frames is None
            else int(warmup_frames)
        )
        if self.warmup_frames < 0 or self.warmup_frames > 300:
            raise ValueError("warmup_frames must be between 0 and 300")
        self.states: list[_CameraState] = []
        self.composite_writer: Any | None = None
        self.timestamp_handle: Any | None = None
        self.timestamp_writer: csv.DictWriter | None = None
        self.composite_frame_count = 0
        self.started_at_utc: str | None = None
        self.started_monotonic_ns: int | None = None
        self.stop_requested = False
        self.stop_reason = "not_started"
        self.errors: list[str] = []
        self.video_finalization: dict[str, dict[str, Any]] = {}
        self._closed = False
        self._snapshot_condition = threading.Condition()
        self._latest_rgbd: dict[str, RolloutRGBDFrame] = {}

    def _start_camera(self, spec: CameraSpec, rs: Any) -> _CameraState:
        cv2 = _require_cv2()
        pipeline = rs.pipeline()
        camera_config = rs.config()
        camera_config.enable_device(spec.serial)
        camera_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.rgb8,
            self.config.fps,
        )
        camera_config.enable_stream(
            rs.stream.depth,
            self.config.width,
            self.config.height,
            rs.format.z16,
            self.config.fps,
        )
        # The installed RealSense SDK uses the ROS2-native .db3 recording
        # container.  Older librealsense builds commonly used .bag here.
        bag_path = self.output_dir / f"camera_{spec.label}.db3"
        if self.record_bag:
            camera_config.enable_record_to_file(str(bag_path))
        try:
            profile = pipeline.start(camera_config)
        except Exception as exc:
            raise RolloutRecorderError(
                f"failed to open Camera {spec.label} ({spec.serial}); stop any viewer or "
                f"perception process that already owns this RealSense: {exc}"
            ) from exc
        try:
            device = profile.get_device()
            depth_scale = float(device.first_depth_sensor().get_depth_scale())
            _configure_color_exposure(device, spec, rs)
            recorder = None
            if self.record_bag:
                try:
                    recorder = device.as_recorder()
                    recorder.pause()
                except Exception:
                    recorder = None
            rgb_writer = _open_writer(
                self.output_dir / f"camera_{spec.label}_rgb.mp4",
                self.config.width,
                self.config.height,
                self.config.fps,
                self.codec,
            )
            depth_writer = (
                _open_writer(
                    self.output_dir / f"camera_{spec.label}_depth.mp4",
                    self.config.width,
                    self.config.height,
                    self.config.fps,
                    self.codec,
                )
                if self.record_depth_video
                else None
            )
            return _CameraState(
                spec=spec,
                pipeline=pipeline,
                align=rs.align(rs.stream.color),
                depth_scale=depth_scale,
                recorder=recorder,
                rgb_writer=rgb_writer,
                depth_writer=depth_writer,
            )
        except BaseException:
            pipeline.stop()
            raise

    def start(self) -> None:
        if self.states:
            raise RolloutRecorderError("recorder has already been started")
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RolloutRecorderError(
                "pyrealsense2 is required; use the configured cali environment"
            ) from exc
        self.output_dir.mkdir(parents=True, exist_ok=False)
        context = rs.context()
        available = {
            device.get_info(rs.camera_info.serial_number)
            for device in context.query_devices()
        }
        configured = {
            spec.serial
            for spec in self.config.cameras
            if spec.label in set(self.config.active_camera_labels)
        }
        missing = sorted(configured - available)
        if missing:
            raise RolloutRecorderError(
                f"configured RealSense serials are missing: {missing}; "
                f"available={sorted(available)}"
            )
        active_specs = [
            spec
            for spec in self.config.cameras
            if spec.label in set(self.config.active_camera_labels)
        ]
        self.record_composite = self.record_composite and {'A', 'B'}.issubset(
            spec.label for spec in active_specs)
        try:
            for spec in active_specs:
                self.states.append(self._start_camera(spec, rs))
            for _ in range(self.warmup_frames):
                for state in self.states:
                    state.pipeline.wait_for_frames(2000)
            for state in self.states:
                if state.recorder is not None:
                    state.recorder.resume()
            if self.record_composite:
                self.composite_writer = _open_writer(
                    self.output_dir / "composite_AB_depth.mp4",
                    self.config.width * 2,
                    self.config.height * 2,
                    self.config.fps,
                    self.codec,
                )
            self.timestamp_handle = (self.output_dir / "frame_timestamps.csv").open(
                "w", newline="", encoding="utf-8"
            )
            fieldnames = [
                "host_utc",
                "host_monotonic_ns",
                "elapsed_s",
                "camera_label",
                "serial",
                "color_frame_number",
                "color_device_timestamp_ms",
                "depth_frame_number",
                "depth_device_timestamp_ms",
                "valid_depth_fraction",
            ]
            self.timestamp_writer = csv.DictWriter(
                self.timestamp_handle, fieldnames=fieldnames
            )
            self.timestamp_writer.writeheader()
            self.timestamp_handle.flush()
            self.started_at_utc = _now()
            self.started_monotonic_ns = time.monotonic_ns()
            self.stop_reason = "recording"
        except BaseException:
            self.close()
            raise

    def _read_camera(self, state: _CameraState) -> _CapturedFrame:
        cv2 = _require_cv2()
        frames = state.align.process(state.pipeline.wait_for_frames(2000))
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RolloutRecorderError(
                f"Camera {state.spec.label} returned an incomplete RGB-D frame"
            )
        rgb = np.asanyarray(color.get_data()).copy()
        depth_m = (
            np.asanyarray(depth.get_data()).astype(np.float32) * state.depth_scale
        )
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_bgr = depth_to_bgr(
            depth_m,
            min_depth_m=self.config.min_depth_m,
            max_depth_m=self.config.max_depth_m,
        )
        valid = (
            np.isfinite(depth_m)
            & (depth_m > self.config.min_depth_m)
            & (depth_m < self.config.max_depth_m)
        )
        return _CapturedFrame(
            label=state.spec.label,
            serial=state.spec.serial,
            rgb=rgb,
            depth_m=depth_m,
            rgb_bgr=rgb_bgr,
            depth_bgr=depth_bgr,
            host_utc=_now(),
            host_monotonic_ns=time.monotonic_ns(),
            color_frame_number=int(color.get_frame_number()),
            color_device_timestamp_ms=float(color.get_timestamp()),
            depth_frame_number=int(depth.get_frame_number()),
            depth_device_timestamp_ms=float(depth.get_timestamp()),
            valid_depth_fraction=float(np.mean(valid)),
        )

    @staticmethod
    def _snapshot_frame(frame: _CapturedFrame) -> RolloutRGBDFrame:
        return RolloutRGBDFrame(
            label=frame.label,
            serial=frame.serial,
            rgb=frame.rgb.copy(),
            depth_m=frame.depth_m.copy(),
            host_utc=frame.host_utc,
            host_monotonic_ns=frame.host_monotonic_ns,
            color_frame_number=frame.color_frame_number,
            depth_frame_number=frame.depth_frame_number,
            valid_depth_fraction=frame.valid_depth_fraction,
        )

    def wait_for_latest_rgbd(
        self,
        *,
        after_monotonic_ns: int = 0,
        timeout_s: float = 3.0,
        labels: tuple[str, ...] = ("A", "B"),
    ) -> dict[str, RolloutRGBDFrame]:
        """Return one fresh synchronized recorder snapshot without reopening cameras."""

        if timeout_s <= 0:
            raise ValueError("snapshot timeout_s must be positive")
        required = tuple(dict.fromkeys(str(label) for label in labels))
        if not required:
            raise ValueError("at least one snapshot camera label is required")
        deadline = time.monotonic() + float(timeout_s)
        with self._snapshot_condition:
            while True:
                ready = all(
                    label in self._latest_rgbd
                    and self._latest_rgbd[label].host_monotonic_ns
                    > int(after_monotonic_ns)
                    for label in required
                )
                if ready:
                    return {
                        label: RolloutRGBDFrame(
                            **{
                                **self._latest_rgbd[label].__dict__,
                                "rgb": self._latest_rgbd[label].rgb.copy(),
                                "depth_m": self._latest_rgbd[label].depth_m.copy(),
                            }
                        )
                        for label in required
                    }
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    available = {
                        label: frame.host_monotonic_ns
                        for label, frame in self._latest_rgbd.items()
                    }
                    raise RolloutRecorderError(
                        "timed out waiting for fresh recorder RGB-D snapshot: "
                        f"required={list(required)} after={after_monotonic_ns} "
                        f"available={available}"
                    )
                if self._closed and not ready:
                    raise RolloutRecorderError(
                        "recorder closed before a fresh RGB-D snapshot became available"
                    )
                self._snapshot_condition.wait(timeout=remaining)

    def latest_pre_lift_rgb(self, *, after_ns: int, before_ns: int):
        """Freeze a recent contact-pose RGB frame without waiting for camera I/O.

        The callback calls this before allowing the next robot action. A frame
        during closure is valid; this does not claim the jaws were fully closed.
        Published snapshot arrays are replaced, never mutated by the recorder.
        """
        with self._snapshot_condition:
            frame = self._latest_rgbd.get('A')
            if (frame is None or not after_ns < frame.host_monotonic_ns <= before_ns
                    or before_ns - frame.host_monotonic_ns > 500_000_000):
                raise RolloutRecorderError('No recent Camera A frame at contact before lift')
            return frame

    def _write_timestamp(self, frame: _CapturedFrame) -> None:
        if self.timestamp_writer is None or self.started_monotonic_ns is None:
            raise RolloutRecorderError("timestamp writer is not initialized")
        self.timestamp_writer.writerow(
            {
                "host_utc": frame.host_utc,
                "host_monotonic_ns": frame.host_monotonic_ns,
                "elapsed_s": (
                    frame.host_monotonic_ns - self.started_monotonic_ns
                )
                / 1e9,
                "camera_label": frame.label,
                "serial": frame.serial,
                "color_frame_number": frame.color_frame_number,
                "color_device_timestamp_ms": frame.color_device_timestamp_ms,
                "depth_frame_number": frame.depth_frame_number,
                "depth_device_timestamp_ms": frame.depth_device_timestamp_ms,
                "valid_depth_fraction": frame.valid_depth_fraction,
            }
        )

    def record(self, duration_s: float = 0.0) -> dict[str, Any]:
        if not self.states or self.started_monotonic_ns is None:
            raise RolloutRecorderError("call start() before record()")
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        start_s = time.monotonic()
        last_report_s = start_s
        try:
            while not self.stop_requested:
                now_s = time.monotonic()
                if duration_s > 0 and now_s - start_s >= duration_s:
                    self.stop_reason = "duration_completed"
                    break
                captured: dict[str, _CapturedFrame] = {}
                for state in self.states:
                    frame = self._read_camera(state)
                    elapsed_s = (
                        frame.host_monotonic_ns - self.started_monotonic_ns
                    ) / 1e9
                    rgb = _label_frame(frame.rgb_bgr, f"Camera {frame.label} RGB", elapsed_s)
                    depth = _label_frame(
                        frame.depth_bgr, f"Camera {frame.label} depth", elapsed_s
                    )
                    desired_count = max(
                        state.encoded_frame_count + 1,
                        max(1, int(round(elapsed_s * self.config.fps))),
                    )
                    for _ in range(desired_count - state.encoded_frame_count):
                        state.rgb_writer.write(rgb)
                        if state.depth_writer is not None:
                            state.depth_writer.write(depth)
                    state.encoded_frame_count = desired_count
                    state.frame_count += 1
                    state.last_frame_number = frame.color_frame_number
                    captured[frame.label] = _CapturedFrame(
                        **{
                            **frame.__dict__,
                            "rgb_bgr": rgb,
                            "depth_bgr": depth,
                        }
                    )
                    self._write_timestamp(frame)
                with self._snapshot_condition:
                    self._latest_rgbd = {
                        label: self._snapshot_frame(frame)
                        for label, frame in captured.items()
                    }
                    self._snapshot_condition.notify_all()
                if self.composite_writer is not None and {"A", "B"}.issubset(captured):
                    composite = compose_four_panel(
                        captured["A"].rgb_bgr,
                        captured["B"].rgb_bgr,
                        captured["A"].depth_bgr,
                        captured["B"].depth_bgr,
                    )
                    composite_elapsed_s = max(
                        (
                            frame.host_monotonic_ns - self.started_monotonic_ns
                        )
                        / 1e9
                        for frame in captured.values()
                    )
                    desired_composite_count = max(
                        self.composite_frame_count + 1,
                        max(1, int(round(composite_elapsed_s * self.config.fps))),
                    )
                    for _ in range(
                        desired_composite_count - self.composite_frame_count
                    ):
                        self.composite_writer.write(composite)
                    self.composite_frame_count = desired_composite_count
                if self.timestamp_handle is not None:
                    self.timestamp_handle.flush()
                now_s = time.monotonic()
                if now_s - last_report_s >= 1.0:
                    counts = " ".join(
                        f"{state.spec.label}={state.frame_count}" for state in self.states
                    )
                    print(f"recording {now_s - start_s:7.1f}s | frames {counts}", flush=True)
                    last_report_s = now_s
        except KeyboardInterrupt:
            self.stop_reason = "keyboard_interrupt"
        except BaseException as exc:
            self.stop_reason = "recording_error"
            self.errors.append(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.close()
        return self.manifest()

    def request_stop(self, reason: str = "stop_requested") -> None:
        self.stop_reason = reason
        self.stop_requested = True
        with self._snapshot_condition:
            self._snapshot_condition.notify_all()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._snapshot_condition:
            self._snapshot_condition.notify_all()
        if self.composite_writer is not None:
            self.composite_writer.release()
            self.composite_writer = None
        for state in self.states:
            try:
                state.rgb_writer.release()
            except Exception:
                pass
            if state.depth_writer is not None:
                try:
                    state.depth_writer.release()
                except Exception:
                    pass
            try:
                state.pipeline.stop()
            except Exception as exc:
                self.errors.append(
                    f"Camera {state.spec.label} stop: {type(exc).__name__}: {exc}"
                )
        if self.timestamp_handle is not None:
            try:
                self.timestamp_handle.flush()
                self.timestamp_handle.close()
            finally:
                self.timestamp_handle = None
                self.timestamp_writer = None
        if self.finalize_h264 and self.output_dir.is_dir():
            video_paths = [
                self.output_dir / f"camera_{state.spec.label}_rgb.mp4"
                for state in self.states
            ]
            if self.record_depth_video:
                video_paths.extend(
                    self.output_dir / f"camera_{state.spec.label}_depth.mp4"
                    for state in self.states
                )
            if self.record_composite:
                video_paths.append(self.output_dir / "composite_AB_depth.mp4")
            for path in video_paths:
                if not path.is_file():
                    continue
                print(f"Finalizing H.264 MP4: {path.name}", flush=True)
                try:
                    self.video_finalization[path.name] = finalize_mp4_h264(
                        path,
                        ffmpeg_binary=self.ffmpeg_binary,
                    )
                except BaseException as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    self.video_finalization[path.name] = {
                        "status": "failed",
                        "error": message,
                    }
                    self.errors.append(f"{path.name} finalization: {message}")
        if self.stop_reason == "recording":
            self.stop_reason = "closed"
        if self.output_dir.is_dir():
            (self.output_dir / "recording_manifest.json").write_text(
                json.dumps(self.manifest(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def manifest(self) -> dict[str, Any]:
        ended_monotonic_ns = time.monotonic_ns()
        duration_s = (
            None
            if self.started_monotonic_ns is None
            else (ended_monotonic_ns - self.started_monotonic_ns) / 1e9
        )
        finalized_outputs = list(self.video_finalization.values())
        output_codec = (
            "h264"
            if finalized_outputs
            and all(item.get("status") == "completed" for item in finalized_outputs)
            else self.codec
        )
        return {
            "created_at": self.started_at_utc,
            "ended_at": _now(),
            "duration_s": duration_s,
            "stop_reason": self.stop_reason,
            "robot_control": False,
            "camera_ownership": "standalone_exclusive",
            "resolution": [self.config.width, self.config.height],
            "fps": self.config.fps,
            "codec": output_codec,
            "capture_codec": self.codec,
            "h264_finalization_enabled": self.finalize_h264,
            "video_finalization": dict(self.video_finalization),
            "record_bag": self.record_bag,
            "record_depth_video": self.record_depth_video,
            "record_composite": self.record_composite,
            "cameras": [
                {
                    "label": state.spec.label,
                    "serial": state.spec.serial,
                    "color_exposure": state.spec.color_exposure,
                    "color_white_balance": state.spec.color_white_balance,
                    "frame_count": state.frame_count,
                    "encoded_video_frame_count": state.encoded_frame_count,
                    "last_color_frame_number": state.last_frame_number,
                    "rgb_video": f"camera_{state.spec.label}_rgb.mp4",
                    "depth_video": (
                        f"camera_{state.spec.label}_depth.mp4"
                        if self.record_depth_video
                        else None
                    ),
                    "native_recording": (
                        f"camera_{state.spec.label}.db3" if self.record_bag else None
                    ),
                }
                for state in self.states
            ],
            "composite_video": (
                "composite_AB_depth.mp4" if self.record_composite else None
            ),
            "composite_encoded_frame_count": self.composite_frame_count,
            "timestamps": "frame_timestamps.csv",
            "errors": list(self.errors),
        }


def capture_observer_rgb(
    serial: str,
    output_dir: Path,
    *,
    label: str = "C",
    width: int = 1280,
    height: int = 720,
    fps: int = 15,
    color_exposure: float | None = 700.0,
    color_white_balance: float | None = 3800.0,
    warmup_frames: int = 20,
) -> dict[str, Any]:
    """Capture one RGB-only frame from an uncalibrated observer camera.

    The observer is deliberately independent of :class:`PerceptionConfig`:
    it does not load an extrinsic transform, does not open a depth stream, and
    never contributes points or robot coordinates to A/B perception.
    """

    if not str(serial).strip():
        raise RolloutRecorderError("observer camera serial must be non-empty")
    if int(width) <= 0 or int(height) <= 0 or int(fps) <= 0:
        raise RolloutRecorderError("observer RGB width/height/fps must be positive")
    if int(warmup_frames) < 0:
        raise RolloutRecorderError("observer warmup_frames must be non-negative")
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RolloutRecorderError(
            "pyrealsense2 is required for the observer RGB camera"
        ) from exc
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = rs.context()
    available = {
        device.get_info(rs.camera_info.serial_number)
        for device in context.query_devices()
    }
    serial_text = str(serial).strip()
    if serial_text not in available:
        raise RolloutRecorderError(
            f"observer camera {serial_text} is not connected; available={sorted(available)}"
        )
    pipeline = rs.pipeline()
    camera_config = rs.config()
    camera_config.enable_device(serial_text)
    camera_config.enable_stream(
        rs.stream.color,
        int(width),
        int(height),
        rs.format.rgb8,
        int(fps),
    )
    camera_spec = CameraSpec(
        str(label),
        serial_text,
        Path("."),
        color_exposure,
        color_white_balance,
    )
    pipeline.start(camera_config)
    try:
        _configure_color_exposure(pipeline.get_active_profile().get_device(), camera_spec, rs)
        for _ in range(int(warmup_frames)):
            pipeline.wait_for_frames(2000)
        frames = pipeline.wait_for_frames(2000)
        host_monotonic_ns = time.monotonic_ns()
        color = frames.get_color_frame()
        if not color:
            raise RolloutRecorderError(
                f"observer camera {serial_text} returned no color frame"
            )
        rgb = np.asanyarray(color.get_data()).copy()
        from PIL import Image

        image_path = output / f"camera_{str(label)}_observer_rgb.png"
        Image.fromarray(rgb).save(image_path)
        manifest = {
            "schema_version": 1,
            "created_at": _now(),
            "label": str(label),
            "serial": serial_text,
            "calibrated": False,
            "geometry_used": False,
            "purpose": "uncalibrated RGB observer for grasp/occlusion evaluation",
            "stream": "color RGB only",
            "resolution": [int(width), int(height)],
            "fps": int(fps),
            "color_frame_number": int(color.get_frame_number()),
            "color_device_timestamp_ms": float(color.get_timestamp()),
            "host_monotonic_ns": host_monotonic_ns,
            "rgb_image": str(image_path.resolve()),
        }
        (output / f"camera_{str(label)}_observer_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest
    finally:
        pipeline.stop()


class ObserverRGBRolloutRecorder:
    """Record an uncalibrated RGB-only observer during one robot action."""

    def __init__(
        self,
        serial: str,
        output_dir: Path,
        *,
        label: str = "C",
        width: int = 1280,
        height: int = 720,
        fps: int = 15,
        color_exposure: float | None = 700.0,
        color_white_balance: float | None = 3800.0,
        codec: str = "mp4v",
        finalize_h264: bool = True,
        ffmpeg_binary: str = "ffmpeg",
        warmup_frames: int = 20,
    ):
        self.serial = str(serial).strip()
        self.label = str(label).strip() or "C"
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.color_exposure = color_exposure
        self.color_white_balance = color_white_balance
        self.codec = str(codec)
        self.finalize_h264 = bool(finalize_h264)
        self.ffmpeg_binary = str(ffmpeg_binary)
        self.warmup_frames = int(warmup_frames)
        self.pipeline: Any | None = None
        self.writer: Any | None = None
        self.started_at_utc: str | None = None
        self.started_monotonic_ns: int | None = None
        self.frame_count = 0
        self.encoded_frame_count = 0
        self.stop_requested = False
        self.stop_reason = "not_started"
        self.errors: list[str] = []
        self.video_finalization: dict[str, Any] = {}
        self._closed = False
        self._snapshot_condition = threading.Condition()
        self._latest_rgb: ObserverRGBFrame | None = None

    @property
    def video_path(self) -> Path:
        return self.output_dir / f"camera_{self.label}_observer_rgb.mp4"

    def start(self) -> None:
        if self.pipeline is not None:
            raise RolloutRecorderError("observer recorder has already been started")
        if not self.serial:
            raise RolloutRecorderError("observer camera serial must be non-empty")
        if min(self.width, self.height, self.fps) <= 0:
            raise RolloutRecorderError("observer RGB width/height/fps must be positive")
        if self.warmup_frames < 0 or self.warmup_frames > 300:
            raise RolloutRecorderError("observer warmup_frames must be between 0 and 300")
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RolloutRecorderError(
                "pyrealsense2 is required for the observer RGB camera"
            ) from exc
        context = rs.context()
        available = {
            device.get_info(rs.camera_info.serial_number)
            for device in context.query_devices()
        }
        if self.serial not in available:
            raise RolloutRecorderError(
                f"observer camera {self.serial} is not connected; available={sorted(available)}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        camera_config = rs.config()
        camera_config.enable_device(self.serial)
        camera_config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.rgb8,
            self.fps,
        )
        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(camera_config)
            camera_spec = CameraSpec(
                self.label,
                self.serial,
                Path("."),
                self.color_exposure,
                self.color_white_balance,
            )
            _configure_color_exposure(profile.get_device(), camera_spec, rs)
            for _ in range(self.warmup_frames):
                pipeline.wait_for_frames(2000)
            self.writer = _open_writer(
                self.video_path,
                self.width,
                self.height,
                self.fps,
                self.codec,
            )
            self.pipeline = pipeline
            self.started_at_utc = _now()
            self.started_monotonic_ns = time.monotonic_ns()
            self.stop_reason = "recording"
        except BaseException:
            try:
                pipeline.stop()
            except Exception:
                pass
            if self.writer is not None:
                try:
                    self.writer.release()
                except Exception:
                    pass
                self.writer = None
            raise

    def record(self) -> dict[str, Any]:
        if self.pipeline is None or self.writer is None or self.started_monotonic_ns is None:
            raise RolloutRecorderError("call start() before record()")
        cv2 = _require_cv2()
        try:
            while not self.stop_requested:
                frames = self.pipeline.wait_for_frames(2000)
                color = frames.get_color_frame()
                if not color:
                    raise RolloutRecorderError(
                        f"observer camera {self.serial} returned no color frame"
                    )
                rgb = np.asanyarray(color.get_data()).copy()
                host_utc = _now()
                host_monotonic_ns = time.monotonic_ns()
                with self._snapshot_condition:
                    self._latest_rgb = ObserverRGBFrame(
                        label=self.label,
                        serial=self.serial,
                        rgb=rgb.copy(),
                        host_utc=host_utc,
                        host_monotonic_ns=host_monotonic_ns,
                        color_frame_number=int(color.get_frame_number()),
                        color_device_timestamp_ms=float(color.get_timestamp()),
                    )
                    self._snapshot_condition.notify_all()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                elapsed_s = (time.monotonic_ns() - self.started_monotonic_ns) / 1e9
                labelled = _label_frame(
                    bgr,
                    f"Camera {self.label} observer RGB",
                    elapsed_s,
                )
                self.writer.write(labelled)
                self.frame_count += 1
                self.encoded_frame_count += 1
        except KeyboardInterrupt:
            self.stop_reason = "keyboard_interrupt"
        except BaseException as exc:
            self.stop_reason = "recording_error"
            self.errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            self.close()
        return self.manifest()

    def request_stop(self, reason: str = "stop_requested") -> None:
        self.stop_reason = str(reason)
        self.stop_requested = True
        with self._snapshot_condition:
            self._snapshot_condition.notify_all()

    def wait_for_latest_rgb(
        self,
        *,
        after_monotonic_ns: int = 0,
        timeout_s: float = 3.0,
    ) -> ObserverRGBFrame:
        """Return a fresh observer frame without reopening Camera C.

        The recorder thread remains the sole owner of the RealSense pipeline.
        This avoids the camera-contention race that would occur if a checkpoint
        callback opened a second pipeline while the rollout video was running.
        """

        if timeout_s <= 0:
            raise ValueError("observer snapshot timeout_s must be positive")
        deadline = time.monotonic() + float(timeout_s)
        with self._snapshot_condition:
            while True:
                frame = self._latest_rgb
                if frame is not None and frame.host_monotonic_ns > int(after_monotonic_ns):
                    return ObserverRGBFrame(
                        label=frame.label,
                        serial=frame.serial,
                        rgb=frame.rgb.copy(),
                        host_utc=frame.host_utc,
                        host_monotonic_ns=frame.host_monotonic_ns,
                        color_frame_number=frame.color_frame_number,
                        color_device_timestamp_ms=frame.color_device_timestamp_ms,
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    latest = None if frame is None else frame.host_monotonic_ns
                    raise RolloutRecorderError(
                        "timed out waiting for a fresh Camera-C observer frame: "
                        f"after={after_monotonic_ns} latest={latest}"
                    )
                if self._closed and frame is None:
                    raise RolloutRecorderError(
                        "observer recorder closed before a fresh snapshot became available"
                    )
                self._snapshot_condition.wait(timeout=remaining)

    def save_snapshot(
        self,
        output_path: Path,
        *,
        after_monotonic_ns: int = 0,
        timeout_s: float = 3.0,
    ) -> dict[str, Any]:
        """Save one fresh RGB frame for an action-boundary evidence checkpoint."""

        frame = self.wait_for_latest_rgb(
            after_monotonic_ns=after_monotonic_ns,
            timeout_s=timeout_s,
        )
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.fromarray(frame.rgb).save(output)
        return {
            "status": "CAPTURED",
            "label": frame.label,
            "serial": frame.serial,
            "image": str(output),
            "host_utc": frame.host_utc,
            "host_monotonic_ns": frame.host_monotonic_ns,
            "color_frame_number": frame.color_frame_number,
            "color_device_timestamp_ms": frame.color_device_timestamp_ms,
            "source": "observer_rollout_recorder_latest_frame",
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._snapshot_condition:
            self._snapshot_condition.notify_all()
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
            self.writer = None
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as exc:
                self.errors.append(f"camera stop: {type(exc).__name__}: {exc}")
            self.pipeline = None
        if self.finalize_h264 and self.video_path.is_file():
            try:
                self.video_finalization[self.video_path.name] = finalize_mp4_h264(
                    self.video_path,
                    ffmpeg_binary=self.ffmpeg_binary,
                )
            except BaseException as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.video_finalization[self.video_path.name] = {
                    "status": "failed",
                    "error": message,
                }
                self.errors.append(f"{self.video_path.name} finalization: {message}")
        if self.stop_reason == "recording":
            self.stop_reason = "closed"
        if self.output_dir.is_dir():
            (self.output_dir / "observer_recording_manifest.json").write_text(
                json.dumps(self.manifest(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def manifest(self) -> dict[str, Any]:
        duration_s = (
            None
            if self.started_monotonic_ns is None
            else (time.monotonic_ns() - self.started_monotonic_ns) / 1e9
        )
        output_codec = (
            "h264"
            if self.video_finalization
            and all(item.get("status") == "completed" for item in self.video_finalization.values())
            else self.codec
        )
        return {
            "schema_version": 1,
            "created_at": self.started_at_utc,
            "ended_at": _now(),
            "duration_s": duration_s,
            "stop_reason": self.stop_reason,
            "robot_control": False,
            "calibrated": False,
            "geometry_used": False,
            "label": self.label,
            "serial": self.serial,
            "resolution": [self.width, self.height],
            "fps": self.fps,
            "codec": output_codec,
            "capture_codec": self.codec,
            "h264_finalization_enabled": self.finalize_h264,
            "video_finalization": dict(self.video_finalization),
            "frame_count": self.frame_count,
            "encoded_video_frame_count": self.encoded_frame_count,
            "rgb_video": self.video_path.name,
            "errors": list(self.errors),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--perception-config", default="config/perception.free_exploration.json"
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="recording duration; 0 records until Ctrl+C",
    )
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument(
        "--no-h264-finalize",
        action="store_true",
        help="keep the OpenCV capture codec instead of producing compatible H.264 MP4 files",
    )
    parser.add_argument("--warmup-frames", type=int)
    parser.add_argument(
        "--no-bag",
        "--no-native-recording",
        dest="no_bag",
        action="store_true",
        help="disable the SDK-native .db3 recording files",
    )
    parser.add_argument("--no-depth-video", action="store_true")
    parser.add_argument("--no-composite", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve()
    config_path = Path(args.perception_config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = PerceptionConfig.load(root, config_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / "results" / "rollout_recordings" / f"rollout_{stamp}"
    )
    recorder = DualRealSenseRolloutRecorder(
        config,
        output_dir,
        record_bag=not args.no_bag,
        record_depth_video=not args.no_depth_video,
        record_composite=not args.no_composite,
        codec=args.codec,
        finalize_h264=not args.no_h264_finalize,
        warmup_frames=args.warmup_frames,
    )

    def stop(signum: int, _frame: Any) -> None:
        recorder.request_stop(f"signal_{signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        print("Opening Camera A/B exclusively...", flush=True)
        recorder.start()
        print(f"RECORDING READY: {output_dir}", flush=True)
        print("Start the robot rollout now. Press Ctrl+C after return-home.", flush=True)
        manifest = recorder.record(args.duration_s)
    except BaseException as exc:
        recorder.errors.append(f"{type(exc).__name__}: {exc}")
        recorder.close()
        print(f"RECORDING FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print(f"RECORDING COMPLETE: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
