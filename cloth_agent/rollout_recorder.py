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
from pathlib import Path
import signal
import shutil
import subprocess
import time
from typing import Any

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
    rgb_bgr: np.ndarray
    depth_bgr: np.ndarray
    host_utc: str
    host_monotonic_ns: int
    color_frame_number: int
    color_device_timestamp_ms: float
    depth_frame_number: int
    depth_device_timestamp_ms: float
    valid_depth_fraction: float


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

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
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
