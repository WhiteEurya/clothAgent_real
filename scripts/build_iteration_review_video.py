#!/usr/bin/env python3
"""Build a chronological rollout review with per-iteration reasoning panels.

The cumulative rollout is already accelerated when it is produced by the main
loop. This tool recovers its iteration boundaries from the retained recording
manifests, applies the remaining speed multiplier, adds a readable intro hold,
and places selection/evaluation context beside the video. Iterations that never
sent robot motion are represented by an explicit no-rollout slate.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
)


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}"
        )


def _probe_duration(video: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"cannot probe {video}")
    duration = float(completed.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"invalid video duration for {video}: {duration}")
    return duration


def _probe_keyframe_times(video: Path) -> list[float]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-skip_frame",
            "nokey",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(video),
        ],
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"cannot probe {video}")
    values: list[float] = []
    for line in completed.stdout.splitlines():
        token = line.strip().rstrip(",").strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if math.isfinite(value) and value >= 0:
            values.append(value)
    return sorted(set(values))


def _find_output_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if (root / "combined_rollout.mp4").is_file():
        return root
    candidates = sorted(root.glob("results/molmo_keypoint_cli/*/combined_rollout.mp4"))
    if not candidates:
        raise FileNotFoundError(
            f"no combined_rollout.mp4 under run/output directory: {root}"
        )
    return candidates[-1].parent


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),)
        if bold
        else ()
    ) + FONT_CANDIDATES
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError("no CJK font found")


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return float(right - left)


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for char in paragraph:
            candidate = current + char
            if current and _text_width(draw, candidate, font) > width:
                lines.append(current.rstrip())
                current = char.lstrip()
            else:
                current = candidate
        if current:
            lines.append(current.rstrip())
    return lines


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    width: int,
    line_height: int,
    max_lines: int,
) -> int:
    x, y = xy
    lines = _wrap_text(draw, text, font, width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip("。.!！ ") + "..."
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def _proposal_for_record(record: dict[str, Any]) -> dict[str, Any]:
    proposal = record.get("proposal")
    if isinstance(proposal, dict):
        return proposal
    rejections = record.get("global_planning_rejections")
    if isinstance(rejections, list):
        for rejection in reversed(rejections):
            if not isinstance(rejection, dict):
                continue
            rejected = rejection.get("rejected_proposal")
            if isinstance(rejected, dict):
                return rejected
    return {}


def _default_logic(record: dict[str, Any]) -> str:
    selected = _proposal_for_record(record).get("selected_grasp")
    if isinstance(selected, dict) and selected.get("reason"):
        return str(selected["reason"])
    return "本轮在生成可执行抓点前失败，没有选择抓取点。"


def _default_outcome(record: dict[str, Any]) -> str:
    error = record.get("error")
    if error:
        return f"未执行：{error}"
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        return "没有保存 evaluator 结果。"
    progress = (evaluation.get("task_progress") or {}).get("status", "UNKNOWN")
    acquisition = (evaluation.get("grasp_acquisition") or {}).get(
        "status", "UNKNOWN"
    )
    structure = (evaluation.get("target_structure_acquired") or {}).get(
        "status", "UNKNOWN"
    )
    earliest = evaluation.get("earliest_failure_stage", "UNKNOWN")
    return (
        f"任务={progress}；抓取={acquisition}；目标层={structure}；"
        f"最早失败阶段={earliest}。"
    )


def _record_metadata(record: dict[str, Any]) -> list[str]:
    proposal = _proposal_for_record(record)
    selected = proposal.get("selected_grasp")
    grounding = record.get("global_grounding") or {}
    measurement = grounding.get("measurement") or {}
    diagnostic = measurement.get("surface_shape_diagnostic") or {}
    lines: list[str] = []
    if isinstance(selected, dict):
        lines.append(
            f"抓点: Camera {selected.get('camera', '?')} {selected.get('pixel_xy', '?')}"
        )
    xyz = measurement.get("base_xyz_median_mm")
    height = measurement.get("height_above_table_median_mm")
    if isinstance(xyz, list) and len(xyz) == 3:
        lines.append(
            "Base XYZ: " + ", ".join(f"{float(value):.1f}" for value in xyz) + " mm"
        )
    if isinstance(height, (int, float)):
        lines.append(f"桌面相对高度: {float(height):.1f} mm")
    shape = diagnostic.get("surface_shape")
    if shape:
        compression = "需要" if diagnostic.get("compression_probe_recommended") else "不需要"
        lines.append(f"局部形态: {shape}；下压试探: {compression}")
    skills = [
        str(item.get("name"))
        for item in proposal.get("skill_invocations") or []
        if isinstance(item, dict) and item.get("name")
    ]
    lines.append(f"Skill: {', '.join(skills) if skills else '未显式调用'}")
    return lines


def _panel_image(
    record: dict[str, Any],
    notes: dict[str, Any],
    output: Path,
    *,
    has_video: bool,
    target_speed: float,
    intro_s: float,
) -> None:
    width, height = 720, 1080
    image = Image.new("RGB", (width, height), (24, 27, 31))
    draw = ImageDraw.Draw(image)
    title_font = _font(42, bold=True)
    section_font = _font(27, bold=True)
    body_font = _font(23)
    small_font = _font(19)
    status = str(record.get("status", "UNKNOWN"))
    status_color = (
        (62, 185, 121)
        if status == "COMPLETED"
        else (229, 170, 61)
        if status == "RECOVERABLE_ERROR"
        else (221, 86, 86)
    )
    draw.rectangle((0, 0, width, 106), fill=(34, 39, 45))
    draw.rectangle((0, 102, width, 106), fill=status_color)
    iteration = int(record.get("iteration", 0))
    draw.text((38, 24), f"ITERATION {iteration:02d}", font=title_font, fill=(247, 249, 251))
    badge = status if has_video else f"{status} / NO ROLLOUT"
    badge_width = int(_text_width(draw, badge, small_font)) + 30
    draw.rounded_rectangle(
        (width - badge_width - 34, 32, width - 34, 74),
        radius=5,
        fill=status_color,
    )
    draw.text(
        (width - badge_width - 19, 39),
        badge,
        font=small_font,
        fill=(16, 19, 22),
    )

    y = 132
    for line in _record_metadata(record):
        y = _draw_wrapped(
            draw,
            (38, y),
            line,
            font=body_font,
            fill=(203, 211, 220),
            width=644,
            line_height=34,
            max_lines=2,
        )
    y += 20
    draw.line((38, y, width - 38, y), fill=(73, 82, 91), width=2)
    y += 24
    draw.text((38, y), "选择逻辑", font=section_font, fill=(247, 249, 251))
    y += 44
    logic = str(notes.get("logic") or _default_logic(record))
    y = _draw_wrapped(
        draw,
        (38, y),
        logic,
        font=body_font,
        fill=(227, 232, 237),
        width=644,
        line_height=35,
        max_lines=10,
    )
    y += 20
    draw.line((38, y, width - 38, y), fill=(73, 82, 91), width=2)
    y += 24
    draw.text((38, y), "实际结果", font=section_font, fill=status_color)
    y += 44
    outcome = str(notes.get("outcome") or _default_outcome(record))
    _draw_wrapped(
        draw,
        (38, y),
        outcome,
        font=body_font,
        fill=(227, 232, 237),
        width=644,
        line_height=35,
        max_lines=7,
    )
    footer = (
        f"动作录像 {target_speed:g}x；开头 {intro_s:g}s 静帧用于阅读"
        if has_video
        else f"本轮无机器人动作；错误页显示 {intro_s:g}s"
    )
    draw.text((38, height - 50), footer, font=small_font, fill=(142, 153, 164))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def _no_video_slate(iteration: int, output: Path) -> None:
    image = Image.new("RGB", (1920, 1080), (15, 17, 20))
    draw = ImageDraw.Draw(image)
    title_font = _font(52, bold=True)
    body_font = _font(30)
    title = f"ITERATION {iteration:02d}"
    body = "本轮没有机器人动作录像\n规划、边界检查或辅助感知阶段提前终止"
    title_w = _text_width(draw, title, title_font)
    draw.text(((1920 - title_w) / 2, 430), title, font=title_font, fill=(238, 241, 244))
    lines = body.splitlines()
    for index, line in enumerate(lines):
        line_w = _text_width(draw, line, body_font)
        draw.text(
            ((1920 - line_w) / 2, 515 + index * 48),
            line,
            font=body_font,
            fill=(156, 166, 176),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def _video_segment(
    source: Path,
    panel: Path,
    output: Path,
    *,
    start_s: float,
    duration_s: float,
    additional_speed: float,
    intro_s: float,
    preset: str,
) -> None:
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_s:.9f}",
            "-t",
            f"{duration_s:.9f}",
            "-i",
            str(source),
            "-loop",
            "1",
            "-framerate",
            "30",
            "-i",
            str(panel),
            "-filter_complex",
            (
                f"[0:v]setpts=PTS/{additional_speed:.9g},"
                f"tpad=start_duration={intro_s:.9g}:start_mode=clone,"
                "scale=1920:1080:force_original_aspect_ratio=decrease,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[left];"
                "[1:v]scale=720:1080,setsar=1[panel];"
                "[left][panel]hstack=inputs=2:shortest=1,format=yuv420p[out]"
            ),
            "-map",
            "[out]",
            "-an",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def _slate_segment(
    slate: Path,
    panel: Path,
    output: Path,
    *,
    duration_s: float,
    preset: str,
) -> None:
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-framerate",
            "30",
            "-i",
            str(slate),
            "-loop",
            "1",
            "-framerate",
            "30",
            "-i",
            str(panel),
            "-filter_complex",
            (
                "[0:v]scale=1920:1080,setsar=1[left];"
                "[1:v]scale=720:1080,setsar=1[panel];"
                "[left][panel]hstack=inputs=2:shortest=1,format=yuv420p[out]"
            ),
            "-map",
            "[out]",
            "-t",
            f"{duration_s:.9f}",
            "-an",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def build(args: argparse.Namespace) -> Path:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required")
    root = _find_output_root(args.run)
    source = root / "combined_rollout.mp4"
    records = [
        _load_json(path)
        for path in sorted(root.glob("iteration_[0-9][0-9][0-9].json"))
    ]
    if not records:
        raise FileNotFoundError(f"no iteration records under {root}")
    notes: dict[str, Any] = {}
    if args.notes is not None:
        loaded = _load_json(args.notes.expanduser().resolve())
        notes = {str(key): value for key, value in loaded.items()}

    recording_by_iteration: dict[int, dict[str, Any]] = {}
    theoretical_durations: list[tuple[int, float]] = []
    base_speed: float | None = None
    for path in sorted(root.glob("iteration_*/rollout_recording.json")):
        recording = _load_json(path)
        if recording.get("status") != "completed":
            continue
        iteration = int(path.parent.name.split("_")[-1])
        manifest = recording.get("manifest") or {}
        frames = manifest.get("composite_encoded_frame_count")
        fps = manifest.get("fps")
        append = recording.get("cumulative_video_append") or {}
        speed = append.get("playback_speed")
        if not all(isinstance(value, (int, float)) for value in (frames, fps, speed)):
            raise ValueError(f"incomplete cumulative recording metadata: {path}")
        if float(fps) <= 0 or float(speed) <= 0 or int(frames) <= 0:
            raise ValueError(f"invalid cumulative recording metadata: {path}")
        if base_speed is None:
            base_speed = float(speed)
        elif not math.isclose(base_speed, float(speed), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("cumulative video segments use inconsistent playback speeds")
        duration = float(frames) / float(fps) / float(speed)
        recording_by_iteration[iteration] = recording
        theoretical_durations.append((iteration, duration))
    if not theoretical_durations or base_speed is None:
        raise ValueError("no completed cumulative rollout segments were found")
    if args.target_speed < base_speed:
        raise ValueError(
            f"target speed {args.target_speed:g}x is below source speed {base_speed:g}x"
        )

    actual_duration = _probe_duration(source)
    theoretical_total = sum(duration for _, duration in theoretical_durations)
    duration_scale = actual_duration / theoretical_total
    keyframes = _probe_keyframe_times(source)
    expected_boundaries: list[float] = []
    expected_cursor = 0.0
    for _, duration in theoretical_durations[:-1]:
        expected_cursor += duration * duration_scale
        expected_boundaries.append(expected_cursor)
    snapped_boundaries: list[float] = []
    previous = 0.0
    for expected in expected_boundaries:
        candidates = [value for value in keyframes if value > previous + 0.1]
        nearest = min(candidates, key=lambda value: abs(value - expected))
        if abs(nearest - expected) > 1.0:
            raise ValueError(
                "could not recover a cumulative-video iteration boundary near "
                f"{expected:.3f}s; nearest keyframe={nearest:.3f}s"
            )
        snapped_boundaries.append(nearest)
        previous = nearest
    all_boundaries = [0.0, *snapped_boundaries, actual_duration]
    segment_bounds: dict[int, tuple[float, float]] = {}
    for index, (iteration, _) in enumerate(theoretical_durations):
        start = all_boundaries[index]
        end = all_boundaries[index + 1]
        segment_bounds[iteration] = (start, end - start)

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / f"iteration_reasoning_{args.target_speed:g}x.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    additional_speed = args.target_speed / base_speed
    with tempfile.TemporaryDirectory(prefix="cloth_iteration_review_") as temp_raw:
        temp = Path(temp_raw)
        segments: list[Path] = []
        manifest_items: list[dict[str, Any]] = []
        for index, record in enumerate(records, start=1):
            iteration = int(record.get("iteration", index))
            has_video = iteration in segment_bounds
            panel = temp / f"panel_{iteration:03d}.png"
            note = notes.get(str(iteration))
            if not isinstance(note, dict):
                note = {}
            _panel_image(
                record,
                note,
                panel,
                has_video=has_video,
                target_speed=args.target_speed,
                intro_s=args.intro_s,
            )
            segment = temp / f"segment_{iteration:03d}.mp4"
            if has_video:
                start_s, duration_s = segment_bounds[iteration]
                _video_segment(
                    source,
                    panel,
                    segment,
                    start_s=start_s,
                    duration_s=duration_s,
                    additional_speed=additional_speed,
                    intro_s=args.intro_s,
                    preset=args.preset,
                )
                source_range: list[float] | None = [start_s, start_s + duration_s]
            else:
                slate = temp / f"slate_{iteration:03d}.png"
                _no_video_slate(iteration, slate)
                _slate_segment(
                    slate,
                    panel,
                    segment,
                    duration_s=args.intro_s,
                    preset=args.preset,
                )
                source_range = None
            segments.append(segment)
            manifest_items.append(
                {
                    "iteration": iteration,
                    "status": record.get("status"),
                    "has_rollout_video": has_video,
                    "source_range_s": source_range,
                    "logic": note.get("logic") or _default_logic(record),
                    "outcome": note.get("outcome") or _default_outcome(record),
                }
            )
            print(
                f"prepared iteration {iteration:03d}/{len(records):03d} "
                f"({'video' if has_video else 'no rollout'})",
                flush=True,
            )
        concat = temp / "segments.txt"
        concat.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in segments),
            encoding="utf-8",
        )
        _run(
            [
                "ffmpeg",
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
                str(concat),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output),
            ]
        )
    review_manifest = output.with_suffix(".json")
    review_manifest.write_text(
        json.dumps(
            {
                "source": str(source),
                "output": str(output),
                "source_playback_speed": base_speed,
                "target_playback_speed": args.target_speed,
                "intro_hold_s": args.intro_s,
                "source_duration_s": actual_duration,
                "output_duration_s": _probe_duration(output),
                "iterations": manifest_items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run",
        type=Path,
        help="run directory or results/molmo_keypoint_cli/<timestamp> directory",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--notes", type=Path, help="optional per-iteration Chinese notes JSON")
    parser.add_argument("--target-speed", type=float, default=32.0)
    parser.add_argument("--intro-s", type=float, default=4.0)
    parser.add_argument(
        "--preset",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium"),
        default="veryfast",
    )
    args = parser.parse_args()
    if not math.isfinite(args.target_speed) or args.target_speed <= 0:
        parser.error("--target-speed must be positive")
    if not math.isfinite(args.intro_s) or args.intro_s <= 0:
        parser.error("--intro-s must be positive")
    output = build(args)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
