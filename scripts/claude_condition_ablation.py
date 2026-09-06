#!/usr/bin/env python3
"""Run a four-condition Claude evidence ablation on saved garment rollouts.

The experiment never opens a camera or robot.  It discovers saved
``recording_manifest.json`` files, extracts before/during/after stills from the
recorded videos, marks the historical candidate when one is available, and
asks Claude the same ten questions under four information conditions:

  A  before/after Camera-A RGB only
  B  A + before/after Camera-B RGB
  C  B + saved RGB-D/height visualizations
  D  before/during/after A/B RGB + height visualizations + action metadata

One Claude call is made per segment and condition.  Results are written as
individual JSON files plus a JSONL ledger and an aggregate summary.  Existing
responses can be reused with ``--resume``.  ``--dry-run`` builds and indexes
the evidence bundles without invoking Claude, which is useful for auditing a
large rollout collection before spending API/CLI time.

Example::

    python scripts/claude_condition_ablation.py \
      --runs-root runs \
      --limit 50 \
      --output-dir runs/claude_condition_ablation_50

The default condition definitions intentionally make A/B/C static before/after
comparisons while D is the temporal/action-aware condition.  This preserves
the proposed information ladder and makes UNKNOWN answers meaningful for
acquisition and motion questions under the weaker conditions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("A", "B", "C", "D")
LABELS = ("YES", "NO", "UNKNOWN")

QUESTIONS: tuple[dict[str, str], ...] = (
    {"id": "Q1", "text": "Is the marked candidate located on a clearly visible garment boundary?"},
    {"id": "Q2", "text": "Does the candidate look like an independent flap or separable layer?"},
    {"id": "Q3", "text": "Was the candidate acquisition successful?"},
    {"id": "Q4", "text": "Did the intended target move with the gripper?"},
    {"id": "Q5", "text": "Did surrounding cloth move together with the target?"},
    {"id": "Q6", "text": "Is there visible local separation between layers?"},
    {"id": "Q7", "text": "Is newly exposed material visible after the action?"},
    {"id": "Q8", "text": "Is a new recognizable garment structure visible after the action?"},
    {"id": "Q9", "text": "Is garment orientation already unambiguous?"},
    {"id": "Q10", "text": "Is there enough evidence to call a folding skill now?"},
)


class AblationError(RuntimeError):
    """Raised for deterministic evidence or invocation errors."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    raise TypeError(f"cannot encode {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _read_json(path: Path, *, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return default


def _parse_time(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _resolve_path(raw: Any, *, base: Path) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _find_recording_manifests(inputs: Sequence[Path]) -> list[Path]:
    """Discover manifests from run roots or explicit recording directories."""

    found: set[Path] = set()
    for raw in inputs:
        root = raw.expanduser().resolve()
        if root.is_file() and root.name == "recording_manifest.json":
            found.add(root)
            continue
        if root.is_dir() and (root / "recording_manifest.json").is_file():
            found.add((root / "recording_manifest.json").resolve())
            continue
        if root.is_dir():
            found.update(path.resolve() for path in root.glob("**/recording_manifest.json"))
    return sorted(found, key=lambda path: str(path))


def _load_video_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False, shell=False)
    if completed.returncode != 0:
        raise AblationError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    try:
        value = float(completed.stdout.strip())
    except ValueError as exc:
        raise AblationError(f"invalid video duration for {path}: {completed.stdout!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise AblationError(f"invalid video duration for {path}: {value}")
    return value


def _extract_frame(video: Path, timestamp_s: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.jpg")
    temporary.unlink(missing_ok=True)
    try:
        # Seeking a few milliseconds before an MP4's nominal duration can
        # still miss when the final packet is held in an encoder buffer.  A
        # one-second fallback keeps the requested phase while guaranteeing a
        # decodable frame for short recordings.
        attempts = [max(0.0, float(timestamp_s))]
        if attempts[0] > 0.75:
            attempts.append(max(0.0, attempts[0] - 1.0))
        last_detail = ""
        for seek_s in attempts:
            temporary.unlink(missing_ok=True)
            command = [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{seek_s:.6f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(temporary),
            ]
            completed = subprocess.run(command, text=True, capture_output=True, check=False, shell=False)
            if completed.returncode == 0 and temporary.is_file() and temporary.stat().st_size > 0:
                temporary.replace(output)
                return
            last_detail = (completed.stderr or completed.stdout).strip()
        raise AblationError(f"frame extraction failed for {video} at {timestamp_s:.3f}s: {last_detail}")
    finally:
        temporary.unlink(missing_ok=True)


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _candidate_from_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    selected = payload.get("selected_grasp")
    if not isinstance(selected, Mapping):
        return None
    camera = selected.get("camera")
    pixel = selected.get("pixel_xy")
    if camera not in {"A", "B"} or not isinstance(pixel, Sequence) or isinstance(pixel, (str, bytes)):
        return None
    if len(pixel) != 2:
        return None
    try:
        x, y = int(round(float(pixel[0]))), int(round(float(pixel[1])))
    except (TypeError, ValueError):
        return None
    return {"camera": camera, "pixel_xy": [x, y], "reason": selected.get("reason")}


def _candidate_from_legacy_selection(*payloads: Mapping[str, Any]) -> dict[str, Any] | None:
    """Recover candidates from collar-style runs without proposal.json."""

    for payload in payloads:
        selection = payload.get("selection")
        if not isinstance(selection, Mapping):
            selection = payload
        camera = selection.get("camera")
        pixel = selection.get("pixel_xy")
        if camera not in {"A", "B"} or not isinstance(pixel, Sequence) or isinstance(pixel, (str, bytes)) or len(pixel) != 2:
            continue
        try:
            x, y = int(round(float(pixel[0]))), int(round(float(pixel[1])))
        except (TypeError, ValueError):
            continue
        return {"camera": camera, "pixel_xy": [x, y], "reason": selection.get("reason") or selection.get("evidence")}
    return None


def _mark_candidate(source: Path, output: Path, candidate: Mapping[str, Any] | None) -> None:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    if candidate is not None:
        width, height = image.size
        x, y = candidate["pixel_xy"]
        x = int(_clamp(x, 0, width - 1))
        y = int(_clamp(y, 0, height - 1))
        radius = max(12, min(width, height) // 45)
        red = (228, 25, 25)
        yellow = (255, 230, 0)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=yellow, width=5)
        draw.line((x - radius * 2, y, x + radius * 2, y), fill=red, width=4)
        draw.line((x, y - radius * 2, x, y + radius * 2), fill=red, width=4)
        label = f"candidate {candidate['camera']} ({x},{y})"
        label_font = _font(18)
        left, top, right, bottom = draw.textbbox((14, 14), label, font=label_font)
        draw.rectangle((8, 8, right + 10, bottom + 8), fill=(0, 0, 0))
        draw.text((14, 14), label, fill=(255, 255, 255), font=label_font)
    image.save(output, quality=94)


def _mark_optional(source: Path | None, output: Path, candidate: Mapping[str, Any] | None) -> Path | None:
    if source is None or not source.is_file():
        return None
    _mark_candidate(source, output, candidate)
    return output


def _load_nearby_perception(
    run_root: Path,
    *,
    target_time: datetime | None,
) -> tuple[Path | None, dict[str, Any]]:
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for result_path in run_root.glob("results/**/result.json"):
        if result_path.parent.name.startswith("iteration_"):
            continue
        payload = _read_json(result_path)
        if not isinstance(payload, Mapping):
            continue
        views = payload.get("views")
        if not isinstance(views, list) and "perception_mode" not in payload:
            continue
        created = _parse_time(payload.get("created_at"))
        if target_time is not None and created is not None:
            distance = abs((created - target_time).total_seconds())
        else:
            distance = abs(result_path.stat().st_mtime - (target_time.timestamp() if target_time else result_path.stat().st_mtime))
        candidates.append((distance, result_path.parent.resolve(), dict(payload)))
    if not candidates:
        return None, {}
    candidates.sort(key=lambda item: (item[0], str(item[1])))
    return candidates[0][1], candidates[0][2]


def _perception_asset(perception_dir: Path | None, names: Sequence[str]) -> Path | None:
    if perception_dir is None:
        return None
    for name in names:
        path = perception_dir / name
        if path.is_file():
            return path.resolve()
    return None


@dataclass
class Segment:
    segment_id: str
    manifest_path: Path
    recording_dir: Path
    run_root: Path
    iteration_dir: Path
    videos: dict[str, Path]
    duration_s: float
    before_s: float
    during_s: float
    after_s: float
    action_start_s: float
    action_end_s: float
    candidate: dict[str, Any] | None
    proposal: dict[str, Any]
    execution: dict[str, Any]
    evaluation: dict[str, Any]
    result: dict[str, Any]
    perception_dir: Path | None
    perception: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("manifest_path", "recording_dir", "run_root", "iteration_dir", "perception_dir"):
            value = payload.get(key)
            payload[key] = str(value) if value is not None else None
        payload["videos"] = {key: str(value) for key, value in self.videos.items()}
        return payload


def _segment_from_manifest(manifest_path: Path) -> Segment | None:
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        return None
    recording_dir = manifest_path.parent.resolve()
    iteration_dir = recording_dir.parent.resolve()
    run_root = iteration_dir
    while run_root.parent != run_root and run_root.name not in {"runs", "results"}:
        if (run_root / "run_metadata.json").is_file() or (run_root / "workspace").is_dir():
            break
        run_root = run_root.parent
    # A recording may live under results/<pipeline>/<timestamp>/iteration_N; in
    # that layout the run root is the ancestor with run_metadata/workspace.
    for ancestor in [recording_dir, *recording_dir.parents]:
        if (ancestor / "run_metadata.json").is_file() or (ancestor / "workspace").is_dir():
            run_root = ancestor.resolve()
            break

    video_names = {
        "A_rgb": "camera_A_rgb.mp4",
        "B_rgb": "camera_B_rgb.mp4",
        "A_depth": "camera_A_depth.mp4",
        "B_depth": "camera_B_depth.mp4",
    }
    videos = {key: recording_dir / name for key, name in video_names.items() if (recording_dir / name).is_file()}
    if "A_rgb" not in videos:
        return None
    # The recorder manifest measures wall-clock ownership time, while an MP4
    # can end a few seconds earlier because encoder buffers are finalized after
    # capture stops.  Use the actual video duration for seek/clamp arithmetic;
    # otherwise an ``after`` seek can land past the final decodable frame.
    video_duration = _load_video_duration(videos["A_rgb"])
    duration_raw = manifest.get("duration_s")
    try:
        manifest_duration = float(duration_raw)
    except (TypeError, ValueError):
        manifest_duration = video_duration
    duration = video_duration if not math.isfinite(manifest_duration) or manifest_duration <= 0 else min(video_duration, manifest_duration)

    execution = _read_json(iteration_dir / "execution.json", default={})
    proposal = _read_json(iteration_dir / "proposal.json", default={})
    legacy_plan = _read_json(iteration_dir / "validated_plan.json", default={})
    legacy_selection = _read_json(iteration_dir / "claude_collar_selection.json", default={})
    evaluation = _read_json(iteration_dir / "evaluation.json", default={})
    result = _read_json(iteration_dir / "result.json", default={})
    if not isinstance(execution, Mapping):
        execution = {}
    if not isinstance(proposal, Mapping):
        proposal = {}
    if not isinstance(evaluation, Mapping):
        evaluation = {}
    if not isinstance(result, Mapping):
        result = {}

    recording_start = _parse_time(manifest.get("created_at"))
    execution_start = _parse_time(execution.get("started_at"))
    execution_end = _parse_time(execution.get("completed_at"))
    if recording_start is not None and execution_start is not None:
        action_start = _clamp((execution_start - recording_start).total_seconds(), 0.0, duration)
    else:
        action_start = duration * 0.25
    if recording_start is not None and execution_end is not None:
        action_end = _clamp((execution_end - recording_start).total_seconds(), action_start, duration)
    else:
        action_end = duration * 0.80
    before = _clamp(action_start - 1.0, 0.0, duration - 0.01)
    after = _clamp(action_end + 1.0, 0.0, duration - 0.01)
    if after <= before:
        after = duration - 0.01
    during = _clamp(action_start + max(0.5, (action_end - action_start) * 0.5), before, after)

    perception_payload = result.get("perception") if isinstance(result, Mapping) else {}
    perception_time = _parse_time(perception_payload.get("created_at")) if isinstance(perception_payload, Mapping) else None
    perception_dir, perception = _load_nearby_perception(run_root, target_time=perception_time or recording_start)
    candidate = _candidate_from_payload(proposal)
    if candidate is None:
        candidate = _candidate_from_legacy_selection(legacy_plan, legacy_selection)
    if candidate is not None and "selected_grasp" not in proposal:
        proposal = {**dict(proposal), "selected_grasp": candidate}
    iteration_name = iteration_dir.name
    digest = hashlib.sha1(str(recording_dir).encode("utf-8")).hexdigest()[:8]
    segment_id = f"{iteration_name}_{digest}"
    return Segment(
        segment_id=segment_id,
        manifest_path=manifest_path.resolve(),
        recording_dir=recording_dir,
        run_root=run_root,
        iteration_dir=iteration_dir,
        videos={key: path.resolve() for key, path in videos.items()},
        duration_s=float(duration),
        before_s=float(before),
        during_s=float(during),
        after_s=float(after),
        action_start_s=float(action_start),
        action_end_s=float(action_end),
        candidate=candidate,
        proposal=dict(proposal),
        execution=dict(execution),
        evaluation=dict(evaluation),
        result=dict(result),
        perception_dir=perception_dir,
        perception=perception,
    )


def _sample_segments(segments: Sequence[Segment], *, limit: int, seed: int) -> list[Segment]:
    if limit <= 0 or len(segments) <= limit:
        return list(segments)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(segments)), limit))
    return [segments[index] for index in indices]


def _prepare_bundle(segment: Segment, bundle_dir: Path) -> dict[str, Any]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    candidate = segment.candidate
    extracted: dict[str, str] = {}
    for view, video_key in (("A", "A_rgb"), ("B", "B_rgb")):
        video = segment.videos.get(video_key)
        if video is None:
            continue
        for phase, timestamp in (("before", segment.before_s), ("during", segment.during_s), ("after", segment.after_s)):
            raw = bundle_dir / f"{phase}_{view}_rgb_raw.jpg"
            marked = bundle_dir / f"{phase}_{view}_rgb.jpg"
            if not raw.is_file():
                _extract_frame(video, timestamp, raw)
            camera_candidate = candidate if candidate and candidate.get("camera") == view else None
            if not marked.is_file():
                _mark_candidate(raw, marked, camera_candidate)
            extracted[f"{phase}_{view}_rgb"] = str(marked)

    height_paths: dict[str, Path | None] = {
        "A_height": _perception_asset(segment.perception_dir, ("camera_A_height_map_heatmap.png", "camera_A_height_map_heatmap_global.png", "camera_A_height_map_boundary.png")),
        "B_height": _perception_asset(segment.perception_dir, ("camera_B_height_map_heatmap.png", "camera_B_height_map_heatmap_global.png", "camera_B_height_map_boundary.png")),
    }
    for key, source in height_paths.items():
        if source is None:
            continue
        view = key[0]
        marked = bundle_dir / f"before_{view}_height.png"
        camera_candidate = candidate if candidate and candidate.get("camera") == view else None
        if not marked.is_file():
            _mark_candidate(source, marked, camera_candidate)
        extracted[key] = str(marked)

    # If no saved heatmap was found, preserve a depth-video still as an
    # explicitly labelled fallback rather than silently calling it height.
    for view, video_key in (("A", "A_depth"), ("B", "B_depth")):
        if f"{view}_height" in extracted or video_key not in segment.videos:
            continue
        raw = bundle_dir / f"before_{view}_depth_raw.jpg"
        marked = bundle_dir / f"before_{view}_depth.png"
        if not raw.is_file():
            _extract_frame(segment.videos[video_key], segment.before_s, raw)
        camera_candidate = candidate if candidate and candidate.get("camera") == view else None
        if not marked.is_file():
            _mark_candidate(raw, marked, camera_candidate)
        extracted[f"{view}_depth_visualization"] = str(marked)

    metadata = {
        "segment": segment.as_dict(),
        "candidate": candidate,
        "timing": {
            "before_s": segment.before_s,
            "during_s": segment.during_s,
            "after_s": segment.after_s,
            "action_start_s": segment.action_start_s,
            "action_end_s": segment.action_end_s,
            "timing_note": "Times are relative to recording_manifest.created_at; execution timestamps are used when available.",
        },
        "extracted": extracted,
    }
    _write_json(bundle_dir / "bundle_manifest.json", metadata)
    return metadata


def _condition_assets(bundle: Mapping[str, Any], condition: str) -> list[Path]:
    extracted = bundle.get("extracted", {})
    if not isinstance(extracted, Mapping):
        return []
    names: list[str]
    if condition == "A":
        names = ["before_A_rgb", "after_A_rgb"]
    elif condition == "B":
        names = ["before_A_rgb", "after_A_rgb", "before_B_rgb", "after_B_rgb"]
    elif condition == "C":
        names = ["before_A_rgb", "after_A_rgb", "before_B_rgb", "after_B_rgb", "A_height", "B_height", "A_depth_visualization", "B_depth_visualization"]
    elif condition == "D":
        names = [
            "before_A_rgb", "during_A_rgb", "after_A_rgb",
            "before_B_rgb", "during_B_rgb", "after_B_rgb",
            "A_height", "B_height", "A_depth_visualization", "B_depth_visualization",
        ]
    else:
        raise ValueError(condition)
    paths: list[Path] = []
    for name in names:
        raw = extracted.get(name)
        if isinstance(raw, str) and Path(raw).is_file():
            paths.append(Path(raw).resolve())
    return paths


def _condition_description(condition: str) -> str:
    return {
        "A": "Camera A before/after RGB only; the candidate is marked in the image when available.",
        "B": "Camera A and Camera B before/after RGB; no depth/height visualization and no action metadata.",
        "C": "Camera A/B before/after RGB plus saved height/depth visualizations; no action metadata.",
        "D": "Camera A/B before/during/after RGB, saved height/depth visualizations, and historical action metadata.",
    }[condition]


def _action_metadata(segment: Segment) -> dict[str, Any]:
    # Keep D's metadata factual and bounded.  The model is explicitly told
    # that commands are not evidence of successful acquisition.
    actions = segment.execution.get("requested_robot_actions")
    if not isinstance(actions, list):
        actions = segment.proposal.get("actions", [])
    compact_actions: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        item: dict[str, Any] = {"name": action.get("name"), "args": action.get("args", {})}
        if "success" in action:
            item["success"] = action.get("success")
        if action.get("error"):
            item["error"] = action.get("error")
        compact_actions.append(item)
    evaluation = segment.evaluation
    evaluation_summary: dict[str, Any] = {}
    if isinstance(evaluation, Mapping):
        for key in (
            "earliest_failure_stage",
            "task_progress",
            "grasp_acquisition",
            "target_structure_acquired",
            "structure_engagement",
            "opening_relevance",
            "transport",
            "laydown",
        ):
            if key in evaluation:
                evaluation_summary[key] = evaluation[key]
    return {
        "candidate": segment.candidate,
        "proposal_selected_grasp": segment.proposal.get("selected_grasp"),
        "historical_actions": compact_actions,
        "execution_status": {
            "physical_execution": segment.execution.get("physical_execution"),
            "preflight_completed": segment.execution.get("preflight_completed"),
            "execution_completed": segment.execution.get("execution_completed"),
            "robot_errors": segment.execution.get("robot_errors"),
        },
        "saved_evaluation_summary": evaluation_summary,
    }


def _schema() -> dict[str, Any]:
    answer_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string", "enum": list(LABELS)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "string", "minLength": 1},
        },
        "required": ["label", "confidence", "evidence"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "segment_id": {"type": "string", "minLength": 1},
            "condition": {"type": "string", "enum": list(CONDITIONS)},
            "answers": {
                "type": "object",
                "additionalProperties": False,
                "properties": {question["id"]: answer_schema for question in QUESTIONS},
                "required": [question["id"] for question in QUESTIONS],
            },
            "overall_notes": {"type": "string", "minLength": 1},
        },
        "required": ["segment_id", "condition", "answers", "overall_notes"],
    }


def _validate_response(payload: Any, *, segment_id: str, condition: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise AblationError("Claude response is not a JSON object")
    if payload.get("segment_id") != segment_id:
        raise AblationError(f"segment_id mismatch: expected {segment_id!r}, got {payload.get('segment_id')!r}")
    if payload.get("condition") != condition:
        raise AblationError(f"condition mismatch: expected {condition!r}, got {payload.get('condition')!r}")
    answers = payload.get("answers")
    if not isinstance(answers, Mapping):
        raise AblationError("response.answers must be an object")
    normalized: dict[str, Any] = {"segment_id": segment_id, "condition": condition, "answers": {}, "overall_notes": str(payload.get("overall_notes", ""))}
    if not normalized["overall_notes"].strip():
        raise AblationError("overall_notes must be non-empty")
    for question in QUESTIONS:
        key = question["id"]
        value = answers.get(key)
        if not isinstance(value, Mapping):
            raise AblationError(f"missing answer object for {key}")
        label = value.get("label")
        if label not in LABELS:
            raise AblationError(f"{key}.label must be one of {LABELS}, got {label!r}")
        try:
            confidence = float(value.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise AblationError(f"{key}.confidence is not numeric") from exc
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise AblationError(f"{key}.confidence must be within [0,1]")
        evidence = value.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            raise AblationError(f"{key}.evidence must be non-empty")
        normalized["answers"][key] = {
            "label": label,
            "confidence": confidence,
            "evidence": evidence,
        }
    return normalized


def _json_from_cli(text: str) -> dict[str, Any]:
    """Extract Claude's structured output from its JSON envelope."""

    candidates = [text.strip()]
    try:
        outer = json.loads(text)
    except json.JSONDecodeError:
        outer = None
    if isinstance(outer, Mapping):
        for key in ("structured_output", "structuredOutput", "result"):
            value = outer.get(key)
            if isinstance(value, Mapping):
                return dict(value)
            if isinstance(value, str):
                candidates.insert(0, value)
        if not any(key in outer for key in ("structured_output", "structuredOutput", "result")):
            return dict(outer)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, Mapping):
                return dict(value)
        except json.JSONDecodeError:
            pass
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping):
                return dict(value)
    raise AblationError("could not extract a JSON object from Claude output")


def _build_prompt(segment: Segment, bundle: Mapping[str, Any], condition: str) -> str:
    assets = _condition_assets(bundle, condition)
    image_lines = "\n".join(f"- {path.name}" for path in assets) or "- none"
    metadata = _action_metadata(segment) if condition == "D" else None
    questions = "\n".join(f"{q['id']}: {q['text']} Answer YES, NO, or UNKNOWN." for q in QUESTIONS)
    prompt = (
        "You are conducting a controlled offline evidence ablation for garment manipulation. "
        "Use only the supplied images and, for condition D, the supplied historical metadata. "
        "Do not execute commands, write files, or infer hidden state. The red/yellow cross marks "
        "the historical candidate when a candidate was available; if no mark is present, answer "
        "candidate-specific questions UNKNOWN.\n\n"
        f"Segment: {segment.segment_id}\n"
        f"Condition {condition}: {_condition_description(condition)}\n"
        "Important: commands in metadata are not proof that the gripper acquired cloth. Static "
        "before/after evidence cannot prove an intermediate event; use UNKNOWN rather than guessing. "
        "Separate the target from surrounding cloth. A whole-body translation is not local separation. "
        "For Q10, answer YES only when the visible garment state is sufficiently understood and a "
        "folding skill would be justified, not merely because an action completed.\n\n"
        f"Images available for Read (inside the bundle directory):\n{image_lines}\n\n"
        f"Questions:\n{questions}\n\n"
        "Return exactly one JSON object with fields segment_id, condition, answers, overall_notes. "
        "Each answers.Q is an object with label (YES/NO/UNKNOWN), confidence (0..1), and concise "
        "evidence tied to visible frames."
    )
    if metadata is not None:
        prompt += "\n\nHistorical action metadata (context only; not visual evidence):\n" + json.dumps(metadata, ensure_ascii=False, indent=2)
    return prompt


def _invoke_claude(
    *,
    prompt: str,
    schema: Mapping[str, Any],
    bundle_dir: Path,
    binary: str,
    timeout_s: int,
) -> dict[str, Any]:
    resolved = shutil.which(binary) if Path(binary).name == binary else binary
    if resolved is None:
        raise AblationError(f"Claude CLI not found: {binary}")
    command = [
        resolved,
        "--print",
        prompt,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema, separators=(",", ":")),
        "--permission-mode",
        "plan",
        "--allowedTools",
        "Read",
        "--tools",
        "Read",
        "--add-dir",
        str(bundle_dir),
        "--safe-mode",
        "--no-session-persistence",
        "--system-prompt",
        (
            "You are a read-only visual evaluator. Use the Read tool only. Do not execute shell "
            "commands, call robot APIs, inspect unrelated directories, or write files. Return only "
            "the requested JSON object."
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=bundle_dir,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise AblationError(f"Claude exited with {completed.returncode}: {detail}")
    return _json_from_cli(completed.stdout)


def _condition_result_path(output_dir: Path, segment_id: str, condition: str) -> Path:
    return output_dir / "responses" / segment_id / f"condition_{condition}.json"


def _write_prompt(path: Path, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")


def _summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(records)
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    confidence: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    valid = Counter()
    prepared = Counter()
    errors = Counter()
    total = Counter()
    for record in materialized:
        condition = str(record.get("condition", "?"))
        total[condition] += 1
        if record.get("status") == "dry_run":
            prepared[condition] += 1
            continue
        if record.get("status") != "completed":
            errors[condition] += 1
            continue
        valid[condition] += 1
        answers = record.get("response", {}).get("answers", {}) if isinstance(record.get("response"), Mapping) else {}
        if not isinstance(answers, Mapping):
            continue
        for question in QUESTIONS:
            value = answers.get(question["id"])
            if not isinstance(value, Mapping):
                continue
            label = value.get("label")
            if label in LABELS:
                counts[condition][question["id"]][str(label)] += 1
            try:
                confidence[condition][question["id"]].append(float(value.get("confidence")))
            except (TypeError, ValueError):
                pass
    by_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        by_condition[condition] = {
            "total": total[condition],
            "prepared": prepared[condition],
            "completed": valid[condition],
            "errors": errors[condition],
            "questions": {
                question["id"]: {
                    "labels": dict(counts[condition][question["id"]]),
                    "mean_confidence": (
                        sum(confidence[condition][question["id"]]) / len(confidence[condition][question["id"]])
                        if confidence[condition][question["id"]]
                        else None
                    ),
                }
                for question in QUESTIONS
            },
        }
    by_segment: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in materialized:
        if record.get("status") == "completed" and isinstance(record.get("response"), Mapping):
            by_segment[str(record.get("segment_id"))][str(record.get("condition"))] = record["response"]
    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(CONDITIONS):
        for right in CONDITIONS[left_index + 1 :]:
            pair_key = f"{left}_vs_{right}"
            pairwise[pair_key] = {}
            for question in QUESTIONS:
                transitions: Counter[str] = Counter()
                for segment in by_segment.values():
                    left_response = segment.get(left)
                    right_response = segment.get(right)
                    if not isinstance(left_response, Mapping) or not isinstance(right_response, Mapping):
                        continue
                    left_answer = left_response.get("answers", {}).get(question["id"], {})
                    right_answer = right_response.get("answers", {}).get(question["id"], {})
                    if isinstance(left_answer, Mapping) and isinstance(right_answer, Mapping):
                        a, b = left_answer.get("label"), right_answer.get("label")
                        if a in LABELS and b in LABELS:
                            transitions[f"{a}->{b}"] += 1
                pairwise[pair_key][question["id"]] = {
                    "label_transitions": dict(transitions),
                    "unknown_to_known": sum(
                        count
                        for transition, count in transitions.items()
                        if transition.startswith("UNKNOWN->") and not transition.endswith("->UNKNOWN")
                    ),
                }
    return {"conditions": by_condition, "pairwise": pairwise}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", action="append", type=Path, default=None, help="run root or recording directory; repeatable (defaults to runs when --recording is omitted)")
    parser.add_argument("--recording", action="append", type=Path, default=[], help="explicit recording directory or recording_manifest.json; repeatable")
    parser.add_argument("--limit", type=int, default=50, help="number of segments to sample; 0 means all")
    parser.add_argument("--seed", type=int, default=20260828, help="deterministic segment sampling seed")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/claude_condition_ablation"))
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--claude-timeout-s", type=int, default=900)
    parser.add_argument("--resume", action="store_true", help="reuse valid condition response files")
    parser.add_argument("--dry-run", action="store_true", help="prepare evidence bundles and prompts without Claude calls")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.claude_timeout_s <= 0:
        raise SystemExit("--claude-timeout-s must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = list(args.runs_root or ([] if args.recording else [Path("runs")]))
    inputs = [path.expanduser().resolve() for path in [*roots, *args.recording]]
    manifests = _find_recording_manifests(inputs)
    segments = [segment for path in manifests if (segment := _segment_from_manifest(path)) is not None]
    segments = _sample_segments(segments, limit=args.limit, seed=args.seed)
    if not segments:
        raise SystemExit("no recording manifests with camera_A_rgb.mp4 were found")

    selected_conditions = tuple(dict.fromkeys(args.conditions))
    run_index = {
        "created_at": _now(),
        "project_root": str(PROJECT_ROOT),
        "inputs": [str(path) for path in inputs],
        "manifest_count": len(manifests),
        "segment_count": len(segments),
        "limit": args.limit,
        "seed": args.seed,
        "conditions": list(selected_conditions),
        "question_count": len(QUESTIONS),
        "questions": list(QUESTIONS),
        "segment_ids": [segment.segment_id for segment in segments],
    }
    _write_json(output_dir / "experiment_index.json", run_index)

    ledger_path = output_dir / "results.jsonl"
    records: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, start=1):
        bundle_dir = output_dir / "bundles" / segment.segment_id
        try:
            bundle = _prepare_bundle(segment, bundle_dir)
        except Exception as exc:
            failure = {"segment_id": segment.segment_id, "status": "bundle_error", "error": f"{type(exc).__name__}: {exc}"}
            print(f"[{index}/{len(segments)}] {segment.segment_id}: bundle error: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise
            for condition in selected_conditions:
                records.append({**failure, "condition": condition})
            continue
        print(f"[{index}/{len(segments)}] {segment.segment_id}: bundle ready", flush=True)
        for condition in selected_conditions:
            result_path = _condition_result_path(output_dir, segment.segment_id, condition)
            prompt_path = output_dir / "prompts" / segment.segment_id / f"condition_{condition}.md"
            if args.resume and result_path.is_file():
                cached = _read_json(result_path)
                if isinstance(cached, Mapping) and cached.get("status") == "completed":
                    records.append(dict(cached))
                    print(f"  condition {condition}: reused", flush=True)
                    continue
            prompt = _build_prompt(segment, bundle, condition)
            _write_prompt(prompt_path, prompt)
            record: dict[str, Any] = {
                "created_at": _now(),
                "segment_id": segment.segment_id,
                "condition": condition,
                "status": "dry_run" if args.dry_run else "running",
                "bundle_dir": str(bundle_dir),
                "prompt_path": str(prompt_path),
                "assets": [str(path) for path in _condition_assets(bundle, condition)],
            }
            if args.dry_run:
                _write_json(result_path, record)
                records.append(record)
                continue
            try:
                raw = _invoke_claude(
                    prompt=prompt,
                    schema=_schema(),
                    bundle_dir=bundle_dir,
                    binary=args.claude_binary,
                    timeout_s=args.claude_timeout_s,
                )
                response = _validate_response(raw, segment_id=segment.segment_id, condition=condition)
                record.update({"status": "completed", "response": response})
                print(f"  condition {condition}: completed", flush=True)
            except Exception as exc:
                record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                print(f"  condition {condition}: error: {exc}", file=sys.stderr, flush=True)
                if args.fail_fast:
                    _write_json(result_path, record)
                    raise
            _write_json(result_path, record)
            records.append(record)

    with ledger_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    summary = {
        "created_at": _now(),
        "experiment_index": str(output_dir / "experiment_index.json"),
        "segment_count": len(segments),
        "condition_count": len(selected_conditions),
        "call_count": len(segments) * len(selected_conditions),
        "dry_run": bool(args.dry_run),
        **_summarize(records),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(output_dir), "segments": len(segments), "conditions": list(selected_conditions), "summary": str(output_dir / "summary.json")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
