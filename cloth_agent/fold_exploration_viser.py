"""Read-only Viser dashboard for the video-backed folding exploration run.

The folding CLI writes artifacts incrementally.  This viewer follows the
run's output directory and keeps every image from every iteration visible:
raw before/after captures, perception overlays, rollout contact sheets, and
any debug overlays produced by a later version of the pipeline.  It never
opens a camera, connects to the robot, or invokes Claude.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import hashlib
import html
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_IMAGE_PATH_RE = re.compile(
    r"(?P<path>/[^\s\]\)\}\"']+\.(?:png|jpg|jpeg|webp|bmp))",
    re.IGNORECASE,
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _image(path: Path) -> np.ndarray | None:
    try:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError):
        return None


def _short(value: Any, limit: int = 1800) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _iteration_dirs(source: Path) -> list[Path]:
    if (source / "claude_image_tools").is_dir():
        return [source]
    return sorted(
        (path for path in source.glob("iteration_*") if path.is_dir()),
        key=lambda path: path.name,
    )


def _run_root(source: Path) -> Path:
    summary = _load_json(source / "summary.json")
    configured = summary.get("run_dir")
    if isinstance(configured, str) and configured.strip():
        path = Path(configured).expanduser().resolve()
        if path.is_dir():
            return path
    # Expected layout: <run>/results/fold_exploration/<timestamp>.
    if len(source.parents) >= 3:
        return source.parents[2]
    return source


def _resolve_run_path(value: Any, run_root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = run_root / path
    try:
        return path.resolve()
    except OSError:
        return None


def _unique_existing_images(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if (
            resolved in seen
            or not resolved.is_file()
            or resolved.suffix.lower() not in IMAGE_SUFFIXES
        ):
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def _unique_images_by_content(paths: list[Path]) -> list[Path]:
    result, seen = [], set()
    for path in _unique_existing_images(paths):
        try:
            key = (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _image_paths_from_text(text: Any) -> list[Path]:
    if not isinstance(text, str):
        return []
    return _unique_existing_images(
        [Path(match.group("path")) for match in _IMAGE_PATH_RE.finditer(text)]
    )


def _prompt_from_payload(payload: Mapping[str, Any]) -> str:
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        return prompt
    command = payload.get("command")
    if isinstance(command, list) and "--print" in command:
        index = command.index("--print") + 1
        if index < len(command) and isinstance(command[index], str):
            return command[index]
    return ""


def _planning_images_from_prompt(text: Any) -> list[Path]:
    if not isinstance(text, str):
        return []
    marker = "Garment images to inspect:\n"
    if marker not in text:
        return _image_paths_from_text(text)
    section = text.split(marker, 1)[1]
    section = section.split("\n\nWhen the canonical upright Camera-A", 1)[0]
    return _image_paths_from_text(section)


def _supervisor_input_images(
    iteration_dir: Path,
    stage: str,
    run_root: Path,
) -> list[Path]:
    payload = _load_json(iteration_dir / f"{stage}.json")
    if "evidence_images" in payload:
        return _unique_existing_images([path for value in payload["evidence_images"]
            if (path := _resolve_run_path(value, run_root)) is not None])
    bundle = payload.get("context_bundle")
    if not isinstance(bundle, Mapping):
        return _image_paths_from_text(_prompt_from_payload(payload))
    read_order = bundle.get("read_order")
    if not isinstance(read_order, list):
        return []
    evidence_path = next(
        (
            _resolve_run_path(item, run_root)
            for item in read_order
            if isinstance(item, str) and item.endswith("04_evidence_manifest.json")
        ),
        None,
    )
    if evidence_path is None or not evidence_path.is_file():
        return []
    evidence = _load_json(evidence_path)
    values: list[Any] = []
    for key in ("images", "rollout_video_contact_sheets"):
        items = evidence.get(key)
        if isinstance(items, list):
            values.extend(items)
    paths = [
        path
        for value in values
        if (path := _resolve_run_path(value, run_root)) is not None
    ]
    return _unique_existing_images(paths)


def _claude_input_groups(iteration_dir: Path, run_root: Path) -> dict[str, list[Path]]:
    """Return the exact raster inputs supplied to each Claude visual stage."""

    groups: dict[str, list[Path]] = {}
    planning = _load_json(iteration_dir / "planning_diagnostics.json")
    visual = planning.get("visual_plan_result") if isinstance(planning, Mapping) else None
    if isinstance(visual, Mapping):
        paths = _planning_images_from_prompt(_prompt_from_payload(visual))
        if paths:
            groups["planning_stage1"] = paths
    for stage in ("supervisor_before", "supervisor_after"):
        paths = _supervisor_input_images(iteration_dir, stage, run_root)
        if paths:
            groups[stage] = paths
    evaluation = _load_json(iteration_dir / "claude_evaluation_result.json")
    paths = (_unique_existing_images([path for value in evaluation["evidence_images"]
             if (path := _resolve_run_path(value, run_root)) is not None])
             if "evidence_images" in evaluation else _image_paths_from_text(_prompt_from_payload(evaluation)))
    if paths:
        groups["evaluation"] = paths
    for manifest in sorted(iteration_dir.glob("planning_attempt_*/*/*_invocation.json")):
        data = _load_json(manifest)
        paths = _unique_existing_images([path for value in data.get("evidence_images", [])
                 if (path := _resolve_run_path(value, run_root)) is not None])
        if paths:
            groups[f"{manifest.parent.parent.name}/{data.get('stage', manifest.stem)}"] = paths
    return groups


def _iter_images(iteration_dir: Path) -> list[Path]:
    """Return every saved raster artifact, in a useful stage order."""

    paths = [
        path.resolve()
        for path in iteration_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        and "claude_image_tools" not in path.relative_to(iteration_dir).parts
    ]

    def order(path: Path) -> tuple[int, str]:
        text = str(path.relative_to(iteration_dir)).lower()
        if "workspace_" in path.name:
            rank = -1
        elif "before_raw" in text:
            rank = 0
        elif "trajectory" in text or "proposal" in text:
            rank = 1
        elif "rollout_recording" in text or "hold_check" in text:
            rank = 2
        elif "after_raw" in text:
            rank = 3
        else:
            rank = 4
        return rank, text

    # Prefer diagnostic artifacts when two files are byte-identical (tests and
    # staged copies can share placeholder bytes); otherwise keep the first
    # useful stage ordering and suppress duplicate copies.
    ordered = sorted(paths, key=order)
    return _unique_images_by_content(ordered)


def _debug_markdown(source: Path) -> str:
    """Compact live phase and timing table, backed by structured events."""
    events = []
    try:
        for line in (source / "debug_events.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Writer may be halfway through its last line.
    except OSError:
        return "Waiting for structured debug events."
    if not events:
        return "Waiting for structured debug events."
    last = events[-1]
    try:
        age = max(0., (datetime.now(timezone.utc) - datetime.fromisoformat(last["timestamp"].replace("Z", "+00:00"))).total_seconds())
    except (KeyError, ValueError, TypeError):
        age = 0.
    lines = [f"Current / last event: **{last.get('stage')} — {_short(last.get('message'), 300)}**",
             f"\nRun elapsed at last event: {last.get('elapsed_s', 0):.1f} s · last update {age:.1f} s ago",
             "\nSSH includes downloads and Claude. Transfer timings and public Claude messages stream as available; message gaps are not pure reasoning time.",
             "\nRecent timed stages (nested durations overlap; do not sum rows):\n",
             "| Run time | Stage | Event | Duration |", "| ---: | --- | --- | ---: |"]
    timed = [e for e in events if isinstance(e.get("fields", {}).get("duration_s"), (float, int))]
    for e in timed[-18:]:
        fields = e["fields"]
        lines.append(f"| {e['elapsed_s']:.1f}s | {e['stage']} | {_short(e['message'], 90)} | {fields['duration_s']:.3f}s |")
    failures = [e for e in events if e.get("fields", {}).get("exception_type")]
    if failures:
        lines += ["\nLatest error:\n", _short(failures[-1]["message"], 1400)]
    return "\n".join(lines)


def _error_html(source: Path) -> str:
    """Keep failures visible even when subsequent recovery messages arrive."""
    events = []
    path = source / "debug_events.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            fields = event.get("fields", {})
            if event.get("level") == "ERROR" or fields.get("exception_type") or fields.get("error"):
                events.append(event)
    if not events:
        return '<div style="color:#2b8a3e">No errors recorded.</div>'
    rows = []
    for event in events[-6:]:
        fields = event.get("fields", {})
        detail = f"+{event.get('elapsed_s', 0):.1f}s · {event.get('stage')} · {event.get('message')}"
        recovery = fields.get("recovery_status") or fields.get("operation") or "See current stage / recovery record"
        rows.append('<div style="margin-bottom:10px"><strong>' + html.escape(detail) +
                    '</strong><br>' + html.escape(str(recovery)) + '</div>')
    return '<div role="alert" style="color:#ff6b6b;border-left:4px solid #e03131;padding:10px;overflow-wrap:anywhere">' + ''.join(rows) + '</div>'


def _workspace_markdown(iteration_dir: Path) -> str:
    lines = []
    for path in sorted(iteration_dir.glob("planning_attempt_*/*/workspace_diagnostics.json")):
        data = _load_json(path)
        lines.extend([f"\nWorkspace: **{data.get('status', 'UNKNOWN')}**", f"\n`{path.relative_to(iteration_dir)}`\n"])
        if data.get("render_error"):
            lines.append(f"Image generation failed: {data['render_error']}")
        if data.get("plan_error"):
            lines.append(f"\n{data['plan_error']}")
        for p in data.get("moves", []):
            if p.get("error"):
                lines.extend([f"Rejected action **#{p['action_index']}** ({p['target']})",
                    f"\nUpright pixel `{p['upright_pixel_xy']}` → base XYZ `{p['base_xyz_mm']}` mm.",
                    f"\n{p['error']}"])
                if p.get("lateral"):
                    lines.append(f"\nSigned side clearances: `{p['lateral']['signed_clearance_mm']}` mm; negative means outside.")
    return "\n".join(lines)


def _trajectory_points(iteration_dir: Path) -> np.ndarray:
    payload = _load_json(iteration_dir / "trajectory.json")
    actions = payload.get("actions")
    if not isinstance(actions, list):
        return np.empty((0, 3), dtype=np.float32)
    points: list[list[float]] = []
    for action in actions:
        if not isinstance(action, dict) or action.get("name") != "move":
            continue
        args = action.get("args")
        if not isinstance(args, dict):
            continue
        try:
            points.append([float(args["x"]) / 1000.0, float(args["y"]) / 1000.0, float(args["z"]) / 1000.0])
        except (KeyError, TypeError, ValueError):
            continue
    return np.asarray(points, dtype=np.float32)


def _markdown_for_iteration(iteration_dir: Path) -> str:
    record = _load_json(iteration_dir / "record.json")
    before = _load_json(iteration_dir / "supervisor_before.json")
    after = _load_json(iteration_dir / "supervisor_after.json")
    evaluation = _load_json(iteration_dir / "evaluation.json")
    trajectory = _load_json(iteration_dir / "trajectory.json")
    screen_before = record.get("screen_before", {}) if record else {}
    screen_after = record.get("screen_after", {}) if record else {}
    proposal = record.get("proposal", {}) if record else {}
    claude_groups = _claude_input_groups(
        iteration_dir,
        _run_root(iteration_dir.parent),
    )
    lines = [f"### {iteration_dir.name}", ""]
    failure = _load_json(iteration_dir / "failure_detection.json")
    recovery = _load_json(iteration_dir / "recovery.json")
    if failure:
        lines.extend([f"Failure detection: **{failure.get('category', 'UNKNOWN')}**",
                      f"\nRelease/Home confirmed: `{failure.get('safe_return_confirmed')}`",
                      f"\nRecovery: `{recovery.get('status', 'See current stage')}`",
                      f"\nExperience: `{'SAVED' if record.get('record_id') else 'PENDING'}`\n"])
    if record.get("skill_review"):
        lines.append(f"Skill candidate: `{record['skill_review'].get('status')}`\n")
    lines.append(_workspace_markdown(iteration_dir))
    lines.append(f"- mode: `{record.get('status', trajectory.get('mode', 'RUNNING'))}`")
    lines.append(f"- images displayed: `{len(_iter_images(iteration_dir))}`")
    lines.append(f"- trajectory actions: `{len(trajectory.get('actions', []))}`")
    lines.append(
        f"- Claude input groups/images: `{len(claude_groups)}` / "
        f"`{sum(len(paths) for paths in claude_groups.values())}`"
    )
    lines.append(f"- before visibility: `{screen_before.get('visibility', 'unknown')}`")
    lines.append(f"- after visibility: `{screen_after.get('visibility', 'unknown')}`")
    lines.append(
        f"- supervisor before: `{before.get('current_step', 'unknown')}` / `{before.get('trajectory_decision', 'unknown')}`"
    )
    lines.append(
        f"- supervisor after: `{after.get('current_step', 'unknown')}` / `{after.get('trajectory_decision', 'unknown')}`"
    )
    task_progress = evaluation.get("task_progress", {}) if isinstance(evaluation, dict) else {}
    if isinstance(task_progress, dict):
        lines.append(f"- evaluation progress: `{task_progress.get('status', 'unknown')}`")
    if proposal:
        lines.extend(["", "**Claude proposal / observation**", "", _short(proposal.get("garment_observation")), "", _short(proposal.get("reveal_strategy"))])
    if before.get("reason"):
        lines.extend(["", "**Supervisor before reason**", "", _short(before.get("reason"), 900)])
    if after.get("reason"):
        lines.extend(["", "**Supervisor after reason**", "", _short(after.get("reason"), 900)])
    if evaluation.get("reason"):
        lines.extend(["", "**Evaluation reason**", "", _short(evaluation.get("reason"), 1200)])
    selection = _load_json(iteration_dir / 'claude_molmo_orientation' / 'selection.json')
    if selection:
        handoff = _load_json(iteration_dir / 'molmo_handoff.json')
        hint = _load_json(iteration_dir / 'molmo_sleeve_hint.json')
        lines.extend(['', '**Claude → Molmo → Claude**', '',
            f"- Orientation: `{selection.get('status')}` / collar UP, hem DOWN",
            f"- Claude selected: `{selection.get('image_id')}`",
            f"- Molmo actual RGB input: `{handoff.get('input_image', 'not handed off')}`",
            f"- RGB digest: `{handoff.get('input_rgb_sha256', 'n/a')}`",
            f"- Sleeve step: `{handoff.get('step', 'n/a')}` / hint: `{hint.get('status', 'pending')}`",
            f"- Point in processed RGB: `{hint.get('processed_pixel_xy')}`",
            f"- Point in fixed camera display: `{hint.get('upright_pixel_xy')}`",
            f"- Point in original Cam A: `{hint.get('raw_pixel_xy')}`",
            '', _short(selection.get('error') or selection.get('reason'), 1200)])
    files = [
        "supervisor_before.json",
        "planning_diagnostics.json",
        "trajectory.json",
        "execution.json",
        "recording.json",
        "claude_evaluation_result.json",
        "evaluation.json",
        "supervisor_after.json",
        "record.json",
        "failure_detection.json",
        "recovery.json",
        "skill_review.json",
        "video_archive.json",
        "perception_artifacts.json",
        "claude_molmo_orientation/selection.json",
        "molmo_handoff.json",
        "molmo_sleeve_locator/pixel_mapping.json",
    ]
    present = [name for name in files if (iteration_dir / name).is_file()]
    if present:
        lines.extend(["", "**Intermediate JSON artifacts**", "", *[f"- `{name}`" for name in present]])
    return "\n".join(lines)


def _image_tool_summary(data):
    lines = [f"**{data.get('stage', 'Claude')} | {data.get('status', 'RUNNING')}**",
        f"Elapsed: {data.get('elapsed_s', 0):.1f}s | "
        f"Audit: {'complete' if data.get('audit_complete') else 'in progress / incomplete'}",
        "READ_COMPLETED means the Read tool succeeded; it does not prove visual understanding."]
    last_message = data.get('last_claude_event')
    if last_message:
        lines.append(f"Claude public messages: {data.get('claude_event_count', 0)}; "
                     f"last: {last_message.get('type')} at +{last_message.get('received_elapsed_s', 0):.1f}s. "
                     'See claude_transcript.md and timing.md below. Hidden reasoning is not available.')
    if not any(e.get("tool") in {"rotate_image", "crop_image", "resize_image"} for e in data.get("events", [])):
        lines.append("No image transformation calls recorded so far.")
    checks = [e for e in data.get('events', []) if e.get('kind') == 'orientation_guard']
    if checks:
        requested = sum(e.get('status') == 'requested' for e in checks)
        lines.append(f"Orientation correction: {requested}/1 in the same session; "
                     f"last audit: {checks[-1].get('classification')}. Edit/turn/time budgets are shared.")
    if data.get("error"):
        lines.append(f"Error: {data['error']}")
    lines.extend(str(e) for e in data.get("errors", []))
    for i, event in enumerate(data.get("events", []), 1):
        budget = event.get('edit_budget')
        if budget is not None:
            lines.append(f"Edit budget: {budget['used']}/{budget['limit']} used; {budget['remaining']} remaining")
        if event.get("kind") == "session":
            continue
        duration = event.get("duration_s")
        suffix = f" | {duration:.3f}s" if isinstance(duration, (int, float)) else ""
        lines.append(f"{i}. +{event.get('received_elapsed_s', 0):.1f}s | {event.get('tool')} | {event.get('status')}{suffix} | "
                     f"{json.dumps(event.get('arguments', {}), ensure_ascii=False)}")
        if event.get("tool") == "map_point":
            lines.append(f"   Original pixel: {json.dumps(event.get('result', {}))}")
        if event.get("error"):
            lines.append(f"   Error: {event['error']}")
    return "\n\n".join(lines)


class _FoldViserState:
    def __init__(self, server: Any, source: Path):
        self.server = server
        self.source = source
        self.run_root = _run_root(source)
        self.lock = threading.Lock()
        self.image_handles: dict[Path, tuple[int, Any]] = {}
        self.claude_image_handles: dict[
            tuple[Path, Path], tuple[int, tuple[str, ...], Any]
        ] = {}
        self.claude_iteration_panels: dict[Path, Any] = {}
        self.iteration_panels: dict[Path, Any] = {}
        self.path_handles: dict[Path, Any] = {}
        self.path_mtimes: dict[Path, int] = {}
        self.image_folders: dict[tuple[Path, str], Any] = {}
        self.tool_panels = {}
        self.tool_image_handles = {}
        self.tool_image_panels = {}
        self.tool_raw_panels = {}
        self.tool_raw_folders = {}
        self.tool_raw_mtimes = {}
        self.visible_iterations = set()
        self.status = server.gui.add_markdown(
            f"### Folding exploration dashboard\n\nFollowing `{source}`. Waiting for iteration artifacts."
        )
        self.timing_panel = server.gui.add_markdown("Waiting for timing events.")
        self.error_panel = server.gui.add_html(_error_html(source))
        with server.gui.add_folder("Raw debug log", expand_by_default=False):
            self.debug_panel = server.gui.add_markdown("Waiting for debug.log.")

    def _folder(self, iteration_dir, group):
        key = (iteration_dir, group)
        if key not in self.image_folders:
            self.image_folders[key] = self.server.gui.add_folder(
                f"{iteration_dir.name} | {group}", expand_by_default=group == "Workspace targets")
        return self.image_folders[key]

    def _relative_to_run(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.run_root))
        except (OSError, ValueError):
            return str(path)

    def _render_claude_image(
        self,
        path: Path,
        iteration_dir: Path,
        stages: list[str],
    ) -> None:
        key = (iteration_dir, path)
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            return
        stage_key = tuple(stages)
        current = self.claude_image_handles.get(key)
        if current is not None and current[0] == mtime and current[1] == stage_key:
            return
        if current is not None:
            try:
                current[2].remove()
            except Exception:
                pass
        image = _image(path)
        if image is None:
            return
        label = (
            f"CLAUDE INPUT | {iteration_dir.name} | {','.join(stages)} | "
            f"{self._relative_to_run(path)}"
        )
        with self._folder(iteration_dir, "Actual Claude RGB inputs"):
            self.claude_image_handles[key] = (mtime, stage_key, self.server.gui.add_image(image, label=label))

    def _render_image(self, path: Path, iteration_dir: Path) -> None:
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            return
        current = self.image_handles.get(path)
        if current is not None and current[0] == mtime:
            return
        if current is not None:
            try:
                current[1].remove()
            except Exception:
                pass
        image = _image(path)
        if image is None:
            return
        relative = path.relative_to(iteration_dir)
        is_static_reference = "flat_reference" in relative.parts
        label = (f"{iteration_dir.name} | STATIC REFERENCE (historical topology; not current camera) | {relative}"
                 if is_static_reference else f"{iteration_dir.name} | {relative}")
        group = ("Static Molmo reference (not current observation)" if is_static_reference else
                 "Workspace targets" if path.name.startswith("workspace_") else
                 "After observation" if "after_raw" in str(relative) else
                 "Before observation" if "before_raw" in str(relative) else
                 "Rollout" if "rollout" in str(relative) else "Other diagnostics")
        with self._folder(iteration_dir, group):
            self.image_handles[path] = (mtime, self.server.gui.add_image(image, label=label))

    def _render_path(self, iteration_dir: Path) -> None:
        trajectory_path = iteration_dir / "trajectory.json"
        try:
            mtime = trajectory_path.stat().st_mtime_ns
        except OSError:
            mtime = -1
        if iteration_dir in self.path_mtimes and self.path_mtimes[iteration_dir] == mtime:
            return
        self.path_mtimes[iteration_dir] = mtime
        points = _trajectory_points(iteration_dir)
        old = self.path_handles.get(iteration_dir)
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass
            self.path_handles.pop(iteration_dir, None)
        if len(points) < 2:
            return
        # Keep each iteration's trajectory in a separate scene node.  Colors
        # cycle so adjacent iterations remain visually distinguishable.
        palette = np.asarray(
            [[230, 45, 45], [40, 130, 255], [40, 180, 80], [220, 150, 30], [170, 70, 210]],
            dtype=np.uint8,
        )
        index = max(0, int(iteration_dir.name.rsplit("_", 1)[-1]) - 1) if "_" in iteration_dir.name else 0
        colors = np.tile(palette[index % len(palette)], (len(points) - 1, 2, 1))
        segments = np.stack([points[:-1], points[1:]], axis=1)
        self.path_handles[iteration_dir] = self.server.scene.add_line_segments(
            f"/trajectories/{iteration_dir.name}",
            points=segments,
            colors=colors,
            line_width=4.0,
        )

    def _render_image_tools(self, iteration_dir):
        for manifest in sorted(iteration_dir.glob("claude_image_tools/*/image_debug.json")):
            data = _load_json(manifest)
            if not data:
                continue
            folder = self._folder(iteration_dir, "Claude image operations / " + manifest.parent.name)
            with folder:
                summary = _image_tool_summary(data)
                if manifest not in self.tool_panels:
                    self.tool_panels[manifest] = self.server.gui.add_markdown(summary)
                else:
                    self.tool_panels[manifest].content = summary
                for index, view in enumerate(data.get("views", [])):
                    path = Path(view.get("path", "")).resolve()
                    if manifest.parent not in path.parents or not path.is_file():
                        continue
                    key = (manifest, view["image_id"])
                    info = (f"**{index:02d} | {view.get('operation', 'Original RGB')} | {view['image_id']}**\n\n"
                        f"{view.get('verification')} | {view.get('read_status')}\n\n"
                        f"Original: image_{view.get('original_image_index')} | "
                        f"Parent: {view.get('parent_image_id')} | Size: {view.get('size')}\n\n"
                        f"Operation duration: {view.get('duration_s', 'n/a')} s | "
                        f"Read durations: {[r.get('duration_s') for r in view.get('reads', [])]} s\n\n"
                        f"Parameters: `{json.dumps(view.get('arguments', {}))}`\n\n"
                        f"{self._relative_to_run(path)}")
                    if key not in self.tool_image_panels:
                        self.tool_image_panels[key] = self.server.gui.add_markdown(info)
                    else:
                        self.tool_image_panels[key].content = info
                    if key not in self.tool_image_handles:
                        pixels = _image(path)
                        if pixels is not None:
                            self.tool_image_handles[key] = self.server.gui.add_image(pixels,
                                label=f"{index:02d} {view.get('operation', 'Original')} | {view['image_id']}")
                for overlay in data.get("point_overlays", []):
                    path = Path(overlay.get("path", "")).resolve()
                    if manifest.parent not in path.parents or not path.is_file():
                        continue
                    key = (manifest, str(path))
                    if key not in self.tool_image_handles:
                        pixels = _image(path)
                        if pixels is not None:
                            label = (f"DEBUG ONLY / NOT SENT | event {overlay['event_sequence']} | "
                                f"{overlay['label']} | {overlay['source_image_id']} | "
                                f"{overlay['pixel_xy']} | source {overlay['source_verification']}")
                            self.tool_image_handles[key] = self.server.gui.add_image(pixels, label=label)
                # Complete request, tool results/matrices/hashes, stdout and
                # stderr remain expandable; do not truncate diagnostic files.
                for name in ("prompt.txt", "system_prompt.txt", "claude_transcript.md", "timing.md",
                             "timing.json", "claude_result.json", "claude_events.jsonl",
                             "request.json", "image_debug.json", "stdout.log", "stderr.log", "exception.log"):
                    path = manifest.parent / name
                    if not path.exists():
                        continue
                    key = (manifest, name)
                    try:
                        mtime = path.stat().st_mtime_ns
                        if self.tool_raw_mtimes.get(key) == mtime:
                            continue
                        content = path.read_text(encoding="utf-8")
                    except OSError:
                        continue
                    self.tool_raw_mtimes[key] = mtime
                    fence = "`" * max(4, 1 + max((len(m.group()) for m in re.finditer(r"`+", content)), default=0))
                    markdown = fence + "text\n" + content + "\n" + fence
                    if name.endswith('.md'):
                        markdown = content
                    if key not in self.tool_raw_panels:
                        self.tool_raw_folders[key] = self.server.gui.add_folder(name, expand_by_default=False)
                        with self.tool_raw_folders[key]:
                            self.tool_raw_panels[key] = self.server.gui.add_markdown(markdown)
                    else:
                        self.tool_raw_panels[key].content = markdown

    def _evict_iteration(self, iteration):
        """Release GUI/scene handles only; saved run artifacts stay on disk."""
        def belongs(key):
            paths = key if isinstance(key, tuple) else (key,)
            return any(isinstance(p, Path) and (p == iteration or iteration in p.parents) for p in paths)
        for name in ('image_handles', 'claude_image_handles', 'claude_iteration_panels',
                     'iteration_panels', 'path_handles', 'tool_panels', 'tool_image_handles',
                     'tool_image_panels', 'tool_raw_panels', 'tool_raw_folders', 'image_folders'):
            handles = getattr(self, name)
            for key in list(handles):
                if not belongs(key):
                    continue
                value = handles.pop(key)
                handle = value[-1] if isinstance(value, tuple) else value
                try:
                    handle.remove()
                except Exception:
                    pass
        for name in ('path_mtimes', 'tool_raw_mtimes'):
            mapping = getattr(self, name)
            for key in list(mapping):
                if belongs(key):
                    mapping.pop(key)

    def update(self) -> None:
        all_iteration_dirs = _iteration_dirs(self.source)
        iteration_dirs = all_iteration_dirs[-2:]
        for old in self.visible_iterations - set(iteration_dirs):
            self._evict_iteration(old)
        self.visible_iterations = set(iteration_dirs)
        displayed = 0
        claude_group_count = 0
        for iteration_dir in iteration_dirs:
            self._render_image_tools(iteration_dir)
            claude_groups = _claude_input_groups(iteration_dir, self.run_root)
            claude_groups = {stage: _unique_images_by_content(paths)
                             for stage, paths in claude_groups.items()}
            claude_group_count += len(claude_groups)
            path_stages: dict[Path, list[str]] = {}
            for stage, paths in claude_groups.items():
                for image_path in paths:
                    path_stages.setdefault(image_path, []).append(stage)
            if claude_groups:
                panel = self.claude_iteration_panels.get(iteration_dir)
                stage_lines = "\n".join(
                    f"- `{stage}`: `{len(paths)}` images"
                    for stage, paths in claude_groups.items()
                )
                content = (
                    f"### Claude Inputs | {iteration_dir.name}\n\n"
                    f"Unique raster files actually supplied: `{len(path_stages)}`. "
                    "If a file was reused, its image label lists every Claude stage.\n\n"
                    f"{stage_lines}"
                )
                if panel is None:
                    panel = self.server.gui.add_markdown(content)
                    self.claude_iteration_panels[iteration_dir] = panel
                else:
                    panel.content = content
                for image_path, stages in path_stages.items():
                    self._render_claude_image(image_path, iteration_dir, stages)
            claude_paths = set(path_stages)
            for image_path in _iter_images(iteration_dir):
                if image_path in claude_paths:
                    continue
                self._render_image(image_path, iteration_dir)
                displayed += 1
            self._render_path(iteration_dir)
            panel = self.iteration_panels.get(iteration_dir)
            if panel is None:
                panel = self.server.gui.add_markdown(_markdown_for_iteration(iteration_dir))
                self.iteration_panels[iteration_dir] = panel
            else:
                panel.content = _markdown_for_iteration(iteration_dir)
        summary = _load_json(self.source / "summary.json")
        self.timing_panel.content = _debug_markdown(self.source)
        self.error_panel.content = _error_html(self.source)
        debug_path = self.source / "debug.log"
        if debug_path.is_file():
            try:
                debug_tail = "\n".join(debug_path.read_text(encoding="utf-8").splitlines()[-40:])
            except OSError as exc:
                debug_tail = f"debug.log read failed: {type(exc).__name__}: {exc}"
        else:
            debug_tail = "Waiting for debug.log."
        self.status.content = (
            "### Folding exploration dashboard\n\n"
            f"- source: `{self.source}`\n"
            f"- iterations discovered: `{len(all_iteration_dirs)}`; showing latest `{len(iteration_dirs)}` (maximum 2)\n"
            f"- raster artifacts displayed: `{len(self.image_handles)}`\n"
            f"- Claude input images displayed: `{len(self.claude_image_handles)}`\n"
            f"- Claude input groups: `{claude_group_count}`\n"
            f"- Claude image-operation views/point overlays: `{len(self.tool_image_handles)}`\n"
            f"- trajectory overlays: `{len(self.path_handles)}`\n"
            f"- run status: `{summary.get('status', 'RUNNING')}`\n"
            "\nThe viewer is read-only; it does not control the robot."
        )
        self.debug_panel.content = "### Debug tail\n\n```text\n" + _short(debug_tail, 12000) + "\n```"


def run_viewer(source: Path, *, host: str = "127.0.0.1", port: int = 8765, refresh_s: float = 0.5, open_browser: bool = False) -> int:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise PermissionError("fold exploration Viser must bind to loopback")
    if not 0.1 <= float(refresh_s) <= 30.0:
        raise ValueError("refresh_s must be between 0.1 and 30 seconds")
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError("Viser is required; install it with python -m pip install 'viser>=1.0,<2'") from exc
    source = Path(source).expanduser().resolve()
    source.mkdir(parents=True, exist_ok=True)
    server = viser.ViserServer(host=host, port=int(port), label="Fold exploration (read-only)")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/workspace/table", width=1.2, height=0.8, cell_size=0.05, section_size=0.25)
    state = _FoldViserState(server, source)
    stop = threading.Event()

    def follow() -> None:
        while not stop.is_set():
            try:
                with state.lock:
                    state.update()
            except Exception as exc:
                state.status.content = f"### Viewer update failed\n\n`{type(exc).__name__}: {exc}`"
            stop.wait(float(refresh_s))

    thread = threading.Thread(target=follow, daemon=True, name="fold-exploration-viser-follow")
    thread.start()
    print(f"Fold exploration Viser: http://{host}:{port}", flush=True)
    print(f"Following run output: {source}", flush=True)
    if open_browser:
        try:
            webbrowser.open(f"http://{host}:{port}", new=2)
        except Exception as exc:
            print(f"Browser auto-open skipped: {type(exc).__name__}: {exc}", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Fold exploration Viser stopped; no camera or robot command was sent.", flush=True)
    finally:
        stop.set()
        thread.join(timeout=max(1.0, float(refresh_s) + 0.5))
        server.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="fold_exploration timestamp output directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh-s", type=float, default=0.5)
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument("--open-browser", dest="open_browser", action="store_true",
                         help="open the viewer URL in a browser (default: only print the URL)")
    browser.add_argument("--no-open-browser", dest="open_browser", action="store_false")
    parser.set_defaults(open_browser=False)
    args = parser.parse_args(argv)
    return run_viewer(args.source, host=args.host, port=args.port, refresh_s=args.refresh_s,
                      open_browser=args.open_browser)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
