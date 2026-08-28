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
    paths = _image_paths_from_text(_prompt_from_payload(evaluation))
    if paths:
        groups["evaluation"] = paths
    return groups


def _iter_images(iteration_dir: Path) -> list[Path]:
    """Return every saved raster artifact, in a useful stage order."""

    paths = [
        path.resolve()
        for path in iteration_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]

    def order(path: Path) -> tuple[int, str]:
        text = str(path.relative_to(iteration_dir)).lower()
        if "before_raw" in text:
            rank = 0
        elif "trajectory" in text or "proposal" in text:
            rank = 1
        elif "rollout_recording" in text:
            rank = 2
        elif "after_raw" in text:
            rank = 3
        else:
            rank = 4
        return rank, text

    return sorted(set(paths), key=order)


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
        f"- supervisor before: `{before.get('next_step', 'unknown')}` / `{before.get('trajectory_decision', 'unknown')}`"
    )
    lines.append(
        f"- supervisor after: `{after.get('next_step', 'unknown')}` / `{after.get('trajectory_decision', 'unknown')}`"
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
        "perception_artifacts.json",
    ]
    present = [name for name in files if (iteration_dir / name).is_file()]
    if present:
        lines.extend(["", "**Intermediate JSON artifacts**", "", *[f"- `{name}`" for name in present]])
    return "\n".join(lines)


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
        self.status = server.gui.add_markdown(
            f"### Folding exploration dashboard\n\nFollowing `{source}`. Waiting for iteration artifacts."
        )
        self.debug_panel = server.gui.add_markdown("### Debug tail\n\nWaiting for debug.log.")

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
        self.claude_image_handles[key] = (
            mtime,
            stage_key,
            self.server.gui.add_image(image, label=label),
        )

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
        label = f"{iteration_dir.name} | {relative}"
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

    def update(self) -> None:
        iteration_dirs = _iteration_dirs(self.source)
        displayed = 0
        claude_group_count = 0
        for iteration_dir in iteration_dirs:
            claude_groups = _claude_input_groups(iteration_dir, self.run_root)
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
            for image_path in _iter_images(iteration_dir):
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
            f"- iterations discovered: `{len(iteration_dirs)}`\n"
            f"- raster artifacts displayed: `{len(self.image_handles)}`\n"
            f"- Claude input images displayed: `{len(self.claude_image_handles)}`\n"
            f"- Claude input groups: `{claude_group_count}`\n"
            f"- trajectory overlays: `{len(self.path_handles)}`\n"
            f"- run status: `{summary.get('status', 'RUNNING')}`\n"
            "\nThe viewer is read-only; it does not control the robot."
        )
        self.debug_panel.content = "### Debug tail\n\n```text\n" + _short(debug_tail, 12000) + "\n```"


def run_viewer(source: Path, *, host: str = "127.0.0.1", port: int = 8765, refresh_s: float = 0.5) -> int:
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
    args = parser.parse_args(argv)
    return run_viewer(args.source, host=args.host, port=args.port, refresh_s=args.refresh_s)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
