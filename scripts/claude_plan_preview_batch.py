"""Generate several independent Claude plans from one saved perception input.

Each sample invokes ``claude_plan_preview.py`` on the same run directory, so
the Camera A/B RGB, raw-depth, height/edge/coordinate, and fused-depth evidence
is identical across samples.  Every sample stops before physical execution.
The resulting top-down and Camera A/B projections are copied into isolated
sample directories and composed into comparison sheets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]


ARTIFACT_NAMES = (
    "animation_frames.npz",
    "claude_plan.json",
    "claude_plan_camera_A.png",
    "claude_plan_camera_B.png",
    "claude_plan_topdown.png",
    "claude_plan_visualization.png",
    "claude_planning_input_images.json",
    "plan_preview_summary.json",
    "proposal.json",
)


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _target_before(
    actions: list[dict[str, Any]], action_name: str, *, after_close: bool = False
) -> dict[str, Any] | None:
    last_move: dict[str, Any] | None = None
    closed = False
    for action in actions:
        name = action.get("name")
        if name == "close_gripper":
            closed = True
        if name == "move":
            last_move = action
        if name == action_name and (not after_close or closed):
            return last_move
    return None


def _sample_summary(sample_index: int, sample_dir: Path) -> dict[str, Any]:
    proposal = json.loads((sample_dir / "proposal.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (sample_dir / "plan_preview_summary.json").read_text(encoding="utf-8")
    )
    actions = proposal["actions"]
    grasp = _target_before(actions, "close_gripper")
    release = _target_before(actions, "open_gripper", after_close=True)
    return {
        "sample": sample_index,
        "confidence": proposal["confidence"],
        "grasp_move": grasp,
        "release_move": release,
        "action_count": len(actions),
        "actions": actions,
        "garment_observation": proposal["garment_observation"],
        "reveal_strategy": proposal["reveal_strategy"],
        "expected_observation": proposal["expected_observation"],
        "preflight": summary["phases"]["preflight"]["status"],
        "controller_ik": summary["phases"]["controller_ik"]["status"],
        "animation": summary["phases"]["animation"]["status"],
        "animation_frame_count": summary["phases"]["animation"].get(
            "frame_count", 0
        ),
        "physical_execution_authority": summary["physical_execution_authority"],
        "physical_commands_sent": summary["physical_commands_sent"],
        "execution_status": summary["execution_status"],
        "directory": str(sample_dir),
    }


def _tile(path: Path, *, title: str, size: tuple[int, int]) -> Image.Image:
    tile = Image.new("RGB", size, (248, 248, 248))
    image = Image.open(path).convert("RGB")
    contained = ImageOps.contain(image, (size[0] - 20, size[1] - 58))
    x = (size[0] - contained.width) // 2
    y = 48 + (size[1] - 48 - contained.height) // 2
    tile.paste(contained, (x, y))
    draw = ImageDraw.Draw(tile)
    draw.text((14, 12), title, fill=(20, 20, 20), font=_font(18))
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(80, 80, 80), width=2)
    return tile


def _compose_grid(
    samples: list[Path], output_path: Path, *, views: tuple[str, ...]
) -> None:
    tile_size = (640, 390)
    canvas = Image.new(
        "RGB",
        (tile_size[0] * len(views), tile_size[1] * len(samples)),
        (255, 255, 255),
    )
    names = {
        "topdown": "claude_plan_topdown.png",
        "camera_A": "claude_plan_camera_A.png",
        "camera_B": "claude_plan_camera_B.png",
    }
    labels = {
        "topdown": "Top-down fused plan",
        "camera_A": "Camera A projection",
        "camera_B": "Camera B projection",
    }
    for row, sample_dir in enumerate(samples):
        for column, view in enumerate(views):
            tile = _tile(
                sample_dir / names[view],
                title=f"Sample {row + 1} | {labels[view]}",
                size=tile_size,
            )
            canvas.paste(tile, (column * tile_size[0], row * tile_size[1]))
    canvas.save(output_path)


def _compose_one_view(
    samples: list[Path], output_path: Path, *, filename: str, label: str
) -> None:
    tile_size = (640, 390)
    canvas = Image.new(
        "RGB", (tile_size[0] * len(samples), tile_size[1]), (255, 255, 255)
    )
    for index, sample_dir in enumerate(samples):
        tile = _tile(
            sample_dir / filename,
            title=f"Sample {index + 1} | {label}",
            size=tile_size,
        )
        canvas.paste(tile, (index * tile_size[0], 0))
    canvas.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--claude-timeout-s", type=int, default=600)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if args.samples < 1 or args.samples > 8:
        raise ValueError("--samples must be between 1 and 8")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir
        / "results"
        / f"claude_plan_preview_batch_{args.samples}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    live_preview_dir = run_dir / "results" / "claude_plan_preview"
    script = root / "scripts" / "claude_plan_preview.py"

    sample_dirs: list[Path] = []
    batch: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "sample_count_requested": args.samples,
        "same_saved_perception_for_all_samples": True,
        "physical_execution_authority": False,
        "physical_commands_sent": False,
        "samples": [],
    }
    for index in range(1, args.samples + 1):
        sample_dir = output_dir / f"sample_{index:02d}"
        sample_dir.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--run-dir",
            str(run_dir),
            "--perception-config",
            str(root / "config" / "perception.free_exploration.json"),
            "--claude-timeout-s",
            str(args.claude_timeout_s),
        ]
        print(f"SAMPLE {index}/{args.samples}: Claude planning started", flush=True)
        log_path = sample_dir / "stdout_stderr.txt"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=root,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.claude_timeout_s + 180,
                check=False,
                shell=False,
            )
        if completed.returncode != 0:
            failure = {
                "sample": index,
                "status": "failed",
                "returncode": completed.returncode,
                "log": str(log_path),
            }
            batch["samples"].append(failure)
            _write_json(output_dir / "batch_summary.json", batch)
            raise RuntimeError(
                f"sample {index} failed with code {completed.returncode}; see {log_path}"
            )
        for name in ARTIFACT_NAMES:
            source = live_preview_dir / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, sample_dir / name)
        experiment_source = run_dir / "workspace" / "experiment_001_claude_plan_preview.py"
        shutil.copy2(experiment_source, sample_dir / experiment_source.name)
        summary = _sample_summary(index, sample_dir)
        batch["samples"].append(summary)
        sample_dirs.append(sample_dir)
        print(
            f"SAMPLE {index}/{args.samples}: completed; "
            f"confidence={summary['confidence']} "
            f"grasp={summary['grasp_move']}",
            flush=True,
        )
        _write_json(output_dir / "batch_summary.json", batch)

    grid_path = output_dir / "claude_three_plan_comparison.png"
    _compose_grid(
        sample_dirs,
        grid_path,
        views=("topdown", "camera_A", "camera_B"),
    )
    camera_a_path = output_dir / "claude_three_camera_A.png"
    camera_b_path = output_dir / "claude_three_camera_B.png"
    topdown_path = output_dir / "claude_three_topdown.png"
    _compose_one_view(
        sample_dirs,
        camera_a_path,
        filename="claude_plan_camera_A.png",
        label="Camera A",
    )
    _compose_one_view(
        sample_dirs,
        camera_b_path,
        filename="claude_plan_camera_B.png",
        label="Camera B",
    )
    _compose_one_view(
        sample_dirs,
        topdown_path,
        filename="claude_plan_topdown.png",
        label="Top-down",
    )
    batch["status"] = "completed"
    batch["sample_count_completed"] = len(sample_dirs)
    batch["visualizations"] = {
        "comparison_grid": str(grid_path),
        "camera_A": str(camera_a_path),
        "camera_B": str(camera_b_path),
        "topdown": str(topdown_path),
    }
    _write_json(output_dir / "batch_summary.json", batch)
    print(json.dumps(batch, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
