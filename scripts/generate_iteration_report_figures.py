"""Backfill deterministic Camera-A report sheets for auto-exploration iterations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.report_figure import compose_camera_perception_report


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _latest_auto_run(run_dir: Path) -> Path:
    root = run_dir / "results" / "auto_exploration"
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no auto-exploration runs under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def generate_for_iteration(run_dir: Path, iteration_dir: Path) -> Path | None:
    perception_path = iteration_dir / "perception.json"
    if not perception_path.is_file():
        return None
    artifact = json.loads(perception_path.read_text(encoding="utf-8"))
    perception = artifact.get("saved_result")
    relative_result = artifact.get("saved_result_path")
    if not isinstance(perception, dict) or not relative_result:
        return None
    result_path = run_dir / str(relative_result)

    selected_reference = None
    visual_path = iteration_dir / "claude_visual_plan.json"
    if visual_path.is_file():
        visual = json.loads(visual_path.read_text(encoding="utf-8"))
        selected_reference = visual.get("decision", {}).get("selected_reference")

    target = None
    target_path = iteration_dir / "grasp_target_visualization.json"
    if target_path.is_file():
        target_payload = json.loads(target_path.read_text(encoding="utf-8"))
        targets = target_payload.get("targets", [])
        if targets:
            target = targets[0]
    overlay_path = iteration_dir / "grasp_target_camera_A.png"
    molmo_overlay_path = iteration_dir / "molmo_all_parts_camera_A.png"
    number = int(iteration_dir.name.rsplit("_", 1)[-1])
    output = iteration_dir / "camera_A_perception_report.png"
    manifest = compose_camera_perception_report(
        perception,
        result_path,
        output,
        camera="A",
        run_name=run_dir.name,
        iteration=number,
        target_overlay_path=overlay_path if overlay_path.is_file() else None,
        molmo_annotation_path=(
            molmo_overlay_path if molmo_overlay_path.is_file() else None
        ),
        selected_reference=selected_reference,
        target=target,
    )
    gallery_dir = run_dir / "results" / "report_figures" / iteration_dir.parent.name
    gallery_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = gallery_dir / f"iteration_{number:03d}_camera_A.png"
    shutil.copy2(output, gallery_path)
    molmo_legend_path = iteration_dir / "molmo_all_parts_camera_A_with_legend.png"
    if molmo_legend_path.is_file():
        molmo_gallery_path = gallery_dir / f"iteration_{number:03d}_molmo_camera_A.png"
        shutil.copy2(molmo_legend_path, molmo_gallery_path)
        manifest["molmo_gallery_image"] = str(molmo_gallery_path)
    try:
        manifest["image"] = str(output.relative_to(run_dir))
        manifest["gallery_image"] = str(gallery_path.relative_to(run_dir))
        if manifest.get("molmo_gallery_image"):
            manifest["molmo_gallery_image"] = str(
                Path(str(manifest["molmo_gallery_image"])).relative_to(run_dir)
            )
    except ValueError:
        pass
    _write_json(iteration_dir / "camera_A_perception_report.json", manifest)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--auto-run-dir")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    auto_run = (
        Path(args.auto_run_dir).expanduser().resolve()
        if args.auto_run_dir
        else _latest_auto_run(run_dir)
    )
    generated = 0
    for iteration_dir in sorted(auto_run.glob("iteration_[0-9][0-9][0-9]")):
        output = generate_for_iteration(run_dir, iteration_dir)
        if output is not None:
            generated += 1
            print(output)
    print(f"generated={generated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
