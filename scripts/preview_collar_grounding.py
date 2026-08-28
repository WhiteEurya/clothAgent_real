#!/usr/bin/env python3
"""Run the saved-image collar grounding pipeline without robot motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.collar_lift_retreat import (  # noqa: E402
    COLLAR_MOLMO_SPECS,
    _selection_overlay,
    invoke_claude_collar_selector,
)
from cloth_agent.molmo_keypoint_pipeline import (  # noqa: E402
    run_molmo_semantic_anchor_pipeline,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _camera_a_evidence(perception_dir: Path, output_dir: Path) -> list[Path]:
    candidates = [
        perception_dir / "camera_0_A.png",
        perception_dir / "camera_A_garment_only.png",
        perception_dir / "camera_A_height_map_heatmap.png",
        perception_dir / "camera_A_height_gradient_edges.png",
        output_dir / "camera_A_semantic_anchors.png",
        output_dir / "camera_A_semantic_anchor_diagnostics.png",
        output_dir / "flat_reference" / "camera_A_flat_reference.png",
        output_dir / "flat_reference" / "camera_A_flat_reference_anchors.png",
    ]
    return [path.resolve() for path in candidates if path.is_file()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--perception-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--molmo-python", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--claude-timeout-s", type=int, default=300)
    parser.add_argument("--min-collar-confidence", type=float, default=0.0)
    parser.add_argument("--skip-claude", action="store_true")
    args = parser.parse_args(argv)

    root = args.project_root.expanduser().resolve()
    perception = args.perception_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    manifest = run_molmo_semantic_anchor_pipeline(
        project_root=root,
        perception_dir=perception,
        artifact_dir=output,
        confidence_threshold=args.confidence_threshold,
        molmo_python=args.molmo_python,
        keypoint_specs=COLLAR_MOLMO_SPECS,
        cameras=("A",),
        max_anchors=len(COLLAR_MOLMO_SPECS),
        install=False,
    )
    summary: dict[str, object] = {
        "status": "MOLMO_READY",
        "perception_dir": str(perception),
        "output_dir": str(output),
        "molmo_manifest": str(output / "molmo_semantic_anchors.json"),
        "molmo_anchor_count": manifest.get("anchor_count"),
        "topology_guide_count": manifest.get("topology_guide_count"),
    }
    if args.skip_claude:
        _write_json(output / "preview_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    reference_path = output / "flat_reference" / "reference_anchors.json"
    if not reference_path.is_file():
        raise RuntimeError("collar preview requires copied flat-reference anchors")
    reference_anchors = json.loads(reference_path.read_text(encoding="utf-8"))
    evidence = _camera_a_evidence(perception, output)
    raw_image = perception / "camera_0_A.png"
    if not raw_image.is_file():
        raise RuntimeError(f"Camera A raw image is missing: {raw_image}")
    from PIL import Image

    with Image.open(raw_image) as image:
        width, height = image.size
    selection, claude_log = invoke_claude_collar_selector(
        evidence,
        run_dir=root,
        image_width=width,
        image_height=height,
        molmo_manifest=manifest,
        reference_anchors=reference_anchors,
        binary=args.claude_binary,
        timeout_s=args.claude_timeout_s,
        min_confidence=args.min_collar_confidence,
    )
    _write_json(output / "claude_collar_selection.json", claude_log)
    if selection.pixel_xy is not None:
        _selection_overlay(raw_image, selection, output / "collar_grasp_overlay.png")
    summary.update(
        {
            "status": selection.status,
            "selection": selection.as_dict(),
            "collar_grasp_overlay": (
                str(output / "collar_grasp_overlay.png")
                if selection.pixel_xy is not None
                else None
            ),
        }
    )
    _write_json(output / "preview_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
