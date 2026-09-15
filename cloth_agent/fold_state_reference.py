"""Static cross-garment fold-state references.

These images describe a visual state transition for a different garment.  They
are deliberately kept separate from Camera-A grounding artifacts: no pixel,
depth, or robot coordinate from a reference state is executable.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping


FOLD_STATE_IDS = (
    "state_00_unfolded",
    "state_01_left_sleeve",
    "state_02_right_sleeve",
    "state_03_left_side",
    "state_04_right_side",
    "state_05_bottom_hem",
)

FOLD_STATE_STEP_PAIRS: dict[str, tuple[str, str]] = {
    "left_sleeve": (FOLD_STATE_IDS[0], FOLD_STATE_IDS[1]),
    "right_sleeve": (FOLD_STATE_IDS[1], FOLD_STATE_IDS[2]),
    "left_side": (FOLD_STATE_IDS[2], FOLD_STATE_IDS[3]),
    "right_side": (FOLD_STATE_IDS[3], FOLD_STATE_IDS[4]),
    "hem_up": (FOLD_STATE_IDS[4], FOLD_STATE_IDS[5]),
    # Older runs may use this name in their supervisor ledger.
    "bottom_hem": (FOLD_STATE_IDS[4], FOLD_STATE_IDS[5]),
}


class FoldStateReferenceError(ValueError):
    """Raised when a static fold-state collection is incomplete or invalid."""


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FoldStateReferenceError(f"fold-state manifest is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FoldStateReferenceError(f"cannot read fold-state manifest: {path}") from exc
    if not isinstance(data, dict) or data.get("reference_type") != "static_cross_garment_fold_states":
        raise FoldStateReferenceError("manifest is not a static cross-garment fold-state collection")
    return data


def _state_files(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    states = manifest.get("states")
    if not isinstance(states, list):
        raise FoldStateReferenceError("manifest.states must be a list")
    result: dict[str, Path] = {}
    for item in states:
        if not isinstance(item, Mapping):
            continue
        state_id = str(item.get("state_id", "")).strip()
        filename = str(item.get("filename", "")).strip()
        if not state_id or not filename or Path(filename).name != filename:
            raise FoldStateReferenceError("each state needs a safe state_id and filename")
        if state_id in result:
            raise FoldStateReferenceError(f"duplicate state_id: {state_id}")
        path = (root / filename).resolve()
        if root.resolve() not in path.parents or not path.is_file() or path.stat().st_size <= 0:
            raise FoldStateReferenceError(f"state image is missing: {path}")
        result[state_id] = path
    missing = [state_id for state_id in FOLD_STATE_IDS if state_id not in result]
    if missing:
        raise FoldStateReferenceError(f"manifest is missing states: {', '.join(missing)}")
    return result


def stage_fold_state_pair(
    reference_dir: Path,
    iteration_dir: Path,
    step: str,
) -> dict[str, Any] | None:
    """Copy the current step's two reference images into the run directory.

    Returns metadata and run-local image paths.  A missing configured directory
    returns ``None`` so existing zero-reference runs remain valid.
    """

    root = Path(reference_dir).expanduser().resolve()
    if not root.exists():
        return None
    if step not in FOLD_STATE_STEP_PAIRS:
        raise FoldStateReferenceError(f"no static reference mapping for fold step: {step}")
    manifest = _read_manifest(root)
    files = _state_files(root, manifest)
    source_id, target_id = FOLD_STATE_STEP_PAIRS[step]
    destination = Path(iteration_dir).resolve() / "fold_state_reference"
    destination.mkdir(parents=True, exist_ok=True)
    source_path = destination / "fold_reference_source.png"
    target_path = destination / "fold_reference_target.png"
    shutil.copy2(files[source_id], source_path)
    shutil.copy2(files[target_id], target_path)
    run_manifest = {
        "schema_version": 1,
        "reference_type": "static_cross_garment_fold_states",
        "role": "semantic_target_state_only",
        "current_step": step,
        "source_state": source_id,
        "target_state": target_id,
        "source_image": source_path.name,
        "target_image": target_path.name,
        "source_collection": str(root),
        "source_manifest": manifest.get("manifest_path", str(root / "manifest.json")),
        "coordinate_policy": "reference pixels, scale, depth, XYZ, and robot actions are non-executable",
    }
    (destination / "reference_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        **run_manifest,
        "images": [source_path, target_path],
        "directory": destination,
    }

