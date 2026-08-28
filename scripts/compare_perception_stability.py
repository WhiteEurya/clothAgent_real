#!/usr/bin/env python3
"""Measure whether the A/B height discrepancy is stable across captures.

This is an offline-only companion to ``diagnose_perception_consistency.py``.
Pass at least two processed perception directories (each containing
``result.json`` and the saved camera arrays).  The script computes the same
within-capture reprojected Camera-B-minus-Camera-A height statistics for every
capture, then reports the across-capture mean, standard deviation, range, and a
simple stability classification.

Example::

    python scripts/compare_perception_stability.py \
      --perception-dir runs/test_01/results/perception_processed/pipeline \
      --perception-dir runs/test_02/results/perception_processed/pipeline \
      --region lower_sleeve=280,540,550,719

The script never opens RealSense, moves the robot, or modifies the input
directories.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnose_perception_consistency import (  # type: ignore[import-not-found]
    _array,
    _finite_stats,
    _load_latest_perception,
    _parse_region,
    _reproject_a_to_b,
    _region_mask,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def _discover(run_dir: Path) -> list[Path]:
    processed = list((run_dir / "results").glob("**/pipeline/result.json"))
    # Prefer explicit processed outputs when present. Otherwise compare the
    # native center_* perception results, but never count both representations
    # of the same capture as two independent observations.
    candidates = processed or list(
        (run_dir / "results" / "perception").glob("center_*/result.json")
    )
    return sorted({path.parent.resolve() for path in candidates}, key=lambda p: p.stat().st_mtime)


def _aggregate(values: list[float]) -> dict[str, Any]:
    return _finite_stats(np.asarray(values, dtype=np.float64))


def _classify(stats: dict[str, Any]) -> str:
    count = int(stats.get("count", 0))
    if count < 2:
        return "INSUFFICIENT_CAPTURES"
    spread = float(stats.get("std", float("inf")))
    span = float(stats.get("max", float("inf")) - stats.get("min", -float("inf")))
    if spread <= 1.0 and span <= 3.0:
        return "STABLE"
    if spread >= 2.0 or span >= 6.0:
        return "UNSTABLE"
    return "VARIABLE"


def _capture_metrics(
    perception_dir: Path,
    *,
    stride: int,
    min_xy_agreement_mm: float,
    regions: dict[str, tuple[int, int, int, int]],
) -> dict[str, Any]:
    result_path = perception_dir / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    a_mask = _array(perception_dir, "camera_A_garment_mask.npy").astype(bool)
    a_height = _array(perception_dir, "camera_A_height_above_table_mm.npy").astype(np.float64)
    b_height = _array(perception_dir, "camera_B_height_above_table_mm.npy").astype(np.float64)
    audit = _reproject_a_to_b(
        result,
        perception_dir,
        stride=max(1, int(stride)),
        min_xy_agreement_mm=float(min_xy_agreement_mm),
    )
    valid = audit.valid
    report: dict[str, Any] = {
        "perception_dir": str(perception_dir),
        "created_at": result.get("created_at"),
        "perception_status": result.get("status"),
        "cross_view_valid_fraction": float(valid.mean()) if len(valid) else 0.0,
        "cross_view_dh_B_minus_A_mm": _finite_stats(audit.dh[valid]),
        "cross_view_dz_B_minus_A_mm": _finite_stats(audit.dz[valid]),
        "table_noise_p90_mm": result.get("depth_fusion", {}).get("table_noise_p90_mm"),
        "camera_z_offsets_mm": (
            result.get("depth_fusion", {})
            .get("table_plane", {})
            .get("camera_z_bias_correction", {})
            .get("offsets_mm", {})
        ),
        "regions": {},
    }
    for name, spec in regions.items():
        mask = _region_mask(a_mask.shape, spec) & a_mask
        sample_mask = valid & np.asarray(
            [bool(mask[y, x]) for x, y in audit.a_pixels], dtype=bool
        )
        report["regions"][name] = {
            "camera_A_height_mm": _finite_stats(a_height[mask]),
            "cross_view_count": int(sample_mask.sum()),
            "cross_view_dh_B_minus_A_mm": _finite_stats(audit.dh[sample_mask]),
            "cross_view_dz_B_minus_A_mm": _finite_stats(audit.dz[sample_mask]),
        }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--perception-dir",
        action="append",
        type=Path,
        help="processed perception directory containing result.json; repeatable",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="discover center_*/result.json and **/pipeline/result.json under this run",
    )
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--min-xy-agreement-mm", type=float, default=8.0)
    parser.add_argument(
        "--region",
        action="append",
        default=["lower_sleeve=280,540,550,719"],
        help="NAME=x0,x1,y0,y1; repeatable",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    directories = [path.expanduser().resolve() for path in (args.perception_dir or [])]
    if args.run_dir:
        directories.extend(_discover(args.run_dir.expanduser().resolve()))
    directories = sorted(set(directories), key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
    if len(directories) < 2:
        raise SystemExit(
            "need at least two processed perception directories; repeat --perception-dir "
            "or pass --run-dir containing multiple captures"
        )
    regions: dict[str, tuple[int, int, int, int]] = {}
    for raw in args.region:
        name, spec = _parse_region(raw)
        regions[name] = spec
    captures = [
        _capture_metrics(
            directory,
            stride=args.stride,
            min_xy_agreement_mm=args.min_xy_agreement_mm,
            regions=regions,
        )
        for directory in directories
    ]
    def series(path: list[str], region: str | None = None) -> list[float]:
        values: list[float] = []
        for item in captures:
            data: Any = item
            if region is not None:
                data = item["regions"].get(region, {})
            for key in path:
                data = data.get(key, {}) if isinstance(data, dict) else {}
            if isinstance(data, (int, float)) and np.isfinite(data):
                values.append(float(data))
        return values

    overall_dh = series(["cross_view_dh_B_minus_A_mm", "p50"])
    overall_dz = series(["cross_view_dz_B_minus_A_mm", "p50"])
    table_noise = [float(item["table_noise_p90_mm"]) for item in captures if item.get("table_noise_p90_mm") is not None]
    offset_a = [float(item.get("camera_z_offsets_mm", {}).get("A")) for item in captures if item.get("camera_z_offsets_mm", {}).get("A") is not None]
    offset_b = [float(item.get("camera_z_offsets_mm", {}).get("B")) for item in captures if item.get("camera_z_offsets_mm", {}).get("B") is not None]
    summary: dict[str, Any] = {
        "created_at": _now(),
        "capture_count": len(captures),
        "captures": captures,
        "stability": {
            "overall_height_bias_dh_B_minus_A_p50_mm": _aggregate(overall_dh),
            "overall_z_bias_dz_B_minus_A_p50_mm": _aggregate(overall_dz),
            "table_noise_p90_mm": _aggregate(table_noise),
            "camera_A_z_offset_mm": _aggregate(offset_a),
            "camera_B_z_offset_mm": _aggregate(offset_b),
            "overall_height_bias_classification": _classify(_aggregate(overall_dh)),
        },
        "regions": {},
    }
    for name in regions:
        values = series(["cross_view_dh_B_minus_A_mm", "p50"], region=name)
        summary["regions"][name] = {
            "height_bias_p50_mm": _aggregate(values),
            "classification": _classify(_aggregate(values)),
        }
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else directories[-1].parent / "stability_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "stability_report.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report_path), "stability": summary["stability"], "regions": summary["regions"]}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
