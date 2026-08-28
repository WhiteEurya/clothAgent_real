#!/usr/bin/env python3
"""Capture, process, and compare repeated A/B perception observations.

This is the one-command orchestration layer for the separate modules:

1. capture calibrated RealSense A/B RGB-D;
2. run the saved RGB-D through the normal table/garment/fusion perception;
3. compute Camera-B-minus-Camera-A height disagreement;
4. aggregate repeatability and classify the bias as STABLE/VARIABLE/UNSTABLE.

No robot command, Claude call, or Viser server is started. The garment and
cameras must remain still for the complete capture series.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capture_rgbd_consistency import _resolve, _save_capture  # type: ignore[import-not-found]
from cloth_agent.config import ExperimentConfig, RobotConfig
from cloth_agent.perception import ClothCenterPerception, PerceptionConfig, capture_two_view_rgbd
from compare_perception_stability import (  # type: ignore[import-not-found]
    _aggregate,
    _capture_metrics,
    _classify,
    _parse_region,
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


def _run_one_capture(
    *,
    root: Path,
    config: PerceptionConfig,
    robot: RobotConfig,
    capture_dir: Path,
    capture_index: int,
    stride: int,
    min_xy_agreement_mm: float,
    regions: dict[str, tuple[int, int, int, int]],
) -> dict[str, Any]:
    raw_dir = capture_dir / "raw"
    perception_dir = capture_dir / "perception"
    raw_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    print(
        f"[stability] capture {capture_index}: capturing Camera A/B RGB-D "
        "(no robot motion)",
        flush=True,
    )
    frames = capture_two_view_rgbd(config)
    capture_manifest = _save_capture(frames, raw_dir)
    print(
        f"[stability] capture {capture_index}: processing table/mask/fused perception",
        flush=True,
    )
    service = ClothCenterPerception(root, robot, config)
    result, _ = service.locate(perception_dir, ExperimentConfig(), frames=frames)
    result_path = perception_dir / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    metrics = _capture_metrics(
        perception_dir,
        stride=stride,
        min_xy_agreement_mm=min_xy_agreement_mm,
        regions=regions,
    )
    metrics["capture_index"] = capture_index
    metrics["duration_s"] = time.monotonic() - started
    metrics["raw_capture_dir"] = str(raw_dir)
    metrics["perception_dir"] = str(perception_dir)
    (capture_dir / "capture_manifest.json").write_text(
        json.dumps(capture_manifest, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (capture_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    bias = metrics["cross_view_dh_B_minus_A_mm"].get("p50")
    lower = metrics["regions"].get("lower_sleeve", {}).get("cross_view_dh_B_minus_A_mm", {}).get("p50")
    print(
        f"[stability] capture {capture_index}: overall B-A={bias:.2f} mm; "
        f"lower_sleeve={lower:.2f} mm",
        flush=True,
    )
    return metrics


def _build_summary(
    captures: list[dict[str, Any]],
    regions: dict[str, tuple[int, int, int, int]],
) -> dict[str, Any]:
    def values(path: tuple[str, ...], region: str | None = None) -> list[float]:
        output: list[float] = []
        for capture in captures:
            data: Any = capture["regions"].get(region, {}) if region else capture
            for key in path:
                data = data.get(key, {}) if isinstance(data, dict) else {}
            if isinstance(data, (int, float)) and np.isfinite(data):
                output.append(float(data))
        return output

    overall = _aggregate(values(("cross_view_dh_B_minus_A_mm", "p50")))
    table_noise = _aggregate(
        [float(item["table_noise_p90_mm"]) for item in captures if item.get("table_noise_p90_mm") is not None]
    )
    offset_a = _aggregate(
        [float(item["camera_z_offsets_mm"]["A"]) for item in captures if item.get("camera_z_offsets_mm", {}).get("A") is not None]
    )
    offset_b = _aggregate(
        [float(item["camera_z_offsets_mm"]["B"]) for item in captures if item.get("camera_z_offsets_mm", {}).get("B") is not None]
    )
    summary: dict[str, Any] = {
        "created_at": _now(),
        "capture_count": len(captures),
        "stability": {
            "overall_height_bias_dh_B_minus_A_p50_mm": overall,
            "table_noise_p90_mm": table_noise,
            "camera_A_z_offset_mm": offset_a,
            "camera_B_z_offset_mm": offset_b,
            "classification": _classify(overall),
        },
        "regions": {},
        "captures": captures,
    }
    for name in regions:
        series = values(("cross_view_dh_B_minus_A_mm", "p50"), region=name)
        stats = _aggregate(series)
        summary["regions"][name] = {
            "height_bias_p50_mm": stats,
            "classification": _classify(stats),
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--perception-config",
        type=Path,
        default=Path("config/perception.free_exploration.json"),
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path("config/robot.example.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new series directory; defaults to results/perception_stability/<timestamp>",
    )
    parser.add_argument("--captures", type=int, default=3, help="number of still captures (default: 3)")
    parser.add_argument("--interval-s", type=float, default=1.0, help="delay between captures (default: 1s)")
    parser.add_argument("--temporal-median-frames", type=int)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--min-xy-agreement-mm", type=float, default=8.0)
    parser.add_argument("--region", action="append", default=None, help="NAME=x0,x1,y0,y1; repeatable")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.captures < 2:
        raise SystemExit("--captures must be at least 2")
    if args.interval_s < 0:
        raise SystemExit("--interval-s must be non-negative")
    root = args.project_root.expanduser().resolve()
    perception_path = _resolve(root, args.perception_config)
    robot_path = _resolve(root, args.robot_config)
    config = PerceptionConfig.load(root, perception_path)
    if args.temporal_median_frames is not None:
        from dataclasses import replace

        config = replace(config, temporal_median_frames=int(args.temporal_median_frames))
        config.validate()
    robot = RobotConfig.load(root, robot_path)
    region_args = args.region or ["lower_sleeve=280,540,550,719"]
    regions = dict(_parse_region(raw) for raw in region_args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "results" / "perception_stability" / stamp
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "perception_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "robot_config.json").write_text(
        json.dumps(asdict(robot), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"[stability] starting {args.captures} captures; keep garment and cameras still; "
        "no robot/Claude/Viser will be used",
        flush=True,
    )
    captures: list[dict[str, Any]] = []
    try:
        for index in range(1, args.captures + 1):
            capture_dir = output_dir / f"capture_{index:03d}"
            captures.append(
                _run_one_capture(
                    root=root,
                    config=config,
                    robot=robot,
                    capture_dir=capture_dir,
                    capture_index=index,
                    stride=args.stride,
                    min_xy_agreement_mm=args.min_xy_agreement_mm,
                    regions=regions,
                )
            )
            if index < args.captures and args.interval_s:
                time.sleep(args.interval_s)
        summary = _build_summary(captures, regions)
        report_path = output_dir / "stability_report.json"
        report_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        print(json.dumps({"report": str(report_path), "stability": summary["stability"], "regions": summary["regions"]}, ensure_ascii=False, indent=2, default=_json_default))
        return 0
    except BaseException as exc:
        (output_dir / "failure.json").write_text(
            json.dumps(
                {"created_at": _now(), "status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "completed_captures": len(captures)},
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
