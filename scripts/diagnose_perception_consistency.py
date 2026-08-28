#!/usr/bin/env python3
"""Audit saved A/B RGB-D perception consistency without commanding a robot.

The report is designed to answer one question before changing grasp policy:
whether a Camera-A surface is actually low, or whether Camera A and Camera B
disagree about the same garment surface.  It reprojects every valid Camera-A
base-XYZ sample into Camera B, compares the two calibrated Z/height values,
summarises table and garment residuals, and writes a colour-coded disagreement
map.

Example::

    /home/CNS2026330003/miniconda3/envs/molmo/bin/python \
      scripts/diagnose_perception_consistency.py \
      --run-dir runs/neat_fold_real_01 \
      --pixel 408,652 --pixel 433,597 \
      --region lower_sleeve=280,540,550,719

This script is read-only with respect to the run and never opens a camera or
connects to the robot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def _finite_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    p = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(p[0]),
        "p05": float(p[1]),
        "p25": float(p[2]),
        "p50": float(p[3]),
        "p75": float(p[4]),
        "p95": float(p[5]),
        "p99": float(p[6]),
    }


def _load_latest_perception(run_dir: Path, explicit: Path | None) -> tuple[Path, dict[str, Any]]:
    if explicit is not None:
        if explicit.is_absolute():
            result_path = explicit
        else:
            run_relative = run_dir / explicit
            result_path = run_relative if run_relative.exists() else explicit
        result_path = result_path.expanduser().resolve()
        if result_path.is_dir():
            result_path = result_path / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        return result_path.parent, json.loads(result_path.read_text(encoding="utf-8"))
    candidates = sorted(
        (run_dir / "results" / "perception").glob("center_*/result.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no results/perception/center_*/result.json under {run_dir}")
    result_path = candidates[-1]
    return result_path.parent, json.loads(result_path.read_text(encoding="utf-8"))


def _array(perception_dir: Path, name: str) -> np.ndarray:
    path = perception_dir / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def _image(perception_dir: Path, name: str) -> np.ndarray:
    path = perception_dir / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("RGB"))


def _plane_residual(xyz: np.ndarray, coefficients: dict[str, Any]) -> np.ndarray:
    a = float(coefficients["a"])
    b = float(coefficients["b"])
    c = float(coefficients["c_mm"])
    return xyz[..., 2] - (a * xyz[..., 0] + b * xyz[..., 1] + c)


@dataclass(frozen=True)
class CrossViewAudit:
    dz: np.ndarray
    dh: np.ndarray
    a_xy_error: np.ndarray
    a_pixels: np.ndarray
    b_pixels: np.ndarray
    valid: np.ndarray


def _reproject_a_to_b(
    result: dict[str, Any],
    perception_dir: Path,
    *,
    stride: int,
    min_xy_agreement_mm: float,
) -> CrossViewAudit:
    views = {str(view["label"]): view for view in result.get("views", [])}
    if "A" not in views or "B" not in views:
        raise ValueError("perception result must contain Camera A and Camera B views")
    a_xyz = _array(perception_dir, "camera_A_base_xyz_mm.npy").astype(np.float64)
    b_xyz = _array(perception_dir, "camera_B_base_xyz_mm.npy").astype(np.float64)
    a_h = _array(perception_dir, "camera_A_height_above_table_mm.npy").astype(np.float64)
    b_h = _array(perception_dir, "camera_B_height_above_table_mm.npy").astype(np.float64)
    a_mask = _array(perception_dir, "camera_A_garment_mask.npy").astype(bool)
    b_mask = _array(perception_dir, "camera_B_garment_mask.npy").astype(bool)

    height, width = a_xyz.shape[:2]
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    points = a_xyz[yy, xx]
    finite = np.all(np.isfinite(points), axis=2)
    flat_points = points[finite]
    flat_yy = yy[finite]
    flat_xx = xx[finite]
    view_b = views["B"]
    intrinsics = np.asarray(view_b["intrinsics"], dtype=np.float64)
    base_from_camera = np.asarray(view_b["X_base_camera"], dtype=np.float64)
    camera_from_base = np.linalg.inv(base_from_camera)
    camera_points = (
        (flat_points / 1000.0) @ camera_from_base[:3, :3].T
        + camera_from_base[:3, 3]
    )
    in_front = camera_points[:, 2] > 1e-6
    u = intrinsics[0, 0] * camera_points[:, 0] / camera_points[:, 2] + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera_points[:, 1] / camera_points[:, 2] + intrinsics[1, 2]
    bx = np.rint(u).astype(np.int64)
    by = np.rint(v).astype(np.int64)
    inside = (
        in_front
        & (bx >= 0)
        & (bx < b_xyz.shape[1])
        & (by >= 0)
        & (by < b_xyz.shape[0])
    )
    flat_points = flat_points[inside]
    flat_yy = flat_yy[inside]
    flat_xx = flat_xx[inside]
    bx = bx[inside]
    by = by[inside]
    b_points = b_xyz[by, bx]
    valid = (
        np.all(np.isfinite(b_points), axis=1)
        & np.isfinite(a_h[flat_yy, flat_xx])
        & np.isfinite(b_h[by, bx])
    )
    xy_error = np.linalg.norm(b_points[:, :2] - flat_points[:, :2], axis=1)
    valid &= xy_error <= float(min_xy_agreement_mm)
    a_pixels = np.column_stack((flat_xx, flat_yy))
    b_pixels = np.column_stack((bx, by))
    dz = b_points[:, 2] - flat_points[:, 2]
    dh = b_h[by, bx] - a_h[flat_yy, flat_xx]
    return CrossViewAudit(dz=dz, dh=dh, a_xy_error=xy_error, a_pixels=a_pixels, b_pixels=b_pixels, valid=valid)


def _region_mask(shape: tuple[int, int], spec: tuple[int, int, int, int]) -> np.ndarray:
    x0, x1, y0, y1 = spec
    mask = np.zeros(shape, dtype=bool)
    mask[max(0, y0) : min(shape[0], y1 + 1), max(0, x0) : min(shape[1], x1 + 1)] = True
    return mask


def _parse_region(raw: str) -> tuple[str, tuple[int, int, int, int]]:
    if "=" not in raw:
        raise ValueError("region must be NAME=x0,x1,y0,y1")
    name, values = raw.split("=", 1)
    parts = [int(item.strip()) for item in values.split(",")]
    if len(parts) != 4:
        raise ValueError("region must contain x0,x1,y0,y1")
    return name.strip() or "region", tuple(parts)  # type: ignore[return-value]


def _auto_regions(mask: np.ndarray) -> dict[str, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {"garment": mask.copy()}
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    lower_start = int(round(y0 + 0.62 * (y1 - y0)))
    return {
        "garment": mask.copy(),
        "garment_lower_band": mask & (np.indices(mask.shape)[0] >= lower_start),
        "garment_left_lower_quadrant": mask
        & (np.indices(mask.shape)[0] >= lower_start)
        & (np.indices(mask.shape)[1] <= (x0 + x1) // 2),
    }


def _save_diverging_map(
    path: Path,
    shape: tuple[int, int],
    pixels: np.ndarray,
    values: np.ndarray,
    *,
    limit_mm: float,
    background: tuple[int, int, int] = (32, 32, 36),
) -> None:
    canvas = np.empty((*shape, 3), dtype=np.uint8)
    canvas[:, :] = np.asarray(background, dtype=np.uint8)
    finite = np.isfinite(values)
    values = np.clip(values, -limit_mm, limit_mm)
    # blue = B lower than A; red = B higher than A; gray = agreement
    t = (values / limit_mm + 1.0) * 0.5
    rgb = np.empty((len(values), 3), dtype=np.uint8)
    low = t <= 0.5
    high = ~low
    rgb[low, 0] = (255.0 * (2.0 * t[low])).astype(np.uint8)
    rgb[low, 1] = (255.0 * (2.0 * t[low])).astype(np.uint8)
    rgb[low, 2] = 255
    rgb[high, 0] = 255
    rgb[high, 1] = (255.0 * (2.0 - 2.0 * t[high])).astype(np.uint8)
    rgb[high, 2] = (255.0 * (2.0 - 2.0 * t[high])).astype(np.uint8)
    rgb[~finite] = np.asarray(background, dtype=np.uint8)
    canvas[pixels[:, 1], pixels[:, 0]] = rgb
    Image.fromarray(canvas).save(path)


def _pixel_report(
    perception_dir: Path,
    result: dict[str, Any],
    x: int,
    y: int,
    radius: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {"pixel_xy": [x, y], "radius_px": radius}
    a_xyz = _array(perception_dir, "camera_A_base_xyz_mm.npy").astype(np.float64)
    if not (0 <= x < a_xyz.shape[1] and 0 <= y < a_xyz.shape[0]):
        return {**out, "A": {"valid": False, "reason": "pixel outside image"}}
    a_point = a_xyz[y, x]
    out["A"] = None
    for cam in ("A", "B"):
        xyz = _array(perception_dir, f"camera_{cam}_base_xyz_mm.npy").astype(np.float64)
        height = _array(perception_dir, f"camera_{cam}_height_above_table_mm.npy").astype(np.float64)
        table = _array(perception_dir, f"camera_{cam}_table_z_mm.npy").astype(np.float64)
        mask = _array(perception_dir, f"camera_{cam}_garment_mask.npy").astype(bool)
        if cam == "A":
            sample_x, sample_y = x, y
        else:
            views = {str(view["label"]): view for view in result.get("views", [])}
            view = views.get("B")
            if view is None or not np.all(np.isfinite(a_point)):
                out[cam] = {"valid": False, "reason": "Camera B reprojection unavailable"}
                continue
            intrinsics = np.asarray(view["intrinsics"], dtype=np.float64)
            camera_from_base = np.linalg.inv(
                np.asarray(view["X_base_camera"], dtype=np.float64)
            )
            camera_point = camera_from_base @ np.r_[a_point / 1000.0, 1.0]
            if camera_point[2] <= 1e-6:
                out[cam] = {"valid": False, "reason": "reprojected point is behind Camera B"}
                continue
            sample_x = int(round(intrinsics[0, 0] * camera_point[0] / camera_point[2] + intrinsics[0, 2]))
            sample_y = int(round(intrinsics[1, 1] * camera_point[1] / camera_point[2] + intrinsics[1, 2]))
            out[cam] = {"reprojected_pixel_xy": [sample_x, sample_y]}
        if not (0 <= sample_x < xyz.shape[1] and 0 <= sample_y < xyz.shape[0]):
            out[cam] = {"valid": False, "reason": "reprojected pixel outside image", "pixel_xy": [sample_x, sample_y]}
            continue
        y0, y1 = max(0, sample_y - radius), min(xyz.shape[0], sample_y + radius + 1)
        x0, x1 = max(0, sample_x - radius), min(xyz.shape[1], sample_x + radius + 1)
        local_xyz = xyz[y0:y1, x0:x1].reshape(-1, 3)
        local_h = height[y0:y1, x0:x1].reshape(-1)
        local_t = table[y0:y1, x0:x1].reshape(-1)
        valid = np.all(np.isfinite(local_xyz), axis=1)
        base = {
            "valid": bool(valid.any()),
            "pixel_xy": [sample_x, sample_y],
            "pixel_xyz_mm": xyz[sample_y, sample_x].tolist(),
            "pixel_height_above_table_mm": float(height[sample_y, sample_x]),
            "pixel_table_z_mm": float(table[sample_y, sample_x]),
            "pixel_garment_mask": bool(mask[sample_y, sample_x]),
            "local_z_mm": _finite_stats(local_xyz[valid, 2]),
            "local_height_mm": _finite_stats(local_h[valid]),
            "local_table_z_mm": _finite_stats(local_t[valid]),
            "local_mask_fraction": float(np.mean(mask[y0:y1, x0:x1])),
        }
        if cam == "B":
            base["z_difference_B_minus_A_mm"] = float(
                xyz[sample_y, sample_x, 2] - a_point[2]
            )
            base["height_difference_B_minus_A_mm"] = float(
                height[sample_y, sample_x] - _array(
                    perception_dir, "camera_A_height_above_table_mm.npy"
                )[y, x]
            )
        out[cam] = {**(out[cam] or {}), **base}
    return out


def diagnose(
    run_dir: Path,
    perception_dir: Path,
    result: dict[str, Any],
    *,
    stride: int,
    min_xy_agreement_mm: float,
    pixels: Iterable[tuple[int, int]],
    regions: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    coeff = result.get("depth_fusion", {}).get("table_plane", {}).get("coefficients", {})
    if not {"a", "b", "c_mm"}.issubset(coeff):
        raise ValueError("perception result has no fitted table-plane coefficients")
    a_xyz = _array(perception_dir, "camera_A_base_xyz_mm.npy").astype(np.float64)
    b_xyz = _array(perception_dir, "camera_B_base_xyz_mm.npy").astype(np.float64)
    a_h = _array(perception_dir, "camera_A_height_above_table_mm.npy").astype(np.float64)
    b_h = _array(perception_dir, "camera_B_height_above_table_mm.npy").astype(np.float64)
    a_mask = _array(perception_dir, "camera_A_garment_mask.npy").astype(bool)
    b_mask = _array(perception_dir, "camera_B_garment_mask.npy").astype(bool)
    a_rgb = _image(perception_dir, "camera_0_A.png")
    b_rgb = _image(perception_dir, "camera_1_B.png")
    audit = _reproject_a_to_b(
        result,
        perception_dir,
        stride=max(1, int(stride)),
        min_xy_agreement_mm=float(min_xy_agreement_mm),
    )
    valid = audit.valid
    report: dict[str, Any] = {
        "created_at": _now(),
        "run_dir": str(run_dir),
        "perception_dir": str(perception_dir),
        "perception_status": result.get("status"),
        "primary_camera": result.get("primary_camera"),
        "table_plane": coeff,
        "table_noise_p90_mm": result.get("depth_fusion", {}).get("table_noise_p90_mm"),
        "garment_relief_threshold_mm": result.get("depth_fusion", {}).get("garment_relief_threshold_mm"),
        "camera_z_bias_correction": result.get("depth_fusion", {}).get("table_plane", {}).get("camera_z_bias_correction"),
        "camera_shapes": {"A": list(a_xyz.shape), "B": list(b_xyz.shape)},
        "cross_view": {
            "sample_stride_px": int(stride),
            "min_xy_agreement_mm": float(min_xy_agreement_mm),
            "sample_count": int(len(valid)),
            "valid_count": int(valid.sum()),
            "valid_fraction": float(valid.mean()) if len(valid) else 0.0,
            "dz_B_minus_A_mm": _finite_stats(audit.dz[valid]),
            "dh_B_minus_A_mm": _finite_stats(audit.dh[valid]),
            "xy_agreement_mm": _finite_stats(audit.a_xy_error[valid]),
        },
        "cameras": {},
        "regions": {},
        "pixels": [_pixel_report(perception_dir, result, x, y, 3) for x, y in pixels],
        "artifacts": {},
    }
    for cam, xyz, height, mask, rgb in (
        ("A", a_xyz, a_h, a_mask, a_rgb),
        ("B", b_xyz, b_h, b_mask, b_rgb),
    ):
        residual = _plane_residual(xyz, coeff)
        finite = np.all(np.isfinite(xyz), axis=2)
        luma = (
            0.2126 * rgb[..., 0].astype(np.float64)
            + 0.7152 * rgb[..., 1].astype(np.float64)
            + 0.0722 * rgb[..., 2].astype(np.float64)
        )
        table_candidate = finite & ~mask & (luma >= np.percentile(luma[finite & ~mask], 95.0))
        report["cameras"][cam] = {
            "finite_fraction": float(finite.mean()),
            "garment_mask_fraction": float(mask.mean()),
            "garment_height_mm": _finite_stats(height[finite & mask]),
            "garment_plane_residual_mm": _finite_stats(residual[finite & mask]),
            "table_candidate_residual_mm": _finite_stats(residual[table_candidate]),
            "table_candidate_count": int(table_candidate.sum()),
            "height_below_zero_fraction_in_mask": float(np.mean(height[finite & mask] < 0.0)),
            "height_below_minus_noise_fraction_in_mask": float(
                np.mean(
                    height[finite & mask]
                    < -float(result.get("depth_fusion", {}).get("table_noise_p90_mm") or 0.0)
                )
            ),
        }
    for name, region_mask in regions.items():
        region_mask = region_mask & a_mask & np.all(np.isfinite(a_xyz), axis=2)
        sample_mask = valid & np.array(
            [bool(region_mask[y, x]) for x, y in audit.a_pixels], dtype=bool
        )
        report["regions"][name] = {
            "pixel_count_A": int(region_mask.sum()),
            "camera_A_height_mm": _finite_stats(a_h[region_mask]),
            "camera_A_plane_residual_mm": _finite_stats(_plane_residual(a_xyz, coeff)[region_mask]),
            "cross_view_count": int(sample_mask.sum()),
            "cross_view_dz_B_minus_A_mm": _finite_stats(audit.dz[sample_mask]),
            "cross_view_dh_B_minus_A_mm": _finite_stats(audit.dh[sample_mask]),
        }
    flags: list[dict[str, Any]] = []
    table_noise = float(report.get("table_noise_p90_mm") or 0.0)
    if table_noise >= 0.75:
        flags.append(
            {
                "level": "WARN",
                "code": "TABLE_NOISE_EXCEEDS_MIN_GRASP_MARGIN",
                "message": (
                    "The measured table noise is larger than the minimum grasp "
                    "compression margin; near-table negative heights cannot be "
                    "treated as a decisive rejection."
                ),
                "table_noise_p90_mm": table_noise,
                "minimum_compression_mm": 0.75,
            }
        )
    offsets = (
        result.get("depth_fusion", {})
        .get("table_plane", {})
        .get("camera_z_bias_correction", {})
        .get("offsets_mm", {})
    )
    if abs(float(offsets.get("A", 0.0))) >= 5.0 or abs(float(offsets.get("B", 0.0))) >= 5.0:
        flags.append(
            {
                "level": "WARN",
                "code": "LARGE_CAMERA_Z_BIAS",
                "message": "At least one camera needs a large global Z bias correction.",
                "offsets_mm": offsets,
            }
        )
    garment_cross = report["regions"].get("garment", {}).get("cross_view_dh_B_minus_A_mm", {})
    if float(garment_cross.get("count", 0)) >= 100 and abs(float(garment_cross.get("p50", 0.0))) >= 3.0:
        flags.append(
            {
                "level": "FAIL",
                "code": "GARMENT_CROSS_VIEW_HEIGHT_DISAGREEMENT",
                "message": (
                    "Camera A and Camera B disagree on garment height. Do not use "
                    "Camera A height alone as the grasp-height truth."
                ),
                "dz_B_minus_A_mm": garment_cross,
            }
        )
    lower = report["regions"].get("lower_sleeve", report["regions"].get("garment_lower_band", {}))
    lower_a = lower.get("camera_A_height_mm", {})
    lower_cross = lower.get("cross_view_dh_B_minus_A_mm", {})
    if float(lower_a.get("count", 0)) >= 100 and float(lower_a.get("p25", 0.0)) < 0.0 and float(lower_cross.get("p50", 0.0)) >= 3.0:
        flags.append(
            {
                "level": "FAIL",
                "code": "LOW_CAMERA_A_LOWER_EDGE_WITH_POSITIVE_CAMERA_B_SUPPORT",
                "message": (
                    "The lower garment band looks below-table in Camera A while "
                    "Camera B reports it higher; this is the exact false-rejection "
                    "pattern seen at the sleeve cuff."
                ),
                "camera_A_height_mm": lower_a,
                "cross_view_dh_B_minus_A_mm": lower_cross,
            }
        )
    if report["cross_view"]["valid_fraction"] < 0.6:
        flags.append(
            {
                "level": "WARN",
                "code": "LOW_CROSS_VIEW_COVERAGE",
                "message": (
                    "Many Camera-A pixels do not have a valid Camera-B reprojection; "
                    "those pixels need an uncertainty fallback rather than a blind "
                    "single-camera decision."
                ),
                "valid_fraction": report["cross_view"]["valid_fraction"],
            }
        )
    report["diagnostic_flags"] = flags
    delta_path = output_dir / "camera_B_minus_A_height_mm.png"
    _save_diverging_map(
        delta_path,
        a_xyz.shape[:2],
        audit.a_pixels[valid],
        audit.dh[valid],
        limit_mm=15.0,
    )
    overlay = Image.fromarray(a_rgb).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for x, y in pixels:
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(255, 0, 0), width=3)
        draw.text((x + 10, y - 10), f"({x},{y})", fill=(255, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
    pixel_overlay_path = output_dir / "camera_A_selected_pixels.png"
    overlay.save(pixel_overlay_path)
    report["artifacts"] = {
        "camera_B_minus_A_height_mm": str(delta_path),
        "camera_A_selected_pixels": str(pixel_overlay_path),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--perception-dir", type=Path)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--min-xy-agreement-mm", type=float, default=8.0)
    parser.add_argument("--pixel", action="append", default=[], help="Camera-A pixel x,y; repeatable")
    parser.add_argument("--region", action="append", default=[], help="NAME=x0,x1,y0,y1; repeatable")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    perception_dir, result = _load_latest_perception(run_dir, args.perception_dir)
    a_mask = _array(perception_dir, "camera_A_garment_mask.npy").astype(bool)
    regions = _auto_regions(a_mask)
    for raw in args.region:
        name, spec = _parse_region(raw)
        regions[name] = _region_mask(a_mask.shape, spec)
    pixels: list[tuple[int, int]] = []
    for raw in args.pixel:
        values = [int(item.strip()) for item in raw.split(",")]
        if len(values) != 2:
            raise ValueError("pixel must be x,y")
        pixels.append((values[0], values[1]))
    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = run_dir / "results" / "perception_diagnostics" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report = diagnose(
        run_dir,
        perception_dir,
        result,
        stride=args.stride,
        min_xy_agreement_mm=args.min_xy_agreement_mm,
        pixels=pixels,
        regions=regions,
        output_dir=output_dir,
    )
    print(json.dumps({
        "report": str(output_dir / "report.json"),
        "perception_dir": str(perception_dir),
        "cross_view_valid_fraction": report["cross_view"]["valid_fraction"],
        "cross_view_dz_B_minus_A_mm": report["cross_view"]["dz_B_minus_A_mm"],
        "regions": report["regions"],
    }, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
