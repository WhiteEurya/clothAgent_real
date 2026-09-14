"""Local-only workspace diagnostics. Never an execution or camera interface."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import SafetyError


def _draw(image):
    draw = ImageDraw.Draw(image)
    try:
        draw.font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        draw.font = ImageFont.load_default()
    return draw


class WorkspaceTargetError(SafetyError):
    def __init__(self, trace):
        self.trace = trace
        point = next(p for p in trace["moves"] if p.get("error"))
        super().__init__(
            f"action {point['action_index']} ({point['target']}, "
            f"upright pixel={point['upright_pixel_xy']}, "
            f"base XYZ mm={point['base_xyz_mm']}): {point['error']}"
        )


def lateral_clearance(bounds, x, y, margin):
    if bounds.lateral_points_mm is None:
        return None
    nx, ny, low, high = bounds.lateral_geometry()
    p = nx * x + ny * y
    return {"normal_xy": [nx, ny], "projection_mm": p,
            "safe_projection_limits_mm": [low + margin, high - margin],
            "signed_clearance_mm": [p - low - margin, high - margin - p]}


def xy_allowed(xyz, config, yaw=90.0):
    """XY-only eligibility; final Z, actual yaw and IK still require validation."""
    xyz = np.asarray(xyz)
    x, y = xyz[..., 0], xyz[..., 1]
    valid = np.all(np.isfinite(xyz), axis=-1) & np.any(xyz != 0, axis=-1)
    bounds, margin = config.boundaries, config.workspace_margin_mm
    if bounds.lateral_points_mm is not None:
        nx, ny, low, high = bounds.lateral_geometry()
        p = nx * x + ny * y
        valid &= (p >= low + margin - 1e-6) & (p <= high - margin + 1e-6)
    for values, limits in ((x, (bounds.x_min, bounds.x_max)),
                           (y, config.y_workspace_bounds_mm(yaw))):
        low, high = limits
        if low is not None:
            valid &= values >= low + margin
        if high is not None:
            valid &= values <= high - margin
    return valid


def save_workspace_debug(directory: Path, views: Path, config, trace=None):
    """Save calibrated XY footprint and proposed TCPs, including rejected ones.

    Red image tint means measured XY is outside the possible workspace; dark
    pixels have no measurement. This is not a depth map or an IK feasibility map.
    The world plot is reconstructed from calibration, not independent ground truth.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(trace or {"moves": [], "status": "CANDIDATE_PRECHECK"})
    report["bounds"] = asdict(config.boundaries)
    report["workspace_margin_mm"] = config.workspace_margin_mm
    report["lower_z_margin_mm"] = config.lower_z_margin_mm
    report["meaning"] = "Calibrated robot base frame, mm. XY eligibility only; not IK approval. Negative signed clearance means outside."
    report["source_views"] = str(views)
    report_path = directory / "workspace_diagnostics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        xyz = np.load(views / "camera_A_base_xyz_mm.npy", allow_pickle=False)
        with Image.open(views / "camera_0_A.png") as source:
            raw = np.asarray(source.convert("RGB")).copy()
        if xyz.shape != (*raw.shape[:2], 3):
            raise ValueError("RGB and saved XYZ map shapes differ")
        valid = np.isfinite(xyz).all(axis=-1) & np.any(xyz != 0, axis=-1)
        allowed = xy_allowed(xyz, config)
        raw[valid & ~allowed] = (raw[valid & ~allowed] * .5 + np.array([128, 0, 0])).astype(np.uint8)
        raw[~valid] //= 3
        raw_image = Image.fromarray(raw)
        upright = raw_image.rotate(-90, expand=True)
        guide_path = views / "camera_A_coordinate_guide.json"
        guide = json.loads(guide_path.read_text()) if guide_path.is_file() else {}
        refs = []
        for sample in guide.get("samples", []):
            pixel, point = sample.get("pixel_xy"), sample.get("base_xyz_mm")
            if pixel is None or point is None:
                continue
            good = bool(xy_allowed(point, config))
            refs.append({"reference_id": sample.get("reference_id"), "pixel_xy": pixel,
                         "base_xyz_mm": point, "xy_eligible": good,
                         "lateral": lateral_clearance(config.boundaries, *point[:2], config.workspace_margin_mm)})
            u, v = pixel
            for im, uv in ((raw_image, (u, v)), (upright, (raw.shape[0] - 1 - v, u))):
                ImageDraw.Draw(im).ellipse((uv[0]-2, uv[1]-2, uv[0]+2, uv[1]+2),
                                         fill="lime" if good else "red")
        report["references"] = refs
        legend = ["LOCAL ONLY | XY workspace eligibility (not IK)",
                  "Red: outside XY bounds | Dark: no measured XYZ", "Green: XY eligible reference",
                  "TCP targets: cyan=passed workspace; red=rejected", "Action numbering is 1-based."]
        for im, key in ((raw_image, "raw_pixel_xy"), (upright, "upright_pixel_xy")):
            groups = {}
            for point in report.get("moves", []):
                groups.setdefault(tuple(point[key]), []).append(point)
            for (u,v), points in groups.items():
                color = "red" if any(p.get("error") for p in points) else "cyan"
                draw = _draw(im)
                draw.ellipse((u-7, v-7, u+7, v+7), outline=color, width=3)
                label = "#" + ",".join(str(p["action_index"]) for p in points)
                draw.text((min(u+8, max(0, im.width-len(label)*9)), min(v+8, im.height-18)), label, fill=color)
        for point in report.get("moves", []):
            coords = ", ".join(f"{n:.1f}" for n in point["base_xyz_mm"])
            legend.append(f"#{point['action_index']} {point['target']} XYZ=({coords}) mm")
            if point.get("error"):
                legend.append(point["error"])
            lateral = point.get("lateral")
            if lateral:
                legend.append("  side clearances mm: " + ", ".join(f"{n:.2f}" for n in lateral["signed_clearance_mm"]))
        for name, im in (("workspace_targets_raw.png", raw_image), ("workspace_targets_upright.png", upright)):
            canvas = Image.new("RGB", (max(im.width, 780), im.height + 22 * len(legend) + 20), "#18202c")
            canvas.paste(im, (0, 0))
            draw = _draw(canvas)
            for i, line in enumerate(legend):
                draw.text((10, im.height + 10 + 22*i), line, fill="white")
            canvas.save(directory / name)
        _save_world_plot(directory, xyz, valid, config, report)
    except Exception as exc:
        # A debug rendering error must never conceal the original safety error.
        report["render_error"] = f"{type(exc).__name__}: {exc}"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _save_world_plot(directory, xyz, valid, config, report):
    cloud = xyz[valid][::max(1, int(valid.sum()) // 6000), :2]
    targets = [p["base_xyz_mm"][:2] for p in report.get("moves", [])]
    reference_xy = [r["base_xyz_mm"][:2] for r in report.get("references", [])
                    if np.isfinite(r["base_xyz_mm"]).all()]
    anchors = [[0., 0.], *targets, *reference_xy, *(config.boundaries.lateral_points_mm or [])]
    if len(cloud):
        anchors.extend(np.percentile(cloud, [1, 99], axis=0).tolist())
    anchors = np.asarray(anchors)
    low, high = anchors.min(axis=0), anchors.max(axis=0)
    span = max(float((high-low).max()) * 1.15, 100.)
    center = (low+high)/2
    low = center-span/2
    size, pad = 760, 70
    grid_x = low[0] + np.arange(size) * span/(size-1)
    grid_y = low[1] + np.arange(size-1, -1, -1) * span/(size-1)
    gx, gy = np.meshgrid(grid_x, grid_y)
    safe = xy_allowed(np.stack([gx, gy, np.ones_like(gx)], axis=-1), config)
    bg = np.where(safe[..., None], [224, 243, 233], [252, 217, 218]).astype(np.uint8)
    canvas = Image.new("RGB", (1020, 990 + 23*len(report.get("moves", []))), "white")
    canvas.paste(Image.fromarray(bg), (pad, pad))
    draw = _draw(canvas)
    def uv(point):
        return (pad+(float(point[0])-low[0])/span*(size-1),
                pad+(1-(float(point[1])-low[1])/span)*(size-1))
    for i in range(6):
        k = pad + i*(size-1)/5
        draw.line((k, pad, k, pad+size-1), fill="#b9c6c2")
        draw.line((pad, k, pad+size-1, k), fill="#b9c6c2")
        draw.text((k-15, pad+size+8), f"{low[0]+i*span/5:.0f}", fill="black")
        draw.text((5, k), f"{low[1]+(5-i)*span/5:.0f}", fill="black")
    for point in cloud:
        u,v = uv(point)
        if pad <= u < pad+size and pad <= v < pad+size:
            draw.ellipse((u-1,v-1,u+1,v+1), fill="#71879b")
    if config.boundaries.lateral_points_mm is not None:
        nx, ny, lower, upper = config.boundaries.lateral_geometry()
        for side, bound in enumerate((lower + config.workspace_margin_mm, upper - config.workspace_margin_mm), 1):
            crossings = []
            for x in (low[0], low[0]+span):
                if abs(ny) > 1e-9:
                    y = (bound-nx*x)/ny
                    if low[1] <= y <= low[1]+span:
                        crossings.append(uv([x,y]))
            for y in (low[1], low[1]+span):
                if abs(nx) > 1e-9:
                    x = (bound-ny*y)/nx
                    if low[0] <= x <= low[0]+span:
                        crossings.append(uv([x,y]))
            if len(crossings) >= 2:
                draw.line((*crossings[0], *crossings[-1]), fill="#b20000", width=3)
                u,v = crossings[0]
                draw.text((min(u+5, 720), min(v+5, 810)), f"Side {side} (with margin)", fill="#b20000")
    previous = None
    groups = {}
    for reference in report.get("references", []):
        if not np.isfinite(reference["base_xyz_mm"]).all():
            continue
        u,v = uv(reference["base_xyz_mm"])
        draw.ellipse((u-2,v-2,u+2,v+2), fill="green" if reference["xy_eligible"] else "red")
    for point in report.get("moves", []):
        u,v = uv(point["base_xyz_mm"])
        if previous is not None:
            draw.line((*previous,u,v), fill="#276fac", width=2)
        previous = (u,v)
        color = "red" if point.get("error") else "blue"
        draw.ellipse((u-6,v-6,u+6,v+6), fill=color)
        groups.setdefault((u,v), []).append(point)
    for (u,v), points in groups.items():
        label = "#" + ",".join(str(p["action_index"]) for p in points)
        draw.text((u+9,v+7), label, fill="red" if any(p.get("error") for p in points) else "blue")
    u,v = uv([0,0])
    draw.text((u+4,v-16), "BASE ORIGIN", fill="black")
    draw.line((u-5,v,u+5,v), fill="black", width=2)
    draw.line((u,v-5,u,v+5), fill="black", width=2)
    draw.text((pad, 16), "LOCAL | Calibrated robot base XY (equal scale, mm)", fill="black")
    draw.text((pad, 36), "Red zone = outside XY limits; green = XY eligible at maximum yaw allowance", fill="black")
    draw.text((400, 865), "Base X (mm) ->", fill="black")
    draw.text((pad, 895), "Base Y increases upward. Grey: measured surface. Small green/red dots: eligible/rejected references.", fill="black")
    draw.text((pad, 918), "Blue: proposed TCP. Red: rejected TCP. Lines: proposed waypoint order, not executed motion.", fill="black")
    draw.text((pad, 941), "Coordinates depend on saved calibration. Heights and failed limits are listed below (mm).", fill="black")
    for i, point in enumerate(report.get("moves", [])):
        coords = ", ".join(f"{v:.2f}" for v in point["base_xyz_mm"])
        label = f"#{point['action_index']} XYZ=({coords})"
        if point.get("lateral"):
            label += " | side clearances: " + ", ".join(f"{v:.2f}" for v in point["lateral"]["signed_clearance_mm"])
        draw.text((pad, 966+23*i), label, fill="red" if point.get("error") else "blue")
    canvas.save(directory / "workspace_base_xy.png")
