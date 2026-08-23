from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, gaussian_filter
from scipy.interpolate import griddata


def font(size: int, bold: bool = False):
    path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def lerp(a, b, t):
    return tuple(int(round(x + (y - x) * float(t))) for x, y in zip(a, b))


def color(v: float):
    v = max(0.0, min(1.0, float(v)))
    stops = [(0.0, (20, 48, 125)), (0.35, (25, 135, 204)), (0.62, (28, 211, 205)), (0.80, (250, 218, 55)), (1.0, (211, 42, 38))]
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        if v <= x1:
            return lerp(c0, c1, (v - x0) / (x1 - x0))
    return stops[-1][1]


def fill_for_display(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate small holes in the garment support without filling the table."""
    yy, xx = np.indices(values.shape, dtype=np.float32)
    points = np.column_stack((xx[valid], yy[valid]))
    linear = griddata(points, values[valid].astype(np.float64), (xx, yy), method="linear")
    # Only bridge nearby holes.  Pixels farther from measured garment cells remain background.
    support = binary_dilation(valid, iterations=2)
    nearest_indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    nearest = values[tuple(nearest_indices)]
    filled = np.where(np.isfinite(linear), linear, nearest)
    filled_valid = support & np.isfinite(filled)
    return filled.astype(np.float32), filled_valid


def robust_global(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    samples = values[valid]
    p02, p95 = np.percentile(samples, [2, 95])
    scaled = np.clip((values - p02) / max(p95 - p02, 1e-6), 0.0, 1.0)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    for y, x in zip(*np.where(valid)):
        rgb[y, x] = color(scaled[y, x])
    return rgb, {"p02_mm": float(p02), "p95_mm": float(p95), "p99_mm": float(np.percentile(samples, 99)), "max_mm": float(np.max(samples))}


def multiscale(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    values, valid = fill_for_display(values, valid)
    base, stats = robust_global(values, valid)
    weights = gaussian_filter(valid.astype(np.float32), sigma=2.5, mode="nearest")
    smooth_values = gaussian_filter(np.where(valid, values, 0.0).astype(np.float32), sigma=2.5, mode="nearest")
    baseline = smooth_values / np.maximum(weights, 1e-4)
    residual = np.where(valid, values - baseline, 0.0)
    mad = float(np.median(np.abs(residual[valid] - np.median(residual[valid]))))
    noise_floor = max(0.8, 1.4826 * mad)
    local_scale = max(noise_floor * 2.0, float(np.percentile(np.abs(residual[valid]), 95)))
    positive = np.clip((residual - noise_floor) / max(local_scale - noise_floor, 1e-6), 0.0, 1.0)
    negative = np.clip((-residual - noise_floor) / max(local_scale - noise_floor, 1e-6), 0.0, 1.0)
    out = base.astype(np.float32)
    for channel, tint in enumerate((np.array([0, 235, 240]), np.array([220, 70, 245]))):
        strength = positive if channel == 0 else negative
        alpha = (0.72 * strength)[..., None]
        out = out * (1.0 - alpha) + tint * alpha
    out[~valid] = (0, 0, 0)
    local_core = positive > 0.28
    local_boundary = local_core & ~binary_erosion(local_core, structure=np.ones((3, 3), dtype=bool))
    extreme = valid & (values >= stats["p99_mm"])
    out[local_boundary] = (0, 255, 255)
    out[extreme] = (255, 246, 150)
    stats.update({"local_mad_mm": mad, "noise_floor_mm": noise_floor, "local_scale_mm": local_scale, "local_positive_pixels": int(local_core.sum()), "extreme_pixels": int(extreme.sum())})
    return out.astype(np.uint8), stats


def enlarge(arr: np.ndarray, scale: int = 6, valid: np.ndarray | None = None) -> Image.Image:
    """Smoothly enlarge a map while keeping invalid table pixels dark."""
    image = Image.fromarray(arr).resize(
        (arr.shape[1] * scale, arr.shape[0] * scale), Image.Resampling.BICUBIC
    )
    if valid is None:
        return image
    alpha = Image.fromarray((valid.astype(np.uint8) * 255)).resize(
        image.size, Image.Resampling.BICUBIC
    )
    background = Image.new("RGB", image.size, (0, 0, 0))
    background.paste(image, mask=alpha)
    return background


def add_frame(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 82), (247, 249, 252))
    canvas.paste(image, (0, 82))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), title, fill=(24, 32, 48), font=font(28, bold=True))
    draw.text((20, 49), subtitle, fill=(88, 98, 116), font=font(17))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perception-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p = args.perception_dir
    values = np.load(p / "fused_height_map_mm.npy").astype(np.float32)
    valid = np.isfinite(values)
    display_values, display_valid = fill_for_display(values, valid)
    global_rgb, stats = robust_global(display_values, display_valid)
    multi_rgb, multi_stats = multiscale(values, valid)
    stats.update(multi_stats)

    # Put the actual captured RGB views above the two height encodings.
    cam_a = Image.open(p / "camera_0_A.png").convert("RGB").resize((640, 360))
    cam_b = Image.open(p / "camera_1_B.png").convert("RGB").resize((640, 360))
    top = Image.new("RGB", (1280, 442), (247, 249, 252))
    top.paste(add_frame(cam_a, "现场采集 · Camera A", "RealSense RGB（本次采集）"), (0, 0))
    top.paste(add_frame(cam_b, "现场采集 · Camera B", "RealSense RGB（本次采集）"), (640, 0))

    current = enlarge(global_rgb, 6, display_valid)
    proposed = enlarge(multi_rgb, 6, display_valid)
    bottom = Image.new("RGB", (1280, max(current.height, proposed.height) + 82), (247, 249, 252))
    bottom.paste(add_frame(current, "当前全局 softmax", "同一帧融合高度图 · 极值会压平其他区域"), (0, 0))
    bottom.paste(add_frame(proposed, "全局高度 + 局部残差", "同一帧融合高度图 · 青色轮廓显示局部变化"), (640, 0))

    out = Image.new("RGB", (1280, top.height + bottom.height + 115), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.text((32, 22), "现场 RGB-D 采集结果：softmax 与多尺度高度可视化对比", fill=(18, 27, 43), font=font(34, bold=True))
    draw.text((34, 67), "这不是示意数据；下方高度图来自刚刚采集的 Cam A/B 融合结果。", fill=(88, 98, 116), font=font(20))
    out.paste(top, (0, 115))
    out.paste(bottom, (0, 115 + top.height))
    y = 115 + top.height + bottom.height + 14
    draw = ImageDraw.Draw(out)
    draw.text((34, y), f"融合高度统计：p02={stats['p02_mm']:.1f} mm · p95={stats['p95_mm']:.1f} mm · p99={stats['p99_mm']:.1f} mm · max={stats['max_mm']:.1f} mm · 局部噪声尺度={stats['noise_floor_mm']:.1f} mm", fill=(60, 69, 84), font=font(17))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output, quality=95)
    print(args.output)
    print(stats)


if __name__ == "__main__":
    main()
