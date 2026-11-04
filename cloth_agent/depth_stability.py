"""Reject locally changing depth, even when the background is stationary."""
import numpy as np


# Depths are float32 metres. Converting their differences to millimetres can
# put an exact threshold value slightly above the limit (e.g. 3.00002098 mm).
# One micrometre covers roundoff over the accepted 0.1–2.5 m depth range.
_COMPARISON_ATOL_MM = 0.001


def table_depth_mask(shape, polygon):
    """Fixed image-space tabletop ROI, independent of the depth values."""
    from PIL import Image, ImageDraw
    points = np.asarray(polygon, dtype=float)
    h, w = shape
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or not np.isfinite(points).all() or np.any(points < 0)
            or np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h)):
        raise ValueError('table_polygon_px requires at least three finite in-image XY points')
    area = abs(np.dot(points[:, 0], np.roll(points[:, 1], 1))
               - np.dot(points[:, 1], np.roll(points[:, 0], 1))) / 2
    if area < 1:
        raise ValueError('table_polygon_px has no area')
    image = Image.new('L', (w, h))
    ImageDraw.Draw(image).polygon([tuple(p) for p in points], fill=1)
    return np.asarray(image, dtype=bool)


def depth_window_report(depths, max_range_mm, *, table_polygon_px=None):
    stack = np.asarray(depths)[:, ::4, ::4]
    valid = np.isfinite(stack) & (stack > .1) & (stack <= 2.5)
    midpoint = len(stack)//2
    medians = []
    coverage = []
    for values, mask in ((stack[:midpoint], valid[:midpoint]),
                         (stack[midpoint:], valid[midpoint:])):
        medians.append(np.ma.median(np.ma.array(values, mask=~mask), axis=0).filled(np.nan))
        coverage.append(mask.mean(axis=0) >= .6)
    common = coverage[0] & coverage[1]
    spread = np.abs(medians[1] - medians[0])
    roi = (table_depth_mask(np.asarray(depths).shape[1:], table_polygon_px)[::4, ::4]
           if table_polygon_px is not None else None)
    return depth_tile_report(common, spread * 1000, max_range_mm, roi=roi, frames=len(stack))


def depth_tile_report(common, spread_mm, max_range_mm, *, roi=None, frames=None):
    """Evaluate local coverage and drift only within the selected surface."""
    h, w = common.shape
    if roi is None:
        roi = np.zeros_like(common, dtype=bool)
        roi[int(h*.15):int(h*.85), int(w*.25):int(w*.8)] = True
    ys_roi, xs_roi = np.where(roi)
    if not len(ys_roi):
        return dict(stable=False, reason='empty_table_roi')
    scores = []
    for row, ys in enumerate(np.array_split(np.arange(ys_roi.min(), ys_roi.max()+1), 6)):
        for col, xs in enumerate(np.array_split(np.arange(xs_roi.min(), xs_roi.max()+1), 6)):
            selected = roi[np.ix_(ys, xs)]
            if not selected.any():
                continue
            mask = common[np.ix_(ys, xs)]
            fraction = float(mask[selected].mean())
            if fraction < .7:
                return dict(stable=False, reason='insufficient_local_depth',
                    tile_row=row, tile_col=col,
                    valid_coverage=fraction, required_coverage=.7, frames=frames)
            scores.append(float(np.median(spread_mm[np.ix_(ys, xs)][mask & selected])))
    worst = max(scores)
    return dict(stable=worst <= max_range_mm + _COMPARISON_ATOL_MM,
                worst_tile_range_mm=worst, max_range_mm=max_range_mm,
                comparison_atol_mm=_COMPARISON_ATOL_MM, tiles=len(scores))
