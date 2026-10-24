"""Reject locally changing depth, even when the background is stationary."""
import numpy as np


# Depths are float32 metres. Converting their differences to millimetres can
# put an exact threshold value slightly above the limit (e.g. 3.00002098 mm).
# One micrometre covers roundoff over the accepted 0.1–2.5 m depth range.
_COMPARISON_ATOL_MM = 0.001


def depth_window_report(depths, max_range_mm):
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
    h, w = common.shape
    scores = []
    # Central work surface; separate tiles prevent a stable background from
    # masking a locally drifting garment. Invalid tiles fail the check.
    for ys in np.array_split(np.arange(int(h*.15), int(h*.85)), 6):
        for xs in np.array_split(np.arange(int(w*.25), int(w*.8)), 6):
            mask = common[np.ix_(ys, xs)]
            if not mask.size or mask.mean() < .7:
                return dict(stable=False, reason='insufficient_local_depth')
            scores.append(float(np.median(spread[np.ix_(ys, xs)][mask])*1000))
    worst = max(scores)
    return dict(stable=worst <= max_range_mm + _COMPARISON_ATOL_MM,
                worst_tile_range_mm=worst, max_range_mm=max_range_mm,
                comparison_atol_mm=_COMPARISON_ATOL_MM, tiles=len(scores))
