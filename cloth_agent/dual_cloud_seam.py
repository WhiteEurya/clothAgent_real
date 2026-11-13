"""Local appearance seams; source coordinates stay intact, ownership may change."""
import cv2
import numpy as np


def harmonize_patch(primary, secondary, coverage, selected):
    """Harmonic log-gain inside small closed patches, retaining source texture.

    The correction is multiplicative (black remains black). Missing secondary
    color is used only to estimate the smooth gain; it is never made measured.
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
    secondary_valid, primary_valid = coverage
    domain = selected | ~(secondary_valid | primary_valid)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(domain.astype(np.uint8), 8)
    corrected = secondary.copy()
    changed = np.zeros(selected.shape, bool)
    height, width = selected.shape
    patches = 0
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 8 or area > 20000 or x < 4 or y < 4 or x+w >= width-4 or y+h >= height-4:
            continue
        region = np.s_[y-3:y+h+3, x-3:x+w+3]
        mask = labels[region] == label
        if np.mean(secondary_valid[region][mask]) < .9:
            continue
        boundary = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~mask
        if not primary_valid[region][boundary].all():
            continue
        source = cv2.inpaint(secondary[region], (~secondary_valid[region]).astype(np.uint8), 3, cv2.INPAINT_TELEA)
        target = primary[region]
        yy, xx = np.nonzero(mask)
        ids = np.full(mask.shape, -1, int)
        ids[yy, xx] = np.arange(len(yy))
        matrix = lil_matrix((len(yy), len(yy)))
        matrix.setdiag(4)
        rhs = np.zeros(len(yy))
        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ny, nx = yy+dy, xx+dx
            neighbors = ids[ny, nx]
            inside = neighbors >= 0
            matrix[np.flatnonzero(inside), neighbors[inside]] = -1
            outside = ~inside
            target_color = target[ny[outside], nx[outside]].astype(float)
            source_color = source[ny[outside], nx[outside]].astype(float)
            # Exposure only: independent channel gains create colored halos.
            ratio = (target_color.mean(1)+8) / (source_color.mean(1)+8)
            # A black-cloth/table boundary mismatch is geometry, not exposure.
            mismatch = (target_color.max(1)<65) != (source_color.max(1)<65)
            ratio[mismatch] = 1.
            rhs[outside] += np.log(np.clip(ratio, .5, 2.))
        gain = np.exp(spsolve(matrix.tocsr(), rhs))
        values = np.clip(np.rint(secondary[region][yy, xx]*gain[:, None]), 0, 255).astype(np.uint8)
        measured = secondary_valid[region][yy, xx] & selected[region][yy, xx]
        corrected[region][yy[measured], xx[measured]] = values[measured]
        changed[region][yy[measured], xx[measured]] = True
        patches += 1
    return corrected, changed, patches


def local_appearance_seam(primary, secondary, coverage, forced_secondary=None, protected_primary=None):
    """Move seams locally, then match low-frequency exposure, without RGB blur.

    Inputs are supersampled appearance images, ordered secondary then primary
    for coverage. Each output pixel still has exactly one geometry source.
    """
    secondary_valid, primary_valid = coverage
    nearby = cv2.dilate((~primary_valid).astype(np.uint8), np.ones((65, 65), np.uint8)).astype(bool)
    allowed_secondary = secondary_valid & nearby
    allowed_primary = primary_valid.copy()
    if protected_primary is not None:
        allowed_secondary[protected_primary & primary_valid] = False
    if forced_secondary is not None:
        forced = forced_secondary & secondary_valid
        allowed_secondary |= forced
        allowed_primary[forced] = False
    masks = [cv2.UMat(v.astype(np.uint8)*255) for v in (allowed_secondary, allowed_primary)]
    cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD').find(
        [secondary.astype(np.float32), primary.astype(np.float32)], [(0, 0), (0, 0)], masks)
    selected = allowed_secondary & (~allowed_primary | (masks[0].get() > 0))
    corrected, changed, patches = harmonize_patch(primary, secondary, coverage, selected)
    return selected, corrected, changed, dict(
        method='local_graphcut_harmonic_log_gain', radius_subpixels=32,
        corrected_patches=patches, corrected_subpixels=int(changed.sum()),
        geometry_modified=False, source_ownership_may_change=True,
        color_correction='shared_channel_luminance_gain',
        texture_source_reprojected=False)
