import numpy as np

from cloth_agent.dual_cloud_seam import harmonize_patch, local_appearance_seam


def test_harmonic_gain_preserves_black_and_does_not_change_other_pixels():
    secondary = np.full((80, 80, 3), 100, np.uint8)
    primary = np.full_like(secondary, 150)
    secondary[38:42, 38:42] = 0
    valid = np.ones((80, 80), bool)
    selected = np.zeros_like(valid)
    selected[25:55, 25:55] = True
    primary_valid = valid & ~selected
    original = secondary.copy()
    result, changed, patches = harmonize_patch(primary, secondary, [valid, primary_valid], selected)
    assert patches == 1
    assert np.array_equal(secondary, original)
    assert np.array_equal(result[~selected], secondary[~selected])
    assert not result[38:42, 38:42].any()
    assert (result[30, 30] > 140).all()
    assert np.array_equal(changed, selected)


def test_seam_never_selects_missing_secondary_or_changes_coverage():
    primary = np.full((100, 100, 3), 150, np.uint8)
    secondary = np.full_like(primary, 100)
    primary_valid = np.ones((100, 100), bool)
    primary_valid[40:60, 40:60] = False
    secondary_valid = np.ones_like(primary_valid)
    secondary_valid[:5] = False
    forced = np.zeros_like(primary_valid)
    forced[65:68, 65:68] = True
    protected = np.zeros_like(primary_valid)
    protected[30:35, 30:35] = True
    originals = [v.copy() for v in (secondary_valid, primary_valid)]
    chosen, _, _, report = local_appearance_seam(primary, secondary, [secondary_valid, primary_valid], forced, protected)
    assert not chosen[~secondary_valid].any()
    assert chosen[~primary_valid & secondary_valid].all()
    assert chosen[forced].all()
    assert not chosen[protected].any()
    assert all(np.array_equal(a, b) for a, b in zip(originals, [secondary_valid, primary_valid]))
    assert report['geometry_modified'] is False


def test_no_gain_completion_for_large_or_open_regions():
    secondary = np.full((180, 180, 3), 100, np.uint8)
    primary = np.full_like(secondary, 150)
    valid = np.ones((180, 180), bool)
    selected = valid.copy()
    result, changed, patches = harmonize_patch(primary, secondary, [valid, ~valid], selected)
    assert patches == 0 and not changed.any()
    assert np.array_equal(result, secondary)


def test_exposure_correction_preserves_color_ratios_without_colored_halos():
    secondary = np.empty((80, 80, 3), np.uint8)
    secondary[:] = [100, 120, 80]
    primary = np.full_like(secondary, 150)
    valid = np.ones((80, 80), bool)
    selected = np.zeros_like(valid)
    selected[25:55, 25:55] = True
    result, _, _ = harmonize_patch(primary, secondary, [valid, valid & ~selected], selected)
    ratios = result[40, 40].astype(float)/secondary[40, 40]
    assert np.ptp(ratios) < .02
