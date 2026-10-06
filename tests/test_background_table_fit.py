from dataclasses import replace
import numpy as np
import pytest
from tests.test_camera_parallel_table import scene
from cloth_agent.perception import _fit_table_plane_from_references, camera_height_map_mm, PerceptionError


def slanted_view(tmp_path):
    frame, config = scene(tmp_path, 175.)
    yy, xx = np.indices(frame.depth_m.shape)
    k, t = frame.intrinsics, frame.X_base_camera
    rays = np.stack([(xx-k[0,2])/k[0,0], (yy-k[1,2])/k[1,1], np.ones_like(xx)], axis=-1)
    # Horizontal physical plane at robot Z=100 mm, observed by oblique camera.
    depth = (.1-t[2,3]) / (rays @ t[:3,:3].T)[:,:,2]
    depth[30:70,30:70] -= .01
    return replace(frame, depth_m=depth), replace(config, table_plane_mode='background_fit')


def test_oblique_camera_recovers_horizontal_table(tmp_path):
    frame, config = slanted_view(tmp_path)
    co, stats = _fit_table_plane_from_references([frame],config,np.zeros(3))
    np.testing.assert_allclose(co,[0,0,100],atol=.03)
    height, valid, inferred = camera_height_map_mm(frame,config)
    np.testing.assert_allclose(inferred,co)
    assert abs(np.median(height[15:25,20:80])) < .03
    assert stats['mode']=='background_fit'
    assert stats['inlier_count'] >= 6


def test_outlier_patch_is_rejected_not_used_as_plane(tmp_path):
    frame, config = slanted_view(tmp_path)
    depth=frame.depth_m.copy();depth[10:25,15:30] += .06
    co, stats = _fit_table_plane_from_references([replace(frame,depth_m=depth)],config,np.zeros(3))
    np.testing.assert_allclose(co,[0,0,100],atol=.03)
    assert stats['inlier_count'] < stats['reference_count']


def test_missing_background_fails_instead_of_fallback(tmp_path):
    frame, config = slanted_view(tmp_path)
    with pytest.raises(PerceptionError):
        _fit_table_plane_from_references([replace(frame,depth_m=np.full_like(frame.depth_m,np.nan))],config,np.array([0,0,100]))


def test_local_depth_spikes_do_not_move_background_plane(tmp_path):
    # Native-size image with small biased depth islands at every reference center.
    # The surrounding bare tabletop remains at Z=100 mm.
    frame, config = scene(tmp_path)
    scale = 12
    k = frame.intrinsics.copy()
    k[:2] *= scale
    frame = replace(frame,
                    rgb=np.repeat(np.repeat(frame.rgb, scale, axis=0), scale, axis=1),
                    depth_m=np.repeat(np.repeat(frame.depth_m, scale, axis=0), scale, axis=1),
                    intrinsics=k)
    config = replace(config, table_plane_mode='background_fit')
    _, clean = _fit_table_plane_from_references([frame], config, np.zeros(3))
    depth = frame.depth_m.copy()
    for index, record in enumerate(clean['cameras']['A']):
        x, y = record['pixel_xy']
        depth[y-1:y+2, x-1:x+2] += ((index % 5) - 2) * .008
    co, stats = _fit_table_plane_from_references(
        [replace(frame, depth_m=depth)], config, np.zeros(3))
    np.testing.assert_allclose(co, [0, 0, 100], atol=.03)
    assert stats['inlier_count'] == stats['reference_count']


def test_compliant_support_tolerance_preserves_measured_surface(tmp_path):
    from cloth_agent.perception import _fit_background_references
    frame, config = scene(tmp_path)
    config = replace(config, table_plane_mode='background_fit')
    records = [dict(pixel_xy=[x,y], depth_median_m=.5 + .004*((i+j)%2*2-1))
               for i,x in enumerate(range(20,81,10))
               for j,y in enumerate(range(20,81,10))]
    with pytest.raises(PerceptionError, match='70%'):
        _fit_background_references(frame, records, {}, config, 0.)
    co, stats, points, _ = _fit_background_references(
        frame, records, {}, replace(config, background_plane_inlier_threshold_mm=6.), 0.)
    assert stats['inlier_count'] == len(records)
    assert stats['inlier_threshold_mm'] == 6.
    assert np.ptp(points[:,2]) == pytest.approx(8.)  # no flattening
    assert abs(co[2]-100) < 1.
    # A substantially displaced surface must still fail, even in sponge mode.
    for r in records:
        r['depth_median_m'] = .5 + (r['depth_median_m']-.5)*10
    with pytest.raises(PerceptionError):
        _fit_background_references(frame, records, {},
                                   replace(config, background_plane_inlier_threshold_mm=6.), 0.)


@pytest.mark.parametrize('value', [0, -1, 11, float('nan'), float('inf'), True, '6'])
def test_invalid_background_tolerance_is_rejected(tmp_path, value):
    _, config = scene(tmp_path)
    with pytest.raises(PerceptionError, match='background_plane_inlier_threshold_mm'):
        replace(config, background_plane_inlier_threshold_mm=value).validate()
