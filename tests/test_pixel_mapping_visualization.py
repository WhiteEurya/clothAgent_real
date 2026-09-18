import json

import numpy as np
from PIL import Image
import pytest

from cloth_agent.pixel_mapping_visualization import PixelMappingView
from scripts.manual_pixel_move import raw_pixel
from scripts.visualize_pixel_mapping import main


def scene():
    yy, xx = np.indices((120, 180))
    rgb = np.full((120, 180, 3), 140, dtype=np.uint8)
    xyz = np.stack((300+xx, 100-yy, np.full_like(xx, 50)), axis=-1).astype(float)
    valid = np.ones((120, 180), dtype=bool)
    return rgb, xyz, valid


def test_xy_axes_equal_scale_and_direction():
    view = PixelMappingView(*scene())
    center = view.xy_to_panel([350, 20])
    dx = view.xy_to_panel([400, 20]) - center
    dy = view.xy_to_panel([350, 70]) - center
    assert dx[0] > 0 and dx[1] == pytest.approx(0)
    assert dy[0] == pytest.approx(0) and dy[1] < 0
    assert dx[0] == pytest.approx(-dy[1])
    assert view.render([(30, 40), (130, 80)]).size == (300, 120)


def test_grid_never_marks_invalid_depth():
    rgb, xyz, valid = scene()
    valid[50:80, 40:70] = False
    xyz[50:80, 40:70] = np.nan
    view = PixelMappingView(rgb, xyz, valid)
    assert np.array_equal(np.asarray(view.grid_rgb)[65, 55], rgb[65, 55])
    with pytest.raises(ValueError, match='Invalid selected pixel'):
        view.render([(55, 65)])
    with pytest.raises(ValueError, match='No finite points'):
        PixelMappingView(rgb, xyz, np.zeros_like(valid))


def test_side_by_side_scaling_does_not_shift_selected_pixel():
    # Composite 2000x720, displayed at 1000x360. Only the left 640 pixels
    # correspond to the raw 1280x720 image; right panel must not enter mapping.
    left_size = (1280 * 1000 / 2000, 360)
    assert raw_pixel(179.5, 275.5, left_size, (1280, 720)) == (359, 551)
    with pytest.raises(ValueError):
        raw_pixel(650, 100, left_size, (1280, 720))


def test_offline_export_reconstructs_xyz_and_applies_saved_z_offset(tmp_path, capsys):
    rgb, _, _ = scene()
    Image.fromarray(rgb).save(tmp_path / 'rgb.png')
    np.save(tmp_path / 'depth.npy', np.full((120, 180), .5))
    (tmp_path / 'result.json').write_text(json.dumps({'views': [{
        'label': 'A', 'image': 'rgb.png', 'depth_m': 'depth.npy',
        'intrinsics': [[100, 0, 90], [0, 100, 60], [0, 0, 1]],
        'X_base_camera': np.eye(4).tolist(), 'base_z_offset_mm': 7,
    }]}))
    output = tmp_path / 'out.png'
    assert main(['--perception', str(tmp_path), '--pixel', '90', '60',
                 '--output', str(output)]) == 0
    assert '[0.0, 0.0, 507.0]' in capsys.readouterr().out
    with Image.open(output) as image:
        assert image.size == (300, 120)
