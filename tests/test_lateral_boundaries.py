import json
from types import SimpleNamespace
import sys

import pytest

from cloth_agent.config import ConfigError, SafetyError, WorkspaceBounds
from scripts.record_xarm_boundaries import record_sides


def strip(points):
    return WorkspaceBounds.from_mapping({
        "lateral_points_mm": points, "z_min": 0., "z_max": 600.,
    })


@pytest.mark.parametrize("points", [((0., 0.), (100., 100.)), ((100., 100.), (0., 0.))])
def test_rotated_strip_and_reversed_selection(points):
    bounds = strip(points)
    assert bounds.complete
    bounds.validate(50, 50, 300, require_complete=True)
    bounds.validate(550, -450, 300)  # Longitudinal movement is unrestricted.
    with pytest.raises(SafetyError, match="left/right"):
        bounds.validate(101, 101, 300, y_extension_mm=1000)
    with pytest.raises(SafetyError):
        bounds.validate(50, 50, -1)
    with pytest.raises(SafetyError, match="left/right"):
        bounds.validate(0, 0, 300, margin_mm=1)


@pytest.mark.parametrize("points", [[[0, 0], [0, 0]], [[0, 0], [float('nan'), 2]], [[0, 0]]])
def test_invalid_selection_rejected(points):
    with pytest.raises(ConfigError):
        strip(points)


def test_floor_only_has_no_software_ceiling():
    bounds = WorkspaceBounds.from_mapping({
        'lateral_points_mm': [[0, 0], [100, 100]], 'z_min': 10,
    })
    assert bounds.complete
    bounds.validate(50, 50, 100000, require_complete=True)
    with pytest.raises(SafetyError):
        bounds.validate(50, 50, 9)


def test_record_floor_removes_old_ceiling(tmp_path, monkeypatch):
    import scripts.record_xarm_z_bounds as module
    path = tmp_path / 'bounds.json'
    points = [[0, 0], [100, 100]]
    path.write_text(json.dumps({'boundary_mm': {
        'lateral_points_mm': points, 'z_min': 0, 'z_max': 500,
    }, 'samples': {'side_1': {'kept': True}, 'z_max': {'old': True}}}))
    monkeypatch.setattr(module, 'capture_points', lambda *a: {
        'z_min': {'tcp_pose_mm_deg': [50, 50, 20]},
    })
    monkeypatch.setattr('builtins.input', lambda _: 'SAVE')
    module.record_z(SimpleNamespace(ip='192.168.2.232', output=path))
    result = json.loads(path.read_text())
    assert result['boundary_mm'] == {'lateral_points_mm': points, 'z_min': 20}
    assert result['samples']['side_1'] == {'kept': True}
    assert 'z_max' not in result['samples']


@pytest.mark.parametrize("cancel", [False, True])
def test_record_sides_read_only_and_cancel_preserves_file(tmp_path, monkeypatch, cancel):
    path = tmp_path / "boundaries.json"
    original = json.dumps({"boundary_mm": {"x_min": 300, "y_min": -100, "y_max": 100,
                                            "z_min": 0, "z_max": 600}})
    path.write_text(original)
    answers = iter(["q"] if cancel else ["", "", "SAVE"])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    positions = iter([[0., 0., 300., 0., 0., 0.], [100., 100., 300., 0., 0., 0.]])
    disconnected = []
    arm = SimpleNamespace(connected=True,
        get_position=lambda: (0, next(positions)),
        get_servo_angle=lambda: (0, [0.] * 7),
        disconnect=lambda: disconnected.append(True))
    monkeypatch.setitem(sys.modules, "xarm.wrapper", SimpleNamespace(XArmAPI=lambda *a, **kw: arm))
    record_sides(SimpleNamespace(ip="192.168.2.232", base_boundaries=path, output=path))
    assert disconnected == [True]
    if cancel:
        assert path.read_text() == original
        assert not list(tmp_path.glob('*.bak'))
    else:
        data = json.loads(path.read_text())
        assert data['boundary_mm']['lateral_points_mm'] == [[0., 0.], [100., 100.]]
        assert 'x_min' not in data['boundary_mm']
        assert WorkspaceBounds.from_mapping(data['boundary_mm']).complete
        assert data['boundary_mm']['z_min'] == 0
        assert data['boundary_mm']['z_max'] == 600
        assert next(tmp_path.glob('*.bak')).read_text() == original


@pytest.mark.parametrize("z_first", [False, True])
def test_independent_capture_order(tmp_path, monkeypatch, z_first):
    from scripts.record_xarm_z_bounds import record_z
    path = tmp_path / 'new.json'
    args = SimpleNamespace(ip='192.168.2.232', output=path, base_boundaries=None)
    answers = iter(['', 'SAVE', '', '', 'SAVE'] if z_first else ['', '', 'SAVE', '', 'SAVE'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    sides = [[0., 0., 300., 0., 0., 0.], [100., 100., 300., 0., 0., 0.]]
    heights = [[50., 50., 10., 0., 0., 0.]]
    positions = iter(heights + sides if z_first else sides + heights)
    arm = SimpleNamespace(connected=True, get_position=lambda: (0, next(positions)),
                          get_servo_angle=lambda: (0, [0.] * 7), disconnect=lambda: None)
    monkeypatch.setitem(sys.modules, 'xarm.wrapper', SimpleNamespace(XArmAPI=lambda *a, **kw: arm))
    first, second = (record_z, record_sides) if z_first else (record_sides, record_z)
    first(args)
    assert not WorkspaceBounds.from_mapping(json.loads(path.read_text())['boundary_mm']).complete
    second(args)
    data = json.loads(path.read_text())
    assert data['boundary_mm'] == {'lateral_points_mm': [[0., 0.], [100., 100.]],
                                    'z_min': 10.}
    assert WorkspaceBounds.from_mapping(data['boundary_mm']).complete
    assert set(data['samples']) == {'side_1', 'side_2', 'z_min'}


def test_invalid_z_keeps_existing_file(tmp_path, monkeypatch):
    from scripts.record_xarm_z_bounds import record_z
    import scripts.record_xarm_z_bounds as module
    path = tmp_path / 'existing.json'
    original = json.dumps({'boundary_mm': {'lateral_points_mm': [[0, 0], [100, 100]]}})
    path.write_text(original)
    monkeypatch.setattr(module, 'capture_points', lambda *a: {
        'z_min': {'tcp_pose_mm_deg': [0, 0, float('nan')]},
    })
    with pytest.raises(ConfigError):
        record_z(SimpleNamespace(ip='192.168.2.232', output=path))
    assert path.read_text() == original
