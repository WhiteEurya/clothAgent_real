import json
import queue
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from cloth_agent.config import RobotConfig
from cloth_agent.perception import RGBDFrame
from scripts import live_tcp_overlay as script


def projection():
    # Camera looking down from base Z=700mm, with a rotated XY frame.
    transform = np.array([[0., 1, 0, .4], [1, 0, 0, .1], [0, 0, -1, .7], [0, 0, 0, 1]])
    k = np.array([[100., 0, 80], [0, 100, 60], [0, 0, 1]])
    return k, transform, script.FrozenCameraProjection(k, transform, (160, 120))


def test_projection_units_rotation_and_frozen_camera_pose():
    k, transform, camera = projection()
    expected = camera.project([450, 200, 200])
    assert expected['status'] == 'VISIBLE'
    assert expected['raw_pixel_xy'] == pytest.approx([100, 70])
    assert expected['camera_depth_m'] == pytest.approx(.5)
    k[0, 0] = 900
    transform[:3, 3] = [9, 9, 9]
    assert camera.project([450, 200, 200]) == expected
    # Height changes projection, even with the same base XY.
    assert camera.project([450, 200, 450])['raw_pixel_xy'] == pytest.approx([120, 80])


def test_behind_outside_and_invalid_points():
    _, _, camera = projection()
    assert camera.project([400, 100, 800]) == {'status': 'BEHIND_CAMERA', 'raw_pixel_xy': None}
    outside = camera.project([400, 2000, 200])
    assert outside['status'] == 'OUTSIDE_IMAGE'
    assert outside['raw_pixel_xy'][0] > 160
    with pytest.raises(ValueError):
        camera.project([np.nan, 0, 0])


def test_failure_or_stale_feedback_removes_marker():
    _, _, camera = projection()
    sample = {'status': 'OK', 'sample_monotonic': 100,
              'projection': camera.project([450, 200, 200])}
    assert script.overlay_status(sample, 100.1)[0] == 'VISIBLE'
    assert script.overlay_status(sample, 102) == ('STALE', None)
    assert script.overlay_status({'status': 'READ_ERROR'}, 100) == ('READ_ERROR', None)
    image = Image.new('RGB', (160, 120), 'black')
    rendered, _ = script.render_overlay(image, sample, 100.1)
    assert np.asarray(rendered).any()
    rendered, _ = script.render_overlay(image, sample, 102)
    assert not np.asarray(rendered).any()


def test_capture_must_be_stationary():
    before = [400, 100, 500, 0, 0, 179.9]
    script.check_capture_stationary(before, [400, 100, 500, 0, 0, -179.9])
    with pytest.raises(RuntimeError, match='moved during capture'):
        script.check_capture_stationary(before, [402, 100, 500, 0, 0, 179.9])
    with pytest.raises(RuntimeError):
        script.check_capture_stationary(before, [400, 100, 500, 0, 2, 179.9])


class ReadOnlyArm:
    connected = True
    tcp_offset = [0, 0, 172, 0, 0, 0]

    def __init__(self):
        self.calls = []

    def get_position(self, *, is_radian):
        self.calls.append('get_position')
        assert is_radian is False
        return 0, [450, 200, 200, 0, 0, 0]

    def disconnect(self):
        self.calls.append('disconnect')
        self.connected = False

    def __getattr__(self, name):
        raise AssertionError(f'Unexpected API in read-only monitoring: {name}')


def config():
    return RobotConfig.load(script.PROJECT_ROOT, script.PROJECT_ROOT / 'config/robot.example.json')


def test_read_only_poll_and_tcp_offset_mismatch():
    arm = ReadOnlyArm()
    _, _, camera = projection()
    sample = script.poll_sample(arm, config(), camera)
    assert sample['status'] == 'OK'
    assert sample['projection']['raw_pixel_xy'] == pytest.approx([100, 70])
    assert arm.calls == ['get_position']
    arm.tcp_offset = [0]*6
    assert script.poll_sample(arm, config(), camera)['status'] == 'READ_ERROR'


def test_latest_mailbox_drops_old_samples():
    latest = queue.Queue(maxsize=1)
    script.publish_latest(latest, {'sequence': 1})
    script.publish_latest(latest, {'sequence': 2})
    assert latest.get_nowait() == {'sequence': 2}


@pytest.mark.parametrize('capture_current', [True, False])
def test_complete_workflow_with_read_only_monitor(monkeypatch, tmp_path, capture_current):
    arm = ReadOnlyArm()
    fake_root = SimpleNamespace(withdraw=Mock(), destroy=Mock())
    monkeypatch.setitem(sys.modules, 'tkinter', SimpleNamespace(Tk=lambda: fake_root))
    monkeypatch.setitem(sys.modules, 'xarm', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'xarm.wrapper', SimpleNamespace(XArmAPI=lambda *a, **k: arm))
    observation = Mock(return_value={'sequence': ['home', 'perception_position']})
    monkeypatch.setattr(script, 'move_robot_to_perception_position', observation)
    k, transform, _ = projection()
    frame = RGBDFrame('A', 'test', np.zeros((120, 160, 3), np.uint8),
                     np.full((120, 160), .5), k, transform)
    monkeypatch.setattr(script, 'capture_two_view_rgbd', lambda cfg: [frame])

    def viewer(root, frame, latest, directory):
        sample = latest.get(timeout=2)
        assert sample['status'] == 'OK'
        assert script.overlay_status(sample, time.monotonic())[0] == 'VISIBLE'

    monkeypatch.setattr(script, 'run_viewer', viewer)
    args = ['--output-dir', str(tmp_path)]
    args += ['--capture-current'] if capture_current else ['--real', '--confirm-real']
    assert script.main(args) == 0
    assert observation.call_count == (0 if capture_current else 1)
    assert set(arm.calls) == {'get_position', 'disconnect'}
    directory, = tmp_path.iterdir()
    assert json.loads((directory / 'session.json').read_text())['status'] == 'CLOSED'
    records = [json.loads(line) for line in (directory / 'tcp_samples.jsonl').read_text().splitlines()]
    assert records[0]['projection']['status'] == 'VISIBLE'
    assert (directory / 'result.json').is_file()


def test_initial_motion_requires_flags(monkeypatch):
    move = Mock()
    monkeypatch.setattr(script, 'move_robot_to_perception_position', move)
    with pytest.raises(SystemExit) as exc:
        script.main([])
    assert exc.value.code == 2
    move.assert_not_called()
