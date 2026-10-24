from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from cloth_agent import one_shot_orientation as orientation, grasp_angle
from cloth_agent.codex_motion import parse_code
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as wire
from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
from cloth_agent.experiments import trajectory_memory_full_episode_v1 as shared
from cloth_agent import codex_recovery
from cloth_agent.pre_action_gate import PreActionGate


def proposal(angle=35.):
    return dict(decision='ACTION', grasp_px=[640, 360], target_px=[680, 360],
                motion_type='lift_pull_place', grasp_angle_deg=angle, short_reason='沿可见边缘夹取。')


@pytest.mark.parametrize('value', [-1, 181, float('nan'), float('inf'), True, '45'])
def test_invalid_angles_are_rejected(value):
    with pytest.raises((ValueError, wire.jsonschema.ValidationError)):
        wire.validate_plan(proposal(value), [1280, 720], require_grasp_rotation=True)


def test_rotation_field_required_live_and_null_on_non_actions():
    legacy = proposal(); legacy.pop('grasp_angle_deg')
    wire.validate_plan(legacy, [1280, 720])  # Original offline records remain readable.
    with pytest.raises(wire.jsonschema.ValidationError):
        wire.validate_plan(legacy, [1280, 720], require_grasp_rotation=True)
    for decision in ('DONE', 'ABSTAIN'):
        p = dict(proposal(), decision=decision, grasp_px=None, target_px=None, motion_type=None)
        with pytest.raises(ValueError):
            wire.validate_plan(p, [1280, 720])
        p['grasp_angle_deg'] = None
        wire.validate_plan(p, [1280, 720], require_grasp_rotation=True)


@pytest.mark.parametrize('angle', [None, 0., 45., 90., 180.])
def test_live_preparation_rotates_before_closing_and_keeps_pose_through_place(tmp_path, monkeypatch, angle):
    k = np.array([[700., 0, 640], [0, 700, 360], [0, 0, 1]])
    wc = np.eye(4); wc[:3, :3] = Rotation.from_euler('xyz', [180, 0, 25], degrees=True).as_matrix()
    wc[2, 3] = 1000.
    home = np.array([180., 0., 15.])
    a, b = np.array([0., 0., 30.]), np.array([40., 0., 30.])
    q = {'left': np.zeros(6), 'right': np.zeros(7)}
    source_seen = []
    def plan(source, initial, **kwargs):
        source_seen.append(source)
        assert kwargs['preserve_command_boundaries'] is True
        return [('gripper', args) if name == 'gripper' else ('move', args)
                for name, args in parse_code(source)]
    approach_seen = []
    runtime = SimpleNamespace(
        config=SimpleNamespace(limits={'max_surface_spread_mm': 8.}, path_validation_enabled=False,
            collision_envelopes_disabled_arms={'left'}, arms={'left': SimpleNamespace(
                tcp_offset=np.array([0., 0., 172., 0., 0., 20.]), world_from_base=np.eye(4), home_joints=q['left'])}),
        camera=SimpleNamespace(intrinsics=k),
        models={'left': SimpleNamespace(forward=lambda _: np.r_[0., 0., 500., home])},
        coordinator=SimpleNamespace(snapshots=lambda: {key: {'joints': val} for key, val in q.items()}),
        arrival=lambda _: None, plan=plan, plan_stats={},
        _inverse_target=lambda arm, pose, seed, phase: (approach_seen.append(pose) or np.zeros(6)))
    img = tmp_path/'rgb.png'; Image.new('RGB', (1280, 720)).save(img)
    before = SimpleNamespace(image=img, meta={'world_from_camera_mm': wc.tolist(), 'joints_deg': q},
                             sample=lambda *args: a, validate_for=lambda *args, **kwargs: None)
    def target_sample(*args):
        wire.save(tmp_path/'target_depth.json', {'source': 'MEASURED_DEPTH', 'measured_depth': True})
        return b
    monkeypatch.setattr(episode.table_target_depth, 'sample_target', target_sample)
    ops, initial = episode.prepare_action(runtime, before, proposal(angle), tmp_path)
    record = wire.read(tmp_path/'grasp_orientation.json')
    selected = np.array(record['selected_rpy_world_deg'])
    assert record['rotation_while_holding'] is False and abs(record['world_z_delta_deg']) <= 90
    np.testing.assert_allclose(Rotation.from_euler('xyz', approach_seen[0][3:], degrees=True).as_matrix(),
                               Rotation.from_euler('xyz', selected, degrees=True).as_matrix(), atol=1e-8)
    calls = parse_code(source_seen[0])
    close = next(i for i, (name, args) in enumerate(calls) if name == 'gripper' and args['action'] == 'close')
    assert calls[close-1][0] == 'move_tcp'  # descend with the already chosen orientation.
    for name, args in calls:
        if name == 'move_tcp':
            np.testing.assert_allclose(Rotation.from_euler('xyz', args['left'][3:]).as_matrix(),
                                       Rotation.from_euler('xyz', selected, degrees=True).as_matrix(), atol=1e-8)
    assert sum(name == 'move_tcp' for name, _ in calls[close+1:]) == 4  # lift, across, place, retreat.
    assert any(kind == 'gripper' and data['action'] == 'release_partial' for kind, data in ops)
    if angle is None:
        np.testing.assert_allclose(selected, home, atol=1e-10)
    else:
        axis_tcp = Rotation.from_euler('xyz', runtime.config.arms['left'].tcp_offset[3:], degrees=True).inv().apply([0, 1, 0])
        axis = Rotation.from_euler('xyz', selected, degrees=True).apply(axis_tcp)
        axis[2] = 0
        pixels = [grasp_angle.project(a+sign*30*axis, k, wc) for sign in (-1, 1)]
        delta = pixels[1]-pixels[0]
        measured = np.degrees(np.arctan2(delta[1], delta[0])) % 180
        assert abs((measured-angle+90) % 180-90) < 1e-6


def test_repeat_checks_distinguish_a_new_orientation_but_not_half_turn(tmp_path):
    path = tmp_path/'rgb.png'; Image.new('RGB', (100, 80)).save(path)
    old = proposal(0)
    history = [dict(plan=old, before=path, acquisition='ACQUISITION_FAILURE')]
    request = {'failures': [{'plan': old}]}
    assert shared.repeated_action(proposal(180), path, history)
    assert not shared.repeated_action(proposal(45), path, history)
    with pytest.raises(orientation.DualArmError):
        codex_recovery.reject_repeated(proposal(180), request)
    codex_recovery.reject_repeated(proposal(45), request)


def test_shadow_gate_receives_angle_and_draws_it(tmp_path):
    path = tmp_path/'rgb.png'; Image.new('RGB', (1280, 720)).save(path)
    def request(folder, body, headers, timeout):
        text = body['input'][0]['content'][0]['text']
        assert '"grasp_angle_deg": 35.0' in text
        assert '夹住后保持姿态' in body['instructions']
        return {'decision': 'TRUST'}, {'wall_latency_s': 0.01}
    gate = PreActionGate(tmp_path, {}, request)
    assert gate.evaluate(path, path, proposal()) == {'decision': 'TRUST'}
    assert gate.metrics['status'] == 'VALID'
    assert wire.read(tmp_path/'gate_input_manifest.json')['proposed_action']['grasp_angle_deg'] == 35.
    with Image.open(tmp_path/'gate_input_overlay.png') as im:
        pixels = np.asarray(im.crop((600, 320, 680, 400)))
        assert np.any(np.all(pixels == (128, 255, 128), axis=-1))
