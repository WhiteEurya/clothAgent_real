import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent import gamepad_collect as collect
from cloth_agent.codex_motion import parse_code
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.gamepad_continuous import ContinuousArm6
from cloth_agent.gamepad_teleop import InputGuard, Intent


def test_transfer_preserves_orientation_and_places_before_release():
    source = collect.transfer_program([100, 200, 150, 175, 0, 25], [100, 200, 100],
                                      [300, 250, 140], clearance=30, place_offset=5)
    calls = parse_code(source)
    assert [name for name, _ in calls] == ['move_tcp']*2 + ['gripper', 'move_tcp']
    poses = [np.r_[np.asarray(args['left'][:3])*1000, np.degrees(args['left'][3:])]
             for name, args in calls if name == 'move_tcp']
    assert [p[2] for p in poses] == pytest.approx([150, 145, 175])
    assert poses[0][:2] == pytest.approx([300, 250])
    for pose in poses:
        assert pose[3:] == pytest.approx([175, 0, 25])
    assert calls[2][1] == dict(arm='left', action='open')


@pytest.mark.parametrize('height', [100, 104.9])
def test_insufficient_manual_lift_blocks_transfer(height):
    with pytest.raises(DualArmError, match='低于放置高度'):
        collect.transfer_program([0, 0, height, 180, 0, 0], [0, 0, 100], [100, 0, 100],
                                 clearance=30, place_offset=5)


def handoff_backend(monkeypatch):
    monkeypatch.setattr('cloth_agent.gamepad_continuous.time.sleep', lambda _: None)
    obj = object.__new__(ContinuousArm6)
    obj.guard = InputGuard(); obj.guard.update(Intent())
    obj.closed = threading.Event(); obj.workers = []; obj.failure = None
    obj.runtime = SimpleNamespace(cancel=threading.Event(), prepared=True,
        coordinator=SimpleNamespace(event=lambda *a, **kw: None))
    obj.mode_entered = True; obj.latency = .05
    obj.fast = SimpleNamespace(mode=4, state=0)
    calls = []
    def mode(value):
        calls.append(('mode', value)); obj.fast.mode = value; return 0
    obj.arm = SimpleNamespace(vc_set_joint_velocity=lambda *a, **kw: calls.append(('zero', a[0])) or 0,
                              set_mode=mode, set_state=lambda x: calls.append(('state', x)) or 0)
    obj.gripper = SimpleNamespace(active=True)
    obj.gripper_api = SimpleNamespace(get_gripper_position=lambda: (0, 321),
        set_gripper_position=lambda value, **kw: calls.append(('hold', value)) or 0,
        disconnect=lambda: calls.append(('disconnect',)))
    obj._feedback = lambda: (np.zeros(6), None, 0)
    obj.stop = lambda: obj.runtime.cancel.set()
    return obj, calls


def test_handoff_holds_gripper_restores_mode_without_stop(monkeypatch):
    obj, calls = handoff_backend(monkeypatch)
    obj.handoff()
    assert ('zero', [0.]*6) in calls and ('hold', 321) in calls
    assert ('mode', 0) in calls and ('state', 4) not in calls
    assert not obj.runtime.cancel.is_set()
    assert not obj.runtime.prepared
    assert obj.gripper_api is None


def test_handoff_active_input_cancels(monkeypatch):
    obj, calls = handoff_backend(monkeypatch)
    obj.guard.update(Intent(enabled=True))
    with pytest.raises(DualArmError, match='neutral'):
        obj.handoff()
    assert obj.runtime.cancel.is_set()
    assert not calls


def test_handoff_moving_feedback_blocks_position_mode(monkeypatch):
    obj, calls = handoff_backend(monkeypatch)
    feedback = iter([np.zeros(6), np.ones(6)])
    obj._feedback = lambda: (next(feedback), None, 0)
    with pytest.raises(DualArmError, match='still moving'):
        obj.handoff()
    assert obj.runtime.cancel.is_set()
    assert ('mode', 0) not in calls


@pytest.mark.parametrize('fail,single', [(False, False), (True, False), (False, True)])
def test_iteration_order_and_manifest_on_failure(tmp_path, monkeypatch, fail, single):
    stages = []
    class Recorder:
        def __init__(self, *args):
            self.started = collect.time.monotonic(); self.error = None; self.samples = 10
        def start(self): pass
        def check(self): pass
        def close(self): pass
        def stage(self, stage): stages.append(stage)
    monkeypatch.setattr(collect, 'TrajectoryRecorder', Recorder)
    obs = SimpleNamespace(validate_for=lambda *a, **kw: None)
    from cloth_agent.dual_arm.execution import DualArmCoordinator
    coordinator = object.__new__(DualArmCoordinator)
    coordinator.directory = tmp_path/'old_removed_iteration'/'controller'
    coordinator.log_lock = threading.Lock(); coordinator.sequence = 0
    def observe(folder):
        coordinator.event('new_iteration_observation')
        stages.append('capture'); folder.mkdir(); (folder/'rgb.png').write_bytes(b'saved')
        return obs
    runtime = SimpleNamespace(calibration_id='test', cancel=threading.Event(),
        config=SimpleNamespace(limits={'max_surface_spread_mm': 8}), observe=observe,
        coordinator=coordinator,
        stop=lambda: stages.append('stop'), move_to_perception=lambda p: stages.append('perception'))
    monkeypatch.setattr(collect, 'select_points', lambda *a, **kw: [
        dict(name='start', world_xyz_mm=[0, 0, 100]), dict(name='target', world_xyz_mm=[100, 0, 100])][:1 if kw['single'] else 2])
    monkeypatch.setattr(collect, 'approach_program', lambda *a: 'approach')
    monkeypatch.setattr(collect, 'world_pose', lambda r: np.array([0, 0, 200, 180, 0, 0.]))
    def execute(r, recorder, folder, stage, source):
        stages.append(stage)
        if fail and stage == 'transfer_place': raise DualArmError('blocked')
    monkeypatch.setattr(collect, 'execute_program', execute)
    monkeypatch.setattr(collect, 'manual_stage', lambda *a: stages.append('manual'))
    args = SimpleNamespace(single_point=single, approach_rpy_deg=[180, 0, 0], approach_height_mm=50,
                           clearance_mm=30, place_offset_mm=5)
    folder = tmp_path/'traj'
    if fail:
        with pytest.raises(DualArmError): collect.run_iteration(runtime, None, args, folder)
    else:
        collect.run_iteration(runtime, None, args, folder)
    assert stages[:4] == ['capture', 'select_points', 'approach', 'manual']
    if single:
        assert stages == ['capture', 'select_points', 'approach', 'manual']
        assert (folder/'final_state.json').exists()
        assert len(json.loads((folder/'points.json').read_text())) == 1
    else:
        assert stages[4] == 'transfer_place'
    report = json.loads((folder/'traj.json').read_text())
    assert report['arm'] == 'left' and report['camera'] == 'A'
    assert report['status'] == ('FAILED' if fail else 'PENDING_CONFIRMATION')
    assert (folder/'observation/rgb.png').exists()
    assert (folder/'points.json').exists()
    assert coordinator.directory == folder/'controller'
    assert 'new_iteration_observation' in (folder/'controller/events.jsonl').read_text()
    if fail:
        assert 'perception' not in stages and runtime.cancel.is_set()
    elif not single:
        assert stages[-1] == 'perception'


def test_handoff_height_is_operator_selected_without_start_clearance():
    source = collect.transfer_program([0, 0, 110, 180, 0, 0], [0, 0, 100],
                                      [100, 0, 100], clearance=30, place_offset=5)
    calls = parse_code(source)
    assert calls[0][1]['left'][:3] == pytest.approx([.1, 0, .110])


def test_approach_solves_endpoint_in_base_and_uses_joint_path():
    transform = np.eye(4); transform[0, 3] = 100
    seed = np.arange(6.)
    captured = {}
    def inverse(arm, base, initial, phase):
        captured.update(arm=arm, base=base, seed=initial, phase=phase)
        return np.arange(6.) + 10
    runtime = SimpleNamespace(
        config=SimpleNamespace(arms={'left': SimpleNamespace(world_from_base=transform)}),
        coordinator=SimpleNamespace(snapshots=lambda: {'left': {'joints': seed}}),
        _inverse_target=inverse)
    calls = parse_code(collect.approach_program(runtime, [500, 200, 114, 180, 0, 0]))
    assert captured['arm'] == 'left'
    assert captured['base'][:3] == pytest.approx([400, 200, 114])
    assert captured['seed'] == pytest.approx(seed)
    assert [name for name, _ in calls] == ['gripper', 'move_joints']
    assert calls[1][1]['left'] == pytest.approx(np.radians(np.arange(6.) + 10))


@pytest.mark.parametrize('accepted', [True, False])
def test_confirmation_publishes_only_accepted_iteration(tmp_path, monkeypatch, accepted):
    from cloth_agent import collection_review
    folder = tmp_path/'pending'/'iter_0000'
    folder.mkdir(parents=True)
    collect.save_json(folder/'traj.json', dict(status='PENDING_CONFIRMATION', samples=10))
    (folder/'rgb.png').write_bytes(b'image')
    destination = tmp_path/'traj_0000'
    def confirm(review_folder):
        assert not destination.exists()
        assert review_folder == folder
        return accepted
    monkeypatch.setattr(collection_review, 'confirm_images', confirm)
    from cloth_agent.dual_arm.execution import DualArmCoordinator
    coordinator = object.__new__(DualArmCoordinator)
    coordinator.directory = folder/'controller'; coordinator.directory.mkdir()
    coordinator.log_lock = threading.Lock(); coordinator.sequence = 0
    coordinator.event('before_confirmation')
    assert collect.confirm_save_iteration(folder, destination, coordinator) is accepted
    coordinator.event('after_confirmation')
    assert destination.exists() is accepted
    saved = destination if accepted else folder
    manifest = json.loads((saved/'traj.json').read_text())
    assert manifest['status'] == ('COMPLETE' if accepted else 'REJECTED')
    assert manifest['operator_accepted'] is accepted
    assert (saved/'rgb.png').read_bytes() == b'image'
    assert coordinator.directory == saved/'controller'
    assert 'after_confirmation' in (saved/'controller/events.jsonl').read_text()


def test_place_release_stays_partial_through_retreat():
    moves = [('move', object()), ('move', object())]
    opening = ('gripper', dict(arm='left', action='open', joints={'left':[0]*6}))
    closing = ('gripper', dict(arm='left', action='close', joints={'left':[0]*6}))
    operations = moves + [opening, ('move', object())]
    result = collect.staged_place_release(operations)
    assert result[2][1]['action'] == 'release_partial'
    assert result[2][1]['opening_delta_pulse'] == 180
    assert result[2][1]['settle_s'] == .5
    assert result[3][0] == 'move'
    assert len(result) == 4
    assert opening[1]['action'] == 'open'
    with pytest.raises(DualArmError):
        collect.staged_place_release(moves + [opening, closing])


@pytest.mark.parametrize('recording_error', [None, 'feedback lost'])
def test_manual_only_records_without_automatic_actions(tmp_path, monkeypatch, recording_error):
    calls = []
    recorder = SimpleNamespace(started=collect.time.monotonic(), samples=12, error=None,
        start=lambda: calls.append('record'),
        close=lambda: setattr(recorder, 'error', recording_error))
    def check():
        if recorder.error:
            raise DualArmError(recorder.error)
    recorder.check = check
    monkeypatch.setattr(collect, 'TrajectoryRecorder', lambda *a: recorder)
    monkeypatch.setattr(collect, 'manual_stage', lambda *a: calls.append('manual'))
    monkeypatch.setattr(collect, 'world_pose', lambda *a: np.zeros(6))
    def unexpected(*a, **kw):
        pytest.fail('manual mode invoked an assisted action')
    monkeypatch.setattr(collect, 'select_points', unexpected)
    monkeypatch.setattr(collect, 'execute_program', unexpected)
    observation = SimpleNamespace(validate_for=lambda *a, **kw: None)
    runtime = SimpleNamespace(coordinator=None, calibration_id='test', config=object(),
        observe=lambda *a: observation, move_to_perception=unexpected)
    folder=tmp_path/'iteration'
    if recording_error:
        with pytest.raises(DualArmError, match=recording_error):
            collect.run_iteration(runtime, None, SimpleNamespace(manual_only=True), folder)
    else:
        collect.run_iteration(runtime, None, SimpleNamespace(manual_only=True), folder)
    manifest=json.loads((folder/'traj.json').read_text())
    assert manifest['status']==('FAILED' if recording_error else 'PENDING_CONFIRMATION')
    assert manifest['collection_mode']=='manual_only'
    assert calls==['record','manual']
    assert (folder/'final_state.json').exists()


def test_recorder_timestamps_after_acquiring_feedback_lock(monkeypatch):
    recorder = object.__new__(collect.TrajectoryRecorder)
    recorder.runtime = SimpleNamespace(config=SimpleNamespace(raw={
        'safety': {'max_feedback_age_s': .1}}))
    clock = [10.]
    connection = SimpleNamespace(_latest_report=None)
    class CallbackDuringLockAcquisition:
        def __enter__(self):
            clock[0] = 10.002
            connection._latest_report = ([0.]*6, [0.]*6, 10.001)
        def __exit__(self, *args): pass
    connection._report_lock = CallbackDuringLockAcquisition()
    monkeypatch.setattr(collect.time, 'monotonic', lambda: clock[0])
    q, pose, sampled, observed = recorder._read_feedback(connection)
    assert observed - sampled == pytest.approx(.001)


@pytest.mark.parametrize('sampled', [9.8, 10.1, float('nan')])
def test_recorder_rejects_bad_feedback_age_with_diagnostics(monkeypatch, sampled):
    recorder = object.__new__(collect.TrajectoryRecorder)
    recorder.runtime = SimpleNamespace(config=SimpleNamespace(raw={
        'safety': {'max_feedback_age_s': .1}}))
    connection = SimpleNamespace(_report_lock=threading.Lock(),
        _latest_report=([0.]*6,[0.]*6,sampled))
    monkeypatch.setattr(collect.time, 'monotonic', lambda: 10.)
    with pytest.raises(DualArmError, match='age_s=.*limit_s='):
        recorder._read_feedback(connection)


def test_automatic_pickup_geometry_and_command_order(monkeypatch):
    monkeypatch.setattr(collect, 'approach_program', lambda *a:
        'motion.gripper(arm="left", action="open")\nmotion.move_joints(left=[0,0,0,0,0,0])\n')
    source = collect.automatic_program(None, [100, 200, 30], [300, 200, 40], [180, 0, 0],
        approach_height=50, grasp_offset=-15, lift_height=50, clearance=30, place_offset=5)
    calls = parse_code(source)
    assert [name for name, _ in calls] == ['gripper', 'move_joints', 'move_tcp', 'gripper',
                                         'move_tcp', 'move_tcp', 'move_tcp', 'gripper', 'move_tcp']
    poses = [np.array(args['left'][:3])*1000 for name, args in calls if name == 'move_tcp']
    assert np.array(poses) == pytest.approx(np.array([[100,200,15], [100,200,90],
        [300,200,90], [300,200,45], [300,200,75]]))
    with pytest.raises(ValueError):
        collect.automatic_program(None, [100,200,30], [300,200,40], [180,0,0],
            approach_height=50, grasp_offset=-15, lift_height=4, clearance=30, place_offset=5)


@pytest.mark.parametrize('manual_grasp',[False,True])
def test_automatic_iteration_skips_gamepad_and_saves_after(tmp_path, monkeypatch,manual_grasp):
    monkeypatch.setattr('cloth_agent.grasp_angle.select_points_and_angle', lambda r, o, rpy, f:
        ([dict(world_xyz_mm=[100,200,30]), dict(world_xyz_mm=[300,200,30])],rpy,manual_grasp))
    stages = []
    class Recorder:
        def __init__(self, *args):
            self.started=collect.time.monotonic(); self.error=None; self.samples=10
        def start(self): pass
        def check(self): pass
        def close(self): pass
        def stage(self, name): stages.append(name)
    monkeypatch.setattr(collect, 'TrajectoryRecorder', Recorder)
    monkeypatch.setattr(collect, 'select_points', lambda *a, **kw: [
        dict(world_xyz_mm=[100,200,30]), dict(world_xyz_mm=[300,200,30])])
    monkeypatch.setattr(collect, 'automatic_program', lambda *a, **kw: 'auto-source')
    monkeypatch.setattr(collect, 'execute_program', lambda r, rec, folder, stage, source: stages.append(stage))
    monkeypatch.setattr(collect, 'manual_stage', lambda *a: pytest.fail('manual control must not run'))
    monkeypatch.setattr(collect, 'assisted_pickup', lambda *a: stages.append('gamepad_grasp'))
    monkeypatch.setattr(collect, 'world_pose', lambda *a: np.array([100,200,100,180,0,0]))
    def capture(folder):
        folder.mkdir(); (folder/'rgb.png').write_bytes(b'frame')
        return SimpleNamespace(validate_for=lambda *a, **kw: None)
    runtime=SimpleNamespace(calibration_id='test', config=SimpleNamespace(limits={'max_surface_spread_mm':8}),
        observe=capture, capture_current=capture, move_to_perception=lambda p: stages.append('home'),
        cancel=threading.Event(), stop=lambda: None)
    args=SimpleNamespace(auto_pick_place=True, approach_rpy_deg=[180,0,0], approach_height_mm=50,
        grasp_offset_mm=-15, lift_height_mm=50, clearance_mm=30, place_offset_mm=5)
    folder=tmp_path/'iteration'
    collect.run_iteration(runtime, None, args, folder)
    expected=['gamepad_grasp','transfer_place'] if manual_grasp else ['automatic_pick_place']
    assert stages == ['select_points',*expected,'return_perception','home','after_observation']
    assert (folder/'after/rgb.png').exists()
    manifest=json.loads((folder/'traj.json').read_text())
    assert manifest['collection_mode']==('assisted' if manual_grasp else 'automatic_pick_place')
    assert manifest['status']=='PENDING_CONFIRMATION'


def test_automatic_execution_preserves_preopen_and_partially_releases(tmp_path):
    operations=[('gripper',dict(arm='left',action='open')), ('move',{}),
                ('gripper',dict(arm='left',action='close')), ('move',{}),
                ('gripper',dict(arm='left',action='open')), ('move',{})]
    seen=[]
    runtime=SimpleNamespace(coordinator=SimpleNamespace(snapshots=lambda: {'left':{'joints':[0]*6}}),
        plan=lambda *a, **kw: operations, arrival=lambda *a: None,
        _execute_operations=lambda ops, folder: seen.extend(ops) or {'status':'EXECUTED'})
    recorder=SimpleNamespace(check=lambda: None,stage=lambda *a: None)
    collect.execute_program(runtime,recorder,tmp_path,'automatic_pick_place','source')
    grippers=[v for k,v in seen if k=='gripper']
    assert [v['action'] for v in grippers] == ['open','close','release_partial']
    assert grippers[-1]['opening_delta_pulse']==180
    assert seen[-1][0]=='move'


@pytest.mark.parametrize('fail',[False,True])
def test_fine_grasp_opens_gamepad_before_approach_and_closes_it(tmp_path,monkeypatch,fail):
    seen=[]
    monkeypatch.setattr(collect,'list_gamepads',lambda:[{'path':'pad'}])
    def connect(path):
        seen.append('connect');return SimpleNamespace(close=lambda:seen.append('disconnect'))
    monkeypatch.setattr(collect,'Gamepad',connect)
    monkeypatch.setattr(collect,'approach_program',lambda *a:'approach')
    monkeypatch.setattr(collect,'execute_program',lambda *a:seen.append('approach'))
    def manual(runtime,recorder,folder,pad,args,start,target):
        seen.append('manual')
        assert args.speed_mm_s==5.
        if fail:raise DualArmError('gamepad failed')
    monkeypatch.setattr(collect,'manual_stage',manual)
    args=SimpleNamespace(approach_height_mm=50,speed_mm_s=20,device=None)
    if fail:
        with pytest.raises(DualArmError):collect.assisted_pickup(None,None,tmp_path,args,np.array([0,0,30]),np.array([100,0,30]),[180,0,0])
    else:
        collect.assisted_pickup(None,None,tmp_path,args,np.array([0,0,30]),np.array([100,0,30]),[180,0,0])
    assert seen==['connect','approach','manual','disconnect']
    assert args.speed_mm_s==20


def test_missing_gamepad_prevents_fine_grasp_approach(tmp_path,monkeypatch):
    monkeypatch.setattr(collect,'list_gamepads',lambda:[])
    monkeypatch.setattr(collect,'execute_program',lambda *a:pytest.fail('must not move'))
    with pytest.raises(DualArmError):
        collect.assisted_pickup(None,None,tmp_path,SimpleNamespace(device=None),None,None,None)


def test_ui_random_reset_does_not_become_unfolding_demonstration(tmp_path,monkeypatch):
    class Recorder:
        def __init__(self,*a):self.started=collect.time.monotonic();self.error=None;self.samples=7;self.closed=0
        def start(self):pass
        def stage(self,*a):pass
        def check(self):pass
        def close(self):self.closed+=1
    rec=Recorder()
    monkeypatch.setattr(collect,'TrajectoryRecorder',lambda *a:rec)
    monkeypatch.setattr('cloth_agent.grasp_angle.select_points_and_angle',lambda *a:(None,[180,0,0],'random_reset'))
    calls=[]
    def reset(runtime,folder,**kwargs):
        assert kwargs['recorder'] is rec
        assert kwargs['lift_height']==300
        calls.append(folder)
    monkeypatch.setattr('cloth_agent.random_fold_reset.run_reset',reset)
    monkeypatch.setattr(collect,'execute_program',lambda *a:pytest.fail('manual point transfer must not run'))
    runtime=SimpleNamespace(calibration_id='test',observe=lambda p:SimpleNamespace(validate_for=lambda *a,**kw:None),
                            config=SimpleNamespace(),cancel=threading.Event(),stop=lambda:None)
    args=SimpleNamespace(auto_pick_place=True,approach_rpy_deg=[180,0,0],reset_radius_mm=30,
                         reset_lift_height_mm=300,grasp_offset_mm=-15)
    folder=tmp_path/'iter'
    collect.run_iteration(runtime,None,args,folder)
    manifest=json.loads((folder/'traj.json').read_text())
    assert manifest['status']=='RESET_COMPLETE'
    assert manifest['collection_mode']=='random_reset' and manifest['not_a_demonstration']
    assert rec.closed==1 and calls==[folder/'random_reset']
    assert not (folder/'points.json').exists()
