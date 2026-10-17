"""Offline batch protocol tests; no robot or RealSense connection."""
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cloth_agent.experiments import grasp_acquire_baseline as base
from cloth_agent.experiments import grasp_acquire_batch as batch


@pytest.fixture(autouse=True)
def offline_confirmation(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda *a: '')


def test_default_schedule_and_no_hardware(monkeypatch, capsys):
    monkeypatch.setattr(batch, 'run', lambda *a: pytest.fail('hardware initialized'))
    assert base.main(['batch']) == 0
    assert str(batch.schedule('compare9', 0)) in capsys.readouterr().out
    assert batch.schedule('shakedown', 0) == [0., -2., -4., -6.]
    with pytest.raises(ValueError, match='requires'):
        batch.schedule('balanced12', 0)


@pytest.mark.parametrize('seed', [0, 1, 42, 999])
def test_compare6_balanced_frozen_and_excludes_deeper_offsets(seed):
    values = batch.schedule('compare6', seed)
    assert len(values) == 6
    assert values.count(0.) == values.count(-2.) == 3
    assert set(values) == {0., -2.}
    assert values == batch.schedule('compare6', seed)


def test_compare6_requires_manual_reset_between_all_six_trials(tmp_path, monkeypatch):
    values = batch.schedule('compare6', 0)
    session = dict(offsets_mm=values, fixed_world_xy_mm=[300., 100.], trials=[])
    events = []
    monkeypatch.setattr('builtins.input', lambda q: events.append('RESET') or '')
    def trial(runtime, args, session, xy, offset, folder, report):
        events.append(offset)
        report.update(status='COMPLETED', returned_safely=True)
    batch.run_trials(runtime_mock([]), None, tmp_path, session, trial)
    assert events[::2] == values
    assert events[1::2] == ['RESET'] * 5
    assert len(session['trials']) == 6


def shakedown(tmp_path):
    session = dict(schedule='shakedown', offsets_mm=batch.OFFSETS, status='COMPLETED', trials=[])
    for offset in batch.OFFSETS:
        path = base.allocate_trial(tmp_path)
        base.save(path/'trial.json', dict(status='COMPLETED', returned_safely=True, operator_label=None))
        session['trials'].append(dict(trial=path.name, z_offset_mm=offset))
    base.save(tmp_path/'session.json', session)
    return session


def test_balanced12_requires_manual_pass_and_is_frozen_reproducible(tmp_path):
    shakedown(tmp_path)
    batch.review_shakedown(tmp_path, 'PASS')
    a = batch.schedule('balanced12', 42, tmp_path)
    assert len(a) == 12 and all(a.count(v) == 3 for v in batch.OFFSETS)
    assert a == batch.schedule('balanced12', 42, tmp_path)
    assert a != batch.schedule('balanced12', 43, tmp_path)
    with pytest.raises(ValueError, match='already'):
        batch.review_shakedown(tmp_path, 'PASS')
    p = tmp_path/'session.json'
    p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError, match='matching'):
        batch.schedule('balanced12', 42, tmp_path)


def test_incomplete_and_failed_shakedown_cannot_enable_expansion(tmp_path):
    session = shakedown(tmp_path)
    session['status'] = 'FAILED'
    base.save(tmp_path/'session.json', session)
    with pytest.raises(ValueError, match='completed'):
        batch.review_shakedown(tmp_path, 'PASS')
    session['status'] = 'COMPLETED'
    base.save(tmp_path/'session.json', session)
    batch.review_shakedown(tmp_path, 'FAIL')
    with pytest.raises(ValueError, match='matching'):
        batch.schedule('balanced12', 42, tmp_path)


def cloud():
    y, x = np.mgrid[:15, :15]
    return np.stack([x*1., y*1., np.full_like(x, 60.)], axis=-1)


def test_fixed_xy_uses_fresh_cloud_not_original_pixel():
    xyz = cloud()
    obs = SimpleNamespace(xyz=xyz)
    assert batch.fixed_xy_pixel(obs, [7., 7.])[0] == [7, 7]
    xyz[:, :, 0] += 2  # camera/cloud moved; physical XY must stay fixed
    assert batch.fixed_xy_pixel(obs, [7., 7.])[0] == [5, 7]
    xyz[:] = np.nan
    with pytest.raises(ValueError, match='no measured'):
        batch.fixed_xy_pixel(obs, [7., 7.])


def test_depth_hole_cannot_silently_jump_to_distant_point():
    xyz = cloud();xyz[3:12, 3:12] = np.nan
    with pytest.raises(ValueError, match='no measured'):
        batch.fixed_xy_pixel(SimpleNamespace(xyz=xyz), [7., 7.])


@pytest.mark.parametrize('offset', batch.OFFSETS)
def test_each_trial_uses_fresh_surface_exact_offset_and_fixed_xy(tmp_path, monkeypatch, offset):
    seen = []
    def prepare(runtime, before, xyz, yaw, folder):
        seen.append(xyz)
        pose = [*(np.array(xyz)/1000), np.pi, 0., 0.]
        lift = pose.copy();lift[2] += .025
        return 'source', pose, lift
    monkeypatch.setattr(base, 'prepare_source', prepare)
    m = dict(sample_error=None, surface_xyz_mm=[301., 99.5, 63.])
    batch.prepare_batch_source(SimpleNamespace(grasp_offset_mm=0), None, m, [300.,100.], offset, 0., tmp_path)
    assert seen == [[300.,100.,63.+offset]]
    assert m['command_grasp_z_mm'] == pytest.approx(63.+offset)
    assert np.array(m['command_lift_world_m_rad'][:3])-m['command_grasp_world_m_rad'][:3] == pytest.approx([0,0,.025])


def test_existing_height_adjustment_cannot_be_stacked_or_bypassed(tmp_path, monkeypatch):
    def prepare(*a):
        return 'source', [.3,.1,.055,np.pi,0,0], [.3,.1,.080,np.pi,0,0]
    monkeypatch.setattr(base, 'prepare_source', prepare)
    m = dict(sample_error=None, surface_xyz_mm=[300.,100.,60.])
    with pytest.raises(ValueError, match='no policy bypass'):
        batch.prepare_batch_source(SimpleNamespace(grasp_offset_mm=3), None, m, [300.,100.], -2, 0, tmp_path)


def test_sampler_or_patch_registration_failure_stops():
    with pytest.raises(ValueError, match='sampler'):
        batch.target_xyz(dict(sample_error='insufficient depth'), [300.,100.], -2)
    with pytest.raises(ValueError, match='does not represent'):
        batch.target_xyz(dict(sample_error=None, surface_xyz_mm=[310.,100.,60.]), [300.,100.], -2)


def runtime_mock(events, fail=None):
    def execute(ops, folder):
        events.append(folder.name)
        return dict(status='EXECUTION_FAILED' if folder.name==fail else 'EXECUTED', error='native rejection')
    return SimpleNamespace(camera=SimpleNamespace(check=lambda:None, stage=''),
        _execute_operations=execute, coordinator=SimpleNamespace(snapshots=lambda:{}))


def test_checkpoints_close_lift_and_no_online_label(tmp_path):
    events = [];runtime = runtime_mock(events)
    def photo(runtime, folder, hold):
        events.append((folder.name, hold));folder.mkdir()
        Image.new('RGB',(8,8)).save(folder/'rgb.png')
    phases = [(n,[]) for n in ['pregrasp_descent','close','lift','replace','release_retreat','return_home']]
    report = dict(phases=[], operator_label=None)
    batch.dispatch_batch(runtime, phases, tmp_path, report, photo)
    assert events == ['pregrasp_descent','close',('close_observation',0.),'lift',('lift_observation',2.),
                      'replace','release_retreat','return_home']
    assert report['operator_label'] is None and report['returned_safely']
    assert (tmp_path/'close.png').exists() and (tmp_path/'lift.png').exists()


@pytest.mark.parametrize('failure', ['close','lift','replace','release_retreat','return_home'])
def test_each_native_failure_stops_without_following_phase(tmp_path, failure):
    events = [];runtime = runtime_mock(events, failure)
    def photo(runtime, folder, hold):
        folder.mkdir();Image.new('RGB',(8,8)).save(folder/'rgb.png')
    names = ['pregrasp_descent','close','lift','replace','release_retreat','return_home']
    with pytest.raises(RuntimeError, match='native rejection'):
        batch.dispatch_batch(runtime, [(n,[]) for n in names], tmp_path, {'phases':[]}, photo)
    assert events == names[:names.index(failure)+1]


def test_checkpoint_stationary_failure_is_not_swallowed(tmp_path):
    events = [];runtime = runtime_mock(events)
    def photo(*a):
        raise RuntimeError('Robot moved while taking hold image')
    with pytest.raises(RuntimeError, match='Robot moved'):
        batch.dispatch_batch(runtime, [('close',[]),('lift',[])], tmp_path, {'phases':[]}, photo)
    assert events == ['close']


def test_failed_trial_aborts_entire_sequence_and_preserves_logs(tmp_path, monkeypatch):
    prompts=[]
    monkeypatch.setattr('builtins.input',lambda q:prompts.append(q) or '')
    runtime = runtime_mock([])
    session = dict(offsets_mm=batch.OFFSETS, fixed_world_xy_mm=[300.,100.], trials=[])
    calls = []
    def trial(runtime,args,session,xy,offset,folder,report):
        calls.append(offset)
        if offset == -2:
            raise RuntimeError('workspace reject')
        report.update(status='COMPLETED',returned_safely=True)
    with pytest.raises(RuntimeError, match='workspace reject'):
        batch.run_trials(runtime,None,tmp_path,session,trial)
    assert calls == [0., -2.]
    assert len(session['trials']) == 2
    failed = json.loads((tmp_path/'trial_000002/trial.json').read_text())
    assert failed['status'] == 'FAILED' and failed['operator_label'] is None
    assert not (tmp_path/'trial_000003').exists()
    assert len(prompts)==1  # never prompt/continue after the failed trial


def test_all_four_require_reset_before_each_following_trial(tmp_path, monkeypatch):
    events=[]
    session = dict(offsets_mm=batch.OFFSETS, fixed_world_xy_mm=[300.,100.], trials=[])
    def confirm(prompt):
        assert session['status']=='WAITING_FOR_MANUAL_RESET'
        assert session['trials'][-1]['status']=='COMPLETED'
        events.append('RESET')
        return ''
    monkeypatch.setattr('builtins.input',confirm)
    def trial(runtime,args,session,xy,offset,folder,report):
        events.append(offset)
        report.update(status='COMPLETED',returned_safely=True)
    batch.run_trials(runtime_mock([]),None,tmp_path,session,trial)
    assert len(session['trials']) == 4
    assert all(row['status']=='COMPLETED' for row in session['trials'])
    assert events==[0.,'RESET',-2.,'RESET',-4.,'RESET',-6.]
    assert len(session['manual_resets'])==3
    assert all(r['confirmation']=='ENTER_HANDS_CLEAR' for r in session['manual_resets'])


def test_session_camera_single_owner_fresh_frames_and_playable_video(tmp_path):
    import cv2
    class Camera:
        intrinsics = np.eye(3)
        def __init__(self): self.count=0;self.threads=set()
        def start(self): pass
        def stop(self): pass
        def read(self):
            self.threads.add(threading.get_ident());time.sleep(.01);self.count+=1
            return np.full((48,64,3), self.count%255,np.uint8),np.full((48,64),self.count,dtype=float)
    raw = Camera();camera = batch.SessionCamera(raw,tmp_path,30,threading.Event())
    camera.start()
    try:
        _,a=camera.read();_,b=camera.read()
        assert b[0,0]>a[0,0]
        assert raw.threads == {camera.thread.ident}
    finally:
        camera.stop()
    camera.stop()  # idempotent shutdown
    video=cv2.VideoCapture(str(tmp_path/'session.mp4'))
    try: assert video.read()[0]
    finally: video.release()
    manifest=json.loads((tmp_path/'recording.json').read_text())
    assert manifest['frames'] >= 2 and manifest['status']=='RECORDED'
    assert len((tmp_path/'frame_times.jsonl').read_text().splitlines()) == manifest['frames']


def test_camera_failure_signals_cancellation(tmp_path):
    class Camera:
        def start(self): pass
        def stop(self): pass
        def read(self): raise RuntimeError('B disconnected')
    cancel=threading.Event();camera=batch.SessionCamera(Camera(),tmp_path,30,cancel)
    try:
        with pytest.raises(RuntimeError,match='B disconnected'):
            camera.read()
        assert cancel.is_set()
    finally: camera.stop()


def test_review_video_marks_missing_checkpoint_without_fabricating_image(tmp_path):
    import cv2
    trial = base.allocate_trial(tmp_path)
    Image.new('RGB',(64,48),'red').save(trial/'before.png')
    session = dict(trials=[dict(trial=trial.name,z_offset_mm=-2.,status='FAILED')])
    batch.make_review(tmp_path,session)
    video=cv2.VideoCapture(str(tmp_path/'review.mp4'))
    try:
        assert int(video.get(cv2.CAP_PROP_FRAME_COUNT))==80
        assert video.read()[0]
    finally:video.release()
    assert not (trial/'lift.png').exists()


def test_full_four_trial_flow_refreshes_depth_and_tolerates_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr('builtins.input',lambda *a:'')
    observed=[];surfaces=[];commands=[];waits=[];photos=[]
    runtime=runtime_mock([])
    runtime.cancel=SimpleNamespace(wait=lambda seconds:waits.append(seconds) or False)
    runtime.camera.intrinsics=np.eye(3)
    runtime.config=SimpleNamespace(limits={'max_surface_spread_mm':8.})
    runtime.grasp_offset_mm=0.
    runtime.arrival=lambda q:None
    def observe(folder):
        observed.append(folder.name);folder.mkdir()
        Image.new('RGB',(15,15)).save(folder/'rgb.png')
        return SimpleNamespace(directory=folder,image=folder/'rgb.png',xyz=cloud(),
            meta={'joints_deg':{'right':[0.]}},validate_for=lambda *a,**kw:None)
    runtime.observe=observe
    def measure(*args):
        z=60.+len(surfaces);surfaces.append(z)
        return dict(sample_error=None,surface_xyz_mm=[7.,7.,z],current_surface_z_mm=z,A={})
    monkeypatch.setattr(base,'measure_pa',measure)
    def fail_a(*args): raise RuntimeError("Couldn't resolve requests")
    monkeypatch.setattr(base,'capture_a',fail_a)
    def prepare(runtime,before,xyz,yaw,folder):
        commands.append(xyz)
        grasp=[*(np.array(xyz)/1000),np.pi,0.,0.]
        lift=grasp.copy();lift[2]+=.025
        return 'source',grasp,lift
    monkeypatch.setattr(base,'prepare_source',prepare)
    phases=[(n,[]) for n in ['pregrasp_descent','close','lift','replace','release_retreat','return_home']]
    monkeypatch.setattr(base,'plan_trial',lambda *a:phases)
    original=batch.dispatch_batch
    def photo(runtime,folder,hold):
        photos.append((folder.name,hold));folder.mkdir()
        Image.new('RGB',(15,15)).save(folder/'rgb.png')
    monkeypatch.setattr(batch,'dispatch_batch',lambda *args:original(*args,photo_fn=photo))
    session=dict(offsets_mm=batch.OFFSETS,fixed_world_xy_mm=[7.,7.],trials=[])
    batch.run_trials(runtime,SimpleNamespace(config='unused',yaw_deg=0),tmp_path,session)
    assert observed==['before','after_reset_observation']*4
    assert surfaces==[60.,61.,62.,63.]
    assert commands==[[7.,7.,60.],[7.,7.,59.],[7.,7.,58.],[7.,7.,57.]]
    assert waits==[2.]*4
    assert photos==[('close_observation',0.),('lift_observation',2.)]*4
    for row in session['trials']:
        trial=tmp_path/row['trial']
        report=json.loads((trial/'trial.json').read_text())
        assert report['status']=='COMPLETED' and report['operator_label'] is None
        assert "Couldn't resolve" in report['measurement']['A']['reason']
        for stage in ['before','close','lift','after_reset']:
            assert (trial/f'{stage}.png').exists()
    batch.label_worksheet(tmp_path,session)
    assert len((tmp_path/'labels_to_fill.csv').read_text().splitlines())==5


def test_batch_orchestrator_one_confirmation_and_stops_on_failure(tmp_path, monkeypatch):
    from cloth_agent import codex_arm7_runtime
    from cloth_agent.dual_arm import observation
    events=[];prompts=[]
    class Camera:
        error=None
        def __init__(self,*args):pass
        def start(self):events.append('camera_start')
        def read(self):events.append('frame')
        def stop(self):events.append('camera_stop')
    class Runtime:
        def __init__(self,*args):
            self.config=SimpleNamespace(raw={'camera':{'fps':30}},limits={'max_surface_spread_mm':8})
            self.cancel=threading.Event()
        def observe(self,folder):
            events.append('observe');return SimpleNamespace(image='image')
        def stop(self):events.append('STOP');return {'right':'sent'}
        def close(self):events.append('CLOSE');self.camera.stop()
    monkeypatch.setattr(codex_arm7_runtime,'Arm7Runtime',Runtime)
    monkeypatch.setattr(observation,'create_camera',lambda *a:object())
    monkeypatch.setattr(batch,'SessionCamera',Camera)
    monkeypatch.setattr(base,'low_speed',lambda *a:{})
    monkeypatch.setattr(base,'select_pa',lambda *a:[7,7])
    monkeypatch.setattr(base,'measure_pa',lambda *a:dict(sample_error=None,surface_xyz_mm=[7.,7.,60.]))
    monkeypatch.setattr('builtins.input',lambda q:prompts.append(q) or 'YES')
    def fail(*a):raise RuntimeError('native workspace rejected')
    monkeypatch.setattr(batch,'run_trials',fail)
    args=SimpleNamespace(schedule='shakedown',random_seed=0,shakedown=None,output=tmp_path,config='unused')
    assert batch.run(args)==1
    assert len(prompts)==1 and events.count('STOP')==1 and events[-2:]==['CLOSE','camera_stop']
    report=json.loads(next(tmp_path.glob('batch_*/session.json')).read_text())
    assert report['status']=='FAILED' and report['safety_confirmation']['response']=='YES'
    assert report['fixed_world_xy_mm']==[7.,7.]


@pytest.mark.parametrize('response', ['q', 'EOF', 'CTRL_C'])
def test_manual_reset_cancel_never_starts_next_trial(tmp_path, monkeypatch, response):
    session=dict(offsets_mm=batch.OFFSETS,fixed_world_xy_mm=[7.,7.],trials=[])
    calls=[]
    def trial(runtime,args,session,xy,offset,folder,report):
        calls.append(offset);report.update(status='COMPLETED',returned_safely=True)
    def answer(*a):
        if response=='EOF':raise EOFError()
        if response=='CTRL_C':raise KeyboardInterrupt()
        return response
    monkeypatch.setattr('builtins.input',answer)
    with pytest.raises((KeyboardInterrupt,EOFError)):
        batch.run_trials(runtime_mock([]),None,tmp_path,session,trial)
    assert calls==[0.]
    assert not (tmp_path/'trial_000002').exists()
    assert session['manual_resets'][0]['status']=='CANCELLED'
    assert json.loads((tmp_path/'trial_000001/trial.json').read_text())['status']=='COMPLETED'


def test_manual_reset_rejects_unrecognized_response_without_motion(tmp_path, monkeypatch):
    session={};answers=iter(['not yet',''])
    calls=[]
    def answer(q):
        calls.append(q);return next(answers)
    monkeypatch.setattr('builtins.input',answer)
    events=[]
    batch.wait_for_manual_reset(runtime_mock(events),tmp_path,session,
        dict(trial='trial_000001',status='COMPLETED',returned_safely=True),-2.)
    assert len(calls)==2 and events==[]
    assert session['status']=='RUNNING'


@pytest.mark.parametrize('seed', [0,1,42])
def test_compare9_includes_minus4_but_never_minus6(seed):
    values=batch.schedule('compare9',seed)
    assert len(values)==9 and set(values)=={0.,-2.,-4.}
    assert all(values.count(x)==3 for x in (0.,-2.,-4.))
    assert values==batch.schedule('compare9',seed)


@pytest.mark.parametrize('answer', ['', 'q', 'EOF'])
def test_release_confirmation_holds_before_replace_and_open(tmp_path,monkeypatch,answer):
    events=[];runtime=runtime_mock(events)
    def photo(runtime,folder,hold):
        folder.mkdir();Image.new('RGB',(8,8)).save(folder/'rgb.png');events.append('PHOTO')
    def confirm(prompt):
        assert events==['close','PHOTO','lift','PHOTO']
        assert (tmp_path/'lift.png').exists()
        assert not (tmp_path/'replace').exists()
        events.append('CONFIRM')
        if answer=='EOF':raise EOFError()
        return answer
    monkeypatch.setattr('builtins.input',confirm)
    report={'phases':[]}
    phases=[(n,[]) for n in ['close','lift','replace','release_retreat','return_home']]
    if answer=='':
        batch.dispatch_batch(runtime,phases,tmp_path,report,photo)
        assert events[-3:]==['replace','release_retreat','return_home']
        assert report['release_confirmation']['status']=='CONFIRMED'
    else:
        with pytest.raises((KeyboardInterrupt,EOFError)):
            batch.dispatch_batch(runtime,phases,tmp_path,report,photo)
        assert events[-1]=='CONFIRM' and not (tmp_path/'replace').exists()
        assert report['release_confirmation']['status']=='CANCELLED'
