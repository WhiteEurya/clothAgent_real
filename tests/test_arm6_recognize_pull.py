from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np
import pytest
from PIL import Image
from cloth_agent.experiments import arm6_recognize_pull as experiment
from cloth_agent.experiments import arm6_corner_pull as pull
from cloth_agent.dual_arm.geometry import DualArmError


def observation(tmp_path):
    p=tmp_path/'rgb.png';Image.new('RGB',(300,300),'purple').save(p)
    return SimpleNamespace(image=p,validate_for=lambda *a,**k:None,
        sample=lambda xy,spread:np.array([xy[0],xy[1],30.]))


def recognition(state,confidence=.95):
    return dict(state='RECOGNIZABLE' if state=='DONE' else state,unfolded=state=='DONE',
                unfolded_evidence=['sleeves and hem extended'] if state=='DONE' else [],confidence=confidence,state_description='visible arrangement',visible_parts=['fabric edge'],
                occluded_or_uncertain_parts=['neckline'],reason='evidence')


class Media:
    def begin(self,*a):pass
    def phase(self,*a):pass
    def snapshot(self,*a):pass
    def edge(self,*a):pass


@pytest.mark.parametrize('sequence,limit,expected,pulls',[
    (['DONE'],3,'MODEL_REPORTED_UNFOLDED',0),
    (['UNRECOGNIZABLE','RECOGNIZABLE','DONE'],3,'MODEL_REPORTED_UNFOLDED',2),
    (['UNRECOGNIZABLE']*3,2,'STEP_LIMIT',2)])
def test_loop_reassesses_after_each_pull(tmp_path,monkeypatch,sequence,limit,expected,pulls):
    obs=observation(tmp_path);calls=[];it=iter(sequence)
    def ask(kind,payload,images):
        calls.append(kind);return recognition(next(it))
    def move(runtime,before,requests,path,media,result,**kw):
        calls.append('pull');result['physical_actions']+=1;result['pull_actions']+=1
        if kw['kind']=='unknown_pull':result['exploration_pulls']=result.get('exploration_pulls',0)+1
        return 'EXECUTED',obs,dict(before=str(obs.image),proposal={})
    monkeypatch.setattr(experiment,'run_once',move)
    runtime=SimpleNamespace(observe=lambda p:obs,cancel=threading.Event())
    result=dict(steps=[],recognitions=[],physical_actions=0,pull_actions=0)
    experiment.loop(runtime,SimpleNamespace(ask=ask),tmp_path,Media(),result,max_steps=limit)
    assert result['status']==expected and result['pull_actions']==pulls
    assert calls==['recognize','pull']*pulls+['recognize']


def test_uncertain_recognition_does_not_trigger_motion(tmp_path,monkeypatch):
    obs=observation(tmp_path)
    monkeypatch.setattr(experiment,'run_once',lambda *a,**kw:pytest.fail('must not pull'))
    result=dict(steps=[],recognitions=[],physical_actions=0,pull_actions=0)
    runtime=SimpleNamespace(observe=lambda p:obs,cancel=threading.Event())
    experiment.loop(runtime,SimpleNamespace(ask=lambda *a:recognition('UNRECOGNIZABLE',.4)),tmp_path,Media(),result)
    assert result['status']=='RECOGNITION_UNCERTAIN'


def proposal():return dict(decision='ACT',confidence=.95,grasp_px=[100,100],target_px=[140,100],
    opening_angle_image_deg=0,visible_corner_evidence='exposed hem',reason='pull open',expected_change='more visible')


def test_depth_and_distance_rejection_never_plan(tmp_path):
    obs=observation(tmp_path);runtime=SimpleNamespace(config=SimpleNamespace(limits={'max_surface_spread_mm':8}))
    p=proposal();p['target_px']=[250,100]
    with pytest.raises(DualArmError,match='distance'):pull.prepare(runtime,obs,p,tmp_path/'long')
    obs.sample=lambda *a:np.array([np.nan,0,0])
    with pytest.raises(DualArmError,match='measured depth'):pull.prepare(runtime,obs,proposal(),tmp_path/'depth')


@pytest.mark.parametrize('fault,expected',[('stop','UNKNOWN_PULL_MODEL_STOP'),('confidence','UNKNOWN_PULL_UNCERTAIN'),
    ('geometry','UNKNOWN_PULL_GEOMETRY_REJECTED'),('cancel',None)])
def test_blocked_pull_never_dispatches(tmp_path,monkeypatch,fault,expected):
    obs=observation(tmp_path);p=proposal();cancel=threading.Event()
    runtime=SimpleNamespace(cancel=cancel)
    monkeypatch.setattr(pull,'observation_context',lambda *a:dict(host_checks=[]))
    def ask(*a):
        if fault=='stop':p['decision']='STOP'
        if fault=='confidence':p['confidence']=.3
        if fault=='cancel':cancel.set()
        return p
    def prepare(*a,**kw):
        if fault=='geometry':raise DualArmError('no depth')
        pytest.fail('should not plan')
    monkeypatch.setattr(pull,'prepare',prepare)
    if fault=='cancel':
        with pytest.raises(KeyboardInterrupt):pull.run_once(runtime,obs,SimpleNamespace(ask=ask),tmp_path,Media(),{})
    else:assert pull.run_once(runtime,obs,SimpleNamespace(ask=ask),tmp_path,Media(),{})[0]==expected


def test_execution_error_is_not_retried(tmp_path,monkeypatch):
    obs=observation(tmp_path);calls=[]
    monkeypatch.setattr(pull,'observation_context',lambda *a:dict(host_checks=[]))
    monkeypatch.setattr(pull,'prepare',lambda *a,**kw:([],{}))
    class Recorder:
        def __init__(self,*a):pass
        def start(self):pass
        def check(self):pass
        def stage(self,*a):pass
        def close(self):calls.append('close_recorder')
    monkeypatch.setattr(pull,'TrajectoryRecorder',Recorder)
    def fail(*a):calls.append('dispatch');raise DualArmError('controller fault')
    runtime=SimpleNamespace(cancel=threading.Event(),arrival=lambda p:None,_execute_operations=fail)
    with pytest.raises(DualArmError,match='controller fault'):
        pull.run_once(runtime,obs,SimpleNamespace(ask=lambda *a:proposal()),tmp_path,Media(),dict(steps=[],physical_actions=0))
    assert calls==['dispatch','close_recorder']


def test_offline_mode_never_creates_runtime(tmp_path,monkeypatch):
    from cloth_agent import gamepad_collect_runtime
    obs=observation(tmp_path);out=tmp_path/'offline'
    monkeypatch.setattr(gamepad_collect_runtime,'CollectionRuntime',lambda *a:pytest.fail('no hardware offline'))
    requests=SimpleNamespace(ask=lambda *a:recognition('UNRECOGNIZABLE'),metrics=lambda:{})
    monkeypatch.setattr(experiment,'make_requests',lambda *a:requests)
    assert experiment.main(['--image',str(obs.image),'--output',str(out)])==0
    import json
    result=json.loads((out/'result.json').read_text())
    assert result['physical_actions']==0 and result['proposed_next']=='PULL'


def test_exploration_limit_stops_without_extra_motion(tmp_path,monkeypatch):
    obs=observation(tmp_path)
    monkeypatch.setattr(experiment,'run_once',lambda *a,**kw:pytest.fail('limit exceeded'))
    result=dict(steps=[],recognitions=[],physical_actions=1,pull_actions=1,exploration_pulls=1)
    runtime=SimpleNamespace(observe=lambda p:obs,cancel=threading.Event())
    experiment.loop(runtime,SimpleNamespace(ask=lambda *a:recognition('UNRECOGNIZABLE')),
                    tmp_path,Media(),result,max_exploration_pulls=1)
    assert result['status']=='EXPLORATION_LIMIT'


@pytest.mark.parametrize('kind',['unknown_pull','unfold','skill'])
def test_pull_returns_home_and_captures_before_reassessment(tmp_path,monkeypatch,kind):
    obs=observation(tmp_path);calls=[]
    monkeypatch.setattr(pull,'observation_context',lambda *a:dict(host_checks=[]))
    monkeypatch.setattr(pull,'prepare',lambda *a,**kw:([],{}))
    class Recorder:
        def __init__(self,*a):pass
        def start(self):pass
        def check(self):pass
        def stage(self,*a):pass
        def close(self):calls.append('close_recorder')
    monkeypatch.setattr(pull,'TrajectoryRecorder',Recorder)
    def execute(*a):calls.append('execute');return dict(status='EXECUTED')
    runtime=SimpleNamespace(cancel=threading.Event(),arrival=lambda p:None,_execute_operations=execute,
        move_to_perception=lambda p:calls.append('home'),capture_current=lambda p:calls.append('photo') or obs)
    requests=SimpleNamespace(ask=lambda k,*a:calls.append(k) or proposal())
    result=dict(steps=[],physical_actions=0)
    if kind=='skill':
        status,after,history=pull.execute_proposal(runtime,obs,proposal(),tmp_path,Media(),result,kind=kind)
    else:
        status,after,history=pull.run_once(runtime,obs,requests,tmp_path,Media(),result,kind=kind)
    assert status=='EXECUTED' and after is obs
    assert calls==([] if kind=='skill' else [kind])+['execute','home','close_recorder','photo']
    assert result['physical_actions']==1 and runtime.open_at_home_on_return is False
    assert result[{'unknown_pull':'exploration_pulls','unfold':'unfold_actions','skill':'skill_actions'}[kind]]==1


def test_prepare_plans_complete_motion_and_stages_only_final_open(tmp_path,monkeypatch):
    obs=observation(tmp_path);seen=[]
    arm=SimpleNamespace(world_from_base=np.eye(4),home_joints=np.zeros(6),tcp_offset=np.zeros(6))
    def program(runtime,start,target,chosen,**kw):
        seen.append(kw);assert np.linalg.norm(target[:2]-start[:2])==40
        return 'whole_motion'
    monkeypatch.setattr(pull,'automatic_program',program)
    monkeypatch.setattr(pull,'transform_for',lambda o:np.eye(4))
    monkeypatch.setattr(pull,'image_angle_to_rpy',lambda *a:(np.array([180.,0,0]),0.))
    monkeypatch.setattr(pull,'draw_preview',lambda *a:None)
    def plan(source,initial,**kw):
        assert source=='whole_motion' and kw['preserve_command_boundaries'] is True
        return [('gripper',dict(arm='left',action=a)) for a in ['open','close','open']]
    runtime=SimpleNamespace(config=SimpleNamespace(arms={'left':arm},limits={'max_surface_spread_mm':8}),
        models={'left':SimpleNamespace(forward=lambda q:[100,100,200,180,0,0])},
        camera=SimpleNamespace(intrinsics=np.eye(3)),plan=plan,
        coordinator=SimpleNamespace(snapshots=lambda:{'left':dict(joints=np.zeros(6))}))
    operations,initial=pull.prepare(runtime,obs,proposal(),tmp_path)
    assert [data['action'] for kind,data in operations]==['open','close','release_partial']
    assert operations[-1][1]['opening_delta_pulse']==180
    assert seen[0]['lift_height']==50 and seen[0]['grasp_offset']==-15
