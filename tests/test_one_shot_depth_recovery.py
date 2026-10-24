from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np
from PIL import Image
import pytest
from cloth_agent import one_shot_depth_recovery as r
from cloth_agent.dual_arm.observation import Observation,DepthUnavailableError


def make_obs(folder,missing=False):
    folder.mkdir(parents=True)
    Image.new('RGB',(100,80),'purple').save(folder/'rgb.png')
    xyz=np.zeros((80,100,3),dtype=float);xyz[:,:,2]=30
    if missing:xyz[:]=np.nan
    np.save(folder/'world_xyz_mm.npy',xyz)
    r.io.save(folder/'observation.json',dict(schema_version=1,frame_id='world_mm',
        joints_deg={'left':[0]*6},world_from_camera_mm=np.eye(4).tolist(),
        sha256={name:r.io.sha(folder/name) for name in ('rgb.png','world_xyz_mm.npy')}))
    return Observation.load(folder)


def plan(decision='ACTION',x=20):
    return dict(decision=decision,grasp_px=[x,30] if decision=='ACTION' else None,
        target_px=[x+20,30] if decision=='ACTION' else None,motion_type='lift_pull_place' if decision=='ACTION' else None,
        grasp_angle_deg=None,short_reason='拉开袖口')


@pytest.mark.parametrize('failures,decision',[(0,'ACTION'),(1,'ACTION'),(2,'ACTION'),(0,'DONE')])
def test_recovery_replans_preserves_evidence_and_stops_at_two(tmp_path,monkeypatch,failures,decision):
    folder=tmp_path/'round_000';folder.mkdir()
    original=make_obs(folder/'pre_execution_observation',True)
    make_obs(folder/'before',True)
    old=plan();r.io.save(folder/'plan.json',old);r.io.save(folder/'model_metrics.json',dict(wall_latency_s=2.))
    timing=dict(planning_s=2.,fast_planner_wall_s=2.,gate_wall_s=1.)
    captures=[];moves=[];calls=[];prepared=[]
    def capture(path):
        # Both observation and confirmation for each retry have matching depth.
        view=int(path.parent.name.split('_')[-1]) if path.parent.name.startswith('view_') else 0
        o=make_obs(path,view<failures);captures.append(o);return o
    runtime=SimpleNamespace(active_arms=('left',),cancel=threading.Event(),arrival=lambda q:None,capture_current=capture)
    from cloth_agent import gamepad_collect
    monkeypatch.setattr(gamepad_collect,'world_pose',lambda _:np.array([400.,0.,500.,180.,0.,0.]))
    monkeypatch.setattr(r.views,'move_view',lambda runtime,pose,folder:moves.append(pose.copy()))
    monkeypatch.setattr(r.checks,'verify_scene',lambda *a,**k:None)
    def replan(root,attempt,fresh,headers,timeout,direct,previous_context,*,view_change):
        assert view_change and previous_context=={'source_round':'previous'}
        calls.append(fresh.image)
        new=plan(decision,30+len(calls))
        for name in ('plan.json','fast_plan.json'):r.io.save(attempt/name,new)
        r.io.save(attempt/'model_metrics.json',dict(wall_latency_s=3.))
        return new,dict(wall_latency_s=3.)
    monkeypatch.setattr(r.planner,'plan_round',replan)
    class Router:
        def __init__(self,*a,**k):pass
        def accept_fast(self,p):pass
        def shadow(self,*a):pass
    monkeypatch.setattr(r,'ShadowPlannerRouter',Router)
    monkeypatch.setattr(r,'PreActionGate',lambda *a:SimpleNamespace(metrics=dict(gate_wall_s=.5)))
    def prepare(runtime,obs,p,attempt):
        assert p['grasp_px']!=old['grasp_px']
        obs.sample(p['grasp_px'],None);prepared.append(p)
        return ['new operations'],{'left':[0]*6}
    args=(runtime,tmp_path,folder,original,old,DepthUnavailableError('missing'),{},10,10,
          None,prepare,{'source_round':'previous'},timing)
    if failures==2:
        with pytest.raises(DepthUnavailableError,match='20 mm'):r.recover(*args)
        assert len(calls)==2 and len(moves)==3 and not prepared
        assert r.io.read(folder/'plan.json')==old
    else:
        before,checked,new,result=r.recover(*args)
        assert r.io.read(folder/'plan.json')==new
        assert r.io.read(folder/'depth_recovery/original/plan.json')==old
        assert before.image==folder/'before/rgb.png'
        assert timing['planning_s']==2+3*len(calls)
        assert r.io.read(folder/'model_metrics.json')['wall_latency_s']+timing['depth_recovery_prior_planning_s']==timing['planning_s']
        if decision=='DONE':assert result is None and not prepared
        else:assert result[0]==['new operations'] and len(prepared)==1
        assert len(calls)==failures+1
    assert moves[0][0]==380
    if len(calls)==2:assert moves[1][0]==420
