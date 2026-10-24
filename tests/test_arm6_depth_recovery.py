from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np
import pytest
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_arm.observation import DepthUnavailableError
from cloth_agent.experiments import arm6_depth_recovery as recovery


def setup(monkeypatch,tmp_path):
    from cloth_agent import gamepad_collect
    origin=np.array([400.,0.,500.,180.,0.,0.])
    monkeypatch.setattr(gamepad_collect,'world_pose',lambda _:origin.copy())
    moves=[];captures=[]
    monkeypatch.setattr(recovery,'move_view',lambda r,p,f:moves.append(p.copy()))
    def capture(p):
        obs=SimpleNamespace(image=Path(p)/'rgb.png',index=len(captures)+1)
        captures.append(obs);return obs
    runtime=SimpleNamespace(active_arms=('left',),cancel=threading.Event(),
        arrival=lambda q:None,capture_current=capture)
    obs=SimpleNamespace(index=0,image=tmp_path/'original.png',
        meta=dict(joints_deg={'left':[0]*6},world_from_camera_mm=np.eye(4).tolist()))
    return runtime,obs,origin,moves,captures


def test_valid_depth_never_moves_or_recaptures(monkeypatch,tmp_path):
    r,o,origin,moves,captures=setup(monkeypatch,tmp_path)
    def unexpected(*args):pytest.fail('unexpected reground')
    actual=recovery.prepare_with_view_retry(r,o,{'old':True},tmp_path/'retry',lambda *a:'plan',unexpected)
    assert actual==(o,{'old':True},'plan')
    assert not moves and not captures


def test_first_view_recovers_with_new_pixels_and_pose(monkeypatch,tmp_path):
    r,o,origin,moves,captures=setup(monkeypatch,tmp_path)
    def prepare(obs,anchors):
        if obs.index==0:raise DepthUnavailableError('missing depth')
        assert anchors=={'view':obs.index}
        return 'fresh plan'
    actual=recovery.prepare_with_view_retry(r,o,{'old':True},tmp_path/'retry',prepare,
                                           lambda obs,folder:{'view':obs.index})
    assert actual[0] is captures[0] and actual[2]=='fresh plan'
    assert len(moves)==1
    assert moves[0]==pytest.approx(origin+[-15,0,0,0,0,0])


def test_two_failed_views_restore_without_grasp(monkeypatch,tmp_path):
    r,o,origin,moves,captures=setup(monkeypatch,tmp_path)
    def missing(*a):raise DepthUnavailableError('missing')
    with pytest.raises(DepthUnavailableError,match='restored'):
        recovery.prepare_with_view_retry(r,o,{},tmp_path/'retry',missing,lambda obs,p:{'view':obs.index})
    assert len(captures)==2 and len(moves)==3
    assert moves[0]==pytest.approx(origin+[-15,0,0,0,0,0])
    assert moves[1]==pytest.approx(origin+[15,0,0,0,0,0])
    assert moves[2]==pytest.approx(origin)


def test_unrelated_planning_error_does_not_trigger_motion(monkeypatch,tmp_path):
    r,o,origin,moves,captures=setup(monkeypatch,tmp_path)
    def bad(*a):raise DualArmError('IK rejected')
    with pytest.raises(DualArmError,match='IK'):
        recovery.prepare_with_view_retry(r,o,{},tmp_path/'retry',bad,lambda *a:{})
    assert not moves


def test_cancel_does_not_trigger_motion(monkeypatch,tmp_path):
    r,o,origin,moves,captures=setup(monkeypatch,tmp_path);r.cancel.set()
    def bad(*a):raise DepthUnavailableError('missing')
    with pytest.raises(KeyboardInterrupt):
        recovery.prepare_with_view_retry(r,o,{},tmp_path/'retry',bad,lambda *a:{})
    assert not moves


def test_lateral_offset_uses_camera_axis_preserves_height_orientation():
    frame=np.eye(4);frame[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    origin=np.array([400,0,500,180,0,3.])
    left,right=recovery.lateral_poses(origin,frame)
    assert left==pytest.approx([400,-15,500,180,0,3])
    assert right==pytest.approx([400,15,500,180,0,3])
    with pytest.raises(DualArmError):recovery.lateral_poses(origin,frame,50)


def test_capture_current_does_not_return_to_perception(monkeypatch,tmp_path):
    from cloth_agent.codex_native_runtime import NativeRuntime
    from cloth_agent.dual_arm import observation
    calls=[]
    camera=object()
    def capture(directory,config,models,snapshots,**kw):
        assert kw['camera'] is camera
        calls.append(directory)
        directory.mkdir()
        return SimpleNamespace(meta=dict(joints_deg={'left':[1,2,3,4,5,6]}))
    monkeypatch.setattr(observation,'capture',capture)
    def unexpected(*a):pytest.fail('unexpected return to perception')
    runtime=SimpleNamespace(coordinator=SimpleNamespace(snapshots=lambda:{}),camera=camera,
        config=SimpleNamespace(arms={}),models={},move_to_perception=unexpected)
    NativeRuntime.capture_current(runtime,tmp_path/'shifted')
    assert calls==[tmp_path/'shifted']
    assert (tmp_path/'shifted/state.json').exists()
