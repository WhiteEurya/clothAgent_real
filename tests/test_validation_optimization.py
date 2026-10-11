from types import SimpleNamespace
import numpy as np
import pytest

from cloth_agent.dual_arm.geometry import Capsule
from cloth_agent.dual_arm.stop_sweep import (
    verify_joint_box,require_safe,ProofIncompleteError,UnsafeBoxError,cell_capsules,
)
from cloth_agent.dual_arm.collision_witness import find_camera_witness
from .test_stop_sweep import arc_scene
from .test_dual_arm_runtime import scene


class CameraModel:
    def __init__(self,offset):self.offset=offset
    def capsules(self,q):
        p=np.array([self.offset+q[0],0.,0.])
        return [Capsule('camera',p,p,1.)]
    def capsule_motion_bounds(self,delta):return {'camera':float(np.abs(delta).sum())}


def cameras():
    config,_=arc_scene();config.obstacles=[]
    models={'left':CameraModel(0),'right':CameraModel(2.5)}
    return config,models,{'left':np.array([0.]),'right':np.array([0.])},{'left':np.array([1.]),'right':np.array([0.])}


def test_fast_witness_is_in_box_verified_and_has_details():
    config,models,lo,hi=cameras()
    result=verify_joint_box(config,models,lo,hi,segment={'id':'test_segment'})
    assert result['status']=='UNSAFE' and result['nodes']==0
    assert result['witness']['objects']==['left/camera','right/camera']
    assert result['witness']['gap_mm']<=0
    for k,q in result['witness']['joints_deg'].items():
        assert np.all(q>=lo[k]) and np.all(q<=hi[k])
    assert result['segment']['id']=='test_segment'
    with pytest.raises(UnsafeBoxError):require_safe(result)


def test_no_witness_still_requires_strict_proof(monkeypatch):
    from cloth_agent.dual_arm import stop_sweep
    config,models,lo,hi=cameras()
    monkeypatch.setattr(stop_sweep,'find_camera_witness',lambda *a,**kw:(None,5))
    result=verify_joint_box(config,models,lo,hi)
    assert result['status']=='UNSAFE' and result['nodes']>0


def test_negative_objective_without_oracle_confirmation_is_not_a_witness(monkeypatch):
    from cloth_agent.dual_arm import collision_witness
    config,models,lo,hi=cameras()
    monkeypatch.setattr(collision_witness,'check_cell_collision',lambda *a,**kw:None)
    witness,count=find_camera_witness(config,models,lo,hi,cell_capsules,lambda:None)
    assert witness is None and count>0


def test_out_of_box_optimizer_candidate_is_clipped_and_rechecked(monkeypatch):
    import scipy.optimize
    config,models,lo,hi=cameras()
    monkeypatch.setattr(scipy.optimize,'minimize',lambda objective,*a,**kw:objective(np.array([100.,-100.])))
    result=verify_joint_box(config,models,lo,hi)
    assert result['status']=='UNSAFE'
    assert result['witness']['joints_deg']=={'left':[1.],'right':[0.]}


@pytest.mark.parametrize('options',[{'max_nodes':1},{'timeout_s':0}])
def test_exhaustion_is_unknown_and_adapter_rejects(options):
    config,models=arc_scene();lo={'left':[-1.25],'right':[0]};hi={'left':[1.35],'right':[0]}
    result=verify_joint_box(config,models,lo,hi,**options)
    assert result['status']=='UNKNOWN'
    with pytest.raises(ProofIncompleteError):require_safe(result)


def test_strict_safe_is_preserved_with_and_without_cache():
    config,models=arc_scene();lo={'left':[-1.25],'right':[0]};hi={'left':[1.35],'right':[0]}
    a=verify_joint_box(config,models,lo,hi,use_cache=False,witness_search=False)
    b=verify_joint_box(config,models,lo,hi)
    assert a['status']==b['status']=='SAFE'
    assert a['nodes']==b['nodes'] and b['pending']==0


def test_pose_cache_is_exact_scoped_and_does_not_alias(scene,monkeypatch):
    from cloth_agent.dual_arm.kinematics import ArmModel
    from cloth_agent.dual_arm.validation_cache import geometry_cache
    m=ArmModel(scene.config.arms['left']);q=np.degrees((m.lower+m.upper)/2)
    calls=[];original=m.urdf.update_cfg
    def update(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(m.urdf,'update_cfg',update)
    before=m.capsules(q);calls.clear()
    with geometry_cache({'left':m}):
        caps=m.capsules(q);m.capsules(q.copy());m.local_capsule_motion_bounds(q,np.zeros_like(q))
        assert len(calls)==1
        small=m.local_capsule_motion_bounds(q,np.full_like(q,.1))
        larger=m.local_capsule_motion_bounds(q,np.full_like(q,.2))
        assert any(larger[k]>small[k] for k in small)
        caps[0].start[0]+=12345
        np.testing.assert_allclose(m.capsules(q)[0].start,before[0].start)
        changed=q.copy();changed[0]+=1e-5;m.capsules(changed)
        assert len(calls)==2
    with geometry_cache({'left':m}):
        m.capsules(changed)
    assert len(calls)==3
    previous=m.capsules(changed)[0].start.copy()
    m.config.world_from_base[0,3]+=10
    with geometry_cache({'left':m}):
        np.testing.assert_allclose(m.capsules(changed)[0].start,previous+[10,0,0])
