import numpy as np

from cloth_agent.dual_arm import homing
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_arm.separation import minimum_gap,separation_steps
from .test_dual_arm_runtime import scene


def close_pose(scene):
    q={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    q['right'][1]=-90
    return q


def test_direction_ranking_refines_overlapping_capsules_with_meshes():
    from .test_mesh_narrowphase import HullModel
    from cloth_agent.dual_arm.geometry import Capsule
    models={'left':HullModel(0),'right':HullModel(10)}
    for model in models.values():
        center=model.points.mean(0)
        model.capsules=lambda q,p=center:[Capsule('link1',p,p,10.)]
    assert minimum_gap(models,{'left':[0.],'right':[0.]})==9.


def test_close_home_separates_before_return_and_keeps_other_arm_still(scene):
    scene.config.execution_mode='controller_sequential'
    initial=close_pose(scene)
    program,attempts=homing.plan_home(initial,scene.config,scene.models)
    first=program[0]
    assert first.phase.name.startswith('home_separate_')
    changed=sum(np.any(np.abs(q[-1]-q[0])>1e-9) for q in first.joints.values())
    assert changed==1
    gaps=[minimum_gap(scene.models,{k:q[i] for k,q in first.joints.items()})
          for i in range(len(first.times))]
    assert np.min(np.diff(gaps)) >= -1e-9
    assert gaps[-1]>gaps[0]+1
    homing.validate_home_program(program,scene.config)
    assert attempts[-1]['status']=='accepted'


def test_no_separation_prefix_when_every_sweep_is_blocked(scene,monkeypatch):
    def blocked(*args,**kwargs):
        raise DualArmError('stopping range blocked')
    monkeypatch.setattr(homing,'joint_motion',blocked)
    attempts=[]
    assert list(separation_steps(close_pose(scene),scene.config,scene.models,attempts))==[]
    assert attempts and all(a['status']=='rejected' for a in attempts)


def test_already_home_does_not_add_separation(scene,monkeypatch):
    import cloth_agent.dual_arm.separation as separation
    scene.config.execution_mode='controller_sequential'
    monkeypatch.setattr(separation,'minimum_gap',lambda *args:-100.)
    def unexpected(*args,**kwargs):
        raise AssertionError('already Home must not move away again')
    monkeypatch.setattr(separation,'separation_steps',unexpected)
    program,_=homing.plan_home({k:a.home_joints.copy() for k,a in scene.config.arms.items()},
                             scene.config,scene.models)
    assert not any(m.phase.name.startswith('home_separate_') for m in program[:-1])


def test_separated_home_uses_coordinated_return(scene,monkeypatch):
    scene.config.execution_mode='controller_sequential'
    original=homing.joint_motion
    def one_axis_only(start,end,*args,**kwargs):
        if sum(np.count_nonzero(np.abs(end[k]-start[k])>1e-9) for k in start)>1:
            raise DualArmError('multi-axis stop box blocked')
        return original(start,end,*args,**kwargs)
    monkeypatch.setattr(homing,'joint_motion',one_axis_only)
    initial=close_pose(scene)
    for q in initial.values():q[2]+=5
    program,attempts=homing.plan_home(initial,scene.config,scene.models)
    assert 'cartesian' in attempts[-1]['strategy']
    homing.validate_home_program(program,scene.config)
    assert any(sum(np.count_nonzero(np.abs(q[-1]-q[0])>1e-9)
                   for q in motion.joints.values())>1 for motion in program[1:-1])
    assert not any('joint_return' in m.phase.name for m in program[:-1])
