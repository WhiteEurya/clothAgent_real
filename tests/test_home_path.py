import numpy as np
import pytest

from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_arm.home_path import check_program_path,settings
from cloth_agent.dual_arm.planning import Motion,Phase
from .test_dual_arm_runtime import scene


def program(scene,delta):
    joints={k:np.tile(a.home_joints,(3,1)) for k,a in scene.config.arms.items()}
    joints['left'][1,:3]+=delta
    poses={k:np.asarray([scene.models[k].forward(q) for q in rows]) for k,rows in joints.items()}
    return [Motion(Phase('home_test','move'),np.array([0.,1.,2.]),joints,poses),
            Phase('home_open_grippers','open')]


def test_home_rejects_unnecessary_high_arc(scene):
    with pytest.raises(DualArmError,match='height limit'):
        check_program_path(program(scene,[0,0,60]),scene.config,scene.models)


def test_home_rejects_sideways_detour(scene):
    with pytest.raises(DualArmError,match='detour limit'):
        check_program_path(program(scene,[80,0,0]),scene.config,scene.models)


def test_small_local_separation_fits_home_limits(scene):
    _,metrics=check_program_path(program(scene,[10,0,0]),scene.config,scene.models)
    assert metrics['left']['length_mm']==pytest.approx(20.)


def test_invalid_home_path_limit_fails(scene):
    scene.config.raw['home_path']={'extra_height_mm':-1}
    with pytest.raises(DualArmError,match='extra_height_mm'):
        settings(scene.config)


def test_redundant_joint_is_guided_to_home_without_final_large_swing(scene,monkeypatch):
    from cloth_agent.dual_arm.home_path import coordinated_return
    scene.config.execution_mode='controller_sequential'
    target={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    initial={k:q.copy() for k,q in target.items()}
    initial['right'][0]-=20;initial['right'][6]=10
    monkeypatch.setattr(scene.models['right'],'inverse',lambda pose,seed:np.r_[pose,seed[6]])
    motions=coordinated_return(initial,target,('right',),scene.config,scene.models,'home_cartesian_test')
    assert len(motions)==1
    q=motions[0].joints['right']
    assert np.all(np.diff(q[:,6])<=1e-9)
    assert np.allclose(q[-1],target['right'])


def test_controller_joint_box_must_fit_height_limit(scene):
    from cloth_agent.dual_arm.sequential_home import certify_segment
    from cloth_agent.dual_arm.home_path import corridor
    scene.config.execution_mode='controller_sequential'
    start={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    end={k:q.copy() for k,q in start.items()};end['left'][0]+=10
    bounds=corridor(start,scene.config,scene.models)
    bounds['left']['height']=start['left'][2]+.1
    with pytest.raises(DualArmError,match='controller segment exceeds Home height limit'):
        certify_segment(start,end,scene.config,scene.models,path_bounds=bounds)


def test_coordinated_joint_return_defers_only_tiny_final_adjustments(scene):
    from cloth_agent.dual_arm.home_path import coordinated_return
    scene.config.execution_mode='controller_sequential'
    target={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    initial={k:q.copy() for k,q in target.items()}
    initial['right'][0]-=20;initial['right'][2]+=5;initial['right'][6]=.1
    motions=coordinated_return(initial,target,('right',),scene.config,scene.models,'home_joint_test',joint=True)
    assert len(motions)==2
    assert np.all(motions[0].joints['right'][:,6]==.1)
    assert np.count_nonzero(motions[0].joints['right'][-1]-initial['right'])==2
    assert np.array_equal(motions[-1].joints['right'][-1],target['right'])


def test_interleaved_return_moves_arms_in_short_coordinated_turns(scene):
    from cloth_agent.dual_arm.home_path import interleaved_return
    scene.config.execution_mode='controller_sequential'
    target={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    initial={k:q.copy() for k,q in target.items()}
    for q in initial.values():q[0]-=20;q[2]-=10
    motions=interleaved_return(initial,target,scene.config,scene.models,'home_interleaved_test',steps=3)
    assert len(motions)==6
    for i,motion in enumerate(motions):
        moving=[k for k,q in motion.joints.items() if not np.array_equal(q[0],q[-1])]
        assert moving==['left' if i%2==0 else 'right']
        assert np.count_nonzero(motion.joints[moving[0]][-1]-motion.joints[moving[0]][0])==2
    for key in target:
        assert not np.array_equal(motions[1].joints[key][-1],target[key])
        assert np.allclose(motions[-1].joints[key][-1],target[key])
