from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent.dual_arm.geometry import Capsule, DualArmError, check_collision
from cloth_agent.dual_arm.stop_sweep import certify_joint_box, native_joint_ranges, validate_native_sweep


class ArcModel:
    def __init__(self, offset):
        self.offset=np.asarray(offset,dtype=float)

    def capsules(self,q):
        theta=np.radians(q[0])
        p=self.offset+np.array([100*np.cos(theta),100*np.sin(theta),0])
        return [Capsule('tool',p,p,1)]

    def capsule_motion_bounds(self,delta):
        return {'tool':100*float(np.abs(np.radians(delta)).sum())}


def arc_scene():
    config=SimpleNamespace(
        arms={k:SimpleNamespace(axis=1) for k in ('left','right')},
        limits={'tracking_error_deg':0.,'stationary_tolerance_deg':0.,
                'tracking_error_mm':0.,'clearance_mm':0.},
        obstacles=[{'name':'side_obstacle','min_mm':[101.4,-5,-2],'max_mm':[103,5,2]}],
        raw={'safety':{'status':'measured','max_sweep_nodes':4096,
             'arms':{k:{'base_error_mm':0.,'geometry_error_mm':0.,'stop_excursion_deg':[1.25]}
                     for k in ('left','right')}}})
    models={'left':ArcModel([0,0,0]),'right':ArcModel([1000,1000,0])}
    return config,models


def test_adaptive_stop_sweep_clears_false_isotropic_collision():
    config,models=arc_scene()
    start={k:np.array([0.]) for k in models}
    end={'left':np.array([.1]),'right':np.array([0.])}
    old={k:[Capsule(c.name,c.start,c.end,c.radius+100*np.radians(1.25))
            for c in m.capsules(start[k])] for k,m in models.items()}
    with pytest.raises(DualArmError,match='obstacle collision'):
        check_collision(old,config.obstacles,0)
    result=validate_native_sweep(config,models,start,end)
    assert result['nodes'] > 1
    assert result['accepted_cells'] > 1
    # Independently sample the complete signed braking interval, not only its ends.
    for angle in np.linspace(-1.25,1.35,101):
        check_collision({'left':models['left'].capsules([angle]),
                         'right':models['right'].capsules([0])},config.obstacles,0)


def test_collision_inside_stopping_range_still_rejected():
    config,models=arc_scene()
    config.obstacles=[{'name':'stop_path','min_mm':[99.8,1,-1],'max_mm':[100.2,1.2,1]}]
    with pytest.raises(DualArmError,match='contains collision'):
        validate_native_sweep(config,models,{'left':[0],'right':[0]}, {'left':[.1],'right':[0]})


def test_stop_sweep_budget_exhaustion_never_passes():
    config,models=arc_scene()
    config.raw['safety']['max_sweep_nodes']=1
    with pytest.raises(DualArmError,match='budget'):
        validate_native_sweep(config,models,{'left':[0],'right':[0]}, {'left':[.1],'right':[0]})


def test_search_timeout_rejects_and_does_not_leak_into_later_proofs():
    from cloth_agent.dual_arm.stop_sweep import search_proof_budget
    config,models=arc_scene()
    with search_proof_budget(0):
        with pytest.raises(DualArmError,match='time budget'):
            validate_native_sweep(config,models,{'left':[0],'right':[0]}, {'left':[.1],'right':[0]})
    assert validate_native_sweep(config,models,{'left':[0],'right':[0]}, {'left':[.1],'right':[0]})['accepted_cells']>0


def test_native_state_is_not_inflated_by_stopping_but_move_still_checks_it():
    from cloth_agent.dual_arm.safety import check_state, validate_sweep
    config,models=arc_scene()
    config.execution_mode='controller_sequential'
    config.obstacles=[{'name':'beyond_start','min_mm':[99.8,1.5,-1],'max_mm':[100.2,1.7,1]}]
    start={'left':np.array([0.]),'right':np.array([0.])}
    check_state(config,models,start)
    with pytest.raises(DualArmError,match='contains collision'):
        validate_sweep(config,models,start,{'left':np.array([.1]),'right':np.array([0.])})


def test_only_commanded_joints_get_stopping_excursions():
    config,_=arc_scene()
    config.limits['stationary_tolerance_deg']=.2
    lo,hi=native_joint_ranges(config,{'left':[0],'right':[3]}, {'left':[-2],'right':[3]})
    assert lo['left']==pytest.approx([-3.25])
    assert hi['left']==pytest.approx([1.25])
    assert lo['right']==pytest.approx([2.8])
    assert hi['right']==pytest.approx([3.2])


def test_independent_joints_do_not_collapse_to_shared_progress():
    class XYModel:
        def __init__(self, offset): self.offset=np.asarray(offset)
        def capsules(self,q):
            p=self.offset+np.r_[q,0.]
            return [Capsule('tool',p,p,.5)]
        def capsule_motion_bounds(self,d): return {'tool':float(np.abs(d).sum())}
    config,_=arc_scene()
    for arm in config.arms.values(): arm.axis=2
    models={'left':XYModel([0,0,0]),'right':XYModel([1000,0,0])}
    config.obstacles=[{'name':'off_diagonal','min_mm':[.5,8.5,-1],'max_mm':[1.5,9.5,1]}]
    with pytest.raises(DualArmError,match='contains collision'):
        certify_joint_box(config,models,{'left':[0,0],'right':[0,0]},
                          {'left':[10,10],'right':[0,0]})


def test_stationary_axes_of_moving_arm_keep_arrival_tolerance():
    config,_=arc_scene()
    for arm in config.arms.values(): arm.axis=2
    for row in config.raw['safety']['arms'].values(): row['stop_excursion_deg']=[1.25,1.25]
    config.limits['stationary_tolerance_deg']=.2
    lo,hi=native_joint_ranges(config,{'left':[0,3],'right':[0,0]},
                             {'left':[1,3],'right':[0,0]})
    assert lo['left']==pytest.approx([-1.25,2.8])
    assert hi['left']==pytest.approx([2.25,3.2])


def test_native_path_checks_intermediate_extrema():
    from cloth_agent.dual_arm.stop_sweep import validate_native_path
    config,models=arc_scene()
    config.obstacles=[{'name':'middle','min_mm':[-2,99,-2],'max_mm':[2,101,2]}]
    with pytest.raises(DualArmError,match='contains collision'):
        validate_native_path(config,models,{'left':np.array([[0.],[90.],[0.]]),
                                           'right':np.array([[0.],[0.],[0.]])})


def test_small_paths_get_full_proof_budget_before_further_subdivision():
    from cloth_agent.dual_arm.stop_sweep import path_proof_budget
    config,_=arc_scene()
    start={'left':np.array([0.]),'right':np.array([0.])}
    assert path_proof_budget(config,start,{'left':np.array([3.]),'right':np.array([0.])}) is None
    assert path_proof_budget(config,start,{'left':np.array([30.]),'right':np.array([0.])})==64


def test_native_path_covers_all_intervals_without_repeated_leaf_checks(monkeypatch):
    import cloth_agent.dual_arm.stop_sweep as sweep
    config,models=arc_scene()
    config.obstacles=[]
    calls=[]
    original=sweep.certify_joint_box
    def record(*args,**kwargs):
        calls.append((args[2],args[3]))
        return original(*args,**kwargs)
    monkeypatch.setattr(sweep,'certify_joint_box',record)
    sweep.validate_native_path(config,models,{'left':np.linspace(0,5,101)[:,None],
                                             'right':np.zeros((101,1))})
    assert len(calls)==1
    assert calls[0][0]['left']==pytest.approx([-1.25])
    assert calls[0][1]['left']==pytest.approx([6.25])
