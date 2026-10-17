import copy
from types import SimpleNamespace
import numpy as np
import pytest
from cloth_agent.experiments.seed_template_offline_audit import pose_errors,geometry_checks


def test_pose_errors_compare_rotations_not_euler_numbers():
    a=[dict(method='grasp',arguments=dict(right=[.4,0,.05,np.pi,0,np.pi]))]
    b=copy.deepcopy(a);b[0]['arguments']['right'][5]=-np.pi
    assert pose_errors(a,b)[0]['rotation_error_deg']<1e-10
    b[0]['arguments']['right'][0]+=.003
    assert pose_errors(a,b)[0]['xyz_error_mm']==pytest.approx(3.)


def test_pose_errors_reject_changed_primitive():
    with pytest.raises(ValueError,match='method'):
        pose_errors([dict(method='grasp')],[dict(method='move_tcp')])


def test_workspace_check_uses_base_frame_and_all_waypoints():
    calls=[]
    def base(pose):
        pose=pose.copy();pose[0]-=400;return pose
    def validate(p):
        calls.append(p.copy())
        if p[0]>20:raise ValueError('outside measured workspace')
    arm=SimpleNamespace(base_pose=base,validate_point=validate)
    slots=[dict(pixel_xy=[10,10],z_above_surface_mm=0),dict(pixel_xy=[30,10],z_above_surface_mm=60)]
    m=[dict(commanded_world_m_rad=[.4,0,.05,0,0,0]),dict(commanded_world_m_rad=[.43,0,.11,0,0,0])]
    with pytest.raises(ValueError,match='workspace'):
        geometry_checks(dict(slots=slots),{},SimpleNamespace(xyz=np.zeros((50,50,3))),arm,m)
    assert len(calls)==2
    assert calls[0][0]==0 and calls[1][0]==pytest.approx(30.)
