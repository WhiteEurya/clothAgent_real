import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from cloth_agent.grasp_orientation import nearest_grasp_orientation
from cloth_agent.codex_motion import parse_code


def test_half_turn_is_removed_for_perception_heading():
    source='motion.grasp(right=[.6,0,.06,3.141592654,0,0])\nmotion.move_tcp(right=[.7,.1,.1,3.141592654,0,0])'
    result,report=nearest_grasp_orientation(source,R.from_euler('xyz',[178.81,-5.12,179.04],degrees=True).as_matrix())
    assert report['applied'] and report['equivalent_rotation_deg']<6
    a=parse_code(source,allow_grasp=True);b=parse_code(result,allow_grasp=True)
    for old,new in zip(a,b):
        assert old[1]['right'][:3]==new[1]['right'][:3]
        assert abs(abs(new[1]['right'][5])-np.pi)<1e-6


def test_tilted_transport_preserves_relative_world_rotation():
    source='motion.grasp(right=[.6,0,.06,3.0,.1,0])\nmotion.move_transfer(right=[.7,.1,.1,2.8,.2,.4])'
    result,report=nearest_grasp_orientation(source,(R.from_euler('xyz',[3.,.1,0])*R.from_rotvec([0,0,np.pi])).as_matrix())
    assert report['applied']
    a=[R.from_euler('xyz',kw['right'][3:]) for _,kw in parse_code(source,allow_grasp=True)]
    b=[R.from_euler('xyz',kw['right'][3:]) for _,kw in parse_code(result,allow_grasp=True)]
    assert np.allclose((a[1]*a[0].inv()).as_matrix(),(b[1]*b[0].inv()).as_matrix())


def test_nearby_pose_and_explicit_commands_unchanged():
    source='motion.grasp(right=[.6,0,.06,3.14,0,0])\nmotion.move_tcp(right=[.7,.1,.1,3.14,0,0])'
    result,report=nearest_grasp_orientation(source,R.from_euler('xyz',[3.14,0,.1]).as_matrix())
    assert result==source and not report['applied']
    raw='motion.move_joints(right=[0,0,0,0,0,0,0])'
    assert nearest_grasp_orientation(raw,np.eye(3))==(raw,None)


@pytest.mark.parametrize('yaw,expected',[(20,-160),(-20,-20),(0,0),(180,0),(90,-90)])
def test_ccw_uses_jaw_half_turn_not_full_revolution(yaw,expected):
    source=f'motion.grasp(right=[.6,0,.06,0,0,{np.radians(yaw)}])\nmotion.move_tcp(right=[.7,.1,.1,.2,.1,.4])'
    result,report=nearest_grasp_orientation(source,np.eye(3),prefer_ccw=True)
    assert report['selected_local_twist_deg']==pytest.approx(expected)
    assert -180<=report['selected_local_twist_deg']<=0
    old=parse_code(source,allow_grasp=True);new=parse_code(result,allow_grasp=True)
    a=[R.from_euler('xyz',kw['right'][3:]) for _,kw in old]
    b=[R.from_euler('xyz',kw['right'][3:]) for _,kw in new]
    np.testing.assert_allclose((a[1]*a[0].inv()).as_matrix(),(b[1]*b[0].inv()).as_matrix(),atol=1e-8)
    for (_,x),(_,y) in zip(old,new):assert x['right'][:3]==y['right'][:3]
