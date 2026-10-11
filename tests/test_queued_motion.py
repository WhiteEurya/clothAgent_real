import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent.queued_motion import queue_moves,check_queued_feedback
from cloth_agent.dual_arm.sequential_home import PositionSegment
from cloth_agent.dual_arm.geometry import DualArmError
from .test_dual_arm_runtime import scene
from .test_codex_native_runtime import runtime_for


def segment(a,b):
    return PositionSegment('left',{'left':np.array([a]*6),'right':np.zeros(7)},
        {'left':np.array([b]*6),'right':np.zeros(7)},5,10,30)


def test_waypoints_preserved_and_gripper_separates_batches():
    a,b,c=segment(0,1),segment(1,2),segment(2,3)
    event=('gripper',{'arm':'left','action':'close'})
    result=queue_moves([('move',a),('move',b),event,('move',c)])
    assert len(result)==3 and result[1]==event
    assert result[0][1].segments==(a,b)
    assert result[2][1] is c
    assert len(result[0][1].to_dict()['queued_segments'])==2
    with pytest.raises(DualArmError,match='discontinuous'):
        queue_moves([('move',a),('move',c)])


def test_feedback_accepts_only_certified_boxes(scene):
    r=runtime_for(scene);q={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    b={k:v.copy() for k,v in q.items()};b['left'][0]+=1
    c={k:v.copy() for k,v in b.items()};c['left'][0]+=1
    batch=queue_moves([('move',PositionSegment('left',q,b,5,10,30)),
                       ('move',PositionSegment('left',b,c,5,10,30))])[0][1]
    try:
        scene.connections['left'].joints=b['left'].copy()
        check_queued_feedback(r.coordinator,batch)
        # Degree/radian conversions can put the final endpoint a few ULPs
        # outside its box; this must not reject an otherwise completed queue.
        scene.connections['left'].joints=c['left'].copy()
        scene.connections['left'].joints[0]=np.nextafter(c['left'][0],np.inf)
        check_queued_feedback(r.coordinator,batch)
        scene.connections['left'].joints[0]=c['left'][0]+1e-5
        with pytest.raises(DualArmError):check_queued_feedback(r.coordinator,batch)
        scene.connections['left'].joints=b['left'].copy()
        scene.connections['right'].joints[0]+=10
        with pytest.raises(DualArmError):check_queued_feedback(r.coordinator,batch)
    finally:r.close()


def test_sdk_queue_waits_only_at_end_and_keeps_exact_path():
    from cloth_agent.dual_arm.execution import XArmConnection
    calls=[]
    arm=SimpleNamespace(mode=0,set_servo_angle=lambda **kw:calls.append(kw) or 0)
    connection=SimpleNamespace(arm=arm,cancel=threading.Event(),_check=lambda name,code:None)
    XArmConnection.move_joint_queue(connection,[segment(0,1),segment(1,2),segment(2,3)])
    assert [c['wait'] for c in calls]==[False,False,True]
    assert all(c['radius']==-1 for c in calls)
    assert [c['angle'][0] for c in calls]==[1,2,3]
    assert connection.position_command_active is False
    connection.cancel.set()
    with pytest.raises(DualArmError):XArmConnection.move_joint_queue(connection,[segment(0,1)])
    assert len(calls)==3
