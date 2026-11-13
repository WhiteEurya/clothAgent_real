from pathlib import Path
from types import SimpleNamespace

from cloth_agent.experiment import _execute
from cloth_agent.robot_api import XArmBackend


def robot():
    backend=object.__new__(XArmBackend);backend.collision=object()
    calls=[]
    r=SimpleNamespace(backend=backend,halted=False,stop=lambda:calls.append('stop'))
    r.motion_sequence=lambda actions:calls.append(('group',actions))
    r.move=lambda **args:calls.append(('move',args))
    for name in ('open_gripper','close_gripper','home','shake','shake_open'):
        setattr(r,name,lambda name=name:calls.append(name))
    return r,calls


def test_lift_transport_and_retreat_home_are_continuous_but_gripper_stops():
    r,calls=robot()
    source='''def run():
    open_gripper()
    move(1,2,80,0)
    move(1,2,30,0)
    close_gripper()
    move(1,2,100,0)
    move(20,2,100,0)
    move(20,2,40,0)
    open_gripper()
    move(20,2,90,0)
    home()
'''
    _,error=_execute(source,Path('test.py'),r)
    assert error is None
    assert [c if isinstance(c,str) else c[0] for c in calls]==['open_gripper','group','close_gripper','group','open_gripper','group']
    assert [a['name'] for a in calls[-1][1]]==['move','home']
    assert len(calls[3][1])==3


def test_press_contact_is_a_required_stop_at_direction_reversal():
    r,calls=robot()
    source='''def run():
    close_gripper()
    move(1,2,100,0)
    move(1,2,35,0)
    move(1,2,100,0)
    open_gripper()
'''
    _,error=_execute(source,Path('test.py'),r)
    assert error is None
    assert [c if isinstance(c,str) else c[0] for c in calls]==['close_gripper','group','move','open_gripper']
    assert calls[1][1][-1]['args']['z']==35


def test_upper_turn_and_duplicate_target_do_not_split_queue():
    r,calls=robot()
    source='''def run():
    close_gripper()
    move(1,2,80,0)
    move(1,2,400,0)
    move(1,2,400,0)
    move(20,2,350,0)
    open_gripper()
    move(20,2,350,0)
    home()
'''
    _,error=_execute(source,Path('test.py'),r)
    assert error is None
    assert [c if isinstance(c,str) else c[0] for c in calls]==['close_gripper','group','open_gripper','group']
    assert [a['args']['z'] for a in calls[1][1]]==[80,400,350]
    # A repeated target after release is a new action, not silently removed.
    assert calls[-1][1][0]['args']['z']==350
