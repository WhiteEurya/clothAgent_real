import copy
import threading

import pytest

from scripts import dual_arm_micro_test as m


class Arm:
    def __init__(self, axis, serial):
        self.axis = axis
        self.control_box_sn = serial
        self.mode = 0
        self.state = 0
        self.motor_enable_states = [1] * axis
        self.motor_brake_states = [1] * axis
        self.tcp_offset = [0] * 6
        self.pose = [400, 0, 300, 180, 0, 0]
        self.commands = []
        self.fail = False
        self.arrival = None
        self.stopped = False

    def get_state(self): return 0, self.state
    def get_err_warn_code(self): return 0, [0, 0]
    def get_robot_sn(self): return 0, 'shared-arm-serial'
    def get_position(self, **kw): return 0, list(self.pose)
    def get_servo_angle(self, **kw): return 0, [0] * self.axis
    def is_tcp_limit(self, *a, **kw): return 0, False
    def is_joint_limit(self, *a, **kw): return 0, False
    def get_inverse_kinematics(self, *a, **kw): return 0, [0] * self.axis
    def set_state(self, state):
        self.stopped = True
        return 0
    def set_position(self, *pose, **kw):
        self.commands.append(pose)
        if self.arrival: self.arrival.wait(timeout=2)
        if self.fail: return -1
        self.pose = list(pose)
        return 0


def setup_pair():
    arms = [Arm(6, 'box-a'), Arm(7, 'box-b')]
    configs = [m.snapshot(a, str(i)) for i,a in enumerate(arms)]
    return arms, configs


def test_both_start_concurrently_and_return_to_distinct_homes():
    arms, _ = setup_pair()
    arms[1].pose[1] = 100
    configs = [m.snapshot(a, str(i)) for i,a in enumerate(arms)]
    rendezvous = threading.Barrier(2)
    for a in arms: a.arrival = rendezvous
    m.execute(arms, configs, 3, 2)
    for a,c in zip(arms, configs):
        assert len(a.commands) == 2
        assert a.commands[0][2] == c['home'][2] + 3
        assert a.pose == c['home']
        assert not a.stopped


@pytest.mark.parametrize('problem', ['state', 'identity', 'offset', 'home', 'ik'])
def test_second_arm_preflight_failure_prevents_all_motion(problem):
    arms, configs = setup_pair()
    if problem == 'state': arms[1].state = 4
    if problem == 'identity': arms[1].control_box_sn = 'replacement'
    if problem == 'offset': arms[1].tcp_offset[2] = 10
    if problem == 'home': arms[1].pose[0] += 2
    if problem == 'ik': arms[1].get_inverse_kinematics = lambda *a, **k: (0, [30]*7)
    with pytest.raises(RuntimeError): m.execute(arms, configs, 3, 2)
    assert not any(a.commands for a in arms)


def test_motion_failure_stops_both_and_never_returns_home():
    arms, configs = setup_pair()
    arms[1].fail = True
    with pytest.raises(RuntimeError): m.execute(arms, configs, 3, 2)
    assert all(a.stopped for a in arms)
    assert len(arms[1].commands) == 1
    assert all(len(a.commands) <= 1 for a in arms)


def test_home_rejects_nonfinite_or_duplicate_devices():
    _, configs = setup_pair()
    bad = copy.deepcopy(configs)
    bad[0]['home'][0] = float('nan')
    with pytest.raises(ValueError): m.validate({'arms': bad})
    bad = copy.deepcopy(configs)
    bad[1]['ip'] = bad[0]['ip']
    with pytest.raises(ValueError): m.validate({'arms': bad})


def test_facing_home_mirrors_position_and_tool_frame():
    # A nontrivial pose checks all Euler components, not only a vertical tool.
    import math
    def rotation(pose):
        r,p,y = map(math.radians, pose[3:])
        cr,sr,cp,sp,cy,sy = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
        return [[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],
                [-sp,cp*sr,cp*cr]]
    pose = [495,-3,370,-178,-4,7]
    target = m.mirror_facing_home(pose)
    assert target[:3] == [495,3,370]
    a,b = rotation(pose),rotation(target)
    d = [1,-1,1]
    for i in range(3):
        for j in range(3):
            assert b[i][j] == pytest.approx(d[i]*a[i][j]*d[j])
    assert m.mirror_facing_home(target) == pose
