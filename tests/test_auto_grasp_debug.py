import copy
from types import SimpleNamespace
import numpy as np
import pytest
from cloth_agent.auto_grasp_debug import close_evidence,prepare_branch
from cloth_agent.codex_native_runtime import NativeRuntime
from cloth_agent.dual_arm.geometry import DualArmError
from tests.test_dual_arm_runtime import scene
from tests.test_codex_native_runtime import runtime_for


def feedback(pos,initial=840):
    return dict(feedback=dict(position_pulse=pos,state='stop',error_code=0,read_errors=[],usable_for_completion=True),
        completion=dict(status='COMPLETED',reason='measured_close_stable',target_position_pulse=5,
            position_tolerance_pulse=5,before_command=dict(position_pulse=initial)))


@pytest.mark.parametrize('pos,expected',[(8,'NEAR_CLOSED'),(700,'WIDE_STALLED'),(200,'CONTACT_OR_UNKNOWN')])
def test_no_full_close_requirement_for_normal_contact(pos,expected):
    assert close_evidence(feedback(pos))==expected


@pytest.mark.parametrize('key,value',[('state','moving'),('error_code',1),('read_errors',['lost']),('usable_for_completion',False)])
def test_uncertain_or_fault_never_triggers(key,value):
    f=feedback(700);f['feedback'][key]=value
    assert close_evidence(f)=='UNKNOWN'


def test_branch_uses_same_planner_and_preserves_transport(tmp_path):
    from cloth_agent.codex_motion import parse_code
    calls=[]
    def plan(s,q):calls.append((parse_code(s),q));return []
    r=SimpleNamespace(plan=plan,plan_stats={})
    q={'right':[0]*7,'left':[0]*6}
    source='motion.grasp(right=[.7,0,.05,3.14,0,0])\nmotion.move_tcp(right=[.7,.1,.1,3.14,0,0])'
    operations=[('gripper',dict(action='close',arm='right',joints=q)),('move',SimpleNamespace(end=q))]
    b=prepare_branch(r,source,operations,tmp_path)
    assert b['close_index']==0
    assert calls[0][0][0][1]['right'][2]==pytest.approx(.052)
    assert calls[0][0][1]==('gripper',{'arm':'right','action':'close'})
    assert calls[0][0][2][1]['right']==[.7,.1,.1,3.14,0,0]


@pytest.mark.parametrize('second,expected',[(8,'EXECUTED'),(700,'EXECUTION_FAILED')])
def test_wide_stall_selects_preplanned_branch_once(scene,tmp_path,monkeypatch,second,expected):
    r=runtime_for(scene);q={k:a.home_joints.tolist() for k,a in scene.config.arms.items()}
    close=('gripper',dict(arm='right',action='close',joints=q))
    original_tail=('gripper',dict(arm='right',action='open',joints=q))
    responses=iter([feedback(700),feedback(second,initial=700),feedback(840)])
    actions=[]
    def gripper(action):actions.append(action);return next(responses)
    monkeypatch.setattr(scene.connections['right'],'gripper',gripper)
    # No fake motion is dispatched; tests branching / reclose / abort order independently of planner.
    r.auto_grasp_branch=dict(close_index=0,operations=[close,original_tail])
    try:
        result=r._execute_operations([close,original_tail],tmp_path)
        assert result['status']==expected
        assert actions==(['close','close','open'] if second==8 else ['close','close'])
        assert not scene.connections['right'].debug_reclose_confirmation
    finally:r.close()


def test_branch_plan_failure_blocks_all_dispatch(scene,tmp_path,monkeypatch):
    r=runtime_for(scene);r.auto_grasp_debug=True
    initial={k:a.home_joints for k,a in scene.config.arms.items()}
    obs=SimpleNamespace(meta={'joints_deg':initial},validate_for=lambda *a,**k:None)
    monkeypatch.setattr(r,'plan',lambda *a:[])
    def reject(*a):raise DualArmError('recovery lift rejected by existing planner')
    monkeypatch.setattr('cloth_agent.auto_grasp_debug.prepare_branch',reject)
    try:
        result=r.execute('motion.gripper(arm="right",action="open")',obs,tmp_path)
        assert result['status']=='PLANNING_FAILED' and not result['physical_execution']
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()

@pytest.mark.parametrize('enabled',[False,True])
def test_stationary_close_confirmation_is_opt_in(monkeypatch,enabled):
    from cloth_agent.dual_arm.single_arm_backend import XArmBackend
    from tests.test_gripper_completion import Arm,sample
    from cloth_agent.config import RobotConfig,WorkspaceBounds
    config=RobotConfig(robot_ip='test',boundaries=WorkspaceBounds(x_min=0,x_max=600,y_min=-300,y_max=300,z_min=0,z_max=500),init_joints_deg=(0,)*6,init_pose_mm_deg=(300,0,200,180,0,0),orientation_roll_deg=180,orientation_pitch_deg=0,gripper_completion_timeout_s=60.)
    now=[0.];monkeypatch.setattr('cloth_agent.dual_arm.single_arm_backend.time.monotonic',lambda:now[0])
    monkeypatch.setattr('cloth_agent.dual_arm.single_arm_backend.time.sleep',lambda s:now.__setitem__(0,now[0]+s))
    b=XArmBackend.__new__(XArmBackend);b.arm=Arm([sample(700)]*90+[KeyboardInterrupt()]);b._state=lambda:([0]*6,{})
    b.debug_reclose_confirmation=enabled
    if enabled:
        result,_=b.close_gripper(config)
        assert result['completion']['duration_s']>=3
        assert result['completion']['debug_stationary_close_confirmation'] is True
        assert result['feedback']['position_pulse']==700
    else:
        with pytest.raises(KeyboardInterrupt):b.close_gripper(config)
