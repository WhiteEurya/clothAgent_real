from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import check_zed_planned_path as checker


def example():
    state={'left':[0.]*7,'right':[0.]*7}
    config={'arms':{arm:dict(axis=7,ip='test',robot_sn=arm,control_box_sn=arm,
                           tcp_offset_mm_deg=[0,0,172,0,0,0]) for arm in state},
            'limits':dict(joint_speed_deg_s=5,joint_accel_deg_s2=10)}
    artifact=dict(active_arm='right',initial_joints_deg=deepcopy(state),operations=[])
    for index in range(3):
        target=deepcopy(state)
        target['right'][0]+=.1
        artifact['operations'].append(dict(kind='move',data=dict(arm='right',start=state,end=target,
            speed_deg_s=5,acceleration_deg_s2=10)))
        state=target
    return artifact,config


@pytest.mark.parametrize('change', ['inactive','discontinuous','speed','nonfinite'])
def test_reject_invalid_export_before_controller_access(change):
    artifact,config=example()
    data=artifact['operations'][0]['data']
    if change=='inactive':data['end']['left'][0]=1
    if change=='discontinuous':data['start']['right'][0]=1
    if change=='speed':data['speed_deg_s']=6
    if change=='nonfinite':data['end']['right'][0]=float('nan')
    with pytest.raises(ValueError):checker.validate_export(artifact,config)


def test_controller_queries_are_only_check_and_leave_physical_state_unchanged(tmp_path,monkeypatch):
    artifact,config=example()
    config_path=tmp_path/'config.json'
    config_path.write_text(json.dumps(config))
    artifact.update(config_path=str(config_path),config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    operations=tmp_path/'operations.json'
    operations.write_text(json.dumps(artifact))
    output=tmp_path/'checks.json'
    commands=[]
    class Robot:
        control_box_sn='right'
        sn='right'
        axis=7
        tcp_offset=[0,0,172,0,0,0]
        world_offset=[0]*6
        mode=0
        only_check_result=0
        def __init__(self,*args,**kwargs):pass
        def connect(self):pass
        def disconnect(self):pass
        def get_robot_sn(self):return 0,self.sn
        def get_servo_angle(self,**kwargs):return 0,[0.]*7
        def get_state(self):return 0,2
        def get_err_warn_code(self):return 0,[0,0]
        def set_servo_angle(self,**kwargs):
            assert kwargs['only_check_type'] in (1,2,3)
            commands.append(kwargs)
            return 0
    monkeypatch.setitem(sys.modules,'xarm.wrapper',SimpleNamespace(XArmAPI=Robot))
    monkeypatch.setattr(checker.socket,'create_connection',lambda *args,**kwargs:nullcontext())
    monkeypatch.setattr(checker.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(sys,'argv',['check','--operations',str(operations),'--output',str(output)])
    assert checker.main()==0
    assert [command['only_check_type'] for command in commands]==[1,3,2]
    report=json.loads(output.read_text())
    assert report['status']=='CONTROLLER_PATH_PASS'
    assert report['joint_drift_deg']==0
    assert report['physical_motion_commands']==report['gripper_commands']==0
