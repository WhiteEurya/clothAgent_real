import json
import threading
import time
from types import SimpleNamespace as S
import numpy as np
import pytest
from cloth_agent.experiments import moving_release_probe as m

@pytest.mark.parametrize('failure',[None,'path','gripper'])
def test_empty_probe_checks_path_and_stops_on_release_failure(tmp_path,monkeypatch,failure):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr('sys.argv',['probe','--execute-empty'])
    events=[];position=[200.,0.,450.,180.,0.,0.];opened=[True]
    cfg=S(home_speed_deg_s=20.,home_acceleration_deg_s2=40.,robot_ip='fake',continuous_joint_stream=True,gripper_open=850,gripper_speed=2000,
          gripper_open_tolerance_pulse=15,relative_yaw_from_absolute_deg=lambda x:x)
    arms={k:S(home_joints=np.zeros(7),serial=k) for k in ('left','right')}
    runtime=S(config=S(arms=arms,limits={'stationary_tolerance_deg':1},raw={},root=tmp_path),
              models={k:S(forward=lambda q:np.array(position)) for k in arms},profile={},config_path='fake',close=lambda:None)
    monkeypatch.setattr(m,'PartitionedReplayRuntime',lambda *a,**kw:runtime)
    monkeypatch.setattr(m,'configure_runtime',lambda r:r)
    monkeypatch.setattr(m,'read_arm',lambda p,k:dict(status='READ_ONLY_CHECKS_PASS',control_box_sn=k,joints_deg=[0]*7))
    monkeypatch.setattr(m,'build_robot_config',lambda *a,**kw:cfg)
    def check(*a):
        events.append('path')
        if failure=='path':raise ValueError('bad path')
    monkeypatch.setattr(m,'RobotCollision',lambda *a,**kw:S(require_execution=check))
    def release(*a,**kw):
        events.append(('release',position[1],kw))
        if failure=='gripper':return 99
        opened[0]=True;return 0
    observer=S(get_position=lambda:(0,list(position)),set_gripper_position=release,disconnect=lambda:None)
    monkeypatch.setattr(m,'_connect_xarm',lambda *a:observer)
    monkeypatch.setattr(m,'_read_gripper_feedback',lambda a:dict(read_errors=[],error_code=0,position_pulse=850 if opened[0] else 5))
    class Backend:
        def __init__(self,*a,**kw):
            events.append('backend');self._stop_requested=threading.Event();self._guard_error=None;self._mode_stop_lock=threading.Lock()
        def close_gripper(self,c):events.append('close');opened[0]=False
        def _check(self,label,code):
            if code:raise RuntimeError('gripper command failure')
        def joint_sequence(self,c,a):
            for y in range(0,61,3):
                if self._stop_requested.is_set():raise RuntimeError('stopped')
                position[1]=y;time.sleep(.02)
            return None,{}
        def home(self,c):events.append('home')
        def stop(self):events.append('stop');self._stop_requested.set()
        def close(self):pass
    monkeypatch.setattr(m,'XArmBackend',Backend)
    m.main()
    result=json.loads(next((tmp_path/'reports').glob('*/result.json')).read_text())
    if failure=='path':
        assert events==['path'] and not result.get('physical_execution',False)
    elif failure=='gripper':
        assert 'stop' in events and 'home' not in events and result['status']=='ERROR'
    else:
        assert result['status']=='COMPLETED' and result['fully_open_before_endpoint']
        release_event=next(e for e in events if isinstance(e,tuple))
        assert 12<=release_event[1]<48
        assert release_event[2]['wait_motion'] is False
        assert events.index('path')<events.index('backend')<events.index('close')
        assert events[-1]=='home'
