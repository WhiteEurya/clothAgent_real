import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cloth_agent.motion_recording import MotionRecorder
from .test_dual_arm_runtime import scene
from .test_codex_native_runtime import runtime_for


class Camera:
    def __init__(self,fail=False):self.calls=0;self.fail=fail
    def read(self):
        time.sleep(.003);self.calls+=1
        if self.fail and self.calls>2:raise RuntimeError('camera disconnected')
        return np.full((48,64,3),[240,20,10],dtype=np.uint8),None


@pytest.mark.parametrize('fail',[False,True])
def test_mp4_finalized_and_camera_failure_recorded(tmp_path,fail):
    r=MotionRecorder(Camera(fail),tmp_path,fps=30)
    r.start();time.sleep(.03);report=r.stop()
    cap=cv2.VideoCapture(str(tmp_path/'execution.mp4'))
    try:
        ok,frame=cap.read();assert ok
        assert frame.shape==(48,64,3)
        assert frame[20,20,2]>200 # RGB red converted to BGR for OpenCV.
    finally:cap.release()
    assert report['frames']>=1
    assert report['status']==('FAILED' if fail else 'RECORDED')
    assert not r.thread.is_alive()
    assert len(json.loads((tmp_path/'recording.json').read_text())['frame_times_s'])==report['frames']


@pytest.mark.parametrize('fails',[False,True])
def test_recording_wraps_execution_and_closes_on_exception(scene,tmp_path,monkeypatch,fails):
    r=runtime_for(scene);r.camera=Camera();events=[]
    initial={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    obs=SimpleNamespace(meta={'joints_deg':initial},validate_for=lambda *a,**kw:None)
    monkeypatch.setattr(r,'plan',lambda *a:[])
    monkeypatch.setattr(r,'arrival',lambda *a:None)
    monkeypatch.setattr(MotionRecorder,'start',lambda self:events.append('start'))
    monkeypatch.setattr(MotionRecorder,'stop',lambda self:events.append('stop') or {'status':'RECORDED'})
    def execute(*args):
        events.append('execute')
        if fails:raise KeyboardInterrupt()
        return {'status':'EXECUTED'}
    monkeypatch.setattr(r,'_execute_operations',execute)
    try:
        if fails:
            with pytest.raises(KeyboardInterrupt):r.execute('motion.move_tcp(left=[.25,-.12,.18,0,0,0])',obs,tmp_path)
        else:
            result=r.execute('motion.move_tcp(left=[.25,-.12,.18,0,0,0])',obs,tmp_path)
            assert result['recording']['status']=='RECORDED'
        assert events==['start','execute','stop']
    finally:r.camera=None;r.close()
