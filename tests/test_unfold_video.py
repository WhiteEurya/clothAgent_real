import json
import shutil
import time
from types import SimpleNamespace
import numpy as np
import pytest
from cloth_agent.unfold_video import UnfoldVideo
from cloth_agent.perception import PerceptionError


class FakeCamera:
    def __init__(self,fail_after=None):
        self.starts=0;self.stops=0;self.frames=0;self.intrinsics=np.eye(3);self.fail_after=fail_after
    def start(self):self.starts+=1
    def read(self):
        time.sleep(.02)
        self.frames+=1
        if self.fail_after and self.frames>=self.fail_after:raise RuntimeError('camera disconnected')
        self.timestamp_ns=self.frames*20000000
        return np.full((48,64,3),self.frames%255,np.uint8),np.full((48,64),self.frames,float)
    def stop(self):self.stops+=1


def make(tmp_path,fail_after=None):
    if not shutil.which('ffmpeg'):pytest.skip('ffmpeg not available')
    spec=SimpleNamespace(label='A',backend='zed',serial='test',extrinsics_file='unused')
    cfg=SimpleNamespace(cameras=[spec],active_camera_labels=['A'],width=64,height=48,fps=30,warmup_frames=2,temporal_median_frames=3)
    camera=FakeCamera(fail_after)
    return UnfoldVideo(cfg,tmp_path/'video',tmp_path,camera=camera),camera


def test_continuous_video_and_fresh_observations_share_one_camera(tmp_path,monkeypatch):
    recorder,camera=make(tmp_path)
    try:
        recorder.start()
        borrowed=recorder.camera_factory(recorder.spec,64,48,30);borrowed.start()
        a,da=borrowed.read();b,db=borrowed.read()
        assert db[0,0]>da[0,0] and b[0,0,0]>a[0,0,0]
        borrowed.stop();assert camera.stops==0
        from cloth_agent.perception import capture_two_view_rgbd
        monkeypatch.setattr('cloth_agent.perception.load_extrinsics',lambda p:np.eye(4))
        frames=capture_two_view_rgbd(recorder.config,camera_factory=recorder.camera_factory)
        assert len(frames)==1 and frames[0].depth_m[0,0]>db[0,0]
        assert camera.starts==1 and camera.stops==0
        count=camera.frames;time.sleep(.25);assert camera.frames>count
        with pytest.raises(PerceptionError,match='configuration mismatch'):
            recorder.camera_factory(recorder.spec,1280,720,30)
    finally:recorder.close()
    recorder.close()
    assert camera.starts==camera.stops==1
    manifest=json.loads((tmp_path/'video_recording.json').read_text())
    assert manifest['status']=='COMPLETED' and manifest['frames']>=2
    assert manifest['playback_speed']==32 and manifest['original_deleted']
    assert not (tmp_path/'video/zed_rgb.mp4').exists()
    assert recorder.path.name=='zed_rgb_32x.mp4'
    import cv2
    cap=cv2.VideoCapture(str(recorder.path));ok,frame=cap.read();cap.release()
    assert ok and frame.shape==(48,64,3)


def test_camera_error_propagates_and_video_is_finalized(tmp_path):
    recorder,camera=make(tmp_path,fail_after=5)
    try:
        recorder.start()
        deadline=time.monotonic()+12
        while recorder.error is None and time.monotonic()<deadline:time.sleep(.05)
        with pytest.raises(PerceptionError,match='reconnect exhausted'):recorder.read()
    finally:recorder.close()
    assert camera.starts==camera.stops==4
    assert json.loads((tmp_path/'video_recording.json').read_text())['status']=='FAILED'


def test_transient_camera_error_reconnects_without_stopping_recording(tmp_path):
    recorder,camera=make(tmp_path)
    original=camera.read;failed=[False]
    def transient():
        if camera.frames>=2 and not failed[0]:
            failed[0]=True
            raise RuntimeError('camera disconnected')
        return original()
    camera.read=transient
    try:
        recorder.start();time.sleep(1.3)
        rgb,_=recorder.read()
        assert rgb.shape==(48,64,3) and recorder.error is None
        assert recorder.camera_reconnects==1
    finally:recorder.close()
    assert camera.starts==camera.stops==2


def test_low_disk_never_opens_camera(tmp_path,monkeypatch):
    recorder,camera=make(tmp_path)
    monkeypatch.setattr('cloth_agent.unfold_video.shutil.disk_usage',lambda p:SimpleNamespace(free=100))
    with pytest.raises(RuntimeError,match='1 GiB'):recorder.start()
    assert camera.starts==0


def test_failed_speed_conversion_keeps_original(tmp_path,monkeypatch):
    recorder,camera=make(tmp_path)
    def fail(*a,**k):raise RuntimeError('encoder unavailable')
    monkeypatch.setattr('cloth_agent.rollout_recorder.speed_up_mp4',fail)
    try:recorder.start();time.sleep(.1)
    finally:recorder.close()
    assert (tmp_path/'video/zed_rgb.mp4').stat().st_size>0
    data=json.loads((tmp_path/'video_recording.json').read_text())
    assert data['conversion_status']=='FAILED'
    assert 'encoder unavailable' in data['conversion_error']
