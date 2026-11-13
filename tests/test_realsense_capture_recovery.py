from types import SimpleNamespace
import pytest
from cloth_agent.dual_cloud_observation import _acquire_frames

FRAME=SimpleNamespace(get_depth_frame=lambda:True,get_color_frame=lambda:True)
TIMEOUT="Frame didn't arrive within 5000"


def camera(scripts,events,start_errors=None):
    scripts=iter(scripts);start_errors=iter(start_errors or [None]*10)
    def pipeline():
        script=iter(next(scripts));error=next(start_errors)
        class Pipe:
            def start(self,config):
                events.append('start')
                if error:raise error
            def wait_for_frames(self,timeout):
                events.append('wait');value=next(script)
                if isinstance(value,BaseException):raise value
                return value
            def stop(self):events.append('stop')
        return Pipe()
    return SimpleNamespace(pipeline=pipeline)


def test_timeout_waits_without_closing_live_stream():
    events=[];rs=camera([[RuntimeError(TIMEOUT),FRAME,FRAME]],events)
    pipe,frames,audit=_acquire_frames(rs,None,'left123',2)
    assert events==['start','wait','wait','wait']
    assert audit['stream_attempts']==1 and frames is FRAME
    pipe.stop()


def test_restart_closes_failed_stream_before_reopening():
    events=[];rs=camera([[RuntimeError(TIMEOUT),RuntimeError(TIMEOUT)],[FRAME]],events)
    pipe,_,audit=_acquire_frames(rs,None,'right456',1)
    assert events==['start','wait','wait','stop','start','wait']
    assert audit['stream_attempts']==2
    pipe.stop()


def test_persistent_timeout_stops_after_two_streams_with_serial():
    events=[];rs=camera([[RuntimeError(TIMEOUT)]*2]*2,events)
    with pytest.raises(RuntimeError,match='right456: warmup failed after 2 stream attempts'):
        _acquire_frames(rs,None,'right456',60)
    assert events.count('start')==2 and events.count('stop')==2 and events.count('wait')==4


def test_keyboard_interrupt_closes_without_retry():
    events=[];rs=camera([[KeyboardInterrupt()]],events)
    with pytest.raises(KeyboardInterrupt):_acquire_frames(rs,None,'serial',60)
    assert events==['start','wait','stop']


def test_start_error_retries_then_reports_identity():
    events=[];rs=camera([[],[]],events,[RuntimeError('disconnected')]*2)
    with pytest.raises(RuntimeError,match='serial: start failed'):_acquire_frames(rs,None,'serial',60)
    assert events==['start','start']
