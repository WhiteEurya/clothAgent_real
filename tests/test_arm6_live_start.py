import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from cloth_agent.experiments.arm6_live_start import PlacementSession


def setup_session(tmp_path, match):
    captures=[]
    def capture(folder):
        folder.mkdir()
        image=folder/'rgb.png'; Image.new('RGB',(40,40),'green').save(image)
        captures.append(image)
        return SimpleNamespace(image=image)
    runtime=SimpleNamespace(capture_current=capture, cancel=threading.Event())
    graph={'nodes':[dict(node_id=n,representative_observations=['o']) for n in ['n092','n019']]}
    session=PlacementSession(runtime,lambda *a:dict(match),graph,'n092',.8,tmp_path/'placement')
    return session,captures


def prediction(**kw):
    return dict(dict(status='MATCH',node_id='n092',candidate_nodes=['n092'],confidence=.9,
                     matched_evidence=['visible correspondence'],violated_conditions=[],unresolved_conditions=[]),**kw)


@pytest.mark.parametrize('changes',[
    dict(node_id='n019',candidate_nodes=['n019']),
    dict(status='NO_MATCH',node_id='',candidate_nodes=[]),
    dict(status='AMBIGUOUS',node_id='',candidate_nodes=['n092','n019']),
    dict(confidence=.79),dict(unresolved_conditions=['hidden boundary']),
])
def test_only_reliable_expected_node_enables_continue(tmp_path,changes):
    session,_=setup_session(tmp_path,prediction(**changes))
    session.check()
    assert not session.ready
    with pytest.raises(ValueError):session.accept()


def test_resume_invalidates_result_and_recaptures_rgbd(tmp_path):
    session,captures=setup_session(tmp_path,prediction())
    with pytest.raises(ValueError):session.accept()
    snapshots=[]
    session.check(snapshots.append)
    obs,match=session.accept()
    assert obs is snapshots[0] and match==prediction()
    session.resume()
    assert session.match is None and session.observation is None
    with pytest.raises(ValueError):session.accept()
    session.check()
    new,_=session.accept()
    assert new.image!=obs.image and len(captures)==2
    assert (tmp_path/'placement/attempt_0000/result.json').is_file()
    assert (tmp_path/'placement/attempt_0001/result.json').is_file()


def test_capture_failure_never_reuses_prior_match(tmp_path):
    session,_=setup_session(tmp_path,prediction())
    session.check()
    def fail(*a):raise RuntimeError('camera disconnected')
    session.runtime.capture_current=fail
    with pytest.raises(RuntimeError):session.check()
    assert not session.ready and session.observation is None
    with pytest.raises(ValueError):session.accept()


@pytest.mark.parametrize('cancel',[False,True])
def test_hidden_tk_live_preview_retry_and_stop_without_hardware(tmp_path,monkeypatch,cancel):
    import os
    import time
    import tkinter as tk
    import numpy as np
    from cloth_agent.experiments.arm6_live_start import prepare_live_start
    if not os.environ.get('DISPLAY'):pytest.skip('requires a display for hidden Tk smoke test')
    session,captures=setup_session(tmp_path,prediction())
    reference=tmp_path/'ref.png';Image.new('RGB',(40,40),'blue').save(reference)
    reads=[]
    def read():
        time.sleep(.01)
        reads.append(True)
        return np.zeros((40,40,3),dtype=np.uint8),None
    session.runtime.camera=SimpleNamespace(read=read)
    calls=[]
    def recognize(*a):
        calls.append(a)
        return prediction(node_id='n019',candidate_nodes=['n019']) if len(calls)==1 else prediction()
    original_tk=tk.Tk
    errors=[]
    def create():
        root=original_tk();root.withdraw()
        phase=[0]
        started=time.monotonic()
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        def drive():
            buttons={w.cget('text'):w for w in descendants(root) if isinstance(w,tk.Button)}
            check=buttons['Recognize / 识别'];resume=buttons['Resume live / 继续摆放']
            accept=buttons['Continue / 确认起点']
            if time.monotonic()-started>6:
                errors.append('UI test timeout');buttons['Stop'].invoke();return
            if cancel and reads:
                buttons['Stop'].invoke();return
            if phase[0]==0 and reads:
                assert accept.cget('state')=='disabled'
                check.invoke();phase[0]=1
            elif phase[0]==1 and resume.cget('state')=='normal':
                assert accept.cget('state')=='disabled'
                resume.invoke();phase[0]=2
            elif phase[0]==2 and check.cget('state')=='normal':
                check.invoke();phase[0]=3
            elif phase[0]==3 and accept.cget('state')=='normal':
                accept.invoke();return
            root.after(40,drive)
        root.after(40,drive)
        return root
    monkeypatch.setattr(tk,'Tk',create)
    args=(session.runtime,reference,recognize,session.graph,'n092',.8,tmp_path/'ui')
    if cancel:
        with pytest.raises(KeyboardInterrupt):prepare_live_start(*args)
        assert not calls
    else:
        observation,match=prepare_live_start(*args)
        assert observation.image==captures[-1] and len(calls)==2
        assert match['node_id']=='n092'
    assert not errors and reads
