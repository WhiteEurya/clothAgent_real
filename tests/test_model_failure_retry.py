import json
import pytest
from cloth_agent.experiments.graph_policy import Requests
from cloth_agent.experiments.model_errors import ModelInvocationError

@pytest.mark.parametrize('retryable,failures,expected',[(True,1,2),(True,5,2),(False,1,1)])
def test_bounded_model_only_retry(tmp_path,retryable,failures,expected):
    class Backend:
        model='fixture'
        calls=0
        def invoke(self,*args):
            self.calls+=1
            if self.calls<=failures:raise ModelInvocationError('offline transport fixture',retryable=retryable,details={'returncode':1})
            return {'text':'{}'}
    backend=Backend();r=Requests(tmp_path,backend,schemas={'test':{'type':'object'}},prompts={'test':'test'},common='')
    if failures==1 and retryable:assert r.ask('test',{},[])=={}
    else:
        with pytest.raises(ModelInvocationError):r.ask('test',{},[])
    assert backend.calls==expected
    rows=[json.loads(l) for l in (tmp_path/'requests.jsonl').read_text().splitlines()]
    assert len(rows)==expected and rows[0]['model_failure']['returncode']==1


def test_interrupt_never_retries(tmp_path):
    class Backend:
        model='fixture';calls=0
        def invoke(self,*a):self.calls+=1;raise KeyboardInterrupt()
    b=Backend();r=Requests(tmp_path,b,schemas={'t':{}},prompts={'t':'test'},common='')
    with pytest.raises(KeyboardInterrupt):r.ask('t',{},[])
    assert b.calls==1


@pytest.mark.parametrize('stderr,retryable',[('Reading prompt from stdin...',True),('bwrap: namespace setup failed',False)])
def test_backend_preserves_failure_event_details(tmp_path,monkeypatch,stderr,retryable):
    from cloth_agent.experiments import discovery_model as dm
    monkeypatch.setattr(dm,'build_command',lambda *a,**k:['fake','--model','fixture','-'])
    monkeypatch.setattr(dm,'isolated_command',lambda *a:[])
    class Process:
        returncode=1
        def communicate(self,prompt,timeout):pass
    def popen(cmd,**kwargs):
        kwargs['stdout'].write(json.dumps(dict(type='error',message='connection reset fixture'))+'\n')
        kwargs['stdout'].flush();kwargs['stderr'].write(stderr);kwargs['stderr'].flush()
        return Process()
    monkeypatch.setattr(dm.subprocess,'Popen',popen)
    backend=dm.CodexMultimodal(binary='true')
    with pytest.raises(ModelInvocationError) as caught:backend.invoke('test',[],{})
    assert caught.value.retryable is retryable
    assert caught.value.details['returncode']==1
    assert caught.value.details['failure_events'][0]['message']=='connection reset fixture'
