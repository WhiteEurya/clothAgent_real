"""Exercise the real CLI with injected, non-hardware runtime failures."""
import json
from types import SimpleNamespace
import pytest
from cloth_agent.experiments import graph_policy as gp
from cloth_agent.experiments import graph_policy_replay as pr
from tests.test_graph_policy_replay import policy,Runtime,Model


@pytest.mark.parametrize('online',[False,True])
def test_initialization_fault_has_footer_and_no_automatic_restart(tmp_path,monkeypatch,capsys,online):
    graph,_=policy(tmp_path);output=tmp_path/'run';calls=[]
    def fail(*args):
        calls.append('init')
        raise ValueError('right: controller fault/state 4, [11, 0]')
    monkeypatch.setattr(gp,'make_runtime',fail)
    monkeypatch.setattr(pr,'policy_requests',lambda *a,**k:Model())
    def forbidden(*args):raise AssertionError('no next episode after hardware fault')
    monkeypatch.setattr('builtins.input',forbidden)
    argv=['--graph',str(graph),'--output',str(output),'--execute']
    if online:argv+=['--online-growth','--episodes','2']
    assert pr.main(argv)==1
    text=capsys.readouterr().out
    assert '状态：CONTROLLER_FAULT' in text and 'error=11' in text
    assert text.rstrip().endswith(str(output/'episode_0001' if online else output))
    folder=output/'episode_0001' if online else output
    result=json.loads((folder/'cli_result.json').read_text())
    assert result['controller_fault']['error_code']==11
    assert 'controller fault' in (folder/'cli_exception.txt').read_text()
    assert calls==['init'] and not (output/'latest.json').exists()


@pytest.mark.parametrize('exc',[RuntimeError('model outage'),KeyboardInterrupt('SIGTERM')])
def test_replay_exception_keeps_saved_evidence_and_prints_after_cleanup(tmp_path,monkeypatch,capsys,exc):
    graph,_=policy(tmp_path);output=tmp_path/'run';runtime=Runtime();closed=[]
    def preview_close():closed.append('preview');print('preview stopped')
    runtime.preview=SimpleNamespace(close=preview_close)
    def fail(*args,**kwargs):
        output.mkdir()
        gp.frozen(output/'final_metrics.json',dict(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',error=f'{type(exc).__name__}: {exc}',
                   transitions=[dict(edge_id='executed-before-failure')],model_calls=3))
        raise exc
    monkeypatch.setattr(gp,'make_runtime',lambda *a:runtime)
    monkeypatch.setattr(pr,'policy_requests',lambda *a,**k:Model())
    monkeypatch.setattr(pr,'replay',fail)
    assert pr.main(['--graph',str(graph),'--output',str(output),'--execute'])==1
    result=json.loads((output/'cli_result.json').read_text())
    assert result['transitions']==[dict(edge_id='executed-before-failure')]
    assert result['model_calls']==3 and runtime.stopped and runtime.closed and closed==['preview']
    assert gp.load_frozen(output/'final_metrics.json')['model_calls']==3
    text=capsys.readouterr().out
    assert text.index('preview stopped') < text.index('Replay 结束原因')
    assert str(exc) in text


def test_cleanup_failure_does_not_hide_primary_or_skip_preview(tmp_path,monkeypatch,capsys):
    graph,_=policy(tmp_path);output=tmp_path/'run';runtime=Runtime();preview=[]
    def close():raise RuntimeError('disconnect failed')
    runtime.close=close;runtime.preview=SimpleNamespace(close=lambda:preview.append(True))
    monkeypatch.setattr(gp,'make_runtime',lambda *a:runtime)
    monkeypatch.setattr(pr,'policy_requests',lambda *a,**k:Model())
    monkeypatch.setattr(pr,'replay',lambda *a,**k:dict(status='ERROR',error='primary failure'))
    assert pr.main(['--graph',str(graph),'--output',str(output),'--execute'])==1
    result=json.loads((output/'cli_result.json').read_text())
    assert result['error']=='primary failure' and preview==[True]
    text=capsys.readouterr().out
    assert 'primary failure' in text and 'disconnect failed' in text


def test_successful_cli_is_still_success(tmp_path,monkeypatch):
    graph,_=policy(tmp_path);output=tmp_path/'run'
    monkeypatch.setattr(gp,'make_runtime',lambda *a:Runtime())
    monkeypatch.setattr(pr,'policy_requests',lambda *a,**k:Model())
    monkeypatch.setattr(pr,'replay',lambda *a,**k:dict(status='COMPLETED'))
    assert pr.main(['--graph',str(graph),'--output',str(output),'--execute'])==0
