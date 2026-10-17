import json
import pytest
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.experiments import graph_policy_replay as pr
from cloth_agent.experiments.replay_failures import controller_fault_details
from cloth_agent.experiments.replay_stop_summary import describe_stop
from tests.test_graph_policy_replay import policy,Runtime,Model

FAULT='right: controller fault/state 4, [11, 0]'

@pytest.mark.parametrize('stage',['initial_observation','recapture','dispatch'])
def test_controller_fault_never_enters_recovery(tmp_path,monkeypatch,stage):
    path,g=policy(tmp_path);runtime=Runtime();model=Model();attempts=[]
    def fail(*args,**kwargs):raise DualArmError(FAULT)
    if stage=='initial_observation':runtime.observe=fail
    elif stage=='recapture':monkeypatch.setattr(pr,'grounding_with_recapture',fail)
    else:monkeypatch.setattr(pr,'execute_policy_edge',fail)
    def forbidden(*a,**k):attempts.append(k);raise AssertionError('must not recover hardware fault')
    result=pr.replay(path,tmp_path/'run',runtime,model,max_fallbacks=3,repair_self_loop=True,
                     applicability_recovery=True,fallback_fn=forbidden)
    assert result['status']=='CONTROLLER_FAULT' and result['controller_fault']['error_code']==11
    assert result['error']==FAULT and not attempts and not runtime.executions
    assert runtime.stopped and runtime.closed
    assert json.loads((tmp_path/'run/final_metrics.json').read_text())['status']=='CONTROLLER_FAULT'


def test_failure_during_stop_preserves_primary_controller_fault(tmp_path):
    path,g=policy(tmp_path);runtime=Runtime()
    def fail(*a,**kw):raise DualArmError(FAULT)
    def stop():raise RuntimeError('stop transport lost')
    runtime.observe=fail;runtime.stop=stop
    result=pr.replay(path,tmp_path/'run',runtime,Model())
    assert result['status']=='CONTROLLER_FAULT' and result['error']==FAULT
    assert 'stop transport lost' in result['stop_error']


def test_legacy_summary_corrects_misleading_failure_category(tmp_path):
    s=describe_stop(tmp_path,dict(status='EDGE_INSTANTIATION_FAILURE',error=FAULT))
    assert s['status']=='CONTROLLER_FAULT' and '不是视觉' in s['reason']


def test_not_a_controller_fault():
    assert controller_fault_details('minimal anchors unresolved/low confidence') is None


def test_returned_fallback_fault_is_not_a_visual_failure(tmp_path):
    from tests.test_graph_policy import match
    from tests.test_online_graph_growth import Requests
    path,g=policy(tmp_path);runtime=Runtime();calls=[]
    def fallback(*a,**k):
        calls.append(1)
        return dict(outcome=dict(status='EXECUTION_FAILED',physical_execution=True,error=FAULT))
    result=pr.replay(path,tmp_path/'run',runtime,Requests([match('NO_MATCH')],node=g['nodes'][0]),max_fallbacks=3,fallback_fn=fallback)
    assert result['status']=='CONTROLLER_FAULT' and result['error']==FAULT
    assert runtime.stopped and len(calls)==1


@pytest.mark.parametrize('stage',['grounding','returned_execution'])
def test_online_controller_fault_preserves_error_and_no_transition(tmp_path,monkeypatch,stage):
    from tests.test_online_graph_growth import Requests,matched
    path,g=policy(tmp_path);runtime=Runtime()
    if stage=='grounding':
        def fail(*a,**k):raise DualArmError(FAULT)
        monkeypatch.setattr(pr,'growth_ground',fail)
    else:
        runtime.execute=lambda *a,**k:dict(status='EXECUTION_FAILED',physical_execution=True,error=FAULT)
    result=pr.online_replay(path,tmp_path/'run',runtime,Requests([matched('N0')]))
    assert result['status']=='CONTROLLER_FAULT' and result['error']==FAULT
    assert result['metrics']['current_transitions']==0 and runtime.stopped
