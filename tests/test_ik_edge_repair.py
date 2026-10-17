import copy,json
import pytest
from tests.test_graph_policy_replay import policy,Runtime,Model,matched
from cloth_agent.experiments import graph_policy_replay as pr
from cloth_agent.experiments.edge_self_loop_repair import repairable_ik_failure


def failure():return dict(status='PLANNING_FAILED',physical_execution=False,failure_kind='IK_FAILED',
    ik_failure=dict(kind='IK_NO_ACCEPTABLE_SOLUTION',required_position_error_mm=.5,required_orientation_error_deg=.2))


def test_recorded_ik_failures_are_eligible():
    from pathlib import Path
    for name in ['policy_replay_20261015T065108688654Z','policy_replay_20261015T065638612773Z']:
        path=Path('reports/seed_contact_tcp_20261015/compiled_policy')/name/'final_metrics.json'
        if not path.exists():pytest.skip('historical run unavailable')
        assert repairable_ik_failure(json.loads(path.read_text())['execution'])


@pytest.mark.parametrize('change',[dict(physical_execution=True),dict(failure_kind='COLLISION'),dict(status='EXECUTION_FAILED'),dict(ik_failure={})])
def test_non_ik_or_dispatched_failure_not_eligible(change):
    assert not repairable_ik_failure({**failure(),**change})


@pytest.mark.parametrize('repair_outcome',[dict(status='EXECUTED',physical_execution=True),failure()])
def test_ik_repair_bounded_and_rematches(tmp_path,monkeypatch,repair_outcome):
    path,g=policy(tmp_path);runtime=Runtime();model=Model([matched('N0'),matched('N1')])
    monkeypatch.setattr(pr,'execute_policy_edge',lambda *a:failure())
    contexts=[]
    def repair(*args,**kw):
        contexts.append(kw['context'])
        return dict(source='motion.grasp(right=[.5,.1,.05,3.14,0,0])\n',outcome=copy.deepcopy(repair_outcome))
    result=pr.replay(path,tmp_path/'run',runtime,model,repair_self_loop=True,fallback_fn=repair)
    assert len(contexts)==1 and result['edge_replans']==1
    context=contexts[0]
    assert context['reason']=='REPAIR_EXISTING_EDGE_AFTER_IK_FAILED'
    assert context['failure']['result_node'] is None
    assert context['failure']['transition']['execution']['ik_failure']['required_position_error_mm']==.5
    if repair_outcome['status']=='EXECUTED':
        assert result['transitions'][0]['actual_node']=='N1'
        assert result['transitions'][0]['mode']=='EDGE_REPAIR'
    else:
        assert result['status']=='SAFETY_REJECT' and result['transitions']==[]


def test_collision_stops_without_repair(tmp_path,monkeypatch):
    path,g=policy(tmp_path)
    monkeypatch.setattr(pr,'execute_policy_edge',lambda *a:{**failure(),'failure_kind':'COLLISION'})
    def forbidden(*a,**kw):raise AssertionError('collision must not trigger repair')
    result=pr.replay(path,tmp_path/'run',Runtime(),Model(),repair_self_loop=True,fallback_fn=forbidden)
    assert result['status']=='SAFETY_REJECT' and result['edge_replans']==0


def test_actual_adapter_blocks_unchanged_repair_before_dispatch(tmp_path):
    from tests.test_graph_policy import SOURCE,observation
    from cloth_agent.experiments.graph_policy_recovery import free_plan_one
    path,g=policy(tmp_path);obs=observation(tmp_path/'current');runtime=Runtime()
    def model_call(prompt,**kw):
        out=kw['output'];out.mkdir(parents=True);(out/'answer.txt').write_text(SOURCE)
        return dict(status='AGENT_FINISHED',session_id='unchanged-repair')
    result=free_plan_one(runtime,obs,tmp_path/'repair',g,model_call=model_call,
        context=dict(reason='REPAIR_EXISTING_EDGE_AFTER_IK_FAILED',failure=dict(transition=dict(command_source=SOURCE))))
    assert result['outcome']['failure_kind']=='UNCHANGED_REPAIR_COMMAND'
    assert result['outcome']['physical_execution'] is False
    assert runtime.executions==[]
