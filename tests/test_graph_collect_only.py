"""Collection with injected planner/runtime; never connects to a robot or GPT."""
import json
from types import SimpleNamespace

import pytest
from cloth_agent.experiments import graph_policy_replay as pr
from cloth_agent.experiments import graph_policy as gp
from cloth_agent.experiments.graph_policy_recovery import free_plan_one
from tests.test_online_graph_growth import Requests, frontier, state
from tests.test_graph_policy_replay import policy, matched
from tests.test_graph_policy import Runtime, SOURCE, match


def test_collect_only_never_selects_or_grounds_existing_edge_and_preserves_history(tmp_path,monkeypatch):
    path,graph=policy(tmp_path)
    previous=pr.OnlineGrowth(path,tmp_path/'previous')
    old_snapshot=previous.latest_path.read_bytes();base=path.read_bytes()
    def forbidden(*args,**kwargs):pytest.fail('collection must never select or ground a Graph action')
    monkeypatch.setattr(pr.OnlineGrowth,'select',forbidden)
    monkeypatch.setattr(pr,'growth_ground',forbidden)
    seen=[]
    def plan(runtime,before,folder,planning_graph,**kwargs):
        assert set(planning_graph)<= {'goal','goal_references'}
        assert kwargs['context']=={'reason':'COLLECT_ONLY_FULL_PLANNER'}
        seen.append(before.image)
        return frontier(runtime,before,folder,planning_graph,**kwargs)
    requests=Requests([matched('N0'),match('NO_MATCH'),matched('N1'),matched('N1')])
    result=pr.online_replay(path,tmp_path/'run',Runtime(),requests,max_steps=2,
        state_path=tmp_path/'previous/latest.json',collect_only=True,frontier_fn=plan)
    assert result['status']=='STEP_LIMIT',result
    assert len(seen)==2 and seen[0]!=seen[1]
    assert result['runner_config']['action_selection']=='FULL_GPT_EVERY_STEP'
    saved=state(result)
    assert len(saved['transitions'])==2
    assert [a['source_node'] for a in saved['attempts']]==['N0','N1']
    assert all(a['planner_source']=='GPT_FRONTIER' for a in saved['attempts'])
    assert saved['actions']['E0']==previous.state['actions']['E0']
    assert all(not a['preferred'] and not a['replayable'] for aid,a in saved['actions'].items() if aid!='E0')
    assert result['metrics']['graph_policy_actions']==0
    assert all(kind!='ground_edge' for kind,_,_ in requests.calls)
    restored=pr.OnlineGrowth(path,tmp_path/'restore',state_path=result['growth_state'])
    assert len(restored.state['transitions'])==2
    assert previous.latest_path.read_bytes()==old_snapshot and path.read_bytes()==base


def test_collect_only_records_new_node_without_reusing_graph(tmp_path):
    path,graph=policy(tmp_path)
    requests=Requests([matched('N0'),match('NO_MATCH'),match('NO_MATCH')],node=graph['nodes'][0])
    result=pr.online_replay(path,tmp_path/'run',Runtime(),requests,max_steps=1,
                           collect_only=True,frontier_fn=frontier)
    saved=state(result)
    assert len(saved['nodes'])==3
    assert saved['transitions'][0]['new_node_created']
    assert saved['transitions'][0]['selected_action']!='E0'
    assert saved['actions']['E0']['outcomes']['N1']['observed_count']==0


def test_collect_only_does_not_stop_on_graph_terminal_node(tmp_path,monkeypatch):
    path,graph=policy(tmp_path)
    original=pr.OnlineGrowth.__init__
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs);self.base['terminal_node_ids']=['N0']
    monkeypatch.setattr(pr.OnlineGrowth,'__init__',init)
    runtime=Runtime()
    result=pr.online_replay(path,tmp_path/'run',runtime,Requests([matched('N0'),matched('N0')]),
        max_steps=1,collect_only=True,frontier_fn=frontier)
    assert result['status']=='STEP_LIMIT' and len(runtime.executions)==1


@pytest.mark.parametrize('outcome,expected',[
    ({'status':'BLOCKED','physical_execution':False,'reason':'NO_ROBUST_ACTION: no cloth handle'},'ACTION_EXECUTION_FAILURE'),
    ({'status':'PLANNING_FAILED','physical_execution':False},'SAFETY_REJECT'),
    ({'status':'MODEL_REPORTED_COMPLETE','physical_execution':False},'FRONTIER_REPORTED_COMPLETE')])
def test_collect_only_refusal_safety_and_completion_do_not_create_success(tmp_path,outcome,expected):
    path,_=policy(tmp_path);calls=[]
    def plan(*args,**kwargs):calls.append(1);return dict(source=None,outcome=outcome)
    result=pr.online_replay(path,tmp_path/'run',Runtime(),Requests([matched('N0')]),
                           max_steps=3,collect_only=True,frontier_fn=plan)
    assert result['status']==expected and calls==[1]
    assert result['metrics']['total_physical_actions']==0
    assert not state(result)['transitions']


def test_collect_full_prompt_has_no_graph_repair_instruction(tmp_path):
    path,graph=policy(tmp_path);runtime=Runtime();before=runtime.observe(tmp_path/'obs');prompts=[]
    def model_call(prompt,output,**kwargs):
        prompts.append(prompt);output.mkdir();(output/'answer.txt').write_text(SOURCE)
        return dict(status='AGENT_FINISHED',session_id='collection-test')
    result=free_plan_one(runtime,before,tmp_path/'planner',{'goal':graph['goal']},
        context={'reason':'COLLECT_ONLY_FULL_PLANNER'},model_call=model_call)
    assert result['outcome']['status']=='EXECUTED'
    assert '独立规划本轮一个完整的布料操作' in prompts[0]
    assert '当前图策略未找到可用动作' not in prompts[0]
    assert '上一动作与最新匹配反馈' not in prompts[0]
    assert json.loads((tmp_path/'planner/round_001/dispatch_intent.json').read_text())['event']=='COLLECT_FULL_PLAN'


def test_episode_runner_passes_collect_only(tmp_path,monkeypatch):
    from cloth_agent.experiments import discovery_model
    path,_=policy(tmp_path)
    monkeypatch.setattr(gp,'make_runtime',lambda *a:Runtime())
    monkeypatch.setattr(discovery_model,'CodexMultimodal',lambda *a:None)
    monkeypatch.setattr(pr,'policy_requests',lambda *a,**k:Requests([matched('N0'),matched('N0')]))
    monkeypatch.setattr(pr,'free_plan_one',frontier)
    args=SimpleNamespace(graph=path,growth_state=None,episodes=1,model='gpt-6-astra',timeout_s=900,
        no_cache=False,max_calls=32,config=None,viser_port=8088,max_steps=1,match_threshold=.8,collect_only=True)
    result=pr.run_growth_episodes(args,tmp_path/'run')
    saved=state(result)
    assert saved['episodes'][-1]['runner_config']['collect_only']
    assert saved['attempts'][-1]['planner_source']=='GPT_FRONTIER'


def test_collect_only_cli_dry_run_and_validation(tmp_path,monkeypatch):
    path,_=policy(tmp_path)
    def forbidden(*args,**kwargs):pytest.fail('dry run must be offline')
    monkeypatch.setattr(gp,'make_runtime',forbidden)
    monkeypatch.setattr(pr,'policy_requests',forbidden)
    assert pr.main(['--graph',str(path),'--online-growth','--collect-only','--dry-run','--output',str(tmp_path/'dry')])==0
    with pytest.raises(SystemExit):pr.main(['--graph',str(path),'--collect-only','--dry-run'])
    with pytest.raises(SystemExit):pr.main(['--graph',str(path),'--online-growth','--collect-only','--grounding-mode','minimal','--dry-run'])
