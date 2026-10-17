"""Offline Graph self-loop replacement tests; fake robot and model only."""
import copy
import json

import pytest

from cloth_agent.experiments import graph_policy_replay as pr
from cloth_agent.experiments import graph_policy as gp
from cloth_agent.experiments.growth_edge_recovery import select_recorded_action
from cloth_agent.experiments.graph_policy_recovery import free_plan_one
from tests.test_graph_policy import Runtime, SOURCE, match
from tests.test_graph_policy_replay import policy, matched
from tests.test_online_graph_growth import Requests, frontier, state


REPAIR=SOURCE.replace('0.035','0.032')


def repair(runtime,before,folder,graph,**kwargs):
    folder.mkdir()
    return dict(source=REPAIR,outcome=runtime.execute(REPAIR,before,folder))


def run(tmp_path, *, matches=None, planner=repair, runtime=None, requests=None, steps=1, live_port=None):
    path,graph=policy(tmp_path)
    requests=requests or Requests(matches or [matched('N0'),matched('N0'),matched('N1')])
    runtime=runtime or Runtime()
    result=pr.online_replay(path,tmp_path/'run',runtime,requests,max_steps=steps,
                           recover_no_progress=True,frontier_fn=planner,live_graph_port=live_port)
    return path,graph,result,state(result),runtime


def test_self_loop_replaces_same_edge_after_success_and_reloads(tmp_path):
    path,graph,result,s,runtime=run(tmp_path,live_port=0)
    assert result['status']=='STEP_LIMIT',result
    assert len(runtime.executions)==2
    assert set(s['actions'])=={'E0'} and set(s['nodes'])=={'N0','N1'}
    edge=s['actions']['E0']
    assert edge['edge']['seed_action']['exact_command']==gp.command(REPAIR)
    assert edge['revisions'][0]['previous_edge']==graph['edges'][0]
    assert [a['planner_source'] for a in s['attempts']]==['GRAPH','GPT_RECOVERY']
    assert result['recoveries'][0]['status']=='REMATCHED'
    assert result['recoveries'][0]['before']!=result['recoveries'][0]['after']
    assert result['recoveries'][0]['replacement']=='COMMITTED'
    assert gp.load_frozen(path)==graph
    restored=pr.OnlineGrowth(path,tmp_path/'restored',state_path=result['growth_state'])
    assert restored.state['actions']['E0']==edge
    live=json.loads((tmp_path/'run/live_graph/state.json').read_text())
    assert live['recoveries'][0]['matched_node']=='N1'
    assert len(live['recoveries'][0]['images'])==2


@pytest.mark.parametrize('matches,effect',[
    ([matched('N0'),matched('N0'),matched('N0')],'PROGRESS'),
    ([matched('N0'),matched('N0'),matched('N1')],'UNKNOWN'),
    ([matched('N0'),matched('N0'),match('AMBIGUOUS')],'PROGRESS'),
])
def test_unverified_recovery_keeps_original_edge_and_no_new_nodes(tmp_path,matches,effect):
    path,graph,result,s,runtime=run(tmp_path,requests=Requests(matches,effect=effect),steps=5)
    assert result['status']=='RECOVERY_UNRESOLVED',result
    assert len(runtime.executions)==2
    assert s['actions']['E0']['edge']==graph['edges'][0]
    assert not s['actions']['E0'].get('revisions')
    assert set(s['nodes'])=={'N0','N1'}
    pr.OnlineGrowth(path,tmp_path/'reload',state_path=result['growth_state'])


def test_unknown_initial_observation_creates_node_and_plans(tmp_path):
    path,graph=policy(tmp_path);runtime=Runtime()
    requests=Requests([match('NO_MATCH'),match('NO_MATCH'),matched('N1')],node=graph['nodes'][0])
    result=pr.online_replay(path,tmp_path/'run',runtime,requests,max_steps=1,
                           recover_no_progress=True,frontier_fn=frontier)
    assert result['status']=='STEP_LIMIT',result
    saved=state(result);new=result['initial_node']
    assert new=='P000002' and saved['nodes'][new]['status']=='PROVISIONAL'
    assert saved['attempts'][0]['source_node']==new
    assert saved['attempts'][0]['planner_source']=='GPT_FRONTIER'
    assert saved['nodes'][new]['node']['representative_observations']
    assert len(runtime.executions)==1
    pr.OnlineGrowth(path,tmp_path/'reload',state_path=result['growth_state'])


def test_unknown_action_outcome_creates_node_and_continues_frontier(tmp_path):
    path,graph=policy(tmp_path);runtime=Runtime()
    requests=Requests([matched('N0'),match('NO_MATCH'),match('NO_MATCH'),
                       match('NO_MATCH'),matched('N1')],node=graph['nodes'][0])
    result=pr.online_replay(path,tmp_path/'run',runtime,requests,max_steps=2,
                           recover_no_progress=True,frontier_fn=frontier)
    assert result['status']=='STEP_LIMIT',result
    saved=state(result)
    assert saved['transitions'][0]['transition_type']=='NEW_NODE'
    assert saved['transitions'][0]['result_node']=='P000002'
    assert saved['attempts'][1]['source_node']=='P000002'
    assert saved['attempts'][1]['planner_source']=='GPT_FRONTIER'
    assert len(runtime.executions)==2 and not result['recoveries']
    assert saved['actions']['E0']['edge']==graph['edges'][0]
    pr.OnlineGrowth(path,tmp_path/'reload',state_path=result['growth_state'])


@pytest.mark.parametrize('effect',['PROGRESS','UNKNOWN'])
def test_unknown_recovery_outcome_creates_node_and_continues_without_claiming_repair(tmp_path,effect):
    path,graph=policy(tmp_path);runtime=Runtime()
    requests=Requests([matched('N0'),matched('N0'),match('NO_MATCH'),match('NO_MATCH'),
                       match('NO_MATCH'),matched('N1')],node=graph['nodes'][0],effect=effect)
    result=pr.online_replay(path,tmp_path/'run',runtime,requests,max_steps=2,
                           recover_no_progress=True,frontier_fn=repair,live_graph_port=0)
    assert result['status']=='STEP_LIMIT',result
    saved=state(result);recovery=result['recoveries'][0]
    assert recovery['new_node_created'] and recovery['matched_node']=='P000002'
    assert recovery['replacement']=='NOT_COMMITTED'
    assert saved['actions']['E0']['edge']==graph['edges'][0]
    assert saved['transitions'][1]['transition_type']=='NEW_NODE'
    assert saved['attempts'][-1]['source_node']=='P000002'
    assert saved['attempts'][-1]['planner_source']=='GPT_FRONTIER'
    assert len(runtime.executions)==3
    live=json.loads((tmp_path/'run/live_graph/state.json').read_text())
    assert any(n['id']=='P000002' and n['images'] for n in live['nodes'])
    pr.OnlineGrowth(path,tmp_path/'reload',state_path=result['growth_state'])


def test_ambiguous_initial_observation_does_not_create_node(tmp_path):
    _,_,result,s,runtime=run(tmp_path,requests=Requests([match('AMBIGUOUS')]))
    assert result['status']=='AMBIGUOUS'
    assert set(s['nodes'])=={'N0','N1'} and not runtime.executions


@pytest.mark.parametrize('status',['BLOCKED','PLANNING_FAILED'])
def test_recovery_refusal_or_safety_reject_stops(tmp_path,status):
    def rejected(*args,**kwargs):
        return dict(source=None,outcome=dict(status=status,physical_execution=False))
    _,graph,result,s,runtime=run(tmp_path,planner=rejected,steps=4)
    assert result['status']=='RECOVERY_UNRESOLVED'
    assert len(runtime.executions)==1
    assert s['actions']['E0']['edge']==graph['edges'][0]


def test_unchanged_recovery_is_not_dispatched(tmp_path):
    def unchanged(runtime,before,folder,graph,**kwargs):
        source=kwargs['context']['failure']['transition']['command_source']
        return dict(source=source,outcome=runtime.execute(source,before,folder))
    _,_,result,_,runtime=run(tmp_path,planner=unchanged)
    assert result['status']=='RECOVERY_UNRESOLVED'
    assert len(runtime.executions)==1


def test_no_recovery_on_execution_failure(tmp_path):
    def forbidden(*a,**k):pytest.fail('failed dispatch must not invoke recovery')
    _,_,result,_,runtime=run(tmp_path,runtime=Runtime('EXECUTION_FAILED'),planner=forbidden)
    assert result['status']=='ACTION_EXECUTION_FAILURE'
    assert len(runtime.executions)==1 and not result['recoveries']


def test_template_failure_does_not_commit(tmp_path):
    class BrokenTemplate(Requests):
        def ask(self,kind,*args,**kwargs):
            if kind=='growth_action':raise RuntimeError('template unavailable')
            return super().ask(kind,*args,**kwargs)
    _,graph,result,s,runtime=run(tmp_path,
        requests=BrokenTemplate([matched('N0'),matched('N0'),matched('N1')]))
    assert result['status']=='RECOVERY_UNRESOLVED'
    assert len(runtime.executions)==2 and len(s['transitions'])==2
    assert s['actions']['E0']['edge']==graph['edges'][0]


def test_only_one_recovery_action_per_episode(tmp_path,monkeypatch):
    path,_=policy(tmp_path)
    first=pr.online_replay(path,tmp_path/'collect',Runtime(),
        Requests([matched('N1'),match('NO_MATCH'),matched('N0')]),
        collect_only=True,max_steps=1,frontier_fn=frontier)
    # N0 -> N0; recovery N0 -> N1; the recorded N1 action then self-loops.
    req=Requests([matched('N0'),matched('N0'),matched('N1'),matched('N1')])
    calls=[]
    def recover(*a,**k):calls.append(1);return repair(*a,**k)
    runtime=Runtime()
    result=pr.online_replay(path,tmp_path/'run',runtime,req,max_steps=5,
        state_path=first['growth_state'],recover_no_progress=True,frontier_fn=recover)
    assert result['status']=='RECOVERY_UNRESOLVED',result
    assert len(calls)==1 and len(runtime.executions)==3
    assert len(result['recoveries'])==1


def test_recovery_wrong_existing_target_does_not_commit(tmp_path):
    path,graph=policy(tmp_path)
    # Add another observed state through the existing collection path.
    collected=pr.online_replay(path,tmp_path/'collect',Runtime(),
        Requests([match('NO_MATCH'),matched('P000002')],node=graph['nodes'][0]),
        max_steps=1,collect_only=True,frontier_fn=frontier)
    result=pr.online_replay(path,tmp_path/'run',Runtime(),
        Requests([matched('N0'),matched('N0'),match('NO_MATCH'),matched('P000002')]),
        state_path=collected['growth_state'],max_steps=1,recover_no_progress=True,frontier_fn=repair)
    assert result['status']=='RECOVERY_UNRESOLVED'
    assert state(result)['actions']['E0']['edge']==graph['edges'][0]


def test_controller_failure_in_recovery_stops(tmp_path):
    class Failure(Runtime):
        def execute(self,*args):
            result=super().execute(*args)
            if len(self.executions)==2:
                result.update(status='EXECUTION_FAILED',error='right: controller fault/state 4, [11, 0]')
            return result
    _,_,result,_,runtime=run(tmp_path,runtime=Failure())
    assert result['status']=='CONTROLLER_FAULT' and runtime.stopped
    assert len(runtime.executions)==2 and len(result['recoveries'])==1


def test_revision_replay_receives_corrected_edge_and_does_not_surface_ground(tmp_path,monkeypatch):
    path,_,first,_,_=run(tmp_path)
    def forbidden(*a,**k):pytest.fail('revised grasp height must not use surface-only grounder')
    monkeypatch.setattr(pr,'growth_ground',forbidden)
    contexts=[]
    def replay_revision(*args,**kwargs):
        contexts.append(kwargs['context']);return repair(*args,**kwargs)
    result=pr.online_replay(path,tmp_path/'next',Runtime(),Requests([matched('N0'),matched('N1')]),
        state_path=first['growth_state'],max_steps=1,recover_no_progress=True,frontier_fn=replay_revision)
    assert result['status']=='STEP_LIMIT'
    assert contexts[0]['reason']=='REPLAY_REVISED_EDGE'
    assert contexts[0]['selected_edge']['seed_action']['exact_command']==gp.command(REPAIR)
    assert state(result)['attempts'][-1]['planner_source']=='GRAPH_REVISED'


def test_tampered_revision_is_rejected(tmp_path):
    path,_,result,s,_=run(tmp_path)
    bad=copy.deepcopy(s)
    bad['actions']['E0']['edge']['seed_action']['exact_command']=gp.command(SOURCE)
    file=tmp_path/'tampered.json';gp.frozen(file,bad)
    with pytest.raises(ValueError,match='revision'):
        pr.OnlineGrowth(path,tmp_path/'reload',state_path=file)


def test_collected_actions_are_selectable_without_falsifying_eligibility(tmp_path):
    path,_=policy(tmp_path)
    collected=pr.online_replay(path,tmp_path/'collected',Runtime(),Requests([matched('N0'),match('NO_MATCH'),matched('N1')]),
        collect_only=True,max_steps=1,frontier_fn=frontier)
    growth=pr.OnlineGrowth(path,tmp_path/'loaded',state_path=collected['growth_state'])
    action=select_recorded_action(growth,'N0')
    assert action['origin']=='GPT_FRONTIER'
    assert not action['preferred'] and not action['replayable']
    # Nodes without any successful/preferred action can trial recorded evidence.
    growth.state['nodes']['N0']['outgoing_actions']=[action['action_id']]
    action['successful_observations']=0
    assert select_recorded_action(growth,'N0') is action


def test_revised_prompt_keeps_height_choice(tmp_path):
    _,graph=policy(tmp_path);runtime=Runtime();before=runtime.observe(tmp_path/'obs');prompts=[]
    def call(prompt,output,**kwargs):
        prompts.append(prompt);output.mkdir();(output/'answer.txt').write_text(REPAIR)
        return dict(status='AGENT_FINISHED',session_id='revision-test')
    result=free_plan_one(runtime,before,tmp_path/'planner',graph,model_call=call,
        context=dict(reason='REPLAY_REVISED_EDGE',selected_edge=graph['edges'][0]))
    assert result['outcome']['status']=='EXECUTED'
    assert '不强制回到布面 Z' in prompts[0]
    assert '上一动作与最新匹配反馈' not in prompts[0]


def test_recovery_cli_validation_and_dry_run(tmp_path,monkeypatch):
    path,_=policy(tmp_path)
    def forbidden(*a,**k):pytest.fail('dry-run must not connect robot/model')
    monkeypatch.setattr(gp,'make_runtime',forbidden)
    monkeypatch.setattr(pr,'policy_requests',forbidden)
    args=['--graph',str(path),'--recover-no-progress','--dry-run']
    with pytest.raises(SystemExit):pr.main(args)
    with pytest.raises(SystemExit):pr.main(args+['--online-growth','--collect-only'])
    assert pr.main(args+['--online-growth','--output',str(tmp_path/'dry')])==0
