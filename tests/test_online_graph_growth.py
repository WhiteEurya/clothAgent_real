"""Injected robot/model tests: no hardware or network calls."""
import copy
import json
from pathlib import Path
import pytest
from cloth_agent.experiments import graph_policy_replay as pr
from cloth_agent.experiments import graph_policy as gp
from tests.test_graph_policy_replay import policy, candidate, matched
from tests.test_graph_policy import Runtime, match, SOURCE


class Requests:
    def __init__(self, matches, node=None, effect='PROGRESS'):
        self.matches=list(matches); self.calls=[]; self.node=node; self.effect=effect
    def ask(self,kind,payload,images,validator=None):
        self.calls.append((kind,copy.deepcopy(payload),list(images)))
        if kind=='node_match': value=copy.deepcopy(self.matches.pop(0))
        elif kind=='growth_effect':
            value=dict(progress=self.effect,task_progress='visible torso and sleeve arrangement',evidence=['visible changed boundary'])
        elif kind=='recovery_node':
            value=copy.deepcopy(self.node)
            value.update(node_id=payload['node_id'],representative_observations=[payload['observation_id']])
        elif kind=='growth_action':
            value=dict(action_family='test transfer',action_template=dict(grasp_relation='visible cuff',target_relation='torso',
                grasp_z_rule='surface with host offset',yaw_rule='preserve orientation',other_constraints=[]))
        elif kind=='ground_edge':
            value=candidate(); value['edge_id']=payload['selected_edge']['edge_id']
            value['action_family']=payload['selected_edge']['action_family']
        else: raise AssertionError(kind)
        if validator:validator(value)
        return value
    def metrics(self): return dict(model_calls=len(self.calls))


def state(result): return gp.load_frozen(result['growth_state'])


def frontier(runtime,before,folder,graph,**kwargs):
    folder.mkdir()
    return dict(source=SOURCE,outcome=runtime.execute(SOURCE,before,folder))


def test_existing_action_self_loop_then_nominal_keeps_identity_and_repeats(tmp_path):
    p,g=policy(tmp_path); r=Runtime(); req=Requests([matched('N0'),matched('N0'),matched('N1')])
    result=pr.online_replay(p,tmp_path/'run',r,req,max_steps=4)
    assert result['status']=='NO_REPLAYABLE_ACTION'
    s=state(result); ts=s['transitions']
    assert [t['transition_type'] for t in ts]==['NO_PROGRESS','EXPECTED']
    assert [t['selected_action'] for t in ts]==['E0','E0']
    assert s['actions']['E0']['outcomes']['N0']['observed_count']==1
    assert s['actions']['E0']['outcomes']['N1']['observed_count']==1
    assert result['metrics']['multi_outcome_actions']==1
    assert result['metrics']['graph_action_ratio']==1
    assert len(r.executions)==2
    assert all(t['planner_source']=='GRAPH' for t in ts)
    assert gp.load_frozen(p)==g
    assert Path(result['growth_state']).with_suffix('.svg').exists()
    assert 'State → Action → Outcome State' in Path(result['growth_state']).with_suffix('.md').read_text()


def test_exact_local_outcomes_then_global_reentry_no_new_node(tmp_path):
    p,g=policy(tmp_path); growth=pr.OnlineGrowth(p,tmp_path/'state'); growth.requests=Requests([],node=g['nodes'][0])
    r=Runtime(); obs=r.observe(tmp_path/'obs')
    provisional=growth.observe_node(obs)
    growth.requests=Requests([match('NO_MATCH'),matched(provisional)])
    node,audit=growth.match(obs,tmp_path,source='N0',action='E0')
    assert node==provisional and not audit['new_node_created']
    calls=growth.requests.calls
    assert [n['node_id'] for n in calls[0][1]['nodes']]==['N0','N1']
    assert set(n['node_id'] for n in calls[1][1]['nodes'])=={'N0','N1',provisional}
    assert audit['global_reentry']['used'] and audit['global_reentry']['result']=='MATCH'
    for _,payload,_ in calls:
        assert not any(k in json.dumps(payload) for k in ['selected_action','nominal_target','outcomes','edge_id'])
    # An observed alternative is local on the next visit, sharing the same action.
    growth.state['actions']['E0']['outcomes'][provisional]=dict(seed_evidence=False,observed_count=0,history=[])
    growth.requests=Requests([matched(provisional)])
    _,audit=growth.match(obs,tmp_path,source='N0',action='E0')
    assert audit['local_match']['candidates']==['N0','N1',provisional]
    assert not audit['global_reentry']['used']


def test_frontier_success_is_reused_in_next_episode_and_global_reentry_recorded(tmp_path):
    p,g=policy(tmp_path); r=Runtime()
    # Start with a genuinely new node. Frontier returns to N0 via global re-entry.
    req=Requests([match('NO_MATCH'),match('NO_MATCH'),matched('N0')],node=g['nodes'][0])
    result=pr.online_replay(p,tmp_path/'ep1',r,req,max_steps=1,frontier_fn=frontier)
    assert result['status']=='STEP_LIMIT',result
    s=state(result); provisional=next(n for n,row in s['nodes'].items() if row['status']=='PROVISIONAL')
    aid=next(a for a,row in s['actions'].items() if row['origin']=='GPT_FRONTIER')
    assert s['actions'][aid]['replayable'] and s['actions'][aid]['successful_observations']==1
    assert s['transitions'][0]['transition_type']=='GLOBAL_REENTRY'
    assert s['nodes'][provisional]['status']=='PROVISIONAL'
    assert s['nodes'][provisional]['observation_count']==1
    assert s['nodes'][provisional]['outcome_history']==['T0000000']
    def forbidden(*args,**kwargs):pytest.fail('successful frontier action must be grounded, not replanned')
    req2=Requests([matched(provisional),matched('N0')])
    second=pr.online_replay(p,tmp_path/'ep2',Runtime(),req2,max_steps=1,state_path=result['growth_state'],frontier_fn=forbidden)
    assert second['status']=='STEP_LIMIT',second
    s2=state(second)
    assert s2['transitions'][-1]['planner_source']=='GRAPH'
    assert s2['transitions'][-1]['selected_action']==aid
    assert second['metrics']['total_physical_actions']==2
    assert second['metrics']['graph_action_ratio']==.5
    assert len(s2['episodes'])==2


def test_double_no_match_creates_provisional_only_after_global(tmp_path):
    p,g=policy(tmp_path)
    req=Requests([matched('N0'),match('NO_MATCH'),match('NO_MATCH')],node=g['nodes'][1])
    result=pr.online_replay(p,tmp_path/'run',Runtime(),req,max_steps=1)
    s=state(result); t=s['transitions'][0]
    assert t['transition_type']=='NEW_NODE' and t['new_node_created']
    row=s['nodes'][t['result_node']]
    assert row['status']=='PROVISIONAL' and row['created_from']=='N0' and row['created_by_action']=='E0'
    assert row['incoming_transitions']==[t['transition_id']]
    assert row['task_progress'] and row['representative_observation']
    kinds=[k for k,_,_ in req.calls]
    assert kinds[:4]==['node_match','ground_edge','node_match','node_match']
    assert kinds.index('recovery_node')>3
    assert result['metrics']['new_node_rate']==1


@pytest.mark.parametrize('response',[match('AMBIGUOUS'),dict(matched('N0'),confidence=.2)])
def test_ambiguity_never_creates_or_dispatches(tmp_path,response):
    p,g=policy(tmp_path); r=Runtime()
    req=Requests([response,response])
    result=pr.online_replay(p,tmp_path/'run',r,req)
    assert result['status']=='AMBIGUOUS'
    assert not r.executions and result['metrics']['new_provisional_nodes']==0


def test_no_progress_frontier_never_promoted(tmp_path):
    p,g=policy(tmp_path)
    req=Requests([match('NO_MATCH'),matched('P000002')],node=g['nodes'][0],effect='PROGRESS')
    result=pr.online_replay(p,tmp_path/'run',Runtime(),req,max_steps=1,frontier_fn=frontier)
    s=state(result); action=next(a for a in s['actions'].values() if a['origin']=='GPT_FRONTIER')
    assert not action['replayable'] and action['successful_observations']==0
    assert s['transitions'][0]['transition_type']=='NO_PROGRESS'


def test_safety_rejection_does_not_fall_back_or_record_success(tmp_path):
    p,g=policy(tmp_path)
    class Rejected(Runtime):
        def execute(self,*args):return dict(status='PLANNING_FAILED',physical_execution=False)
    def forbidden(*args,**kwargs):pytest.fail('safety rejection cannot trigger free planning')
    req=Requests([matched('N0')])
    result=pr.online_replay(p,tmp_path/'run',Rejected(),req,frontier_fn=forbidden)
    assert result['status']=='SAFETY_REJECT'
    assert result['metrics']['total_physical_actions']==0
    assert result['metrics']['current_transitions']==0


def test_partial_execution_remains_in_physical_denominator(tmp_path):
    p,g=policy(tmp_path); req=Requests([matched('N0')]); r=Runtime('EXECUTION_FAILED')
    result=pr.online_replay(p,tmp_path/'run',r,req)
    assert result['status']=='ACTION_EXECUTION_FAILURE'
    assert result['metrics']['total_physical_actions']==1
    assert result['metrics']['unmatched_physical_attempts']==1
    assert result['metrics']['current_transitions']==0


def test_growth_reload_rejects_policy_mutation(tmp_path):
    p,g=policy(tmp_path); growth=pr.OnlineGrowth(p,tmp_path/'state')
    tampered=copy.deepcopy(growth.state); tampered['actions']['E0']['edge']['action_family']='different strategy'
    bad=tmp_path/'bad.json'; gp.frozen(bad,tampered)
    with pytest.raises(ValueError,match='frozen policy'):
        pr.OnlineGrowth(p,tmp_path/'reload',state_path=bad)


def test_frontier_action_template_is_not_absolute_coordinate_replay(tmp_path):
    p,g=policy(tmp_path); req=Requests([match('NO_MATCH'),match('NO_MATCH'),matched('N0')],node=g['nodes'][0])
    result=pr.online_replay(p,tmp_path/'run',Runtime(),req,max_steps=1,frontier_fn=frontier)
    s=state(result); a=next(a for a in s['actions'].values() if a['origin']=='GPT_FRONTIER')
    assert a['edge']['seed_action']['exact_command']==gp.command(SOURCE)
    assert a['edge']['action_template']['grasp_relation']=='visible cuff'
    assert len(a['outcomes'])==1 and a['nominal_target']=='N0'


def test_ambiguous_provisional_equivalence_is_audit_only(tmp_path):
    p,g=policy(tmp_path); growth=pr.OnlineGrowth(p,tmp_path/'state')
    growth.requests=Requests([],node=g['nodes'][0]); obs=Runtime().observe(tmp_path/'obs')
    new=growth.observe_node(obs)
    ambiguous=match('AMBIGUOUS'); ambiguous['candidate_nodes']=['N0',new]
    growth.requests=Requests([ambiguous,ambiguous])
    target,audit=growth.match(obs,tmp_path)
    assert target is None
    assert growth.state['merge_candidates'][0]['status']=='MERGE_CANDIDATE'
    assert growth.state['merge_candidates'][0]['permanent_merge'] is False
    assert set(growth.state['nodes'])=={'N0','N1',new}
    assert growth.state['nodes'][new]['status']=='PROVISIONAL'


def test_progress_unknown_does_not_promote_but_preserves_transition(tmp_path):
    p,g=policy(tmp_path)
    req=Requests([match('NO_MATCH'),match('NO_MATCH'),matched('N0')],node=g['nodes'][0],effect='UNKNOWN')
    result=pr.online_replay(p,tmp_path/'run',Runtime(),req,max_steps=1,frontier_fn=frontier)
    s=state(result)
    assert len(s['transitions'])==1
    assert not next(a for a in s['actions'].values() if a['origin']=='GPT_FRONTIER')['replayable']


def test_template_failure_keeps_observed_outcome_and_action_ineligible(tmp_path):
    p,g=policy(tmp_path)
    class Broken(Requests):
        def ask(self,kind,*args,**kwargs):
            if kind=='growth_action':raise RuntimeError('model unavailable')
            return super().ask(kind,*args,**kwargs)
    req=Broken([match('NO_MATCH'),match('NO_MATCH'),matched('N0')],node=g['nodes'][0])
    result=pr.online_replay(p,tmp_path/'run',Runtime(),req,max_steps=1,frontier_fn=frontier)
    assert len(state(result)['transitions'])==1
    assert not next(a for a in state(result)['actions'].values() if a['origin']=='GPT_FRONTIER')['replayable']


def test_interrupted_frontier_postprocessing_does_not_lose_physical_attempt(tmp_path):
    p,g=policy(tmp_path)
    def interrupted(runtime,before,folder,graph,**kwargs):
        folder.mkdir(); runtime.execute(SOURCE,before,folder)
        raise KeyboardInterrupt()
    result=pr.online_replay(p,tmp_path/'run',Runtime(),Requests([match('NO_MATCH')],node=g['nodes'][0]),frontier_fn=interrupted)
    assert result['status']=='INTERRUPTED'
    assert result['metrics']['total_physical_actions']==1
    assert result['metrics']['gpt_frontier_actions']==1
    assert state(result)['attempts'][0]['command_source']==SOURCE


def test_resume_latest_pointer_and_ineligible_candidate_not_replayed(tmp_path):
    p,g=policy(tmp_path); growth=pr.OnlineGrowth(p,tmp_path/'state')
    restored=pr.OnlineGrowth(p,tmp_path/'restored',state_path=tmp_path/'state/latest.json')
    assert restored.metrics()==growth.metrics()
    pointer=json.loads((tmp_path/'state/latest.json').read_text()); pointer['sha256']='bad'
    (tmp_path/'state/latest.json').write_text(json.dumps(pointer))
    with pytest.raises(ValueError,match='pointer hash'):
        pr.OnlineGrowth(p,tmp_path/'bad',state_path=tmp_path/'state/latest.json')


def test_multiple_episodes_carry_graph_with_manual_reset(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from cloth_agent.experiments import discovery_model
    p,g=policy(tmp_path); robots=[]; prompts=[]
    def make(*args):
        r=Runtime(); robots.append(r);return r
    monkeypatch.setattr(gp,'make_runtime',make)
    monkeypatch.setattr(discovery_model,'CodexMultimodal',lambda *args:None)
    monkeypatch.setattr(pr,'policy_requests',lambda *args,**kwargs:Requests([matched('N0'),matched('N1')]))
    monkeypatch.setattr('builtins.input',lambda text:prompts.append(text) or '')
    args=SimpleNamespace(graph=p,growth_state=None,episodes=2,model='gpt-6-astra',timeout_s=900,
                         no_cache=False,max_calls=32,config=None,viser_port=8088,max_steps=1,match_threshold=.8)
    result=pr.run_growth_episodes(args,tmp_path/'episodes')
    assert result['episodes']==2 and len(prompts)==1 and all(r.closed for r in robots)
    assert state(result)['episodes'][-1]['metrics']['total_physical_actions']==2
    assert len(state(result)['transitions'])==2


def test_alternative_known_outcome_is_local_not_new_action(tmp_path):
    p,g=policy(tmp_path); growth=pr.OnlineGrowth(p,tmp_path/'state')
    growth.requests=Requests([],node=g['nodes'][1]); r=Runtime(); before=r.observe(tmp_path/'before')
    other=growth.observe_node(before)
    a=growth.state['actions']['E0']; a['outcomes'][other]=dict(seed_evidence=False,observed_count=0,history=[])
    growth.requests=Requests([matched(other)])
    target,audit=growth.match(before,tmp_path,source='N0',action='E0')
    attempt=dict(attempt_id='test',source_node='N0',selected_action='E0',planner_source='GRAPH',
        before=gp.evidence_file(before.image),after=gp.evidence_file(before.image),
        execution=dict(status='EXECUTED',physical_execution=True))
    growth.state['attempts'].append(attempt)
    transition=growth.transition(attempt,target,audit,dict(progress='PROGRESS',task_progress='visible',evidence=['changed']))
    assert transition['transition_type']=='ALTERNATIVE_OUTCOME'
    assert transition['selected_action']=='E0' and len(growth.state['actions'])==1


def test_out_of_local_set_is_rejected_not_silently_accepted(tmp_path):
    p,g=policy(tmp_path); growth=pr.OnlineGrowth(p,tmp_path/'state')
    growth.requests=Requests([],node=g['nodes'][0]); obs=Runtime().observe(tmp_path/'obs')
    other=growth.observe_node(obs); growth.requests=Requests([matched(other)])
    with pytest.raises(ValueError,match='unknown'):
        growth.match(obs,tmp_path,source='N0',action='E0')


def test_cli_growth_dry_run_never_creates_robot_or_model(tmp_path,monkeypatch):
    p,g=policy(tmp_path)
    def forbidden(*args,**kwargs):pytest.fail('dry run must be offline')
    monkeypatch.setattr(gp,'make_runtime',forbidden)
    monkeypatch.setattr(pr,'policy_requests',forbidden)
    assert pr.main(['--graph',str(p),'--online-growth','--dry-run','--output',str(tmp_path/'dry')])==0
    pointer=json.loads((tmp_path/'dry/latest.json').read_text())
    assert gp.load_frozen(pointer['snapshot'])['protocol']==pr.GROWTH_PROTOCOL


def test_from_scratch_cli_has_zero_prior_nodes_actions_and_no_hardware(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('dry run must not use hardware or model')
    monkeypatch.setattr(gp,'make_runtime',forbidden)
    monkeypatch.setattr(pr,'policy_requests',forbidden)
    output=tmp_path/'empty'
    assert pr.main(['--from-scratch','--online-growth','--dry-run','--output',str(output)])==0
    pointer=json.loads((output/'latest.json').read_text())
    s=gp.load_frozen(pointer['snapshot'])
    assert s['nodes']=={} and s['actions']=={} and s['transitions']==[]
    base=gp.load_frozen(output/'empty_policy.json')
    assert base['protocol']==pr.EMPTY_PROTOCOL
    assert 'seed_trajectory_path' not in base
    assert pr.validate_growth_base(base)==dict(observations={},steps=[])
    restored=pr.OnlineGrowth(output/'empty_policy.json',tmp_path/'resume',state_path=output/'latest.json')
    assert restored.metrics()['total_physical_actions']==0


def test_from_scratch_first_action_frontier_then_learned_action_reused(tmp_path,monkeypatch):
    monkeypatch.setattr(Runtime,'grasp_offset_mm',0.)
    _,fixture=policy(tmp_path)
    base=pr.create_empty_base(tmp_path/'empty.json')
    req=Requests([match('NO_MATCH'),match('NO_MATCH')],node=fixture['nodes'][0])
    r=Runtime()
    result=pr.online_replay(base,tmp_path/'first',r,req,max_steps=1,frontier_fn=frontier)
    assert result['status']=='STEP_LIMIT',result
    s=state(result)
    assert set(s['nodes'])=={'P000000','P000001'}
    assert s['attempts'][0]['planner_source']=='GPT_FRONTIER'
    assert s['actions']['F0000000']['replayable']
    assert req.calls[0][0]=='recovery_node'  # No matching against invented priors.
    req2=Requests([matched('P000000'),matched('P000001')])
    result2=pr.online_replay(base,tmp_path/'second',Runtime(),req2,max_steps=1,
                            state_path=result['growth_state'],frontier_fn=lambda *a,**k:pytest.fail('must reuse'))
    assert result2['status']=='STEP_LIMIT',result2
    assert state(result2)['attempts'][-1]['planner_source']=='GRAPH'
    assert result2['metrics']['graph_policy_actions']==1
    assert result2['metrics']['gpt_frontier_actions']==1


@pytest.mark.parametrize('args',[
    ['--from-scratch'],
    ['--from-scratch','--online-growth','--growth-state','old.json'],
    ['--from-scratch','--online-growth','--graph','old.json'],
])
def test_from_scratch_rejects_conflicting_initialization(args):
    with pytest.raises(SystemExit):pr.main(args)


def test_empty_base_cannot_smuggle_seed_nodes(tmp_path):
    base=gp.load_frozen(pr.create_empty_base(tmp_path/'empty.json'))
    base['nodes']=[{'node_id':'S0'}]
    with pytest.raises(ValueError,match='invalid empty'):
        pr.validate_growth_base(base)


def test_from_scratch_episode_runner_creates_independent_base(tmp_path,monkeypatch):
    monkeypatch.setattr(Runtime,'grasp_offset_mm',0.)
    from types import SimpleNamespace
    from cloth_agent.experiments import discovery_model
    _,fixture=policy(tmp_path)
    monkeypatch.setattr(gp,'make_runtime',lambda *a:Runtime())
    monkeypatch.setattr(discovery_model,'CodexMultimodal',lambda *a:None)
    monkeypatch.setattr(pr,'free_plan_one',frontier)
    monkeypatch.setattr(pr,'policy_requests',lambda *a,**k:Requests(
        [match('NO_MATCH'),match('NO_MATCH')],node=fixture['nodes'][0]))
    args=SimpleNamespace(from_scratch=True,graph=None,growth_state=None,episodes=1,
        model='gpt-6-astra',timeout_s=900,no_cache=False,max_calls=32,config=None,
        viser_port=8088,max_steps=1,match_threshold=.8)
    output=tmp_path/'fresh'
    result=pr.run_growth_episodes(args,output)
    assert result['status']=='STEP_LIMIT',result
    assert gp.load_frozen(output/'empty_policy.json')['nodes']==[]
    restored=pr.OnlineGrowth(output/'empty_policy.json',tmp_path/'restore',state_path=output/'latest.json')
    assert set(restored.state['nodes'])=={'P000000','P000001'}
    assert restored.metrics()['gpt_frontier_actions']==1
