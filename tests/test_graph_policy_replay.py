"""No network or robot: contract tests with fresh observations and injected model outputs."""
import copy
import json
from pathlib import Path

import pytest

from cloth_agent.experiments import graph_policy as gp
from cloth_agent.experiments import graph_policy_replay as pr
from tests.test_graph_policy import fixture_graph, write_graph, Runtime, candidate as legacy_candidate, match


def candidate():
    value = legacy_candidate()
    del value['unresolved_conditions']
    value.update(visual_violations=[], execution_risk=[], missing_visual_parameters=[],
                 garment_frame=dict(collar_pixel_xy=[10,2], hem_center_pixel_xy=[10,18],
                                    visual_evidence='Visible collar and hem'))
    return value


def policy(tmp_path):
    g, t = fixture_graph(tmp_path)
    g['node_mapping'] = {'old0': 'N0', 'old1': 'N1'}
    source = write_graph(tmp_path, g, t)
    graph = gp.load_frozen(source)
    graph.update(protocol=pr.PROTOCOL, match_nodes=copy.deepcopy(g['nodes']),
        matcher_projection_omissions=[], policy_provenance=dict(authority='USER',
            source_graph=str(source), source_sha256=gp.file_hash(source)))
    graph['edges'][0].update(replayable=True, preference='PREFERRED', policy_outcome='SUCCESS',
        evidence_status='ONE_SHOT_HYPOTHESIS', stop_reason=None)
    path = tmp_path / 'policy.json'; gp.frozen(path, graph)
    return path, graph


def matched(node):
    value = match(); value.update(node_id=node, candidate_nodes=[node])
    return value


class Model:
    def __init__(self, matches=None, grounding=None):
        self.matches = list(matches or [matched('N0'), matched('N1')])
        self.grounding = grounding or candidate()
        self.calls = []
    def ask(self, kind, payload, images, validator=None):
        self.calls.append((kind, copy.deepcopy(payload), images))
        if kind == 'node_match':
            value = self.matches[0] if self.matches else self.last_match
            ids={n['node_id'] for n in payload['nodes']}
            if value['status']=='MATCH' and value['node_id'] not in ids:
                return match('NO_MATCH')
            if self.matches:self.last_match=self.matches.pop(0)
        else:value=self.grounding
        if validator: validator(value)
        return copy.deepcopy(value)
    def metrics(self):
        return dict(model_calls=len(self.calls), tool_calls=0, reasoning_time_s=0.)


def test_fresh_blind_match_after_dispatch_and_host_selects(tmp_path):
    path, g = policy(tmp_path); runtime = Runtime(); model = Model()
    result = pr.replay(path, tmp_path/'run', runtime, model)
    assert result['status'] == 'NO_REPLAYABLE_EDGE'
    assert result['transitions'][0]['reached_expected_target_node']
    assert [x[0] for x in model.calls] == ['node_match','ground_edge','node_match']
    assert model.calls[0][1] == model.calls[2][1]  # no expected target or history leakage
    assert model.calls[0][2][0][1] != model.calls[2][2][0][1]  # genuinely fresh RGB
    forbidden = ['seed_action', 'action_family', 'expected_target', 'replayable', 'preference', 'edges']
    assert not any(s in json.dumps(model.calls[0][1]) for s in forbidden)
    grounded = model.calls[1][1]
    assert grounded['selected_edge']['edge_id'] == 'E0'
    assert 'exact_command' not in json.dumps(grounded)
    assert len(runtime.executions) == 1 and not result['fallback_used']
    assert gp.load_frozen(path) == g


@pytest.mark.parametrize('node,status', [('N0','SOURCE_NODE_UNCHANGED'), ('N2','UNEXPECTED_NODE')])
def test_source_or_unexpected_node_stops(tmp_path, node, status):
    path,g = policy(tmp_path)
    if node == 'N2':
        # Add a third observed node to the test's source and policy, not during replay.
        sourcepath = Path(g['policy_provenance']['source_graph'])
        source = gp.load_frozen(sourcepath)
        n = copy.deepcopy(source['nodes'][1]); n['node_id']='N2'
        source['nodes'].append(n)
        sourcepath.unlink(); Path(str(sourcepath)+'.sha256.json').unlink();gp.frozen(sourcepath,source)
        g['nodes'].append(n);g['match_nodes'].append(copy.deepcopy(n))
        g['policy_provenance']['source_sha256']=gp.file_hash(sourcepath)
        path.unlink();Path(str(path)+'.sha256.json').unlink();gp.frozen(path,g)
    r=Runtime(); result=pr.replay(path,tmp_path/'run',r,Model([matched('N0'),matched(node)]))
    assert result['status']==status and len(r.executions)==1
    assert not result['graph_only_success']


@pytest.mark.parametrize('status',['NO_MATCH','AMBIGUOUS'])
def test_nonmatch_no_grounding_no_dispatch(tmp_path,status):
    p,_=policy(tmp_path);r=Runtime();m=Model([match(status)])
    out=pr.replay(p,tmp_path/'run',r,m)
    assert out['status']==status and not r.executions
    assert len(m.calls)==1  # Representatives are attached on the first call.


@pytest.mark.parametrize('confidence', [.95, .3])
def test_recovery_reenters_alternative_known_node_using_its_own_edge(tmp_path, confidence):
    path, graph = policy(tmp_path)
    source_path = Path(graph['policy_provenance']['source_graph'])
    source = gp.load_frozen(source_path)
    node = copy.deepcopy(source['nodes'][1]); node['node_id'] = 'N2'
    edge = copy.deepcopy(source['edges'][0]); edge.update(edge_id='E2', source_node='N2')
    source['nodes'].append(node); source['edges'].append(edge)
    source_path.unlink(); Path(str(source_path)+'.sha256.json').unlink()
    gp.frozen(source_path, source)
    graph['nodes'].append(node); graph['match_nodes'].append(copy.deepcopy(node))
    replay_edge = copy.deepcopy(graph['edges'][0]); replay_edge.update(edge_id='E2', source_node='N2')
    graph['edges'].append(replay_edge)
    graph['policy_provenance']['source_sha256'] = gp.file_hash(source_path)
    path.unlink(); Path(str(path)+'.sha256.json').unlink(); gp.frozen(path, graph)

    class SelectedEdgeModel(Model):
        def ask(self, kind, payload, images, validator=None):
            if kind == 'ground_edge':
                self.grounding = candidate()
                self.grounding['edge_id'] = payload['selected_edge']['edge_id']
            return super().ask(kind, payload, images, validator)

    alternative = matched('N2'); alternative['confidence'] = confidence
    model = SelectedEdgeModel([matched('N0'), alternative, matched('N1')])
    runtime = Runtime()
    def forbidden(*args, **kwargs):
        raise AssertionError('Reliable known-node reentry does not require free planning')
    result = pr.replay(path, tmp_path/'run', runtime, model, max_fallbacks=1, fallback_fn=forbidden)
    assert gp.load_frozen(path) == graph
    assert result['fallback_steps'] == 0
    if confidence < .8:
        assert result['status'] == 'AMBIGUOUS'
        assert len(runtime.executions) == 1
        assert not result.get('graph_reentries')
    else:
        assert result['status'] == 'NO_REPLAYABLE_EDGE'
        assert result['executed_edges'] == ['E0', 'E2']
        assert result['graph_reentries'][0]['node_id'] == 'N2'
        assert result['transitions'][0]['actual_node'] == 'N2'
        assert not result['transitions'][0]['reached_expected_target_node']
        extended = gp.load_frozen(result['extended_graph']); pr.validate_policy(extended)
        branch = extended['recovery_extension']['transitions'][0]
        assert branch['source_node'] == 'N0' and branch['target_node'] == 'N2'
        assert branch['reused_edge_id'] == 'E0' and not branch['replayable']


def test_low_confidence_match_is_ambiguous(tmp_path):
    p,_=policy(tmp_path);r=Runtime();v=matched('N0');v['confidence']=.3
    out=pr.replay(p,tmp_path/'run',r,Model([v]))
    assert out['status']=='AMBIGUOUS' and not r.executions


@pytest.mark.parametrize('change',['family','edge','slots','height','blocked','uncertainty'])
def test_grounding_cannot_change_edge_or_skeleton(tmp_path,change):
    p,_=policy(tmp_path);r=Runtime();v=candidate()
    if change=='family':v['action_family']='new strategy'
    if change=='edge':v['edge_id']='invented'
    if change=='slots':v['slots'].reverse()
    if change=='height':v['slots'][0]['z_above_surface_mm']=5
    if change=='blocked':v['status']='BLOCKED'
    if change=='uncertainty':v.update(status='NEED_MORE_INFORMATION',missing_visual_parameters=['Cuff anchor occluded'])
    out=pr.replay(p,tmp_path/'run',r,Model(grounding=v))
    assert out['status']=='EDGE_INSTANTIATION_FAILURE' and not r.executions


@pytest.mark.parametrize('status',['PLANNING_FAILED','PLANNING_INCOMPLETE'])
def test_existing_safety_rejection_never_replans(tmp_path,status):
    p,_=policy(tmp_path);r=Runtime(status);m=Model()
    out=pr.replay(p,tmp_path/'run',r,m)
    assert out['status']=='SAFETY_REJECT' and out['executed_edges']==[]
    assert len(m.calls)==2 and not out['fallback_used']


def test_after_last_allowed_execution_still_matches(tmp_path):
    p,_=policy(tmp_path);m=Model();out=pr.replay(p,tmp_path/'run',Runtime(),m,max_steps=1)
    assert len(m.calls)==3 and out['transitions'][0]['actual_node']=='N1'


def test_multiple_preferred_rejected(tmp_path):
    _,g=policy(tmp_path)
    g['edges'].append(copy.deepcopy(g['edges'][0]))
    assert pr.choose(g,'N0')==(None,'NO_REPLAYABLE_EDGE')
    with pytest.raises(ValueError):pr.validate_policy(g)


def test_validation_barrier_never_instantiates(tmp_path):
    p,g=policy(tmp_path);g['edges'][0].update(replayable=False,preference='EXCLUDED',
        evidence_status='UNVERIFIED_OUTCOME',stop_reason='REQUIRES_FRESH_VALIDATION')
    p.unlink();Path(str(p)+'.sha256.json').unlink();gp.frozen(p,g)
    r=Runtime();m=Model();out=pr.replay(p,tmp_path/'run',r,m)
    assert out['status']=='REQUIRES_FRESH_VALIDATION' and len(m.calls)==1 and not r.executions


def test_dry_run_default_never_constructs_model_or_runtime(tmp_path,monkeypatch):
    p,_=policy(tmp_path)
    def forbidden(*a,**k):raise AssertionError('hardware/model forbidden')
    monkeypatch.setattr(gp,'make_runtime',forbidden)
    monkeypatch.setattr(pr,'policy_requests',forbidden)
    assert pr.main(['--graph',str(p),'--output',str(tmp_path/'dry')])==0
    result=gp.load_frozen(tmp_path/'dry/dry_run.json')
    assert result['model_calls']==result['robot_calls']==0
    assert result['mode']=='OFFLINE_STATIC_PREFLIGHT'
    with pytest.raises(SystemExit):pr.main(['--graph',str(p),'--execute','--dry-run'])


def test_requests_real_multimodal_arguments_and_clean_prompt(tmp_path):
    p,g=policy(tmp_path);t=pr.validate_policy(g)
    class Backend:
        model='fake-gpt'
        def invoke(self,prompt,images,schema):
            self.prompt=prompt;self.images=images
            return {'text':json.dumps(matched('N0'))}
    backend=Backend();request=pr.policy_requests(tmp_path/'request',backend)
    images=pr.matcher_images(g,t)
    request.ask('node_match',pr.matcher_payload(g),images)
    assert backend.images==[p for _,p in images]
    assert 'parallel-jaw' not in backend.prompt and 'seed_action' not in backend.prompt
    assert 'action_family' not in backend.prompt and 'graph_id' not in backend.prompt


def test_policy_mutation_before_dispatch_stops(tmp_path):
    p,_=policy(tmp_path);r=Runtime()
    class Mutator(Model):
        def ask(self,*a,**k):
            value=super().ask(*a,**k)
            if a[0]=='ground_edge':p.write_text(p.read_text()+' ')
            return value
    with pytest.raises(ValueError,match='changed before dispatch'):
        pr.replay(p,tmp_path/'run',r,Mutator())
    assert not r.executions and r.stopped


def test_matching_graph_path_never_calls_recovery(tmp_path):
    p,_=policy(tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('matching graph must not free-plan')
    model=Model()
    result=pr.replay(p,tmp_path/'run',Runtime(),model,max_fallbacks=3,fallback_fn=forbidden)
    assert not result['fallback_used']
    assert [x[0] for x in model.calls]==['node_match','ground_edge','node_match']


def test_consolidation_preserves_commands_and_explicit_exclusions(tmp_path):
    g,t=fixture_graph(tmp_path)
    artifact=t['observations']['o0']['rgb']
    t['observations']={f'o{i:03d}':{'rgb':artifact} for i in range(14)}
    groups=[[0],[1],[2],[3],[4],[5],[6,7,8],[9,10],[11],[12],[13],[]]
    template_node=copy.deepcopy(g['nodes'][0]);template_edge=copy.deepcopy(g['edges'][0])
    g['nodes']=[];g['edges']=[];t['steps']=[]
    for i,ids in enumerate(groups):
        n=copy.deepcopy(template_node);n.update(node_id=f'S{i}',representative_observations=[f'o{j:03d}' for j in ids])
        if not ids:n.update(confidence=0.,observation_condition=dict(positive_evidence=[],negative_evidence=[],uncertainty_conditions=['no RGB']))
        g['nodes'].append(n)
    owner={v:i for i,ids in enumerate(groups) for v in ids}
    for i in range(14):
        e=copy.deepcopy(template_edge)
        before=f'o{i:03d}';after=f'o{i+1:03d}' if i<13 else None
        e.update(edge_id=f'e{i:03d}',seed_step_id=f's{i:03d}',source_node=f'S{owner[i]}',
            target_node=f'S{owner[i+1]}' if i<13 else 'S11',outcome_evidence=[before])
        e['seed_action']['source_observation']=before
        if i in {1,6,7,9}:e.update(seed_outcome='FAILURE',generalization_status='OBSERVED_FAILURE')
        if i in {12,13}:e.update(seed_outcome='UNKNOWN')
        g['edges'].append(e)
        t['steps'].append(dict(step_id=e['seed_step_id'],before_observation=before,after_observation=after,
            commanded_action=copy.deepcopy(e['seed_action']['exact_command']),
            actual_execution=dict(physical_execution=True,status='EXECUTED' if i<13 else 'UNAVAILABLE')))
    g['node_mapping']={f'n{i:03d}':f'S{owner[i]}' for i in range(14)}
    seed=tmp_path/'seed_trajectory.json';seed.unlink();Path(str(seed)+'.sha256.json').unlink();gp.frozen(seed,t)
    src=write_graph(tmp_path,g,t);old_hash=gp.file_hash(src)
    policy=pr.consolidate(src,tmp_path/'policy.json')
    assert gp.file_hash(src)==old_hash
    assert pr.summary(policy)==dict(nodes=12,matchable_nodes=11,replayable_edges=10,failed_edges=3,validation_blocked_edges=1)
    edges={e['edge_id']:e for e in policy['edges']}
    assert edges['e001']['seed_outcome']=='PROGRESS' and edges['e001']['replayable']
    assert pr.choose(policy,'S6')[0]['edge_id']=='e008'
    assert pr.choose(policy,'S7')[0]['edge_id']=='e010'
    assert pr.choose(policy,'S9')[0]['edge_id']=='e012'
    assert policy['terminal_node_ids']==['S10']
    assert pr.choose(policy,'S10')==(None,'COMPLETED')
    assert edges['e012']['seed_outcome']=='UNKNOWN'
    for eid in ['e006','e007','e009','e013']:assert not edges[eid]['replayable']
    for original in g['edges']:assert edges[original['edge_id']]['seed_action']==original['seed_action']
    for eid in ['e006','e007','e009','e013']:
        bad=copy.deepcopy(policy)
        next(e for e in bad['edges'] if e['edge_id']==eid)['replayable']=True
        with pytest.raises(ValueError):pr.validate_policy(bad)


def test_after_execution_no_match_never_assumes_target(tmp_path):
    p,_=policy(tmp_path);r=Runtime()
    out=pr.replay(p,tmp_path/'run',r,Model([matched('N0'),match('NO_MATCH')]))
    assert out['status']=='NO_MATCH' and len(r.executions)==1
    assert not out['transitions'][0]['reached_expected_target_node']


def test_execution_failure_has_no_visual_success_claim(tmp_path):
    p,_=policy(tmp_path);r=Runtime('EXECUTION_FAILED');m=Model()
    out=pr.replay(p,tmp_path/'run',r,m)
    assert out['status']=='ACTION_EXECUTION_FAILURE' and len(m.calls)==2
    assert not out['graph_only_success'] and out['executed_edges']==[]


def test_grounding_block_preserves_reasons_without_dispatch(tmp_path):
    p,_=policy(tmp_path);r=Runtime();v=candidate()
    v.update(status='BLOCKED',explanation='Receiving region is visibly obstructed.',
             visual_violations=[dict(constraint='Exposed receiving torso',
                                    current_visual_evidence='Opposite sleeve visibly covers the receiving region')])
    m=Model(grounding=v)
    out=pr.replay(p,tmp_path/'run',r,m)
    assert out['status']=='EDGE_INSTANTIATION_FAILURE'
    detail=out['instantiation_failure']
    assert detail['edge_id']=='E0' and detail['model_status']=='BLOCKED'
    assert detail['visual_violations']==v['visual_violations']
    assert detail['explanation']==v['explanation']
    assert not detail['safety_validation_attempted'] and not detail['dispatch_attempted']
    assert 'Opposite sleeve visibly covers' in out['error']
    assert not r.executions and len(m.calls)==2


@pytest.mark.parametrize('host_status', [None, 'PLANNING_FAILED', 'PLANNING_INCOMPLETE'])
def test_unknown_cloth_risk_reaches_host_but_never_overrides_rejection(tmp_path, host_status):
    p,g=policy(tmp_path);r=Runtime(host_status);v=candidate()
    v['execution_risk']=['Hidden layer relations unknown', 'Torso may be drawn by cloth coupling']
    m=Model(grounding=v)
    out=pr.replay(p,tmp_path/'run',r,m)
    assert len(r.executions)==1  # candidate submitted to existing runtime validator
    intent=gp.read(tmp_path/'run/step_000/dispatch_intent.json')
    assert intent['execution_risk']==v['execution_risk']
    assert gp.load_frozen(p)==g
    assert m.calls[1][1]['constraint_ownership']['host']==pr.OWNERSHIP['host']
    if host_status:
        assert out['status']=='SAFETY_REJECT' and out['executed_edges']==[]
        assert len(m.calls)==2
    else:
        assert out['transitions'][0]['execution_risk']==v['execution_risk']
        assert len(m.calls)==3


def test_unknown_risk_cannot_be_reported_as_evidence_free_blocked():
    v=candidate();v.update(status='BLOCKED',execution_risk=['Hidden layers unknown'])
    with pytest.raises(ValueError,match='positive visual violation'):
        pr.validate_grounding(v)


def test_ready_with_visible_violation_rejected(tmp_path):
    p,_=policy(tmp_path);r=Runtime();v=candidate()
    v['visual_violations']=[dict(constraint='Exposed cuff',current_visual_evidence='Cuff covered by torso')]
    out=pr.replay(p,tmp_path/'run',r,Model(grounding=v))
    assert out['status']=='EDGE_INSTANTIATION_FAILURE' and not r.executions


def test_grounding_contract_is_isolated_from_original_experiment():
    assert 'execution_risk' in pr.GROUNDING['required']
    assert 'unresolved_conditions' not in pr.GROUNDING['properties']
    assert 'unresolved_conditions' in gp.SCHEMAS['instantiate']['required']
    assert 'execution_risk' not in gp.SCHEMAS['instantiate']['properties']


@pytest.mark.parametrize('collar,hem,left_x,right_x', [([10,2],[10,18],5,15),([10,18],[10,2],15,5)])
def test_sleeve_side_uses_garment_rotation_not_raw_image(tmp_path,collar,hem,left_x,right_x):
    from tests.test_graph_policy import observation
    obs=observation(tmp_path/'obs')
    v=candidate();v['garment_frame'].update(collar_pixel_xy=collar,hem_center_pixel_xy=hem)
    for side,x in [('left',left_x),('right',right_x)]:
        e=dict(action_family=side+'_sleeve_inward_fold')
        v['slots'][0]['pixel_xy']=[x,10]
        assert pr.validate_sleeve_side(v,e,obs)['side']==side
        v['slots'][0]['pixel_xy']=[20-x,10]
        with pytest.raises(ValueError,match='opposite/ambiguous'):
            pr.validate_sleeve_side(v,e,obs)


def test_grounding_receives_graph_side_convention(tmp_path):
    from tests.test_graph_policy import observation
    _,g=policy(tmp_path);g['goal']['side_convention']='Rotate collar above hem without mirroring; viewer sides'
    payload=pr.grounding_payload(g,g['edges'][0],observation(tmp_path/'obs'))
    assert payload['task']['side_convention']==g['goal']['side_convention']


class RecoveryModel(Model):
    def ask(self,kind,payload,images,validator=None):
        if kind!='recovery_node':return super().ask(kind,payload,images,validator)
        self.calls.append((kind,copy.deepcopy(payload),images))
        n=dict(node_id=payload['node_id'],task_context='New observed configuration',
               observation_condition=dict(positive_evidence=['Visible unusual flap'],negative_evidence=[],uncertainty_conditions=[]),
               representative_observations=[payload['observation_id']],confidence=.8)
        if validator:validator(n)
        return n


def fake_fallback(runtime,before,folder,graph,**kwargs):
    from tests.test_graph_policy import SOURCE
    Path(folder).mkdir()
    outcome=runtime.execute(SOURCE,before,folder)
    return dict(result={'status':'ROUND_LIMIT'},outcome=outcome,source=SOURCE,metrics={})


def test_no_match_recovers_then_resumes_graph_and_saves_new_node(tmp_path):
    p,g=policy(tmp_path);r=Runtime()
    m=RecoveryModel([match('NO_MATCH'),matched('N0'),matched('N1')])
    out=pr.replay(p,tmp_path/'run',r,m,max_fallbacks=3,fallback_fn=fake_fallback)
    assert out['status']=='NO_REPLAYABLE_EDGE' and out['fallback_used']
    assert out['fallback_steps']==1 and len(r.executions)==2
    assert out['executed_edges']==['E0'] and not out['graph_only_success']
    assert out['fallback_transitions'][0]['actual_node']=='N0'
    assert out['graph_resumptions'][0]['node_id']=='N0'
    assert [x[0] for x in m.calls]==['node_match','recovery_node','node_match','ground_edge','node_match']
    extended=gp.load_frozen(out['extended_graph']);pr.validate_policy(extended)
    assert len(extended['nodes'])==len(g['nodes'])+1
    assert extended['edges']==g['edges'] and gp.load_frozen(p)==g
    assert extended['recovery_extension']['transitions'][0]['mode']=='FALLBACK_FREE_PLAN'
    assert not extended['recovery_extension']['transitions'][0]['replayable']
    assert 'edges' not in m.calls[2][1]


def test_off_graph_after_edge_records_actual_branch_and_recovers(tmp_path):
    p,_=policy(tmp_path);r=Runtime()
    m=RecoveryModel([matched('N0'),match('NO_MATCH'),matched('N1')])
    out=pr.replay(p,tmp_path/'run',r,m,max_fallbacks=1,fallback_fn=fake_fallback)
    assert out['fallback_steps']==1 and len(r.executions)==2
    assert not out['transitions'][0]['reached_expected_target_node']
    assert out['fallback_transitions'][0]['actual_node']=='N1'
    ext=gp.load_frozen(out['extended_graph']);pr.validate_policy(ext)
    transitions=ext['recovery_extension']['transitions']
    assert [e['mode'] for e in transitions]==['GRAPH_REUSE','FALLBACK_FREE_PLAN']


def test_recovery_is_bounded_and_ambiguity_does_not_fallback(tmp_path):
    p,_=policy(tmp_path);r=Runtime()
    m=RecoveryModel([match('NO_MATCH'),match('NO_MATCH')])
    out=pr.replay(p,tmp_path/'run',r,m,max_fallbacks=1,fallback_fn=fake_fallback)
    assert out['status']=='FALLBACK_LIMIT' and len(r.executions)==1
    pr.validate_policy(gp.load_frozen(out['extended_graph']))
    r2=Runtime()
    out2=pr.replay(p,tmp_path/'ambiguous',r2,RecoveryModel([match('AMBIGUOUS')]),
                   max_fallbacks=3,fallback_fn=fake_fallback)
    assert out2['status']=='AMBIGUOUS' and not r2.executions and not out2['fallback_used']


def test_recovery_preserves_safety_rejection(tmp_path):
    p,_=policy(tmp_path);r=Runtime('PLANNING_FAILED')
    m=RecoveryModel([match('NO_MATCH')])
    out=pr.replay(p,tmp_path/'run',r,m,max_fallbacks=3,fallback_fn=fake_fallback)
    assert out['status']=='SAFETY_REJECT' and out['fallback_steps']==1
    assert out['executed_edges']==[] and len(m.calls)==2


def test_extended_graph_can_be_loaded_and_original_edges_cannot_change(tmp_path):
    p,_=policy(tmp_path);r=Runtime()
    out=pr.replay(p,tmp_path/'run',r,RecoveryModel([match('NO_MATCH'),matched('N1')]),
                  max_fallbacks=1,fallback_fn=fake_fallback)
    extended=gp.load_frozen(out['extended_graph'])
    pr.dry_run(out['extended_graph'],tmp_path/'extended_dry')
    bad=copy.deepcopy(extended);bad['edges'][0]['replayable']=False
    with pytest.raises(ValueError,match='altered original policy'):pr.validate_policy(bad)


def test_actual_arm7_single_round_adapter_reuses_observation_and_validator(tmp_path):
    from tests.test_graph_policy import SOURCE, observation
    from cloth_agent.experiments.graph_policy_recovery import free_plan_one
    p,g=policy(tmp_path);obs=observation(tmp_path/'current');r=Runtime()
    def model_call(prompt,**kwargs):
        assert str(obs.directory) in prompt
        assert kwargs['images'][0]==obs.image
        out=kwargs['output'];out.mkdir(parents=True)
        (out/'answer.txt').write_text(SOURCE)
        return dict(status='AGENT_FINISHED',session_id='test-recovery')
    result=free_plan_one(r,obs,tmp_path/'fallback',g,model_call=model_call)
    assert result['outcome']['status']=='EXECUTED' and len(r.executions)==1
    assert result['source']==SOURCE and not r.closed


def test_recovery_and_replay_imports_do_not_load_physical_runtime():
    import subprocess
    import sys
    script='''
import sys
import cloth_agent.experiments.graph_policy_recovery
import cloth_agent.experiments.graph_policy_replay
for name in ('cloth_agent.codex_loop', 'cloth_agent.codex_arm7_runtime',
             'cloth_agent.codex_native_runtime', 'xarm', 'pyrealsense2'):
    assert name not in sys.modules, name
'''
    subprocess.run([sys.executable,'-c',script],check=True,capture_output=True,text=True)


@pytest.mark.parametrize('family,expected', [('right_sleeve_inward_fold',15.),
    ('right_sleeve_curl_inward_adjustment',15.),('left_sleeve_inward_fold',15.),
    ('right_torso_side_inward_fold',15.),
    ('right_torso_hem_corner_inward_fold',15.),('right_upper_torso_inward_tuck',15.),
    ('left_torso_side_inward_fold',15.),('bottom_hem_toward_collar_fold',15.)])
def test_replay_keeps_recorded_base_without_side_delta(tmp_path,family,expected):
    from tests.test_graph_policy import observation, SOURCE
    _,g=policy(tmp_path);e=copy.deepcopy(g['edges'][0]);e['action_family']=family
    obs=observation(tmp_path/'obs');folder=tmp_path/'exec';folder.mkdir()
    class Capture(Runtime):
        def execute(self,source,before,directory):
            self.offset=self.grasp_offset_mm
            assert source==SOURCE
            return super().execute(source,before,directory)
    r=Capture()
    pr.execute_policy_edge(r,SOURCE,obs,folder,g,e)
    assert r.offset==expected and r.grasp_offset_mm==15.
    assert pr.grounding_payload(g,e,obs)['grasp_offset_mm']==expected
    assert gp.read(folder/'grasp_offset_override.json')['applied_offset_mm']==expected
    assert gp.read(folder/'grasp_offset_override.json')['extra_offset_mm']==expected-15.
    assert pr.grounding_payload(g,e,obs)['grasp_offset_override']['extra_mm']==expected-15.


def test_right_sleeve_offset_restored_on_execution_exception(tmp_path):
    from tests.test_graph_policy import observation, SOURCE
    _,g=policy(tmp_path);e=copy.deepcopy(g['edges'][0]);e['action_family']='right_sleeve_inward_fold'
    class Fail(Runtime):
        def execute(self,*args):
            assert self.grasp_offset_mm==15.
            raise RuntimeError('dispatch failed')
    r=Fail();folder=tmp_path/'exec';folder.mkdir()
    with pytest.raises(RuntimeError,match='dispatch failed'):
        pr.execute_policy_edge(r,SOURCE,observation(tmp_path/'obs'),folder,g,e)
    assert r.grasp_offset_mm==15.


def test_unchanged_graph_node_triggers_fallback_with_feedback(tmp_path):
    p,g=policy(tmp_path);r=Runtime();contexts=[]
    def recover(*args,**kwargs):
        contexts.append(kwargs['context']);return fake_fallback(*args,**kwargs)
    m=RecoveryModel([matched('N0'),matched('N0'),matched('N1')])
    out=pr.replay(p,tmp_path/'run',r,m,max_fallbacks=2,fallback_fn=recover)
    assert out['fallback_used'] and out['fallback_steps']==1 and len(r.executions)==2
    assert contexts[0]['reason']=='SOURCE_NODE_UNCHANGED'
    assert contexts[0]['previous_transition']['edge_id']=='E0'
    assert out['fallback_transitions'][0]['actual_node']=='N1'
    assert not out['transitions'][0]['reached_expected_target_node']
    ext=gp.load_frozen(out['extended_graph']);pr.validate_policy(ext)
    assert ext['nodes']==g['nodes'] and ext['edges']==g['edges']
    assert ext['recovery_extension']['transitions'][0]['source_node']=='N0'
    assert ext['recovery_extension']['transitions'][0]['target_node']=='N0'


def test_repeated_unchanged_fallback_hits_limit_without_replaying_edge(tmp_path):
    p,_=policy(tmp_path);r=Runtime()
    m=RecoveryModel([matched('N0')]*4)
    out=pr.replay(p,tmp_path/'run',r,m,max_fallbacks=2,fallback_fn=fake_fallback)
    assert out['status']=='FALLBACK_LIMIT' and out['fallback_steps']==2
    assert len(r.executions)==3 and out['executed_edges']==['E0']
    assert not out['graph_only_success']
    pr.validate_policy(gp.load_frozen(out['extended_graph']))


def test_terminal_requires_fresh_match_and_no_extra_action(tmp_path):
    path, g = policy(tmp_path)
    g['terminal_node_ids'] = ['N1']
    path.unlink(); Path(str(path)+'.sha256.json').unlink(); gp.frozen(path,g)
    runtime=Runtime(); model=Model()
    result=pr.replay(path,tmp_path/'terminal_run',runtime,model)
    assert result['status']=='COMPLETED' and result['terminal_node']=='N1'
    assert result['graph_only_success']
    assert len(runtime.executions)==1
    assert [x[0] for x in model.calls]==['node_match','ground_edge','node_match']
    runtime=Runtime()
    result=pr.replay(path,tmp_path/'unchanged_run',runtime,Model([matched('N0'),matched('N0')]))
    assert result['status']=='SOURCE_NODE_UNCHANGED'
    assert not result['graph_only_success']


def test_hem_direction_rejects_reported_reversal_and_wrong_transfer():
    frame=dict(collar_pixel_xy=[644,609], hem_center_pixel_xy=[653,143])
    c=candidate()
    c['garment_frame'].update(collar_pixel_xy=[653,143],hem_center_pixel_xy=[644,609])
    c['slots'][0]['pixel_xy']=[644,609]
    with pytest.raises(ValueError,match='contradicts'):
        pr.validate_hem_direction(c,frame)
    c['garment_frame'].update(frame)
    with pytest.raises(ValueError,match='not near verified hem'):
        pr.validate_hem_direction(c,frame)
    c['slots'][0]['pixel_xy']=[653,143]
    c['slots'][-1]['pixel_xy']=[650,90]
    with pytest.raises(ValueError,match='does not approach'):
        pr.validate_hem_direction(c,frame)
    c['slots'][-1]['pixel_xy']=[646,560]
    assert pr.validate_hem_direction(c,frame)['status']=='CONSISTENT'


def test_hem_direction_supports_rotated_garment():
    frame=dict(collar_pixel_xy=[15,50],hem_center_pixel_xy=[95,50])
    c=candidate();c['garment_frame'].update(frame)
    c['slots'][0]['pixel_xy']=[94,51];c['slots'][-1]['pixel_xy']=[25,49]
    assert pr.validate_hem_direction(c,frame)['status']=='CONSISTENT'


def test_unknown_hem_identity_blocks_before_grounding_and_execution(tmp_path):
    path,g=policy(tmp_path)
    sourcepath=Path(g['policy_provenance']['source_graph'])
    source=gp.load_frozen(sourcepath)
    source['edges'][0]['action_family']='bottom_hem_toward_collar_fold'
    sourcepath.unlink();Path(str(sourcepath)+'.sha256.json').unlink();gp.frozen(sourcepath,source)
    g['edges'][0]['action_family']='bottom_hem_toward_collar_fold'
    g['policy_provenance']['source_sha256']=gp.file_hash(sourcepath)
    path.unlink();Path(str(path)+'.sha256.json').unlink();gp.frozen(path,g)
    class UnknownFrameModel(Model):
        def ask(self,kind,payload,images,validator=None):
            if kind!='hem_frame':return super().ask(kind,payload,images,validator)
            self.calls.append((kind,payload,images))
            assert all(Path(p).is_file() for _,p in images)
            assert 'action_family' not in json.dumps(payload)
            return dict(status='UNKNOWN',collar_pixel_xy=[0,0],hem_center_pixel_xy=[0,0],
                collar_evidence='',hem_evidence='',correspondence_evidence='hidden neck')
    runtime=Runtime();model=UnknownFrameModel()
    result=pr.replay(path,tmp_path/'hem_unknown',runtime,model)
    assert result['status']=='EDGE_INSTANTIATION_FAILURE'
    assert not runtime.executions
    assert [x[0] for x in model.calls]==['node_match','hem_frame']
    assert (tmp_path/'hem_unknown/step_000/hem_orientation.json').exists()
