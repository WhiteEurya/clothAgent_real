"""Offline replay transitions; no physical runtime or model service is constructed."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from cloth_agent.experiments import graph_policy as gp
from cloth_agent.experiments import graph_policy_replay as pr
from cloth_agent.experiments.graph_policy_recovery import free_plan_one
from cloth_agent.experiments.graph_minimal_grounding import MinimalGrounder, NoRobustAction
from cloth_agent.experiments.robustness_recovery import REASON
from tests.test_graph_policy import Runtime, SOURCE, observation, match
from tests.test_graph_policy_replay import policy, candidate, matched, Model


def rejected():
    value=candidate()
    value.update(status='NEED_MORE_INFORMATION',slots=[],
        missing_visual_parameters=['NO_ROBUST_ACTION: grasp layer and jaw clearance unresolved'])
    return value


class ChangingRuntime(Runtime):
    def __init__(self, *, unchanged=False, failure=None):
        super().__init__(failure);self.observations=[];self.unchanged=unchanged
    def observe(self, folder):
        self.observations.append(folder)
        return observation(folder, 'white' if self.unchanged else (20*len(self.observations),30,40))
    def execute(self, source, obs, folder):
        self.executions.append(source)
        return dict(status=self.failure or 'EXECUTED',physical_execution=self.failure is None)


class GateModel(Model):
    def __init__(self, matches=None, groundings=None):
        super().__init__(matches);self.groundings=list(groundings or [rejected(),candidate()])
    def ask(self, kind, payload, images, validator=None):
        if kind=='ground_edge':
            self.grounding=self.groundings.pop(0)
            self.grounding['edge_id']=payload['selected_edge']['edge_id']
        return super().ask(kind,payload,images,validator)


def runner(runtime, *, answer=SOURCE, exception=None):
    calls=[];prompts=[]
    def model_call(prompt,output,**kwargs):
        prompts.append(prompt)
        if exception:raise exception
        output.mkdir();(output/'answer.txt').write_text(answer)
        return dict(status='AGENT_FINISHED',session_id='recovery-test-session')
    def run(rt,obs,folder,graph,**kwargs):
        calls.append(dict(observation=obs,context=kwargs['context']))
        assert rt is runtime
        # The model budget double lacks counters; production uses the shared Requests budget.
        kwargs.pop('budget')
        return free_plan_one(rt,obs,folder,graph,model_call=model_call,**kwargs)
    return run,calls,prompts


def audit(output):return json.loads((output/'robustness_recovery/audit.json').read_text())


def test_recovery_then_fresh_same_node_reenters_graph_without_extension(tmp_path):
    path,graph=policy(tmp_path);original=path.read_bytes()
    runtime=ChangingRuntime();model=GateModel([matched('N0'),matched('N0'),matched('N1')])
    run,calls,prompts=runner(runtime);out=tmp_path/'run'
    result=pr.replay(path,out,runtime,model,fallback_fn=run,max_recovery_attempts=1)
    assert result['status']=='NO_REPLAYABLE_EDGE'
    assert len(calls)==1 and len(prompts)==1 and len(runtime.executions)==2
    assert result['recovery_attempts']==1 and result['executed_edges']==['E0']
    assert [c[0] for c in model.calls]==['node_match','ground_edge','node_match','ground_edge','node_match']
    assert calls[0]['observation'].directory==out/'initial_observation'
    context=calls[0]['context']
    assert context['reason']==REASON and context['recommended_edge']['edge_id']=='E0'
    assert 'jaw clearance' in str(context['gate_rejection_reasons'])
    assert '不是强行完成该 fold' in prompts[0] and 'Hard dispatch gate' in prompts[0]
    assert model.calls[2][1]==model.calls[0][1]  # Fresh, unconditioned node matcher.
    assert model.calls[2][2][0][1]!=model.calls[0][2][0][1]
    row=audit(out)['attempts'][0]
    assert row['rematch']['result']['node_id']=='N0'
    assert row['target_edge_reject_to_dispatch'] is True
    assert row['subsequent_matches'][-1]['result']['node_id']=='N1'
    assert row['subsequent_graph_outcomes'][0]['execution']['status']=='EXECUTED'
    assert row['subsequent_graph_outcomes'][0]['after']['path']==row['subsequent_matches'][-1]['current_rgb']['path']
    assert row['before']['path']!=row['after']['path'] and row['graph_unchanged']
    assert path.read_bytes()==original and gp.load_frozen(path)==graph
    assert not list(out.glob('policy_extended_*')) and not (out/'applicability_recovery').exists()
    assert not list(out.rglob('grounding_retry_*'))


def test_recovery_may_rematch_different_node_and_use_its_edge(tmp_path):
    path,graph=policy(tmp_path)
    sourcepath=Path(graph['policy_provenance']['source_graph']);base=gp.load_frozen(sourcepath)
    node=copy.deepcopy(base['nodes'][0]);node['node_id']='N2'
    base['nodes'].append(node)
    e=copy.deepcopy(base['edges'][0]);e.update(edge_id='E2',source_node='N2');base['edges'].append(e)
    sourcepath.unlink();Path(str(sourcepath)+'.sha256.json').unlink();gp.frozen(sourcepath,base)
    graph['nodes'].append(node);graph['match_nodes'].append(copy.deepcopy(node))
    e=copy.deepcopy(graph['edges'][0]);e.update(edge_id='E2',source_node='N2');graph['edges'].append(e)
    graph['policy_provenance']['source_sha256']=gp.file_hash(sourcepath)
    path.unlink();Path(str(path)+'.sha256.json').unlink();gp.frozen(path,graph)
    rt=ChangingRuntime();m=GateModel([matched('N0'),matched('N2'),matched('N1')]);run,calls,_=runner(rt)
    out=tmp_path/'run';result=pr.replay(path,out,rt,m,fallback_fn=run)
    assert result['executed_edges']==['E2'] and len(calls)==1
    row=audit(out)['attempts'][0]
    assert row['rematch']['result']['node_id']=='N2' and not row['target_edge_reject_to_dispatch']
    assert row['subsequent_graph_outcomes'][0]['edge_id']=='E2'
    assert gp.load_frozen(path)==graph


@pytest.mark.parametrize('answer',[
    'motion.blocked(reason="NO_ROBUST_ACTION: no reliable recovery handle")',
    'motion.complete()',
    'motion.gripper(arm="right",action="close")',
    SOURCE+SOURCE,  # Two acquisitions are not one recovery action.
])
def test_no_reliable_or_invalid_recovery_stops_without_execution(tmp_path,answer):
    path,_=policy(tmp_path);rt=ChangingRuntime();m=GateModel();run,calls,_=runner(rt,answer=answer)
    out=tmp_path/'run';result=pr.replay(path,out,rt,m,fallback_fn=run)
    assert result['status']=='RECOVERY_UNRESOLVED' and len(calls)==1
    assert not rt.executions and len(rt.observations)==1
    assert result['recovery_attempts']==1 and rt.stopped
    assert audit(out)['attempts'][0]['status']=='RECOVERY_UNRESOLVED'


@pytest.mark.parametrize('failure',['PLANNING_FAILED','PLANNING_INCOMPLETE','EXECUTION_FAILED'])
def test_recovery_rejection_stops_before_any_graph_retry(tmp_path,failure):
    path,_=policy(tmp_path);rt=ChangingRuntime(failure=failure);m=GateModel();run,calls,_=runner(rt)
    out=tmp_path/'run';result=pr.replay(path,out,rt,m,fallback_fn=run)
    assert result['status']=='RECOVERY_UNRESOLVED' and len(calls)==1
    assert len(rt.executions)==1 and len(rt.observations)==1
    assert [c[0] for c in m.calls]==['node_match','ground_edge']
    row=audit(out)['attempts'][0]
    assert row['recovery_execution']['status']==failure and not row['target_edge_reject_to_dispatch']


@pytest.mark.parametrize('unchanged',[False,True])
def test_repeated_rejection_never_runs_second_recovery_or_same_snapshot_grounding(tmp_path,unchanged):
    path,_=policy(tmp_path);rt=ChangingRuntime(unchanged=unchanged)
    m=GateModel([matched('N0'),matched('N0')],[rejected(),rejected()]);run,calls,_=runner(rt)
    out=tmp_path/'run';result=pr.replay(path,out,rt,m,fallback_fn=run)
    assert result['status']=='RECOVERY_UNRESOLVED' and len(calls)==1 and len(rt.executions)==1
    assert len([c for c in m.calls if c[0]=='ground_edge'])==(1 if unchanged else 2)
    data=audit(out);assert len(data['gate_rejections'])==2 and data['recovery_attempts']==1
    assert not data['attempts'][0]['target_edge_reject_to_dispatch']


@pytest.mark.parametrize('status',['NO_MATCH','AMBIGUOUS'])
def test_unmatched_recovery_stops_without_creating_node(tmp_path,status):
    path,graph=policy(tmp_path);rt=ChangingRuntime();m=GateModel([matched('N0'),match(status)])
    run,calls,_=runner(rt);out=tmp_path/'run'
    result=pr.replay(path,out,rt,m,fallback_fn=run)
    assert result['status']=='RECOVERY_UNRESOLVED' and len(calls)==1
    assert not any(c[0]=='recovery_node' for c in m.calls)
    assert gp.load_frozen(path)==graph and not list(out.glob('policy_extended_*'))
    assert audit(out)['attempts'][0]['rematch']['result']['status']==status


def test_minimal_null_gate_goes_to_recovery_without_recapture(tmp_path):
    path,_=policy(tmp_path);rt=ChangingRuntime();m=GateModel()
    grounder=object.__new__(MinimalGrounder)
    grounder.path=tmp_path/'templates.json'
    grounder.template=lambda edge:dict(spec={},image_size=[20,20],confidence_min=.8)
    m.minimal_grounder=grounder
    original=m.ask
    def ask(kind,payload,images,validator=None):
        if kind!='ground_edge_minimal':return original(kind,payload,images,validator)
        m.calls.append((kind,payload,images))
        value=dict(grasp_px=None,target_px=None,confidence=0,robustness=dict(tags=[],confidence=0,
                   reason='NO_ROBUST_ACTION: ambiguous layers'))
        validator(value);return value
    m.ask=ask
    run,calls,_=runner(rt,answer='motion.blocked(reason="NO_ROBUST_ACTION: recovery unavailable")')
    out=tmp_path/'run';result=pr.replay(path,out,rt,m,fallback_fn=run)
    assert result['status']=='RECOVERY_UNRESOLVED' and len(calls)==1
    assert [c[0] for c in m.calls]==['node_match','ground_edge_minimal']
    assert len(rt.observations)==1 and not list(out.rglob('grounding_retry_*'))
    assert 'ambiguous layers' in str(calls[0]['context']['gate_rejection_reasons'])


def test_model_failure_consumes_one_attempt_without_retry(tmp_path):
    path,_=policy(tmp_path);rt=ChangingRuntime();m=GateModel();run,calls,_=runner(rt,exception=RuntimeError('model unavailable'))
    result=pr.replay(path,tmp_path/'run',rt,m,fallback_fn=run)
    assert result['status']=='RECOVERY_UNRESOLVED' and result['recovery_attempts']==1
    assert len(calls)==1 and not rt.executions


def test_last_step_still_rematches_after_recovery(tmp_path):
    path,_=policy(tmp_path);rt=ChangingRuntime();m=GateModel([matched('N0'),matched('N0')]);run,calls,_=runner(rt)
    out=tmp_path/'run';result=pr.replay(path,out,rt,m,fallback_fn=run,max_steps=1)
    assert result['status']=='STEP_LIMIT' and len(rt.executions)==1
    assert audit(out)['attempts'][0]['rematch']['result']['node_id']=='N0'


def test_recovery_budget_and_no_graph_extension_modes(tmp_path):
    path,_=policy(tmp_path)
    for kwargs in (dict(max_recovery_attempts=2),dict(max_recovery_attempts=1,max_fallbacks=1),
                   dict(max_recovery_attempts=1,applicability_recovery=True)):
        with pytest.raises(ValueError):pr.replay(path,tmp_path/'unused',ChangingRuntime(),GateModel(),**kwargs)
    rt=ChangingRuntime();run,calls,_=runner(rt)
    result=pr.replay(path,tmp_path/'disabled',rt,GateModel(),fallback_fn=run,max_recovery_attempts=0)
    assert result['status']=='RECOVERY_UNRESOLVED' and not calls and not rt.executions


def test_recovery_capture_failure_is_unresolved_and_does_not_reuse_old_observation(tmp_path):
    path,_=policy(tmp_path)
    class CaptureFailure(ChangingRuntime):
        def observe(self,folder):
            if self.observations:raise RuntimeError('camera unavailable after recovery')
            return super().observe(folder)
    rt=CaptureFailure();m=GateModel();run,calls,_=runner(rt);out=tmp_path/'run'
    result=pr.replay(path,out,rt,m,fallback_fn=run)
    assert result['status']=='RECOVERY_UNRESOLVED' and len(rt.executions)==1
    assert len([c for c in m.calls if c[0]=='ground_edge'])==1
    row=audit(out)['attempts'][0]
    assert row['after'] is None and row['rematch'] is None
    assert 'camera unavailable' in row['error']


def test_cli_defaults_to_nonextending_gate_recovery(tmp_path,monkeypatch):
    path,_=policy(tmp_path);seen=[]
    monkeypatch.setattr(gp,'make_runtime',lambda *a:ChangingRuntime())
    monkeypatch.setattr(pr,'policy_requests',lambda *a,**kw:Model())
    def replay(*args,**kwargs):
        seen.append(kwargs)
        return dict(status='RECOVERY_UNRESOLVED')
    monkeypatch.setattr(pr,'replay',replay)
    out=tmp_path/'run'
    assert pr.main(['--graph',str(path),'--output',str(out),'--execute','--no-live-graph'])==1
    assert seen[0]['max_recovery_attempts']==1 and seen[0]['max_fallbacks']==0
    assert seen[0]['applicability_recovery'] is False
    assert json.loads((out/'stop_summary.json').read_text())['status']=='RECOVERY_UNRESOLVED'


def test_cli_rejects_combined_extending_recovery_before_hardware(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('hardware must not initialize')
    monkeypatch.setattr(gp,'make_runtime',forbidden)
    with pytest.raises(SystemExit):
        pr.main(['--graph',str(tmp_path/'unused'),'--execute','--max-fallbacks','1'])
