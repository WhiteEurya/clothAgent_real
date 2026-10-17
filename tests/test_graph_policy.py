"""Offline only: no robot runtime construction, network calls or real model."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from cloth_agent.experiments import graph_policy as gp


def observation(folder, color='white'):
    folder = Path(folder); folder.mkdir(parents=True)
    Image.new('RGB', (20, 20), color).save(folder / 'rgb.png')
    xyz = np.full((20, 20, 3), [100., 200., 50.])
    np.save(folder / 'world_xyz_mm.npy', xyz)
    (folder / 'observation.json').write_text('{}')
    return SimpleNamespace(directory=folder, image=folder / 'rgb.png', xyz=xyz,
                           sample=lambda pixel, spread: xyz[pixel[1], pixel[0]])


SOURCE = 'motion.grasp(right=[0.1,0.2,0.035,3.14,0,0])\nmotion.move_tcp(right=[0.2,0.3,0.1,3.14,0,0])\n'


def fixture_graph(tmp_path):
    obs = observation(tmp_path / 'old')
    trajectory = dict(goal={'subgoal': 'right_sleeve'}, grasp_offset_mm=15.,
        observations={'o0': {'rgb': gp.evidence_file(obs.image)}, 'o1': {'rgb': gp.evidence_file(obs.image)}},
        steps=[dict(step_id='s0', before_observation='o0', after_observation='o1',
                    commanded_action=gp.command(SOURCE), actual_execution={'physical_execution': True, 'status': 'EXECUTED'})],
        metrics={'model_calls': 2}, result={'status': 'MODEL_REPORTED_COMPLETE'}, limitations=[])
    node = dict(node_id='N0', task_context='right_sleeve', observation_condition=dict(
        positive_evidence=['visible cuff'], negative_evidence=[], uncertainty_conditions=[]),
        representative_observations=['o0'], confidence=.9)
    target = copy.deepcopy(node); target.update(node_id='N1', representative_observations=['o1'])
    edge = dict(edge_id='E0', source_node='N0', target_node='N1', action_family='test transfer',
        action_template=dict(grasp_relation='cuff', target_relation='torso', grasp_z_rule='measured surface with host lowering',
                             yaw_rule='preserve relation', other_constraints=[]),
        seed_step_id='s0', seed_action=dict(exact_command=gp.command(SOURCE), source_observation='o0'),
        observed_outcome='one observed fold', outcome_evidence=['o1', 's0'], seed_outcome='SUCCESS',
        generalization_status='ONE_SHOT_HYPOTHESIS')
    graph = dict(graph_id='G', nodes=[node, target], edges=[edge], limitations=['one shot'])
    gp.frozen(tmp_path / 'seed_trajectory.json', trajectory)
    return graph, trajectory


def write_graph(tmp_path, graph, trajectory):
    graph = copy.deepcopy(graph)
    graph.update(protocol=gp.PROTOCOL, goal=trajectory['goal'], grasp_offset_mm=15.,
                 seed_trajectory_hash=gp.digest(trajectory), seed_trajectory_path=str(tmp_path / 'seed_trajectory.json'))
    gp.frozen(tmp_path / 'graph_v1.json', graph)
    return tmp_path / 'graph_v1.json'


def match(status='MATCH'):
    return dict(status=status, node_id='N0' if status == 'MATCH' else '', candidate_nodes=['N0'] if status != 'NO_MATCH' else [],
                matched_evidence=['current cuff'], violated_conditions=[], unresolved_conditions=[], confidence=.95)


def candidate():
    return dict(status='READY', edge_id='E0', action_family='test transfer',
        slots=[dict(operation_index=i, pixel_xy=[5+i, 5], z_above_surface_mm=0 if i == 0 else 50,
                    rpy_rad=[3.14, 0., 0.], relation_evidence='current anchor') for i in range(2)],
        explanation='existing relations', unresolved_conditions=[])


def assessment():
    return dict(expected_target='REACHED', matched_node_id='N1', task_goal='COMPLETE', acquisition='UNKNOWN',
                evidence=['after: cuff inboard'], failure_category='NONE', limitations=[], confidence=.95)


class FakeRequests:
    def __init__(self, overrides=None): self.calls=[]; self.overrides=overrides or {}
    def ask(self, kind, payload, images, validator=None):
        self.calls.append((kind, payload, images))
        value = copy.deepcopy(self.overrides.get(kind, dict(match=match(), instantiate=candidate(),
            conformance=dict(status='CONFORMS', evidence=['current anchor conforms'], violated_constraints=[],
                             unresolved_conditions=[], confidence=.95), assess=assessment())[kind]))
        if validator: validator(value)
        return value
    def metrics(self): return dict(model_calls=len(self.calls), tool_calls=0, visual_tool_calls=0, reasoning_time_s=.1, cache_hits=0)


class Runtime:
    physical=True; grasp_offset_mm=15.; active_arms=('right',)
    planning_policy='arm7 test policy'; motion_contract='normal contract'; preparation_description='home then perception'
    def __init__(self, failure=None): self.executions=[]; self.closed=False; self.stopped=False; self.failure=failure
    def observe(self, folder): return observation(folder)
    def execute(self, source, obs, folder):
        self.executions.append(source)
        return dict(status=self.failure or 'EXECUTED', physical_execution=True)
    def close(self): self.closed=True
    def stop(self): self.stopped=True


@pytest.mark.parametrize('status', ['NO_MATCH', 'AMBIGUOUS'])
def test_nonmatch_never_instantiates_or_executes(tmp_path, status):
    g,t=fixture_graph(tmp_path); p=write_graph(tmp_path,g,t); runtime=Runtime(); req=FakeRequests({'match':match(status)})
    result=gp.replay(p,tmp_path/'replay',runtime,req)
    assert len(req.calls)==1 and not runtime.executions
    assert not result['graph_only_success'] and not result['fallback_used'] and runtime.closed


def test_graph_replay_existing_edge_and_surface_not_seed_xyz(tmp_path):
    g,t=fixture_graph(tmp_path); p=write_graph(tmp_path,g,t); runtime=Runtime(); req=FakeRequests()
    result=gp.replay(p,tmp_path/'replay',runtime,req)
    assert [c[0] for c in req.calls]==['match','instantiate','conformance','assess']
    assert result['graph_only_success'] and result['executed_edges']==['E0']
    assert '0.05' in runtime.executions[0]  # original measured Z, no double 15mm lowering
    assert gp.read(tmp_path/'replay/step_000/dispatch_intent.json')['edge_id']=='E0'
    assert gp.load_frozen(p)['edges']==g['edges']
    assert gp.read(tmp_path/'replay/final_metrics.json')['model_calls']==4


@pytest.mark.parametrize('change', ['family','edge','order','offset','left','extra','unknown'])
def test_illegal_instantiations_rejected(tmp_path, change):
    g,_=fixture_graph(tmp_path); c=candidate(); obs=observation(tmp_path/'new')
    if change=='family': c['action_family']='new strategy'
    if change=='edge': c['edge_id']='invented'
    if change=='order': c['slots'].reverse()
    if change=='offset': c['slots'][0]['z_above_surface_mm']=2
    if change=='left': g['edges'][0]['seed_action']['exact_command']['operations'][0]['arguments']={'left':[0]*6}
    if change=='extra': c['slots'].append(c['slots'][1])
    if change=='unknown': c['unresolved_conditions']=['unresolved grasp relation']
    with pytest.raises(ValueError): gp.instantiate(c,g['edges'],obs)


def test_conformance_uncertainty_no_execution(tmp_path):
    g,t=fixture_graph(tmp_path); p=write_graph(tmp_path,g,t); runtime=Runtime()
    req=FakeRequests({'conformance':dict(status='UNKNOWN', evidence=[], violated_constraints=[], unresolved_conditions=['occluded'],confidence=.2)})
    result=gp.replay(p,tmp_path/'replay',runtime,req)
    assert not runtime.executions and result['failure_category']=='EDGE_GENERALIZATION_FAILURE'


@pytest.mark.parametrize('change', ['execution','unknown_outcome','wrong_target','low_confidence'])
def test_execution_and_visual_success_are_not_conflated(tmp_path, change):
    g,t=fixture_graph(tmp_path); p=write_graph(tmp_path,g,t); runtime=Runtime('EXECUTION_FAILED' if change=='execution' else None)
    a=assessment()
    if change=='unknown_outcome': a['task_goal']='UNKNOWN'
    if change=='wrong_target': a['matched_node_id']='N0'
    if change=='low_confidence': a['confidence']=.2
    result=gp.replay(p,tmp_path/'replay',runtime,FakeRequests({'assess':a}))
    assert not result['graph_only_success']
    if change=='execution': assert result['failure_category']=='ACTION_EXECUTION_FAILURE'


def test_failed_edge_never_preferred(tmp_path):
    g,t=fixture_graph(tmp_path); g['edges'][0].update(seed_outcome='FAILURE',generalization_status='OBSERVED_FAILURE')
    p=write_graph(tmp_path,g,t); runtime=Runtime(); req=FakeRequests()
    result=gp.replay(p,tmp_path/'replay',runtime,req)
    assert not runtime.executions and len(req.calls)==1 and result['status']=='NO_ELIGIBLE_EDGE'


def test_graph_grounding_and_failure_contract(tmp_path):
    g,t=fixture_graph(tmp_path); gp.graph_validation(g,t)
    bad=copy.deepcopy(g); bad['edges'][0]['seed_action']['exact_command']['source']='fabricated'
    with pytest.raises(ValueError,match='exact_command'): gp.graph_validation(bad,t)
    bad=copy.deepcopy(g); bad['edges']=[]
    with pytest.raises(ValueError,match='every physical'): gp.graph_validation(bad,t)
    t['steps'][0]['actual_execution']['status']='EXECUTION_FAILED'
    with pytest.raises(ValueError,match='OBSERVED_FAILURE'): gp.graph_validation(g,t)


def test_cache_real_image_hash_and_frozen_graph(tmp_path):
    img=observation(tmp_path/'im').image
    class Backend:
        model='fake'
        def __init__(self): self.calls=[]
        def invoke(self,prompt,images,schema):
            self.calls.append(images)
            assert images==[img] and 'image_ids' in prompt
            return dict(text=json.dumps(match()), usage={'tokens':1})
    b=Backend(); req=gp.Requests(tmp_path/'requests',b)
    req.ask('match',{'graph':{'version':1}},[('current',img)])
    req.ask('match',{'graph':{'version':1}},[('current',img)])
    assert len(b.calls)==1 and req.cache_hits==1
    Image.new('RGB',(20,20),'red').save(img)
    req.ask('match',{'graph':{'version':1}},[('current',img)])
    req.ask('match',{'graph':{'version':2}},[('current',img)])
    assert len(b.calls)==3
    gp.frozen(tmp_path/'frozen.json',{'a':1})
    with pytest.raises(ValueError): gp.frozen(tmp_path/'frozen.json',{'a':2})


def test_seed_uses_normal_loop_session_and_complete_trajectory(tmp_path):
    runtime=Runtime(); prompts=[]
    def model(prompt,output,**kwargs):
        prompts.append(prompt); output.mkdir()
        (output/'answer.txt').write_text(SOURCE if len(prompts)==1 else 'motion.complete()')
        assert kwargs['session_id']==(None if len(prompts)==1 else 'same')
        return dict(status='AGENT_FINISHED',session_id='same')
    gp.seed(tmp_path/'seedrun',runtime,subgoal='right_sleeve',model_call=model,max_rounds=3)
    t=gp.load_frozen(tmp_path/'seedrun/seed_trajectory.json')
    assert len(runtime.executions)==1 and runtime.closed
    assert t['steps'][0]['after_observation']==t['steps'][1]['before_observation']
    assert len(t['observations'])==2 and t['metrics']['model_calls']==2
    assert all('right_sleeve' in p for p in prompts)
    assert not any('graph_v1' in p for p in prompts)


def test_seed_bound_captures_terminal_after_only_after_success(tmp_path):
    for failure in (None,'EXECUTION_FAILED'):
        root=tmp_path/str(failure); runtime=Runtime(failure)
        def model(prompt,output,**kwargs):
            output.mkdir(); (output/'answer.txt').write_text(SOURCE)
            return dict(status='AGENT_FINISHED',session_id='test')
        gp.seed(root,runtime,model_call=model,max_rounds=1)
        t=gp.load_frozen(root/'seed_trajectory.json')
        assert bool(t['steps'][0]['after_observation'])==(failure is None)


def test_replay_cannot_resume_physical_dispatch(tmp_path):
    g,t=fixture_graph(tmp_path); p=write_graph(tmp_path,g,t)
    gp.replay(p,tmp_path/'replay',Runtime(),FakeRequests())
    with pytest.raises(FileExistsError): gp.replay(p,tmp_path/'replay',Runtime(),FakeRequests())


def test_compile_attaches_all_seed_rgb_and_never_overwrites(tmp_path):
    g,t=fixture_graph(tmp_path)
    class Compiler:
        def __init__(self): self.calls=0
        def ask(self,kind,payload,images,validator):
            self.calls+=1; assert kind=='compile' and len(images)==2
            assert payload['steps']==t['steps']; validator(g); return copy.deepcopy(g)
        def metrics(self): return {'model_calls':self.calls}
    c=Compiler(); gp.compile_graph(tmp_path,c); gp.compile_graph(tmp_path,c)
    assert c.calls==1 and (tmp_path/'graph_v1.md').exists()


def test_low_confidence_match_stops_and_empty_terminal_is_not_reuse(tmp_path):
    g,t=fixture_graph(tmp_path); p=write_graph(tmp_path,g,t)
    low=match(); low['confidence']=.5
    runtime=Runtime()
    result=gp.replay(p,tmp_path/'low',runtime,FakeRequests({'match':low}))
    assert not runtime.executions and result['status']=='AMBIGUOUS'
    terminal=match(); terminal.update(node_id='N1',candidate_nodes=['N1'])
    runtime=Runtime()
    result=gp.replay(p,tmp_path/'terminal',runtime,FakeRequests({'match':terminal}))
    assert not result['graph_only_success'] and not result['graph_reused'] and not runtime.executions


def test_multistep_replay_is_bounded_and_never_expands_graph(tmp_path):
    g,t=fixture_graph(tmp_path)
    g['nodes'][0]['representative_observations']=['o0','o1']
    g['edges'][0]['target_node']='N0'
    p=write_graph(tmp_path,g,t); before_hash=gp.file_hash(p)
    a=assessment(); a.update(task_goal='INCOMPLETE',matched_node_id='N0')
    runtime=Runtime()
    result=gp.replay(p,tmp_path/'replay',runtime,FakeRequests({'assess':a}),max_steps=2)
    assert len(runtime.executions)==2 and result['status']=='STEP_LIMIT'
    assert not result['graph_only_success'] and gp.file_hash(p)==before_hash


def test_image_mutation_rejected_before_compilation(tmp_path):
    g,t=fixture_graph(tmp_path)
    Path(t['observations']['o0']['rgb']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='evidence modified'):
        gp.compile_graph(tmp_path,FakeRequests())


def test_bounded_format_repair_and_recorded_tool_metrics(tmp_path):
    img=observation(tmp_path/'im').image
    class Broken:
        model='fake'
        def invoke(self,*args):
            return dict(text='not json',tool_calls=2,visual_tool_calls=1,completed_turns=1)
    req=gp.Requests(tmp_path/'requests',Broken())
    with pytest.raises(ValueError): req.ask('match',{},[('current',img)])
    assert req.model_calls==2 and req.metrics()['tool_calls']==4 and req.metrics()['visual_tool_calls']==2
    rows=[json.loads(line) for line in (tmp_path/'requests/requests.jsonl').read_text().splitlines()]
    assert [r['retry_count'] for r in rows]==[0,1] and all('error' in r for r in rows)


def test_missing_depth_prevents_execution(tmp_path):
    g,t=fixture_graph(tmp_path); p=write_graph(tmp_path,g,t)
    class NoDepth(Runtime):
        def observe(self,folder):
            obs=super().observe(folder)
            obs.sample=lambda *args: np.array([np.nan]*3)
            return obs
    runtime=NoDepth()
    result=gp.replay(p,tmp_path/'replay',runtime,FakeRequests())
    assert not runtime.executions and result['status']=='INSTANTIATION_REJECTED'


def test_failed_transport_preserves_error_report_and_stops(tmp_path):
    g,t=fixture_graph(tmp_path); p=write_graph(tmp_path,g,t)
    class BrokenRequests(FakeRequests):
        def ask(self,*args,**kwargs): raise RuntimeError('model unavailable')
    runtime=Runtime()
    with pytest.raises(RuntimeError,match='model unavailable'):
        gp.replay(p,tmp_path/'replay',runtime,BrokenRequests())
    assert runtime.closed and runtime.stopped and not runtime.executions
    assert gp.read(tmp_path/'replay/final_metrics.json')['failure_category']=='UNKNOWN'


def test_execution_event_slice_excludes_earlier_events(tmp_path):
    runtime=Runtime(); runtime.coordinator=SimpleNamespace(directory=tmp_path/'controller')
    runtime.coordinator.directory.mkdir()
    events=runtime.coordinator.directory/'events.jsonl'
    events.write_text('{"event":"earlier"}\n')
    def execute(*args):
        with events.open('a') as f: f.write('{"event":"astra_command_complete","index":1}\n')
        return {'status':'EXECUTED'}
    runtime.execute=execute
    folder=tmp_path/'execution'; folder.mkdir()
    gp.execute_recorded(runtime,SOURCE,None,folder)
    rows=gp.read(folder/'execution_events.json')['events']
    assert rows==[{'event':'astra_command_complete','index':1}]


def test_import_and_compile_cannot_load_robot_runtime(tmp_path):
    import subprocess
    import sys
    g,t=fixture_graph(tmp_path)
    gp.save(tmp_path/'builder_response.json',g)
    script = '''
import sys, importlib.abc, json
from pathlib import Path
class DenyRobot(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if any(part in fullname.split('.') for part in ('codex_loop','codex_arm7_runtime','codex_native_runtime',
                   'xarm','pyrealsense2','kinematics')):
            raise AssertionError('unexpected robot import: '+fullname)
sys.meta_path.insert(0,DenyRobot())
from cloth_agent import RobotAPI
def forbid_robot(*args,**kwargs): raise AssertionError('RobotAPI construction forbidden')
RobotAPI.__init__=forbid_robot
from cloth_agent.experiments.graph_policy import Requests, compile_graph
root=Path(sys.argv[1])
class Backend:
    model='offline-test'
    def invoke(self,prompt,images,schema):
        assert len(images)==2
        return {'text':(root/'builder_response.json').read_text()}
compile_graph(root,Requests(root/'compile',Backend()))
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path)],check=True,cwd=gp.ROOT)
    assert (tmp_path/'graph_v1.json').is_file()


def test_cleanup_failure_does_not_erase_audit(tmp_path):
    g,t=fixture_graph(tmp_path); p=write_graph(tmp_path,g,t)
    class CloseFailure(Runtime):
        def close(self): raise RuntimeError('connection close failed')
    result=gp.replay(p,tmp_path/'replay',CloseFailure(),FakeRequests({'match':match('NO_MATCH')}))
    assert result['status']=='GRAPH_MISS' and 'cleanup_error' in result
    assert (tmp_path/'replay/final_metrics.json').is_file()


def test_full_fold_default_passes_entire_demonstration_and_preserves_multistep(tmp_path):
    runtime=Runtime(); prompts=[]; attached=[]
    def model(prompt,output,**kwargs):
        prompts.append(prompt); attached.append(kwargs['images']); output.mkdir()
        (output/'answer.txt').write_text(SOURCE if len(prompts)<4 else 'motion.complete()')
        return dict(status='AGENT_FINISHED',session_id='full-session')
    gp.seed(tmp_path/'full',runtime,model_call=model)
    t=gp.load_frozen(tmp_path/'full/seed_trajectory.json')
    assert t['goal']['subgoal']=='full_fold' and len(t['goal_references'])==5
    assert all(len(images)==6 for images in attached)
    assert len(runtime.executions)==3 and len(t['steps'])==4
    assert all('完成一个袖子或某个中间步骤不能输出complete' in p for p in prompts)
    assert all('仅完成右袖子目标' not in p for p in prompts)
    assert gp.read(tmp_path/'full/experiment.json')['max_rounds']==30
    assert t['coverage']['status']=='SEED_PLANNER_REPORTED_COMPLETE'
    for before,after in zip(t['steps'],t['steps'][1:]):
        assert before['after_observation']==after['before_observation']


def test_graph_rejects_disconnected_consecutive_steps(tmp_path):
    g,t=fixture_graph(tmp_path)
    t['observations']['o2']=copy.deepcopy(t['observations']['o1'])
    second=copy.deepcopy(t['steps'][0]); second.update(step_id='s1',before_observation='o1',after_observation='o2')
    t['steps'].append(second)
    node=copy.deepcopy(g['nodes'][1]); node.update(node_id='N2',representative_observations=['o1','o2'])
    g['nodes'].append(node)
    edge=copy.deepcopy(g['edges'][0]); edge.update(edge_id='E1',source_node='N2',target_node='N2',seed_step_id='s1')
    edge['seed_action']['source_observation']='o1'; g['edges'].append(edge)
    with pytest.raises(ValueError,match='share a graph node'): gp.graph_validation(g,t)
    edge['source_node']='N1'
    gp.graph_validation(g,t)


def test_partial_full_fold_is_not_reported_as_complete(tmp_path):
    runtime=Runtime()
    def model(prompt,output,**kwargs):
        output.mkdir(); (output/'answer.txt').write_text(SOURCE)
        return dict(status='AGENT_FINISHED',session_id='bounded')
    gp.seed(tmp_path/'partial',runtime,max_rounds=1,model_call=model)
    t=gp.load_frozen(tmp_path/'partial/seed_trajectory.json')
    assert t['coverage']['scope']=='full_fold' and t['coverage']['status']=='PARTIAL_TRAJECTORY'
    assert t['coverage']['physical_attempt_count']==1


def test_export_missing_outcome_retains_interrupted_dispatch(tmp_path):
    run=tmp_path/'run'; folder=run/'seed/round_001'
    obs=observation(folder/'before')
    (folder/'motion.py').write_text(SOURCE)
    gp.save(folder/'execution_events.json',{'events':[
        {'event':'astra_gripper_start'}, {'event':'astra_command_complete'}, {'event':'stop_both'}]})
    gp.save(run/'experiment.json',{'task_contract':{'goal':{'subgoal':'full_fold'}},
        'subgoal':'full_fold','grasp_offset_mm':15.})
    gp.export_seed(run,[obs],{'status':'INTERRUPTED'})
    t=gp.load_frozen(run/'seed_trajectory.json'); step=t['steps'][0]
    assert step['actual_execution']['physical_execution'] is True
    assert step['actual_execution']['completion_unknown'] is True
    assert step['actual_execution']['completed_commands']==1
    assert step['after_observation'] is None
    assert t['coverage']['status']=='PARTIAL_TRAJECTORY'


def test_dense_evidence_projection_preserves_commands_events_and_fault_statistics():
    from cloth_agent.experiments.graph_evidence import builder_steps
    samples=[{'elapsed_s':i*.1,'feedback':{'error_code':7 if i==40 else 0,
        'mechanical_grasp_detected':i>60,'position_pulse':100-i}} for i in range(100)]
    step={'step_id':'s0','commanded_action':gp.command(SOURCE),'before_observation':'o0','after_observation':'o1',
          'actual_execution':{'status':'EXECUTED'},'intermediate_evidence':{
              'gripper_003.json':{'artifact':{'path':'raw','sha256':'hash'},'content':{'samples':samples}},
              'execution_events.json':{'content':{'events':[{'event':'move','index':i,'waypoints':list(range(100))} for i in range(30)]}}}}
    old=copy.deepcopy(step); result=builder_steps([step])[0]
    assert step==old and result['commanded_action']==step['commanded_action']
    assert result['before_observation']=='o0' and result['after_observation']=='o1'
    summary=result['intermediate_evidence']['gripper_003.json']['content']['samples']
    assert summary['original_count']==100 and summary['telemetry_statistics']['feedback.error_code']['max']==7
    assert summary['telemetry_statistics']['feedback.mechanical_grasp_detected']['most_common_values']
    events=result['intermediate_evidence']['execution_events.json']['content']['events']
    assert [e['index'] for e in events]==list(range(30))
    assert events[0]['waypoints']['original_count']==100
    assert result['intermediate_evidence']['gripper_003.json']['artifact']=={'path':'raw','sha256':'hash'}


def test_oversized_request_rejected_before_model(tmp_path):
    class Backend:
        model='fake'
        def invoke(self,*args): raise AssertionError('must not invoke')
    requests=gp.Requests(tmp_path,Backend())
    with pytest.raises(ValueError,match='before model call'):
        requests.ask('compile',{'huge':'x'*800_000},[])
    assert requests.model_calls==0


def test_unknown_sink_preserves_gap_but_is_never_replayable(tmp_path):
    g,t=fixture_graph(tmp_path)
    t['steps'][0]['after_observation']=None
    t['steps'][0]['actual_execution']['status']='UNAVAILABLE'
    n=g['nodes'][1]; n.update(representative_observations=[],confidence=0)
    n['observation_condition'].update(positive_evidence=[],uncertainty_conditions=['after unavailable'])
    g['edges'][0]['seed_outcome']='UNKNOWN'
    gp.graph_validation(g,t)
    assert gp.eligible_edges(g,'N0')==[]
    selected=match(); selected.update(node_id='N1',candidate_nodes=['N1'])
    with pytest.raises(ValueError):gp.reliable_match(selected,g,.8)
    bad=copy.deepcopy(g);bad['nodes'][1]['confidence']=.5
    with pytest.raises(ValueError,match='zero-confidence sink'):gp.graph_validation(bad,t)
    bad=copy.deepcopy(g);bad['edges'][0]['source_node']='N1'
    with pytest.raises(ValueError,match='zero-confidence sink'):gp.graph_validation(bad,t)
    bad=copy.deepcopy(g);bad['edges'][0]['seed_outcome']='SUCCESS'
    with pytest.raises(ValueError,match='cannot establish'):gp.graph_validation(bad,t)
    bad=copy.deepcopy(g);bad['nodes'][1]['representative_observations']=['o1']
    bad['nodes'][1]['observation_condition']['positive_evidence']=['made up']
    with pytest.raises(ValueError,match='missing after'):gp.graph_validation(bad,t)


def test_observed_target_cannot_be_relabelled_unobserved(tmp_path):
    g,t=fixture_graph(tmp_path)
    g['nodes'][1].update(representative_observations=[],confidence=0)
    g['nodes'][1]['observation_condition']['uncertainty_conditions']=['unknown']
    with pytest.raises(ValueError,match='missing after evidence'):gp.graph_validation(g,t)


def reviewed_terminal_graph(tmp_path):
    g,t=fixture_graph(tmp_path); source=write_graph(tmp_path,g,t)
    new=gp.load_frozen(source)
    new['terminal_node_ids']=['N1']
    new['user_review']={'authority':'USER','instruction':'N1 is terminal',
        'source_graph':str(source),'source_sha256':gp.file_hash(source),'excluded_seed_steps':[]}
    path=tmp_path/'reviewed.json';gp.frozen(path,new)
    return path,new,t


def test_user_terminal_stops_after_verified_transition_even_if_old_goal_unknown(tmp_path):
    path,g,t=reviewed_terminal_graph(tmp_path)
    a=assessment(); a['task_goal']='UNKNOWN'
    runtime=Runtime();req=FakeRequests({'assess':a})
    result=gp.replay(path,tmp_path/'replay',runtime,req,max_steps=5)
    assert result['graph_only_success'] and result['reached_terminal_node']
    assert result['completion_basis']=='USER_DESIGNATED_TERMINAL_MATCH'
    assert len(runtime.executions)==1 and len(req.calls)==4


def test_initial_terminal_never_moves_or_claims_reuse(tmp_path):
    path,g,t=reviewed_terminal_graph(tmp_path)
    m=match();m.update(node_id='N1',candidate_nodes=['N1'])
    runtime=Runtime();req=FakeRequests({'match':m})
    result=gp.replay(path,tmp_path/'replay',runtime,req)
    assert not runtime.executions and len(req.calls)==1
    assert result['reached_terminal_node'] and not result['graph_only_success']


def test_explicit_terminal_does_not_accept_weak_visual_match(tmp_path):
    path,g,t=reviewed_terminal_graph(tmp_path)
    a=assessment();a['confidence']=.3
    result=gp.replay(path,tmp_path/'replay',Runtime(),FakeRequests({'assess':a}))
    assert not result['graph_only_success']


def test_progress_is_eligible_but_controller_failure_remains_rejected(tmp_path):
    g,t=fixture_graph(tmp_path);g['edges'][0]['seed_outcome']='PROGRESS'
    gp.graph_validation(g,t)
    assert len(gp.eligible_edges(g,'N0'))==1
    t['steps'][0]['actual_execution']['status']='EXECUTION_FAILED'
    with pytest.raises(ValueError,match='OBSERVED_FAILURE'):gp.graph_validation(g,t)


def test_terminal_or_exclusion_needs_review_provenance(tmp_path):
    path,g,t=reviewed_terminal_graph(tmp_path)
    bad=copy.deepcopy(g);bad.pop('user_review')
    with pytest.raises(ValueError,match='provenance'):gp.validate_replay_graph(bad,t)
    bad=copy.deepcopy(g);bad['user_review']['excluded_seed_steps']=['s0']
    with pytest.raises(ValueError,match='exclusion coverage'):gp.validate_replay_graph(bad,t)


@pytest.mark.parametrize('outcome',['NO_PROGRESS','FAILURE','UNKNOWN'])
def test_nonprogress_experience_is_not_an_eligible_policy_action(tmp_path,outcome):
    graph,_=fixture_graph(tmp_path)
    graph['edges'][0]['seed_outcome']=outcome
    assert gp.eligible_edges(graph,'N0')==[]


def test_failure_telemetry_projection_covers_all_duplicates():
    from cloth_agent.experiments.graph_evidence import builder_steps
    samples=[dict(elapsed_s=i,feedback=dict(position_pulse=901,error_code=0)) for i in range(100)]
    execution=dict(status='EXECUTION_FAILED',physical_execution=True,error='timeout',
                   gripper_completion=dict(samples=samples))
    step=dict(commanded_action=gp.command(SOURCE),actual_execution=execution,
        outcome=dict(controller=copy.deepcopy(execution)),intermediate_evidence={
            'native_plan.json':dict(content=[dict(kind='move',data=dict(queued_segments=[dict(start=[1],end=[2])]*5))])})
    original=copy.deepcopy(step);projected=builder_steps([step])[0]
    assert step==original and projected['commanded_action']==step['commanded_action']
    for value in [projected['actual_execution'],projected['outcome']['controller']]:
        assert value['status']=='EXECUTION_FAILED' and value['physical_execution'] is True
        assert value['gripper_completion']['samples']['original_count']==100
    assert projected['intermediate_evidence']['native_plan.json']['content'][0]['data']['queued_segments']['original_count']==5


def test_compiler_projection_retains_physical_attempts_and_task_commands():
    from cloth_agent.experiments.graph_evidence import compiler_steps
    step=dict(step_id='s0',commanded_action=gp.command(SOURCE),before_observation='o0',after_observation=None,
        actual_execution=dict(status='EXECUTION_FAILED',physical_execution=True,error='timeout'),
        intermediate_evidence={'native_plan.json':dict(artifact=dict(path='raw',sha256='hash'),content=[{'kind':'move'}]),
            'execution_events.json':dict(content=dict(events=[dict(event='gripper_failure',error='timeout',sequence=7)]))})
    old=copy.deepcopy(step);out=compiler_steps([step])[0]
    assert step==old
    for key in ['step_id','commanded_action','before_observation','after_observation','actual_execution']:assert out[key]==step[key]
    assert out['intermediate_evidence']['native_plan.json']['artifact']==step['intermediate_evidence']['native_plan.json']['artifact']
    assert out['intermediate_evidence']['execution_events.json']['content']['event_sequence'][0]['error']=='timeout'
