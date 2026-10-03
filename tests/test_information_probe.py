import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from cloth_agent.harness.common import read_json
from cloth_agent.harness.information_probe import (
    ROOT, TASKS, ProbeBackend, compact_skill, main, prepare_observation, run_arm, validate_result,
)
from cloth_agent.harness.policy import PolicyError
from cloth_agent.image_tools_mcp import pixel_hash


def sample_result():
    return {'findings': [{'need_id': t['id'], 'status': 'UNKNOWN', 'answer': '无法判断',
        'source_image_ids': ['image_0'], 'method_used': ['查看全图'],
        'success_check': '细节不足', 'missing_information': '缺少清晰局部'} for t in TASKS]}


def sample_manifest(tmp_path):
    image = tmp_path / 'source.png'
    im = Image.new('RGB', (40, 60), 'purple'); im.save(image)
    return {'decisions': [{'decision_id': 'test', 'iteration_id': 'iteration',
        'pre_decision': {'observation_id': 'obs', 'candidate_registry': 'SECRET_SELECTION',
            'images': [{'role': role, 'source': 'fold_reference_target.png', 'status': 'AVAILABLE',
                       'path': str(image), 'size': list(im.size), 'rgb_sha256': pixel_hash(im)}
                       for role in ['clean', 'overlay', 'reference', 'hint']]},
        'post_decision': 'SECRET_OUTCOME', 'tool_trace': 'SECRET_TRACE'}]}


def test_observation_and_skill_do_not_leak_history(tmp_path):
    obs, images, _ = prepare_observation(sample_manifest(tmp_path), tmp_path / 'out')
    assert len(images) == 3
    assert 'SECRET' not in json.dumps(obs)
    assert 'path' not in json.dumps(obs)
    artifact = read_json(ROOT / 'data/skills/experimental/visual_information.json')
    skill = compact_skill(artifact)
    assert len(skill['skills']) == 8
    assert 'evidence' not in json.dumps(skill)
    assert 'decision_id' not in json.dumps(skill)
    assert all('UNKNOWN' in s['on_insufficient'] for s in skill['skills'])
    manifest = sample_manifest(tmp_path)
    manifest['decisions'][0]['pre_decision']['images'][0]['rgb_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='changed'):
        prepare_observation(manifest, tmp_path / 'bad')


def test_unknown_requires_reason_and_all_tasks():
    result = sample_result(); validate_result(result)
    result['findings'][0]['missing_information'] = ''
    with pytest.raises(PolicyError, match='missing_information'):
        validate_result(result)
    result = sample_result()
    result['findings'][0] = copy.deepcopy(result['findings'][1])
    with pytest.raises(PolicyError, match='exactly once'):
        validate_result(result)


class FakeBackend:
    requests = []
    def __init__(self, *, ssh_host, timeout_s, image_tools):
        self.progress_callback = None
    def invoke(self, **kwargs):
        self.requests.append(kwargs)
        stdout = json.dumps({'type': 'result', 'structured_output': sample_result(), 'modelUsage': {}})
        return SimpleNamespace(stdout=stdout, stderr='', timings={'remote_claude_s': 0.1})


def test_both_arms_use_equal_inputs_and_budgets(tmp_path, monkeypatch):
    from cloth_agent.harness import information_probe as module
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(sample_manifest(tmp_path)))
    FakeBackend.requests = []
    monkeypatch.setattr(module, 'ProbeBackend', FakeBackend)
    output = tmp_path / 'pilot'
    assert main(['--manifest', str(manifest_path), '--output', str(output)]) == 0
    skill, baseline = FakeBackend.requests
    assert skill['image_paths'] == baseline['image_paths']
    for key in ('schema', 'system_prompt', 'model', 'max_turns', 'image_edit_limit', 'overall_timeout_s'):
        assert skill[key] == baseline[key]
    a = read_json(output / 'with_skill/input.json')
    b = read_json(output / 'without_skill/input.json')
    a.pop('visual_information_skill')
    assert a == b
    report = read_json(output / 'report.json')
    assert report['speedup_established'] is False
    assert report['arms']['with_skill']['unknown'] == 4
    assert report['arms']['with_skill']['satisfied'] == 0


def test_failed_call_saves_timing(tmp_path):
    class Failing:
        last_timings = {'upload_0_s': 2}
        def invoke(self, **kwargs):
            raise TimeoutError('timeout')
    report = run_arm(Failing(), {}, [], None, tmp_path / 'failure',
                     model='test', timeout=1, edits=0, turns=2)
    assert report['status'] == 'FAILED'
    assert report['timings']['upload_0_s'] == 2
    assert (tmp_path / 'failure/report.json').is_file()


def test_backend_settings_isolation():
    backend = ProbeBackend(image_tools=True)
    backend._image_edit_limit = 4
    _, flags, _ = backend._image_tool_setup('/tmp/probe', 4)
    assert "--setting-sources user" in flags
    assert '--tools Read' in flags
    assert 'image_tools.settings.json' in flags


def test_tool_metrics_use_full_remote_trace_and_deduplicate(tmp_path):
    from cloth_agent.harness.information_probe import tool_metrics
    event = {'message': {'content': [{'type': 'tool_use', 'id': 'crop_one',
                                     'name': 'mcp__cloth_image__crop_image'}]}}
    (tmp_path / 'claude_stdout.txt').write_text(json.dumps(event) + '\n' + json.dumps(event))
    metrics = tool_metrics('{"type":"result"}', tmp_path)
    assert metrics['tool_call_count'] == 1
    assert metrics['image_edit_calls'] == 1


def test_direct_executes_full_recipe_without_intermediate_model_calls(tmp_path):
    from cloth_agent.harness.direct_information_probe import run_direct
    obs, images, _ = prepare_observation(sample_manifest(tmp_path), tmp_path / 'input')
    artifact = read_json(ROOT / 'data/skills/experimental/visual_information.json')
    binding = {'roi': [0.1, 0.2, 0.8, 0.9], 'paired_crop': True, 'enlarge': True,
               'target_description': '局部边缘', 'reason': '检查细节与标注遮挡'}
    class Model:
        def __init__(self): self.calls=[]; self.requests=[]
        def invoke(self, **kw):
            self.requests.append(kw); self.calls.append({'exploratory_tool_round_trips': 0})
            return binding if len(self.requests)==1 else sample_result()
    model=Model();out=tmp_path/'direct'
    result=run_direct(model,obs,images,artifact,out)
    assert result['status']=='COMPLETED'
    assert result['model_invocations']==2
    assert result['host_image_ops']==4
    assert len(model.requests[1]['images'])==len(images)+4
    execution=read_json(out/'execution/execution.json')
    assert [s['operation'] for s in execution['steps']]==['crop_image','crop_image','resize_image','resize_image']
    assert execution['steps'][0]['arguments']==execution['steps'][1]['arguments']
    assert execution['steps'][2]['arguments']==execution['steps'][3]['arguments']
    assert 'SECRET' not in model.requests[0]['prompt']+model.requests[1]['prompt']


def test_direct_cannot_replan_or_execute_bad_binding(tmp_path):
    from cloth_agent.harness.direct_information_probe import execute_recipe
    obs,images,_=prepare_observation(sample_manifest(tmp_path),tmp_path/'input')
    recipe=read_json(ROOT/'data/skills/experimental/visual_information.json')['execution_recipe']
    binding={'roi':None,'paired_crop':False,'enlarge':False,'target_description':'现有图足够','reason':'无需处理'}
    paths,_,steps=execute_recipe(obs,images,binding,recipe,tmp_path/'skip')
    assert len(paths)==len(images)
    assert all(s['status']=='SKIPPED_CONDITION' for s in steps)
    binding['enlarge']=True
    with pytest.raises(PolicyError,match='require ROI'):
        execute_recipe(obs,images,binding,recipe,tmp_path/'invalid')
    binding['roi']=[0.8,0.2,0.1,0.9]
    with pytest.raises(PolicyError,match='reversed'):
        execute_recipe(obs,images,binding,recipe,tmp_path/'reversed')


def test_planning_arms_share_original_contract_without_verification(tmp_path):
    from cloth_agent.auto_exploration import VISUAL_PLAN_JSON_SCHEMA
    from cloth_agent.harness.planning_probe import run_planning_arm, validate_plan
    obs,images,_=prepare_observation(sample_manifest(tmp_path),tmp_path/'input')
    case={'observation':obs,'context_hash':'same-context','fold_goal':'hem_up',
          'prompt':'Original planning instructions','system_prompt':'Original system',
          'schema':VISUAL_PLAN_JSON_SCHEMA,'context_files':{'manifest.json':'{}','00.json':'{"task":"hem_up"}'},
          'context':{'approved_skill_names':[],'fold_state_reference':{'current_step':'hem_up'}},
          'registry':{'binding':'RAW_RGB_HASH_VERIFIED','candidates':[
              {'candidate_id':'R001','camera':'A','pixel_xy':[12,15]}]}}
    plan={'garment_observation':'visible fabric','opening_strategy':'fold requested hem',
          'confidence':0.6,'selected_reference':{'camera':'A','reference_id':'R001','reason':'visible hem'},
          'motion_intent':'lift then inward laydown','expected_observation':'hem covers body',
          'safety_notes':['ground locally before any motion']}
    class Backend:
        def __init__(self):self.requests=[]
        def invoke(self,**kw):
            self.requests.append(kw)
            return SimpleNamespace(stdout=json.dumps({'type':'result','structured_output':plan,'modelUsage':{}}),stderr='',timings={})
    class Binder:
        def __init__(self):self.calls=[]
        def invoke(self,**kw):
            assert kw['stage']=='information_skill_bind'
            self.calls.append({'stage':kw['stage']})
            return {'roi':[.1,.2,.8,.9],'paired_crop':True,'enlarge':True,'reason':'task boundary',
                    'target_description':'current hem'}
    artifact=read_json(ROOT/'data/skills/experimental/visual_information.json')
    base,prepared=Backend(),Backend(); b1,b2=Binder(),Binder()
    ra=run_planning_arm(case,images,artifact,tmp_path/'base',use_skill=False,model=b1,backend=base,model_name='same')
    rb=run_planning_arm(case,images,artifact,tmp_path/'prepared',use_skill=True,model=b2,backend=prepared,model_name='same',
                        skill_flow='legacy-preprocess')
    assert ra['status']==rb['status']=='COMPLETED'
    assert len(b1.calls)==0 and len(b2.calls)==1
    assert rb['standalone_verification_calls']==0
    for key in ('schema','context_files','system_prompt','model','max_turns'):
        assert base.requests[0][key]==prepared.requests[0][key]
    assert base.requests[0]['image_edit_limit']==6
    assert prepared.requests[0]['image_edit_limit']==0
    assert len(prepared.requests[0]['image_paths'])==len(images)+4
    assert base.requests[0]['prompt']=='Original planning instructions'
    assert prepared.requests[0]['prompt'].startswith(base.requests[0]['prompt'])
    assert 'information_skill_verify' not in json.dumps(rb)
    bad=copy.deepcopy(plan);bad['selected_reference']['reference_id']='R999'
    with pytest.raises(PolicyError,match='absent'):
        validate_plan(bad,case)
    case['context']['rejected_references']=[{'camera':'A','reference_id':'R001'}]
    with pytest.raises(PolicyError,match='rejected'):
        validate_plan(plan,case)


def test_multistep_coverage_counts_actual_separate_edit_responses(tmp_path):
    from cloth_agent.harness.planning_probe import operation_trace,processing_coverage
    rows=[]
    for mid,tid,name in [('m1','t1','crop_image'),('m1','t2','crop_image'),('m2','t3','resize_image')]:
        rows.append({'type':'assistant','message':{'id':mid,'content':[{'id':tid,'type':'tool_use',
                     'name':'mcp__cloth_image__'+name,'input':{'image_id':'image_0'}}]}})
    (tmp_path/'claude_stdout.txt').write_text('\n'.join(json.dumps(e) for e in [*rows,rows[-1]]))
    calls=operation_trace(tmp_path)
    assert len(calls)==3
    assert [c['response_group'] for c in calls]==[1,1,2]
    arms={'original':{'status':'COMPLETED','image_edit_calls':3,'image_edit_response_groups':2},
          'skill_then_reasoning':{'status':'COMPLETED','host_image_ops':4}}
    assert processing_coverage(arms)['qualified_multi_step_comparison'] is True
    arms['original']['image_edit_response_groups']=1
    assert processing_coverage(arms)['qualified_multi_step_comparison'] is False
