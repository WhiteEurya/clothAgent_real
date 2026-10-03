"""Synthetic fixtures and mocked Claude only; not real learning/physical evidence."""
import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from cloth_agent.harness.candidate_patch import (baseline_bundle, materialize_candidate, registry_from_bundle,
    run_gates, verify_candidate, _request, _fixtures)
from cloth_agent.harness.common import digest, read_json, write_json
from cloth_agent.harness.executors.observation import RegisteredObservationHost
from cloth_agent.harness.executors.restricted import RestrictedProgram
from cloth_agent.harness.information_flow import global_schema
from cloth_agent.harness.patch_evolution import evolve, evaluate_promotion, main
from cloth_agent.harness.patch_lifecycle import freeze_reviewed, review_outcome
from cloth_agent.harness.policy import PolicyError, validate_schema
from cloth_agent.harness.skills import ObservationSkill, builtin_registry


@pytest.fixture
def scene(tmp_path):
    obs, images = _fixtures(tmp_path/'inputs', [100,80])
    return {'schema_version':1,'observation_id':'synthetic','fold_goal':'Fold the current visible flap',
        'images':[{**i,'path':str(p),'status':'AVAILABLE'} for i,p in zip(obs['images'],images)],
        'candidate_registry':{'observation_id':'synthetic','binding':'RAW_RGB_HASH_VERIFIED',
            'raw_size':[100,80],'to_raw':[1,0,0,0,1,0],
            'candidates':[{'camera':'A','candidate_id':'R001','pixel_xy':[20,30],'raw_pixel_xy':[20,30]}]}}


def plan():
    return {'observation_id':'synthetic','status':'READY','concepts':[],
        'action':{'selected_reference':{'camera':'A','reference_id':'R001','reason':'SYNTHETIC_PRIOR_ACTION_SECRET'},
                  'target':{'pixel_xy':[50,30],'relation':'toward','anchor_pixel_xy':[60,40],'reason':'SYNTHETIC_TARGET_SECRET'}},
        'evidence_summary':'SYNTHETIC_PRIOR_REASON_SECRET','missing_information':'','residual_uncertainty':'Physical checks pending'}


def observe(requests=None):
    return {'information':[{'id':'geometry','need':'Resolve boundary','status':'UNKNOWN' if requests else 'KNOWN',
        'finding':'' if requests else 'Synthetic visible boundary','missing_information':'Boundary detail' if requests else '',
        'source_image_ids':['image_0']}], 'observation_requests': requests or []}


class Model:
    configuration={'actual_measurement':False,'backend':'MOCK_TEST_ONLY'}
    def __init__(self,*outputs): self.outputs=list(outputs);self.calls=[];self.requests=[]
    def invoke(self,**kwargs):
        self.calls.append({'backend_invoked':True,'response_received':True,'tool_round_trips':0})
        self.requests.append(kwargs)
        value=self.outputs.pop(0)
        if isinstance(value,Exception): raise value
        return value(kwargs) if callable(value) else copy.deepcopy(value)


def proposal(level='SKILL_CODE',target='overlay_occlusion'):
    return {'level':level,'target':target,'problem':'Synthetic duplicated clean crop',
            'proposed_change':'Reuse identical clean crop and prepare only missing paired overlay',
            'expected_effect':'Potentially fewer host edits; untested',
            'must_preserve':['Aligned pair','Provenance','No robot'],'evidence_rollouts':['baseline_r00'],
            'unverified':'Semantic sufficiency and speed are not established'}


def code_implementation(new=False):
    base=baseline_bundle();registry=registry_from_bundle(base)
    skill=registry.get('overlay_occlusion')
    spec=copy.deepcopy(skill.specification)
    spec.update(id='paired_roi_reuse' if new else 'overlay_occlusion',version=1 if new else 2)
    source=skill.source.replace('False','True')
    config=copy.deepcopy(base['config'])
    if new: config['enabled_skills'].append(spec['id'])
    cases=[]
    for roi in ([.1,.2,.8,.9],[.2,.1,.7,.8]):
        request=_request(spec['id'],roi=roi)
        expected={'roles':['clean','overlay'],'operations':[{'op':'crop','roi':roi}],'reuse_existing':True}
        cases.append({'request':request,'source_size':[40,60],'expected_recipe':expected})
    return {'config':config,'skill':{'specification':spec,'source':source,'tests':cases}}


def prompt_implementation():
    config=copy.deepcopy(baseline_bundle()['config'])
    config['observation_instruction']+=' Stop acquiring views when sufficient.'
    return {'config':config,'skill':None}


def build_candidate(tmp_path,*,new=False):
    path=tmp_path/'candidate';path.mkdir()
    materialize_candidate(path,proposal('NEW_TOOL','paired_roi_reuse') if new else proposal(),code_implementation(new),baseline_bundle())
    return path


def test_registry_versions_and_builtin_compatibility():
    registry=builtin_registry();original=registry.snapshot()
    skill=registry.get('orientation')
    with pytest.raises(PolicyError,match='overwrite'): registry.register(skill)
    skill.specification['version']=2
    registry.register(skill)
    assert registry.get('orientation').key=='orientation@v2'
    assert registry.get('orientation@v1').content_hash==original['orientation']['hash']
    copy_skill=registry.get('orientation');copy_skill.specification['method']='tamper returned copy'
    assert registry.get('orientation').specification['method']!='tamper returned copy'


@pytest.mark.parametrize('source',[
    'import os\ndef prepare(request, source, available): return {}',
    'def prepare(request, source, available): return source.__class__',
    'def prepare(request, source, available): return open("robot.py")',
    'def prepare(request, source, available):\n while True: pass',
    'def prepare(request, source, available): return [x for x in source]',
    'def prepare(request, source, available): return 2 ** 999',
    'def prepare(request, source, available): return __import__("socket")',
    'def prepare(request, source, available):\n source["size"] = [1, 1]\n return source',
    'def prepare(request, source, available): return eval("1")',
])
def test_forbidden_code_is_rejected_without_execution(source):
    with pytest.raises(PolicyError): RestrictedProgram(source)


def test_sequence_allocation_and_nonfinite_numbers_are_bounded():
    with pytest.raises(PolicyError): RestrictedProgram('def prepare(request, source, available): return "x" * 1000000').run({}, {}, {})
    with pytest.raises(PolicyError): RestrictedProgram('def prepare(request, source, available): return 1e999')


def test_aliased_containers_have_an_aggregate_validation_budget():
    source = 'def prepare(request, source, available):\n    a = [0]\n'
    source += '    a = [a, a, a, a, a, a, a, a]\n' * 6
    source += '    return a\n'
    with pytest.raises(PolicyError, match='aggregate value budget'):
        RestrictedProgram(source).run({}, {}, {})


def test_generated_cases_can_name_their_own_information_gap(tmp_path):
    implementation = code_implementation()
    for case in implementation['skill']['tests']:
        case['request']['gap_id'] = 'synthetic_boundary_detail'
    path = tmp_path/'candidate'
    path.mkdir()
    materialize_candidate(path, proposal(), implementation, baseline_bundle())
    assert run_gates(path)['status'] == 'PASSED'


def test_new_skill_real_code_registered_after_gates(tmp_path):
    path=build_candidate(tmp_path,new=True)
    gates=run_gates(path)
    assert gates['status']=='PASSED',gates
    bundle=verify_candidate(path);registry=registry_from_bundle(bundle)
    assert registry.get('paired_roi_reuse').key=='paired_roi_reuse@v1'
    assert 'paired_roi_reuse' in global_schema(registry)['properties']['observation_requests']['items']['properties']['skill_id']['enum']
    assert (path/'patch.diff').is_file()
    assert (path/'cloth_agent/harness/skills/paired_roi_reuse/implementation.py').is_file()
    assert any(c['name']=='generated_unit_tests' for c in gates['checks'])


def test_partial_pair_reuses_clean_without_duplicate_delivery(tmp_path):
    path=build_candidate(tmp_path,new=True);assert run_gates(path)['status']=='PASSED'
    registry=registry_from_bundle(verify_candidate(path))
    obs,images=_fixtures(tmp_path/'reuse_inputs',[40,60])
    host=RegisteredObservationHost(obs,images,{'skills':registry.catalog()},tmp_path/'host',3,registry=registry)
    info=[{'id':'geometry','status':'UNKNOWN'}]
    assert host.execute([_request('local_boundary',roi=[.1,.2,.8,.9])],info) is None
    assert host.ops==1 and len(host.paths)==3
    assert host.execute([_request('paired_roi_reuse',roi=[.1,.2,.8,.9])],info) is None
    assert host.ops==2 and len(host.paths)==4
    assert host.history[-1]['reused_image_ids']==['image_2']
    assert host.catalog[2]['to_original']==host.catalog[3]['to_original']
    assert host.execute([_request('paired_roi_reuse',roi=[.1,.2,.8,.9])],info)=='REPEATED_OBSERVATION'
    assert host.ops==2


def test_generated_test_failure_blocks_gate(tmp_path):
    implementation=code_implementation()
    implementation['skill']['tests'][0]['expected_recipe']['reuse_existing']=False
    path=tmp_path/'candidate';path.mkdir()
    materialize_candidate(path,proposal(),implementation,baseline_bundle())
    assert run_gates(path)['status']=='REJECTED'
    with pytest.raises(PolicyError,match='passing tests'): verify_candidate(path)


def test_tamper_after_gate_cannot_replay(tmp_path):
    path=build_candidate(tmp_path);assert run_gates(path)['status']=='PASSED'
    source=path/'cloth_agent/harness/skills/overlay_occlusion/implementation.py'
    source.write_text(source.read_text()+'\n# changed after tests\n')
    with pytest.raises(PolicyError,match='changed'): verify_candidate(path)


def test_out_of_scope_paths_or_extra_fields_rejected(tmp_path):
    implementation=code_implementation();implementation['skill']['files']={'cloth_agent/robot_api.py':'danger'}
    with pytest.raises(PolicyError): materialize_candidate(tmp_path,proposal(),implementation,baseline_bundle())
    assert not (tmp_path/'bundle.json').exists()


def test_code_patch_cannot_sneak_in_planner_change(tmp_path):
    implementation=code_implementation();implementation['config']['observation_instruction']='Copy prior answer'
    with pytest.raises(PolicyError,match='preserve baseline'): materialize_candidate(tmp_path,proposal(),implementation,baseline_bundle())


def test_source_version_collision_rejected(tmp_path):
    implementation=code_implementation();implementation['skill']['specification']['version']=1
    with pytest.raises(PolicyError,match='next version'): materialize_candidate(tmp_path,proposal(),implementation,baseline_bundle())


def test_full_loop_runtime_proposes_implements_tests_and_replays_without_answers(scene,tmp_path,monkeypatch):
    def forbidden(*args,**kwargs): pytest.fail('Evolution touched robot')
    monkeypatch.setattr('cloth_agent.robot_api.RobotAPI.move',forbidden)
    monkeypatch.setattr('cloth_agent.robot_api.RobotAPI.home',forbidden)
    requests = [_request('local_boundary', roi=[.1,.2,.8,.9]),
                _request('overlay_occlusion', roi=[.1,.2,.8,.9])]
    selected = plan()
    selected['concepts'] = [{'name':'geometry', 'finding':'Synthetic paired boundary is visible',
                             'source_image_ids':['image_2','image_3']}]
    new_requests = [requests[0], {**requests[1], 'skill_id':'paired_roi_reuse'}]
    model=Model(observe(),plan(),proposal(),code_implementation(),observe(requests),selected,
                proposal('NEW_TOOL','paired_roi_reuse'),code_implementation(True),observe(new_requests),selected)
    report=evolve(scene,tmp_path/'experiment',model,patches=2,repeats=1)
    assert report['status']=='NO_PROMOTION',report.get('error')
    assert len(report['rollouts'])==3
    assert all(r['status']=='READY' for r in report['rollouts'])
    assert [r['metrics']['host_image_ops'] for r in report['rollouts']] == [0,2,2]
    assert report['rollouts'][2]['observation_trace'][-1]['skill_version'] == 'paired_roi_reuse@v1'
    assert report['rollouts'][2]['observation_trace'][-1]['reused_image_ids'] == ['image_2']
    assert len(report['candidates'])==2
    assert all(c['gates']['status']=='PASSED' for c in report['candidates'])
    assert len({r['root_evidence_hash'] for r in report['rollouts']})==1
    replay=[r for r in model.requests if r['stage'] in ('patch_observe','reasoning_rollout')]
    assert len(replay)==6
    assert all('SECRET' not in r['prompt'] for r in replay)
    assert 'SECRET' in model.requests[2]['prompt']  # diagnosis can inspect public baseline
    assert not (tmp_path/'experiment/working').exists() # never promote mock measurements
    assert (tmp_path/'experiment/index.html').is_file()
    assert (tmp_path/'experiment/candidates/patch_00/evaluation.json').is_file()


def test_gate_failure_prevents_any_candidate_planning(scene,tmp_path):
    implementation=code_implementation();implementation['skill']['tests'][1]['expected_recipe']['roles']=['clean']
    model=Model(observe(),plan(),proposal(),implementation)
    report=evolve(scene,tmp_path/'exp',model,patches=1,repeats=1)
    assert len(report['rollouts'])==1
    assert report['candidates'][0]['status']=='REJECTED'
    assert len(model.requests)==4


def test_invalid_baseline_stops_before_diagnosis(scene,tmp_path):
    bad=plan();bad['action']['selected_reference']['reference_id']='R999'
    model=Model(observe(),bad)
    report=evolve(scene,tmp_path/'exp',model,patches=2,repeats=1)
    assert report['status']=='BLOCKED_BASELINE'
    assert len(model.requests)==2 and not report['candidates']


def test_budget_stops_before_candidate_implementation(scene,tmp_path):
    model=Model(observe(),plan(),proposal())
    report=evolve(scene,tmp_path/'exp',model,patches=5,repeats=1,max_calls=3)
    assert len(model.requests)==3
    assert report['candidates'][0]['status']=='REJECTED'
    assert report['candidates'][0]['budget_exhausted']


def test_gap_is_not_resolved_by_host_execution(scene,tmp_path):
    model=Model(observe([_request('orientation',angle=90)]),plan())
    report=evolve(scene,tmp_path/'exp',model,patches=1,repeats=1)
    assert report['rollouts'][0]['status']=='NEEDS_LEARNING'
    assert report['rollouts'][0]['reason']=='GAP_NOT_EXPLICITLY_INTERPRETED'
    assert len(model.requests)==2


def test_fresh_views_and_gap_state_reach_planner(scene,tmp_path):
    result=plan();result['concepts']=[{'name':'geometry','finding':'Synthetic visible boundary after rotation','source_image_ids':['image_2']}]
    model=Model(observe([_request('orientation',angle=90)]),result,proposal('PROMPT','observation'),prompt_implementation(),observe(),plan())
    report=evolve(scene,tmp_path/'exp',model,patches=1,repeats=1)
    assert report['rollouts'][0]['status']=='READY'
    assert report['rollouts'][0]['metrics']['host_image_ops']==2
    assert len(model.requests[1]['images'])==4
    assert 'observation_requirements' in model.requests[1]['prompt']
    assert report['rollouts'][0]['derived_evidence_hash']!=report['rollouts'][0]['root_evidence_hash']


def _promotion_rows():
    return [{'harness_hash':name,'root_evidence_hash':'shared_input','actual_measurement':True,'status':'READY',
             'action':{'candidate_legal':True},'metrics':{'elapsed_s':seconds,'response_count':2}}
            for name,seconds in [('base',10),('candidate',7),('other',8)]]


def test_promotion_requires_real_improvement_tests_and_stable_baseline():
    consensus={'status':'SELECTED','selected':{'harness_hash':'candidate'},'clusters':[{'harnesses':['base','candidate','other']}]}
    rows=_promotion_rows()
    result=evaluate_promotion('base','candidate',rows,consensus,gates_passed=True)
    assert result['status']=='WORKING'
    assert result['physical_status']=='PENDING' and result['frozen_eligible'] is False
    rows[1]['metrics']['elapsed_s']=9.5
    assert evaluate_promotion('base','candidate',rows,consensus,gates_passed=True)['status']=='REJECTED'
    rows[1]['metrics']['elapsed_s']=7
    assert evaluate_promotion('base','candidate',rows,consensus,gates_passed=False)['status']=='REJECTED'
    consensus['clusters'][0]['harnesses'].remove('base')
    assert evaluate_promotion('base','candidate',rows,consensus,gates_passed=True)['status']=='REJECTED'


@pytest.mark.parametrize('fault',['mock','different_input','not_grounded','unknown','not_cheapest'])
def test_promotion_rejection_reasons(fault):
    rows=_promotion_rows();consensus={'status':'SELECTED','selected':{'harness_hash':'candidate'},'clusters':[{'harnesses':['base','candidate','other']}]}
    if fault=='mock': rows[1]['actual_measurement']=False
    if fault=='different_input': rows[1]['root_evidence_hash']='different'
    if fault=='not_grounded': rows[1]['action']['candidate_legal']=False
    if fault=='unknown': rows[1]['status']='NEEDS_LEARNING'
    if fault=='not_cheapest': consensus['selected']['harness_hash']='other'
    assert evaluate_promotion('base','candidate',rows,consensus,gates_passed=True)['status']=='REJECTED'


def test_prepare_only_and_existing_directory(scene,tmp_path):
    source=tmp_path/'evidence.json';write_json(source,scene)
    args=['--evidence',str(source),'--output',str(tmp_path/'prepared'),'--prepare-only']
    assert main(args)==0
    assert read_json(tmp_path/'prepared/report.json')['totals']['model_calls']==0
    assert main(args)==2
    assert read_json(tmp_path/'prepared/report.json')['status']=='PREPARED'


def test_missing_source_writes_blocker(tmp_path):
    assert main(['--evidence',str(tmp_path/'missing.json'),'--output',str(tmp_path/'blocked')])==2
    assert read_json(tmp_path/'blocked/blocked.json')['model_calls']==0


def test_comment_only_code_patch_is_not_a_new_program(tmp_path):
    implementation=code_implementation()
    implementation['skill']['source']=registry_from_bundle(baseline_bundle()).get('overlay_occlusion').source+'\n# cosmetic only\n'
    with pytest.raises(PolicyError,match='No executable change'):
        materialize_candidate(tmp_path,proposal(),implementation,baseline_bundle())


def test_python_source_cannot_declare_nested_functions():
    with pytest.raises(PolicyError,match='Nested'):
        RestrictedProgram('def prepare(request, source, available):\n def other(): return 1\n return {}')


def test_patch_then_new_observation_does_not_reuse_old_views(tmp_path):
    path=build_candidate(tmp_path,new=True);assert run_gates(path)['status']=='PASSED'
    registry=registry_from_bundle(verify_candidate(path))
    obs,images=_fixtures(tmp_path/'roots',[40,60])
    artifact={'skills':registry.catalog()};info=[{'id':'geometry','status':'UNKNOWN'}]
    a=RegisteredObservationHost(obs,images,artifact,tmp_path/'a',12,registry=registry)
    b=RegisteredObservationHost(obs,images,artifact,tmp_path/'b',12,registry=registry)
    req=_request('paired_roi_reuse',roi=[.1,.2,.8,.9])
    assert a.execute([req],info) is None and b.execute([req],info) is None
    assert a.ops==b.ops==2
    assert not b.history[0]['reused_image_ids']


def test_pair_atomic_budget_with_partial_cached_view(tmp_path):
    path=build_candidate(tmp_path,new=True);assert run_gates(path)['status']=='PASSED'
    registry=registry_from_bundle(verify_candidate(path));obs,images=_fixtures(tmp_path/'roots',[40,60])
    host=RegisteredObservationHost(obs,images,{'skills':registry.catalog()},tmp_path/'host',1,registry=registry)
    info=[{'id':'geometry','status':'UNKNOWN'}]
    host.execute([_request('local_boundary',roi=[.1,.2,.8,.9])],info)
    assert host.execute([_request('paired_roi_reuse',roi=[.1,.2,.8,.9])],info)=='OBSERVATION_BUDGET_EXHAUSTED'
    assert host.ops==1 and len(host.paths)==3


def test_runtime_claude_adapter_is_used_for_diagnosis_and_implementation(scene,tmp_path,monkeypatch):
    from cloth_agent.harness.model import RuntimeClaude
    from cloth_agent.planner_backend import BackendResult
    outputs=[observe(),plan(),proposal(),code_implementation(),observe(),plan()]
    stages=[]
    def invoke(self,**kwargs):
        stages.append(kwargs['usage_stage'])
        message=json.loads(kwargs['input_data'])['message']
        assert len([b for b in message['content'] if b['type']=='image'])==2
        result={'type':'result','structured_output':outputs.pop(0)}
        return BackendResult(json.dumps(result),'',0,tuple(kwargs['command']))
    monkeypatch.setattr('cloth_agent.harness.model.shutil.which',lambda _: '/synthetic/claude')
    monkeypatch.setattr('cloth_agent.planner_backend.LocalClaudeBackend.invoke',invoke)
    report=evolve(scene,tmp_path/'exp',RuntimeClaude(),patches=1,repeats=1)
    assert report['status']=='NO_PROMOTION'
    assert stages==['patch_observe','reasoning_rollout','patch_diagnose','patch_implement','patch_observe','reasoning_rollout']
    assert (tmp_path/'exp/candidates/patch_00/implement/input.jsonl').exists()


def _working_fixture(tmp_path):
    path=build_candidate(tmp_path);assert run_gates(path)['status']=='PASSED'
    bundle=verify_candidate(path);key=digest(bundle)
    rows=_promotion_rows()
    for row in rows:
        if row['harness_hash']=='candidate': row['harness_hash']=key
    consensus={'status':'SELECTED','selected':{'harness_hash':key},'clusters':[{'harnesses':['base',key,'other']}]}
    evaluation=evaluate_promotion('base',key,rows,consensus,gates_passed=True)
    working=tmp_path/'working'
    write_json(working/'bundle.json',bundle)
    write_json(working/'evidence.json',{'bundle_hash':key,'status':'WORKING_OFFLINE_ONLY',
                                      'candidate_directory':str(path),'evaluation':evaluation})
    return working,key


def _review_fixture(tmp_path,key,index=0,*,outcome='SUCCESS',patch_attribution='CLEARED',image_hash=None):
    observation=f'synthetic_state_{index}';evidence_hash=digest(observation)
    action={'synthetic_action':index}
    physical={'bundle_hash':key,'observation_id':observation,'root_evidence_hash':evidence_hash,
              'action_hash':digest(action),'outcome':outcome,'synthetic_fixture':True}
    report={'rollouts':[{'harness_hash':key,'root_evidence_hash':evidence_hash,'observation_id':observation,
                         'observation_rgb_sha256':image_hash or digest(['synthetic_rgb',index]),
                         'status':'READY','actual_measurement':True,'action':action}]}
    # These flags imitate trusted external records only for testing admission;
    # fixture files are never published as real execution results.
    write_json(tmp_path/f'physical_{index}.json',physical)
    write_json(tmp_path/f'replay_{index}.json',report)
    return {'schema_version':1,'bundle_hash':key,'observation_id':observation,'root_evidence_hash':evidence_hash,
        'action_hash':digest(action),'physical_record_path':str(tmp_path/f'physical_{index}.json'),
        'physical_record_hash':digest(physical),'outcome':outcome,'reviewed_by':'SYNTHETIC_TEST_REVIEWER',
        'attribution':{'task_decision':'CLEARED','grasp_depth':'CAUSE' if outcome=='FAILURE' else 'CLEARED',
            'target':'CLEARED','observation_patch':patch_attribution,'rationale':'Synthetic external review only'},
        'replay_report_path':str(tmp_path/f'replay_{index}.json'),'replay_report_hash':digest(report)}


def test_physical_outcome_with_unresolved_attribution_cannot_freeze(tmp_path):
    working,key=_working_fixture(tmp_path)
    review=_review_fixture(tmp_path,key,outcome='FAILURE',patch_attribution='UNKNOWN')
    row=review_outcome(working,review,tmp_path/'ledger/one.json')
    assert row['status']=='HOLD'
    with pytest.raises(PolicyError,match='Unresolved'):
        freeze_reviewed(working,tmp_path/'ledger',tmp_path/'frozen')
    assert not (tmp_path/'frozen').exists()


def test_three_distinct_reviewed_states_can_export_without_activating(tmp_path):
    working,key=_working_fixture(tmp_path)
    for index in range(3):
        review=_review_fixture(tmp_path,key,index)
        assert review_outcome(working,review,tmp_path/'ledger'/f'{index}.json')['status']=='REVIEWED_CLEAR'
    path=freeze_reviewed(working,tmp_path/'ledger',tmp_path/'frozen')
    artifact=read_json(path/'evidence.json')
    assert artifact['status']=='FROZEN_OFFLINE_LIBRARY'
    assert artifact['distinct_states']==3
    assert artifact['auto_enabled'] is False and artifact['robot_executable'] is False
    assert read_json(working/'evidence.json')['status']=='WORKING_OFFLINE_ONLY'


def test_repeated_image_with_different_ids_is_not_cross_state_evidence(tmp_path):
    working,key=_working_fixture(tmp_path)
    for index in range(3):
        review_outcome(working,_review_fixture(tmp_path,key,index,image_hash='same_actual_pixels'),tmp_path/'ledger'/f'{index}.json')
    with pytest.raises(PolicyError,match='distinct'):
        freeze_reviewed(working,tmp_path/'ledger',tmp_path/'frozen')


def test_physical_evidence_hash_and_action_binding_are_verified(tmp_path):
    working,key=_working_fixture(tmp_path)
    review=_review_fixture(tmp_path,key)
    review['action_hash']='not_the_replayed_action'
    with pytest.raises(PolicyError,match='bound'):
        review_outcome(working,review,tmp_path/'ledger/one.json')
    review=_review_fixture(tmp_path,key)
    write_json(review['physical_record_path'],{'changed':True})
    with pytest.raises(PolicyError,match='changed'):
        review_outcome(working,review,tmp_path/'ledger/two.json')
