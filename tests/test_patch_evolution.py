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


def test_code_patch_can_include_matching_planner_instruction(tmp_path):
    implementation=code_implementation()
    implementation['config']['observation_instruction']='Reuse the paired crop for all relevant current visual gaps.'
    bundle=materialize_candidate(tmp_path,proposal(),implementation,baseline_bundle())
    assert bundle['config']==implementation['config']
    assert run_gates(tmp_path)['status']=='PASSED'


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


def test_blocking_gap_still_stops_when_planner_cannot_resolve_it(scene,tmp_path):
    blocked=plan();blocked.update(status='NEEDS_LEARNING',action=None,missing_information='Cuff boundary remains occluded')
    model=Model(observe([_request('orientation',angle=90)]),blocked)
    report=evolve(scene,tmp_path/'exp',model,patches=1,repeats=1)
    assert report['status']=='BLOCKED_BASELINE'
    assert report['rollouts'][0]['status']=='NEEDS_LEARNING'
    assert report['rollouts'][0]['reason']=='Cuff boundary remains occluded'
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


def test_semantic_timing_context_closes_on_experiment_failure(scene, tmp_path, monkeypatch):
    from cloth_agent.harness import patch_evolution
    from cloth_agent.pipeline_timing import active_timing
    source = tmp_path/'evidence.json'
    write_json(source, scene)
    def fail(*args, **kwargs):
        assert active_timing().semantic_phases
        with active_timing().span('test.failure'):
            raise RuntimeError('test interrupted run')
    monkeypatch.setattr(patch_evolution, 'evolve', fail)
    args = ['--evidence', str(source), '--output', str(tmp_path/'experiment'),
            '--backend', 'remote', '--semantic-timing']
    assert main(args) == 2
    assert active_timing() is None
    report = read_json(tmp_path/'experiment_timing/summary.json')
    assert report['status'] == 'ERROR'
    assert report['spans'][0]['name'] == 'test.failure'
    assert (tmp_path/'experiment_timing/semantic_timeline.csv').exists()
    with pytest.raises(SystemExit):
        main(['--evidence', str(source), '--output', str(tmp_path/'local'), '--semantic-timing'])


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
    transport_proposal=proposal(); transport_proposal['must_preserve']='; '.join(transport_proposal['must_preserve'])
    outputs=[observe(),plan(),transport_proposal,code_implementation(),observe(),plan()]
    stages=[]
    def invoke(self,**kwargs):
        stages.append(kwargs['usage_stage'])
        message=json.loads(kwargs['input_data'])['message']
        assert len([b for b in message['content'] if b['type']=='image'])==2
        result={'type':'result','structured_output':outputs.pop(0)}
        return BackendResult(json.dumps(result),'',0,tuple(kwargs['command']))
    monkeypatch.setattr('cloth_agent.harness.model.shutil.which',lambda _: '/synthetic/claude')
    monkeypatch.setattr('cloth_agent.planner_backend.LocalClaudeBackend.invoke',invoke)
    class MockedRuntime(RuntimeClaude):
        @property
        def configuration(self):
            return {**super().configuration, 'actual_measurement': False}
    report=evolve(scene,tmp_path/'exp',MockedRuntime(),patches=1,repeats=1)
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


def test_one_patch_can_form_two_version_consensus_and_gets_time_objective(scene,tmp_path):
    model=Model(observe(),plan(),proposal(),code_implementation(),observe(),plan())
    report=evolve(scene,tmp_path/'one',model,patches=1,repeats=1)
    assert report['settings']['consensus']['min_harnesses']==2
    assert report['consensus']['status']=='SELECTED'
    assert report['consensus']['selected']['votes']==2
    assert len(model.requests)==6
    diagnosis=model.requests[2]['prompt']
    assert 'REDUCE TOTAL' in diagnosis
    assert 'evaluation_policy' in diagnosis and 'min_latency_reduction' in diagnosis
    assert 'optimization_objective' in model.requests[3]['prompt']
    assert not (tmp_path/'one/working').exists()  # mocks still cannot establish benefit


@pytest.mark.parametrize('patches,explicit,expected',[(1,None,2),(3,None,3),(1,3,3),(3,2,2)])
def test_cli_consensus_default_and_override(scene,tmp_path,patches,explicit,expected):
    source=tmp_path/'source.json';write_json(source,scene)
    args=['--evidence',str(source),'--output',str(tmp_path/'prepared'),
          '--patches',str(patches),'--repeats','1','--prepare-only']
    if explicit is not None: args+=['--min-harnesses',str(explicit)]
    assert main(args)==0
    report=read_json(tmp_path/'prepared/report.json')
    assert report['settings']['consensus']['min_harnesses']==expected
    assert report['totals']['model_calls']==0


def test_physical_unknowns_and_free_concept_names_do_not_block_observation(scene,tmp_path):
    request=_request('local_boundary',roi=[.1,.2,.8,.9])
    observation=observe([request])
    observation['information'].append({'id':'physical','need':'Physical hold verification','status':'UNKNOWN',
        'finding':'RGB cannot establish friction','missing_information':'Later physical feedback',
        'source_image_ids':['image_0']})
    decision=plan();decision['concepts']=[{'name':'visible_cuff',
        'finding':'Synthetic cuff boundary is visible','source_image_ids':['image_2']}]
    model=Model(observation,decision,proposal('PROMPT','observation'),prompt_implementation(),observe(),plan())
    report=evolve(scene,tmp_path/'exp',model,patches=1,repeats=1)
    baseline=report['rollouts'][0]
    assert baseline['status']=='READY'
    assert baseline['metrics']['host_image_ops']==1
    assert 'Later physical feedback' in model.requests[1]['prompt']
    assert 'residual_uncertainty' in model.requests[1]['prompt']


def test_patch_reuses_one_crop_for_four_information_tasks(scene,tmp_path):
    requests=[{**_request('local_boundary',roi=[.1,.2,.8,.9]),'gap_id':f'gap{i}'} for i in range(4)]
    observation=observe(requests)
    observation['information']=[{**observation['information'][0],'id':r['gap_id']} for r in requests]
    model=Model(observation,plan(),proposal('PROMPT','observation'),prompt_implementation(),observe(),plan())
    report=evolve(scene,tmp_path/'exp',model,patches=1,repeats=1)
    baseline=report['rollouts'][0]
    assert baseline['status']=='READY'
    assert baseline['metrics']['host_image_ops']==1
    assert len(baseline['observation_trace'])==4
    assert baseline['observation_trace'][-1]['reused_image_ids']==['image_2']
    assert len(model.requests[1]['images'])==3


@pytest.mark.parametrize('text',[
    'Nothing further needed at this stage; finer edge placement is covered by I2.',
    'None for registration itself; sleeve membership is a boundary question.',
])
def test_known_explanatory_missing_text_reaches_planner_unchanged(scene,tmp_path,text):
    observation=observe();observation['information'][0]['missing_information']=text
    original=copy.deepcopy(observation)
    model=Model(observation,plan(),proposal('PROMPT','observation'),prompt_implementation(),observe(),plan())
    report=evolve(scene,tmp_path/'exp',model,patches=1,repeats=1)
    assert report['rollouts'][0]['status']=='READY'
    assert len(report['candidates'])==1
    assert text in model.requests[1]['prompt']
    assert 'KNOWN_WITH_MISSING_INFORMATION' in model.requests[1]['prompt']
    saved=read_json(tmp_path/'exp/replays/baseline_r00/observation.json')
    assert saved==original
    audit=read_json(tmp_path/'exp/replays/baseline_r00/information_validation.json')
    assert audit['status']=='ACCEPTED_WITH_WARNINGS'
    assert audit['warnings'][0]['missing_information']==text


def test_real_known_gap_not_silently_cleared(scene,tmp_path):
    observation=observe();observation['information'][0]['missing_information']='Target cuff is hidden'
    blocked=plan();blocked.update(status='NEEDS_LEARNING',action=None,missing_information='Target cuff is hidden')
    model=Model(observation,blocked)
    report=evolve(scene,tmp_path/'exp',model,patches=1,repeats=1)
    assert 'Target cuff is hidden' in model.requests[1]['prompt']
    assert report['status']=='BLOCKED_BASELINE'
    assert report['rollouts'][0]['reason']=='Target cuff is hidden'


def test_prose_citations_normalized_without_extra_call(scene,tmp_path):
    p=proposal('PROMPT','observation')
    p['evidence_rollouts']=['baseline_r00 observation_trace: crop used',
                            'image_2 is only supplemental image evidence',
                            'baseline_r00 planning metrics: recorded time']
    original=copy.deepcopy(p)
    model=Model(observe(),plan(),p,prompt_implementation(),observe(),plan())
    report=evolve(scene,tmp_path/'exp',model,patches=1,repeats=1)
    assert len(model.requests)==6
    assert len(report['rollouts'])==2
    directory=tmp_path/'exp/candidates/patch_00'
    assert read_json(directory/'proposal.json')==original
    normalized=read_json(directory/'proposal_normalized.json')
    assert normalized['evidence_rollouts']==['baseline_r00']
    assert read_json(directory/'citation_audit.json')['entries'][1]['status']=='UNRESOLVED'
    assert 'supplemental image evidence' in model.requests[3]['prompt']
    assert model.requests[2]['schema']['properties']['evidence_rollouts']['items']['enum']==['baseline_r00']


@pytest.mark.parametrize('fixed',[[],['baseline_r00']])
def test_missing_citation_gets_one_field_only_repair(scene,tmp_path,fixed):
    p=proposal('PROMPT','observation');p['evidence_rollouts']=['image_2']
    responses=[observe(),plan(),p,{'evidence_rollouts':fixed}]
    if fixed: responses += [prompt_implementation(),observe(),plan()]
    model=Model(*responses)
    report=evolve(scene,tmp_path/'exp',model,patches=1,repeats=1)
    assert sum(r['stage']=='patch_citation_repair' for r in model.requests)==1
    assert sum(r['stage']=='patch_diagnose' for r in model.requests)==1
    if fixed:
        normalized=read_json(tmp_path/'exp/candidates/patch_00/proposal_normalized.json')
        assert {k:v for k,v in normalized.items() if k!='evidence_rollouts'}=={k:v for k,v in p.items() if k!='evidence_rollouts'}
        assert len(report['rollouts'])==2
    else:
        assert len(model.requests)==4
        assert report['candidates'][0]['status']=='REJECTED'


def test_citation_boundaries_do_not_guess_ids():
    from cloth_agent.harness.candidate_patch import normalize_citations
    p=proposal()
    p['evidence_rollouts']=['baseline_r000','prefixbaseline_r00','baseline_r00-other',
                            'baseline_r00 and baseline_r01 both mentioned']
    normalized,audit=normalize_citations(p,['baseline_r00','baseline_r01'])
    assert normalized['evidence_rollouts']==[]
    assert audit['status']=='NEEDS_CITATION_REPAIR'


def test_known_rotation_does_not_block_valid_crop_batch(scene,tmp_path):
    requests=[_request('local_boundary',roi=[.1,.2,.8,.9]),
              {**_request('orientation',angle=270),'gap_id':'camera_rotation'}]
    observation=observe(requests)
    observation['information'].append({'id':'camera_rotation','need':'Camera orientation',
        'status':'KNOWN','finding':'The transform gives the rotation angle',
        'missing_information':'','source_image_ids':['image_0']})
    original=copy.deepcopy(observation)
    model=Model(observation,plan(),proposal('PROMPT','observation'),prompt_implementation(),observe(),plan())
    report=evolve(scene,tmp_path/'exp',model,patches=1,repeats=1)
    baseline=report['rollouts'][0]
    assert baseline['status']=='READY'
    assert baseline['metrics']['host_image_ops']==3
    assert [x['information_status_at_request'] for x in baseline['observation_trace']]==['UNKNOWN','KNOWN']
    assert read_json(tmp_path/'exp/replays/baseline_r00/observation.json')==original
    assert len(model.requests[1]['images'])==5


def test_exclusive_phase_costs_and_actual_implementation_effects(scene, tmp_path):
    from cloth_agent.harness.patch_evolution import implementation_effects
    report = evolve(scene, tmp_path/'timed', Model(observe(),plan(),proposal('PROMPT','observation'),prompt_implementation(),observe(),plan()), patches=1, repeats=1)
    metrics = report['rollouts'][0]['metrics']
    assert sum(metrics['exclusive_phases_s'].values()) == pytest.approx(metrics['elapsed_s'])
    assert metrics['exclusive_phases_s']['observation_call_s'] > 0
    assert metrics['exclusive_phases_s']['planning_s'] > 0
    prompt = implementation_effects(baseline_bundle(), prompt_implementation())
    assert prompt['observation_prompt_changed']
    assert not prompt['skill_code_generated'] and not prompt['host_dispatch_code_changed']
    assert not prompt['stage_topology_changed']
    assert implementation_effects(baseline_bundle(), code_implementation())['skill_code_generated']


def test_failed_replay_has_no_speedup_measurement():
    rows = _promotion_rows(); rows[1]['status'] = 'ERROR'
    consensus = {'status':'SELECTED','selected':{'harness_hash':'candidate'},'clusters':[{'harnesses':['base','candidate']}]}
    result = evaluate_promotion('base','candidate',rows,consensus,gates_passed=True)
    assert result['latency_reduction'] is None
    assert not result['checks']['latency_improved']


def test_budget_defers_only_overflow_and_still_plans(scene, tmp_path):
    requests = [_request('local_boundary',roi=[.1,.1,.4,.4]),
                _request('overlay_occlusion',roi=[.6,.6,.9,.9])]
    model = Model(observe(requests), plan(), proposal('PROMPT','observation'),
                  prompt_implementation(), observe(), plan())
    report = evolve(scene,tmp_path/'partial',model,patches=1,repeats=1,max_ops=1)
    baseline = report['rollouts'][0]
    assert baseline['status'] == 'READY'
    assert baseline['metrics']['host_image_ops'] == 1
    assert len(baseline['deferred_observations']) == 1
    assert baseline['deferred_observations'][0]['delivered_image_ids'] == []
    assert 'DEFERRED' in model.requests[1]['prompt']
    assert 'NOT executed' in model.requests[1]['prompt']


def _cached_outputs():
    return [observe(),plan(),proposal('PROMPT','observation'),prompt_implementation(),observe(),plan()]


def test_experience_mode_reuses_baseline_and_executes_host_code_without_speed_gate(scene,tmp_path,monkeypatch):
    evolve(scene,tmp_path/'old',Model(*_cached_outputs()),patches=1,repeats=1)
    config=copy.deepcopy(baseline_bundle()['config'])
    config['reasoning_harness']['stages']=[
        {'id':'measure','context':[],'allow_ready':False,
         'instruction':'Identify current geometry and output p, a, b measurements. Stop if unsupported.'},
        {'id':'decide','context':['measure'],'allow_ready':True,
         'instruction':'Use the computed reflection with current images to choose a supported visual action.',
         'host_operations':[{'id':'mirror','op':'reflect_point','source_stage':'measure',
                             'bindings':{'point':'p','line_start':'a','line_end':'b'}}]}]
    measurement=plan();measurement.update(status='CONTINUE',action=None,
        measurements={'p':[20,30],'a':[10,10],'b':[10,50]})
    def final(kw):
        assert '"status":"COMPUTED"' in kw['prompt']
        assert '"output":[0.0,30.0]' in kw['prompt']
        answer=plan();answer['action']['target']['pixel_xy']=[0,30]
        return answer
    model=Model(proposal('HARNESS','reasoning'),{'config':config,'skill':None},measurement,final)
    def forbidden(*a,**k):raise AssertionError('Experience mode must not run the speed promotion gate')
    monkeypatch.setattr('cloth_agent.harness.patch_evolution.evaluate_promotion',forbidden)
    result=evolve(scene,tmp_path/'new',model,patches=1,repeats=1,
                  reuse_baseline_run=tmp_path/'old',learning_mode='experience')
    assert result['status']=='EXPERIMENT_COMPLETE',result
    assert result['baseline_cache']['status']=='HIT'
    assert [r['stage'] for r in model.requests]==['patch_diagnose','patch_implement','reasoning_rollout','reasoning_rollout']
    candidate=result['candidates'][0]
    assert candidate['experience_evaluation']['computed_operations']==1
    assert candidate['status']=='TESTED' and result['promotions']==[]
    assert not result['automatic_activation']
    revised=copy.deepcopy(config)
    revised['reasoning_harness']['stages'][0]['instruction']+=' Verify hinge applicability before reporting measurements.'
    resumed_model=Model(proposal('HARNESS','reasoning'),{'config':revised,'skill':None},measurement,final)
    resumed=evolve(scene,tmp_path/'continued',resumed_model,patches=2,repeats=1,
                   reuse_baseline_run=tmp_path/'old',resume_run=tmp_path/'new',learning_mode='experience')
    assert resumed['status']=='EXPERIMENT_COMPLETE',resumed
    assert len(resumed['candidates'])==2
    assert resumed['candidates'][0]['resumed_from']==str((tmp_path/'new').resolve())
    assert resumed['candidates'][1]['experience_evaluation']['computed_operations']==1
    assert resumed_model.requests[0]['stage']=='patch_diagnose'
    assert 'mirror' in resumed_model.requests[0]['prompt']


def test_completed_baseline_cache_skips_calls_and_preserves_historical_timing(scene,tmp_path):
    cache = tmp_path/'cache'
    first = evolve(scene,tmp_path/'first',Model(*_cached_outputs()),patches=1,repeats=1,baseline_cache_dir=cache)
    assert first['baseline_cache']['stored']
    second_model = Model(*_cached_outputs()[2:])
    second = evolve(scene,tmp_path/'second',second_model,patches=1,repeats=1,baseline_cache_dir=cache)
    assert second['baseline_cache']['status'] == 'HIT'
    assert second['totals']['model_calls'] == 4
    assert second['rollouts'][0]['baseline_cache']['reused']
    assert second['totals']['cached_baseline_seconds'] == first['rollouts'][0]['metrics']['elapsed_s']
    assert second['totals']['replay_seconds'] == second['rollouts'][1]['metrics']['elapsed_s']
    assert second_model.requests[0]['stage'] == 'patch_diagnose'
    assert 'SECRET' not in second_model.requests[2]['prompt']
    assert (tmp_path/'second/cached_baseline/replays/baseline_r00/result.json').is_file()


def test_cache_misses_when_inputs_or_settings_change_and_rejects_tampering(scene,tmp_path):
    cache = tmp_path/'cache'
    first = evolve(scene,tmp_path/'first',Model(*_cached_outputs()),patches=1,repeats=1,baseline_cache_dir=cache)
    changed = copy.deepcopy(scene); changed['fold_goal'] = 'A different fold objective'
    miss = evolve(changed,tmp_path/'changed',Model(*_cached_outputs()),patches=1,repeats=1,baseline_cache_dir=cache)
    assert miss['baseline_cache']['status'] == 'MISS'
    setting = evolve(scene,tmp_path/'setting',Model(*_cached_outputs()),patches=1,repeats=1,max_ops=10,baseline_cache_dir=cache)
    assert setting['baseline_cache']['status'] == 'MISS'
    entry = cache/first['baseline_cache']['key']
    (entry/'artifacts/replays/baseline_r00/result.json').write_text('{}')
    invalid = evolve(scene,tmp_path/'invalid',Model(*_cached_outputs()),patches=1,repeats=1,baseline_cache_dir=cache)
    assert invalid['baseline_cache']['status'] == 'INVALID'
    assert invalid['totals']['model_calls'] == 6


def test_failed_baseline_is_not_cached(scene,tmp_path):
    cache = tmp_path/'cache'
    invalid = plan(); invalid['action']['selected_reference']['reference_id'] = 'R999'
    report = evolve(scene,tmp_path/'failed',Model(observe(),invalid),patches=1,repeats=1,baseline_cache_dir=cache)
    assert report['status'] == 'BLOCKED_BASELINE'
    assert not report['baseline_cache']['stored']
    assert not cache.exists()


def test_historical_usage_not_counted_in_new_session(tmp_path):
    from cloth_agent.harness.reasoning_learning import call_metrics
    (tmp_path/'cached_baseline').mkdir()
    (tmp_path/'cached_baseline/stdout.jsonl').write_text('invalid historical text')
    result = {'type':'result','subtype':'success','result':'{}','usage':{
        'input_tokens':10,'output_tokens':20,'cache_read_input_tokens':0,'cache_creation_input_tokens':0}}
    (tmp_path/'stdout.jsonl').write_text(json.dumps(result))
    metrics = call_metrics([{'backend_invoked':True,'response_received':True,'tool_round_trips':0}],
                           tmp_path,exclude_dirs=('cached_baseline',))
    assert metrics['model_calls'] == 1
    assert metrics['usage']['total_tokens'] == 30


def planning_only_implementation():
    result={'config':copy.deepcopy(baseline_bundle()['config']),'skill':None}
    result['config']['reasoning_harness']['stages'][0]['instruction'] += ' Avoid repeating evidence descriptions.'
    return result


def test_planning_only_patch_reuses_exact_preplanning_evidence(scene,tmp_path):
    model=Model(observe([_request('orientation',angle=90)]),plan(),
                proposal('PROMPT','plan_instruction'),planning_only_implementation(),plan())
    report=evolve(scene,tmp_path/'fixed',model,patches=1,repeats=1)
    base,candidate=report['rollouts']
    assert base['status']==candidate['status']=='READY'
    assert len(model.requests)==5
    assert candidate['metrics']['model_calls']==1
    assert candidate['metrics']['host_image_ops']==0
    assert candidate['metrics']['exclusive_phases_s']['observation_call_s']==0
    assert base['derived_evidence_hash']==candidate['derived_evidence_hash']
    assert candidate['observation_reuse']['source_rollout']=='baseline_r00'
    assert 'SECRET' not in model.requests[-1]['prompt']
    assert not (tmp_path/'fixed/replays/patch_00_r00/observe').exists()
    assert report['candidates'][0]['promotion']['comparison_scope']=='SHARED_OBSERVATION_RECONSTRUCTED_TOTAL'
    expected=base['metrics']['elapsed_s']-base['metrics']['exclusive_phases_s']['planning_s']+candidate['metrics']['exclusive_phases_s']['planning_s']
    assert candidate['metrics']['comparison_elapsed_s']==pytest.approx(expected)


def test_explicit_historical_baseline_needs_only_three_new_calls(scene,tmp_path):
    first=evolve(scene,tmp_path/'old',Model(*_cached_outputs()),patches=1,repeats=1)
    model=Model(proposal('PROMPT','plan_instruction'),planning_only_implementation(),plan())
    second=evolve(scene,tmp_path/'new',model,patches=1,repeats=1,reuse_baseline_run=tmp_path/'old')
    assert second['status']=='NO_PROMOTION'
    assert second['baseline_cache']['mode']=='EXPLICIT_HISTORICAL_RUN'
    assert second['rollouts'][1]['status']=='READY'
    assert second['totals']['model_calls']==3
    assert second['rollouts'][1]['derived_evidence_hash']==first['rollouts'][0]['derived_evidence_hash']


def test_changed_observation_not_reused(scene,tmp_path):
    model=Model(*_cached_outputs())
    report=evolve(scene,tmp_path/'changed_observer',model,patches=1,repeats=1)
    assert len(model.requests)==6
    assert not report['rollouts'][1].get('observation_reuse')


def test_historical_prepared_image_tamper_blocks_reuse(scene,tmp_path):
    evolve(scene,tmp_path/'old',Model(*_cached_outputs()),patches=1,repeats=1)
    path=tmp_path/'old/replays/baseline_r00/prepared/image_0.png'
    Image.new('RGB',(100,80),'red').save(path)
    model=Model()
    report=evolve(scene,tmp_path/'new',model,patches=1,repeats=1,reuse_baseline_run=tmp_path/'old')
    assert report['status']=='ERROR'
    assert not model.requests


def test_invalid_stage_context_gets_one_local_repair(scene,tmp_path):
    initial=planning_only_implementation()
    initial['config']['reasoning_harness']['stages'][0]['context']=['observation']
    model=Model(observe(),plan(),proposal('HARNESS','plan'),initial,
        {'status':'REPAIRED','reason':'Use explicitly supplied observation evidence; no session-resume claim.',
         'implementation':planning_only_implementation()},plan())
    result=evolve(scene,tmp_path/'repaired',model,patches=1,repeats=1)
    assert result['candidates'][0]['implementation_repaired']
    assert result['rollouts'][-1]['status']=='READY'
    repair=next(r for r in model.requests if r['stage']=='patch_implementation_repair')
    assert 'available earlier stages: []' in repair['prompt']
    assert len([r for r in model.requests if r['stage']=='patch_observe'])==1
    assert (tmp_path/'repaired/candidates/patch_00/implementation_initial.json').is_file()


def test_unsupported_mechanism_is_not_silently_repaired_into_prompt(scene,tmp_path):
    initial=planning_only_implementation();initial['config']['reasoning_harness']['stages'][0]['context']=['observation']
    model=Model(observe(),plan(),proposal('HARNESS','plan'),initial,
                {'status':'NEEDS_RUNTIME_SUPPORT','reason':'Session continuation has no backend API','implementation':None})
    report=evolve(scene,tmp_path/'unsupported',model,patches=1,repeats=1)
    assert report['candidates'][0]['status']=='NEEDS_RUNTIME_SUPPORT'
    assert len(report['rollouts'])==1
    assert not (tmp_path/'unsupported/candidates/patch_00/bundle.json').exists()


def test_stage_usage_separates_planning_and_observation():
    from cloth_agent.harness.patch_evolution import stage_cost_summary
    row={'rollout_id':'r','metrics':{'elapsed_s':350,'exclusive_phases_s':{'planning_s':260},
        'usage':{'cache_creation_input_tokens':47521,'output_tokens':21920}},
        'reasoning':{'metrics':{'usage':{'cache_creation_input_tokens':32028,'output_tokens':17029}}}}
    summary=stage_cost_summary(row)
    assert summary['observation_usage']=={'cache_creation_input_tokens':15493,'output_tokens':4891}
    assert summary['planning_usage']['output_tokens']==17029


def test_proposal_prose_transport_normalizes_without_changing_invariants():
    from cloth_agent.harness.candidate_patch import proposal_transport_schema,normalize_proposal_transport,PROPOSAL_SCHEMA
    raw=proposal();raw['must_preserve']='Keep coordinates; do not hide blocking gaps.\nPreserve all source images.'
    validate_schema(raw,proposal_transport_schema(['baseline_r00']))
    normalized=normalize_proposal_transport(raw)
    validate_schema(normalized,PROPOSAL_SCHEMA)
    assert ''.join(normalized['must_preserve'])==raw['must_preserve']
    assert normalize_proposal_transport(proposal())==proposal()


def test_retry_saved_candidate_only_repeats_planning(scene,tmp_path):
    first=Model(observe(),plan(),proposal('PROMPT','plan_instruction'),planning_only_implementation(),plan())
    evolve(scene,tmp_path/'first',first,patches=1,repeats=1)
    retry=Model(plan())
    report=evolve(scene,tmp_path/'retry',retry,patches=1,repeats=1,
        reuse_baseline_run=tmp_path/'first',retry_candidate=tmp_path/'first/candidates/patch_00')
    assert report['rollouts'][-1]['status']=='READY'
    assert report['totals']['model_calls']==1
    assert retry.requests[0]['stage']=='reasoning_rollout'
    assert 'SECRET' not in retry.requests[0]['prompt']
    assert report['candidates'][0]['source_candidate'].endswith('first/candidates/patch_00')


def test_next_iteration_receives_actual_patch_results_and_can_cite_them(scene,tmp_path):
    second=proposal('NEW_TOOL','paired_roi_reuse')
    second['evidence_rollouts']=['patch_00_r00']
    model=Model(observe(),plan(),proposal(),code_implementation(),observe(),plan(),
                second,code_implementation(True),observe(),plan())
    report=evolve(scene,tmp_path/'feedback',model,patches=2,repeats=1)
    assert all(r['status']=='READY' for r in report['rollouts'])
    context=read_json(tmp_path/'feedback/candidates/patch_01/diagnosis_context.json')
    assert context['allowed_evidence_rollouts']==['baseline_r00','patch_00_r00']
    prior=context['iteration_feedback'][0]
    assert prior['implementation']==code_implementation()
    assert prior['rollouts'][0]['action']==report['rollouts'][1]['action']
    assert prior['rollouts'][0]['costs']['total_inclusive_s']>0
    assert prior['provisional_evaluation']['checks']['valid_complete_plans']
    assert 'failed_checks' in prior
    assert report['candidates'][1]['proposal']['evidence_rollouts']==['patch_00_r00']
    # Historical actions remain absent from planning input; only diagnosis/implementation get feedback.
    replays=[r for r in model.requests if r['stage']=='reasoning_rollout']
    assert all('iteration_feedback' not in r['prompt'] for r in replays)


def test_consecutive_failed_calls_stop_unattended_generation(scene,tmp_path):
    model=Model(observe(),plan(),TimeoutError('remote timeout'),TimeoutError('remote timeout'),
                TimeoutError('must never call this'))
    report=evolve(scene,tmp_path/'stopped',model,patches=5,repeats=1,max_consecutive_failures=2)
    assert report['stop_reason']=='CONSECUTIVE_EXECUTION_FAILURES'
    assert len(report['candidates'])==2
    assert len(model.requests)==4
    ctx=read_json(tmp_path/'stopped/candidates/patch_01/diagnosis_context.json')
    assert 'remote timeout' in ctx['iteration_feedback'][0]['reason']
    assert ctx['iteration_feedback'][0]['provisional_evaluation'] is None


def test_successful_replay_resets_failure_streak_even_without_promotion(scene,tmp_path):
    model=Model(observe(),plan(),TimeoutError('temporary'),proposal(),code_implementation(),observe(),plan(),
                TimeoutError('temporary again'))
    report=evolve(scene,tmp_path/'reset',model,patches=3,repeats=1,max_consecutive_failures=2)
    assert [c['feedback']['consecutive_execution_failures'] for c in report['candidates']]==[1,0,1]
    assert 'stop_reason' not in report


def test_transport_failure_feedback_does_not_fill_next_prompt():
    from cloth_agent.harness.patch_evolution import feedback_context
    raw='PlannerBackendError: exit=124: __CLOTH_PROGRESS__ '+('{"error_status":524,"phase":"waiting"}\n'*4000)
    source={'reason':raw,'nested':[{'error':raw}], 'finding':'visible seam '*3000}
    result=feedback_context(source)
    assert len(result['reason'])<2000
    assert '524' in result['reason']
    assert len(result['nested'][0]['error'])<2000
    assert result['finding']==source['finding']
    assert source['reason']==raw


def test_resume_keeps_prior_feedback_without_repeating_calls(scene,tmp_path):
    first=evolve(scene,tmp_path/'first',Model(observe(),plan(),proposal(),code_implementation(),observe(),plan()),
                 patches=1,repeats=1)
    model=Model(proposal('NEW_TOOL','paired_roi_reuse'),code_implementation(True),observe(),plan())
    resumed=evolve(scene,tmp_path/'resumed',model,patches=2,repeats=1,
                   reuse_baseline_run=tmp_path/'first',resume_run=tmp_path/'first')
    assert len(model.requests)==4
    assert len(resumed['candidates'])==2
    assert len(resumed['rollouts'])==3
    assert resumed['candidates'][0]['resumed_from']
    assert resumed['rollouts'][1]['resumed_from']
    assert resumed['totals']['replay_seconds']==resumed['rollouts'][2]['metrics']['elapsed_s']
    ctx=read_json(tmp_path/'resumed/candidates/patch_01/diagnosis_context.json')
    assert ctx['iteration_feedback'][0]['candidate_id']=='patch_00'
    assert (tmp_path/'resumed/replays/patch_00_r00').is_dir()
    assert read_json(tmp_path/'first/report.json')==first


def test_resume_rejects_changed_settings(scene,tmp_path):
    evolve(scene,tmp_path/'first',Model(observe(),plan(),proposal(),code_implementation(),observe(),plan()),patches=1,repeats=1)
    model=Model()
    result=evolve(scene,tmp_path/'resumed',model,patches=2,repeats=1,call_timeout=181,
                  reuse_baseline_run=tmp_path/'first',resume_run=tmp_path/'first')
    assert result['status']=='ERROR'
    assert not model.requests
