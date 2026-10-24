"""Sequence provenance, evidence contracts, and robot-free skill-policy tests."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest
from PIL import Image
from cloth_agent.experiments import cloth_skill_library as lib
from cloth_agent.experiments import arm6_skill_policy as policy
from cloth_agent.experiments.discovery_data import save, read
from cloth_agent.experiments.graph_policy import frozen, evidence_file, Requests


def case(ids):
    return dict(title='Reveal sleeve',applicability='A visible panel covers a sleeve edge',
        subgoal='Expose a graspable sleeve edge',grasp_relation='free covering edge',
        move_relation='flip inward to clear the sleeve',expected_effect='new sleeve edge visible',
        completion_test='visible accessible edge',next_step='grasp newly exposed sleeve',
        evidence_step_ids=ids,visual_evidence=['before covered; after edge exposed'],
        intent_basis='SEQUENCE_INFERENCE',outcome='OBSERVED_SUBGOAL',limitations=['layer count uncertain'],confidence=.93)


def selection(done=False,effect='INITIAL'):
    return dict(decision='DONE' if done else 'ACT',skill_id='' if done else 'reveal-sleeve',confidence=.95,
        assessment='visible folds',subgoal='Expose sleeve',applicability_evidence=['covering layer'],uncertainties=[],
        reason='clear grasp access',previous_effect=effect,previous_effect_evidence=['sleeve now visible'] if effect!='INITIAL' else [],
        completion_evidence=['sleeves and hem unfolded'] if done else [])


def action():
    return dict(decision='ACT',confidence=.95,grasp_px=[20,30],target_px=[60,30],opening_angle_image_deg=45,
        visible_corner_evidence='visible free edge',reason='clear covering layer',expected_change='expose sleeve edge',
        skill_id='reveal-sleeve',subgoal='Expose sleeve',grasp_role='covering edge',target_role='clearance region',
        adaptation_reason='current layer points in a different direction')


class Backend:
    model='fixture'
    def __init__(self):self.calls=[];self.done=False
    def invoke(self,prompt,images,schema):
        data=json.loads(prompt.split('\n')[-1])['payload'];self.calls.append((data,images))
        keys=schema['properties']
        if 'links' in keys:
            ids=data['window']['step_ids']
            value=dict(links=[dict(before_step=a,next_step=b,continuity='CONTINUOUS',evidence='same visible layout') for a,b in zip(ids,ids[1:])],
                       cases=[case(ids)],limitations=['fixture only'])
        elif 'skills' in keys:
            ids=[c['case_id'] for c in data['cases']]
            value=dict(skills=[dict(skill_id='reveal-sleeve',name='Expose hidden sleeve',
                applicability='covering layer',subgoal='Expose sleeve',grasp_relation='free visible covering edge',
                move_relation='inward flip clears the sleeve',completion_test='new accessible sleeve edge',
                next_step='unfold the exposed sleeve',support_case_ids=ids,counterexample_case_ids=[],limitations=['fixture'])],omitted_cases=[])
        elif 'previous_effect' in keys:value=selection(self.done,'SUBGOAL_REACHED' if self.done else 'INITIAL')
        else:value=action()
        return dict(text=json.dumps(value),elapsed_s=.01,tool_calls=0,visual_tool_calls=0,completed_turns=1)


@pytest.fixture
def archive(tmp_path):
    observations={};steps=[]
    for i in range(4):
        folder=tmp_path/'session'/f'traj_{i:04d}';folder.mkdir(parents=True)
        save(folder/'traj.json',dict(episode_id=0,preceding_reset=None))
        for phase in ('before','after'):
            path=folder/f'{phase}.png';Image.new('RGB',(120,100),'navy').save(path)
            observations[f'{phase}{i}']=dict(rgb=evidence_file(path))
        steps.append(dict(step_id=f's{i:03d}',session='session',iteration=i,episode_id=0,source_folder=str(folder),
            before_observation=f'before{i}',after_observation=f'after{i}',
            outcome=dict(after_provenance='dedicated_after'),commanded_action=dict(
                manifest=evidence_file(folder/'traj.json'),selected_points=[],automated_phases=[],collection_mode='automatic_pick_place')))
    value=dict(protocol='arm6-archive-experience-v1',steps=steps,observations=observations)
    path=tmp_path/'trajectory.json';frozen(path,value)
    return path,value


def test_segments_split_gaps_episodes_sessions_and_reset_markers(archive):
    _,data=archive
    assert len(lib.segments(data))==1
    broken=copy.deepcopy(data);broken['steps'][2]['episode_id']=1;broken['steps'][3]['episode_id']=1
    assert [len(g) for g in lib.segments(broken)]==[2,2]
    broken=copy.deepcopy(data);broken['steps'].pop(1)
    assert [len(g) for g in lib.segments(broken)]==[1,2]
    broken=copy.deepcopy(data);broken['steps'][3]['session']='second'
    assert len(lib.segments(broken))==2
    reset=Path(data['steps'][0]['source_folder']).parent/'resets'/'after_iter_0001';reset.mkdir(parents=True)
    assert [len(g) for g in lib.segments(data)]==[2,2]
    assert lib.segments(data,holdout_sessions=['session'])==[]
    with pytest.raises(ValueError,match='overlap'):lib.segments(data,['session'],['session'])


def test_window_coverage_does_not_bridge_segments(archive):
    _,data=archive;groups=[data['steps'][:2],data['steps'][2:]]
    windows=lib.make_windows(groups,3,1)
    assert [w['step_ids'] for w in windows]==[['s000','s001'],['s002','s003']]


@pytest.mark.parametrize('fault',['unknown','skip','boundary','missing_after'])
def test_analysis_rejects_invented_delayed_credit(archive,fault):
    _,data=archive;steps={s['step_id']:s for s in data['steps']}
    window=dict(step_ids=list(steps))
    value=dict(links=[dict(before_step=a,next_step=b,continuity='CONTINUOUS',evidence='same') for a,b in zip(steps,list(steps)[1:])],
               cases=[case(list(steps))],limitations=[])
    if fault=='unknown':value['cases'][0]['evidence_step_ids']=['absent']
    if fault=='skip':value['cases'][0]['evidence_step_ids']=['s000','s002']
    if fault=='boundary':value['links'][0]['continuity']='CHANGED'
    if fault=='missing_after':steps['s001']['after_observation']=None
    with pytest.raises(ValueError):lib.validate_analysis(value,window,steps)


def test_compile_exports_portable_visual_skill_and_resumes_without_model(archive,tmp_path):
    path,_=archive;backend=Backend();output=tmp_path/'compiled'
    summary=lib.compile_library(path,output,size=3,overlap=1,backend=backend)
    assert summary['skills']==1 and summary['robot_calls']==0
    library=lib.load_library(output/'skill/library.json')
    assert library['skills'][0]['move_relation'].startswith('inward')
    assert all(len(c['images'])==len(c['evidence_step_ids']) for c in library['cases'].values())
    calls=len(backend.calls);lib.compile_library(path,output,size=3,overlap=1,backend=backend)
    assert len(backend.calls)==calls
    (output/'skill'/library['cases']['w000_c00']['images'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='evidence'):lib.load_library(output/'skill/library.json')


def test_prepare_only_does_not_call_model_and_hash_change_rejected(archive,tmp_path):
    path,data=archive;backend=Backend()
    summary=lib.compile_library(path,tmp_path/'prepared',prepare_only=True,backend=backend)
    assert summary['status']=='PREPARED' and not backend.calls
    Path(data['observations']['before0']['rgb']['path']).write_bytes(b'mutated')
    with pytest.raises(ValueError,match='evidence'):lib.compile_library(path,tmp_path/'bad',prepare_only=True)


def test_possible_or_counterexample_cannot_be_promoted_to_support():
    value=dict(skills=[dict(skill_id='reveal',name='Reveal',applicability='cover',subgoal='access',grasp_relation='edge',
        move_relation='inward',completion_test='edge visible',next_step='unfold',support_case_ids=['c'],counterexample_case_ids=[],limitations=[])],omitted_cases=[])
    c=case(['s000']);c['outcome']='POSSIBLE'
    with pytest.raises(ValueError,match='evidence role'):lib.validate_synthesis(value,{'c':c})


class Media:
    def begin(self,*a):pass
    def phase(self,*a):pass
    def snapshot(self,*a):pass
    def edge(self,*a):pass


@pytest.fixture
def bundle(archive,tmp_path):
    path,_=archive;out=tmp_path/'compile';lib.compile_library(path,out,size=3,overlap=1,backend=Backend())
    p=out/'skill/library.json';return p,lib.load_library(p)


def test_offline_cli_never_creates_robot_and_uses_examples(bundle,tmp_path,monkeypatch):
    from cloth_agent import gamepad_collect_runtime
    path,library=bundle;image=tmp_path/'current.png';Image.new('RGB',(120,100)).save(image)
    backend=Backend();monkeypatch.setattr(policy,'CodexMultimodal',lambda *a:backend)
    monkeypatch.setattr(gamepad_collect_runtime,'CollectionRuntime',lambda *a:pytest.fail('no hardware in offline mode'))
    out=tmp_path/'offline'
    assert policy.main(['--library',str(path),'--image',str(image),'--output',str(out)])==0
    result=read(out/'result.json')
    assert result['status']=='OFFLINE_PLAN_ONLY' and result['physical_actions']==0
    assert (out/'proposal_preview.png').is_file()
    assert len(backend.calls)==2 and len(backend.calls[1][1])>1
    assert backend.calls[1][0]['observation_context']['measured_depth_available'] is False


def test_loop_evaluates_subgoal_before_next_action(bundle,tmp_path,monkeypatch):
    path,library=bundle;backend=Backend();out=tmp_path/'run';out.mkdir()
    image=tmp_path/'current.png';Image.new('RGB',(120,100)).save(image)
    obs=SimpleNamespace(image=image);runtime=SimpleNamespace(cancel=threading.Event(),observe=lambda p:obs)
    args=SimpleNamespace(max_steps=2,max_stalled_steps=2,confidence_threshold=.8,max_pull_mm=150,grasp_offset_mm=-15)
    req=policy.make_requests(out,SimpleNamespace(model='fixture',timeout_s=10,max_calls=8),lib.ENTRYPOINT,backend)
    monkeypatch.setattr(policy,'motion_context',lambda *a:{})
    moved=[]
    def execute(*a,**kw):
        moved.append(kw['kind']);backend.done=True
        a[5]['physical_actions']+=1
        return 'EXECUTED',obs,dict(before=str(image),proposal=a[2])
    monkeypatch.setattr(policy,'execute_proposal',execute)
    result=dict(steps=[],assessments=[],physical_actions=0)
    policy.loop(runtime,req,library,path.parent,out,Media(),result,args)
    assert result['status']=='MODEL_REPORTED_UNFOLDED' and moved==['skill']
    assert backend.calls[-1][0]['previous_action']['expected_change']=='expose sleeve edge'
    assert result['assessments'][-1]['previous_effect']=='SUBGOAL_REACHED'


@pytest.mark.parametrize('reason',['uncertain','stop','limit','cancel'])
def test_no_dispatch_on_selection_stop_or_cancellation(bundle,tmp_path,monkeypatch,reason):
    path,library=bundle;out=tmp_path/'run';out.mkdir();image=tmp_path/'current.png';Image.new('RGB',(120,100)).save(image)
    obs=SimpleNamespace(image=image);runtime=SimpleNamespace(cancel=threading.Event(),observe=lambda p:obs)
    args=SimpleNamespace(max_steps=0 if reason=='limit' else 2,max_stalled_steps=2,confidence_threshold=.8,max_pull_mm=150,grasp_offset_mm=-15)
    def assess(*a,**kw):
        v=selection()
        if reason=='uncertain':v=selection(done=True);v['confidence']=.4
        if reason=='stop':v['decision']='STOP';v['skill_id']=''
        if reason=='cancel':runtime.cancel.set()
        return v
    monkeypatch.setattr(policy,'assess',assess)
    monkeypatch.setattr(policy,'execute_proposal',lambda *a,**kw:pytest.fail('must not dispatch'))
    monkeypatch.setattr(policy,'ground',lambda *a,**kw:pytest.fail('must not ground'))
    result=dict(steps=[],assessments=[],physical_actions=0)
    if reason=='cancel':
        with pytest.raises(KeyboardInterrupt):policy.loop(runtime,None,library,path.parent,out,Media(),result,args)
    else:
        policy.loop(runtime,None,library,path.parent,out,Media(),result,args)
        assert result['physical_actions']==0


def test_stalled_subgoal_stops_repeated_actions(bundle,tmp_path,monkeypatch):
    path,library=bundle;out=tmp_path/'stalled';out.mkdir()
    image=tmp_path/'current.png';Image.new('RGB',(120,100)).save(image)
    obs=SimpleNamespace(image=image);runtime=SimpleNamespace(cancel=threading.Event(),observe=lambda p:obs)
    args=SimpleNamespace(max_steps=8,max_stalled_steps=2,confidence_threshold=.8,max_pull_mm=150,grasp_offset_mm=-15)
    assessments=iter([selection(),selection(effect='NO_CHANGE'),selection(effect='NO_CHANGE')]);moves=[]
    monkeypatch.setattr(policy,'assess',lambda *a:next(assessments))
    monkeypatch.setattr(policy,'motion_context',lambda *a:{})
    monkeypatch.setattr(policy,'ground',lambda *a:action())
    def execute(*a,**kw):
        moves.append(a[2]);a[5]['physical_actions']+=1
        return 'EXECUTED',obs,dict(before=str(image),proposal=a[2])
    monkeypatch.setattr(policy,'execute_proposal',execute)
    result=dict(steps=[],assessments=[],physical_actions=0)
    policy.loop(runtime,None,library,path.parent,out,Media(),result,args)
    assert result['status']=='SUBGOAL_STALLED' and len(moves)==2


def test_selection_rejects_unknown_skill_and_unsubstantiated_completion(bundle):
    _,library=bundle;v=selection();v['skill_id']='invented'
    with pytest.raises(ValueError,match='unknown skill'):policy.validate_selection(v,library)
    v=selection(done=True);v['completion_evidence']=[]
    with pytest.raises(ValueError,match='evidence'):policy.validate_selection(v,library)


@pytest.mark.parametrize('ground_decision,ground_confidence', [('ACT',.7),('STOP',.95)])
def test_tentative_selection_gets_examples_but_rejected_grounding_cannot_move(bundle,tmp_path,monkeypatch,ground_decision,ground_confidence):
    from cloth_agent.experiments import arm6_corner_pull as pull
    path,library=bundle;out=tmp_path/'tentative';out.mkdir()
    image=tmp_path/'current.png';Image.new('RGB',(120,100)).save(image)
    obs=SimpleNamespace(image=image);runtime=SimpleNamespace(cancel=threading.Event(),observe=lambda p:obs)
    args=SimpleNamespace(max_steps=2,max_stalled_steps=2,confidence_threshold=.8,max_pull_mm=150,grasp_offset_mm=-15)
    class TentativeBackend(Backend):
        def invoke(self,prompt,images,schema):
            response=super().invoke(prompt,images,schema);value=json.loads(response['text'])
            if 'previous_effect' in schema['properties']:value['confidence']=.74
            else:value.update(decision=ground_decision,confidence=ground_confidence)
            response['text']=json.dumps(value);return response
    backend=TentativeBackend()
    requests=policy.make_requests(out,SimpleNamespace(model='fixture',timeout_s=10,max_calls=4),lib.ENTRYPOINT,backend)
    monkeypatch.setattr(policy,'motion_context',lambda *a:{})
    monkeypatch.setattr(pull,'prepare',lambda *a,**kw:pytest.fail('rejected grounding must not plan or dispatch'))
    result=dict(steps=[],assessments=[],physical_actions=0)
    policy.loop(runtime,requests,library,path.parent,out,Media(),result,args)
    assert len(backend.calls)==2 and len(backend.calls[1][1])>1
    assert result['status']==('SKILL_UNCERTAIN' if ground_decision=='ACT' else 'SKILL_MODEL_STOP')
    assert result['physical_actions']==0


def test_tentative_choice_and_completion_have_separate_gates():
    v=selection();v['confidence']=.74
    assert policy.selection_stop(v,.8) is None
    v=selection(done=True);v['confidence']=.74
    assert policy.selection_stop(v,.8)=='SKILL_COMPLETION_UNCERTAIN'
    v=selection(done=True)
    assert policy.selection_stop(v,.8)=='MODEL_REPORTED_UNFOLDED'


def test_zero_threshold_cli_accepts_zero_confidence_pixel_plan(bundle,tmp_path,monkeypatch):
    path,_=bundle;image=tmp_path/'current.png';Image.new('RGB',(120,100)).save(image)
    class LowConfidenceBackend(Backend):
        def invoke(self,*a):
            response=super().invoke(*a);value=json.loads(response['text']);value['confidence']=0
            response['text']=json.dumps(value);return response
    monkeypatch.setattr(policy,'CodexMultimodal',lambda *a:LowConfidenceBackend())
    out=tmp_path/'no_threshold'
    assert policy.main(['--library',str(path),'--image',str(image),'--output',str(out),'--confidence-threshold','0'])==0
    result=read(out/'result.json')
    assert result['confidence_gating_enabled'] is False and result['proposed_next']=='PIXEL_PLAN_ONLY'
    assert result['proposal']['confidence']==0 and result['physical_actions']==0
    done=selection(done=True);done['confidence']=0
    assert policy.selection_stop(done,0)=='MODEL_REPORTED_UNFOLDED'


def test_zero_threshold_passes_to_geometry_without_bypassing_it(tmp_path,monkeypatch):
    from cloth_agent.experiments import arm6_corner_pull as pull
    from cloth_agent.dual_arm.geometry import DualArmError
    p=action();p['confidence']=0;seen=[]
    def invalid_geometry(*a,**kw):seen.append('geometry');raise DualArmError('measured depth missing')
    monkeypatch.setattr(pull,'prepare',invalid_geometry)
    runtime=SimpleNamespace(cancel=threading.Event())
    status,_,_=pull.execute_proposal(runtime,None,p,tmp_path,Media(),{},threshold=0,kind='skill')
    assert status=='SKILL_GEOMETRY_REJECTED' and seen==['geometry']
    p['decision']='STOP';seen.clear()
    status,_,_=pull.execute_proposal(runtime,None,p,tmp_path,Media(),{},threshold=0,kind='skill')
    assert status=='SKILL_MODEL_STOP' and seen==[]
