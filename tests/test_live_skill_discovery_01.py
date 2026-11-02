"""Offline only: all model calls, robot connections and movements are mocked."""
import importlib.util,json,tempfile,threading,contextlib,io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import pytest
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('live_study',ROOT/'experiments/live_skill_discovery_01/legacy_skill_loop.py')
live=importlib.util.module_from_spec(spec);spec.loader.exec_module(live)

PLAN=dict(decision='ACTION',grasp_px=[10,10],target_px=[20,20],motion_type='lift_pull_place',grasp_angle_deg=None,short_reason='local fold')
INTENT={**PLAN,'target_region':'original sleeve','visible_fold':'visible fold','expected_visible_change':'original crease reduced'}
TEACHER={'decision':'NEW_SKILL','observed_gap':'gap','physical_hypothesis':'unverified hypothesis','mechanism_difference':'different material interaction','unknowns':'layers unknown','skill':dict(trigger='可见局部卷折',grasp_selection='可见布边',direction_selection='朝空桌面',expected_effect='卷折减少')}

def episode(tmp_path,outcomes,*,failure=False,teacher_decision='NEW_SKILL'):
    class Runtime:
        coordinator=object();cancel=threading.Event()
        def capture_current(self,folder):
            folder.mkdir(parents=True,exist_ok=True);im=folder/'rgb.png';Image.new('RGB',(64,64),'pink').save(im)
            return SimpleNamespace(image=im,directory=folder,meta={'joints_deg':{'left':[0]*6,'right':[0]*7}})
        close=Mock();stop=Mock()
    runtime=Runtime();seen_goals=[]
    def plan(folder,before,skill=None,original_goal=None):
        seen_goals.append(original_goal);live.save(folder/'plan.json',PLAN);return PLAN,INTENT
    def dispatch(runtime,root,state,folder,before,plan,intent,**kwargs):
        if failure:raise live.StopExperiment('Host rejected')
        state['dispatches']+=1
        for n,d in [('execution.json',{'status':'EXECUTED','physical_execution':True}),('points.json',[]),('motion_profile.json',{})]:live.save(folder/n,d)
        return runtime.capture_current(folder/'after')
    teacher={**TEACHER,'decision':teacher_decision}
    def request(folder,*args):
        live.save(folder/'request_payload.json',{'test':True})
        index=int(folder.parent.name.split('_')[-1])
        return {**teacher,'decision':teacher_decision[index]} if isinstance(teacher_decision,list) else teacher
    with patch.object(live,'make_runtime',return_value=runtime),patch.object(live,'runtime_policy',return_value={}),patch.object(live,'connect_stationary'),patch.object(live,'startup_reset') as reset,patch.object(live,'planner',side_effect=plan) as planner,patch.object(live,'dispatch_once',side_effect=dispatch),patch.object(live,'observe_outcome',side_effect=[{'choice':o,'note':'observed change'} for o in outcomes]),patch.object(live,'request',side_effect=request) as req,contextlib.redirect_stdout(io.StringIO()):
        out=tmp_path/'episode';live.run(SimpleNamespace(output=out,config=tmp_path/'unused'))
    reset.assert_called_once()
    return live.read(out/'status.json'),planner.call_count,req.call_count,seen_goals,out

@pytest.mark.parametrize('outcome',['IMPROVED','UNKNOWN'])
def test_each_visible_outcome_is_reviewed(tmp_path,outcome):
    count=6
    state,n,t,_,_=episode(tmp_path,[outcome]*count);assert state['dispatches']==count and n==count and t==6

def test_host_failure_never_becomes_cloth_failure(tmp_path):
    state,n,t,_,_=episode(tmp_path,[],failure=True);assert state['dispatches']==0 and t==0 and state['termination']=='STOPPED'

def test_no_new_skill_continues(tmp_path):
    state,n,t,_,_=episode(tmp_path,['NO_CLEAR_IMPROVEMENT']*6,teacher_decision='NO_NEW_SKILL')
    assert (state['dispatches'],n,t)==(6,6,6)
    assert state['termination']=='ITERATION_LIMIT'

def test_six_rounds_each_teacher_and_original_goal_preserved(tmp_path):
    state,n,t,goals,out=episode(tmp_path,['NO_CLEAR_IMPROVEMENT']*5+['IMPROVED'])
    assert (state['dispatches'],n,t)==(6,6,6)
    assert goals[0] is None and goals[1]['target_region']=='original sleeve'
    assert state['termination']=='ITERATION_LIMIT'
    assert live.sha(out/'skill_frozen.json')==live.read(out/'skill_freeze.json')['sha256']

def test_dry_run_never_constructs_runtime():
    with patch.object(live,'make_runtime',side_effect=AssertionError('hardware')),contextlib.redirect_stdout(io.StringIO()):assert live.main(['--dry-run'])==0

def test_authorization_does_not_claim_human_review(tmp_path):
    with patch('builtins.input',side_effect=AssertionError('no terminal input')):
        live.execution_authorization(tmp_path,{'native_sha256':'abc'})
    result=live.read(tmp_path/'execution_authorization.json')
    assert result['mode']=='automatic' and result['per_action_human_review'] is False
    assert result['binding']['native_sha256']=='abc'

def test_host_rejection_blocks_automatic_dispatch(tmp_path):
    runtime=SimpleNamespace(_execute_operations=Mock())
    with patch.object(live,'prepare',side_effect=live.StopExperiment('Host rejected')):
        with pytest.raises(live.StopExperiment,match='Host rejected'):
            live.dispatch_once(runtime,tmp_path,{'dispatches':0},tmp_path,None,PLAN,INTENT)
    runtime._execute_operations.assert_not_called()

def test_budget_blocks_seventh_dispatch(tmp_path):
    with pytest.raises(live.StopExperiment,match='预算'):live.dispatch_once(None,tmp_path,{'dispatches':6},tmp_path,None,PLAN,INTENT)

def test_runtime_reuses_existing_arm6_configuration_without_overrides():
    from cloth_agent.experiments import trajectory_memory_full_episode_v1 as episode
    config=SimpleNamespace(path_validation_enabled=False,collision_envelopes_disabled_arms={'left'},home_recover_stopped=True)
    runtime=SimpleNamespace(config=config,direct_joint_moves=True)
    with patch.object(episode,'EpisodeCollectionRuntime',return_value=runtime) as factory:
        assert live.make_runtime(Path('unused')) is runtime
    factory.assert_called_once_with(Path('unused'))
    live.safety_assert(runtime)
    assert not config.path_validation_enabled and config.collision_envelopes_disabled_arms=={'left'}
    assert config.home_recover_stopped and runtime.direct_joint_moves
    config.path_validation_enabled=True
    with pytest.raises(live.StopExperiment):live.safety_assert(runtime)

def test_abstain_schema_accepts_null_action():
    import jsonschema
    jsonschema.validate({**INTENT,'decision':'ABSTAIN','grasp_px':None,'target_px':None,'motion_type':None},live.PLAN)

def test_actual_request_is_isolated_and_saved(tmp_path):
    from cloth_agent.experiments import one_shot_goal_full_episode_v1 as ep
    wire=ep.planner
    im=tmp_path/'before.png';Image.new('RGB',(64,64),'pink').save(im)
    captured=[]
    def direct(folder,body,headers,timeout):
        captured.append(body);live.save(folder/'request_payload.json',body);return INTENT,{}
    with patch.object(ep,'direct_once',side_effect=direct),patch.object(wire,'auth_headers',return_value={}):
        live.request(tmp_path/'model',live.BASE,'image size 64x64',[im],live.PLAN)
    body=captured[0]
    assert body['instructions']==live.BASE
    assert body['tools']==[] and len(body['input'])==1 and 'previous_response_id' not in body
    assert len(body['input'][0]['content'])==2
    assert live.read(tmp_path/'model/audited_payload.json')==body
    assert not any(x in body['instructions'] for x in ['四端部','对侧牵拉','法线','Reveal'])

@pytest.mark.parametrize('choice',['NOT_DISTINCT','UNKNOWN'])
def test_novelty_assessment_does_not_replace_scene_safety_gate(tmp_path,choice):
    image=tmp_path/'before.png';Image.new('RGB',(64,64)).save(image)
    runtime=SimpleNamespace(capture_current=Mock(side_effect=live.StopExperiment('scene check')), _execute_operations=Mock())
    with patch.object(live,'prepare',return_value=([],{},'hash')),patch.object(live,'show_review'),patch.object(live,'review_mechanism',return_value={'choice':choice}) as review:
        with pytest.raises(live.StopExperiment,match='scene check'):
            live.dispatch_once(runtime,tmp_path,{'dispatches':1},tmp_path,SimpleNamespace(image=image),PLAN,INTENT,base_plan=PLAN)
    review.assert_called_once();runtime._execute_operations.assert_not_called()

def test_final_iteration_records_skill_without_seventh_action(tmp_path):
    state,n,t,_,out=episode(tmp_path,['IMPROVED']*5+['NO_CLEAR_IMPROVEMENT'])
    assert (state['dispatches'],n,t)==(6,6,6)
    assert state['termination']=='ITERATION_LIMIT'
    assert not live.read(out/'round_005/skill_transition.json')['has_next_round']

def test_worsening_retires_skill_and_returns_to_base_without_reliable_replacement(tmp_path):
    state,n,t,_,out=episode(tmp_path,['NO_CLEAR_IMPROVEMENT','WORSENED']+['IMPROVED']*4,
        teacher_decision=['NEW_SKILL']+['NO_NEW_SKILL']*5)
    rows=live.read(out/'rounds.json')
    assert [r['mode'] for r in rows[:3]]==['BASE','LEARNED','BASE']
    assert rows[1]['skill_transition']['retired_due_to_worsening']
    assert rows[1]['skill_transition']['next_mode']=='BASE'
    assert (state['dispatches'],t)==(6,6)

def test_no_motion_is_possible_missed_grasp_not_confirmed_failure(tmp_path):
    state,_,t,_,out=episode(tmp_path,['NO_VISIBLE_CHANGE']*6,teacher_decision='NO_NEW_SKILL')
    rows=live.read(out/'rounds.json')
    assert all(r['skill_transition']['possible_missed_grasp'] and not r['skill_transition']['missed_grasp_confirmed'] for r in rows)
    assert (state['dispatches'],t)==(6,6)

def test_startup_reset_no_input_and_not_iteration_budget(tmp_path):
    import numpy as np
    from cloth_agent.experiments import one_shot_goal_full_episode_v1 as ep
    from cloth_agent.dual_arm.observation import Observation
    from cloth_agent import preference_one_shot
    config=SimpleNamespace(path_validation_enabled=True,collision_envelopes_disabled_arms={'left','right'},arms={'left':1,'right':2})
    before=SimpleNamespace(image=tmp_path/'before.png');Image.new('RGB',(64,64)).save(before.image)
    fresh=SimpleNamespace(sample=lambda *a:np.zeros(3))
    actual=Mock(return_value={'status':'EXECUTED'})
    runtime=SimpleNamespace(config=config,_execute_operations=actual,arrival=Mock(),capture_current=lambda p:fresh)
    runtime._live_skill_policy=live.runtime_policy(runtime)
    operations=[('gripper',{'joints':{'left':[0]*6,'right':[0]*7},'arm':'left','action':'open'})]
    state={'dispatches':0,'reset_manipulations':0,'reset_motion_dispatches':0}
    def direct(folder,body,*a):live.save(folder/'request_payload.json',body);return {},{}
    def initialize(rt,folder,headers,*a,**kw):
        folder.mkdir()
        live.save(folder/'grasp.json',{'pixel_xy':[10,10],'world_xyz_mm':[0,0,0]})
        body={'instructions':ep.reset.PROMPT+' Return only schema-valid JSON.','tools':[],'input':[{'content':[{},{}]}]}
        ep.direct_once(folder/'region_request',body,{},1)
        for name in ['before_perception','lift_and_drop','return_perception']:rt._execute_operations(operations,folder/name)
        return {'status':'EXECUTED'}
    with patch.object(ep,'initialize',side_effect=initialize),patch.object(ep,'direct_once',side_effect=direct) as wire,patch.object(ep.planner,'auth_headers',return_value={}),patch('builtins.input',side_effect=AssertionError('no input')),patch.object(Observation,'load',return_value=before),patch.object(preference_one_shot,'verify_scene'):
        live.startup_reset(runtime,tmp_path,state)
        assert ep.direct_once is wire
    assert runtime._execute_operations is actual
    assert state['dispatches']==0
    assert state['reset_manipulations']==1
    assert actual.call_count==3

def test_only_arm6_sdk_connection_arm7_is_static(tmp_path):
    from cloth_agent.dual_arm import execution,static_arm
    left=SimpleNamespace(ip='arm6');right=SimpleNamespace(ip='arm7')
    runtime=SimpleNamespace(config=SimpleNamespace(arms={'left':left,'right':right},static_arm_obstacles={'right':{'joint_angles_deg':[0]*7}}),models={'right':object()},cancel=threading.Event())
    connected=Mock();obstacle=Mock()
    with patch.object(execution,'XArmConnection',return_value=connected) as sdk,patch.object(static_arm,'StaticArmObstacle',return_value=obstacle) as static:
        result=live.connect_with_diagnostics(runtime,tmp_path)
    sdk.assert_called_once_with(left,runtime.cancel,recover_stopped=False)
    static.assert_called_once()
    assert result=={'left':connected,'right':obstacle}
    assert live.read(tmp_path/'connection_audit.json')['arms'][1]['live_feedback'] is False

def test_connection_error_names_arm_and_endpoint(tmp_path):
    from cloth_agent.dual_arm import execution
    runtime=SimpleNamespace(config=SimpleNamespace(arms={'left':SimpleNamespace(ip='arm6')}),cancel=threading.Event())
    with patch.object(execution,'XArmConnection',side_effect=Exception('connect socket failed')):
        with pytest.raises(live.StopExperiment,match='left.*arm6'):live.connect_with_diagnostics(runtime,tmp_path)
    assert live.read(tmp_path/'connection_audit.json')['status']=='FAILED'

@pytest.mark.parametrize('choice',['IMPROVED','NO_CLEAR_IMPROVEMENT','WORSENED','NO_VISIBLE_CHANGE','UNKNOWN'])
def test_outcome_is_model_evidence_not_default_answer(tmp_path,choice):
    before=SimpleNamespace(image=tmp_path/'before.png')
    after=SimpleNamespace(image=tmp_path/'after.png')
    for ob in [before,after]:Image.new('RGB',(64,64),'pink').save(ob.image)
    with patch('builtins.input',side_effect=AssertionError('no input')),patch.object(live,'request',return_value={'choice':choice,'note':'image evidence','unknowns':'occlusion'}) as model:
        result=live.observe_outcome(tmp_path,before,after,INTENT)
    assert result['choice']==choice and result['human_verified'] is False
    assert model.call_args.args[3]==[before.image,after.image]
    assert live.read(tmp_path/'outcome_review.json')['source']=='model_visual_assessment'

def test_noninteractive_execute_entrypoint(tmp_path):
    output=tmp_path/'run'
    def run(args):live.save(args.output/'status.json',{'termination':'ITERATION_LIMIT'})
    with patch.object(live,'run',side_effect=run),patch('builtins.input',side_effect=AssertionError('no input')):
        assert live.main(['--execute','--output',str(output)])==0


def test_unknown_continues_and_clears_stale_target_without_claiming_failure(tmp_path):
    state,n,t,goals,out=episode(tmp_path,['NO_CLEAR_IMPROVEMENT','UNKNOWN']+['UNKNOWN']*4,
        teacher_decision=['NEW_SKILL']+['NO_NEW_SKILL']*5)
    assert state['termination']=='ITERATION_LIMIT'
    assert (state['dispatches'],n,t)==(6,6,6)
    assert goals[1]['target_region']=='original sleeve'
    assert all(goal is None for goal in goals[2:])
    rows=live.read(out/'rounds.json')
    assert all(row['outcome']['choice']=='UNKNOWN' for row in rows[1:])
    assert not any(row['skill_transition']['retired_due_to_worsening'] for row in rows[1:])


@pytest.mark.parametrize('distance,accepted',[(19.99,False),(20,True),(30,True),(40,True),(40.01,False),(float('nan'),False)])
def test_experiment_pull_distance_uses_metric_xy(tmp_path,distance,accepted):
    live.save(tmp_path/'points.json',[{'world_xyz_mm':[0,0,100]},{'world_xyz_mm':[distance,0,200]}])
    live.save(tmp_path/'motion_profile.json',{'lift_height_mm':20,'max_translation_mm':150})
    if accepted:
        live.check_pull_distance(tmp_path)
        profile=live.read(tmp_path/'motion_profile.json')
        assert profile['min_translation_mm']==20 and profile['max_translation_mm']==40
        assert profile['lift_height_mm']==20
    else:
        with pytest.raises(live.StopExperiment,match='牵拉距离'):live.check_pull_distance(tmp_path)
    assert live.read(tmp_path/'pull_distance_check.json')['passed']==accepted


def test_planner_keeps_multisentence_reason_but_rejects_bad_coordinates(tmp_path):
    before=SimpleNamespace(image=tmp_path/'before.png')
    Image.new('RGB',(64,64)).save(before.image)
    value={**INTENT,'short_reason':'第一句。第二句。'}
    with patch.object(live,'request',return_value=value):
        plan,_=live.planner(tmp_path,before)
    assert plan['short_reason']==value['short_reason']
    assert live.read(tmp_path/'plan.json')['short_reason']==value['short_reason']
    with patch.object(live,'request',return_value={**value,'grasp_px':[100,10]}):
        with pytest.raises(ValueError,match='coordinate invalid'):live.planner(tmp_path,before)
