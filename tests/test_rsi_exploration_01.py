import importlib.util,json
from pathlib import Path
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('explore',ROOT/'experiments/rsi_exploration_01/run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def candidate(id='a',x=20):
 return {'id':id,'grasp_px':[x,20],'target_px':[x+10,50],'mechanism_hypothesis':'待检验假设',
 'visible_basis':'可见边缘','target_material':'局部','target_deformation':'预测变化','other_regions_motion':'预测运动',
 'fold_change':'预测折叠变化','distinction_from_others':'相对运动关系不同','predictions':[
 {'id':'p1','region':'局部','predicted_change':'位置变化','supporting_observation':'缝线变化','falsifying_observation':'缝线不变'},
 {'id':'p2','region':'其他区域','predicted_change':'保持','supporting_observation':'保持','falsifying_observation':'移动'}],
 'task_progress_basis':'可见依据','hypotheses_to_distinguish':'待检验','history_comparison':'历史未知','uncertainty':'层数未知'}

def test_schema_and_candidate_validation():
 import jsonschema
 value={'decision':'CANDIDATES','reason':'依据','local_goal':'局部目标','goal_completion_observation':'可见标志','history_assessment':'不确定','candidates':[candidate(),candidate('b',60)]}
 jsonschema.validate(value,m.GENERATE_SCHEMA);m.validate_candidates(value,[100,100])
 for bad in [[candidate()],[candidate(),candidate('b',22)],[candidate(),candidate()]]:
  with pytest.raises(ValueError):m.validate_candidates({**value,'candidates':bad},[100,100])

def test_host_rejected_candidate_cannot_be_selected():
 sel={'decision':'SELECT','selected_id':'a','distinct_mechanisms_supported':True}
 with pytest.raises(ValueError):m.validate_selection(sel,[candidate()],{'a':{'status':'REJECT'}})
 assert m.validate_selection({**sel,'distinct_mechanisms_supported':False},[candidate()],{'a':{'status':'PASS'}}) is None
 assert m.validate_selection(sel,[candidate()],{'a':{'status':'PASS'}})['id']=='a'

def test_automatic_execution_records_bound_authorization_without_input(tmp_path):
 binding={'candidate_id':'a','native':'hash','predictions':'frozen'}
 with patch('builtins.input',side_effect=AssertionError('must not request input')):
  m.authorize_execution(tmp_path,binding)
 record=m.io.read(tmp_path/'execution_authorization.json')
 assert record['binding']==binding and record['binding_sha256']==m.io.digest(binding)
 assert record['mode']=='automatic' and record['per_action_human_review'] is False
 assert not (tmp_path/'approval.json').exists()

def test_history_contains_actual_effects_not_expert_prompts():
 memory=m.load_memory()
 assert len(memory['observed_experiences'])==11
 for row in memory['observed_experiences']:
  assert row['labels_human_verified'] is False
  assert 'short_reason' not in row['action']
  assert row['historical_predictions_available'] is False

def test_dry_run_does_not_run_hardware():
 with patch.object(m,'run',side_effect=AssertionError('hardware')):assert m.main([])==0


def reset_module():
 spec=importlib.util.spec_from_file_location('reset_stage_test',ROOT/'experiments/rsi_exploration_01/reset_stage.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module

def test_experiment_enables_existing_path_checks_and_preparation_checks_only_locally():
 from types import SimpleNamespace
 runtime=SimpleNamespace(config=SimpleNamespace(path_validation_enabled=False,collision_envelopes_disabled_arms={'left'},home_recover_stopped=True))
 with patch.object(m.io,'make_runtime',return_value=runtime):assert m.make_checked_runtime(Path('unused')) is runtime
 assert runtime.config.path_validation_enabled
 assert runtime._live_skill_policy['path_validation_enabled']
 assert runtime._plan_preparation.__self__ is runtime
 assert runtime.config.collision_envelopes_disabled_arms=={'left'}


@pytest.mark.parametrize('radius',[20.,30.,40.])
def test_reset_reuses_full_episode_prompt_and_sampler(tmp_path,radius):
 from types import SimpleNamespace
 from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
 reset=reset_module()
 folder=tmp_path/'initialization/before';folder.mkdir(parents=True)
 (folder/'rgb.png').write_bytes(b'offline image placeholder')
 calls=[]
 original_sample=episode.reset.sample_grasp
 body={'instructions':episode.reset.PROMPT+' Return only schema-valid JSON.',
       'text':{'format':{'schema':episode.reset.REGION}}}
 def transport(directory,payload,headers,timeout):
  calls.append(payload)
  assert payload is body
  return {'region':'unchanged'},{'wall_latency_s':1}
 def initialize(runtime,folder,headers,timeout,**kwargs):
  assert kwargs=={'center_radius_mm':radius,'press_after_drop':False}
  assert episode.reset.sample_grasp is original_sample
  return episode.direct_once(folder/'region_request',body,headers,timeout)
 execute=lambda *args: None
 runtime=SimpleNamespace(_execute_operations=execute)
 with patch.object(episode,'direct_once',transport),patch.object(episode,'initialize',initialize),patch.object(episode.planner,'auth_headers',return_value={}),patch.object(reset.random.SystemRandom,'uniform',return_value=radius):
  result=reset.initialize(runtime,tmp_path,{'reset_dispatches':0},m.io,m.authorize_execution)
  assert episode.direct_once is transport
 assert result==({'region':'unchanged'},{'wall_latency_s':1})
 assert len(calls)==1
 assert runtime._execute_operations is execute
 assert episode.reset.sample_grasp is original_sample
 assert m.io.read(tmp_path/'initialization/region_request/audited_payload.json')==body


def test_prepare_relaxes_only_orientation_reason_sentence_count(tmp_path):
 from cloth_agent import one_shot_orientation
 from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
 original=one_shot_orientation.validate_plan
 class ReachedGeometry(Exception):pass
 def prepare_action(runtime,before,plan,folder):
  one_shot_orientation.validate_plan(plan,[100,100],require_grasp_rotation=True)
  raise ReachedGeometry()
 c=candidate();c['mechanism_hypothesis']='第一条假设。第二条预测。'
 with patch.object(episode,'prepare_action',prepare_action):
  with pytest.raises(ReachedGeometry):m.prepare(None,None,c,tmp_path)
  bad={**c,'grasp_px':[101,20]}
  with pytest.raises(ValueError,match='coordinate invalid'):m.prepare(None,None,bad,tmp_path)
 assert one_shot_orientation.validate_plan is original
 with pytest.raises(ValueError,match='exceeds one sentence'):
  original(m.plan_for(c),[100,100],require_grasp_rotation=True)


@pytest.mark.parametrize('resume',[False,True])
def test_continuous_carries_memory_resets_once_and_stops_on_rejection(tmp_path,resume):
 from types import SimpleNamespace
 previous=tmp_path/'previous'
 m.io.save(previous/'status.json',{'status':'COMPLETE_ONE_CYCLE'})
 m.io.save(previous/'memory_after.json',{'saved':True})
 args=SimpleNamespace(output=tmp_path/'new',memory=None,resume_from=previous if resume else None,max_rounds=0)
 calls=[]
 def cycle(arg):
  calls.append(arg)
  m.io.save(arg.output/'status.json',{'status':'COMPLETE_ONE_CYCLE' if len(calls)<3 else 'NO_VALID_EXPLORATION',
      'dispatches':1 if len(calls)<3 else 0,'reset_dispatches':int(not arg.no_reset)})
  m.io.save(arg.output/'memory_after.json',{'round':len(calls)})
 with patch.object(m,'run',side_effect=cycle):assert m.run_continuous(args)==1
 assert len(calls)==3
 assert [x.no_reset for x in calls]==[resume,True,True]
 assert calls[0].memory==(args.output/'resume_memory.json' if resume else None)
 assert calls[1].memory==calls[0].output/'memory_after.json'
 assert calls[2].memory==calls[1].output/'memory_after.json'
 state=m.io.read(args.output/'status.json')
 assert state['completed_cycles']==2 and state['dispatches']==2
 assert state['reset_dispatches']==int(not resume)


def test_done_requests_one_reset_next_round(tmp_path):
 from types import SimpleNamespace
 args=SimpleNamespace(output=tmp_path/'run',memory=None,resume_from=None,max_rounds=3)
 calls=[]
 def cycle(arg):
  calls.append(arg)
  status='DONE_RESET_REQUIRED' if len(calls)==1 else 'COMPLETE_ONE_CYCLE'
  m.io.save(arg.output/'status.json',{'status':status,'dispatches':int(status=='COMPLETE_ONE_CYCLE'),'reset_dispatches':int(not arg.no_reset)})
  if status=='COMPLETE_ONE_CYCLE':m.io.save(arg.output/'memory_after.json',{})
 with patch.object(m,'run',side_effect=cycle):assert m.run_continuous(args)==0
 assert [c.no_reset for c in calls]==[False,False,True]
 assert m.io.read(args.output/'status.json')['done_detections']==1


def test_resume_interrupted_analysis_preserves_executed_action(tmp_path):
 root=tmp_path/'old';a=root/'round_000';b=root/'round_001'
 m.io.save(a/'status.json',{'status':'COMPLETE_ONE_CYCLE','dispatches':1})
 m.io.save(a/'memory_after.json',{'hypothesis_records':[{'old':'evidence'}]})
 m.io.save(b/'status.json',{'status':'EXECUTING','dispatches':1})
 m.io.save(b/'execution.json',{'status':'EXECUTED'})
 m.io.save(b/'selected.json',{'candidate':'frozen'})
 m.io.save(b/'physical_observation.json',{'visible':'change'})
 path,reset=m.resume_memory(root,tmp_path/'resumed.json')
 saved=m.io.read(path)
 assert not reset and saved['interrupted_round']['selected']=={'candidate':'frozen'}
 assert saved['hypothesis_records']==[{'old':'evidence'}]
 m.io.save(b/'execution.json',{'status':'EXECUTION_EXCEPTION'})
 with pytest.raises(ValueError,match='未确认完整执行'):m.resume_memory(root,tmp_path/'bad.json')
