"""Continuous RSI exploration; reset once, one checked action per cycle."""
from pathlib import Path
import argparse,importlib.util,json,signal,sys,html,time
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('feedback',ROOT/'experiments/live_skill_discovery_01/run.py')
feedback=importlib.util.module_from_spec(spec);spec.loader.exec_module(feedback)
io=feedback.io
S=io.S;obj=io.obj;enum=io.enum
STRINGS={'type':'array','items':S}
CLAIM=obj({'id':S,'region':S,'predicted_change':S,'supporting_observation':S,'falsifying_observation':S})
CANDIDATE=obj({'id':S,'grasp_px':io.POINT,'target_px':io.POINT,'mechanism_hypothesis':S,
 'visible_basis':S,'target_material':S,'target_deformation':S,'other_regions_motion':S,
 'fold_change':S,'distinction_from_others':S,'predictions':{'type':'array','items':CLAIM,'minItems':2,'maxItems':5},
 'task_progress_basis':S,'hypotheses_to_distinguish':S,'history_comparison':S,'uncertainty':S})
GENERATE_SCHEMA=obj({'decision':enum('CANDIDATES','NO_VALID_EXPLORATION'),'reason':S,
 'local_goal':S,'goal_completion_observation':S,'history_assessment':S,
 'candidates':{'type':'array','items':CANDIDATE,'maxItems':3}})
GENERATE='''训练阶段只提出操作假设，不执行动作。阅读当前RGB和已有真实经验记录，选择当前未完成的一个局部目标，提出2–3种有可见依据、具有不同物理作用机制的候选。不能把几像素或小角度微调称为不同机制，不预设任何正确操作。历史UNKNOWN不是失败或零收益；抓空/滑移只是未证实可能性，不应直接否定操作机制。对同一目标重复无明确进展应优先考虑不同机制；有明确进展可继续原机制。说明哪些历史状态可比、哪些不可比。
接口仅单次lift_pull_place，Pa/Pb为当前原图整数像素；固定Home方向，10–150 mm实测XY范围。仅Arm6，不增加动作阶段或机器人能力。保守考虑可见工作空间，安全由Host检查，不能自称通过安全检查。
每个候选明确抓点、落点、目标材料、目标及其他区域的预测运动、折叠变化、可证伪观察。给出2–5条带唯一id的可观察预测。机制解释只是假设。候选区分依据应是不同的作用位置/相对运动关系或折叠释放方式，而非名称不同。不能安全且有依据地产生至少两种不同方案时NO_VALID_EXPLORATION。不要编造成功概率，不生成Skill。'''
SELECT_SCHEMA=obj({'decision':enum('SELECT','NO_VALID_EXPLORATION'),'selected_id':S,
 'task_progress_reason':S,'information_gain_reason':S,'diversity_reason':S,
 'distinct_mechanisms_supported':{'type':'boolean'},'why_not_others':S,'unknowns':S})
SELECT='''只选择一个已冻结且Host检查PASS的候选，不修改动作或预测。综合Task Progress、Information Gain、Exploration Diversity，引用具体预测及历史记录，不按自报成功概率排序。核对候选是否有可见依据的不同机制；不同名称、微调抓点或小角度变化不够。当前机制持续有明确进展时允许继续；同一目标反复无明确进展时优先不同机制，但UNKNOWN不得视为失败。不能从证据判断安全或机制差异，或者没有有效候选时NO_VALID_EXPLORATION。未执行候选只有预测，没有真实结果。'''
VERDICT=obj({'prediction_id':S,'verdict':enum('SUPPORTED','REFUTED','UNKNOWN'),'observed_evidence':S,'uncertainty':S})
UPDATE_SCHEMA=obj({'prediction_checks':{'type':'array','items':VERDICT},
 'relative_motion':S,'target_fold_change':S,'whole_garment_motion':S,'adverse_folds':S,
 'acquisition':enum('CONFIRMED','UNCERTAIN'),'acquisition_evidence':S,
 'new_information':S,'new_information_status':enum('OBSERVATIONAL_EVIDENCE','HYPOTHESIS_ONLY','UNKNOWN'),
 'hypothesis_update':obj({'mechanism':S,'state_conditions':S,'supported':STRINGS,'refuted':STRINGS,'unresolved':STRINGS,'next_test':S,'next_action_selection_change':S}),
 'limitations':S})
UPDATE='''这次只真实执行了一个候选。逐条比较该候选事先冻结的预测与独立第一层观察、原始Before/After和执行记录；每条预测只能SUPPORTED/REFUTED/UNKNOWN，必须给出具体区域形变证据。UNKNOWN不是反驳或零收益；命令完成不证明抓住布料；整体拖移不等于局部展开。先判断布料是否被有效带动；抓取不确定时不能把无变化归因于物理假设错误。
分别描述相对运动、目标折叠、整衣移动、新的不利折叠。说明获得了此前没有的什么信息：只支持观察关系时不能宣称因果机制已证实。生成一条简短假设更新，保留支持、反驳、未知项、下次值得检验的内容以及这将如何改变下一次动作选择。不因一次失败永久排除机制，不生成Skill，不规划或执行下一动作。不得为未执行候选填写真实效果。'''
DONE_SCHEMA=obj({'decision':enum('DONE','NOT_DONE','UNKNOWN'),'visible_evidence':S,'remaining_folds':S,'visibility_limits':S})
DONE_PROMPT='仅根据当前RGB判断整件T-shirt展平任务是否完成，不提出动作或Skill。DONE要求整件衣物主要区域可见、衣身和袖部已展开，没有明确残留的大面积折叠、自覆盖或卷折；轻微自然细皱不单独判未完成。只完成某个局部目标不足以DONE。存在明确残留折叠为NOT_DONE；出画、遮挡、曝光或层别不清而无法核验整件衣物时UNKNOWN，不能猜测隐藏区域。记录具体可见证据、残余折叠及观察限制。'

DEFAULT_HISTORY=['live_skill_discovery_01_20261028_220104_597305','live_skill_discovery_01_20261028_221501_600721','live_skill_discovery_01_20261028_225412_484535']

def plan_for(c):
 return {'decision':'ACTION','grasp_px':c['grasp_px'],'target_px':c['target_px'],
 'motion_type':'lift_pull_place','grasp_angle_deg':None,'short_reason':c['mechanism_hypothesis']}

def load_memory(path=None):
 if path:
  record=io.read(path)
  return {'previous_memory':record,'source_path':str(path),'source_sha256':io.sha(path)}
 rows=[]
 for name in DEFAULT_HISTORY:
  for folder in sorted((ROOT/'reports'/name).glob('round_*')):
   if not (folder/'reason_analysis.json').exists():continue
   execution=io.read(folder/'execution.json')
   if execution.get('status')!='EXECUTED':continue
   a=io.read(folder/'reason_analysis.json');p=io.read(folder/'plan.json')
   rows.append({'id':name+'/'+folder.name,'action':{k:p.get(k) for k in ['grasp_px','target_px','motion_type']},
    'state_changes':{k:{'change':v['visible_change'],'correspondence':v['correspondence']} for k,v in io.read(folder/'physical_observation.json')['regions'].items()} if (folder/'physical_observation.json').exists() else {},
    'interpretation':{k:{'label':a[k]['label'],'evidence':a[k]['evidence']} for k in ['target_completed','local_progress','adverse_change']},
    'labels_human_verified':False,'historical_predictions_available':False,'mechanism_not_yet_categorized':True})
 return {'observed_experiences':rows[-12:],'warning':'same evolving cloth; no independent causal validation; UNKNOWN unresolved'}

def validate_candidates(value,size):
 from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as wire
 if value['decision']=='NO_VALID_EXPLORATION':return
 cs=value['candidates']
 if len(cs) not in [2,3] or len({c['id'] for c in cs})!=len(cs):raise ValueError('expected 2–3 unique candidates')
 for c in cs:
  wire.validate_plan(plan_for(c),size,require_grasp_rotation=True,require_single_sentence=False)
  ids=[p['id'] for p in c['predictions']]
  if len(set(ids))!=len(ids):raise ValueError('duplicate prediction ids')
 for i,c in enumerate(cs):
  for d in cs[:i]:
   if max(abs(c[k][j]-d[k][j]) for k in ['grasp_px','target_px'] for j in [0,1])<=8:
    raise ValueError('near-identical coordinates do not establish different mechanisms')

def prepare(runtime,before,c,folder):
 import numpy as np
 from unittest.mock import patch
 from cloth_agent import one_shot_orientation
 from cloth_agent.experiments.one_shot_goal_full_episode_v1 import prepare_action
 plan=plan_for(c);io.save(folder/'plan.json',plan)
 original_validate=one_shot_orientation.validate_plan
 def validate_reason(*args,**kwargs):
  kwargs['require_single_sentence']=False
  return original_validate(*args,**kwargs)
 # Orientation validates the plan again using its own imported validator.
 # Relax only prose sentence count; preserve schema, coordinates and motion checks.
 with patch.object(one_shot_orientation,'validate_plan',validate_reason):
  operations,initial=prepare_action(runtime,before,plan,folder)
 kind,last=operations[-1];end=last.end if kind=='move' else {k:np.asarray(v) for k,v in last['joints'].items()}
 owner,target=runtime.perception_target();old=getattr(runtime,'open_at_home_on_return',False)
 runtime.open_at_home_on_return=True
 try:_,source=runtime._preparation_program(end,owner,target)
 finally:runtime.open_at_home_on_return=old
 (folder/'return_motion.py').write_text(source)
 operations+=runtime._plan_preparation(source,end)
 native=[{'kind':k,'data':v.to_dict() if k=='move' else v} for k,v in operations]
 io.save(folder/'native_plan.json',native)
 return operations,initial,io.digest(native)

def validate_selection(selection,candidates,host):
 if selection['decision']!='SELECT':return None
 if not selection['distinct_mechanisms_supported']:return None
 selected=next((c for c in candidates if c['id']==selection['selected_id']),None)
 if selected is None or host[selected['id']]['status']!='PASS':raise ValueError('selected candidate absent or rejected by Host')
 return selected

def authorize_execution(root,binding):
 # Record session authorization without fabricating a per-action human review.
 io.save(root/'execution_authorization.json',{'mode':'automatic',
  'source':'user explicitly requested automatic approval',
  'per_action_human_review':False,'binding':binding,'binding_sha256':io.digest(binding),
  'at':datetime.now().isoformat()})
 print('检查通过后自动执行；动作记录：'+str(root/'review.html'),flush=True)


def make_checked_runtime(config):
    # Extra verification is local to this experiment; no production/config files changed.
    import types
    from cloth_agent.codex_native_runtime import NativeRuntime
    runtime=io.make_runtime(config)
    runtime.config.path_validation_enabled=True
    runtime._plan_preparation=types.MethodType(NativeRuntime._plan_preparation,runtime)
    runtime._live_skill_policy=io.runtime_policy(runtime)
    return runtime


def run(args):
 from PIL import Image
 from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as wire
 from cloth_agent.preference_one_shot import verify_scene
 from cloth_agent.gamepad_collect import TrajectoryRecorder
 from cloth_agent import codex_recovery
 root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
 state={'status':'PREPARING','dispatches':0,'max_actions':1,'reset_dispatches':0};io.save(root/'status.json',state)
 runtime=None;handlers={};selected=None;update=None;selection=None;generated=None
 try:
  memory=load_memory(args.memory)
  memory['current_state_context']={'reset_this_cycle':not getattr(args,'no_reset',False),'history_is_prior_states':True}
  io.save(root/'memory_before.json',memory)
  io.save(root/'protocol.json',{'max_actions':1,'reset':not getattr(args,'no_reset',False),'max_reset_actions':0 if getattr(args,'no_reset',False) else 1,'reset_grasp':'original full-episode region prompt and sampler; measured XY disk around detected patch center; radius bound uniformly sampled in [20,40] mm','production_prompt_modified':False,
   'specific_human_approval_required':False,'execution_authorization':'session-authorized automatic; existing checks unchanged','model':'gpt-6-astra','effort':'high','new_skill_deployment':False})
  runtime=make_checked_runtime(args.config)
  def interrupted(*_):runtime.cancel.set();raise KeyboardInterrupt()
  for sig in [signal.SIGINT,signal.SIGTERM]:handlers[sig]=signal.signal(sig,interrupted)
  io.connect_stationary(runtime,root)
  io.save(root/'runtime_policy.json',io.runtime_policy(runtime))
  if not getattr(args,'no_reset',False):
   state['status']='STARTUP_RESET';io.save(root/'status.json',state)
   reset_spec=importlib.util.spec_from_file_location('rsi_reset_stage',Path(__file__).with_name('reset_stage.py'))
   reset_stage=importlib.util.module_from_spec(reset_spec);reset_spec.loader.exec_module(reset_stage)
   reset_stage.initialize(runtime,root,state,io,authorize_execution)
  state['status']='GENERATING_CANDIDATES';io.save(root/'status.json',state)
  before=runtime.capture_current(root/'before')
  size=list(Image.open(before.image).size)
  done=io.request(root/'done_check',DONE_PROMPT,'判断当前整件衣物是否已展开。',[before.image],DONE_SCHEMA)
  io.save(root/'done_check.json',done)
  if done['decision']=='DONE':
   state['status']='DONE_RESET_REQUIRED'
   return
  generated=io.request(root/'generation',GENERATE,json.dumps({'image_size':size,'memory':memory},ensure_ascii=False),[before.image],GENERATE_SCHEMA)
  validate_candidates(generated,size)
  io.save(root/'frozen_predictions.json',generated);frozen_hash=io.sha(root/'frozen_predictions.json')
  if generated['decision']!='CANDIDATES':state['status']='NO_VALID_EXPLORATION';return
  prepared={};host={}
  for index,c in enumerate(generated['candidates']):
   folder=root/f'candidate_{index:02d}';folder.mkdir()
   wire.overlay(before.image,plan_for(c),folder/'action.png',c['id'])
   try:
    operations,initial,h=prepare(runtime,before,c,folder)
    prepared[c['id']]=(operations,initial,h,folder)
    host[c['id']]={'status':'PASS','native_sha256':h,'full_path_validation':runtime.config.path_validation_enabled}
   except Exception as exc:
    host[c['id']]={'status':'REJECT','error':f'{type(exc).__name__}: {exc}'}
   io.save(folder/'host_check.json',host[c['id']])
  selection=io.request(root/'selection',SELECT,json.dumps({'frozen_candidates':generated,'host_checks':host,'memory':memory},ensure_ascii=False),[before.image],SELECT_SCHEMA)
  selected=validate_selection(selection,generated['candidates'],host)
  if selected is None:state['status']='NO_VALID_EXPLORATION';return
  operations,initial,native_hash,folder=prepared[selected['id']]
  binding={'before':io.sha(before.image),'predictions':frozen_hash,'selection':io.digest(selection),'native_plan':native_hash,'candidate_id':selected['id'],'memory':io.sha(root/'memory_before.json')}
  io.save(root/'selected.json',{'candidate':selected,'binding':binding,'selection':selection})
  (root/'review.html').write_text('<meta charset="utf-8"><h1>一次RSI探索：自动执行记录</h1><img width="1000" src="'+str(folder.relative_to(root))+'/action.png"><pre>'+html.escape(json.dumps({'goal':generated['local_goal'],'candidate':selected,'selection':selection,'checks':host[selected['id']],'runtime_policy':io.runtime_policy(runtime)},ensure_ascii=False,indent=2))+'</pre><p>未执行候选仅为假设；当前配置的完整路径验证状态以上述记录为准。</p>')
  state['status']='READY_TO_EXECUTE';io.save(root/'status.json',state)
  authorize_execution(root,binding)
  fresh=runtime.capture_current(root/'pre_dispatch')
  verify_scene(before,fresh,plan_for(selected),root,automatic=False)
  codex_recovery.check_stationary(runtime,before.meta['joints_deg'],root/'stationary_check.json')
  for path,expected in [(root/'frozen_predictions.json',frozen_hash),(before.image,binding['before']),(root/'memory_before.json',binding['memory'])]:
   if io.sha(path)!=expected:raise io.StopExperiment('规划后输入变化')
  if io.digest(io.read(folder/'native_plan.json'))!=native_hash:raise io.StopExperiment('规划后轨迹变化')
  runtime.arrival(initial);io.safety_assert(runtime)
  if state['dispatches']!=0:raise io.StopExperiment('一次动作预算已用')
  recorder=TrajectoryRecorder(runtime,folder)
  try:
   recorder.start();recorder.stage('automatic_one_exploration');recorder.check()
   state.update(status='EXECUTING',dispatches=1);io.save(root/'status.json',state)
   io.save(root/'dispatch_intent.json',{'binding':binding,'physical_execution':True,'no_automatic_retry':True})
   try:execution=runtime._execute_operations(operations,folder)
   except BaseException as exc:
    io.save(root/'execution.json',{'status':'EXECUTION_EXCEPTION','physical_execution':True,'error':str(exc)});raise
   io.save(root/'execution.json',execution)
   if execution['status']!='EXECUTED':raise io.StopExperiment('执行异常，不算布料操作失败')
   recorder.check()
  finally:recorder.close()
  after=runtime.capture_current(root/'after')
  observation=io.request(root/'observation',feedback.OBSERVATION_PROMPT,'第一张Before，第二张After；只记录可见物理变化。',[before.image,after.image],feedback.OBSERVATION_SCHEMA)
  io.save(root/'physical_observation.json',observation)
  update=io.request(root/'hypothesis_update',UPDATE,json.dumps({'selected_candidate':selected,'original_goal':generated['local_goal'],
   'observation':observation,'execution':execution,'memory_before':memory},ensure_ascii=False),[before.image,after.image],UPDATE_SCHEMA)
  ids=[x['prediction_id'] for x in update['prediction_checks']]
  if len(ids)!=len(set(ids)) or set(ids)!={p['id'] for p in selected['predictions']}:raise ValueError('prediction checks must cover exactly the selected candidate')
  io.save(root/'memory_after.json',{'previous_memory':str(root/'memory_before.json'),'previous_sha256':binding['memory'],
   'latest_experiment':str(root),'selected':selected,'observation':observation,'update':update,
   'observed_experiences':memory.get('observed_experiences',memory.get('previous_memory',{}).get('observed_experiences',[]))[-12:],
   'hypothesis_records':(memory.get('previous_memory',{}).get('hypothesis_records',[])+[{'experiment':str(root),'candidate':selected,'update':update}])[-8:],
   'unexecuted_candidate_ids':[c['id'] for c in generated['candidates'] if c['id']!=selected['id']],
   'unexecuted_outcomes':'UNKNOWN_UNTESTED','skill_ready':False,'permanent_exclusion':False})
  state['status']='COMPLETE_ONE_CYCLE'
 except BaseException as exc:
  import traceback
  (root/'traceback.txt').write_text(traceback.format_exc())
  state.update(status='STOPPED',error=f'{type(exc).__name__}: {exc}')
  if runtime is not None and runtime.coordinator is not None:
   try:runtime.stop()
   except Exception as err:state['stop_error']=str(err)
 finally:
  if runtime:
   try:runtime.close()
   except Exception as exc:state['close_error']=str(exc)
  for sig,handler in handlers.items():signal.signal(sig,handler)
  io.save(root/'status.json',state)
  report='本次状态：'+state['status']+'\n\n实际下发次数：'+str(state['dispatches'])+'\n\n'
  if update:report+='新信息：'+update['new_information']+'\n\n证据层次：'+update['new_information_status']+'\n\n下一次动作选择应如何改变：'+update['hypothesis_update']['next_action_selection_change']+'\n\n下次值得检验：'+update['hypothesis_update']['next_test']+'\n\n限制：'+update['limitations']
  elif state['status']=='DONE_RESET_REQUIRED':
   report+='当前视觉判断为DONE；未下发展开动作，下一轮先reset。模型判断不等于人工验证。'
  else:
   report+='未获得完整验证结果，不宣称学到新知识。'+state.get('error','')
   if state['status']=='NO_VALID_EXPLORATION':
    explanation=selection if selection is not None else generated
    report+='\n\n停止依据：\n'+json.dumps(explanation,ensure_ascii=False,indent=2)
    io.save(root/'memory_after.json',{'previous_memory':str(root/'memory_before.json'),
      'physical_knowledge_updated':False,'unexecuted_candidates':generated['candidates'] if generated else [],
      'outcomes':'UNKNOWN_UNTESTED','selection':selection,'permanent_exclusion':False,'skill_ready':False})
  (root/'report.md').write_text(report+'\n\n本轮结束；未执行候选无真实效果标签，未部署Skill。\n')
  print(report+'\n记录：'+str(root),flush=True)

def resume_memory(previous,destination):
 """Read a paused run without replaying any saved action or inventing feedback."""
 previous=previous.resolve()
 rounds=sorted(previous.glob('round_*'))
 if not rounds:rounds=[previous]
 latest=rounds[-1];status=io.read(latest/'status.json')
 if status.get('dispatches',0):
  execution=latest/'execution.json'
  if not execution.exists() or io.read(execution).get('status')!='EXECUTED':
   raise ValueError('末轮动作未确认完整执行；不能自动续跑部分执行的动作')
 complete=[d for d in rounds if (d/'memory_after.json').exists() and io.read(d/'status.json')['status']=='COMPLETE_ONE_CYCLE']
 if complete:
  memory=io.read(complete[-1]/'memory_after.json')
 else:
  saved=io.read(latest/'memory_before.json')
  memory=saved.get('previous_memory',saved)
 if status['status']!='COMPLETE_ONE_CYCLE':
  interrupted={'source':str(latest),'status':status,'interpretation':'incomplete analysis; effects unresolved; never replay action'}
  for name in ['selected','execution','physical_observation']:
   path=latest/(name+'.json')
   if path.exists():interrupted[name]=io.read(path)
  memory={**memory,'interrupted_round':interrupted}
 io.save(destination,memory)
 return destination,status['status']=='DONE_RESET_REQUIRED'


def run_continuous(args):
 from copy import copy
 root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
 memory=args.memory
 reset_pending=not bool(args.resume_from)
 if args.resume_from:
  memory,reset_pending=resume_memory(args.resume_from,root/'resume_memory.json')
 state={'status':'RUNNING','completed_cycles':0,'dispatches':0,'reset_dispatches':0,
        'max_rounds':args.max_rounds or None,'resume_from':str(args.resume_from) if args.resume_from else None}
 io.save(root/'status.json',state)
 try:
  index=0
  while args.max_rounds==0 or index<args.max_rounds:
   cycle=copy(args);cycle.output=root/f'round_{index:03d}';cycle.memory=memory
   cycle.no_reset=not reset_pending
   state['current_round']=str(cycle.output);io.save(root/'status.json',state)
   print(f'RSI ROUND {index:03d} reset={not cycle.no_reset}',flush=True)
   run(cycle)
   result=io.read(cycle.output/'status.json')
   state['dispatches']+=result['dispatches'];state['reset_dispatches']+=result['reset_dispatches']
   if result.get('close_error') or result.get('stop_error'):
    state.update(status='STOPPED',reason=result);break
   if result['status']=='DONE_RESET_REQUIRED':
    reset_pending=True
    state['done_detections']=state.get('done_detections',0)+1
    io.save(root/'status.json',state)
    index+=1
    continue
   if result['status']!='COMPLETE_ONE_CYCLE':
    state.update(status='STOPPED',reason=result);break
   reset_pending=False
   memory=cycle.output/'memory_after.json';io.read(memory)
   state['completed_cycles']+=1;state['latest_memory']=str(memory)
   io.save(root/'status.json',state)
   index+=1
  else:state['status']='ROUND_LIMIT_REACHED'
 except KeyboardInterrupt:
  state['status']='USER_INTERRUPTED'
 except Exception as exc:
  state.update(status='STOPPED',error=f'{type(exc).__name__}: {exc}')
 finally:
  io.save(root/'status.json',state)
  print('连续探索已停止：'+state['status']+'\n记录：'+str(root),flush=True)
 return 0 if state['status']=='ROUND_LIMIT_REACHED' else 1


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path)
 p.add_argument('--memory',type=Path);p.add_argument('--config',type=Path,default=ROOT/'config/dual_arm.local.json')
 p.add_argument('--max-rounds',type=int,default=0,help='0 = unlimited; stop on rejection/error')
 p.add_argument('--resume-from',type=Path,help='paused run or completed cycle; fresh observation, never replay old actions')
 args=p.parse_args(argv)
 if args.max_rounds<0:p.error('--max-rounds must be >=0')
 if args.memory and args.resume_from:p.error('--memory and --resume-from are mutually exclusive')
 if not args.execute:print('连续探索；开场reset一次，每轮一个动作并传递假设记录。默认不限轮数，Ctrl+C停止；拒绝或异常停止。');return 0
 args.output=args.output or ROOT/'reports'/('rsi_exploration_01_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
 return run_continuous(args)
if __name__=='__main__':raise SystemExit(main())
