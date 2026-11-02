"""15 bounded independent model calls: observe, explain, compress, predict, plan, reveal."""
from common import *
from data import cases,packet
from transport import call
import static_checks

OBS=obj({'operations':{'type':'array','items':obj({'id':S,'reliable_local_intention':S,'OBSERVED':SS,'INFERRED':SS,'UNKNOWN':SS,'insufficient_goal_evidence':SS})},'common_phenomena':{'type':'array','items':obj({'phenomenon':S,'evidence_ids':SS,'epistemic_status':{'type':'string','enum':['OBSERVED','INFERRED','UNKNOWN']}})},'observation_limits':SS})
HYP=obj({'chosen_failure':S,'failure_evidence_ids':SS,'competing_hypotheses':{'type':'array','minItems':1,'maxItems':3,'items':obj({'id':S,'cause':S,'explained_observations':SS,'inconsistent_or_missing_observations':SS,'discriminating_observation':S})},'selected_hypothesis_id':S,'why_most_valuable':S,'specific_alternative_operation':obj({'change_from_executed_action':S,'how_to_choose_Pa':S,'how_to_choose_Pb':S,'unchanged_parameters':S,'illustrative_discovery_case_action':S}), 'predicted_observable_if_effective':SS,'falsifying_observation':SS,'UNKNOWN':SS})
SKILL=obj({k:S for k in ['Trigger','Action','Expected Effect','Failure Condition','Evidence','Uncertainty']})
PRED=obj({'id':S,'applicability':{'type':'string','enum':['APPLICABLE','NOT_APPLICABLE','UNKNOWN']},'applicability_evidence':SS,'OBSERVED':SS,'INFERRED':SS,'predictions':{'type':'array','minItems':1,'maxItems':4,'items':obj({'id':S,'target_region_prediction':S,'possible_comoving_regions':S,'support_criterion':S,'contradiction_criterion':S,'confidence':{'type':'number'}})},'does_executed_action_match_alternative':S,'UNKNOWN':SS})
POINT={'type':['array','null'],'items':{'type':'integer'},'minItems':2,'maxItems':2}
PLAN=obj({'id':S,'decision':{'type':'string','enum':['ACTION','STOP','ABSTAIN']},'Pa':POINT,'Pb':POINT,'motion_type':{'type':'string','enum':['lift_pull_place','none']},'goal':S,'reason':S,'rule_trigger':{'type':'string','enum':['YES','NO','UNKNOWN','NOT_PROVIDED']},'trigger_evidence':SS,'rule_effect_on_action':S,'UNKNOWN':SS})
EVAL=obj({'id':S,'overall_verdict':{'type':'string','enum':['SUPPORTED','CONTRADICTED','INCONCLUSIVE','NOT_APPLICABLE']},'applicability':S,'prediction_checks':{'type':'array','items':obj({'prediction_id':S,'verdict':{'type':'string','enum':['SUPPORTED','CONTRADICTED','INCONCLUSIVE','NOT_APPLICABLE']},'OBSERVED':SS,'INFERRED':SS,'UNKNOWN':SS})},'hypothesis_counterexample':S,'alternative_operation_effectiveness':S,'planner_trigger_audit':obj({'B_trigger_correctness':{'type':'string','enum':['SUPPORTED_BY_PRE','CONTRADICTED_BY_PRE','INCONCLUSIVE']},'pre_only_evidence':SS,'unnecessary_abstain':S,'action_change_matches_rule':S,'attribution_limit':S}),'evidence_limitations':SS})

def packet_content(c,full):
 f=OUT/'cases'/c['id'];facts=read(f/('facts.json' if full else 'pre.json'));imgs=[f/'before.png']
 if full:
  if (f/'process.jpg').exists():imgs.append(f/'process.jpg')
  imgs.append(f/'after.png')
 text='案例 '+c['id']+'，附件顺序：'+','.join(p.name for p in imgs)+'。before/after 1280×720；process每格320×180，按从左到右、从上到下时间顺序，时间为UTC；无法分辨的接触细节不得臆造。\n'+json.dumps(facts,ensure_ascii=False)+'\n'
 return text,imgs

def discovery():
 verify_freezes();prompt=INTERFACE+'\n阶段一，只做观察，不提出或修改策略规则。对每次操作回答：可靠输入能否确定希望发生的局部变化？若不能，真实意图写UNKNOWN，另在INFERRED中明确说可能目的。直接观察有哪些布料变化？哪些表明任务未充分完成？不同操作中有哪些值得继续研究的共同现象？分别输出OBSERVED、INFERRED、UNKNOWN，不把原因假设写成事实。\n';imgs=[]
 for c in cases(True):t,ims=packet_content(c,True);prompt+=t;imgs+=ims
 obs=call('01_observe',prompt,imgs,OBS)
 if not (OUT/'observations.json').exists():save(OUT/'observations.json',obs);freeze('observations',[OUT/'observations.json'])
 prompt=INTERFACE+'\n阶段二。从本轮观察中自主选择一个最值得解释的未达目标现象。提出最多三个竞争性物理原因：各自解释什么、哪些观察不符合或缺证据。选一个最值得检验的原因；给具体替代操作，包括Pa/Pb选择方式与相对实际动作的改变、保持什么参数不变、成功时应观察什么、什么结果可推翻。不能只说更好抓点/避免错误。不要求证明。不要新增多条技能。\n本轮冻结观察原文：\n'+(OUT/'teacher_calls/01_observe/raw_output.txt').read_text()+'\n原始发现证据复核：\n';imgs=[]
 for c in cases(True):t,ims=packet_content(c,True);prompt+=t;imgs+=ims
 hyp=call('02_hypothesize',prompt,imgs,HYP)
 if not (OUT/'hypotheses.json').exists():save(OUT/'hypotheses.json',hyp);freeze('hypotheses',[OUT/'hypotheses.json'])
 prompt=INTERFACE+'\n阶段三。只把本轮选中的核心操作假设压缩成一条简短候选Skill。不要改变其核心机制，不要引入新原因或额外技能。完整填Trigger、Action、Expected Effect、Failure Condition、Evidence、Uncertainty；尽量每字段一到两句，不照搬Episode专用绝对坐标或固定图像方向。保留实际需要的时间上下文/不可判断条件，不能为了容易触发就放宽证据。清楚说明未验证的替代动作。只输出一条规则。\n观察：'+(OUT/'teacher_calls/01_observe/raw_output.txt').read_text()+'\n竞争性假设与选定操作：'+(OUT/'teacher_calls/02_hypothesize/raw_output.txt').read_text()
 skill=call('03_compress',prompt,[],SKILL)
 if not (OUT/'candidate_skill.json').exists():save(OUT/'candidate_skill.json',skill);freeze('candidate_skill',[OUT/'candidate_skill.json',OUT/'teacher_calls/03_compress/raw_output.txt'])

def predict():
 verify_freezes();assert (OUT/'freezes/candidate_skill.json').exists()
 for i,c in enumerate(cases(False)):
  packet(c,False);t,ims=packet_content(c,False)
  prompt=INTERFACE+'\n独立留出预测阶段：只有动作前观测与实际执行的动作数值，不提供结果、过程、夹爪反馈。根据冻结假设和规则，判断适用条件；证据缺失填UNKNOWN，条件明确不满足填NOT_APPLICABLE。预测目标区域可能变化、可能共同移动的其他区域，并给可观察的支持/反驳标准。限制2–3个具体预测，每个带ID。没有时间证据时不能虚构此前操作。说明历史动作是否真的实施了候选替代动作。预测一经保存不能改。\n冻结假设：'+json.dumps(read(OUT/'hypotheses.json'),ensure_ascii=False)+'\n唯一冻结规则：'+json.dumps(read(OUT/'candidate_skill.json'),ensure_ascii=False)+'\n'+t
  p=call(f'{i+4:02d}_predict_{c["id"]}',prompt,ims,PRED)
  target=OUT/'predictions'/f'{c["id"]}.json'
  if not target.exists():save(target,p)
 if not (OUT/'freezes/predictions.json').exists():freeze('predictions',list((OUT/'predictions').glob('*.json')))

def static_audit(c,p):
 result=static_checks.check(c,p);geo=read(OUT/'cases'/c['id']/'pre.json')['geometry']['primitive'];cfg=read(OUT/'static_environment.json');xyz=result.get('values',{}).get('world_points_mm');extra={}
 if xyz:
  pa,pb=np.asarray(xyz);T=np.linalg.inv(np.asarray(cfg['world_from_base_mm']));z=max(pa[2],pb[2])+geo['lift_height_mm'];poses=[pa+[0,0,geo['grasp_offset_mm']],pb+[0,0,geo['place_offset_mm']],[pa[0],pa[1],z],[pb[0],pb[1],z]];violations=[]
  for n,pos in zip(['grasp','place','lift','transfer'],poses):
   base=(T@np.r_[pos,1])[:3]
   for j,axis in enumerate('xyz'):
    lo=cfg['workspace']['min_mm'][j];hi=cfg['workspace']['max_mm'][j]
    if lo is not None and base[j]<lo:violations.append({'point':n,'axis':axis,'mm':float(base[j]),'min':lo})
    if hi is not None and base[j]>hi:violations.append({'point':n,'axis':axis,'mm':float(base[j]),'max':hi})
  extra={'endpoint_bounds_pass':not violations,'violations':violations}
  if violations:result['status']='FAIL'
 result['endpoint_bounds']=extra;result['scope']='Offline format, depth/metric and endpoint boundary screen only; no live robot/Host/IK/collision/path validation.'
 return result

def planners():
 assert (OUT/'freezes/predictions.json').exists();verify_freezes()
 for i,c in enumerate(cases(False)):
  f=OUT/'cases'/c['id'];primitive=read(f/'pre.json')['geometry']['primitive']
  base='''你是单步T-shirt展开Planner。目标是展开并铺平，减少可见折叠与重叠。只看本次1280×720当前RGB，生成一个lift_pull_place(Pa,Pb)，原点左上、x向右、y向下，Pa/Pb为图内整数坐标，Pa在可见布料上。XY移动10–150mm，图像约1.3px/mm仅为粗略尺度参照，不代表实测验证。没有历史动作或后图；只能提出未执行建议。可输出STOP或ABSTAIN，但需具体依据。不要调用工具。输出目的、理由、是否触发附加规则和不确定性。没有附加规则填NOT_PROVIDED。\n'''
  base+='固定基础原语参数：'+json.dumps(primitive,ensure_ascii=False)+'\n归档TCP边界（不可把毫米直接当像素）：'+json.dumps(read(OUT/'static_environment.json')['workspace'])+'\n案例ID：'+c['id']+'\n'
  for arm in (['A','B'] if i%2==0 else ['B','A']):
   name=f'planner_{c["id"]}_{arm}';prompt=base
   if arm=='B':prompt+='唯一新增候选规则（待验证，保持其适用条件，不强行触发）：\n'+json.dumps(read(OUT/'candidate_skill.json'),ensure_ascii=False)
   p=call(name,prompt,[f/'before.png'],PLAN);folder=OUT/'planner_comparison'/c['id'];folder.mkdir(parents=True,exist_ok=True)
   if not (folder/f'{arm}.json').exists():save(folder/f'{arm}.json',p);save(folder/f'{arm}_static.json',static_audit(c,p));annotate(f/'before.png',p['Pa'],p['Pb'],folder/f'{arm}.png')
  if not (folder/'comparison.json').exists():save(folder/'comparison.json',static_checks.compare(read(folder/'A.json'),read(folder/'B.json')))
 if not (OUT/'freezes/planners.json').exists():freeze('planners',[f for f in (OUT/'planner_comparison').rglob('*') if f.is_file()])

def reveal():
 verify_freezes();assert (OUT/'freezes/predictions.json').exists();assert (OUT/'freezes/planners.json').exists();results=[]
 for c in cases(False):
  packet(c,True);t,ims=packet_content(c,True);pair=OUT/'planner_comparison'/c['id']
  prompt=INTERFACE+'\n你是独立评价请求。下列预测在揭示结果前已冻结，不得改写预测或规则。用真实后图与过程图逐条判SUPPORTED/CONTRADICTED/INCONCLUSIVE；规则触发条件明确不成立时标NOT_APPLICABLE。没有执行候选替代操作时，只能检验相关现象预测，不能证明替代操作更好。现象符合不是机制证实。具体反例应要求满足对应前提。\n此外核查Planner B声称的规则触发：只允许引用before图与其可用单帧信息，不能用后图补上B当时缺少的触发证据。判断是否强行适用、有无无法证实的弃权成本、动作变化是否与规则一致。A/B未执行，不能评价物理好坏。\n冻结假设：'+json.dumps(read(OUT/'hypotheses.json'),ensure_ascii=False)+'\n冻结规则：'+json.dumps(read(OUT/'candidate_skill.json'),ensure_ascii=False)+'\n原始冻结预测：'+json.dumps(read(OUT/'predictions'/f'{c["id"]}.json'),ensure_ascii=False)+'\nPlanner A：'+json.dumps(read(pair/'A.json'),ensure_ascii=False)+'\nPlanner B：'+json.dumps(read(pair/'B.json'),ensure_ascii=False)+'\n'+t
  prompt+='\n评价口径澄清（在揭示任何留出结果之前记录）：冻结假设中的illustrative_discovery_case_action是D1示例坐标，不自动成为不同案例的绝对坐标要求。请按冻结Skill的实际相对操作与Trigger判断；如果预测把示例坐标当作跨案例必要条件，在applicability中指出该过度绑定，但不得改写预测或放宽Skill明写的历史、初态及参数条件。分别判断Trigger可验证性、历史Action是否匹配、具体预测有无观察支持。'
  d=call('evaluate_'+c['id'],prompt,ims,EVAL);results.append(d)
 if not (OUT/'heldout_evaluation.json').exists():save(OUT/'heldout_evaluation.json',{'cases':results,'independent_calls':True,'candidate_skill_sha256':sha(OUT/'candidate_skill.json'),'alternative_actions_executed':False});freeze('heldout_evaluation',[OUT/'heldout_evaluation.json'])

def context_audit():
 m=read(OUT/'teacher_calls/01_observe/input_manifest.json');cmd=m['command'];flags=cmd[2:cmd.index('--output-schema')];flags=[x for x in flags if x not in ['--ephemeral','--skip-git-repo-check','--json']];i=flags.index('--color');del flags[i:i+2]
 r=subprocess.run(['codex']+flags+['debug','prompt-input','Isolation audit only. No experiment data.'],capture_output=True,text=True,timeout=180)
 (OUT/'context_audit_raw.json').write_text(r.stdout);(OUT/'context_audit_stderr.txt').write_text(r.stderr);d=json.loads(r.stdout) if r.returncode==0 else [];text='\n'.join(v.get('text','') for x in d for v in x.get('content',[]))
 save(OUT/'context_audit.json',{'at':utc(),'returncode':r.returncode,'custom_base_present':BASE in text,'forbidden_markers':{s:s in text for s in ['hem_sleeve_pull_rule','Rule Mining','Policy Evolution','clothAgent_real','Expert Prompt']},'generic_skills_present':'Available skills' in text,'generic_tool_metadata':any(x.get('type')=='additional_tools' for x in d),'boundary':'Model may see generic CLI metadata, no project or clothing rules; zero actual tool events required.'})
 print('CONTEXT_AUDIT_DONE')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['discover','predict','plan','reveal','audit']);a=p.parse_args();{'discover':discovery,'predict':predict,'plan':planners,'reveal':reveal,'audit':context_audit}[a.stage]()
