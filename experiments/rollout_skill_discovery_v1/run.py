"""Offline independent direct GPT calls; no runtime construction or robot connection."""
import sys,json,hashlib,base64,time
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
TRAIN=['reports/one_shot_goal_continuous_v1_20261021_110536_963158048/episode_0001','reports/one_shot_goal_continuous_v1_20261021_113055_912326188/episode_0001','reports/automatic_top5_batch_20261023_154316_327624848/episode_0001']
VALID=['reports/one_shot_goal_continuous_v1_20261021_102354_200485125/episode_0001','reports/one_shot_goal_continuous_v1_20261021_115018_870388145/episode_0001','reports/automatic_top5_batch_20261023_173631_672851883/episode_0003']
def read(p):return json.loads(Path(p).read_text())
def save(p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def obj(d):return dict(type='object',properties=d,required=list(d),additionalProperties=False)
S={'type':'string'}
def enum(*v):return {'type':'string','enum':list(v)}
SK=obj({k:S for k in ['trigger','grasp_selection','direction_selection','expected_visible_effect']})
EV=obj({'initial_difficulty':S,'trajectory_uncertainty':S,'operations':{'type':'array','items':obj({'id':S,'trigger':enum('YES','NO','UNKNOWN'),'before_evidence':S,'action_conforms':enum('YES','NO','UNKNOWN'),'expected_effect_observed':enum('YES','NO_CLEAR_EVIDENCE','UNKNOWN'),'after_evidence':S,'uncertainty':S})},'overall_assessment':S})
BASE='你是独立的离线视觉研究评价者。只依据本次请求文字与图像，不使用工具、历史会话或外部知识库。任务是展开桌面上的T-shirt。动作lift_pull_place表示在当前图Pa处抓取，抬起、平移至Pb后释放。坐标为原始Before图的像素，左上为原点，x向右y向下。只输出指定JSON。'
def collect(paths,prefix,outcomes):
 rows=[];imgs=[];manifest=[]
 for j,path in enumerate(paths,1):
  rid=f'{prefix}{j}';ops=[]
  for r in sorted((ROOT/path).glob('round_*')):
   if not (r/'execution.json').exists():continue
   e=read(r/'execution.json'); plan=read(r/'dispatch_intent.json').get('plan') if (r/'dispatch_intent.json').exists() else (read(r/'plan.json') if (r/'plan.json').exists() else None)
   rec={'id':f'{rid}/{r.name}','execution_status':e.get('status'),'physical_execution':e.get('physical_execution',False),'action':{k:plan[k] for k in ['decision','grasp_px','target_px','motion_type','grasp_angle_deg'] if k in plan} if plan else None}
   if (r/'points.json').exists():rec['measured_points']=read(r/'points.json')
   if (r/'geometry.json').exists():
    g=read(r/'geometry.json');rec['geometry']={k:v for k,v in g.items() if k in ['distance_mm','rpy_world_deg','primitive']}
   for field,rel in [('before','before/rgb.png'),('after','after/rgb.png')]:
    p=r/rel
    if field=='after' and not outcomes:continue
    if p.exists():
     rec[field+'_attachment']=len(imgs)+1;rec[field+'_size']=list(Image.open(p).size);imgs.append(p)
     manifest.append({'id':rec['id'],'phase':field,'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
    else:rec[field+'_missing']=True
   ops.append(rec)
  rows.append({'rollout':rid,'operations_in_original_order':ops})
 return rows,imgs,manifest

def sheets(rows,imgs,folder):
 for roll in rows:
  ops=roll['operations_in_original_order'];canvas=Image.new('RGB',(1040,max(1,len(ops))*330),'white');d=ImageDraw.Draw(canvas)
  for i,o in enumerate(ops):
   d.text((5,i*330+3),o['id']+' '+str(o['action']),fill='black')
   for phase,x in [('before',0),('after',520)]:
    if phase+'_attachment' in o:
     im=Image.open(imgs[o[phase+'_attachment']-1]).convert('RGB');im.thumbnail((515,290));canvas.paste(im,(x,i*330+30))
  folder.mkdir(parents=True,exist_ok=True);canvas.save(folder/(roll['rollout']+'.jpg'))

def prepare():
 if (OUT/'protocol.json').exists():return
 rows,imgs,man=collect(TRAIN,'T',True)
 save(OUT/'train_data.json',rows);save(OUT/'train_manifest.json',man);sheets(rows,imgs,OUT/'review')
 save(OUT/'protocol.json',{'created':now(),'teacher_rollouts':TRAIN,'validation_rollouts':VALID,'model':'gpt-6-astra','effort':'high','planned_calls':4,'teacher_calls':1,'revision_budget':0,'validation_calls':3,'validation':'retrospective independent full-rollout review, NOT blinded prediction','teacher_expert_prompt':False,'historical_planner_reasons':False,'reference_group':False,'robot_execution':False,'selection':'Train episodes from three distinct run groups with multiple existing verified unfolding gains; entire episode order retained. Validation uses three separate existing Development groups; outcomes opened only after skill freeze.','selection_bias':'Purposive positive-progress training selection, no population success-rate inference','skill_limit':'four fields combined <=150 Chinese characters; no fixed coordinates','transport':'Existing direct ChatGPT Responses transport; no CLI/session memory/project skills/tools','scientific_retries':0})
 save(OUT/'context_audit.json',{'teacher_inputs':'Neutral task/interface + sanitized chronological Before/actual Action/After only; no labels, no source planner rationale, no expert prompt, no previous skills','request_audit':'Exact serialized HTTP payload saved before dispatch; attachments hashes saved; no project tool context supplied','provider_internal_context':'not inspectable','coordinator':'Has previous conversation context; isolated Teacher does not receive it','validation_limitation':'After and later states visible in retrospective request; trigger/visible-effect association, not independent predictive accuracy'})
 print('Prepared training:',[(r['rollout'],len(r['operations_in_original_order'])) for r in rows])

def call(name,prompt,rows,imgs,manifest,schema):
 from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once
 from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as wire
 folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'parsed.json').exists():return read(folder/'parsed.json')
 if (folder/'attempt.json').exists():raise RuntimeError('Existing attempt; no automatic scientific retry: '+name)
 text=prompt+'\n按附件编号对应的完整顺序记录：\n'+json.dumps(rows,ensure_ascii=False)
 content=[{'type':'input_text','text':text}]+[{'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode(),'detail':'high'} for p in imgs]
 body={'model':'gpt-6-astra','instructions':BASE,'input':[{'role':'user','content':content}],'tools':[],'tool_choice':'none','parallel_tool_calls':False,'reasoning':{'effort':'high'},'text':{'format':{'type':'json_schema','name':'rollout_skill_study','strict':True,'schema':schema}},'store':False,'stream':True}
 save(folder/'attachments.json',manifest);(folder/'instructions.txt').write_text(BASE);(folder/'prompt.txt').write_text(text);save(folder/'schema.json',schema)
 save(folder/'context_check.json',{'instructions_exact':body['instructions']==BASE,'one_user_message':len(body['input'])==1,'tools_empty':body['tools']==[],'no_previous_response_id':'previous_response_id' not in body,'no_expert_prompt_or_prior_skill_files_supplied':True,'attachment_count':len(imgs),'audited_at':now()})
 print('START',name,'images',len(imgs),flush=True)
 result,metrics=direct_once(folder,body,wire.auth_headers(),600)
 save(folder/'parsed.json',result);print('END',name,'seconds',metrics['wall_latency_s'],flush=True);return result

def teacher():
 rows,imgs,man=collect(TRAIN,'T',True)
 prompt='观察三条真实完整Rollout，自行判断哪些动作及其真实结果最值得学习。不要假定每个动作有效；保留失败、无明显变化和遮挡不确定性。从执行前可见结构自主归纳最多一条能用于新状态的Skill，包含适用状态、抓点选择、牵拉方向和预期可见效果。四字段合计不超过150个汉字，不允许用固定坐标或固定操作顺序代替规律。不提供专家操作规则，不要求发现特定规律。证据不足可trigger=NO_RELIABLE_SKILL，其余字段说明不足。只输出一条Skill，不输出多条候选或轨迹总结。'
 r=call('teacher',prompt,rows,imgs,man,SK)
 n=sum(len(v) for v in r.values());hanz=sum('\u4e00'<=c<='\u9fff' for v in r.values() for c in v)
 if hanz>150:raise RuntimeError('Skill over 150 Chinese characters; no rewrite permitted')
 save(OUT/'skill_frozen.json',r);save(OUT/'freeze.json',{'at':now(),'sha256':sha(OUT/'skill_frozen.json'),'characters':n,'han_characters':hanz,'revised':False})
 print(json.dumps(r,ensure_ascii=False),flush=True)

def validate():
 assert sha(OUT/'skill_frozen.json')==read(OUT/'freeze.json')['sha256']
 skill=read(OUT/'skill_frozen.json')
 for j,path in enumerate(VALID,1):
  rows,imgs,man=collect([path],f'V{j}_',True)
  sheets(rows,imgs,OUT/'review')
  prompt='这是冻结Skill在未参与归纳的真实Rollout中的事后检验，不修订规则。Skill：'+json.dumps(skill,ensure_ascii=False)+'\n逐项检查每个真实操作：trigger只能引用该步Before可见证据，不能因After有效就倒推触发；动作是否符合抓点/方向；After是否出现预期可见效果（必须说明具体变化，区分整体平移与局部展开）。未执行/After缺失时效果UNKNOWN。NO/不符合不自动等于无收益。保留初始难度、动作参数差异、遮挡、抓取层数和深度不确定性，不作因果或统计显著性声明；同状态不要求完全相同。每个记录id恰好评一次，保持顺序。'
  r=call(f'validation_{j}',prompt,rows,imgs,man,EV)
  assert [o['id'] for o in r['operations']]==[o['id'] for o in rows[0]['operations_in_original_order']]
 report()

def report():
 vs=[read(OUT/f'validation_{j}/parsed.json') for j in range(1,4)]
 rows=[o for v in vs for o in v['operations']];matched=[o for o in rows if o['trigger']=='YES' and o['action_conforms']=='YES']
 summary={'operations':len(rows),'trigger_yes':sum(o['trigger']=='YES' for o in rows),'trigger_unknown':sum(o['trigger']=='UNKNOWN' for o in rows),'trigger_and_conform':len(matched),'matched_effects':{k:sum(o['expected_effect_observed']==k for o in matched) for k in ['YES','NO_CLEAR_EVIDENCE','UNKNOWN']},'validation':vs,'skill':read(OUT/'skill_frozen.json'),'limitations':['Retrospective, not outcome-blind prediction','Single Teacher, no prior-only/action-only baseline','Positive-progress selection bias','No new robot execution; no measured task improvement']}
 save(OUT/'results.json',summary)
 costs=[]
 for name in ['teacher','validation_1','validation_2','validation_3']:costs.append({'request':name,**read(OUT/name/'model_metrics.json')})
 save(OUT/'costs.json',{'calls':costs,'total_wall_s':sum(x['wall_latency_s'] for x in costs),'input_tokens':sum(x.get('usage',{}).get('input_tokens',0) for x in costs),'output_tokens':sum(x.get('usage',{}).get('output_tokens',0) for x in costs),'currency_cost':None})
 print(json.dumps({k:v for k,v in summary.items() if k!='validation'},ensure_ascii=False),flush=True)

if __name__=='__main__':
 prepare()
 stage=sys.argv[1]
 if stage=='prepare':pass
 elif stage=='teacher':teacher()
 elif stage=='validate':validate()
 elif stage=='report':report()
