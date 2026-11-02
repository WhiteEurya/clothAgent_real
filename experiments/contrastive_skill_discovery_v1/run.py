"""Offline experiment. No robot imports. Requests use isolated ephemeral CLI sessions."""
from pathlib import Path
import json,hashlib,subprocess,time,sys,shutil,concurrent.futures,math,difflib
from datetime import datetime,timezone
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parent
DATA=OUT.parent/'contrastive_skill_dataset_v1'
CLEAN=Path('/tmp/csd_v1_isolated');CLEAN.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def obj(d):return {'type':'object','properties':d,'required':list(d),'additionalProperties':False}
S={'type':'string'}
def enum(*v):return {'type':'string','enum':list(v)}
def arr(v):return {'type':'array','items':v}
SK=obj({k:arr(S) if k=='evidence_ids' else S for k in ['skill','trigger','grasp_selection','direction_selection','expected_effect','evidence_ids','limitations']})
BASE='你是独立离线实验模型。仅使用本次请求文字与附件，不使用工具、文件、网络、技能、历史会话。任务是展开桌面上的T-shirt。动作接口为当前RGB像素坐标Pa抓点和Pb释放目标点，左上为原点，x向右y向下；机器人抓取Pa，抬起、平移到Pb、放下释放。只输出要求的JSON。'
TASK='最多自主提出一条可复用操作Skill；也可skill=NO_RELIABLE_SKILL。trigger只能依赖当前RGB可见结构；具体说明抓点与方向如何选择，不用固定历史坐标，不新增算法或多轮规划。skill+trigger+grasp_selection+direction_selection合计尽量不超过150汉字。不要仅总结成败，要尝试提出执行前可见结构如何指导选点和方向；不确定之处写入limitations。不能从相关性宣称因果。'
PRED=obj({'cases':arr(obj({'id':S,'trigger':enum('YES','NO','UNKNOWN'),'conforms':enum('YES','NO','UNKNOWN'),'prediction':enum('YES','NO_CLEAR_EVIDENCE','UNKNOWN'),'reason':S,'uncertainty':S}))})
POINT=arr({'type':'number'})
PLAN=obj({'Pa':POINT,'Pb':POINT,'grasp_region':S,'pull_direction':S,'skill_trigger':enum('YES','NO','UNKNOWN','BASE'),'reason':S})
REV=obj({'decision':enum('KEEP','REVISE'),'problem':S,'evidence':S,'changed_component':S,'generalization_reason':S,'final_skill':SK})
def flags(base):
 f=['-C',str(CLEAN),'-s','read-only','-c','approval_policy="never"','-c',f'model_instructions_file="{base}"','-c','developer_instructions=""','-c','project_doc_max_bytes=0','-c','web_search="disabled"','-c',f'model_catalog_json="{DATA}/local_model_catalog.json"','-c','features.skip_host_skill_discovery=true','-c','skills.config=[{path="/home/sja/.codex/skills/cloth-contour-normal-grasp/SKILL.md",enabled=false}]','-m','gpt-6-astra','-c','model_reasoning_effort="high"']
 for x in ['shell_tool','shell_snapshot','view_image','multi_agent','apps','plugins','hooks','memories','browser_use','computer_use','image_generation','skill_search','goals','sleep_tool','unbounded_connection_retries']:f+=['--disable',x]
 return f

def call(name,prompt,images,schema):
 folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'result.json').exists():
  assert read(folder/'result.json')['status']=='COMPLETE',name
  return read(folder/'parsed.json')
 assert not (folder/'attempt.json').exists(),f'Unfinished request {name}'
 (folder/'base.txt').write_text(BASE);(folder/'prompt.txt').write_text(prompt);save(folder/'schema.json',schema)
 attachments=[]
 for i,p in enumerate(images):
  dst=folder/'attachments'/f'{i+1:02d}.png';dst.parent.mkdir(exist_ok=True);shutil.copyfile(p,dst);attachments.append(dst)
 ff=flags(folder/'base.txt')
 assert '-p' not in ff and not any('dmxapi' in x for x in ff)
 # Audit the actual configured model-visible context before EVERY request.
 auditcmd=['codex']+ff+['debug','prompt-input',prompt]
 a=subprocess.run(auditcmd,capture_output=True,text=True,timeout=120,cwd=CLEAN)
 (folder/'context_raw.json').write_text(a.stdout);(folder/'context_stderr.txt').write_text(a.stderr)
 forbidden=['cloth-contour-normal-grasp','hem_sleeve_pull_rule','failure_driven_rsi','HEM_REVEAL','clothAgent_real','cloth-contour']
 hits=[x for x in forbidden if x in a.stdout]
 save(folder/'context_check.json',{'at':now(),'returncode':a.returncode,'forbidden_hits':hits,'context_sha256':sha(folder/'context_raw.json'),'attachments_audited_separately':True})
 if a.returncode or hits:raise RuntimeError('Context audit rejected '+name+str(hits))
 cmd=['codex','exec']+ff+['--ephemeral','--skip-git-repo-check','--json','--color','never','--output-schema',str(folder/'schema.json'),'-o',str(folder/'raw_output.txt')]
 for p in attachments:cmd+=['-i',str(p)]
 cmd+=['-']
 save(folder/'input_manifest.json',{'command':cmd,'prompt_sha256':sha(folder/'prompt.txt'),'attachments':[{'source':str(src),'file':str(dst.relative_to(OUT)),'sha256':sha(dst),'size':Image.open(dst).size} for src,dst in zip(images,attachments)],'context_sha256':sha(folder/'context_raw.json')})
 save(folder/'attempt.json',{'at':now(),'automatic_retries':'native provider default; not overridden'});print('START',name,flush=True)
 start=time.monotonic();terminal=None
 with (folder/'events.jsonl').open('w') as so,(folder/'stderr.txt').open('w') as se:
  p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=so,stderr=se,text=True,cwd=CLEAN,start_new_session=True);p.stdin.write(prompt);p.stdin.close()
  while p.poll() is None:
   ev=(folder/'events.jsonl').read_text()
   if '"type":"turn.completed"' in ev or '"type":"turn.failed"' in ev:
    terminal=terminal or time.monotonic()
    if (folder/'raw_output.txt').exists() or time.monotonic()-terminal>8:p.terminate();break
   if time.monotonic()-start>900:p.terminate();break
   time.sleep(.5)
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:p.kill();p.wait()
 es=[]
 for line in (folder/'events.jsonl').read_text().splitlines():
  try:es.append(json.loads(line))
  except ValueError:pass
 done=[e for e in es if e.get('type')=='turn.completed'];tool=[e for e in es if e.get('item',{}).get('type','reasoning') not in ['error','agent_message','reasoning']]
 ok=len(done)==1 and not tool and (folder/'raw_output.txt').exists()
 save(folder/'result.json',{'at':now(),'status':'COMPLETE' if ok else 'FAILED','wall_s':(terminal or time.monotonic())-start,'usage':done[0].get('usage') if done else None,'tool_events':tool,'exit_code':p.returncode,'model':'gpt-6-astra','effort':'high'})
 if not ok:raise RuntimeError('Request failed: '+name)
 r=read(folder/'raw_output.txt');save(folder/'parsed.json',r);print('END',name,flush=True);return r

def cases(split):
 # Never serialize split rows wholesale. Label field inaccessible to consumers below.
 return read(DATA/'splits'/f'{split}.json')['cases']
def inputs(split,outcomes=False):
 rows=[];images=[]
 for x in cases(split):
  im=DATA/x['Observation'];r={'id':x['id'],'action':x['Action'],'image_size':Image.open(im).size,'before_attachment':len(images)+1};images.append(im)
  if outcomes:
   r['after_attachment']=len(images)+1;images.append(DATA/x['NextObservation']);r['verified_effects']=x['OutcomeEvidence']
  rows.append(r)
 return rows,images

def init():
 if (OUT/'protocol.json').exists():return
 save(OUT/'protocol.json',{'created':now(),'model':'gpt-6-astra','effort':'high','teachers':['A','B','C'],'teacher_calls_each':2,'evaluation_calls_each':2,'replay_states':['E29','E30'],'replay_repetitions':2,'replay_conditions':['Base','A','B','C'],'planned_total_calls':28,'optional_D':False,'primary_label':'unfolding_gain','positive':'YES','low_benefit':'NO_CLEAR_EVIDENCE','unknown':'excluded from binary denominators, reported separately','metrics':['positive precision','positive recall','low-benefit recall','trigger applicability','prediction UNKNOWN','class-specific trigger and conformity coverage'],'revisions':'one equal budget; A/B only own blinded dev predictions and aggregate scores, never case outcomes; C own dev predictions and verified dev outcomes. No cross-arm feedback.','replay_selection':'first two Test IDs, before only; not used in skill learning; not claimed unseen by evaluation','stopping':'no scientific retries; no extra sampling based on performance','limitation':'orchestrator accidentally exposed to test labels before freeze; teachers/evaluators isolated; full experiment is not strictly orchestrator-blind','robot_execution':False})
 save(OUT/'context_audit.json',{'incident':{'at':now(),'stage':'preflight, before any teacher calls','description':'cat of source split manifests exposed embedded development/test labels to root orchestration conversation','affected':'root orchestration context','quarantine':'root is not a Teacher/evaluator; no root-generated rule or label-specific feedback included in requests','cannot_claim':'fully blinded orchestration'},'teacher_isolation':'ephemeral independent model requests, clean cwd, no tools, disabled project skills, per-request debug prompt-input audit','source_context_audit':read(DATA/'context_audit.json')})
 manifest={'sources':{f:sha(DATA/f) for f in ['experiences.json','effects_verified.json','source_audit.json','context_audit.json','splits/train.json','splits/development.json','splits/test.json']},'splits':{},'reference_excluded':True}
 groups=[]
 for s,n in [('train',20),('development',8),('test',8)]:
  cs=cases(s);assert len(cs)==n
  manifest['splits'][s]=[{'id':x['id'],'group':x['source']['group'],'before_sha256':sha(DATA/x['Observation'])} for x in cs];groups.append({x['source']['group'] for x in cs})
 assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
 save(OUT/'data_manifest.json',manifest)

def teacher(g):
 rows,images=inputs('train',g=='C') if g!='A' else ([],[])
 prompt=TASK+'\n'+({'A':'无历史经验，自主提出可能有用的Skill。','B':'只提供历史Before和实际动作，无结果。不要把被选择的动作假定为有效。','C':'提供Before、实际动作、After及已有四维效果和不确定性。寻找能解释不同结果的执行前可见结构特征。UNKNOWN不能当作失败。'}[g])+'\n附件与记录：'+json.dumps(rows,ensure_ascii=False)
 r=call(f'teacher_{g}/initial',prompt,images,SK);save(OUT/'skills'/f'{g}_initial.json',r)

def evaluate(split,version,groups="ABC"):
 rows,images=inputs(split)
 def evaluate_arm(g):
  sk=read(OUT/'skills'/f'{g}_{version}.json')
  p='对每条历史动作盲评：是否满足Skill触发条件、动作是否符合原则、依该Skill预测局部展开收益(YES=明确收益,NO_CLEAR_EVIDENCE=无明确收益,UNKNOWN=无法判断)。不触发不自动等于无收益；不符合也不自动等于失败。明确可见证据及不可判断事项。只依当前Before和动作，不能知道实际结果。\nSkill:'+json.dumps(sk,ensure_ascii=False)+'\n记录:'+json.dumps(rows,ensure_ascii=False)
  r=call(f'evaluation/{split}_{g}',p,images,PRED)
  assert sorted(x['id'] for x in r['cases'])==sorted(x['id'] for x in rows)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  list(pool.map(evaluate_arm,groups))
 if groups!='ABC':return
 save(OUT/f'{split}_predictions_frozen.json',{'at':now(),'hashes':{g:sha(OUT/f'evaluation/{split}_{g}/parsed.json') for g in 'ABC'}})
 score(split)

def score(split):
 assert (OUT/f'{split}_predictions_frozen.json').exists()
 truth={x['id']:x['OutcomeEvidence'] for x in cases(split)}
 result={'scored_at':now(),'primary_dimension':'unfolding_gain','unknown_not_failure':True,'arms':{}}
 for g in 'ABC':
  rs=read(OUT/f'evaluation/{split}_{g}/parsed.json')['cases']
  for r in rs:r['verified_effects']=truth[r['id']];r['truth']=truth[r['id']]['unfolding_gain']['value']
  def ratio(a,b):return {'numerator':a,'denominator':b,'rate':a/b if b else None}
  known=[r for r in rs if r['truth']!='UNKNOWN'];pos=[r for r in known if r['truth']=='YES'];neg=[r for r in known if r['truth']=='NO_CLEAR_EVIDENCE'];predpos=[r for r in known if r['prediction']=='YES']
  m={'positive_precision':ratio(sum(r['truth']=='YES' for r in predpos),len(predpos)),'positive_recall':ratio(sum(r['prediction']=='YES' for r in pos),len(pos)),'low_benefit_recall':ratio(sum(r['prediction']=='NO_CLEAR_EVIDENCE' for r in neg),len(neg)),'applicability':ratio(sum(r['trigger']=='YES' for r in rs),len(rs)),'prediction_unknown':ratio(sum(r['prediction']=='UNKNOWN' for r in rs),len(rs)),'truth_unknown':ratio(sum(r['truth']=='UNKNOWN' for r in rs),len(rs))}
  for label,subset in [('positive',pos),('low_benefit',neg)]:
   for field in ['trigger','conforms']:m[label+'_'+field+'_coverage']=ratio(sum(r[field]=='YES' for r in subset),len(subset))
  m['confusion']={t:{p:sum(r['truth']==t and r['prediction']==p for r in rs) for p in ['YES','NO_CLEAR_EVIDENCE','UNKNOWN']} for t in ['YES','NO_CLEAR_EVIDENCE','UNKNOWN']}
  result['arms'][g]={'metrics':m,'cases':rs}
 save(OUT/f'{split}_evaluation.json',result)

def revise(g):
 rows,images=inputs('train',g=='C') if g!='A' else ([],[])
 dev=read(OUT/'development_evaluation.json')['arms'][g]
 feedback={'aggregate_metrics':dev['metrics'],'own_blind_predictions':read(OUT/f'evaluation/development_{g}/parsed.json')}
 if g=='C':feedback['verified_development_effects']=[{'id':x['id'],'effects':x['verified_effects']} for x in dev['cases']]
 p=TASK+'\n只允许一次最小修订，也可KEEP。不得累加条件凑答案。说明具体问题、证据、修改Trigger/抓点/方向中的哪一项和泛化理由。仅用原组允许的经验及本组反馈。A/B未获得逐例结果，聚合评分不支持反推出单例标签。\n原Skill:'+json.dumps(read(OUT/'skills'/f'{g}_initial.json'),ensure_ascii=False)+'\n原经验:'+json.dumps(rows,ensure_ascii=False)+'\nDevelopment反馈:'+json.dumps(feedback,ensure_ascii=False)
 r=call(f'teacher_{g}/revision',p,images,REV);final=read(OUT/'skills'/f'{g}_initial.json') if r['decision']=='KEEP' else r['final_skill'];save(OUT/'skills'/f'{g}_final.json',final)
 (OUT/f'teacher_{g}/version.diff').write_text(''.join(difflib.unified_diff((OUT/'skills'/f'{g}_initial.json').read_text().splitlines(True),(OUT/'skills'/f'{g}_final.json').read_text().splitlines(True),fromfile='initial',tofile='final')))

def replay():
 selected=cases('test')[:2]
 jobs=[]
 for rep in range(2):
  for x in selected:
   for g in (['Base','A','B','C'] if rep==0 else ['C','B','A','Base']):jobs.append((rep,x,g))
 def replay_one(job):
  rep,x,g=job
  im=DATA/x['Observation'];sk=None if g=='Base' else read(OUT/'skills'/f'{g}_deployment.json')
  p='依据当前RGB，只生成一次Pa/Pb展开动作。坐标必须在图像范围内，Pa须为可见布料上的抓取位置，Pb为释放目标；说明抓取区域、方向、Skill是否触发。图像尺寸:'+str(Image.open(im).size)+'\n可选Skill:'+json.dumps(sk,ensure_ascii=False)
  name=f'planner_replay/{x["id"]}_{g}_r{rep+1}';r=call(name,p,[im],PLAN)
  w,h=Image.open(im).size;pa,pb=r['Pa'],r['Pb'];valid=all(len(p)==2 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in p) and 0<=p[0]<w and 0<=p[1]<h for p in [pa,pb]);dist=math.dist(pa,pb) if len(pa)==len(pb)==2 else None
  save(OUT/name/'geometry.json',{'finite_2d_in_bounds':valid,'nonzero_pixel_displacement':bool(dist and dist>0),'distance_px':dist,'on_cloth':'NOT_VERIFIED_BY_INDEPENDENT_SEGMENTATION','robot_reachability':'NOT_ASSESSED','physical_success':'NOT_EXECUTED'})
  if valid:
   image=Image.open(im).convert('RGB');d=ImageDraw.Draw(image);d.line([tuple(pa),tuple(pb)],fill='red',width=5)
   for point,c,label in [(pa,'lime','Pa'),(pb,'blue','Pb')]:a,b=point;d.ellipse((a-8,b-8,a+8,b+8),fill=c);d.text((a+10,b),label,fill=c)
   image.save(OUT/name/'action_overlay.jpg')

 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  list(pool.map(replay_one,jobs))

def main():
 init();stage=sys.argv[1]
 if stage=='teachers':
  for g in 'ABC':teacher(g)
  save(OUT/'skills/initial_freeze.json',{'at':now(),'sha256':{g:sha(OUT/'skills'/f'{g}_initial.json') for g in 'ABC'}})
 elif stage=='retry-development-B':
  old=OUT/'quarantine/development_B_usage_limit_20261028'
  m=read(old/'input_manifest.json')
  call('evaluation/development_B',(old/'prompt.txt').read_text(),[Path(x['source']) for x in m['attachments']],read(old/'schema.json'))
 elif stage=='development-C':evaluate('development','initial','C')
 elif stage=='development':evaluate('development','initial')
 elif stage=='revisions':
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   list(pool.map(revise,'ABC'))
  save(OUT/'skills/final_freeze.json',{'at':now(),'sha256':{g:sha(OUT/'skills'/f'{g}_final.json') for g in 'ABC'}})
 elif stage=='freeze-deployment':
  assert (OUT/'skills/final_freeze.json').exists()
  for g in 'ABC':
   final=read(OUT/'skills'/f'{g}_final.json')
   save(OUT/'skills'/f'{g}_deployment.json',{k:final[k] for k in ['skill','trigger','grasp_selection','direction_selection']})
  save(OUT/'skills/deployment_freeze.json',{'at':now(),'derivation':'exact unchanged four operational fields from final skills; identical allowlist for all arms','sha256':{g:sha(OUT/'skills'/f'{g}_deployment.json') for g in 'ABC'}})
 elif stage=='test':
  assert (OUT/'skills/final_freeze.json').exists();evaluate('test','final')
 elif stage=='replay':replay()
if __name__=='__main__':main()
