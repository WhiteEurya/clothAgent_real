"""Two independent visual descriptions per case, with conservative agreement labels."""
from common import *
from transport import call
from concurrent.futures import ThreadPoolExecutor,as_completed
import collections,jsonschema

DIMS=['unfolding_gain','whole_garment_transport','new_folding','exposure_gain']
EFFECT=obj({'value':enum('YES','NO_CLEAR_EVIDENCE','UNKNOWN'),'confidence':NUM,'visible_evidence':S,'alternative_explanations_or_limits':S})
CASE=obj({'id':S,'endpoint_comparability':enum('GOOD','LIMITED','UNKNOWN'),'comparison_evidence':S,'pre_structure':S,'observed_changes':SS,'INFERRED':SS,'UNKNOWN':SS,'effects':obj({k:EFFECT for k in DIMS}),'process_evidence':S})
SCHEMA=obj({'cases':{'type':'array','items':CASE}})
INSTRUCTIONS='''任务：只描述已经执行的T-shirt操作使布形发生了什么变化，不归纳操作规则、不推荐抓点、方向或操作顺序。历史Planner理由、效果标签和专家规则均不提供。
每个案例独立评估四个量：
unfolding_gain：可见局部折拢/重叠减少、布面舒展的证据。不能仅凭轮廓面积增大、拉长或整体旋转说有展开。
whole_garment_transport：衣物多个相距较远的区域相对桌面共同平移/旋转的证据；不是任何局部运动就算整体搬移。可以与局部展开同时发生。
new_folding：原本相对展开区域出现新的折拢/重叠/聚拢的可见证据；区分光照/褶线改向与新增覆盖。
exposure_gain：此前被覆盖或难辨的布边/布面结构变得可见的证据，可发生在原轮廓内部。新轮廓位置、面积增加、或把已可见边移到新位置本身不算显露。无法建立对应时UNKNOWN。
输出YES、NO_CLEAR_EVIDENCE或UNKNOWN，逐项给图像证据、0到1的主观置信度、竞争性解释。NO_CLEAR_EVIDENCE仅表示可比图中未见明确证据，不是严格证明没有变化。四量不得压成成功分数。
先比较桌面/背景参照和可辨衣物结构，判断前后视角可比性GOOD/LIMITED/UNKNOWN。遮挡、出画、位姿差异妨碍判断时保留UNKNOWN。不能把图像整体移动当成布料移动。过程为移动腕部相机，不可从其像素运动断言接触、抓层、滑脱或连续传力。执行完成不等于抓到布料。
pre_structure只能描述Before中可见的结构，不引入After事实。observed_changes为直接前后现象，INFERRED为解释，UNKNOWN保存缺口。不得臆造旧Planner真实意图。每项证据1–2句，优先具体位置和可见变化，不写长篇研究结论。'''

def batches():
 cases=read(OUT/'selection_manifest.json')['cases'];out=[]
 for reviewer in ['R1','R2']:
  index=0
  for split in ['train','development','test','reference']:
   rows=[c for c in cases if c['split']==split]
   rows=sorted(rows,key=lambda c:c['id'] if reviewer=='R1' else key('review2|'+c['id']))
   for i in range(0,len(rows),4):
    index+=1;out.append({'name':f'clean_effect_{reviewer}_{index:02d}','reviewer':reviewer,'split':split,'ids':[r['id'] for r in rows[i:i+4]]})
 return out

def execute_batch(batch):
 prompt=INSTRUCTIONS+'\n';images=[]
 if batch['reviewer']=='R1':prompt+='本次仅给原始Before/After，故执行接触和具体动作过程均UNKNOWN。请先独立观察末态差异。\n'
 else:prompt+='本次给Before、实际动作事实、按时序抽取的过程总览和After；用过程核对可见证据，但不让命令数值替代结果观察。\n'
 for cid in batch['ids']:
  f=OUT/'cases'/cid;facts=read(f/'facts.json');ims=[f/'before.png']
  if batch['reviewer']=='R2' and (f/'process.jpg').exists():ims.append(f/'process.jpg')
  ims.append(f/'after.png');images+=ims
  prompt+='案例'+cid+'；附件顺序：'+','.join(i.name for i in ims)+'。原始前后图1280×720；过程每格320×180，左至右、上至下时间有序。\n'
  if batch['reviewer']=='R2':prompt+=json.dumps(facts,ensure_ascii=False)+'\n'
  else:prompt+=json.dumps({k:facts[k] for k in ['id','before_at','after_at','recorded_camera_transform_equal']},ensure_ascii=False)+'；两端观测来自真实已执行记录，登记相机变换相同不替代图像配准。\n'
 result=call(batch['name'],prompt,images,SCHEMA);jsonschema.validate(result,SCHEMA)
 assert sorted(x['id'] for x in result['cases'])==sorted(batch['ids'])
 return {'name':batch['name'],'status':'COMPLETE'}

def run():
 verify_freezes();bs=batches()
 if not (OUT/'effect_batches_clean.json').exists():save(OUT/'effect_batches_clean.json',bs);freeze('clean_effect_schedule',[OUT/'effect_batches_clean.json'])
 # Independent read-only requests run two at a time. No agent delegation or shared context.
 jobs=sorted(bs,key=lambda b:(int(b['name'].split('_')[-1]),b['reviewer']))
 results=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures={pool.submit(execute_batch,b):b for b in jobs}
  for f in as_completed(futures):
   b=futures[f]
   try:r=f.result()
   except Exception as e:r={'name':b['name'],'status':'FAILED','error':str(e)}
   results.append(r);write(OUT/'effect_run_status.json',results);print('BATCH',json.dumps(r),flush=True)
 consolidate()

def consolidate(target='effects.json',retries=None):
 bs=read(OUT/'effect_batches_clean.json');reviews=collections.defaultdict(dict)
 for b in bs:
  callname=(retries or {}).get(b['name'],b['name']);f=OUT/'teacher_calls'/callname/'parsed.json'
  if f.exists():
   for row in read(f)['cases']:reviews[row['id']][b['reviewer']]={'call':callname,**row}
 output=[];agreement=collections.Counter();totals=collections.Counter()
 for c in read(OUT/'selection_manifest.json')['cases']:
  rr=reviews[c['id']];z={}
  for dim in DIMS:
   vals=[rr[r]['effects'][dim]['value'] for r in rr];confs=[rr[r]['effects'][dim]['confidence'] for r in rr];quality=[rr[r]['endpoint_comparability'] for r in rr]
   usable=len(rr)==2 and len(set(vals))==1 and vals[0]!='UNKNOWN' and min(confs)>=.65 and all(v=='GOOD' for v in quality)
   z[dim]={'value':vals[0] if usable else 'UNKNOWN','review_values':dict(zip(rr,vals)),'accepted':usable,'reason':'two independent descriptions agree with adequate comparability/confidence' if usable else 'review missing, disagreement, low confidence or limited comparability'}
   totals[dim]+=1;agreement[dim]+=len(rr)==2 and len(set(vals))==1
  u,t,n,e=[z[k]['value'] for k in DIMS]
  group='A_LOCAL_UNFOLDING' if u=='YES' and n=='NO_CLEAR_EVIDENCE' else 'C_MIXED' if u=='YES' and n=='YES' else 'B_TRANSPORT_WITHOUT_CLEAR_UNFOLDING' if t=='YES' and u=='NO_CLEAR_EVIDENCE' else 'NO_CLEAR_SHAPE_GAIN' if u==t==n==e=='NO_CLEAR_EVIDENCE' else 'OTHER_OR_UNCERTAIN'
  output.append({'id':c['id'],'split':c['split'],'round':c['round'],'z':z,'contrast_group':group,'reviews':rr,'future_affordance_gain':'NOT_ASSESSED','physical_outcome_of_proposed_actions':'NO_PROPOSED_ACTIONS'})
 result={'at':utc(),'cases':output,'dimension_exact_agreement':{k:{'agree':agreement[k],'total':totals[k]} for k in DIMS},'not_independent_ground_truth':'same-model fresh requests, different evidence subsets; agreement is reproducibility only'}
 save(OUT/target,result);freeze(Path(target).stem,[OUT/target]);print('CONSOLIDATED',collections.Counter(x['contrast_group'] for x in output),flush=True)

def retry_failed():
 retries={}
 for batch in read(OUT/'effect_batches_clean.json'):
  folder=OUT/'teacher_calls'/batch['name'];result=read(folder/'result.json')
  if result['status']=='COMPLETE':continue
  events=(folder/'events.jsonl').read_text()
  if 'Connection failed' not in events:raise RuntimeError('Non-network failure needs explicit investigation: '+batch['name'])
  new={**batch,'name':batch['name']+'_network_retry1'};execute_batch(new)
  a=read(folder/'input_manifest.json');b=read(OUT/'teacher_calls'/new['name']/'input_manifest.json')
  assert a['prompt_sha256']==b['prompt_sha256'] and [x['sha256'] for x in a['transport_images']]==[x['sha256'] for x in b['transport_images']]
  retries[batch['name']]=new['name']
 save(OUT/'retry_map.json',retries);freeze('network_retries',[OUT/'retry_map.json']);consolidate('effects_verified.json',retries)

def audit_context():
 m=read(OUT/'teacher_calls/effect_R1_01/input_manifest.json');cmd=m['command'];flags=cmd[2:cmd.index('--output-schema')];flags=[x for x in flags if x not in ['--ephemeral','--skip-git-repo-check','--json']];i=flags.index('--color');del flags[i:i+2]
 flags+=['-c','skills.config=[{path="/home/sja/.codex/skills/cloth-contour-normal-grasp/SKILL.md",enabled=false}]']
 r=subprocess.run(['codex']+flags+['debug','prompt-input','Offline isolation audit only.'],capture_output=True,text=True,timeout=120)
 (OUT/'context_audit_raw.json').write_text(r.stdout);(OUT/'context_audit_stderr.txt').write_text(r.stderr)
 markers=['cloth-contour-normal-grasp','clothAgent_real','hem_sleeve_pull_rule','failure_driven_rsi','Reveal','法线']
 d={'at':utc(),'returncode':r.returncode,'forbidden_markers':{x:x in r.stdout for x in markers},'generic_skills_present':'Available skills' in r.stdout,'raw_bytes':len(r.stdout)};write(OUT/'context_audit.json',d);print(json.dumps(d,ensure_ascii=False))

if __name__=='__main__':
 import sys
 if len(sys.argv)>1 and sys.argv[1]=='stop-contaminated':
  import signal
  victims=[];processes={}
  for path in Path('/proc').glob('[0-9]*'):
   try:
    args=(path/'cmdline').read_bytes().decode().strip('\0').split('\0');ppid=int((path/'stat').read_text().split(') ')[1].split()[1]);processes[int(path.name)]=(args,ppid)
   except (OSError,ValueError):pass
  for pid,(args,ppid) in processes.items():
   if pid!=os.getpid() and len(args)==2 and args[1]==str(OUT/'effects.py'):
    os.kill(pid,signal.SIGSTOP);descendants={pid}
    for _ in range(10):descendants.update(p for p,(a,parent) in processes.items() if parent in descendants)
    for child in descendants-{pid}:
     try:os.kill(child,signal.SIGTERM)
     except ProcessLookupError:pass
    os.kill(pid,signal.SIGKILL);victims.append(pid)
  print('Stopped experiment supervisors only:',victims)
 elif len(sys.argv)>1 and sys.argv[1]=='audit-context':audit_context()
 elif len(sys.argv)>1 and sys.argv[1]=='retry-failed':retry_failed()
 else:run()
