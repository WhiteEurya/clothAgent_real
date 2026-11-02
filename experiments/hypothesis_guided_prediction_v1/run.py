"""Frozen small paired prediction experiment. No robot imports or source writes."""
import concurrent.futures,hashlib,json,os,random,signal,subprocess,tempfile,time
from pathlib import Path
from PIL import Image
import jsonschema

ROOT=Path('/mnt/newssd/sja/clothAgent_real')
SRC=ROOT/'reports/intention_outcome_hypothesis_bank_v1'
OUT=ROOT/'reports/hypothesis_guided_prediction_20261102'
SEED=2026110219
REGIONS=['IMAGE_LEFT_SLEEVE','IMAGE_RIGHT_SLEEVE','IMAGE_LEFT_HEM','IMAGE_RIGHT_HEM']
METRICS=['EXPOSED_REGION_AREA','VISIBLE_TERMINAL_EDGE_LENGTH','COVERING_OVERLAP_EXTENT']
CHANGES=['INCREASE','DECREASE','UNCHANGED','UNKNOWN']
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def obj(d):return {'type':'object','properties':d,'required':list(d),'additionalProperties':False}
def enum(*xs):return {'type':'string','enum':list(xs)}
S={'type':'string'}
def arr(x):return {'type':'array','items':x}
BOX={'type':'array','items':{'type':'integer'},'minItems':4,'maxItems':4}
SELECT=obj({'cases':arr(obj({'id':S,'accept':{'type':'boolean'},'applicability':enum('APPLICABLE_PROVISIONALLY','NOT_APPLICABLE','INSUFFICIENT'),
 'visible_conditions':S,'action_relation':S,'target_region':enum(*REGIONS,'UNKNOWN'),'reason':S,'limitations':S}))})
PRED=obj({'primary_target':enum(*REGIONS,'UNKNOWN'),'semantic_corner':enum('LS','RS','LH','RH','UNKNOWN'),
 'target_bbox_xyxy':BOX,'semantic_identity_basis':S,
 'visibility_change':enum(*CHANGES),'geometry_metric':enum(*METRICS),'geometry_change':enum(*CHANGES),
 'static_after_check':S,'prediction_basis':S,'uncertainty':S,'confidence':enum('LOW','MEDIUM','HIGH')})
MEASURE=obj({'change':enum(*CHANGES),'evidence_quality':enum('CLEAR','PROVISIONAL','INSUFFICIENT'),'evidence':S})
OBS=obj({'image_comparability':S,'regions':arr(obj({'region':enum(*REGIONS),'semantic_corner':enum('LS','RS','LH','RH','UNKNOWN'),
 'region_correspondence':enum('CLEAR','PROVISIONAL','INSUFFICIENT'),'before_bbox_xyxy':BOX,'after_bbox_xyxy':BOX,
 'identity_evidence':S,'visibility':MEASURE,'exposed_region_area':MEASURE,'visible_terminal_edge_length':MEASURE,'covering_overlap_extent':MEASURE,
 'corner_reveal':enum('CONFIRMED','PROVISIONAL','NO_REVEAL','UNKNOWN'),'corner_reveal_evidence':S})),
 'limitations':S})
DEFINITIONS='''任务：四个语义端点Corner Reveal。LS/RS是穿着者左/右袖口目标点，LH/RH是左/右真实下摆角。可辨识性提高不自动等于从自遮挡中显露。材料身份不足时必须保留UNKNOWN。
为了避免图像左右和穿着者左右混淆，region键以Before图中的袖部/下摆区域作锚，After中仍追踪同一片区域，而不是按After绝对左右重新命名。语义侧另外输出；不确定可用UNKNOWN。UNKNOWN不等于失败或不变。
静态指标固定如下：
visibility：该语义端点所在区域的可见/可辨识程度（不是自动确认端点Reveal）。
EXPOSED_REGION_AREA：同一目标区域的可见投影布面范围；不推断隐藏材料面积，不以全衣面积代替。
VISIBLE_TERMINAL_EDGE_LENGTH：同一袖口末端边缘或下摆端部相邻真实边缘可看清的长度；不能用新假折边冒充。
COVERING_OVERLAP_EXTENT：同一目标区域被相邻可见布层覆盖的投影范围。减少表示覆盖变少；层界不足则UNKNOWN。
变化为INCREASE/DECREASE/UNCHANGED/UNKNOWN。只有图像中明确可辨的变化才给有方向的确定判断；UNCHANGED需证据支持基本相同，不等于没看清。没有标定分割噪声，本轮是视觉分类代理评价，不输出伪精确百分比。主要预测不能依赖过程中的接触、受力、运动先后、悬空阶段或夹持成败。'''
ISOLATION='你是独立离线分析Codex。仅使用随请求提供的信息与图像；禁止使用工具、读文件、联网或读取旧会话。中文JSON输出。不要生成新Skill或机器人动作。'
SELECT_PROMPT=ISOLATION+DEFINITIONS+'''现在只判断5个随机抽样案例是否适合检验给定假设，不评价已发生结果。图片全为Before，顺序与T01..T05一致。图像保持原始尺寸。逐一审核且不得遗漏。接受须同时有具体可见适用条件、可定位的潜在目标区域以及动作可能作用到相关布层；不要求确认隐藏层序，不能仅因有袖子就接受。证据不足/不适用应拒绝，不为凑数放宽。只依据Before与记录动作；没有After、历史意图或结果标签。不得更换抽样或生成动作。'''
PRED_PROMPT=ISOLATION+DEFINITIONS+'''对这张新的Before和真实记录动作预测其静态After效果。选一个最值得预测变化的端点区域，给其可见性变化和一种具体静态几何指标的变化，并说明After中如何检查。原图坐标的框只粗定位当前可见区域，不能虚构隐藏端点坐标。信息不足用UNKNOWN，但应区分两项预测，不必同时未知。不提供新动作。动作包含索引字段和原生指令，不一致时保留差异，不改成统一参数。命令完成不证明成功夹持。
预测应是对这个动作最终静态结果的最佳判断，不要只写“若条件全部成立则可能”。证据不足时可以明确UNKNOWN并说明缺少什么。只选择一个几何指标，不额外生成大量备选结果。'''
OBS_PROMPT=ISOLATION+DEFINITIONS+'''你是独立结果评价器。图1Before，图2After。你不知道预测、实验组、假设、原意图或动作。必须逐一记录四个region；对每个region的visibility及三个静态指标分别评价。给出能在这两张图上定位的证据。只有同一区域对应清楚且该指标变化明确才标CLEAR；疑似但材料对应/边界不足标PROVISIONAL，无法判断标INSUFFICIENT和UNKNOWN。不要用整体面积增加代替端点变化，不用过程机制解释作为判定。两图相似或看不清不能自动标UNCHANGED。每个region另外判断严格Corner Reveal，但不能从区域更清楚直接升级CONFIRMED。'''

def call(name,prompt,data,schema,images):
 folder=OUT/'calls'/name;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'result.json').exists():return load(folder/'result.json')['output']
 for n in range(1,3):
  d=folder/f'attempt_{n:02}';d.mkdir(parents=True,exist_ok=True)
  if (d/'failed.json').exists():continue
  (d/'input.txt').write_text(prompt+'\n'+json.dumps(data,ensure_ascii=False));save(d/'schema.json',schema)
  work=Path(tempfile.mkdtemp(prefix='hypothesis_effect_blind_'))
  cmd=['codex','exec','--ephemeral','--json','-s','read-only','--skip-git-repo-check','-C',str(work),'-c','model="gpt-6-astra"','-c','model_reasoning_effort="medium"','--output-schema',str(d/'schema.json'),'-o',str(d/'output.json')]
  ims=[]
  for k,path in enumerate(images,1):
   target=d/f'image_{k:02}.jpg'
   with Image.open(path) as im:im.convert('RGB').save(target,quality=94)
   cmd+=['-i',str(target)];ims.append({'source':str(path),'source_sha256':sha(path),'upload_sha256':sha(target)})
  cmd+=['-'];start=time.time();save(d/'manifest.json',{'command':cmd,'images':ims,'started_at':start,'input_sha256':sha(d/'input.txt')})
  print('START',name,n,flush=True)
  try:
   with (d/'input.txt').open() as fi,(d/'events.jsonl').open('w') as fo,(d/'stderr.log').open('w') as fe:
    proc=subprocess.Popen(cmd,stdin=fi,stdout=fo,stderr=fe,start_new_session=True)
    try:rc=proc.wait(timeout=900)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10);raise
   events=[]
   for line in (d/'events.jsonl').read_text().splitlines():
    try:events.append(json.loads(line))
    except ValueError:pass
   if any(e.get('item',{}).get('type') in ['command_execution','mcp_tool_call','web_search','file_change'] for e in events):raise RuntimeError('Forbidden tool call')
   if rc:raise RuntimeError('Codex failed; see events/stderr')
   result=load(d/'output.json');jsonschema.validate(result,schema)
   if schema==SELECT:assert sorted(x['id'] for x in result['cases'])==[f'T{i:02}' for i in range(1,len(images)+1)]
   if schema==OBS:assert sorted(x['region'] for x in result['regions'])==sorted(REGIONS)
   saved={'output':result,'started_at':start,'finished_at':time.time(),'latency_s':time.time()-start,'usage':next((e['usage'] for e in reversed(events) if 'usage'in e),None),'attempt_dir':str(d),'tool_calls':0}
   save(folder/'result.json',saved);print('FINISH',name,round(saved['latency_s']),flush=True);return result
  except Exception as e:
   save(d/'failed.json',{'error':repr(e),'latency_s':time.time()-start});print('FAILED',name,repr(e),flush=True)
   if 'Forbidden tool' in str(e):raise
 raise RuntimeError('Both attempts failed: '+name)

def prepare():
 if (OUT/'protocol.json').exists():return
 OUT.mkdir(parents=True)
 bank=load(SRC/'hypothesis_bank.json');h=next(x for x in bank if x['hypothesis_id']=='X0152-H001')
 save(OUT/'source_hypothesis.json',h)
 supplied={k:h[k] for k in ['observable_conditions','hypothesized_mechanism','predicted_effect','falsification_condition','evidence_level','uncertainty']}
 save(OUT/'provided_hypothesis.json',supplied)
 idx=load(SRC/'experience_index.json');rev=load(ROOT/'reports/hypothesis_retrieval_20261102/retrieval_analysis.json')['reviews']
 exposed={r['source_experience_id'] for r in rev}|{'X0152','X0339'}
 excluded_runs={r['run'] for r in idx if r['experience_id'] in exposed}
 groups={};excluded=[]
 for r in idx:
  if r['run'] in excluded_runs:excluded.append({'experience_id':r['experience_id'],'run':r['run'],'reason':'source run or previously After-exposed run'});continue
  p=load(r['payload']);act=p['indexed_executed_action']
  if p['execution_information'].get('status')!='EXECUTED' or act.get('motion_type')!='lift_pull_place' or not act.get('grasp_px') or not act.get('target_px') or not Path(r['before']).is_file() or not Path(r['after']).is_file():
   excluded.append({'experience_id':r['experience_id'],'reason':'not complete indexed lift-pull-place pair'});continue
  groups.setdefault(r['run'],[]).append(r)
 rng=random.Random(SEED);runs=sorted(groups);rng.shuffle(runs);sample=[]
 for run in runs[:5]:
  rows=sorted(groups[run],key=lambda x:x['experience_id']);r=rows[rng.randrange(len(rows))];cid=f'T{len(sample)+1:02}';p=load(r['payload'])
  save(OUT/'actions'/f'{cid}.json',{'indexed_action':p['indexed_executed_action'],'native_command_sequence':p['native_command_sequence'],'limits':'Recorded commands and indexed parameters; no assertion of successful material acquisition.'})
  sample.append({'id':cid,'experience_id':r['experience_id'],'run':r['run'],'source':r['source'],'before':r['before'],'after':r['after'],'before_sha256':r['before_sha256'],'after_expected_sha256':r['after_sha256'],'payload':r['payload'],'payload_sha256':sha(r['payload'])})
 save(OUT/'sample_manifest.json',sample);save(OUT/'metadata_exclusions.json',excluded)
 order=[{'id':r['id'],'group':g} for r in sample for g in ['Baseline','Hindsight']];rng.shuffle(order);save(OUT/'prediction_order.json',order)
 prompts={'selection':SELECT_PROMPT,'prediction_shared':PRED_PROMPT,'observation':OBS_PROMPT}
 for k,v in prompts.items():(OUT/f'{k}_prompt.txt').write_text(v)
 for k,v in [('selection',SELECT),('prediction',PRED),('observation',OBS)]:save(OUT/f'{k}_schema.json',v)
 save(OUT/'protocol.json',{'version':'hypothesis_guided_static_prediction_v1','frozen_at':time.time(),'seed':SEED,'hypothesis':'X0152-H001','hypothesis_rationale':'Contains explicit change of non-manipulated sleeve exposure and visible cover boundary. Chosen before any current test After; only static endpoint-region outcomes scored, never process mechanism. Previous X0339 was inconclusive and its full run excluded. This is an exploratory follow-up, not an independent validation of all mining.','max_random_cases':5,'eligible_runs':len(groups),'sampled_cases':len(sample),'sampling':'Uniform shuffled run list then uniform eligible action within run; no outcome filtering. Five only, rejected cases not replaced.','acceptance':'Observable hypothesis conditions + localizable target sleeve + action possibly engages relevant cloth; no required hidden-layer proof. INS UFFICIENT rejected.','conditions':'Same clean Before, same raw action, same Corner Reveal definitions, same schema/model; Hindsight only adds provided_hypothesis. No original intent or result labels.','evaluator':'New independent session per accepted case; Before+After only. No hypothesis, predictions, action, condition or selection rationale. All four regions and three metrics assessed before deterministic comparison.','prediction_scope':'One primary target region; visibility and one static geometric direction. UNKNOWN preserved. Semantic-side mismatch reported separately.','scoring':'Only CLEAR region correspondence and CLEAR non-UNKNOWN reference label are judgeable. UNKNOWN prediction = ABSTAIN, never correct/incorrect. Unknown reference = UNASSESSABLE. Known label equality = CORRECT else INCORRECT. Report conditional accuracy AND correct/judgeable coverage; no unknown imputation. Paired comparison only cases judgeable and definite for both; common target/metric subset separately.','information':'Fixed two prediction slots per case: definite counts, judgeable definite counts, correct concrete geometry counts, and response length/tokens/latency. More words is not more correct information. No unsupported causal detail scored.','reference_limit':'Blind model visual judgments are proxy labels, not independent physical ground truth. CLEAR is a preregistered visual rubric, not calibrated measurement. No statistical/generalization claims from <=5 actions.','models':{'name':'gpt-6-astra','effort':'medium','profile':None,'workers':2,'repetitions':1,'max_attempts':2},'source_bank_sha256':sha(SRC/'hypothesis_bank.json'),'source_index_sha256':sha(SRC/'experience_index.json'),'code_sha256':sha(__file__),'prompt_hashes':{k:sha(OUT/f'{k}_prompt.txt') for k in prompts},'no_robot':True,'no_new_skill':True})
 print('PREPARED',len(sample),'random cases',flush=True)

def run():
 prepare();p=load(OUT/'protocol.json');assert p['code_sha256']==sha(__file__)
 rows=load(OUT/'sample_manifest.json');h=load(OUT/'provided_hypothesis.json');by={r['id']:r for r in rows}
 selected=call('applicability',SELECT_PROMPT,{'hypothesis':h,'cases':[{'id':r['id'],'image_size':list(Image.open(r['before']).size),'action':load(OUT/'actions'/f'{r["id"]}.json')} for r in rows]},SELECT,[r['before'] for r in rows])
 accepted=[r['id'] for r in selected['cases'] if r['accept'] and r['applicability']=='APPLICABLE_PROVISIONALLY']
 save(OUT/'applicability_frozen.json',{'accepted_ids':accepted,'all_cases':selected['cases'],'frozen_at':time.time(),'output_sha256':sha(Path(load(OUT/'calls/applicability/result.json')['attempt_dir'])/'output.json')})
 if not accepted:
  save(OUT/'result.json',{'status':'NO_APPLICABLE_TEST_CASES','protocol':p,'sample':rows,'selection':selected,'predictions':{},'observations':{}});return
 jobs=[j for j in load(OUT/'prediction_order.json') if j['id'] in accepted]
 def predict(j):
  row=by[j['id']];data={'image_size':list(Image.open(row['before']).size),'recorded_action':load(OUT/'actions'/f'{row["id"]}.json')}
  if j['group']=='Hindsight':data['historical_hypothesis']=h
  return call('predict_'+j['id']+'_'+j['group'],PRED_PROMPT,data,PRED,[row['before']])
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
  outputs=list(ex.map(predict,jobs))
 predictions={cid:{j['group']:out for j,out in zip(jobs,outputs) if j['id']==cid} for cid in accepted}
 save(OUT/'predictions.json',predictions)
 if not (OUT/'predictions_frozen.json').exists():save(OUT/'predictions_frozen.json',{'frozen_at':time.time(),'predictions_sha256':sha(OUT/'predictions.json'),'hypothesis_sha256':sha(OUT/'provided_hypothesis.json'),'accepted_ids':accepted,'call_outputs':{j['id']+'_'+j['group']:sha(Path(load(OUT/'calls'/('predict_'+j['id']+'_'+j['group'])/'result.json')['attempt_dir'])/'output.json') for j in jobs},'all_predictions_before_any_test_after':True})
 f=load(OUT/'predictions_frozen.json');assert sha(OUT/'predictions.json')==f['predictions_sha256']
 print('ALL PREDICTIONS FROZEN',accepted,flush=True)
 # No test After image contents opened until all prediction outputs above are frozen.
 if not (OUT/'after_disclosure.json').exists():
  hashes={cid:sha(by[cid]['after']) for cid in accepted}
  for cid in accepted:assert hashes[cid]==by[cid]['after_expected_sha256']
  save(OUT/'after_disclosure.json',{'disclosed_at':time.time(),'after_sha256':hashes,'prediction_freeze_sha256':sha(OUT/'predictions_frozen.json')})
 def observe(cid):return call('observe_'+cid,OBS_PROMPT,{},OBS,[by[cid]['before'],by[cid]['after']])
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:obs=dict(zip(accepted,ex.map(observe,accepted)))
 assert sha(OUT/'predictions.json')==f['predictions_sha256']
 assert sha(SRC/'hypothesis_bank.json')==p['source_bank_sha256']
 assert sha(SRC/'experience_index.json')==p['source_index_sha256']
 save(OUT/'result.json',{'status':'COMPLETE','protocol':p,'sample':rows,'selection':selected,'predictions':predictions,'observations':obs,'prediction_freeze':f,'after_disclosure':load(OUT/'after_disclosure.json')})
 print('COMPLETE',len(accepted),'paired cases',flush=True)
if __name__=='__main__':run()
