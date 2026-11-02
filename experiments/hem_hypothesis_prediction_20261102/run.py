"""One historical action prediction, frozen before outcome disclosure. No robot imports."""
import argparse, hashlib, json, random, subprocess, time, tempfile, os, signal
from pathlib import Path
import jsonschema
from PIL import Image

ROOT=Path('/mnt/newssd/sja/clothAgent_real')
SRC=ROOT/'reports/intention_outcome_hypothesis_bank_v1'
OUT=ROOT/'reports/x0152_h001_prediction_20261102'
SEED=2026110207
def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def obj(d):return dict(type='object',properties=d,required=list(d),additionalProperties=False)
S={'type':'string'}
def enum(*x):return {'type':'string','enum':list(x)}
def arr(x):return {'type':'array','items':x}
BOX={'type':'array','items':{'type':'integer'},'minItems':4,'maxItems':4}
SELECT=obj({'candidates':arr(obj({'id':S,'applicability':enum('COMPARABLE','PARTIAL','NOT_COMPARABLE'),
 'before_evidence':S,'action_relation':S,'limitations':S})), 'selected_id':S,'selection_reason':S})
PRED=obj({'target_sleeve':S,'semantic_identity':S,'target_bbox_xyxy':BOX,'covering_edge_bbox_xyxy':BOX,
 'before_evidence':S,'hypothesis_applicability':enum('APPLICABLE_PROVISIONALLY','PARTIAL','INSUFFICIENT'),
 'action_interpretation':S,'cover_edge_retreat_prediction':enum('YES','NO','UNCERTAIN'),
 'prediction_confidence':enum('LOW','MEDIUM','HIGH'),'relative_motion_prediction':S,
 'target_sleeve_prediction':S,'corner_reveal_prediction':enum('LIKELY','UNLIKELY','UNCERTAIN'),
 'supporting_observations':arr(S),'falsifying_observations':arr(S),'non_falsifying_failures':arr(S),
 'alternative_outcome':S,'limits_of_two_images':S})
OBS=obj({'image_quality_and_view_change':S,'before_sleeve_regions':S,'after_sleeve_regions':S,
 'covering_boundary_changes':arr(S),'relative_motion_evidence':S,'visible_sleeve_changes':arr(S),
 'corner_reveal_evidence':S,'grasp_or_contact_evidence':S,'unresolved_material_identity':S})
CMP=obj({'prediction_vs_observation':arr(obj({'prediction':S,'observation':S,'verdict':enum('CONSISTENT','INCONSISTENT','UNRESOLVED')})),
 'cover_edge_retreat_result':enum('SUPPORTED_PROVISIONALLY','NOT_OBSERVED','UNRESOLVED'),
 'target_sleeve_result':S,'corner_reveal_result':enum('CONFIRMED','PROVISIONAL','NO_REVEAL','UNCERTAIN'),
 'hypothesis_result':enum('PRELIMINARY_SUPPORT','WEAKENED','INCONCLUSIVE'),
 'falsification_conditions_met':S,'conclusion':S,'limitations':arr(S)})
COMMON='你是独立离线视觉分析Codex。没有此前聊天或研究报告。只用本请求提供的文字与图片；禁止调用任何工具、读取文件、联网或运行代码。输出中文JSON。不得把轮廓变化等同语义端点显露，不得把未观测布层身份当作事实。'
SELP=COMMON+'''这一步只选择一个历史状态用于未来结果预测，不预测已发生结果。提供的12张图全部是Before，顺序对应C01到C12。每张的真实记录动作在文字中。只依据原假设适用条件、Before的可见结构以及动作作用关系评价全部12个候选。优先存在上胸覆盖边、部分外露目标袖部、动作可能牵动相关布层的状态，不能只按图像左右/同向拉动选。不要假设隐藏袖口。选最可比较的一例；若全部缺少相关线索返回selected_id=NONE。不能指定新动作。'''
PREDP=COMMON+'''只提供一个既有假设及适用条件、一个新的Before RGB、该状态实际记录的动作。你没有After、新案例原始意图、效果标签或既有观察。独立预测这个已执行动作的结果：覆盖边会否相对目标袖部退开？目标袖部怎样变化？具体什么结果会反驳假设？允许YES/NO/UNCERTAIN。框在原图像素中仅粗定位目标袖部与覆盖边，不能虚构隐藏端点坐标。材料语义侧不明用画面位置，不能强制判LS/RS。清晰区分条件预测与无条件预测、袖面外露与同一真实袖口中点解除遮挡。动作记录有索引字段和底层指令；如抬升高度等不一致，保留差异，不能自行适配。底层指令不是接触传感器证据。不能从假设成立推出该次一定抓住正确布层。不要提出新动作。'''
OBSP=COMMON+'''图1为Before，图2为After。你不知道待验证假设、预测、原始意图或动作。独立描述整件衣服的可见变化，尤其袖部与相邻覆盖边。用画面位置、可追踪文字/缝边作证据。说明整体运动、局部相对变化，袖面变宽、翻转、出入视野与真正袖口解除遮挡的区别。没有过程帧不得确认材料点轨迹、夹持成败或隐藏层序。只观察，不构造策略或物理机制。'''
CMPP=COMMON+'''这是冻结预测后的只读核对。提供原假设、冻结预测、独立且未读预测的Before/After观察以及Before/After原图。逐项比较，不修改预测，不生成新机制或策略。原假设关于层间运动的反证有前提：夹持成功、作用传入相关层且可追踪；缺少这些证据不能把无改善自动判机制被证伪。不能将两帧因果解释升级已验证技能。区分覆盖边退开、袖面更外露、真实语义袖口点恢复。'''

def call(name,prompt,data,schema,images):
 d=OUT/'calls'/name;d.mkdir(parents=True,exist_ok=True)
 if (d/'result.json').exists():return load(d/'result.json')['output']
 input_text=prompt+'\n'+json.dumps(data,ensure_ascii=False)
 (d/'input.txt').write_text(input_text);save(d/'schema.json',schema)
 work=Path(tempfile.mkdtemp(prefix='x0152_prediction_empty_'))
 cmd=['codex','exec','--ephemeral','--json','-s','read-only','--skip-git-repo-check','-C',str(work),'-c','model="gpt-6-astra"','-c','model_reasoning_effort="medium"','--output-schema',str(d/'schema.json'),'-o',str(d/'output.json')]
 manifests=[]
 for i,p in enumerate(images):
  dest=d/f'image_{i+1:02}.jpg'
  with Image.open(p) as im:
   im=im.convert('RGB')
   if name=='selection':im.thumbnail((960,960))
   im.save(dest,quality=94)
  cmd+=['-i',str(dest)];manifests.append({'ordinal':i+1,'source':str(p),'sha256':sha(p),'uploaded_sha256':sha(dest)})
 cmd+=['-'];start=time.time();save(d/'manifest.json',{'command':cmd,'started_at':start,'images':manifests,'input_sha256':sha(d/'input.txt')})
 print('START',name,flush=True)
 with (d/'input.txt').open() as fi,(d/'events.jsonl').open('w') as fo,(d/'stderr.log').open('w') as fe:
  proc=subprocess.Popen(cmd,stdin=fi,stdout=fo,stderr=fe,start_new_session=True)
  try:rc=proc.wait(timeout=900)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10);raise
 if rc:raise RuntimeError(f'{name} codex exit={rc}: inspect {d}')
 events=[]
 for line in (d/'events.jsonl').read_text().splitlines():
  try:events.append(json.loads(line))
  except ValueError:pass
 bad=[e for e in events if e.get('item',{}).get('type') in ['command_execution','mcp_tool_call','web_search','file_change']]
 if bad:raise RuntimeError('Isolation violation')
 out=load(d/'output.json');jsonschema.validate(out,schema)
 save(d/'result.json',{'output':out,'latency_s':time.time()-start,'finished_at':time.time(),'usage':next((e['usage'] for e in reversed(events) if 'usage'in e),None),'tool_calls':0})
 print('FINISH',name,round(time.time()-start),flush=True);return out

def prepare():
 if (OUT/'protocol.json').exists():return
 OUT.mkdir(parents=True)
 h=next(x for x in load(SRC/'hypothesis_bank.json') if x['hypothesis_id']=='X0152-H001')
 save(OUT/'source_hypothesis.json',h)
 # Prediction gets only hypothesis/conditions, not its historical image, intention or observation labels.
 supplied={k:h[k] for k in ['observable_conditions','hypothesized_mechanism','predicted_effect','falsification_condition','evidence_level','uncertainty']}
 save(OUT/'provided_hypothesis.json',supplied)
 idx=load(SRC/'experience_index.json');origin=next(x for x in idx if x['experience_id']=='X0152')
 reviewed={x['source_experience_id'] for x in load(ROOT/'reports/hypothesis_retrieval_20261102/retrieval_analysis.json')['reviews']}
 # No new-case observations or bank entries are passed to or used in selection.
 groups={};excluded=[]
 for x in idx:
  if x['run']==origin['run'] or x['experience_id'] in reviewed:
   excluded.append({'id':x['experience_id'],'reason':'source run or previously visually reviewed'});continue
  groups.setdefault(x['run'],[]).append(x)
 rng=random.Random(SEED);runs=sorted(groups);rng.shuffle(runs);chosen=[]
 for run in runs:
  pool=sorted(groups[run],key=lambda x:x['experience_id']);rng.shuffle(pool)
  for x in pool:
   p=load(x['payload']);a=p['indexed_executed_action']
   if p['execution_information'].get('status')!='EXECUTED' or not a.get('grasp_px') or not Path(x['before']).is_file() or not Path(x['after']).is_file():continue
   cid=f'C{len(chosen)+1:02}';save(OUT/'actions'/f'{cid}.json',{'indexed_action':a,'native_command_sequence':p['native_command_sequence'],'warning':'Indexed fields and native commands retained separately; commands do not establish successful material grasp.'})
   chosen.append({'id':cid,'experience_id':x['experience_id'],'run':run,'source':x['source'],'before':x['before'],'after':x['after'],'payload':x['payload'],'before_sha256':x['before_sha256'],'after_expected_sha256':x['after_sha256'],'payload_sha256':sha(x['payload'])});break
  if len(chosen)==12:break
 save(OUT/'sample_manifest.json',chosen);save(OUT/'excluded.json',excluded)
 save(OUT/'protocol.json',{'version':'single_hypothesis_prediction_v1','frozen_at':time.time(),'source_hypothesis':'X0152-H001','seed':SEED,'sample_size':len(chosen),'eligible_runs':len(groups),'sampling':'Seeded shuffle runs, one eligible EXECUTED action per run; exclude source run and all previous 25 visual-review source operations. No outcome-dependent selection.','selector':'Before+recorded action+source hypothesis only; all candidates assessed; one selected, no replacement based on After.','predictor':'Fresh default codex ephemeral session, NO profile, only hypothesis+selected Before+action; no selector rationale or source IDs.','observer':'Separate fresh session, Before+After only, no hypothesis/prediction/action/intent.','model':'gpt-6-astra','effort':'medium','max_calls':4,'no_robot':True,'code_sha256':sha(__file__),'source_bank_sha256':sha(SRC/'hypothesis_bank.json'),'source_index_sha256':sha(SRC/'experience_index.json'),'pre_exposure_limit':'Coordinator knows source hypothesis and prior report; new-case prediction uses independent tool-free sessions. Not a statistically independent physical trial.'})
 for n,s in [('selection',SELECT),('prediction',PRED),('observation',OBS),('comparison',CMP)]:save(OUT/f'{n}_schema.json',s)
 print('PREPARED',len(chosen),'Before-only candidates',flush=True)

def run():
 prepare();protocol=load(OUT/'protocol.json')
 assert protocol['code_sha256']==sha(__file__),'Frozen runner changed'
 rows=load(OUT/'sample_manifest.json');h=load(OUT/'provided_hypothesis.json')
 selection=call('selection',SELP,{'hypothesis':h,'candidates':[{'id':r['id'],'action':load(OUT/'actions'/f'{r["id"]}.json')} for r in rows]},SELECT,[r['before'] for r in rows])
 assert sorted(x['id'] for x in selection['candidates'])==sorted(r['id'] for r in rows)
 if selection['selected_id']=='NONE':save(OUT/'status.json',{'status':'NO_COMPARABLE_BEFORE_FOUND'});return
 row=next(r for r in rows if r['id']==selection['selected_id']);save(OUT/'selected_case.json',row)
 action=load(OUT/'actions'/f'{row["id"]}.json')
 pred=call('prediction',PREDP,{'hypothesis':h,'recorded_action':action},PRED,[row['before']])
 freeze=OUT/'prediction_frozen.json'
 if not freeze.exists():save(freeze,{'frozen_at':time.time(),'prediction_sha256':sha(OUT/'calls/prediction/output.json'),'before_sha256':sha(row['before']),'action_sha256':sha(OUT/'actions'/f'{row["id"]}.json'),'hypothesis_sha256':sha(OUT/'provided_hypothesis.json'),'selected_case_sha256':sha(OUT/'selected_case.json'),'no_after_in_selection_or_prediction':True,'selection_input_sha256':sha(OUT/'calls/selection/input.txt'),'prediction_input_sha256':sha(OUT/'calls/prediction/input.txt')})
 print('PREDICTION FROZEN',row['experience_id'],flush=True)
 assert load(freeze)['prediction_sha256']==sha(OUT/'calls/prediction/output.json')
 # The first read of any new-case After image content occurs only below.
 assert sha(row['after'])==row['after_expected_sha256']
 if not (OUT/'after_revealed.json').exists():save(OUT/'after_revealed.json',{'revealed_at':time.time(),'after_sha256':sha(row['after']),'freeze_sha256':sha(freeze)})
 obs=call('observation',OBSP,{},OBS,[row['before'],row['after']])
 cmp=call('comparison',CMPP,{'hypothesis':h,'frozen_prediction':pred,'independent_observation':obs},CMP,[row['before'],row['after']])
 assert sha(SRC/'hypothesis_bank.json')==protocol['source_bank_sha256']
 assert sha(SRC/'experience_index.json')==protocol['source_index_sha256']
 for r in rows:assert sha(r['payload'])==r['payload_sha256']
 save(OUT/'result.json',{'source_hypothesis':load(OUT/'source_hypothesis.json'),'selected_case':row,'selection':selection,'prediction':pred,'observation':obs,'comparison':cmp,'protocol':protocol,'prediction_freeze':load(freeze),'after_disclosure':load(OUT/'after_revealed.json')})
 save(OUT/'status.json',{'status':'COMPLETE','finished_at':time.time(),'prediction_unmodified':sha(OUT/'calls/prediction/output.json')==load(freeze)['prediction_sha256'],'source_unchanged':True})
if __name__=='__main__':run()
