"""Frozen, complete bank retrieval. Reads evidence only; never imports robot code."""
import argparse,concurrent.futures,hashlib,json,os,random,re,signal,subprocess,time,threading
from pathlib import Path
import jsonschema
BASE=Path('/mnt/newssd/sja/clothAgent_real')
SOURCE=BASE/'reports/intention_outcome_hypothesis_bank_v1'
OUT=BASE/'reports/hypothesis_retrieval_20261102'
SEED=2026110201

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def obj(props):return {'type':'object','properties':props,'required':list(props),'additionalProperties':False}
def text():return {'type':'string'}
def enum(*vals):return {'type':'string','enum':list(vals)}
def score(n):return {'type':'integer','minimum':0,'maximum':n}
SCREEN_ROW=obj({'blind_id':text(),'goal_relevance':score(3),'goal_basis':text(),
 'effect_expectation':enum('EXPLICIT_EXPECTED','UNSPECIFIED','CONTRADICTED','INDETERMINATE'),
 'relation_expectation':enum('EXPLICIT_EXPECTED','UNSPECIFIED','CONTRADICTED','INDETERMINATE'),
 'expectation_basis':text(),'operational_novelty':score(2),'new_vs_original':text(),
 'testability':score(4),'testability_basis':text(),'evidence_strength':score(3),'evidence_basis':text(),
 'observed_benefit_clue':{'type':'boolean'},'relation_quote':text(),'prediction_quote':text(),'limitations':text()})
SCREEN_SCHEMA=obj({'rows':{'type':'array','items':SCREEN_ROW}})
REVIEW_SCHEMA=obj({'blind_id':text(),
 'visual_support':enum('SUPPORTED_PROVISIONAL','WEAK','NOT_SUPPORTED','UNASSESSABLE'),
 'visible_before_after_evidence':text(),'semantic_identity_limits':text(),
 'expectation_gap':enum('EXPLICIT_EXPECTED','UNSPECIFIED','CONTRADICTED','INDETERMINATE'),
 'already_in_original':text(),'added_relation':text(),'relation_is_specific':{'type':'boolean'},
 'existing_prediction_quote':text(),'existing_falsification_quote':text(),'evidence_vs_mechanism':text(),
 'inversion_potential':enum('WORTH_CONSIDERING_PROVISIONALLY','INSUFFICIENT_EVIDENCE','ALREADY_EXPRESSED','NOT_GOAL_RELEVANT'),
 'reason':text(),'limitations':text()})
SCREEN_PROMPT='''你是只读Hypothesis Bank检索评估者。独立评估下面每一条，不能遗漏。仅根据提供的原始决策理由、已有观察、实际动作和原假设评估，不产生新物理假设，不写新动作，不合并，不归纳Skill。你不知道历史成功案例，也不能调用任何工具或读取其他文件。标识已匿名。输出中文解释；每条解释简洁但具体。
目标仅为穿着者LS/RS袖口目标点、LH/RH真实下摆角显露或可能具体影响其显露的关系。不得预设抓取侧、方向或机制。不得仅凭类别标签判断；普通布片面积、缝边、标签不是语义端点。未确认原目标不等于证明失败。UNSPECIFIED不是明确预测不变。区分新观察、新解释、已验证规律，所有机制未验证。Novelty只相对于该条原始意图，不是相对于模型全部知识。
逐条评分，标准固定：
A goal_relevance 0无关，1泛泛可能影响布料但缺乏目标连接，2明确解释某个端部区域的遮挡/释放/受力关系（语义点可未确认），3直接涉及某语义端点显露及其关系。
B effect_expectation是该假设所解释效果对原意图的关系；relation_expectation是它提出的操作关系对原意图的关系。EXPLICIT_EXPECTED=明确表达，UNSPECIFIED=未提及，CONTRADICTED=与明示预测不一致，INDETERMINATE=不足以判断。预期目标与新增机制可不同，分别判定。
C operational_novelty 0没有新增关系/只是换说法，1新增部分具体关系但与原意图有重叠或不确定，2有具体动作—效果关系未在原理由表达。不得凭不同措辞打高分。new_vs_original并列比较。
D testability 0空泛，1只有现象，2有关系但关键条件或区分性预测欠缺，3有Before可见条件、作用关系、结果预测、可支持/反驳观察但存在必要条件不明，4四项都具体且可区分解释。仅有字段不够。
E evidence_strength 0无相关观察支撑，1有区域形变但端点及因果弱，2有相关端部变化的具体视觉记录且语义身份/遮挡未定，3原记录对相关语义端点变化有强支持（不升级原PROVISIONAL）。此阶段没有实际图像，不得假称独立看图。
observed_benefit_clue仅当原观察记录支持目标端部或其相关区域改善/显露线索时true，不能把假设预测当作已经发生。
relation_quote和prediction_quote必须逐字摘引原假设机制/预测中能支持评价的片段（可以完整摘录）；没有则空字符串。只评价，不能补写缺失内容。不要将其他案例的规则带入本条。
'''
REVIEW_PROMPT='''只读原始证据复核。你仅知道此匿名案例，不知道它的检索分数、排名、入选状态，也不读取其他分析。附图1为完整Before，图2为完整After。核对现存假设，不新增、修改或合并物理假设，不生成动作/Skill。目标限定为LS/RS袖口目标点、LH/RH真实下摆角。
分别检查实际效果的视觉依据、是否已在原意图明确提出、新假设多解释了什么、该新增部分是否具体、有无原文中已有的可证伪预测。普通轮廓变化、重新进入视野、缝边或面积增加不能自动视为角点显露。材料身份、画面侧/穿着者侧不明确时保留不确定性。
只能引用/评价输入中的机制和预测；existing_prediction_quote和existing_falsification_quote必须逐字摘引已有假设，不补写新预测。PROVISIONAL不升级CONFIRMED，没有实机新验证，不产生技能成功标签。
inversion_potential只表示现有假设是否值得后续研究，不要执行Goal Inversion或提出新抓点。输出中文，禁止工具调用。
'''

def eligible(r):return r['goal_relevance']>=2 and r['operational_novelty']>=1 and r['testability']>=3 and r['evidence_strength']>=1 and r['relation_expectation']!='EXPLICIT_EXPECTED'
def rank_key(r):return (-int(r['observed_benefit_clue']),-r['goal_relevance'],-r['evidence_strength'],-r['operational_novelty'],-r['testability'],r['blind_id'])
def exclusions(r):
 out=[]
 for yes,reason in [(r['goal_relevance']<2,'GOAL_CONNECTION_TOO_GENERIC'),(r['operational_novelty']<1,'NO_NEW_OPERATIONAL_RELATION'),(r['testability']<3,'INSUFFICIENT_TESTABILITY'),(r['evidence_strength']<1,'NO_OBSERVATION_SUPPORT'),(r['relation_expectation']=='EXPLICIT_EXPECTED','RELATION_ALREADY_EXPLICIT')]:
  if yes:out.append(reason)
 return out

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'protocol.json').exists():return
 bank=load(SOURCE/'hypothesis_bank.json'); obs={x['experience_id']:x for x in load(SOURCE/'experience_observations.json')}
 assert len(bank)==1222 and len(obs)==532
 shuffled=sorted(bank,key=lambda h:h['hypothesis_id']);random.Random(SEED).shuffle(shuffled)
 mapping={}; jobs=[];rawrefs={}; source_hashes={str(SOURCE/n):sha(SOURCE/n) for n in ['hypothesis_bank.json','experience_observations.json','experience_index.json']}
 for i,h in enumerate(shuffled,1):
  bid=f'B{i:04d}';eid=h['source_experience_id'];o=obs[eid];pp=SOURCE/h['evidence']['original_record'];payload=load(pp);source_hashes[str(pp)]=sha(pp)
  hh={k:v for k,v in h.items() if k not in ['hypothesis_id','source_experience_id','evidence','mismatch_ids']}
  mm=[m for m in o['observation']['mismatches'] if m['mismatch_id'] in h['mismatch_ids']]
  mm=[{k:v for k,v in m.items() if k!='mismatch_id'} for m in mm]
  raw={'blind_id':bid,'original_intention':payload['original_intention'],'executed_action':payload['indexed_executed_action'],
   'execution_information':payload['execution_information'],'observation':o['observation']['observed_effect'],
   'corners':o['observation']['corners'],'linked_comparisons':mm,'hypothesis':hh}
  # Remove incidental old IDs in free text; no historical report, ranking or manual case explanation supplied.
  raw=json.loads(re.sub(r'X\d{4}(?:-[HM]\d{3})?','[source-reference]',json.dumps(raw,ensure_ascii=False)))
  save(OUT/'inputs'/f'{bid}.json',raw)
  mapping[bid]={'hypothesis_id':h['hypothesis_id'],'source_experience_id':eid,'source':o['source'],'original_hypothesis':h,
   'before':str(SOURCE/h['evidence']['before']),'after':str(SOURCE/h['evidence']['after']),'payload':str(pp)}
 for n in range(0,len(mapping),25):jobs.append({'job_id':f'text_{n//25:03d}','ids':list(mapping)[n:n+25]})
 save(OUT/'mapping.json',mapping);save(OUT/'text_jobs.json',jobs)
 (OUT/'screen_prompt.txt').write_text(SCREEN_PROMPT);(OUT/'review_prompt.txt').write_text(REVIEW_PROMPT)
 save(OUT/'screen_schema.json',SCREEN_SCHEMA);save(OUT/'review_schema.json',REVIEW_SCHEMA)
 protocol={'version':'bank_retrieval_v1','frozen_at':time.time(),'seed':SEED,'hypotheses':len(bank),'operations':len(obs),
 'profile':None,'command':'codex exec (NO -p)','model':'gpt-6-astra','reasoning_effort':'medium','provider':'openai',
 'batch_size':25,'text_jobs':len(jobs),'workers':2,'max_attempts':2,'timeout_s':900,'visual_top_limit':15,'random_rejected_sample':10,
 'candidate_rule':'A>=2,C>=1,D>=3,E>=1,relation_expectation != EXPLICIT_EXPECTED; no category/tag prefilter',
 'ranking':'observed benefit clue DESC, A DESC, E DESC, C DESC, D DESC, blind_id ASC',
 'random_sampling':'random.Random(seed+1).sample(sorted excluded blind IDs, 10); no reranking after review',
 'isolation':'Fresh ephemeral context per request; anonymous IDs; no historical reports or conversation. Rows within text batch share context; not a fully blinded independent measurement.',
 'known_case_diagnostic':'Only after analysis_frozen.json; source case ID supplied at separate diagnostic command.',
 'source_hashes':source_hashes,'code_sha256':sha(__file__),
 'prompt_hashes':{n:sha(OUT/n) for n in ['screen_prompt.txt','review_prompt.txt','screen_schema.json','review_schema.json']},
 'limits':'No new hypotheses, no skills, no physical actions. Text scoring is LLM retrieval judgment, not validated knowledge or success.'}
 save(OUT/'protocol.json',protocol);save(OUT/'progress.json',{'stage':'PREPARED','processed_hypotheses':0,'total_hypotheses':1222})
 print('PREPARED 1222 hypotheses / 49 text requests; anonymous mapping frozen',flush=True)

def verify():
 p=load(OUT/'protocol.json')
 if sha(__file__)!=p['code_sha256']:raise ValueError('Runner changed since freeze')
 for f,h in p['source_hashes'].items():
  if sha(f)!=h:raise ValueError('Source changed: '+f)
 for f,h in p['prompt_hashes'].items():
  if sha(OUT/f)!=h:raise ValueError('Prompt/schema changed')
 return p

def call(job,stage,stop):
 folder=OUT/'calls'/job['job_id'];folder.mkdir(parents=True,exist_ok=True)
 if (folder/'result.json').exists():
  r=load(folder/'result.json')
  if r['status']=='COMPLETE':return r
 raw=[load(OUT/'inputs'/f'{b}.json') for b in job['ids']]
 prompt=(SCREEN_PROMPT if stage=='text' else REVIEW_PROMPT)+'\n'+json.dumps(raw if stage=='text' else raw[0],ensure_ascii=False)
 schema=SCREEN_SCHEMA if stage=='text' else REVIEW_SCHEMA
 for n in range(1,3):
  ad=folder/f'attempt_{n:02d}'
  if (ad/'result.json').exists():continue
  if stop.is_set():return {'status':'NOT_PROCESSED','job_id':job['job_id']}
  ad.mkdir(parents=True,exist_ok=True);work=ad/'empty_work';work.mkdir(exist_ok=True)
  (ad/'input.txt').write_text(prompt);save(ad/'schema.json',schema)
  cmd=['codex','exec','--ephemeral','--json','-s','read-only','--skip-git-repo-check','-C',str(work),'-c','model="gpt-6-astra"','-c','model_reasoning_effort="medium"','--output-schema',str(ad/'schema.json'),'-o',str(ad/'output.json')]
  image_manifest=[]
  if stage!='text':
   from PIL import Image
   mapping=load(OUT/'mapping.json')[job['ids'][0]]
   for k in ['before','after']:
    f=Path(mapping[k]);up=ad/(k+'.jpg')
    with Image.open(f) as im:im.convert('RGB').save(up,quality=92)
    cmd.extend(['-i',str(up)]);image_manifest.append({'role':k,'source':str(f),'source_sha256':sha(f),'upload_sha256':sha(up)})
  cmd+=['-'];start=time.time();save(ad/'manifest.json',{'command':cmd,'input_sha256':sha(ad/'input.txt'),'images':image_manifest,'time':start})
  print('START',job['job_id'],n,flush=True)
  result={'status':'FAILED','job_id':job['job_id'],'retryable':True};proc=None
  try:
   with (ad/'input.txt').open() as fi,(ad/'events.jsonl').open('w') as fo,(ad/'stderr.log').open('w') as fe:
    proc=subprocess.Popen(cmd,stdin=fi,stdout=fo,stderr=fe,start_new_session=True)
    while proc.poll() is None:
     if stop.wait(.5) or time.time()-start>900:
      os.killpg(proc.pid,signal.SIGTERM)
      try:proc.wait(timeout=5)
      except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
      raise RuntimeError('Interrupted or 900s timeout')
   ev=[]
   for line in (ad/'events.jsonl').read_text().splitlines():
    try:ev.append(json.loads(line))
    except ValueError:pass
   bad=[e for e in ev if e.get('item',{}).get('type') in ['command_execution','mcp_tool_call','web_search','file_change']]
   if bad:result['retryable']=False;stop.set();raise RuntimeError('Forbidden tool call in analysis context')
   if proc.returncode:
    err=(ad/'stderr.log').read_text()+(ad/'events.jsonl').read_text()
    if any(t in err.lower() for t in ['额度不足','quota exceeded','usage limit']):result['retryable']=False;stop.set()
    raise RuntimeError('Codex failed; inspect preserved stderr/events')
   out=load(ad/'output.json');jsonschema.validate(out,schema)
   rows=out['rows'] if stage=='text' else [out]
   if sorted(r['blind_id'] for r in rows)!=sorted(job['ids']):raise ValueError('Missing/duplicate/foreign blind IDs')
   for r in rows:
    h=load(OUT/'inputs'/f'{r["blind_id"]}.json')['hypothesis']
    pairs=[('relation_quote','hypothesized_mechanism'),('prediction_quote','predicted_effect')] if stage=='text' else [('existing_prediction_quote','predicted_effect'),('existing_falsification_quote','falsification_condition')]
    for field,source in pairs:
     if r[field] and r[field] not in h[source]:raise ValueError('Non-verbatim quote: '+field+' '+r['blind_id'])
   usage=next((e['usage'] for e in reversed(ev) if 'usage'in e),None)
   result={'status':'COMPLETE','job_id':job['job_id'],'output':out,'usage':usage,'retryable':False}
  except Exception as e:result['error']=str(e)
  result.update(latency_s=time.time()-start,finished_at=time.time());save(ad/'result.json',result)
  print('END',job['job_id'],result['status'],round(result['latency_s']),result.get('error',''),flush=True)
  if result['status']=='COMPLETE' or not result['retryable']:save(folder/'result.json',result);return result
 save(folder/'result.json',result);return result

def batch(jobs,stage):
 stop=threading.Event();start=time.time();done=0
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  pending={};it=iter(jobs)
  def submit():
   if stop.is_set():return
   j=next(it,None)
   if j:pending[pool.submit(call,j,stage,stop)]=j
  for _ in range(2):submit()
  last=0
  while pending:
   ready,_=concurrent.futures.wait(pending,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in ready:
    job=pending.pop(f);r=f.result();done+=1;submit()
   if time.time()-last>15 or ready:
    complete=sum((OUT/'calls'/j['job_id']/'result.json').exists() and load(OUT/'calls'/j['job_id']/'result.json')['status']=='COMPLETE' for j in jobs)
    save(OUT/'progress.json',{'stage':stage,'jobs_complete':complete,'jobs_total':len(jobs),'elapsed_s':time.time()-start,'active':[j['job_id'] for j in pending.values()]});last=time.time()
 if any(not (OUT/'calls'/j['job_id']/'result.json').exists() or load(OUT/'calls'/j['job_id']/'result.json')['status']!='COMPLETE' for j in jobs):raise RuntimeError('Incomplete stage; no ranking/absence claims allowed. Checkpoints retained.')

def rank():
 if (OUT/'ranking_frozen.json').exists():return load(OUT/'ranking_frozen.json')
 rows=[]
 for job in load(OUT/'text_jobs.json'):rows.extend(load(OUT/'calls'/job['job_id']/'result.json')['output']['rows'])
 assert len(rows)==1222 and len({r['blind_id'] for r in rows})==1222
 candidates=sorted([r for r in rows if eligible(r)],key=rank_key)
 excluded=sorted(r['blind_id'] for r in rows if not eligible(r))
 sample=random.Random(SEED+1).sample(excluded,min(10,len(excluded)))
 for r in rows:r['preliminary_candidate']=eligible(r);r['exclusion_reasons']=exclusions(r)
 save(OUT/'all_screening.json',rows)
 frozen={'frozen_at':time.time(),'screening_sha256':sha(OUT/'all_screening.json'),'candidate_ids':[r['blind_id'] for r in candidates],
 'top_ids':[r['blind_id'] for r in candidates[:15]],'random_excluded_ids':sample,'candidate_count':len(candidates),'not_selected_count':len(excluded)}
 save(OUT/'ranking_frozen.json',frozen);return frozen

def finalize():
 mapping=load(OUT/'mapping.json');rows=load(OUT/'all_screening.json');r=load(OUT/'ranking_frozen.json');reviews=[]
 for group,ids in [('top',r['top_ids']),('random_excluded',r['random_excluded_ids'])]:
  for bid in ids:
   rr=load(OUT/'calls'/('review_'+bid)/'result.json');reviews.append({'group':group,'blind_id':bid,'hypothesis_id':mapping[bid]['hypothesis_id'],'source_experience_id':mapping[bid]['source_experience_id'],'review':rr['output']})
 full=[dict(x,**mapping[x['blind_id']]) for x in rows]
 report={'protocol':load(OUT/'protocol.json'),'coverage':{'total':1222,'screened':len(rows),'pending':0},'ranking':r,
 'candidate_source_operations':len({mapping[b]['source_experience_id'] for b in r['candidate_ids']}),'all_hypotheses':full,'reviews':reviews,
 'limits':['检索和复核均为模型判断，不是独立物理真值。','相对于原意图新增解释不等于模型从未知道该关系。','仅复核前15条和10条未入选抽样，其余候选保留未复核。','不合并、不新增机制、不生成动作或Skill。']}
 save(OUT/'retrieval_analysis.json',report)
 save(OUT/'analysis_frozen.json',{'frozen_at':time.time(),'analysis_sha256':sha(OUT/'retrieval_analysis.json'),'ranking_sha256':sha(OUT/'ranking_frozen.json')})
 save(OUT/'progress.json',{'stage':'COMPLETE','hypotheses':len(rows),'candidates':len(r['candidate_ids']),'visual_reviews':len(reviews)})
 verify()
 print('COMPLETE',len(rows),'screened;',len(r['candidate_ids']),'candidates;',len(reviews),'reviews',flush=True)

def diagnostic(eid):
 freeze=load(OUT/'analysis_frozen.json');assert sha(OUT/'retrieval_analysis.json')==freeze['analysis_sha256']
 a=load(OUT/'retrieval_analysis.json');rows=[r for r in a['all_hypotheses'] if r['source_experience_id']==eid]
 for row in rows:
  ids=a['ranking']['candidate_ids'];row['frozen_candidate_rank']=ids.index(row['blind_id'])+1 if row['blind_id'] in ids else None
 save(OUT/'postfreeze_known_case_diagnostic.json',{'experience_id':eid,'checked_after_analysis_freeze':True,'freeze':freeze,'rows':rows,'ranking_modified':False})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run','diagnostic']);ap.add_argument('--experience-id');args=ap.parse_args()
 if args.mode=='prepare':prepare()
 elif args.mode=='diagnostic':diagnostic(args.experience_id)
 else:
  prepare();verify();batch(load(OUT/'text_jobs.json'),'text');r=rank();jobs=[{'job_id':'review_'+b,'ids':[b]} for b in r['top_ids']+r['random_excluded_ids']];batch(jobs,'review');finalize()
