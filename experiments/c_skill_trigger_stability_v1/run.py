"""Nine independent trigger-only GPT calls; no action generation or robot imports."""
from pathlib import Path
import importlib.util,json,shutil,concurrent.futures,collections,itertools,statistics
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'contrastive_skill_discovery_v1'
spec=importlib.util.spec_from_file_location('audited_transport',PARENT/'run.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
t.OUT=HERE;t.CLEAN=Path('/tmp/c_skill_trigger_stability_v1_clean');t.CLEAN.mkdir(exist_ok=True)
SCHEMA=t.obj({'trigger':t.enum('YES','NO','UNKNOWN'),'reason':t.S,'uncertainty':t.S})
SOURCE=PARENT/'skills/C_deployment.json'

def prepare():
 if (HERE/'protocol.json').exists():return
 assert t.sha(SOURCE)==t.read(PARENT/'skills/deployment_freeze.json')['sha256']['C']
 shutil.copyfile(SOURCE,HERE/'C_skill_frozen.json')
 skill=t.read(SOURCE)
 prompt='仅根据本次附图的当前Before RGB，判断以下固定Skill的trigger是否满足。YES=可见证据支持满足触发条件；NO=可见证据支持不满足触发条件；UNKNOWN=当前图像不足以确定。说明可观察的判断依据和不确定性。只判断触发，不生成Pa/Pb或其他动作，不预测操作效果，不修改Skill。只输出指定JSON。\n固定Skill：'+json.dumps(skill,ensure_ascii=False)
 (HERE/'prompt_frozen.txt').write_text(prompt);(HERE/'base_frozen.txt').write_text(t.BASE);t.save(HERE/'schema_frozen.json',SCHEMA)
 jobs=[(case,rep) for rep in [1,2,3] for case in ['E29','E30','E35']]
 t.save(HERE/'protocol.json',{'at':t.now(),'model':'gpt-6-astra','effort':'high','provider':'native openai; no api profile','cases':['E29','E30','E35'],'independent_calls_per_image':3,'planned_total_calls':9,'max_concurrency':2,'execution_order':[{'case':c,'repeat':r} for c,r in jobs],'model_visible_case_ids':False,'model_visible_repeat_ids':False,'skill_representation':'Exact frozen four operational fields previously injected in Planner Replay; not full seven-field evaluation JSON','skill_source':str(SOURCE),'skill_sha256':t.sha(SOURCE),'prompt_sha256':t.sha(HERE/'prompt_frozen.txt'),'base_sha256':t.sha(HERE/'base_frozen.txt'),'schema_sha256':t.sha(HERE/'schema_frozen.json'),'prompt_policy':'Previous shared BASE unchanged; one trigger-only task instruction frozen before first request, identical across all 9 calls; no prompt optimization or between-call edits','after_images':False,'actions_supplied':False,'effect_labels_supplied':False,'robot_execution':False,'scientific_retry_budget':0,'native_transport_retries':'CLI defaults; inspect event logs','stopping':'Exactly nine logical requests; no extra samples based on consistency','metrics':['per-image counts YES/NO/UNKNOWN','unanimity','majority fraction','within-image pairwise agreement','UNKNOWN rate'],'unknown_is_separate_category':True,'comparison_limit':'This measures stability under a fixed trigger-only context, not whether the previous full-skill historical-action context and four-field Planner context should agree.'})
 t.save(HERE/'data_manifest.json',{'cases':[{'id':c,'source':str(t.DATA/f'cases/{c}/before.png'),'sha256':t.sha(t.DATA/f'cases/{c}/before.png')} for c in ['E29','E30','E35']],'no_split_files_or_outcome_labels_read':True,'previous_coordinator_had_seen_results':'Coordinator already knows earlier E30 discrepancy and prior evaluation results; those are not passed to isolated callers.'})

def verify_fixed():
 p=t.read(HERE/'protocol.json')
 for f,k in [('C_skill_frozen.json','skill_sha256'),('prompt_frozen.txt','prompt_sha256'),('base_frozen.txt','base_sha256'),('schema_frozen.json','schema_sha256')]:assert t.sha(HERE/f)==p[k],f
 assert t.sha(SOURCE)==p['skill_sha256'];assert t.BASE==(HERE/'base_frozen.txt').read_text()

def one(job):
 verify_fixed();case,rep=job
 result=t.call(f'calls/{case}_r{rep}',(HERE/'prompt_frozen.txt').read_text(),[t.DATA/f'cases/{case}/before.png'],SCHEMA)
 assert set(result)=={'trigger','reason','uncertainty'}
 return result

def report():
 verify_fixed();rows=[];checks=[]
 def ck(name,ok):checks.append({'check':name,'pass':bool(ok)})
 by_case={};costs=[];audits=[];transport=[]
 proto=t.read(HERE/'protocol.json')
 for case in proto['cases']:
  rs=[]
  for rep in [1,2,3]:
   f=HERE/f'calls/{case}_r{rep}';r=t.read(f/'parsed.json');meta=t.read(f/'result.json');manifest=t.read(f/'input_manifest.json');audit=t.read(f/'context_check.json')
   row={'case':case,'repeat':rep,**r,'wall_s':meta['wall_s'],'usage':meta['usage']};rows.append(row);rs.append(row);costs.append({'case':case,'repeat':rep,**meta});audits.append({'case':case,'repeat':rep,**audit})
   ck(f'{case}/{rep} prompt unchanged',t.sha(f/'prompt.txt')==proto['prompt_sha256']);ck(f'{case}/{rep} base unchanged',t.sha(f/'base.txt')==proto['base_sha256']);ck(f'{case}/{rep} schema unchanged',t.sha(f/'schema.json')==proto['schema_sha256']);ck(f'{case}/{rep} no actions generated',set(r)=={'trigger','reason','uncertainty'});ck(f'{case}/{rep} isolated context',not audit['forbidden_hits'] and audit['returncode']==0);ck(f'{case}/{rep} no tool calls',meta['tool_events']==[]);ck(f'{case}/{rep} success',meta['status']=='COMPLETE');ck(f'{case}/{rep} correct image',len(manifest['attachments'])==1 and manifest['attachments'][0]['sha256']==t.sha(t.DATA/f'cases/{case}/before.png'));ck(f'{case}/{rep} attachment bytes unchanged',t.sha(f/'attachments/01.png')==t.sha(t.DATA/f'cases/{case}/before.png'))
   es=[json.loads(x) for x in (f/'events.jsonl').read_text().splitlines()];ck(f'{case}/{rep} one completed turn',sum(e.get('type')=='turn.completed' for e in es)==1)
   messages=[e.get('message',e.get('item',{}).get('message','')) for e in es];transport += [{'case':case,'repeat':rep,'message':m} for m in messages if 'Reconnect' in m or 'Falling back' in m or 'usage limit' in m]
  counts={v:sum(r['trigger']==v for r in rs) for v in ['YES','NO','UNKNOWN']};agree=sum(a['trigger']==b['trigger'] for a,b in itertools.combinations(rs,2));by_case[case]={'counts':counts,'sequence':[r['trigger'] for r in rs],'unanimous':max(counts.values())==3,'majority_fraction':max(counts.values())/3,'pairwise_agreement':{'numerator':agree,'denominator':3,'rate':agree/3},'unknown_rate':counts['UNKNOWN']/3,'judgments':rs}
 all_agree=sum(v['pairwise_agreement']['numerator'] for v in by_case.values());summary={'at':t.now(),'cases':by_case,'unanimous_image_count':sum(x['unanimous'] for x in by_case.values()),'image_count':3,'within_image_pairwise_agreement':{'numerator':all_agree,'denominator':9,'rate':all_agree/9},'unknown_count':sum(r['trigger']=='UNKNOWN' for r in rows),'request_count':9,'not_physical_or_trigger_ground_truth':True}
 t.save(HERE/'results.json',summary);t.save(HERE/'costs.json',{'requests':costs,'request_count':len(costs),'known_totals':{k:sum((r['usage'] or {}).get(k,0) for r in costs) for k in ['input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens']},'mean_wall_s':statistics.mean(r['wall_s'] for r in rows),'currency_cost':None,'transport_incidents':transport,'usage_may_exclude_failed_native_transport_attempts':True})
 ck('exactly nine logical requests',len(list((HERE/'calls').glob('*/attempt.json')))==9);ck('unique independent sessions',len({json.loads((f/'events.jsonl').read_text().splitlines()[0])['thread_id'] for f in (HERE/'calls').iterdir()})==9)
 t.save(HERE/'context_audit.json',{'checks':audits,'forbidden_project_context_hits':sum(len(a['forbidden_hits']) for a in audits),'prompt_snapshot_note':'Configured debug prompt-input context, plus exact prompt/base/schema and attachment hashes; not a wire packet capture','no_prior_judgments_or_case_history_sent':True})
 t.save(HERE/'validation.json',{'checks':checks,'passed':sum(c['pass'] for c in checks),'failed':sum(not c['pass'] for c in checks)})
 lines=['# C Skill 触发稳定性小实验','', '固定上一轮Planner实际部署的C Skill四个操作字段；gpt-6-astra/high，三张Before RGB各3次独立会话，共9次逻辑请求。9次共享相同Prompt、BASE和JSON schema，仅图像变化；不提供历史动作、After、标签或前次判断，不生成Pa/Pb。', '', '## 一致性', '', '|图像|第1次|第2次|第3次|YES / NO / UNKNOWN|多数占比|两两一致率|', '|---|---|---|---|---|---|---|']
 for case,d in by_case.items():lines.append('|'+case+'|'+'|'.join(d['sequence'])+'|'+ '/'.join(str(d['counts'][v]) for v in ['YES','NO','UNKNOWN'])+f"|{d['majority_fraction']:.1%}|{d['pairwise_agreement']['numerator']}/3|")
 lines+=['',f"三次完全一致的图像：{summary['unanimous_image_count']}/3；图像内两两一致：{all_agree}/9；UNKNOWN：{summary['unknown_count']}/9。UNKNOWN是独立类别；稳定输出UNKNOWN不代表稳定识别了适用状态。",'', '## 每次判断依据','', '|图像|重复|判断|依据|不确定性|','|---|---|---|---|---|']
 for r in rows:lines.append('|'+ '|'.join(str(r[k]).replace('|','/').replace('\n',' ') for k in ['case','repeat','trigger','reason','uncertainty'])+'|')
 lines+=['','## 固定Skill','','```json',json.dumps(t.read(HERE/'C_skill_frozen.json'),ensure_ascii=False,indent=2),'```','','## 解释边界','','本实验只检验相同输入条件下的重复一致性，不验证判断是否正确或动作是否有效。每图仅3次，不宣称统计显著性或稳定率的精确估计。','', '上一轮历史评价还带实际动作与完整七字段Skill，Planner使用四字段且要求生成动作。本实验使用四字段、仅判断Trigger，因而不能仅凭这9次结果将原先E30的UNKNOWN/YES差异归因于模型随机性，也不能证明跨任务上下文稳定。','', '原共享BASE保持原文。本次任务要求只判断Trigger，因此一次性固定了对应任务指令；没有在9次之间改写或优化Prompt，也未修改生产Planner Prompt。','',f"校验：{sum(c['pass'] for c in checks)}项通过、{sum(not c['pass'] for c in checks)}项失败。全部真实请求、附件、原始输出和Token/耗时分别见calls/、costs.json。"]
 (HERE/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'sequences':{k:v['sequence'] for k,v in by_case.items()},'unanimous_images':summary['unanimous_image_count'],'validation_failures':sum(not c['pass'] for c in checks)},ensure_ascii=False),flush=True)

def main():
 prepare();verify_fixed();p=t.read(HERE/'protocol.json');jobs=[(r['case'],r['repeat']) for r in p['execution_order']]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,jobs))
 report()
if __name__=='__main__':main()
