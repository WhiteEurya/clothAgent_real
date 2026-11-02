"""Run-level held-out retrieval A/B. No robot imports; never writes source evidence."""
import concurrent.futures,collections,hashlib,importlib.util,json,math,os,random,re,time
from pathlib import Path
from PIL import Image
ROOT=Path('/mnt/newssd/sja/clothAgent_real');SRC=ROOT/'reports/intention_outcome_hypothesis_bank_v1';OUT=ROOT/'reports/autonomous_retrieval_ab_20261102'
SEED=2026110223
spec=importlib.util.spec_from_file_location('model_io',ROOT/'experiments/hypothesis_guided_prediction_v1/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=OUT
obj,S,arr,enum=m.obj,m.S,m.arr,m.enum
GOALS=['LS','RS','LH','RH'];TARGETS=GOALS+['BODY'];DIRECTIONS=['IMPROVED','WORSENED','UNCHANGED','UNKNOWN'];GEO=['INCREASE','DECREASE','UNCHANGED','UNKNOWN'];METRICS=['EXPOSED_AREA','CONTOUR_EXTENSION','FOLD_COVERAGE']
QUERY=obj({'visible_structure':S,'grasp_region':S,'action_direction_and_distance':S,'queries':arr(S),'uncertainty':S})
RETRIEVAL=obj({'selection':enum('SELECTED','NONE'),'chosen':{'type':'array','maxItems':3,'items':obj({'hypothesis_id':S,'applicability_evidence':S,'potential_additional_prediction_information':S,'limitations':S})},'none_or_selection_reason':S})
PRED=obj({'regions':arr(obj({'goal':enum(*GOALS),'image_region_description':S,'semantic_identity_confidence':enum('CLEAR','PROVISIONAL','UNKNOWN'),'visibility_change':enum(*DIRECTIONS),'strict_corner_reveal':enum('YES','NO','UNKNOWN'),'basis':S,'uncertainty':S})),
 'operated_region_local_deformation':S,'non_direct_region_changes':S,'global_motion_or_out_of_view':S,
 'static_effect':obj({'target':enum(*TARGETS,'UNKNOWN'),'metric':enum(*METRICS),'change':enum(*GEO),'checkable_statement':S,'basis':S,'uncertainty':S}),
 'overall_uncertainty':S})
OBSVAL=obj({'value':enum(*DIRECTIONS),'quality':enum('CLEAR','PROVISIONAL','INSUFFICIENT'),'evidence':S})
GEOVAL=obj({'value':enum(*GEO),'quality':enum('CLEAR','PROVISIONAL','INSUFFICIENT'),'evidence':S})
GEOMETRY=obj({k:GEOVAL for k in METRICS})
OBS=obj({'view_comparability':S,'regions':arr(obj({'goal':enum(*GOALS),'image_region_description':S,'semantic_correspondence':enum('CLEAR','PROVISIONAL','UNKNOWN'),
 'direct_grasp_relation':enum('DIRECT','NON_DIRECT','UNKNOWN'),'grasp_relation_basis':S,'visibility_change':OBSVAL,
 'strict_corner_reveal':enum('CONFIRMED','PROVISIONAL','NO_REVEAL','UNKNOWN'),'reveal_evidence':S,'geometry':GEOMETRY})),
 'body_geometry':GEOMETRY,'operated_region_observed_deformation':S,'global_motion_and_out_of_view':S,'limits':S})
COMMON='你是独立离线Codex。仅根据本请求信息回答，不调用任何工具、不读文件、不联网、不读其他会话。使用中文JSON。禁止新机器人动作、生成新假设或Skill、修改已有假设。'
DEFS='''Corner Reveal任务定义：LS/RS为穿着者左/右袖口目标区域，LH/RH为穿着者左/右真实下摆角区域。画面左右不一定等于语义左右；衣服旋转时按其结构识别。身份不清保留UNKNOWN。
四个语义端点区域的可见性分IMPROVED/WORSENED/UNCHANGED/UNKNOWN；这是区域可辨识程度，不要求精确材料中点追踪。严格Corner Reveal是同一真实语义端点从自遮挡到显露，必须单独报告，不能把轮廓变宽、区域更清楚、折线或标签出现视为Reveal。端点出入视野及整体移动/旋转另行注明。
静态几何指标：EXPOSED_AREA=同一目标区域可见投影布面范围；CONTOUR_EXTENSION=该区域轮廓相对其连接部位的伸展程度，排除仅整体平移；FOLD_COVERAGE=可观察的局部折叠覆盖范围（看不清覆盖层界就UNKNOWN）。它们取INCREASE/DECREASE/UNCHANGED/UNKNOWN。BODY指衣身主体。不要以全衣面积代替袖部指标，不输出未标定的精确百分比。
命令/索引参数不等于测得布料运动或夹持成功。索引与底层指令不同则保留差异。UNKNOWN不是不变，更不是失败。没有过程图像时不能确认受力、接触滞留或运动先后。'''
QP=COMMON+DEFS+'''你只看到新Before RGB和真实记录动作，尚不知道After或实际效果。生成用于检索历史操作假设的1至3条具体中文查询，依据可见结构、抓点所处区域、方向和距离。描述与动作可能关联的几何/语义条件，但不能将未来变化写成事实。不要预设哪一部位一定改善，不编造历史成功结果。'''
RP=COMMON+DEFS+'''你看到新Before及记录动作，以及从隔离训练库按当前查询检索到的最多24条既有假设。仅依据适用条件和动作关系自主选择0至3条有用假设，不需凑满；没有适用条目返回NONE和空列表。保留竞争机制；不得改写或新建假设。说明每条为何可能适用、可能提供哪些额外预测信息、哪些条件未知。历史结果只是历史事实，不是新状态结果。选择ID必须来自提供列表。不能按检索分数直接当真或把机制当成已验证规律。'''
PP=COMMON+DEFS+'''请预测此Before下真实记录动作的最终效果。你没有After、原始Agent理由、结果标签或历史案例报告。对LS/RS/LH/RH各输出一次可见性变化及严格Reveal预测，不得遗漏或重复。还要预测被操作区域局部形变、非直接抓取区域可能变化、整体移动/旋转或出画。
提供一个在静态After中可直接检查的具体几何效果，使用规定target/metric/change。证据不足可UNKNOWN并说明。不要为了覆盖四个部位而预测它们全部变化；也不要只写空泛条件句来避免给出最佳判断。每项分别保留不确定性，不能事后补写。若提供历史假设，可利用或拒绝它，但不将其当作新案例事实。不得假设没有提供的假设，也不写新机制。'''
OP=COMMON+DEFS+'''你是独立结果评价器。图1Before，图2After，并提供记录动作以定位抓点；没有任何预测、组别、检索假设或选择理由。逐一观察LS/RS/LH/RH，不能遗漏。可见性判断关注区域可辨识程度，不额外要求已定位精确袖口中点；但身份/区域对应不清仍须UNKNOWN或PROVISIONAL。有明确视觉证据才标CLEAR。
逐区域另外判断严格Corner Reveal，以及三个静态几何指标。区分区域可辨性、材料端点解除自遮挡、刚性位移/旋转投影和出入视野。直接操作关系按Before抓点位于哪一语义区域判断：DIRECT、NON_DIRECT、UNKNOWN。这只是几何抓点关系，不等于知道原Agent目标或真实夹住哪层布料。BODY抓取可使四角均为NON_DIRECT；不能从After反推抓取身份。
对每个判断引用可见轮廓/缝线/文字/邻接关系；边界不足不得硬猜。不能从命令完成推断抓取成功。只输出观察，不产生新动作或物理假设。'''

def toks(text):
 text=text.lower();latin=re.findall(r'[a-z0-9_]+',text);chunks=re.findall(r'[\u4e00-\u9fff]+',text)
 return latin+[s[i:i+2] for s in chunks for i in range(len(s)-1)]
def duplicate_keys(r):
 keys={('source',str(Path(r['source']).resolve())),('index',r['source_index_id']),('pair',r['before_sha256'],r['after_sha256']),('original_pair',str(Path(r['original_before']).resolve()),str(Path(r['original_after']).resolve()))}
 return keys
def prepare():
 if (OUT/'protocol.json').exists():return
 OUT.mkdir(parents=True)
 idx=m.load(SRC/'experience_index.json');bank=m.load(SRC/'hypothesis_bank.json');assert len(idx)==532 and len(bank)==1222
 rng=random.Random(SEED);runs=sorted({r['run'] for r in idx});rng.shuffle(runs);test_runs=runs[:4]
 testrows=[r for r in idx if r['run'] in test_runs];excluded_ids={r['experience_id'] for r in testrows};keys=set().union(*(duplicate_keys(r) for r in testrows));duplicates=[]
 while True:
  add=[r for r in idx if r['experience_id'] not in excluded_ids and duplicate_keys(r)&keys]
  if not add:break
  for r in add:excluded_ids.add(r['experience_id']);keys.update(duplicate_keys(r));duplicates.append(r['experience_id'])
 held_ids=set(excluded_ids)
 # Remove any train hypothesis explicitly referencing held-out operations or result paths.
 forbidden_tokens=test_runs+[r['source'] for r in idx if r['experience_id'] in held_ids]+[r['after_sha256'] for r in idx if r['experience_id'] in held_ids]
 allowed=[];excluded_h=[]
 sourceby={r['experience_id']:r for r in idx}
 fields=['hypothesis_id','source_experience_id','intended_effect','observed_effect','mismatch','observable_conditions','hypothesized_mechanism','predicted_effect','falsification_condition','evidence_level','uncertainty','related_corner_goals','physical_mechanism_validated']
 for h in bank:
  serialized=json.dumps(h,ensure_ascii=False);refs=set(re.findall(r'X\d{4}',serialized));reasons=[]
  if h['source_experience_id'] in excluded_ids:reasons.append('HELDOUT_RUN_OR_DUPLICATE_EXECUTION')
  if refs&held_ids:reasons.append('EXPLICIT_HELDOUT_CASE_REFERENCE')
  if any(token in serialized for token in forbidden_tokens):reasons.append('HELDOUT_PATH_OR_HASH_REFERENCE')
  if reasons:excluded_h.append({'hypothesis_id':h['hypothesis_id'],'source_experience_id':h['source_experience_id'],'reasons':reasons});continue
  entry={k:h[k] for k in fields};entry['source_run']=sourceby[h['source_experience_id']]['run'];allowed.append(entry)
 eligible=[];invalid=[]
 for r in testrows:
  p=m.load(r['payload']);act=p['indexed_executed_action']
  if p['execution_information'].get('status')=='EXECUTED' and act.get('grasp_px') and act.get('target_px') and Path(r['before']).is_file() and Path(r['after']).is_file():eligible.append(r)
  else:invalid.append({'experience_id':r['experience_id'],'reason':'missing verified executed action metadata or pair file'})
 # Two per run where possible, then fill remaining slots from these same four runs.
 chosen=[];remaining=[]
 for run in test_runs:
  pool=sorted([r for r in eligible if r['run']==run],key=lambda r:r['experience_id']);rng.shuffle(pool);chosen+=pool[:2];remaining+=pool[2:]
 rng.shuffle(remaining);chosen=(chosen+remaining)[:8];sample=[]
 for n,r in enumerate(chosen,1):
  cid=f'T{n:02}';p=m.load(r['payload']);m.save(OUT/'actions'/f'{cid}.json',{'indexed_action':p['indexed_executed_action'],'native_command_sequence':p['native_command_sequence'],'limits':'Executed command record; material acquisition not inferred.'})
  sample.append({'id':cid,'experience_id':r['experience_id'],'run':r['run'],'source':r['source'],'before':r['before'],'after':r['after'],'before_sha256':r['before_sha256'],'after_expected_sha256':r['after_sha256'],'payload':r['payload'],'payload_sha256':m.sha(r['payload'])})
 m.save(OUT/'sample_manifest.json',sample);m.save(OUT/'training_index.json',allowed);os.chmod(OUT/'training_index.json',0o444)
 m.save(OUT/'isolation_audit.json',{'test_runs':test_runs,'whole_test_run_operations':len(testrows),'heldout_ids':sorted(held_ids),'duplicate_expansion_ids':duplicates,'excluded_hypotheses':excluded_h,'allowed_hypotheses':len(allowed),'invalid_test_records':invalid,'test_run_choice_uses':'run identifiers only, no After or result labels','exclusion_fields':'canonical source, source_index_id, canonical original pair paths, paired image hashes; transitive closure; explicit cross-case ID/path/hash scan','derived_reports_used':[],'limits':'No prior retrieval reports or case lists read. No semantic guarantee against unmarked paraphrased cross-case content; source mining was recorded as per-experience. Fresh analysis sessions cannot access files via tools.'})
 jobs=[{'id':r['id'],'group':g} for r in sample for g in ['Baseline','Hindsight']];rng.shuffle(jobs);m.save(OUT/'prediction_order.json',jobs)
 prompts={'query':QP,'retrieval':RP,'prediction':PP,'observation':OP}
 for k,v in prompts.items():(OUT/f'{k}_prompt.txt').write_text(v)
 for k,v in [('query',QUERY),('retrieval',RETRIEVAL),('prediction',PRED),('observation',OBS)]:m.save(OUT/f'{k}_schema.json',v)
 m.save(OUT/'protocol.json',{'version':'autonomous_retrieval_ab_v1','seed':SEED,'frozen_at':time.time(),'sample_count':len(sample),'test_runs':test_runs,'test_runs_count':len(test_runs),'source_counts':{'operations':len(idx),'hypotheses':len(bank)},'train_hypotheses':len(allowed),
 'sampling':'Uniform seeded permutation of all run IDs, first four entire runs held out. Up to two random eligible actions per run; fill to eight within same four runs, no replacement based on applicability or outcomes.',
 'retrieval':'Fresh image/action query generation (1-3 queries), BM25 over every allowed hypothesis, top24 text shortlist, fresh visual selector chooses0-3. No manually nominated hypotheses, no known-case/Top15 reports. Shortlist recall is a limitation.',
 'bm25':{'k1':1.5,'b':0.75,'tokenizer':'Chinese character bigrams + Latin alphanumeric words','document_fields':'observable_conditions x2, mechanism, prediction, observed_effect, intended_effect, uncertainty','shortlist':24},
 'conditions':'Same Before/action/task prompt/schema/model. Hindsight prediction receives selected existing hypotheses only, not query/selector reasoning; Baseline no hypotheses. Selector rationale saved separately. Fresh session for every call. NONE still tested.',
 'freeze':'All predictions frozen and hashed before first test After content read. Independent observer sees Before+After+action but no prediction, hypothesis or group.',
 'scoring':'Four endpoint-region visibility items plus one static geometric item. CLEAR semantic correspondence and CLEAR nonUNKNOWN observation required. Unknown predicted ->NA_ABSTAIN; unknown observation ->NA_REFERENCE. Else label match correct or incorrect. Strict Reveal separately, never inferred from ordinary visibility improvement.',
 'non_direct':'Observer maps grasp location to DIRECT/NON_DIRECT/UNKNOWN; NON_DIRECT is geometric non-grasp region, not inferred original planning intent. Changed clear regions are denominator; directionally predicted matching changes are hits.',
 'unsupported':'Changed prediction contradicted by clear UNCHANGED/opposite label counted contradicted. Unknown reference is NA, not unsupported or wrong. No inference from absent evidence.',
 'specific_information':'Compare fixed semantic/geometry predictions, not verbosity. Free-text deformation predictions preserved but not post-hoc given subjective success points. Same-region paired accuracy and additional correct non-direct changes reported.',
 'limits':'At most8 actions and4run clusters, one stochastic draw percondition, descriptive only. Blind model judgments are proxies not physical ground truth. No prior-case blacklist used to pick runs; coordinator may have seen some raw cases previously, independent model requests receive no prior conclusions.',
 'models':{'model':'gpt-6-astra','effort':'medium','profile':None,'workers':2,'attempts_max':2},'source_bank_sha256':m.sha(SRC/'hypothesis_bank.json'),'source_index_sha256':m.sha(SRC/'experience_index.json'),'training_index_sha256':m.sha(OUT/'training_index.json'),'runner_sha256':m.sha(__file__),'call_helper_sha256':m.sha(ROOT/'experiments/hypothesis_guided_prediction_v1/run.py'),'prompt_hashes':{k:m.sha(OUT/f'{k}_prompt.txt') for k in prompts},'no_robot':True,'no_new_hypotheses':True})
 print('PREPARED',len(sample),'cases /',len(testrows),'held-out operations /',len(allowed),'train hypotheses',flush=True)

def bm25(queries,index):
 docs=[]
 for h in index:
  text=h['observable_conditions']+' '+h['observable_conditions']+' '+' '.join(h[k] for k in ['hypothesized_mechanism','predicted_effect','observed_effect','intended_effect','uncertainty']);docs.append(collections.Counter(toks(text)))
 lengths=[sum(d.values()) for d in docs];avg=sum(lengths)/len(lengths);df=collections.Counter(t for d in docs for t in d);q=set(toks(' '.join(queries)));N=len(docs);rank=[]
 for h,doc,dl in zip(index,docs,lengths):
  score=0.0
  for t in q:
   tf=doc[t]
   if tf:score+=math.log(1+(N-df[t]+.5)/(df[t]+.5))*tf*2.5/(tf+1.5*(1-.75+.75*dl/avg))
  rank.append({'hypothesis_id':h['hypothesis_id'],'score':score})
 rank.sort(key=lambda x:(-x['score'],x['hypothesis_id']));return rank
def validate_goals(result,key='regions'):
 assert sorted(x['goal'] for x in result[key])==sorted(GOALS),'Missing/duplicated semantic region'
def parallel(jobs,fn,label):
 results={};completed=0
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
  fut={ex.submit(fn,j):j for j in jobs}
  for f in concurrent.futures.as_completed(fut):
   j=fut[f];key=j if isinstance(j,str) else j['id']+'_'+j['group'];results[key]=f.result();completed+=1
   m.save(OUT/'progress.json',{'stage':label,'completed':completed,'total':len(jobs),'updated_at':time.time()});print(label,completed,'/',len(jobs),flush=True)
 return results

def main():
 prepare();protocol=m.load(OUT/'protocol.json');assert m.sha(__file__)==protocol['runner_sha256'];assert m.sha(OUT/'training_index.json')==protocol['training_index_sha256']
 rows=m.load(OUT/'sample_manifest.json');by={r['id']:r for r in rows};ids=list(by);index=m.load(OUT/'training_index.json');hb={h['hypothesis_id']:h for h in index}
 def base(cid):
  row=by[cid]
  with Image.open(row['before']) as im:size=list(im.size)
  return {'image_size':size,'recorded_action':m.load(OUT/'actions'/f'{cid}.json')}
 def retrieve(cid):
  row=by[cid];query=m.call('query_'+cid,QP,base(cid),QUERY,[row['before']]);assert 1<=len(query['queries'])<=3
  ranks=bm25(query['queries'],index);m.save(OUT/'retrieval'/f'{cid}_all_scores.json',ranks);short=[hb[r['hypothesis_id']] for r in ranks[:24]]
  data={**base(cid),'current_observation_query':query,'candidate_hypotheses':short}
  selected=m.call('retrieve_'+cid,RP,data,RETRIEVAL,[row['before']]);chosen=[r['hypothesis_id'] for r in selected['chosen']]
  assert len(chosen)==len(set(chosen)) and set(chosen)<={h['hypothesis_id'] for h in short}
  assert (selected['selection']=='NONE')==(len(chosen)==0)
  m.save(OUT/'retrieval'/f'{cid}.json',{'query':query,'shortlist_ids':[h['hypothesis_id'] for h in short],'selection':selected,'selected_hypotheses':[hb[h] for h in chosen]});return selected
 retrieval=parallel(ids,retrieve,'RETRIEVAL')
 jobs=m.load(OUT/'prediction_order.json')
 def predict(j):
  cid=j['id'];data=base(cid)
  if j['group']=='Hindsight':data['retrieved_hypotheses']=m.load(OUT/'retrieval'/f'{cid}.json')['selected_hypotheses']
  out=m.call('predict_'+cid+'_'+j['group'],PP,data,PRED,[by[cid]['before']]);validate_goals(out);return out
 outputs=parallel(jobs,predict,'PREDICTION');pred={cid:{g:outputs[cid+'_'+g] for g in ['Baseline','Hindsight']} for cid in ids};m.save(OUT/'predictions.json',pred)
 if not (OUT/'predictions_frozen.json').exists():m.save(OUT/'predictions_frozen.json',{'frozen_at':time.time(),'predictions_sha256':m.sha(OUT/'predictions.json'),'call_outputs':{key:m.sha(Path(m.load(OUT/'calls'/('predict_'+key)/'result.json')['attempt_dir'])/'output.json') for key in outputs},'retrieval_hashes':{cid:m.sha(OUT/'retrieval'/f'{cid}.json') for cid in ids},'training_index_sha256':m.sha(OUT/'training_index.json')})
 frozen=m.load(OUT/'predictions_frozen.json');assert m.sha(OUT/'predictions.json')==frozen['predictions_sha256'];print('ALL PREDICTIONS FROZEN; NOW REVEAL AFTER',flush=True)
 if not (OUT/'after_disclosure.json').exists():
  hashes={cid:m.sha(by[cid]['after']) for cid in ids}
  for cid in ids:assert hashes[cid]==by[cid]['after_expected_sha256']
  m.save(OUT/'after_disclosure.json',{'disclosed_at':time.time(),'after_sha256':hashes,'prediction_freeze_sha256':m.sha(OUT/'predictions_frozen.json')})
 def observe(cid):
  out=m.call('observe_'+cid,OP,{'recorded_action':base(cid)['recorded_action']},OBS,[by[cid]['before'],by[cid]['after']]);validate_goals(out);return out
 observations=parallel(ids,observe,'OBSERVATION')
 for r in rows:assert m.sha(r['payload'])==r['payload_sha256'];assert m.sha(r['before'])==r['before_sha256']
 assert m.sha(SRC/'hypothesis_bank.json')==protocol['source_bank_sha256'];assert m.sha(OUT/'predictions.json')==frozen['predictions_sha256']
 m.save(OUT/'result.json',{'status':'COMPLETE','protocol':protocol,'sample':rows,'retrieval':{cid:m.load(OUT/'retrieval'/f'{cid}.json') for cid in ids},'predictions':pred,'observations':observations,'prediction_freeze':frozen,'after_disclosure':m.load(OUT/'after_disclosure.json'),'finished_at':time.time()})
 m.save(OUT/'progress.json',{'stage':'COMPLETE','cases':len(rows),'finished_at':time.time()});print('COMPLETE',len(rows),'paired cases',flush=True)
if __name__=='__main__':main()
