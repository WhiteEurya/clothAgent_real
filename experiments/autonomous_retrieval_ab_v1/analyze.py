"""Deterministic comparisons of frozen predictions with blind observation labels."""
import collections,hashlib,json
from pathlib import Path
R=Path('/mnt/newssd/sja/clothAgent_real/reports/autonomous_retrieval_ab_20261102')
def load(p):return json.loads(Path(p).read_text())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def verdict(pred,ref,judgeable):
 if not judgeable:return 'NA_REFERENCE'
 if pred=='UNKNOWN':return 'NA_ABSTAIN'
 return 'CORRECT' if pred==ref else 'INCORRECT'
def ratio(n,d):return n/d if d else None
def percentile(values,q):
 v=sorted(values)
 if not v:return None
 x=(len(v)-1)*q;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
def main():
 a=load(R/'result.json');cases={r['id']:r for r in a['sample']};vis=[];geo=[];reveal=[];pair=[];differences=[]
 for cid,preds in a['predictions'].items():
  obs={r['goal']:r for r in a['observations'][cid]['regions']};bp={r['goal']:r for r in preds['Baseline']['regions']};hp={r['goal']:r for r in preds['Hindsight']['regions']}
  for goal,ref in obs.items():
   v=ref['visibility_change'];judge=ref['semantic_correspondence']=='CLEAR' and v['quality']=='CLEAR' and v['value']!='UNKNOWN'
   rs={}
   for group,pr in [('Baseline',bp[goal]),('Hindsight',hp[goal])]:
    change=pr['visibility_change'];vv=verdict(change,v['value'],judge)
    row={'case_id':cid,'experience_id':cases[cid]['experience_id'],'run':cases[cid]['run'],'group':group,'goal':goal,'prediction':change,'reference':v['value'],'reference_evidence':v['evidence'],'reference_quality':v['quality'],'reference_correspondence':ref['semantic_correspondence'],'judgeable':judge,'verdict':vv,'direct_grasp_relation':ref['direct_grasp_relation'],
     'predicted_change':change in ['IMPROVED','WORSENED'],'observed_non_direct_change':bool(judge and ref['direct_grasp_relation']=='NON_DIRECT' and v['value'] in ['IMPROVED','WORSENED']),
     'contradicted_change_prediction':bool(judge and change in ['IMPROVED','WORSENED'] and change!=v['value']),'unknown_reference_change_prediction':bool(not judge and change in ['IMPROVED','WORSENED'])}
    vis.append(row);rs[group]=row
    rv=ref['strict_corner_reveal'];rj=ref['semantic_correspondence']=='CLEAR' and rv in ['CONFIRMED','NO_REVEAL'];target={'CONFIRMED':'YES','NO_REVEAL':'NO'}.get(rv,'UNKNOWN')
    reveal.append({'case_id':cid,'group':group,'goal':goal,'prediction':pr['strict_corner_reveal'],'reference':rv,'judgeable':rj,'verdict':verdict(pr['strict_corner_reveal'],target,rj),'evidence':ref['reveal_evidence']})
   pair.append({'case_id':cid,'experience_id':cases[cid]['experience_id'],'run':cases[cid]['run'],'goal':goal,'reference':v,'judgeable':judge,'non_direct':ref['direct_grasp_relation']=='NON_DIRECT','Baseline':rs['Baseline']['verdict'],'Hindsight':rs['Hindsight']['verdict'],
    'predictions_differ':bp[goal]['visibility_change']!=hp[goal]['visibility_change'],
    'additional_correct_non_direct_effect':rs['Hindsight']['observed_non_direct_change'] and rs['Hindsight']['verdict']=='CORRECT' and rs['Baseline']['verdict']!='CORRECT',
    'lost_correct_non_direct_effect':rs['Baseline']['observed_non_direct_change'] and rs['Baseline']['verdict']=='CORRECT' and rs['Hindsight']['verdict']!='CORRECT'})
  for group,pr in preds.items():
   se=pr['static_effect'];target=se['target'];ref=a['observations'][cid]['body_geometry'][se['metric']] if target=='BODY' else obs[target]['geometry'][se['metric']] if target in obs else None
   judge=bool(ref and ref['quality']=='CLEAR' and ref['value']!='UNKNOWN' and (target=='BODY' or obs[target]['semantic_correspondence']=='CLEAR'))
   geo.append({'case_id':cid,'group':group,'target':target,'metric':se['metric'],'prediction':se['change'],'reference':ref,'judgeable':judge,'verdict':verdict(se['change'],ref['value'] if ref else 'UNKNOWN',judge),'statement':se['checkable_statement']})
  differences.append({'case_id':cid,'visibility_difference_goals':[g for g in obs if bp[g]['visibility_change']!=hp[g]['visibility_change']],
   'strict_reveal_difference_goals':[g for g in obs if bp[g]['strict_corner_reveal']!=hp[g]['strict_corner_reveal']],
   'same_static_claim':all(preds['Baseline']['static_effect'][k]==preds['Hindsight']['static_effect'][k] for k in ['target','metric','change'])})
 summary={}
 for group in ['Baseline','Hindsight']:
  rr=[r for r in vis if r['group']==group];c=collections.Counter(r['verdict'] for r in rr);judge=sum(r['judgeable'] for r in rr);nd=[r for r in rr if r['observed_non_direct_change']];ndcorrect=sum(r['verdict']=='CORRECT' for r in nd)
  gg=[r for r in geo if r['group']==group];rv=[r for r in reveal if r['group']==group]
  summary[group]={'visibility_items':len(rr),'judgeable_visibility':judge,'reference_coverage':ratio(judge,len(rr)),'correct':c['CORRECT'],'incorrect':c['INCORRECT'],'na_abstain':c['NA_ABSTAIN'],'na_reference':c['NA_REFERENCE'],'unknown_predictions':sum(r['prediction']=='UNKNOWN' for r in rr),'unknown_ratio':ratio(sum(r['prediction']=='UNKNOWN' for r in rr),len(rr)),
   'conditional_accuracy':ratio(c['CORRECT'],c['CORRECT']+c['INCORRECT']),'scored_over_judgeable':ratio(c['CORRECT']+c['INCORRECT'],judge),'overall_scored_coverage':ratio(c['CORRECT']+c['INCORRECT'],len(rr)),'correct_over_judgeable':ratio(c['CORRECT'],judge),'non_direct_changed_items':len(nd),'correct_non_direct_changes':ndcorrect,'non_direct_change_recall':ratio(ndcorrect,len(nd)),
   'contradicted_change_predictions':sum(r['contradicted_change_prediction'] for r in rr),'change_predictions_with_unknown_reference':sum(r['unknown_reference_change_prediction'] for r in rr),
   'static_geometry':dict(collections.Counter(r['verdict'] for r in gg)),'strict_reveal':dict(collections.Counter(r['verdict'] for r in rv)),
   'mean_output_characters':sum(len(json.dumps(p[group],ensure_ascii=False)) for p in a['predictions'].values())/len(a['predictions'])}
 gains=[x for x in pair if x['additional_correct_non_direct_effect']];losses=[x for x in pair if x['lost_correct_non_direct_effect']]
 runsummary=[]
 for run in a['protocol']['test_runs']:
  item={'run':run,'actions':sum(r['run']==run for r in a['sample'])}
  for group in summary:
   rr=[r for r in vis if r['run']==run and r['group']==group];item[group]={'correct':sum(r['verdict']=='CORRECT' for r in rr),'incorrect':sum(r['verdict']=='INCORRECT' for r in rr),'correct_non_direct_changes':sum(r['verdict']=='CORRECT' and r['observed_non_direct_change'] for r in rr)}
  runsummary.append(item)
 calls=[]
 for path in (R/'calls').glob('*/result.json'):
  c=load(path);failed=[];usage=collections.Counter(c['usage'] or {});started=c['started_at']
  for ff in sorted(path.parent.glob('attempt_*/failed.json')):
   failure=load(ff);manifest=load(ff.parent/'manifest.json');started=min(started,manifest['started_at']);events=[]
   for line in (ff.parent/'events.jsonl').read_text().splitlines():
    try:events.append(json.loads(line))
    except ValueError:pass
   tokens=next((e['usage'] for e in reversed(events) if 'usage'in e),None)
   usage.update(tokens or {});failed.append({'error':failure['error'],'latency_s':failure['latency_s'],'usage':tokens})
  calls.append({'name':path.parent.name,'latency_s':c['latency_s']+sum(x['latency_s'] for x in failed),'successful_attempt_latency_s':c['latency_s'],'failed_attempts':failed,'started_at':started,'finished_at':c['finished_at'],'usage':dict(usage),'usage_complete':all(x['usage'] is not None for x in failed)})
 def cost(cc):
  return {'jobs':len(cc),'attempts':sum(1+len(c['failed_attempts']) for c in cc),'total_request_s':sum(c['latency_s'] for c in cc),'successful_attempt_s':sum(c['successful_attempt_latency_s'] for c in cc),'failed_attempt_s':sum(x['latency_s'] for c in cc for x in c['failed_attempts']),'mean_job_s':sum(c['latency_s'] for c in cc)/len(cc) if cc else None,'median_job_s':percentile([c['latency_s'] for c in cc],.5),'p95_job_s':percentile([c['latency_s'] for c in cc],.95),'tokens':dict(sum((collections.Counter(c['usage'] or {}) for c in cc),collections.Counter())),'token_accounting_complete':all(c['usage_complete'] for c in cc)}
 percasecost=[]
 cb={c['name']:c for c in calls}
 for cid in cases:
  b=cb['predict_'+cid+'_Baseline']['latency_s'];h=sum(cb[k+cid]['latency_s'] for k in ['query_','retrieve_'])+cb['predict_'+cid+'_Hindsight']['latency_s'];percasecost.append({'case_id':cid,'Baseline_prediction_s':b,'Hindsight_prediction_s':cb['predict_'+cid+'_Hindsight']['latency_s'],'Hindsight_retrieval_s':h-cb['predict_'+cid+'_Hindsight']['latency_s'],'Hindsight_total_request_s':h})
 costs={'Baseline_prediction':cost([c for c in calls if c['name'].endswith('_Baseline')]),'Hindsight_prediction':cost([c for c in calls if c['name'].endswith('_Hindsight')]),'Hindsight_retrieval':cost([c for c in calls if c['name'].startswith(('query_','retrieve_'))]),'blind_evaluation':cost([c for c in calls if c['name'].startswith('observe_')]),'per_case':percasecost,'overall':cost(calls),'wall_clock_s':max(c['finished_at'] for c in calls)-min(c['started_at'] for c in calls)}
 costs['Hindsight_per_case_request_sum']={'mean_s':sum(x['Hindsight_total_request_s'] for x in percasecost)/len(percasecost),'median_s':percentile([x['Hindsight_total_request_s'] for x in percasecost],.5),'p95_s':percentile([x['Hindsight_total_request_s'] for x in percasecost],.95),'note':'Sum of query, selection and prediction request times; not measured online deployment latency. Service waiting and possible retries included.'}
 overall={'cases':len(cases),'independent_run_clusters':len(a['protocol']['test_runs']),'new_correct_non_direct_effects':len(gains),'lost_correct_non_direct_effects':len(losses),'net_correct_non_direct_effects':len(gains)-len(losses),'visibility_different_items':sum(x['predictions_differ'] for x in pair),'cases_with_visibility_difference':sum(bool(x['visibility_difference_goals']) for x in differences),'retrieval_NONE':sum(x['selection']['selection']=='NONE' for x in a['retrieval'].values())}
 save(R/'analysis.json',{'summary':summary,'overall':overall,'gains':gains,'losses':losses,'per_region':vis,'paired_visibility':pair,'static_geometry':geo,'strict_reveal':reveal,'substantive_differences':differences,'per_run':runsummary,'costs':costs,'calls':calls,'analysis_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limits':['Blind model observations are proxy labels, not independently measured physical truth.','Four run clusters, eight actions, one stochastic prediction per arm: no statistical or discovery-process generalization claim.','NON_DIRECT means outside directly grasped semantic region, not original Agent intent.','Unknown reference is NA; not counted as error or as unsupported.','Comparison aligns declared semantic region labels; free-text material grounding is not independently validated.','Additional correct effects may reflect model sampling variation; this run does not isolate that variance.','BM25 shortlist may miss relevant hypotheses even though every allowed record is indexed.','All eight tests retained; no outcome-based resampling or prompt edits.']})
 print(json.dumps({'overall':overall,'summary':summary},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
