"""Deterministic preregistered label comparison; no model calls."""
import json,hashlib
from pathlib import Path
from collections import Counter
R=Path('/mnt/newssd/sja/clothAgent_real/reports/hypothesis_guided_prediction_20261102')
def load(p):return json.loads(Path(p).read_text())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
MAP={'EXPOSED_REGION_AREA':'exposed_region_area','VISIBLE_TERMINAL_EDGE_LENGTH':'visible_terminal_edge_length','COVERING_OVERLAP_EXTENT':'covering_overlap_extent'}
def compare(pred,ref,region_clear):
 judgeable=bool(ref and region_clear and ref['evidence_quality']=='CLEAR' and ref['change']!='UNKNOWN')
 definite=pred!='UNKNOWN'
 verdict='UNASSESSABLE' if not judgeable else ('ABSTAIN' if not definite else ('CORRECT' if pred==ref['change'] else 'INCORRECT'))
 return {'predicted_change':pred,'reference':ref,'reference_judgeable':judgeable,'prediction_definite':definite,'verdict':verdict}
def main():
 a=load(R/'result.json');rows=[];pair_rows=[]
 for cid,pair in a['predictions'].items():
  refs={r['region']:r for r in a['observations'][cid]['regions']}
  for group,p in pair.items():
   reg=refs.get(p['primary_target']);clear=reg and reg['region_correspondence']=='CLEAR'
   semantic_comparable=bool(reg and clear and p['semantic_corner']!='UNKNOWN' and reg['semantic_corner']!='UNKNOWN')
   for slot,metric,label in [('visibility','visibility',p['visibility_change']),('geometry',MAP[p['geometry_metric']],p['geometry_change'])]:
    row={'case_id':cid,'group':group,'slot':slot,'target':p['primary_target'],'metric':metric,**compare(label,reg[metric] if reg else None,clear),
     'semantic_comparable':semantic_comparable,'semantic_match':p['semantic_corner']==reg['semantic_corner'] if semantic_comparable else None,
     'reference_region_correspondence':reg['region_correspondence'] if reg else 'INSUFFICIENT'}
    rows.append(row)
  for slot in ['visibility','geometry']:
   b=next(x for x in rows if x['case_id']==cid and x['group']=='Baseline' and x['slot']==slot);h=next(x for x in rows if x['case_id']==cid and x['group']=='Hindsight' and x['slot']==slot)
   pair_rows.append({'case_id':cid,'slot':slot,'Baseline':b['verdict'],'Hindsight':h['verdict'],'same_target_and_metric':b['target']==h['target'] and b['metric']==h['metric'],'both_judgeable':b['reference_judgeable'] and h['reference_judgeable'],'both_definite_and_judgeable':all(x['reference_judgeable'] and x['prediction_definite'] for x in [b,h])})
 summaries={}
 for group in ['Baseline','Hindsight']:
  rr=[r for r in rows if r['group']==group];c=Counter(r['verdict'] for r in rr);n=len(rr);ref=sum(r['reference_judgeable'] for r in rr);graded=c['CORRECT']+c['INCORRECT'];definite=sum(r['prediction_definite'] for r in rr)
  def ratio(x,y):return x/y if y else None
  summaries[group]={'cases':len(a['predictions']),'slots':n,'definite_predictions':definite,'prediction_definite_rate':ratio(definite,n),'reference_judgeable':ref,'reference_judgeable_rate':ratio(ref,n),'graded':graded,'correct':c['CORRECT'],'incorrect':c['INCORRECT'],'abstain_on_judgeable':c['ABSTAIN'],'unassessable':c['UNASSESSABLE'],'unknown_prediction_total':n-definite,'conditional_accuracy':ratio(c['CORRECT'],graded),'correct_over_reference_judgeable':ratio(c['CORRECT'],ref),'correct_concrete_geometry':sum(r['slot']=='geometry' and r['verdict']=='CORRECT' for r in rr),'semantic_label_conflicts':len({r['case_id'] for r in rr if r['semantic_match'] is False}),'mean_response_characters':sum(len(json.dumps(p[group],ensure_ascii=False)) for p in a['predictions'].values())/len(a['predictions']) if a['predictions'] else None}
 calls=[]
 for f in (R/'calls').glob('*/result.json'):
  v=load(f);calls.append({'name':f.parent.name,'latency_s':v['latency_s'],'usage':v['usage']})
 for group in summaries:
  cc=[c for c in calls if c['name'].endswith('_'+group)];summaries[group]['mean_latency_s']=sum(c['latency_s'] for c in cc)/len(cc) if cc else None;summaries[group]['usage']=dict(sum((Counter(c['usage'] or {}) for c in cc),Counter()))
 # No significance claim or forced winner from five actions and one sample per arm.
 matched=[r for r in pair_rows if r['same_target_and_metric'] and r['both_definite_and_judgeable']]
 paired={'same_target_metric_both_graded':len(matched),'Baseline_correct':sum(r['Baseline']=='CORRECT' for r in matched),'Hindsight_correct':sum(r['Hindsight']=='CORRECT' for r in matched),'Hindsight_only_correct':sum(r['Hindsight']=='CORRECT' and r['Baseline']!='CORRECT' for r in matched),'Baseline_only_correct':sum(r['Baseline']=='CORRECT' and r['Hindsight']!='CORRECT' for r in matched)}
 save(R/'analysis.json',{'status':a['status'],'groups':summaries,'paired_common_slots':paired,'per_slot':rows,'per_pair':pair_rows,'calls':calls,'total_model_usage':dict(sum((Counter(c['usage'] or {}) for c in calls),Counter())),'analysis_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limits':['Labels are independent blind model observations, not human-verified physical ground truth.','Unknown reference is UNASSESSABLE; unknown prediction on known reference is ABSTAIN; neither counted correct or incorrect.','Conditional accuracy must be read with abstention and reference coverage.','Different chosen targets/metrics can change denominators; common-target/metric pairs reported separately.','Static corner-region visibility is not verified semantic Corner Reveal.','At most five actions, one stochastic prediction per condition, no estimate separating sampling noise from stable prompt effects.','Free-text causal explanations are not scored as success or information gain.','No new actions, no skill, no changes to the Hypothesis Bank.']})
 print(json.dumps({'groups':summaries,'paired':paired},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
