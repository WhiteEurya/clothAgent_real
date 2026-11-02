from run import *
def main():
 assert (OUT/'test_predictions_frozen.json').exists()
 diagnostics={'at':now(),'posthoc_reporting_only':True,'not_sent_to_teachers':True,'labels_not_changed':True,'splits':{}}
 for split in ['development','test']:
  f=OUT/f'{split}_evaluation.json';ev=read(f);ds={}
  for g,arm in ev['arms'].items():
   rs=arm['cases'];n=len(rs)
   for field in ['trigger','conforms']:
    num=sum(r[field]=='UNKNOWN' for r in rs);arm['metrics'][field+'_unknown']={'numerator':num,'denominator':n,'rate':num/n}
   flagged=[r['id'] for r in rs if r['prediction']=='NO_CLEAR_EVIDENCE' and r['trigger']!='YES']
   ds[g]={'NCE_prediction_without_confirmed_trigger_ids':flagged,'interpretation':'Scope-risk flag only, not automatically invalid. Some rationales may confuse inability of a skill to predict benefit with predicting no clear benefit. Preserve original predictions; no relabeling.','trigger_unknown':arm['metrics']['trigger_unknown'],'conforms_unknown':arm['metrics']['conforms_unknown']}
  rs=ev['arms']['A']['cases'];known=[r for r in rs if r['truth']!='UNKNOWN'];positive=sum(r['truth']=='YES' for r in known)
  ds['truth_counts']={v:sum(r['truth']==v for r in rs) for v in ['YES','NO_CLEAR_EVIDENCE','UNKNOWN']}
  ds['constant_positive_descriptive_reference']={'positive_recall':1 if positive else None,'accuracy_on_known':positive/len(known) if known else None,'not_model_arm':True,'posthoc':True}
  diagnostics['splits'][split]=ds;ev['reporting_derived_at']=now();save(f,ev)
 save(OUT/'evaluation_diagnostics.json',diagnostics)
 effects={x['id']:x['z'] for x in read(DATA/'effects_verified.json')['cases']};checks={x['id']:x['OutcomeEvidence']==effects[x['id']] for split in ['train','development','test'] for x in cases(split)}
 save(OUT/'label_parity.json',{'all_match':all(checks.values()),'per_case':checks,'verified_after_test_predictions_frozen':True,'label_source_note':read(DATA/'effects_verified.json')['not_independent_ground_truth']})
 print(json.dumps({'label_parity':all(checks.values()),'diagnostics':diagnostics},ensure_ascii=False))
if __name__=='__main__':main()
