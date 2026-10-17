import json,re
from pathlib import Path
from cloth_agent.experiments.visual_program_inheritance_qwen import classify,metrics,GRAMMARS,PIDS

def program():return json.loads(Path('reports/visual_program_S1_S2/visual_program.json').read_text())
def values(v):return json.dumps({p:dict(value=x,confidence=.9) for p,x in zip(PIDS,v)})
def test_valid_s2_derived_by_host():
 text=values(['YES','NO','YES','YES']);r=classify(text,'inheritance',program())
 assert re.fullmatch(GRAMMARS['inheritance'],text)
 assert r['prediction']=='S2' and r['status']=='OK'
def test_failure_is_not_unknown():
 r=classify('{bad','inheritance',program())
 assert r['prediction']=='PARSE_ERROR' and r['status']=='PARSE_ERROR'
def test_no_new_predicate_allowed():
 s=json.loads(values(['YES','YES','NO','YES']));s['new']=dict(value='NO',confidence=.8)
 assert classify(json.dumps(s),'inheritance',program())['status']=='PARSE_ERROR'
def test_confidence_grammar_bounds():
 s=values(['YES','YES','NO','YES']).replace('0.9','1.1')
 assert re.fullmatch(GRAMMARS['inheritance'],s) is None
 assert classify(s,'inheritance',program())['status']=='PARSE_ERROR'
def test_real_unknown_is_not_error():
 assert classify(values(['YES','UNKNOWN','NO','YES']),'inheritance',program())['prediction']=='UNKNOWN'
 assert classify('UNKNOWN','zero_shot',program())['status']=='OK'
def test_metrics_separate_failure_from_rejection():
 rows=[]
 for sid,label,status,pred in [('obs_7f66d64e53de5d54','S2','PARSE_ERROR','PARSE_ERROR'),('obs_cb681fd8ddcdf275','OTHER','INFERENCE_ERROR','INFERENCE_ERROR')]:
  rows.append(dict(sample_id=sid,label=label,status=status,prediction=pred,elapsed_s=1,peak_allocated_mib=1,peak_reserved_mib=1))
 s=metrics(rows)
 assert s['unknown']==0 and s['other_false_accept']==0 and s['parse_errors']==1 and s['inference_errors']==1 and s['known_correct']==0
