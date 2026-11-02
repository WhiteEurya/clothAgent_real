"""Validate isolation, chronology, completeness and unchanged source hashes."""
import hashlib,json,re
from pathlib import Path
ROOT=Path('/mnt/newssd/sja/clothAgent_real');R=ROOT/'reports/autonomous_retrieval_ab_20261102'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=load(R/'result.json');p=a['protocol'];f=load(R/'predictions_frozen.json');d=load(R/'after_disclosure.json');audit=load(R/'isolation_audit.json');idx=load(R/'training_index.json');allowed={h['hypothesis_id'] for h in idx};held=set(audit['heldout_ids'])
 assert len(a['sample'])==8 and len({r['run'] for r in a['sample']})>=4
 assert sha(ROOT/'experiments/autonomous_retrieval_ab_v1/run.py')==p['runner_sha256']
 assert sha(ROOT/'reports/intention_outcome_hypothesis_bank_v1/hypothesis_bank.json')==p['source_bank_sha256']
 assert sha(ROOT/'reports/intention_outcome_hypothesis_bank_v1/experience_index.json')==p['source_index_sha256']
 assert sha(R/'training_index.json')==p['training_index_sha256']
 assert not any(set(re.findall(r'X\d{4}',json.dumps(h,ensure_ascii=False)))&held for h in idx)
 assert sha(R/'predictions.json')==f['predictions_sha256'];assert f['frozen_at']<d['disclosed_at']
 scoring=load(R/'scoring_code_frozen.json');assert scoring['frozen_at']<d['disclosed_at'];assert sha(ROOT/'experiments/autonomous_retrieval_ab_v1/analyze.py')==scoring['analysis_code_sha256']
 for k,v in p['prompt_hashes'].items():assert sha(R/f'{k}_prompt.txt')==v
 for row in a['sample']:
  cid=row['id'];assert sha(Path(row['before']))==row['before_sha256'];assert sha(Path(row['after']))==row['after_expected_sha256'];assert sha(Path(row['payload']))==row['payload_sha256']
  ret=a['retrieval'][cid];ids={h['hypothesis_id'] for h in ret['selected_hypotheses']};assert ids<=allowed and len(ids)<=3
  assert sha(R/'retrieval'/f'{cid}.json')==f['retrieval_hashes'][cid]
  data={};man={}
  for group in ['Baseline','Hindsight']:
   rr=load(R/'calls'/f'predict_{cid}_{group}'/'result.json');ad=Path(rr['attempt_dir']);assert rr['finished_at']<=f['frozen_at'];assert sha(ad/'output.json')==f['call_outputs'][cid+'_'+group]
   raw=(ad/'input.txt').read_text();prefix=(R/'prediction_prompt.txt').read_text()+'\n';assert raw.startswith(prefix);data[group]=json.loads(raw[len(prefix):]);man[group]=load(ad/'manifest.json')
  assert data['Hindsight'].pop('retrieved_hypotheses')==ret['selected_hypotheses'];assert data['Baseline']==data['Hindsight'];assert man['Baseline']['images']==man['Hindsight']['images']
  ob=load(R/'calls'/f'observe_{cid}'/'result.json');assert ob['started_at']>d['disclosed_at'];ad=Path(ob['attempt_dir']);raw=(ad/'input.txt').read_text();prefix=(R/'observation_prompt.txt').read_text()+'\n';assert raw.startswith(prefix);payload=json.loads(raw[len(prefix):]);assert set(payload)=={'recorded_action'}
  assert payload['recorded_action']==data['Baseline']['recorded_action']
  assert len(load(ad/'manifest.json')['images'])==2
 for file in (R/'calls').glob('*/result.json'):
  rr=load(file);ad=Path(rr['attempt_dir']);man=load(ad/'manifest.json');assert '-p' not in man['command'];assert man['input_sha256']==sha(ad/'input.txt')
  events=[json.loads(x) for x in (ad/'events.jsonl').read_text().splitlines()];assert not any(e.get('item',{}).get('type') in ['command_execution','mcp_tool_call','web_search','file_change'] for e in events)
 result={'status':'PASS','cases':8,'run_clusters':4,'model_jobs':len(list((R/'calls').glob('*/result.json'))),'all_eight_retained':True,'full_run_exclusion_and_reference_scan':True,'source_bank_and_records_unchanged':True,'paired_inputs_identical_except_retrieved_hypotheses':True,'all_predictions_frozen_before_after_disclosure':True,'evaluator_inputs_exclude_predictions_hypotheses_and_group':True,'no_model_tool_calls':True,'no_profile':True,'runner_and_prompts_unchanged':True}
 (R/'delivery_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
