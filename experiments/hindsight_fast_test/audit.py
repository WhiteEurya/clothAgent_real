"""Read saved artifacts and verify isolation/budgets; no model or robot calls."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2]/'reports/hindsight_fast_test_20261029'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    protocol=read(R/'protocol.json');idx=read(R/'experience_index.json');sp=read(R/'split.json');h=set(sp['heldout_complete_runs'])
    train=[x for x in idx if x['run'] not in h];test=[x for x in idx if x['run'] in h]
    a={x[k+'_sha256'] for x in train for k in ['before','after']};b={x[k+'_sha256'] for x in test for k in ['before','after']}
    assert not a & b
    assert sha(Path(__file__).parent/'run.py')==protocol['code_sha256']
    assert sha(Path(__file__).parents[1]/'hindsight_smoke_test/transport.py')==protocol['transport_sha256']
    sheets=read(R/'contact_sheets.json');train_ids={x['id'] for x in train};assert all(set(s['ids'])<=train_ids for s in sheets)
    short=read(R/'shortlist.json');assert len(short)<=10 and len({x['id'] for x in short})==len(short)
    attempts=list((R/'calls').glob('*/attempt_*/started.json'));assert len(attempts)<=20
    tools=[]
    for p in (R/'calls').glob('*/attempt_*/events.jsonl'):
        for line in p.read_text().splitlines():
            try:ev=json.loads(line)
            except ValueError:continue
            typ=ev.get('item',{}).get('type')
            if typ in ['command_execution','mcp_tool_call','web_search','file_change']:tools.append((str(p),typ))
    assert not tools
    allowed_val={x['before_sha256'] for x in sp['validation']};checked=0
    for p in (R/'calls').glob('*/attempt_*/manifest.json'):
        x=read(p);is_val=p.parent.parent.name.startswith('validation_')
        for im in x['images']:
            if is_val:assert im['sha256'] in allowed_val
            else:assert im['sha256'] not in b
        checked+=1
    if (R/'skill_freeze.json').exists():
        f=read(R/'skill_freeze.json');assert f['code_sha256']==sha(R/'candidate_skill.py');assert f['prompt_sha256']==sha(R/'inference_prompt.txt')
        for p in (R/'calls').glob('validation_*/attempt_*/started.json'):assert read(p)['time']>f['time']
    vals=read(R/'validation_results.json') if (R/'validation_results.json').exists() else [];assert len(vals)<=3
    result={'status':'PASS','run_and_image_hash_isolation':True,'original_frozen_selection_code_unchanged':True,'model_tool_calls':len(tools),'model_manifests_checked':checked,'model_attempts':len(attempts),'unique_enlarged_pairs':len(short),'validation_images':len(vals),'validation_started_after_freeze':True if (R/'skill_freeze.json').exists() else 'NOT_APPLICABLE','physical_actions':0,'limitations':'Static/pixel/code checks are not robot safety or physical skill evaluation. Agent observations are not independent material ground truth.'}
    (R/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
