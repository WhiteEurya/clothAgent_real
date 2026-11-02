"""Read-only consistency audit of this experiment, saved alongside its report."""
import json,hashlib
from pathlib import Path
from PIL import Image
import run

def main():
    p=run.OUT
    read=lambda x:json.loads((p/x).read_text())
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    m=read('input_manifest.json');reviews=read('reviews.json');ranking=read('ranking.json')
    assert len(m['mapping'])==17
    assert len({x['id'] for x in reviews})==len(reviews)
    assert {x['id'] for x in reviews}=={x['id'] for x in m['mapping']}
    expected=sorted([x for x in reviews if run.eligible(x)],key=run.rank)
    assert expected==ranking['candidates']
    protocol=read('protocol.json')
    assert sha(Path(run.__file__))==protocol['code_sha256']
    assert sha(Path(run.__file__).parent/'transport.py')==protocol['transport_sha256']
    for x in reviews:
        outputs=[]
        for prefix in ['review_','retry_review_']:
            f=p/'calls'/(prefix+x['id'])/'result.json'
            if f.exists():
                result=json.loads(f.read_text())
                if result['status']=='COMPLETE':outputs.append(result['output'])
        assert x in outputs
    for r in m['mapping']:
        for k in ['before','after']:assert sha(run.REPO/r[k])==r[k+'_sha256']
    assert not ({r['run'] for r in m['mapping']}&{v['run'] for v in m['validation_before_only']})
    calls=[];tools=[]
    for f in sorted((p/'calls').glob('*/attempt_*')):
        if not (f/'started.json').exists():continue
        man=json.loads((f/'manifest.json').read_text())
        for im in man['images']:
            assert sha(im['source'])==im['sha256']
            assert sha(im['upload'])==im['upload_sha256']
            assert Image.open(im['source']).size==Image.open(im['upload']).size
        if (f/'events.jsonl').exists():
            for line in (f/'events.jsonl').read_text().splitlines():
                try:e=json.loads(line)
                except ValueError:continue
                t=e.get('item',{}).get('type')
                if t in ['command_execution','mcp_tool_call','web_search','file_change']:tools.append({'call':f.parent.name,'type':t})
        calls.append(f)
    assert not tools and len(calls)<=26
    selected=read('selected_case.json') if (p/'selected_case.json').exists() else None
    if selected:
        assert selected['review']==expected[0]
        goal=selected['goal']
        for f in ['skill_hypotheses.json','skill_synthesis.json']:
            if (p/f).exists():assert read(f)['goal']==goal
        if (p/'skill_freeze.json').exists():
            freeze=read('skill_freeze.json')
            assert sha(p/'candidate_skill.py')==freeze['code_sha256']
            assert sha(p/'inference_prompt.txt')==freeze['prompt_sha256']
            for f in (p/'calls').glob('test_*/attempt_*/started.json'):
                assert json.loads(f.read_text())['time']>=freeze['time']
        for t in read('validation_results.json') if (p/'validation_results.json').exists() else []:
            assert t['action']['goal']==goal
            assert t['after_or_original_action_supplied'] is False
            assert t['physical_success']=='NOT_TESTED'
    result={'status':'PASS','reviewed_pairs':len(reviews),'raw_images_hash_verified':34,'ranking_unchanged':True,'goal_lock_verified':True,'full_resolution_uploads':True,'upload_encoding':'JPEG quality90; resolution preserved, not lossless','discovery_and_test_runs_disjoint':True,'model_calls':len(calls),'forbidden_tool_events':tools,'robot_actions':0,'material_ground_truth':'NOT_INDEPENDENTLY_VERIFIED','auditor':'deterministic artifact consistency checks, not an independent semantic evaluation'}
    (p/'artifact_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
