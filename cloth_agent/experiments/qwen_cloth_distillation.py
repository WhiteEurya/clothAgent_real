"""Offline predicate distillation; immutable test exclusion and grouped splits."""
import argparse,csv,hashlib,json,random,re,time,shutil,html
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from PIL import Image
from .visual_decision_program import save,sha,obj,call
from .qwen_cloth_teacher import TeacherMultimodal as CodexMultimodal
from .qwen_predicate_diagnosis import PREFIX
from .visual_program_inheritance import SIDE,PIDS,load
OUT=Path('reports/qwen_cloth_predicate_distillation')
PROGRAM=Path('reports/visual_program_S1_S2/visual_program.json')
TEST=Path('reports/qwen_predicate_diagnosis/manifest.json')
SEED=20261014

def jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def readl(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def pixels(p):
    im=Image.open(p).convert('RGB');return hashlib.sha256(im.tobytes()).hexdigest()
def thumb(p):return np.asarray(Image.open(p).convert('RGB').resize((64,64)),dtype=np.float32)

def prepare():
    if (OUT/'manifest.json').exists():raise RuntimeError('Already frozen')
    tests=load(TEST)['queries'];testhash={r['pixel_sha256'] for r in tests};testruns={r['run_id'] for r in tests}
    for r in tests:assert sha(r['image'])==r['file_sha256'] and pixels(r['image'])==r['pixel_sha256']
    testthumb=[thumb(r['image']) for r in tests]
    candidates=load(OUT/'candidate_inventory.json');groups=defaultdict(list);excluded=[]
    for r in candidates:
        d=re.search(r'2026\d{4}',r['run']);date=d.group() if d else '';t0=thumb(r['image'])
        reason=None
        if not date or date>='20261012':reason='date_quarantine_all_Oct12_and_later'
        elif r['run'] in testruns or r['pixel_sha256'] in testhash:reason='test_run_or_exact_copy'
        elif min(float(np.abs(t0-t).mean()) for t in testthumb)<2:reason='near_test_image_MAE_lt_2_of_255'
        if reason:excluded.append(dict(**r,reason=reason));continue
        r=dict(**r,date=date);groups[r['run']].append(r)
    rng=random.Random(SEED);selected=[]
    for run,rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:r['image']);rng.shuffle(rows);selected+=rows[:100]
    rng.shuffle(selected);selected=selected[:240]
    # Entire acquisition day held out; no test outputs used to choose this date.
    for r in selected:
        r['split']='val' if r['date']=='20260923' else 'train';r['sample_id']='trainobs_'+r['pixel_sha256'][:16];r['file_sha256']=sha(r['image'])
    # Drop train near-duplicates of val, including possible restarted runs.
    vt=[thumb(r['image']) for r in selected if r['split']=='val'];kept=[]
    for r in selected:
        t0=thumb(r['image'])
        if r['split']=='train' and vt and min(float(np.abs(t0-t).mean()) for t in vt)<2:
            excluded.append(dict(**r,reason='near_val_MAE_lt_2_of_255'));continue
        kept.append(r)
    assert not {r['run'] for r in kept if r['split']=='train'} & {r['run'] for r in kept if r['split']=='val'}
    assert len({r['pixel_sha256'] for r in kept})==len(kept)
    program=load(PROGRAM);prompts={p['id']:PREFIX+p['question'] for p in program['predicates']}
    previous=load(TEST)['prompts']['A_atomic_original']
    for p,v in previous.items():assert prompts[p]==v
    (OUT/'visual_program.json').write_bytes(PROGRAM.read_bytes())
    save(OUT/'manifest.json',dict(seed=SEED,program_sha256=sha(PROGRAM),test_manifest_sha256=sha(TEST),excluded_test_runs=sorted(testruns),images=kept,atomic_prompts=prompts,selection='pre-20261012 only; <=100 images/run; random <=240; whole 20260923 validation; exact and MAE<2/255 near-duplicate exclusions',confidence_cutoff=.85,epochs=2,audit_random_images=40,teacher_model='gpt-6-astra',freeze_unix_s=time.time()))
    save(OUT/'exclusions.json',excluded)
    print('Frozen',Counter(r['split'] for r in kept),'runs',len({r['run'] for r in kept}),flush=True)

def teacher():
    m=load(OUT/'manifest.json');program=load(OUT/'visual_program.json');assert sha(PROGRAM)==m['program_sha256']
    schema=obj({p:obj(dict(value=dict(type='string',enum=['YES','NO','UNKNOWN']),confidence=dict(type='number',minimum=0,maximum=1))) for p in PIDS})
    prompt=SIDE+'\nLabel only the four frozen visual predicates on this single current RGB. Assess each independently from visible evidence. If evidence is occluded or ambiguous, use UNKNOWN. Do not infer from action history. Return only the four predicate keys, each with value YES/NO/UNKNOWN and confidence [0,1]. Do not output node labels, actions, explanations, or reasoning.\nFrozen region definitions and predicate questions:\n'+json.dumps({k:program[k] for k in ['regions','predicates']})
    save(OUT/'teacher_protocol.json',dict(prompt=prompt,schema=schema,model='gpt-6-astra',images_per_request=1,tool_isolation='bubblewrap',test_images_attached=False))
    def one(r):
        assert sha(r['image'])==r['file_sha256']
        for attempt in range(4):
            try:
                d,meta=call(CodexMultimodal('gpt-6-astra',900),prompt,[r['image']],schema,OUT/'teacher_requests'/r['sample_id']);break
            except Exception as e:
                if attempt==3:raise
                save(OUT/'teacher_retry_log'/f"{r['sample_id']}_{attempt}.json",dict(error=str(e),backoff_s=30*(attempt+1)))
                time.sleep(30*(attempt+1))
        return dict(**r,predicates=d,teacher_model=meta['model'],elapsed_s=meta['elapsed_s'])
    rows=[];errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        audit_ids={r['sample_id'] for r in random.Random(SEED+1).sample(sorted(m['images'],key=lambda r:r['sample_id']),40)}
        ordered=sorted(m['images'],key=lambda r:(r['sample_id'] not in audit_ids,r['sample_id']))
        jobs={pool.submit(one,r):r for r in ordered}
        for f in as_completed(jobs):
            r=jobs[f]
            try:rows.append(f.result())
            except Exception as e:errors.append(dict(sample_id=r['sample_id'],error=str(e)))
            jsonl(OUT/'teacher_labels.jsonl',sorted(rows,key=lambda r:r['sample_id']));save(OUT/'teacher_errors.json',errors)
            print(f'TEACHER {len(rows)}/{len(m["images"])} errors={len(errors)}',flush=True)
    if errors:raise RuntimeError('Teacher failures saved; resume uses verified cache')

def audit():
    chosen=random.Random(SEED+1).sample(sorted(load(OUT/'manifest.json')['images'],key=lambda r:r['sample_id']),40)
    p=OUT/'human_audit';p.mkdir(exist_ok=True);data=[]
    for i,r in enumerate(chosen):
        name=f'{i+1:02d}.png';shutil.copyfile(r['image'],p/name);data.append(dict(audit_id=i+1,sample_id=r['sample_id'],image=name))
    save(p/'manifest.json',data)
    with (p/'human_labels.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['audit_id','sample_id',*PIDS,'reviewer']);w.writeheader();w.writerows({k:r[k] for k in ['audit_id','sample_id']} for r in data)
    questions=load(p/'questions_zh.json') if (p/'questions_zh.json').exists() else {q['id']:q['question'] for q in load(PROGRAM)['predicates']}
    page='''<!doctype html><meta charset="utf-8"><title>Cloth predicate human audit</title><style>body{font:17px sans-serif;margin:24px;max-width:1200px}img{width:100%;max-height:650px;object-fit:contain}section{border-bottom:2px solid #aaa;padding:20px}label{display:block;margin:12px}select,input,button{font-size:18px}header{position:sticky;top:0;background:white;padding:12px}.answers{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 22px}.answers button{cursor:pointer;padding:12px 24px;border:2px solid #aaa;background:#fff;border-radius:8px;min-width:100px}.answers button[aria-pressed=true]{background:#d9ecff;border-color:#0066cc;color:#004a99;font-weight:bold}.answers button:focus-visible{outline:3px solid #111;outline-offset:3px}</style><header><b>40 张盲审（隐藏 teacher 答案）</b><p>先在脑中旋转衣物为领口朝上、下摆朝下；不镜像。左右按此方向的观察者定义。不清楚请选 UNKNOWN。</p>审核人 <input id="who"><button onclick="download()">导出已填写答案（可只填一部分）</button><span id="progress"></span></header><main></main><script>'''
    page+='const rows='+json.dumps(data)+';const questions='+json.dumps(questions)+';'
    page+='''const key='cloth-distillation-audit-20261014';const stored=JSON.parse(localStorage.getItem(key)||'{}');const main=document.querySelector('main');for(const r of rows){const s=document.createElement('section');s.innerHTML=`<h2>${r.audit_id}/40</h2><img src="${r.image}">`;for(const [p,q] of Object.entries(questions)){const l=document.createElement('label');l.textContent=q+' ';const group=document.createElement('div');group.className='answers';group.setAttribute('role','group');group.setAttribute('aria-label',q);for(const v of ['YES','NO','UNKNOWN']){const btn=document.createElement('button');btn.type='button';btn.textContent=({'YES':'是','NO':'否','UNKNOWN':'无法确定／看不清'})[v];btn.dataset.value=v;btn.setAttribute('aria-pressed',String(stored[r.audit_id+'_'+p]===v));btn.onclick=()=>{stored[r.audit_id+'_'+p]=v;localStorage.setItem(key,JSON.stringify(stored));for(const sibling of group.children)sibling.setAttribute('aria-pressed',String(sibling.dataset.value===v));progress()};group.append(btn)}l.append(group);s.append(l)}main.append(s)}function progress(){document.querySelector('#progress').textContent=' 已填 '+Object.values(stored).filter(Boolean).length+'/160'}progress();function download(){if(!document.querySelector('#who').value.trim()){alert('请填写审核人');return}const quote=x=>'"'+String(x).replaceAll('"','""')+'"';const lines=[['audit_id','sample_id',...Object.keys(questions),'reviewer'],...rows.map(r=>[r.audit_id,r.sample_id,...Object.keys(questions).map(p=>stored[r.audit_id+'_'+p]||''),document.querySelector('#who').value])];const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.map(r=>r.map(quote).join(',')).join('\\n')],{type:'text/csv'}));a.download='human_labels.csv';a.click()}
</script>'''
    (p/'index.html').write_text(page)
    (OUT/'audit.md').write_text('# Human teacher audit\n\nPENDING HUMAN LABELS. 40 images randomly selected without replacement (seed 20261015), four frozen predicates each. Teacher labels hidden in human_audit/index.html. No AI self-review is counted as human truth.\n\nAfter receipt, report per-predicate exact agreement and YES precision = teacher YES & human YES / teacher YES; also NO precision and denominators. Human UNKNOWN is not counted as verified correct YES/NO. Do not change frozen test/prompt/threshold.\n')
    print('Audit ready',p/'index.html',flush=True)

def build_samples():
    m=load(OUT/'manifest.json');rows=readl(OUT/'teacher_labels.jsonl');atomic=[]
    assert len(rows)==len(m['images'])
    for r in rows:
        for p,d in r['predicates'].items():
            if d['value']=='UNKNOWN' or d['confidence']<m['confidence_cutoff']:continue
            atomic.append(dict(sample_id=r['sample_id'],image=r['image'],file_sha256=r['file_sha256'],run=r['run'],split=r['split'],predicate=p,value=d['value'],confidence=d['confidence'],prompt=m['atomic_prompts'][p]))
    # Balance via inverse-frequency sampler, not duplicated images; never balance using test.
    counts=Counter((r['predicate'],r['value']) for r in atomic if r['split']=='train')
    for r in atomic:r['sampling_weight']=1/counts[(r['predicate'],r['value'])] if r['split']=='train' else 1
    jsonl(OUT/'atomic_samples.jsonl',atomic)
    distributions={s:{p:dict(Counter(r['predicates'][p]['value'] for r in rows if r['split']==s)) for p in PIDS} for s in ['train','val']}
    save(OUT/'dataset_stats.json',dict(images=len(rows),splits=dict(Counter(r['split'] for r in rows)),runs={s:len({r['run'] for r in rows if r['split']==s}) for s in ['train','val']},teacher_distributions=distributions,retained_atomic={s:{p:dict(Counter(r['value'] for r in atomic if r['split']==s and r['predicate']==p)) for p in PIDS} for s in ['train','val']},retained_images={s:len({r['sample_id'] for r in atomic if r['split']==s}) for s in ['train','val']},atomic_samples=dict(Counter(r['split'] for r in atomic))))
    (OUT/'dataset_summary.md').write_text('# Dataset summary\n\nOnly pre-20261012 observations; all Oct12+ runs quarantined. Exact pixel dedup, held-out 29 hashes checked, near-copy exclusions. Validation uses entire 20260923 acquisition day. No test labels used. Unique images are not claimed to be independent physical states. Adjacent frames may remain within one split.\n\nTeacher confidence >=0.85 and YES/NO only retained. Inverse-frequency sampling balances predicate x answer cells (if present), without asserting that duplicated draws add diversity. Validation remains naturally distributed.\n\n```json\n'+(OUT/'dataset_stats.json').read_text()+'\n```\n')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','teacher','audit','samples']);args=a.parse_args();dict(prepare=prepare,teacher=teacher,audit=audit,samples=build_samples)[args.stage]()
