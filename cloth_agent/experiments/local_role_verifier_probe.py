"""Offline fixed-candidate L/G verifier. No localization or robot access.

prepare --source reports/local_experience_transfer_probe_20261020 --output NEW_DIR
run --experiment NEW_DIR
report --experiment NEW_DIR
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

import jsonschema
from PIL import Image, ImageDraw, ImageFont, ImageOps

MODEL='gpt-6-astra'
EFFORT='high'
SIDE=320
DISPLAY_ARROW_PX=110
IDS=('q01','q03','q04')
SCHEMA={'type':'object','additionalProperties':False,
        'properties':{k:{'type':'string','enum':['YES','NO','UNCERTAIN']}
                      for k in ('same_manipulation_role','same_effect_likely')},
        'required':['same_manipulation_role','same_effect_likely']}
PROMPT='''Offline pairwise verification of ONE FROZEN manipulation candidate.
The marked demo Pa and marked query candidate are fixed. Do not search for a
different point, relocate either point, change an action, or propose an action.
Do not match the whole garment state. Do not classify or name garment parts;
the task is structural-role judgment, not semantic naming or material-point identity.
Red marks identify the fixed points. Yellow arrows show the frozen movement
directions at a uniform display length; arrow length does not encode distance.
All local images are 320x320 native-pixel crops at equal scale, without rotation
or resizing. The demo AFTER crop uses exactly the same image-coordinate window
as the demo BEFORE crop, not a tracked point. Gray padding denotes pixels outside
the captured image. If full views are provided, they show these same marked
points and directions in the surrounding garment, not alternative candidates.

Judge ONLY the marked query candidate relative to the marked demonstration:
same_manipulation_role: Does this candidate, within its garment context, have
the same operable structural role as demo Pa? Consider how the locally grasped
fabric relates to adjacent and connected fabric, boundaries, folds and coverage.
Local appearance resemblance alone is insufficient, but exact whole-object
shape and semantic part identity are not required.
same_effect_likely: Is there sufficient visible evidence that applying the frozen
transferred movement at this candidate would produce a local change similar to
the demonstrated BEFORE-to-AFTER change? A generic useful pull is not by itself
evidence of the same effect. Do not require identical final whole-garment shape.
Judge each field separately. Use UNCERTAIN when the supplied views do not support
a decision; do not invent hidden structure. Return ONLY the two-field schema JSON.
No explanation, garment-part labels, confidence, coordinates or new movement.
'''


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def font(size=20): return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def snapshot(root): return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def marked(image,pa,direction):
    im=image.convert('RGB').copy();d=ImageDraw.Draw(im)
    x,y=pa;dx,dy=direction;end=(x+DISPLAY_ARROW_PX*dx,y+DISPLAY_ARROW_PX*dy)
    d.line([(x,y),end],fill='#ffe747',width=4)
    d.polygon([end,(end[0]-14*dx+6*dy,end[1]-14*dy-6*dx),
               (end[0]-14*dx-6*dy,end[1]-14*dy+6*dx)],fill='#ffe747')
    d.ellipse((x-6,y-6,x+6,y+6),fill='#ff364f',outline='white',width=2)
    d.text((x+9,y+8),'Pa',font=font(18),fill='white',stroke_width=2,stroke_fill='black')
    return im


def crop_box(pa):
    x,y=pa;left=int(round(x-SIDE/2));top=int(round(y-SIDE/2))
    return [left,top,left+SIDE,top+SIDE]


def crop_native(im,box):
    canvas=Image.new('RGB',(SIDE,SIDE),(75,75,75))
    x0,y0,x1,y1=box
    visible=(max(0,x0),max(0,y0),min(im.width,x1),min(im.height,y1))
    if visible[2]<=visible[0] or visible[3]<=visible[1]:raise ValueError('crop misses image')
    canvas.paste(im.crop(visible),(visible[0]-x0,visible[1]-y0));return canvas


def request_images(root,qid,condition):
    base=Path(root)/'inputs'/qid
    local=[('DEMO local BEFORE with historical action',base/'demo_local.png'),
           ('QUERY local fixed candidate with transferred action',base/'query_local.png'),
           ('DEMO local AFTER (same coordinate window)',base/'after_local.png')]
    if condition=='L':return local
    if condition!='G':raise ValueError('unknown condition')
    return local+[
        ('DEMO full BEFORE with historical action',base/'demo_full.png'),
        ('QUERY full image with fixed candidate and action',base/'query_full.png'),
        ('DEMO full AFTER',base/'after_full.png')]


def compose(images,destination):
    board=Image.new('RGB',(1440,400 if len(images)==3 else 740),'#121c2d');d=ImageDraw.Draw(board)
    for i,(role,path) in enumerate(images):
        column=i%3;row=i//3;x=column*480;y=0 if row==0 else 400
        with Image.open(path) as im:
            shown=im.convert('RGB') if row==0 else ImageOps.contain(im.convert('RGB'),(470,265))
        board.paste(shown,(x+(480-shown.width)//2,y+65))
        d.text((x+12,y+8),f'Image {i+1}',font=font(),fill='white')
        short=['DEMO local + action','QUERY local + frozen action','DEMO local AFTER',
               'DEMO full + action','QUERY full + frozen action','DEMO full AFTER'][i]
        d.text((x+12,y+35),short,font=font(18),fill='#b7cee8')
    board.save(destination)


def prepare(source,root):
    source=Path(source).resolve();root=Path(root).resolve()
    if root==source or source in root.parents:raise ValueError('new experiment must be outside previous results')
    before_snapshot=snapshot(source)
    root.mkdir(parents=True,exist_ok=False)
    m=read(source/'dataset/manifest.json');demo_pa=m['demo_pa'];demo_target=m['demo_target']
    import math
    v=[b-a for a,b in zip(demo_pa,demo_target)];length=math.hypot(*v);demo_dir=[x/length for x in v]
    box=crop_box(demo_pa);frozen={};files={}
    for qid in IDS:
        result=read(source/f'run/{qid}_A/result.json')
        if result['status']!='OK' or result['answer']['applicable']!='YES':raise ValueError('missing frozen YES candidate')
        answer=result['answer'];base=root/'inputs'/qid;base.mkdir(parents=True)
        frozen[qid]=dict(source_condition='A',source_result=str(source/f'run/{qid}_A/result.json'),
            source_sha256=sha(source/f'run/{qid}_A/result.json'),answer=answer,
            query_crop_box=crop_box(answer['grasp_px']))
        with Image.open(source/'dataset/public/demo_before.png') as im:
            full=marked(im,demo_pa,demo_dir);full.save(base/'demo_full.png');crop_native(full,box).save(base/'demo_local.png')
        with Image.open(source/f'dataset/public/{qid}.png') as im:
            full=marked(im,answer['grasp_px'],answer['move_direction_px'])
            full.save(base/'query_full.png');crop_native(full,frozen[qid]['query_crop_box']).save(base/'query_local.png')
        with Image.open(source/'dataset/public/demo_after.png') as im:
            im=im.convert('RGB');im.save(base/'after_full.png');crop_native(im,box).save(base/'after_local.png')
        for p in base.glob('*.png'):files[str(p.relative_to(root))]=sha(p)
    truth={'source':'User-fixed evaluation labels for this probe, before invocation',
           'q01':'ACCEPT','q03':'ACCEPT','q04':'REJECT',
           'q04_private_reason':'用户人工确认：condition A 的 Pa 在衣身角落，与 demo 右袖操作不同。此说明不得发送模型。'}
    save(root/'private/evaluation_labels.json',truth)
    save(root/'private/source_snapshot.json',before_snapshot)
    schedule=[['q01','L'],['q01','G'],['q03','G'],['q03','L'],['q04','L'],['q04','G']]
    save(root/'manifest.json',dict(protocol='local-role-verifier-v1',model=MODEL,effort=EFFORT,
        source=str(source),frozen_candidates=frozen,demo_pa=demo_pa,demo_target=demo_target,
        demo_direction=demo_dir,demo_crop_box=box,crop_side_px=SIDE,display_arrow_length_px=DISPLAY_ARROW_PX,
        files=files,schedule=schedule,max_calls=6,prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),schema=SCHEMA,
        labels_sha256=sha(root/'private/evaluation_labels.json'),source_snapshot_sha256=sha(root/'private/source_snapshot.json'),
        acceptance_rule='ACCEPT iff both YES; REJECT if either NO; otherwise UNCERTAIN',created_unix=time.time()))
    for qid,condition in schedule:
        folder=root/'requests'/f'{qid}_{condition}';folder.mkdir(parents=True)
        images=request_images(root,qid,condition)
        prompt=PROMPT+'\nAttached images, in order:\n'+'\n'.join(f'Image {i+1}: {role}' for i,(role,_) in enumerate(images))
        (folder/'prompt.txt').write_text(prompt)
        save(folder/'request.json',dict(model=MODEL,reasoning_effort=EFFORT,schema=SCHEMA,
             images=[dict(role=role,path=str(p.relative_to(root)),sha256=sha(p)) for role,p in images],prompt_sha256=sha(folder/'prompt.txt')))
        compose(images,folder/'input_composite.png')
    assert snapshot(source)==before_snapshot
    return root/'manifest.json'


def verify(root):
    m=read(root/'manifest.json')
    if m['model']!=MODEL or m['effort']!=EFFORT or m['schema']!=SCHEMA or m['prompt_sha256']!=hashlib.sha256(PROMPT.encode()).hexdigest():raise ValueError('frozen protocol changed')
    if m['max_calls']!=6 or len(m['schedule'])!=6 or set(map(tuple,m['schedule']))!={(q,c) for q in IDS for c in ('L','G')}:raise ValueError('invalid six-call schedule')
    for name,expected in m['files'].items():
        if sha(root/name)!=expected:raise ValueError('frozen image changed')
    for name,key in [('evaluation_labels.json','labels_sha256'),('source_snapshot.json','source_snapshot_sha256')]:
        if sha(root/'private'/name)!=m[key]:raise ValueError('frozen labels or source snapshot changed')
    for qid,condition in m['schedule']:
        folder=root/'requests'/f'{qid}_{condition}';req=read(folder/'request.json')
        expected=PROMPT+'\nAttached images, in order:\n'+'\n'.join(f'Image {i+1}: {role}' for i,(role,_) in enumerate(request_images(root,qid,condition)))
        if (folder/'prompt.txt').read_text()!=expected:raise ValueError('request prompt changed')
        expected_images=[dict(role=role,path=str(p.relative_to(root)),sha256=sha(p)) for role,p in request_images(root,qid,condition)]
        if req['images']!=expected_images or req['schema']!=SCHEMA or req['model']!=MODEL or req['reasoning_effort']!=EFFORT:raise ValueError('request changed')
    return m


def invoke_once(prompt,images,folder,timeout):
    from .discovery_model import isolated_command
    binary=shutil.which('codex')
    if not binary:raise RuntimeError('codex missing')
    with tempfile.TemporaryDirectory(prefix='local_role_verifier_') as tmp:
        work=Path(tmp);save(work/'schema.json',SCHEMA)
        for i,path in enumerate(images):
            with Image.open(path) as im:im.convert('RGB').save(work/f'image_{i}.png')
        auth=Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex')))/'auth.json'
        command=isolated_command(binary,work,auth)+['/opt/codex','exec','--model',MODEL,
            '-c',f'model_reasoning_effort="{EFFORT}"','--cd','/work','--json','--color','never',
            '--sandbox','read-only','--skip-git-repo-check','--ephemeral','--ignore-user-config','--ignore-rules',
            '-c','features.shell_tool=false','-c','web_search="disabled"',
            '--output-schema','/work/schema.json','--output-last-message','/work/answer.txt']
        for i in range(len(images)):command+=['--image',f'/work/image_{i}.png']
        command+=['-']
        with (folder/'events.jsonl').open('w') as events,(folder/'stderr.txt').open('w') as errors:
            proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=events,stderr=errors,text=True,
                start_new_session=True,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'})
            try:proc.communicate(prompt,timeout=timeout)
            except BaseException:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                if (work/'answer.txt').exists():shutil.copy2(work/'answer.txt',folder/'raw_response.txt')
                raise
        if (work/'answer.txt').exists():shutil.copy2(work/'answer.txt',folder/'raw_response.txt')
        if proc.returncode:raise RuntimeError(f'Codex exit {proc.returncode}; see raw events/stderr')
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines() if line.strip()]
        from cloth_agent.codex_event_status import completion_status
        status=completion_status(events)
        if status['errors'] or not status['final_turn_completed']:raise RuntimeError('incomplete model turn')
        save(folder/'transport.json',dict(completion=status,
            threads=[e.get('thread_id') for e in events if e.get('type')=='thread.started'],
            turns=[e for e in events if e.get('type')=='turn.completed']))
        return json.loads((folder/'raw_response.txt').read_text())


def verdict(answer):
    if answer is None:return 'ERROR'
    values=list(answer.values())
    if all(v=='YES' for v in values):return 'ACCEPT'
    if 'NO' in values:return 'REJECT'
    return 'UNCERTAIN'


def run(root,timeout=240):
    root=Path(root).resolve();m=verify(root)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for qid,condition in m['schedule']:
            folder=root/'calls'/f'{qid}_{condition}'
            if folder.exists():
                if not (folder/'result.json').exists():save(folder/'result.json',dict(status='INTERRUPTED_NOT_RETRIED',answer=None,wall_latency_s=None))
                continue
            folder.mkdir(parents=True);request=root/'requests'/folder.name
            save(folder/'started.json',dict(unix=time.time(),request_sha256=sha(request/'request.json')))
            started=time.perf_counter();result=dict(status='ERROR',answer=None)
            try:
                answer=invoke_once((request/'prompt.txt').read_text(),[p for _,p in request_images(root,qid,condition)],folder,timeout)
                save(folder/'parsed_response.json',answer);jsonschema.validate(answer,SCHEMA)
                result.update(status='OK',answer=answer)
            except Exception as exc:result['error']=f'{type(exc).__name__}: {exc}'
            finally:
                result['wall_latency_s']=time.perf_counter()-started;save(folder/'result.json',result)
            print(json.dumps(dict(query=qid,condition=condition,**result)),flush=True)
        report(root)


def report(root):
    root=Path(root);m=verify(root);truth=read(root/'private/evaluation_labels.json');rows=[]
    for qid in IDS:
        for condition in ('L','G'):
            p=root/'calls'/f'{qid}_{condition}'/'result.json'
            if not p.exists():continue
            result=read(p);a=result['answer']
            rows.append(dict(query=qid,condition=condition,expected=truth[qid],
                same_manipulation_role=a['same_manipulation_role'] if a else 'ERROR',
                same_effect_likely=a['same_effect_likely'] if a else 'ERROR',
                verdict=verdict(a),wall_latency_s=result['wall_latency_s']))
    prior_unchanged=snapshot(Path(m['source']))==read(root/'private/source_snapshot.json')
    paired={q:{r['condition']:r for r in rows if r['query']==q} for q in IDS}
    complete=len(rows)==6
    outcomes={}
    if complete:
        outcomes=dict(G_retains_q01=paired['q01']['G']['verdict']=='ACCEPT',
            G_retains_q03=paired['q03']['G']['verdict']=='ACCEPT',G_rejects_q04=paired['q04']['G']['verdict']=='REJECT',
            q04_L_accept_to_G_reject=paired['q04']['L']['verdict']=='ACCEPT' and paired['q04']['G']['verdict']=='REJECT',
            q04_prior_A_accept_to_G_reject=paired['q04']['G']['verdict']=='REJECT')
    save(root/'metrics.json',dict(rows=rows,complete=complete,criteria=outcomes,
        previous_experiment_unchanged=prior_unchanged,robot_calls=0,application_retries=0,
        attempted_calls=len(list((root/'calls').glob('*/started.json')))))
    if not rows:return
    with (root/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    combined=Image.new('RGB',(1920,3*690),'#121c2d')
    for qi,qid in enumerate(IDS):
        board=Image.new('RGB',(1920,690),'#121c2d');d=ImageDraw.Draw(board)
        for ci,c in enumerate(('L','G')):
            r=paired[qid].get(c)
            if not r:continue
            x=ci*960
            d.text((x+16,12),f'{qid} / {c}: {r["verdict"]} | role={r["same_manipulation_role"]} | effect={r["same_effect_likely"]}',font=font(20),fill='white')
            latency=r['wall_latency_s'];lat='n/a' if latency is None else f'{latency:.2f}s'
            d.text((x+16,42),f'Wall: {lat} | user-fixed expected: {truth[qid]}',font=font(18),fill='#a9c3df')
            with Image.open(root/'requests'/f'{qid}_{c}'/'input_composite.png') as im:
                shown=ImageOps.contain(im.convert('RGB'),(950,600))
            board.paste(shown,(x,85))
        board.save(root/f'{qid}_LG.png');combined.paste(board,(0,qi*690))
    combined.save(root/'LG_results.png')
    lines=['# Local role verifier probe','',
        'Six independent gpt-6-astra/high calls; one frozen historical experience and three frozen condition-A candidates. No new Pa, new action, localization, retrieval or robot connection.',
        '', '## Fixed protocol','',
        '- L: demo local BEFORE/action, query local candidate/action, demo local AFTER.',
        '- G: the identical three local files plus full demo BEFORE/action, full query/candidate/action, full demo AFTER.',
        '- Crops: 320×320 native pixels, centered on the frozen Pa, no resizing or rotation. AFTER uses the same coordinate window as BEFORE. Out-of-frame pixels are gray padding; q04 has 37 padded rows at the bottom.',
        '- Arrows display frozen directions at 110 pixels for legibility; this is not a new movement magnitude. Original target regions and historical endpoint remain preserved in manifest.json. No target is regenerated.',
        '- Models receive separate original-resolution input images; composites are saved contact sheets of those exact images for inspection. L and G share byte-identical local inputs.',
        '- ACCEPT requires both YES; any NO is REJECT; otherwise UNCERTAIN. Invalid/failed output is ERROR. No repair, retry, prompt tuning or additional model invocation.',
        '- q01/q03 expected ACCEPT; q04 expected REJECT. These labels were fixed by the user; the private q04 garment-part explanation is not in any model prompt or input.',
        '', '## Paired judgments','',
        '| Candidate | Condition | Same role | Same effect | Decision | Wall seconds |','|---|---|---|---|---|---:|']
    for r in rows:
        lat=r['wall_latency_s'];lat='—' if lat is None else f'{lat:.2f}'
        lines.append(f'| {r["query"]} | {r["condition"]} | {r["same_manipulation_role"]} | {r["same_effect_likely"]} | {r["verdict"]} | {lat} |')
    lines+=['','## Primary criteria','']
    for k,v in outcomes.items():lines.append(f'- {k}: **{v}**')
    if complete:
        l,g=paired['q04']['L']['verdict'],paired['q04']['G']['verdict']
        lines+=['',f'q04: prior proposal A accepted this point; local verifier L = {l}; global verifier G = {g}.',
            'A rejection by G alone does not demonstrate the benefit of added context if L already rejects. An UNCERTAIN result is not counted as successful rejection. Positive retention and negative rejection must be considered together.']
    lines+=['',f'Previous experiment unchanged (all source file SHA-256 hashes): **{prior_unchanged}**.',
        '', '## Images and raw outputs','', '- [All L/G results](LG_results.png)']
    for qid in IDS:
        lines.append(f'- [{qid} L/G comparison]({qid}_LG.png)')
        for c in ('L','G'):lines.append(f'  - [{c} input composite](requests/{qid}_{c}/input_composite.png), [raw response](calls/{qid}_{c}/raw_response.txt), [parsed JSON](calls/{qid}_{c}/parsed_response.json)')
    lines+=['','## Limits','',
        'This is a three-candidate exploratory contrast with one sample per condition, not a general accuracy estimate or causal proof of a context benefit. Global views add image content and image count; the experiment cannot separate these factors. A 320px crop may already expose role cues, and the fixed-coordinate AFTER crop does not track the moved fabric. The verifier judges visible plausibility, not physical outcomes. All six calls use new ephemeral isolated sessions; only specified images, prompt and schema are exposed. CLI-internal reconnects, if any, are retained in events; no application retry is made.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    for name in ('run','report'):
        a=sub.add_parser(name);a.add_argument('--experiment',type=Path,required=True)
        if name=='run':a.add_argument('--timeout-s',type=float,default=240)
    a=p.parse_args(argv)
    if a.command=='prepare':print(prepare(a.source,a.output))
    elif a.command=='run':run(a.experiment,a.timeout_s)
    else:report(a.experiment)


if __name__=='__main__':main()
