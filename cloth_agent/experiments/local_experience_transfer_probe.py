"""Independent offline, eight-call local experience transfer probe.

prepare --spec selection.json --output DATASET
run --dataset DATASET --output RUN
report --dataset DATASET --output RUN

No robot, graph, planner, matcher, retrieval, crop or depth dependencies.
Only the existing isolated Codex transport's namespace helper is reused.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

import jsonschema
from PIL import Image, ImageDraw, ImageFont, ImageOps

MODEL = 'gpt-6-astra'
EFFORT = 'high'
PROTOCOL = 'local-experience-transfer-probe-v1'
POINT = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}
BOX = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4}
SCHEMA = {'type': 'object', 'additionalProperties': False,
          'properties': {'applicable': {'type': 'string', 'enum': ['YES', 'NO', 'AMBIGUOUS']},
                         'grasp_px': {'anyOf': [POINT, {'type': 'null'}]},
                         'move_direction_px': {'anyOf': [POINT, {'type': 'null'}]},
                         'target_region_px': {'anyOf': [BOX, {'type': 'null'}]}},
          'required': ['applicable', 'grasp_px', 'move_direction_px', 'target_region_px']}
PROMPT = '''You are evaluating zero-shot local manipulation experience transfer, offline.
Do not match the whole garment state. Do not require the same semantic garment
part or the same material point. Decide whether ANY local region in the complete
QUERY image gives reason for the historical manipulation to produce a similar
effect. Similar color, a wrinkle, or a rolled edge alone is not sufficient.
The demonstrated action is a grasp at red Pa, followed by movement in the yellow
arrow direction. Transfer the manipulation relative to local cloth configuration;
do not copy absolute image coordinates or the screen-space direction blindly.
Images are ordered and identified below. No query ROI or candidate location is
provided. Independently search the full query. In condition A no demo AFTER is
provided: infer the likely local effect from BEFORE and the demonstrated action.
In condition B use the observed AFTER to refine that effect. Do not require the
whole query to become identical to the demonstration AFTER.
Return ONLY the exact schema JSON, no reasoning or additional fields.
applicable is YES, NO, or AMBIGUOUS. Use AMBIGUOUS when applicability cannot be
determined from visible evidence. For YES give grasp_px=[x,y], a Euclidean UNIT
move_direction_px=[dx,dy], and target_region_px=[xmin,ymin,xmax,ymax], a nonempty
axis-aligned rectangle of reasonable release positions. All coordinates refer
to the original complete QUERY pixels, origin top-left, x right, y down. The
direction points from grasp toward the intended release region. For NO or
AMBIGUOUS all three coordinate fields MUST be null. No confidence threshold.
'''


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(vector):
    n = math.hypot(*vector)
    if not math.isfinite(n) or n < 1e-8: raise ValueError('invalid direction')
    return [float(v) / n for v in vector]


def angular_error(a, b):
    a, b = unit(a), unit(b)
    return math.degrees(math.acos(max(-1, min(1, sum(x*y for x, y in zip(a, b))))))


def inside(point, polygon):
    x, y = point; result = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (x-a[0])*(b[1]-a[1])-(y-a[1])*(b[0]-a[0])
        if abs(cross) < 1e-7 and min(a[0],b[0]) <= x <= max(a[0],b[0]) and min(a[1],b[1]) <= y <= max(a[1],b[1]): return True
        if (a[1] > y) != (b[1] > y) and x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]: result = not result
    return result


def font(size=22):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)


def arrow(draw, start, end, color='#ffe747'):
    draw.line([tuple(start), tuple(end)], fill=color, width=5)
    u = unit([end[0]-start[0], end[1]-start[1]])
    tip = [tuple(end), (end[0]-17*u[0]+8*u[1], end[1]-17*u[1]-8*u[0]),
           (end[0]-17*u[0]-8*u[1], end[1]-17*u[1]+8*u[0])]
    draw.polygon(tip, fill=color)
    x, y = start; draw.ellipse((x-8,y-8,x+8,y+8), fill='#ff364f', outline='white', width=2)
    draw.text((x+10,y+5), 'Pa', fill='white', font=font(), stroke_width=2, stroke_fill='black')


def validate_answer(answer, size):
    jsonschema.validate(answer, SCHEMA)
    keys = ('grasp_px', 'move_direction_px', 'target_region_px')
    if answer['applicable'] != 'YES':
        if any(answer[k] is not None for k in keys): raise ValueError('non-YES coordinates must be null')
        return
    if any(answer[k] is None for k in keys): raise ValueError('YES needs all coordinates')
    if any(not math.isfinite(x) for k in keys for x in answer[k]): raise ValueError('nonfinite coordinates')
    x, y = answer['grasp_px']; w, h = size
    if not (0 <= x < w and 0 <= y < h): raise ValueError('grasp outside query')
    if abs(math.hypot(*answer['move_direction_px']) - 1) > .01: raise ValueError('direction not unit length')
    a, b, c, d = answer['target_region_px']
    if not (0 <= a < c < w and 0 <= b < d < h): raise ValueError('invalid target rectangle')


def prepare(spec_path, output):
    spec = read(spec_path); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    if len(spec['queries']) != 4: raise ValueError('exactly four queries required')
    labels = [q['ground_truth']['applicable'] for q in spec['queries']]
    if labels.count('YES') != 2 or labels.count('NOT_APPLICABLE') != 2: raise ValueError('need two positives and two negatives')
    demo = spec['demo']; files = {}; provenance = {}; truth = {}
    def copy_rgb(source, relative):
        destination = output / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as im: im.convert('RGB').save(destination)
        files[relative] = digest(destination)
        provenance[relative] = dict(source=str(Path(source).resolve()), source_sha256=digest(source))
        return destination
    before = copy_rgb(demo['before'], 'public/demo_before.png')
    copy_rgb(demo['after'], 'public/demo_after.png')
    with Image.open(before) as im:
        im = im.convert('RGB'); arrow(ImageDraw.Draw(im), demo['pa'], demo['target'])
        im.save(output/'public/demo_action_overlay.png')
    files['public/demo_action_overlay.png'] = digest(output/'public/demo_action_overlay.png')
    queries = []
    for index, q in enumerate(spec['queries'], 1):
        qid = f'q{index:02d}'; relative = f'public/{qid}.png'; p = copy_rgb(q['image'], relative)
        gt = q['ground_truth']
        with Image.open(p) as im: size = list(im.size)
        if gt['applicable'] == 'YES':
            if not gt['regions']: raise ValueError('positive needs a region')
            for r in gt['regions']:
                if len(r['polygon_px']) < 3 or not inside(r['reference_pa'], r['polygon_px']): raise ValueError('invalid positive annotation')
                for x, y in r['polygon_px'] + [r['reference_pa']]:
                    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < size[0] and 0 <= y < size[1]): raise ValueError('annotation outside image')
                r['move_direction_px'] = unit(r['move_direction_px'])
        elif gt.get('regions'): raise ValueError('negative must not contain correct regions')
        truth[qid] = gt; queries.append(dict(id=qid, image=relative, size=size))
    if len({files[q['image']] for q in queries}) != 4: raise ValueError('duplicate query images')
    if any(files[q['image']] in [files['public/demo_before.png'], files['public/demo_after.png']] for q in queries): raise ValueError('query duplicates demonstration')
    save(output/'private/ground_truth.json', dict(annotation_status=spec['annotation_status'], queries=truth))
    save(output/'private/selection.json', spec)
    save(output/'private/provenance.json', provenance)
    schedule = [(q['id'], c) for i, q in enumerate(queries) for c in (('A','B') if i % 2 == 0 else ('B','A'))]
    save(output/'manifest.json', dict(protocol=PROTOCOL, model=MODEL, reasoning_effort=EFFORT,
         queries=queries, files=files, ground_truth_sha256=digest(output/'private/ground_truth.json'),
         prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(), schema=SCHEMA,
         schedule=schedule, max_calls=8, demo_pa=demo['pa'], demo_target=demo['target'],
         annotation_status=spec['annotation_status'], created_unix=time.time(), robot_calls=0))
    for q in queries:
        with Image.open(output/q['image']) as im: im = im.convert('RGB')
        draw = ImageDraw.Draw(im)
        for region in truth[q['id']].get('regions', []):
            draw.line([tuple(p) for p in region['polygon_px'] + region['polygon_px'][:1]], fill='#40f3b5', width=4)
            p=region['reference_pa']; v=region['move_direction_px']; arrow(draw,p,[p[0]+110*v[0],p[1]+110*v[1]])
        im.save(output/f'private/{q["id"]}_annotation.png')
    return output/'manifest.json'


def inputs_for(dataset, manifest, qid, condition):
    q = next(q for q in manifest['queries'] if q['id'] == qid)
    names = ['demo_before.png', 'demo_action_overlay.png']
    if condition == 'B': names.append('demo_after.png')
    names.append(qid+'.png')
    images = [Path(dataset)/'public'/name for name in names]
    roles = ['DEMO BEFORE', 'DEMO ACTION OVERLAY'] + (['DEMO AFTER'] if condition == 'B' else []) + ['QUERY']
    prompt = PROMPT + f'\nCondition {condition}. Query size: {q["size"]}.\n'
    prompt += '\n'.join(f'Image {i+1}: {role}.' for i, role in enumerate(roles))
    return prompt, images


def invoke_once(prompt, images, folder, timeout):
    # A fresh ephemeral namespace for every call. No labels, dataset directory,
    # history, graph, output directory or host config is mounted inside it.
    from .discovery_model import isolated_command
    binary = shutil.which('codex')
    if not binary: raise RuntimeError('codex CLI missing')
    with tempfile.TemporaryDirectory(prefix='local_transfer_probe_') as tmp:
        work = Path(tmp); save(work/'schema.json', SCHEMA)
        for i, path in enumerate(images):
            with Image.open(path) as im: im.convert('RGB').save(work/f'image_{i}.png')
        auth = Path(os.environ.get('CODEX_HOME', str(Path.home()/'.codex')))/'auth.json'
        cmd = isolated_command(binary, work, auth) + ['/opt/codex', 'exec', '--model', MODEL,
            '-c', f'model_reasoning_effort="{EFFORT}"', '--cd', '/work', '--json', '--color', 'never',
            '--sandbox', 'read-only', '--skip-git-repo-check', '--ephemeral', '--ignore-user-config',
            '--ignore-rules', '-c', 'features.shell_tool=false', '-c', 'web_search="disabled"',
            '--output-schema', '/work/schema.json', '--output-last-message', '/work/answer.txt']
        for i in range(len(images)): cmd += ['--image', f'/work/image_{i}.png']
        cmd += ['-']
        with (folder/'events.jsonl').open('w') as events, (folder/'stderr.txt').open('w') as errors:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=events, stderr=errors,
                text=True, start_new_session=True, env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'})
            try: proc.communicate(prompt, timeout=timeout)
            except BaseException:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                if (work/'answer.txt').exists(): shutil.copy2(work/'answer.txt',folder/'answer.txt')
                raise
        if (work/'answer.txt').exists(): shutil.copy2(work/'answer.txt', folder/'answer.txt')
        if proc.returncode: raise RuntimeError(f'Codex exit {proc.returncode}; see stderr/events')
        events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines() if line.strip()]
        from cloth_agent.codex_event_status import completion_status
        status = completion_status(events)
        if status['errors'] or not status['final_turn_completed']: raise RuntimeError('incomplete model turn')
        # Explicitly retain CLI reconnects and usage. No application retry/repair.
        save(folder/'transport.json', dict(completion=status, model=MODEL, reasoning_effort=EFFORT,
            turns=[e for e in events if e.get('type') == 'turn.completed']))
        return json.loads((folder/'answer.txt').read_text())


def verify_dataset(dataset):
    m = read(dataset/'manifest.json')
    if m['protocol'] != PROTOCOL or m['model'] != MODEL or m['reasoning_effort'] != EFFORT: raise ValueError('protocol mismatch')
    if m['max_calls'] != 8 or len(m['schedule']) != 8 or len(set(map(tuple,m['schedule']))) != 8: raise ValueError('invalid eight-call schedule')
    if set(map(tuple,m['schedule'])) != {(q['id'],c) for q in m['queries'] for c in ('A','B')}: raise ValueError('schedule must cover four paired queries')
    for relative, sha in m['files'].items():
        if digest(dataset/relative) != sha: raise ValueError('input image changed: '+relative)
    if digest(dataset/'private/ground_truth.json') != m['ground_truth_sha256']: raise ValueError('labels changed after preparation')
    if hashlib.sha256(PROMPT.encode()).hexdigest() != m['prompt_sha256'] or m['schema'] != SCHEMA: raise ValueError('protocol changed after freeze')
    return m


def run(dataset, output, timeout=240):
    dataset=Path(dataset).resolve(); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    m=verify_dataset(dataset)
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=dict(dataset=str(dataset),manifest_sha256=digest(dataset/'manifest.json'),model=MODEL,reasoning_effort=EFFORT,max_calls=8)
        if (output/'run_config.json').exists() and read(output/'run_config.json') != frozen: raise ValueError('run input changed')
        save(output/'run_config.json',frozen)
        for qid,condition in m['schedule']:
            folder=output/f'{qid}_{condition}'
            if folder.exists():
                if not (folder/'result.json').exists():
                    save(folder/'result.json',dict(status='INTERRUPTED_NOT_RETRIED',wall_latency_s=None,answer=None))
                continue  # Never repeat a started, failed, or interrupted call.
            folder.mkdir(); prompt,images=inputs_for(dataset,m,qid,condition)
            (folder/'prompt.txt').write_text(prompt)
            save(folder/'request.json',dict(model=MODEL,reasoning_effort=EFFORT,images=[dict(role=i,sha256=digest(p)) for i,p in enumerate(images)],schema=SCHEMA))
            save(folder/'started.json',dict(unix=time.time(),condition=condition,query=qid))
            started=time.perf_counter(); result=dict(status='ERROR',answer=None)
            try:
                answer=invoke_once(prompt,images,folder,timeout)
                save(folder/'raw_answer.json',answer)
                validate_answer(answer,next(q['size'] for q in m['queries'] if q['id']==qid))
                result.update(status='OK',answer=answer)
            except Exception as exc: result['error']=f'{type(exc).__name__}: {exc}'
            finally:
                result['wall_latency_s']=time.perf_counter()-started
                save(folder/'result.json',result)
            print(json.dumps(dict(query=qid,condition=condition,**result),ensure_ascii=False),flush=True)
        report(dataset,output)


def score(answer, truth):
    result=dict(pa_in_region=None, angular_error_deg=None, matched_region=None)
    if truth['applicable'] != 'YES' or answer is None or answer['applicable'] != 'YES': return result
    regions=truth['regions']; hits=[i for i,r in enumerate(regions) if inside(answer['grasp_px'],r['polygon_px'])]
    result['pa_in_region']=bool(hits)
    # Direction is only meaningful relative to the local region selected.
    # Outside all admissible regions report undefined, not an arbitrary angle.
    if hits:
        i=hits[0];result['matched_region']=i
        result['angular_error_deg']=angular_error(answer['move_direction_px'],regions[i]['move_direction_px'])
    return result


def model_overlay(image, result, destination, title):
    with Image.open(image) as im: im=im.convert('RGB')
    draw=ImageDraw.Draw(im); answer=result.get('answer')
    status=answer['applicable'] if answer else result['status']
    if answer and status=='YES':
        draw.rectangle(tuple(answer['target_region_px']),outline='#54dfff',width=4)
        p=answer['grasp_px'];v=answer['move_direction_px']
        # Fixed 110px arrow encodes direction only; not a movement distance.
        arrow(draw,p,[p[0]+110*v[0],p[1]+110*v[1]])
    board=Image.new('RGB',(im.width,im.height+70),'#121c2d');board.paste(im,(0,70))
    d=ImageDraw.Draw(board);lat=result.get('wall_latency_s');lat='n/a' if lat is None else f'{lat:.2f}s'
    d.text((14,8),f'{title} | {status} | wall {lat}',font=font(),fill='white')
    d.text((14,39),'Red: Pa | Yellow: unit direction (display 110px) | Cyan: target region',font=font(18),fill='#b3c7dd')
    board.save(destination)


def report(dataset, output):
    dataset=Path(dataset);output=Path(output);m=verify_dataset(dataset)
    gt=read(dataset/'private/ground_truth.json');rows=[]
    # Review-only overview; these private annotations are never model inputs.
    board=Image.new('RGB',(1600,1040),'#121c2d');draw=ImageDraw.Draw(board)
    for index,q in enumerate(m['queries']):
        x=(index%2)*800;y=(index//2)*520
        with Image.open(dataset/f'private/{q["id"]}_annotation.png') as im:
            panel=ImageOps.contain(im.convert('RGB'),(800,450))
        board.paste(panel,(x,y+55))
        label=gt['queries'][q['id']]['applicable']
        draw.text((x+12,y+10),q['id']+' | proxy label: '+label,font=font(),fill='white')
    board.save(output/'query_annotations.png')
    demo_board=Image.new('RGB',(1920,440),'#121c2d');draw=ImageDraw.Draw(demo_board)
    for index,name in enumerate(('demo_before','demo_action_overlay','demo_after')):
        with Image.open(dataset/f'public/{name}.png') as im:
            panel=ImageOps.contain(im.convert('RGB'),(640,360))
        demo_board.paste(panel,(640*index,60));draw.text((640*index+12,12),name,font=font(),fill='white')
    demo_board.save(output/'demonstration.png')
    for q in m['queries']:
        panels=[]
        for condition in ('A','B'):
            folder=output/f'{q["id"]}_{condition}';p=folder/'result.json'
            if not p.exists(): continue
            result=read(p);answer=result.get('answer');prediction=answer['applicable'] if answer else 'ERROR'
            row=dict(query=q['id'],condition=condition,truth=gt['queries'][q['id']]['applicable'],
                     prediction=prediction,wall_latency_s=result.get('wall_latency_s'),**score(answer,gt['queries'][q['id']]))
            rows.append(row)
            model_overlay(dataset/q['image'],result,folder/'overlay.png',q['id']+' / '+condition)
            with Image.open(folder/'overlay.png') as im: panels.append(ImageOps.contain(im.convert('RGB'),(800,520)))
        if len(panels)==2:
            board=Image.new('RGB',(1600,520),'#121c2d')
            for i,panel in enumerate(panels):board.paste(panel,(i*800,0))
            board.save(output/f'{q["id"]}_AB.png')
    if not rows:return
    with (output/'benchmark.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    counts=[]
    for condition in ('A','B'):
        for truth in ('YES','NOT_APPLICABLE'):
            counts.append(dict(condition=condition,truth=truth,**{p:sum(r['condition']==condition and r['truth']==truth and r['prediction']==p for r in rows) for p in ('YES','NO','AMBIGUOUS','ERROR')}))
    save(output/'metrics.json',dict(annotation_status=gt['annotation_status'],rows=rows,confusion=counts,
         attempted_calls=len(list(output.glob('*/started.json'))),completed_records=len(rows),
         application_retries=0,robot_calls=0))
    with (output/'confusion.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(counts[0]));writer.writeheader();writer.writerows(counts)
    lines=['# Local experience transfer probe','',f'Model: {MODEL}; reasoning: {EFFORT}. One fixed demonstration, four complete RGB queries, two conditions; planned exactly eight independent ephemeral invocations.',
        '',f'Annotation status: **{gt["annotation_status"]}**. Labels were fixed before model calls and never supplied to the model.',
        '', 'A receives BEFORE + action overlay + QUERY. B receives the same inputs plus AFTER. No textual effect hint is supplied in either condition. No training, query ROI, retrieval, crop, depth, robot or graph-state matching.',
        '', '## Applicability confusion counts','', '| Condition | Reference label | YES | NO | AMBIGUOUS | ERROR |','|---|---|---:|---:|---:|---:|']
    for c in counts:lines.append('| '+' | '.join(str(c[k]) for k in ('condition','truth','YES','NO','AMBIGUOUS','ERROR'))+' |')
    lines+=['','AMBIGUOUS is retained as abstention, never silently mapped to NO. ERROR includes invalid JSON/coordinates or transport failure; no repairs or application retries.',
            '', '## Per-query localization and latency','', '| Query | Condition | Label | Prediction | Pa inside | Direction error (deg) | Wall seconds |','|---|---|---|---|---|---:|---:|']
    for r in rows:
        vals=[r[k] for k in ('query','condition','truth','prediction','pa_in_region','angular_error_deg','wall_latency_s')]
        lines.append('| '+' | '.join('—' if x is None else f'{x:.2f}' if isinstance(x,float) else str(x) for x in vals)+' |')
    lines+=['','Direction error uses the reference direction of the predeclared region containing predicted Pa. It is undefined for NO/AMBIGUOUS/ERROR or a grasp outside every accepted region. Alternate valid regions are explicitly annotated beforehand. Target rectangles are visualized but not scored: no target-region ground truth was requested.', '', '## Paired outcome','']
    for c in ('A','B'):
        group=[r for r in rows if r['condition']==c];pos=[r for r in group if r['truth']=='YES'];lats=[r['wall_latency_s'] for r in group if r['wall_latency_s'] is not None]
        import statistics
        correct=sum(r['prediction']==('YES' if r['truth']=='YES' else 'NO') for r in group)
        hit=sum(r['pa_in_region'] is True for r in pos)
        angles=[r['angular_error_deg'] for r in pos if r['angular_error_deg'] is not None]
        lines.append(f'- {c}: applicability exact agreement {correct}/{len(group)}; positive Pa hits {hit}/{len(pos)} (abstentions count as misses); angle mean {statistics.mean(angles) if angles else "undefined"} deg, n={len(angles)}; wall mean {statistics.mean(lats) if lats else "undefined"} s, median {statistics.median(lats) if lats else "undefined"} s.')
    if len(rows)==8:
        paired={q['id']:{r['condition']:r for r in rows if r['query']==q['id']} for q in m['queries']}
        changes=sum(p['A']['prediction']!=p['B']['prediction'] for p in paired.values())
        false_positives=[r['query']+'/'+r['condition'] for r in rows if r['truth']=='NOT_APPLICABLE' and r['prediction']=='YES']
        lines+=['','## Interpretation','',
            f'Adding AFTER changed the applicability decision on {changes}/4 query pairs. Proxy false positives: {", ".join(false_positives) or "none"}.',
            'Positive grasp hits and direction agreement provide limited evidence of local correspondence under zero additional training, not proof of physical applicability. A false positive means a visually plausible pull was proposed where the frozen annotation did not support the demonstrated effect. Compare the paired overlays, not just the YES/NO counts. Do not interpret conditional angular averages with different sample counts as an AFTER improvement. Four selected queries and one sample per condition cannot establish a general capability or a reliable treatment effect.']
    lines+=['','## Visual comparisons','',
            '- [Demonstration: BEFORE / historical action / AFTER](demonstration.png)',
            '- [Four queries and frozen proxy annotations (evaluation only)](query_annotations.png)', '']
    for q in m['queries']:lines.append(f'- [{q["id"]} A/B]({q["id"]}_AB.png)')
    lines+=['','## Limits','', 'This is an exploratory four-query probe, not a performance estimate. The task is zero additional training with one in-context manipulation demonstration, not zero-context inference. Positive labels, admissible grasp areas and direction references are provisional assistant visual annotations, not human-verified or experimentally confirmed applicability. The two negative images contain sleeve rolls/wrinkles without the demonstrated broad chest-covering panel; how hard they are is not established independently. One positive comes from the demonstration recording session but a different trajectory/global configuration; the other is from another session. Multiple physically useful grasps may exist beyond the annotations. No offline score proves manipulation success. A/B has one draw per query and cannot establish a statistically reliable AFTER benefit.', '', 'The frozen private selection contains source paths, the visible demonstration effect, query selection rationales and labeling uncertainty. Files in private/ are never mounted in model sessions. Full raw responses, prompts, events, stderr and measured invocation wall latency are retained. CLI transport reconnects may occur internally and are logged; the script makes no extra model invocations.', '']
    (output/'report.md').write_text('\n'.join(lines))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--spec',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    for name in ('run','report'):
        a=sub.add_parser(name);a.add_argument('--dataset',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
        if name=='run':a.add_argument('--timeout-s',type=float,default=240)
    a=p.parse_args(argv)
    if a.command=='prepare':print(prepare(a.spec,a.output))
    elif a.command=='run':run(a.dataset,a.output,a.timeout_s)
    else:report(a.dataset,a.output)


if __name__=='__main__':main()
