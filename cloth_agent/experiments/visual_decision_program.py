"""Offline S1/S2 visual program: isolated compile, frozen predicates, host rules."""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import re
import time

import jsonschema
from PIL import Image, ImageDraw
from .discovery_model import CodexMultimodal

HARD='obs_7f66d64e53de5d54'

def obj(properties):
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)

def arr(items,**kwargs):return dict(type='array',items=items,**kwargs)
S={'type':'string'}
ID=dict(type='string',pattern='^[a-z][a-z0-9_]*$')
ROI=arr(dict(type='integer',minimum=0),minItems=4,maxItems=4)
PROGRAM=obj(dict(regions=arr(obj(dict(id=ID,semantic_region=S)),minItems=1,maxItems=3),
    predicates=arr(obj(dict(id=ID,region=ID,question=S)),minItems=1,maxItems=6),
    rules=arr(obj({'if':S,'result':dict(enum=['S1','S2'])}),minItems=2,maxItems=6),
    fallback=dict(enum=['UNKNOWN'])))
COMPILE='''Compile one reusable executable visual decision program for S1 versus S2.
You receive ONLY their actual node descriptions and representative RGB images. No queries.
Derive regions, predicates and rules from this evidence, not a generic folding ideal.
Use the supplied side_convention consistently and encode it in semantic_region text for the executor; never assume image-left equals garment-right. Do not treat
perfect flattening as necessary unless the node descriptions require it. Do not infer hidden layers.
Return the supplied JSON schema only, no chain-of-thought. At most 3 semantic regions and 6
YES/NO predicates. Questions must be concrete visual tests, not requests to classify a node.
Do not mention S1/S2, labels, or classification in region IDs, predicate IDs, or questions.
Rules use only predicate_id == YES or predicate_id == NO, joined by && / || and parentheses.
No Python, functions, arithmetic or other operators. Provide at least one rule for each class.
Host evaluates all rules: conflicting matches or any UNKNOWN predicate referenced by the rules
produce UNKNOWN; otherwise a unique matched class wins, no match gives UNKNOWN.
Do not include explanations, model confidence thresholds or test-specific adjustments.
Image order and descriptions follow:\n'''

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tree(expression,ids):
    parsed=ast.parse(expression.replace('&&',' and ').replace('||',' or '),mode='eval').body
    def check(n):
        if isinstance(n,ast.BoolOp) and isinstance(n.op,(ast.And,ast.Or)):
            for v in n.values:check(v)
        elif (isinstance(n,ast.Compare) and len(n.ops)==1 and isinstance(n.ops[0],ast.Eq)
              and isinstance(n.left,ast.Name) and n.left.id in ids and len(n.comparators)==1
              and isinstance(n.comparators[0],ast.Name) and n.comparators[0].id in ('YES','NO')):pass
        else:raise ValueError('unsupported rule expression')
    check(parsed);return parsed

def validate_program(p):
    jsonschema.validate(p,PROGRAM)
    regions=[r['id'] for r in p['regions']];ids=[r['id'] for r in p['predicates']]
    if len(set(regions))!=len(regions) or len(set(ids))!=len(ids):raise ValueError('duplicate IDs')
    for pred in p['predicates']:
        if pred['region'] not in regions:raise ValueError('unknown region')
        if re.search(r'\bS[12]\b',json.dumps(pred),re.I):raise ValueError('predicate leaks class label')
    for rule in p['rules']:tree(rule['if'],ids)
    if {r['result'] for r in p['rules']}!={'S1','S2'}:raise ValueError('need rules for both classes')

def decide(p,results):
    validate_program(p)
    ids={x['id'] for x in p['predicates']}
    if len(results)!=len(ids) or {r['predicate_id'] for r in results}!=ids:
        raise ValueError('missing, extra or duplicate predicate results')
    values={r['predicate_id']:r['value'] for r in results}
    if any(v not in ('YES','NO','UNKNOWN') for v in values.values()):raise ValueError('invalid value')
    trees=[tree(r['if'],ids) for r in p['rules']]
    needed={n.left.id for t in trees for n in ast.walk(t) if isinstance(n,ast.Compare)}
    unknown=sorted(i for i in needed if values[i]=='UNKNOWN')
    if unknown:return dict(decision='UNKNOWN',matched_rules=[],unknown_required=unknown,reason='unknown_required_predicate')
    def evaluate(n):
        if isinstance(n,ast.Compare):return values[n.left.id]==n.comparators[0].id
        vs=[evaluate(v) for v in n.values]
        return all(vs) if isinstance(n.op,ast.And) else any(vs)
    matched=[i for i,t in enumerate(trees) if evaluate(t)]
    labels={p['rules'][i]['result'] for i in matched}
    return dict(decision=next(iter(labels)) if len(labels)==1 else 'UNKNOWN',matched_rules=matched,
                unknown_required=[],reason='unique_class' if len(labels)==1 else 'conflict' if labels else 'no_rule')

def call(model,prompt,images,schema,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    prompt += '\nReturn ONLY JSON matching this schema:\n'+json.dumps(schema)
    request=dict(prompt=prompt,images=[dict(path=str(Path(x).resolve()),sha256=sha(x)) for x in images],schema=schema)
    if (folder/'response.json').exists() and json.loads((folder/'request.json').read_text())!=request:
        raise ValueError('cached response request mismatch')
    save(folder/'request.json',request)
    if (folder/'response.json').exists():
        result=json.loads((folder/'response.json').read_text())
    else:
        result=model.invoke(prompt,images,schema);save(folder/'response.json',result)
    raw=result['text'].strip()
    if raw.startswith('```'):raw=re.sub(r'^```(?:json)?\s*|\s*```$','',raw)
    value=json.loads(raw);jsonschema.validate(value,schema)
    save(folder/'parsed.json',value)
    return value,result

def prepare(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    dataset=list(csv.DictReader((args.dataset/'samples.csv').open()))
    graph=json.loads(args.graph.read_text())
    trajectory=json.loads(Path(graph['seed_trajectory_path']).read_text())
    nodes=[n for n in graph['nodes'] if n['node_id'] in ('S1','S2')]
    reps=[];descriptions=[]
    for n in nodes:
        images=[]
        for oid in n['representative_observations']:
            artifact=trajectory['observations'][oid]['rgb'];src=Path(artifact['path'])
            if sha(src)!=artifact['sha256']:raise ValueError('representative changed')
            target=out/'compile'/'representatives'/f'{n["node_id"]}_{oid}.png'
            target.parent.mkdir(parents=True,exist_ok=True)
            Image.open(src).convert('RGB').save(target);images.append(len(reps));reps.append(target)
        descriptions.append(dict(node_id=n['node_id'],task_context=n['task_context'],
                                 side_convention=graph['goal'].get('side_convention',''),
                                 observation_condition=n['observation_condition'],image_indices=images))
    if len(nodes)!=2:raise ValueError('S1/S2 required')
    pixel=lambda f:hashlib.sha256(Image.open(f).convert('RGB').tobytes()).hexdigest()
    rep_hashes={pixel(p) for p in reps}
    compile_runs={r['run_id'] for r in dataset if r['pixel_sha256'] in rep_hashes}
    queries=[r for r in dataset if r['node_label'] in ('S1','S2') and r['run_id'] not in compile_runs]
    if not queries or not any(r['sample_id']==HARD for r in queries):raise ValueError('missing hard negative')
    if any(r['pixel_sha256'] in rep_hashes for r in queries):raise ValueError('split leakage')
    for r in queries:
        if sha(r['image'])!=r['file_sha256']:raise ValueError('query image changed')
    manifest=dict(protocol='visual-decision-program-v1',model=args.model,
        compiler_inputs=descriptions,representatives=[dict(path=str(p.resolve()),sha256=sha(p)) for p in reps],
        queries=queries,compile_runs=sorted(compile_runs),split='exclude entire representative runs; all remaining S1/S2 samples',
        label_provenance='Existing graph/matcher labels, not independent human ground truth',
        dataset_sha256=sha(args.dataset/'samples.csv'),rules_semantics='all matched rules; conflict/no match/any required UNKNOWN -> UNKNOWN',
        thresholds='No confidence threshold; model UNKNOWN propagated; no test-time tuning')
    f=out/'manifest.json'
    if f.exists() and json.loads(f.read_text())!=manifest:raise ValueError('manifest differs; use a new experiment directory')
    save(f,manifest)
    return descriptions,reps,queries

def compile_program(args,descriptions,reps):
    out=args.output;path=out/'visual_program.json'
    if path.exists():
        p=json.loads(path.read_text());validate_program(p)
        if json.loads((out/'program_freeze.json').read_text())['sha256']!=sha(path):raise ValueError('frozen program changed')
        return p
    p,_=call(CodexMultimodal(args.model,args.timeout),COMPILE+json.dumps(descriptions),reps,PROGRAM,out/'compile'/'request')
    validate_program(p);save(path,p)
    save(out/'visual_program_S1_S2.json',p)
    save(out/'program_freeze.json',dict(sha256=sha(path),compiled_unix_s=time.time(),compiler_inputs_sha256=sha(out/'compile/request/request.json')))
    path.chmod(0o444);(out/'visual_program_S1_S2.json').chmod(0o444)
    return p

def execute_query(args,program,row):
    began=time.monotonic();folder=args.output/'audit_images'/row['sample_id'];folder.mkdir(parents=True,exist_ok=True)
    program_hash=sha(args.output/'visual_program.json')
    expected=json.loads((args.output/'program_freeze.json').read_text())['sha256']
    if expected!=program_hash:raise ValueError('program changed before execution')
    original=folder/'original.png';Image.open(row['image']).convert('RGB').save(original)
    w,h=Image.open(original).size
    locate_schema=obj(dict(regions=arr(obj(dict(region_id=dict(enum=[r['id'] for r in program['regions']]),
        roi=ROI,visual_evidence=S)),minItems=len(program['regions']),maxItems=len(program['regions']))))
    prompt=('Locate each frozen semantic region in this query RGB. Return a bounding box [x0,y0,x1,y1] in original pixels, '
        f'image size {w}x{h}. Use anatomy, not fixed screen sides. Include surrounding fabric needed to judge the question. '
        'If uncertain use the full image and state uncertainty. Do not classify a node, answer predicates, or invent regions. '
        'No chain-of-thought; only concise visible localization evidence. Frozen regions: '+json.dumps(program['regions']))
    model=CodexMultimodal(args.model,args.timeout)
    loc,lmeta=call(model,prompt,[original],locate_schema,folder/'localize')
    if {r['region_id'] for r in loc['regions']}!={r['id'] for r in program['regions']}:raise ValueError('region IDs mismatch')
    crops=[];mapping=[];boxes={}
    im=Image.open(original).convert('RGB');overlay=im.copy();draw=ImageDraw.Draw(overlay)
    for i,r in enumerate(loc['regions']):
        x0,y0,x1,y1=r['roi']
        if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('invalid ROI')
        path=folder/f'crop_{r["region_id"]}.png';im.crop(r['roi']).save(path);crops.append(path)
        mapping.append(dict(region_id=r['region_id'],image_index=i+1,roi=r['roi']))
        boxes[r['region_id']]=r['roi'];draw.rectangle(r['roi'],outline='red',width=3);draw.text((x0,y0),r['region_id'],fill='red')
    overlay.save(folder/'roi_overlay.png')
    result_schema=obj(dict(results=arr(obj(dict(predicate_id=dict(enum=[p['id'] for p in program['predicates']]),
        value=dict(enum=['YES','NO','UNKNOWN']),confidence=dict(type='number',minimum=0,maximum=1),
        visual_evidence=S,roi=ROI)),minItems=len(program['predicates']),maxItems=len(program['predicates']))))
    prompt=('Execute EVERY frozen predicate independently, in the given order. Image 0 is the original query; '
        'remaining images are the actual region crops listed below. Assess the visible spatial relation asked, '
        'not garment similarity or a likely action history. Answer only YES/NO/UNKNOWN, confidence and brief visual evidence; '
        'no chain-of-thought, node classification, new predicates, or rule edits. Use UNKNOWN if anatomy or the required relation '
        'cannot be verified. For roi echo the corresponding region bounding box in original-image coordinates. '
        'The host alone computes the final node; you are not given its rules or class labels.\n'+json.dumps(dict(
            regions=program['regions'],predicates=program['predicates'],attached_crops=mapping)))
    values,vmeta=call(model,prompt,[original,*crops],result_schema,folder/'execute')
    for v in values['results']:
        pred=next(p for p in program['predicates'] if p['id']==v['predicate_id'])
        if v['roi']!=boxes[pred['region']]:raise ValueError('predicate ROI differs from supplied crop')
    if sha(args.output/'visual_program.json')!=expected:raise ValueError('program changed during execution')
    decision=decide(program,values['results'])
    save(folder/'predicate_results.json',values)
    save(folder/'host_decision.json',dict(**decision,program_sha256=expected,
        model_elapsed_s=lmeta['elapsed_s']+vmeta['elapsed_s'],wall_s=time.monotonic()-began,
        crop_audit='These exact crops were attached to executor; internal model attention is not observable.'))
    return dict(sample_id=row['sample_id'],run_id=row['run_id'],label=row['node_label'],
        prediction=decision['decision'],correct=decision['decision']==row['node_label'],
        elapsed_s=lmeta['elapsed_s']+vmeta['elapsed_s'],audit=str(folder.relative_to(args.output)),
        hard_negative=row['sample_id']==HARD)

def report(args,program,rows):
    rows=sorted(rows,key=lambda r:r['sample_id'])
    baselines={}
    for name,path in [('DINO_CLS',args.dataset/'rejection_audit/predictions.csv'),('RoMa',args.dataset/'roma/predictions.csv')]:
        data=list(csv.DictReader(path.open()))
        if name=='DINO_CLS':data=[r for r in data if r['feature']=='cls' and r['rule']=='similarity_and_margin']
        baselines[name]={r['sample_id']:r for r in data}
        for r in rows:r[name]=baselines[name][r['sample_id']]['prediction']
    with (args.output/'predictions.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Visual Decision Program — S1 / S2','',
        'Compile: only S1/S2 node descriptions and seed representative RGBs. All queries exclude the entire seed run. '
        'One frozen program; no test-image recompilation, added predicates, or threshold tuning.',
        'Labels are existing graph/matcher labels, not independent human ground truth. Only one held-out S2; no generalization claim.',
        '',f'Program SHA256: `{sha(args.output/"visual_program.json")}`','',
        f'Correct (UNKNOWN counts incorrect): {sum(r["correct"] for r in rows)}/{len(rows)}. '
        f'Coverage: {sum(r["prediction"]!="UNKNOWN" for r in rows)}/{len(rows)}.',
        '', '| Label | Count | Correct | S1 | S2 | UNKNOWN |','|---|---:|---:|---:|---:|---:|']
    for label in ('S1','S2'):
        subset=[r for r in rows if r['label']==label]
        lines.append(f'| {label} | {len(subset)} | {sum(r["correct"] for r in subset)} | '+ ' | '.join(str(sum(r['prediction']==p for r in subset)) for p in ('S1','S2','UNKNOWN'))+' |')
    hard=next(r for r in rows if r['hard_negative'])
    lines+=['',f'Hard negative `{HARD}`: **{hard["prediction"]}**, DINO CLS: {hard["DINO_CLS"]}, RoMa: {hard["RoMa"]}.',
        '', '## Frozen program','```json',json.dumps(program,ensure_ascii=False,indent=2),'```',
        '', '## Predicate audit','YES/NO/UNKNOWN counts per true label; association on this small set is descriptive, not causal importance.',
        '| Predicate | S1 YES/NO/UNKNOWN | S2 YES/NO/UNKNOWN | UNKNOWN total |','|---|---|---|---:|']
    results={r['sample_id']:json.loads((args.output/r['audit']/'predicate_results.json').read_text())['results'] for r in rows}
    for pred in program['predicates']:
        counts=[];unknown=0
        for label in ('S1','S2'):
            vals=[next(v['value'] for v in results[r['sample_id']] if v['predicate_id']==pred['id']) for r in rows if r['label']==label]
            counts.append('/'.join(str(vals.count(x)) for x in ('YES','NO','UNKNOWN')));unknown+=vals.count('UNKNOWN')
        lines.append(f'| {pred["id"]} | {counts[0]} | {counts[1]} | {unknown} |')
    lines+=['','## Query results','| Query | Label | Host | DINO CLS | RoMa | Seconds |','|---|---|---|---|---|---:|']
    for r in rows:lines.append(f'| [{r["sample_id"]}]({r["audit"]}/roi_overlay.png) | {r["label"]} | {r["prediction"]} | {r["DINO_CLS"]} | {r["RoMa"]} | {r["elapsed_s"]:.2f} |')
    lines+=['','## Baseline comparability',
        'Existing DINO/RoMa are frozen S0/S1/UNKNOWN open-set benchmarks with leave-one-run-out banks, not S1/S2 classifiers. '
        'Their S2 samples are OTHER. The table compares the same query IDs and especially false S1 acceptance; '
        'their aggregate S1/S2 accuracy is not an equivalent-task comparison. No baseline was retrained or retuned.',
        f'Mean visual-program query latency (localize + predicate execution): {sum(r["elapsed_s"] for r in rows)/len(rows):.2f}s. '
        'Includes model latency; compile excluded. Existing RoMa pair-inference timing excludes model startup; not directly comparable.',
        '', 'Audit contains original RGB, proposed ROI overlay, exact crops actually attached to executor, '
        'requests/responses, predicate results and host decision. ROI attachment does not prove internal visual attention. '
        'Execution receives no node labels, node descriptions, representative images or decision rules.']
    lines += ['', '## 同图基线对照',
        '| 方法 | S1 正确接受 | S1 UNKNOWN | S2 hard negative |',
        '|---|---:|---:|---|']
    for name,key in [('Visual Program','prediction'),('DINO CLS + margin','DINO_CLS'),('RoMa','RoMa')]:
        known=[r for r in rows if r['label']=='S1']
        lines.append(f'| {name} | {sum(r[key]=="S1" for r in known)}/{len(known)} | {sum(r[key]=="UNKNOWN" for r in known)} | {hard[key]} |')
    durations=sorted(r['elapsed_s'] for r in rows)
    compile_response=json.loads((args.output/'compile/request/response.json').read_text())
    lines += ['', f'Compile 模型耗时：{compile_response["elapsed_s"]:.2f}s；query 中位耗时：{durations[len(durations)//2]:.2f}s。',
        '首次预编译请求未内联 JSON schema，返回格式验证失败，已归档在 compile/invalid_schema_attempt_001。'
        '补齐 schema 与原图的左右约定后重新编译；冻结之前未提供任何测试图。测试阶段没有改动程序。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    save(args.output/'summary.json',dict(rows=rows,program_sha256=sha(args.output/'visual_program.json')))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',type=Path,default=Path('reports/local_node_classifier_e000'))
    ap.add_argument('--graph',type=Path,default=Path('runs/graph_policy_full02/policy_graph_v2.json'))
    ap.add_argument('--output',type=Path,default=Path('reports/visual_program_S1_S2'))
    ap.add_argument('--model',default='gpt-6-astra');ap.add_argument('--timeout',type=int,default=900)
    ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args();descriptions,reps,queries=prepare(args)
    print(f'COMPILE representatives={len(reps)}, held_out={len(queries)}',flush=True)
    program=compile_program(args,descriptions,reps)
    print('FROZEN '+json.dumps(program),flush=True)
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(execute_query,args,program,r):r for r in queries}
        for future in as_completed(jobs):
            result=future.result();rows.append(result);print('QUERY '+json.dumps(result),flush=True)
    report(args,program,rows);print('DONE '+str(args.output/'report.md'),flush=True)

if __name__=='__main__':main()
