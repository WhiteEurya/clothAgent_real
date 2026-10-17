"""One isolated visual call per query: fixed predicates plus e001 pixel entities.
No runtime, controller, depth sampler, IK, collision planner or action dispatcher.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import json
import math
from pathlib import Path
import shutil
import statistics
from PIL import Image,ImageDraw,ImageFont
from .visual_decision_program import obj,arr,call,decide,sha,save,validate_program
from .discovery_model import CodexMultimodal

SOURCE=Path('reports/visual_program_S1_S2')
OUT=Path('reports/visual_state_grounding_e001')
PRED_IDS=['left_inward_flap','right_extended','right_rounded_cuff','torso_open']
POINT={'anyOf':[arr(dict(type='integer',minimum=0),minItems=2,maxItems=2),dict(type='null')]}
BOX={'anyOf':[arr(dict(type='integer',minimum=0),minItems=4,maxItems=4),dict(type='null')]}
CONF=dict(type='number',minimum=0,maximum=1)
SCHEMA=obj(dict(entities=obj(dict(
    right_cuff=obj(dict(visible=dict(type='boolean'),bbox=BOX,anchor_px=POINT,confidence=CONF)),
    right_sleeve_root=obj(dict(anchor_px=POINT,confidence=CONF)),
    right_upper_torso=obj(dict(bbox=BOX,target_anchor_px=POINT,confidence=CONF)))),
    predicates=obj({k:dict(enum=['YES','NO','UNKNOWN']) for k in PRED_IDS})))

PROMPT='''Extract visual variables from CURRENT RGB, never choose, plan or execute an action.
Image 0 is the current query. Images 1/2 are the fixed S1/S2 representatives, only visual references.
You are supplied the frozen visual program and fixed e001 action template. Do not change either.
In this ONE response, evaluate every frozen predicate independently and locate each semantic entity
in image 0. Do not output a node label, action decision, motion code, trajectory, yaw, depth or XYZ.
Only the Host applies rules and decides whether the fixed e001 template is applicable.
The current image may belong to neither reference state. Do not force-fit it to an expected outcome.
Use the program's garment-frame convention: collar above hem without mirroring; left/right are
viewer sides in that orientation. Coordinates always refer to the unrotated ORIGINAL query pixels.
Cuff bbox encloses the distal right cuff fabric, not the entire sleeve. Grasp anchor is visibly
exposed fabric just inside the cuff edge, never the dark opening of a curled cuff or background.
Sleeve root anchor marks the visible right sleeve-to-torso junction. Torso bbox is the right upper
receiving torso region; its target anchor is on visible torso fabric inward of the sleeve root,
on the garment-right side, not on the already folded opposite sleeve or in the collar opening.
Return entities even when the predicates do not support e001. If a semantic location cannot be
established from visible evidence, use null coordinates and low confidence; cuff visible=false
requires bbox=null and anchor_px=null. Do not fabricate anchors on occluded anatomy.
Bboxes are [x0,y0,x1,y1]; anchors [x,y]. Give only the exact JSON schema, no chain-of-thought.
Predicate values and entities must describe the SAME current image. The Host, not you, will
handle pixel-to-XYZ, depth and all physical validation later; this experiment is offline only.
'''

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    freeze=json.loads((SOURCE/'program_freeze.json').read_text())
    assert sha(SOURCE/'visual_program.json')==freeze['sha256']
    program=json.loads((SOURCE/'visual_program.json').read_text());validate_program(program)
    assert [p['id'] for p in program['predicates']]==PRED_IDS
    graph=json.loads(Path('runs/graph_policy_full02/policy_graph_v2.json').read_text())
    edge=next(e for e in graph['edges'] if e['edge_id']=='e001')
    template={k:edge[k] for k in ['edge_id','source_node','target_node','action_family','action_template']}
    assert edge['source_node']=='S1' and edge['replayable']
    srcmanifest=json.loads((SOURCE/'manifest.json').read_text())
    reps=[]
    for index,rep in enumerate(srcmanifest['representatives']):
        assert sha(rep['path'])==rep['sha256']
        dest=OUT/'inputs'/f'representative_{index}.png';dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(rep['path'],dest)
        assert sha(dest)==rep['sha256'];reps.append(dest)
    queries=json.loads((SOURCE/'audit/manifest.json').read_text())['queries']
    for row in queries:assert sha(row['image'])==row['file_sha256']
    manifest=dict(model=srcmanifest['model'],program_sha256=freeze['sha256'],queries=queries,
        e001=template,representatives=[dict(path=str(p.resolve()),sha256=sha(p)) for p in reps],
        grounding_confidence_threshold=0.8,threshold_set_before_queries=True,
        policy='e001 semantic grounding only if Host S1 and all entities valid with confidence >= 0.8; no physical approval',
        missing_entities='null coordinates; invisible cuff requires null box and anchor',
        calls='Exactly one isolated visual invocation per query, no repair/second grounding call',
        stability_scope='Across held-out observations and archived same-image e001 anchors; no repeated-query variance experiment')
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    save(OUT/'manifest.json',manifest)
    for name,value in [('visual_program.json',program),('e001_action_template.json',template)]:
        f=OUT/name
        if not f.exists():save(f,value);f.chmod(0o444)
        assert json.loads(f.read_text())==value
    assert sha(OUT/'visual_program.json')==freeze['sha256']
    return program,template,reps,queries,manifest

def host(value,program,size):
    w,h=size
    import jsonschema
    jsonschema.validate(value,SCHEMA)
    decision=decide(program,[dict(predicate_id=k,value=v) for k,v in value['predicates'].items()])
    es=value['entities'];issues=[]
    def point(p):return p is not None and 0<=p[0]<w and 0<=p[1]<h
    def box(b):return b is not None and 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h
    def inside(p,b):return point(p) and box(b) and b[0]<=p[0]<b[2] and b[1]<=p[1]<b[3]
    cuff=es['right_cuff'];torso=es['right_upper_torso'];root=es['right_sleeve_root']
    checks=dict(cuff_visible=cuff['visible'],cuff_anchor_in_bbox=inside(cuff['anchor_px'],cuff['bbox']),
        torso_target_in_bbox=inside(torso['target_anchor_px'],torso['bbox']),root_in_image=point(root['anchor_px']),
        confidence_sufficient=all(e['confidence']>=.8 for e in es.values()))
    for k,v in checks.items():
        if not v:issues.append(k)
    if not cuff['visible'] and (cuff['bbox'] is not None or cuff['anchor_px'] is not None):issues.append('invisible_cuff_has_coordinates')
    # Relation diagnostic only: this projection is not a robot-safety or anatomy proof.
    t=None
    if point(cuff['anchor_px']) and point(root['anchor_px']) and point(torso['target_anchor_px']):
        a=cuff['anchor_px'];b=torso['target_anchor_px'];r=root['anchor_px'];v=[b[i]-a[i] for i in (0,1)];norm=sum(x*x for x in v)
        if norm:t=sum((r[i]-a[i])*v[i] for i in (0,1))/norm
    allowed=decision['decision']=='S1'
    grounding=None
    if allowed and not issues:
        grounding=dict(edge_id='e001',grasp_anchor_px=cuff['anchor_px'],sleeve_root_anchor_px=root['anchor_px'],
            target_anchor_px=torso['target_anchor_px'],units='original_rgb_pixels',
            status='SEMANTIC_ONLY_PENDING_HOST_PHYSICAL_VALIDATION',physical_execution=False)
    return dict(node=decision,eligible_source_node=allowed,grounding=grounding,checks=checks,issues=issues,
        root_projection_fraction=t,root_between_cuff_and_target=None if t is None else 0<=t<=1,
        physical_execution=False)

def draw(folder,value,decision):
    im=Image.open(folder/'original.png').convert('RGB');w,h=im.size
    canvas=Image.new('RGB',(w,h+115),'white');canvas.paste(im,(0,0));d=ImageDraw.Draw(canvas)
    e=value['entities']
    for key,color,anchor in [('right_cuff','red','anchor_px'),('right_sleeve_root','yellow','anchor_px'),('right_upper_torso','cyan','target_anchor_px')]:
        ent=e[key];b=ent.get('bbox');p=ent[anchor]
        if b and b[2]>b[0] and b[3]>b[1]:d.rectangle(b,outline=color,width=3)
        if p:
            x,y=p;d.ellipse((x-5,y-5,x+5,y+5),fill=color,outline='black');d.text((x+8,y),key,fill=color)
    d.text((12,h+8),f'Host node: {decision["node"]["decision"]} | e001 semantic grounding: {decision["grounding"] is not None} | NOT ROBOT APPROVAL',fill='black')
    for i,(k,v) in enumerate(value['predicates'].items()):d.text((12+(i%2)*600,h+35+(i//2)*25),f'{k}: {v}',fill='black')
    canvas.save(folder/'overlay.png')

def query(row,program,template,reps,manifest):
    folder=OUT/'audit_images'/row['sample_id'];folder.mkdir(parents=True,exist_ok=True)
    original=folder/'original.png';Image.open(row['image']).convert('RGB').save(original)
    size=Image.open(original).size
    prompt=PROMPT+json.dumps(dict(query_size=list(size),visual_program=program,e001_action_template=template,
        representative_image_order=['S1','S2']))
    try:
        value,meta=call(CodexMultimodal(manifest['model'],900),prompt,[original,*reps],SCHEMA,folder/'request')
        decision=host(value,program,size);save(folder/'variables.json',value);save(folder/'host_result.json',decision)
        draw(folder,value,decision)
        assert sha(SOURCE/'visual_program.json')==sha(OUT/'visual_program.json')==manifest['program_sha256']
        return dict(sample_id=row['sample_id'],label=row['node_label'],original_node=row['original_node_label'],
            prediction=decision['node']['decision'],semantic_grounding=decision['grounding'] is not None,
            elapsed_s=meta['elapsed_s'],model_calls=1,status='OK',audit=str(folder.relative_to(OUT)))
    except Exception as exc:
        save(folder/'error.json',dict(error=str(exc)))
        raise  # No second visual repair call.

def archived_anchors():
    result={}
    for run in Path('runs/graph_policy_full02').glob('policy_replay_*'):
        for f in run.glob('step_*/dispatch_intent.json'):
            d=json.loads(f.read_text());g=f.parent/'grounding.json'
            if d.get('edge_id')!='e001' or not g.exists() or not d.get('before'):continue
            data=json.loads(g.read_text());slots=data.get('slots',[])
            if not slots:continue
            result[d['before']['sha256']]=dict(grasp=slots[0]['pixel_xy'],target=slots[-1]['pixel_xy'],path=str(g),
                garment_frame=data.get('garment_frame'))
    return result

def summarize(rows,queries):
    byid={r['sample_id']:r for r in queries};old=archived_anchors()
    baseline={r['sample_id']:r for r in json.loads((SOURCE/'audit/results.json').read_text())}
    distance=lambda a,b:math.dist(a,b) if a and b else None
    for r in rows:
        folder=OUT/r['audit'];value=json.loads((folder/'variables.json').read_text());h=json.loads((folder/'host_result.json').read_text())
        r.update(previous_prediction=baseline[r['sample_id']]['prediction'],
            cuff_visible=value['entities']['right_cuff']['visible'],
            cuff_confidence=value['entities']['right_cuff']['confidence'],
            torso_confidence=value['entities']['right_upper_torso']['confidence'],
            root_between=h['root_between_cuff_and_target'],issues=';'.join(h['issues']))
        prior=old.get(byid[r['sample_id']]['file_sha256'])
        r['prior_grasp_delta_px']=distance(value['entities']['right_cuff']['anchor_px'],prior['grasp']) if prior else None
        r['prior_target_delta_px']=distance(value['entities']['right_upper_torso']['target_anchor_px'],prior['target']) if prior else None
        if prior:save(folder/'archived_grounding_comparison.json',dict(prior=prior,
            grasp_delta_px=r['prior_grasp_delta_px'],target_delta_px=r['prior_target_delta_px'],
            limitation='Archived GPT-selected anchors are not human or physical ground truth.'))
    with (OUT/'predictions.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(sorted(rows,key=lambda r:r['sample_id']))
    known=[r for r in rows if r['label'] in ('S1','S2')];other=[r for r in rows if r['label']=='OTHER'];s1=[r for r in rows if r['label']=='S1']
    stats=dict(total=len(rows),calls=sum(r['model_calls'] for r in rows),
        known_correct=sum(r['prediction']==r['label'] for r in known),known_total=len(known),
        previous_known_correct=sum(r['previous_prediction']==r['label'] for r in known),
        unknown_total=sum(r['prediction']=='UNKNOWN' for r in rows),
        other_false_accept=sum(r['prediction']!='UNKNOWN' for r in other),other_total=len(other),
        s1_semantic_grounding=sum(r['semantic_grounding'] for r in s1),
        s1_root_between=sum(r['root_between'] is True for r in s1),
        prediction_agreement=sum(r['prediction']==r['previous_prediction'] for r in rows),
        mean_elapsed_s=statistics.mean(r['elapsed_s'] for r in rows),median_elapsed_s=statistics.median(r['elapsed_s'] for r in rows))
    for key in ['prior_grasp_delta_px','prior_target_delta_px']:
        ds=[r[key] for r in s1 if r[key] is not None]
        stats[key]=dict(count=len(ds),median=statistics.median(ds) if ds else None,max=max(ds) if ds else None)
    save(OUT/'summary.json',dict(metrics=stats,rows=rows))
    lines=['# Visual State + Grounding Program — e001','',
        '每图恰好一次隔离 GPT-6 视觉调用，同时输出四个冻结 predicates 和 cuff/root/torso entities。Host 按冻结规则计算节点；仅 S1 且实体有效、confidence ≥0.8 时生成 semantic grounding。阈值在执行前固定。未读取深度、未调用机器人或规划器。',
        '输入是同一组 29 张 held-out（10 S1、1 S2、18 OTHER）及原 S1/S2 representatives。规则未修改；节点标签不是独立人工真值。',
        '', '## 自动统计','```json',json.dumps(stats,indent=2),'```',
        '', '## 逐图结果','| Image | Label | Host | e001 grounding | Cuff confidence | Torso confidence | Time(s) |','|---|---|---|---|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:r['sample_id']):lines.append(f'| [{r["sample_id"]}]({r["audit"]}/overlay.png) | {r["original_node"]} | {r["prediction"]} | {r["semantic_grounding"]} | {r["cuff_confidence"]:.2f} | {r["torso_confidence"]:.2f} | {r["elapsed_s"]:.2f} |')
    lines+=['','## 定位与一致性的边界',
        '自动检查：坐标在图内、anchor 在对应 bbox 内、cuff 可见性、confidence，和 sleeve-root 在 cuff→target 方向上的投影（仅诊断，非安全约束）。这些不证明语义锚点正确，需复核 overlay。',
        '历史同图 e001 锚点的像素偏差仅描述与另一次 GPT grounding 的一致性，不是标注误差；不同的合法布料抓点可能相距数十像素。',
        '本实验每图只调用一次，未做重复查询方差测试，因此“稳定”只指跨样本定位与旧结果对照。',
        '', '## 是否能合并两次调用',
        '在语义输出接口上，state predicates 与 cuff/torso 锚点可来自同一次调用，Host 再顺序计算节点和 e001 eligibility。',
        '这不是完整 replay 动作实例化：旧 ground_edge 还生成 garment_frame、5 个 primitive slots、路径中间锚点及朝向等；本实验的三个实体不替代这些输出，更不替代 depth/IK/collision。',
        '原 Visual Decision Program 的两次调用是 ROI 定位 + predicates，不能混称为 node_match + edge_grounding。历史 replay 延迟对照另行匹配同图请求，不把两种基线混用。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(stats),flush=True)

def main():
    program,template,reps,queries,manifest=prepare();rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(query,r,program,template,reps,manifest) for r in queries]
        for f in as_completed(jobs):
            r=f.result();rows.append(r);print(json.dumps(r),flush=True)
    summarize(rows,queries)

if __name__=='__main__':main()
