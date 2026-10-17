"""Offline measured geometry -> frozen predicates/rules; no robot imports."""
import csv,json,math,shutil,statistics,time,itertools,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from PIL import Image,ImageDraw
import jsonschema
from .visual_decision_program import obj,arr,call,save,sha,decide
from .visual_state_grounding_e001 import POINT,BOX,CONF,archived_anchors
from .discovery_model import CodexMultimodal
OUT=Path('reports/unified_visual_representation_e001')
BASE=Path('reports/visual_state_grounding_e001')
PIDS=['left_inward_flap','right_extended','right_rounded_cuff','torso_open']
POLY={'anyOf':[arr(arr(dict(type='integer',minimum=0),minItems=2,maxItems=2),minItems=3,maxItems=12),dict(type='null')]}
def entity(**kw):return obj(dict(**kw,confidence=CONF))
SCHEMA=obj(dict(
 garment=entity(collar_anchor=POINT,hem_anchor=POINT),
 left_sleeve=entity(bbox=BOX,polygon=POLY,distal_anchor=POINT,root_anchor=POINT),
 right_sleeve=entity(bbox=BOX,polygon=POLY,distal_anchor=POINT,root_anchor=POINT),
 right_cuff=entity(bbox=BOX,grasp_anchor=POINT,opening_visible=dict(enum=['YES','NO','UNKNOWN'])),
 torso=entity(bbox=BOX,left_boundary=arr(POINT,minItems=3,maxItems=3),right_boundary=arr(POINT,minItems=3,maxItems=3),target_anchor=POINT,
 print_visible_fraction={'anyOf':[CONF,dict(type='null')]},surface_ridges=arr(obj(dict(start=POINT,end=POINT,confidence=CONF)),maxItems=8))))
PROMPT='''Measure only visible garment entities, pixel geometry and local physical appearance in this single RGB.
Do not classify a state, name a folding stage, output predicates, recommend actions, or infer hidden anatomy.
Mentally rotate collar above hem without mirroring; left/right are viewer sides in that orientation.
All coordinates refer to original unrotated image pixels. Bboxes [x0,y0,x1,y1].
For sleeves give visible fabric polygon, bbox, distal endpoint and sleeve/torso junction root.
For right cuff give only distal cuff fabric bbox, a grasp anchor on exposed fabric (not hole/background),
and whether a visible aperture between cuff fabric layers is present. UNKNOWN if aperture cannot be resolved.
Torso bbox excludes sleeves. Each side boundary has exactly three visible contour points ordered upper torso,
mid torso, hem. Do not reconstruct a hidden unfolded silhouette: use null when uncertain.
Target anchor is on visible upper torso fabric on garment-right, inward of right sleeve root, avoiding collar
hole and fabric from the other sleeve. Collar and hem anchors mark their respective middle points.
Print visible fraction is the estimated fraction of the garment's central graphic visible, null if not assessable.
Surface ridges are conspicuous interior doubled-fabric/fold edges: endpoints in pixels, omit ordinary fine wrinkles.
Record their geometry without calling them panels or assigning folding status. Null unlocatable coordinates;
confidence describes visibility/localization reliability, not a class. No explanations or extra keys.
'''

def load(p):return json.loads(Path(p).read_text())
def extract(row,split):
    folder=OUT/split/row['sample_id'];folder.mkdir(parents=True,exist_ok=True)
    image=folder/'original.png'
    if not image.exists():shutil.copy2(row['image'],image)
    assert sha(image)==row['file_sha256']
    v,m=call(CodexMultimodal('gpt-6-astra',900),PROMPT+'\nImage dimensions: '+str(Image.open(image).size),[image],SCHEMA,folder/'request')
    save(folder/'variables.json',v)
    print(split,row['sample_id'],round(m['elapsed_s'],2),flush=True)
    return row

def features(v,size):
    jsonschema.validate(v,SCHEMA);w,h=size
    def valid(p):return p is not None and 0<=p[0]<w and 0<=p[1]<h
    g=v['garment'];t=v['torso'];l=v['left_sleeve'];r=v['right_sleeve'];c=v['right_cuff']
    required=[g['collar_anchor'],g['hem_anchor'],*t['left_boundary'],*t['right_boundary']]
    if not all(valid(p) for p in required):return dict(valid=False)
    collar,hem=required[:2];axis=[hem[i]-collar[i] for i in (0,1)];length=math.hypot(*axis)
    if length<1:return dict(valid=False)
    down=[a/length for a in axis];right=[down[1],-down[0]]
    project=lambda p:sum((p[i]-collar[i])*right[i] for i in (0,1))
    widths=[project(b)-project(a) for a,b in zip(t['left_boundary'],t['right_boundary'])]
    if min(widths)<=1:return dict(valid=False)
    width=widths[0];left=project(t['left_boundary'][0]);rightedge=project(t['right_boundary'][0])
    def pointfeat(p,fn):return fn(p) if valid(p) else None
    def overlap(poly):
        if not poly or not all(valid(p) for p in poly):return None
        a=Image.new('1',(w,h));b=Image.new('1',(w,h));ImageDraw.Draw(a).polygon([tuple(p) for p in poly],fill=1)
        box=t['bbox']
        if box is None or not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):return None
        ImageDraw.Draw(b).rectangle(box,fill=1)
        import numpy as np
        aa=np.asarray(a);bb=np.asarray(b)
        return float((aa&bb).sum()/aa.sum()) if aa.sum() else None
    b=c['bbox'];aspect=None
    if b and 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h:aspect=min(b[2]-b[0],b[3]-b[1])/max(b[2]-b[0],b[3]-b[1])
    ridges=[]
    for line in t['surface_ridges']:
        if line['confidence']<.7 or not valid(line['start']) or not valid(line['end']):continue
        delta=[line['end'][i]-line['start'][i] for i in (0,1)]
        along=abs(sum(delta[i]*down[i] for i in (0,1)));across=abs(sum(delta[i]*right[i] for i in (0,1)))
        if along>2*across:ridges.append(along/length)
    return dict(valid=True,confidence={k:v[k]['confidence'] for k in v},
        left_inset=pointfeat(l['distal_anchor'],lambda p:(project(p)-left)/width),left_overlap=overlap(l['polygon']),
        right_extension=pointfeat(r['distal_anchor'],lambda p:(project(p)-rightedge)/width),
        cuff_aspect=aspect,cuff_root_distance=math.dist(c['grasp_anchor'],r['root_anchor'])/width if valid(c['grasp_anchor']) and valid(r['root_anchor']) else None,
        opening=c['opening_visible'],hem_width_ratio=widths[2]/max(widths[:2]),print_fraction=t['print_visible_fraction'],long_ridge=max(ridges,default=0))

def derive(f,p):
    result={k:'UNKNOWN' for k in PIDS}
    if not f['valid']:return result
    base=min(f['confidence']['garment'],f['confidence']['torso'])>=p['confidence']
    def ok(keys,entity):return base and f['confidence'][entity]>=p['confidence'] and all(f[k] is not None for k in keys)
    yn=lambda b:'YES' if b else 'NO'
    if ok(['left_inset','left_overlap'],'left_sleeve'):result[PIDS[0]]=yn(f['left_inset']>=p['inset'] and f['left_overlap']>=p['overlap'])
    if ok(['right_extension'],'right_sleeve'):result[PIDS[1]]=yn(f['right_extension']>=p['extension'])
    if ok(['cuff_aspect','cuff_root_distance'],'right_cuff') and f['confidence']['right_sleeve']>=p['confidence'] :
        geometry=f['cuff_aspect']>=p['aspect'] and f['cuff_root_distance']<=p['near']
        result[PIDS[2]]='NO' if not geometry or f['opening']=='NO' else 'YES' if f['opening']=='YES' else 'UNKNOWN'
    if ok(['hem_width_ratio','print_fraction'],'torso'):result[PIDS[3]]=yn(f['hem_width_ratio']>=p['hem'] and f['print_fraction']>=p['print'] and f['long_ridge']<p['ridge'])
    return result

def host(v,p,program,size):
    f=features(v,size);pred=derive(f,p);node=decide(program,[dict(predicate_id=k,value=x) for k,x in pred.items()])
    c=v['right_cuff'];t=v['torso'];w,h=size
    def inside(a,b):return a is not None and b is not None and 0<=b[0]<=a[0]<b[2]<=w and 0<=b[1]<=a[1]<b[3]<=h
    valid=inside(c['grasp_anchor'],c['bbox']) and inside(t['target_anchor'],t['bbox']) and min(c['confidence'],t['confidence'])>=p['confidence']
    grounding=dict(edge_id='e001',grasp_anchor_px=c['grasp_anchor'],target_anchor_px=t['target_anchor'],physical_execution=False,status='SEMANTIC_ONLY') if node['decision']=='S1' and valid else None
    return dict(features=f,predicates=pred,node=node,grounding=grounding,anchors_valid=valid,physical_execution=False)

GRID=dict(confidence=[.65,.8],inset=[0,.08,.16],overlap=[.25,.45,.65],extension=[.08,.16,.24,.32],aspect=[.25,.4,.55],near=[.45,.7,1.0],hem=[.65,.8,.95],print=[.5,.75,.9],ridge=[.35,.5,.7])
DEFAULT=dict(confidence=.8,inset=.08,overlap=.45,extension=.16,aspect=.4,near=.7,hem=.8,print=.75,ridge=.5)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    prior=load(BASE/'manifest.json');test=prior['queries'];train=[r for r in csv.DictReader(open('reports/local_node_classifier_e000/samples.csv')) if r['run_id']=='seed']
    assert not ({r['pixel_sha256'] for r in train}&{r['pixel_sha256'] for r in test})
    for r in train+test:assert sha(r['image'])==r['file_sha256']
    if not (OUT/'visual_program.json').exists():shutil.copy2(BASE/'visual_program.json',OUT/'visual_program.json')
    assert sha(OUT/'visual_program.json')==prior['program_sha256']
    manifest=dict(train=train,test=test,model='gpt-6-astra',program_sha256=prior['program_sha256'],schema=SCHEMA,prompt=PROMPT,grid=GRID,default=DEFAULT,
        calibration='train only; minimize S1/S2 predicate errors then all train node errors; deterministic distance-to-default tie break',
        train_predicate_labels={'S1':['YES','YES','NO','YES'],'S2':['YES','NO','YES','YES']},
        limitation='Only one train S1 and one train S2; test previously reported in A/B but not used to fit C; existing matcher labels not human truth')
    if (OUT/'manifest.json').exists():assert load(OUT/'manifest.json')==manifest
    else:save(OUT/'manifest.json',manifest)
    return train,test,load(OUT/'visual_program.json')

def batch(rows,split):
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(extract,r,split) for r in rows]):f.result()

def calibrate(train,program):
    path=OUT/'host_calibration.json'
    if path.exists():
        frozen=load(OUT/'freeze.json');assert sha(path)==frozen['calibration_sha256'] and sha(__file__)==frozen['implementation_sha256'];return load(path)['thresholds']
    records=[]
    for r in train:
        d=OUT/'train'/r['sample_id'];records.append((r,features(load(d/'variables.json'),Image.open(d/'original.png').size)))
    # The predicates are separable, so independently minimize train predicate error first.
    groups=[('left_inward_flap',['confidence','inset','overlap']),('right_extended',['confidence','extension']),('right_rounded_cuff',['confidence','aspect','near']),('torso_open',['confidence','hem','print','ridge'])]
    # Exhaustive joint grid is modest; evaluate cached four predicate tables, not repeated image geometry.
    tables={}
    for pid,keys in groups:
        tab={}
        for vals in itertools.product(*(GRID[k] for k in keys)):
            p=DEFAULT|dict(zip(keys,vals));tab[vals]=[derive(f,p)[pid] for _,f in records]
        tables[pid]=(keys,tab)
    best=None;chosen=None
    labels=load(OUT/'manifest.json')['train_predicate_labels']
    for vals in itertools.product(*GRID.values()):
        p=dict(zip(GRID,vals));outputs={pid:table[tuple(p[k] for k in keys)] for pid,(keys,table) in tables.items()}
        pe=ne=0
        for i,(r,_) in enumerate(records):
            label=r['node_label'];pred=[outputs[k][i] for k in PIDS]
            if label in ('S1','S2'):pe+=sum(a!=b for a,b in zip(pred,labels[label]))
            s1=pred==['YES','YES','NO','YES'];s2=pred==['YES','NO','YES','YES'];decision='S1' if s1 else 'S2' if s2 else 'UNKNOWN'
            ne+=decision!=(label if label in ('S1','S2') else 'UNKNOWN')
        dist=sum(abs(GRID[k].index(p[k])-GRID[k].index(DEFAULT[k])) for k in GRID)
        score=(pe,ne,dist)
        if best is None or score<best:best=score;chosen=p
    save(path,dict(thresholds=chosen,train_objective=list(best),grid=GRID,selected_unix_s=time.time(),train_count=len(train)))
    save(OUT/'freeze.json',dict(calibration_sha256=sha(path),implementation_sha256=sha(__file__),program_sha256=sha(OUT/'visual_program.json'),manifest_sha256=sha(OUT/'manifest.json'),frozen_unix_s=time.time()))
    print('FROZEN',chosen,best,flush=True);return chosen

def overlay(folder,v,h):
    im=Image.open(folder/'original.png').convert('RGB');w,height=im.size;canvas=Image.new('RGB',(w,height+100),'white');canvas.paste(im);d=ImageDraw.Draw(canvas)
    for key,col in [('left_sleeve','lime'),('right_sleeve','orange'),('right_cuff','red'),('torso','cyan'),('garment','yellow')]:
        e=v[key];b=e.get('bbox')
        if b and b[0]<b[2] and b[1]<b[3]:d.rectangle(b,outline=col,width=3)
        for k,a in e.items():
            if 'anchor' in k and a:
                x,y=a;d.ellipse((x-4,y-4,x+4,y+4),fill=col);d.text((x+5,y),key+':'+k,fill=col)
        for name in ['left_boundary','right_boundary']:
            pts=e.get(name,[])
            if pts and all(pts):d.line([tuple(p) for p in pts],fill=col,width=3)
    d.text((10,height+8),'Host: '+h['node']['decision']+' | e001 semantic grounding: '+str(h['grounding'] is not None),fill='black')
    for i,(k,val) in enumerate(h['predicates'].items()):d.text((10+(i%2)*570,height+33+(i//2)*25),k+': '+val,fill='black')
    canvas.save(folder/'overlay.png')

def report(test,p,program):
    rows=[];old=archived_anchors();a={r['sample_id']:r for r in load('reports/visual_program_S1_S2/audit/results.json')};b={r['sample_id']:r for r in load(BASE/'summary.json')['rows']}
    for r in test:
        folder=OUT/'test'/r['sample_id'];v=load(folder/'variables.json');h=host(v,p,program,Image.open(folder/'original.png').size);save(folder/'host_result.json',h);overlay(folder,v,h)
        prior=old.get(r['file_sha256']);c=v['right_cuff']['grasp_anchor'];t=v['torso']['target_anchor']
        row=dict(sample_id=r['sample_id'],label=r['node_label'],prediction=h['node']['decision'],A=a[r['sample_id']]['prediction'],B=b[r['sample_id']]['prediction'],grounding=h['grounding'] is not None,anchors_valid=h['anchors_valid'],elapsed_s=load(folder/'request/response.json')['elapsed_s'],**h['predicates'],grasp_delta_px=math.dist(c,prior['grasp']) if prior and c else None,target_delta_px=math.dist(t,prior['target']) if prior and t else None)
        rows.append(row)
    with (OUT/'predictions.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    known=[r for r in rows if r['label']!='OTHER'];others=[r for r in rows if r['label']=='OTHER']
    metrics=dict(total=len(rows),known_total=len(known),accuracy_comparison={k:sum(r[k]==r['label'] for r in known) for k in ['A','B','prediction']},unknown=sum(r['prediction']=='UNKNOWN' for r in rows),other_false_accept=sum(r['prediction']!='UNKNOWN' for r in others),other_total=len(others),mean_latency_s=statistics.mean(r['elapsed_s'] for r in rows),median_latency_s=statistics.median(r['elapsed_s'] for r in rows),groundings=sum(r['grounding'] for r in rows),hard_negative=next(r for r in rows if r['sample_id']=='obs_7f66d64e53de5d54'))
    for key in ['grasp_delta_px','target_delta_px']:
        ds=[r[key] for r in rows if r['label']=='S1' and r[key] is not None];metrics[key]=dict(count=len(ds),median=statistics.median(ds) if ds else None,max=max(ds) if ds else None)
    save(OUT/'summary.json',dict(metrics=metrics,rows=rows));print(json.dumps(metrics),flush=True)
    lines=['# Unified Visual Representation — S1/S2 + e001','','每张图一次 GPT-6 调用，仅输入当前 RGB 和低层测量 schema；不输入节点、冻结 predicates/rules、测试标签或历史预测。Host 从实体几何派生 predicates，再执行原冻结 rules。无机器人执行。',
    'Train 为完整 seed run 13 张，test 为与 A/B 相同的 29 张，像素哈希无交集。仅 train 校准，测试前冻结派生实现、阈值、schema 和 manifest。train 中 S1/S2 各仅 1 张，校准泛化证据有限。既有标签来自 graph/matcher，不是独立人工真值。',
    '这是此前已公开结果的数据集上的新表示实验，不是从未被研究者见过的盲测；没有用 test 输出或标签优化阈值。','','## Host 派生',
    '- 左袖：distal 在左 torso boundary 内的宽度归一化距离，以及 sleeve polygon 与 torso bbox 的重叠比例（包围盒近似）。',
    '- 右袖：distal 超出右 torso boundary 的宽度归一化距离。',
    '- 袖口：opening_visible、bbox 短长边比、grasp 到 sleeve root 的归一化距离。',
    '- torso：hem 宽度比、print visible fraction，以及与 collar→hem 轴平行的长 ridge。',
    '缺失坐标或低置信度传播 UNKNOWN。这些是原谓词的几何近似，不是语义等价的证明；袖口 bbox 比例受拍摄方向影响，普通折痕与折叠边缘也可能混淆。',
    '```json',json.dumps(load(OUT/'host_calibration.json'),ensure_ascii=False,indent=2),'```','','## Held-out 结果','```json',json.dumps(metrics,ensure_ascii=False,indent=2),'```',
    '历史 grounding 偏差仅衡量与旧 GPT 锚点的一致性，不是人工真值误差。一次/图未测重复调用稳定性。',
    '', '| Sample | Label | A | B | C | Grounding | Seconds |','|---|---|---|---|---|---|---:|']
    for r in rows:lines.append(f"| [{r['sample_id']}](test/{r['sample_id']}/overlay.png) | {r['label']} | {r['A']} | {r['B']} | {r['prediction']} | {r['grounding']} | {r['elapsed_s']:.2f} |")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['train','test','all'],default='all');args=ap.parse_args()
    train,test,program=prepare()
    if args.stage in ['train','all']:batch(train,'train')
    p=calibrate(train,program)
    if args.stage in ['test','all']:batch(test,'test');report(test,p,program)
if __name__=='__main__':main()
