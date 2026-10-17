"""Standalone two-ROI GPT ablation. Imports existing registration read-only.
Never modifies Replay/minimal modules, executes a robot, or uses learned correspondence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import jsonschema
import numpy as np
from PIL import Image,ImageDraw
from .graph_minimal_grounding import register_frame
from .discovery_model import CodexMultimodal

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/roi_minimal_grounding_ablation'
OLD=ROOT/'reports/minimal_edge_grounding'
ROIS={'grasp':[320,390,570,660],'target':[490,390,760,650]}
UV={'anyOf':[{'type':'array','items':{'type':'number','minimum':0,'maximum':1},'minItems':2,'maxItems':2},{'type':'null'}]}
SCHEMA={'type':'object','properties':{'grasp_uv':UV,'target_uv':UV,'confidence':{'type':'number','minimum':0,'maximum':1}},'required':['grasp_uv','target_uv','confidence'],'additionalProperties':False}
TEXT='''Image A (first): select a grasp point on exposed right cuff / distal sleeve fabric.
Image B (second): select a target point on upper-torso fabric suitable for receiving the sleeve.
Return only grasp_uv, target_uv, confidence. Each [u,v] is normalized to its own image, top-left [0,0], bottom-right [1,1]. Use null if unresolved. No explanation.
'''
PROMPT=TEXT+json.dumps(SCHEMA,separators=(',',':'))

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def transform_roi(box,affine,size):
    x0,y0,x1,y1=box
    # Transform continuous crop boundaries; retain all of the rotated rectangle in an AABB.
    corners=np.array([[x0,y0,1],[x1,y0,1],[x1,y1,1],[x0,y1,1]],float)@np.array(affine).T
    bounds=[int(np.floor(corners[:,0].min())),int(np.floor(corners[:,1].min())),int(np.ceil(corners[:,0].max())),int(np.ceil(corners[:,1].max()))]
    a,b,c,d=bounds
    if not np.isfinite(corners).all() or a<0 or b<0 or c>size[0] or d>size[1] or min(c-a,d-b)<16:raise ValueError('Transformed ROI outside image or too small; no clipping/fallback')
    return dict(reference_xyxy=box,polygon_full_px=corners.tolist(),crop_xyxy=bounds,crop_size=[c-a,d-b])

def map_uv(uv,box):
    if uv is None:return None
    if len(uv)!=2 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=1 for v in uv):raise ValueError('invalid normalized coordinates')
    x0,y0,x1,y1=box
    # Exact pixel-center convention; no x1/y1 off-by-one at u/v=1.
    return [x0+uv[0]*(x1-x0-1),y0+uv[1]*(y1-y0-1)]

def validate(value):
    jsonschema.validate(value,SCHEMA)
    if not math.isfinite(value['confidence']):raise ValueError('nonfinite confidence')
    for k in ['grasp_uv','target_uv']:map_uv(value[k],[0,0,100,100])

def input_files():
    files=[OLD/n for n in ['manifest.json','host_templates.json','results.json','summary.json']]
    files += [ROOT/'reports/action_anchor_transfer/manifest.json']
    files += [ROOT/'cloth_agent/experiments'/n for n in ['graph_policy_replay.py','graph_minimal_grounding.py','minimal_grounding_ablation.py']]
    old=read(OLD/'manifest.json')
    files += [Path(q['image']) for q in old['queries']]
    template=read(OLD/'host_templates.json')['edges']['e001'];files.append(Path(template['reference_image']))
    return {str(f):sha(f) for f in files}

def verify():
    m=read(OUT/'manifest.json')
    for f,h in m['immutable_inputs'].items():
        if sha(f)!=h:raise ValueError('Frozen input changed: '+f)
    return m

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    old=read(OLD/'manifest.json');template=read(OLD/'host_templates.json')['edges']['e001'];anchor=read(ROOT/'reports/action_anchor_transfer/manifest.json')
    assert len(old['queries'])==7 and [q['sample_id'] for q in old['queries']]==[q['sample_id'] for q in anchor['queries']]
    m=dict(model=old['model'],reference=anchor['reference'],queries=old['queries'],reference_rois=ROIS,
           immutable_inputs=input_files(),prompt=PROMPT,schema=SCHEMA,registration='Unchanged graph_minimal_grounding.register_frame',
           roi_protocol='Reference-only fixed rectangles, reused from prior anchor-transfer experiment; same for every query. Affine-transformed corners -> floor/ceil enclosing crop. Out-of-image registration/crops fail; no full-image GPT fallback.',
           coordinate_mapping='x=x0+u*(width-1), y=y0+v*(height-1); preserve float full-image coordinates',
           calls='One uncached invocation per registered query per condition; no format repair or retry; two workers.',
           controls='Optional roi_2x uses identical ROI content and identical prompt/schema, bicubic 2x size; no full RGB supplied to GPT.',
           thresholds='Existing registration thresholds untouched; no post-result ROI/threshold adjustments; confidence reported, not calibrated or used to alter results.')
    p=OUT/'manifest.json'
    if p.exists():assert read(p)==m,'Protocol changed'
    else:save(p,m)
    # Freeze the protocol BEFORE registration or any new model outputs.
    save(OUT/'protocol_sha256.json',{'sha256':sha(p)})
    for q in m['queries']:
        folder=OUT/'samples'/q['sample_id'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'roi.json').exists():continue
        image=Path(q['image']);obs=SimpleNamespace(image=image,meta=read(image.parent/'observation.json'))
        start=time.perf_counter()
        try:
            frame,audit=register_frame(template,obs)
            im=Image.open(image).convert('RGB');regions={k:transform_roi(b,audit['affine'],im.size) for k,b in ROIS.items()}
            for k,r in regions.items():
                crop=im.crop(r['crop_xyxy']);crop.save(folder/(k+'.png'))
                crop.resize((crop.width*2,crop.height*2),Image.Resampling.BICUBIC).save(folder/(k+'_2x.png'))
            result=dict(status='OK',registration=audit,regions=regions,elapsed_s=time.perf_counter()-start)
            overlay=im.copy();d=ImageDraw.Draw(overlay)
            for k,r in regions.items():
                color='red' if k=='grasp' else 'cyan';pts=[tuple(v) for v in r['polygon_full_px']]
                d.line(pts+[pts[0]],fill=color,width=3);d.rectangle(r['crop_xyxy'],outline=color,width=2)
                d.text(tuple(r['crop_xyxy'][:2]),k+' ROI',fill=color,stroke_width=1,stroke_fill='black')
            (OUT/'overlays').mkdir(exist_ok=True);overlay.save(OUT/'overlays'/f"{q['sample_id']}_roi.png")
        except Exception as exc:
            result=dict(status='FAILURE',failure_stage='ROI_REGISTRATION',error=f'{type(exc).__name__}: {exc}',elapsed_s=time.perf_counter()-start)
        save(folder/'roi.json',result)
    verify();return m

def query(q,condition,m):
    folder=OUT/'samples'/q['sample_id'];out=folder/condition;out.mkdir(exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Uncached experiment cannot overwrite/reuse a completed condition')
    roi=read(folder/'roi.json');r=dict(sample_id=q['sample_id'],condition=condition,status='FAILURE',grasp_px=None,target_px=None,confidence=None,
                                    latency_s=None,model_calls=0,registration_status=roi['status'],null_output=False,structured_failure=False,transport_failure=False)
    if roi['status']!='OK':r['failure_stage']='ROI_REGISTRATION';save(out/'result.json',r);return r
    suffix='' if condition=='roi' else '_2x';images=[folder/f'{k}{suffix}.png' for k in ['grasp','target']]
    save(out/'request.json',dict(model=m['model'],prompt=PROMPT,schema=SCHEMA,images=[dict(path=str(p),sha256=sha(p),size=list(Image.open(p).size)) for p in images],cache_disabled=True))
    start=time.perf_counter();r['model_calls']=1
    try:
        response=CodexMultimodal(m['model'],900).invoke(PROMPT,images,SCHEMA)
        r['latency_s']=time.perf_counter()-start
        save(out/'response.json',response)
        try:value=json.loads(response['text']);validate(value)
        except (ValueError,jsonschema.ValidationError) as exc:
            r.update(structured_failure=True,failure_stage='STRUCTURED_OUTPUT',error=str(exc))
        else:
            r.update(grasp_uv=value['grasp_uv'],target_uv=value['target_uv'],confidence=value['confidence'],
                     grasp_px=map_uv(value['grasp_uv'],roi['regions']['grasp']['crop_xyxy']),target_px=map_uv(value['target_uv'],roi['regions']['target']['crop_xyxy']))
            r['null_output']=value['grasp_uv'] is None or value['target_uv'] is None
            r['status']='NULL' if r['null_output'] else 'OK'
    except Exception as exc:
        r.update(latency_s=time.perf_counter()-start,transport_failure=True,failure_stage='TRANSPORT',error=f'{type(exc).__name__}: {exc}')
    save(out/'result.json',r);print(condition,q['sample_id'],r['status'],round(r['latency_s'] or 0,2),'s',flush=True);return r

def run(condition):
    m=prepare();jobs=[q for q in m['queries']]
    if any((OUT/'samples'/q['sample_id']/condition/'result.json').exists() for q in jobs):raise FileExistsError('Existing condition results preserved; no implicit cache or selective rerun')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(query,q,condition,m) for q in jobs]
        for f in as_completed(futures):f.result()
    verify();report()

def report():
    m=verify();old=read(OLD/'results.json');byold={(r['sample_id'],r['mode']):r for r in old};rows=[];summary={}
    for condition in ['roi','roi_2x']:
        for q in m['queries']:
            f=OUT/'samples'/q['sample_id']/condition/'result.json'
            if not f.exists():continue
            r=read(f)
            for mode in ['full','minimal']:
                for part,key in [('grasp','Pa_pred'),('target','Pb_pred')]:
                    gt=byold[(q['sample_id'],mode)][key];r[part+'_vs_'+mode+'_px']=math.dist(r[part+'_px'],gt) if r[part+'_px'] is not None and gt is not None else None
            rows.append(r)
        rs=[r for r in rows if r['condition']==condition]
        if not rs:continue
        times=[r['latency_s'] for r in rs if r['latency_s'] is not None]
        s=dict(n=len(rs),model_calls=sum(r['model_calls'] for r in rs),registration_failures=sum(r['registration_status']!='OK' for r in rs),
               nulls=sum(r['null_output'] for r in rs),structured_failures=sum(r['structured_failure'] for r in rs),transport_failures=sum(r['transport_failure'] for r in rs),
               mean_latency_s=statistics.mean(times) if times else None,median_latency_s=statistics.median(times) if times else None)
        for key in ['grasp_vs_full_px','target_vs_full_px','grasp_vs_minimal_px','target_vs_minimal_px']:
            v=[r[key] for r in rs if r[key] is not None];s['median_'+key]=statistics.median(v) if v else None
        usages=[read(OUT/'samples'/r['sample_id']/condition/'response.json').get('usage',{}) for r in rs if (OUT/'samples'/r['sample_id']/condition/'response.json').exists()]
        for key in ['input_tokens','cached_input_tokens','output_tokens']:
            v=[u[key] for u in usages if key in u]
            s['mean_reported_'+key]=statistics.mean(v) if v else None
        s['calls_with_provider_cached_input']=sum(u.get('cached_input_tokens',0)>0 for u in usages)
        summary[condition]=s
    summary['historical_baselines']=read(OLD/'summary.json')
    save(OUT/'summary.json',summary)
    if rows:
        keys=sorted(set().union(*(r.keys() for r in rows)))
        with (OUT/'predictions.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    for r in rows:
        q=next(q for q in m['queries'] if q['sample_id']==r['sample_id'])
        im=Image.open(q['image']).convert('RGB');canvas=Image.new('RGB',(1280,820),'white');canvas.paste(im,(0,100));d=ImageDraw.Draw(canvas)
        d.text((10,8),q['sample_id']+' '+r['condition']+' '+r['status']+' | filled=ROI; circles=previous minimal; X=previous full',fill='black')
        d.text((10,32),'RED grasp / CYAN target; coordinates compared with noisy GPT references, not human truth',fill='black')
        for part,key,color in [('grasp','Pa_pred','red'),('target','Pb_pred','cyan')]:
            for mode,kind in [('full','X'),('minimal','circle'),('roi','filled')]:
                point=r[part+'_px'] if mode=='roi' else byold[(q['sample_id'],mode)][key]
                if point is None:continue
                x,y=point;y+=100
                if kind=='X':d.line((x-9,y-9,x+9,y+9),fill=color,width=3);d.line((x-9,y+9,x+9,y-9),fill=color,width=3)
                else:d.ellipse((x-7,y-7,x+7,y+7),outline=color,fill=color if kind=='filled' else None,width=3)
        suffix='' if r['condition']=='roi' else '_2x'
        canvas.save(OUT/'overlays'/f"{q['sample_id']}_anchors{suffix}.png")
    thumbs=[]
    for q in m['queries']:
        f=OUT/'overlays'/f"{q['sample_id']}_anchors.png"
        if f.exists():thumbs.append(Image.open(f).resize((640,410)))
    if thumbs:
        canvas=Image.new('RGB',(1280,410*math.ceil(len(thumbs)/2)),'white')
        for i,im in enumerate(thumbs):canvas.paste(im,((i%2)*640,(i//2)*410))
        canvas.save(OUT/'montage.jpg')
    lines=['# Two-ROI minimal grounding ablation','',
           'Independent offline experiment; existing Graph Replay/minimal implementations and all historical inputs remain byte-identical (manifest hashes rechecked). No robot runtime, training, DINO/RoMa/Qwen or safety changes.','',
           '## Frozen setup',
           'Same seven e001/S1 query images, same successful reference and Pa_ref/Pb_ref. Reference-only ROIs: grasp [320,390,570,660], target [490,390,760,650], reused unchanged from the earlier reference-ROI specification. No per-query annotation. Existing Host SIFT/partial-affine garment registration transfers the rectangles; AABB crops enclose the transformed corners. Any registration/out-of-bounds failure skips GPT and is explicitly FAILURE.',
           'C receives exactly two unresized crops and a short fixed prompt. Only normalized grasp_uv/target_uv/confidence returned; null allowed. Full RGB, graph, node, garment frame, reference, history and geometry are never sent. Host maps UV via x0+u*(width-1), y0+v*(height-1). All per-query crops, affine matrices, polygons and overlays are saved.',
           'One fresh uncached call per registered image/condition, no repair/retry. Timing encloses the same CodexMultimodal.invoke transport used previously, including request preparation and response handling, excluding registration/cropping. Two query workers. Confidence is reported without a new acceptance threshold.','',
           '| Method | n | Mean GPT s | Median GPT s | Null | Structured failure | Registration failure |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for key,title in [('full','A archived fresh full'),('minimal','B archived fresh minimal full-image')]:
        s=summary['historical_baselines'][key];lines.append(f"| {title} | {s['n']} | {s['mean_latency_s']:.2f} | {s['median_latency_s']:.2f} | 0 | {s['structured_failures']}/{s['n']} | N/A |")
    for key in ['roi','roi_2x']:
        if key not in summary:continue
        s=summary[key];fmt=lambda v:'N/A' if v is None else f'{v:.2f}'
        lines.append(f"| {key} | {s['n']} | {fmt(s['mean_latency_s'])} | {fmt(s['median_latency_s'])} | {s['nulls']}/{s['model_calls']} calls | {s['structured_failures']}/{s['model_calls']} calls | {s['registration_failures']}/{s['n']} |")
    lines+=['','## Deviation from frozen previous outputs','Historical full/minimal outputs are noisy references, not human truth. Distances are original full-image pixels; no query coordinates used for crop generation.','```json',json.dumps({k:v for k,v in summary.items() if k!='historical_baselines'},indent=2),'```','',
            '## Attribution limits',
            'A/B are unchanged historical calls, not simultaneous randomized reruns. C changes image field of view/size, prompt text, coordinate format (pixels→UV), and one image→two. Therefore C versus B alone cannot attribute a latency change specifically to reduced visual input. Input pixel area is not a direct measurement of internal vision tokens or FLOPs; the serving stack may resize crops.',
            'If roi_2x is present, it uses the SAME two crop contents (bicubic 2x dimensions), byte-identical prompt/schema and UV convention, without providing any full RGB. This is only a resolution diagnostic: paired content is controlled, but call scheduling/time and internal image resizing remain confounders; it does not isolate loss of full-image context or establish a causal C-versus-B effect. No ROI or threshold is changed after results.',
            'Local prediction/cache reuse is disabled and every inference record is a real invocation. Provider usage nevertheless reports cached_input_tokens on some calls (also present in historical A/B); this transport exposes no provider-cache-off control. Those calls are retained in the timing statistics. Thus this is not a guaranteed cold-provider-cache benchmark.',
            '', '## Qualitative review','See semantic_review.json and overlays/*_anchors.png. Review is assistant visual inspection, not independently human-verified truth or physical success.',
            '', '## Reproduce','```bash','/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.roi_minimal_grounding_ablation prepare',
            '/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.roi_minimal_grounding_ablation run',
            '# Optional same-prompt two-crop resolution diagnostic:',
            '/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.roi_minimal_grounding_ablation control',
            '```','Existing completed conditions are refused, not read from an inference cache or overwritten. Use report to regenerate summaries without model calls.']
    if 'roi' in summary:
        c=summary['roi'];b=summary['historical_baselines']['minimal']
        lines += ['', '## Findings',
                  f"C median {c['median_latency_s']:.2f}s versus frozen B {b['median_latency_s']:.2f}s: no further observed median reduction in this run. C mean {c['mean_latency_s']:.2f}s; slower calls retained, not discarded as outliers.",
                  f"C versus previous fresh minimal median deviation: grasp {c['median_grasp_vs_minimal_px']:.2f}px, target {c['median_target_vs_minimal_px']:.2f}px. Coarse region plausibility does not imply preserving the same action geometry or fold outcome."]
    if 'roi_2x' in summary:
        d=summary['roi_2x'];lines += [f"Same-prompt 2x-crop control median {d['median_latency_s']:.2f}s, mean {d['mean_latency_s']:.2f}s. It provides a resolution-only comparison under sequential-call/service/cache confounders, not causal proof of the whole-image-to-ROI effect.",
                                    'Original-size and 2x medians are very close despite fourfold input pixel area. No convincing further speed benefit is observed, and there is no basis to attribute a benefit specifically to visual cropping rather than prompt/format changes. This experiment does not justify replacing the existing full-image minimal mode.']
    if (OUT/'semantic_review.json').exists():
        lines += ['', 'Qualitative review: no obvious opposite-sleeve, neck-opening or background points in the seven original-size ROI outputs. Grasp remains within distal cuff/sleeve fabric. Targets remain on the upper torso but shift away from earlier points, often toward the torso interior/print. This is assistant inspection, not human annotation or validated folding success.']
    if (OUT/'tests.log').exists():lines += ['', '## Tests','```',(OUT/'tests.log').read_text().strip(),'```']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','control','report']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='report':report()
    else:run('roi' if a.stage=='run' else 'roi_2x')
