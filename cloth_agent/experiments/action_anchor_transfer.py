"""Offline e001 reference-conditioned pixel transfer. No GPT or robot imports/calls."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/action_anchor_transfer'
ASSETS = Path('/mnt/newssd/sja/clothAgent_real/model_assets/local_node_benchmark')
# Set from the reference alone, BEFORE inference. Fixed-camera ROIs, no query GT.
ROIS = {'grasp': [320,390,570,660], 'target': [490,390,760,650]}
POLICY = dict(dino_size=392, descriptor_min_cosine=.6, descriptor_nonlocal_margin=.015,
              descriptor_cycle_max_cells=2., roma_min_confidence=.5, roma_cycle_max_px=20.,
              rois=ROIS, roi_source='Manually fixed once using only the reference image; no query anchor/annotation used',
              threshold_source='Fixed engineering gates before model inference; uncalibrated, not tuned on query errors')


def load(p): return json.loads(Path(p).read_text())
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inside(p,b): return bool(np.isfinite(p).all() and b[0]<=p[0]<b[2] and b[1]<=p[1]<b[3])
def norm_point(p,box): return [(p[i]-box[i]+.5)/(box[i+2]-box[i])*2-1 for i in (0,1)]
def pixel_point(p,box): return [(p[i]+1)/2*(box[i+2]-box[i])-.5+box[i] for i in (0,1)]

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    base=ROOT/'runs/graph_policy_full02'; rows=[];excluded=[];seen=set()
    for f in sorted(base.glob('policy_replay_*/step_*/dispatch_intent.json')):
        d=load(f)
        if d.get('edge_id')!='e001':continue
        g=load(f.parent/'grounding.json');n=load(f.parent/'node_match.json');h=load(f.parent/'host_selection.json')
        image=Path(d['before']['path']); im=Image.open(image).convert('RGB'); obs=load(image.parent/'observation.json')
        reason=None
        if not(n['result']['status']=='MATCH' and n['result']['node_id']=='S1' and n['result']['confidence']>=.8 and h['node_id']=='S1'):reason='source not confirmed S1'
        if n['current_rgb']['sha256']!=sha(image) or d['before']['sha256']!=sha(image):raise ValueError('source hash mismatch')
        if g['status']!='READY' or g['edge_id']!='e001':reason='grounding not ready'
        if im.size!=(1280,720) or obs['camera_serial']!='233622079809':reason='camera/resolution mismatch'
        pixhash=hashlib.sha256(im.tobytes()).hexdigest()
        if pixhash in seen:reason='duplicate RGB pixels'
        if reason:excluded.append(dict(path=str(f),reason=reason));continue
        seen.add(pixhash)
        slots=sorted(g['slots'],key=lambda s:s['operation_index'])
        assert slots[0]['operation_index']==0 and slots[-1]['operation_index']==4
        ex=load(f.parent/'execution.json')
        nxt=f.parent.parent/f'step_{int(f.parent.name[5:])+1:03d}'/'node_match.json'
        after=load(nxt) if nxt.exists() else None
        rows.append(dict(sample_id='anchor_'+pixhash[:16],run=f.parent.parent.name,image=str(image),sha256=sha(image),pixel_sha256=pixhash,
                         Pa=slots[0]['pixel_xy'],Pb=slots[-1]['pixel_xy'],source_node='S1',edge_id='e001',
                         source_match=n['result'],grounding_file=str(f.parent/'grounding.json'),dispatch_file=str(f),
                         execution_file=str(f.parent/'execution.json'),execution_status=ex['status'],after_match_file=str(nxt) if after else None,
                         after_match=after['result'] if after else None,camera_serial=obs['camera_serial'],joints_deg=obs['joints_deg']))
    # Success = actual executed e001 and observed expected S2, not mere command completion.
    ref=next(r for r in rows if r['execution_status']=='EXECUTED' and r['after_match'] and r['after_match']['status']=='MATCH' and r['after_match']['node_id']=='S2')
    queries=[r for r in rows if r['run']!=ref['run']]
    m=dict(edge='e001',source_node='S1',reference=ref,queries=queries,excluded=excluded,policy=POLICY,
           reference_selection='Earliest chronological executed e001 with subsequent MATCH S2; chosen without correspondence results',
           success_scope='Expected graph intermediate S2 (curled inward cuff), NOT perfect flattening or final fold success',
           historical_grounding='Noisy archived GPT coordinates; not manual ground truth or exact material-point identity',
           dataset_scope='All eligible e001 dispatch records in original graph_policy_full02 replay bank; other online schemas not inferred as this edge',
           graph_sha256=sha(base/'policy_graph_v2.json'),gpt_calls=0,robot_calls=0,training=False)
    path=OUT/'manifest.json'
    if path.exists():assert load(path)==m,'Frozen manifest changed'
    else:save(path,m)
    # Separate inference input: never carries query Pa/Pb, grounding, descriptions or frame anchors.
    inference=dict(reference_image=ref['image'],Pa_ref=ref['Pa'],Pb_ref=ref['Pb'],queries=[{k:r[k] for k in ['sample_id','image','sha256']} for r in queries],policy=POLICY)
    save(OUT/'inference_inputs.json',inference)
    canvas=Image.new('RGB',(1280,400*math.ceil(len(rows)/2)),'white');d=ImageDraw.Draw(canvas)
    for i,r in enumerate(rows):
        x=(i%2)*640;y=(i//2)*400
        canvas.paste(Image.open(r['image']).resize((640,360)),(x,y+40))
        d.text((x+5,y+5),('REFERENCE' if r==ref else 'QUERY')+' '+r['run'],fill='black')
    canvas.save(OUT/'dataset_montage.jpg')
    print('Prepared',len(queries),'queries; reference',ref['run'],ref['Pa'],ref['Pb'],flush=True)


def sample_map(chw,coords):
    import torch
    import torch.nn.functional as F
    grid=torch.as_tensor(coords,dtype=torch.float32,device=chw.device).reshape(1,-1,1,2)
    return F.grid_sample(chw[None],grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,:,0].T


def descriptor_transfer(ref,query,anchor,box):
    """Local descriptor NN plus nonlocal uniqueness and backward consistency. No labels."""
    import torch
    import torch.nn.functional as F
    ref=F.normalize(ref.float(),dim=0);query=F.normalize(query.float(),dim=0)
    h,w=query.shape[1:]
    vec=F.normalize(sample_map(ref,[norm_point(anchor,box)])[0],dim=0)
    sim=(query*vec[:,None,None]).sum(0)
    idx=int(sim.argmax());iy,ix=divmod(idx,w);best=float(sim[iy,ix])
    yy,xx=torch.meshgrid(torch.arange(h,device=sim.device),torch.arange(w,device=sim.device),indexing='ij')
    far=(xx-ix)**2+(yy-iy)**2>4
    margin=best-float(sim[far].max()) if far.any() else 0.
    back=(ref*query[:,iy,ix,None,None]).sum(0);bi=int(back.argmax());by,bx=divmod(bi,ref.shape[2])
    an=norm_point(anchor,box);ax=(an[0]+1)*ref.shape[2]/2-.5;ay=(an[1]+1)*ref.shape[1]/2-.5
    cycle=math.hypot(bx-ax,by-ay)
    pred=pixel_point([(ix+.5)/w*2-1,(iy+.5)/h*2-1],box)
    reasons=[]
    if best<POLICY['descriptor_min_cosine']:reasons.append('LOW_COSINE')
    if margin<POLICY['descriptor_nonlocal_margin']:reasons.append('AMBIGUOUS_NONLOCAL_MATCH')
    if cycle>POLICY['descriptor_cycle_max_cells']:reasons.append('CYCLE_INCONSISTENT')
    if not inside(pred,box):reasons.append('OUTSIDE_ROI')
    return dict(status='FAILURE' if reasons else 'OK',point=None if reasons else pred,confidence=best,
                cosine=best,nonlocal_margin=margin,cycle_cells=cycle,failure_reasons=reasons)


def roma_transfer(warp,conf,anchor,box):
    # RoMa symmetric halves: [Axy, B(A)xy] and [A(B)xy, Bxy].
    if warp.ndim==4:
        if warp.shape[0]!=1 or conf.shape[0]!=1:raise ValueError('Expected a single image pair')
        warp=warp[0];conf=conf[0]
    if warp.ndim!=3 or conf.ndim!=2 or tuple(warp.shape[:2])!=tuple(conf.shape):
        raise ValueError('Invalid symmetric warp/confidence shapes')
    half=warp.shape[1]//2
    ab=warp[:,:half,2:4].permute(2,0,1);ba=warp[:,half:,:2].permute(2,0,1)
    an=norm_point(anchor,box);bn=sample_map(ab,[an])[0]
    cf=float(sample_map(conf[:,:half][None],[an])[0,0])
    back=sample_map(ba,bn[None])[0];cb=float(sample_map(conf[:,half:][None],bn[None])[0,0])
    pred=pixel_point(bn.tolist(),box);backpx=pixel_point(back.tolist(),box);cycle=math.dist(backpx,anchor)
    confidence=min(cf,cb);reasons=[]
    if confidence<POLICY['roma_min_confidence']:reasons.append('LOW_DENSE_CONFIDENCE')
    if cycle>POLICY['roma_cycle_max_px']:reasons.append('CYCLE_INCONSISTENT')
    if not inside(pred,box):reasons.append('OUTSIDE_ROI')
    return dict(status='FAILURE' if reasons else 'OK',point=None if reasons else pred,confidence=confidence,
                forward_confidence=cf,reverse_confidence=cb,cycle_px=cycle,failure_reasons=reasons)


class Dino:
    def __init__(self):
        import torch
        from transformers import AutoModel,AutoImageProcessor
        path=ASSETS/'dinov2-small'
        self.model=AutoModel.from_pretrained(path,local_files_only=True).cuda().eval().requires_grad_(False)
        p=AutoImageProcessor.from_pretrained(path,local_files_only=True,use_fast=False)
        self.mean=torch.tensor(p.image_mean,device='cuda')[:,None,None];self.std=torch.tensor(p.image_std,device='cuda')[:,None,None]
        assert self.model.config.model_type=='dinov2' and self.model.config.patch_size==14
        self.meta=dict(model='facebook/dinov2-small',weights={p.name:sha(p) for p in path.glob('*.safetensors')},crop_resize=POLICY['dino_size'])
    def feature(self,im):
        import torch
        n=POLICY['dino_size'];im=im.resize((n,n),Image.Resampling.BICUBIC)
        x=torch.from_numpy(np.asarray(im,dtype=np.float32).copy()/255).permute(2,0,1).cuda()
        t=self.model(pixel_values=((x-self.mean)/self.std)[None]).last_hidden_state[0,1:]
        return t.T.reshape(-1,n//14,n//14)


class Matcha:
    def __init__(self):
        for p in ['matcha_deps','roma_deps']:sys.path.insert(0,str(ASSETS/p))
        from .local_node_matcha_e000 import load_model,prepare_tensor
        weights=ASSETS/'matcha_pretrained.pth'
        if not weights.is_file():raise FileNotFoundError('Official MATCHA fusion weights absent; official README Drive link HTTP 404 (2026-10-14).')
        self.model,meta=load_model(ASSETS/'matcha_source',weights,ASSETS/'matcha_sd21',ASSETS)
        self.prepare=prepare_tensor;self.meta=dict(model='nv-dvl/matcha semantic dense',weights_sha256=sha(weights),loading=meta)
    def feature(self,im):
        import torch
        torch.manual_seed(0)
        return self.model(self.prepare(im,[0,0,*im.size]),semantic_mode=True)[0]


class Roma:
    def __init__(self):
        sys.path.insert(0,str(ASSETS/'roma_deps'))
        import torch
        from romatch import roma_outdoor
        a=ASSETS/'roma_weights/roma_outdoor.pth';b=ASSETS/'roma_weights/dinov2_vitl14_pretrain.pth'
        self.model=roma_outdoor(device='cuda',weights=torch.load(a,map_location='cpu',weights_only=True),dinov2_weights=torch.load(b,map_location='cpu',weights_only=True),use_custom_corr=False).eval().requires_grad_(False)
        assert self.model.symmetric
        self.meta=dict(model='romatch 0.1.2 roma_outdoor',weights_sha256=sha(a),backbone_sha256=sha(b),coarse=560,upsample=864,symmetric=True)
    def pair(self,a,b):return self.model.match(a,b,device='cuda')


def run(method):
    inp=load(OUT/'inference_inputs.json');out=OUT/method;out.mkdir(exist_ok=True)
    if (out/'predictions.json').exists():raise FileExistsError('Existing results preserved; no implicit re-run')
    assert inp['policy']==POLICY
    import torch
    torch.set_num_threads(4);torch.manual_seed(0)
    start=time.perf_counter()
    try:model={'dinov2':Dino,'roma':Roma,'matcha':Matcha}[method]()
    except FileNotFoundError as e:
        if method!='matcha':raise
        save(out/'status.json',dict(status='BLOCKED_MISSING_CHECKPOINT',reason=str(e),predictions=None,latency_ms=None,model_calls=0))
        return
    save(out/'model.json',model.meta)
    ref=Image.open(inp['reference_image']).convert('RGB');refs={k:ref.crop(b) for k,b in ROIS.items()}
    with torch.inference_mode():
        if method=='roma':model.pair(refs['grasp'],refs['grasp']);ref_feats=None
        else:ref_feats={k:model.feature(im) for k,im in refs.items()}
        torch.cuda.synchronize();save(out/'setup.json',dict(load_reference_warmup_s=time.perf_counter()-start,reference_cached=method!='roma'))
        rows=[]
        for q in inp['queries']:
            assert sha(q['image'])==q['sha256']
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            im=Image.open(q['image']).convert('RGB');details={}
            for k,anchor in [('grasp',inp['Pa_ref']),('target',inp['Pb_ref'])]:
                patch=im.crop(ROIS[k])
                if method=='roma':
                    warp,confidence=model.pair(refs[k],patch)
                    details[k]=roma_transfer(warp,confidence,anchor,ROIS[k]);del warp,confidence
                else:details[k]=descriptor_transfer(ref_feats[k],model.feature(patch),anchor,ROIS[k])
            torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
            row=dict(sample_id=q['sample_id'],method=method,status='OK' if all(v['status']=='OK' for v in details.values()) else 'FAILURE',
                     Pa_pred=details['grasp']['point'],Pb_pred=details['target']['point'],confidence={k:v['confidence'] for k,v in details.items()},
                     latency_ms=elapsed,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,details=details)
            rows.append(row);save(out/'predictions.partial.json',rows);print(method,q['sample_id'],row['status'],round(elapsed),'ms',flush=True)
    save(out/'predictions.json',rows);save(out/'status.json',dict(status='COMPLETE',n=len(rows)))


def draw(ref,q,row,path):
    ims=[Image.open(r['image']).convert('RGB') for r in [ref,q]]
    canvas=Image.new('RGB',(2560,820),'white');d=ImageDraw.Draw(canvas)
    for i,im in enumerate(ims):canvas.paste(im,(1280*i,80))
    colors={'grasp':'#ff3232','target':'#00cfff'}
    def mark(p,offset,color,kind,label):
        if p is None:return
        x,y=p[0]+offset,p[1]+80
        if kind=='circle':d.ellipse([x-9,y-9,x+9,y+9],outline=color,width=4)
        else:d.line((x-9,y-9,x+9,y+9),fill=color,width=4);d.line((x-9,y+9,x+9,y-9),fill=color,width=4)
        d.text((x+12,y-15),label,fill=color,stroke_width=1,stroke_fill='black')
    for k,a,b in [('grasp','Pa','Pa_pred'),('target','Pb','Pb_pred')]:
        color=colors[k];mark(ref[a],0,color,'circle',a+'_ref');mark(row[b],1280,color,'circle',a+'_pred');mark(q[a],1280,'#ffff00','cross',a+'_historical_GPT')
        for offset in [0,1280]:
            box=ROIS[k];d.rectangle([box[0]+offset,box[1]+80,box[2]+offset,box[3]+80],outline=color,width=2)
    d.text((15,10),f"REFERENCE e001 S1 {ref['run']} | red=grasp cyan=target; rectangles=fixed search ROI",fill='black')
    d.text((1290,10),f"{row['method']} {q['sample_id']} {row['status']} | circles=prediction; yellow X=historical GPT (noisy)",fill='black')
    d.text((1290,35),f"grasp error={row.get('grasp_error_px')} target error={row.get('target_error_px')}",fill='black')
    if 'details' in row:
        d.text((1290,55),' | '.join(k+':'+','.join(v['failure_reasons']) for k,v in row['details'].items()),fill='black')
    canvas.save(path)


def selfcheck():
    """Identity positive control, diagnostic only; never changes gates or query results."""
    import gc
    import torch
    torch.set_num_threads(4);torch.manual_seed(0)
    inp=load(OUT/'inference_inputs.json');ref=Image.open(inp['reference_image']).convert('RGB');results={}
    with torch.inference_mode():
        for name,cls in [('dinov2',Dino),('roma',Roma)]:
            model=cls();details={}
            for k,a in [('grasp',inp['Pa_ref']),('target',inp['Pb_ref'])]:
                patch=ref.crop(ROIS[k])
                if name=='roma':
                    warp,c=model.pair(patch,patch);r=roma_transfer(warp,c,a,ROIS[k]);del warp,c
                else:
                    f=model.feature(patch);r=descriptor_transfer(f,f,a,ROIS[k]);del f
                r['identity_error_px']=math.dist(a,r['point']) if r['point'] is not None else None
                details[k]=r
            results[name]=details;del model;gc.collect();torch.cuda.empty_cache()
    save(OUT/'identity_control.json',results)
    print(json.dumps(results,indent=2))


def report():
    m=load(OUT/'manifest.json');lookup={r['sample_id']:r for r in m['queries']};allrows=[];summaries={}
    overlays=OUT/'overlays';overlays.mkdir(exist_ok=True)
    for method in ['dinov2','roma','matcha']:
        p=OUT/method/'predictions.json'
        if not p.exists():summaries[method]=load(OUT/method/'status.json');continue
        rows=load(p)
        for row in rows:
            q=lookup[row['sample_id']];row.update(Pa_gt=q['Pa'],Pb_gt=q['Pb'])
            for k,key,gt in [('grasp','Pa_pred','Pa_gt'),('target','Pb_pred','Pb_gt')]:row[k+'_error_px']=math.dist(row[key],row[gt]) if row[key] is not None else None
            draw(m['reference'],q,row,overlays/f"{method}_{row['sample_id']}.png")
        summary=dict(status='COMPLETE',n=len(rows),joint_failure_count=sum(r['status']!='OK' for r in rows),failure_rate=sum(r['status']!='OK' for r in rows)/len(rows),mean_latency_ms=float(np.mean([r['latency_ms'] for r in rows])),median_latency_ms=float(np.median([r['latency_ms'] for r in rows])))
        for k in ['grasp','target']:
            errors=[r[k+'_error_px'] for r in rows if r[k+'_error_px'] is not None]
            summary[k]=dict(n_accepted=len(errors),failures=len(rows)-len(errors),median_error_px=float(np.median(errors)) if errors else None,p95_error_px=float(np.percentile(errors,95)) if errors else None)
        summaries[method]=summary;allrows+=rows
    save(OUT/'predictions.json',allrows);save(OUT/'summary.json',summaries)
    keys=['sample_id','method','status','Pa_pred','Pb_pred','Pa_gt','Pb_gt','grasp_error_px','target_error_px','confidence','latency_ms']
    with (OUT/'predictions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in allrows:w.writerow({k:json.dumps(r[k]) if isinstance(r[k],(list,dict)) else r[k] for k in keys})
    lines=['# Reference-conditioned action anchor transfer — e001','',
           'Offline only: no GPT calls, no training, no robot execution or robot-runtime code changes. Original image coordinates, 1280×720.','',
           '## Data and provenance',
           f"Reference: `{m['reference']['run']}`; Pa_ref={m['reference']['Pa']}, Pb_ref={m['reference']['Pb']}. {len(lookup)} queries from separate runs, all archived MATCH S1 + READY e001 grounding. Source hashes, commands, next-state matching and raw source paths in manifest.json.",
           'Reference selected chronologically from successful historical transitions, not optimized on query errors. Success means executed S1→S2 (curled inward sleeve); does not mean perfect flattening. The original seed e001 is marked PROGRESS; historical reference avoids reconstructing pixel anchors from XYZ.',
           'Historical Pa_gt/Pb_gt are noisy GPT choices, not manual truth. A correspondence transports a material/appearance point, whereas GPT can choose another valid action point; deviation conflates these effects. All queries retained regardless of their subsequent execution outcome. Same garment, camera, orientation and narrow pose range; not a general deployment benchmark.','',
           '## Frozen local matching protocol',
           'Separate sleeve/cuff ROI [320,390,570,660] and upper-torso ROI [490,390,760,650], drawn from reference only. No query semantic labels, GPT frame anchors or historical query Pa/Pb passed into inference; these are joined only for scoring. Fixed camera/placement assumption is explicit. No whole-image global nearest-neighbor fallback.',
           'DINOv2-S/14: each local ROI resized to 392²; bilinear reference descriptor sampling, cosine nearest patch in the corresponding query ROI, nonlocal uniqueness gap ≥0.015, cosine ≥0.6, backward cycle ≤2 grid cells. Predictions are patch centers.',
           'RoMa Outdoor: two independent local dense matches, default 560/864 resolutions, symmetric warp. Bilinear forward-anchor transport with minimum forward/backward confidence ≥0.5 and backward cycle ≤20 original pixels. MATCHA adapter uses official semantic dense descriptors with the same local descriptor transfer checks. Gates are predeclared engineering heuristics, not calibrated probabilities or guarantees; no error-based tuning.',
           'Unavailable anchors are null with explicit FAILURE, separate per-anchor status/reason. Median/P95 below are conditional on accepted anchors, including the available anchor of a joint FAILURE. Joint failure means either anchor failed; missing model is N/A, not a measured 100% correspondence failure.','',
           '| Method | grasp median px | target median px | P95 grasp / target px | failure rate | median latency ms |',
           '|---|---:|---:|---:|---:|---:|']
    def fmt(v):return 'N/A' if v is None else f'{v:.2f}'
    for name,s in summaries.items():
        if s['status']!='COMPLETE':lines.append(f'| {name} | N/A | N/A | N/A | NOT RUN | N/A |');continue
        lines.append(f"| {name} | {fmt(s['grasp']['median_error_px'])} | {fmt(s['target']['median_error_px'])} | {fmt(s['grasp']['p95_error_px'])} / {fmt(s['target']['p95_error_px'])} | {s['joint_failure_count']}/{s['n']} | {s['median_latency_ms']:.1f} |")
    lines += ['', '## Findings in this run',
              'Neither tested backend produced an accepted Pa/Pb pair on any of the seven queries under the frozen gates. This is an abstention result for these specific implementations and gates, not proof that every raw correspondence was geometrically wrong or that all possible DINO/RoMa configurations fail.',
              'Grasp: DINO accepted 2/7; RoMa accepted 1/7. RoMa has lower conditional error on its one accepted point, but there is no statistically credible stability winner at this coverage. Target: DINO accepted only 1/7, RoMa 0/7. Neither is stable; target is the principal coverage bottleneck.',
              'DINO cosine remains approximately 0.86–0.92 despite frequent nonlocal ambiguity and reverse-cycle failure. RoMa target confidence is approximately 4.6e-9–1.5e-5 even though reverse-cycle error is small. Cycle consistency alone does not establish a unique, meaningful cloth point.',
              'The lack of distinctive texture around the receiving torso anchor is a plausible cause, based on the reference and overlays; this is a hypothesis, not an ablation-proven cause. Historical target choices can also vary within a valid receiving region.',
              'Cases where globally similar S1 garments do not yield a stable local anchor:',
              '- anchor_c9411e9b7be750b2: DINO accepts the grasp, yet it is about 41.4 px from historical GPT; target is rejected. See overlays/dinov2_anchor_c9411e9b7be750b2.png.',
              '- anchor_38c53de7bce4f24e: DINO accepts the target but it differs by 41.3 px from historical GPT, while grasp is rejected. This establishes disagreement, not human-verified material-point error.',
              '- anchor_94934bb37c5c9e84: RoMa accepts grasp (18.3 px deviation) but rejects target with confidence about 3.3e-7. Overall silhouette similarity is insufficient for confident local transport.',
              'Current replacement verdict: no evidence that this setup can replace GPT grounding. Local correspondence may be a proposal source, but accepted-pair coverage is zero here. No gates were relaxed after looking at errors. MATCHA is blocked, so the requested three-backend empirical comparison remains incomplete.']
    if (OUT/'identity_control.json').exists():
        lines += ['', '## Identity positive control (reference matched to itself)',
                  '**Critical limitation:** RoMa rejects the identical-image torso target (confidence 0.000487); DINO rejects the identical-image grasp due to nonlocal ambiguity (margin 0.00676). RoMa identity grasp is accepted with 0.011 px error; DINO identity target is accepted with 10.86 px error, showing the patch-localization floor. Thus 100% joint failure is substantially a gate/texture ambiguity result and must NOT be read as 100% geometrically wrong transport. Confidence thresholds are uncalibrated for cloth, and the current experiment cannot rank raw geometry quality for rejected points.',
                  'Diagnostic performed after the fixed query run; no thresholds or predictions modified. These gates can reject even identical input, not only difficult cross-run deformations. Future reliability calibration would require separate validation data, not tuning these seven query errors.',
                  '```json',json.dumps(load(OUT/'identity_control.json'),indent=2),'```']
    lines += ['', 'Latency: synchronized image decode + two local transfers; excludes model initialization and warmup. DINO/MATCHA reference features cached; RoMa computes both local pairs. No confidence comparability across backends. Seven pairs are too few for a stable tail estimate.',
              '', '## MATCHA availability',
              'Official fusion checkpoint is missing locally. On 2026-10-14 the [official README](https://github.com/nv-dvl/matcha) still links to a Drive checkpoint returning HTTP 404. No random weights or substitute matcher used. Interface exists but real MATCHA inference remains unverified; its comparison is incomplete.',
              '', '## Per-anchor diagnostics','```json',json.dumps(summaries,indent=2),'```','',
              '## Reproduce','```bash','/home/sja/miniconda3/envs/molmo/bin/python -m cloth_agent.experiments.action_anchor_transfer prepare',
              '# Run once per method; existing predictions are preserved and re-run refused.',
              '/home/sja/miniconda3/envs/molmo/bin/python -m cloth_agent.experiments.action_anchor_transfer dinov2',
              '/home/sja/miniconda3/envs/molmo/bin/python -m cloth_agent.experiments.action_anchor_transfer roma',
              '/home/sja/miniconda3/envs/molmo/bin/python -m cloth_agent.experiments.action_anchor_transfer matcha',
              '/home/sja/miniconda3/envs/molmo/bin/python -m cloth_agent.experiments.action_anchor_transfer report','```']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    for method in ['dinov2','roma']:
        thumbs=Image.new('RGB',(1280,len(lookup)*410),'white')
        for i,q in enumerate(m['queries']):
            f=overlays/f"{method}_{q['sample_id']}.png"
            if f.exists():thumbs.paste(Image.open(f).resize((1280,410)),(0,i*410))
        thumbs.save(OUT/(method+'_montage.jpg'))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','dinov2','roma','matcha','report','selfcheck']);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='report':report()
    elif a.stage=='selfcheck':selfcheck()
    else:run(a.stage)
