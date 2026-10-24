"""Two independent dense feature evaluations; no GPT or robot imports.

Discovery uses no annotation; precision waits for returned-candidate reviews.
Localization uses only operator-confirmed experience groups and human start clicks.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .local_dense_proposal_probe import read,save,sha,pixel_hash,resize_mapping,protocol,CHECKPOINT,DEMO_SOURCE

METHODS=('A','B','C')
NMS_PX=56
HIT_PX=32
MAX_PER_IMAGE=3


def font(size=20):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def csv_out(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def arrow(im,start,target):
    im=im.copy();d=ImageDraw.Draw(im);p=np.array(start,float);q=np.array(target,float);v=q-p
    norm=np.linalg.norm(v)
    if norm>0:
        v=v/norm;u=np.array([-v[1],v[0]])
        d.line([tuple(p),tuple(q)],fill='yellow',width=4)
        d.polygon([tuple(q),tuple(q-15*v+7*u),tuple(q-15*v-7*u)],fill='yellow')
    x,y=p;d.ellipse((x-7,y-7,x+7,y+7),fill='red',outline='white',width=2)
    return im


def step_card(before,after,pa,target,dest,title):
    board=Image.new('RGB',(1440,330),'#121c2d');d=ImageDraw.Draw(board)
    for i,key in enumerate(('before','action','after')):
        path=after if key=='after' else before
        with Image.open(path) as im:
            im=im.convert('RGB')
            if key=='action' and pa is not None:im=arrow(im,pa,target)
            im=ImageOps.contain(im,(475,268))
        board.paste(im,(i*480,55));d.text((i*480+8,30),key,font=font(17),fill='#b3d9df')
    d.text((8,5),title,font=font(18),fill='white');board.save(dest)


def prepare(root,data=Path('data/demonstrations')):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    (root/'gallery').mkdir();(root/'step_cards').mkdir()
    images={};by_hash={};steps=[];excluded=[];source_files={}
    paths=sorted(Path(data).glob('20261019*/traj_*/traj.json'))
    for manifest in paths:
        folder=manifest.parent;meta=read(manifest);source_files[str(manifest.resolve())]=sha(manifest)
        if meta.get('status')!='COMPLETE' or meta.get('operator_accepted') is not True:
            excluded.append(dict(path=str(folder),reason='not complete/operator accepted'));continue
        step_id=f's{len(steps):03d}';session=folder.parent.name;image_ids={}
        for phase,relative in [('before','observation/rgb.png'),('after','after/rgb.png')]:
            path=folder/relative
            if not path.exists():continue
            ph=pixel_hash(path);source_files[str(path.resolve())]=sha(path)
            occurrence=dict(step_id=step_id,session=session,phase=phase,source=str(path.resolve()))
            if ph not in by_hash:
                iid=f'i{len(images):04d}';by_hash[ph]=iid;dest=root/'gallery'/f'{iid}.png'
                with Image.open(path) as im:im=im.convert('RGB');size=list(im.size);im.save(dest)
                images[iid]=dict(image_id=iid,path=str(dest.relative_to(root)),sha256=sha(dest),pixel_sha256=ph,
                    size=size,mapping=resize_mapping(size),occurrences=[])
            iid=by_hash[ph];images[iid]['occurrences'].append(occurrence);image_ids[phase]=iid
        points=folder/'points.json';pa=target=None;reason='no human start/target record'
        if points.exists():
            source_files[str(points.resolve())]=sha(points)
            selected={p['name']:p.get('pixel_xy') for p in read(points)}
            pa=selected.get('start');target=selected.get('target')
        eligible=meta.get('collection_mode')=='automatic_pick_place' and pa is not None and target is not None and set(image_ids)=={'before','after'}
        if eligible:reason='human UI start/target for automatic_pick_place; reviewer must confirm experience type'
        elif meta.get('collection_mode')!='automatic_pick_place':reason='assisted start may be an approach point, not the actual human grasp; excluded from anchor GT'
        episode=meta.get('episode_id',meta.get('parameters',{}).get('reset_episode'))
        trajectory_unit=session if episode is None else f'{session}/episode_{episode}'
        step=dict(step_id=step_id,session=session,trajectory=str(folder.resolve()),trajectory_unit=trajectory_unit,
            images=image_ids,pa=pa,target=target,collection_mode=meta.get('collection_mode'),anchor_eligible=eligible,
            anchor_evidence=reason,points_source=str(points.resolve()) if points.exists() else None)
        steps.append(step)
        if set(image_ids)=={'before','after'}:
            step_card(root/images[image_ids['before']]['path'],root/images[image_ids['after']]['path'],pa,target,
                      root/'step_cards'/f'{step_id}.jpg',f'{step_id}  {session}/{folder.name} | {meta.get("collection_mode")}')
    demo_meta=read(DEMO_SOURCE/'manifest.json');demo_hash=pixel_hash(DEMO_SOURCE/'public/demo_before.png')
    demo_id=by_hash[demo_hash]
    demo_step=next(s for s in steps if s['images'].get('before')==demo_id and s['pa']==demo_meta['demo_pa'])
    pp=protocol(CHECKPOINT)
    pp['evaluation']=dict(discovery='Precision only; operator YES/NO/MAYBE for returned candidates. No recall denominator.',
        localization='Known human Pa; hit iff Top-K nearest Euclidean original-pixel distance <= 32px; K=1/5/10',
        primary='cross-session references only, excluding held-out trajectory unit and pixel-identical before/after',
        secondary='leave-one-continuous-trajectory-unit-out; do not use adjacent steps of same episode',
        unit_definition='session/episode_id when recorded; otherwise entire session',
        pooling_reference='max cosine over remaining confirmed demonstrations in the same human experience group; held-out Pa never used to propose')
    pp['discovery']=dict(top_n=50,max_per_image=MAX_PER_IMAGE,nms_radius_original_px=NMS_PX,
        exclusions='fixed demo step before/after, all occurrences from demo continuous trajectory unit, and pixel-identical images',
        crop_side_original_px=320,score_threshold=None)
    pp['localization_hit_radius_original_px']=HIT_PX
    save(root/'protocol.json',pp)
    manifest=dict(version=2,experiment='local_dense_proposal_probe',cohort='operator-saved COMPLETE trajectories in 20261019 unfolding collection; excludes pending/reset folders; all before/after RGB scanned',
        images=images,steps=steps,excluded=excluded,source_files=source_files,
        demo=dict(image_id=demo_id,step_id=demo_step['step_id'],pa=demo_meta['demo_pa'],trajectory_unit=demo_step['trajectory_unit'],session=demo_step['session']),
        protocol_sha256=sha(root/'protocol.json'),created_unix=time.time(),gpt_calls=0,robot_calls=0)
    save(root/'manifest.json',manifest)
    save(root/'experience_groups.draft.json',dict(origin='HUMAN_REVIEW_REQUIRED',annotator='',manifest_sha256=sha(root/'manifest.json'),
        steps={s['step_id']:dict(group='',confirmed=False) for s in steps}))
    save(root/'candidate_labels.json',dict(origin='HUMAN_REVIEW_REQUIRED',annotator='',labels={}))
    csv_out(root/'step_index.csv',[dict(step_id=s['step_id'],session=s['session'],trajectory=s['trajectory'],trajectory_unit=s['trajectory_unit'],
        before=s['images'].get('before'),after=s['images'].get('after'),human_pa=json.dumps(s['pa']),human_target=json.dumps(s['target']),
        anchor_eligible=s['anchor_eligible'],experience_group='UNREVIEWED') for s in steps])
    (root/'review.html').write_text('<!doctype html><meta charset="utf-8"><h2>Local dense proposal review</h2><p>启动本地 review 服务后打开显示的网址。所有标签留空，没有导入 GPT 分组。</p><pre>python -m cloth_agent.experiments.local_dense_proposal_probe review --experiment '+str(root)+'</pre>')
    summarize(root)
    return dict(steps=len(steps),rgb_occurrences=sum(len(i['occurrences']) for i in images.values()),unique_images=len(images),human_anchor_eligible=sum(s['anchor_eligible'] for s in steps))


def verify(root):
    m=read(root/'manifest.json')
    if m['version']!=2 or sha(root/'protocol.json')!=m['protocol_sha256']:raise ValueError('v2 protocol changed')
    for item in m['images'].values():
        if sha(root/item['path'])!=item['sha256']:raise ValueError('image changed')
    return m


def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def pool(grid,valid,window):
    r=window//2;h,w,d=grid.shape;out=np.zeros_like(grid);counts=np.zeros((h,w,1),np.float32)
    for dy in range(-r,r+1):
        for dx in range(-r,r+1):
            ys=slice(max(0,-dy),min(h,h-dy));xs=slice(max(0,-dx),min(w,w-dx))
            yt=slice(max(0,dy),min(h,h+dy));xt=slice(max(0,dx),min(w,w+dx))
            mask=valid[ys,xs,None];out[yt,xt]+=grid[ys,xs]*mask;counts[yt,xt]+=mask
    return normalize(out/np.maximum(counts,1))


def geometry(mapping):
    gw,gh=mapping['grid_wh'];sx,sy=mapping['scale_xy'];rw,rh=mapping['resized_size']
    yy,xx=np.mgrid[:gh,:gw];xy=np.stack(((xx+.5)*14/sx,(yy+.5)*14/sy),axis=-1)
    valid=((xx+.5)*14<rw)&((yy+.5)*14<rh)
    return xy,valid


def descriptors(grid,valid):
    single=normalize(grid);core=pool(single,valid,3);context=pool(single,valid,9)
    return dict(A=single,B=core,C=normalize(np.concatenate([core,context],axis=-1)))


def nms(scores,xy,valid,k,radius=NMS_PX):
    flat=scores.reshape(-1);points=xy.reshape(-1,2);available=np.flatnonzero(valid.reshape(-1)&np.isfinite(flat))
    order=available[np.argsort(-flat[available],kind='stable')];chosen=[]
    for idx in order:
        if all(np.linalg.norm(points[idx]-points[j])>radius for j in chosen):chosen.append(int(idx))
        if len(chosen)==k:break
    return [dict(flat_index=j,xy=points[j].tolist(),score=float(flat[j])) for j in chosen]


class Encoder:
    def __init__(self,pp):
        import torch
        from transformers import AutoModel
        self.torch=torch;torch.set_num_threads(4)
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        path=Path(pp['model_path'])
        if {p.name:sha(p) for p in path.iterdir() if p.is_file()}!=pp['checkpoint_files']:raise ValueError('checkpoint changed')
        self.model=AutoModel.from_pretrained(path,local_files_only=True).eval().requires_grad_(False).to(self.device)
        self.mean=np.asarray(pp['preprocess']['image_mean'],np.float32);self.std=np.asarray(pp['preprocess']['image_std'],np.float32)
    def encode(self,path,mapping):
        t=self.torch
        with Image.open(path) as im:
            im=im.convert('RGB').resize(tuple(mapping['resized_size']),Image.Resampling.BICUBIC)
            arr=(np.asarray(im,dtype=np.float32)/255-self.mean)/self.std
        pw,ph=mapping['padded_size'];rw,rh=mapping['resized_size'];padded=np.zeros((ph,pw,3),np.float32);padded[:rh,:rw]=arr
        tensor=t.from_numpy(padded).permute(2,0,1).unsqueeze(0).to(self.device)
        with t.inference_mode():tokens=self.model(pixel_values=tensor).last_hidden_state[0,1:].float()
        gw,gh=mapping['grid_wh']
        if tokens.shape!=(gw*gh,384):raise ValueError('unexpected patch tokens; no CLS/extra tokens permitted')
        return normalize(tokens.cpu().numpy().reshape(gh,gw,384)).astype(np.float32)


def ensure_features(root,m):
    pp=read(root/'protocol.json');cache=root/'features';cache.mkdir(exist_ok=True);encoder=None;times=[]
    contract=dict(protocol_sha256=m['protocol_sha256'],images={iid:i['sha256'] for iid,i in m['images'].items()})
    if (cache/'contract.json').exists() and read(cache/'contract.json')!=contract:raise ValueError('cache contract mismatch')
    save(cache/'contract.json',contract)
    for index,(iid,item) in enumerate(m['images'].items(),1):
        dest=cache/f'{iid}.npy';meta=cache/f'{iid}.json'
        if dest.exists() and meta.exists():
            if sha(dest)!=read(meta)['sha256']:raise ValueError('cached feature changed')
            continue
        if encoder is None:encoder=Encoder(pp)
        start=time.perf_counter();array=encoder.encode(root/item['path'],item['mapping']);elapsed=time.perf_counter()-start
        np.save(dest,array,allow_pickle=False);save(meta,dict(sha256=sha(dest),shape=list(array.shape),encode_wall_s=elapsed,device=encoder.device))
        times.append(elapsed)
        if index%10==0:print(json.dumps(dict(encoded=index,total=len(m['images']))),flush=True)
    save(cache/'status.json',dict(complete=True,image_count=len(m['images']),new_encodes=len(times),new_encode_wall_s=sum(times),gpt_calls=0,robot_calls=0))


def load_descriptors(root,item):
    grid=np.load(root/'features'/f'{item["image_id"]}.npy',allow_pickle=False)
    xy,valid=geometry(item['mapping']);return descriptors(grid,valid),xy,valid


def anchor_vector(desc,item,pa):
    sx,sy=item['mapping']['scale_xy'];col=int(math.floor(pa[0]*sx/14));row=int(math.floor(pa[1]*sy/14))
    if not (0<=row<desc.shape[0] and 0<=col<desc.shape[1]):raise ValueError('human Pa out of image')
    return desc[row,col]


def heatmap_plot(image,scores,item,path):
    # Deterministic color map across all methods/images; no score threshold.
    a=np.clip((scores+1)/2,0,1);rgb=np.stack([np.clip(2*a,0,1),np.clip(2-abs(4*a-2),0,1),np.clip(2-2*a,0,1)],axis=-1)
    hm=Image.fromarray(np.uint8(rgb*255)).resize(tuple(item['mapping']['padded_size']),Image.Resampling.BILINEAR)
    rw,rh=item['mapping']['resized_size'];hm=hm.crop((0,0,rw,rh)).resize(tuple(item['size']),Image.Resampling.BILINEAR)
    with Image.open(image) as im:Image.blend(im.convert('RGB'),hm,.48).save(path)


def crop_center(im,xy,side=320):
    x,y=xy;left=round(x-side/2);top=round(y-side/2)
    return im.crop((left,top,left+side,top+side))


def contact_sheet(root,rows,path,title):
    # Five pages of ten candidates each; page filenames preserve rank.
    for page in range(math.ceil(len(rows)/10)):
        subset=rows[page*10:(page+1)*10];board=Image.new('RGB',(1600,730),'#121c2d');d=ImageDraw.Draw(board)
        d.text((12,8),title,font=font(22),fill='white')
        for i,c in enumerate(subset):
            x=(i%5)*320;y=45+(i//5)*340
            with Image.open(root/'candidate_cards'/f'{c["candidate_id"]}_crop.png') as im:shown=ImageOps.contain(im,(300,300))
            board.paste(shown,(x+10,y));d.text((x+10,y+302),f'#{page*10+i+1} {c["image_id"]} {c["score"]:.4f}',font=font(17),fill='white')
        board.save(path.with_name(path.stem+f'_{page+1:02d}.jpg'))


def discovery(root):
    root=Path(root);m=verify(root);out=root/'discovery'
    if (out/'results.json').exists():raise ValueError('discovery already complete; no silent reranking/tuning')
    out.mkdir(exist_ok=True);(out/'heatmaps').mkdir(exist_ok=True)
    ensure_features(root,m)
    demo=m['demo'];di=m['images'][demo['image_id']];dd,_,_=load_descriptors(root,di)
    vectors={method:anchor_vector(dd[method],di,demo['pa']) for method in METHODS}
    step_lookup={s['step_id']:s for s in m['steps']}
    excluded_ids={iid for iid,item in m['images'].items() if any(step_lookup[o['step_id']]['trajectory_unit']==demo['trajectory_unit'] for o in item['occurrences'])}
    pools={method:[] for method in METHODS};stats=[];start=time.perf_counter()
    for index,(iid,item) in enumerate(m['images'].items(),1):
        if iid in excluded_ids:continue
        ds,xy,valid=load_descriptors(root,item)
        for method in METHODS:
            scores=np.einsum('hwd,d->hw',ds[method],vectors[method],optimize=False)
            np.save(out/'heatmaps'/f'{iid}_{method}.npy',scores,allow_pickle=False)
            heatmap_plot(root/item['path'],scores,item,out/'heatmaps'/f'{iid}_{method}.jpg')
            peaks=nms(scores,xy,valid,MAX_PER_IMAGE)
            for peak in peaks:
                cid=f'{iid}_p{peak["flat_index"]:04d}'
                pools[method].append(dict(candidate_id=cid,image_id=iid,**peak))
            vals=scores[valid]
            stats.append(dict(image_id=iid,method=method,peak=float(vals.max()),mean=float(vals.mean()),std=float(vals.std()),
                              p05=float(np.quantile(vals,.05)),median=float(np.median(vals)),p95=float(np.quantile(vals,.95))))
        if index%20==0:print(json.dumps(dict(scanned=index,total=len(m['images']))),flush=True)
    rankings={method:sorted(pool,key=lambda c:(-c['score'],c['image_id'],c['flat_index']))[:50] for method,pool in pools.items()}
    cards=root/'candidate_cards';cards.mkdir(exist_ok=True);unique={c['candidate_id']:c for rs in rankings.values() for c in rs}
    for cid,c in unique.items():
        item=m['images'][c['image_id']]
        with Image.open(root/item['path']) as im:im=im.convert('RGB')
        d=ImageDraw.Draw(im);x,y=c['xy'];d.ellipse((x-7,y-7,x+7,y+7),outline='red',width=3)
        im.save(cards/f'{cid}_full.jpg');crop_center(im,c['xy']).save(cards/f'{cid}_crop.png')
    rows=[]
    for method,rs in rankings.items():
        contact_sheet(root,rs,out/f'{method}_top50.jpg',f'{method} discovery Top-50 — operator labels pending')
        for rank,c in enumerate(rs,1):rows.append(dict(method=method,rank=rank,**c))
    save(out/'results.json',dict(rankings=rankings,unique_candidates=unique,excluded_image_ids=sorted(excluded_ids),
        searched_images=len(m['images'])-len(excluded_ids),scan_wall_s=time.perf_counter()-start,
        manifest_sha256=sha(root/'manifest.json'),protocol_sha256=m['protocol_sha256'],gpt_calls=0,robot_calls=0))
    csv_out(out/'candidates.csv',[dict(method=r['method'],rank=r['rank'],candidate_id=r['candidate_id'],image_id=r['image_id'],x=r['xy'][0],y=r['xy'][1],score=r['score']) for r in rows])
    csv_out(out/'score_distribution.csv',stats)
    labels=read(root/'candidate_labels.json');labels['ranking_sha256']=sha(out/'results.json')
    for cid in unique:labels['labels'].setdefault(cid,'UNREVIEWED')
    save(root/'candidate_labels.json',labels);summarize(root)
    return dict(methods=3,candidates_per_method={k:len(v) for k,v in rankings.items()},unique_candidates=len(unique),annotation_required=True)


def precision(rows,labels,k):
    selected=rows[:k];values=[labels.get(c['candidate_id'],'UNREVIEWED') for c in selected]
    n=len(selected);yes=values.count('YES');no=values.count('NO');maybe=values.count('MAYBE');pending=values.count('UNREVIEWED')
    return dict(k=k,n=n,yes=yes,no=no,maybe=maybe,unreviewed=pending,
        precision_strict=None if pending or not n else yes/n,
        precision_upper=None if pending or not n else (yes+maybe)/n,
        precision_decisive=None if pending or not yes+no else yes/(yes+no))


def freeze_groups(root):
    root=Path(root);m=verify(root);a=read(root/'experience_groups.draft.json')
    if not a.get('annotator','').strip() or a.get('origin')!='HUMAN_REVIEW_REQUIRED':raise ValueError('operator identity/confirmation required')
    if a['manifest_sha256']!=sha(root/'manifest.json'):raise ValueError('group source mismatch')
    groups=defaultdict(list);steps={s['step_id']:s for s in m['steps']}
    for sid,label in a['steps'].items():
        if label.get('confirmed'):
            if not steps[sid]['anchor_eligible'] or not label.get('group','').strip():raise ValueError('confirmed entry needs eligible human Pa and group name')
            groups[label['group'].strip()].append(sid)
    usable={g:sids for g,sids in groups.items() if len({steps[s]['trajectory_unit'] for s in sids})>=2}
    if not usable:raise ValueError('need >=2 confirmed continuous trajectory units in at least one experience group')
    p=root/'experience_groups.frozen.json'
    if p.exists():raise ValueError('groups already frozen')
    save(p,dict(origin='HUMAN_CONFIRMED',annotator=a['annotator'],groups=dict(groups),manifest_sha256=sha(root/'manifest.json'),frozen_unix=time.time()))
    save(root/'groups_freeze.json',dict(sha256=sha(p)));summarize(root);return p


def reference_steps(query,group,steps,scope):
    qhashes=set(query['images'].values());refs=[]
    for sid in group:
        s=steps[sid]
        if s['trajectory_unit']==query['trajectory_unit'] or qhashes.intersection(s['images'].values()):continue
        if scope=='cross_session_primary' and s['session']==query['session']:continue
        refs.append(s)
    return refs


def nearest_pa(peaks,pa,k):
    return min((float(np.linalg.norm(np.asarray(p['xy'])-pa)) for p in peaks[:k]),default=float('inf'))


def localize(root):
    root=Path(root);m=verify(root);fp=root/'experience_groups.frozen.json'
    if not fp.exists():raise ValueError('human experience grouping must be frozen first; GPT groups are not GT')
    groups=read(fp)
    if groups.get('origin')!='HUMAN_CONFIRMED' or groups['manifest_sha256']!=sha(root/'manifest.json') or sha(fp)!=read(root/'groups_freeze.json')['sha256']:raise ValueError('frozen groups changed')
    out=root/'localization'
    if out.exists():raise ValueError('localization output already exists')
    out.mkdir();ensure_features(root,m);steps={s['step_id']:s for s in m['steps']};rows=[];skips=[]
    for group,sids in groups['groups'].items():
        for sid in sids:
            query=steps[sid];item=m['images'][query['images']['before']]
            qdesc,xy,valid=load_descriptors(root,item)
            for scope in ('cross_session_primary','leave_trajectory_out_secondary'):
                refs=reference_steps(query,sids,steps,scope)
                if not refs:skips.append(dict(group=group,query=sid,scope=scope,reason='no independent reference'));continue
                refvec={method:[] for method in METHODS}
                for ref in refs:
                    ri=m['images'][ref['images']['before']];ds,_,_=load_descriptors(root,ri)
                    for method in METHODS:refvec[method].append(anchor_vector(ds[method],ri,ref['pa']))
                for method in METHODS:
                    # Query Pa is deliberately absent from ranking computations.
                    sims=np.einsum('hwd,rd->hwr',qdesc[method],np.asarray(refvec[method]),optimize=False)
                    heat=sims.max(axis=-1);peaks=nms(heat,xy,valid,10)
                    for c in peaks:
                        row,col=divmod(c['flat_index'],heat.shape[1]);c['reference_step']=refs[int(sims[row,col].argmax())]['step_id']
                    distances={k:nearest_pa(peaks,query['pa'],k) for k in (1,5,10)}
                    folder=out/scope/sid/method;folder.mkdir(parents=True)
                    save(folder/'candidates.json',dict(group=group,query_step=sid,reference_steps=[s['step_id'] for s in refs],candidates=peaks,gt_human_pa=query['pa']))
                    np.save(folder/'heatmap.npy',heat);heatmap_plot(root/item['path'],heat,item,folder/'heatmap.jpg')
                    with Image.open(root/item['path']) as im:im=im.convert('RGB')
                    d=ImageDraw.Draw(im)
                    for rank,c in enumerate(peaks,1):
                        x,y=c['xy'];d.ellipse((x-6,y-6,x+6,y+6),outline='yellow',width=3);d.text((x+7,y),str(rank),font=font(18),fill='yellow')
                    x,y=query['pa'];d.ellipse((x-HIT_PX,y-HIT_PX,x+HIT_PX,y+HIT_PX),outline='#30ff90',width=3);im.save(folder/'overlay.jpg')
                    for k,dist in distances.items():rows.append(dict(group=group,query_step=sid,session=query['session'],trajectory_unit=query['trajectory_unit'],scope=scope,method=method,k=k,min_distance_px=dist,hit=dist<=HIT_PX,reference_count=len(refs)))
    save(out/'results.json',dict(rows=rows,skipped=skips,groups_sha256=sha(fp),hit_radius_px=HIT_PX));csv_out(out/'results.csv',rows);summarize(root)


def summarize(root):
    root=Path(root);m=verify(root);all_metrics=[]
    lines=['# Local dense proposal probe v2','',
        '两个评价分开：retrieval_discovery 只报告返回候选的 Precision；known_positive_localization 只在人工确认的同类 experience 内评价 human Pa 定位。没有 GPT、depth、mask、训练或动作执行。',
        '',f'数据：{len(m["steps"])} 个已接受 step；{sum(len(i["occurrences"]) for i in m["images"].values())} 张原始 before/after；按 RGB 像素去重后 {len(m["images"])} 张。cohort 为 20261019 的 unfolding 采集，明确排除 pending 和 resets。',
        '', '## 预先固定的规则','',
        '- A: single patch；B: 3×3 mean + normalize；C: normalized 3×3 core / 9×9 context 等权拼接并 normalize。使用 frozen DINOv2-small patch tokens，不用 CLS。',
        '- 长边 896，保持比例，右/下补到 14 的倍数；NMS 为原图 56px。每张图最多 3 个候选，全库按分数取每方法 Top-50。',
        '- discovery 排除 demo 所属连续 trajectory unit 的全部帧及像素重复图；不是只删除 demo 文件名。',
        '- localization 的 trajectory unit 是 session + reset episode；缺失 episode 时使用整个 session。主要结果只用跨 session reference；同 session 不同 episode 仅出现在 secondary。',
        '- localization hit 为候选中心距离 held-out human Pa ≤32px；同时报告连续像素距离。这是示范点定位 Recall，不是所有合法抓点的覆盖率。',
        '', '## Retrieval discovery','']
    result_path=root/'discovery/results.json'
    if result_path.exists():
        res=read(result_path);lab=read(root/'candidate_labels.json')
        if lab.get('ranking_sha256')!=sha(result_path):raise ValueError('candidate review belongs to another ranking')
        labels=lab['labels'];valid_origin=lab.get('origin')=='HUMAN_REVIEW_REQUIRED' and bool(lab.get('annotator','').strip())
        rows=[]
        for method,cs in res['rankings'].items():
            for k in (5,10,20,50):rows.append(dict(method=method,**precision(cs,labels if valid_origin else {},k)))
        csv_out(root/'discovery/precision.csv',rows)
        all_metrics.extend(dict(task='retrieval_discovery',**r) for r in rows)
        lines += [f'排除 demo trajectory 后扫描 {res["searched_images"]} 张图；A/B/C 各返回 {len(res["rankings"]["A"])} 个候选，跨方法去重后需审核 {len(res["unique_candidates"])} 个候选。',
            '', '| Method | K | YES | NO | MAYBE | 未审核 | strict P@K | upper P@K |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:lines.append('| '+' | '.join('—' if r[k] is None else f'{r[k]:.3f}' if isinstance(r[k],float) else str(r[k]) for k in ('method','k','yes','no','maybe','unreviewed','precision_strict','precision_upper'))+' |')
        lines += ['', 'MAYBE 保留：strict=YES/K，upper=(YES+MAYBE)/K；未审核时不输出 Precision。另存只按 YES/NO 计算的 decisive precision。没有穷举正例，因此不声称 Recall。',
            '', '候选分数和坐标：[candidates.csv](discovery/candidates.csv)；每图完整分数分布：[score_distribution.csv](discovery/score_distribution.csv)。']
        for method,cs in res['rankings'].items():
            scores=np.asarray([c['score'] for c in cs]);lines.append(f'- {method} Top-50 scores: min={scores.min():.4f}, median={np.median(scores):.4f}, max={scores.max():.4f}; [contact sheet 1](discovery/{method}_top50_01.jpg)')
        lines.append('Cross-image 不等于独立场景：全库可能包含同一条其他轨迹的相邻 before/after，这些会相关。下表披露覆盖范围，不把 50 个位置当 50 个独立实验。')
        coverage=[];step_by_id={s['step_id']:s for s in m['steps']}
        for method,cs in res['rankings'].items():
            occurrences=[o for c in cs for o in m['images'][c['image_id']]['occurrences']]
            cov=dict(method=method,images=len({c['image_id'] for c in cs}),sessions=len({o['session'] for o in occurrences}),
                     trajectory_units=len({step_by_id[o['step_id']]['trajectory_unit'] for o in occurrences}))
            coverage.append(cov);lines.append(f'- {method}: {cov["images"]} images / {cov["trajectory_units"]} continuous trajectories / {cov["sessions"]} sessions')
        csv_out(root/'discovery/coverage.csv',coverage)
        # Examples become successes/failures ONLY after operator review.
        if valid_origin:
            for label in ('YES','NO','MAYBE'):
                cases=[c for cid,c in sorted(res['unique_candidates'].items()) if labels.get(cid)==label][:10]
                if cases:
                    contact_sheet(root,cases,root/'discovery'/f'reviewed_{label}.jpg',f'Operator-reviewed {label} examples (not ranked by score)')
                    lines.append(f'- [人工审核 {label} 案例](discovery/reviewed_{label}_01.jpg)')
    else:lines.append('尚未计算候选；discovery 无需 ROI/experience 标签即可运行。')
    lines+=['','## Known-positive localization','']
    if (root/'localization/results.json').exists():
        result=read(root/'localization/results.json');agg=[]
        lines+=['| Scope | Method | K | N held-out steps | Recall | Mean nearest distance px | Median px |','|---|---|---:|---:|---:|---:|---:|']
        for scope in ('cross_session_primary','leave_trajectory_out_secondary'):
            for method in METHODS:
                for k in (1,5,10):
                    rs=[r for r in result['rows'] if (r['scope'],r['method'],r['k'])==(scope,method,k)]
                    if not rs:continue
                    a=dict(scope=scope,method=method,k=k,n=len(rs),recall=float(np.mean([r['hit'] for r in rs])),mean_distance_px=float(np.mean([r['min_distance_px'] for r in rs])),median_distance_px=float(np.median([r['min_distance_px'] for r in rs])))
                    agg.append(a);lines.append(f'| {scope} | {method} | {k} | {a["n"]} | {a["recall"]:.3f} | {a["mean_distance_px"]:.2f} | {a["median_distance_px"]:.2f} |')
        csv_out(root/'localization/recall_comparison.csv',agg)
        all_metrics.extend(dict(task='known_positive_localization',**r) for r in agg)
        lines.append(f'没有独立 reference 的折数：{len(result["skipped"])}；这些不伪装为成功或失败。完整逐步距离在 localization/results.csv。')
    else:lines.append('等待操作者确认少量同类动作分组。没有把历史 GPT action family 或 graph 节点当作 experience 标签，因此当前不报告 Recall。')
    lines+=['','## Review','',
        '启动：`python -m cloth_agent.experiments.local_dense_proposal_probe review --experiment '+str(root)+'`',
        '候选页：只对返回位置标 YES/NO/MAYBE，不画 ROI。分组页：看 before/action/after，为明显同类动作填写相同 group 名称并确认；不需要完整 cloth state 描述。可只选择少量清楚的动作，其余保持未审核。',
        '', '## 解释范围','',
        '候选相似度不等于动作可执行性。动作分组必须由操作者确认；human Pa 是一次已记录的示范位置，不是唯一合法位置。主要结果以跨 session 为准，不将同一连续操作内的相邻帧当迁移证据。没有自动执行 Top-1。']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    csv_out(root/'results.csv',all_metrics)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare-v2');a.add_argument('--output',type=Path,required=True)
    for name in ('discover','review','freeze-groups','localize','summarize'):
        a=sub.add_parser(name);a.add_argument('--experiment',type=Path,required=True)
        if name=='review':a.add_argument('--port',type=int,default=8767)
    a=p.parse_args(argv)
    if a.command=='prepare-v2':print(json.dumps(prepare(a.output),ensure_ascii=False))
    elif a.command=='discover':print(json.dumps(discovery(a.experiment),ensure_ascii=False))
    elif a.command=='review':
        from .local_dense_review import serve
        serve(a.experiment,a.port)
    elif a.command=='freeze-groups':print(freeze_groups(a.experiment))
    elif a.command=='localize':localize(a.experiment)
    else:summarize(a.experiment)


if __name__=='__main__':main()
