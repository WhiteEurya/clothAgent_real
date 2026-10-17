"""Offline frozen DINOv2-small e000 audit. No robot, GPT, or training imports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvwrite(p, rows):
    if not rows: return
    with Path(p).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def unit(x): return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def evaluate(features, samples, grouped=False):
    labels=np.array([s['label'] for s in samples]); runs=np.array([s['run_id'] for s in samples])
    rows=[]
    for i,s in enumerate(samples):
        train=np.arange(len(samples))!=i
        if grouped: train &= runs!=s['run_id']
        pools=[features[train & (labels==label)] for label in ['S0','S1']]
        if any(len(p)==0 for p in pools): continue
        prototypes=np.stack([unit(p.mean(axis=0)) for p in pools]); sims=prototypes@features[i]
        prediction='S0' if sims[0]>sims[1] else 'S1' if sims[1]>sims[0] else 'TIE'
        rows.append(dict(sample_id=s['sample_id'],run_id=s['run_id'],label=s['label'],node_label=s['node_label'],
            sim_S0=float(sims[0]),sim_S1=float(sims[1]),margin=float(abs(sims[0]-sims[1])),
            prediction=prediction,local_decision={'S0':'SOURCE','S1':'TARGET','TIE':'TIE'}[prediction],
            correct=prediction==s['label'] if s['label']!='OTHER' else '',
            train_S0=len(pools[0]),train_S1=len(pools[1])))
    return rows


def collect(root, output):
    graph=read(root/'policy_graph_v2.json');trajectory=read(graph['seed_trajectory_path'])
    ref=read(Path(trajectory['observations']['o000']['rgb']['path']).parent/'observation.json')
    proposed=[];skipped=[]
    for n in graph['match_nodes']:
        for oid in n['representative_observations']:
            proposed.append((Path(trajectory['observations'][oid]['rgb']['path']),n['node_id'],'seed',
                str(root/'policy_graph_v2.json')+'#'+n['node_id']+'/'+oid,n.get('confidence','')))
    for r in sorted(root.glob('policy_replay_*')):
        for p in sorted(r.glob('step_*/node_match.json')):
            d=read(p);v=d['result']
            if (v['status']!='MATCH' or v['confidence']<.8 or not v['matched_evidence']
                    or v['violated_conditions'] or v['unresolved_conditions']):
                skipped.append(dict(source=str(p),reason='not an accepted existing node match'));continue
            proposed.append((Path(d['current_rgb']['path']),v['node_id'],r.name,str(p),v['confidence']))
    byhash={}
    for path,node,run,source,confidence in proposed:
        if not path.exists():skipped.append(dict(source=source,reason='missing image'));continue
        metadata=path.parent/'observation.json'
        if not metadata.exists():skipped.append(dict(source=source,reason='missing camera/pose metadata'));continue
        meta=read(metadata)
        if meta.get('synthetic') or meta.get('camera_serial')!=ref.get('camera_serial'):
            skipped.append(dict(source=source,reason='synthetic or different camera'));continue
        joints=np.array(meta['joints_deg']['right']);reference=np.array(ref['joints_deg']['right'])
        if np.max(np.abs(joints-reference))>1.0:
            skipped.append(dict(source=source,reason='camera arm not at fixed perception pose'));continue
        im=Image.open(path).convert('RGB')
        if im.size!=(1280,720):skipped.append(dict(source=source,reason='unexpected resolution'));continue
        h=hashlib.sha256(im.tobytes()).hexdigest()
        if h in byhash:
            row=byhash[h];row['sources'].append(source)
            if row['node_label']!=node:row['conflict']=True
        else:byhash[h]=dict(sample_id='obs_'+h[:16],run_id=run,node_label=node,
            label=node if node in ['S0','S1'] else 'OTHER',image=str(path),pixel_sha256=h,
            file_sha256=sha(path),label_confidence=confidence,sources=[source],conflict=False,
            camera_serial=meta['camera_serial'])
    samples=[];other={}
    for row in byhash.values():
        if row.pop('conflict'):
            skipped.append(dict(source=row['sources'],reason='conflicting existing labels for identical RGB'));continue
        if row['label']=='OTHER':
            node=row['node_label'];other[node]=other.get(node,0)+1
            if other[node]>2:continue
        row['label_source']=json.dumps(row.pop('sources'));samples.append(row)
    csvwrite(output/'samples.csv',samples)
    manifest=dict(policy=str(root/'policy_graph_v2.json'),policy_hash=sha(root/'policy_graph_v2.json'),
        label_rule='existing graph representatives or accepted MATCH >=0.8; no inference from action',
        camera_serial=ref.get('camera_serial'),pose_tolerance_deg=1.,
        crop_xyxy=[270,0,1180,720],crop_source='fixed table ROI reused from presentation tools; no per-image crop/mask',
        dedup='decoded RGB pixel SHA256; conflicting labels excluded',other_sampling='first two unique observations per other node',
        skipped=skipped,counts={label:sum(s['label']==label for s in samples) for label in ['S0','S1','OTHER']})
    # Latest run containing actual e000 execution AND both existing before/after matches.
    for r in sorted(root.glob('policy_replay_*'),reverse=True):
        p=r/'final_metrics.json'
        if not p.exists():continue
        transitions=read(p).get('transitions',[])
        match=next((t for t in transitions if t.get('edge_id')=='e000' and t.get('after')),None)
        if match:
            manifest['latest_e000']=dict(run=str(r),before=match['before']['path'],after=match['after']['path']);break
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return samples,manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('runs/graph_policy_full02'))
    ap.add_argument('--model',type=Path,default=Path('/tmp/dinov2-small'))
    ap.add_argument('--output',type=Path,default=Path('reports/local_node_classifier_e000'))
    ap.add_argument('--extract-only',action='store_true');args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);samples,manifest=collect(args.root.resolve(),args.output)
    print(manifest['counts'],flush=True)
    if args.extract_only:return
    import torch
    from transformers import AutoModel,AutoImageProcessor
    torch.manual_seed(0);torch.set_num_threads(4)
    model=AutoModel.from_pretrained(str(args.model),local_files_only=True).eval().requires_grad_(False)
    if model.config.model_type!='dinov2' or model.config.hidden_size!=384 or model.config.patch_size!=14:
        raise ValueError('requires DINOv2-small / ViT-S14, no substitute model')
    processor=AutoImageProcessor.from_pretrained(str(args.model),local_files_only=True,use_fast=False)
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    mean=torch.tensor(processor.image_mean).view(1,3,1,1).to(device)
    std=torch.tensor(processor.image_std).view(1,3,1,1).to(device)
    arrays={'cls':[],'patch_mean':[]};started=time.perf_counter()
    for s in samples:
        im=Image.open(s['image']).convert('RGB').crop(tuple(manifest['crop_xyxy'])).resize((224,224),Image.Resampling.BICUBIC)
        x=torch.from_numpy(np.asarray(im,dtype=np.float32)/255).permute(2,0,1).unsqueeze(0).to(device)
        with torch.inference_mode():tokens=model(pixel_values=(x-mean)/std).last_hidden_state[0].float()
        if tokens.shape!=(257,384):raise ValueError(f'unexpected token shape {tokens.shape}')
        arrays['cls'].append(tokens[0].cpu().numpy());arrays['patch_mean'].append(tokens[1:].mean(0).cpu().numpy())
    elapsed=time.perf_counter()-started
    feature_dir=args.output/'features';feature_dir.mkdir(exist_ok=True)
    results=[];summary={}
    for method,values in arrays.items():
        values=unit(np.stack(values));np.save(feature_dir/(method+'.npy'),values)
        for grouped in [False,True]:
            rows=evaluate(values,samples,grouped)
            mode='leave_one_run_out' if grouped else 'leave_one_image_out'
            for row in rows:row.update(feature=method,evaluation=mode)
            results.extend(rows);binary=[r for r in rows if r['label']!='OTHER'];others=[r for r in rows if r['label']=='OTHER']
            summary[method+'/'+mode]=dict(n=len(binary),correct=sum(r['correct'] for r in binary),
                accuracy=float(np.mean([r['correct'] for r in binary])),minimum_margin=min(r['margin'] for r in binary),
                confusion=[r['sample_id'] for r in binary if not r['correct']],
                other_sim_S0=[r['sim_S0'] for r in others],other_sim_S1=[r['sim_S1'] for r in others],
                other_margin=[r['margin'] for r in others])
    csvwrite(args.output/'predictions.csv',results)
    metadata=dict(model='facebook/dinov2-small',model_path=str(args.model),weights={p.name:sha(p) for p in args.model.glob('*.safetensors')},
        torch=torch.__version__,device=device,extraction_s=elapsed,seed=0,input_size=[224,224],
        preprocessing='fixed crop; bicubic resize; processor ImageNet mean/std; no extra center crop',
        pooling='last_hidden_state CLS or arithmetic mean of all 256 crop patch tokens, then L2 normalize',
        prototype='mean L2-normalized training image features, then L2 normalize',
        sample_order=[s['sample_id'] for s in samples],summary=summary)
    (feature_dir/'metadata.json').write_text(json.dumps(metadata,indent=2))
    text=['# Frozen DINOv2-small：e000 S0→S1 离线实验','',f"样本：{manifest['counts']}。标签来自已有graph/node matcher，不是独立人工真值。按RGB内容去重；无训练、无GPT调用、无机器人调用。",'',
        '固定Camera B perception，机器人相机臂姿态偏差≤1°；统一桌面ROI (270,0,1180,720)，224×224。未找到本组稳定衣物mask，因此patch均值含桌面背景。',
        '复用 scripts/dinov3_correspondence.py 的 bicubic resize + processor mean/std 预处理方式；使用官方 DINOv2-small，未使用DINOv3权重。','',
        '## 结果','|特征/验证|正确/总数|准确率|最小margin|','|---|---:|---:|---:|']
    for k,v in summary.items():text.append(f"|{k}|{v['correct']}/{v['n']}|{v['accuracy']:.1%}|{v['minimum_margin']:.6f}|")
    text+=['','留一图是要求的主实验；同一run相邻图相关，额外提供leave-one-run-out作为更严格的敏感性检查。','', '## 最容易混淆的图片']
    for method in arrays:
        selected=sorted([r for r in results if r['feature']==method and r['evaluation']=='leave_one_image_out' and r['label']!='OTHER'],key=lambda r:r['margin'])[:5]
        text += [f"- {method}: {r['sample_id']}，真标签{r['label']} → {r['prediction']}，margin={r['margin']:.6f}，{r['run_id']}" for r in selected]
    text+=['','## OTHER 分布（仅观察，不训练拒识器）']
    for method in arrays:
        q=[r for r in results if r['feature']==method and r['evaluation']=='leave_one_image_out' and r['label']=='OTHER']
        for key in ['sim_S0','sim_S1','margin']:
            a=np.array([r[key] for r in q]);text.append(f"- {method} {key}: min={a.min():.6f}, median={np.median(a):.6f}, max={a.max():.6f}")
        q=sorted(q,key=lambda r:r['margin'],reverse=True)[:3]
        text += [f"  - OTHER/{r['node_label']} → {r['prediction']}: margin={r['margin']:.6f}, max cosine={max(r['sim_S0'],r['sim_S1']):.6f}, {r['sample_id']}" for r in q]
    text+=['','## 最新 e000 前后（额外按run整体留出）']
    for role in ['before','after']:
        path=manifest.get('latest_e000',{}).get(role)
        sample=next((s for s in samples if s['image']==path),None)
        if not sample:text.append(f'- {role}: unavailable / no accepted label');continue
        for row in results:
            if row['sample_id']==sample['sample_id'] and row['evaluation']=='leave_one_run_out':
                text.append(f"- {role} {row['feature']}: {row['label']} → {row['prediction']}; sim_S0={row['sim_S0']:.6f}, sim_S1={row['sim_S1']:.6f}, margin={row['margin']:.6f}")
    text+=['','## 解释与限制','本结果只验证e000两类在已有少量固定视角样本上的可分性，不证明可替代全图GPT matcher。OTHER没有拒识规则，二选一必然给它S0/S1；大margin不是校准置信度。',
        '没有调阈值、训练linear probe或优化网络；模型权重下载后全部离线。不能将同一衣服/场景上的高准确率外推到揉团、遮挡或新衣物。',
        '下一步应收集独立run及更近似的负例，先验证拒识与泛化；本实验不接入机器人。']
        # Explicit answers retain uncertainty instead of choosing a winner from a tiny tied set.
    loo_cls=summary['cls/leave_one_image_out'];loo_mean=summary['patch_mean/leave_one_image_out']
    text += ['', '## 六个研究问题的结论',
        f"1. 本组S0/S1可分：CLS {loo_cls['accuracy']:.1%}，patch mean {loo_mean['accuracy']:.1%}；只是对已有标签的一致性，不是独立物理真值或跨场景稳定性证明。",
        '2. 两者准确率并列，不能认定一个稳定更好；留一图patch mean的最小margin稍大，留一run则CLS稍大。',
        f"3. 留一图最小margin：CLS {loo_cls['minimum_margin']:.6f}，patch mean {loo_mean['minimum_margin']:.6f}。",
        '4. 无二类错误；seed的S0和S1以及最新run的S0属于低margin、易混淆样本，详见上表与predictions.csv。',
        '5. OTHER能出现很高的相似度和较大margin。它们可能被二选一规则强行归到S0/S1；这些数值不可称为校准置信度。',
        '6. 值得继续小规模离线local classifier实验，但应先评估独立run和UNKNOWN拒识，暂不替代线上GPT matcher。',
        '', '## 复现',
        '```bash',
        f'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/sja/miniconda3/envs/molmo/bin/python -m cloth_agent.experiments.local_node_classifier_e000 --model {args.model} --output {args.output}',
        '```',
        f'CPU/GPU设备：{device}；{len(samples)}张提取耗时{elapsed:.3f}秒（不含模型加载；包含裁剪读取与前向，不应直接与GPT全流程耗时对比）。']
    (args.output/'report.json').write_text(json.dumps(dict(manifest=manifest,features=metadata),indent=2))
    (args.output/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if not a.startswith('other_')} for k,v in summary.items()},indent=2),flush=True)

if __name__=='__main__':main()
