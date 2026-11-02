"""Real outcome diagnostics before allowing any new policy-model experiment."""
import argparse,collections,html,json,shutil,time
from pathlib import Path
from dataclasses import asdict
import cv2
import numpy as np
from PIL import Image,ImageDraw
from .runtime.dataset import load_observation,action_from
from .cv_harness.outcome_observer import observe,execution_status
from .candidates.silhouette_area import observe_candidate
from .experiment import save,read,sha,overlay

ROOT=Path(__file__).resolve().parents[1]

def contact(folder,out):
    files=sorted((folder/'execution_images').glob('*.jpg'))
    if not files:return []
    selected=[files[i] for i in np.linspace(0,len(files)-1,min(9,len(files)),dtype=int)]
    canvas=Image.new('RGB',(960,3*205),'#f0f0f0');draw=ImageDraw.Draw(canvas)
    for i,f in enumerate(selected):
        im=Image.open(f).convert('RGB');im.thumbnail((320,180));x=(i%3)*320;y=(i//3)*205;canvas.paste(im,(x,y+22));draw.text((x+5,y+4),f.name,fill='black')
    canvas.save(out/'execution_contact.jpg');return [str(f) for f in selected]

def run(root):
    root.mkdir(parents=True,exist_ok=False);snapshot=root/'source_snapshot';shutil.copytree(Path(__file__).parent,snapshot,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(ROOT/'geometry_report.py',snapshot/'external_geometry_report.py')
    prior=Path('reports/cv_feedback_v0_v2_20261025_142045');labels=read(prior/'oracle_labels_D_frozen.json');pairs=read(prior/'pairs.json');cases=[]
    for p in pairs:
        label=labels.get(p['id'],{})
        if not label.get('confirmed_by'):continue
        group='HUMAN_IMPROVED' if label['labels']['visible_unfolding_progress']=='IMPROVED' else 'HUMAN_RIGID_NO_PROGRESS'
        cases.append({'source':p['source'],'pair':p,'group':group,'label_provenance':'existing user-confirmed outcome labels','human_labels':label['labels']})
    for path in ['ggo_rollout_20261025_135142_857883813/episode_0001/round_004','ggo_rollout_20261024_235254_998166311/episode_0001/round_007','ggo_rollout_20261024_201047_587656891/episode_0001/round_007']:
        cases.append({'source':str((Path('reports')/path).resolve()),'group':'NEW_COVERAGE_CANDIDATE','label_provenance':'assistant visual screening; not independent ground truth'})
    for path in ['ggo_rollout_20261025_142045_387801620/episode_0002/round_010','ggo_rollout_20261025_135142_857883813/episode_0001/round_007','reset_radius_sweep_20261024_131929_882888513/episode_0001/round_009']:
        cases.append({'source':str((Path('reports')/path).resolve()),'group':'EXECUTION_EXCEPTION','label_provenance':'controller and interruption logs; no assumption of grasp failure'})
    save(root/'selection.json',cases);rows=[]
    for i,case in enumerate(cases):
        key=f'S{i+1:02d}';folder=root/key;folder.mkdir();source=Path(case['source']);row={**case,'id':key};log=read(source/'execution.json');row['execution_log']=log
        row['execution_contact_sources']=contact(source,folder)
        for side in ('before','after'):
            image=source/side/'rgb.png'
            if image.exists():shutil.copyfile(image,folder/(side+'.png'));row[side+'_rgb_sha256']=sha(image)
        if not (source/'after/observation.json').exists():
            row['status']='OUTCOME_UNOBSERVABLE';row['execution_analysis']=execution_status(log);row['reason']='no safe final RGB-D; sampled moving-camera execution frames are not a substitute'
        else:
            b=load_observation(source/'before');a=load_observation(source/'after');plan=read(source/'dispatch_intent.json')['plan'];p={'plan':plan};action=action_from(p)
            baseline,mb,ma=observe(b,action,a,log);candidate,fb,fa,holes=observe_candidate(b,action,a,log)
            row.update(status='ANALYZED',baseline=asdict(baseline),candidate=asdict(candidate))
            for name,value in [('baseline',baseline),('candidate',candidate)]:save(folder/(name+'.json'),asdict(value))
            for name,rgb,mask in [('baseline_before',b.rgb,mb),('baseline_after',a.rgb,ma),('candidate_before',b.rgb,fb),('candidate_after',a.rgb,fa)]:Image.fromarray(overlay(rgb,mask)).save(folder/(name+'.png'))
            if baseline.coverage['delta_ratio'] is not None and candidate.coverage['delta_ratio'] is not None:row['area_delta_change_pp']=100*(candidate.coverage['delta_ratio']-baseline.coverage['delta_ratio'])
        save(folder/'result.json',row);rows.append(row)
    summary={'cases':len(rows),'groups':dict(collections.Counter(r['group'] for r in rows)),'model_calls':0,'robot_actions':0,'gpt_gate':'NOT_PASSED',
       'reason':'current observer does not classify unfolding/rigid-only; insufficient calibrated evidence and unknown coverage. Candidate changes measurement, not validated semantic capability.',
       'independent_outcome_label_cases':6,'new_coverage_ground_truth':False,'confirmed_empty_grasps_or_slips':0,'candidate_deployment':'NOT_ACCEPTED'}
    for variant in ('baseline','candidate'):
        data=[r[variant] for r in rows if r['status']=='ANALYZED'];summary[variant]={'paired_cases':len(data),'area_available':sum(d['coverage']['delta_ratio'] is not None for d in data),'registration_ok':sum(d['motion']['status']=='OK' for d in data),'semantic_unfolding_or_rigid_only_predictions':0,'mean_ms':float(np.mean([d['latency_ms'] for d in data])),'p95_ms':float(np.percentile([d['latency_ms'] for d in data],95))}
    save(root/'summary.json',summary);save(root/'results.json',rows);save(root/'source_hashes.json',{str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*.py')})
    render(root,rows,summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def render(root,rows,summary):
    description='CV验证优先：3组人工确认展开、3组人工确认主要拖动且无进展、3组疑似新覆盖、3组执行异常。没有GPT请求。疑似新覆盖只是筛选分组，不是独立真值；执行异常不等于抓空。'
    parts=['<!doctype html><meta charset="utf-8"><title>CV结果诊断</title><style>body{font:17px system-ui;margin:25px}.pair{display:flex;gap:12px}.pair img{width:49%}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}section{border-top:1px solid #aaa;padding-top:16px;margin-top:20px}</style><h1>CV能否区分展开与拖动？</h1><p>'+description+'</p><p><b>当前结论：尚不能可靠区分，因此本轮不进入GPT实验。</b>面积/形状变化不直接等同于展开；刚体运动与局部展开可同时发生。</p><a href="report.md">文字报告</a><table><tr><th>样本</th><th>筛选分组</th><th>原面积变化</th><th>候选外轮廓变化</th><th>原刚体对齐IoU</th></tr>']
    md=['# CV结果诊断','',description,'','当前未通过GPT实验前置条件。','', '|案例|分组|原面积变化|外轮廓面积变化|原对齐IoU|','|---|---|---:|---:|---:|']
    for r in rows:
        if r['status']=='ANALYZED':
            c=r['baseline']['coverage']['delta_ratio'];d=r['candidate']['coverage']['delta_ratio'];iou=r['baseline']['motion'].get('aligned_iou');vals=[f'{v*100:+.1f}%' if v is not None else 'UNKNOWN' for v in (c,d)]+[f'{iou:.3f}' if iou is not None else 'UNKNOWN']
        else:vals=['无安全后图']*3
        parts.append('<tr><td><a href="#'+r['id']+'">'+r['id']+'</a></td><td>'+r['group']+'</td>'+''.join('<td>'+v+'</td>' for v in vals)+'</tr>');md.append('|'+ '|'.join([r['id'],r['group'],*vals])+'|')
    parts.append('</table>')
    for r in rows:
        key=r['id'];parts.append(f'<section id="{key}"><h2>{key} · {r["group"]}</h2><p>'+html.escape(r['source'])+'</p><p>'+html.escape(r['label_provenance'])+'</p>')
        if r.get('human_labels'):parts.append('<pre>'+html.escape(json.dumps(r['human_labels'],ensure_ascii=False,indent=2))+'</pre>')
        parts.append('<div class="pair"><img src="'+key+'/before.png">'+('<img src="'+key+'/after.png">' if r.get('after_rgb_sha256') else '<p>缺少动作后观测：不填补、不复用别轮照片。</p>')+'</div>')
        if r['status']=='ANALYZED':
            parts.append('<details><summary>原CV与候选测量、mask对比</summary><p>上排原颜色mask；下排候选外轮廓。填补印花/阴影形成的封闭孔洞只用于外轮廓面积，绝不用于抓点验证。</p>')
            for prefix in ('baseline','candidate'):parts.append(f'<div class="pair"><img src="{key}/{prefix}_before.png"><img src="{key}/{prefix}_after.png"></div><pre>'+html.escape(json.dumps(r[prefix],ensure_ascii=False,indent=2))+'</pre>')
            parts.append('</details>')
        parts.append('<pre>'+html.escape(json.dumps(r['execution_log'],ensure_ascii=False,indent=2))+'</pre>')
        if r['execution_contact_sources']:parts.append(f'<details><summary>执行视频序列的9张采样帧（移动相机，不作为对齐后的效果图）</summary><img style="max-width:100%" src="{key}/execution_contact.jpg"></details>')
        parts.append('</section>')
    parts.append('<h2>汇总与边界</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')
    md+=['','## 汇总','',json.dumps(summary,ensure_ascii=False,indent=2),'','候选改进仅将颜色支持面积与外轮廓包围面积分开，减少封闭印花孔洞的影响；尚未解决相机位姿变化、衣物出画或纹理不足。未部署、未改评价标签、没有宣称分类准确率提升。']
    (root/'index.html').write_text('\n'.join(parts));(root/'report.md').write_text('\n'.join(md))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output.resolve())
