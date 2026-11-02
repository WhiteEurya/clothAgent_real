"""Run frozen A0/A1/A2 after oracle verification, then condition-blind evaluation."""
import argparse,html,json,random,shutil,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from PIL import Image
from .core import read,save,sha,validate_annotation,build_request,endpoint_match,PARTS
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io


def task(args):
    folder,case,condition=args;folder=Path(folder);started=time.perf_counter();row={'case':case['id'],'condition':condition,'status':'FAILED'}
    try:
        from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once
        request=read(folder/'request.json');plan,metrics=direct_once(folder/'model',request,io.auth_headers(),240)
        io.validate_plan({k:plan[k] for k in io.ROTATION_SCHEMA['properties']},case['size'],require_grasp_rotation=True)
        save(folder/'plan.json',plan);save(folder/'metrics.json',metrics)
        row.update(status='COMPLETE',plan=plan,request_s=metrics['wall_latency_s'],usage=metrics.get('usage'))
    except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'
    row['total_request_attempt_s']=time.perf_counter()-started
    save(folder/'result.json',row);return row


def run(root):
    root=Path(root);cases=read(root/'cases.json');annotations=read(root/'annotations.json');protocol=read(root/'protocol.json')
    for c in cases:
        validate_annotation(annotations.get(c['id'],{}),c['size'])
        if annotations[c['id']].get('rgb_sha256')!=c['rgb_sha256'] or sha(root/'images'/f"{c['id']}.png")!=c['rgb_sha256']:raise ValueError('annotation/image hash mismatch')
    freeze=root/'frozen/oracle_annotations.json'
    if freeze.exists() and read(freeze)!=annotations:raise ValueError('oracle changed after freeze; create another experiment')
    save(freeze,annotations)
    frozen_policy=read(root/'frozen/policy.json')
    if io.MODEL!=frozen_policy['model'] or io.EFFORT!=frozen_policy['effort']:raise ValueError('model settings changed')
    if sha(root/'goal.png')!=frozen_policy['goal_sha256']:raise ValueError('goal changed')
    for name in ('core.py','experiment.py'):
        src=Path(__file__).parent/name;dst=root/'frozen'/name
        if dst.exists() and sha(dst)!=sha(src):raise ValueError('frozen experiment source changed')
        shutil.copyfile(src,dst)
    protocol['status']='RUNNING';protocol['frozen_annotation_sha256']=sha(freeze);save(root/'protocol.json',protocol)
    tasks=[]
    for c in cases:
        for condition in ('A0','A1','A2'):
            folder=root/'trials'/c['id']/condition;folder.mkdir(parents=True,exist_ok=True);body=build_request(root,c,condition,annotations[c['id']])
            if (folder/'request.json').exists() and read(folder/'request.json')!=body:raise ValueError('frozen request changed')
            save(folder/'request.json',body)
            if (folder/'result.json').exists():continue
            if (folder/'model/attempt.json').exists():raise ValueError('unfinished dispatched request: no silent retry')
            tasks.append((str(folder),c,condition))
    random.Random(20261025).shuffle(tasks);save(root/'request_order.json',[(c['id'],a) for _,c,a in tasks]);started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(task,t) for t in tasks]):
            r=future.result();print(json.dumps({k:r.get(k) for k in ('case','condition','status','request_s','error')},ensure_ascii=False),flush=True)
    rows=[read(p) for p in sorted((root/'trials').glob('*/*/result.json'))];save(root/'results.json',rows)
    actions=[r for r in rows if r['status']=='COMPLETE' and r['plan']['decision']=='ACTION'];random.Random(631).shuffle(actions)
    blind=[{'review_id':f'R{i+1:03d}','case':r['case'],'condition':r['condition'],'pick_px':r['plan']['grasp_px']} for i,r in enumerate(actions)]
    save(root/'blind_manifest_private.json',blind);protocol.update(status='WAITING_BLIND_EVALUATION',request_batch_s=time.perf_counter()-started);save(root/'protocol.json',protocol);evaluate(root)


def evaluate(root):
    root=Path(root);rows=read(root/'results.json');annotations=read(root/'frozen/oracle_annotations.json');labels=read(root/'blind_labels.json');blind=read(root/'blind_manifest_private.json');mapping={(x['case'],x['condition']):x['review_id'] for x in blind}
    summary={};details=[]
    for condition in ('A0','A1','A2'):
        group=[r for r in rows if r['condition']==condition];complete=[r for r in group if r['status']=='COMPLETE'];actions=[r for r in complete if r['plan']['decision']=='ACTION'];known=[]
        for r in actions:
            review=labels.get(mapping[(r['case'],condition)],{})
            if review.get('confirmed_by') and review.get('judgment')!='UNCERTAIN':known.append(review)
        positive=sum(v['judgment']=='TRUE_ENDPOINT' for v in known);false=sum(v['judgment']=='FALSE_FOLD_CORNER' for v in known)
        summary[condition]={'requests':len(group),'completed':len(complete),'api_errors':len(group)-len(complete),'actions':len(actions),'DONE':sum(r['plan']['decision']=='DONE' for r in complete),'ABSTAIN':sum(r['plan']['decision']=='ABSTAIN' for r in complete),
            'independently_evaluated_actions':len(known),'evaluation_coverage':len(known)/len(actions) if actions else None,
            'true_endpoint_rate_among_evaluated_actions':positive/len(known) if known else None,'false_fold_corner_rate_among_evaluated_actions':false/len(known) if known else None,
            'true_endpoint_rate_all_actions':positive/len(actions) if actions and len(known)==len(actions) else None,'false_corner_rate_all_actions':false/len(actions) if actions and len(known)==len(actions) else None,
            'true_endpoint_confirmed_lower_bound_all_responses':positive/len(group) if group else None,
            'mean_request_s':float(np.mean([r['request_s'] for r in complete])) if complete else None,'p95_request_s':float(np.percentile([r['request_s'] for r in complete],95)) if complete else None}
        for radius in (10,20,30):summary[condition][f'coordinate_hit_rate_{radius}px']=sum(endpoint_match(r['plan'],annotations[r['case']],radius)['status']=='WITHIN_RADIUS' for r in actions)/len(actions) if actions else None
        for r in group:
            entry={**r,'coordinate_evaluation':endpoint_match(r['plan'],annotations[r['case']]) if r['status']=='COMPLETE' else None,'independent_review':labels.get(mapping.get((r['case'],condition),''))};details.append(entry)
    save(root/'summary.json',summary);save(root/'evaluation_details.json',details)
    import csv
    with (root/'results.csv').open('w') as stream:
        fields=['case','condition','status','decision','grasp_px','target_px','request_s','total_request_attempt_s','input_tokens','output_tokens','total_tokens','coordinate_part','independent_part','independent_judgment']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in details:
            p=r.get('plan',{});u=r.get('usage') or {};coord=r.get('coordinate_evaluation') or {};review=r.get('independent_review') or {}
            writer.writerow({**{k:r.get(k) for k in ['case','condition','status','request_s','total_request_attempt_s']},**{k:p.get(k) for k in ['decision','grasp_px','target_px']},**{k:u.get(k) for k in ['input_tokens','output_tokens','total_tokens']},'coordinate_part':coord.get('part'),'independent_part':review.get('part'),'independent_judgment':review.get('judgment')})
    # Conditions remain hidden in the annotation app; the final comparison is released only after every ACTION has an explicit blind review, including UNCERTAIN.
    reviewed=all(labels.get(x['review_id'],{}).get('confirmed_by') for x in blind)
    save(root/'evaluation_status.json',{'all_actions_reviewed':reviewed,'missing_reviews':[x['review_id'] for x in blind if not labels.get(x['review_id'],{}).get('confirmed_by')]})
    if reviewed:render(root,details,summary)
    else:(root/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>模型请求完成，等待独立盲审</h1><p>条件对照暂不公开，避免影响假角点复核。通过标注服务的第二阶段完成后生成完整报告。</p>')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def render(root,rows,summary):
    cases=read(root/'cases.json');parts=['<!doctype html><meta charset="utf-8"><style>body{font:17px system-ui;margin:25px}.row{display:flex}.cell{width:33%;padding:8px}img{width:100%}pre{white-space:pre-wrap}</style><h1>Semantic Corner Grounding Oracle</h1><p>离线抓点选择实验，无动作执行或成功率推断。A2得到真值候选，指标衡量候选使用而非模型自主识别能力。像素邻近与独立语义复核分别报告。</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'];md=['# Semantic Corner Grounding Oracle','',json.dumps(summary,ensure_ascii=False,indent=2),'','离线提议未执行，不能视为成功动作。A2依赖人工Oracle；端部选择正确不保证动作收益。']
    for case in cases:
        parts.append('<h2>'+case['id']+'</h2><div class="row">')
        for r in sorted((r for r in rows if r['case']==case['id']),key=lambda r:r['condition']):
            cond=r['condition'];folder=root/'trials'/case['id']/cond;parts.append('<div class="cell"><h3>'+cond+'</h3>')
            if r['status']=='COMPLETE':
                io.overlay(root/'images'/f"{case['id']}.png",r['plan'],folder/'action.png',cond)
                parts.append(f'<img src="trials/{case["id"]}/{cond}/action.png"><p>'+html.escape(r['plan']['short_reason'])+'</p><pre>'+html.escape(json.dumps({'request_s':r['request_s'],'usage':r.get('usage'),'coordinate_match':r['coordinate_evaluation'],'independent_review':r['independent_review']},ensure_ascii=False,indent=2))+'</pre>')
                if r.get('independent_review',{} ) and r['independent_review'].get('judgment') not in ('TRUE_ENDPOINT',):md.append(f"\n{case['id']} {cond}: {r['independent_review']}；理由：{r['plan']['short_reason']}")
            else:parts.append(html.escape(r.get('error','')))
            parts.append('</div>')
        parts.append('</div>')
    (root/'index.html').write_text('\n'.join(parts));(root/'report.md').write_text('\n'.join(md))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','evaluate']);p.add_argument('--root',type=Path,required=True);a=p.parse_args();{'run':run,'evaluate':evaluate}[a.command](a.root)
