"""V0–V2: prepare real outcome pairs, then run one-call feedback ablations."""
import argparse,copy,csv,hashlib,html,json,random,shutil,time
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .runtime.dataset import discover,load_observation,action_from,read
from .runtime.gpt_policy import payload
from .cv_harness.outcome_observer import observe
from .cv_harness.feedback_generator import compress
from .evaluation.evaluator import LABELS,validate_label,evaluate_outcomes
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io

def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def overlay(rgb,mask):
    a=rgb.copy();a[mask]=(a[mask]*.6+np.array([0,255,150])*.4).astype('uint8');return a

def render_cv(root,pairs,rows):
    page=['<!doctype html><meta charset="utf-8"><title>CV Feedback V0–V2</title><style>body{font:17px system-ui;margin:24px}img{width:32%}pre{white-space:pre-wrap}section{border-top:1px solid #aaa;margin-top:20px}</style><h1>真实动作前后：轻量CV反馈</h1><p>左：实际执行Pa→Pb；中：动作前mask；右：动作后mask。只描述观察事实，不把控制器完成当成抓取成功、不把轮廓面积当成展平收益。无机器人动作、无模型识别调用。当前所有数据是开发演示集，不是独立验证。</p><a href="abcd/index.html">A/B/C/D 规划对照</a> · <a href="human_review.html">人工盲审前后图</a> · <a href="report.md">文字报告</a>']
    for p in pairs:
        n=p['id'];r=rows.get(n);page.append('<section><h2>'+n+'</h2>')
        if r is None:continue
        if 'error' in r:page.append(html.escape(r['error']));continue
        page.append('<pre>'+html.escape(r['feedback'])+'</pre>')
        page.extend([f'<img src="{n}/{im}.png">' for im in ('executed','mask_before','mask_after')]);page.append(f'<p><a href="{n}/outcome.json">完整CV报告</a>；CV {r["latency_ms"]:.1f}ms</p></section>')
    (root/'index.html').write_text('\n'.join(page))

def prepare(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);pairs,excluded=discover(a.source);save(root/'pairs.json',pairs);save(root/'excluded.json',excluded)
    snapshot=root/'source_snapshot';shutil.copytree(Path(__file__).parent,snapshot,ignore=shutil.ignore_patterns('__pycache__'));shutil.copyfile(Path(__file__).parents[1]/'geometry_report.py',snapshot/'external_geometry_report.py')
    hashes={str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*.py')};save(root/'source_hashes.json',hashes)
    rows={};labels={};blind=['<!doctype html><meta charset="utf-8"><h1>人工结果标注：只看真实前后图，不显示CV预测</h1><p>在oracle_labels.json填写五个枚举字段、confirmed_by与evidence。UNKNOWN允许保留。不提供下一步动作建议。这里全部是已看过的开发数据。</p>']
    for p in pairs:
        folder=root/p['id'];folder.mkdir();start=time.perf_counter()
        try:
            b=load_observation(p['before']);c=load_observation(p['after']);action=action_from(p)
            out,mb,ma=observe(b,action,c,p['execution']);feedback=compress(out,action);save(folder/'outcome.json',asdict(out));(folder/'feedback.txt').write_text(feedback)
            Image.fromarray(overlay(b.rgb,mb)).save(folder/'mask_before.png');Image.fromarray(overlay(c.rgb,ma)).save(folder/'mask_after.png')
            shutil.copyfile(Path(p['before'])/'rgb.png',folder/'before.png');shutil.copyfile(Path(p['after'])/'rgb.png',folder/'after.png')
            io.overlay(folder/'before.png',p['plan'],folder/'executed.png','ACTUALLY DISPATCHED')
            rows[p['id']]={'feedback':feedback,'latency_ms':out.latency_ms,'io_and_cv_ms':(time.perf_counter()-start)*1000,'registration':out.motion['status']}
            save(folder/'evidence.json',{'before_rgb_sha256':sha(folder/'before.png'),'after_rgb_sha256':sha(folder/'after.png'),
                'before_xyz_sha256':sha(Path(p['before'])/'world_xyz_mm.npy'),'after_xyz_sha256':sha(Path(p['after'])/'world_xyz_mm.npy'),
                'execution_sha256':sha(Path(p['source'])/'execution.json'),'dispatch_sha256':sha(Path(p['source'])/'dispatch_intent.json')})
            labels[p['id']]={'confirmed_by':None,'evidence':[],'labels':{k:'UNKNOWN' for k in LABELS}}
            blind.append(f'<h2>{p["id"]}</h2><p>实际动作 {p["plan"]["grasp_px"]} → {p["plan"]["target_px"]}，控制器 {p["execution"]["status"]}（不代表抓住）。</p><img width="48%" src="{p["id"]}/before.png"><img width="48%" src="{p["id"]}/after.png">')
        except Exception as exc:rows[p['id']]={'error':f'{type(exc).__name__}: {exc}'}
    save(root/'cv_results.json',rows);save(root/'oracle_labels.json',labels);save(root/'label_schema.json',{k:sorted(v) for k,v in LABELS.items()});(root/'human_review.html').write_text('\n'.join(blind));render_cv(root,pairs,rows)
    lat=[r['latency_ms'] for r in rows.values() if 'error' not in r]
    summary={'pairs':len(pairs),'processed':len(lat),'errors':sum('error' in r for r in rows.values()),'excluded':excluded,
        'cv_mean_ms':float(np.mean(lat)) if lat else None,'cv_p95_ms':float(np.percentile(lat,95)) if lat else None,
        'registration_ok':sum(r.get('registration')=='OK' for r in rows.values()),'independent_labels':0,'feedback_accuracy':None,
        'data_split':'development only; episodes never split across sets','area_threshold':'not calibrated; no categorical significance claim'}
    save(root/'summary.json',summary);save(root/'protocol.json',{'source':str(a.source.resolve()),'robot_execution':False,'model_calls_in_cv':0,'conditions':{'A':'current+fixed goal','B':'current+goal+previous actual before/action','C':'current+goal+automatic compact feedback','D':'current+goal+human-confirmed outcome'},'policy_output':'unchanged Pa/Pb schema','candidate_source_hashes':hashes,'development_episodes':sorted(set(p['episode_id'] for p in pairs))})
    (root/'report.md').write_text('# V0–V2 开发数据结果\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\nAccuracy未提供：没有独立人工标签。该页面仅验证数据流、耗时和可观测数值。UNKNOWN不等于失败或无变化。\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

def trial(task):
    folder,pair,condition=task;folder=Path(folder);start=time.perf_counter();r={'case':pair['id'],'condition':condition,'status':'FAILED'}
    try:
        from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once
        from cloth_agent.restricted_context_one_shot import validate
        body=read(folder/'request.json');plan,metrics=direct_once(folder/'model',body,io.auth_headers(),240);validate(plan,Image.open(Path(pair['after'])/'rgb.png').size,False)
        save(folder/'plan.json',plan);io.overlay(Path(pair['after'])/'rgb.png',plan,folder/'action.png',condition+' - OFFLINE')
        r.update(status='COMPLETE',plan=plan,model_s=metrics['wall_latency_s'],usage=metrics.get('usage'))
    except Exception as exc:r['error']=f'{type(exc).__name__}: {exc}'
    r['wall_s']=time.perf_counter()-start;save(folder/'result.json',r);return r

def render_ab(root,rows,skipped):
    page=['<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:20px}.row{display:flex}.cell{width:33%;padding:8px}img{width:100%}pre{white-space:pre-wrap}</style><h1>Feedback A/B/C/D</h1><p>A 当前图；B 前图＋实际动作；C 自动CV反馈；D 人工确认反馈。所有组均带同一展平目标、同一prompt和Pa/Pb输出。各组一次模型调用，不预测真实执行收益。</p>']
    for case in sorted(set(r['case'] for r in rows)):
        page.append('<h2>'+case+'</h2><div class="row">')
        for r in sorted((r for r in rows if r['case']==case),key=lambda r:r['condition']):
            c=r['condition'];page.append('<div class="cell"><h3>'+c+'</h3>')
            if r['status']=='COMPLETE':
                p=r['plan'];page.append(f'<img src="{case}/{c}/action.png"><p>{r["model_s"]:.1f}s · {p["decision"]} {p["grasp_px"]} → {p["target_px"]}</p><p>'+html.escape(p['short_reason'])+'</p><p>'+html.escape(str(p['action_purpose']))+'</p>')
            else:page.append(html.escape(r.get('error','')))
            page.append('</div>')
        page.append('</div>')
    page.append('<h2>未运行条件</h2><pre>'+html.escape(json.dumps(skipped,ensure_ascii=False,indent=2))+'</pre>');(root/'index.html').write_text('\n'.join(page));save(root/'results.json',rows)

def run(a):
    root=a.output.resolve();pairs=read(root/'pairs.json');cv=read(root/'cv_results.json');labels=read(a.labels or root/'oracle_labels.json')
    # Fixed deterministic selection spanning episodes; no selection based on generated policy results.
    eligible=[p for p in pairs if 'error' not in cv[p['id']] and p['execution']['status']=='EXECUTED']
    if a.limit and len(eligible)>a.limit:
        indices=np.linspace(0,len(eligible)-1,a.limit,dtype=int);eligible=[eligible[i] for i in indices]
    out=root/'abcd';out.mkdir(exist_ok=True);tasks=[];skipped=[];previous=[]
    for p in eligible:
        for c in a.conditions:
            label=labels.get(p['id']);folder=out/p['id']/c
            if c=='D':
                if label:validate_label(label)
                if not label or not label.get('confirmed_by') or all(v=='UNKNOWN' for v in label['labels'].values()):
                    skipped.append({'case':p['id'],'condition':'D','reason':'awaiting informative independent human labels'});continue
            body=payload(p,c,cv[p['id']]['feedback'],label)
            folder.mkdir(parents=True,exist_ok=True)
            if (folder/'request.json').exists():
                if read(folder/'request.json')!=body:raise ValueError('existing request differs; use a new experiment directory')
                if (folder/'result.json').exists():previous.append(read(folder/'result.json'));continue
                if (folder/'model/attempt.json').exists():raise ValueError('interrupted attempt: no silent retry')
            save(folder/'request.json',body);tasks.append((str(folder),p,c))
    random.Random(731).shuffle(tasks)
    save(out/('protocol_'+''.join(a.conditions)+'.json'),{'selected_cases':[p['id'] for p in eligible],'conditions':a.conditions,'repeats':1,'workers':2,'call_order':[(p['id'],c) for _,p,c in tasks],
        'policy_sha256':sha(Path(__file__).parent/'runtime/gpt_policy.py'),'model':io.MODEL,'effort':io.EFFORT,'robot_execution':False,'labels_sha256':sha(a.labels or root/'oracle_labels.json')})
    rows=previous;start=time.perf_counter();render_ab(out,rows,skipped)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(trial,t) for t in tasks]):
            row=f.result();rows.append(row);render_ab(out,rows,skipped);print(json.dumps({k:row.get(k) for k in ('case','condition','status','model_s','error')},ensure_ascii=False),flush=True)
    allrows=[read(p) for p in sorted(out.glob('*/*/result.json'))];render_ab(out,allrows,skipped)
    summary={'elapsed_s':time.perf_counter()-start,'completed':sum(r['status']=='COMPLETE' for r in allrows),'failed':sum(r['status']!='COMPLETE' for r in allrows),'skipped':skipped,'decision_quality':'UNMEASURED: no independent action labels or actual counterfactual execution'}
    for c in 'ABCD':
        vals=[r['model_s'] for r in allrows if r['condition']==c and r['status']=='COMPLETE'];summary[c]={'n':len(vals),'mean_s':float(np.mean(vals)) if vals else None,'p95_s':float(np.percentile(vals,95)) if vals else None}
    save(out/'summary.json',summary);(out/'report.md').write_text('# Feedback 离线对照\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n动作不同不能证明动作更好；D未标注时不运行，也不由模型代替人工。\n');print(json.dumps(summary,ensure_ascii=False),flush=True)

def evaluate(a):
    root=a.output.resolve();outcomes={p.parent.name:read(p) for p in root.glob('*/outcome.json')};labels=read(a.labels or root/'oracle_labels.json');result=evaluate_outcomes(outcomes,labels);save(root/'evaluation.json',result);print(json.dumps(result))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','evaluate']);p.add_argument('--source',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--labels',type=Path);p.add_argument('--limit',type=int,default=6);p.add_argument('--conditions',nargs='+',choices=list('ABCD'),default=list('ABCD'));a=p.parse_args()
    if a.command=='prepare' and a.source is None:p.error('prepare requires --source')
    {'prepare':prepare,'run':run,'evaluate':evaluate}[a.command](a)
if __name__=='__main__':main()
