"""Replay frozen action requests with purpose-guided planning; no robot calls."""
import argparse,copy,csv,difflib,hashlib,html,json,math,shutil,statistics,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from PIL import Image
from cloth_agent.ggo import live
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once


def distance(plan):
    return math.dist(plan['grasp_px'],plan['target_px']) if plan.get('decision')=='ACTION' else None


def task(paths):
    source,folder=map(Path,paths);start=time.perf_counter();row={'round':source.name,'status':'FAILED'}
    try:
        body=io.read(folder/'request.json');old=io.read(source/'plan.json');m=io.read(source/'model_metrics.json')
        row.update(old_plan=old,old_action_s=m.get('action_model_wall_s'),old_distance_px=distance(old))
        plan,metrics=direct_once(folder/'model',body,io.auth_headers(),240)
        live.validate(plan,Image.open(folder/'before.png').size,False)
        io.save(folder/'plan.json',plan)
        io.overlay(folder/'before.png',old,folder/'old.png','ORIGINAL')
        io.overlay(folder/'before.png',plan,folder/'new.png','PURPOSE GUIDED (OFFLINE)')
        row.update(status='COMPLETE',plan=plan,new_action_s=metrics['wall_latency_s'],new_distance_px=distance(plan))
    except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'
    row['case_wall_s']=time.perf_counter()-start;io.save(folder/'result.json',row)
    return row


def render(root,rows):
    content=['<!doctype html><meta charset="utf-8"><title>动作目的离线对照</title><style>body{font:18px system-ui;margin:25px}.pair{display:flex;gap:10px}.pair img{width:49%}section{border-top:1px solid #aaa;padding:15px}pre{white-space:pre-wrap}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>13:51 run：动作目的＋预期停止状态</h1><p>同一current/goal/历史图、实际历史动作及几何报告。只修改规划指令与输出字段，无机器人执行。左图原动作，右图新离线动作。像素距离不是毫米，也不代表动作效果。</p><a href="report.md">文字与耗时报告</a>']
    md=['# 动作目的离线对照','', '同一图片、历史动作和几何报告，只更新目的判断prompt/schema。原动作来自历史实机记录，新动作各采样一次；没有机器人执行，不能判断实际成功率。','', '|iter|原动作模型秒|新动作模型秒|原像素距离|新像素距离|动作目的|预期停止状态|','|---|---:|---:|---:|---:|---|---|']
    for r in sorted(rows,key=lambda x:x['round']):
        n=r['round'];content.append('<section><h2>'+n+'</h2>')
        if r['status']=='COMPLETE':
            p=r['plan'];old=r['old_plan']
            content.append('<p><b>目的：</b>'+html.escape(str(p['action_purpose']))+'</p><p><b>预期停止状态：</b>'+html.escape(str(p['expected_stop_state']))+'</p>')
            content.append('<p>原理由：'+html.escape(old['short_reason'])+'</p><p>新理由：'+html.escape(p['short_reason'])+'</p>')
            content.append(f'<p>原Pa/Pb：{old["grasp_px"]} → {old["target_px"]}；新：{p["grasp_px"]} → {p["target_px"]}。新动作模型 {r["new_action_s"]:.1f}s；原动作模型 {r["old_action_s"]:.1f}s。</p><div class="pair"><img src="{n}/old.png"><img src="{n}/new.png"></div>')
            md.append('|'+n+'|'+str(round(r['old_action_s'],1))+'|'+str(round(r['new_action_s'],1))+'|'+str(round(r['old_distance_px'],1) if r['old_distance_px'] is not None else '—')+'|'+str(round(r['new_distance_px'],1) if r['new_distance_px'] is not None else '—')+'|'+str(p['action_purpose'])+'|'+str(p['expected_stop_state'])+'|')
        else:content.append('<p>'+html.escape(r.get('error',''))+'</p>')
        content.append('</section>')
    (root/'index.html').write_text('\n'.join(content));(root/'report.md').write_text('\n'.join(md));io.save(root/'results.json',rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=a.source.resolve();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);tasks=[]
    for before in sorted(source.glob('round_*/before/rgb.png')):
        s=before.parent.parent;folder=root/s.name;folder.mkdir()
        body=io.read(s/'request_payload.json');old=copy.deepcopy(body)
        body['instructions']=live.PROMPT;body['text']['format']['schema']=copy.deepcopy(live.SCHEMA)
        assert body['input']==old['input']
        io.save(folder/'request.json',body);shutil.copyfile(before,folder/'before.png')
        (folder/'prompt.diff').write_text(''.join(difflib.unified_diff(old['instructions'].splitlines(True),body['instructions'].splitlines(True))))
        io.save(folder/'source.json',dict(source=str(s),source_request_sha256=io.sha(s/'request_payload.json'),rgb_sha256=io.sha(before),
            input_sha256=hashlib.sha256(json.dumps(body['input'],ensure_ascii=False,sort_keys=True).encode()).hexdigest(),identical_input=True))
        tasks.append((str(s),str(folder)))
    io.save(root/'protocol.json',dict(source=str(source),cases=len(tasks),model=io.MODEL,effort=io.EFFORT,workers=2,repeats=1,
        robot_execution=False,geometry_recomputed=False,anchors_recomputed=False,only_changes=['instructions','output schema'],
        history='recorded actual execution; offline proposals never used as executed history',timing='action model request latency, excludes original anchor and robot execution time'))
    snap=root/'source_snapshot';snap.mkdir()
    for f in [Path(__file__),Path('cloth_agent/restricted_context_one_shot.py'),Path('cloth_agent/ggo/live.py')]:shutil.copyfile(f,snap/f.name)
    rows=[];start=time.perf_counter();render(root,rows)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(task,t) for t in tasks]):
            row=future.result();rows.append(row);render(root,rows);print(json.dumps(row,ensure_ascii=False),flush=True)
    complete=[r for r in rows if r['status']=='COMPLETE'];paired=[r for r in complete if r['new_distance_px'] is not None and r['old_distance_px'] is not None]
    summary=dict(cases=len(tasks),complete=len(complete),elapsed_s=time.perf_counter()-start,
        old_mean_action_s=statistics.mean(r['old_action_s'] for r in complete) if complete else None,
        new_mean_action_s=statistics.mean(r['new_action_s'] for r in complete) if complete else None,
        shorter=sum(r['new_distance_px']<r['old_distance_px'] for r in paired),longer=sum(r['new_distance_px']>r['old_distance_px'] for r in paired),
        note='Pixel length comparisons include changes of grasp point and direction; no physical efficacy inference.')
    io.save(root/'summary.json',summary)
    with (root/'timing.csv').open('w',newline='') as f:
        fields=['round','status','old_action_s','new_action_s','case_wall_s','old_distance_px','new_distance_px'];w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(sorted(rows,key=lambda r:r['round']))
    text='\n\n## 汇总\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n'
    with (root/'report.md').open('a') as f:f.write(text)
    with (root/'index.html').open('a') as f:f.write('<h2>耗时汇总</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')

if __name__=='__main__':main()
