"""Replay recorded before observations through GGO; never instantiate robot runtime."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import html
import json
from pathlib import Path
import shutil
import time
from cloth_agent.ggo import live
from cloth_agent.dual_arm.observation import Observation
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io


def run_case(task):
    root,source,context=task
    folder=root/source.name;folder.mkdir()
    began=time.perf_counter()
    row=dict(round=source.name,source=str(source),status='FAILED')
    try:
        obs=Observation.load(source/'before')
        (folder/'before').symlink_to(obs.directory,target_is_directory=True)
        io.save(folder/'source.json',dict(source=str(source),rgb_sha256=io.sha(obs.image),previous_context=context,robot_execution=False))
        if (source/'plan.json').exists():
            old=io.read(source/'plan.json');row['original_plan']=old
            io.overlay(obs.image,old,folder/'original_action.png','ORIGINAL RECORDED PLAN')
        if (source/'ggo/report.json').exists():row['original_registration']=io.read(source/'ggo/report.json')['registration']
        plan,metrics=live.plan_round(root,folder,obs,io.auth_headers(),240,direct_once,previous_context=context)
        report=io.read(folder/'ggo/report.json')
        row.update(status='COMPLETE',plan=plan,registration=report['registration'],deformation=report['deformation']['status'],
                   anchor_wall_s=metrics.get('anchor_wall_s'),action_model_wall_s=metrics.get('action_model_wall_s'),
                   geometry_wall_s=metrics.get('geometry_wall_s'),model_wall_s=metrics['wall_latency_s'])
    except Exception as exc:
        row['error']=f'{type(exc).__name__}: {exc}'
    row['total_wall_s']=time.perf_counter()-began
    io.save(folder/'offline_result.json',row)
    return row


def render(root,rows):
    parts=['<!doctype html><meta charset="utf-8"><title>离线 GGO 回放</title><style>body{font:18px system-ui;margin:25px}section{border-top:1px solid #aaa;margin:25px 0}.pair{display:flex;gap:12px}.pair img{width:49%}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style>',
           '<h1>被中断实验：逐轮 before 离线重规划</h1><p>新锚点识别＋修复后几何＋一次动作规划；历史上下文来自实际已执行动作。无机器人执行；原动作是历史记录，并非随机对照实验。</p>']
    md=['# 被中断实验：逐轮 before 离线重规划','', '无机器人执行。重新识别锚点，使用修复后几何，再调用一次动作模型。原动作仅为历史比较，不代表新动作实际更有效。','', '|轮次|原配准|新配准|新动作 Pa→Pb|锚点秒|动作模型秒|几何秒|理由|','|---|---|---|---|---:|---:|---:|---|']
    for r in sorted(rows,key=lambda r:r['round']):
        n=r['round'];parts.append(f'<section><h2>{n} · {r["status"]}</h2>')
        if r['status']=='COMPLETE':
            p=r['plan'];old=r.get('original_plan',{});reg=r['registration'];coords=f'{p["grasp_px"]} → {p["target_px"]}'
            parts.append('<p>原理由：'+html.escape(old.get('short_reason','无'))+'</p><p>新理由：'+html.escape(p['short_reason'])+'</p>')
            parts.append(f'<p>新配准：{reg["status"]}；形变：{r["deformation"]}；锚点 {r["anchor_wall_s"]:.1f}s，动作模型 {r["action_model_wall_s"]:.1f}s，几何 {r["geometry_wall_s"]:.2f}s。</p>')
            parts.append(f'<div class="pair"><img src="{n}/original_action.png"><img src="{n}/proposed_action_overlay.png"></div><p>左：原动作；右：新离线动作。</p><a href="{n}/ggo/index.html">几何可视化与锚点</a> · <a href="{n}/geometry_observation.txt">GPT收到的几何信息</a>')
            md.append(f'|{n}|{r.get("original_registration",{}).get("status","—")}|{reg["status"]}|{coords}|{r["anchor_wall_s"]:.1f}|{r["action_model_wall_s"]:.1f}|{r["geometry_wall_s"]:.2f}|{p["short_reason"]}|')
        else:
            parts.append('<pre>'+html.escape(r.get('error',''))+'</pre>');md.append(f'|{n}|—|FAILED|—|—|—|—|{r.get("error","")}|')
        parts.append('</section>')
    (root/'index.html').write_text('\n'.join(parts));(root/'report.md').write_text('\n'.join(md)+'\n')
    io.save(root/'results.json',sorted(rows,key=lambda r:r['round']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2,choices=[1,2]);args=parser.parse_args()
    source=args.source.resolve();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    (root/'goal').mkdir();shutil.copyfile(source/'goal/target.png',root/'goal/target.png')
    shutil.copyfile(source/'ggo_reference_seed.json',root/'ggo_reference_seed.json')
    config=io.read(source/'ggo_live_config.json');config['automatic_anchors']=True;io.save(root/'ggo_live_config.json',config)
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for p in [*Path('cloth_agent/ggo').glob('*.py'),Path(__file__)]:shutil.copyfile(p,snapshot/p.name)
    tasks=[];previous=None
    for folder in sorted(source.glob('round_*')):
        if not (folder/'before/observation.json').exists():continue
        tasks.append((root,folder,previous))
        execution=io.read(folder/'execution.json') if (folder/'execution.json').exists() else {}
        if execution.get('status')=='EXECUTED':
            previous=dict(source_round=folder.name,before_image=str(folder/'before/rgb.png'),after_image=str(folder/'after/rgb.png'),
                          plan=io.read(folder/'plan.json'),completed_commands=execution.get('completed_commands'))
    io.save(root/'protocol.json',dict(source=str(source),cases=len(tasks),workers=args.workers,model=io.MODEL,effort=io.EFFORT,
        robot_execution=False,reidentify_anchors=True,previous_context='recorded executed action only',retries=0,model_calls_per_case=2))
    rows=[];began=time.perf_counter();render(root,rows)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_case,t) for t in tasks]
        for future in as_completed(futures):
            row=future.result();rows.append(row);render(root,rows)
            print(json.dumps({k:row.get(k) for k in ['round','status','total_wall_s','registration','error']},ensure_ascii=False),flush=True)
    io.save(root/'summary.json',dict(cases=len(tasks),completed=sum(r['status']=='COMPLETE' for r in rows),elapsed_s=time.perf_counter()-began))
    print('REPORT '+str(root/'index.html'),flush=True)


if __name__=='__main__':main()
