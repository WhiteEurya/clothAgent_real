"""Frozen-log action ablation. Does not construct any robot/camera runtime.

prepare: freeze identical payloads except a span-report text block.
run: one tool-free model request per trial, bounded workers, explicit no retries.
analyze: read-only metric projection and HTML/Markdown action comparisons.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import random
import shutil
import time
import numpy as np
from PIL import Image
from cloth_agent import restricted_context_one_shot as restricted, preference_one_shot as shared
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once
from cloth_agent.dual_arm.observation import Observation


def prepare(root):
    root=Path(root);manifest=io.read(root/'sample_manifest.json')
    protocol_file=root/'action_protocol.json'
    if protocol_file.exists():return io.read(protocol_file)
    prompt=restricted.PROMPT.replace('附带Geometry Report来自本轮RGB-D的本地计算，','若附带几何报告，它来自本轮RGB-D的本地计算，')
    # Replace contour-specific instruction with the same span-only instruction in both arms.
    prompt=prompt.replace('几何报告仅作辅助：轮廓候选不是已识别的下摆，需结合RGB辨认；未知长度保持未知，不根据候选跨度比直接决定DONE。报告不可用时按RGB规划，不猜隐藏长度。不要求复述报告或增加输出字段。',
        '若有几何报告，它仅是整体跨度的辅助证据，不是下摆长度或展开百分比；跨度接近参考不保证局部没有折叠。不要机械追求比例为1，没有固定DONE阈值。不要求复述报告或增加输出字段。')
    schema=copy.deepcopy(restricted.SCHEMA);tasks=[];rng=random.Random(241024)
    snapshots=root/'action_source_snapshot';snapshots.mkdir()
    for path in [Path(__file__),Path(restricted.__file__),Path(shared.__file__)]:shutil.copyfile(path,snapshots/path.name)
    (root/'action_prompt.txt').write_text(prompt);io.save(root/'action_schema.json',schema)
    for row in manifest:
        folder=Path(row['case_dir']);obs=Observation.load(folder/'before')
        ctx=io.read(folder/'previous_action_context.json') if (folder/'previous_action_context.json').exists() else None
        base=shared.make_payload(root,folder,obs,prompt,schema,previous_context=ctx,assess_previous_effect=False)
        base['text']['format']['name']='offline_geometry_action_ab'
        for condition in ['A','B']:
            body=copy.deepcopy(base)
            if condition=='B':
                span=io.read(folder/'span_report.json')
                body['input'][0]['content'].append(dict(type='input_text',text='整体跨度报告（仅辅助证据）：\n'+json.dumps(span,ensure_ascii=False,separators=(',',':'))))
            io.save(folder/f'action_payload_{condition}.json',body)
        for repeat in range(1,4):
            conditions=['A','B'];rng.shuffle(conditions)
            for condition in conditions:
                tasks.append(dict(sample_id=row['sample_id'],repeat=repeat,condition=condition,
                    case_dir=str(folder),payload=str(folder/f'action_payload_{condition}.json'),
                    output=str(root/'action_trials'/row['sample_id']/f'repeat_{repeat}_{condition}')))
    # Shuffle paired blocks, preserving randomized condition order within each block.
    blocks=[tasks[i:i+2] for i in range(0,len(tasks),2)];rng.shuffle(blocks);tasks=[t for block in blocks for t in block]
    for index,t in enumerate(tasks):t['scheduled_index']=index;t['payload_sha256']=io.sha(t['payload'])
    protocol=dict(created_at=io.utc(),experiment='2: effect on next action',offline=True,robot_execution=False,
        model=io.MODEL,effort=io.EFFORT,samples=len(manifest),repeats=3,requests=len(tasks),
        contrast='A RGB+goal+same previous-action context; B adds only whole-garment PCA span report',
        output_schema=schema,prompt_sha256=io.sha(root/'action_prompt.txt'),
        no_extra_tools=True,no_shadow_gate=True,http_retries=0,concurrency=3,
        done_accuracy_evaluation='deferred by user; report decisions without truth labels',
        sample_strata='assistant-selected sampling strata, not human ground truth',
        scheduling_seed=241024,model_random_seed=None,model_settings_identical=True,
        caveats=['Correlated states from same runs; repetitions measure variation, not independent garments.',
                 'Offline action changes do not establish improved physical unfolding or acquisition.',
                 'Latency includes server load/cache; parallel requests are not rollout latency measurements.'],tasks=tasks)
    io.save(protocol_file,protocol);return protocol


def run(root,workers=3,limit=None):
    protocol=io.read(root/'action_protocol.json');headers=io.auth_headers()
    tasks=[t for t in protocol['tasks'] if not (Path(t['output'])/'result.json').exists()]
    if limit is not None:tasks=tasks[:limit]
    def once(task):
        out=Path(task['output']);out.mkdir(parents=True,exist_ok=True)
        result={**task,'started_at':io.utc(),'status':'ERROR'}
        started=time.perf_counter()
        try:
            if (out/'attempt.json').exists():raise RuntimeError('prior attempt exists; no silent retry')
            if io.sha(task['payload'])!=task['payload_sha256']:raise RuntimeError('frozen payload changed')
            body=io.read(task['payload']);value,metrics=direct_once(out,body,headers,240.)
            case=Path(task['case_dir']);size=Image.open(case/'before/rgb.png').size
            # Frozen schema plus coordinate/null/type checks; no hardware imports or calls here.
            io.jsonschema.validate(value,protocol['output_schema'])
            io.validate_plan({k:value[k] for k in io.ROTATION_SCHEMA['properties']},size,require_grasp_rotation=True)
            if value['decision']!='ACTION' and value['is_smoothing']:raise ValueError('non-action smoothing true')
            io.save(out/'plan.json',value)
            io.overlay(case/'before/rgb.png',value,out/'action_overlay.png',f"{task['sample_id']} {task['condition']} repeat {task['repeat']}")
            result.update(status='OK',plan=value,metrics=metrics)
        except Exception as exc:
            result['error']=f'{type(exc).__name__}: {exc}'
            if (out/'model_metrics.json').exists():result['metrics']=io.read(out/'model_metrics.json')
        result.update(finished_at=io.utc(),job_wall_s=time.perf_counter()-started)
        io.save(out/'result.json',result)
        print(json.dumps({k:result.get(k) for k in ['sample_id','repeat','condition','status','error','job_wall_s']},ensure_ascii=False),flush=True)
        return result
    t0=time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(once,t) for t in tasks]
        for f in as_completed(futures):f.result()
    io.save(root/f'action_run_session_{datetime.now().strftime("%H%M%S")}.json',dict(jobs=len(tasks),workers=workers,wall_s=time.monotonic()-t0))


def measure(case,plan):
    if plan['decision']!='ACTION':return dict(status='NO_ACTION')
    obs=Observation.load(case/'before');a=np.array(plan['grasp_px']);b=np.array(plan['target_px'])
    v=b-a
    result=dict(status='PIXEL_ONLY',pixel_distance=float(np.linalg.norm(v)),image_direction_deg=float(math.degrees(math.atan2(v[1],v[0]))%360))
    try:
        grasp=obs.sample(a,None)
        result['grasp_measured_xyz_mm']=grasp.tolist()
        mask=np.asarray(Image.open(case/'geometry_mask.png'))>0
        result['pa_inside_color_mask']=bool(mask[a[1],a[0]])
        # Reuse the original observation's measured support-plane fit, never target depth.
        plane_file=case/'source_target_depth.json'
        if not plane_file.exists():plane_file=case/'analysis_support_plane.json'
        audit=io.read(plane_file);coef=np.array(audit['plane_z_equals_ax_by_c'])
        result['plane_source']=str(plane_file)
        K=np.array(__import__('yaml').safe_load(Path('config/calibration/dual_arm_working_20261008/camA_intrinsics.yaml').read_text())['K'])
        T=np.array(obs.meta['world_from_camera_mm']);origin=T[:3,3];direction=T[:3,:3]@np.linalg.solve(K,np.r_[b,1.])
        normal=np.array([-coef[0],-coef[1],1.]);den=float(normal@direction)
        if abs(den)<1e-8:raise ValueError('ray parallel to plane')
        scalar=float((coef[2]-normal@origin)/den)
        if scalar<=0:raise ValueError('plane behind camera')
        target=origin+scalar*direction;distance=float(np.linalg.norm(target[:2]-grasp[:2]))
        limit=30 if plan['is_smoothing'] else 150
        roi=audit['appearance_config']['settings']['table_roi_xyxy'];w,h=Image.open(obs.image).size
        result.update(status='MEASURED_PA_PLANE_PB',target_plane_xyz_mm=target.tolist(),xy_distance_mm=distance,
            translation_limit_mm=limit,translation_within_limits=10<=distance<=limit,
            pb_inside_source_table_roi=bool(int(roi[0]*w)<=b[0]<int(roi[2]*w) and int(roi[1]*h)<=b[1]<int(roi[3]*h)),
            limitation='Offline XY/ROI checks only; no IK/collision/graspability/actual execution validation.')
    except Exception as exc:result['metric_unavailable_reason']=f'{type(exc).__name__}: {exc}'
    return result


def stats(values):
    values=[float(x) for x in values if x is not None]
    return dict(n=len(values),mean=float(np.mean(values)) if values else None,median=float(np.median(values)) if values else None)


def analyze(root):
    protocol=io.read(root/'action_protocol.json');rows=[]
    for task in protocol['tasks']:
        path=Path(task['output'])/'result.json'
        if not path.exists():continue
        row=io.read(path)
        if row['status']=='OK':
            row['action_measurement']=measure(Path(task['case_dir']),row['plan'])
            io.save(Path(task['output'])/'offline_action_measurement.json',row['action_measurement'])
        rows.append(row)
    summary={};good=[r for r in rows if r['status']=='OK']
    for condition in ['A','B']:
        group=[r for r in good if r['condition']==condition];actions=[r for r in group if r['plan']['decision']=='ACTION']
        summary[condition]=dict(valid=len(group),errors=sum(r['status']!='OK' and r['condition']==condition for r in rows),
            decisions=dict(Counter(r['plan']['decision'] for r in group)),smoothing_actions=sum(r['plan']['is_smoothing'] for r in actions),
            xy_distance_mm=stats([r['action_measurement'].get('xy_distance_mm') for r in actions]),
            translation_limit_violations=sum(r['action_measurement'].get('translation_within_limits') is False for r in actions),
            pa_outside_color_mask=sum(r['action_measurement'].get('pa_inside_color_mask') is False for r in actions),
            request_latency_s=stats([r.get('metrics',{}).get('wall_latency_s') for r in group]),
            output_tokens=stats([r.get('metrics',{}).get('usage',{}).get('output_tokens') for r in group]))
    lookup={(r['sample_id'],r['repeat'],r['condition']):r for r in good};pairs=[]
    for sid in sorted({r['sample_id'] for r in rows}):
        for repeat in range(1,4):
            a=lookup.get((sid,repeat,'A'));b=lookup.get((sid,repeat,'B'))
            if not (a and b):continue
            item=dict(sample_id=sid,repeat=repeat,A=a['plan']['decision'],B=b['plan']['decision'],
                latency_B_minus_A_s=b['metrics']['wall_latency_s']-a['metrics']['wall_latency_s'])
            if a['plan']['decision']==b['plan']['decision']=='ACTION':
                item['pa_shift_px']=float(np.linalg.norm(np.array(a['plan']['grasp_px'])-b['plan']['grasp_px']))
                item['pb_shift_px']=float(np.linalg.norm(np.array(a['plan']['target_px'])-b['plan']['target_px']))
                angle=abs(a['action_measurement']['image_direction_deg']-b['action_measurement']['image_direction_deg'])
                item['direction_difference_deg']=min(angle,360-angle)
                item['smoothing_changed']=a['plan']['is_smoothing']!=b['plan']['is_smoothing']
                da=a['action_measurement'].get('xy_distance_mm');db=b['action_measurement'].get('xy_distance_mm')
                if da is not None and db is not None:item['distance_B_minus_A_mm']=db-da
            pairs.append(item)
    # Compare across-condition shifts with within-condition stochastic variation.
    variations=[]
    for sid in sorted({r['sample_id'] for r in rows}):
        for condition in ['A','B']:
            group=[r for r in good if r['sample_id']==sid and r['condition']==condition and r['plan']['decision']=='ACTION']
            for i in range(len(group)):
                for j in range(i+1,len(group)):
                    variations.append(dict(sample_id=sid,condition=condition,pa_shift_px=float(np.linalg.norm(np.array(group[i]['plan']['grasp_px'])-group[j]['plan']['grasp_px']))))
    result=dict(completed=len(rows),planned=len(protocol['tasks']),conditions=summary,pairs=pairs,
        paired_action_pa_shift_px=stats([p.get('pa_shift_px') for p in pairs]),
        within_condition_pa_shift_px={c:stats([r['pa_shift_px'] for r in variations if r['condition']==c]) for c in ['A','B']},
        paired_action_distance_delta_mm=stats([p.get('distance_B_minus_A_mm') for p in pairs]),
        paired_direction_difference_deg=stats([p.get('direction_difference_deg') for p in pairs]),
        paired_latency_delta_s=stats([p['latency_B_minus_A_s'] for p in pairs]),
        decision_transitions=dict(Counter(p['A']+' -> '+p['B'] for p in pairs)),
        human_correctness_labels=False,robot_execution=False)
    io.save(root/'action_results.json',rows);io.save(root/'action_summary.json',result)
    write_report(root,protocol,rows,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def write_report(root,protocol,rows,summary):
    notes=['实验2：整体跨度报告对下一动作的影响；实验1按用户要求暂缓。',
        '15张今天的日志图，每图每条件3次；A无几何报告，B仅增加整体跨度，不含轮廓路径比或高度起伏。',
        '所有图、目标、历史上下文、提示词和模型设置固定。每次请求独立，无新动作执行。',
        '每个重复是一次随机模型输出；需要结合组内变化看A/B差异。',
        '采样分组不是真值；不评价DONE准确率，不把动作改变称为物理效果改善。',
        '毫米距离使用Pa实测点与Pb射线/原记录桌面平面交点；没有运行IK或完整执行检查。',
        'A/B可能利用历史动作反推出部分尺度；该对照只测试额外几何文字的增量影响。']
    lines=['# 实验2：距离信息与下一步动作','','\n\n'.join(notes),'','## 汇总','','```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','']
    page=['<!doctype html><meta charset="utf-8"><title>动作几何 A/B 实验</title>',
        '<style>body{font:16px system-ui;max-width:1500px;margin:25px auto}pre{white-space:pre-wrap}table{width:100%;table-layout:fixed;border-collapse:collapse}td{vertical-align:top;border:1px solid #aaa;padding:10px}img{width:100%}section{margin:35px 0}</style>',
        '<h1>实验2：距离信息与下一步动作</h1>',*[f'<p>{html.escape(n)}</p>' for n in notes],
        '<details><summary>统计汇总</summary><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></details>']
    lookup={(r['sample_id'],r['repeat'],r['condition']):r for r in rows}
    for sample in io.read(root/'sample_manifest.json'):
        sid=sample['sample_id'];span=io.read(Path(sample['case_dir'])/'span_report.json')
        lines.extend(['## '+sid,'',f"来源：{sample['run']} / {sample['episode']} / {sample['round']}",'',f"整体跨度比例：{span['current_over_reference']}",''])
        page.extend([f'<section><h2>{sid}</h2><p>{html.escape(sample["run"]+" / "+sample["episode"]+" / "+sample["round"])}</p>',
            '<p>整体跨度比例：'+html.escape(str(span['current_over_reference']))+'</p>','<table><tr><th>A：无几何</th><th>B：有整体跨度</th></tr>'])
        for repeat in range(1,4):
            page.append('<tr>')
            for condition in ['A','B']:
                row=lookup.get((sid,repeat,condition));page.append('<td>')
                if row is None:page.append('尚未完成');page.append('</td>');continue
                if row['status']!='OK':text=row.get('error','ERROR')
                else:
                    plan=row['plan'];m=row['action_measurement']
                    text=f"重复{repeat} {condition} · {plan['decision']} · is_smoothing={plan['is_smoothing']}\nPa={plan['grasp_px']} → Pb={plan['target_px']}\n理由：{plan['short_reason']}\n平移={m.get('xy_distance_mm','不可测')} mm；请求耗时={row['metrics']['wall_latency_s']:.2f}s\n距离限制通过={m.get('translation_within_limits','不适用')}"
                    img=Path(row['output'])/'action_overlay.png';relative=img.relative_to(root).as_posix()
                    page.append(f'<a href="{relative}"><img loading="lazy" src="{relative}"></a>')
                    lines.extend([f'### 重复 {repeat} · {condition}','',text,'',f'![{sid} {condition}]({relative})',''])
                page.append('<pre>'+html.escape(text)+'</pre></td>')
            page.append('</tr>')
        page.append('</table></section>')
    (root/'action_report.md').write_text('\n'.join(lines));(root/'index.html').write_text('\n'.join(page))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','run','analyze']);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--workers',type=int,default=3);parser.add_argument('--limit',type=int)
    args=parser.parse_args();root=args.root.resolve()
    if args.mode=='prepare':print(json.dumps({k:v for k,v in prepare(root).items() if k!='tasks'},ensure_ascii=False,indent=2))
    elif args.mode=='run':run(root,args.workers,args.limit)
    else:analyze(root)


if __name__=='__main__':main()
