"""Controlled offline prompt ablation on recorded images; no robot execution."""
import argparse,copy,difflib,html,json,random,shutil,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from PIL import Image
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once

OLD=Path('reports/reset_radius_sweep_20261024_161453_855154284/episode_0001')
NEW=Path('reports/ggo_rollout_20261024_201047_587656891')
GROUPS={'A':'旧动作规则（无几何报告）','B':'A＋整理型动作规则','C':'B＋新版几何相关规划指令','D':'C＋历史全UNKNOWN报告'}


def run_one(task):
    folder=Path(task);body=io.read(folder/'payload.json');start=time.perf_counter()
    row={'case':folder.parent.name,'group':folder.name,'status':'FAILED'}
    try:
        plan,metrics=direct_once(folder/'model',body,io.auth_headers(),240)
        io.validate_plan({k:plan[k] for k in io.ROTATION_SCHEMA['properties']},Image.open(folder.parent/'current.png').size,require_grasp_rotation=True)
        io.save(folder/'plan.json',plan);io.overlay(folder.parent/'current.png',plan,folder/'action.png',row['case']+' '+row['group'])
        dx=dy=None
        if plan['decision']=='ACTION':dx=plan['target_px'][0]-plan['grasp_px'][0];dy=plan['target_px'][1]-plan['grasp_px'][1]
        row.update(status='COMPLETE',plan=plan,dx_px=dx,dy_px=dy,model_s=metrics['wall_latency_s'])
    except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'
    row['total_s']=time.perf_counter()-start;io.save(folder/'result.json',row)
    return row


def render(root,rows):
    parts=['<!doctype html><meta charset="utf-8"><title>Prompt changes ablation</title><style>body{font:18px system-ui;margin:25px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}img{width:100%}article{border:1px solid #ccc;padding:10px}pre{white-space:pre-wrap}</style><h1>几何接入前后：非几何因素对照</h1><p>6张固定20mm实验观测，每张4组，同一current/goal/previous图和实际动作上下文。A–C没有几何报告，D只加入历史UNKNOWN。每组一次，不能当作实机成功率或统计结论。</p>']
    for case in sorted(set(r['case'] for r in rows)):
        parts.append('<h2>'+case+'</h2><div class="grid">')
        for row in sorted((r for r in rows if r['case']==case),key=lambda r:r['group']):
            g=row['group'];parts.append('<article><h3>'+g+' '+GROUPS[g]+'</h3>')
            if row['status']=='COMPLETE':
                p=row['plan'];parts.append('<p>'+html.escape(p['decision']+' '+p['short_reason'])+'</p><p>'+html.escape(str(p['grasp_px'])+' → '+str(p['target_px'])+f'；{row["model_s"]:.1f}s；整理='+str(p.get('is_smoothing','旧版无此字段')))+'</p><img src="'+case+'/'+g+'/action.png">')
            else:parts.append(html.escape(row.get('error','')))
            parts.append('</article>')
        parts.append('</div>')
    (root/'index.html').write_text('\n'.join(parts));io.save(root/'results.json',rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    old=io.read(OLD/'round_000/request_payload.json');new=io.read(NEW/'episode_0001/round_004/request_payload.json')
    oldprompt=old['instructions'];newprompt=new['instructions']
    (root/'historical_prompt.diff').write_text(''.join(difflib.unified_diff(oldprompt.splitlines(True),newprompt.splitlines(True),fromfile='old_161453',tofile='new_201047')))
    (root/'historical_old_prompt.txt').write_text(oldprompt);(root/'historical_new_prompt.txt').write_text(newprompt)
    # Remove references to absent reports equally; this is an ablation, not exact historical replay.
    def normalize(text):
        text='\n'.join(l for l in text.splitlines() if not l.startswith(('几何报告仅作辅助','结构距离报告')))
        return text.replace('附带Geometry Report来自本轮RGB-D的本地计算，不臆造像素到毫米换算；几何可行性仍由Host检查。','不臆造像素到毫米换算；几何可行性仍由Host检查。')
    a=normalize(oldprompt);c=normalize(newprompt)
    b=a
    newlines=c.splitlines()
    motion=[l for l in newlines if l.startswith(('仅使用lift_pull_place','常规动作抬升'))]
    b='\n'.join(('\n'.join(motion) if l.startswith('仅使用lift_pull_place') else next(x for x in newlines if x.startswith('ACTION输出')) if l.startswith('ACTION输出') else l) for l in a.splitlines())
    prompts={'A':a,'B':b,'C':c,'D':c}
    for k,v in prompts.items():(root/(k+'_prompt.txt')).write_text(v)
    tasks=[];cases=[]
    for ep in range(1,4):
        for rn in [4,10]:
            source=NEW/f'episode_{ep:04d}'/f'round_{rn:03d}';name=f'E{ep}_R{rn:02d}';folder=root/name;folder.mkdir()
            frozen=io.read(source/'request_payload.json');geometry=[x for x in frozen['input'][0]['content'] if x.get('text','').startswith('Geometry Observation:')]
            assert len(geometry)==1
            common=[x for x in frozen['input'][0]['content'] if x not in geometry]
            shutil.copyfile(source/'before/rgb.png',folder/'current.png')
            io.save(folder/'original_plan.json',io.read(source/'plan.json'))
            case=dict(case=name,source=str(source.resolve()),rgb_sha256=io.sha(source/'before/rgb.png'),payload_sha256=io.sha(source/'request_payload.json'))
            cases.append(case);io.save(folder/'source.json',case)
            for g in GROUPS:
                d=folder/g;d.mkdir();body=copy.deepcopy(frozen);body['instructions']=prompts[g]
                body['input'][0]['content']=copy.deepcopy(common+(geometry if g=='D' else []))
                body['text']['format']['schema']=copy.deepcopy(old['text']['format']['schema'] if g=='A' else new['text']['format']['schema'])
                io.save(d/'payload.json',body);tasks.append(str(d))
            assert io.read(folder/'B/payload.json')['input']==io.read(folder/'C/payload.json')['input']==io.read(folder/'A/payload.json')['input']
    io.save(root/'protocol.json',dict(groups=GROUPS,cases=cases,model=io.MODEL,effort=io.EFFORT,robot_execution=False,repeats=1,
        controls='identical image bytes, history, model and effort per case; B/C/D same output schema',
        limitation='A is normalized old rules without its historical geometry report; not exact historical replay; fresh stochastic calls',
        normalization='remove absent geometry-report references; no changes to visible grasp restriction, history, DONE rule, workspace or rotation rule',seed=241024))
    random.Random(241024).shuffle(tasks);rows=[];began=time.perf_counter();render(root,rows)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run_one,t) for t in tasks]):
            row=future.result();rows.append(row);render(root,rows);print(json.dumps(row,ensure_ascii=False),flush=True)
    io.save(root/'timing.json',dict(elapsed_s=time.perf_counter()-began,completed=sum(r['status']=='COMPLETE' for r in rows)))


if __name__=='__main__':main()
