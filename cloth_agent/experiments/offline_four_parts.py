"""Four-part grounding and chord checks on frozen observations; no robot runtime."""
import argparse, copy, html, itertools, json, shutil, statistics, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from cloth_agent import restricted_context_one_shot as planner
from cloth_agent.dual_arm.observation import Observation
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once

KEYS=('SL','SR','HL','HR')
POINT={'type':'object','properties':{'visibility':{'type':'string','enum':['VISIBLE','HIDDEN','UNCERTAIN']},'px':{'anyOf':[{'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2},{'type':'null'}]},'confidence':{'type':'number','minimum':0,'maximum':1},'reason':{'type':'string'}},'required':['visibility','px','confidence','reason'],'additionalProperties':False}
SCHEMA={'type':'object','properties':{k:POINT for k in KEYS},'required':list(KEYS),'additionalProperties':False}
RULES='''
本次四部位报告取代上半身配准报告。SL/SR是参考标签对应的左右袖口，HL/HR是左右真实下摆角；不是当前画面的左右。参考点是人工种子，未经用户真值确认。先结合原图复核这些身份。
报告是两端点的三维直线距离，不是沿折叠布面的长度。短于参考可能是真实折叠，不能据此否认点身份；超过参考是身份或测量存疑，也不能断言假点。四点距离相容不证明身份正确。
只抓VISIBLE且身份可信的袖口/下摆带。HIDDEN、UNCERTAIN或几何SUSPECT端点不能当已确认部位直接抓；缺深度仅表示几何不可测，不等于视觉不可见。禁止另猜同一隐藏角点绕过约束。
如果目标部位不可见/存疑，优先考虑抓其可信可见对向对应部位做REVEAL显露动作；必须说明何种牵拉关系能显露它，不机械地拉对向、不要默认要大拉。没有可靠动作则ABSTAIN。
Pa可从可信部位向同一可见布面内侧微调以夹牢。保持原有目的/预期停止状态输出，short_reason点明展开自身或显露对向。四点齐全不是DONE充分条件，按当前RGB检查主要衣身/双袖/下摆。
'''
PROMPT='\n'.join(s for s in planner.PROMPT.splitlines() if not s.startswith(('几何报告仅作辅助','结构距离报告')))+RULES

def measure(obs,p):
    if p is None:return {'status':'UNMEASURABLE','reason':'no visible point'}
    x,y=p;h,w=obs.xyz.shape[:2]
    if not (2<=x<w-2 and 2<=y<h-2):return {'status':'UNMEASURABLE','reason':'outside depth support'}
    patch=obs.xyz[y-2:y+3,x-2:x+3].reshape(-1,3);v=patch[np.isfinite(patch).all(1)]
    if not np.isfinite(obs.xyz[y,x]).all() or len(v)<18:return {'status':'UNMEASURABLE','reason':'insufficient measured depth','valid_count':len(v)}
    median=np.median(v,axis=0)
    return {'status':'MEASURED','xyz_mm':median.tolist(),'valid_count':len(v),'patch_radius_mm':float(np.max(np.linalg.norm(v-median,axis=1)))}

def geometry(obs,ref,points,seed):
    current={k:measure(obs,points[k]['px']) for k in KEYS};reference={k:measure(ref,seed[k]) for k in KEYS}
    status={k:('VISUALLY_UNVERIFIED' if points[k]['visibility']!='VISIBLE' else 'COMPATIBLE_NOT_VERIFIED') for k in KEYS};pairs=[]
    for a,b in itertools.combinations(KEYS,2):
        r={'a':a,'b':b,'status':'UNMEASURABLE'};samples=[current[a],current[b],reference[a],reference[b]]
        if all(s['status']=='MEASURED' for s in samples):
            c=float(np.linalg.norm(np.array(current[a]['xyz_mm'])-current[b]['xyz_mm']));d=float(np.linalg.norm(np.array(reference[a]['xyz_mm'])-reference[b]['xyz_mm']))
            # Exploratory tolerance, not calibrated identity test. Include observed local patch spread.
            tolerance=max(15.,.10*d)+sum(s['patch_radius_mm'] for s in samples)
            state='EXCEEDS_REFERENCE_SUSPECT' if c>d+tolerance else ('SHORTER_FOLD_OR_MISIDENTIFICATION' if c<d-tolerance else 'WITHIN_TOLERANCE')
            r.update(status=state,current_mm=c,reference_mm=d,ratio=c/d if d else None,tolerance_mm=tolerance)
            if state=='EXCEEDS_REFERENCE_SUSPECT':status[a]=status[b]='SUSPECT'
        pairs.append(r)
    for k in KEYS:
        if current[k]['status']!='MEASURED' and points[k]['visibility']=='VISIBLE':status[k]='GEOMETRY_UNMEASURABLE'
    return dict(points=points,current_depth=current,reference_depth=reference,identity_checks=status,pairs=pairs,
        limitations='3D chords, not material/geodesic length. No rigid registration. Compatible does not verify identity; shortened does not disprove identity. Reference labels are assistant seed, not human GT.',
        tolerance_policy='Exploratory max(15mm,10% reference length) plus summed patch radii; not calibrated.')

def draw_points(image,points,report,out):
    im=Image.open(image).convert('RGB');d=ImageDraw.Draw(im)
    for pair in report['pairs']:
        a,b=pair['a'],pair['b'];pa,pb=points[a]['px'],points[b]['px']
        if pa and pb:
            color='#ff5555' if 'SUSPECT' in pair['status'] else '#49dfff'
            d.line([tuple(pa),tuple(pb)],fill=color,width=2)
            if 'current_mm' in pair:d.text(((pa[0]+pb[0])/2,(pa[1]+pb[1])/2),f"{pair['current_mm']:.0f}/{pair['reference_mm']:.0f}mm",fill='white',stroke_width=2,stroke_fill='black')
    for k,p in points.items():
        if p['px']:
            x,y=p['px'];d.ellipse((x-6,y-6,x+6,y+6),fill='yellow');d.text((x+8,y-15),k,fill='yellow',stroke_width=2,stroke_fill='black')
    im.save(out)

def task(args):
    s,f,refpath,seed=args;s=Path(s);f=Path(f);start=time.perf_counter();r={'case':f.name,'status':'FAILED'}
    try:
        obs=Observation.load(s/'before');ref=Observation.load(refpath)
        frozen=io.read(f/'original_request.json');body=copy.deepcopy(frozen)
        images=[c for c in body['input'][0]['content'] if c['type']=='input_image'][:2]
        body['input']=[{'role':'user','content':[{'type':'input_text','text':f'第一张当前图尺寸{Image.open(obs.image).size}，第二张展平参考图。'},*images]}]
        body['instructions']='只识别当前图四个材料部位，不规划动作，不评价历史动作。SL/SR左右袖口、HL/HR左右真实下摆角。以下参考图像素定义标签对应位置，当前衣服旋转后也保持材料身份，不能按屏幕左右重命名。袖口选择与参考相同的边缘位置。不可见或不能区分折角与真实边角时标HIDDEN/UNCERTAIN且px=null；VISIBLE必须有当前图整数像素。置信度是主观估计，不是真值。每点简述证据。参考坐标：'+json.dumps(seed)
        body['text']['format'].update(name='four_visible_parts',schema=SCHEMA)
        pts,m=direct_once(f/'grounding',body,io.auth_headers(),240);io.save(f/'points.json',pts);r['grounding_s']=m['wall_latency_s']
        w,h=Image.open(obs.image).size
        for p in pts.values():
            if p['visibility']!='VISIBLE' and p['px'] is not None:raise ValueError('invisible point has coordinates')
            if p['visibility']=='VISIBLE' and (p['px'] is None or not (0<=p['px'][0]<w and 0<=p['px'][1]<h)):raise ValueError('visible point lacks valid coordinates')
        t=time.perf_counter();g=geometry(obs,ref,pts,seed);r['geometry_s']=time.perf_counter()-t;io.save(f/'geometry.json',g)
        draw_points(obs.image,pts,g,f/'points.png')
        body=copy.deepcopy(frozen);body['instructions']=PROMPT;body['text']['format']['schema']=copy.deepcopy(planner.SCHEMA)
        for msg in body['input']:
            if isinstance(msg.get('content'),list):msg['content']=[c for c in msg['content'] if not(c['type']=='input_text' and c['text'].startswith('Geometry Observation:'))]
        body['input'][0]['content'].append({'type':'input_text','text':'Four-part Geometry Observation:\n'+json.dumps(g,ensure_ascii=False)})
        plan,m=direct_once(f/'action',body,io.auth_headers(),240);planner.validate(plan,(w,h),False);io.save(f/'plan.json',plan)
        old=io.read(f/'original_plan.json');io.overlay(obs.image,old,f/'old.png','RECORDED');io.overlay(obs.image,plan,f/'new.png','FOUR PARTS - OFFLINE')
        r.update(status='COMPLETE',plan=plan,old_plan=old,action_s=m['wall_latency_s'],visible=sum(p['visibility']=='VISIBLE' for p in pts.values()),identity_checks=g['identity_checks'])
        oldmetrics=io.read(s/'model_metrics.json');r['old_planning_s']=oldmetrics.get('wall_latency_s');r['old_action_s']=oldmetrics.get('action_model_wall_s')
    except Exception as e:r['error']=f'{type(e).__name__}: {e}'
    r['elapsed_s']=time.perf_counter()-start;io.save(f/'result.json',r);return r

def render(root,rows):
    parts=['<!doctype html><meta charset="utf-8"><title>四部位离线实验</title><style>body{font:17px system-ui;margin:24px}img{width:32%}pre{white-space:pre-wrap}section{border-top:1px solid #bbb;padding:15px}</style><h1>14:20 run：四部位识别＋几何检查＋展开/显露</h1><p>左：历史动作；中：四点与三维直线长度（当前/参考）；右：新离线动作。虚拟动作未执行，历史始终是真实动作。短距不能判错，距离相容不证明部位身份。新流程各图采样一次，没有新旧随机重复对照。</p><a href="report.md">文字报告</a>']
    md=['# 四部位离线重规划','', '来源14:20 run；不执行机器人。历史真实规划与新流程单次输出比较，不能推断成功率。距离为3D直线，不是沿折叠布面长度。参考种子非人工确认GT，容差为探索性设置。','']
    for r in sorted(rows,key=lambda r:r['case']):
        n=r['case'];parts.append(f'<section><h2>{n} {r["status"]}</h2>');md.append('## '+n)
        if r['status']=='COMPLETE':
            p=r['plan'];g=io.read(root/n/'geometry.json')
            lines=[f"原：{r['old_plan']['decision']} {r['old_plan']['grasp_px']} → {r['old_plan']['target_px']}；{r['old_plan']['short_reason']}",f"新：{p['decision']} {p['grasp_px']} → {p['target_px']}；{p['short_reason']}",f"目的：{p['action_purpose']}；预期：{p['expected_stop_state']}",f"耗时：识别 {r['grounding_s']:.1f}s，几何 {r['geometry_s']:.3f}s，规划 {r['action_s']:.1f}s；总计 {r['elapsed_s']:.1f}s"]
            lines += [f"{k}: {v['visibility']}，{v['px']}，{g['identity_checks'][k]}；{v['reason']}" for k,v in g['points'].items()]
            parts.append('<pre>'+html.escape('\n'.join(lines))+'</pre>'+''.join(f'<img src="{n}/{im}.png">' for im in ('old','points','new'))+f'<p><a href="{n}/geometry.json">完整距离和深度报告</a></p>');md.extend(['',*lines,''])
        else:parts.append(html.escape(r.get('error','')));md.append(r.get('error',''))
        parts.append('</section>')
    done=[r for r in rows if r['status']=='COMPLETE'];summary={'completed':len(done),'failed':len(rows)-len(done)}
    if done:
        summary.update({k:statistics.mean(r[k] for r in done) for k in ['grounding_s','geometry_s','action_s','elapsed_s']});summary['new_DONE']=[r['case'] for r in done if r['plan']['decision']=='DONE'];summary['all_four_visible']=sum(r['visible']==4 for r in done)
    parts.append('<h2>汇总</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>');md+=['## 汇总',json.dumps(summary,ensure_ascii=False,indent=2)]
    (root/'index.html').write_text('\n'.join(parts));(root/'report.md').write_text('\n'.join(md));io.save(root/'summary.json',summary);io.save(root/'results.json',rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    tasks=[]
    for s in sorted(a.source.resolve().glob('episode_*/round_*')):
        if not all((s/n).exists() for n in ['request_payload.json','plan.json','before/observation.json']):continue
        f=root/(s.parent.name+'_'+s.name);f.mkdir();shutil.copyfile(s/'request_payload.json',f/'original_request.json');shutil.copyfile(s/'plan.json',f/'original_plan.json')
        seed=io.read(s.parent/'ggo_reference_seed.json');ref=io.read(s.parent/'ggo_live_config.json')['reference']
        if io.sha(Path(ref)/'rgb.png')!=seed['reference_rgb_sha256']:raise ValueError('reference hash mismatch')
        tasks.append((str(s),str(f),ref,{k:seed['reference'][k] for k in KEYS}));io.save(f/'source.json',{'path':str(s),'rgb_sha256':io.sha(s/'before/rgb.png'),'request_sha256':io.sha(f/'original_request.json')})
    shutil.copyfile(__file__,root/'source_snapshot.py');io.save(root/'protocol.json',{'cases':len(tasks),'source':str(a.source),'workers':2,'repeats':1,'robot_execution':False,'geometry':'6 pairwise 3D chords, no registration, no hidden point imputation','history':'original actual history, no offline action feedback','model':io.MODEL,'effort':io.EFFORT,'tasks':tasks})
    rows=[];start=time.perf_counter();render(root,rows)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(task,t) for t in tasks]):
            r=future.result();rows.append(r);render(root,rows);print(json.dumps({k:r.get(k) for k in ('case','status','visible','elapsed_s','error')},ensure_ascii=False),flush=True)
    io.save(root/'batch_timing.json',{'elapsed_s':time.perf_counter()-start});print('REPORT '+str(root/'index.html'),flush=True)
if __name__=='__main__':main()
