"""Fixed E/M, equal pixel length, eight directions; model sees no arrows."""
from pathlib import Path
import argparse
import csv
import html
import json
import math
import random
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from cloth_agent.dual_arm.observation import Observation
from cloth_agent.dual_arm.config import DualConfig
from cloth_agent import table_target_depth
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as wire
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once

BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/'reports/one_shot_goal_full_episode_v1_20261021_232513_053641787/round_002/after'
OUT=BASE/'reports/em_fixed_pa_direction_ablation_20261022'
POINTS={'E':[795,606],'M':[702,626]}
L=120.
PROMPT='''你是单臂衣物展开动作评估器。输入只有一张当前图，仅标有 E、M 两个固定抓点，没有箭头。
目标是展开衣身和袖部，减少主要折叠与遮挡；允许整体平移/旋转和普通小褶皱，但不要把单纯搬动当作展开。
E/M 仅为候选标识，不表示哪一个更好。请独立观察图像，不要根据命名或输入顺序预设优先级。
候选动作的 Pa、Pb 和方向由 Host 固定生成，以文字给出。你只评分，不修改任何坐标、方向、长度或夹爪姿态。
它们是同一初始状态下的互斥备选，不是连续执行。所有动作都是 lift_pull_place：固定同一个 Home 夹爪姿态，抓取、抬升约50毫米、平移至 Pb、放下；不调整抓前角度，也不持布旋转。
图像坐标原点左上，x向右、y向下。theta=0°向右，90°向下，180°向左，270°向上。
全部候选的理想像素长度为120，Pb=Pa+120*(cos(theta),sin(theta))；小数坐标保留几何关系，不自行取整或修正。
对每个提供的候选返回0–100整数 score，评价完整动作值得作为下一步执行的程度；不是成功概率，分数不要求总和100。
考虑该方向可能打开哪一层、改善多大范围的折叠，以及多层牵连、整体移动或新折叠风险。
reason 用一句中文解释评分；expected_effect 说明预期局部变化；uncertainty 说明图像不能确定的事项。
同一标准评价 E 与 M，以及不同方向。preferred_id 必须是最高分候选之一，selection_reason 说明选择依据。
候选已通过 Host 的历史观测离线几何筛选；这不等于实机可执行认证。没有实时观测，不声称动作已执行或经过完整轨迹验证。
只输出指定 JSON；不得调用工具或生成动作代码。
'''


def schema(ids):
    item=dict(type='object',additionalProperties=False,properties=dict(
        id=dict(type='string',enum=ids),score=dict(type='integer',minimum=0,maximum=100),
        reason=dict(type='string'),expected_effect=dict(type='string'),uncertainty=dict(type='string')),
        required=['id','score','reason','expected_effect','uncertainty'])
    return dict(type='object',additionalProperties=False,properties=dict(
        evaluations=dict(type='array',minItems=len(ids),maxItems=len(ids),items=item),
        preferred_id=dict(type='string',enum=ids),selection_reason=dict(type='string')),
        required=['evaluations','preferred_id','selection_reason'])


def prepare():
    OUT.mkdir(parents=True,exist_ok=False)
    obs=Observation.load(SOURCE);config=DualConfig.load(BASE/'config/dual_arm.local.json',root=BASE,homing_only=True)
    obs.validate_for(config,live=False)
    arm=config.arms['left'];world_to_base=np.linalg.inv(arm.world_from_base)
    spread=config.limits['max_surface_spread_mm']
    shutil.copyfile(obs.image,OUT/'current.png')
    image=Image.open(obs.image).convert('RGB');w,h=image.size
    draw=ImageDraw.Draw(image);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',24)
    for label,(x,y) in POINTS.items():
        draw.ellipse((x-5,y-5,x+5,y+5),fill='white',outline='black',width=2)
        draw.text((x+11,y-24),label,font=font,fill='white',stroke_width=2,stroke_fill='black')
    image.save(OUT/'gpt_input_em.png')
    generated=[]
    for label,pa in POINTS.items():
        start=obs.sample(pa,spread)
        for i in range(8):
            theta=45*i;t=math.radians(theta)
            pb=[pa[0]+L*math.cos(t),pa[1]+L*math.sin(t)]
            sampled=[round(v) for v in pb]
            row=dict(id=f'{label}{i+1}',point=label,grasp_px=pa,theta_deg=theta,target_px=pb,
                nominal_length_px=L,actual_length_px=math.dist(pa,pb),depth_sample_px=sampled,
                depth_sampling_roundoff_px=math.dist(pb,sampled),status='REJECTED',reason='')
            try:
                if not (2<=pb[0]<w-2 and 2<=pb[1]<h-2):raise ValueError('Pb outside supported image interior')
                # Use recorded depth and the existing target-only plane fallback;
                # only the depth sample rounds to a raster pixel, not the candidate Pb.
                target=table_target_depth.sample_target(obs,sampled,pa,spread,OUT/f'{row["id"]}_depth.json')
                distance=float(np.linalg.norm(target[:2]-start[:2]))
                if not 10<=distance<=150:raise ValueError(f'metric translation {distance:.3f} mm outside 10–150 mm')
                top=max(start[2],target[2])+50.
                waypoints=[start+[0,0,50],start+[0,0,-15],np.r_[start[:2],top],np.r_[target[:2],top],target+[0,0,5],target+[0,0,30]]
                for p in waypoints:arm.validate_point((world_to_base@np.r_[p,1])[:3])
                row.update(status='ACCEPTED_OFFLINE',start_world_mm=start.tolist(),target_world_mm=target.tolist(),
                    metric_translation_mm=distance,depth_source=wire.read(OUT/f'{row["id"]}_depth.json')['source'])
            except Exception as error:row['reason']=f'{type(error).__name__}: {error}'
            generated.append(row)
    accepted=[r for r in generated if r['status']=='ACCEPTED_OFFLINE']
    assert accepted and len({r['point'] for r in accepted})==2
    model_candidates=[{k:r[k] for k in ('id','point','grasp_px','target_px','theta_deg','nominal_length_px')} for r in accepted]
    random.Random(20261022).shuffle(model_candidates)
    wire.save(OUT/'candidates.json',generated);wire.save(OUT/'model_candidates.json',model_candidates)
    ids=[r['id'] for r in model_candidates]
    body=wire.payload(OUT/'gpt_input_em.png',OUT/'gpt_input_em.png',[w,h])
    body['instructions']=PROMPT
    body['input'][0]['content']=[dict(type='input_text',text=f'当前图尺寸 {w}×{h}。固定点：'+json.dumps(POINTS)+'\n候选动作：\n'+json.dumps(model_candidates,ensure_ascii=False)),
        dict(type='input_image',image_url=wire.image_data(OUT/'gpt_input_em.png'),detail='high')]
    body['text']['format'].update(name='fixed_pa_direction_scores',schema=schema(ids))
    wire.save(OUT/'prepared_request.json',body)
    (OUT/'prompt.txt').write_text(PROMPT+'\n\n'+body['input'][0]['content'][0]['text'])
    wire.save(OUT/'protocol.json',dict(created_at=wire.utc(),source=str(SOURCE),model=wire.MODEL,effort=wire.EFFORT,
        points=POINTS,length_px=L,theta_degrees=list(range(0,360,45)),gripper_orientation='same fixed Home for every candidate',
        current_sha256=wire.sha(OUT/'current.png'),model_image_sha256=wire.sha(OUT/'gpt_input_em.png'),
        image_count=1,model_image_annotations=['E','M'],model_image_arrows=False,goal_image_supplied=False,
        old_scores_supplied=False,history_supplied=False,generated_count=len(generated),accepted_count=len(accepted),
        filter='recorded depth (target-only existing plane fallback), image interior, base-frame workspace at primitive waypoints, 10–150 mm planar translation',
        workspace=arm.raw['workspace'],calibration_id=obs.meta['calibration_id'],config_sha256=wire.sha(BASE/'config/dual_arm.local.json'),
        exact_float_Pb_preserved=True,depth_sampling='nearest raster pixel only; offsets logged',
        robot_execution=False,full_IK_and_collision_certification=False,
        limitation='M is the previously selected lower folded edge, not a geometrically central cloth point; candidate sets may differ after filtering.'))
    shutil.copyfile(__file__,OUT/'experiment_source.py')
    print('PREPARED',len(generated),'generated;',len(accepted),'accepted',flush=True)
    for r in generated:
        if r['status']!='ACCEPTED_OFFLINE':print('REJECT',r['id'],r['theta_deg'],r['reason'],flush=True)


def validate(output):
    ids=[r['id'] for r in wire.read(OUT/'model_candidates.json')]
    wire.jsonschema.validate(output,schema(ids))
    scores=output['evaluations'];assert len({s['id'] for s in scores})==len(ids)
    assert {s['id'] for s in scores}==set(ids)
    winner=next(s for s in scores if s['id']==output['preferred_id'])
    assert winner['score']==max(s['score'] for s in scores)


def run():
    output,metrics=direct_once(OUT,wire.read(OUT/'prepared_request.json'),wire.auth_headers(),240)
    wire.save(OUT/'output.json',output);validate(output)
    print(json.dumps(output,ensure_ascii=False),flush=True);print('MODEL_SECONDS',metrics['wall_latency_s'],flush=True)
    report()


def report():
    output=wire.read(OUT/'output.json');validate(output)
    evaluations={s['id']:s for s in output['evaluations']};all_rows=wire.read(OUT/'candidates.json')
    rows=[{**r,**evaluations[r['id']]} for r in all_rows if r['id'] in evaluations]
    rows.sort(key=lambda r:(-r['score'],r['id']))
    for i,r in enumerate(rows):
        r['rank']=i+1
        plan=dict(decision='ACTION',grasp_px=r['grasp_px'],target_px=r['target_px'],motion_type='lift_pull_place',
            short_reason=r['reason'],grasp_angle_deg=None)
        wire.overlay(OUT/'gpt_input_em.png',plan,OUT/f'{r["id"]}_arrow.png',f"{r['id']} | {r['theta_deg']} deg | SCORE {r['score']} | REVIEW ONLY")
    wire.save(OUT/'results.json',rows)
    with (OUT/'results.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','point','theta_deg','Pa','Pb_exact','score','reason','expected_effect','uncertainty'])
        for r in rows:writer.writerow([r['id'],r['point'],r['theta_deg'],r['grasp_px'],r['target_px'],r['score'],r['reason'],r['expected_effect'],r['uncertainty']])
    scores_by_id={r['id']:r for r in rows};matrix=[]
    common=[]
    for i in range(8):
        cells=[]
        if f'E{i+1}' in scores_by_id and f'M{i+1}' in scores_by_id:common.append(45*i)
        for point in ('E','M'):
            identifier=f'{point}{i+1}'
            if identifier in scores_by_id:
                r=scores_by_id[identifier];cells.append(f'<td><button data-id="{identifier}">{identifier} · {r["score"]} 分</button><br>Pb ({r["target_px"][0]:.2f}, {r["target_px"][1]:.2f})</td>')
            else:
                r=next(r for r in all_rows if r['id']==identifier);cells.append(f'<td class="excluded">剔除 {identifier}<br>{html.escape(r["reason"])}</td>')
        matrix.append(f'<tr><td>{45*i}°</td>'+''.join(cells)+'</tr>')
    metrics=wire.read(OUT/'model_metrics.json')
    cards=[]
    for r in rows:
        cards.append(f'<button class="card" data-id="{r["id"]}"><img src="{r["id"]}_arrow.png"><strong>第 {r["rank"]} · {r["id"]} · {r["theta_deg"]}° · {r["score"]} 分</strong><p>{html.escape(r["reason"])}</p></button>')
    summary=dict(preferred_id=output['preferred_id'],common_directions_deg=common,
        best_by_point={p:max((r for r in rows if r['point']==p),key=lambda r:r['score'])['id'] for p in POINTS},
        model_wall_s=metrics['wall_latency_s'])
    wire.save(OUT/'summary.json',summary)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
    best_image=Image.new('RGB',(1280,450),'white');bd=ImageDraw.Draw(best_image)
    for col,p in enumerate(('E','M')):
        r=scores_by_id[summary['best_by_point'][p]]
        im=Image.open(OUT/f'{r["id"]}_arrow.png');im.thumbnail((630,355));best_image.paste(im,(col*640+5,5))
        bd.text((col*640+12,365),f"{p} 最优：{r['theta_deg']}°，{r['score']} 分；长度 120 px",font=font,fill='#172033')
        bd.text((col*640+12,402),f"Pa {r['grasp_px']} → Pb {[round(v,2) for v in r['target_px']]}",font=font,fill='#172033')
    best_image.save(OUT/'best_E_vs_M.png')
    score_range={p:[min(r['score'] for r in rows if r['point']==p),max(r['score'] for r in rows if r['point']==p)] for p in POINTS}
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>固定 E/M · 八方向等长拉动实验</title>
<style>body{font-family:system-ui,sans-serif;background:#f3f5f9;color:#182235;margin:24px}p{line-height:1.65}.panel{background:white;padding:18px;border-radius:12px;margin-bottom:18px}.grid{display:grid;grid-template-columns:2fr 1fr;gap:20px}.grid img{width:100%}.cards{display:grid;grid-template-columns:1fr 1fr;gap:14px}.card{background:white;border:2px solid transparent;text-align:left;padding:12px;border-radius:10px;font:inherit;cursor:pointer}.card img{width:100%}.active{outline:2px solid #db9b26}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:12px;border-bottom:1px solid #ddd}.excluded{color:#805d33;font-size:13px;max-width:420px}button,select{padding:7px;cursor:pointer}a{color:#2059a3}#detail{white-space:pre-line}.metric{font-size:24px;font-weight:bold}@media(max-width:900px){.grid,.cards{grid-template-columns:1fr}}</style>
<h1>实验 1：固定 Pa，只改变 Pb</h1><p>2026-10-21 23:25 run · round_002 动作后。同一初态的互斥备选，非连续执行。</p>
<div class="panel"><p><b>E=(795,606)，M=(702,626)，L=120 px。</b>θ=0°右、90°下、180°左、270°上。所有候选保持同一个 Home 夹爪姿态。</p><p>Pb 精确按三角函数生成，不取整改变长度；仅历史深度采样取最近像素。M 是此前完整动作胜出的下端折边，不是衣身几何中心。</p>
<div class="grid"><div><h3>实际发给 GPT 的唯一图像</h3><img src="gpt_input_em.png"><p>仅标 E、M；没有黄色箭头、Pb 标记、目标图或旧评分。候选坐标作为文字输入。</p></div><div><p class="metric">模型耗时 '''+f"{metrics['wall_latency_s']:.2f} 秒"+'''</p><p>首选：<b>'''+output['preferred_id']+'''</b></p><p>'''+html.escape(output['selection_reason'])+'''</p><p>通过离线筛选：'''+str(len(rows))+''' / 16。共同有效方向：'''+', '.join(str(d)+'°' for d in common)+'''。</p><p>只用共同方向做 E/M 的逐方向比较；全部候选的最佳分同时展示，不把不对称候选数量当作优势。</p></div></div></div>
<div class="panel"><h2>两点各自最优方向</h2><a href="best_E_vs_M.png"><img src="best_E_vs_M.png" style="width:100%;max-width:1280px"></a><p>同一抓点仅改变方向，E 得分范围 '''+str(score_range['E'])+'，M 得分范围 '+str(score_range['M'])+'''。这些是单次模型评分，不是物理收益；方向造成的分数变化比两点最优动作的分差更大。</p><p>相对上一轮自由动作评分，本次还固定了长度与 Home 姿态，且只提供当前图；不把跨实验的排名变化单独归因于某一个因素。</p></div>
<div class="panel"><h2>给你看的动作箭头（未发送给模型）</h2><select id="candidate">'''+''.join(f'<option value="{r["id"]}">{r["id"]} · {r["theta_deg"]}° · {r["score"]} 分</option>' for r in rows)+'''</select><div class="grid"><img id="preview"><p id="detail"></p></div></div>
<div class="panel"><h2>同方向 E/M 对照与剔除记录</h2><table><thead><tr><th>方向</th><th>E</th><th>M</th></tr></thead><tbody>'''+''.join(matrix)+'''</tbody></table></div><div class="cards">'''+''.join(cards)+'''</div>
<div class="panel" style="margin-top:18px"><p>离线筛选检查历史实测深度/既有目标平面回退、图像边界、TCP 工作区及10–150 mm位移。未连接机器人，未做完整 IK/碰撞轨迹认证；保留候选不代表已认证可执行。分数是模型主观预期，尚无物理结果。</p>
<a href="gpt_input_em.png">模型输入图</a> · <a href="prompt.txt">完整 prompt 与文字候选</a> · <a href="candidates.json">全部16个候选与筛选证据</a> · <a href="results.csv">评分 CSV</a> · <a href="output.json">原始答案</a> · <a href="protocol.json">协议</a></div>
<script>const rows='''+json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')+''';function select(id){const r=rows.find(r=>r.id===id);document.getElementById('candidate').value=id;document.getElementById('preview').src=id+'_arrow.png';document.getElementById('detail').textContent=`${id} · ${r.theta_deg}° · ${r.score}/100\nPa ${r.grasp_px} → Pb (${r.target_px.map(v=>v.toFixed(3)).join(', ')})\n像素长度 ${r.actual_length_px.toFixed(3)}\n夹爪：固定 Home\n\n理由：${r.reason}\n\n预期效果：${r.expected_effect}\n\n不确定性：${r.uncertainty}`;document.querySelectorAll('[data-id]').forEach(e=>e.classList.toggle('active',e.dataset.id===id));}document.getElementById('candidate').onchange=e=>select(e.target.value);document.querySelectorAll('[data-id]').forEach(e=>e.onclick=()=>select(e.dataset.id));select('''+json.dumps(output['preferred_id'])+''');</script></html>'''
    (OUT/'index.html').write_text(page);print('REPORT',OUT/'index.html',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','report'])
    globals()[parser.parse_args().command]()
