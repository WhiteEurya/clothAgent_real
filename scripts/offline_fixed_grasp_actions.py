"""Complete ten fixed grasp candidates into alternative single-step actions."""
from pathlib import Path
import argparse
import copy
import csv
import html
import json
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as wire
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once

SOURCE=Path('reports/grasp_point_scores_2325_round002_20261022')
OUT=Path('reports/fixed_grasp_actions_2325_round002_20261022')
PROMPT='''你是单臂衣物展开的单步规划器。第一张是当前 RGB，第二张是展平目标 RGB。
给定十个固定候选抓点 Pa，为每个候选设计一个最佳的完整单步动作，再比较并选择最值得执行的一个。
这是同一当前状态下的十个互斥备选动作，不是依次执行的十步。每个动作都必须基于第一张当前图。
任务是使衣物主要部位的展开布局和折叠/遮挡状态接近目标；允许整体平移或旋转、普通小褶皱，不要求背景或绝对桌面位置一致。
每个候选的 id 只是标识，不表示优先级；输入顺序也不表示优先级。必须原样保留每个 id 和 grasp_px，不能移动、替换或增加抓点。
为每个 Pa 选择 target_px（Pb）和 grasp_angle_deg，并评价这个完整动作的预期展开价值。
只允许现有 lift_pull_place：抓前可以调整夹爪方向，抓住后保持该方向完成抬升约50毫米、平移10–150毫米、放下；不允许持布旋转。
所有像素均以第一张原图左上角为原点，x向右、y向下；坐标为整数，不把目标图坐标直接搬到当前图。
Pb 需要落在当前可见工作台内，方向和幅度应有助于打开可见折叠，避免只是搬动整个衣服或制造新的折叠。
没有深度输入；不要臆造像素到毫米换算或声称通过了深度、IK、碰撞、边界检查。
十个候选均需返回。能提出有依据的动作时 decision=ACTION、motion_type=lift_pull_place；若某个固定抓点确实无法支持合理单步动作，可返回 ABSTAIN，target_px/motion_type/grasp_angle_deg 为 null，action_score=0，并说明原因。
action_score 是0–100整数，表示你对完整动作“值得优先执行”的主观评分，不是成功概率，各项不要求总和100。
同时考虑预期减少折叠/遮挡的收益、夹取单层的依据、方向合理性、多层牵连或整体移动风险。
short_reason 用一句中文解释动作选择；expected_effect 描述预期局部变化；uncertainty 说明不确定性。
输出 actions 按 id 从1到10排列。preferred_id 选最高 action_score 的 ACTION（并列任选一个），无可推荐动作时为 null。
selection_reason 解释为什么这个完整动作优于其他候选。不得输出机器人代码或执行任何动作。
'''+wire.GRASP_ANGLE_PROMPT+'\n'+wire.GRIPPER_BOUNDARY_PROMPT
ACTION=copy.deepcopy(wire.ROTATION_SCHEMA)
ACTION['properties']['decision']['enum']=['ACTION','ABSTAIN']
ACTION['properties'].update(id=dict(type='integer',minimum=1,maximum=10),
    action_score=dict(type='integer',minimum=0,maximum=100),
    expected_effect=dict(type='string'),uncertainty=dict(type='string'))
ACTION['required']+=['id','action_score','expected_effect','uncertainty']
SCHEMA=dict(type='object',additionalProperties=False,properties=dict(
    actions=dict(type='array',minItems=10,maxItems=10,items=ACTION),
    preferred_id=dict(type=['integer','null'],minimum=1,maximum=10),
    selection_reason=dict(type='string')),required=['actions','preferred_id','selection_reason'])


def prepare():
    OUT.mkdir(parents=True,exist_ok=False)
    old=wire.read(SOURCE/'output.json')
    for name in ('current.png','goal.png'):shutil.copyfile(SOURCE/name,OUT/name)
    wire.save(OUT/'original_point_scores.json',old)
    points=[dict(id=p['id'],grasp_px=p['grasp_px']) for p in old['candidates']]
    random.Random(20261022).shuffle(points)
    wire.save(OUT/'fixed_points_input.json',points)
    size=list(Image.open(OUT/'current.png').size)
    body=wire.payload(OUT/'current.png',OUT/'goal.png',size)
    body['instructions']=PROMPT
    body['text']['format'].update(name='fixed_grasp_complete_actions',schema=SCHEMA)
    body['input'][0]['content'].append(dict(type='input_text',text='固定抓点（id仅为标识）：\n'+json.dumps(points,ensure_ascii=False)))
    wire.save(OUT/'prepared_request.json',body)
    (OUT/'prompt.txt').write_text(PROMPT+'\n\n'+'\n'.join(c['text'] for c in body['input'][0]['content'] if c['type']=='input_text'))
    wire.save(OUT/'protocol.json',dict(created_at=wire.utc(),source=str(SOURCE.resolve()),
        source_label=wire.read(SOURCE/'protocol.json')['source_label'],model=wire.MODEL,effort=wire.EFFORT,
        image_size=size,current_sha256=wire.sha(OUT/'current.png'),goal_sha256=wire.sha(OUT/'goal.png'),
        fixed_points=points,input_order_seed=20261022,old_scores_supplied=False,old_reasons_supplied=False,
        historical_actions_supplied=False,old_AB_answers_supplied=False,
        alternative_actions_from_same_current_state=True,robot_execution=False,
        host_geometry_or_depth_validation=False,retries=0))
    shutil.copyfile(__file__,OUT/'experiment_source.py')
    print('PREPARED',OUT,flush=True)


def validate(output):
    wire.jsonschema.validate(output,SCHEMA)
    protocol=wire.read(OUT/'protocol.json')
    fixed={p['id']:p['grasp_px'] for p in protocol['fixed_points']}
    actions=output['actions']
    assert [a['id'] for a in actions]==list(range(1,11)), 'one result per fixed point, ordered by ID'
    for action in actions:
        assert action['grasp_px']==fixed[action['id']], 'model changed fixed Pa'
        plan={k:action[k] for k in wire.ROTATION_SCHEMA['required']}
        if action['decision']=='ABSTAIN':
            assert action['action_score']==0
            plan['grasp_px']=None  # Preserve fixed Pa in output, use legacy semantic validator for the rest.
        wire.validate_plan(plan,protocol['image_size'],require_grasp_rotation=True)
        if action['decision']=='ACTION':assert action['target_px']!=action['grasp_px']
    eligible=[a for a in actions if a['decision']=='ACTION']
    if eligible:
        chosen=next(a for a in eligible if a['id']==output['preferred_id'])
        assert chosen['action_score']==max(a['action_score'] for a in eligible)
    else:assert output['preferred_id'] is None


def run():
    output,metrics=direct_once(OUT,wire.read(OUT/'prepared_request.json'),wire.auth_headers(),240)
    wire.save(OUT/'output.json',output)
    validate(output)
    print(json.dumps(output,ensure_ascii=False),flush=True)
    print('MODEL_SECONDS',metrics['wall_latency_s'],flush=True)
    report()


def report():
    output=wire.read(OUT/'output.json');validate(output)
    metrics=wire.read(OUT/'model_metrics.json');protocol=wire.read(OUT/'protocol.json')
    point_latency=wire.read(SOURCE/'model_metrics.json')['wall_latency_s']
    wire.save(OUT/'stage_timing.json',dict(point_generation_s=point_latency,
        action_completion_s=metrics['wall_latency_s'],
        two_stage_model_total_s=point_latency+metrics['wall_latency_s']))
    old={p['id']:p for p in wire.read(OUT/'original_point_scores.json')['candidates']}
    ranked=sorted(output['actions'],key=lambda a:(-a['action_score'],a['id']))
    ranks={a['id']:i+1 for i,a in enumerate(ranked)}
    rows=[];cards=[];data=[]
    sheet=Image.new('RGB',(1280,5*420),'#eef2f7');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    for i,a in enumerate(ranked):
        identifier=a['id'];folder=OUT/f'candidate_{identifier:02d}';folder.mkdir(exist_ok=True)
        plan={k:a[k] for k in wire.ROTATION_SCHEMA['required']}
        if plan['decision']=='ABSTAIN':plan['grasp_px']=None
        wire.save(folder/'plan.json',plan)
        wire.overlay(OUT/'current.png',plan,folder/'overlay.png',f"#{identifier} | ACTION SCORE {a['action_score']} | NOT EXECUTED")
        im=Image.open(folder/'overlay.png');im.thumbnail((630,355));x=(i%2)*640;y=(i//2)*420
        sheet.paste(im,(x+5,y+5))
        draw.text((x+12,y+365),f"Rank {i+1} | ID {identifier} | Point {old[identifier]['score']} -> Action {a['action_score']}",font=font,fill='#172033')
        draw.text((x+12,y+392),f"Pa {a['grasp_px']} -> Pb {a['target_px']}",font=font,fill='#172033')
        row=dict(**a,action_rank=ranks[identifier],old_point_score=old[identifier]['score'],old_point_rank=identifier,
                 part=old[identifier]['part'],image=f'candidate_{identifier:02d}/overlay.png')
        data.append(row)
        rows.append(f'''<tr data-id="{identifier}" tabindex="0"><td>{ranks[identifier]}</td><td>#{identifier}</td><td>{old[identifier]['score']}</td><td><b>{a['action_score']}</b></td><td>{a['grasp_px']} → {a['target_px']}</td><td>{a['grasp_angle_deg']}°</td><td>{html.escape(a['short_reason'])}</td></tr>''')
        cards.append(f'''<button class="card" data-id="{identifier}"><img src="candidate_{identifier:02d}/overlay.png"><strong>完整动作第 {ranks[identifier]} 名 · 点 #{identifier} · {a['action_score']} 分</strong><span>{html.escape(a['short_reason'])}</span></button>''')
    sheet.save(OUT/'all_actions.png')
    wire.save(OUT/'comparison.json',data)
    with (OUT/'comparison.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['point_id','old_point_rank','old_point_score','action_rank','action_score','Pa','Pb','grasp_angle_deg','short_reason','expected_effect','uncertainty'])
        for a in data:writer.writerow([a['id'],a['old_point_rank'],a['old_point_score'],a['action_rank'],a['action_score'],a['grasp_px'],a['target_px'],a['grasp_angle_deg'],a['short_reason'],a['expected_effect'],a['uncertainty']])
    preferred=output['preferred_id']
    rank_change=f"原抓点第一名 #1：{old[1]['score']} 分；补全动作后排第 {ranks[1]}，{next(a['action_score'] for a in ranked if a['id']==1)} 分。原抓点第三名 #3：{old[3]['score']} 分；补全动作后排第 {ranks[3]}，{next(a['action_score'] for a in ranked if a['id']==3)} 分。"
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>10 个固定抓点 → 完整动作</title>
<style>body{font-family:system-ui,sans-serif;background:#f3f5f9;color:#182235;margin:24px}p{line-height:1.65}h1{margin-bottom:10px}.panel{background:white;padding:18px;border-radius:12px;margin-bottom:18px}.viewer{display:grid;grid-template-columns:3fr 1fr;gap:20px}.viewer img{width:100%;display:block}#detail{line-height:1.7}.metric{font-size:25px;font-weight:bold}.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{text-align:left;cursor:pointer;border:2px solid transparent;border-radius:10px;background:white;padding:12px;font:inherit}.card img{width:100%}.card span,.card strong{display:block;padding:7px 0}.card.active{border-color:#d08b00}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:12px;border-bottom:1px solid #ddd}tr[data-id]{cursor:pointer}tr.active{background:#fff2ce}a{color:#2059a3}button{padding:7px 14px;cursor:pointer}select{padding:8px}@media(max-width:900px){.viewer,.cards{grid-template-columns:1fr}}</style>
<h1>固定 10 个抓点，分别补全 Pa → Pb 并重评完整动作</h1><p>'''+html.escape(protocol['source_label'])+'''</p>
<p>10 个 Pa 原样锁定；输入仍为当前图＋目标图。旧分数、旧理由和 A/B 答案没有提供给模型，候选输入顺序已打乱。这里是同一状态下的十个备选动作，不是连续十步。</p>
<div class="panel"><div class="metric">本次补全动作耗时 '''+f"{metrics['wall_latency_s']:.2f} 秒"+''' · 首选点 #'''+str(preferred)+'''</div><p>'''+html.escape(output['selection_reason'])+'''</p><p>gpt-6-astra / high · 一次请求比较全部候选 · 无工具、无重试</p><p>上次生成抓点 '''+f"{point_latency:.2f} 秒；两阶段模型请求合计 {point_latency+metrics['wall_latency_s']:.2f} 秒"+'''。均不含拍照和机器人执行。</p></div>
<div class="panel"><p><b>'''+html.escape(rank_change)+'''</b></p><p>这一次完整动作评估重新选择了先前 A/B 偏好的下端正面区域；抓点单独评分与完整动作评分的排序不同。仍是单次模型判断，不代表实测效果。</p></div>
<div class="panel"><p>选择候选：<select id="candidate">'''+''.join(f'<option value="{a["id"]}">点 #{a["id"]} · 完整动作第 {a["action_rank"]} 名 · {a["action_score"]} 分</option>' for a in data)+'''</select> <button id="toggle">切换原图 / 动作</button></p>
<div class="viewer"><div><img id="preview" src="current.png"><p>红点 Pa：抓点；青点 Pb：终点；黄色箭头：拉动；绿色：抓取前夹爪开合轴。</p></div><div id="detail"></div></div></div>
<div class="panel"><h2>抓点评分 → 完整动作评分</h2><table><thead><tr><th>新排名</th><th>固定点 ID / 旧排名</th><th>旧抓点分</th><th>完整动作分</th><th>Pa → Pb</th><th>夹爪角度</th><th>理由</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<div class="cards">'''+''.join(cards)+'''</div><div class="panel" style="margin-top:18px"><p>分数均为模型主观优先级，抓点分与完整动作分的评价对象不同，不代表成功率或实测收益。此次仅核对 JSON、Pa 固定、像素范围和参数格式，没有进行深度、IK、碰撞或毫米位移校验，也没有执行机器人动作。</p>
<p><a href="all_actions.png">十组动作总览 PNG</a> · <a href="comparison.csv">对照 CSV</a> · <a href="comparison.json">完整对照数据</a> · <a href="output.json">模型原始答案</a> · <a href="prompt.txt">完整 prompt</a> · <a href="model_metrics.json">耗时</a> · <a href="protocol.json">实验协议</a> · <a href="goal.png">目标图</a></p></div>
<script>const actions='''+json.dumps(data,ensure_ascii=False).replace('<','\\u003c')+''';let selected=null;let raw=false;
function select(id){selected=actions.find(a=>a.id===id);document.getElementById('candidate').value=id;document.getElementById('preview').src=raw?'current.png':selected.image;document.querySelectorAll('[data-id]').forEach(e=>e.classList.toggle('active',Number(e.dataset.id)===id));const d=document.getElementById('detail');d.replaceChildren();for(const text of [`点 #${id} · ${selected.part}`,`旧抓点：第 ${selected.old_point_rank} 名 / ${selected.old_point_score} 分`,`完整动作：第 ${selected.action_rank} 名 / ${selected.action_score} 分`,`Pa ${selected.grasp_px} → Pb ${selected.target_px}`,`抓前夹爪方向：${selected.grasp_angle_deg}°`, `理由：${selected.short_reason}`,`预期效果：${selected.expected_effect}`,`不确定性：${selected.uncertainty}`]){let p=document.createElement('p');p.textContent=text;d.appendChild(p);}}
document.getElementById('candidate').onchange=e=>select(Number(e.target.value));document.getElementById('toggle').onclick=()=>{raw=!raw;select(selected.id)};
document.querySelectorAll('[data-id]').forEach(e=>{e.onclick=()=>select(Number(e.dataset.id));e.onkeydown=event=>{if(event.key==='Enter'){select(Number(e.dataset.id))}}});select('''+str(preferred or ranked[0]['id'])+''');</script></html>'''
    (OUT/'index.html').write_text(page)
    print('REPORT',(OUT/'index.html').resolve(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','report'])
    globals()[parser.parse_args().command]()
