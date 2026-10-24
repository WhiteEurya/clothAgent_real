"""Experiment 3: same clean RGB, free one-shot versus 30 fixed actions."""
from pathlib import Path
import argparse
import copy
import html
import json
import math
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as wire
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once

BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/'reports/grasp_point_scores_2325_round002_20261022'
OUT=BASE/'reports/free_vs_candidate_ranking_20261022'
COMMON='''你是单臂衣物展开的单步规划器。输入只有一张当前 RGB。
任务：展平衣服，选择最可能改善当前衣物展开状态的一个动作。目标是展开主要衣身和袖部、减少折叠与遮挡；允许整体平移或旋转、普通小褶皱，不追求背景或绝对桌面位置一致。
仅支持现有 lift_pull_place：抓取可见布料、抬升约50毫米、平移10–150毫米、放下。
尽量抓取清楚可见、可夹住的衣物边缘或角点，优先能带动展开的外露边角，避免抓衣身中部或多层堆叠处；不要为抓边角而猜测隐藏角点。
Pa 必须落在当前图可见、可抓的布料上；Pb 应在当前可见工作台内，选择有助展开的方向，不猜隐藏衣角或布层。
像素坐标以当前原图左上角为原点，x向右、y向下，必须为整数。
本次没有深度输入；坐标仅为待 Host 检查的像素建议，不声称已通过实测深度、IK或安全检查，不臆造像素到毫米的换算。
只输出指定 JSON。short_reason 最多一句中文，解释为什么选择这个动作；uncertainty 简短说明主要不确定性。
'''+wire.GRASP_ANGLE_PROMPT+wire.GRIPPER_BOUNDARY_PROMPT
FREE='''自主选择 Pa、Pb，分别输出 grasp_px、target_px。
只有当前衣物已达到主要部位展开状态才输出 DONE；无法提出有依据的单步动作时输出 ABSTAIN。
ACTION 时 motion_type 为 lift_pull_place；DONE/ABSTAIN 时 grasp_px、target_px、motion_type、grasp_angle_deg 全为 null。
'''
RANK='''你只能从文字列表给出的30个固定候选动作中选择，不能新增、替换或移动任何 Pa 或 Pb，也不能改变动作长度或方向。
这些动作是同一当前状态下的互斥备选，不是连续执行。图像没有编号或箭头，通过文字坐标定位。
编号和列表顺序只是标识，不表示质量或优先级。给每个候选一个0–100整数 score，评价这个完整动作作为下一步的预期展开价值，不是成功概率，分数不要求总和100。
scores 中每个候选恰好出现一次。selected_id 必须是最高分候选之一，只选择一个。
可为选中动作输出抓前 grasp_angle_deg（或 null 沿用 Home），不能持布旋转。
'''
A_SCHEMA=copy.deepcopy(wire.ROTATION_SCHEMA)
A_SCHEMA['properties']['uncertainty']=dict(type='string')
A_SCHEMA['required'].append('uncertainty')


def b_schema(ids):
    item=dict(type='object',additionalProperties=False,properties=dict(
        id=dict(type='string',enum=ids),score=dict(type='integer',minimum=0,maximum=100)),required=['id','score'])
    return dict(type='object',additionalProperties=False,properties=dict(
        scores=dict(type='array',items=item,minItems=len(ids),maxItems=len(ids)),
        selected_id=dict(type='string',enum=ids),
        grasp_angle_deg=copy.deepcopy(wire.ROTATION_SCHEMA['properties']['grasp_angle_deg']),
        short_reason=dict(type='string'),uncertainty=dict(type='string')),
        required=['scores','selected_id','grasp_angle_deg','short_reason','uncertainty'])


def prepare():
    OUT.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(SOURCE/'current.png',OUT/'current.png')
    assert wire.sha(OUT/'current.png')=='a522f3b0f0c23266c4f999e189d7d235c9821d617bc7e591f8acae58630b12fa'
    old=wire.read(SOURCE/'output.json')['candidates']
    # Seeded relabeling removes the old score/rank order and E/M semantic labels.
    rng=random.Random(20261023);rng.shuffle(old)
    points=[dict(id=f'P{i:02d}',source_id=p['id'],grasp_px=p['grasp_px']) for i,p in enumerate(old,1)]
    generated=[]
    for p in points:
        for theta,dx,dy in [(0,120,0),(180,-120,0),(270,0,-120)]:
            pa=p['grasp_px'];pb=[pa[0]+dx,pa[1]+dy]
            assert 0<=pb[0]<1280 and 0<=pb[1]<720
            generated.append(dict(point_id=p['id'],source_id=p['source_id'],grasp_px=pa,target_px=pb,
                                  theta_deg=theta,displacement=[dx,dy],length_px=120))
    rng.shuffle(generated)
    for i,c in enumerate(generated,1):c['id']=f'C{i:02d}'
    assert len(generated)==30
    model_candidates=[{k:c[k] for k in ('id','point_id','grasp_px','target_px','displacement','theta_deg','length_px')} for c in generated]
    wire.save(OUT/'points.json',points);wire.save(OUT/'candidates.json',generated)
    wire.save(OUT/'model_candidates.json',model_candidates)
    for v,tail,schema in [('A',FREE,A_SCHEMA),('B',RANK,b_schema([c['id'] for c in generated]))]:
        text='当前图尺寸1280×720。请规划下一步展平动作。'
        if v=='B':text+='\n固定候选动作：\n'+json.dumps(model_candidates,ensure_ascii=False)
        body=wire.payload(OUT/'current.png',OUT/'current.png',[1280,720])
        body['instructions']=COMMON+tail
        body['input'][0]['content']=[dict(type='input_text',text=text),dict(type='input_image',image_url=wire.image_data(OUT/'current.png'),detail='high')]
        body['text']['format'].update(name='free_action' if v=='A' else 'candidate_action_ranking',schema=schema)
        wire.save(OUT/f'prepared_{v}.json',body)
        (OUT/f'prompt_{v}.txt').write_text(body['instructions']+'\n\n'+text)
    # Review-only candidate markers: never included in either request.
    image=Image.open(OUT/'current.png').convert('RGB');d=ImageDraw.Draw(image)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',20)
    for p in points:
        x,y=p['grasp_px'];label=p['id']+(' / E' if p['source_id']==1 else ' / M' if p['source_id']==3 else '')
        d.ellipse((x-4,y-4,x+4,y+4),fill='white',outline='black',width=1)
        d.text((x+8,y-24),label,font=font,fill='white',stroke_width=2,stroke_fill='black')
    image.save(OUT/'candidate_points_review.png')
    wire.save(OUT/'protocol.json',dict(created_at=wire.utc(),source_run='2026-10-21 23:25 run round_002 after',
        model=wire.MODEL,effort=wire.EFFORT,source_sha256=wire.sha(OUT/'current.png'),image_count=1,
        annotations_supplied=False,goal_image_supplied=False,old_scores_or_reasons_supplied=False,
        E=[795,606],M=[702,626],points=10,directions_deg=[0,180,270],candidate_length_px=120,
        candidate_count=30,candidate_seed=20261023,edge_priority_shared=True,pregrasp_rotation_shared=True,
        repetitions_per_version=1,order=['A','B'],retries=0,robot_execution=False,
        reason_request='generic choice reason in both arms; no prompt demanding fold-relation explanations',
        online_prompt_adaptation='One current image and textual flattening objective; no right-view/goal image. Preserves edge priority, pregrasp rotation, primitive and boundary rules. Not a byte-identical live-system replay.',
        ranking_restriction='B fixed 30 actions, scores all, chooses maximum; A generates free coordinates/length/direction and may DONE/ABSTAIN.',
        geometry_checks='Image bounds only for proposed endpoints; no depth/IK/collision/physical certification.',
        candidate_source=str(SOURCE),candidate_creation_is_reused=True,
        historical_candidate_generation_s=wire.read(SOURCE/'model_metrics.json')['wall_latency_s'],
        interpretation='M is a lower front folded edge, not a geometrical cloth-center GT. Classify each new Pa from actual image, not nearest-label alone. One sample each, no physical outcomes; different action spaces and output token budgets.'))
    shutil.copyfile(__file__,OUT/'experiment_source.py')
    print('PREPARED: 10 Pa x 3 directions; same unmarked image for A and B.',flush=True)


def selected_plan(version,output):
    if version=='A':
        wire.jsonschema.validate(output,A_SCHEMA)
        plan={k:output[k] for k in wire.ROTATION_SCHEMA['required']}
    else:
        candidates={c['id']:c for c in wire.read(OUT/'candidates.json')}
        wire.jsonschema.validate(output,b_schema(list(candidates)))
        scores=output['scores'];assert len({r['id'] for r in scores})==len(candidates)
        assert {r['id'] for r in scores}==set(candidates)
        assert next(r['score'] for r in scores if r['id']==output['selected_id'])==max(r['score'] for r in scores)
        c=candidates[output['selected_id']]
        plan=dict(decision='ACTION',grasp_px=c['grasp_px'],target_px=c['target_px'],motion_type='lift_pull_place',
                  grasp_angle_deg=output['grasp_angle_deg'],short_reason=output['short_reason'])
    wire.validate_plan(plan,[1280,720],require_grasp_rotation=True)
    if plan['decision']=='ACTION':assert plan['grasp_px']!=plan['target_px']
    return plan


def run():
    headers=wire.auth_headers()
    for v in ('A','B'):
        folder=OUT/v
        if (folder/'attempt.json').exists():
            print('PRESERVE existing attempt',v,flush=True);continue
        print('REQUEST',v,flush=True)
        try:
            output,metrics=direct_once(folder,wire.read(OUT/f'prepared_{v}.json'),headers,300)
            wire.save(folder/'output.json',output)
            plan=selected_plan(v,output);wire.save(folder/'plan.json',plan)
            print(json.dumps(dict(version=v,seconds=metrics['wall_latency_s'],plan=plan,uncertainty=output['uncertainty']),ensure_ascii=False),flush=True)
        except Exception as error:
            wire.save(folder/'validation_or_request_error.json',dict(error=f'{type(error).__name__}: {error}'))
            print('FAILED',v,type(error).__name__,str(error),flush=True)
    report()


def report():
    results={};cards=[];text=['实验 3：自由 one-shot 与候选 ranking。','使用2026-10-21 23:25 run的round_002动作后图像；两版输入完全相同的干净原图，无编号、箭头或目标图。',
        'A自由生成Pa、Pb；B使用已有10个固定抓点，每点向右、向左、向上各120像素，共30个候选，只能选其中一个。两版共同保留外露边角优先、仅抓前调整夹爪的规则；各调用一次gpt-6-astra / high。']
    for v in ('A','B'):
        folder=OUT/v
        if not (folder/'output.json').exists() or (folder/'validation_or_request_error.json').exists():
            text.append(v+'尚无通过验证的结果，请查阅请求/验证记录。');continue
        output=wire.read(folder/'output.json');plan=selected_plan(v,output);metrics=wire.read(folder/'model_metrics.json')
        results[v]=dict(plan=plan,uncertainty=output['uncertainty'],seconds=metrics['wall_latency_s'],usage=metrics.get('usage'))
        if plan['decision']=='ACTION':
            results[v]['length_px']=math.dist(plan['grasp_px'],plan['target_px'])
            results[v]['distance_to_E_px']=math.dist(plan['grasp_px'],[795,606])
            results[v]['distance_to_M_px']=math.dist(plan['grasp_px'],[702,626])
        wire.overlay(OUT/'current.png',plan,folder/'overlay.png',f'{v} | '+('FREE ONE-SHOT' if v=='A' else 'FIXED CANDIDATE RANKING')+' | REVIEW ONLY')
        desc=f'{v}：Pa {plan["grasp_px"]} → Pb {plan["target_px"]}；抓前夹爪角度 {plan["grasp_angle_deg"]}°；请求耗时 {metrics["wall_latency_s"]:.2f} 秒。'
        text += [desc,'模型原始理由：'+output['short_reason'],'模型不确定性：'+output['uncertainty']]
        cards.append(f'<section><h2>{html.escape(desc)}</h2><img src="{v}/overlay.png"><p>{html.escape(output["short_reason"])}</p><p>不确定性：{html.escape(output["uncertainty"])}</p><a href="{v}/output.json">完整答案</a> · <a href="{v}/model_metrics.json">耗时与 token</a></section>')
    ranking=[]
    if 'B' in results:
        candidates={c['id']:c for c in wire.read(OUT/'candidates.json')}
        output=wire.read(OUT/'B/output.json')
        ranking=sorted([{**candidates[s['id']],'score':s['score']} for s in output['scores']],key=lambda r:-r['score'])
        results['B']['selected_id']=output['selected_id']
        results['B']['selected_source_id']=candidates[output['selected_id']]['source_id']
        for i,c in enumerate(ranking[:5],1):
            plan=dict(decision='ACTION',grasp_px=c['grasp_px'],target_px=c['target_px'],motion_type='lift_pull_place',short_reason='ranking',grasp_angle_deg=None)
            wire.overlay(OUT/'current.png',plan,OUT/f'rank_{i}.png',f'RANK {i} | {c["id"]} | {c["score"]}/100 | REVIEW ONLY')
        wire.save(OUT/'ranking.json',ranking)
    assessment_path=OUT/'assessment.json'
    assessment=wire.read(assessment_path) if assessment_path.exists() else None
    if assessment:text+=['结果解读：',*assessment['paragraphs']]
    text += ['耗时只统计本轮模型请求，不含拍照和机器人执行；B复用了已有抓点，本次没有重新生成抓点。',
        '这是单图适配的one-shot范式比较，不是线上多视角加目标图系统的完整复现。A可自由选择位移，B限定三个方向和120像素，且B额外输出30项评分，因此差异包含搜索空间和输出负担的影响。',
        '每版仅一次，评分不是成功率，理由也不是布层关系的真值。M是先前定义的下端正面折边，不应直接等同于衣服几何中心。未执行实机动作；未进行深度、IK或碰撞验证。']
    (OUT/'report.txt').write_text('\n\n'.join(text)+'\n')
    wire.save(OUT/'summary.json',dict(results=results,assessment=assessment))
    rows=''.join(f'<tr><td>{i}</td><td>{r["id"]}</td><td>{r["point_id"]}'+(' / E' if r['source_id']==1 else ' / M' if r['source_id']==3 else '')+f'</td><td>{r["grasp_px"]} → {r["target_px"]}</td><td>{r["theta_deg"]}°</td><td>{r["score"]}</td></tr>' for i,r in enumerate(ranking,1))
    top=''.join(f'<figure><img src="rank_{i}.png"><figcaption>第{i}名 · {r["id"]} · {r["score"]}分</figcaption></figure>' for i,r in enumerate(ranking[:5],1))
    analysis_html=''.join('<p>'+html.escape(p)+'</p>' for p in (assessment['paragraphs'] if assessment else ['等待人工检查动作位置和理由。']))
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>实验3 · 自由动作 vs 候选 ranking</title>
<style>body{font-family:system-ui,sans-serif;background:#f3f5f9;color:#182235;margin:24px}p{line-height:1.7}section{background:white;padding:18px;border-radius:12px;margin:16px 0}.pair,.top{display:grid;grid-template-columns:1fr 1fr;gap:18px}img{width:100%}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}a{color:#2059a3}figure{margin:0}pre{white-space:pre-wrap;line-height:1.7}@media(max-width:900px){.pair,.top{grid-template-columns:1fr}}</style>
<h1>实验3：自由 one-shot vs 候选 ranking</h1><p>2026-10-21 23:25 run · round_002 动作后。离线比较，各请求一次，无重试。</p>
<section><h2>本次判断</h2>'''+analysis_html+'''<a href="report.txt">纯文字报告</a></section>
<div class="pair">'''+''.join(cards)+'''</div><div class="pair"><section><h2>两版实际使用同一张干净图</h2><img src="current.png"><p>没有编号、箭头、目标图或旧评分。仅B额外得到候选坐标文字。</p></section><section><h2>候选抓点分布（仅供你看）</h2><img src="candidate_points_review.png"><p>10个固定抓点，每点3个方向：右、左、上；长度均120px。E/M标签未提供给模型。</p></section></div>
<section><h2>B 前五名（展示箭头未输入模型）</h2><div class="top">'''+top+'''</div></section>
<section><h2>B 全部30项评分</h2><table><tr><th>排名</th><th>动作</th><th>抓点</th><th>Pa → Pb</th><th>方向</th><th>分数</th></tr>'''+rows+'''</table></section>
<section><h2>完整文字报告</h2><pre>'''+html.escape('\n\n'.join(text))+'''</pre><a href="prompt_A.txt">A prompt</a> · <a href="prompt_B.txt">B prompt</a> · <a href="protocol.json">实验协议</a> · <a href="candidates.json">候选来源与坐标</a> · <a href="summary.json">汇总</a></section></html>'''
    (OUT/'index.html').write_text(page)
    print('REPORT',OUT/'index.html',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','report'])
    globals()[parser.parse_args().command]()
