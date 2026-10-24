"""One offline Codex-model request for ten scored grasp points, no endpoints."""
from pathlib import Path
import argparse
import csv
import html
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as wire
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once

SOURCE = Path('reports/one_shot_goal_full_episode_v1_20261022_120645_870170377')
OUT = Path('reports/grasp_point_scores_round009_20261022')
ROUND = 'round_009'
SOURCE_LABEL = '2026-10-22 12:06 run · round_009（第 10 个动作）后的图，与第一次 A/B 实验的当前图完全相同。'
REFERENCE = Path('reports/one_shot_state_ab_20261022/inputs/current.png')
PROMPT = '''你是单臂衣物展开规划器。本次只做候选抓点生成与评分，不执行动作。
第一张是当前 RGB（某次真实动作后的状态），第二张是展平目标 RGB。
任务：根据当前衣物的折叠、遮挡和可见边缘，提出下一步应考虑抓取的 10 个不同候选点，并按你自己的判断打分。
目标是衣物主要部位的展开布局和折叠/遮挡状态接近目标，允许整体平移或旋转与普通小褶皱，不要求背景或绝对桌面位置一致。
已有动作能力是单臂 lift_pull_place，抓前可调整夹爪方向，抓住后保持方向、抬升约50毫米、平移10–150毫米再放下。
本次只输出抓点 Pa：不要输出终点 Pb、拉动向量、动作路径或夹爪角度。
每个 grasp_px 都是第一张原图的整数像素坐标，原点左上角，x 向右、y 向下。必须位于可见布料上，不得借用目标图坐标。
优先考虑清楚可见、可夹住、有助展开的外露边缘或角点，避免无依据地猜测隐藏角点或布层。
给出不同的候选位置，尽量覆盖有意义的替代抓取部位，不要用同一个位置的微小抖动凑够十个；也不要为了分散而选择背景。
score 为 0–100 的整数，表达“在当前状态下，这个点有多值得作为下一步优先抓点”。
综合你认为重要的因素：可见性、可夹取性、减少折叠/遮挡的潜力、相对其他部位的优先级，以及叠层不确定性。
评分不是单纯的布料检测置信度，不是已验证的成功概率；各点独立评分，不要求总和为100，不强制等差或满分。
按 score 从高到低输出，id 为1到10。part 简述可见局部，reason 用一句中文解释该评分，低分说明为何不优先。
summary 简述整体优先级判断；uncertainty 简述仅凭RGB无法确定的事项。
没有深度、接触或机器人检查输入，不声称已验证抓取成功、可达或安全。只输出指定JSON。
'''
SCHEMA = dict(type='object', additionalProperties=False,
    properties=dict(summary=dict(type='string'), uncertainty=dict(type='string'),
        candidates=dict(type='array',minItems=10,maxItems=10,items=dict(type='object',additionalProperties=False,
            properties=dict(id=dict(type='integer',minimum=1,maximum=10),
                grasp_px=dict(type='array',items=dict(type='integer'),minItems=2,maxItems=2),
                score=dict(type='integer',minimum=0,maximum=100),part=dict(type='string'),reason=dict(type='string')),
            required=['id','grasp_px','score','part','reason']))),
    required=['summary','uncertainty','candidates'])


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    paths=dict(current=SOURCE/ROUND/'after/rgb.png',goal=SOURCE/'goal/target.png')
    for name,path in paths.items():shutil.copyfile(path,OUT/f'{name}.png')
    assert wire.sha(paths['current'])==wire.sha(REFERENCE)
    with Image.open(OUT/'current.png') as im:size=list(im.size)
    body=wire.payload(OUT/'current.png',OUT/'goal.png',size)
    body['instructions']=PROMPT
    body['text']['format'].update(name='ten_scored_grasp_points',schema=SCHEMA)
    wire.save(OUT/'prepared_request.json',body)
    (OUT/'prompt.txt').write_text(PROMPT+'\n'+body['input'][0]['content'][0]['text'])
    wire.save(OUT/'protocol.json',dict(created_at=wire.utc(),experiment='ten_scored_grasp_points',
        source=str(SOURCE.resolve()),round=ROUND,source_label=SOURCE_LABEL,current_observation='after',image_size=size,
        input_images={n:dict(source=str(p.resolve()),sha256=wire.sha(p)) for n,p in paths.items()},
        model=wire.MODEL,effort=wire.EFFORT,transport='Existing Codex Responses backend, single tool-free model request',
        prior_AB_answers_supplied=False,previous_action_supplied=False,
        score_definition='0–100 subjective next-grasp priority, not calibrated success probability',
        endpoint_generation=False,robot_execution=False,retries=0,
        visualization='Fixed linear light-to-dark blue scale 0–100 at candidate points only; no interpolation.'))
    shutil.copyfile(__file__,OUT/'experiment_source.py')
    print('PREPARED',OUT,flush=True)


def validate(output):
    wire.jsonschema.validate(output,SCHEMA)
    size=wire.read(OUT/'protocol.json')['image_size']
    points=output['candidates']
    assert [p['id'] for p in points]==list(range(1,11)), 'IDs must match ranked output'
    assert [p['score'] for p in points]==sorted([p['score'] for p in points],reverse=True)
    assert len({tuple(p['grasp_px']) for p in points})==10, 'duplicate points'
    for p in points:
        x,y=p['grasp_px'];assert 0<=x<size[0] and 0<=y<size[1]
        assert p['reason'].strip() and p['part'].strip()


def run():
    output,metrics=direct_once(OUT,wire.read(OUT/'prepared_request.json'),wire.auth_headers(),180)
    wire.save(OUT/'output.json',output)
    validate(output)
    print(json.dumps(output,ensure_ascii=False),flush=True)
    print('MODEL_SECONDS',metrics['wall_latency_s'],flush=True)
    report()


def color(score):
    return tuple(round(a+(b-a)*score/100) for a,b in zip((222,235,247),(8,48,107)))


def report():
    output=wire.read(OUT/'output.json');validate(output)
    metrics=wire.read(OUT/'model_metrics.json');points=output['candidates']
    image=Image.open(OUT/'current.png').convert('RGB');w,h=image.size
    canvas=Image.new('RGB',(w,h+110),'white');canvas.paste(image,(0,0));draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    shapes=[];rows=[];positions=[]
    for p in points:
        x,y=p['grasp_px'];score=p['score'];c=color(score);hexcolor='#%02x%02x%02x'%c
        draw.ellipse((x-11,y-11,x+11,y+11),fill=c,outline='white',width=2)
        label=f"{p['id']}: {score}"
        tw=draw.textbbox((0,0),label,font=small)[2]+10;th=27
        candidates=[(x+17,y-30),(x+17,y+5),(x-tw-17,y-30),(x-tw-17,y+5),(x+17,y+35),(x-tw-17,y-60)]
        def overlap(a,b):return a[0]<b[2] and a[2]>b[0] and a[1]<b[3] and a[3]>b[1]
        lx,ly=candidates[0]
        for tx,ty in candidates:
            rect=(tx,ty,tx+tw,ty+th)
            if tx>=0 and ty>=0 and tx+tw<=w and ty+th<=h and not any(overlap(rect,r) for r in positions):
                lx,ly=tx,ty;break
        positions.append((lx,ly,lx+tw,ly+th))
        draw.line((x,y,lx+tw/2,ly+th/2),fill='white',width=1)
        draw.rounded_rectangle((lx,ly,lx+tw,ly+th),radius=4,fill=(255,255,255),outline=c,width=2)
        draw.text((lx+5,ly+3),label,font=small,fill=(16,30,48))
        title=html.escape(f"#{p['id']} · {score}/100 · {p['part']} · {p['reason']}")
        shapes.append(f'''<g class="candidate" data-id="{p['id']}" tabindex="0" role="button" aria-label="{title}"><title>{title}</title>
<circle cx="{x}" cy="{y}" r="11" fill="{hexcolor}" stroke="white" stroke-width="2"/>
<line x1="{x}" y1="{y}" x2="{lx+tw/2}" y2="{ly+th/2}" stroke="white"/>
<rect x="{lx}" y="{ly}" width="{tw}" height="{th}" rx="4" fill="white" stroke="{hexcolor}" stroke-width="2"/>
<text x="{lx+5}" y="{ly+20}" font-size="17" fill="#101e30">{label}</text></g>''')
        rows.append(f'''<tr id="point-{p['id']}" data-id="{p['id']}" tabindex="0"><td><span class="dot" style="background:{hexcolor}"></span>{p['id']}</td><td><b>{score}</b></td><td>{p['grasp_px']}</td><td>{html.escape(p['part'])}</td><td>{html.escape(p['reason'])}</td></tr>''')
    draw.text((25,h+12),'Codex grasp priority | 0-100 | Darker = higher | NOT success probability',font=font,fill='#182235')
    for x in range(500):draw.line((25+x,h+48,25+x,h+66),fill=color(x/499*100))
    draw.text((25,h+72),'0',font=small,fill='black');draw.text((490,h+72),'100',font=small,fill='black')
    draw.text((570,h+48),f"10 points | {metrics['wall_latency_s']:.2f} s | No endpoints",font=font,fill='#182235')
    canvas.save(OUT/'scored_grasp_points.png')
    (OUT/'points.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">' + ''.join(shapes)+'</svg>')
    with (OUT/'points.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['rank','x','y','score','part','reason'])
        for p in points:writer.writerow([p['id'],*p['grasp_px'],p['score'],p['part'],p['reason']])
    usage=metrics.get('usage') or {}
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>实验二 · 10 个抓点与模型评分</title>
<style>body{font-family:system-ui,sans-serif;background:#f3f5f9;color:#182235;margin:24px}h1{margin-bottom:10px}p{line-height:1.65}.panel{background:white;border-radius:12px;padding:18px;margin-bottom:18px}.views{display:grid;grid-template-columns:3fr 1fr;gap:18px}.view{position:relative}.view img{width:100%;display:block}.view svg{position:absolute;inset:0;width:100%;height:100%;font-family:Arial,sans-serif}aside img{width:100%}.candidate{cursor:pointer}.candidate.active circle{stroke:#ffb000;stroke-width:6}.candidate.active rect{stroke:#ffb000;stroke-width:3}button{padding:7px 14px;cursor:pointer}.legend{display:inline-block;width:300px;height:16px;background:linear-gradient(to right,rgb(222,235,247),rgb(8,48,107));vertical-align:middle;border:1px solid #ccd4df}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px;border-bottom:1px solid #e1e6ee}tr[data-id]{cursor:pointer}tr.active{background:#fff2ce}.dot{display:inline-block;width:16px;height:16px;border-radius:50%;margin-right:8px;border:1px solid #aab5c5}a{color:#2059a3}.metric{font-size:26px;font-weight:bold}#selected{min-height:48px}pre{white-space:pre-wrap} @media(max-width:850px){.views{grid-template-columns:1fr}aside{max-width:320px}}</style>
<h1>实验二：10 个候选抓点，按模型自己的优先级打分</h1>
<p>来源：'''+html.escape(wire.read(OUT/'protocol.json').get('source_label', SOURCE_LABEL))+'''</p>
<p>输入：当前图＋展平目标图；没有输入之前 A/B 的答案。只生成抓点，不生成终点、拉动箭头或夹爪角度。</p>
<div class="panel"><div class="views"><div><div class="view"><img src="current.png"><svg id="points" viewBox="0 0 '''+f'{w} {h}'+'''">'''+''.join(shapes)+'''</svg></div><p><button id="toggle">隐藏 / 显示抓点</button>　点标签：编号 : 分数</p><p>低分 0　<span class="legend"></span>　100 高分（颜色越深越优先）</p><div id="selected">点击图上抓点或下方行，查看对应理由。</div></div><aside><img src="goal.png"><p>展平目标，仅用于判断当前应优先抓哪里。</p><div class="metric">'''+f"{metrics['wall_latency_s']:.2f} 秒"+'''</div><p>gpt-6-astra / high<br>单次 Codex 模型请求，无工具、无重试</p><p>输入 token '''+str(usage.get('input_tokens'))+' · 输出 token '+str(usage.get('output_tokens'))+'''</p></aside></div></div>
<div class="panel"><p><b>模型整体判断：</b>'''+html.escape(output['summary'])+'''</p><p><b>模型不确定性：</b>'''+html.escape(output['uncertainty'])+'''</p></div>
<div class="panel"><table><thead><tr><th>排名</th><th>分数</th><th>抓点 (x,y)</th><th>局部</th><th>模型评分理由</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<div class="panel"><p>分数是模型对“这个点是否值得优先抓”的主观判断，不是实测成功率。只在这 10 个离散候选点上着色，没有推断整张图的稠密热力图。没有执行抓取，也没有做深度、IK 或碰撞验证。</p>
<a href="scored_grasp_points.png">下载标注图</a> · <a href="points.csv">10 点 CSV</a> · <a href="output.json">完整模型输出</a> · <a href="prompt.txt">完整 prompt</a> · <a href="model_metrics.json">耗时与 token</a> · <a href="protocol.json">来源与图像哈希</a></div>
<script>const points='''+json.dumps(points,ensure_ascii=False).replace('<','\\u003c')+''';
function select(id){document.querySelectorAll('[data-id]').forEach(e=>e.classList.toggle('active',Number(e.dataset.id)===id));const p=points.find(p=>p.id===id);document.getElementById('selected').textContent=`#${id} · ${p.score}/100 · (${p.grasp_px.join(', ')}) · ${p.part}：${p.reason}`;}
document.querySelectorAll('[data-id]').forEach(e=>{e.addEventListener('click',()=>select(Number(e.dataset.id)));e.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();select(Number(e.dataset.id));}});});
document.getElementById('toggle').onclick=()=>{let s=document.getElementById('points');s.style.display=s.style.display==='none'?'':'none';};select(1);</script></html>'''
    (OUT/'index.html').write_text(page)
    print('REPORT',(OUT/'index.html').resolve(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','report'])
    parser.add_argument('--case',choices=['round009','s2'],default='round009')
    args=parser.parse_args()
    if args.case=='s2':
        SOURCE=Path('reports/one_shot_goal_full_episode_v1_20261021_232513_053641787')
        OUT=Path('reports/grasp_point_scores_2325_round002_20261022')
        ROUND='round_002'
        SOURCE_LABEL='2026-10-21 23:25 run · round_002（第 3 个动作）后的图，与三组 A/B 实验的 S2 当前图完全相同。'
        REFERENCE=Path('reports/interrupted_state_ab_20261022/S2/inputs/current.png')
    globals()[args.command]()
