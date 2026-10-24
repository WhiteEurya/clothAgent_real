"""Paired offline representation experiment; never connects to the robot."""
from pathlib import Path
import argparse
import html
import json
import math
import shutil
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as wire
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once

BASE = Path(__file__).resolve().parents[1]
PREVIOUS = BASE / 'reports/em_fixed_pa_direction_ablation_20261022'
OUT = BASE / 'reports/action_overlay_bias_20261022'
PROMPT = '''你是单臂衣物展开动作评估器。观察当前图，比较文字指定的两个固定动作。
目标是展开衣身和袖部，减少主要折叠与遮挡；允许整体平移/旋转和普通小褶皱，但不要把单纯搬动当作展开。
图上的小编号 1、2 表示两个抓点 Pa，编号不表示优先级。坐标原点在左上，x 向右、y 向下。以文字坐标定义动作。
两个动作是同一初态的互斥备选，不是连续执行。均为 lift_pull_place：同一个固定 Home 夹爪姿态，抓取、抬升约50毫米、平移至 Pb、放下；不调整抓前角度，不持布旋转。
动作1和动作2，哪一个更有可能有效展开衣服？只能选择一个，并说明预计哪一处布料关系会发生改变。
请根据可见布料结构评价打开遮挡/折叠的可能性，同时考虑多层牵连、整体移动和新折叠风险。不要改动动作。
selected_action 只能是 "1" 或 "2"。reason 用中文简短解释为什么优于另一个动作。
expected_relation_change 必须具体说明哪一片/边缘相对于哪一片布料的遮挡、覆盖或折叠关系预计改变。
uncertainty 说明从这张图无法确定的关键事项。没有实机结果，不声称动作已经成功。
只输出指定 JSON，不调用工具。
'''
SCHEMA = dict(type='object', additionalProperties=False, properties={
    'selected_action': dict(type='string', enum=['1', '2']),
    **{k: dict(type='string') for k in ('reason', 'expected_relation_change', 'uncertainty')}
}, required=['selected_action', 'reason', 'expected_relation_change', 'uncertainty'])


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    previous = wire.read(PREVIOUS / 'summary.json')
    rows = {r['id']: r for r in wire.read(PREVIOUS / 'results.json')}
    actions = []
    for number, point in [('1', 'E'), ('2', 'M')]:
        r = rows[previous['best_by_point'][point]]
        actions.append(dict(id=number, source_id=r['id'], point=point,
                            Pa=r['grasp_px'], Pb=r['target_px'],
                            displacement=[b-a for a,b in zip(r['grasp_px'], r['target_px'])]))
    assert [a['source_id'] for a in actions] == ['E1', 'M5']
    assert wire.sha(PREVIOUS / 'current.png') == 'a522f3b0f0c23266c4f999e189d7d235c9821d617bc7e591f8acae58630b12fa'
    shutil.copyfile(PREVIOUS / 'current.png', OUT / 'current.png')
    raw = Image.open(OUT / 'current.png').convert('RGB')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
    for version in ('A', 'B'):
        im = raw.copy(); draw = ImageDraw.Draw(im)
        if version == 'A':
            for action in actions:
                pa, pb = action['Pa'], action['Pb']
                theta = math.atan2(pb[1]-pa[1], pb[0]-pa[0])
                draw.line([tuple(pa), tuple(pb)], fill='#ffdc00', width=5)
                wings = [(pb[0]-15*math.cos(theta+d), pb[1]-15*math.sin(theta+d)) for d in (-0.48, 0.48)]
                draw.polygon([tuple(pb), *wings], fill='#ffdc00')
        # Identical small Pa markers, drawn last in both versions.
        for action in actions:
            x,y = action['Pa']
            draw.ellipse((x-5,y-5,x+5,y+5), fill='white', outline='black', width=2)
            draw.text((x+10,y-26), action['id'], font=font, fill='white', stroke_width=2, stroke_fill='black')
        im.save(OUT / f'input_{version}.png')
    text_actions = [{k:a[k] for k in ('id','Pa','Pb','displacement')} for a in actions]
    text = '当前图尺寸 1280×720。两个动作长度均为120像素。候选动作（displacement = Pb − Pa）：\n' + json.dumps(text_actions, ensure_ascii=False)
    for version in ('A','B'):
        body = wire.payload(OUT / f'input_{version}.png', OUT / f'input_{version}.png', [1280,720])
        body['instructions'] = PROMPT
        body['input'][0]['content'] = [dict(type='input_text', text=text),
            dict(type='input_image', image_url=wire.image_data(OUT / f'input_{version}.png'), detail='high')]
        body['text']['format'].update(name='fixed_action_pair_choice', schema=SCHEMA)
        wire.save(OUT / f'prepared_{version}.json', body)
    aa = wire.read(OUT / 'prepared_A.json'); bb = wire.read(OUT / 'prepared_B.json')
    aa['input'][0]['content'][1]['image_url'] = bb['input'][0]['content'][1]['image_url'] = '<IMAGE>'
    assert aa == bb, 'Only image may differ'
    (OUT / 'prompt.txt').write_text(PROMPT+'\n'+text)
    wire.save(OUT / 'protocol.json', dict(created_at=wire.utc(), source_experiment=str(PREVIOUS),
        source_run='2026-10-21 23:25 round_002 after', actions=actions,
        model=wire.MODEL, effort=wire.EFFORT, repetitions_per_version=3,
        dispatch_order=['A','B','B','A','A','B'], sequential_requests=True,
        independent_contexts=True, retries_per_request=0, robot_execution=False,
        only_request_difference='input image: A has yellow arrows; B only Pa numbers',
        same_text_coordinates_in_both_versions=True, fixed_gripper='Home', lift_mm=50,
        goal_image_supplied=False, previous_scores_supplied=False,
        source_sha256=wire.sha(OUT / 'current.png'),
        image_sha256={v:wire.sha(OUT / f'input_{v}.png') for v in ('A','B')},
        limitation='One state, three fresh contexts per representation; fixed labels; stochastic/backend/cache effects possible. No physical outcomes. A reversal shows representation sensitivity, not proof of a perception mechanism.'))
    shutil.copyfile(__file__, OUT / 'experiment_source.py')
    print('PREPARED', OUT, flush=True)


def run():
    protocol = wire.read(OUT / 'protocol.json')
    headers = wire.auth_headers()
    counts = {'A':0,'B':0}
    for sequence, version in enumerate(protocol['dispatch_order'], 1):
        counts[version] += 1
        folder = OUT / f'{version}_{counts[version]:02d}'
        if (folder / 'attempt.json').exists():
            print('PRESERVED EXISTING ATTEMPT', folder.name, flush=True)
            continue
        print('REQUEST', sequence, folder.name, flush=True)
        try:
            output, metrics = direct_once(folder, wire.read(OUT / f'prepared_{version}.json'), headers, 240)
            wire.save(folder / 'output.json', output)
            print(json.dumps(dict(trial=folder.name, seconds=metrics['wall_latency_s'], **output), ensure_ascii=False), flush=True)
        except Exception as error:
            print('FAILED', folder.name, type(error).__name__, str(error), flush=True)
        report()
    report()


def report():
    protocol = wire.read(OUT / 'protocol.json')
    trials = []
    counts = {'A':0,'B':0}
    for sequence, version in enumerate(protocol['dispatch_order'], 1):
        counts[version] += 1
        name = f'{version}_{counts[version]:02d}'; folder = OUT / name
        if not (folder / 'model_metrics.json').exists(): continue
        metrics = wire.read(folder / 'model_metrics.json')
        row = dict(trial=name, sequence=sequence, version=version, pair=counts[version],
                   seconds=metrics.get('wall_latency_s'), usage=metrics.get('usage'), error=metrics.get('error'))
        if (folder / 'output.json').exists():
            output = wire.read(folder / 'output.json'); wire.jsonschema.validate(output, SCHEMA); row.update(output)
        trials.append(row)
    stats = {}
    for version in ('A','B'):
        valid = [r for r in trials if r['version']==version and 'selected_action' in r]
        stats[version] = dict(valid=len(valid), E=sum(r['selected_action']=='1' for r in valid),
            M=sum(r['selected_action']=='2' for r in valid),
            choices=['E' if r['selected_action']=='1' else 'M' for r in valid],
            mean_seconds=statistics.mean(r['seconds'] for r in valid) if valid else None)
    complete = all(stats[v]['valid']==3 for v in ('A','B'))
    majority_flip = complete and ((stats['A']['E'] > stats['A']['M']) != (stats['B']['E'] > stats['B']['M']))
    stable_flip = complete and ((stats['A']['E']==3 and stats['B']['M']==3) or (stats['A']['M']==3 and stats['B']['E']==3))
    same_unanimous = complete and (stats['A']['E']==stats['B']['E']==3 or stats['A']['M']==stats['B']['M']==3)
    if stable_flip:
        conclusion = '两种表示在本图的 3 次判断中各自一致，但选择完全翻转：观察到了对动作表示的敏感性。它可能影响状态理解，也可能影响注意力或动作评价；仅凭选择翻转还不能定位机制。'
    elif same_unanimous:
        conclusion = '两种表示的 3 次判断全部选择同一个动作，本次没有观察到箭头导致的排序翻转；不能据此证明其他图片或箭头样式没有影响。'
    elif majority_flip:
        conclusion = '多数选择发生翻转：A 更常选 E，B 更常选 M。B 内部也出现选择波动，没有两版各自 3/3 一致的完全翻转。这是表示敏感性的线索，但每版仅三次，不能确认稳定效应，更不能直接断言状态感知已被改变。' if stats['A']['E']>stats['A']['M'] else '多数选择发生翻转，但存在重复采样波动；每版仅三次，不能确认稳定效应或定位到状态感知机制。'
    elif complete:
        conclusion = '本次选择存在波动，但没有出现两种表示各自 3/3 一致的完全翻转。样本太少，不能确认稳定的表示效应。'
    else:
        conclusion = '实验尚未得到每版 3 个有效回答，请查看各次结果与错误记录。'
    summary = dict(complete=complete, versions=stats, majority_reversal=majority_flip, stable_complete_reversal=stable_flip,
        conclusion=conclusion, total_request_seconds=sum(r['seconds'] or 0 for r in trials))
    wire.save(OUT / 'results.json', trials); wire.save(OUT / 'summary.json', summary)
    esc = html.escape
    cards = []
    for row in trials:
        if 'selected_action' in row:
            winner = '1 / E 向右' if row['selected_action']=='1' else '2 / M 向左'
            content = f'<h3>{row["trial"]}：选择 {winner} · {row["seconds"]:.2f} 秒</h3>'
            for key,label in [('reason','原因'),('expected_relation_change','预计改变的布料关系'),('uncertainty','不确定性')]:
                content += f'<p><b>{label}：</b>{esc(row[key])}</p>'
        else:
            content = f'<h3>{row["trial"]} 请求失败</h3><p>{esc(str(row["error"]))}</p>'
        cards.append(f'<article>{content}<a href="{row["trial"]}/raw_output.txt">原始回答</a> · <a href="{row["trial"]}/model_metrics.json">耗时与 token</a></article>')
    table = ''.join(f'<tr><td>{v}</td><td>{s["valid"]}/3</td><td>{s["E"]}</td><td>{s["M"]}</td><td>{s["mean_seconds"]:.2f} 秒</td></tr>' if s['valid'] else f'<tr><td>{v}</td><td colspan="4">暂无有效结果</td></tr>' for v,s in stats.items())
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>实验 2 · 箭头表示对照</title>
<style>body{font-family:system-ui,sans-serif;background:#f3f5f9;color:#182235;margin:24px}p{line-height:1.65}section,article{background:white;padding:18px;border-radius:12px;margin:16px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}img{width:100%}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:12px;border-bottom:1px solid #ddd}a{color:#2059a3}.note{color:#735217}@media(max-width:900px){.pair{grid-template-columns:1fr}}</style>
<h1>实验 2：箭头是否改变 GPT 的动作选择？</h1>
<p>2026-10-21 23:25 run · round_002 动作后；固定实验 1 的两个最优动作。纯离线评估。</p>
<section><p><b>动作 1 / E：</b>(795,606) → (915,606)，向右 120 px。<br><b>动作 2 / M：</b>(702,626) → (582,626)，向左 120 px。</p><p>两者为互斥备选，固定 Home 夹爪姿态，抬升约 50 mm 后平移、放下。编号不代表优先级；M 是此前的下端折边点。</p></section>
<div class="pair"><section><h2>Version A：黄色动作箭头</h2><a href="input_A.png"><img src="input_A.png"></a></section><section><h2>Version B：仅 Pa 小编号</h2><a href="input_B.png"><img src="input_B.png"></a></section></div>
<section><p>以上是实际提交的输入图片。两版同样提供 Pa、Pb 和 displacement 文字坐标；请求唯一差异是图片中的黄色箭头。没有目标图、旧评分或历史回答。</p><p>两动作均向外拉，箭头主要落在桌面，遮挡衣服纹理很少；本次对“遮住褶皱造成状态误判”的检验力度较弱。</p>
<p>相同问题：动作1和动作2，哪一个更有可能有效展开衣服？只能选择一个，并说明预计哪一处布料关系会发生改变。</p>
<p>gpt-6-astra / high；每版 3 次独立上下文，串行顺序 A1 → B1 → B2 → A2 → A3 → B3；无重试。</p></section>
<section><h2>选择与耗时</h2><table><tr><th>表示</th><th>有效回答</th><th>选择 E</th><th>选择 M</th><th>平均请求耗时</th></tr>'''+table+'''</table><p><b>'''+esc(conclusion)+'''</b></p><p class="note">一次状态、每版三次，不是成功率实验。没有物理执行结果。固定编号未做标签互换；请求耗时包含网络与服务处理，缓存和服务负载会影响耗时，不能单独归因于画箭头。</p></section>
<h2>每次理由与预计局部变化</h2>'''+''.join(cards)+'''
<section><a href="prompt.txt">完整相同 prompt</a> · <a href="protocol.json">冻结协议</a> · <a href="results.json">逐次结果</a> · <a href="summary.json">汇总</a> · <a href="../em_fixed_pa_direction_ablation_20261022/index.html">实验 1</a></section></html>'''
    (OUT / 'index.html').write_text(page)
    print('REPORT', json.dumps(summary,ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['prepare','run','report'])
    globals()[parser.parse_args().command]()
