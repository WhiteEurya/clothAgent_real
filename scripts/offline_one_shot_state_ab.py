"""Offline paired next-action experiment; never constructs a robot runtime."""
from pathlib import Path
import argparse
import copy
import html
import json
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as wire
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once

SOURCE = Path('reports/one_shot_goal_full_episode_v1_20261022_120645_870170377')
OUT = Path('reports/one_shot_state_ab_20261022')


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    source_round = SOURCE / 'round_009'
    execution = wire.read(source_round/'execution.json')
    assert execution['status'] == 'EXECUTED' and execution['physical_execution']
    inputs = OUT/'inputs'
    inputs.mkdir()
    paths = {'previous': source_round/'before/rgb.png',
             'current': source_round/'after/rgb.png', 'goal': SOURCE/'goal/target.png'}
    for name, path in paths.items():
        shutil.copyfile(path, inputs/f'{name}.png')
    previous_plan = wire.read(source_round/'plan.json')
    geometry = wire.read(source_round/'geometry.json')
    actual_action = dict(
        coordinate_frame='previous.png 原图像素；不是 current.png 坐标',
        motion_type=previous_plan['motion_type'], grasp_px=previous_plan['grasp_px'],
        target_px=previous_plan['target_px'], grasp_angle_deg=previous_plan['grasp_angle_deg'],
        translation_mm=geometry['distance_mm'], lift_height_mm=geometry['primitive']['lift_height_mm'],
        execution_status=execution['status'], completed_commands=execution['completed_commands'],
        cloth_acquisition=execution['acquisition'], rotation_while_holding=False,
        note='这是已执行的抓放命令；执行完成不证明成功夹住布料，也不证明布料到达 Pb。')
    wire.save(inputs/'actual_action.json', actual_action)
    shutil.copyfile(source_round/'motion.py', inputs/'executed_motion.py')
    for name in ('execution.json', 'geometry.json', 'grasp_orientation.json', 'plan.json'):
        shutil.copyfile(source_round/name, inputs/('previous_'+name))
    wire.overlay(inputs/'previous.png', previous_plan, inputs/'previous_action.png', 'PREVIOUS EXECUTED COMMAND')
    size = Image.open(inputs/'current.png').size
    # Keep both arms of the comparison on the same current action policy.
    # A retains the original two-image/no-history formulation, with today's
    # corner preference and pre-grasp rotation enabled identically in A and B.
    common = wire.PROMPT.replace('固定 Home 夹爪姿态，', '抓取前可调整夹爪方向，夹住后保持姿态，')
    common += '\n尽量抓取清楚可见、可夹住的衣物边缘或角点，优先能带动展开的外露边角，避免抓衣身中部或多层堆叠处。\n'
    common += wire.GRASP_ANGLE_PROMPT.replace('第一张 perception 图', '当前图')
    b_prompt = common.replace('输入第一张是当前 RGB，第二张是目标 RGB。',
        '输入第一张是上一步动作前 RGB，随后是实际执行动作记录，第二张是动作后的当前 RGB；没有目标图。')
    b_prompt = b_prompt.replace('任务：选择最可能使当前衣物接近目标状态的一个动作。目标是衣物主要部位的展开布局和折叠/遮挡状态接近参考；',
        '任务：选择最可能使当前衣物继续展平的一个动作；目标是衣身和袖部展开，减少折叠与遮挡。')
    b_prompt = b_prompt.replace('只有当前衣物已达到参考图的主要布局与展开程度才输出 DONE',
        '只有当前衣物主要部位已经展开、没有明显大面积折叠或遮挡才输出 DONE')
    b_prompt += '''\n先在 local_effect 中判断上一动作的局部效果，再输出下一动作。
assessment 取 IMPROVED / NO_CLEAR_IMPROVEMENT / WORSENED / UNCERTAIN；
evidence 用一两句中文描述前后可见变化，uncertainty 描述无法从两张 RGB 确认的事项。
不要仅凭命令完成认定抓取成功；不要把整体平移当作展开。根据证据选择下一步，允许不确定。
下一动作的 grasp_px、target_px、grasp_angle_deg 只使用第二张当前图；上一动作的坐标仅属于第一张图。
'''
    b_schema = copy.deepcopy(wire.ROTATION_SCHEMA)
    b_schema['properties'] = {'local_effect': dict(type='object', additionalProperties=False,
        properties=dict(assessment=dict(type='string', enum=['IMPROVED', 'NO_CLEAR_IMPROVEMENT', 'WORSENED', 'UNCERTAIN']),
                        evidence=dict(type='string'), uncertainty=dict(type='string')),
        required=['assessment', 'evidence', 'uncertainty']), **b_schema['properties']}
    b_schema['required'] = ['local_effect', *b_schema['required']]
    for label, prompt, schema, image_names in (
            ('A', common, wire.ROTATION_SCHEMA, ['current', 'goal']),
            ('B', b_prompt, b_schema, ['previous', 'current'])):
        folder = OUT/label
        folder.mkdir()
        body = wire.payload(inputs/'current.png', inputs/'goal.png', size)
        body['instructions'] = prompt
        body['text']['format'].update(name='offline_next_action_'+label.lower(), schema=schema)
        text = f'当前原图尺寸 {size[0]}×{size[1]} 像素。'
        content = [dict(type='input_text', text=text)]
        for name in image_names:
            content += [dict(type='input_text', text={'current': '当前图（下一动作坐标基准）',
                'previous': '上一步动作前图', 'goal': '展平目标图'}[name]),
                dict(type='input_image', image_url=wire.image_data(inputs/f'{name}.png'), detail='high')]
            if label == 'B' and name == 'previous':
                content.append(dict(type='input_text', text='实际执行动作记录：\n'+json.dumps(actual_action, ensure_ascii=False)))
        body['input'][0]['content'] = content
        wire.save(folder/'prepared_request.json', body)
        (folder/'prompt.txt').write_text(prompt+'\n\n'+'\n'.join(c['text'] for c in content if c['type']=='input_text'))
    wire.save(OUT/'protocol.json', dict(created_at=wire.utc(), source=str(SOURCE.resolve()), source_round='round_009',
        current_definition='round_009/after/rgb.png (identical bytes in A and B)',
        sample_selection='Visual failure candidate: previously extended left sleeve folds across upper chest; garment shifts toward right table edge. Not a human-scored GT failure.',
        selection_before_model_calls=True, model=wire.MODEL, effort=wire.EFFORT,
        trials_per_version=1, order=['A','B'], right_view_supplied=False,
        A_inputs=['current','goal'], B_inputs=['previous','actual_executed_command','current'],
        common_policy='Same current corner preference and pre-grasp rotation; A uses original no-history two-image structure, not byte-identical historical prompt.',
        limitations=['Single selected sample and one request per version; no success-rate or stable speed estimate.',
            'B removes goal image as well as adding history/effect assessment; not an isolated history ablation.',
            'B has additional output fields; latency includes extra assessment generation.',
            'No depth/IK/gate checks or robot execution of proposals.'],
        robot_execution=False, size=list(size),
        source_images={k:dict(path=str(v.resolve()), sha256=wire.sha(v)) for k,v in paths.items()}))
    shutil.copyfile(__file__, OUT/'experiment_source.py')
    print('PREPARED', OUT, flush=True)


def run():
    headers = wire.auth_headers()  # Never saved or printed.
    size = wire.read(OUT/'protocol.json')['size']
    for label in ('A', 'B'):
        folder = OUT/label
        if (folder/'attempt.json').exists():
            print(label, 'existing attempt retained; not repeated', flush=True)
            continue
        started = time.perf_counter()
        try:
            value, metrics = direct_once(folder, wire.read(folder/'prepared_request.json'), headers, 180)
            wire.save(folder/'output.json', value)
            plan = {k:v for k,v in value.items() if k != 'local_effect'}
            wire.validate_plan(plan, size, require_grasp_rotation=True)
            wire.save(folder/'plan.json', plan)
            wire.overlay(OUT/'inputs/current.png', plan, folder/'overlay.png', label+' NEXT ACTION (NOT EXECUTED)')
            print(label, json.dumps(value, ensure_ascii=False), 'seconds', metrics['wall_latency_s'], flush=True)
        except Exception as exc:
            wire.save(folder/'error.json', dict(error=f'{type(exc).__name__}: {exc}'))
            print(label, type(exc).__name__, 'see saved error', flush=True)
        finally:
            wire.save(folder/'client_timing.json', dict(inference_and_validation_s=time.perf_counter()-started))
    report()


def report():
    rows = []
    cards = []
    esc = html.escape
    for label in ('A','B'):
        folder = OUT/label
        metrics = wire.read(folder/'model_metrics.json') if (folder/'model_metrics.json').exists() else {}
        output = wire.read(folder/'output.json') if (folder/'output.json').exists() else {}
        usage = metrics.get('usage') or {}
        rows.append(dict(version=label, output=output, metrics=metrics))
        latency = metrics.get('wall_latency_s', 0)
        title = '当前图＋展平目标' if label == 'A' else '上一步图＋实际动作＋当前图'
        effect = output.get('local_effect')
        effect_html = '' if not effect else '<div class="effect"><b>上一动作：'+esc(effect['assessment'])+'</b><p>'+esc(effect['evidence'])+'</p><p>不确定性：'+esc(effect['uncertainty'])+'</p></div>'
        image_html = f'<img src="{label}/overlay.png">' if (folder/'overlay.png').exists() else '<p>请求失败，参见原始记录。</p>'
        cards.append(f'''<article><h2>{label} · {title}</h2><div class="metric">{latency:.2f} 秒 <small>请求至完整响应</small></div>
<p>输入 token {usage.get('input_tokens','—')} · 输出 token {usage.get('output_tokens','—')} · 推理 token {(usage.get('output_tokens_details') or {}).get('reasoning_tokens','—')}</p>
{image_html}<p><b>{esc(str(output.get('decision','FAILED')))} · Pa {output.get('grasp_px')} → Pb {output.get('target_px')} · 夹爪 {output.get('grasp_angle_deg')}°</b></p>
<p>{esc(output.get('short_reason',''))}</p>{effect_html}
<details><summary>完整模型输出</summary><pre>{esc(json.dumps(output, ensure_ascii=False, indent=2))}</pre></details>
<p><a href="{label}/prompt.txt">完整 prompt</a> · <a href="{label}/model_metrics.json">耗时和 token</a> · <a href="{label}/raw_output.txt">原始输出</a></p></article>''')
    wire.save(OUT/'results.json', rows)
    summary_html = ''
    if all(row['output'].get('decision') == 'ACTION' for row in rows):
        import math
        a, b = [row['output'] for row in rows]
        ta, tb = [row['metrics']['wall_latency_s'] for row in rows]
        pa_delta = math.dist(a['grasp_px'], b['grasp_px'])
        pb_delta = math.dist(a['target_px'], b['target_px'])
        summary_html = f'''<div class="panel" style="margin-top:20px"><h2>这一个样本说明什么</h2>
<p>A 用时 {ta:.2f} 秒，B 用时 {tb:.2f} 秒；B 本次多 {tb-ta:.2f} 秒（{(tb/ta-1)*100:.1f}%）。耗时不包含拍照、机器人移动或 gate 检查。</p>
<p>抓取点相差 {pa_delta:.1f} 像素，两版都选择左侧外露袖口；放置点相差 {pb_delta:.1f} 像素。A 向左上拉，B 更偏向左拉，夹爪角度也不同。</p>
<p>B 明确判断“没有明确改善”：看到了整体位移与剩余肩胸叠折，并说明避免继续向右侧边界拉动。A 单看当前图也发现了胸前袖部压叠；加入历史后未改变抓取部位，主要改变了拉动方向与解释。</p>
<p>这证明 B 能输出基于前后观测的效果评估，尚不能证明 B 的下一动作会更有效。两版新动作均未实机执行。</p></div>'''
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>One-shot 状态输入 A/B 实验</title>
<style>body{font-family:system-ui,sans-serif;margin:28px;background:#f3f5f9;color:#182235}h1{margin:0 0 12px}p{line-height:1.6}a{color:#2059a3}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.inputs{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}article,.panel{background:white;padding:20px;border-radius:12px}img{width:100%;display:block}figure{margin:0}figcaption{padding:8px 0}.metric{font-size:28px;font-weight:700}small{font-size:14px;font-weight:400}.effect{background:#edf3ff;padding:12px;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.note{color:#785200}button{padding:7px 14px}@media(max-width:900px){.grid,.inputs{grid-template-columns:1fr}}</style>
<h1>同一失败候选状态：无历史 A / 有状态 B</h1>
<p>来源：2026-10-22 12:06 run · round_009（第 10 个动作）。下一步输入统一使用该动作的 after 原图，1280×720。</p>
<p>选样依据：动作执行后，原先向左展开的袖部横折至上胸，衣物更靠近右边缘；这是目测效果退化候选，未作人工 GT 评分。该判断未提供给 A/B。</p>
<div class="panel"><div class="inputs"><figure><img id="previous" src="inputs/previous_action.png"><figcaption>动作前＋已执行命令：Pa (914,307) → Pb (1007,339)，夹爪 0°，平移 74.33 mm。<button onclick="const i=document.getElementById('previous');i.src=i.getAttribute('src').includes('previous_action')?'inputs/previous.png':'inputs/previous_action.png'">切换标注</button></figcaption></figure><figure><img src="inputs/current.png"><figcaption>动作后：A、B 共用的当前图</figcaption></figure><figure><img src="inputs/goal.png"><figcaption>展平目标（仅 A 输入）</figcaption></figure></div></div>
<p>共同设置：gpt-6-astra / high；各一次、顺序 A→B；两张图、high detail；相同边角偏好、允许抓前旋转；无工具、无 right view、无自动重试。B 先报告局部效果再提议下一动作。</p>
<div class="grid">''' + ''.join(cards) + '</div>' + summary_html + '''
<div class="panel" style="margin-top:20px"><p class="note">这里只比较规划输出，两个新动作均未执行，也未经过深度、IK 或 gate 检查。箭头不是实际运动结果；命令执行完成不代表夹住了布料。</p>
<p>按指定输入，B 同时加入历史并移除目标图，且输出更多文字，因此差异不能只归因于“有状态”。单样本、单次耗时不能代表总体准确率或平均速度。</p>
<p>A 使用原始 one-shot 的双图、无历史结构，同时与 B 保持当前边角与夹爪旋转规则一致，不是逐字复刻旧 prompt。</p>
<a href="protocol.json">实验协议与原图哈希</a> · <a href="results.json">完整结果</a> · <a href="inputs/actual_action.json">上一实际执行命令</a></div></html>'''
    (OUT/'index.html').write_text(page)
    print('REPORT', (OUT/'index.html').resolve(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','run','report'])
    args = parser.parse_args()
    globals()[args.command]()
