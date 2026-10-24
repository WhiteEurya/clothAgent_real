"""Three offline A/B cases from two user-interrupted October 21 episodes."""
from pathlib import Path
import copy
import html
import json
import math
import shutil
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as wire
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once

OUT = Path('reports/interrupted_state_ab_20261022')
TEMPLATE = Path('reports/one_shot_state_ab_20261022')
CASES = [
    ('S1', '230743_643820346', 1, 2, '23:07 run · round_001 后',
     '衣身仍叠压，拉动后整体布局改变，部位边界不清；下一动作 round_002 被用户中断。'),
    ('S2', '232513_053641787', 2, 4, '23:25 run · round_002 后',
     '衣身呈窄长叠层，上一拉动前后几乎没有可见变化；该 run 后来在 round_004 被用户中断。'),
    ('S3', '232513_053641787', 3, 4, '23:25 run · round_003 后',
     '衣身仍呈窄长叠层，动作后有整体转向/位移；下一动作 round_004 被用户中断。'),
]


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    templates = {v: wire.read(TEMPLATE/v/'prepared_request.json') for v in ('A','B')}
    manifest = []
    for key, suffix, previous_index, interrupted_index, title, selection in CASES:
        run = Path('reports')/('one_shot_goal_full_episode_v1_20261021_'+suffix)
        previous = run/f'round_{previous_index:03d}'
        interrupted = run/f'round_{interrupted_index:03d}'
        interrupted_execution = wire.read(interrupted/'execution.json')
        assert interrupted_execution['status']=='USER_INTERRUPTED'
        execution = wire.read(previous/'execution.json')
        assert execution['status']=='EXECUTED' and execution['physical_execution']
        root = OUT/key
        inputs = root/'inputs'
        inputs.mkdir(parents=True)
        paths = dict(previous=previous/'before/rgb.png', current=previous/'after/rgb.png', goal=run/'goal/target.png')
        for name, path in paths.items():
            shutil.copyfile(path, inputs/f'{name}.png')
        plan = wire.read(previous/'plan.json')
        geometry = wire.read(previous/'geometry.json')
        action = dict(coordinate_frame='previous.png 原图像素；不是 current.png 坐标',
            motion_type=plan['motion_type'], grasp_px=plan['grasp_px'], target_px=plan['target_px'],
            grasp_angle_deg=plan.get('grasp_angle_deg'),
            orientation=geometry['primitive']['orientation'], rpy_world_deg=geometry['rpy_world_deg'],
            translation_mm=geometry['distance_mm'], lift_height_mm=geometry['primitive']['lift_height_mm'],
            execution_status=execution['status'], completed_commands=execution['completed_commands'],
            cloth_acquisition=execution['acquisition'], rotation_while_holding=False,
            note='这是已执行的抓放命令；执行完成不证明成功夹住布料，也不证明布料到达 Pb。此历史动作固定 Home 姿态，未提供图像夹爪角度。')
        wire.save(inputs/'actual_action.json', action)
        for name in ('motion.py','execution.json','geometry.json','plan.json'):
            shutil.copyfile(previous/name, inputs/('previous_'+name))
        wire.save(inputs/'later_interrupted_execution.json', interrupted_execution)
        wire.overlay(inputs/'previous.png', plan, inputs/'previous_action.png', key+' PREVIOUS COMPLETED COMMAND')
        with Image.open(inputs/'current.png') as im:
            size = list(im.size)
        for version in ('A','B'):
            folder = root/version
            folder.mkdir()
            body = copy.deepcopy(templates[version])
            content = [dict(type='input_text',text=f'当前原图尺寸 {size[0]}×{size[1]} 像素。')]
            for name in (['current','goal'] if version=='A' else ['previous','current']):
                content.extend([dict(type='input_text', text={'current':'当前图（下一动作坐标基准）',
                    'previous':'上一步动作前图','goal':'展平目标图'}[name]),
                    dict(type='input_image',image_url=wire.image_data(inputs/f'{name}.png'),detail='high')])
                if name=='previous':
                    content.append(dict(type='input_text',text='实际执行动作记录：\n'+json.dumps(action,ensure_ascii=False)))
            body['input'][0]['content'] = content
            wire.save(folder/'prepared_request.json',body)
            (folder/'prompt.txt').write_text(body['instructions']+'\n\n'+'\n'.join(c['text'] for c in content if c['type']=='input_text'))
        meta = dict(case=key, title=title, selection_reason=selection, source=str(run.resolve()),
            previous_round=previous.name, interrupted_round=interrupted.name,
            interrupted_execution=interrupted_execution, completed_previous_execution=execution,
            current_definition=str(paths['current'].resolve()), size=size,
            source_images={k:dict(path=str(p.resolve()),sha256=wire.sha(p)) for k,p in paths.items()},
            order=['B','A'] if key=='S2' else ['A','B'])
        wire.save(root/'case.json',meta)
        manifest.append(meta)
    wire.save(OUT/'protocol.json',dict(created_at=wire.utc(), cases=manifest,
        model=wire.MODEL,effort=wire.EFFORT,trials_per_version_per_case=1,
        A_inputs=['current','goal'],B_inputs=['previous','actual_executed_command','current'],
        image_count_each=2,right_view_supplied=False,robot_execution=False,
        selection_before_requests=True,history_depth=1,
        shared_action_policy='Current corner preference and pre-grasp rotation in both A and B; historical executed commands used fixed Home orientation.',
        limitations=['Only two independent episodes; S2 and S3 are consecutive states of the same episode.',
            'User interruption is not an independently scored failure label; no incomplete action was represented as completed.',
            'B removes goal and adds history and effect output; not a pure history ablation.',
            'Single request per version per state; no depth/IK/gate or robot execution of proposals.']))
    shutil.copyfile(__file__,OUT/'experiment_source.py')
    print('PREPARED',OUT,flush=True)


def run():
    headers=wire.auth_headers()
    for case in wire.read(OUT/'protocol.json')['cases']:
        for version in case['order']:
            folder=OUT/case['case']/version
            if (folder/'attempt.json').exists():
                print(case['case'],version,'existing attempt retained',flush=True)
                continue
            try:
                result,metrics=direct_once(folder,wire.read(folder/'prepared_request.json'),headers,180)
                wire.save(folder/'output.json',result)
                plan={k:v for k,v in result.items() if k!='local_effect'}
                wire.validate_plan(plan,case['size'],require_grasp_rotation=True)
                wire.save(folder/'plan.json',plan)
                wire.overlay(folder.parent/'inputs/current.png',plan,folder/'overlay.png',case['case']+' '+version+' (NOT EXECUTED)')
                print(case['case'],version,round(metrics['wall_latency_s'],3),json.dumps(result,ensure_ascii=False),flush=True)
            except Exception as exc:
                wire.save(folder/'error.json',dict(error=f'{type(exc).__name__}: {exc}'))
                print(case['case'],version,type(exc).__name__,'see saved error',flush=True)
    report()


def report():
    esc=html.escape
    cases=wire.read(OUT/'protocol.json')['cases']
    sections=[]; rows=[]; table=[];times={'A':[],'B':[]}
    for case in cases:
        key=case['case'];cards=[]; outputs={}
        action=wire.read(OUT/key/'inputs/actual_action.json')
        for version in ('A','B'):
            folder=OUT/key/version
            metrics=wire.read(folder/'model_metrics.json') if (folder/'model_metrics.json').exists() else {}
            output=wire.read(folder/'output.json') if (folder/'output.json').exists() else {}
            outputs[version]=output
            rows.append(dict(case=key,version=version,output=output,metrics=metrics))
            latency=metrics.get('wall_latency_s',0)
            if metrics.get('status')=='VALID_JSON':times[version].append(latency)
            usage=metrics.get('usage') or {}
            effect=output.get('local_effect')
            effect_html='' if not effect else '<div class="effect"><b>局部效果：'+esc(effect['assessment'])+'</b><p>'+esc(effect['evidence'])+'</p><p>不确定性：'+esc(effect['uncertainty'])+'</p></div>'
            image_html=f'<img src="{key}/{version}/overlay.png">' if (folder/'overlay.png').exists() else '<p>请求未成功；查看原始记录。</p>'
            cards.append(f'''<article><h3>{version} · {'当前＋目标' if version=='A' else '上一步＋动作＋当前'}</h3>
<div class="metric">{latency:.2f} 秒</div><p>输入 token {usage.get('input_tokens','—')} · 输出 token {usage.get('output_tokens','—')} · 缓存 token {(usage.get('input_tokens_details') or {}).get('cached_tokens','—')}</p>
{image_html}<p><b>{esc(str(output.get('decision','FAILED')))} · Pa {output.get('grasp_px')} → Pb {output.get('target_px')} · 夹爪 {output.get('grasp_angle_deg')}°</b></p>
<p>{esc(output.get('short_reason',''))}</p>{effect_html}
<details><summary>完整输出</summary><pre>{esc(json.dumps(output,ensure_ascii=False,indent=2))}</pre></details>
<p><a href="{key}/{version}/prompt.txt">完整 prompt</a> · <a href="{key}/{version}/model_metrics.json">耗时和 token</a> · <a href="{key}/{version}/raw_output.txt">原始输出</a></p></article>''')
        a,b=outputs['A'],outputs['B']
        diff=''
        if a.get('decision')==b.get('decision')=='ACTION':
            pa=math.dist(a['grasp_px'],b['grasp_px']);pb=math.dist(a['target_px'],b['target_px'])
            diff=f'两版抓取点相差 {pa:.1f} px；放置点相差 {pb:.1f} px。'
        effect=b.get('local_effect',{}).get('assessment','—')
        latencies=[next(r['metrics'].get('wall_latency_s',0) for r in rows if r['case']==key and r['version']==v) for v in ('A','B')]
        table.append(f'<tr><td><a href="#{key}">{key} · {esc(case["title"])}</a></td><td>{latencies[0]:.2f} s</td><td>{latencies[1]:.2f} s</td><td>{esc(effect)}</td><td>{diff}</td></tr>')
        sections.append(f'''<section id="{key}"><h2>{key} · {esc(case['title'])}</h2><p>{esc(case['selection_reason'])}</p>
<div class="inputs panel"><figure><img id="{key}-previous" src="{key}/inputs/previous_action.png"><figcaption>上一步已执行：Pa {action['grasp_px']} → Pb {action['target_px']}，平移 {action['translation_mm']:.2f} mm，固定 Home 姿态。
<button onclick="let i=document.getElementById('{key}-previous');i.src=i.getAttribute('src').includes('previous_action')?'{key}/inputs/previous.png':'{key}/inputs/previous_action.png'">切换标注</button></figcaption></figure>
<figure><img src="{key}/inputs/current.png"><figcaption>动作后当前图 · A/B 完全相同</figcaption></figure><figure><img src="{key}/inputs/goal.png"><figcaption>目标图 · 仅 A 输入</figcaption></figure></div>
<div class="grid">{''.join(cards)}</div><p>{diff}</p><p><a href="{key}/case.json">来源与原图哈希</a> · <a href="{key}/inputs/actual_action.json">实际执行命令</a></p></section>''')
    aggregate={v:dict(n=len(t),mean_s=statistics.mean(t),median_s=statistics.median(t),min_s=min(t),max_s=max(t)) for v,t in times.items() if t}
    wire.save(OUT/'results.json',rows)
    wire.save(OUT/'latency_summary.json',aggregate)
    summary=' · '.join(f'{v}：本批均值 {m["mean_s"]:.2f} s / 中位数 {m["median_s"]:.2f} s' for v,m in aggregate.items())
    analysis=(OUT/'analysis.html').read_text() if (OUT/'analysis.html').exists() else ''
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>被中断 run：三组状态 A/B 对照</title>
<style>body{font-family:system-ui,sans-serif;margin:24px;background:#f3f5f9;color:#182235}h1{margin:0 0 12px}p{line-height:1.65}a{color:#2059a3}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:16px}.inputs{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}article,.panel{background:white;padding:18px;border-radius:12px}img{width:100%;display:block}figure{margin:0}figcaption{padding:8px 0}.metric{font-size:26px;font-weight:700}.effect{background:#edf3ff;padding:12px;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere}button{padding:6px 12px}section{margin-top:36px;border-top:2px solid #d3dbe8}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px;border-bottom:1px solid #ddd}@media(max-width:900px){.grid,.inputs{grid-template-columns:1fr}}</style>
<h1>昨天被你中断的 run：3 个折叠状态的 A/B 实验</h1>
<p>来源：2026-10-21 23:07、23:25 两次明确记录 USER_INTERRUPTED 的 run。使用其中已执行完成动作的前后观测；中断动作本身没有 after 图，没有当作完整动作使用。当前图均是对应完整动作后的原图。</p>
<p>A：当前图＋目标图；B：上一步图＋实际执行命令＋当前图，并先输出局部效果判断。两版同为 gpt-6-astra / high、两张原图、相同边角与抓前旋转规则，各一次调用；无 right view、无工具、无重试。调用顺序 S1 A→B、S2 B→A、S3 A→B。</p>
<p>历史动作固定 Home 姿态；本次两版的下一动作都允许抓前旋转。选样理由、原 run 的后续动作和中断信息没有提供给模型。</p>
<div class="panel"><table><thead><tr><th>状态</th><th>A 耗时</th><th>B 耗时</th><th>B 对上一步的判断</th><th>动作差异</th></tr></thead><tbody>'''+''.join(table)+f'</tbody></table><p>{summary}</p></div>'+analysis+''.join(sections)+'''
<div class="panel"><p>耗时为请求发送到完整响应，不含拍照、机器人移动、深度/IK/gate 检查。新动作均未执行，箭头只是规划提议，不能据此判定物理成功率。</p>
<p>只有两个独立 run，S2/S3 是相邻状态；中断不自动等于失败。B 同时移除了目标图并增加效果评估输出，因此这是两套输入方案的比较，不是纯历史消融。</p>
<a href="protocol.json">实验协议</a> · <a href="results.json">全部输出</a> · <a href="latency_summary.json">耗时统计</a></div></html>'''
    (OUT/'index.html').write_text(page)
    print('REPORT',OUT/'index.html',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','report'])
    globals()[parser.parse_args().command]()
