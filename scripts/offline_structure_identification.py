"""Repeated independent, single-image structure descriptions. No robot actions."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/independent_structure_M_20261022'
SOURCE = ROOT / 'reports/visual_fact_ab_M_20261022/current.png'
PROMPT = '''你是衣物图像的结构观察者。本任务只描述单张图片中指定位置的结构，不评价或生成任何动作。
根据可见证据，分别判断：
1. M 对应什么衣物部位，位于该部位的什么位置。
2. M 附近边缘或线条的性质，以及 M 是否处于边界、转角或内部。
3. M 与附近布料的连接、覆盖和层次关系。
每项先写可见证据，再给出解释；明确区分直接可见事实与推测，不把图像轮廓形状直接当成已确认的衣物结构。
如果有多个合理解释，可以列出备选及无法区分的原因。不要为了给出具体部位而猜测隐藏结构。
每项选择 evidence_status：SUPPORTED（可见证据足以支持该项结论）、UNCERTAIN（有线索但不能排除其他解释）、INSUFFICIENT（证据不足，无法判断）。
证据不足是有效答案，不必给出具体部位；某项无法判断时可以直接写“证据不足”。
最后说明整体判断是否能成立，以及还缺少哪些视觉证据。不得评价抓取价值、拉动方向或动作效果。
用中文回答，只输出指定 JSON，不调用工具。'''


def schema():
    aspect = dict(type='object', additionalProperties=False, properties=dict(
        visible_evidence=dict(type='string'), conclusion=dict(type='string'),
        evidence_status=dict(type='string', enum=['SUPPORTED', 'UNCERTAIN', 'INSUFFICIENT']),
        alternatives=dict(type='array', items=dict(type='string'))),
        required=['visible_evidence', 'conclusion', 'evidence_status', 'alternatives'])
    props = {k: aspect for k in ('garment_part', 'edge_nature', 'nearby_cloth_relations')}
    props.update(overall_conclusion=dict(type='string'), missing_visual_evidence=dict(type='string'))
    return dict(type='object', additionalProperties=False, properties=props, required=list(props))


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(SOURCE, OUT/'current.png')
    assert io.sha(OUT/'current.png') == 'a522f3b0f0c23266c4f999e189d7d235c9821d617bc7e591f8acae58630b12fa'
    user_text = '图像尺寸1280×720。像素坐标以原图左上角为原点，x向右、y向下。待观察点 M 的位置为 (702,626)。'
    body = dict(model=io.MODEL, instructions=PROMPT,
        input=[dict(role='user', content=[dict(type='input_text', text=user_text),
            dict(type='input_image', image_url=io.image_data(OUT/'current.png'), detail='high')])],
        tools=[], tool_choice='none', parallel_tool_calls=False,
        reasoning=dict(effort=io.EFFORT), store=False, stream=True,
        text=dict(format=dict(type='json_schema', name='independent_garment_structure', strict=True, schema=schema())))
    io.save(OUT/'prepared_request.json', body)
    (OUT/'prompt.txt').write_text(PROMPT+'\n\n'+user_text+'\n')
    io.save(OUT/'protocol.json', dict(created_at=io.utc(), source=str(SOURCE), point=[702,626],
        model=io.MODEL, effort=io.EFFORT, repeats=5, independent_contexts=True, max_workers=3,
        identical_payload=True, retries=0, robot_execution=False,
        input_images=['current.png'], image_sha256=io.sha(OUT/'current.png'),
        no_goal_no_action_no_history_no_identity_hint=True,
        evaluation_only_user_reference='下摆正中间；不在模型请求内。仅识别到下摆而未确认中点，算部分识别。',
        evaluation_rule='按原文分别记录明确确认下摆中点、仅部分识别、其他明确识别、无法确定；备选提及不算明确确认。证据不足单独统计，不能当作识别正确。',
        limitation='同图重复请求只描述该图上的输出稳定性，不是跨图准确率；用户参考身份未经此实验独立验证。'))
    # Viewer-only marker; the model receives the unmodified source RGB.
    im=Image.open(OUT/'current.png').convert('RGB');draw=ImageDraw.Draw(im)
    draw.ellipse((692,616,712,636), outline='#00ffff', width=3)
    draw.text((718,626), 'M', fill='#00ffff', stroke_width=1, stroke_fill='black')
    im.save(OUT/'M_location_viewer_only.png')
    shutil.copyfile(__file__, OUT/'experiment_source.py')
    print('PREPARED', OUT, flush=True)


def run():
    protocol=io.read(OUT/'protocol.json');body=io.read(OUT/'prepared_request.json');headers=io.auth_headers()
    def one(i):
        folder=OUT/f'trial_{i:02d}'
        if (folder/'attempt.json').exists():
            return dict(trial=i, preserved_attempt=True)
        try:
            output,metrics=direct_once(folder, body, headers, 300)
            io.save(folder/'output.json',output)
            return dict(trial=i,seconds=metrics['wall_latency_s'],part=output['garment_part'],overall=output['overall_conclusion'])
        except Exception as exc:
            io.save(folder/'error.json',dict(error=f'{type(exc).__name__}: {exc}'))
            return dict(trial=i,error=f'{type(exc).__name__}: {exc}')
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=protocol['max_workers']) as pool:
        futures=[pool.submit(one,i) for i in range(1,protocol['repeats']+1)]
        for future in as_completed(futures):
            print(json.dumps(future.result(),ensure_ascii=False),flush=True)
    io.save(OUT/'batch_metrics.json',dict(batch_wall_s=time.perf_counter()-started))
    report()


def report():
    protocol=io.read(OUT/'protocol.json');rows=[];hashes=[]
    fields=[('garment_part','衣物部位'),('edge_nature','边缘性质'),('nearby_cloth_relations','附近布料关系')]
    lines=['独立结构识别：M=(702,626)', '来源：2026-10-21 23:25 run · round_002 对应的既有干净图。',
        '5次相同输入、独立上下文；仅一张干净RGB和坐标，无目标图、动作、历史评价或部位答案。',
        f"模型：{protocol['model']}；推理设置：{protocol['effort']}；允许证据不足；不执行机器人。",'']
    for i in range(1,6):
        folder=OUT/f'trial_{i:02d}'
        if not (folder/'output.json').exists():
            lines.append(f'第{i}次：未取得有效结果，见该次错误记录。');continue
        o=io.read(folder/'output.json');m=io.read(folder/'model_metrics.json')
        hashes.append(hashlib.sha256((folder/'request_payload.json').read_bytes()).hexdigest())
        rows.append(dict(trial=i,seconds=m['wall_latency_s'],output=o))
        lines += [f"第{i}次：{m['wall_latency_s']:.2f}秒"]
        for field,label in fields:
            a=o[field]
            lines += [f"{label} [{a['evidence_status']}]",'可见证据：'+a['visible_evidence'],
                '结论：'+a['conclusion'],'备选解释：'+('；'.join(a['alternatives']) or '无')]
        lines += ['整体判断：'+o['overall_conclusion'],'缺少证据：'+o['missing_visual_evidence'],'']
    assert len(set(hashes))<=1,'Requests must be byte-identical'
    counts={k:{s:sum(r['output'][k]['evidence_status']==s for r in rows)
        for s in ('SUPPORTED','UNCERTAIN','INSUFFICIENT')} for k,_ in fields}
    summary=dict(valid_responses=len(rows),expected_responses=5,results=rows,status_counts=counts,
        identical_dispatched_payloads=len(hashes)==5 and len(set(hashes))==1,
        summed_request_s=sum(r['seconds'] for r in rows))
    if (OUT/'batch_metrics.json').exists():summary.update(io.read(OUT/'batch_metrics.json'))
    io.save(OUT/'summary.json',summary)
    lines += ['注意：相同图片重复请求的结果比例不能解释为总体识别准确率；模型自报证据强度不是校准后的置信度。',
        '请求耗时包含网络和生成；并发执行，累计请求耗时与整批墙钟耗时不同。']
    if (OUT/'interpretation.txt').exists():lines=[(OUT/'interpretation.txt').read_text(),'——逐次原始回答——','',*lines]
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    page='<!doctype html><meta charset="utf-8"><title>M 独立结构识别</title><style>body{font:17px system-ui;max-width:1100px;margin:32px auto;padding:20px;color:#182235;background:#f4f6f8}pre{white-space:pre-wrap;line-height:1.7;background:white;padding:24px}img{width:100%}</style>'
    page+='<h1>M 点独立结构识别 · 5 次</h1><a href="report.txt">纯文字报告</a> · <a href="prompt.txt">完整 prompt</a> · <a href="protocol.json">实验协议</a><p>下图标记只供人查看；模型收到的是不带标记的原图和坐标。</p><img src="M_location_viewer_only.png"><pre>'+html.escape('\n'.join(lines))+'</pre>'
    (OUT/'index.html').write_text(page)
    print('REPORT',len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','report'])
    globals()[p.parse_args().command]()
