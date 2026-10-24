"""Offline A/B/C: baseline, independent structure report, in-request self-check."""
from pathlib import Path
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
import html
import json
import shutil
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once
from offline_visual_fact_ab import PROMPT, schema, plan

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/visual_fact_ab_M_20261022'
STRUCTURE=ROOT/'reports/independent_structure_M_20261022/trial_01/output.json'
OUT=ROOT/'reports/structure_decision_ABC_20261022'
C_CHECK='''作决定前，请自主检查候选动作评价所依赖的衣物结构假设：哪些部位、边缘性质、连接与覆盖关系有直接图像证据，哪些只是推测？
对会影响动作效果的关键假设，检查是否存在其他合理结构解释，以及这些解释是否会改变动作评价。若证据不足，应保留不确定性，不把未验证的假设当成事实。
据此完成同样的候选评分与选择。没有预设应选择或排除的动作；结论可以保持，也可以改变。最终在既有理由、风险和不确定性字段中简要说明相关依据，不需额外输出逐步思考过程。'''


def prepare():
    OUT.mkdir(parents=True,exist_ok=False)
    prior=io.read(SOURCE/'protocol.json');actions=prior['actions'];target=prior['target']
    shutil.copyfile(SOURCE/'current.png',OUT/'current.png')
    shutil.copyfile(STRUCTURE,OUT/'independent_structure.json')
    assert io.sha(OUT/'current.png')==prior['image_sha256']['current.png']
    base_prompt=PROMPT.replace('第一张是perception当前RGB，第二张是展平目标RGB。',
        '唯一图片是当前RGB。本次目标是展平衣服，没有提供目标图片。').replace('不把目标图坐标搬到当前图。','')
    candidate_text='图像尺寸1280×720；仅提供当前干净原图。待评动作：'+target['id']+'。\n固定候选：\n'+json.dumps(
        [{k:v for k,v in a.items() if k!='source_id'} for a in actions],ensure_ascii=False)
    base=dict(model=io.MODEL,instructions=base_prompt,
        input=[dict(role='user',content=[dict(type='input_text',text=candidate_text),
            dict(type='input_image',image_url=io.image_data(OUT/'current.png'),detail='high')])],
        tools=[],tool_choice='none',parallel_tool_calls=False,reasoning=dict(effort=io.EFFORT),store=False,stream=True,
        text=dict(format=dict(type='json_schema',name='structure_decision_comparison',strict=True,
            schema=schema([a['id'] for a in actions]))))
    additions=dict(A='',B='以下是对同一图片、M=(702,626) 的一次独立视觉结构识别结果，供参考；它不是已确认的结构标注，也不包含动作评价。\n'+
        (OUT/'independent_structure.json').read_text(),C=C_CHECK)
    for condition,extra in additions.items():
        body=copy.deepcopy(base)
        if extra:body['input'][0]['content'].append(dict(type='input_text',text=extra))
        io.save(OUT/f'prepared_{condition}.json',body)
        (OUT/f'prompt_{condition}.txt').write_text(base_prompt+'\n\n'+candidate_text+'\n\n'+extra+'\n')
        stripped=copy.deepcopy(body)
        if extra:stripped['input'][0]['content'].pop()
        assert stripped==base
    schedule=[(c,r) for r,order in enumerate(('ABC','BCA','CAB'),1) for c in order]
    io.save(OUT/'protocol.json',dict(created_at=io.utc(),source=str(SOURCE),target=target,actions=actions,
        image_sha256=io.sha(OUT/'current.png'),input_images=['current.png'],model=io.MODEL,effort=io.EFFORT,
        repeats_per_condition=3,requests=9,retries=0,max_workers=3,dispatch_schedule=schedule,
        independent_contexts=True,robot_execution=False,goal_image=False,arrows_in_model_input=False,
        prior_scores_outcomes_identity_correction_supplied=False,
        B_structure_source=str(STRUCTURE),B_source_sha256=io.sha(STRUCTURE),
        B_selection_rule='固定复用第1次完整结果，按原编号选择，不按后续决策效果筛选；本实验不重新调用视觉模型。',
        C_implementation='同一次模型请求内加入自主结构检查指令；相同high设置，无额外模型回合、工具或外部结构答案。',
        evaluation='比较M分数/排名/首选频数与理由是否依赖未确认结构；不把换动作等同物理成功。',
        caveats=['同图每条件3次，探索性结果，不代表泛化能力。','B提供额外结构描述和注意位置，不能单独归因于不确定性措辞。',
                 'C仅测试一次请求内的提示式自检，不代表所有Harness方案。','历史A/B含目标图；本次仅当前图，各条件重新请求，不混用历史基线。']))
    io.overlay(OUT/'current.png',plan(target),OUT/'M_action.png','Fixed M action; viewer only')
    shutil.copyfile(__file__,OUT/'experiment_source.py')
    print('PREPARED',OUT,flush=True)


def run():
    protocol=io.read(OUT/'protocol.json');headers=io.auth_headers()
    expected={a['id'] for a in protocol['actions']}
    def one(condition,rep):
        folder=OUT/f'{condition}_{rep:02d}'
        if (folder/'attempt.json').exists():return dict(condition=condition,rep=rep,preserved=True)
        try:
            output,metrics=direct_once(folder,io.read(OUT/f'prepared_{condition}.json'),headers,300)
            if len(output['scores'])!=10 or {r['id'] for r in output['scores']}!=expected:
                raise ValueError('duplicate or missing candidate scores')
            io.save(folder/'output.json',output)
            m=next(r['action_score'] for r in output['scores'] if r['id']==protocol['target']['id'])
            return dict(condition=condition,rep=rep,seconds=metrics['wall_latency_s'],M_score=m,
                selected=output['selected_id'],judgment=output['target_judgment'],identity=output['pa_identity'])
        except Exception as exc:
            io.save(folder/'error.json',dict(error=f'{type(exc).__name__}: {exc}'))
            return dict(condition=condition,rep=rep,error=f'{type(exc).__name__}: {exc}')
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(one,c,r) for c,r in protocol['dispatch_schedule']]
        for f in as_completed(futures):print(json.dumps(f.result(),ensure_ascii=False),flush=True)
    io.save(OUT/'batch_metrics.json',dict(batch_wall_s=time.perf_counter()-started))
    report()


def report():
    protocol=io.read(OUT/'protocol.json');actions={a['id']:a for a in protocol['actions']};target=protocol['target']['id']
    results=[];csv_rows=[];cards=[]
    for c in 'ABC':
        for rep in range(1,4):
            name=f'{c}_{rep:02d}';folder=OUT/name
            if not (folder/'output.json').exists() or (folder/'error.json').exists():continue
            o=io.read(folder/'output.json');metrics=io.read(folder/'model_metrics.json');scores={r['id']:r['action_score'] for r in o['scores']}
            row=dict(condition=c,rep=rep,seconds=metrics['wall_latency_s'],M_score=scores[target],
                M_rank=1+sum(v>scores[target] for v in scores.values()),selected_id=o['selected_id'],
                M_margin_to_best_other=scores[target]-max(v for k,v in scores.items() if k!=target),
                selected_action=actions[o['selected_id']],judgment=o['target_judgment'],output=o,
                selected_score_consistent=scores[o['selected_id']]==max(scores.values()))
            results.append(row)
            for action in protocol['actions']:
                id=action['id'];csv_rows.append(dict(condition=c,rep=rep,id=id,source_id=action['source_id'],
                    Pa=action['Pa'],Pb=action['Pb'],score=scores[id],rank=1+sum(v>scores[id] for v in scores.values()),selected=id==o['selected_id']))
            io.overlay(OUT/'current.png',plan(row['selected_action']),OUT/f'{name}_selected.png',f'{name}: {o["selected_id"]}; M score {row["M_score"]}, rank {row["M_rank"]}')
            cards.append(f'<article><h3>{name}: {o["selected_id"]} · M {row["M_score"]}分 / 第{row["M_rank"]}</h3><img src="{name}_selected.png"><p>{html.escape(o["selection_reason"])}</p></article>')
    aggregates={}
    for c in 'ABC':
        rows=[r for r in results if r['condition']==c]
        if rows:aggregates[c]=dict(n=len(rows),M_score_mean=statistics.mean(r['M_score'] for r in rows),
            M_score_range=[min(r['M_score'] for r in rows),max(r['M_score'] for r in rows)],
            M_rank_mean=statistics.mean(r['M_rank'] for r in rows),M_selected=sum(r['selected_id']==target for r in rows),
            M_margin_to_best_other_mean=statistics.mean(r['M_margin_to_best_other'] for r in rows),
            selections=dict(Counter(r['selected_id'] for r in rows)),judgments=dict(Counter(r['judgment'] for r in rows)),
            mean_seconds=statistics.mean(r['seconds'] for r in rows))
    summary=dict(complete=len(results)==9,aggregates=aggregates,results=results,
        summed_request_s=sum(r['seconds'] for r in results))
    if (OUT/'batch_metrics.json').exists():summary.update(io.read(OUT/'batch_metrics.json'))
    io.save(OUT/'summary.json',summary)
    if csv_rows:
        with (OUT/'scores.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    lines=['结构信息与自主检查 A/B/C 对照',
        '相同干净当前图和10个固定完整动作；M=C08，(702,626) → (540,600)，抓前90°。',
        'A：基线；B：加入独立结构识别第1次完整回答；C：要求在同一次请求中自主检查结构假设。',
        '每条件3次，相同模型gpt-6-astra / high、候选顺序、输出格式。仅当前图，无目标图、箭头、历史分数、失败结果或正确部位答案。',
        '所有请求独立；B只提供M的识别结果。C不访问B的结果，不额外调用模型。','',
        '组别 | M均分(范围) | M平均排名 | 选择M | 首选频数 | 平均请求秒数']
    for c,a in aggregates.items():lines.append(f"{c} | {a['M_score_mean']:.2f} ({a['M_score_range']}) | {a['M_rank_mean']:.2f} | {a['M_selected']}/{a['n']} | {a['selections']} | {a['mean_seconds']:.2f}")
    lines+=['','逐次记录：']
    for r in results:
        lines += [f"{r['condition']}_{r['rep']:02d}：M {r['M_score']}分 / 第{r['M_rank']}；{r['judgment']}；首选{r['selected_id']}；{r['seconds']:.2f}秒。"]
        for k,label in [('pa_identity','部位'),('visible_cloth_relations','布层关系'),('expected_relation_change','预期变化'),('main_risk','风险'),('uncertainty','不确定性'),('selection_reason','选择理由')]:
            lines.append(label+'：'+r['output'][k])
        lines.append('')
    lines += ['实验边界：',*protocol['caveats'],f"累计请求耗时 {summary['summed_request_s']:.2f}秒；整批并发墙钟 {summary.get('batch_wall_s',0):.2f}秒。",'B复用了既有视觉识别结果；上述耗时不包含此前生成该报告的51.18秒。']
    if (OUT/'interpretation.txt').exists():lines=[(OUT/'interpretation.txt').read_text(),'——协议与逐次原文——','',*lines]
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    page='<!doctype html><meta charset="utf-8"><title>结构信息与自主检查 ABC</title><style>body{font:16px system-ui;background:#f2f5f8;color:#182235;max-width:1400px;margin:auto;padding:24px}pre{white-space:pre-wrap;line-height:1.7;background:white;padding:24px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}article{background:white;padding:12px}img{width:100%}@media(max-width:950px){.grid{grid-template-columns:1fr}}</style>'
    page+='<h1>结构信息与自主检查 A/B/C</h1><p><a href="report.txt">纯文字报告</a> · <a href="scores.csv">全部评分</a> · '+ ' · '.join(f'<a href="prompt_{c}.txt">{c} prompt</a>' for c in 'ABC')+'</p>'
    if (OUT/'interpretation.txt').exists():page+='<pre>'+html.escape((OUT/'interpretation.txt').read_text())+'</pre>'
    page+='<p>以下箭头仅供人查看，模型输入均为干净原图。</p><div class="grid">'+''.join(cards)+'</div><details><summary>展开协议与逐次完整理由</summary><pre>'+html.escape('\n'.join(lines))+'</pre></details>'
    (OUT/'index.html').write_text(page)
    print('REPORT',len(results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','report']);globals()[p.parse_args().command]()
