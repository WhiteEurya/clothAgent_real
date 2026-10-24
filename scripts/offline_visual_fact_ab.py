"""One paired diagnostic: only one supplied garment-part fact differs. No robot."""
from pathlib import Path
import argparse
import copy
import html
import json
import random
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once
from cloth_agent.preference_one_shot import COMMON

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/fixed_grasp_actions_2325_round002_20261022'
OUT=ROOT/'reports/visual_fact_ab_M_20261022'
PROMPT=COMMON+'''
现在对给定的10个固定完整动作进行独立评价，并选出你认为最有利于展开衣服的一个。
这些动作是同一初态的互斥备选，不是连续执行。Pa、Pb和抓前角度均固定，不生成新的动作。
提供的候选列表已经固定，顺序和编号不代表好坏。不给你其他模型的评分或理由。
对每个动作给action_score（0–100，综合动作预期展开效果，分数不是成功概率）及一句中文reason。
另外详细评价指定的待评动作：先说明你认为Pa对应的衣物部位和可见布层关系，再说明Pa移动到Pb后预计哪些布料关系改变、主要风险和剩余不确定性。
target_judgment为HELPFUL（预期有助展开）、HARMFUL（预期主要增加折叠/遮挡）、LITTLE_CHANGE（主要移动或收益很小）、UNCERTAIN（不足以判断）之一。
最后selected_id选一个你认为最好的动作，selection_reason说明选择依据。所有理由基于本次输入，不引用其他试验结果。
'''


def schema(ids):
    item=dict(type='object',additionalProperties=False,properties=dict(
        id=dict(type='string',enum=ids),action_score=dict(type='integer',minimum=0,maximum=100),reason=dict(type='string')),
        required=['id','action_score','reason'])
    fields={key:dict(type='string') for key in ('pa_identity','visible_cloth_relations','expected_relation_change','main_risk','uncertainty','selection_reason')}
    fields.update(target_judgment=dict(type='string',enum=['HELPFUL','HARMFUL','LITTLE_CHANGE','UNCERTAIN']),
                  selected_id=dict(type='string',enum=ids),
                  scores=dict(type='array',items=item,minItems=len(ids),maxItems=len(ids)))
    return dict(type='object',additionalProperties=False,properties=fields,required=list(fields))


def prepare():
    OUT.mkdir(parents=True,exist_ok=False)
    source=io.read(SOURCE/'output.json')
    actions=copy.deepcopy(source['actions']);random.Random(20261022).shuffle(actions)
    actions=[dict(id=f'C{i:02d}',source_id=r['id'],Pa=r['grasp_px'],Pb=r['target_px'],
                  grasp_angle_deg=r['grasp_angle_deg'],displacement=[b-a for a,b in zip(r['grasp_px'],r['target_px'])])
             for i,r in enumerate(actions,1)]
    target=next(a for a in actions if a['source_id']==3)
    assert target['Pa']==[702,626] and target['Pb']==[540,600] and target['grasp_angle_deg']==90
    for name in ('current.png','goal.png'):shutil.copyfile(SOURCE/name,OUT/name)
    assert io.sha(OUT/'current.png')=='a522f3b0f0c23266c4f999e189d7d235c9821d617bc7e591f8acae58630b12fa'
    inputs=[{k:v for k,v in a.items() if k!='source_id'} for a in actions]
    text=f"待评动作：{target['id']}。\n固定候选：\n"+json.dumps(inputs,ensure_ascii=False)
    fact=f"仅补充一个已确认的视觉事实：待评动作 {target['id']} 的 Pa，并不是衣服向内翻折形成的自由折角，而是衣服下摆的正中间。"
    for version in ('A','B'):
        body=io.payload(OUT/'current.png',OUT/'goal.png',list(Image.open(OUT/'current.png').size))
        body['instructions']=PROMPT
        body['input'][0]['content'].append(dict(type='input_text',text=text))
        if version=='B':body['input'][0]['content'].append(dict(type='input_text',text=fact))
        body['text']['format'].update(name='fixed_action_visual_fact_diagnostic',schema=schema([a['id'] for a in actions]))
        io.save(OUT/f'prepared_{version}.json',body)
        (OUT/f'prompt_{version}.txt').write_text(PROMPT+'\n'+text+('\n'+fact if version=='B' else '')+'\n')
    a=io.read(OUT/'prepared_A.json');b=io.read(OUT/'prepared_B.json')
    assert b['input'][0]['content'].pop()==dict(type='input_text',text=fact)
    assert a==b,'Only the single visual-fact message may differ'
    io.save(OUT/'protocol.json',dict(created_at=io.utc(),source=str(SOURCE),
        source_observation='2026-10-21 23:25 run round_002 AFTER RGB',
        target=target,actions=actions,model=io.MODEL,effort=io.EFFORT,requests_per_version=1,
        dispatch_order=['A','B'],independent_contexts=True,robot_execution=False,retries=0,
        supplied_fact=fact,fact_source='User-provided garment identity; not independently verified by this experiment',
        input_images=['current.png','goal.png'],image_sha256={n:io.sha(OUT/n) for n in ('current.png','goal.png')},
        arrows_in_model_input=False,prior_scores_reasons_outcomes_supplied=False,
        comparison_scope='fixed existing complete-action ranking; not a replay of the earlier action-generation request',
        limitation='One sample per condition. Response to a supplied fact may reflect semantic conditioning or sampling, not proof of an internal perception/physics mechanism. If A does not reproduce the original preference, there is no reproduced choice to reverse.'))
    shutil.copyfile(__file__,OUT/'experiment_source.py')
    io.overlay(OUT/'current.png',plan(target),OUT/'target_action.png','Original selected M: (702,626) -> (540,600); user display only')
    print('PREPARED',OUT,'TARGET',target['id'],flush=True)


def plan(action):
    return dict(decision='ACTION',grasp_px=action['Pa'],target_px=action['Pb'],
                grasp_angle_deg=action['grasp_angle_deg'],motion_type='lift_pull_place',short_reason='')


def run():
    protocol=io.read(OUT/'protocol.json');headers=io.auth_headers()
    for version in protocol['dispatch_order']:
        folder=OUT/version
        if (folder/'attempt.json').exists():
            print('PRESERVED ATTEMPT',version,flush=True);continue
        print('REQUEST',version,flush=True)
        try:
            output,metrics=direct_once(folder,io.read(OUT/f'prepared_{version}.json'),headers,300)
            io.save(folder/'output.json',output)
            expected={r['id'] for r in protocol['actions']}
            if {r['id'] for r in output['scores']}!=expected or len(output['scores'])!=len(expected):
                raise ValueError('duplicate or missing candidate scores')
            print(json.dumps(dict(version=version,seconds=metrics['wall_latency_s'],selected=output['selected_id'],
                target_judgment=output['target_judgment'],pa_identity=output['pa_identity']),ensure_ascii=False),flush=True)
        except Exception as e:
            io.save(folder/'error.json',dict(error=f'{type(e).__name__}: {e}'))
            print('FAILED',version,type(e).__name__,str(e),flush=True)
        report()


def report():
    protocol=io.read(OUT/'protocol.json');target=protocol['target'];actions={r['id']:r for r in protocol['actions']}
    results={}
    for v in ('A','B'):
        folder=OUT/v
        if not (folder/'output.json').exists() or (folder/'error.json').exists():continue
        result=io.read(folder/'output.json');scores={r['id']:r for r in result['scores']}
        target_score=scores[target['id']]['action_score']
        selected=actions[result['selected_id']]
        results[v]=dict(output=result,seconds=io.read(folder/'model_metrics.json')['wall_latency_s'],
            target_score=target_score,target_rank=1+sum(r['action_score']>target_score for r in result['scores']),
            target_selected=result['selected_id']==target['id'],selected_action=selected,
            selected_score_consistent=scores[result['selected_id']]['action_score']==max(r['action_score'] for r in result['scores']))
        io.overlay(OUT/'current.png',plan(selected),OUT/f'{v}_selected.png',f'{v} selected {selected["id"]} (original point {selected["source_id"]})')
    summary=dict(complete=len(results)==2,results=results,target=target,
        total_request_s=sum(r['seconds'] for r in results.values()))
    if len(results)==2:
        a,b=results['A'],results['B']
        summary.update(target_score_delta=b['target_score']-a['target_score'],
            selected_action_changed=a['output']['selected_id']!=b['output']['selected_id'],
            A_reproduced_target_choice=a['target_selected'],
            target_choice_reversal=a['target_selected'] and not b['target_selected'])
    io.save(OUT/'summary.json',summary)
    lines=['视觉事实纠正 A/B 诊断（单图，每组一次，离线）',
        '原图：2026-10-21 23:25 run · round_002 动作后；沿用当时完整动作排序的M动作。',
        f"待评动作 {target['id']}：Pa={target['Pa']} → Pb={target['Pb']}；抓前角度90°。",
        '两版相同干净当前图、展平目标、10个固定完整动作、prompt与输出格式。没有旧评分、箭头、失败结果。',
        'B唯一新增：'+protocol['supplied_fact'],'']
    for version,result in results.items():
        o=result['output'];a=result['selected_action']
        lines += [f"{version}：{result['seconds']:.2f}秒；M评分{result['target_score']}/100，排名{result['target_rank']}/10；选择M：{result['target_selected']}。",
            f"部位识别：{o['pa_identity']}",f"布层理解：{o['visible_cloth_relations']}",
            f"动作判断：{o['target_judgment']}",f"预计变化：{o['expected_relation_change']}",
            f"主要风险：{o['main_risk']}",f"不确定性：{o['uncertainty']}",
            f"最佳动作：{a['id']}（原point {a['source_id']}），{a['Pa']} → {a['Pb']}，抓前角度{a['grasp_angle_deg']}°。",
            '选择理由：'+o['selection_reason'],'']
    lines += ['解释边界：B接受视觉事实并改变判断，是“视觉描述会影响动作判断”的线索，不能单独证明物理推理正确。',
        'B没有改变也不等于已证明物理模型缺失；需检查是否真正接受事实，以及当前单图是否仍有关键布层歧义。',
        '若A没有选M，则未重现历史选择，不能把B选其他动作直接当成纠错成功。只请求各一次，未估计随机波动。',
        '失败由用户指明；本次不向GPT提供结果，不执行机器人，也不改动alpha/Rθ。']
    if len(results)==2:
        a,b=results['A'],results['B']
        table=['全部候选评分（固定动作；0–100主观分数，不是成功率）',
               '编号 | 原point | Pa → Pb | A分数/排名 | B分数/排名']
        for action in protocol['actions']:
            entries=[]
            for result in (a,b):
                scores=result['output']['scores']
                score=next(r['action_score'] for r in scores if r['id']==action['id'])
                rank=1+sum(r['action_score']>score for r in scores)
                entries.append(f'{score}/{rank}')
            table.append(f"{action['id']} | {action['source_id']} | {action['Pa']} → {action['Pb']} | {' | '.join(entries)}")
        lines += ['',*table,f"两次模型请求总耗时：{summary['total_request_s']:.2f}秒；包含网络与生成，不是纯推理时间。"]
    interpretation=OUT/'interpretation.txt'
    if interpretation.exists():
        lines=[interpretation.read_text().strip(),'','——实验记录与模型原始理由——','',*lines]
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    cards=''.join(f'<section><h2>{v} 的首选动作</h2><img src="{v}_selected.png"></section>' for v in results)
    page='<!doctype html><meta charset="utf-8"><title>视觉事实纠正 A/B 诊断</title><style>body{font-family:system-ui;color:#182235;background:#f3f5f9;margin:24px}pre{white-space:pre-wrap;line-height:1.7}section{background:white;padding:18px;margin:14px 0}img{width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:900px){.pair{grid-template-columns:1fr}}</style>'
    page+='<h1>只纠正一个视觉事实，会不会改变动作判断？</h1><a href="report.txt">纯文字报告</a> · <a href="protocol.json">协议</a> · <a href="prompt_A.txt">A prompt</a> · <a href="prompt_B.txt">B prompt</a><section><pre>'+html.escape('\n'.join(lines))+'</pre></section>'
    page+='<div class="pair"><section><h2>干净输入图</h2><img src="current.png"></section><section><h2>固定待评M动作（仅展示给人）</h2><img src="target_action.png"></section></div><div class="pair">'+cards+'</div>'
    (OUT/'index.html').write_text(page)
    print('REPORT',summary['complete'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','report'])
    globals()[parser.parse_args().command]()
