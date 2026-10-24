"""Unlearned GPT action choice with separate Ps/Pt Top5 evidence."""
import html
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from . import preference_one_shot as scoring
from .experiments import one_shot_goal_planner_probe_v1 as io

PROMPT=scoring.PT_PROMPT+'''
在给出所有候选的Pt后，另行选择你综合认为最值得实际执行的一个完整动作selected_id。
selected_id也可以是DONE，表示当前满足四项结构条件、不再执行动作：衣身无独立翻折布片覆盖、两袖主要向外、下摆不分成两片垂落、没有遮挡大块布面的翻折层。
第二张goal图是合格参考，图中长斜皱纹不构成继续动作的理由；不要把一条皱纹当成边界明确的翻折布片。
只看见T恤轮廓或领袖也不够；胸腹被翻折布片盖住、下摆出现深凹并分成两片时，仍应选能释放该结构的动作。
抓点阶段completion_check供复核，不要机械沿用；在本阶段也返回你自己核查后的completion_check。
你的四项检查都明确为true时选DONE；任一项false时选能解除具体结构问题的固定动作，并在selection_reason中指出该问题。
候选Pt是条件轨迹的相对收益分，Pt最高也不代表此时值得继续；不要为了用满轮数而行动。
Pt仍仅评价成功抓住后的条件展开收益；完整动作选择还需结合当前图可见的抓取可行性，不能自动把Pt最高等同于最佳完整动作。
不得改变候选坐标或夹爪角度。没有人工偏好分数或学习修正。
selection_reason用一句中文说明为何选择动作或DONE；选DONE时，其余说明字段写明“不执行”及残余褶皱为何可以接受。
选动作时，pa_identity说明抓点部位判断；expected_change说明预计解除哪处主要覆盖；uncertainty说明结构证据不足；failure_risk说明主要失败方式。
这些字段只需简短的依据和结论，不需要逐步思考过程。'''


def trajectory_schema(ids):
    s=scoring.trajectory_schema(ids)
    fields={k:dict(type='string',minLength=1) for k in ('selection_reason','pa_identity','expected_change','uncertainty','failure_risk')}
    fields['selected_id']=dict(type='string',enum=[*ids,'DONE'])
    fields['completion_check']=scoring.COMPLETION_CHECK_SCHEMA
    s['properties'].update(fields);s['required']+=list(fields)
    return s


def record_points(folder,before,output):
    top=sorted(output['points'],key=lambda p:(-p['Ps'],p['id']))[:5]
    io.save(folder/'points_top5.json',dict(decision=output['decision'],ranking='Ps descending; id breaks ties for display only',points=top))
    im=Image.open(before.image).convert('RGB');draw=ImageDraw.Draw(im)
    for n,p in enumerate(top,1):
        x,y=p['grasp_px'];draw.ellipse((x-7,y-7,x+7,y+7),fill='#00ffff',outline='black',width=2)
        draw.text((x+10,y-10),f"{n} {p['id']} Ps={p['Ps']}",fill='white',stroke_width=2,stroke_fill='black')
    im.save(folder/'points_top5.png')


def choose(folder,before,bundle,cancel):
    if cancel.is_set():raise KeyboardInterrupt('cancelled before automatic selection')
    if 'stop_plan' in bundle:
        io.save(folder/'trajectories_top5.json',dict(actions=[],reason=bundle['stop_plan']['decision']))
        return bundle['stop_plan']
    rows=bundle['candidates'];s=bundle['selection']
    check=s['completion_check']
    ready=all(check[key] for key in ('torso_not_covered','both_sleeves_out','hem_open','no_major_fold'))
    if (s['selected_id']=='DONE') != ready:
        raise ValueError('selected_id conflicts with structural completion checklist; no action dispatched')
    top=sorted(rows,key=lambda r:(-r['Pt'],r['id']))[:5]
    io.save(folder/'trajectories_top5.json',dict(ranking='Pt descending; id breaks ties for display only',actions=top))
    for i,row in enumerate(top,1):io.overlay(before.image,scoring.pref.plan_for(row),folder/f'trajectory_top{i}.png',f"Pt #{i}: {row['id']} Pt={row['Pt']}")
    if s['selected_id']=='DONE':
        evidence=dict(mode='GPT_EXPLICIT_SELECTION',selected_id='DONE',selected_action=None,
            candidate_count=len(rows),model_selection={k:v for k,v in s.items() if k!='scores'},
            completion_check=check,
            alpha_used=False,residual_used=False,learning=False,
            observation_sha256=io.sha(before.image),reason='garment sufficiently unfolded; no dispatch')
        if 'previous_effect' in bundle:
            evidence['previous_effect_assessment']=bundle['previous_effect']
            evidence['previous_action_context']=io.read(folder/'previous_action_context.json')
        io.save(folder/'automatic_selection.json',evidence)
        plan=dict(decision='DONE',grasp_px=None,target_px=None,motion_type=None,
                  grasp_angle_deg=None,short_reason=s['selection_reason'])
        io.validate_plan(plan,list(Image.open(before.image).size),require_grasp_rotation=True)
        return plan
    matches=[r for r in rows if r['id']==s['selected_id']]
    if len(matches)!=1:raise ValueError('GPT selected an absent or duplicate candidate')
    selected=matches[0]
    point_scores={r['point_id']:r['Ps'] for r in rows}
    evidence=dict(mode='GPT_EXPLICIT_SELECTION',selected_id=selected['id'],selected_action=selected,
        Ps_rank_among_eligible_points=1+sum(v>selected['Ps'] for v in point_scores.values()),
        Pt_rank=1+sum(r['Pt']>selected['Pt'] for r in rows),
        Pt_gap_to_best=max(r['Pt'] for r in rows)-selected['Pt'],
        candidate_count=len(rows),model_selection={k:v for k,v in s.items() if k!='scores'},
        completion_check=check,
        alpha_used=False,residual_used=False,learning=False,
        selection_is_not_argmax_Pt=True,observation_sha256=io.sha(before.image))
    if 'previous_effect' in bundle:
        evidence['previous_effect_assessment']=bundle['previous_effect']
        evidence['previous_action_context']=io.read(folder/'previous_action_context.json')
    io.save(folder/'automatic_selection.json',evidence)
    plan=scoring.pref.plan_for(selected);plan['short_reason']=s['selection_reason']
    io.validate_plan(plan,list(Image.open(before.image).size),require_grasp_rotation=True)
    return plan


def finalize_round(folder,item):
    """Persist observable outcomes without inventing grasp/success labels."""
    evidence=dict(execution=item['execution'],timing=item['timing'],grasp_success='UNREVIEWED',
        unfolding_progress='UNREVIEWED',model_reason_is_not_outcome=True)
    a=folder/'before/rgb.png';b=folder/'after/rgb.png'
    if a.exists() and b.exists():
        x=np.asarray(Image.open(a).convert('RGB'),dtype=float);y=np.asarray(Image.open(b).convert('RGB'),dtype=float)
        evidence['observation_hashes']=dict(before=io.sha(a),after=io.sha(b))
        if x.shape==y.shape:
            delta=np.abs(x-y)
            evidence['rgb_change']=dict(mean_abs=float(delta.mean()),q95_abs=float(np.quantile(delta,.95)),
                pixel_fraction_max_channel_over_20=float((delta.max(axis=2)>20).mean()),
                limitation='image change only; not cloth area, grasp success or flattening improvement')
    io.save(folder/'outcome_evidence.json',evidence)
    render_round(folder)


def render_round(folder):
    points=io.read(folder/'points_top5.json') if (folder/'points_top5.json').exists() else {}
    trajectories=io.read(folder/'trajectories_top5.json') if (folder/'trajectories_top5.json').exists() else {}
    selection=io.read(folder/'automatic_selection.json') if (folder/'automatic_selection.json').exists() else {}
    completion=io.read(folder/'completion_check.json') if (folder/'completion_check.json').exists() else {}
    outcome=io.read(folder/'outcome_evidence.json') if (folder/'outcome_evidence.json').exists() else {}
    import json
    page='<!doctype html><meta charset="utf-8"><style>body{font:17px system-ui;background:#eef2f6;padding:24px;color:#182235}img{width:100%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}pre{white-space:pre-wrap;background:white;padding:20px}</style>'
    page+='<h1>'+folder.name+' · GPT 自动选择</h1><h2>停止条件检查</h2><pre>'+html.escape(json.dumps(completion,ensure_ascii=False,indent=2))+'</pre>'
    page+='<pre>'+html.escape(json.dumps(selection,ensure_ascii=False,indent=2))+'</pre>'
    page+='<div class="grid">'
    if (folder/'context_previous_before.png').exists():
        context=io.read(folder/'previous_action_context.json')
        effect=io.read(folder/'previous_effect.json') if (folder/'previous_effect.json').exists() else {}
        page+='<section><h2>上一轮动作前图</h2><img src="context_previous_before.png"></section>'
        page+='<section><h2>上一轮动作及本轮效果判断</h2><pre>'+html.escape(json.dumps(
            dict(context=context,effect=effect),ensure_ascii=False,indent=2))+'</pre></section>'
    for name,label in [('before/rgb.png','动作前'),('after/rgb.png','动作后'),('points_top5.png','抓点 Top5 / Ps'),('proposed_action_overlay.png','GPT 选择执行的动作')]:
        if (folder/name).exists():page+=f'<section><h2>{label}</h2><img src="{name}"></section>'
    page+='</div><h2>轨迹 Top5 / Pt</h2><div class="grid">'
    for i,r in enumerate(trajectories.get('actions',[]),1):
        page+=f'<section><img src="trajectory_top{i}.png"><p>'+html.escape(r['trajectory_reason'])+'</p></section>'
    page+='</div><pre>'+html.escape(json.dumps(dict(points=points,outcome=outcome),ensure_ascii=False,indent=2))+'</pre>'
    (folder/'top5.html').write_text(page)


def append_report(root,summary):
    lines=['','自动候选模式：Ps/Pt各保存Top5；执行GPT明确选择的完整动作。未加载alpha或Rθ，未训练。',
           '停止标准：衣身没有大片覆盖、两袖主要向外、下摆大部分展开且没有大折层；四项都成立才DONE。抓点评分与轨迹评分阶段均记录检查结果。','']
    if summary.get('stateful_context_enabled'):
        lines+=['状态上下文：从第二个动作起，Ps与Pt都读取上一轮动作前原图、已执行Pa→Pb和本轮当前图；'
                'Ps另记录上一动作的局部效果判断。第一轮与无上下文基线相同。','']
    else:
        lines+=['无状态基线：每轮只读取当前图与目标图，不向模型提供上一动作。','']
    for item in summary['rounds']:lines.append(f"- [{item['round']} Top5及结果]({item['round']}/top5.html)")
    with (root/'report.md').open('a') as f:f.write('\n'.join(lines)+'\n')
