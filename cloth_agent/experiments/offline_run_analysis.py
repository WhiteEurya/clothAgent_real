"""On-demand, read-only analysis of saved episodes. No robot or model imports.

python -m cloth_agent.experiments.offline_run_analysis RUN --output REPORT_DIR
Optional labels: {"episode_0001/round_001": {"grasp": "EMPTY", "evidence": "..."}}
Labels describe human-reviewed evidence; execution completion never implies acquisition.
"""
from __future__ import annotations
import argparse
from collections import Counter
import html
import json
from pathlib import Path

GRASP_LABELS = {'CAUGHT', 'EMPTY', 'SLIPPED', 'MULTILAYER', 'UNCERTAIN'}


def read(path, warnings):
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('expected object')
        return value
    except (OSError, ValueError) as exc:
        warnings.append(f'{path}: {exc}')
        return {}


def collect(run, labels=None):
    run = Path(run).resolve()
    if not run.is_dir():
        raise ValueError(f'Run directory does not exist: {run}')
    warnings = []
    labels = labels or {}
    # Direct children only: recovery archives are attempts, never additional actions.
    episodes = [run] if list(run.glob('round_*')) else sorted(run.glob('episode_*'))
    rows = []
    for episode in episodes:
        for folder in sorted(episode.glob('round_*')):
            if not folder.is_dir():
                continue
            key = str(folder.relative_to(run))
            plan = read(folder/'plan.json', warnings)
            execution = read(folder/'execution.json', warnings)
            evidence = read(folder/'decision_evidence.json', warnings)
            timing = read(folder/'timing.json', warnings)
            status = execution.get('status', 'MISSING')
            dispatched = execution.get('physical_execution') is True
            annotation = labels.get(key, {})
            grasp = annotation.get('grasp', 'UNCERTAIN')
            if grasp not in GRASP_LABELS:
                raise ValueError(f'Invalid grasp label for {key}: {grasp}')
            if grasp != 'UNCERTAIN' and not annotation.get('evidence'):
                raise ValueError(f'Human grasp label requires evidence: {key}')
            if not dispatched and grasp != 'UNCERTAIN':
                raise ValueError(f'Cannot label unexecuted grasp: {key}')
            assets = {}
            for name in ('before/rgb.png', 'proposed_action_overlay.png', 'after/rgb.png',
                         'execution.json', 'timing.json', 'decision_evidence.json', 'target_depth.json',
                         'target_replan/feedback.json', 'depth_recovery/summary.json'):
                if (folder/name).exists():
                    assets[name] = str(folder/name)
            rows.append(dict(id=key, episode=str(episode), round=folder.name,
                decision=plan.get('decision'), pa=plan.get('grasp_px'), pb=plan.get('target_px'),
                reason=plan.get('short_reason'), structure=evidence.get('primary_structure'),
                expected_change=evidence.get('expected_change'), risk=evidence.get('failure_risk'),
                execution_status=status, dispatched=dispatched,
                error=execution.get('error'), failure_phase=execution.get('failure_phase'),
                grasp=grasp if dispatched else 'NOT_EXECUTED',
                grasp_evidence=annotation.get('evidence'), annotation=annotation,
                effect_model=None, effect_source=None, timing=timing, assets=assets,
                attempts=[str(p) for p in sorted(folder.glob('target_replan/**/plan.json'))]
                    + [str(p) for p in sorted(folder.glob('depth_recovery/**/plan.json'))],
                previous_context=read(folder/'previous_action_context.json', warnings),
                previous_effect=evidence.get('previous_effect')))
    by_key = {(r['episode'], r['round']): r for r in rows}
    for row in rows:
        context = row.pop('previous_context')
        effect = row.pop('previous_effect')
        if effect and context.get('source_round'):
            previous = by_key.get((row['episode'], context['source_round']))
            if previous and previous['execution_status'] == 'EXECUTED':
                previous['effect_model'] = effect
                previous['effect_source'] = row['id']
    completed = [r for r in rows if r['execution_status'] == 'EXECUTED']
    reviewed = [r for r in rows if r['grasp'] in GRASP_LABELS-{'UNCERTAIN'}]
    summary = dict(rounds=len(rows), dispatched=sum(r['dispatched'] for r in rows),
        completed_commands_actions=len(completed),
        rejected_before_dispatch=sum(r['execution_status']=='SAFETY_REJECTED' for r in rows),
        grasp_labels=dict(Counter(r['grasp'] for r in rows)), human_reviewed_grasps=len(reviewed),
        model_effects=dict(Counter((r['effect_model'] or {}).get('verdict','UNASSESSED') for r in completed)),
        # Timing fields overlap. Never sum planning_s with fast_planner_wall_s.
        timing_s={k:sum(r['timing'].get(k,0) or 0 for r in rows) for k in
                  ('total_round_s','planning_s','deliberative_planner_wall_s','robot_execution_s','gate_wall_s')})
    return dict(run=str(run), summary=summary, rounds=rows, warnings=warnings,
        limitations=['EXECUTED仅表示指令完成，不证明抓到衣服。',
                    '模型的改善/DONE判断未经人工确认，不作为真实成功率。',
                    '空抓、滑脱及多层抓取需要过程证据或人工标注；缺少证据保持不确定。',
                    '耗时为已保存轮次数据之和，不包含初始化和轮次间等待；缺失记录不补算。'])


def write_report(report, output):
    output = Path(output).resolve()
    # Never replace existing reports or acquisition artifacts.
    output.mkdir(parents=True, exist_ok=False)
    (output/'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    lines = ['# 离线运行分析', '', f"来源：{report['run']}", '', '## 统计', '',
             '```json', json.dumps(report['summary'], ensure_ascii=False, indent=2), '```', '',
             *[f'- {x}' for x in report['limitations']], '']
    cards = []
    for row in report['rounds']:
        text = (f"{row['id']}\nPa {row['pa']} → Pb {row['pb']}\n理由：{row['reason']}\n"
                f"执行：{row['execution_status']}；抓取：{row['grasp']}\n"
                f"错误：{row['error']}\n模型后续效果评价：{row['effect_model']}\n"
                f"效果评价来源：{row['effect_source']}\n人工证据：{row['grasp_evidence']}\n"
                f"耗时：{row['timing']}")
        lines += ['## '+row['id'], '', text, '']
        card = '<section><pre>'+html.escape(text)+'</pre>'
        for name, path in row['assets'].items():
            lines += [f'[{name}]({path})', '']
            uri = html.escape(Path(path).as_uri(), quote=True)
            card += f'<p><a href="{uri}">{html.escape(name)}</a></p>'
            if name.endswith('.png'):
                card += f'<img src="{uri}" loading="lazy">'
        cards.append(card+'</section>')
    lines += ['## 记录读取问题', '', *report['warnings']]
    (output/'report.md').write_text('\n'.join(lines))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>离线分析</title>'
        '<style>body{max-width:1200px;margin:24px auto;font:16px sans-serif}pre{white-space:pre-wrap}'
        'img{max-width:48%}section{border-top:1px solid #aaa;padding:20px 0}</style>'
        '<h1>离线运行分析</h1><pre>'+html.escape(json.dumps(report['summary'],ensure_ascii=False,indent=2))
        +'\n'+html.escape('\n'.join(report['limitations']))+'</pre>'+''.join(cards))
    template = {r['id']:dict(grasp='UNCERTAIN', evidence='') for r in report['rounds'] if r['dispatched']}
    (output/'human_labels_template.json').write_text(json.dumps(template,ensure_ascii=False,indent=2))
    return output/'report.md'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--labels',type=Path)
    args=parser.parse_args()
    labels=json.loads(args.labels.read_text()) if args.labels else None
    print(write_report(collect(args.run,labels),args.output))


if __name__=='__main__':
    main()
