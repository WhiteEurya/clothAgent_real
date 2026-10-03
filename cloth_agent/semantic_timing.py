"""Observe declared planning phases; never infer topics from hidden thinking."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re


PHASES = {
    'context': '任务与约束读取',
    'orientation': '衣服方向判断',
    'correspondence': '跨图与坐标对应',
    'candidate': '抓取候选评估',
    'motion_target': '目标点与运动方案',
    'tool_recovery': '工具问题处理',
    'submission': '结果整理与提交',
}
PLANNING_CALLS = {'fold_supervisor', 'molmo_orientation', 'visual_planning', 'pixel_motion', 'skill_observation'}
CATEGORIES = ('thinking', 'text', 'tool_arguments', 'tool_wait', 'response_wait', 'other')
_MARKER = re.compile(r'^\[\[phase:([a-z_]+):(start|end)\]\]$')


def diagnostic_instructions():
    return """
Optional diagnostic phase timing (changes observability, not the task or schema):
In public progress text, emit a standalone [[phase:NAME:start]] line BEFORE
starting a task, and [[phase:NAME:end]] AFTER finishing that task. Names:
context = reading task/constraints; orientation = garment orientation/left-right;
correspondence = matching views/coordinate transforms; candidate = grasp selection;
motion_target = destination/motion proposal; tool_recovery = handling tool errors;
submission = assembling/submitting the final schema.
Mark only tasks actually performed, in their natural order. Do not add work just
to fill phases. Keep one phase active at a time; end it before switching tasks.
Revisiting a task uses another start/end pair with the same name. Keep tool use
inside the relevant phase; close submission immediately before submitting the
final result through the available output mechanism. If its validation fails,
open a new submission phase. Timing never requires finding or calling a tool.
These markers are brief status labels, NOT a request to expose chain-of-thought,
private reasoning, or a retrospective reasoning transcript. Never invent times,
backfill markers for earlier work, or claim the markers measure internal compute.
For this diagnostic only, progress markers are permitted before the final result
despite any 'JSON only' instruction. The FINAL result must still use the exact
original structured schema; put no markers or extra fields inside that result.
Use only available tools. Do not add tool calls or model turns just for timing.
""".strip()


def analyze_semantic_events(stdout, event_analysis=None):
    """Partition producer wall time using streamed *public text* markers only.

    Missing/mismatched end markers do not establish a measured phase. Complete
    assistant copies and thinking deltas are intentionally not parsed for labels.
    """
    if event_analysis is None:
        from .harness.planner_profile import analyze_events
        event_analysis = analyze_events(stdout)
    markers, warnings, buffers = [], [], {}
    last_t = 0.0
    valid_clock = True
    saw_timestamp = False

    def consume(line, t):
        match = _MARKER.fullmatch(line.strip())
        if match:
            phase, action = match.groups()
            markers.append({'phase': phase, 'action': action, 'time_s': t})

    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        t = event.get('_cloth_timing', {}).get('elapsed_s')
        if t is None:
            continue
        if type(t) not in (float, int) or not math.isfinite(t) or t < last_t:
            valid_clock = False
            continue
        saw_timestamp = True
        last_t = t
        if event.get('type') != 'stream_event':
            continue
        stream = event.get('event', {})
        kind, index = stream.get('type'), stream.get('index')
        if kind == 'message_start':
            buffers.clear()
        elif kind == 'content_block_start':
            block = stream.get('content_block', {})
            if block.get('type') == 'text':
                buffers[index] = block.get('text', '')
        elif kind == 'content_block_delta' and index in buffers:
            delta = stream.get('delta', {})
            if delta.get('type') != 'text_delta':
                continue
            buffers[index] += delta.get('text', '')
            while '\n' in buffers[index]:
                complete, buffers[index] = buffers[index].split('\n', 1)
                consume(complete, t)
        elif kind == 'content_block_stop' and index in buffers:
            consume(buffers.pop(index), t)

    occurrences = []
    active = None

    def finish(t, status):
        nonlocal active
        if active is not None:
            occurrences.append({**active, 'end_s': t, 'status': status,
                                'duration_s': t-active['start_s'] if status == 'OBSERVED' else None})
            active = None

    if not valid_clock:
        warnings.append('Non-monotonic/invalid producer timestamps; no semantic attribution.')
    for marker in markers if valid_clock else []:
        phase, action, t = marker['phase'], marker['action'], marker['time_s']
        if phase not in PHASES:
            warnings.append(f'Unknown phase {phase} at {t}; active interval is UNKNOWN.')
            finish(t, 'INVALID')
        elif action == 'start':
            if active is not None:
                warnings.append(f'Missing end for {active["phase"]} before {phase}.')
                finish(t, 'INCOMPLETE')
            active = {'phase': phase, 'start_s': t}
        elif active is not None and active['phase'] == phase:
            if t == active['start_s']:
                warnings.append(f'Coalesced start/end for {phase}; no measurable boundary interval.')
                finish(t, 'UNRESOLVED')
            else:
                finish(t, 'OBSERVED')
        else:
            warnings.append(f'Unmatched end for {phase} at {t}.')
            finish(t, 'INVALID')
    if active is not None:
        warnings.append(f'Missing final end for {active["phase"]}.')
        finish(last_t, 'INCOMPLETE')

    # Intersect phase windows with observable event types. Tool execution may
    # overlap streamed output; give output priority so every second counts once.
    intervals = {key: [] for key in CATEGORIES}
    for row in event_analysis.get('rounds', []):
        first = row['first_event_s']
        intervals['response_wait'].append((first-row['wait_before_first_event_s'], first))
        for block in row.get('blocks', []):
            category = {'thinking': 'thinking', 'text': 'text', 'tool_use': 'tool_arguments'}.get(block['kind'])
            if category:
                intervals[category].append((block['start_s'], block['end_s']))
    for call in event_analysis.get('tool_calls', []):
        if call.get('result_s') is not None:
            intervals['tool_wait'].append((call['submitted_s'], call['result_s']))

    totals = {key: {'phase': key, 'label': label, 'status': 'NOT_OBSERVED',
                    'visits': 0, 'wall_s': None, **{k+'_s': None for k in CATEGORIES}}
              for key, label in PHASES.items()}
    totals['UNKNOWN'] = {'phase': 'UNKNOWN', 'label': '未归因', 'status': 'UNKNOWN',
                         'visits': 0, 'wall_s': 0.0, **{k+'_s': 0.0 for k in CATEGORIES}}
    complete = [r for r in occurrences if r['status'] == 'OBSERVED']
    for row in complete:
        row.update({k+'_s': 0.0 for k in CATEGORIES})
        total = totals[row['phase']]
        if total['visits'] == 0:
            total.update(wall_s=0.0, status='OBSERVED', **{k+'_s': 0.0 for k in CATEGORIES})
        total['visits'] += 1
    boundaries = {0.0, last_t}
    for rows in intervals.values():
        for start, end in rows:
            boundaries.update((max(0.0, min(last_t, start)), max(0.0, min(last_t, end))))
    for row in complete:
        boundaries.update((row['start_s'], row['end_s']))
    points = sorted(boundaries)
    for start, end in zip(points, points[1:]):
        mid, duration = (start+end)/2, end-start
        category = next((k for k in CATEGORIES if any(a <= mid < b for a, b in intervals[k])), 'other')
        row = next((r for r in complete if r['start_s'] <= mid < r['end_s']), None)
        total = totals[row['phase'] if row else 'UNKNOWN']
        total['wall_s'] += duration
        total[category+'_s'] += duration
        if row is not None:
            row[category+'_s'] += duration
    if not saw_timestamp:
        for total in totals.values():
            total.update(wall_s=None, **{k+'_s': None for k in CATEGORIES})
    return {'status': 'NO_TIMESTAMPS' if not saw_timestamp else
            'INVALID_TIMESTAMPS' if not valid_clock else 'NO_MARKERS' if not markers else
            'PARTIAL' if warnings else 'OBSERVED',
            'elapsed_s': last_t if saw_timestamp else None,
            'attribution': 'model_declared_public_phase_windows',
            'markers': markers, 'occurrences': occurrences, 'totals': list(totals.values()),
            'warnings': warnings,
            'notes': [
                'Phase labels are model declarations, not verified internal reasoning topics.',
                'Times are producer-side CLI observations, not internal compute measurements.',
                'No retrospective topic inference; missing/incomplete intervals remain UNKNOWN.',
                'Thinking, text, tool arguments, tool wait, response wait, other partition each window.',
                'Tool wait excludes overlaps with emitted blocks; waits include dispatch overhead.',
                'End marker emission is included; final StructuredOutput after submission end is UNKNOWN.',
                'Prompt instrumentation can change model behavior and latency.']}


def write_semantic_report(directory, calls):
    directory = Path(directory)
    (directory/'semantic_timing.json').write_text(json.dumps(calls, ensure_ascii=False, indent=2)+'\n')
    lines = ['# Claude 声明的任务阶段耗时', '',
             '诊断模式保留单次调用，增加公开阶段标记，可能影响模型行为和耗时。',
             '阶段标签由模型声明；thinking 是流式块的观测窗口，不是内部纯计算时间。',
             '缺失、不匹配或未闭合的阶段不推断，时间计入 UNKNOWN。未观测到的阶段显示 —，不表示耗时为零。',
             '工具等待扣除了与输出块重叠的部分；各列可相加，重复访问同一阶段累计且另存逐次记录。', '',
             '| 调用 / iteration | 阶段 | 访问次数 | 总秒 | thinking | 文字 | 工具参数 | 工具等待 | 响应前等待 | 其他 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def number(value):
        return '—' if value is None else f'{value:.3f}'
    for call in calls:
        report = call.get('semantic', {})
        lines += [f"| {call.get('span_id')} {call.get('stage')} / {call.get('iteration')} | {r['label']} | {r['visits']} | " +
                  ' | '.join(number(r[k]) for k in ('wall_s', *(c+'_s' for c in CATEGORIES))) + ' |'
                  for r in report.get('totals', [])]
        for warning in report.get('warnings', []):
            lines.append(f"\n调用 {call.get('span_id')}：{warning}\n")
        if call.get('error'):
            lines.append(f"\n调用 {call.get('span_id')}：解析失败 {call['error']}，无可用归因。\n")
    lines += ['', '逐次阶段边界、UNKNOWN 和解析警告见 semantic_timing.json；原始调用路径包含在该文件中。']
    (directory/'semantic_timing.md').write_text('\n'.join(lines)+'\n')
