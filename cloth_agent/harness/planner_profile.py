"""Replay one original visual planner call, with producer-side event timing."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from ..planner_backend import RemoteClaudeBackend, parse_claude_json
from .common import read_json, write_json
from .planning_probe import load_case, validate_plan


def analyze_events(stdout):
    """Use remote timestamps only; never use buffered SSH receipt timestamps."""
    rounds, calls, blocks = [], {}, {}
    current = None
    ready = 0.0
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        t = event.get('_cloth_timing', {}).get('elapsed_s')
        if t is None:
            continue
        if event.get('type') == 'stream_event':
            stream = event['event']; kind = stream['type']
            if kind == 'message_start':
                current = {'round': len(rounds)+1, 'first_event_s': t,
                           'wait_before_first_event_s': max(0, t-ready),
                           'emission_s': None, 'blocks': [], 'tools': []}
                rounds.append(current)
                blocks = {}
            elif kind == 'content_block_start' and current is not None:
                block = stream['content_block']
                blocks[stream['index']] = {'kind': block['type'], 'start_s': t,
                                           'tool': block.get('name')}
            elif kind == 'content_block_stop' and current is not None:
                block = blocks.pop(stream['index'], None)
                if block is not None:
                    block.update(end_s=t, elapsed_s=t-block['start_s'])
                    current['blocks'].append(block)
            elif kind == 'message_stop' and current is not None:
                current.update(end_s=t, emission_s=t-current['first_event_s'])
                ready = max(ready,t)
            continue
        content = (event.get('message') or {}).get('content', [])
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get('type') == 'tool_use':
                key = block['id']
                if key in calls:
                    continue
                args = block.get('input', {})
                call = {'id':key, 'tool':block['name'], 'submitted_s':t,
                        'arguments':args, 'result_s':None, 'result_wait_s':None}
                if block['name'] == 'StructuredOutput':
                    call['candidate'] = args.get('selected_reference')
                calls[key] = call
                if current is not None:
                    current['tools'].append(key)
            elif block.get('type') == 'tool_result' and block.get('tool_use_id') in calls:
                call = calls[block['tool_use_id']]
                if call['result_s'] is None:
                    call.update(result_s=t, result_wait_s=t-call['submitted_s'],
                                is_error=bool(block.get('is_error')))
                    # Store schema errors, not image payloads or hidden reasoning.
                    if call['is_error']:
                        value = block.get('content')
                        call['error'] = value[:2000] if isinstance(value,str) else 'Tool returned an error'
                ready = max(ready,t)
    return {'rounds':rounds, 'tool_calls':list(calls.values()),
            'notes':[
                'Times are remote CLI line-emission observations, not provider-internal inference measurements.',
                'Wait before first event includes startup, prompt processing, queueing and API/network wait.',
                'Thinking block durations measure exposed block emission windows only; hidden reasoning topics cannot be timed.',
                'Tool result waits include dispatch and serialization; parallel tool intervals must not be summed.',
                'Task labels must come from public tools/arguments, not an invented breakdown of hidden reasoning.']}


def write_report(output, report):
    write_json(output/'profile.json',report)
    lines=['# 原始视觉规划流程：单次计时', '',
           f"状态：{report['status']}；总耗时：{report['elapsed_s']:.3f} 秒。", '',
           '使用保存的原始 prompt、system prompt、schema、context 和四张图；不接入观察 skill，不执行机器人。', '',
           '## 调用阶段', '', '| 阶段 | 秒 |', '|---|---:|']
    for key,value in report.get('stages_s',{}).items():
        lines.append(f'| {key} | {value:.3f} |')
    lines += ['', '下列 backend 计时存在包含关系（SSH 包含远端 Claude），不可直接相加。', '',
              '| 后端阶段 | 秒 |','|---|---:|']
    for key,value in report.get('backend_timings',{}).items():
        lines.append(f'| {key} | {value:.3f} |')
    lines += ['', '## 每轮模型响应', '',
              '等待包括 API/排队/网络/输入处理；流式输出窗口包括思考块、文本或工具参数生成，不等于纯推理。', '',
              '| 轮次 | 首事件之前等待 s | 流式输出 s | 本轮公开动作 |', '|---|---:|---:|---|']
    analysis=report.get('events',{});calls={c['id']:c for c in analysis.get('tool_calls',[])}
    for row in analysis.get('rounds',[]):
        names=[]
        for key in row['tools']:
            call=calls[key];args=call['arguments']
            detail = args.get('file_path') or args.get('image_id') or args.get('query') or ''
            if call['tool']=='StructuredOutput':
                detail=str((args.get('selected_reference') or {}).get('reference_id',''))
            names.append(call['tool']+' '+str(detail))
        emission=row['emission_s']
        lines.append(f"| {row['round']} | {row['wait_before_first_event_s']:.3f} | {emission if emission is None else round(emission,3)} | {'; '.join(names)} |")
    lines += ['', '## 工具结果与错误', '']
    for call in calls.values():
        lines.append(f"- +{call['submitted_s']:.3f}s `{call['tool']}`；等待结果 {call['result_wait_s']} 秒；错误：{call.get('error','无')}。")
    if report.get('error'):
        lines += ['', '错误：'+report['error']]
    lines += ['', '详细数值见 [profile.json](profile.json)，完整调用记录见 trace/。', '',
              '不能从这些日志得出“识别领口思考了 X 秒”这样的内部语义计时；可以准确定位哪一轮观察/工具调用/输出修正附近耗时最长。']
    (output/'profile.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest',type=Path,default=Path('results/harness_image_processing_20261002/manifest.json'))
    p.add_argument('--decision-id',default='fd15207c944ff34b5e17eb41')
    p.add_argument('--model',default='claude-opus-5')
    p.add_argument('--ssh-host',default='company-planner')
    p.add_argument('--timeout',type=int,default=900)
    a=p.parse_args(argv);output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();stages={}
    report={'status':'RUNNING','stages_s':stages,'model':a.model,
            'scope':'Original saved visual-planning call only; no live cameras, grounding, IK or robot.'}
    backend=RemoteClaudeBackend(ssh_host=a.ssh_host,timeout_s=a.timeout,record_event_timing=True)
    stdout=''
    try:
        t=time.monotonic();case,images=load_case(read_json(a.manifest),output,a.decision_id)
        stages['load_saved_case']=time.monotonic()-t
        snapshots=output/'source_snapshot';snapshots.mkdir()
        for source in [Path(__file__),Path('cloth_agent/planner_backend.py'),Path('cloth_agent/remote_output.py')]:
            shutil.copyfile(source,snapshots/source.name)
        report['source']=case['source'];report['original_budget']=case['original_budget']
        write_json(output/'profile.json',report)
        def progress(stage,event,duration_s=None,**details):
            print(json.dumps({'stage':stage,'event':event,'duration_s':duration_s},ensure_ascii=False),flush=True)
        backend.progress_callback=progress
        t=time.monotonic()
        try:
            response=backend.invoke(prompt=case['prompt'],image_paths=images,schema=case['schema'],
                system_prompt=case['system_prompt'],context_files=case['context_files'],
                max_turns=case['original_budget']['max_turns'],
                image_edit_limit=case['original_budget']['image_edit_limit'],model=a.model,
                overall_timeout_s=a.timeout,debug_dir=output/'trace',usage_run_dir=output,
                usage_stage='original_planner_profile')
            stdout=response.stdout
        finally:
            stages['planner_backend']=time.monotonic()-t
        t=time.monotonic();result=parse_claude_json(stdout)
        write_json(output/'result.json',result)
        report['validation']=validate_plan(result,case)
        stages['parse_and_validate']=time.monotonic()-t
        report.update(status='COMPLETED',selected_reference=result['selected_reference'])
    except Exception as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
    finally:
        # The original backend returns only the final envelope; the debug
        # spool, unlike response.stdout, contains all timestamped events.
        if (output/'trace/claude_stdout.txt').exists():
            stdout=(output/'trace/claude_stdout.txt').read_text()
        report.update(elapsed_s=time.monotonic()-start,backend_timings=backend.last_timings,
                      events=analyze_events(stdout))
        write_report(output,report)
        print(json.dumps({k:report[k] for k in ('status','elapsed_s')},ensure_ascii=False),flush=True)
    return 0 if report['status']=='COMPLETED' else 1


if __name__=='__main__':
    raise SystemExit(main())
