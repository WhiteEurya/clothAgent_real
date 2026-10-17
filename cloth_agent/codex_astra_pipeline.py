"""Minimal task-to-local-Codex pipeline with reproducible run artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .run_storage import runs_root
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'gpt-6-astra'
EFFORT = 'high'
DEFAULT_TASK = '使用本项目已配置的相机和两台机械臂，把桌上的衣服整理并折叠整齐。请自主观察、规划、执行并检查结果。'


def build_prompt(task: str) -> str:
    if not task.strip():
        raise ValueError('任务描述不能为空')
    return task.strip() + '\n使用项目现有标定和观测找抓点；本轮只生成运动代码，不执行机器人动作。'


def build_command(binary: str, workspace: Path, output: Path, *, read_only: bool, session_id: str | None = None,
                  reasoning_effort: str = EFFORT) -> list[str]:
    command = [binary, 'exec', '--model', MODEL, '-c', f'model_reasoning_effort="{reasoning_effort}"',
               '--cd', str(workspace), '--json', '--color', 'never',
               '--output-last-message', str(output / 'answer.txt')]
    command += ['--sandbox', 'read-only'] if read_only else ['--approve-for-me']
    if session_id is not None:
        if not isinstance(session_id, str) or not session_id.strip() or session_id.startswith('-'):
            raise ValueError('invalid Codex session id')
        command += ['resume', session_id]
    return command + ['-']


def run(task: str, *, workspace: Path = ROOT, output: Path | None = None,
        binary: str = 'codex', timeout_s: float = 1800, read_only: bool = False,
        dry_run: bool = False, scene: Path | None = None, state: Path | None = None,
        images=(), raw_prompt: bool = False, session_id: str | None = None,
        trace_output: Path | None = None, trace_iteration: int | None = None,
        reasoning_effort: str = EFFORT) -> dict:
    if timeout_s <= 0:
        raise ValueError('timeout_s must be positive')
    prompt = task if raw_prompt else build_prompt(task)
    if read_only:
        prompt = '本次仅做只读分析，不连接或操作机器人，不修改项目文件。\n' + prompt
    else:
        from .codex_motion import CONTRACT
        prompt += '\n' + CONTRACT
    if (scene is None) != (state is None):
        raise ValueError('--scene and --state must be provided together')
    if scene is not None:
        scene, state = scene.resolve(), state.resolve()
        if not scene.is_file() or not state.is_file():
            raise ValueError('scene/state input file is missing')
        prompt += f'\n规划场景：{scene}\n初始关节状态：{state}'
    if trace_output is not None:
        from .planning_trace import trace_prompt
        prompt += trace_prompt()
    workspace = workspace.resolve()
    if not workspace.is_dir():
        raise ValueError(f'工作目录不存在: {workspace}')
    executable = shutil.which(binary)
    if executable is None:
        raise FileNotFoundError(f'找不到本机 Codex: {binary}')
    output = (output or runs_root(ROOT) / 'codex_astra' /
              datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = build_command(executable, workspace, output, read_only=read_only, session_id=session_id,
                            reasoning_effort=reasoning_effort)
    for image in images:
        image = Path(image).resolve()
        if not image.is_file():
            raise ValueError(f'image is missing: {image}')
        command[-1:-1] = ['--image', str(image)]
    request = dict(model=MODEL, reasoning_effort=reasoning_effort, task=task, prompt=prompt,
                   workspace=str(workspace), command=command, timeout_s=timeout_s,
                   read_only=read_only, dry_run=dry_run, requested_session_id=session_id)
    (output / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2)+'\n')
    (output / 'prompt.txt').write_text(prompt+'\n')
    result = dict(status='DRY_RUN' if dry_run else 'RUNNING', output_directory=str(output),
                  model=MODEL, reasoning_effort=reasoning_effort)
    if dry_run:
        (output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    started = time.monotonic()
    process = None
    trace = None
    try:
        with (output / 'events.jsonl').open('w') as events, (output / 'stderr.txt').open('w') as errors:
            if trace_output is not None:
                from .planning_trace import TraceCollector
                try:
                    trace=TraceCollector(output/'events.jsonl',trace_output,output.parent/'planning_trace.jsonl',
                                         trace_iteration,session_id)
                    trace.start()
                except Exception as exc:
                    result['planning_trace_error']=f'{type(exc).__name__}: {exc}'
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=events, stderr=errors,
                                       text=True, cwd=workspace, start_new_session=True)
            try:
                process.communicate(prompt, timeout=timeout_s)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                # Stop the Codex process group; this is not proof a controller has
                # stopped. Hardware outcomes must be read from execution records.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                result['status'] = 'TIMEOUT' if isinstance(exc, subprocess.TimeoutExpired) else 'INTERRUPTED'
            else:
                result['status'] = 'AGENT_FINISHED' if process.returncode == 0 else 'FAILED'
            result['returncode'] = process.returncode
        usage = {}
        turns = 0
        events_parsed = []
        for line in (output / 'events.jsonl').read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events_parsed.append(event)
            if event.get('type') == 'thread.started':
                result['session_id'] = event.get('thread_id')
            if event.get('type') == 'turn.completed':
                turns += 1
                for key, value in (event.get('usage') or {}).items():
                    if isinstance(value, (int, float)):
                        usage[key] = usage.get(key, 0) + value
        from .codex_event_status import completion_status
        completion = completion_status(events_parsed)
        result.update(usage=usage, completed_turns=turns, **completion)
        if (completion['errors'] or not completion['final_turn_completed']) and result['status'] == 'AGENT_FINISHED':
            result['status'] = 'FAILED'
        if session_id is not None and result.get('session_id') != session_id:
            result.update(status='FAILED', error='Codex resumed session ID mismatch or missing')
        # Agent process success is not a physical task success verdict.
        result['task_success'] = None
        if result['status'] == 'AGENT_FINISHED' and (not (output / 'answer.txt').is_file()
                or not (output / 'answer.txt').read_text().strip()):
            result.update(status='FAILED', error='Codex did not write a final answer')
        if result['status'] == 'AGENT_FINISHED' and not read_only:
            from .codex_motion import parse_code, handoff
            source = (output / 'answer.txt').read_text()
            (output / 'motion.py').write_text(source)
            calls = parse_code(source)
            if calls[0][0] == 'blocked':
                result.update(status='BLOCKED', reason=calls[0][1]['reason'])
            elif scene is None:
                result.update(status='CODE_READY', planner_status='requires --scene and --state')
            else:
                planned = handoff(output / 'motion.py', scene, state, output / 'motion_plan.json')
                result.update(status='PLANNED' if planned['success'] else 'PLANNING_FAILED',
                              physical_execution=False, planner_result=str(output / 'motion_plan.json'))
    except Exception as exc:
        result.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
    finally:
        if trace is not None:
            result['planning_trace']=trace.close()
        result['duration_s'] = time.monotonic()-started
        (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default=DEFAULT_TASK)
    parser.add_argument('--workspace', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--codex-binary', default='codex')
    parser.add_argument('--timeout-s', type=float, default=1800)
    parser.add_argument('--read-only', action='store_true', help='只读评估，不执行机器人任务')
    parser.add_argument('--dry-run', action='store_true', help='只保存提示词和调用参数')
    parser.add_argument('--scene', type=Path, help='昨天规划器使用的 collision scene JSON')
    parser.add_argument('--state', type=Path, help='初始关节 JSON: units=rad, left[6], right[7]')
    args = parser.parse_args(argv)
    try:
        result = run(args.task, workspace=args.workspace, output=args.output,
                     binary=args.codex_binary, timeout_s=args.timeout_s,
                     read_only=args.read_only, dry_run=args.dry_run, scene=args.scene, state=args.state)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] in ('AGENT_FINISHED', 'DRY_RUN', 'CODE_READY', 'PLANNED') else 1


if __name__ == '__main__':
    raise SystemExit(main())
