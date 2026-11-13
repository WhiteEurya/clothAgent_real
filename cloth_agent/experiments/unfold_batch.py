"""Bounded sequential unfold runs using the existing single-run executable."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys

SUCCESS_STATUSES = {'MODEL_REPORTED_DONE', 'HUMAN_REPORTED_DONE'}
INCONCLUSIVE_STATUSES = {'ABSTAIN', 'PLANNING_REJECTED', 'ACTION_LIMIT'}
CONTINUE_STATUSES = SUCCESS_STATUSES | INCONCLUSIVE_STATUSES


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)


def execute(command):
    # Wait for normal single-run cleanup (including video) before starting another.
    with subprocess.Popen(command) as child:
        try:
            return child.wait()
        except KeyboardInterrupt:
            child.send_signal(signal.SIGINT)
            child.wait()
            raise


def run_batch(root, count, arguments, execute_one=execute, *, start_index=1, continue_from=None, manual_reset=False, confirm=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', run_limit=count, runs=[],
                  continuation_statuses=sorted(CONTINUE_STATUSES), automatic_reset=not manual_reset, manual_reset=manual_reset)
    if continue_from:
        report.update(continued_from=str(continue_from), start_run=start_index)
    arguments = list(arguments)
    if manual_reset and '--no-reset' not in arguments:
        arguments.append('--no-reset')
    confirm = input if confirm is None else confirm
    capture_goal = '--capture-goal' in arguments
    summary = root/'batch_summary.json'
    save(summary, report)
    try:
        for index in range(start_index, count+1):
            output = root/f'{root.name}_run_{index:02d}'
            report['current_run'] = index
            save(summary, report)
            continuing = continue_from is not None and index == start_index
            if manual_reset:
                report.update(status='WAITING_FOR_MANUAL_SETUP', current_run=index)
                save(summary, report)
                try:
                    prompt=('保留当前衣服状态，手离开台面；继续本轮（不做 reset）？' if continuing else '请手动摆好衣服并将手离开台面；直接开始 unfold（不做 reset）？')
                    answer=confirm(f'\n[Run {index}/{count}] {prompt}[y/N] ')
                except EOFError:
                    answer=''
                if answer.strip().lower()!='y':
                    report.update(status='STOPPED', reason='未确认手动摆放；没有启动下一轮')
                    break
                report['status']='RUNNING'
                report.setdefault('manual_setup_confirmations',[]).append(index)
                save(summary, report)
            run_arguments = [*arguments, '--continue-from', str(continue_from)] if continuing else arguments
            label = '重新观察 → 继续 unfold（不 reset）' if continuing else ('手动摆放 → unfold（不 reset）' if manual_reset else '自动 reset → unfold')
            print(f'\n[连续实验 {index}/{count}] {label}\n记录：{output}', flush=True)
            code = execute_one([sys.executable, '-u', '-m',
                'cloth_agent.experiments.partitioned_unfold', *run_arguments, '--output', str(output)])
            episode_file = output/'episode_summary.json'
            if not episode_file.is_file():
                report.update(status='ERROR', child_exit_code=code,
                    missing_episode_summary=str(episode_file),
                    reason=f'Run {index} 子进程退出（exit {code}），未生成 episode_summary.json；请查看上方子进程原始错误。常见原因是参数或续跑路径无效。')
                save(summary, report)
                break
            result = json.loads(episode_file.read_text(encoding='utf-8'))
            status = result.get('status', 'MISSING_STATUS')
            outcome_class = ('SUCCESS' if status in SUCCESS_STATUSES else
                             'INCONCLUSIVE' if status in INCONCLUSIVE_STATUSES else 'FAILED')
            report['runs'].append(dict(run=index, output=str(output), status=status,
                                      outcome_class=outcome_class,
                                      exit_code=code, physical_actions=result.get('physical_actions'),
                                      error=result.get('error')))
            save(summary, report)
            if code != 0 or status not in CONTINUE_STATUSES:
                report.update(status='STOPPED', reason=f'Run {index}: {status} (exit {code})')
                break
            if capture_goal and index == 1 and index < count:
                goal = output/'goal/target.png'
                if not goal.is_file():
                    raise FileNotFoundError(f'首轮目标图未保存，停止后续运行：{goal}')
                # Capture only before the first reset. Never recapture a folded garment.
                reused = []
                skip_value = False
                for arg in arguments:
                    if skip_value:
                        skip_value = False
                        continue
                    if arg == '--goal-image':
                        skip_value = True
                    elif arg != '--capture-goal' and not arg.startswith('--goal-image='):
                        reused.append(arg)
                arguments = [*reused, '--goal-image', str(goal)]
                report['shared_goal_image'] = str(goal)
                save(summary, report)
            print(f'[Run {index}/{count}] {status}；'+
                  (('等待下一轮手动摆放。' if manual_reset else '进入下一轮自动 reset。') if index < count else '已达到总轮数上限。'), flush=True)
        else:
            report['status'] = 'COMPLETED'
    except KeyboardInterrupt:
        report.update(status='INTERRUPTED', reason='用户或信号中断；不启动下一轮')
    except Exception as exc:
        report.update(status='ERROR', reason=f'{type(exc).__name__}: {exc}')
    finally:
        save(summary, report)
    print(f"\n[批次结束] {report['status']} · {len(report['runs'])}/{count} 个 run 已返回\n"
          f"原因：{report.get('reason', '达到总轮数上限')}\n记录：{summary}", flush=True)
    return 0 if report['status']=='COMPLETED' else 130 if report['status']=='INTERRUPTED' else 1


def continuation_arguments(directory):
    directory=Path(directory).resolve()
    protocol=json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    # Reuse the saved experiment settings and target for all subsequent resets.
    args=['--goal-image',str(directory/'goal/target.png'), '--config',protocol['config'],
          '--max-actions',str(protocol['max_actions']),
          '--planning-mode',protocol.get('planning_mode','skills'),
          '--planner-transport',protocol.get('planner_transport','openai-cli'),
          '--reset-profile',protocol.get('reset_profile','legacy'),
          '--reset-repeats',str(protocol.get('reset_repeats',1)),
          '--initialize-center-radius-mm',str(protocol.get('reset_radius_mm',0))]
    if protocol.get('camera_mode')=='dual-wrist':
        args += ['--camera-mode','dual-wrist','--fixed-wrist-config',protocol['fusion_config'],
                 '--topdown-config',protocol['topdown_config']]
    elif protocol.get('fusion_config'):
        args += ['--fusion-config',protocol['fusion_config']]
    else:args += ['--single-camera']
    return args


def batch_continuation(directory):
    directory=Path(directory).resolve()
    batch=json.loads((directory.parent/'batch_summary.json').read_text(encoding='utf-8'))
    if batch.get('status') in ('RUNNING','STARTED'):
        raise ValueError('source batch is still active')
    runs=batch['runs']
    if not runs or Path(runs[-1]['output']).resolve()!=directory:
        raise ValueError('continue-batch must reference the last attempted run')
    args=continuation_arguments(directory)
    completed=runs[-1]['status'] in CONTINUE_STATUSES
    return int(runs[-1]['run'])+int(completed), int(batch['run_limit']), args


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--continue-batch',type=Path)
    parser.add_argument('--continue-from',type=Path,help='resume this run first without reset; --runs includes this resumed run')
    parser.add_argument('--manual-reset',dest='manual_reset',action='store_true')
    parser.add_argument('--auto-reset',dest='manual_reset',action='store_false')
    parser.set_defaults(manual_reset=False)
    parser.add_argument('--runs', type=int, default=6)
    parser.add_argument('--start-run', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args, rest=parser.parse_known_args(argv)
    if args.manual_reset and any(flag in rest for flag in ('--capture-goal','--reset-only')):
        parser.error('手动摆放模式跳过 reset；请用 --goal-image 指定已保存的展开目标图，不能同时 capture-goal/reset-only')
    if args.start_run < 1 or args.start_run > args.runs:
        parser.error('--start-run must be between 1 and --runs (last run number)')
    if args.continue_batch and args.start_run != 1:
        parser.error('--start-run cannot combine with --continue-batch')
    if args.runs < 1:
        parser.error('--runs must be positive')
    if args.continue_from:
        if args.continue_batch or args.start_run != 1 or any(x in rest for x in ('--capture-goal','--capture-only','--reset-only')):
            parser.error('continue-from cannot combine with continue-batch/start-run/capture-goal/capture-only/reset-only')
        from .partitioned_unfold import continuation_state, main as single
        state=continuation_state(args.continue_from)
        if state['start_round']>=state['protocol']['max_actions']:
            parser.error('source run has reached its action limit')
        # First run always resumes saved state/settings; following runs reuse its
        # target and camera setup, but follow the selected normal reset mode.
        rest=[*rest,*continuation_arguments(args.continue_from)]
        supplied=list(sys.argv[1:] if argv is None else argv)
        if not any(x in supplied for x in ('--manual-reset','--auto-reset')):
            previous_batch=args.continue_from.parent/'batch_summary.json'
            if previous_batch.exists():
                args.manual_reset=bool(json.loads(previous_batch.read_text(encoding='utf-8')).get('manual_reset',False))
        if '--execute' not in rest or '--help' in rest:
            return single([*rest,'--continue-from',str(args.continue_from),'--output',str(args.output)])
        if '--no-reset' in rest and args.runs>1 and not args.manual_reset:
            parser.error('multiple resumed runs require manual-reset or normal automatic reset')
    if args.continue_batch:
        if '--continue-from' in rest or '--capture-goal' in rest or '--no-reset' in rest:
            parser.error('continue-batch cannot combine with continue-from/capture-goal/no-reset')
        start_index, count, inherited = batch_continuation(args.continue_batch)
        rest = [*rest, *inherited]
        if start_index>count:parser.error('source batch has no remaining runs')
        previous=json.loads((args.continue_batch/'episode_summary.json').read_text(encoding='utf-8'))
        resume_source=None if previous.get('status') in CONTINUE_STATUSES else args.continue_batch
        if '--execute' not in rest:
            from .partitioned_unfold import main as single
            return single([*rest, *(['--continue-from',str(resume_source)] if resume_source else []), '--output', str(args.output)])
    # Preserve existing help, dry-run, capture-only and explicit single-run modes.
    if '--help' in rest or '--execute' not in rest or '--capture-only' in rest or (args.runs == 1 and not args.continue_batch and not args.continue_from and not args.manual_reset):
        from .partitioned_unfold import main as single
        return single([*rest, '--output', str(args.output)])
    if '--no-reset' in rest and not args.manual_reset and not args.continue_from:
        parser.error('连续实验每轮需要 reset；保留当前衣物状态请使用 --runs 1 --no-reset')
    old=signal.getsignal(signal.SIGTERM)
    def terminate(*_):
        raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, terminate)
    try:
        if args.continue_from:
            return run_batch(args.output,args.runs,rest,continue_from=args.continue_from,manual_reset=args.manual_reset)
        if args.continue_batch:
            return run_batch(args.output, count, rest, start_index=start_index, continue_from=resume_source, manual_reset=args.manual_reset)
        return run_batch(args.output, args.runs, rest, start_index=args.start_run, manual_reset=args.manual_reset)
    finally:
        signal.signal(signal.SIGTERM, old)


if __name__ == '__main__':
    raise SystemExit(main())
