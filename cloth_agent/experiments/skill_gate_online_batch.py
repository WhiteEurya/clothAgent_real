"""Six-run pilot with manual garment restoration between runs."""
from pathlib import Path
import argparse
import json
import shutil
import signal
import traceback
from datetime import datetime, timezone

from .discovery_data import read, save, file_hash
from .skill_gate_online import ROOT, PROBABILITIES, run_episode


def validate_config(path):
    cfg = read(path)
    expected = dict(mode='PRE_PLANNING_RANDOMIZATION', run_count=6,
        probabilities=PROBABILITIES, max_actions=12,
        max_planning_requests_per_iteration=1, automatic_initialization=True,
        fallback_to_other_skill=False, garment_id='black_mario', gate_training=False)
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError('pilot protocol mismatch: '+key)
    if type(cfg.get('seed')) is not int or cfg['seed'] < 0:
        raise ValueError('fixed nonnegative seed required')
    if not cfg.get('goal_image'):
        raise ValueError('black Mario goal missing')
    goal = ROOT/cfg['goal_image']
    if not goal.is_file() or file_hash(goal) != cfg.get('goal_sha256'):
        raise ValueError('black Mario goal hash mismatch')
    manifest = read(goal.parent/'manifest.json')
    if manifest.get('garment_id') != 'black_mario' or manifest.get('sha256') != cfg['goal_sha256']:
        raise ValueError('black Mario goal manifest mismatch')
    return cfg, goal


def confirm_manual_reset(run_id):
    """No motion here; require an explicit operator acknowledgement."""
    print(f'\a[Run {run_id}/6 准备] 请手动把衣服完全展开并放回台面中央。'
          '手离开工作区后输入 y；随后才会执行原有自动抓起放下初始化。', flush=True)
    while True:
        try:
            answer = input('已手动复位，继续？[y/N]：').strip().lower()
        except EOFError:
            return False
        if answer == 'y':
            return True
        if answer in ('', 'n', 'stop', 'q', 'quit'):
            return False
        print('请输入 y 或 n；直接回车默认不继续。', flush=True)


def run_batch(root, cfg, goal, execute_one, confirm_reset=None):
    """Bounded sequential orchestration; errors stop the entire batch."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    save(root/'protocol.json', dict(cfg, manual_restoration_before_each_run=True), immutable=True)
    summary = dict(status='RUNNING', run_limit=6, runs=[], gate_training=False)
    save(root/'batch_summary.json', summary)
    try:
        for run_id in range(1, 7):
            summary.update(status='WAITING_FOR_MANUAL_RESET', next_run_id=run_id,
                           reason='请手动完全展开衣物并放回台面中央，确认后才允许下一次自动初始化')
            save(root/'batch_summary.json', summary)
            print('\a'+summary['reason'], flush=True)
            if confirm_reset is None or confirm_reset(run_id) is not True:
                break
            save(root/f'manual_reset_{run_id:02d}.json', dict(
                status='OPERATOR_CONFIRMED', run_id=run_id,
                confirmed_at=datetime.now(timezone.utc).isoformat()))
            summary.update(status='RUNNING', reason=None)
            folder = root/f'run_{run_id:02d}'
            (folder/'goal').mkdir(parents=True)
            shutil.copyfile(goal, folder/'goal/target.png')
            if file_hash(folder/'goal/target.png') != cfg['goal_sha256']:
                raise ValueError('goal changed before run')
            save(folder/'goal/manifest.json', dict(garment_id='black_mario',
                 source=str(goal), sha256=cfg['goal_sha256']))
            save(folder/'hem_sleeve_pull_rule.json', dict(enabled=False, rgb_only=True))
            # Separate deterministic reset stream; never changed using results.
            reset_seed = cfg['seed']+run_id
            row = dict(run_id=run_id, output=str(folder), status='STARTED', reset_seed=reset_seed)
            summary['runs'].append(row)
            save(root/'batch_summary.json', summary)
            print(json.dumps(row), flush=True)
            print(f'[Run {run_id}/6] 自动初始化开始；reset seed={reset_seed}', flush=True)
            result = execute_one(folder, run_id, reset_seed)
            row.update(status=result['status'], physical_actions=result.get('physical_actions', 0))
            row['reason'] = result.get('error') or result.get('initialization', {}).get('error')
            print(f"[Run {run_id}/6] 结束：{result['status']}；原因：{row['reason'] or '无异常说明'}", flush=True)
            initialization = result.get('initialization', {})
            failed_request = any(read(p).get('failure_kind') == 'PLANNING_EXCEPTION'
                                 for p in folder.glob('round_*/exploration.json'))
            if (initialization.get('status') != 'EXECUTED' or failed_request or
                    result['status'] not in ('MODEL_REPORTED_DONE', 'ACTION_LIMIT', 'ABSTAIN', 'PLANNING_REJECTED')):
                summary.update(status='STOPPED', reason=result.get('error') or 'reset/execution/request failure')
                break
            save(root/'batch_summary.json', summary)
        else:
            summary['status'] = 'COMPLETED_SIX_RUNS'
            print('六次 run 已结束；如需复位，请手动整理衣物。不会自动 reset。', flush=True)
    except KeyboardInterrupt:
        summary['status'] = 'INTERRUPTED'
        if summary['runs'] and summary['runs'][-1]['status']=='STARTED':
            summary['runs'][-1]['status']='INTERRUPTED'
        raise
    except Exception as exc:
        (root/'batch_exception.txt').write_text(traceback.format_exc(), encoding='utf-8')
        summary.update(status='ERROR', error=f'{type(exc).__name__}: {exc}')
        if summary['runs'] and summary['runs'][-1]['status']=='STARTED':
            summary['runs'][-1].update(status='ERROR',error=summary['error'])
        raise
    finally:
        save(root/'batch_summary.json', summary)
        from .skill_gate_online_status import describe_batch, print_report
        report = describe_batch(root)
        save(root/'diagnosis.json', report)
        print_report(report)
    return summary


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'config/skill_gate_online_pilot01.json')
    parser.add_argument('--output', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--status', metavar='BATCH', help='read saved batch or latest; no hardware')
    parser.add_argument('--json', action='store_true', help='machine-readable status output')
    args = parser.parse_args(argv)
    if args.status:
        from .skill_gate_online_status import resolve_batch, describe_batch, print_report
        report = describe_batch(resolve_batch(args.status, ROOT/'reports'))
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print_report(report)
        return 0
    cfg, goal = validate_config(args.protocol)
    if not args.execute:
        print(json.dumps(dict(status='DRY_RUN_OK', goal=str(goal), run_count=6,
                              robot_calls=0, camera_calls=0, model_calls=0)))
        print('配置与目标哈希检查通过。未验证设备在线、IK可达或网络；启动：bash scripts/run_skill_gate_online.sh start', flush=True)
        return 0
    from .partitioned_unfold import configure_runtime, PartitionedReplayRuntime, DEFAULT_CONFIG
    from .one_shot_goal_full_episode_v1 import direct_once
    from .one_shot_goal_planner_probe_v1 import auth_headers
    from cloth_agent.dual_cloud_observation import load_config
    from cloth_agent.perception import PerceptionConfig
    from cloth_agent.unfold_video import UnfoldVideo
    import numpy as np
    headers = auth_headers()
    cfg = dict(cfg, execute=True)
    root = (args.output or ROOT/'reports'/('skill_gate_online_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    print('实验目录：'+str(root), flush=True)
    def execute_one(folder, run_id, reset_seed):
        runtime = configure_runtime(PartitionedReplayRuntime(DEFAULT_CONFIG, planning_only=False))
        try:
            runtime.dual_cloud_config = load_config(ROOT/'config/fusion.unfold_dual.json')
            runtime.video_recorder = UnfoldVideo(
                PerceptionConfig.load(ROOT, ROOT/'config/perception.zed_dual_xarm7_20261103.json'),
                ROOT/'videos'/root.name/folder.name, folder)
            runtime.video_recorder.start()
            return run_episode(runtime, folder, seed=cfg['seed'], run_id=run_id,
                headers=headers, timeout=900, direct=direct_once, initialize_seed=reset_seed,
                radius_mm=float(np.random.default_rng(reset_seed).uniform(20, 40)))
        finally:
            runtime.close()
    old = signal.getsignal(signal.SIGTERM)
    def terminate(*_):
        raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, terminate)
    try:
        result = run_batch(root, cfg, goal, execute_one, confirm_reset=confirm_manual_reset)
        print(json.dumps(dict(output=str(root), **result)), flush=True)
        return 0 if result['status']=='COMPLETED_SIX_RUNS' else 1
    finally:
        signal.signal(signal.SIGTERM, old)


def main(argv=None):
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print('实验已中断；已保存当前批次日志。', flush=True)
        return 130
    except Exception as exc:
        print(f'实验启动或运行失败：{type(exc).__name__}: {exc}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
