#!/usr/bin/env python3
"""Replay the recorded ten-action fold without cameras or model calls.

Default is a simulated run. --real --confirm-real executes the exact saved
coordinates through the normal workspace, controller IK and gripper gates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.config import RobotConfig
from cloth_agent.experiment import ExperimentRunner


# User-supplied coordinates, in robot base millimetres; no automatic rewriting.
RECORDED_SOURCE = """def run():
    move(410.527, -149.384, 116.690, 0.000)
    open_gripper()
    move(410.527, -149.384, 36.690, 0.000)
    close_gripper()
    move(410.527, -149.384, 106.690, 0.000)
    move(382.736, -79.166, 106.690, 0.000)
    move(382.736, -79.166, 42.690, 0.000)
    open_gripper()
    move(382.736, -79.166, 131.690, 0.000)
    home()
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', '--robot-config', type=Path,
                        default=Path('config/robot.example.json'), help='same robot configuration as the fold loop')
    parser.add_argument('--source', type=Path, help='saved run command source; defaults to the original recorded trajectory')
    parser.add_argument('--real', action='store_true', help='execute on xArm; requires --confirm-real')
    parser.add_argument('--confirm-real', action='store_true', help='confirm the recorded physical trajectory')
    args = parser.parse_args(argv)
    if args.real and not args.confirm_real:
        parser.error('--real requires --confirm-real')
    if args.confirm_real and not args.real:
        parser.error('--confirm-real requires --real')

    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    recorded_source = args.source.expanduser().resolve().read_text(encoding="utf-8") if args.source else RECORDED_SOURCE
    config = RobotConfig.load(PROJECT_ROOT, config_path)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir = PROJECT_ROOT / 'runs' / f'gripper_replay_{stamp}'
    run_dir.mkdir(parents=True, exist_ok=False)
    runner = ExperimentRunner(run_dir, config)
    source = runner.workspace / 'recorded_gripper_test.py'
    source.write_text(recorded_source, encoding='utf-8')
    summary = {'status': 'RUNNING', 'physical_execution': args.real,
               'run_dir': str(run_dir), 'source': str(source),
               'robot_config_path': str(config_path.resolve()), 'robot_config': asdict(config)}
    summary_path = run_dir / 'summary.json'
    def save_summary():
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    save_summary()
    print(f'Replay mode: {"REAL xArm" if args.real else "SIMULATED (no robot connection)"}', flush=True)
    print(f'Run: {run_dir}', flush=True)
    print(f'Command source: {args.source or "built-in recorded trajectory"}', flush=True)
    print('Uses the saved commands and coordinates, not the current garment image.', flush=True)
    started = time.monotonic()
    terminal = sys.stdout
    last_finished = started

    def action_finished(index, action):
        nonlocal last_finished
        now = time.monotonic()
        print(f'[replay {index+1}] {action["name"]} completed | '
              f'action_s={now-last_finished:.3f} total_s={now-started:.3f}', file=terminal, flush=True)
        with (run_dir / 'action_events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'index': index+1, 'elapsed_s': now-started, 'action': action}) + '\n')
        last_finished = now

    try:
        result = runner.run_experiment(source.name, real=args.real, confirmed=args.confirm_real,
            notes='Exact user-recorded gripper replay; no camera, Claude, Molmo or automatic recovery.',
            action_callback=action_finished)
        summary.update(status='COMPLETED' if result['execution_completed'] else 'FAILED',
                       robot_errors=result['robot_errors'],
                       result=str(runner.results_dir / 'recorded_gripper_test.json'))
        if result['robot_errors']:
            print('Replay failed: ' + '; '.join(result['robot_errors']), flush=True)
        return 0 if result['execution_completed'] else 1
    except KeyboardInterrupt:
        summary.update(status='INTERRUPTED')
        print('Replay interrupted. No extra release or Home command was sent.', flush=True)
        return 130
    except Exception as exc:
        summary.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        print(f'Replay failed: {summary["error"]}', file=sys.stderr, flush=True)
        return 1
    finally:
        summary['duration_s'] = time.monotonic()-started
        save_summary()
        print(f'Replay status: {summary["status"]}', flush=True)
        print(f'Trace: {runner.results_dir / "recorded_gripper_test.trace.json"}', flush=True)
        print(f'Summary: {summary_path}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
