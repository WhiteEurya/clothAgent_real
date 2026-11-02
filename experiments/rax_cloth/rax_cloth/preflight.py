"""Operator diagnostic: configuration by default; explicit stationary live reads.

Never arms, plans, enables motors, homes or dispatches a motion. Uses the same
NativePort as the RAX backend; close disconnects without sending a STOP command.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .baseline import register
from .session import write_json


def inspect(config_path, output, live=False, port_factory=None):
    output = Path(output).resolve()
    requested = output
    suffix = 0
    while True:
        try:
            output.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
            output = requested.with_name(requested.name+f'_{suffix:03d}')
    print(f'检查结果目录：{output}', file=sys.stderr, flush=True)
    report = {'live_requested': live, 'physical_actions': 0, 'task_success': None,
              'checks': {}, 'status': 'failed', 'output_directory': str(output)}
    checks = report['checks']
    port = None
    start = time.monotonic()
    try:
        checkout = register()
        checks['upstream_commit'] = subprocess.check_output(
            ['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
        checks['executables'] = {name: shutil.which(name) for name in ('git', 'codex', 'bwrap', 'socat')}
        from cloth_agent.codex_native_runtime import NativeRuntime
        from .xarm_driver import configure_arm6
        runtime = NativeRuntime(config_path)
        try:
            configure_arm6(runtime)
            cfg = runtime.config
            # NativeRuntime already validates its primitive-execution configuration.
            # Do not invoke the unrelated high-level garment grasp configuration.
            checks['axes'] = {name: arm.axis for name, arm in cfg.arms.items()}
            if checks['axes'] != {'left': 6, 'right': 7}:
                raise ValueError('expected left=arm6 and right=arm7')
            if (not getattr(cfg, 'path_validation_enabled', True)
                    or getattr(cfg, 'collision_envelopes_disabled_arms', ())):
                raise ValueError('existing safety validation must be fully enabled')
            checks['camera_mounts'] = {'camA': cfg.raw['camera']['mount_arm']}
            if checks['camera_mounts'] != {'camA': 'left'}:
                raise ValueError('camera ownership mismatch')
            checks['inactive_arm7'] = 'saved static obstacle only; not connected or measured'
            checks['configuration'] = 'passed; models/calibration loaded, no hardware connection'
        finally:
            runtime.close()
        if live:
            if port_factory is None:
                from .xarm_driver import NativePort
                port_factory = NativePort
            port = port_factory(config_path, output/'robot')
            checks['state_before'] = port.state()
            for camera in ('camA',):
                began = time.monotonic()
                obs = port.capture(camera, output/camera)
                port.validate_observation(obs, camera)
                checks[camera] = {'observation_id': obs.meta['observation_id'],
                                  'seconds': time.monotonic()-began,
                                  'calibration_id': obs.meta['calibration_id']}
            checks['state_after'] = port.state()
        report['status'] = 'live_reads_passed' if live else 'configuration_passed'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if port is not None:
            try:
                port.close()
            except Exception as exc:
                report['status'] = 'failed'
                report['close_error'] = f'{type(exc).__name__}: {exc}'
        report['elapsed_seconds'] = time.monotonic()-start
        report['limitations'] = 'Does not validate motion, stopping latency, operator presence or task success.'
        write_json(output/'preflight.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-config', type=Path, default=Path('config/dual_arm.local.json'))
    parser.add_argument('--output', type=Path, required=True, help='Diagnostic directory; existing paths get a new numbered suffix')
    parser.add_argument('--live-read', action='store_true', help='Connect only to arm6 and capture camA, without motion')
    args = parser.parse_args()
    report = inspect(args.robot_config.resolve(), args.output, args.live_read)
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report['status'] == 'failed' else 0)


if __name__ == '__main__':
    main()
