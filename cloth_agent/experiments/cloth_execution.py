"""Unified offline entry for grasp and PyRoki dual-arm Home validation.

The CLI never connects to hardware. execute_home accepts an existing coordinator
and replans from its live snapshots before invoking the checked executor.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.geometry import DualArmError, finite
from cloth_agent.dual_arm.kinematics import ArmModel
from cloth_agent.dual_arm.pyroki_home import plan_home, execute_home
from cloth_agent.site_observation import load_site
from .full_grasp_adapter import DEFAULT_CONFIG, ROOT, validate_offline


def validate_home(home_state_path, output, config_path=DEFAULT_CONFIG):
    execution = json.loads(Path(config_path).read_text())
    state = json.loads(Path(home_state_path).read_text())
    profile = load_site()
    if execution['calibration_id'] != profile['calibration_id']:
        raise ValueError('Calibration identity mismatch')
    initial = {}
    for arm in ('left', 'right'):
        recorded, configured, site = state['arms'][arm], execution['arms'][arm], profile['arms'][arm]
        if not (recorded['axis'] == configured['axis'] == site['robot_axis'] == 7):
            raise ValueError(f'{arm}: expected seven axes')
        if recorded['status'] != 'READ_ONLY_CHECKS_PASS':
            raise ValueError(f'{arm}: invalid reference state')
        if not (recorded['ip'] == configured['ip'] == site['robot_ip']):
            raise ValueError(f'{arm}: device mismatch')
        for key in ('robot_sn', 'control_box_sn'):
            if not (recorded[key] == configured[key] == site['identity_verification'][key]):
                raise ValueError(f'{arm}: identity mismatch')
        for key, expected in (('tcp_offset_mm_deg', configured['tcp_offset_mm_deg']),
                              ('world_offset_mm_deg', [0.] * 6)):
            if not np.allclose(recorded[key], expected, atol=.01, rtol=0) or not np.allclose(recorded[key], site[key], atol=.01, rtol=0):
                raise ValueError(f'{arm}: offset mismatch')
        if recorded.get('errors_warnings', [0, 0])[0] != 0 or recorded.get('state', 2) not in (0, 2):
            raise ValueError(f'{arm}: recorded controller not ready')
        initial[arm] = finite(recorded['joints_deg'], (7,), 'recorded current joints')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        'status': 'BLOCKED',
        'backend': 'PyRoki / joint_motion / prepare_segments / execute_sequential_home',
        'physical_execution': False, 'execution_allowed': False, 'robot_connections': 0,
        'reference_state': str(home_state_path),
        'initial_joints_deg': {key: value.tolist() for key, value in initial.items()},
        'controller_path_check': 'NOT_RUN', 'dual_arm_collision_check': 'NOT_RUN',
        'state_policy': 'Recorded actual state for offline validation only; execution replans from live snapshots.',
    }
    try:
        config = DualConfig.load(config_path, root=ROOT, homing_only=True)
        config.execution_mode = 'controller_sequential'
        models = {key: ArmModel(arm) for key, arm in config.arms.items()}
        report['dual_arm_collision_check'] = 'RUNNING'
        program, segments, attempts = plan_home(initial, config, models)
        report.update(status='PATH_CHECK_PASS', dual_arm_collision_check='PASS', attempts=attempts,
                      segments=[segment.to_dict() for segment in segments])
        for index, motion in enumerate(program[:-1]):
            np.savez_compressed(output / f'home_{index}.npz', times=motion.times,
                                **{key: value for key, value in motion.joints.items()})
        try:
            config.require_real(homing=True)
        except DualArmError as exc:
            report['real_execution_blocker'] = str(exc)
    except (DualArmError, ValueError) as exc:
        if report['dual_arm_collision_check'] == 'RUNNING':
            report['dual_arm_collision_check'] = 'NOT_COMPLETED'
        report['error'] = f'{type(exc).__name__}: {exc}'
    (output / 'integration_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    home = commands.add_parser('home')
    home.add_argument('--robot-state', '--home-state', dest='home_state', type=Path, required=True)
    fold = commands.add_parser('fold')
    fold.add_argument('--proposals', type=Path, required=True)
    fold.add_argument('--robot-state', type=Path, required=True)
    for command in (home, fold):
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    if args.action == 'home':
        report = validate_home(args.home_state, args.output, args.config)
    else:
        report = validate_offline(args.proposals, args.robot_state, args.output, config_path=args.config)
    print(json.dumps(report, indent=2))
    return 2 if report['status'] == 'BLOCKED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
