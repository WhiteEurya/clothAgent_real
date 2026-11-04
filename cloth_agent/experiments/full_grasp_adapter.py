"""Compile a current-site proposal for the existing complete grasp executor.

This entry uses ExperimentRunner's simulated execution and original configuration
checks only. It never creates a controller connection or authorizes execution.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from cloth_agent.config import RobotConfig, WorkspaceBounds
from cloth_agent.experiment import ExperimentRunner
from cloth_agent.site_observation import DEFAULT_PROFILE, convert_pose_mm_deg, load_site
from .prepare_partitioned_trial import pose_errors, validate_source
from .robotcamcalib_adapter import vector


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'config/dual_arm.zed_dual_xarm7_20261103.json'


def compile_proposal(plan, profile, execution, state, *, config_path=DEFAULT_CONFIG):
    if plan['calibration_id'] != profile['calibration_id'] or execution['calibration_id'] != profile['calibration_id']:
        raise ValueError('Calibration identity mismatch')
    active, inactive = plan['selected_arm'], plan['inactive_arm']
    if {active, inactive} != {'left', 'right'}:
        raise ValueError('Expected one selected arm and one inactive arm')
    for arm in ('left', 'right'):
        site, config, recorded = profile['arms'][arm], execution['arms'][arm], state['arms'][arm]
        if config['axis'] != 7 or site['robot_axis'] != 7 or recorded['axis'] != 7:
            raise ValueError('Current adapter requires 7+7 axes')
        if recorded['status'] != 'READ_ONLY_CHECKS_PASS' or not (config['ip'] == site['robot_ip'] == recorded['ip']):
            raise ValueError(f'{arm}: device mismatch')
        for key in ('robot_sn', 'control_box_sn'):
            if recorded[key] != config[key] or recorded[key] != site['identity_verification'][key]:
                raise ValueError(f'{arm}: identity mismatch')
        for key in ('tcp_offset_mm_deg', 'world_offset_mm_deg'):
            expected = config['tcp_offset_mm_deg'] if key == 'tcp_offset_mm_deg' else [0.] * 6
            observed = vector(recorded[key], 6, key)
            if not np.allclose(observed, expected, atol=.01, rtol=0) or not np.allclose(observed, site[key], atol=.01, rtol=0):
                raise ValueError(f'{arm}: offset mismatch')
        vector(recorded['joints_deg'], 7, 'joints')
    operations = plan['operations']
    if not operations or any(op['arm'] != active for op in operations):
        raise ValueError('Only the selected arm may act')
    poses = [vector(op['pose_base_mm_deg'], 6, 'pose') for op in operations if op['kind'] == 'move_tcp']
    if not poses:
        raise ValueError('Proposal contains no motion')
    roll, pitch = poses[0][3:5]
    config = build_robot_config(execution, state, active, roll, pitch, config_path=config_path)
    return compile_operations(operations, profile, active, config, roll, pitch, execution['arms'][active]['gripper'])


def build_robot_config(execution, home_state, active, roll, pitch, *, config_path=DEFAULT_CONFIG):
    arm_config, limits = execution['arms'][active], execution['limits']
    workspace, gripper = arm_config['workspace'], arm_config['gripper']
    bounds = WorkspaceBounds.from_mapping({
        'boundary_policy': workspace.get('boundary_policy', 'legacy_complete'),
        'halfspaces_mm': workspace.get('halfspaces_mm', []),
        **{
        f'{axis}_{side}': workspace[f'{side}_mm'][index]
        for index, axis in enumerate(('x', 'y', 'z')) for side in ('min', 'max')
        },
    })
    from cloth_agent.dual_arm.config import DualConfig
    from cloth_agent.dual_arm.kinematics import ArmModel
    geometry = DualConfig.parse(execution, ROOT, homing_only=True, geometry_only=True)
    home_pose = ArmModel(geometry.arms[active]).forward(geometry.arms[active].home_joints)
    config = RobotConfig(
        robot_ip=arm_config['ip'], boundaries=bounds,
        collision_scene_path=str(Path(config_path).resolve()), collision_active_arm=active,
        init_joints_deg=tuple(arm_config['home_joints_deg']),
        init_pose_mm_deg=tuple(home_pose),
        orientation_roll_deg=float(roll), orientation_pitch_deg=float(pitch),
        expected_tcp_offset_mm_deg=tuple(arm_config['tcp_offset_mm_deg']),
        speed_mm_s=limits['cartesian_speed_mm_s'],
        acceleration_mm_s2=limits['cartesian_accel_mm_s2'],
        home_speed_deg_s=limits['joint_speed_deg_s'],
        home_acceleration_deg_s2=limits['joint_accel_deg_s2'],
        gripper_open=gripper['open'], gripper_close=gripper['close'],
        gripper_speed=gripper['speed'], gripper_open_tolerance_pulse=gripper['open_tolerance'],
        gripper_completion_timeout_s=gripper['timeout_s'],
    )
    return config


def compile_operations(operations, profile, active, config, roll, pitch, gripper):
    lines, mapping = ['def run():'], []
    for index, operation in enumerate(operations):
        kind = operation['kind']
        if kind == 'move_tcp':
            pose = vector(operation['pose_base_mm_deg'], 6, 'pose')
            common = vector(operation['pose_common_mm_deg'], 6, 'common pose')
            expected = convert_pose_mm_deg(profile, common, 'common', active)
            distance, angle = pose_errors(pose, expected)
            if distance > 1e-6 or angle > 1e-6:
                raise ValueError('Common/base transform mismatch')
            yaw = config.relative_yaw_from_absolute_deg(float(pose[5]))
            represented = [*pose[:3], roll, pitch, config.command_yaw_deg(yaw)]
            if pose_errors(pose, represented)[1] > 1e-6:
                raise ValueError(f'{operation["phase"]}: variable roll/pitch is unsupported by legacy move')
            for key, expected_value in (('speed_mm_s', config.speed_mm_s), ('acceleration_mm_s2', config.acceleration_mm_s2)):
                if key in operation and operation[key] != expected_value:
                    raise ValueError(f'{operation["phase"]}: per-action {key} differs from legacy global config')
            arguments = [float(value) for value in pose[:3]] + [yaw]
            call = 'move(' + ', '.join(repr(value) for value in arguments) + ')'
        elif kind == 'gripper' and operation['action'] in ('open', 'close'):
            action = operation['action']
            for key, expected_value in (('position_pulse', gripper[action]), ('speed_pulse_s', gripper['speed'])):
                if key in operation and operation[key] != expected_value:
                    raise ValueError(f'{operation["phase"]}: explicit gripper {key} mismatch')
            call = f'{action}_gripper()'
        else:
            raise ValueError(f'Unsupported operation: {kind}')
        lines.append('    ' + call)
        mapping.append({'index': index, 'phase': operation['phase'], 'call': call})
    return config, '\n'.join(lines) + '\n', mapping


def validate_offline(proposals, state_path, output, *, profile_path=DEFAULT_PROFILE, config_path=DEFAULT_CONFIG):
    profile = load_site(profile_path)
    plan = validate_source(Path(proposals), profile)
    execution = json.loads(Path(config_path).read_text())
    state = json.loads(Path(state_path).read_text())
    config, source, mapping = compile_proposal(plan, profile, execution, state, config_path=config_path)
    from cloth_agent.robot_collision import RobotCollision
    collision = RobotCollision(config_path, {k: a['joints_deg'] for k, a in state['arms'].items()},
                               plan['selected_arm'], root=ROOT)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    runner = ExperimentRunner(output, config, collision_checker=collision)
    source_path = runner.workspace / 'fold_sleeve.py'
    source_path.write_text(source)
    (output / 'robot_config.json').write_text(json.dumps(asdict(config), indent=2) + '\n')
    result = runner.run_experiment(source_path, real=False, confirmed=False)
    try:
        collision_report = collision.inspect_actions(config, result['requested_robot_actions'])
    except ValueError as exc:
        collision_report = {'status': 'COLLISION_REJECTED', 'error': str(exc), 'execution_allowed': False}
    report = {
        'status': 'SIMULATED_EXECUTION_COMPLETE' if result['execution_completed'] else 'PREFLIGHT_FAILED',
        'execution_allowed': False, 'physical_execution': False, 'robot_connections': 0,
        'backend': 'ExperimentRunner / RobotAPI / SimulatedBackend',
        'selected_arm': plan['selected_arm'], 'mapping': mapping,
        'source_proposals': str(Path(proposals).resolve()),
        'source_config': str(Path(config_path).resolve()),
        'source_state': str(Path(state_path).resolve()),
        'action_count': len(result['actual_robot_actions']),
        'parameter_policy': {
            'coordinates': 'Unchanged base-frame XYZ, mm; absolute yaw converted to Home-relative yaw',
            'height_adjustment_mm': 0,
            'motion_defaults_from': 'current execution config limits; proposal has no per-action override',
            'speed_mm_s': config.speed_mm_s, 'acceleration_mm_s2': config.acceleration_mm_s2,
            'gripper_open': config.gripper_open, 'gripper_close': config.gripper_close,
            'gripper_speed': config.gripper_speed,
        },
        'controller_path_check': {'status': 'NOT_RUN', 'reason': 'No robot connection permitted'},
        'external_collision_check': collision_report,
    }
    try:
        config.validate_for_real()
        report['original_execution_config_check'] = {'status': 'PASS'}
    except ValueError as exc:
        report['original_execution_config_check'] = {
            'status': 'BLOCKED', 'error': f'{type(exc).__name__}: {exc}',
            'boundaries': asdict(config.boundaries),
        }
    (output / 'integration_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposals', type=Path, required=True)
    parser.add_argument('--robot-state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = validate_offline(args.proposals, args.robot_state, args.output)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'SIMULATED_EXECUTION_COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
