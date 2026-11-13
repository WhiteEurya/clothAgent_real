"""Read-only controller diagnostics for the calibrated ZED / dual-xArm7 site.

CLI/default reads use getters only. Replay may explicitly opt into fault-free
STOP recovery; it restores position mode after Servo, without clearing faults
or sending motion commands.
This is a commissioning report, not execution authorization or collision proof.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

import numpy as np

from .site_observation import DEFAULT_PROFILE, load_site, convert_pose_mm_deg


def value(response, name):
    if not isinstance(response, (tuple, list)) or len(response) != 2 or response[0] != 0:
        raise RuntimeError(f'{name}: invalid controller response {response}')
    return response[1]


def recover_stopped_controller(robot, cancel):
    """Explicit startup recovery from stopped position/Servo mode only."""
    if (cancel.is_set() or robot.mode not in (0, 1)
            or value(robot.get_state(), 'state before recovery') != 4
            or list(value(robot.get_err_warn_code(), 'recovery faults')) != [0, 0]
            or robot.get_is_moving()
            or value(robot.get_cmdnum(), 'command queue') != 0):
        raise RuntimeError('STOP recovery requires fault-free stationary position/Servo mode and an empty queue')
    commands = 2
    if robot.mode == 1:
        code = robot.set_mode(0)
        if code != 0:
            raise RuntimeError(f'restore position mode failed: {code}')
        deadline = time.monotonic() + 1.
        while robot.mode != 0:
            if cancel.wait(.01) or time.monotonic() >= deadline:
                raise RuntimeError('position mode restoration not confirmed')
        commands += 1
    for name, command in (
        ('motion_enable', lambda: robot.motion_enable(enable=True)),
        ('set_state(normal)', lambda: robot.set_state(0)),
    ):
        if (cancel.is_set() or robot.mode != 0
                or value(robot.get_state(), 'recovery state') not in (4, 5)
                or list(value(robot.get_err_warn_code(), 'recovery faults')) != [0, 0]
                or robot.get_is_moving()
                or value(robot.get_cmdnum(), 'recovery queue') != 0):
            raise RuntimeError('STOP recovery cancelled or controller changed')
        code = command()
        if code != 0:
            raise RuntimeError(f'{name} failed: {code}')
    deadline = time.monotonic() + 2.
    while value(robot.get_state(), 'recovered state') not in (0, 2):
        if cancel.wait(.02) or time.monotonic() >= deadline:
            raise RuntimeError('controller did not leave STOP')
    return commands


def read_arm(profile, arm, *, factory=None, recover_stopped=False):
    config = profile['arms'][arm]
    if factory is None:
        from xarm.wrapper import XArmAPI
        factory = XArmAPI
    robot = None
    report = dict(arm=arm, ip=config['robot_ip'], started_at=datetime.now(timezone.utc).isoformat(),
        motion_commands_sent=0, configuration_commands_sent=0, execution_allowed=False)
    ready = threading.Event()
    count = [0]
    def received(row):
        if 'joints' in row and 'cartesian' in row:
            count[0] += 1
            if count[0] >= 2:
                ready.set()
    try:
        from .robot_api import _connect_xarm
        def setup(connection):
            count[0] = 0
            ready.clear()
            connection.register_report_callback(received)
        robot = _connect_xarm(factory, config['robot_ip'], before_connect=setup)
        if not ready.wait(3):
            raise RuntimeError('Initial coherent controller reports unavailable')
        value(robot.get_robot_sn(), 'robot identity')
        q0 = np.asarray(value(robot.get_servo_angle(is_radian=False), 'initial joints'), dtype=float)
        pose = value(robot.get_position(is_radian=False), 'TCP pose')
        report.update(axis=int(robot.axis), control_box_sn=robot.control_box_sn,
            robot_sn=robot.sn, state=value(robot.get_state(), 'state'), mode=int(robot.mode),
            errors_warnings=value(robot.get_err_warn_code(), 'errors/warnings'),
            joints_deg=q0.tolist(), pose_base_mm_deg=pose,
            pose_common_mm_deg=convert_pose_mm_deg(profile, pose, arm, 'common').tolist(),
            tcp_offset_mm_deg=list(robot.tcp_offset), world_offset_mm_deg=list(robot.world_offset),
            tcp_load=list(robot.tcp_load), motor_enable_states=list(robot.motor_enable_states),
            motor_brake_states=list(robot.motor_brake_states),
            gripper_position_response=robot.get_gripper_position(),
            gripper_error_response=robot.get_gripper_err_code())
        time.sleep(.25)
        q1 = np.asarray(value(robot.get_servo_angle(is_radian=False), 'final joints'), dtype=float)
        report['final_state'] = value(robot.get_state(), 'final state')
        report['final_errors_warnings'] = value(robot.get_err_warn_code(), 'final errors/warnings')
        report['joint_drift_deg'] = float(np.max(np.abs(q1 - q0)))
        report['home_joint_max_error_deg'] = float(np.max(np.abs(q1 - config['home']['reference_joints_deg'])))
        # Only explicit top-level expectations are enforcement bindings. The
        # nested identity_verification record is provenance, so fixture and
        # historical read-only profiles remain observational.
        expected_box = config.get('expected_control_box_sn')
        expected_robot = config.get('expected_robot_sn')
        checks = dict(axis=report['axis'] == config['robot_axis'],
            control_box_identity=(not expected_box
                                  or report['control_box_sn'] == expected_box),
            robot_identity=(not expected_robot
                            or report['robot_sn'] == expected_robot),
            tcp_offset=bool(np.allclose(report['tcp_offset_mm_deg'], config['tcp_offset_mm_deg'], atol=.1, rtol=0)),
            world_offset=bool(np.allclose(report['world_offset_mm_deg'], config['world_offset_mm_deg'], atol=.01, rtol=0)),
            idle_position_mode=report['state'] == report['final_state'] == 2 and report['mode'] == 0,
            no_faults=report['errors_warnings'] == report['final_errors_warnings'] == [0, 0],
            stationary=report['joint_drift_deg'] <= .1,
            identity_observed=bool(report['control_box_sn']),
            motors_enabled=len(report['motor_enable_states']) >= 7 and all(report['motor_enable_states'][:7]),
            brakes_released=len(report['motor_brake_states']) >= 7 and all(report['motor_brake_states'][:7]))
        report.update(checks=checks, status='READ_ONLY_CHECKS_PASS' if all(checks.values()) else 'NEEDS_ATTENTION')
        report['identity_binding'] = ('Compared with the site profile identity evidence.'
            if (expected_box or expected_robot) else
            'Observed identity only; site profile has not yet bound an expected controller serial.')
        if recover_stopped and report['state'] == report['final_state'] == 4:
            identity = config['identity_verification']
            if (not all(v for k, v in checks.items() if k != 'idle_position_mode')
                    or report['control_box_sn'] != identity['control_box_sn']
                    or report['robot_sn'] != identity['robot_sn']):
                raise RuntimeError('STOP recovery requires valid identity, offsets, stationary feedback and no faults')
            report['configuration_commands_sent'] = recover_stopped_controller(robot, threading.Event())
            report['stop_recovered'] = True
    except Exception as exc:
        report.update(status='READ_FAILED', error=f'{type(exc).__name__}: {exc}')
    finally:
        if robot is not None:
            robot.disconnect()
    if report.get('stop_recovered'):
        fresh = read_arm(profile, arm, factory=factory)
        fresh.update(stop_recovered=True, configuration_commands_sent=report['configuration_commands_sent'],
                     mode_before_recovery=report['mode'],
                     state_before_recovery=4)
        return fresh
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    profile = load_site(args.profile)
    if args.output.exists():
        parser.error('Output already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(calibration_id=profile['calibration_id'], execution_allowed=False,
        motion_commands_sent=0, configuration_commands_sent=0, arms={})
    for arm in ('left', 'right'):
        try:
            report['arms'][arm] = read_arm(profile, arm)
        except Exception as exc:
            report['arms'][arm] = dict(status='CONNECTION_FAILED', error=f'{type(exc).__name__}: {exc}')
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if all(r['status'] == 'READ_ONLY_CHECKS_PASS' for r in report['arms'].values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
