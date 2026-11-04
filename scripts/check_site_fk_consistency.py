"""Compare stationary controller feedback and FK with the configured URDF."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time

import numpy as np
from scipy.spatial.transform import Rotation

from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.geometry import pose_matrix
from cloth_agent.dual_arm.kinematics import ArmModel
from cloth_agent.dual_arm.factory_kinematics import read_origins, calibrated_urdf
from cloth_agent.site_robot_check import value


def difference(predicted, observed):
    predicted, observed = pose_matrix(predicted), pose_matrix(observed)
    return dict(position_mm=float(np.linalg.norm(predicted[:3, 3] - observed[:3, 3])),
                rotation_deg=float(np.degrees(Rotation.from_matrix(
                    predicted[:3, :3].T @ observed[:3, :3]).magnitude())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path('config/dual_arm.zed_dual_xarm7_20261103.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--factory-calibration', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    config = DualConfig.load(args.config, root=Path(__file__).resolve().parents[1])
    report = dict(started_at=datetime.now(timezone.utc).isoformat(),
                  motion_commands_sent=0, configuration_commands_sent=0, arms={})
    from xarm.wrapper import XArmAPI
    for name, arm in config.arms.items():
        entry = report['arms'][name] = dict(status='STARTED', samples=[])
        robot = None
        try:
            address = config.raw['arms'][name]['ip']
            with socket.create_connection((address, 502), timeout=3):
                pass
            robot = XArmAPI(address, is_radian=False, do_not_open=True)
            robot.connect()
            time.sleep(.7)
            value(robot.get_robot_sn(), 'identity')
            entry.update(robot_sn=robot.sn, control_box_sn=robot.control_box_sn,
                         dh_params=robot.get_dh_params(), tcp_offset=list(robot.tcp_offset),
                         world_offset=list(robot.world_offset))
            expected = config.raw['arms'][name]
            if robot.sn != expected['robot_sn'] or robot.axis != arm.axis:
                raise RuntimeError('Robot identity mismatch')
            if not np.allclose(robot.tcp_offset, arm.tcp_offset, atol=.1, rtol=0):
                raise RuntimeError('TCP offset mismatch')
            if not np.allclose(robot.world_offset, 0, atol=.01, rtol=0):
                raise RuntimeError('Nonzero world offset')
            if value(robot.get_state(), 'state') != 2 or robot.mode != 0:
                raise RuntimeError('Robot is not stationary in position mode')
            if value(robot.get_err_warn_code(), 'faults') != [0, 0]:
                raise RuntimeError('Controller fault')
            model = ArmModel(arm)
            calibrated = None
            if args.factory_calibration:
                origins = read_origins(address, arm.axis)
                entry['factory_origins_m_rad'] = origins.tolist()
                candidate = args.output.parent / f'{args.output.stem}_{name}.urdf'
                candidate.parent.mkdir(parents=True, exist_ok=True)
                with candidate.open('x') as output:
                    output.write(calibrated_urdf(arm.urdf, origins))
                entry['candidate_urdf'] = str(candidate)
                calibrated = ArmModel(replace(arm, urdf=candidate.resolve()))
            initial = value(robot.get_servo_angle(is_radian=False), 'joints')
            for index in range(3):
                joints = value(robot.get_servo_angle(is_radian=False), 'joints')
                feedback = value(robot.get_position(is_radian=False), 'position')
                controller = value(robot.get_forward_kinematics(joints,
                    input_is_radian=False, return_is_radian=False), 'controller FK')
                predicted = model.forward(joints).tolist()
                entry['samples'].append(dict(joints_deg=joints, feedback=feedback,
                    controller_fk=controller, urdf_fk=predicted,
                    urdf_vs_feedback=difference(predicted, feedback),
                    urdf_vs_controller_fk=difference(predicted, controller),
                    controller_fk_vs_feedback=difference(controller, feedback)))
                if calibrated is not None:
                    entry['samples'][-1]['factory_vs_controller_fk'] = difference(
                        calibrated.forward(joints), controller)
                time.sleep(.1)
            if calibrated is not None:
                entry['virtual_fk_samples'] = []
                for joint_index in range(arm.axis):
                    for delta in (-5., 5.):
                        query_joints = np.asarray(initial).copy()
                        query_joints[joint_index] += delta
                        controller = value(robot.get_forward_kinematics(query_joints.tolist(),
                            input_is_radian=False, return_is_radian=False), 'virtual FK')
                        entry['virtual_fk_samples'].append(dict(joints_deg=query_joints.tolist(),
                            controller_fk=controller, nominal_vs_controller_fk=difference(
                                model.forward(query_joints), controller),
                            factory_vs_controller_fk=difference(calibrated.forward(query_joints), controller)))
            final = value(robot.get_servo_angle(is_radian=False), 'final joints')
            entry['joint_drift_deg'] = float(np.max(np.abs(np.asarray(final)-initial)))
            entry['final_faults'] = value(robot.get_err_warn_code(), 'final faults')
            entry['status'] = 'READ_ONLY_COMPLETE'
        except Exception as exc:
            entry.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        finally:
            if robot is not None:
                robot.disconnect()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if all(entry['status'] == 'READ_ONLY_COMPLETE'
                    for entry in report['arms'].values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
