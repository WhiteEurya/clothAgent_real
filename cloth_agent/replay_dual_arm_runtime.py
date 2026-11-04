"""Route the legacy logical Arm7 grasp to one of two live xArm7 devices."""
import json
import threading
from pathlib import Path

import numpy as np

from .codex_motion import parse_code
from .dual_arm.geometry import DualArmError, pose_matrix


def route_grasp(source, tcp_world_mm):
    """Bisect live TCP XY positions; preserve every world-frame target pose."""
    calls = parse_code(source, allow_grasp=True)
    if calls[0][0] != 'grasp' or any(set(args) != {'right'} for _, args in calls):
        raise DualArmError('partitioned replay requires one logical right grasp task')
    left, right = (np.asarray(tcp_world_mm[k], dtype=float) for k in ('left', 'right'))
    if any(p.shape != (3,) or not np.isfinite(p).all() for p in (left, right)):
        raise DualArmError('invalid live TCP positions')
    axis = right[:2] - left[:2]
    length = np.linalg.norm(axis)
    if length < 1.:
        raise DualArmError('TCP centers are indistinguishable in the table plane')
    axis /= length
    midpoint = (left + right) / 2.
    grasp_mm = np.asarray(calls[0][1]['right'][:3]) * 1000.
    distance = float((grasp_mm[:2] - midpoint[:2]) @ axis)
    arm = 'left' if distance < 0. else 'right'
    routed = '\n'.join(f'motion.{name}({arm}={args["right"]!r})' for name, args in calls) + '\n'
    return routed, dict(active_arm=arm, inactive_arm='right' if arm == 'left' else 'left',
                       tcp_world_mm={k: np.asarray(v).tolist() for k, v in tcp_world_mm.items()},
                       midpoint_world_mm=midpoint.tolist(), right_axis_xy=axis.tolist(),
                       grasp_world_mm=grasp_mm.tolist(), signed_distance_mm=distance,
                       center_tie_arm='right', rule='live_tcp_xy_perpendicular_bisector')


class PartitionedReplayRuntime:
    """Graph/observation adapter to ExperimentRunner, with no motion backend."""
    active_arms = ('left', 'right')
    grasp_offset_mm = 0.
    preview = None
    planning_policy = '输出逻辑 right 的单夹爪 grasp 和搬运；宿主按两夹爪中点选择物理臂。'
    motion_contract = '只允许一个 motion.grasp(right=[x,y,z,roll,pitch,yaw])，随后使用 right 的 move_tcp，保持 roll/pitch；单位 m/rad。'
    preparation_description = '固定 ZED 采图；读取两台控制器的实际静止姿态，不自动回位。'

    def __init__(self, config_path, *, planning_only=False):
        from .dual_arm.config import DualConfig
        from .dual_arm.kinematics import ArmModel
        from .site_observation import ROOT, load_site
        self.config_path = Path(config_path).resolve()
        self.config = DualConfig.load_geometry(config_path, root=ROOT)
        if any(a.axis != 7 for a in self.config.arms.values()):
            raise DualArmError('partitioned replay requires two xArm7 robots')
        if self.config.raw['camera'].get('backend') != 'zed' or self.config.raw['camera'].get('mount_arm') is not None:
            raise DualArmError('partitioned replay requires the calibrated fixed ZED')
        self.models = {k: ArmModel(a) for k, a in self.config.arms.items()}
        self.profile = load_site()
        self.task_policy = json.loads((ROOT / 'config/folding.partitioned.json').read_text())
        self.calibration_id = self.config.raw['calibration_id']
        self.planning_only = planning_only
        self.physical = not planning_only
        self._routing_lock = threading.Lock()
        self._lease = None

    def observe(self, directory):
        if self.planning_only:
            raise DualArmError('planning-only adapter cannot connect hardware')
        import fcntl
        from .site_robot_check import read_arm
        from .dual_arm.observation import capture
        if self._lease is None:
            self._lease = open('/tmp/clothagent_codex_dual_arm.lock', 'w')
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def snapshots():
            result = {}
            for key in self.models:
                row = read_arm(self.profile, key)
                if row['status'] != 'READ_ONLY_CHECKS_PASS':
                    raise DualArmError(f'{key}: controller not stationary/ready: {row["checks"]}')
                if row['control_box_sn'] != self.config.arms[key].serial:
                    raise DualArmError(f'{key}: controller identity mismatch')
                result[key] = dict(joints=row['joints_deg'])
            return result
        # Fixed camera, sequential observe/execute: query both arms before and
        # after capture, not by reconnecting controllers for every video frame.
        return capture(directory, self.config, self.models, snapshots, check_each_frame=False)

    def compile(self, source, observation):
        from .experiments.full_grasp_adapter import build_robot_config
        from .site_observation import convert_pose_mm_deg
        from .dual_arm.geometry import pose_error
        tcp = {k: (a.world_from_base @ pose_matrix(self.models[k].forward(observation.meta['joints_deg'][k])))[:3, 3]
               for k, a in self.config.arms.items()}
        routed, assignment = route_grasp(source, tcp)
        active = assignment['active_arm']
        calls = parse_code(routed, allow_grasp=True)
        if any(name not in {'grasp', 'move_tcp'} for name, _ in calls):
            raise DualArmError('RobotAPI does not reproduce Native move_transfer detours')
        poses = [np.r_[np.array(kw[active][:3])*1000, np.degrees(kw[active][3:])] for _, kw in calls]
        poses = [convert_pose_mm_deg(self.profile, p, 'common', active) for p in poses]
        state = {'arms': {k: {'pose_base_mm_deg': self.models[k].forward(a.home_joints).tolist()}
                          for k, a in self.config.arms.items()}}
        config = build_robot_config(self.config.raw, state, active, poses[0][3], poses[0][4], config_path=self.config_path)
        lines = ['def run():', '    open_gripper()']
        def move(pose):
            yaw = config.relative_yaw_from_absolute_deg(float(pose[5]))
            represented = [*pose[:3], config.orientation_roll_deg, config.orientation_pitch_deg, config.command_yaw_deg(yaw)]
            if pose_error(pose, represented)[1] > 1e-6:
                raise DualArmError('RobotAPI fixed roll/pitch cannot represent this graph action')
            args = [float(v) for v in pose[:3]] + [yaw]
            lines.append('    move(' + ', '.join(repr(v) for v in args) + ')')
        up = self.config.arms[active].world_from_base[:3, :3].T @ np.array([0., 0., 1.])
        hover = poses[0].copy(); hover[:3] += self.task_policy['approach_clearance_mm'] * up
        move(hover); move(poses[0]); lines.append('    close_gripper()')
        for pose in poses[1:]: move(pose)
        lines.append('    open_gripper()')
        retreat = poses[-1].copy(); retreat[:3] += self.task_policy['retreat_clearance_mm'] * up
        move(retreat); lines.append('    home()')
        return config, '\n'.join(lines) + '\n', assignment

    def execute(self, source, observation, folder):
        if self.planning_only:
            raise DualArmError('planning-only adapter cannot execute motion')
        if not self._routing_lock.acquire(blocking=False):
            raise DualArmError('a partitioned replay action is already running')
        try:
            from .experiment import ExperimentRunner
            from .robot_collision import RobotCollision
            observation.validate_for(self.config, live=True, check_age=False)
            config, program, assignment = self.compile(source, observation)
            folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
            (folder / 'arm_assignment.json').write_text(json.dumps(assignment, indent=2)+'\n')
            (folder / 'routed_motion.py').write_text(program)
            collision = RobotCollision(self.config_path, observation.meta['joints_deg'], assignment['active_arm'], root=self.config.root)
            runner = ExperimentRunner(folder / 'experiment', config, collision_checker=collision)
            path = runner.workspace / 'graph_action.py'; path.write_text(program)
            result = runner.run_experiment(path, real=True, confirmed=True)
            return dict(status='EXECUTED' if result['execution_completed'] else 'EXECUTION_FAILED',
                        physical_execution=result['physical_execution'], arm_assignment=assignment, experiment=result)
        except (ValueError, RuntimeError) as exc:
            return dict(status='PLANNING_FAILED', physical_execution=False, error=f'{type(exc).__name__}: {exc}')
        finally:
            self._routing_lock.release()

    def stop(self):
        # Motion lifecycle and exception-stop belong to RobotAPI. This adapter
        # has only short-lived read-only controller connections, never motors.
        pass

    def close(self):
        if self._lease is not None:
            self._lease.close(); self._lease = None
