"""Route the legacy logical Arm7 grasp to one of two live xArm7 devices."""
import json
import threading
from pathlib import Path

import numpy as np

from .codex_motion import parse_code
from .dual_arm.geometry import DualArmError, pose_matrix
from .single_arm_runtime import SingleArmRuntime


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


class PartitionedReplayRuntime(SingleArmRuntime):
    """Both arms supply live feedback; only the selected arm moves per action.

    Graphs still describe a logical ``right`` gripper. Routing happens once,
    before IK, symmetry selection and dispatch; stored graph evidence is intact.
    """
    active_arms = ('left', 'right')
    camera_label = 'A'
    motion_contract = SingleArmRuntime.motion_contract.replace(' 或 arm6', '') + (
        '\n本入口 right 为 Graph 的逻辑夹爪槽位；宿主按当前两个夹爪中点选择物理执行臂。'
        '只输出一个 grasp，后续仅使用 right 的 move_tcp/move_transfer；不要输出关节角或双臂动作。'
    )
    planning_policy = (
        '沿用 Graph 的单夹爪动作格式，使用逻辑 right 输出一个 grasp 及后续搬运。'
        'right 是模板槽位，不表示固定物理右臂。宿主以当前两个夹爪 TCP 的中点为界，'
        '抓点在左侧则整步由左 xArm7 执行，右侧则由右 xArm7 执行。'
        '动作使用当前观测公共坐标系；每步只动一臂，不输出双臂或关节角指令。'
    )
    preparation_description = (
        '固定 ZED 采图，不移动相机臂；两臂读取实时关节姿态。'
        '按抓点选臂后，沿用原单夹爪抓取、搬运、分段释放和回 Home 流程。'
    )

    def __init__(self, config_path, *, planning_only=False):
        self._routing_lock = threading.Lock()
        super().__init__(config_path, planning_only=planning_only)

    def load_config(self, config_path):
        config = super().load_config(config_path)
        if any(arm.axis != 7 for arm in config.arms.values()):
            raise DualArmError('partitioned replay requires two xArm7 robots')
        camera = config.raw['camera']
        if camera.get('backend') != 'zed' or camera.get('mount_arm') is not None:
            raise DualArmError('partitioned replay requires the fixed calibrated ZED camera')
        # Unlike the old single-arm runtime, both devices provide live feedback.
        config.static_arm_obstacles = {}
        config.raw['execution_policy'] = dict(mode='live_tcp_partitioned_replay',
                                             simultaneous_motion=False)
        return config

    def execute(self, source, observation, folder):
        if getattr(self, 'planning_only', False):
            raise DualArmError('planning-only runtime cannot execute motion')
        # Reject overlapping callers instead of queuing a stale observation.
        if not self._routing_lock.acquire(blocking=False):
            raise DualArmError('a partitioned replay action is already running')
        try:
            tcp = {}
            for key, arm in self.config.arms.items():
                pose = self.models[key].forward(observation.meta['joints_deg'][key])
                tcp[key] = (arm.world_from_base @ pose_matrix(pose))[:3, 3]
            routed, routing = route_grasp(source, tcp)
            folder = Path(folder)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'arm_assignment.json').write_text(json.dumps(routing, indent=2) + '\n')
            (folder / 'routed_motion.py').write_text(routed)
            self.arm_key, self.other_key = routing['active_arm'], routing['inactive_arm']
            self.robot_name, self.other_robot_name = f'{self.arm_key} xArm7', f'{self.other_key} xArm7'
            result = super().execute(routed, observation, folder)
            result['arm_assignment'] = routing
            return result
        finally:
            self._routing_lock.release()
