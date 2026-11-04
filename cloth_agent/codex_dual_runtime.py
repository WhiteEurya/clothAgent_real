"""Dual-arm folding using the configured calibrated RGB-D observation."""
import json
from pathlib import Path

import numpy as np

from .codex_native_runtime import NativeRuntime
from .codex_motion import CONTRACT
from .dual_arm.geometry import DualArmError
from .codex_grasp_task import TASK_CONTRACT,lower_task_grasps
from .codex_motion import parse_code


class DualArmRuntime(NativeRuntime):
    motion_contract=TASK_CONTRACT+(
        '\n以上宿主辅助动作及 grasp 格式限制仅适用于 grasp 模式。'
        'grasp 可仅含 left、仅含 right 或同时含两臂，各自目标可不同。'
        '也可选择以下显式模式，自行安排单臂/双臂移动、先后抓取、开合爪和释放；'
        '显式模式不写 grasp，不与 grasp 模式混用，结束前自行放稳并松开衣物。\n'+CONTRACT)

    def __init__(self,config_path,*,planning_only=False):
        super().__init__(config_path,planning_only=planning_only)
        for model in self.models.values():
            model.ik_max_iterations=40
            model.ik_seed_limit=2

    def plan(self,source,initial,**kwargs):
        self._ik_exact_cache={}
        return super().plan(source,initial,**kwargs)

    def _inverse_target(self,arm,target_base,seed,phase):
        import numpy as np
        key=(arm,tuple(np.asarray(target_base,dtype=float)),tuple(np.asarray(seed,dtype=float)))
        cache=getattr(self,'_ik_exact_cache',{})
        if key in cache:
            self.plan_stats['ik_cache_hits']=self.plan_stats.get('ik_cache_hits',0)+1
            return cache[key].copy()
        q=super()._inverse_target(arm,target_base,seed,phase)
        if len(cache)<1024:cache[key]=q.copy()
        self._ik_exact_cache=cache
        return q

    preparation_description=(
        '采集使用当前配置指定的 RGB-D 相机；固定 third-view ZED 不移动相机臂，'
        '腕部相机才执行对应的 perception 位姿准备。两臂已在准备终点时直接采集；否则打开两爪、回 Home。'
        'grasp 模式由宿主负责最后松开与回 Home；显式模式自行安排释放，下一轮宿主准备拍照。\n')

    @property
    def planning_policy(self):
        from .codex_loop import COOPERATION_POLICY
        return COOPERATION_POLICY+(
            '两臂所有抓点均来自本轮当前 RGB-D 相机的 world_xyz_mm.npy；不读取未启用的相机。'
            '自行选择一臂或两臂、不同抓点和动作顺序，不要求两爪同时抓取或抓同一位置。grasp 模式由宿主规划辅助动作，显式模式按输出代码执行。'
            '宿主将 grasp 或显式 close 前的抓取 TCP 目标额外降低 10 mm。'
            '请输出观测原始抓取高度，不要重复减去该偏移。\n')

    def load_config(self,config_path):
        from .codex_camera_b import with_arm7_camera
        config=with_arm7_camera(super().load_config(config_path))
        if (type(self) is DualArmRuntime and
                config.raw.get('execution_policy', {}).get('mode') ==
                'sequential_single_gripper_static_other'):
            raise DualArmError(
                'current ZED site requires --active-arm left|right; '
                'the inactive xArm7 must be loaded as a static collision model'
            )
        config.raw['astra_grasp_height']={'mode':'independent_grasp_lowering'}
        return config

    def adjust_motion(self,source,folder):
        task=lower_task_grasps(source,folder)
        if task is not None:return task
        from .codex_grasp_height import lower_grasp_z
        changes=[]
        for arm in ('left','right'):
            source,rows=lower_grasp_z(source,arm=arm,offset_mm=10.)
            changes.extend(rows)
        if changes:
            (Path(folder)/'effective_motion.py').write_text(source)
            (Path(folder)/'grasp_height_policy.json').write_text(json.dumps(
                dict(mode='dual_grasp_lowering',offset_mm=10.,changes=changes),indent=2)+'\n')
        return source


class ZedSingleArmRuntime(DualArmRuntime):
    """Current fixed-ZED mode: one real xArm7, the other arm is static geometry.

    The inactive arm is deliberately represented by :class:`StaticArmObstacle`.
    It is included in every collision proof, but it is never connected, enabled,
    sampled as live feedback, or sent a command.  A new round can select the
    other arm explicitly after that arm has been physically placed at its saved
    static reference pose.
    """

    def __init__(self, config_path, active_arm, *, planning_only=False):
        if active_arm not in ('left', 'right'):
            raise DualArmError('active_arm must be left or right')
        self.active_arm = active_arm
        self.inactive_arm = 'right' if active_arm == 'left' else 'left'
        self.active_arms = (active_arm,)
        super().__init__(config_path, planning_only=planning_only)

    def load_config(self, config_path):
        config = super().load_config(config_path)
        records = config.raw.get('static_arm_models', {})
        record = records.get(self.inactive_arm)
        if not isinstance(record, dict):
            raise DualArmError(
                f'missing static_arm_models.{self.inactive_arm} for single-arm execution'
            )
        joints = np.asarray(record.get('joint_angles_deg'), dtype=float)
        axis = config.arms[self.inactive_arm].axis
        if joints.shape != (axis,) or not np.isfinite(joints).all():
            raise DualArmError(
                f'invalid static_arm_models.{self.inactive_arm}.joint_angles_deg'
            )
        config.static_arm_obstacles = {
            self.inactive_arm: {
                **record,
                'joint_angles_deg': joints.tolist(),
                'source': record.get('source', 'config.static_arm_models'),
            }
        }
        # The static arm has no live stopping/tracking uncertainty.  Its
        # nominal URDF geometry remains in every collision proof.
        config.collision_envelopes_disabled_arms = frozenset({self.inactive_arm})
        config.raw['execution_policy'] = {
            'mode': 'sequential_single_gripper_static_other',
            'active_arm': self.active_arm,
            'inactive_arm': self.inactive_arm,
            'inactive_arm_controller_connection': False,
            'inactive_arm_collision_model': 'saved_joint_pose_urdf_capsules',
        }
        config.raw['astra_grasp_height'] = {'mode': 'source_surface', 'offset_mm': 0.}
        return config

    @property
    def planning_policy(self):
        side = '左' if self.active_arm == 'left' else '右'
        other = '右' if self.active_arm == 'left' else '左'
        return (
            f'本轮只使用物理{side}臂（{self.active_arm}）和一个夹爪完成动作；'
            f'物理{other}臂（{self.inactive_arm}）保持保存的静态姿态，只作为碰撞模型，'
            '不参与抓取、固定、同步或任何控制器命令。'
            '选择单夹爪可以完成的抓取和折叠动作，所有动作按顺序执行。'
            '抓点使用本轮 world_xyz_mm.npy 的毫米坐标，不重复修正。\n'
        )

    preparation_description = (
        '固定 ZED third-view 不移动相机。每轮只准备当前 active_arm；另一条 xArm7 '
        '不连接、不使能、不回 Home，由保存的关节姿态作为静态碰撞模型。'
        '动作必须在本轮结束前放稳衣物并松开当前夹爪。\n'
    )

    @property
    def motion_contract(self):
        arm = self.active_arm
        return TASK_CONTRACT.replace(
            'left=[x,y,z,roll,pitch,yaw], right=[x,y,z,roll,pitch,yaw]',
            f'{arm}=[x,y,z,roll,pitch,yaw]',
        ).replace(
            '宿主对 grasp 的原始观测 Z 额外降低 10 mm；不要自己重复减去偏移，中间动作目标不加此偏移。',
            '当前 ZED 现场保留原始观测抓取高度，不额外降低抓点；中间动作目标同样保持原值。',
        ) + (
            f'\n只允许 {arm}；不要输出 {self.inactive_arm} 的 grasp、move_tcp、'
            f'move_transfer、move_joints 或 gripper。'
        )

    def plan(self, source, initial, **kwargs):
        calls = parse_code(source, allow_grasp=True)
        for name, args in calls:
            if name in ('grasp', 'move_tcp', 'move_transfer', 'move_joints'):
                if set(args) - {self.active_arm}:
                    raise DualArmError(
                        f'single-arm runtime rejects inactive arm {self.inactive_arm} command'
                    )
            elif name == 'gripper' and args.get('arm') != self.active_arm:
                raise DualArmError(
                    f'single-arm runtime rejects inactive arm {self.inactive_arm} gripper command'
                )
        return super().plan(source, initial, **kwargs)

    def adjust_motion(self, source, folder):
        (Path(folder) / 'grasp_height_policy.json').write_text(json.dumps(
            dict(mode='source_surface', active_arm=self.active_arm, offset_mm=0., changes=[]),
            indent=2) + '\n')
        return source

    def _preparation_program(self, initial, owner, target):
        if owner is not None:
            raise DualArmError('fixed ZED execution does not move a camera arm')
        home = self.config.arms[self.active_arm].home_joints
        source = (
            f'motion.gripper(arm="{self.active_arm}",action="open")\n'
            f'motion.move_joints({self.active_arm}={np.radians(home).tolist()})\n'
        )
        final = {k: np.asarray(v).copy() for k, v in initial.items()}
        final[self.active_arm] = home.copy()
        return final, source

    def _execute_operations(self, operations, folder):
        for kind, data in operations:
            arm = data.arm if kind == 'move' else data['arm']
            if arm != self.active_arm:
                raise DualArmError(
                    f'single-arm dispatcher rejects inactive arm {self.inactive_arm} operation'
                )
            if kind == 'move':
                for segment in getattr(data, 'segments', (data,)):
                    if segment.arm != self.active_arm:
                        raise DualArmError('inactive static arm appears in a commanded segment')
                    if not np.allclose(
                        segment.start[self.inactive_arm],
                        segment.end[self.inactive_arm],
                        atol=1e-9,
                        rtol=0,
                    ):
                        raise DualArmError('static arm changed in a single-arm path')
        return super()._execute_operations(operations, folder)
