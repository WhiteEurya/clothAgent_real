"""Arm6-only folding, retaining both-arm geometry and the calibrated camera."""
import json
from pathlib import Path
import numpy as np

from .codex_native_runtime import NativeRuntime
from .codex_motion import parse_code
from .dual_arm.geometry import DualArmError
from .codex_grasp_task import TASK_CONTRACT,lower_task_grasps


class Arm6Runtime(NativeRuntime):
    grasp_offset_mm=15.
    active_arms=('left',)
    planning_policy=(
        '只使用 arm6（left）单臂整理并折叠衣服，按顺序参考图完成。'
        'arm7（right）保持当前位置，不参与抓取、固定或折叠。'
        '选择单夹爪可完成的抓取和折入动作。'
        '尽可能和human demonstration相似。'
        '抓点使用 world_xyz_mm.npy 中的毫米坐标，不重复修正。'
        '输出观测原始抓取高度；宿主将抓取 TCP 额外降低 15 mm，不要自行重复扣减。\n')
    preparation_description=(
        '拍照前只打开 arm6 夹爪，让 arm6 回 Home，再到 arm6 保存的 perception 位姿。'
        '使用 arm6 上的 Camera A 拍照，arm7 全程保持原位。'
        '宿主负责最后松开与回 Home，GPT 只给抓点及持布期间的动作。\n')
    motion_contract=TASK_CONTRACT.replace('10 mm','15 mm').replace(
        'left=[x,y,z,roll,pitch,yaw], right=[x,y,z,roll,pitch,yaw]',
        'left=[x,y,z,roll,pitch,yaw]')+'\n仅允许 left，不允许 right 或 arm7 动作。'

    def adjust_motion(self,source,folder):
        task=lower_task_grasps(source,folder,offset_mm=self.grasp_offset_mm)
        if task is not None:return task
        from .codex_grasp_height import lower_grasp_z
        effective,changes=lower_grasp_z(source,arm='left',offset_mm=self.grasp_offset_mm)
        if changes:
            (Path(folder)/'effective_motion.py').write_text(effective)
            (Path(folder)/'grasp_height_policy.json').write_text(json.dumps(
                dict(mode='arm6_grasp_lowering',offset_mm=self.grasp_offset_mm,changes=changes),indent=2)+'\n')
        return effective

    def plan(self,source,initial,**kwargs):
        self.preview_candidates=[];self.preview_failures=[]
        self.planning_context={'stage':'arm6_only_command_check'}
        for name,args in parse_code(source,allow_grasp=True):
            if (name in ('grasp','move_tcp','move_transfer','move_joints') and set(args)!={'left'}) or (name=='gripper' and args['arm']!='left'):
                raise DualArmError('arm6-only pipeline rejects every arm7/right command')
        operations=super().plan(source,initial,**kwargs)
        if not kwargs.get('preserve_command_boundaries',False):
            from .queued_motion import queue_moves
            operations=queue_moves(operations)
            self.plan_stats['queued_move_batches']=sum(kind=='move' for kind,_ in operations)
        return operations

    def _execute_operations(self,operations,folder):
        # Enforce again at dispatch, independently of the model prompt/parser.
        for kind,data in operations:
            arm=data.arm if kind=='move' else data['arm']
            if arm!='left':raise DualArmError('arm6-only dispatcher rejects arm7 operation')
            # pack()/unpack() convert degrees <-> radians; a parked arm can
            # differ by a few floating-point ULPs without a commanded motion.
            if kind=='move':
                for segment in getattr(data,'segments',(data,)):
                    if segment.arm!='left' or not np.allclose(segment.start['right'],segment.end['right'],atol=1e-9,rtol=0):
                        raise DualArmError('arm6-only path must keep arm7 stationary')
        return super()._execute_operations(operations,folder)

    def load_config(self,config_path):
        config=super().load_config(config_path)
        camera=config.raw['camera']
        if camera['mount_arm']!='left' or str(camera['serial'])!='317222073552':
            raise DualArmError('arm6-only observation requires Camera A on arm6 (317222073552)')
        # Runtime-local policy: retain nominal geometry and swept command ranges,
        # but omit arm6's additional collision uncertainty/stop envelopes.
        config.collision_envelopes_disabled_arms=frozenset({'left'})
        camera.update(depth_exposure_us=8500., depth_gain=16.,
                      depth_stability={'warmup_frames':30, 'max_frames':90,
                                       'window_frames':5, 'stable_windows':3,
                                       'max_range_mm':3.})
        return config

    def _preparation_program(self,initial,owner,target):
        if owner!='left':raise DualArmError('arm6-only observation requires Camera A on arm6')
        final={k:v.copy() for k,v in initial.items()};final['left']=target.copy()
        home=self.config.arms['left'].home_joints
        source=('motion.gripper(arm="left",action="open")\n'
                f'motion.move_joints(left={np.radians(home).tolist()})\n'
                f'motion.move_joints(left={np.radians(target).tolist()})\n')
        return final,source
