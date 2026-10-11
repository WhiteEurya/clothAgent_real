"""Arm7-only folding, retaining both-arm geometry and the calibrated camera."""
import json
from pathlib import Path
import numpy as np

from .codex_native_runtime import NativeRuntime
from .codex_motion import parse_code
from .dual_arm.geometry import DualArmError,pose_error
from .codex_grasp_task import TASK_CONTRACT,lower_task_grasps


def wrist_equivalent(angle, seed, lower, upper, *, prefer_ccw=False):
    """Choose a winding of the same pose; arm -> tool view: negative is CCW."""
    turns=range(int(np.ceil((lower-angle)/360)),int(np.floor((upper-angle)/360))+1)
    candidates=[angle+360*n for n in turns]
    if not candidates:raise DualArmError('no joint7 equivalent within limits')
    nearest=min(candidates,key=lambda value:abs(value-seed))
    # Only large approach rotations get a direction preference. Small Cartesian
    # continuation steps must never acquire an extra revolution.
    if prefer_ccw and nearest-seed>90.:
        ccw=[value for value in candidates if value<seed]
        if ccw:return max(ccw)
    return nearest


class Arm7Runtime(NativeRuntime):
    grasp_offset_mm=15.
    active_arms=('right',)
    planning_policy=(
        '只使用 arm7（right）单臂整理并折叠衣服，按顺序参考图完成。'
        'arm6（left）保持当前位置，不参与抓取、固定或折叠。'
        '选择单夹爪可完成的抓取和折入动作。'
        '尽可能和human demonstration相似。'
        '抓点使用 world_xyz_mm.npy 中的毫米坐标，不重复修正。'
        '宿主会将闭合前的抓取 TCP 目标额外降低 15 mm，输出观测原始抓取高度，不要重复减去该偏移。\n')
    preparation_description=(
        '拍照前只打开 arm7 夹爪，让 arm7 回 Home，再到 arm7 保存的 perception 位姿。'
        '使用 arm7 上的 Camera B 拍照，arm6 全程保持原位。'
        '宿主负责最后松开与回 Home，GPT 只给抓点及持布期间的动作。\n')
    motion_contract=TASK_CONTRACT.replace('10 mm','15 mm').replace(
        'left=[x,y,z,roll,pitch,yaw], right=[x,y,z,roll,pitch,yaw]',
        'right=[x,y,z,roll,pitch,yaw]')+'\n仅允许 right，不允许 left 或 arm6 动作。'

    def _inverse_target(self,arm,target_base,seed,phase):
        q=super()._inverse_target(arm,target_base,seed,phase)
        if arm!='right':return q
        model=self.models[arm]
        # Synthetic test models do not represent a periodic revolute wrist.
        if not hasattr(model,'urdf'):return q
        joint=model.urdf.joint_map['joint7']
        if joint.type!='revolute' or not np.allclose(joint.axis,[0,0,1]):
            raise DualArmError('arm7 cable policy requires the calibrated +Z joint7 axis')
        selected=np.asarray(q).copy()
        selected[-1]=wrist_equivalent(q[-1],seed[-1],
            np.degrees(model.lower[-1]),np.degrees(model.upper[-1]),
            prefer_ccw=phase in ('grasp_goal_ik','host_approach_ik'))
        distance,angle=pose_error(model.forward(selected),target_base)
        if distance>.5 or angle>.2:
            raise DualArmError('arm7 wrist equivalent failed TCP verification')
        if abs(selected[-1]-q[-1])>1e-6:
            self.planning_context['wrist_winding']=dict(
                phase=phase,original_deg=float(q[-1]),selected_deg=float(selected[-1]),
                seed_deg=float(seed[-1]),view='arm_toward_gripper',ccw_sign=-1)
        return selected

    def adjust_motion(self,source,folder):
        task=lower_task_grasps(source,folder,offset_mm=self.grasp_offset_mm)
        if task is not None:return task
        from .codex_grasp_height import lower_grasp_z
        effective,changes=lower_grasp_z(source,arm='right',offset_mm=self.grasp_offset_mm)
        if changes:
            (Path(folder)/'effective_motion.py').write_text(effective)
            (Path(folder)/'grasp_height_policy.json').write_text(json.dumps(
                dict(mode='arm7_grasp_lowering',offset_mm=self.grasp_offset_mm,changes=changes),indent=2)+'\n')
        return effective

    def plan(self,source,initial,**kwargs):
        self.preview_candidates=[];self.preview_failures=[]
        self.planning_context={'stage':'arm7_only_command_check'}
        for name,args in parse_code(source,allow_grasp=True):
            if (name in ('grasp','move_tcp','move_transfer','move_joints') and set(args)!={'right'}) or (name=='gripper' and args['arm']!='right'):
                raise DualArmError('arm7-only pipeline rejects every arm6/left command')
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
            if arm!='right':raise DualArmError('arm7-only dispatcher rejects arm6 operation')
            # pack()/unpack() convert degrees <-> radians; a parked arm can
            # differ by a few floating-point ULPs without a commanded motion.
            if kind=='move':
                for segment in getattr(data,'segments',(data,)):
                    if segment.arm!='right' or not np.allclose(segment.start['left'],segment.end['left'],atol=1e-9,rtol=0):
                        raise DualArmError('arm7-only path must keep arm6 stationary')
        return super()._execute_operations(operations,folder)

    def load_config(self,config_path):
        from .codex_camera_b import with_arm7_camera
        config=with_arm7_camera(super().load_config(config_path))
        # Match arm6's runtime-local policy for the active arm only.
        config.collision_envelopes_disabled_arms=frozenset({'right'})
        parked_file=config.root/'data/robot/arm6_parked_for_arm7.json'
        parked=json.loads(parked_file.read_text())
        if parked.get('arm')!='left':raise DualArmError('arm7 static obstacle must be arm6/left')
        joints=np.asarray(parked['joint_angles_deg'],dtype=float)
        if joints.shape!=(config.arms['left'].axis,) or not np.isfinite(joints).all():
            raise DualArmError('invalid arm6 static obstacle joints')
        parked['reference_file']=str(parked_file)
        config.static_arm_obstacles={'left':parked}
        return config

    def _preparation_program(self,initial,owner,target):
        if owner!='right':raise DualArmError('arm7-only observation requires Camera B on arm7')
        final={k:v.copy() for k,v in initial.items()};final['right']=target.copy()
        home=self.config.arms['right'].home_joints
        source=('motion.gripper(arm="right",action="open")\n'
                f'motion.move_joints(right={np.radians(home).tolist()})\n'
                f'motion.move_joints(right={np.radians(target).tolist()})\n')
        return final,source
