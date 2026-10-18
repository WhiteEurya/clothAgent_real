"""Shared single-arm execution policies extracted from the Arm7 runtime."""
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
    if not candidates:raise DualArmError('no wrist equivalent within limits')
    nearest=min(candidates,key=lambda value:abs(value-seed))
    # A direction preference must never create a near-full wrist revolution.
    # The 180-degree jaw symmetry is selected for the WHOLE task before IK.
    if prefer_ccw:
        bounded=[value for value in candidates if abs(value-seed)<=180.+1e-6]
        if not bounded:raise DualArmError('initial wrist alignment exceeds 180 degrees within limits')
        ccw=[value for value in bounded if value-seed<=1e-6]
        return min(ccw or bounded,key=lambda value:abs(value-seed))
    return nearest


class SingleArmRuntime(NativeRuntime):
    arm_key='right'
    other_key='left'
    robot_name='arm7'
    other_robot_name='arm6'
    wrist_joint='joint7'
    camera_label='B'

    # Parallel-jaw symmetry applies to folding and graph replay as well as unfold.
    parallel_jaw_grasp_symmetry=True
    grasp_offset_mm=0.
    active_arms=('right',)
    staged_release=True
    release_open_delta_pulse=120.
    planning_policy=(
        '只使用 arm7（right）单臂整理并折叠衣服，按顺序参考图完成。'
        'arm6（left）保持当前位置，不参与抓取、固定或折叠。'
        '选择单夹爪可完成的抓取和折入动作。'
        '尽可能和human demonstration相似。'
        '平行夹爪绕工具轴相差180度等价；宿主会在等价姿态中优先选择逆时针对齐的一支，不绕整圈，'
        '并对后续持布位姿同步变换以保持相对旋转，不要主动添加半圈旋转。'
        'world_xyz_mm.npy 提供观测的 World 毫米坐标；根据当前观察和动作需要，自主选择抓取及后续动作的 TCP 高度。\n')
    preparation_description=(
        '拍照前只打开 arm7 夹爪，让 arm7 回 Home，再到 arm7 保存的 perception 位姿。'
        '使用 arm7 上的 Camera B 拍照，arm6 全程保持原位。'
        '最终放置点应先让布料受到接收区支撑；宿主先少量开爪放布，再竖直抬升后完全开爪并回 Home。'
        'GPT 只给抓点及持布期间的动作，不添加开爪或撤离指令。\n')
    motion_contract=TASK_CONTRACT.replace('宿主对 grasp 的原始观测 Z 额外降低 10 mm；不要自己重复减去偏移，中间动作目标不加此偏移。',
        '').replace(
        'left=[x,y,z,roll,pitch,yaw], right=[x,y,z,roll,pitch,yaw]',
        'right=[x,y,z,roll,pitch,yaw]')+'\n仅允许 right，不允许 left 或 arm6 动作。'

    def _inverse_target(self,arm,target_base,seed,phase):
        q=super()._inverse_target(arm,target_base,seed,phase)
        if arm!=self.arm_key:return q
        model=self.models[arm]
        # Synthetic test models do not represent a periodic revolute wrist.
        if not hasattr(model,'urdf'):return q
        joint=model.urdf.joint_map[self.wrist_joint]
        if joint.type!='revolute' or not np.allclose(joint.axis,[0,0,1]):
            raise DualArmError(f'{self.robot_name} cable policy requires the calibrated +Z {self.wrist_joint} axis')
        selected=np.asarray(q).copy()
        selected[-1]=wrist_equivalent(q[-1],seed[-1],
            np.degrees(model.lower[-1]),np.degrees(model.upper[-1]),
            prefer_ccw=phase in ('grasp_goal_ik','host_approach_ik'))
        distance,angle=pose_error(model.forward(selected),target_base)
        if distance>.5 or angle>.2:
            raise DualArmError(f'{self.robot_name} wrist equivalent failed TCP verification')
        if phase in ('grasp_goal_ik','host_approach_ik') or abs(selected[-1]-q[-1])>1e-6:
            self.planning_context['wrist_winding']=dict(
                phase=phase,original_deg=float(q[-1]),selected_deg=float(selected[-1]),
                seed_deg=float(seed[-1]),view='arm_toward_gripper',ccw_sign=-1,
                prefer_ccw=phase in ('grasp_goal_ik','host_approach_ik'),
                selected_delta_deg=float(selected[-1]-seed[-1]),
                clockwise_selected=bool(phase in ('grasp_goal_ik','host_approach_ik')
                    and selected[-1]-seed[-1]>1e-6))
        return selected

    def execute(self,source,observation,folder):
        calls=parse_code(source,allow_grasp=True)
        symmetry=(getattr(self,'parallel_jaw_grasp_symmetry',False)
                  or getattr(self,'unfold_grasp_symmetry',False))
        if symmetry and calls[0][0]=='grasp' and set(calls[0][1])=={self.arm_key}:
            from .dual_arm.geometry import pose_matrix
            from .grasp_orientation import nearest_grasp_orientation
            joints=observation.meta['joints_deg'][self.arm_key]
            world=self.config.arms[self.arm_key].world_from_base@pose_matrix(self.models[self.arm_key].forward(joints))
            source,report=nearest_grasp_orientation(source,world[:3,:3],prefer_ccw=True,arm=self.arm_key)
            if report is not None:
                (Path(folder)/'grasp_orientation_policy.json').write_text(json.dumps(report,indent=2)+'\n')
        return super().execute(source,observation,folder)

    def adjust_motion(self,source,folder):
        task=lower_task_grasps(source,folder,offset_mm=self.grasp_offset_mm)
        if task is not None:return task
        from .codex_grasp_height import lower_grasp_z
        effective,changes=lower_grasp_z(source,arm=self.arm_key,offset_mm=self.grasp_offset_mm)
        if changes:
            (Path(folder)/'effective_motion.py').write_text(effective)
            (Path(folder)/'grasp_height_policy.json').write_text(json.dumps(
                dict(mode=f'{self.robot_name}_grasp_lowering',offset_mm=self.grasp_offset_mm,changes=changes),indent=2)+'\n')
        return effective

    def plan(self,source,initial,**kwargs):
        self.preview_candidates=[];self.preview_failures=[]
        self.planning_context={'stage':f'{self.robot_name}_only_command_check'}
        for name,args in parse_code(source,allow_grasp=True):
            if (name in ('grasp','move_tcp','move_transfer','move_joints') and set(args)!={self.arm_key}) or (name=='gripper' and args['arm']!=self.arm_key):
                raise DualArmError(f'{self.robot_name}-only pipeline rejects every {self.other_robot_name}/{self.other_key} command')
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
            if arm!=self.arm_key:raise DualArmError(f'{self.robot_name}-only dispatcher rejects {self.other_robot_name} operation')
            # pack()/unpack() convert degrees <-> radians; a parked arm can
            # differ by a few floating-point ULPs without a commanded motion.
            if kind=='move':
                for segment in getattr(data,'segments',(data,)):
                    if segment.arm!=self.arm_key or not np.allclose(segment.start[self.other_key],segment.end[self.other_key],atol=1e-9,rtol=0):
                        raise DualArmError(f'{self.robot_name}-only path must keep {self.other_robot_name} stationary')
        return super()._execute_operations(operations,folder)

    def _preparation_program(self,initial,owner,target):
        if owner!=self.arm_key:raise DualArmError(f'{self.robot_name}-only observation requires Camera {self.camera_label} on {self.robot_name}')
        final={k:v.copy() for k,v in initial.items()};final[self.arm_key]=target.copy()
        home=self.config.arms[self.arm_key].home_joints
        source=(f'motion.gripper(arm="{self.arm_key}",action="open")\n'
                f'motion.move_joints({self.arm_key}={np.radians(home).tolist()})\n'
                f'motion.move_joints({self.arm_key}={np.radians(target).tolist()})\n')
        return final,source
