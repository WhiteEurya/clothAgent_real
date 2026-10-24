"""Sequential native execution of Astra motion code using installed envelopes."""
from __future__ import annotations
import fcntl
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .codex_motion import parse_code
from .codex_astra_pipeline import ROOT
from .dual_arm.geometry import DualArmError, matrix_pose, pose_matrix, pose_error
from .dual_arm.motion.home import HomeValidator, pack, unpack
from .dual_arm.motion.path import PathPlanner
from .dual_arm.planning import validate_sample
from .dual_arm.stop_sweep import ProofIncompleteError, complete_proof
from .dual_arm.sequential_home import (PositionSegment, certify_segment, command_limits,
                                      check_feedback, wait_operation)


class NativeRuntime:
    physical=True

    def __init__(self, config_path):
        from .dual_arm.config import DualConfig
        from .dual_arm.kinematics import ArmModel
        self.config=self.load_config(config_path)
        self.config.execution_mode='controller_sequential'
        self.config.home_recover_stopped=True
        self.config.require_real(homing=True)
        self.calibration_id=self.config.raw['calibration_id']
        self.models={k:ArmModel(a) for k,a in self.config.arms.items()}
        self.cancel=threading.Event();self.coordinator=None;self._lease=None
        self.prepared=False
        self.paths=PathPlanner(SimpleNamespace(models=self.models),self,
                               HomeValidator(self.config,self.models,None))
        self.paths.edge_timeout_s=.5
        self.paths.complete_final_proof=True
        self.paths.validator.cancel=self.cancel
        self.paths.validator.progress=self._proof_progress

    def load_config(self,config_path):
        from .dual_arm.config import DualConfig
        return DualConfig.load(config_path,root=ROOT,homing_only=True)

    def capability_report(self):
        from .robot_capabilities import build_capabilities
        return build_capabilities(self.config,self.models,getattr(self,'active_arms',('left','right')))

    def _proof_progress(self, report):
        print('路径验证进行中：'+json.dumps(dict(context=getattr(self,'planning_context',{}),**report),ensure_ascii=False),flush=True)

    def perception_target(self):
        """Load the recorded pose by camera/robot identity, trimming SDK padding."""
        camera=self.config.raw['camera']
        owner=camera['mount_arm']
        if owner not in self.config.arms:
            raise DualArmError('perception transition requires a configured camera arm')
        arm=self.config.arms[owner]
        path=Path(camera.get('perception_pose_file','data/robot/xarm_perception_pose.json'))
        if not path.is_absolute():path=self.config.root/path
        recorded=json.loads(path.read_text())
        if (recorded.get('robot_ip')!=arm.ip or recorded.get('axis')!=arm.axis
                or recorded.get('control_box_sn')!=arm.serial
                or str(recorded.get('camera_serial'))!=str(camera['serial'])):
            raise DualArmError('perception pose does not match camera/robot identity')
        joints=np.asarray(recorded['joint_angles_deg'],dtype=float)
        offset=np.asarray(recorded['tcp_offset_mm_deg'],dtype=float)
        if joints.shape not in ((arm.axis,),(7,)) or not np.isfinite(joints).all():
            raise DualArmError('invalid perception joint vector')
        if offset.shape!=(6,) or not np.isfinite(offset).all() or np.max(np.abs(offset-arm.tcp_offset))>.5:
            raise DualArmError('perception pose TCP calibration differs from current config')
        return owner,joints[:arm.axis].copy()

    def move_to_perception(self,directory):
        owner,target=self.perception_target()
        snapshot=self.coordinator.snapshots()
        initial={k:np.asarray(v['joints']) for k,v in snapshot.items()}
        final,source=self._preparation_program(initial,owner,target)
        folder=Path(directory);folder.mkdir(parents=True,exist_ok=False)
        if (not getattr(self, 'open_at_home_on_return', False)
                and all(np.max(np.abs(initial[k]-final[k]))<=self.config.limits['stationary_tolerance_deg'] for k in final)):
            self.arrival(final)
            result=dict(status='ALREADY_AT_PERCEPTION',physical_execution=False)
        else:
            # Match the original acquisition sequence, but certify both arms
            # together instead of invoking the old single-arm SDK helper.
            (folder/'motion.py').write_text(source)
            from .trajectory_preview import record_preview
            try:
                operations=self._plan_preparation(source,initial)
            except Exception as exc:
                try:record_preview(self,source,initial,None,f'{type(exc).__name__}: {exc}',folder)
                except Exception as preview_error:print(f'预览记录失败：{preview_error}',flush=True)
                raise
            record_preview(self,source,initial,operations,None,folder)
            preview=getattr(self,'preview',None)
            if preview is not None:preview.await_execution(self.cancel)
            self.arrival(initial)
            result=self._execute_preparation(operations,folder)
            if result['status']=='EXECUTED':self.arrival(final)
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if result['status'] not in ('EXECUTED','ALREADY_AT_PERCEPTION'):
            raise DualArmError(f'perception transition failed: {result}')
        self.perception_initialized=True

    def _preparation_program(self,initial,owner,target):
        final={k:a.home_joints.copy() for k,a in self.config.arms.items()}
        final[owner]=target
        home=', '.join(f'{k}={np.radians(a.home_joints).tolist()}' for k,a in self.config.arms.items())
        source=('motion.gripper(arm="left", action="open")\n'
                'motion.gripper(arm="right", action="open")\n'
                f'motion.move_joints({home})\nmotion.move_joints({owner}={np.radians(target).tolist()})\n')
        return final,source

    def _plan_preparation(self,source,initial):
        return self.plan(source,initial,preserve_command_boundaries=True)

    def _execute_preparation(self,operations,folder):
        return self._execute_operations(operations,folder)

    def check(self,left,right):
        try:
            for k,q in [('left',left),('right',right)]:
                if np.any(q<self.models[k].lower) or np.any(q>self.models[k].upper):
                    raise DualArmError('joint target exceeds URDF limits')
            validate_sample(self.config,self.models,unpack(np.r_[left,right]),holding=False)
            return {'safe':True}
        except DualArmError as exc:return {'safe':False,'reason':str(exc)}

    def observe(self,directory):
        from .dual_arm.cli import connect
        from .dual_arm.execution import DualArmCoordinator
        from .dual_arm.observation import capture, create_camera
        directory=Path(directory)
        if self.coordinator is None:
            self._lease=open('/tmp/clothagent_codex_dual_arm.lock','w')
            fcntl.flock(self._lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            connections=connect(self.config,self.models,self.cancel,True)
            self.coordinator=DualArmCoordinator(self.config,self.models,connections,self.cancel,
                                                directory.parent/'controller',realtime=True)
        self.move_to_perception(directory.with_name(directory.name+'_perception'))
        return self.capture_current(directory)

    def capture_current(self,directory):
        """Capture at the actual current pose without moving back to perception."""
        from .dual_arm.observation import capture, create_camera
        directory=Path(directory)
        if self.coordinator is None:
            raise DualArmError('capture_current requires connected stationary arms')
        if getattr(self,'camera',None) is None:
            self.camera=create_camera(self.config)
        obs=capture(directory,self.config,self.models,self.coordinator.snapshots,camera=self.camera)
        q={k:np.asarray(v) for k,v in obs.meta['joints_deg'].items()}
        poses={k:matrix_pose(a.world_from_base@pose_matrix(self.models[k].forward(q[k]))).tolist()
               for k,a in self.config.arms.items()}
        (directory/'state.json').write_text(json.dumps(dict(units='mm_deg',joints=q_to_lists(q),world_tcp=poses),indent=2))
        return obs

    def _segments(self,a,b):
        if not getattr(self.config, 'path_validation_enabled', True):
            moving=[k for k in a if not np.allclose(a[k],b[k],atol=1e-9,rtol=0)]
            if len(moving)>1:raise DualArmError('native code moves one arm at a time')
            for state in (a,b):
                for k,q in state.items():
                    q=np.asarray(q,dtype=float)
                    if (q.shape!=(self.config.arms[k].axis,) or not np.isfinite(q).all()
                            or np.any(np.radians(q)<self.models[k].lower)
                            or np.any(np.radians(q)>self.models[k].upper)):
                        raise DualArmError('joint target exceeds URDF limits')
            if not moving:return []
            k=moving[0]
            speed,accel,timeout=command_limits(self.config,self.models[k],self.config.arms[k],a[k],b[k])
            return [PositionSegment(k,a,b,speed,accel,timeout)]
        if hasattr(self,'preview_candidates'):
            self.preview_candidates.append(dict(start=q_to_lists(a),end=q_to_lists(b),context=dict(self.planning_context)))
        pending=[(a,b,0)];result=[]
        while pending:
            start,end,depth=pending.pop()
            if self.cancel.is_set():raise KeyboardInterrupt('planning cancelled')
            moving=[k for k in start if np.max(np.abs(end[k]-start[k]))>1e-8]
            if not moving:continue
            if len(moving)!=1:raise DualArmError('native code moves one arm at a time')
            try:
                # Includes continuous geometry, workspace and stopping bounds.
                self.paths.validator.certify_box(pack(start),pack(end))
            except ProofIncompleteError as exc:
                self._preview_failure(exc)
                # Splitting the commanded motion does not shrink the fixed
                # stop/idle uncertainty; do not repeat an unfinished proof.
                raise
            except DualArmError as exc:
                if depth==0:self._preview_failure(exc)
                if depth>=12 or max(np.max(np.abs(end[k]-start[k])) for k in start)<.05:raise
                mid={k:(start[k]+end[k])/2 for k in start}
                pending.extend([(mid,end,depth+1),(start,mid,depth+1)]);continue
            k=moving[0]
            speed,accel,timeout=command_limits(self.config,self.models[k],self.config.arms[k],start[k],end[k])
            result.append(PositionSegment(k,start,end,speed,accel,timeout))
            if len(result)>256:raise DualArmError('native code exceeds segment budget')
        return result

    def plan(self,source,initial,*,preserve_command_boundaries=False):
        self.preview_candidates=[];self.preview_failures=[]
        self.planning_context={'stage':'parse_motion'}
        self.plan_stats={'checked_segments':0,'dispatched_segments':0}
        calls=parse_code(source,allow_grasp=True)
        if calls[0][0]=='blocked':
            self.planning_context={'stage':'model_blocked'}
            raise DualArmError(calls[0][1]['reason'])
        if calls[0][0]=='grasp':
            from .codex_grasp_task import plan_grasp_task
            return plan_grasp_task(self,calls,initial)
        q={k:np.asarray(v).copy() for k,v in initial.items()};operations=[]
        boundaries=set()
        for command_index,(name,kw) in enumerate(calls,1):
            self.planning_context={'stage':'plan_command','command_index':command_index,
                                   'command':name,'arguments':kw}
            first_operation=len(operations)
            if name=='gripper':
                # Attachment envelopes already cover the full gripper range.
                if getattr(self.config, 'path_validation_enabled', True):
                    with complete_proof(cancel=self.cancel,progress=self._proof_progress):
                        certify_segment(q,q,self.config,self.models)
                operations.append(('gripper',dict(**kw,joints=q_to_lists(q))))
                continue
            if name=='move_joints':
                goal={k:np.degrees(kw[k]) if k in kw else v.copy() for k,v in q.items()}
                if getattr(self, 'direct_joint_moves', False):
                    # Validate the direct joint interpolation; no path search
                    # or detour. A failed proof propagates to the caller.
                    operations.extend(('move', segment) for segment in self._segments(q, goal))
                else:
                    moving=[k for k in q if not np.allclose(q[k],goal[k],atol=1e-9,rtol=0)]
                    orders=(moving[0]+'_first',) if len(moving)==1 else ('left_first','right_first')
                    search_failures=[]
                    for attempt in range(1,3):
                        if self.cancel.is_set():raise KeyboardInterrupt('planning cancelled')
                        failures=[];path=None;incomplete=False
                        for order in orders:
                            try:
                                path=self.paths.joint_path(pack(q),pack(goal),order=order,timeout_s=30);break
                            except DualArmError as exc:
                                failures.append(str(exc))
                                incomplete |= isinstance(exc,ProofIncompleteError) or 'time budget' in str(exc) or 'planning budget' in str(exc)
                        if path is not None:break
                        if not incomplete:raise DualArmError('; '.join(failures))
                        search_failures.extend(failures)
                        if attempt==2:
                            self.planning_context['path_search_failure']=dict(
                                kind='PATH_SEARCH_BUDGET_EXHAUSTED',attempt_rounds=attempt,
                                orders=list(orders),per_search_timeout_s=30,errors=search_failures,
                                goal_joints_deg={k:v.tolist() for k,v in goal.items()})
                            raise DualArmError('PATH_SEARCH_BUDGET_EXHAUSTED: 路径搜索两轮仍未找到通过碰撞验证的完整路径；不是 IK 失败。'+ '; '.join(failures))
                        print(f'路径搜索第 {attempt}/2 轮未完成，最后再试一轮（不是 IK 求解）：'+ '; '.join(failures),flush=True)
                    for a,b in zip(path,path[1:]):
                        operations.extend(('move',s) for s in self._segments(unpack(a),unpack(b)))
                q=goal
            else:
                for k in ('left','right'):
                    if k not in kw:continue
                    segments,q=self._tcp_segments(k,kw[k],q)
                    operations.extend(('move',segment) for segment in segments)
            if preserve_command_boundaries:boundaries.add(first_operation)
            if len(operations)>8192:raise DualArmError('round exceeds planning segment budget')
        operations=self._join_moves(operations,boundaries)
        if len(operations)>512:raise DualArmError('round exceeds command budget')
        return operations

    def _preview_failure(self,error):
        report=getattr(error,'result',None)
        if report is not None and hasattr(self,'preview_failures'):
            self.preview_failures.append(dict(context=dict(self.planning_context),report=report))

    def _tcp_segments(self,k,pose,q,*,allow_detour=True):
        """Try Cartesian continuation, then replan this entire TCP move by RRT.

        Never keep the prefix of a failed straight approach: its last point may
        itself have no stopping clearance. Nothing is dispatched during search.
        """
        initial={a:v.copy() for a,v in q.items()}
        arm=self.config.arms[k]
        start=arm.world_from_base@pose_matrix(self.models[k].forward(q[k]))
        target=pose_matrix(np.r_[np.asarray(pose[:3])*1000,np.degrees(pose[3:])])
        rotation=Rotation.from_matrix(np.stack([start[:3,:3],target[:3,:3]]))
        steps=max(1,int(np.ceil(np.linalg.norm(target[:3,3]-start[:3,3])/5)),
                  int(np.ceil(np.degrees((rotation[0].inv()*rotation[1]).magnitude())/2)))
        segments=[]
        try:
            if steps>256:raise DualArmError('Cartesian target exceeds step budget')
            slerp=Slerp([0,1],rotation)
            for fraction in np.linspace(0,1,steps+1)[1:]:
                if self.cancel.is_set():raise KeyboardInterrupt('planning cancelled')
                self.planning_context.update(arm=k,route='tcp_direct',cartesian_fraction=float(fraction),cartesian_steps=steps)
                frame=start.copy();frame[:3,3]=start[:3,3]+fraction*(target[:3,3]-start[:3,3])
                frame[:3,:3]=slerp(fraction).as_matrix()
                goal={a:v.copy() for a,v in q.items()}
                goal[k]=self._inverse_target(k,matrix_pose(np.linalg.inv(arm.world_from_base)@frame),q[k],'tcp_direct_ik')
                checked=self.check(pack(goal)[:6],pack(goal)[6:])
                if not checked['safe']:raise DualArmError(checked['reason'])
                segments.extend(self._segments(q,goal));q=goal
            return segments,q
        except DualArmError as exc:
            if not allow_detour:raise
            direct_failure=f'{type(exc).__name__}: {exc}'
        if self.cancel.is_set():raise KeyboardInterrupt('planning cancelled')
        self.planning_context.update(arm=k,route='tcp_rrtconnect',direct_failure=direct_failure)
        self.planning_context.pop('cartesian_fraction',None)
        print(f'{k}: TCP 直达路径未通过，先求绕行终点 IK：{direct_failure}',flush=True)
        goal={a:v.copy() for a,v in initial.items()}
        target_base=matrix_pose(np.linalg.inv(arm.world_from_base)@target)
        # A partial continuation can provide a better IK seed, but it is never
        # used as the start of the detour or dispatched as a failed prefix.
        try:goal[k]=self._inverse_target(k,target_base,q[k],'rrt_goal_ik')
        except DualArmError:goal[k]=self._inverse_target(k,target_base,initial[k],'rrt_goal_ik')
        print(f'{k}: 终点 IK 通过，开始 RRTConnect 绕行搜索',flush=True)
        path=self.paths.joint_path(pack(initial),pack(goal),order=k+'_first',timeout_s=120)
        segments=[]
        for a,b in zip(path,path[1:]):
            if self.cancel.is_set():raise KeyboardInterrupt('planning cancelled')
            segments.extend(self._segments(unpack(a),unpack(b)))
        self.plan_stats['tcp_rrt_detours']=self.plan_stats.get('tcp_rrt_detours',0)+1
        return segments,goal

    def _inverse_target(self,arm,target_base,seed,phase):
        self.planning_context.pop('ik_failure',None)
        try:return self.models[arm].inverse(target_base,seed)
        except DualArmError as exc:
            report=dict(kind='IK_NO_ACCEPTABLE_SOLUTION',phase=phase,
                arm=arm,robot='arm6' if arm=='left' else 'arm7',
                command_index=self.planning_context.get('command_index'),
                target_base_mm_deg=np.asarray(target_base).tolist(),
                seed_joints_deg=np.asarray(seed).tolist(),
                required_position_error_mm=.5,required_orientation_error_deg=.2,
                candidate=getattr(exc,'ik_candidate',None),error=str(exc),
                diagnostics=getattr(exc,'ik_diagnostics',None),
                task_phase=self.planning_context.get('task_phase'),
                target_world_m_rad=np.r_[matrix_pose(self.config.arms[arm].world_from_base@pose_matrix(target_base))[:3]/1000,
                    np.radians(matrix_pose(self.config.arms[arm].world_from_base@pose_matrix(target_base))[3:])].tolist(),
                repair_instruction='Revise the reported target/pose or intermediate path using the saved observation. Do not execute a rejected candidate or lower acceptance tolerances.')
            self.planning_context['ik_failure']=report
            exc.ik_failure=report
            raise

    def _join_moves(self,operations,boundaries=()):
        """Use one native command across compatible adjacent source moves.

        Keep the ORIGINAL IK vertices until this pass: repeatedly compacting
        already compacted paths would accumulate the Cartesian deviation limit.
        A source statement alone is not a stop; grippers, arm switches and
        explicit arrival boundaries (Home/perception) are.
        """
        if not getattr(self.config, 'path_validation_enabled', True):
            self.plan_stats.update(path_validation='disabled_by_user',scope='all_collection',rrt=False)
            return operations
        output=[];pending=[]
        for index,(kind,data) in enumerate(operations):
            if pending and (index in boundaries or kind!='move' or data.arm!=pending[-1][1].arm):
                output.extend(self._compact_moves(pending));pending=[]
            if kind=='move':pending.append((kind,data))
            else:output.append((kind,data))
        if pending:output.extend(self._compact_moves(pending))
        self.plan_stats['removed_stops']=self.plan_stats['checked_segments']-self.plan_stats['dispatched_segments']
        return output

    def _compact_moves(self,operations):
        """Retain fine checks, but dispatch longer independently certified boxes.

        Called on adjacent same-arm moves: never crosses a gripper boundary.
        Reject shortcuts through joint reversals or appreciably curved IK paths.
        """
        self.plan_stats['checked_segments']+=len(operations)
        deadline=time.monotonic()+5.
        output=[]
        pending=[(0,len(operations))] if operations else []
        while pending:
            lo,hi=pending.pop()
            if hi-lo==1:
                output.append(operations[lo]);continue
            first,last=operations[lo][1],operations[hi-1][1]
            arm=first.arm
            merge=all(kind=='move' and segment.arm==arm for kind,segment in operations[lo:hi])
            a,b=first.start[arm],last.end[arm]
            delta=b-a;length2=float(delta@delta)
            if merge and length2>1e-16:
                if any(any(not np.allclose(previous.end[k],following.start[k],atol=1e-9,rtol=0)
                               for k in previous.end)
                       for (_,previous),(_,following) in zip(operations[lo:hi-1],operations[lo+1:hi])):
                    raise DualArmError('discontinuous native motion program')
                if any(np.any((s.end[arm]-s.start[arm])*delta < -1e-10)
                       for _,s in operations[lo:hi]):
                    merge=False
                for _,segment in operations[lo:hi-1]:
                    if not merge:break
                    vertex=segment.end[arm]
                    if np.any(vertex<np.minimum(a,b)-1e-8) or np.any(vertex>np.maximum(a,b)+1e-8):
                        merge=False;break
                    fraction=float(np.clip((vertex-a)@delta/length2,0,1))
                    error,angle=pose_error(self.models[arm].forward(vertex),
                                          self.models[arm].forward(a+fraction*delta))
                    if error>2. or angle>1.:
                        merge=False;break
            else:merge=False
            if merge and time.monotonic()<deadline:
                try:
                    self.paths.validator.certify_box(pack(first.start),pack(last.end),
                        deadline=min(deadline,time.monotonic()+.5),max_nodes=128)
                except DualArmError:
                    merge=False
            else:merge=False
            if merge:
                speed,accel,timeout=command_limits(self.config,self.models[arm],self.config.arms[arm],a,b)
                output.append(('move',PositionSegment(arm,first.start,last.end,speed,accel,timeout)))
            else:
                mid=(lo+hi)//2
                pending.extend([(mid,hi),(lo,mid)])
        self.plan_stats['dispatched_segments']+=len(output)
        return output

    def arrival(self,target):
        c=self.coordinator
        state=c.snapshots()
        for k,q in target.items():
            if np.max(np.abs(np.asarray(state[k]['joints'])-q))>self.config.limits['stationary_tolerance_deg']:
                raise DualArmError('robot moved from expected stationary state')
            distance,angle=pose_error(state[k]['pose'],self.models[k].forward(state[k]['joints']))
            if distance>2 or angle>1:
                raise DualArmError(f'{k}: controller/model FK mismatch ({distance:.3f} mm, {angle:.3f} deg)')
        validate_sample(self.config,self.models,{k:np.asarray(v['joints']) for k,v in state.items()},holding=False)

    def adjust_motion(self,source,folder):
        return source

    def execute(self,source,observation,folder):
        from .trajectory_preview import record_preview
        self.preview_candidates=[];self.preview_failures=[]
        self.planning_context={'stage':'validate_observation_and_height_policy'}
        initial={k:np.asarray(v) for k,v in observation.meta['joints_deg'].items()}
        timings={};stage='trajectory_planning_s';began=time.perf_counter()
        try:
            source=self.adjust_motion(source,folder)
            if self.config.raw.get('astra_grasp_height',{}).get('mode')=='shared_arm7_target_z':
                from .codex_grasp_height import share_arm7_grasp_z
                source,changes=share_arm7_grasp_z(source)
                (Path(folder)/'effective_motion.py').write_text(source)
                (Path(folder)/'grasp_height_policy.json').write_text(json.dumps(
                    dict(mode='shared_arm7_target_z',changes=changes),indent=2)+'\n')
            observation.validate_for(self.config,live=True,check_age=False)
            operations=self.plan(source,initial)
            self.auto_grasp_branch=None
            if getattr(self,'auto_grasp_debug',False):
                from .auto_grasp_debug import prepare_branch
                self.auto_grasp_branch=prepare_branch(self,source,operations,folder)
            timings[stage]=time.perf_counter()-began
            stage='preview_s';began=time.perf_counter()
            record_preview(self,source,initial,operations,None,folder)
            timings[stage]=time.perf_counter()-began
            stage='preview_wait_s';began=time.perf_counter()
            preview=getattr(self,'preview',None)
            if preview is not None:preview.await_execution(self.cancel)
            timings[stage]=time.perf_counter()-began
            stage='pre_dispatch_check_s';began=time.perf_counter()
            self.planning_context={'stage':'validate_before_dispatch'}
            observation.validate_for(self.config,live=True,check_age=False)
            self.arrival(initial)
        except Exception as exc:
            timings[stage]=time.perf_counter()-began
            try:
                record_preview(self,source,initial,None,f'{type(exc).__name__}: {exc}',folder)
            except Exception as preview_error:
                print(f'预览记录失败：{preview_error}',flush=True)
            return dict(status='PLANNING_INCOMPLETE' if isinstance(exc,ProofIncompleteError) else 'PLANNING_FAILED',error=f'{type(exc).__name__}: {exc}',physical_execution=False,
                        failure_context=self.planning_context,timings_s=timings,
                        failure_kind='IK_FAILED' if hasattr(exc,'ik_failure') else 'PLANNING_FAILED',
                        ik_failure=getattr(exc,'ik_failure',None),
                        rejected_motion_source=source)
        timings[stage]=time.perf_counter()-began
        began=time.perf_counter()
        recorder=None
        # Preparation calls _execute_operations directly and is not recorded.
        # Here the full grasp/fold/release/Home plan has passed validation.
        if getattr(self,'camera',None) is not None:
            from .motion_recording import MotionRecorder
            recorder=MotionRecorder(self.camera,folder,fps=self.config.raw['camera']['fps'],
                                    camera_label=self.config.raw['camera'].get('label','A'))
            try:
                recorder.start()
            except Exception as exc:
                return dict(status='EXECUTION_FAILED',physical_execution=False,
                            error=f'Recording initialization failed: {exc}',timings_s=timings)
        try:
            result=self._execute_operations(operations,folder)
        finally:
            recording=recorder.stop() if recorder is not None else None
        if recording is not None:result['recording']=recording
        timings['execution_s']=time.perf_counter()-began
        result['timings_s']=timings
        return result

    def _execute_operations(self,operations,folder):
        folder=Path(folder)
        (folder/'planning_stats.json').write_text(json.dumps(getattr(self,'plan_stats',{}),indent=2)+'\n')
        (folder/'native_plan.json').write_text(json.dumps([
            dict(kind=kind,data=data.to_dict() if kind=='move' else data) for kind,data in operations],indent=2))
        c=self.coordinator;completed=0;kind='prepare';data={}
        try:
            if not self.prepared:
                for key in getattr(self,'active_arms',tuple(c.connections)):
                    c.connections[key].prepare_position()
                self.prepared=True
            queue=list(operations); unstick_used=False
            while queue:
                kind,data=queue.pop(0)
                if kind=='move':
                    self.arrival(data.start)
                    c.event('astra_move_start',**data.to_dict())
                    if hasattr(data,'target_pose'):
                        from .queued_motion import check_queued_feedback
                        wait_operation(c,lambda s=data:c.connections[s.arm].move_tcp_position(
                            s.target_pose,s.cartesian_speed,s.cartesian_acceleration,s.timeout),
                            data.timeout,lambda s=data:check_queued_feedback(c,s))
                    elif hasattr(data,'segments'):
                        from .queued_motion import check_queued_feedback
                        wait_operation(c,lambda s=data:c.connections[s.arm].move_joint_queue(s.segments),
                                       data.timeout,lambda s=data:check_queued_feedback(c,s))
                    else:
                        wait_operation(c,lambda s=data:c.connections[s.arm].move_joint_position(s.end[s.arm],s.speed,s.acceleration,s.timeout),
                                       data.timeout,lambda s=data:check_feedback(c,s))
                    self.arrival(data.end)
                else:
                    target={k:np.asarray(v) for k,v in data['joints'].items()}
                    self.arrival(target)
                    c.event('astra_gripper_start',index=completed+1,arm=data['arm'],action=data['action'],
                            opening_delta_pulse=data.get('opening_delta_pulse'),
                            speed=self.config.arms[data['arm']].gripper.gripper_speed)
                    feedback_box={}
                    def command_gripper(d=data,index=completed+1):
                        try:
                            connection=c.connections[d['arm']]
                            old_confirmation=getattr(connection,'debug_reclose_confirmation',False)
                            connection.debug_reclose_confirmation=bool(getattr(self,'auto_grasp_branch',None) and d['action']=='close')
                            try:
                                if d['action']=='release_partial':
                                    feedback=connection.gripper(d['action'],opening_delta_pulse=d['opening_delta_pulse'])
                                else:
                                    feedback=connection.gripper(d['action'])
                            finally:
                                connection.debug_reclose_confirmation=old_confirmation
                            feedback_box['value']=feedback
                        except BaseException as error:
                            record=dict(arm=d['arm'],action=d['action'],error=f'{type(error).__name__}: {error}',
                                        completion=getattr(error,'gripper_completion',None))
                            (folder/f'gripper_{index:03d}.json').write_text(json.dumps(record,indent=2)+'\n')
                            raise
                        (folder/f'gripper_{index:03d}.json').write_text(json.dumps(
                            dict(arm=d['arm'],action=d['action'],feedback=feedback),indent=2)+'\n')
                        return feedback
                    wait_operation(c,command_gripper,
                                   self.config.arms[data['arm']].gripper.gripper_completion_timeout_s+1,
                                   lambda:self.arrival(target))
                    if data.get('settle_s', 0):
                        c.event('release_settle', seconds=data['settle_s'])
                        deadline = time.monotonic() + data['settle_s']
                        while time.monotonic() < deadline:
                            if self.cancel.wait(min(.02, max(0., deadline-time.monotonic()))):
                                raise DualArmError('cancelled during release settling')
                            self.arrival(target)
                    branch=getattr(self,'auto_grasp_branch',None)
                    if branch and data['action']=='close':
                        from .auto_grasp_debug import close_evidence
                        decision=close_evidence(feedback_box.get('value',{}))
                        (folder/f'auto_close_{completed+1:03d}.json').write_text(json.dumps(
                            dict(decision=decision,unstick_used=unstick_used),indent=2))
                        if unstick_used:
                            if decision in {'WIDE_STALLED','UNKNOWN'}:
                                raise DualArmError('automatic unstick unresolved; transport blocked')
                        elif completed==branch['close_index'] and decision=='WIDE_STALLED':
                            # Already stopped and measured stable; no early return from closing wait.
                            # All alternate moves were planned/certified before initial dispatch.
                            if getattr(self,'debug_unstick_attempts',0)>=1:
                                raise DualArmError('automatic unstick budget exhausted; transport blocked')
                            self.debug_unstick_attempts=getattr(self,'debug_unstick_attempts',0)+1
                            queue=list(branch['operations']);unstick_used=True
                            c.event('astra_auto_unstick',lift_mm=2,reason=decision)
                completed+=1;c.event('astra_command_complete',index=completed,kind=kind)
            return dict(status='EXECUTED',physical_execution=True,completed_commands=completed)
        except BaseException as exc:
            stopped=self.stop()
            pending=getattr(c,'native_operation_future',None)
            if pending is not None:
                from concurrent.futures import wait
                wait([pending],timeout=2)
            if isinstance(exc,KeyboardInterrupt):raise
            return dict(status='EXECUTION_FAILED',error=f'{type(exc).__name__}: {exc}',
                        physical_execution=True,completed_commands=completed,stop_results=stopped,
                        failed_command=dict(index=completed+1,kind=kind,
                            data=data.to_dict() if kind=='move' else data),
                        gripper_completion=getattr(exc,'gripper_completion',None))

    def stop(self):
        return self.coordinator.stop_all() if self.coordinator is not None else {}

    def close(self):
        try:
            if getattr(self,'camera',None) is not None:
                self.camera.stop()
        finally:
            self.camera=None
            self._close_connections()

    def _close_connections(self):
        if self.coordinator is not None:
            self.coordinator.close();self.coordinator=None
        if self._lease is not None:
            self._lease.close();self._lease=None


def q_to_lists(q):return {k:np.asarray(v).tolist() for k,v in q.items()}
