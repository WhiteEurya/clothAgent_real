"""Host-planned empty-hand approach/return around model-provided cloth motion."""
import ast
from pathlib import Path
import json

import numpy as np

from .dual_arm.geometry import DualArmError,matrix_pose,pose_matrix
from .dual_arm.motion.home import pack,unpack
from .dual_arm.sequential_home import certify_segment
from .dual_arm.stop_sweep import complete_proof


TASK_CONTRACT='''最终回答只输出 Python 动作代码，不加 Markdown。格式为：
motion.grasp(left=[x,y,z,roll,pitch,yaw], right=[x,y,z,roll,pitch,yaw])
motion.move_tcp(left=[x,y,z,roll,pitch,yaw], right=[x,y,z,roll,pitch,yaw])
motion.move_tcp(...)
第一条 grasp 只声明真实抓点和夹爪方向，不是抓点上方的接近点。
后续 move_tcp / move_transfer 列表只描述抓住衣物后，从抓点到最终放置目标的必要动作；最后的位置应适合松开衣物。
move_tcp 保持指定的笛卡尔路径，用于折叠关键段。
motion.move_transfer(left=[x,y,z,roll,pitch,yaw], right=[x,y,z,roll,pitch,yaw])
move_transfer 只约束终点位姿，中间允许关节路径绕行及改变姿态；仅用于衣物允许自由转移的段，不用于必须保持形状或双爪相对距离的段。两臂仍先后运动。
工作区边缘不要强制竖直抬升或保持竖直姿态：可写向各自基座收回并抬升的可达转移点，必要时改变该点倾角。指定终点仍须通过 IK，宿主不会偷偷移动抓点或折叠关键点。
TCP 使用 World 坐标的米/弧度。move_tcp 可省略保持不动的一侧。
宿主负责开爪、接近、下降抓取、闭合，以及最后松开、撤离和回 Home。
不要输出这些前后辅助动作，不输出 gripper、move_joints、import 或 SDK 指令。
宿主对 grasp 的原始观测 Z 额外降低 10 mm；不要自己重复减去偏移，中间动作目标不加此偏移。
缺少坐标依据时单独输出 motion.blocked(reason="具体原因")。
已叠好时单独输出 motion.complete()。'''


def lower_task_grasps(source,folder,*,offset_mm=10.):
    """Apply the configured grasp offset exactly once to the declaration."""
    from .codex_motion import parse_code
    calls=parse_code(source,allow_grasp=True)
    if calls[0][0]!='grasp':return None
    tree=ast.parse(source);changes=[]
    for keyword in tree.body[0].value.keywords:
        pose=list(calls[0][1][keyword.arg]);old=pose[2];pose[2]-=offset_mm/1000
        keyword.value=ast.parse(repr(pose),mode='eval').body
        changes.append(dict(arm=keyword.arg,original_z_mm=old*1000,applied_z_mm=pose[2]*1000))
    effective=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    Path(folder,'effective_motion.py').write_text(effective)
    Path(folder,'grasp_height_policy.json').write_text(json.dumps(
        dict(mode='declared_grasp_lowering',offset_mm=offset_mm,changes=changes),indent=2)+'\n')
    return effective


def _copy(q):return {k:np.asarray(v).copy() for k,v in q.items()}


def _context(runtime,index,phase,arm=None,**details):
    runtime.planning_context=dict(stage='plan_command',command_index=index,
        command='grasp' if index==1 else 'move_tcp',task_phase=phase,**details)
    if arm is not None:runtime.planning_context['arm']=arm


def _grippers(runtime,q,arms,action):
    with complete_proof(cancel=runtime.cancel,progress=runtime._proof_progress):
        certify_segment(q,q,runtime.config,runtime.models)
    return [('gripper',dict(arm=k,action=action,joints={a:v.tolist() for a,v in q.items()})) for k in arms]


def _joint_goal(runtime,arm,goal,q):
    path=runtime.paths.joint_path(pack(q),pack(goal),order=arm+'_first',timeout_s=120)
    operations=[]
    for a,b in zip(path,path[1:]):
        operations.extend(('move',s) for s in runtime._segments(unpack(a),unpack(b)))
    return operations,goal


def _clearance_targets(runtime,arm,pose):
    """Prefer existing lifts; also try inward diagonal access at workspace edges."""
    pose=np.asarray(pose,dtype=float)
    for height in (60.,40.,20.,10.):
        target=pose.copy();target[2]+=height/1000
        yield height,0.,target.tolist()
    toward=runtime.config.arms[arm].world_from_base[:2,3]/1000-pose[:2]
    length=np.linalg.norm(toward)
    if length<1e-9:return
    toward/=length
    for inward in (20.,40.,60.):
        for height in (20.,10.):
            target=pose.copy();target[:2]+=toward*inward/1000;target[2]+=height/1000
            yield height,inward,target.tolist()


def _transfer(runtime,arm,pose,q):
    world=pose_matrix(np.r_[np.array(pose[:3])*1000,np.degrees(pose[3:])])
    base=matrix_pose(np.linalg.inv(runtime.config.arms[arm].world_from_base)@world)
    goal=_copy(q)
    goal[arm]=runtime._inverse_target(arm,base,q[arm],'transfer_goal_ik')
    return _joint_goal(runtime,arm,goal,q)


def _approach(runtime,arm,pose,q):
    """Choose a joint-space route to a pregrasp, then descend without detours."""
    failures=[]
    for height_mm,inward_mm,hover in _clearance_targets(runtime,arm,pose):
        if runtime.cancel.is_set():raise KeyboardInterrupt('planning cancelled')
        _context(runtime,1,'host_approach',arm,grasp_world_m_rad=pose,
                 approach_height_mm=height_mm,inward_mm=inward_mm,generated_target_world_m_rad=hover)
        try:
            frame=pose_matrix(np.r_[np.array(hover[:3])*1000,np.degrees(hover[3:])])
            base=matrix_pose(np.linalg.inv(runtime.config.arms[arm].world_from_base)@frame)
            goal=_copy(q)
            goal[arm]=runtime._inverse_target(arm,base,q[arm],'host_approach_ik')
            moves,at_hover=_joint_goal(runtime,arm,goal,q)
            _context(runtime,1,'host_grasp_descent',arm,grasp_world_m_rad=pose,approach_height_mm=height_mm,inward_mm=inward_mm)
            descent,at_grasp=runtime._tcp_segments(arm,pose,at_hover,allow_detour=False)
            return moves+[('move',s) for s in descent],at_grasp
        except DualArmError as exc:
            failures.append(dict(height_mm=height_mm,inward_mm=inward_mm,error=str(exc),context=dict(runtime.planning_context)))
            last=exc
    runtime.planning_context['host_attempts']=failures
    raise last


def _retreat(runtime,arm,q,index):
    world=runtime.config.arms[arm].world_from_base@pose_matrix(runtime.models[arm].forward(q[arm]))
    pose=matrix_pose(world);pose=np.r_[pose[:3]/1000,np.radians(pose[3:])]
    failures=[]
    for height_mm,inward_mm,hover in _clearance_targets(runtime,arm,pose):
        if runtime.cancel.is_set():raise KeyboardInterrupt('planning cancelled')
        _context(runtime,index,'host_retreat',arm,retreat_height_mm=height_mm,
                 inward_mm=inward_mm,generated_target_world_m_rad=hover)
        try:
            segments,raised=runtime._tcp_segments(arm,hover, q,allow_detour=False)
            goal=_copy(raised);goal[arm]=runtime.config.arms[arm].home_joints.copy()
            _context(runtime,index,'host_return_home',arm,retreat_height_mm=height_mm)
            moves,home=_joint_goal(runtime,arm,goal,raised)
            return [('move',s) for s in segments]+moves,home
        except DualArmError as exc:
            failures.append(dict(height_mm=height_mm,inward_mm=inward_mm,error=str(exc),context=dict(runtime.planning_context)))
            last=exc
    runtime.planning_context['host_attempts']=failures
    raise last


def _ordered_phase(runtime,q,arms,operation):
    failures=[]
    orders=[arms] if len(arms)==1 else [arms,tuple(reversed(arms))]
    for order in orders:
        trial=_copy(q);moves=[]
        try:
            for arm in order:
                chunk,trial=operation(arm,trial)
                moves.extend(chunk)
            return moves,trial
        except DualArmError as exc:
            failures.append(dict(order=order,error=str(exc),context=dict(runtime.planning_context)))
            last=exc
    runtime.planning_context['phase_attempts']=failures
    raise last


def plan_grasp_task(runtime,calls,initial):
    """Build the entire pick/fold/place/return program before any dispatch."""
    grasps=calls[0][1];arms=tuple(k for k in ('left','right') if k in grasps)
    q=_copy(initial)
    for arm,pose in grasps.items():
        _context(runtime,1,'host_grasp_goal_check',arm,grasp_world_m_rad=pose)
        world=pose_matrix(np.r_[np.array(pose[:3])*1000,np.degrees(pose[3:])])
        base=matrix_pose(np.linalg.inv(runtime.config.arms[arm].world_from_base)@world)
        runtime._inverse_target(arm,base,q[arm],'grasp_goal_ik')
    _context(runtime,1,'host_open')
    operations=_grippers(runtime,q,arms,'open')
    moves,q=_ordered_phase(runtime,q,arms,lambda k,state:_approach(runtime,k,grasps[k],state))
    operations.extend(moves)
    _context(runtime,1,'host_close')
    operations.extend(_grippers(runtime,q,arms,'close'))
    for index,(name,targets) in enumerate(calls[1:],2):
        for arm,pose in targets.items():
            _context(runtime,index,'gpt_cloth_motion',arm,arguments=targets)
            runtime.planning_context['command']=name
            if name=='move_transfer':
                runtime.planning_context['task_phase']='gpt_cloth_transfer'
                moves,q=_transfer(runtime,arm,pose,q)
                operations.extend(moves)
            else:
                # Critical cloth motion retains its Cartesian path exactly.
                segments,q=runtime._tcp_segments(arm,pose,q,allow_detour=False)
                operations.extend(('move',s) for s in segments)
            if len(operations)>8192:raise DualArmError('round exceeds planning segment budget')
    _context(runtime,len(calls),'host_release')
    operations.extend(_grippers(runtime,q,arms,'open'))
    moves,q=_ordered_phase(runtime,q,arms,lambda k,state:_retreat(runtime,k,state,len(calls)))
    operations.extend(moves)
    operations=runtime._join_moves(operations,set())
    if len(operations)>512:raise DualArmError('round exceeds command budget')
    runtime.plan_stats['host_managed_grasp_task']=True
    return operations
