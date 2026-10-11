"""Parse declarative Python motion calls and hand them to the OMPL planner."""
import ast
import json
from pathlib import Path

import numpy as np

CONTRACT = '''最终回答只输出 Python 动作代码，不加 Markdown。只允许以下调用，参数使用数字列表/字符串字面量：
motion.move_joints(left=[6个关节弧度], right=[7个关节弧度])
motion.move_tcp(left=[x,y,z,roll,pitch,yaw], right=[x,y,z,roll,pitch,yaw])
motion.gripper(arm="left", action="open")
移动可省略保持不动的一侧；TCP 使用场景 World 坐标中的米/弧度，夹爪 arm 为 left/right，action 为 open/close。
可以按任务需要重复调用；不要输出 import、直接 SDK 指令或执行运动。宿主会把代码交给现有规划器。缺少观测或坐标依据时输出 motion.blocked(reason="具体原因")，不要编造坐标。'''


def parse_code(source, *, allow_complete=False, allow_grasp=False):
    tree = ast.parse(source)
    if not tree.body or len(tree.body)>256:
        raise ValueError('motion code must contain 1..256 calls')
    calls=[]
    for statement in tree.body:
        call=statement.value if isinstance(statement,ast.Expr) else None
        if not (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                and isinstance(call.func.value,ast.Name) and call.func.value.id=='motion'
                and call.func.attr in ({'move_joints','move_tcp','gripper','blocked'} | ({'complete'} if allow_complete else set()) | ({'grasp','move_transfer'} if allow_grasp else set())) and not call.args):
            raise ValueError('only declarative motion calls with literal keywords are allowed')
        kw={}
        for arg in call.keywords:
            if arg.arg is None or arg.arg in kw:
                raise ValueError('duplicate/expanded motion arguments')
            kw[arg.arg]=ast.literal_eval(arg.value)
        name=call.func.attr
        if name in ('move_joints','move_tcp','move_transfer','grasp'):
            if not kw or set(kw)-{'left','right'}:
                raise ValueError('move requires left and/or right')
            for arm,value in kw.items():
                size=7 if name=='move_joints' and arm=='right' else 6
                a=np.asarray(value,dtype=float)
                if a.shape!=(size,) or not np.isfinite(a).all():
                    raise ValueError('invalid motion target')
        elif name=='gripper':
            if set(kw)!={'arm','action'} or kw['arm'] not in ('left','right') or kw['action'] not in ('open','close'):
                raise ValueError('invalid gripper event')
        elif name=='complete':
            if kw:raise ValueError('complete takes no arguments')
        elif set(kw)!={'reason'} or not isinstance(kw['reason'],str) or not kw['reason'].strip():
            raise ValueError('blocked requires a reason')
        calls.append((name,kw))
    if any(name in ('blocked','complete') for name,_ in calls) and len(calls)!=1:
        raise ValueError('blocked/complete cannot be mixed with motion')
    if any(name=='grasp' for name,_ in calls):
        if calls[0][0]!='grasp' or len(calls)<2 or any(name not in ('move_tcp','move_transfer') for name,_ in calls[1:]):
            raise ValueError('grasp task requires one initial grasp followed only by move_tcp/move_transfer actions')
        arms=set(calls[0][1]);moved=set()
        for _,kw in calls[1:]:
            if set(kw)-arms:raise ValueError('fold moves must use only the grasping arms')
            moved.update(kw)
        if moved!=arms:raise ValueError('each grasping arm requires a fold/place target')
    if any(name=='move_transfer' for name,_ in calls) and calls[0][0]!='grasp':
        raise ValueError('move_transfer requires a grasp task')
    return calls


def compile_code(source, planner, state):
    from cloth_agent.dual_arm.geometry import pose_matrix
    from cloth_agent.dual_arm.motion.trajectory import parameterize, validate_timed
    calls=parse_code(source)
    if calls[0][0]=='blocked':
        return {'success':False,'reason':calls[0][1]['reason'],'physical_execution_supported':False}
    if state.get('units')!='rad':
        raise ValueError('state must declare units=rad')
    q=np.r_[state['left'],state['right']].astype(float)
    if len(state['left'])!=6 or len(state['right'])!=7 or not np.isfinite(q).all():
        raise ValueError('state requires 6 left and 7 right joints')
    initial=q.copy()
    if not planner.checker.check(q[:6],q[6:])['safe']:
        raise ValueError('initial state rejected by collision checker')
    edges=[];events=[]
    for index,(name,kw) in enumerate(calls):
        if name=='gripper':
            events.append(dict(kind='gripper',waypoint=len(edges),**kw))
            continue
        corridors=None
        if name=='move_joints':
            goal=np.r_[kw.get('left',q[:6]),kw.get('right',q[6:])]
            path=planner.paths.joint_path(q,goal,order='left_first',timeout_s=30)
        else:
            targets=planner.scene.tcp_world(q[:6],q[6:])
            for arm,pose in kw.items():
                # Existing pose_matrix takes mm/degree; scene uses m/rad.
                frame=pose_matrix(np.r_[np.array(pose[:3])*1000,np.degrees(pose[3:])])
                frame[:3,3]/=1000
                targets[arm]=frame
            path,corridors=planner.paths.cartesian(q,targets,timeout_s=30)
        for i,(a,b) in enumerate(zip(path,path[1:])):
            edge=dict(phase=f'code_{index}',start_q_rad=np.asarray(a).tolist(),end_q_rad=np.asarray(b).tolist())
            if corridors is not None:
                edge['cartesian']=corridors[i]
            edges.append(edge)
        q=np.asarray(path[-1])
    if not edges:
        raise ValueError('no movement supplied; gripper-only execution is not supported')
    urdf_velocity=np.array([m.urdf.joint_map[n].limit.velocity for m in planner.scene.models.values() for n in m.names])
    timed=parameterize(edges,np.minimum(urdf_velocity,np.full(13,np.deg2rad(5))),np.full(13,np.deg2rad(10)))
    verification=validate_timed(timed,planner.validator)
    for event in events:
        event['time_s']=timed['waypoint_times_s'][event['waypoint']]
    return dict(success=True,units='m_rad_s',initial_q=initial.tolist(),events=events,
                verification=verification,physical_execution_supported=False,execution_authorized=False,**timed)


def handoff(code_path, scene_path, state_path, output):
    from cloth_agent.dual_arm.collision import CollisionScene
    from cloth_agent.dual_arm.motion import DualArmPlanner
    output=Path(output)
    try:
        result=compile_code(Path(code_path).read_text(),DualArmPlanner(CollisionScene.load(scene_path)),
                            json.loads(Path(state_path).read_text()))
    except Exception as exc:
        result=dict(success=False,reason=f'{type(exc).__name__}: {exc}',physical_execution_supported=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result
