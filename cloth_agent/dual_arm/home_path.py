"""Home path shape limits and coordinated Cartesian return motions."""
import numpy as np

from .config import number
from .geometry import DualArmError, matrix_pose, pose_matrix
from .planning import Phase, compile_motion


def settings(config):
    raw=config.raw.get('home_path',{})
    if not isinstance(raw,dict):
        raise DualArmError('home_path must be an object')
    return {key:number(raw.get(key,default),f'home_path.{key}',0,500)
            for key,default in [('extra_height_mm',20.),('corridor_radius_mm',100.),
                                ('extra_path_length_mm',100.),('separation_distance_mm',50.)]}


def world_pose(model,arm,q):
    return matrix_pose(arm.world_from_base@pose_matrix(model.forward(q)))


def corridor(initial,config,models):
    limits=settings(config)
    result={}
    for key,arm in config.arms.items():
        start=world_pose(models[key],arm,initial[key])[:3]
        end=world_pose(models[key],arm,arm.home_joints)[:3]
        result[key]={'start':start,'end':end,'height':max(start[2],end[2])+limits['extra_height_mm'],
                     'radius':limits['corridor_radius_mm'],
                     'length':float(np.linalg.norm(end-start))+limits['extra_path_length_mm']}
    return result


def deviation(points,bounds):
    points=np.asarray(points)
    delta=bounds['end']-bounds['start'];length2=float(delta@delta)
    t=np.clip((points-bounds['start'])@delta/length2,0,1) if length2 else np.zeros(points.shape[:-1])
    return np.linalg.norm(points-bounds['start']-t[...,None]*delta,axis=-1)


def check_program_path(program,config,models):
    initial={k:q[0] for k,q in program[0].joints.items()}
    bounds=corridor(initial,config,models)
    metrics={}
    for key,arm in config.arms.items():
        points=np.asarray([world_pose(models[key],arm,q)[:3]
                           for motion in program[:-1] for q in motion.joints[key]])
        length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
        height=float(points[:,2].max());offset=float(deviation(points,bounds[key]).max())
        if height>bounds[key]['height']+1e-6:
            raise DualArmError(f'{key}: Home path exceeds height limit')
        if offset>bounds[key]['radius']+1e-6 or length>bounds[key]['length']+1e-6:
            raise DualArmError(f'{key}: Home path exceeds detour limit')
        metrics[key]={'max_height_mm':height,'height_limit_mm':bounds[key]['height'],
                      'length_mm':length,'length_limit_mm':bounds[key]['length'],
                      'max_deviation_mm':offset}
    return bounds,metrics


def clearance_options(current,target,group,config,models):
    limits=settings(config);candidates=[]
    for key in group:
        arm=config.arms[key];start=world_pose(models[key],arm,current[key])[:3]
        for j in range(arm.axis):
            delta=target[key][j]-current[key][j]
            for magnitude in (.5,1.,2.,5.):
                if abs(delta)<1e-9:
                    continue
                end={k:q.copy() for k,q in current.items()}
                end[key][j]+=np.sign(delta)*min(magnitude,abs(delta))
                point=world_pose(models[key],arm,end[key])[:3]
                rise=point[2]-start[2];travel=float(np.linalg.norm(point-start))
                if 1.<rise<=limits['extra_height_mm'] and travel<=limits['separation_distance_mm']:
                    candidates.append((rise/(travel+1),key,j,end))
    return sorted(candidates,key=lambda r:r[0],reverse=True)


def coordinated_return(current,target,group,config,models,name,*,clearance=False,joint=False):
    """Straight TCP reference with seeded IK, then a checked exact-joint finish."""
    from .homing import joint_motion
    from .stop_sweep import search_proof_budget
    if clearance:
        # A small clearance step can make the straight return possible near a
        # table. Its TCP rise is capped at 20 mm, never a full-joint Home swing.
        failures=[]
        for _,key,j,end in clearance_options(current,target,group,config,models):
            try:
                with search_proof_budget(20.):
                    prep=joint_motion(current,end,config,models,name+f'_clearance_{key}_{j+1}')
                    rest=coordinated_return(end,target,group,config,models,name+'_return',joint=joint)
                motions=[prep,*rest]
                check_program_path([*motions,Phase('home_open_grippers','open')],config,models)
                return motions
            except DualArmError as exc:
                failures.append(str(exc))
        raise DualArmError(f'no low clearance waypoint for coordinated Home: {failures}')
    if joint:
        exact={k:target[k].copy() if k in group else q.copy() for k,q in current.items()}
        intermediate={k:q.copy() for k,q in exact.items()}
        for key in group:
            tiny=np.abs(exact[key]-current[key])<=config.limits['stationary_tolerance_deg']
            intermediate[key][tiny]=current[key][tiny]
        with search_proof_budget(20.):
            result=[joint_motion(current,intermediate,config,models,name)]
            if any(np.max(np.abs(intermediate[k]-exact[k]))>1e-9 for k in current):
                result.append(joint_motion(intermediate,exact,config,models,name+'_final_trim'))
        check_program_path([*result,Phase('home_open_grippers','open')],config,models)
        return result
    starts={k:world_pose(models[k],a,current[k]) for k,a in config.arms.items()}
    targets={k:world_pose(models[k],a,target[k]) if k in group else starts[k]
             for k,a in config.arms.items()}
    phase=Phase(name,'move',targets,fixed_arms=tuple(k for k in current if k not in group))
    inverse={k:m.inverse for k,m in models.items()}
    for key in group:
        if config.arms[key].axis!=7:
            continue
        start_base=np.asarray(models[key].forward(current[key]))[:3]
        delta_base=np.asarray(models[key].forward(target[key]))[:3]-start_base
        length2=float(delta_base@delta_base)
        if length2<=1e-9:
            continue
        def guided(pose,seed,*,key=key,start=start_base,delta=delta_base,length2=length2):
            progress=float(np.clip((np.asarray(pose)[:3]-start)@delta/length2,0,1))
            # Guide the redundant degree of freedom continuously toward the
            # taught Home solution instead of leaving a large final joint swing.
            guide=(1-progress)*current[key]+progress*target[key]
            return models[key].inverse(pose,guide)
        inverse[key]=guided
    with search_proof_budget(20.):
        motion=compile_motion(phase,current,config,models,inverse)
    result=[motion]
    end={k:q[-1].copy() for k,q in motion.joints.items()}
    exact={k:target[k] if k in group else end[k] for k in end}
    if any(np.max(np.abs(end[k]-exact[k]))>1e-9 for k in end):
        with search_proof_budget(20.):
            result.append(joint_motion(end,exact,config,models,name+'_exact_joints'))
    check_program_path([*result,Phase('home_open_grippers','open')],config,models)
    return result


def interleaved_return(initial,target,config,models,name,*,steps=6,clearance=False):
    """Alternate short coordinated arm motions along their Home joint paths.

    Both arms make progress, instead of forcing one to stay beside the other
    until the other arm has completed its whole return.
    """
    from .homing import joint_motion
    from .stop_sweep import search_proof_budget
    keys=tuple(initial);current={k:q.copy() for k,q in initial.items()};program=[]
    if clearance:
        for key in keys:
            failures=[]
            for _,_,j,end in clearance_options(current,target,(key,),config,models):
                try:
                    with search_proof_budget(20.):
                        prep=joint_motion(current,end,config,models,f'{name}_clearance_{key}_{j+1}')
                    program.append(prep);current={k:q[-1].copy() for k,q in prep.joints.items()}
                    break
                except DualArmError as exc:
                    failures.append(str(exc))
            else:
                raise DualArmError(f'no short clearance move for {key}: {failures}')
        program.extend(interleaved_return(current,target,config,models,name+'_return',steps=steps))
        check_program_path([*program,Phase('home_open_grippers','open')],config,models)
        return program
    goals={k:q.copy() for k,q in target.items()}
    for k in keys:
        tiny=np.abs(goals[k]-initial[k])<=config.limits['stationary_tolerance_deg']
        goals[k][tiny]=initial[k][tiny]
    for step in range(1,steps+1):
        waypoint={k:initial[k]+(goals[k]-initial[k])*step/steps for k in keys}
        failures=[]
        for order in (keys,keys[::-1]):
            q={k:v.copy() for k,v in current.items()};moves=[]
            try:
                for key in order:
                    end={k:waypoint[k] if k==key else v for k,v in q.items()}
                    with search_proof_budget(20.):
                        motion=joint_motion(q,end,config,models,f'{name}_{step}_{key}')
                    moves.append(motion);q={k:v[-1].copy() for k,v in motion.joints.items()}
                break
            except DualArmError as exc:
                failures.append(str(exc))
        else:
            raise DualArmError(f'no coordinated interleaved continuation: {failures}')
        program.extend(moves);current=q
    for key in keys:
        end={k:target[k] if k==key else v for k,v in current.items()}
        if np.max(np.abs(end[key]-current[key]))>1e-9:
            with search_proof_budget(20.):
                program.append(joint_motion(current,end,config,models,f'{name}_trim_{key}'))
            current={k:v.copy() for k,v in end.items()}
    check_program_path([*program,Phase('home_open_grippers','open')],config,models)
    return program
