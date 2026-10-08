"""Plan single-joint escape steps that increase the closest inter-arm gap."""
import numpy as np

from .geometry import DualArmError, segment_distance
from .mesh_narrowphase import separating_gap
from .stop_sweep import search_proof_budget


def minimum_gap(models, joints):
    """Conservative shape separation estimate, used for direction ranking only.

    Actual acceptance still uses the full mesh, uncertainty and stopping proof.
    Taking the minimum over ALL pairs prevents improving one pair at the
    expense of an even closer camera, tool, or link pair.
    """
    keys=tuple(models)
    caps={k:models[k].capsules(joints[k]) for k in keys}
    pairs=sorted(((segment_distance(a.start,a.end,b.start,b.end)-a.radius-b.radius,a,b)
                  for a in caps[keys[0]] for b in caps[keys[1]]),key=lambda p:p[0])
    shapes={}
    def shape(key,cap):
        identity=(key,cap.name)
        if identity not in shapes:
            hull=models[key].collision_hull(joints[key],cap.name) if hasattr(models[key],'collision_hull') else None
            shapes[identity]=(hull if hull is not None else
                              (np.vstack([cap.start,cap.end]),np.empty((0,3)),cap.radius))
        return shapes[identity]
    gap=float('inf')
    for coarse,a,b in pairs:
        if coarse>=gap:
            continue
        pa,na,ma=shape(keys[0],a);pb,nb,mb=shape(keys[1],b)
        refined=coarse
        if len(na) or len(nb):
            refined=max(coarse,separating_gap(pa,na,pb,nb)-ma-mb)
        gap=min(gap,refined)
    return gap


def separation_steps(initial, config, models, attempts, *, max_steps=8):
    """Yield certified, cumulatively separating prefixes; never dispatch them.

    Move a single joint at a time so other axes do not acquire artificial
    commanded stopping excursions. Try both arms and both joint directions.
    """
    from .homing import joint_motion
    from .home_path import settings,world_pose
    current={k:np.asarray(q).copy() for k,q in initial.items()}
    origins={k:world_pose(models[k],a,initial[k])[:3] for k,a in config.arms.items()}
    limits=settings(config)
    prefix=[]
    for step in range(max_steps):
        before=minimum_gap(models,current)
        candidates=[]
        for key,arm in config.arms.items():
            for joint in range(arm.axis):
                for delta in (-2.,2.,-5.,5.,-10.,10.):
                    end={k:q.copy() for k,q in current.items()}
                    end[key][joint]+=delta
                    try:
                        gap=minimum_gap(models,end)
                        # Reject endpoint workspace violations before spending
                        # time on the continuous collision certificate.
                        arm.validate_point(models[key].forward(end[key])[:3])
                        point=world_pose(models[key],arm,end[key])[:3]
                        if np.linalg.norm(point-origins[key])>limits['separation_distance_mm']:
                            continue
                        if point[2]>origins[key][2]+limits['extra_height_mm']:
                            continue
                    except DualArmError:
                        continue
                    if gap > before+1.:
                        change=np.zeros(arm.axis);change[joint]=abs(delta)
                        travel=(models[key].local_capsule_motion_bounds(current[key],change)
                                if hasattr(models[key],'local_capsule_motion_bounds') else
                                models[key].capsule_motion_bounds(change))
                        efficiency=(gap-before)/(1.+max(travel.values(),default=0.))
                        candidates.append((efficiency,gap,key,joint,delta,end))
        accepted=None
        for _,gap,key,joint,delta,end in sorted(candidates,key=lambda x:x[0],reverse=True):
            label=f'separate_{step}_{key}_joint{joint+1}_{delta:+g}'
            try:
                # Reject directions whose nominal intermediate geometry initially
                # approaches, even if their endpoint is farther away.
                previous=before
                for fraction in np.linspace(0,1,11)[1:]:
                    q={k:current[k]+fraction*(end[k]-current[k]) for k in current}
                    value=minimum_gap(models,q)
                    if value < previous-0.05:
                        raise DualArmError('separation direction initially reduces closest gap')
                    previous=value
                    for key,arm in config.arms.items():
                        point=world_pose(models[key],arm,q[key])[:3]
                        if (np.linalg.norm(point-origins[key])>limits['separation_distance_mm'] or
                                point[2]>origins[key][2]+limits['extra_height_mm']):
                            raise DualArmError('separation exceeds short retreat limits')
                with search_proof_budget():
                    motion=joint_motion(current,end,config,models,f'home_{label}')
                # Recheck the timed path as well as its stopping envelope,
                # already covered by joint_motion's full sweep validation.
                previous=before
                for i in range(len(motion.times)):
                    value=minimum_gap(models,{k:q[i] for k,q in motion.joints.items()})
                    if value < previous-0.05:
                        raise DualArmError('separation path reduces closest gap')
                    previous=value
                accepted=motion
                attempts.append({'strategy':label,'status':'separation_step_checked',
                                 'gap_before_mm':before,'gap_after_mm':gap})
                break
            except DualArmError as exc:
                attempts.append({'strategy':label,'status':'rejected','reason':str(exc)})
        if accepted is None:
            return
        prefix.append(accepted)
        current={k:q[-1].copy() for k,q in accepted.joints.items()}
        yield current,list(prefix)
