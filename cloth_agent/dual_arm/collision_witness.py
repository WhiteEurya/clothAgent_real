"""Bounded falsification only: every candidate is checked by the strict oracle."""
import numpy as np
from .geometry import DualArmError, segment_distance, segment_box_distance
from .mesh_narrowphase import check_cell_collision


class WitnessFound(Exception):
    def __init__(self,witness):self.witness=witness


class WitnessSearchDone(Exception):
    pass


def collision_record(config,models,q,capsules,error,*,tool_contact=False):
    text=str(error)
    if not text.startswith(('inter-arm collision: ', 'obstacle collision: ')):
        return None
    objects=text.split(': ',1)[1].split(', ')
    by_name={f'{k}/{c.name}':c for k,rows in capsules.items() for c in rows}
    gap=None
    if all(name in by_name for name in objects):
        a,b=(by_name[name] for name in objects)
        gap=segment_distance(a.start,a.end,b.start,b.end)-a.radius-b.radius-config.limits['clearance_mm']
    else:
        cap=by_name.get(objects[0]);box=next((b for b in config.obstacles if b['name']==objects[1]),None)
        if cap is not None and box is not None:
            margin=config.limits['clearance_mm'];arm_id=objects[0].split('/')[0]
            if ((tool_contact is True or arm_id in (tool_contact or ())) and cap.name=='tool'
                    and 'tool_contact_allowance_mm' in box):margin=-float(box['tool_contact_allowance_mm'])
            gap=segment_box_distance(cap.start,cap.end,box['min_mm'],box['max_mm'])-cap.radius-margin
    return dict(objects=objects,joints_deg={k:np.asarray(v).tolist() for k,v in q.items()},gap_mm=gap,
                gap_definition='signed configured capsule clearance including position margins; not mesh penetration depth',reason=text)


def find_camera_witness(config,models,low,high,cell_capsules,checkpoint,*,tool_contact=False,max_evaluations=96,enclosure=None):
    """Search closest camera pairs in the unchanged joint box; no SAFE return."""
    from scipy.optimize import minimize
    keys=list(models);center={k:(low[k]+high[k])/2 for k in keys};half={k:(high[k]-low[k])/2 for k in keys}
    checkpoint();coarse=enclosure if enclosure is not None else cell_capsules(config,models,center,half)
    pairs=[];zero={k:np.zeros_like(v) for k,v in center.items()}
    nominal=None
    for a in coarse[keys[0]]:
        for b in coarse[keys[1]]:
            if 'camera' not in a.name or 'camera' not in b.name:continue
            gap=segment_distance(a.start,a.end,b.start,b.end)-a.radius-b.radius-config.limits['clearance_mm']
            if gap<=0:
                if nominal is None:nominal=cell_capsules(config,models,center,zero)
                na=next(c for c in nominal[keys[0]] if c.name==a.name)
                nb=next(c for c in nominal[keys[1]] if c.name==b.name)
                nominal_gap=segment_distance(na.start,na.end,nb.start,nb.end)-na.radius-nb.radius-config.limits['clearance_mm']
                pairs.append((nominal_gap,a.name,b.name))
    if not pairs:return None,0
    sizes=[len(low[k]) for k in keys];cuts=np.cumsum([0,*sizes]);mid=np.concatenate([center[k] for k in keys]);width=np.concatenate([half[k] for k in keys]);zero={k:np.zeros_like(v) for k,v in center.items()}
    evaluations=0
    for _,left_name,right_name in sorted(pairs):
        def objective(x):
            nonlocal evaluations
            checkpoint()
            if evaluations>=max_evaluations:raise WitnessSearchDone()
            evaluations+=1
            values=mid+np.clip(x,-1,1)*width
            # Clip and validate again: an out-of-box optimizer result cannot reject a path.
            q={k:np.clip(values[cuts[i]:cuts[i+1]],low[k],high[k]) for i,k in enumerate(keys)}
            caps=cell_capsules(config,models,q,zero)
            a=next(c for c in caps[keys[0]] if c.name==left_name);b=next(c for c in caps[keys[1]] if c.name==right_name)
            gap=segment_distance(a.start,a.end,b.start,b.end)-a.radius-b.radius-config.limits['clearance_mm']
            if gap<=0:
                try:check_cell_collision(config,models,q,caps,tool_contact=tool_contact)
                except DualArmError as error:
                    record=collision_record(config,models,q,caps,error,tool_contact=tool_contact)
                    if record is not None:raise WitnessFound(record)
            return float(gap)
        try:
            minimize(objective,np.zeros(len(mid)),method='L-BFGS-B',bounds=[(-1.,1.)]*len(mid),
                     options={'maxiter':8,'maxfun':max_evaluations,'eps':1e-5,'ftol':1e-9})
        except WitnessFound as found:return found.witness,evaluations
        except WitnessSearchDone:break
    return None,evaluations
