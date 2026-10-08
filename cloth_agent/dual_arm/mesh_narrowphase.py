"""Separating-plane proofs for URDF convex hulls plus all configured margins."""
import itertools
import numpy as np

from .geometry import check_collision


def separating_gap(points_a, normals_a, points_b, normals_b, required_gap=None):
    """Any positive projection gap proves separation; absent proof is NOT clear."""
    delta=points_b.mean(axis=0)-points_a.mean(axis=0)
    # Most broad-phase false positives separate on a coordinate or center axis.
    # Avoid thousands of face-normal projections and tiny multi-threaded BLAS
    # calls unless the cheap support-plane tests cannot prove separation.
    gap=float(max(np.max(points_a.min(0)-points_b.max(0)),
                  np.max(points_b.min(0)-points_a.max(0))))
    if np.linalg.norm(delta)>1e-12:
        direction=delta/np.linalg.norm(delta)
        a=np.einsum('ij,j->i',points_a,direction)
        b=np.einsum('ij,j->i',points_b,direction)
        gap=max(gap,float(max(a.min()-b.max(),b.min()-a.max())))
    if required_gap is not None and gap>required_gap:
        return gap
    axes=np.unique(np.vstack((normals_a,normals_b)),axis=0)
    length=np.linalg.norm(axes,axis=1)
    axes=axes[length>1e-12]/length[length>1e-12,None]
    for offset in range(0,len(axes),32):
        batch=axes[offset:offset+32]
        a=np.einsum('ij,kj->ik',points_a,batch)
        b=np.einsum('ij,kj->ik',points_b,batch)
        gap=max(gap,float(max(np.max(a.min(axis=0)-b.max(axis=0)),
                              np.max(b.min(axis=0)-a.max(axis=0)))))
        if required_gap is not None and gap>required_gap:
            return gap
    return gap


def check_cell_collision(config,models,center,capsules,*,half_ranges=None,tool_contact=False):
    cache={}
    static_shells={}

    def shape(key,cap):
        identity=(key,cap.name)
        if identity in cache:
            return cache[identity]
        model=models[key]
        hull=model.collision_hull(center[key],cap.name) if hasattr(model,'collision_hull') else None
        if hull is None:
            value=(np.vstack([cap.start,cap.end]),np.empty((0,3)),cap.radius,False)
            static_shells[identity]=next((c['radius_mm'] for c in getattr(getattr(model,'config',None),'capsules',[])
                                          if c['name']==cap.name),cap.radius)
        else:
            points,normals,shell=hull
            nominal=next(c['radius_mm'] for c in model.config.capsules if c['name']==cap.name)
            # Endpoint travel bounds alone do not cover rotation of off-axis
            # mesh vertices. Add the capsule radial reach for every joint.
            # Counting non-ancestor joints as well is conservative.
            radial_motion = (0. if half_ranges is None or cap.name=='link_base' else
                             nominal*float(np.abs(np.radians(half_ranges[key])).sum()))
            value=(points,normals,shell+cap.radius-nominal+radial_motion,True)
            static_shells[identity]=shell
        cache[identity]=value
        return value

    def projected(key,cap,points,direction):
        lo,hi=models[key].projection_motion_interval(center[key],half_ranges[key],cap.name,points,direction)
        error=0.
        if config.raw['safety']['status']=='measured':
            row=config.raw['safety']['arms'][key]
            error=row['base_error_mm']+row['geometry_error_mm']+config.limits['tracking_error_mm']
        margin=static_shells[(key,cap.name)]+error
        return lo-margin,hi+margin

    def directional_available(*keys):
        return half_ranges is not None and all(hasattr(models[k],'projection_motion_interval') for k in keys)

    def directions(pa,pb):
        delta=pb.mean(0)-pa.mean(0)
        return [*np.eye(3),*([delta/np.linalg.norm(delta)] if np.linalg.norm(delta)>1e-12 else [])]

    def pair_separated(ka,a,kb,b,clearance):
        pa,na,ma,ha=shape(ka,a);pb,nb,mb,hb=shape(kb,b)
        required=ma+mb+clearance+1e-6
        if (ha or hb) and separating_gap(pa,na,pb,nb,required_gap=required) > required:
            return True
        if directional_available(ka,kb):
            for direction in directions(pa,pb):
                alo,ahi=projected(ka,a,pa,direction);blo,bhi=projected(kb,b,pb,direction)
                if max(alo-bhi,blo-ahi)>clearance+1e-6:
                    return True
        return False

    def box_separated(key,cap,box,clearance):
        points,normals,margin,is_hull=shape(key,cap)
        corners=np.array(list(itertools.product(*zip(box['min_mm'],box['max_mm']))))
        required=margin+clearance+1e-6
        if is_hull and separating_gap(points,normals,corners,np.eye(3),required_gap=required) > required:
            return True
        if directional_available(key):
            for direction in directions(points,corners):
                lo,hi=projected(key,cap,points,direction)
                projection=corners@direction
                if max(lo-projection.max(),projection.min()-hi)>clearance+1e-6:
                    return True
        return False

    check_collision(capsules,config.obstacles,config.limits['clearance_mm'],
                    allow_tool_contact=tool_contact,pair_separated=pair_separated,
                    box_separated=box_separated)
