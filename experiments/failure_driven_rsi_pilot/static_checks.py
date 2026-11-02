"""Reused pure numeric checks, supplemented with recorded Cartesian bounds."""
from common import *

def compare(a,b):
    if a['decision']!='ACTION' or b['decision']!='ACTION':return {'decision_changed':a['decision']!=b['decision'],'meaningful':a['decision']!=b['decision']}
    pa=float(np.linalg.norm(np.array(a['Pa'])-b['Pa']));pb=float(np.linalg.norm(np.array(a['Pb'])-b['Pb']))
    va=np.array(a['Pb'])-a['Pa'];vb=np.array(b['Pb'])-b['Pa']
    angle=float(np.degrees(np.arccos(np.clip(np.dot(va,vb)/(np.linalg.norm(va)*np.linalg.norm(vb)),-1,1)))) if np.linalg.norm(va)*np.linalg.norm(vb)>0 else None
    return {'Pa_delta_px':pa,'Pb_delta_px':pb,'direction_delta_deg':angle,'meaningful':pa>15 or pb>15 or (angle is not None and angle>15)}

def check(c,plan):
    r=ROOT/c['round'];im=np.asarray(Image.open(OUT/'cases'/c['id']/'before.png').convert('RGB')).astype(float);h,w=im.shape[:2]
    checks={};values={}
    if plan['decision']!='ACTION':return {'status':'NO_ACTION','checks':{'null_points':plan['Pa'] is None and plan['Pb'] is None},'limitation':'No proposed physical motion'}
    a,b=plan['Pa'],plan['Pb']
    checks['image_bounds']=all(isinstance(p,list) and len(p)==2 and all(isinstance(v,int) for v in p) and 0<=p[0]<w and 0<=p[1]<h for p in [a,b])
    checks['primitive']=plan['motion_type']=='lift_pull_place'
    if not checks['image_bounds']:return {'status':'FAIL','checks':checks}
    checks['nonzero']=a!=b
    x,y=a;rgb=im[max(0,y-3):min(h,y+4),max(0,x-3):min(w,x+4)]
    cloth=(rgb[:,:,0]>rgb[:,:,1]+10)&(rgb[:,:,2]>rgb[:,:,1]+15)&(rgb[:,:,0]>70)
    values['grasp_purple_mask_fraction_7x7']=float(cloth.mean());checks['grasp_on_cloth_coarse']=bool(cloth.mean()>=.5)
    cloud=np.load(r/'before/world_xyz_mm.npy')
    coords=[]
    for x,y in [a,b]:
        p=cloud[max(0,y-2):min(h,y+3),max(0,x-2):min(w,x+3)].reshape(-1,3)
        good=np.isfinite(p).all(axis=1)&(np.linalg.norm(p,axis=1)>10)
        coords.append(np.median(p[good],axis=0) if good.sum()>=5 else None)
    checks['grasp_measured_depth']=coords[0] is not None
    values['target_mapping']='median recorded depth 5x5'
    if coords[1] is None:
        td=read(r/'target_depth.json') if (r/'target_depth.json').exists() else {}
        coef=td.get('plane_z_equals_ax_by_c')
        if coef is not None:
            meta=read(r/'before/observation.json');T=np.asarray(meta['world_from_camera_mm'])
            ray=T[:3,:3]@np.array([(b[0]-624.7428588867188)/910.2265014648438,(b[1]-371.5201416015625)/908.62109375,1.])
            origin=T[:3,3];normal=np.array([-coef[0],-coef[1],1.]);denom=normal@ray
            if abs(denom)>1e-9:
                distance=(coef[2]-normal@origin)/denom
                if distance>0:coords[1]=origin+distance*ray;values['target_mapping']='recorded support-plane ray intersection (no live Host call)'
    checks['target_depth_or_recorded_plane']=coords[1] is not None
    if all(p is not None for p in coords):
        dist=float(np.linalg.norm(coords[0][:2]-coords[1][:2]));values['translation_xy_mm']=dist
        checks['translation_10_to_150_mm']=10<=dist<=150
        values['world_points_mm']=[p.tolist() for p in coords]
    else:checks['translation_10_to_150_mm']=False
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'values':values,'limitation':'Coarse offline screen only. No live robot, IK, collision, reachability or production safety certification. Color mask is garment-specific and may reject edge grasps.'}
