"""Deterministic target-contour chord reorientation; no hardware or model calls."""
import cv2
import numpy as np
from cloth_agent.contour_normal_grasp import NormalGraspError, ray_intersections

NAME='garment-chord-reorient'

def cross(a,b):return float(a[0]*b[1]-a[1]*b[0])

def contour_crossings(a,b):
    hits=[];q=b;s=np.roll(b,-1,axis=0)-b
    for p,end in zip(a,np.roll(a,-1,axis=0)):
        r=end-p;den=r[0]*s[:,1]-r[1]*s[:,0];good=abs(den)>1e-9;v=q-p
        t=np.full(len(b),np.inf);u=t.copy()
        t[good]=(v[good,0]*s[good,1]-v[good,1]*s[good,0])/den[good]
        u[good]=(v[good,0]*r[1]-v[good,1]*r[0])/den[good]
        for hit in p+t[(t>=0)&(t<=1)&(u>=0)&(u<=1),None]*r:
            if not any(np.linalg.norm(hit-old)<1e-5 for old in hits):hits.append(hit)
    return sorted(hits,key=lambda p:(p[1],p[0]))

def nearest(point,polygon):
    edge=np.roll(polygon,-1,axis=0)-polygon
    u=np.clip(np.sum((point-polygon)*edge,axis=1)/np.maximum(np.sum(edge*edge,axis=1),1e-12),0,1)
    q=polygon+u[:,None]*edge;idx=np.argmin(np.linalg.norm(q-point,axis=1))
    return q[idx],float(np.linalg.norm(q[idx]-point))

def geometry(mask,goal_polygon,mm_per_pixel,pull_mm=50.,inset_px=8.):
    mask=np.asarray(mask);goal=np.asarray(goal_polygon,float)
    if mask.ndim!=2 or not np.isfinite(mask).all() or goal.ndim!=2 or goal.shape[1]!=2 or len(goal)<3 or not np.isfinite(goal).all():
        raise NormalGraspError('invalid current mask or goal contour')
    if not np.isfinite([mm_per_pixel,pull_mm,inset_px]).all() or mm_per_pixel<=0 or not 10<=pull_mm<=150 or inset_px<3:
        raise NormalGraspError('invalid scale/inset or pull outside existing 10–150 mm range')
    mask=(mask>0).astype('uint8');count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if count<2:raise NormalGraspError('no garment component')
    areas=stats[1:,cv2.CC_STAT_AREA].tolist();mask=labels==1+int(np.argmax(areas))
    if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():raise NormalGraspError('garment contour clipped by image boundary')
    contour=max(cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0],key=cv2.contourArea)[:,0].astype(float)
    hits=contour_crossings(contour,goal)
    if len(hits)<2:raise NormalGraspError('reorientation needs at least two current/goal contour intersections')
    A,B=hits[0],hits[-1];M=(A+B)/2;length=np.linalg.norm(B-A)
    if length<1e-6:raise NormalGraspError('coincident chord endpoints')
    t=(B-A)/length;outside=[];polygon=goal.astype('float32').reshape(-1,1,2)
    for p in contour:
        if cv2.pointPolygonTest(polygon,tuple(p),False)<0:
            q,d=nearest(p,goal);outside.append((d,p,q))
    if not outside:raise NormalGraspError('no protruding contour region')
    dist,C,Q=max(outside,key=lambda v:v[0]);n=np.array([-t[1],t[0]])
    side=np.dot(n,C-M)
    if abs(side)<1e-6:raise NormalGraspError('protrusion side of chord is ambiguous')
    if side>0:n=-n
    ray_hits=ray_intersections(contour,M,n)
    if not ray_hits:raise NormalGraspError('no opposite contour intersection')
    E=M+ray_hits[-1]*n;G=np.rint(E-inset_px*n).astype(int);height,width=mask.shape
    if not (0<=G[0]<width and 0<=G[1]<height) or not mask[G[1],G[0]]:raise NormalGraspError('inset grasp not on visible cloth')
    y,x=np.nonzero(mask);O=np.array([x.mean(),y.mean()]);world=lambda p:np.asarray(p)*mm_per_pixel*[1,-1]
    a=cross(world(C-O),world(Q-C));b=cross(world(E-O),t*[1,-1]);bg=cross(world(G-O),t*[1,-1])
    if abs(a)<1e-6 or abs(b)<1e-6 or abs(bg)<1e-6 or b*bg<=0:raise NormalGraspError('reorientation torque sign ambiguous at grasp')
    direction=t if a*b>0 else -t;target=np.rint(G+pull_mm/mm_per_pixel*direction).astype(int)
    if not (3<=target[0]<width-3 and 3<=target[1]<height-3):raise NormalGraspError('target outside image interior')
    return dict(skill=NAME,grasp_px=G.tolist(),target_px=target.tolist(),A_px=A.tolist(),B_px=B.tolist(),M_px=M.tolist(),C_px=C.tolist(),Q_px=Q.tolist(),E_px=E.tolist(),O_px=O.tolist(),all_intersections_px=[p.tolist() for p in hits],normal_unit_xy=n.tolist(),pull_unit_xy=direction.tolist(),rotation='CCW' if a>0 else 'CW',a_mm2=a,b_mm=b,b_inset_mm=bg,requested_pull_mm=pull_mm,component_areas_px=areas,rule='topmost/bottommost crossings; chord midpoint; farthest opposite intersection; parallel pull with cross-product sign')
