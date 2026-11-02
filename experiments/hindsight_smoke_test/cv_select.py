from pathlib import Path
import hashlib,json
import cv2,numpy as np
from transport import read,save,sha
SEED='smoke-v0-20261029'
def stable(x):return hashlib.sha256((SEED+x).encode()).hexdigest()
def run_id(r):
 p=Path(r['source']).parts
 return '/'.join(p[:3] if p[0]=='data' else p[:2])
def mask(im):
 h,w=im.shape[:2];hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
 m=np.uint8((hsv[:,:,1]>35)&(hsv[:,:,2]>80))*255
 roi=np.zeros_like(m);roi[int(.03*h):int(.97*h),int(.15*w):int(.85*w)]=255;m=cv2.bitwise_and(m,roi)
 m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8));m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
 c=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
 if not c:raise ValueError('no foreground')
 c=max(c,key=cv2.contourArea);m[:]=0;cv2.drawContours(m,[c],-1,255,-1);frac=float((m>0).mean())
 if not .01<frac<.65:raise ValueError('foreground area outside quality range')
 x,y,bw,bh=cv2.boundingRect(c);area=cv2.contourArea(c);hull=cv2.contourArea(cv2.convexHull(c));ys,xs=np.where(m>0);pts=np.column_stack((xs,ys));cov=np.cov(pts.T);val,vec=np.linalg.eigh(cov);axis=vec[:,-1];ang=float(np.degrees(np.arctan2(axis[1],axis[0]))%180)
 return m,{'fraction':frac,'aspect':bw/max(1,bh),'solidity':area/max(1,hull),'orientation_deg':ang,'shape_bucket':f'{int(ang//45)}_{int(bw>bh)}_{int(area/max(1,hull)>.8)}'}
def measure(r):
 a=cv2.imread(r['before']);b=cv2.imread(r['after'])
 if a is None or b is None or a.shape!=b.shape:raise ValueError('bad image or shape mismatch')
 a=cv2.resize(a,(480,270));b=cv2.resize(b,(480,270));ma,shape=mask(a);mb,_=mask(b)
 orb=cv2.ORB_create(nfeatures=600);ka,da=orb.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),ma);kb,db=orb.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),mb)
 M=None;quality='CENTROID_FALLBACK';inliers=0
 if da is not None and db is not None and len(db)>1:
  matches=[p[0] for p in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2) if len(p)==2 and p[0].distance<.75*p[1].distance]
  if len(matches)>=4:
   x=np.float64([ka[m.queryIdx].pt for m in matches]);y=np.float64([kb[m.trainIdx].pt for m in matches]);est,ins=cv2.estimateAffinePartial2D(x,y,method=cv2.RANSAC,ransacReprojThreshold=3)
   if est is not None and ins.sum()>=4 and .8<np.linalg.norm(est[:,0])<1.25:
    xx=x[ins[:,0]>0];yy=y[ins[:,0]>0];xc=xx.mean(0);yc=yy.mean(0);u,_,v=np.linalg.svd((xx-xc).T@(yy-yc));rot=v.T@u.T
    if np.linalg.det(rot)<0:v[-1]*=-1;rot=v.T@u.T
    M=np.column_stack((rot,yc-rot@xc));quality='RIGID_FEATURE_FIT';inliers=int(ins.sum())
 if M is None:
  ya,xa=np.where(ma>0);yb,xb=np.where(mb>0);M=np.array([[1.,0.,xb.mean()-xa.mean()],[0.,1.,yb.mean()-ya.mean()]])
 warped=cv2.warpAffine(ma,M,(480,270),flags=cv2.INTER_NEAREST)>0;cur=mb>0;union=np.logical_or(warped,cur).sum();iou=float(np.logical_and(warped,cur).sum()/max(1,union));new=float(np.logical_and(cur,~warped).sum()/max(1,cur.sum()))
 return {'id':r['id'],'run':run_id(r),'cv_status':'OK','alignment_status':quality,'inliers':inliers,'aligned_iou':iou,'new_projected_fraction':new,'score':1-iou,'shape':shape,'before_to_after_small_image':M.tolist(),'semantic_label':None}

def select(root,index):
 if (root/'selected_pairs.json').exists():return read(root/'selected_pairs.json'),read(root/'validation_inputs.json')
 runs=sorted({run_id(r) for r in index},key=stable);heldout=set(runs[:4]);val=[]
 for run in runs[:4]:
  r=min([r for r in index if run_id(r)==run],key=lambda r:stable(r['id']))
  val.append({'id':r['id'],'run':run,'before':r['before'],'before_sha256':sha(r['before'])})
 save(root/'validation_inputs.json',val)
 metrics=[]
 for i,r in enumerate(index):
  try:x=measure(r)
  except Exception as e:x={'id':r['id'],'run':run_id(r),'cv_status':'UNMEASURABLE','error':str(e)}
  metrics.append(x)
  if (i+1)%100==0:print('CV',i+1,'/',len(index),flush=True)
 save(root/'cv_metrics.json',metrics)
 pool=[x for x in metrics if x['cv_status']=='OK' and x['run'] not in heldout];pool.sort(key=lambda x:(x['score'],stable(x['id'])))
 thirds=np.array_split(np.array(pool,dtype=object),3);selected=[];usedruns=set();shapes=set()
 def pick(items,n,tag):
  for _ in range(n):
   available=[x for x in items if x['run'] not in usedruns and all(x['id']!=y['id'] for y in selected)]
   if not available:break
   z=min(available,key=lambda x:(x['shape']['shape_bucket'] in shapes,stable(x['id'])));selected.append({**z,'selection_stratum':tag});usedruns.add(z['run']);shapes.add(z['shape']['shape_bucket'])
 # 4 high, 3 medium, 3 low; deterministic random controls from remaining.
 for items,n,tag in [(thirds[2].tolist(),4,'HIGH_DEFORMATION'),(thirds[1].tolist(),3,'MEDIUM_DEFORMATION'),(thirds[0].tolist(),3,'LOW_DEFORMATION')]:pick(items,n,tag)
 controls=[x for x in pool if x['run'] not in usedruns];controls.sort(key=lambda x:stable('control'+x['id']))
 for z in controls:
  if len(selected)>=12:break
  if z['run'] in usedruns:continue
  selected.append({**z,'selection_stratum':'RANDOM_CONTROL'});usedruns.add(z['run'])
 save(root/'selected_pairs.json',selected);return selected,val
