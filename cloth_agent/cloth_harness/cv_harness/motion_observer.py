"""Adjacent-frame textured-cloth rigid motion hypothesis, with explicit abstention."""
import cv2
import numpy as np

def fit_rigid_matches(a,b,inliers,mask):
    a,b=np.asarray(a,float),np.asarray(b,float)
    keep=np.asarray(inliers).ravel().astype(bool);a,b=a[keep],b[keep]
    if len(a)<12:return {'status':'UNKNOWN','reason':'fewer than 12 inlier matches'}
    hull=cv2.convexHull(a.astype('float32'))
    support=float(cv2.contourArea(hull))/max(1,int(mask.sum()))
    if support<.12:return {'status':'UNKNOWN','reason':'matches cover too little of garment','support_ratio':support}
    aa,bb=a-a.mean(0),b-b.mean(0)
    u,_,vt=np.linalg.svd(aa.T@bb);r=vt.T@u.T
    if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
    t=b.mean(0)-r@a.mean(0);res=np.linalg.norm(a@r.T+t-b,axis=1)
    if np.percentile(res,90)>4:return {'status':'UNKNOWN','reason':'rigid residual too large','residual_p90_px':float(np.percentile(res,90))}
    return {'status':'OK','matrix':np.column_stack((r,t)).tolist(),'inliers':len(a),'support_ratio':support,
            'residual_p90_px':float(np.percentile(res,90)),'rotation_deg':float(np.degrees(np.arctan2(r[1,0],r[0,0])))}

def register(before,after,mb,ma):
    cv2.setRNGSeed(731)
    orb=cv2.ORB_create(nfeatures=1800)
    k1,d1=orb.detectAndCompute(cv2.cvtColor(before,cv2.COLOR_RGB2GRAY),mb.astype('uint8')*255)
    k2,d2=orb.detectAndCompute(cv2.cvtColor(after,cv2.COLOR_RGB2GRAY),ma.astype('uint8')*255)
    if d1 is None or d2 is None:return {'status':'UNKNOWN','reason':'no reliable texture descriptors'}
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    forward=matcher.knnMatch(d1,d2,k=2);reverse=matcher.knnMatch(d2,d1,k=2)
    back={v[0].queryIdx:v[0].trainIdx for v in reverse if len(v)==2 and v[0].distance<.75*v[1].distance}
    matches=[v[0] for v in forward if len(v)==2 and v[0].distance<.75*v[1].distance and back.get(v[0].trainIdx)==v[0].queryIdx]
    if len(matches)<12:return {'status':'UNKNOWN','reason':'insufficient mutual matches','matches':len(matches)}
    a=np.float32([k1[m.queryIdx].pt for m in matches]);b=np.float32([k2[m.trainIdx].pt for m in matches])
    aff,keep=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=2000,confidence=.99)
    if aff is None or keep is None:return {'status':'UNKNOWN','reason':'RANSAC failed'}
    scale=float(np.linalg.norm(aff[:,0]));fraction=float(keep.mean())
    if abs(scale-1)>.03 or fraction<.6:return {'status':'UNKNOWN','reason':'scale change or weak consensus','scale':scale,'inlier_ratio':fraction}
    result=fit_rigid_matches(a,b,keep,mb);result.update(matches=len(matches),inlier_ratio=fraction,diagnostic_scale=scale)
    if result['status']!='OK':return result
    transform=np.array(result['matrix']);h,w=ma.shape
    warped=cv2.warpAffine(mb.astype('uint8'),transform,(w,h),flags=cv2.INTER_NEAREST).astype(bool)
    if warped.sum()<.98*mb.sum():return {'status':'UNKNOWN','reason':'alignment moves mask out of field of view'}
    union=np.logical_or(warped,ma).sum();iou=float(np.logical_and(warped,ma).sum()/max(1,union))
    y,x=np.where(mb);center=np.array([x.mean(),y.mean()]);delta=transform[:,:2]@center+transform[:,2]-center
    result.update(aligned_iou=iou,shape_change=1-iou,new_area_ratio=float((ma&~warped).sum()/max(1,mb.sum())),
                  lost_area_ratio=float((warped&~ma).sum()/max(1,mb.sum())),centroid_motion_px=delta.tolist(),translation_px=float(np.linalg.norm(delta)),translation_mm=None)
    return result
