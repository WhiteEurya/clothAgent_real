"""Color-only alignment of measured overlapping top-down views."""
import numpy as np
import cv2
from scipy.optimize import lsq_linear

def fit_affine(source,target):
    gains=[];offsets=[]
    for c in range(3):
        x=source[:,c].astype(float);y=target[:,c].astype(float)
        if np.std(x)<8:raise ValueError('Insufficient color variation to estimate gain')
        A=np.c_[x,np.ones(len(x))];weight=np.ones(len(x))
        for _ in range(10):
            fit=lsq_linear(A*weight[:,None],y*weight,bounds=([.5,-80],[2.,80])).x
            residual=A@fit-y;scale=max(2.,1.4826*np.median(np.abs(residual-np.median(residual))))
            weight=np.sqrt(np.minimum(1.,1.5*scale/np.maximum(np.abs(residual),1e-6)))
        if not (.5<=fit[0]<=2 and abs(fit[1])<=80):raise ValueError(f'Color transform outside preview bounds: channel={c}, fit={fit}')
        gains.append(float(fit[0]));offsets.append(float(fit[1]))
    return np.array(gains),np.array(offsets)


def match_colors(rgb, xyz, fallback):
    valid=[np.isfinite(x).all(2) for x in xyz]
    overlap=valid[0]&valid[1]&(np.abs(xyz[0][:,:,2]-xyz[1][:,:,2])<5)
    sample=overlap.copy()
    for im in rgb:
        gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(float)
        gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)/8
        gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)/8
        sample&=(np.hypot(gx,gy)<8)&(im.min(2)>25)&(im.max(2)<245)
    sample=cv2.erode(sample.astype(np.uint8),np.ones((3,3),np.uint8))>0
    y,x=np.indices(sample.shape);train=sample&((x//24+y//24)%2==0);test=sample&~train
    info={'depth_modified':False,'fit_pixels':int(train.sum()),'validation_pixels':int(test.sum())}
    gain=np.array(fallback['gain_rgb']);offset=np.array(fallback['offset_rgb'])
    try:
        if min(train.sum(),test.sum())<300:raise ValueError('insufficient smooth overlap')
        g,o=fit_affine(rgb[1][train],rgb[0][train])
        old=np.median(np.mean(np.abs(rgb[1][test].astype(float)-rgb[0][test]),axis=1))
        new=np.median(np.mean(np.abs(np.clip(rgb[1][test]*g+o,0,255)-rgb[0][test]),axis=1))
        if new>old:raise ValueError('held-out color mismatch worsened')
        gain,offset=g,o;info.update(status='FITTED_CURRENT_CAPTURE',before=float(old),after=float(new))
    except ValueError as exc:
        info.update(status='SAVED_COLOR_FALLBACK',reason=str(exc))
    # Smooth bright table pixels alone do not validate a transform for black
    # cloth. Check all held-out overlapping colors, including dark material.
    validation=overlap&((x//24+y//24)%2!=0)
    if validation.sum()>=300:
        a=rgb[1][validation].astype(float);b=rgb[0][validation].astype(float)
        old=float(np.mean(np.abs(a-b)));new=float(np.mean(np.abs(np.clip(a*gain+offset,0,255)-b)))
        dark=(a.max(1)<65)&(b.max(1)<65)
        dark_old=float(np.mean(np.abs(a[dark]-b[dark]))) if dark.any() else None
        dark_new=float(np.mean(np.abs(np.clip(a[dark]*gain+offset,0,255)-b[dark]))) if dark.any() else None
        accepted=new<old and (dark.sum()<100 or dark_new<=dark_old+2)
        info.update(all_color_before=old,all_color_after=new,dark_pixels=int(dark.sum()),dark_before=dark_old,dark_after=dark_new)
    else:accepted=False
    if not accepted:
        info.update(rejected_status=info['status'],status='IDENTITY_UNVALIDATED_COLOR',
                    validation_reason='no demonstrated current-frame improvement across table and dark material')
        gain=np.ones(3);offset=np.zeros(3)
    info.update(gain_rgb=gain.tolist(),offset_rgb=offset.tolist())
    return gain,offset,info
