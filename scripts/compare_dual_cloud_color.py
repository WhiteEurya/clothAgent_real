"""Offline color-only alignment; preserve highest-Z geometry and unknown pixels."""
from pathlib import Path
import argparse,json
import numpy as np
import cv2
from scipy.optimize import lsq_linear
from PIL import Image,ImageDraw


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


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rgb=[np.asarray(Image.open(a.source/f'{n}_topdown.png').convert('RGB')) for n in ['zed','realsense']]
    xyz=[np.load(a.source/f'{n}_topdown_xyz_mm.npy') for n in ['zed','realsense']];valid=[np.isfinite(x).all(2) for x in xyz]
    both=valid[0]&valid[1];dz=np.abs(xyz[0][:,:,2]-xyz[1][:,:,2]);sample=both&(dz<5)
    for image in rgb:
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(float)
        gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)/8;gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)/8
        sample&=(np.hypot(gx,gy)<8)&(image.min(2)>25)&(image.max(2)<245)
    sample=cv2.erode(sample.astype(np.uint8),np.ones((3,3),np.uint8))>0
    y,x=np.indices(sample.shape);train=sample&(((x//24+y//24)%2)==0);test=sample&~train
    if min(train.sum(),test.sum())<300:raise ValueError('Insufficient consistent overlap for color fitting')
    gain,offset=fit_affine(rgb[1][train],rgb[0][train]);corrected=np.clip(np.rint(rgb[1]*gain+offset),0,255).astype(np.uint8)
    # Only replace colors from the same winning camera; no depth alteration or filling.
    use_rs=valid[1]&(~valid[0]|(xyz[1][:,:,2]>xyz[0][:,:,2]));union=valid[0]|valid[1]
    before=np.full_like(rgb[0],[24,30,39]);after=before.copy();before[valid[0]]=rgb[0][valid[0]];after[valid[0]]=rgb[0][valid[0]];before[use_rs]=rgb[1][use_rs];after[use_rs]=corrected[use_rs]
    Image.fromarray(before).save(a.output/'before.png');Image.fromarray(after).save(a.output/'after.png')
    np.save(a.output/'valid.npy',union);np.save(a.output/'source_camera.npy',np.where(use_rs,1,np.where(valid[0],0,-1)).astype(np.int8))
    Image.fromarray(corrected).save(a.output/'realsense_color_matched.png')
    canvas=Image.new('RGB',(1280,680),'#181e27');canvas.paste(Image.fromarray(before),(0,40));canvas.paste(Image.fromarray(after),(640,40));d=ImageDraw.Draw(canvas);d.text((15,12),'BEFORE | original camera colors',fill='white');d.text((655,12),'AFTER | color alignment only, identical depth and source selection',fill='white');canvas.save(a.output/'comparison.png')
    err0=np.mean(np.abs(rgb[1][test].astype(float)-rgb[0][test]),axis=1);err1=np.mean(np.abs(corrected[test].astype(float)-rgb[0][test]),axis=1)
    report={'source':str(a.source),'method':'Robust per-channel affine match RealSense to ZED on smooth overlap with absolute height difference below 5 mm; held-out spatial tiles. Highest-Z camera selection unchanged.','gain_rgb':gain.tolist(),'offset_rgb':offset.tolist(),'fit_pixels':int(train.sum()),'validation_pixels':int(test.sum()),'held_out_median_channel_absolute_difference_before':float(np.median(err0)),'held_out_median_channel_absolute_difference_after':float(np.median(err1)),'depth_modified':False,'holes_filled':False,'execution_config_modified':False,'limitation':'Display color consistency only, not colorimetric or geometric calibration. Fit belongs to this capture; auto exposure/white balance changes require re-evaluation.'}
    assert np.array_equal(before[~union],after[~union])
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
