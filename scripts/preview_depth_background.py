"""Offline native-depth empty-table subtraction and top-down boundary preview."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from PIL import Image,ImageDraw
from cloth_agent.depth_background import fit_background,subtract_background
from cloth_agent.dual_cloud_observation import scale_world
from cloth_agent.depth_stability import table_depth_mask
from cloth_agent.cloth_harness.cv_harness.segmentation import external_contour_band


def run(background,capture,output):
    background=Path(background);capture=Path(capture);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    bm=json.loads((background/'report.json').read_text());cm=json.loads((capture/'report.json').read_text())
    bc=json.loads((background/'calibration_used.json').read_text());cal=json.loads((capture/'calibration_used.json').read_text())
    for key in ['zed_serial','realsense_serial','world_from_zed_mm','world_from_realsense_color_mm']:
        if bc[key]!=cal[key]:raise ValueError('background camera/calibration mismatch: '+key)
    for key in ['zed_resolution','realsense_depth_size_wh','depth_mode']:
        if bm[key]!=cm[key]:raise ValueError('background capture settings mismatch: '+key)
    cfg=json.loads(Path('config/fusion.unfold_dual.json').read_text())
    roi=json.loads(Path('config/topdown.unfold_zed.json').read_text())
    center=np.array(cm['topdown_center_mm']);span=cm['topdown_span_mm']
    zT=np.array(cal['world_from_zed_mm']);zK=np.array(cm['zed_K'])
    table=table_depth_mask(np.load(capture/'zed_depth_m.npy').shape,roi['table_polygon_px'])
    reports={};masks=[];knowns=[]
    for name in ['zed','realsense']:
        stack=np.load(background/(name+'_depth_frames_m.npy'))
        # Freeze using the first 20 frames; remaining frames are empty-table validation.
        if len(stack)<30:raise ValueError('need 30 frames: 20 background + 10 held-out')
        model=fit_background(stack[:20]);np.savez_compressed(output/(name+'_background.npz'),**model)
        depth=np.load(capture/('zed_depth_m.npy' if name=='zed' else 'realsense_native_depth_m.npy'))
        foreground,known,delta,threshold=subtract_background(depth,model)
        np.save(output/(name+'_delta_mm.npy'),delta)
        np.save(output/(name+'_threshold_mm.npy'),threshold)
        Image.fromarray(foreground.astype('uint8')*255).save(output/(name+'_native_mask.png'))
        xyz=np.load(capture/(name+'_world_xyz_mm.npy')).reshape(*depth.shape,3)
        T=zT if name=='zed' else np.array(cal['world_from_realsense_color_mm'])@np.array(cm['realsense_depth_to_color_mm'])
        xyz=scale_world(xyz,T,cfg['depth_scale'][name]);finite=np.isfinite(xyz).all(2)
        yy,xx=np.nonzero(finite);points=xyz[yy,xx]
        camera=(points-zT[:3,3])@zT[:3,:3];uv=camera@zK.T
        uv=np.rint(uv[:,:2]/uv[:,2:]).astype(int)
        inside=(camera[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<table.shape[1])&(uv[:,1]>=0)&(uv[:,1]<table.shape[0])
        ids=np.flatnonzero(inside);inside[ids]&=table[uv[ids,1],uv[ids,0]]
        inside&=(points[:,2]>=-20)&(points[:,2]<=400)
        yy,xx=yy[inside],xx[inside];points=points[inside]
        px=np.rint(320+(points[:,0]-center[0])*520/span).astype(int)
        py=np.rint(320-(points[:,1]-center[1])*520/span).astype(int)
        fits=(px>=0)&(px<640)&(py>=0)&(py<640)
        px,py=px[fits],py[fits];yy,xx=yy[fits],xx[fits]
        projected=np.zeros((640,640),np.uint8);support=np.zeros_like(projected)
        fg=foreground[yy,xx];kn=known[yy,xx]
        projected[py[fg],px[fg]]=1;support[py[kn],px[kn]]=1
        masks.append(projected);knowns.append(support)
        holdout=[]
        for frame in stack[20:]:
            f,k,_,_=subtract_background(frame,model)
            # Same observed table rays; report no-data separately, not as background.
            good=k[yy,xx];holdout.append(float(f[yy,xx][good].mean()) if good.any() else None)
        reports[name]=dict(background_supported_fraction=float(model['supported'].mean()),
            native_foreground_pixels=int(foreground.sum()),table_foreground_samples=int(fg.sum()),
            empty_table_heldout_false_positive_fraction=holdout,
            median_threshold_mm=float(np.nanmedian(threshold[yy,xx])),
            known_table_samples=int(kn.sum()),table_samples=len(kn))
    # Preserve each source mask. Union is a preview, not proof of per-pixel agreement.
    merged=masks[0]|masks[1]
    closed=cv2.morphologyEx(merged,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(closed,8)
    final=np.zeros_like(merged)
    if n>1:
        largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        if stats[largest,cv2.CC_STAT_AREA]>=100:final[labels==largest]=1
    Image.fromarray(final*255).save(output/'mask.png')
    rgb=np.array(Image.open(capture/'appearance/fused_rgb.png'))
    sheet=Image.new('RGB',(1920,670),'white');draw=ImageDraw.Draw(sheet)
    for i,(name,mask) in enumerate(zip(['ZED depth difference','RealSense depth difference','Fused depth foreground'],[*masks,final])):
        overlay=rgb.copy();overlay[external_contour_band(mask,thickness=1)>0]=[0,255,70]
        Image.fromarray(overlay).save(output/(['zed','realsense','fused'][i]+'_overlay.png'))
        sheet.paste(Image.fromarray(overlay),(640*i,30));draw.text((640*i+10,8),name,fill='black')
    sheet.save(output/'comparison.png')
    reports.update(background=str(background),capture=str(capture),depth_modified=False,
        method='native optical depth: background median - current > 5 mm + 3 sigma(MAD)',
        mask_processing='3x3 close, largest component >=100 output pixels; appearance only',
        background_frames=20,heldout_frames=len(stack)-20,robot_connections=0,default_changed=False)
    (output/'report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps(reports),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--background',required=True);p.add_argument('--capture',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.background,a.capture,a.output)
