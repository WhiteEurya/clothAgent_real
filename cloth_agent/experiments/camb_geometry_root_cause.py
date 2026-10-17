"""Offline depth-plane/frame-semantics diagnostics; never applies corrections."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from cloth_agent.experiments.apriltag_geometry_audit import csv_write

def stats(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    return dict(count=len(a),median=float(np.median(a)),p05=float(np.percentile(a,5)),p95=float(np.percentile(a,95)),
        min=float(a.min()),max=float(a.max())) if len(a) else dict(count=0)

def optical_z_in_color(depth_z,ray,R,t):
    # p_depth = R.T @ (ray_color*z_color - t), solve its third component.
    return (depth_z+(R.T@t)[2])/(np.asarray(ray)@R[:,2])

def robust_plane(points):
    A=np.c_[points[:,:2],np.ones(len(points))];z=points[:,2];keep=np.ones(len(points),bool)
    for _ in range(5):
        beta=np.linalg.lstsq(A[keep],z[keep],rcond=None)[0]
        residual=z-A@beta
        mad=np.median(np.abs(residual-np.median(residual)))
        keep=np.abs(residual-np.median(residual))<=max(1.,3*1.4826*mad)
    normal=np.array([-beta[0],-beta[1],1.]);normal/=np.linalg.norm(normal)
    return beta,normal,residual

def run(session,output):
    session=Path(session);out=Path(output)
    meta=json.loads((out/'device_metadata.json').read_text())
    data=session/'multiboard_measured_142mm'
    camera=json.loads((session/'camera.json').read_text());K=np.asarray(camera['K'])
    R=np.asarray(meta['depth_to_color']['rotation_column_major']).reshape(3,3,order='F')
    t=np.asarray(meta['depth_to_color']['translation_m'])*1000
    samples=np.load(session/'last_depth_window_m.npy');valid=np.isfinite(samples)&(samples>.1)&(samples<2.5)
    depth=np.ma.median(np.ma.array(samples,mask=~valid),axis=0).filled(np.nan)*1000
    yy,xx=np.indices(depth.shape);rays=np.stack(((xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)),axis=-1)
    rgb=cv2.imread(str(session/'last_aligned_rgb.png'));gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    rows=[];summary=[]
    for f in sorted(data.glob('*_board/analysis.json')):
        a=json.loads(f.read_text());name=f.parent.name
        mask=np.zeros(depth.shape,np.uint8)
        for tag in a['detections']:
            cv2.fillConvexPoly(mask,np.rint(tag['corners']).astype(np.int32),1)
        mask=cv2.erode(mask,np.ones((7,7),np.uint8))
        selected=mask.astype(bool)&np.isfinite(depth)&(valid.mean(axis=0)>=.8)
        zc=optical_z_in_color(depth[selected],rays[selected],R,t)
        points=rays[selected]*zc[:,None]
        beta,n,resid=robust_plane(points)
        pn=np.asarray(a['T_camera_board_m'])[:3,2]
        tilt=float(np.degrees(np.arccos(np.clip(abs(n@pn),0,1))))
        light=gray[selected]
        before=[];after=[];deltas=[];temporal=[]
        for row in a['rows']:
            if row['depth_status']!='VALID' or not row['local_stable']:continue
            u,v=row['sample_u'],row['sample_v'];zd=row['depth_camera_z_mm']
            corrected=float(optical_z_in_color(zd,rays[v,u],R,t))
            dz=corrected-row['board_camera_z_mm']
            window=samples[:,v-2:v+3,u-2:u+3]*1000
            m=valid[:,v-2:v+3,u-2:u+3]
            perframe=np.ma.median(np.ma.array(window.reshape(len(samples),-1),mask=~m.reshape(len(samples),-1)),axis=1).filled(np.nan)
            jitter=float(np.nanstd(perframe))
            rowout=dict(board=name,tag_id=row['tag_id'],u=row['u'],v=row['v'],
                raw_depth_z_mm=zd,estimated_color_optical_z_mm=corrected,
                frame_semantics_delta_mm=corrected-zd,
                original_depth_minus_pnp_mm=row['dz_mm'],color_z_minus_pnp_mm=dz,
                frame_to_frame_depth_std_mm=jitter)
            rows.append(rowout);before.append(row['dz_mm']);after.append(dz);deltas.append(corrected-zd);temporal.append(jitter)
        black=resid[light<100];white=resid[light>180]
        summary.append(dict(board=name,points_used=len(points),
            tag_center_depth_jitter_mm=stats(temporal),
            frame_semantics_delta_mm=stats(deltas),original_depth_minus_pnp_mm=stats(before),
            color_z_minus_pnp_mm=stats(after),
            plane_residual_mm=stats(resid),plane_abs_residual_mm=stats(np.abs(resid)),
            black_plane_residual_mm=stats(black),white_plane_residual_mm=stats(white),
            black_minus_white_residual_median_mm=float(np.median(black)-np.median(white)) if len(black) and len(white) else None,
            pnp_vs_depth_normal_angle_deg=tilt,depth_plane_coefficients=beta.tolist(),
            theoretical_width_precision_mm_per_1mm_z=142/float(np.median(points[:,2]))))
    csv_write(out/'tag_diagnostics.csv',rows,['board','tag_id'])
    (out/'depth_diagnostics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('session');p.add_argument('output');args=p.parse_args();run(args.session,args.output)
