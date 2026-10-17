"""RGB vs IR PnP and stereo triangulation, no calibration updates."""
import json,argparse
from pathlib import Path
import cv2,numpy as np
from pupil_apriltags import Detector
from cloth_agent.experiments.apriltag_multiboard_audit import measured_board
from cloth_agent.experiments.apriltag_geometry_audit import fit_pose,project_rgb,transform_points,csv_write
from cloth_agent.experiments.camb_geometry_root_cause import stats

def transform(m,label):
    e=m['extrinsics']['depth_to_'+label];t=np.eye(4)
    t[:3,:3]=np.array(e['rotation_column_major']).reshape(3,3,order='F');t[:3,3]=e['translation_m']
    return t

def run(on,off,out):
    on=Path(on);off=Path(off);out=Path(out);m=json.loads((off/'metadata.json').read_text())
    detector=Detector(families='tag36h11',quad_decimate=1,nthreads=2)
    alltags={}
    for label in ('rgb','ir1','ir2'):
        alltags[label]={int(t.tag_id):t for t in detector.detect(cv2.imread(str(off/f'{label}_14.png'),0)) if t.hamming==0 and t.decision_margin>=20}
    common=sorted(set.intersection(*[set(t) for t in alltags.values()]))
    if len(common)<4:raise ValueError('Insufficient shared tags')
    board,_=measured_board(142.)
    obj=np.concatenate([board['corners'][i] for i in common])
    poses={};summary={}
    for label,tags in alltags.items():
        pix=np.concatenate([tags[i].corners for i in common])
        K=np.array(m['profiles'][label]['K']);dist=np.array(m['profiles'][label]['coeffs'])
        if np.any(dist):raise ValueError('Nonzero distortion needs explicit model handling')
        pose,amb=fit_pose(obj,pix,K,dist)
        err=np.linalg.norm(project_rgb(pose,obj,K,dist)-pix,axis=1)
        depth_pose=np.linalg.inv(transform(m,label))@pose
        poses[label]=depth_pose
        summary[label]=dict(tags=len(tags),shared=len(common),pnp_ambiguous=bool(amb),
            reprojection_rms_px=float(np.sqrt(np.mean(err**2))),board_origin_in_depth_camera_m=depth_pose[:3,3].tolist(),
            T_depth_board=depth_pose.tolist())

    # Check RGB consistency across the emitter-on and off acquisitions.
    old={int(t.tag_id):t for t in detector.detect(cv2.imread(str(on/'rgb_14.png'),0))}
    movement=np.concatenate([alltags['rgb'][i].corners-old[i].corners for i in common if i in old])
    motion=float(np.sqrt(np.mean(movement**2)))
    if motion>1.:raise ValueError(f'Board moved between captures: {motion} px')
    depth_u16=np.load(on/'native_depth_u16.npy');onmeta=json.loads((on/'metadata.json').read_text())
    depths=depth_u16.astype(float)*onmeta['depth_scale']*1000
    valid=(depths>100)&(depths<2500)
    depth=np.ma.median(np.ma.array(depths,mask=~valid),axis=0).filled(np.nan)
    Kd=np.array(onmeta['profiles']['depth']['K']);rows=[]
    for i in common:
        center=board['corners'][i].mean(axis=0)
        pir=transform_points(poses['ir1'],[center])[0]
        uv=(Kd@pir)[:2]/pir[2];u,v=np.rint(uv).astype(int)
        patch=depth[v-2:v+3,u-2:u+3];values=patch[np.isfinite(patch)]
        if len(values)<18:continue
        z=float(np.median(values))
        row=dict(tag_id=i,depth_u=u,depth_v=v,raw_native_depth_mm=z)
        for label,p in poses.items():
            point=transform_points(p,[center])[0]*1000
            row[label+'_pnp_z_mm']=float(point[2]);row['depth_minus_'+label+'_pnp_mm']=float(z-point[2])
        rows.append(row)

    # Stereo triangulation in depth coordinates, factory IR intrinsics/extrinsics.
    projections=[];pixels=[]
    for label in ('ir1','ir2'):
        projections.append(np.array(m['profiles'][label]['K'])@transform(m,label)[:3])
        pixels.append(np.concatenate([alltags[label][i].corners for i in common]))
    homogeneous=cv2.triangulatePoints(*projections,pixels[0].T,pixels[1].T)
    stereo=(homogeneous[:3]/homogeneous[3:]).T
    rgbpred=project_rgb(transform(m,'rgb'),stereo,np.array(m['profiles']['rgb']['K']),np.zeros(5))
    rgbactual=np.concatenate([alltags['rgb'][i].corners for i in common])
    rgbdelta=rgbpred-rgbactual
    lengths=[]
    for i in range(len(common)):
        corners=stereo[i*4:i*4+4]
        lengths.extend(np.linalg.norm(corners-np.roll(corners,-1,axis=0),axis=1)*1000)
    summary['stereo']=dict(tag_edge_length_mm=stats(lengths),expected_edge_mm=board['tag_size_mm'],
        rgb_reprojection_error_px=stats(np.linalg.norm(rgbdelta,axis=1)),
        rgb_signed_du_px=stats(rgbdelta[:,0]),rgb_signed_dv_px=stats(rgbdelta[:,1]),
        note='IR triangulation uses factory stereo calibration, not independent physical truth; corner depth noise affects edge lengths')
    # Independent direct native-depth -> RGB correspondence check (no PnP/board size).
    ir1_profile=np.asarray(m['profiles']['ir1']['K'])
    depth_ir1=transform(m,'ir1')
    direct=[];raw_points=[]
    for tag_id in common:
        for index,uv_ir in enumerate(alltags['ir1'][tag_id].corners):
            # IR1/depth have the same native calibration on this device.
            if not np.allclose(depth_ir1,np.eye(4),atol=1e-8) or not np.allclose(ir1_profile,Kd):
                raise ValueError('Direct IR1/depth indexing not valid on this profile')
            u,v=np.rint(uv_ir).astype(int);patch=depth[v-2:v+3,u-2:u+3]
            vals=patch[np.isfinite(patch)]
            if len(vals)<18:continue
            z=float(np.median(vals))/1000
            ray=np.linalg.solve(Kd,[*uv_ir,1.])
            pd=ray*z
            actual=alltags['rgb'][tag_id].corners[index]
            expected=project_rgb(transform(m,'rgb'),[pd],np.asarray(m['profiles']['rgb']['K']),np.zeros(5))[0]
            errors=[]
            for offset in (-.030,.030):
                altered=project_rgb(transform(m,'rgb'),[ray*(z+offset)],np.asarray(m['profiles']['rgb']['K']),np.zeros(5))[0]
                errors.append(float(np.linalg.norm(altered-expected)))
            direct.append(dict(tag_id=tag_id,corner=index,ir_u=float(uv_ir[0]),ir_v=float(uv_ir[1]),
                rgb_observed_u=float(actual[0]),rgb_observed_v=float(actual[1]),
                rgb_predicted_u=float(expected[0]),rgb_predicted_v=float(expected[1]),
                du_px=float(expected[0]-actual[0]),dv_px=float(expected[1]-actual[1]),
                error_px=float(np.linalg.norm(expected-actual)),max_shift_for_30mm_depth_error_px=max(errors)))
    csv_write(out/'direct_native_depth_projection.csv',direct,['tag_id','corner'])
    summary['direct_native_depth_to_rgb']=dict(count=len(direct),
        error_px=stats([r['error_px'] for r in direct]),
        du_px=stats([r['du_px'] for r in direct]),dv_px=stats([r['dv_px'] for r in direct]),
        projection_change_for_depth_plus_minus_30mm_px=stats([r['max_shift_for_30mm_depth_error_px'] for r in direct]),
        method='IR1 observed corner + native emitter-on depth -> factory depth_to_RGB -> observed RGB corner; no board geometry or PnP')
    summary['raw_depth_comparison']={label:stats([r['depth_minus_'+label+'_pnp_mm'] for r in rows]) for label in poses}
    summary['motion_between_captures_rms_px']=motion;summary['shared_tag_ids']=common
    summary['assumptions']='142 mm outer span, original gap/tag ratio, square flat board; factory intrinsics/extrinsics; no calibration writes'
    (out/'cross_stream.json').write_text(json.dumps(summary,indent=2))
    csv_write(out/'cross_stream_tags.csv',rows,['tag_id'])
    canvas=cv2.imread(str(off/'rgb_14.png'))
    for a,b in zip(rgbactual,rgbpred):
        cv2.circle(canvas,tuple(np.rint(a).astype(int)),3,(0,255,0),1)
        cv2.drawMarker(canvas,tuple(np.rint(b).astype(int)),(0,0,255),cv2.MARKER_CROSS,7,1)
    cv2.putText(canvas,'Green RGB observed; red IR-stereo projected (actual pixel scale)',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
    cv2.imwrite(str(out/'rgb_vs_ir_projection.png'),canvas)
    print(json.dumps(summary,indent=2))
    return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('on');p.add_argument('off');p.add_argument('output');a=p.parse_args();run(a.on,a.off,a.output)
