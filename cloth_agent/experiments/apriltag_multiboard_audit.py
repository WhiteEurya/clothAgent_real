"""Offline camera-frame audit of three separated boards; no device/control calls."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image
from pupil_apriltags import Detector
from cloth_agent.experiments.apriltag_geometry_audit import (
    board_config, DEFAULT_BOARD,A4_SCALE,analyze,save_outputs,csv_write,distribution)
from cloth_agent.dual_arm.observation import Observation

def measured_board(outer_width_mm=None):
    original=board_config(DEFAULT_BOARD,1.)
    points=np.concatenate(list(original['corners'].values()))
    span=float(np.ptp(points[:,0])*1000)
    if outer_width_mm is not None and (not np.isfinite(outer_width_mm) or outer_width_mm<=0):
        raise ValueError('Measured black-edge outer width must be positive and finite')
    scale=A4_SCALE if outer_width_mm is None else float(outer_width_mm)/span
    board=board_config(DEFAULT_BOARD,scale)
    return board,dict(measured_outer_black_width_mm=outer_width_mm,
        original_outer_black_width_mm=span,previous_assumed_outer_black_width_mm=span*A4_SCALE,
        scale=scale,relative_to_previous_scale=scale/A4_SCALE,
        tag_black_edge_mm=board['tag_size_mm'],pitch_mm=board['pitch_mm'],
        assumption='Uniform print scaling in both axes; existing gap/tag ratio; same size for all three boards')

def run(session,outer_width_mm=None,output=None):
    session=Path(session); out=Path(output) if output else session/'multiboard'
    board,geometry=measured_board(outer_width_mm)
    out.mkdir(exist_ok=False)
    (out/'board_geometry.json').write_text(json.dumps(geometry,indent=2))
    rgb=np.asarray(Image.open(session/'last_aligned_rgb.png'))
    meta=json.loads((session/'camera.json').read_text());k=np.asarray(meta['K'])
    samples=np.load(session/'last_depth_window_m.npy')
    valid=np.isfinite(samples)&(samples>.1)&(samples<=2.5)
    depth=np.ma.median(np.ma.array(samples,mask=~valid),axis=0).filled(np.nan)
    depth[valid.sum(axis=0)<3]=np.nan
    # Identical pinhole formula to online capture, in camera frame only.
    yy,xx=np.indices(depth.shape)
    xyz=np.stack(((xx-k[0,2])/k[0,0]*depth,(yy-k[1,2])/k[1,1]*depth,depth),axis=-1)*1000
    obsdir=out/'camera_observation';obsdir.mkdir()
    Image.fromarray(rgb).save(obsdir/'rgb.png')
    np.save(obsdir/'camera_xyz_mm.npy',xyz)
    obs=Observation(obsdir,{'world_from_camera_mm':np.eye(4).tolist()},xyz)
    tags=Detector(families=board['family'],quad_decimate=1).detect(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))
    # Image-specific grouping: split at largest two horizontal inter-column gaps.
    ordered=sorted(tags,key=lambda t:t.center[0])
    gaps=np.diff([t.center[0] for t in ordered])
    split=sorted((np.argsort(gaps)[-2:]+1).tolist())
    groups=[ordered[:split[0]],ordered[split[0]:split[1]],ordered[split[1]:]]
    if any(len(g)!=16 or len(set(t.tag_id for t in g))!=16 for g in groups):
        raise ValueError('Three spatial groups must each contain unique IDs 0–15; refusing guessed association')
    rows=[]; summaries=[]; overlay=rgb.copy()
    colors=[(255,80,80),(60,255,100),(60,160,255)]
    for name,tags,color in zip(('left_board','middle_board','right_board'),groups,colors):
        detector=SimpleNamespace(detect=lambda *a,**kw:tags)
        r=analyze(obs,k,meta,board,detector,meta['max_surface_spread_mm'],name)
        for row in r['rows']:
            u,v=row['sample_u'],row['sample_v']
            a=samples[:,v-2:v+3,u-2:u+3];mask=valid[:,v-2:v+3,u-2:u+3]
            half=len(a)//2
            med=[np.ma.median(np.ma.array(part,mask=~m),axis=0).filled(np.nan) for part,m in ((a[:half],mask[:half]),(a[half:],mask[half:]))]
            common=(mask[:half].mean(axis=0)>=.6)&(mask[half:].mean(axis=0)>=.6)
            drift=float(np.nanmedian(np.abs(med[1]-med[0])[common])*1000) if common.any() else None
            row.update(board=name,temporal_valid_ratio=float(mask.mean()),stable_pixel_ratio=float(common.mean()),temporal_drift_mm=drift,
                local_stable=bool(common.mean()>=.7 and drift is not None and drift<=2))
            if row['depth_status']=='VALID':
                b=np.array([row['board_camera_x_mm'],row['board_camera_y_mm'],row['board_camera_z_mm']])
                radial=b[:2]/b[2]*row['dz_mm']
                row['depth_scale_induced_dx_mm']=float(radial[0]);row['depth_scale_induced_dy_mm']=float(radial[1])
                row['residual_xy_after_radial_mm']=float(np.linalg.norm(np.array([row['dx_mm'],row['dy_mm']])-radial))
            for t in tags:
                cv2.polylines(overlay,[np.rint(t.corners).astype(int)],True,color,1)
                uv=tuple(np.rint(t.center).astype(int));cv2.putText(overlay,str(t.tag_id),uv,cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
            if row['depth_status']=='VALID':
                cv2.arrowedLine(overlay,(u,v),(round(u+row['dx_mm']*5),round(v+row['dy_mm']*5)),color,2,tipLength=.2)
        folder=out/name;folder.mkdir()
        # Save camera-only fields; synthetic identity is NOT a world calibration.
        for row in r['rows']:
            for field in list(row):
                if 'world' in field:del row[field]
        r.pop('T_world_camera_mm',None);r.pop('world_correction_affine_mm',None)
        r['measurement_scope']='CAMERA_ONLY; raw aligned temporal median; full-workspace stability FAILED; no corrected/world result'
        (folder/'analysis.json').write_text(json.dumps(r,indent=2))
        rows.extend(r['rows'])
        good=[x for x in r['rows'] if x['depth_status']=='VALID' and x['local_stable']]
        summaries.append(dict(board=name,tags=len(tags),valid_stable=len(good),pose_status=r['status'],
            reprojection_rms_px=r['corner_reprojection_rms_px'],
            xy=distribution([x['xy_error_mm'] for x in good]),
            signed_dx=distribution([x['dx_mm'] for x in good]),signed_dy=distribution([x['dy_mm'] for x in good]),
            signed_z=distribution([x['dz_mm'] for x in good]),
            nonradial_xy=distribution([x['residual_xy_after_radial_mm'] for x in good])))
    cv2.putText(overlay,'Camera XY residual arrows x5 px/mm; separate PnP per board',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,0),2)
    Image.fromarray(overlay).save(out/'all_tags_xy.png')
    csv_write(out/'results.csv',rows,['board','tag_id','u','v','dx_mm','dy_mm','dz_mm','xy_error_mm'])
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    lines=['# Three-board camera-only audit','',
        f'48 tags, 3 independent PnP fits. Assumed tag black edge {board["tag_size_mm"]:.6f} mm; geometry provenance in board_geometry.json.',
        'Full-workspace online depth gate FAILED. Results below use raw aligned depth window with local tag-center checks; no world compensation, no robot motion.',
        '', '| Board | valid/stable | RGB RMS px | XY median/max mm | signed Z median mm | non-radial XY median mm |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['board']} | {s['valid_stable']}/16 | {s['reprojection_rms_px']:.3f} | {s['xy']['median']:.3f}/{s['xy']['max']:.3f} | {s['signed_z']['median']:.3f} | {s['nonradial_xy']['median']:.3f} |")
    lines+=['','XY is camera-frame depth minus PnP, not measured robot grasp error.',
        'Depth scale and board print size errors produce radial XY error increasing away from optical center.',
        'Small non-radial error is not independent proof of RGB-depth alignment: both estimates use the same pixel ray.',
        'No world/extrinsic/TCP accuracy claim. Verify physical printed size and board flatness before attributing error to depth.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summaries,indent=2))
    print(out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('session');p.add_argument('--outer-black-width-mm',type=float);p.add_argument('--output',type=Path)
    a=p.parse_args();run(a.session,a.outer_black_width_mm,a.output)
