"""Independent Camera B registration field: three placements, no robot control."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import threading
import time
import traceback

import cv2
import numpy as np
from PIL import Image
from cloth_agent.experiments.camb_native_ir_capture import describe
from cloth_agent.experiments.camb_cross_stream_audit import transform
from cloth_agent.experiments.apriltag_geometry_audit import ROOT,project_rgb
from cloth_agent.experiments.apriltag_multiboard_audit import measured_board

PLACEMENTS=('L-M','C-M','R-M')
TARGETS={'L-M':(.30,.55),'C-M':(.50,.55),'R-M':(.72,.55)}
FIELDS=['placement','repeat','attempt','tag_id','corner_id','rgb_u','rgb_v','native_depth_z_mm',
        'native_depth_patch_valid_count','native_depth_patch_spread_mm','stereo_depth_z_mm','stereo_du','stereo_dv','stereo_error_px','stereo_valid','stereo_reason',
        'direct_du','direct_dv','direct_error_px','direct_valid','direct_reason',
        'rgb_motion_rms_px','repeat_status','status','reject_reason']
METHODS=('stereo','direct')
REASON_ZH={
 'BOARD_MOTION_GT_1PX':'切换前后板移动超过 1 像素：请放稳并移开手',
 'INSUFFICIENT_MOTION_MATCHES':'前后共同 tag 不足：请保持整张板可见',
 'INSUFFICIENT_RGB_BOARD':'RGB 检出的 tag 不足 12 个：请移除遮挡',
 'OUTSIDE_PLACEMENT_TARGET':'板中心不在当前绿色区域内：请移回目标位置',
 'BOARD_ORIENTATION_CHANGED':'板的朝向或倾角改变过大：请平放并保持初始朝向',
 'INSUFFICIENT_PAIRED_CORNERS':'两种方法共同有效角点不足 24 个：可能存在角点深度空洞。若板边近水平，请停止后将板在桌面内斜放约30°再开始新session；采集中勿移动板'}
def explain_rejection(reasons):
    return '；'.join(REASON_ZH.get(r,r) for r in reasons)


def save(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def detect(detector,rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY) if rgb.ndim==3 else rgb
    tags=detector.detect(np.ascontiguousarray(gray),estimate_tag_pose=False)
    result={}
    for t in tags:
        if not 0<=t.tag_id<16 or t.hamming!=0 or t.decision_margin<20:continue
        if t.tag_id in result:raise ValueError('DUPLICATE_TAG_IDS: only one board allowed')
        if np.asarray(t.corners).shape!=(4,2) or not np.isfinite(t.corners).all():continue
        result[int(t.tag_id)]=np.asarray(t.corners,float)
    return result

def motion_rms(reference,current,min_tags=4):
    ids=sorted(set(reference)&set(current))
    if len(ids)<min_tags:return None
    delta=np.concatenate([current[i]-reference[i] for i in ids])
    # RMS of 2-D corner displacement, not RMS of separate scalar components.
    return float(np.sqrt(np.mean(np.sum(delta**2,axis=1))))

def validate_metadata(meta):
    for label in ('rgb','depth','ir1','ir2'):
        p=meta['profiles'][label]
        if np.any(np.asarray(p['coeffs'])!=0):
            raise ValueError('Nonzero distortion needs explicit model handling; abort, never silently ignore')
        if not np.isfinite(p['K']).all():raise ValueError('Invalid factory intrinsics')
    if not np.allclose(transform(meta,'ir1'),np.eye(4),atol=1e-8):
        raise ValueError('Native depth and IR1 do not share coordinates')
    if not np.allclose(meta['profiles']['depth']['K'],meta['profiles']['ir1']['K']):
        raise ValueError('Native depth and IR1 pixel grids differ')
    if meta['profiles']['depth']['width']!=meta['profiles']['ir1']['width'] or meta['profiles']['depth']['height']!=meta['profiles']['ir1']['height']:
        raise ValueError('Native depth and IR1 dimensions differ')

def calculate_corners(meta,tags,depth_stack,placement,repeat,attempt):
    validate_metadata(meta)
    kd=np.asarray(meta['profiles']['depth']['K'])
    kc=np.asarray(meta['profiles']['rgb']['K'])
    T=transform(meta,'rgb')
    projections=[np.asarray(meta['profiles'][l]['K'])@transform(meta,l)[:3] for l in ('ir1','ir2')]
    depths=depth_stack.astype(float)*meta['depth_scale']*1000
    good=np.isfinite(depths)&(depths>100)&(depths<2500)
    depth=np.ma.median(np.ma.array(depths,mask=~good),axis=0).filled(np.nan)
    h,w=depth.shape
    rows=[]
    for tag_id,rgbcorners in sorted(tags['rgb'].items()):
        for corner_id,rgbuv in enumerate(rgbcorners):
            row=dict(placement=placement,repeat=repeat,attempt=attempt,tag_id=tag_id,corner_id=corner_id,
                rgb_u=float(rgbuv[0]),rgb_v=float(rgbuv[1]),stereo_valid=False,direct_valid=False,
                stereo_reason='MISSING_IR_MATCH',direct_reason='MISSING_IR1_MATCH')
            if tag_id in tags['ir1']:
                uv=np.asarray(tags['ir1'][tag_id][corner_id])
                u,v=np.rint(uv).astype(int)
                row['direct_reason']='DEPTH_OUT_OF_BOUNDS'
                if 2<=u<w-2 and 2<=v<h-2:
                    patch=depth[v-2:v+3,u-2:u+3];values=patch[np.isfinite(patch)]
                    row['direct_reason']='INSUFFICIENT_DEPTH'
                    row['native_depth_patch_valid_count']=len(values)
                    row['native_depth_patch_spread_mm']=float(np.ptp(values)) if len(values) else None
                    if len(values)>=18:
                        # Preserve the previously verified native-depth method; log spread,
                        # do not add a grasp-specific discontinuity gate to a camera experiment.
                        row['native_depth_z_mm']=float(np.median(values))
                        point=np.linalg.solve(kd,[*uv,1.])*row['native_depth_z_mm']/1000
                        pred=project_rgb(T,[point],kc,np.zeros(5))[0]
                        delta=pred-rgbuv
                        row.update(direct_valid=True,direct_reason='',direct_du=float(delta[0]),
                            direct_dv=float(delta[1]),direct_error_px=float(np.linalg.norm(delta)))
                if tag_id in tags['ir2']:
                    uv2=np.asarray(tags['ir2'][tag_id][corner_id])
                    hom=cv2.triangulatePoints(*projections,uv.reshape(2,1),uv2.reshape(2,1))[:,0]
                    row['stereo_reason']='INVALID_TRIANGULATION'
                    if np.isfinite(hom).all() and abs(hom[3])>1e-10:
                        point=hom[:3]/hom[3]
                        projected=[];zs=[]
                        for label in ('ir1','ir2'):
                            ts=transform(meta,label);ps=ts[:3,:3]@point+ts[:3,3];zs.append(ps[2])
                            projected.append(project_rgb(ts,[point],np.asarray(meta['profiles'][label]['K']),np.zeros(5))[0])
                        # IR reprojection/cheirality checks only. Never threshold the RGB residual being measured.
                        ir_error=max(np.linalg.norm(projected[0]-uv),np.linalg.norm(projected[1]-uv2))
                        row['stereo_reason']='IR_GEOMETRY_REJECT'
                        if all(z>.1 and z<2.5 for z in zs) and ir_error<=2.:
                            pred=project_rgb(T,[point],kc,np.zeros(5))[0];delta=pred-rgbuv
                            row.update(stereo_valid=True,stereo_reason='',stereo_depth_z_mm=float(point[2]*1000),
                                stereo_du=float(delta[0]),stereo_dv=float(delta[1]),stereo_error_px=float(np.linalg.norm(delta)))
            rows.append(row)
    return rows

def board_geometry(tags,meta):
    if len(tags)<4:return None
    board,_=measured_board(142.)
    ids=sorted(tags)
    obj=np.concatenate([board['corners'][i] for i in ids])
    pix=np.concatenate([tags[i] for i in ids])
    H,_=cv2.findHomography(obj[:,:2],pix,0)
    axis=cv2.perspectiveTransform(np.array([[[0.,0.],[.05,0.]]],dtype=float),H)[0]
    delta=axis[1]-axis[0]
    ok,r,t=cv2.solvePnP(obj,pix,np.asarray(meta['profiles']['rgb']['K']),np.zeros(5))
    if not ok:return None
    normal=cv2.Rodrigues(r)[0][:,2]
    if normal[2]<0:normal=-normal
    return dict(center=axis[0].tolist(),rgb_tags=len(tags),inplane_deg=float(np.degrees(np.arctan2(delta[1],delta[0]))),normal=normal.tolist())

def finalize_repeat(rows,motions,geometry,reference_geometry,meta,placement,min_corners=24):
    reasons=[]
    if not motions or any(m is None for m in motions):reasons.append('INSUFFICIENT_MOTION_MATCHES')
    motion=max((m for m in motions if m is not None),default=None)
    if motion is not None and motion>1.:reasons.append('BOARD_MOTION_GT_1PX')
    if geometry is None or geometry.get('rgb_tags',0)<12:reasons.append('INSUFFICIENT_RGB_BOARD')
    else:
        w=meta['profiles']['rgb']['width'];h=meta['profiles']['rgb']['height']
        x,y=geometry['center'];tx,ty=TARGETS[placement]
        if abs(x/w-tx)>.065 or abs(y/h-ty)>.10:reasons.append('OUTSIDE_PLACEMENT_TARGET')
        if reference_geometry is not None:
            turn=abs((geometry['inplane_deg']-reference_geometry['inplane_deg']+180)%360-180)
            angle=np.degrees(np.arccos(np.clip(np.dot(geometry['normal'],reference_geometry['normal']),-1,1)))
            if turn>10 or angle>10:reasons.append('BOARD_ORIENTATION_CHANGED')
    paired=sum(r['direct_valid'] and r['stereo_valid'] for r in rows)
    if paired<min_corners:reasons.append('INSUFFICIENT_PAIRED_CORNERS')
    accepted=not reasons
    for r in rows:
        r.update(rgb_motion_rms_px=motion,repeat_status='VALID' if accepted else 'REJECTED',
            status=('VALID' if r['direct_valid'] and r['stereo_valid'] else 'PARTIAL') if accepted else 'REJECTED',
            reject_reason=';'.join(reasons))
    return dict(accepted=accepted,reasons=reasons,rgb_motion_rms_px=motion,paired_corners=paired,geometry=geometry)

def append_rows(path,rows):
    exists=Path(path).exists()
    with Path(path).open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore')
        if not exists:writer.writeheader()
        writer.writerows(rows)

def check_cancel(event):
    if event.is_set():raise InterruptedError('Operator stopped session')

class NativeCamera:
    """Camera only; existing adapter owns exposure save/restore."""
    def __init__(self,config_path):
        from cloth_agent.dual_arm.config import DualConfig
        from cloth_agent.codex_camera_b import with_arm7_camera
        from cloth_agent.dual_arm.observation import create_camera
        config=with_arm7_camera(DualConfig.load(config_path,root=ROOT,homing_only=True))
        self.cam=create_camera(config);self.rs=self.cam.rs;self.sensor=None;self.original_emitter=None
        self.depth_options={}
        for index in (1,2):
            self.cam.config.enable_stream(self.rs.stream.infrared,index,self.cam.width,self.cam.height,self.rs.format.y8,self.cam.fps)
    def start(self):
        self.cam.start()
        self.sensor=self.cam.pipeline.get_active_profile().get_device().first_depth_sensor()
        self.original_emitter=self.sensor.get_option(self.rs.option.emitter_enabled)
        for name in ('enable_auto_exposure','exposure','gain'):
            opt=getattr(self.rs.option,name)
            if self.sensor.supports(opt):self.depth_options[name]=self.sensor.get_option(opt)
        profile=self.cam.pipeline.get_active_profile()
        streams={'rgb':profile.get_stream(self.rs.stream.color),'depth':profile.get_stream(self.rs.stream.depth),
            'ir1':profile.get_stream(self.rs.stream.infrared,1),'ir2':profile.get_stream(self.rs.stream.infrared,2)}
        meta=dict(profiles={k:describe(p) for k,p in streams.items()},depth_scale=self.cam.depth_scale,
            serial=self.cam.spec.serial,original_emitter=self.original_emitter,depth_capture_options=self.depth_options,
            robot_connected=False,extrinsics={})
        for name,p in streams.items():
            e=streams['depth'].get_extrinsics_to(p)
            meta['extrinsics']['depth_to_'+name]=dict(rotation_column_major=list(e.rotation),translation_m=list(e.translation))
        validate_metadata(meta)
        return meta
    def emitter(self,value):
        self.sensor.set_option(self.rs.option.emitter_enabled,float(value))
        measured=self.sensor.get_option(self.rs.option.emitter_enabled)
        if measured!=float(value):raise RuntimeError(f'Emitter setting failed: {value} -> {measured}')
    def read(self):
        f=self.cam.pipeline.wait_for_frames(timeout_ms=5000)
        frames=dict(rgb=f.get_color_frame(),depth=f.get_depth_frame(),ir1=f.get_infrared_frame(1),ir2=f.get_infrared_frame(2))
        if not all(frames.values()):raise RuntimeError('Incomplete RGB/IR/depth frameset')
        result={k:np.asanyarray(v.get_data()).copy() for k,v in frames.items()}
        result['timestamps']={k:v.get_timestamp() for k,v in frames.items()}
        return result
    def restore_emitter(self):
        if self.original_emitter is not None:
            self.emitter(self.original_emitter)
            return dict(original=self.original_emitter,restored=self.sensor.get_option(self.rs.option.emitter_enabled),verified=True)
        return dict(verified=False,reason='Emitter not initialized')
    def close(self):
        try:return self.restore_emitter()
        finally:self.cam.stop()

def capture_pair(camera,folder,cancel,preview,status):
    """Always restore emitter, including partial acquisition/cancellation."""
    evidence={};windows=[];timestamps=[]
    def grab():
        check_cancel(cancel);frame=camera.read();preview(frame['rgb']);return frame
    def settle(n):
        for _ in range(n):grab()
    try:
        status('采集 emitter-on 原生深度；请勿移动板')
        camera.emitter(1);settle(30)
        for i in range(15):
            f=grab();windows.append(f['depth']);timestamps.append(dict(phase='on',**f['timestamps']))
            if i in (0,14):
                key='on_first' if i==0 else 'on_last';evidence[key]=f['rgb']
                Image.fromarray(f['rgb']).save(folder/f'{key}.png')
                # Same-phase IR is essential to diagnose saturation/depth holes.
                for label in ('ir1','ir2'):
                    Image.fromarray(f[label]).save(folder/f'{key}_{label}.png')
        np.savez_compressed(folder/'native_depth.npz',depth=np.asarray(windows))
        status('采集 emitter-off RGB/IR；请勿移动板')
        camera.emitter(0);settle(30)
        for i in range(15):
            f=grab();timestamps.append(dict(phase='off',**f['timestamps']))
            if i in (0,14):
                key='off_first' if i==0 else 'off_last';evidence[key]=f['rgb']
                Image.fromarray(f['rgb']).save(folder/f'{key}.png')
            if i==14:
                evidence['ir1']=f['ir1'];evidence['ir2']=f['ir2']
                Image.fromarray(f['ir1']).save(folder/'ir1.png');Image.fromarray(f['ir2']).save(folder/'ir2.png')
    finally:
        # If restoration fails, propagate and abort the whole session, never retry hardware faults.
        restored=camera.restore_emitter();save(folder/'emitter_restoration.json',restored)
    settle(10);f=grab();evidence['post_restore']=f['rgb']
    Image.fromarray(f['rgb']).save(folder/'post_restore.png')
    timestamps.append(dict(phase='restored',**f['timestamps']));save(folder/'timestamps.json',timestamps)
    return evidence,np.asarray(windows)

def summarize(session):
    from cloth_agent.experiments.registration_field_report import generate
    return generate(Path(session))

def run_session(args,ui):
    from pupil_apriltags import Detector
    session=args.output/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    session.mkdir(parents=True,exist_ok=False);ui.session=str(session)
    manifest=dict(schema='registration_field_v1',status='STARTED',placements=list(PLACEMENTS),valid_repeats=args.repeats,
        max_attempts_per_placement=args.max_attempts,motion_threshold_px=1.,motion_definition='sqrt(mean(du^2+dv^2))',
        min_paired_corners=24,placement_frame='Original RGB left/center/right; M=image middle row, not a prescribed optical depth',
        residual_sign='projected RGB minus observed RGB',robot_connected=False,calibration_written=False,attempts=[],
        placement_targets=TARGETS,registration_correction_fitted=False)
    save(session/'session.json',manifest);camera=None;lease=None
    try:
        lease=open('/tmp/clothagent_codex_dual_arm.lock','a')
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        camera=NativeCamera(args.config)
        meta=camera.start();save(session/'camera.json',meta)
        detector=Detector(families='tag36h11',quad_decimate=1,nthreads=2)
        reference=None
        for placement in PLACEMENTS:
            ui.placement=placement;ui.ready.clear();ui.waiting=True
            ui.update('请将一张板平放到高亮区域：图案边与画面水平线约30°，三处保持同一朝向；手移开后按 Enter。',0,args.repeats)
            while not ui.ready.is_set():
                check_cancel(ui.cancel)
                rgb=camera.read()['rgb'];ui.preview(rgb)
                now=time.monotonic()
                if now-getattr(ui,'last_detection_time',0)>.4:
                    ui.last_detection_time=now
                    try:
                        seen=detect(detector,rgb);g=board_geometry(seen,meta)
                        ui.detected_center=g['center'] if g else None;ui.detected_tags=len(seen)
                    except ValueError:
                        ui.detected_center=None;ui.detected_tags=0
            ui.waiting=False;accepted=0
            for attempt in range(1,args.max_attempts+1):
                check_cancel(ui.cancel)
                if accepted>=args.repeats:break
                repeat=accepted+1
                folder=session/placement/f'attempt_{attempt:03d}';folder.mkdir(parents=True)
                ui.update(f'自动采集 {placement}：有效 {accepted}/{args.repeats}，尝试 {attempt}',accepted,args.repeats)
                evidence,depth=capture_pair(camera,folder,ui.cancel,ui.preview,
                    lambda msg:ui.update(msg,accepted,args.repeats))
                rows=[];result=None
                try:
                    rgbmaps={key:detect(detector,evidence[key]) for key in ('on_first','on_last','off_first','off_last','post_restore')}
                    tags=dict(rgb=rgbmaps['off_last'],ir1=detect(detector,evidence['ir1']),ir2=detect(detector,evidence['ir2']))
                    rows=calculate_corners(meta,tags,depth,placement,repeat,attempt)
                    motions=[motion_rms(rgbmaps['on_first'],rgbmaps[k]) for k in ('on_last','off_first','off_last','post_restore')]
                    geometry=board_geometry(tags['rgb'],meta)
                    result=finalize_repeat(rows,motions,geometry,reference,meta,placement)
                    save(folder/'detections.json',{key:{str(i):c.tolist() for i,c in ts.items()} for key,ts in tags.items()})
                    result['motion_checks_px']=motions
                except ValueError as exc:
                    # Detection ambiguity is a rejected attempt; hardware/config failures are not caught here.
                    result=dict(accepted=False,reasons=[str(exc)],rgb_motion_rms_px=None,paired_corners=0)
                    for row in rows:row.update(repeat_status='REJECTED',status='REJECTED',rgb_motion_rms_px=None,reject_reason=str(exc))
                if not rows:
                    rows=[dict(placement=placement,repeat=repeat,attempt=attempt,repeat_status='REJECTED',
                        status='REJECTED',reject_reason=';'.join(result['reasons']),stereo_valid=False,direct_valid=False)]
                save(folder/'result.json',result);save(folder/'corner_results.json',rows)
                append_rows(session/'all_samples.csv',rows)
                if result['accepted']:
                    accepted+=1
                    if reference is None:reference=result['geometry']
                manifest['attempts'].append(dict(placement=placement,repeat=repeat,attempt=attempt,**result))
                save(session/'session.json',manifest)
                counts=f"（stereo {sum(bool(r.get('stereo_valid')) for r in rows)}；direct depth {sum(bool(r.get('direct_valid')) for r in rows)}；共同 {result.get('paired_corners',0)}）"
                message='有效 repeat 已保存'+counts if result['accepted'] else '已拒绝并自动重采：'+explain_rejection(result['reasons'])+counts
                ui.update(message,accepted,args.repeats);print(f'{placement} {accepted}/{args.repeats}: {message}',flush=True)
            if accepted<args.repeats:raise RuntimeError(f'{placement}: reached {args.max_attempts} attempts with only {accepted} valid repeats')
        manifest['status']='COMPLETE'
    except (Exception,KeyboardInterrupt) as exc:
        manifest.update(status='INTERRUPTED' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'FAILED',
            error=f'{type(exc).__name__}: {exc}')
        (session/'failure.txt').write_text(traceback.format_exc())
    finally:
        if camera is not None:
            try:save(session/'final_emitter_restoration.json',camera.close())
            except Exception as exc:
                manifest.update(status='RESTORATION_FAILED',error=str(exc))
        if lease is not None:lease.close()
        save(session/'session.json',manifest)
        try:summarize(session)
        except Exception as exc:
            manifest['report_error']=str(exc)
            manifest['capture_status']=manifest['status'];manifest['status']='REPORT_FAILED'
            manifest['error']=manifest.get('error','')+f' Report generation failed: {exc}'
            save(session/'session.json',manifest)
        ui.finished(manifest['status'],str(session),manifest.get('error',''))
    return session

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run');run.add_argument('--output',type=Path,default=ROOT/'runs/registration_field')
    run.add_argument('--config',type=Path,default=ROOT/'config/dual_arm.local.json')
    run.add_argument('--repeats',type=int,default=10)
    run.add_argument('--max-attempts',type=int,default=40)
    review=sub.add_parser('review');review.add_argument('session',type=Path)
    args=parser.parse_args()
    if args.command=='review':summarize(args.session);return
    if args.repeats<1 or args.max_attempts<args.repeats:parser.error('Require max-attempts >= repeats >= 1')
    from cloth_agent.experiments.registration_field_ui import launch
    raise SystemExit(launch(lambda ui:run_session(args,ui)))
if __name__=='__main__':main()
