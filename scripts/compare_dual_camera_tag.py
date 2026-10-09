#!/usr/bin/env python3
"""Observe one physical AprilTag with two stationary wrist cameras. No robot API.
Fit B->A on initial samples, freeze it, evaluate only later independent samples.
Tag size is the measured outer BLACK square edge, not paper size.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import sys,time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from cloth_agent.dual_camera_tag import estimate_tag,fit_camera_transform,compare_tag,summarize,rigid


class Camera:
    def __init__(self,serial,width,height,fps):
        import pyrealsense2 as rs
        self.rs=rs;self.serial=serial;self.pipeline=rs.pipeline();self.started=False
        c=rs.config();c.enable_device(serial);c.enable_stream(rs.stream.color,width,height,rs.format.bgr8,fps)
        try:
            profile=self.pipeline.start(c);self.started=True
            intr=profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            if intr.model not in (rs.distortion.none,rs.distortion.brown_conrady):
                raise ValueError(f'Unsupported RGB distortion model: {intr.model}; do not ignore it')
            self.k=np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1.]])
            self.dist=np.asarray(intr.coeffs) if intr.model!=rs.distortion.none else np.zeros(5)
            self.metadata={'serial':serial,'width':intr.width,'height':intr.height,'K':self.k.tolist(),
                           'distortion_model':str(intr.model),'coefficients':self.dist.tolist()}
            for _ in range(15): self.pipeline.wait_for_frames(5000)
        except BaseException:
            self.close();raise
    def read(self):
        frame=self.pipeline.wait_for_frames(5000).get_color_frame()
        if not frame: raise RuntimeError('No RGB frame')
        host=time.monotonic();raw=np.asanyarray(frame.get_data()).copy()
        return raw,cv2.undistort(raw,self.k,self.dist),{'host_received_monotonic_s':host,
                'device_timestamp_ms':frame.get_timestamp(),'timestamp_domain':str(frame.get_frame_timestamp_domain()),
                'frame_number':frame.get_frame_number()}
    def close(self):
        if self.started:self.pipeline.stop();self.started=False


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serial-a',default='317222073552');p.add_argument('--serial-b',default='233622079809')
    p.add_argument('--family',default='tag36h11');p.add_argument('--tag-id',type=int,required=True)
    p.add_argument('--tag-size-mm',type=float,required=True)
    p.add_argument('--fit-samples',type=int,default=10);p.add_argument('--validation-samples',type=int,default=20)
    p.add_argument('--interval',type=float,default=.5);p.add_argument('--timeout',type=float,default=180)
    p.add_argument('--max-reprojection-px',type=float,default=2)
    p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720);p.add_argument('--fps',type=int,default=30)
    p.add_argument('--transform',type=Path,help='Independent or previously saved B->A transform JSON; skip fitting')
    p.add_argument('--no-preview',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'results'/('dual_tag_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    args=p.parse_args(argv)
    if (args.serial_a==args.serial_b or not np.isfinite(args.tag_size_mm) or args.tag_size_mm<=0
        or args.tag_id<0 or args.fit_samples<3 or args.validation_samples<1
        or not 0<args.interval<=60 or not 0<args.timeout<86400 or not 0<args.max_reprojection_px<100):
        p.error('Invalid camera identity, tag size, counts, timing or reprojection threshold')
    frozen=None
    if args.transform:
        doc=json.loads(args.transform.read_text())
        if doc.get('serial_a')!=args.serial_a or doc.get('serial_b')!=args.serial_b:
            p.error('Transform camera identities do not match')
        frozen=rigid(doc['A_from_B_mm'])
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    save('settings.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    from pupil_apriltags import Detector
    detector=Detector(families=args.family,quad_decimate=1,nthreads=2)
    cams=[];fit=[];rows=[];accepted=0;status='UNKNOWN';error=None;rejects=0
    print('Keep BOTH cameras fixed. Use ONE physical tag visible to both; same ID printed twice is invalid.\n'
          'No hardware synchronization: keep tag still during each pair. Move tag between samples to test spatial coverage.\n'
          'No robot connections or motion. Output: '+str(args.output),flush=True)
    try:
        for serial in (args.serial_a,args.serial_b):cams.append(Camera(serial,args.width,args.height,args.fps))
        save('intrinsics.json',{'A':cams[0].metadata,'B':cams[1].metadata})
        began=time.monotonic();last=-1e20;index=0
        with ThreadPoolExecutor(max_workers=2) as pool, (args.output/'observations.jsonl').open('w') as log:
            while time.monotonic()-began<args.timeout and len(rows)<args.validation_samples:
                futures=[pool.submit(c.read) for c in cams];frames=[f.result() for f in futures]
                index+=1;record={'attempt':index,'at':datetime.now(timezone.utc).isoformat(),
                                 'frames':{k:f[2] for k,f in zip(('A','B'),frames)}}
                # Host receive skew is a diagnostic, NOT exposure synchronization.
                record['host_receive_skew_ms']=abs(frames[0][2]['host_received_monotonic_s']-frames[1][2]['host_received_monotonic_s'])*1000
                display=[f[1].copy() for f in frames];detected=[]
                try:
                    for k,c,f,img in zip(('A','B'),cams,frames,display):
                        d=estimate_tag(cv2.cvtColor(f[1],cv2.COLOR_BGR2GRAY),c.k,detector,args.tag_id,args.tag_size_mm,args.max_reprojection_px)
                        detected.append(d);cv2.polylines(img,[np.int32(d['corners_rectified_px'])],True,(0,255,0),2)
                        xyz=np.array(d['camera_from_tag_mm'])[:3,3]
                        cv2.putText(img,f'{k} tag {args.tag_id}: {xyz.round(1)} mm',(15,35),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
                    if time.monotonic()-last>=args.interval:
                        a,b=[np.array(d['camera_from_tag_mm']) for d in detected]
                        record['detections']={'A':detected[0],'B':detected[1]}
                        if frozen is None:
                            record['phase']='FIT';fit.append((a,b))
                            if len(fit)==args.fit_samples:
                                frozen=fit_camera_transform(fit)
                                save('camera_transform.json',{'serial_a':args.serial_a,'serial_b':args.serial_b,
                                    'A_from_B_mm':frozen.tolist(),'fit_samples':len(fit),'source':'initial_disjoint_fit_samples',
                                    'scope':'camera-to-camera at fixed wrist poses; NOT base-to-base or hand-eye calibration'})
                        else:
                            record['phase']='VALIDATION';result=compare_tag(a,b,frozen);record['residual']=result;rows.append(result)
                            print(f"Validation {len(rows)}/{args.validation_samples}: delta={np.round(result['delta_B_minus_A_mm'],2)} mm; norm={result['position_error_mm']:.2f} mm; angle={result['orientation_error_deg']:.2f} deg",flush=True)
                        accepted+=1;last=time.monotonic()
                        for k,f in zip(('A','B'),frames):
                            name=f'{accepted:04d}_{k}_raw.png'
                            if not cv2.imwrite(str(args.output/name),f[0]):raise RuntimeError('Image save failed')
                            record.setdefault('images',{})[k]=name
                        log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
                except ValueError as exc:
                    rejects+=1;record.update(phase='REJECTED',reason=str(exc));log.write(json.dumps(record)+'\n');log.flush()
                if not args.no_preview:
                    preview=np.hstack([cv2.resize(im,(640,360)) for im in display])
                    cv2.putText(preview,f'FIT {len(fit)}/{args.fit_samples} | VALIDATION {len(rows)}/{args.validation_samples} | Q exit',(15,345),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                    cv2.imshow('A/B AprilTag alignment - stationary cameras',preview)
                    if cv2.waitKey(1)&255 in (27,ord('q')):status='CANCELLED';break
            if len(rows)>=args.validation_samples:status='COMPLETED'
            elif status!='CANCELLED':status='TIMEOUT'
    except KeyboardInterrupt:status='CANCELLED'
    except Exception as exc:status='FAILED';error=f'{type(exc).__name__}: {exc}'
    finally:
        for c in cams:c.close()
        if not args.no_preview:cv2.destroyAllWindows()
        summary={'status':status,'error':error,'fit_count':len(fit),'rejected_frames':rejects,
                 'validation':summarize(rows),'transform_input':str(args.transform) if args.transform else None,
                 'limitations':['Camera frames are different: raw XYZ subtraction is not alignment error.',
                    'Fit samples are excluded from validation; no ground truth accuracy claimed.',
                    'Cameras must stay fixed; timestamp records do not guarantee simultaneous exposure.',
                    'Stationary repeated tag observations measure repeatability; use held-out tag placements for spatial agreement.',
                    'Single planar tag may be ambiguous; metric scale depends on measured black-border size.',
                    'Cannot determine robot base alignment without both hand-eye transforms and robot poses.']}
        save('summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return 0 if status=='COMPLETED' else 1

if __name__=='__main__':raise SystemExit(main())
