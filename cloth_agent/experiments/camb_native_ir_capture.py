"""Read-only RGB/IR/native-depth evidence capture; no robot connection."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import fcntl,json
import numpy as np
from PIL import Image
from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.codex_camera_b import with_arm7_camera
from cloth_agent.dual_arm.observation import create_camera
from cloth_agent.experiments.apriltag_geometry_audit import ROOT
def describe(p):
    i=p.as_video_stream_profile().get_intrinsics()
    return dict(K=[[i.fx,0,i.ppx],[0,i.fy,i.ppy],[0,0,1]],coeffs=list(i.coeffs),model=str(i.model),width=i.width,height=i.height)
def run(emitter_off=False):
    out=ROOT/'reports/camb_geometry_root_cause_20261016'/('native_ir_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    config=with_arm7_camera(DualConfig.load(ROOT/'config/dual_arm.local.json',root=ROOT,homing_only=True))
    cam=create_camera(config);rs=cam.rs;sensor=None;original_emitter=None
    cam.config.enable_stream(rs.stream.infrared,1,1280,720,rs.format.y8,15)
    cam.config.enable_stream(rs.stream.infrared,2,1280,720,rs.format.y8,15)
    with open('/tmp/clothagent_codex_dual_arm.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            cam.start()
            sensor=cam.pipeline.get_active_profile().get_device().first_depth_sensor()
            if emitter_off:
                original_emitter=sensor.get_option(rs.option.emitter_enabled)
                sensor.set_option(rs.option.emitter_enabled,0.)
            for _ in range(30):cam.pipeline.wait_for_frames()
            depths=[];aligned=[];timestamps=[]
            for i in range(15):
                f=cam.pipeline.wait_for_frames()
                depth=f.get_depth_frame();rgb=f.get_color_frame();ir1=f.get_infrared_frame(1);ir2=f.get_infrared_frame(2)
                depths.append(np.asanyarray(depth.get_data()).copy())
                aligned.append(np.asanyarray(cam.align.process(f).get_depth_frame().get_data()).copy())
                timestamps.append(dict(depth=depth.get_timestamp(),rgb=rgb.get_timestamp(),ir1=ir1.get_timestamp(),ir2=ir2.get_timestamp()))
                # Endpoints retained for checking whether board moved during window.
                if i in (0,14):
                    for label,frame in [('rgb',rgb),('ir1',ir1),('ir2',ir2)]:
                        Image.fromarray(np.asanyarray(frame.get_data()).copy()).save(out/f'{label}_{i:02d}.png')
            np.save(out/'native_depth_u16.npy',np.asarray(depths))
            np.save(out/'aligned_depth_u16.npy',np.asarray(aligned))
            profiles=dict(rgb=rgb.profile,depth=depth.profile,ir1=ir1.profile,ir2=ir2.profile)
            metadata=dict(depth_scale=cam.depth_scale,profiles={k:describe(p) for k,p in profiles.items()},timestamps=timestamps,
                robot_connected=False,emitter_off=emitter_off,depth_use='DO_NOT_USE_FOR_DEPTH_EVALUATION' if emitter_off else 'NATIVE_DEPTH',frames=15,serial=config.raw['camera']['serial'])
            metadata['extrinsics']={}
            for label,p in profiles.items():
                e=profiles['depth'].get_extrinsics_to(p)
                metadata['extrinsics']['depth_to_'+label]=dict(rotation_column_major=list(e.rotation),translation_m=list(e.translation))
            (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
            print(out,flush=True)
        finally:
            try:
                if original_emitter is not None:
                    sensor.set_option(rs.option.emitter_enabled,original_emitter)
                    restored=sensor.get_option(rs.option.emitter_enabled)
                    (out/'emitter_restoration.json').write_text(json.dumps(dict(original=original_emitter,restored=restored)))
                    if restored!=original_emitter:raise RuntimeError('Emitter restoration verification failed')
            finally:cam.stop()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ir-only-emitter-off',action='store_true');a=p.parse_args();run(a.ir_only_emitter_off)
