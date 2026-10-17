"""Independent Camera B RGB intrinsic audit; no robot imports or calibration writes."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import fcntl
import json
import time
import numpy as np
from PIL import Image
from cloth_agent.experiments.apriltag_geometry_audit import ROOT,DEFAULT_BOARD,A4_SCALE,board_config

def capture_rgb(output,seconds):
    from cloth_agent.dual_arm.config import DualConfig
    from cloth_agent.codex_camera_b import with_arm7_camera
    from cloth_agent.dual_arm.observation import create_camera
    folder=Path(output)/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True,exist_ok=False);(folder/'images').mkdir()
    config=with_arm7_camera(DualConfig.load(ROOT/'config/dual_arm.local.json',root=ROOT,homing_only=True))
    camera=create_camera(config)
    with open('/tmp/clothagent_codex_dual_arm.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            camera.start()
            for _ in range(15):camera.read()
            intr=camera.pipeline.get_active_profile().get_stream(camera.rs.stream.color).as_video_stream_profile().get_intrinsics()
            meta=dict(K=camera.intrinsics.tolist(),dist=list(intr.coeffs),distortion_model=str(intr.model),
                image_size=[intr.width,intr.height],serial=camera.spec.serial,robot_connected=False,
                board_scale=A4_SCALE,board_source=str(DEFAULT_BOARD),duration_requested_s=seconds)
            (folder/'camera.json').write_text(json.dumps(meta,indent=2))
            start=time.monotonic();next_save=0;index=0;records=[]
            print('CAPTURE_STARTED '+str(folder),flush=True)
            while time.monotonic()-start<seconds:
                rgb,_=camera.read();elapsed=time.monotonic()-start
                if elapsed>=next_save:
                    name=f'images/frame_{index:04d}.png';Image.fromarray(rgb).save(folder/name)
                    records.append(dict(image=name,elapsed_s=elapsed))
                    index+=1;next_save=elapsed+.4
                    if index%20==0:print(f'CAPTURING elapsed={elapsed:.1f}s frames={index}',flush=True)
            (folder/'frames.json').write_text(json.dumps(records,indent=2))
            print('CAPTURE_FINISHED '+str(folder),flush=True)
        finally:camera.stop()
    return folder

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/camb_intrinsics_audit')
    p.add_argument('--seconds',type=float,default=60);a=p.parse_args()
    capture_rgb(a.output,a.seconds)
