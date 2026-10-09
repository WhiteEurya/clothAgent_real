#!/usr/bin/env python3
"""Prepare UNVERIFIED model envelopes. Does not enable real execution or move arms."""
import json,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cloth_agent.dual_arm.setup import mesh_capsules
from cloth_agent.dual_arm.viewer_accessories import load_hand_eye


def main():
    path=ROOT/'config/dual_arm.local.json';raw=json.loads(path.read_text())
    for key,arm in raw['arms'].items():
        arm['collision_capsules']=mesh_capsules(ROOT/arm['urdf'],arm['axis'],tool_radius_mm=45,tcp_offset=arm['tcp_offset_mm_deg'])
        eye=load_hand_eye(ROOT/('config/extrinsics_A.yaml' if key=='left' else 'config/calibration/dual_arm_working_20261008/camB_extrinsics.yaml'))
        # Encloses the viewer's 90x25x25 body with a capsule along optical X.
        a=eye@np.array([-.045,0,-.0125,1]);b=eye@np.array([.045,0,-.0125,1])
        arm['collision_capsules'].append({'name':'camera','frame':'link_eef','start_mm':(a[:3]*1000).tolist(),
            'end_mm':(b[:3]*1000).tolist(),'radius_mm':float(np.hypot(12.5,12.5)),
            'source':'D435 90x25x25 mm visual body; optical-to-housing offset approximate'})
    raw['obstacles']=[{'name':'table_z0_user_confirmed','min_mm':[-10000,-10000,-10000],
                      'max_mm':[10000,10000,0],'excluded_capsules':['left/link_base','right/link_base'],
                      'source':'User: installation planes flush with tabletop. Broad slab represents z<=0 over modeled workspace; table edges not measured.'}]
    raw['geometry_provenance']={'status':'approximate_not_measured','tool_radius_mm':45,
        'camera_optical_to_body_offset':'approximate','printed_mount':'omitted at user request; not independently proven contained by camera envelope',
        'payload':'user confirmed empty','table':'user confirmed flush; world z=0 plane',
        'note':'URDF link capsules + provisional tool/camera capsules. No stop/tracking bounds supplied.'}
    raw['collision_geometry_verified']=False
    backup=path.with_name(path.name+'.'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.bak')
    shutil.copy2(path,backup);path.write_text(json.dumps(raw,ensure_ascii=False,indent=2)+'\n')
    print('Saved approximate geometry; real execution flags remain disabled:',path)

if __name__=='__main__':main()
