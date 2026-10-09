#!/usr/bin/env python3
"""Offline nominal-geometry Home checks. No execution, no safety certification."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cloth_agent.dual_arm.kinematics import ArmModel
from cloth_agent.dual_arm.geometry import segment_distance,segment_box_distance,DualArmError,pose_error
from cloth_agent.dual_arm.safety import synthetic_safety,validate_sweep
from scripts.preview_dual_home_viser import route_joints,STRATEGIES


def diagnose(directory):
    raw=json.loads((directory/'installation_config.json').read_text());snap=json.loads((directory/'snapshot.json').read_text())
    models={};arms={}
    for k,a in raw['arms'].items():
        arms[k]=SimpleNamespace(arm_id=k,axis=a['axis'],raw=a,urdf=ROOT/a['urdf'],capsules=a['collision_capsules'],
            tcp_offset=np.array(a['tcp_offset_mm_deg']),world_from_base=np.array(a['world_from_base_mm']))
        models[k]=ArmModel(arms[k])
    # Nominal swept geometry only; never write synthetic safety into hardware config.
    config=SimpleNamespace(arms=arms,limits=raw['limits'],obstacles=raw['obstacles'],raw={'safety':synthetic_safety(arms)})
    initial={k:np.array(v['joints_deg']) for k,v in snap['arms'].items()}
    home={k:np.array(v['home_joints_deg']) for k,v in raw['arms'].items()}
    def distances(q):
        caps={k:m.capsules(q[k]) for k,m in models.items()}
        inter=min((float(segment_distance(a.start,a.end,b.start,b.end)-a.radius-b.radius),a.name,b.name)
                  for a in caps['left'] for b in caps['right'])
        env=min((float(segment_box_distance(c.start,c.end,box['min_mm'],box['max_mm'])-c.radius),k,c.name,box['name'])
            for k,shapes in caps.items() for c in shapes for box in config.obstacles if f'{k}/{c.name}' not in box.get('excluded_capsules',[]))
        return {'inter_arm_clearance_mm':inter[0],'inter_arm_pair':inter[1:],
                'environment_clearance_mm':env[0],'environment_pair':env[1:]}
    report={'physical_execution':False,'scope':'nominal geometry only, provisional tools/cameras; no measured tracking/stop margins',
            'current':distances(initial),'home':distances(home),'routes':{},'controller':{}}
    for k,m in models.items():
        report['controller'][k]={'state':snap['arms'][k]['state'],'errors':snap['arms'][k]['errors'],
            'tcp_offset_matches':bool(np.allclose(snap['arms'][k]['tcp_offset_mm_deg'],m.config.tcp_offset,atol=.1)),
            'fk_vs_controller_mm_deg':list(pose_error(m.forward(initial[k]),snap['arms'][k]['tcp_pose_mm_deg']))}
    for strategy in STRATEGIES:
        try:
            last=initial;nodes=0
            for progress in np.linspace(0,1,101)[1:]:
                q=route_joints(initial,home,strategy,progress)
                nodes+=validate_sweep(config,models,last,q)
                last=q
            report['routes'][strategy]={'status':'NOMINAL_GEOMETRY_ONLY_PASSED','nodes':nodes}
        except DualArmError as exc:report['routes'][strategy]={'status':'REJECTED','reason':str(exc)}
    (directory/'geometry_diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    diagnose(p.parse_args().directory)
