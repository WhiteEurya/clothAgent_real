#!/usr/bin/env python3
"""Offline old-SciPy/PyRoki comparison; no robot SDK or camera connection."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.geometry import matrix_pose,pose_matrix,pose_error
from cloth_agent.dual_arm.kinematics import ArmModel


def old_scipy(model,pose,seed_deg):
    """Frozen pre-migration solver for comparison only, never a fallback."""
    target=pose_matrix(pose);seed=np.radians(seed_deg)
    def residual(joints):
        actual=pose_matrix(model.forward(np.degrees(joints)))
        return np.r_[(actual[:3,3]-target[:3,3])/1000,
            Rotation.from_matrix(target[:3,:3].T@actual[:3,:3]).as_rotvec(),1e-4*(joints-seed)]
    result=least_squares(residual,np.clip(seed,model.lower,model.upper),
        bounds=(model.lower,model.upper),max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    q=np.degrees(result.x);distance,angle=pose_error(model.forward(q),pose)
    if not result.success or distance>.5 or angle>.2:
        raise ValueError(f'old IK rejected ({distance:.6f} mm, {angle:.6f} deg)')
    return q


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/dual_arm.local.json')
    parser.add_argument('--round',type=Path,default=ROOT/'results/codex_astra/loop_20261009T125018455816Z/round_001')
    parser.add_argument('--output',type=Path,default=ROOT/'results/pyroki_ik_migration/comparison.json')
    args=parser.parse_args();config=DualConfig.load(args.config,root=ROOT,homing_only=True)
    models={k:ArmModel(a) for k,a in config.arms.items()};rows=[]
    def compare(arm,name,target,seed):
        m=models[arm]
        for backend,solve in [('scipy',lambda:old_scipy(m,target,seed)),('pyroki',lambda:m.inverse(target,seed))]:
            t=time.perf_counter()
            try:
                q=solve();distance,angle=pose_error(m.forward(q),target)
                data=dict(status='IK_OK',distance_mm=distance,angle_deg=angle,joints_deg=q.tolist())
            except Exception as e:data=dict(status='REJECTED',error=str(e))
            rows.append(dict(arm=arm,case=name,backend=backend,elapsed_s=time.perf_counter()-t,**data))
    for arm,m in models.items():
        seed=config.arms[arm].home_joints.copy();target=m.forward(seed);target[2]+=3
        compare(arm,'home_lift_cold',target,seed)
        for i in range(5):
            target=m.forward(seed);target[2]+=i+1
            compare(arm,f'home_lift_{i+1}mm_warm',target,seed)
    if args.round.exists():
        outcome=json.loads((args.round/'outcome.json').read_text())
        state=json.loads((args.round/'before/state.json').read_text())
        context=outcome['failure_context'];arm=context['arm'];pose=context['arguments'][arm]
        target=matrix_pose(np.linalg.inv(config.arms[arm].world_from_base)@
            pose_matrix(np.r_[np.array(pose[:3])*1000,np.degrees(pose[3:])]))
        compare(arm,'saved_failed_target',target,state['joints'][arm])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(physical_execution=False,cases=rows),indent=2)+'\n')
    print(args.output)
    for backend in ('scipy','pyroki'):
        warm=[r['elapsed_s'] for r in rows if r['backend']==backend and r['case'].endswith('_warm')]
        print(backend,'warm_median_ms',1000*float(np.median(warm)))


if __name__=='__main__':main()
