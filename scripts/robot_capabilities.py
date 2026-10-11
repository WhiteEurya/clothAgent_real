#!/usr/bin/env python3
"""Generate model guidance or check a World TCP target offline, without hardware."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.kinematics import ArmModel
from cloth_agent.dual_arm.geometry import matrix_pose,pose_matrix,pose_error
from cloth_agent.robot_capabilities import build_capabilities,render_capabilities


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/dual_arm.local.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--arm',choices=['left','right'])
    parser.add_argument('--pose-mm-deg',type=float,nargs=6)
    parser.add_argument('--grasp',action='store_true')
    parser.add_argument('--grasp-offset-mm',type=float,default=10.)
    args=parser.parse_args()
    if not np.isfinite(args.grasp_offset_mm) or args.grasp_offset_mm<0:parser.error('invalid grasp offset')
    if bool(args.arm)!=bool(args.pose_mm_deg):parser.error('--arm and --pose-mm-deg must be used together')
    if args.grasp and not args.pose_mm_deg:parser.error('--grasp requires a target')
    config=DualConfig.load(args.config,root=ROOT,homing_only=True)
    models={k:ArmModel(a) for k,a in config.arms.items()}
    if args.pose_mm_deg:
        target=np.asarray(args.pose_mm_deg);target[2]-=args.grasp_offset_mm if args.grasp else 0
        arm=config.arms[args.arm];model=models[args.arm]
        base=matrix_pose(np.linalg.inv(arm.world_from_base)@pose_matrix(target))
        result=dict(physical_execution=False,world_tcp_mm_deg=target.tolist(),base_tcp_mm_deg=base.tolist())
        try:
            q=model.inverse(base,arm.home_joints)
            result.update(status='IK_OK',joints_deg=q.tolist(),error_mm_deg=pose_error(model.forward(q),base))
        except Exception as exc:result.update(status='IK_REJECTED',error=str(exc),candidate=getattr(exc,'ik_candidate',None))
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    report=build_capabilities(config,models);text=render_capabilities(report)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)
        args.output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(args.output)
    else:print(text)


if __name__=='__main__':main()
