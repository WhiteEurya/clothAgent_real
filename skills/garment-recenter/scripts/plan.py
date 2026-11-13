"""Offline world-XY recenter candidate; never connects to hardware."""
import argparse
import json
from pathlib import Path
import numpy as np


def plan(centroid_xy_mm, table_center_xy_mm, collar_grasp_xy_mm):
    c=np.asarray(centroid_xy_mm,float)
    t=np.asarray(table_center_xy_mm,float)
    grasp=np.asarray(collar_grasp_xy_mm,float)
    if any(x.shape!=(2,) or not np.isfinite(x).all() for x in (c,t,grasp)):
        raise ValueError('finite world XY centers and collar grasp required')
    delta=t-c
    distance=float(np.linalg.norm(delta))
    if distance<1e-6:
        return dict(status='NO_TRANSLATION',center_delta_world_mm=delta.tolist())
    return dict(status='CANDIDATE_REQUIRES_EXISTING_VALIDATION',
                skill='garment-recenter',coordinate_frame='world_xy_mm',
                garment_center_xy_mm=c.tolist(),table_center_xy_mm=t.tolist(),
                center_delta_world_mm=delta.tolist(),distance_mm=distance,
                grasp_world_xy_mm=grasp.tolist(),target_world_xy_mm=(grasp+delta).tolist(),
                physical_execution=False,
                required_checks=['grasp_inside_cloth','visible_collar_area',
                                 'depth','workspace','full_path'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input',type=Path,help='JSON: centroid_xy_mm, table_center_xy_mm, collar_grasp_xy_mm')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=plan(**json.loads(args.input.read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
