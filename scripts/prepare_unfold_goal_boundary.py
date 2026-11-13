"""Offline goal silhouette preparation. No camera or robot connections."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
from cloth_agent.grounded_sam2 import segment_file,close_worker
from cloth_agent.experiments.unfold_goal_boundary import digest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goal',type=Path,required=True)
    parser.add_argument('--view',type=Path,required=True)
    args=parser.parse_args();meta=json.loads(args.view.read_text(encoding='utf-8'))
    if Image.open(args.goal).size!=(640,640):raise ValueError('expected canonical 640x640 topdown goal')
    try:mask,report=segment_file(args.goal,args.goal.with_suffix('.mask.png'))
    finally:close_worker()
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:raise ValueError('no goal silhouette')
    contour=max(contours,key=cv2.contourArea)
    contour=cv2.approxPolyDP(contour,1.,True).reshape(-1,2)
    world=(contour-320)*float(meta['mm_per_pixel'])*np.array([1.,-1.])+meta['center_world_xy_mm']
    data=dict(goal_sha256=digest(args.goal),world_xy_mm=world.tolist(),view=str(args.view.resolve()),segmentation=report)
    args.goal.with_suffix('.boundary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    rgb=np.array(Image.open(args.goal).convert('RGB'));cv2.polylines(rgb,[contour],True,(30,240,60),1)
    Image.fromarray(rgb).save(args.goal.with_suffix('.boundary.png'))
    print(args.goal.with_suffix('.boundary.png'))

if __name__=='__main__':main()
