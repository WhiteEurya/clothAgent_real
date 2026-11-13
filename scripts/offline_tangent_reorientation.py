"""Saved round007 geometry only: choose a tangential sign using cross products."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image,ImageDraw
from cloth_agent.contour_normal_grasp import geometry,NormalGraspError

SOURCE=Path('reports/offline_pose_guidance_20261108')
ROUND=Path('reports/arm7_unfold_20261108_160630_770586/round_007')
GOAL=Path('reports/target_capture_20261107T163224250153Z')
OUT=Path('reports/round007_tangent_reorientation_20261108')

def cross(a,b):return float(a[0]*b[1]-a[1]*b[0])
def nearest(point,polygon):
 a=polygon;b=np.roll(a,-1,axis=0);ab=b-a
 u=np.clip(np.sum((point-a)*ab,axis=1)/np.maximum(np.sum(ab*ab,axis=1),1e-12),0,1)
 q=a+u[:,None]*ab;i=np.argmin(np.linalg.norm(q-point,axis=1));return q[i],float(np.linalg.norm(q[i]-point))

def main():
 OUT.mkdir(exist_ok=True)
 mask=np.array(Image.open(SOURCE/'current_mask.png'))>0;goal=np.array(Image.open(SOURCE/'goal_mask.png'))>0
 count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
 component_areas=stats[1:,cv2.CC_STAT_AREA].tolist();mask=labels==(1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])))
 cm=json.loads((ROUND/'topdown/view.json').read_text());gm=json.loads((GOAL/'topdown/view.json').read_text());scale=cm['mm_per_pixel'];center=np.array(cm['center_world_xy_mm'])
 toworld=lambda p:(np.asarray(p)-320)*scale*[1,-1]+center
 topixel=lambda p:(np.asarray(p)-center)/scale*[1,-1]+320
 contour=max(cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0],key=cv2.contourArea)[:,0]
 gc=max(cv2.findContours(goal.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0],key=cv2.contourArea)[:,0]
 gw=(gc-320)*gm['mm_per_pixel']*[1,-1]+gm['center_world_xy_mm'];gp=topixel(gw)
 y,x=np.nonzero(mask);O=toworld([x.mean(),y.mean()]);candidates=[]
 for c in contour:
  cw=toworld(c)
  if cv2.pointPolygonTest(gp.astype('float32').reshape(-1,1,2),tuple(map(float,c)),False)>=0:continue
  q,dist=nearest(cw,gw);candidates.append((dist,c,q))
 candidates.sort(key=lambda z:-z[0]);failures=[];selected=None
 for dist,c,q in candidates:
  try:g=geometry(mask,c,pull_px=40,max_snap_px=1)
  except NormalGraspError as exc:
   failures.append(dict(C_px=c.tolist(),outside_mm=dist,reason=str(exc)));continue
  C=toworld(g['origin_px']);E=toworld(g['intersection_px']);G=toworld(g['grasp_px']);t=np.array(g['tangent_unit_xy'])*[1,-1]
  a=cross(C-O,q-C);b=cross(E-O,t);bg=cross(G-O,t)
  if abs(a)<1e-6 or abs(b)<1e-6 or abs(bg)<1e-6:
   failures.append(dict(C_px=c.tolist(),outside_mm=dist,reason='degenerate cross product'));continue
  direction=t if a*b>0 else -t;Gpx=np.array(g['grasp_px']);dpx=direction*[1,-1]
  # 50mm arrow is illustrative only, not a validated release point.
  selected=dict(C_px=g['origin_px'],Q_px=topixel(q).tolist(),E_px=g['intersection_px'],G_px=g['grasp_px'],O_px=topixel(O).tolist(),outside_mm=dist,t_world_xy=t.tolist(),a_mm2=a,b_mm=b,b_at_inset_grasp_mm=bg,selected_direction_world_xy=direction.tolist(),selected_direction_image_xy=dpx.tolist(),expected_rotation='CCW' if a>0 else 'CW',tangent_variance_ratio=g['tangent_variance_ratio'],illustration_length_mm=50,illustration_target_px=(Gpx+50/scale*dpx).tolist(),normal_unit_image_xy=g['normal_unit_xy'],sign_consistent_at_inset_grasp=bool(b*bg>0))
  break
 report=dict(component_areas_px=component_areas,mask_cleanup='largest connected component',source_round=str(ROUND),source_masks=str(SOURCE),robot_calls=0,model_calls=0,goal_mapping='world XY; no centering or scaling alignment of garment',rule='farthest outside contour point first; skip only original tangent geometry failures; select tangent sign by cross product at E',extreme_point=dict(C_px=candidates[0][1].tolist(),outside_mm=candidates[0][0]),skipped_candidates=failures,selected=selected,limitations=['2D rigid-body torque proxy, not garment deformation prediction','illustrative 50mm arrow only; no depth, arm-partition, IK or path validation','existing tangent algorithm unchanged; new direction not executable by original normal-pull skill'])
 (OUT/'geometry.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 if selected:
  rgb=np.array(Image.open(SOURCE/'current.png').convert('RGB'));cv2.polylines(rgb,[np.rint(gp).astype('int32')],True,(40,240,60),1)
  im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);s=selected
  c=np.array(s['C_px']);e=np.array(s['E_px']);q=np.array(s['Q_px']);g=np.array(s['G_px']);d=np.array(s['selected_direction_image_xy'])
  draw.line([tuple(c),tuple(e)],fill='yellow',width=2);draw.line([tuple(c),tuple(q)],fill='orange',width=2)
  draw.line([tuple(c-35*d),tuple(c+35*d)],fill='magenta',width=2)
  def arrow(start,end,color):
   cv2.arrowedLine(rgb,tuple(np.rint(start).astype(int)),tuple(np.rint(end).astype(int)),color,2,tipLength=.22)
  rgb=np.array(im);arrow(g,g+50/scale*d,(0,255,255));arrow(g,g-50/scale*d,(150,150,150));im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
  for key,color in [('C','orange'),('Q','lime'),('E','yellow'),('G','red'),('O','white')]:
   p=np.array(s[key+'_px']);draw.ellipse((p[0]-3,p[1]-3,p[0]+3,p[1]+3),fill=color);draw.text((p[0]+5,p[1]-12),key,fill=color,stroke_width=1,stroke_fill='black')
  im.resize((1280,1280)).save(OUT/'geometry.png')
 print(json.dumps({k:v for k,v in report.items() if k!='skipped_candidates'},ensure_ascii=False));print('skipped',len(failures),failures[:2])
if __name__=='__main__':main()
