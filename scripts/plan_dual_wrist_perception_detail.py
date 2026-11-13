"""Offline candidate search for calibrated dual wrist RGB coverage. No robot IO."""
import sys,json,itertools
from pathlib import Path
import numpy as np,cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.kinematics import ArmModel
from cloth_agent.robot_collision import RobotCollision
from cloth_agent.experiments.full_grasp_adapter import build_robot_config
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/dual_wrist_perception_detail_20261110'

def main():
 OUT.mkdir(exist_ok=True)
 cfg=DualConfig.load_geometry(ROOT/'reports/dual_wrist_home_20261110/motion_config.json',root=ROOT)
 frames=json.loads((ROOT/'config/calibration/dual_wrist_20261110/frames.json').read_text())
 models={k:ArmModel(a) for k,a in cfg.arms.items()};home={k:a.home_joints.copy() for k,a in cfg.arms.items()}
 eye={k:np.array(v['eef_from_color_mm']) for k,v in frames['arms'].items()}
 sessions={'left':'reports/realsense_263622075308_calibration_20261110/manual_01/session.json','right':'reports/wrist_handeye_20261110/manual_02/session.json'}
 K={k:np.array(json.loads((ROOT/p).read_text())['K']) for k,p in sessions.items()}
 # Existing reviewed ZED tabletop polygon, intersected with assumed z=40 mm.
 roi=json.loads((ROOT/'config/topdown.unfold_zed.json').read_text())
 fusion=json.loads((ROOT/'config/fusion.unfold_dual.json').read_text());z=np.array(fusion['calibration']['world_from_zed_mm'])
 kz=np.array(fusion['calibration']['zed_K']);kz[:2]*=2/3 # calibration 1920x1080 -> ROI 1280x720
 uv=np.c_[np.array(roi['table_polygon_px']),np.ones(len(roi['table_polygon_px']))]
 rays=(uv@np.linalg.inv(kz).T)@z[:3,:3].T;scale=(40-z[2,3])/rays[:,2];poly=z[:3,3]+scale[:,None]*rays
 if np.any(scale<=0):raise ValueError('Table polygon ray does not meet plane')
 xx,yy=np.meshgrid(np.arange(poly[:,0].min(),poly[:,0].max(),15),np.arange(poly[:,1].min(),poly[:,1].max(),15));points=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,40.)]
 contour=poly[:,:2].astype(np.float32);points=points[[cv2.pointPolygonTest(contour,tuple(p[:2]),False)>=0 for p in points]]
 # Representative garment region: saved goal mask, dilated 40 mm in XY.
 goal_dir=ROOT/'data/reference/pink_kirby_goal_20261109'
 meta=json.loads((goal_dir/'view.json').read_text());mask=cv2.imread(str(goal_dir/'goal.mask.png'),0)>0
 pad=int(np.ceil(40/meta['mm_per_pixel']));mask=cv2.dilate(mask.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*pad+1,2*pad+1)))
 uv_goal=np.c_[320+(points[:,0]-meta['center_world_xy_mm'][0])/meta['mm_per_pixel'],320-(points[:,1]-meta['center_world_xy_mm'][1])/meta['mm_per_pixel']].round().astype(int)
 inside=(uv_goal[:,0]>=0)&(uv_goal[:,0]<640)&(uv_goal[:,1]>=0)&(uv_goal[:,1]<640)
 garment=np.zeros(len(points),bool);garment[inside]=mask[uv_goal[inside,1],uv_goal[inside,0]]>0
 if garment.sum()<20:raise ValueError('Representative garment not inside table model')
 np.save(OUT/'garment_region.npy',garment)
 def detail(key,t,visible):
  c=(points-t[:3,3])@t[:3,:3];z=c[:,2];rot=t[:3,:3];fx,fy=K[key][0,0],K[key][1,1]
  j=np.empty((len(c),2,2))
  for axis in (0,1):
   j[:,0,axis]=fx*(rot[axis,0]*z-c[:,0]*rot[axis,2])/z**2
   j[:,1,axis]=fy*(rot[axis,1]*z-c[:,1]*rot[axis,2])/z**2
  density=np.linalg.svd(j,compute_uv=False)[:,-1]
  valid=garment&visible
  return {'garment_coverage':float(visible[garment].mean()),'garment_px_per_mm_p10':float(np.percentile(density[valid],10)) if valid.any() else 0.,'garment_px_per_mm_median':float(np.median(density[valid])) if valid.any() else 0.,'tilt_from_vertical_deg':float(np.degrees(np.arccos(np.clip(-t[2,2],-1,1)))),'height_above_plane_mm':float(t[2,3]-40)},density
 def camera(k,q):return cfg.arms[k].world_from_base@models[k].frame_at(q,'link_eef')@eye[k]
 def visible(k,q):
  t=camera(k,q);c=(points-t[:3,3])@t[:3,:3];uv=c@K[k].T;uv=uv[:,:2]/np.maximum(c[:,2:],1e-9)
  mask=(c[:,2]>250)&(c[:,2]<1500)&(uv[:,0]>=12)&(uv[:,0]<1268)&(uv[:,1]>=12)&(uv[:,1]<708)
  return mask,t
 rng=np.random.default_rng(20261110);candidates={}
 for key in models:
  choices=[home[key]]
  # Saved manually sampled poses add known-camera-view alternatives.
  manual=ROOT/Path(sessions[key]).parent
  for p in sorted(manual.glob('point_*/observation.json')):
   choices.append(np.array(json.loads(p.read_text())['robot_before']['joints_deg']))
  choices.extend(home[key]+rng.uniform(-1,1,(12000,7))*[45,60,40,60,45,65,50])
  rows=[]
  for q in choices:
   if np.any(np.radians(q)<models[key].lower) or np.any(np.radians(q)>models[key].upper):continue
   pose=models[key].forward(q)
   try:cfg.arms[key].validate_point(pose[:3])
   except Exception:continue
   mask,t=visible(key,q)
   if t[2,3]<400 or t[2,3]>1100 or t[2,2]>-.35:continue
   metrics,density=detail(key,t,mask)
   rows.append({'detail':metrics,'density':density,'q':q,'mask':mask,'t':t,'coverage':float(mask.mean()),'distance':float(np.linalg.norm(q-home[key]))})
  candidates[key]=rows
  print(key,'feasible individual candidates',len(rows),flush=True)
 pairs=[]
 for primary in ('left','right'):
  other='right' if primary=='left' else 'left'
  main=[r for r in candidates[primary] if r['detail']['garment_coverage']>=.98 and r['detail']['tilt_from_vertical_deg']<40]
  main.sort(key=lambda r:r['detail']['garment_px_per_mm_p10']-.004*r['detail']['tilt_from_vertical_deg'],reverse=True)
  # Sample secondaries across both detailed close views and broad table views.
  supplements=sorted(candidates[other],key=lambda r:r['coverage'],reverse=True)[:120]+sorted(candidates[other],key=lambda r:r['detail']['garment_px_per_mm_p10'],reverse=True)[:120]
  center=points[garment].mean(axis=0)
  for m in main[:100]:
   for n in supplements:
    l,r=(m,n) if primary=='left' else (n,m)
    union=l['mask']|r['mask'];overlap=l['mask']&r['mask']
    if union.mean()<.96:continue
    a=m['t'][:3,3]-center;b=n['t'][:3,3]-center
    separation=float(np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))
    if separation<20:continue
    added=float((n['mask']&~m['mask']).mean())
    score=1.5*m['detail']['garment_px_per_mm_p10']-.008*m['detail']['tilt_from_vertical_deg']+union.mean()+.4*added+.15*min(separation,60)/60-.0003*(m['distance']+n['distance'])
    pairs.append((score,l,r,primary,separation,added))
 pairs.sort(key=lambda x:x[0],reverse=True);rejected=[];accepted=[]
 for score,l,r,primary,separation,added in pairs[:250]:
  if len(accepted)>=3:break
  if any(np.linalg.norm(l['q']-np.array(a['joints_deg']['left']))+np.linalg.norm(r['q']-np.array(a['joints_deg']['right']))<12 for a in accepted):continue
  target={'left':l['q'],'right':r['q']};current={k:q.copy() for k,q in home.items()};paths=[]
  try:
   for key in ('left','right'):
    checker=RobotCollision(ROOT/'reports/dual_wrist_home_20261110/motion_config.json',current,key,root=ROOT)
    p=models[key].forward(target[key]);robot=build_robot_config(cfg.raw,{},key,*p[3:5],config_path=ROOT/'reports/dual_wrist_home_20261110/motion_config.json')
    goal={k:q.copy() for k,q in current.items()};goal[key]=target[key]
    path=checker.refine_commanded_path([current,goal],robot,.2);paths.append({'arm':key,'path':[{k:q.tolist() for k,q in row.items()} for row in path]});current=goal
   accepted.append({'primary_arm':primary,'secondary_arm':'right' if primary=='left' else 'left','view_separation_deg':separation,'secondary_added_table_fraction':added,'detail_metrics':{'left':l['detail'],'right':r['detail']},'joints_deg':{k:q.tolist() for k,q in target.items()},'RGB_coverage':{'left':l['coverage'],'right':r['coverage'],'union':float((l['mask']|r['mask']).mean()),'overlap':float((l['mask']&r['mask']).mean())},'world_from_color_mm':{k:camera(k,target[k]).tolist() for k in models},'paths_from_home':paths,'status':'OFFLINE_PATH_CLEAR_NOT_EXECUTED'})
  except Exception as e:rejected.append(str(e))
 home_masks={k:visible(k,home[k])[0] for k in models}
 result={'status':'CANDIDATES' if accepted else 'NO_CLEAR_CANDIDATE','common_frame':'right_arm_base','table_polygon_world_mm':poly.tolist(),'table_plane_z_mm':40,'table_source':'Reviewed fixed-ZED ROI projected to assumed z=40 mm plane; not a new table measurement','sampling_step_mm':15,'home_joints_deg':{k:q.tolist() for k,q in home.items()},'home_RGB_coverage':{**{k:float(m.mean()) for k,m in home_masks.items()},'union':float((home_masks['left']|home_masks['right']).mean()),'overlap':float((home_masks['left']&home_masks['right']).mean())},'candidates':accepted,'rejected_path_count':len(rejected),'rejected_examples':rejected[:5],'objective':'Primary complete representative garment with higher worst-direction pixel density; secondary adds table coverage and angular diversity.','garment_roi_source':'data/reference/pink_kirby_goal_20261109/goal.mask.png with 40 mm margin','limitations':['Not evaluated on current live garment; angular diversity is not proof of seeing hidden cloth','RGB field-of-view estimate, not guaranteed valid depth','Coverage before mesh occlusion evaluation','Table z=40 mm is an assumption','Paths checked with conservative camera/support envelopes; no robot motion']}
 (OUT/'candidates.json').write_text(json.dumps(result,indent=2));np.save(OUT/'table_samples_mm.npy',points)
 print(json.dumps({k:result[k] for k in ['status','home_RGB_coverage','rejected_path_count']}));print([{k:x[k] for k in ['primary_arm','RGB_coverage','detail_metrics','view_separation_deg','secondary_added_table_fraction']} for x in accepted])
if __name__=='__main__':main()
