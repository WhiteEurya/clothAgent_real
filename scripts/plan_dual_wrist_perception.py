"""Offline candidate search for calibrated dual wrist RGB coverage. No robot IO."""
import sys,json,itertools
from pathlib import Path
import numpy as np,cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.kinematics import ArmModel
from cloth_agent.robot_collision import RobotCollision
from cloth_agent.experiments.full_grasp_adapter import build_robot_config
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/dual_wrist_perception_20261110'

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
  choices.extend(home[key]+rng.uniform(-1,1,(900,7))*[20,30,18,30,22,30,22])
  rows=[]
  for q in choices:
   if np.any(np.radians(q)<models[key].lower) or np.any(np.radians(q)>models[key].upper):continue
   pose=models[key].forward(q)
   try:cfg.arms[key].validate_point(pose[:3])
   except Exception:continue
   mask,t=visible(key,q)
   if t[2,3]<400 or t[2,3]>1100 or t[2,2]>-.35:continue
   rows.append({'q':q,'mask':mask,'t':t,'coverage':float(mask.mean()),'distance':float(np.linalg.norm(q-home[key]))})
  rows.sort(key=lambda r:r['coverage']-.0002*r['distance'],reverse=True);candidates[key]=rows[:24]
  print(key,'candidates',len(rows),'best RGB coverage',rows[0]['coverage'],flush=True)
 pairs=[]
 for l,r in itertools.product(candidates['left'],candidates['right']):
  union=l['mask']|r['mask'];overlap=l['mask']&r['mask'];score=union.mean()+.12*overlap.mean()-.0001*(l['distance']+r['distance'])
  pairs.append((score,l,r))
 pairs.sort(key=lambda x:x[0],reverse=True);rejected=[];accepted=[]
 for score,l,r in pairs[:100]:
  if len(accepted)>=3:break
  target={'left':l['q'],'right':r['q']};current={k:q.copy() for k,q in home.items()};paths=[]
  try:
   for key in ('left','right'):
    checker=RobotCollision(ROOT/'reports/dual_wrist_home_20261110/motion_config.json',current,key,root=ROOT)
    p=models[key].forward(target[key]);robot=build_robot_config(cfg.raw,{},key,*p[3:5],config_path=ROOT/'reports/dual_wrist_home_20261110/motion_config.json')
    goal={k:q.copy() for k,q in current.items()};goal[key]=target[key]
    path=checker.refine_commanded_path([current,goal],robot,.2);paths.append({'arm':key,'path':[{k:q.tolist() for k,q in row.items()} for row in path]});current=goal
   accepted.append({'joints_deg':{k:q.tolist() for k,q in target.items()},'RGB_coverage':{'left':l['coverage'],'right':r['coverage'],'union':float((l['mask']|r['mask']).mean()),'overlap':float((l['mask']&r['mask']).mean())},'world_from_color_mm':{k:camera(k,target[k]).tolist() for k in models},'paths_from_home':paths,'status':'OFFLINE_PATH_CLEAR_NOT_EXECUTED'})
  except Exception as e:rejected.append(str(e))
 home_masks={k:visible(k,home[k])[0] for k in models}
 result={'status':'CANDIDATES' if accepted else 'NO_CLEAR_CANDIDATE','common_frame':'right_arm_base','table_polygon_world_mm':poly.tolist(),'table_plane_z_mm':40,'table_source':'Reviewed fixed-ZED ROI projected to assumed z=40 mm plane; not a new table measurement','sampling_step_mm':15,'home_joints_deg':{k:q.tolist() for k,q in home.items()},'home_RGB_coverage':{**{k:float(m.mean()) for k,m in home_masks.items()},'union':float((home_masks['left']|home_masks['right']).mean()),'overlap':float((home_masks['left']&home_masks['right']).mean())},'candidates':accepted,'rejected_path_count':len(rejected),'rejected_examples':rejected[:5],'limitations':['RGB field-of-view estimate, not guaranteed valid depth','Coverage before mesh occlusion evaluation','Table z=40 mm is an assumption','Paths checked with conservative camera/support envelopes; no robot motion']}
 (OUT/'candidates.json').write_text(json.dumps(result,indent=2));np.save(OUT/'table_samples_mm.npy',points)
 print(json.dumps({k:result[k] for k in ['status','home_RGB_coverage','rejected_path_count']}));print([x['RGB_coverage'] for x in accepted])
if __name__=='__main__':main()
