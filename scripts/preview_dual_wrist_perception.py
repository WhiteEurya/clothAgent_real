"""Saved-state dual gripper/camera viewer and conservative mesh visibility audit."""
import sys,json,time,argparse
from pathlib import Path
from functools import partial
import numpy as np,trimesh,yourdfpy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.kinematics import ArmModel
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/dual_wrist_perception_20261110'

def setup():
 cfg=DualConfig.load_geometry(ROOT/'reports/dual_wrist_home_20261110/motion_config.json',root=ROOT)
 frames=json.loads((ROOT/'config/calibration/dual_wrist_20261110/frames.json').read_text())
 models={k:ArmModel(v) for k,v in cfg.arms.items()};grip_path=ROOT/'assets/robots/xarm7/xarm7.urdf'
 grip=yourdfpy.URDF.load(grip_path,load_meshes=False,load_collision_meshes=False);keep={'link_eef'}
 for _ in range(12):
  for j in grip.robot.joints:
   if j.parent in keep:keep.add(j.child)
 meshes={}
 def extract(u,path,links):
  items=[]
  for name in links:
   for v in u.link_map[name].visuals:
    if v.geometry.mesh is None:continue
    src=v.geometry.mesh;mesh=trimesh.load(path.parent/src.filename,force='mesh')
    if src.scale is not None:mesh.apply_scale(src.scale)
    if v.origin is not None:mesh.apply_transform(v.origin)
    items.append((name,mesh))
  return items
 for key,model in models.items():meshes[key]=extract(model.urdf,model.config.urdf,model.urdf.link_map)
 gripper=extract(grip,grip_path,keep-{'link_eef'})
 return cfg,frames,models,grip,meshes,gripper

def make_scene(data,joints):
 cfg,frames,models,grip,meshes,gripper=data;items=[];eyes={};grip.update_cfg({'drive_joint':0.})
 for key in models:
  color=[65,155,245,255] if key=='left' else [245,155,55,255];model=models[key];model.update(joints[key]);base=cfg.arms[key].world_from_base.copy();base[:3,3]/=1000
  for link,src in meshes[key]:
   t=model.frame(link);t[:3,3]/=1000;m=src.copy();m.apply_transform(base@t);m.visual.face_colors=color;items.append((key+'/'+link,m))
  eef=model.frame('link_eef');eef[:3,3]/=1000;eef=base@eef
  for link,src in gripper:
   m=src.copy();m.apply_transform(eef@grip.get_transform(link,'link_eef'));m.visual.face_colors=[190,195,205,255];items.append((key+'/gripper/'+link,m))
  hand=np.array(frames['arms'][key]['eef_from_color_mm']);hand[:3,3]/=1000;eye=eef@hand;eyes[key]=eye
  body=trimesh.creation.box([.09,.025,.025]);body.apply_translation([0,0,-.0125]);body.apply_transform(eye);body.visual.face_colors=[80,85,90,255];items.append((key+'/camera',body))
  endpoint=hand[:3,3]+hand[:3,:3]@np.array([0,.0125,-.0125]);start=np.array([0,0,.003]);delta=endpoint-start
  support=trimesh.creation.box([.018,.005,np.linalg.norm(delta)]);t=trimesh.geometry.align_vectors([0,0,1],delta);t[:3,3]=(start+endpoint)/2;support.apply_transform(eef@t);support.visual.face_colors=[65,75,90,255];items.append((key+'/support',support))
 return items,eyes

def audit(data,report):
 points=np.load(OUT/'table_samples_mm.npy')/1000;cases=[('Home',report['home_joints_deg'])]+[(f'Candidate {i+1}',r['joints_deg']) for i,r in enumerate(report['candidates'])];result=[]
 sessions={'left':'reports/realsense_263622075308_calibration_20261110/manual_01/session.json','right':'reports/wrist_handeye_20261110/manual_02/session.json'}
 for name,q in cases:
  items,eyes=make_scene(data,q)
  # Convex arm links conservatively fill visual concavities; gripper geometry is exact URDF.
  occluders=[m if '/gripper/' in label or '/camera' in label or '/support' in label else m.convex_hull for label,m in items]
  scene=trimesh.util.concatenate(occluders);visible={};blocked={}
  for key,eye in eyes.items():
   k=np.array(json.loads((ROOT/sessions[key]).read_text())['K']);c=(points-eye[:3,3])@eye[:3,:3];uv=c@k.T;uv=uv[:,:2]/np.maximum(c[:,2:],1e-9)
   fov=(c[:,2]>.25)&(c[:,2]<1.5)&(uv[:,0]>=12)&(uv[:,0]<1268)&(uv[:,1]>=12)&(uv[:,1]<708);ids=np.flatnonzero(fov);directions=points[ids]-eye[:3,3];lengths=np.linalg.norm(directions,axis=1);directions/=lengths[:,None]
   loc,ray,_=scene.ray.intersects_location(np.repeat(eye[None,:3,3],len(ids),axis=0),directions,multiple_hits=True)
   obstruction=np.zeros(len(points),bool)
   if len(ray):
    dist=np.linalg.norm(loc-eye[:3,3],axis=1);valid=(dist>.002)&(dist<lengths[ray]-.005);obstruction[ids[ray[valid]]]=True
   visible[key]=fov&~obstruction;blocked[key]=float((fov&obstruction).mean())
  colors=np.full((len(points),3),[80,80,80],dtype=np.uint8);colors[visible['left']]=[65,155,245];colors[visible['right']]=[245,155,55];colors[visible['left']&visible['right']]=[75,220,120]
  slug=name.replace(' ','_');np.save(OUT/(slug+'_coverage_colors.npy'),colors)
  row={'name':name,'joints_deg':q,'coverage_after_model_occlusion':{k:float(v.mean()) for k,v in visible.items()},'union':float((visible['left']|visible['right']).mean()),'overlap':float((visible['left']&visible['right']).mean()),'occluded_fraction_of_table':blocked,'world_from_color_m':{k:v.tolist() for k,v in eyes.items()}}
  result.append(row);print(name,row['union'],row['overlap'],blocked,flush=True)
  if name!='Home':
   trimesh.util.concatenate([m for label,m in items]).export(OUT/(slug+'_robots.glb'))
 (OUT/'visibility_audit.json').write_text(json.dumps({'cases':result,'gripper_model':'assets/robots/xarm7/xarm7.urdf; drive_joint=0 nominal open, not measured jaw calibration','camera_and_support':'Approximate housing/support visuals; hand-eye optical transforms calibrated','occlusion_model':'Convex arm links and URDF gripper meshes; tabletop flat; no cloth occlusion','table_assumption':report['table_source'],'not_motion_authorization':True},indent=2))
 return result

def main():
 global OUT
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=OUT);p.add_argument('--serve',action='store_true');p.add_argument('--port',type=int,default=8094);args=p.parse_args();OUT=args.directory.resolve();data=setup();report=json.loads((OUT/'candidates.json').read_text())
 if not args.serve:audit(data,report);return
 import viser,viser.transforms as tf
 results=json.loads((OUT/'visibility_audit.json').read_text())['cases'];server=viser.ViserServer(host='127.0.0.1',port=args.port,label='双腕相机 perception · 离线模型');server.scene.set_up_direction('+z');server.gui.configure_theme(dark_mode=True)
 server.gui.add_markdown('## 双夹爪与双腕相机\n仅离线展示，没有机器人连接。蓝：237；橙：230；绿色点：双相机共同可见。\n夹爪复用 URDF，显示名义张开状态。相机外壳和支架为近似模型。')
 pick=server.gui.add_dropdown('姿态',options=[r['name'] for r in results],initial_value='Candidate 1');info=server.gui.add_markdown('');handles=[];points=np.load(OUT/'table_samples_mm.npy')/1000
 poly=np.array(report['table_polygon_world_mm'])/1000;server.scene.add_line_segments('/table/outline',points=np.stack([poly,np.roll(poly,-1,axis=0)],axis=1),colors=(180,180,180),line_width=2)
 server.scene.add_frame('/common',axes_length=.15,axes_radius=.003);server.scene.add_label('/common/label','right_arm_base · 公共坐标系',position=(0,0,.1))
 def render(_=None):
  for h in handles:h.remove()
  handles.clear();r=next(r for r in results if r['name']==pick.value);items,eyes=make_scene(data,r['joints_deg'])
  for label,m in items:handles.append(server.scene.add_mesh_trimesh('/robots/'+label,m))
  for key,t in eyes.items():
   handles.append(server.scene.add_frame('/optical/'+key,position=t[:3,3],wxyz=tf.SO3.from_matrix(t[:3,:3]).wxyz,axes_length=.09,axes_radius=.002))
   serial='263622075308' if key=='left' else '254522070945';kp=ROOT/('reports/realsense_263622075308_calibration_20261110/manual_01/session.json' if key=='left' else 'reports/wrist_handeye_20261110/manual_02/session.json');k=np.array(json.loads(kp.read_text())['K']);rays=np.array([[0,0,1],[1280,0,1],[1280,720,1],[0,720,1]])@np.linalg.inv(k).T;ends=t[:3,3]+(rays@t[:3,:3].T)*.65
   handles.append(server.scene.add_line_segments('/fov/'+key,points=np.stack([np.repeat(t[None,:3,3],4,axis=0),ends],axis=1),colors=(65,155,245) if key=='left' else (245,155,55),line_width=1))
  handles.append(server.scene.add_point_cloud('/coverage',points=points,colors=np.load(OUT/(r['name'].replace(' ','_')+'_coverage_colors.npy')),point_size=.012))
  extra=''
  if r['name']!='Home':
   candidate=report['candidates'][int(r['name'].split()[-1])-1]
   if 'primary_arm' in candidate:
    arm=candidate['primary_arm'];metrics=candidate['detail_metrics'][arm]
    extra=f"主视角：{arm}；主体像素密度 P10：{metrics['garment_px_per_mm_p10']:.2f} px/mm；偏离垂直：{metrics['tilt_from_vertical_deg']:.1f}°。\n\n"
  info.content=extra+f"**{r['name']}**\n\n桌面模型联合可见：{r['union']:.1%}；双目重叠：{r['overlap']:.1%}。\n\n桌面为已有 ROI 投影到假设 Z=40 mm 平面，未包含衣服遮挡；不是实测深度覆盖。候选路径仅离线检查，未执行。"
 pick.on_update(render);render()
 @server.on_client_connect
 def connect(client):
  client.camera.position=(1.5,-1.65,1.4);client.camera.look_at=(.53,-.3,.3);client.camera.up_direction=(0,0,1)
 print(f'VIEWER http://127.0.0.1:{args.port}',flush=True)
 while True:time.sleep(1)
if __name__=='__main__':main()
