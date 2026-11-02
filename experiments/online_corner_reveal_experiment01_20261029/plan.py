"""Offline safety planning from a fresh real observation. No live connections."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,traceback,argparse
import numpy as np
from PIL import Image,ImageDraw
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'corner_reveal_physical_E353_20261029'))
from scene_runtime import CheckedRuntime,ROOT
from binary_check import execution_permission
from cloth_agent.dual_arm.observation import Observation
from cloth_agent.dual_arm.geometry import matrix_pose,pose_matrix
from cloth_agent.gamepad_collect import automatic_program,staged_place_release
from cloth_agent.one_shot_orientation import resolve

BASE=ROOT/'reports/online_corner_reveal_experiment01_20261029'
OUT=BASE/'planned_action_01'

def save(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 global OUT
 parser=argparse.ArgumentParser();parser.add_argument("--search-approach",action="store_true");parser.add_argument("--observation",type=Path,default=BASE/"before_capture/observation");parser.add_argument("--output",type=Path);parser.add_argument("--grasp",nargs=2,type=int,default=[844,270]);parser.add_argument("--target",nargs=2,type=int,default=[951,320]);args=parser.parse_args()
 if args.search_approach: OUT=BASE/"planned_action_02"
 if args.output: OUT=args.output
 OUT.mkdir(exist_ok=False)
 result={'status':'PLANNING','physical_actions':0,'path_validation':True,'disabled_collision_envelopes':[]}
 try:
  obs=Observation.load(args.observation)
  rt=CheckedRuntime(ROOT/'config/dual_arm.local.json')
  if args.search_approach:rt.direct_joint_moves=False
  snapshot=json.loads((obs.directory/'observation.json').read_text())['snapshot_after']
  rt.coordinator=SimpleNamespace(snapshots=lambda:snapshot)
  K=yaml.safe_load((ROOT/'config/calibration/dual_arm_working_20261008/camA_intrinsics.yaml').read_text())['K']
  rt.camera=SimpleNamespace(intrinsics=np.asarray(K))
  proposal=json.loads((BASE/'prediction_before_execution.json').read_text())['proposal']
  assert proposal['decision']=='EXPLORATORY_ACTION'
  args.grasp=proposal['grasp_px'];args.target=proposal['target_px']
  plan={'decision':'ACTION','grasp_px':args.grasp,'target_px':args.target,'motion_type':'lift_pull_place','grasp_angle_deg':proposal['grasp_angle_deg'],'short_reason':'Test a newly grounded cuff reveal hypothesis.'}
  save('plan.json',plan)
  im=Image.open(obs.image).convert('RGB');d=ImageDraw.Draw(im);g=tuple(plan['grasp_px']);t=tuple(plan['target_px']);d.line([g,t],fill='red',width=5);d.ellipse((g[0]-7,g[1]-7,g[0]+7,g[1]+7),outline='yellow',width=3);d.ellipse((t[0]-6,t[1]-6,t[0]+6,t[1]+6),outline='red',width=3);im.save(OUT/'action_overlay.png')
  obs.validate_for(rt.config,live=True,check_age=False)
  start=obs.sample(g,rt.config.limits['max_surface_spread_mm']);target=obs.sample(t,rt.config.limits['max_surface_spread_mm'])
  distance=float(np.linalg.norm(target[:2]-start[:2]));assert 10<=distance<=150
  arm=rt.config.arms['left'];home=matrix_pose(arm.world_from_base@pose_matrix(rt.models['left'].forward(arm.home_joints)))[3:]
  rpy=resolve(rt,obs,plan,start,home,OUT)
  save('geometry.json',{'distance_mm':distance,'start_world_mm':start.tolist(),'target_world_mm':target.tolist(),'rpy_world_deg':rpy.tolist(),'nominal_lift_height_mm':20,'grasp_offset_mm':-15,'lift_reference':'max grasp/target surface Z + 20; not rise from closed TCP','placement_release':'partial 180 pulses then retreat','arm7_physically_present':False,'arm7_absence_source':'operator confirmation 2026-10-29'})
  initial={k:np.asarray(v['joints']) for k,v in snapshot.items()}
  # Current state is checked before any planned motion; failure is not bypassed.
  rt.arrival(initial)
  source=automatic_program(rt,start,target,rpy,approach_height=50,grasp_offset=-15,lift_height=20,clearance=30,place_offset=5)
  (OUT/'motion.py').write_text(source)
  operations=rt.plan(source,initial,preserve_command_boundaries=True)
  opens=[i for i,(kind,data) in enumerate(operations) if kind=='gripper' and data['action']=='open'];assert len(opens)==2
  i=opens[-1];operations=operations[:i]+staged_place_release(operations[i:])
  save('native_plan.json',[{'kind':k,'data':v.to_dict() if k=='move' else v} for k,v in operations])
  result.update(status='SAFETY_PLAN_PASSED_PENDING_HUMAN_APPROVAL',operation_count=len(operations),planning_stats=rt.plan_stats)
 except BaseException as e:
  result.update(status='SAFETY_PLAN_REJECTED_NO_MOTION',error=f'{type(e).__name__}: {e}')
  (OUT/'error.txt').write_text(traceback.format_exc())
 finally:
  result['execution_permission']=execution_permission(result)
  save('result.json',result)
  (OUT/'execution_permission.txt').write_text(result['execution_permission']+'\n')
  print(result['execution_permission'],flush=True)
if __name__=='__main__':main()
