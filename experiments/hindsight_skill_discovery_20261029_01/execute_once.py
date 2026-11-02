"""One explicitly authorized physical trial; durable one-dispatch guard."""
from pathlib import Path
import sys,json,fcntl,hashlib,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'online_corner_reveal_experiment01_20261029'))
from scene_runtime import CheckedRuntime,ROOT
from binary_check import execution_permission
from cloth_agent.codex_native_runtime import NativeRuntime
from cloth_agent.dual_arm.cli import connect
from cloth_agent.dual_arm.execution import DualArmCoordinator
from cloth_agent.dual_arm.observation import Observation
from cloth_agent.gamepad_collect import staged_place_release, TrajectoryRecorder
from cloth_agent.experiments.trajectory_memory_full_episode_v1 import ExecutionImages
from cloth_agent.preference_one_shot import verify_scene
from cloth_agent import codex_recovery
R=ROOT/'reports/hindsight_skill_discovery_20261029_01';P=R/'planned_action_01'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def main():
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--execute',action='store_true');args=a.parse_args()
 if not args.execute:raise SystemExit('Explicit --execute required')
 assert execution_permission(json.loads((P/'result.json').read_text()))=='YES'
 if (R/'dispatch_intent.json').exists():raise RuntimeError('Action already dispatched; no retry')
 out=R/'live_execution';out.mkdir(exist_ok=False)
 recorder=None;images=None
 rt=CheckedRuntime(ROOT/'config/dual_arm.local.json');status={'physical_actions':0,'status':'PREFLIGHT'}
 try:
  rt._lease=open('/tmp/clothagent_codex_dual_arm.lock','w');fcntl.flock(rt._lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  connections=connect(rt.config,rt.models,rt.cancel,True)
  rt.coordinator=DualArmCoordinator(rt.config,rt.models,connections,rt.cancel,out/'controller',realtime=True)
  before=Observation.load(R/'before_capture/observation')
  initial={k:np.asarray(v) for k,v in before.meta['joints_deg'].items()}
  codex_recovery.check_stationary(rt,before.meta['joints_deg'],out/'stationary_check.json')
  fresh=rt.capture_current(out/'pre_dispatch')
  plan=json.loads((P/'plan.json').read_text());verify_scene(before,fresh,plan,out,automatic=False)
  source=(P/'motion.py').read_text()
  # Include safe return for a comparable post-release observation, no second grasp.
  source+='\nmotion.move_joints(left='+repr(np.radians(initial['left']).tolist())+')\n'
  (out/'motion_with_observation_return.py').write_text(source)
  ops=rt.plan(source,initial,preserve_command_boundaries=True)
  opens=[i for i,(k,v) in enumerate(ops) if k=='gripper' and v['action']=='open'];assert len(opens)==2
  i=opens[-1];ops=ops[:i]+staged_place_release(ops[i:])
  assert sum(k=='gripper' and v['action']=='close' for k,v in ops)==1
  assert all(v.arm=='left' if k=='move' else v['arm']=='left' for k,v in ops)
  save(out/'approved_execution_plan.json',[{'kind':k,'data':v.to_dict() if k=='move' else v} for k,v in ops])
  rt.arrival(initial)
  if getattr(rt,'auto_grasp_branch',None):raise RuntimeError('Automatic retry branch must be disabled')
  recorder=TrajectoryRecorder(rt,out);recorder.start();recorder.stage('one_hindsight_exploration');recorder.check()
  images=ExecutionImages(rt,out/'execution_images');images.start()
  with (R/'dispatch_intent.json').open('x') as f:json.dump({'time':time.time(),'max_actions':1,'user_authorized':True,'minimum_physical_tool_z_mm':-14.9,'prediction_sha256':hashlib.sha256((R/'prediction_before_execution.json').read_bytes()).hexdigest(),'before_sha256':hashlib.sha256(before.image.read_bytes()).hexdigest(),'no_retry':True},f)
  status.update(physical_actions=1,status='DISPATCHED')
  save(out/'status.json',status)
  result=NativeRuntime._execute_operations(rt,ops,out)
  save(out/'execution.json',result)
  status['status']=result['status']
  if result['status']!='EXECUTED':raise RuntimeError('Execution incomplete; no retry')
  recorder.check();images.close();images=None;recorder.close();recorder=None
  after=rt.capture_current(out/'after');status['after']=str(after.image)
 except BaseException as e:
  status.update(status='STOPPED',error=f'{type(e).__name__}: {e}')
  import traceback
  (out/'error.txt').write_text(traceback.format_exc())
 finally:
  if images is not None:images.close()
  if recorder is not None:recorder.close()
  save(out/'status.json',status);rt.close();print(json.dumps(status,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
