"""Read-only controller feedback + stationary RGB-D. Never enables/moves either arm.
Strict collision/path settings are local to this experiment, not formal policy edits.
"""
from pathlib import Path
import sys, json, fcntl, argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from cloth_agent.gamepad_collect_runtime import CollectionRuntime
from cloth_agent.dual_arm.cli import connect
from cloth_agent.dual_arm.execution import DualArmCoordinator

sys.path.insert(0,str(ROOT/'experiments/online_corner_reveal_experiment01_20261029'))
from scene_runtime import CheckedRuntime

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    runtime=None;result={'physical_actions':0,'controller_writes':0,'status':'PREFLIGHT_STARTED'}
    try:
        runtime=CheckedRuntime(ROOT/'config/dual_arm.local.json')
        runtime._lease=open('/tmp/clothagent_codex_dual_arm.lock','w')
        fcntl.flock(runtime._lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        connections=connect(runtime.config,runtime.models,runtime.cancel,True)
        runtime.coordinator=DualArmCoordinator(runtime.config,runtime.models,connections,runtime.cancel,args.output/'controller',realtime=True)
        state=runtime.coordinator.snapshots()
        (args.output/'live_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
        obs=runtime.capture_current(args.output/'observation')
        result.update(status='CAPTURED_NO_MOTION',image=str(obs.image),path_validation=True,disabled_collision_envelopes=[])
    except BaseException as exc:
        result.update(status='PREFLIGHT_FAILED_NO_MOTION',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if runtime is not None:runtime.close()
        (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
