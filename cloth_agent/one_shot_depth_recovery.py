"""Bounded Arm6 view recovery for missing Pa depth, with fresh contextual plans."""
from pathlib import Path
import shutil
import time

from .dual_arm.observation import Observation, DepthUnavailableError
from .experiments import arm6_depth_recovery as views
from .experiments import one_shot_goal_planner_probe_v1 as io
from . import restricted_context_one_shot as planner, preference_one_shot as checks
from .pre_action_gate import PreActionGate
from .unfolding_planners import ShadowPlannerRouter


def recover(runtime,root,folder,observation,plan,error,headers,timeout_s,gate_timeout_s,
            direct,prepare,previous_context,timing,*,history=None):
    """Called only for missing measured Pa depth before dispatch; no grasp retries."""
    recovery=folder/'depth_recovery'
    # Archive the original proposal before publishing a new view/plan as canonical.
    recovery.mkdir(exist_ok=False)
    original=recovery/'original';original.mkdir()
    for path in list(folder.iterdir()):
        if path==recovery:continue
        if path.is_dir():shutil.copytree(path,original/path.name)
        else:shutil.copyfile(path,original/path.name)
    initial_model_s=io.read(folder/'model_metrics.json')['wall_latency_s']
    new_model_s=0.;selected={};active_attempt=None
    began=time.monotonic()
    timing['depth_recovery_host_s']=0.
    timing['depth_recovery_model_calls']=0

    def reground(fresh,attempt):
        nonlocal active_attempt,new_model_s
        active_attempt=attempt
        shutil.copytree(fresh.directory,attempt/'before')
        try:
            new_plan,_=planner.plan_round(root,attempt,fresh,headers,timeout_s,direct,
                                         previous_context,view_change=True)
        finally:
            timing['depth_recovery_model_calls']+=1
            if (attempt/'model_metrics.json').exists():
                new_model_s+=io.read(attempt/'model_metrics.json').get('wall_latency_s',0.)
            # Until promotion, the main metrics still describe the original request.
            timing['depth_recovery_prior_planning_s']=new_model_s
        selected.update(plan=new_plan,attempt=attempt)
        return new_plan

    def prepare_fresh(fresh,new_plan):
        if fresh is observation:raise error
        attempt=active_attempt
        if new_plan['decision']!='ACTION':
            io.save(attempt/'gate_status.json',dict(status='NOT_APPLICABLE',mode='SHADOW',
                    reason=new_plan['decision'],affects_execution=False,model_called=False))
            io.save(attempt/'gate_result.json',dict(decision='UNCERTAIN'))
            return None
        if history:
            from .experiments.trajectory_memory_full_episode_v1 import repeated_action
            from .dual_arm.geometry import DualArmError
            if repeated_action(new_plan,fresh.image,history):
                raise DualArmError('unchanged observation with repeated same action after view recovery; no redispatch')
        router=ShadowPlannerRouter(attempt,recovery_enabled=False);router.accept_fast(new_plan)
        gate=PreActionGate(attempt,headers,direct,gate_timeout_s)
        try:router.shadow(gate,fresh.image,root/'goal/target.png')
        finally:timing['gate_wall_s']+=gate.metrics.get('gate_wall_s',0.)
        started=time.monotonic()
        try:
            checked=runtime.capture_current(attempt/'pre_execution_observation')
            checks.verify_scene(fresh,checked,new_plan,attempt,automatic=True)
            # Never recover arbitrary planner/IK/target errors as missing Pa depth.
            checked.sample(new_plan['grasp_px'],None)
            try:prepared=prepare(runtime,checked,new_plan,attempt)
            except DepthUnavailableError as exc:
                # Pa passed above; a later missing-depth error is not an eligible grasp retry.
                from .dual_arm.geometry import DualArmError
                raise DualArmError('non-grasp depth failure during recovery: '+str(exc)) from exc
            selected['checked']=checked
            return prepared
        finally:timing['depth_recovery_host_s']+=time.monotonic()-started

    try:
        fresh,new_plan,prepared=views.prepare_with_view_retry(runtime,observation,plan,
            recovery/'views',prepare_fresh,reground,distance_mm=20.)
        attempt=selected['attempt']
        # Preserve every attempt under depth_recovery; top-level artifacts match the final plan.
        for path in attempt.iterdir():
            if path.name in ('move','observation'):continue
            dest=folder/path.name
            if path.is_dir():shutil.copytree(path,dest,dirs_exist_ok=True)
            else:shutil.copyfile(path,dest)
        if prepared is None:
            # No current checked-depth observation is needed for DONE/ABSTAIN.
            shutil.copytree(fresh.directory,folder/'pre_execution_observation',dirs_exist_ok=True)
        timing['depth_recovery_prior_planning_s']=initial_model_s+new_model_s-io.read(folder/'model_metrics.json')['wall_latency_s']
        timing['planning_s']=initial_model_s+new_model_s
        timing['fast_planner_wall_s']=timing['planning_s']
        io.save(recovery/'summary.json',dict(status='REPLANNED',selected_attempt=str(attempt.relative_to(folder)),
            offset_mm=20,max_views=2,model_calls=timing['depth_recovery_model_calls'],
            no_garment_action_during_recovery=True,previous_effect_counted_once=True,
            final_decision=new_plan['decision']))
        if (folder/'progress_stop.json').exists():
            progress=io.read(folder/'progress_stop.json')
            progress.update(assessment_view='depth_recovery/original/before/rgb.png',
                            recovery_assessments_counted=False,final_decision=new_plan['decision'])
            io.save(folder/'progress_stop.json',progress)
        return (Observation.load(folder/'before'),Observation.load(folder/'pre_execution_observation'),
                new_plan,prepared)
    finally:
        timing['depth_recovery_wall_s']=time.monotonic()-began
        timing['planning_s']=initial_model_s+new_model_s
        timing['fast_planner_wall_s']=timing['planning_s']
