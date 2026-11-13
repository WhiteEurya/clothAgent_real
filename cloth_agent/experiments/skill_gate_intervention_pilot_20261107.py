"""Four-run online intervention pilot: one fixed Simple/Tangent intervention per run."""
from pathlib import Path
import argparse, copy, hashlib, json, signal, traceback
from datetime import datetime, timezone
import numpy as np

from .discovery_data import read, save, file_hash
from .skill_gate_online import ROOT

CONFIG = ROOT/'config/skill_gate_online_pilot01.json'
RUNS = [
    dict(run_id=1, phase='EARLY', intervention_round=2, intervention_skill='SIMPLE', reset_seed=2026110701),
    dict(run_id=2, phase='EARLY', intervention_round=2, intervention_skill='TANGENT', reset_seed=2026110702),
    dict(run_id=3, phase='LATE', intervention_round=8, intervention_skill='SIMPLE', reset_seed=2026110703),
    dict(run_id=4, phase='LATE', intervention_round=8, intervention_skill='TANGENT', reset_seed=2026110704),
]
PROBABILITIES={'SIMPLE':0.7,'TANGENT':0.3}

def validate_target():
    cfg=read(CONFIG); goal=ROOT/cfg['goal_image']
    if cfg.get('garment_id')!='black_mario': raise ValueError('wrong garment id')
    if not goal.is_file() or file_hash(goal)!=cfg.get('goal_sha256'): raise ValueError('goal hash mismatch')
    manifest=read(goal.parent/'manifest.json')
    if manifest.get('garment_id')!='black_mario' or manifest.get('sha256')!=cfg['goal_sha256']: raise ValueError('goal manifest mismatch')
    perception=read(ROOT/'config/perception.zed_dual_xarm7_20261103.json')
    fusion=read(ROOT/'config/fusion.unfold_dual.json')
    if perception.get('active_cameras')!=['A'] or perception.get('table_plane_mode')!='background_fit': raise ValueError('unexpected perception configuration')
    return cfg,goal,perception,fusion

def choose_skill(step, plan):
    # Only the planned intervention differs from the normal Simple policy.
    return plan['intervention_skill'] if step==plan['intervention_round'] else 'SIMPLE'

class InterventionRequest:
    def __init__(self,root,plan,runtime,headers,timeout,direct):
        self.root=Path(root); self.plan=plan; self.runtime=runtime; self.headers=headers; self.timeout=timeout; self.direct=direct
        if not (self.root/'goal/target.png').is_file(): raise ValueError('missing explicit goal')
    def __call__(self,folder,before,context=None):
        from .partitioned_unfold import plan_round
        from .unfold_skill_planner import abstain
        folder=Path(folder); step=int(next(p.name[6:] for p in (folder,*folder.parents) if p.name.startswith('round_')))
        skill=choose_skill(step,self.plan); intervention=(step==self.plan['intervention_round'])
        record=dict(protocol='four_run_intervention_v1',run_id=self.plan['run_id'],phase=self.plan['phase'],round=step,
                    intervention_round=self.plan['intervention_round'],intervention_skill=self.plan['intervention_skill'],
                    intervention=intervention,selection_source='PRESET_INTERVENTION' if intervention else 'NORMAL_SIMPLE_POLICY',
                    selected_skill=skill,selection_reason_is_autonomous=False,
                    observation_id=getattr(before,'meta',{}).get('observation_id'),before_rgb=str(before.image),before_sha256=file_hash(before.image),
                    goal_sha256=file_hash(self.root/'goal/target.png'),planning_status='PENDING',planning_success=None,model_requests=0,
                    availability={'SIMPLE':{'status':'NOT_EVALUATED','reason':'not selected'} if skill=='TANGENT' else {'status':'SELECTED_NORMAL','reason':'normal Simple before/after intervention'},
                                  'TANGENT':{'status':'NOT_EVALUATED','reason':'not selected'} if skill=='SIMPLE' else {'status':'SELECTED_INTERVENTION','reason':'preset intervention'}},
                    action=None,execution={'status':'NOT_DISPATCHED'},intervention_outcome='PENDING')
        path=folder/'intervention.json'; save(path,record,immutable=True)
        calls=0
        def counted(destination,body,headers,timeout):
            nonlocal calls
            calls+=1
            if calls>1: raise RuntimeError('one-shot request budget exhausted')
            return self.direct(destination,body,headers,timeout)
        try:
            plan_value,metrics=plan_round(self.root,folder,before,self.headers,self.timeout,counted,runtime=self.runtime,forced_skill=skill)
            record.update(action=plan_value,planning_status=plan_value['decision'],abstain_reason=plan_value.get('short_reason') if plan_value['decision']=='ABSTAIN' else None)
            record['availability'][skill].update(status='CANDIDATE_GENERATED' if plan_value['decision']=='ACTION' else 'NO_ACTION',reason=plan_value.get('short_reason'))
            if intervention and plan_value['decision']!='ACTION': record['intervention_outcome']='ABSTAIN_OR_DONE'
            return plan_value,metrics
        except Exception as exc:
            reason=f'{type(exc).__name__}: {exc}'; value=abstain(reason)
            record.update(action=value,planning_status='ABSTAIN',abstain_reason=reason,failure_kind='PLANNING_EXCEPTION',intervention_outcome='ABSTAIN_OR_DONE' if intervention else 'NORMAL_STEP_ERROR')
            record['availability'][skill]={'status':'UNAVAILABLE','reason':reason}
            return value,{'wall_latency_s':None,'selection_attempts':calls,'error':reason}
        finally:
            record['model_requests']=calls; record['planning_success']=record['planning_status']=='ACTION'
            save(path,record)
            # retain the historical filename used by the online pilot for audit tools
            save(folder/'exploration.json',record)

def collect_run(root,plan):
    root=Path(root); summary=read(root/'episode_summary.json'); rows=[]
    for step in summary.get('steps',[]):
        folder=root/f"round_{step['round']:03d}"; rec=read(folder/'intervention.json')
        rec['execution']=step.get('execution',{'status':'NOT_DISPATCHED'}); rec['episode_status']=summary.get('status')
        rec['after_rgb']=str(folder/'after/rgb.png') if (folder/'after/rgb.png').exists() else None
        rec['next_normal_observation']=rec['after_rgb']
        rec['raw_images']={k:str(folder/k) for k in ('after/rgb.png','after/topdown/planning_rgb.png','topdown/planning_rgb.png') if (folder/k).exists()}
        rec['outcome_label']='UNKNOWN_DEFERRED_OFFLINE'
        save(folder/'intervention.json',rec); save(folder/'exploration.json',rec); rows.append(rec)
    # If the episode stopped before the preset round, preserve this explicitly.
    reached=any(r['round']==plan['intervention_round'] for r in rows)
    intervention_status='EXECUTED' if any(r['round']==plan['intervention_round'] and r['execution'].get('status')=='EXECUTED' for r in rows) else ('ABSTAIN' if any(r['round']==plan['intervention_round'] and r.get('planning_status')=='ABSTAIN' for r in rows) else ('EARLY_DONE_BEFORE_INTERVENTION' if not reached else 'NOT_EXECUTED'))
    save(root/'intervention_summary.json',dict(protocol='four_run_intervention_v1',run_id=plan['run_id'],phase=plan['phase'],intervention_round=plan['intervention_round'],intervention_skill=plan['intervention_skill'],intervention_status=intervention_status,episode_status=summary.get('status'),steps=rows,final_quality={'completion':'UNKNOWN','flatness':'UNKNOWN','wrinkles':'UNKNOWN','stacking':'UNKNOWN','reason':'independent offline evaluation pending'}))
    return rows

def execute_batch(out):
    from .partitioned_unfold import configure_runtime, PartitionedReplayRuntime, DEFAULT_CONFIG
    from .one_shot_goal_full_episode_v1 import direct_once
    from .one_shot_goal_planner_probe_v1 import auth_headers
    from cloth_agent.dual_cloud_observation import load_config
    from cloth_agent.unfold_video import UnfoldVideo
    from cloth_agent.perception import PerceptionConfig
    cfg,goal,perception_cfg,fusion=validate_target()
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    save(out/'protocol.json',dict(protocol='four_run_intervention_v1',runs=RUNS,order=[p['run_id'] for p in RUNS],max_actions=12,normal_policy='SIMPLE',one_planner_request_per_step=True,fallback_to_other_skill=False,goal_image=str(goal),goal_sha256=cfg['goal_sha256'],perception_config='config/perception.zed_dual_xarm7_20261103.json',fusion_config='config/fusion.unfold_dual.json',z_policy='existing runtime; no height experiment',no_gate_training=True,no_router_update=True),immutable=True)
    save(out/'batch_summary.json',dict(status='RUNNING',runs=[]))
    headers=auth_headers()
    for plan in RUNS:
        folder=out/f"run_{plan['run_id']:02d}"; (folder/'goal').mkdir(parents=True)
        from shutil import copyfile
        copyfile(goal,folder/'goal/target.png'); save(folder/'goal/manifest.json',dict(garment_id='black_mario',source=str(goal),sha256=cfg['goal_sha256']))
        radius=float(np.random.default_rng(plan['reset_seed']).uniform(20,40)); plan=dict(plan,radius_mm=radius)
        save(folder/'run_plan.json',plan,immutable=True)
        runtime=configure_runtime(PartitionedReplayRuntime(DEFAULT_CONFIG,planning_only=False))
        try:
            runtime.dual_cloud_config=load_config(ROOT/'config/fusion.unfold_dual.json')
            runtime.video_recorder=UnfoldVideo(PerceptionConfig.load(ROOT,ROOT/'config/perception.zed_dual_xarm7_20261103.json'),ROOT/'videos'/out.name/folder.name,folder); runtime.video_recorder.start()
            req=InterventionRequest(folder,plan,runtime,headers,900,direct_once)
            from .partitioned_unfold import episode,initialize
            def reset(before,d): return initialize(runtime,before,d,headers=headers,timeout=900,direct=direct_once,seed=plan['reset_seed'],radius_mm=radius)
            result=episode(runtime,folder,max_actions=12,request=req,reset_request=reset,max_plan_attempts=1)
            save(folder/'episode_summary.json',result)
        finally:
            try: runtime.close()
            except Exception: pass
        collect_run(folder,plan)
        print(json.dumps({'run_id':plan['run_id'],'status':read(folder/'episode_summary.json').get('status'),'intervention_status':read(folder/'intervention_summary.json').get('intervention_status')},ensure_ascii=False),flush=True)
        episode_status=read(folder/'episode_summary.json').get('status')
        summary=read(out/'batch_summary.json'); summary['runs'].append({'run_id':plan['run_id'],'status':episode_status,'intervention_status':read(folder/'intervention_summary.json').get('intervention_status')});
        if episode_status not in ('MODEL_REPORTED_DONE','ACTION_LIMIT','ABSTAIN'):
            summary.update(status='STOPPED',reason=f'run {plan["run_id"]} ended with {episode_status}; no further run allowed')
            save(out/'batch_summary.json',summary)
            break
        save(out/'batch_summary.json',summary)
    summary=read(out/'batch_summary.json')
    if summary.get('status') == 'RUNNING' and len(summary.get('runs',[])) == len(RUNS):
        summary['status']='COMPLETED_FOUR_RUNS'
    save(out/'batch_summary.json',summary)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--check',action='store_true'); ap.add_argument('--execute',action='store_true'); a=ap.parse_args()
    cfg,goal,perception,fusion=validate_target()
    if a.check:
        print(json.dumps({'status':'DRY_RUN_OK','goal':str(goal),'goal_sha256':file_hash(goal),'runs':RUNS,'active_cameras':perception.get('active_cameras'),'table_plane_mode':perception.get('table_plane_mode'),'robot_calls':0,'model_calls':0},ensure_ascii=False)); return 0
    if not a.execute: raise SystemExit('use --check or --execute')
    execute_batch(a.output); return 0
if __name__=='__main__': raise SystemExit(main())
