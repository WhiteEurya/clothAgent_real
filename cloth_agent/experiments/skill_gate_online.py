"""Pre-planning randomization adapter; no hardware entry point or learned gate.

The caller supplies the existing runtime. This module does not start cameras,
robots, or a six-run batch on import. The online pilot requires an explicit goal.
"""
from pathlib import Path
import copy
import hashlib
import json

from .discovery_data import save, read, file_hash

ROOT=Path(__file__).resolve().parents[2]
PROBABILITIES={'SIMPLE':0.7,'TANGENT':0.3}


def assignment(seed,run_id,iteration):
    """Stateless reproducible draw; observations and outcomes cannot bias it."""
    if type(seed) is not int or seed<0:raise ValueError('nonnegative integer seed required')
    if not 1<=run_id<=6 or iteration<0:raise ValueError('pilot has exactly six run IDs')
    key=f'skill-gate-online-v1:{seed}:{run_id}:{iteration}'.encode()
    n=int.from_bytes(hashlib.sha256(key).digest()[:8],'big')>>11
    u=n/2**53
    skill='TANGENT' if u<0.3 else 'SIMPLE'
    return dict(seed=seed,run_id=run_id,iteration=iteration,draw=u,forced_skill=skill,
                selection_probability=PROBABILITIES[skill],probabilities=PROBABILITIES.copy(),
                algorithm='SHA256 keyed by fixed seed/run/iteration; first 53 bits uniform')


def plan_forced(root,folder,before,headers,timeout,direct,*,runtime,forced_skill):
    """Reuse existing candidate computation; exactly one model request, no fallback."""
    from PIL import Image
    from . import partitioned_unfold as unfold
    from . import one_shot_goal_planner_probe_v1 as io
    from .unfold_topdown import primary, PRIMARY_GUIDANCE
    from .unfold_planning_skills import workspace_prompt,planning_text
    from .unfold_skill_planner import run
    if forced_skill not in PROBABILITIES:raise ValueError('forced_skill must be SIMPLE or TANGENT')
    expected=None if forced_skill=='SIMPLE' else 'garment-tangent-grasp'
    def one_request(destination,payload,auth,seconds):
        nonlocal calls
        if calls:raise RuntimeError('one-shot request budget exhausted')
        calls+=1
        payload=copy.deepcopy(payload)
        payload['text']['format']['schema']['properties']['skill']=(
            {'type':'null'} if expected is None else
            {'anyOf':[{'type':'null'},{'type':'string','enum':[expected]}]})
        value,metrics=direct(destination,payload,auth,seconds)
        if value.get('decision')=='ACTION' and value.get('skill')!=expected:
            raise ValueError('model did not produce an action for assigned skill')
        return value,metrics
    calls=0
    if forced_skill=='SIMPLE':
        return unfold.plan_round_pure(root,folder,before,headers,timeout,one_request,
                                      runtime=runtime,max_attempts=1)
    view=primary(before,folder)
    raw=(ROOT/'skills/garment-tangent-grasp/SKILL.md').read_text(encoding='utf-8')
    method=planning_text('garment-tangent-grasp',raw).split('\n\n',1)[1]
    method=method.replace('每轮根据新图重新选择普通抓放或技能。','')
    body=io.payload(view.image,Path(root)/'goal/target.png',Image.open(view.image).size)
    body['instructions']=('第一张是当前衣物，第二张是展开目标。本步已在规划前随机分配 Tangent Skill。'
        '仅提出该技能的一步动作，不选择其他技能。无法提出有效候选则ABSTAIN；已展开则DONE。\n'
        +method+'\n'+PRIMARY_GUIDANCE)
    if runtime is not None:body['instructions']+='\n'+workspace_prompt(runtime,before,view,folder)
    unfold.add_previous_image(body,folder,None)
    contract=('ACTION时skill=garment-tangent-grasp，boundary_point_px给可见边界点C；'
        'grasp_px、target_px为null，chord_endpoints_px=[]，pull_distance_mm为10–150。'
        'region_name/evidence说明定位及依据。非ACTION时skill及所有点为null，'
        'chord_endpoints_px=[]，pull_distance_mm=10。')
    save(Path(folder)/'planning_skills.json',dict(mode='online_forced_tangent',
         skills=['garment-tangent-grasp'],local_cv_executed=False,semantic_review=False))
    value,metrics=run(root,folder,view,body,headers,timeout,one_request,
                      selection_prompt=contract,max_attempts=1)
    return unfold.finish_plan(folder,before,view,value,metrics)


class RandomizedRequest:
    """Observation -> persisted random choice -> one selected planning request."""
    def __init__(self,root,*,seed,run_id,runtime,headers,timeout,direct):
        self.root=Path(root);self.seed=seed;self.run_id=run_id;self.runtime=runtime
        self.headers=headers;self.timeout=timeout;self.direct=direct
        if not (self.root/'goal/target.png').is_file():
            raise ValueError('explicit black-garment goal required; no red-goal default')
        assignment(seed,run_id,0)

    def __call__(self,folder,before,context=None):
        from .partitioned_unfold import plan_round
        from .unfold_skill_planner import abstain
        folder=Path(folder)
        step=int(next(p.name[6:] for p in (folder,*folder.parents) if p.name.startswith('round_')))
        record=assignment(self.seed,self.run_id,step)
        obs_id=getattr(before,'meta',{}).get('observation_id')
        record.update(observation_id=obs_id,before_rgb=str(before.image),
            before_sha256=file_hash(before.image),goal_sha256=file_hash(self.root/'goal/target.png'),
            selection_stage='BEFORE_PLANNING',planning_status='PENDING',planning_success=None,model_requests=0,
            availability={k:{'status':'NOT_EVALUATED','reason':'not planned'} for k in PROBABILITIES},
            action=None,execution={'status':'NOT_DISPATCHED'},abstain_reason=None)
        path=folder/'exploration.json'
        if path.exists():raise ValueError('iteration already assigned; no redraw or replanning')
        save(path,record,immutable=True)
        print(f"[Run {self.run_id}/6 iter {step:03d}] 分配 {record['forced_skill']}，概率={record['selection_probability']:.0%}；开始单次规划", flush=True)
        calls=0
        def counted(destination,body,headers,timeout):
            nonlocal calls
            calls+=1
            if calls>1:raise RuntimeError('one-shot request budget exhausted')
            return self.direct(destination,body,headers,timeout)
        try:
            plan,metrics=plan_round(self.root,folder,before,self.headers,self.timeout,counted,
                runtime=self.runtime,forced_skill=record['forced_skill'])
            decision=plan['decision']
            record.update(action=plan,planning_status=decision,
                abstain_reason=plan.get('short_reason') if decision=='ABSTAIN' else None)
            record['availability'][record['forced_skill']]={
                'status':'CANDIDATE_GENERATED' if decision=='ACTION' else 'NO_ACTION',
                'reason':plan.get('short_reason'),
                'motion_validation':'NOT_YET_EVALUATED' if decision=='ACTION' else 'NOT_APPLICABLE'}
            return plan,metrics
        except Exception as exc:
            reason=f'{type(exc).__name__}: {exc}'
            plan=abstain(reason)
            record.update(action=plan,planning_status='ABSTAIN',abstain_reason=reason,failure_kind='PLANNING_EXCEPTION')
            record['availability'][record['forced_skill']]={'status':'UNAVAILABLE','reason':reason}
            return plan,{'wall_latency_s':None,'selection_attempts':calls,'error':reason}
        finally:
            record['model_requests']=calls
            record['planning_success']=record['planning_status']=='ACTION'
            save(path,record)
            print(f"[Run {self.run_id}/6 iter {step:03d}] 规划={record['planning_status']}；请求={calls}；原因={record['abstain_reason'] or '无'}", flush=True)


def collect_records(root):
    """Append raw evidence references after the episode, never an online judge."""
    root=Path(root);summary=read(root/'episode_summary.json');rows=[]
    for step in summary.get('steps',[]):
        folder=root/f"round_{step['round']:03d}"
        r=read(folder/'exploration.json')
        if not r:continue
        r['execution']=step.get('execution',{'status':'NOT_DISPATCHED'})
        r['action_geometry']=read(folder/'geometry.json') if (folder/'geometry.json').exists() else None
        r['actual_robot_actions']=r['execution'].get('experiment',{}).get('actual_robot_actions',[])
        r['episode_status']=summary.get('status')
        r['episode_error']=summary.get('error')
        rejected=read(folder/'translation_rejection.json')
        if rejected or r['execution'].get('status')=='PLANNING_FAILED':
            reason=rejected or r['execution'].get('error')
            r.update(planning_status='ABSTAIN',abstain_reason=reason)
            r['availability'][r['forced_skill']]={'status':'UNAVAILABLE','reason':reason}
        after=folder/'after/rgb.png'
        r['after_rgb']=str(after) if after.exists() else None
        r['after_observation_id']=read(folder/'after/observation.json').get('observation_id')
        r['planning_success']=r['planning_status']=='ACTION'
        planned_rgb=folder/'topdown/planning_rgb.png'
        r['planning_before_rgb']=str(planned_rgb) if planned_rgb.exists() else r['before_rgb']
        r['observed_change']={'status':'UNKNOWN','evaluation':'deferred offline; before/after raw evidence retained'}
        r['evidence_files']={name:str(folder/name) for name in
            ('plan.json','geometry.json','motion.py','robotapi_preview.py','execution.json',
             'skill_selection/request_payload.json','skill_selection/selection.json') if (folder/name).exists()}
        save(folder/'exploration.json',r);rows.append(r)
    (root/'iterations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    save(root/'exploration_summary.json',dict(status=summary.get('status'),
        model_reported_complete=summary.get('status')=='MODEL_REPORTED_DONE',
        task_complete_verified=None,initialization=summary.get('initialization'),
        physical_actions_including_reset=summary.get('physical_actions',0),
        executed_skills={k:sum(r['forced_skill']==k and r['execution'].get('status')=='EXECUTED' for r in rows) for k in PROBABILITIES},
        final_rgb=next((r['after_rgb'] for r in reversed(rows) if r.get('after_rgb')),
                      rows[-1]['before_rgb'] if rows else None),
        semantic_endpoints={k:'UNKNOWN' for k in ('left_sleeve','right_sleeve','left_hem_corner','right_hem_corner')},
        final_flatness='UNKNOWN',residual_wrinkles='UNKNOWN',stacking='UNKNOWN',counterfactual_preference=None))
    return rows


def run_episode(runtime,root,*,seed,run_id,headers,timeout,direct,initialize_seed,radius_mm):
    """Library adapter only. Caller must supply a ready existing runtime and goal."""
    from .partitioned_unfold import episode,initialize
    root=Path(root)
    if (root/'exploration_protocol.json').exists():
        raise ValueError('run already started; no automatic rerun or replacement')
    goal=read(root/'goal/manifest.json')
    if goal.get('garment_id')!='black_mario' or not goal.get('source'):
        raise ValueError('black Mario goal is not configured; old red reference is not permitted')
    if goal.get('sha256')!=file_hash(root/'goal/target.png'):
        raise ValueError('explicit goal differs from the prepared black-garment reference')
    request=RandomizedRequest(root,seed=seed,run_id=run_id,runtime=runtime,
        headers=headers,timeout=timeout,direct=direct)
    save(root/'exploration_protocol.json',dict(run_id=run_id,seed=seed,probabilities=PROBABILITIES,
        max_runs=6,max_actions=12,preplanning=True,model_requests_per_step=1,
        automatic_initialization=True,initialize_seed=initialize_seed,radius_mm=radius_mm,
        no_fallback=True,unselected_candidate='NOT_EVALUATED',gate_training=False),immutable=True)
    try:
        return episode(runtime,root,max_actions=12,max_plan_attempts=1,request=request,
            reset_request=lambda before,folder:initialize(runtime,before,folder,headers=headers,
                timeout=timeout,direct=direct,seed=initialize_seed,radius_mm=radius_mm))
    finally:
        collect_records(root)
