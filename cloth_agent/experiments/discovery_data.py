"""Conservative adapters for recorded fold_exploration iterations and Codex rounds."""
from __future__ import annotations
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from PIL import Image

VERSION = 'fold-record-v1'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    p=Path(path)
    return json.loads(p.read_text()) if p.is_file() else {}


def save(path, value, *, immutable=False):
    p=Path(path);p.parent.mkdir(parents=True, exist_ok=True)
    text=json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if immutable and p.exists():
        if read(p)!=value:raise ValueError(f'refusing to overwrite frozen artifact: {p}')
        return
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(text);tmp.replace(p)


def choose_rgb(folder, record, when):
    # Known raw camera names only; never use overlays, segmentation or annotated plans.
    names=('camera_A_rgb_upright.png','camera_0_A.png','camera_A_rgb.png')
    candidates=[folder/f'{when}_raw'/n for n in names]
    listed=record.get(f'{when}_images')
    if not isinstance(listed,list):listed=[]
    candidates += [Path(p) for p in listed if isinstance(p,str) and Path(p).name in names]
    for p in candidates:
        if p.is_file():
            try:
                with Image.open(p) as im:im.verify()
                return str(p.resolve())
            except (OSError,ValueError):pass
    return None


def clean_actions(proposal):
    # Strip observation, reveal_strategy, reasoning and all execution annotations.
    if not isinstance(proposal,dict):return None
    rows=proposal.get('actions')
    if not isinstance(rows,list) or not rows:return None
    out=[]
    for row in rows:
        if not isinstance(row,dict) or row.get('name') not in ('move','open_gripper','close_gripper','home'):return None
        args=row.get('args',{})
        if not isinstance(args,dict):return None
        permitted={'x','y','z','yaw'} if row['name']=='move' else set()
        if set(args)!=permitted:return None
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in args.values()):return None
        out.append(dict(name=row['name'],args=args))
    return out


def action_record(actions):
    if not actions:return None
    # Only one unambiguous close -> first release cycle. Other programs retain raw commands.
    closes=[i for i,a in enumerate(actions) if a['name']=='close_gripper']
    pa=pb=None
    if len(closes)==1:
        c=closes[0];previous=[a['args'] for a in actions[:c] if a['name']=='move']
        release=next((i for i in range(c+1,len(actions)) if actions[i]['name']=='open_gripper'),None)
        held=[a['args'] for a in actions[c+1:release] if a['name']=='move'] if release is not None else []
        if previous and held:pa=[previous[-1][k] for k in ('x','y','z')];pb=[held[-1][k] for k in ('x','y','z')]
    return dict(raw=actions,grasp=pa,target=pb,normalized=dict(
        coordinate_system='legacy RobotAPI Cartesian command frame',units='mm',
        delta=[b-a for a,b in zip(pa,pb)] if pa is not None else None,
        garment_relative=None,normalization_unavailable='No verified garment mask/bbox in this adapter',
        endpoint_rule='last move before unique close; last held move before first open'))


def extract(roots, subgoal='auto'):
    roots=sorted({str(Path(r).resolve()) for r in roots})
    dirs=set()
    for root in roots:
        p=Path(root)
        if not p.is_dir():raise ValueError(f'runs root missing: {root}')
        dirs.update(x.resolve() for pattern in ('**/iteration_*','**/round_*') for x in p.glob(pattern) if x.is_dir() and x.name.split('_')[-1].isdigit())
    samples=[]
    for folder in sorted(dirs):
        issues=[];sources={};record={}
        try:
            record=read(folder/'record.json')
            if not record:record=read(folder/'partial_record.json')
            if not isinstance(record,dict):raise ValueError('record not an object')
        except (ValueError,OSError) as e:
            record={};issues.append(f'invalid_record: {type(e).__name__}')
        step=record.get('planned_step');label_source='record.json:planned_step'
        if not step:
            step=(record.get('supervisor_before') if isinstance(record.get('supervisor_before'),dict) else {}).get('current_step');label_source='record.json:supervisor_before.current_step'
        mode=record.get('action_mode') or record.get('mode')
        if not isinstance(step,str) or not step.strip():step=None;issues.append('missing_explicit_subgoal')
        # Do not mix probing, revealing or repairing with normal folding.
        if mode!='FOLD':issues.append('not_explicit_FOLD_action_mode')
        if step:sources['subgoal']=dict(file=str(folder/('record.json' if (folder/'record.json').exists() else 'partial_record.json')),field=label_source.split(':',1)[1],mapping='identity',mode=mode)
        before=choose_rgb(folder,record,'before');after=choose_rgb(folder,record,'after')
        if not before:issues.append('missing_clean_camera_A_before')
        if not after:issues.append('missing_clean_camera_A_after')
        proposal=record.get('execution_proposal');action_source='record.json:execution_proposal.actions'
        if not proposal:
            try:proposal=read(folder/'execution_plan.json');action_source='execution_plan.json:actions'
            except (ValueError,OSError):proposal={};issues.append('invalid_execution_plan')
        action=action_record(clean_actions(proposal))
        if not action:issues.append('missing_supported_action')
        else:sources['action']=dict(file=str(folder/action_source.split(':')[0]),field=action_source.split(':')[1],semantics='commanded plan, not measured trajectory')
        execution=record.get('execution') or {}
        if not execution:
            try:execution=read(folder/'execution.json')
            except (ValueError,OSError):execution={}
        physical=execution.get('physical_execution') if isinstance(execution,dict) else None
        evaluation=record.get('evaluation');policy=record.get('outcome')
        if not evaluation:
            try:evaluation=read(folder/'evaluation.json') or None
            except (ValueError,OSError):issues.append('invalid_evaluation')
        outcome=dict(physical_execution=physical,evaluation=evaluation,policy_labels=policy,
            execution_telemetry={k:execution[k] for k in ('execution_completed','gripper_completion_failed','robot_errors') if isinstance(execution,dict) and k in execution},
            provenance='historical evaluator / host policy; not ground truth',
            warning='UNCHANGED/ACQUISITION labels do not prove empty jaws or shallow Z; execution completion is not physical success') if (evaluation or policy or physical is not None) else None
        if physical is not True:issues.append('physical_execution_not_confirmed')
        if not evaluation and not policy and not after:issues.append('missing_outcome_evidence')
        run_id=folder.parent.name+'_'+digest(str(folder.parent))[:10]
        sid='s_'+digest(str(folder))[:16]
        eligible=before is not None and step is not None and mode=='FOLD'
        sample=dict(sample_id=sid,run_id=run_id,iteration_id=folder.name,subgoal=step,
            before_rgb=before,action=action,after_rgb=after,outcome=outcome,
            source_directory=str(folder),sources=sources,issues=issues,
            image_hashes={k:file_hash(v) for k,v in [('before',before),('after',after)] if v},
            eligible=dict(image_only=eligible,action=eligible and action is not None,
                outcome=eligible and action is not None and physical is True and bool(after or evaluation or policy)))
        samples.append(sample)
    counts=Counter(s['subgoal'] for s in samples if s['eligible']['image_only'])
    chosen=(sorted(counts,key=lambda k:(-counts[k],k))[0] if counts else None) if subgoal=='auto' else subgoal
    selected=[s['sample_id'] for s in samples if s['subgoal']==chosen and s['eligible']['image_only']]
    hash_groups={}
    for s in samples:
        if s.get('image_hashes',{}).get('before'):
            hash_groups.setdefault(s['image_hashes']['before'],[]).append(s['sample_id'])
    from .discovery_goal import task_contract
    return dict(schema_version=2,adapter_version=VERSION,roots=roots,selected_subgoal=chosen,
        task_contract=task_contract(chosen),
        all_explicit_subgoal_counts=dict(Counter(s['subgoal'] for s in samples if s['subgoal'])),
        duplicate_before_image_groups={k:v for k,v in hash_groups.items() if len(v)>1},
        field_completeness={goal:{field:sum(bool(s[field]) for s in samples if s['subgoal']==goal and s['eligible']['image_only']) for field in ('before_rgb','action','after_rgb','outcome')} for goal in sorted(counts)},
        selection_rule='largest number of explicit FOLD + clean Camera A before; deterministic lexical tie-break',
        subgoal_counts=dict(sorted(counts.items())),total_observations=len(samples),selected_sample_ids=selected,
        stage_counts={stage:sum(s['subgoal']==chosen and s['eligible'][stage] for s in samples) for stage in ('image_only','action','outcome')},
        issue_counts=dict(Counter(i for s in samples for i in s['issues'])),samples=samples,
        limitations=['Adapter accepts explicit FOLD only; repairs/probes not merged.',
          'Codex rounds without explicit subgoal metadata are listed but excluded.',
          'Repeated or identical images are retained and hash groups reported; no assumption of independent samples.',
          'No visual taxonomy supplied by host; no reliable bbox normalization available.'])


def stage_sample(sample, stage):
    if stage not in ('image_only','action','outcome'):raise ValueError(stage)
    # Explicit allowlists; even run IDs and filesystem paths are hidden from the model.
    row=dict(sample_id=sample['sample_id'])
    if stage in ('action','outcome'):row['action']=sample['action']
    if stage=='outcome':
        row['outcome']=sample['outcome']
        row['after_observation_available']=bool(sample['after_rgb'])
    return row
