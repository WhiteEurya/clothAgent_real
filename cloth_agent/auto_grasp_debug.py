"""Opt-in bounded grasp diagnostics. No controller calls or relaxed geometry checks."""
import copy
import json
import numpy as np
from pathlib import Path


def close_evidence(feedback):
    """A wide stopped gap is only suspected obstruction, never proof of sponge."""
    c=feedback.get('completion',{});f=feedback.get('feedback',{})
    if c.get('status')!='COMPLETED' or c.get('reason')!='measured_close_stable':return 'UNKNOWN'
    if not f.get('usable_for_completion') or f.get('read_errors') or f.get('error_code')!=0 or f.get('state') not in {'stop','grasp'}:return 'UNKNOWN'
    pos=f.get('position_pulse');target=c.get('target_position_pulse');initial=c.get('before_command',{}).get('position_pulse')
    if any(v is None or not np.isfinite(v) for v in (pos,target,initial)):return 'UNKNOWN'
    if abs(pos-target)<=c.get('position_tolerance_pulse',5):return 'NEAR_CLOSED'
    opening=initial-target
    if opening>20 and (pos-target)/opening>=.8:return 'WIDE_STALLED'
    return 'CONTACT_OR_UNKNOWN'


def prepare_branch(runtime, source, operations, folder):
    """Pre-plan lift+reclose+entire continuation before any physical dispatch."""
    from .codex_motion import parse_code
    calls=parse_code(source,allow_grasp=True)
    if calls[0][0]!='grasp' or set(calls[0][1])!={'right'}:raise ValueError('auto grasp debug needs one right grasp')
    close_indices=[i for i,(k,d) in enumerate(operations) if k=='gripper' and d['action']=='close']
    if len(close_indices)!=1:raise ValueError('auto grasp debug needs exactly one initial close')
    i=close_indices[0];q=operations[i][1]['joints'];pose=list(calls[0][1]['right']);pose[2]+=.002
    if any(name not in {'move_tcp','move_transfer'} for name,kw in calls[1:]):raise ValueError('unsupported debug transport skeleton')
    last_kind,last=operations[-1]
    if last_kind!='move':raise ValueError('debug continuation needs known final joint pose')
    # Restore the original validated final joint posture through the existing planner.
    lines=[f'motion.move_tcp(right={pose!r})','motion.gripper(arm="right", action="close")']
    lines += ['motion.'+name+'('+','.join(k+'='+repr(v) for k,v in kw.items())+')' for name,kw in calls[1:]]
    lines += ['motion.gripper(arm="right", action="open")',f'motion.move_joints(right={np.radians(last.end["right"]).tolist()!r})']
    branch_source='\n'.join(lines)+'\n'
    stats=copy.deepcopy(runtime.plan_stats)
    branch=runtime.plan(branch_source,{k:np.asarray(v) for k,v in q.items()})
    branch_stats=copy.deepcopy(runtime.plan_stats);runtime.plan_stats=stats
    folder=Path(folder);(folder/'auto_unstick_motion.py').write_text(branch_source)
    (folder/'auto_unstick_plan.json').write_text(json.dumps(dict(trigger='WIDE_STALLED',lift_mm=2,
        note='suspected obstruction only; no force/contact classification',stats=branch_stats,
        operations=[dict(kind=k,data=d.to_dict() if k=='move' else d) for k,d in branch]),indent=2))
    return dict(close_index=i,operations=branch)
