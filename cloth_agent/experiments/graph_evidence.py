"""Deterministic, explicitly lossy projection of dense controller telemetry.

Raw evidence is never modified. This is not a cloth-state classifier.
"""
from collections import Counter
import math
from .discovery_data import digest

SERIES_FIELDS = (
    ('phase',), ('completion_reason',), ('elapsed_s',), ('progress_pulse',),
    ('feedback', 'state'), ('feedback', 'error_code'), ('feedback', 'status_code'),
    ('feedback', 'position_code'), ('feedback', 'position_pulse'),
    ('feedback', 'available'), ('feedback', 'mechanical_grasp_detected'),
    ('feedback', 'read_errors'),
)


def series_statistics(rows):
    result = {}
    for path in SERIES_FIELDS:
        values = []
        for row in rows:
            value = row
            for key in path:
                if not isinstance(value, dict) or key not in value: break
                value = value[key]
            else: values.append(value)
        if not values: continue
        stats = {'present_count': len(values), 'missing_count': len(rows)-len(values)}
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            stats.update(min=min(values), max=max(values))
        else:
            # Includes mechanical-grasp true/false and read-error distributions;
            # these remain mechanical signals, never acquisition ground truth.
            counts = Counter(str(v) for v in values)
            stats['most_common_values'] = counts.most_common(12)
            stats['other_value_count'] = sum(n for _,n in counts.most_common()[12:])
        result['.'.join(path)] = stats
    return result


def compact(value, *, retain_sequence=False):
    if isinstance(value, dict):
        return {k: (dict(summary_type='INTERPOLATION_SEGMENTS_SUMMARY',
                        original_count=len(v), original_sha256=digest(v),
                        limitation='Controller interpolation omitted; exact task commands and original artifact retained')
                    if k=='queued_segments' and isinstance(v,list)
                    else compact(v, retain_sequence=k == 'events')) for k,v in value.items()}
    if not isinstance(value, list): return value
    if retain_sequence or len(value) <= 12:
        return [compact(v) for v in value]
    result = dict(summary_type='DENSE_SEQUENCE_SUMMARY', original_count=len(value),
                  original_sha256=digest(value), first=compact(value[0]), last=compact(value[-1]),
                  limitation='Interior sequence details omitted from model input; raw artifact retained. '
                             'Do not infer continuous contact or absence of transient faults from endpoints.')
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value):
        result.update(min=min(value), max=max(value))
    if all(isinstance(v, dict) for v in value): result['telemetry_statistics'] = series_statistics(value)
    return result


def compact_evidence(evidence):
    return {name: {**record, 'content': compact(record.get('content'), retain_sequence=name == 'native_plan.json'),
                   'projection': 'dense-telemetry-v1; artifact points to original unmodified evidence'}
            for name,record in evidence.items()}


def builder_steps(steps):
    result=[]
    for step in steps:
        row=dict(step)
        if 'intermediate_evidence' in step:
            row['intermediate_evidence']=compact_evidence(step['intermediate_evidence'])
        # Failures embed the same high-rate gripper trace both here and in outcome.
        # Keep commands/observation IDs exact; summarize telemetry on every copy.
        for key in ('actual_execution','outcome'):
            if key in step:row[key]=compact(step[key])
        result.append(row)
    return result


def compiler_steps(steps):
    """Compile visual state/action graph without duplicating native motion telemetry."""
    result=builder_steps(steps)
    for row in result:
        for name,record in row.get('intermediate_evidence',{}).items():
            if name in {'native_plan.json','execution_events.json','trajectory_preview.json','planning_stats.json'}:
                content=record.get('content')
                events=content.get('events',[]) if isinstance(content,dict) else []
                record['content']=dict(summary_type='CONTROLLER_ARTIFACT_REFERENCE',
                    projected_sha256=digest(content),
                    event_sequence=[{k:e[k] for k in ('event','sequence','index','kind','arm','action','error','reason') if k in e} for e in events],
                    limitation='Detailed controller paths omitted from visual graph compiler only; exact task command, execution outcome and original artifact remain available.')
    return result
