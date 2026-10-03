"""External physical-evidence review for working patches; never runs a robot.

Freeze exports an offline library artifact only, never changes the active registry.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .candidate_patch import registry_from_bundle, verify_candidate
from .common import digest, now, read_json, write_json
from .policy import PolicyError, obj, validate_schema

TEXT={'type':'string','minLength':1,'maxLength':2400}
OUTCOME_SCHEMA=obj({'schema_version':{'const':1},'bundle_hash':TEXT,'observation_id':TEXT,
    'root_evidence_hash':TEXT,'action_hash':TEXT,'physical_record_path':TEXT,'physical_record_hash':TEXT,
    'outcome':{'enum':['SUCCESS','FAILURE','UNKNOWN']},'reviewed_by':TEXT,
    'attribution':obj({'task_decision':{'enum':['CAUSE','CLEARED','UNKNOWN']},
        'grasp_depth':{'enum':['CAUSE','CLEARED','UNKNOWN']},'target':{'enum':['CAUSE','CLEARED','UNKNOWN']},
        'observation_patch':{'enum':['CAUSE','CLEARED','UNKNOWN']},'rationale':TEXT}),
    'replay_report_path':TEXT,'replay_report_hash':TEXT})


def verify_working(working):
    working=Path(working)
    evidence=read_json(working/'evidence.json');bundle=read_json(working/'bundle.json')
    if digest(bundle)!=evidence['bundle_hash'] or evidence.get('status')!='WORKING_OFFLINE_ONLY':
        raise PolicyError('Working artifact identity mismatch')
    registry_from_bundle(bundle)
    candidate=verify_candidate(evidence['candidate_directory'])
    if digest(candidate)!=digest(bundle): raise PolicyError('Working candidate differs from tested version')
    evaluation=evidence['evaluation']
    required={'tests_passed','valid_complete_plans','baseline_valid','same_predecision_inputs','actual_replays',
              'cheapest_in_dominant_cluster','baseline_in_dominant_cluster','latency_improved','no_extra_unresolved_information'}
    if (evaluation.get('status')!='WORKING' or set(evaluation.get('checks',{}))!=required
            or not all(evaluation['checks'].values())):
        raise PolicyError('No passing inner promotion evidence')
    return bundle,evidence


def review_outcome(working,review,output,*,base=None):
    """Record explicit externally reviewed evidence, never infer attribution."""
    bundle,evidence=verify_working(working)
    validate_schema(review,OUTCOME_SCHEMA)
    if review['bundle_hash']!=digest(bundle): raise PolicyError('Outcome belongs to another implementation')
    image_hash=None
    for key in ('physical_record','replay_report'):
        path=Path(review[key+'_path'])
        path=path if path.is_absolute() else Path(base or '.')/path
        value=read_json(path)
        if digest(value)!=review[key+'_hash']: raise PolicyError(f'{key} changed')
        review={**review,key+'_path':str(path.resolve())}
        if key=='replay_report':
            rows=value.get('rollouts',[])
            matched=[r for r in rows if r.get('harness_hash')==review['bundle_hash'] and r.get('root_evidence_hash')==review['root_evidence_hash']]
            if not matched or not all(r.get('actual_measurement') and r['status']=='READY' for r in matched):
                raise PolicyError('No real valid replay evidence for this observation and version')
            if any(r.get('observation_id')!=review['observation_id'] for r in matched):
                raise PolicyError('Replay observation mismatch')
            if not any(digest(r.get('action'))==review['action_hash'] for r in matched):
                raise PolicyError('Executed action is not a candidate replay action')
            hashes={r.get('observation_rgb_sha256') for r in matched}
            if len(hashes)!=1 or None in hashes: raise PolicyError('Missing immutable state image identity')
            image_hash=hashes.pop()
        else:
            # External adapter must explicitly bind physical execution to this
            # working bundle; unrelated old run outcomes cannot certify it.
            for field in ('bundle_hash','observation_id','root_evidence_hash','action_hash','outcome'):
                if value.get(field)!=review[field]: raise PolicyError('Physical record is not bound to this patch/observation/outcome')
    attribution=review['attribution']
    clear=review['outcome']!='UNKNOWN' and attribution['observation_patch']=='CLEARED'
    if review['outcome']=='FAILURE' and not any(attribution[k]=='CAUSE' for k in ('task_decision','grasp_depth','target')):
        clear=False
    record={'status':'REVIEWED_CLEAR' if clear else 'HOLD','review':review,'observation_rgb_sha256':image_hash,'recorded_at':now(),
            'authority':'Explicit external reviewer; not model-inferred or host-verified physical causality'}
    write_json(output,record,exclusive=True)
    return record


def freeze_reviewed(working,reviews,output,*,minimum_states=3):
    if type(minimum_states) is not int or minimum_states<3: raise PolicyError('Need at least three distinct reviewed states')
    bundle,evidence=verify_working(working)
    # Caller must include all review records in this dedicated ledger directory.
    ledger=Path(reviews)
    records=[read_json(p) for p in sorted(ledger.glob('*.json'))]
    if not records: raise PolicyError('No physical review ledger')
    states=set()
    for record in records:
        review=record['review']
        validate_schema(review,OUTCOME_SCHEMA)
        if record['status']!='REVIEWED_CLEAR' or review['bundle_hash']!=digest(bundle):
            raise PolicyError('Unresolved failure/attribution or mixed version in review ledger')
        # Revalidate record hashes and original eligibility rather than trusting status.
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            checked=review_outcome(working,review,Path(temp)/'review.json')
        if checked['status']!='REVIEWED_CLEAR': raise PolicyError('Review is no longer eligible')
        states.add(checked['observation_rgb_sha256'])
    if len(states)<minimum_states: raise PolicyError('Insufficient distinct pre-decision states')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'bundle.json',bundle,exclusive=True)
    write_json(output/'evidence.json',{'status':'FROZEN_OFFLINE_LIBRARY','bundle_hash':digest(bundle),
        'inner_evidence':evidence,'physical_reviews':records,'distinct_states':len(states),'frozen_at':now(),
        'auto_enabled':False,'robot_executable':False},exclusive=True)
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    r=commands.add_parser('review');r.add_argument('--working',type=Path,required=True);r.add_argument('--review',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    f=commands.add_parser('freeze');f.add_argument('--working',type=Path,required=True);f.add_argument('--reviews',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.command=='review': review_outcome(a.working,read_json(a.review),a.output,base=a.review.parent)
        else: freeze_reviewed(a.working,a.reviews,a.output)
        return 0
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}');return 2


if __name__=='__main__': raise SystemExit(main())
