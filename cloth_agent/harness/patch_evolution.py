"""Plan -> diagnose -> propose -> implement -> gate -> replay -> working/reject.

Standalone offline entry point; no production writes or robot execution.
"""
from __future__ import annotations

import argparse
import copy
import html
import json
import math
import shutil
import statistics
import time
import traceback
from pathlib import Path

from .action_consensus import select_consensus
from .candidate_patch import (CONFIG_SCHEMA, IMPLEMENTATION_SCHEMA, PROPOSAL_SCHEMA, RECIPE_SCHEMA, TEST_SCHEMA,
    baseline_bundle, behavior_hash, bundle_hash, materialize_candidate, registry_from_bundle, run_gates, verify_candidate)
from .common import canonical, digest, now, read_json, write_json
from .executors.observation import RegisteredObservationHost
from .information_flow import global_schema, validate_state
from .model import RuntimeClaude
from .policy import PolicyError, validate_schema
from .reasoning_contract import evidence_from_manifest, freeze_evidence, freeze_harness, load_evidence, verify_evidence
from .reasoning_learning import BudgetExceeded, CallBudget, DebugLog, call_metrics, run_rollout

DIAGNOSE = '''Diagnose the public planning trace and costs; propose ONE testable change to HOW the
agent acquires or reasons about evidence, not the action to take. Select PROMPT, HARNESS, SKILL_CODE,
or NEW_TOOL. Preserve coordinate alignment, original provenance, bounded budgets, no robot access,
and pre-decision input isolation. Separate observed repetition/cost from unverified usefulness.
Do not claim a tool is unnecessary merely because final prose omitted it. Output a PatchProposal only;
it does not change runtime behavior. For code, target ONLY observation skill implementations.
No historical candidate, target, fixed scene ROI, answer lookup or covert answer encoding in the patch.
'''
IMPLEMENT = '''Implement the approved candidate proposal in this isolated workspace contract. Return
complete replacement config and, only for SKILL_CODE/NEW_TOOL, a semantic specification, Python source,
and at least two declarative unit tests with expected recipes. Host will generate the exact diff and
only write the selected skill directory under cloth_agent/harness/skills/. Do not supply arbitrary paths.
No production edits, imports, attributes, loops, comprehensions, eval/exec, file IO, network or robot access.
Code is a bounded pure function: def prepare(request, source, available). It receives JSON values only.
It may use assignments, if/else, return, numeric arithmetic, comparisons, lists/dicts/indexing, and
min/max/abs/round/int/float/len/floor/ceil. No recursion, power or sequence arithmetic. Return a recipe
with roles [clean] or [clean, overlay], operations (crop normalized ROI, rotate right angles, resize
bounded scale), and reuse_existing bool. Host executes operations, records affine chains and may reuse
identical cached operation prefixes. A new ID/program extends the registry, not the trusted IO kernel.
Source has image_id, role and size; available has cached_views and remaining_ops.
Request keys are gap_id, skill_id, source_image_id, roi, degrees_clockwise, enlarge, expected_information_gain.
ROI is in normalized current clean ROOT coordinates initially; later recipe crops are relative to
preceding recipe output. Both roles get the identical operation sequence.
Implement dynamic geometry from current inputs; never insert baseline answers, historical pixels,
image-specific constants or hidden lookup tables. Source tests are synthetic geometry, not live answers.
SKILL_CODE: target existing ID, version increments exactly once. Preserve all original valid inputs.
NEW_TOOL: new ID, version one. Code patches preserve baseline reasoning and observation prompts;
only NEW_TOOL may add its ID to enabled_skills. PROMPT changes text only. HARNESS may change configuration
and stage topology within allowed schemas. No code for configuration-only patches.
'''
OBSERVE = '''On these CURRENT pre-decision inputs identify selection-critical visual information gaps.
Do not select grasp or target. A known finding must cite an attached image; an UNKNOWN gap must say
what is missing. Bind a bounded set of registered observation requests to the current clean ROOT.
No old planner crops, previous rollout conclusions or action/evaluation are available. The selected
implementation executes on Host; a processed image alone does not prove that a gap has been resolved.
'''


def replay_bundle(bundle, frozen, evidence_dir, model, output, budget, *, rollout_id, timeout=300, max_ops=12):
    """Candidate gets the same roots; any new views must be generated afresh here."""
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    start, calls_start=time.monotonic(),len(model.calls)
    version_hash=bundle_hash(bundle)
    row={'rollout_id':rollout_id,'harness_hash':version_hash,'root_evidence_hash':frozen['evidence_hash'],
         'observation_id':frozen['evidence']['observation_id'],
         'observation_rgb_sha256':next(i['rgb_sha256'] for i in frozen['evidence']['images'] if i['role']=='clean'),
         'status':'ERROR','action':None,'reason':'UNFINISHED','actual_measurement':model.configuration.get('actual_measurement',True)}
    host=None
    try:
        registry=registry_from_bundle(bundle)
        root=copy.deepcopy(frozen['evidence'])
        paths=verify_evidence(frozen,evidence_dir)
        config=bundle['config']
        catalog=[s for s in registry.catalog() if s['id'] in config['enabled_skills']]
        host=RegisteredObservationHost(root,paths,{'skills':catalog},output/'observations',max_ops,registry=registry)
        # Restrict model-visible IDs to enabled entries, not all installed capabilities.
        from .skills import SkillRegistry
        enabled=SkillRegistry()
        for name in config['enabled_skills']: enabled.register(registry.get(name))
        payload={'current_evidence':root,'instruction':config['observation_instruction'],'skill_specifications':catalog,
                 'remaining_host_ops':max_ops}
        observation=budget.invoke(model,frozen=frozen,evidence_dir=evidence_dir,prompt=OBSERVE+canonical(payload),
            schema=global_schema(enabled),output=output/'observe',stage='patch_observe',deadline=start+timeout)
        validate_schema(observation,global_schema(enabled))
        validate_state(observation['information'],host.catalog)
        write_json(output/'observation.json',observation)
        requested=observation['observation_requests']
        gaps={i['id'] for i in observation['information'] if i['status']=='UNKNOWN'}
        if gaps - {r['gap_id'] for r in requested}:
            row.update(status='NEEDS_LEARNING',reason='UNRESOLVED_GAP_WITHOUT_SUPPORTED_OBSERVATION')
        else:
            stop=host.execute(requested,observation['information'])
            if stop: row.update(status='NEEDS_LEARNING',reason=stop)
            else:
                # Fresh roots plus candidate-generated views; separate root/derived hashes.
                prepared=output/'prepared'; prepared.mkdir()
                augmented=copy.deepcopy(root)
                augmented['observation_requirements']={
                    'information':observation['information'],
                    'instruction':'Interpret each UNKNOWN gap using the attached freshly generated views. For READY, include a concept with the exact gap ID as its name, cited image sources and a finding that resolves it. If any gap remains blocking, return NEEDS_LEARNING. Host execution is not semantic proof.'}
                for i,path in enumerate(host.paths): shutil.copyfile(path,prepared/f'image_{i}.png')
                for item in host.catalog[len(root['images']):]:
                    augmented['images'].append({k:item[k] for k in ('image_id','role','size','rgb_sha256','original_image_id','to_original')} |
                        {'role':item['role']+'_crop','file':item['image_id']+'.png'})
                wrapped={'evidence':augmented,'evidence_hash':digest(augmented)}
                write_json(prepared/'evidence.json',wrapped)
                reasoning=freeze_harness(config['reasoning_harness'],output/'reasoning_version.json',source='candidate_bundle')
                result=run_rollout(reasoning,wrapped,prepared,model,output/'reasoning',budget,
                                   rollout_id=rollout_id,timeout=max(.001,timeout-(time.monotonic()-start)))
                row.update(status=result['status'],reason=result['reason'],action=result['action'],
                           reasoning=result,derived_evidence_hash=wrapped['evidence_hash'])
                # A fresh planner must interpret the requested gaps explicitly. Reuse
                # its public concepts as evidence; never infer KNOWN from host execution.
                if row['status']=='READY' and gaps:
                    concepts=result['stages'][-1]['judgment']['concepts']
                    resolved={c['name'] for c in concepts}
                    if not gaps <= resolved:
                        row.update(status='NEEDS_LEARNING',reason='GAP_NOT_EXPLICITLY_INTERPRETED',action=None)
                if time.monotonic()-start>timeout: row.update(status='BUDGET_EXHAUSTED',reason='REPLAY_DEADLINE',action=None)
        verify_evidence(frozen,evidence_dir)
        if bundle_hash(bundle)!=version_hash: raise PolicyError('Bundle mutated during replay')
    except BudgetExceeded as exc: row.update(status='BUDGET_EXHAUSTED',reason=str(exc),action=None)
    except Exception as exc:
        row.update(status='ERROR',reason=f'{type(exc).__name__}: {exc}',action=None)
        (output/'exception.txt').write_text(traceback.format_exc())
    finally:
        calls=copy.deepcopy(model.calls[calls_start:])
        row['metrics']={**call_metrics(calls,output),'elapsed_s':time.monotonic()-start,
            'host_image_ops':host.ops if host else 0,'host_seconds':host.elapsed_s if host else 0}
        row['observation_trace']=host.history if host else []
        write_json(output/'call_audits.json',calls)
        write_json(output/'result.json',row)
        budget.debug.event('patch_replay_end',rollout_id=rollout_id,status=row['status'],reason=row['reason'],metrics=row['metrics'])
    return row


def evaluate_promotion(baseline_hash,candidate_hash,rows,consensus,*,gates_passed,min_latency_reduction=.1):
    checks={}
    candidate=[r for r in rows if r['harness_hash']==candidate_hash]
    baseline=[r for r in rows if r['harness_hash']==baseline_hash]
    checks['tests_passed']=gates_passed
    checks['valid_complete_plans']=bool(candidate) and all(r['status']=='READY' and (r.get('action') or {}).get('candidate_legal') for r in candidate)
    checks['baseline_valid']=bool(baseline) and all(r['status']=='READY' for r in baseline)
    checks['same_predecision_inputs']=len({r.get('root_evidence_hash') for r in candidate+baseline})==1 and all(r.get('root_evidence_hash') for r in candidate+baseline)
    checks['actual_replays']=bool(candidate) and all(r.get('actual_measurement') and r['metrics'].get('response_count',0)>0 for r in candidate+baseline)
    selected=consensus.get('selected') or {}
    checks['cheapest_in_dominant_cluster']=consensus['status']=='SELECTED' and selected.get('harness_hash')==candidate_hash
    clusters=consensus.get('clusters',[])
    dominant=clusters[0]['harnesses'] if clusters else []
    checks['baseline_in_dominant_cluster']=baseline_hash in dominant
    old=statistics.median(r['metrics']['elapsed_s'] for r in baseline) if baseline else None
    new=statistics.median(r['metrics']['elapsed_s'] for r in candidate) if candidate else None
    reduction=(old-new)/old if old and new is not None else None
    checks['latency_improved']=reduction is not None and reduction>=min_latency_reduction
    checks['no_extra_unresolved_information']=all(r['status']=='READY' for r in candidate)
    return {'status':'WORKING' if all(checks.values()) else 'REJECTED', 'checks':checks,
            'latency_reduction':reduction,'required_latency_reduction':min_latency_reduction,
            'baseline_median_seconds':old,'candidate_median_seconds':new,
            'grounding_scope':'Current-registry and visual pixel coordinates only; physical grounding not performed',
            'physical_status':'PENDING','frozen_eligible':False,'robot_executable':False}


def save_report(output,report):
    write_json(output/'report.json',report)
    lines=['# Candidate patch evolution','',f"Status: {report['status']}",
        'Offline only. Candidate code is tested before replay; working is not frozen and does not authorize robot execution.','',
        '| Version | State | Reason |','|---|---|---|']
    for c in report['candidates']:
        lines.append(f"| {c['id']} | {c['status']} | {str(c.get('reason','')).replace('|','/')} |")
    lines+=['','## Measurements and promotion','','```json',json.dumps({k:report.get(k) for k in ('consensus','promotions','totals')},ensure_ascii=False,indent=2),'```',
            '', 'Same-state compression/evolution only. Stable consensus does not prove physical correctness. All generation/testing/reflection overhead is reported separately.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    page='<h1>Candidate patch evolution</h1><a href="report.json">Report JSON</a> · <a href="events.jsonl">Debug events</a><pre>'+html.escape('\n'.join(lines))+'</pre>'
    for c in report['candidates']:
        name=c['id']
        page+=f'<details><summary>{name}: {html.escape(c["status"])}</summary>'
        for filename in ('proposal.json','implementation.json','patch.diff','gates.json','evaluation.json'):
            page+=f'<p><a href="candidates/{name}/{filename}">{filename}</a></p>'
        page+='<pre>'+html.escape(json.dumps(c,ensure_ascii=False,indent=2))+'</pre></details>'
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Patch evolution debug</title><style>body{max-width:1200px;margin:auto;padding:24px;font:16px system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ccc;padding:12px;margin:12px}</style>'+page)


def evolve(evidence,output,model,*,baseline=None,patches=3,repeats=2,max_calls=60,max_seconds=1800,
           call_timeout=180,replay_timeout=300,max_ops=12,min_latency_reduction=.1,consensus_options=None,
           source_base=None,prepare_only=False):
    if not 1<=patches<=8 or not 1<=repeats<=5 or not 1<=max_calls<=200 or not 0<=max_ops<=24:
        raise ValueError('Invalid finite evolution budget')
    if any(not math.isfinite(v) or v<=0 for v in (max_seconds,call_timeout,replay_timeout)) or not 0<min_latency_reduction<1:
        raise ValueError('Invalid timeout or promotion threshold')
    options=consensus_options or {}
    select_consensus([],[],repeats=repeats,**options)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    debug=DebugLog(output); start=time.monotonic(); call_start=len(model.calls)
    budget=CallBudget(max_calls=max_calls,max_seconds=max_seconds,call_timeout=call_timeout,prompt_chars=200000,debug=debug)
    report={'status':'PREPARING','candidates':[],'rollouts':[],'promotions':[],
            'actual_measurement':model.configuration.get('actual_measurement',True),'model_configuration':model.configuration,
            'settings':{'patches':patches,'repeats':repeats,'max_calls':max_calls,'max_seconds':max_seconds,
                'replay_timeout':replay_timeout,'call_timeout':call_timeout,'max_ops':max_ops,
                'min_latency_reduction':min_latency_reduction,'consensus':options},'robot_actions':0}
    save_report(output,report)
    try:
        baseline=copy.deepcopy(baseline or baseline_bundle())
        registry=registry_from_bundle(baseline)
        frozen=freeze_evidence(evidence,output/'evidence',base=source_base)
        baseline_id=bundle_hash(baseline)
        report.update(root_evidence_hash=frozen['evidence_hash'],baseline_hash=baseline_id)
        write_json(output/'baseline_bundle.json',baseline,exclusive=True)
        if prepare_only:
            report['status']='PREPARED'
            return report
        def evaluate(bundle,name):
            for repeat in range(repeats):
                rid=f'{name}_r{repeat:02d}'
                report['rollouts'].append(replay_bundle(bundle,frozen,output/'evidence',model,output/'replays'/rid,budget,
                    rollout_id=rid,timeout=replay_timeout,max_ops=max_ops))
                save_report(output,report)
                if report['rollouts'][-1]['status']=='BUDGET_EXHAUSTED': break
        report['status']='BASELINE'
        evaluate(baseline,'baseline')
        parent=[r for r in report['rollouts'] if r['harness_hash']==baseline_id]
        # Serialization/infrastructure failure is not a visual optimization trace.
        if not any(r['status']=='READY' for r in parent):
            report['status']='BLOCKED_BASELINE';return report
        admitted=[baseline_id]; signatures={behavior_hash(baseline['config'],registry)}
        for index in range(patches):
            cid=f'patch_{index:02d}'; directory=output/'candidates'/cid; directory.mkdir(parents=True)
            candidate={'id':cid,'status':'DIAGNOSING'};report['candidates'].append(candidate)
            started=time.monotonic()
            debug.event('candidate_start',candidate=cid)
            try:
                diagnosis={'baseline':baseline,'observed_rollouts':parent,
                           'previous_proposals':[c.get('proposal') for c in report['candidates'][:-1]],
                           'available_levels':['PROMPT','HARNESS','SKILL_CODE','NEW_TOOL']}
                proposal=budget.invoke(model,frozen=frozen,evidence_dir=output/'evidence',prompt=DIAGNOSE+canonical(diagnosis),
                    schema=PROPOSAL_SCHEMA,output=directory/'diagnose',stage='patch_diagnose')
                write_json(directory/'proposal.json',proposal,exclusive=True)
                validate_schema(proposal,PROPOSAL_SCHEMA)
                if not set(proposal['evidence_rollouts'])<={r['rollout_id'] for r in parent}: raise PolicyError('Unknown proposal trace citation')
                candidate.update(status='IMPLEMENTING',proposal=proposal)
                save_report(output,report)
                implementation=budget.invoke(model,frozen=frozen,evidence_dir=output/'evidence',
                    prompt=IMPLEMENT+canonical({'proposal':proposal,'baseline':baseline,'recipe_schema':RECIPE_SCHEMA,
                                               'test_schema':TEST_SCHEMA,'config_schema':CONFIG_SCHEMA}),
                    schema=IMPLEMENTATION_SCHEMA,output=directory/'implement',stage='patch_implement')
                write_json(directory/'implementation.json',implementation,exclusive=True)
                bundle=materialize_candidate(directory,proposal,implementation,baseline)
                candidate.update(status='TESTING',bundle_hash=bundle_hash(bundle))
                debug.event('candidate_test_start',candidate=cid,bundle_hash=bundle_hash(bundle))
                gate=run_gates(directory,baseline=baseline)
                candidate['gates']=gate
                if gate['status']!='PASSED': raise PolicyError('Mandatory tests failed: '+gate.get('error','unknown'))
                signature=behavior_hash(bundle['config'],registry_from_bundle(bundle))
                if signature in signatures: raise PolicyError('Duplicate executable patch; no extra consensus vote')
                if time.monotonic()-start>=max_seconds: raise BudgetExceeded('SESSION_BUDGET_EXHAUSTED')
                signatures.add(signature);admitted.append(bundle_hash(bundle))
                verify_candidate(directory)
                candidate['status']='REPLAYING';save_report(output,report)
                evaluate(bundle,cid)
                verify_candidate(directory)
                candidate['status']='EVALUATED'
            except Exception as exc:
                candidate.update(status='REJECTED',reason=f'{type(exc).__name__}: {exc}')
                (directory/'exception.txt').write_text(traceback.format_exc())
                # Invalidate every rollout of a tampered candidate before consensus.
                if candidate.get('bundle_hash'):
                    for row in report['rollouts']:
                        if row['harness_hash']==candidate['bundle_hash']:
                            row.update(status='ERROR',reason='CANDIDATE_REJECTED_AFTER_REPLAY',action=None)
                if isinstance(exc,BudgetExceeded):
                    candidate['budget_exhausted']=True
            finally:
                candidate['generation_test_replay_seconds']=time.monotonic()-started
                write_json(directory/'evaluation.json',candidate)
                debug.event('candidate_end',candidate=cid,status=candidate['status'],reason=candidate.get('reason'))
                save_report(output,report)
            if candidate.get('budget_exhausted'): break
        consensus=select_consensus(report['rollouts'],admitted,repeats=repeats,**options)
        report['consensus']=consensus
        for candidate in report['candidates']:
            if candidate['status']!='EVALUATED': continue
            directory=output/'candidates'/candidate['id']
            verify_candidate(directory)
            evaluation=evaluate_promotion(baseline_id,candidate['bundle_hash'],report['rollouts'],consensus,
                gates_passed=candidate['gates']['status']=='PASSED',min_latency_reduction=min_latency_reduction)
            candidate['status']=evaluation['status'];candidate['promotion']=evaluation
            report['promotions'].append({'candidate':candidate['id'],**evaluation})
            write_json(directory/'evaluation.json',candidate)
            if evaluation['status']=='WORKING':
                bundle=verify_candidate(directory)
                working=output/'working'/candidate['bundle_hash'];working.mkdir(parents=True,exist_ok=False)
                write_json(working/'bundle.json',bundle,exclusive=True)
                write_json(working/'evidence.json',{'schema_version':1,'bundle_hash':candidate['bundle_hash'],
                    'root_evidence_hash':frozen['evidence_hash'],'candidate_directory':str(directory.resolve()),
                    'evaluation':evaluation,'consensus':consensus,'physical_outcomes':[],
                    'status':'WORKING_OFFLINE_ONLY','created_at':now()},exclusive=True)
        report['status']='WORKING' if any(p['status']=='WORKING' for p in report['promotions']) else 'NO_PROMOTION'
    except Exception as exc:
        report.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
        (output/'exception.txt').write_text(traceback.format_exc())
    finally:
        report['totals']={**call_metrics(model.calls[call_start:],output),'elapsed_s':time.monotonic()-start,
            'call_attempts':budget.attempts,'replay_seconds':sum(r['metrics']['elapsed_s'] for r in report['rollouts']),
            'host_image_ops':sum(r['metrics']['host_image_ops'] for r in report['rollouts'])}
        report['totals']['learning_overhead_seconds']=max(0,report['totals']['elapsed_s']-report['totals']['replay_seconds'])
        debug.event('evolution_end',status=report['status'],totals=report['totals'])
        save_report(output,report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest',type=Path);source.add_argument('--evidence',type=Path)
    p.add_argument('--decision-id');p.add_argument('--output',required=True,type=Path)
    p.add_argument('--baseline-bundle',type=Path)
    p.add_argument('--backend',choices=['local','remote'],default='local');p.add_argument('--model')
    p.add_argument('--ssh-host',default='company-planner');p.add_argument('--patches',type=int,default=3)
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--max-calls',type=int,default=60)
    p.add_argument('--max-seconds',type=float,default=1800);p.add_argument('--call-timeout',type=float,default=180)
    p.add_argument('--replay-timeout',type=float,default=300);p.add_argument('--max-host-ops',type=int,default=12)
    p.add_argument('--min-latency-reduction',type=float,default=.1)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args(argv)
    try:
        if a.output.exists(): raise ValueError('Output exists; choose a new directory')
        evidence=evidence_from_manifest(read_json(a.manifest),a.decision_id) if a.manifest else load_evidence(a.evidence)
        model=RuntimeClaude(backend=a.backend,ssh_host=a.ssh_host,model=a.model,timeout_s=a.call_timeout)
        baseline=None
        if a.baseline_bundle:
            from .patch_lifecycle import verify_working
            baseline,_=verify_working(a.baseline_bundle.parent)
            if read_json(a.baseline_bundle)!=baseline: raise PolicyError('Baseline path is not the tested working bundle')
        result=evolve(evidence,a.output,model,baseline=baseline,
            source_base=(a.manifest or a.evidence).parent,patches=a.patches,repeats=a.repeats,max_calls=a.max_calls,
            max_seconds=a.max_seconds,call_timeout=a.call_timeout,replay_timeout=a.replay_timeout,max_ops=a.max_host_ops,
            min_latency_reduction=a.min_latency_reduction,prepare_only=a.prepare_only)
        return 0 if result['status'] in {'WORKING','PREPARED'} else 2
    except Exception as exc:
        if not a.output.exists():
            write_json(a.output/'blocked.json',{'status':'BLOCKED','error':f'{type(exc).__name__}: {exc}',
                       'model_calls':0,'robot_actions':0})
        print(f'{type(exc).__name__}: {exc}',flush=True);return 2


if __name__=='__main__': raise SystemExit(main())
