"""Plan -> diagnose -> propose -> implement -> gate -> replay -> working/reject.

Standalone offline entry point; no production writes or robot execution.
"""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import copy
import html
import json
import math
import re
import shutil
import statistics
import time
import traceback
from pathlib import Path

from .action_consensus import select_consensus
from .candidate_patch import (CONFIG_SCHEMA, IMPLEMENTATION_SCHEMA, PROPOSAL_SCHEMA, RECIPE_SCHEMA, TEST_SCHEMA,
    baseline_bundle, behavior_hash, bundle_hash, materialize_candidate, registry_from_bundle, run_gates, verify_candidate,
    proposal_schema, normalize_citations, validate_implementation, proposal_transport_schema, normalize_proposal_transport)
from .baseline_cache import cache_contract, load_baseline, store_baseline, load_baseline_run
from .common import canonical, digest, now, read_json, write_json
from .executors.observation import RegisteredObservationHost
from .information_flow import global_schema, validate_state
from .model import RuntimeClaude
from .policy import PolicyError, validate_schema
from .reasoning_contract import evidence_from_manifest, freeze_evidence, freeze_harness, load_evidence, verify_evidence
from .reasoning_learning import BudgetExceeded, CallBudget, DebugLog, call_metrics, run_rollout
from .host_operations import CATALOG as HOST_OPERATION_CATALOG

EXPERIENCE_OBJECTIVE = {
    'primary': 'Extract and implement a reusable reasoning method from recorded planning; test it once with real deterministic Host computation.',
    'constraints': 'Preserve blocking uncertainty and current-image grounding; do not copy historical answers.',
    'secondary': 'Report time and tokens only. No latency-based rejection or automatic promotion.',
}
EXPERIENCE_DIAGNOSE = '''Use the recorded planning and reviewed experience to propose ONE executable
HARNESS patch for grasp-and-target reasoning. Keep observation configuration unchanged: reuse the
baseline prepared images. The patch must bind at least one available deterministic host operation
between model stages: fresh measurements -> Host Python calculation -> visual decision using the
result. Pure prompt edits or image-recipe edits do not satisfy this experiment. Existing Python
primitives are reused, not newly authored code; the generated artifact is an executable harness
program with explicit data bindings. Do not promise unavailable code. Do not default all semantic
uncertainty to nonblocking; applicability of geometric calculations must be justified by current
evidence. Do not repeat the baseline or encode its action, pixels or candidate IDs in the patch.
Use level HARNESS. Cite the supplied allowed_evidence_rollouts exactly. Treat source_reasoning as
additional fallible diagnosis data and its review as corrections, not established truth. Each stage
is an independent call; context must name earlier stage IDs. Time is measured but is not an acceptance
criterion. Return the supplied proposal schema, including a concise falsifiable expected effect.
'''
EXPERIENCE_IMPLEMENT = '''Implement the proposed HARNESS using the complete supplied config schema.
Keep all baseline config outside reasoning_harness exactly unchanged. skill=null because
this is a reasoning harness implementation, not an image recipe. Use fresh measurements from an
earlier stage and host_operations on a later stage, with source_stage present in its context and
bindings exactly matching host_operation_catalog. Require at least one operation actually relevant
to the final decision. Earlier measurement stages must not allow READY before code executes.
Later reasoning must inspect computed results and current images; UNKNOWN operation output is not
usable geometry. Report blocking gaps honestly. Never copy past candidate IDs/coordinates or force
the final plan to agree with history. Return executable config, not prose suggesting future code.
Coordinate frame contracts matter: named measurements use CURRENT FULL CLEAN coordinates, so do
not apply crop-to-root transforms to measurements already in that frame. No speed gate, no robot.
'''

OPTIMIZATION_OBJECTIVE = {
    'primary': 'Reduce complete replay wall-clock time: observation calls, Host processing, image delivery and all planning stages.',
    'constraints': 'Preserve decision quality and information sufficiency; agreement alone does not prove correctness.',
    'secondary': 'Report tokens and model calls. Report diagnosis, implementation and testing overhead separately.',
    'single_repeat': 'One baseline and one candidate are an exploratory comparison, not a measurement of repeatability.',
}

RUNTIME_CAPABILITIES = {
    'editable': {'PROMPT': 'Observation/planning instruction text',
                 'HARNESS': 'Enabled skills and ordered reasoning stages, including host_operations bound to earlier measurements; each stage is a new independent call',
                 'SKILL_CODE': 'Existing pure prepare(request,source,available) image recipe',
                 'NEW_TOOL': 'New pure image-recipe skill using the same bounded API'},
    'context_semantics': 'context contains IDs of earlier stages in reasoning_harness.stages only. First stage uses []. Observation information/images are supplied automatically, never through context=[observation].',
    'skill_patch_scope': 'Exactly ONE skill object per candidate. For SKILL_CODE target must equal an existing baseline skill ID and implementation.skill.specification.id. For NEW_TOOL target must equal the new skill ID. A descriptive group name is not a valid skill target. Test multi-skill changes as separate candidates; do not claim an implementation edits several skills.',
    'not_implemented': ['Session resume/continuation across calls', 'Provider prompt-cache control',
                        'Host transport or dispatch code edits', 'Changing model or provider configuration'],
    'evidence_semantics': 'expected_information_gain describes a question, not a finding. EXECUTED_NOT_YET_INTERPRETED means a view exists, not that the information is KNOWN.',
    'cost_semantics': 'Whole-replay token counters include observation AND planning. output_tokens already includes thinking_tokens; do not add them.',
}


def stage_cost_summary(row):
    total = row['metrics']
    planning = (row.get('reasoning') or {}).get('metrics') or {}
    total_usage, plan_usage = total.get('usage') or {}, planning.get('usage') or {}
    obs_usage = {k: total_usage[k]-plan_usage[k]
                 if isinstance(total_usage.get(k),(int,float)) and isinstance(plan_usage.get(k),(int,float)) else None
                 for k in total_usage}
    return {'rollout_id':row['rollout_id'], 'total_inclusive_s':total['elapsed_s'],
            'exclusive_stage_seconds':total.get('exclusive_phases_s'),
            'total_usage_inclusive':total_usage,'planning_usage':plan_usage,
            'observation_usage':obs_usage,
            'note':'Observation usage is total minus planning; nested planning is NOT additional time or tokens.'}


def feedback_context(value):
    """Keep semantic evidence intact; summarize embedded transport stderr only.

    Reports and transport artifacts retain the complete errors. Repeating heartbeat
    dumps in every later diagnosis otherwise exhausts the prompt limit.
    """
    if isinstance(value, list):
        return [feedback_context(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in {'reason', 'error'} and isinstance(item, str) and len(item) > 2000 and (
                '__CLOTH_' in item or 'REMOTE_CLI_FAILED' in item):
            statuses = sorted(set(re.findall(r'"error_status"\s*:\s*(\d+)', item)))
            result[key] = (item.split('__CLOTH_', 1)[0][:500] +
                '\n[Transport log summarized; full text remains in the source report and transport/stderr.log.]' +
                '\nRemote HTTP error statuses: ' + ', '.join(statuses) + '\nTail: ' + item[-1000:])
        else:
            result[key] = feedback_context(item)
    return result


DIAGNOSE = '''Read iteration_feedback before proposing. It contains actual earlier implementations, outcomes,
measured costs, changed actions and provisional failed checks. Revise an earlier mechanism or choose a
new one based on these results; explain which feedback motivates the change in evidence_explanation.
Do not simply repeat a failed proposal. Treat execution failures as unavailable measurements, not fast
visual decisions. Single-run latency includes service waiting and is not proof of model reasoning cost.
All implementations remain complete replacements relative to the fixed baseline, not incremental diffs.
Previous outputs are diagnosis evidence only: never encode their answers in the new patch.
Your primary objective is to REDUCE TOTAL observation-plus-planning wall-clock time
while preserving decision quality and information sufficiency. Use the supplied measured stage costs
to identify work a patch can remove, combine, reuse or execute more cheaply. Freely choose an approach;
these are possibilities, not a fixed recipe. Do not add stages merely because answers differ or optimize
for longer explanations. Explain which work is replaced and why total time, including added model calls,
image delivery and Host execution, should fall. Tokens and call count are secondary metrics, not speed
proof. State a falsifiable expected effect; never force READY by hiding a blocking information gap.
The evaluation_policy specifies the required measured latency reduction and consensus settings.
Single-repeat comparisons are exploratory. Correctness and generalization still need independent review.
Diagnose the public planning trace and costs; propose ONE testable change to HOW the
agent acquires or reasons about evidence, not the action to take. Select PROMPT, HARNESS, SKILL_CODE,
or NEW_TOOL. Preserve coordinate alignment, original provenance, bounded budgets, no robot access,
and pre-decision input isolation. Separate observed repetition/cost from unverified usefulness.
Do not claim a tool is unnecessary merely because final prose omitted it. Output a PatchProposal only;
it does not change runtime behavior. For code, target ONLY observation skill implementations.
No historical candidate, target, fixed scene ROI, answer lookup or covert answer encoding in the patch.
evidence_rollouts must contain only exact IDs from allowed_evidence_rollouts. Put citation explanations
in evidence_explanation, not inside an ID. Image IDs are not rollout IDs.
must_preserve is ONE quoted JSON string in the transport schema. Separate invariants within that string;
never emit an unquoted sentence or repeat the must_preserve key. Keep prose concise.
Use metrics.exclusive_phases_s for stage costs: elapsed_s ALREADY includes planning. Never add nested
reasoning elapsed to total. Aggregate model_calls includes observation AND planning; image operations
are Host work, not separate model round-trips. Do not infer stage durations from token counts.
Planning-only candidates automatically reuse the baseline pre-planning images and information.
Their planning time is directly compared; total comparison shares the historical observation cost
and is reconstructed, not a new full-pipeline measurement.
Read runtime_capabilities before choosing a mechanism. Session continuation is NOT supported;
context references earlier reasoning stages only and does not create sessions or cache reuse.
Choose an executable mechanism within these capabilities, or explain the unsupported dependency
in unverified; never disguise a missing backend capability as prompt text.
Only PROMPT text, HARNESS config/topology, and bounded skill programs can be changed here. There is no
editable Host-dispatch gate or resolved-gap ledger API. A text instruction is model-guided behavior,
not a deterministic implementation. For deterministic image work propose SKILL_CODE/NEW_TOOL within
the recipe API. KNOWN does not imply an observation is redundant or its artifact already delivered.
'''
IMPLEMENT = '''Implement the candidate proposal in this isolated workspace contract; its benefit is
not yet approved or verified. Host dispatch is not editable in this contract. Do not describe
a prompt instruction as a deterministic gate; config has no ALREADY_RESOLVED/DEFERRED dispatcher.
If a proposal exceeds capabilities, implement its feasible part and treat timing predictions for the
unimplemented part as unverified. Preserve its time-saving mechanism and information-sufficiency conditions.
Do not add unrelated stages or checks that change the experiment. Total replay time includes observation,
image handling, Host operations and planning; faster arithmetic alone is not a measured speedup. Return
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
ROI is in normalized selected source image coordinates initially; later recipe crops are relative to
preceding recipe output. Both roles get the identical operation sequence.
Implement dynamic geometry from current inputs; never insert baseline answers, historical pixels,
image-specific constants or hidden lookup tables. Source tests are synthetic geometry, not live answers.
SKILL_CODE: target existing ID, version increments exactly once. Preserve all original valid inputs.
NEW_TOOL: new ID, version one. Code patches may also update observation/reasoning prompts to use the changed implementation. Explain combined changes; evaluation measures the whole patch, not isolated code benefit;
only NEW_TOOL may add its ID to enabled_skills. PROMPT changes text only. HARNESS may change configuration
and stage topology within allowed schemas. No code for configuration-only patches.
'''
OBSERVE = '''On these CURRENT pre-decision inputs identify selection-critical visual information gaps.
Do not select grasp or target. A known finding must cite an attached image; an UNKNOWN gap must say
what is missing. Bind a bounded set of registered observation requests to any available image ID, including intermediate views. Preserve the source ID; Host records the full reference chain and root-coordinate mapping.
No old planner crops, previous rollout conclusions or action/evaluation are available. The selected
implementation executes on Host; a processed image alone does not prove that a gap has been resolved.
UNKNOWN is not automatically blocking: depth, friction and physical execution checks may remain unknown
for RGB-only planning. Request useful observations, not one request per unknown. A view may answer
multiple questions. The final planner must decide which unresolved gaps actually block a visual action.
KNOWN and UNKNOWN describe information confidence, not permission to process images. You may request
an operation for either status, e.g. rotate by a known angle to simplify later interpretation. gap_id
must reference an existing information item; explain the expected benefit in expected_information_gain.
Keep its original status unless new evidence changes it; do not relabel KNOWN as UNKNOWN to request a view.
'''


@contextmanager
def measured_phase(phases, name):
    started = time.monotonic()
    try:
        yield
    finally:
        phases[name] += time.monotonic() - started


def implementation_effects(baseline, implementation):
    old, new = baseline['config'], implementation['config']
    def topology(h):
        return {**h, 'name': '', 'applicability': '',
                'stages': [{**stage, 'instruction': ''} for stage in h['stages']]}
    return {
        'observation_prompt_changed': old['observation_instruction'] != new['observation_instruction'],
        'reasoning_harness_changed': old['reasoning_harness'] != new['reasoning_harness'],
        'stage_topology_changed': topology(old['reasoning_harness']) != topology(new['reasoning_harness']),
        'enabled_skills_changed': old['enabled_skills'] != new['enabled_skills'],
        'skill_code_generated': implementation['skill'] is not None,
        'host_dispatch_code_changed': False,
        'scope': 'Measured config/source changes; prose does not implement a deterministic Host gate.'}


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
    phases = dict(observation_call_s=0., host_dispatch_s=0., planning_s=0.)
    try:
        registry=registry_from_bundle(bundle)
        root=copy.deepcopy(frozen['evidence'])
        paths=verify_evidence(frozen,evidence_dir)
        config=bundle['config']
        catalog=[s for s in registry.catalog() if s['id'] in config['enabled_skills']]
        host=RegisteredObservationHost(root,paths,{'skills':catalog},output/'observations',max_ops,registry=registry,reuse_observations=True,defer_over_budget=True)
        # Restrict model-visible IDs to enabled entries, not all installed capabilities.
        from .skills import SkillRegistry
        enabled=SkillRegistry()
        for name in config['enabled_skills']: enabled.register(registry.get(name))
        payload={'current_evidence':root,'instruction':config['observation_instruction'],'skill_specifications':catalog,
                 'remaining_host_ops':max_ops}
        with measured_phase(phases, 'observation_call_s'):
            observation=budget.invoke(model,frozen=frozen,evidence_dir=evidence_dir,prompt=OBSERVE+canonical(payload),
                schema=global_schema(enabled),output=output/'observe',stage='patch_observe',deadline=start+timeout)
        validate_schema(observation,global_schema(enabled))
        warnings=validate_state(observation['information'],host.catalog)
        row['information_warnings']=warnings
        write_json(output/'information_validation.json',{'status':'ACCEPTED_WITH_WARNINGS' if warnings else 'ACCEPTED',
                                                        'warnings':warnings,'original_preserved':True})
        if warnings:
            budget.debug.event('information_warnings',rollout_id=rollout_id,warnings=warnings)
        write_json(output/'observation.json',observation)
        requested=observation['observation_requests']
        with measured_phase(phases, 'host_dispatch_s'):
            stop=host.execute(requested,observation['information'])
        if stop: row.update(status='NEEDS_LEARNING',reason=stop)
        else:
            # Fresh roots plus candidate-generated views; separate root/derived hashes.
            prepared=output/'prepared'; prepared.mkdir()
            augmented=copy.deepcopy(root)
            augmented['observation_requirements']={
                'information':observation['information'], 'information_warnings':warnings,
                'observation_results':host.history, 'deferred_observations':host.deferred,
                'instruction':'Deferred observations were NOT executed and provide no new evidence. Assess whether existing views suffice; if a deferred gap blocks the decision, return NEEDS_LEARNING. Use the supplied views to assess the recorded information needs. UNKNOWN does not automatically block RGB visual planning. State genuinely blocking gaps in missing_information and return NEEDS_LEARNING; retain non-blocking physical checks in residual_uncertainty. Never hide a blocking gap to force READY. Cite images supporting the decision; concept names are free, exact gap IDs are not required. Reused images are not independent new evidence. Host execution alone does not resolve any gap.'}
            for i,path in enumerate(host.paths): shutil.copyfile(path,prepared/f'image_{i}.png')
            for item in host.catalog[len(root['images']):]:
                augmented['images'].append({k:item[k] for k in ('image_id','role','size','rgb_sha256','original_image_id','to_original','parent_image_id','to_parent','operation','arguments','lineage')} |
                    {'role':item['role']+'_crop','file':item['image_id']+'.png'})
            wrapped={'evidence':augmented,'evidence_hash':digest(augmented)}
            write_json(prepared/'evidence.json',wrapped)
            reasoning=freeze_harness(config['reasoning_harness'],output/'reasoning_version.json',source='candidate_bundle')
            with measured_phase(phases, 'planning_s'):
                result=run_rollout(reasoning,wrapped,prepared,model,output/'reasoning',budget,
                                   rollout_id=rollout_id,timeout=max(.001,timeout-(time.monotonic()-start)))
            row.update(status=result['status'],reason=result['reason'],action=result['action'],
                       reasoning=result,derived_evidence_hash=wrapped['evidence_hash'])
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
        phases['other_s'] = max(0., row['metrics']['elapsed_s'] - sum(phases.values()))
        row['metrics']['exclusive_phases_s'] = phases
        row['metrics']['timing_semantics'] = 'elapsed_s is TOTAL observation + Host + planning + other. Exclusive phases sum to total. Nested reasoning metrics and host_seconds are already included; never add them again. Model calls/tool_calls are aggregate, not observation round counts.'
        row['deferred_observations']=host.deferred if host else []
        row['observation_trace']=host.history if host else []
        write_json(output/'call_audits.json',calls)
        write_json(output/'result.json',row)
        budget.debug.event('patch_replay_end',rollout_id=rollout_id,status=row['status'],reason=row['reason'],metrics=row['metrics'])
    return row


def observation_signature(bundle):
    return digest({'instruction':bundle['config']['observation_instruction'],
                   'enabled_skills':bundle['config']['enabled_skills'],
                   'registry':registry_from_bundle(bundle).snapshot()})


def replay_fixed_observation(bundle, baseline_row, baseline_directory, frozen, evidence_dir,
                             model, output, budget, *, rollout_id, timeout):
    """Only pre-planning evidence is reused; baseline decisions never enter the planner."""
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); call_start=len(model.calls)
    row={'rollout_id':rollout_id,'harness_hash':bundle_hash(bundle),
         'root_evidence_hash':frozen['evidence_hash'], 'observation_id':baseline_row['observation_id'],
         'observation_rgb_sha256':baseline_row['observation_rgb_sha256'],
         'status':'ERROR','action':None,'reason':'UNFINISHED',
         'actual_measurement':model.configuration.get('actual_measurement',True)}
    phases=dict(observation_call_s=0.,host_dispatch_s=0.,planning_s=0.)
    try:
        verify_evidence(frozen,evidence_dir)
        source=Path(baseline_directory)/'prepared'
        wrapped=read_json(source/'evidence.json')
        verify_evidence(wrapped,source)
        if baseline_row['root_evidence_hash']!=frozen['evidence_hash'] or wrapped['evidence_hash']!=baseline_row['derived_evidence_hash']:
            raise PolicyError('Baseline observation identity mismatch')
        prepared=output/'prepared'; shutil.copytree(source,prepared)
        verify_evidence(wrapped,prepared)
        row['observation_reuse']={'source_rollout':baseline_row['rollout_id'],
            'source_directory':str(Path(baseline_directory).resolve()),'evidence_hash':wrapped['evidence_hash'],
            'scope':'Identical pre-planning images and information; no baseline action or reasoning supplied'}
        reasoning=freeze_harness(bundle['config']['reasoning_harness'],output/'reasoning_version.json',source='candidate_bundle')
        with measured_phase(phases,'planning_s'):
            result=run_rollout(reasoning,wrapped,prepared,model,output/'reasoning',budget,
                               rollout_id=rollout_id,timeout=max(.001,timeout-(time.monotonic()-started)))
        row.update(status=result['status'],reason=result['reason'],action=result['action'],reasoning=result,
                   derived_evidence_hash=wrapped['evidence_hash'])
        verify_evidence(wrapped,prepared)
        if time.monotonic()-started>timeout: row.update(status='BUDGET_EXHAUSTED',reason='REPLAY_DEADLINE',action=None)
    except Exception as exc:
        row.update(status='ERROR',reason=f'{type(exc).__name__}: {exc}',action=None)
        (output/'exception.txt').write_text(traceback.format_exc())
    finally:
        calls=copy.deepcopy(model.calls[call_start:])
        elapsed=time.monotonic()-started
        phases['other_s']=max(0.,elapsed-sum(phases.values()))
        row['metrics']={**call_metrics(calls,output),'elapsed_s':elapsed,'host_seconds':0.,
            'exclusive_phases_s':phases,'timing_semantics':'Actual current planning replay only. Observation is reused, not timed again.'}
        base_phases=baseline_row['metrics']['exclusive_phases_s']
        row['metrics']['comparison_elapsed_s']=baseline_row['metrics']['elapsed_s']-base_phases['planning_s']+phases['planning_s']
        row['metrics']['comparison_semantics']='Reconstructed total with shared historical baseline non-planning cost; not measured end-to-end latency.'
        write_json(output/'call_audits.json',calls); write_json(output/'result.json',row)
    return row


def comparison_rows(rows):
    """Use the same shared observation cost for consensus latency ranking."""
    result=copy.deepcopy(rows)
    for row in result:
        row['metrics']['elapsed_s']=row['metrics'].get('comparison_elapsed_s',row['metrics']['elapsed_s'])
    return result


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
    new=statistics.median(r['metrics'].get('comparison_elapsed_s',r['metrics']['elapsed_s']) for r in candidate) if candidate else None
    reduction=(old-new)/old if old and new is not None and checks['valid_complete_plans'] and checks['baseline_valid'] else None
    checks['latency_improved']=reduction is not None and reduction>=min_latency_reduction
    checks['no_extra_unresolved_information']=all(r['status']=='READY' for r in candidate)
    return {'status':'WORKING' if all(checks.values()) else 'REJECTED', 'checks':checks,
            'latency_reduction':reduction,'required_latency_reduction':min_latency_reduction,
            'baseline_median_seconds':old,'candidate_median_seconds':new,
            'comparison_scope':'SHARED_OBSERVATION_RECONSTRUCTED_TOTAL' if any(r.get('observation_reuse') for r in candidate) else 'MEASURED_FULL_REPLAY',
            'baseline_planning_seconds':statistics.median(r['metrics']['exclusive_phases_s']['planning_s'] for r in baseline) if baseline and all('exclusive_phases_s' in r['metrics'] for r in baseline) else None,
            'candidate_planning_seconds':statistics.median(r['metrics']['exclusive_phases_s']['planning_s'] for r in candidate) if candidate and all('exclusive_phases_s' in r['metrics'] for r in candidate) else None,
            'grounding_scope':'Current-registry and visual pixel coordinates only; physical grounding not performed',
            'physical_status':'PENDING','frozen_eligible':False,'robot_executable':False}


def save_report(output,report):
    write_json(output/'report.json',report)
    lines=['# Candidate patch evolution','',f"Status: {report['status']}",
        'Offline only. Candidate code is tested before replay; working is not frozen and does not authorize robot execution.','',
        '| Version | State | Reason |','|---|---|---|']
    for c in report['candidates']:
        lines.append(f"| {c['id']} | {c['status']} | {str(c.get('reason','')).replace('|','/')} |")
    lines+=['','## Measurements and promotion','','```json',json.dumps({k:report.get(k) for k in ('stop_reason','baseline_cache','consensus','promotions','totals')},ensure_ascii=False,indent=2),'```',
            '', 'Same-state compression/evolution only. Stable consensus does not prove physical correctness. All generation/testing/reflection overhead is reported separately.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    page='<h1>Candidate patch evolution</h1><a href="report.json">Report JSON</a> · <a href="events.jsonl">Debug events</a><pre>'+html.escape('\n'.join(lines))+'</pre>'
    for c in report['candidates']:
        name=c['id']
        page+=f'<details><summary>{name}: {html.escape(c["status"])}</summary>'
        for filename in ('diagnosis_context.json','feedback.json','proposal.json','implementation.json','patch.diff','gates.json','evaluation.json'):
            page+=f'<p><a href="candidates/{name}/{filename}">{filename}</a></p>'
        page+='<pre>'+html.escape(json.dumps(c,ensure_ascii=False,indent=2))+'</pre></details>'
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Patch evolution debug</title><style>body{max-width:1200px;margin:auto;padding:24px;font:16px system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ccc;padding:12px;margin:12px}</style>'+page)


def evolve(evidence,output,model,*,baseline=None,patches=3,repeats=2,max_calls=60,max_seconds=1800,
           call_timeout=180,replay_timeout=300,max_ops=12,min_latency_reduction=.1,consensus_options=None,
           source_base=None,prepare_only=False,baseline_cache_dir=None,reuse_baseline_run=None,retry_candidate=None,max_consecutive_failures=3,resume_run=None,
           learning_mode='optimization',reasoning_record=None):
    if learning_mode not in ('optimization','experience'):
        raise ValueError('Unknown learning mode')
    experience_mode=learning_mode=='experience'
    if reasoning_record and not experience_mode:
        raise ValueError('Reasoning record requires experience mode')
    objective=EXPERIENCE_OBJECTIVE if experience_mode else OPTIMIZATION_OBJECTIVE
    if not 1<=patches<=8 or not 1<=repeats<=5 or not 1<=max_calls<=200 or not 0<=max_ops<=24:
        raise ValueError('Invalid finite evolution budget')
    if any(not math.isfinite(v) or v<=0 for v in (max_seconds,call_timeout,replay_timeout)) or not 0<min_latency_reduction<1:
        raise ValueError('Invalid timeout or promotion threshold')
    if not 1 <= max_consecutive_failures <= 8:
        raise ValueError('Invalid consecutive failure limit')
    if retry_candidate and patches != 1:
        raise ValueError('Retry one saved candidate with --patches 1')
    if retry_candidate and resume_run:
        raise ValueError('Choose retry-candidate or resume-run, not both')
    options={'min_harnesses': min(3, patches+1), **(consensus_options or {})}
    select_consensus([],[],repeats=repeats,**options)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    debug=DebugLog(output); start=time.monotonic(); call_start=len(model.calls)
    budget=CallBudget(max_calls=max_calls,max_seconds=max_seconds,call_timeout=call_timeout,prompt_chars=200000,debug=debug)
    report={'status':'PREPARING','candidates':[],'rollouts':[],'promotions':[],
            'actual_measurement':model.configuration.get('actual_measurement',True),'model_configuration':model.configuration,
            'settings':{'patches':patches,'repeats':repeats,'max_calls':max_calls,'max_seconds':max_seconds,
                'replay_timeout':replay_timeout,'call_timeout':call_timeout,'max_ops':max_ops,
                'min_latency_reduction':min_latency_reduction,'consensus':options,
                'max_consecutive_failures':max_consecutive_failures,'learning_mode':learning_mode},'robot_actions':0,
            'optimization_objective':objective,'automatic_activation':False}
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
                if name!='baseline' and observation_signature(bundle)==observation_signature(baseline):
                    base=parent[repeat]
                    root_dir=output/'cached_baseline' if base.get('baseline_cache',{}).get('reused') else output
                    row=replay_fixed_observation(bundle,base,root_dir/'replays'/base['rollout_id'],
                        frozen,output/'evidence',model,output/'replays'/rid,budget,rollout_id=rid,timeout=replay_timeout)
                else:
                    row=replay_bundle(bundle,frozen,output/'evidence',model,output/'replays'/rid,budget,
                        rollout_id=rid,timeout=replay_timeout,max_ops=max_ops)
                report['rollouts'].append(row)
                save_report(output,report)
                if report['rollouts'][-1]['status']=='BUDGET_EXHAUSTED': break
        report['status']='BASELINE'
        contract=cache_contract(frozen,baseline_id,model.configuration,repeats=repeats,max_ops=max_ops,
                                replay_timeout=replay_timeout,call_timeout=call_timeout)
        if reuse_baseline_run:
            cached,cache_info=load_baseline_run(reuse_baseline_run,contract,output)
        else:
            cached, cache_info = load_baseline(baseline_cache_dir,contract,output) if baseline_cache_dir else ([], {'status':'DISABLED'})
        report['baseline_cache']=cache_info
        if cached:
            report['rollouts'].extend(cached)
        else:
            evaluate(baseline,'baseline')
            rows=[r for r in report['rollouts'] if r['harness_hash']==baseline_id]
            if baseline_cache_dir:
                try:
                    report['baseline_cache']['stored']=store_baseline(baseline_cache_dir,contract,rows,output)
                except OSError as exc:
                    report['baseline_cache'].update(stored=False,write_warning=str(exc))
        parent=[r for r in report['rollouts'] if r['harness_hash']==baseline_id]
        # Serialization/infrastructure failure is not a visual optimization trace.
        if not any(r['status']=='READY' for r in parent):
            report['status']='BLOCKED_BASELINE';return report
        source_reasoning=None
        if reasoning_record:
            source=Path(reasoning_record)
            source_report=read_json(source/'report.json')
            if source_report['evidence_hash']!=parent[0]['derived_evidence_hash']:
                raise PolicyError('Supplemental reasoning used different prepared images')
            source_reasoning={'source':str(source.resolve()),
                'result':read_json(source/'reasoning/result.json'),
                'draft_experience':read_json(source/'experience.json'),
                'review':(source/'review.md').read_text()}
            write_json(output/'source_reasoning.json',source_reasoning)
        admitted=[baseline_id]; signatures={behavior_hash(baseline['config'],registry)}
        if resume_run:
            source=Path(resume_run)
            previous=read_json(source/'report.json')
            if previous['settings'].get('learning_mode','optimization')!=learning_mode:
                raise PolicyError('Resume learning mode mismatch')
            if previous['status'] not in {'NO_PROMOTION','WORKING','ERROR','EXPERIMENT_COMPLETE','EXPERIMENT_INCOMPLETE'}:
                raise PolicyError('Resume requires a terminal source report')
            if previous['root_evidence_hash']!=frozen['evidence_hash'] or previous['baseline_hash']!=baseline_id:
                raise PolicyError('Resume evidence or baseline mismatch')
            for key in ('repeats','call_timeout','replay_timeout','max_ops'):
                if previous['settings'][key]!=report['settings'][key]:
                    raise PolicyError('Resume setting mismatch: '+key)
            if previous['model_configuration']!=report['model_configuration']:
                raise PolicyError('Resume model configuration mismatch')
            completed=copy.deepcopy(previous['candidates'])
            # A prompt-limit stop before diagnosis produced no proposal or replay;
            # retry that index after fixing the oversized input, retaining its source log.
            if completed and completed[-1].get('budget_exhausted') and not completed[-1].get('proposal'):
                completed.pop()
            if len(completed)>=patches:
                raise PolicyError('No remaining candidates; increase --patches to the total desired count')
            for candidate in completed:
                cid=candidate['id']
                if 'feedback' not in candidate:
                    raise PolicyError('Resume candidate lacks completed feedback: '+cid)
                if candidate.get('gates',{}).get('status')=='PASSED':
                    bundle=verify_candidate(source/'candidates'/cid)
                    signatures.add(behavior_hash(bundle['config'],registry_from_bundle(bundle)))
                    admitted.append(bundle_hash(bundle))
                shutil.copytree(source/'candidates'/cid,output/'candidates'/cid)
                candidate['resumed_from']=str(source.resolve())
                if candidate.get('promotion'):
                    candidate['status']='EVALUATED'
                report['candidates'].append(candidate)
                for old in previous['rollouts']:
                    if old['rollout_id'].startswith(cid+'_r'):
                        row=copy.deepcopy(old)
                        row['resumed_from']=str(source.resolve())
                        shutil.copytree(source/'replays'/row['rollout_id'],output/'replays'/row['rollout_id'])
                        report['rollouts'].append(row)
            report['resumed_from']=str(source.resolve())
            debug.event('evolution_resumed',source=str(source.resolve()),completed_candidates=len(completed))
        consecutive_failures=0
        for index in range(len(report['candidates']),patches):
            if consecutive_failures >= max_consecutive_failures:
                report['stop_reason']='CONSECUTIVE_EXECUTION_FAILURES'
                break
            if budget.attempts >= max_calls or time.monotonic()-start >= max_seconds:
                report['stop_reason']='SESSION_BUDGET_EXHAUSTED'
                break
            cid=f'patch_{index:02d}'; directory=output/'candidates'/cid; directory.mkdir(parents=True)
            candidate={'id':cid,'status':'DIAGNOSING'};report['candidates'].append(candidate)
            report['status']='EVOLVING'
            save_report(output,report)
            started=time.monotonic()
            debug.event('candidate_start',candidate=cid)
            try:
                if retry_candidate:
                    source=Path(retry_candidate)
                    verify_candidate(source)
                    if read_json(source/'base_version.json')['bundle_hash'] != baseline_id:
                        raise PolicyError('Saved candidate uses a different baseline bundle')
                    source_report=read_json(source.parent.parent/'report.json')
                    if source_report['root_evidence_hash'] != frozen['evidence_hash']:
                        raise PolicyError('Saved candidate was generated for different evidence')
                    proposal=read_json(source/('proposal_normalized.json' if (source/'proposal_normalized.json').exists() else 'proposal.json'))
                    implementation=read_json(source/'implementation.json')
                    candidate.update(status='IMPLEMENTING',proposal=proposal,source_candidate=str(source.resolve()))
                    write_json(directory/'proposal.json',proposal,exclusive=True)
                    write_json(directory/'proposal_normalized.json',proposal,exclusive=True)
                    debug.event('candidate_reused',candidate=cid,source=str(source.resolve()))
                else:
                    observed=feedback_context(report['rollouts'])
                    feedback=feedback_context([c['feedback'] for c in report['candidates'][:-1]])
                    diagnosis={'baseline':baseline,'observed_rollouts':observed,
                               'iteration_feedback':feedback,
                               'allowed_evidence_rollouts':[r['rollout_id'] for r in observed],
                               'optimization_objective':objective,
                               'evaluation_policy':({'latency_gate':False,'automatic_promotion':False,'require_actual_host_code':True}
                                   if experience_mode else {'min_latency_reduction':min_latency_reduction,
                                                    'consensus':options,'repeats':repeats}),
                               'previous_proposals':[c.get('proposal') for c in report['candidates'][:-1]],
                               'available_levels':['HARNESS'] if experience_mode else ['PROMPT','HARNESS','SKILL_CODE','NEW_TOOL'],
                               'runtime_capabilities':RUNTIME_CAPABILITIES,
                               'host_operation_catalog':HOST_OPERATION_CATALOG,
                               'source_reasoning':source_reasoning,
                               'stage_costs':[stage_cost_summary(r) for r in observed]}
                    write_json(directory/'diagnosis_context.json',diagnosis,exclusive=True)
                    proposal=budget.invoke(model,frozen=frozen,evidence_dir=output/'evidence',prompt=(EXPERIENCE_DIAGNOSE if experience_mode else DIAGNOSE)+canonical(diagnosis),
                        schema=proposal_transport_schema(r['rollout_id'] for r in observed),output=directory/'diagnose',stage='patch_diagnose')
                    write_json(directory/'proposal_transport.json',proposal,exclusive=True)
                    proposal=normalize_proposal_transport(proposal)
                    write_json(directory/'proposal.json',proposal,exclusive=True)
                    proposal, citation_audit=normalize_citations(proposal,[r['rollout_id'] for r in observed])
                    write_json(directory/'citation_audit.json',citation_audit)
                    if not proposal['evidence_rollouts']:
                        # Repair only references, once; never rerun the baseline or replace the proposal.
                        repair_schema={'type':'object','additionalProperties':False,'required':['evidence_rollouts'],
                            'properties':{'evidence_rollouts':{'type':'array','maxItems':20,'uniqueItems':True,
                                'items':{'type':'string','enum':[r['rollout_id'] for r in observed]}}}}
                        repair=budget.invoke(model,frozen=frozen,evidence_dir=output/'evidence',
                            prompt='Repair citation IDs only. Select actual supplied rollouts supporting this unchanged proposal. '
                                   'Do not invent support; return an empty list if none support it. '+canonical({
                                       'proposal':citation_audit['original'],'observed_rollouts':observed}),
                            schema=repair_schema,output=directory/'citation_repair',stage='patch_citation_repair')
                        write_json(directory/'citation_repair.json',repair)
                        validate_schema(repair,repair_schema)
                        proposal['evidence_rollouts']=repair['evidence_rollouts']
                        if not proposal['evidence_rollouts']:
                            raise PolicyError('No supported rollout citation after one local citation repair')
                    validate_schema(proposal,proposal_schema(r['rollout_id'] for r in observed))
                    write_json(directory/'proposal_normalized.json',proposal)
                    candidate.update(status='IMPLEMENTING',proposal=proposal)
                    save_report(output,report)
                    implementation=budget.invoke(model,frozen=frozen,evidence_dir=output/'evidence',
                        prompt=(EXPERIENCE_IMPLEMENT if experience_mode else IMPLEMENT)+canonical({'proposal':proposal,'citation_notes':citation_audit['entries'],
                                                   'baseline':baseline,'iteration_feedback':feedback,'recipe_schema':RECIPE_SCHEMA,
                                                   'optimization_objective':objective,'host_operation_catalog':HOST_OPERATION_CATALOG,
                                                   'test_schema':TEST_SCHEMA,'config_schema':CONFIG_SCHEMA,
                                                   'runtime_capabilities':RUNTIME_CAPABILITIES}),
                        schema=IMPLEMENTATION_SCHEMA,output=directory/'implement',stage='patch_implement')
                write_json(directory/'implementation_initial.json',implementation,exclusive=True)
                try:
                    validate_implementation(proposal,implementation,baseline)
                except PolicyError as exc:
                    repair_context={'proposal':proposal, 'implementation':implementation,
                        'validation_error':str(exc),'baseline':baseline,
                        'runtime_capabilities':RUNTIME_CAPABILITIES,
                        'host_operation_catalog':HOST_OPERATION_CATALOG,'learning_mode':learning_mode,
                        'config_schema':CONFIG_SCHEMA,'recipe_schema':RECIPE_SCHEMA,'test_schema':TEST_SCHEMA}
                    write_json(directory/'implementation_repair_request.json',repair_context)
                    debug.event('implementation_repair_start',candidate=cid,error=str(exc))
                    # One bounded repair, no baseline rerun or silent reference deletion.
                    repair_schema={'type':'object','additionalProperties':False,
                        'properties':{'status':{'enum':['REPAIRED','NEEDS_RUNTIME_SUPPORT']},
                            'reason':{'type':'string'},
                            'implementation':{'anyOf':[IMPLEMENTATION_SCHEMA,{'type':'null'}]}},
                        'required':['status','reason','implementation']}
                    repair=budget.invoke(model,frozen=frozen,evidence_dir=output/'evidence',
                        prompt='Repair this implementation once, preserving the proposed mechanism. Fix structural errors using the actual runtime contract. Do not delete an unavailable session reference and claim session continuation works. If the mechanism needs unsupported runtime behavior, return NEEDS_RUNTIME_SUPPORT and null implementation. No new proposal or baseline run. '+canonical(repair_context),
                        schema=repair_schema,output=directory/'implementation_repair',stage='patch_implementation_repair')
                    write_json(directory/'implementation_repair.json',repair)
                    validate_schema(repair,repair_schema)
                    if repair['status']=='NEEDS_RUNTIME_SUPPORT':
                        candidate.update(status='NEEDS_RUNTIME_SUPPORT',reason=repair['reason'])
                        continue
                    implementation=repair['implementation']
                    validate_implementation(proposal,implementation,baseline)
                    candidate['implementation_repaired']=True
                write_json(directory/'implementation.json',implementation,exclusive=True)
                if experience_mode:
                    if proposal['level']!='HARNESS' or implementation.get('skill') is not None or any(
                            implementation['config'][key]!=baseline['config'][key]
                            for key in baseline['config'] if key!='reasoning_harness'):
                        raise PolicyError('Experience mode requires a reasoning-only HARNESS patch')
                    if not any(s.get('host_operations') for s in implementation['config']['reasoning_harness']['stages']):
                        raise PolicyError('Experience implementation must bind executable Host operations, not prose only')
                candidate['implementation_effects'] = implementation_effects(baseline, implementation)
                write_json(directory/'implementation_effects.json', candidate['implementation_effects'])
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
                rows=[r for r in report['rollouts'] if r['rollout_id'].startswith(cid+'_r')]
                interim=select_consensus(comparison_rows(report['rollouts']),admitted,repeats=repeats,**options)
                assessment=evaluate_promotion(baseline_id,candidate['bundle_hash'],report['rollouts'],interim,
                    gates_passed=candidate.get('gates',{}).get('status')=='PASSED',
                    min_latency_reduction=min_latency_reduction) if candidate.get('bundle_hash') and not experience_mode else None
                execution_failed=(any(r['status'] in {'ERROR','BUDGET_EXHAUSTED'} for r in rows)
                                  or (not rows and candidate['status']=='REJECTED'))
                consecutive_failures=consecutive_failures+1 if execution_failed else 0
                candidate['feedback']={'candidate_id':cid,'status':candidate['status'],
                    'reason':candidate.get('reason'),'proposal':candidate.get('proposal'),
                    'implementation':read_json(directory/'implementation.json') if (directory/'implementation.json').exists() else None,
                    'implementation_effects':candidate.get('implementation_effects'),
                    'rollouts':[{'rollout_id':r['rollout_id'],'status':r['status'],'reason':r.get('reason'),
                                 'action':r.get('action'),'costs':stage_cost_summary(r)} for r in rows],
                    'provisional_evaluation':assessment,
                    'failed_checks':[k for k,v in assessment['checks'].items() if not v] if assessment else [],
                    'evaluation_note':('Experience trial: time is telemetry, not a rejection criterion; actual code execution is checked.' if experience_mode else
                        'Provisional against versions tested so far; consensus is recomputed after all candidates. Failed execution is not a latency improvement.'),
                    'host_execution':[op for r in rows for op in (r.get('reasoning') or {}).get('host_operations',[])],
                    'consecutive_execution_failures':consecutive_failures}
                write_json(directory/'feedback.json',candidate['feedback'])
                candidate['generation_test_replay_seconds']=time.monotonic()-started
                write_json(directory/'evaluation.json',candidate)
                debug.event('candidate_end',candidate=cid,status=candidate['status'],reason=candidate.get('reason'))
                save_report(output,report)
            if candidate.get('budget_exhausted'):
                report['stop_reason']='SESSION_BUDGET_EXHAUSTED'
                break
        consensus=select_consensus(comparison_rows(report['rollouts']),admitted,repeats=repeats,**options)
        report['consensus']=consensus
        if experience_mode:
            for candidate in report['candidates']:
                if candidate['status']!='EVALUATED': continue
                rows=[r for r in report['rollouts'] if r['rollout_id'].startswith(candidate['id']+'_r')]
                ops=[op for r in rows for op in (r.get('reasoning') or {}).get('host_operations',[])]
                candidate['experience_evaluation']={'host_code_executed':bool(ops),
                    'computed_operations':sum(op['status']=='COMPUTED' for op in ops),
                    'planning_statuses':[r['status'] for r in rows],
                    'semantic_correctness':'NOT_INDEPENDENTLY_VERIFIED','latency_gate':False}
                candidate['status']='TESTED' if any(op['status']=='COMPUTED' for op in ops) else 'CODE_NOT_EXERCISED'
                write_json(output/'candidates'/candidate['id']/'evaluation.json',candidate)
            report['status']='EXPERIMENT_COMPLETE' if any(c['status']=='TESTED' for c in report['candidates']) else 'EXPERIMENT_INCOMPLETE'
            return report
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
        excluded=['cached_baseline']
        excluded.extend('candidates/'+c['id'] for c in report['candidates'] if c.get('resumed_from'))
        excluded.extend('replays/'+r['rollout_id'] for r in report['rollouts'] if r.get('resumed_from'))
        report['totals']={**call_metrics(model.calls[call_start:],output,exclude_dirs=excluded),'elapsed_s':time.monotonic()-start,
            'call_attempts':budget.attempts,'replay_seconds':sum(r['metrics']['elapsed_s'] for r in report['rollouts'] if not r.get('baseline_cache',{}).get('reused') and not r.get('resumed_from')),
            'host_image_ops':sum(r['metrics']['host_image_ops'] for r in report['rollouts'] if not r.get('baseline_cache',{}).get('reused') and not r.get('resumed_from'))}
        report['totals']['answer_cache_reused']=report.get('baseline_cache',{}).get('status')=='HIT'
        report['totals']['cached_baseline_seconds']=sum(r['metrics']['elapsed_s'] for r in report['rollouts'] if r.get('baseline_cache',{}).get('reused'))
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
    p.add_argument('--retry-candidate',type=Path,help='Reuse a saved tested candidate; skip diagnosis/implementation and rerun its evaluation')
    p.add_argument('--resume-run',type=Path,help='Continue a terminal run with saved candidate feedback; --patches is the total desired count')
    p.add_argument('--reuse-baseline-run',type=Path,help='Explicit historical READY baseline run to reuse across runtime updates')
    p.add_argument('--baseline-cache-dir',type=Path, help='Default: output parent/.patch_baseline_cache')
    p.add_argument('--no-baseline-cache',action='store_true',help='Measure a fresh baseline instead of reusing a completed one')
    p.add_argument('--backend',choices=['local','remote'],default='local');p.add_argument('--model')
    p.add_argument('--ssh-host',default='company-planner');p.add_argument('--patches',type=int,default=3)
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--max-calls',type=int,default=60)
    p.add_argument('--max-seconds',type=float,default=1800);p.add_argument('--call-timeout',type=float,default=180)
    p.add_argument('--replay-timeout',type=float,default=300);p.add_argument('--max-host-ops',type=int,default=12)
    p.add_argument('--max-consecutive-failures',type=int,default=3,help='Stop after this many consecutive candidate execution/validation failures; valid slow plans do not count')
    p.add_argument('--min-latency-reduction',type=float,default=.1)
    p.add_argument('--min-harnesses',type=int,
                   help='Consensus minimum: defaults to 2 for one patch, otherwise 3; explicit values are preserved')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--learning-mode',choices=['optimization','experience','code-extraction'],default='optimization')
    p.add_argument('--max-code-request-rounds',type=int,default=2,
                   help='Optional generated-function request rounds per original planner replay')
    p.add_argument('--reuse-code-extraction-run',type=Path,
                   help='Reuse saved code-extraction outputs after a Host fix; no new extraction model calls')
    p.add_argument('--reasoning-record',type=Path,help='Saved fresh reasoning plus reviewed draft experience for diagnosis only')
    p.add_argument('--semantic-timing',action='store_true',
                   help='Record public selection phase windows and Host timings (remote only; may affect latency)')
    a=p.parse_args(argv)
    if a.semantic_timing and a.backend != 'remote':
        p.error('--semantic-timing requires --backend remote')
    from contextlib import ExitStack
    from ..pipeline_timing import PipelineTiming
    try:
        if a.output.exists(): raise ValueError('Output exists; choose a new directory')
        with ExitStack() as timing_context:
            if a.semantic_timing:
                timing_context.enter_context(PipelineTiming(a.output.parent/(a.output.name+'_timing'), semantic_phases=True))
            evidence=evidence_from_manifest(read_json(a.manifest),a.decision_id) if a.manifest else load_evidence(a.evidence)
            model=RuntimeClaude(backend=a.backend,ssh_host=a.ssh_host,model=a.model,timeout_s=a.call_timeout)
            if a.learning_mode == 'code-extraction':
                from tempfile import TemporaryDirectory
                from .reasoning_code import run_extraction
                if not a.reuse_baseline_run or a.repeats != 1 or not 1 <= a.patches <= 2:
                    raise ValueError('code-extraction requires --reuse-baseline-run, --repeats 1 and --patches 1 or 2')
                if a.resume_run or a.retry_candidate or a.baseline_bundle or a.prepare_only:
                    raise ValueError('code-extraction starts a bounded library pilot; use --reasoning-record for prior feedback')
                if not 0 <= a.max_code_request_rounds <= 4:
                    raise ValueError('Code request rounds must be between zero and four')
                extractor=RuntimeClaude(backend=a.backend,ssh_host=a.ssh_host,model=a.model,
                                        timeout_s=a.call_timeout,text_only=True)
                with TemporaryDirectory(prefix='reasoning-code-evidence-') as temp:
                    root_hash=freeze_evidence(evidence,Path(temp)/'evidence',base=(a.manifest or a.evidence).parent)['evidence_hash']
                result=run_extraction(a.reuse_baseline_run,a.reasoning_record,a.output,extractor,model,
                    iterations=a.patches,call_timeout=a.call_timeout,replay_timeout=a.replay_timeout,
                    max_seconds=a.max_seconds,max_calls=a.max_calls,max_requests=a.max_code_request_rounds,
                    expected_root_hash=root_hash,reuse_extractions=a.reuse_code_extraction_run)
                return 0 if result['status']=='COMPLETED' else 2
            baseline=None
            if a.baseline_bundle:
                from .patch_lifecycle import verify_working
                baseline,_=verify_working(a.baseline_bundle.parent)
                if read_json(a.baseline_bundle)!=baseline: raise PolicyError('Baseline path is not the tested working bundle')
            result=evolve(evidence,a.output,model,baseline=baseline,
                source_base=(a.manifest or a.evidence).parent,patches=a.patches,repeats=a.repeats,max_calls=a.max_calls,
                max_seconds=a.max_seconds,call_timeout=a.call_timeout,replay_timeout=a.replay_timeout,max_ops=a.max_host_ops,
                min_latency_reduction=a.min_latency_reduction,prepare_only=a.prepare_only,
                reuse_baseline_run=a.reuse_baseline_run,retry_candidate=a.retry_candidate,
                max_consecutive_failures=a.max_consecutive_failures,
                resume_run=a.resume_run,
                learning_mode=a.learning_mode,reasoning_record=a.reasoning_record,
                baseline_cache_dir=None if a.no_baseline_cache else (a.baseline_cache_dir or a.output.parent/'.patch_baseline_cache'),
                consensus_options={'min_harnesses':a.min_harnesses} if a.min_harnesses is not None else None)
            return 0 if result['status'] in {'WORKING','PREPARED','EXPERIMENT_COMPLETE'} else 2
    except Exception as exc:
        if not a.output.exists():
            write_json(a.output/'blocked.json',{'status':'BLOCKED','error':f'{type(exc).__name__}: {exc}',
                       'model_calls':0,'robot_actions':0})
        print(f'{type(exc).__name__}: {exc}',flush=True);return 2


if __name__=='__main__': raise SystemExit(main())
