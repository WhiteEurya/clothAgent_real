"""Standalone inner learning loop: fixed evidence, runtime reflection, no robot.

python -m cloth_agent.harness.reasoning_learning --help
"""
from __future__ import annotations

import argparse
import copy
import difflib
import html
import json
import math
import statistics
import time
import traceback
from pathlib import Path

from ..planner_backend import claude_result_envelope
from ..token_usage import parse_usage
from .action_consensus import select_consensus
from .common import canonical, digest, now, read_json, write_json
from .model import RuntimeClaude
from .host_operations import CATALOG as HOST_CATALOG, execute as execute_host
from .optimization import OPTIMIZATION_OBJECTIVE, compare_versions, operation_candidates
from .policy import PolicyError, validate_schema
from .format_preflight import repair_format, preflight_contracts
from .reasoning_contract import (
    HARNESS_SCHEMA, JUDGMENT_SCHEMA, REFLECTION_SCHEMA, baseline_harness,
    evidence_from_manifest, freeze_evidence, freeze_harness, load_evidence, load_harness,
    validate_harness, validate_judgment, verify_evidence,
)

PLANNING_CONTRACT = '''Perform a fresh visual planning rollout using ONLY this fixed evidence package.
No tools, new images, image processing, filesystem access or physical actions. All supplied images remain
available in every stage. Follow this stage's instruction and only its explicitly bound prior-stage state.
Concept names and representation are yours to choose; provide concise evidence findings, not private
chain-of-thought. Never infer a cached conclusion from a previous rollout. READY is allowed only when
this stage permits it and both grasp and target are supported; otherwise CONTINUE or NEEDS_LEARNING.
The grasp must be a current Camera-A candidate. Target and anchor use pixel centers of the CURRENT
FULL CLEAN image, never crop/reference coordinates. Relation is toward, onto, across, away_from or hold
relative to the stated anchor. This is a visual target proposal, NOT XYZ, Z, IK, trajectory or robot command.
Host verifies identity, coordinates and budgets, not the semantic truth of the proposal.
Output contract: READY requires allow_ready=true, a complete action, and missing_information="".
CONTINUE and NEEDS_LEARNING require action=null. NEEDS_LEARNING also requires a specific blocking
gap in missing_information. Put non-blocking limitations in residual_uncertainty, which may be
nonempty even for READY. Lack of depth/IK is not automatically a blocker for this RGB-only visual
proposal; decide whether each uncertainty actually prevents the current visual choice. Never hide
a truly blocking gap as residual uncertainty. A later physical check is not a claim of physical success.
If later stages bind measurements, emit their named numeric values in optional measurements, using
the CURRENT FULL CLEAN pixel frame; emit null for an unknown measurement. host_results are deterministic
computations from earlier measurements, not new visual evidence or proof of on-fabric validity. Reuse
COMPUTED results rather than redoing their arithmetic. UNKNOWN results have no usable output: resolve
the missing evidence or report NEEDS_LEARNING, never substitute invented geometry.
Include a concise decision_log where evidence permits: the information need, actual observation or
operation, source image/result IDs, factual result, and any candidate kept/rejected/selected with a
brief evidence-based reason. Describe reported observations and decisions, not private thinking.
Do not invent candidate comparisons or tool use to fill the log. Direct inspection is a model image
assessment; it is not a Host crop or computation. Use depends_on only for explicit prior log records.
These are provenance records, not a prescribed workflow and not measured per-step timing.
'''
REFLECT_CONTRACT = '''You are the runtime harness meta-agent. Your primary objective is to REDUCE total
visual-planning wall-clock time while preserving decision quality and stability. Token usage and model
calls are secondary costs to report, not a token budget. Do not optimize for longer or more detailed reasoning.
Use the supplied stage timings, calls, tokens and public results to identify expensive information tasks.
Freely propose a concrete time-saving hypothesis: you may remove, combine, reorder or condition work,
reuse measured intermediate information, shorten output, or finish earlier. These are possibilities,
not a required recipe. Do not automatically add stages when answers disagree. An added stage needs
an explicit explanation of which more expensive work it replaces and why TOTAL planning may be faster.
Changing instructions cannot establish a speedup until measured; repeated agreement is not correctness.
Return the abstraction in operations: information_goal, method, outputs, success_check, on_insufficient,
replaces and expected_time_saving. Describe how to acquire information, not the answer to this scene.
Positive or negative findings can satisfy an information goal. If visibility is insufficient, report UNKNOWN
and the specific gap rather than claiming absence or repeating an observation that cannot resolve it.
Implement the proposed method in the returned harness so the next rollout actually tests it. operations
is an explanatory sidecar, not another model call. The host_operation_catalog lists implemented Python
computations. To use one, have an earlier model stage output named measurements; attach host_operations
to a later stage, with source_stage in its context and bindings mapping catalog argument names to those
measurement names. Host executes these before the later model call and supplies host_results. No numeric
answers from history may be bound. A host operation is optional: justify any additional model call by total
expected savings. Every proposal becomes an executable_patch.json including its actual host bindings.
If it needs unavailable host code or tools, do not pretend
they ran: explain the dependency and STOP, or propose an executable alternative within current capabilities.
Improve HOW the supplied planning program
uses fixed evidence, intermediate concepts, information ordering and stopping conditions. Do not optimize,
copy, prescribe or hard-code this observation's action. Do not redo the physical task. Return a reusable
replacement harness, or STOP with a reason. Propose a specific falsifiable change; its benefit is UNVERIFIED
until another fresh rollout. You may merge/split/reorder stages, modify instructions and context bindings,
or enable an earlier READY. All stages receive exactly the same fixed images, goal and registry. No new
images, external tools, arbitrary code, image transformations or cross-rollout cache in this version.
Only the explicitly supplied host_operation_catalog is executable. Intermediate concepts
may be newly named but are recomputed each rollout. Missing telemetry is UNKNOWN, not zero. Public
summaries cannot prove which private reasoning was repeated or useless. Separate observed costs from
hypotheses and unverified necessity. Never claim ablation evidence from an untested patch.
Parameterize the reusable harness wherever practical: bind grasp candidates, targets, anchors, image
regions and geometric measurements from the CURRENT observation, goal and registry on each rollout.
Describe how to derive those values rather than copying a historical action, coordinate or image ID.
Numbers, numbered steps, counts, thresholds and ordinary technical examples are allowed. Explain the
applicability of any fixed heuristic; prefer scene-relative quantities where they support transfer.
An example ID, path or coordinate in explanatory prose is not a binding to a current answer or permission
to access external evidence. The fixed-input and no-tool execution scope remains unchanged.
Put observations, rationale and source rollout citations ONLY in changes, outside the executable harness.
Treat contract/schema rejection separately from visual failure: a READY_BLOCKING_GAP or
NON_READY_ACTION error identifies conflicting output fields, not a need for more geometry stages.
Do not infer grasp quality or missing intermediate-concept requirements from a serialization rejection.
Each stage has a fixed judgment schema provided below. context can reference only earlier stage IDs.
A non-READY last stage falls back to NEEDS_LEARNING. Output is a bounded JSON program, not Python.
'''
LIMITS = [
    'Fixed evidence on one observation only; no generalization or physical-success claim.',
    'Consensus is decision stability among correlated harnesses, not independent proof of correctness.',
    'Visual target pixels are not grounded robot targets; no physical action or experience update occurs.',
    'Planning time includes fixed-input checks, image delivery, Claude, output validation and debug writes; excludes reflection.',
    'Total inner-loop cost includes every rollout and reflection; no speedup claim from the cheapest rollout alone.',
    'Provider prompt caches and fixed execution order may affect latency; no stored answer cache is used.',
    'Thinking tokens are unavailable unless explicitly reported; output tokens are not a substitute.',
    'Executable prose is structurally screened but its semantic reusability is not certified.',
]


class BudgetExceeded(RuntimeError):
    pass


class DebugLog:
    def __init__(self, output):
        self.output, self.started = Path(output), time.monotonic()

    def event(self, kind, **details):
        row = {'at': now(), 'elapsed_s': time.monotonic() - self.started, 'event': kind, **details}
        with (self.output / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(canonical(row) + '\n')
        print('[reasoning-learning] ' + canonical(row), flush=True)


class CallBudget:
    def __init__(self, *, max_calls, max_seconds, call_timeout, prompt_chars, debug, max_tokens=None):
        self.max_calls, self.max_seconds = max_calls, max_seconds
        self.call_timeout, self.prompt_chars = call_timeout, prompt_chars
        self.started, self.attempts, self.debug = time.monotonic(), 0, debug
        self.max_tokens, self.tokens_used, self.unknown_usage_calls = max_tokens, 0, 0

    def token_report(self):
        return {'limit': self.max_tokens, 'known_total_tokens': self.tokens_used,
                'unknown_usage_calls': self.unknown_usage_calls,
                'remaining_tokens': None if self.max_tokens is None or self.unknown_usage_calls else max(0, self.max_tokens-self.tokens_used),
                'stop_reason': (None if self.max_tokens is None else
                                'TOKEN_USAGE_UNKNOWN' if self.unknown_usage_calls else
                                'TOKEN_BUDGET_EXHAUSTED' if self.max_tokens is not None and self.tokens_used >= self.max_tokens else None),
                'scope': 'input + output + cache read + cache creation; thinking is part of output, not added again',
                'enforcement': 'between calls; an in-flight call can exceed the remaining budget'}

    def invoke(self, model, *, frozen, evidence_dir, prompt, schema, output, stage, deadline=None):
        if self.max_tokens is not None and self.token_report()['stop_reason']:
            raise BudgetExceeded(self.token_report()['stop_reason'])
        remaining = self.max_seconds - (time.monotonic() - self.started)
        if deadline is not None:
            remaining = min(remaining, deadline - time.monotonic())
        if self.attempts >= self.max_calls or remaining <= 0:
            raise BudgetExceeded('SESSION_OR_ROLLOUT_BUDGET_EXHAUSTED')
        if len(prompt) > self.prompt_chars:
            raise BudgetExceeded('PROMPT_BUDGET_EXHAUSTED; no silent truncation')
        images = verify_evidence(frozen, evidence_dir)
        if getattr(model, 'text_only', False):
            images = []
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Saved even when model setup fails before RuntimeClaude creates its call folder.
        write_json(output.parent / (output.name + '_input.json'), {'stage': stage, 'prompt': prompt,
            'schema': schema, 'evidence_hash': frozen['evidence_hash'],
            'image_ids': [i['image_id'] for i in frozen['evidence']['images']] if images else []})
        self.attempts += 1
        self.debug.event('call_start', stage=stage, call_attempt=self.attempts,
                         evidence_hash=frozen['evidence_hash'], artifact=str(output), timeout_s=min(remaining, self.call_timeout))
        started = time.monotonic()
        call_start = len(model.calls)
        try:
            result = model.invoke(prompt=prompt, schema=schema, images=images, output=output,
                                  stage=stage, timeout_s=min(remaining, self.call_timeout))
            write_json(output / 'returned.json', result)
            verify_evidence(frozen, evidence_dir)
            if time.monotonic() - started > remaining:
                raise BudgetExceeded('MODEL_RESPONSE_AFTER_DEADLINE')
            self.debug.event('call_return', stage=stage, elapsed_s=time.monotonic()-started)
            return result
        except Exception as exc:
            write_json(output.parent / (output.name + '_failure.json'),
                       {'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()})
            self.debug.event('call_failed', stage=stage, error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            invoked = sum(bool(c.get('backend_invoked')) for c in model.calls[call_start:])
            # Count only RuntimeClaude's canonical terminal file, not duplicate
            # transport traces. Failed/invalid replies still consume tokens.
            path = output / 'stdout.jsonl'
            usage = parse_usage(path.read_text()) if path.is_file() else {}
            total = usage.get('total_tokens')
            if invoked and total is not None:
                self.tokens_used += total
            if invoked and (total is None or invoked != 1):
                self.unknown_usage_calls += invoked if total is None else invoked-1
            self.debug.event('token_usage', stage=stage, call_tokens=total,
                             **self.token_report())
            write_json(self.debug.output / 'token_budget.json', self.token_report())


def call_metrics(calls, directory, *, exclude_dirs=()):
    """Use terminal usage only; hidden/provider reasoning is never reconstructed."""
    usages, thinking = [], []
    excluded = [(Path(directory)/name).resolve() for name in exclude_dirs]
    for path in sorted(Path(directory).rglob('stdout.jsonl')):
        if any(path.resolve().is_relative_to(root) for root in excluded):
            continue
        stdout = path.read_text()
        usages.append(parse_usage(stdout))
        try:
            usage = claude_result_envelope(stdout).get('usage') or {}
            details = usage.get('output_tokens_details') or {}
            value = usage.get('thinking_tokens')
            if value is None:
                value = details.get('thinking_tokens')
            if value is None:
                value = details.get('reasoning_tokens')
            thinking.append(value if type(value) is int and value >= 0 else None)
        except (RuntimeError, ValueError, TypeError):
            thinking.append(None)
    invocations = sum(c.get('backend_invoked', False) for c in calls)
    complete = bool(invocations) and len(usages) == invocations
    sums = {key: sum(u[key] for u in usages) if complete and all(u.get(key) is not None for u in usages) else None
            for key in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens', 'total_tokens', 'total_cost_usd')}
    return {'model_calls': invocations, 'response_count': sum(c.get('response_received', False) for c in calls),
            'tool_calls': sum(c['tool_round_trips'] for c in calls) if all(c.get('tool_round_trips') is not None for c in calls) else None,
            'thinking_tokens': sum(thinking) if complete and all(v is not None for v in thinking) else None,
            'usage': sums, 'host_image_ops': 0, 'usage_source': 'terminal Claude usage; no estimates',
            'configuration': [c.get('configuration') for c in calls], 'answer_cache_reused': False}


def execution_signature(harness):
    positions = {s['id']: i for i, s in enumerate(harness['stages'])}
    return digest({'applicability': harness['applicability'], 'stages': [
        {'instruction': s['instruction'], 'context': [positions[k] for k in s['context']], 'allow_ready': s['allow_ready'],
         'host_operations': [{**op, 'source_stage': positions[op['source_stage']]}
                             for op in s.get('host_operations', [])]}
        for s in harness['stages']]})


def run_rollout(version, frozen, evidence_dir, model, output, budget, *, rollout_id, timeout):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start, call_start = time.monotonic(), len(model.calls)
    harness = copy.deepcopy(version['harness'])
    row = {'rollout_id': rollout_id, 'harness_hash': version['harness_hash'], 'evidence_hash': frozen['evidence_hash'],
           'status': 'NEEDS_LEARNING', 'reason': 'STAGES_EXHAUSTED', 'action': None, 'stages': [],
           'actual_measurement': model.configuration.get('actual_measurement', True)}
    write_json(output / 'harness.json', version, exclusive=True)
    budget.debug.event('rollout_start', rollout_id=rollout_id, harness_hash=version['harness_hash'])
    try:
        validate_harness(harness)
        if digest(harness) != version['harness_hash']:
            raise PolicyError('Frozen harness changed')
        state = {}
        row['host_operations'] = []
        for stage in harness['stages']:
            stage_start, stage_calls = time.monotonic(), len(model.calls)
            host_results = execute_host(stage, state, frozen['evidence'])
            row['host_operations'].extend(host_results)
            write_json(output / 'host_execution.json', row['host_operations'])
            payload = {'fixed_evidence': frozen['evidence'], 'applicability': harness['applicability'],
                       'stage': stage, 'bound_state': {name: state[name] for name in stage['context']},
                       'host_results': host_results}
            result = budget.invoke(model, frozen=frozen, evidence_dir=evidence_dir,
                prompt=PLANNING_CONTRACT + canonical(payload), schema=JUDGMENT_SCHEMA,
                output=output / 'calls' / stage['id'], stage='reasoning_rollout', deadline=start+timeout)
            result = repair_format(result, 'judgment', output / 'calls' / stage['id'] / 'format_repair.json')
            stage_row = {'stage_id': stage['id'], 'judgment': result, 'valid': False,
                         'elapsed_s': time.monotonic()-stage_start,
                         'metrics': call_metrics(model.calls[stage_calls:], output / 'calls' / stage['id'])}
            row['stages'].append(stage_row)
            try:
                action = validate_judgment(result, frozen['evidence'], allow_ready=stage['allow_ready'])
            except PolicyError as exc:
                stage_row['validation_error'] = str(exc)
                raise
            stage_row['valid'] = True
            state[stage['id']] = result
            write_json(output / 'state.json', state)
            budget.debug.event('stage_validated', rollout_id=rollout_id, stage_id=stage['id'], status=result['status'])
            if result['status'] == 'NEEDS_LEARNING':
                row['reason'] = result['missing_information']
                break
            if action:
                row.update(status='READY', reason=result['evidence_summary'], action=action)
                break
    except BudgetExceeded as exc:
        row.update(reason=str(exc), status='BUDGET_EXHAUSTED')
    except Exception as exc:
        row.update(status='ERROR', reason=f'{type(exc).__name__}: {exc}')
        row['failure_kind'] = 'CONTRACT_REJECTION' if isinstance(exc, PolicyError) else 'CALL_OR_RUNTIME_ERROR'
        (output / 'exception.txt').write_text(traceback.format_exc())
    finally:
        if digest(version['harness']) != version['harness_hash']:
            row.update(status='ERROR', reason='HARNESS_MUTATED_DURING_ROLLOUT', action=None)
        calls = copy.deepcopy(model.calls[call_start:])
        row['metrics'] = call_metrics(calls, output / 'calls')
        row['metrics']['elapsed_s'] = time.monotonic() - start
        row['metrics']['host_operation_count'] = len(row.get('host_operations', []))
        row['metrics']['host_operation_seconds'] = sum(op['elapsed_s'] for op in row.get('host_operations', []))
        row['metrics']['call_attempts'] = len(list((output / 'calls').glob('*_input.json')))
        try:
            from ..public_process import collect_rollout_trace, write_trace
            row['operation_trace'] = collect_rollout_trace(row, output)
            write_trace(row['operation_trace'], output/'trace')
        except Exception as exc:
            row['trace_error'] = f'{type(exc).__name__}: {exc}'
        write_json(output / 'call_audits.json', calls)
        write_json(output / 'result.json', row)
        budget.debug.event('rollout_end', rollout_id=rollout_id, status=row['status'], metrics=row['metrics'])
    return row


def reflect(parent, parent_rollouts, frozen, evidence_dir, model, output, budget, previous_changes):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start, call_start = time.monotonic(), len(model.calls)
    report = {'status': 'ERROR', 'parent_hash': parent['harness_hash']}
    try:
        if parent_rollouts and all(r.get('failure_kind') == 'CONTRACT_REJECTION' for r in parent_rollouts):
            report.update(status='STOP', reason='FORMAT_BLOCKED: all parent rollouts failed contract validation; no visual-learning evidence. Inspect format_repair and validation_error before another experiment.')
            return report
        # A rejected serialization is not evidence for changing visual reasoning.
        observed = copy.deepcopy(parent_rollouts)
        for row in observed:
            if row['status'] == 'ERROR':
                row['stages'] = []
                row['learning_instruction'] = 'Infrastructure/contract rejection only. Do not infer visual failure or add reasoning stages to explain it.'
        payload = {'parent_harness': parent['harness'], 'observed_rollouts': observed,
                   'optimization_objective': OPTIMIZATION_OBJECTIVE,
                   'host_operation_catalog': HOST_CATALOG,
                   'already_proposed_changes': previous_changes,
                   'fixed_evidence': frozen['evidence'], 'harness_schema': HARNESS_SCHEMA,
                   'judgment_schema': JUDGMENT_SCHEMA}
        proposal = budget.invoke(model, frozen=frozen, evidence_dir=evidence_dir,
            prompt=REFLECT_CONTRACT + canonical(payload), schema=REFLECTION_SCHEMA,
            output=output / 'call', stage='reasoning_reflection')
        write_json(output / 'proposal.json', proposal)
        proposal = repair_format(proposal, 'reflection', output / 'format_repair.json')
        validate_schema(proposal, REFLECTION_SCHEMA)
        if proposal['status'] == 'STOP':
            if proposal['harness'] is not None:
                raise PolicyError('STOP must not supply an executable harness')
            report.update(status='STOP', reason=proposal['reason'])
        else:
            validate_harness(proposal['harness'])
            known = {r['rollout_id'] for r in parent_rollouts}
            if not proposal['changes'] or any(not set(c['source_rollout_ids']) <= known for c in proposal['changes']):
                raise PolicyError('Patch must cite actual parent rollout evidence')
            if execution_signature(proposal['harness']) == execution_signature(parent['harness']):
                raise PolicyError('Reflection did not change the executable harness')
            report.update(status='VALID', proposal=proposal)
            old = json.dumps(parent['harness'], ensure_ascii=False, indent=2).splitlines()
            new = json.dumps(proposal['harness'], ensure_ascii=False, indent=2).splitlines()
            (output / 'harness.diff').write_text('\n'.join(difflib.unified_diff(old, new, fromfile='parent', tofile='proposed')) + '\n')
    except BudgetExceeded as exc:
        report.update(status='BUDGET_EXHAUSTED', reason=str(exc))
    except Exception as exc:
        report.update(reason=f'{type(exc).__name__}: {exc}')
        (output / 'exception.txt').write_text(traceback.format_exc())
    finally:
        calls = copy.deepcopy(model.calls[call_start:])
        report['metrics'] = {**call_metrics(calls, output), 'elapsed_s': time.monotonic()-start}
        write_json(output / 'call_audits.json', calls)
        write_json(output / 'validation.json', report)
        budget.debug.event('reflection_end', status=report['status'], parent_hash=parent['harness_hash'],
                           reason=report.get('reason'), metrics=report['metrics'])
    return report


def write_report(output, report):
    write_json(output / 'report.json', report)
    lines = ['# Fixed-evidence reasoning learning', '', f"Status: {report['status']}",
             f"Actual model measurements: {report['actual_measurement']}",
             f"Evidence hash: {report.get('evidence_hash', 'unavailable')}",
             f"Search: {report['settings']['search']}; robot actions: 0", '',
             '| Rollout | Harness | Status | Grasp | Target pixels / relation | Seconds | Model calls | Tool calls | Thinking tokens |',
             '|---|---|---|---|---|---:|---:|---:|---:|']
    for r in report['rollouts']:
        action, metrics = r.get('action') or {}, r['metrics']
        target = action.get('target', {})
        lines.append(f"| {r['rollout_id']} | {r['harness_hash'][:12]} | {r['status']} | {action.get('selected_reference', {}).get('reference_id')} | {target.get('pixel_xy')} / {target.get('relation')} | {metrics['elapsed_s']:.3f} | {metrics['model_calls']} | {metrics['tool_calls']} | {metrics['thinking_tokens']} |")
    lines += ['', '## Time-first optimization comparison', '',
              '| Version | Median seconds | Median calls | Median tokens | Stability | Baseline/candidate time |',
              '|---|---:|---:|---:|---|---:|']
    for comparison in report.get('optimization_comparison', {}).get('versions', []):
        lines.append('| ' + ' | '.join(str(comparison[k]) for k in (
            'version', 'median_seconds', 'median_model_calls', 'median_total_tokens',
            'within_version_stability', 'baseline_over_candidate_time_ratio')) + ' |')
    lines += ['', 'Quality is NOT_EVALUATED. Faster timing alone does not approve a skill.',
              '', '## Baseline comparison', '', '```json', json.dumps(report.get('baseline_comparison'), indent=2), '```',
              '', '## Consensus and costs', '', '```json', json.dumps(report.get('consensus'), ensure_ascii=False, indent=2), '```',
              '', '## Total work', '', '```json', json.dumps(report.get('totals'), indent=2), '```',
              '', '## Limitations', '', *['- ' + s for s in LIMITS]]
    if report.get('error'):
        lines += ['', 'Error: ' + report['error']]
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    content = '<h1>固定证据：脑内 Harness Learning</h1><p>仅离线视觉决策稳定性；机器人动作数为零。</p>'
    content += '<nav><a href="report.json">JSON</a> · <a href="report.md">Markdown</a> · <a href="events.jsonl">事件日志</a> · <a href="optimization_comparison.json">优化对比</a> · <a href="operation_candidates.json">抽象操作候选</a></nav>'
    content += '<pre>' + html.escape('\n'.join(lines[:9])) + '</pre>'
    for r in report['rollouts']:
        name = r['rollout_id']
        content += f'<details><summary>{html.escape(name)} — {html.escape(r["status"])}</summary>'
        content += f'<a href="rollouts/{name}/result.json">结果与公开阶段输出</a> · <a href="rollouts/{name}/harness.json">本轮 Harness</a>'
        content += '<pre>' + html.escape(json.dumps(r, ensure_ascii=False, indent=2)) + '</pre></details>'
    for index, reflection in enumerate(report['reflections'], 1):
        content += f'<details><summary>Reflection {index}: {html.escape(reflection["status"])}</summary>'
        content += f'<a href="reflections/patch_{index:02d}/validation.json">校验、成本与提案</a> · <a href="reflections/patch_{index:02d}/harness.diff">Harness diff</a>'
        content += '<pre>' + html.escape(json.dumps(reflection, ensure_ascii=False, indent=2)) + '</pre></details>'
    content += '<h2>Consensus</h2><pre>' + html.escape(json.dumps(report.get('consensus'), ensure_ascii=False, indent=2)) + '</pre>'
    content += '<h2>优化对比（正确性尚未验证）</h2><pre>' + html.escape(json.dumps(report.get('optimization_comparison'), ensure_ascii=False, indent=2)) + '</pre>'
    content += '<h2>Totals / limits</h2><pre>' + html.escape(json.dumps(report.get('totals'), indent=2) + '\n' + '\n'.join(LIMITS)) + '</pre>'
    (output / 'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Reasoning learning debug</title><style>body{max-width:1200px;margin:auto;padding:24px;font:16px/1.6 system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f7;padding:16px}details{border:1px solid #ddd;margin:12px 0;padding:12px}</style>' + content + '</html>')


def run_learning(evidence, output, model, *, initial=None, source_base=None, search='serial', variants=3,
                 repeats=1, max_calls=40, max_seconds=1800, rollout_timeout=300, call_timeout=180,
                 max_prompt_chars=160000, consensus_options=None, prepare_only=False, max_tokens=None):
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
        raise ValueError('max_tokens must be a positive integer')
    if search not in {'serial', 'branch'} or not 0 <= variants <= 8 or not 1 <= repeats <= 5:
        raise ValueError('Invalid bounded search settings')
    if not 1 <= max_calls <= 200 or any(not math.isfinite(v) or v <= 0 for v in (max_seconds, rollout_timeout, call_timeout)):
        raise ValueError('Invalid call or time budget')
    if not 1000 <= max_prompt_chars <= 500000:
        raise ValueError('Invalid prompt budget')
    # Validate comparison options before spending on Claude.
    options = consensus_options or {}
    select_consensus([], [], repeats=repeats, **options)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    debug = DebugLog(output)
    start = time.monotonic()
    budget = CallBudget(max_calls=max_calls, max_seconds=max_seconds, call_timeout=call_timeout,
                        prompt_chars=max_prompt_chars, debug=debug, max_tokens=max_tokens)
    settings = dict(search=search, variants=variants, repeats=repeats, max_calls=max_calls,
                    max_seconds=max_seconds, rollout_timeout=rollout_timeout, call_timeout=call_timeout,
                    max_prompt_chars=max_prompt_chars, max_tokens=max_tokens, consensus=options, prepare_only=prepare_only)
    report = {'schema_version': 1, 'status': 'PREPARING', 'settings': settings, 'rollouts': [], 'reflections': [],
              'versions': [], 'actual_measurement': model.configuration.get('actual_measurement', True),
              'model_configuration': model.configuration, 'robot_actions': 0, 'limitations': LIMITS}
    write_report(output, report)
    try:
        write_json(output / 'format_preflight.json', preflight_contracts())
        frozen = freeze_evidence(evidence, output / 'evidence', base=source_base)
        report['evidence_hash'] = frozen['evidence_hash']
        first = freeze_harness(initial or baseline_harness(), output / 'harnesses/h_zero.json',
                               source='user_supplied_baseline' if initial else 'developer_baseline_not_learned')
        versions, signatures = [first], {execution_signature(first['harness'])}
        report['versions'].append(first)
        debug.event('evidence_frozen', evidence_hash=frozen['evidence_hash'], harness_hash=first['harness_hash'])
        if prepare_only:
            report['status'] = 'PREPARED'
        else:
            def evaluate(version, index):
                for repetition in range(repeats):
                    rid = f'h{index:02d}_r{repetition:02d}'
                    row = run_rollout(version, frozen, output / 'evidence', model, output / 'rollouts' / rid,
                                      budget, rollout_id=rid, timeout=rollout_timeout)
                    report['rollouts'].append(row)
                    write_report(output, report)
                    if row['status'] == 'BUDGET_EXHAUSTED':
                        break
            report['status'] = 'RUNNING'
            evaluate(first, 0)
            changes = []
            for index in range(1, variants+1):
                parent = versions[-1] if search == 'serial' else first
                observations = [r for r in report['rollouts'] if r['harness_hash'] == parent['harness_hash']]
                reflection = reflect(parent, observations, frozen, output / 'evidence', model,
                                     output / 'reflections' / f'patch_{index:02d}', budget, changes)
                report['reflections'].append(reflection)
                write_report(output, report)
                if reflection['status'] in {'STOP', 'BUDGET_EXHAUSTED'}:
                    break
                if reflection['status'] != 'VALID':
                    continue
                proposed = reflection['proposal']['harness']
                signature = execution_signature(proposed)
                if signature in signatures:
                    reflection.update(status='DUPLICATE', reason='Executable behavior already evaluated; no extra vote')
                    write_json(output / 'reflections' / f'patch_{index:02d}' / 'validation.json', reflection)
                    debug.event('duplicate_patch', patch=index)
                    continue
                version = freeze_harness(proposed, output / 'harnesses' / f'h_{index:02d}.json',
                                         parent_hash=parent['harness_hash'])
                write_json(output / 'reflections' / f'patch_{index:02d}' / 'executable_patch.json', {
                    'status': 'IMPLEMENTED_PENDING_EVALUATION', 'parent_hash': parent['harness_hash'],
                    'harness_hash': version['harness_hash'], 'harness': proposed,
                    'implementation': 'Host interpreter executes stage bindings and model calls; no arbitrary source execution.',
                    'host_operation_count': sum(len(s.get('host_operations', [])) for s in proposed['stages'])})
                signatures.add(signature)
                versions.append(version)
                report['versions'].append(version)
                changes.append(reflection['proposal']['changes'])
                evaluate(version, index)
                patch_path = output / 'reflections' / f'patch_{index:02d}' / 'executable_patch.json'
                patch = read_json(patch_path)
                evaluated = [r for r in report['rollouts'] if r['harness_hash'] == version['harness_hash']]
                patch.update(status='EVALUATED', evaluation=[{
                    'rollout_id': r['rollout_id'], 'status': r['status'],
                    'host_operation_count': r['metrics']['host_operation_count'],
                    'elapsed_s': r['metrics']['elapsed_s']} for r in evaluated],
                    benefit='UNVERIFIED; inspect optimization_comparison and review visual quality')
                write_json(patch_path, patch)
            report['consensus'] = select_consensus(report['rollouts'], [v['harness_hash'] for v in versions], repeats=repeats, **options)
            selected = report['consensus']['selected']
            report['status'] = 'SELECTED' if selected else 'NO_SELECTION'
            if selected:
                chosen = next(v for v in versions if v['harness_hash'] == selected['harness_hash'])
                write_json(output / 'selected_harness.json', chosen, exclusive=True)
                write_json(output / 'selected_action.json', {**selected, 'evidence_hash': frozen['evidence_hash'],
                           'actual_measurement': report['actual_measurement'], 'physical_success': 'UNKNOWN'}, exclusive=True)
                baseline_rows = [r for r in report['rollouts'] if r['harness_hash'] == first['harness_hash']]
                selected_rows = [r for r in report['rollouts'] if r['harness_hash'] == chosen['harness_hash']]
                baseline_ready = len(baseline_rows) == repeats and all(r['status'] == 'READY' for r in baseline_rows)
                baseline_s = statistics.median(r['metrics']['elapsed_s'] for r in baseline_rows) if baseline_ready else None
                selected_s = statistics.median(r['metrics']['elapsed_s'] for r in selected_rows)
                report['baseline_comparison'] = {'baseline_median_seconds': baseline_s, 'selected_median_seconds': selected_s,
                    'measured_planning_ratio': baseline_s / selected_s
                        if report['actual_measurement'] and baseline_s is not None and selected_s > 0 else None,
                    'note': 'Planning-stage ratio only; not total inner-loop speedup. Synthetic runs never report a measured ratio.'}
    except Exception as exc:
        report.update(status='ERROR', error=f'{type(exc).__name__}: {exc}')
        (output / 'exception.txt').write_text(traceback.format_exc())
        debug.event('session_failed', error=report['error'])
    finally:
        report['optimization_objective'] = OPTIMIZATION_OBJECTIVE
        report['optimization_comparison'] = compare_versions(report['versions'], report['rollouts'],
            repeats=repeats, options=options, actual=report['actual_measurement'])
        write_json(output / 'optimization_comparison.json', report['optimization_comparison'])
        write_json(output / 'operation_candidates.json', operation_candidates(report))
        rows = report['rollouts'] + report['reflections']
        report['totals'] = {'elapsed_s': time.monotonic()-start, 'call_attempts': budget.attempts,
            'token_budget': budget.token_report(),
            'model_calls': sum(r['metrics']['model_calls'] for r in rows),
            'rollout_seconds': sum(r['metrics']['elapsed_s'] for r in report['rollouts']),
            'reflection_seconds': sum(r['metrics']['elapsed_s'] for r in report['reflections']),
            'note': 'Optimization overhead is paid in this experiment, not amortized or hidden.'}
        for key in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens', 'total_tokens', 'total_cost_usd'):
            report['totals'][key] = (sum(r['metrics']['usage'][key] for r in rows)
                if rows and all(r['metrics']['usage'][key] is not None for r in rows) else None)
        debug.event('session_end', status=report['status'], totals=report['totals'])
        write_report(output, report)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest', type=Path, help='Collector manifest; only pre_decision is read')
    source.add_argument('--evidence', type=Path, help='Explicit pre-decision evidence JSON; paths relative to its directory')
    parser.add_argument('--decision-id')
    parser.add_argument('--initial-harness', type=Path, help='Frozen harness from a prior experiment; recomputes all state')
    parser.add_argument('--output', required=True, type=Path, help='New directory; never overwritten')
    parser.add_argument('--search', choices=['serial', 'branch'], default='serial')
    parser.add_argument('--variants', '--k', dest='variants', type=int, default=3,
                        help='maximum new-harness proposal attempts, excluding initial H0 (default: 3)')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-calls', type=int, default=40)
    parser.add_argument('--max-tokens', type=int,
                        help='total input/output/cache token budget across planning and reflection; checked between calls, not a hard in-flight cap')
    parser.add_argument('--max-seconds', type=float, default=1800)
    parser.add_argument('--rollout-timeout', type=float, default=300)
    parser.add_argument('--call-timeout', type=float, default=180)
    parser.add_argument('--max-prompt-chars', type=int, default=160000)
    parser.add_argument('--backend', choices=['local', 'remote'], default='local')
    parser.add_argument('--model')
    parser.add_argument('--ssh-host', default='company-planner')
    parser.add_argument('--claude-binary', default='claude')
    parser.add_argument('--grasp-mode', choices=['candidate', 'distance'], default='candidate')
    parser.add_argument('--epsilon-grasp', type=float, default=12)
    parser.add_argument('--epsilon-target', type=float, default=20)
    parser.add_argument('--epsilon-anchor', type=float, default=20)
    parser.add_argument('--min-support', type=float, default=.6)
    parser.add_argument('--min-harnesses', type=int, default=3)
    parser.add_argument('--cost-weights', type=float, nargs=4, default=[1, 0, 0, 0], metavar=('TIME', 'CALLS', 'TOOLS', 'THINKING'))
    parser.add_argument('--prepare-only', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError('Output already exists; use a new experiment directory')
        evidence = (evidence_from_manifest(read_json(args.manifest), args.decision_id) if args.manifest else load_evidence(args.evidence))
        initial = load_harness(args.initial_harness) if args.initial_harness else None
        model = RuntimeClaude(backend=args.backend, binary=args.claude_binary, ssh_host=args.ssh_host,
                              model=args.model, timeout_s=args.call_timeout)
        report = run_learning(evidence, args.output, model, initial=initial,
            source_base=args.evidence.parent if args.evidence else args.manifest.parent,
            search=args.search, variants=args.variants, repeats=args.repeats, max_calls=args.max_calls,
            max_seconds=args.max_seconds, rollout_timeout=args.rollout_timeout, call_timeout=args.call_timeout,
            max_tokens=args.max_tokens,
            max_prompt_chars=args.max_prompt_chars, prepare_only=args.prepare_only,
            consensus_options={'grasp_mode': args.grasp_mode, 'epsilon_grasp': args.epsilon_grasp,
                'epsilon_target': args.epsilon_target, 'epsilon_anchor': args.epsilon_anchor,
                'min_support': args.min_support, 'min_harnesses': args.min_harnesses,
                'weights': dict(zip(('elapsed_s', 'model_calls', 'tool_calls', 'thinking_tokens'), args.cost_weights))})
        return 0 if report['status'] in {'SELECTED', 'PREPARED'} else 2
    except Exception as exc:
        # Missing source conditions still leave a reviewable debug artifact.
        if not args.output.exists():
            args.output.mkdir(parents=True)
            write_json(args.output / 'blocked.json', {'status': 'BLOCKED', 'error': f'{type(exc).__name__}: {exc}',
                        'model_calls': 0, 'robot_actions': 0})
        print(f'{type(exc).__name__}: {exc}', flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
