"""Snapshot an existing baseline, replay its complete reasoning, then extract draft methods."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from .baseline_cache import tree_hashes
from .common import canonical, digest, read_json, write_json
from .experience_review import schema as experience_schema
from .model import RuntimeClaude
from .policy import validate_schema
from .reasoning_contract import verify_evidence
from .reasoning_learning import CallBudget, DebugLog, call_metrics, run_rollout

REFLECT = '''Summarize reusable REASONING methods from this fresh visual grasp/target planning record.
This is not an image-processing-only task and not latency optimization. Cover task interpretation,
structure identification, evidence requirements, candidate comparison, grasp selection, target relation,
coordinate handling, uncertainty and stopping where supported by the record. Do not force missing steps.
The record contains explicit findings and final decisions, NOT the private thinking process: never
invent thought order or claim to reconstruct hidden reasoning. You have text only, not images.
For each method state the problem, needed information, how to obtain it, decision rule, success_check,
on_insufficient, and what deterministic calculation could be delegated to code. Code suggestions
are UNIMPLEMENTED unless the record proves execution. Parameterize methods; do not prescribe this
scene's candidate IDs or coordinates as reusable answers. Keep exact historical values in evidence
discussion only if needed. supported_by/failed_in concern a method's documented information or
consistency result, never robot success. Unverified semantic claims belong in unresolved. A validated
READY means a legal visual proposal, not correctness. A suggestion is not demonstrated reusable
learning until another run tests it. No promotion, benchmark gate or automatic skill activation.
Use concise Chinese, at most six lessons. Include contradictions and missing evidence explicitly.
'''


def summary_schema(ids):
    spec = experience_schema(ids)
    lesson = spec['properties']['lessons']['items']
    lesson['properties']['scope']['enum'] = ['reasoning_decision', 'visual_information', 'workflow']
    lesson['properties']['decision_rule'] = {'type': 'string'}
    lesson['properties']['code_candidate'] = {'type': 'string'}
    lesson['required'] += ['decision_rule', 'code_candidate']
    spec['properties']['lessons']['maxItems'] = 6
    return spec


def run(baseline, output, vision, reflection, *, timeout=900):
    baseline, output = Path(baseline).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'status': 'RUNNING', 'baseline_source': str(baseline), 'robot_actions': 0,
              'automatic_activation': False, 'motion_code_generated': False,
              'scope': 'Complete saved baseline visual reasoning: grasp and target; then draft method extraction.'}
    debug = DebugLog(output)
    try:
        before = tree_hashes(baseline)
        saved = output/'baseline'
        shutil.copytree(baseline, saved)
        if tree_hashes(saved) != before or tree_hashes(baseline) != before:
            raise ValueError('Baseline changed while snapshotting')
        write_json(output/'baseline_manifest.json', {'source': str(baseline), 'files': before,
                   'snapshot_hash': digest(before), 'historical_result_is_not_fresh_model_input': True})
        frozen = read_json(saved/'prepared/evidence.json')
        paths = verify_evidence(frozen, saved/'prepared')
        version = read_json(saved/'reasoning_version.json')
        report.update(evidence_hash=frozen['evidence_hash'], images=len(paths),
                      harness_hash=version['harness_hash'])
        write_json(output/'report.json', report)
        budget = CallBudget(max_calls=len(version['harness']['stages']), max_seconds=timeout,
                            call_timeout=timeout, prompt_chars=240000, debug=debug)
        # Only the pre-decision image package and harness enter planning, never result.json.
        result = run_rollout(version, frozen, saved/'prepared', vision, output/'reasoning',
                             budget, rollout_id='fresh_reasoning', timeout=timeout)
        report['planning_status'] = result['status']
        report['action'] = result.get('action')
        report['planning_metrics'] = result['metrics']
        write_json(output/'report.json', report)
        stages = result['stages']
        records = [{'record_id': 'fresh_'+s['stage_id'], 'stage': s,
                    'instruction': next(x['instruction'] for x in version['harness']['stages']
                                        if x['id']==s['stage_id'])} for s in stages]
        if not records:
            report.update(status='PLANNING_FAILED', reason=result['reason'])
            return report
        context = {'goal': frozen['evidence']['fold_goal'], 'image_catalog': frozen['evidence']['images'],
                   'records': records, 'host_operations': result.get('host_operations', []),
                   'final_status': result['status'], 'action': result.get('action'),
                   'physical_validation': 'NOT_PERFORMED'}
        spec = summary_schema([r['record_id'] for r in records])
        prompt = REFLECT + canonical(context)
        write_json(output/'reflection_input.json', {'prompt': prompt, 'schema': spec, 'images_sent': 0})
        debug.event('reflection_start')
        experience = reflection.invoke(prompt=prompt, schema=spec, images=[],
                                       output=output/'reflection', stage='reasoning_experience')
        write_json(output/'reflection_returned.json', experience)
        validate_schema(experience, spec)
        write_json(output/'experience.json', experience)
        lines = ['# 抓取规划 reasoning 经验（草案）', '', experience['summary'],
                 '', '仅根据公开判断总结，未独立验证视觉正确性，未自动接入后续运行。']
        for lesson in experience['lessons']:
            lines += ['', '## '+lesson['name'], '', '需要的信息：'+lesson['information_need'],
                      '', '适用条件：'+lesson['when_to_use'], '', '判断规则：'+lesson['decision_rule'], '']
            lines += [f'{i}. {step}' for i, step in enumerate(lesson['method'], 1)]
            lines += ['', '成功判据：'+lesson['success_check'], '', '信息不足：'+lesson['on_insufficient'],
                      '', '可代码化部分（建议）：'+lesson['code_candidate'],
                      '', '证据：'+canonical(lesson['evidence']), '', '局限：'+'；'.join(lesson['limitations'])]
        lines += ['', '## 待验证', ''] + ['- '+q for q in experience['open_questions']]
        (output/'experience.md').write_text('\n'.join(lines)+'\n')
        report.update(status='SUMMARIZED', lesson_count=len(experience['lessons']),
                      semantic_review='PENDING', learned_policy_replayed=False)
        debug.event('reflection_complete', lessons=len(experience['lessons']))
    except Exception as exc:
        report.update(status='ERROR', error=f'{type(exc).__name__}: {exc}')
    finally:
        report['elapsed_s'] = time.monotonic()-started
        report['metrics'] = call_metrics(vision.calls+reflection.calls, output,
                                       exclude_dirs=('baseline',))
        write_json(output/'report.json', report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ssh-host', default='company-planner')
    p.add_argument('--timeout', type=float, default=900)
    a = p.parse_args(argv)
    opts = dict(backend='remote', ssh_host=a.ssh_host, timeout_s=a.timeout)
    r = run(a.baseline, a.output, RuntimeClaude(**opts), RuntimeClaude(**opts, text_only=True), timeout=a.timeout)
    print(canonical(r), flush=True)
    return 0 if r['status']=='SUMMARIZED' else 2


if __name__=='__main__':
    raise SystemExit(main())
