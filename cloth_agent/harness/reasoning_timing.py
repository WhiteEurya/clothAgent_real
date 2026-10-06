"""One fresh fixed-image planning rollout with timing, no learning or robot."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from ..pipeline_timing import PipelineTiming
from .baseline_cache import tree_hashes
from .common import read_json, write_json
from .model import RuntimeClaude
from .reasoning_contract import load_harness, verify_evidence
from .reasoning_learning import CallBudget, DebugLog, run_rollout


def run_once(baseline_run, output, model, *, timeout=900):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'mode': 'planning-timing-only', 'status': 'RUNNING', 'robot_actions': 0,
              'learning_calls': 0, 'baseline_answer_reused': False,
              'scope': 'Fresh point selection from saved prepared images; no image-preparation replay.'}
    try:
        with PipelineTiming(output/'timing', semantic_phases=True) as timing:
            with timing.span('planning.prepare_saved_inputs'):
                source = Path(baseline_run)/'replays/baseline_r00'
                load_harness(source/'reasoning_version.json')
                version = read_json(source/'reasoning_version.json')
                stages = version['harness']['stages']
                if len(stages) != 1 or not stages[0]['allow_ready']:
                    raise ValueError('Timing-only requires one original complete planning stage')
                prepared = source/'prepared'
                hashes = tree_hashes(prepared)
                shutil.copytree(prepared, output/'prepared')
                if tree_hashes(prepared) != hashes or tree_hashes(output/'prepared') != hashes:
                    raise ValueError('Prepared evidence changed during copy')
                frozen = read_json(output/'prepared/evidence.json')
                verify_evidence(frozen, output/'prepared')
                write_json(output/'input_manifest.json', {'source': str(prepared.resolve()), 'files': hashes,
                    'evidence_hash': frozen['evidence_hash'], 'harness_hash': version['harness_hash']})
            budget = CallBudget(max_calls=1, max_seconds=timeout, call_timeout=timeout,
                                prompt_chars=400000, debug=DebugLog(output))
            with timing.span('planning.single_rollout'):
                result = run_rollout(version, frozen, output/'prepared', model, output/'reasoning',
                                     budget, rollout_id='timed_once', timeout=timeout)
            report.update(status=result['status'], action=result.get('action'),
                          reason=result.get('reason'), metrics=result['metrics'],
                          call_attempts=budget.attempts, tokens=budget.token_report())
    except Exception as exc:
        report.update(status='ERROR', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-baseline-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ssh-host', default='company-planner')
    parser.add_argument('--model')
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args(argv)
    if not 0 < args.timeout < float('inf'):
        parser.error('--timeout must be finite and positive')
    model = RuntimeClaude(backend='remote', ssh_host=args.ssh_host, model=args.model,
                          timeout_s=args.timeout)
    try:
        result = run_once(args.reuse_baseline_run, args.output, model, timeout=args.timeout)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', flush=True)
        return 2
    print(f"{result['status']}: {args.output}/report.json", flush=True)
    return 0 if result['status'] == 'READY' else 2


if __name__ == '__main__':
    raise SystemExit(main())
