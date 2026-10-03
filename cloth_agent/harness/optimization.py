"""Time-first exploratory comparisons; no automatic correctness or skill verdict."""
from __future__ import annotations

import itertools
import statistics

from .action_consensus import compatible


OPTIMIZATION_OBJECTIVE = {
    'primary': 'Reduce end-to-end visual-planning wall-clock seconds per rollout.',
    'constraints': 'Preserve visual decision quality and stability; do not trade missing evidence for speed.',
    'secondary': 'Report tokens and model calls; no token budget unless explicitly configured.',
    'capabilities': 'Fixed supplied images, prompt/stage/context changes, early READY and supplied deterministic host operations; no arbitrary code or external tools.',
    'evaluation': 'Compare fresh baseline and candidate rollouts. Report learning overhead separately. '
                  'Agreement and READY are not correctness labels; quality remains unverified without independent review.',
    'abstraction': 'Reusable information-acquisition operations with success and insufficiency conditions, '
                   'implemented in the candidate harness and retained as unverified candidates.',
}


def compare_versions(versions, rollouts, *, repeats, options, actual):
    thresholds = {k: options.get(k, default) for k, default in (
        ('grasp_mode', 'candidate'), ('epsilon_grasp', 12.),
        ('epsilon_target', 20.), ('epsilon_anchor', 20.))}
    comparisons = []
    for index, version in enumerate(versions):
        rows = [r for r in rollouts if r['harness_hash'] == version['harness_hash']]
        complete = len(rows) == repeats
        ready = complete and all(r['status'] == 'READY' for r in rows)
        measured = [r['metrics']['elapsed_s'] for r in rows]
        def median_metric(key):
            values = [r['metrics']['usage'].get('total_tokens') if key == 'total_tokens'
                      else r['metrics'].get(key) for r in rows]
            return statistics.median(values) if values and all(v is not None for v in values) else None
        comparisons.append({
            'version': f'H{index}', 'harness_hash': version['harness_hash'],
            'completed_repetitions': len(rows), 'expected_repetitions': repeats,
            'statuses': [r['status'] for r in rows], 'all_ready': ready,
            'within_version_stability': ('NOT_EVALUABLE' if not ready else 'UNMEASURED_SINGLE_REPEAT'
                if len(rows) < 2 else 'STABLE' if all(compatible(a, b, **thresholds)
                    for a, b in itertools.combinations(rows, 2)) else 'UNSTABLE'),
            'median_seconds': statistics.median(measured) if measured else None,
            'median_model_calls': median_metric('model_calls'),
            'median_total_tokens': median_metric('total_tokens'),
            'baseline_over_candidate_time_ratio': None,
            'quality': 'NOT_EVALUATED', 'promoted_to_skill': False,
        })
    if comparisons:
        baseline = comparisons[0]
        for row in comparisons[1:]:
            if actual and baseline['all_ready'] and row['all_ready'] and row['median_seconds'] > 0:
                row['baseline_over_candidate_time_ratio'] = baseline['median_seconds'] / row['median_seconds']
    return {'objective': OPTIMIZATION_OBJECTIVE, 'versions': comparisons, 'thresholds': thresholds,
            'note': 'Ratios above one mean lower observed planning time, not validated acceleration. '
                    'Independent quality review is required; timing includes all stages. '
                    'No consensus is needed to display this comparison. Single repeats cannot measure stability. '
                    'Fixed order, one scene and provider caching limit causal/generalization claims.'}


def operation_candidates(report):
    candidates = []
    for index, reflection in enumerate(report['reflections'], 1):
        if reflection['status'] != 'VALID':
            continue
        proposal = reflection.get('proposal', {})
        version = next((v for v in report['versions'] if v['harness'] == proposal.get('harness')), None)
        rows = [r['rollout_id'] for r in report['rollouts']
                if version and r['harness_hash'] == version['harness_hash']]
        for operation in proposal.get('operations', []):
            candidates.append({**operation, 'status': 'UNVERIFIED',
                'source_reflection': f'patch_{index:02d}',
                'tested_harness_hash': version['harness_hash'] if version else None,
                'evidence': {'supported_by': [], 'failed_in': [], 'unresolved': rows},
                'evidence_meaning': 'Whether this observation method obtained its requested information. '
                    'Whole-harness READY, runtime and consensus do not establish operation-level success.'})
    return {'operations': candidates, 'approved': False,
            'note': 'Candidate abstractions only; no approved skill library is changed.'}
