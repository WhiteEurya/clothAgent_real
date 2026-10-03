"""Bounded complete-link action consensus and cost selection; no physical verdict."""
from __future__ import annotations

import itertools
import math
import statistics


def compatible(left, right, *, grasp_mode, epsilon_grasp, epsilon_target, epsilon_anchor):
    a, b = left['action'], right['action']
    grasp = (a['selected_reference']['reference_id'] == b['selected_reference']['reference_id']
             if grasp_mode == 'candidate' else math.dist(a['grasp_pixel_xy'], b['grasp_pixel_xy']) <= epsilon_grasp)
    return (grasp and a['target']['relation'] == b['target']['relation']
            and math.dist(a['target']['pixel_xy'], b['target']['pixel_xy']) <= epsilon_target
            and math.dist(a['target']['anchor_pixel_xy'], b['target']['anchor_pixel_xy']) <= epsilon_anchor)


def select_consensus(rollouts, versions, *, repeats=1, grasp_mode='candidate', epsilon_grasp=12.,
                     epsilon_target=20., epsilon_anchor=20., min_support=0.6, min_harnesses=3,
                     weights=None):
    weights = weights if weights is not None else {'elapsed_s': 1., 'model_calls': 0., 'tool_calls': 0., 'thinking_tokens': 0.}
    if set(weights) != {'elapsed_s', 'model_calls', 'tool_calls', 'thinking_tokens'}:
        raise ValueError('Unknown cost dimensions')
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in weights.values())
            or not any(weights.values())):
        raise ValueError('Cost weights must be finite, nonnegative and not all zero')
    if (grasp_mode not in {'candidate', 'distance'} or not .5 < min_support <= 1 or min_harnesses < 2
            or not 1 <= repeats <= 5 or len(versions) > 9):
        raise ValueError('Invalid consensus settings')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
           for v in (epsilon_grasp, epsilon_target, epsilon_anchor)):
        raise ValueError('Distance thresholds must be finite positive pixel distances')
    kwargs = dict(grasp_mode=grasp_mode, epsilon_grasp=epsilon_grasp,
                  epsilon_target=epsilon_target, epsilon_anchor=epsilon_anchor)
    eligible, rejected = {}, {}
    for key in versions:
        rows = [r for r in rollouts if r['harness_hash'] == key]
        if len(rows) != repeats or any(r['status'] != 'READY' for r in rows):
            rejected[key] = 'INCOMPLETE_OR_NON_READY_REPETITIONS'
        elif not all(compatible(a, b, **kwargs) for a, b in itertools.combinations(rows, 2)):
            rejected[key] = 'UNSTABLE_WITHIN_HARNESS'
        else:
            eligible[key] = rows
    keys = sorted(eligible)
    pairs = {(a, b): all(compatible(x, y, **kwargs) for x in eligible[a] for y in eligible[b])
             for a, b in itertools.combinations(keys, 2)}
    # At most nine distinct harnesses: enumerate maximal complete-link clusters.
    # No single-link chaining where close A/B and B/C hide distant A/C.
    clusters = []
    for size in range(len(keys), 0, -1):
        for members in itertools.combinations(keys, size):
            if any(set(members) < set(cluster) for cluster in clusters):
                continue
            if all(pairs[(a, b)] for a, b in itertools.combinations(members, 2)):
                clusters.append(members)
    report = {'status': 'NO_STABLE_CONSENSUS', 'selected': None, 'rejected_harnesses': rejected,
              'distinct_harnesses': len(versions), 'eligible_harnesses': len(eligible),
              'thresholds': {**kwargs, 'min_support': min_support, 'min_harnesses': min_harnesses},
              'weights': weights, 'clusters': [{'harnesses': list(c), 'votes': len(c),
                  'support': len(c) / len(versions)} for c in clusters],
              'pairwise_compatible': [{'left': a, 'right': b, 'compatible': value} for (a, b), value in pairs.items()],
              'costs': {}, 'meaning': 'Decision stability only; correlated reasoning paths are not independent physical evidence.'}
    if not clusters:
        return report
    largest = [c for c in clusters if len(c) == len(clusters[0])]
    if len(largest) != 1:
        report['reason'] = 'TIED_DOMINANT_CLUSTERS'
        return report
    dominant = largest[0]
    if len(dominant) < min_harnesses or len(dominant) / len(versions) < min_support:
        report['reason'] = 'INSUFFICIENT_DISTINCT_HARNESS_SUPPORT'
        return report
    ranked = []
    for key in dominant:
        medians = {metric: (statistics.median(r['metrics'][metric] for r in eligible[key])
                    if all(r['metrics'].get(metric) is not None for r in eligible[key]) else None)
                   for metric in weights}
        missing = [m for m, w in weights.items() if w > 0 and medians[m] is None]
        cost = None if missing else sum(w * medians[m] for m, w in weights.items() if w > 0)
        report['costs'][key] = {'median_metrics': medians, 'cost': cost, 'missing_weighted_metrics': missing}
        if cost is not None:
            ranked.append((cost, key))
    # Do not compare partial costs against fully measured costs.
    if len(ranked) != len(dominant):
        report.update(status='COST_UNAVAILABLE', reason='A_WEIGHTED_METRIC_IS_UNAVAILABLE')
        return report
    cost, chosen = min(ranked)
    rows = eligible[chosen]
    median = statistics.median(r['metrics']['elapsed_s'] for r in rows)
    exemplar = min(rows, key=lambda r: (abs(r['metrics']['elapsed_s'] - median), r['rollout_id']))
    report.update(status='SELECTED', selected={'harness_hash': chosen, 'rollout_id': exemplar['rollout_id'],
                  'action': exemplar['action'], 'cost': cost, 'votes': len(dominant),
                  'support': len(dominant) / len(versions), 'robot_executable': False})
    return report
