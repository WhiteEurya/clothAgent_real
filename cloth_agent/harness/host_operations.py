"""Deterministic, current-observation computations used by executable harness patches."""
from __future__ import annotations

import math
import time

from .policy import PolicyError, obj

NAME = {'type': 'string', 'pattern': '^[a-z][a-z0-9_]{0,47}$'}
CATALOG = {
    'reflect_point': {'inputs': ['point', 'line_start', 'line_end'],
                      'output': 'point reflected across an infinite line; does not certify fabric occupancy'},
    'affine_point': {'inputs': ['point', 'matrix'],
                     'output': 'point under six coefficients [a,b,c,d,e,f]: [a*x+b*y+c,d*x+e*y+f]'},
    'rank_candidates': {'inputs': ['candidate_ids', 'origin', 'axis'],
                        'output': 'current-registry candidates sorted by descending projection onto axis; no fabric classification'},
}
HOST_SCHEMA = {'type': 'array', 'maxItems': 8, 'items': obj({
    'id': NAME, 'op': {'enum': list(CATALOG)}, 'source_stage': NAME,
    'bindings': {'type': 'object', 'additionalProperties': NAME},
})}
MEASUREMENTS_SCHEMA = {'type': 'object', 'maxProperties': 32, 'additionalProperties': {
    'anyOf': [{'type': 'null'}, {'type': 'number'}, {'type': 'string'},
              {'type': 'array', 'maxItems': 512, 'items': {'type': ['number', 'string']}}]}}


def validate_bindings(stage):
    seen = set()
    for op in stage.get('host_operations', []):
        if op['id'] in seen or op['source_stage'] not in stage['context']:
            raise PolicyError('Host operation IDs must be unique and source_stage must be in stage context')
        seen.add(op['id'])
        if set(op['bindings']) != set(CATALOG[op['op']]['inputs']):
            raise PolicyError('Host operation bindings must match its catalog inputs')


def vector(value, size):
    if (not isinstance(value, list) or len(value) != size
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError(f'Expected {size} finite coordinates')
    return value


def compute(name, args, registry):
    if name == 'reflect_point':
        p, a, b = [vector(args[key], 2) for key in ('point', 'line_start', 'line_end')]
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = dx*dx+dy*dy
        if length <= 1e-12:
            raise ValueError('Fold line has zero length')
        t = ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length
        result = [2*(a[0]+t*dx)-p[0], 2*(a[1]+t*dy)-p[1]]
    elif name == 'affine_point':
        x, y = vector(args['point'], 2)
        a, b, c, d, e, f = vector(args['matrix'], 6)
        result = [a*x+b*y+c, d*x+e*y+f]
    elif name == 'rank_candidates':
        origin, axis = vector(args['origin'], 2), vector(args['axis'], 2)
        length = math.hypot(*axis)
        if length <= 1e-12:
            raise ValueError('Ranking axis has zero length')
        ids = args['candidate_ids']
        if not isinstance(ids, list) or not ids or any(not isinstance(k, str) for k in ids):
            raise ValueError('Need a nonempty list of current candidate IDs')
        current = {r['candidate_id']: r for r in registry['candidates']}
        if len(set(ids)) != len(ids) or not set(ids) <= current.keys():
            raise ValueError('Unknown or duplicate candidate ID')
        result = []
        for key in ids:
            p = vector(current[key]['pixel_xy'], 2)
            projection = sum((p[i]-origin[i])*axis[i]/length for i in range(2))
            if not math.isfinite(projection):
                raise ValueError('Non-finite projection')
            result.append({'candidate_id': key, 'pixel_xy': p, 'projection': projection})
        return sorted(result, key=lambda r: (-r['projection'], r['candidate_id']))
    else:
        raise ValueError('Unknown host operation')
    return vector(result, 2)


def execute(stage, state, registry):
    records = []
    for op in stage.get('host_operations', []):
        start = time.monotonic()
        record = {'id': op['id'], 'op': op['op'], 'source_stage': op['source_stage'],
                  'status': 'UNKNOWN', 'inputs': {}, 'output': None}
        try:
            values = state[op['source_stage']].get('measurements', {})
            record['inputs'] = {key: values[value] for key, value in op['bindings'].items()}
            record['output'] = compute(op['op'], record['inputs'], registry)
            record['status'] = 'COMPUTED'
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            record['missing_information'] = str(exc)
        record['elapsed_s'] = time.monotonic()-start
        records.append(record)
    return records
