"""Conservative, deterministic formatting repairs; never repair a decision."""
from __future__ import annotations

import copy

from .common import write_json
from .reasoning_contract import HARNESS_SCHEMA, JUDGMENT_SCHEMA, baseline_harness, validate_harness
from .policy import validate_schema


def repair_format(value, kind, output):
    result = copy.deepcopy(value)
    changes = []
    if isinstance(result, dict):
        if kind == 'judgment' and 'residual_uncertainty' not in result:
            # Newly added informational field: do not infer or clear blockers.
            result['residual_uncertainty'] = ''
            changes.append({'field': 'residual_uncertainty', 'before': None, 'after': '',
                            'rule': 'missing optional information represented as empty; no inferred conclusion'})
    write_json(output, {'kind': kind, 'before': value, 'after': result, 'changes': changes,
                        'model_calls': 0, 'status': 'REPAIRED_PENDING_VALIDATION' if changes else 'UNCHANGED'})
    return result


def preflight_contracts():
    """No model/robot call: catch inconsistencies before spending on rollouts."""
    validate_schema(baseline_harness(), HARNESS_SCHEMA)
    validate_harness(baseline_harness())
    common = dict(observation_id='preflight', concepts=[], evidence_summary='Synthetic preflight',
                  missing_information='', residual_uncertainty='Physical checks not evaluated', action=None)
    validate_schema(dict(common, status='CONTINUE'), JUDGMENT_SCHEMA)
    validate_schema(dict(common, status='NEEDS_LEARNING', missing_information='Target is hidden'), JUDGMENT_SCHEMA)
    validate_schema(dict(common, status='READY', action={
        'selected_reference': {'camera': 'A', 'reference_id': 'R001', 'reason': 'Synthetic'},
        'target': {'pixel_xy': [0, 0], 'anchor_pixel_xy': [0, 0], 'relation': 'hold', 'reason': 'Synthetic'}}), JUDGMENT_SCHEMA)
    return {'status': 'PASSED', 'source': 'synthetic contract checks, not real visual evidence', 'model_calls': 0}
