"""Explicit end-state policy for same-view fold evaluations."""
from __future__ import annotations

import copy
import math


COMPARISON_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'status': {'enum': ['UNCHANGED', 'CHANGED', 'UNCOMPARABLE']},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'evidence': {'type': 'array', 'minItems': 1, 'maxItems': 12,
                     'items': {'type': 'string', 'minLength': 1}},
    },
    'required': ['status', 'confidence', 'evidence'],
}

COMPARISON_INSTRUCTION = (
    'FINAL PERCEPTION COMPARISON POLICY: Compare the current attempt Camera A before RGB '
    'at the calibrated perception position with after RGB freshly captured after release, '
    'Home, and return to that same perception position. Return perception_comparison '
    'with status UNCHANGED, CHANGED, or UNCOMPARABLE, confidence, and visual evidence. '
    'Compare garment location, silhouette, sleeve positions, folds and overlap; ignore '
    'minor exposure, sensor noise and small camera alignment differences. Equal area '
    'alone is insufficient: a displaced or rotated garment is CHANGED. Missing, dark, '
    'occluded, stale, or incompatible views are UNCOMPARABLE, never UNCHANGED. '
    'UNCHANGED is the user-defined unsuccessful-grasp criterion: the host will record '
    'grasp_acquisition=FAILURE and earliest_failure_stage=ACQUISITION. This rule applies '
    'also to sleeve repairs and reversible probes, even if lift evidence suggests '
    'temporary contact. It is an end-state policy, not direct observation of empty jaws. '
    'CHANGED alone does not establish a successful grasp or completed fold. '
    'Lift photos and video are supplementary; inability to interpret them must not '
    'override a clear UNCHANGED comparison at the perception position. '
    'Do not learn a universal physical rule that unchanged images prove empty jaws. '
    'UNCHANGED alone provides no grasp-depth diagnosis and must not imply TOO_SHALLOW '
    'or a deeper-Z retry. The separate host grasp_experience starts UNKNOWN/NONE and '
    'can update only from independent depth-specific interaction evidence.'
)


def comparison_schema(base):
    schema = copy.deepcopy(base)
    schema['properties']['perception_comparison'] = copy.deepcopy(COMPARISON_SCHEMA)
    schema['required'].append('perception_comparison')
    return schema


def validate_comparison(value):
    if not isinstance(value, dict) or set(value) != {'status', 'confidence', 'evidence'}:
        raise ValueError('perception_comparison requires status, confidence and evidence')
    confidence = value['confidence']
    evidence = value['evidence']
    if (value['status'] not in ('UNCHANGED', 'CHANGED', 'UNCOMPARABLE')
            or isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1
            or not isinstance(evidence, list) or not 1 <= len(evidence) <= 12
            or any(not isinstance(item, str) or not item.strip() for item in evidence)):
        raise ValueError('invalid perception_comparison status, confidence or evidence')
    return copy.deepcopy(value)


def apply_comparison_policy(payload):
    result = copy.deepcopy(payload)
    comparison = validate_comparison(result.get('perception_comparison'))
    if comparison['status'] != 'UNCHANGED':
        return result
    reason = ('Final perception view unchanged: treated as unsuccessful acquisition '
              'under the configured end-state policy, not a direct observation of empty jaws.')
    result['grasp_acquisition'] = {
        'status': 'FAILURE', 'confidence': comparison['confidence'],
        'evidence': [reason, *comparison['evidence'][:11]],
    }
    result['earliest_failure_stage'] = 'ACQUISITION'
    result['task_progress']['status'] = 'NEUTRAL'
    result['task_progress']['confidence'] = comparison['confidence']
    result['next_experiment']['reason'] = reason
    result['next_experiment']['change'] = [
        'Reassess contact location, jaw alignment and entry from current RGB; '
        'test a revised grasp without advancing the fold step.'
    ]
    # Do not persist contradictory model-generated success/empty-jaw lessons.
    # Depth diagnosis is generated separately after this policy with runtime
    # surface/command geometry; FAILURE here cannot supply its causal evidence.
    result.pop('skill_update', None)
    return result
