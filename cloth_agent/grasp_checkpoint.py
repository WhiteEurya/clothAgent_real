"""Small lift followed by an explicit visual gate, before fold transport."""
from __future__ import annotations

import copy
import math
import numpy as np
from PIL import Image
from dataclasses import replace
from pathlib import Path

from .free_exploration import ExplorationPlanningError
from .planner_backend import parse_claude_json


class GraspCheckpointRejected(RuntimeError):
    """Physical checkpoint denied continuation; do not auto-restart this run."""


GRASP_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'classification': {'enum': ['GRASP_CONFIRMED', 'EMPTY', 'UNKNOWN']},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'evidence': {'type': 'array', 'minItems': 1, 'maxItems': 4,
                     'items': {'type': 'string', 'minLength': 1}},
        'reason': {'type': 'string', 'minLength': 1},
    },
    'required': ['classification', 'confidence', 'evidence', 'reason'],
}


def compile_grasp_checkpoint(proposal, lift_mm=10.0):
    """Insert a vertical micro-lift into an explicit model plan; never extend it."""
    if not math.isfinite(lift_mm) or not 0 < lift_mm <= 20:
        raise ExplorationPlanningError('grasp check lift must be in (0,20] mm')
    actions = copy.deepcopy(list(proposal.actions))
    close_indices = [i for i, a in enumerate(actions) if a['name'] == 'close_gripper']
    if len(close_indices) != 1:
        raise ExplorationPlanningError('fold grasp gate requires exactly one closure')
    close = close_indices[0]
    if close == 0 or actions[close - 1]['name'] != 'move':
        raise ExplorationPlanningError('fold grasp gate requires a contact move before closure')
    if close + 1 >= len(actions) or actions[close + 1]['name'] != 'move':
        raise ExplorationPlanningError('fold grasp gate requires an immediate vertical lift after closure')
    grasp = actions[close - 1]['args']
    lift = actions[close + 1]['args']
    if any(not math.isfinite(float(p[k])) for p in (grasp, lift) for k in ('x', 'y', 'z', 'yaw')):
        raise ExplorationPlanningError('grasp gate contains nonfinite poses')
    if math.hypot(lift['x'] - grasp['x'], lift['y'] - grasp['y']) > 1e-6:
        raise ExplorationPlanningError('fold grasp gate requires vertical lift before transport')
    height = min(float(lift_mm), lift['z'] - grasp['z'])
    if height <= 0:
        raise ExplorationPlanningError('fold grasp gate requires a positive lift')
    small_pose = {**grasp, 'z': grasp['z'] + height}
    checkpoint = close + 1
    # Keep the original full lift (including yaw) after the decision if needed.
    inserted = any(abs(small_pose[k] - lift[k]) > 1e-6 for k in ('x', 'y', 'z', 'yaw'))
    if inserted:
        actions.insert(checkpoint, {'name': 'move', 'args': small_pose})
    abort = [
        {'name': 'move', 'args': dict(grasp)},
        {'name': 'open_gripper', 'args': {}},
        {'name': 'move', 'args': dict(small_pose)},
        {'name': 'home', 'args': {}},
    ]
    plan = {'checkpoint_action_index': checkpoint, 'lift_mm': height,
            'inserted_micro_lift': inserted, 'abort_actions': abort,
            'required_classification': 'GRASP_CONFIRMED', 'confidence_threshold': 0.8}
    return replace(proposal, actions=tuple(actions)), plan


def validate_grasp_decision(payload):
    if not isinstance(payload, dict) or set(payload) != set(GRASP_SCHEMA['required']):
        raise ValueError('invalid grasp checkpoint fields')
    if payload['classification'] not in {'GRASP_CONFIRMED', 'EMPTY', 'UNKNOWN'}:
        raise ValueError('invalid grasp checkpoint classification')
    confidence = payload['confidence']
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('invalid grasp checkpoint confidence')
    evidence = payload['evidence']
    if (not isinstance(evidence, list) or not 1 <= len(evidence) <= 4
            or any(not isinstance(s, str) or not s.strip() for s in evidence)
            or not isinstance(payload['reason'], str) or not payload['reason'].strip()):
        raise ValueError('grasp checkpoint requires visual evidence and reason')
    allowed = payload['classification'] == 'GRASP_CONFIRMED' and confidence >= .8
    return {**payload, 'status': 'ASSESSED', 'continue_transport': allowed,
            'runtime_decision': 'CONTINUE' if allowed else 'ABORT_RELEASE'}


def inspect_grasp(backend, images, directory: Path):
    """One bounded read-only call. No default affirmative or automatic retry."""
    expected = ['camera_A_grasp_after_close.png', 'camera_A_grasp_after_lift.png']
    if [p.name for p in images] != expected or any(not p.is_file() for p in images):
        raise ValueError('both fresh closure and micro-lift RGB images are required')
    quality = []
    for path in images:
        with Image.open(path) as image:
            luma = np.asarray(image.convert('L'))
        quality.append({'image': str(path), 'p99_luma': float(np.percentile(luma, 99)),
                        'near_black_fraction': float(np.mean(luma < 8))})
    if any(item['p99_luma'] < 12 for item in quality):
        return {'status': 'EVIDENCE_UNUSABLE', 'classification': 'UNKNOWN', 'confidence': 0.0,
                'evidence': ['At least one grasp image is nearly black; no grasp conclusion is supported.'],
                'reason': 'Recapture from a usable observation pose; verify illumination/exposure and wrist occlusion.',
                'image_quality': quality, 'continue_transport': False, 'runtime_decision': 'ABORT_RELEASE'}
    result = backend.invoke(
        prompt=('This is a real T-shirt folding experiment paused at a small vertical lift. '
                'Use view_image to inspect BOTH images: image_0 is after confirmed jaw closure, BEFORE lifting; '
                'image_1 is after the small lift, BEFORE any transport. '
                'Assess only whether fabric is visibly retained by the gripper and lifted from its support. '
                'The camera is wrist-mounted and moves with the arm. Camera motion, jaw closure, '
                'wrinkles alone, or an intended plan do not prove acquisition. '
                'Return GRASP_CONFIRMED only with clear visual evidence of retained lifted fabric; '
                'EMPTY when clearly empty, UNKNOWN if occluded, ambiguous or insufficient. '
                'This is not a single-layer classification task. Do not infer hidden cloth. '
                'Do not edit images, design a fold, or prescribe actions. Request the two views in one batch '
                'where possible, then return the requested JSON.'),
        system_prompt='Visual grasp evidence assessor for a paused cloth-folding experiment. No robot access.',
        image_paths=images, schema=GRASP_SCHEMA,
        debug_dir=directory, image_edit_limit=0, max_turns=4, timeout_s=90,
        overall_timeout_s=90,
    )
    return validate_grasp_decision(parse_claude_json(result.stdout))
