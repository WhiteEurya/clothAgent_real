"""Contracts for offline, fixed-evidence reasoning rollouts, not robot commands."""
from __future__ import annotations

import copy
from pathlib import Path

from ..image_tools_mcp import pixel_hash, transform_point
from PIL import Image
from .common import digest, now, read_json, write_json
from .executor import _coordinates, inverse, validate_observation
from .policy import PolicyError, obj, validate_schema
from .host_operations import HOST_SCHEMA, MEASUREMENTS_SCHEMA, validate_bindings

TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 2400}
NAME = {'type': 'string', 'pattern': '^[a-z][a-z0-9_]{0,47}$'}
POINT = {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'number'}}
ACTION_SCHEMA = obj({
    'selected_reference': obj({'camera': {'const': 'A'}, 'reference_id': {'type': 'string', 'pattern': '^R[0-9]+$'}, 'reason': TEXT}),
    'target': obj({'pixel_xy': POINT, 'relation': {'enum': ['toward', 'onto', 'across', 'away_from', 'hold']},
                   'anchor_pixel_xy': POINT, 'reason': TEXT}),
})
JUDGMENT_SCHEMA = obj({
    'observation_id': TEXT, 'status': {'enum': ['CONTINUE', 'READY', 'NEEDS_LEARNING']},
    'concepts': {'type': 'array', 'maxItems': 16, 'items': obj({
        'name': TEXT, 'finding': TEXT,
        'source_image_ids': {'type': 'array', 'minItems': 1, 'maxItems': 16, 'uniqueItems': True, 'items': TEXT}})},
    'action': {'anyOf': [{'type': 'null'}, ACTION_SCHEMA]},
    'evidence_summary': TEXT,
    'missing_information': {'type': 'string', 'maxLength': 2400,
        'description': 'Only unresolved information that blocks this visual decision. READY requires the empty string.'},
    'residual_uncertainty': {'type': 'string', 'maxLength': 2400,
        'description': 'Non-blocking limitations and later physical checks. May be nonempty for READY. Never move a truly blocking gap here.'},
})
JUDGMENT_SCHEMA['allOf'] = [
    {'if': {'properties': {'status': {'const': 'READY'}}},
     'then': {'properties': {'action': ACTION_SCHEMA, 'missing_information': {'const': ''}}},
     'else': {'properties': {'action': {'type': 'null'}}}},
    {'if': {'properties': {'status': {'const': 'NEEDS_LEARNING'}}},
     'then': {'properties': {'missing_information': {'type': 'string', 'pattern': r'\S'}}}},
]
JUDGMENT_SCHEMA['properties']['measurements'] = MEASUREMENTS_SCHEMA
HARNESS_SCHEMA = obj({
    'schema_version': {'const': 1}, 'name': NAME, 'applicability': TEXT,
    'stages': {'type': 'array', 'minItems': 1, 'maxItems': 6, 'items': obj({
        'id': NAME, 'instruction': TEXT,
        'context': {'type': 'array', 'maxItems': 5, 'uniqueItems': True, 'items': NAME},
        'allow_ready': {'type': 'boolean'},
    })},
    'on_exhaustion': {'const': 'NEEDS_LEARNING'},
})
REFLECTION_SCHEMA = obj({
    'status': {'enum': ['PROPOSE', 'STOP']}, 'reason': TEXT,
    'harness': {'anyOf': [{'type': 'null'}, HARNESS_SCHEMA]},
    'changes': {'type': 'array', 'maxItems': 12, 'items': obj({
        'field': TEXT, 'observed': TEXT, 'hypothesis': TEXT, 'unverified': TEXT,
        'source_rollout_ids': {'type': 'array', 'minItems': 1, 'maxItems': 20, 'uniqueItems': True, 'items': TEXT},
    })},
})
HARNESS_SCHEMA['properties']['stages']['items']['properties']['host_operations'] = HOST_SCHEMA
# Optional for backward compatibility: an absent abstraction must not discard an
# otherwise executable proposal. These are candidate methods, not approved skills.
REFLECTION_SCHEMA['properties']['operations'] = {
    'type': 'array', 'maxItems': 8, 'items': obj({
        'name': TEXT, 'information_goal': TEXT, 'method': TEXT,
        'outputs': {'type': 'array', 'minItems': 1, 'maxItems': 16, 'items': TEXT},
        'success_check': TEXT, 'on_insufficient': TEXT,
        'replaces': TEXT, 'expected_time_saving': TEXT,
    })}


def baseline_harness():
    """An explicit developer baseline, never presented as a learned artifact."""
    return {'schema_version': 1, 'name': 'baseline',
            'applicability': 'Current garment evidence and fold goal are sufficient for a visual grasp and target proposal.',
            'stages': [{'id': 'plan', 'context': [], 'allow_ready': True,
                        'instruction': 'Inspect the supplied fixed evidence. Determine the task-relevant cloth structure and any useful intermediate concepts. Propose a current grasp candidate and visual target with a concise evidence summary. Stop if the evidence is insufficient.'}],
            'on_exhaustion': 'NEEDS_LEARNING'}


def validate_harness(harness):
    validate_schema(harness, HARNESS_SCHEMA)
    seen = set()
    for stage in harness['stages']:
        if stage['id'] in seen or not set(stage['context']) <= seen:
            raise PolicyError('Stage IDs must be unique and context must reference earlier stages')
        seen.add(stage['id'])
        validate_bindings(stage)
    if not any(s['allow_ready'] for s in harness['stages']):
        raise PolicyError('Harness has no decision output stage')
    # Parameterization is a generation instruction, not a lexical validity test.
    # Digits, example IDs and paths in prose do not establish answer leakage.
    # Keep structural checks here; semantic reusability remains unverified.
    return copy.deepcopy(harness)


def freeze_harness(harness, output, *, parent_hash=None, source='runtime_reflection'):
    harness = validate_harness(harness)
    frozen = {'schema_version': 1, 'harness_hash': digest(harness), 'harness': harness,
              'parent_hash': parent_hash, 'source': source, 'created_at': now(), 'offline_only': True}
    write_json(output, frozen, exclusive=True)
    return frozen


def load_harness(path):
    frozen = read_json(path)
    harness = validate_harness(frozen['harness'])
    if frozen.get('harness_hash') != digest(harness) or frozen.get('offline_only') is not True:
        raise PolicyError('Invalid frozen harness hash or scope')
    return harness


def evidence_from_manifest(manifest, decision_id=None):
    decisions = manifest.get('decisions', [])
    if not decisions:
        raise PolicyError('Manifest contains no pre-decision traces; restore the original run images and registry before learning')
    if decision_id:
        decisions = [d for d in decisions if d['decision_id'] == decision_id]
    if len(decisions) != 1:
        raise PolicyError('Select one decision explicitly with --decision-id; no usable unique decision found')
    trace = decisions[0]
    before = copy.deepcopy(trace['pre_decision'])
    validate_observation(before)
    # Do not read request prompts, historical ROIs, selected overlays, feedback,
    # planner crops, motion intentions or post-decision reasoning.
    images = [i for i in before['images'] if i.get('role') in {'clean', 'overlay', 'reference', 'hint'}
              and i.get('status') == 'AVAILABLE']
    for item in images:
        if item['role'] == 'reference':
            name = Path(item.get('source') or item['path']).name.lower()
            item['reference_kind'] = 'target' if 'target' in name else 'source' if 'source' in name else 'unspecified'
    return {'schema_version': 1, 'observation_id': before['observation_id'],
            'fold_goal': before['fold_goal'], 'candidate_registry': before['candidate_registry'],
            'images': images}


def freeze_evidence(evidence, output, *, base=None):
    """Copy and hash the exact Z once. Optional crops need explicit lineage."""
    if set(evidence) != {'schema_version', 'observation_id', 'fold_goal', 'candidate_registry', 'images'}:
        raise PolicyError('Evidence must contain only the fixed pre-decision contract')
    if evidence['schema_version'] != 1 or not isinstance(evidence['fold_goal'], str) or not evidence['fold_goal'].strip():
        raise PolicyError('Evidence needs a version and a current textual fold goal')
    evidence = copy.deepcopy(evidence)
    for item in evidence['images']:
        path = Path(item['path'])
        item['path'] = str((Path(base or '.') / path).resolve()) if not path.is_absolute() else str(path)
    pair = validate_observation(evidence)
    if not 2 <= len(evidence['images']) <= 16:
        raise PolicyError('Fixed evidence requires two to sixteen images')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    catalog = []
    allowed = {'clean', 'overlay', 'reference', 'hint', 'clean_crop', 'overlay_crop'}
    for index, item in enumerate(evidence['images']):
        if item.get('role') not in allowed or item.get('status') != 'AVAILABLE':
            raise PolicyError('Unknown, missing or post-action evidence image')
        with Image.open(item['path']) as im:
            if list(im.size) != item['size'] or pixel_hash(im) != item['rgb_sha256']:
                raise PolicyError('Evidence image hash or size changed')
            if im.width * im.height > 16_777_216:
                raise PolicyError('Evidence exceeds image pixel budget')
            filename = f'image_{index}.png'
            im.convert('RGB').save(output / filename)
        row = {'image_id': f'image_{index}', 'file': filename, 'role': item['role'],
               'size': item['size'], 'rgb_sha256': item['rgb_sha256']}
        if item['role'] == 'reference':
            kind = item.get('reference_kind', 'unspecified')
            if kind not in {'source', 'target', 'unspecified'}:
                raise PolicyError('Unknown reference role')
            row['reference_kind'] = kind
        if item['role'].endswith('_crop'):
            lineage = item.get('lineage', {})
            root = pair[item['role'].split('_')[0]]
            matrix = lineage.get('to_original')
            if (lineage.get('availability') != 'PRE_DECISION' or lineage.get('parent_rgb_sha256') != root['rgb_sha256']
                    or not isinstance(matrix, list) or len(matrix) != 6):
                raise PolicyError('Prepared crop requires explicit pre-decision parent hash and transform')
            validate_schema(matrix, {'type': 'array', 'items': {'type': 'number'}})
            inverse(matrix)
            w, h = item['size']
            if not all(_coordinates(transform_point(matrix, p), root['size'])
                       for p in ([0, 0], [w-1, 0], [0, h-1], [w-1, h-1])):
                raise PolicyError('Prepared crop transform is outside its parent')
            row['lineage'] = {k: lineage[k] for k in ('availability', 'parent_rgb_sha256', 'to_original')}
        catalog.append(row)
    registry = evidence['candidate_registry']
    payload = {'schema_version': 1, 'observation_id': evidence['observation_id'], 'fold_goal': evidence['fold_goal'],
               'images': catalog, 'coordinate_frame': 'current_clean_pixel_centers',
               'raw_size': registry['raw_size'], 'to_raw': registry['to_raw'],
               'candidates': [{k: c[k] for k in ('candidate_id', 'camera', 'pixel_xy', 'raw_pixel_xy')}
                              for c in registry['candidates']]}
    frozen = {'evidence_hash': digest(payload), 'evidence': payload}
    write_json(output / 'evidence.json', frozen, exclusive=True)
    return frozen


def verify_evidence(frozen, directory):
    evidence = frozen['evidence']
    if digest(evidence) != frozen['evidence_hash']:
        raise PolicyError('Fixed evidence contract changed')
    if read_json(Path(directory) / 'evidence.json') != frozen:
        raise PolicyError('Frozen evidence file changed')
    paths = []
    for image in evidence['images']:
        path = Path(directory) / image['file']
        with Image.open(path) as im:
            if list(im.size) != image['size'] or pixel_hash(im) != image['rgb_sha256']:
                raise PolicyError('Fixed evidence pixels changed')
        paths.append(path)
    return paths


def load_evidence(path):
    """Accept an explicit input package or reuse this tool's frozen Z unchanged."""
    path = Path(path)
    value = read_json(path)
    if set(value) != {'evidence_hash', 'evidence'}:
        return value
    verify_evidence(value, path.parent)
    evidence = value['evidence']
    images = [{**{k: image[k] for k in ('role', 'size', 'rgb_sha256')},
               'path': str((path.parent / image['file']).resolve()), 'status': 'AVAILABLE',
               **({'reference_kind': image['reference_kind']} if 'reference_kind' in image else {}),
               **({'lineage': image['lineage']} if 'lineage' in image else {})} for image in evidence['images']]
    return {'schema_version': 1, 'observation_id': evidence['observation_id'], 'fold_goal': evidence['fold_goal'],
            'images': images, 'candidate_registry': {'observation_id': evidence['observation_id'],
                'binding': 'RAW_RGB_HASH_VERIFIED', 'candidates': evidence['candidates'],
                'raw_size': evidence['raw_size'], 'to_raw': evidence['to_raw']}}


def validate_judgment(result, evidence, *, allow_ready):
    # Explain cross-field errors precisely before generic JSON-schema checks.
    # Reflection must not mistake a serialization failure for bad geometry.
    if isinstance(result, dict):
        status = result.get('status')
        if status in {'CONTINUE', 'NEEDS_LEARNING'} and result.get('action') is not None:
            raise PolicyError(f'NON_READY_ACTION: status={status} requires action=null; '
                              'record intermediate findings in concepts. This is a field conflict, not a grasp-quality verdict.')
        if status == 'READY':
            if not allow_ready:
                raise PolicyError('READY_NOT_ALLOWED: this stage has allow_ready=false; use CONTINUE or NEEDS_LEARNING with action=null.')
            if result.get('action') is None:
                raise PolicyError('READY_ACTION_MISSING: READY requires a complete action.')
            gap = result.get('missing_information')
            if isinstance(gap, str) and gap:
                raise PolicyError('READY_BLOCKING_GAP: missing_information must be empty for READY. '
                                  'If this gap blocks the visual proposal, use NEEDS_LEARNING and action=null. '
                                  'Only non-blocking limitations belong in residual_uncertainty. '
                                  'No conclusion about grasp or target correctness was reached.')
    validate_schema(result, JUDGMENT_SCHEMA)
    if result['observation_id'] != evidence['observation_id']:
        raise PolicyError('Wrong observation ID')
    images = {i['image_id'] for i in evidence['images']}
    for concept in result['concepts']:
        if not set(concept['source_image_ids']) <= images:
            raise PolicyError('Concept cites an unavailable image')
    if result['status'] != 'READY':
        if result['action'] is not None:
            raise PolicyError('Non-ready judgment cannot contain an action')
        if result['status'] == 'NEEDS_LEARNING' and not result['missing_information'].strip():
            raise PolicyError('NEEDS_LEARNING must explain missing information')
        return None
    action = result['action']
    candidate = next((c for c in evidence['candidates'] if c['candidate_id'] == action['selected_reference']['reference_id']), None)
    if not candidate:
        raise PolicyError('Candidate absent from the current observation')
    size = next(i['size'] for i in evidence['images'] if i['role'] == 'clean')
    for field in ('pixel_xy', 'anchor_pixel_xy'):
        if not _coordinates(action['target'][field], size):
            raise PolicyError('Target or anchor outside current clean image')
    target_raw = transform_point(evidence['to_raw'], action['target']['pixel_xy'])
    if not _coordinates(target_raw, evidence['raw_size']):
        raise PolicyError('Mapped target outside raw image')
    return {'selected_reference': action['selected_reference'], 'grasp_pixel_xy': candidate['pixel_xy'],
            'grasp_raw_pixel_xy': candidate['raw_pixel_xy'], 'target': action['target'],
            'target_raw_pixel_xy': target_raw,
            'candidate_legal': True, 'physical_validity': 'NOT_EVALUATED', 'robot_executable': False}
