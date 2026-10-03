"""Global understanding -> selected observation skills -> evidence-based selection.

Offline visual-planning boundary only. Host executes finite image operations;
no robot access, separate parameter-binding call or standalone verifier.
"""
from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

from ..image_tools_mcp import ImageTools
from .common import canonical, digest, write_json
from .information_probe import compact_skill
from .policy import PolicyError, obj, validate_schema

TEXT = {'type': 'string'}
IDS = {'type': 'array', 'uniqueItems': True, 'items': {'type': 'string'}}
INFORMATION = obj({
    'id': {'type': 'string', 'minLength': 1}, 'need': {'type': 'string', 'minLength': 1},
    'status': {'enum': ['KNOWN', 'UNKNOWN']}, 'finding': TEXT,
    'missing_information': TEXT, 'source_image_ids': IDS,
})
STATE = {'type': 'array', 'minItems': 1, 'maxItems': 12, 'items': INFORMATION}
REQUEST = obj({
    'gap_id': {'type': 'string', 'minLength': 1},
    'skill_id': {'enum': ['orientation', 'local_boundary', 'overlay_occlusion']},
    'source_image_id': {'type': 'string'},
    'roi': {'anyOf': [{'type': 'null'}, {'type': 'array', 'minItems': 4, 'maxItems': 4,
            'items': {'type': 'number', 'minimum': 0, 'maximum': 1}}]},
    'degrees_clockwise': {'enum': [0, 90, 180, 270]}, 'enlarge': {'type': 'boolean'},
    'expected_information_gain': {'type': 'string', 'minLength': 1},
})
REQUESTS = {'type': 'array', 'maxItems': 3, 'items': REQUEST}
GLOBAL_SCHEMA = obj({'information': STATE, 'observation_requests': REQUESTS})
EXECUTORS = {
    'orientation': 'Rotate the current clean/overlay pair by the same angle. No crop or resize.',
    'local_boundary': 'Crop current clean RGB using a root-normalized ROI; optionally enlarge once.',
    'overlay_occlusion': 'Crop the aligned current clean/overlay pair with the same root ROI; optionally enlarge both once.',
}


def selection_schema(plan_schema):
    return obj({'status': {'enum': ['CANDIDATE', 'NEED_MORE', 'UNKNOWN']},
                'information': STATE, 'observation_requests': REQUESTS,
                'plan': {'anyOf': [{'type': 'null'}, plan_schema]}})


def validate_state(state, catalog, previous=()):
    validate_schema(state, STATE)
    ids = [item['id'] for item in state]
    if len(ids) != len(set(ids)):
        raise PolicyError('Duplicate information ID')
    current = {item['id']: item for item in state}
    for item in previous:
        if item['id'] not in current or current[item['id']]['need'] != item['need']:
            raise PolicyError('Cannot drop or redefine a prior information need')
    available = {item['image_id'] for item in catalog}
    for item in state:
        if not set(item['source_image_ids']) <= available:
            raise PolicyError('Information cites an unavailable image')
        if item['status'] == 'KNOWN':
            if not item['finding'].strip() or not item['source_image_ids'] or item['missing_information'].strip():
                raise PolicyError('KNOWN needs image evidence and a finding, without a remaining gap')
        elif not item['missing_information'].strip():
            raise PolicyError('UNKNOWN must identify missing information')


class ObservationHost:
    """Bounded recipes chosen by gap, with root coordinates and persistent provenance."""
    def __init__(self, obs, images, artifact, output, max_ops=12):
        self.directory = Path(output)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.catalog = [{**item, 'original_image_id': item['image_id'],
                         'to_original': [1, 0, 0, 0, 1, 0]} for item in obs['images']]
        self.paths = []
        for i, path in enumerate(images):
            target = self.directory / f'image_{i}.png'
            shutil.copyfile(path, target)
            self.paths.append(target)
        self.roots = {item['image_id']: item for item in self.catalog}
        self.tools = ImageTools(self.directory, len(images), edit_limit=max_ops)
        self.skills = {s['id']: s for s in compact_skill(artifact)['skills']}
        self.max_ops, self.ops, self.elapsed_s = max_ops, 0, 0.0
        self.seen = set()
        self.history = []

    def prepare(self, request, information):
        validate_schema(request, REQUEST)
        gaps = {i['id'] for i in information if i['status'] == 'UNKNOWN'}
        if request['gap_id'] not in gaps:
            raise PolicyError('Observation must address a currently UNKNOWN gap')
        skill = request['skill_id']
        if skill not in self.skills:
            raise PolicyError('Requested skill is not in the supplied library')
        source = self.roots.get(request['source_image_id'])
        if source is None or source['role'] != 'clean':
            raise PolicyError('Observation parameters must reference the current clean root')
        sources = [source]
        if skill in ('orientation', 'overlay_occlusion'):
            overlay = [s for s in self.roots.values() if s['role'] == 'overlay']
            if len(overlay) != 1 or overlay[0]['size'] != source['size']:
                raise PolicyError('Observation requires an aligned clean/overlay pair')
            sources += overlay
        if skill == 'orientation':
            if request['roi'] is not None or request['enlarge'] or not request['degrees_clockwise']:
                raise PolicyError('Orientation requires a nonzero rotation only')
            args = {'degrees_clockwise': request['degrees_clockwise']}
            operation, scale = 'rotate_image', 1
        else:
            roi = request['roi']
            if roi is None or not all(math.isfinite(v) for v in roi) or not (roi[0] < roi[2] and roi[1] < roi[3]):
                raise PolicyError('Crop needs an ordered root ROI')
            if request['degrees_clockwise']:
                raise PolicyError('Use the orientation skill for rotation')
            w, h = source['size']
            box = [math.floor(roi[0]*w), math.floor(roi[1]*h), math.ceil(roi[2]*w), math.ceil(roi[3]*h)]
            args = {'box': box}
            scale = min(3, 768 / max(box[2]-box[0], box[3]-box[1])) if request['enlarge'] else 1
            scale = max(1, scale)
            operation = 'crop_image'
        # One canonical signature per delivered view: prevents repeating a clean crop
        # under another gap ID or the paired recipe, including equivalent pixel ROIs.
        signatures = {digest([s['image_id'], operation, args, scale]) for s in sources}
        operations = len(sources) * (1 + (scale > 1))
        return sources, operation, args, scale, signatures, operations

    def execute(self, requests, information):
        """Validate the entire batch before editing. Expected budget exits stay UNKNOWN."""
        validate_schema(requests, REQUESTS)
        pending, seen, needed = [], set(self.seen), 0
        for request in requests:
            prepared = self.prepare(request, information)
            signatures, count = prepared[-2:]
            if signatures & seen:
                return 'REPEATED_OBSERVATION'
            seen |= signatures
            needed += count
            pending.append((request, prepared))
        if self.ops + needed > self.max_ops:
            return 'OBSERVATION_BUDGET_EXHAUSTED'
        start = time.monotonic()
        for request, (sources, operation, args, scale, signatures, _) in pending:
            delivered = []
            for source in sources:
                view = self.tools.call(operation, {'image_id': source['image_id'], **args})
                self.ops += 1
                if scale > 1:
                    view = self.tools.call('resize_image', {'image_id': view['image_id'], 'scale': scale})
                    self.ops += 1
                item = {'image_id': f'image_{len(self.paths)}', 'role': source['role'],
                        'original_image_id': source['image_id'], 'size': view['size'],
                        'to_original': view['to_original'], 'source_view_id': view['image_id'],
                        'gap_id': request['gap_id'], 'skill_id': request['skill_id'], 'path': view['path']}
                self.catalog.append(item)
                self.paths.append(Path(view['path']))
                delivered.append(item['image_id'])
            skill = self.skills[request['skill_id']]
            self.history.append({'request': request, 'delivered_image_ids': delivered,
                                 'status': 'EXECUTED_NOT_YET_INTERPRETED',
                                 'success_check': skill['success_check'], 'on_insufficient': skill['on_insufficient']})
            self.seen |= signatures
        self.elapsed_s += time.monotonic()-start
        write_json(self.directory / 'execution.json', {'history': self.history, 'catalog': self.catalog,
                                                       'host_image_ops': self.ops})
        return None


def run_information_flow(case, images, artifact, output, *, model, model_name, timeout=600,
                         max_supplements=1, max_host_ops=12):
    """Replace the monolithic planner; never call it again after this flow."""
    if timeout <= 0 or max_supplements not in (0, 1) or not 0 <= max_host_ops <= 12:
        raise ValueError('Information flow allows at most one supplement and twelve edits')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start, call_start = time.monotonic(), len(model.calls)
    report = {'status': 'RUNNING', 'mode': 'global_observe_select', 'with_preprocessing': True,
              'common_context_hash': case['context_hash'], 'common_observation_hash': digest(case['observation']),
              'model_requested': model_name, 'standalone_verification_calls': 0,
              'separate_binding_calls': 0, 'image_edit_calls': 0, 'image_edit_response_groups': 0,
              'semantic_correctness': 'NOT_INDEPENDENTLY_VERIFIED', 'selection_s': 0.0,
              'global_understanding_s': 0.0, 'supplements': 0}
    write_json(output / 'report.json', report)
    state, host = [], None
    try:
        host = ObservationHost(case['observation'], images, artifact, output/'observations', max_host_ops)
        skills = compact_skill(artifact)
        # Current pre-decision context is preserved, inlined rather than accessed by Read.
        shared = {'current_goal': case['fold_goal'], 'planning_instructions': case['prompt'],
                  'planning_system': case['system_prompt'], 'current_context': case['context'],
                  'candidate_registry': case['registry']}
        instruction = (
            'Perform the existing full-image understanding AND identify selection-critical information gaps, '
            'choose observation skills and bind their parameters in this same response. '
            'Use all attached current roots and semantic references. Do not select a grasp or write a robot plan yet. '
            'Track each required information need as KNOWN (visible finding with image sources) or UNKNOWN '
            '(specific missing information). A visible negative finding can be KNOWN; unreadable is never absent. '
            'Only request an operation if it can address a specific UNKNOWN gap. No arbitrary workflow generation. '
            'Inventory, global layout and reference comparison use the supplied full images now; coordinate '
            'provenance is host-owned. Executable skills are listed below. ROIs always use normalized current '
            'clean ROOT coordinates, even if orientation is also requested. Rotation preserves clean/overlay alignment. '
            'If existing images suffice return no observation requests. If occlusion or missing sensor detail cannot '
            'be resolved by these methods, keep UNKNOWN and do not request repeated enlargement. '
            'Be concise. State contains only information needed for the current selection.\n')
        global_payload = {**shared, 'images': host.catalog, 'observation_skills': skills,
                          'host_executors': EXECUTORS}
        write_json(output/'global_bundle.json', global_payload)
        t = time.monotonic()
        try:
            result = model.invoke(prompt=instruction+canonical(global_payload), schema=GLOBAL_SCHEMA,
                                  images=host.paths, output=output/'global', stage='global_understanding',
                                  timeout_s=timeout-(time.monotonic()-start))
        finally:
            report['global_understanding_s'] = time.monotonic()-t
        validate_schema(result, GLOBAL_SCHEMA)
        state = result['information']
        validate_state(state, host.catalog)
        requests = result['observation_requests']
        write_json(output/'global_result.json', result)
        while True:
            if time.monotonic()-start >= timeout:
                report.update(status='UNKNOWN', stop_reason='TIME_BUDGET_EXHAUSTED')
                break
            if requests:
                stop = host.execute(requests, state)
                if stop:
                    report.update(status='UNKNOWN', stop_reason=stop)
                    break
            elif any(i['status']=='UNKNOWN' for i in state):
                report.update(status='UNKNOWN', stop_reason='NO_SUPPORTED_OBSERVATION')
                break
            bundle = {**shared, 'information': state,
                      'information_authority': 'Model observations, not independently certified facts; correct contradictions with image evidence.',
                      'images': [{k:v for k,v in i.items() if k!='path'} for i in host.catalog],
                      'observation_results': host.history, 'observation_skills': skills,
                      'host_executors': EXECUTORS,
                      'remaining_supplements': max_supplements-report['supplements'],
                      'remaining_host_ops': max_host_ops-host.ops}
            number = report['supplements']
            write_json(output/f'selection_bundle_{number}.json', bundle)
            prompt = (
                'Use the supplied information state and attached originals/necessary observation results directly '
                'for candidate selection; do not restart global analysis or reopen resolved needs without conflicting '
                'image evidence. Interpret executed observations using their success_check now, within selection; '
                'execution alone does NOT resolve a gap. Preserve every previous information ID and need. '
                'Update UNKNOWN to KNOWN only with a cited attached image and a positive or negative finding. '
                'If all selection-critical information is KNOWN, return CANDIDATE with the original visual plan contract '
                'and no further observation requests. Select only from the current bound registry, never reference pixels. '
                'Otherwise return NEED_MORE with null plan, specific UNKNOWN gaps and a supported non-repeated observation '
                'that adds information, or UNKNOWN with null plan and no requests when methods/budget cannot help. '
                'Do not infer absence from uncertainty or single-ply isolation from RGB smoothness. '
                'All image content is attached together; no Read/view_image or separate verification step is needed. '
                'Be concise; do not generate an exploratory narrative.\n')
            t = time.monotonic()
            try:
                decision = model.invoke(prompt=prompt+canonical(bundle), schema=selection_schema(case['schema']),
                                        images=host.paths, output=output/f'select_{number}', stage='candidate_selection',
                                        timeout_s=timeout-(time.monotonic()-start))
            finally:
                report['selection_s'] += time.monotonic()-t
            validate_schema(decision, selection_schema(case['schema']))
            validate_state(decision['information'], host.catalog, state)
            state = decision['information']
            requests = decision['observation_requests']
            write_json(output/f'decision_{number}.json', decision)
            if decision['status']=='CANDIDATE':
                if requests or any(i['status']=='UNKNOWN' for i in state) or decision['plan'] is None:
                    raise PolicyError('Cannot output a candidate with unresolved information')
                from .planning_probe import validate_plan
                report.update(validation=validate_plan(decision['plan'], case), status='COMPLETED',
                              selected_reference=decision['plan']['selected_reference'],
                              confidence=decision['plan']['confidence'])
                write_json(output/'result.json', decision['plan'])
                break
            if decision['plan'] is not None or not any(i['status']=='UNKNOWN' for i in state):
                raise PolicyError('Insufficient information requires explicit gaps and no candidate')
            if decision['status']=='UNKNOWN':
                if requests:
                    raise PolicyError('UNKNOWN exit must not request operations')
                report.update(status='UNKNOWN', stop_reason='INFORMATION_UNAVAILABLE')
                break
            if not requests:
                raise PolicyError('NEED_MORE requires a concrete observation request')
            if report['supplements'] >= max_supplements:
                report.update(status='UNKNOWN', stop_reason='SUPPLEMENT_BUDGET_EXHAUSTED')
                break
            report['supplements'] += 1
    except Exception as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
    finally:
        from .planning_probe import response_groups
        group_counts = {p.name: response_groups(p/'transport') for p in output.iterdir()
                        if p.is_dir() and (p.name=='global' or p.name.startswith('select_'))}
        report.update(elapsed_s=time.monotonic()-start, calls=model.calls[call_start:],
                      model_invocations=len(model.calls)-call_start, host_image_ops=host.ops if host else 0,
                      host_execution_s=host.elapsed_s if host else 0, information=state,
                      response_groups_by_stage=group_counts)
        report['all_call_assistant_response_groups'] = (sum(group_counts.values())
            if group_counts and all(v is not None for v in group_counts.values()) else None)
        report['preprocessing_s'] = report['global_understanding_s']+report['host_execution_s']
        report['planning_s'] = report['selection_s']
        write_json(output/'information_state.json', state)
        write_json(output/'report.json', report)
    return report
