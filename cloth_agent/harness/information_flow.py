"""Global understanding -> selected observation skills -> evidence-based selection.

Offline visual-planning boundary only. Host executes finite image operations;
no robot access, separate parameter-binding call or standalone verifier.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

from .common import canonical, digest, write_json
from .information_probe import compact_skill
from .policy import PolicyError, obj, validate_schema
from .skills import builtin_registry

# Legacy A/B callers still import this catalog. Implementations are owned by
# the registry, not this compatibility view of their descriptions.
EXECUTORS = {s['id']: s['method'] for s in builtin_registry().catalog()}

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
    'skill_id': {'type': 'string', 'pattern': '^[a-z][a-z0-9_]{0,47}$'},
    'source_image_id': {'type': 'string'},
    'roi': {'anyOf': [{'type': 'null'}, {'type': 'array', 'minItems': 4, 'maxItems': 4,
            'items': {'type': 'number', 'minimum': 0, 'maximum': 1}}]},
    'degrees_clockwise': {'enum': [0, 90, 180, 270]}, 'enlarge': {'type': 'boolean'},
    'expected_information_gain': {'type': 'string', 'minLength': 1},
})
REQUESTS = {'type': 'array', 'maxItems': 12, 'items': REQUEST}
GLOBAL_SCHEMA = obj({'information': STATE, 'observation_requests': REQUESTS})


def selection_schema(plan_schema, registry=None):
    return obj({'status': {'enum': ['CANDIDATE', 'NEED_MORE', 'UNKNOWN']},
                'information': STATE, 'observation_requests': requests_schema(registry),
                'plan': {'anyOf': [{'type': 'null'}, plan_schema]}})


def validate_state(state, catalog, previous=()):
    """Reject invalid evidence bindings; report prose conflicts without rewriting them."""
    validate_schema(state, STATE)
    warnings = []
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
            if not item['finding'].strip() or not item['source_image_ids']:
                raise PolicyError('KNOWN needs image evidence and a finding')
            if item['missing_information'].strip():
                warnings.append({'code': 'KNOWN_WITH_MISSING_INFORMATION', 'information_id': item['id'],
                    'missing_information': item['missing_information'],
                    'instruction': 'Read this text in context: it may say no information is missing, '
                        'refer to another item, or describe a real gap. Do not clear it automatically '
                        'or treat KNOWN as proof. Resolve any decision-blocking contradiction with '
                        'image evidence or report insufficient information.'})
        elif not item['missing_information'].strip():
            raise PolicyError('UNKNOWN must identify missing information')
    return warnings


from .executors.observation import RegisteredObservationHost as ObservationHost


def request_schema(registry=None):
    from .skills import builtin_registry
    schema = copy.deepcopy(REQUEST)
    schema['properties']['skill_id'] = {'enum': [s['id'] for s in (registry or builtin_registry()).catalog()]}
    return schema


def requests_schema(registry=None):
    return {'type': 'array', 'maxItems': 12, 'items': request_schema(registry)}


def global_schema(registry=None):
    return obj({'information': STATE, 'observation_requests': requests_schema(registry)})


def run_information_flow(case, images, artifact, output, *, model, model_name, timeout=600,
                         max_supplements=1, max_host_ops=12, registry=None):
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
        host = ObservationHost(case['observation'], images, artifact, output/'observations', max_host_ops, registry=registry)
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
            'Request operations tied to existing information items, whether KNOWN or UNKNOWN, when they '
            'resolve a gap or simplify subsequent interpretation; explain the expected benefit. '
            'Do not relabel a known fact as unknown merely to request image processing. '
            'Inventory, global layout and reference comparison use the supplied full images now; coordinate '
            'provenance is host-owned. Executable skills are listed below. ROIs always use normalized current '
            'selected source image coordinates, even if orientation is also requested. Rotation preserves clean/overlay alignment. '
            'If existing images suffice return no observation requests. If occlusion or missing sensor detail cannot '
            'be resolved by these methods, keep UNKNOWN and do not request repeated enlargement. '
            'Be concise. State contains only information needed for the current selection.\n')
        global_payload = {**shared, 'images': host.catalog, 'observation_skills': skills,
                          'host_executors': host.registry.snapshot()}
        write_json(output/'global_bundle.json', global_payload)
        t = time.monotonic()
        try:
            result = model.invoke(prompt=instruction+canonical(global_payload), schema=global_schema(host.registry),
                                  images=host.paths, output=output/'global', stage='global_understanding',
                                  timeout_s=timeout-(time.monotonic()-start))
        finally:
            report['global_understanding_s'] = time.monotonic()-t
        validate_schema(result, global_schema(host.registry))
        state = result['information']
        state_warnings = validate_state(state, host.catalog)
        report['information_warnings'] = list(state_warnings)
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
                      'information_warnings': state_warnings,
                      'information_authority': 'Model observations, not independently certified facts; correct contradictions with image evidence.',
                      'images': [{k:v for k,v in i.items() if k!='path'} for i in host.catalog],
                      'observation_results': host.history, 'observation_skills': skills,
                      'host_executors': host.registry.snapshot(),
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
                decision = model.invoke(prompt=prompt+canonical(bundle), schema=selection_schema(case['schema'], host.registry),
                                        images=host.paths, output=output/f'select_{number}', stage='candidate_selection',
                                        timeout_s=timeout-(time.monotonic()-start))
            finally:
                report['selection_s'] += time.monotonic()-t
            validate_schema(decision, selection_schema(case['schema'], host.registry))
            state_warnings = validate_state(decision['information'], host.catalog, state)
            report['information_warnings'].extend(state_warnings)
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
