"""Public operation/evidence records, with explicit provenance and coverage gaps.

Never export thinking blocks or infer hidden steps/timings from final prose.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


PROCESS_INSTRUCTIONS = '''Public execution record (observability only; do not change the task):
For actual tool work, give a SHORT factual purpose before the request and a SHORT observation/result
after it when available. Optionally emit [[process]] followed by one JSON object with id,
information_need, operation, evidence_refs, result_summary, candidate_updates, depends_on.
Record only actions actually performed and explicit observations; a supplied image is not proof it
was inspected. Candidate updates may state candidate_id, outcome (considered/kept/rejected/selected)
and a brief evidence-based reason. Use current image/tool IDs, and prior record IDs for depends_on.
Do not narrate private thinking, invent comparisons, manufacture a fixed workflow, or add calls to
fill the record. Missing observations stay unknown. Final output must still follow the task schema.
'''


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def public_value(value):
    """Strip opaque/private blocks and binary payloads; preserve public result data."""
    if isinstance(value, list):
        return [public_value(v) for v in value if not isinstance(v, dict) or
                v.get('type') not in {'thinking', 'redacted_thinking', 'thinking_delta', 'signature_delta'}]
    if not isinstance(value, dict): return value
    if value.get('type') in {'thinking', 'redacted_thinking', 'thinking_delta', 'signature_delta'}:
        return {'omitted': 'non_public_process_block'}
    if value.get('type') in {'image', 'image_url'}:
        return {'type': 'image', 'binary_omitted': True}
    return {k: public_value(v) for k, v in value.items()
            if k not in {'thinking', 'redacted_thinking', 'signature', 'partial_json', 'inspection_history'}
            and not (k == 'data' and value.get('type') == 'base64')}


def stream_process(path):
    """Keep stream order and tool-use/result bindings; final deltas are not duplicated."""
    path = Path(path)
    events, seen_calls, seen_messages, warnings = [], set(), set(), []
    if not path.exists(): return {'events': [], 'warnings': ['STREAM_NOT_AVAILABLE']}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try: envelope = json.loads(line)
        except ValueError:
            warnings.append(f'MALFORMED_LINE:{line_number}')
            continue
        if not isinstance(envelope, dict): continue
        event = envelope.get('event', envelope)
        if not isinstance(event, dict) or event.get('type') not in {'assistant', 'user'}: continue
        message = event.get('message') or {}
        if not isinstance(message, dict): continue
        content = message.get('content', [])
        if isinstance(content, str): content = [{'type': 'text', 'text': content}]
        if not isinstance(content, list): continue
        for block_index, block in enumerate(content):
            if not isinstance(block, dict): continue
            kind = block.get('type')
            if kind not in {'text', 'tool_use', 'tool_result'}: continue
            source = {'file': str(path), 'line': line_number, 'block': block_index,
                      'received_elapsed_s': envelope.get('received_elapsed_s'),
                      'timing_scope': 'receipt time, not topic thinking duration'}
            row = {'source': source, 'origin': 'RECORDED_PUBLIC_EVENT'}
            if kind == 'tool_use':
                identity, name = block.get('id'), block.get('name', '')
                if identity in seen_calls: continue
                seen_calls.add(identity)
                args = block.get('input', {})
                plumbing = name == 'StructuredOutput' or (name == 'ToolSearch' and
                            isinstance(args, dict) and 'StructuredOutput' in str(args.get('query', '')))
                row.update(kind='output_submission' if plumbing else 'tool_request',
                           id=identity, operation=name, inputs=None if plumbing else public_value(args),
                           result='Output-channel operation; not image evidence' if plumbing else None)
            elif kind == 'tool_result':
                row.update(kind='tool_result', tool_use_id=block.get('tool_use_id'),
                           is_error=bool(block.get('is_error')), result=public_value(block.get('content')))
                # Submission acknowledgements add no evidence; retain the request linkage only.
                if any(e.get('kind') == 'output_submission' and e.get('id') == row['tool_use_id'] for e in events):
                    row['result'] = 'Output submission acknowledgement'
            else:
                if event.get('type') != 'assistant': continue
                text = block.get('text', '')
                key = (message.get('id'), block_index, text)
                if key in seen_messages: continue
                seen_messages.add(key)
                row.update(kind='public_statement', result=text)
                if text.strip().startswith('[[process]]'):
                    try:
                        record = json.loads(text.strip()[len('[[process]]'):])
                        if isinstance(record, dict): row.update(kind='decision_record', result=public_value(record))
                    except ValueError:
                        warnings.append(f'UNPARSED_PROCESS_NOTE:{line_number}')
            events.append(row)
    returned = {e.get('tool_use_id') for e in events if e['kind'] == 'tool_result'}
    for request in events:
        if request['kind'] == 'tool_request' and request.get('id') not in returned:
            warnings.append('TOOL_RESULT_NOT_RECORDED:'+str(request.get('id')))
    return {'events': events, 'warnings': warnings}


def collect_rollout_trace(row, directory=None):
    """Combine observable phases; ordering within model cognition remains unknown."""
    root = Path(directory) if directory is not None else None
    nodes, edges, warnings, artifacts = [], [], [], []
    def add(kind, data, source, *, origin='RECORDED_ARTIFACT', depends=()):
        identity = f'e{len(nodes):04d}'
        nodes.append({'id': identity, 'kind': kind, 'origin': origin,
                      'source': source, 'data': public_value(data)})
        edges.extend({'from': parent, 'to': identity, 'relation': relation} for parent, relation in depends)
        return identity
    def stream(directory, phase):
        path = directory/'transport/claude_events.jsonl'
        if not path.exists(): path = directory/'stdout.jsonl'
        parsed = stream_process(path)
        warnings.extend(f'{phase}:{w}' for w in parsed['warnings'])
        calls, notes, previous = {}, {}, None
        if path.exists(): artifacts.append(str(path))
        for event in parsed['events']:
            links = [(previous, 'RECORDED_EVENT_ORDER')] if previous else []
            tool_id = event.get('tool_use_id')
            if tool_id in calls: links.append((calls[tool_id], 'TOOL_RESULT_FOR_REQUEST'))
            note = event.get('result') if event['kind'] == 'decision_record' else None
            if isinstance(note, dict):
                for dependency in note.get('depends_on', []):
                    if isinstance(dependency, str) and dependency in notes:
                        links.append((notes[dependency], 'MODEL_DECLARED_DEPENDENCY'))
                    else:
                        warnings.append(f'{phase}:UNRESOLVED_NOTE_DEPENDENCY:{dependency}')
            node = add(event['kind'], {**event, 'phase': phase}, event['source'], depends=links)
            if event.get('id'): calls[event['id']] = node
            if isinstance(note, dict) and isinstance(note.get('id'), str): notes[note['id']] = node
            previous = node
    image_producers, gap_nodes = {}, {}
    if root:
        observation = read(root/'observe/returned.json', {})
        if (root/'observe').exists(): stream(root/'observe', 'observation')
        for info in observation.get('information', []):
            gap_nodes[info['id']] = add('information_assessment', info, 'observe/returned.json',
                                       origin='MODEL_REPORTED_FINDING')
    for index, operation in enumerate(row.get('observation_trace', [])):
        request = operation.get('request', {})
        gap = gap_nodes.get(request.get('gap_id'))
        event = add('observation_request', {'request': request,
                    'information_status_at_request': operation.get('information_status_at_request')},
                    f'result.json#/observation_trace/{index}',
                    depends=[(gap, 'ADDRESSES_INFORMATION_NEED')] if gap else [])
        for transform in operation.get('lineage', []):
            image_id = transform.get('image_id')
            if image_id in image_producers: continue
            links = [(event, 'HOST_OPERATION_FOR_REQUEST')]
            parent = image_producers.get(transform.get('parent_image_id'))
            if parent: links.append((parent, 'IMAGE_DERIVED_FROM'))
            image_producers[image_id] = add('image_operation', transform,
                f'result.json#/observation_trace/{index}/lineage', depends=links)
        add('observation_delivery', {k: v for k, v in operation.items() if k not in {'request', 'lineage'}},
            f'result.json#/observation_trace/{index}', depends=[(event, 'DELIVERY_FOR_REQUEST')])
    if root:
        raw_ops = root/'observations/image_tool_calls.jsonl'
        if raw_ops.exists():
            artifacts.append(str(raw_ops))
            for number, line in enumerate(raw_ops.read_text().splitlines(), 1):
                try: op = json.loads(line)
                except ValueError:
                    warnings.append(f'MALFORMED_HOST_OPERATION:{number}'); continue
                add('host_image_execution', op, f'{raw_ops}:{number}')
        package = read(root/'prepared/evidence.json', {})
        if package:
            add('prepared_evidence', package['evidence'].get('images', []), 'prepared/evidence.json',
                origin='HOST_DELIVERED_INPUT_NOT_PROOF_OF_INSPECTION')
    reasoning = row.get('reasoning') or row
    stages = reasoning.get('stages', [])
    if root:
        reasoning_root = root/'reasoning' if (root/'reasoning').is_dir() else root
        if stages:
            for stage in stages:
                # Artifact names are not arbitrary model-controlled paths.
                sid = str(stage.get('stage_id', ''))
                if sid and Path(sid).name == sid and sid not in {'.', '..'}:
                    stream(reasoning_root/'calls'/sid, sid)
        else:
            for call in sorted((reasoning_root/'calls').glob('call_*')):
                if call.is_dir(): stream(call, call.name)
    computations_key = 'host_operations' if 'host_operations' in reasoning else 'executions'
    for index, computation in enumerate(reasoning.get(computations_key, [])):
        add('host_computation', computation, f'reasoning/result.json#/{computations_key}/{index}')
    for request_round in reasoning.get('requests', []):
        add('code_request_and_result', request_round, 'reasoning/result.json#/requests')
    judgments = [(s['judgment'], f'reasoning/result.json#/stages/{i}/judgment')
                 for i, s in enumerate(stages) if 'judgment' in s]
    if reasoning.get('judgment'): judgments.append((reasoning['judgment'], 'reasoning/result.json#/judgment'))
    for judgment, source in judgments:
        jid = add('model_judgment', judgment, source,
                  origin='MODEL_REPORTED_FINDINGS_NOT_INNER_THOUGHT_ORDER')
        for concept in judgment.get('concepts', []):
            for image_id in concept.get('source_image_ids', []):
                if image_id in image_producers:
                    edges.append({'from': image_producers[image_id], 'to': jid,
                                  'relation': 'MODEL_CITES_IMAGE_NOT_CAUSAL_PROOF', 'image_id': image_id})
        notes = {}
        for note in judgment.get('decision_log', []):
            links = []
            for dependency in note.get('depends_on', []):
                if dependency in notes:
                    links.append((notes[dependency], 'MODEL_DECLARED_DEPENDENCY'))
                else:
                    warnings.append(f'{source}:UNRESOLVED_NOTE_DEPENDENCY:{dependency}')
            identity = add('decision_record', note, source+'/decision_log',
                origin='MODEL_REPORTED_DECISION_RECORD_NO_MEASURED_STEP_TIME', depends=links)
            if note.get('id'):
                if note['id'] in notes: warnings.append(f'{source}:DUPLICATE_NOTE_ID:{note["id"]}')
                notes[note['id']] = identity
    add('final_selection', {'status': row.get('status'), 'action': row.get('action'),
                           'reason': row.get('reason')}, 'result.json')
    counts = dict(Counter(n['kind'] for n in nodes))
    gaps = ['PRIVATE_THINKING_NOT_INCLUDED', 'SEMANTIC_TRUTH_NOT_INDEPENDENTLY_VERIFIED',
            'NO_PER_DECISION_THINKING_DURATION']
    comparisons = False
    for node in nodes:
        if node['kind'] != 'decision_record': continue
        note = node['data']
        if isinstance(note.get('result'), dict): note = note['result']
        comparisons = comparisons or bool(note.get('candidate_updates'))
    if not comparisons: gaps.append('CANDIDATE_COMPARISON_HISTORY_NOT_EXPLICITLY_RECORDED')
    if not counts.get('observation_request') and not counts.get('tool_request'):
        gaps.append('NO_OBSERVATION_OPERATION_TRACE_AVAILABLE')
    return {'schema_version': 1, 'nodes': nodes, 'edges': edges,
            'coverage': {'status': 'PARTIAL_PUBLIC_TRACE', 'counts': counts, 'gaps': gaps,
                         'note': 'Node order is presentation only. Only explicit edges and stream receipt order establish relations; it does not reconstruct internal reasoning.'},
            'warnings': warnings, 'artifacts': artifacts}


def write_trace(trace, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'trajectory.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 公开操作与证据轨迹', '', trace['coverage']['note'], '',
             '缺失或未验证：'+', '.join(trace['coverage']['gaps']), '',
             '| 事件 | 类型 | 记录内容 |', '|---|---|---|']
    for node in trace['nodes']:
        text = json.dumps(node['data'], ensure_ascii=False).replace('|', '\\|').replace('\n', ' ')
        lines.append(f'| {node["id"]} | {node["kind"]} | {text} |')
    (output/'trajectory.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists(): raise ValueError('Output exists; choose a new directory')
    trace = collect_rollout_trace(read(args.replay/'result.json'), args.replay)
    write_trace(trace, args.output)
    print(json.dumps(trace['coverage'], ensure_ascii=False))


if __name__ == '__main__': main()
