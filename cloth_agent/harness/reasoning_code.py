"""Extract optional parameterized Python operations without rewriting the planner."""
from __future__ import annotations

from ..pipeline_timing import timed_stage

import copy
import json
import math
import shutil
import time
from pathlib import Path

from jsonschema import Draft202012Validator

from ..public_process import collect_rollout_trace, write_trace

from .baseline_cache import tree_hashes
from .common import canonical, digest, read_json, write_json
from .executors.restricted import RestrictedProgram
from .format_preflight import repair_format
from .policy import PolicyError, obj, validate_schema
from .reasoning_contract import JUDGMENT_SCHEMA, validate_judgment, verify_evidence
from .reasoning_learning import CallBudget, DebugLog, PLANNING_CONTRACT, call_metrics

TEXT = {'type': 'string', 'maxLength': 2400}
JSON_TEXT = {'type': 'string', 'maxLength': 16000}
NAME = {'type': 'string', 'pattern': '^[a-z][a-z0-9_]{0,47}$'}
STRINGS = {'type': 'array', 'maxItems': 12, 'items': TEXT}
FUNCTION_SCHEMA = obj({
    'id': NAME, 'purpose': TEXT, 'extracted_operation': TEXT,
    'when_to_use': TEXT, 'success_check': TEXT, 'on_insufficient': TEXT,
    'input_schema_json': JSON_TEXT, 'output_schema_json': JSON_TEXT,
    'source': {'type': 'string', 'maxLength': 16000},
    'evidence': obj({'supported_by': STRINGS, 'failed_in': STRINGS, 'unresolved': STRINGS}),
    'tests': {'type': 'array', 'minItems': 3, 'maxItems': 6, 'items': obj({
        'name': TEXT, 'arguments_json': JSON_TEXT, 'expected_json': JSON_TEXT,
        'kind': {'enum': ['synthetic', 'record_replay']}, 'record_id': TEXT,
        'explanation': TEXT,
    })},
})
EXTRACTION_SCHEMA = obj({
    'summary': TEXT,
    'functions': {'type': 'array', 'maxItems': 2, 'items': FUNCTION_SCHEMA},
    'left_to_model': STRINGS,
})
RESPONSE_SCHEMA = obj({
    'kind': {'enum': ['REQUEST_CODE', 'FINAL']},
    'calls': {'type': 'array', 'maxItems': 4, 'items': obj({
        'function_id': NAME, 'arguments_json': JSON_TEXT, 'reason': TEXT,
        'source_image_ids': STRINGS,
    })},
    'judgment': {'anyOf': [{'type': 'null'}, JUDGMENT_SCHEMA]},
})

EXTRACT = '''Extract reusable PARAMETERIZED PYTHON OPERATIONS from the supplied public reasoning
records. Scope is the WHOLE grasp-and-target reasoning process, not only image processing and not
only mirror geometry. Find repeated computation/data manipulation whose inputs can be supplied
explicitly and whose outputs are deterministic. Do NOT redesign, split, reorder or prescribe the
planner's workflow. Do NOT force a geometric construction or new checks onto every decision.
The original planner chooses whether/when to call each operation. Semantic visual judgments and
choice of method remain with the planner. Do not reconstruct hidden thinking from public findings.
Read operation_trace first: it includes initial information needs, actual observation requests,
image transformations/lineage, public tool requests/results, explicit findings and final selection.
Follow explicit dependency edges, not an invented chronology inside one model call. A delivered
image does not prove inspection, an information request does not prove the information was obtained,
and a final explanation does not prove every intermediate candidate was compared. Coverage gaps
are missing evidence; never fill them with plausible imagined steps.
Look for reusable multi-operation fragments with explicit input/output contracts, such as coordinate
alignment plus candidate comparison or parameterized filtering plus a derived geometry result,
when actually supported. Do not default to an isolated arithmetic helper simply because the final
answer contains a formula. State which recorded operations each function replaces, and which
visual observations/judgments must remain outside it. A fragment requiring unavailable image or
model operations belongs in left_to_model/unresolved; do not pretend a tiny helper implements it.
Generate at most TWO functions (or none if no supported operation). Existing library functions carry
forward; reuse an ID only to revise that function from actual feedback. Do not regenerate unchanged
code or hard-code a historical answer. Source records are fallible evidence, not ground truth.
Each source is real Python: def run(arguments):, receiving one JSON object. Define input/output
JSON Schemas as JSON strings. Use required keys and additionalProperties=false for input objects.
Supported: local assignments, if/else, return, for NAME in a bounded JSON array, numeric arithmetic
(+ - * / // %), comparisons, and/or/not, conditional expressions, dict/list/tuple literals,
indexing, list concatenation, and min/max/abs/round/int/float/len/floor/ceil. No imports, attributes,
recursion, while, comprehensions, power operator, mutation of inputs, file/network/robot access.
Loops and outputs are bounded by Host. List inputs have at most 128 items. Implement actual code,
not calls to unavailable helper functions. Constants for mathematics are allowed; scene geometry,
thresholds and candidate IDs should be parameters. The function cannot see any images or history.
Return usable JSON data, or an explicit UNKNOWN with a reason for insufficient/degenerate inputs.
Include >=3 small tests: normal inputs, different parameter values, and an edge/unknown case.
Synthetic tests verify internal behavior only, not visual correctness. For record_replay, expected
must be an actual JSON value in that cited record; do not fabricate a historic output. It is fine
to supply only synthetic tests when no measured numeric output was recorded: say so in unresolved.
supported_by/failed_in contain exact record IDs showing the operation's observed information/result,
not robot outcomes; distinguish repetition demonstrated in records from hypothesized reuse.
Keep metadata as callable-operation documentation, never a new overall planning policy. State
which decisions remain with the model. No latency gate, no production promotion.
'''

OPTIONAL_CODE = '''
OPTIONAL LEARNED CODE LIBRARY: The original planning task and stage instruction are unchanged.
You may request one or more listed pure functions when you have a concrete computation to perform.
There is NO required function, stage order, mirror construction, extra validation stage, or usage quota.
Choose the reasoning method yourself. If no listed operation helps, finish normally without calling it.
Request via kind=REQUEST_CODE, calls=[...], judgment=null. arguments_json is a JSON object grounded
in current evidence; explain the computation briefly and cite current image IDs. Host will run the
code and return its actual result. Never claim to have executed a function yourself. A result proves
only the computation, not visual correctness or applicability. Resolve UNKNOWN or report a real gap.
Finish via kind=FINAL, calls=[], judgment=the ordinary judgment schema. A complete grounded plan may
be returned immediately. You retain the entire visual reasoning task; do not output private thoughts.
Only current evidence, your own requests and returned code results are supplied, no historical answers.
'''


def schemas(spec):
    result = []
    for key in ('input_schema_json', 'output_schema_json'):
        schema = json.loads(spec[key])
        # Resolve no external or recursive schema references in generated contracts.
        def inspect(value):
            if isinstance(value, dict):
                if any(k in value for k in ('$ref', '$dynamicRef', '$recursiveRef')):
                    raise PolicyError('Generated contract references are unsupported')
                for item in value.values(): inspect(item)
            elif isinstance(value, list):
                for item in value: inspect(item)
        inspect(schema)
        Draft202012Validator.check_schema(schema)
        result.append(schema)
    if not isinstance(result[0], dict) or result[0].get('type') != 'object':
        raise PolicyError('Function arguments schema must describe a JSON object')
    return result


@timed_stage('harness.execute_function')
def execute_function(spec, arguments):
    started = time.monotonic()
    row = {'function_id': spec['id'], 'source_hash': digest(spec['source']),
           'arguments': arguments, 'status': 'UNKNOWN', 'output': None}
    try:
        inp, out = schemas(spec)
        validate_schema(arguments, inp)
        program = RestrictedProgram(spec['source'], function_name='run',
                                    parameters=('arguments',), allow_loops=True)
        value = program.run(copy.deepcopy(arguments))
        validate_schema(value, out)
        row.update(status='RETURNED', output=value)
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    row['elapsed_s'] = time.monotonic() - started
    return row


def equivalent(a, b):
    if type(a) is int and type(b) is int:
        return a == b
    if type(a) in (float, int) and type(b) in (float, int):
        return math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-6)
    if type(a) != type(b): return False
    if isinstance(a, dict): return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list): return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    return a == b


def contains_value(record, value):
    if equivalent(record, value): return True
    if isinstance(record, dict): return any(contains_value(v, value) for v in record.values())
    if isinstance(record, list): return any(contains_value(v, value) for v in record)
    return False


@timed_stage('harness.test_function')
def test_function(spec, records):
    validate_schema(spec, FUNCTION_SCHEMA)
    record_map = {r['record_id']: r for r in records}
    citations = []
    for key in ('supported_by', 'failed_in'):
        for citation in spec['evidence'][key]:
            matches = [rid for rid in record_map if citation == rid or citation.startswith(rid+':')]
            if len(matches) != 1:
                raise PolicyError('Function evidence cites an unknown or ambiguous record')
            citations.append({'field': key, 'original': citation, 'record_id': matches[0]})
    schemas(spec)
    RestrictedProgram(spec['source'], function_name='run', parameters=('arguments',), allow_loops=True)
    tests = []
    for test in spec['tests']:
        args, expected = json.loads(test['arguments_json']), json.loads(test['expected_json'])
        result = execute_function(spec, args)
        provenance = (test['kind'] == 'synthetic' or
                      test['record_id'] in record_map and contains_value(record_map[test['record_id']], expected))
        tests.append({'name': test['name'], 'kind': test['kind'], 'record_id': test['record_id'],
                      'expected': expected, 'result': result, 'record_value_present': provenance,
                      'passed': provenance and result['status'] == 'RETURNED' and equivalent(result['output'], expected)})
    # Different examples must actually exercise different inputs; no numeric-literal ban.
    diverse = len({canonical(json.loads(t['arguments_json'])) for t in spec['tests']}) >= 3
    return {'passed': diverse and all(t['passed'] for t in tests), 'distinct_inputs': diverse,
            'citation_resolution': citations,
            'tests': tests, 'semantic_correctness': 'NOT_INDEPENDENTLY_VERIFIED',
            'note': 'Record value presence is a provenance check, not proof of the binding interpretation; generated tests are not an independent oracle.'}


def catalog(library):
    # No test fixtures, historical evidence, or scene answers enter fresh planning.
    return [{k: f[k] for k in ('id', 'purpose', 'when_to_use', 'success_check', 'on_insufficient',
                              'input_schema_json', 'output_schema_json')} for f in library.values()]


@timed_stage('harness.run_optional_planner')
def run_optional_planner(version, frozen, evidence_dir, model, budget, output, library,
                         *, timeout, max_requests=2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started, first_call = time.monotonic(), len(model.calls)
    original = copy.deepcopy(version['harness'])
    if len(original['stages']) != 1 or not original['stages'][0]['allow_ready']:
        raise PolicyError('This pilot requires the original single complete planning stage')
    stage = original['stages'][0]
    history, executions = [], []
    result = {'status': 'RUNNING', 'action': None, 'requests': history, 'executions': executions,
              'original_harness_hash': version['harness_hash'], 'workflow_modified': False}
    write_json(output/'original_harness.json', version)
    write_json(output/'function_catalog.json', catalog(library))
    # The original task/coordinate/output contracts remain; only pure computation requests are added.
    contract = PLANNING_CONTRACT.replace(
        'No tools, new images, image processing, filesystem access or physical actions.',
        'Only listed pure-code computation requests are allowed; no new images, image processing, filesystem access or physical actions.')
    try:
        for step in range(max_requests + 1):
            payload = {'fixed_evidence': frozen['evidence'], 'applicability': original['applicability'],
                       'stage': stage, 'bound_state': {}, 'host_results': [],
                       'optional_functions': catalog(library), 'current_rollout_history': history,
                       'remaining_code_request_rounds': max_requests-step}
            reply = budget.invoke(model, frozen=frozen, evidence_dir=evidence_dir,
                prompt=contract+OPTIONAL_CODE+canonical(payload), schema=RESPONSE_SCHEMA,
                output=output/'calls'/f'call_{step:02d}', stage='reasoning_optional_code', deadline=started+timeout)
            validate_schema(reply, RESPONSE_SCHEMA)
            if reply['kind'] == 'FINAL':
                if reply['calls'] or reply['judgment'] is None: raise PolicyError('FINAL requires judgment and no pending calls')
                judgment = repair_format(reply['judgment'], 'judgment', output/f'format_repair_{step}.json')
                action = validate_judgment(judgment, frozen['evidence'], allow_ready=True)
                result.update(status=judgment['status'], action=action, judgment=judgment)
                if judgment['status'] == 'CONTINUE':
                    result.update(status='NEEDS_LEARNING', reason='Original stage ended without a final decision')
                break
            if reply['judgment'] is not None or not reply['calls']:
                raise PolicyError('REQUEST_CODE requires nonempty calls and no judgment')
            if step == max_requests:
                result.update(status='NEEDS_LEARNING', reason='CODE_REQUEST_LIMIT', pending_calls=reply['calls'])
                break
            outputs = []
            for request in reply['calls']:
                try:
                    ids = {i['image_id'] for i in frozen['evidence']['images']}
                    if not set(request['source_image_ids']) <= ids: raise PolicyError('Unknown current image citation')
                    args = json.loads(request['arguments_json'])
                    spec = library[request['function_id']]
                    row = execute_function(spec, args)
                except Exception as exc:
                    row = {'function_id': request['function_id'], 'status': 'UNKNOWN', 'output': None,
                           'error': f'{type(exc).__name__}: {exc}'}
                outputs.append(row)
                executions.append(row)
            history.append({'requests': reply['calls'], 'host_results': outputs})
            write_json(output/'host_execution.json', executions)
            budget.debug.event('code_executed', functions=[r['function_id'] for r in outputs],
                               statuses=[r['status'] for r in outputs])
    except Exception as exc:
        result.update(status='ERROR', reason=f'{type(exc).__name__}: {exc}')
    finally:
        result['metrics'] = {**call_metrics(model.calls[first_call:], output),
                             'elapsed_s': time.monotonic()-started,
                             'code_calls': len(executions),
                             'code_seconds': sum(r.get('elapsed_s', 0) for r in executions)}
        result['code_usage'] = 'USED' if executions else 'NOT_USED'
        try:
            result['operation_trace'] = collect_rollout_trace(result, output)
            write_trace(result['operation_trace'], output/'trace')
        except Exception as exc:
            result['trace_error'] = f'{type(exc).__name__}: {exc}'
        write_json(output/'result.json', result)
    return result


def public_records(report, prefix, directory=None):
    records = []
    for row in report.get('rollouts', []):
        rid = str(row['rollout_id'])
        artifact = None
        if directory and Path(rid).name == rid and rid not in {'.', '..'}:
            path = Path(directory)/'replays'/rid
            if path.is_dir(): artifact = path
        records.append({'record_id': prefix+'_'+row['rollout_id'], 'status': row['status'],
                        'action': row.get('action'),
                        'operation_trace': collect_rollout_trace(row, artifact)})
    if directory and report.get('mode') == 'code-extraction':
        for iteration in report.get('iterations', []):
            index = iteration.get('iteration')
            if type(index) is not int or index < 0: continue
            artifact = Path(directory)/f'iter_{index:02d}'/'reasoning'
            if not (artifact/'result.json').exists(): continue
            row = read_json(artifact/'result.json')
            records.append({'record_id': f'{prefix}_iteration_{index:02d}', 'status': row['status'],
                            'action': row.get('action'), 'operation_trace': collect_rollout_trace(row, artifact)})
    return records


@timed_stage('harness.run_extraction')
def run_extraction(baseline_run, reasoning_record, output, extractor, planner, *,
                   iterations=2, call_timeout=600, replay_timeout=1200, max_seconds=3600,
                   max_calls=10, max_requests=2, expected_root_hash=None, reuse_extractions=None):
    if not 1 <= iterations <= 2: raise ValueError('Pilot supports one or two extraction iterations')
    source = Path(baseline_run)
    baseline_report = read_json(source/'report.json')
    if expected_root_hash and baseline_report['root_evidence_hash'] != expected_root_hash:
        raise PolicyError('Baseline root evidence mismatch')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    debug = DebugLog(output)
    budget = CallBudget(max_calls=max_calls, max_seconds=max_seconds, call_timeout=call_timeout,
                        prompt_chars=400000, debug=debug)
    started = time.monotonic()
    report = {'status': 'RUNNING', 'mode': 'code-extraction', 'iterations': [],
              'robot_actions': 0, 'automatic_activation': False, 'baseline_reused': True,
              'original_workflow_preserved': True, 'semantic_correctness': 'NOT_INDEPENDENTLY_VERIFIED'}
    library, feedback = {}, []
    try:
        src = source/'replays/baseline_r00'
        before = tree_hashes(src)
        saved = output/'cached_baseline'
        shutil.copytree(src, saved)
        if tree_hashes(src) != before or tree_hashes(saved) != before: raise PolicyError('Baseline changed while copying')
        write_json(output/'baseline_manifest.json', {'source': str(src.resolve()), 'files': before})
        frozen = read_json(saved/'prepared/evidence.json')
        verify_evidence(frozen, saved/'prepared')
        version = read_json(saved/'reasoning_version.json')
        report.update(evidence_hash=frozen['evidence_hash'], original_harness_hash=version['harness_hash'])
        if reuse_extractions:
            cached_report = read_json(Path(reuse_extractions)/'report.json')
            if (cached_report['evidence_hash'] != frozen['evidence_hash'] or
                    cached_report['original_harness_hash'] != version['harness_hash']):
                raise PolicyError('Saved extraction evidence or original planner mismatch')
            report['extraction_reuse'] = {'source': str(Path(reuse_extractions).resolve()),
                'fresh_extraction_calls': 0,
                'note': 'Re-evaluates the same saved extraction iterations after a Host fix; saved generation did not see these new replay results.'}
        records = public_records(baseline_report, 'baseline', source)
        if reasoning_record:
            previous = read_json(Path(reasoning_record)/'report.json')
            if (previous.get('root_evidence_hash') != baseline_report['root_evidence_hash'] and
                    previous.get('evidence_hash') != frozen['evidence_hash']):
                raise PolicyError('Learning records belong to different evidence')
            records += public_records(previous, 'history', reasoning_record)
        write_json(output/'source_records.json', records)
        for index in range(iterations):
            directory = output/f'iter_{index:02d}'
            directory.mkdir()
            row = {'iteration': index, 'status': 'EXTRACTING', 'functions': []}
            report['iterations'].append(row)
            write_json(output/'report.json', report)
            debug.event('extraction_start', iteration=index)
            try:
                context = {'records': records, 'existing_library': list(library.values()), 'iteration_feedback': feedback}
                if reuse_extractions:
                    source_path = Path(reuse_extractions)/f'iter_{index:02d}'/'extraction.json'
                    generated = read_json(source_path)
                    row['extraction_reused'] = {'source': str(source_path.resolve()), 'hash': digest(generated)}
                else:
                    generated = budget.invoke(extractor, frozen=frozen, evidence_dir=saved/'prepared',
                        prompt=EXTRACT+canonical(context), schema=EXTRACTION_SCHEMA,
                        output=directory/'extract', stage='extract_parameterized_code')
                validate_schema(generated, EXTRACTION_SCHEMA)
                write_json(directory/'extraction.json', generated)
                ids = [f['id'] for f in generated['functions']]
                if len(set(ids)) != len(ids): raise PolicyError('Duplicate generated function IDs')
                for spec in generated['functions']:
                    dest = directory/'functions'/spec['id']
                    dest.mkdir(parents=True)
                    (dest/'function.py').write_text(spec['source']+'\n')
                    write_json(dest/'specification.json', spec)
                    try:
                        gate = test_function(spec, records)
                    except Exception as exc:
                        gate = {'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
                    write_json(dest/'tests.json', gate)
                    row['functions'].append({'id': spec['id'], 'gate': gate})
                    if gate['passed']: library[spec['id']] = spec
                write_json(directory/'library.json', list(library.values()))
                write_json(output/'library.json', list(library.values()))
                if not library:
                    row.update(status='NO_EXECUTABLE_FUNCTION')
                    feedback.append(copy.deepcopy(row))
                    continue
                row['status'] = 'REPLAYING'
                write_json(output/'report.json', report)
                result = run_optional_planner(version, frozen, saved/'prepared', planner, budget,
                    directory/'reasoning', library, timeout=replay_timeout, max_requests=max_requests)
                row.update(status='TESTED', planning_status=result['status'], code_usage=result['code_usage'],
                           action=result['action'], metrics=result['metrics'])
                record = {'record_id': f'iteration_{index:02d}', 'status': result['status'],
                          'operation_trace': result.get('operation_trace'),
                          'trace_error': result.get('trace_error')}
                records.append(record)
                feedback.append({'iteration': index, 'functions': row['functions'], 'planning_record': record,
                                 'metrics': result['metrics'], 'error': result.get('reason', '')[:1000]})
            except Exception as exc:
                row.update(status='ERROR', error=f'{type(exc).__name__}: {exc}')
                feedback.append(copy.deepcopy(row))
            finally:
                write_json(directory/'feedback.json', feedback[-1] if feedback else row)
                write_json(output/'report.json', report)
                debug.event('extraction_end', iteration=index, status=row['status'], planning_status=row.get('planning_status'))
        report['status'] = 'COMPLETED'
    except Exception as exc:
        report.update(status='ERROR', error=f'{type(exc).__name__}: {exc}')
    finally:
        report['totals'] = {**call_metrics(extractor.calls+planner.calls, output, exclude_dirs=('cached_baseline',)),
                            'elapsed_s': time.monotonic()-started, 'token_accounting': budget.token_report()}
        write_json(output/'report.json', report)
        debug.event('code_extraction_end', status=report['status'], elapsed_s=report['totals']['elapsed_s'])
    return report
